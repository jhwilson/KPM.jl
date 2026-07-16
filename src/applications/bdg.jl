"""
    ScaledOperator(op, a, b)

Lazy matrix-free representation of `(op - b I) / a` for duck-typed
operators. This is used where `normalizeH`, which requires `ishermitian` and
Arpack `eigs`, cannot be applied.
"""
struct ScaledOperator{TO}
    op::TO
    a::Float64
    b::Float64
end

Base.size(S::ScaledOperator) = size(S.op)
Base.size(S::ScaledOperator, d::Int) = size(S.op, d)
Base.eltype(S::ScaledOperator) = promote_type(eltype(S.op), Float64)

function LinearAlgebra.mul!(Y::AbstractVecOrMat, S::ScaledOperator,
                            X::AbstractVecOrMat, α::Number, β::Number)
    mul!(Y, S.op, X, α / S.a, β)
    iszero(S.b) || (@. Y -= (α * S.b / S.a) * X)
    return Y
end

LinearAlgebra.mul!(Y::AbstractVecOrMat, S::ScaledOperator, X::AbstractVecOrMat) =
    mul!(Y, S, X, true, false)

"""
    spectral_radius(op; tol=1e-4, maxiter=300, rng=Xoshiro(0), v0=nothing) -> (radius, v)

Estimate the spectral radius of a Hermitian matrix-free operator with norm-ratio
power iteration. The norm ratio converges to the spectral radius even when the
extremal eigenvalues occur as a `±Emax` pair, as in BdG systems. The default
deterministic RNG makes restarts reproducible, and `v0` enables warm starts.
"""
function spectral_radius(op; tol=1e-4, maxiter=300, rng=Xoshiro(0), v0=nothing)
    N = size(op, 1)
    v = v0 === nothing ? randn(rng, ComplexF64, N) : collect(ComplexF64, v0)
    length(v) == N || throw(ArgumentError("spectral_radius: v0 has length $(length(v)); expected $N"))
    nv = norm(v)
    iszero(nv) && throw(ArgumentError("spectral_radius: v0 must be nonzero"))
    v ./= nv
    w = similar(v)
    λ_prev = Inf

    for iter in 1:maxiter
        mul!(w, op, v)
        λ = norm(w)
        iszero(λ) && return 0.0, v
        if abs(λ - λ_prev) <= tol * max(λ, eps())
            v .= w ./ λ
            return Float64(λ), v
        end
        v .= w ./ λ
        λ_prev = λ
    end

    @debug "spectral_radius reached maxiter without convergence" maxiter tol λ=λ_prev
    return Float64(λ_prev), v
end

"""
    BdGOperator(h; mu, U, n=zeros(size(h, 1)), Delta=zeros(ComplexF64, size(h, 1)))

Matrix-free reduced spin-singlet Nambu BdG operator with particle-hole layout
`[particle; hole]` and hole index `i + N`:

    H_BdG = [ ξ                         Diagonal(Δ)       ]
            [ Diagonal(conj(Δ))         -ξ                ]

with `ξ = h - μ I - Diagonal(U n / 2)`.

Here `U > 0` is attractive, with `H_int = -U Σ n↑n↓`, Hartree shift
`-(U/2)n`, and `Δ_i = -U_i⟨c_{i↓}c_{i↑}⟩`. This CPU-only reduced convention
uses the same `h` in the hole block and presumes `h_{-K}^* = h_K`; for matrix
inputs this is exact when `h` is real-symmetric. Matrix-free `h` is the
caller's responsibility.
"""
mutable struct BdGOperator{TH}
    const h::TH
    μ::Float64
    const U::Vector{Float64}
    const n::Vector{Float64}
    const Δ::Vector{ComplexF64}
    const N::Int
end

function BdGOperator(h; mu::Real, U, n=zeros(size(h, 1)), Delta=zeros(ComplexF64, size(h, 1)))
    N = size(h, 1)
    size(h, 2) == N || throw(ArgumentError("BdGOperator: h must be square (got $(size(h)))"))
    U_vec = U isa Number ? fill(Float64(U), N) : collect(Float64, U)
    n_vec = collect(Float64, n)
    Δ_vec = collect(ComplexF64, Delta)
    length(U_vec) == N || throw(ArgumentError("BdGOperator: U has length $(length(U_vec)); expected $N"))
    length(n_vec) == N || throw(ArgumentError("BdGOperator: n has length $(length(n_vec)); expected $N"))
    length(Δ_vec) == N || throw(ArgumentError("BdGOperator: Delta has length $(length(Δ_vec)); expected $N"))

    if h isa AbstractMatrix
        ishermitian(h) || throw(ArgumentError("BdGOperator: matrix h must be Hermitian"))
        h ≈ transpose(h) || @warn "BdGOperator reduced convention presumes h_{-K}^* = h_K (i.e. symmetric h); the spectrum need not be particle-hole symmetric."
    end
    return BdGOperator(h, Float64(mu), U_vec, n_vec, Δ_vec, N)
end

Base.size(B::BdGOperator) = (2B.N, 2B.N)
Base.size(B::BdGOperator, d::Int) = d <= 2 ? 2B.N : 1
Base.eltype(::BdGOperator) = ComplexF64

_nambu_block(x::AbstractVector, N, blk) = view(x, (blk - 1) * N + 1:blk * N)
_nambu_block(x::AbstractMatrix, N, blk) = view(x, (blk - 1) * N + 1:blk * N, :)

function LinearAlgebra.mul!(Y::AbstractVecOrMat, B::BdGOperator,
                            X::AbstractVecOrMat, α::Number, β::Number)
    N = B.N
    Yp = _nambu_block(Y, N, 1); Yh = _nambu_block(Y, N, 2)
    Xp = _nambu_block(X, N, 1); Xh = _nambu_block(X, N, 2)
    mul!(Yp, B.h, Xp, α, β)
    mul!(Yh, B.h, Xh, -α, β)
    @. Yp += α * ((-B.μ - (B.U / 2) * B.n) * Xp + B.Δ * Xh)
    @. Yh += α * (conj(B.Δ) * Xp - (-B.μ - (B.U / 2) * B.n) * Xh)
    return Y
end

LinearAlgebra.mul!(Y::AbstractVecOrMat, B::BdGOperator, X::AbstractVecOrMat) =
    mul!(Y, B, X, true, false)

"""
    bdg_site_moments(Hs, N, sites, NC; batch_size=64, verbose=0)

Compute the raw local BdG moments
`mu_rho[m, c] = real(<i,p|T_{m-1}(Hs)|i,p>)` and
`mu_delta[m, c] = conj(<i,h|T_{m-1}(Hs)|i,p>)` for `i = sites[c]`.
Both moments are read from one recurrence seeded only with `|i,p>`; the
particle and hole entries of each Chebyshev vector supply the two results.

Sites are processed in batches, using a CPU-only workspace of
`2N * batch_size * 2` complex numbers (or less when there are fewer sites).
"""
function bdg_site_moments(Hs, N::Integer, sites::AbstractVector{<:Integer}, NC::Integer;
                          batch_size::Integer=64, verbose::Integer=0)
    NC >= 2 || throw(ArgumentError("bdg_site_moments: NC must be at least 2 (got $NC)"))
    batch_size > 0 || throw(ArgumentError("bdg_site_moments: batch_size must be positive (got $batch_size)"))
    all(i -> 1 <= i <= N, sites) ||
        throw(ArgumentError("bdg_site_moments: all sites must satisfy 1 <= site <= N=$N"))

    ns = length(sites)
    mu_rho = zeros(dt_real, NC, ns)
    mu_delta = zeros(dt_cplx, NC, ns)
    iszero(ns) && return mu_rho, mu_delta

    batch_capacity = min(Int(batch_size), ns)
    psi = zeros(dt_cplx, 2Int(N), batch_capacity, 2)
    batch_starts = 1:batch_capacity:ns
    verbose >= 1 && println("NC = $(NC), sites = $(ns), batch_size = $(batch_capacity)")

    for first_site in batch_starts
        last_site = min(first_site + batch_capacity - 1, ns)
        B = last_site - first_site + 1
        psi_active = view(psi, :, 1:B, :)
        fill!(psi_active, zero(dt_cplx))
        psi_views = map(i -> view(psi_active, :, :, i), 1:2)

        for (c, cg) in enumerate(first_site:last_site)
            psi_active[sites[cg], c, 1] = one(dt_cplx)
        end

        function extract_moment!(m, slot)
            for (c, cg) in enumerate(first_site:last_site)
                i = sites[cg]
                mu_rho[m, cg] = real(slot[i, c])
                mu_delta[m, cg] = conj(slot[i + N, c])
            end
            return nothing
        end

        extract_moment!(1, psi_views[1])
        mul!(psi_views[2], Hs, psi_views[1])
        extract_moment!(2, psi_views[2])

        ip = 2
        ipp = 1
        for m in 3:NC
            chebyshev_iter_single(Hs, psi_views[ipp], psi_views[ip])
            extract_moment!(m, psi_views[ipp])
            ip = 3 - ip
            ipp = 3 - ipp
        end
    end

    return mu_rho, mu_delta
end

"""
    bdg_update(mu_rho, mu_delta, a; U, beta, g_rho=1.0,
               kernel=JacksonKernel, Np=2size(mu_rho, 1))

Update local BdG fields at Fermi level zero from the Jackson-dressed moments:

    n_c = g_rho / Np * sum_k gamma_rho_c(x_k) f(a x_k)
    Delta_c = -U_c / Np * sum_k gamma_delta_c(x_k) f(a x_k)

where `gamma_c(x) = sum_m kernel(m-1, NC) hn(m-1) mu[m,c]
T_{m-1}(x)` and `x_k` are Gauss-Chebyshev nodes. There is no extra
`a`-Jacobian: the spectral `1/a` cancels the `dE = a dx` measure exactly.
"""
function bdg_update(mu_rho::AbstractMatrix{<:Real},
                    mu_delta::AbstractMatrix{<:Complex}, a::Real;
                    U::AbstractVector{<:Real}, beta::Real, g_rho::Real=1.0,
                    kernel=JacksonKernel, Np::Integer=2 * size(mu_rho, 1))
    size(mu_rho) == size(mu_delta) ||
        throw(ArgumentError("bdg_update: mu_rho and mu_delta must have the same size (got $(size(mu_rho)) and $(size(mu_delta)))"))
    NC, ns = size(mu_rho)
    length(U) == ns || throw(ArgumentError("bdg_update: U has length $(length(U)); expected $ns"))
    beta > 0 || throw(ArgumentError("bdg_update: beta must be positive (got $beta)"))
    a > 0 || throw(ArgumentError("bdg_update: a must be positive (got $a)"))
    Np > 0 || throw(ArgumentError("bdg_update: Np must be positive (got $Np)"))

    gh = kernel.(0:NC-1, NC) .* hn.(0:NC-1)
    nodes, _ = gausschebyshevt(Np)
    C = cos.((0:NC-1) .* acos.(nodes'))
    wf = fermiFunctions(0.0, Float64(beta)).(a .* nodes) ./ Np
    integrated = C * wf
    n_new = g_rho .* real.(transpose(gh .* mu_rho) * integrated)
    Delta_new = -collect(Float64, U) .* (transpose(gh .* mu_delta) * integrated)
    return collect(Float64, n_new), collect(ComplexF64, Delta_new)
end

"""
    BdGSCFResult

Summary of a BdG self-consistency solve. `history` stores one entry per
fixed-point iteration, including entries restored from a checkpoint.
"""
struct BdGSCFResult
    converged::Bool
    iterations::Int
    residual_delta::Float64
    residual_n::Float64
    a::Float64
    history::Vector{NamedTuple}
end

Base.show(io::IO, r::BdGSCFResult) = print(io,
    "BdGSCFResult(converged=$(r.converged), iterations=$(r.iterations), " *
    "residual_delta=$(r.residual_delta), residual_n=$(r.residual_n), a=$(r.a))")

"""
    bdg_checkpoint(path, op, history, v_power) -> nothing

Atomically write the plain-data state needed to restart a BdG
self-consistency solve. The operator and callback are never serialized.
"""
function bdg_checkpoint(path::AbstractString, op::BdGOperator, history, v_power)
    state = (version=1,
             delta=copy(op.Δ),
             n=copy(op.n),
             mu=op.μ,
             U=copy(op.U),
             N=op.N,
             history=copy(history),
             v_power=v_power === nothing ? nothing : copy(v_power))
    tmp_path = "$(path).tmp"
    open(tmp_path, "w") do io
        serialize(io, state)
    end
    mv(tmp_path, path; force=true)
    return nothing
end

"""
    bdg_restore!(op, path) -> (history, v_power)

Restore a checkpointed BdG field state into `op`. The checkpoint must have
the same number of sites as `op`.
"""
function bdg_restore!(op::BdGOperator, path::AbstractString)
    state = open(deserialize, path)
    state.version == 1 || throw(ArgumentError("bdg_restore!: unsupported checkpoint version $(state.version)"))
    state.N == op.N || throw(ArgumentError("bdg_restore!: checkpoint has N=$(state.N); expected N=$(op.N)"))
    length(state.delta) == op.N || throw(ArgumentError("bdg_restore!: checkpoint delta has invalid length"))
    length(state.n) == op.N || throw(ArgumentError("bdg_restore!: checkpoint n has invalid length"))
    op.Δ .= state.delta
    op.n .= state.n
    op.μ = state.mu
    return copy(state.history), state.v_power === nothing ? nothing : copy(state.v_power)
end

"""
    bdg_solve!(op; beta, NC=512, g_rho=1.0, mix=0.1, tol_delta=1e-6,
               tol_n=1e-6, maxiter=500, kernel=JacksonKernel, Np=2NC,
               batch_size=64, update_density=true, target_filling=nothing,
               mu_bracket=(-Inf, Inf), mu_tol=1e-4, mu_maxiter=60,
               rescale_eps=0.2, callback=nothing, checkpoint_path=nothing,
               checkpoint_every=10, restart=nothing, verbose=0) -> BdGSCFResult

Solve the local reduced BdG fixed-point equations by linear mixing. A field
update must meet both channel tolerances on two consecutive iterations before
the solve is reported converged. With `target_filling`, a bisection over
`op.μ` performs a full inner solve at every chemical-potential evaluation;
the fields carry over between evaluations as a warm start.
"""
function bdg_solve!(op::BdGOperator; beta::Real, NC::Integer=512,
                    g_rho::Real=1.0, mix::Real=0.1,
                    tol_delta::Real=1e-6, tol_n::Real=1e-6,
                    maxiter::Integer=500, kernel=JacksonKernel,
                    Np::Integer=2 * NC, batch_size::Integer=64,
                    update_density::Bool=true,
                    target_filling::Union{Nothing, Real}=nothing,
                    mu_bracket::Tuple{Real, Real}=(-Inf, Inf), mu_tol::Real=1e-4,
                    mu_maxiter::Integer=60, rescale_eps::Real=0.2,
                    callback=nothing,
                    checkpoint_path::Union{Nothing, AbstractString}=nothing,
                    checkpoint_every::Integer=10,
                    restart::Union{Nothing, AbstractString}=nothing,
                    verbose::Integer=0)
    beta > 0 || throw(ArgumentError("bdg_solve!: beta must be positive (got $beta)"))
    NC >= 2 || throw(ArgumentError("bdg_solve!: NC must be at least 2 (got $NC)"))
    Np > 0 || throw(ArgumentError("bdg_solve!: Np must be positive (got $Np)"))
    0 < mix <= 1 || throw(ArgumentError("bdg_solve!: mix must satisfy 0 < mix <= 1 (got $mix)"))
    tol_delta >= 0 || throw(ArgumentError("bdg_solve!: tol_delta must be nonnegative (got $tol_delta)"))
    tol_n >= 0 || throw(ArgumentError("bdg_solve!: tol_n must be nonnegative (got $tol_n)"))
    maxiter > 0 || throw(ArgumentError("bdg_solve!: maxiter must be positive (got $maxiter)"))
    batch_size > 0 || throw(ArgumentError("bdg_solve!: batch_size must be positive (got $batch_size)"))
    0 < rescale_eps < 2 || throw(ArgumentError("bdg_solve!: rescale_eps must satisfy 0 < rescale_eps < 2 (got $rescale_eps)"))
    mu_tol >= 0 || throw(ArgumentError("bdg_solve!: mu_tol must be nonnegative (got $mu_tol)"))
    mu_maxiter > 0 || throw(ArgumentError("bdg_solve!: mu_maxiter must be positive (got $mu_maxiter)"))
    checkpoint_every > 0 || throw(ArgumentError("bdg_solve!: checkpoint_every must be positive (got $checkpoint_every)"))

    history, v_power = restart === nothing ? (NamedTuple[], nothing) : bdg_restore!(op, restart)

    function inner_solve!()
        consecutive = false
        converged = false
        res_d_abs = Inf
        res_n_abs = update_density ? Inf : 0.0
        a = NaN
        iter_offset = isempty(history) ? 0 : last(history).iter

        for local_iter in 1:Int(maxiter)
            iter = iter_offset + local_iter
            # In translation-invariant problems an exact prior power vector
            # can remain a nonextremal eigenvector as the fields change.
            # A deterministic perturbation retains the warm start while
            # restoring overlap with every eigenspace; using `iter` preserves
            # checkpoint/restart bitwise reproducibility.
            if v_power !== nothing
                v_power .+= 0.1 .* randn(Xoshiro(iter), ComplexF64, length(v_power))
            end
            rad, v_power_new = spectral_radius(op; v0=v_power)
            rad > 0 || throw(ArgumentError("bdg_solve!: BdG operator has zero spectral radius"))
            v_power = v_power_new
            a = 2rad / (2 - rescale_eps)
            Hs = ScaledOperator(op, a, 0.0)
            mu_rho, mu_delta = bdg_site_moments(Hs, op.N, 1:op.N, Int(NC);
                                                 batch_size=Int(batch_size))
            n_new, Delta_new = bdg_update(mu_rho, mu_delta, a;
                                           U=op.U, beta=beta, g_rho=g_rho,
                                           kernel=kernel, Np=Int(Np))

            res_d_abs = norm(Delta_new .- op.Δ, Inf)
            res_d_rel = res_d_abs / max(norm(Delta_new, Inf), eps())
            delta_pass = res_d_abs <= tol_delta || res_d_rel <= tol_delta
            if update_density
                res_n_abs = norm(n_new .- op.n, Inf)
                res_n_rel = res_n_abs / max(norm(n_new, Inf), eps())
                n_pass = res_n_abs <= tol_n || res_n_rel <= tol_n
            else
                res_n_abs = 0.0
                n_pass = true
            end
            passes = delta_pass && n_pass
            converged = consecutive && passes
            consecutive = passes

            @. op.Δ = (1 - mix) * op.Δ + mix * Delta_new
            if update_density
                @. op.n = (1 - mix) * op.n + mix * n_new
            end

            entry = (iter=iter, res_delta=Float64(res_d_abs), res_n=Float64(res_n_abs),
                     max_delta=Float64(maximum(abs, op.Δ)),
                     mean_n=Float64(sum(op.n) / op.N), mu=op.μ, a=Float64(a))
            push!(history, entry)
            callback === nothing || callback(op, iter, entry)
            if checkpoint_path !== nothing &&
                    (iter % checkpoint_every == 0 || converged || local_iter == maxiter)
                bdg_checkpoint(checkpoint_path, op, history, v_power)
            end
            verbose >= 1 && println("BdG iter $(iter): res_delta=$(res_d_abs), res_n=$(res_n_abs), mu=$(op.μ), a=$(a)")
            converged && break
        end

        iterations = isempty(history) ? 0 : last(history).iter
        return BdGSCFResult(converged, iterations, Float64(res_d_abs),
                            Float64(res_n_abs), Float64(a), history)
    end

    target_filling === nothing && return inner_solve!()

    mu_lo, mu_hi = mu_bracket
    isfinite(mu_lo) && isfinite(mu_hi) ||
        throw(ArgumentError("bdg_solve!: target_filling requires finite mu_bracket endpoints"))
    mu_lo < mu_hi || throw(ArgumentError("bdg_solve!: mu_bracket must satisfy lo < hi"))

    function filling_error!(mu)
        op.μ = Float64(mu)
        result = inner_solve!()
        return sum(op.n) / op.N - target_filling, result
    end

    err_lo, result = filling_error!(mu_lo)
    abs(err_lo) <= mu_tol && return result
    err_hi, result = filling_error!(mu_hi)
    abs(err_hi) <= mu_tol && return result
    signbit(err_lo) == signbit(err_hi) &&
        throw(ArgumentError("bdg_solve!: target_filling is not bracketed by mu_bracket"))

    for _ in 1:Int(mu_maxiter)
        mu_mid = (mu_lo + mu_hi) / 2
        err_mid, result = filling_error!(mu_mid)
        abs(err_mid) <= mu_tol && return result
        if signbit(err_mid) == signbit(err_lo)
            mu_lo, err_lo = mu_mid, err_mid
        else
            mu_hi, err_hi = mu_mid, err_mid
        end
    end
    return result
end
