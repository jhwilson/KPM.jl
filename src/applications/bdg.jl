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
    spectral_radius(op; tol=1e-4, maxiter=300, miniter=32, restarts=2,
                    rng=Xoshiro(0), v0=nothing) -> (radius, v)

Estimate the spectral radius of a Hermitian matrix-free operator with norm-ratio
power iteration. The norm ratio converges to the spectral radius even when the
extremal eigenvalues occur as a `±Emax` pair, as in BdG systems. The default
deterministic RNG makes restarts reproducible, and `v0` enables a warm-started
primary run. No run exits for convergence before `miniter`, and `restarts`
additional runs use fresh random vectors. The maximum estimate and its vector
are returned. This remains a lower estimate; see [`rescale(::BdGOperator)`](@ref)
for certified bounds and runtime guards.
"""
function spectral_radius(op; tol=1e-4, maxiter=300, miniter::Integer=32,
                         restarts::Integer=2, rng=Xoshiro(0), v0=nothing)
    N = size(op, 1)
    maxiter > 0 || throw(ArgumentError("spectral_radius: maxiter must be positive (got $maxiter)"))
    1 <= miniter <= maxiter ||
        throw(ArgumentError("spectral_radius: miniter must satisfy 1 <= miniter <= maxiter (got miniter=$miniter, maxiter=$maxiter)"))
    restarts >= 0 || throw(ArgumentError("spectral_radius: restarts must be nonnegative (got $restarts)"))

    function power_run(v_start)
        v = collect(ComplexF64, v_start)
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
            v .= w ./ λ
            if iter >= miniter && abs(λ - λ_prev) <= tol * max(λ, eps())
                return Float64(λ), v
            end
            λ_prev = λ
        end

        @debug "spectral_radius run reached maxiter without convergence" maxiter miniter tol λ=λ_prev
        return Float64(λ_prev), v
    end

    primary = v0 === nothing ? randn(rng, ComplexF64, N) : v0
    radius_best, v_best = power_run(primary)
    for _ in 1:restarts
        radius, v = power_run(randn(rng, ComplexF64, N))
        if radius > radius_best
            radius_best, v_best = radius, v
        end
    end
    return radius_best, v_best
end

"""
    BdGOperator(h; mu, U, n=zeros(size(h, 1)),
                Delta=zeros(ComplexF64, size(h, 1)),
                hole_convention=:intervalley, h_hole=nothing,
                assume_intervalley=false)

Matrix-free reduced spin-singlet Nambu BdG operator with particle-hole layout
`[particle; hole]` and hole index `i + N`:

    H_BdG = [ ξ                         Diagonal(Δ)       ]
            [ Diagonal(conj(Δ))         hole              ]

with `ξ = h - μ I - Diagonal(U n / 2)`.

Here `U > 0` is attractive, with `H_int = -U Σ n↑n↓`, Hartree shift
`-(U/2)n`, and `Δ_i = -U_i⟨c_{i↓}c_{i↑}⟩`. This CPU-only reduced convention
supports two hole-block conventions:

  * `hole_convention=:intervalley` (the default) uses `hole = -ξ`, hence the
    same `h` in both blocks, and presumes `h_{-K}^* = h_K`. For matrix inputs a
    complex Hermitian `h` is rejected unless `assume_intervalley=true`
    explicitly asserts that it is the intervalley-identified operator. For
    complex `h` with a nonuniform gap phase its spectrum need not be
    particle-hole symmetric; the package's `b=0` rescaling remains a safe
    radial bound, not a symmetry statement.
  * `hole_convention=:singlet` is the standard same-valley
    `(c_up, c_down^dagger)` convention and uses `hole = -conj(ξ)`. It obeys the
    exact particle-hole symmetry `tau_y * conj(H) * tau_y = -H` for any complex
    Hermitian `h` and any spatially varying complex `Δ`. For an assembled
    matrix the conjugated hole operator is built automatically. Matrix-free
    callers must supply it as `h_hole`.

The conventions coincide identically for real-symmetric `h`.
"""
mutable struct BdGOperator{TH, THH}
    const h::TH
    const h_hole::THH
    μ::Float64
    const U::Vector{Float64}
    const n::Vector{Float64}
    const Δ::Vector{ComplexF64}
    const N::Int
    const hole_convention::Symbol
end

function BdGOperator(h; mu::Real, U, n=zeros(size(h, 1)),
                     Delta=zeros(ComplexF64, size(h, 1)),
                     hole_convention::Symbol=:intervalley, h_hole=nothing,
                     assume_intervalley::Bool=false)
    N = size(h, 1)
    size(h, 2) == N || throw(ArgumentError("BdGOperator: h must be square (got $(size(h)))"))
    hole_convention in (:intervalley, :singlet) ||
        throw(ArgumentError("BdGOperator: hole_convention must be :intervalley or :singlet (got $hole_convention)"))
    U_vec = U isa Number ? fill(Float64(U), N) : collect(Float64, U)
    n_vec = collect(Float64, n)
    Δ_vec = collect(ComplexF64, Delta)
    length(U_vec) == N || throw(ArgumentError("BdGOperator: U has length $(length(U_vec)); expected $N"))
    length(n_vec) == N || throw(ArgumentError("BdGOperator: n has length $(length(n_vec)); expected $N"))
    length(Δ_vec) == N || throw(ArgumentError("BdGOperator: Delta has length $(length(Δ_vec)); expected $N"))

    h_is_matrix = h isa AbstractMatrix
    if h_is_matrix
        ishermitian(h) || throw(ArgumentError("BdGOperator: matrix h must be Hermitian"))
    end

    h_hole_stored = if hole_convention === :intervalley
        h_hole === nothing ||
            throw(ArgumentError("BdGOperator: h_hole must be nothing for hole_convention=:intervalley"))
        if h_is_matrix && eltype(h) <: Complex
            values = h isa SparseMatrixCSC ? nonzeros(h) : h
            max_imag = maximum(x -> abs(imag(x)), values; init=0.0)
            max_abs = maximum(abs, values; init=0.0)
            symmetric = max_imag <= sqrt(eps()) * max(max_abs, 1e-300)
            if !symmetric && !assume_intervalley
                throw(ArgumentError("BdGOperator: for complex Hermitian h the reduced same-h hole block is only the correct physics if h is the intervalley-identified operator (h_{-K}^* = h_K); pass assume_intervalley=true to assert that, or use a real-symmetric h."))
            end
        end
        h
    else
        assume_intervalley &&
            throw(ArgumentError("BdGOperator: assume_intervalley is meaningless for hole_convention=:singlet; leave it false"))
        if h_hole === nothing
            h_is_matrix ||
                throw(ArgumentError("BdGOperator: matrix-free h with hole_convention=:singlet requires h_hole, the conjugated normal-state operator"))
            conj(h)
        else
            size(h_hole) == size(h) ||
                throw(ArgumentError("BdGOperator: h_hole has size $(size(h_hole)); expected $(size(h))"))
            h_hole
        end
    end
    return BdGOperator(h, h_hole_stored, Float64(mu), U_vec, n_vec, Δ_vec, N,
                       hole_convention)
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
    mul!(Yh, B.h_hole, Xh, -α, β)
    @. Yp += α * ((-B.μ - (B.U / 2) * B.n) * Xp + B.Δ * Xh)
    @. Yh += α * (conj(B.Δ) * Xp - (-B.μ - (B.U / 2) * B.n) * Xh)
    return Y
end

LinearAlgebra.mul!(Y::AbstractVecOrMat, B::BdGOperator, X::AbstractVecOrMat) =
    mul!(Y, B, X, true, false)

"""
    gershgorin_bound(op::BdGOperator) -> Float64

Certified Gershgorin upper bound on the spectral radius of an assembled BdG
operator. Off-diagonal hopping row sums are accumulated from the stored matrix
entries, while the diagonal is replaced explicitly by
`abs(h[i,i] - mu - U[i]*n[i]/2)` and the pairing contribution `abs(Delta[i])`
is added. Both Nambu blocks have the same bound by symmetry of this
construction. Matrix-free normal operators are not supported.
"""
function gershgorin_bound(op::BdGOperator)
    op.h isa AbstractMatrix ||
        throw(ArgumentError("gershgorin_bound: BdGOperator must have an assembled matrix h"))
    # The singlet hole block has identical row sums because abs(conj(h)) == abs(h).
    N = op.N
    iszero(N) && return 0.0
    rowsums = zeros(Float64, N)
    diagonal = zeros(ComplexF64, N)

    if op.h isa SparseMatrixCSC
        rows = rowvals(op.h)
        values = nonzeros(op.h)
        for j in axes(op.h, 2)
            for k in nzrange(op.h, j)
                i = rows[k]
                if i == j
                    diagonal[i] = values[k]
                else
                    rowsums[i] += abs(values[k])
                end
            end
        end
    else
        for j in axes(op.h, 2), i in axes(op.h, 1)
            if i == j
                diagonal[i] = op.h[i, j]
            else
                rowsums[i] += abs(op.h[i, j])
            end
        end
    end

    @inbounds for i in 1:N
        rowsums[i] += abs(diagonal[i] - op.μ - op.U[i] * op.n[i] / 2) + abs(op.Δ[i])
    end
    return maximum(rowsums)
end

function _check_chebyshev_columns(slot::AbstractMatrix, iteration::Integer)
    max_norm2 = 0.0
    @inbounds for c in axes(slot, 2)
        norm2 = 0.0
        for i in axes(slot, 1)
            norm2 += abs2(slot[i, c])
        end
        max_norm2 = max(max_norm2, norm2)
    end
    max_norm = sqrt(max_norm2)
    if !(max_norm <= 1.5)
        error("Chebyshev recurrence is unstable at iteration $iteration (maximum column norm $max_norm > 1.5); use rescale(...; bound=:gershgorin) or a larger eps.")
    end
    return nothing
end

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
        NC == 2 && _check_chebyshev_columns(psi_views[2], 1)

        ip = 2
        ipp = 1
        for m in 3:NC
            chebyshev_iter_single(Hs, psi_views[ipp], psi_views[ip])
            extract_moment!(m, psi_views[ipp])
            iteration = m - 1
            (iteration % 16 == 0 || m == NC) &&
                _check_chebyshev_columns(psi_views[ipp], iteration)
            ip = 3 - ip
            ipp = 3 - ipp
        end
    end

    return mu_rho, mu_delta
end

"""
    chebyshev_stability_probe(Hs, NH::Integer, NC::Integer; rng=Xoshiro(1)) -> Float64

Run the plain two-slot Chebyshev recurrence from one random unit vector for
`NC` moments and return the maximum vector norm encountered. Values above
`1.5` indicate that the rescaled spectrum has escaped the stable interval.
"""
function chebyshev_stability_probe(Hs, NH::Integer, NC::Integer; rng=Xoshiro(1))
    NH > 0 || throw(ArgumentError("chebyshev_stability_probe: NH must be positive (got $NH)"))
    size(Hs) == (NH, NH) ||
        throw(ArgumentError("chebyshev_stability_probe: Hs has size $(size(Hs)); expected ($NH, $NH)"))
    NC > 0 || throw(ArgumentError("chebyshev_stability_probe: NC must be positive (got $NC)"))

    previous = randn(rng, ComplexF64, NH)
    previous ./= norm(previous)
    max_norm = 1.0
    NC == 1 && return max_norm

    current = similar(previous)
    mul!(current, Hs, previous)
    max_norm = max(max_norm, norm(current))
    T = ComplexF64
    for _ in 3:NC
        # two-address Chebyshev step: previous <- 2 Hs current - previous
        mul!(previous, Hs, current, T(2), T(-1))
        max_norm = max(max_norm, norm(previous))
        previous, current = current, previous
    end
    return Float64(max_norm)
end

"""
    bdg_update(mu_rho, mu_delta, a; U, beta, g_rho=2.0,
               kernel=JacksonKernel, Np=2size(mu_rho, 1))

Update local BdG fields at Fermi level zero from the Jackson-dressed moments:

    n_c = g_rho / Np * sum_k gamma_rho_c(x_k) f(a x_k)
    Delta_c = -U_c / Np * sum_k gamma_delta_c(x_k) f(a x_k)

where `gamma_c(x) = sum_m kernel(m-1, NC) hn(m-1) mu[m,c]
T_{m-1}(x)` and `x_k` are Gauss-Chebyshev nodes. There is no extra
`a`-Jacobian: the spectral `1/a` cancels the `dE = a dx` measure exactly.
The reduced block integrates one spin species, so the default `g_rho=2`
reconstructs the full spin-singlet site density. Set `g_rho=1` for per-spin
density; the caller then owns the corresponding Hartree interpretation.
"""
function bdg_update(mu_rho::AbstractMatrix{<:Real},
                    mu_delta::AbstractMatrix{<:Complex}, a::Real;
                    U::AbstractVector{<:Real}, beta::Real, g_rho::Real=2.0,
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
    bdg_checkpoint(path, op, history, v_power, params) -> nothing

Atomically write the plain-data state needed to restart a BdG
self-consistency solve. The operator and callback are never serialized.
"""
function bdg_checkpoint(path::AbstractString, op::BdGOperator, history, v_power, params)
    state = (version=2,
             delta=copy(op.Δ),
             n=copy(op.n),
             mu=op.μ,
             U=copy(op.U),
             N=op.N,
             history=copy(history),
             v_power=v_power === nothing ? nothing : copy(v_power),
             params=params)
    tmp_path = "$(path).tmp"
    open(tmp_path, "w") do io
        serialize(io, state)
    end
    mv(tmp_path, path; force=true)
    return nothing
end

"""
    bdg_restore!(op, path) -> (history, v_power, params)

Restore a checkpointed BdG field state into `op`. The checkpoint must have
the same number of sites and interaction field as `op`. Version-1 checkpoints
are accepted and return `nothing` for their missing solver parameters.
"""
function bdg_restore!(op::BdGOperator, path::AbstractString)
    state = open(deserialize, path)
    state.version in (1, 2) || throw(ArgumentError("bdg_restore!: unsupported checkpoint version $(state.version)"))
    state.N == op.N || throw(ArgumentError("bdg_restore!: checkpoint has N=$(state.N); expected N=$(op.N)"))
    state.U ≈ op.U || throw(ArgumentError("bdg_restore!: checkpoint interaction U does not match the operator"))
    length(state.delta) == op.N || throw(ArgumentError("bdg_restore!: checkpoint delta has invalid length"))
    length(state.n) == op.N || throw(ArgumentError("bdg_restore!: checkpoint n has invalid length"))
    op.Δ .= state.delta
    op.n .= state.n
    op.μ = state.mu
    params = state.version == 1 ? nothing : state.params
    return copy(state.history), state.v_power === nothing ? nothing : copy(state.v_power), params
end

"""
    bdg_solve!(op; beta, NC=512, g_rho=2.0, mix=0.1, tol_delta=1e-6,
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
The reduced block integrates one spin species; default `g_rho=2` reconstructs
the full spin-singlet site density used by the Hartree term. On restart,
solver-parameter differences are warned about but allowed. Restart cannot be
combined with `target_filling` because the outer bisection state is not
checkpointed.
"""
function bdg_solve!(op::BdGOperator; beta::Real, NC::Integer=512,
                    g_rho::Real=2.0, mix::Real=0.1,
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
    restart !== nothing && target_filling !== nothing &&
        throw(ArgumentError("bdg_solve!: restart cannot be combined with target_filling because the bisection state is not checkpointed"))

    kernel_name = try
        string(nameof(kernel))
    catch
        string(nameof(typeof(kernel)))
    end
    params = (beta=beta, NC=NC, Np=Np, g_rho=g_rho, mix=mix,
              tol_delta=tol_delta, tol_n=tol_n, kernel=kernel_name,
              update_density=update_density)
    history, v_power, saved_params = restart === nothing ?
        (NamedTuple[], nothing, nothing) : bdg_restore!(op, restart)
    if saved_params !== nothing
        differences = String[]
        for name in propertynames(params)
            saved = getproperty(saved_params, name)
            current = getproperty(params, name)
            isequal(saved, current) || push!(differences, "$(name): saved=$(repr(saved)), current=$(repr(current))")
        end
        if !isempty(differences)
            diff_text = join(differences, "; ")
            @warn "bdg_solve!: restart parameters differ from checkpoint: $diff_text"
        end
    end

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
                bdg_checkpoint(checkpoint_path, op, history, v_power, params)
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
        result.converged ||
            error("bdg_solve!: target-filling inner solve did not converge at mu=$(op.μ); raise maxiter before continuing the bisection")
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
