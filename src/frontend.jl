# Typed KPM front end. Typed wrappers for cpge, dc_long, and ldos_mu remain
# follow-up work because those APIs take rescaled-unit energies.

"""
    RescaledHamiltonian(H, a, b)

A Hamiltonian rescaled as `H = (H_original - b I) / a`, with spectrum inside
`(-1, 1)`. `a` and `b` are in the physical energy units of the original
Hamiltonian.
"""
struct RescaledHamiltonian{TH}
    H::TH
    a::Float64
    b::Float64
end

"""
    AbstractMoments

Abstract supertype for KPM moment values that retain the Hamiltonian
rescaling and stochastic-trace metadata used to compute them.
"""
abstract type AbstractMoments end

"""
    DosMoments(mu, a, b, NH, NR)

One-dimensional density-of-states moments `mu` for `(H_original - b I) / a`.
`NH` is the Hilbert-space dimension and `NR` is the number of probe vectors.
`a` and `b` are in physical energy units.
"""
struct DosMoments{V<:AbstractVector{<:Real}} <: AbstractMoments
    mu::V
    a::Float64
    b::Float64
    NH::Int
    NR::Int
    function DosMoments(mu::V, a, b, NH, NR) where {V<:AbstractVector{<:Real}}
        NH > 0 && NR > 0 || throw(
            ArgumentError("DosMoments: NH and NR must be positive (got NH=$NH, NR=$NR)"),
        )
        new{V}(mu, a, b, NH, NR)
    end
end

"""
    ConductivityMoments(mu, a, b, NH, NR)

Two-dimensional conductivity moments `mu` for `(H_original - b I) / a`.
`NH` is the Hilbert-space dimension and `NR` is the number of probe vectors.
`a` and `b` are in physical energy units.
"""
struct ConductivityMoments{M<:AbstractMatrix{<:Complex}} <: AbstractMoments
    mu::M
    a::Float64
    b::Float64
    NH::Int
    NR::Int
    function ConductivityMoments(mu::M, a, b, NH, NR) where {M<:AbstractMatrix{<:Complex}}
        size(mu, 1) == size(mu, 2) || throw(
            ArgumentError(
                "ConductivityMoments: the moment matrix must be square (got $(size(mu)))",
            ),
        )
        NH > 0 && NR > 0 || throw(
            ArgumentError(
                "ConductivityMoments: NH and NR must be positive (got NH=$NH, NR=$NR)",
            ),
        )
        new{M}(mu, a, b, NH, NR)
    end
end

"""
    CurrentMoments(mu, a, b, NH, NR)

One-dimensional current moments `mu[n] = Tr[J*T_n(H_norm)]/NH` for
`H_norm = (H_original - b*I)/a`. `NH` is the Hilbert-space dimension and
`NR` the number of stochastic-trace probes.
"""
struct CurrentMoments{T} <: AbstractMoments
    mu::Vector{T}
    a::Float64
    b::Float64
    NH::Int
    NR::Int
end

"""
    GreenMoments(mu, a, b, NH)

Matrix-element moments `mu[n, p] = ⟨u_p|T_{n-1}(H_norm)|v_p⟩` (size
NC × npairs, complex) for `(H_original - b I) / a`. Unlike `DosMoments`,
these are deterministic matrix elements for caller-supplied probe pairs, not
a stochastic trace, so there is no `NR`; which pair a column belongs to is
the caller's bookkeeping (models are user data). Reconstruct with
[`greens`](@ref), [`ldos`](@ref), or [`spectral_function`](@ref).
"""
struct GreenMoments{M<:AbstractMatrix{<:Complex}} <: AbstractMoments
    mu::M
    a::Float64
    b::Float64
    NH::Int
    function GreenMoments(mu::M, a, b, NH) where {M<:AbstractMatrix{<:Complex}}
        NH > 0 || throw(ArgumentError("GreenMoments: NH must be positive (got NH=$NH)"))
        size(mu, 1) >= 1 || throw(
            ArgumentError(
                "GreenMoments: mu must contain at least one moment (got size $(size(mu)))",
            ),
        )
        (isfinite(a) && a > 0) ||
            throw(ArgumentError("GreenMoments: a must be finite and positive (got a=$a)"))
        isfinite(b) || throw(ArgumentError("GreenMoments: b must be finite (got b=$b)"))
        new{M}(mu, a, b, NH)
    end
end

# number of Chebyshev moments (NC is derived from the stored moments, not stored)
nc(m::DosMoments) = length(m.mu)
nc(m::ConductivityMoments) = size(m.mu, 1)
nc(m::CurrentMoments) = length(m.mu)
nc(m::GreenMoments) = size(m.mu, 1)
npairs(m::GreenMoments) = size(m.mu, 2)

Base.size(h::RescaledHamiltonian) = size(h.H)
Base.size(h::RescaledHamiltonian, d) = size(h.H, d)

Base.show(io::IO, h::RescaledHamiltonian) =
    print(io, "RescaledHamiltonian(NH=$(size(h.H, 1)), a=$(h.a), b=$(h.b))")
Base.show(io::IO, m::DosMoments) =
    print(io, "DosMoments(NC=$(nc(m)), NR=$(m.NR), NH=$(m.NH), a=$(m.a), b=$(m.b))")
Base.show(io::IO, m::ConductivityMoments) = print(
    io,
    "ConductivityMoments(NC=$(nc(m)), NR=$(m.NR), NH=$(m.NH), a=$(m.a), b=$(m.b))",
)
Base.show(io::IO, m::CurrentMoments) =
    print(io, "CurrentMoments(NC=$(nc(m)), NR=$(m.NR), NH=$(m.NH), a=$(m.a), b=$(m.b))")
Base.show(io::IO, m::GreenMoments) = print(
    io,
    "GreenMoments(NC=$(nc(m)), npairs=$(npairs(m)), NH=$(m.NH), a=$(m.a), b=$(m.b))",
)

"""
    rescale(H; center=false, eps=0.1, fixed_a=0.0)

Rescale Hermitian `H` for Chebyshev expansion and retain its physical-energy
provenance. With `center=true`, use `(H - b I) / a`; otherwise `b = 0`.
"""
function rescale(H; center::Bool = false, eps::Float64 = 0.1, fixed_a::Number = 0.0)
    if center
        a, b, H_norm = normalizeH(H; eps = eps, fixed_a = fixed_a, center = true)
    else
        a, H_norm = normalizeH(H; eps = eps, fixed_a = fixed_a)
        b = 0.0
    end
    af = Float64(a)
    bf = Float64(b)
    (isfinite(af) && isfinite(bf)) ||
        throw(ArgumentError("rescale: a and b must be finite as Float64 (got a=$a, b=$b)"))
    return RescaledHamiltonian(H_norm, af, bf)
end

"""
    rescale(op::BdGOperator; eps=0.2, bound=:power, radius=nothing, kwargs...)

BdG radial rescaling with center `b = 0` and
`a = 2 * radius / (2 - eps)`. `bound=:power` uses the hardened multi-start
[`spectral_radius`](@ref), which remains a lower estimate protected by
recurrence guards. `bound=:gershgorin` uses the certified
[`gershgorin_bound`](@ref) for assembled operators; its potentially larger
scale reduces physical resolution at fixed `NC`. A caller-supplied `radius`
overrides either estimator, for example when a bandwidth bound is known.
"""
function rescale(
    op::BdGOperator;
    eps::Real = 0.2,
    bound::Symbol = :power,
    radius::Union{Nothing,Real} = nothing,
    kwargs...,
)
    0 < eps < 2 || throw(ArgumentError("rescale: eps must satisfy 0 < eps < 2 (got $eps)"))
    bound in (:power, :gershgorin) ||
        throw(ArgumentError("rescale: bound must be :power or :gershgorin (got $bound)"))
    rad = if radius !== nothing
        Float64(radius)
    elseif bound == :power
        first(spectral_radius(op; kwargs...))
    else
        isempty(kwargs) || throw(
            ArgumentError(
                "rescale: power-iteration keyword arguments are not used with bound=:gershgorin",
            ),
        )
        gershgorin_bound(op)
    end
    a = 2 * rad / (2 - eps)
    isfinite(a) && a > 0 || throw(
        ArgumentError(
            "rescale: BdG scale a must be finite and positive (got $a); the zero operator cannot be rescaled",
        ),
    )
    return RescaledHamiltonian(ScaledOperator(op, a, 0.0), a, 0.0)
end

"""
    random_phase_vectors(rng, NH, NR)

Create `NR` unit-norm random-phase probes of length `NH` using `rng`. This is
the front-end reproducibility contract: the same fresh RNG state produces the
same probes, with no mean centering of the stochastic trace estimator.
"""
function random_phase_vectors(rng, NH::Integer, NR::Integer)
    psi = exp.(rand(rng, dt_real, NH, NR) .* (2im * pi))
    normalize_by_col(psi, NR)
    return psi
end

"""
    dos_moments(h; NC=1024, NR=12, rng=nothing, psi_in=nothing, verbose=0)

Compute DOS moments of the rescaled Hamiltonian in `h`. With `rng`, probes
follow `random_phase_vectors`: equal fresh RNG states reproduce equal moments.
With `psi_in`, its columns are used as supplied and determine `NR`.
"""
function dos_moments(
    h::RescaledHamiltonian;
    NC::Integer = 1024,
    NR::Integer = 12,
    rng = nothing,
    psi_in = nothing,
    verbose = 0,
)
    rng !== nothing &&
        psi_in !== nothing &&
        throw(ArgumentError("pass either rng or psi_in, not both"))
    iseven(NC) || throw(
        ArgumentError(
            "kpm_1d computes NC moments from NC/2 recurrence steps; NC must be even",
        ),
    )

    NH = size(h.H, 1)
    NC_int = Int(NC)
    NR_int = Int(NR)
    if psi_in !== nothing
        size(psi_in, 1) == NH ||
            throw(ArgumentError("psi_in has $(size(psi_in, 1)) rows; expected $NH"))
        NR_int = size(psi_in, 2)
    elseif rng !== nothing
        psi_in = random_phase_vectors(rng, NH, NR_int)
    end

    mu = if psi_in === nothing
        kpm_1d(h.H, NC_int, NR_int; verbose = verbose)
    else
        kpm_1d(h.H, NC_int, NR_int; psi_in = psi_in, verbose = verbose)
    end
    return DosMoments(mu, h.a, h.b, NH, NR_int)
end

"""
    cond_moments(h, Jα, Jβ; NC=256, NR=8, rng=nothing, psi_in=nothing, kwargs...)

Compute conductivity moments for `h` and current operators `Jα`, `Jβ`.
Random probes use `random_phase_vectors`, so equal fresh RNG states reproduce
equal moments. Current operators follow the bond convention
`(J_α)_ij = H_ij (r_i - r_j)_α` built from the original unrescaled `H`;
building them from `H_norm` divides results by `a²`.
"""
function cond_moments(
    h::RescaledHamiltonian,
    Jα,
    Jβ;
    NC::Integer = 256,
    NR::Integer = 8,
    rng = nothing,
    psi_in = nothing,
    arr_size = :auto,
    right_block = :auto,
    workspace_bytes = :auto,
    kwargs...,
)
    rng !== nothing &&
        psi_in !== nothing &&
        throw(ArgumentError("pass either rng or psi_in, not both"))

    NH = size(h.H, 1)
    NC_int = Int(NC)
    NR_int = Int(NR)
    if psi_in !== nothing
        size(psi_in, 1) == NH ||
            throw(ArgumentError("psi_in has $(size(psi_in, 1)) rows; expected $NH"))
        NR_int = size(psi_in, 2)
    elseif rng !== nothing
        psi_in = random_phase_vectors(rng, NH, NR_int)
    end

    mu = if psi_in === nothing
        kpm_2d(
            h.H,
            Jα,
            Jβ,
            NC_int,
            NR_int,
            NH;
            arr_size = arr_size,
            right_block = right_block,
            workspace_bytes = workspace_bytes,
            kwargs...,
        )
    else
        kpm_2d(
            h.H,
            Jα,
            Jβ,
            NC_int,
            NR_int,
            NH;
            psi_in = psi_in,
            arr_size = arr_size,
            right_block = right_block,
            workspace_bytes = workspace_bytes,
            kwargs...,
        )
    end
    return ConductivityMoments(mu, h.a, h.b, NH, NR_int)
end

"""
    current_moments(h, J, NC, NR; rng=nothing, kwargs...)

Compute one-dimensional current moments for the rescaled Hamiltonian in `h`.
With `rng`, probes follow [`random_phase_vectors`](@ref), so equal fresh RNG
states reproduce equal moments. `J` is user data and must be built from the
original, unrescaled Hamiltonian; building it from `h.H` changes its physical
normalization. Extra keywords are forwarded to [`kpm_1d_current`](@ref).
"""
function current_moments(
    h::RescaledHamiltonian,
    J,
    NC::Integer,
    NR::Integer;
    rng = nothing,
    kwargs...,
)
    NH = size(h.H, 1)
    NC_int = Int(NC)
    NR_int = Int(NR)
    mu = if rng === nothing
        kpm_1d_current(h.H, J, NC_int, NR_int, NH; kwargs...)
    else
        psi = random_phase_vectors(rng, NH, NR_int)
        kpm_1d_current(h.H, J, NC_int, NR_int, NH; psi_in = psi, kwargs...)
    end
    return CurrentMoments(mu, h.a, h.b, NH, NR_int)
end

function _check_optical_metadata(m2::ConductivityMoments, m1::CurrentMoments)
    (m1.a, m1.b, m1.NH) == (m2.a, m2.b, m2.NH) || throw(
        ArgumentError("current and conductivity moments must share a, b, and NH"),
    )
    return nothing
end

function _typed_optical_node_function(mu2s, mu1s, NC, omega_tilde, lambda_tilde, a)
    delta = zeros(ComplexF64, NC)
    gR = similar(delta)
    gA = similar(delta)
    rightR = [similar(delta) for _ in mu2s]
    rightD = [similar(delta) for _ in mu2s]
    function F!(out, theta)
        chebyshev_delta_theta!(delta, theta)
        x = cos(theta)
        green_coefficients!(gR, x + omega_tilde, lambda_tilde, Val(:R))
        green_coefficients!(gA, x - omega_tilde, lambda_tilde, Val(:A))
        @inbounds for k in eachindex(mu2s)
            mul!(rightR[k], mu2s[k], delta)
            mul!(rightD[k], mu2s[k], gA)
            paramagnetic = sum(gR .* rightR[k]) + sum(delta .* rightD[k])
            diamagnetic = mu1s[k] === nothing ? 0.0 + 0.0im : sum(mu1s[k] .* delta)
            out[k] = diamagnetic + paramagnetic / a
        end
        return out
    end
    return _SpectralNodeFunction(F!, length(mu2s))
end

"""
    optical_cond(m2, omega; area, Ef=0, beta=Inf, m1=nothing, lambda=0,
                 kernel=JacksonKernel, quad_rtol=1e-8, quad_atol=0,
                 maxevals=10^6)

Reconstruct the physical linear optical conductivity at physical frequency
`omega`, returning `sigma/(e^2/h)` for a two-dimensional sample of `area`.
`Ef`, `omega`, and `lambda` use the original Hamiltonian's energy units, and
`beta` is inverse physical energy; `lambda` implements JL's
`i0 -> i*lambda`. Current operators used to produce the moments must be built
from the unrescaled Hamiltonian.

The package table
`mu2[n,m] = Tr[Jalpha*T_m(H_norm)*Jbeta*T_n(H_norm)]/NH` is contracted
directly with `Lambda_nm`. This orientation reproduces
[`kubo_bastin_cond`](@ref) as `omega -> 0`, including the package's
ED/FHS-anchored convention: `Jalpha` is response, `Jbeta` is field, and
`sigma_xy = +C`. In the labels of Joao--Lopes Eqs. 26, 42, and 44, this is
their `sigma^{beta alpha}` because the two textbook Kubo forms differ by
exactly this relabeling.

The reconstruction integrates the diamagnetic term from optional `m1` and
the `1/a`-weighted paramagnetic term in one node function, preserving their
low-frequency cancellation before the quadrature error estimate. The integral
must satisfy `error[k] <= quad_atol + quad_rtol*abs(I[k])` for every batched
component; otherwise it throws with the available tolerance,
evaluation-budget, and broadening knobs. A nonzero `quad_atol` is needed for
exactly zero components. Cost is `O(N_nodes*NC^2)` per tensor component.
"""
function optical_cond(
    m2::ConductivityMoments,
    omega::Real;
    area::Real,
    Ef::Real = 0.0,
    beta::Real = Inf,
    m1::Union{CurrentMoments,Nothing} = nothing,
    lambda::Real = 0.0,
    kernel = JacksonKernel,
    quad_rtol::Real = 1e-8,
    quad_atol::Real = 0.0,
    maxevals::Integer = 10^6,
)
    values = optical_cond(
        (m2,),
        omega;
        area = area,
        Ef = Ef,
        beta = beta,
        m1s = m1 === nothing ? nothing : (m1,),
        lambda = lambda,
        kernel = kernel,
        quad_rtol = quad_rtol,
        quad_atol = quad_atol,
        maxevals = maxevals,
    )
    return only(values)
end

function optical_cond(
    m2::ConductivityMoments,
    omega::AbstractVector{<:Real};
    kwargs...,
)
    return ComplexF64[optical_cond(m2, frequency; kwargs...) for frequency in omega]
end

function optical_cond(
    m2s::Tuple{Vararg{ConductivityMoments}},
    omega::Real;
    area::Real,
    Ef::Real = 0.0,
    beta::Real = Inf,
    m1s = nothing,
    lambda::Real = 0.0,
    kernel = JacksonKernel,
    quad_rtol::Real = 1e-8,
    quad_atol::Real = 0.0,
    maxevals::Integer = 10^6,
)
    isempty(m2s) && return ComplexF64[]
    iszero(omega) && throw(ArgumentError("optical conductivity is singular at omega = 0"))
    area > 0 || throw(ArgumentError("area must be positive"))
    lambda >= 0 || throw(ArgumentError("lambda must be nonnegative (got $lambda)"))

    reference = first(m2s)
    NC = nc(reference)
    for moments in m2s
        (moments.a, moments.b, moments.NH, nc(moments)) ==
            (reference.a, reference.b, reference.NH, NC) || throw(
            ArgumentError("batched conductivity moments must share a, b, NH, and NC"),
        )
    end
    current = if m1s === nothing
        ntuple(_ -> nothing, length(m2s))
    else
        length(m1s) == length(m2s) ||
            throw(ArgumentError("m1s and m2s must have the same length"))
        for (m2, m1) in zip(m2s, m1s)
            m1 === nothing || _check_optical_metadata(m2, m1)
        end
        m1s
    end

    mu2s = tuple(
        (maybe_to_host(mu2D_apply_kernel_and_h(m.mu, NC, kernel)) for m in m2s)...,
    )
    mu1s = tuple(
        (
            m === nothing ?
            nothing :
            maybe_to_host(muND_apply_kernel_and_h(view(m.mu, 1:NC), NC, kernel; dims = [1])) for
            m in current
        )...,
    )
    omega_tilde = omega / reference.a
    lambda_tilde = lambda / reference.a
    xF = (Ef - reference.b) / reference.a
    beta_a = beta * reference.a
    F! = _typed_optical_node_function(
        mu2s,
        mu1s,
        NC,
        omega_tilde,
        lambda_tilde,
        reference.a,
    )
    integral, _ = _spectral_integral(
        F!,
        NC,
        (omega_tilde, -omega_tilde),
        xF,
        beta_a,
        lambda_tilde;
        rtol = quad_rtol,
        atol = quad_atol,
        maxevals = maxevals,
    )
    prefactor = -2pi * im * reference.NH / (area * omega)
    return ComplexF64.(prefactor .* integral)
end

"""
    green_moments(h, psi_l, psi_r; NC=1024, verbose=0)

Compute complex matrix-element moments `⟨u_p|T_n(H_norm)|v_p⟩` for
independent bra/ket probe blocks: column p of `psi_l` pairs with column p of
`psi_r`. This path cannot use moment doubling, so it runs the full NC-step
recurrence (2x the matvecs of the equal-vector method) and accepts odd `NC`.
Vectors are accepted as single-pair blocks. Returns [`GreenMoments`](@ref).
"""
function green_moments(
    h::RescaledHamiltonian,
    psi_l::AbstractVecOrMat,
    psi_r::AbstractVecOrMat;
    NC::Integer = 1024,
    verbose = 0,
)
    NC >= 2 || throw(ArgumentError("NC must be at least 2 (got $NC)"))
    NH = size(h.H, 1)
    ψl = psi_l isa AbstractVector ? reshape(psi_l, :, 1) : psi_l
    ψr = psi_r isa AbstractVector ? reshape(psi_r, :, 1) : psi_r
    size(ψl, 1) == NH || throw(ArgumentError("psi_l has $(size(ψl, 1)) rows; expected $NH"))
    size(ψl) == size(ψr) || throw(
        ArgumentError(
            "psi_l and psi_r must have equal size (got $(size(ψl)) and $(size(ψr)))",
        ),
    )
    M = size(ψl, 2)
    mu_all = zeros(dt_cplx, M, Int(NC))
    kpm_1d!(h.H, Int(NC), M, NH, mu_all, ψl, ψr; verbose = verbose)
    return GreenMoments(permutedims(mu_all), h.a, h.b, NH)
end

"""
    green_moments(h, psi; NC=1024, verbose=0)

Equal-probe diagonal block: moments `⟨u_p|T_n(H_norm)|u_p⟩` for each column
of `psi`, using the moment-doubling recurrence (`NC` must be even). The
stored moments are complex with zero imaginary part to roundoff for a
Hermitian operator. Returns [`GreenMoments`](@ref).
"""
function green_moments(
    h::RescaledHamiltonian,
    psi::AbstractVecOrMat;
    NC::Integer = 1024,
    verbose = 0,
)
    NC >= 2 || throw(ArgumentError("NC must be at least 2 (got $NC)"))
    iseven(NC) || throw(
        ArgumentError(
            "kpm_1d computes NC moments from NC/2 recurrence steps; NC must be even",
        ),
    )
    NH = size(h.H, 1)
    ψ = psi isa AbstractVector ? reshape(psi, :, 1) : psi
    size(ψ, 1) == NH || throw(ArgumentError("psi has $(size(ψ, 1)) rows; expected $NH"))
    M = size(ψ, 2)
    mu_all = zeros(dt_cplx, M, Int(NC))
    kpm_1d!(h.H, Int(NC), M, NH, mu_all, ψ; verbose = verbose)
    return GreenMoments(permutedims(mu_all), h.a, h.b, NH)
end

"""
    ldos_moments(h; sites, NC=1024, batch_size=64, verbose=0)

Site-diagonal moments `⟨i|T_n(H_norm)|i⟩` for the listed site indices, seeded
with unit basis vectors in batches of `batch_size` columns (each batch is one
doubling recurrence, so `NC` must be even). Returns [`GreenMoments`](@ref)
with one column per site, in the order given. Site indices are positions in
the caller's Hilbert-space basis; no lattice structure is inferred.
"""
function ldos_moments(
    h::RescaledHamiltonian;
    sites::AbstractVector{<:Integer},
    NC::Integer = 1024,
    batch_size::Integer = 64,
    verbose = 0,
)
    NC >= 2 || throw(ArgumentError("NC must be at least 2 (got $NC)"))
    iseven(NC) || throw(
        ArgumentError(
            "kpm_1d computes NC moments from NC/2 recurrence steps; NC must be even",
        ),
    )
    batch_size > 0 || throw(ArgumentError("batch_size must be positive (got $batch_size)"))
    NH = size(h.H, 1)
    all(s -> 1 <= s <= NH, sites) || throw(ArgumentError("site indices must lie in 1:$NH"))
    NC_int = Int(NC)
    mu = zeros(dt_cplx, NC_int, length(sites))
    for lo = 1:Int(batch_size):length(sites)
        hi = min(lo + Int(batch_size) - 1, length(sites))
        nb = hi - lo + 1
        ψ = zeros(dt_cplx, NH, nb)
        for (c, s) in enumerate(view(sites, lo:hi))
            ψ[s, c] = 1
        end
        mu_batch = zeros(dt_cplx, nb, NC_int)
        kpm_1d!(h.H, NC_int, nb, NH, mu_batch, ψ; verbose = verbose)
        mu[:, lo:hi] .= permutedims(mu_batch)
    end
    return GreenMoments(mu, h.a, h.b, NH)
end

"""
    greens(m::GreenMoments, E; kernel=..., eta=..., branch=:retarded, NC=...)

Reconstruct the full complex retarded/advanced Green function at physical
energies `E` from typed moments; see [`greens`](@ref) for the formula and the
two broadening routes. The stored rescaling shift cannot be overridden.
"""
function greens(m::GreenMoments, E; kwargs...)
    haskey(kwargs, :b) &&
        throw(ArgumentError("b is stored in the moments object; do not pass it separately"))
    return greens(m.mu, m.a, E; b = m.b, kwargs...)
end

"""
    ldos(m::GreenMoments, E; kernel=..., eta=..., NC=...)

Local density of states `-Im G^R/π` at physical energies `E` from typed
site-diagonal moments (see [`ldos_moments`](@ref)); real output, one column
per probe. The stored rescaling shift cannot be overridden.
"""
function ldos(m::GreenMoments, E; kwargs...)
    haskey(kwargs, :b) &&
        throw(ArgumentError("b is stored in the moments object; do not pass it separately"))
    return ldos(m.mu, m.a, E; b = m.b, kwargs...)
end

"""
    spectral_function(m::GreenMoments, E; kernel=..., eta=..., NC=...)

Spectral function `A = (i/2π)(G^R - G^A)` at physical energies `E` from typed
moments; complex for independent bra/ket pairs, real and nonnegative on
diagonal pairs. The stored rescaling shift cannot be overridden.
"""
function spectral_function(m::GreenMoments, E; kwargs...)
    haskey(kwargs, :b) &&
        throw(ArgumentError("b is stored in the moments object; do not pass it separately"))
    return spectral_function(m.mu, m.a, E; b = m.b, kwargs...)
end

"""
    spectral_weights(m::GreenMoments)

Integrated spectral weight per probe pair, `∫ A_uv(E) dE = μ₀ = ⟨u|v⟩`,
exact by Chebyshev orthogonality (1 for a normalized diagonal probe). Use it
as the sum-rule reference for a reconstructed spectral function.
"""
spectral_weights(m::GreenMoments) = m.mu[1, :]

"""
    evolve(h::RescaledHamiltonian, psi0, t::Real; NC=0, tol=1e-12, check_every=16, verbose=0)
    evolve(h::RescaledHamiltonian, psi0, ts::AbstractVector{<:Real}; ...)

Unitary time evolution ``e^{-iHt}\\,ψ_0`` for the time-independent
``H = a\\,H_{\\rm norm} + b\\,I`` by Chebyshev propagation,

```math
e^{-iHt}|ψ_0\\rangle = e^{-ibt}\\Big[J_0(at)\\,T_0(H_{\\rm norm})
    + 2\\sum_{n\\ge 1}(-i)^n J_n(at)\\,T_n(H_{\\rm norm})\\Big]|ψ_0\\rangle .
```

No Jackson or Lorentz kernel is applied — the converged series is unitary,
and the tests pin norm conservation, reversibility (`t < 0` is valid), and
the group property. By default the order is chosen adaptively from the
superexponential Bessel tail ([`evolution_order`](@ref), tolerance `tol`);
a caller-fixed `NC` that truncates too early triggers a tail warning.

`psi0` is a host vector or `NH × NR` block; times may be a scalar or a
vector. Returns a plain host array of states, dropping singleton axes:
`NH`, `NH × NR`, `NH × NT`, or `NH × NR × NT`. All requested times share one
Chebyshev recurrence with `NT` accumulators (`NT` extra state blocks of
memory; loop over scalar calls instead if that is the binding constraint).
For plain dense/sparse `h.H` the recurrence runs on the active device;
operators without device-transfer methods (e.g. `Hermitian`/`Symmetric`
wrappers) stay on the host, workspaces following the operator. Propagation
is rejected by the relative stability guard of [`chebyshev_action!`](@ref)
when the recurrence grows unstably in the propagated states (a symptom of a
rescaled spectrum outside the Chebyshev domain; the guard sees only the
probe subspace).
"""
function evolve(
    h::RescaledHamiltonian,
    psi0::AbstractVecOrMat,
    ts::AbstractVector{<:Real};
    NC::Integer = 0,
    tol::Real = 1e-12,
    check_every::Integer = 16,
    verbose::Integer = 0,
)
    NH = size(h.H, 1)
    size(psi0, 1) == NH ||
        throw(ArgumentError("psi0 has $(size(psi0, 1)) rows; expected NH = $NH"))
    isempty(ts) && throw(ArgumentError("ts must not be empty"))
    psi_mat = psi0 isa AbstractVector ? reshape(psi0, :, 1) : psi0
    Hd = maybe_to_device(h.H, dt_cplx)
    out = device_zeros_of(Hd, dt_cplx, NH, size(psi_mat, 2), length(ts))
    evolve!(
        out,
        Hd,
        h.a,
        h.b,
        psi_mat,
        ts;
        NC = NC,
        tol = tol,
        check_every = check_every,
        verbose = verbose,
    )
    out_host = maybe_to_host(out)
    return psi0 isa AbstractVector ? dropdims(out_host; dims = 2) : out_host
end

function evolve(h::RescaledHamiltonian, psi0::AbstractVecOrMat, t::Real; kwargs...)
    out = evolve(h, psi0, [t]; kwargs...)
    return dropdims(out; dims = ndims(out))
end

"""
    fermi_projector(h::RescaledHamiltonian, V; Ef, beta=Inf, NC,
                    kernel=JacksonKernel, check_every=16, verbose=0)

Apply the Fermi operator ``P = f_\\beta(H - E_F)`` to a host vector or
`NH × NR` block `V`, returning a plain host array of the same shape
(complex). `h.H` must be Hermitian (the rescaling contract). At
`beta = Inf` (default) `P` projects onto the occupied subspace below `Ef` —
sharp and accurate when the Jackson-damped resolution
``\\Delta E \\approx \\pi\\,a/NC`` sits well inside a spectral gap at `Ef`;
an eigenvalue exactly at `Ef` takes weight `1/2` (the midpoint convention
of the package step [`fermiFunctions`](@ref), and what the Chebyshev series
converges to at a jump), so place `Ef` in a gap for projector semantics. At
finite `beta` the occupation is thermally smoothed and the operator is
deliberately **not** idempotent — a Fermi–Dirac operator, not a subspace
projector. Coefficients come from [`fermi_coefficients`](@ref) (see there
for the required-`NC` policy and kernel choice), the action from
[`chebyshev_action!`](@ref), which also supplies the recurrence stability
guard and device residence: the recurrence runs on the active device for
plain dense/sparse `h.H`.
"""
function fermi_projector(
    h::RescaledHamiltonian,
    V::AbstractVecOrMat;
    Ef::Real,
    beta::Real = Inf,
    NC::Integer,
    kernel = JacksonKernel,
    check_every::Integer = 16,
    verbose::Integer = 0,
)
    NH = size(h.H, 1)
    size(V, 1) == NH || throw(ArgumentError("V has $(size(V, 1)) rows; expected NH = $NH"))
    Vmat = V isa AbstractVector ? reshape(V, :, 1) : V
    C = fermi_coefficients(h.a, h.b, Ef; beta = beta, NC = NC, kernel = kernel)
    Hd = maybe_to_device(h.H, dt_cplx)
    out = device_zeros_of(Hd, dt_cplx, NH, size(Vmat, 2))
    chebyshev_action!(out, Hd, Vmat, C; check_every = check_every, verbose = verbose)
    out_host = maybe_to_host(out)
    return V isa AbstractVector ? dropdims(out_host; dims = 2) : out_host
end

"""
    chern_marker(h::RescaledHamiltonian, x, y; Ef, sites, beta=Inf, NC,
                 kernel=JacksonKernel, batch_size=32, check_every=16,
                 verbose=0) -> Vector{Float64}

Bianco–Resta local Chern marker ``m_i = -4\\pi\\,\\mathrm{Im}\\,\\langle i| P\\,X\\,Q\\,Y\\,P|i\\rangle`` at the basis indices `sites`, with the
KPM Fermi projector ``P = f_\\beta(H - E_F)`` ([`fermi_coefficients`](@ref);
`NC` is required — see there), ``Q = I - P``, and diagonal position
operators from the caller-supplied coordinate vectors `x`, `y`. The sign
convention matches the package's Hall conductivity: the bulk average over
complete cells ([`chern_marker_average`](@ref), explicit area) equals the
same `+C` as ``\\sigma_{xy} = +C\\,e^2/h``.

Geometry is user data: coordinates, site grouping, and areas are inputs,
never inferred from `H` — the raw markers are per **orbital** and carry
units of x·y. `h.H` must be Hermitian (the contraction uses ``\\langle Pv|`` as the bra). Valid for **open boundaries only**: a diagonal position
operator is not a legal position observable on a torus. At `beta = Inf` the
marker summed over the whole finite sample converges to the exact identity
``\\mathrm{Im}\\,\\mathrm{Tr}[PXQYP] = 0`` — topology is read from a bulk
average with the boundary excluded, never the full trace. At finite `beta`
the routine returns the marker built from the smoothed Fermi operator, not
the finite-temperature Hall conductivity ``\\sigma_{xy}(T)``: its local
spectral weight is ``f_a f_c(1-f_b)`` rather than the Kubo--Bastin
``[\\min(f_a,f_c)-f_b]_+``, and the two coincide only at zero temperature.
The finite-`beta` whole-sample sum can therefore be genuinely nonzero because
the Fermi--Dirac operator is not idempotent, while its bulk average still
tracks `C` for temperatures well below the gap. Cost is two `NC`-step
recurrences per `batch_size` sites (five complex `NH × batch_size` device
workspaces, ≈ `80·NH·batch_size` bytes); for a regional average that does not need
every site, see [`chern_marker_region`](@ref).
"""
function chern_marker(
    h::RescaledHamiltonian,
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real};
    Ef::Real,
    sites::AbstractVector{<:Integer},
    beta::Real = Inf,
    NC::Integer,
    kernel = JacksonKernel,
    batch_size::Integer = 32,
    check_every::Integer = 16,
    verbose::Integer = 0,
)
    NH = size(h.H, 1)
    length(x) == NH || throw(ArgumentError("x has length $(length(x)); expected NH = $NH"))
    length(y) == NH || throw(ArgumentError("y has length $(length(y)); expected NH = $NH"))
    all(isfinite, x) && all(isfinite, y) ||
        throw(ArgumentError("coordinate vectors must be finite"))
    isempty(sites) && throw(ArgumentError("sites must not be empty"))
    all(s -> 1 <= s <= NH, sites) || throw(ArgumentError("sites must lie in 1:$NH"))
    batch_size >= 1 || throw(ArgumentError("batch_size must be >= 1"))

    C = fermi_coefficients(h.a, h.b, Ef; beta = beta, NC = NC, kernel = kernel)
    Hd = maybe_to_device(h.H, dt_cplx)
    xd = to_device_of(Hd, Vector{dt_real}(x))
    yd = to_device_of(Hd, Vector{dt_real}(y))

    markers = Vector{dt_real}(undef, length(sites))
    nb = min(Int(batch_size), length(sites))
    U, YU, W, s1, s2 = (device_zeros_of(Hd, dt_cplx, NH, nb) for _ = 1:5)
    for lo = 1:batch_size:length(sites)
        hi = min(lo + batch_size - 1, length(sites))
        if hi - lo + 1 != nb   # smaller final batch: fresh plain workspaces
            nb = hi - lo + 1
            U, YU, W, s1, s2 = (device_zeros_of(Hd, dt_cplx, NH, nb) for _ = 1:5)
        end
        V = zeros(dt_cplx, NH, nb)
        for (c, s) in enumerate(view(sites, lo:hi))
            V[s, c] = 1
        end
        r = _pxqyp_imdiag!(
            Hd,
            C,
            xd,
            yd,
            V,
            U,
            YU,
            W,
            (s1, s2);
            check_every = check_every,
            verbose = verbose,
        )
        markers[lo:hi] .= (4π) .* r
    end
    return markers
end

"""
    chern_marker_region(h::RescaledHamiltonian, x, y; Ef, region, rng,
                        NR=16, beta=Inf, NC, kernel=JacksonKernel,
                        batch_size=32, check_every=16, verbose=0)
                        -> Vector{Float64}

Stochastic estimate of the **region-summed** local Chern marker
``\\sum_{i \\in R} m_i`` ([`chern_marker`](@ref)) from `NR` random-phase
probes supported on the basis indices `region` (duplicates are rejected).
Returns the length-`NR` vector of independent per-probe estimates: `mean`
of it is the estimate, `std/√NR` its statistical error, and
`chern_marker_average(mean(est); area)` (explicit region area) gives the
regional Chern-number estimate. **Each entry already estimates the entire
region sum** — do not pass the returned vector itself to
[`chern_marker_average`](@ref), which would multiply the answer by `NR`;
average the probes first. Probes come from
[`random_phase_vectors`](@ref) under the package reproducibility contract
(`rng = Xoshiro(seed)` ⇒ identical probes on every device).

Cost is two `NC`-step recurrences per `batch_size` probes — independent of
`|R|` — so this replaces `|R|` deterministic site columns with `NR` probes;
for small regions the deterministic mode (`sites = region`, then `sum`) is
both exact and cheaper.
"""
function chern_marker_region(
    h::RescaledHamiltonian,
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real};
    Ef::Real,
    region::AbstractVector{<:Integer},
    rng,
    NR::Integer = 16,
    beta::Real = Inf,
    NC::Integer,
    kernel = JacksonKernel,
    batch_size::Integer = 32,
    check_every::Integer = 16,
    verbose::Integer = 0,
)
    NH = size(h.H, 1)
    length(x) == NH || throw(ArgumentError("x has length $(length(x)); expected NH = $NH"))
    length(y) == NH || throw(ArgumentError("y has length $(length(y)); expected NH = $NH"))
    all(isfinite, x) && all(isfinite, y) ||
        throw(ArgumentError("coordinate vectors must be finite"))
    isempty(region) && throw(ArgumentError("region must not be empty"))
    all(s -> 1 <= s <= NH, region) || throw(ArgumentError("region must lie in 1:$NH"))
    allunique(region) || throw(ArgumentError("region must not contain duplicates"))
    NR >= 1 || throw(ArgumentError("NR must be >= 1"))
    batch_size >= 1 || throw(ArgumentError("batch_size must be >= 1"))

    C = fermi_coefficients(h.a, h.b, Ef; beta = beta, NC = NC, kernel = kernel)
    Hd = maybe_to_device(h.H, dt_cplx)
    xd = to_device_of(Hd, Vector{dt_real}(x))
    yd = to_device_of(Hd, Vector{dt_real}(y))

    # unit-norm columns on |R| rows estimate tr_R[·]/|R|; the |R| factor
    # below converts each probe to a region-summed estimate
    phases = random_phase_vectors(rng, length(region), Int(NR))

    estimates = Vector{dt_real}(undef, NR)
    nb = min(Int(batch_size), Int(NR))
    U, YU, W, s1, s2 = (device_zeros_of(Hd, dt_cplx, NH, nb) for _ = 1:5)
    for lo = 1:batch_size:NR
        hi = min(lo + batch_size - 1, NR)
        if hi - lo + 1 != nb   # smaller final batch: fresh plain workspaces
            nb = hi - lo + 1
            U, YU, W, s1, s2 = (device_zeros_of(Hd, dt_cplx, NH, nb) for _ = 1:5)
        end
        V = zeros(dt_cplx, NH, nb)
        for (c, p) in enumerate(lo:hi), (k, s) in enumerate(region)
            V[s, c] = phases[k, p]
        end
        r = _pxqyp_imdiag!(
            Hd,
            C,
            xd,
            yd,
            V,
            U,
            YU,
            W,
            (s1, s2);
            check_every = check_every,
            verbose = verbose,
        )
        estimates[lo:hi] .= (4π * length(region)) .* r
    end
    return estimates
end

"""
    dos(m::DosMoments; kwargs...)

Reconstruct the DOS from typed moments at physical energies. The center shift
`b` is retained by `m` and cannot be overridden.
"""
function dos(m::DosMoments; kwargs...)
    haskey(kwargs, :b) &&
        throw(ArgumentError("b is stored in the moments object; do not pass it separately"))
    return dos(m.mu, m.a; b = m.b, kwargs...)
end

"""
    dos(h::RescaledHamiltonian; NC=1024, NR=12, rng=nothing, psi_in=nothing, verbose=0, kwargs...)

Compute typed DOS moments for `h` and reconstruct the physical-energy DOS.
Probe reproducibility is the same as for `dos_moments`.
"""
function dos(
    h::RescaledHamiltonian;
    NC::Integer = 1024,
    NR::Integer = 12,
    rng = nothing,
    psi_in = nothing,
    verbose = 0,
    kwargs...,
)
    m = dos_moments(h; NC = NC, NR = NR, rng = rng, psi_in = psi_in, verbose = verbose)
    return dos(m; kwargs...)
end

"""
    dos0(m::DosMoments; kwargs...)

Evaluate the DOS reconstruction at the rescaling center, physical energy
`E = m.b`.
"""
dos0(m::DosMoments; kwargs...) = dos0(m.mu, m.a; kwargs...)

"""
    kubo_bastin_cond(m::ConductivityMoments, Ef; area, kwargs...)

Evaluate Kubo–Bastin conductivity at physical Fermi energy `Ef`. The stored
rescaling shift and Hilbert-space dimension cannot be overridden.
"""
function kubo_bastin_cond(m::ConductivityMoments, Ef::Real; area::Real, kwargs...)
    haskey(kwargs, :b) &&
        throw(ArgumentError("b is stored in the moments object; do not pass it separately"))
    haskey(kwargs, :NH) && throw(
        ArgumentError("NH is stored in the moments object; do not pass it separately"),
    )
    return kubo_bastin_cond(m.mu, m.a, Ef; b = m.b, NH = m.NH, area = area, kwargs...)
end

"""
    d_dc_cond(m::ConductivityMoments, E; kwargs...)

Evaluate the differential DC conductivity at physical energies `E`. A vector
input returns a vector; `b` is retained by the moments object.
"""
function d_dc_cond(m::ConductivityMoments, E::AbstractVector{<:Real}; kwargs...)
    haskey(kwargs, :b) &&
        throw(ArgumentError("b is stored in the moments object; do not pass it separately"))
    return d_dc_cond(m.mu, m.a, collect(Float64, E); b = m.b, kwargs...)
end

"""
    d_dc_cond(m::ConductivityMoments, E; kwargs...)

Evaluate differential DC conductivity at physical scalar energy `E`. Unlike
the legacy scalar method, this typed method unwraps the length-one result.
"""
d_dc_cond(m::ConductivityMoments, E::Real; kwargs...) = only(d_dc_cond(m, [E]; kwargs...))

"""
    dc_cond0(m::ConductivityMoments; kwargs...)

Evaluate the bare DC conductivity sum at the rescaling center, physical
energy `E = m.b`.
"""
dc_cond0(m::ConductivityMoments; kwargs...) = dc_cond0(m.mu, m.a; kwargs...)

"""
    dc_cond_single(m::ConductivityMoments, Ef; kwargs...)

Evaluate the bare DC conductivity sum at physical Fermi energy `Ef`; the
rescaling shift `b` stored by `m` cannot be overridden.
"""
function dc_cond_single(m::ConductivityMoments, Ef::Real; kwargs...)
    haskey(kwargs, :b) &&
        throw(ArgumentError("b is stored in the moments object; do not pass it separately"))
    return dc_cond_single(m.mu, m.a, Ef; b = m.b, kwargs...)
end

"""
    two_energy_response(m::ConductivityMoments; beta, eta, volume, kwargs...)

Reconstruct a generic two-energy response from typed moments. The stored
rescaling shift and Hilbert-space dimension cannot be overridden.
"""
function two_energy_response(
    m::ConductivityMoments;
    beta::Real,
    eta::Real,
    volume::Real,
    kwargs...,
)
    haskey(kwargs, :b) &&
        throw(ArgumentError("b is stored in the moments object; do not pass it separately"))
    haskey(kwargs, :NH) && throw(
        ArgumentError("NH is stored in the moments object; do not pass it separately"),
    )
    return two_energy_response(
        m.mu,
        m.a;
        b = m.b,
        beta = beta,
        eta = eta,
        NH = m.NH,
        volume = volume,
        kwargs...,
    )
end

"""
    transport_distribution(m::ConductivityMoments, E; volume, kwargs...)

Reconstruct the symmetric, dissipative transport distribution at physical
energies `E` from typed moments. A vector input returns a vector and a scalar
input returns a scalar. Units are as in [`transport_distribution`](@ref); only
the symmetric part of `sigma_alphabeta` is reconstructed. The stored rescaling
shift and Hilbert-space dimension cannot be overridden.
"""
function transport_distribution(m::ConductivityMoments, E; volume::Real, kwargs...)
    haskey(kwargs, :b) &&
        throw(ArgumentError("b is stored in the moments object; do not pass it separately"))
    haskey(kwargs, :NH) && throw(
        ArgumentError("NH is stored in the moments object; do not pass it separately"),
    )
    return transport_distribution(
        m.mu,
        m.a,
        E;
        b = m.b,
        NH = m.NH,
        volume = volume,
        kwargs...,
    )
end

"""
    transport_integrals(m::ConductivityMoments, mu_chem; beta, volume, kwargs...)

Compute Chester–Thellung/Kubo–Greenwood transport integrals from typed moments
at physical chemical potential `mu_chem`. See [`transport_integrals`](@ref) for
units and numerical controls. The stored rescaling shift and Hilbert-space
dimension cannot be overridden.
"""
function transport_integrals(
    m::ConductivityMoments,
    mu_chem::Real;
    beta::Real,
    volume::Real,
    kwargs...,
)
    haskey(kwargs, :b) &&
        throw(ArgumentError("b is stored in the moments object; do not pass it separately"))
    haskey(kwargs, :NH) && throw(
        ArgumentError("NH is stored in the moments object; do not pass it separately"),
    )
    return transport_integrals(
        m.mu,
        m.a,
        mu_chem;
        beta = beta,
        b = m.b,
        NH = m.NH,
        volume = volume,
        kwargs...,
    )
end

"""
    thermoelectric(m::ConductivityMoments, mu_chem; beta, volume, kwargs...)

Compute a [`ThermoelectricResult`](@ref) from typed conductivity moments at
physical chemical potential `mu_chem`. The stored rescaling shift and
Hilbert-space dimension cannot be overridden. See [`thermoelectric`](@ref) for
the physical scope, units, and numerical controls.
"""
function thermoelectric(
    m::ConductivityMoments,
    mu_chem::Real;
    beta::Real,
    volume::Real,
    kwargs...,
)
    haskey(kwargs, :b) &&
        throw(ArgumentError("b is stored in the moments object; do not pass it separately"))
    haskey(kwargs, :NH) && throw(
        ArgumentError("NH is stored in the moments object; do not pass it separately"),
    )
    return thermoelectric(
        m.mu,
        m.a,
        mu_chem;
        beta = beta,
        b = m.b,
        NH = m.NH,
        volume = volume,
        kwargs...,
    )
end

"""
    thermoelectric(M::AbstractMatrix{<:ConductivityMoments}, mu_chem;
                   beta, volume, g_J=1.0, kernel=JacksonKernel,
                   NC=minimum(nc(m) for m in M), quad_N=8NC,
                   edge_cutoff=1e-3, sigma_min=nothing)

Compose a two- or three-dimensional thermoelectric tensor from typed
conductivity moments. `M[i, j]` holds moments for the current pair `(J_i, J_j)`
in the caller's own axis ordering; no direction is inferred from matrix indices,
from `H`, or from the moments. A caller exploiting symmetry must place the same
moments object in both off-diagonal slots explicitly, for example
`M[1, 2] = M[2, 1] = mxy`.

Exact Kubo–Greenwood moments give symmetric `L_r` tensors by trace cyclicity.
The skew part produced when components are estimated independently is treated
as stochastic noise (or an inconsistent-moments diagnostic) and projected out
before the open-circuit solve. The final `S = -L0 \\ (beta*L1)` is not itself
symmetrized: symmetric `L0` and `L1` need not commute, so a nonsymmetric `S` is
correct.

!!! warning

    This composition yields the **symmetric part** of the transport tensors
    only. The equal-energy Kubo–Greenwood contraction cannot produce
    antisymmetric (Hall-like) components, so for a time-reversal-broken model
    this is the symmetric projection, not the full conductivity or Seebeck
    tensor, and must not be presented as such.

This typed-only composition has no raw-`mu2D` tensor counterpart. The default
`sigma_min` is `SEEBECK_SIGMA_FLOOR_RTOL` times the largest absolute diagonal
band transport distribution. `neg_weight` records the maximum negative weight
over diagonal components only, because off-diagonal transport distributions
can legitimately be negative; the tensor wrapper does not emit the scalar
method's negative-weight warning. See
[`thermoelectric`](@ref) for units and physical scope.
"""
function thermoelectric(
    M::AbstractMatrix{<:ConductivityMoments},
    mu_chem::Real;
    beta::Real,
    volume::Real,
    g_J::Real = 1.0,
    kernel = JacksonKernel,
    NC::Int64 = minimum(nc(m) for m in M),
    quad_N::Int64 = 8 * NC,
    edge_cutoff::Real = 1e-3,
    sigma_min::Union{Nothing,Real} = nothing,
)
    size(M, 1) == size(M, 2) && size(M, 1) in (2, 3) || throw(
        ArgumentError(
            "thermoelectric: the moments matrix must be square with size 2 or 3 (got $(size(M)))",
        ),
    )

    reference = M[1, 1]
    for i in axes(M, 1), j in axes(M, 2)
        m = M[i, j]
        if m.a != reference.a
            throw(
                ArgumentError(
                    "thermoelectric: moments at index ($i, $j) have a=$(m.a) but expected a=$(reference.a); all entries must share identical (a, b, NH, NC)",
                ),
            )
        elseif m.b != reference.b
            throw(
                ArgumentError(
                    "thermoelectric: moments at index ($i, $j) have b=$(m.b) but expected b=$(reference.b); all entries must share identical (a, b, NH, NC)",
                ),
            )
        elseif m.NH != reference.NH
            throw(
                ArgumentError(
                    "thermoelectric: moments at index ($i, $j) have NH=$(m.NH) but expected NH=$(reference.NH); all entries must share identical (a, b, NH, NC)",
                ),
            )
        elseif nc(m) != nc(reference)
            throw(
                ArgumentError(
                    "thermoelectric: moments at index ($i, $j) have nc=$(nc(m)) but expected nc=$(nc(reference)); all entries must share identical (a, b, NH, NC)",
                ),
            )
        end
    end

    L0 = Matrix{Float64}(undef, size(M))
    L1 = Matrix{Float64}(undef, size(M))
    L2 = Matrix{Float64}(undef, size(M))
    neg_weight = 0.0
    for i in axes(M, 1), j in axes(M, 2)
        m = M[i, j]
        integrals = transport_integrals(
            m.mu,
            m.a,
            mu_chem;
            beta = beta,
            b = m.b,
            NH = m.NH,
            volume = volume,
            g_J = g_J,
            kernel = kernel,
            NC = NC,
            quad_N = quad_N,
            edge_cutoff = edge_cutoff,
        )
        L0[i, j] = integrals.L0
        L1[i, j] = integrals.L1
        L2[i, j] = integrals.L2
        if i == j
            neg_weight = max(neg_weight, integrals.neg_weight)
        end
    end

    L0_sym = (L0 + transpose(L0)) / 2
    L1_sym = (L1 + transpose(L1)) / 2
    L2_sym = (L2 + transpose(L2)) / 2
    skew = opnorm(L0 - transpose(L0)) / max(2 * opnorm(L0_sym), eps())
    if skew > 0.05
        @warn "Thermoelectric L0 tensor has a large skew fraction; stochastic noise or inconsistent component moments may be significant before symmetrization." skew=skew
    end

    if sigma_min === nothing
        max_sigma = 0.0
        for i in axes(M, 1)
            m = M[i, i]
            af = m.a
            bf = m.b
            w = af * (1 - Float64(edge_cutoff))
            n_scan = max(257, 4 * NC + 1)
            E_band = collect(range(bf - w, bf + w; length = n_scan))
            sigma_band = transport_distribution(
                m.mu,
                m.a,
                E_band;
                b = m.b,
                NH = m.NH,
                volume = volume,
                g_J = g_J,
                kernel = kernel,
                NC = NC,
                edge_cutoff = edge_cutoff,
            )
            max_sigma = max(max_sigma, maximum(abs, sigma_band))
        end
        sigma_min = SEEBECK_SIGMA_FLOOR_RTOL * max_sigma
    end
    S = seebeck_solve(L0_sym, L1_sym, Float64(beta); sigma_min = Float64(sigma_min))
    return ThermoelectricResult(
        L0_sym,
        L1_sym,
        L2_sym,
        S,
        Float64(mu_chem),
        Float64(beta),
        neg_weight,
    )
end

"""
    LocalBdGMoments(mu_rho, mu_delta, a, b, sites, NH, g_rho, U)

Local density and pairing moments for a reduced BdG Hamiltonian, together
with the rescaling and field-update metadata used to compute them.
"""
struct LocalBdGMoments{MR<:AbstractMatrix{<:Real},MC<:AbstractMatrix{<:Complex}} <:
       AbstractMoments
    mu_rho::MR
    mu_delta::MC
    a::Float64
    b::Float64
    sites::Vector{Int}
    NH::Int
    g_rho::Float64
    U::Vector{Float64}
    function LocalBdGMoments(
        mu_rho::MR,
        mu_delta::MC,
        a,
        b,
        sites,
        NH,
        g_rho,
        U,
    ) where {MR<:AbstractMatrix{<:Real},MC<:AbstractMatrix{<:Complex}}
        size(mu_rho) == size(mu_delta) || throw(
            ArgumentError(
                "LocalBdGMoments: mu_rho and mu_delta must have the same size (got $(size(mu_rho)) and $(size(mu_delta)))",
            ),
        )
        sites_vec = collect(Int, sites)
        U_vec = collect(Float64, U)
        ns = size(mu_rho, 2)
        ns == length(sites_vec) == length(U_vec) || throw(
            ArgumentError(
                "LocalBdGMoments: moment columns, sites, and U must have equal lengths (got $ns, $(length(sites_vec)), and $(length(U_vec)))",
            ),
        )
        b == 0.0 || throw(
            ArgumentError(
                "LocalBdGMoments: b must be 0.0 for reduced BdG radial rescaling (got $b)",
            ),
        )
        NH > 0 || throw(ArgumentError("LocalBdGMoments: NH must be positive (got $NH)"))
        af = Float64(a)
        isfinite(af) && af > 0 ||
            throw(ArgumentError("LocalBdGMoments: a must be finite and positive (got $a)"))
        new{MR,MC}(
            mu_rho,
            mu_delta,
            af,
            Float64(b),
            sites_vec,
            Int(NH),
            Float64(g_rho),
            U_vec,
        )
    end
end

nc(m::LocalBdGMoments) = size(m.mu_rho, 1)

Base.show(io::IO, m::LocalBdGMoments) = print(
    io,
    "LocalBdGMoments(NC=$(nc(m)), NS=$(length(m.sites)), NH=$(m.NH), a=$(m.a), b=$(m.b))",
)

"""
    bdg_local_moments(h; sites=nothing, NC=512, g_rho=2.0, batch_size=64)

Compute typed local density and pairing moments for a rescaled reduced BdG
Hamiltonian. By default, moments are computed at every physical site. The
reduced block integrates one spin species, so `g_rho=2` reconstructs the full
spin-singlet site density; `g_rho=1` stores per-spin density and leaves the
Hartree interpretation to the caller.
"""
function bdg_local_moments(
    h::RescaledHamiltonian{<:ScaledOperator{<:BdGOperator}};
    sites = nothing,
    NC::Integer = 512,
    g_rho::Real = 2.0,
    batch_size::Integer = 64,
)
    op = h.H.op
    sites_vec = sites === nothing ? collect(1:op.N) : collect(Int, sites)
    mu_rho, mu_delta =
        bdg_site_moments(h.H, op.N, sites_vec, Int(NC); batch_size = batch_size)
    return LocalBdGMoments(
        mu_rho,
        mu_delta,
        h.a,
        h.b,
        sites_vec,
        2op.N,
        g_rho,
        op.U[sites_vec],
    )
end

"""
    bdg_update(m::LocalBdGMoments; beta, kernel=JacksonKernel, Np=2nc(m))

Update local density and pairing fields using the rescaling, interaction, and
density-degeneracy metadata stored in `m`.
"""
function bdg_update(
    m::LocalBdGMoments;
    beta::Real,
    kernel = JacksonKernel,
    Np::Integer = 2 * nc(m),
)
    return bdg_update(
        m.mu_rho,
        m.mu_delta,
        m.a;
        U = m.U,
        beta = beta,
        g_rho = m.g_rho,
        kernel = kernel,
        Np = Np,
    )
end

"""
    ChannelBdGMoments(mu_rho, mu_F, a, b, sites, directed_bonds,
                      channels, NH, g_rho)

Density and directed anomalous moments for user-defined pairing channels,
together with their BdG rescaling and reconstruction metadata.
"""
struct ChannelBdGMoments{MR<:AbstractMatrix{<:Real},MC<:AbstractMatrix{<:Complex}} <:
       AbstractMoments
    mu_rho::MR
    mu_F::MC
    a::Float64
    b::Float64
    sites::Vector{Int}
    directed_bonds::Vector{Tuple{Int,Int}}
    channels::Vector{PairingChannel}
    NH::Int
    g_rho::Float64
    function ChannelBdGMoments(
        mu_rho::MR,
        mu_F::MC,
        a,
        b,
        sites,
        directed_bonds,
        channels,
        NH,
        g_rho,
    ) where {MR<:AbstractMatrix{<:Real},MC<:AbstractMatrix{<:Complex}}
        size(mu_rho, 1) == size(mu_F, 1) || throw(
            ArgumentError(
                "ChannelBdGMoments: mu_rho and mu_F must have the same number of rows (got $(size(mu_rho, 1)) and $(size(mu_F, 1)))",
            ),
        )
        sites_vec = collect(Int, sites)
        bonds_vec = collect(Tuple{Int,Int}, directed_bonds)
        channels_vec = collect(PairingChannel, channels)
        size(mu_rho, 2) == length(sites_vec) || throw(
            ArgumentError(
                "ChannelBdGMoments: mu_rho columns and sites must have equal lengths (got $(size(mu_rho, 2)) and $(length(sites_vec)))",
            ),
        )
        size(mu_F, 2) == length(bonds_vec) || throw(
            ArgumentError(
                "ChannelBdGMoments: mu_F columns and directed_bonds must have equal lengths (got $(size(mu_F, 2)) and $(length(bonds_vec)))",
            ),
        )
        b == 0.0 || throw(
            ArgumentError(
                "ChannelBdGMoments: b must be 0.0 for reduced BdG radial rescaling (got $b)",
            ),
        )
        NH > 0 || throw(ArgumentError("ChannelBdGMoments: NH must be positive (got $NH)"))
        iseven(NH) || throw(
            ArgumentError(
                "ChannelBdGMoments: NH must be even for a BdG Hamiltonian (got $NH)",
            ),
        )
        af = Float64(a)
        isfinite(af) && af > 0 || throw(
            ArgumentError("ChannelBdGMoments: a must be finite and positive (got $a)"),
        )
        N = Int(NH) ÷ 2
        all(i -> 1 <= i <= N, sites_vec) || throw(
            ArgumentError("ChannelBdGMoments: all sites must satisfy 1 <= site <= $N"),
        )
        site_set = Set(sites_vec)
        all(bond -> 1 <= bond[1] <= N && 1 <= bond[2] <= N, bonds_vec) || throw(
            ArgumentError(
                "ChannelBdGMoments: all directed bond indices must satisfy 1 <= index <= $N",
            ),
        )
        all(bond -> bond[1] in site_set, bonds_vec) || throw(
            ArgumentError(
                "ChannelBdGMoments: every directed bond source must occur in sites",
            ),
        )
        _validate_channel_ownership(channels_vec, "ChannelBdGMoments")
        directed_set = Set(bonds_vec)
        length(directed_set) == length(bonds_vec) || throw(
            ArgumentError("ChannelBdGMoments: directed_bonds must not contain duplicates"),
        )
        for channel in channels_vec, (i, j) in channel.bonds
            i <= N && j <= N || throw(
                ArgumentError(
                    "ChannelBdGMoments: channel bond ($i, $j) exceeds physical size $N",
                ),
            )
            (i, j) in directed_set ||
                throw(ArgumentError("ChannelBdGMoments: missing directed bond ($i, $j)"))
            i == j ||
                (j, i) in directed_set ||
                throw(ArgumentError("ChannelBdGMoments: missing directed bond ($j, $i)"))
        end
        new{MR,MC}(
            mu_rho,
            mu_F,
            af,
            Float64(b),
            sites_vec,
            bonds_vec,
            channels_vec,
            Int(NH),
            Float64(g_rho),
        )
    end
end

nc(m::ChannelBdGMoments) = size(m.mu_rho, 1)

Base.show(io::IO, m::ChannelBdGMoments) = print(
    io,
    "ChannelBdGMoments(NC=$(nc(m)), NS=$(length(m.sites)), NB=$(length(m.directed_bonds)), NH=$(m.NH), a=$(m.a), b=$(m.b))",
)

"""
    bdg_channel_moments(h, channels; sites=nothing, NC=512, g_rho=2.0,
                        batch_size=64)

Compute typed density and directed anomalous moments for all bonds required
by `channels`. By default every physical site is seeded.
"""
function bdg_channel_moments(
    h::RescaledHamiltonian{<:ScaledOperator{<:BdGOperator}},
    channels::Vector{PairingChannel};
    sites = nothing,
    NC::Integer = 512,
    g_rho::Real = 2.0,
    batch_size::Integer = 64,
)
    op = h.H.op
    sites_vec = sites === nothing ? collect(1:op.N) : collect(Int, sites)
    directed_bonds = _channel_directed_bonds(channels)
    mu_rho, mu_F = bdg_channel_moments(
        h.H,
        op.N,
        sites_vec,
        directed_bonds,
        Int(NC);
        batch_size = batch_size,
    )
    return ChannelBdGMoments(
        mu_rho,
        mu_F,
        h.a,
        h.b,
        sites_vec,
        directed_bonds,
        channels,
        2op.N,
        g_rho,
    )
end

"""
    bdg_update(m::ChannelBdGMoments; beta, kernel=JacksonKernel, Np=2nc(m))

Update density and all per-bond pairing amplitudes using the channel and
rescaling metadata stored in `m`.
"""
function bdg_update(
    m::ChannelBdGMoments;
    beta::Real,
    kernel = JacksonKernel,
    Np::Integer = 2 * nc(m),
)
    return bdg_channel_update(
        m.mu_rho,
        m.mu_F,
        m.a;
        channels = m.channels,
        directed_bonds = m.directed_bonds,
        beta = beta,
        g_rho = m.g_rho,
        kernel = kernel,
        Np = Np,
    )
end
