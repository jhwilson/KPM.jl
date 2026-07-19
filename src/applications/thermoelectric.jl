# k_B/|e| in microvolt per kelvin
const KB_OVER_E_UV_PER_K = 86.17333262

# Default relative conductivity floor: a thermal-window L0 whose (symmetric-part)
# scale is more than this factor below the band maximum of |Sigma(E)| is treated
# as numerically insulating.
const SEEBECK_SIGMA_FLOOR_RTOL = 1e-6

# Bare diagonal Chebyshev contraction at rescaled nodes. The node chunks keep
# each NC x K reconstruction workspace near 16 MB.
function _transport_nodes(mu_tilde::AbstractMatrix, x::AbstractVector{Float64}, NC::Int)
    r = zeros(Float64, length(x))
    ns = 0:(NC-1)
    chunk = max(1, min(length(x), cld(1_000_000, NC)))
    for lo = 1:chunk:length(x)
        hi = min(lo + chunk - 1, length(x))
        xs = transpose(view(x, lo:hi))
        theta = acos.(xs)
        C = cos.(ns .* theta)
        U = mu_tilde * C
        r[lo:hi] .= vec(real.(sum(C .* U; dims = 1)))
    end
    return r
end

function _transport_from_prepared(
    mu_tilde::AbstractMatrix,
    x::Vector{Float64},
    a::Float64,
    NH::Integer,
    volume::Float64,
    g_J::Float64,
    NC::Int,
    edge_cutoff::Float64,
)
    sigma = zeros(Float64, length(x))
    inside = findall(abs.(x) .< 1 - edge_cutoff)
    isempty(inside) && return sigma
    xi = x[inside]
    contractions = _transport_nodes(mu_tilde, xi, NC)
    sigma[inside] .= @. -(2 * g_J * NH) * contractions / (volume * a^2 * (1 - xi^2))
    return sigma
end

"""
    transport_distribution(mu2D, a, E; b=0.0, NH, volume, g_J=1.0,
                           kernel=JacksonKernel, NC=size(mu2D, 1),
                           edge_cutoff=1e-3)

Reconstruct the symmetric Kubo–Greenwood transport distribution
`Sigma_alphabeta(E)` at physical energies `E`. Values outside the usable
rescaled band, `abs((E-b)/a) < 1-edge_cutoff`, are zero.

`mu2D` are the moments from
`kpm_2d(H_norm, Jalpha, Jbeta, NC, NR, NH)`, with
`H_norm = (H - b*I)/a`. The current operators must be built from the
**original, unrescaled** Hamiltonian using
`(J_alpha)_ij = H_ij (r_i-r_j)_alpha = i*hbar*(v_alpha)_ij`. With unit-norm
probes, the moment convention is
`mu2D[n,m] ≈ Tr[Jalpha T_m(H_norm) Jbeta T_n(H_norm)]/NH` for the zero-based
Chebyshev indices held at Julia indices `n+1,m+1`.

The returned distribution is in `(e^2/h) * length^(2-d)`. Thus, in three
dimensions, with `volume` in `length^3`, it is in `(e^2/h)/length`. The
equal-energy contraction reconstructs only the symmetric, dissipative part of
`sigma_alphabeta`; it does not contain Hall or other antisymmetric components.
GPU moment matrices are transferred to the host for reconstruction.
"""
function transport_distribution(
    mu2D,
    a::Real,
    E::AbstractVector{<:Real};
    b::Real = 0.0,
    NH::Integer,
    volume::Real,
    g_J::Real = 1.0,
    kernel = JacksonKernel,
    NC::Int64 = size(mu2D, 1),
    edge_cutoff::Real = 1e-3,
)
    isfinite(a) && a > 0 ||
        throw(ArgumentError("transport_distribution: a must be finite and positive."))
    isfinite(b) || throw(ArgumentError("transport_distribution: b must be finite."))
    isfinite(volume) && volume > 0 ||
        throw(ArgumentError("transport_distribution: volume must be finite and positive."))
    NH > 0 || throw(ArgumentError("transport_distribution: NH must be positive."))
    isfinite(g_J) && g_J > 0 ||
        throw(ArgumentError("transport_distribution: g_J must be finite and positive."))
    all(isfinite, E) ||
        throw(ArgumentError("transport_distribution: all energies E must be finite."))
    0 <= edge_cutoff < 1 ||
        throw(ArgumentError("transport_distribution: edge_cutoff must be in [0, 1)."))

    NC = min(NC, size(mu2D, 1), size(mu2D, 2))
    NC > 0 || throw(ArgumentError("transport_distribution: NC must be positive."))
    mu_tilde = maybe_to_host(mu2D_apply_kernel_and_h(mu2D[1:NC, 1:NC], NC, kernel))
    af = Float64(a)
    bf = Float64(b)
    x = Float64.((E .- bf) ./ af)
    return _transport_from_prepared(
        mu_tilde,
        x,
        af,
        NH,
        Float64(volume),
        Float64(g_J),
        NC,
        Float64(edge_cutoff),
    )
end

transport_distribution(mu2D, a::Real, E::Real; kwargs...) =
    only(transport_distribution(mu2D, a, [E]; kwargs...))

"""
    transport_integrals(mu2D, a, mu_chem; beta, b=0.0, NH, volume,
                        g_J=1.0, kernel=JacksonKernel, NC=size(mu2D, 1),
                        quad_N=8*NC, edge_cutoff=1e-3)

Compute the Chester–Thellung/Kubo–Greenwood moments

    L_r = integral dE (E-mu_chem)^r (-df/dE) Sigma(E),  r = 0,1,2,

using Gauss–Legendre quadrature over the intersection of the usable band and
`mu_chem +/- 40/beta`. At 40 thermal energies the omitted Fermi-window weight
is approximately `4e-18`. `beta` must be finite and positive; the CTKG thermal
window does not support the zero-temperature delta-function limit.

The thermal-resolution warning compares `1/beta` with the Jackson-kernel
estimate `pi*a/NC`. For `LorentzKernels(lambda)`, the broadening instead scales
with `lambda` and can be larger; no kernel introspection is attempted.

Returns `(L0, L1, L2, neg_weight)`, where `neg_weight` is the fraction of the
absolute thermally weighted transport distribution contributed by negative
values. Negative values are retained in all three integrals.
"""
function transport_integrals(
    mu2D,
    a::Real,
    mu_chem::Real;
    beta::Real,
    b::Real = 0.0,
    NH::Integer,
    volume::Real,
    g_J::Real = 1.0,
    kernel = JacksonKernel,
    NC::Int64 = size(mu2D, 1),
    quad_N::Int64 = 8*NC,
    edge_cutoff::Real = 1e-3,
)
    isfinite(a) && a > 0 ||
        throw(ArgumentError("transport_integrals: a must be finite and positive."))
    isfinite(b) || throw(ArgumentError("transport_integrals: b must be finite."))
    isfinite(volume) && volume > 0 ||
        throw(ArgumentError("transport_integrals: volume must be finite and positive."))
    NH > 0 || throw(ArgumentError("transport_integrals: NH must be positive."))
    isfinite(g_J) && g_J > 0 ||
        throw(ArgumentError("transport_integrals: g_J must be finite and positive."))
    quad_N > 0 || throw(ArgumentError("transport_integrals: quad_N must be positive."))
    0 <= edge_cutoff < 1 ||
        throw(ArgumentError("transport_integrals: edge_cutoff must be in [0, 1)."))
    isfinite(beta) && beta > 0 || throw(
        ArgumentError(
            "transport_integrals: beta must be finite and positive; T=0 is not supported by the CTKG window.",
        ),
    )
    isfinite(mu_chem) ||
        throw(ArgumentError("transport_integrals: mu_chem must be finite."))

    NC = min(NC, size(mu2D, 1), size(mu2D, 2))
    NC > 0 || throw(ArgumentError("transport_integrals: NC must be positive."))
    af = Float64(a)
    bf = Float64(b)
    beta_f = Float64(beta)
    mu_f = Float64(mu_chem)
    edge_f = Float64(edge_cutoff)
    if inv(beta_f) < pi * af / NC
        @warn "Thermal width 1/beta is narrower than the Jackson-kernel KPM energy-resolution estimate pi*a/NC; thermoelectric results may not be converged. Lorentz-kernel broadening scales with lambda and can be larger." beta=beta_f jackson_resolution=pi *
                                                                                                                                                                                                                                                             af /
                                                                                                                                                                                                                                                             NC
    end

    band_halfwidth = af * (1 - edge_f)
    band_lower = bf - band_halfwidth
    band_upper = bf + band_halfwidth
    tail_lower = mu_f - 40 / beta_f
    tail_upper = mu_f + 40 / beta_f
    lower = max(band_lower, tail_lower)
    upper = min(band_upper, tail_upper)
    lower < upper || return (L0 = 0.0, L1 = 0.0, L2 = 0.0, neg_weight = 0.0)

    if lower == band_lower || upper == band_upper
        f_lower = fermiFunction(lower, mu_f, beta_f)
        f_upper = fermiFunction(upper, mu_f, beta_f)
        lost_mass = 1 - (f_lower - f_upper)
        if lost_mass > 1e-6
            @warn "The thermal window extends past the usable band; clipped Fermi-window mass can bias L1 and the Seebeck coefficient." lost_fermi_window_mass=lost_mass
        end
    end

    nodes, weights = gausslegendre(quad_N)
    halfwidth = (upper - lower) / 2
    midpoint = (upper + lower) / 2
    energies = Float64.(midpoint .+ halfwidth .* nodes)
    quad_weights = halfwidth .* weights

    mu_tilde = maybe_to_host(mu2D_apply_kernel_and_h(mu2D[1:NC, 1:NC], NC, kernel))
    x = (energies .- bf) ./ af
    sigma = _transport_from_prepared(
        mu_tilde,
        x,
        af,
        NH,
        Float64(volume),
        Float64(g_J),
        NC,
        edge_f,
    )
    fw = fermi_window(mu_f, beta_f)
    thermal_weights = quad_weights .* fw.(energies)
    delta = energies .- mu_f

    L0 = sum(thermal_weights .* sigma)
    L1 = sum(thermal_weights .* delta .* sigma)
    L2 = sum(thermal_weights .* delta .^ 2 .* sigma)
    neg = sum(thermal_weights .* max.(-sigma, 0.0))
    tot = sum(thermal_weights .* abs.(sigma))
    neg_weight = tot > 0 ? neg / tot : 0.0
    return (
        L0 = Float64(L0),
        L1 = Float64(L1),
        L2 = Float64(L2),
        neg_weight = Float64(neg_weight),
    )
end

"""
    ThermoelectricResult

Eager thermoelectric reconstruction result. `L0`, `L1`, and `L2` have units
`(e^2/h) * length^(2-d) * energy^r` for `r = 0,1,2`, respectively.
`S_over_kB_over_e` is dimensionless and gives the Seebeck coefficient in units
of `k_B/|e|`; `mu_chem` and `beta` record the physical chemical potential and
inverse thermal energy, and `neg_weight` diagnoses negative spectral weight.
The matrix variant contains only the symmetric part of the transport tensors.
"""
struct ThermoelectricResult{T<:Union{Float64,Matrix{Float64}}}
    L0::T
    L1::T
    L2::T
    S_over_kB_over_e::T
    mu_chem::Float64
    beta::Float64
    neg_weight::Float64
end

"""
    seebeck_uVK(r::ThermoelectricResult)

Convert the Seebeck result to microvolt per kelvin, using
`k_B/|e| = 86.17333262 microvolt/K`.
"""
seebeck_uVK(r::ThermoelectricResult) = KB_OVER_E_UV_PER_K * r.S_over_kB_over_e

function Base.show(io::IO, r::ThermoelectricResult)
    print(
        io,
        "ThermoelectricResult(S=",
        r.S_over_kB_over_e,
        " k_B/|e|, mu_chem=",
        r.mu_chem,
        ", beta=",
        r.beta,
        ", neg_weight=",
        r.neg_weight,
        ")",
    )
end

function Base.show(io::IO, ::MIME"text/plain", r::ThermoelectricResult)
    println(io, "ThermoelectricResult")
    println(io, "  S (k_B/|e|): ", r.S_over_kB_over_e)
    println(io, "  S (microvolt/K): ", seebeck_uVK(r))
    println(io, "  L0: ", r.L0)
    println(io, "  L1: ", r.L1)
    println(io, "  L2: ", r.L2)
    println(io, "  mu_chem: ", r.mu_chem)
    println(io, "  beta: ", r.beta)
    print(io, "  neg_weight: ", r.neg_weight)
end

"""
    seebeck_solve(L0, L1, beta; sigma_min=0.0)

Solve for the dimensionless electron-convention Seebeck coefficient,
`S = -L0 \\ (beta*L1)`. `sigma_min` is an absolute floor in the units of `L0`;
its default of `0.0` applies only the sign, finiteness, and definiteness checks.
`sigma_min` must be finite and nonnegative. For scalars, non-finite `L0` or
`L1`, `L0 <= 0`, or `L0 < sigma_min` produces a warning and `NaN`. For
matrices, non-finite entries, a symmetric-part eigenvalue `<= 0` or
`< sigma_min`, a singular matrix, or condition number above `1e12` produces a
warning and an all-`NaN` matrix. The matrix method uses a left solve and never
forms an explicit inverse.
"""
function seebeck_solve(L0::Float64, L1::Float64, beta::Float64; sigma_min::Real = 0.0)
    isfinite(sigma_min) && sigma_min >= 0 ||
        throw(ArgumentError("seebeck_solve: sigma_min must be finite and nonnegative."))
    if !isfinite(L1)
        @warn "Cannot compute Seebeck coefficient because L1 is non-finite; returning NaN." L1=L1
        return NaN
    end
    if !isfinite(L0) || L0 <= 0 || L0 < sigma_min
        @warn "Cannot compute Seebeck coefficient for insulating/non-positive or below-conductivity-floor L0; returning NaN. Pass sigma_min explicitly, e.g. sigma_min=0.0 to disable the floor." L0=L0 sigma_min=sigma_min
        return NaN
    end
    return -beta * L1 / L0
end

function seebeck_solve(
    L0::Matrix{Float64},
    L1::Matrix{Float64},
    beta::Float64;
    sigma_min::Real = 0.0,
)
    isfinite(sigma_min) && sigma_min >= 0 ||
        throw(ArgumentError("seebeck_solve: sigma_min must be finite and nonnegative."))
    size(L0, 1) == size(L0, 2) ||
        throw(DimensionMismatch("seebeck_solve: L0 must be square."))
    size(L1) == size(L0) ||
        throw(DimensionMismatch("seebeck_solve: L0 and L1 must have matching sizes."))
    if !all(isfinite, L0)
        @warn "Cannot compute Seebeck tensor because L0 is non-finite; returning NaNs."
        return fill(NaN, size(L0))
    end
    if !all(isfinite, L1)
        @warn "Cannot compute Seebeck tensor because L1 is non-finite; returning NaNs."
        return fill(NaN, size(L0))
    end
    lam_min = eigmin(Symmetric((L0 + transpose(L0)) / 2))
    if lam_min <= 0 || lam_min < sigma_min
        @warn "Cannot compute Seebeck tensor for an insulating, below-conductivity-floor, or non-positive-definite symmetric part of L0; returning NaNs. Pass sigma_min explicitly, e.g. sigma_min=0.0 to disable the floor." lam_min=lam_min sigma_min=sigma_min
        return fill(NaN, size(L0))
    end
    F = lu(L0; check = false)
    if !issuccess(F)
        @warn "Cannot compute Seebeck tensor because L0 is singular; returning NaNs."
        return fill(NaN, size(L0))
    end
    kappa = cond(L0)
    if !isfinite(kappa) || kappa > 1e12
        @warn "Cannot compute Seebeck tensor because L0 is singular or hopelessly ill-conditioned; returning NaNs." condition_number=kappa
        return fill(NaN, size(L0))
    end
    return -(F \ (beta * L1))
end

"""
    thermoelectric(mu2D, a, mu_chem; beta, b=0.0, NH, volume, g_J=1.0,
                   kernel=JacksonKernel, NC=size(mu2D, 1), quad_N=8*NC,
                   edge_cutoff=1e-3, sigma_min=nothing)

Compute the longitudinal electronic Seebeck coefficient from the
Chester–Thellung/Kubo–Greenwood (CTKG), or Jonson–Mahan, relations

    L_r = integral dE (E-mu_chem)^r (-df/dE) Sigma(E),
    S/(k_B/|e|) = -beta*L1/L0.

`mu2D` must come from `kpm_2d(H_norm, Jalpha, Jbeta, NC, NR, NH)`, where
`H_norm = (H-b*I)/a` and both current operators are built from the original,
unrescaled Hamiltonian using `(J_alpha)_ij = H_ij(r_i-r_j)_alpha`.

The electron convention `q=-|e|` fixes the minus sign; carrier character
emerges from the particle-hole asymmetry of `Sigma(E)`. `L_r` has units
`(e^2/h) * length^(2-d) * energy^r`, and the stored Seebeck value is
dimensionless in `k_B/|e|`. The equal-energy Kubo–Greenwood contraction yields
only the symmetric part of the transport response, not Hall or other
antisymmetric components.

`beta` must be finite and positive. The `beta=Inf` limit is rejected because
the thermal window becomes a delta function and returns the trivial `L1=0`,
hiding the physically relevant low-temperature Mott slope. See also
[`transport_distribution`](@ref), [`transport_integrals`](@ref), and
[`kubo_bastin_cond`](@ref).

By default, `sigma_min` is
`SEEBECK_SIGMA_FLOOR_RTOL * max_E |Sigma(E)|` over the usable band. An
insulating thermal window therefore yields `S = NaN` with a warning while
`L0`, `L1`, and `L2` are still reported. Pass `sigma_min` explicitly (for
example, `sigma_min=0.0`) to override this floor.
"""
function thermoelectric(
    mu2D,
    a::Real,
    mu_chem::Real;
    beta::Real,
    b::Real = 0.0,
    NH::Integer,
    volume::Real,
    g_J::Real = 1.0,
    kernel = JacksonKernel,
    NC::Int64 = size(mu2D, 1),
    quad_N::Int64 = 8*NC,
    edge_cutoff::Real = 1e-3,
    sigma_min::Union{Nothing,Real} = nothing,
)
    isfinite(mu_chem) || throw(ArgumentError("thermoelectric: mu_chem must be finite."))
    isfinite(beta) && beta > 0 || throw(
        ArgumentError(
            "thermoelectric: beta must be finite and positive; T=0 is not supported by the CTKG window.",
        ),
    )
    integrals = transport_integrals(
        mu2D,
        a,
        mu_chem;
        beta = beta,
        b = b,
        NH = NH,
        volume = volume,
        g_J = g_J,
        kernel = kernel,
        NC = NC,
        quad_N = quad_N,
        edge_cutoff = edge_cutoff,
    )
    if integrals.neg_weight > 1e-2
        @warn "Transport distribution has significant negative weight in the thermal window; increase NR/NC or broadening — result may not be converged." neg_weight=integrals.neg_weight
    end
    beta_f = Float64(beta)
    if sigma_min === nothing
        af = Float64(a);
        bf = Float64(b);
        w = af * (1 - Float64(edge_cutoff))
        n_scan = max(257, 4 * NC + 1)
        E_band = collect(range(bf - w, bf + w; length = n_scan))
        sigma_band = transport_distribution(
            mu2D,
            a,
            E_band;
            b = b,
            NH = NH,
            volume = volume,
            g_J = g_J,
            kernel = kernel,
            NC = NC,
            edge_cutoff = edge_cutoff,
        )
        sigma_min = SEEBECK_SIGMA_FLOOR_RTOL * maximum(abs, sigma_band)
    end
    S = seebeck_solve(integrals.L0, integrals.L1, beta_f; sigma_min = Float64(sigma_min))
    return ThermoelectricResult(
        integrals.L0,
        integrals.L1,
        integrals.L2,
        S,
        Float64(mu_chem),
        beta_f,
        integrals.neg_weight,
    )
end
