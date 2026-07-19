using FastGaussQuadrature

## Fermi projector and Bianco–Resta local Chern marker on the shared
## matrix-function action. The projector is the kernel-damped Chebyshev
## series of the occupation f_β(H − E_F); the marker contracts two projector
## actions per probe batch (see chern_marker in frontend.jl). Geometry is
## user data throughout: coordinate vectors, site lists, and areas are
## caller-supplied, never inferred from H.

"""
    fermi_coefficients(a, b, Ef; beta=Inf, NC, kernel=JacksonKernel, Np=2NC)

Chebyshev coefficients of the occupation ``f_\\beta(H - E_F)`` as a function
of the rescaled `H_norm = (H - b I)/a`, fully formed for verbatim
[`chebyshev_action!`](@ref) consumption: the ``(2 - \\delta_{n0})`` weights
and the damping `kernel` are already folded in. The Fermi-projector action
``P|v\\rangle`` *is* `chebyshev_action(H_norm, v, fermi_coefficients(...))`
— there is no separate raw projector routine (same layering as `evolve`).

At `beta = Inf` the coefficients are the closed-form step series
(``c_0 = 1 - \\theta_F/\\pi``, ``c_n = -(2/\\pi)\\sin(n\\theta_F)/n`` with
``\\theta_F = \\arccos \\tilde x_F``, ``\\tilde x_F = (E_F - b)/a``); at
finite `beta` they are evaluated by `Np`-point Gauss–Chebyshev quadrature of
the smooth Fermi factor. At the jump itself the sharp series converges to
the midpoint value `1/2` — the same convention as the package step
[`fermiFunctions`](@ref) — so a state with an eigenvalue exactly at `Ef`
receives half occupation. `Ef` must map strictly inside the rescaled
window, `-1 < x̃_F < 1`: an `Ef` outside the spectrum would make every
coefficient route a near-constant occupation (`P ≈ 0` or `≈ I`), which is
far more likely a rescaling bookkeeping error than intent, so it is
rejected rather than silently returning a trivial operator.

The step is non-analytic, so the series has no rigorous truncation bound and
`NC` is a **required** keyword (deliberately unlike the `NC = 1024` moment
defaults): the Jackson-damped resolution is ``\\Delta E \\approx \\pi a/NC``,
which must sit well inside the spectral gap at `Ef` — choose
``NC \\gtrsim 4\\pi a/\\Delta_{\\mathrm{gap}}`` and verify by doubling `NC`.
`JacksonKernel` is the default damping (undamped truncation Gibbs-rings the
projector); pass any `kernel(n, N)`, e.g. `(n, N) -> 1.0` for the bare
series. At finite `beta` the effective smearing is
``\\max(\\Delta E, \\sim 4/\\beta)``; explicit thermal smoothing is the
recommended regularizer when `Ef` approaches a band edge or a transition.
"""
function fermi_coefficients(
    a::Real,
    b::Real,
    Ef::Real;
    beta::Real = Inf,
    NC::Integer,
    kernel = JacksonKernel,
    Np::Integer = 2 * NC,
)
    isfinite(a) && a > 0 || throw(ArgumentError("a must be finite and positive, got $a"))
    isfinite(b) || throw(ArgumentError("b must be finite, got $b"))
    isfinite(Ef) || throw(ArgumentError("Ef must be finite, got $Ef"))
    beta > 0 || throw(ArgumentError("beta must be positive, got $beta"))
    NC >= 1 || throw(ArgumentError("NC must be >= 1, got $NC"))
    Np >= 1 || throw(ArgumentError("Np must be >= 1, got $Np"))
    xF = (Ef - b) / a
    -1 < xF < 1 || throw(
        ArgumentError(
            "Ef = $Ef maps to x̃_F = $xF outside the rescaled spectral window (-1, 1)",
        ),
    )

    c = Vector{Float64}(undef, NC)
    if isinf(beta)
        θF = acos(xF)
        c[1] = 1 - θF / π
        for n = 1:(NC-1)
            c[n+1] = -(2 / π) * sin(n * θF) / n
        end
    else
        # c_n = hn(n)/Np · Σ_k T_n(x_k) f_β(a x_k + b − Ef), Gauss–Chebyshev
        # nodes x_k (the bdg_update pattern, in physical units).
        nodes, _ = gausschebyshevt(Np)
        wf = fermiFunctions(Float64(Ef), Float64(beta)).(a .* nodes .+ b) ./ Np
        c .= hn.(0:(NC-1)) .* (cos.((0:(NC-1)) .* acos.(nodes')) * wf)
    end
    c .*= kernel.(0:(NC-1), NC)
    return c
end

# Per-column contraction of the marker sequence for one probe batch:
# with u = P v, Im⟨u|X Q Y|u⟩ = −Im⟨u|X P Y|u⟩ identically (Σ_j x_j y_j
# |u_j|² is exactly real for real coordinates), so Q never materializes and
# two projector actions suffice. Returns the host vector
# r[c] = Σ_j x[j] · Im(conj(U[j,c]) · W[j,c]),   U = P·V, W = P·(y .* U),
# so that the Bianco–Resta marker of a unit-site probe column is
# m = −4π·Im⟨e|P X Q Y P|e⟩ = +4π·r. Workspaces U, YU, W and the recurrence
# slots are caller-supplied and resident with Hn; V may be host or resident.
function _pxqyp_imdiag!(
    Hn,
    C::AbstractVector,
    xd,
    yd,
    V,
    U,
    YU,
    W,
    slots;
    check_every::Integer = 16,
    verbose::Integer = 0,
)
    chebyshev_action!(
        U,
        Hn,
        V,
        C;
        slots = slots,
        check_every = check_every,
        verbose = verbose,
    )
    YU .= yd .* U
    chebyshev_action!(
        W,
        Hn,
        YU,
        C;
        slots = slots,
        check_every = check_every,
        verbose = verbose,
    )
    return maybe_to_host(vec(sum(xd .* imag.(conj.(U) .* W); dims = 1)))
end

"""
    chern_marker_average(markers; area) -> Float64

Sum of raw orbital markers divided by the explicit `area` they cover. When
`markers` holds the [`chern_marker`](@ref) values of **complete bulk cells**
(all orbitals of each cell, away from the boundary) and `area` is the total
area of those cells, this is the local-marker estimate of the Chern number.
The area is caller data — cell and sample geometry are never inferred (the
raw markers carry units of x·y, so the normalization decides
dimensionlessness). Note the sharp (`beta = Inf`) marker summed over an
*entire* finite open sample is ≈ 0: topology comes from a bulk average with
the boundary excluded, not from the full trace.

The scalar method serves an already-summed value — in particular
`chern_marker_average(mean(est); area)` for the per-probe region-sum
estimates of [`chern_marker_region`](@ref). Do not pass that estimate
vector to the vector method: its entries each estimate the whole region
sum, so summing them overcounts by `NR`.
"""
function chern_marker_average(markers::AbstractVector{<:Real}; area::Real)
    isfinite(area) && area > 0 ||
        throw(ArgumentError("area must be finite and positive, got $area"))
    return sum(markers) / area
end

chern_marker_average(marker::Real; area::Real) = chern_marker_average([marker]; area = area)
