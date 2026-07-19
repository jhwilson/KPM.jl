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
the smooth Fermi factor. `Ef` must map strictly inside the rescaled window,
`-1 < x̃_F < 1`.

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
