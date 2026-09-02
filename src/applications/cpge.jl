# The paper coefficients (u, v, w) multiply the package tensor slots listed
# here.  JL/Wu's Gamma[n,m,p] = mu[m,n,p], so u -> 2, v -> 1, and w -> 3.
# Keeping this permutation in one constant makes the labeling choice explicit.
const _CPGE_SLOTS = (2, 1, 3)

function _check_cpge_inputs(mu3, NC, omega1, omega2, lambda, delta)
    NC > 0 || throw(ArgumentError("NC must be positive"))
    ndims(mu3) == 3 || throw(ArgumentError("mu3 must be a three-dimensional moment table"))
    all(size(mu3, d) >= NC for d = 1:3) ||
        throw(ArgumentError("NC exceeds the supplied moment dimensions"))
    lambda >= 0 || throw(ArgumentError("lambda must be nonnegative (got $lambda)"))
    delta >= 0 || throw(ArgumentError("delta must be nonnegative (got $delta)"))
    Omega = omega1 + omega2
    # RR shift differences are omega1, RA shift differences are Omega, and AA
    # shift differences are omega2. Equal edges coincide at zero difference;
    # opposite Chebyshev edges coincide when the difference is +/-2.
    coincidences = (omega1, omega2, Omega)
    if iszero(lambda) && iszero(delta) && any(x -> x in (-2, 0, 2), coincidences)
        throw(
            ArgumentError(
                "coincident shifted Green-function edges require lambda > 0 or " *
                "delta > 0 when omega1, omega2, or Omega = omega1 + omega2 is " *
                "0 or +/-2; see the cpge docstring for the regularization recipe.",
            ),
        )
    end
    return nothing
end

function _cpge_green_coefficients!(g, z, lambda, delta, branch)
    if delta > 0 && 1 - delta < abs(z) < 1 + delta
        fill!(g, 0)
        return g
    end
    return green_coefficients!(g, z, lambda, branch)
end

function _cpge_exclusion_breakpoints(shifts, delta)
    iszero(delta) && return Float64[]
    points = Float64[]
    for shift in shifts, radius in (1 - delta, 1 + delta)
        radius < 0 && continue
        for edge in (-radius, radius)
            xstar = edge - shift
            -1 < xstar < 1 && push!(points, acos(xstar))
        end
    end
    return _deduplicate_sorted(points)
end

function _cpge_slot_coefficients(u, v, w)
    coefficients = (u, v, w)
    return ntuple(
        slot -> coefficients[findfirst(==(slot), _CPGE_SLOTS)],
        Val(3),
    )
end

function _cpge_contract!(collapsed, right, mu_matrix, u, v, w)
    slot1, slot2, slot3 = _cpge_slot_coefficients(u, v, w)
    mul!(collapsed, transpose(mu_matrix), slot1)
    mul!(right, reshape(collapsed, length(slot2), length(slot3)), slot3)
    return sum(slot2 .* right) # plain transpose: Chebyshev contractions are not sesquilinear
end

function _cpge_node_function(mu_tilde, NC, omega1, omega2, lambda, delta)
    Delta = zeros(ComplexF64, NC)
    gR_Omega = similar(Delta)
    gR_omega2 = similar(Delta)
    gR_omega1 = similar(Delta)
    gA_momega2 = similar(Delta)
    gA_momega1 = similar(Delta)
    gA_mOmega = similar(Delta)
    collapsed = zeros(ComplexF64, NC^2)
    right = zeros(ComplexF64, NC)
    mu_matrix = reshape(mu_tilde, NC, NC^2)
    Omega = omega1 + omega2

    function F!(out, theta)
        chebyshev_delta_theta!(Delta, theta)
        x = cos(theta)
        _cpge_green_coefficients!(gR_Omega, x + Omega, lambda, delta, Val(:R))
        _cpge_green_coefficients!(gR_omega2, x + omega2, lambda, delta, Val(:R))
        _cpge_green_coefficients!(gR_omega1, x + omega1, lambda, delta, Val(:R))
        _cpge_green_coefficients!(gA_momega2, x - omega2, lambda, delta, Val(:A))
        _cpge_green_coefficients!(gA_momega1, x - omega1, lambda, delta, Val(:A))
        _cpge_green_coefficients!(gA_mOmega, x - Omega, lambda, delta, Val(:A))

        rr = _cpge_contract!(
            collapsed,
            right,
            mu_matrix,
            gR_Omega,
            gR_omega2,
            Delta,
        )
        ra = _cpge_contract!(
            collapsed,
            right,
            mu_matrix,
            gR_omega1,
            Delta,
            gA_momega2,
        )
        aa = _cpge_contract!(
            collapsed,
            right,
            mu_matrix,
            Delta,
            gA_momega1,
            gA_mOmega,
        )
        out[1] = rr + ra + aa
        return out
    end
    return _SpectralNodeFunction(F!, 1, [all(iszero, mu_tilde)])
end

function _cpge_kernel_moments(mu3, NC, kernel)
    view3 = view(mu3, 1:NC, 1:NC, 1:NC)
    return maybe_to_host(mu3D_apply_kernel_and_h(view3, Int(NC), kernel))
end

"""
    cpge(mu3, NC, omega1_tilde, omega2_tilde; E_f=0, beta=Inf,
         lambda=0, delta=0, kernel=JacksonKernel, quad_rtol=1e-8,
         quad_atol=0, maxevals=10^6)

Return the bare three-current second-order response in rescaled units,

```math
i\\int_0^\\pi d\\theta\\,\\tilde f(\\cos\\theta)\\,[RR(\\theta)+RA(\\theta)+AA(\\theta)].
```

There is no `Omega` prefactor and no `1/(omega1_tilde*omega2_tilde)` factor;
`omega2_tilde = -omega1_tilde` is legal with the regularization described
below. `E_f`, both frequencies, and `lambda` are rescaled energies; `beta` is
inverse rescaled energy (`beta = beta_physical*a`). In units `e = hbar = 1`,
the physical response of Wu et al., PRB 110, 014201, Eqs. 6/A1
(three-current term only), is
`chi_alpha_beta_gamma = D/(V*a^2) * cpge(...)`, in their `e^3/hbar^2` units.

The package table is
`mu3[n,m,p] = Tr[Jalpha*T_m*Jbeta*T_n*Jgamma*T_p]/D`; the operator order in
the trace is the argument order of `kpm_3d`. The paper tensor is
`Gamma[n,m,p] = mu3[m,n,p]`: coefficient vectors `(u,v,w)` therefore contract
tensor slots `(2,1,3)`, respectively. Unlike the two-index optical route,
which uses the Kubo--Bastin orientation, the three-index contraction follows
Joao--Lopes/Wu et al. literally.

For the injection-current convention of Wu et al., form
`y(omega1,omega2) = Im((chi_alpha_beta_gamma(omega1,omega2) +
chi_beta_alpha_gamma(omega2,omega1))/(omega1*omega2))`, then
`beta(omega) = lim_(Omega->0) Omega*y(omega,Omega-omega)`. Their numerical
protocol sets `Omega = 0` exactly, uses Jackson broadening, and identifies an
`NC`-linear plateau calibrated against `beta0*C = pi*e^3*C/(3h^2)`. The old
calibration constant was fitted with the previous reconstruction and must be
re-fitted for this implementation.

When any of `omega1_tilde`, `omega2_tilde`, or
`Omega = omega1_tilde + omega2_tilde` equals `0`, `+2`, or `-2`, at least one
of `lambda > 0` or `delta > 0` is required. These are exactly the coincident
same- or opposite-band-edge Green singularities; either regularizer resolves
every such coincidence. `lambda` is the rescaled Green broadening.
`delta` instead zeroes every *shifted* Green coefficient on the annulus
`1-delta < |z| < 1+delta`; it is never applied to the delta coefficient.
Unregularized calls at any of the listed coincidences throw.

Adaptive quadrature reuses the shared shifted-edge breakpoints and endpoint
maps and requires `error <= quad_atol + quad_rtol*abs(integral)`. It throws
if `maxevals` is insufficient; a nonzero `quad_atol` is needed for components
that vanish by cancellation. Exact-zero components are returned as zero.
Exclusion-annulus boundaries are additional finite-jump breakpoints. Cost is
`O(N_nodes*NC^3)` with `O(NC^2)` node workspace.
"""
function cpge(
    mu3,
    NC::Integer,
    omega1_tilde::Real,
    omega2_tilde::Real;
    E_f::Real = 0.0,
    beta::Real = Inf,
    lambda::Real = 0.0,
    delta::Real = 0.0,
    kernel = JacksonKernel,
    quad_rtol::Real = 1e-8,
    quad_atol::Real = 0.0,
    maxevals::Integer = 10^6,
)
    _check_cpge_inputs(mu3, NC, omega1_tilde, omega2_tilde, lambda, delta)
    mu_tilde = _cpge_kernel_moments(mu3, NC, kernel)
    F! = _cpge_node_function(
        mu_tilde,
        Int(NC),
        omega1_tilde,
        omega2_tilde,
        lambda,
        delta,
    )
    Omega = omega1_tilde + omega2_tilde
    shifts = (Omega, omega2_tilde, omega1_tilde, -omega2_tilde, -omega1_tilde, -Omega)
    integral, _ = _spectral_integral(
        F!,
        NC,
        shifts,
        E_f,
        beta,
        lambda;
        rtol = quad_rtol,
        atol = quad_atol,
        maxevals = maxevals,
        breakpoints = _cpge_exclusion_breakpoints(shifts, delta),
    )
    return ComplexF64(im * only(integral))
end

"""
    d_cpge(mu3, NC, omega1_tilde, omega2_tilde, x;
           lambda=0, delta=0, kernel=JacksonKernel)

Return the host-side energy-resolved bracket `RR + RA + AA` per unit `dx` at
`-1 < x < 1`, with `Delta_n(x) = T_n(x)/(pi*sqrt(1-x^2))`. The Fermi weight
and the overall factor `im` in [`cpge`](@ref) are not included.

This resolved function therefore takes no `beta`; in a manual integral,
`beta` is inverse rescaled energy.

The grid method without `x` returns `(energy_grid, values)` and accepts
`N_int=2NC` and `e_range=[-1,1]`.
"""
function d_cpge(
    mu3,
    NC::Integer,
    omega1_tilde::Real,
    omega2_tilde::Real,
    x::Real;
    lambda::Real = 0.0,
    delta::Real = 0.0,
    kernel = JacksonKernel,
)
    _check_cpge_inputs(mu3, NC, omega1_tilde, omega2_tilde, lambda, delta)
    -1 < x < 1 || throw(DomainError(x, "the resolved integrand requires -1 < x < 1"))
    mu_tilde = _cpge_kernel_moments(mu3, NC, kernel)
    F! = _cpge_node_function(
        mu_tilde,
        Int(NC),
        omega1_tilde,
        omega2_tilde,
        lambda,
        delta,
    )
    out = zeros(ComplexF64, 1)
    F!(out, acos(x))
    return ComplexF64(only(out) / sqrt(1 - x^2))
end

function d_cpge(
    mu3,
    NC::Integer,
    omega1_tilde::Real,
    omega2_tilde::Real;
    lambda::Real = 0.0,
    delta::Real = 0.0,
    kernel = JacksonKernel,
    N_int::Integer = 2NC,
    e_range = [-1.0, 1.0],
)
    _check_cpge_inputs(mu3, NC, omega1_tilde, omega2_tilde, lambda, delta)
    N_int > 0 || throw(ArgumentError("N_int must be positive"))
    length(e_range) == 2 || throw(ArgumentError("e_range must have two endpoints"))
    e_range[1] < e_range[2] || throw(ArgumentError("e_range endpoints must be increasing"))
    grid = collect(((0.5:N_int) ./ N_int) .* (e_range[2] - e_range[1]) .+ e_range[1])
    all(x -> -1 < x < 1, grid) ||
        throw(DomainError(e_range, "resolved grid nodes must lie strictly inside (-1,1)"))

    mu_tilde = _cpge_kernel_moments(mu3, NC, kernel)
    F! = _cpge_node_function(
        mu_tilde,
        Int(NC),
        omega1_tilde,
        omega2_tilde,
        lambda,
        delta,
    )
    out = zeros(ComplexF64, 1)
    values = ComplexF64[]
    sizehint!(values, N_int)
    for x in grid
        F!(out, acos(x))
        push!(values, only(out) / sqrt(1 - x^2))
    end
    return grid, values
end
