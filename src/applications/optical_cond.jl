function _check_optical_inputs(mu, NC, lambda = 0.0; omega_tilde = nothing)
    lambda >= 0 || throw(ArgumentError("lambda must be nonnegative (got $lambda)"))
    NC > 0 || throw(ArgumentError("NC must be positive"))
    all(size(mu, d) >= NC for d = 1:ndims(mu)) ||
        throw(ArgumentError("NC exceeds the supplied moment dimensions"))
    if omega_tilde !== nothing && iszero(lambda) && abs(omega_tilde) == 2
        throw(
            ArgumentError(
                "the shifted band edge coincides with the bare-band endpoint at " *
                "|omega_tilde| = 2; lambda > 0 (λ > 0) is required to " *
                "regularize the logarithmic singularity.",
            ),
        )
    end
    return nothing
end

"""
    optical_cond1(mu1, NC, omega_tilde; E_f=0, beta=Inf,
                  kernel=JacksonKernel, quad_rtol=1e-8, quad_atol=0,
                  maxevals=10^6)

Return the bare diamagnetic optical term
`(-im/omega_tilde) sum_n mu1_tilde[n] Lambda_n` in rescaled energy units.
`E_f` and `omega_tilde` are rescaled, and `beta` is inverse rescaled energy
(`beta = beta_physical*a`). For a two-dimensional physical result in `e^2/h`,
multiply by `2pi*D/(A*a)`, or use [`optical_cond`](@ref).

The moments are kernel- and `hn`-improved on the host. At zero temperature
`Lambda_n` is analytic; finite-temperature quadrature enforces
`error <= quad_atol + quad_rtol*abs(I)` on the scalar contracted integrand and
throws if `maxevals` is insufficient. A nonzero `quad_atol` is needed for a
component that vanishes by cancellation.
"""
function optical_cond1(
    mu1,
    NC::Integer,
    omega_tilde::Real;
    E_f::Real = 0.0,
    beta::Real = Inf,
    kernel = JacksonKernel,
    quad_rtol::Real = 1e-8,
    quad_atol::Real = 0.0,
    maxevals::Integer = 10^6,
)
    iszero(omega_tilde) &&
        throw(ArgumentError("optical conductivity is singular at omega = 0"))
    _check_optical_inputs(mu1, NC)
    mu_tilde = maybe_to_host(muND_apply_kernel_and_h(view(mu1, 1:NC), Int(NC), kernel; dims = [1]))
    integral = if isinf(beta)
        sum(mu_tilde .* _lambda_coefficients(NC, E_f))
    else
        active = findall(!iszero, mu_tilde)
        function node(theta)
            return sum(
                mu_tilde[n] * cos((n - 1) * theta) for n in active
            ) / pi / (1 + exp(beta * (cos(theta) - E_f)))
        end
        value, estimate = try
            quadgk(
                node,
                0.0,
                pi;
                rtol = quad_rtol / 10,
                atol = quad_atol / 10,
                maxevals = maxevals,
                order = 31,
                norm = abs,
            )
        catch err
            err isa InterruptException && rethrow()
            _quad_contract_error(
                quad_rtol,
                quad_atol,
                maxevals,
                0.0,
                sprint(showerror, err),
            )
        end
        estimate <= quad_atol + quad_rtol * abs(value) || _quad_contract_error(
            quad_rtol,
            quad_atol,
            maxevals,
            0.0,
            "estimated error $estimate",
        )
        value
    end
    return ComplexF64(-im * integral / omega_tilde)
end

function _optical_node_function(mus, NC, omega_tilde, lambda)
    delta = zeros(ComplexF64, NC)
    gR = similar(delta)
    gA = similar(delta)
    rightR = [similar(delta) for _ in mus]
    rightD = [similar(delta) for _ in mus]
    function F!(out, theta)
        chebyshev_delta_theta!(delta, theta)
        x = cos(theta)
        green_coefficients!(gR, x + omega_tilde, lambda, Val(:R))
        green_coefficients!(gA, x - omega_tilde, lambda, Val(:A))
        @inbounds for k in eachindex(mus)
            mul!(rightR[k], mus[k], delta)
            mul!(rightD[k], mus[k], gA)
            out[k] = sum(gR .* rightR[k]) + sum(delta .* rightD[k])
        end
        return out
    end
    return _SpectralNodeFunction(F!, length(mus), [all(iszero, mu) for mu in mus])
end

"""
    optical_cond2(mu2, NC, omega_tilde; E_f=0, beta=Inf, lambda=0,
                  kernel=JacksonKernel, quad_rtol=1e-8, quad_atol=0,
                  maxevals=10^6)

Return the bare paramagnetic optical term `(-im/omega_tilde) integral B` in
rescaled units. Here
`B = gR'*(mu2_tilde*Delta) + Delta'*(mu2_tilde*gA)`, with plain transposes.
The package table `mu2[n,m] = Tr[Jalpha T_m Jbeta T_n]/D` is contracted
directly with `Lambda_nm`. This orientation reproduces [`kubo_bastin_cond`](@ref)
as `omega_tilde -> 0`, including the package's ED/FHS-anchored Hall convention:
`Jalpha` is the response direction, `Jbeta` is the field direction, and
`sigma_xy = +C`. In the labels of Joao--Lopes Eqs. 26, 42, and 44, the same
quantity is their `sigma^{beta alpha}` because the two textbook Kubo forms
differ by exactly this index relabeling.

`E_f`, `omega_tilde`, and `lambda` are rescaled, while `beta` is inverse
rescaled energy (`beta = beta_physical*a`); finite `lambda` replaces JL's
`i0` by `i*lambda`. For a two-dimensional result in `e^2/h`, multiply by
`2pi*D/(A*a^2)`, or use [`optical_cond`](@ref). The adaptive integral
enforces `error[k] <= quad_atol + quad_rtol*abs(I[k])` for every batched
component and throws if `maxevals` is insufficient. Exact-zero components are
returned as zero, while components that vanish by cancellation require a
nonzero `quad_atol`. When `abs(omega_tilde) == 2`, the unregularized integral
has a logarithmic endpoint singularity and therefore requires `lambda > 0`.
Cost is `O(N_nodes*NC^2)`.
"""
function optical_cond2(
    mu2,
    NC::Integer,
    omega_tilde::Real;
    E_f::Real = 0.0,
    beta::Real = Inf,
    lambda::Real = 0.0,
    kernel = JacksonKernel,
    quad_rtol::Real = 1e-8,
    quad_atol::Real = 0.0,
    maxevals::Integer = 10^6,
)
    return only(
        optical_cond2(
            (mu2,),
            NC,
            omega_tilde;
            E_f = E_f,
            beta = beta,
            lambda = lambda,
            kernel = kernel,
            quad_rtol = quad_rtol,
            quad_atol = quad_atol,
            maxevals = maxevals,
        ),
    )
end

function optical_cond2(
    mus::Tuple,
    NC::Integer,
    omega_tilde::Real;
    E_f::Real = 0.0,
    beta::Real = Inf,
    lambda::Real = 0.0,
    kernel = JacksonKernel,
    quad_rtol::Real = 1e-8,
    quad_atol::Real = 0.0,
    maxevals::Integer = 10^6,
)
    isempty(mus) && return ComplexF64[]
    iszero(omega_tilde) &&
        throw(ArgumentError("optical conductivity is singular at omega = 0"))
    for mu in mus
        _check_optical_inputs(mu, NC, lambda; omega_tilde = omega_tilde)
    end
    mu_tilde = tuple(
        (maybe_to_host(mu2D_apply_kernel_and_h(view(mu, 1:NC, 1:NC), Int(NC), kernel)) for mu in mus)...,
    )
    F! = _optical_node_function(mu_tilde, NC, omega_tilde, lambda)
    integral, _ = _spectral_integral(
        F!,
        NC,
        (omega_tilde, -omega_tilde),
        E_f,
        beta,
        lambda;
        rtol = quad_rtol,
        atol = quad_atol,
        maxevals = maxevals,
    )
    return ComplexF64.((-im / omega_tilde) .* integral)
end

function _delta_coefficients_x(NC, x)
    -1 < x < 1 || throw(DomainError(x, "the resolved integrand requires -1 < x < 1"))
    delta = zeros(ComplexF64, NC)
    chebyshev_delta_theta!(delta, acos(x))
    delta ./= sqrt(1 - x^2)
    return delta
end

"""
    d_optical_cond1(mu1, NC, x; kernel=JacksonKernel)

The rescaled-energy integrand `sum_n mu1_tilde[n] Delta_n(x)` per unit `dx`,
for `x` strictly inside `(-1,1)`. The Fermi weight is omitted, so this
resolved function takes no `beta`; in a manual integral, `beta` is inverse
rescaled energy.
"""
function d_optical_cond1(mu1, NC::Integer, x::Real; kernel = JacksonKernel)
    _check_optical_inputs(mu1, NC)
    mu_tilde = maybe_to_host(muND_apply_kernel_and_h(view(mu1, 1:NC), Int(NC), kernel; dims = [1]))
    return ComplexF64(sum(mu_tilde .* _delta_coefficients_x(NC, x)))
end

function d_optical_cond1(
    mu1,
    NC::Integer;
    kernel = JacksonKernel,
    N_int::Integer = 2NC,
    e_range = [-1.0, 1.0],
)
    grid = collect(((0.5:N_int) ./ N_int) .* (e_range[2] - e_range[1]) .+ e_range[1])
    return grid, [d_optical_cond1(mu1, NC, x; kernel = kernel) for x in grid]
end

"""
    d_optical_cond2(mu2, NC, omega_tilde, x; lambda=0, kernel=JacksonKernel)

The rescaled-energy paramagnetic integrand per unit `dx`. The package layout
`mu2[n,m] = Tr[Jalpha T_m Jbeta T_n]/D` is contracted directly with
`Lambda_nm`. This is the orientation whose zero-frequency limit matches
[`kubo_bastin_cond`](@ref): `Jalpha` is response, `Jbeta` is field, and
`sigma_xy = +C` in the package's ED/FHS-anchored convention. In the labels of
Joao--Lopes Eqs. 26, 42, and 44, this is their `sigma^{beta alpha}` because
the two textbook Kubo forms differ by exactly this relabeling.
`lambda >= 0` is the rescaled Green broadening. The Fermi weight is omitted,
so this resolved function takes no `beta`; in a manual integral, `beta` is
inverse rescaled energy. Unlike [`optical_cond2`](@ref), the resolved
integrand is finite and accepted at `omega_tilde = 0` away from its Green
singular points.
"""
function d_optical_cond2(
    mu2,
    NC::Integer,
    omega_tilde::Real,
    x::Real;
    lambda::Real = 0.0,
    kernel = JacksonKernel,
)
    _check_optical_inputs(mu2, NC, lambda)
    mu_tilde = maybe_to_host(mu2D_apply_kernel_and_h(view(mu2, 1:NC, 1:NC), Int(NC), kernel))
    delta = _delta_coefficients_x(NC, x)
    gR = zeros(ComplexF64, NC)
    gA = similar(gR)
    green_coefficients!(gR, x + omega_tilde, lambda, Val(:R))
    green_coefficients!(gA, x - omega_tilde, lambda, Val(:A))
    return ComplexF64(sum(gR .* (mu_tilde * delta)) + sum(delta .* (mu_tilde * gA)))
end

function d_optical_cond2(
    mu2,
    NC::Integer,
    omega_tilde::Real;
    lambda::Real = 0.0,
    kernel = JacksonKernel,
    N_int::Integer = 2NC,
    e_range = [-1.0, 1.0],
)
    grid = collect(((0.5:N_int) ./ N_int) .* (e_range[2] - e_range[1]) .+ e_range[1])
    values = [
        d_optical_cond2(mu2, NC, omega_tilde, x; lambda = lambda, kernel = kernel) for
        x in grid
    ]
    return grid, values
end
