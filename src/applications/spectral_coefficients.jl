using QuadGK

struct _SpectralNodeFunction{F}
    f!::F
    ncomp::Int
end

(F::_SpectralNodeFunction)(out, theta) = F.f!(out, theta)
_spectral_ncomp(F::_SpectralNodeFunction, NC) = F.ncomp
_spectral_ncomp(F, NC) = NC

"""
    chebyshev_delta_theta!(delta, theta)

Fill `delta[n+1] = cos(n*theta)/pi`. These are the package-normalized
Chebyshev delta coefficients after `x = cos(theta)`, including the Jacobian.
"""
function chebyshev_delta_theta!(delta::AbstractVector, theta::Real)
    isempty(delta) && return delta
    c = cos(theta)
    delta[1] = inv(pi)
    length(delta) == 1 && return delta
    delta[2] = c / pi
    @inbounds for n = 3:length(delta)
        delta[n] = 2c * delta[n-1] - delta[n-2]
    end
    return delta
end

"""
    green_coefficients!(g, z, lambda, Val(:R))
    green_coefficients!(g, z, lambda, Val(:A))

Fill the package-normalized Chebyshev coefficients of the retarded or
advanced resolvent. `lambda >= 0` is the broadening in rescaled energy units.
For `lambda == 0`, all real `z` except the singular points `z = +/-1` are
supported, including the real resolvent tails outside the Chebyshev interval.
"""
function green_coefficients!(
    g::AbstractVector{<:Complex},
    z::Real,
    lambda::Real,
    branch::Union{Val{:R},Val{:A}},
)
    lambda >= 0 || throw(ArgumentError("lambda must be nonnegative (got $lambda)"))
    isempty(g) && return g

    if iszero(lambda)
        abs(z) == 1 && throw(DomainError(z, "z = +/-1 is singular when lambda = 0"))
        if abs(z) < 1
            phi = acos(z)
            s = sin(phi)
            if branch isa Val{:R}
                g[1] = -im / s
                q = cis(-phi)
            else
                g[1] = im / s
                q = cis(phi)
            end
        elseif z > 1
            g[1] = inv(sqrt(z^2 - 1))
            q = exp(-acosh(z))
        else
            g[1] = -inv(sqrt(z^2 - 1))
            q = -exp(-acosh(-z))
        end
    elseif branch isa Val{:R}
        w = complex(z, lambda)
        phi = acos(w)
        g[1] = -im / sqrt(1 - w^2)
        q = exp(-im * phi)
    else
        w = complex(z, -lambda)
        phi = acos(w)
        g[1] = im / sqrt(1 - w^2)
        q = exp(im * phi)
    end

    @inbounds for n = 2:length(g)
        g[n] = g[n-1] * q
    end
    return g
end

function _spectral_fermi(x, xF, beta_a)
    isinf(beta_a) && return ((x < xF) + (x <= xF)) / 2
    y = beta_a * (x - xF)
    if y > 0
        ey = exp(-y)
        return ey / (1 + ey)
    end
    return inv(1 + exp(y))
end

function _deduplicate_sorted(points; tol = 1e-12)
    sort!(points)
    out = Float64[]
    for point in points
        (isempty(out) || abs(point - out[end]) > tol) && push!(out, point)
    end
    return out
end

function _quad_contract_error(rtol, atol, maxevals, lambda, detail = "")
    suffix = isempty(detail) ? "" : " ($detail)"
    return error(
        "spectral quadrature did not meet its error contract$suffix; adjust " *
        "quad_rtol, quad_atol, maxevals, or lambda. A nonzero quad_atol is " *
        "needed for symmetry-forbidden components that are exactly zero.",
    )
end

"""
    _spectral_integral(F!, NC, shifts, xF, beta_a, lambda;
                       rtol=1e-8, atol=0, maxevals=10^6, order=7)

Integrate `f(cos(theta))*F(theta)` over the occupied part of `(0, pi)`.
Shifted real-axis Green-function edges are split out and regularized by an
endpoint-squaring map when `lambda == 0`. The summed QuadGK estimate must
satisfy `error[k] <= atol + rtol*abs(integral[k])` for every component or the
call throws. Vector-valued integrands use a scale-estimating first pass and a
weighted maximum-norm second pass; both share `maxevals`.
"""
function _spectral_integral(
    F!,
    NC::Integer,
    shifts,
    xF::Real,
    beta_a::Real,
    lambda::Real;
    rtol::Real = 1e-8,
    atol::Real = 0.0,
    maxevals::Integer = 10^6,
    order::Integer = 7,
    breakpoints = (),
)
    NC > 0 || throw(ArgumentError("NC must be positive"))
    lambda >= 0 || throw(ArgumentError("lambda must be nonnegative (got $lambda)"))
    rtol >= 0 || throw(ArgumentError("rtol must be nonnegative"))
    atol >= 0 || throw(ArgumentError("atol must be nonnegative"))
    maxevals > 0 || throw(ArgumentError("maxevals must be positive"))
    order > 0 || throw(ArgumentError("order must be positive"))

    ncomp = _spectral_ncomp(F!, NC)
    ncomp > 0 || throw(ArgumentError("the node function must have at least one component"))
    empty_integral = zeros(ComplexF64, ncomp)
    isinf(beta_a) && xF <= -1 && return empty_integral, 0.0

    theta_F = acos(clamp(xF, -1, 1))
    theta_lo = isinf(beta_a) ? theta_F : 0.0
    theta_hi = pi
    points = Float64[theta_lo, theta_hi]
    singular = Float64[]
    for shift in shifts
        for edge in (-1.0, 1.0)
            xstar = edge - shift
            if -1 < xstar < 1
                theta = acos(xstar)
                theta_lo - 1e-12 <= theta <= theta_hi + 1e-12 || continue
                abs(theta - theta_lo) <= 1e-12 && (theta = theta_lo)
                abs(theta - theta_hi) <= 1e-12 && (theta = theta_hi)
                push!(points, theta)
                push!(singular, theta)
            end
        end
    end
    if !isinf(beta_a) && beta_a > 50 && -1 < xF < 1
        push!(points, theta_F)
    end
    for theta in breakpoints
        theta_lo < theta < theta_hi && push!(points, Float64(theta))
    end
    points = _deduplicate_sorted(points)
    singular = _deduplicate_sorted(singular)
    is_singular(theta) = any(abs(theta - s) <= 1e-12 for s in singular)
    piece_count = sum(
        iszero(lambda) && is_singular(points[k]) && is_singular(points[k+1]) ? 2 : 1 for
        k = 1:(length(points)-1)
    )
    evaluations = Ref(0)
    function node(theta)
        evaluations[] += 1
        out = zeros(ComplexF64, ncomp)
        F!(out, theta)
        out .*= _spectral_fermi(cos(theta), xF, beta_a)
        return out
    end

    function integrate_piece(a, b, left_singular, right_singular, pass_rtol, pass_atol, pass_norm)
        a == b && return zeros(ComplexF64, ncomp), 0.0
        if left_singular && right_singular
            mid = (a + b) / 2
            I1, E1 = integrate_piece(a, mid, true, false, pass_rtol, pass_atol, pass_norm)
            I2, E2 = integrate_piece(mid, b, false, true, pass_rtol, pass_atol, pass_norm)
            return I1 + I2, E1 + E2
        end
        width = b - a
        integrand = if iszero(lambda) && left_singular
            u -> node(a + width * u^2) .* (2width * u)
        elseif iszero(lambda) && right_singular
            u -> node(b - width * u^2) .* (2width * u)
        else
            node
        end
        qa, qb = iszero(lambda) && (left_singular || right_singular) ? (0.0, 1.0) : (a, b)
        remaining = maxevals - evaluations[]
        remaining > 0 || _quad_contract_error(rtol, atol, maxevals, lambda, "maxevals exhausted")
        try
            return quadgk(
                integrand,
                qa,
                qb;
                rtol = pass_rtol,
                atol = pass_atol,
                maxevals = remaining,
                order = order,
                norm = pass_norm,
            )
        catch err
            err isa InterruptException && rethrow()
            _quad_contract_error(rtol, atol, maxevals, lambda, sprint(showerror, err))
        end
    end

    function integrate_pass(pass_rtol, pass_atol, pass_norm)
        integral = zeros(ComplexF64, ncomp)
        total_error = 0.0
        for k = 1:(length(points)-1)
            a, b = points[k], points[k+1]
            value, estimate = integrate_piece(
                a,
                b,
                is_singular(a),
                is_singular(b),
                pass_rtol,
                pass_atol,
                pass_norm,
            )
            integral .+= value
            total_error += estimate
        end
        return integral, total_error
    end

    # Leave headroom for cancellation when the independently estimated piece
    # errors are summed against the norm of the final integral.
    piece_rtol = rtol / (10piece_count)
    piece_atol = atol / (10piece_count)
    first_integral, first_error = integrate_pass(piece_rtol, piece_atol, LinearAlgebra.norm)
    first_error <= atol + rtol * LinearAlgebra.norm(first_integral) ||
        _quad_contract_error(
            rtol,
            atol,
            maxevals,
            lambda,
            "estimated error $first_error",
        )
    ncomp == 1 && return first_integral, first_error

    scales = atol .+ rtol .* abs.(first_integral)
    any(iszero, scales) && _quad_contract_error(
        rtol,
        atol,
        maxevals,
        lambda,
        "quad_atol is required for exactly-zero components",
    )
    weights = inv.(scales)
    component_norm = x -> maximum(k -> abs(x[k]) * weights[k], eachindex(weights))
    integral, weighted_error = integrate_pass(
        0.0,
        1 / (10piece_count),
        component_norm,
    )
    weighted_error <= 1 || _quad_contract_error(
        rtol,
        atol,
        maxevals,
        lambda,
        "maximum scaled estimated error $weighted_error",
    )
    return integral, weighted_error
end

function _lambda_coefficients(
    NC::Integer,
    xF::Real,
)
    xF <= -1 && return zeros(ComplexF64, NC)
    theta_F = acos(clamp(xF, -1, 1))
    lambda_n = zeros(ComplexF64, NC)
    lambda_n[1] = 1 - theta_F / pi
    @inbounds for n = 1:(NC-1)
        lambda_n[n+1] = -sin(n * theta_F) / (n * pi)
    end
    return lambda_n
end
