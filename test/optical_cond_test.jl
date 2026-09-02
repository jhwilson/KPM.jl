using Test
using LinearAlgebra
using SparseArrays
using Random
using QuadGK
using KPM

isdefined(@__MODULE__, :EDReference) || include("ed_reference.jl")
using .EDReference

function _chebyshev_matrices(A, NC)
    D = size(A, 1)
    T = Vector{Matrix{ComplexF64}}(undef, NC)
    T[1] = Matrix{ComplexF64}(I, D, D)
    NC >= 2 && (T[2] = Matrix{ComplexF64}(A))
    for k = 3:NC
        T[k] = 2 * Matrix(A) * T[k-1] - T[k-2]
    end
    return T
end

function _ref_chebyshev(x, NC)
    T = zeros(promote_type(typeof(x), Float64), NC)
    NC == 0 && return T
    T[1] = 1
    NC == 1 && return T
    T[2] = x
    for n = 3:NC
        T[n] = 2x * T[n-1] - T[n-2]
    end
    return T
end

_ref_hn(n) = n == 0 ? 1.0 : 2.0
function _ref_jackson(n, NC)
    alpha = pi / (NC + 1)
    return ((NC + 1 - n) * cos(n * alpha) + sin(n * alpha) * cot(alpha)) / (NC + 1)
end

function _ref_delta(x, NC)
    return _ref_chebyshev(x, NC) ./ (pi * sqrt(1 - x^2))
end

function _ref_green(z::Real, NC, lambda, branch)
    if iszero(lambda)
        if abs(z) < 1
            phi = acos(z)
            sign = branch === :R ? -1 : 1
            prefactor = branch === :R ? -im / sin(phi) : im / sin(phi)
            return ComplexF64[prefactor * exp(sign * im * n * phi) for n = 0:(NC-1)]
        elseif z > 1
            eta = acosh(z)
            return ComplexF64[exp(-n * eta) / sqrt(z^2 - 1) for n = 0:(NC-1)]
        elseif z < -1
            eta = acosh(-z)
            return ComplexF64[-(-1)^n * exp(-n * eta) / sqrt(z^2 - 1) for n = 0:(NC-1)]
        end
        throw(DomainError(z))
    end
    w = branch === :R ? complex(z, lambda) : complex(z, -lambda)
    if branch === :R
        return ComplexF64[-im * exp(-im * n * acos(w)) / sqrt(1 - w^2) for n = 0:(NC-1)]
    end
    return ComplexF64[im * exp(im * n * acos(w)) / sqrt(1 - w^2) for n = 0:(NC-1)]
end

function _ref_fermi(x, xF, beta)
    isinf(beta) && return ((x < xF) + (x <= xF)) / 2
    y = beta * (x - xF)
    y > 0 && return exp(-y) / (1 + exp(-y))
    return inv(1 + exp(y))
end

function _ref_integral(fun, shifts, xF, beta, lambda; rtol = 2e-11, atol = 2e-13)
    isinf(beta) && xF <= -1 && return 0.0 + 0.0im
    lo = isinf(beta) ? acos(clamp(xF, -1, 1)) : 0.0
    points = Float64[lo, pi]
    singular = Float64[]
    for shift in shifts, edge in (-1.0, 1.0)
        xstar = edge - shift
        if -1 < xstar < 1
            theta = acos(xstar)
            if lo < theta < pi
                push!(points, theta)
                push!(singular, theta)
            end
        end
    end
    !isinf(beta) && beta > 50 && -1 < xF < 1 && push!(points, acos(xF))
    sort!(points)
    unique!(points)
    isedge(theta) = any(s -> abs(theta - s) < 1e-13, singular)
    node(theta) = sin(theta) * _ref_fermi(cos(theta), xF, beta) * fun(cos(theta))

    function piece(a, b, leftedge, rightedge)
        if leftedge && rightedge
            mid = (a + b) / 2
            return piece(a, mid, true, false) + piece(mid, b, false, true)
        elseif iszero(lambda) && leftedge
            width = b - a
            return first(quadgk(u -> node(a + width * u^2) * 2width * u, 0, 1;
                rtol = rtol, atol = atol, order = 15))
        elseif iszero(lambda) && rightedge
            width = b - a
            return first(quadgk(u -> node(b - width * u^2) * 2width * u, 0, 1;
                rtol = rtol, atol = atol, order = 15))
        end
        return first(quadgk(node, a, b; rtol = rtol, atol = atol, order = 15))
    end

    return sum(
        piece(points[k], points[k+1], isedge(points[k]), isedge(points[k+1])) for
        k = 1:(length(points)-1)
    )
end

# Independent T=0 reference for a shifted Green singularity at the active
# Fermi boundary. Unlike `_ref_integral`, boundary singularities are retained
# explicitly and every singular endpoint is mapped with theta = theta_* +/- w*u^2.
function _ref_endpoint_integral(fun, shifts, xF; rtol = 2e-11, atol = 2e-13)
    lo = acos(clamp(xF, -1, 1))
    points = Float64[lo, pi]
    singular = Float64[]
    for shift in shifts, edge in (-1.0, 1.0)
        xstar = edge - shift
        -1 < xstar < 1 || continue
        theta = acos(xstar)
        lo - 1e-13 <= theta <= pi + 1e-13 || continue
        abs(theta - lo) <= 1e-13 && (theta = lo)
        abs(theta - pi) <= 1e-13 && (theta = pi)
        push!(points, theta)
        push!(singular, theta)
    end
    sort!(points)
    unique!(points)
    isedge(theta) = any(s -> abs(theta - s) <= 1e-13, singular)
    node(theta) = sin(theta) * fun(cos(theta))

    function piece(a, b, leftedge, rightedge)
        if leftedge && rightedge
            mid = (a + b) / 2
            return piece(a, mid, true, false) + piece(mid, b, false, true)
        end
        width = b - a
        if leftedge
            return first(quadgk(
                u -> node(a + width * u^2) * (2width * u),
                0.0,
                1.0;
                rtol = rtol,
                atol = atol,
                order = 21,
            ))
        elseif rightedge
            return first(quadgk(
                u -> node(b - width * u^2) * (2width * u),
                0.0,
                1.0;
                rtol = rtol,
                atol = atol,
                order = 21,
            ))
        end
        return first(quadgk(node, a, b; rtol = rtol, atol = atol, order = 21))
    end

    return sum(
        piece(points[k], points[k+1], isedge(points[k]), isedge(points[k+1])) for
        k = 1:(length(points)-1)
    )
end

function _eigenbasis_reference(Hnorm, Ja, Jb, NC, omega; xF, beta, lambda)
    F = eigen(Hermitian(Matrix(Hnorm)))
    A = F.vectors' * Matrix(Ja) * F.vectors
    B = F.vectors' * Matrix(Jb) * F.vectors
    T = hcat((_ref_chebyshev(E, NC) for E in F.values)...)
    kh = [_ref_jackson(n, NC) * _ref_hn(n) for n = 0:(NC-1)]
    function integrand(x)
        deltaK = transpose(kh .* _ref_delta(x, NC)) * T
        greenR = transpose(kh .* _ref_green(x + omega, NC, lambda, :R)) * T
        greenA = transpose(kh .* _ref_green(x - omega, NC, lambda, :A)) * T
        acc = 0.0 + 0.0im
        # mu[n,m] = Tr[Ja*T_m*Jb*T_n]/D
        #         = sum_ab Ja_ab*T_m(E_b)*Jb_ba*T_n(E_a)/D.
        # Direct contraction with Lambda_nm therefore pairs the retarded
        # coefficient at n with E_a and delta at m with E_b; the advanced
        # term pairs delta at n with E_a and the advanced coefficient at m
        # with E_b.
        for a in eachindex(F.values), b in eachindex(F.values)
            acc += A[a, b] * B[b, a] * (
                greenR[a] * deltaK[b] + deltaK[a] * greenA[b]
            )
        end
        return acc / length(F.values)
    end
    return (-im / omega) * _ref_integral(integrand, (omega, -omega), xF, beta, lambda)
end

function _eigenbasis_endpoint_reference(Hnorm, Ja, Jb, NC, omega; xF)
    F = eigen(Hermitian(Matrix(Hnorm)))
    A = F.vectors' * Matrix(Ja) * F.vectors
    B = F.vectors' * Matrix(Jb) * F.vectors
    T = hcat((_ref_chebyshev(E, NC) for E in F.values)...)
    kh = [_ref_jackson(n, NC) * _ref_hn(n) for n = 0:(NC-1)]
    function integrand(x)
        deltaK = transpose(kh .* _ref_delta(x, NC)) * T
        greenR = transpose(kh .* _ref_green(x + omega, NC, 0.0, :R)) * T
        greenA = transpose(kh .* _ref_green(x - omega, NC, 0.0, :A)) * T
        return sum(
            A[a, b] * B[b, a] *
            (greenR[a] * deltaK[b] + deltaK[a] * greenA[b]) for
            a in eachindex(F.values), b in eachindex(F.values)
        ) / length(F.values)
    end
    return (-im / omega) * _ref_endpoint_integral(integrand, (omega, -omega), xF)
end

function _diamagnetic_reference(Hnorm, Jaa, NC, omega; xF, beta)
    F = eigen(Hermitian(Matrix(Hnorm)))
    J = F.vectors' * Matrix(Jaa) * F.vectors
    T = hcat((_ref_chebyshev(E, NC) for E in F.values)...)
    kh = [_ref_jackson(n, NC) * _ref_hn(n) for n = 0:(NC-1)]
    function integrand(x)
        deltaK = transpose(kh .* _ref_delta(x, NC)) * T
        return sum(J[a, a] * deltaK[a] for a in eachindex(F.values)) / length(F.values)
    end
    return (-im / omega) * _ref_integral(integrand, (), xF, beta, 0.0)
end

function _second_current(H, J)
    Hd = Matrix(H)
    Jd = Matrix(J)
    displacement = [
        abs(Hd[i, j]) > 0 ? Jd[i, j] / Hd[i, j] : 0.0 + 0.0im for
        i in axes(Hd, 1), j in axes(Hd, 2)
    ]
    return sparse(Hd .* displacement .^ 2)
end

function _lehmann_optical_reference(Hnorm, Ja, Jb, omega, lambda, xF)
    F = eigen(Hermitian(Matrix(Hnorm)))
    A = F.vectors' * Matrix(Ja) * F.vectors
    B = F.vectors' * Matrix(Jb) * F.vectors
    lorentz(x, E) = lambda / (pi * ((x - E)^2 + lambda^2))
    function integrand(x)
        acc = 0.0 + 0.0im
        for a in eachindex(F.values), b in eachindex(F.values)
            greenR = inv(x + omega - F.values[a] + im * lambda)
            greenA = inv(x - omega - F.values[b] - im * lambda)
            acc += A[a, b] * B[b, a] * (
                greenR * lorentz(x, F.values[b]) +
                lorentz(x, F.values[a]) * greenA
            )
        end
        return acc / length(F.values)
    end
    integral = first(quadgk(integrand, -Inf, xF; rtol = 1e-11, atol = 1e-13))
    return (-im / omega) * integral
end

# Small exact-trace Haldane fixture.
Hop, Jx, Jy, area = haldane_model(3, 3; t = 1.0, t2 = 0.2, ϕ = pi / 2, m = 0.0)
D = size(Hop, 1)
let eigenvalues = eigvals(Hermitian(Matrix(Hop)))
    global b_r = (maximum(eigenvalues) + minimum(eigenvalues)) / 2
    global a_r = (maximum(eigenvalues) - minimum(eigenvalues)) / (2 - 0.1)
end
H_norm = sparse((Matrix(Hop) - b_r * I) / a_r)
Hm = Matrix(Hop)
dx = [abs(Hm[i, j]) > 0 ? Matrix(Jx)[i, j] / Hm[i, j] : 0.0 + 0im for i = 1:D, j = 1:D]
Jxx = sparse(Hm .* dx .^ 2)
NC = 16
psi = Matrix{ComplexF64}(I, D, D)
Tmat = _chebyshev_matrices(H_norm, NC)

@testset "kpm_1d_current moments vs dense Chebyshev matrices" begin
    @test norm(Jxx - Jxx') == 0
    @test norm(Jx + Jx') == 0
    @test maximum(abs, eigvals(Hermitian(Matrix(H_norm)))) < 1

    mu = KPM.kpm_1d_current(H_norm, Jxx, NC, D, D; psi_in = psi)
    ref = [tr(Jxx * Tmat[n]) / D for n = 1:NC]
    @test mu ≈ ref rtol = 1e-12
    @test maximum(abs, imag.(mu)) < 1e-14

    mu_ser = KPM.kpm_1d_current(H_norm, Jxx, NC, D, D; psi_in = psi, NR_parallel = false)
    @test mu_ser ≈ mu rtol = 1e-12

    mu_all = KPM.kpm_1d_current(H_norm, Jxx, NC, D, D; psi_in = psi, avg_output = false)
    @test size(mu_all) == (D, NC)
    @test vec(sum(mu_all, dims = 1) ./ D) ≈ mu rtol = 1e-12

    rng = Xoshiro(4242)
    NRs = 4
    psi_u = KPM.random_phase_vectors(rng, D, NRs) .* reshape([1.0, 3.0, 0.25, 7.0], 1, NRs)
    p_forced = copy(psi_u)
    mu_forced = KPM.kpm_1d_current(H_norm, Jxx, NC, NRs, D;
        psi_in = p_forced, force_norm = true)
    p_pre = copy(psi_u)
    KPM.normalize_by_col(p_pre, NRs)
    mu_pre = KPM.kpm_1d_current(H_norm, Jxx, NC, NRs, D; psi_in = p_pre)
    @test mu_forced ≈ mu_pre rtol = 1e-12
end

@testset "kpm_1d_current: Tr[J T_n] for an anti-Hermitian bond current" begin
    N = 12
    hring, _, ringdisp = flux_ring_model(N; t = 1.0, phi = 0.35)
    Jring = sparse([hring[i, j] * ringdisp(i, j)[1] for i = 1:N, j = 1:N])
    @test norm(Jring + Jring') == 0

    ar = 2 * maximum(abs, eigvals(Hermitian(Matrix(hring)))) / (2 - 0.1)
    hn_ring = sparse(Matrix(hring) / ar)
    NCr = 12
    Tring = _chebyshev_matrices(hn_ring, NCr)
    psir = Matrix{ComplexF64}(I, N, N)
    tr_J = [tr(Jring * Tring[n]) / N for n = 1:NCr]
    @test maximum(abs, tr_J) > 0.5
    @test maximum(abs, real.(tr_J)) < 1e-14

    mu_default = KPM.kpm_1d_current(hn_ring, Jring, NCr, N, N; psi_in = psir)
    @test mu_default isa Vector{ComplexF64}
    @test mu_default ≈ tr_J rtol = 1e-12 atol = 1e-14
    @test maximum(abs, real.(mu_default)) < 1e-14
    @test maximum(abs, mu_default) > 0.5

    mu_all = KPM.kpm_1d_current(hn_ring, Jring, NCr, N, N;
        psi_in = psir, avg_output = false)
    mu_cplx = vec(sum(mu_all, dims = 1) ./ N)
    @test mu_cplx ≈ tr_J rtol = 1e-12 atol = 1e-14
    @test maximum(abs, real.(mu_cplx)) < 1e-14
    @test maximum(abs, mu_cplx) > 0.5
end

@testset "kpm_1d_current: independent bra/ket is documented as unimplemented" begin
    psil = KPM.random_phase_vectors(Xoshiro(1), D, 1)
    psir = KPM.random_phase_vectors(Xoshiro(2), D, 1)
    @test_throws String KPM.kpm_1d_current(H_norm, Jxx, NC, 1, D;
        psi_in_l = psil, psi_in_r = psir)
    @test_throws String KPM.kpm_1d_current!(H_norm, Jxx, NC, 1, D,
        zeros(ComplexF64, 1, NC), psil, psir)
end

@testset "spectral coefficients" begin
    NCg = 12
    for z in (0.3, 1.7, -2.2), branch in (:R, :A)
        g0 = zeros(ComplexF64, NCg)
        gp = similar(g0)
        KPM.green_coefficients!(g0, z, 0.0, Val(branch))
        KPM.green_coefficients!(gp, z, 1e-9, Val(branch))
        @test gp ≈ g0 rtol = 1e-6 atol = 1e-9
    end

    for z in (1.5, -1.8)
        Nsum, E = 200, 0.4
        g = zeros(ComplexF64, Nsum)
        KPM.green_coefficients!(g, z, 0.0, Val(:R))
        series = sum(_ref_hn(n) * g[n+1] * _ref_chebyshev(E, Nsum)[n+1] for n = 0:(Nsum-1))
        @test series ≈ inv(z - E) rtol = 1e-10
    end

    for branch in (:R, :A)
        Nsum, z, E, lambda = 2000, 0.3, 0.4, 0.05
        g = zeros(ComplexF64, Nsum)
        KPM.green_coefficients!(g, z, lambda, Val(branch))
        T = _ref_chebyshev(E, Nsum)
        series = sum(_ref_hn(n) * g[n+1] * T[n+1] for n = 0:(Nsum-1))
        target = inv(z - E + (branch === :R ? im * lambda : -im * lambda))
        @test series ≈ target rtol = 1e-6
    end

    z = 0.37
    gR = zeros(ComplexF64, NCg)
    gA = similar(gR)
    KPM.green_coefficients!(gR, z, 0.0, Val(:R))
    KPM.green_coefficients!(gA, z, 0.0, Val(:A))
    delta = _ref_delta(z, NCg)
    @test gR - gA ≈ -2pi * im .* delta rtol = 1e-12

    theta = 0.73
    delta_theta = zeros(ComplexF64, NCg)
    KPM.chebyshev_delta_theta!(delta_theta, theta)
    @test delta_theta ≈ _ref_chebyshev(cos(theta), NCg) ./ pi rtol = 1e-14
    @test_throws DomainError KPM.green_coefficients!(gR, 1.0, 0.0, Val(:R))
    @test_throws DomainError KPM.green_coefficients!(gA, -1.0, 0.0, Val(:A))
end

@testset "same-kernel eigenbasis optical reference" begin
    NCo = 24
    psi_exact = Matrix{ComplexF64}(I, D, D)
    mu2 = KPM.kpm_2d(H_norm, Jx, Jy, NCo, D, D; psi_in = psi_exact)
    mu1 = KPM.kpm_1d_current(H_norm, Jxx, NCo, D, D; psi_in = psi_exact)
    for (xF, beta) in ((0.1, Inf), (-0.1, 20.0)), lambda in (0.0, 0.03), omega in (0.35, 0.9)
        ref2 = _eigenbasis_reference(H_norm, Jx, Jy, NCo, omega;
            xF = xF, beta = beta, lambda = lambda)
        value2 = KPM.optical_cond2(mu2, NCo, omega;
            E_f = xF, beta = beta, lambda = lambda,
            quad_rtol = 2e-10, quad_atol = 2e-12)
        @test value2 ≈ ref2 rtol = 1e-8 atol = 2e-10

        ref1 = _diamagnetic_reference(H_norm, Jxx, NCo, omega;
            xF = xF, beta = beta)
        value1 = KPM.optical_cond1(mu1, NCo, omega;
            E_f = xF, beta = beta, quad_rtol = 2e-10, quad_atol = 2e-12)
        @test value1 ≈ ref1 rtol = 1e-8 atol = 2e-10
    end

    # The independent reference is converged beyond the asserted comparison.
    loose = _eigenbasis_reference(H_norm, Jx, Jy, NCo, 0.35;
        xF = 0.1, beta = Inf, lambda = 0.0)
    tight = let
        F = eigen(Hermitian(Matrix(H_norm)))
        A = F.vectors' * Matrix(Jx) * F.vectors
        B = F.vectors' * Matrix(Jy) * F.vectors
        T = hcat((_ref_chebyshev(E, NCo) for E in F.values)...)
        kh = [_ref_jackson(n, NCo) * _ref_hn(n) for n = 0:(NCo-1)]
        integrand(x) = begin
            d = transpose(kh .* _ref_delta(x, NCo)) * T
            r = transpose(kh .* _ref_green(x + 0.35, NCo, 0.0, :R)) * T
            a = transpose(kh .* _ref_green(x - 0.35, NCo, 0.0, :A)) * T
            sum(A[i, j] * B[j, i] * (r[i] * d[j] + d[i] * a[j])
                for i in eachindex(F.values), j in eachindex(F.values)) / D
        end
        (-im / 0.35) * _ref_integral(integrand, (0.35, -0.35), 0.1, Inf, 0.0;
            rtol = 2e-12, atol = 2e-14)
    end
    @test loose ≈ tight rtol = 1e-10 atol = 1e-12

    omega_edge = 0.3
    for xF_edge in (1 - omega_edge, -1 + omega_edge)
        endpoint_ref = _eigenbasis_endpoint_reference(
            H_norm,
            Jx,
            Jy,
            NCo,
            omega_edge;
            xF = xF_edge,
        )
        endpoint_value = KPM.optical_cond2(
            mu2,
            NCo,
            omega_edge;
            E_f = xF_edge,
            quad_rtol = 2e-10,
            quad_atol = 2e-12,
        )
        @test endpoint_value ≈ endpoint_ref rtol = 1e-8 atol = 2e-10
    end

    full_band = KPM.optical_cond2(
        mu2,
        NCo,
        omega_edge;
        E_f = 1.0,
        lambda = 0.03,
        quad_atol = 2e-10,
    )
    @test KPM.optical_cond2(
        mu2,
        NCo,
        omega_edge;
        E_f = 1.2,
        lambda = 0.03,
        quad_atol = 2e-10,
    ) ≈ full_band rtol = 1e-12 atol = 1e-12
    @test iszero(KPM.optical_cond2(mu2, NCo, omega_edge; E_f = -1.0))
    @test iszero(KPM.optical_cond2(mu2, NCo, omega_edge; E_f = -1.2))
end

@testset "optical moment orientation" begin
    NCo, omega = 7, 0.43
    Ts = _chebyshev_matrices(H_norm, NCo)
    mu_dense = [tr(Matrix(Jx) * Ts[m] * Matrix(Jy) * Ts[n]) / D
        for n = 1:NCo, m = 1:NCo]
    kh = [_ref_jackson(n, NCo) * _ref_hn(n) for n = 0:(NCo-1)]
    lambda_nm = [
        _ref_integral(
            x -> begin
                delta = _ref_delta(x, NCo)
                gR = _ref_green(x + omega, NCo, 0.0, :R)
                gA = _ref_green(x - omega, NCo, 0.0, :A)
                gR[n] * delta[m] + delta[n] * gA[m]
            end,
            (omega, -omega),
            0.1,
            Inf,
            0.0,
        ) for n = 1:NCo, m = 1:NCo
    ]
    direct = (-im / omega) * sum(mu_dense .* (kh * transpose(kh)) .* lambda_nm)
    mu_xy = KPM.kpm_2d(H_norm, Jx, Jy, NCo, D, D; psi_in = psi)
    value = KPM.optical_cond2(mu_xy, NCo, omega;
        E_f = 0.1, quad_rtol = 2e-10, quad_atol = 2e-12)
    @test value ≈ direct rtol = 1e-8 atol = 1e-10

    mu_yx = KPM.kpm_2d(H_norm, Jy, Jx, NCo, D, D; psi_in = psi)
    xy, yx = KPM.optical_cond2((mu_xy, mu_yx), NCo, omega;
        E_f = 0.1, quad_rtol = 2e-10, quad_atol = 2e-12)
    symmetric_xy = (xy + yx) / 2
    antisymmetric_xy = (xy - yx) / 2
    symmetric_yx = (yx + xy) / 2
    antisymmetric_yx = (yx - xy) / 2
    @test symmetric_xy ≈ symmetric_yx rtol = 1e-12 atol = 1e-12
    @test antisymmetric_xy ≈ -antisymmetric_yx rtol = 1e-12 atol = 1e-12
end

@testset "batch and resolved optical integrands" begin
    NCo, omega, x = 12, 0.51, -0.23
    mu_xy = KPM.kpm_2d(H_norm, Jx, Jy, NCo, D, D; psi_in = psi)
    mu_yx = KPM.kpm_2d(H_norm, Jy, Jx, NCo, D, D; psi_in = psi)
    kwargs = (; E_f = -0.1, beta = 13.0, lambda = 0.03,
        quad_rtol = 2e-11, quad_atol = 2e-13)
    batch = KPM.optical_cond2((mu_xy, mu_yx), NCo, omega; kwargs...)
    separate = [KPM.optical_cond2(mu, NCo, omega; kwargs...) for mu in (mu_xy, mu_yx)]
    @test batch ≈ separate rtol = 1e-13 atol = 1e-13

    NCscale = 64
    easy = zeros(ComplexF64, NCscale, NCscale)
    hard = zeros(ComplexF64, NCscale, NCscale)
    easy[1, 1] = 10
    hard[end, end] = 1
    scale_kwargs = (;
        E_f = 0.2,
        lambda = 0.02,
        quad_rtol = 1e-8,
        quad_atol = 0.0,
    )
    scale_batch = KPM.optical_cond2((easy, hard), NCscale, 0.3; scale_kwargs...)
    scale_separate = [
        KPM.optical_cond2(mu, NCscale, 0.3; scale_kwargs...) for mu in (easy, hard)
    ]
    component_errors = abs.(scale_batch .- scale_separate) ./ abs.(scale_separate)
    @info "unequal-scale optical batch component errors" component_errors
    @test all(component_errors .< 1e-8)

    identity_kernel = (n, N) -> 1.0
    mu1_high = zeros(ComplexF64, NCscale)
    mu1_high[end] = 1
    beta_high, xF_high = 12.0, 0.2
    value_high = KPM.optical_cond1(
        mu1_high,
        NCscale,
        0.3;
        E_f = xF_high,
        beta = beta_high,
        kernel = identity_kernel,
        quad_rtol = 1e-8,
    )
    n_high = NCscale - 1
    lambda_high = first(quadgk(
        theta -> cos(n_high * theta) / pi /
                 (1 + exp(beta_high * (cos(theta) - xF_high))),
        0.0,
        pi;
        rtol = 1e-13,
        atol = 1e-18,
        order = 63,
    ))
    reference_high = (-im / 0.3) * 2lambda_high
    high_moment_error = abs(value_high - reference_high) / abs(reference_high)
    @info "finite-temperature high-moment optical_cond1 error" high_moment_error
    @test high_moment_error < 1e-8

    mu_dense = [tr(Matrix(Jx) * S * Matrix(Jy) * T) / D
        for T in _chebyshev_matrices(H_norm, NCo), S in _chebyshev_matrices(H_norm, NCo)]
    kh = [_ref_jackson(n, NCo) * _ref_hn(n) for n = 0:(NCo-1)]
    delta = _ref_delta(x, NCo)
    gR = _ref_green(x + omega, NCo, 0.03, :R)
    gA = _ref_green(x - omega, NCo, 0.03, :A)
    ref = sum(mu_dense .* (kh * transpose(kh)) .*
        (gR * transpose(delta) + delta * transpose(gA)))
    @test KPM.d_optical_cond2(mu_xy, NCo, omega, x; lambda = 0.03) ≈ ref rtol = 1e-10

    # grid methods agree with the scalar ones (rtol contract: the moments are
    # recomputed per call and device reductions may reorder the sums)
    mu1_grid = KPM.kpm_1d_current(H_norm, Jxx, NCo, D, D; psi_in = psi)
    grid1, values1 = KPM.d_optical_cond1(mu1_grid, NCo; N_int = 7, e_range = [-0.8, 0.8])
    @test values1 ≈ [KPM.d_optical_cond1(mu1_grid, NCo, x) for x in grid1] rtol = 1e-12
    grid2, values2 = KPM.d_optical_cond2(mu_xy, NCo, omega;
        lambda = 0.03, N_int = 7, e_range = [-0.8, 0.8])
    @test values2 ≈ [KPM.d_optical_cond2(mu_xy, NCo, omega, x; lambda = 0.03)
        for x in grid2] rtol = 1e-12
end

@testset "optical input and quadrature errors" begin
    mu = zeros(ComplexF64, 8, 8)
    @test_throws ArgumentError KPM.optical_cond2(mu, 8, 0.0)
    @test isfinite(KPM.d_optical_cond2(mu, 8, 0.0, 0.0))
    @test_throws ArgumentError KPM.optical_cond2(mu, 8, 0.3; lambda = -0.1)
    @test_throws ErrorException KPM.optical_cond2(
        ones(ComplexF64, 8, 8), 8, 0.3; maxevals = 5)
    zero_component_error = try
        KPM.optical_cond2(
            (ones(ComplexF64, 8, 8), zeros(ComplexF64, 8, 8)),
            8,
            0.3,
        )
        nothing
    catch err
        err
    end
    @test zero_component_error isa ErrorException
    @test occursin("quad_atol is required", sprint(showerror, zero_component_error))
end

@testset "typed optical layer and physical anchors" begin
    Hbig, Jxbig, Jybig, areabig =
        haldane_model(12, 12; t = 1.0, t2 = 0.2, ϕ = pi / 2, m = 0.0)
    Dbig = size(Hbig, 1)
    hbig = KPM.rescale(Hbig; center = true)
    psibig = Matrix{ComplexF64}(I, Dbig, Dbig)
    NCbig = 256
    mxy = KPM.cond_moments(hbig, Jxbig, Jybig; NC = NCbig, psi_in = copy(psibig))
    mxx = KPM.cond_moments(hbig, Jxbig, Jxbig; NC = NCbig, psi_in = copy(psibig))
    Jxxbig = _second_current(Hbig, Jxbig)
    m1xx = KPM.current_moments(hbig, Jxxbig, NCbig, Dbig; psi_in = copy(psibig))

    @test m1xx isa KPM.CurrentMoments
    @test KPM.nc(m1xx) == NCbig
    @test m1xx.mu[1] ≈ tr(Jxxbig) / Dbig rtol = 1e-12 atol = 1e-14
    @test occursin("CurrentMoments", string(m1xx))

    omega_api = 0.35hbig.a
    scalar = KPM.optical_cond(
        mxy,
        omega_api;
        area = areabig,
        Ef = hbig.b,
        quad_atol = 1e-12,
    )
    vector = KPM.optical_cond(
        mxy,
        [omega_api, 0.4hbig.a];
        area = areabig,
        Ef = hbig.b,
        quad_atol = 1e-12,
    )
    batch = KPM.optical_cond(
        (mxy, mxy),
        omega_api;
        area = areabig,
        Ef = hbig.b,
        quad_atol = 1e-12,
    )
    @test scalar isa ComplexF64
    @test vector isa Vector{ComplexF64}
    @test vector[1] ≈ scalar rtol = 1e-13 atol = 1e-13
    @test batch ≈ [scalar, scalar] rtol = 1e-13 atol = 1e-13

    mixed_batch = KPM.optical_cond(
        (mxx, mxy),
        omega_api;
        area = areabig,
        Ef = hbig.b,
        m1s = (m1xx, nothing),
        quad_atol = 1e-12,
    )
    mixed_separate = [
        KPM.optical_cond(
            mxx,
            omega_api;
            area = areabig,
            Ef = hbig.b,
            m1 = m1xx,
            quad_atol = 1e-12,
        ),
        KPM.optical_cond(
            mxy,
            omega_api;
            area = areabig,
            Ef = hbig.b,
            quad_atol = 1e-12,
        ),
    ]
    @test mixed_batch ≈ mixed_separate rtol = 1e-10 atol = 1e-12

    beta_physical = 3.2
    omega_beta = 0.35hbig.a
    Ef_beta = hbig.b + 0.1hbig.a
    lambda_beta = 0.03hbig.a
    typed_beta = KPM.optical_cond(
        mxy,
        omega_beta;
        area = areabig,
        Ef = Ef_beta,
        beta = beta_physical,
        lambda = lambda_beta,
        quad_rtol = 2e-10,
        quad_atol = 2e-12,
    )
    bare_beta = KPM.optical_cond2(
        mxy.mu,
        NCbig,
        omega_beta / hbig.a;
        E_f = (Ef_beta - hbig.b) / hbig.a,
        beta = beta_physical * hbig.a,
        lambda = lambda_beta / hbig.a,
        quad_rtol = 2e-10,
        quad_atol = 2e-12,
    )
    bare_beta *= 2pi * Dbig / (areabig * hbig.a^2)
    @test typed_beta ≈ bare_beta rtol = 1e-10 atol = 1e-12

    sigma(q) = real(KPM.optical_cond(
        mxy,
        q * hbig.a;
        area = areabig,
        Ef = hbig.b,
        quad_atol = 1e-12,
    ))
    sigma_half = sigma(5e-4)
    sigma_h = sigma(1e-3)
    sigma_2h = sigma(2e-3)
    sigma_dc = (4sigma_h - sigma_2h) / 3
    sigma_dc_half = (4sigma_half - sigma_h) / 3
    sigma_bastin = KPM.kubo_bastin_cond(mxy, hbig.b; area = areabig)
    dc_relative_error = abs(sigma_dc - sigma_bastin) / abs(sigma_bastin)
    dc_step_stability = abs(sigma_dc_half - sigma_dc) / abs(sigma_dc)
    @info "optical DC Richardson anchor" sigma_bastin sigma_dc sigma_dc_half dc_relative_error dc_step_stability
    @test dc_step_stability < 1e-8
    @test sigma_dc ≈ sigma_bastin rtol = 1e-6

    # This anchor uses a 12x12 torus rather than the 6x6 torus in
    # kubo_bastin_test.jl; the same generous 5e-2 KPM-to-C tolerance is used.
    C_fhs = chern_number_fhs(
        haldane_bloch(; t = 1.0, t2 = 0.2, ϕ = pi / 2, m = 0.0);
        Nk = 30,
    )
    @test C_fhs ≈ 1.0 atol = 1e-10
    @test sigma_bastin ≈ C_fhs atol = 5e-2

    frequencies = [hbig.a, 1.5hbig.a]
    spectrum = eigvals(Hermitian(Matrix(Hbig)))
    gap = 2minimum(abs.(spectrum .- hbig.b))
    bandwidth = maximum(spectrum) - minimum(spectrum)
    @test all((gap .< frequencies) .& (frequencies .< bandwidth))
    sigma_xx = KPM.optical_cond(
        mxx,
        frequencies;
        area = areabig,
        Ef = hbig.b,
        m1 = m1xx,
        quad_atol = 1e-12,
    )
    @test real.(sigma_xx) ≈ [1.24501644500, 0.07108598834] rtol = 2e-8
    @test all(real.(sigma_xx) .> 0)

    omega_tilde = 1e-3
    dia = real(
        im * omega_tilde * KPM.optical_cond1(
            m1xx.mu,
            NCbig,
            omega_tilde;
            E_f = 0.0,
            quad_atol = 1e-12,
        ),
    )
    para = real(
        im * omega_tilde * KPM.optical_cond2(
            mxx.mu,
            NCbig,
            omega_tilde;
            E_f = 0.0,
            quad_atol = 1e-12,
        ),
    ) / hbig.a
    # Measured |dia + Re(para)/a|/|dia| = 5.4316e-4 at NC=256.
    gauge_residual = abs(dia + para) / abs(dia)
    @test gauge_residual < 2e-3

    Hsmall, Jxsmall, _, _ =
        haldane_model(3, 3; t = 1.0, t2 = 0.2, ϕ = pi / 2, m = 0.0)
    Dsmall = size(Hsmall, 1)
    hsmall = KPM.rescale(Hsmall; center = true)
    NCsmall = 64
    psismall = Matrix{ComplexF64}(I, Dsmall, Dsmall)
    muxx_small = KPM.kpm_2d(
        hsmall.H,
        Jxsmall,
        Jxsmall,
        NCsmall,
        Dsmall,
        Dsmall;
        psi_in = psismall,
    )
    omega_small = 0.5
    spectrum_small = eigvals(Hermitian(Matrix(Hsmall)))
    physical_bandwidth = maximum(spectrum_small) - minimum(spectrum_small)
    lambda_small = 0.02physical_bandwidth / hsmall.a
    value_small = KPM.optical_cond2(
        muxx_small,
        NCsmall,
        omega_small;
        E_f = 0.0,
        lambda = lambda_small,
        quad_atol = 1e-12,
    )
    ref_small = _lehmann_optical_reference(
        hsmall.H,
        Jxsmall,
        Jxsmall,
        omega_small,
        lambda_small,
        0.0,
    )
    # At NC=64 the Jackson-smeared, compact-support KPM spectrum differs from
    # the exact Lorentzian tails; the measured relative difference is 9.95%.
    @test value_small ≈ ref_small rtol = 0.1

    bad_m1 = KPM.CurrentMoments(m1xx.mu, m1xx.a + 0.1, m1xx.b, m1xx.NH, m1xx.NR)
    @test_throws ArgumentError KPM.optical_cond(
        mxx,
        omega_api;
        area = areabig,
        m1 = bad_m1,
    )
    @test_throws ArgumentError KPM.optical_cond(mxy, 0.0; area = areabig)
    @test_throws ArgumentError KPM.optical_cond(
        mxy,
        omega_api;
        area = areabig,
        lambda = -0.1,
    )
    @test_throws ArgumentError KPM.optical_cond(mxy, omega_api; area = 0.0)
end
