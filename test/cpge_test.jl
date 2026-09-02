using Test
using LinearAlgebra
using SparseArrays
using Random
using QuadGK
using KPM

# Model geometry is explicit user data.  The three bond currents are built
# from the unrescaled Hamiltonian and the helix displacements.
function helix_cluster(; t = 1.0, t2 = 0.45, phi = 0.7, w = 0.3)
    N = 6
    pos = [
        [cos(2pi * (j - 1) / 3), sin(2pi * (j - 1) / 3), 0.4 * (j - 1)] for
        j = 1:N
    ]
    Hd = zeros(ComplexF64, N, N)
    for j = 1:(N-1)
        Hd[j+1, j] += -t
        Hd[j, j+1] += -t
    end
    for j = 1:(N-2)
        Hd[j+2, j] += t2 * cis(phi)
        Hd[j, j+2] += t2 * cis(-phi)
    end
    for j = 1:N
        Hd[j, j] = w * (-1)^j
    end
    @assert Hd ≈ Hd'
    J = [
        sparse([Hd[i, j] * (pos[i][alpha] - pos[j][alpha]) for i = 1:N, j = 1:N]) for
        alpha = 1:3
    ]
    return sparse(Hd), J, pos
end

function cpge_chebyshev_matrices(A, NC)
    D = size(A, 1)
    T = Vector{Matrix{ComplexF64}}(undef, NC)
    T[1] = Matrix{ComplexF64}(I, D, D)
    NC >= 2 && (T[2] = Matrix{ComplexF64}(A))
    for n = 3:NC
        T[n] = 2 * Matrix(A) * T[n-1] - T[n-2]
    end
    return T
end

function cpge_chebyshev(x, NC)
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

cpge_ref_hn(n) = n == 0 ? 1.0 : 2.0
function cpge_ref_jackson(n, NC)
    alpha = pi / (NC + 1)
    return ((NC + 1 - n) * cos(n * alpha) + sin(n * alpha) * cot(alpha)) / (NC + 1)
end

cpge_ref_delta(x, NC) = cpge_chebyshev(x, NC) ./ (pi * sqrt(1 - x^2))

function cpge_ref_green(z::Real, NC, lambda, branch; delta = 0.0)
    if delta > 0 && 1 - delta < abs(z) < 1 + delta
        return zeros(ComplexF64, NC)
    end
    if iszero(lambda)
        if abs(z) < 1
            phi = acos(z)
            prefactor = branch === :R ? -im / sin(phi) : im / sin(phi)
            sign = branch === :R ? -1 : 1
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
        return ComplexF64[
            -im * exp(-im * n * acos(w)) / sqrt(1 - w^2) for n = 0:(NC-1)
        ]
    end
    return ComplexF64[
        im * exp(im * n * acos(w)) / sqrt(1 - w^2) for n = 0:(NC-1)
    ]
end

function cpge_ref_fermi(x, xF, beta)
    isinf(beta) && return ((x < xF) + (x <= xF)) / 2
    y = beta * (x - xF)
    y > 0 && return exp(-y) / (1 + exp(-y))
    return inv(1 + exp(y))
end

function cpge_unique_points(points; tol = 1e-13)
    sort!(points)
    out = Float64[]
    for point in points
        (isempty(out) || abs(point - out[end]) > tol) && push!(out, point)
    end
    return out
end

# Independent adaptive integration: its coefficients, breakpoint construction,
# endpoint maps, tolerances, and quadrature order do not call package helpers.
function cpge_ref_integral(
    fun,
    shifts,
    xF,
    beta,
    lambda;
    delta = 0.0,
    rtol = 2e-10,
    atol = 2e-12,
    order = 15,
)
    lo = isinf(beta) ? acos(clamp(xF, -1, 1)) : 0.0
    points = Float64[lo, pi]
    singular = Float64[]
    for shift in shifts, edge in (-1.0, 1.0)
        xstar = edge - shift
        if -1 < xstar < 1
            theta = acos(xstar)
            if lo < theta < pi
                push!(points, theta)
                iszero(lambda) && iszero(delta) && push!(singular, theta)
            end
        end
    end
    if delta > 0
        for shift in shifts, radius in (1 - delta, 1 + delta)
            radius < 0 && continue
            for edge in (-radius, radius)
                xstar = edge - shift
                -1 < xstar < 1 || continue
                theta = acos(xstar)
                lo < theta < pi && push!(points, theta)
            end
        end
    end
    !isinf(beta) && beta > 50 && -1 < xF < 1 && push!(points, acos(xF))
    points = cpge_unique_points(points)
    singular = cpge_unique_points(singular)
    isedge(theta) = any(s -> abs(theta - s) < 1e-12, singular)
    node(theta) = sin(theta) * cpge_ref_fermi(cos(theta), xF, beta) * fun(cos(theta))

    function piece(a, b, leftedge, rightedge)
        if leftedge && rightedge
            mid = (a + b) / 2
            return piece(a, mid, true, false) + piece(mid, b, false, true)
        elseif leftedge
            width = b - a
            return first(
                quadgk(
                    u -> node(a + width * u^2) * (2width * u),
                    0.0,
                    1.0;
                    rtol = rtol,
                    atol = atol,
                    order = order,
                    norm = LinearAlgebra.norm,
                ),
            )
        elseif rightedge
            width = b - a
            return first(
                quadgk(
                    u -> node(b - width * u^2) * (2width * u),
                    0.0,
                    1.0;
                    rtol = rtol,
                    atol = atol,
                    order = order,
                    norm = LinearAlgebra.norm,
                ),
            )
        end
        return first(
            quadgk(node, a, b; rtol = rtol, atol = atol, order = order, norm = LinearAlgebra.norm),
        )
    end

    values = [
        piece(points[k], points[k+1], isedge(points[k]), isedge(points[k+1])) for
        k = 1:(length(points)-1)
    ]
    return reduce(+, values)
end

function cpge_eigen_integrand(F, A, B, C, T, kh, NC, omega1, omega2, x, lambda, delta)
    Omega = omega1 + omega2
    deltaK = transpose(kh .* cpge_ref_delta(x, NC)) * T
    gR_Omega = transpose(kh .* cpge_ref_green(x + Omega, NC, lambda, :R; delta = delta)) * T
    gR_omega2 = transpose(kh .* cpge_ref_green(x + omega2, NC, lambda, :R; delta = delta)) * T
    gR_omega1 = transpose(kh .* cpge_ref_green(x + omega1, NC, lambda, :R; delta = delta)) * T
    gA_momega2 = transpose(kh .* cpge_ref_green(x - omega2, NC, lambda, :A; delta = delta)) * T
    gA_momega1 = transpose(kh .* cpge_ref_green(x - omega1, NC, lambda, :A; delta = delta)) * T
    gA_mOmega = transpose(kh .* cpge_ref_green(x - Omega, NC, lambda, :A; delta = delta)) * T
    acc = 0.0 + 0.0im
    # Tr[Jalpha*T_n*Jbeta*T_m*Jgamma*T_p]
    # = sum_abc Jalpha_ab*T_n(E_b)*Jbeta_bc*T_m(E_c)*Jgamma_ca*T_p(E_a).
    # Thus RR evaluates its (n,m,p) coefficient sums at (E_b,E_c,E_a),
    # and the same eigenvalue pattern gives the RA and AA terms below.
    for a in eachindex(F.values), b in eachindex(F.values), c in eachindex(F.values)
        acc += A[a, b] * B[b, c] * C[c, a] * (
            gR_Omega[b] * gR_omega2[c] * deltaK[a] +
            gR_omega1[b] * deltaK[c] * gA_momega2[a] +
            deltaK[b] * gA_momega1[c] * gA_mOmega[a]
        )
    end
    return acc / length(F.values)
end

function cpge_eigen_reference(
    Hnorm,
    Jalpha,
    Jbeta,
    Jgamma,
    NC,
    omega1,
    omega2;
    xF,
    beta,
    lambda,
    delta = 0.0,
    rtol = 2e-10,
    atol = 2e-12,
    order = 15,
)
    F = eigen(Hermitian(Matrix(Hnorm)))
    A = F.vectors' * Matrix(Jalpha) * F.vectors
    B = F.vectors' * Matrix(Jbeta) * F.vectors
    C = F.vectors' * Matrix(Jgamma) * F.vectors
    T = hcat((cpge_chebyshev(E, NC) for E in F.values)...)
    kh = [cpge_ref_jackson(n, NC) * cpge_ref_hn(n) for n = 0:(NC-1)]
    shifts = (
        omega1 + omega2,
        omega2,
        omega1,
        -omega2,
        -omega1,
        -omega1 - omega2,
    )
    integrand(x) = cpge_eigen_integrand(
        F,
        A,
        B,
        C,
        T,
        kh,
        NC,
        omega1,
        omega2,
        x,
        lambda,
        delta,
    )
    return im * cpge_ref_integral(
        integrand,
        shifts,
        xF,
        beta,
        lambda;
        delta = delta,
        rtol = rtol,
        atol = atol,
        order = order,
    )
end

function cpge_lambda_table(NC, omega1, omega2; xF, beta, lambda)
    Omega = omega1 + omega2
    shifts = (Omega, omega2, omega1, -omega2, -omega1, -Omega)
    function integrand(x)
        Delta = cpge_ref_delta(x, NC)
        rO = cpge_ref_green(x + Omega, NC, lambda, :R)
        r2 = cpge_ref_green(x + omega2, NC, lambda, :R)
        r1 = cpge_ref_green(x + omega1, NC, lambda, :R)
        a2 = cpge_ref_green(x - omega2, NC, lambda, :A)
        a1 = cpge_ref_green(x - omega1, NC, lambda, :A)
        aO = cpge_ref_green(x - Omega, NC, lambda, :A)
        return [
            rO[n] * r2[m] * Delta[p] + r1[n] * Delta[m] * a2[p] +
            Delta[n] * a1[m] * aO[p] for n = 1:NC, m = 1:NC, p = 1:NC
        ]
    end
    return cpge_ref_integral(
        integrand,
        shifts,
        xF,
        beta,
        lambda;
        rtol = 2e-10,
        atol = 2e-12,
    )
end

function dense_rescale(H; eps = 0.1)
    eigenvalues = eigvals(Hermitian(Matrix(H)))
    b = (maximum(eigenvalues) + minimum(eigenvalues)) / 2
    a = (maximum(eigenvalues) - minimum(eigenvalues)) / (2 - eps)
    return a, b, sparse((Matrix(H) - b * I) / a)
end

H, J, pos = helix_cluster()
D = size(H, 1)
a, b, H_norm = dense_rescale(H)
psi = Matrix{ComplexF64}(I, D, D)

@testset "kpm_3d moments vs dense Chebyshev matrices" begin
    NC = 8
    Tmat = cpge_chebyshev_matrices(H_norm, NC)
    @test maximum(abs, eigvals(Hermitian(Matrix(H_norm)))) < 1

    mu3 = KPM.kpm_3d(H_norm, J[1], J[2], J[3], NC, D, D; psi_in = psi)
    @test size(mu3) == (NC, NC, NC)

    # Implemented layout (cyclically written with Jalpha last):
    # mu[n,m,p] = Tr[T_m(Hnorm)*Jbeta*T_n(Hnorm)*Jgamma*T_p(Hnorm)*Jalpha]/D.
    mu_ref = [
        tr(Tmat[m] * J[2] * Tmat[n] * J[3] * Tmat[p] * J[1]) / D for n = 1:NC,
        m = 1:NC, p = 1:NC
    ]
    @test mu3 ≈ mu_ref rtol = 1e-12

    mu_ip = zeros(ComplexF64, NC, NC, NC)
    KPM.kpm_3d!(H_norm, J[1], J[2], J[3], NC, D, D, mu_ip, psi, psi)
    @test mu_ip ≈ mu3 rtol = 1e-12

    rng = Xoshiro(20260901)
    psi_r = KPM.random_phase_vectors(rng, D, 1024)
    mu_r = KPM.kpm_3d(H_norm, J[1], J[2], J[3], NC, 1024, D; psi_in = psi_r)
    @test maximum(abs, mu_r .- mu_ref) < 0.15 * maximum(abs, mu_ref)
end

@testset "same-kernel eigenbasis CPGE reference" begin
    NC = 12
    mu3 = KPM.kpm_3d(H_norm, J[1], J[2], J[3], NC, D, D; psi_in = psi)
    for (omega1, omega2) in ((0.31, -0.29), (0.3, 0.2)),
        (xF, beta) in ((0.15, Inf), (0.15, 12.0)), lambda in (0.0, 0.03)
        ref = cpge_eigen_reference(
            H_norm,
            J[1],
            J[2],
            J[3],
            NC,
            omega1,
            omega2;
            xF = xF,
            beta = beta,
            lambda = lambda,
        )
        value = KPM.cpge(
            mu3,
            NC,
            omega1,
            omega2;
            E_f = xF,
            beta = beta,
            lambda = lambda,
            quad_rtol = 1e-8,
            quad_atol = 2e-11,
        )
        @test value ≈ ref rtol = 1e-8 atol = 2e-9
    end

    loose = cpge_eigen_reference(
        H_norm,
        J[1],
        J[2],
        J[3],
        NC,
        0.3,
        0.2;
        xF = 0.15,
        beta = 12.0,
        lambda = 0.03,
        rtol = 2e-10,
        atol = 2e-12,
        order = 15,
    )
    tight = cpge_eigen_reference(
        H_norm,
        J[1],
        J[2],
        J[3],
        NC,
        0.3,
        0.2;
        xF = 0.15,
        beta = 12.0,
        lambda = 0.03,
        rtol = 2e-12,
        atol = 2e-14,
        order = 21,
    )
    @test loose ≈ tight rtol = 1e-10 atol = 1e-12
end

@testset "paper-tensor index orientation" begin
    NC = 6
    omega1, omega2 = 0.3, 0.2
    Tmat = cpge_chebyshev_matrices(H_norm, NC)
    Gamma = [
        tr(J[1] * Tmat[n] * J[2] * Tmat[m] * J[3] * Tmat[p]) / D for n = 1:NC,
        m = 1:NC, p = 1:NC
    ]
    Lambda = cpge_lambda_table(NC, omega1, omega2; xF = 0.15, beta = 12.0, lambda = 0.03)
    kh = [cpge_ref_jackson(n, NC) * cpge_ref_hn(n) for n = 0:(NC-1)]
    kernel3 = reshape(kh, NC, 1, 1) .* reshape(kh, 1, NC, 1) .* reshape(kh, 1, 1, NC)
    direct = im * sum(Gamma .* kernel3 .* Lambda)

    mu3 = KPM.kpm_3d(H_norm, J[1], J[2], J[3], NC, D, D; psi_in = psi)
    value = KPM.cpge(
        mu3,
        NC,
        omega1,
        omega2;
        E_f = 0.15,
        beta = 12.0,
        lambda = 0.03,
        quad_rtol = 1e-9,
        quad_atol = 2e-11,
    )
    @test value ≈ direct rtol = 1e-8 atol = 2e-10

    mu_permuted = KPM.kpm_3d(H_norm, J[1], J[3], J[2], NC, D, D; psi_in = psi)
    permuted = KPM.cpge(
        mu_permuted,
        NC,
        omega1,
        omega2;
        E_f = 0.15,
        beta = 12.0,
        lambda = 0.03,
        quad_rtol = 1e-9,
        quad_atol = 2e-11,
    )
    @test !isapprox(permuted, direct; rtol = 1e-3, atol = 1e-8)
end

@testset "Omega = 0 regularization" begin
    NC = 12
    omega1, omega2 = 0.31, -0.31
    mu3 = KPM.kpm_3d(H_norm, J[1], J[2], J[3], NC, D, D; psi_in = psi)
    err = try
        KPM.cpge(mu3, NC, omega1, omega2)
        nothing
    catch caught
        caught
    end
    @test err isa ArgumentError
    @test occursin("lambda > 0", sprint(showerror, err))
    @test occursin("delta > 0", sprint(showerror, err))
    @test occursin("Omega != 0", sprint(showerror, err))

    lambda = 0.05
    value_lambda = KPM.cpge(
        mu3,
        NC,
        omega1,
        omega2;
        E_f = 0.15,
        lambda = lambda,
        quad_rtol = 1e-9,
        quad_atol = 2e-11,
    )
    ref_lambda = cpge_eigen_reference(
        H_norm,
        J[1],
        J[2],
        J[3],
        NC,
        omega1,
        omega2;
        xF = 0.15,
        beta = Inf,
        lambda = lambda,
    )
    @test isfinite(value_lambda)
    @test value_lambda ≈ ref_lambda rtol = 1e-8 atol = 2e-9

    value_delta4 = KPM.cpge(
        mu3,
        NC,
        omega1,
        omega2;
        E_f = 0.15,
        delta = 1e-4,
        quad_rtol = 2e-8,
        quad_atol = 2e-10,
    )
    value_delta5 = KPM.cpge(
        mu3,
        NC,
        omega1,
        omega2;
        E_f = 0.15,
        delta = 1e-5,
        quad_rtol = 2e-8,
        quad_atol = 2e-10,
    )
    delta_sensitivity = abs(value_delta4 - value_delta5) / abs(value_delta5)
    @info "CPGE Omega=0 delta sensitivity" value_delta4 value_delta5 delta_sensitivity
    @test isfinite(value_delta4)
    @test isfinite(value_delta5)
    # Measured at 3.32% for this finite-NC helix: the edge-exclusion dependence
    # is weak compared with the response, but is not a quadrature-level effect.
    @test delta_sensitivity < 5e-2
end

# A finite-N exact time-reversal identity for the requested antisymmetrized
# frequency combination was not established cleanly: transpose symmetry gives
# Gamma^{abc}_{nmp} = -Gamma^{acb}_{pmn}, which does not by itself exchange
# alpha and beta in y.  We therefore do not encode a potentially false TR test.

@testset "resolved CPGE integrand" begin
    NC = 12
    omega1, omega2, x, lambda = 0.3, 0.2, -0.23, 0.03
    mu3 = KPM.kpm_3d(H_norm, J[1], J[2], J[3], NC, D, D; psi_in = psi)
    F = eigen(Hermitian(Matrix(H_norm)))
    A = F.vectors' * Matrix(J[1]) * F.vectors
    B = F.vectors' * Matrix(J[2]) * F.vectors
    C = F.vectors' * Matrix(J[3]) * F.vectors
    T = hcat((cpge_chebyshev(E, NC) for E in F.values)...)
    kh = [cpge_ref_jackson(n, NC) * cpge_ref_hn(n) for n = 0:(NC-1)]
    ref = cpge_eigen_integrand(F, A, B, C, T, kh, NC, omega1, omega2, x, lambda, 0.0)
    @test KPM.d_cpge(mu3, NC, omega1, omega2, x; lambda = lambda) ≈ ref rtol = 1e-10 atol =
        1e-12

    grid, values = KPM.d_cpge(
        mu3,
        NC,
        omega1,
        omega2;
        lambda = lambda,
        N_int = 7,
        e_range = [-0.8, 0.8],
    )
    @test values ≈ [
        KPM.d_cpge(mu3, NC, omega1, omega2, node; lambda = lambda) for node in grid
    ] rtol = 1e-12
end

@testset "CPGE input and quadrature errors" begin
    mu3 = ones(ComplexF64, 5, 5, 5)
    @test_throws ArgumentError KPM.cpge(mu3, 5, 0.3, 0.2; lambda = -0.1)
    @test_throws ArgumentError KPM.cpge(mu3, 5, 0.3, 0.2; delta = -0.1)
    @test_throws ErrorException KPM.cpge(mu3, 5, 0.3, 0.2; maxevals = 5)
end
