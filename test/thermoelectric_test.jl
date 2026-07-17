using Test
using LinearAlgebra
using SparseArrays
using Random
using Logging
using KPM

isdefined(@__MODULE__, :EDReference) || include("ed_reference.jl")
using .EDReference

# Thermoelectric reconstruction pins: Kubo–Greenwood normalization, CTKG
# quadrature, particle-hole/sign conventions, insulating-window guards, and
# the tensor left-solve policy.

# Exact deterministic moments mu2D[n,m] = Tr[J T_m(Ht) J T_n(Ht)]/D in the
# eigenbasis of Ht: with C[i,n] = T_{n-1}(lam_i) and W_ij = Jt_ij*Jt_ji,
# mu2D = transpose(C) * W * C.
function exact_mu2D(H::Matrix{ComplexF64}, J::Matrix{ComplexF64}, a, b, NC)
    N = size(H, 1)
    F = eigen(Hermitian((H - b * I) / a))
    lam, V = F.values, F.vectors
    Jt = V' * J * V
    W = Jt .* transpose(Jt)
    theta = acos.(clamp.(lam, -1.0, 1.0))
    C = [cos((n - 1) * theta[i]) for i in 1:N, n in 1:NC]
    return (transpose(C) * W * C) ./ N
end

rng = Xoshiro(7)
N = 10
A0 = randn(rng, ComplexF64, N, N)
H = Matrix(Hermitian(A0 + A0'))
r = randn(rng, N)
J = ComplexF64[H[i, j] * (r[i] - r[j]) for i in 1:N, j in 1:N]
ev = eigvals(Hermitian(H))
margin = 0.05 * (ev[end] - ev[1])
a = (ev[end] - ev[1] + 2margin) / 2
b = (ev[end] + ev[1]) / 2
NC = 64
mu2D = exact_mu2D(H, J, a, b, NC)
vol = 3.7
gJ = 2.0
E_test = b .+ a .* [-0.8, -0.35, 0.0, 0.22, 0.6, 0.93]

@testset "transport_distribution: operator-level normalization vs direct contraction" begin
    @test norm(J + J') < 1e-12

    Ht = (H - b * I) / a
    F = eigen(Hermitian(Ht))
    lam, V = F.values, F.vectors
    Tn = [V * Diagonal(cos.(n .* acos.(lam))) * V' for n in 0:(NC - 1)]
    g = [KPM.JacksonKernel(n, NC) for n in 0:(NC - 1)]
    h = [n == 0 ? 1.0 : 2.0 for n in 0:(NC - 1)]
    function deltaK(E)
        x = (E - b) / a
        return sum(h[n + 1] * g[n + 1] * cos(n * acos(x)) * Tn[n + 1]
                   for n in 0:(NC - 1)) / (a * pi * sqrt(1 - x^2))
    end
    sig_direct = [-(2 * pi^2 * gJ / vol) * real(tr(J * deltaK(E) * J * deltaK(E)))
                  for E in E_test]
    sig_kpm = KPM.transport_distribution(mu2D, a, E_test; b=b, NH=N,
                                         volume=vol, g_J=gJ, NC=NC)

    # exact-moment vs direct-contraction agreement is ~5e-11 elementwise
    # (dominated by the 1/(1-x^2)^2 amplification near the band edge); any
    # normalization or sign error would be O(1), so 1e-9 pins every factor
    # while staying robust to BLAS-dependent roundoff
    @test sig_kpm ≈ sig_direct rtol=1e-9
    @test all(sig_kpm .>= 0)
    @test KPM.transport_distribution(mu2D, a, E_test[2]; b=b, NH=N,
                                     volume=vol, g_J=gJ, NC=NC) == sig_kpm[2]
    @test KPM.transport_distribution(mu2D, a, E_test; b=b, NH=N, volume=2vol,
                                     g_J=gJ, NC=NC) ≈ sig_kpm ./ 2
    @test KPM.transport_distribution(mu2D, a, b + 1.5a; b=b, NH=N,
                                     volume=vol, g_J=gJ, NC=NC) == 0.0
end

@testset "transport_integrals: quadrature accuracy, rigid shift, argument checks" begin
    beta = 4.0
    mu_chem = b + 0.1a
    kwargs = (; b=b, NH=N, volume=vol, g_J=gJ, NC=NC)
    res = Logging.with_logger(Logging.NullLogger()) do
        KPM.transport_integrals(mu2D, a, mu_chem; beta=beta, kwargs...)
    end

    Eg = range(b - a * (1 - 1e-3), b + a * (1 - 1e-3); length=200_001)
    Eg_vec = collect(Eg)
    sg = KPM.transport_distribution(mu2D, a, Eg_vec; kwargs...)
    w = KPM.fermi_window(mu_chem, beta).(Eg_vec)
    dE = step(Eg)
    L0b = sum(w .* sg) * dE
    L1b = sum((Eg_vec .- mu_chem) .* w .* sg) * dE
    L2b = sum((Eg_vec .- mu_chem) .^ 2 .* w .* sg) * dE
    @test res.L0 ≈ L0b rtol=1e-5
    @test res.L1 ≈ L1b rtol=1e-5
    @test res.L2 ≈ L2b rtol=1e-5

    res_fine = Logging.with_logger(Logging.NullLogger()) do
        KPM.transport_integrals(mu2D, a, mu_chem; beta=beta,
                                quad_N=4 * 8 * NC, kwargs...)
    end
    @test res.L0 ≈ res_fine.L0 rtol=1e-10
    @test res.L1 ≈ res_fine.L1 rtol=1e-10
    @test res.L2 ≈ res_fine.L2 rtol=1e-10

    c = 1.3
    res_shift = Logging.with_logger(Logging.NullLogger()) do
        KPM.transport_integrals(mu2D, a, mu_chem + c; beta=beta,
                                b=b + c, NH=N, volume=vol, g_J=gJ, NC=NC)
    end
    @test res_shift.L0 ≈ res.L0 rtol=1e-10
    @test res_shift.L1 ≈ res.L1 rtol=1e-10
    @test res_shift.L2 ≈ res.L2 rtol=1e-10

    empty = Logging.with_logger(Logging.NullLogger()) do
        KPM.transport_integrals(mu2D, a, b + 3a; beta=1000.0, kwargs...)
    end
    @test empty.L0 == 0.0
    @test empty.L1 == 0.0
    @test empty.L2 == 0.0
    @test empty.neg_weight == 0.0
    @test_throws ArgumentError KPM.transport_integrals(mu2D, a, mu_chem; beta=Inf, kwargs...)
    @test_throws ArgumentError KPM.transport_integrals(mu2D, a, mu_chem; beta=0.0, kwargs...)
    @test_throws ArgumentError KPM.transport_integrals(mu2D, a, mu_chem; beta=-1.0, kwargs...)
    @test_throws ArgumentError KPM.thermoelectric(mu2D, a, mu_chem; beta=Inf, kwargs...)
end

@testset "particle-hole symmetry and Seebeck sign on a ring" begin
    Nr = 64
    Hr = spzeros(ComplexF64, Nr, Nr)
    Jr = spzeros(ComplexF64, Nr, Nr)
    for i in 1:Nr
        j = mod1(i + 1, Nr)
        Hr[i, j] = -1.0
        Hr[j, i] = -1.0
        Jr[i, j] = Hr[i, j] * (-1.0)
        Jr[j, i] = Hr[j, i] * (1.0)
    end
    ar, br, Hn = KPM.normalizeH(Hr; center=true)
    psi = Matrix{ComplexF64}(I, Nr, Nr)
    mur = KPM.kpm_2d(Hn, Jr, Jr, 64, Nr, Nr; psi_in=psi)

    r0 = KPM.thermoelectric(mur, ar, 0.0; beta=5.0, NH=Nr, volume=Float64(Nr), b=br)
    @test r0.L0 > 0 && isfinite(r0.L0)
    @test abs(r0.L1) < 1e-7
    @test abs(r0.S_over_kB_over_e) < 1e-8

    rp = KPM.thermoelectric(mur, ar, 1.0; beta=5.0, NH=Nr, volume=Float64(Nr), b=br)
    rm = KPM.thermoelectric(mur, ar, -1.0; beta=5.0, NH=Nr, volume=Float64(Nr), b=br)
    @test rp.S_over_kB_over_e ≈ -rm.S_over_kB_over_e rtol=1e-7
    @test rp.L0 ≈ rm.L0 rtol=1e-7
    @test rm.S_over_kB_over_e < 0
    @test KPM.seebeck_uVK(rm) == 86.17333262 * rm.S_over_kB_over_e
end

@testset "insulating thermal window yields NaN, not a huge unqualified S" begin
    Nd = 64
    Hd = spzeros(ComplexF64, Nd, Nd)
    Jd = spzeros(ComplexF64, Nd, Nd)
    for i in 1:(Nd - 1)
        t = isodd(i) ? -1.0 : -0.1
        Hd[i, i + 1] = t
        Hd[i + 1, i] = t
        Jd[i, i + 1] = t * (-1.0)
        Jd[i + 1, i] = t * (1.0)
    end
    ad, bd, Hdn = KPM.normalizeH(Hd; center=true)
    mud = KPM.kpm_2d(Hdn, Jd, Jd, 64, Nd, Nd;
                     psi_in=Matrix{ComplexF64}(I, Nd, Nd))
    dkwargs = (; beta=20.0, NH=Nd, volume=Float64(Nd), b=bd)

    # This off-center point remains thermally insulating at NC=64 while giving L1 != 0.
    rg = @test_logs (:warn, r"below-conductivity-floor") match_mode=:any begin
        KPM.thermoelectric(mud, ad, 0.1; dkwargs...)
    end
    @test isnan(rg.S_over_kB_over_e)
    @test 0 <= rg.L0 < 1e-4

    res = Logging.with_logger(Logging.NullLogger()) do
        KPM.thermoelectric(mud, ad, 0.1; sigma_min=0.0, dkwargs...)
    end
    @test isfinite(res.S_over_kB_over_e)
    rmet = Logging.with_logger(Logging.NullLogger()) do
        KPM.thermoelectric(mud, ad, -1.0; dkwargs...)
    end
    @test isfinite(rmet.S_over_kB_over_e)
    @test rmet.L0 > 1e-2
end

@testset "seebeck_solve: left-solve pinning and guard policy" begin
    beta = 4.0
    L0m = [2.0 0.3; -0.1 1.5]
    L1m = [0.4 -0.2; 0.7 0.1]
    S = KPM.seebeck_solve(L0m, L1m, beta)
    @test S ≈ -beta .* (L0m \ L1m) rtol=1e-12
    @test norm(S - (-beta .* (L1m / L0m))) > 1e-3
    @test norm(S - (-beta .* L1m ./ L0m)) > 1e-3

    @test KPM.seebeck_solve(2.0, 0.5, beta) == -beta * 0.5 / 2.0
    @test_logs (:warn, r"below-conductivity-floor") begin
        @test isnan(KPM.seebeck_solve(0.0, 0.5, beta))
    end
    @test isnan(Logging.with_logger(Logging.NullLogger()) do
        KPM.seebeck_solve(-1e-3, 0.5, beta)
    end)
    @test isnan(Logging.with_logger(Logging.NullLogger()) do
        KPM.seebeck_solve(NaN, 0.5, beta)
    end)
    @test isnan(Logging.with_logger(Logging.NullLogger()) do
        KPM.seebeck_solve(1e-300, 0.5, beta; sigma_min=1e-10)
    end)

    @test all(isnan, Logging.with_logger(Logging.NullLogger()) do
        KPM.seebeck_solve([-2.0 0.0; 0.0 -1.5], L1m, beta)
    end)
    @test all(isnan, Logging.with_logger(Logging.NullLogger()) do
        KPM.seebeck_solve(diagm([1e-15, 2e-15]), L1m, beta; sigma_min=1e-10)
    end)
    @test all(isnan, Logging.with_logger(Logging.NullLogger()) do
        KPM.seebeck_solve([1.0 1.0; 1.0 1.0], L1m, beta)
    end)
    @test_throws DimensionMismatch KPM.seebeck_solve(zeros(2, 3), zeros(2, 3), beta)
    @test_throws DimensionMismatch KPM.seebeck_solve(zeros(2, 2), zeros(3, 3), beta)
end

@testset "KG transport distribution consistent with the ED-pinned Kubo–Bastin" begin
    rng = Xoshiro(11)
    N = 8
    A0 = randn(rng, ComplexF64, N, N)
    H = Matrix(Hermitian(A0 + A0'))
    r = randn(rng, N)
    J = ComplexF64[H[i, j] * (r[i] - r[j]) for i in 1:N, j in 1:N]
    ev = eigvals(Hermitian(H))
    a = (ev[end] - ev[1]) / 2 * 1.1
    b = (ev[end] + ev[1]) / 2
    Ef = ev[4]
    for NC in (256, 2048)
        mu2D = exact_mu2D(H, J, a, b, NC)
        kg = KPM.transport_distribution(mu2D, a, Ef; b=b, NH=N, volume=1.0)
        kb = KPM.kubo_bastin_cond(mu2D, a, Ef; b=b, NH=N, area=1.0)
        @test kb / kg ≈ 1 atol=0.1
        @test kg > 0
    end
end

@testset "identity-probe kpm_2d vs recurrence-built kernel delta operators" begin
    rng_local = Xoshiro(101)
    N_local = 16
    A_local = randn(rng_local, ComplexF64, N_local, N_local)
    H_local = sparse(Matrix(Hermitian(A_local + A_local')))
    positions = randn(rng_local, N_local)
    J_local = sparse(ComplexF64[
        H_local[i, j] * (positions[i] - positions[j])
        for i in 1:N_local, j in 1:N_local
    ])
    a_local, b_local, H_norm = KPM.normalizeH(H_local; center=true)
    NC_local = 32
    psi_local = Matrix{ComplexF64}(I, N_local, N_local)
    mu_local = KPM.kpm_2d(H_norm, J_local, J_local, NC_local,
                          N_local, N_local; psi_in=psi_local)

    Ht = Matrix((H_local - b_local * I) / a_local)
    Tm = Vector{Matrix{ComplexF64}}(undef, NC_local)
    Tm[1] = Matrix{ComplexF64}(I, N_local, N_local)
    Tm[2] = Ht
    for n in 3:NC_local
        Tm[n] = 2 * Ht * Tm[n - 1] - Tm[n - 2]
    end
    kernel_weights = [KPM.JacksonKernel(n, NC_local) for n in 0:(NC_local - 1)]
    function recurrence_delta(E)
        x = (E - b_local) / a_local
        tx = Vector{Float64}(undef, NC_local)
        tx[1] = 1.0
        tx[2] = x
        for n in 3:NC_local
            tx[n] = 2x * tx[n - 1] - tx[n - 2]
        end
        delta = zeros(ComplexF64, N_local, N_local)
        for n in 1:NC_local
            hn = n == 1 ? 1.0 : 2.0
            delta .+= hn * kernel_weights[n] * tx[n] .* Tm[n]
        end
        return delta / (a_local * π * sqrt(1 - x^2))
    end

    volume_local = 2.9
    gJ_local = 2.0
    energies = b_local .+ a_local .* [-0.65, -0.35, 0.0, 0.3, 0.65]
    sigma_direct = [
        -(2π^2 * gJ_local / volume_local) *
        real(tr(J_local * recurrence_delta(E) * J_local * recurrence_delta(E)))
        for E in energies
    ]
    sigma_kpm = KPM.transport_distribution(
        mu_local, a_local, energies; b=b_local, NH=N_local,
        volume=volume_local, g_J=gJ_local, NC=NC_local)

    # The recurrence and KPM routes contract the identical finite Jackson
    # expansion; the observed maximum elementwise relative error is 4.3e-13.
    @test all(isapprox.(sigma_kpm, sigma_direct; rtol=1e-12))
end

@testset "disordered cubic thermoelectric integrals vs matched-broadening ED" begin
    H3, Jx3, Jy3, Jz3, volume3 = cubic_model(
        4; t=1.0, W=2.0, rng=Xoshiro(1))
    N3 = size(H3, 1)
    a3, b3, H3_norm = KPM.normalizeH(H3; center=true)
    NC3 = 128
    psi3 = Matrix{ComplexF64}(I, N3, N3)
    mu3 = KPM.kpm_2d(H3_norm, Jx3, Jx3, NC3, N3, N3; psi_in=psi3)
    lambda = 4.0
    lorentz = KPM.LorentzKernels(lambda)

    energies = b3 .+ a3 .* [-0.55, -0.25, 0.05, 0.3]
    sigma_kpm = KPM.transport_distribution(
        mu3, a3, energies; b=b3, NH=N3, volume=volume3,
        kernel=lorentz, NC=NC3)
    sigma_ed = [
        ed_transport_distribution(
            H3, Jx3, Jx3, volume3; E=E,
            eta=a3 * lambda * sqrt(1 - ((E - b3) / a3)^2) / NC3)
        for E in energies
    ]
    # The maximum observed pointwise relative error is below 5%; 15% is a
    # threefold broadening-match margin, not a stochastic tolerance.
    @test all(isapprox.(sigma_kpm, sigma_ed; rtol=0.15))

    mu_chem3 = b3 - 0.25a3
    beta3 = 10.0
    xmu = (mu_chem3 - b3) / a3
    eta = a3 * lambda * sqrt(1 - xmu^2) / NC3
    kpm_integrals = Logging.with_logger(Logging.NullLogger()) do
        KPM.transport_integrals(
            mu3, a3, mu_chem3; beta=beta3, b=b3, NH=N3,
            volume=volume3, kernel=lorentz, NC=NC3)
    end
    ed_integrals = ed_transport_integrals(
        H3, Jx3, Jx3, volume3; mu_chem=mu_chem3, beta=beta3, eta=eta)
    S_kpm = -beta3 * kpm_integrals.L1 / kpm_integrals.L0
    S_ed = -beta3 * ed_integrals.L1 / ed_integrals.L0
    # Relative errors are 0.42%, 0.42%, and 0.84% for L0, L1, and S;
    # 3% leaves more than a threefold deterministic broadening-match margin.
    @test kpm_integrals.L0 ≈ ed_integrals.L0 rtol=0.03
    @test kpm_integrals.L1 ≈ ed_integrals.L1 rtol=0.03
    @test S_kpm ≈ S_ed rtol=0.03
end

@testset "thermoelectric energy-scale covariance" begin
    scale = 2.7
    beta_local = 4.0
    mu_chem = b + 0.1a
    base_integrals, scaled_integrals, base_result, scaled_result =
        Logging.with_logger(Logging.NullLogger()) do
            base_i = KPM.transport_integrals(
                mu2D, a, mu_chem; beta=beta_local, b=b, NH=N,
                volume=vol, g_J=gJ, NC=NC)
            scaled_i = KPM.transport_integrals(
                mu2D .* scale^2, scale * a, scale * mu_chem;
                beta=beta_local / scale, b=scale * b, NH=N,
                volume=vol, g_J=gJ, NC=NC)
            base_r = KPM.thermoelectric(
                mu2D, a, mu_chem; beta=beta_local, b=b, NH=N,
                volume=vol, g_J=gJ, NC=NC)
            scaled_r = KPM.thermoelectric(
                mu2D .* scale^2, scale * a, scale * mu_chem;
                beta=beta_local / scale, b=scale * b, NH=N,
                volume=vol, g_J=gJ, NC=NC)
            base_i, scaled_i, base_r, scaled_r
        end
    @test scaled_integrals.L0 ≈ base_integrals.L0 rtol=1e-12
    @test scaled_integrals.L1 ≈ scale * base_integrals.L1 rtol=1e-12
    @test scaled_integrals.L2 ≈ scale^2 * base_integrals.L2 rtol=1e-12
    @test scaled_result.S_over_kB_over_e ≈ base_result.S_over_kB_over_e rtol=1e-12
end

@testset "low-temperature Mott relation" begin
    Nr = 64
    Hr = spzeros(ComplexF64, Nr, Nr)
    Jr = spzeros(ComplexF64, Nr, Nr)
    for i in 1:Nr
        j = mod1(i + 1, Nr)
        Hr[i, j] = -1.0
        Hr[j, i] = -1.0
        Jr[i, j] = Hr[i, j] * (-1.0)
        Jr[j, i] = Hr[j, i] * (1.0)
    end
    ar, br, Hrn = KPM.normalizeH(Hr; center=true)
    NCr = 128
    mur = KPM.kpm_2d(
        Hrn, Jr, Jr, NCr, Nr, Nr;
        psi_in=Matrix{ComplexF64}(I, Nr, Nr))
    # Strong Lorentz damping makes the finite-ring transport distribution
    # smooth on both thermal scales, which is the hypothesis of the Mott
    # expansion rather than an NC-resolution convergence claim.
    mu_chem = -1.35
    smooth_kernel = KPM.LorentzKernels(20.0)
    dE = 1e-4 * ar
    sigma_minus = KPM.transport_distribution(
        mur, ar, mu_chem - dE; b=br, NH=Nr, volume=Float64(Nr), NC=NCr,
        kernel=smooth_kernel)
    sigma_plus = KPM.transport_distribution(
        mur, ar, mu_chem + dE; b=br, NH=Nr, volume=Float64(Nr), NC=NCr,
        kernel=smooth_kernel)
    dlnsigma = (log(sigma_plus) - log(sigma_minus)) / (2dE)

    deviations = Float64[]
    for beta_local in (40.0, 80.0)
        result = Logging.with_logger(Logging.NullLogger()) do
            KPM.thermoelectric(
                mur, ar, mu_chem; beta=beta_local, b=br, NH=Nr,
                volume=Float64(Nr), NC=NCr, kernel=smooth_kernel,
                sigma_min=0.0)
        end
        mott = -(π^2 / (3beta_local)) * dlnsigma
        push!(deviations, abs(result.S_over_kB_over_e - mott))
        # Relative deviations are about 0.63% and 0.32% at beta=40 and 80.
        @test result.S_over_kB_over_e ≈ mott rtol=0.02
    end
    # The leading Sommerfeld correction decreases as beta doubles.
    @test deviations[2] < deviations[1]
end

@testset "thermoelectric volume routing" begin
    beta_local = 4.0
    mu_chem = b + 0.1a
    result_v, result_2v = Logging.with_logger(Logging.NullLogger()) do
        rv = KPM.thermoelectric(
            mu2D, a, mu_chem; beta=beta_local, b=b, NH=N,
            volume=vol, g_J=gJ, NC=NC)
        r2v = KPM.thermoelectric(
            mu2D, a, mu_chem; beta=beta_local, b=b, NH=N,
            volume=2vol, g_J=gJ, NC=NC)
        rv, r2v
    end
    @test result_2v.L0 ≈ result_v.L0 / 2 rtol=1e-12
    @test result_2v.L1 ≈ result_v.L1 / 2 rtol=1e-12
    @test result_2v.L2 ≈ result_v.L2 / 2 rtol=1e-12
    @test result_2v.S_over_kB_over_e ≈ result_v.S_over_kB_over_e rtol=1e-12
end

@testset "quadrature, expansion-order, and kernel convergence" begin
    Nr = 64
    Hr = spzeros(ComplexF64, Nr, Nr)
    Jr = spzeros(ComplexF64, Nr, Nr)
    for i in 1:Nr
        j = mod1(i + 1, Nr)
        Hr[i, j] = -1.0
        Hr[j, i] = -1.0
        Jr[i, j] = Hr[i, j] * (-1.0)
        Jr[j, i] = Hr[j, i] * (1.0)
    end
    ar, br, Hrn = KPM.normalizeH(Hr; center=true)
    NCmax = 256
    mur = KPM.kpm_2d(
        Hrn, Jr, Jr, NCmax, Nr, Nr;
        psi_in=Matrix{ComplexF64}(I, Nr, Nr))
    mu_chem = -1.0
    beta_local = 8.0
    lorentz = KPM.LorentzKernels(4.0)

    quad_default, quad_fine = Logging.with_logger(Logging.NullLogger()) do
        q1 = KPM.transport_integrals(
            mur, ar, mu_chem; beta=beta_local, b=br, NH=Nr,
            volume=Float64(Nr), kernel=lorentz, NC=128, quad_N=8 * 128)
        q2 = KPM.transport_integrals(
            mur, ar, mu_chem; beta=beta_local, b=br, NH=Nr,
            volume=Float64(Nr), kernel=lorentz, NC=128, quad_N=16 * 128)
        q1, q2
    end
    @test quad_default.L0 ≈ quad_fine.L0 rtol=1e-10
    @test quad_default.L1 ≈ quad_fine.L1 rtol=1e-10
    @test quad_default.L2 ≈ quad_fine.L2 rtol=1e-10

    jackson128, jackson256, lorentz256 = Logging.with_logger(Logging.NullLogger()) do
        r128 = KPM.thermoelectric(
            mur, ar, mu_chem; beta=beta_local, b=br, NH=Nr,
            volume=Float64(Nr), NC=128, sigma_min=0.0)
        r256 = KPM.thermoelectric(
            mur, ar, mu_chem; beta=beta_local, b=br, NH=Nr,
            volume=Float64(Nr), NC=256, sigma_min=0.0)
        rl = KPM.thermoelectric(
            mur, ar, mu_chem; beta=beta_local, b=br, NH=Nr,
            volume=Float64(Nr), kernel=lorentz, NC=256, sigma_min=0.0)
        r128, r256, rl
    end
    # Observed S values (Jackson-128, Jackson-256, Lorentz-256) are
    # (-0.025771, -0.025898, -0.025718), differing by 0.49% and 0.70%.
    @test jackson128.S_over_kB_over_e ≈ jackson256.S_over_kB_over_e rtol=0.05
    @test lorentz256.S_over_kB_over_e ≈ jackson256.S_over_kB_over_e rtol=0.1
end

@testset "Haldane equal-energy KG excludes antisymmetric Hall response" begin
    Hh, Jxh, Jyh, area_h = haldane_model(
        6, 6; t=1.0, t2=0.2, ϕ=π / 2, m=0.0)
    Dh = size(Hh, 1)
    ah, bh, Hh_norm = KPM.normalizeH(Hh; center=true)
    NCh = 96
    psih = Matrix{ComplexF64}(I, Dh, Dh)
    muh_xy = KPM.kpm_2d(Hh_norm, Jxh, Jyh, NCh, Dh, Dh; psi_in=psih)

    sigma_xy = KPM.kubo_bastin_cond(
        muh_xy, ah, 0.0; b=bh, NH=Dh, area=area_h)
    @test abs(sigma_xy) ≈ 1.0 atol=0.05

    ti = Logging.with_logger(Logging.NullLogger()) do
        KPM.transport_integrals(
            muh_xy, ah, 0.0; beta=20.0, b=bh, NH=Dh,
            volume=area_h, NC=NCh)
    end
    sigma_kg = KPM.transport_distribution(
        muh_xy, ah, 0.0; b=bh, NH=Dh, volume=area_h, NC=NCh)
    # The Fermi-sea Bastin antisymmetric response is invisible to the
    # equal-energy KG contraction by design: KG reconstructs only the
    # symmetric part and therefore vanishes in the insulating gap.
    @test abs(ti.L0) < 0.1
    @test abs(sigma_kg) < 0.1
end
