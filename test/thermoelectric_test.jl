using Test
using LinearAlgebra
using SparseArrays
using Random
using Logging
using KPM

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
