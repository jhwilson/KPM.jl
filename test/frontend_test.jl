using Test
using Random
using SparseArrays
using LinearAlgebra
using Logging
using KPM

NH = 64
NR = 4
H = spdiagm(-1 => ones(NH - 1), 1 => ones(NH - 1))
Jx = spdiagm(-1 => -im .* ones(NH - 1), 1 => im .* ones(NH - 1))
Jy = 0.7 .* Jx
h = KPM.rescale(H)

@testset "rescale provenance" begin
    @test Matrix(h.a .* h.H) + h.b * I ≈ Matrix(H)
    @test all(
        (-1 .< eigvals(Hermitian(Matrix(h.H)))) .& (eigvals(Hermitian(Matrix(h.H))) .< 1),
    )

    H_shifted = H + 0.7 * I
    h_centered = KPM.rescale(H_shifted; center = true)
    @test Matrix(h_centered.a .* h_centered.H) + h_centered.b * I ≈ Matrix(H_shifted)
    lambda_centered = eigvals(Hermitian(Matrix(h_centered.H)))
    @test all((-1 .< lambda_centered) .& (lambda_centered .< 1))

    h_fixed = KPM.rescale(H; fixed_a = 3)
    @test h_fixed.a === 3.0
    @test h_fixed.b == 0.0
end

# Typed wrappers must delegate exactly: bitwise on the CPU. GPU kernels are
# not run-to-run bitwise-deterministic (two identical same-seed kpm_1d calls
# differ at ulps), so with the GPU device active the same delegation is
# asserted at ulp-level tolerance instead.
delegates_exactly(x, y) =
    KPM.whichcore() ? isapprox(x, y; rtol = 1e-9, atol = 1e-12) : x == y
delegates_exactly(x::Tuple, y::Tuple) = all(map(delegates_exactly, x, y))

@testset "typed DOS moments and reconstruction" begin
    rng = Xoshiro(11)
    psi = exp.(rand(rng, Float64, NH, NR) .* (2im * pi))
    KPM.normalize_by_col(psi, NR)
    m = KPM.dos_moments(h; NC = 64, psi_in = copy(psi))
    mu_ref = KPM.kpm_1d(h.H, 64, NR; psi_in = copy(psi))
    @test delegates_exactly(m.mu, mu_ref)
    @test delegates_exactly(KPM.dos(m), KPM.dos(mu_ref, h.a; b = h.b))
    @test delegates_exactly(KPM.dos0(m), KPM.dos0(mu_ref, h.a))
    @test delegates_exactly(KPM.dos0(m; dE_order = 2), KPM.dos0(mu_ref, h.a; dE_order = 2))

    grid = collect(range(-1.5, 1.5, length = 101))
    E, rho = KPM.dos(m; E_grid = grid)
    @test all(isfinite, rho)
    E_range, rho_range = KPM.dos(m; E_range = [-1.5, 1.5], N_tilde = 100)
    @test E ≈ E_range
    @test rho ≈ rho_range
    E_explicit, rho_explicit = KPM.dos(m; E_grid = E_range)
    @test E_explicit == E_range
    @test rho_explicit == rho_range
    _, rho_integer_grid = KPM.dos(m; E_grid = collect(-2:2))
    @test eltype(rho_integer_grid) == Float64
    @test all(isfinite, rho_integer_grid)
end

@testset "typed conductivity moments and reconstruction" begin
    rng = Xoshiro(11)
    psi = exp.(rand(rng, Float64, NH, NR) .* (2im * pi))
    KPM.normalize_by_col(psi, NR)
    m2 = KPM.cond_moments(h, Jx, Jy; NC = 32, psi_in = copy(psi))
    mu2_ref = KPM.kpm_2d(h.H, Jx, Jy, 32, NR, size(h.H, 1); psi_in = copy(psi))
    @test delegates_exactly(m2.mu, mu2_ref)
    @test delegates_exactly(
        KPM.kubo_bastin_cond(m2, 0.1; area = 1.0),
        KPM.kubo_bastin_cond(mu2_ref, h.a, 0.1; b = h.b, NH = size(h.H, 1), area = 1.0),
    )
    @test delegates_exactly(
        KPM.dc_cond_single(m2, 0.1),
        KPM.dc_cond_single(mu2_ref, h.a, 0.1; b = h.b),
    )
    @test delegates_exactly(KPM.dc_cond0(m2), KPM.dc_cond0(mu2_ref, h.a))
    @test delegates_exactly(
        KPM.d_dc_cond(m2, [0.0, 0.1]),
        KPM.d_dc_cond(mu2_ref, h.a, [0.0, 0.1]; b = h.b),
    )
    @test KPM.d_dc_cond(m2, 0.0:0.05:0.2) == KPM.d_dc_cond(m2, collect(0.0:0.05:0.2))
    @test KPM.d_dc_cond(m2, 0.1) isa Real
    @test delegates_exactly(
        KPM.d_dc_cond(m2, 0.1),
        only(KPM.d_dc_cond(mu2_ref, h.a, [0.1]; b = h.b)),
    )

    @test_throws ArgumentError KPM.kubo_bastin_cond(m2, 0.1; area = 1.0, NH = 10)
    @test_throws ArgumentError KPM.ConductivityMoments(
        zeros(ComplexF64, 3, 2),
        1.0,
        0.0,
        NH,
        1,
    )
end

@testset "centered rescaling parity (b != 0)" begin
    # every energy-dependent delegation must forward the stored b: these
    # comparisons are run away from E = b so a dropped shift changes the numbers
    H_shifted = H + 0.7 * I
    hc = KPM.rescale(H_shifted; center = true)
    @test hc.b != 0.0
    psi = KPM.random_phase_vectors(Xoshiro(21), NH, NR)

    mc = KPM.dos_moments(hc; NC = 64, psi_in = copy(psi))
    mu_ref = KPM.kpm_1d(hc.H, 64, NR; psi_in = copy(psi))
    @test delegates_exactly(mc.mu, mu_ref)
    @test delegates_exactly(KPM.dos(mc), KPM.dos(mu_ref, hc.a; b = hc.b))
    grid = collect(range(hc.b - 0.5, hc.b + 0.5, length = 21))
    @test delegates_exactly(
        KPM.dos(mc; E_grid = grid)[2],
        KPM.dos(mu_ref, hc.a; b = hc.b, E_grid = grid)[2],
    )

    m2c = KPM.cond_moments(hc, Jx, Jy; NC = 32, psi_in = copy(psi))
    mu2_ref = KPM.kpm_2d(hc.H, Jx, Jy, 32, NR, NH; psi_in = copy(psi))
    @test delegates_exactly(
        KPM.kubo_bastin_cond(m2c, 0.3; area = 1.0),
        KPM.kubo_bastin_cond(mu2_ref, hc.a, 0.3; b = hc.b, NH = NH, area = 1.0),
    )
    @test delegates_exactly(
        KPM.dc_cond_single(m2c, 0.3),
        KPM.dc_cond_single(mu2_ref, hc.a, 0.3; b = hc.b),
    )
    @test delegates_exactly(
        KPM.d_dc_cond(m2c, [0.3, 0.9]),
        KPM.d_dc_cond(mu2_ref, hc.a, [0.3, 0.9]; b = hc.b),
    )
end

@testset "typed front-end validation and reproducibility" begin
    m1 = KPM.dos_moments(h; NC = 64, NR = 4, rng = Xoshiro(42))
    m2 = KPM.dos_moments(h; NC = 64, NR = 4, rng = Xoshiro(42))
    m3 = KPM.dos_moments(h; NC = 64, NR = 4, rng = Xoshiro(43))
    @test delegates_exactly(m1.mu, m2.mu)   # equal seeds reproduce (bitwise on CPU)
    @test !(m1.mu ≈ m3.mu)
    # unit-norm probes give mu_0 = 1 exactly as in the default random path
    @test m1.mu[1] ≈ 1.0

    # rng path reproduces the package's probe recipe exactly (random phase,
    # column-normalized, no mean centering)
    rng_manual = Xoshiro(5)
    psi_manual = exp.(rand(rng_manual, Float64, NH, NR) .* (2im * pi))
    KPM.normalize_by_col(psi_manual, NR)
    @test KPM.random_phase_vectors(Xoshiro(5), NH, NR) == psi_manual

    psi = KPM.random_phase_vectors(Xoshiro(9), NH, NR)
    moments = KPM.dos_moments(h; NC = 64, psi_in = copy(psi))
    conductivity_moments = KPM.cond_moments(h, Jx, Jy; NC = 32, psi_in = copy(psi))
    @test_throws ArgumentError KPM.dos_moments(h; NC = 64, rng = Xoshiro(1), psi_in = psi)
    @test_throws ArgumentError KPM.dos_moments(h; NC = 63)
    @test_throws ArgumentError KPM.dos(moments; b = 0.2)
    @test_throws ArgumentError KPM.dos_moments(h; NC = 64, psi_in = zeros(NH - 1, NR))
    @test_throws ArgumentError KPM.cond_moments(
        h,
        Jx,
        Jy;
        NC = 32,
        psi_in = zeros(NH - 1, NR),
    )
    @test_throws ArgumentError KPM.kubo_bastin_cond(
        conductivity_moments,
        0.1;
        area = 1.0,
        b = 0.2,
    )

    # DOS sum rule, deterministically. With unit-norm probes μ₀ = Tr[T₀]/D = 1
    # exactly (identity probes make the stochastic trace exact), and
    # ∫_{-1}^{1} Tₙ(x)/(π√(1-x²)) dx = δ_{n0} with g₀ = h₀ = 1 for every
    # kernel, so ∫ρ dE = 1 identically. Evaluating that integral with the
    # Gauss–Chebyshev rule that inverts the 1/√(1-x²) weight,
    #   ∫ρ dE = (1/N) Σ_k a π √(1-x_k²) ρ(a x_k + b),  x_k = cos((k-½)π/N),
    # is exact for the NC-term series once N > NC — hence rtol 1e-12, not the
    # 0.15 that a stochastic trace on a coarse grid used to need.
    psi_full = Matrix{ComplexF64}(I, NH, NH)
    m_full = KPM.dos_moments(h; NC = 128, psi_in = psi_full)
    @test m_full.mu[1] ≈ 1.0 rtol = 1e-12
    Nq = 512
    xq = [cos((k - 0.5) * pi / Nq) for k = 1:Nq]
    _, rho_q = KPM.dos(m_full; E_grid = h.a .* xq .+ h.b)
    @test sum(h.a * pi .* sqrt.(1 .- xq .^ 2) .* rho_q) / Nq ≈ 1.0 rtol = 1e-12

    # The same sum rule read off a uniform grid: the default E_range is
    # b ± 0.99a, so the trapezoid integral misses the band-edge tails. The
    # measured deficit is 2.3e-5 and is N_tilde-independent above ~4096
    # (window truncation, not quadrature error); 1e-4 is 4× that.
    E_t, rho_t = KPM.dos(m_full; N_tilde = 4096)
    integral = sum((rho_t[1:(end-1)] .+ rho_t[2:end]) .* diff(E_t)) / 2
    @test isapprox(integral, 1.0; atol = 1e-4)

    E, rho = KPM.dos(h; NC = 128, NR = 4, rng = Xoshiro(7))
    @test all(isfinite, rho)
    @test occursin("DosMoments", string(moments))
    @test occursin("RescaledHamiltonian", string(h))
end

@testset "typed thermoelectric reconstruction parity" begin
    psi = KPM.random_phase_vectors(Xoshiro(31), NH, NR)
    # The file fixture Jx is Hermitian (velocity-like); thermoelectric uses
    # the package current J=i*v, which is anti-Hermitian and gives L0 > 0.
    Jte = im .* Jx
    m2 = KPM.cond_moments(h, Jte, Jte; NC = 32, psi_in = copy(psi))
    mu2_ref = KPM.kpm_2d(h.H, Jte, Jte, 32, NR, NH; psi_in = copy(psi))
    energies = [-0.2, 0.0, 0.2]

    @test delegates_exactly(
        KPM.transport_distribution(m2, energies; volume = 1.0),
        KPM.transport_distribution(mu2_ref, h.a, energies; b = h.b, NH = NH, volume = 1.0),
    )
    @test delegates_exactly(
        KPM.transport_distribution(m2, 0.1; volume = 1.0),
        KPM.transport_distribution(mu2_ref, h.a, 0.1; b = h.b, NH = NH, volume = 1.0),
    )

    typed_integrals, raw_integrals = Logging.with_logger(Logging.NullLogger()) do
        typed = KPM.transport_integrals(m2, 0.1; beta = 5.0, volume = 1.0)
        raw = KPM.transport_integrals(
            mu2_ref,
            h.a,
            0.1;
            beta = 5.0,
            b = h.b,
            NH = NH,
            volume = 1.0,
        )
        typed, raw
    end
    @test delegates_exactly(typed_integrals.L0, raw_integrals.L0)
    @test delegates_exactly(typed_integrals.L1, raw_integrals.L1)
    @test delegates_exactly(typed_integrals.L2, raw_integrals.L2)
    @test delegates_exactly(typed_integrals.neg_weight, raw_integrals.neg_weight)

    typed_result, raw_result = Logging.with_logger(Logging.NullLogger()) do
        typed = KPM.thermoelectric(m2, 0.1; beta = 5.0, volume = 1.0)
        raw = KPM.thermoelectric(
            mu2_ref,
            h.a,
            0.1;
            beta = 5.0,
            b = h.b,
            NH = NH,
            volume = 1.0,
        )
        typed, raw
    end
    @test isfinite(typed_result.S_over_kB_over_e)
    @test isfinite(raw_result.S_over_kB_over_e)
    @test delegates_exactly(typed_result.L0, raw_result.L0)
    @test delegates_exactly(typed_result.L1, raw_result.L1)
    @test delegates_exactly(typed_result.L2, raw_result.L2)
    @test delegates_exactly(typed_result.S_over_kB_over_e, raw_result.S_over_kB_over_e)

    # Pin stored scalar integrals directly so L2 cannot be accidentally routed from L1.
    scalar_integrals = Logging.with_logger(Logging.NullLogger()) do
        KPM.transport_integrals(m2, 0.1; beta = 5.0, volume = 1.0)
    end
    scalar_result = Logging.with_logger(Logging.NullLogger()) do
        KPM.thermoelectric(m2, 0.1; beta = 5.0, volume = 1.0)
    end
    @test scalar_result.L0 == scalar_integrals.L0
    @test scalar_result.L1 == scalar_integrals.L1
    @test scalar_result.L2 == scalar_integrals.L2

    @test_throws ArgumentError KPM.transport_distribution(m2, [0.1]; volume = 1.0, b = 0.0)
    @test_throws ArgumentError KPM.transport_distribution(m2, [0.1]; volume = 1.0, NH = NH)
    @test_throws ArgumentError KPM.transport_integrals(
        m2,
        0.1;
        beta = 5.0,
        volume = 1.0,
        b = 0.0,
    )
    @test_throws ArgumentError KPM.transport_integrals(
        m2,
        0.1;
        beta = 5.0,
        volume = 1.0,
        NH = NH,
    )
    @test_throws ArgumentError KPM.thermoelectric(
        m2,
        0.1;
        beta = 5.0,
        volume = 1.0,
        b = 0.0,
    )
    @test_throws ArgumentError KPM.thermoelectric(
        m2,
        0.1;
        beta = 5.0,
        volume = 1.0,
        NH = NH,
    )
end

@testset "typed anisotropic thermoelectric tensor" begin
    # A rectangular hopping tensor plus deterministic disorder makes anisotropy
    # genuine, while identity probes remove stochastic ambiguity from the
    # componentwise-reference and symmetrization assertions.
    Lx = 5
    Ly = 5
    Nt = Lx * Ly
    tx = 1.0
    ty = 0.6
    site(x, y) = mod(x, Lx) + 1 + Lx * mod(y, Ly)
    minimum_image(d, L) = mod(d + fld(L, 2), L) - fld(L, 2)
    positions = [(x, y) for y = 0:(Ly-1) for x = 0:(Lx-1)]
    Ht = spzeros(ComplexF64, Nt, Nt)
    Jxt = spzeros(ComplexF64, Nt, Nt)
    Jyt = spzeros(ComplexF64, Nt, Nt)
    rng_t = Xoshiro(2026)
    for i = 1:Nt
        Ht[i, i] = 0.15 * (rand(rng_t) - 0.5)
    end
    for y = 0:(Ly-1), x = 0:(Lx-1)
        i = site(x, y)
        for (j, hopping) in ((site(x + 1, y), tx), (site(x, y + 1), ty))
            Ht[i, j] = Ht[j, i] = -hopping
            dx = minimum_image(positions[i][1] - positions[j][1], Lx)
            dy = minimum_image(positions[i][2] - positions[j][2], Ly)
            Jxt[i, j] = Ht[i, j] * dx
            Jxt[j, i] = Ht[j, i] * -dx
            Jyt[i, j] = Ht[i, j] * dy
            Jyt[j, i] = Ht[j, i] * -dy
        end
    end

    ht = KPM.rescale(Ht; center = true)
    psi_t = Matrix{ComplexF64}(I, Nt, Nt)
    mxx = KPM.cond_moments(ht, Jxt, Jxt; NC = 64, psi_in = copy(psi_t))
    myy = KPM.cond_moments(ht, Jyt, Jyt; NC = 64, psi_in = copy(psi_t))
    mxy = KPM.cond_moments(ht, Jxt, Jyt; NC = 64, psi_in = copy(psi_t))
    myx = KPM.cond_moments(ht, Jyt, Jxt; NC = 64, psi_in = copy(psi_t))
    M = [mxx mxy; myx myy]
    mu_t = ht.b + 0.12ht.a
    beta_t = 2.0
    kwargs_t =
        (; beta = beta_t, volume = Float64(Nt), NC = 64, quad_N = 8 * 64, sigma_min = 0.0)

    raw = Matrix{Any}(undef, 2, 2)
    for i = 1:2, j = 1:2
        raw[i, j] = Logging.with_logger(Logging.NullLogger()) do
            KPM.transport_integrals(
                M[i, j],
                mu_t;
                beta = beta_t,
                volume = Float64(Nt),
                NC = 64,
                quad_N = 8 * 64,
            )
        end
    end
    raw_L0 = [raw[i, j].L0 for i = 1:2, j = 1:2]
    raw_L1 = [raw[i, j].L1 for i = 1:2, j = 1:2]
    raw_L2 = [raw[i, j].L2 for i = 1:2, j = 1:2]
    L0_ref = (raw_L0 + transpose(raw_L0)) / 2
    L1_ref = (raw_L1 + transpose(raw_L1)) / 2
    L2_ref = (raw_L2 + transpose(raw_L2)) / 2
    result = Logging.with_logger(Logging.NullLogger()) do
        KPM.thermoelectric(M, mu_t; kwargs_t...)
    end
    @test result.L0 ≈ L0_ref rtol=1e-12
    @test result.L1 ≈ L1_ref rtol=1e-12
    @test result.L2 ≈ L2_ref rtol=1e-12
    @test result.S_over_kB_over_e ≈
          KPM.seebeck_solve(L0_ref, L1_ref, beta_t; sigma_min = 0.0) rtol=1e-12
    @test result.L0[1, 1] != result.L0[2, 2]

    bad_a = KPM.ConductivityMoments(mxy.mu, mxy.a + 0.1, mxy.b, mxy.NH, mxy.NR)
    bad_NH = KPM.ConductivityMoments(mxy.mu, mxy.a, mxy.b, mxy.NH + 1, mxy.NR)
    M_bad = copy(M)
    M_bad[1, 2] = bad_a
    @test_throws ArgumentError KPM.thermoelectric(M_bad, mu_t; kwargs_t...)
    M_bad[1, 2] = bad_NH
    @test_throws ArgumentError KPM.thermoelectric(M_bad, mu_t; kwargs_t...)
    @test_throws ArgumentError KPM.thermoelectric(
        reshape(fill(mxx, 6), 2, 3),
        mu_t;
        kwargs_t...,
    )
    @test_throws ArgumentError KPM.thermoelectric(fill(mxx, 4, 4), mu_t; kwargs_t...)

    insulating = @test_logs (:warn, r"below-conductivity-floor") match_mode=:any begin
        KPM.thermoelectric(
            M,
            mu_t;
            beta = beta_t,
            volume = Float64(Nt),
            NC = 64,
            quad_N = 8 * 64,
            sigma_min = 1e100,
        )
    end
    @test all(isnan, insulating.S_over_kB_over_e)

    off12 = KPM.ConductivityMoments(10 .* mxx.mu, mxx.a, mxx.b, mxx.NH, mxx.NR)
    off21 = KPM.ConductivityMoments(-10 .* mxx.mu, mxx.a, mxx.b, mxx.NH, mxx.NR)
    M_skew = [mxx off12; off21 myy]
    skew_result = @test_logs (:warn, r"skew") match_mode=:any begin
        KPM.thermoelectric(
            M_skew,
            mu_t;
            beta = beta_t,
            volume = Float64(Nt),
            NC = 64,
            quad_N = 8 * 64,
            sigma_min = 0.0,
        )
    end
    diagonal_neg = maximum(raw[i, i].neg_weight for i = 1:2)
    @test skew_result.neg_weight ≈ diagonal_neg rtol=1e-12
end
