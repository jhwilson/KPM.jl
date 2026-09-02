using Test
using LinearAlgebra
using SparseArrays
using KPM

include("ed_reference.jl")
using .EDReference

# Units and normalization of the KPM Kubo–Bastin conductivity, validated
# against exact diagonalization on the Haldane model (Chern insulator):
#  - Hall channel, Fermi level in the gap: exact η-free ED reference (TKNN),
#    KPM must give the quantized σ_xy = C e²/h.
#  - Longitudinal channel, Fermi level in the band: ED with Lorentzian
#    broadening of both δ(ε-H) and G(ε), matched to the KPM Lorentz kernel.
#
# The trace is evaluated deterministically (identity probe matrix), so the
# only KPM errors are kernel broadening and quadrature — no stochastic noise.

t2 = 0.2      # gap 6√3 t2 ≈ 2.1 — large, so modest NC resolves it
H, Jx, Jy, area = haldane_model(6, 6; t = 1.0, t2 = t2, ϕ = π/2, m = 0.0)
D = size(H, 1)

a, b, H_norm = KPM.normalizeH(H; center = true)
NC = 96
psi = Matrix{ComplexF64}(I, D, D)          # full deterministic trace
mu2D_xy = KPM.kpm_2d(H_norm, Jx, Jy, NC, D, D; psi_in = psi)
mu2D_xx = KPM.kpm_2d(H_norm, Jx, Jx, NC, D, D; psi_in = psi)

@testset "quantum Hall: σ_xy quantized and matches ED (units pinned)" begin
    # independent sign anchor: Fukui–Hatsugai–Suzuki Berry flux of the same
    # model's Bloch Hamiltonian — wavefunction overlaps only, no velocity
    # operators, so a simultaneous sign error in the ED and KPM Kubo formulas
    # cannot fool it. TKNN: σ_xy = +C e²/h.
    C_fhs = chern_number_fhs(haldane_bloch(; t = 1.0, t2 = t2, ϕ = π/2, m = 0.0); Nk = 30)
    C = round(C_fhs)
    @test abs(C) == 1
    @test C_fhs ≈ C atol = 1e-10

    # exact ED reference: quantized to the FHS Chern number, including sign
    σxy_ed = ed_hall_conductivity_T0(H, Jx, Jy, area; Ef = 0.0)
    @test abs(σxy_ed - C) < 0.02

    # the two independent ED routines agree in the gap
    @test ed_kubo_bastin(H, Jx, Jy, area; Ef = 0.0, eta = 0.01) ≈ σxy_ed atol = 1e-2
    @test ed_kubo_bastin_broadened(H, Jx, Jy, area; Ef = 0.0, eta = 0.05) ≈ σxy_ed atol =
        1e-2

    # KPM Kubo–Bastin in e²/h units: quantized, correct sign, matches ED
    σxy_kpm = KPM.kubo_bastin_cond(mu2D_xy, a, 0.0; b = b, NH = D, area = area)
    @test σxy_kpm ≈ σxy_ed atol = 0.03
    @test σxy_kpm ≈ C atol = 0.05

    # finite temperature deep in the gap changes nothing (T ≪ gap)
    σxy_T = KPM.kubo_bastin_cond(mu2D_xy, a, 0.0; b = b, NH = D, area = area, beta = 50.0)
    @test σxy_T ≈ σxy_kpm atol = 0.01
end

@testset "shifted spectrum: center shift b flows through the Kubo–Bastin chain" begin
    # same Chern insulator, spectrum rigidly shifted: b ≈ shift, and the
    # Fermi function argument f(a x + b) plus x_F = (Ef - b)/a must all be
    # exercised for the plateau to survive at Ef = shift
    shift = 2.5
    Hs = H + shift * sparse(I, D, D)
    as, bs, Hs_norm = KPM.normalizeH(Hs; center = true)
    @test abs(bs - shift) < 0.05

    mu_s = KPM.kpm_2d(Hs_norm, Jx, Jy, NC, D, D; psi_in = psi)
    σxy_s = KPM.kubo_bastin_cond(mu_s, as, shift; b = bs, NH = D, area = area)
    σxy_ed = ed_hall_conductivity_T0(H, Jx, Jy, area; Ef = 0.0)
    @test σxy_s ≈ σxy_ed atol = 0.03
end

@testset "trivial phase: KPM tracks the exact (non-quantized) ED value" begin
    # m > 3√3 t2 sinϕ puts the model in the C = 0 phase. At this small size
    # the gap is only ≈ 0.56, so σ_xy carries visible finite-size corrections
    # (≈ -0.11 e²/h, not exactly 0) — KPM must reproduce the exact ED value,
    # not the idealized thermodynamic-limit one.
    Ht, Jxt, Jyt, areat = haldane_model(6, 6; t = 1.0, t2 = t2, ϕ = π/2, m = 1.6)
    σxy_ed = ed_hall_conductivity_T0(Ht, Jxt, Jyt, areat; Ef = 0.0)
    @test abs(σxy_ed) < 0.2   # clearly not ±1

    at, bt, Ht_norm = KPM.normalizeH(Ht; center = true)
    mu_t = KPM.kpm_2d(Ht_norm, Jxt, Jyt, NC, D, D; psi_in = psi)
    σxy_kpm = KPM.kubo_bastin_cond(mu_t, at, 0.0; b = bt, NH = D, area = areat)
    @test σxy_kpm ≈ σxy_ed atol = 0.03
end

@testset "currents must come from the unrescaled H: the a² counterfactual" begin
    # Package convention (CLAUDE.md, kubo_bastin_cond docstring): the bond
    # currents are (J_α)_ij = H_ij (r_i - r_j)_α built from the ORIGINAL H.
    # Applying the same rule to H_norm = (H - b I)/a instead scales every
    # off-diagonal entry by 1/a — the b-shift is diagonal and carries no
    # displacement — so both current operators shrink by a, the moments by a²,
    # and, since σ ∝ Re[Γ μ̃]/a² is linear in μ, the reported σ by a².
    #
    # The bond displacements are recovered from the model's own J and H (they
    # are user data, not something inferred from the matrix) and the bond rule
    # is then re-applied to H_norm, so the counterfactual really is "the same
    # construction on the wrong Hamiltonian".
    Hd = Matrix(H)
    Hnd = Matrix(H_norm)
    disp(Jm) = [abs(Hd[i, j]) > 0 ? Jm[i, j] / Hd[i, j] : 0.0 + 0im for i = 1:D, j = 1:D]
    dxm, dym = disp(Matrix(Jx)), disp(Matrix(Jy))
    Jx_wrong = sparse(Hnd .* dxm)
    Jy_wrong = sparse(Hnd .* dym)
    @test Jx_wrong ≈ Jx ./ a rtol = 1e-12          # the claim above, checked
    @test Jy_wrong ≈ Jy ./ a rtol = 1e-12

    mu_wrong = KPM.kpm_2d(H_norm, Jx_wrong, Jy_wrong, NC, D, D; psi_in = psi)
    @test mu_wrong ≈ mu2D_xy ./ a^2 rtol = 1e-10   # moment-level statement

    σ_right = KPM.kubo_bastin_cond(mu2D_xy, a, 0.0; b = b, NH = D, area = area)
    σ_wrong = KPM.kubo_bastin_cond(mu_wrong, a, 0.0; b = b, NH = D, area = area)
    σxy_ed = ed_hall_conductivity_T0(H, Jx, Jy, area; Ef = 0.0)

    @test σ_right ≈ σxy_ed atol = 0.03             # correct convention: matches ED
    @test σ_wrong ≈ σ_right / a^2 rtol = 1e-10     # wrong convention: off by a²
    @test a^2 > 5                                  # ...and a² is not close to 1,
    @test !isapprox(σ_wrong, σxy_ed; atol = 0.3)   # so the error is not subtle
end

@testset "σ_xx: zero in the gap, matches broadened ED in the band" begin
    # insulating: no longitudinal response at Ef in the gap
    σxx_gap = KPM.kubo_bastin_cond(mu2D_xx, a, 0.0; b = b, NH = D, area = area)
    @test abs(σxx_gap) < 1e-4

    # metallic: Lorentz kernel λ ↔ Lorentzian broadening η = a λ √(1-x_F²)/NC
    Ef = -2.0
    xf = (Ef - b) / a
    for λ in (3.0, 4.0)
        η = a * λ * sqrt(1 - xf^2) / NC
        σxx_kpm = KPM.kubo_bastin_cond(
            mu2D_xx,
            a,
            Ef;
            b = b,
            NH = D,
            area = area,
            kernel = KPM.LorentzKernels(λ),
        )
        σxx_ed = ed_kubo_bastin_broadened(H, Jx, Jx, area; Ef = Ef, eta = η)
        @test σxx_kpm ≈ σxx_ed rtol = 0.15
    end
end
