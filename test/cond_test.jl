using Test
using LinearAlgebra
using SparseArrays
using Random
using KPM

# Consistency checks for the conductivity helpers. `dc_cond0` and
# `dc_cond_single(..., Ef=0)` evaluate the same Eq.-25 sum through independent
# code paths (mod-4 index slicing vs closed-form Tn(0) weights), so they must
# agree for arbitrary, non-symmetric moment matrices — the old dc_cond0
# subtracted `onethree` twice, which only cancels for symmetric μ.

@testset "dc_cond0 == dc_cond_single at Ef = 0" begin
    rng = Xoshiro(5)
    NC = 32
    a = 2.3
    mu = randn(rng, ComplexF64, NC, NC)   # generic: μ_nm ≠ μ_mn
    v0 = KPM.dc_cond0(mu, a; NC = NC)
    v1 = KPM.dc_cond_single(mu, a, 0.0; NC = NC)
    @test v0 ≈ v1 atol = 1e-12
end

@testset "Γnmμnmαβ factorized contraction vs direct sum" begin
    rng = Xoshiro(6)
    NC = 24
    mu = randn(rng, ComplexF64, NC, NC)
    for ε in (-0.83, 0.0, 0.37)
        direct = sum(KPM.Γnm.(0:(NC-1), (0:(NC-1))', ε) .* mu)
        @test KPM.Γnmμnmαβ(mu, ε, NC) ≈ direct rtol = 1e-11
    end
end

@testset "d_dc_cond: zero (not garbage) outside the band" begin
    rng = Xoshiro(8)
    NC = 16
    a = 1.7
    mu = randn(rng, ComplexF64, NC, NC)
    dσ = KPM.d_dc_cond(mu, a, [0.3, 2.5 * a])
    @test isfinite(dσ[1])
    @test dσ[2] == 0.0
end

@testset "d_dc_cond(dE_order=1) vs central finite differences" begin
    # The dE_order = 1 path differentiates the same reconstruction through
    # Zygote.forwarddiff, so it must reproduce a finite-difference derivative
    # of the dE_order = 0 curve.
    #
    # Step size: the fourth-order central stencil
    #   f'(E) ≈ [f(E-2h) - 8f(E-h) + 8f(E+h) - f(E+2h)] / (12h)
    # has truncation error h⁴ f⁽⁵⁾/30. Measured relative deviation from the
    # analytic derivative at this μ, NC = 16, a = 1.7:
    #   h/a = 1e-2 → 1.1e-5,  3e-3 → 8.8e-8,  1e-3 → 1.1e-9   (h⁴ scaling).
    # Roundoff is eps·|f|/(h|f'|) ≈ 1e-14 at h = 1e-3·a, so the total error at
    # that step is ~1e-9 — two orders below the 1e-7 tolerance asserted here.
    rng = Xoshiro(8)
    NC = 16
    a = 1.7
    b = 0.35
    mu = randn(rng, ComplexF64, NC, NC)
    f0(E) = KPM.d_dc_cond(mu, a, [E]; b = b, NC = NC)[1]
    h = 1e-3 * a
    for E in (b - 0.5a, b + 0.3a, b + 0.72a)
        d1 = KPM.d_dc_cond(mu, a, [E]; b = b, NC = NC, dE_order = 1)[1]
        fd = (f0(E - 2h) - 8 * f0(E - h) + 8 * f0(E + h) - f0(E + 2h)) / (12h)
        @test d1 ≈ fd rtol = 1e-7
    end

    # scalar-energy method delegates to the array one
    @test KPM.d_dc_cond(mu, a, b + 0.3a; b = b, NC = NC, dE_order = 1) ==
          KPM.d_dc_cond(mu, a, [b + 0.3a]; b = b, NC = NC, dE_order = 1)
end

@testset "d_dc_cond(dE_order=2) is currently unsupported" begin
    # PINNED LIMITATION. The throw originates in Zygote's `forward_jacobian`
    # when Zygote.forwarddiff is nested, not mutation in Γnmμnmαβ (which is
    # now a pure matvec pair in src/applications/dc_cond_util.jl:21). The
    # replacement check is the fourth-order second-derivative stencil below.
    rng = Xoshiro(8)
    NC = 16
    a = 1.7
    b = 0.35
    mu = randn(rng, ComplexF64, NC, NC)
    E = b
    f0(E) = KPM.d_dc_cond(mu, a, [E]; b = b, NC = NC)[1]
    h = 1e-3 * a  # documented step: balances this stencil's O(h⁴) truncation and roundoff
    stencil = (-f0(E + 2h) + 16f0(E + h) - 30f0(E) + 16f0(E - h) - f0(E - 2h)) /
              (12h^2)
    d2 = try
        KPM.d_dc_cond(mu, a, [E]; b = b, NC = NC, dE_order = 2)[1]
    catch
        NaN
    end
    @test_broken d2 ≈ stencil rtol = 1e-7
end

@testset "kpm_2d moment_parity selects the documented index parity" begin
    # `moment_parity=:ODD` keeps μnm with mod(n+m, 2) == 1 (0-based n, m) and
    # zeroes the rest; `:EVEN` keeps the complement. The kept entries must be
    # bit-identical to the :NONE calculation — the parity option only skips
    # work, it must not change any number.
    rng = Xoshiro(31)
    D = 10
    NC = 8
    A = randn(rng, ComplexF64, D, D)
    Hd = (A + A') / 2
    Hd ./= (2 * maximum(abs, eigvals(Hermitian(Hd))))
    H = sparse(Hd)
    Jα = sparse(randn(rng, ComplexF64, D, D))
    Jβ = sparse(randn(rng, ComplexF64, D, D))
    psi = Matrix{ComplexF64}(I, D, D)          # exact trace

    mu_none = KPM.kpm_2d(H, Jα, Jβ, NC, D, D; psi_in = psi, moment_parity = :NONE)
    mu_odd = KPM.kpm_2d(H, Jα, Jβ, NC, D, D; psi_in = psi, moment_parity = :ODD)
    mu_even = KPM.kpm_2d(H, Jα, Jβ, NC, D, D; psi_in = psi, moment_parity = :EVEN)

    mask_odd = [mod((n - 1) + (m - 1), 2) == 1 for n = 1:NC, m = 1:NC]
    @test mu_odd == mu_none .* mask_odd
    @test mu_even == mu_none .* .!mask_odd
    @test mu_odd .+ mu_even == mu_none
    @test count(!iszero, mu_odd) == NC^2 ÷ 2

    @test_throws ArgumentError KPM.kpm_2d(
        H,
        Jα,
        Jβ,
        NC,
        D,
        D;
        psi_in = psi,
        moment_parity = :SOMETHING,
    )
end

@testset "kernel application: no-mutate variants match" begin
    NC = 15
    mu2 = ones(NC, NC)
    t_mut = KPM.mu2D_apply_kernel_and_h(copy(mu2), NC, KPM.JacksonKernel)
    t_nomut = KPM.mu2D_apply_kernel_and_h_no_mutate(mu2, NC, KPM.JacksonKernel)
    @test t_mut ≈ t_nomut
    @test all(mu2 .== 1)  # input untouched
end
