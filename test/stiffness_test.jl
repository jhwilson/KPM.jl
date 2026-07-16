# Generic two-energy response reconstruction checks against dense ED.
using Test
using LinearAlgebra
using SparseArrays
using Random
using KPM

isdefined(@__MODULE__, :EDReference) || include("ed_reference.jl")
using .EDReference

const STIFFNESS_D = 6
const STIFFNESS_RNG = Xoshiro(42)
const STIFFNESS_A = randn(STIFFNESS_RNG, ComplexF64, STIFFNESS_D, STIFFNESS_D)
const STIFFNESS_H0 = Matrix(Hermitian((STIFFNESS_A + STIFFNESS_A') / 2))
const STIFFNESS_JL = randn(STIFFNESS_RNG, ComplexF64, STIFFNESS_D, STIFFNESS_D)
const STIFFNESS_JR = randn(STIFFNESS_RNG, ComplexF64, STIFFNESS_D, STIFFNESS_D)
const STIFFNESS_JL_H = Matrix(Hermitian((STIFFNESS_JL + STIFFNESS_JL') / 2))
const STIFFNESS_JR_H = Matrix(Hermitian((STIFFNESS_JR + STIFFNESS_JR') / 2))
const STIFFNESS_EV = eigvals(Hermitian(STIFFNESS_H0))
const STIFFNESS_B0 = 0.2
const STIFFNESS_A0 = maximum(abs, STIFFNESS_EV .- STIFFNESS_B0) / 0.9
const STIFFNESS_HN = (STIFFNESS_H0 - STIFFNESS_B0 * I) / STIFFNESS_A0
const STIFFNESS_PSI = Matrix{ComplexF64}(I, STIFFNESS_D, STIFFNESS_D)

stiffness_moments(NC) = KPM.kpm_2d(
    STIFFNESS_HN, STIFFNESS_JL, STIFFNESS_JR,
    NC, STIFFNESS_D, STIFFNESS_D;
    psi_in_l=STIFFNESS_PSI, psi_in_r=STIFFNESS_PSI)

mu2D_64 = stiffness_moments(64)
mu2D_256 = stiffness_moments(256)

@testset "two-point Lehmann oracle (omega≠0, complex vertices)" begin
    Pi_kpm_64 = KPM.two_energy_response(
        mu2D_64, STIFFNESS_A0; b=STIFFNESS_B0, beta=8.0, eta=0.25,
        omega=0.3, Ef=0.15, NH=STIFFNESS_D, volume=1.0)
    Pi_kpm_256 = KPM.two_energy_response(
        mu2D_256, STIFFNESS_A0; b=STIFFNESS_B0, beta=8.0, eta=0.25,
        omega=0.3, Ef=0.15, NH=STIFFNESS_D, volume=1.0)
    Pi_ed = ed_two_energy_response(
        STIFFNESS_H0, STIFFNESS_JL, STIFFNESS_JR;
        beta=8.0, eta=0.25, omega=0.3, Ef=0.15)
    err64 = abs(Pi_kpm_64 - Pi_ed)
    err256 = abs(Pi_kpm_256 - Pi_ed)
    println("two-point oracle: ED=$(Pi_ed), KPM NC=64=$(Pi_kpm_64), KPM NC=256=$(Pi_kpm_256)")
    println("two-point oracle errors: NC=64 $(err64), NC=256 $(err256)")

    @test err256 < err64
    @test abs(real(Pi_kpm_256 - Pi_ed)) < abs(real(Pi_kpm_64 - Pi_ed))
    @test abs(imag(Pi_kpm_256 - Pi_ed)) < abs(imag(Pi_kpm_64 - Pi_ed))
    TOL_DYNAMIC = 0.015
    @test isapprox(Pi_kpm_256, Pi_ed; rtol=TOL_DYNAMIC)
end

@testset "static eta=0 divided-difference kernel" begin
    # For the deliberately non-Hermitian oracle vertices the divided-
    # difference kernel is real, but its contraction need not be. Preserve
    # that generic comparison, then check the requested reality property on
    # Hermitian versions of the same fixed vertices.
    Pi_kpm_generic = KPM.two_energy_response(
        mu2D_256, STIFFNESS_A0; b=STIFFNESS_B0, beta=8.0, eta=0.0,
        omega=0.0, Ef=0.15, NH=STIFFNESS_D, volume=1.0)
    Pi_ed_generic = ed_two_energy_response(
        STIFFNESS_H0, STIFFNESS_JL, STIFFNESS_JR;
        beta=8.0, eta=0.0, omega=0.0, Ef=0.15)
    println("static divided difference (non-Hermitian): ED=$(Pi_ed_generic), KPM NC=256=$(Pi_kpm_generic), error=$(abs(Pi_kpm_generic - Pi_ed_generic))")
    @test isapprox(Pi_kpm_generic, Pi_ed_generic; rtol=0.012)

    mu2D_static = KPM.kpm_2d(
        STIFFNESS_HN, STIFFNESS_JL_H, STIFFNESS_JR_H,
        256, STIFFNESS_D, STIFFNESS_D;
        psi_in_l=STIFFNESS_PSI, psi_in_r=STIFFNESS_PSI)
    Pi_kpm = KPM.two_energy_response(
        mu2D_static, STIFFNESS_A0; b=STIFFNESS_B0, beta=8.0, eta=0.0,
        omega=0.0, Ef=0.15, NH=STIFFNESS_D, volume=1.0)
    Pi_ed = ed_two_energy_response(
        STIFFNESS_H0, STIFFNESS_JL_H, STIFFNESS_JR_H;
        beta=8.0, eta=0.0, omega=0.0, Ef=0.15)
    println("static divided difference (Hermitian): ED=$(Pi_ed), KPM NC=256=$(Pi_kpm), error=$(abs(Pi_kpm - Pi_ed))")
    @test imag(Pi_kpm) ≈ 0 atol=1e-10
    @test imag(Pi_ed) ≈ 0 atol=1e-10
    TOL_STATIC = 0.05
    @test isapprox(Pi_kpm, Pi_ed; rtol=TOL_STATIC)
    @test_throws ArgumentError KPM.two_energy_response(
        mu2D_256, STIFFNESS_A0; b=STIFFNESS_B0, beta=8.0, eta=0.0,
        omega=0.3, NH=STIFFNESS_D, volume=1.0)
    @test_throws ArgumentError KPM.two_energy_response(
        mu2D_256, STIFFNESS_A0; b=STIFFNESS_B0, beta=Inf, eta=0.0,
        omega=0.0, NH=STIFFNESS_D, volume=1.0)
end

@testset "typed wrapper parity and beta=Inf" begin
    m = KPM.ConductivityMoments(
        mu2D_256, STIFFNESS_A0, STIFFNESS_B0, STIFFNESS_D, STIFFNESS_D)
    raw = KPM.two_energy_response(
        mu2D_256, STIFFNESS_A0; b=STIFFNESS_B0, beta=8.0, eta=0.25,
        omega=0.3, Ef=0.15, NH=STIFFNESS_D, volume=1.0)
    wrapped = KPM.two_energy_response(
        m; beta=8.0, eta=0.25, omega=0.3, Ef=0.15, volume=1.0)
    @test wrapped == raw
    @test_throws ArgumentError KPM.two_energy_response(
        m; beta=8.0, eta=0.25, volume=1.0, b=STIFFNESS_B0)
    @test_throws ArgumentError KPM.two_energy_response(
        m; beta=8.0, eta=0.25, volume=1.0, NH=STIFFNESS_D)

    Pi_kpm = KPM.two_energy_response(
        m; beta=Inf, eta=0.25, omega=0.3, Ef=0.15, volume=1.0)
    Pi_ed = ed_two_energy_response(
        STIFFNESS_H0, STIFFNESS_JL, STIFFNESS_JR;
        beta=Inf, eta=0.25, omega=0.3, Ef=0.15)
    println("beta=Inf oracle: ED=$(Pi_ed), KPM NC=256=$(Pi_kpm), error=$(abs(Pi_kpm - Pi_ed))")
    @test isfinite(real(Pi_kpm)) && isfinite(imag(Pi_kpm))
    TOL_BETA_INF = 0.02
    @test isapprox(Pi_kpm, Pi_ed; rtol=TOL_BETA_INF)
end
