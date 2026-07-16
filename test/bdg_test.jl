# Matrix-free reduced BdG operator and rescaling checks against dense ED.
using Test
using LinearAlgebra
using SparseArrays
using Random
using KPM

isdefined(@__MODULE__, :EDReference) || include("ed_reference.jl")
using .EDReference

const BdG_N = 4
const BdG_t = 1.0
const BdG_mu = -0.5
const BdG_U = 2.0
const BdG_n = fill(0.5, BdG_N)
const BdG_Delta = fill(0.3 + 0.4im, BdG_N)
const BdG_h, BdG_pos, BdG_disp = ring_model(BdG_N; t=BdG_t)
const BdG_op = KPM.BdGOperator(BdG_h; mu=BdG_mu, U=BdG_U, n=BdG_n, Delta=BdG_Delta)
const BdG_Hd = bdg_matrix(BdG_h, BdG_mu, BdG_U, BdG_n, BdG_Delta)

@testset "BdG matrix-free action vs assembled" begin
    Hmf = zeros(ComplexF64, 2BdG_N, 2BdG_N)
    for j in axes(Hmf, 2)
        e_j = zeros(ComplexF64, 2BdG_N)
        e_j[j] = 1
        mul!(view(Hmf, :, j), BdG_op, e_j)
    end
    @test Hmf ≈ BdG_Hd atol=1e-12

    rng = Xoshiro(7)
    x = randn(rng, ComplexF64, 2BdG_N)
    y = randn(rng, ComplexF64, 2BdG_N)
    α = 1.3 - 0.2im
    β = 0.7 + 0.5im
    @test mul!(copy(y), BdG_op, x, α, β) ≈ α * BdG_Hd * x + β * y atol=1e-12

    y_overwrite = randn(rng, ComplexF64, 2BdG_N)
    mul!(y_overwrite, BdG_op, x, true, false)
    @test y_overwrite ≈ BdG_Hd * x atol=1e-12
    y2 = randn(rng, ComplexF64, 2BdG_N)
    mul!(y2, BdG_op, x)
    @test y2 ≈ BdG_Hd * x atol=1e-12

    X = randn(rng, ComplexF64, 2BdG_N, 3)
    Y = randn(rng, ComplexF64, 2BdG_N, 3)
    @test mul!(copy(Y), BdG_op, X, α, β) ≈ α * BdG_Hd * X + β * Y atol=1e-12
    @test BdG_Hd ≈ BdG_Hd'
end

@testset "particle-hole symmetry (complex Δ)" begin
    ev = eigvals(Hermitian(BdG_Hd))
    @test sort(ev) ≈ -reverse(sort(ev)) atol=1e-10
end

@testset "global Δ-phase invariance" begin
    φ = 0.77
    Hphase = bdg_matrix(BdG_h, BdG_mu, BdG_U, BdG_n, exp(im * φ) .* BdG_Delta)
    @test eigvals(Hermitian(BdG_Hd)) ≈ eigvals(Hermitian(Hphase)) atol=1e-10
end

@testset "ScaledOperator and spectral_radius bounds" begin
    rad, v = KPM.spectral_radius(BdG_op)
    exact = maximum(abs, eigvals(Hermitian(BdG_Hd)))
    @test rad ≈ exact rtol=2e-3
    @test rad <= exact + 1e-8
    @test length(v) == 2BdG_N

    rh = KPM.rescale(BdG_op)
    @test rh.b == 0.0
    @test all(abs.(eigvals(Hermitian(BdG_Hd ./ rh.a))) .< 1.0)

    rng = Xoshiro(8)
    x = randn(rng, ComplexF64, 2BdG_N)
    y = randn(rng, ComplexF64, 2BdG_N)
    mul!(y, rh.H, x)
    @test y ≈ (BdG_Hd / rh.a) * x atol=1e-12

    S2 = KPM.ScaledOperator(BdG_op, 2.0, 0.3)
    α = 1.3 - 0.2im
    β = 0.7 + 0.5im
    y = randn(rng, ComplexF64, 2BdG_N)
    @test mul!(copy(y), S2, x, α, β) ≈ α * ((BdG_Hd - 0.3I) / 2.0) * x + β * y atol=1e-12

    m = KPM.dos_moments(rh; NC=32, NR=2, rng=Xoshiro(1))
    @test all(isfinite, m.mu)
    @test m.mu[1] ≈ 1 atol=1e-10
end
