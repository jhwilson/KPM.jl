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

@testset "one-site analytic update (Eqs 26-27)" begin
    h1 = spzeros(Float64, 1, 1)
    mu = 0.3
    U = 1.5
    n = [0.0]
    Delta = [0.3 + 0.4im]
    beta = 5.0
    g_rho = 1.0

    xi = -mu
    E = sqrt(xi^2 + abs2(only(Delta)))
    Delta_exact = U * only(Delta) * tanh(beta * E / 2) / (2E)
    n_exact = 1 / 2 - xi * tanh(beta * E / 2) / (2E)

    op = KPM.BdGOperator(h1; mu=mu, U=U, n=n, Delta=Delta)
    rh = KPM.rescale(op)
    m128 = KPM.bdg_local_moments(rh; NC=128, g_rho=g_rho)
    m512 = KPM.bdg_local_moments(rh; NC=512, g_rho=g_rho)
    n128, Delta128 = KPM.bdg_update(m128; beta=beta)
    n512, Delta512 = KPM.bdg_update(m512; beta=beta)

    err128_Delta = abs(only(Delta128) - Delta_exact)
    err512_Delta = abs(only(Delta512) - Delta_exact)
    err128_n = abs(only(n128) - n_exact)
    err512_n = abs(only(n512) - n_exact)
    println("one-site errors: Delta NC=128 $(err128_Delta), NC=512 $(err512_Delta); n NC=128 $(err128_n), NC=512 $(err512_n)")

    TOL_Delta = 3e-5
    TOL_n = 1.2e-5
    @test err512_Delta < err128_Delta
    @test err512_n < err128_n
    @test err512_Delta < TOL_Delta
    @test err512_n < TOL_n
    @test TOL_Delta <= 1e-3
    @test TOL_n <= 1e-3

    m512_g2 = KPM.bdg_local_moments(rh; NC=512, g_rho=2)
    n512_g2, Delta512_g2 = KPM.bdg_update(m512_g2; beta=beta)
    @test n512_g2 ≈ 2 .* n512 atol=1e-12 rtol=0
    @test Delta512_g2 ≈ Delta512 atol=1e-12 rtol=0
    @test angle(only(Delta512)) ≈ angle(Delta_exact) atol=1e-3
end

@testset "ring update vs exact eigenvector sums" begin
    N = 4
    h, _, _ = ring_model(N; t=1.0)
    mu = -0.3
    U = fill(2.0, N)
    n = fill(0.4, N)
    Delta = fill(0.25 + 0.35im, N)
    beta = 10.0
    g_rho = 1.0

    H = bdg_matrix(h, mu, U, n, Delta)
    F = eigen(Hermitian(Matrix(H)))
    f = KPM.fermiFunctions(0.0, beta)
    occupations = f.(F.values)
    Delta_exact = ComplexF64[-U[i] * sum(F.vectors[i, j] * conj(F.vectors[i + N, j]) * occupations[j]
                                        for j in eachindex(F.values)) for i in 1:N]
    n_exact = Float64[sum(abs2(F.vectors[i, j]) * occupations[j]
                          for j in eachindex(F.values)) for i in 1:N]

    op = KPM.BdGOperator(h; mu=mu, U=U, n=n, Delta=Delta)
    rh = KPM.rescale(op)
    m64 = KPM.bdg_local_moments(rh; NC=64, g_rho=g_rho, batch_size=3)
    m256 = KPM.bdg_local_moments(rh; NC=256, g_rho=g_rho, batch_size=3)
    n64, Delta64 = KPM.bdg_update(m64; beta=beta)
    n256, Delta256 = KPM.bdg_update(m256; beta=beta)

    err64_Delta = maximum(abs.(Delta64 .- Delta_exact))
    err256_Delta = maximum(abs.(Delta256 .- Delta_exact))
    err64_n = maximum(abs.(n64 .- n_exact))
    err256_n = maximum(abs.(n256 .- n_exact))
    println("ring errors: Delta NC=64 $(err64_Delta), NC=256 $(err256_Delta); n NC=64 $(err64_n), NC=256 $(err256_n)")

    TOL_Delta = 1e-3
    TOL_n = 3e-4
    @test err256_Delta < err64_Delta
    @test err256_n < err64_n
    @test err256_Delta < TOL_Delta
    @test err256_n < TOL_n
    @test TOL_Delta <= 1e-3
    @test TOL_n <= 1e-3

    mu_rho, mu_delta = KPM.bdg_site_moments(rh.H, op.N, collect(1:op.N), 256;
                                             batch_size=3)
    n_raw, Delta_raw = KPM.bdg_update(mu_rho, mu_delta, rh.a;
                                      U=op.U, beta=beta, g_rho=g_rho)
    @test n_raw == n256
    @test Delta_raw == Delta256
    @test maximum(abs.(n256 .- first(n256))) < 1e-10
    @test maximum(abs.(Delta256 .- first(Delta256))) < 1e-10
end
