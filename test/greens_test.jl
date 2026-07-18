using Test
using LinearAlgebra
using SparseArrays
using Random
using KPM

# Complex matrix-element moments ⟨ψl|T_n(H)|ψr⟩ with independent bra/ket, and
# the Green-function reconstruction built on them. Moment checks are exact
# (direct recurrence at 1e-12, ED spectral sums at 1e-8); reconstruction is
# anchored by the CPGF route at finite η, which is exact up to the series tail.

# Shared prescaled test Hamiltonian: dense complex Hermitian, spectrum inside
# (-1, 1), as in dos_test.jl.
function _prescaled_hermitian(rng, NH; margin = 0.95)
    A = randn(rng, ComplexF64, NH, NH)
    Hd = Matrix((A + A') / 2)
    a = maximum(abs, eigvals(Hermitian(Hd))) / margin
    Hd ./= a
    return Hd, sparse(Hd)
end

@testset "left/right moments vs direct recurrence and ED" begin
    rng = Xoshiro(7)
    NH = 64
    NC = 33   # odd on purpose: the non-doubling path must accept it
    NR = 2
    Hd, H = _prescaled_hermitian(rng, NH)

    ψl = randn(rng, ComplexF64, NH, NR)
    ψr = randn(rng, ComplexF64, NH, NR)
    foreach(j -> ψl[:, j] ./= norm(ψl[:, j]), 1:NR)
    foreach(j -> ψr[:, j] ./= norm(ψr[:, j]), 1:NR)

    mu_all = zeros(ComplexF64, NR, NC)
    KPM.kpm_1d!(H, NC, NR, NH, mu_all, ψl, ψr)

    # direct three-term recurrence, complex dots
    mu_direct = zeros(ComplexF64, NR, NC)
    for j in 1:NR
        α_prev = ψr[:, j]
        α_curr = Hd * ψr[:, j]
        mu_direct[j, 1] = dot(ψl[:, j], α_prev)
        mu_direct[j, 2] = dot(ψl[:, j], α_curr)
        for n in 3:NC
            α_prev, α_curr = α_curr, 2 .* (Hd * α_curr) .- α_prev
            mu_direct[j, n] = dot(ψl[:, j], α_curr)
        end
    end
    @test mu_all ≈ mu_direct atol = 1e-12

    # ED: μ_n = Σ_m ⟨ψl|m⟩⟨m|ψr⟩ T_n(λ_m), complex
    λ, V = eigen(Hermitian(Hd))
    for j in 1:NR
        w = conj.(V' * ψl[:, j]) .* (V' * ψr[:, j])
        mu_ed = [sum(w .* cos.(n .* acos.(clamp.(λ, -1, 1)))) for n in 0:(NC - 1)]
        @test vec(mu_all[j, :]) ≈ mu_ed atol = 1e-8
    end

    # genuinely complex off-diagonal moments (not an accidental real case)
    @test maximum(abs.(imag.(mu_all))) > 1e-3
end

@testset "left == right: non-doubling path agrees with doubling path" begin
    rng = Xoshiro(11)
    NH = 48
    NC = 32   # even so the doubling path is legal
    NR = 3
    _, H = _prescaled_hermitian(rng, NH)

    ψ = exp.(2im * pi * rand(rng, NH, NR))
    KPM.normalize_by_col(ψ, NR)

    mu_doubling = zeros(ComplexF64, NR, NC)
    KPM.kpm_1d!(H, NC, NR, NH, mu_doubling, ψ)

    mu_lr = zeros(ComplexF64, NR, NC)
    KPM.kpm_1d!(H, NC, NR, NH, mu_lr, copy(ψ), copy(ψ))

    @test mu_lr ≈ mu_doubling atol = 1e-12
    # diagonal moments of a Hermitian H are real to roundoff
    @test maximum(abs.(imag.(mu_lr))) < 1e-12
end

@testset "kpm_1d wrapper: complex average for left/right pairs" begin
    rng = Xoshiro(23)
    NH = 32
    NC = 17
    NR = 2
    _, H = _prescaled_hermitian(rng, NH)

    ψl = randn(rng, ComplexF64, NH, NR)
    ψr = randn(rng, ComplexF64, NH, NR)

    mu_avg = KPM.kpm_1d(H, NC, NR; psi_in_l = copy(ψl), psi_in_r = copy(ψr))
    @test eltype(mu_avg) <: Complex
    @test size(mu_avg) == (NC,)

    mu_all = KPM.kpm_1d(H, NC, NR; psi_in_l = copy(ψl), psi_in_r = copy(ψr),
                        avg_output = false)
    @test vec(sum(mu_all, dims = 1) ./ NR) ≈ mu_avg atol = 1e-12

    mu_ser = KPM.kpm_1d(H, NC, NR; psi_in_l = copy(ψl), psi_in_r = copy(ψr),
                        NR_parallel = false)
    @test mu_ser ≈ mu_avg atol = 1e-12

    # equal-vector path still returns real averages
    ψ = exp.(2im * pi * rand(rng, NH, NR))
    KPM.normalize_by_col(ψ, NR)
    mu_eq = KPM.kpm_1d(H, 16, NR; psi_in = ψ)
    @test eltype(mu_eq) <: Real
end
