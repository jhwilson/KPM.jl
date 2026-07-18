using Test
using LinearAlgebra
using SparseArrays
using Random
using KPM

# The coefficient-accumulating matrix-function action
# out[:, :, k] = Σ_n C[n, k] T_{n-1}(Hn) V. Checks are exact: direct
# three-term recurrence at 1e-12, ED spectral sums at 1e-8. Coefficients are
# used verbatim (no hn, no kernel), so references carry no extra factors.

function _prescaled_hermitian_action(rng, NH; margin = 0.95)
    A = randn(rng, ComplexF64, NH, NH)
    Hd = Matrix((A + A') / 2)
    a = maximum(abs, eigvals(Hermitian(Hd))) / margin
    Hd ./= a
    return Hd, sparse(Hd)
end

# Reference: dense column-by-column three-term recurrence.
function _action_reference(Hd, V, C)
    NH, NR = size(V)
    NC, K = size(C)
    out = zeros(ComplexF64, NH, NR, K)
    for j in 1:NR
        α_prev = ComplexF64.(V[:, j])
        α_curr = Hd * α_prev
        for k in 1:K
            out[:, j, k] .+= C[1, k] .* α_prev
            NC >= 2 && (out[:, j, k] .+= C[2, k] .* α_curr)
        end
        for n in 3:NC
            α_prev, α_curr = α_curr, 2 .* (Hd * α_curr) .- α_prev
            for k in 1:K
                out[:, j, k] .+= C[n, k] .* α_curr
            end
        end
    end
    return out
end

@testset "linear combination vs direct recurrence and ED" begin
    rng = Xoshiro(151)
    NH, NC, NR, K = 48, 16, 2, 3
    Hd, Hs = _prescaled_hermitian_action(rng, NH)
    V = randn(rng, ComplexF64, NH, NR)
    C = randn(rng, ComplexF64, NC, K)

    ref = _action_reference(Hd, V, C)
    @test KPM.chebyshev_action(Hd, V, C) ≈ ref atol = 1e-12
    @test KPM.chebyshev_action(Hs, V, C) ≈ ref atol = 1e-12

    # ED: Σ_n C[n,k] T_{n-1}(λ) in the eigenbasis
    λ, U = eigen(Hermitian(Hd))
    θ = acos.(clamp.(λ, -1, 1))
    for k in 1:K
        f = [sum(C[n + 1, k] * cos(n * θi) for n in 0:(NC - 1)) for θi in θ]
        @test KPM.chebyshev_action(Hd, V, C)[:, :, k] ≈ U * (f .* (U' * V)) atol = 1e-8
    end
end

@testset "eltype promotion: real coefficients and real probes" begin
    rng = Xoshiro(157)
    NH, NC, NR = 40, 12, 2
    Hd, Hs = _prescaled_hermitian_action(rng, NH)
    Vc = randn(rng, ComplexF64, NH, NR)
    Vr = randn(rng, NH, NR)
    Cr = randn(rng, NC)
    Cc = randn(rng, ComplexF64, NC)

    # real C × complex V and complex C × real V both give complex output
    @test KPM.chebyshev_action(Hs, Vc, Cr) ≈
          _action_reference(Hd, Vc, reshape(complex.(Cr), :, 1))[:, :, 1] atol = 1e-12
    @test KPM.chebyshev_action(Hs, Vr, Cc) ≈
          _action_reference(Hd, complex.(Vr), reshape(Cc, :, 1))[:, :, 1] atol = 1e-12

    # real-symmetric sparse H with the same contract
    Hr = real.(Hd)
    a = maximum(abs, eigvals(Symmetric(Hr))) / 0.95
    Hr ./= a
    @test KPM.chebyshev_action(sparse(Hr), Vr, Cc) ≈
          _action_reference(complex.(Hr), complex.(Vr), reshape(Cc, :, 1))[:, :, 1] atol = 1e-12
end

@testset "vector / K=1 / allocating / preallocated-slots equivalence" begin
    rng = Xoshiro(163)
    NH, NC, NR, K = 32, 10, 3, 2
    Hd, Hs = _prescaled_hermitian_action(rng, NH)
    V = randn(rng, ComplexF64, NH, NR)
    C = randn(rng, ComplexF64, NC, K)

    full = KPM.chebyshev_action(Hs, V, C)
    # vector-C method == first coefficient column
    @test KPM.chebyshev_action(Hs, V, C[:, 1]) ≈ full[:, :, 1] atol = 1e-14
    # vector-V methods drop the probe axis
    @test KPM.chebyshev_action(Hs, V[:, 1], C) ≈ full[:, 1, :] atol = 1e-14
    @test KPM.chebyshev_action(Hs, V[:, 1], C[:, 1]) ≈ full[:, 1, 1] atol = 1e-14

    # in-place matrix form and preallocated slots reproduce the allocating call
    out = zeros(ComplexF64, NH, NR, K)
    slots = (zeros(ComplexF64, NH, NR), zeros(ComplexF64, NH, NR))
    KPM.chebyshev_action!(out, Hs, V, C; slots = slots)
    @test out ≈ full atol = 1e-14
    out2 = zeros(ComplexF64, NH, NR)
    KPM.chebyshev_action!(out2, Hs, V, C[:, 2])
    @test out2 ≈ full[:, :, 2] atol = 1e-14

    # SubArray probes are materialized, not mutated
    Vbig = randn(rng, ComplexF64, NH, NR + 2)
    Vview = view(Vbig, :, 2:(NR + 1))
    Vcopy = copy(Vview)
    @test KPM.chebyshev_action(Hs, Vview, C) ≈ KPM.chebyshev_action(Hs, Vcopy, C) atol = 1e-14
    @test Vview == Vcopy
end

@testset "zero padding is exact" begin
    rng = Xoshiro(167)
    NH, NC, NR, K = 32, 9, 2, 2
    Hd, Hs = _prescaled_hermitian_action(rng, NH)
    V = randn(rng, ComplexF64, NH, NR)
    C = randn(rng, ComplexF64, NC, K)
    Cpad = vcat(C, zeros(ComplexF64, 8, K))
    @test KPM.chebyshev_action(Hs, V, Cpad) ≈ KPM.chebyshev_action(Hs, V, C) atol = 1e-14
end

@testset "accumulator branch coverage across K" begin
    rng = Xoshiro(173)
    NH, NC, NR = 24, 8, 2
    Hd, Hs = _prescaled_hermitian_action(rng, NH)
    V = randn(rng, ComplexF64, NH, NR)
    # K = 1 and K > Threads.nthreads() exercise the finer_mt_ and mt_ paths
    for K in (1, Threads.nthreads() + 1)
        C = randn(rng, ComplexF64, NC, K)
        @test KPM.chebyshev_action(Hs, V, C) ≈ _action_reference(Hd, V, C) atol = 1e-12
    end
end

@testset "stability guard" begin
    rng = Xoshiro(179)
    NH, NR = 32, 2
    Hd, Hs = _prescaled_hermitian_action(rng, NH)
    V = randn(rng, ComplexF64, NH, NR)
    C = randn(rng, ComplexF64, 64, 1)

    # spectrum outside (-1, 1): the recurrence grows and must be rejected
    err = try
        KPM.chebyshev_action(1.2 .* Hs, V, C)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("Chebyshev recurrence is unstable", sprint(showerror, err))
    # check_every=0 disables the guard (result is garbage but the call runs)
    @test KPM.chebyshev_action(1.2 .* Hs, V, C; check_every = 0) isa Array

    # the guard is relative to seed norms: large unnormalized probes are fine
    Vbig = 100 .* V
    @test KPM.chebyshev_action(Hs, Vbig, C) ≈
          100 .* KPM.chebyshev_action(Hs, V, C) atol = 1e-9
end

@testset "argument validation" begin
    rng = Xoshiro(181)
    NH, NC, NR, K = 24, 8, 2, 2
    Hd, Hs = _prescaled_hermitian_action(rng, NH)
    V = randn(rng, ComplexF64, NH, NR)
    C = randn(rng, ComplexF64, NC, K)

    # wrong probe row count
    @test_throws ArgumentError KPM.chebyshev_action(Hs, randn(rng, ComplexF64, NH + 1, NR), C)
    # output shape mismatches
    @test_throws ArgumentError KPM.chebyshev_action!(zeros(ComplexF64, NH, NR, K + 1), Hs, V, C)
    @test_throws ArgumentError KPM.chebyshev_action!(zeros(ComplexF64, NH, NR + 1, K), Hs, V, C)
    # empty coefficient table
    @test_throws ArgumentError KPM.chebyshev_action(Hs, V, zeros(ComplexF64, 0, K))
    # negative guard cadence
    @test_throws ArgumentError KPM.chebyshev_action(Hs, V, C; check_every = -1)
    # aliased out/V
    Vc = randn(rng, ComplexF64, NH, NR)
    @test_throws ArgumentError KPM.chebyshev_action!(reshape(Vc, NH, NR, 1), Hs, Vc, C[:, 1:1])
    # aliased or malformed slots
    w = zeros(ComplexF64, NH, NR)
    @test_throws ArgumentError KPM.chebyshev_action!(zeros(ComplexF64, NH, NR, 1), Hs, V,
                                                     C[:, 1:1]; slots = (w, w))
    @test_throws ArgumentError KPM.chebyshev_action!(zeros(ComplexF64, NH, NR, 1), Hs, V,
                                                     C[:, 1:1];
                                                     slots = (w, zeros(ComplexF64, NH + 1, NR)))
end
