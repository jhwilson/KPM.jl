using Test
using LinearAlgebra
using SparseArrays
using Random
using KPM

# Unitary Chebyshev propagation e^{-iHt} = e^{-ibt} Σ (2-δ_n0)(-i)^n J_n(at) T_n.
# Anchored against dense ED propagation (ed_evolve); norm conservation,
# reversibility, and the group property double as the proof that no kernel
# damping is applied anywhere in the evolution path.

isdefined(@__MODULE__, :EDReference) || include("ed_reference.jl")
using .EDReference: ring_model, flux_ring_model, ed_evolve

# Deterministic physical-scale fixture: exact spectral bounds (dense eigvals),
# margin 0.95, so H = a·H_norm + b·I with genuine (a, b) and no Arpack call.
function _evolution_fixture(H)
    λ = eigvals(Hermitian(Matrix(H)))
    b = (maximum(λ) + minimum(λ)) / 2
    a = (maximum(λ) - minimum(λ)) / 2 / 0.95
    Hn = (H - b * I) / a
    return KPM.RescaledHamiltonian(Hn, a, b), Matrix(Hermitian(Matrix(H)))
end

@testset "dense exp(-iHt) comparison" begin
    rng = Xoshiro(191)
    N = 96
    ring_h, _, _ = ring_model(N)
    flux_h, _, _ = flux_ring_model(N)
    # + 0.7I forces a genuine center shift b ≈ 0.7 (e^{-ibt} phase is live)
    for H_phys in (sparse(ring_h + 0.7 * I), sparse(flux_h))
        h, Hd = _evolution_fixture(H_phys)
        ψ0 = randn(rng, ComplexF64, N)
        ψ0 ./= norm(ψ0)
        for t in (0.0, 0.7, 3.3, -1.9)
            @test KPM.evolve(h, ψ0, t) ≈ ed_evolve(Hd, ψ0, t) atol = 1e-10
        end
    end
end

@testset "center-shift phase isolation" begin
    rng = Xoshiro(193)
    N = 64
    ring_h, _, _ = ring_model(N)
    h, _ = _evolution_fixture(sparse(ring_h + 0.7 * I))
    h0 = KPM.RescaledHamiltonian(h.H, h.a, 0.0)
    ψ0 = randn(rng, ComplexF64, N)
    ψ0 ./= norm(ψ0)
    for t in (1.3, -2.1)
        @test KPM.evolve(h, ψ0, t) ≈ cis(-h.b * t) .* KPM.evolve(h0, ψ0, t) atol = 1e-13
    end
end

@testset "t = 0 identity and norm conservation" begin
    rng = Xoshiro(197)
    N = 64
    flux_h, _, _ = flux_ring_model(N)
    h, _ = _evolution_fixture(sparse(flux_h))
    ψ0 = randn(rng, ComplexF64, N)
    ψ0 ./= norm(ψ0)

    @test KPM.evolve(h, ψ0, 0.0) ≈ ψ0 atol = 1e-13

    for t in (0.5, 5.0, -11.0, 40.0)
        @test abs(norm(KPM.evolve(h, ψ0, t)) - 1) < 1e-11
    end
end

@testset "reversibility and group property" begin
    rng = Xoshiro(199)
    N = 64
    ring_h, _, _ = ring_model(N)
    h, _ = _evolution_fixture(sparse(ring_h + 0.7 * I))
    ψ0 = randn(rng, ComplexF64, N)
    ψ0 ./= norm(ψ0)

    for t in (0.9, 4.7)
        @test KPM.evolve(h, KPM.evolve(h, ψ0, t), -t) ≈ ψ0 atol = 1e-10
    end

    t1, t2 = 1.7, -0.6
    @test KPM.evolve(h, ψ0, t1 + t2) ≈ KPM.evolve(h, KPM.evolve(h, ψ0, t1), t2) atol = 1e-10
end

@testset "multi-time interface and output shapes" begin
    rng = Xoshiro(211)
    N = 48
    NR = 3
    flux_h, _, _ = flux_ring_model(N)
    h, Hd = _evolution_fixture(sparse(flux_h))
    ψ0 = randn(rng, ComplexF64, N)
    ψ0 ./= norm(ψ0)
    Ψ0 = randn(rng, ComplexF64, N, NR)
    foreach(j -> Ψ0[:, j] ./= norm(Ψ0[:, j]), 1:NR)
    ts = [-1.9, 0.0, 0.7, 3.3]
    NT = length(ts)

    # shape table for all four psi0 × time arities
    @test size(KPM.evolve(h, ψ0, 0.7)) == (N,)
    @test size(KPM.evolve(h, Ψ0, 0.7)) == (N, NR)
    @test size(KPM.evolve(h, ψ0, ts)) == (N, NT)
    @test size(KPM.evolve(h, Ψ0, ts)) == (N, NR, NT)

    # one shared recurrence must reproduce the independent scalar calls (the
    # shared call pads to the largest adaptive order, so agreement is limited
    # by the smaller call's dropped tail, ~2 tol)
    multi = KPM.evolve(h, Ψ0, ts)
    for (k, t) in enumerate(ts)
        @test multi[:, :, k] ≈ KPM.evolve(h, Ψ0, t) atol = 5e-12
    end

    # block states: columnwise scalar calls and dense ED on the block
    single = KPM.evolve(h, Ψ0, 3.3)
    for j in 1:NR
        @test single[:, j] ≈ KPM.evolve(h, Ψ0[:, j], 3.3) atol = 5e-12
    end
    @test single ≈ ed_evolve(Hd, Ψ0, 3.3) atol = 1e-10
end

@testset "large |at| with adaptive order" begin
    rng = Xoshiro(223)
    N = 512
    ring_h, _, _ = ring_model(N)
    h, Hd = _evolution_fixture(sparse(ring_h))
    t = 200.0 / h.a   # |a t| ≈ 200
    ψ0 = randn(rng, ComplexF64, N)
    ψ0 ./= norm(ψ0)

    NC_auto = KPM.evolution_order(h.a, t)
    @test 200 < NC_auto < 320

    ψt = KPM.evolve(h, ψ0, t)
    @test abs(norm(ψt) - 1) < 1e-10
    @test KPM.evolve(h, ψt, -t) ≈ ψ0 atol = 1e-9
    @test ψt ≈ ed_evolve(Hd, ψ0, t) atol = 1e-8
end

@testset "adaptive order and truncation warning" begin
    # order grows monotonically with |t| and shrinks with looser tol
    orders = [KPM.evolution_order(2.0, t) for t in (0.5, 2.0, 10.0, 50.0)]
    @test issorted(orders)
    @test KPM.evolution_order(2.0, 10.0; tol = 1e-6) <= KPM.evolution_order(2.0, 10.0)
    # t = 0 hits the NC_min floor (NC_min is the minimum *returned* order)
    @test KPM.evolution_order(2.0, 0.0) == 8
    @test KPM.evolution_order(2.0, 0.0; NC_min = 32) == 32

    # a deliberately truncated series warns and visibly loses norm
    rng = Xoshiro(227)
    N = 48
    ring_h, _, _ = ring_model(N)
    h, _ = _evolution_fixture(sparse(ring_h))
    t = 50.0 / h.a
    ψ0 = randn(rng, ComplexF64, N)
    ψ0 ./= norm(ψ0)
    C, NC_used, tail = @test_logs (:warn, r"evolution series tail") KPM.evolution_coefficients(h.a, h.b, [t]; NC = 8)
    @test NC_used == 8
    @test maximum(tail) > 1e-12
    ψt = @test_logs (:warn, r"evolution series tail") KPM.evolve(h, ψ0, t; NC = 8)
    @test abs(norm(ψt) - 1) > 1e-3
end

@testset "truncation-tolerance contract (scalar oracle)" begin
    # 1×1 H_norm = [x]: the exact evolution is the phase e^{-i(ax+b)t}, so the
    # measured error is pure truncation + recurrence roundoff. The adaptive
    # order must meet the default tol = 1e-12 outright (the tail sum is a
    # rigorous bound; the adversarial review's counterexamples at x = 0 are
    # included).
    for (x, at) in ((0.0, 1995.4), (0.0, 2989.2), (0.3, 500.0), (-0.7, 3000.0))
        a = 2.0
        t = at / a
        b = 0.4
        hx = KPM.RescaledHamiltonian(sparse(reshape([x], 1, 1)), a, b)
        exact = cis(-(a * x + b) * t)
        @test abs(KPM.evolve(hx, ComplexF64[1.0], t)[1] - exact) <= 1e-12
    end

    # the reported tail is the full dropped sum and bounds the actual error
    z = 1995.4
    NC = KPM.evolution_order(1.0, z)
    @test KPM.evolution_tail(z, NC) < 1e-12
    @test KPM.evolution_tail(z, NC) >= 2 * abs(KPM.besselj(NC, z))
end

@testset "threshold-adjacent truncation warning" begin
    # at NC = 3116, |at| = 2989.2 the two leading dropped terms alone are just
    # below 1e-12 but the full tail is above it — the warning must still fire
    # (the adversarial review's false-negative case)
    @test_logs (:warn, r"evolution series tail") KPM.evolution_coefficients(1.0, 0.0, [2989.2]; NC = 3116)
    # the adaptive order for the same time must be silent
    NC_ok = KPM.evolution_order(1.0, 2989.2)
    @test NC_ok > 3116
    @test_logs KPM.evolution_coefficients(1.0, 0.0, [2989.2]; NC = NC_ok)
end

@testset "order cap and overflow safety" begin
    # |at| beyond the cap (or overflowing to Inf) is an ArgumentError up
    # front, never an InexactError from integer conversion
    @test_throws ArgumentError KPM.evolution_order(1.0, 1e20)
    @test_throws ArgumentError KPM.evolution_order(1e308, 2.0)
    @test_throws ArgumentError KPM.evolution_order(1.0, 100.0; NC_cap = 50)
    @test_throws ArgumentError KPM.evolution_order(1.0, 1.0; NC_min = 100, NC_cap = 10)
    # NC_min == NC_cap is legal when the required order fits
    @test KPM.evolution_order(1.0, 0.0; NC_min = 8, NC_cap = 8) == 8
end

@testset "stability-guard rejection" begin
    rng = Xoshiro(229)
    N = 48
    ring_h, _, _ = ring_model(N)
    h, _ = _evolution_fixture(sparse(ring_h))
    bad = KPM.RescaledHamiltonian(1.2 .* h.H, h.a, h.b)
    ψ0 = randn(rng, ComplexF64, N)
    ψ0 ./= norm(ψ0)
    err = try
        KPM.evolve(bad, ψ0, 30.0 / h.a)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("Chebyshev recurrence is unstable", sprint(showerror, err))
end

@testset "argument validation" begin
    rng = Xoshiro(233)
    N = 32
    ring_h, _, _ = ring_model(N)
    h, _ = _evolution_fixture(sparse(ring_h))
    ψ0 = randn(rng, ComplexF64, N)

    @test_throws ArgumentError KPM.evolve(h, ψ0, NaN)
    @test_throws ArgumentError KPM.evolve(h, ψ0, Inf)
    @test_throws ArgumentError KPM.evolve(h, ψ0, Float64[])
    @test_throws ArgumentError KPM.evolve(h, randn(rng, ComplexF64, N + 1), 1.0)
    @test_throws ArgumentError KPM.evolve(h, ψ0, 1.0; tol = 0.0)
    @test_throws ArgumentError KPM.evolve(h, ψ0, 1.0; NC = 1)
    @test_throws ArgumentError KPM.evolution_order(-1.0, 1.0)
    @test_throws ArgumentError KPM.evolution_coefficients(0.0, 0.0, [1.0])
end
