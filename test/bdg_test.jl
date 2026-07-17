# Matrix-free reduced BdG operator and rescaling checks against dense ED.
using Test
using LinearAlgebra
using SparseArrays
using Random
using Serialization
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

    # bdg_assemble is the operator the GPU path runs on: it must reproduce
    # the matrix-free action exactly, including a general sparse pairing block.
    H_asm = KPM.bdg_assemble(BdG_op)
    @test H_asm isa SparseMatrixCSC{ComplexF64, Int}
    @test Matrix(H_asm) ≈ BdG_Hd atol=1e-12
    @test ishermitian(H_asm)

    bonds = [(1, 2), (3, 4)]
    channel = KPM.PairingChannel(bonds, 1.0, 1.0, :even)
    D_bond = KPM.pairing_matrix(BdG_N, [channel]; amplitude=0.25 + 0.1im)
    op_bond = KPM.BdGOperator(BdG_h; mu=BdG_mu, U=BdG_U, n=BdG_n, D=D_bond,
                              hole_convention=:conjugate)
    H_bond = KPM.bdg_assemble(op_bond)
    Hmf_bond = zeros(ComplexF64, 2BdG_N, 2BdG_N)
    for j in axes(Hmf_bond, 2)
        e_j = zeros(ComplexF64, 2BdG_N)
        e_j[j] = 1
        mul!(view(Hmf_bond, :, j), op_bond, e_j)
    end
    @test Matrix(H_bond) ≈ Hmf_bond atol=1e-12

    op_free = KPM.BdGOperator(KPM.ScaledOperator(BdG_h, 1.0, 0.0);
                              mu=BdG_mu, U=BdG_U, n=BdG_n, Delta=BdG_Delta,
                              h_hole=KPM.ScaledOperator(conj(BdG_h), 1.0, 0.0),
                              hole_convention=:conjugate)
    @test_throws ArgumentError KPM.bdg_assemble(op_free)
end

@testset "complex hopping intervalley opt-in and sparse constructor scaling" begin
    h_complex = ComplexF64[0 im; -im 0]
    n = zeros(2)
    Delta = ComplexF64[0.3 + 0.1im, 0.2 - 0.4im]
    @test_throws ArgumentError KPM.BdGOperator(
        h_complex; mu=0.2, U=0.0, n=n, Delta=Delta)

    op_complex = KPM.BdGOperator(
        h_complex; mu=0.2, U=0.0, n=n, Delta=Delta,
        assume_intervalley=true)
    Hmf = zeros(ComplexF64, 4, 4)
    for j in axes(Hmf, 2)
        e_j = zeros(ComplexF64, 4)
        e_j[j] = 1
        mul!(view(Hmf, :, j), op_complex, e_j)
    end
    @test Hmf ≈ bdg_matrix(h_complex, 0.2, zeros(2), n, Delta) atol=1e-12

    # Sanity only (no CI timing assertion): this exercises the O(nnz)
    # complex-real-valued sparse symmetry check at production-like size.
    Nlarge = 50_000
    h_large = spdiagm(0 => ones(ComplexF64, Nlarge))
    op_large = KPM.BdGOperator(
        h_large; mu=0.0, U=0.0, n=zeros(Nlarge),
        Delta=zeros(ComplexF64, Nlarge))
    @test op_large.N == Nlarge
end

@testset "self-consistency: fixed point, seeds, U=0, phase, restart" begin
    N = 4
    h, _, _ = ring_model(N; t=1.0)
    mu = -1.0
    U = fill(3.0, N)
    beta = 10.0
    n_initial = fill(0.5, N)
    Delta_initial = fill(0.1 + 0.0im, N)

    # Dense ED reference for the same linearly mixed fixed-point map.
    n_ed = copy(n_initial)
    Delta_ed = copy(Delta_initial)
    f = KPM.fermiFunctions(0.0, beta)
    res_delta = Inf
    res_n = Inf
    for _ in 1:10_000
        F = eigen(Hermitian(Matrix(bdg_matrix(h, mu, U, n_ed, Delta_ed))))
        occupations = f.(F.values)
        Delta_new = ComplexF64[-U[i] * sum(F.vectors[i, j] * conj(F.vectors[i + N, j]) * occupations[j]
                                           for j in eachindex(F.values)) for i in 1:N]
        n_new = Float64[2sum(abs2(F.vectors[i, j]) * occupations[j]
                              for j in eachindex(F.values)) for i in 1:N]
        res_delta = norm(Delta_new .- Delta_ed, Inf)
        res_n = norm(n_new .- n_ed, Inf)
        @. Delta_ed = 0.7 * Delta_ed + 0.3 * Delta_new
        @. n_ed = 0.7 * n_ed + 0.3 * n_new
        max(res_delta, res_n) < 1e-12 && break
    end
    @test max(res_delta, res_n) < 1e-12

    op_b = KPM.BdGOperator(h; mu=mu, U=U, n=n_initial, Delta=Delta_initial)
    res_b = KPM.bdg_solve!(op_b; beta=beta, NC=512, mix=0.3,
                            tol_delta=1e-9, tol_n=1e-9, maxiter=400)
    @test res_b.converged
    @test maximum(abs.(op_b.Δ .- sum(op_b.Δ) / N)) < 1e-8
    @test maximum(abs.(op_b.n .- sum(op_b.n) / N)) < 1e-8
    err_delta = maximum(abs.(abs.(op_b.Δ) .- abs.(Delta_ed)))
    err_n = abs(sum(op_b.n) / N - sum(n_ed) / N)
    println("SCF calibration: Delta=$(err_delta), n=$(err_n)")
    TOL_SCF_DELTA = 2e-3
    TOL_SCF_N = 2e-3
    @test abs.(op_b.Δ) ≈ abs.(Delta_ed) rtol=TOL_SCF_DELTA
    @test sum(op_b.n) / N ≈ sum(n_ed) / N rtol=TOL_SCF_N
    @test abs(op_b.Δ[1]) > 0.1

    op_seed = KPM.BdGOperator(h; mu=mu, U=U, n=n_initial,
                              Delta=fill(0.4 - 0.2im, N))
    res_seed = KPM.bdg_solve!(op_seed; beta=beta, NC=512, mix=0.15,
                               tol_delta=1e-9, tol_n=1e-9, maxiter=1200)
    @test res_seed.converged
    @test abs.(op_seed.Δ) ≈ abs.(op_b.Δ) atol=1e-6 rtol=0
    @test op_seed.n ≈ op_b.n atol=1e-6 rtol=0

    phi = 0.9
    op_phase = KPM.BdGOperator(h; mu=mu, U=U, n=n_initial,
                               Delta=exp(im * phi) .* Delta_initial)
    res_phase = KPM.bdg_solve!(op_phase; beta=beta, NC=512, mix=0.3,
                                tol_delta=1e-9, tol_n=1e-9, maxiter=400)
    @test res_phase.converged
    @test abs.(op_phase.Δ) ≈ abs.(op_b.Δ) atol=1e-6 rtol=0
    @test op_phase.n ≈ op_b.n atol=1e-6 rtol=0
    phase_error = angle(exp(im * (angle(op_phase.Δ[1]) - angle(op_b.Δ[1]) - phi)))
    @test phase_error ≈ 0.0 atol=1e-3

    op_u0 = KPM.BdGOperator(h; mu=mu, U=0.0, n=n_initial,
                            Delta=fill(0.3 + 0.4im, N))
    res_u0 = KPM.bdg_solve!(op_u0; beta=beta, NC=512, mix=0.3,
                             tol_delta=1e-10, tol_n=1e-9, maxiter=400)
    @test res_u0.converged
    @test norm(op_u0.Δ, Inf) < 1e-8

    mktempdir() do dir
        checkpoint = joinpath(dir, "bdg-checkpoint.bin")
        op_a = KPM.BdGOperator(h; mu=mu, U=U, n=n_initial, Delta=Delta_initial)
        KPM.bdg_solve!(op_a; beta=beta, NC=512, mix=0.3,
                       tol_delta=1e-14, tol_n=1e-14, maxiter=7,
                       checkpoint_path=checkpoint, checkpoint_every=7)
        state = open(deserialize, checkpoint)
        @test state.version == 2
        @test state.params.g_rho == 2.0

        op_bad_u = KPM.BdGOperator(h; mu=mu, U=U .+ 1, n=n_initial,
                                   Delta=Delta_initial)
        @test_throws ArgumentError KPM.bdg_restore!(op_bad_u, checkpoint)

        op_bad_target = KPM.BdGOperator(h; mu=mu, U=U, n=n_initial,
                                        Delta=Delta_initial)
        @test_throws ArgumentError KPM.bdg_solve!(
            op_bad_target; beta=beta, NC=512, mix=0.3,
            target_filling=0.6, mu_bracket=(-4.0, 4.0), restart=checkpoint)

        op_warn = KPM.BdGOperator(h; mu=mu, U=U, n=n_initial,
                                  Delta=Delta_initial)
        @test_logs (:warn, r"restart parameters differ.*NC") KPM.bdg_solve!(
            op_warn; beta=beta, NC=256, mix=0.3,
            tol_delta=1e-14, tol_n=1e-14, maxiter=1, restart=checkpoint)

        op_b_restart = KPM.BdGOperator(h; mu=mu, U=U, n=n_initial, Delta=Delta_initial)
        KPM.bdg_solve!(op_b_restart; beta=beta, NC=512, mix=0.3,
                       tol_delta=1e-14, tol_n=1e-14, maxiter=7, restart=checkpoint)
        op_c = KPM.BdGOperator(h; mu=mu, U=U, n=n_initial, Delta=Delta_initial)
        KPM.bdg_solve!(op_c; beta=beta, NC=512, mix=0.3,
                       tol_delta=1e-14, tol_n=1e-14, maxiter=14)
        if KPM.whichcore()
            # GPU kernels are not guaranteed bitwise-deterministic; the
            # bitwise checkpoint/restart contract holds on the CPU path.
            @test op_b_restart.Δ ≈ op_c.Δ atol=1e-10 rtol=0
            @test op_b_restart.n ≈ op_c.n atol=1e-10 rtol=0
        else
            @test op_b_restart.Δ == op_c.Δ
            @test op_b_restart.n == op_c.n
        end
    end

    op_filling = KPM.BdGOperator(h; mu=mu, U=U, n=n_initial, Delta=Delta_initial)
    res_filling = KPM.bdg_solve!(op_filling; beta=beta, NC=512, mix=0.3,
                                  target_filling=0.6, mu_bracket=(-4.0, 4.0),
                                  mu_tol=1e-3)
    @test res_filling.converged
    @test abs(sum(op_filling.n) / N - 0.6) <= 1e-3
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

    @test KPM.gershgorin_bound(BdG_op) >= exact

    Nedge = 20_000
    true_radius = 2.0
    h_edge = spdiagm(0 => vcat(true_radius, ones(Nedge - 1)))
    op_edge = KPM.BdGOperator(
        h_edge; mu=0.0, U=0.0, n=zeros(Nedge),
        Delta=zeros(ComplexF64, Nedge))
    rh_edge = KPM.rescale(op_edge)
    rh_edge_certified = KPM.rescale(op_edge; bound=:gershgorin)
    required_scale = true_radius / (1 - 0.2 / 2)
    println("isolated-edge scaling: power a=$(rh_edge.a), Gershgorin a=$(rh_edge_certified.a), required=$(required_scale)")
    @test rh_edge.a >= required_scale || rh_edge_certified.a >= required_scale

    Hs_too_small = KPM.ScaledOperator(op_edge, 0.5true_radius, 0.0)
    @test_throws ErrorException KPM.bdg_site_moments(
        Hs_too_small, Nedge, [1], 2; batch_size=1)

    op_zero = KPM.BdGOperator(
        spzeros(Float64, 2, 2); mu=0.0, U=0.0, n=zeros(2),
        Delta=zeros(ComplexF64, 2))
    @test_throws ArgumentError KPM.rescale(op_zero)
    @test_throws ArgumentError KPM.rescale(BdG_op; eps=0.0)
    @test_throws ArgumentError KPM.rescale(BdG_op; eps=2.0)
end

@testset "one-site analytic update (Eqs 26-27)" begin
    h1 = spzeros(Float64, 1, 1)
    mu = 0.3
    U = 1.5
    n = [0.0]
    Delta = [0.3 + 0.4im]
    beta = 5.0
    xi = -mu
    E = sqrt(xi^2 + abs2(only(Delta)))
    Delta_exact = U * only(Delta) * tanh(beta * E / 2) / (2E)
    n_exact = 1 - xi * tanh(beta * E / 2) / E

    op = KPM.BdGOperator(h1; mu=mu, U=U, n=n, Delta=Delta)
    rh = KPM.rescale(op)
    m128 = KPM.bdg_local_moments(rh; NC=128)
    m512 = KPM.bdg_local_moments(rh; NC=512)
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

    m512_g1 = KPM.bdg_local_moments(rh; NC=512, g_rho=1)
    n512_g1, Delta512_g1 = KPM.bdg_update(m512_g1; beta=beta)
    @test n512_g1 ≈ 0.5 .* n512 atol=1e-12 rtol=0
    @test Delta512_g1 ≈ Delta512 atol=1e-12 rtol=0
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
    H = bdg_matrix(h, mu, U, n, Delta)
    F = eigen(Hermitian(Matrix(H)))
    f = KPM.fermiFunctions(0.0, beta)
    occupations = f.(F.values)
    Delta_exact = ComplexF64[-U[i] * sum(F.vectors[i, j] * conj(F.vectors[i + N, j]) * occupations[j]
                                        for j in eachindex(F.values)) for i in 1:N]
    n_exact = Float64[2sum(abs2(F.vectors[i, j]) * occupations[j]
                           for j in eachindex(F.values)) for i in 1:N]

    op = KPM.BdGOperator(h; mu=mu, U=U, n=n, Delta=Delta)
    rh = KPM.rescale(op)
    m64 = KPM.bdg_local_moments(rh; NC=64, batch_size=3)
    m256 = KPM.bdg_local_moments(rh; NC=256, batch_size=3)
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
                                      U=op.U, beta=beta)
    @test n_raw == n256
    @test Delta_raw == Delta256
    @test maximum(abs.(n256 .- first(n256))) < 1e-10
    @test maximum(abs.(Delta256 .- first(Delta256))) < 1e-10
end

@testset "singlet hole convention (complex flux ring)" begin
    N = 4
    hf, posf, dispf = flux_ring_model(N; phi=0.35)
    mu = -0.4
    U = fill(2.0, N)
    n = fill(0.9, N)
    Delta = [0.4exp(0.3im), 0.4exp(1.1im),
             0.4exp(-0.7im), 0.4exp(2.0im)]
    Hs = bdg_matrix_singlet(hf, mu, U, n, Delta)
    op_s = KPM.BdGOperator(
        hf; mu=mu, U=U, n=n, Delta=Delta, hole_convention=:singlet)

    Hmf = zeros(ComplexF64, 2N, 2N)
    for j in axes(Hmf, 2)
        e_j = zeros(ComplexF64, 2N)
        e_j[j] = 1
        mul!(view(Hmf, :, j), op_s, e_j)
    end
    @test Hmf ≈ Hs atol=1e-12
    @test op_s.h_hole isa SparseMatrixCSC

    rng = Xoshiro(91)
    x = randn(rng, ComplexF64, 2N)
    y = randn(rng, ComplexF64, 2N)
    alpha = 1.2 - 0.3im
    beta_mul = -0.4 + 0.7im
    @test mul!(copy(y), op_s, x, alpha, beta_mul) ≈
          alpha * Hs * x + beta_mul * y atol=1e-12

    ev = sort(eigvals(Hermitian(Hs)))
    @test ev ≈ -reverse(ev) atol=1e-10
    op_i = KPM.BdGOperator(
        hf; mu=mu, U=U, n=n, Delta=Delta,
        hole_convention=:intervalley, assume_intervalley=true)
    Hi = zeros(ComplexF64, 2N, 2N)
    for j in axes(Hi, 2)
        e_j = zeros(ComplexF64, 2N)
        e_j[j] = 1
        mul!(view(Hi, :, j), op_i, e_j)
    end
    ev_i = sort(eigvals(Hermitian(Hi)))
    intervalley_pairing_mismatch = maximum(abs.(ev_i .+ reverse(ev_i)))
    println("flux-ring PH mismatch: singlet=$(maximum(abs.(ev .+ reverse(ev)))), intervalley=$(intervalley_pairing_mismatch)")
    @test intervalley_pairing_mismatch > 1e-3

    beta = 10.0
    F = eigen(Hermitian(Hs))
    occupations = KPM.fermiFunctions(0.0, beta).(F.values)
    Delta_exact = ComplexF64[
        -U[i] * sum(F.vectors[i, j] * conj(F.vectors[i + N, j]) * occupations[j]
                    for j in eachindex(F.values)) for i in 1:N]
    n_exact = Float64[
        2sum(abs2(F.vectors[i, j]) * occupations[j]
             for j in eachindex(F.values)) for i in 1:N]
    rh = KPM.rescale(op_s)
    # NC=512: at NC=256 the flux ring's sharp levels leave ~3e-3 kernel
    # broadening error; 512 comfortably meets the 2e-3 gate.
    moments = KPM.bdg_local_moments(rh; NC=512, batch_size=3)
    n_kpm, Delta_kpm = KPM.bdg_update(moments; beta=beta)
    update_delta_relerr = maximum(abs.(Delta_kpm .- Delta_exact)) / maximum(abs, Delta_exact)
    update_n_relerr = maximum(abs.(n_kpm .- n_exact)) / maximum(abs, n_exact)
    println("flux-ring singlet update: Delta relerr=$(update_delta_relerr), n relerr=$(update_n_relerr)")
    @test Delta_kpm ≈ Delta_exact rtol=2e-3
    @test n_kpm ≈ n_exact rtol=2e-3

    # SCF with a genuinely gapped fixed point: place mu on the xi = 0 level
    # (flux-ring single-particle level at -2 sin(phi) = -0.686, Hartree
    # -(U/2) n = -1.125 with the density frozen) so pairing is strong. At the
    # original mu = -0.4 no level sits near the Fermi energy and the true
    # fixed point is the normal state (Delta -> 0), which cannot anchor a
    # relative comparison. Density feedback is already exercised by the
    # real-ring SCF testset; here update_density=false keeps the level
    # placement exact.
    mu_scf = -2 * sin(0.35) - 1.125
    U_scf = fill(2.5, N)
    Delta_initial = fill(0.1 + 0.0im, N)
    op_scf = KPM.BdGOperator(
        hf; mu=mu_scf, U=U_scf, n=n, Delta=Delta_initial,
        hole_convention=:singlet)
    result = KPM.bdg_solve!(
        op_scf; beta=beta, NC=512, mix=0.3, update_density=false,
        tol_delta=1e-8, tol_n=1e-8, maxiter=800)
    @test result.converged
    @test maximum(abs.(abs.(op_scf.Δ) .- sum(abs, op_scf.Δ) / N)) < 1e-8
    @test abs(op_scf.Δ[1]) > 0.05

    Delta_ed = copy(Delta_initial)
    residual_ed = Inf
    for _ in 1:10_000
        F_ed = eigen(Hermitian(bdg_matrix_singlet(hf, mu_scf, U_scf, n, Delta_ed)))
        occupations_ed = KPM.fermiFunctions(0.0, beta).(F_ed.values)
        Delta_new = ComplexF64[
            -U_scf[i] * sum(F_ed.vectors[i, j] * conj(F_ed.vectors[i + N, j]) * occupations_ed[j]
                            for j in eachindex(F_ed.values)) for i in 1:N]
        residual_ed = norm(Delta_new .- Delta_ed, Inf)
        @. Delta_ed = 0.7 * Delta_ed + 0.3 * Delta_new
        residual_ed < 1e-12 && break
    end
    @test residual_ed < 1e-12
    scf_delta_relerr = maximum(abs.(abs.(op_scf.Δ) .- abs.(Delta_ed))) / maximum(abs, Delta_ed)
    println("flux-ring singlet SCF: |Delta_ed|=$(abs(Delta_ed[1])), Delta relerr=$(scf_delta_relerr)")
    @test abs.(op_scf.Δ) ≈ abs.(Delta_ed) rtol=2e-3

    @test_throws ArgumentError KPM.BdGOperator(
        hf; mu=mu, U=U, n=n, Delta=Delta,
        hole_convention=:intervalley, h_hole=conj(hf),
        assume_intervalley=true)
    hf_matrix_free = KPM.ScaledOperator(hf, 1.0, 0.0)
    @test_throws ArgumentError KPM.BdGOperator(
        hf_matrix_free; mu=mu, U=U, n=n, Delta=Delta,
        hole_convention=:singlet)
end

@testset "Anderson mixing: same fixed point, fewer iterations" begin
    N = 4
    h, _, _ = ring_model(N; t=1.0)
    mu = -1.0
    U = fill(3.0, N)
    beta = 10.0
    n_initial = fill(0.5, N)
    Delta_initial = fill(0.1 + 0.0im, N)

    op_lin = KPM.BdGOperator(h; mu=mu, U=U, n=n_initial, Delta=Delta_initial)
    res_lin = KPM.bdg_solve!(op_lin; beta=beta, NC=512, mix=0.3,
                             tol_delta=1e-9, tol_n=1e-9, maxiter=400)
    @test res_lin.converged

    op_and = KPM.BdGOperator(h; mu=mu, U=U, n=n_initial, Delta=Delta_initial)
    res_and = KPM.bdg_solve!(op_and; beta=beta, NC=512, mix=0.3,
                             mixing=:anderson, anderson_history=6,
                             tol_delta=1e-9, tol_n=1e-9, maxiter=400)
    println("mixing iterations: linear=$(res_lin.iterations), anderson=$(res_and.iterations)")
    @test res_and.converged
    @test abs.(op_and.Δ) ≈ abs.(op_lin.Δ) atol=1e-6 rtol=0
    @test op_and.n ≈ op_lin.n atol=1e-6 rtol=0
    @test res_and.iterations < res_lin.iterations

    # frozen-density path
    op_and_nd = KPM.BdGOperator(h; mu=mu, U=U, n=n_initial, Delta=Delta_initial)
    res_and_nd = KPM.bdg_solve!(op_and_nd; beta=beta, NC=512, mix=0.3,
                                mixing=:anderson, update_density=false,
                                tol_delta=1e-9, maxiter=400)
    @test res_and_nd.converged

    @test_throws ArgumentError KPM.bdg_solve!(
        KPM.BdGOperator(h; mu=mu, U=U, n=n_initial, Delta=Delta_initial);
        beta=beta, mixing=:bogus)
    @test_throws ArgumentError KPM.bdg_solve!(
        KPM.BdGOperator(h; mu=mu, U=U, n=n_initial, Delta=Delta_initial);
        beta=beta, mixing=:anderson, anderson_history=0)
end
