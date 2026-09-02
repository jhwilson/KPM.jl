using Test
using LinearAlgebra
using Random
using SparseArrays
using KPM

function _blocking_model(rng, L)
    N = L^2
    H = zeros(ComplexF64, N, N)
    for y = 1:L, x = 1:L
        i = x + L * (y - 1)
        x < L && (H[i, i+1] = H[i+1, i] = -1)
        y < L && (H[i, i+L] = H[i+L, i] = -1)
    end
    A = randn(rng, ComplexF64, N, N)
    H .+= 0.03 .* Matrix(Hermitian(A))
    H ./= 1.2 * opnorm(H)
    B = randn(rng, ComplexF64, N, N)
    C = randn(rng, ComplexF64, N, N)
    Jα = Matrix(Hermitian(B))
    Jβ = Matrix(Hermitian(C))
    return H, Jα, Jβ
end

function _chebyshev_matrices(H, NC)
    N = size(H, 1)
    T = Vector{Matrix{ComplexF64}}(undef, NC)
    T[1] = Matrix{ComplexF64}(I, N, N)
    NC == 1 && return T
    T[2] = Matrix{ComplexF64}(H)
    for n = 3:NC
        T[n] = 2 .* H * T[n-1] .- T[n-2]
    end
    return T
end

function _kpm2d_dense_reference(H, Jα, Jβ, ψl, ψr, NC)
    NR = size(ψl, 2)
    T = _chebyshev_matrices(H, NC)
    μ = zeros(ComplexF64, NC, NC)
    for m = 1:NC, n = 1:NC, i = 1:NR
        μ[n, m] += dot(T[m] * ψl[:, i], Jβ * T[n] * Jα * ψr[:, i]) / NR
    end
    return μ
end

function _kpm3d_dense_reference(H, Jα, Jβ, Jγ, ψl, ψr, NC)
    NR = size(ψl, 2)
    T = _chebyshev_matrices(H, NC)
    μ = zeros(ComplexF64, NC, NC, NC)
    for n1 = 1:NC, n3 = 1:NC, n2 = 1:NC, i = 1:NR
        μ[n2, n3, n1] += dot(
            T[n3] * ψl[:, i],
            Jγ * T[n2] * Jβ * T[n1] * Jα * ψr[:, i],
        ) / NR
    end
    return μ
end

function _kpm2d_workspaces(NH, NR, K_left, K_right)
    return (
        ψ0r = zeros(ComplexF64, NH, NR),
        Jψ0r = zeros(ComplexF64, NH, NR),
        JTnHJψr = zeros(ComplexF64, NH, NR, K_right),
        ψall_r = zeros(ComplexF64, NH, NR, 3),
        ψ0l = zeros(ComplexF64, NH, NR),
        ψall_l = zeros(ComplexF64, NH, NR, K_left),
        ψw = zeros(ComplexF64, NH, NR),
        μblock = zeros(ComplexF64, K_right, K_left),
    )
end

@testset "blocked GEMM vs independent dense reference" begin
    rng = Xoshiro(801)
    NH, NR = 36, 3
    H, Jα, Jβ = _blocking_model(rng, 6)
    ψl = KPM.random_phase_vectors(rng, NH, NR)
    ψr = KPM.random_phase_vectors(rng, NH, NR)
    for NC in (17, 24)
        ref = _kpm2d_dense_reference(H, Jα, Jβ, ψl, ψr, NC)
        got = KPM.kpm_2d(
            sparse(H),
            sparse(Jα),
            sparse(Jβ),
            NC,
            NR,
            NH;
            psi_in_l = ψl,
            psi_in_r = ψr,
            arr_size = 5,
            right_block = 3,
        )
        @test got ≈ ref rtol = 1e-12 atol = 1e-12
    end
end

@testset "blocking invariance" begin
    rng = Xoshiro(802)
    NH, NR, NC = 36, 3, 17
    H, Jα, Jβ = _blocking_model(rng, 6)
    ψ = KPM.random_phase_vectors(rng, NH, NR)
    baseline = KPM.kpm_2d(H, Jα, Jβ, NC, NR, NH; psi_in = ψ, arr_size = 3, right_block = 3)
    for K_left in (2, 3, 5, NC, NC + 7), K_right in (1, 3, 16, NC, NC + 7)
        got = KPM.kpm_2d(
            H,
            Jα,
            Jβ,
            NC,
            NR,
            NH;
            psi_in = ψ,
            arr_size = K_left,
            right_block = K_right,
        )
        @test got ≈ baseline rtol = 1e-12 atol = 1e-12
    end
end

@testset "blocking choices" begin
    NH, NR, NC = 40, 3, 50
    one_block = NH * NR * sizeof(ComplexF64)
    choice = KPM.kpm_2d_blocking(
        NH,
        NR,
        NC;
        workspace_bytes = 30 * one_block,
        right_block = 4,
    )
    # footprint = (K_right + 7) blocks + K_left * (block + K_right * 16 B):
    # base = 11 blocks = 21120 B, per_left = 1920 + 64 = 1984 B,
    # fld(57600 - 21120, 1984) = 18
    @test choice.arr_size == 18
    @test choice.right_block == 4
    @test choice.bytes == 11 * one_block + 18 * (one_block + 4 * sizeof(ComplexF64))
    @test choice.bytes <= 30 * one_block
    @test_throws ArgumentError KPM.kpm_2d_blocking(
        NH,
        NR,
        NC;
        workspace_bytes = 10 * one_block - 1,
    )
    explicit = KPM.kpm_2d_blocking(
        NH,
        NR,
        NC;
        workspace_bytes = 10 * one_block,
        arr_size = 27,
        right_block = 9,
    )
    @test explicit.arr_size == 27
    @test explicit.right_block == 9
end

@testset "parity and mn_sym semantics" begin
    rng = Xoshiro(803)
    NH, NR, NC = 36, 3, 17
    H, Jα, Jβ = _blocking_model(rng, 6)
    ψ = KPM.random_phase_vectors(rng, NH, NR)
    full = KPM.kpm_2d(H, Jα, Jβ, NC, NR, NH; psi_in = ψ, arr_size = 5, right_block = 3)
    odd = KPM.kpm_2d(
        H,
        Jα,
        Jβ,
        NC,
        NR,
        NH;
        psi_in = ψ,
        arr_size = 5,
        right_block = 3,
        moment_parity = :ODD,
    )
    even = KPM.kpm_2d(
        H,
        Jα,
        Jβ,
        NC,
        NR,
        NH;
        psi_in = ψ,
        arr_size = 5,
        right_block = 3,
        moment_parity = :EVEN,
    )
    odd_mask = [isodd(n + m) for n = 1:NC, m = 1:NC]
    @test odd ≈ full .* odd_mask rtol = 1e-12 atol = 1e-12
    @test even ≈ full .* .!odd_mask rtol = 1e-12 atol = 1e-12
    @test odd + even ≈ full rtol = 1e-12 atol = 1e-12

    sym_ref = copy(full)
    for m = 1:NC, n = (m+1):NC
        sym_ref[m, n] = real(sym_ref[m, n])
        sym_ref[n, m] = sym_ref[m, n]
    end
    sym = KPM.kpm_2d(
        H,
        Jα,
        Jβ,
        NC,
        NR,
        NH;
        psi_in = ψ,
        arr_size = 5,
        right_block = 3,
        mn_sym = true,
    )
    @test sym ≈ sym_ref rtol = 1e-12 atol = 1e-12
end

@testset "kpm_3d dense reference" begin
    rng = Xoshiro(804)
    NH, NR, NC = 12, 2, 6
    A = randn(rng, ComplexF64, NH, NH)
    H = Matrix(Hermitian(A))
    H ./= 1.2 * opnorm(H)
    Jα = Matrix(Hermitian(randn(rng, ComplexF64, NH, NH)))
    Jβ = Matrix(Hermitian(randn(rng, ComplexF64, NH, NH)))
    Jγ = Matrix(Hermitian(randn(rng, ComplexF64, NH, NH)))
    ψl = KPM.random_phase_vectors(rng, NH, NR)
    ψr = KPM.random_phase_vectors(rng, NH, NR)
    ref = _kpm3d_dense_reference(H, Jα, Jβ, Jγ, ψl, ψr, NC)
    got = KPM.kpm_3d(
        H,
        Jα,
        Jβ,
        Jγ,
        NC,
        NR,
        NH;
        psi_in_l = ψl,
        psi_in_r = ψr,
        arr_size = 3,
        right_block = 3,
    )
    @test got ≈ ref rtol = 1e-12 atol = 1e-12
end

@testset "kpm_1d and current dense references" begin
    rng = Xoshiro(805)
    NH, NR, NC = 20, 3, 24
    A = randn(rng, ComplexF64, NH, NH)
    H = Matrix(Hermitian(A))
    H ./= 1.2 * opnorm(H)
    J = Matrix(Hermitian(randn(rng, ComplexF64, NH, NH)))
    ψ = KPM.random_phase_vectors(rng, NH, NR)
    ψl = KPM.random_phase_vectors(rng, NH, NR)
    ψr = KPM.random_phase_vectors(rng, NH, NR)
    T = _chebyshev_matrices(H, NC)

    ref_equal = [dot(ψ[:, i], T[n] * ψ[:, i]) for i = 1:NR, n = 1:NC]
    got_equal = KPM.kpm_1d(H, NC, NR, NH; psi_in = ψ, avg_output = false)
    @test got_equal ≈ ref_equal rtol = 1e-12 atol = 1e-12

    ref_lr = [dot(ψl[:, i], T[n] * ψr[:, i]) for i = 1:NR, n = 1:NC]
    got_lr = KPM.kpm_1d(
        H,
        NC,
        NR,
        NH;
        psi_in_l = ψl,
        psi_in_r = ψr,
        avg_output = false,
    )
    @test got_lr ≈ ref_lr rtol = 1e-12 atol = 1e-12

    ref_current = [dot(J * ψ[:, i], T[n] * ψ[:, i]) for i = 1:NR, n = 1:NC]
    got_current = KPM.kpm_1d_current(H, J, NC, NR, NH; psi_in = ψ, avg_output = false)
    @test got_current ≈ ref_current rtol = 1e-12 atol = 1e-12
end

function _allocated_kpm2d!(H, J, μ, ψ, ws, NC, NR, NH)
    return @allocated KPM.kpm_2d!(
        H,
        J,
        J,
        NC,
        NR,
        NH,
        μ,
        ψ;
        ws...,
        arr_size = size(ws.ψall_l, 3),
        right_block = size(ws.JTnHJψr, 3),
    )
end

# CPU-only: on the GPU device @allocated also counts host staging buffers.
KPM.whichcore() || @testset "supplied-workspace allocation guard" begin
    # With K_left = 2 and K_right = 1 the number of block pairs is NC²/2, so
    # any per-block (or per-step) allocation shows up as growth in the
    # allocation count between NC = 32 and NC = 64; the per-call overhead
    # (view arrays, kwargs) is NC-independent up to the μ-view maps.
    NH, NR = 4096, 2
    H = spdiagm(-1 => fill(0.1, NH - 1), 0 => fill(0.2, NH), 1 => fill(0.1, NH - 1))
    J = spdiagm(0 => ones(NH))
    ψ = KPM.random_phase_vectors(Xoshiro(806), NH, NR)
    ws = _kpm2d_workspaces(NH, NR, 2, 1)
    alloc = Dict{Int,Int}()
    for NC in (32, 64)
        μ = zeros(ComplexF64, NC, NC)
        KPM.kpm_2d!(H, J, J, NC, NR, NH, μ, ψ; ws...)
        alloc[NC] = _allocated_kpm2d!(H, J, μ, ψ, ws, NC, NR, NH)
    end
    @test alloc[64] < 1_000_000
    # 512 vs 2048 block pairs: a 100-byte per-pair leak would add ~150 KB.
    # SparseArrays on Julia 1.10 allocates ~64 B inside every 5-arg sparse
    # mul! (64 B × 3072 extra recurrence steps = 196608 B here); the
    # per-block guard is only meaningful where the stdlib kernel is
    # allocation-free (Julia ≥ 1.11).
    VERSION >= v"1.11" && @test alloc[64] - alloc[32] < 65_536
end
