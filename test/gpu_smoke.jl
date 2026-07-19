# GPU validation of the CUDA extension — run manually on a CUDA machine:
#   julia --project=<env with KPM + CUDA + Arpack> test/gpu_smoke.jl
#
# Not part of the CI suite (needs a functional GPU). Checks that every GPU
# code path produces the same physics as the CPU path on the same inputs:
#   1. kpm_1d moments, GPU vs CPU (deterministic probes)
#   2. DoS reconstruction on the device
#   3. kpm_2d moments + kubo_bastin_cond: quantized Hall plateau from GPU moments
#   4. dc_long
#   5. chebyshev_action + evolve (matrix-function action, unitary evolution,
#      device-resident probe seeding)
#   6. Fermi projector + local Chern marker (deterministic and stochastic
#      regional modes) on the open Haldane flake, GPU vs CPU and vs FHS
#   7. BdG: assembled device operator, onsite + pairing-channel SCF, and
#      superfluid stiffness, each GPU vs CPU
#   8. timing comparisons (informational)

using Test
using LinearAlgebra
using SparseArrays
using Random
using CUDA
using KPM

@assert CUDA.functional() "gpu_smoke.jl needs a functional CUDA GPU"
@assert KPM.whichcore() "CUDA extension did not activate the GPU device"
println(
    "== GPU: ",
    CUDA.name(CUDA.device()),
    " | ",
    Base.format_bytes(CUDA.total_memory()),
    " ==",
)

include("ed_reference.jl")
using .EDReference

# helper: evaluate f() with the device temporarily forced to CPU
function on_cpu(f)
    dev = KPM.ACTIVE_DEVICE[]
    KPM.ACTIVE_DEVICE[] = KPM.CPUDevice()
    try
        return f()
    finally
        KPM.ACTIVE_DEVICE[] = dev
    end
end

@testset "kpm_1d: GPU moments == CPU moments" begin
    rng = Xoshiro(42)
    NH = 4096
    NC = 256
    NR = 8
    H = spdiagm(1 => fill(-0.45 + 0im, NH - 1), -1 => fill(-0.45 + 0im, NH - 1))
    psi_in = exp.(2im * pi * rand(rng, NH, NR))
    KPM.normalize_by_col(psi_in, NR)

    mu_gpu = KPM.kpm_1d(H, NC, NR; psi_in = copy(psi_in))
    mu_cpu = on_cpu(() -> KPM.kpm_1d(H, NC, NR; psi_in = copy(psi_in)))
    @test mu_gpu ≈ mu_cpu atol = 1e-8
end

@testset "kpm_1d left/right moments: GPU == CPU (incl. serial wrapper)" begin
    rng = Xoshiro(43)
    NH = 4096
    NC = 257   # odd: the non-doubling left/right recurrence must accept it
    NR = 4
    H = spdiagm(1 => fill(-0.45 + 0im, NH - 1), -1 => fill(-0.45 + 0im, NH - 1))
    ψl = exp.(2im * pi * rand(rng, NH, NR))
    ψr = exp.(2im * pi * rand(rng, NH, NR))
    KPM.normalize_by_col(ψl, NR)
    KPM.normalize_by_col(ψr, NR)

    mu_gpu = zeros(ComplexF64, NR, NC)
    KPM.kpm_1d!(H, NC, NR, NH, mu_gpu, ψl, ψr)
    mu_cpu = on_cpu(() -> begin
        mu = zeros(ComplexF64, NR, NC)
        KPM.kpm_1d!(H, NC, NR, NH, mu, ψl, ψr)
        mu
    end)
    @test mu_gpu ≈ mu_cpu atol = 1e-8

    # serial wrapper path feeds one-column host SubArrays into the device
    # workspaces — the adversarial-review hazard; must match the batched path
    mu_ser =
        KPM.kpm_1d(H, 32, NR; psi_in_l = copy(ψl), psi_in_r = copy(ψr), NR_parallel = false)
    mu_par = KPM.kpm_1d(H, 32, NR; psi_in_l = copy(ψl), psi_in_r = copy(ψr))
    @test mu_ser ≈ mu_par atol = 1e-10

    # typed layer end to end on the device: site-diagonal LDOS moments and a
    # greens reconstruction, GPU vs CPU
    h = KPM.rescale(H)
    sites = [1, 7, NH ÷ 2, NH]
    m_gpu = KPM.ldos_moments(h; sites = sites, NC = 256, batch_size = 3)
    m_cpu = on_cpu(() -> KPM.ldos_moments(h; sites = sites, NC = 256, batch_size = 3))
    @test m_gpu.mu ≈ m_cpu.mu atol = 1e-8
    E = collect(range(-0.8 * h.a, 0.8 * h.a; length = 21))
    @test KPM.ldos(m_gpu, E; eta = 0.05 * h.a) ≈ KPM.ldos(m_cpu, E; eta = 0.05 * h.a) atol =
        1e-8
end

@testset "dos: device reconstruction matches host" begin
    rng = Xoshiro(7)
    NH = 4096
    NC = 256
    H = spdiagm(1 => fill(-0.45 + 0im, NH - 1), -1 => fill(-0.45 + 0im, NH - 1))
    psi_in = exp.(2im * pi * rand(rng, NH, 16))
    KPM.normalize_by_col(psi_in, 16)
    mu = on_cpu(() -> KPM.kpm_1d(H, NC, 16; psi_in = psi_in))

    E_g, rho_g = KPM.dos(mu, 1.0; N_tilde = 512)
    E_c, rho_c = on_cpu(() -> KPM.dos(mu, 1.0; N_tilde = 512))
    @test rho_g ≈ rho_c atol = 1e-8
    dE = E_g[2] - E_g[1]
    @test isapprox(sum(rho_g) * dE, 1.0; atol = 2e-2)   # sum rule on device
end

@testset "quantum Hall from GPU kpm_2d moments" begin
    H, Jx, Jy, area = haldane_model(6, 6; t = 1.0, t2 = 0.2, ϕ = π/2, m = 0.0)
    D = size(H, 1)
    a, b, H_norm = KPM.normalizeH(H; center = true)
    NC = 96
    psi = Matrix{ComplexF64}(I, D, D)         # deterministic full trace

    mu_gpu = KPM.kpm_2d(H_norm, Jx, Jy, NC, D, D; psi_in = psi)
    mu_cpu = on_cpu(() -> KPM.kpm_2d(H_norm, Jx, Jy, NC, D, D; psi_in = psi))
    @test mu_gpu ≈ mu_cpu atol = 1e-7

    σxy_gpu = KPM.kubo_bastin_cond(mu_gpu, a, 0.0; b = b, NH = D, area = area)
    σxy_ed = ed_hall_conductivity_T0(H, Jx, Jy, area; Ef = 0.0)
    @test σxy_gpu ≈ σxy_ed atol = 0.03        # quantized plateau, GPU end to end
end

@testset "dc_long on GPU" begin
    rng = Xoshiro(11)
    NH = 2048
    H = spdiagm(1 => fill(-0.45 + 0im, NH - 1), -1 => fill(-0.45 + 0im, NH - 1))
    # bond current for the chain: (Jx)_ij = H_ij (i - j)
    Jx = spdiagm(1 => fill(-0.45 + 0im, NH - 1), -1 => fill(0.45 + 0im, NH - 1))
    psi_in = exp.(2im * pi * rand(rng, NH, 4))
    KPM.normalize_by_col(psi_in, 4)

    c_gpu = KPM.dc_long(H, Jx, 1.0, [64, 128], 4, NH; psi_in = copy(psi_in))
    c_cpu = on_cpu(() -> KPM.dc_long(H, Jx, 1.0, [64, 128], 4, NH; psi_in = copy(psi_in)))
    @test c_gpu ≈ c_cpu atol = 1e-6
end

@testset "chebyshev_action + evolve: GPU == CPU" begin
    rng = Xoshiro(13)
    NH = 4096
    NC = 256
    NR = 3
    H = spdiagm(1 => fill(-0.45 + 0im, NH - 1), -1 => fill(-0.45 + 0im, NH - 1))
    V = randn(rng, ComplexF64, NH, NR)
    C = randn(rng, ComplexF64, NC, 3)

    # the action follows the residence of Hn: device CSR vs host sparse
    Hn_dev = KPM.maybe_to_device(H, ComplexF64)
    act_gpu = Array(KPM.chebyshev_action(Hn_dev, V, C))
    act_cpu = on_cpu(() -> KPM.chebyshev_action(H, V, C))
    @test act_gpu ≈ act_cpu atol = 1e-8

    # device-resident probe block: the direct-assignment seeding path must
    # reproduce the host-seeded result
    act_dev_v = Array(KPM.chebyshev_action(Hn_dev, CuArray{ComplexF64}(V), C))
    @test act_dev_v ≈ act_cpu atol = 1e-8

    # typed evolve: scalar time and shared-recurrence time grid
    # (H's spectrum is ±0.9 cos k ⊂ (−1, 1), so it serves directly as H_norm)
    h = KPM.RescaledHamiltonian(H, 2.0, 0.3)
    ψ0 = randn(rng, ComplexF64, NH)
    ψ0 ./= norm(ψ0)
    ts = [1.0, 5.0, -3.0]
    ev_gpu = KPM.evolve(h, ψ0, 5.0)
    ev_cpu = on_cpu(() -> KPM.evolve(h, ψ0, 5.0))
    @test ev_gpu ≈ ev_cpu atol = 1e-8
    @test abs(norm(ev_gpu) - 1) < 1e-10
    evs_gpu = KPM.evolve(h, ψ0, ts)
    evs_cpu = on_cpu(() -> KPM.evolve(h, ψ0, ts))
    @test evs_gpu ≈ evs_cpu atol = 1e-8
end

@testset "Fermi projector and Chern marker: GPU == CPU" begin
    Lx = Ly = 12
    NC = 512
    H, pos, Ac = haldane_open_model(Lx, Ly; t = 1.0, t2 = 0.2, ϕ = π/2, m = 0.0)
    ev = eigvals(Hermitian(Matrix(H)))
    b = (maximum(ev) + minimum(ev)) / 2
    a = (maximum(ev) - minimum(ev)) / 2 / 0.95
    h = KPM.RescaledHamiltonian((H - b * I) ./ a, a, b)
    x, y = pos[:, 1], pos[:, 2]

    # projector action on a block
    V = randn(Xoshiro(17), ComplexF64, size(H, 1), 3)
    PV_gpu = KPM.fermi_projector(h, V; Ef = 0.0, NC = NC)
    PV_cpu = on_cpu(() -> KPM.fermi_projector(h, V; Ef = 0.0, NC = NC))
    @test PV_gpu ≈ PV_cpu atol = 1e-8

    # deterministic marker over the central 4×4 cells, and the FHS anchor
    # end to end on the device
    bulk = Int[]
    for cx = 5:8, cy = 5:8
        c = cy + Ly * (cx - 1)
        push!(bulk, 2 * c - 1, 2 * c)
    end
    mk_gpu = KPM.chern_marker(h, x, y; Ef = 0.0, sites = bulk, NC = NC)
    mk_cpu = on_cpu(() -> KPM.chern_marker(h, x, y; Ef = 0.0, sites = bulk, NC = NC))
    @test mk_gpu ≈ mk_cpu atol = 1e-8
    C = round(chern_number_fhs(haldane_bloch(; t = 1.0, t2 = 0.2, ϕ = π/2, m = 0.0)))
    @test KPM.chern_marker_average(mk_gpu; area = 16 * Ac) ≈ C atol = 0.1

    # stochastic regional estimator: identical seed ⇒ identical probes
    est_gpu = KPM.chern_marker_region(
        h,
        x,
        y;
        Ef = 0.0,
        region = bulk,
        rng = Xoshiro(31),
        NR = 8,
        NC = NC,
    )
    est_cpu = on_cpu(
        () -> KPM.chern_marker_region(
            h,
            x,
            y;
            Ef = 0.0,
            region = bulk,
            rng = Xoshiro(31),
            NR = 8,
            NC = NC,
        ),
    )
    @test est_gpu ≈ est_cpu atol = 1e-8
end

square_site(ix, iy, Lx, Ly) = mod1(ix, Lx) + (mod1(iy, Ly) - 1) * Lx

@testset "BdG operator moves to the device as an assembled matrix" begin
    h, pos, disp = square_model(4, 4)
    op = KPM.BdGOperator(
        h;
        mu = -0.4,
        U = 2.0,
        n = fill(0.5, 16),
        Delta = fill(0.3 + 0.1im, 16),
        hole_convention = :conjugate,
    )
    op_dev = KPM.maybe_to_device(op)
    @test !(op_dev isa KPM.BdGOperator)          # assembled + moved
    x = randn(Xoshiro(5), ComplexF64, 32)
    y_dev = Array(op_dev * CuArray(x))
    y_host = zeros(ComplexF64, 32)
    mul!(y_host, op, x)
    @test y_dev ≈ y_host atol=1e-10

    rad_g, _ = KPM.spectral_radius(op_dev)
    rad_c, _ = KPM.spectral_radius(op)
    @test rad_g ≈ rad_c rtol=1e-6
end

@testset "bdg_channel_moments: raw GPU moments == CPU (nonuniform, partial batch)" begin
    rng = Xoshiro(21)
    N = 45
    A = sprandn(rng, ComplexF64, N, N, 0.15)
    h = (A + A') / 2
    op = KPM.BdGOperator(
        h;
        mu = 0.17,
        U = 1.3,
        n = rand(rng, N),
        Delta = 0.3 .* randn(rng, ComplexF64, N),
        hole_convention = :conjugate,
    )
    # Shuffled partial site set: 31 sites with batch_size=16 exercises the
    # final-partial-batch buffer reallocation; the bond list mixes onsite
    # bonds, several targets per source, and both directed orders.
    sites = Random.shuffle(rng, collect(1:N))[1:31]
    directed = Tuple{Int,Int}[]
    for k = 1:8
        i, j = sites[k], sites[k+1]
        push!(directed, (i, i))
        push!(directed, (i, j))
        push!(directed, (j, i))
        push!(directed, (i, sites[k+2]))
    end
    unique!(directed)
    a = 2 * first(KPM.spectral_radius(op)) / (2 - 0.2)
    NC = 64

    Hs_g = KPM.ScaledOperator(KPM.maybe_to_device(op), a, 0.0)
    rho_g, F_g = KPM.bdg_channel_moments(Hs_g, N, sites, directed, NC; batch_size = 16)
    Hs_c = KPM.ScaledOperator(op, a, 0.0)
    rho_c, F_c = KPM.bdg_channel_moments(Hs_c, N, sites, directed, NC; batch_size = 16)
    @test rho_g ≈ rho_c atol=1e-9 rtol=0
    @test F_g ≈ F_c atol=1e-9 rtol=0
end

@testset "BdG onsite SCF: GPU fixed point == CPU fixed point" begin
    Lx, Ly = 8, 8
    N = Lx * Ly
    h, pos, disp = square_model(Lx, Ly)
    make_op() = KPM.BdGOperator(
        h;
        mu = -0.4,
        U = 2.5,
        n = fill(0.5, N),
        Delta = fill(0.3 + 0.0im, N),
        hole_convention = :conjugate,
    )
    kw = (beta = 20.0, NC = 256, mix = 0.4, tol_delta = 1e-8, tol_n = 1e-8, maxiter = 300)

    op_g = make_op()
    res_g = KPM.bdg_solve!(op_g; kw...)
    op_c = make_op()
    res_c = on_cpu(() -> KPM.bdg_solve!(op_c; kw...))

    @test res_g.converged
    @test res_c.converged
    @test op_g.Δ ≈ op_c.Δ rtol=1e-5 atol=1e-7
    @test op_g.n ≈ op_c.n rtol=1e-5 atol=1e-7
    println(
        "onsite SCF: GPU $(res_g.iterations) iters, CPU $(res_c.iterations) iters, ",
        "max|Δ| = $(round(maximum(abs, op_g.Δ), digits=6))",
    )
end

@testset "BdG pairing-channel SCF: GPU == CPU" begin
    Lx, Ly = 6, 6
    N = Lx * Ly
    h, pos, disp = square_model(Lx, Ly)
    xbonds = [
        (square_site(ix, iy, Lx, Ly), square_site(ix + 1, iy, Lx, Ly)) for iy = 1:Ly for
        ix = 1:Lx
    ]
    channel = KPM.PairingChannel(xbonds, 1.0, 2.0, :even)
    make_op() = KPM.BdGOperator(
        h;
        mu = -0.3,
        U = 0.0,
        n = fill(0.4, N),
        D = KPM.pairing_matrix(N, [channel]; amplitude = 0.2),
        hole_convention = :conjugate,
    )
    kw = (beta = 15.0, NC = 256, mix = 0.4, tol_delta = 1e-8, tol_n = 1e-8, maxiter = 400)

    op_g = make_op()
    res_g = KPM.bdg_solve!(op_g, [channel]; kw...)
    op_c = make_op()
    res_c = on_cpu(() -> KPM.bdg_solve!(op_c, [channel]; kw...))

    @test res_g.converged
    @test res_c.converged
    @test nonzeros(op_g.D) ≈ nonzeros(op_c.D) rtol=1e-5 atol=1e-7
    @test op_g.n ≈ op_c.n rtol=1e-5 atol=1e-7
end

@testset "superfluid stiffness: GPU == CPU (same probes)" begin
    Lx, Ly = 8, 8
    N = Lx * Ly
    h, pos, disp = square_model(Lx, Ly)
    op = KPM.BdGOperator(
        h;
        mu = -0.5,
        U = 0.0,
        n = zeros(N),
        Delta = fill(0.4 + 0.0im, N),
        hole_convention = :conjugate,
    )
    q = [0.0, 2pi / Ly]
    psi_in = KPM.random_phase_vectors(Xoshiro(3), 2N, 6)
    kw = (
        beta = 20.0,
        eta = 0.2,
        dir = 1,
        disp = disp,
        NC = 128,
        volume = Float64(N),
        include_diamagnetic = true,
    )

    res_g = KPM.superfluid_stiffness(op, pos, q; psi_in = copy(psi_in), kw...)
    res_c = on_cpu(() -> KPM.superfluid_stiffness(op, pos, q; psi_in = copy(psi_in), kw...))

    @test res_g.Pi_SC ≈ res_c.Pi_SC rtol=1e-5
    @test res_g.Pi_N ≈ res_c.Pi_N rtol=1e-5
    @test res_g.Dia_SC ≈ res_c.Dia_SC rtol=1e-5
    @test res_g.Dia_N ≈ res_c.Dia_N rtol=1e-5
    # Compare the subtracted stiffness directly (not scaled by the large
    # unsubtracted responses, which would slacken the cancellation check).
    @test res_g.Ds_over_pi ≈ res_c.Ds_over_pi rtol=1e-4 atol=1e-8
    @test res_g.Ds_over_pi_complete ≈ res_c.Ds_over_pi_complete rtol=1e-4 atol=1e-8
    println(
        "stiffness: Ds/π = $(round(res_g.Ds_over_pi_complete, digits=6)) ",
        "(GPU) vs $(round(res_c.Ds_over_pi_complete, digits=6)) (CPU)",
    )
end

println("\n== timing: BdG moments, 64x64 square lattice, NC = 256 ==")
let
    Lx, Ly = 64, 64
    N = Lx * Ly
    h, pos, disp = square_model(Lx, Ly)
    op = KPM.BdGOperator(
        h;
        mu = -0.4,
        U = 2.5,
        n = fill(0.5, N),
        Delta = fill(0.3 + 0.0im, N),
        hole_convention = :conjugate,
    )
    NC = 256
    a = 2 * first(KPM.spectral_radius(op)) / (2 - 0.2)

    Hs_g = KPM.ScaledOperator(KPM.maybe_to_device(op), a, 0.0)
    KPM.bdg_site_moments(Hs_g, N, 1:128, NC)                      # warm-up
    t_gpu = @elapsed KPM.bdg_site_moments(Hs_g, N, 1:N, NC)
    Hs_c = KPM.ScaledOperator(op, a, 0.0)
    on_cpu(() -> KPM.bdg_site_moments(Hs_c, N, 1:128, NC))        # warm-up
    t_cpu = @elapsed on_cpu(() -> KPM.bdg_site_moments(Hs_c, N, 1:N, NC))
    println(
        "bdg_site_moments: GPU $(round(t_gpu, digits=2)) s | CPU $(round(t_cpu, digits=2)) s ",
        "| speedup ×$(round(t_cpu / t_gpu, digits=1))",
    )
end

println("\n== timing: 1D chain, NH = 2^17, NC = 1024, NR = 8 ==")
let
    NH = 2^17
    NC = 1024
    NR = 8
    H = spdiagm(1 => fill(-0.45 + 0im, NH - 1), -1 => fill(-0.45 + 0im, NH - 1))
    rng = Xoshiro(1)
    psi_in = exp.(2im * pi * rand(rng, NH, NR))
    KPM.normalize_by_col(psi_in, NR)

    KPM.kpm_1d(H, 64, NR; psi_in = psi_in)                    # GPU warm-up
    t_gpu = @elapsed KPM.kpm_1d(H, NC, NR; psi_in = psi_in)
    on_cpu(() -> KPM.kpm_1d(H, 64, NR; psi_in = psi_in))      # CPU warm-up
    t_cpu = @elapsed on_cpu(() -> KPM.kpm_1d(H, NC, NR; psi_in = psi_in))
    println(
        "kpm_1d: GPU $(round(t_gpu, digits=2)) s | CPU $(round(t_cpu, digits=2)) s ",
        "| speedup ×$(round(t_cpu / t_gpu, digits=1))",
    )
end

println("GPU SMOKE COMPLETE")
