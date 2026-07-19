using Test
using LinearAlgebra
using SparseArrays
using Random
using Statistics
using KPM

# Fermi projector and Bianco–Resta local Chern marker. Anchors: exact dense
# projectors (ED), and the FHS Chern number of the Haldane model — the same
# sign convention pinned by kubo_bastin_test.jl (σ_xy = +C e²/h).

isdefined(@__MODULE__, :EDReference) || include("ed_reference.jl")
using .EDReference

# Evaluate a Chebyshev series with coefficients c at x ∈ (-1, 1).
_cheb_eval(c, x) = sum(c[n+1] * cos(n * acos(x)) for n = 0:(length(c)-1))

# Orbital indices of the cells (x, y) ∈ xs × ys of an Lx×Ly open Haldane
# flake (both sublattices), matching haldane_open_model's site layout.
function _haldane_cells(Ly, xs, ys)
    sites = Int[]
    for x in xs, y in ys
        c = y + Ly * (x - 1)
        push!(sites, 2 * c - 1, 2 * c)
    end
    return sites
end

@testset "fermi_coefficients: closed form, quadrature, and error paths" begin
    a, b, Ef, NC = 2.3, 0.4, 0.7, 256
    xF = (Ef - b) / a

    # closed form (beta = Inf) is the large-beta quadrature limit; the
    # integrand is then a step, so node discretization limits agreement to
    # O(1/Np) — assert the level and its shrinkage with Np
    c_step = KPM.fermi_coefficients(a, b, Ef; NC = NC)
    c_quad = KPM.fermi_coefficients(a, b, Ef; beta = 1e8, NC = NC)
    @test maximum(abs, c_step .- c_quad) < 2e-3
    c_quad_fine = KPM.fermi_coefficients(a, b, Ef; beta = 1e8, NC = NC, Np = 8192)
    @test maximum(abs, c_step .- c_quad_fine) < 1e-4

    # undamped series reproduces the step away from x̃_F (Gibbs at the edge)
    c_bare = KPM.fermi_coefficients(a, b, Ef; NC = NC, kernel = (n, N) -> 1.0)
    for x in (-0.9, -0.5, xF - 0.2, xF + 0.2, 0.6, 0.9)
        step = x < xF ? 1.0 : 0.0
        @test _cheb_eval(c_bare, x) ≈ step atol = 1e-2
    end
    # Jackson damping: accurate outside a ~π/NC resolution window
    for x in (-0.9, -0.5, xF - 0.2, xF + 0.2, 0.6, 0.9)
        step = x < xF ? 1.0 : 0.0
        @test _cheb_eval(c_step, x) ≈ step atol = 1e-3
    end

    # finite beta matches the exact Fermi factor (smooth: quadrature is exact
    # to machine precision at moderate NC)
    beta = 20.0
    c_T = KPM.fermi_coefficients(a, b, Ef; beta = beta, NC = NC)
    for x in (-0.9, -0.3, xF, 0.5, 0.9)
        f = 1 / (exp((a * x + b - Ef) * beta) + 1)
        @test _cheb_eval(c_T, x) ≈ f atol = 1e-3
    end

    # Ef outside the rescaled spectral window throws
    @test_throws ArgumentError KPM.fermi_coefficients(a, b, b + 1.5 * a; NC = NC)
    @test_throws ArgumentError KPM.fermi_coefficients(a, b, b - a; NC = NC)
    # argument validation
    @test_throws ArgumentError KPM.fermi_coefficients(-a, b, Ef; NC = NC)
    @test_throws ArgumentError KPM.fermi_coefficients(a, b, Ef; NC = 0)
    @test_throws ArgumentError KPM.fermi_coefficients(a, b, Ef; NC = NC, beta = 0.0)
    @test_throws ArgumentError KPM.fermi_coefficients(a, b, Ef; NC = NC, Np = 0)
    # NC is deliberately required
    @test_throws UndefKeywordError KPM.fermi_coefficients(a, b, Ef)
end

@testset "fermi_projector vs exact dense projector" begin
    rng = Xoshiro(211)
    NH, NC = 64, 512
    a, b = 2.3, 0.4
    # controlled gapped spectrum: rescaled eigenvalues in ±[0.2, 0.9], so the
    # step at x̃_F = 0 (Ef = b) sits in a gap of width 0.4 ≫ π/NC
    λ = vcat(-0.9 .+ 0.7 .* rand(rng, NH ÷ 2), 0.2 .+ 0.7 .* rand(rng, NH ÷ 2))
    Q = Matrix(qr(randn(rng, ComplexF64, NH, NH)).Q)
    Hn = Matrix(Hermitian(Q * Diagonal(λ) * Q'))
    h = KPM.RescaledHamiltonian(Hn, a, b)
    Ef = b

    P_ed = Q * Diagonal(Float64.(λ .< 0)) * Q'
    V = randn(rng, ComplexF64, NH, 3)
    PV = KPM.fermi_projector(h, V; Ef = Ef, NC = NC)
    @test PV ≈ P_ed * V atol = 1e-3

    # vector input keeps vector shape
    v = V[:, 1]
    Pv = KPM.fermi_projector(h, v; Ef = Ef, NC = NC)
    @test Pv isa Vector
    @test Pv ≈ PV[:, 1] atol = 1e-12

    # Hermiticity and (approximate) idempotency of the damped projector
    u = randn(rng, ComplexF64, NH)
    Pu = KPM.fermi_projector(h, u; Ef = Ef, NC = NC)
    @test dot(u, Pv) ≈ dot(Pu, v) atol = 1e-10
    @test norm(KPM.fermi_projector(h, Pv; Ef = Ef, NC = NC) - Pv) < 2e-3

    # finite temperature matches the dense U f_β(E − Ef) U' action
    beta = 20.0
    P_T = Q * Diagonal(1 ./ (exp.(a .* λ .* beta) .+ 1)) * Q'
    PV_T = KPM.fermi_projector(h, V; Ef = Ef, beta = beta, NC = NC)
    @test PV_T ≈ P_T * V atol = 1e-3

    # validation: row mismatch and required NC
    @test_throws ArgumentError KPM.fermi_projector(
        h,
        randn(rng, ComplexF64, NH + 1, 2);
        Ef = Ef,
        NC = NC,
    )
    @test_throws UndefKeywordError KPM.fermi_projector(h, V; Ef = Ef)
end

# Deterministic rescaling from exact eigenvalues (the evolution_test pattern).
function _marker_fixture(H; margin = 0.95)
    ev = eigvals(Hermitian(Matrix(H)))
    b = (maximum(ev) + minimum(ev)) / 2
    a = (maximum(ev) - minimum(ev)) / 2 / margin
    return KPM.RescaledHamiltonian((H - b * I) ./ a, a, b)
end

@testset "open-boundary Haldane fixture: exact marker matches FHS" begin
    # validates the ED oracle itself before KPM touches it
    Lx = Ly = 10
    H, pos, Ac = haldane_open_model(Lx, Ly)   # topological defaults
    C = round(chern_number_fhs(haldane_bloch(; t = 1.0, t2 = 0.2, ϕ = π/2, m = 0.0)))
    @test abs(C) == 1

    mk = ed_chern_marker(H, pos[:, 1], pos[:, 2], 0.0)
    bulk = _haldane_cells(Ly, 4:7, 4:7)
    @test sum(mk[bulk]) / (16 * Ac) ≈ C atol = 0.01
    # exact identity: Im Tr[P X Q Y P] = 0 — the whole-sample sum vanishes
    @test abs(sum(mk)) < 1e-8
    # geometry sanity: N sites, cell area of the honeycomb lattice vectors
    @test size(pos) == (2 * Lx * Ly, 2)
    @test Ac ≈ 3 * sqrt(3) / 2 atol = 1e-12

    # trivial phase: bulk marker near zero
    m_triv = 1.6
    Ht, post, _ = haldane_open_model(Lx, Ly; m = m_triv)
    Ct = round(chern_number_fhs(haldane_bloch(; t = 1.0, t2 = 0.2, ϕ = π/2, m = m_triv)))
    @test Ct == 0
    mkt = ed_chern_marker(Ht, post[:, 1], post[:, 2], 0.0)
    @test abs(sum(mkt[bulk]) / (16 * Ac)) < 0.01
end

@testset "topological phase: bulk marker quantized, sign anchored to FHS" begin
    Lx = Ly = 10
    NC = 512
    H, pos, Ac = haldane_open_model(Lx, Ly)
    h = _marker_fixture(H)
    C = round(chern_number_fhs(haldane_bloch(; t = 1.0, t2 = 0.2, ϕ = π/2, m = 0.0)))
    bulk = _haldane_cells(Ly, 4:7, 4:7)

    mk = KPM.chern_marker(h, pos[:, 1], pos[:, 2]; Ef = 0.0, sites = bulk, NC = NC)
    @test KPM.chern_marker_average(mk; area = 16 * Ac) ≈ C atol = 0.05
    # every bulk cell individually quantized (sites list is cell-contiguous)
    for k = 1:16
        @test KPM.chern_marker_average(mk[(2k-1):(2k)]; area = Ac) ≈ C atol = 0.1
    end
    # per-orbital agreement with the exact-projector marker
    mk_ed = ed_chern_marker(H, pos[:, 1], pos[:, 2], 0.0)
    @test mk ≈ mk_ed[bulk] atol = 0.02
end

@testset "boundary semantics: whole-sample sum vanishes, edge is not bulk" begin
    Lx = Ly = 10
    NC = 512
    H, pos, Ac = haldane_open_model(Lx, Ly)
    h = _marker_fixture(H)
    C = round(chern_number_fhs(haldane_bloch(; t = 1.0, t2 = 0.2, ϕ = π/2, m = 0.0)))

    all_sites = collect(1:(2*Lx*Ly))
    mk = KPM.chern_marker(h, pos[:, 1], pos[:, 2]; Ef = 0.0, sites = all_sites, NC = NC)
    # the marker over the entire open sample carries no topology: the exact
    # identity Im Tr[P X Q Y P] = 0 (ED sum ≈ 1e-13) is approached with NC —
    # the residual is boundary-dominated truncation error, small against the
    # gross weight Σ|m| and shrinking superlinearly (measured ≈ ×9 per
    # doubling: 6.8 → 0.58 → 0.065 at NC = 256/512/1024)
    @test abs(sum(mk)) < 2e-3 * sum(abs, mk)
    mk_2NC = KPM.chern_marker(
        h,
        pos[:, 1],
        pos[:, 2];
        Ef = 0.0,
        sites = all_sites,
        NC = 2 * NC,
    )
    @test abs(sum(mk_2NC)) < abs(sum(mk)) / 4
    # an edge-column average is clearly not the bulk value
    edge = _haldane_cells(Ly, 1:1, 1:Ly)
    @test abs(sum(mk[edge]) / (Ly * Ac) - C) > 0.3
    # batching is exact bookkeeping: odd batch_size == one big batch
    mk7 = KPM.chern_marker(
        h,
        pos[:, 1],
        pos[:, 2];
        Ef = 0.0,
        sites = all_sites,
        NC = NC,
        batch_size = 7,
    )
    mk200 = KPM.chern_marker(
        h,
        pos[:, 1],
        pos[:, 2];
        Ef = 0.0,
        sites = all_sites,
        NC = NC,
        batch_size = 200,
    )
    @test mk7 ≈ mk atol = 1e-12
    @test mk200 ≈ mk atol = 1e-12
end

@testset "trivial phase: bulk marker vanishes" begin
    Lx = Ly = 10
    NC = 512
    m_triv = 1.6
    H, pos, Ac = haldane_open_model(Lx, Ly; m = m_triv)
    h = _marker_fixture(H)
    bulk = _haldane_cells(Ly, 4:7, 4:7)

    mk = KPM.chern_marker(h, pos[:, 1], pos[:, 2]; Ef = 0.0, sites = bulk, NC = NC)
    @test abs(KPM.chern_marker_average(mk; area = 16 * Ac)) < 0.05
    mk_ed = ed_chern_marker(H, pos[:, 1], pos[:, 2], 0.0)
    @test mk ≈ mk_ed[bulk] atol = 0.02
end

@testset "finite temperature: thermal projector keeps the bulk marker" begin
    Lx = Ly = 10
    NC = 512
    beta = 20.0   # T well below the bulk gap ≈ 2.1
    H, pos, Ac = haldane_open_model(Lx, Ly)
    h = _marker_fixture(H)
    C = round(chern_number_fhs(haldane_bloch(; t = 1.0, t2 = 0.2, ϕ = π/2, m = 0.0)))
    bulk = _haldane_cells(Ly, 4:7, 4:7)

    mk = KPM.chern_marker(
        h,
        pos[:, 1],
        pos[:, 2];
        Ef = 0.0,
        sites = bulk,
        beta = beta,
        NC = NC,
    )
    @test KPM.chern_marker_average(mk; area = 16 * Ac) ≈ C atol = 0.1
    mk_ed = ed_chern_marker(H, pos[:, 1], pos[:, 2], 0.0; beta = beta)
    @test mk ≈ mk_ed[bulk] atol = 0.02
end

@testset "disorder robustness: marker survives weak onsite disorder" begin
    Lx = Ly = 10
    NC = 512
    W = 0.5
    rng = Xoshiro(227)
    H, pos, Ac = haldane_open_model(Lx, Ly)
    Hdis = H + spdiagm(0 => ComplexF64.(W .* (rand(rng, 2 * Lx * Ly) .- 0.5)))
    h = _marker_fixture(Hdis)
    C = round(chern_number_fhs(haldane_bloch(; t = 1.0, t2 = 0.2, ϕ = π/2, m = 0.0)))
    bulk = _haldane_cells(Ly, 4:7, 4:7)

    mk = KPM.chern_marker(h, pos[:, 1], pos[:, 2]; Ef = 0.0, sites = bulk, NC = NC)
    @test KPM.chern_marker_average(mk; area = 16 * Ac) ≈ C atol = 0.15
    mk_ed = ed_chern_marker(Hdis, pos[:, 1], pos[:, 2], 0.0)
    @test mk ≈ mk_ed[bulk] atol = 0.05
end

@testset "stochastic regional estimator" begin
    Lx = Ly = 10
    NC = 512
    NR = 32
    H, pos, Ac = haldane_open_model(Lx, Ly)
    h = _marker_fixture(H)
    C = round(chern_number_fhs(haldane_bloch(; t = 1.0, t2 = 0.2, ϕ = π/2, m = 0.0)))
    region = _haldane_cells(Ly, 4:7, 4:7)

    # deterministic reference: exact region sum from per-site markers
    det_sum =
        sum(KPM.chern_marker(h, pos[:, 1], pos[:, 2]; Ef = 0.0, sites = region, NC = NC))

    est = KPM.chern_marker_region(
        h,
        pos[:, 1],
        pos[:, 2];
        Ef = 0.0,
        region = region,
        rng = Xoshiro(31),
        NR = NR,
        NC = NC,
    )
    @test length(est) == NR
    stderr_est = std(est) / sqrt(NR)
    @test abs(mean(est) - det_sum) < 4 * stderr_est
    # regional Chern estimate through the same explicit-area contract
    @test KPM.chern_marker_average([mean(est)]; area = 16 * Ac) ≈ C atol = 0.2

    # identical seed ⇒ identical probes ⇒ batching is exact bookkeeping
    est5 = KPM.chern_marker_region(
        h,
        pos[:, 1],
        pos[:, 2];
        Ef = 0.0,
        region = region,
        rng = Xoshiro(31),
        NR = NR,
        NC = NC,
        batch_size = 5,
    )
    @test est5 ≈ est atol = 1e-12

    # validation specific to the regional mode
    @test_throws ArgumentError KPM.chern_marker_region(
        h,
        pos[:, 1],
        pos[:, 2];
        Ef = 0.0,
        region = [1, 1],
        rng = Xoshiro(1),
        NC = NC,
    )
    @test_throws ArgumentError KPM.chern_marker_region(
        h,
        pos[:, 1],
        pos[:, 2];
        Ef = 0.0,
        region = region,
        rng = Xoshiro(1),
        NR = 0,
        NC = NC,
    )
    @test_throws UndefKeywordError KPM.chern_marker_region(
        h,
        pos[:, 1],
        pos[:, 2];
        Ef = 0.0,
        region = region,
        NR = 2,
        NC = NC,
    )
end

@testset "argument validation" begin
    H, pos, Ac = haldane_open_model(4, 4)
    h = _marker_fixture(H)
    N = 2 * 4 * 4
    x, y = pos[:, 1], pos[:, 2]

    @test_throws ArgumentError KPM.chern_marker(h, x[1:(N-1)], y; Ef = 0.0, sites = [1], NC = 64)
    @test_throws ArgumentError KPM.chern_marker(h, x, y[1:(N-1)]; Ef = 0.0, sites = [1], NC = 64)
    @test_throws ArgumentError KPM.chern_marker(h, x, y; Ef = 0.0, sites = Int[], NC = 64)
    @test_throws ArgumentError KPM.chern_marker(h, x, y; Ef = 0.0, sites = [0], NC = 64)
    @test_throws ArgumentError KPM.chern_marker(h, x, y; Ef = 0.0, sites = [N + 1], NC = 64)
    @test_throws ArgumentError KPM.chern_marker(
        h,
        x,
        y;
        Ef = 0.0,
        sites = [1],
        NC = 64,
        batch_size = 0,
    )
    # Ef outside the rescaled window propagates from fermi_coefficients
    @test_throws ArgumentError KPM.chern_marker(
        h,
        x,
        y;
        Ef = h.b + 2 * h.a,
        sites = [1],
        NC = 64,
    )
    @test_throws UndefKeywordError KPM.chern_marker(h, x, y; Ef = 0.0, sites = [1])
    @test_throws ArgumentError KPM.chern_marker_average([1.0]; area = 0.0)
    @test_throws ArgumentError KPM.chern_marker_average([1.0]; area = -1.0)
end
