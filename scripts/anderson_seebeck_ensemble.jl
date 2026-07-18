# One antithetic disorder pair of the 3D Anderson mobility-edge Seebeck
# ensemble (the cluster stage of the
# Anderson mobility-edge benchmark).
#
#   julia --project=<env with KPM + JLD2> scripts/anderson_seebeck_ensemble.jl \
#       --manifest scripts/anderson_seebeck_manifest.toml --task N --outdir DIR
#
# Task N enumerates (size, pair) points column-major in the manifest order:
# task = (size_index - 1) * pairs_per_size + pair_index. Each task builds
# H(+eps) and H(-eps) for one disorder draw, computes the two conductivity
# moment matrices with Gamma-paired probes at the fixed manifest rescaling,
# and writes them to one JLD2 file. Averaging over pairs, reconstruction, and
# fits happen offline (locally); this script never averages.
#
# The model construction mirrors examples/AndersonMobilityEdgeSeebeck.jl
# (which stays standalone by repo convention); the antithetic identities are
# asserted here as well, so a silent divergence between the two would fail
# loudly. Runs on CPU or GPU unchanged: KPM's CUDA extension activates when
# a functional device is present, and the moment matrices are host-side
# either way. Existing outputs with matching parameters are not recomputed.

using KPM
using LinearAlgebra
using SparseArrays
using Random
using Printf
using TOML
using JLD2

# KPM's CUDA extension only activates when CUDA.jl is loaded in the session,
# so load it when the environment provides it. Nodes without a functional
# device (and environments without CUDA) fall back to the CPU path.
try
    @eval using CUDA
catch
    println("CUDA.jl not available in this environment — CPU path")
end

site_index(ix, iy, iz, L) = ix + 1 + L * (iy + L * iz)
minimum_image(delta, L) = mod(delta + fld(L, 2), L) - fld(L, 2)

function cubic_hamiltonian_and_current(L, disorder; t=1.0)
    @assert iseven(L) "antithetic particle-hole pairing requires even L"
    NH = L^3
    length(disorder) == NH || throw(DimensionMismatch("expected $NH disorder values"))

    rows = Int[]; cols = Int[]; hvals = ComplexF64[]
    jrows = Int[]; jcols = Int[]; jvals = ComplexF64[]
    positions = Vector{NTuple{3,Int}}(undef, NH)
    gamma = Vector{Float64}(undef, NH)

    for iz in 0:(L - 1), iy in 0:(L - 1), ix in 0:(L - 1)
        i = site_index(ix, iy, iz, L)
        positions[i] = (ix, iy, iz)
        gamma[i] = iseven(ix + iy + iz) ? 1.0 : -1.0
    end

    for i in 1:NH
        push!(rows, i); push!(cols, i); push!(hvals, disorder[i] + 0im)
    end

    function add_hopping!(i, j, value)
        ri = positions[i]; rj = positions[j]
        dx = minimum_image(ri[1] - rj[1], L)
        push!(rows, i); push!(cols, j); push!(hvals, value)
        if !iszero(dx)
            push!(jrows, i); push!(jcols, j); push!(jvals, value * dx)
        end
    end

    for iz in 0:(L - 1), iy in 0:(L - 1), ix in 0:(L - 1)
        i = site_index(ix, iy, iz, L)
        for (jx, jy, jz) in ((mod(ix + 1, L), iy, iz),
                             (ix, mod(iy + 1, L), iz),
                             (ix, iy, mod(iz + 1, L)))
            j = site_index(jx, jy, jz, L)
            add_hopping!(i, j, -t + 0im)
            add_hopping!(j, i, -t + 0im)
        end
    end

    H = sparse(rows, cols, hvals, NH, NH)
    Jx = sparse(jrows, jcols, jvals, NH, NH)
    dropzeros!(Jx)
    @assert maximum(abs, Jx + Jx') <= 1e-13 * maximum(abs, Jx)
    return H, Jx, gamma
end

function parse_cli(args)
    manifest = nothing; task = nothing; outdir = nothing
    i = 1
    while i <= length(args)
        if args[i] == "--manifest"
            manifest = args[i + 1]; i += 2
        elseif args[i] == "--task"
            task = parse(Int, args[i + 1]); i += 2
        elseif args[i] == "--outdir"
            outdir = args[i + 1]; i += 2
        else
            error("unknown argument: $(args[i])")
        end
    end
    manifest === nothing && error("--manifest is required")
    task === nothing && error("--task is required")
    outdir === nothing && error("--outdir is required")
    return (; manifest, task, outdir)
end

function main()
    cli = parse_cli(ARGS)
    mf = TOML.parsefile(cli.manifest)
    model = mf["model"]; kpm = mf["kpm"]; ens = mf["ensemble"]

    W = Float64(model["W"]); t = Float64(model["t"])
    a = Float64(model["a"]); b = Float64(model["b"])
    NR = Int(kpm["NR"]); arr_size = Int(kpm["arr_size"])
    sizes = Int.(ens["sizes"]); NCs = Int.(ens["NC"])
    pairs_per_size = Int(ens["pairs_per_size"])
    seed_base = Int(ens["seed_base"])
    length(NCs) == length(sizes) || error("manifest: NC list must match sizes list")

    n_tasks = length(sizes) * pairs_per_size
    1 <= cli.task <= n_tasks || error("task $(cli.task) outside 1:$n_tasks")
    size_index = fld(cli.task - 1, pairs_per_size) + 1
    pair_index = mod(cli.task - 1, pairs_per_size) + 1
    L = sizes[size_index]; NC = NCs[size_index]
    NH = L^3
    # Disjoint seed blocks per size so adding sizes never reuses a stream.
    seed = seed_base + 10_000 * size_index + pair_index

    params = Dict{String,Any}(
        "W" => W, "t" => t, "a" => a, "b" => b, "L" => L, "NC" => NC,
        "NR" => NR, "arr_size" => arr_size, "seed" => seed,
        "pair_index" => pair_index, "schema" => 1)

    outfile = joinpath(cli.outdir,
        @sprintf("anderson_seebeck_L%02d_NC%04d_pair%02d.jld2", L, NC, pair_index))
    if isfile(outfile)
        existing = try
            load(outfile, "params")
        catch
            nothing
        end
        if existing == params
            println("SKIP: $outfile exists with matching parameters")
            return
        end
        error("$outfile exists with DIFFERENT parameters — refusing to overwrite")
    end
    mkpath(cli.outdir)

    @printf("task %d/%d: L=%d NH=%d NC=%d NR=%d pair=%d seed=%d\n",
            cli.task, n_tasks, L, NH, NC, NR, pair_index, seed)
    device = KPM.whichcore() ? "gpu" : "cpu"
    println("compute device: ", device)

    rng = Xoshiro(seed)
    disorder = W .* (rand(rng, NH) .- 0.5)
    psi_plus = KPM.random_phase_vectors(rng, NH, NR)

    t_build = @elapsed begin
        H_plus, Jx, gamma = cubic_hamiltonian_and_current(L, disorder; t=t)
        H_minus, Jx_minus, _ = cubic_hamiltonian_and_current(L, -disorder; t=t)
    end
    @assert Jx == Jx_minus
    Gamma = spdiagm(0 => gamma)
    @assert maximum(abs, H_minus + Gamma * H_plus * Gamma) /
            maximum(abs, H_minus) < 1e-13
    psi_minus = reshape(gamma, :, 1) .* psi_plus

    h_plus = KPM.RescaledHamiltonian(H_plus ./ a, a, b)
    h_minus = KPM.RescaledHamiltonian(H_minus ./ a, a, b)

    t_plus = @elapsed m_plus = KPM.cond_moments(h_plus, Jx, Jx;
        NC=NC, psi_in=psi_plus, arr_size=arr_size)
    t_minus = @elapsed m_minus = KPM.cond_moments(h_minus, Jx, Jx;
        NC=NC, psi_in=psi_minus, arr_size=arr_size)

    # Bit-exactness of the antithetic identity is CPU-specific (GPU GEMM order
    # differs), so check it as a small-residual identity here.
    pair_residual = maximum(abs, [abs(m_minus.mu[n, m] -
                        (isodd((n - 1) + (m - 1)) ? -1 : 1) * m_plus.mu[n, m])
                        for n in 1:NC, m in 1:NC]) / maximum(abs, m_plus.mu)
    @printf("antithetic pair identity residual = %.3e\n", pair_residual)
    pair_residual < 1e-8 || @warn "pair identity residual unexpectedly large" pair_residual

    meta = Dict{String,Any}(
        "hostname" => gethostname(),
        "device" => device,
        "julia_version" => string(VERSION),
        "git_sha" => something(
            try readchomp(`git -C $(dirname(@__DIR__)) rev-parse HEAD`) catch; nothing end,
            isfile("GIT_SHA") ? strip(read("GIT_SHA", String)) : "unknown"),
        "pair_residual" => pair_residual,
        "t_build" => t_build, "t_moments_plus" => t_plus,
        "t_moments_minus" => t_minus)

    jldsave(outfile; params=params, mu_plus=m_plus.mu, mu_minus=m_minus.mu,
            meta=meta)
    @printf("build %.2fs, moments +eps %.2fs, -eps %.2fs\n", t_build, t_plus, t_minus)
    println("WROTE: $outfile")
end

main()
