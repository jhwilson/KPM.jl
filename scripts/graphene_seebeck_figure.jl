# Extract the publication-figure data for the clean-graphene CTKG Seebeck
# benchmark from a moment cache written by
#   julia --project=. examples/GrapheneSeebeck.jl --cache CACHE
# Reconstruction only — never recomputes moments. Writes three CSVs consumed
# by scripts/graphene_seebeck_figure.py.
#
#   julia --project=. scripts/graphene_seebeck_figure.jl --cache CACHE --outdir DIR

using KPM
using Serialization
using DelimitedFiles

try
    @eval using QuadGK
catch err
    println(
        stderr,
        "QuadGK.jl is required (same example-only dependency as GrapheneSeebeck.jl).",
    )
    rethrow(err)
end

# These must match examples/GrapheneSeebeck.jl (asserted against the cache).
const Nx = 48
const Ny = 48
const t = 1.0
const a = 3.1
const b = 0.0
const NC = 512
const NR = 4
const arr_size = 16
const kBT_values = (0.04, 0.06)
const eta_values = collect(range(-8.0, 8.0; length = 33))
const quad_N = 4 * NC
const seed = Int(0x6e617068)
const area = Nx * Ny * 3sqrt(3) / 2
const g_J = 2.0
const E_ref = 0.2

expected_params() = (;
    schema = 1,
    Nx,
    Ny,
    t,
    a,
    b,
    NC,
    NR,
    arr_size,
    seed,
    model = :GrapheneLattice_gapless,
    twist_scheme = :cartesian_0_plusminus_pi_over_2_pi,
    probe_api = :one_seeded_rng_stream,
)

fermi_weight(u::Real) = begin
    q = exp(-abs(u))
    q / (1 + q)^2
end

# Clean-Dirac reference: S/(kB/|e|) = -A1/A0 with Sigma(E) ∝ |E|.
function seebeck_reference(eta::Real)
    eta_f = Float64(eta)
    lower, upper = eta_f - 60.0, eta_f + 60.0
    A0, _ = quadgk(
        y -> abs(y) * fermi_weight(y - eta_f),
        lower,
        0.0,
        upper;
        rtol = 2e-13,
        atol = 1e-14,
    )
    A1, _ = quadgk(
        y -> (y - eta_f) * abs(y) * fermi_weight(y - eta_f),
        lower,
        0.0,
        upper;
        rtol = 2e-13,
        atol = 1e-14,
    )
    return -KPM.KB_OVER_E_UV_PER_K * A1 / A0
end

# Exact bipartite particle-hole projection (Delta=0, twists included).
function particle_hole_project(mu)
    projected = copy(mu)
    for j in axes(projected, 2), i in axes(projected, 1)
        if isodd((i - 1) + (j - 1))
            projected[i, j] = zero(eltype(projected))
        end
    end
    return projected
end

function parse_cli(args)
    cache = nothing
    outdir = nothing
    i = 1
    while i <= length(args)
        if args[i] == "--cache" && i < length(args)
            cache = args[i+1];
            i += 2
        elseif args[i] == "--outdir" && i < length(args)
            outdir = args[i+1];
            i += 2
        else
            error("usage: graphene_seebeck_figure.jl --cache CACHE --outdir DIR")
        end
    end
    (cache === nothing || outdir === nothing) &&
        error("usage: graphene_seebeck_figure.jl --cache CACHE --outdir DIR")
    return (; cache, outdir)
end

function main()
    cli = parse_cli(ARGS)
    payload = open(deserialize, cli.cache)
    payload.params == expected_params() ||
        error("cache parameters do not match examples/GrapheneSeebeck.jl fast defaults")
    NH = 2 * Nx * Ny
    mxx = KPM.ConductivityMoments(particle_hole_project(payload.mu), a, b, NH, NR)

    E_grid = collect(range(-0.4, 0.4; length = 401))
    sigma = KPM.transport_distribution(mxx, E_grid; volume = area, g_J = g_J)
    i_ref = argmin(abs.(E_grid .- E_ref))
    sigma_normalized = sigma ./ sigma[i_ref]

    S = Dict{Float64,Vector{Float64}}()
    for kBT in kBT_values
        S[kBT] = [
            KPM.seebeck_uVK(
                KPM.thermoelectric(
                    mxx,
                    eta * kBT;
                    beta = inv(kBT),
                    volume = area,
                    g_J = g_J,
                    sigma_min = 0.0,
                    quad_N = quad_N,
                ),
            ) for eta in eta_values
        ]
    end
    accepted = Dict(
        kBT => [abs(eta * kBT) <= 0.35t + 10eps(Float64) for eta in eta_values] for
        kBT in kBT_values
    )

    eta_dense = collect(range(first(eta_values), last(eta_values); length = 321))
    S_dense = seebeck_reference.(eta_dense)

    mkpath(cli.outdir)
    open(joinpath(cli.outdir, "sigma.csv"), "w") do io
        println(io, "E,sigma_normalized")
        writedlm(io, hcat(E_grid, sigma_normalized), ',')
    end
    kT1, kT2 = kBT_values
    open(joinpath(cli.outdir, "seebeck.csv"), "w") do io
        println(io, "eta,S_ref,S_kT1,acc_kT1,S_kT2,acc_kT2")
        writedlm(
            io,
            hcat(
                eta_values,
                seebeck_reference.(eta_values),
                S[kT1],
                Int.(accepted[kT1]),
                S[kT2],
                Int.(accepted[kT2]),
            ),
            ',',
        )
    end
    open(joinpath(cli.outdir, "reference_dense.csv"), "w") do io
        println(io, "eta,S_ref")
        writedlm(io, hcat(eta_dense, S_dense), ',')
    end
    repo = dirname(@__DIR__)
    sha = try
        readchomp(`git -C $repo rev-parse HEAD`)
    catch
        "unknown"
    end
    dirty = try
        isempty(readchomp(`git -C $repo status --porcelain`)) ? "clean" : "dirty"
    catch
        "unknown"
    end
    open(joinpath(cli.outdir, "provenance.txt"), "w") do io
        println(io, "cache = ", abspath(cli.cache))
        println(io, "git_sha = ", sha, " (", dirty, ")")
        println(io, "params = ", payload.params)
    end
    println(
        "Wrote sigma.csv, seebeck.csv, reference_dense.csv, provenance.txt to $(abspath(cli.outdir))",
    )
end

main()
