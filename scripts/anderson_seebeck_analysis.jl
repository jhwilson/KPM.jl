# Offline analysis of the Anderson-Seebeck ensemble (cluster stage of
# examples/AndersonSeebeckBenchmarkPlan.md). Consumes the JLD2 task files
# written by scripts/anderson_seebeck_ensemble.jl; never recomputes moments.
#
#   julia --project=<env with KPM + JLD2 + QuadGK> \
#       scripts/anderson_seebeck_analysis.jl --manifest MANIFEST --datadir DIR
#
# For each size L (and each requested NC truncation of the stored moments):
#   * average the moment matrices over all pairs (+eps and -eps partners),
#   * report the particle-hole residual and how many pairs were found,
#   * reconstruct the disorder-averaged Sigma(E) band scan,
#   * free-(Ec, x) power-law fits of the transport edge (the convergence
#     observable: does (Ec_eff, x_eff) flow toward the critical values?),
#   * S(z) at the manifest kBT values against the universal x=1.5 curve,
#     with the plan's gate metrics printed per (L, NC, kBT).
# Results print as aligned tables; pass --save PATH.jld2 to also store them.

using KPM
using LinearAlgebra
using Printf
using Statistics
using TOML
using JLD2
using QuadGK

function parse_cli(args)
    manifest = nothing; datadir = nothing; save_path = nothing
    i = 1
    while i <= length(args)
        if args[i] == "--manifest"
            manifest = args[i + 1]; i += 2
        elseif args[i] == "--datadir"
            datadir = args[i + 1]; i += 2
        elseif args[i] == "--save"
            save_path = args[i + 1]; i += 2
        else
            error("unknown argument: $(args[i])")
        end
    end
    (manifest === nothing || datadir === nothing) &&
        error("--manifest and --datadir are required")
    return (; manifest, datadir, save_path)
end

# Universal upper-edge reference (windowed, overflow-safe; anchors pinned in
# examples/AndersonMobilityEdgeSeebeck.jl).
fermi_weight(e) = (q = exp(-abs(e)); q / (1 + q)^2)
function seebeck_ref_uVK(z; x=1.5)
    up = min(z, 40.0)
    up <= -40.0 && return NaN
    K0, _ = quadgk(e -> (z - e)^x * fermi_weight(e), -40.0, up; rtol=1e-10)
    K1, _ = quadgk(e -> e * (z - e)^x * fermi_weight(e), -40.0, up; rtol=1e-10)
    return -KPM.KB_OVER_E_UV_PER_K * K1 / K0
end

function particle_hole_residual(mu)
    odd = 0.0
    for j in axes(mu, 2), i in axes(mu, 1)
        isodd((i - 1) + (j - 1)) && (odd = max(odd, abs(mu[i, j])))
    end
    return odd / maximum(abs, mu)
end

# Least-squares Sigma ~= A*(Ec-E)^x on [E_lo, E_hi]; A analytic, (Ec, x) grid.
function edge_fit(E, sigma, E_lo, E_hi; x_fixed=nothing)
    sel = findall(e -> E_lo <= e <= E_hi, E)
    best = (Inf, NaN, NaN, NaN)
    for Ec in 7.0:0.005:9.4
        Ec <= E_hi + 0.05 && continue
        xs = x_fixed === nothing ? (0.3:0.05:3.0) : (x_fixed:x_fixed)
        for x in xs
            basis = (Ec .- E[sel]) .^ x
            A = (basis' * sigma[sel]) / (basis' * basis)
            sse = sum(abs2, sigma[sel] .- A .* basis)
            sse < best[1] && (best = (sse, Ec, x, A))
        end
    end
    return best
end

function main()
    cli = parse_cli(ARGS)
    mf = TOML.parsefile(cli.manifest)
    model = mf["model"]; ens = mf["ensemble"]; an = mf["analysis"]
    a = Float64(model["a"]); b = Float64(model["b"])
    sizes = Int.(ens["sizes"]); NCs_stored = Int.(ens["NC"])
    pairs_per_size = Int(ens["pairs_per_size"])
    kBT_values = Float64.(an["kBT"])
    Ec_ref = Float64(an["Ec_ref"]); x_ref = Float64(an["x_ref"])
    z_values = collect(range(-4.0, 8.0; length=25))
    gate_sel = findall(z -> 1.0 <= z <= 6.0, z_values)
    S_reference = [seebeck_ref_uVK(z; x=x_ref) for z in z_values]

    summary = Dict{String,Any}()
    for (si, L) in pairs(sizes)
        NC_stored = NCs_stored[si]
        NH = L^3
        volume = Float64(NH)
        files = filter(f -> occursin(@sprintf("L%02d_NC%04d", L, NC_stored), f) &&
                            endswith(f, ".jld2"), readdir(cli.datadir; join=true))
        if isempty(files)
            println("L=$L: no task files found — skipping")
            continue
        end
        mu_sum = nothing
        n_real = 0
        for f in files
            mu_sum === nothing && (mu_sum = zeros(ComplexF64, NC_stored, NC_stored))
            mu_sum .+= load(f, "mu_plus")
            mu_sum .+= load(f, "mu_minus")
            n_real += 2
        end
        mu_avg = mu_sum ./ n_real
        ph = particle_hole_residual(mu_avg)
        @printf("\n================ L=%d: %d pairs (%d/%d), PH residual %.2e ================\n",
                L, length(files), length(files), pairs_per_size, ph)

        for NC in unique(clamp.((256, 384, 512, 768, 1024), 0, NC_stored))
            m = KPM.ConductivityMoments(mu_avg[1:NC, 1:NC], a, b, NH,
                                        Int(mf["kpm"]["NR"]))
            E_grid = collect(range(4.0, 9.5; length=1101))
            sigma = KPM.transport_distribution(m, E_grid; volume=volume)
            _, Ec_f, x_f, _ = edge_fit(E_grid, sigma, 6.0, 8.1)
            _, Ec_15, _, _ = edge_fit(E_grid, sigma, 6.0, 8.1; x_fixed=x_ref)
            @printf("L=%2d NC=%4d (pi*a/NC=%.3ft): Ec_eff=%.3f x_eff=%.2f (x=%.1f fixed: Ec=%.3f)\n",
                    L, NC, pi * a / NC, Ec_f, x_f, x_ref, Ec_15)

            n_band = max(257, 4 * NC + 1)
            w = a * (1 - 1e-3)
            sigma_band = KPM.transport_distribution(m,
                collect(range(b - w, b + w; length=n_band)); volume=volume)
            sigma_floor = 1e-6 * maximum(abs, sigma_band)
            for kBT in kBT_values
                S = [KPM.seebeck_uVK(KPM.thermoelectric(m, Ec_ref - z * kBT;
                        beta=inv(kBT), volume=volume, sigma_min=sigma_floor,
                        quad_N=4 * NC)) for z in z_values]
                devs = [abs(S[i] / S_reference[i] - 1) for i in gate_sel]
                @printf("    kBT=%.2f: S(z=0)=%8.3f uV/K, median|S/S_ref-1| (1<=z<=6) = %5.1f%%\n",
                        kBT, S[findfirst(==(0.0), z_values)], 100 * median(devs))
                summary["L$(L)_NC$(NC)_kBT$(kBT)"] =
                    Dict("S" => S, "z" => z_values, "median_dev" => median(devs),
                         "Ec_eff" => Ec_f, "x_eff" => x_f, "ph_residual" => ph,
                         "n_realizations" => n_real)
            end
        end
    end

    if cli.save_path !== nothing
        jldsave(cli.save_path; summary=summary, S_reference=S_reference,
                z_values=z_values)
        println("\nSaved summary: $(abspath(cli.save_path))")
    end
    println("\nReminder: finite-size + broadening study; this does not measure")
    println("the thermodynamic mobility edge or critical exponent.")
end

main()
