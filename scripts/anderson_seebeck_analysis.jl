# Offline analysis of the Anderson-Seebeck ensemble (cluster stage of
# the Anderson mobility-edge benchmark). Consumes the JLD2 task files
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
    allow_partial = false
    i = 1
    while i <= length(args)
        if args[i] == "--manifest"
            manifest = args[i + 1]; i += 2
        elseif args[i] == "--datadir"
            datadir = args[i + 1]; i += 2
        elseif args[i] == "--save"
            save_path = args[i + 1]; i += 2
        elseif args[i] == "--allow-partial"
            allow_partial = true; i += 1
        else
            error("unknown argument: $(args[i])")
        end
    end
    (manifest === nothing || datadir === nothing) &&
        error("--manifest and --datadir are required")
    return (; manifest, datadir, save_path, allow_partial)
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
# Ec must exceed E_hi for the basis to be real, so the scan is bounded below at
# E_hi + 0.05; an optimum ON that bound is a constraint artifact, not a
# measurement, and is flagged as pinned. Fit Ec across several windows
# (edge_fit_profile) and trust only window-stable interior optima.
function edge_fit(E, sigma, E_lo, E_hi; x_fixed=nothing)
    sel = findall(e -> E_lo <= e <= E_hi, E)
    Ec_grid = filter(>(E_hi + 0.05), 7.0:0.005:9.4)
    best = (Inf, NaN, NaN, NaN)
    for Ec in Ec_grid
        xs = x_fixed === nothing ? (0.3:0.05:3.0) : (x_fixed:x_fixed)
        for x in xs
            basis = (Ec .- E[sel]) .^ x
            A = (basis' * sigma[sel]) / (basis' * basis)
            sse = sum(abs2, sigma[sel] .- A .* basis)
            sse < best[1] && (best = (sse, Ec, x, A))
        end
    end
    pinned = best[2] ≈ first(Ec_grid)
    return (; sse=best[1], Ec=best[2], x=best[3], A=best[4], pinned)
end

edge_windows(E_hi_max) = [(6.0, E_hi) for E_hi in (7.0, 7.5, E_hi_max)]

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
        # Fail-closed loading: exactly one file per expected pair, each with a
        # params dict that matches the manifest (seed included). The +eps/-eps
        # partners share one probe stream, so a pair is ONE independent
        # disorder draw stored as two (parity-related) moment matrices.
        seed_base = Int(ens["seed_base"])
        expected = Dict{String,Any}(
            "W" => Float64(model["W"]), "t" => Float64(model["t"]),
            "a" => a, "b" => b, "L" => L, "NC" => NC_stored,
            "NR" => Int(mf["kpm"]["NR"]),
            "arr_size" => Int(mf["kpm"]["arr_size"]), "schema" => 1)
        mu_sum = zeros(ComplexF64, NC_stored, NC_stored)
        n_pairs = 0
        missing_pairs = Int[]
        for pair_index in 1:pairs_per_size
            f = joinpath(cli.datadir,
                @sprintf("anderson_seebeck_L%02d_NC%04d_pair%02d.jld2",
                         L, NC_stored, pair_index))
            if !isfile(f)
                push!(missing_pairs, pair_index)
                continue
            end
            p = load(f, "params")
            expected["seed"] = seed_base + 10_000 * si + pair_index
            expected["pair_index"] = pair_index
            for (k, v) in expected
                get(p, k, nothing) == v ||
                    error("$f: params[$k] = $(get(p, k, nothing)) != manifest-expected $v")
            end
            mu_plus = load(f, "mu_plus"); mu_minus = load(f, "mu_minus")
            size(mu_plus) == size(mu_minus) == (NC_stored, NC_stored) ||
                error("$f: moment matrices are not $(NC_stored)x$(NC_stored)")
            mu_sum .+= mu_plus .+ mu_minus
            n_pairs += 1
        end
        if !isempty(missing_pairs)
            msg = "L=$L: missing task files for pairs $missing_pairs"
            cli.allow_partial || error(msg * " (pass --allow-partial to proceed)")
            @warn msg * " — proceeding on the partial ensemble (--allow-partial)"
        end
        if n_pairs == 0
            println("L=$L: no task files found — skipping")
            continue
        end
        mu_avg = mu_sum ./ (2 * n_pairs)
        ph = particle_hole_residual(mu_avg)
        @printf("\n================ L=%d: %d/%d independent pairs (%d moment matrices), PH residual %.2e ================\n",
                L, n_pairs, pairs_per_size, 2 * n_pairs, ph)

        for NC in unique(clamp.((256, 384, 512, 768, 1024), 0, NC_stored))
            m = KPM.ConductivityMoments(mu_avg[1:NC, 1:NC], a, b, NH,
                                        Int(mf["kpm"]["NR"]))
            E_grid = collect(range(4.0, 9.5; length=1101))
            sigma = KPM.transport_distribution(m, E_grid; volume=volume)
            fits = [(w, edge_fit(E_grid, sigma, w...)) for w in edge_windows(8.1)]
            for ((E_lo, E_hi), r) in fits
                @printf("L=%2d NC=%4d (pi*a/NC=%.3ft) window [%.1f, %.1f]: Ec_eff=%.3f x_eff=%.2f%s\n",
                        L, NC, pi * a / NC, E_lo, E_hi, r.Ec, r.x,
                        r.pinned ? "  [PINNED at scan bound — not a measurement]" : "")
            end
            r15 = edge_fit(E_grid, sigma, 6.0, 7.5; x_fixed=x_ref)
            @printf("    x=%.1f fixed, window [6.0, 7.5]: Ec=%.3f%s\n",
                    x_ref, r15.Ec, r15.pinned ? "  [PINNED]" : "")
            Ec_f, x_f = fits[2][2].Ec, fits[2][2].x
            any_pinned = any(r.pinned for (_, r) in fits)

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
                         "Ec_eff" => Ec_f, "x_eff" => x_f,
                         "edge_fit_pinned" => any_pinned, "ph_residual" => ph,
                         "n_pairs" => n_pairs, "n_matrices" => 2 * n_pairs)
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
