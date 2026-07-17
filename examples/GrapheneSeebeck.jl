"""
GrapheneSeebeck.jl

Benchmark KPM.jl's Chester--Thellung/Kubo--Greenwood thermoelectric
reconstruction against the full finite-temperature clean-Dirac Seebeck curve
for pristine nearest-neighbor graphene. The example checks the normalized
low-energy transport distribution, particle-hole symmetry, temperature
collapse, the universal finite-temperature peak, and moment-resolution
stability.

Perfectly clean infinite graphene has a zero-frequency Drude delta rather
than an ordinary finite dissipative bulk DC conductivity. Jackson damping
regularizes that delta, so the absolute clean transport distribution and L0
are kernel and resolution dependent. The benchmarked observable is the
ratio S = -beta*L1/L0, together with its symmetry and convergence properties.

The fast defaults use temperatures chosen for numerical resolution, not for
a room-temperature graphene claim. QuadGK.jl and Plots.jl are example-only
dependencies. If either is missing, activate the intended environment and
run `import Pkg; Pkg.add(["QuadGK", "Plots"])`.
"""

using KPM
using LinearAlgebra
using Random
using Printf
using Statistics
using Serialization

try
    @eval using QuadGK
catch err
    println(stderr, "QuadGK.jl is required for this example.")
    println(stderr, "Install the example-only dependency with:")
    println(stderr, "  import Pkg; Pkg.add(\"QuadGK\")")
    rethrow(err)
end

# GR otherwise tries to open its Qt workstation on some headless macOS runs,
# even when the only requested action is saving a PNG.
if "--out" in ARGS
    ENV["GKSwstype"] = "100"
end

try
    @eval using Plots
catch err
    println(stderr, "Plots.jl is required for this example.")
    println(stderr, "Install the example-only dependency with:")
    println(stderr, "  import Pkg; Pkg.add(\"Plots\")")
    rethrow(err)
end

include(joinpath(@__DIR__, "GrapheneModel.jl"))

# Provisional fast parameters from examples/GRAPHENE_SEEBECK_PLAN.md.
const Nx = 48
const Ny = 48
const t = 1.0
const a = 3.1
const b = 0.0
const NC = 512
const NR = 4
const arr_size = 16
const kBT_values = (0.04, 0.06)
const eta_values = collect(range(-8.0, 8.0; length=33))
const quad_N = 4 * NC
const seed = Int(0x6e617068)
const area = Nx * Ny * 3sqrt(3) / 2
const g_J = 2.0
const CACHE_PATH = nothing

module DiracReference

using QuadGK
import ..KPM

export A0, seebeck_dimensionless, seebeck_uVK, verify

# Algebraically 1/(4cosh(u/2)^2), without overflow at large |u|.
fermi_weight(u::Real) = begin
    q = exp(-abs(u))
    q / (1 + q)^2
end

function moments(eta::Real)
    eta_f = Float64(eta)
    lower = eta_f - 60.0
    upper = eta_f + 60.0
    A0_value, _ = quadgk(lower, 0.0, upper; rtol=2e-13, atol=1e-14) do y
        abs(y) * fermi_weight(y - eta_f)
    end
    A1_value, _ = quadgk(lower, 0.0, upper; rtol=2e-13, atol=1e-14) do y
        (y - eta_f) * abs(y) * fermi_weight(y - eta_f)
    end
    return A0_value, A1_value
end

A0(eta::Real) = first(moments(eta))

function seebeck_dimensionless(eta::Real)
    A0_value, A1_value = moments(eta)
    return -A1_value / A0_value
end

# KPM.seebeck_uVK intentionally has no scalar overload.
seebeck_uVK(eta::Real) =
    KPM.KB_OVER_E_UV_PER_K * seebeck_dimensionless(eta)

function golden_maximum(f, left::Real, right::Real; atol=1e-10)
    phi = (sqrt(5.0) - 1) / 2
    lo = Float64(left)
    hi = Float64(right)
    c = hi - phi * (hi - lo)
    d = lo + phi * (hi - lo)
    fc = f(c)
    fd = f(d)
    while hi - lo > atol
        if fc > fd
            hi = d
            d = c
            fd = fc
            c = hi - phi * (hi - lo)
            fc = f(c)
        else
            lo = c
            c = d
            fc = fd
            d = lo + phi * (hi - lo)
            fd = f(d)
        end
    end
    x = (lo + hi) / 2
    return x, f(x)
end

function verify()
    anchors = ((0.0, 0.000000000),
               (1.0, -0.804632010),
               (2.0, -1.001885714),
               (4.0, -0.761028423),
               (10.0, -0.328874870))
    for (eta, expected) in anchors
        actual = seebeck_dimensionless(eta)
        @assert isapprox(actual, expected; atol=1e-8, rtol=0.0) "Dirac anchor failed at eta=$eta: got $actual, expected $expected"
    end

    denominator_checks = Tuple{Float64,Float64,Float64}[]
    for eta in (0.0, 1.0, 2.0, 4.0, 10.0)
        actual = A0(eta)
        closed = 2log(2cosh(eta / 2))
        @assert isapprox(actual, closed; atol=1e-10, rtol=0.0) "A0 closed-form check failed at eta=$eta"
        push!(denominator_checks, (eta, actual, closed))
    end

    @assert abs(seebeck_dimensionless(0.0)) < 1e-12
    for eta in (0.5, 1.0, 2.0, 4.0, 10.0)
        oddness = abs(seebeck_dimensionless(eta) + seebeck_dimensionless(-eta))
        @assert oddness < 1e-10 "Dirac oddness check failed at eta=$eta: $oddness"
    end

    peak_eta, peak_value = golden_maximum(
        eta -> abs(seebeck_dimensionless(eta)), 1.0, 3.0)
    @assert abs(peak_eta - 1.94877546) < 1e-6
    @assert abs(peak_value - 1.002258028) < 1e-6

    exact_ten = seebeck_dimensionless(10.0)
    mott_ten = -pi^2 / 30
    mott_rel = abs(exact_ten / mott_ten - 1)
    @assert mott_rel < 1e-3
    return (; anchors, denominator_checks, peak_eta, peak_value, mott_rel)
end

end # module DiracReference

function parse_cli(args)
    out_path = nothing
    cache_path = CACHE_PATH
    i = 1
    while i <= length(args)
        if args[i] == "--out"
            i < length(args) || error("--out requires a path")
            out_path = args[i + 1]
            i += 2
        elseif args[i] == "--cache"
            i < length(args) || error("--cache requires a path")
            cache_path = args[i + 1]
            i += 2
        elseif args[i] in ("-h", "--help")
            println("usage: julia GrapheneSeebeck.jl [--cache PATH] [--out PATH.png]")
            exit(0)
        else
            error("unknown argument: $(args[i])")
        end
    end
    return (; out_path, cache_path)
end

function cache_parameters()
    return (; schema=1, Nx, Ny, t, a, b, NC, NR, arr_size, seed,
            model=:GrapheneLattice_gapless,
            twist_scheme=:cartesian_0_plusminus_pi_over_2_pi,
            probe_api=:one_seeded_rng_stream)
end

function read_cache(path, params)
    path === nothing && return nothing
    isfile(path) || return nothing
    try
        payload = open(deserialize, path)
        if hasproperty(payload, :params) && payload.params == params &&
           hasproperty(payload, :mu) && size(payload.mu) == (NC, NC) &&
           hasproperty(payload, :mu_periodic) && size(payload.mu_periodic) == (NC, NC)
            return payload
        end
        println("Cache exists but parameters/schema do not match; recomputing moments.")
    catch err
        @warn "Could not read moment cache; recomputing." path exception=(err, catch_backtrace())
    end
    return nothing
end

function write_cache(path, payload)
    path === nothing && return
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        serialize(io, payload)
    end
end

relative_difference(x, y) =
    (!isfinite(x) || !isfinite(y)) ? Inf :
    abs(x - y) / max(abs(x), abs(y), eps(Float64))

passfail(flag) = flag ? "PASS" : "FAIL"

function particle_hole_project(mu)
    projected = copy(mu)
    odd_scale = 0.0
    full_scale = maximum(abs, mu)
    for j in axes(projected, 2), i in axes(projected, 1)
        if isodd((i - 1) + (j - 1))
            odd_scale = max(odd_scale, abs(projected[i, j]))
            projected[i, j] = zero(eltype(projected))
        end
    end
    return projected, odd_scale / full_scale
end

function accepted_indices(kBT)
    return findall(eta -> abs(eta * kBT) <= 0.35t + 10eps(Float64), eta_values)
end

function main()
    cli = parse_cli(ARGS)
    total_start = time()
    NH = 2 * Nx * Ny
    kT1, kT2 = kBT_values
    sigma_window = (0.08, 0.35)
    E_ref = 0.2

    println("\n==============================================================================")
    println("Clean-graphene CTKG Seebeck benchmark (fast default)")
    println("==============================================================================")
    @printf("Nx=Ny=%d (NH=%d), t=%.1f, NC=%d, NR=%d, arr_size=%d\n",
            Nx, NH, t, NC, NR, arr_size)
    @printf("fixed a=%.1f, b=%.1f, pi*a/NC=%.6f; kBT=%s\n",
            a, b, pi * a / NC, string(kBT_values))
    @printf("eta grid=[%.1f, %.1f], %d points; area=%.6f; g_J=%.1f; seed=%d\n",
            first(eta_values), last(eta_values), length(eta_values), area, g_J, seed)
    println("Clean finite-size control: inversion-symmetric 4x4 twist grid (16 moment matrices).")
    println("The periodic twist is included; stochastic probes come from one Xoshiro(seed) stream.")
    println("Dirac gate restriction: |mu| <= 0.35t at each temperature.")
    println("Points outside that restriction are computed and plotted with hollow markers, but excluded from every gate.")

    println("\n-- Deterministic finite-temperature Dirac reference --")
    reference_start = time()
    verification = DiracReference.verify()
    for (eta, expected) in verification.anchors
        actual = DiracReference.seebeck_dimensionless(eta)
        @printf("eta=%5.1f  quadrature=% .9f  anchor=% .9f  |delta|=%.2e\n",
                eta, actual, expected, abs(actual - expected))
    end
    for (eta, actual, closed) in verification.denominator_checks
        @printf("A0 eta=%4.1f  quadrature=%.12f  closed=%.12f  |delta|=%.2e\n",
                eta, actual, closed, abs(actual - closed))
    end
    @printf("Peak: |S|/(kB/|e|)=%.9f at |eta|=%.8f\n",
            verification.peak_value, verification.peak_eta)
    @printf("Mott check at eta=10: relative difference = %.6f%%\n",
            100 * verification.mott_rel)
    println("Deterministic reference assertions passed: anchors, A0, S(0), oddness, peak, and Mott tail.")
    reference_seconds = time() - reference_start

    println("\n-- Graphene operators and fixed rescaling --")
    build_start = time()
    H, Jx, _, _, _, _ = GrapheneLattice(Nx, Ny; t=t, Δ=0.0)
    h_residual = norm(H - H') / norm(H)
    j_residual = norm(Jx + Jx') / norm(Jx)
    @assert h_residual <= 1e-13 "H Hermiticity residual is $h_residual"
    @assert j_residual <= 1e-13 "Jx anti-Hermiticity residual is $j_residual"
    @printf("H Hermitian relative residual       = %.3e\n", h_residual)
    @printf("Jx anti-Hermitian relative residual = %.3e\n", j_residual)
    println("Operator assertions passed (relative tolerance 1e-13).")
    h = KPM.RescaledHamiltonian(H ./ a, a, b)
    build_seconds = time() - build_start

    println("\n-- Conductivity moments --")
    params = cache_parameters()
    payload = read_cache(cli.cache_path, params)
    moment_seconds = 0.0
    if payload === nothing
        twist_angles = (0.0, pi / 2, pi, -pi / 2)
        twists = vec(collect(Iterators.product(twist_angles, twist_angles)))
        @assert length(twists) == 16
        probe_rng = Xoshiro(seed)
        mu_sum = zeros(ComplexF64, NC, NC)
        mu_periodic = nothing
        for (twist_index, (theta_x, theta_y)) in enumerate(twists)
            if iszero(theta_x) && iszero(theta_y)
                H_twist, Jx_twist = H, Jx
                h_twist = h
            else
                twist_build_start = time()
                H_twist, Jx_twist, _, _, _, _ = GrapheneLattice(
                    Nx, Ny; t=t, Δ=0.0,
                    bc_factor_x=exp(im * theta_x),
                    bc_factor_y=exp(im * theta_y))
                build_seconds += time() - twist_build_start
                @assert norm(H_twist - H_twist') / norm(H_twist) <= 1e-13
                @assert norm(Jx_twist + Jx_twist') / norm(Jx_twist) <= 1e-13
                h_twist = KPM.RescaledHamiltonian(H_twist ./ a, a, b)
            end
            @printf("twist %2d/%d: (theta_x/pi, theta_y/pi)=(% .2f,% .2f)\n",
                    twist_index, length(twists), theta_x / pi, theta_y / pi)
            moment_start = time()
            m_twist = KPM.cond_moments(
                h_twist, Jx_twist, Jx_twist; NC=NC, NR=NR,
                rng=probe_rng, arr_size=arr_size)
            moment_seconds += time() - moment_start
            mu_sum .+= m_twist.mu
            if iszero(theta_x) && iszero(theta_y)
                mu_periodic = copy(m_twist.mu)
            end
        end
        @assert mu_periodic !== nothing
        payload = (; params, mu=mu_sum ./ length(twists), mu_periodic)
        if cli.cache_path === nothing
            println("Moment cache disabled (use --cache PATH to enable it).")
        else
            write_cache(cli.cache_path, payload)
            println("Wrote matching moment cache: $(abspath(cli.cache_path))")
        end
    else
        println("Reloaded matching moment cache: $(abspath(cli.cache_path))")
    end
    @assert size(payload.mu) == (NC, NC)
    mu_projected, raw_ph_residual = particle_hole_project(payload.mu)
    mu_periodic_projected, periodic_ph_residual =
        particle_hole_project(payload.mu_periodic)
    mxx = KPM.ConductivityMoments(mu_projected, a, b, NH, NR)
    mxx_periodic = KPM.ConductivityMoments(
        mu_periodic_projected, a, b, NH, NR)
    @printf("Stored twist-averaged conductivity moment matrix: %d x %d ComplexF64\n",
            size(payload.mu)...)
    @printf("Raw odd-(n+m) moment residual: average %.3e, periodic %.3e\n",
            raw_ph_residual, periodic_ph_residual)
    println("Applied the exact bipartite particle-hole projection before reconstruction.")

    println("\n-- Transport distribution and thermoelectric reconstruction --")
    reconstruction_start = time()
    E_grid = collect(range(-0.4, 0.4; length=401))
    sigma = KPM.transport_distribution(mxx, E_grid; volume=area, g_J=g_J)
    sigma_periodic = KPM.transport_distribution(
        mxx_periodic, E_grid; volume=area, g_J=g_J)
    i_ref = argmin(abs.(E_grid .- E_ref))
    @assert isapprox(E_grid[i_ref], E_ref; atol=100eps(Float64), rtol=0.0)
    sigma_ref = sigma[i_ref]
    shape_indices = findall(E -> sigma_window[1] <= abs(E) <= sigma_window[2], E_grid)
    sigma_normalized = sigma ./ sigma_ref
    dirac_shape = abs.(E_grid) ./ E_ref
    shape_deviations = abs.(sigma_normalized[shape_indices] .-
                            dirac_shape[shape_indices]) ./ dirac_shape[shape_indices]
    sigma_shape_deviation = maximum(shape_deviations)
    sigma_evenness = maximum(abs.(sigma[shape_indices] .-
                                  reverse(sigma)[shape_indices])) /
                     maximum(abs, sigma[shape_indices])
    sigma_periodic_normalized = sigma_periodic ./ sigma_periodic[i_ref]
    periodic_shape_deviation = maximum(
        abs.(sigma_periodic_normalized[shape_indices] .-
             dirac_shape[shape_indices]) ./ dirac_shape[shape_indices])
    periodic_evenness = maximum(abs.(sigma_periodic[shape_indices] .-
                                     reverse(sigma_periodic)[shape_indices])) /
                        maximum(abs, sigma_periodic[shape_indices])
    @printf("Sigma(E) shape window: %.2f <= |E| <= %.2f; E_ref=%.2f\n",
            sigma_window..., E_ref)
    @printf("Periodic-only shell diagnostic: max shape deviation %.3f%%; evenness %.3e\n",
            100 * periodic_shape_deviation, periodic_evenness)
    @printf("Sigma(E_ref)=%.6e; max normalized |E|-shape relative deviation = %.3f%%\n",
            sigma_ref, 100 * sigma_shape_deviation)
    @printf("Sigma evenness residual over that window = %.3e\n", sigma_evenness)

    results = Dict{Float64,Vector{Any}}()
    S_values = Dict{Float64,Vector{Float64}}()
    L0_values = Dict{Float64,Vector{Float64}}()
    accepted = Dict(kBT => accepted_indices(kBT) for kBT in kBT_values)
    for kBT in kBT_values
        beta = inv(kBT)
        r = [KPM.thermoelectric(mxx, eta * kBT;
                               beta=beta, volume=area, g_J=g_J,
                               sigma_min=0.0, quad_N=quad_N)
             for eta in eta_values]
        results[kBT] = r
        S_values[kBT] = KPM.seebeck_uVK.(r)
        L0_values[kBT] = getfield.(r, :L0)
        @assert all(isfinite, L0_values[kBT]) && all(>(0.0), L0_values[kBT])
        @printf("kBT=%.2f: accepted %d/%d points (|eta| <= %.1f); min L0 over all points = %.6e\n",
                kBT, length(accepted[kBT]), length(eta_values),
                maximum(abs, eta_values[accepted[kBT]]), minimum(L0_values[kBT]))
    end
    println("L0 > 0 assertion passed at every evaluated thermoelectric point (sigma_min=0.0).")

    S_reference = DiracReference.seebeck_uVK.(eta_values)
    zero_index = findfirst(iszero, eta_values)
    println("\nKPM particle-hole diagnostics (exact bipartite projection; raw stochastic residual reported above):")
    for kBT in kBT_values
        oddness = maximum(abs.(S_values[kBT] .+ reverse(S_values[kBT])))
        @printf("  kBT=%.2f: S(0)=% .6f uV/K; max |S(eta)+S(-eta)|=% .6f uV/K\n",
                kBT, S_values[kBT][zero_index], oddness)
    end

    println("\nQuadrature-order spot checks (sigma_min=0.0):")
    for eta in (1.0, 4.0)
        i = findfirst(==(eta), eta_values)
        r4 = results[kT1][i]
        r8 = KPM.thermoelectric(mxx, eta * kT1;
                               beta=inv(kT1), volume=area, g_J=g_J,
                               sigma_min=0.0, quad_N=8 * NC)
        @printf("  eta=%3.1f, kBT=%.2f: S4=% .6f, S8=% .6f uV/K, rel.dS=%.3e; L0_4=%.6e, L0_8=%.6e, rel.dL0=%.3e\n",
                eta, kT1, KPM.seebeck_uVK(r4), KPM.seebeck_uVK(r8),
                relative_difference(KPM.seebeck_uVK(r4), KPM.seebeck_uVK(r8)),
                r4.L0, r8.L0, relative_difference(r4.L0, r8.L0))
    end

    println("\nNC-resolution diagnostic from the stored moment matrix (kBT=0.04):")
    NC_values = (256, 384, 512)
    trunc_results = Dict{Int,Vector{Any}}()
    trunc_S = Dict{Int,Vector{Float64}}()
    trunc_L0 = Dict{Int,Vector{Float64}}()
    for NCt in NC_values
        if NCt == NC
            r = results[kT1]
        else
            mt = KPM.ConductivityMoments(
                mu_projected[1:NCt, 1:NCt], a, b, NH, NR)
            r = [KPM.thermoelectric(mt, eta * kT1;
                                   beta=inv(kT1), volume=area, g_J=g_J,
                                   sigma_min=0.0, quad_N=4 * NCt)
                 for eta in eta_values]
        end
        trunc_results[NCt] = r
        trunc_S[NCt] = KPM.seebeck_uVK.(r)
        trunc_L0[NCt] = getfield.(r, :L0)
    end
    @printf("  eta    NC          S (uV/K)              L0        L0/L0_512\n")
    println("  ----------------------------------------------------------------")
    for eta in (1.0, 2.0, 4.0)
        i = findfirst(==(eta), eta_values)
        for NCt in NC_values
            @printf("  %3.1f   %3d     %13.6f      %12.6e       %9.6f\n",
                    eta, NCt, trunc_S[NCt][i], trunc_L0[NCt][i],
                    trunc_L0[NCt][i] / trunc_L0[NC][i])
        end
    end
    nc_indices = accepted[kT1]
    nc_max_change = maximum(abs.(trunc_S[384][nc_indices] .-
                                 trunc_S[512][nc_indices]))

    common_indices = intersect(accepted[kT1], accepted[kT2])
    collapse_difference = maximum(abs.(S_values[kT1][common_indices] .-
                                       S_values[kT2][common_indices]))
    comparison_errors = Float64[]
    for kBT in kBT_values
        indices = filter(i -> 1 <= abs(eta_values[i]) <= 6, accepted[kBT])
        append!(comparison_errors, abs.(S_values[kBT][indices] .- S_reference[indices]))
    end
    median_reference_error = median(comparison_errors)
    max_neg_weight = maximum(results[kBT][i].neg_weight
                             for kBT in kBT_values for i in accepted[kBT])

    peak_ok = true
    peak_lines = String[]
    for kBT in kBT_values
        for (carrier, side, expected_sign) in (("holes", -1, 1), ("electrons", 1, -1))
            indices = filter(i -> sign(eta_values[i]) == side, accepted[kBT])
            local_i = argmax(abs.(S_values[kBT][indices]))
            i = indices[local_i]
            peak_eta = eta_values[i]
            peak_S = S_values[kBT][i]
            this_ok = abs(abs(peak_S) / 86.3679 - 1) < 0.10 &&
                      sign(peak_S) == expected_sign && 1.0 <= abs(peak_eta) <= 3.0
            peak_ok &= this_ok
            push!(peak_lines,
                  @sprintf("  kBT=%.2f %-9s peak: eta=% .2f, S=% .6f uV/K (%s)",
                           kBT, carrier, peak_eta, peak_S, passfail(this_ok)))
        end
    end

    println("\nFast-run validation gates (diagnostic, non-aborting):")
    @printf("  [%s] Sigma ~ |E| within 5%% on %.2f <= |E| <= %.2f: max deviation %.3f%%\n",
            passfail(sigma_shape_deviation < 0.05), sigma_window...,
            100 * sigma_shape_deviation)
    @printf("  [%s] two-temperature collapse < %.4f uV/K on common accepted points: %.6f uV/K\n",
            passfail(collapse_difference < 0.05 * KPM.KB_OVER_E_UV_PER_K),
            0.05 * KPM.KB_OVER_E_UV_PER_K, collapse_difference)
    @printf("  [%s] KPM peaks within 10%% of 86.3679 uV/K with correct sign and side\n",
            passfail(peak_ok))
    foreach(println, peak_lines)
    @printf("  [%s] median |S_KPM-S_ref| < 5 uV/K for 1 <= |eta| <= 6 (accepted only): %.6f uV/K\n",
            passfail(median_reference_error < 5.0), median_reference_error)
    @printf("  [%s] neg_weight < 1%% at accepted points: maximum %.6f%%\n",
            passfail(max_neg_weight < 0.01), 100 * max_neg_weight)
    @printf("  [%s] max |S_NC384-S_NC512| < 5 uV/K on full kBT=0.04 accepted grid: %.6f uV/K\n",
            passfail(nc_max_change < 5.0), nc_max_change)
    println("Load-bearing qualification: the clean absolute L0 is a kernel-regularized Drude weight, not a physical bulk DC conductivity; S is the benchmarked observable.")

    reconstruction_seconds = time() - reconstruction_start

    println("\nTerminal summary (all S columns in uV/K; A=accepted, O=outside Dirac window):")
    @printf("    eta    S_ref      S(%.2f)       L0(%.2f)  %s    S(%.2f)       L0(%.2f)  %s\n",
            kT1, kT1, "gate", kT2, kT2, "gate")
    println("  --------------------------------------------------------------------------------")
    for (i, eta) in pairs(eta_values)
        flag1 = i in accepted[kT1] ? "A" : "O"
        flag2 = i in accepted[kT2] ? "A" : "O"
        @printf("  %6.2f  %9.4f  %11.4f  %12.5e  %s  %11.4f  %12.5e  %s\n",
                eta, S_reference[i], S_values[kT1][i], L0_values[kT1][i], flag1,
                S_values[kT2][i], L0_values[kT2][i], flag2)
    end

    println("\n-- Four-panel figure --")
    plotting_start = time()
    colors = (:royalblue, :darkorange)
    legend_style = (; legend_background_color=:transparent,
                    legend_foreground_color=:transparent)

    p1 = Plots.plot(E_grid, sigma_normalized; color=:royalblue, linewidth=2,
                    xlabel="E/t", ylabel="Sigma(E) / Sigma(0.2t)",
                    title=@sprintf("Dirac weight: %.2f <= |E| <= %.2f", sigma_window...),
                    label="KPM, 16 clean twists", legend=:topleft, legend_style...)
    Plots.plot!(p1, E_grid, dirac_shape; color=:black, linestyle=:dash,
                linewidth=2, label="|E| / 0.2t")
    Plots.vspan!(p1, [-sigma_window[1], sigma_window[1]]; color=:gray,
                 alpha=0.22, label="kernel-rounded exclusion")
    Plots.vline!(p1, [-sigma_window[2], sigma_window[2]]; color=:gray,
                 linestyle=:dot, label="Dirac-window edge")
    Plots.ylims!(p1, (-0.15, 2.2))

    eta_dense = collect(range(first(eta_values), last(eta_values); length=161))
    S_dense = DiracReference.seebeck_uVK.(eta_dense)
    p2 = Plots.plot(eta_dense, S_dense; color=:black, linewidth=2.5,
                    xlabel="eta = mu/(kBT)", ylabel="S (uV/K)",
                    title="Full finite-T Dirac curve", label="analytic Dirac",
                    legend=:bottomright, legend_style...)
    for (color, kBT) in zip(colors, kBT_values)
        Plots.plot!(p2, eta_values, S_values[kBT]; color=color, linewidth=1.5,
                    label=@sprintf("KPM kBT=%.2f", kBT))
        inside = accepted[kBT]
        outside = setdiff(eachindex(eta_values), inside)
        Plots.scatter!(p2, eta_values[inside], S_values[kBT][inside];
                       marker=:circle, markersize=4, markercolor=color,
                       markerstrokecolor=color, label=false)
        isempty(outside) || Plots.scatter!(
            p2, eta_values[outside], S_values[kBT][outside]; marker=:circle,
            markersize=4.5, markercolor=:white, markerstrokecolor=color,
            markerstrokewidth=1.5, label=@sprintf("outside |mu|<=0.35, %.2f", kBT))
    end

    p3 = Plots.plot(; xlabel="eta = mu/(kBT)", ylabel="S_KPM - S_ref (uV/K)",
                    title="Residuals and Mott reference", legend=:topleft,
                    legend_style...)
    Plots.hline!(p3, [0.0]; color=:black, linewidth=1, label=false)
    for (color, kBT) in zip(colors, kBT_values)
        residual = S_values[kBT] .- S_reference
        Plots.plot!(p3, eta_values, residual; color=color, linewidth=1.5,
                    label=@sprintf("residual kBT=%.2f", kBT))
        inside = accepted[kBT]
        outside = setdiff(eachindex(eta_values), inside)
        Plots.scatter!(p3, eta_values[inside], residual[inside]; color=color,
                       marker=:circle, markersize=3.5, label=false)
        isempty(outside) || Plots.scatter!(
            p3, eta_values[outside], residual[outside]; marker=:circle,
            markersize=4, markercolor=:white, markerstrokecolor=color,
            markerstrokewidth=1.3, label=false)
    end
    mott_eta_negative = collect(range(-8.0, -3.0; length=51))
    mott_eta_positive = collect(range(3.0, 8.0; length=51))
    mott_scale = KPM.KB_OVER_E_UV_PER_K * pi^2 / 3
    p3_mott = Plots.twinx(p3)
    Plots.plot!(p3_mott, mott_eta_negative, -mott_scale ./ mott_eta_negative;
                color=:gray35, linestyle=:dash, linewidth=2,
                ylabel="Mott S (uV/K)", label="Mott S, |eta|>=3",
                legend=:topright, legend_style...)
    Plots.plot!(p3_mott, mott_eta_positive, -mott_scale ./ mott_eta_positive;
                color=:gray35, linestyle=:dash, linewidth=2, label=false)

    panel_etas = (1.0, 2.0, 4.0)
    p4 = Plots.plot(; xlabel="NC", ylabel="S (uV/K)",
                    title="Stored-moment truncation, kBT=0.04",
                    xticks=(collect(NC_values), string.(NC_values)),
                    legend=:bottomleft, legend_style...)
    panel_colors = (:seagreen, :purple, :firebrick)
    for (color, eta) in zip(panel_colors, panel_etas)
        i = findfirst(==(eta), eta_values)
        Plots.plot!(p4, collect(NC_values), [trunc_S[n][i] for n in NC_values];
                    color=color, marker=:circle, linewidth=2,
                    label=@sprintf("S, eta=%.0f", eta))
    end
    p4_L0 = Plots.twinx(p4)
    for (color, eta) in zip(panel_colors, panel_etas)
        i = findfirst(==(eta), eta_values)
        normalized_L0 = [trunc_L0[n][i] / trunc_L0[NC][i] for n in NC_values]
        Plots.plot!(p4_L0, collect(NC_values), normalized_L0; color=color,
                    marker=:diamond, linestyle=:dash, linewidth=1.7,
                    ylabel="L0 / L0(NC=512)",
                    label=@sprintf("L0 norm., eta=%.0f", eta),
                    legend=:topright, legend_style...)
    end

    fig = Plots.plot(p1, p2, p3, p4; layout=(2, 2), size=(1400, 1050),
                     margin=6 * Plots.mm, right_margin=14 * Plots.mm,
                     plot_title="Clean graphene CTKG benchmark -- S is the benchmarked observable")
    show(devnull, MIME("image/png"), fig)
    if cli.out_path === nothing
        display(fig)
        println("Displayed figure (use --out PATH.png to save instead).")
    else
        mkpath(dirname(abspath(cli.out_path)))
        Plots.savefig(fig, cli.out_path)
        println("Saved figure: $(abspath(cli.out_path))")
    end
    plotting_seconds = time() - plotting_start
    total_seconds = time() - total_start

    println("\nTiming summary:")
    @printf("  Deterministic reference:          %8.3f s\n", reference_seconds)
    @printf("  Hamiltonian/current construction: %8.3f s\n", build_seconds)
    @printf("  Conductivity moments:             %8.3f s\n", moment_seconds)
    @printf("  Reconstruction/diagnostics:       %8.3f s\n", reconstruction_seconds)
    @printf("  Plotting/rendering:                %8.3f s\n", plotting_seconds)
    @printf("  Total runtime:                     %8.3f s\n", total_seconds)
end

main()
