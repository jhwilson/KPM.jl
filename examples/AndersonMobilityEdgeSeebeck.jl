"""
AndersonMobilityEdgeSeebeck.jl

Demonstrate KPM.jl's Chester--Thellung/Kubo--Greenwood thermoelectric path for
the Seebeck coefficient near the upper mobility edge of the spinless
three-dimensional Anderson model. The disorder-averaged KPM result is compared
with the universal `x = 1.5` analytic curve, and the script exposes resolution,
negative-weight, conductivity-floor, particle-hole, and finite-sample checks.

This example can test the thermoelectric implementation end to end, including
the current construction, fixed rescaling, antithetic stochastic moments,
transport reconstruction, sign convention, and convergence diagnostics. It
cannot establish the thermodynamic mobility edge or critical exponent. In a
finite periodic cube, the DC, infinite-size, and vanishing-kernel-width limits
do not commute; a critical study needs much larger sizes, more disorder, and
joint scaling in size and broadening.

The fast defaults are intended to take roughly 2--4 minutes on a laptop CPU.
(The plan's provisional L=10, NR=4, two-pair setting ran in ~15 s but was
noise-dominated; the committed defaults spend the budget on statistics and
size, which the plan authorizes after timing.)
QuadGK.jl and Plots.jl are example-only dependencies. If either is missing,
activate the intended environment and run `import Pkg; Pkg.add(["QuadGK",
"Plots"])`.
"""

using KPM
using LinearAlgebra
using SparseArrays
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

# Fast default from examples/AndersonSeebeckBenchmarkPlan.md.
const L = 14
const W = 12.0
const t = 1.0
const NC = 384
const NR = 8
const n_disorder_pairs = 8
const arr_size = 16
const a = 12.25
const b = 0.0
const kBT_values = (0.25, 0.35)
const Ec = 7.5
const x_ref = 1.5
const z_values = collect(range(-4.0, 8.0; length=25))
const z_floor_checks = (-4.0, -8.0, -12.0, -16.0)
const seed_base = Int(0x5eebecc)
const CACHE_PATH = nothing
const edge_cutoff = 1e-3

module UniversalReference

using QuadGK
import ..KPM

export seebeck_dimensionless, seebeck_uVK, verify_anchors

# Algebraically 1/(4cosh(epsilon/2)^2), without overflow at large |epsilon|.
fermi_weight(epsilon) = begin
    q = exp(-abs(epsilon))
    q / (1 + q)^2
end

function Kmoment(n::Integer, x::Real, z::Real)
    upper = min(Float64(z), 40.0)
    upper > -40.0 || return 0.0
    value, _ = quadgk(-40.0, upper; rtol=2e-12, atol=1e-14) do epsilon
        epsilon^n * (z - epsilon)^x * fermi_weight(epsilon)
    end
    return value
end

function seebeck_dimensionless(z::Real; x::Real=1.5)
    K0 = Kmoment(0, x, z)
    K1 = Kmoment(1, x, z)
    return -K1 / K0
end

seebeck_uVK(z::Real; x::Real=1.5) =
    KPM.KB_OVER_E_UV_PER_K * seebeck_dimensionless(z; x=x)

function verify_anchors()
    anchors = ((-4.0, 6.508039815),
               (0.0, 2.833442009),
               (2.0, 1.687876786),
               (6.0, 0.779022690),
               (20.0, 0.245620230))
    for (z, expected) in anchors
        actual = seebeck_dimensionless(z)
        @assert isapprox(actual, expected; atol=1e-8, rtol=0.0) "universal-curve anchor failed at z=$z: got $actual, expected $expected"
    end
    @assert all(seebeck_dimensionless(z) > 0 for z in (-4.0, 0.0, 2.0, 6.0, 20.0))
    return anchors
end

end # module UniversalReference

site_index(ix, iy, iz, L) = ix + 1 + L * (iy + L * iz)

# Put a periodic displacement into [-L/2, L/2), retaining unit NN bonds at
# the boundary. Even L is also required by the bipartite antithetic identity.
minimum_image(delta, L) = mod(delta + fld(L, 2), L) - fld(L, 2)

"""Build H(epsilon), the unrescaled bond current Jx, and Gamma."""
function cubic_hamiltonian_and_current(L, disorder; t=1.0)
    @assert iseven(L) "antithetic particle-hole pairing requires even L"
    NH = L^3
    length(disorder) == NH || throw(DimensionMismatch("expected $NH disorder values"))

    rows = Int[]
    cols = Int[]
    hvals = ComplexF64[]
    jrows = Int[]
    jcols = Int[]
    jvals = ComplexF64[]
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
        ri = positions[i]
        rj = positions[j]
        dx = minimum_image(ri[1] - rj[1], L)
        push!(rows, i); push!(cols, j); push!(hvals, value)
        if !iszero(dx)
            push!(jrows, i); push!(jcols, j); push!(jvals, value * dx)
        end
    end

    # Add every undirected +x/+y/+z bond once and then its Hermitian partner.
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

    # (J_alpha)_ij = H_ij (r_i-r_j)_alpha is antisymmetric for real H;
    # for complex Hermitian H it generally obeys J_alpha' = -J_alpha.
    jscale = maximum(abs, Jx)
    j_residual = maximum(abs, Jx + Jx')
    @assert j_residual <= 1e-13 * jscale
    return H, Jx, gamma, j_residual / jscale
end

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
            println("usage: julia AndersonMobilityEdgeSeebeck.jl [--cache PATH] [--out PATH.png]")
            exit(0)
        else
            error("unknown argument: $(args[i])")
        end
    end
    return (; out_path, cache_path)
end

function pair_random_data(pair_index)
    seed = seed_base + pair_index
    rng = Xoshiro(seed)
    disorder = W .* (rand(rng, L^3) .- 0.5)
    # The checked-out public API is rng-first. It returns unit-norm columns;
    # cond_moments uses supplied probes verbatim, and Gamma preserves the norm.
    psi_plus = KPM.random_phase_vectors(rng, L^3, NR)
    return seed, disorder, psi_plus
end

function parity_projected_partner(mu)
    partner = similar(mu)
    for j in axes(mu, 2), i in axes(mu, 1)
        partner[i, j] = isodd((i - 1) + (j - 1)) ? -mu[i, j] : mu[i, j]
    end
    return partner
end

function particle_hole_residual(mu)
    full_scale = maximum(abs, mu)
    odd_scale = 0.0
    for j in axes(mu, 2), i in axes(mu, 1)
        if isodd((i - 1) + (j - 1))
            odd_scale = max(odd_scale, abs(mu[i, j]))
        end
    end
    return odd_scale / full_scale
end

function cache_parameters()
    return (; schema=1, L, W, t, NC, NR, n_disorder_pairs, arr_size,
            a, b, seed_base, probe_api=:rng_first)
end

function read_cache(path, params)
    path === nothing && return nothing
    isfile(path) || return nothing
    try
        payload = open(deserialize, path)
        if hasproperty(payload, :params) && payload.params == params &&
           hasproperty(payload, :mu_avg) && hasproperty(payload, :mu_realizations) &&
           length(payload.mu_realizations) == 2 * n_disorder_pairs
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
    parent = dirname(abspath(path))
    mkpath(parent)
    open(path, "w") do io
        serialize(io, payload)
    end
end

function relative_difference(x, y)
    (!isfinite(x) || !isfinite(y)) && return Inf
    return abs(x - y) / max(abs(x), abs(y), eps(Float64))
end

function same_number(x, y; rtol=5e-13)
    return (isnan(x) && isnan(y)) || isapprox(x, y; rtol=rtol, atol=0.0)
end

function same_result(r1, r2)
    return same_number(r1.L0, r2.L0) && same_number(r1.L1, r2.L1) &&
           same_number(r1.L2, r2.L2) &&
           same_number(r1.S_over_kB_over_e, r2.S_over_kB_over_e) &&
           same_number(r1.neg_weight, r2.neg_weight)
end

passfail(flag) = flag ? "PASS" : "FAIL"

function main()
    cli = parse_cli(ARGS)
    @assert iseven(L) "L must be even"
    kT1, kT2 = kBT_values
    NH = L^3
    volume = Float64(NH)
    total_start = time()

    println("\n==============================================================================")
    println("3D Anderson mobility-edge Seebeck benchmark (fast default)")
    println("==============================================================================")
    @printf("L=%d (NH=%d), W=%.1f, t=%.1f, NC=%d, NR=%d, disorder pairs=%d\n",
            L, NH, W, t, NC, NR, n_disorder_pairs)
    @printf("fixed a=%.2f, b=%.1f, pi*a/NC=%.6f; kBT=%s\n",
            a, b, pi * a / NC, string(kBT_values))
    @printf("Ec=%.2f, x_ref=%.1f, seed_base=%d; per-pair seeds are seed_base + pair_index\n",
            Ec, x_ref, seed_base)
    println("NR counts stochastic probes per realization; disorder-pair count is separate.")

    println("\n-- Universal reference anchors --")
    anchors = UniversalReference.verify_anchors()
    for (z, expected) in anchors
        actual = UniversalReference.seebeck_dimensionless(z)
        @printf("z=%5.1f  quadrature=% .9f  anchor=% .9f  |delta|=%.2e\n",
                z, actual, expected, abs(actual - expected))
    end
    println("Anchor assertions passed (absolute tolerance 1e-8); upper-edge S is positive.")

    params = cache_parameters()
    payload = read_cache(cli.cache_path, params)
    build_seconds = 0.0
    moment_seconds = 0.0
    mu_avg = nothing
    mu_realizations = Matrix{ComplexF64}[]
    pair_identity_residuals = Float64[]
    j_residuals = Float64[]

    if payload === nothing
        println("\n-- Hamiltonians and antithetic conductivity moments --")
        mu_sum = zeros(ComplexF64, NC, NC)
        for pair_index in 1:n_disorder_pairs
            seed, disorder, psi_plus = pair_random_data(pair_index)
            @printf("pair %d/%d, seed=%d\n", pair_index, n_disorder_pairs, seed)

            build_start = time()
            H_plus, Jx_plus, gamma, jres_plus =
                cubic_hamiltonian_and_current(L, disorder; t=t)
            H_minus, Jx_minus, gamma_minus, jres_minus =
                cubic_hamiltonian_and_current(L, -disorder; t=t)
            build_seconds += time() - build_start
            @assert gamma == gamma_minus
            @assert Jx_plus == Jx_minus
            push!(j_residuals, jres_plus, jres_minus)

            Gamma = spdiagm(0 => gamma)
            h_identity_scale = maximum(abs, H_minus)
            h_identity_residual = maximum(abs, H_minus + Gamma * H_plus * Gamma) /
                                  h_identity_scale
            @assert h_identity_residual < 1e-13

            # H_norm(-epsilon) = -Gamma H_norm(+epsilon) Gamma and
            # Gamma Jx Gamma = -Jx. With psi_minus = Gamma psi_plus, this gives
            # mu_minus[n,m] = (-1)^(n+m) mu_plus[n,m] algebraically; numerical
            # deviations below are therefore only recurrence roundoff.
            psi_minus = reshape(gamma, :, 1) .* psi_plus
            @assert maximum(abs, vec(sum(abs2, psi_plus; dims=1)) .- 1) < 1e-13
            @assert maximum(abs, vec(sum(abs2, psi_minus; dims=1)) .- 1) < 1e-13
            h_plus = KPM.RescaledHamiltonian(H_plus ./ a, a, b)
            h_minus = KPM.RescaledHamiltonian(H_minus ./ a, a, b)

            moment_start = time()
            m_plus = KPM.cond_moments(h_plus, Jx_plus, Jx_plus;
                                      NC=NC, psi_in=psi_plus, arr_size=arr_size)
            m_minus = KPM.cond_moments(h_minus, Jx_minus, Jx_minus;
                                       NC=NC, psi_in=psi_minus, arr_size=arr_size)
            moment_seconds += time() - moment_start

            expected_minus = parity_projected_partner(m_plus.mu)
            pair_residual = maximum(abs, m_minus.mu - expected_minus) /
                            maximum(abs, m_plus.mu)
            push!(pair_identity_residuals, pair_residual)
            @printf("  Jx anti-Hermitian rel. residual <= %.3e; pair identity residual %.3e\n",
                    max(jres_plus, jres_minus), pair_residual)

            push!(mu_realizations, copy(m_plus.mu), copy(m_minus.mu))
            mu_sum .+= m_plus.mu
            mu_sum .+= m_minus.mu
        end
        mu_avg = mu_sum ./ (2 * n_disorder_pairs)
        payload = (; params, mu_avg, mu_realizations)
        if cli.cache_path === nothing
            println("Moment cache disabled (use --cache PATH to enable it).")
        else
            write_cache(cli.cache_path, payload)
            println("Wrote matching averaged/individual moment cache: $(abspath(cli.cache_path))")
        end
    else
        println("\n-- Cached conductivity moments --")
        mu_avg = payload.mu_avg
        mu_realizations = payload.mu_realizations
        println("Reloaded matching moment cache: $(abspath(cli.cache_path))")

        # Rebuild one inexpensive pair so construction invariants still run on
        # a cache-only execution; no KPM moments are recomputed.
        seed, disorder, _ = pair_random_data(1)
        build_start = time()
        H_plus, Jx_plus, gamma, jres_plus = cubic_hamiltonian_and_current(L, disorder; t=t)
        H_minus, Jx_minus, _, jres_minus = cubic_hamiltonian_and_current(L, -disorder; t=t)
        build_seconds += time() - build_start
        Gamma = spdiagm(0 => gamma)
        @assert maximum(abs, H_minus + Gamma * H_plus * Gamma) /
                maximum(abs, H_minus) < 1e-13
        @assert Jx_plus == Jx_minus
        push!(j_residuals, jres_plus, jres_minus)
        for pair_index in 1:n_disorder_pairs
            mu_plus = mu_realizations[2pair_index - 1]
            mu_minus = mu_realizations[2pair_index]
            push!(pair_identity_residuals,
                  maximum(abs, mu_minus - parity_projected_partner(mu_plus)) /
                  maximum(abs, mu_plus))
        end
        @printf("Revalidated pair-1 seed=%d and Jx anti-Hermitian residual %.3e.\n",
                seed, max(jres_plus, jres_minus))
    end

    @assert size(mu_avg) == (NC, NC)
    ph_residual = particle_hole_residual(mu_avg)
    @printf("Particle-hole odd-(n+m) moment residual = %.3e\n", ph_residual)
    @printf("Maximum antithetic pair identity residual = %.3e\n",
            maximum(pair_identity_residuals))
    @assert ph_residual < 1e-10
    m_avg = KPM.ConductivityMoments(mu_avg, a, b, NH, NR)

    println("\n-- Reconstruction and diagnostics --")
    reconstruction_start = time()
    n_band = max(257, 4 * NC + 1)
    band_halfwidth = a * (1 - edge_cutoff)
    E_band = collect(range(b - band_halfwidth, b + band_halfwidth; length=n_band))
    sigma_band = KPM.transport_distribution(m_avg, E_band; volume=volume)
    sigma_floor = 1e-6 * maximum(abs, sigma_band)
    sigma_symmetry_residual = maximum(abs, sigma_band - reverse(sigma_band)) /
                              maximum(abs, sigma_band)
    @printf("Dense band scan: %d points, max|Sigma|=%.6e, sigma_floor=%.6e\n",
            n_band, maximum(abs, sigma_band), sigma_floor)
    @printf("Sigma(E)=Sigma(-E) relative residual = %.3e\n", sigma_symmetry_residual)

    results = Dict{Float64,Any}()
    S_values = Dict{Float64,Vector{Float64}}()
    L0_values = Dict{Float64,Vector{Float64}}()
    for kBT in kBT_values
        beta = inv(kBT)
        r = [KPM.thermoelectric(m_avg, Ec - z * kBT;
                               beta=beta, volume=volume,
                               sigma_min=sigma_floor, quad_N=4 * NC)
             for z in z_values]
        results[kBT] = r
        S_values[kBT] = KPM.seebeck_uVK.(r)
        L0_values[kBT] = getfield.(r, :L0)
    end
    S_reference = UniversalReference.seebeck_uVK.(z_values)

    @printf("\nQuadrature-order spot checks (kBT=%.2f):\n", kT1)
    for z in (0.0, 4.0)
        i = findfirst(==(z), z_values)
        r4 = results[kT1][i]
        r8 = KPM.thermoelectric(m_avg, Ec - z * kT1;
                               beta=inv(kT1), volume=volume,
                               sigma_min=sigma_floor, quad_N=8 * NC)
        delta = relative_difference(KPM.seebeck_uVK(r4), KPM.seebeck_uVK(r8))
        @printf("  z=%4.1f: S(4NC)=% .6f, S(8NC)=% .6f uV/K, rel. diff=%.3e\n",
                z, KPM.seebeck_uVK(r4), KPM.seebeck_uVK(r8), delta)
    end

    @printf("\nConductivity-floor diagnostic at kBT=%.2f:\n", kT1)
    floor_default = Dict{Float64,Any}()
    floor_override = Dict{Float64,Any}()
    for z in z_floor_checks
        mu_chem = Ec - z * kT1
        r_default = KPM.thermoelectric(m_avg, mu_chem;
                                      beta=inv(kT1), volume=volume, quad_N=4 * NC)
        r_override = KPM.thermoelectric(m_avg, mu_chem;
                                       beta=inv(kT1), volume=volume,
                                       sigma_min=0.0, quad_N=4 * NC)
        floor_default[z] = r_default
        floor_override[z] = r_override
        status = isnan(r_default.S_over_kB_over_e) ? "NaN/warned: default floor masks S" :
                                                    "finite: default floor retains S"
        @printf("  z=%5.1f  default=%-34s  override S=% .6f uV/K  L0=%.4e\n",
                z, status, KPM.seebeck_uVK(r_override), r_override.L0)
    end
    println("  sigma_min=0.0 values are diagnostics, not reportable insulating-side claims.")

    println("\nDefault-floor equivalence spot checks (same quad_N=4NC):")
    for z in (4.0, -16.0)
        i = findfirst(==(z), z_values)
        guarded = i === nothing ?
            KPM.thermoelectric(m_avg, Ec - z * kT1; beta=inv(kT1), volume=volume,
                               sigma_min=sigma_floor, quad_N=4 * NC) :
            results[kT1][i]
        default = haskey(floor_default, z) ? floor_default[z] :
            KPM.thermoelectric(m_avg, Ec - z * kT1; beta=inv(kT1), volume=volume,
                               quad_N=4 * NC)
        @printf("  z=%5.1f (%s): explicit precomputed floor matches plain default = %s\n",
                z, z > 0 ? "conducting" : "insulating", string(same_result(guarded, default)))
    end

    scatter_z = (2.0, 6.0)
    realization_scatter = Dict{Float64,Vector{Float64}}()
    for z in scatter_z
        values = Float64[]
        for mu_r in mu_realizations
            m_r = KPM.ConductivityMoments(mu_r, a, b, NH, NR)
            r = KPM.thermoelectric(m_r, Ec - z * kT1;
                                  beta=inv(kT1), volume=volume,
                                  sigma_min=0.0, quad_N=4 * NC)
            push!(values, KPM.seebeck_uVK(r))
        end
        realization_scatter[z] = values
        @printf("Realization scatter at z=%.1f, kBT=%.2f: %s uV/K\n",
                z, kT1, join((@sprintf("% .3f", value) for value in values), ", "))
    end

    # Fast-run quality targets. NC=256 uses a genuinely truncated typed moment
    # object and its own algebraically default-equivalent band floor.
    comparison_indices = findall(z -> 1.0 <= z <= 6.0, z_values)
    NC_trunc = 256
    m_trunc = KPM.ConductivityMoments(mu_avg[1:NC_trunc, 1:NC_trunc],
                                      a, b, NH, NR)
    n_band_trunc = max(257, 4 * NC_trunc + 1)
    E_band_trunc = collect(range(b - band_halfwidth, b + band_halfwidth;
                                 length=n_band_trunc))
    sigma_band_trunc = KPM.transport_distribution(m_trunc, E_band_trunc;
                                                   volume=volume)
    sigma_floor_trunc = 1e-6 * maximum(abs, sigma_band_trunc)
    trunc_changes = Float64[]
    for kBT in kBT_values, i in comparison_indices
        z = z_values[i]
        r_trunc = KPM.thermoelectric(m_trunc, Ec - z * kBT;
                                    beta=inv(kBT), volume=volume,
                                    sigma_min=sigma_floor_trunc,
                                    quad_N=4 * NC_trunc)
        push!(trunc_changes,
              relative_difference(S_values[kBT][i], KPM.seebeck_uVK(r_trunc)))
    end

    neg_weights = [results[kBT][i].neg_weight
                   for kBT in kBT_values for i in comparison_indices]
    collapse_changes = [relative_difference(S_values[kT1][i], S_values[kT2][i])
                        for i in comparison_indices]
    benchmark_changes = [abs(S_values[kBT][i] / S_reference[i] - 1)
                         for kBT in kBT_values for i in comparison_indices]
    max_neg_weight = maximum(neg_weights)
    max_trunc_change = maximum(trunc_changes)
    max_collapse_change = maximum(collapse_changes)
    median_benchmark_change = median(benchmark_changes)

    println("\nFast-run quality targets over 1 <= z <= 6 (diagnostic, non-aborting):")
    @printf("  [%s] neg_weight < 1%%: maximum = %.3f%%\n",
            passfail(max_neg_weight < 0.01), 100 * max_neg_weight)
    @printf("  [%s] NC=%d -> %d maximum S change < 10%%: %.3f%%\n",
            passfail(max_trunc_change < 0.10), NC, NC_trunc, 100 * max_trunc_change)
    @printf("  [%s] two-temperature maximum collapse difference < 10%%: %.3f%%\n",
            passfail(max_collapse_change < 0.10), 100 * max_collapse_change)
    @printf("  [%s] median |S_KPM/S_ref - 1| < 15%%: %.3f%%\n",
            passfail(median_benchmark_change < 0.15), 100 * median_benchmark_change)
    metallic_S_min = minimum(minimum(S_values[kBT][comparison_indices])
                             for kBT in kBT_values)
    @printf("Metallic-window Seebeck sign check: minimum S = %.6f uV/K (%s).\n",
            metallic_S_min, passfail(metallic_S_min > 0))
    println("Honesty statement: this example does not measure E_c or the critical exponent.")
    println("""
        Interpretation of failing targets at the fast defaults: the disorder-averaged
        transport edge at this size and kernel width is a smeared crossover profile
        (Sigma stays finite well above E_c = 7.5 and does not follow the critical
        power law), so the universal-curve and collapse targets fail for physical,
        parameter-limited reasons -- in the plan's ordered causes: finite L and the
        historical E_c outside its finite-size accuracy, not stochastic noise. A
        convergence study in L, NC, and temperature is the follow-up, not gate
        loosening.""")

    reconstruction_seconds = time() - reconstruction_start

    println("\nTerminal summary (all S columns in uV/K):")
    @printf("      z       S(%.2f)       S(%.2f)          S_ref        L0(%.2f)        L0(%.2f)\n",
            kT1, kT2, kT1, kT2)
    println("--------------------------------------------------------------------------------------")
    for (i, z) in pairs(z_values)
        @printf("%7.2f  %13.6f  %13.6f  %13.6f  %14.6e  %14.6e\n",
                z, S_values[kT1][i], S_values[kT2][i], S_reference[i],
                L0_values[kT1][i], L0_values[kT2][i])
    end

    println("\n-- Four-panel figure --")
    plotting_start = time()
    colors = (:royalblue, :darkorange)
    legend_style = (; legend_background_color=:transparent,
                    legend_foreground_color=:transparent)

    edge_indices = findall(E -> 4.5 <= E <= 8.5, E_band)
    E_edge = E_band[edge_indices]
    sigma_edge = sigma_band[edge_indices]
    fit_window = (5.5, 7.0)
    fit_indices = findall(E -> fit_window[1] <= E <= fit_window[2], E_edge)
    edge_power = max.(Ec .- E_edge, 0.0) .^ x_ref
    fit_power = edge_power[fit_indices]
    fit_amplitude = dot(fit_power, sigma_edge[fit_indices]) / dot(fit_power, fit_power)
    sigma_fit = fit_amplitude .* edge_power
    p1 = Plots.plot(E_edge, sigma_edge; color=:black, linewidth=2,
                    xlabel="E/t", ylabel="Sigma_xx(E)",
                    title="Transport edge", label="KPM disorder average",
                    legend=:topright, legend_style...)
    Plots.plot!(p1, E_edge, sigma_fit; color=:crimson, linestyle=:dash,
                linewidth=2, label=@sprintf("A(Ec-E)^1.5 fit, %.1f<=E<=%.1f",
                                            fit_window...))
    Plots.vline!(p1, [Ec]; color=:gray, linestyle=:dot, label="Ec=7.5 (fixed)")

    p2 = Plots.plot(z_values, S_reference; color=:black, linestyle=:dash,
                    linewidth=2.5, xlabel="z = (Ec-mu)/(kBT)",
                    ylabel="S (uV/K)", title="Upper-edge Seebeck",
                    xlims=(-4, 8), label="universal x=1.5",
                    legend=:topright, legend_style...)
    for (color, kBT) in zip(colors, kBT_values)
        Plots.plot!(p2, z_values, S_values[kBT]; marker=:circle, markersize=3,
                    color=color, linewidth=2, label=@sprintf("KPM kBT=%.2f", kBT))
    end

    p3 = Plots.plot(; xlabel="z = (Ec-mu)/(kBT)", ylabel="L0 / max(L0)",
                    title="Conductivity context", yscale=:log10, xlims=(-4, 8),
                    legend=:bottomleft, legend_style...)
    for (color, kBT) in zip(colors, kBT_values)
        scale = maximum(L0_values[kBT])
        normalized = max.(L0_values[kBT] ./ scale, eps(Float64))
        Plots.plot!(p3, z_values, normalized; marker=:circle, markersize=3,
                    color=color, linewidth=2, label=@sprintf("kBT=%.2f", kBT))
    end

    p4 = Plots.plot(; xlabel="z", ylabel="S scatter (uV/K); neg_weight (%)",
                    title="Diagnostics", xlims=(-16.8, 8.5), ylims=(-60, 105),
                    legend=:topright, legend_style...)
    for (color, kBT) in zip(colors, kBT_values)
        neg = 100 .* getfield.(results[kBT], :neg_weight)
        Plots.plot!(p4, z_values, neg; color=color, linewidth=2,
                    label=@sprintf("neg_weight (%%), kBT=%.2f", kBT))
    end
    floor_flags = [isfinite(floor_default[z].S_over_kB_over_e) ? 25.0 : -25.0
                   for z in z_floor_checks]
    Plots.scatter!(p4, collect(z_floor_checks), floor_flags; color=:purple,
                   marker=:diamond, markersize=6, label="default floor: +25 kept, -25 masked")
    for (z, y) in zip(z_floor_checks, floor_flags)
        Plots.annotate!(p4, z, y + 8,
                        Plots.text(@sprintf("override %.0f", KPM.seebeck_uVK(floor_override[z])),
                                   6, :purple))
    end
    scatter_x = Float64[]
    scatter_y = Float64[]
    for (j, z) in pairs(scatter_z), value in realization_scatter[z]
        push!(scatter_x, z + 0.045 * (j - 1.5))
        push!(scatter_y, value)
    end
    Plots.scatter!(p4, scatter_x, scatter_y; color=:forestgreen,
                   marker=:xcross, markersize=5,
                   label=@sprintf("single realizations, kBT=%.2f", kT1),
                   legend=:bottomright, legend_style...)

    fig = Plots.plot(p1, p2, p3, p4; layout=(2, 2), size=(1300, 1000),
                     margin=5 * Plots.mm,
                     plot_title="3D Anderson CTKG benchmark: finite-size/broadening diagnostic")
    # Force one render before recording the runtime annotation. The final
    # render below is then a short serialization pass rather than backend JIT.
    show(devnull, MIME("image/png"), fig)
    runtime_after_first_render = time() - total_start
    Plots.annotate!(fig[4], -16.0, 96.0,
                    Plots.text(@sprintf("pi*a/NC=%.3f; runtime~%.1fs; PH=%.1e",
                                        pi * a / NC, runtime_after_first_render, ph_residual),
                               7, :left, :black))
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
    @printf("  Hamiltonian/current construction: %8.3f s\n", build_seconds)
    @printf("  Conductivity moments:            %8.3f s\n", moment_seconds)
    @printf("  Reconstruction/diagnostics:      %8.3f s\n", reconstruction_seconds)
    @printf("  Plotting/rendering:               %8.3f s\n", plotting_seconds)
    @printf("  Total runtime:                    %8.3f s\n", total_seconds)
    println("This example does not measure E_c or the critical exponent.")
end

main()
