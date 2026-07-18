"""
SpectralFunctionKspace.jl

Momentum-resolved spectral function A(k, E) and full complex Green function
G(k, E) for a weakly disordered periodic chain, in the style of Pixley et
al., PRB 95, 235101 (2017): the momentum content lives entirely in
caller-built Fourier probes u_k[j] = e^{ikj}/sqrt(N) — KPM.jl infers no
positions or reciprocal vectors (models are user data). The disorder-averaged
moments are reconstructed once with the direct CPGF route (exact causal
resolvent at E + i*eta), so Re G is the Kramers-Kronig partner of A(k, E) by
construction — analytically in Chebyshev space, with no numerical Hilbert
transform.

From the same G(k, E) the script extracts the quasiparticle dispersion E(k)
(peak of A), the residue Z(k) = [d Re G^{-1}/dE at E(k)]^{-1}, and the
linewidth gamma(k) = Z(k)/|Im G(k, E(k))|, and compares E(k) with the clean
dispersion -2t*cos(k). For weak disorder gamma(k) is set by the Born rate
~ (W^2/12) * rho(E_k) per unit hopping, giving sharp peaks near the band
center that broaden toward the band edges; eta only regularizes the tail.

Plots.jl is an example-only dependency. If missing, activate the intended
environment and run `import Pkg; Pkg.add("Plots")`.
"""

using KPM
using LinearAlgebra
using Printf
using Random
using SparseArrays
using Statistics

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

# Okabe-Ito
const C_BLUE = "#0072B2"
const C_VERM = "#D55E00"
const C_GREEN = "#009E73"
const C_ORANGE = "#E69F00"

const N = 1024              # chain sites
const t = 1.0               # hopping
const W = 0.6               # box-disorder width, V_j in [-W/2, W/2]
const NC = 4096             # Chebyshev order (doubling path: even)
const N_DIS = 12            # disorder realizations
const ETA = 0.02            # CPGF broadening, physical units
const SEED = 20260718

function parse_cli(args)
    out_path = nothing
    i = 1
    while i <= length(args)
        if args[i] == "--out"
            i == length(args) && error("--out requires a path")
            out_path = args[i + 1]
            i += 2
        else
            error("unknown argument $(args[i]); supported: --out PATH.png")
        end
    end
    return (; out_path)
end

"Periodic chain with per-sample box disorder; fixed (a, b) across samples so moments average."
function chain_hamiltonian(rng)
    H = spdiagm(1 => fill(-t, N - 1), -1 => fill(-t, N - 1))
    H[1, N] = -t
    H[N, 1] = -t
    return H + spdiagm(0 => W .* (rand(rng, N) .- 0.5))
end

function main()
    cli = parse_cli(ARGS)
    rng = Xoshiro(SEED)

    # momenta and Fourier probes: user data, built from explicit positions 1:N
    ks = 2 * pi .* round.(Int, range(8, N ÷ 2 - 8; length = 12)) ./ N
    probes = hcat([exp.(im .* k .* (1:N)) ./ sqrt(N) for k in ks]...)

    # one fixed rescaling for every sample: |spectrum| <= 2t + W/2
    a_fixed = (2 * t + W / 2) * 1.1

    moment_start = time()
    mu_avg = zeros(ComplexF64, NC, length(ks))
    for _ in 1:N_DIS
        h = KPM.rescale(chain_hamiltonian(rng); fixed_a = a_fixed)
        m = KPM.green_moments(h, probes; NC = NC)
        mu_avg .+= m.mu ./ N_DIS
    end
    m_avg = KPM.GreenMoments(mu_avg, a_fixed, 0.0, N)
    moment_seconds = time() - moment_start

    # one reconstruction: A and Re G come from the same moments
    E = collect(range(-2.4, 2.4; length = 1201))
    G = KPM.greens(m_avg, E; eta = ETA)
    A = -imag.(G) ./ pi

    # quasiparticle extraction from the pole form G ~ Z/(E - E_k + i*gamma)
    dE = E[2] - E[1]
    Ek = similar(ks)
    Zk = similar(ks)
    γk = similar(ks)
    for p in eachindex(ks)
        ipk = argmax(view(A, :, p))
        Ek[p] = E[ipk]
        invG = 1 ./ G[:, p]
        dReinvG = (real(invG[ipk + 1]) - real(invG[ipk - 1])) / (2 * dE)
        Zk[p] = 1 / dReinvG
        γk[p] = Zk[p] / abs(imag(G[ipk, p]))
    end

    println("Disorder-averaged quasiparticles (N=$N, W=$W, NC=$NC, eta=$ETA, $N_DIS samples):")
    println("      k/pi      E(k)   -2t cos k      Z(k)  gamma(k)")
    for p in eachindex(ks)
        @printf("  %8.4f  %8.4f  %10.4f  %8.4f  %8.4f\n",
                ks[p] / pi, Ek[p], -2 * t * cos(ks[p]), Zk[p], γk[p])
    end
    drift = maximum(abs.(Ek .+ 2 .* t .* cos.(ks)))
    println("max |E(k) - eps_k| = $(round(drift, sigdigits=3)) (weak-disorder shift + grid step)")

    # (a) A(k, E) waterfall: each momentum offset vertically, directly labeled
    plotting_start = time()
    offset = 0.85 * maximum(A[:, 2:end])
    p1 = Plots.plot(; xlabel = "E / t", ylabel = "A(k, E) + offset",
                    legend = false, grid = false, title = "spectral function")
    for p in eachindex(ks)
        Plots.plot!(p1, E, A[:, p] .+ (p - 1) * offset; color = C_BLUE, lw = 1.2)
        Plots.annotate!(p1, [(2.35, (p - 1) * offset + 0.12 * offset,
                              Plots.text(@sprintf("k=%.2fπ", ks[p] / pi), 7,
                                         :right, :black))])
    end

    # (b) causal pair at one k: A and Re G from the same moments
    psel = 6
    p2 = Plots.plot(E, A[:, psel]; color = C_BLUE, lw = 1.6, grid = false,
                    legend = false, xlabel = "E / t",
                    title = @sprintf("causal pair at k = %.2fπ", ks[psel] / pi))
    Plots.plot!(p2, E, real.(G[:, psel]); color = C_VERM, lw = 1.6)
    Plots.hline!(p2, [0.0]; color = :gray, lw = 0.4)
    ymax = maximum(A[:, psel])
    Plots.annotate!(p2, [(Ek[psel] - 0.75, 0.9 * ymax, Plots.text("A(k, E)", 9, :left, Plots.RGB(0 / 255, 114 / 255, 178 / 255))),
                         (Ek[psel] + 0.55, 0.55 * ymax, Plots.text("Re G(k, E)", 9, :left, Plots.RGB(213 / 255, 94 / 255, 0 / 255)))])

    # (c) dispersion and linewidth
    kdense = range(0, pi; length = 200)
    p3 = Plots.plot(collect(kdense) ./ pi, -2 .* t .* cos.(kdense); color = :gray,
                    lw = 1.0, grid = false, legend = false,
                    xlabel = "k / π", ylabel = "E / t", title = "dispersion")
    Plots.scatter!(p3, ks ./ pi, Ek; color = C_GREEN, markerstrokewidth = 0, ms = 4)
    Plots.annotate!(p3, [(0.32, 1.15, Plots.text("E(k) from peak of A", 9, :left, Plots.RGB(0 / 255, 158 / 255, 115 / 255))),
                         (0.55, -1.6, Plots.text("-2t cos k", 9, :left, :gray))])

    p4 = Plots.scatter(ks ./ pi, γk; color = C_ORANGE, markerstrokewidth = 0, ms = 4,
                       grid = false, legend = false, xlabel = "k / π",
                       ylabel = "γ(k) / t", title = "linewidth")
    Plots.hline!(p4, [ETA]; color = :gray, lw = 0.6, ls = :dash)
    Plots.annotate!(p4, [(0.62, ETA * 1.12, Plots.text("η floor", 8, :left, :gray))])

    fig = Plots.plot(p1, p2, p3, p4; layout = (2, 2), size = (1250, 950),
                     margin = 6 * Plots.mm,
                     plot_title = "Disordered chain: A(k, E) and Kramers–Kronig Re G from one set of moments")
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

    println("\nTiming summary:")
    @printf("  Moments (%d samples x %d momenta):  %8.3f s\n", N_DIS, length(ks), moment_seconds)
    @printf("  Plotting/rendering:                %8.3f s\n", plotting_seconds)
end

main()
