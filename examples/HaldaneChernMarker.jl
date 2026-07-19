# Local Chern marker of the Haldane model on an open flake.
#
# Demonstrates the Fermi-projector/Chern-marker workflow: build an
# open-boundary Haldane flake with explicit site coordinates (geometry is
# user data — KPM.jl never infers positions), compute the site-resolved
# Bianco–Resta marker, average complete bulk cells to read the Chern number,
# and estimate a regional average stochastically with error bars.
#
# Run:  julia --project=. examples/HaldaneChernMarker.jl

using KPM
using LinearAlgebra
using SparseArrays
using Random
using Statistics

"""
    haldane_flake(Lx, Ly; t, t2, ϕ, m) -> (H, x, y, cell_area)

Haldane model on an open Lx×Ly honeycomb flake: NN hopping -t, complex NNN
hopping t2·e^{±iϕ} (chirality +iϕ on sublattice A, -iϕ on B), staggered
mass ±m. Returns the sparse Hamiltonian, the site-coordinate vectors, and
the unit-cell area. Sites are indexed (A, B) per cell, cells column-major
in (x, y).
"""
function haldane_flake(Lx, Ly; t = 1.0, t2 = 0.2, ϕ = π/2, m = 0.0)
    a1 = [sqrt(3), 0.0]
    a2 = [sqrt(3) / 2, 1.5]
    δAB = [sqrt(3) / 2, 0.5]

    inbounds(cx, cy) = 1 <= cx <= Lx && 1 <= cy <= Ly
    cell(cx, cy) = cy + Ly * (cx - 1)
    siteA(cx, cy) = 2 * cell(cx, cy) - 1
    siteB(cx, cy) = 2 * cell(cx, cy)

    N = 2 * Lx * Ly
    x = zeros(N)
    y = zeros(N)
    for cx = 1:Lx, cy = 1:Ly
        rA = (cx - 1) .* a1 .+ (cy - 1) .* a2
        x[siteA(cx, cy)], y[siteA(cx, cy)] = rA
        x[siteB(cx, cy)], y[siteB(cx, cy)] = rA .+ δAB
    end

    I_, J_, V_ = Int[], Int[], ComplexF64[]
    function addhop!(i, j, amp)
        push!(I_, i), push!(J_, j), push!(V_, amp)
        push!(I_, j), push!(J_, i), push!(V_, conj(amp))
    end
    # NNN displacements a1, a2-a1, -a2 in cell offsets
    nnn = ((1, 0), (-1, 1), (0, -1))
    for cx = 1:Lx, cy = 1:Ly
        addhop!(siteB(cx, cy), siteA(cx, cy), -t + 0im)
        inbounds(cx + 1, cy) && addhop!(siteB(cx, cy), siteA(cx + 1, cy), -t + 0im)
        inbounds(cx, cy + 1) && addhop!(siteB(cx, cy), siteA(cx, cy + 1), -t + 0im)
        for (ox, oy) in nnn
            if inbounds(cx + ox, cy + oy)
                addhop!(siteA(cx + ox, cy + oy), siteA(cx, cy), t2 * cis(ϕ))
                addhop!(siteB(cx + ox, cy + oy), siteB(cx, cy), t2 * cis(-ϕ))
            end
        end
        push!(I_, siteA(cx, cy)), push!(J_, siteA(cx, cy)), push!(V_, m + 0im)
        push!(I_, siteB(cx, cy)), push!(J_, siteB(cx, cy)), push!(V_, -m + 0im)
    end
    H = sparse(I_, J_, V_, N, N)
    cell_area = abs(a1[1] * a2[2] - a1[2] * a2[1])
    return H, x, y, cell_area
end

# orbital indices of the cells in xs × ys
function cell_orbitals(Ly, xs, ys)
    sites = Int[]
    for cx in xs, cy in ys
        c = cy + Ly * (cx - 1)
        push!(sites, 2c - 1, 2c)
    end
    return sites
end

Lx = Ly = 16
NC = 1024                       # resolution πa/NC must sit inside the gap
bulk = cell_orbitals(Ly, 6:11, 6:11)          # central 6×6 cells
edge = cell_orbitals(Ly, 1:1, 1:Ly)           # one boundary column

for (label, m) in (("topological (m = 0)", 0.0), ("trivial (m = 1.6)", 1.6))
    H, x, y, Ac = haldane_flake(Lx, Ly; m = m)
    h = KPM.rescale(H; center = true)
    println("== $label: gap-resolving NC = $NC, ΔE ≈ ", round(π * h.a / NC; digits = 3))

    # site-resolved marker over the whole flake
    mk = KPM.chern_marker(h, x, y; Ef = 0.0, sites = collect(1:size(H, 1)), NC = NC)
    C_bulk = KPM.chern_marker_average(mk[bulk]; area = length(bulk) ÷ 2 * Ac)
    C_edge = KPM.chern_marker_average(mk[edge]; area = Ly * Ac)
    println("   bulk 6×6-cell average  : ", round(C_bulk; digits = 3))
    println("   edge-column average    : ", round(C_edge; digits = 3), "  (not quantized)")
    println(
        "   whole-sample sum       : ",
        round(sum(mk); digits = 3),
        "  (",
        round(sum(mk) / sum(abs, mk); sigdigits = 2),
        " of the gross weight Σ|m| — no net topology on an open flake)",
    )

    # stochastic regional estimate with error bars (cost independent of |R|)
    est = KPM.chern_marker_region(
        h,
        x,
        y;
        Ef = 0.0,
        region = bulk,
        rng = Xoshiro(1),
        NR = 64,
        NC = NC,
    )
    C_est = KPM.chern_marker_average([mean(est)]; area = length(bulk) ÷ 2 * Ac)
    C_err = std(est) / sqrt(length(est)) / (length(bulk) ÷ 2 * Ac)
    println(
        "   stochastic bulk average: ",
        round(C_est; digits = 3),
        " ± ",
        round(C_err; digits = 3),
    )
end
