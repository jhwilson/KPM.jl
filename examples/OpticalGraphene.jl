using KPM
using LinearAlgebra
using Random
using Plots
using LaTeXStrings

include("GrapheneModel.jl")

L = 40
t = 2.3
Ham, Jx, Jy, Jxx, Jxy, Jyy = GrapheneLattice(L, L; t = t)
h = KPM.rescale(Ham; center = true)

NC = 128
NR = 4
NH = size(Ham, 1)
probes = KPM.random_phase_vectors(Xoshiro(42), NH, NR)
m2yy = KPM.cond_moments(h, Jy, Jy; NC = NC, psi_in = copy(probes))
m1yy = KPM.current_moments(h, Jyy, NC, NR; psi_in = copy(probes))

# The honeycomb primitive-cell area is 3sqrt(3)/2 in the bond-length units
# used by GrapheneModel.jl. Energies below are in eV because t is in eV.
area = L^2 * 3sqrt(3) / 2
Ef = 0.466
lambda = 38.8e-3
omegas = collect(range(0.1, 2.5; length = 40))
sigma_yy = KPM.optical_cond(
    m2yy,
    omegas;
    area = area,
    Ef = Ef,
    m1 = m1yy,
    lambda = lambda,
    quad_rtol = 1e-6,
)

plot(
    omegas,
    [real.(sigma_yy) imag.(sigma_yy)];
    label = [L"\mathrm{Re}\;\sigma^{yy}" L"\mathrm{Im}\;\sigma^{yy}"],
    ylabel = L"\sigma^{yy}\;(e^2/h)",
    xlabel = L"\hbar\omega\;(\mathrm{eV})",
    framestyle = :box,
    grid = false,
    legend = :topright,
)
