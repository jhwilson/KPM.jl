#!/usr/bin/env julia

# Self-consistent spinless p-wave (Kitaev) chain via a user-defined bond
# pairing channel, with an ED cross-check and the open-chain Majorana signature.
# Run with: julia --project=. examples/KitaevChain.jl

using KPM
using LinearAlgebra
using Printf
using SparseArrays

"""
Spinless chain with nearest-neighbor hopping; returns h and the bond list.
"""
function kitaev_chain(N::Int; t::Real = 1.0, periodic::Bool = true)
    h = spzeros(Float64, N, N)
    bonds = [(i, i + 1) for i = 1:(N-1)]
    periodic && push!(bonds, (N, 1))
    for (i, j) in bonds
        h[i, j] = -Float64(t)
        h[j, i] = -Float64(t)
    end
    return h, bonds
end

"""
Dense BdG matrix for the ED cross-checks (conjugate hole convention).
"""
function dense_bdg(h, mu, D)
    N = size(h, 1)
    xi = Matrix{ComplexF64}(h) - mu * I
    return [
        xi Matrix{ComplexF64}(D);
        adjoint(Matrix{ComplexF64}(D)) -conj(xi)
    ]
end

N = 64
t = 1.0
mu = 0.2
V = 3.0          # nearest-neighbor attraction; the two-site analog gaps for V >= 2t
beta = 20.0

h, bonds = kitaev_chain(N; t = t)

# The pairing structure is user data: a single odd (p-wave) bond channel.
channel = KPM.PairingChannel(bonds, 1.0, V, :odd)
op = KPM.BdGOperator(
    h;
    mu = mu,
    U = 0.0,
    D = KPM.pairing_matrix(N, [channel]; amplitude = 0.3),
    hole_convention = :conjugate,
)

scf = KPM.bdg_solve!(
    op,
    [channel];
    beta = beta,
    NC = 512,
    mix = 0.3,
    tol_delta = 1e-8,
    update_density = false,
    g_rho = 1,
    mixing = :anderson,
)
amps = [op.D[i, j] for (i, j) in bonds]
proj = KPM.channel_amplitude(channel, amps)
@printf(
    "KPM channel SCF: converged=%s after %d iterations, |Delta_bond| = %.6f\n",
    scf.converged,
    scf.iterations,
    abs(proj)
)

# ED cross-check of the same per-bond fixed point.
Delta_ed = 0.3
for _ = 1:2000
    H = dense_bdg(h, mu, KPM.pairing_matrix(N, [channel]; amplitude = Delta_ed))
    F = eigen(Hermitian(H))
    occ = 1 ./ (exp.(beta .* F.values) .+ 1)
    i, j = bonds[1]
    Fij = sum(F.vectors[i, s] * conj(F.vectors[j+N, s]) * occ[s] for s = 1:2N)
    Fji = sum(F.vectors[j, s] * conj(F.vectors[i+N, s]) * occ[s] for s = 1:2N)
    Delta_new = -V * (Fij - Fji) / 2
    global Delta_ed = 0.5 * Delta_ed + 0.5 * real(Delta_new)
end
@printf(
    "ED fixed point:  |Delta_bond| = %.6f (relative mismatch %.2e)\n",
    abs(Delta_ed),
    abs(abs(proj) - abs(Delta_ed)) / abs(Delta_ed)
)

# Open-chain Majorana signature at the converged gap: two near-zero modes for
# |mu| < 2t, none for |mu| > 2t.
h_open, bonds_open = kitaev_chain(N; t = t, periodic = false)
channel_open = KPM.PairingChannel(bonds_open, 1.0, V, :odd)
for (mu_test, label) in ((0.2, "topological, |mu| < 2t"), (3.0, "trivial, |mu| > 2t"))
    D_open = KPM.pairing_matrix(N, [channel_open]; amplitude = abs(proj))
    E = abs.(eigvals(Hermitian(dense_bdg(h_open, mu_test, D_open))))
    sort!(E)
    @printf(
        "open chain, mu=%.1f (%s): |E1|=%.2e, |E2|=%.2e, |E3|=%.2e\n",
        mu_test,
        label,
        E[1],
        E[2],
        E[3]
    )
end
