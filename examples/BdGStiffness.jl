#!/usr/bin/env julia

# Self-consistent reduced-Nambu BdG calculation and finite-q stiffness check.
# Run with: julia --project=. examples/BdGStiffness.jl

using KPM
using LinearAlgebra
using Printf
using SparseArrays

"""Periodic square lattice with nearest-neighbor hopping and minimum-image bonds."""
function square_lattice(Lx::Int, Ly::Int; t::Real=1.0)
    N = Lx * Ly
    site(ix, iy) = mod1(ix, Lx) + (mod1(iy, Ly) - 1) * Lx
    h = spzeros(Float64, N, N)
    pos = zeros(Float64, N, 2)

    for iy in 1:Ly, ix in 1:Lx
        i = site(ix, iy)
        pos[i, :] .= (ix - 1, iy - 1)
        for (jx, jy) in ((ix + 1, iy), (ix - 1, iy),
                         (ix, iy + 1), (ix, iy - 1))
            h[i, site(jx, jy)] = -Float64(t)
        end
    end

    function disp(i, j)
        dx = pos[i, 1] - pos[j, 1]
        dy = pos[i, 2] - pos[j, 2]
        dx > Lx / 2 && (dx -= Lx)
        dx < -Lx / 2 && (dx += Lx)
        dy > Ly / 2 && (dy -= Ly)
        dy < -Ly / 2 && (dy += Ly)
        return [dx, dy]
    end
    return h, pos, disp
end

"""Dense reduced-Nambu BdG matrix used only for the ED cross-checks."""
function dense_bdg(h, mu, U, n, Delta)
    xi = Matrix(h) - mu * I - Diagonal(U .* n ./ 2)
    D = Diagonal(Delta)
    return ComplexF64[xi D; Diagonal(conj.(Delta)) -xi]
end

"""Dense Lehmann response in the same convention as `KPM.two_energy_response`."""
function ed_two_energy_response(H, Jleft, Jright; beta, eta, omega=0.0, Ef=0.0,
                                volume=1.0, g_J=1.0)
    F = eigen(Hermitian(Matrix(H)))
    Jl = F.vectors' * Matrix(Jleft) * F.vectors
    Jr = F.vectors' * Matrix(Jright) * F.vectors
    fermi(E) = 1 / (exp(beta * (E - Ef)) + 1)
    occupations = fermi.(F.values)
    response = zero(ComplexF64)
    for p in eachindex(F.values), q in eachindex(F.values)
        response += Jl[p, q] * Jr[q, p] * (occupations[p] - occupations[q]) /
                    (omega + F.values[p] - F.values[q] + im * eta)
    end
    return ComplexF64(g_J * response / volume)
end

Lx = 6
Ly = 6
t = 1.0
mu = -0.5
U = 2.5
beta = 8.0
Delta_initial = 0.2 + 0.0im
n_initial = 0.5
mix = 0.3
NC_scf = 256
Np_scf = 2 * NC_scf
tol_delta = 1e-7
tol_n = 1e-7

h, pos, disp = square_lattice(Lx, Ly; t=t)
N = size(h, 1)
U_field = fill(U, N)
op = KPM.BdGOperator(h; mu=mu, U=U_field, n=fill(n_initial, N),
                     Delta=fill(Delta_initial, N))

println("Self-consistent BdG on a periodic square lattice")
println("  Lx=$(Lx), Ly=$(Ly), N=$(N), t=$(t)")
println("  mu=$(mu), U=$(U), beta=$(beta)")
println("  initial Delta=$(Delta_initial) (uniform), initial n=$(n_initial)")
println("  mix=$(mix), NC_scf=$(NC_scf), Np=$(Np_scf), tol_delta=$(tol_delta), tol_n=$(tol_n)")
println("  moment_count=NC stores Chebyshev orders 0:NC-1 (here 0:$(NC_scf - 1)).")

result = KPM.bdg_solve!(op; beta=beta, NC=NC_scf, Np=Np_scf, mix=mix,
                        tol_delta=tol_delta, tol_n=tol_n, maxiter=500, verbose=0)
result.converged || error("BdG SCF did not converge: $(result)")

println("\nSCF residual history (every 10th entry and the last)")
println("  iter       res_delta          res_n       max|Delta|       mean(n)          a")
for (history_index, entry) in enumerate(result.history)
    if history_index % 10 == 0 || history_index == length(result.history)
        @printf("  %4d  %14.6e  %14.6e  %14.6e  %12.8f  %10.6f\n",
                entry.iter, entry.res_delta, entry.res_n, entry.max_delta,
                entry.mean_n, entry.a)
    end
end
println("  converged=$(result.converged), iterations=$(result.iterations), a=$(result.a)")

# Compare the KPM update map at the converged fields with its dense exact form.
Hd = dense_bdg(h, mu, U_field, op.n, op.Δ)
F = eigen(Hermitian(Hd))
fermi(E) = 1 / (exp(beta * E) + 1)
occupations = fermi.(F.values)
Delta_exact = ComplexF64[-U_field[i] * sum(F.vectors[i, state] *
                                            conj(F.vectors[i + N, state]) * occupations[state]
                                            for state in eachindex(F.values))
                           for i in 1:N]
n_exact = Float64[2sum(abs2(F.vectors[i, state]) * occupations[state]
                       for state in eachindex(F.values)) for i in 1:N]

println("\nConverged-state KPM update versus dense exact update")
println("  NC       max|Delta_KPM-Delta_ED|       max|n_KPM-n_ED|")
update_errors = Dict{Int, NamedTuple}()
for NC in (64, 256)
    rh = KPM.rescale(op; eps=0.2)
    moments = KPM.bdg_local_moments(rh; NC=NC)  # default g_rho=2: full site density
    n_kpm, Delta_kpm = KPM.bdg_update(moments; beta=beta, Np=2 * NC)
    errors = (Delta=maximum(abs.(Delta_kpm .- Delta_exact)),
              n=maximum(abs.(n_kpm .- n_exact)))
    update_errors[NC] = errors
    @printf("  %3d       %18.10e       %18.10e\n", NC, errors.Delta, errors.n)
end
println("  Increasing NC from 64 to 256 changes the update errors by factors " *
        "Delta=$(update_errors[64].Delta / update_errors[256].Delta), " *
        "n=$(update_errors[64].n / update_errors[256].n) (a decrease means improved resolution).")

q = [0.0, 2pi / Ly]
eta = 0.3
volume = Float64(Lx * Ly)
common_stiffness = (; beta=beta, eta=eta, dir=1, disp=disp, NC=256,
                    volume=volume, g_J=1.0, Np=512, verbose=0)
stochastic = KPM.superfluid_stiffness(op, pos, q; NR=8, common_stiffness...)
psi_identity = Matrix{ComplexF64}(I, 2N, 2N)
deterministic = KPM.superfluid_stiffness(op, pos, q;
                                         psi_in=psi_identity,
                                         include_diamagnetic=true,
                                         common_stiffness...)

Jq = KPM.nambu_current_q(h, pos, q; dir=1, disp=disp)
Jmq = KPM.nambu_current_q(h, pos, -q; dir=1, disp=disp)
Hsc = dense_bdg(h, mu, U_field, op.n, op.Δ)
Hnormal = dense_bdg(h, mu, U_field, op.n, zeros(ComplexF64, N))
Pi_SC_ed = ed_two_energy_response(Hsc, Jq, Jmq; beta=beta, eta=eta, volume=volume)
Pi_N_ed = ed_two_energy_response(Hnormal, Jq, Jmq; beta=beta, eta=eta, volume=volume)
Ds_ed = real(Pi_N_ed) - real(Pi_SC_ed)

println("\nFinite-q transverse stiffness: q=$(q), dir=1, eta=$(eta), volume=$(volume)")
println("  stochastic result metadata (NR=8, default RNG):")
for (name, value) in pairs(stochastic)
    println("    $(name) = $(value)")
end
println("  deterministic trace uses psi_in = I_$(2N) (one probe per Nambu basis state).")
println("\n  response                 stochastic                 deterministic                  ED")
@printf("  Pi_SC       % .10e%+ .10eim  % .10e%+ .10eim  % .10e%+ .10eim\n",
        real(stochastic.Pi_SC), imag(stochastic.Pi_SC),
        real(deterministic.Pi_SC), imag(deterministic.Pi_SC), real(Pi_SC_ed), imag(Pi_SC_ed))
@printf("  Pi_N        % .10e%+ .10eim  % .10e%+ .10eim  % .10e%+ .10eim\n",
        real(stochastic.Pi_N), imag(stochastic.Pi_N),
        real(deterministic.Pi_N), imag(deterministic.Pi_N), real(Pi_N_ed), imag(Pi_N_ed))
@printf("  Ds/pi       % .10e              % .10e              % .10e\n",
        stochastic.Ds_over_pi, deterministic.Ds_over_pi, Ds_ed)
println("  deviations from ED:")
@printf("    stochastic:  |Delta Pi_SC|=%.6e, |Delta Pi_N|=%.6e, |Delta Ds/pi|=%.6e\n",
        abs(stochastic.Pi_SC - Pi_SC_ed), abs(stochastic.Pi_N - Pi_N_ed),
        abs(stochastic.Ds_over_pi - Ds_ed))
@printf("    deterministic: |Delta Pi_SC|=%.6e, |Delta Pi_N|=%.6e, |Delta Ds/pi|=%.6e\n",
        abs(deterministic.Pi_SC - Pi_SC_ed), abs(deterministic.Pi_N - Pi_N_ed),
        abs(deterministic.Ds_over_pi - Ds_ed))

# Lattice-complete stiffness: add the SC-vs-normal diamagnetic difference
# (paramagnetic-only subtraction is exact only for linear dispersion).
Dhat = Matrix(KPM.nambu_diamagnetic(h, pos; dir=1, disp=disp))
fermi_w(E) = 1 / (exp(beta * E) + 1)
function dia_ed(Hdense)
    F = eigen(Hermitian(Hdense))
    return real(sum(fermi_w(F.values[k]) * dot(view(F.vectors, :, k),
                    Dhat, view(F.vectors, :, k))
                    for k in eachindex(F.values))) / volume
end
Dia_SC_ed = dia_ed(Hsc)
Dia_N_ed = dia_ed(Hnormal)
Ds_complete_ed = Ds_ed + Dia_SC_ed - Dia_N_ed
println("\nLattice-complete stiffness (include_diamagnetic=true, deterministic trace):")
@printf("  Dia_SC: KPM=% .10e  ED=% .10e\n", deterministic.Dia_SC, Dia_SC_ed)
@printf("  Dia_N:  KPM=% .10e  ED=% .10e\n", deterministic.Dia_N, Dia_N_ed)
@printf("  Ds_complete/pi: KPM=% .10e  ED=% .10e  (paramagnetic-only KPM=% .10e)\n",
        deterministic.Ds_over_pi_complete, Ds_complete_ed, deterministic.Ds_over_pi)

println("\nConventions used")
println("  reduced Nambu block [particle; hole], with hole index i+N; hole_convention=:intervalley (== :singlet for this real h)")
println("  Ds/pi is the paramagnetic-only pair; Ds_complete/pi adds the lattice diamagnetic difference")
println("  Fermi level 0; Ds/pi = Re Pi_N - Re Pi_SC")
println("  g_rho=2 (n is the full spin-singlet site density, filling in [0,2]); g_J=1")
println("  volume=lattice-site count=area (unit lattice constant)")
println("  q is transverse to dir=1 and commensurate with the periodic torus")
