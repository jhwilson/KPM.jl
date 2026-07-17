```@meta
CurrentModule = KPM
```

# KPM

```@index
```

## Installation

This package is currently unregistered. Install the latest version directly from GitHub:

```bash
] add https://github.com/Pixley-Research-Group-in-CMT/KPM.jl
```

Notes:

- The package supports CUDA.jl versions 4 and 5.
- After installation import with:

```julia
using KPM
```

- To update the package run:

```bash
] update KPM
```
and provide your GitHub username/password if prompted.

For more details see the project's README.

## Design principle: models are user data

KPM.jl computes spectral quantities from operators. It never infers model
content. What an index means (site, orbital, spin, sublattice, cell), which
bonds exist, where things sit in space, which degeneracies are implicit, and
what the volume is are all **user-supplied data**, assembled with whatever
model-building tools you prefer and passed in as plain arrays, callables, and
numbers:

  * operators: `H` (and, for BdG, the pairing structure) — duck-typed, only
    `size` and `mul!` required;
  * geometry: positions `pos` and bond displacements `disp(i, j)` — these
    *define* the current operator `(J_dir)_ij = H_ij (r_i - r_j)_dir`, the
    Peierls coupling, and the diamagnetic operator;
  * normalization content: degeneracy factors (`g_rho`, `g_J`), `volume`,
    interaction couplings.

Why this matters — the SSH example: the same SSH Hamiltonian matrix can be
modeled as `2N` lattice sites with split positions, or as `N` cells with two
orbitals each (co-located or split). These embeddings have **different
current operators** — an intra-cell hop carries current only if the user's
positions say its displacement is nonzero — and therefore different f-sum
rules and transport responses. All embeddings are legitimate physics; only
the user knows which one describes their system, so the package takes the
displacement data verbatim and never derives geometry from the matrix.
The same applies to degeneracies: nothing multiplies your results by 2 for
spin unless you ask for it (`g_rho`, `g_J` are explicit, with documented
defaults).

Practical consequence: if a function needs geometry, degeneracy, or volume,
it takes them as explicit arguments. If you see a result that seems off by a
geometric or combinatorial factor, check the model data you supplied before
suspecting the algorithm — and see the conventions tables below for what
each factor means.

## Conventions and rescaling

All quantities are expanded in Chebyshev polynomials of the rescaled
Hamiltonian `H_norm = (H - b I)/a`, whose spectrum must lie inside (-1, 1).
`KPM.normalizeH(H)` returns `(a, H/a)` assuming a spectrum symmetric about
zero; `KPM.normalizeH(H; center=true)` finds both spectral edges and returns
`(a, b, H_norm)` — use this for spectra that are not particle-hole symmetric,
and pass `b` on to `dos(mu, a; b=b)`.

The DOS moments are `μ_n = Tr[T_n(H_norm)]/D`, estimated with `NR`
unit-normalized random-phase vectors, so `μ_0 = 1` and the DOS integrates to
one. `kpm_1d` uses the moment-doubling trick (`NC` moments from `NC/2`
matrix-vector recurrence steps), so `NC` must be even. The reconstruction is

```
ρ(E) = Σ_n h_n g_n μ_n T_n(x) / (a π √(1-x²)),   x = (E - b)/a,
```

with `h_0 = 1`, `h_n = 2` for `n ≥ 1`, and `g_n` a damping kernel
(Jackson by default).

### Conductivity units

`KPM.kubo_bastin_cond(mu2D, a, Ef; b, NH, area)` returns the Kubo–Bastin DC
conductivity **in units of e²/h**, where `mu2D = kpm_2d(H_norm, Jα, Jβ, NC,
NR, NH)` and the current operators follow the bond convention
`(J_α)_ij = H_ij (r_i - r_j)_α` (i.e. `J_α = iħ v_α`, as in
`examples/GrapheneModel.jl`). `α` is the response direction, `β` the field
direction, and `σ_xy = +1 e²/h` corresponds to Chern number `C = +1`. The
absolute normalization — including the `1/a²` rescaling, the `D/A`
trace-to-density conversion, and the sign — is validated against exact
diagonalization on the Haldane model in `test/kubo_bastin_test.jl`
(quantized Hall plateau to ~1%, longitudinal channel to ~5% with matched
Lorentzian broadening).

The older `dc_cond0` / `dc_cond_single` return bare Chebyshev Fermi-surface
sums, proportional to the longitudinal Kubo–Greenwood conductivity but not
in physical units; prefer `kubo_bastin_cond` for quantitative work.
`d_dc_cond` returns the Kubo–Bastin integrand dσ(E), related to the physical
conductivity by σ = -(2e²/h)·(D/(A·a))·∫dE f(E) dσ(E).

The optical (`optical_cond1/2`) and nonlinear (`cpge`) responses are quoted
in the natural units stated in their docstrings; their absolute
normalizations have **not** been validated against exact diagonalization.

```@docs; canonical=false
kubo_bastin_cond
```

### Thermoelectric (Seebeck) response

The same `cond_moments` / `mu2D` used by `kubo_bastin_cond` are reused to
reconstruct the equal-energy Kubo–Greenwood transport distribution
`Sigma_ab(E)`, the dissipative, symmetric part of the conductivity tensor. The
Chester–Thellung–Kubo–Greenwood (Jonson–Mahan) integrals are
`L_r = integral dE (E-mu)^r (-df/dE) Sigma(E)` for `r = 0, 1, 2`, and the
electron-convention (`q = -|e|`) Seebeck coefficient is
`S = -L0 \ (beta * L1)` in units of `k_B/|e|`; the carrier sign emerges from
the particle-hole asymmetry of `Sigma(E)`, never from flipping the charge.
`seebeck_uVK` converts with `k_B/|e| = 86.17333262 uV/K`.

In 3D, with caller-supplied `volume` in `length^3`, `L0` is in
`(e^2/h)/length` (in general `Sigma` and `L_r` carry
`(e^2/h) * length^(2-d)`, with `L_r` gaining `energy^r`). `S` is independent
of a common volume and of the `g_J` degeneracy factor, but both are normalized
correctly rather than relying on cancellation. By default,
`sigma_min = 1e-6 * max_E |Sigma(E)|` over an NC-resolved usable-band scan with
`max(257, 4NC+1)` points; an insulating thermal window yields `S = NaN` with a
warning while `L0`, `L1`, and `L2` are still reported. Pass `sigma_min`
explicitly, for example `sigma_min=0.0`, to disable this floor. `neg_weight` is
the fraction of thermally weighted `|Sigma|` from negative values, a convergence
diagnostic that is never clipped; for tensor results it is maximized over the
diagonal components only because off-diagonal distributions may legitimately
be negative.

!!! warning
    This route reconstructs the **symmetric part only**: it contains no
    antisymmetric (Hall-like) components, including the zero-field anomalous
    Hall/Nernst response of time-reversal-broken models such as the Haldane
    model in this package's own tests (use `kubo_bastin_cond` for those).
    It covers elastic/static scattering only — not phonon drag, inelastic, or
    interacting transport. Kernel broadening is a physical/numerical choice,
    and clean ballistic DC transport is **not** a finite bulk conductivity;
    check system size, `NC`, `NR`, and broadening before quoting bulk values.
    `beta=Inf` is rejected: at `T=0` the thermal window is a delta function
    and `L1` vanishes trivially; the correct `T -> 0` statement is the Mott
    relation.

```julia
h   = KPM.rescale(H; center=true)
mxx = KPM.cond_moments(h, Jx, Jx; NC=512, NR=12, rng=Xoshiro(42))
r   = KPM.thermoelectric(mxx, mu_chem; beta=beta, volume=V)  # ThermoelectricResult
r.L0; r.S_over_kB_over_e; KPM.seebeck_uVK(r)

M = Matrix{KPM.ConductivityMoments}(undef, 2, 2)  # caller's own axis ordering
M[1, 1] = mxx; M[1, 2] = M[2, 1] = mxy; M[2, 2] = myy  # mxy = cond_moments(h, Jx, Jy; ...)
r_tensor = KPM.thermoelectric(M, mu_chem; beta=beta, volume=V) # symmetric tensors; left solve, no inverse
```

Tensor components are symmetrized before the open-circuit solve, and a skew
fraction above 5% triggers a stochastic-noise/inconsistent-moments warning.

```@docs; canonical=false
transport_distribution
transport_integrals
thermoelectric
ThermoelectricResult
seebeck_uVK
fermi_window
```

## Typed front end

The recommended interface packages the rescaling and the moment metadata into
small value types, so `(a, b, NH, NR)` never have to be threaded by hand:

```julia
using KPM, Random

h = KPM.rescale(H; center=true)          # RescaledHamiltonian: fields H, a, b
m = KPM.dos_moments(h; NC=1024, NR=12)   # DosMoments: records a, b, NH, NR
E, rho = KPM.dos(m)                      # rescaling applied automatically
rho0 = KPM.dos0(m)                       # DOS at E = h.b

m2  = KPM.cond_moments(h, Jx, Jy; NC=256, NR=8)
σxy = KPM.kubo_bastin_cond(m2, Ef; area=A)   # e²/h; NH, a, b come from m2
dσE = KPM.d_dc_cond(m2, E_values)
```

Notes:

- The current operators `Jx, Jy` must be built from the **original,
  unrescaled** Hamiltonian with the bond convention
  `(J_α)_ij = H_ij (r_i - r_j)_α` (building them from `h.H` divides
  conductivities by `a²`).
- Pass `rng=Xoshiro(seed)` to `dos_moments` / `cond_moments` for reproducible
  random-phase probe vectors; pass `psi_in` to supply your own.
- The typed methods are thin wrappers over the raw-array functions documented
  below — same code paths, same conventions — and the raw interface remains
  fully supported.
- Kwargs stored in the objects (`b`, `NH`) cannot be overridden in the typed
  calls; passing them raises an `ArgumentError` instead of silently
  disagreeing with the stored provenance.
- `optical_cond1/2` and `cpge` do not have typed wrappers yet and take
  energies in rescaled units — see their docstrings.

```@docs; canonical=false
rescale
RescaledHamiltonian
dos_moments
cond_moments
DosMoments
ConductivityMoments
```

## Bogoliubov–de Gennes and superfluid stiffness

The BdG layer is a matrix-free Nambu wrapper around any duck-typed normal
operator that supports `size` and `mul!`.  The pairing block is a general
operator `D` (onsite `Diagonal`, sparse bond matrix, or a matrix-free operator
with `size` and five-argument `mul!` for both `D` and `adjoint(D)`);
self-consistent pairing and density fields are solved from one-recurrence local
moments: the particle and hole entries of each recurrence vector provide both
channels without assembling the BdG matrix.

`superfluid_stiffness` evaluates a transverse finite-wavevector response as a
paired superconducting/normal two-point KPM calculation.  The normal reference
keeps the assembled hopping, chemical potential, interaction, and converged
Hartree density, zeroing only the pairing block.  This paramagnetic-only
subtraction is exact for strictly linear (continuum-Dirac-like) dispersions;
for lattice models pass `include_diamagnetic=true` to add the
superconducting-vs-normal difference of the diamagnetic (kinetic) term —
`O(Delta^2)`, zero at `Delta = 0` — as a Fermi-weighted single-operator trace
(see the conventions table and the `superfluid_stiffness` docstring).

```julia
using KPM

op = KPM.BdGOperator(h; mu=-0.5, U=2.5, n=fill(0.5, size(h, 1)),
                     Delta=fill(0.2 + 0im, size(h, 1)))
scf = KPM.bdg_solve!(op; beta=8.0, NC=256, Np=512, mix=0.3,
                     tol_delta=1e-7, tol_n=1e-7)
q = [0.0, 2pi / Ly]
stiffness = KPM.superfluid_stiffness(op, pos, q; beta=8.0, eta=0.3,
                                     dir=1, disp=disp, NC=256, NR=8,
                                     volume=Float64(Lx * Ly))
```

### User-defined pairing channels

Pairing structure is model content, so — like geometry and degeneracies — it
is supplied by the user as plain data, never inferred from the matrix.  A
[`PairingChannel`](@ref) declares which index pairs pair, with what form
factor, coupling, and parity:

```julia
# spinless p-wave (Kitaev) chain: odd bond channel
channel = KPM.PairingChannel([(i, i + 1) for i in 1:N-1], 1.0, V, :odd)
op = KPM.BdGOperator(h; mu=mu, U=0.0,
                     D=KPM.pairing_matrix(N, [channel]; amplitude=0.1),
                     hole_convention=:conjugate)
scf = KPM.bdg_solve!(op, [channel]; beta=8.0, NC=512, mix=0.3,
                     update_density=false, g_rho=1)
```

The self-consistency unknowns are the **per-bond amplitudes**
`Delta_b = -V_b (F_ij + s F_ji)/2` (`s = ±1` per the channel parity;
`F_ij = <c_j c_i>`-type anomalous averages extracted from the same
one-recurrence local moments, O(1) entry reads per bond).  Channel `weights`
act as the seed pattern and as the projection diagnostic
([`channel_amplitude`](@ref)); they do not restrict the variational space.
Channels must have disjoint bond support: per-bond unknowns make same-bond
channel decompositions degenerate, and mixed-parity pairing on a shared bond
is representable only as fixed background entries in `D`, not as competing
self-consistent channels. Onsite bonds `(i, i)` are only legal in `:even`
channels; entries of `D` outside the declared channels are held fixed by the
solver. With channels supplied, `U` drives only the Hartree term — an onsite
`:even` channel with `V = U` reproduces the legacy onsite solver. The built-in
`U`/`n` Hartree term `-(U/2)n` is the reduced-singlet or spinless prescription;
a flattened explicit-spin Hubbard model instead needs the partner-spin field
`-U n_partner`, which the package cannot construct because models are user
data and it has no spin ontology. Explicit-spin callers must fold such shifts
into `h` themselves (using any fixed `n` field as caller-owned input), keep
`update_density=false`, and disable the built-in Hartree channel with `U=0`,
as in `test/rashba_test.jl`. The onsite-singlet pairing channel itself remains
fully supported, with `g_rho=1` for densities callers reconstruct themselves.
A caller-supplied Hartree operator is explicit future work.

### Conventions (load-bearing)

| Topic | Convention |
| --- | --- |
| Nambu layout | `[particle; hole]`, with hole index `i+N`. |
| Hole-block convention | `hole_convention=:intervalley` (default): hole block `-xi` with the **same** `h`, presuming `h_{-K}^* = h_K` (exact for real-symmetric `h`; non-symmetric complex `h` requires `assume_intervalley=true`). `hole_convention=:conjugate` (formerly `:singlet`, which remains an accepted alias): the fully general `-conj(xi)` hole block for any normal operator, including explicit-spin / spin-orbit-coupled `h` — singlet vs triplet content lives in the structure of `D`, not in the hole block. Both conventions coincide for real-symmetric `h`; the current and diamagnetic vertices are convention-aware. |
| Particle-hole symmetry (parity-resolved) | With the `:conjugate` hole block, `tau_y K` is an exact PH symmetry iff `transpose(D) == +D` (even parity: onsite, extended-s, d-wave) and `tau_x K` iff `transpose(D) == -D` (odd parity: p-wave, explicit-spin onsite singlet `i sigma_y Delta`). Either way the spectrum has exact `±E` pairs. Mixed-parity `D` has no exact PH symmetry — nothing breaks: `b=0` remains a radial bound protected by the runtime guards; only Chebyshev resolution is paid. `moment_parity=:EVEN` eligibility inherits the same conditions. |
| Interaction sign | `U > 0` is attractive: `H_int = -U sum n_up n_down` and `Delta_i = -U_i<c_down c_up>`. |
| Hartree | `-(U/2) n_i`, with full site density; there is no double-counting correction, and the absorbed constant is **not** split from `mu`. |
| Chemical potential | It is inside `H_BdG`, so all Fermi factors are at quasiparticle energy `0`. |
| Degeneracies | The reduced block integrates one spin species, so the default `g_rho=2` reconstructs the full spin-singlet site density (filling in `[0,2]`, consistent with the `-(U/2)n` Hartree term); `g_rho=1` gives per-spin density. `g_J` multiplies the response. Neither is applied silently. |
| Volume | The response is per caller-supplied `volume`, in the caller's units. |
| Rescaling | `b=0` (radial bound); `a=2*radius/(2-eps)` from hardened multi-start power iteration with default `eps=0.2`. Runtime recurrence guards abort loudly if the spectrum escapes `(-1,1)`; `rescale(op; bound=:gershgorin)` gives a certified upper bound for assembled operators, and `radius=...` accepts a known bound. |
| Stiffness definition | `Ds/pi = Re Pi_N - Re Pi_SC` (paramagnetic-only, exact for linear dispersion), with paired probes, `NC`, kernel, `Np`, `eta`, chemical potential, and Hartree field; both states share a common Chebyshev scale `a_common = max(a_SC, a_N)` so the finite-`NC` broadening is identical in the subtraction. `include_diamagnetic=true` adds the lattice diamagnetic difference: `Ds_over_pi_complete = (Re Pi_N - Re Pi_SC) + (Dia_SC - Dia_N)`, anchored against the free-energy curvature `(2 g_J/V)(F''_SC - F''_N)` in the tests. |
| Rigid-`Delta` convention (bond pairing) | This is the package's fixed-gauge, rigid-`D` functional: the vector potential adds Peierls phases to kinetic bonds only, while `D` contributes no current or diamagnetic vertex. It reduces to the standard mean-field Kubo (Scalapino–White–Zhang) treatment for onsite pairing, and the free-energy-curvature anchor tests exactly this functional. Local gauge covariance of a nonlocal pair field and the charge-`2e` pairing response are out of scope. |
| `q` and `eta` guidance | Choose `q=2pi/L` transverse to `dir` and commensurate with the torus. Choose `eta` between the finite-size level spacing and the gap, with `eta >= 5 a pi / NC`. |

With a CUDA GPU active, `bdg_solve!` and `superfluid_stiffness` run their
Chebyshev recurrences on the device: the operator is assembled into one
sparse BdG matrix (`bdg_assemble`) and moved to the GPU, while fields,
mixing, checkpoints, and spectral reconstruction stay on the host.  This
requires a sparse `h` and matrix `D` blocks; operators that cannot be
assembled to the device (matrix-free or dense blocks) run entirely on the
CPU in `bdg_solve!`, and throw in `superfluid_stiffness`/`diamagnetic_term`
(whose workspaces follow the globally active device).

```@docs; canonical=false
BdGOperator
ScaledOperator
bdg_assemble
spectral_radius
rescale(::BdGOperator)
bdg_site_moments
bdg_update
bdg_solve!
BdGSCFResult
bdg_local_moments
LocalBdGMoments
PairingChannel
pairing_matrix
channel_amplitude
bdg_channel_moments
bdg_channel_update
ChannelBdGMoments
bdg_checkpoint
bdg_restore!
nambu_current_q
nambu_diamagnetic
diamagnetic_term
two_energy_response
superfluid_stiffness
```

Convergence of the self-consistency loop can be accelerated with
`bdg_solve!(...; mixing=:anderson)` (safeguarded Type-II Anderson; see the
docstring — the accelerated step must be kept away from the unstable
normal-state root of the gap equation, which the built-in delay and step cap
handle).

## Quick examples

### 1) Density of States (DOS) — concise example

```julia
using KPM, LinearAlgebra, SparseArrays, Plots

# small 1D tight-binding (periodic)
function tb1dchain(N; t=1.0)
  H = spzeros(N,N)
  for i in 1:N-1
    H[i,i+1] = -t; H[i+1,i] = -t
  end
  H[1,N] = -t; H[N,1] = -t
  return H
end

N = 1000
NC = 256
NR = 4
H = tb1dchain(N)
# rescale H to (-1,1)
a, Hn = KPM.normalizeH(H)
mu = KPM.kpm_1d(Hn, NC, NR)
E, rho = KPM.dos(mu, a; kernel=KPM.JacksonKernel, N_tilde=500)

plot(E, rho, xlabel="E", ylabel="DOS", legend=false)
```

Reference (full example):

```julia
using KPM
using LinearAlgebra
using SparseArrays

# Simple dense 1D tight-binding Hamiltonian (periodic)
function tb1dchain(N::Integer; t::Real=1.0)
    H = zeros(Float64, N, N)
    for i in 1:(N-1)
        H[i, i+1] = -t
        H[i+1, i] = -t
    end
    H[1, N] = -t
    H[N, 1] = -t
    return H
end

# Parameters
N = 1000               # system size
NC = 1024               # Chebyshev order
NR = 10               # number of random vectors for stochastic trace
nE = 1000             # output energy grid points

H = tb1dchain(N)
# Rescale H -> (-1, 1)
a, H_norm = KPM.normalizeH(H)

# Compute Chebyshev moments (DOS)
mu = KPM.kpm_1d(H_norm, NC, NR)    # returns moments (array-like)

# Reconstruct DOS on a grid and map energies back to physical scale
E, rho1024 = KPM.dos(mu, a; kernel = KPM.JacksonKernel, N_tilde=nE)
E, rho64 = KPM.dos(mu[1:64], a; kernel = KPM.JacksonKernel, N_tilde=nE)
E, rho32 = KPM.dos(mu[1:32], a; kernel = KPM.JacksonKernel, N_tilde=nE)

# Analytical DOS 
rho_exact = zeros(length(E))
mask = abs.(E) .< 2
rho_exact[mask] = 1.0 ./ (π * sqrt.(4 .- E[mask].^2))

# plot the DOS
plot(xlabel=L"E", ylabel="DOS"*L"\;\rho(E)",
        legend = :top, 
        #xlim=[-0.1,0.8586],ylim=[-0.001,0.035],
        framestyle = :box,grid=false,
        xtickfontsize=12, ytickfontsize=12,
        xguidefontsize=12, yguidefontsize=12,
        legendfontsize=12,
        )
plot!(E, [rho1024 rho64 rho32], lw=[4 3 2],label=[L"N_C=1024" L"N_C=64" L"N_C=32"])
plot!(E, rho_exact, c=:black, ls=:dash, label=L"\mathrm{Analytic}")
```

### 2) Optical conductivity (graphene) — concise example

```julia
using KPM, Plots
include("examples/GrapheneModel.jl") # provides GrapheneLattice

L = 60
Ham, Jx, Jy, Jxx, Jxy, Jyy = GrapheneLattice(L, L)
a = 3.5
Hn = Ham / a
NC = 256; NR = 6
mu2d = zeros(ComplexF64, NC, NC)
psi = exp.(2π*1im*rand(Hn.n, NR))
KPM.normalize_by_col(psi, NR)
KPM.kpm_2d!(Hn, Jy, Jy, NC, NR, Hn.n, mu2d, psi, psi)

# compute a sample optical conductivity (2D contribution) at ω=0.5
ω = 0.5
σ2 = KPM.d_optical_cond2(mu2d, NC, ω, 0.0)
println("Optical conductivity (2D part) at ω=", ω, " : ", σ2)
```

Reference (full example):

```julia
using Plots
using LaTeXStrings
using KPM

include("GrapheneModel.jl") # Include the GrapheneLattice function and related structures

function full_optical_condT0(mu1d,mu2d, NC, ω; δ=1e-5, λ=0.0, kernel=KPM.JacksonKernel,
    h = 0.001, Emin= -0.8, Emax = 0.0
    )
    # This function is used to calculate the full optical conductivity
    # by combining the 1D and 2D contributions.
    x_all = collect(Emin:h:Emax)
    y_1 = zeros(ComplexF64, length(x_all))
    y_2 = zeros(ComplexF64, length(x_all))
    mu1d_dev = KPM.maybe_to_device(mu1d[1:NC])
    mu2d_dev = KPM.maybe_to_device(mu2d[1:NC, 1:NC])

    for (i, x) in enumerate(x_all)
        y_1[i] += KPM.d_optical_cond1(mu1d_dev, NC, x; δ=δ, λ=λ, kernel=kernel)
        y_2[i] += KPM.d_optical_cond2(mu2d_dev, NC, ω, x; δ=δ, λ=λ, kernel=kernel)
    end
    return (sum(y_1) * h * (-1im / ω), sum(y_2) * h * (-1im / ω))
    #y_all = y_1 .+ y_2;
    #y_integral = sum(y_all) * h;
    
    #return y_integral*(-1im / ω) # -ie^2 / (ħ^2 * ω)
end

L = 200
Ham, Jx, Jy,Jxx,Jxy,Jyy = GrapheneLattice(L,L);

a = 3.5
H_norm = Ham ./ a
NC = 512 #512
NR = 10
NH = H_norm.n
mu_2d_yy = zeros(ComplexF64, NC, NC)
psi_in_l = exp.(2pi * 1im * rand(NH, NR));
KPM.normalize_by_col(psi_in_l, NR)
psi_in_r = psi_in_l
@time KPM.kpm_2d!(H_norm, Jy, Jy, NC, NR, NH, mu_2d_yy, psi_in_l, psi_in_r; verbose=1);

mu_1d_yy = KPM.kpm_1d_current(H_norm,Jyy, NC, NR; verbose=1)

t = 2.3
μ = 0.466
Ef = μ/t/a
λ = 38.8*10^(-3)/t/a
ωs = collect(LinRange(0.03, 0.982, 100))
res = zeros(ComplexF64, length(ωs))
res2 = zeros(ComplexF64, length(ωs))
for (i, ω) in enumerate(ωs)
    res[i], res2[i] = full_optical_condT0(mu_1d_yy,mu_2d_yy, NC, ω;λ=λ,Emax=Ef)
    #res[i] = full_optical_condT0(mu_1d_yy,mu_2d_yy, NC, ω;λ=λ,Emax=Ef)
    println(i)
end
σyyreal = real.(res2)./a
σyyimag = imag.(t*a*res.+res2)./a
ωsreno = ωs*t*a

plot(ylabel = L"\sigma^{yy}/\sigma_0",xlabel = L"\hbar \omega(\mathrm{eV})",
     framestyle = :box,grid=false,legend=:topright,
        xtickfontsize=12, ytickfontsize=12,
        xguidefontsize=12, yguidefontsize=12,
        legendfontsize=12,#titlefontsize=12,
         ylim=(-2,8)
        )
scatter!(ωs*t*a, σyyreal, label="real",markerstrokewidth=0.0)
scatter!(ωs*t*a, σyyimag, label="imag",markerstrokewidth=0.0)
```

# Moment calculation

The first step in KPM is calculating moments using Hamiltonians (and current operators for conductivity, etc.).
Functions with `!` are the more efficient in-place versions; functions without `!` are convenient wrappers that call the in-place implementations.

```@docs; canonical=false
kpm_1d
kpm_1d!
kpm_2d
kpm_2d!
```

# Applications

## DOS

To calculate the density of states (DOS) first calculate moments using `kpm_1d` / `kpm_1d!` with default (random) input vectors.
Then use the returned moments (`mu`) to evaluate the DOS.
There is also a convenience overload that accepts a Hamiltonian directly and performs the moment calculation for you via `dos`.

```@docs; canonical=false
dos
```

# Kernels

Kernels are functions with signature
```
kernel(n::Int64, N::Int64) -> Float64
```
such that when `n == 0` the kernel returns `1`, and when `n == N-1` it returns a small number close to `0`.

The package provides the `JacksonKernel` (the default for most applications) and `LorentzKernels`.

```@docs; canonical=false
JacksonKernel
```

The Lorentz kernel is useful for Green's functions because it preserves certain symmetries.
`LorentzKernels(λ)` returns a kernel function parameterized by λ.

```@docs; canonical=false
LorentzKernels
```

## API overview

Below is a concise list of the main public APIs provided by the package.

- Typed front end (recommended):
  - `rescale` → `RescaledHamiltonian`
  - `dos_moments` → `DosMoments`; `cond_moments` → `ConductivityMoments`
  - reconstruction via the same names as the raw interface: `dos(m)`,
    `dos0(m)`, `kubo_bastin_cond(m, Ef; area)`, `d_dc_cond(m, E)`,
    `dc_cond0(m)`, `dc_cond_single(m, Ef)`
  - thermoelectric reconstruction: `transport_distribution(m, E; volume)`,
    `transport_integrals(m, mu; beta, volume)`,
    `thermoelectric(m, mu; beta, volume)` → `ThermoelectricResult`,
    `seebeck_uVK`

- Moment / KPM core:
  - `kpm_1d`, `kpm_1d!`
  - `kpm_1d_current`, `kpm_1d_current!`
  - `kpm_2d`, `kpm_2d!`
  - `kpm_3d`, `kpm_3d!`

- DOS / LDOS:
  - `dos`, `dos0`
  - `ldos_mu`

- Conductivity (DC / optical):
  - `kubo_bastin_cond` (absolute units, e²/h; ED-validated)
  - `d_dc_cond`, `dc_cond0`, `dc_cond_single`
  - `optical_cond1`, `d_optical_cond1`
  - `optical_cond2`, `d_optical_cond2`

- Nonlinear / CPGE:
  - `cpge`, `d_cpge`
  - Integration helpers: `Λnmp`, `Λn`, `Λnm`, `gn_R`, `gn_A`, `Δn`

- Kernels:
  - `JacksonKernel`, `LorentzKernels`

- Utilities (KPM.Utils / device helpers):
  - `wrapAdd`, `normalizeH`, `isNotBoundary`, `timestamp`
  - device helpers: `whichcore`, `maybe_to_device`, `maybe_to_host`, `maybe_on_device_rand`, `maybe_on_device_zeros`

For more details see the full API reference below.

```@autodocs
Modules = [KPM]
```
