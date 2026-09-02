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

`KPM.optical_cond(m2, omega; area, Ef, m1, lambda)` accepts physical energies
and returns the two-dimensional optical conductivity in e²/h. The optional
`m1::CurrentMoments` supplies the diamagnetic term. The stored table
`mu[n,m] = Tr[Jalpha T_m Jbeta T_n]/D` is contracted directly with the optical
kernel in the same response/field orientation as `kubo_bastin_cond`; its
zero-frequency Hall limit therefore uses the ED/FHS-anchored convention
`sigma_xy = +C`. The bare `optical_cond1/2` methods instead use rescaled
energies and the unit conversions in their docstrings. The nonlinear `cpge`
method returns the bare three-current ``\chi_{\alpha\beta\gamma}`` bracket in
rescaled units, with no ``\Omega`` or ``1/(\omega_1\omega_2)`` prefactor; see
its docstring for the physical conversion and Ω=0 regularization.

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
σxyω = KPM.optical_cond(m2, ω; area=A, Ef=Ef) # physical energies, e²/h

mxx = KPM.cond_moments(h, Jx, Jx; NC=256, NR=8, rng=Xoshiro(42))
m1xx = KPM.current_moments(h, Jxx, 256, 8; rng=Xoshiro(42))
σxxω = KPM.optical_cond(mxx, ω; area=A, Ef=Ef, m1=m1xx, lambda=λ)
```

Notes:

- The current operators `Jx, Jy` must be built from the **original,
  unrescaled** Hamiltonian with the bond convention
  `(J_α)_ij = H_ij (r_i - r_j)_α` (building them from `h.H` divides
  conductivities by `a²`).
- Pass `rng=Xoshiro(seed)` to `dos_moments`, `cond_moments`, or
  `current_moments` for reproducible
  random-phase probe vectors; pass `psi_in` to supply your own.
- The typed methods are thin wrappers over the raw-array functions documented
  below — same code paths, same conventions — and the raw interface remains
  fully supported.
- Kwargs stored in the objects (`b`, `NH`) cannot be overridden in the typed
  calls; passing them raises an `ArgumentError` instead of silently
  disagreeing with the stored provenance.
- `optical_cond` uses physical energies and returns e²/h in two dimensions;
  the bare `optical_cond1/2`, `cpge`, and `d_cpge` methods use rescaled
  energies — see their docstrings. For CPGE, form
  ``y=\operatorname{Im}[(\chi_{\alpha\beta\gamma}+\chi_{\beta\alpha\gamma})/(\omega_1\omega_2)]``
  and ``\beta(\omega)=\lim_{\Omega\to0}\Omega y(\omega,\Omega-\omega)``.

```@docs; canonical=false
rescale
RescaledHamiltonian
dos_moments
cond_moments
current_moments
DosMoments
ConductivityMoments
CurrentMoments
optical_cond
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
using KPM, Plots, Random
include("examples/GrapheneModel.jl") # provides GrapheneLattice

L = 40
t = 2.3
Ham, Jx, Jy, Jxx, Jxy, Jyy = GrapheneLattice(L, L; t=t)
h = KPM.rescale(Ham; center=true)
NC = 128; NR = 4
psi = KPM.random_phase_vectors(Xoshiro(42), size(Ham, 1), NR)
m2yy = KPM.cond_moments(h, Jy, Jy; NC=NC, psi_in=copy(psi))
m1yy = KPM.current_moments(h, Jyy, NC, NR; psi_in=copy(psi))

area = L^2 * 3sqrt(3) / 2
ω = 1.0       # eV
Ef = 0.466    # eV
λ = 38.8e-3  # eV
σyy = KPM.optical_cond(m2yy, ω; area=area, Ef=Ef, m1=m1yy, lambda=λ)
println("σyy(ω) = ", σyy, " e²/h")
```

`omega`, `Ef`, and `lambda` are in the original Hamiltonian's energy units.
`area` is explicit geometry supplied by the model. The wrapper combines the
diamagnetic and paramagnetic terms in one adaptive integral and returns e²/h
for a two-dimensional sample. See `examples/OpticalGraphene.jl` for a
frequency sweep.

## Unitary time evolution

For a time-independent Hamiltonian rescaled as ``H = a\,H_{\mathrm{norm}} + b\,I``,
[`evolve`](@ref) propagates states by the Chebyshev/Bessel expansion
(Tal-Ezer & Kosloff),

```math
e^{-iHt}|\psi_0\rangle = e^{-ibt}\left[J_0(at)\,T_0(H_{\mathrm{norm}})
 + 2\sum_{n\ge 1}(-i)^n J_n(at)\,T_n(H_{\mathrm{norm}})\right]|\psi_0\rangle .
```

```julia
using KPM

h  = KPM.rescale(H; center=true)
ψt = KPM.evolve(h, ψ0, 2.5)                  # one state, one time
Ψt = KPM.evolve(h, Ψ0, [0.5, 1.0, -1.0])     # NH×NR block × time grid
```

Key properties, pinned by the test suite against dense `exp(-iHt)`:

- **No kernel damping.** The converged series is unitary; Jackson/Lorentz
  kernels are never applied on this path (they would lose norm). Norm
  conservation, reversibility (`t < 0` is valid), and the group property hold
  to the truncation tolerance.
- **Adaptive order.** By default `NC` is chosen from the superexponential
  Bessel tail ``|J_n(at)|`` via [`evolution_order`](@ref) (tolerance `tol`,
  default `1e-12`); a caller-fixed `NC` that truncates too early triggers an
  "evolution series tail" warning.
- **Multi-time memory trade.** A vector of times shares one Chebyshev
  recurrence with one accumulator per time (extra memory `NH × NR` per time);
  loop over scalar calls instead when memory is the binding constraint.
- **Stability guard.** Propagation is rejected (an error, not silent decay)
  when the recurrence grows unstably in the propagated states — the symptom
  of a rescaled spectrum outside the Chebyshev domain. The guard watches the
  propagated probe subspace, not the full spectrum.

The evolution front end is built on the internal coefficient-accumulating
matrix-function action ``f(H)|V\rangle = \sum_n c_n T_n(H_{\mathrm{norm}})|V\rangle``
(`KPM.chebyshev_action!`), shared with the Fermi projector below and
forthcoming filter applications; public APIs expose the physical operation,
not the accumulator.

```@docs; canonical=false
evolve
evolve!
evolution_order
```

## Fermi projector and local Chern marker

[`fermi_projector`](@ref) applies the KPM Fermi operator
``P = f_\beta(H - E_F)`` to a vector or block on the shared matrix-function
action, with coefficients from [`fermi_coefficients`](@ref): the closed-form
step series at ``\beta = \infty``, Gauss–Chebyshev quadrature of the smooth
Fermi factor at finite ``\beta``, Jackson-damped by default (the step is not
analytic — an undamped truncation Gibbs-rings the projector).

Unlike the moment APIs there is **no default `NC`**: the step series has no
rigorous tail bound, and a silent default could return a confidently wrong
topological invariant. The Jackson-damped resolution is
``\Delta E \approx \pi a/N_C``, which must sit well inside the spectral gap
at ``E_F`` — choose ``N_C \gtrsim 4\pi a/\Delta_{\mathrm{gap}}`` and verify
by doubling ``N_C``. Near a transition (or a band edge), finite ``\beta`` is
the recommended regularizer; the effective smearing is
``\max(\Delta E, \sim 4/\beta)``.

On the projector, [`chern_marker`](@ref) evaluates the Bianco–Resta local
Chern marker at requested basis indices,

```math
m_i = -4\pi\,\mathrm{Im}\,\langle i|\,P\,X\,Q\,Y\,P\,|i\rangle,
\qquad Q = I - P,
```

using two projector actions per site batch (the ``Q`` term never
materializes: for ``u = P e_i``,
``\mathrm{Im}\langle u|XQY|u\rangle = -\mathrm{Im}\langle u|XPY|u\rangle``
identically). The sign convention is anchored to the package's Hall
conductivity: a bulk average over complete cells equals the same ``+C`` as
``\sigma_{xy} = +C\,e^2/h`` (pinned by the Haldane/Fukui–Hatsugai–Suzuki
test fixture).

```julia
using KPM

h  = KPM.rescale(H; center=true)
Pv = KPM.fermi_projector(h, v; Ef=0.0, NC=1024)

# site-resolved markers at `sites`, coordinates x, y (caller data)
m  = KPM.chern_marker(h, x, y; Ef=0.0, sites=sites, NC=1024)
C  = KPM.chern_marker_average(m; area=n_cells * cell_area)

# stochastic regional estimate: NR probes on a region, mean ± std/√NR;
# each entry of est already estimates the WHOLE region sum, so average the
# probes before normalizing (never pass est itself to chern_marker_average)
est = KPM.chern_marker_region(h, x, y; Ef=0.0, region=region,
                              rng=Xoshiro(1), NR=32, NC=1024)
C ≈ KPM.chern_marker_average(mean(est); area=region_area)
```

Semantics, pinned by the test suite against exact dense projectors:

- **Geometry is user data.** Coordinates enter as plain vectors `x`, `y`
  (diagonal position operators); site lists, cell groupings, and areas are
  explicit inputs. The raw markers are per **orbital** with units of x·y;
  [`chern_marker_average`](@ref) requires the explicit area that makes the
  result dimensionless.
- **Open boundaries only.** A diagonal position operator is not a valid
  position observable on a torus; periodic-coordinate marker formulations
  are out of scope. At ``\beta = \infty`` the marker summed over an entire
  finite open sample is ``\approx 0`` (exactly
  ``\mathrm{Im}\,\mathrm{Tr} = 0`` for an idempotent projector): topology
  is read from a **bulk** average with the boundary excluded, never from
  the full trace.
- **Finite temperature is a smooth diagnostic, not a projector marker.**
  The Fermi–Dirac operator at finite ``\beta`` is deliberately not
  idempotent, so the zero-trace identity does *not* hold — the thermal
  whole-sample sum is genuinely nonzero (pinned against dense references in
  the tests). The bulk average still tracks ``C`` for temperatures well
  below the gap, and thermal smoothing is the recommended regularizer near
  a transition.
- **Deterministic vs stochastic cost.** Site-resolved maps cost two
  ``N_C``-step recurrences per `batch_size` sites. For a regional average,
  [`chern_marker_region`](@ref) replaces ``|R|`` site columns with `NR`
  random-phase probes restricted to the region (cost independent of
  ``|R|``), returning per-probe estimates so the statistical error is
  explicit.

```@docs; canonical=false
fermi_projector
fermi_coefficients
chern_marker
chern_marker_region
chern_marker_average
```

# Moment calculation

The first step in KPM is calculating moments using Hamiltonians (and current operators for conductivity, etc.).
Functions with `!` are the more efficient in-place versions; functions without `!` are convenient wrappers that call the in-place implementations.

```@docs; canonical=false
kpm_1d
kpm_1d!
kpm_2d
kpm_2d!
kpm_2d_blocking
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
  - `dos_moments` → `DosMoments`; `cond_moments` → `ConductivityMoments`;
    `current_moments` → `CurrentMoments`
  - reconstruction via the same names as the raw interface: `dos(m)`,
    `dos0(m)`, `kubo_bastin_cond(m, Ef; area)`, `d_dc_cond(m, E)`,
    `dc_cond0(m)`, `dc_cond_single(m, Ef)`, `optical_cond(m2, omega; area)`
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

- Unitary evolution:
  - `evolve` (typed; adaptive `NC` via `evolution_order`)

- Fermi projector / local Chern marker:
  - `fermi_projector` (typed; coefficients via `fermi_coefficients`)
  - `chern_marker`, `chern_marker_region`, `chern_marker_average`

- Conductivity (DC / optical):
  - `kubo_bastin_cond` (absolute units, e²/h; ED-validated)
  - `d_dc_cond`, `dc_cond0`, `dc_cond_single`
  - `optical_cond` (typed, physical energies, e²/h in 2D)
  - `optical_cond1`, `d_optical_cond1`
  - `optical_cond2`, `d_optical_cond2`

- Nonlinear / CPGE:
  - `cpge`, `d_cpge`

- Kernels:
  - `JacksonKernel`, `LorentzKernels`

- Utilities (KPM.Utils / device helpers):
  - `wrapAdd`, `normalizeH`, `isNotBoundary`, `timestamp`
  - device helpers: `whichcore`, `maybe_to_device`, `maybe_to_host`, `maybe_on_device_rand`, `maybe_on_device_zeros`

For more details see the full API reference below.

```@autodocs
Modules = [KPM]
```
