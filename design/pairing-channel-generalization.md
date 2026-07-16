# Plan: general pairing channels for the BdG layer

Status: implemented (branch `feature/pairing-channels`, 2026-07-16); §3–§5
realized as staged commits with the §4 gates in `test/kitaev_test.jl`,
`test/pairing_channel_test.jl`, `test/dwave_test.jl`, `test/rashba_test.jl`,
`test/ssh_invariance_test.jl`, and `test/bond_stiffness_test.jl`. One
deviation from §3.3: the SCF unknowns are the per-bond amplitudes (full
Hartree–Fock–Bogoliubov mean field); channel weights are the seed pattern and
projection diagnostic (`channel_amplitude`), not a variational restriction.
Scope: generalize the BdG +
superfluid-stiffness layer (PR #4) from onsite spin-singlet pairing to
user-defined pairing channels (singlet, triplet, bond, orbital-matrix), with
the particle-hole and vertex content following from user-supplied model data
rather than package assumptions.

## 1. Division of responsibility (design principle)

KPM.jl supplies **spectral algorithms**: Chebyshev recurrences, moment
collection, kernel reconstruction, self-consistency iteration, two-point
response, rescaling and stability guards. It must not presuppose **model
content**: lattice geometry, unit cells, what an index means (site, orbital,
spin, sublattice), which bonds exist, or where anything sits in space. That
content is assembled by the user (with whatever model-building package they
prefer) and passed in as plain data.

**The SSH example (canonical ambiguity).** The same SSH chain can be
represented as `2N` lattice sites at positions `0, 1, 2, ...` or as `N` cells
with 2 orbitals each — with the orbitals placed at their physical split
positions or co-located at the cell center. All are legitimate models of the
same Hamiltonian matrix, but they have **different current operators**: the
bond current is `J_ij = h_ij (r_i - r_j)_dir`, so an intra-cell hop carries
current in the split-position embedding and none in the co-located one. The
f-sum rule, the diamagnetic operator, and the stiffness differ accordingly.
Only the user knows which embedding is physical. Therefore:

> Every place the algorithm needs geometry, it takes the user's displacement
> data (`pos`, `disp(i, j)`) verbatim and never infers positions, bonds, or
> internal structure from the matrix. Every place it needs degeneracy or
> normalization content (`g_rho`, `g_J`, `volume`, channel weights), the user
> supplies a number or a small table — plain data, no ontology to learn.

This principle already holds for `h` (duck-typed), `pos`/`disp`, `g_rho`,
`g_J`, and `volume`. The generalization below extends it to the pairing
structure, which is currently hard-wired to onsite-diagonal.

**What the user supplies** (the full contract, kept deliberately plain):

| Input | Form | Used for |
| --- | --- | --- |
| `h` | matrix or duck-typed operator (`size`, 3-/5-arg `mul!`) | normal-state kinetics |
| `h_hole` | conjugated operator (matrix-free only; assembled `h` is conjugated automatically) | hole block |
| pairing structure | `D` operator, or a channel table: bond list `(i, j)` + form-factor weights + coupling `V` + parity | pairing block and gap equation |
| `pos`, `disp(i, j)` | `N x ndim` matrix + displacement callable (minimum image on tori) | current & diamagnetic vertices, Peierls phases |
| `g_rho`, `g_J`, `volume` | numbers | degeneracy and normalization routing |
| `U` / `V`, `mu`, `beta`, target filling | numbers/vectors | interaction channels, thermodynamics |

## 2. Where the current implementation is model-restrictive

Findings from the 2026-07 review of the PR-#4 code:

- **R1 (the real restriction): onsite diagonal pairing.** `D = Diagonal(Δ)`
  is hard-wired in three places: the `mul!` broadcast (`B.Δ .* Xh`), the
  anomalous-moment extraction (reads only the diagonal Nambu entry
  `⟨i,p|T_m|i,h⟩`), and the gap equation `Δ_i = -U_i⟨c_↓c_↑⟩` in
  `bdg_update`. No bond pairing (p-, d-, extended-s wave), no orbital-matrix
  local pairing.
- **R2: misleading naming, not a real restriction.** The `:singlet` hole
  block `-conj(ξ)` is the fully general BdG hole block for *any* normal
  operator, including explicit-spin / spin-orbit-coupled `h` (where singlet
  vs triplet lives in the structure of `D`, not in the hole block). Rename to
  `:conjugate` (keep `:singlet` as an alias). `:intervalley` remains the
  genuinely special, gated option. With explicit-spin `h`, the density
  degeneracy is `g_rho = 1` — already a kwarg, needs documenting.
- **R3: particle-hole symmetry is parity-resolved, not architectural.** With
  the conjugate hole block and pairing matrix `D`: `τ_y K` is an exact PH
  symmetry iff `Dᵀ = +D` (even parity: onsite/extended-s/d-wave); `τ_x K`
  iff `Dᵀ = -D` (odd parity: p-wave/triplet-reduced). Either way the
  spectrum has exact ±E pairs. Mixed-parity `D` has no exact PH symmetry —
  and nothing breaks: `b = 0` is a radial bound, protected by the runtime
  guards and the Gershgorin option; only Chebyshev resolution is paid.
  `moment_parity=:EVEN` eligibility inherits the same parity conditions.
- **R4: Hartree channel.** The scalar `-(U/2)n` diagonal is minimal but the
  `n` vector is already an arbitrary site potential. Fock exchange and
  inter-site Hartree are out of scope here (separate plan if needed).
- **R5: stiffness with bond pairing needs one physics decision.** A bond
  pair field couples to the vector potential with charge 2e. The standard
  mean-field Kubo treatment (Scalapino–White–Zhang) holds `Δ_ij` rigid and
  uses kinetic-only current vertices; the self-consistently re-gauged phase
  response is a different (harder) calculation. Ship the **rigid-Δ**
  convention, documented; the free-energy-curvature anchor tests exactly the
  declared convention because the test controls `H(A)`.

Not requiring generalization: `two_energy_response` (vertex-agnostic),
`kpm_2d`, `ScaledOperator`/`rescale`/stability guards, the Anderson driver,
checkpointing.

## 3. Design

### 3.1 Pairing block: `D` as an operator

`BdGOperator` stores `D` (and `D'` for the lower block) instead of the `Δ`
vector; `mul!` uses sparse/operator products with a `Diagonal` fast path so
the current onsite case keeps its allocation-free broadcasts. The operator
stays Hermitian by construction (`[[ξ, D], [D', -conj(ξ)]]`).
`gershgorin_bound` adds `|D|` row sums. Backward compatibility: the current
`Delta::Vector` keyword keeps working (constructs `Diagonal`).

### 3.2 Channel abstraction (user-facing, plain data)

```julia
PairingChannel(
    bonds    :: Vector{Tuple{Int,Int}},   # (i, j) pairs; (i, i) = onsite
    weights  :: Vector{<:Number},         # form factor per bond (e.g. d-wave ±1)
    V        :: Real or Vector,           # coupling
    parity   :: Symbol,                   # :even (Dᵀ=+D) or :odd (Dᵀ=-D)
)
```

The user writes down bonds and weights in *their* index convention — the
package never derives bonds from geometry. A model with orbitals is simply a
model whose indices the user has already flattened; channels can couple any
index pairs (orbital-resolved local pairing = "bonds" between co-located
indices).

### 3.3 Moments and gap equation

`bdg_site_moments` generalizes to channel moments: one particle-seeded
recurrence per column j, extracting the hole entries `i + N` for every bond
`(i, j)` in the channel — O(z) entry reads per seed per step instead of 1;
the recurrence engine is untouched. Reconstruction reuses the
Jackson/Gauss–Chebyshev path per bond:
`F_ij = ∫ dE f(E) S_ij(E)`, then `Δ_ij = -V_ij · F_ij` (or the
form-factor-projected channel amplitude), symmetrized per the channel parity
before assembling `D`. Bond-counting factors (each bond once vs twice) are
pinned by ED, as everything else has been.

### 3.4 Self-consistency

The SCF driver packs channel amplitudes instead of the onsite `Δ` vector; the
Anderson machinery, residual bookkeeping, and checkpoints are agnostic to
what is packed. Checkpoint format gains the channel amplitudes (version
bump).

### 3.5 Vertices, diamagnetic operator, stiffness

Unchanged in structure: `nambu_current_q` and `nambu_diamagnetic` are built
from `h`, `pos`, `disp` — the user's embedding decides the physics (SSH
example above). Under the rigid-Δ convention the pairing block contributes no
vertex for any channel; this is stated in the docstrings, and the Peierls
test fixtures define `H(A)` accordingly so the finite-difference and
free-energy-curvature anchors remain exact for the declared convention.
Gauge-coupled (2e) pairing response is explicitly out of scope (future plan).

## 4. Minimal test models (gates, in the established style)

1. **Kitaev chain (spinless p-wave), 4-site ring, BdG dim 8** — the primary
   gate; not representable at all today, so it tests the generalization
   rather than regressions:
   - assembled vs matrix-free action with antisymmetric bond `D` (1e-12);
   - exact ±E pairing via `τ_x K` (odd parity); a deliberate mixed-parity
     (onsite + bond) case documents the loss of exact pairing;
   - dispersion vs `E_k = sqrt((2t cos k + μ)² + 4Δ_p² sin²k)`;
   - **two-site "one-bond" analytic fixed point** (closed-form tanh analog of
     the one-site Eq.-26 gate; derive at implementation time);
   - bond-channel moments vs exact eigenvector sums `F_{i,i+1}` on every bond;
   - SCF with nearest-neighbor attraction: uniform bond gap vs ED fixed
     point, U(1) covariance;
   - open chain: two near-zero Majorana modes for `|μ| < 2t`, absent for
     `|μ| > 2t` (free topological sanity check of the whole stack).
2. **d-wave on the existing 3×3 square** (`Dᵀ = +D`, weights ±1 on x/y
   bonds): channel projection, `τ_y K` branch, and the rigid-Δ stiffness with
   zero-gap cancellation + curvature anchor carried over verbatim (2-leg
   ladder or small 2D lattice for a genuine transverse q).
3. **Rashba dimer with explicit-spin `h` and onsite singlet `D = iσ_y Δ`**
   (antisymmetric in the combined site⊗spin index): stresses the parity
   bookkeeping and the `g_rho = 1` explicit-spin convention.
4. **SSH representation-invariance test** (enforces §1 directly): the same
   SSH-BdG matrix built (a) as 2N sites with split positions and (b) as N
   cells × 2 co-located orbitals must give identical spectra, DOS, and local
   observables; the current operators must *differ* exactly as the embeddings
   dictate (intra-cell bonds carry current only in (a)), and with identical
   position data the two representations must give identical stiffness.

## 5. Staging

1. `D`-operator generalization + rename `:singlet → :conjugate` (alias kept)
   + Kitaev action/PH/dispersion gates.
2. Channel abstraction + bond moments + gap equation + Kitaev SCF and
   two-site analytic gates.
3. d-wave + Rashba + SSH-invariance fixtures; parity-resolved PH docs.
4. Rigid-Δ stiffness for bond channels + anchors; docs conventions table.

Estimated effort: 2–4 focused days in the established gate-test workflow.
Out of scope: Fock/off-site Hartree channels, gauge-coupled pairing response,
GPU (still pending hardware).
