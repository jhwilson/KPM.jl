```@meta
CurrentModule = KPM
```

# Development roadmap

This page records the next substantial application areas under consideration.
It is a design roadmap, not a promise that the sketched names or signatures are
already public API. The ordering below separates **implementation difficulty**
from **recommended implementation order**: a feature can be easy in isolation
but be more useful after a shared lower-level primitive exists.

## Difficulty ranking

Difficulty is relative to the present KPM.jl architecture, including its typed
rescaling front end, two-slot Chebyshev recurrence, CPU/CUDA device helpers, and
BdG self-consistency machinery.

| Rank (easiest first) | Feature | Difficulty | What the package can reuse | Main new work |
| --- | --- | --- | --- | --- |
| 1 | Unitary evolution | Low | Rescaling, two-slot recurrence, block-vector and device paths | Complex Bessel coefficients, adaptive truncation, time-grid API |
| 2 | Spectral function, LDOS, and full real-space Green function | Low--medium | `ldos_mu`, `dos`, kernels, typed moment metadata | Complex off-diagonal moments and a causal reconstruction returning both real and imaginary parts |
| 3 | Local Chern markers | Medium | Matrix-function recurrence, stochastic probes, existing Haldane/quantized-Hall tests | Fermi-projector action, explicit geometry/region normalization, boundary-safe marker evaluation |
| 4 | Eigenstate filtering (filter-and-shake) | High | DOS estimates, rescaling, block recurrences, `Arpack` for comparisons | Band-pass filters, rank-revealing orthogonalization, Rayleigh--Ritz/locking, completeness diagnostics |
| 5 | Real-space Green-function Hartree--Fock | Very high | Local/bond moment extraction, SCF mixing and checkpoints from BdG | General interaction callback, selected density-matrix elements, filling control, nonlocal exchange, noisy nonlinear convergence |

The recommended delivery order is **spectral/Green-function foundation →
unitary evolution → Chern markers → eigenstate filtering → Hartree--Fock**.
The first item provides matrix-element and matrix-function machinery used by the
later topology and mean-field work; unitary evolution is then a small, contained
validation of the same matrix-function action.

## Shared prerequisite: a general matrix-function action

Most of this roadmap needs the same operation,

```math
f(H)|V\rangle \simeq \sum_{n=0}^{N_C-1} c_n T_n(H_{\mathrm{norm}})|V\rangle,
```

for one vector or a block of vectors. The current recurrence already computes
the Chebyshev vectors efficiently on CPU and CUDA. The missing reusable layer
should:

- accumulate a vector result with real or complex coefficients without storing
  every Chebyshev vector;
- optionally extract complex matrix elements for independent left and right
  probes (the different-left/right `kpm_1d!` methods are now implemented for
  the 1D path — see the spectral-function milestone status — but the
  coefficient-accumulating action itself is still open);
- batch deterministic basis probes and arbitrary user-supplied probes;
- follow the operator's device residence and retain recurrence stability checks;
- carry `(a, b, NC)` provenance in typed value objects.

This should be an internal primitive first. Public APIs should describe the
physical operation (Green function, projector, evolution, filter), not expose a
second competing KPM core.

## 1. Topological properties from local Chern markers

### Target and scope

The first target is the two-dimensional, independent-particle local Chern
marker. For an occupied-state projector ``P`` and ``Q=I-P``, the Bianco--Resta
marker is a diagonal matrix element of a product such as
``P X Q Y P`` (equivalent commutator forms differ by sign convention). Its bulk
or cell average gives the Chern number, while its spatial variation diagnoses
disorder, interfaces, and boundaries without Bloch momentum.

This fits the package's real-space focus, but geometry remains user data. The
caller must supply coordinates and the grouping/area used to turn orbital
markers into cell or regional markers. The implementation must not infer unit
cells, orbital multiplicities, spatial dimension, or boundary geometry from
`H`.

### Rough implementation

1. Add a zero- or finite-temperature Fermi-projector action
   ``P|v\rangle \simeq f_\beta(H-E_F)|v\rangle`` using the shared
   matrix-function recurrence. A sharp zero-temperature step is appropriate in
   a resolved spectral gap; finite temperature or explicit smoothing should be
   available near a transition.
2. Represent ``X`` and ``Y`` as multiplication by caller-supplied coordinate
   vectors. Apply the projector sequence to batches of site basis vectors for
   site-resolved output. For a regional average, also support stochastic trace
   probes restricted to the requested region so that every site need not be
   seeded.
3. Return raw orbital markers separately from caller-defined cell/region
   averages. Require an explicit `cell_area`/`region_area` (or explicit weights)
   for dimensionless normalization.
4. Default the first implementation to open boundaries. A naive diagonal
   position operator is not valid on a torus; periodic-coordinate formulations
   should be a separate, explicitly tested extension.
5. Document that the marker summed over an entire finite open sample is zero:
   topology is obtained from a bulk/cell average with the boundary excluded,
   not from the full finite trace.

### Validation and risks

- Use the existing Haldane exact-diagonalization fixture and conductivity sign
  anchor: the bulk marker must converge to the same signed Chern number as
  `kubo_bastin_cond`.
- Test a trivial phase, a topological phase, disorder, and a two-region
  heterojunction; compare small systems with an exact projector.
- Converge independently in system size, distance from the boundary, `NC`, and
  any projector temperature/broadening.
- The cost of a full site map is many projector applications. Batching makes
  sparse matrix--block-vector products efficient but does not remove that
  scaling; regional stochastic estimates should therefore be a first-class
  mode rather than an afterthought.

Difficulty is **medium**: the recurrence is present, while the load-bearing
work is projector accuracy, geometry normalization, boundary semantics, and a
scalable choice between deterministic local markers and stochastic averages.

References: [Bianco and Resta, *Mapping topological order in coordinate
space*](https://doi.org/10.1103/PhysRevB.84.241106); [Varjas *et al.*,
KPM-localized projectors and stochastic traces for disordered topological
invariants](https://doi.org/10.1103/PhysRevResearch.2.013229).

## 2. Hartree--Fock from the real-space Green function

### Target and scope

The feasible package-level target is **lattice Hartree--Fock for a
caller-specified interaction decoupling**, not an ab-initio quantum-chemistry
Hartree--Fock engine. Given the one-particle density matrix

```math
\rho_{ij}=\langle c_j^\dagger c_i\rangle
=-\frac{1}{\pi}\int dE\,f_\beta(E-\mu)\,\mathrm{Im}\,G^R_{ij}(E),
```

the caller supplies a function that builds the Hartree and Fock contributions
on the interaction support. KPM.jl must not guess spin, orbital, Coulomb
matrix elements, background charge, double-counting terms, or which exchange
matrix elements are legal.

Although the real-space Green function gives the public and conceptual route,
production SCF updates should usually integrate the Chebyshev expansion of the
Fermi operator directly. This is algebraically the same density matrix and
avoids reconstructing and numerically integrating a dense energy grid on every
iteration.

### Rough implementation

1. Build on the complex real-space Green-function/matrix-element moments in the
   spectral-function milestone. Add a batched extractor for a caller-supplied
   list of density-matrix entries `(i, j)`. The gather/scatter design in
   `bdg_channel_moments` is a useful model.
2. Define a typed interaction/mean-field boundary such as a callback
   `build_mean_field(rho, model_data)` returning the effective one-body operator
   and any double-counting energy. Provide small convenience constructors only
   for explicitly defined models (for example, onsite Hubbard Hartree and
   specified finite-range density--density Hartree--Fock).
3. Add an SCF driver that rebuilds and rescales the effective Hamiltonian,
   recomputes the requested density-matrix elements, mixes fields, and tests
   convergence in both fields and density. Reuse the linear/Anderson mixing,
   recurrence guards, and checkpoint patterns from `bdg_solve!` after factoring
   them into model-independent helpers.
4. Support both fixed chemical potential and fixed filling. Fixed filling needs
   a safeguarded scalar solve for `mu` using the KPM particle number at every
   SCF step. Temperature, degeneracy, and background charge must be explicit.
5. Keep the set of random probes fixed across iterations whenever stochastic
   density estimation is used; changing them turns sampling noise into a moving
   SCF target. Deterministic selected-entry evaluation should be the baseline
   for short-range lattice interactions.
6. Never materialize the full ``N\times N`` density matrix by default. Exchange
   cost is controlled by the caller's interaction support or an explicit
   locality threshold; truly long-range exchange requires a separate scaling
   strategy.

### Validation and risks

- Start with a small spin-resolved Hubbard or finite-range density--density
  model and compare the complete self-consistent density matrix, energy, and
  fields with exact diagonalization of each mean-field iterate.
- Test symmetry-broken and symmetry-preserving seeds, fixed-`mu` and
  fixed-filling ensembles, free-energy/double-counting conventions, and restart
  reproducibility.
- Separate KPM truncation, quadrature, stochastic, and SCF errors in the
  convergence tests.
- A changing mean-field Hamiltonian can change both spectral center and width;
  every iteration needs a safe rescaling strategy. Metals and low temperature
  also demand much larger polynomial order for the sharp Fermi operator.

Difficulty is **very high**: this is a new general nonlinear solver with
model-dependent nonlocal state, not merely another reconstruction of existing
moments. The existing BdG SCF is valuable infrastructure but its present
Hartree and pairing semantics are intentionally specialized and should not be
silently generalized.

References: [Baer, Neuhauser, and Rabani, stochastic density evaluation without
occupied-orbital diagonalization](https://doi.org/10.1103/PhysRevLett.111.106402);
[Cytter *et al.*, finite-temperature stochastic Fermi-operator
evaluation](https://arxiv.org/abs/1801.02163); [Tang *et al.*, self-consistent
random real-space Green functions and density matrices](https://arxiv.org/abs/2311.18161).

## 3. Spectral function, LDOS, and the complete Green function

### Current status

**Implemented** (the first delivered milestone of this roadmap). The general
1D path computes complex moments for independent left and right probe blocks
(`kpm_1d!` with `psi_in_l`/`psi_in_r`; full ``N_C``-step recurrence, since
moment doubling folds ``T_m T_n`` products of one and the same ket — twice
the matrix--vector cost of the equal-vector path, which retains doubling).
The typed layer provides `GreenMoments` with `green_moments` (independent
pairs or equal-probe diagonal blocks) and `ldos_moments` (batched unit-site
seeds), and reconstruction via `greens`, `ldos`, `spectral_function`, and the
exact `spectral_weights` sum rule. The legacy `ldos_mu(H, NC, site)` raw path
remains as a thin, metadata-free alternative.

### Rough implementation

1. Add typed diagonal and general matrix-element moments carrying `(a, b, NC)`.
   The general recurrence must keep complex moments and accept arbitrary
   caller-supplied bra/ket probe blocks. Retain the moment-doubling optimization
   only for eligible equal-vector diagonal cases.
2. Add a reconstruction for the retarded/advanced resolvent

   ```math
   G^{R/A}_{uv}(E)=\langle u|(E-H\pm i\eta)^{-1}|v\rangle
   ```

   that returns the **full complex value**. Define spectral functions as
   ``A_{uv}(E)=\tfrac{i}{2\pi}\left(G^R-G^A\right)_{uv}`` — the matrix element
   of ``\delta(E-H)``, complex for independent bra/ket pairs — rather than as
   a separate moment engine. Only the diagonal reduces to
   ``A_{uu}=-\mathrm{Im}\,G^R_{uu}/\pi \ge 0``, which is the LDOS.
3. Offer diagonal `ldos`, general `greens`, and `spectral_function` APIs
   (the delivered names; a separate projected variant proved unnecessary —
   arbitrary probe pairs already cover projected spectral functions).
   Momentum and orbital content remain user
   data: an ``A(k,E)`` calculation supplies its own Fourier/orbital probe matrix
   instead of asking KPM.jl to infer positions, reciprocal vectors, or bands.
4. Support batches of sites, bonds, or arbitrary probes and energy grids;
   provide sum-rule helpers for integrated spectral weight.

### Lorentz kernel and the real part

The Lorentz kernel is the right **kernel-route choice when finite-order
damping is intended to represent Lorentzian lifetime broadening** (`greens`
has no default route: exactly one of `kernel` or `eta` must be chosen). With
order `NC` and parameter ``\lambda``, the damping is uniform in
``\tilde\theta = \arccos\tilde x``, so the equivalent physical broadening is
position dependent, ``\eta(E)\approx (a\lambda/N_C)\sqrt{1-\tilde x^2}`` (of
order ``a\lambda/N_C`` at the band center); the correspondence requires the
kernel's reflected ``e^{-2\lambda}`` term to be negligible, i.e.
``\lambda\gtrsim 2``, with few-percent accuracy only by ``\lambda\approx 4``. The real and imaginary parts must be
reconstructed as one retarded/advanced analytic object; independently damping
only the spectral part would not define a controlled causal Green function.

The Kramers--Kronig transform that supplies the real part is **analytic in
Chebyshev space** — no numerical Hilbert transform is involved. Both routes
below are one closed-form expression for the resolvent
[Pixley *et al.*, PRB **95**, 235101 (2017), Eq. (14)],

```math
G^{R/A}(E) = \frac{-s\,i}{a\,\sin\tilde\theta}\Big[\mu_0 g_0
+ 2\sum_{n\ge 1}\mu_n g_n e^{-i s n\tilde\theta}\Big],\qquad
\tilde\theta=\arccos\frac{E-b+is\eta}{a},\quad s=\pm 1,
```

differing only in ``(g_n, \operatorname{Im}\tilde z)``: real and imaginary
parts of ``G`` come from the *same* moments. The principal branch of the
complex ``\arccos`` places the series on the decaying side for either sign of
``s``, and real energies outside the band continue to the correct purely real
``G``. The two deliberate choices exposed by `greens` are:

- a Lorentz-kernel KPM route (``g_n`` from `LorentzKernels`, real energies),
  convenient when `NC` and ``\lambda`` define the desired Lorentzian
  resolution; and
- a direct CPGF route (``g_n = 1``, complex energy ``E + is\eta``),
  preferable when the caller specifies a physical broadening ``\eta`` and
  needs the most direct causal resolvent — exact up to the series tail,
  ``\sim e^{-N_C\eta/(a\sqrt{1-\tilde x^2})}`` strictly in band (for
  ``\eta/a \ll 1-\tilde x^2``), softening to ``\sim e^{-N_C\sqrt{2\eta/a}}``
  at the band edges; the implementation's tail warning uses the exact
  complex ``\arccos`` and covers all regimes.

Jackson remains useful for positive spectral densities with near-Gaussian
resolution, but it should not be presented as equivalent to a constant
imaginary self-energy. Whichever route is selected, `real(G)`, `imag(G)`, and
``-\mathrm{Im}(G)/\pi`` come from the same convention and broadening by
construction.

### Validation and risks

- Compare diagonal and off-diagonal Green functions with exact resolvents for a
  one-level system, a short chain, and a small complex Hermitian model.
- Test retarded/advanced conjugacy, Hermitian matrix-element symmetry,
  Kramers--Kronig consistency, positivity of diagonal spectral functions, and
  the integrated-weight sum rule.
- Verify that summing LDOS over sites reproduces the existing DOS convention
  after accounting for its per-state normalization.
- Test the physical `a` and `b` mapping explicitly; Green functions acquire a
  `1/a` factor, while ``\eta`` must be converted to rescaled units.

Difficulty is **low--medium**: diagonal moments and spectral reconstruction are
already close, but a correct full complex resolvent requires removing the
current real/equal-vector assumptions and pinning analytic conventions.

References: [Weiße *et al.*, KPM review, including the Lorentz
kernel](https://doi.org/10.1103/RevModPhys.78.275); [Pixley *et al.*,
momentum-resolved ``G(\mathbf{k},\omega)`` from KPM moments with the
closed-form causal reconstruction](https://doi.org/10.1103/PhysRevB.95.235101);
[João and Viana Parente
Lopes, review of KPM and direct CPGF reconstruction](https://doi.org/10.1016/j.physrep.2020.10.001);
[KITE implementation paper and comparison of Lorentz damping with exact CPGF
coefficients](https://doi.org/10.1098/rsos.191809).

## 4. Unitary evolution with Chebyshev methods

### Target and scope

For a time-independent Hermitian Hamiltonian already rescaled as
``H=aH_{\mathrm{norm}}+bI``, use

```math
e^{-iHt}|\psi_0\rangle = e^{-ibt}\left[
J_0(at)T_0(H_{\mathrm{norm}})
+2\sum_{n=1}^{N_C-1}(-i)^nJ_n(at)T_n(H_{\mathrm{norm}})
\right]|\psi_0\rangle.
```

The operation is unitary in the converged series and needs no Jackson or
Lorentz kernel. This first milestone should not claim support for a general
time-dependent ``H(t)``; that needs time ordering and a separate splitting or
Magnus-type design.

### Rough implementation

1. Reuse the shared matrix-function action with Bessel coefficients and the
   existing two-slot recurrence. Add a direct dependency that supplies stable
   Bessel functions rather than relying on a transitive package dependency.
2. Provide allocating and in-place APIs for one state or a block of states,
   accepting either a `RescaledHamiltonian` or explicit safe `(a, b)` metadata.
3. Choose `NC` adaptively from the tail of ``J_n(|at|)`` when the caller does
   not provide it. Report or expose a truncation estimate, and reject propagation
   if the rescaling stability guard fails.
4. For many requested times, choose explicitly between one recurrence with
   multiple accumulators (low matvec count, more output memory) and independent
   time calls (low working memory). Do not store all Chebyshev vectors by
   default.
5. Keep observables outside the propagator: callers may request states at times
   or supply a callback/reduction so large trajectories need not be retained.

### Validation and risks

- Compare with dense `exp(-im * H * t)` for real and complex Hermitian test
  matrices, including nonzero center shift `b`.
- Test norm conservation, reversibility, the group property, block states,
  `t=0`, large ``|at|``, and CPU/GPU agreement.
- Kernel damping must never be silently applied to unitary evolution; it would
  cause norm loss.

Difficulty is **low**: the sparse matrix--vector recurrence and device behavior
are already the expensive part. The new concerns are coefficient generation,
adaptive order, and a memory-safe multi-time interface.

References: [Tal-Ezer and Kosloff, original Chebyshev propagation
scheme](https://doi.org/10.1063/1.448136); [Fehske *et al.*, numerical
time-evolution review and benchmarks](https://arxiv.org/abs/0907.3022).

## 5. KPM eigenstate filtering (filter-and-shake)

### Target and terminology

The target is a set of eigenpairs in a narrow **interior** energy window without
shift-and-invert factorization. “Filter-and-shake” is used somewhat informally
in the physics literature; the stable numerical baseline is Chebyshev filter
diagonalization: approximate a window projector by a polynomial, apply it to an
overcomplete block of search vectors, orthogonalize, diagonalize the projected
Hamiltonian, and repeat while refreshing directions that have collapsed or are
missing. The precise “shake” policy should be specified in a short design note
before becoming public API rather than encoded as an unexplained heuristic.

### Rough implementation

1. Use an existing DOS calculation to estimate the number ``N_T`` of states in
   the target interval, then allocate an overcomplete search block. Published
   ChebFD guidance uses roughly ``2N_T``--``4N_T`` search vectors when memory
   permits.
2. Construct a Chebyshev approximation to a band-pass/window function. Filter
   design and damping must be independent of the DOS Jackson default; Lanczos
   damping is a strong initial choice for filter diagonalization because it
   suppresses leakage without broadening the target window as strongly as
   Jackson.
3. Apply the filter to the entire block with the shared recurrence, then use a
   **rank-revealing** QR/SVD-style orthogonalization. The package's current
   classical `gram_schmidt!` is not sufficient for a nearly rank-deficient
   filtered block.
4. Form the small projected Hamiltonian, solve the Rayleigh--Ritz problem,
   compute full-space residual norms, lock converged in-window Ritz pairs, and
   discard large-residual “ghosts”.
5. Shake/reseed unconverged directions using new random components projected
   away from locked vectors, or use several Chebyshev-evolved versions of the
   filtered seeds. Refilter until all in-window residuals meet tolerance and the
   state count agrees with the DOS estimate within its uncertainty.
6. Return energies, eigenvectors, residuals, target/search intervals, and a
   completeness diagnostic. Keep filtered block vectors on the active device;
   move only small Gram/projected matrices to the host if needed.

### Validation and risks

- Compare with dense diagonalization and `Arpack` on small sparse matrices;
  cover clustered/degenerate states, complex Hermitian matrices, states on a
  window edge, empty windows, and deliberately underestimated state counts.
- Verify both residual accuracy and completeness. Small residuals alone do not
  prove that every eigenpair in the interval was found.
- Orthogonalization and block storage, not sparse matvecs, become the scaling
  limit when the target window contains many states. GPU support therefore
  requires an explicit dense-linear-algebra and host-transfer audit.

Difficulty is **high**: polynomial application is already available in rough
form, but a reliable interior eigensolver also needs robust subspace numerical
linear algebra, degeneracy handling, locking, state-count estimates, and
failure diagnostics.

References: [Pieper *et al.*, Chebyshev filter diagonalization and parameter
selection](https://doi.org/10.1016/j.jcp.2016.08.027); [Guan and Zhang,
filter/evolution/subspace diagonalization for many interior
states](https://doi.org/10.21468/SciPostPhys.11.5.103); [Zhang, Wilson, and
Foster, a package-relevant use of Chebyshev filtering/shaking for narrow-band
wave functions](https://doi.org/10.1103/PhysRevB.111.024207).

## Proposed milestones

Each milestone should be a separately reviewable change with exact-diagonalization
tests before a higher-level solver depends on it.

1. **Matrix elements and matrix-function action:** complex independent
   bra/ket moments, batched coefficient accumulation, typed provenance, and
   CPU/GPU tests. *Moments and typed provenance are delivered (see the
   spectral-function milestone status above); the coefficient-accumulating
   matrix-function action `f(H)|V⟩` remains open for the projector and
   evolution milestones.*
2. **Spectral and propagation APIs:** complete LDOS/full complex Green function
   (Lorentz and direct-CPGF choices) — *delivered* — followed by
   time-independent unitary evolution.
3. **Fermi projector and Chern marker:** deterministic local maps plus stochastic
   regional averages, pinned to the existing Haldane sign convention.
4. **Filtered interior eigensolver:** ChebFD baseline, then a documented
   filter-and-shake refresh strategy.
5. **General lattice Hartree--Fock:** selected density-matrix elements,
   interaction callback, fixed-`mu`/fixed-filling SCF, mixing, checkpoints, and
   explicit energy conventions.

