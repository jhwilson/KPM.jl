# Response to critique-bdg-gpu.md

Each criticism judged explicitly; fixes landed on `feature/bdg-gpu` in the
commit following the critique.

1. **Mixed-residence matrix-free fallback in stiffness paths — LEGITIMATE
   (blocker).** Accepted as stated: `kpm_2d!`/`kpm_1d_current!` workspaces
   follow the global device flag, so a host operator entering them under an
   active GPU mixes residences. Fix: `_device_operator(op, caller)` — the
   stiffness paths now throw an `ArgumentError` with guidance when the GPU is
   active but the operator is not device-assemblable, instead of crashing
   with scalar indexing (or worse). The silent host fallback remains only in
   the SCF path, whose workspaces are residence-keyed end to end. Making
   `kpm_2d!` itself residence-keyed is the correct long-term fix but touches
   validated non-BdG code; deferred.

2. **Full suite under `Pkg.test` never ran BdG tests on GPU — LEGITIMATE
   (major).** Correct: CUDA loaded only in `optional_cuda_test.jl`, after all
   BdG tests. The A100 "full suite with GPU" claim was therefore mostly a CPU
   run; the real GPU validation was the smoke stage. Fix: `KPM_TEST_GPU=1`
   hook at the top of `runtests.jl` (loads CUDA first, asserts
   `CUDA.functional()` and `KPM.whichcore()`), exported by the sbatch stage 3.
   The restart/Anderson/target-filling/callback GPU coverage requested in
   criticism 4 now comes for free from the existing test files.

3. **Smoke fixtures too symmetric; raw moments never compared; partial-batch
   branch untested — LEGITIMATE (major).** Fix: new smoke testset comparing
   raw `bdg_channel_moments` outputs GPU vs CPU on a random complex Hermitian
   sparse `h` (nonuniform complex `Δ`, `:conjugate`), 31 shuffled sites with
   `batch_size=16` (exercises the final-partial-batch buffer reallocation),
   onsite bonds, both directed orders, and several targets per source
   (atol 1e-9). The dense `T_n(H)` oracle for the CPU implementation already
   exists in `pairing_channel_test.jl` (bond moments vs exact eigenvector
   sums), which now also runs under GPU via criticism 2's fix.

4. **Untested GPU control flow — LEGITIMATE, resolved via #2.** Anderson,
   target filling, checkpoint/restart, callbacks, and the
   `v_power`→`maybe_to_host` flow are all exercised by the existing
   `bdg_test.jl`/`pairing_channel_test.jl` testsets once the suite actually
   runs with the GPU active. No duplicate smoke tests added.

5. **Per-moment gather/launch overhead — ACKNOWLEDGED, deferred.** The
   gathers are O(B) per moment against O(nnz·B) for the matvec, and the
   measured GPU speedup (×6.2 at a deliberately small 64×64 lattice) shows
   the overhead is not crippling; the 1.85× CPU allocation increase is small
   in absolute terms and vanishes relative to matvec cost at production
   sizes. A fused extraction kernel is a known optimization, not a
   correctness issue. Will profile at production scale before any
   performance-sensitive campaign.

6. **Redundant reassembly/reuploads in `include_diamagnetic` — LEGITIMATE
   (minor in impact).** Fix: `diamagnetic_term` accepts an internal `Hs`
   keyword (validated against the requested scale); `superfluid_stiffness`
   passes its already-resident `Hs_sc`/`Hs_n`, eliminating both re-uploads.
   Per-call vertex re-transfers inside `kpm_2d!` remain (API-internal);
   stiffness is called once per parameter point, so upload cost is not on the
   hot path.

7. **Per-iteration assembly cost and dense-`h` blow-up — LEGITIMATE on the
   dense trap; assembly cost acknowledged.** Fix: the extension only
   assembles to the device when `h isa AbstractSparseMatrix`; dense `h`
   stays on the host (documented). Per-iteration sparse reassembly is O(nnz)
   against an O(NC·N·nnz/B) recurrence per iteration — negligible; caching
   the sparsity pattern and updating values in place is a noted future
   optimization.

8. **Checkpoint relaxations — PARTIALLY LEGITIMATE.** Accepted for `op.n`
   with `update_density=false`: the solver never touches it on any device, so
   bitwise equality is restored unconditionally. For Δ/D under GPU, the
   1e-10 atol bounds run-to-run kernel accumulation drift over the restarted
   iterations (fields are O(1)); a comment records the rationale and the
   intent to tighten once measured GPU discrepancies are available (the
   quantity was unmeasurable before #2's fix because the tests never ran on
   GPU).

9. **Π-scaled stiffness tolerance — LEGITIMATE (test sharpness).** Fix: the
   smoke test now compares `Ds_over_pi` (and `_complete`) directly with
   rtol=1e-4/atol=1e-8; the individual Π and Dia terms keep their rtol=1e-5
   assertions, so the cancellation is bounded on both ends.

10. **CUDA 5.x version-matrix and Int32 index ceiling — PARTIALLY
    LEGITIMATE.** The dispatch union names (`AbstractCuSparseMatrix`,
    concrete CSR/CSC) exist across all of CUDA 5.x, covering both the old and
    new (≥5.11 GPUArrays-rebased) type trees; a multi-version GPU CI matrix
    is not available on this cluster and is noted as a limitation. The
    CUSPARSE `Cint` (32-bit) device-index ceiling is now documented in
    `_device_operator`'s docstring and CLAUDE.md; matrices anywhere near
    `typemax(Int32)` indices exceed A100 memory long before overflow, and the
    constructor raises `InexactError` rather than corrupting on overflow.

11. **Bitwise-unchanged claims not pinned; guard reduction changed; missing
    block combos — PARTIALLY LEGITIMATE.** The missing `bdg_assemble` block
    combinations (dense `h`, `Hermitian`-wrapped sparse `h`, complex
    `:intervalley` with `assume_intervalley=true`) are now tested against the
    matrix-free action. A permanent scalar-extraction reference is REJECTED:
    the moments are already pinned against exact eigenvector-sum oracles in
    `pairing_channel_test.jl`/`bdg_test.jl`, which is the physically
    meaningful regression gate; bitwise-vs-old-code was a one-time transition
    claim (verified independently by the critique's own oracle to be exact).
    The stability-guard reduction-order concern is REJECTED as immaterial:
    the guard detects exponential runaway of an unstable recurrence, where
    norms cross 1.5 by orders of magnitude, not by ulps.
