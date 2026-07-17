# Adversarial review: `feature/bdg-stiffness...feature/bdg-gpu`

Reviewed commits:

- `7cf33d5 GPU support for BdG SCF and superfluid stiffness`
- `e65934a Fix CUDA 5.11 dispatch: CuSparseMatrixCSR left AbstractCuSparseMatrix`

I also ran the affected CPU tests (`bdg_test.jl`, `pairing_channel_test.jl`, and
`stiffness_test.jl`) and the focused oracle in `adversarial/analyze_bdg_gpu.jl`.
The ordinary CPU tests pass. The oracle found no BdG block-sign error in the
five matrix combinations it exercised (maximum action discrepancy
`2.01e-16`), and the new gather implementation was bitwise equal to the old
CPU loop for the tested shuffled sites, duplicate directed bonds, onsite
bonds, partial batches, and `NC in (2, 3, 17)`. Those positive checks do not
remove the release blockers below.

## Criticisms

1. **[blocker] The documented matrix-free CPU fallback is broken in both stiffness paths because they return to global-device KPM routines.**

   Evidence:

   - `ext/KPMCUDAExt.jl:45-47` deliberately leaves a non-assemblable operator
     on the host:

     ```julia
     function KPM.to_device(dev::CUDADevice, op::KPM.BdGOperator, expect_eltype)
         KPM._bdg_assemblable(op) || return op
         return KPM.to_device(dev, KPM.bdg_assemble(op), expect_eltype)
     end
     ```

   - `src/applications/stiffness.jl:421-422` can therefore construct
     `Hs_sc = ScaledOperator(host_BdGOperator, ...)`, but
     `src/applications/stiffness.jl:442` allocates the `kpm_2d!` workspace
     through `_kpm2d_workspace`, whose entries at lines 322-328 all use
     `maybe_on_device_zeros`, i.e. the global CUDA flag.
   - `src/moment.jl:663-674` then globally uploads the probes and vertices.
     `maybe_to_device(Hs_sc)` still contains the host matrix-free BdG operator,
     while `Jalpha`, `Jbeta`, and all workspaces are device arrays. The first
     `mul!` is consequently mixed-residence.
   - The same defect exists in `diamagnetic_term`: the host fallback can be
     wrapped at `src/applications/stiffness.jl:284`, while
     `kpm_1d_current!` globally allocates at `src/moment.jl:520-521`, globally
     converts at lines 524-525, and writes a device probe into the workspace at
     line 531.
   - This is not a hypothetical unsupported input. A `D` that only implements
     `size`, `mul!`, and `adjoint` is explicitly supported by `BdGOperator`.
     `adversarial/analyze_bdg_gpu.jl:143-167` constructs such a block and
     obtains a finite stiffness on CPU while `_bdg_assemblable(op) == false`.

   What would settle it: on a CUDA runner, call both `superfluid_stiffness`
   and `diamagnetic_term` (the latter with a supplied `a`) using an assembled
   `h` and matrix-free `D`; assert CPU equality and assert that the complete
   computation stays on the host. The implementation needs residence-keyed
   workspaces/transfers in the current-moment routines, or an explicit scoped
   CPU execution path—not only in `bdg_channel_moments`.

2. **[major] The claimed “full package suite with GPU active” does not run the BdG or stiffness suite on the GPU under normal `Pkg.test`.**

   Evidence:

   - `test/runtests.jl:1` loads `KPM`, but not `CUDA`.
   - All BdG and stiffness tests run at `test/runtests.jl:24-54`.
   - `CUDA` is first loaded by `test/optional_cuda_test.jl:3`, included only at
     `test/runtests.jl:64-65`, after all affected tests have finished. A weak
     dependency does not activate its extension merely because a physical GPU
     is present.
   - `test/gpu_smoke.jl` is manual and is not included by `runtests.jl`.

   Thus a green full suite on an A100 mainly establishes CPU behavior for the
   changed BdG/stiffness tests. It also explains why the new
   `KPM.whichcore()` branches in the ordinary restart tests are not exercised
   by the standard test order.

   What would settle it: create a real GPU test entry point that executes
   `using CUDA` before `using KPM` and before including the affected tests, or
   run/register `gpu_smoke.jl` in GPU CI. Log and assert `KPM.whichcore()` at
   the start of every GPU-required test group.

3. **[major] The GPU smoke fixtures are too symmetric to validate the gather/scatter indexing, and no GPU test compares the raw gathered moments.**

   Evidence:

   - The pairing-channel fixture at `test/gpu_smoke.jl:141-162` is a
     translation-invariant square lattice with identical real weights and
     amplitudes on every x bond. A source/target permutation or column
     permutation can leave its final uniform fixed point unchanged.
   - It only compares converged fields, not `mu_rho` or `mu_F`.
   - The onsite case uses `N=64` and default `batch_size=64`; the channel case
     uses `N=36 < 64`. Neither reaches the width-change branch at
     `src/applications/bdg.jl:571-573`. The timing case uses `N=4096`, exactly
     divisible by 64. Therefore the final-partial-batch device reallocation is
     never tested.
   - No GPU fixture covers shuffled/partial site sets, repeated directed bonds,
     onsite bonds mixed with offsite bonds, several bonds with one source,
     complex nonuniform bond amplitudes, or an odd channel.

   What would settle it: directly compare complete GPU and CPU
   `bdg_channel_moments` arrays for a small nonuniform complex Hamiltonian,
   shuffled sites, `length(sites) % batch_size != 0`, several targets per
   source, duplicate directed requests, and onsite requests. Compare every
   column to an independently assembled dense `T_n(H)` oracle, not only to the
   same algorithm on CPU.

4. **[major] Several advertised GPU-reachable control-flow paths are not tested at all.**

   Evidence:

   - `test/gpu_smoke.jl:119-163` exercises only linear mixing at fixed chemical
     potential. There is no GPU test for `target_filling`, Anderson mixing,
     callbacks, or checkpoint/restart.
   - The only spectral-radius check at lines 114-116 starts from the default
     RNG. It does not exercise a caller-supplied `v0`, returned vector
     residence, the `maybe_to_host(v_power_new)` flow at
     `src/applications/bdg.jl:1038-1040`, or the per-iteration perturbation at
     lines 1031-1033.
   - There is no GPU-active matrix-free fallback test, dense-`h` test,
     `Hermitian`-wrapped-`h` test, or complex `:intervalley` opt-in test.

   What would settle it: add small GPU-vs-CPU trajectory tests (not just fixed
   points) for Anderson and target filling, plus a checkpoint at iteration
   `k` followed by restart and comparison with an uninterrupted run. Assert
   the iteration history, `v_power`, fields, chemical potential, and returned
   iteration count.

5. **[major] The gather rewrite introduces per-moment temporary device arrays and kernel-launch pressure inside the hottest loop.**

   Evidence:

   - `src/applications/bdg.jl:597-600` evaluates
     `slot[rho_idx]` and `slot[bond_idx]` for every moment. Advanced indexing
     materializes gather results before assigning them into `rho_work[:, m]`
     and `F_work[:, m]`.
   - These calls are inside the `m=3:NC` loop at lines 610-612. With two
     gathers and two destination writes, the number of device operations grows
     as `O(NC * number_of_batches)` independently of the sparse SpMMs.
   - The stability guard at lines 614-615 additionally allocates a reduction
     result and synchronizes often enough to inspect it on the host.
   - Even on CPU, the focused `N=512, NC=64, batch=64` check in
     `adversarial/analyze_bdg_gpu.jl:118-141` increased allocations from
     2,964,384 bytes to 5,487,456 bytes (1.85x), although runtime happened to
     be similar for that small case. GPU allocation/launch cost is likely much
     less forgiving.
   - The only benchmark at `test/gpu_smoke.jl:191-209` prints a time; it has no
     performance threshold, allocation measurement, or profile, so it cannot
     prevent a path that is correct but slower than CPU.

   What would settle it: report `CUDA.@profile`, device allocations, kernel
   count, and synchronized wall time for representative `N`, `NC`, and batch
   counts. Replace the per-moment advanced indexing with a fused kernel that
   writes the moment columns directly, or otherwise demonstrate that the
   gathers do not dominate.

6. **[major] `superfluid_stiffness(...; include_diamagnetic=true)` redundantly reassembles/reuploads operators and vertices.**

   Evidence:

   - The SC and normal BdG matrices are assembled and uploaded at
     `src/applications/stiffness.jl:421-422`.
   - `kpm_2d!` globally converts `Jq` and `Jmq` at
     `src/moment.jl:672-674` for the SC calculation, then repeats those
     conversions for the normal calculation at
     `src/applications/stiffness.jl:445-450`.
   - With diamagnetism enabled, lines 469-476 call `diamagnetic_term` twice;
     each call reassembles/reuploads its BdG operator at line 284. The same
     operators already resident for the paramagnetic calculation are not
     reused. The kinetic `Dhat` is also rebuilt and transferred independently
     for the SC and normal references even though it is identical here.

   What would settle it: profile host-to-device bytes and assembly time for one
   complete stiffness call. Introduce an internal API accepting already
   resident `Hs`, vertices, probes, and workspaces, then assert that each
   invariant object is uploaded once.

7. **[major] The SCF implementation pays full sparse construction and upload cost every iteration, and the dense-matrix case is especially dangerous.**

   Evidence:

   - `src/applications/bdg.jl:1037` calls `maybe_to_device(op)` every SCF
     iteration.
   - That invokes `bdg_assemble`, which allocates `onsite`, `xi`, `hole`, the
     converted `D`, and the block concatenation at lines 287-294 before the
     upload in `ext/KPMCUDAExt.jl:47`.
   - `_bdg_assemblable` accepts every `AbstractMatrix` at lines 264-268, but
     `_as_sparse_cplx` at line 271 converts dense `h`/`D` to CSC. A genuinely
     dense model is therefore expanded into a nearly full sparse 2N-by-2N
     matrix, with sparse index overhead, and rebuilt on every iteration. There
     is no dense GPU path and no warning or host fallback.

   This can turn “GPU support” into an O(N^2) memory blow-up or make upload and
   assembly dominate the recurrence. The timing in `gpu_smoke.jl` excludes the
   SCF reassembly/upload loop entirely: it uploads once at line 202 and times
   only `bdg_site_moments` at line 204.

   What would settle it: benchmark full SCF iteration time, including
   assembly/upload, for sparse and dense inputs. Cache the invariant sparsity
   pattern and kinetic values, updating only Hartree/pairing values, or
   explicitly reject/fallback for dense blocks until a dense device path
   exists.

8. **[major] The checkpoint relaxations are not justified by the fixtures and one of them weakens an invariant that should remain exact.**

   Evidence:

   - `test/bdg_test.jl:203-213` runs with solver tolerances `1e-14` but accepts
     checkpoint/restart differences of `1e-10`, four orders of magnitude
     larger, without recording the observed GPU discrepancy or comparing
     history, iteration count, or `v_power`.
   - `test/pairing_channel_test.jl:339-351` also uses `tol_delta=1e-14`, but
     relaxes both `D` and `n` to `1e-10`.
   - That channel fixture passes `update_density=false` at lines 341 and 345.
     Density lives on the host and must not be updated at all; GPU
     nondeterminism cannot justify relaxing `op.n`. The new assertion could
     hide an actual accidental density mutation.
   - Both branches key off global `KPM.whichcore()`, not the actual residence
     of the operator under test. A host-fallback computation would also receive
     the weaker assertion whenever CUDA is globally active.

   What would settle it: retain exact equality for all host-only state and for
   `n` when `update_density=false`. On actual GPUs, log the maximum observed
   restart discrepancy over repeated runs and CUDA versions, then choose a
   tolerance tied to measured floating-point accumulation (or ULPs), solver
   tolerance, and field scale. Compare the full saved/restored state, not only
   final fields.

9. **[major] The stiffness equality tolerance is scaled by the large unsubtracted responses, so it can accept a badly wrong small stiffness.**

   Evidence:

   - `test/gpu_smoke.jl:184-186` defines

     ```julia
     scale = max(abs(res_c.Pi_SC), abs(res_c.Pi_N))
     @test res_g.Ds_over_pi ≈ res_c.Ds_over_pi atol=1e-5 * scale
     ```

   - Stiffness is precisely the cancellation `Re(Pi_N - Pi_SC)`. Near the
     normal state or in other cancellation-heavy fixtures,
     `abs(Ds) << max(abs(Pi_SC), abs(Pi_N))`; this assertion can therefore
     tolerate an error many times larger than the quantity being validated.
     The same loose scale is used for the complete stiffness, without including
     the diamagnetic scale.

   What would settle it: test a nonzero stiffness against an ED/reference value
   with a tolerance on `Ds` itself, plus a GPU zero-gap cancellation case with
   a tight absolute tolerance. Separately bound errors in each large term and
   propagate those measured errors into a justified subtraction tolerance.

10. **[major] The branch claims compatibility with all CUDA.jl 5.x releases, but validation covers one version and does not pin the sparse type/index contracts.**

    Evidence:

    - `Project.toml:28-31` declares `CUDA = "5"`, not `>= 5.11`.
    - The second commit exists because the original dispatch assumption was
      already wrong. `ext/KPMCUDAExt.jl:54-61` now names three specific sparse
      types in a union, but neither the normal test suite nor a version matrix
      checks that those names, subtyping relationships, and methods work across
      the declared range.
    - `ext/KPMCUDAExt.jl:26` uses `CuSparseMatrixCSR{expect_eltype}(x)` without
      asserting the resulting index type. In the locally installed CUDA.jl
      5.9.6 and 5.11.3 sources, that one-parameter constructor defaults to
      `Cint` (`~/.julia/packages/CUDA/FJf6p/lib/cusparse/array.jl:219-221` and
      `~/.julia/packages/CUDA/FQwqy/lib/cusparse/array.jl:233-235`), while
      `bdg_assemble` promises host `Int` indices. Large matrices therefore have
      an undocumented 32-bit device-index ceiling.
    - Neither the concrete CSR identity dispatch nor five-argument ComplexF64
      sparse SpMM is directly tested on the oldest supported CUDA.jl release.

    What would settle it: run GPU CI against the oldest supported CUDA 5.x and
    the latest 5.x, asserting the converted CSR type/index type and exercising
    both three- and five-argument vector and matrix `mul!`. Either document and
    guard the `Cint` limit or use a supported 64-bit sparse-index construction.

11. **[minor] The CPU “bitwise unchanged” and universal assembly claims are not actually pinned by the committed tests.**

    Evidence:

    - `test/pairing_channel_test.jl:309-314` compares
      `bdg_site_moments` with `bdg_channel_moments`, but the former is now only
      a wrapper around the latter (`src/applications/bdg.jl:634-639`). This is
      a comparison of the new implementation with itself, not with the old
      scalar extraction.
    - The CPU reduction used by the stability guard changed from an explicit
      left-to-right sum to `sum(abs2, slot; dims=1)` at
      `src/applications/bdg.jl:502-505`. Even if moments remain identical, a
      near-threshold input can change whether the guard fires.
    - `test/bdg_test.jl:50-69` covers a real sparse intervalley case and one
      sparse conjugate bond case, but not dense `h`, dense `D`, a
      `Hermitian` wrapper, or complex `:intervalley` with
      `assume_intervalley=true`.
    - My independent oracle (`adversarial/analyze_bdg_gpu.jl:74-111`) did find
      assembly agreement to `2.01e-16` and bitwise moment equality for its
      tested edge cases. That is encouraging, but it is not a permanent
      regression gate and it did not run CUSPARSE.

    What would settle it: retain an independent old-style scalar reference in
    tests (or compare to explicitly formed dense Chebyshev polynomials), add
    the missing block combinations, and test stability values just below and
    above the `1.5` threshold.

## Overall verdict

**Not safe to PR.** The matrix-free stiffness/diamagnetic fallback violates the
stated residence contract and should block merging. The current full-suite GPU
claim is also misleading because CUDA is loaded only after the affected test
groups. Before PR, fix the mixed-residence path, make GPU execution an actual
CI/test precondition, add direct adversarial moment and checkpoint tests, and
profile the per-moment gathers plus repeated assembly/upload costs. The CPU
algebra checks are encouraging, but they do not compensate for a broken
documented fallback and insufficient GPU coverage.
