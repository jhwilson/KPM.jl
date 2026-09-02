# KPM compute-core benchmarks

Run the current working tree with a stable eight-thread setting:

```bash
julia -t 8 --project=benchmarks benchmarks/core_bench.jl work
```

`core_bench.jl` develops the adjacent package path before loading KPM. This is
intentional: Julia 1.10 does not support the `[sources]` project-table needed
to express the local path declaratively. Results are printed as Markdown and
saved to `benchmarks/results/<label>.json`; omit the label to use the current
short git revision.

Compare the working tree with a reference revision (default: `main`):

```bash
benchmarks/compare.sh main
```

The comparison script creates a detached temporary worktree, seeds it with
the current harness (so it can compare revisions that predate `benchmarks/`),
benchmarks it with `julia -t 8`, benchmarks the current tree with the same
command, then prints `reference/work` for each case and removes the temporary
worktree.

The suite measures only prebuilt, deterministic workloads. All probe vectors
come from `Xoshiro(1234)`, and Hamiltonian construction and rescaling happen
outside the timed regions.

| Case | What it measures |
| --- | --- |
| `kpm_1d_dos_NH65536_NC1024_NR8` | DOS moments on a 256x256 periodic square lattice. |
| `kpm_1d_dos_NH262144_NC512_NR4` | Larger DOS-moment workload on a 512x512 lattice. |
| `kpm_2d_cond_NH16384_NC64_NR4_*` | Conductivity moments on a 128x128 lattice, using minimum-image bond currents `Jα[i,j] = H[i,j](rᵢ-rⱼ)α`; runs package-default, `arr_size=3`, and (when accepted) `arr_size=64`. This is shrunk from the requested `NC=256` via `NC=128` because both full sampled configurations exceeded five minutes on the baseline laptop. |
| `chebyshev_action_NH65536_NC1024_NR8_K2` | Two Jackson-damped Fermi-step polynomial actions sharing one recurrence. |
| `chebyshev_iter_single_NH262144_NR8` | One raw, preallocated Chebyshev recurrence step. |

Each reported time is BenchmarkTools' minimum of at most three one-evaluation
samples, with a 60-second sampling cap per case. Values are machine-specific:
use a consistent laptop setup to detect local regressions, and run the same
suite on the cluster for a pull-request performance table. If a case exceeds
five minutes on the target machine, halve `NC` or `NH` and update both this
table and the workload metadata stored in its JSON result. The present
baseline made this adjustment only for the 2D cases: `NC=256` to `NC=64` (two
successive halvings); every other workload retains its requested size.
