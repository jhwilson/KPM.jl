#!/usr/bin/env bash
set -euo pipefail

REV="${1:-main}"
ROOT="$(git rev-parse --show-toplevel)"
TMPROOT="$(mktemp -d "${TMPDIR:-/tmp}/kpm-core-bench.XXXXXX")"
REF_TREE="$TMPROOT/reference"
SAFE_REV="${REV//\//-}"
REF_LABEL="ref-$SAFE_REV"

cleanup() {
    git -C "$ROOT" worktree remove --force "$REF_TREE" 2>/dev/null || true
    rmdir "$TMPROOT" 2>/dev/null || true
}
trap cleanup EXIT

git -C "$ROOT" worktree add --detach "$REF_TREE" "$REV"
# The reference can predate this suite.  Seed only the harness files from the
# working tree; `core_bench.jl` still develops the reference worktree's `..`.
mkdir -p "$REF_TREE/benchmarks"
cp "$ROOT/benchmarks/Project.toml" "$ROOT/benchmarks/core_bench.jl" "$REF_TREE/benchmarks/"

(cd "$REF_TREE" && julia -t 8 --project=benchmarks benchmarks/core_bench.jl "$REF_LABEL")
(cd "$ROOT" && julia -t 8 --project=benchmarks benchmarks/core_bench.jl work)

REF_JSON="$REF_TREE/benchmarks/results/$REF_LABEL.json"
WORK_JSON="$ROOT/benchmarks/results/work.json"
CASES=(
  kpm_1d_dos_NH65536_NC1024_NR8
  kpm_1d_dos_NH262144_NC512_NR4
  kpm_2d_cond_NH16384_NC64_NR4_default
  kpm_2d_cond_NH16384_NC64_NR4_arr_size3
  kpm_2d_cond_NH16384_NC64_NR4_arr_size64
  chebyshev_action_NH65536_NC1024_NR8_K2
  chebyshev_iter_single_NH262144_NR8
)

seconds() {
    local file="$1" case_name="$2"
    awk -v case_name="$case_name" '
        $0 ~ "\\\"" case_name "\\\"" {
            sub(/^.*:[[:space:]]*/, "")
            sub(/,.*/, "")
            gsub(/[[:space:]\"]/, "")
            print
            exit
        }
    ' "$file"
}

echo "| case | reference (s) | work (s) | ref/work |"
echo "| --- | ---: | ---: | ---: |"
for case_name in "${CASES[@]}"; do
    ref_seconds="$(seconds "$REF_JSON" "$case_name")"
    work_seconds="$(seconds "$WORK_JSON" "$case_name")"
    if [[ "$ref_seconds" == "n/a" || "$work_seconds" == "n/a" || -z "$ref_seconds" || -z "$work_seconds" ]]; then
        printf '| %s | %s | %s | n/a |\n' "$case_name" "${ref_seconds:-n/a}" "${work_seconds:-n/a}"
    else
        ratio="$(awk -v ref="$ref_seconds" -v work="$work_seconds" 'BEGIN { printf "%.3f", ref / work }')"
        printf '| %s | %.6f | %.6f | %s |\n' "$case_name" "$ref_seconds" "$work_seconds" "$ratio"
    fi
done
