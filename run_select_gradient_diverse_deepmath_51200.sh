#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PRODUCTION_ROOT="/home/mchen/FiRe-OPD"
PYTHON_BIN="${PYTHON_BIN:-/home/mchen/miniconda3/envs/gvendi-opd/bin/python}"
REFERENCE_REPO="${REFERENCE_REPO:-/home/mchen/prismatic-synthesis-reference}"
SELECTION_PROFILE="${SELECTION_PROFILE:-vanilla_51200}"
TARGET_ROWS="${TARGET_ROWS:-51200}"
EXPECTED_SOURCE_ROWS="${EXPECTED_SOURCE_ROWS:-57046}"
EXPECTED_ELIGIBLE_ROWS="${EXPECTED_ELIGIBLE_ROWS:-57045}"
EXPECTED_EXCLUDED_ID="${EXPECTED_EXCLUDED_ID:-deepmath-level6-038794}"
SOURCE_SHA256="${SOURCE_SHA256:-de3350fdd00bc0410550098ea65179e2be873da99e4075f80de575fc17670597}"
PREPARED_JSONL_SHA256="${PREPARED_JSONL_SHA256:-ee8d55c943577888f14052b3953a0a8aa07d26076e50ddffa41a102b60027344}"
PREPARED_MANIFEST_SHA256="${PREPARED_MANIFEST_SHA256:-5add2e965473d3647f2318c73f957434fbe23344516089be9adfe89db9c6a70e}"
ELIGIBILITY_REPORT_SHA256="${ELIGIBILITY_REPORT_SHA256:-cc8d5b888def37761519e15c6ef722e5a22d91ae2342c6d669f008bcff3dcbb6}"
GRADIENT_MANIFEST_SHA256="${GRADIENT_MANIFEST_SHA256:-9a534118a08e736a15e806933d5cf90c2a99822fee28bf18644e5ff344ce3ae2}"
REFERENCE_COMMIT="${REFERENCE_COMMIT:-d9484cd3b5991030b901ac4a3a9e2472dbfac2ad}"
REFERENCE_TREE="${REFERENCE_TREE:-a0079d8c5e15cb18bb4790f99c43cc19bb9ecd50}"
OLD_PARQUET_SHA256="${OLD_PARQUET_SHA256:-caf303c5d151fdaed2e21eebc257917f13660c589906e3da0e569ffdfd59b059}"
OLD_MANIFEST_SHA256="${OLD_MANIFEST_SHA256:-a1a45382ee577e24f9b455386f9f3adcab210ff40a83ea760c2110fb96f8f90a}"
OLD_SELECTED_IDS_SHA256="${OLD_SELECTED_IDS_SHA256:-a78c04cff10c148f45bf9c828e07cc9cf01398b48efeb707253273af359a037e}"
OLD_DIAGNOSTICS_SHA256="${OLD_DIAGNOSTICS_SHA256:-d2105179f2ce5a7c796aef07136bbc1dce79b07b5bc2de4d2319b18b47ea761f}"
PRIMARY_CLUSTER_RATIO="${PRIMARY_CLUSTER_RATIO:-0.10}"
SENSITIVITY_CLUSTER_RATIO="${SENSITIVITY_CLUSTER_RATIO:-0.01}"
CLUSTER_SEEDS="${CLUSTER_SEEDS:-42,43}"
CLUSTER_ITERATIONS="${CLUSTER_ITERATIONS:-20}"
SELECTION_SEED="${SELECTION_SEED:-42}"
GRADIENT_PREFIX="${GRADIENT_PREFIX:-deepmath}"
EXPECTED_NUM_SHARDS="${EXPECTED_NUM_SHARDS:-4}"

SOURCE_PARQUET="${PRODUCTION_ROOT}/data/g-opd/DeepMath-103K/train_filtered_level6.parquet"
OUTPUT_ROOT="${PRODUCTION_ROOT}/data/gradient_diversity"
PREPARED_JSONL="${OUTPUT_ROOT}/deepmath_level6_r1_solution1.jsonl"
PREPARED_MANIFEST="${OUTPUT_ROOT}/deepmath_level6_r1_solution1.manifest.json"
ELIGIBILITY_REPORT="${OUTPUT_ROOT}/deepmath_level6_r1_solution1.eligibility.json"
GRADIENT_DIR="${OUTPUT_ROOT}/gradients/qwen2.5-0.5b-instruct"
GRADIENT_MANIFEST="${GRADIENT_DIR}/gradient.manifest.json"
OLD_PARQUET="${OUTPUT_ROOT}/DeepMath-103K/train_gradient_diverse_12800.parquet"
OLD_SELECTION_DIR="${OUTPUT_ROOT}/selection"
OLD_MANIFEST="${OLD_SELECTION_DIR}/manifest.json"
OLD_SELECTED_IDS="${OLD_SELECTION_DIR}/selected_ids.jsonl"
OLD_DIAGNOSTICS="${OLD_SELECTION_DIR}/diagnostics.json"
SELECTION_DIR="${OUTPUT_ROOT}/selection_51200"
OUTPUT_PARQUET="${OUTPUT_ROOT}/DeepMath-103K/train_gradient_diverse_51200.parquet"
SELECTION_MANIFEST="${SELECTION_DIR}/manifest.json"
SELECTED_IDS="${SELECTION_DIR}/selected_ids.jsonl"
DIAGNOSTICS="${SELECTION_DIR}/diagnostics.json"
MASTER_LOG="${PRODUCTION_ROOT}/logs/gradient_diversity/deepmath_gradient_diverse_51200.log"
MASTER_LOCK="${OUTPUT_ROOT}/.deepmath_gradient_diverse_51200.launch.lock"


die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

require_pinned_value() {
  local name="$1"
  local actual="$2"
  local expected="$3"
  [[ "$actual" == "$expected" ]] || \
    die "${name} must remain pinned to ${expected} (got ${actual})"
}

require_pinned_value SELECTION_PROFILE "$SELECTION_PROFILE" vanilla_51200
require_pinned_value TARGET_ROWS "$TARGET_ROWS" 51200
require_pinned_value EXPECTED_SOURCE_ROWS "$EXPECTED_SOURCE_ROWS" 57046
require_pinned_value EXPECTED_ELIGIBLE_ROWS "$EXPECTED_ELIGIBLE_ROWS" 57045
require_pinned_value EXPECTED_EXCLUDED_ID "$EXPECTED_EXCLUDED_ID" deepmath-level6-038794
require_pinned_value SOURCE_SHA256 "$SOURCE_SHA256" de3350fdd00bc0410550098ea65179e2be873da99e4075f80de575fc17670597
require_pinned_value PREPARED_JSONL_SHA256 "$PREPARED_JSONL_SHA256" ee8d55c943577888f14052b3953a0a8aa07d26076e50ddffa41a102b60027344
require_pinned_value PREPARED_MANIFEST_SHA256 "$PREPARED_MANIFEST_SHA256" 5add2e965473d3647f2318c73f957434fbe23344516089be9adfe89db9c6a70e
require_pinned_value ELIGIBILITY_REPORT_SHA256 "$ELIGIBILITY_REPORT_SHA256" cc8d5b888def37761519e15c6ef722e5a22d91ae2342c6d669f008bcff3dcbb6
require_pinned_value GRADIENT_MANIFEST_SHA256 "$GRADIENT_MANIFEST_SHA256" 9a534118a08e736a15e806933d5cf90c2a99822fee28bf18644e5ff344ce3ae2
require_pinned_value REFERENCE_COMMIT "$REFERENCE_COMMIT" d9484cd3b5991030b901ac4a3a9e2472dbfac2ad
require_pinned_value REFERENCE_TREE "$REFERENCE_TREE" a0079d8c5e15cb18bb4790f99c43cc19bb9ecd50
require_pinned_value OLD_PARQUET_SHA256 "$OLD_PARQUET_SHA256" caf303c5d151fdaed2e21eebc257917f13660c589906e3da0e569ffdfd59b059
require_pinned_value OLD_MANIFEST_SHA256 "$OLD_MANIFEST_SHA256" a1a45382ee577e24f9b455386f9f3adcab210ff40a83ea760c2110fb96f8f90a
require_pinned_value OLD_SELECTED_IDS_SHA256 "$OLD_SELECTED_IDS_SHA256" a78c04cff10c148f45bf9c828e07cc9cf01398b48efeb707253273af359a037e
require_pinned_value OLD_DIAGNOSTICS_SHA256 "$OLD_DIAGNOSTICS_SHA256" d2105179f2ce5a7c796aef07136bbc1dce79b07b5bc2de4d2319b18b47ea761f
require_pinned_value PRIMARY_CLUSTER_RATIO "$PRIMARY_CLUSTER_RATIO" 0.10
require_pinned_value SENSITIVITY_CLUSTER_RATIO "$SENSITIVITY_CLUSTER_RATIO" 0.01
require_pinned_value CLUSTER_SEEDS "$CLUSTER_SEEDS" 42,43
require_pinned_value CLUSTER_ITERATIONS "$CLUSTER_ITERATIONS" 20
require_pinned_value SELECTION_SEED "$SELECTION_SEED" 42
require_pinned_value GRADIENT_PREFIX "$GRADIENT_PREFIX" deepmath
require_pinned_value EXPECTED_NUM_SHARDS "$EXPECTED_NUM_SHARDS" 4

selection_command=(
  "$PYTHON_BIN" -m math_eval.select_gradient_diverse_deepmath
  --source-parquet "$SOURCE_PARQUET"
  --prepared-jsonl "$PREPARED_JSONL"
  --prepared-manifest "$PREPARED_MANIFEST"
  --eligibility-report "$ELIGIBILITY_REPORT"
  --gradient-dir "$GRADIENT_DIR"
  --reference-repo "$REFERENCE_REPO"
  --device cuda:0
  --output-parquet "$OUTPUT_PARQUET"
  --selected-ids "$SELECTED_IDS"
  --diagnostics "$DIAGNOSTICS"
  --manifest "$SELECTION_MANIFEST"
  --expected-source-count 57046
  --expected-eligible-count 57045
  --target-size 51200
  --selection-profile vanilla_51200
  --frozen-prefix-selected-ids "$OLD_SELECTED_IDS"
)

print_contract() {
  printf 'selection_profile=%s\n' "$SELECTION_PROFILE"
  printf 'target_rows=%s\n' "$TARGET_ROWS"
  printf 'expected_source_rows=%s\n' "$EXPECTED_SOURCE_ROWS"
  printf 'expected_eligible_rows=%s\n' "$EXPECTED_ELIGIBLE_ROWS"
  printf 'source_sha256=%s\n' "$SOURCE_SHA256"
  printf 'prepared_jsonl_sha256=%s\n' "$PREPARED_JSONL_SHA256"
  printf 'prepared_manifest_sha256=%s\n' "$PREPARED_MANIFEST_SHA256"
  printf 'eligibility_report_sha256=%s\n' "$ELIGIBILITY_REPORT_SHA256"
  printf 'gradient_manifest_sha256=%s\n' "$GRADIENT_MANIFEST_SHA256"
  printf 'reference_commit=%s\n' "$REFERENCE_COMMIT"
  printf 'reference_tree=%s\n' "$REFERENCE_TREE"
  printf 'primary_cluster_ratio=%s\n' "$PRIMARY_CLUSTER_RATIO"
  printf 'sensitivity_cluster_ratio=%s\n' "$SENSITIVITY_CLUSTER_RATIO"
  printf 'cluster_seeds=%s\n' "$CLUSTER_SEEDS"
  printf 'cluster_iterations=%s\n' "$CLUSTER_ITERATIONS"
  printf 'selection_seed=%s\n' "$SELECTION_SEED"
  printf 'source_parquet=%s\n' "$SOURCE_PARQUET"
  printf 'gradient_dir=%s\n' "$GRADIENT_DIR"
  printf 'frozen_prefix_selected_ids=%s\n' "$OLD_SELECTED_IDS"
  printf 'output_parquet=%s\n' "$OUTPUT_PARQUET"
  printf 'selected_ids=%s\n' "$SELECTED_IDS"
  printf 'diagnostics=%s\n' "$DIAGNOSTICS"
  printf 'manifest=%s\n' "$SELECTION_MANIFEST"
  printf 'master_log=%s\n' "$MASTER_LOG"
  printf 'selection_command='
  printf '%q ' "${selection_command[@]}"
  printf '\n'
}

require_sha256() {
  local path="$1"
  local expected="$2"
  [[ -f "$path" ]] || die "required frozen artifact is missing: $path"
  local actual
  actual="$(sha256sum -- "$path" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || \
    die "SHA-256 mismatch for $path: expected $expected, got $actual"
}

require_clean_repository() {
  local repository="$1"
  local description="$2"
  [[ -d "$repository/.git" || -f "$repository/.git" ]] || \
    die "$description is not a Git worktree: $repository"
  local status
  status="$(cd "$repository" && git status --porcelain=v1 --untracked-files=all)"
  [[ -z "$status" ]] || die "$description must be clean: $status"
}

verify_frozen_inputs() {
  require_sha256 "$SOURCE_PARQUET" "$SOURCE_SHA256"
  require_sha256 "$PREPARED_JSONL" "$PREPARED_JSONL_SHA256"
  require_sha256 "$PREPARED_MANIFEST" "$PREPARED_MANIFEST_SHA256"
  require_sha256 "$ELIGIBILITY_REPORT" "$ELIGIBILITY_REPORT_SHA256"
  require_sha256 "$GRADIENT_MANIFEST" "$GRADIENT_MANIFEST_SHA256"
  require_sha256 "$OLD_PARQUET" "$OLD_PARQUET_SHA256"
  require_sha256 "$OLD_MANIFEST" "$OLD_MANIFEST_SHA256"
  require_sha256 "$OLD_SELECTED_IDS" "$OLD_SELECTED_IDS_SHA256"
  require_sha256 "$OLD_DIAGNOSTICS" "$OLD_DIAGNOSTICS_SHA256"

  require_clean_repository "$REFERENCE_REPO" "reference repository"
  local actual_reference_commit actual_reference_tree
  actual_reference_commit="$(git -C "$REFERENCE_REPO" rev-parse HEAD)"
  actual_reference_tree="$(git -C "$REFERENCE_REPO" rev-parse 'HEAD^{tree}')"
  [[ "$actual_reference_commit" == "$REFERENCE_COMMIT" ]] || \
    die "reference commit mismatch: expected $REFERENCE_COMMIT, got $actual_reference_commit"
  [[ "$actual_reference_tree" == "$REFERENCE_TREE" ]] || \
    die "reference tree mismatch: expected $REFERENCE_TREE, got $actual_reference_tree"
}

validate_output_state() {
  local outputs=(
    "$OUTPUT_PARQUET"
    "$SELECTED_IDS"
    "$DIAGNOSTICS"
    "$SELECTION_MANIFEST"
  )
  local existing=0
  local output
  for output in "${outputs[@]}"; do
    if [[ -e "$output" ]]; then
      [[ -f "$output" ]] || die "selection output path is not a file: $output"
      ((existing += 1))
    fi
  done
  if (( existing != 0 && existing != ${#outputs[@]} )); then
    die "partial 51,200 selection artifact set exists (${existing}/${#outputs[@]})"
  fi
}

run_global_coverage_gate() {
  "$PYTHON_BIN" -m math_eval.collect_prismatic_gradients \
    --validate-global-only \
    --prepared-jsonl "$PREPARED_JSONL" \
    --prepared-manifest "$PREPARED_MANIFEST" \
    --eligibility-report "$ELIGIBILITY_REPORT" \
    --output-dir "$GRADIENT_DIR" \
    --prefix "$GRADIENT_PREFIX" \
    --num-shards "$EXPECTED_NUM_SHARDS"
}

print_provenance() {
  local code_head code_tree code_status_sha code_diff_sha
  code_head="$(git -C "$CODE_ROOT" rev-parse HEAD)"
  code_tree="$(git -C "$CODE_ROOT" rev-parse 'HEAD^{tree}')"
  code_status_sha="$(git -C "$CODE_ROOT" status --porcelain=v1 --untracked-files=all | sha256sum | awk '{print $1}')"
  code_diff_sha="$(git -C "$CODE_ROOT" diff --binary HEAD | sha256sum | awk '{print $1}')"
  printf 'started_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'code_root=%s\n' "$CODE_ROOT"
  printf 'code_head=%s\n' "$code_head"
  printf 'code_tree=%s\n' "$code_tree"
  printf 'code_status_sha256=%s\n' "$code_status_sha"
  printf 'code_diff_sha256=%s\n' "$code_diff_sha"
  printf 'python_bin=%s\n' "$PYTHON_BIN"
  printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-}"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-}"
  print_contract
  "$PYTHON_BIN" - <<'PY'
import importlib.metadata
import json
import platform

packages = {}
for name in ("numpy", "pyarrow", "safetensors", "scikit-learn", "torch"):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
print(json.dumps({"packages": packages, "python": platform.python_version()}, sort_keys=True))
PY
}

case "${SFTGRAD51200_DRY_RUN:-0}" in
  0|1) ;;
  *) die "SFTGRAD51200_DRY_RUN must be 0 or 1" ;;
esac
case "${SFTGRAD51200_PREFLIGHT_ONLY:-0}" in
  0|1) ;;
  *) die "SFTGRAD51200_PREFLIGHT_ONLY must be 0 or 1" ;;
esac

if [[ "${SFTGRAD51200_DRY_RUN:-0}" == "1" ]]; then
  print_contract
  exit 0
fi

require_pinned_value CODE_ROOT "$CODE_ROOT" /home/mchen/FiRe-OPD/.worktrees/vanilla-sft-gradient-51200-opd
require_pinned_value REFERENCE_REPO "$REFERENCE_REPO" /home/mchen/prismatic-synthesis-reference
[[ -x "$PYTHON_BIN" ]] || die "PYTHON_BIN is not executable: $PYTHON_BIN"
require_pinned_value PYTHON_BIN "$PYTHON_BIN" /home/mchen/miniconda3/envs/gvendi-opd/bin/python
for command in awk flock git sha256sum tee; do
  command -v "$command" >/dev/null 2>&1 || die "required command is missing: $command"
done

export PYTHONPATH="${CODE_ROOT}/verl:${CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "$CODE_ROOT"
require_clean_repository "$CODE_ROOT" "selection code worktree"
verify_frozen_inputs
validate_output_state
mkdir -p "$SELECTION_DIR" "$(dirname -- "$OUTPUT_PARQUET")" "$(dirname -- "$MASTER_LOG")"
exec 9<>"$MASTER_LOCK"
flock -n 9 || die "another selection launcher holds $MASTER_LOCK"
printf '%s\n' "$$" >&9
exec > >(tee -a "$MASTER_LOG") 2>&1

printf '%s\n' '=== DeepMath frozen-gradient 51,200 selection preflight ==='
print_provenance
run_global_coverage_gate
verify_frozen_inputs
validate_output_state
printf '%s\n' 'SELECTION_PREFLIGHT=PASS'

if [[ "${SFTGRAD51200_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  printf '%s\n' 'selection_gpu_work=SKIPPED_PREFLIGHT_ONLY'
  exit 0
fi

[[ -n "${SLURM_JOB_ID:-}" ]] || die "SLURM_JOB_ID must be nonempty"
allocated_tokens="$($PYTHON_BIN - <<'PY'
from math_eval.run_opd_proxy_gradient_verify import validate_opd_cli_runtime

print(",".join(validate_opd_cli_runtime(expected_gpus=4, require_idle=True)))
PY
)"
[[ -n "$allocated_tokens" ]] || die "runtime validation returned no CUDA tokens"
selection_token="${allocated_tokens%%,*}"
printf 'validated_allocation_tokens=%s\n' "$allocated_tokens"
printf 'selection_cuda_token=%s\n' "$selection_token"

unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES="$selection_token"
printf '%s\n' '=== DeepMath frozen-gradient 51,200 selection execution ==='
"${selection_command[@]}"

verify_frozen_inputs
validate_output_state
for output in "$OUTPUT_PARQUET" "$SELECTED_IDS" "$DIAGNOSTICS" "$SELECTION_MANIFEST"; do
  [[ -f "$output" ]] || die "selection did not publish required output: $output"
  printf 'published_sha256=%s path=%s\n' \
    "$(sha256sum -- "$output" | awk '{print $1}')" "$output"
done
printf '%s\n' 'SELECTION_51200_PUBLICATION=PASS'
