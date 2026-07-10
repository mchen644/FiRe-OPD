#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$SCRIPT_DIR"
cd "$REPO_ROOT"

GPU_IDS="${GPU_IDS-0,1,2,3}"
PYTHON_BIN="${PYTHON_BIN:-/home/mchen/miniconda3/envs/gvendi-opd/bin/python}"
REFERENCE_REPO="${REFERENCE_REPO:-/home/mchen/prismatic-synthesis-reference}"
SOURCE_PARQUET="${SOURCE_PARQUET:-$REPO_ROOT/data/g-opd/DeepMath-103K/train_filtered_level6.parquet}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/data/gradient_diversity}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/gradient_diversity}"

EXPECTED_SOURCE_ROWS="${EXPECTED_SOURCE_ROWS:-57046}"
EXPECTED_ELIGIBLE_ROWS="${EXPECTED_ELIGIBLE_ROWS:-57045}"
EXPECTED_EXCLUDED_ID="${EXPECTED_EXCLUDED_ID:-deepmath-level6-038794}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-32768}"
TARGET_ROWS="${TARGET_ROWS:-12800}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
MODEL_REVISION="${MODEL_REVISION:-7ae557604adf67be50417f59c2c2f167def9a775}"
DATASET_NAME="${DATASET_NAME:-zwhe99/DeepMath-103K}"
DATASET_REVISION="${DATASET_REVISION:-5cf055d1fe3d7a2eb19719ac020211469736ae44}"
REFERENCE_COMMIT="${REFERENCE_COMMIT:-d9484cd3b5991030b901ac4a3a9e2472dbfac2ad}"
PRIMARY_CLUSTER_RATIO="${PRIMARY_CLUSTER_RATIO:-0.10}"
SENSITIVITY_CLUSTER_RATIO="${SENSITIVITY_CLUSTER_RATIO:-0.01}"
CLUSTER_SEEDS="${CLUSTER_SEEDS:-42,43}"
MAX_GPU_MEMORY_USED_MIB="${MAX_GPU_MEMORY_USED_MIB:-16}"
GRADIENT_PREFIX="${GRADIENT_PREFIX:-deepmath}"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

require_pinned_value() {
  local name="$1"
  local actual="$2"
  local expected="$3"
  [[ "$actual" == "$expected" ]] || die "$name must remain pinned to $expected (got $actual)"
}

trim_whitespace() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

require_pinned_value EXPECTED_SOURCE_ROWS "$EXPECTED_SOURCE_ROWS" 57046
require_pinned_value EXPECTED_ELIGIBLE_ROWS "$EXPECTED_ELIGIBLE_ROWS" 57045
require_pinned_value EXPECTED_EXCLUDED_ID "$EXPECTED_EXCLUDED_ID" deepmath-level6-038794
require_pinned_value MAX_CONTEXT_TOKENS "$MAX_CONTEXT_TOKENS" 32768
require_pinned_value TARGET_ROWS "$TARGET_ROWS" 12800
require_pinned_value MODEL_NAME "$MODEL_NAME" Qwen/Qwen2.5-0.5B-Instruct
require_pinned_value MODEL_REVISION "$MODEL_REVISION" 7ae557604adf67be50417f59c2c2f167def9a775
require_pinned_value DATASET_NAME "$DATASET_NAME" zwhe99/DeepMath-103K
require_pinned_value DATASET_REVISION "$DATASET_REVISION" 5cf055d1fe3d7a2eb19719ac020211469736ae44
require_pinned_value REFERENCE_COMMIT "$REFERENCE_COMMIT" d9484cd3b5991030b901ac4a3a9e2472dbfac2ad
require_pinned_value PRIMARY_CLUSTER_RATIO "$PRIMARY_CLUSTER_RATIO" 0.10
require_pinned_value SENSITIVITY_CLUSTER_RATIO "$SENSITIVITY_CLUSTER_RATIO" 0.01
require_pinned_value CLUSTER_SEEDS "$CLUSTER_SEEDS" 42,43
require_pinned_value GRADIENT_PREFIX "$GRADIENT_PREFIX" deepmath
[[ "$MAX_GPU_MEMORY_USED_MIB" =~ ^[0-9]+$ ]] || die "MAX_GPU_MEMORY_USED_MIB must be a nonnegative integer"

[[ "$GPU_IDS" =~ ^[[:space:]]*[0-9]+([[:space:]]*,[[:space:]]*[0-9]+)*[[:space:]]*$ ]] || \
  die "GPU_IDS must be a nonempty comma-separated list of physical numeric GPU IDs"
IFS=',' read -r -a raw_gpu_ids <<< "$GPU_IDS"
gpu_ids=()
declare -A seen_gpu_ids=()
for raw_gpu_id in "${raw_gpu_ids[@]}"; do
  gpu_id="$(trim_whitespace "$raw_gpu_id")"
  [[ -z "${seen_gpu_ids[$gpu_id]+present}" ]] || die "GPU_IDS contains duplicate physical GPU $gpu_id"
  seen_gpu_ids["$gpu_id"]=1
  gpu_ids+=("$gpu_id")
done
(( ${#gpu_ids[@]} > 0 )) || die "GPU_IDS must contain at least one physical GPU"

[[ -x "$PYTHON_BIN" ]] || die "PYTHON_BIN is not executable: $PYTHON_BIN"
[[ -f "$SOURCE_PARQUET" ]] || die "SOURCE_PARQUET does not exist: $SOURCE_PARQUET"
[[ -d "$REFERENCE_REPO" ]] || die "REFERENCE_REPO does not exist: $REFERENCE_REPO"
command -v flock >/dev/null 2>&1 || die "flock is required"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required"

PYTHON_BIN="$(realpath -e -- "$PYTHON_BIN")"
SOURCE_PARQUET="$(realpath -e -- "$SOURCE_PARQUET")"
REFERENCE_REPO="$(realpath -e -- "$REFERENCE_REPO")"
OUTPUT_ROOT="$(realpath -m -- "$OUTPUT_ROOT")"
LOG_ROOT="$(realpath -m -- "$LOG_ROOT")"

PREPARED_JSONL="$OUTPUT_ROOT/deepmath_level6_r1_solution1.jsonl"
PREPARED_MANIFEST="$OUTPUT_ROOT/deepmath_level6_r1_solution1.manifest.json"
ELIGIBILITY_REPORT="$OUTPUT_ROOT/deepmath_level6_r1_solution1.eligibility.json"
GRADIENT_DIR="$OUTPUT_ROOT/gradients/qwen2.5-0.5b-instruct"
SELECTION_DIR="$OUTPUT_ROOT/selection"
DIAGNOSTICS="$SELECTION_DIR/diagnostics.json"
SELECTED_IDS="$SELECTION_DIR/selected_ids.jsonl"
SELECTION_MANIFEST="$SELECTION_DIR/manifest.json"
OUTPUT_PARQUET="$OUTPUT_ROOT/DeepMath-103K/train_gradient_diverse_12800.parquet"
MASTER_LOG="$LOG_ROOT/deepmath_gradient_diverse_12800.log"
MASTER_LOCK="$OUTPUT_ROOT/.deepmath_gradient_diverse_12800.launch.lock"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$GRADIENT_DIR" "$SELECTION_DIR" "$(dirname -- "$OUTPUT_PARQUET")"
exec 9>"$MASTER_LOCK"
flock -n 9 || die "another launcher already holds the production lock: $MASTER_LOCK"
exec > >(tee -a "$MASTER_LOG") 2>&1

printf '=== gradient-diverse DeepMath production launcher ===\n'
printf 'started_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'repo_root=%s\n' "$REPO_ROOT"
printf 'gpu_ids=%s\n' "$(IFS=,; printf '%s' "${gpu_ids[*]}")"
printf 'python_bin=%s\n' "$PYTHON_BIN"
printf 'reference_repo=%s\n' "$REFERENCE_REPO"
printf 'reference_commit=%s\n' "$REFERENCE_COMMIT"
printf 'source_parquet=%s\n' "$SOURCE_PARQUET"
printf 'output_root=%s\n' "$OUTPUT_ROOT"
printf 'master_log=%s\n' "$MASTER_LOG"
printf 'expected_source_rows=%s\n' "$EXPECTED_SOURCE_ROWS"
printf 'expected_eligible_rows=%s\n' "$EXPECTED_ELIGIBLE_ROWS"
printf 'expected_excluded_id=%s\n' "$EXPECTED_EXCLUDED_ID"
printf 'max_context_tokens=%s\n' "$MAX_CONTEXT_TOKENS"
printf 'target_rows=%s\n' "$TARGET_ROWS"
printf 'dataset_name=%s\n' "$DATASET_NAME"
printf 'dataset_revision=%s\n' "$DATASET_REVISION"
printf 'model_name=%s\n' "$MODEL_NAME"
printf 'model_revision=%s\n' "$MODEL_REVISION"
printf 'primary_cluster_ratio=%s\n' "$PRIMARY_CLUSTER_RATIO"
printf 'sensitivity_cluster_ratio=%s\n' "$SENSITIVITY_CLUSTER_RATIO"
printf 'cluster_seeds=%s\n' "$CLUSTER_SEEDS"
printf 'max_gpu_memory_used_mib=%s\n' "$MAX_GPU_MEMORY_USED_MIB"
printf 'prepared_jsonl=%s\n' "$PREPARED_JSONL"
printf 'prepared_manifest=%s\n' "$PREPARED_MANIFEST"
printf 'eligibility_report=%s\n' "$ELIGIBILITY_REPORT"
printf 'gradient_dir=%s\n' "$GRADIENT_DIR"
printf 'output_parquet=%s\n' "$OUTPUT_PARQUET"

collector_pids=()
cleanup_collectors() {
  local status=$?
  trap - EXIT INT TERM
  if (( status != 0 && ${#collector_pids[@]} > 0 )); then
    printf 'terminating %s collector child process(es) after failure\n' "${#collector_pids[@]}" >&2
    for pid in "${collector_pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        kill -TERM "$pid" 2>/dev/null || true
      fi
    done
    for pid in "${collector_pids[@]}"; do
      wait "$pid" 2>/dev/null || true
    done
  fi
  exit "$status"
}
trap cleanup_collectors EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

printf '%s\n' '--- stage: prepare pinned pool ---'
"$PYTHON_BIN" -m math_eval.prepare_deepmath_gradient_pool \
  --source-parquet "$SOURCE_PARQUET" \
  --output-jsonl "$PREPARED_JSONL" \
  --manifest "$PREPARED_MANIFEST" \
  --dataset-name "$DATASET_NAME" \
  --dataset-revision "$DATASET_REVISION" \
  --expected-count "$EXPECTED_SOURCE_ROWS"

printf '%s\n' '--- stage: build and validate eligibility ---'
"$PYTHON_BIN" -m math_eval.build_gradient_eligibility \
  --prepared-jsonl "$PREPARED_JSONL" \
  --prepared-manifest "$PREPARED_MANIFEST" \
  --output "$ELIGIBILITY_REPORT" \
  --model-name "$MODEL_NAME" \
  --model-revision "$MODEL_REVISION" \
  --max-context-tokens "$MAX_CONTEXT_TOKENS" \
  --max-excluded 1 \
  --expected-excluded-id "$EXPECTED_EXCLUDED_ID"

"$PYTHON_BIN" - "$ELIGIBILITY_REPORT" "$EXPECTED_SOURCE_ROWS" "$EXPECTED_ELIGIBLE_ROWS" "$EXPECTED_EXCLUDED_ID" <<'PY'
import json
import pathlib
import sys

report_path = pathlib.Path(sys.argv[1])
report = json.loads(report_path.read_text(encoding="utf-8"))
expected_source = int(sys.argv[2])
expected_eligible = int(sys.argv[3])
expected_excluded = sys.argv[4]
excluded_rows = report.get("excluded_rows")
excluded_ids = (
    [row.get("id") for row in excluded_rows]
    if isinstance(excluded_rows, list)
    and all(isinstance(row, dict) for row in excluded_rows)
    else None
)
expected = {
    "source_row_count": expected_source,
    "eligible_row_count": expected_eligible,
    "excluded_row_count": 1,
}
for field, value in expected.items():
    if report.get(field) != value:
        raise SystemExit(
            f"eligibility production gate failed: {field}="
            f"{report.get(field)!r}, expected {value!r}"
        )
if excluded_ids != [expected_excluded]:
    raise SystemExit(
        "eligibility production gate failed: excluded IDs="
        f"{excluded_ids!r}, expected {[expected_excluded]!r}"
    )
print(
    "eligibility production gate passed: "
    f"source={expected_source} eligible={expected_eligible} "
    f"excluded={expected_excluded}"
)
PY

check_requested_gpu_free() {
  local physical_id="$1"
  local gpu_info
  local compute_pids
  local reported_index
  local uuid
  local memory_used
  local extra

  if ! gpu_info="$(nvidia-smi --id="$physical_id" --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits)"; then
    die "failed to query requested GPU $physical_id"
  fi
  [[ -n "$gpu_info" && "$gpu_info" != *$'\n'* ]] || die "GPU $physical_id query returned an unexpected row count"
  IFS=',' read -r reported_index uuid memory_used extra <<< "$gpu_info"
  reported_index="$(trim_whitespace "$reported_index")"
  uuid="$(trim_whitespace "$uuid")"
  memory_used="$(trim_whitespace "$memory_used")"
  [[ -z "${extra:-}" ]] || die "GPU $physical_id query returned unexpected fields"
  [[ "$reported_index" == "$physical_id" ]] || die "requested GPU $physical_id resolved to index $reported_index"
  [[ -n "$uuid" ]] || die "GPU $physical_id returned an empty UUID"
  [[ "$memory_used" =~ ^[0-9]+$ ]] || die "GPU $physical_id returned invalid memory.used=$memory_used"

  if ! compute_pids="$(nvidia-smi --id="$physical_id" --query-compute-apps=pid --format=csv,noheader,nounits)"; then
    die "failed to query compute processes for requested GPU $physical_id"
  fi
  printf 'gpu_preflight physical_id=%s uuid=%s memory_used_mib=%s compute_pids=%s\n' \
    "$physical_id" "$uuid" "$memory_used" "${compute_pids:-none}"
  [[ -z "$(trim_whitespace "$compute_pids")" ]] || die "GPU $physical_id has active compute process(es): $compute_pids"
  (( memory_used <= MAX_GPU_MEMORY_USED_MIB )) || \
    die "GPU $physical_id uses ${memory_used} MiB, above idle threshold ${MAX_GPU_MEMORY_USED_MIB} MiB"
}

printf '%s\n' '--- stage: requested GPU availability gate ---'
for physical_id in "${gpu_ids[@]}"; do
  check_requested_gpu_free "$physical_id"
done

printf '%s\n' '--- stage: parallel official gradient collection ---'
num_shards="${#gpu_ids[@]}"
collector_logs=()
for logical_shard in "${!gpu_ids[@]}"; do
  physical_id="${gpu_ids[$logical_shard]}"
  shard_log="$LOG_ROOT/deepmath_gradient_shard_${logical_shard}_gpu_${physical_id}.log"
  collector_logs+=("$shard_log")
  (
    export CUDA_VISIBLE_DEVICES="$physical_id"
    "$PYTHON_BIN" -m math_eval.collect_prismatic_gradients \
      --reference-repo "$REFERENCE_REPO" \
      --prepared-jsonl "$PREPARED_JSONL" \
      --prepared-manifest "$PREPARED_MANIFEST" \
      --eligibility-report "$ELIGIBILITY_REPORT" \
      --output-dir "$GRADIENT_DIR" \
      --prefix "$GRADIENT_PREFIX" \
      --model-name "$MODEL_NAME" \
      --model-revision "$MODEL_REVISION" \
      --shard-index "$logical_shard" \
      --num-shards "$num_shards" \
      --device cuda:0
  ) >"$shard_log" 2>&1 &
  collector_pids+=("$!")
  printf 'collector_started logical_shard=%s physical_gpu=%s pid=%s log=%s\n' \
    "$logical_shard" "$physical_id" "$!" "$shard_log"
done

collector_failed=0
for collector_index in "${!collector_pids[@]}"; do
  pid="${collector_pids[$collector_index]}"
  if wait "$pid"; then
    printf 'collector_finished pid=%s log=%s\n' "$pid" "${collector_logs[$collector_index]}"
  else
    status=$?
    printf 'collector_failed pid=%s status=%s log=%s\n' \
      "$pid" "$status" "${collector_logs[$collector_index]}" >&2
    collector_failed=1
  fi
done
collector_pids=()
(( collector_failed == 0 )) || die "one or more gradient collectors failed; global validation was not run"

printf '%s\n' '--- stage: exact global gradient coverage validation ---'
"$PYTHON_BIN" -m math_eval.collect_prismatic_gradients \
  --validate-global-only \
  --prepared-jsonl "$PREPARED_JSONL" \
  --prepared-manifest "$PREPARED_MANIFEST" \
  --eligibility-report "$ELIGIBILITY_REPORT" \
  --output-dir "$GRADIENT_DIR" \
  --prefix "$GRADIENT_PREFIX" \
  --num-shards "$num_shards"

printf '%s\n' '--- stage: exact fixed-pool selection ---'
export CUDA_VISIBLE_DEVICES="${gpu_ids[0]}"
"$PYTHON_BIN" -m math_eval.select_gradient_diverse_deepmath \
  --source-parquet "$SOURCE_PARQUET" \
  --prepared-jsonl "$PREPARED_JSONL" \
  --prepared-manifest "$PREPARED_MANIFEST" \
  --eligibility-report "$ELIGIBILITY_REPORT" \
  --gradient-dir "$GRADIENT_DIR" \
  --reference-repo "$REFERENCE_REPO" \
  --device cuda:0 \
  --output-parquet "$OUTPUT_PARQUET" \
  --selected-ids "$SELECTED_IDS" \
  --diagnostics "$DIAGNOSTICS" \
  --manifest "$SELECTION_MANIFEST" \
  --expected-source-count "$EXPECTED_SOURCE_ROWS" \
  --expected-eligible-count "$EXPECTED_ELIGIBLE_ROWS" \
  --target-size "$TARGET_ROWS"

printf 'gradient-diverse selection complete output_parquet=%s manifest=%s\n' \
  "$OUTPUT_PARQUET" "$SELECTION_MANIFEST"
