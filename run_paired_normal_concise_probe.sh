#!/usr/bin/env bash
set -euo pipefail

WORKTREE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT=/home/mchen/FiRe-OPD
DATASET="${PAIRED_PROBE_DATASET:-${ROOT}/data/g-opd/DeepMath-103K/train_filtered_level6.parquet}"
DATASET_SHA256=de3350fdd00bc0410550098ea65179e2be873da99e4075f80de575fc17670597
BASE_MODEL="${PAIRED_PROBE_BASE_MODEL:-${ROOT}/models/Qwen3-4B}"
ADAPTIVE_MODEL="${PAIRED_PROBE_ADAPTIVE_MODEL:-${ROOT}/checkpoints/opd-adaptive-conciseprobe-cap50-tokenneutral-rawprompt-4gpu-tp4-step50/global_step_50_hf}"
OUTPUT_ROOT="${PAIRED_PROBE_OUTPUT_ROOT:-${ROOT}/math_eval/paired_normal_concise_probe_outputs}"
LOG_ROOT="${PAIRED_PROBE_LOG_ROOT:-${ROOT}/math_eval/paired_normal_concise_probe_logs}"
CPU_PYTHON=/home/mchen/miniconda3/envs/gvendi-opd/bin/python
GPU_PYTHON=/home/mchen/miniconda3/envs/verl/bin/python
JOB_ID=429
SAMPLE_SIZE=128
SEED=42
EXPECTED_CUDA=0,1,2,3
DRY_RUN="${PAIRED_PROBE_DRY_RUN:-0}"
SOURCE_COMMIT="$(git -C "$WORKTREE" rev-parse HEAD)"
RUN_ID="${PAIRED_PROBE_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-${SOURCE_COMMIT:0:12}}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
LOG_PATH="${LOG_ROOT}/${RUN_ID}.log"
SAMPLE_FILE="${RUN_DIR}/sample.jsonl"
MANIFEST_FILE="${RUN_DIR}/manifest.json"

finish() {
  local rc=$?
  trap - EXIT
  echo "PAIRED_NORMAL_CONCISE_PROBE_DONE_${RUN_ID}:${rc}"
  exit "$rc"
}
trap finish EXIT

print_command() {
  printf 'COMMAND:'
  printf ' %q' "$@"
  printf '\n'
}

run_command() {
  print_command "$@"
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

printf 'WORKTREE=%s\n' "$WORKTREE"
printf 'SOURCE_COMMIT=%s\n' "$SOURCE_COMMIT"
printf 'RUN_ID=%s\n' "$RUN_ID"
printf 'RUN_DIR=%s\n' "$RUN_DIR"
printf 'LOG_PATH=%s\n' "$LOG_PATH"
printf 'DATASET=%s\n' "$DATASET"
printf 'DATASET_SHA256=%s\n' "$DATASET_SHA256"
printf 'BASE_MODEL=%s\n' "$BASE_MODEL"
printf 'ADAPTIVE_MODEL=%s\n' "$ADAPTIVE_MODEL"
printf 'SAMPLE_SIZE=%s\n' "$SAMPLE_SIZE"
printf 'SEED=%s\n' "$SEED"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-$EXPECTED_CUDA}"

PREPARE_COMMAND=(
  env "PYTHONPATH=${WORKTREE}/verl:${WORKTREE}"
  "$CPU_PYTHON" "${WORKTREE}/math_eval/paired_normal_concise_probe.py" prepare
  --dataset "$DATASET"
  --sample-file "$SAMPLE_FILE"
  --manifest "$MANIFEST_FILE"
  --sample-size "$SAMPLE_SIZE"
  --seed "$SEED"
  --source-commit "$SOURCE_COMMIT"
  --expected-dataset-sha256 "$DATASET_SHA256"
)
BASE_COMMAND=(
  env "PYTHONPATH=${WORKTREE}/verl:${WORKTREE}" VLLM_WORKER_MULTIPROC_METHOD=spawn
  "$GPU_PYTHON" "${WORKTREE}/math_eval/paired_normal_concise_probe.py" generate
  --sample-file "$SAMPLE_FILE"
  --model-path "$BASE_MODEL"
  --model-label base
  --output-dir "${RUN_DIR}/base"
  --expected-count "$SAMPLE_SIZE"
  --tensor-parallel-size 4
  --max-model-len 40960
  --max-num-seqs 128
  --gpu-memory-utilization 0.90
)
ADAPTIVE_COMMAND=(
  env "PYTHONPATH=${WORKTREE}/verl:${WORKTREE}" VLLM_WORKER_MULTIPROC_METHOD=spawn
  "$GPU_PYTHON" "${WORKTREE}/math_eval/paired_normal_concise_probe.py" generate
  --sample-file "$SAMPLE_FILE"
  --model-path "$ADAPTIVE_MODEL"
  --model-label adaptive_step50
  --output-dir "${RUN_DIR}/adaptive_step50"
  --expected-count "$SAMPLE_SIZE"
  --tensor-parallel-size 4
  --max-model-len 40960
  --max-num-seqs 128
  --gpu-memory-utilization 0.90
)
ANALYZE_COMMAND=(
  env "PYTHONPATH=${WORKTREE}/verl:${WORKTREE}"
  "$CPU_PYTHON" "${WORKTREE}/math_eval/paired_normal_concise_probe.py" analyze
  --run-dir "$RUN_DIR"
  --expected-count "$SAMPLE_SIZE"
)
VALIDATE_COMMAND=(
  "$CPU_PYTHON" "${WORKTREE}/math_eval/validate_paired_normal_concise_probe.py"
  --run-dir "$RUN_DIR"
  --expected-count "$SAMPLE_SIZE"
)

if [[ "$DRY_RUN" == "1" ]]; then
  print_command "${PREPARE_COMMAND[@]}"
  print_command "${BASE_COMMAND[@]}"
  print_command "${ADAPTIVE_COMMAND[@]}"
  print_command "${ANALYZE_COMMAND[@]}"
  print_command "${VALIDATE_COMMAND[@]}"
  exit 0
fi

if [[ -e "$RUN_DIR" ]]; then
  echo "run namespace already exists: $RUN_DIR" >&2
  exit 1
fi
if [[ -e "$LOG_PATH" ]]; then
  echo "log namespace already exists: $LOG_PATH" >&2
  exit 1
fi
if [[ "${CUDA_VISIBLE_DEVICES:-}" != "$EXPECTED_CUDA" ]]; then
  echo "CUDA_VISIBLE_DEVICES must equal $EXPECTED_CUDA" >&2
  exit 1
fi
if [[ -n "$(git -C "$WORKTREE" status --porcelain)" ]]; then
  echo "source worktree must be clean before launch" >&2
  exit 1
fi
if [[ ! -f "$DATASET" ]]; then
  echo "dataset is missing: $DATASET" >&2
  exit 1
fi
actual_dataset_sha256="$(sha256sum "$DATASET" | awk '{print $1}')"
if [[ "$actual_dataset_sha256" != "$DATASET_SHA256" ]]; then
  echo "dataset SHA256 mismatch" >&2
  exit 1
fi
if [[ ! -d "$BASE_MODEL" || ! -f "${BASE_MODEL}/config.json" ]]; then
  echo "base model is incomplete: $BASE_MODEL" >&2
  exit 1
fi
if [[ ! -d "$ADAPTIVE_MODEL" || ! -f "${ADAPTIVE_MODEL}/config.json" ]]; then
  echo "adaptive model is incomplete: $ADAPTIVE_MODEL" >&2
  exit 1
fi
mapfile -t adaptive_shards < <(find "$ADAPTIVE_MODEL" -maxdepth 1 -type f -name 'model-*.safetensors' -size +0c | sort)
if [[ "${#adaptive_shards[@]}" -ne 2 ]]; then
  echo "adaptive model must contain exactly two nonempty safetensor shards" >&2
  exit 1
fi
job_state="$(squeue -h -j "$JOB_ID" -o '%T' | head -1)"
if [[ "$job_state" != "RUNNING" ]]; then
  echo "Slurm job $JOB_ID is not RUNNING: ${job_state:-missing}" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
mkdir "$RUN_DIR"
exec > >(tee "$LOG_PATH") 2>&1

run_command "${PREPARE_COMMAND[@]}"
run_command "${BASE_COMMAND[@]}"
run_command "${ADAPTIVE_COMMAND[@]}"
run_command "${ANALYZE_COMMAND[@]}"
run_command "${VALIDATE_COMMAND[@]}"
