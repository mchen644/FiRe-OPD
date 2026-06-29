#!/bin/bash
set -euo pipefail

# Create FiRe-OPD parquet files where student rollout keeps the raw prompt and
# teacher/ref log-prob uses a TALE-style budget-aware prompt stored in teacher_prompt.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/mchen/miniconda3/envs/verl/bin/python}"
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/verl:${PYTHONPATH:-}"

STUDENT_MODEL="${STUDENT_MODEL:-${REPO_DIR}/models/Qwen3-4B}"
INPUT_ROOT="${INPUT_ROOT:-${REPO_DIR}/data/g-opd}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/data/g-opd-student-raw-teacher-tale-budget}"
BUDGET_CACHE_ROOT="${BUDGET_CACHE_ROOT:-${OUTPUT_ROOT}/budget_records}"
TEACHER_PROMPT_KEY="${TEACHER_PROMPT_KEY:-teacher_prompt}"
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
MIN_BUDGET="${MIN_BUDGET:-128}"
MAX_BUDGET="${MAX_BUDGET:-8192}"
ROUND_TO="${ROUND_TO:-64}"
FALLBACK_BUDGET="${FALLBACK_BUDGET:-2048}"
LIMIT_ROWS="${LIMIT_ROWS:-}"

convert_one() {
  local rel_path="$1"
  local input_file="${INPUT_ROOT}/${rel_path}"
  local output_file="${OUTPUT_ROOT}/${rel_path}"
  local cache_file="${BUDGET_CACHE_ROOT}/${rel_path%.parquet}.jsonl"

  if [ ! -f "${input_file}" ]; then
    echo "ERROR: missing input file: ${input_file}" >&2
    exit 1
  fi

  mkdir -p "$(dirname "${output_file}")" "$(dirname "${cache_file}")"

  local args=(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/tale_budget_parquet.py"
    --input_file "${input_file}"
    --output_file "${output_file}"
    --output_prompt_key "${TEACHER_PROMPT_KEY}"
    --min_budget "${MIN_BUDGET}"
    --max_budget "${MAX_BUDGET}"
    --round_to "${ROUND_TO}"
    --fallback_budget "${FALLBACK_BUDGET}"
  )

  if [ -f "${cache_file}" ]; then
    args+=(--budget_records "${cache_file}")
  else
    args+=(
      --student_model "${STUDENT_MODEL}"
      --save_budget_records "${cache_file}"
      --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}"
    )
  fi

  if [ -n "${LIMIT_ROWS}" ]; then
    args+=(--limit_rows "${LIMIT_ROWS}")
  fi

  echo "Converting ${rel_path}"
  echo "  input : ${input_file}"
  echo "  output: ${output_file}"
  echo "  cache : ${cache_file}"
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${args[@]}"
}

convert_one "DeepMath-103K/train_filtered_level6.parquet"
convert_one "AIME2024/test.parquet"
convert_one "AIME2025/test.parquet"

echo "Done. Output root: ${OUTPUT_ROOT}"
