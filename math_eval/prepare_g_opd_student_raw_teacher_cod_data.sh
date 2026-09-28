#!/bin/bash
set -euo pipefail

# Prepare OPD math parquet files where:
#   - prompt keeps the original student rollout prompt
#   - teacher_prompt stores the Chain-of-Draft prompt for teacher/ref log-prob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/mchen/miniconda3/envs/verl/bin/python}"
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/verl:${PYTHONPATH:-}"

CHAIN_OF_DRAFT_DIR="${CHAIN_OF_DRAFT_DIR:-$(cd "${REPO_DIR}/.." && pwd)/chain-of-draft}"
COD_SHOT="${COD_SHOT:-5}"
COD_TASK="${COD_TASK:-gsm8k}"

INPUT_ROOT="${INPUT_ROOT:-${REPO_DIR}/data/g-opd}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/data/g-opd-student-raw-teacher-cod-shot${COD_SHOT}}"
TEACHER_PROMPT_KEY="${TEACHER_PROMPT_KEY:-teacher_prompt}"

convert_one() {
  local input_file="$1"
  local output_file="$2"
  if [ ! -f "${input_file}" ]; then
    echo "ERROR: input file not found: ${input_file}" >&2
    exit 1
  fi
  "${PYTHON_BIN}" -m math_eval.cod_parquet \
    --input_file "${input_file}" \
    --output_file "${output_file}" \
    --cod_repo_dir "${CHAIN_OF_DRAFT_DIR}" \
    --cod_task "${COD_TASK}" \
    --cod_shot "${COD_SHOT}" \
    --output_prompt_key "${TEACHER_PROMPT_KEY}"
}

mkdir -p "${OUTPUT_ROOT}/DeepMath-103K" "${OUTPUT_ROOT}/AIME2024" "${OUTPUT_ROOT}/AIME2025"

convert_one \
  "${INPUT_ROOT}/DeepMath-103K/train_filtered_level6.parquet" \
  "${OUTPUT_ROOT}/DeepMath-103K/train_filtered_level6.parquet"

convert_one \
  "${INPUT_ROOT}/AIME2024/test.parquet" \
  "${OUTPUT_ROOT}/AIME2024/test.parquet"

convert_one \
  "${INPUT_ROOT}/AIME2025/test.parquet" \
  "${OUTPUT_ROOT}/AIME2025/test.parquet"

cat <<EOF
Student-raw / teacher-CoD OPD data prepared:
  OUTPUT_ROOT=${OUTPUT_ROOT}
  COD_SHOT=${COD_SHOT}
  COD_TASK=${COD_TASK}
  TEACHER_PROMPT_KEY=${TEACHER_PROMPT_KEY}
EOF
