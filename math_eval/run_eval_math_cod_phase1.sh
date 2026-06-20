#!/bin/bash
set -euo pipefail

# Phase 1 CoD direct-eval matrix for OPD compression:
#   student_4b baseline vs CoD
#   teacher_30b baseline vs CoD
#
# This script does not train or merge checkpoints. It evaluates HuggingFace model
# directories directly with math_eval/eval_math.py.
#
# Common usage:
#   # Single run: 4B student with CoD, zero-shot CoD prompt
#   MODEL_KEY=student_4b PROMPT_STYLE=cod COD_SHOT=0 bash math_eval/run_eval_math_cod_phase1.sh
#
#   # Full 2x2 matrix: student/teacher x baseline/CoD
#   RUN_MATRIX=1 COD_SHOT=0 bash math_eval/run_eval_math_cod_phase1.sh
#
# Useful overrides:
#   DATASETS="aime24 aime25"              # subset of benchmarks
#   N_SAMPLES=8                            # vLLM n per problem
#   GPU_IDS=0,1,2,3                        # CUDA_VISIBLE_DEVICES for every dataset in this run
#   MAX_MODEL_LEN=32768                    # pass through to vLLM
#   MODEL_PATH=/path/to/model MODEL_KEY=x  # custom model path/name

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/mchen/miniconda3/envs/verl/bin/python}"
export PYTHONPATH="${REPO_DIR}/verl:${REPO_DIR}:${PYTHONPATH:-}"

CHAIN_OF_DRAFT_DIR="${CHAIN_OF_DRAFT_DIR:-$(cd "${REPO_DIR}/.." && pwd)/chain-of-draft}"

RUN_MATRIX="${RUN_MATRIX:-0}"
if [ "${RUN_MATRIX}" = "1" ]; then
  MODEL_KEYS="${MODEL_KEYS:-student_4b teacher_30b}"
  PROMPT_STYLES="${PROMPT_STYLES:-baseline cod}"
  for matrix_model_key in ${MODEL_KEYS}; do
    for matrix_prompt_style in ${PROMPT_STYLES}; do
      echo "==== Matrix run: MODEL_KEY=${matrix_model_key}, PROMPT_STYLE=${matrix_prompt_style} ===="
      RUN_MATRIX=0 MODEL_KEY="${matrix_model_key}" PROMPT_STYLE="${matrix_prompt_style}" \
        bash "${SCRIPT_PATH}"
    done
  done
  exit 0
fi

MODEL_KEY="${MODEL_KEY:-student_4b}"
PROMPT_STYLE="${PROMPT_STYLE:-baseline}"
COD_SHOT="${COD_SHOT:-0}"  # -1 means all few-shot examples from Chain-of-Draft's YAML

case "${MODEL_KEY}" in
  student_4b)
    DEFAULT_MODEL_PATH="${REPO_DIR}/models/Qwen3-4B"
    DEFAULT_GPU_IDS="${DEFAULT_GPU_IDS:-0}"
    ;;
  teacher_30b)
    DEFAULT_MODEL_PATH="${REPO_DIR}/models/Qwen3-30B-A3B-Instruct-2507"
    if [ -n "${SLURM_JOB_ID:-}" ]; then
      DEFAULT_GPU_IDS="${DEFAULT_GPU_IDS:-0,1,2,3}"
    else
      DEFAULT_GPU_IDS="${DEFAULT_GPU_IDS:-6,7,8,9}"
    fi
    ;;
  *)
    if [ -z "${MODEL_PATH:-}" ]; then
      echo "ERROR: unknown MODEL_KEY=${MODEL_KEY}. Set MODEL_PATH for custom keys." >&2
      exit 1
    fi
    DEFAULT_MODEL_PATH="${MODEL_PATH}"
    DEFAULT_GPU_IDS="${DEFAULT_GPU_IDS:-0}"
    ;;
esac

MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
GPU_IDS="${GPU_IDS:-${DEFAULT_GPU_IDS}}"

DATASETS="${DATASETS:-aime24 aime25 hmmt25_feb hmmt25_nov math500 minervamath olympiadbench amc2023}"
N_SAMPLES="${N_SAMPLES:-32}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
SEED="${SEED:-42}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"
NO_EXTRA_PROMPT="${NO_EXTRA_PROMPT:-1}"

RUN_NAME="${RUN_NAME:-${MODEL_KEY}-${PROMPT_STYLE}-shot${COD_SHOT}-n${N_SAMPLES}-seed${SEED}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/cod_phase1_outputs}"
MISTAKES_ROOT="${MISTAKES_ROOT:-${SCRIPT_DIR}/cod_phase1_mistakes}"
LOG_ROOT="${LOG_ROOT:-${SCRIPT_DIR}/cod_phase1_logs}"
mkdir -p "${OUTPUT_ROOT}" "${MISTAKES_ROOT}" "${LOG_ROOT}"

if [ ! -d "${CHAIN_OF_DRAFT_DIR}" ]; then
  echo "ERROR: Chain-of-Draft repo not found: ${CHAIN_OF_DRAFT_DIR}" >&2
  exit 1
fi
if [ ! -d "${MODEL_PATH}" ]; then
  echo "ERROR: MODEL_PATH not found: ${MODEL_PATH}" >&2
  exit 1
fi
if [ "${PROMPT_STYLE}" != "baseline" ] && [ "${PROMPT_STYLE}" != "cod" ]; then
  echo "ERROR: PROMPT_STYLE must be baseline or cod, got ${PROMPT_STYLE}" >&2
  exit 1
fi

run_one_dataset() {
  local dataset="$1"
  local input_file="${REPO_DIR}/data/${dataset}/test.jsonl"
  local output_dir="${OUTPUT_ROOT}/${dataset}"
  local mistakes_dir="${MISTAKES_ROOT}/${dataset}"
  local log_file="${LOG_ROOT}/${dataset}-${RUN_NAME}.log"
  local output_file="${output_dir}/${RUN_NAME}.jsonl"
  local mistakes_file="${mistakes_dir}/${RUN_NAME}.jsonl"

  if [ ! -f "${input_file}" ]; then
    echo "ERROR: input file not found: ${input_file}" >&2
    exit 1
  fi
  mkdir -p "${output_dir}" "${mistakes_dir}"

  local cmd=(
    "${PYTHON_BIN}" eval_math.py
    --input_file "${input_file}"
    --model_path "${MODEL_PATH}"
    --output_file "${output_file}"
    --max_tokens "${MAX_TOKENS}"
    --temperature "${TEMPERATURE}"
    --top_p "${TOP_P}"
    --max_num_seqs "${MAX_NUM_SEQS}"
    --n "${N_SAMPLES}"
    --begin_idx -1
    --end_idx -1
    --seed "${SEED}"
    --prompt_style "${PROMPT_STYLE}"
    --cod_repo_dir "${CHAIN_OF_DRAFT_DIR}"
    --cod_shot "${COD_SHOT}"
    --mistakes_file "${mistakes_file}"
  )

  if [ "${NO_EXTRA_PROMPT}" = "1" ]; then
    cmd+=(--no_extra_prompt)
  fi
  if [ "${ENABLE_THINKING}" = "1" ]; then
    cmd+=(--enable_thinking)
  fi
  if [ -n "${MAX_MODEL_LEN}" ]; then
    cmd+=(--max_model_len "${MAX_MODEL_LEN}")
  fi

  echo "[$(date '+%F %T')] dataset=${dataset} model_key=${MODEL_KEY} prompt_style=${PROMPT_STYLE} cod_shot=${COD_SHOT} gpu_ids=${GPU_IDS}"
  echo "  output_file=${output_file}"
  echo "  mistakes_file=${mistakes_file}"
  echo "  log_file=${log_file}"
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${cmd[@]}" 2>&1 | tee "${log_file}"
}

echo "MODEL_KEY=${MODEL_KEY}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "PROMPT_STYLE=${PROMPT_STYLE}"
echo "COD_SHOT=${COD_SHOT}"
echo "CHAIN_OF_DRAFT_DIR=${CHAIN_OF_DRAFT_DIR}"
echo "GPU_IDS=${GPU_IDS}"
echo "DATASETS=${DATASETS}"
echo "N_SAMPLES=${N_SAMPLES}, MAX_TOKENS=${MAX_TOKENS}, TEMPERATURE=${TEMPERATURE}, TOP_P=${TOP_P}, SEED=${SEED}"

for dataset in ${DATASETS}; do
  run_one_dataset "${dataset}"
done

echo "All CoD phase-1 eval jobs finished for RUN_NAME=${RUN_NAME}."
