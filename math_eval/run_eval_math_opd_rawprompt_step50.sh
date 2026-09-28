#!/bin/bash
set -euo pipefail

# Evaluate Original OPD strong-to-weak step-50 checkpoint on the 8 math datasets.
#
# This wrapper is intentionally NORMAL-PROMPT ONLY:
#   - hard-runs eval_math.py with --prompt_style baseline
#   - never passes alternate prompt-construction arguments
#   - rejects any non-baseline PROMPT_STYLE if accidentally provided
#   - uses --no_extra_prompt by default so the prompt is exactly the normal
#     problem text already stored in data/<dataset>/test.jsonl.
#
# Default checkpoint:
#   checkpoints/opd-strong-to-weak-rawprompt-4gpu-tp4-refmb4-rollmb4/global_step_50/actor
#
# Usage inside an srun 4-GPU allocation:
#   bash /home/mchen/FiRe-OPD/math_eval/run_eval_math_opd_rawprompt_step50.sh
#
# Useful overrides:
#   SKIP_MERGE=1 bash ...             # require existing global_step_50_hf
#   FORCE_MERGE=1 bash ...            # delete/recreate global_step_50_hf
#   GPU_IDS=0,1,2,3 bash ...          # override CUDA_VISIBLE_DEVICES for eval
#   N_SAMPLES=8 bash ...              # faster smoke eval
#   DATASETS="aime24 aime25" bash ... # subset eval
#   MAX_MODEL_LEN=32768 bash ...      # pass through to vLLM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/mchen/miniconda3/envs/verl/bin/python}"
export PYTHONPATH="${REPO_DIR}/verl:${REPO_DIR}:${PYTHONPATH:-}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-opd-strong-to-weak-rawprompt-4gpu-tp4-refmb4-rollmb4}"
STEP="${STEP:-50}"
FSDP_CKPT_DIR="${FSDP_CKPT_DIR:-${REPO_DIR}/checkpoints/${EXPERIMENT_NAME}/global_step_${STEP}}"
FSDP_ACTOR_DIR="${FSDP_ACTOR_DIR:-${FSDP_CKPT_DIR}/actor}"
HF_MODEL_DIR="${HF_MODEL_DIR:-${REPO_DIR}/checkpoints/${EXPERIMENT_NAME}/global_step_${STEP}_hf}"
MODEL_PATH="${MODEL_PATH:-${HF_MODEL_DIR}}"

MODEL_KEY="${MODEL_KEY:-opd_strong_to_weak_raw_step${STEP}}"
MODEL_NAME="${MODEL_NAME:-${EXPERIMENT_NAME}-step${STEP}-baseline}"
RUN_NAME="${RUN_NAME:-${MODEL_NAME}-n${N_SAMPLES:-32}-seed${SEED:-42}}"

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
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/opd_rawprompt_step_eval_outputs}"
MISTAKES_ROOT="${MISTAKES_ROOT:-${SCRIPT_DIR}/opd_rawprompt_step_eval_mistakes}"
LOG_ROOT="${LOG_ROOT:-${SCRIPT_DIR}/opd_rawprompt_step_eval_logs}"
SKIP_MERGE="${SKIP_MERGE:-0}"
FORCE_MERGE="${FORCE_MERGE:-0}"

if [ "${PROMPT_STYLE:-baseline}" != "baseline" ]; then
  echo "ERROR: This wrapper is normal-prompt only. Use PROMPT_STYLE=baseline or unset PROMPT_STYLE." >&2
  exit 1
fi

if [ "${NO_EXTRA_PROMPT}" != "1" ]; then
  echo "ERROR: This wrapper expects NO_EXTRA_PROMPT=1 so eval uses the normal prompt exactly as stored in data/*." >&2
  exit 1
fi

ensure_hf_model() {
  if [ "${MODEL_PATH}" != "${HF_MODEL_DIR}" ]; then
    if [ ! -d "${MODEL_PATH}" ]; then
      echo "ERROR: MODEL_PATH does not exist: ${MODEL_PATH}" >&2
      exit 1
    fi
    echo "Using MODEL_PATH override; skipping automatic merge: ${MODEL_PATH}"
    return
  fi

  if [ "${SKIP_MERGE}" = "1" ]; then
    if [ ! -f "${MODEL_PATH}/model.safetensors.index.json" ] && [ ! -f "${MODEL_PATH}/model.safetensors" ]; then
      echo "ERROR: SKIP_MERGE=1 but HF model files were not found in: ${MODEL_PATH}" >&2
      exit 1
    fi
    echo "SKIP_MERGE=1 and HF model exists: ${MODEL_PATH}"
    return
  fi

  if [ "${FORCE_MERGE}" = "1" ] && [ -d "${HF_MODEL_DIR}" ]; then
    echo "FORCE_MERGE=1: removing existing HF dir: ${HF_MODEL_DIR}"
    rm -rf "${HF_MODEL_DIR}"
  fi

  if [ -f "${HF_MODEL_DIR}/model.safetensors.index.json" ] || [ -f "${HF_MODEL_DIR}/model.safetensors" ]; then
    echo "HF model already exists; skipping merge: ${HF_MODEL_DIR}"
    return
  fi

  if [ ! -d "${FSDP_ACTOR_DIR}" ]; then
    echo "ERROR: FSDP actor checkpoint not found: ${FSDP_ACTOR_DIR}" >&2
    exit 1
  fi

  if [ -d "${HF_MODEL_DIR}" ] && [ -n "$(find "${HF_MODEL_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
    echo "ERROR: HF target dir exists but does not look complete: ${HF_MODEL_DIR}" >&2
    echo "Set FORCE_MERGE=1 to delete and recreate it." >&2
    exit 1
  fi

  echo "Merging FSDP checkpoint to HuggingFace format..."
  echo "  local_dir : ${FSDP_ACTOR_DIR}"
  echo "  target_dir: ${HF_MODEL_DIR}"
  cd "${REPO_DIR}"
  "${PYTHON_BIN}" -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${FSDP_ACTOR_DIR}" \
    --target_dir "${HF_MODEL_DIR}"
  cd "${SCRIPT_DIR}"
}

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
  mkdir -p "${output_dir}" "${mistakes_dir}" "${LOG_ROOT}"

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
    --prompt_style baseline
    --no_extra_prompt
    --mistakes_file "${mistakes_file}"
  )

  if [ "${ENABLE_THINKING}" = "1" ]; then
    cmd+=(--enable_thinking)
  fi
  if [ -n "${MAX_MODEL_LEN}" ]; then
    cmd+=(--max_model_len "${MAX_MODEL_LEN}")
  fi

  echo "[$(date '+%F %T')] dataset=${dataset} model_key=${MODEL_KEY} prompt_style=baseline gpu_ids=${GPU_IDS}"
  echo "  output_file=${output_file}"
  echo "  mistakes_file=${mistakes_file}"
  echo "  log_file=${log_file}"
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${cmd[@]}" 2>&1 | tee "${log_file}"
}

ensure_hf_model
mkdir -p "${OUTPUT_ROOT}" "${MISTAKES_ROOT}" "${LOG_ROOT}"

cat <<EOF
Eval config:
  MODEL_PATH=${MODEL_PATH}
  MODEL_KEY=${MODEL_KEY}
  RUN_NAME=${RUN_NAME}
  DATASETS=${DATASETS}
  PROMPT_STYLE=baseline
  NO_EXTRA_PROMPT=1
  GPU_IDS=${GPU_IDS}
  N_SAMPLES=${N_SAMPLES}
  MAX_TOKENS=${MAX_TOKENS}
  TEMPERATURE=${TEMPERATURE}
  TOP_P=${TOP_P}
  MAX_NUM_SEQS=${MAX_NUM_SEQS}
  SEED=${SEED}
  OUTPUT_ROOT=${OUTPUT_ROOT}
  LOG_ROOT=${LOG_ROOT}
EOF

for dataset in ${DATASETS}; do
  run_one_dataset "${dataset}"
done

echo ""
echo "Summary for RUN_NAME=${RUN_NAME}:"
for dataset in ${DATASETS}; do
  log_file="${LOG_ROOT}/${dataset}-${RUN_NAME}.log"
  echo "===== ${dataset} ====="
  if [ -f "${log_file}" ]; then
    grep -E '^(Accuracy:|pass@k:|avg_length:)' "${log_file}" || true
  else
    echo "missing log: ${log_file}"
  fi
done

echo "All normal-prompt OPD rawprompt evaluations done for ${RUN_NAME}."
