#!/bin/bash
set -euo pipefail

# Evaluate the Table-2 FiRe-OPD strong-to-weak checkpoint on the math datasets.
#
# Default target:
#   checkpoints/opd-strong-to-weak-studentraw-teachercod-shot5-4gpu-tp4-refmb4-rollmb4/global_step_50/actor
#
# Behavior:
#   1. Merge the FSDP actor checkpoint to HuggingFace format if needed.
#   2. Run math_eval/eval_math.py through run_eval_math_cod_phase1.sh with the raw/baseline student prompt by default.
#   3. Print a compact summary containing Accuracy, pass@k, and avg_length per dataset.
#
# Common usage inside a 4-GPU Slurm allocation:
#   bash /home/mchen/FiRe-OPD/math_eval/run_eval_math_cod_step100_table2.sh
#
# Useful overrides:
#   SKIP_MERGE=1 bash ...                         # require existing global_step_50_hf
#   FORCE_MERGE=1 bash ...                        # delete/recreate global_step_50_hf
#   GPU_IDS=0,1,2,3 bash ...                      # logical CUDA ids for eval
#   N_SAMPLES=8 bash ...                          # faster smoke eval
#   DATASETS="aime24 aime25" bash ...             # subset eval
#   MAX_MODEL_LEN=32768 bash ...                  # pass through to vLLM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/mchen/miniconda3/envs/verl/bin/python}"
export PYTHONPATH="${REPO_DIR}/verl:${REPO_DIR}:${PYTHONPATH:-}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-opd-strong-to-weak-studentraw-teachercod-shot5-4gpu-tp4-refmb4-rollmb4}"
STEP="${STEP:-50}"
FSDP_CKPT_DIR="${FSDP_CKPT_DIR:-${REPO_DIR}/checkpoints/${EXPERIMENT_NAME}/global_step_${STEP}}"
FSDP_ACTOR_DIR="${FSDP_ACTOR_DIR:-${FSDP_CKPT_DIR}/actor}"
HF_MODEL_DIR="${HF_MODEL_DIR:-${REPO_DIR}/checkpoints/${EXPERIMENT_NAME}/global_step_${STEP}_hf}"
MODEL_PATH="${MODEL_PATH:-${HF_MODEL_DIR}}"

PROMPT_STYLE="${PROMPT_STYLE:-baseline}"
COD_SHOT="${COD_SHOT:-5}"
CHAIN_OF_DRAFT_DIR="${CHAIN_OF_DRAFT_DIR:-$(cd "${REPO_DIR}/.." && pwd)/chain-of-draft}"
NO_EXTRA_PROMPT="${NO_EXTRA_PROMPT:-1}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"

DATASETS="${DATASETS:-aime24 aime25 hmmt25_feb hmmt25_nov math500 minervamath olympiadbench amc2023}"
N_SAMPLES="${N_SAMPLES:-32}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
SEED="${SEED:-42}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"

if [ -n "${SLURM_JOB_ID:-}" ]; then
  DEFAULT_GPU_IDS="0,1,2,3"
else
  DEFAULT_GPU_IDS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
fi
GPU_IDS="${GPU_IDS:-${DEFAULT_GPU_IDS}}"

MODEL_KEY="${MODEL_KEY:-table2_fire_opd_step${STEP}}"
if [ "${PROMPT_STYLE}" = "cod" ]; then
  DEFAULT_MODEL_NAME="${EXPERIMENT_NAME}-step${STEP}-cod-shot${COD_SHOT}"
else
  DEFAULT_MODEL_NAME="${EXPERIMENT_NAME}-step${STEP}-${PROMPT_STYLE}"
fi
MODEL_NAME="${MODEL_NAME:-${DEFAULT_MODEL_NAME}}"
RUN_NAME="${RUN_NAME:-${MODEL_NAME}-n${N_SAMPLES}-seed${SEED}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/cod_step_eval_outputs}"
MISTAKES_ROOT="${MISTAKES_ROOT:-${SCRIPT_DIR}/cod_step_eval_mistakes}"
LOG_ROOT="${LOG_ROOT:-${SCRIPT_DIR}/cod_step_eval_logs}"
SKIP_MERGE="${SKIP_MERGE:-0}"
FORCE_MERGE="${FORCE_MERGE:-0}"

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

if [ "${PROMPT_STYLE}" != "baseline" ] && [ "${PROMPT_STYLE}" != "cod" ]; then
  echo "ERROR: PROMPT_STYLE must be baseline or cod, got ${PROMPT_STYLE}" >&2
  exit 1
fi
if [ ! -d "${CHAIN_OF_DRAFT_DIR}" ]; then
  echo "ERROR: Chain-of-Draft repo not found: ${CHAIN_OF_DRAFT_DIR}" >&2
  exit 1
fi

ensure_hf_model

mkdir -p "${OUTPUT_ROOT}" "${MISTAKES_ROOT}" "${LOG_ROOT}"

cat <<EOF
Eval config:
  MODEL_PATH=${MODEL_PATH}
  MODEL_KEY=${MODEL_KEY}
  RUN_NAME=${RUN_NAME}
  DATASETS=${DATASETS}
  PROMPT_STYLE=${PROMPT_STYLE}
  COD_SHOT=${COD_SHOT}
  CHAIN_OF_DRAFT_DIR=${CHAIN_OF_DRAFT_DIR}
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

RUN_MATRIX=0 \
MODEL_KEY="${MODEL_KEY}" \
MODEL_PATH="${MODEL_PATH}" \
PROMPT_STYLE="${PROMPT_STYLE}" \
COD_SHOT="${COD_SHOT}" \
CHAIN_OF_DRAFT_DIR="${CHAIN_OF_DRAFT_DIR}" \
GPU_IDS="${GPU_IDS}" \
DATASETS="${DATASETS}" \
N_SAMPLES="${N_SAMPLES}" \
MAX_TOKENS="${MAX_TOKENS}" \
TEMPERATURE="${TEMPERATURE}" \
TOP_P="${TOP_P}" \
MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
SEED="${SEED}" \
MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
NO_EXTRA_PROMPT="${NO_EXTRA_PROMPT}" \
ENABLE_THINKING="${ENABLE_THINKING}" \
RUN_NAME="${RUN_NAME}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
MISTAKES_ROOT="${MISTAKES_ROOT}" \
LOG_ROOT="${LOG_ROOT}" \
bash "${SCRIPT_DIR}/run_eval_math_cod_phase1.sh"

echo ""
echo "Summary for RUN_NAME=${RUN_NAME}:"
for dataset in ${DATASETS}; do
  log_file="${LOG_ROOT}/${dataset}-${RUN_NAME}.log"
  if [ -f "${log_file}" ]; then
    echo "===== ${dataset} ====="
    grep -E '^(Accuracy:|pass@k:|avg_length:)' "${log_file}" || true
  else
    echo "===== ${dataset} ====="
    echo "missing log: ${log_file}"
  fi
done
