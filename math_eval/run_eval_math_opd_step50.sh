#!/bin/bash
set -euo pipefail

# Merge and evaluate OPD Table 3 single-teacher step-50 checkpoint.
#
# Default behavior:
#   1. Merge the FSDP actor checkpoint at
#      checkpoints/opd-table3-single-teacher-4b-math/global_step_50/actor
#      into HuggingFace format at
#      checkpoints/opd-table3-single-teacher-4b-math/global_step_50_hf
#      if the HF directory is not already present.
#   2. Evaluate all 8 math benchmarks with the same settings used for the
#      prior FiRe-OPD step-50 eval.
#
# Usage:
#   bash /home/mchen/FiRe-OPD/math_eval/run_eval_math_opd_step50.sh
#
# Useful overrides:
#   SKIP_MERGE=1 bash ...                         # require existing HF model dir
#   FORCE_MERGE=1 bash ...                        # delete/recreate HF model dir
#   GPU_PAIR_A=0,1 GPU_PAIR_B=2,3 bash ...        # override GPU assignment
#   MODEL_PATH=/path/to/hf_model bash ...         # evaluate an already-merged model
#   MODEL_NAME=my-output-name bash ...            # output jsonl basename
#   N_SAMPLES=8 bash ...                          # change number of samples

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/mchen/miniconda3/envs/verl/bin/python}"
export PYTHONPATH="${REPO_DIR}/verl:${PYTHONPATH:-}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-opd-table3-single-teacher-4b-math}"
STEP="${STEP:-50}"
FSDP_CKPT_DIR="${FSDP_CKPT_DIR:-${REPO_DIR}/checkpoints/${EXPERIMENT_NAME}/global_step_${STEP}}"
FSDP_ACTOR_DIR="${FSDP_ACTOR_DIR:-${FSDP_CKPT_DIR}/actor}"
HF_MODEL_DIR="${HF_MODEL_DIR:-${REPO_DIR}/checkpoints/${EXPERIMENT_NAME}/global_step_${STEP}_hf}"

MODEL_PATH="${MODEL_PATH:-${HF_MODEL_DIR}}"
MODEL_NAME="${MODEL_NAME:-${EXPERIMENT_NAME}-step${STEP}}"
SKIP_MERGE="${SKIP_MERGE:-0}"
FORCE_MERGE="${FORCE_MERGE:-0}"

# If running inside a Slurm GPU allocation, CUDA ids are usually logical ids.
# Outside Slurm, default to the same last-four physical GPUs used by the prior
# FiRe-OPD eval script. Override GPU_PAIR_A/GPU_PAIR_B if needed.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  DEFAULT_GPU_PAIR_A="0,1"
  DEFAULT_GPU_PAIR_B="2,3"
else
  DEFAULT_GPU_PAIR_A="6,7"
  DEFAULT_GPU_PAIR_B="8,9"
fi
GPU_PAIR_A="${GPU_PAIR_A:-${DEFAULT_GPU_PAIR_A}}"
GPU_PAIR_B="${GPU_PAIR_B:-${DEFAULT_GPU_PAIR_B}}"

N_SAMPLES="${N_SAMPLES:-32}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
SEED="${SEED:-42}"

cleanup() {
  local pids
  pids="$(jobs -pr || true)"
  if [ -n "${pids}" ]; then
    echo "Stopping background eval processes: ${pids}" >&2
    kill ${pids} 2>/dev/null || true
  fi
}
trap cleanup INT TERM EXIT

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

run_math_eval() {
  local gpu_ids="$1"
  local dataset="$2"
  local input_file="${REPO_DIR}/data/${dataset}/test.jsonl"
  local output_dir="${SCRIPT_DIR}/eval_outputs/${dataset}"
  local output_file="${output_dir}/${MODEL_NAME}.jsonl"

  if [ ! -f "${input_file}" ]; then
    echo "ERROR: input file not found: ${input_file}" >&2
    exit 1
  fi

  mkdir -p "${output_dir}"
  echo "[$(date '+%F %T')] Starting ${dataset} on CUDA_VISIBLE_DEVICES=${gpu_ids}"
  CUDA_VISIBLE_DEVICES="${gpu_ids}" "${PYTHON_BIN}" eval_math.py \
    --input_file "${input_file}" \
    --model_path "${MODEL_PATH}" \
    --output_file "${output_file}" \
    --max_tokens "${MAX_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --max_num_seqs "${MAX_NUM_SEQS}" \
    --n "${N_SAMPLES}" \
    --begin_idx -1 \
    --end_idx -1 \
    --seed "${SEED}" \
    --no_extra_prompt
  echo "[$(date '+%F %T')] Finished ${dataset}; output: ${output_file}"
}

run_pair() {
  local dataset_a="$1"
  local dataset_b="$2"

  run_math_eval "${GPU_PAIR_A}" "${dataset_a}" &
  local pid_a=$!
  run_math_eval "${GPU_PAIR_B}" "${dataset_b}" &
  local pid_b=$!

  wait "${pid_a}"
  wait "${pid_b}"
}

ensure_hf_model

for dataset in aime24 aime25 hmmt25_feb hmmt25_nov math500 minervamath olympiadbench amc2023; do
  mkdir -p "${SCRIPT_DIR}/eval_outputs/${dataset}"
done

echo "MODEL_PATH=${MODEL_PATH}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "FSDP_ACTOR_DIR=${FSDP_ACTOR_DIR}"
echo "HF_MODEL_DIR=${HF_MODEL_DIR}"
echo "GPU_PAIR_A=${GPU_PAIR_A}"
echo "GPU_PAIR_B=${GPU_PAIR_B}"
echo "N_SAMPLES=${N_SAMPLES}, MAX_TOKENS=${MAX_TOKENS}, TEMPERATURE=${TEMPERATURE}, TOP_P=${TOP_P}, SEED=${SEED}"

run_pair aime24 aime25
run_pair hmmt25_feb hmmt25_nov
run_pair math500 minervamath
run_pair olympiadbench amc2023

trap - INT TERM EXIT
echo "All math evaluations done for ${MODEL_NAME}."
