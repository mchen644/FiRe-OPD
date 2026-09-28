#!/bin/bash
set -euo pipefail

# Evaluate FiRe-OPD Table 3 math benchmarks on the last four physical GPUs.
# Default model is the merged HuggingFace checkpoint from global_step_50.
# Run from anywhere:
#   bash /home/mchen/FiRe-OPD/math_eval/run_eval_math_step50_last4gpus.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SCRIPT_DIR}"

MODEL_PATH="${MODEL_PATH:-${REPO_DIR}/checkpoints/fire-opd-table3-single-teacher-4b-math/global_step_50_hf}"
MODEL_NAME="${MODEL_NAME:-fire-opd-table3-single-teacher-4b-math-step50}"
PYTHON_BIN="${PYTHON_BIN:-/home/mchen/miniconda3/envs/verl/bin/python}"

# Physical GPU ids. Override if needed, e.g. GPU_PAIR_A=4,5 GPU_PAIR_B=6,7 bash ...
GPU_PAIR_A="${GPU_PAIR_A:-6,7}"
GPU_PAIR_B="${GPU_PAIR_B:-8,9}"

N_SAMPLES="${N_SAMPLES:-32}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
SEED="${SEED:-42}"

if [ ! -d "${MODEL_PATH}" ]; then
  echo "ERROR: MODEL_PATH does not exist: ${MODEL_PATH}" >&2
  echo "Please merge the FSDP checkpoint first, or set MODEL_PATH=/path/to/hf_model." >&2
  exit 1
fi

for dataset in aime24 aime25 hmmt25_feb hmmt25_nov math500 minervamath olympiadbench amc2023; do
  mkdir -p "${SCRIPT_DIR}/eval_outputs/${dataset}"
done

run_math_eval() {
  local gpu_ids="$1"
  local dataset="$2"
  local input_file="${REPO_DIR}/data/${dataset}/test.jsonl"
  local output_file="${SCRIPT_DIR}/eval_outputs/${dataset}/${MODEL_NAME}.jsonl"

  echo "[$(date '+%F %T')] Starting ${dataset} on GPUs ${gpu_ids}"
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

echo "MODEL_PATH=${MODEL_PATH}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "GPU_PAIR_A=${GPU_PAIR_A}"
echo "GPU_PAIR_B=${GPU_PAIR_B}"
echo "N_SAMPLES=${N_SAMPLES}, MAX_TOKENS=${MAX_TOKENS}, TEMPERATURE=${TEMPERATURE}, TOP_P=${TOP_P}, SEED=${SEED}"

run_pair aime24 aime25
run_pair hmmt25_feb hmmt25_nov
run_pair math500 minervamath
run_pair olympiadbench amc2023

echo "All math evaluations done for ${MODEL_NAME}."
