#!/bin/bash
set -euo pipefail

# Dataset-major Phase 1 CoD direct-eval order.
#
# Order:
#   for each dataset:
#     for each model key:
#       baseline, then CoD shot=5
#
# This is useful when we want an immediate per-dataset comparison instead of
# running one setting across all datasets first.
#
# Defaults run all 4 settings on all Table-3 math datasets using 4 visible GPUs:
#   student_4b baseline
#   student_4b cod shot5
#   teacher_30b baseline
#   teacher_30b cod shot5
#
# Resume behavior:
#   If the corresponding log already contains "Accuracy:", this script skips it.
#   Set FORCE_RERUN=1 to rerun completed entries.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASETS="${DATASETS:-aime24 aime25 hmmt25_feb hmmt25_nov math500 minervamath olympiadbench amc2023}"
MODEL_KEYS="${MODEL_KEYS:-student_4b teacher_30b}"
PROMPT_STYLES="${PROMPT_STYLES:-baseline cod}"
COD_SHOT="${COD_SHOT:-5}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
N_SAMPLES="${N_SAMPLES:-32}"
SEED="${SEED:-42}"
FORCE_RERUN="${FORCE_RERUN:-0}"
LOG_ROOT="${LOG_ROOT:-${SCRIPT_DIR}/cod_phase1_logs}"

run_name_for() {
  local model_key="$1"
  local prompt_style="$2"
  echo "${model_key}-${prompt_style}-shot${COD_SHOT}-n${N_SAMPLES}-seed${SEED}"
}

is_completed() {
  local dataset="$1"
  local model_key="$2"
  local prompt_style="$3"
  local run_name
  run_name="$(run_name_for "${model_key}" "${prompt_style}")"
  local log_file="${LOG_ROOT}/${dataset}-${run_name}.log"
  [ -f "${log_file}" ] && grep -q '^Accuracy:' "${log_file}"
}

for dataset in ${DATASETS}; do
  echo "================ DATASET: ${dataset} ================"
  for model_key in ${MODEL_KEYS}; do
    for prompt_style in ${PROMPT_STYLES}; do
      run_name="$(run_name_for "${model_key}" "${prompt_style}")"
      if [ "${FORCE_RERUN}" != "1" ] && is_completed "${dataset}" "${model_key}" "${prompt_style}"; then
        echo "[SKIP completed] dataset=${dataset} run=${run_name}"
        continue
      fi

      echo "[RUN] dataset=${dataset} model_key=${model_key} prompt_style=${prompt_style} cod_shot=${COD_SHOT} gpu_ids=${GPU_IDS}"
      RUN_MATRIX=0 \
      DATASETS="${dataset}" \
      MODEL_KEY="${model_key}" \
      PROMPT_STYLE="${prompt_style}" \
      COD_SHOT="${COD_SHOT}" \
      GPU_IDS="${GPU_IDS}" \
      N_SAMPLES="${N_SAMPLES}" \
      SEED="${SEED}" \
      bash "${SCRIPT_DIR}/run_eval_math_cod_phase1.sh"
    done
  done
done

echo "Dataset-major Phase 1 CoD eval finished."
