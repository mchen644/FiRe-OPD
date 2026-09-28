#!/bin/bash
set -euo pipefail

# Ready-to-run normal-prompt evaluation for the online TALE budget OPD step-50 checkpoint.
#
# Run from anywhere:
#   bash /home/mchen/FiRe-OPD/math_eval/run_eval_math_tale_budget_step50_ready.sh
#
# This script intentionally hardcodes the eval settings used for the step-50 comparison.
# It delegates the merge/eval mechanics to run_eval_math_tale_budget_step50_table2.sh.

REPO_DIR="/home/mchen/FiRe-OPD"
BASE_SCRIPT="${REPO_DIR}/math_eval/run_eval_math_tale_budget_step50_table2.sh"

# ----- Checkpoint -----
export EXPERIMENT_NAME="opd-strong-to-weak-studentraw-teachertale-budget-selectn1-4gpu-tp4-refmb4-rollmb4"
export STEP="50"
export FSDP_CKPT_DIR="${REPO_DIR}/checkpoints/${EXPERIMENT_NAME}/global_step_${STEP}"
export FSDP_ACTOR_DIR="${FSDP_CKPT_DIR}/actor"
export HF_MODEL_DIR="${REPO_DIR}/checkpoints/${EXPERIMENT_NAME}/global_step_${STEP}_hf"
export MODEL_PATH="${HF_MODEL_DIR}"

# Merge FSDP -> HF automatically if global_step_50_hf is absent.
export SKIP_MERGE="0"
export FORCE_MERGE="0"

# ----- Eval protocol: normal prompt only -----
export PROMPT_STYLE="baseline"
export NO_EXTRA_PROMPT="1"
export ENABLE_THINKING="0"

# Table-2 full math suite used by the wrapper.
export DATASETS="aime24 aime25 hmmt25_feb hmmt25_nov math500 minervamath olympiadbench amc2023"

# Sampling / generation settings.
export N_SAMPLES="32"
export MAX_TOKENS="16384"
export TEMPERATURE="1.0"
export TOP_P="1.0"
export MAX_NUM_SEQS="256"
export SEED="42"
# Leave empty unless vLLM needs a smaller context for the local environment.
export MAX_MODEL_LEN=""

# GPU setting.
# Inside an srun allocation, SLURM usually remaps the allocated 4 GPUs to
# CUDA_VISIBLE_DEVICES=0,1,2,3. Inherit that value so eval_math.py sees all
# allocated GPUs and sets tensor_parallel_size=torch.cuda.device_count()=4.
# If CUDA_VISIBLE_DEVICES is unset, fall back to local GPUs 0,1,2,3.
export GPU_IDS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# Output locations.
export MODEL_KEY="tale_budget_opd_step50"
export MODEL_NAME="online-tale-budget-opd-step50-normalprompt"
export RUN_NAME="${MODEL_NAME}-n${N_SAMPLES}-seed${SEED}"
export OUTPUT_ROOT="${REPO_DIR}/math_eval/tale_budget_step50_table2_eval_outputs"
export MISTAKES_ROOT="${REPO_DIR}/math_eval/tale_budget_step50_table2_eval_mistakes"
export LOG_ROOT="${REPO_DIR}/math_eval/tale_budget_step50_table2_eval_logs"

if [ ! -d "${FSDP_ACTOR_DIR}" ]; then
  echo "ERROR: missing FSDP actor checkpoint: ${FSDP_ACTOR_DIR}" >&2
  exit 1
fi

cat <<EOF
Ready eval settings:
  BASE_SCRIPT=${BASE_SCRIPT}
  EXPERIMENT_NAME=${EXPERIMENT_NAME}
  STEP=${STEP}
  FSDP_ACTOR_DIR=${FSDP_ACTOR_DIR}
  HF_MODEL_DIR=${HF_MODEL_DIR}
  MODEL_PATH=${MODEL_PATH}
  SKIP_MERGE=${SKIP_MERGE}
  FORCE_MERGE=${FORCE_MERGE}
  PROMPT_STYLE=${PROMPT_STYLE}
  NO_EXTRA_PROMPT=${NO_EXTRA_PROMPT}
  ENABLE_THINKING=${ENABLE_THINKING}
  DATASETS=${DATASETS}
  N_SAMPLES=${N_SAMPLES}
  MAX_TOKENS=${MAX_TOKENS}
  TEMPERATURE=${TEMPERATURE}
  TOP_P=${TOP_P}
  MAX_NUM_SEQS=${MAX_NUM_SEQS}
  SEED=${SEED}
  GPU_IDS=${GPU_IDS}
  OUTPUT_ROOT=${OUTPUT_ROOT}
  LOG_ROOT=${LOG_ROOT}
EOF

bash "${BASE_SCRIPT}"
