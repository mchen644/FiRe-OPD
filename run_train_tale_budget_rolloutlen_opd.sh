#!/bin/bash
# Rollout-length TALE-budget OPD training.
# Student rollout uses the normal prompt. Teacher/ref prompt budget is derived from
# the current student rollout length:
#   L_i = response length
#   B_i = round(1.0 * L_i) = L_i
#   N_i = 0.2 * B_i
# Actor OPD loss is supervised only on the first N_i response tokens.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/mchen/FiRe-OPD}"
cd "${REPO_DIR}"

# Make python3 resolve to the verified veRL environment even in a fresh shell.
export PATH="/home/mchen/miniconda3/envs/verl/bin:${PATH}"

# Fixed settings for this run.
export ROLLOUT_N=1
export TALE_BUDGET_SOURCE=rollout_length
export TALE_ROLLOUT_ALPHA=1.0
export TALE_ESR_BETA=0.2
export TALE_ROLLOUT_MAX_BUDGET=null
export TALE_MIN_BUDGET=1
export TALE_ROUND_TO=1
export TALE_TRUNCATE_TO_ESR=True
export DATA_ROOT="${DATA_ROOT:-${REPO_DIR}/data/g-opd}"
export TRAIN_DATA="${TRAIN_DATA:-${DATA_ROOT}/DeepMath-103K/train_filtered_level6.parquet}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
export ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-4}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-opd-strong-to-weak-studentraw-teachertale-rolloutlen-a1.0-b0.2-hardtrunc-selectn1-4gpu-tp4-refmb4-rollmb4}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_DIR}/checkpoints/${EXPERIMENT_NAME}}"

bash verl/examples/fire_opd/run_opd_strong_to_weak_student_raw_teacher_tale_budget30b.sh \
  algorithm.tale_budget.min_budget=${TALE_MIN_BUDGET} \
  algorithm.tale_budget.round_to=${TALE_ROUND_TO} \
  algorithm.tale_budget.truncate_to_esr=${TALE_TRUNCATE_TO_ESR} \
  trainer.val_before_train=False \
  trainer.test_freq=-1 \
  "$@"
