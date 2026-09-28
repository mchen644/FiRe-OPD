#!/bin/bash
# Rollout-length ESR / hard-truncation ablation with a concise teacher prompt.
#
# Purpose:
#   Test whether directly asking the teacher to be concise improves OPD when
#   supervision is still restricted to the first 20% of the student rollout-length
#   budget. Student rollout remains the normal raw prompt. Teacher/ref log-prob
#   uses a concise prompt without a numeric token budget:
#
#     Solve concisely. Avoid unnecessary explanation. Put your final answer within \boxed{}.
#
#   B_i = round(1.0 * student_response_length)
#   supervised_tokens_i = round(0.2 * B_i)
#   hard truncate = True
#
# Difference vs run_train_tale_budget_rolloutlen_opd.sh:
#   algorithm.tale_budget.teacher_prompt_style=concise
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/mchen/FiRe-OPD}"
cd "${REPO_DIR}"

# Make python3 resolve to the verified veRL environment even in a fresh shell.
export PATH="/home/mchen/miniconda3/envs/verl/bin:${PATH}"

export ROLLOUT_N=1
export TALE_BUDGET_SOURCE=rollout_length
export TALE_ROLLOUT_ALPHA=1.0
export TALE_ESR_BETA=0.2
export TALE_ROLLOUT_MAX_BUDGET=null
export TALE_TEACHER_PROMPT_STYLE=concise
export TALE_MIN_BUDGET=1
export TALE_ROUND_TO=1
export TALE_TRUNCATE_TO_ESR=True
export DATA_ROOT="${DATA_ROOT:-${REPO_DIR}/data/g-opd}"
export TRAIN_DATA="${TRAIN_DATA:-${DATA_ROOT}/DeepMath-103K/train_filtered_level6.parquet}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
export ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-4}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-opd-strong-to-weak-studentraw-teacherconcise-rolloutlen-a1.0-b0.2-hardtrunc-selectn1-4gpu-tp4-refmb4-rollmb4}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_DIR}/checkpoints/${EXPERIMENT_NAME}}"

bash verl/examples/fire_opd/run_opd_strong_to_weak_student_raw_teacher_tale_budget30b.sh \
  algorithm.tale_budget.min_budget=${TALE_MIN_BUDGET} \
  algorithm.tale_budget.round_to=${TALE_ROUND_TO} \
  algorithm.tale_budget.truncate_to_esr=${TALE_TRUNCATE_TO_ESR} \
  algorithm.tale_budget.teacher_prompt_style=${TALE_TEACHER_PROMPT_STYLE} \
  trainer.val_before_train=False \
  trainer.test_freq=-1 \
  "$@"
