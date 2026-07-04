#!/usr/bin/env bash
# Rollout-length ESR / hard-truncation ablation with budget-aware teacher prompts (budget20).
#
# Uses rollout-length budgets and hard truncation (ESR mask) while enabling
# Rethinking OPD probe CSV logging.
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
export TALE_MIN_BUDGET=1
export TALE_ROUND_TO=1
export TALE_TRUNCATE_TO_ESR=True
export TALE_USE_BUDGET_TEACHER_PROMPT=True
export DATA_ROOT="${DATA_ROOT:-${REPO_DIR}/data/g-opd}"
export TRAIN_DATA="${TRAIN_DATA:-${DATA_ROOT}/DeepMath-103K/train_filtered_level6.parquet}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
export ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-4}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-opd-budget20-rethinking-probe}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_DIR}/checkpoints/${EXPERIMENT_NAME}}"

bash verl/examples/fire_opd/run_opd_strong_to_weak_student_raw_teacher_tale_budget30b.sh \
  algorithm.tale_budget.enabled=True \
  algorithm.tale_budget.source=${TALE_BUDGET_SOURCE} \
  algorithm.tale_budget.rollout_length_alpha=${TALE_ROLLOUT_ALPHA} \
  algorithm.tale_budget.esr_beta=${TALE_ESR_BETA} \
  algorithm.tale_budget.rollout_length_max_budget=${TALE_ROLLOUT_MAX_BUDGET} \
  algorithm.tale_budget.truncate_to_esr=${TALE_TRUNCATE_TO_ESR} \
  algorithm.tale_budget.use_budget_teacher_prompt=${TALE_USE_BUDGET_TEACHER_PROMPT} \
  algorithm.tale_budget.min_budget=${TALE_MIN_BUDGET} \
  algorithm.tale_budget.round_to=${TALE_ROUND_TO} \
  algorithm.tale_budget.teacher_prompt_style=auto \
  algorithm.rethinking_opd_probe.enabled=True \
  algorithm.rethinking_opd_probe.top_k=16 \
  algorithm.rethinking_opd_probe.chunk_size=1024 \
  algorithm.rethinking_opd_probe.csv_path=math_eval/opd_training_dynamics_audit/training_probe_budget20.csv \
  trainer.experiment_name=${EXPERIMENT_NAME} \
  trainer.val_before_train=False \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  "$@"
