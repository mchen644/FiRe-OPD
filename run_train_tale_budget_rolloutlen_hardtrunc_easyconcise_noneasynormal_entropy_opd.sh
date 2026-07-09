#!/usr/bin/env bash
# Binary easy/non-easy difficulty-aware OPD diagnostic.
#
# Behavior:
#   - easy samples: concise teacher prompt + 20% ESR supervision
#   - non-easy samples: normal/original teacher prompt + 20% ESR supervision
#   - hard exploration: continuous entropy bonus on wrong low-confidence samples
#
# There is intentionally no budget-prompt middle bucket. This keeps the ablation
# focused on "Compress Easy, Explore Hard": concise for easy, normal for the rest.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_DIR="${REPO_DIR:-${SCRIPT_DIR}}"
cd "${REPO_DIR}"

# Base Budget20 ESR for every sample. Routing changes prompt style and entropy
# only; it does not widen hard supervision.
export TALE_ESR_BETA="${TALE_ESR_BETA:-0.20}"
export TALE_TRUNCATE_TO_ESR="${TALE_TRUNCATE_TO_ESR:-True}"
export TALE_USE_BUDGET_TEACHER_PROMPT="${TALE_USE_BUDGET_TEACHER_PROMPT:-True}"

# Compress easy samples with concise teacher prompt, but keep ESR at 20%.
export DA_EASY_PROMPT_THRESHOLD="${DA_EASY_PROMPT_THRESHOLD:-0.7}"
export DA_EASY_PROMPT_STYLE="${DA_EASY_PROMPT_STYLE:-concise}"
export DA_MIN_EASY_ESR_BETA="${DA_MIN_EASY_ESR_BETA:-0.20}"
export DA_EASY_ESR_DELTA="${DA_EASY_ESR_DELTA:-0.0}"

# Remove the middle budget bucket: all non-easy samples default to normal prompt.
export DA_DEFAULT_PROMPT_STYLE="${DA_DEFAULT_PROMPT_STYLE:-normal}"
export DA_HARD_PROMPT_THRESHOLD="${DA_HARD_PROMPT_THRESHOLD:-0.5}"
export DA_HARD_PROMPT_STYLE="${DA_HARD_PROMPT_STYLE:-normal}"

# Entropy weight remains difficulty-aware: hard = wrong * (1 - confidence_rank).
export DA_HARD_ENTROPY_COEF="${DA_HARD_ENTROPY_COEF:-0.003}"

export TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:-20}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-opd-budget20-easyconcise-noneasynormal-entropy003-esr20-noprobe}"

bash run_train_tale_budget_rolloutlen_hardtrunc_hardnormal_opd.sh \
  algorithm.rethinking_opd_probe.enabled=False \
  trainer.save_freq="${TRAINER_SAVE_FREQ}" \
  trainer.val_before_train=False \
  trainer.test_freq=-1 \
  "$@"
