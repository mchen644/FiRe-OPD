#!/usr/bin/env bash
# Easy-concise / hard-normal entropy Budget20 OPD diagnostic.
#
# Behavior:
#   - easy samples: concise teacher prompt + 20% ESR supervision
#   - hard samples: normal/original teacher prompt + 20% ESR supervision
#                   + difficulty-aware current-policy entropy bonus
#   - middle samples: budget teacher prompt + 20% ESR supervision
#
# This tests whether compressing easy samples while encouraging hard-sample
# exploration can retain the pass@k gains of hard-normal routing without the
# length inflation from hard ESR50.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_DIR="${REPO_DIR:-${SCRIPT_DIR}}"
cd "${REPO_DIR}"

# Base Budget20 ESR for all samples. Easy routing changes only the prompt;
# hard routing changes prompt + entropy, not supervision width.
export TALE_ESR_BETA="${TALE_ESR_BETA:-0.20}"
export TALE_TRUNCATE_TO_ESR="${TALE_TRUNCATE_TO_ESR:-True}"
export TALE_USE_BUDGET_TEACHER_PROMPT="${TALE_USE_BUDGET_TEACHER_PROMPT:-True}"

# Compress easy samples with the concise prompt, but keep ESR at 20%.
export DA_EASY_PROMPT_THRESHOLD="${DA_EASY_PROMPT_THRESHOLD:-0.7}"
export DA_EASY_PROMPT_STYLE="${DA_EASY_PROMPT_STYLE:-concise}"
export DA_MIN_EASY_ESR_BETA="${DA_MIN_EASY_ESR_BETA:-0.20}"
export DA_EASY_ESR_DELTA="${DA_EASY_ESR_DELTA:-0.0}"

# Keep middle samples on the budget prompt.
export DA_DEFAULT_PROMPT_STYLE="${DA_DEFAULT_PROMPT_STYLE:-budget}"

# Explore hard samples with normal teacher prompt plus current-policy entropy.
export DA_HARD_PROMPT_THRESHOLD="${DA_HARD_PROMPT_THRESHOLD:-0.5}"
export DA_HARD_PROMPT_STYLE="${DA_HARD_PROMPT_STYLE:-normal}"
export DA_HARD_ENTROPY_COEF="${DA_HARD_ENTROPY_COEF:-0.003}"

export TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:-20}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-opd-budget20-easyconcise-hardnormal-entropy003-esr20-noprobe}"

bash run_train_tale_budget_rolloutlen_hardtrunc_hardnormal_opd.sh \
  algorithm.rethinking_opd_probe.enabled=False \
  trainer.save_freq="${TRAINER_SAVE_FREQ}" \
  trainer.val_before_train=False \
  trainer.test_freq=-1 \
  "$@"
