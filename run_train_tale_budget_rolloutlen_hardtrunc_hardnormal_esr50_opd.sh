#!/usr/bin/env bash
# Hard-normal ESR50 Budget20 OPD diagnostic.
#
# Behavior:
#   - non-hard samples: budget teacher prompt + 20% ESR supervision
#   - hard samples: normal/original teacher prompt + 50% ESR supervision
#   - hard entropy disabled by default to isolate prompt + supervision width effects
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_DIR="${REPO_DIR:-${SCRIPT_DIR}}"
cd "${REPO_DIR}"

export TALE_ESR_BETA="${TALE_ESR_BETA:-0.20}"
export DA_HARD_PROMPT_THRESHOLD="${DA_HARD_PROMPT_THRESHOLD:-0.5}"
export DA_HARD_PROMPT_STYLE="${DA_HARD_PROMPT_STYLE:-normal}"
export DA_HARD_ESR_THRESHOLD="${DA_HARD_ESR_THRESHOLD:-0.5}"
export DA_HARD_ESR_BETA="${DA_HARD_ESR_BETA:-0.50}"
export DA_HARD_ENTROPY_COEF="${DA_HARD_ENTROPY_COEF:-0}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-opd-budget20-hardnormal-esr50-noprobe}"

bash run_train_tale_budget_rolloutlen_hardtrunc_hardnormal_opd.sh \
  algorithm.difficulty_aware_opd.hard_esr_threshold="${DA_HARD_ESR_THRESHOLD}" \
  algorithm.difficulty_aware_opd.hard_esr_beta="${DA_HARD_ESR_BETA}" \
  "$@"
