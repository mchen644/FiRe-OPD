#!/usr/bin/env bash
# Hard-normal Budget20 OPD diagnostic.
#
# Behavior:
#   - keep the original Budget20 hard truncation for every sample (20% ESR)
#   - route hard samples to the normal/original teacher prompt (no budget prompt)
#   - route all non-hard samples to the budget teacher prompt
#   - disable hard entropy by default so the run isolates the prompt-routing effect
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_DIR="${REPO_DIR:-${SCRIPT_DIR}}"
cd "${REPO_DIR}"

export TALE_ESR_BETA="${TALE_ESR_BETA:-0.20}"
export TALE_TRUNCATE_TO_ESR="${TALE_TRUNCATE_TO_ESR:-True}"
export TALE_USE_BUDGET_TEACHER_PROMPT="${TALE_USE_BUDGET_TEACHER_PROMPT:-True}"

# Disable easy-specific compression/prompting: non-hard samples remain Budget20.
export DA_EASY_PROMPT_STYLE="${DA_EASY_PROMPT_STYLE:-budget}"
export DA_DEFAULT_PROMPT_STYLE="${DA_DEFAULT_PROMPT_STYLE:-budget}"
export DA_EASY_PROMPT_THRESHOLD="${DA_EASY_PROMPT_THRESHOLD:-1.1}"
export DA_MIN_EASY_ESR_BETA="${DA_MIN_EASY_ESR_BETA:-0.20}"
export DA_EASY_ESR_DELTA="${DA_EASY_ESR_DELTA:-0.0}"

# First diagnostic isolates hard-normal prompt routing; entropy can be re-enabled later.
export DA_HARD_ENTROPY_COEF="${DA_HARD_ENTROPY_COEF:-0}"
export DA_HARD_PROMPT_THRESHOLD="${DA_HARD_PROMPT_THRESHOLD:-0.7}"
export DA_HARD_PROMPT_STYLE="${DA_HARD_PROMPT_STYLE:-normal}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-opd-budget20-hardnormal-noprobe}"

bash run_train_tale_budget_rolloutlen_hardtrunc_difficultyrouted_rethinking_probe_opd.sh \
  algorithm.difficulty_aware_opd.hard_prompt_threshold="${DA_HARD_PROMPT_THRESHOLD}" \
  algorithm.difficulty_aware_opd.hard_prompt_style="${DA_HARD_PROMPT_STYLE}" \
  algorithm.rethinking_opd_probe.enabled=False \
  "$@"
