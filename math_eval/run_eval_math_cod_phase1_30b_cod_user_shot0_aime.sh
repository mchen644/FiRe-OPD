#!/bin/bash
set -euo pipefail

# Run 30B CoD user-role prompt with zero few-shot examples on AIME24/AIME25:
#   - CoD instruction is placed in the user message, matching FiRe-OPD's
#     baseline prompt role placement and Chain-of-Draft's composed request.
#   - CoD's "at the end" final-answer constraint is adapted to \boxed{}.
#   - FiRe-OPD's trailing "Please reason step by step..." user instruction is
#     stripped before composing the CoD prompt.
#
# This isolates the CoD instruction from GSM8K few-shot examples.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASETS="${DATASETS:-aime24 aime25}"
N_SAMPLES="${N_SAMPLES:-32}"
SEED="${SEED:-42}"
COD_SHOT="${COD_SHOT:-0}"
RUN_NAME="${RUN_NAME:-teacher_30b-cod-user-strip-step-shot${COD_SHOT}-n${N_SAMPLES}-seed${SEED}}"

RUN_MATRIX=0 \
MODEL_KEY="teacher_30b" \
PROMPT_STYLE="cod" \
COD_SHOT="${COD_SHOT}" \
GPU_IDS="${GPU_IDS:-0,1,2,3}" \
DATASETS="${DATASETS}" \
N_SAMPLES="${N_SAMPLES}" \
SEED="${SEED}" \
RUN_NAME="${RUN_NAME}" \
bash "${SCRIPT_DIR}/run_eval_math_cod_phase1.sh"
