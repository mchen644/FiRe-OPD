#!/bin/bash
set -euo pipefail

# Run the Phase 1 CoD direct-eval 4-setting matrix sequentially:
#   1. student_4b baseline
#   2. student_4b cod, 5-shot CoD examples
#   3. teacher_30b baseline
#   4. teacher_30b cod, 5-shot CoD examples
#
# Intended for an interactive srun allocation with 4 visible GPUs.
# Under Slurm, these should be logical ids 0,1,2,3.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_MATRIX=1 \
MODEL_KEYS="${MODEL_KEYS:-student_4b teacher_30b}" \
PROMPT_STYLES="${PROMPT_STYLES:-baseline cod}" \
COD_SHOT="${COD_SHOT:-5}" \
GPU_IDS="${GPU_IDS:-0,1,2,3}" \
bash "${SCRIPT_DIR}/run_eval_math_cod_phase1.sh"
