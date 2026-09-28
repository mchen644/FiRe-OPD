#!/bin/bash
# BAGEN-style OPD budget-awareness probe on AIME24 base-student rollouts.
#
# Protocol:
#   - student source: un-OPD-trained Qwen3-4B AIME24 normal-prompt eval outputs
#   - prefixes: 10%, 20%, 40%
#   - teacher: current Qwen3-30B-A3B-Instruct-2507
#   - task: teacher predicts possible/impossible for student finishing correctly
#
# Dry-run prompt construction only:
#   DRY_RUN=1 bash math_eval/run_bagen_opd_probe_aime24_base_student_teacher30b.sh
#
# Full teacher probe, recommended inside a 4-GPU allocation:
#   srun --gres=gpu:4 --cpus-per-task=16 bash math_eval/run_bagen_opd_probe_aime24_base_student_teacher30b.sh
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/mchen/FiRe-OPD}"
cd "${REPO_DIR}"
export PATH="/home/mchen/miniconda3/envs/verl/bin:${PATH}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

INPUT_FILE="${INPUT_FILE:-${REPO_DIR}/math_eval/cod_phase1_outputs/aime24/student_4b-baseline-shot5-n32-seed42.jsonl}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${REPO_DIR}/models/Qwen3-4B}"
TEACHER_MODEL="${TEACHER_MODEL:-${REPO_DIR}/models/Qwen3-30B-A3B-Instruct-2507}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/math_eval/bagen_opd_probe_outputs/aime24_base4b_teacher30b_prefix10_20_40}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${OUTPUT_ROOT}/probe_records.jsonl}"
SUMMARY_JSON="${SUMMARY_JSON:-${OUTPUT_ROOT}/summary.json}"
PREFIX_FRACTIONS="${PREFIX_FRACTIONS:-0.10 0.20 0.40}"
MAX_ROLLOUTS_PER_PROBLEM="${MAX_ROLLOUTS_PER_PROBLEM:-32}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
TP_SIZE="${TP_SIZE:-${TENSOR_PARALLEL_SIZE:-4}}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_TOKENS="${MAX_TOKENS:-64}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
SEED="${SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"

if [ ! -f "${INPUT_FILE}" ]; then
  echo "ERROR: missing input eval output: ${INPUT_FILE}" >&2
  exit 1
fi
if [ ! -d "${TOKENIZER_PATH}" ]; then
  echo "ERROR: missing tokenizer path: ${TOKENIZER_PATH}" >&2
  exit 1
fi
if [ "${DRY_RUN}" != "1" ] && [ ! -d "${TEACHER_MODEL}" ]; then
  echo "ERROR: missing teacher model: ${TEACHER_MODEL}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

CMD=(
  python math_eval/bagen_opd_probe.py
  --input-file "${INPUT_FILE}"
  --output-jsonl "${OUTPUT_JSONL}"
  --summary-json "${SUMMARY_JSON}"
  --dataset aime24
  --tokenizer-path "${TOKENIZER_PATH}"
  --teacher-model "${TEACHER_MODEL}"
  --prefix-fractions ${PREFIX_FRACTIONS}
  --max-rollouts-per-problem "${MAX_ROLLOUTS_PER_PROBLEM}"
  --tensor-parallel-size "${TP_SIZE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --max-tokens "${MAX_TOKENS}"
  --seed "${SEED}"
)

if [ -n "${MAX_SAMPLES}" ]; then
  CMD+=(--max-samples "${MAX_SAMPLES}")
fi
if [ "${DRY_RUN}" = "1" ]; then
  CMD+=(--dry-run)
fi

cat <<EOF
BAGEN-style OPD probe settings:
  INPUT_FILE=${INPUT_FILE}
  TOKENIZER_PATH=${TOKENIZER_PATH}
  TEACHER_MODEL=${TEACHER_MODEL}
  PREFIX_FRACTIONS=${PREFIX_FRACTIONS}
  MAX_ROLLOUTS_PER_PROBLEM=${MAX_ROLLOUTS_PER_PROBLEM}
  MAX_SAMPLES=${MAX_SAMPLES:-<none>}
  DRY_RUN=${DRY_RUN}
  OUTPUT_JSONL=${OUTPUT_JSONL}
  SUMMARY_JSON=${SUMMARY_JSON}
EOF

"${CMD[@]}"
