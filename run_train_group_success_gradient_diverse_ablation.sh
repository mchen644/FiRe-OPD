#!/usr/bin/env bash
set -euo pipefail

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

require_pinned_value() {
  local name="$1"
  local actual="$2"
  local expected="$3"
  [[ "$actual" == "$expected" ]] || die "$name must remain pinned to $expected (got $actual)"
}

PRODUCTION_ROOT="/home/mchen/FiRe-OPD"
REPO_DIR="${REPO_DIR:-${PRODUCTION_ROOT}}"
GROUP_LAUNCHER="${REPO_DIR}/run_train_group_success_difficulty_opd.sh"
BASE_LAUNCHER="${REPO_DIR}/verl/examples/fire_opd/run_opd_strong_to_weak_student_raw_teacher_tale_budget30b.sh"
PYTHON_BIN="${PYTHON_BIN:-/home/mchen/miniconda3/envs/verl/bin/python}"

SOURCE_DATA="${PRODUCTION_ROOT}/data/g-opd/DeepMath-103K/train_filtered_level6.parquet"
SELECTED_DATA="${PRODUCTION_ROOT}/data/gradient_diversity/DeepMath-103K/train_gradient_diverse_12800.parquet"
SELECTION_MANIFEST="${PRODUCTION_ROOT}/data/gradient_diversity/selection/manifest.json"
SELECTED_IDS="${PRODUCTION_ROOT}/data/gradient_diversity/selection/selected_ids.jsonl"
SELECTED_SHA256="caf303c5d151fdaed2e21eebc257917f13660c589906e3da0e569ffdfd59b059"
SOURCE_SHA256="de3350fdd00bc0410550098ea65179e2be873da99e4075f80de575fc17670597"

BASELINE_EXPERIMENT="opd-n4-easy4of4-concise20-noneasynormal50-purerkl-step50"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-opd-n4-graddiv12800-easy4of4-concise20-noneasynormal50-purerkl-step50}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PRODUCTION_ROOT}/checkpoints/${EXPERIMENT_NAME}}"
LOG_FILE="${LOG_FILE:-${PRODUCTION_ROOT}/logs/difficulty_routed/${EXPERIMENT_NAME}.log}"
TRAIN_DATA="${TRAIN_DATA:-${SELECTED_DATA}}"
DATA_ROOT="${DATA_ROOT:-${PRODUCTION_ROOT}/data/g-opd}"
STUDENT_MODEL="${STUDENT_MODEL:-${PRODUCTION_ROOT}/models/Qwen3-4B}"
TEACHER_MODEL="${TEACHER_MODEL:-${PRODUCTION_ROOT}/models/Qwen3-30B-A3B-Instruct-2507}"
VAL_DATA="${VAL_DATA:-['${DATA_ROOT}/AIME2024/test.parquet', '${DATA_ROOT}/AIME2025/test.parquet']}"
TEACHER_PROMPT_KEY="${TEACHER_PROMPT_KEY:-teacher_prompt}"
ROLLOUT_N="${ROLLOUT_N:-4}"
PROMPT_BATCH_SIZE="${PROMPT_BATCH_SIZE:-256}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-256}"
TOTAL_TRAJECTORIES="${TOTAL_TRAJECTORIES:-1024}"
EXPECTED_GROUP_SIZE="${EXPECTED_GROUP_SIZE:-4}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-50}"
SAVE_FREQ="${SAVE_FREQ:-20}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-4}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"

require_pinned_value PRODUCTION_ROOT "${PRODUCTION_ROOT}" /home/mchen/FiRe-OPD
if [[ "${ABLATION_DRY_RUN:-0}" != "1" ]]; then
  require_pinned_value REPO_DIR "${REPO_DIR}" "${PRODUCTION_ROOT}"
fi
require_pinned_value TRAIN_DATA "${TRAIN_DATA}" "${SELECTED_DATA}"
require_pinned_value EXPERIMENT_NAME "${EXPERIMENT_NAME}" \
  opd-n4-graddiv12800-easy4of4-concise20-noneasynormal50-purerkl-step50
require_pinned_value CHECKPOINT_DIR "${CHECKPOINT_DIR}" \
  "${PRODUCTION_ROOT}/checkpoints/${EXPERIMENT_NAME}"
require_pinned_value LOG_FILE "${LOG_FILE}" \
  "${PRODUCTION_ROOT}/logs/difficulty_routed/${EXPERIMENT_NAME}.log"
require_pinned_value DATA_ROOT "${DATA_ROOT}" "${PRODUCTION_ROOT}/data/g-opd"
require_pinned_value STUDENT_MODEL "${STUDENT_MODEL}" "${PRODUCTION_ROOT}/models/Qwen3-4B"
require_pinned_value TEACHER_MODEL "${TEACHER_MODEL}" \
  "${PRODUCTION_ROOT}/models/Qwen3-30B-A3B-Instruct-2507"
require_pinned_value VAL_DATA "${VAL_DATA}" \
  "['${PRODUCTION_ROOT}/data/g-opd/AIME2024/test.parquet', '${PRODUCTION_ROOT}/data/g-opd/AIME2025/test.parquet']"
require_pinned_value TEACHER_PROMPT_KEY "${TEACHER_PROMPT_KEY}" teacher_prompt
require_pinned_value ROLLOUT_N "${ROLLOUT_N}" 4
require_pinned_value PROMPT_BATCH_SIZE "${PROMPT_BATCH_SIZE}" 256
require_pinned_value PPO_MINI_BATCH_SIZE "${PPO_MINI_BATCH_SIZE}" 256
require_pinned_value TOTAL_TRAJECTORIES "${TOTAL_TRAJECTORIES}" 1024
require_pinned_value EXPECTED_GROUP_SIZE "${EXPECTED_GROUP_SIZE}" 4
require_pinned_value TOTAL_TRAINING_STEPS "${TOTAL_TRAINING_STEPS}" 50
require_pinned_value SAVE_FREQ "${SAVE_FREQ}" 20
require_pinned_value N_GPUS_PER_NODE "${N_GPUS_PER_NODE}" 4
require_pinned_value ROLLOUT_TP_SIZE "${ROLLOUT_TP_SIZE}" 4
require_pinned_value MAX_PROMPT_LENGTH "${MAX_PROMPT_LENGTH}" 2048

[[ -f "${GROUP_LAUNCHER}" ]] || die "group-success launcher does not exist: ${GROUP_LAUNCHER}"
[[ -f "${BASE_LAUNCHER}" ]] || die "base launcher does not exist: ${BASE_LAUNCHER}"
export REPO_DIR DATA_ROOT TRAIN_DATA VAL_DATA STUDENT_MODEL TEACHER_MODEL
export TEACHER_PROMPT_KEY ROLLOUT_N PROMPT_BATCH_SIZE PPO_MINI_BATCH_SIZE
export TOTAL_TRAJECTORIES EXPECTED_GROUP_SIZE TOTAL_TRAINING_STEPS SAVE_FREQ
export N_GPUS_PER_NODE ROLLOUT_TP_SIZE MAX_PROMPT_LENGTH EXPERIMENT_NAME CHECKPOINT_DIR

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

BASELINE_CHECKPOINT="${PRODUCTION_ROOT}/checkpoints/${BASELINE_EXPERIMENT}"
GROUP_SUCCESS_DRY_RUN=1 \
TRAIN_DATA="${SOURCE_DATA}" \
EXPERIMENT_NAME="${BASELINE_EXPERIMENT}" \
CHECKPOINT_DIR="${BASELINE_CHECKPOINT}" \
bash "${GROUP_LAUNCHER}" >"${tmpdir}/baseline.contract"

GROUP_SUCCESS_DRY_RUN=1 \
TRAIN_DATA="${SELECTED_DATA}" \
EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
CHECKPOINT_DIR="${CHECKPOINT_DIR}" \
bash "${GROUP_LAUNCHER}" >"${tmpdir}/candidate.contract"

if ! python3 - \
  "${tmpdir}/baseline.contract" "${tmpdir}/candidate.contract" \
  "${SOURCE_DATA}" "${SELECTED_DATA}" \
  "${BASELINE_EXPERIMENT}" "${EXPERIMENT_NAME}" \
  "${BASELINE_CHECKPOINT}" "${CHECKPOINT_DIR}" <<'PY'
import difflib
from pathlib import Path
import sys

baseline_path, candidate_path = map(Path, sys.argv[1:3])
source_data, selected_data, baseline_name, candidate_name, baseline_ckpt, candidate_ckpt = sys.argv[3:]

def normalize(text, replacements):
    for actual, marker in sorted(replacements, key=lambda pair: len(pair[0]), reverse=True):
        text = text.replace(actual, marker)
    return text

baseline = normalize(
    baseline_path.read_text(encoding="utf-8"),
    [(baseline_ckpt, "<CHECKPOINT_DIR>"), (source_data, "<TRAIN_DATA>"), (baseline_name, "<EXPERIMENT_NAME>")],
)
candidate = normalize(
    candidate_path.read_text(encoding="utf-8"),
    [(candidate_ckpt, "<CHECKPOINT_DIR>"), (selected_data, "<TRAIN_DATA>"), (candidate_name, "<EXPERIMENT_NAME>")],
)
if baseline != candidate:
    sys.stderr.writelines(
        difflib.unified_diff(
            baseline.splitlines(keepends=True),
            candidate.splitlines(keepends=True),
            fromfile="normalized-baseline",
            tofile="normalized-candidate",
        )
    )
    raise SystemExit("data-only contract comparison failed")
PY
then
  die "data-only contract comparison failed"
fi
printf 'single_variable_contract=PASS\n'

if [[ "${ABLATION_DRY_RUN:-0}" == "1" ]]; then
  printf 'artifact_validation=SKIPPED_DRY_RUN\n'
  cat "${tmpdir}/candidate.contract"
  printf 'bash %q\n' "${GROUP_LAUNCHER}"
  printf 'selected_parquet_sha256=%s\n' "${SELECTED_SHA256}"
  printf 'LOG_FILE=%s\n' "${LOG_FILE}"
  exit 0
fi

[[ -x "${PYTHON_BIN}" ]] || die "PYTHON_BIN must be executable: ${PYTHON_BIN}"
artifact_report="$(
  "${PYTHON_BIN}" -m math_eval.validate_gradient_diverse_training_data \
    --selected-parquet "${SELECTED_DATA}" \
    --source-parquet "${SOURCE_DATA}" \
    --selection-manifest "${SELECTION_MANIFEST}" \
    --selected-ids "${SELECTED_IDS}" \
    --tokenizer-path "${STUDENT_MODEL}" \
    --expected-selected-sha256 "${SELECTED_SHA256}" \
    --expected-source-sha256 "${SOURCE_SHA256}" \
    --expected-rows 12800 \
    --expected-source-rows 57046 \
    --expected-eligible-rows 57045 \
    --max-prompt-tokens 2048 \
    --checkpoint-dir "${CHECKPOINT_DIR}"
)"

GIT_HEAD="$(git -C "${REPO_DIR}" rev-parse HEAD)"
GIT_DIFF_SHA256="$(git -C "${REPO_DIR}" diff --no-ext-diff --binary HEAD -- | sha256sum | awk '{print $1}')"
GROUP_LAUNCHER_SHA256="$(sha256sum "${GROUP_LAUNCHER}" | awk '{print $1}')"
BASE_LAUNCHER_SHA256="$(sha256sum "${BASE_LAUNCHER}" | awk '{print $1}')"
ABLATION_LAUNCHER_SHA256="$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')"

print_provenance() {
  printf 'git_head=%s\n' "${GIT_HEAD}"
  printf 'git_status_begin\n'
  git -C "${REPO_DIR}" status --short
  printf 'git_status_end\n'
  printf 'git_diff_sha256=%s\n' "${GIT_DIFF_SHA256}"
  printf 'group_launcher_sha256=%s\n' "${GROUP_LAUNCHER_SHA256}"
  printf 'base_launcher_sha256=%s\n' "${BASE_LAUNCHER_SHA256}"
  printf 'ablation_launcher_sha256=%s\n' "${ABLATION_LAUNCHER_SHA256}"
  printf 'selected_parquet_sha256=%s\n' "${SELECTED_SHA256}"
  printf 'artifact_report=%s\n' "${artifact_report}"
  printf 'resolved_contract_begin\n'
  cat "${tmpdir}/candidate.contract"
  printf 'resolved_contract_end\n'
}

if [[ "${ABLATION_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  print_provenance
  exit 0
fi

mkdir -p "$(dirname -- "${LOG_FILE}")"
exec > >(tee -a "${LOG_FILE}") 2>&1
print_provenance
rm -rf "${tmpdir}"
trap - EXIT
cd "${REPO_DIR}"
exec bash "${GROUP_LAUNCHER}"
