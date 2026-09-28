#!/bin/bash
# Resume rollout-length TALE hard-truncate OPD from the saved global_step_50 checkpoint.
#
# Do NOT run this concurrently with another training job using the same CHECKPOINT_DIR.
# It is meant for restarting after stopping the current run.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/mchen/FiRe-OPD}"
cd "${REPO_DIR}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-opd-strong-to-weak-studentraw-teachertale-rolloutlen-a1.0-b0.2-hardtrunc-selectn1-4gpu-tp4-refmb4-rollmb4}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_DIR}/checkpoints/${EXPERIMENT_NAME}}"
RESUME_STEP="${RESUME_STEP:-50}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-${CHECKPOINT_DIR}/global_step_${RESUME_STEP}}"

if [ ! -d "${RESUME_FROM_PATH}/actor" ]; then
  echo "ERROR: missing resume actor checkpoint: ${RESUME_FROM_PATH}/actor" >&2
  exit 1
fi

cat <<EOF
Resume training settings:
  EXPERIMENT_NAME=${EXPERIMENT_NAME}
  CHECKPOINT_DIR=${CHECKPOINT_DIR}
  RESUME_FROM_PATH=${RESUME_FROM_PATH}
  TALE settings inherited from run_train_tale_budget_rolloutlen_opd.sh:
    budget = rollout length
    supervised tokens = 0.2 * budget
    hard truncate = True
EOF

EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
CHECKPOINT_DIR="${CHECKPOINT_DIR}" \
bash run_train_tale_budget_rolloutlen_opd.sh \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="${RESUME_FROM_PATH}" \
  "$@"
