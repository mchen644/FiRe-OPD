#!/bin/bash
# Resume rollout-length TALE hard-truncate OPD from global_step_50 and stop/save at global_step_75.
#
# This gives a clean checkpoint for eval/resume:
#   checkpoints/<experiment>/global_step_75
#
# Do NOT run this concurrently with another training job using the same CHECKPOINT_DIR.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/mchen/FiRe-OPD}"
cd "${REPO_DIR}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-opd-strong-to-weak-studentraw-teachertale-rolloutlen-a1.0-b0.2-hardtrunc-selectn1-4gpu-tp4-refmb4-rollmb4}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_DIR}/checkpoints/${EXPERIMENT_NAME}}"
RESUME_STEP="${RESUME_STEP:-50}"
TARGET_STEP="${TARGET_STEP:-75}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-${CHECKPOINT_DIR}/global_step_${RESUME_STEP}}"

if [ ! -d "${RESUME_FROM_PATH}/actor" ]; then
  echo "ERROR: missing resume actor checkpoint: ${RESUME_FROM_PATH}/actor" >&2
  exit 1
fi

cat <<EOF
Resume-to-target training settings:
  EXPERIMENT_NAME=${EXPERIMENT_NAME}
  CHECKPOINT_DIR=${CHECKPOINT_DIR}
  RESUME_FROM_PATH=${RESUME_FROM_PATH}
  TARGET_STEP=${TARGET_STEP}
  SAVE_FREQ=25
  Expected new checkpoint: ${CHECKPOINT_DIR}/global_step_${TARGET_STEP}
EOF

EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
CHECKPOINT_DIR="${CHECKPOINT_DIR}" \
bash run_train_tale_budget_rolloutlen_opd.sh \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="${RESUME_FROM_PATH}" \
  trainer.total_training_steps="${TARGET_STEP}" \
  trainer.save_freq=25 \
  "$@"
