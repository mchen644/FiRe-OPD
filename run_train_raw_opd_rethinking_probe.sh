#!/usr/bin/env bash
# Raw OPD training with Rethinking OPD probe logging enabled.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/mchen/FiRe-OPD}"
cd "${REPO_DIR}"

# Make python3 resolve to the verified veRL environment even in a fresh shell.
export PATH="/home/mchen/miniconda3/envs/verl/bin:${PATH}"

export DATA_ROOT="${DATA_ROOT:-${REPO_DIR}/data/g-opd}"
export TRAIN_DATA="${TRAIN_DATA:-${DATA_ROOT}/DeepMath-103K/train_filtered_level6.parquet}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
export ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-4}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-opd-raw-rethinking-probe}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_DIR}/checkpoints/${EXPERIMENT_NAME}}"

bash verl/examples/fire_opd/run_opd_strong_to_weak_raw_teacher30b.sh \
  algorithm.rethinking_opd_probe.enabled=True \
  algorithm.rethinking_opd_probe.top_k=16 \
  algorithm.rethinking_opd_probe.chunk_size=1024 \
  algorithm.rethinking_opd_probe.csv_path=math_eval/opd_training_dynamics_audit/training_probe_raw_opd.csv \
  trainer.experiment_name=${EXPERIMENT_NAME} \
  trainer.val_before_train=False \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  "$@"
