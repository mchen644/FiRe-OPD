#!/bin/bash
# One-step online TALE-budget OPD smoke test for an interactive/srun allocation.
# Assumes SLURM/srun already assigned the intended GPUs via CUDA_VISIBLE_DEVICES.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/mchen/FiRe-OPD}"
cd "${REPO_DIR}"

export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-offline}"

# Keep the smoke small; override from the environment if needed.
SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE:-8}"
SMOKE_TOTAL_STEPS="${SMOKE_TOTAL_STEPS:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-4}"
ROLLOUT_N="${ROLLOUT_N:-1}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-smoke-online-tale-budget-opd}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/tmp/${EXPERIMENT_NAME}}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-${SLURM_CPUS_PER_TASK:-16}}"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "WARNING: CUDA_VISIBLE_DEVICES is not set. In srun this is usually set automatically." >&2
else
  echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

echo "Smoke config: batch=${SMOKE_BATCH_SIZE}, steps=${SMOKE_TOTAL_STEPS}, gpus=${N_GPUS_PER_NODE}, tp=${ROLLOUT_TP_SIZE}, rollout_n=${ROLLOUT_N}"

N_GPUS_PER_NODE="${N_GPUS_PER_NODE}" \
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE}" \
ROLLOUT_N="${ROLLOUT_N}" \
EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
CHECKPOINT_DIR="${CHECKPOINT_DIR}" \
RAY_NUM_CPUS="${RAY_NUM_CPUS}" \
bash run_train_tale_budget_opd.sh \
  trainer.total_training_steps="${SMOKE_TOTAL_STEPS}" \
  trainer.total_epochs=1 \
  trainer.val_before_train=False \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  trainer.logger='["console"]' \
  trainer.log_val_generations=0 \
  trainer.resume_mode=disable \
  data.train_batch_size="${SMOKE_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${SMOKE_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  "$@"
