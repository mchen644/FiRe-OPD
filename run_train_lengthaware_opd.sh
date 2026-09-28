#!/bin/bash
set -euo pipefail

cd /home/mchen/FiRe-OPD

if [ -n "${SLURM_JOB_ID:-}" ]; then
  # Inside srun/sbatch, use Slurm's assigned visible GPUs.
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
else
  # Outside Slurm, default to physical GPUs 1-4.
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4}"
fi

LENGTH_PENALTY_COEF=0.05 \
LENGTH_PENALTY_GATE=incorrect_or_low_teacher \
LENGTH_CORRECT_REWARD_THRESHOLD=0.5 \
LENGTH_TEACHER_REJECT_PERCENTILE=20.0 \
bash verl/examples/fire_opd/run_opd_strong_to_weak_lengthaware_teacher30b.sh
