#!/bin/bash
set -euo pipefail

cd /home/mchen/FiRe-OPD

# srun/sbatch should control the visible logical GPU ids. Outside Slurm,
# default to physical GPUs 1-4 to match the local interactive convention.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
else
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4}"
fi

ROLLOUT_N="${ROLLOUT_N:-2}" \
LENGTH_PENALTY_COEF="${LENGTH_PENALTY_COEF:-0.02}" \
LENGTH_PENALTY_GATE="${LENGTH_PENALTY_GATE:-incorrect_or_low_teacher}" \
LENGTH_CORRECT_REWARD_THRESHOLD="${LENGTH_CORRECT_REWARD_THRESHOLD:-0.5}" \
LENGTH_TEACHER_REJECT_PERCENTILE="${LENGTH_TEACHER_REJECT_PERCENTILE:-20.0}" \
CANDIDATE_SELECTION_ENABLED="${CANDIDATE_SELECTION_ENABLED:-True}" \
CANDIDATE_SELECTION_METHOD="${CANDIDATE_SELECTION_METHOD:-shortest_correct_else_teacher}" \
CANDIDATE_SELECTION_KEEP_PER_UID="${CANDIDATE_SELECTION_KEEP_PER_UID:-1}" \
CANDIDATE_SELECTION_CORRECT_REWARD_THRESHOLD="${CANDIDATE_SELECTION_CORRECT_REWARD_THRESHOLD:-0.5}" \
bash verl/examples/fire_opd/run_opd_strong_to_weak_lengthaware_select_teacher30b.sh
