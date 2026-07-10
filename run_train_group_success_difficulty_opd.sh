#!/usr/bin/env bash
# Clean Stage-1 OPD: online n=4 group success routes teacher prompt and ESR only.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/mchen/FiRe-OPD}"
BASE_LAUNCHER="${REPO_DIR}/verl/examples/fire_opd/run_opd_strong_to_weak_student_raw_teacher_tale_budget30b.sh"

ROLLOUT_N="${ROLLOUT_N:-4}"
PROMPT_BATCH_SIZE="${PROMPT_BATCH_SIZE:-256}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-256}"
TOTAL_TRAJECTORIES="${TOTAL_TRAJECTORIES:-1024}"
EXPECTED_GROUP_SIZE="${EXPECTED_GROUP_SIZE:-4}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-50}"
SAVE_FREQ="${SAVE_FREQ:-20}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-opd-n4-easy4of4-concise20-noneasynormal50-purerkl-step50}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_DIR}/checkpoints/${EXPERIMENT_NAME}}"

for integer_name in ROLLOUT_N PROMPT_BATCH_SIZE PPO_MINI_BATCH_SIZE TOTAL_TRAJECTORIES EXPECTED_GROUP_SIZE TOTAL_TRAINING_STEPS SAVE_FREQ; do
  integer_value="${!integer_name}"
  if [[ ! "${integer_value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: ${integer_name} must be a positive integer, got ${integer_value}" >&2
    exit 2
  fi
done

if (( ROLLOUT_N != EXPECTED_GROUP_SIZE )); then
  echo "ERROR: ROLLOUT_N must equal EXPECTED_GROUP_SIZE" >&2
  exit 2
fi
if (( PROMPT_BATCH_SIZE * ROLLOUT_N != TOTAL_TRAJECTORIES )); then
  echo "ERROR: prompt batch times rollout count must equal TOTAL_TRAJECTORIES" >&2
  exit 2
fi
if (( PPO_MINI_BATCH_SIZE != PROMPT_BATCH_SIZE )); then
  echo "ERROR: PPO_MINI_BATCH_SIZE must equal PROMPT_BATCH_SIZE before worker rollout expansion" >&2
  exit 2
fi

export PATH="/home/mchen/miniconda3/envs/verl/bin:${PATH}"
export REPO_DIR
export ROLLOUT_N
export TALE_BUDGET_SOURCE=rollout_length
export TALE_ROLLOUT_ALPHA=1.0
export TALE_ESR_BETA=0.5
export TALE_ROLLOUT_MAX_BUDGET=null
export TALE_TEACHER_PROMPT_STYLE=normal
export DATA_ROOT="${DATA_ROOT:-${REPO_DIR}/data/g-opd}"
export TRAIN_DATA="${TRAIN_DATA:-${DATA_ROOT}/DeepMath-103K/train_filtered_level6.parquet}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
export ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-4}"
export EXPERIMENT_NAME
export CHECKPOINT_DIR

fixed_args=(
  "data.train_batch_size=${PROMPT_BATCH_SIZE}"
  "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
  "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
  "algorithm.tale_budget.enabled=True"
  "algorithm.tale_budget.source=rollout_length"
  "algorithm.tale_budget.rollout_length_alpha=1.0"
  "algorithm.tale_budget.esr_beta=0.5"
  "algorithm.tale_budget.rollout_length_max_budget=null"
  "algorithm.tale_budget.truncate_to_esr=True"
  "algorithm.tale_budget.use_budget_teacher_prompt=False"
  "algorithm.tale_budget.teacher_prompt_style=normal"
  "algorithm.tale_budget.min_budget=1"
  "algorithm.tale_budget.round_to=1"
  "algorithm.difficulty_aware_opd.enabled=True"
  "algorithm.difficulty_aware_opd.method=group_success_prompt_esr"
  "algorithm.difficulty_aware_opd.correct_reward_threshold=0.5"
  "algorithm.difficulty_aware_opd.expected_group_size=${EXPECTED_GROUP_SIZE}"
  "algorithm.difficulty_aware_opd.easy_group_correct_count=4"
  "algorithm.difficulty_aware_opd.easy_prompt_style=concise"
  "algorithm.difficulty_aware_opd.default_prompt_style=normal"
  "algorithm.difficulty_aware_opd.easy_esr_beta=0.20"
  "algorithm.difficulty_aware_opd.non_easy_esr_beta=0.50"
  "algorithm.difficulty_aware_opd.hard_entropy_coef=0.0"
  "algorithm.candidate_selection.enabled=False"
  "algorithm.rethinking_opd_probe.enabled=False"
  "actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=True"
  "actor_rollout_ref.actor.policy_loss.length_aware_opd=False"
  "actor_rollout_ref.actor.entropy_coeff=0"
  "actor_rollout_ref.actor.kl_loss_coef=0"
  "algorithm.use_kl_in_reward=False"
  "actor_rollout_ref.actor.use_torch_compile=False"
  "actor_rollout_ref.ref.use_torch_compile=False"
  "trainer.experiment_name=${EXPERIMENT_NAME}"
  "trainer.default_local_dir=${CHECKPOINT_DIR}"
  "trainer.val_before_train=False"
  "trainer.test_freq=-1"
  "trainer.save_freq=${SAVE_FREQ}"
  "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}"
  "trainer.resume_mode=disable"
)

if [[ "${GROUP_SUCCESS_DRY_RUN:-0}" == "1" ]]; then
  printf 'REPO_DIR=%s\n' "${REPO_DIR}"
  printf 'ROLLOUT_N=%s\n' "${ROLLOUT_N}"
  printf 'PROMPT_BATCH_SIZE=%s\n' "${PROMPT_BATCH_SIZE}"
  printf 'PPO_MINI_BATCH_SIZE=%s\n' "${PPO_MINI_BATCH_SIZE}"
  printf 'TOTAL_TRAJECTORIES=%s\n' "${TOTAL_TRAJECTORIES}"
  printf 'EXPERIMENT_NAME=%s\n' "${EXPERIMENT_NAME}"
  printf 'bash %q' "${BASE_LAUNCHER}"
  printf ' %q' "$@" "${fixed_args[@]}"
  printf '\n'
  exit 0
fi

cd "${REPO_DIR}"
exec bash "${BASE_LAUNCHER}" "$@" "${fixed_args[@]}"
