#!/usr/bin/env bash
set -euo pipefail

PRODUCTION_ROOT="/home/mchen/FiRe-OPD"
REPO_DIR="${REPO_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)}"
PYTHON_BIN="${PYTHON_BIN:-/home/mchen/miniconda3/envs/verl/bin/python}"
RUN_MODE="${ADAPTIVE_CONCISE_RUN_MODE:-full}"
DRY_RUN="${ADAPTIVE_CONCISE_DRY_RUN:-0}"
DATA_ROOT="${DATA_ROOT:-${PRODUCTION_ROOT}/data/g-opd}"
TRAIN_DATA="${TRAIN_DATA:-${DATA_ROOT}/DeepMath-103K/train_filtered_level6.parquet}"
VAL_DATA="${VAL_DATA:-['${DATA_ROOT}/AIME2024/test.parquet', '${DATA_ROOT}/AIME2025/test.parquet']}"
STUDENT_MODEL="${STUDENT_MODEL:-${PRODUCTION_ROOT}/models/Qwen3-4B}"
TEACHER_MODEL="${TEACHER_MODEL:-${PRODUCTION_ROOT}/models/Qwen3-30B-A3B-Instruct-2507}"
ROLLOUT_N="${ROLLOUT_N:-1}"
PROMPT_BATCH_SIZE="${PROMPT_BATCH_SIZE:-1024}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-1024}"
CONCISE_CAP_RATIO="${CONCISE_CAP_RATIO:-0.5}"
EXPECTED_QUESTIONS_PER_STEP="${EXPECTED_QUESTIONS_PER_STEP:-1024}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
DATA_SEED="${DATA_SEED:-42}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
WARMUP_RATIO="${WARMUP_RATIO:-0.0}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-4}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-${SLURM_CPUS_PER_TASK:-16}}"
TEACHER_PROMPT_KEY="teacher_prompt"


die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}


require_pinned_value() {
  local name="$1"
  local actual="$2"
  local expected="$3"
  [[ "$actual" == "$expected" ]] || \
    die "${name} must remain pinned to ${expected} (got ${actual})"
}


case "$RUN_MODE" in
  profile)
    EXPECTED_STEPS=1
    EXPECTED_SAVE_FREQ=-1
    EXPECTED_TEST_FREQ=-1
    EXPECTED_VAL_BEFORE=False
    EXPECTED_EXPERIMENT="opd-adaptive-conciseprobe-cap50-tokenneutral-rawprompt-4gpu-tp4-profile-step1"
    EXPECTED_LOG_FILE="${PRODUCTION_ROOT}/logs/adaptive_concise_opd/profile-step1.log"
    ;;
  full)
    EXPECTED_STEPS=50
    EXPECTED_SAVE_FREQ=50
    EXPECTED_TEST_FREQ=10
    EXPECTED_VAL_BEFORE=True
    EXPECTED_EXPERIMENT="opd-adaptive-conciseprobe-cap50-tokenneutral-rawprompt-4gpu-tp4-step50"
    EXPECTED_LOG_FILE="${PRODUCTION_ROOT}/logs/adaptive_concise_opd/${EXPECTED_EXPERIMENT}.log"
    ;;
  *)
    die "ADAPTIVE_CONCISE_RUN_MODE must be profile or full (got ${RUN_MODE})"
    ;;
esac

TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-${EXPECTED_STEPS}}"
SAVE_FREQ="${SAVE_FREQ:-${EXPECTED_SAVE_FREQ}}"
TEST_FREQ="${TEST_FREQ:-${EXPECTED_TEST_FREQ}}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-${EXPECTED_VAL_BEFORE}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${EXPECTED_EXPERIMENT}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PRODUCTION_ROOT}/checkpoints/${EXPERIMENT_NAME}}"
LOG_FILE="${LOG_FILE:-${EXPECTED_LOG_FILE}}"

require_pinned_value REPO_DIR "$REPO_DIR" /home/mchen/FiRe-OPD/.worktrees/vanilla-sft-gradient-51200-opd
require_pinned_value DATA_ROOT "$DATA_ROOT" /home/mchen/FiRe-OPD/data/g-opd
require_pinned_value TRAIN_DATA "$TRAIN_DATA" /home/mchen/FiRe-OPD/data/g-opd/DeepMath-103K/train_filtered_level6.parquet
require_pinned_value STUDENT_MODEL "$STUDENT_MODEL" /home/mchen/FiRe-OPD/models/Qwen3-4B
require_pinned_value TEACHER_MODEL "$TEACHER_MODEL" /home/mchen/FiRe-OPD/models/Qwen3-30B-A3B-Instruct-2507
require_pinned_value ROLLOUT_N "$ROLLOUT_N" 1
require_pinned_value PROMPT_BATCH_SIZE "$PROMPT_BATCH_SIZE" 1024
require_pinned_value PPO_MINI_BATCH_SIZE "$PPO_MINI_BATCH_SIZE" 1024
require_pinned_value CONCISE_CAP_RATIO "$CONCISE_CAP_RATIO" 0.5
require_pinned_value EXPECTED_QUESTIONS_PER_STEP "$EXPECTED_QUESTIONS_PER_STEP" 1024
require_pinned_value MAX_PROMPT_LENGTH "$MAX_PROMPT_LENGTH" 2048
require_pinned_value MAX_RESPONSE_LENGTH "$MAX_RESPONSE_LENGTH" 16384
require_pinned_value DATA_SEED "$DATA_SEED" 42
require_pinned_value LEARNING_RATE "$LEARNING_RATE" 1e-6
require_pinned_value WARMUP_RATIO "$WARMUP_RATIO" 0.0
require_pinned_value ROLLOUT_TEMPERATURE "$ROLLOUT_TEMPERATURE" 1.0
require_pinned_value ROLLOUT_TOP_P "$ROLLOUT_TOP_P" 1.0
require_pinned_value N_GPUS_PER_NODE "$N_GPUS_PER_NODE" 4
require_pinned_value ROLLOUT_TP_SIZE "$ROLLOUT_TP_SIZE" 4
require_pinned_value TOTAL_EPOCHS "$TOTAL_EPOCHS" 3
require_pinned_value TOTAL_TRAINING_STEPS "$TOTAL_TRAINING_STEPS" "$EXPECTED_STEPS"
require_pinned_value SAVE_FREQ "$SAVE_FREQ" "$EXPECTED_SAVE_FREQ"
require_pinned_value TEST_FREQ "$TEST_FREQ" "$EXPECTED_TEST_FREQ"
require_pinned_value VAL_BEFORE_TRAIN "$VAL_BEFORE_TRAIN" "$EXPECTED_VAL_BEFORE"
require_pinned_value EXPERIMENT_NAME "$EXPERIMENT_NAME" "$EXPECTED_EXPERIMENT"
require_pinned_value CHECKPOINT_DIR "$CHECKPOINT_DIR" "${PRODUCTION_ROOT}/checkpoints/${EXPECTED_EXPERIMENT}"
require_pinned_value LOG_FILE "$LOG_FILE" "$EXPECTED_LOG_FILE"
[[ "$RAY_NUM_CPUS" =~ ^[1-9][0-9]*$ ]] || die "RAY_NUM_CPUS must be positive"
(( $# == 0 )) || die "positional overrides are forbidden"
case "$DRY_RUN" in
  0|1) ;;
  *) die "ADAPTIVE_CONCISE_DRY_RUN must be 0 or 1" ;;
esac

fixed_args=(
  "algorithm.adv_estimator=grpo"
  "algorithm.rollout_correction.rollout_is=token"
  "algorithm.rollout_correction.rollout_is_threshold=5.0"
  "algorithm.rollout_correction.rollout_rs=null"
  "algorithm.rollout_correction.bypass_mode=false"
  "algorithm.adaptive_concise_opd.enabled=True"
  "algorithm.adaptive_concise_opd.correct_reward_threshold=0.5"
  "algorithm.adaptive_concise_opd.concise_cap_ratio=0.5"
  "algorithm.adaptive_concise_opd.teacher_prompt_key=${TEACHER_PROMPT_KEY}"
  "algorithm.adaptive_concise_opd.temperature=1.0"
  "algorithm.adaptive_concise_opd.top_p=1.0"
  "algorithm.adaptive_concise_opd.expected_questions_per_step=1024"
  "algorithm.tale_budget.enabled=False"
  "algorithm.difficulty_aware_opd.enabled=False"
  "algorithm.candidate_selection.enabled=False"
  "algorithm.rethinking_opd_probe.enabled=False"
  "algorithm.use_kl_in_reward=False"
  "actor_rollout_ref.rollout.calculate_log_probs=true"
  "data.train_files=${TRAIN_DATA}"
  "data.val_files=${VAL_DATA}"
  "data.train_batch_size=1024"
  "data.max_prompt_length=2048"
  "data.max_response_length=16384"
  "data.filter_overlong_prompts=True"
  "data.truncation=error"
  "data.shuffle=True"
  "data.seed=42"
  "data.return_raw_chat=True"
  "+data.ref_raw_prompt_key=${TEACHER_PROMPT_KEY}"
  "+data.apply_chat_template_kwargs.enable_thinking=False"
  "actor_rollout_ref.model.path=${STUDENT_MODEL}"
  "+actor_rollout_ref.ref.model.path=${TEACHER_MODEL}"
  "actor_rollout_ref.actor.optim.lr=1e-6"
  "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0"
  "actor_rollout_ref.model.use_remove_padding=True"
  "actor_rollout_ref.model.enable_gradient_checkpointing=True"
  "actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=True"
  "actor_rollout_ref.actor.policy_loss.length_aware_opd=False"
  "actor_rollout_ref.actor.loss_agg_mode=token-mean"
  "actor_rollout_ref.actor.ppo_mini_batch_size=1024"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.actor.use_kl_loss=True"
  "actor_rollout_ref.actor.kl_loss_coef=0"
  "actor_rollout_ref.actor.kl_loss_type=low_var_kl"
  "actor_rollout_ref.actor.entropy_coeff=0"
  "actor_rollout_ref.actor.entropy_from_logits_with_chunking=True"
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768"
  "actor_rollout_ref.actor.fsdp_config.param_offload=False"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False"
  "actor_rollout_ref.actor.use_torch_compile=False"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=4"
  "actor_rollout_ref.rollout.name=vllm"
  "actor_rollout_ref.rollout.mode=sync"
  "actor_rollout_ref.rollout.gpu_memory_utilization=0.6"
  "actor_rollout_ref.rollout.n=1"
  "actor_rollout_ref.rollout.max_num_batched_tokens=32768"
  "actor_rollout_ref.rollout.temperature=1.0"
  "actor_rollout_ref.rollout.top_p=1.0"
  "actor_rollout_ref.rollout.val_kwargs.do_sample=True"
  "actor_rollout_ref.rollout.val_kwargs.temperature=1.0"
  "actor_rollout_ref.rollout.val_kwargs.top_p=1.0"
  "actor_rollout_ref.rollout.val_kwargs.n=8"
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4"
  "actor_rollout_ref.ref.fsdp_config.param_offload=True"
  "actor_rollout_ref.ref.use_torch_compile=False"
  "reward_model.reward_manager=naive"
  "reward_model.launch_reward_fn_async=False"
  "trainer.critic_warmup=0"
  "trainer.val_before_train=${VAL_BEFORE_TRAIN}"
  "trainer.logger=[\"console\",\"wandb\"]"
  "trainer.log_val_generations=10"
  "trainer.project_name=fire-opd"
  "trainer.experiment_name=${EXPERIMENT_NAME}"
  "trainer.n_gpus_per_node=4"
  "trainer.nnodes=1"
  "trainer.save_freq=${SAVE_FREQ}"
  "trainer.default_local_dir=${CHECKPOINT_DIR}"
  "trainer.test_freq=${TEST_FREQ}"
  "trainer.total_epochs=3"
  "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}"
  "trainer.resume_mode=disable"
  "ray_kwargs.ray_init.num_cpus=${RAY_NUM_CPUS}"
)


print_contract() {
  printf 'RUN_MODE=%s\n' "$RUN_MODE"
  printf 'REPO_DIR=%s\n' "$REPO_DIR"
  printf 'PYTHON_BIN=%s\n' "$PYTHON_BIN"
  printf 'TRAIN_DATA=%s\n' "$TRAIN_DATA"
  printf 'VAL_DATA=%s\n' "$VAL_DATA"
  printf 'STUDENT_MODEL=%s\n' "$STUDENT_MODEL"
  printf 'TEACHER_MODEL=%s\n' "$TEACHER_MODEL"
  printf 'EXPERIMENT_NAME=%s\n' "$EXPERIMENT_NAME"
  printf 'CHECKPOINT_DIR=%s\n' "$CHECKPOINT_DIR"
  printf 'LOG_FILE=%s\n' "$LOG_FILE"
  printf 'ROLLOUT_N=%s\n' "$ROLLOUT_N"
  printf 'PROMPT_BATCH_SIZE=%s\n' "$PROMPT_BATCH_SIZE"
  printf 'PPO_MINI_BATCH_SIZE=%s\n' "$PPO_MINI_BATCH_SIZE"
  printf 'CONCISE_CAP_RATIO=%s\n' "$CONCISE_CAP_RATIO"
  printf 'EXPECTED_QUESTIONS_PER_STEP=%s\n' "$EXPECTED_QUESTIONS_PER_STEP"
  printf 'TOTAL_TRAINING_STEPS=%s\n' "$TOTAL_TRAINING_STEPS"
  printf 'SAVE_FREQ=%s\n' "$SAVE_FREQ"
  printf 'TEST_FREQ=%s\n' "$TEST_FREQ"
  printf 'VAL_BEFORE_TRAIN=%s\n' "$VAL_BEFORE_TRAIN"
  printf 'SOURCE_COMMIT=%s\n' "$(git -C "$REPO_DIR" rev-parse HEAD)"
  printf 'adaptive_command='
  printf '%q ' "$PYTHON_BIN" -m verl.trainer.main_ppo "${fixed_args[@]}"
  printf '\n'
}

if [[ "$DRY_RUN" == "1" ]]; then
  print_contract
  exit 0
fi

[[ -n "${SLURM_JOB_ID:-}" ]] || die "SLURM_JOB_ID is required; launch inside the existing allocation"
require_pinned_value CUDA_VISIBLE_DEVICES "${CUDA_VISIBLE_DEVICES:-}" 0,1,2,3
[[ -x "$PYTHON_BIN" ]] || die "PYTHON_BIN is not executable: $PYTHON_BIN"
[[ -f "$TRAIN_DATA" ]] || die "TRAIN_DATA is missing: $TRAIN_DATA"
[[ -f "${DATA_ROOT}/AIME2024/test.parquet" ]] || die "AIME2024 validation data is missing"
[[ -f "${DATA_ROOT}/AIME2025/test.parquet" ]] || die "AIME2025 validation data is missing"
[[ -d "$STUDENT_MODEL" ]] || die "STUDENT_MODEL is missing: $STUDENT_MODEL"
[[ -d "$TEACHER_MODEL" ]] || die "TEACHER_MODEL is missing: $TEACHER_MODEL"
[[ -d "$REPO_DIR/verl/verl" ]] || die "REPO_DIR does not contain VERL"
[[ ! -e "$CHECKPOINT_DIR" ]] || die "checkpoint destination already exists: $CHECKPOINT_DIR"
[[ ! -e "$LOG_FILE" ]] || die "log destination already exists: $LOG_FILE"
[[ -z "$(git -C "$REPO_DIR" status --porcelain=v1 --untracked-files=all)" ]] || die "source worktree must be clean"

mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$CHECKPOINT_DIR")"
if ! (set -o noclobber; : > "$LOG_FILE") 2>/dev/null; then
  die "failed to atomically claim log destination: $LOG_FILE"
fi
exec >"$LOG_FILE" 2>&1
print_contract

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_DIR}/verl:${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
cd "$REPO_DIR"
exec "$PYTHON_BIN" -m verl.trainer.main_ppo "${fixed_args[@]}"
