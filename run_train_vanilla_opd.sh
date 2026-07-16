#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/mchen/FiRe-OPD}"
PYTHON_BIN="${PYTHON_BIN:-/home/mchen/miniconda3/envs/verl/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/mchen/FiRe-OPD/data/g-opd}"
TRAIN_DATA="${TRAIN_DATA:-${DATA_ROOT}/DeepMath-103K/train_filtered_level6.parquet}"
VAL_DATA="${VAL_DATA:-['${DATA_ROOT}/AIME2024/test.parquet', '${DATA_ROOT}/AIME2025/test.parquet']}"
STUDENT_MODEL="${STUDENT_MODEL:-/home/mchen/FiRe-OPD/models/Qwen3-4B}"
TEACHER_MODEL="${TEACHER_MODEL:-/home/mchen/FiRe-OPD/models/Qwen3-30B-A3B-Instruct-2507}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-opd-strong-to-weak-rawprompt-4gpu-tp4-refmb4-rollmb4}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/home/mchen/FiRe-OPD/checkpoints/${EXPERIMENT_NAME}}"
ROLLOUT_N="${ROLLOUT_N:-1}"
PROMPT_BATCH_SIZE="${PROMPT_BATCH_SIZE:-1024}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-1024}"
TOTAL_TRAJECTORIES="${TOTAL_TRAJECTORIES:-1024}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-50}"
SAVE_FREQ="${SAVE_FREQ:-50}"
TEST_FREQ="${TEST_FREQ:-10}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-4}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
DATA_SEED="${DATA_SEED:-42}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
WARMUP_RATIO="${WARMUP_RATIO:-0.0}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
VALIDATION_N="${VALIDATION_N:-8}"
VALIDATION_TEMPERATURE="${VALIDATION_TEMPERATURE:-1.0}"
VALIDATION_TOP_P="${VALIDATION_TOP_P:-1.0}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-${SLURM_CPUS_PER_TASK:-16}}"


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

require_pinned_value ROLLOUT_N "$ROLLOUT_N" 1
require_pinned_value PROMPT_BATCH_SIZE "$PROMPT_BATCH_SIZE" 1024
require_pinned_value PPO_MINI_BATCH_SIZE "$PPO_MINI_BATCH_SIZE" 1024
require_pinned_value TOTAL_TRAJECTORIES "$TOTAL_TRAJECTORIES" 1024
require_pinned_value TOTAL_TRAINING_STEPS "$TOTAL_TRAINING_STEPS" 50
require_pinned_value SAVE_FREQ "$SAVE_FREQ" 50
require_pinned_value TEST_FREQ "$TEST_FREQ" 10
require_pinned_value N_GPUS_PER_NODE "$N_GPUS_PER_NODE" 4
require_pinned_value ROLLOUT_TP_SIZE "$ROLLOUT_TP_SIZE" 4
require_pinned_value MAX_PROMPT_LENGTH "$MAX_PROMPT_LENGTH" 2048
require_pinned_value MAX_RESPONSE_LENGTH "$MAX_RESPONSE_LENGTH" 16384
require_pinned_value DATA_SEED "$DATA_SEED" 42
require_pinned_value LEARNING_RATE "$LEARNING_RATE" 1e-6
require_pinned_value WARMUP_RATIO "$WARMUP_RATIO" 0.0
require_pinned_value ROLLOUT_TEMPERATURE "$ROLLOUT_TEMPERATURE" 1.0
require_pinned_value ROLLOUT_TOP_P "$ROLLOUT_TOP_P" 1.0
require_pinned_value VALIDATION_N "$VALIDATION_N" 8
require_pinned_value VALIDATION_TEMPERATURE "$VALIDATION_TEMPERATURE" 1.0
require_pinned_value VALIDATION_TOP_P "$VALIDATION_TOP_P" 1.0
require_pinned_value TOTAL_EPOCHS "$TOTAL_EPOCHS" 3
[[ "$RAY_NUM_CPUS" =~ ^[1-9][0-9]*$ ]] || die "RAY_NUM_CPUS must be positive"
(( PROMPT_BATCH_SIZE * ROLLOUT_N == TOTAL_TRAJECTORIES )) || \
  die "PROMPT_BATCH_SIZE * ROLLOUT_N must equal TOTAL_TRAJECTORIES"
(( $# == 0 )) || die "unreviewed command-line overrides are forbidden"

fixed_args=(
  "algorithm.adv_estimator=grpo"
  "algorithm.rollout_correction.rollout_is=token"
  "algorithm.rollout_correction.rollout_is_threshold=5.0"
  "algorithm.rollout_correction.rollout_rs=null"
  "algorithm.rollout_correction.bypass_mode=false"
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
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768"
  "actor_rollout_ref.actor.fsdp_config.param_offload=False"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=4"
  "actor_rollout_ref.rollout.name=vllm"
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
  "reward_model.reward_manager=naive"
  "trainer.critic_warmup=0"
  "trainer.val_before_train=True"
  "trainer.logger=[\"console\",\"wandb\"]"
  "trainer.log_val_generations=10"
  "trainer.project_name=fire-opd"
  "trainer.experiment_name=${EXPERIMENT_NAME}"
  "trainer.n_gpus_per_node=4"
  "trainer.nnodes=1"
  "trainer.save_freq=50"
  "trainer.default_local_dir=${CHECKPOINT_DIR}"
  "trainer.test_freq=10"
  "trainer.total_epochs=3"
  "trainer.total_training_steps=50"
  "trainer.resume_mode=disable"
  "ray_kwargs.ray_init.num_cpus=${RAY_NUM_CPUS}"
)

print_contract() {
  printf 'REPO_DIR=%s\n' "$REPO_DIR"
  printf 'PYTHON_BIN=%s\n' "$PYTHON_BIN"
  printf 'TRAIN_DATA=%s\n' "$TRAIN_DATA"
  printf 'VAL_DATA=%s\n' "$VAL_DATA"
  printf 'STUDENT_MODEL=%s\n' "$STUDENT_MODEL"
  printf 'TEACHER_MODEL=%s\n' "$TEACHER_MODEL"
  printf 'EXPERIMENT_NAME=%s\n' "$EXPERIMENT_NAME"
  printf 'CHECKPOINT_DIR=%s\n' "$CHECKPOINT_DIR"
  printf 'ROLLOUT_N=%s\n' "$ROLLOUT_N"
  printf 'PROMPT_BATCH_SIZE=%s\n' "$PROMPT_BATCH_SIZE"
  printf 'PPO_MINI_BATCH_SIZE=%s\n' "$PPO_MINI_BATCH_SIZE"
  printf 'TOTAL_TRAJECTORIES=%s\n' "$TOTAL_TRAJECTORIES"
  printf 'TOTAL_TRAINING_STEPS=%s\n' "$TOTAL_TRAINING_STEPS"
  printf 'MAX_PROMPT_LENGTH=%s\n' "$MAX_PROMPT_LENGTH"
  printf 'MAX_RESPONSE_LENGTH=%s\n' "$MAX_RESPONSE_LENGTH"
  printf 'DATA_SEED=%s\n' "$DATA_SEED"
  printf 'LEARNING_RATE=%s\n' "$LEARNING_RATE"
  printf 'WARMUP_RATIO=%s\n' "$WARMUP_RATIO"
  printf 'SAVE_FREQ=%s\n' "$SAVE_FREQ"
  printf 'TEST_FREQ=%s\n' "$TEST_FREQ"
  printf 'VALIDATION_N=%s\n' "$VALIDATION_N"
  printf 'N_GPUS_PER_NODE=%s\n' "$N_GPUS_PER_NODE"
  printf 'ROLLOUT_TP_SIZE=%s\n' "$ROLLOUT_TP_SIZE"
  printf 'RAY_NUM_CPUS=%s\n' "$RAY_NUM_CPUS"
  printf 'vanilla_command='
  printf '%q ' "$PYTHON_BIN" -m verl.trainer.main_ppo "${fixed_args[@]}"
  printf '\n'
}

case "${VANILLA_OPD_DRY_RUN:-0}" in
  0|1) ;;
  *) die "VANILLA_OPD_DRY_RUN must be 0 or 1" ;;
esac
if [[ "${VANILLA_OPD_DRY_RUN:-0}" == "1" ]]; then
  print_contract
  exit 0
fi

[[ -x "$PYTHON_BIN" ]] || die "PYTHON_BIN is not executable: $PYTHON_BIN"
[[ -d "$REPO_DIR/verl/verl" ]] || die "REPO_DIR does not contain VERL: $REPO_DIR"
[[ -f "$TRAIN_DATA" ]] || die "TRAIN_DATA is not a regular file: $TRAIN_DATA"
[[ -f "$DATA_ROOT/AIME2024/test.parquet" ]] || die "AIME2024 validation data is missing"
[[ -f "$DATA_ROOT/AIME2025/test.parquet" ]] || die "AIME2025 validation data is missing"
[[ -d "$STUDENT_MODEL" ]] || die "STUDENT_MODEL is not a directory: $STUDENT_MODEL"
[[ -d "$TEACHER_MODEL" ]] || die "TEACHER_MODEL is not a directory: $TEACHER_MODEL"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_DIR}/verl:${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
cd "$REPO_DIR"
exec "$PYTHON_BIN" -m verl.trainer.main_ppo "${fixed_args[@]}"
