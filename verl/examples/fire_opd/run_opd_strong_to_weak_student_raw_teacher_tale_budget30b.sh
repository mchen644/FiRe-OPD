#!/bin/bash
# OPD strong-to-weak with normal student prompts and TALE-style budgeted teacher prompts.
set -x
set -euo pipefail

export PYTHONUNBUFFERED=1
REPO_DIR="${REPO_DIR:-/home/mchen/FiRe-OPD}"
export PYTHONPATH="${REPO_DIR}/verl:${REPO_DIR}:${PYTHONPATH:-}"

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

if [ -n "${SLURM_JOB_ID:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
else
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
fi

RAY_NUM_CPUS="${RAY_NUM_CPUS:-${SLURM_CPUS_PER_TASK:-16}}"
TEACHER_PROMPT_KEY="${TEACHER_PROMPT_KEY:-teacher_prompt}"
DATA_ROOT="${DATA_ROOT:-${REPO_DIR}/data/g-opd-student-raw-teacher-tale-budget}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-4}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
ROLLOUT_N="${ROLLOUT_N:-1}"

TRAIN_DATA="${TRAIN_DATA:-${DATA_ROOT}/DeepMath-103K/train_filtered_level6.parquet}"
VAL_DATA="${VAL_DATA:-['${DATA_ROOT}/AIME2024/test.parquet', '${DATA_ROOT}/AIME2025/test.parquet']}"
STUDENT_MODEL="${STUDENT_MODEL:-${REPO_DIR}/models/Qwen3-4B}"
TEACHER_MODEL="${TEACHER_MODEL:-${REPO_DIR}/models/Qwen3-30B-A3B-Instruct-2507}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-opd-strong-to-weak-studentraw-teachertale-budget-selectn${ROLLOUT_N}-${N_GPUS_PER_NODE}gpu-tp${ROLLOUT_TP_SIZE}-refmb4-rollmb4}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_DIR}/checkpoints/${EXPERIMENT_NAME}}"

if [ ! -f "${TRAIN_DATA}" ]; then
  echo "ERROR: TRAIN_DATA not found: ${TRAIN_DATA}" >&2
  echo "Run: bash ${REPO_DIR}/math_eval/prepare_g_opd_student_raw_teacher_tale_budget_data.sh" >&2
  exit 1
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=5.0 \
    algorithm.rollout_correction.rollout_rs=null \
    algorithm.rollout_correction.bypass_mode=false \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    data.train_files=${TRAIN_DATA} \
    data.val_files="${VAL_DATA}" \
    data.train_batch_size=1024 \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    data.seed=42 \
    data.return_raw_chat=True \
    +data.ref_raw_prompt_key=${TEACHER_PROMPT_KEY} \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=${STUDENT_MODEL} \
    +actor_rollout_ref.ref.model.path=${TEACHER_MODEL} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=True \
    actor_rollout_ref.actor.policy_loss.length_aware_opd=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=1024 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP_SIZE} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=naive \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.logger='["console","wandb"]' \
    trainer.log_val_generations=10 \
    trainer.project_name='fire-opd' \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.default_local_dir=${CHECKPOINT_DIR} \
    trainer.test_freq=10 \
    trainer.total_epochs=3 \
    trainer.resume_mode=auto \
    ray_kwargs.ray_init.num_cpus=${RAY_NUM_CPUS} \
    "$@"
