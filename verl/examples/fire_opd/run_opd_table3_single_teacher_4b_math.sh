#!/bin/bash
# Pure OPD Table 3: Single-Teacher Reverse-KL Distillation
# Student: Qwen3-4B
# Teacher: Qwen3-4B-Non-Thinking-RL-Math-Step500
#
# This is the OPD baseline counterpart of:
#   run_fire_opd_table3_single_teacher_4b_math.sh
#
# It keeps only_reverse_kl_advantages=True and does not enable FiRe-OPD
# filtering or token reweighting.
set -x

export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/mchen/FiRe-OPD/verl:${PYTHONPATH:-}

# This is a CUDA/H200 run. Some Slurm environments export ROCm/HIP
# visibility variables as well; verl rejects having them alongside CUDA.
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

# Use the four GPUs visible inside the Slurm allocation.
# Override CUDA_VISIBLE_DEVICES manually if running outside Slurm.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

# Ray must be constrained under Slurm; otherwise it may see all node CPUs
# and prestart hundreds of workers, causing apparent startup hangs.
RAY_NUM_CPUS="${RAY_NUM_CPUS:-${SLURM_CPUS_PER_TASK:-16}}"

# ============ Data paths ============
TRAIN_DATA=/home/mchen/FiRe-OPD/data/g-opd/DeepMath-103K/train_filtered_level6.parquet
VAL_DATA="['/home/mchen/FiRe-OPD/data/g-opd/AIME2024/test.parquet', '/home/mchen/FiRe-OPD/data/g-opd/AIME2025/test.parquet']"

# ============ Model paths ============
STUDENT_MODEL=/home/mchen/FiRe-OPD/models/Qwen3-4B
TEACHER_MODEL=/home/mchen/FiRe-OPD/models/Qwen3-4B-Non-Thinking-RL-Math-Step500

# ============ Output ============
EXPERIMENT_NAME=opd-table3-single-teacher-4b-math
CHECKPOINT_DIR=/home/mchen/FiRe-OPD/checkpoints/${EXPERIMENT_NAME}

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
    data.max_prompt_length=2048 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    data.seed=42 \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=${STUDENT_MODEL} \
    +actor_rollout_ref.ref.model.path=${TEACHER_MODEL} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=True \
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
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=1 \
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
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.default_local_dir=${CHECKPOINT_DIR} \
    trainer.test_freq=10 \
    trainer.total_epochs=3 \
    trainer.resume_mode=auto \
    ray_kwargs.ray_init.num_cpus=${RAY_NUM_CPUS}
