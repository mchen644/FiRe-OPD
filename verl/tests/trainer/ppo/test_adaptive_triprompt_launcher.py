from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from verl.trainer.main_ppo import validate_adaptive_triprompt_runtime_config

REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_DIR = str((REPO_ROOT / "verl/verl/trainer/config").resolve())


def _compose_valid_config():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(
            config_name="ppo_trainer",
            overrides=[
                "algorithm.adaptive_triprompt_opd.enabled=True",
                "algorithm.adaptive_triprompt_opd.correct_reward_threshold=0.5",
                "algorithm.adaptive_triprompt_opd.diagnostic_max_response_length=16384",
                "algorithm.adaptive_triprompt_opd.budget_alpha=1.0",
                "algorithm.adaptive_triprompt_opd.teacher_prompt_key=teacher_prompt",
                "algorithm.adaptive_triprompt_opd.temperature=1.0",
                "algorithm.adaptive_triprompt_opd.top_p=1.0",
                "algorithm.adaptive_triprompt_opd.expected_questions_per_step=1024",
                "algorithm.adaptive_concise_opd.enabled=False",
                "algorithm.tale_budget.enabled=False",
                "algorithm.difficulty_aware_opd.enabled=False",
                "algorithm.candidate_selection.enabled=False",
                "algorithm.rethinking_opd_probe.enabled=False",
                "algorithm.opd_proxy_verify_capture.enabled=False",
                "algorithm.rollout_correction.rollout_is=token",
                "algorithm.rollout_correction.rollout_is_threshold=5.0",
                "algorithm.rollout_correction.rollout_rs=null",
                "algorithm.rollout_correction.bypass_mode=False",
                "algorithm.use_kl_in_reward=False",
                "data.train_files=/home/mchen/FiRe-OPD/data/g-opd/DeepMath-103K/train_filtered_level6.parquet",
                "data.train_batch_size=1024",
                "data.max_prompt_length=2048",
                "data.max_response_length=16384",
                "data.shuffle=True",
                "data.truncation=error",
                "data.seed=42",
                "data.return_raw_chat=True",
                "+data.ref_raw_prompt_key=teacher_prompt",
                "+data.apply_chat_template_kwargs.enable_thinking=False",
                "actor_rollout_ref.model.path=/home/mchen/FiRe-OPD/models/Qwen3-4B",
                "+actor_rollout_ref.ref.model.path=/home/mchen/FiRe-OPD/models/Qwen3-30B-A3B-Instruct-2507",
                "actor_rollout_ref.rollout.name=vllm",
                "actor_rollout_ref.rollout.mode=sync",
                "actor_rollout_ref.rollout.n=1",
                "actor_rollout_ref.rollout.calculate_log_probs=True",
                "actor_rollout_ref.rollout.tensor_model_parallel_size=4",
                "actor_rollout_ref.rollout.temperature=1.0",
                "actor_rollout_ref.rollout.top_p=1.0",
                "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4",
                "actor_rollout_ref.actor.optim.lr=1e-6",
                "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0",
                "actor_rollout_ref.actor.ppo_mini_batch_size=1024",
                "actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=True",
                "actor_rollout_ref.actor.policy_loss.length_aware_opd=False",
                "actor_rollout_ref.actor.policy_loss.entropy_aware_distill=False",
                "actor_rollout_ref.actor.loss_agg_mode=token-mean",
                "actor_rollout_ref.actor.entropy_coeff=0",
                "actor_rollout_ref.actor.entropy_from_logits_with_chunking=True",
                "actor_rollout_ref.actor.use_kl_loss=True",
                "actor_rollout_ref.actor.kl_loss_coef=0",
                "reward_model.reward_manager=naive",
                "reward_model.launch_reward_fn_async=False",
                "trainer.n_gpus_per_node=4",
                "trainer.nnodes=1",
                "trainer.total_training_steps=50",
                "trainer.resume_mode=disable",
            ],
        )


def test_triprompt_runtime_contract_accepts_only_frozen_shape() -> None:
    contract = validate_adaptive_triprompt_runtime_config(_compose_valid_config())
    assert contract == {
        "correct_reward_threshold": 0.5,
        "diagnostic_max_response_length": 16384,
        "budget_alpha": 1.0,
        "teacher_prompt_key": "teacher_prompt",
        "expected_questions_per_step": 1024,
    }


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("algorithm.adaptive_triprompt_opd.correct_reward_threshold", 0.4),
        ("algorithm.adaptive_triprompt_opd.diagnostic_max_response_length", 8192),
        ("algorithm.adaptive_triprompt_opd.budget_alpha", 0.5),
        ("algorithm.adaptive_triprompt_opd.teacher_prompt_key", "raw_prompt"),
        ("algorithm.adaptive_triprompt_opd.temperature", 0.7),
        ("algorithm.adaptive_triprompt_opd.top_p", 0.9),
        ("algorithm.adaptive_triprompt_opd.expected_questions_per_step", 256),
        ("algorithm.adaptive_concise_opd.enabled", True),
        ("algorithm.tale_budget.enabled", True),
        ("algorithm.difficulty_aware_opd.enabled", True),
        ("algorithm.candidate_selection.enabled", True),
        ("algorithm.rethinking_opd_probe.enabled", True),
        ("algorithm.opd_proxy_verify_capture.enabled", True),
        ("algorithm.rollout_correction.rollout_is", "sequence"),
        ("algorithm.rollout_correction.rollout_is_threshold", 2.0),
        ("algorithm.rollout_correction.rollout_rs", "token"),
        ("algorithm.rollout_correction.bypass_mode", True),
        ("algorithm.use_kl_in_reward", True),
        ("data.train_batch_size", 256),
        ("data.max_prompt_length", 1024),
        ("data.max_response_length", 8192),
        ("data.shuffle", False),
        ("data.seed", 43),
        ("data.ref_raw_prompt_key", "raw_prompt"),
        ("actor_rollout_ref.rollout.n", 2),
        ("actor_rollout_ref.rollout.mode", "async"),
        ("actor_rollout_ref.rollout.tensor_model_parallel_size", 2),
        ("actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu", 1),
        ("actor_rollout_ref.actor.optim.lr", 2e-6),
        ("actor_rollout_ref.actor.ppo_mini_batch_size", 256),
        ("actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages", False),
        ("actor_rollout_ref.actor.policy_loss.length_aware_opd", True),
        ("actor_rollout_ref.actor.policy_loss.entropy_aware_distill", True),
        ("actor_rollout_ref.actor.loss_agg_mode", "seq-mean-token-sum"),
        ("actor_rollout_ref.actor.entropy_coeff", 0.001),
        ("actor_rollout_ref.actor.entropy_from_logits_with_chunking", False),
        ("actor_rollout_ref.actor.kl_loss_coef", 0.1),
        ("reward_model.launch_reward_fn_async", True),
        ("trainer.n_gpus_per_node", 8),
        ("trainer.resume_mode", "auto"),
    ],
)
def test_triprompt_runtime_contract_rejects_mutation(path: str, value) -> None:
    config = _compose_valid_config()
    OmegaConf.update(config, path, value, merge=True)
    with pytest.raises(ValueError, match=path):
        validate_adaptive_triprompt_runtime_config(config)


def test_triprompt_runtime_contract_is_disabled_by_default() -> None:
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        config = compose(config_name="ppo_trainer")
    assert validate_adaptive_triprompt_runtime_config(config) is None
