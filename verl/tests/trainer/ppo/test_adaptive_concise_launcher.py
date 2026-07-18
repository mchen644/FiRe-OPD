import os
import shlex
import subprocess
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from verl.trainer.main_ppo import validate_adaptive_concise_runtime_config

REPO_ROOT = Path(__file__).resolve().parents[4]
LAUNCHER = REPO_ROOT / "run_train_adaptive_concise_token_neutral_opd.sh"
CONFIG_DIR = str((REPO_ROOT / "verl/verl/trainer/config").resolve())


def _compose_valid_config():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(
            config_name="ppo_trainer",
            overrides=[
                "algorithm.adaptive_concise_opd.enabled=True",
                "algorithm.adaptive_concise_opd.correct_reward_threshold=0.5",
                "algorithm.adaptive_concise_opd.concise_cap_ratio=0.5",
                "algorithm.adaptive_concise_opd.teacher_prompt_key=teacher_prompt",
                "algorithm.adaptive_concise_opd.temperature=1.0",
                "algorithm.adaptive_concise_opd.top_p=1.0",
                "algorithm.adaptive_concise_opd.expected_questions_per_step=1024",
                "algorithm.tale_budget.enabled=False",
                "algorithm.difficulty_aware_opd.enabled=False",
                "algorithm.candidate_selection.enabled=False",
                "algorithm.rethinking_opd_probe.enabled=False",
                "algorithm.rollout_correction.rollout_is=token",
                "algorithm.rollout_correction.rollout_is_threshold=5.0",
                "algorithm.rollout_correction.rollout_rs=null",
                "algorithm.rollout_correction.bypass_mode=False",
                "algorithm.use_kl_in_reward=False",
                "data.train_batch_size=1024",
                "data.max_prompt_length=2048",
                "data.max_response_length=16384",
                "data.shuffle=True",
                "data.truncation=error",
                "data.seed=42",
                "data.return_raw_chat=True",
                "+data.ref_raw_prompt_key=teacher_prompt",
                "actor_rollout_ref.rollout.name=vllm",
                "actor_rollout_ref.rollout.mode=sync",
                "actor_rollout_ref.rollout.n=1",
                "actor_rollout_ref.rollout.calculate_log_probs=True",
                "actor_rollout_ref.rollout.tensor_model_parallel_size=4",
                "actor_rollout_ref.rollout.temperature=1.0",
                "actor_rollout_ref.rollout.top_p=1.0",
                "actor_rollout_ref.actor.optim.lr=1e-6",
                "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0",
                "actor_rollout_ref.actor.ppo_mini_batch_size=1024",
                "actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=True",
                "actor_rollout_ref.actor.policy_loss.length_aware_opd=False",
                "actor_rollout_ref.actor.loss_agg_mode=token-mean",
                "actor_rollout_ref.actor.entropy_coeff=0",
                "actor_rollout_ref.actor.use_kl_loss=True",
                "actor_rollout_ref.actor.kl_loss_coef=0",
                "reward_model.launch_reward_fn_async=False",
                "trainer.n_gpus_per_node=4",
                "trainer.nnodes=1",
                "trainer.total_training_steps=50",
                "trainer.resume_mode=disable",
            ],
        )


def test_adaptive_runtime_contract_accepts_only_the_frozen_shape() -> None:
    contract = validate_adaptive_concise_runtime_config(_compose_valid_config())
    assert contract == {
        "correct_reward_threshold": 0.5,
        "concise_cap_ratio": 0.5,
        "teacher_prompt_key": "teacher_prompt",
        "expected_questions_per_step": 1024,
    }


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("data.train_batch_size", 256),
        ("data.seed", 43),
        ("data.ref_raw_prompt_key", "raw_prompt"),
        ("actor_rollout_ref.rollout.n", 2),
        ("actor_rollout_ref.rollout.mode", "async"),
        ("actor_rollout_ref.actor.ppo_mini_batch_size", 256),
        ("actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages", False),
        ("actor_rollout_ref.actor.policy_loss.length_aware_opd", True),
        ("actor_rollout_ref.actor.entropy_coeff", 0.001),
        ("reward_model.launch_reward_fn_async", True),
        ("algorithm.adaptive_concise_opd.concise_cap_ratio", 0.4),
        ("algorithm.tale_budget.enabled", True),
        ("algorithm.difficulty_aware_opd.enabled", True),
        ("algorithm.candidate_selection.enabled", True),
        ("algorithm.rethinking_opd_probe.enabled", True),
        ("trainer.resume_mode", "auto"),
    ],
)
def test_adaptive_runtime_contract_rejects_changed_fields(path: str, value) -> None:
    config = _compose_valid_config()
    OmegaConf.update(config, path, value, merge=True)
    with pytest.raises(ValueError, match=path):
        validate_adaptive_concise_runtime_config(config)


def _run_launcher(mode: str = "full", **overrides):
    env = os.environ.copy()
    env.update(
        {
            "ADAPTIVE_CONCISE_DRY_RUN": "1",
            "ADAPTIVE_CONCISE_RUN_MODE": mode,
            **overrides,
        }
    )
    return subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("mode", "steps", "save_freq", "test_freq", "val_before"),
    [
        ("profile", "1", "-1", "-1", "False"),
        ("full", "50", "50", "10", "True"),
    ],
)
def test_launcher_pins_profile_and_full_contracts(
    mode: str, steps: str, save_freq: str, test_freq: str, val_before: str
) -> None:
    completed = _run_launcher(mode)
    assert completed.returncode == 0, completed.stderr
    required = [
        f"RUN_MODE={mode}",
        "REPO_DIR=/home/mchen/FiRe-OPD/.worktrees/vanilla-sft-gradient-51200-opd",
        "TRAIN_DATA=/home/mchen/FiRe-OPD/data/g-opd/DeepMath-103K/train_filtered_level6.parquet",
        "ROLLOUT_N=1",
        "PROMPT_BATCH_SIZE=1024",
        "PPO_MINI_BATCH_SIZE=1024",
        "CONCISE_CAP_RATIO=0.5",
        f"TOTAL_TRAINING_STEPS={steps}",
        f"trainer.total_training_steps={steps}",
        f"trainer.save_freq={save_freq}",
        f"trainer.test_freq={test_freq}",
        f"trainer.val_before_train={val_before}",
        "data.train_batch_size=1024",
        "data.seed=42",
        "data.return_raw_chat=True",
        "+data.ref_raw_prompt_key=teacher_prompt",
        "actor_rollout_ref.rollout.n=1",
        "actor_rollout_ref.actor.ppo_mini_batch_size=1024",
        "actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=True",
        "actor_rollout_ref.actor.loss_agg_mode=token-mean",
        "algorithm.rollout_correction.rollout_is=token",
        "algorithm.rollout_correction.rollout_is_threshold=5.0",
        "algorithm.adaptive_concise_opd.enabled=True",
        "algorithm.adaptive_concise_opd.concise_cap_ratio=0.5",
        "algorithm.adaptive_concise_opd.expected_questions_per_step=1024",
        "algorithm.adaptive_concise_opd.teacher_prompt_key=teacher_prompt",
        "algorithm.tale_budget.enabled=False",
        "algorithm.difficulty_aware_opd.enabled=False",
        "algorithm.candidate_selection.enabled=False",
        "algorithm.rethinking_opd_probe.enabled=False",
        "actor_rollout_ref.actor.policy_loss.length_aware_opd=False",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.entropy_from_logits_with_chunking=True",
        "trainer.resume_mode=disable",
    ]
    for value in required:
        assert value in completed.stdout


def test_launcher_command_composes_and_passes_runtime_validation() -> None:
    completed = _run_launcher("full")
    assert completed.returncode == 0, completed.stderr
    command_line = next(
        line for line in completed.stdout.splitlines() if line.startswith("adaptive_command=")
    )
    argv = shlex.split(command_line.split("=", 1)[1])
    assert argv[1:3] == ["-m", "verl.trainer.main_ppo"]
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        config = compose(config_name="ppo_trainer", overrides=argv[3:])
    contract = validate_adaptive_concise_runtime_config(config)
    assert contract["concise_cap_ratio"] == 0.5
    assert config.trainer.total_training_steps == 50


def test_launcher_chunks_actor_entropy_without_changing_old_log_prob_micro_batch() -> None:
    completed = _run_launcher("full")
    assert completed.returncode == 0, completed.stderr
    command_line = next(
        line for line in completed.stdout.splitlines() if line.startswith("adaptive_command=")
    )
    argv = shlex.split(command_line.split("=", 1)[1])
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        config = compose(config_name="ppo_trainer", overrides=argv[3:])
    assert config.actor_rollout_ref.actor.entropy_from_logits_with_chunking is True
    assert config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu == 4


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("ROLLOUT_N", "2"),
        ("PROMPT_BATCH_SIZE", "256"),
        ("PPO_MINI_BATCH_SIZE", "256"),
        ("CONCISE_CAP_RATIO", "0.4"),
        ("DATA_SEED", "43"),
        ("LEARNING_RATE", "2e-6"),
        ("MAX_RESPONSE_LENGTH", "8192"),
        ("N_GPUS_PER_NODE", "8"),
        ("ROLLOUT_TP_SIZE", "2"),
    ],
)
def test_launcher_rejects_changed_scientific_pin(name: str, value: str) -> None:
    completed = _run_launcher("full", **{name: value})
    assert completed.returncode == 2
    assert f"{name} must remain pinned" in completed.stderr


def test_launcher_rejects_invalid_mode_and_positional_arguments() -> None:
    completed = _run_launcher("invalid")
    assert completed.returncode == 2
    assert "ADAPTIVE_CONCISE_RUN_MODE" in completed.stderr

    env = os.environ.copy()
    env.update({"ADAPTIVE_CONCISE_DRY_RUN": "1", "ADAPTIVE_CONCISE_RUN_MODE": "full"})
    positional = subprocess.run(
        ["bash", str(LAUNCHER), "algorithm.gamma=0"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert positional.returncode == 2
    assert "positional overrides" in positional.stderr
