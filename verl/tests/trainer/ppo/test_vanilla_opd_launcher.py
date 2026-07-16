import os
import shlex
import subprocess
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

REPO_ROOT = Path(__file__).resolve().parents[4]
LAUNCHER = REPO_ROOT / "run_train_vanilla_opd.sh"


def _run(**overrides):
    env = os.environ.copy()
    env.update({"VANILLA_OPD_DRY_RUN": "1", **overrides})
    return subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_vanilla_launcher_pins_canonical_step50_contract() -> None:
    completed = _run()
    assert completed.returncode == 0, completed.stderr
    required = [
        "ROLLOUT_N=1",
        "PROMPT_BATCH_SIZE=1024",
        "PPO_MINI_BATCH_SIZE=1024",
        "TOTAL_TRAJECTORIES=1024",
        "TOTAL_TRAINING_STEPS=50",
        "data.train_batch_size=1024",
        "data.seed=42",
        "data.max_response_length=16384",
        "data.return_raw_chat=True",
        "actor_rollout_ref.rollout.n=1",
        "actor_rollout_ref.actor.ppo_mini_batch_size=1024",
        "actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=True",
        "actor_rollout_ref.actor.loss_agg_mode=token-mean",
        "algorithm.rollout_correction.rollout_is=token",
        "algorithm.rollout_correction.rollout_is_threshold=5.0",
        "algorithm.tale_budget.enabled=False",
        "algorithm.difficulty_aware_opd.enabled=False",
        "algorithm.candidate_selection.enabled=False",
        "algorithm.rethinking_opd_probe.enabled=False",
        "actor_rollout_ref.actor.policy_loss.length_aware_opd=False",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.kl_loss_coef=0",
        "algorithm.use_kl_in_reward=False",
        "trainer.val_before_train=True",
        "trainer.test_freq=10",
        "trainer.save_freq=50",
        "trainer.total_training_steps=50",
        "trainer.total_epochs=3",
        "trainer.resume_mode=disable",
    ]
    for value in required:
        assert value in completed.stdout
    assert "ref_raw_prompt_key" not in completed.stdout


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("ROLLOUT_N", "4"),
        ("PROMPT_BATCH_SIZE", "256"),
        ("PPO_MINI_BATCH_SIZE", "256"),
        ("TOTAL_TRAJECTORIES", "2048"),
        ("TOTAL_TRAINING_STEPS", "51"),
        ("SAVE_FREQ", "20"),
        ("TEST_FREQ", "-1"),
        ("N_GPUS_PER_NODE", "8"),
        ("ROLLOUT_TP_SIZE", "2"),
        ("MAX_PROMPT_LENGTH", "1024"),
        ("MAX_RESPONSE_LENGTH", "8192"),
        ("DATA_SEED", "43"),
        ("LEARNING_RATE", "2e-6"),
        ("WARMUP_RATIO", "0.1"),
        ("ROLLOUT_TEMPERATURE", "0.8"),
        ("ROLLOUT_TOP_P", "0.95"),
        ("VALIDATION_N", "4"),
        ("VALIDATION_TEMPERATURE", "0.8"),
        ("VALIDATION_TOP_P", "0.95"),
        ("TOTAL_EPOCHS", "4"),
    ],
)
def test_vanilla_launcher_rejects_changed_scientific_pin(
    name: str, value: str
) -> None:
    completed = _run(**{name: value})
    assert completed.returncode == 2
    assert f"{name} must remain pinned" in completed.stderr


def test_vanilla_command_composes_into_resolved_ppo_config() -> None:
    completed = _run()
    command_line = next(
        line
        for line in completed.stdout.splitlines()
        if line.startswith("vanilla_command=")
    )
    argv = shlex.split(command_line.split("=", 1)[1])
    assert argv[1:3] == ["-m", "verl.trainer.main_ppo"]

    config_dir = str((REPO_ROOT / "verl/verl/trainer/config").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name="ppo_trainer", overrides=argv[3:])

    assert config.data.train_batch_size == 1_024
    assert config.actor_rollout_ref.rollout.n == 1
    assert config.actor_rollout_ref.actor.ppo_mini_batch_size == 1_024
    assert config.actor_rollout_ref.rollout.val_kwargs.n == 8
    assert config.algorithm.tale_budget.enabled is False
    assert config.algorithm.difficulty_aware_opd.enabled is False
    assert config.algorithm.candidate_selection.enabled is False
    assert config.algorithm.rethinking_opd_probe.enabled is False
    assert config.trainer.total_training_steps == 50
    assert config.trainer.resume_mode == "disable"
