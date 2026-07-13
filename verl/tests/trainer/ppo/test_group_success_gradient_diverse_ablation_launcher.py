import os
from pathlib import Path
import subprocess

import pytest


REPO_DIR = Path(__file__).resolve().parents[4]
LAUNCHER = REPO_DIR / "run_train_group_success_gradient_diverse_ablation.sh"
RUN_NAME = "opd-n4-graddiv12800-easy4of4-concise20-noneasynormal50-purerkl-step50"


def _run(**overrides):
    env = os.environ.copy()
    env.update(
        {
            "ABLATION_DRY_RUN": "1",
            "PYTHON_BIN": "/path/not/used/in/dry-run",
            "REPO_DIR": str(REPO_DIR),
            **overrides,
        }
    )
    return subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_DIR,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_ablation_dry_run_is_artifact_independent_and_data_only():
    completed = _run()

    assert completed.returncode == 0, completed.stderr
    assert "artifact_validation=SKIPPED_DRY_RUN\n" in completed.stdout
    assert "single_variable_contract=PASS\n" in completed.stdout
    assert f"EXPERIMENT_NAME={RUN_NAME}\n" in completed.stdout
    assert "TRAIN_DATA=/home/mchen/FiRe-OPD/data/gradient_diversity/DeepMath-103K/train_gradient_diverse_12800.parquet\n" in completed.stdout
    assert "ROLLOUT_N=4\n" in completed.stdout
    assert "PROMPT_BATCH_SIZE=256\n" in completed.stdout
    assert "PPO_MINI_BATCH_SIZE=256\n" in completed.stdout
    assert "TOTAL_TRAJECTORIES=1024\n" in completed.stdout
    assert "trainer.total_training_steps=50" in completed.stdout
    assert "trainer.resume_mode=disable" in completed.stdout
    required = [
        "algorithm.difficulty_aware_opd.easy_group_correct_count=4",
        "algorithm.difficulty_aware_opd.easy_prompt_style=concise",
        "algorithm.difficulty_aware_opd.default_prompt_style=normal",
        "algorithm.difficulty_aware_opd.easy_esr_beta=0.20",
        "algorithm.difficulty_aware_opd.non_easy_esr_beta=0.50",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.kl_loss_coef=0",
        "algorithm.use_kl_in_reward=False",
        "algorithm.candidate_selection.enabled=False",
        "actor_rollout_ref.actor.policy_loss.length_aware_opd=False",
        "algorithm.rethinking_opd_probe.enabled=False",
        "trainer.save_freq=20",
        "trainer.total_training_steps=50",
    ]
    for value in required:
        assert value in completed.stdout


def test_ablation_rejects_changed_scientific_volume():
    completed = _run(ROLLOUT_N="8")

    assert completed.returncode == 2
    assert "ROLLOUT_N must remain pinned to 4 (got 8)" in completed.stderr


def test_ablation_rejects_changed_training_data():
    completed = _run(TRAIN_DATA="/tmp/random.parquet")

    assert completed.returncode == 2
    assert "TRAIN_DATA must remain pinned" in completed.stderr


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("PROMPT_BATCH_SIZE", "128"),
        ("PPO_MINI_BATCH_SIZE", "1024"),
        ("TOTAL_TRAJECTORIES", "2048"),
        ("EXPECTED_GROUP_SIZE", "8"),
        ("TOTAL_TRAINING_STEPS", "49"),
        ("SAVE_FREQ", "50"),
        ("STUDENT_MODEL", "/tmp/student"),
        ("TEACHER_MODEL", "/tmp/teacher"),
    ],
)
def test_ablation_rejects_changed_pinned_value(name, value):
    completed = _run(**{name: value})

    assert completed.returncode == 2
    assert f"{name} must remain pinned" in completed.stderr


def test_ablation_delegates_to_existing_group_success_launcher():
    completed = _run()

    assert completed.returncode == 0, completed.stderr
    expected = str(REPO_DIR / "run_train_group_success_difficulty_opd.sh")
    assert f"bash {expected}" in completed.stdout
    assert "algorithm.difficulty_aware_opd.method=group_success_prompt_esr" in completed.stdout
    assert "actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=True" in completed.stdout
