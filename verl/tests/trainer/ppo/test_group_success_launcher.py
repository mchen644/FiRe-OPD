import os
from pathlib import Path
import subprocess


REPO_DIR = Path(__file__).resolve().parents[4]
LAUNCHER = REPO_DIR / "run_train_group_success_difficulty_opd.sh"


def _run_launcher(**env_overrides):
    env = os.environ.copy()
    env.update({"GROUP_SUCCESS_DRY_RUN": "1", **env_overrides})
    return subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_DIR,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_group_success_launcher_dry_run_prints_clean_production_contract():
    completed = _run_launcher()

    assert completed.returncode == 0, completed.stderr
    required = [
        "data.train_batch_size=256",
        "actor_rollout_ref.rollout.n=4",
        "actor_rollout_ref.actor.ppo_mini_batch_size=256",
        "algorithm.difficulty_aware_opd.method=group_success_prompt_esr",
        "algorithm.difficulty_aware_opd.easy_esr_beta=0.20",
        "algorithm.difficulty_aware_opd.non_easy_esr_beta=0.50",
        "algorithm.difficulty_aware_opd.hard_entropy_coef=0.0",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=True",
        "algorithm.candidate_selection.enabled=False",
        "algorithm.rethinking_opd_probe.enabled=False",
        "trainer.total_training_steps=50",
        "trainer.resume_mode=disable",
    ]
    for value in required:
        assert value in completed.stdout


def test_group_success_launcher_rejects_inconsistent_total_trajectory_count():
    completed = _run_launcher(PROMPT_BATCH_SIZE="255")

    assert completed.returncode == 2
    assert "prompt batch times rollout count must equal TOTAL_TRAJECTORIES" in completed.stderr


def test_group_success_launcher_requires_prompt_level_ppo_mini_batch_size():
    completed = _run_launcher(PPO_MINI_BATCH_SIZE="1024")

    assert completed.returncode == 2
    assert "PPO_MINI_BATCH_SIZE must equal PROMPT_BATCH_SIZE" in completed.stderr


def test_group_success_launcher_rejects_non_integer_volume_setting_cleanly():
    completed = _run_launcher(PROMPT_BATCH_SIZE="many")

    assert completed.returncode == 2
    assert "PROMPT_BATCH_SIZE must be a positive integer, got many" in completed.stderr
    assert "bad substitution" not in completed.stderr
