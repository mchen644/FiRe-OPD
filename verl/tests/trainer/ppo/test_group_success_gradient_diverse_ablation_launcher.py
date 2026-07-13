import os
from pathlib import Path
import subprocess

import pytest


REPO_DIR = Path(__file__).resolve().parents[4]
LAUNCHER = REPO_DIR / "run_train_group_success_gradient_diverse_ablation.sh"
PRODUCTION_REPO_DIR = Path("/home/mchen/FiRe-OPD")
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


def _run_preflight(tmp_path, **overrides):
    invocation_record = tmp_path / "validator-invocation.txt"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
{
  printf 'cwd=%s\\n' "$PWD"
  printf 'arg=%s\\n' "$@"
} >"${FAKE_PYTHON_RECORD:?}"
printf '{"fake_validator":"ok"}\\n'
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "ABLATION_DRY_RUN": "0",
            "ABLATION_PREFLIGHT_ONLY": "1",
            "GROUP_SUCCESS_DRY_RUN": "0",
            "PYTHON_BIN": str(fake_python),
            "REPO_DIR": str(PRODUCTION_REPO_DIR),
            "FAKE_PYTHON_RECORD": str(invocation_record),
            **overrides,
        }
    )
    completed = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_DIR,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed, invocation_record


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


def test_ablation_preflight_rejects_nonproduction_repo_before_validator(tmp_path):
    completed, invocation_record = _run_preflight(tmp_path, REPO_DIR=str(REPO_DIR))

    assert completed.returncode == 2
    assert "REPO_DIR must remain pinned to /home/mchen/FiRe-OPD" in completed.stderr
    assert not invocation_record.exists()


@pytest.mark.parametrize("inherited_value", ["1", "true"])
def test_ablation_preflight_rejects_inherited_child_dry_run_before_validator(
    tmp_path, inherited_value
):
    completed, invocation_record = _run_preflight(
        tmp_path, GROUP_SUCCESS_DRY_RUN=inherited_value
    )

    assert completed.returncode == 2
    assert "GROUP_SUCCESS_DRY_RUN must be unset or 0" in completed.stderr
    assert not invocation_record.exists()


def test_ablation_preflight_runs_validator_from_production_repo_with_pinned_arguments(
    tmp_path,
):
    completed, invocation_record = _run_preflight(tmp_path)

    assert completed.returncode == 0, completed.stderr
    recorded = invocation_record.read_text(encoding="utf-8").splitlines()
    assert recorded[0] == f"cwd={PRODUCTION_REPO_DIR}"
    assert [line.removeprefix("arg=") for line in recorded[1:]] == [
        "-m",
        "math_eval.validate_gradient_diverse_training_data",
        "--selected-parquet",
        "/home/mchen/FiRe-OPD/data/gradient_diversity/DeepMath-103K/train_gradient_diverse_12800.parquet",
        "--source-parquet",
        "/home/mchen/FiRe-OPD/data/g-opd/DeepMath-103K/train_filtered_level6.parquet",
        "--selection-manifest",
        "/home/mchen/FiRe-OPD/data/gradient_diversity/selection/manifest.json",
        "--selected-ids",
        "/home/mchen/FiRe-OPD/data/gradient_diversity/selection/selected_ids.jsonl",
        "--tokenizer-path",
        "/home/mchen/FiRe-OPD/models/Qwen3-4B",
        "--expected-selected-sha256",
        "caf303c5d151fdaed2e21eebc257917f13660c589906e3da0e569ffdfd59b059",
        "--expected-source-sha256",
        "de3350fdd00bc0410550098ea65179e2be873da99e4075f80de575fc17670597",
        "--expected-rows",
        "12800",
        "--expected-source-rows",
        "57046",
        "--expected-eligible-rows",
        "57045",
        "--max-prompt-tokens",
        "2048",
        "--checkpoint-dir",
        f"/home/mchen/FiRe-OPD/checkpoints/{RUN_NAME}",
    ]


def test_ablation_preflight_prints_provenance_after_validation_without_delegating(
    tmp_path,
):
    completed, invocation_record = _run_preflight(tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert invocation_record.exists()
    assert completed.stdout.index("single_variable_contract=PASS") < completed.stdout.index(
        'artifact_report={"fake_validator":"ok"}'
    )
    required = [
        "git_head=",
        "git_status_begin\n",
        "git_status_end\n",
        "git_diff_sha256=",
        "group_launcher_sha256=",
        "base_launcher_sha256=",
        "ablation_launcher_sha256=",
        "selected_parquet_sha256=caf303c5d151fdaed2e21eebc257917f13660c589906e3da0e569ffdfd59b059",
        'artifact_report={"fake_validator":"ok"}',
        "resolved_contract_begin\n",
        "resolved_contract_end\n",
    ]
    for value in required:
        assert value in completed.stdout
    assert (
        f"bash {PRODUCTION_REPO_DIR / 'run_train_group_success_difficulty_opd.sh'}"
        not in completed.stdout
    )
