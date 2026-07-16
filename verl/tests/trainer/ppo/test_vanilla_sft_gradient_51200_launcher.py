import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
LAUNCHER = REPO_ROOT / "run_train_vanilla_sft_gradient_51200.sh"
RUN_NAME = "opd-strong-to-weak-sftgrad51200-rawprompt-4gpu-tp4-refmb4-rollmb4"


def _run(**overrides):
    env = os.environ.copy()
    env.update(
        {
            "VANILLA_SFTGRAD_DRY_RUN": "1",
            "PYTHON_BIN": "/not/used/in/dry-run",
            "REPO_DIR": str(REPO_ROOT),
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


def test_candidate_dry_run_is_a_data_only_vanilla_contract() -> None:
    completed = _run()
    assert completed.returncode == 0, completed.stderr
    required = [
        "single_variable_contract=PASS",
        "artifact_validation=SKIPPED_DRY_RUN",
        f"EXPERIMENT_NAME={RUN_NAME}",
        "train_gradient_diverse_51200.parquet",
        "ROLLOUT_N=1",
        "PROMPT_BATCH_SIZE=1024",
        "TOTAL_TRAINING_STEPS=50",
        "trainer.val_before_train=True",
        "trainer.test_freq=10",
        "trainer.resume_mode=disable",
        "bash " + str(REPO_ROOT / "run_train_vanilla_opd.sh"),
    ]
    for value in required:
        assert value in completed.stdout


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("TRAIN_DATA", "/tmp/other.parquet"),
        ("EXPERIMENT_NAME", "other"),
        ("ROLLOUT_N", "4"),
        ("PROMPT_BATCH_SIZE", "256"),
        ("TOTAL_TRAINING_STEPS", "49"),
        ("STUDENT_MODEL", "/tmp/student"),
        ("TEACHER_MODEL", "/tmp/teacher"),
        ("DATA_ROOT", "/tmp/data"),
        ("VAL_DATA", "['/tmp/a.parquet']"),
        ("CHECKPOINT_DIR", "/tmp/checkpoint"),
        ("LOG_FILE", "/tmp/run.log"),
        ("PPO_MINI_BATCH_SIZE", "512"),
        ("TOTAL_TRAJECTORIES", "2048"),
        ("SAVE_FREQ", "25"),
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
def test_candidate_rejects_changed_pin(name: str, value: str) -> None:
    completed = _run(**{name: value})
    assert completed.returncode == 2
    assert f"{name} must remain pinned" in completed.stderr


def test_candidate_wrapper_contains_artifact_provenance_and_gpu_gates() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    required = [
        "validate_gradient_diverse_training_data",
        "--expected-selection-profile",
        "vanilla_51200",
        "--frozen-prefix-selected-ids",
        "--expected-frozen-prefix-rows",
        "12800",
        "--checkpoint-dir",
        "git status --porcelain=v1",
        "validate_opd_cli_runtime",
        "expected_gpus=4",
        "flock -n",
        "VANILLA_SFTGRAD_PREFLIGHT_ONLY",
        "runtime_versions",
    ]
    for value in required:
        assert value in text
    assert "pip install" not in text
    assert "conda install" not in text
    assert "srun " not in text
