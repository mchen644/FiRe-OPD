import fcntl
import json
import os
from pathlib import Path
import subprocess

import pytest


REPO_DIR = Path(__file__).resolve().parents[4]
LAUNCHER = REPO_DIR / "run_train_group_success_gradient_diverse_ablation.sh"
PRODUCTION_REPO_DIR = Path("/home/mchen/FiRe-OPD")
RUN_NAME = "opd-n4-graddiv12800-easy4of4-concise20-noneasynormal50-purerkl-step50"
SELECTED_DATA = "/home/mchen/FiRe-OPD/data/gradient_diversity/DeepMath-103K/train_gradient_diverse_12800.parquet"
SOURCE_DATA = "/home/mchen/FiRe-OPD/data/g-opd/DeepMath-103K/train_filtered_level6.parquet"
SELECTION_MANIFEST = "/home/mchen/FiRe-OPD/data/gradient_diversity/selection/manifest.json"
SELECTED_IDS = "/home/mchen/FiRe-OPD/data/gradient_diversity/selection/selected_ids.jsonl"
DIAGNOSTICS = "/home/mchen/FiRe-OPD/data/gradient_diversity/selection/diagnostics.json"
SELECTED_SHA256 = "caf303c5d151fdaed2e21eebc257917f13660c589906e3da0e569ffdfd59b059"
SOURCE_SHA256 = "de3350fdd00bc0410550098ea65179e2be873da99e4075f80de575fc17670597"
MANIFEST_SHA256 = "a1a45382ee577e24f9b455386f9f3adcab210ff40a83ea760c2110fb96f8f90a"
SELECTED_IDS_SHA256 = "a78c04cff10c148f45bf9c828e07cc9cf01398b48efeb707253273af359a037e"
DIAGNOSTICS_SHA256 = "d2105179f2ce5a7c796aef07136bbc1dce79b07b5bc2de4d2319b18b47ea761f"


def _valid_validator_report(**overrides) -> str:
    report = {
        "diagnostics": DIAGNOSTICS,
        "diagnostics_sha256": DIAGNOSTICS_SHA256,
        "eligible_rows": 57045,
        "max_prompt_tokens": 668,
        "median_prompt_tokens": 256.0,
        "min_prompt_tokens": 32,
        "p95_prompt_tokens": 512,
        "p99_prompt_tokens": 640,
        "prompt_token_limit": 2048,
        "prompts_over_limit": 0,
        "schema_equal": True,
        "selected_ids": 12800,
        "selected_ids_path": SELECTED_IDS,
        "selected_ids_sha256": SELECTED_IDS_SHA256,
        "selected_parquet": SELECTED_DATA,
        "selected_rows": 12800,
        "selected_sha256": SELECTED_SHA256,
        "selection_manifest": SELECTION_MANIFEST,
        "selection_manifest_sha256": MANIFEST_SHA256,
        "source_parquet": SOURCE_DATA,
        "source_rows": 57046,
        "source_rows_equal": True,
        "source_sha256": SOURCE_SHA256,
        "unique_prompts": 12800,
    }
    report.update(overrides)
    return json.dumps(report, separators=(",", ":"), sort_keys=True)


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


def _run_nondry(
    tmp_path,
    *,
    preflight_only="1",
    validator_output=None,
    validator_exit_code="0",
    **overrides,
):
    invocation_record = tmp_path / "validator-invocation.txt"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
{
  printf 'cwd=%s\\n' "$PWD"
  printf 'arg=%s\\n' "$@"
} >"${FAKE_PYTHON_RECORD:?}"
printf '%s\\n' "${FAKE_VALIDATOR_OUTPUT:?}"
exit "${FAKE_VALIDATOR_EXIT_CODE:?}"
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "ABLATION_DRY_RUN": "0",
            "ABLATION_PREFLIGHT_ONLY": preflight_only,
            "GROUP_SUCCESS_DRY_RUN": "0",
            "PYTHON_BIN": str(fake_python),
            "REPO_DIR": str(PRODUCTION_REPO_DIR),
            "FAKE_PYTHON_RECORD": str(invocation_record),
            "FAKE_VALIDATOR_OUTPUT": (
                _valid_validator_report()
                if validator_output is None
                else validator_output
            ),
            "FAKE_VALIDATOR_EXIT_CODE": validator_exit_code,
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


def _run_preflight(tmp_path, **overrides):
    return _run_nondry(tmp_path, **overrides)


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
        SELECTED_IDS,
        "--diagnostics",
        DIAGNOSTICS,
        "--tokenizer-path",
        "/home/mchen/FiRe-OPD/models/Qwen3-4B",
        "--expected-selected-sha256",
        "caf303c5d151fdaed2e21eebc257917f13660c589906e3da0e569ffdfd59b059",
        "--expected-source-sha256",
        SOURCE_SHA256,
        "--expected-manifest-sha256",
        MANIFEST_SHA256,
        "--expected-selected-ids-sha256",
        SELECTED_IDS_SHA256,
        "--expected-diagnostics-sha256",
        DIAGNOSTICS_SHA256,
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
        f"artifact_report={_valid_validator_report()}"
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
        f"artifact_report={_valid_validator_report()}",
        "resolved_contract_begin\n",
        "resolved_contract_end\n",
    ]
    for value in required:
        assert value in completed.stdout
    assert (
        f"bash {PRODUCTION_REPO_DIR / 'run_train_group_success_difficulty_opd.sh'}"
        not in completed.stdout
    )


def test_ablation_preflight_rejects_empty_validator_stdout_before_provenance(tmp_path):
    completed, invocation_record = _run_preflight(tmp_path, PYTHON_BIN="/bin/true")

    assert completed.returncode == 2
    assert "validator report" in completed.stderr
    assert not invocation_record.exists()
    assert "git_head=" not in completed.stdout
    assert (
        f"bash {PRODUCTION_REPO_DIR / 'run_train_group_success_difficulty_opd.sh'}"
        not in completed.stdout
    )


@pytest.mark.parametrize(
    ("validator_output", "case"),
    [
        ('{"fake_validator":"ok"}', "partial"),
        ("{not-json", "malformed"),
        ('{"value":NaN}', "nonfinite"),
        (_valid_validator_report(max_prompt_tokens=667), "mismatched"),
        (_valid_validator_report(unexpected=True), "unexpected-field"),
        (
            '{"selected_rows":12800,' + _valid_validator_report()[1:],
            "duplicate-field",
        ),
    ],
)
def test_ablation_preflight_rejects_untrusted_validator_stdout_before_provenance(
    tmp_path, validator_output, case
):
    completed, invocation_record = _run_nondry(
        tmp_path,
        validator_output=validator_output,
    )

    assert completed.returncode == 2, case
    assert "validator report" in completed.stderr
    assert invocation_record.exists()
    assert "git_head=" not in completed.stdout
    assert (
        f"bash {PRODUCTION_REPO_DIR / 'run_train_group_success_difficulty_opd.sh'}"
        not in completed.stdout
    )


def test_ablation_real_launch_rejects_held_experiment_lock_before_validator(tmp_path):
    lock_path = Path(f"/tmp/fire-opd-{os.getuid()}-{RUN_NAME}.launch.lock")
    try:
        with lock_path.open("w", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            completed, invocation_record = _run_nondry(
                tmp_path,
                preflight_only="0",
                validator_exit_code="79",
            )

        assert completed.returncode == 2
        assert f"training launch lock is already held: {lock_path}" in completed.stderr
        assert not invocation_record.exists()
        assert (
            f"bash {PRODUCTION_REPO_DIR / 'run_train_group_success_difficulty_opd.sh'}"
            not in completed.stdout
        )
    finally:
        lock_path.unlink(missing_ok=True)
