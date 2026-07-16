import os
from pathlib import Path
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "run_select_gradient_diverse_deepmath_51200.sh"


def _run(**overrides):
    env = os.environ.copy()
    env.update(
        {
            "SFTGRAD51200_DRY_RUN": "1",
            "PYTHON_BIN": "/not/used/in/dry-run",
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


def test_51200_selection_dry_run_uses_only_frozen_gradients() -> None:
    completed = _run()
    assert completed.returncode == 0, completed.stderr
    required = [
        "selection_profile=vanilla_51200",
        "target_rows=51200",
        "expected_source_rows=57046",
        "expected_eligible_rows=57045",
        "selection_51200/selected_ids.jsonl",
        "train_gradient_diverse_51200.parquet",
        "--selection-profile vanilla_51200",
        "--target-size 51200",
        "--frozen-prefix-selected-ids",
        "math_eval.select_gradient_diverse_deepmath",
        "ee8d55c943577888f14052b3953a0a8aa07d26076e50ddffa41a102b60027344",
        "d9484cd3b5991030b901ac4a3a9e2472dbfac2ad",
        "caf303c5d151fdaed2e21eebc257917f13660c589906e3da0e569ffdfd59b059",
        "a1a45382ee577e24f9b455386f9f3adcab210ff40a83ea760c2110fb96f8f90a",
        "a78c04cff10c148f45bf9c828e07cc9cf01398b48efeb707253273af359a037e",
        "d2105179f2ce5a7c796aef07136bbc1dce79b07b5bc2de4d2319b18b47ea761f",
    ]
    for value in required:
        assert value in completed.stdout
    assert "collect_prismatic_gradients" not in completed.stdout
    assert "prepare_deepmath_gradient_pool" not in completed.stdout


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("TARGET_ROWS", "12800"),
        ("SELECTION_PROFILE", "coreset_12800"),
        ("PRIMARY_CLUSTER_RATIO", "0.01"),
        ("CLUSTER_SEEDS", "42"),
        ("SOURCE_SHA256", "0" * 64),
        ("PREPARED_JSONL_SHA256", "0" * 64),
        ("REFERENCE_COMMIT", "0" * 40),
    ],
)
def test_51200_selection_rejects_changed_pin(name: str, value: str) -> None:
    completed = _run(**{name: value})
    assert completed.returncode == 2
    assert f"{name} must remain pinned" in completed.stderr


def test_51200_selection_launcher_contains_fail_closed_runtime_gates() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    required = [
        "SFTGRAD51200_PREFLIGHT_ONLY",
        "sha256sum",
        "flock -n",
        "validate_opd_cli_runtime",
        "expected_gpus=4",
        "SLURM_JOB_ID",
        "CUDA_VISIBLE_DEVICES",
        "git status --porcelain=v1",
        "--validate-global-only",
        "REFERENCE_TREE",
    ]
    for value in required:
        assert value in text
    assert "pip install" not in text
    assert "conda install" not in text
    assert "srun " not in text
