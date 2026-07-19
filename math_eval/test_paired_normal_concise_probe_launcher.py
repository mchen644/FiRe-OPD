from __future__ import annotations

import os
import subprocess
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[1]
LAUNCHER = WORKTREE / "run_paired_normal_concise_probe.sh"


def test_launcher_dry_run_pins_two_model_sequential_probe_contract(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    log_root = tmp_path / "logs"
    env = os.environ.copy()
    env.update(
        {
            "PAIRED_PROBE_DRY_RUN": "1",
            "PAIRED_PROBE_RUN_ID": "test-dry-run",
            "PAIRED_PROBE_OUTPUT_ROOT": str(output_root),
            "PAIRED_PROBE_LOG_ROOT": str(log_root),
        }
    )

    completed = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=WORKTREE,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    output = completed.stdout + completed.stderr

    required = [
        "train_filtered_level6.parquet",
        "de3350fdd00bc0410550098ea65179e2be873da99e4075f80de575fc17670597",
        "/models/Qwen3-4B",
        "global_step_50_hf",
        "--model-label base",
        "--model-label adaptive_step50",
        "--sample-size 128",
        "--seed 42",
        "--expected-count 128",
        "--tensor-parallel-size 4",
        "--max-model-len 40960",
        "paired_normal_concise_probe.py prepare",
        "paired_normal_concise_probe.py generate",
        "paired_normal_concise_probe.py analyze",
        "validate_paired_normal_concise_probe.py",
        "CUDA_VISIBLE_DEVICES=0,1,2,3",
        "VLLM_WORKER_MULTIPROC_METHOD=spawn",
        "PAIRED_NORMAL_CONCISE_PROBE_DONE_test-dry-run:0",
    ]
    for token in required:
        assert token in output
    assert output.index("--model-label base") < output.index("--model-label adaptive_step50")
    assert output.index("--model-label adaptive_step50") < output.index(
        "paired_normal_concise_probe.py analyze"
    )
    assert not output_root.exists()
    assert not log_root.exists()


def test_launcher_source_forbids_nested_scheduler_and_mutating_modes() -> None:
    source = LAUNCHER.read_text()

    assert "srun" not in source
    assert "wandb" not in source.lower()
    assert "resume" not in source.lower()
    assert "overwrite" not in source.lower()
    assert "CUDA_VISIBLE_DEVICES" in source
    assert "JOB_ID=429" in source
    assert "set -euo pipefail" in source
