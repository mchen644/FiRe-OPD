from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "run_prismatic_lite_qwen3_2k_pilot.sh"


def _text() -> str:
    return LAUNCHER.read_text(encoding="utf-8")


def test_launcher_exists_is_executable_and_has_strict_shell_contract() -> None:
    text = _text()

    assert LAUNCHER.stat().st_mode & 0o111
    assert text.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    assert not re.search(r"\bsrun\b", text)


def test_launcher_pins_worktree_models_inputs_and_environments() -> None:
    text = _text()

    required = (
        "/home/mchen/FiRe-OPD/.worktrees/vanilla-sft-gradient-51200-opd",
        "/home/mchen/FiRe-OPD/models/Qwen3-30B-A3B-Instruct-2507",
        "/home/mchen/prismatic-synthesis-reference",
        "/home/mchen/miniconda3/envs/gvendi-opd/bin/python",
        "/home/mchen/miniconda3/envs/verl/bin/python",
        "ee8d55c943577888f14052b3953a0a8aa07d26076e50ddffa41a102b60027344",
        "cc8d5b888def37761519e15c6ef722e5a22d91ae2342c6d669f008bcff3dcbb6",
        "9a534118a08e736a15e806933d5cf90c2a99822fee28bf18644e5ff344ce3ae2",
        "d9484cd3b5991030b901ac4a3a9e2472dbfac2ad",
    )
    assert all(value in text for value in required)
    assert "status --porcelain=v1 --untracked-files=all" in text
    assert "HEAD^{tree}" in text


def test_dry_run_is_cpu_only_checks_absent_targets_and_prints_exact_commands() -> None:
    text = _text()

    assert 'case "${DRY_RUN:-0}"' in text
    dry_section = text[text.index('if [[ "${DRY_RUN:-0}" == "1" ]]') :]
    assert "verify_frozen_inputs" in dry_section
    assert "verify_model_tree" in dry_section
    assert "require_absent_initial_targets" in dry_section
    assert "print_phase_commands" in dry_section
    assert "QwenVllmBackend" not in dry_section.split("fi", 1)[0]
    assert "CUDA_VISIBLE_DEVICES=" not in dry_section.split("fi", 1)[0]


def test_launcher_runtime_gate_requires_opd_cli_exact_tokens_idle_and_walltime() -> None:
    text = _text()

    assert "validate_opd_cli_runtime" in text
    assert '[[ "$allocated_tokens" == "0,1,2,3" ]]' in text
    assert "require_idle=True" in text
    assert "opd-CLI" in text
    assert "require_remaining_walltime" in text
    assert "REQUIRED_REMAINING_SECONDS=172800" in text
    assert "require_no_stale_processes" in text
    assert "wait_for_gpu_drain" in text


def test_launcher_has_frozen_phase_order_and_calibration_stop_gate() -> None:
    text = _text()
    ordered = (
        "run_prepare",
        "run_calibration_generation",
        "run_calibration_gradients",
        "run_calibration_analysis",
        "run_problem_generation",
        "run_candidate_solution_generation",
        "run_quality_review",
        "run_candidate_gradients",
        "run_selection_analysis",
        "run_final_validation",
    )
    positions = [text.index(f"{name}\n", text.index("=== canonical execution ===")) for name in ordered]

    assert positions == sorted(positions)
    assert "NO_GO_EXIT_CODE=20" in text
    assert "handle_early_no_go" in text
    assert "STAGE_COMPLETE.json" in text
    assert "PRISMATIC_LITE_DONE:" in text


def test_qwen_and_proxy_phases_have_exact_gpu_topology() -> None:
    text = _text()

    assert text.count('CUDA_VISIBLE_DEVICES="0,1,2,3"') >= 3
    assert 'for shard_index in 0 1 2 3; do' in text
    assert 'CUDA_VISIBLE_DEVICES="$shard_index"' in text
    assert "--tensor-parallel-size" not in text  # topology is internal and frozen in Python
    assert "--num-shards 4" in text
    assert "--device cuda:0" in text
    assert "--validate-global-only" in text


def test_launcher_forbids_resume_and_overwrite_and_uses_one_master_log() -> None:
    text = _text()

    assert "resume" not in " ".join(
        line for line in text.splitlines() if "forbidden" not in line.lower()
    ).lower()
    assert "require_absent" in text
    assert "flock -n" in text
    assert "tee -a" in text
    assert "failed_attempts" in text
    assert "archive" in text.lower()
