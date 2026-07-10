from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPOSITORY_ROOT / "run_select_gradient_diverse_deepmath.sh"


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _fake_runtime(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    runtime = tmp_path / "runtime with spaces"
    bin_dir = runtime / "bin"
    bin_dir.mkdir(parents=True)
    calls_path = runtime / "python-calls.jsonl"
    gpu_calls_path = runtime / "gpu-calls.jsonl"
    source_path = runtime / "source pool.parquet"
    source_path.write_bytes(b"fixture")
    reference_repo = runtime / "reference repo"
    reference_repo.mkdir()

    fake_python = bin_dir / "fake-python"
    _write_executable(
        fake_python,
        """#!/usr/bin/env python3
import json
import os
import pathlib
import sys
import time


def option(name):
    index = sys.argv.index(name)
    return sys.argv[index + 1]


if len(sys.argv) >= 2 and sys.argv[1] == "-":
    os.execv(os.environ["REAL_PYTHON"], [os.environ["REAL_PYTHON"], *sys.argv[1:]])

record = {
    "argv": sys.argv[1:],
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
}
with open(os.environ["FAKE_PYTHON_CALLS"], "a", encoding="utf-8") as handle:
    handle.write(json.dumps(record, sort_keys=True) + "\\n")

if sys.argv[1:3] == ["-m", "math_eval.prepare_deepmath_gradient_pool"]:
    time.sleep(float(os.environ.get("PREPARE_SLEEP", "0")))
    pathlib.Path(option("--output-jsonl")).write_text("{}\\n", encoding="utf-8")
    pathlib.Path(option("--manifest")).write_text("{}\\n", encoding="utf-8")
elif sys.argv[1:3] == ["-m", "math_eval.build_gradient_eligibility"]:
    report = {
        "source_row_count": int(os.environ.get("FAKE_SOURCE_ROWS", "57046")),
        "eligible_row_count": int(os.environ.get("FAKE_ELIGIBLE_ROWS", "57045")),
        "excluded_row_count": 1,
        "excluded_rows": [{"id": os.environ.get("FAKE_EXCLUDED_ID", "deepmath-level6-038794")}],
    }
    pathlib.Path(option("--output")).write_text(json.dumps(report), encoding="utf-8")
elif sys.argv[1:3] == ["-m", "math_eval.collect_prismatic_gradients"]:
    if "--validate-global-only" not in sys.argv:
        shard = option("--shard-index")
        time.sleep(float(os.environ.get("COLLECT_SLEEP", "0")))
        if shard == os.environ.get("FAIL_SHARD"):
            raise SystemExit(23)
elif sys.argv[1:3] == ["-m", "math_eval.select_gradient_diverse_deepmath"]:
    for flag in ("--output-parquet", "--selected-ids", "--diagnostics", "--manifest"):
        pathlib.Path(option(flag)).write_text("fixture\\n", encoding="utf-8")
else:
    raise SystemExit("unexpected fake Python invocation: " + repr(sys.argv))
""",
    )

    fake_nvidia_smi = bin_dir / "nvidia-smi"
    _write_executable(
        fake_nvidia_smi,
        """#!/usr/bin/env python3
import json
import os
import sys

gpu_id = next((arg.split("=", 1)[1] for arg in sys.argv[1:] if arg.startswith("--id=")), None)
with open(os.environ["FAKE_GPU_CALLS"], "a", encoding="utf-8") as handle:
    handle.write(json.dumps({"argv": sys.argv[1:], "gpu_id": gpu_id}, sort_keys=True) + "\\n")
if gpu_id is None:
    raise SystemExit(4)
if any("--query-gpu=" in arg for arg in sys.argv[1:]):
    print(f"{gpu_id}, GPU-fake-{gpu_id}, 5")
elif any("--query-compute-apps=" in arg for arg in sys.argv[1:]):
    if gpu_id == os.environ.get("OCCUPIED_GPU"):
        print("424242")
else:
    raise SystemExit(5)
""",
    )

    output_root = runtime / "output root"
    log_root = runtime / "log root"
    env = os.environ.copy()
    env.update(
        {
            "GPU_IDS": "7,2",
            "PYTHON_BIN": str(fake_python),
            "REFERENCE_REPO": str(reference_repo),
            "SOURCE_PARQUET": str(source_path),
            "OUTPUT_ROOT": str(output_root),
            "LOG_ROOT": str(log_root),
            "FAKE_PYTHON_CALLS": str(calls_path),
            "FAKE_GPU_CALLS": str(gpu_calls_path),
            "REAL_PYTHON": sys.executable,
            "PATH": f"{bin_dir}{os.pathsep}{env['PATH']}",
        }
    )
    return env, calls_path, gpu_calls_path


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _module_call_names(calls: list[dict]) -> list[str]:
    return [call["argv"][1] for call in calls]


def test_launcher_static_contract_contains_pinned_production_gates():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "set -euo pipefail" in text
    for pinned in (
        "/home/mchen/miniconda3/envs/gvendi-opd/bin/python",
        "/home/mchen/prismatic-synthesis-reference",
        "57046",
        "57045",
        "deepmath-level6-038794",
        "12800",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "7ae557604adf67be50417f59c2c2f167def9a775",
        "5cf055d1fe3d7a2eb19719ac020211469736ae44",
        "d9484cd3b5991030b901ac4a3a9e2472dbfac2ad",
        "0.10",
        "0.01",
        "42,43",
    ):
        assert pinned in text
    assert "flock -n" in text
    assert "nvidia-smi" in text
    assert "--query-gpu=index,uuid,memory.used" in text
    assert "--query-compute-apps=pid" in text
    assert "CUDA_VISIBLE_DEVICES=" in text
    assert "--device cuda:0" in text
    assert "--eligibility-report" in text
    assert "--validate-global-only" in text
    assert "pip install" not in text
    assert "conda install" not in text

    prepare = text.index("-m math_eval.prepare_deepmath_gradient_pool")
    eligibility = text.index("-m math_eval.build_gradient_eligibility")
    collection = text.index("--shard-index")
    global_validation = text.index("--validate-global-only")
    selection = text.index("-m math_eval.select_gradient_diverse_deepmath")
    assert prepare < eligibility < collection < global_validation < selection


def test_launcher_fake_cli_runs_stages_in_order_and_maps_logical_shards(tmp_path):
    env, calls_path, gpu_calls_path = _fake_runtime(tmp_path)

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPOSITORY_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    calls = _read_jsonl(calls_path)
    names = _module_call_names(calls)
    assert names[0:2] == [
        "math_eval.prepare_deepmath_gradient_pool",
        "math_eval.build_gradient_eligibility",
    ]
    collector_calls = [
        call
        for call in calls
        if call["argv"][1] == "math_eval.collect_prismatic_gradients"
        and "--validate-global-only" not in call["argv"]
    ]
    assert len(collector_calls) == 2
    by_shard = {
        call["argv"][call["argv"].index("--shard-index") + 1]: call
        for call in collector_calls
    }
    assert by_shard["0"]["cuda_visible_devices"] == "7"
    assert by_shard["1"]["cuda_visible_devices"] == "2"
    for shard, call in by_shard.items():
        assert call["argv"][call["argv"].index("--num-shards") + 1] == "2"
        assert call["argv"][call["argv"].index("--device") + 1] == "cuda:0"
        assert "--eligibility-report" in call["argv"]
        assert shard in {"0", "1"}

    global_index = next(
        index
        for index, call in enumerate(calls)
        if "--validate-global-only" in call["argv"]
    )
    selection_index = names.index("math_eval.select_gradient_diverse_deepmath")
    assert global_index > max(calls.index(call) for call in collector_calls)
    assert selection_index > global_index
    assert "--eligibility-report" in calls[global_index]["argv"]
    assert calls[selection_index]["cuda_visible_devices"] == "7"

    gpu_calls = _read_jsonl(gpu_calls_path)
    assert {call["gpu_id"] for call in gpu_calls} == {"7", "2"}
    for gpu_id in ("7", "2"):
        queries = [call["argv"] for call in gpu_calls if call["gpu_id"] == gpu_id]
        assert any("--query-gpu=index,uuid,memory.used" in args for args in queries)
        assert any("--query-compute-apps=pid" in args for args in queries)

    assert "GPU-fake-7" in result.stdout
    assert "GPU-fake-2" in result.stdout
    assert "gradient-diverse selection complete" in result.stdout


def test_launcher_propagates_collector_failure_before_global_validation(tmp_path):
    env, calls_path, _ = _fake_runtime(tmp_path)
    env["FAIL_SHARD"] = "1"

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPOSITORY_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode != 0
    calls = _read_jsonl(calls_path)
    assert not any("--validate-global-only" in call["argv"] for call in calls)
    assert "math_eval.select_gradient_diverse_deepmath" not in _module_call_names(calls)


def test_launcher_refuses_requested_occupied_gpu_before_collectors(tmp_path):
    env, calls_path, gpu_calls_path = _fake_runtime(tmp_path)
    env["OCCUPIED_GPU"] = "2"

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPOSITORY_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode != 0
    assert "GPU 2" in result.stdout + result.stderr
    calls = _read_jsonl(calls_path)
    assert not any(
        call["argv"][1] == "math_eval.collect_prismatic_gradients"
        for call in calls
    )
    assert {call["gpu_id"] for call in _read_jsonl(gpu_calls_path)} <= {"7", "2"}


@pytest.mark.parametrize("gpu_ids", ["", " ", "7,7", "7,", "7,a"])
def test_launcher_rejects_empty_duplicate_or_invalid_gpu_ids(tmp_path, gpu_ids):
    env, calls_path, gpu_calls_path = _fake_runtime(tmp_path)
    env["GPU_IDS"] = gpu_ids

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPOSITORY_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode != 0
    assert _read_jsonl(calls_path) == []
    assert _read_jsonl(gpu_calls_path) == []


def test_launcher_master_lock_rejects_duplicate_concurrent_launch(tmp_path):
    env, calls_path, _ = _fake_runtime(tmp_path)
    env["PREPARE_SLEEP"] = "1.0"
    first = subprocess.Popen(
        ["bash", str(LAUNCHER)],
        cwd=REPOSITORY_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if _read_jsonl(calls_path):
                break
            time.sleep(0.02)
        else:
            pytest.fail("first launcher did not enter preparation")

        second = subprocess.run(
            ["bash", str(LAUNCHER)],
            cwd=REPOSITORY_ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=5,
        )
        assert second.returncode != 0
        assert "already holds" in second.stdout + second.stderr
        assert first.wait(timeout=10) == 0
    finally:
        if first.poll() is None:
            first.terminate()
            first.wait(timeout=5)


def test_launcher_can_resume_by_rerunning_the_same_cli_contract(tmp_path):
    env, calls_path, _ = _fake_runtime(tmp_path)

    for _ in range(2):
        result = subprocess.run(
            ["bash", str(LAUNCHER)],
            cwd=REPOSITORY_ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=20,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    names = _module_call_names(_read_jsonl(calls_path))
    assert names.count("math_eval.prepare_deepmath_gradient_pool") == 2
    assert names.count("math_eval.build_gradient_eligibility") == 2
    assert names.count("math_eval.collect_prismatic_gradients") == 6
    assert names.count("math_eval.select_gradient_diverse_deepmath") == 2
