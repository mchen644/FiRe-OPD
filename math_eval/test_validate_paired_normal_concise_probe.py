from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from math_eval.paired_normal_concise_probe import analyze_run
from math_eval.test_paired_normal_concise_probe import _complete_analyzable_run
from math_eval.validate_paired_normal_concise_probe import validate_run


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _rehash(run_dir: Path, relative: str) -> None:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"][relative] = hashlib.sha256((run_dir / relative).read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))


def _validated_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    _complete_analyzable_run(run_dir)
    analyze_run(run_dir=run_dir, expected_count=4)
    return run_dir


def test_validate_run_accepts_complete_independently_recomputed_artifacts(tmp_path: Path) -> None:
    run_dir = _validated_run(tmp_path)

    report = validate_run(run_dir=run_dir, expected_count=4)

    assert report["gate"] == "pass"
    assert report["models"] == ["base", "adaptive_step50"]
    assert report["record_count_per_model"] == 4


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("cap", "concise cap"),
        ("seed", "sample"),
        ("prefix", "prefix"),
        ("counterfactual_mode", "counterfactual mode"),
        ("forced_prefix_length", "prefix length"),
        ("continuation_seed", "continuation seed"),
        ("continuation_max_tokens", "remaining budget"),
        ("continuation_length", "continuation length"),
        ("continuation_token", "combined token IDs"),
        ("continuation_prompt_token", "prompt capped suffix"),
        ("continuation_prompt_length", "prompt length"),
        ("manifest_mode", "manifest counterfactual mode"),
        ("generation_mode", "generation counterfactual mode"),
        ("route", "quadrant"),
        ("summary", "summary"),
        ("order", "sample identities"),
    ],
)
def test_validate_run_rejects_semantic_tampering(
    tmp_path: Path, mutation: str, match: str
) -> None:
    run_dir = _validated_run(tmp_path)
    records_path = run_dir / "base" / "records.jsonl"
    records = _read_jsonl(records_path)

    if mutation == "cap":
        records[0]["concise"]["max_tokens"] -= 1
        _write_jsonl(records_path, records)
        _rehash(run_dir, "base/records.jsonl")
    elif mutation == "seed":
        records[0]["request_seed"] += 1
        _write_jsonl(records_path, records)
        _rehash(run_dir, "base/records.jsonl")
    elif mutation == "prefix":
        records[1]["relaxed_concise"]["token_ids"][0] = 999
        _write_jsonl(records_path, records)
        _rehash(run_dir, "base/records.jsonl")
    elif mutation == "counterfactual_mode":
        records[1]["relaxed_concise"]["counterfactual_mode"] = "independent_resample"
        _write_jsonl(records_path, records)
        _rehash(run_dir, "base/records.jsonl")
    elif mutation == "forced_prefix_length":
        records[1]["relaxed_concise"]["forced_prefix_length"] -= 1
        _write_jsonl(records_path, records)
        _rehash(run_dir, "base/records.jsonl")
    elif mutation == "continuation_seed":
        records[1]["relaxed_concise"]["continuation_seed"] += 1
        _write_jsonl(records_path, records)
        _rehash(run_dir, "base/records.jsonl")
    elif mutation == "continuation_max_tokens":
        records[1]["relaxed_concise"]["continuation_max_tokens"] += 1
        _write_jsonl(records_path, records)
        _rehash(run_dir, "base/records.jsonl")
    elif mutation == "continuation_length":
        records[1]["relaxed_concise"]["continuation_length"] += 1
        _write_jsonl(records_path, records)
        _rehash(run_dir, "base/records.jsonl")
    elif mutation == "continuation_token":
        records[1]["relaxed_concise"]["continuation_token_ids"][0] = 999
        _write_jsonl(records_path, records)
        _rehash(run_dir, "base/records.jsonl")
    elif mutation == "continuation_prompt_token":
        records[1]["relaxed_concise"]["continuation_prompt_token_ids"][-1] = 999
        _write_jsonl(records_path, records)
        _rehash(run_dir, "base/records.jsonl")
    elif mutation == "continuation_prompt_length":
        records[1]["relaxed_concise"]["continuation_prompt_length"] -= 1
        _write_jsonl(records_path, records)
        _rehash(run_dir, "base/records.jsonl")
    elif mutation == "manifest_mode":
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["protocol"]["relaxed_counterfactual_mode"] = "independent_resample"
        manifest_path.write_text(json.dumps(manifest))
    elif mutation == "generation_mode":
        generation_path = run_dir / "base" / "generation.json"
        generation = json.loads(generation_path.read_text())
        generation["relaxed_counterfactual_mode"] = "independent_resample"
        generation_path.write_text(json.dumps(generation))
        _rehash(run_dir, "base/generation.json")
    elif mutation == "route":
        records[0]["quadrant"] = "both_wrong"
        _write_jsonl(records_path, records)
        _rehash(run_dir, "base/records.jsonl")
    elif mutation == "summary":
        summary_path = run_dir / "base" / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["quadrant_counts"]["compression_safe"] += 1
        summary_path.write_text(json.dumps(summary))
        _rehash(run_dir, "base/summary.json")
    elif mutation == "order":
        records[0], records[1] = records[1], records[0]
        _write_jsonl(records_path, records)
        _rehash(run_dir, "base/records.jsonl")
    else:
        raise AssertionError(mutation)

    with pytest.raises(ValueError, match=match):
        validate_run(run_dir=run_dir, expected_count=4)


def test_validate_run_rejects_artifact_hash_tampering(tmp_path: Path) -> None:
    run_dir = _validated_run(tmp_path)
    with (run_dir / "comparison.md").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")

    with pytest.raises(ValueError, match="artifact SHA256"):
        validate_run(run_dir=run_dir, expected_count=4)
