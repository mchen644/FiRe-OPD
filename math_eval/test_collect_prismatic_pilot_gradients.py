from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from math_eval import collect_prismatic_pilot_gradients as pilot
from math_eval.collect_prismatic_gradients import (
    MODEL_NAME,
    MODEL_REVISION,
    PROJECTION_DIM,
    PROJECTION_SEED,
)
from math_eval.prismatic_lite_pilot_artifacts import sha256_file


FIELDS = ("id", "question_id", "solution_index", "prompt", "completion")


def _write_rows(path: Path, count: int = 6) -> list[dict]:
    rows = [
        {
            "id": f"q-{index // 2}.solution-{index % 2}",
            "question_id": f"q-{index // 2}",
            "solution_index": index % 2,
            "prompt": f"Prompt {index // 2}",
            "completion": f"Solution {index}",
        }
        for index in range(count)
    ]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return rows


def test_load_pilot_gradient_input_requires_hash_exact_fields_and_order(
    tmp_path: Path,
) -> None:
    path = tmp_path / "input.jsonl"
    expected = _write_rows(path)

    assert pilot.load_pilot_gradient_input(path, sha256_file(path)) == expected
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        pilot.load_pilot_gradient_input(path, "0" * 64)

    bad = dict(expected[0])
    bad["extra"] = 1
    path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exact fields"):
        pilot.load_pilot_gradient_input(path, sha256_file(path))


def test_load_pilot_gradient_input_rejects_question_solution_mismatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "input.jsonl"
    rows = _write_rows(path)
    rows[0]["id"] = "wrong"
    path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="derived ID"):
        pilot.load_pilot_gradient_input(path, sha256_file(path))


def test_validate_pilot_manifest_binds_exact_file_and_sha(tmp_path: Path) -> None:
    manifest = tmp_path / "pilot-manifest.json"
    manifest.write_text('{"pilot":"qwen3_2k"}\n', encoding="utf-8")

    value = pilot.validate_pilot_manifest(manifest, sha256_file(manifest))

    assert value == {"pilot": "qwen3_2k"}
    with pytest.raises(ValueError, match="pilot manifest SHA-256 mismatch"):
        pilot.validate_pilot_manifest(manifest, "f" * 64)


def test_build_manifest_preserves_frozen_projection_contract(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    rows = _write_rows(input_path)
    pilot_manifest = tmp_path / "manifest.json"
    pilot_manifest.write_text('{"pilot":"qwen3_2k"}\n', encoding="utf-8")

    result = pilot.build_pilot_gradient_manifest(
        input_jsonl=input_path,
        input_sha256=sha256_file(input_path),
        rows=rows,
        pilot_manifest=pilot_manifest,
        pilot_manifest_sha256=sha256_file(pilot_manifest),
        reference_repo=tmp_path / "reference",
        reference_tree="a" * 40,
        prefix="pilot",
        num_shards=4,
        trl_version="0.17.0",
        package_versions={"torch": "x"},
    )

    assert result["input_fields"] == list(FIELDS)
    assert result["input_row_count"] == len(rows)
    assert result["model_name"] == MODEL_NAME
    assert result["model_revision"] == MODEL_REVISION
    assert result["projection"]["dimension"] == PROJECTION_DIM
    assert result["projection"]["seed"] == PROJECTION_SEED
    assert result["resume_allowed"] is False


def test_assert_fresh_shard_rejects_any_prior_owned_chunk(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    pilot.assert_fresh_shard(tmp_path, "pilot", total=12, num_shards=4, shard_index=1)

    (tmp_path / "pilot.4.txt").write_text('{"id":"x"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="resume is forbidden.*pilot.4.txt"):
        pilot.assert_fresh_shard(
            tmp_path, "pilot", total=12, num_shards=4, shard_index=1
        )


def test_assert_fresh_shard_rejects_malformed_prefix_artifacts(tmp_path: Path) -> None:
    (tmp_path / "pilot.bad.safetensors").write_bytes(b"bad")
    with pytest.raises(ValueError, match="malformed gradient artifact"):
        pilot.assert_fresh_shard(
            tmp_path, "pilot", total=12, num_shards=4, shard_index=0
        )


def test_run_collection_calls_official_collector_once_from_absolute_shard_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "input.jsonl"
    rows = _write_rows(input_path, count=6)
    pilot_manifest = tmp_path / "pilot-manifest.json"
    pilot_manifest.write_text('{"pilot":"qwen3_2k"}\n', encoding="utf-8")
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "prismatic-synthesis/gradient_modules").mkdir(parents=True)
    (reference / "prismatic-synthesis/gradient_modules/gradient_computer.py").write_text(
        "# fixture\n", encoding="utf-8"
    )
    output = tmp_path / "output"
    calls: list[tuple[list[dict], str, Path, int]] = []

    class FakeCollector:
        def compute_project_store_gradients(self, samples, prefix, output_dir, start):
            calls.append((list(samples), prefix, Path(output_dir), start))

    monkeypatch.setattr(pilot, "verify_reference_repo", lambda *_: None)
    monkeypatch.setattr(pilot, "_reference_tree", lambda *_: "b" * 40)
    monkeypatch.setattr(pilot, "_installed_trl_version", lambda: "0.17.0")
    monkeypatch.setattr(pilot, "_installed_package_versions", lambda: {"torch": "x"})
    monkeypatch.setattr(pilot, "validate_single_cuda_device", lambda *_: None)
    monkeypatch.setattr(
        pilot, "load_official_gradient_computer_class", lambda *_: object
    )
    monkeypatch.setattr(
        pilot,
        "load_model_and_tokenizer",
        lambda *_: (SimpleNamespace(config=SimpleNamespace(max_position_embeddings=32768)), object()),
    )
    monkeypatch.setattr(pilot, "construct_strict_collector", lambda *_args, **_kw: FakeCollector())
    monkeypatch.setattr(pilot, "preflight_samples", lambda samples, *_: {"sample_count": len(samples)})
    monkeypatch.setattr(
        pilot,
        "validate_global_gradient_coverage",
        lambda *_args, **_kwargs: {"row_count": 6, "chunk_count": 3, "shard_count": 3},
    )

    result = pilot.run_collection(
        reference_repo=reference,
        input_jsonl=input_path,
        input_sha256=sha256_file(input_path),
        pilot_manifest=pilot_manifest,
        pilot_manifest_sha256=sha256_file(pilot_manifest),
        output_dir=output,
        prefix="pilot",
        model_name=MODEL_NAME,
        model_revision=MODEL_REVISION,
        shard_index=1,
        num_shards=3,
        device="cuda:0",
        validate_after_collection=False,
    )

    assert result == {
        "status": "collected",
        "shard_start": 3,
        "shard_end": 6,
        "sample_count": 3,
    }
    assert calls == [(rows[3:6], "pilot", output, 3)]
    assert (output / "gradient.manifest.json").is_file()


def test_cli_exposes_no_resume_switch_and_requires_all_provenance() -> None:
    parser = pilot._build_parser()
    destinations = {action.dest for action in parser._actions}

    assert "resume" not in destinations
    assert {
        "input_jsonl",
        "input_sha256",
        "pilot_manifest",
        "pilot_manifest_sha256",
    } <= destinations
