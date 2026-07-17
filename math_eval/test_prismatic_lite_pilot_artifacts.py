from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from math_eval.prismatic_lite_pilot_artifacts import (
    PilotPaths,
    artifact_record,
    atomic_publish_json,
    atomic_publish_jsonl,
    atomic_publish_npz,
    strict_json_file,
    strict_jsonl,
    validate_exact_fields,
    write_or_validate_manifest,
)


def test_pilot_paths_expose_approved_namespace(tmp_path: Path) -> None:
    paths = PilotPaths.from_root(tmp_path)

    assert paths.manifest == tmp_path / "manifest.json"
    assert paths.calibration_sample == tmp_path / "calibration/sample.jsonl"
    assert paths.calibration_solutions == tmp_path / "calibration/solutions.jsonl"
    assert paths.calibration_gradient_input == tmp_path / "calibration/gradient_input.jsonl"
    assert paths.calibration_gradients == tmp_path / "calibration/paired_gradients"
    assert paths.calibration_report == tmp_path / "calibration/report.json"
    assert paths.candidate_problems == tmp_path / "candidates/problems.jsonl"
    assert paths.candidate_solutions == tmp_path / "candidates/solutions.jsonl"
    assert paths.quality_passed == tmp_path / "candidates/quality_passed.jsonl"
    assert paths.candidate_gradient_input == tmp_path / "candidates/gradient_input.jsonl"
    assert paths.candidate_gradients == tmp_path / "candidates/new_gradients"
    assert paths.cluster_seed42 == tmp_path / "selection/cluster_state_seed42.npz"
    assert paths.cluster_seed43 == tmp_path / "selection/cluster_state_seed43.npz"
    assert paths.accepted == tmp_path / "selection/accepted.jsonl"
    assert paths.rejected == tmp_path / "selection/rejected.jsonl"
    assert paths.report_json == tmp_path / "report.json"
    assert paths.report_md == tmp_path / "report.md"
    assert paths.stage_complete == tmp_path / "STAGE_COMPLETE.json"


def test_strict_json_file_rejects_duplicate_keys_and_nonfinite(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate.*a"):
        strict_json_file(duplicate, "fixture")

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        strict_json_file(nonfinite, "fixture")


def test_strict_jsonl_requires_exact_fields_unique_ids_and_no_blank_lines(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(
        json.dumps({"id": "a", "value": 1}) + "\n\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="blank line 2"):
        strict_jsonl(path, exact_fields=("id", "value"))

    path.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in ({"id": "a", "value": 1}, {"id": "a", "value": 2})
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate id.*a"):
        strict_jsonl(path, exact_fields=("id", "value"))

    path.write_text(json.dumps({"id": "a", "extra": 1}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exact fields"):
        strict_jsonl(path, exact_fields=("id", "value"))


def test_validate_exact_fields_names_missing_and_extra_fields() -> None:
    with pytest.raises(ValueError, match=r"missing=\['value'\].*extra=\['other'\]"):
        validate_exact_fields({"id": "x", "other": 1}, ("id", "value"), "row")


def test_atomic_publishers_refuse_existing_targets(tmp_path: Path) -> None:
    json_path = tmp_path / "nested/value.json"
    jsonl_path = tmp_path / "nested/rows.jsonl"
    npz_path = tmp_path / "nested/state.npz"

    atomic_publish_json(json_path, {"status": "ok"})
    atomic_publish_jsonl(jsonl_path, [{"id": "x"}])
    atomic_publish_npz(
        npz_path,
        {"labels": np.array([1, 0], dtype=np.int64), "vectors": np.eye(2)},
    )

    assert strict_json_file(json_path, "value") == {"status": "ok"}
    assert strict_jsonl(jsonl_path) == [{"id": "x"}]
    with np.load(npz_path, allow_pickle=False) as values:
        assert values.files == ["labels", "vectors"]
        np.testing.assert_array_equal(values["labels"], np.array([1, 0]))

    for publish, path, value in (
        (atomic_publish_json, json_path, {"status": "changed"}),
        (atomic_publish_jsonl, jsonl_path, [{"id": "y"}]),
        (atomic_publish_npz, npz_path, {"labels": np.array([2])}),
    ):
        with pytest.raises(FileExistsError, match=str(path)):
            publish(path, value)


def test_atomic_publish_npz_rejects_object_and_nonfinite_arrays(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="object dtype"):
        atomic_publish_npz(
            tmp_path / "object.npz", {"bad": np.array([object()], dtype=object)}
        )
    with pytest.raises(ValueError, match="non-finite"):
        atomic_publish_npz(
            tmp_path / "nan.npz", {"bad": np.array([float("nan")])}
        )


def test_manifest_creation_refuses_to_bless_partial_owned_artifact(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    partial = tmp_path / "calibration/sample.jsonl"
    partial.parent.mkdir(parents=True)
    partial.write_text('{"id":"partial"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="manifest is missing.*sample.jsonl"):
        write_or_validate_manifest(
            manifest, {"version": 1}, owned_paths=(partial,)
        )
    assert not manifest.exists()


def test_manifest_is_idempotent_only_for_exact_canonical_value(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    expected = {"version": 1, "sha256": "a" * 64}

    assert write_or_validate_manifest(path, expected) == expected
    assert write_or_validate_manifest(path, expected) == expected
    with pytest.raises(ValueError, match="manifest mismatch"):
        write_or_validate_manifest(path, {"version": 2, "sha256": "a" * 64})


def test_artifact_record_binds_absolute_path_size_and_hash(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"abc")

    assert artifact_record(path) == {
        "path": str(path.resolve()),
        "size": 3,
        "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    }
