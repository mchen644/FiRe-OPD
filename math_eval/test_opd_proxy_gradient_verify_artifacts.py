from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file as load_safetensors
from safetensors.numpy import save_file as save_safetensors

from math_eval.opd_proxy_gradient_verify_artifacts import (
    SourceFileSpec,
    TrajectoryKey,
    atomic_save_npy,
    atomic_save_safetensors,
    atomic_write_json,
    atomic_write_jsonl,
    build_runtime_metadata,
    build_source_snapshot,
    canonical_json_bytes,
    load_npy_strict,
    load_vector_set,
    recursive_file_manifest,
    sha256_file,
    sha256_id_lines,
    sha256_int_rows,
    sha256_ordered_id_lines,
    validate_exact_key_coverage,
    write_or_validate_manifest,
)


def test_canonical_json_bytes_are_ascii_sorted_compact_and_newline_terminated():
    payload = canonical_json_bytes({"β": 2, "a": 1})
    assert payload == b'{"a":1,"\\u03b2":2}\n'
    with pytest.raises(ValueError, match="JSON"):
        canonical_json_bytes({"bad": float("nan")})


def test_sha256_id_lines_uses_utf8_terminal_newlines():
    assert sha256_id_lines(["a", "β"]) == (
        "d3c5672deb0c99f78c72cf77d08b03ca61f7685b7fb1689d9861bcfd48afe094"
    )


def test_ordered_id_hash_allows_repeated_stable_ids_without_losing_order():
    expected = hashlib.sha256(b"q0\nq0\nq1\n").hexdigest()
    assert sha256_ordered_id_lines(["q0", "q0", "q1"]) == expected
    assert sha256_ordered_id_lines(["q0", "q1", "q0"]) != expected


def test_sha256_int_rows_uses_compact_json_and_terminal_newlines():
    rows = np.array([[0, 2], [1, 3]], dtype=np.int32)
    assert sha256_int_rows(rows) == (
        "74201e550190c3ead9a6c11a336e2d25ed25cb5d36b30166eba198b0596104c2"
    )
    with pytest.raises(ValueError, match="two-dimensional integer"):
        sha256_int_rows(np.array([0, 1], dtype=np.int32))


def test_source_snapshot_hashes_untracked_file_bytes(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "untracked.py"
    source.write_bytes(b"VALUE = 1\n")
    first = build_source_snapshot(
        {"main": repo}, [SourceFileSpec("main", "untracked.py")]
    )
    source.write_bytes(b"VALUE = 2\n")
    second = build_source_snapshot(
        {"main": repo}, [SourceFileSpec("main", "untracked.py")]
    )
    assert first["files"][0]["sha256"] != second["files"][0]["sha256"]
    assert first["manifest_sha256"] != second["manifest_sha256"]


def test_source_snapshot_is_sorted_and_rejects_escape(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "b.py").write_text("b\n", encoding="utf-8")
    (repo / "a.py").write_text("a\n", encoding="utf-8")
    snapshot = build_source_snapshot(
        {"main": repo},
        [SourceFileSpec("main", "b.py"), SourceFileSpec("main", "a.py")],
    )
    assert [(row["repository"], row["path"]) for row in snapshot["files"]] == [
        ("main", "a.py"),
        ("main", "b.py"),
    ]
    with pytest.raises(ValueError, match="relative|escape"):
        build_source_snapshot({"main": repo}, [SourceFileSpec("main", "../outside.py")])


def test_recursive_model_manifest_is_relative_sorted_and_byte_sensitive(tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "b.bin").write_bytes(b"b")
    (model / "a.json").write_bytes(b"a")
    first = recursive_file_manifest(model)
    assert [row["path"] for row in first["files"]] == ["a.json", "b.bin"]
    (model / "b.bin").write_bytes(b"changed")
    assert recursive_file_manifest(model)["manifest_sha256"] != first["manifest_sha256"]


def test_recursive_model_manifest_rejects_broken_and_root_escaping_symlinks(
    tmp_path: Path,
):
    model = tmp_path / "model"
    model.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    (model / "escape.bin").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes root"):
        recursive_file_manifest(model)
    (model / "escape.bin").unlink()
    (model / "broken.bin").symlink_to(model / "missing.bin")
    with pytest.raises(ValueError, match="broken symlink"):
        recursive_file_manifest(model)


def test_key_coverage_rejects_duplicate_and_missing_slots():
    expected = {
        TrajectoryKey("q0", 42, 0),
        TrajectoryKey("q0", 42, 1),
    }
    with pytest.raises(ValueError, match="duplicate trajectory key"):
        validate_exact_key_coverage(
            [TrajectoryKey("q0", 42, 0), TrajectoryKey("q0", 42, 0)],
            expected,
        )
    with pytest.raises(ValueError, match="missing trajectory keys"):
        validate_exact_key_coverage([TrajectoryKey("q0", 42, 0)], expected)
    with pytest.raises(ValueError, match="unexpected trajectory keys"):
        validate_exact_key_coverage(
            [
                TrajectoryKey("q0", 42, 0),
                TrajectoryKey("q0", 42, 1),
                TrajectoryKey("q0", 42, 2),
            ],
            expected,
        )


def test_atomic_json_jsonl_npy_and_safetensors_leave_no_temporary_files(
    tmp_path: Path,
):
    atomic_write_json(tmp_path / "value.json", {"b": 2, "a": 1})
    atomic_write_jsonl(tmp_path / "rows.jsonl", [{"id": "a"}, {"id": "β"}])
    atomic_save_npy(
        tmp_path / "rows.npy",
        np.asfortranarray(np.array([[1, 2], [3, 4]], dtype=">i4")),
    )
    atomic_save_safetensors(
        tmp_path / "vectors.safetensors",
        {"vector": np.array([[1.0, 2.0]], dtype=np.float32)},
    )
    assert (tmp_path / "value.json").read_bytes() == b'{"a":1,"b":2}\n'
    assert (tmp_path / "rows.jsonl").read_bytes() == (b'{"id":"a"}\n{"id":"\\u03b2"}\n')
    loaded = load_npy_strict(tmp_path / "rows.npy", dtype=np.dtype("<i4"), shape=(2, 2))
    assert loaded.dtype == np.dtype("<i4")
    assert loaded.flags.c_contiguous
    assert loaded.tolist() == [[1, 2], [3, 4]]
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".")]


def test_load_npy_strict_rejects_wrong_dtype_shape_and_non_contiguous_payload(
    tmp_path: Path,
):
    path = tmp_path / "array.npy"
    with path.open("wb") as handle:
        np.save(handle, np.array([[1.0, 2.0]], dtype=np.float64), allow_pickle=False)
    with pytest.raises(ValueError, match="dtype"):
        load_npy_strict(path, dtype=np.dtype("<i4"), shape=(1, 2))
    with pytest.raises(ValueError, match="shape"):
        load_npy_strict(path, dtype=np.dtype("<f8"), shape=(2, 1))


def test_write_or_validate_manifest_rejects_changed_parent_hash(tmp_path: Path):
    path = tmp_path / "manifest.json"
    original = {
        "schema_version": 1,
        "artifact_type": "fixture",
        "parent_hashes": {"sample": "a" * 64},
    }
    assert write_or_validate_manifest(path, original) == original
    assert write_or_validate_manifest(path, dict(original)) == original
    changed = {**original, "parent_hashes": {"sample": "b" * 64}}
    with pytest.raises(ValueError, match="manifest mismatch"):
        write_or_validate_manifest(path, changed)


def _version_resolver(versions: dict[str, str | None]):
    def resolve(name: str) -> str:
        value = versions.get(name)
        if value is None:
            raise LookupError(name)
        return value

    return resolve


def test_runtime_profiles_require_only_their_packages_and_keep_common_schema():
    capture_versions = {
        "torch": "2.6.0",
        "transformers": "4.51.1",
        "vllm": "0.8.5.post1",
        "numpy": "1.26.4",
        "safetensors": "0.8.0",
        "pyarrow": "24.0.0",
    }
    metadata = build_runtime_metadata(
        "verl_capture",
        version_resolver=_version_resolver(capture_versions),
        python_executable="/env/bin/python",
        python_version="3.10.0",
    )
    assert metadata["runtime_profile"] == "verl_capture"
    assert metadata["python_executable"] == "/env/bin/python"
    assert metadata["package_versions"]["scipy"] is None
    assert metadata["package_versions"]["fast-jl"] is None
    assert set(metadata["package_versions"]) == {
        "torch",
        "transformers",
        "vllm",
        "numpy",
        "scipy",
        "scikit-learn",
        "safetensors",
        "pyarrow",
        "traker",
        "fast-jl",
    }

    with pytest.raises(ValueError, match="required package torch"):
        build_runtime_metadata(
            "verl_capture",
            version_resolver=_version_resolver({**capture_versions, "torch": None}),
        )


def test_runtime_metadata_changes_when_a_package_version_changes():
    names = {
        "torch": "1",
        "transformers": "1",
        "vllm": "1",
        "numpy": "1",
        "scipy": "1",
        "scikit-learn": "1",
        "safetensors": "1",
        "pyarrow": "1",
        "traker": "1",
        "fast-jl": "1",
    }
    first = build_runtime_metadata(
        "gvendi_analysis", version_resolver=_version_resolver(names)
    )
    second = build_runtime_metadata(
        "gvendi_analysis",
        version_resolver=_version_resolver({**names, "numpy": "2"}),
    )
    assert first != second


def _write_vector_fixture(
    directory: Path,
    *,
    extra_manifest: dict[str, object] | None = None,
    include_complete: bool = True,
    add_unknown_tensor: bool = False,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    vector_ids = ["V:q1", "V:q0"]
    stable_ids = ["q1", "q0"]
    tensors = {
        "projected_gradient": np.array([[0.0, 2.0], [1.0, 0.0]], dtype=np.float32),
        "full_gradient_norm": np.array([2.0, 1.0], dtype=np.float32),
        "projected_gradient_norm": np.array([2.0, 1.0], dtype=np.float32),
        "valid_token_count": np.array([4.0, 3.0], dtype=np.float32),
        "response_length": np.array([5.0, 4.0], dtype=np.float32),
        "sampled_reverse_kl": np.array([0.2, 0.1], dtype=np.float32),
        "opd_signal_rms": np.array([0.4, 0.3], dtype=np.float32),
    }
    if add_unknown_tensor:
        tensors["surprise"] = np.ones(2, dtype=np.float32)
    save_safetensors(tensors, directory / "vectors_0_2.safetensors")
    sidecars = [
        {"vector_id": vector_id, "stable_id": stable_id, "tensor_row": index}
        for index, (vector_id, stable_id) in enumerate(zip(vector_ids, stable_ids))
    ]
    (directory / "vectors_0_2.jsonl").write_bytes(
        b"".join(canonical_json_bytes(row) for row in sidecars)
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "vector_set",
        "representation": "V",
        "vector_count": 2,
        "vector_dimension": 2,
        "chunk_count": 1,
        "vector_ids_sha256": sha256_id_lines(vector_ids),
        "stable_ids_sha256": sha256_id_lines(stable_ids),
        "parent_hashes": {"sample": "a" * 64},
        "source_snapshot": {"manifest_sha256": "b" * 64},
        "repository": {
            "head": "c" * 40,
            "status": "",
            "status_sha256": hashlib.sha256(b"").hexdigest(),
        },
        "runtime": {"runtime_profile": "gvendi_analysis"},
        "verifier": {"status": "not_computed"},
        "metadata": {},
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    (directory / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    if include_complete:
        complete = {
            "schema_version": 1,
            "artifact_type": "vector_set_complete",
            "manifest_sha256": sha256_file(directory / "manifest.json"),
            "vector_count": 2,
            "chunk_count": 1,
        }
        (directory / "COMPLETE.json").write_bytes(canonical_json_bytes(complete))


def test_vector_loader_reorders_sidecar_rows_to_requested_stable_id_order(
    tmp_path: Path,
):
    _write_vector_fixture(tmp_path)
    # An interrupted hidden sibling must never become a discovered chunk.
    (tmp_path / ".vectors_2_4.safetensors.dead.tmp").write_bytes(b"partial")
    vectors = load_vector_set(tmp_path, expected_stable_ids=("q0", "q1"))
    assert vectors.stable_ids == ("q0", "q1")
    assert vectors.vector_ids == ("V:q0", "V:q1")
    np.testing.assert_array_equal(
        vectors.vectors, np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    )
    np.testing.assert_array_equal(vectors.full_gradient_norm, [1.0, 2.0])
    assert vectors.verifier_correct_count is None
    assert vectors.verifier_total is None


def test_vector_loader_requires_complete_marker_and_exact_parent_hash(tmp_path: Path):
    incomplete = tmp_path / "incomplete"
    _write_vector_fixture(incomplete, include_complete=False)
    with pytest.raises(ValueError, match="COMPLETE.json"):
        load_vector_set(incomplete)

    complete = tmp_path / "complete"
    _write_vector_fixture(complete)
    with pytest.raises(ValueError, match="parent hash"):
        load_vector_set(complete, expected_parent_hashes={"sample": "d" * 64})


def test_vector_loader_rejects_unknown_manifest_and_tensor_fields(tmp_path: Path):
    unknown_manifest = tmp_path / "unknown_manifest"
    _write_vector_fixture(unknown_manifest, extra_manifest={"surprise": 1})
    with pytest.raises(ValueError, match="unknown vector manifest fields"):
        load_vector_set(unknown_manifest)

    unknown_tensor = tmp_path / "unknown_tensor"
    _write_vector_fixture(unknown_tensor, add_unknown_tensor=True)
    with pytest.raises(ValueError, match="unknown vector tensor fields"):
        load_vector_set(unknown_tensor)


def test_vector_loader_binds_verifier_status_to_optional_count_tensors(
    tmp_path: Path,
):
    computed_without_counts = tmp_path / "computed_without_counts"
    _write_vector_fixture(computed_without_counts)
    manifest_path = computed_without_counts / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["verifier"] = {"status": "computed"}
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    complete_path = computed_without_counts / "COMPLETE.json"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    complete["manifest_sha256"] = sha256_file(manifest_path)
    complete_path.write_bytes(canonical_json_bytes(complete))
    with pytest.raises(ValueError, match="requires verifier count"):
        load_vector_set(computed_without_counts)

    counts_without_status = tmp_path / "counts_without_status"
    _write_vector_fixture(counts_without_status)
    tensor_path = counts_without_status / "vectors_0_2.safetensors"
    tensors = dict(load_safetensors(tensor_path))
    tensors["verifier_correct_count"] = np.array([1.0, 0.0], dtype=np.float32)
    tensors["verifier_total"] = np.array([1.0, 1.0], dtype=np.float32)
    save_safetensors(tensors, tensor_path)
    with pytest.raises(ValueError, match="not_computed.*forbids"):
        load_vector_set(counts_without_status)


def test_vector_loader_rejects_nonfinite_and_zero_vectors(tmp_path: Path):
    _write_vector_fixture(tmp_path)
    path = tmp_path / "vectors_0_2.safetensors"
    tensors = {
        "projected_gradient": np.array([[0.0, 0.0], [np.nan, 1.0]], dtype=np.float32),
        "full_gradient_norm": np.array([1.0, 1.0], dtype=np.float32),
        "projected_gradient_norm": np.array([1.0, 1.0], dtype=np.float32),
        "valid_token_count": np.array([1.0, 1.0], dtype=np.float32),
        "response_length": np.array([1.0, 1.0], dtype=np.float32),
        "sampled_reverse_kl": np.array([1.0, 1.0], dtype=np.float32),
        "opd_signal_rms": np.array([1.0, 1.0], dtype=np.float32),
    }
    save_safetensors(tensors, path)
    with pytest.raises(ValueError, match="finite|zero"):
        load_vector_set(tmp_path)
