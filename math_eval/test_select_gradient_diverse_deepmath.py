import hashlib
import json
import multiprocessing
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from safetensors.torch import save_file

from math_eval import select_gradient_diverse_deepmath as selection
from math_eval.select_gradient_diverse_deepmath import (
    PROJECTION_DIM,
    apply_production_eligibility_report,
    build_selection_gradient_provenance,
    cluster_official,
    compute_selection_diagnostics,
    load_official_reference_classes,
    load_globally_validated_gradients,
    load_projected_gradients,
    load_validated_gradient_manifest,
    main,
    publish_selection_artifacts,
    write_selected_parquet,
)


def _ids(*indices: int) -> list[str]:
    return [f"deepmath-level6-{index:06d}" for index in indices]


def _write_chunk(
    directory: Path,
    *,
    prefix: str,
    start: int,
    sidecar_ids: list[str],
    tensor_ids: list[str] | None = None,
    dtype: torch.dtype = torch.float32,
    width: int = PROJECTION_DIM,
    value: float = 1.0,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    stored_ids = sidecar_ids if tensor_ids is None else tensor_ids
    # Reverse insertion order deliberately: loader order must come from eligibility.
    tensors = {
        sample_id: torch.full((width,), value + offset, dtype=dtype)
        for offset, sample_id in enumerate(reversed(stored_ids))
    }
    save_file(tensors, directory / f"{prefix}.{start}.safetensors")
    (directory / f"{prefix}.{start}.txt").write_text(
        "".join(json.dumps({"id": sample_id}) + "\n" for sample_id in sidecar_ids),
        encoding="utf-8",
    )


def _source_table() -> pa.Table:
    # Parquet reads list children back with the canonical ``element`` name; this
    # matches the actual training parquet schema instead of pa.list_'s in-memory
    # default child name (``item``).
    prompt_type = pa.list_(
        pa.field(
            "element",
            pa.struct([("role", pa.string()), ("content", pa.string())]),
        )
    )
    schema = pa.schema(
        [
            ("prompt", prompt_type),
            ("reward_model", pa.struct([("ground_truth", pa.string())])),
            (
                "extra_info",
                pa.struct([("index", pa.int64()), ("split", pa.string())]),
            ),
        ],
        metadata={b"dataset": b"deepmath-level6", b"nested": b"preserve-me"},
    )
    rows = [
        {
            "prompt": [{"role": "user", "content": f"question-{index}"}],
            "reward_model": {"ground_truth": str(index)},
            "extra_info": {"index": index, "split": "train"},
        }
        for index in range(4)
    ]
    return pa.Table.from_pylist(rows, schema=schema)


def _hold_selection_locks(paths, attempting, acquired, release) -> None:
    attempting.set()
    with selection._selection_locks(paths):
        acquired.set()
        release.wait(timeout=5)


def test_load_projected_gradients_uses_eligibility_order_with_stable_id_gap(
    tmp_path: Path,
) -> None:
    expected = _ids(0, 2, 3)
    _write_chunk(
        tmp_path,
        prefix="deepmath",
        start=2,
        sidecar_ids=expected[2:],
        value=20.0,
    )
    _write_chunk(
        tmp_path,
        prefix="deepmath",
        start=0,
        sidecar_ids=expected[:2],
        value=10.0,
    )

    loaded_ids, gradients = load_projected_gradients(tmp_path, expected)

    assert loaded_ids == expected
    assert gradients.shape == (3, PROJECTION_DIM)
    assert gradients.dtype == torch.float32
    assert gradients.device.type == "cpu"
    assert gradients[:, 0].tolist() == [11.0, 10.0, 20.0]


@pytest.mark.parametrize("problem", ["missing", "unexpected", "duplicate"])
def test_load_projected_gradients_rejects_non_exact_coverage(
    tmp_path: Path, problem: str
) -> None:
    expected = _ids(0, 2, 3)
    if problem == "missing":
        _write_chunk(tmp_path, prefix="deepmath", start=0, sidecar_ids=expected[:2])
    elif problem == "unexpected":
        _write_chunk(
            tmp_path,
            prefix="deepmath",
            start=0,
            sidecar_ids=[expected[0], _ids(1)[0], expected[2]],
        )
    else:
        _write_chunk(
            tmp_path,
            prefix="deepmath",
            start=0,
            sidecar_ids=[expected[0], expected[0], expected[2]],
        )

    with pytest.raises(ValueError):
        load_projected_gradients(tmp_path, expected)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dtype": torch.float16}, "float32"),
        ({"width": PROJECTION_DIM - 1}, "shape"),
        ({"value": 0.0}, "zero"),
        ({"value": float("nan")}, "finite"),
    ],
)
def test_load_projected_gradients_rejects_invalid_tensor_rows(
    tmp_path: Path, kwargs: dict, message: str
) -> None:
    expected = _ids(0)
    _write_chunk(
        tmp_path,
        prefix="deepmath",
        start=0,
        sidecar_ids=expected,
        **kwargs,
    )

    with pytest.raises(ValueError, match=message):
        load_projected_gradients(tmp_path, expected)


@pytest.mark.parametrize("orphan_suffix", ["txt", "safetensors"])
def test_load_projected_gradients_rejects_unpaired_chunks(
    tmp_path: Path, orphan_suffix: str
) -> None:
    expected = _ids(0)
    _write_chunk(tmp_path, prefix="deepmath", start=0, sidecar_ids=expected)
    (tmp_path / f"deepmath.0.{orphan_suffix}").unlink()

    with pytest.raises(ValueError, match="paired"):
        load_projected_gradients(tmp_path, expected)


def test_load_projected_gradients_rejects_unfinished_chunk_artifacts(
    tmp_path: Path,
) -> None:
    expected = _ids(0)
    _write_chunk(tmp_path, prefix="deepmath", start=0, sidecar_ids=expected)
    (tmp_path / "deepmath.1.safetensors.tmp").write_bytes(b"unfinished")

    with pytest.raises(ValueError, match="malformed gradient artifact"):
        load_projected_gradients(tmp_path, expected)


def test_load_projected_gradients_rejects_duplicate_sidecar_json_keys(
    tmp_path: Path,
) -> None:
    expected = _ids(0)
    _write_chunk(tmp_path, prefix="deepmath", start=0, sidecar_ids=expected)
    (tmp_path / "deepmath.0.txt").write_text(
        f'{{"id":"{expected[0]}","id":"{expected[0]}"}}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_projected_gradients(tmp_path, expected)


def test_load_projected_gradients_rejects_a_nonmanifest_prefix(
    tmp_path: Path,
) -> None:
    expected = _ids(0)
    _write_chunk(tmp_path, prefix="deepmath", start=0, sidecar_ids=expected)
    _write_chunk(tmp_path, prefix="foreign", start=1, sidecar_ids=_ids(1))

    with pytest.raises(ValueError, match="unexpected gradient prefix"):
        load_projected_gradients(tmp_path, expected, prefix="deepmath")


def test_load_globally_validated_gradients_uses_manifest_sharding_contract(
    tmp_path: Path,
) -> None:
    expected = _ids(0, 2)
    _write_chunk(tmp_path, prefix="deepmath", start=0, sidecar_ids=expected)
    calls = []

    def exact_validator(ids, directory, prefix, num_shards):
        calls.append((list(ids), Path(directory), prefix, num_shards))
        return {"row_count": 2, "chunk_count": 1, "shard_count": 4}

    loaded_ids, gradients, coverage = load_globally_validated_gradients(
        tmp_path,
        expected,
        {"prefix": "deepmath", "num_shards": 4},
        coverage_validator=exact_validator,
    )

    assert calls == [(expected, tmp_path, "deepmath", 4)]
    assert loaded_ids == expected
    assert gradients.shape == (2, PROJECTION_DIM)
    assert coverage == {"row_count": 2, "chunk_count": 1, "shard_count": 4}


def test_write_selected_parquet_preserves_nested_schema_metadata_and_order(
    tmp_path: Path,
) -> None:
    source = _source_table()
    output = tmp_path / "selected.parquet"

    write_selected_parquet(source, [3, 1], output)

    selected = pq.read_table(output)
    assert selected.schema.equals(source.schema, check_metadata=True)
    assert selected.to_pylist() == source.take(pa.array([3, 1])).to_pylist()


@pytest.mark.parametrize("indices", [[0, 0], [-1], [4]])
def test_write_selected_parquet_rejects_invalid_source_indices(
    tmp_path: Path, indices: list[int]
) -> None:
    with pytest.raises(ValueError):
        write_selected_parquet(_source_table(), indices, tmp_path / "selected.parquet")


def test_write_selected_parquet_rejects_duplicate_prompts(tmp_path: Path) -> None:
    source = _source_table()
    rows = source.to_pylist()
    rows[2]["prompt"] = rows[0]["prompt"]
    duplicate_source = pa.Table.from_pylist(rows, schema=source.schema)

    with pytest.raises(ValueError, match="unique prompts"):
        write_selected_parquet(
            duplicate_source, [0, 2], tmp_path / "selected.parquet"
        )


class _FakeClusterManager:
    calls: list[tuple[torch.Tensor, int, int, bool]] = []

    @staticmethod
    def cluster_kmeans(data, k, num_iter, use_tqdm=False):
        _FakeClusterManager.calls.append((data.clone(), k, num_iter, use_tqdm))
        labels = torch.arange(data.shape[0], device=data.device) % k
        return labels, torch.empty((k, data.shape[1]), device=data.device)


def test_cluster_official_normalizes_seed_permutates_and_inverts_labels() -> None:
    _FakeClusterManager.calls.clear()
    gradients = torch.arange(1, 61, dtype=torch.float32).reshape(10, 6)

    labels = cluster_official(
        gradients,
        ratio=0.2,
        iterations=20,
        seed=42,
        reference_repo=Path("unused-under-injection"),
        cluster_manager_class=_FakeClusterManager,
    )

    received, k, iterations, use_tqdm = _FakeClusterManager.calls[-1]
    expected_permutation = torch.randperm(10, generator=torch.Generator().manual_seed(42))
    normalized = torch.nn.functional.normalize(gradients, dim=1)
    assert torch.allclose(received, normalized[expected_permutation])
    assert torch.allclose(torch.linalg.vector_norm(received, dim=1), torch.ones(10))
    assert (k, iterations, use_tqdm) == (2, 20, False)
    expected_labels = torch.empty(10, dtype=torch.int64)
    expected_labels[expected_permutation] = torch.arange(10) % 2
    assert np.array_equal(labels, expected_labels.numpy())


@pytest.mark.parametrize("ratio", [0.0, -0.1, 1.1])
def test_cluster_official_rejects_invalid_ratio(ratio: float) -> None:
    with pytest.raises(ValueError, match="ratio"):
        cluster_official(
            torch.ones((3, 2)),
            ratio,
            20,
            42,
            Path("unused"),
            cluster_manager_class=_FakeClusterManager,
        )


def test_cluster_official_requires_twenty_iterations() -> None:
    with pytest.raises(ValueError, match="20"):
        cluster_official(
            torch.ones((3, 2)),
            0.1,
            19,
            42,
            Path("unused"),
            cluster_manager_class=_FakeClusterManager,
        )


def test_cluster_official_real_path_requires_explicit_cuda_device() -> None:
    with pytest.raises(ValueError, match="explicit.*cuda:0"):
        cluster_official(
            torch.ones((3, 2)),
            0.1,
            20,
            42,
            Path("/home/mchen/prismatic-synthesis-reference"),
        )


class _FakeVendi:
    calls: list[tuple[torch.Tensor, int]] = []

    @staticmethod
    def compute_vendi_score(covariance, n):
        _FakeVendi.calls.append((covariance.clone(), n))
        return 7.5


def test_compute_selection_diagnostics_reports_two_seeds_and_official_vendi() -> None:
    _FakeVendi.calls.clear()
    gradients = torch.tensor(
        [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0], [2.0, 1.0]],
        dtype=torch.float32,
    )
    rows = [
        {"topic": "A", "difficulty": 6},
        {"topic": "B", "difficulty": 7},
        {"topic": "A", "difficulty": 6},
        {"topic": "C", "difficulty": 8},
    ]
    label_runs = {
        (0.5, 42): np.array([0, 0, 1, 1]),
        (0.5, 43): np.array([0, 1, 0, 1]),
    }

    diagnostics = compute_selection_diagnostics(
        gradients,
        label_runs,
        rows,
        target_size=3,
        selection_seed=42,
        vendi_class=_FakeVendi,
    )

    ratio = diagnostics["ratios"]["0.5"]
    assert set(ratio["runs"]) == {"42", "43"}
    assert ratio["adjusted_rand_index"] == pytest.approx(-0.5)
    assert 0.0 <= ratio["selected_set_jaccard"] <= 1.0
    assert ratio["runs"]["42"]["selected_g_vendi"] == 7.5
    assert ratio["runs"]["42"]["selected_topic_coverage"] >= 2
    assert sum(ratio["runs"]["42"]["difficulty_distribution"].values()) == 3
    assert len(_FakeVendi.calls) == 2
    for covariance, n in _FakeVendi.calls:
        assert covariance.shape == (2, 2)
        assert n == 3


def test_load_official_reference_classes_uses_pinned_sources() -> None:
    manager, vendi, provenance = load_official_reference_classes(
        Path("/home/mchen/prismatic-synthesis-reference")
    )

    assert manager.__name__ == "ClusterManager"
    assert vendi.__name__ == "Vendi"
    assert provenance["reference_commit"] == (
        "d9484cd3b5991030b901ac4a3a9e2472dbfac2ad"
    )
    assert provenance["reference_tree"] == (
        "a0079d8c5e15cb18bb4790f99c43cc19bb9ecd50"
    )


def test_load_validated_gradient_manifest_delegates_the_complete_contract(
    tmp_path: Path,
) -> None:
    prepared_manifest_path = tmp_path / "pool.manifest.json"
    eligibility_path = tmp_path / "pool.eligibility.json"
    prepared_manifest = {"prepared": "provenance"}
    eligibility = {"eligible": "provenance"}
    reference_repo = Path("/home/mchen/prismatic-synthesis-reference").resolve()
    gradient_manifest_path = tmp_path / "gradient.manifest.json"
    gradient_manifest = {
        "manifest_version": 1,
        "prefix": "deepmath",
        "num_shards": 4,
        "official_gradient_module": "/pinned/gradient_computer.py",
        "max_context_tokens": 32768,
        "trl_version": "0.17.0",
        "package_versions": {"torch": "2.6.0", "trak": "0.3.2"},
        "projection": {"response_marker": "<|im_start|>assistant"},
    }
    gradient_manifest_path.write_text(
        json.dumps(gradient_manifest) + "\n", encoding="utf-8"
    )
    calls = []

    def exact_validator(path, **kwargs):
        calls.append((Path(path), kwargs))
        return dict(gradient_manifest)

    loaded = load_validated_gradient_manifest(
        gradient_manifest_path,
        prepared_manifest_path=prepared_manifest_path,
        prepared_manifest=prepared_manifest,
        eligibility_report_path=eligibility_path,
        eligibility_report=eligibility,
        reference_repo=reference_repo,
        manifest_validator=exact_validator,
    )
    assert loaded == gradient_manifest
    assert calls == [
        (
            gradient_manifest_path.resolve(),
            {
                "prepared_manifest_path": prepared_manifest_path.resolve(),
                "prepared_manifest": prepared_manifest,
                "eligibility_report_path": eligibility_path.resolve(),
                "eligibility_report": eligibility,
                "reference_repo": reference_repo,
                "expected_prefix": None,
                "expected_num_shards": None,
            },
        )
    ]


def test_apply_production_eligibility_binds_the_callers_manifest_path(
    tmp_path: Path,
) -> None:
    rows = [{"id": _ids(0)[0]}]
    manifest = {"prepared": "mapping"}
    manifest_path = tmp_path / "prepared.manifest.json"
    report_path = tmp_path / "eligibility.json"
    calls = []

    def apply_report(received_rows, received_manifest, received_report, **kwargs):
        calls.append(
            (received_rows, received_manifest, Path(received_report), kwargs)
        )
        return list(received_rows), {"validated": True}

    result = apply_production_eligibility_report(
        rows,
        manifest,
        report_path,
        prepared_manifest_path=manifest_path,
        apply_report=apply_report,
    )

    assert result == (rows, {"validated": True})
    assert calls == [
        (
            rows,
            manifest,
            report_path.resolve(),
            {"prepared_manifest_path": manifest_path.resolve()},
        )
    ]


def test_load_validated_gradient_manifest_rejects_duplicate_keys_before_delegate(
    tmp_path: Path,
) -> None:
    path = tmp_path / "gradient.manifest.json"
    path.write_text('{"prefix":"deepmath","prefix":"deepmath"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_validated_gradient_manifest(
            path,
            prepared_manifest_path=tmp_path / "pool.manifest.json",
            prepared_manifest={},
            eligibility_report_path=tmp_path / "eligibility.json",
            eligibility_report={},
            reference_repo=Path("/home/mchen/prismatic-synthesis-reference"),
            manifest_validator=lambda *_args, **_kwargs: pytest.fail(
                "duplicate JSON reached collector validator"
            ),
        )


def test_load_validated_gradient_manifest_reuses_collector_exact_validator(
    tmp_path: Path, monkeypatch
) -> None:
    from math_eval import collect_prismatic_gradients as collector

    prepared_jsonl = tmp_path / "pool.jsonl"
    prepared_manifest_path = tmp_path / "pool.manifest.json"
    eligibility_path = tmp_path / "eligibility.json"
    gradient_path = tmp_path / "gradients" / collector.GRADIENT_MANIFEST_NAME
    prepared_manifest = {
        "prepared_jsonl": str(prepared_jsonl.resolve()),
        "prepared_jsonl_sha256": "1" * 64,
        "prepared_row_count": 3,
        "source_sha256": "2" * 64,
    }
    prepared_manifest_path.write_text(
        json.dumps(prepared_manifest) + "\n", encoding="utf-8"
    )
    eligibility = {
        "eligible_row_count": 2,
        "excluded_row_count": 1,
        "eligible_ids_sha256": "3" * 64,
    }
    eligibility_path.write_text(json.dumps(eligibility) + "\n", encoding="utf-8")
    reference_repo = Path("/home/mchen/prismatic-synthesis-reference").resolve()
    package_versions = {
        "torch": "test-torch",
        "transformers": "test-transformers",
        "trl": "0.17.0",
        "trak": collector.TRAKER_VERSION,
        "fast_jl": collector.FAST_JL_VERSION,
    }
    monkeypatch.setattr(collector, "_installed_trl_version", lambda: "0.17.0")
    monkeypatch.setattr(
        collector, "_installed_package_versions", lambda: package_versions
    )
    expected = collector.build_gradient_manifest(
        prepared_jsonl,
        prepared_manifest_path,
        prepared_manifest,
        reference_repo,
        collector.REFERENCE_COMMIT,
        collector._reference_tree(reference_repo),
        prefix="deepmath",
        num_shards=1,
        trl_version="0.17.0",
        package_versions=package_versions,
        eligibility_report_path=eligibility_path,
        eligibility_report=eligibility,
    )
    collector.write_or_validate_gradient_manifest(gradient_path, expected)

    loaded = load_validated_gradient_manifest(
        gradient_path,
        prepared_manifest_path=prepared_manifest_path,
        prepared_manifest=prepared_manifest,
        eligibility_report_path=eligibility_path,
        eligibility_report=eligibility,
        reference_repo=reference_repo,
    )

    assert loaded == expected
    assert loaded["projection"]["response_marker"] == "<|im_start|>assistant"
    assert loaded["official_gradient_module"].endswith("gradient_computer.py")


def test_build_selection_gradient_provenance_embeds_the_complete_manifest(
    tmp_path: Path,
) -> None:
    path = tmp_path / "gradient.manifest.json"
    path.write_text("manifest-bytes\n", encoding="utf-8")
    gradient_manifest = {
        "prefix": "deepmath",
        "num_shards": 4,
        "official_gradient_module": "/reference/gradient_computer.py",
        "max_context_tokens": 32768,
        "trl_version": "0.17.0",
        "package_versions": {
            "torch": "2.6.0",
            "transformers": "4.51.3",
            "trak": "0.3.2",
            "fast_jl": "0.1.3",
        },
        "projection": {"response_marker": "<|im_start|>assistant"},
    }

    provenance = build_selection_gradient_provenance(path, gradient_manifest)

    assert provenance["gradient_provenance"] == gradient_manifest
    assert provenance["gradient_manifest"] == str(path.resolve())
    assert provenance["gradient_manifest_sha256"] == hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    assert provenance["gradient_prefix"] == "deepmath"
    assert provenance["gradient_num_shards"] == 4
    assert provenance["gradient_runtime"] == {
        "trl_version": "0.17.0",
        "package_versions": gradient_manifest["package_versions"],
    }


def test_main_requires_the_eligibility_report() -> None:
    with pytest.raises(SystemExit) as error:
        main([])
    assert error.value.code == 2


def test_publish_selection_artifacts_is_locked_reusable_and_mismatch_closed(
    tmp_path: Path,
) -> None:
    source = _source_table()
    output_parquet = tmp_path / "selected.parquet"
    selected_ids_path = tmp_path / "selected_ids.jsonl"
    diagnostics_path = tmp_path / "diagnostics.json"
    manifest_path = tmp_path / "manifest.json"
    selected_rows = [
        {"id": _ids(3)[0], "source_row_index": 3},
        {"id": _ids(1)[0], "source_row_index": 1},
    ]
    base_manifest = {"manifest_version": 1, "eligible_row_count": 3}
    diagnostics = {"primary_ratio": 0.1, "primary_seed": 42}

    first = publish_selection_artifacts(
        source=source,
        selected_rows=selected_rows,
        diagnostics=diagnostics,
        base_manifest=base_manifest,
        output_parquet=output_parquet,
        selected_ids_path=selected_ids_path,
        diagnostics_path=diagnostics_path,
        manifest_path=manifest_path,
    )
    second = publish_selection_artifacts(
        source=source,
        selected_rows=selected_rows,
        diagnostics=diagnostics,
        base_manifest=base_manifest,
        output_parquet=output_parquet,
        selected_ids_path=selected_ids_path,
        diagnostics_path=diagnostics_path,
        manifest_path=manifest_path,
    )

    assert second == first
    assert first["selected_row_count"] == 2
    assert pq.read_table(output_parquet).to_pylist() == source.take(
        pa.array([3, 1])
    ).to_pylist()
    for target in (
        output_parquet,
        selected_ids_path,
        diagnostics_path,
        manifest_path,
    ):
        assert target.with_name(f".{target.name}.lock").is_file()

    diagnostics_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        publish_selection_artifacts(
            source=source,
            selected_rows=selected_rows,
            diagnostics=diagnostics,
            base_manifest=base_manifest,
            output_parquet=output_parquet,
            selected_ids_path=selected_ids_path,
            diagnostics_path=diagnostics_path,
            manifest_path=manifest_path,
        )


def test_publish_selection_artifacts_rejects_partial_existing_set(
    tmp_path: Path,
) -> None:
    diagnostics_path = tmp_path / "diagnostics.json"
    diagnostics_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="partial"):
        publish_selection_artifacts(
            source=_source_table(),
            selected_rows=[{"id": _ids(0)[0], "source_row_index": 0}],
            diagnostics={},
            base_manifest={"manifest_version": 1},
            output_parquet=tmp_path / "selected.parquet",
            selected_ids_path=tmp_path / "selected_ids.jsonl",
            diagnostics_path=diagnostics_path,
            manifest_path=tmp_path / "manifest.json",
        )


def test_selection_locks_serialize_shared_outputs_with_different_manifests(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("fork")
    common = [
        tmp_path / "selected.parquet",
        tmp_path / "selected_ids.jsonl",
        tmp_path / "diagnostics.json",
    ]
    first_attempting, second_attempting = context.Event(), context.Event()
    first_acquired, first_release = context.Event(), context.Event()
    second_acquired, second_release = context.Event(), context.Event()
    first = context.Process(
        target=_hold_selection_locks,
        args=(
            common + [tmp_path / "manifest-a.json"],
            first_attempting,
            first_acquired,
            first_release,
        ),
    )
    second = context.Process(
        target=_hold_selection_locks,
        args=(
            common + [tmp_path / "manifest-b.json"],
            second_attempting,
            second_acquired,
            second_release,
        ),
    )
    try:
        first.start()
        assert first_acquired.wait(timeout=2)
        second.start()
        assert second_attempting.wait(timeout=2)
        assert not second_acquired.wait(timeout=0.25)
        first_release.set()
        assert second_acquired.wait(timeout=2)
        second_release.set()
        first.join(timeout=2)
        second.join(timeout=2)
        assert first.exitcode == 0
        assert second.exitcode == 0
    finally:
        first_release.set()
        second_release.set()
        for process in (first, second):
            if process.is_alive():
                process.terminate()
            process.join(timeout=2)


def test_publish_selection_artifacts_rejects_nonfinite_json(tmp_path: Path) -> None:
    targets = {
        "output_parquet": tmp_path / "selected.parquet",
        "selected_ids_path": tmp_path / "selected_ids.jsonl",
        "diagnostics_path": tmp_path / "diagnostics.json",
        "manifest_path": tmp_path / "manifest.json",
    }

    with pytest.raises(ValueError, match="JSON"):
        publish_selection_artifacts(
            source=_source_table(),
            selected_rows=[{"id": _ids(0)[0], "source_row_index": 0}],
            diagnostics={"nonfinite": float("nan")},
            base_manifest={"manifest_version": 1},
            **targets,
        )
    assert not any(path.exists() for path in targets.values())


@pytest.mark.parametrize("artifact", ["manifest", "diagnostics", "selected_ids"])
def test_publish_selection_artifacts_rejects_duplicate_json_keys_on_reuse(
    tmp_path: Path, artifact: str
) -> None:
    source = _source_table()
    output_parquet = tmp_path / "selected.parquet"
    selected_ids_path = tmp_path / "selected_ids.jsonl"
    diagnostics_path = tmp_path / "diagnostics.json"
    manifest_path = tmp_path / "manifest.json"
    selected_rows = [{"id": _ids(0)[0], "source_row_index": 0}]
    diagnostics = {"primary_ratio": 0.1}
    base_manifest = {"manifest_version": 1}
    publish_selection_artifacts(
        source=source,
        selected_rows=selected_rows,
        diagnostics=diagnostics,
        base_manifest=base_manifest,
        output_parquet=output_parquet,
        selected_ids_path=selected_ids_path,
        diagnostics_path=diagnostics_path,
        manifest_path=manifest_path,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if artifact == "manifest":
        original = manifest_path.read_text(encoding="utf-8")
        manifest_path.write_text(
            '{"manifest_version":1,' + original.lstrip()[1:], encoding="utf-8"
        )
    elif artifact == "diagnostics":
        diagnostics_path.write_text(
            '{"primary_ratio":0.1,"primary_ratio":0.1}\n', encoding="utf-8"
        )
        manifest["diagnostics_sha256"] = hashlib.sha256(
            diagnostics_path.read_bytes()
        ).hexdigest()
        manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    else:
        selected_ids_path.write_text(
            f'{{"id":"{_ids(0)[0]}","id":"{_ids(0)[0]}",'
            '"source_row_index":0}\n',
            encoding="utf-8",
        )
        manifest["selected_ids_sha256"] = hashlib.sha256(
            selected_ids_path.read_bytes()
        ).hexdigest()
        manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key"):
        publish_selection_artifacts(
            source=source,
            selected_rows=selected_rows,
            diagnostics=diagnostics,
            base_manifest=base_manifest,
            output_parquet=output_parquet,
            selected_ids_path=selected_ids_path,
            diagnostics_path=diagnostics_path,
            manifest_path=manifest_path,
        )
