import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from safetensors.torch import save_file

from math_eval.select_gradient_diverse_deepmath import (
    PROJECTION_DIM,
    cluster_official,
    compute_selection_diagnostics,
    load_official_reference_classes,
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


def test_load_validated_gradient_manifest_binds_eligibility_and_projection(
    tmp_path: Path,
) -> None:
    prepared_jsonl = tmp_path / "pool.jsonl"
    prepared_jsonl.write_text("{}\n", encoding="utf-8")
    source = tmp_path / "source.parquet"
    source.write_bytes(b"source")
    prepared_manifest_path = tmp_path / "pool.manifest.json"
    prepared_manifest = {
        "prepared_jsonl": str(prepared_jsonl.resolve()),
        "prepared_jsonl_sha256": __import__("hashlib").sha256(b"{}\n").hexdigest(),
        "prepared_row_count": 4,
        "source_sha256": __import__("hashlib").sha256(b"source").hexdigest(),
        "dataset_name": "zwhe99/DeepMath-103K",
        "dataset_revision": "5cf055d1fe3d7a2eb19719ac020211469736ae44",
    }
    prepared_manifest_path.write_text(
        json.dumps(prepared_manifest) + "\n", encoding="utf-8"
    )
    eligibility_path = tmp_path / "pool.eligibility.json"
    eligibility = {
        "eligible_row_count": 3,
        "excluded_row_count": 1,
        "eligible_ids_sha256": "e" * 64,
    }
    eligibility_path.write_text(json.dumps(eligibility) + "\n", encoding="utf-8")
    reference_repo = Path("/home/mchen/prismatic-synthesis-reference").resolve()
    gradient_manifest_path = tmp_path / "gradient.manifest.json"
    gradient_manifest = {
        "manifest_version": 1,
        "prepared_jsonl": str(prepared_jsonl.resolve()),
        "prepared_jsonl_sha256": prepared_manifest["prepared_jsonl_sha256"],
        "prepared_manifest": str(prepared_manifest_path.resolve()),
        "prepared_manifest_sha256": __import__("hashlib").sha256(
            prepared_manifest_path.read_bytes()
        ).hexdigest(),
        "prepared_row_count": 4,
        "source_sha256": prepared_manifest["source_sha256"],
        "dataset_name": prepared_manifest["dataset_name"],
        "dataset_revision": prepared_manifest["dataset_revision"],
        "reference_repo": str(reference_repo),
        "reference_commit": "d9484cd3b5991030b901ac4a3a9e2472dbfac2ad",
        "reference_tree": "a0079d8c5e15cb18bb4790f99c43cc19bb9ecd50",
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        "eligibility_report": str(eligibility_path.resolve()),
        "eligibility_report_sha256": __import__("hashlib").sha256(
            eligibility_path.read_bytes()
        ).hexdigest(),
        "eligible_row_count": 3,
        "excluded_row_count": 1,
        "eligible_ids_sha256": "e" * 64,
        "prefix": "deepmath",
        "num_shards": 4,
        "projection": {
            "backend": "CudaProjector",
            "dimension": 1024,
            "seed": 0,
            "type": "rademacher",
            "input_dtype": "float16",
            "output_dtype": "float32",
            "project_interval": 4,
            "save_interval": 500,
            "completion_only_loss": True,
            "full_parameter_gradients": True,
        },
    }
    gradient_manifest_path.write_text(
        json.dumps(gradient_manifest) + "\n", encoding="utf-8"
    )

    loaded = load_validated_gradient_manifest(
        gradient_manifest_path,
        prepared_manifest_path=prepared_manifest_path,
        prepared_manifest=prepared_manifest,
        eligibility_report_path=eligibility_path,
        eligibility_report=eligibility,
        reference_repo=reference_repo,
    )
    assert loaded == gradient_manifest

    eligibility["eligible_ids_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="eligible_ids_sha256"):
        load_validated_gradient_manifest(
            gradient_manifest_path,
            prepared_manifest_path=prepared_manifest_path,
            prepared_manifest=prepared_manifest,
            eligibility_report_path=eligibility_path,
            eligibility_report=eligibility,
            reference_repo=reference_repo,
        )


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
