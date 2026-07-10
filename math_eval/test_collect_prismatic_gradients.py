from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors.torch import save_file

from math_eval.collect_prismatic_gradients import (
    ASSISTANT_RESPONSE_MARKER,
    DATASET_NAME,
    DATASET_REVISION,
    MAX_CONTEXT_TOKENS,
    MODEL_NAME,
    MODEL_REVISION,
    PROJECT_INTERVAL,
    PROJECTION_DIM,
    PROJECTION_SEED,
    REFERENCE_COMMIT,
    SAVE_INTERVAL,
    TRL_VERSION,
    _exclusive_shard_lock,
    build_gradient_manifest,
    construct_strict_collector,
    load_prepared_pool,
    load_model_and_tokenizer,
    load_official_gradient_computer_class,
    preflight_samples,
    resolve_resume_start,
    run_collection,
    run_global_validation,
    validate_global_gradient_coverage,
    validate_single_cuda_device,
    verify_reference_repo,
    write_or_validate_gradient_manifest,
)
from math_eval.deepmath_gradient_diversity import official_shard_bounds


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "reference"
    repository.mkdir()
    _run_git(repository, "init", "--quiet")
    _run_git(repository, "config", "user.email", "test@example.com")
    _run_git(repository, "config", "user.name", "Test User")
    (repository / "tracked.txt").write_text("clean\n", encoding="utf-8")
    _run_git(repository, "add", "tracked.txt")
    _run_git(repository, "commit", "--quiet", "-m", "initial")
    return repository, _run_git(repository, "rev-parse", "HEAD")


def _official_git_repository(tmp_path: Path) -> tuple[Path, str, str, Path]:
    repository = tmp_path / "reference"
    source = (
        repository
        / "prismatic-synthesis"
        / "gradient_modules"
        / "gradient_computer.py"
    )
    source.parent.mkdir(parents=True)
    source.write_text("class GradientComputer:\n    pass\n", encoding="utf-8")
    _run_git(repository, "init", "--quiet")
    _run_git(repository, "config", "user.email", "test@example.com")
    _run_git(repository, "config", "user.name", "Test User")
    _run_git(repository, "add", ".")
    _run_git(repository, "commit", "--quiet", "-m", "official fixture")
    commit = _run_git(repository, "rev-parse", "HEAD")
    tree = _run_git(repository, "rev-parse", "HEAD^{tree}")
    return repository, commit, tree, source


def _prepared_fixture(
    tmp_path: Path, count: int = 3
) -> tuple[Path, Path, list[dict], dict]:
    source = tmp_path / "source.parquet"
    source.write_bytes(b"source parquet bytes")
    prepared = tmp_path / "prepared.jsonl"
    rows = [
        {
            "id": f"deepmath-level6-{index:06d}",
            "prompt": f"Question {index}",
            "completion": f"Solution {index}",
            "source_row_index": index,
            "original_dataset_index": index + 10,
        }
        for index in range(count)
    ]
    prepared.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    manifest = {
        "manifest_version": 1,
        "dataset_name": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "dataset_split": "train",
        "source_parquet": str(source.resolve()),
        "source_sha256": _sha256(source),
        "source_row_count": count,
        "prepared_jsonl": str(prepared.resolve()),
        "prepared_jsonl_sha256": _sha256(prepared),
        "prepared_row_count": count,
        "expected_count": count,
    }
    manifest_path = tmp_path / "prepared.manifest.json"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    return prepared, manifest_path, rows, manifest


def _ids(count: int) -> list[str]:
    return [f"deepmath-level6-{index:06d}" for index in range(count)]


def _write_chunk(
    directory: Path,
    *,
    prefix: str,
    start: int,
    ids: list[str],
    dtype: torch.dtype = torch.float32,
    width: int = PROJECTION_DIM,
    value: float = 1.0,
    tensor_ids: list[str] | None = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    stored_ids = ids if tensor_ids is None else tensor_ids
    tensors = {
        sample_id: torch.full((width,), value, dtype=dtype)
        for sample_id in stored_ids
    }
    save_file(tensors, directory / f"{prefix}.{start}.safetensors")
    (directory / f"{prefix}.{start}.txt").write_text(
        "".join(json.dumps({"id": sample_id}) + "\n" for sample_id in ids),
        encoding="utf-8",
    )


def _write_matching_gradient_manifest(
    output_dir: Path,
    prepared_jsonl: Path,
    prepared_manifest_path: Path,
    reference_repo: Path,
    reference_commit: str,
    *,
    prefix: str,
    num_shards: int,
) -> dict:
    import math_eval.collect_prismatic_gradients as collection

    prepared_manifest = json.loads(
        prepared_manifest_path.read_text(encoding="utf-8")
    )
    expected = build_gradient_manifest(
        prepared_jsonl,
        prepared_manifest_path,
        prepared_manifest,
        reference_repo,
        reference_commit,
        _run_git(reference_repo, "rev-parse", "HEAD^{tree}"),
        prefix=prefix,
        num_shards=num_shards,
        trl_version=TRL_VERSION,
        package_versions=collection._installed_package_versions(),
    )
    return write_or_validate_gradient_manifest(
        output_dir / collection.GRADIENT_MANIFEST_NAME, expected
    )


def test_verify_reference_repo_accepts_exact_clean_commit(tmp_path: Path) -> None:
    repository, commit = _git_repository(tmp_path)

    verify_reference_repo(repository, commit)


def test_verify_reference_repo_rejects_wrong_head(tmp_path: Path) -> None:
    repository, _ = _git_repository(tmp_path)

    with pytest.raises(ValueError, match="HEAD"):
        verify_reference_repo(repository, "0" * 40)


@pytest.mark.parametrize("dirty_kind", ["tracked", "untracked"])
def test_verify_reference_repo_rejects_all_dirty_files(
    tmp_path: Path, dirty_kind: str
) -> None:
    repository, commit = _git_repository(tmp_path)
    if dirty_kind == "tracked":
        (repository / "tracked.txt").write_text("dirty\n", encoding="utf-8")
    else:
        (repository / "untracked.txt").write_text("untracked\n", encoding="utf-8")

    with pytest.raises(ValueError, match="clean"):
        verify_reference_repo(repository, commit)


def test_load_prepared_pool_validates_manifest_and_every_row(tmp_path: Path) -> None:
    prepared, manifest_path, expected_rows, manifest = _prepared_fixture(tmp_path)

    rows, loaded_manifest = load_prepared_pool(prepared, manifest_path)

    assert rows == expected_rows
    assert loaded_manifest == manifest


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("prepared_jsonl", "/wrong/path.jsonl", "prepared_jsonl"),
        ("prepared_jsonl_sha256", "0" * 64, "prepared_jsonl_sha256"),
        ("prepared_row_count", 2, "row count"),
        ("source_sha256", "0" * 64, "source_sha256"),
        ("dataset_revision", "wrong-revision", "dataset_revision"),
    ],
)
def test_load_prepared_pool_rejects_manifest_provenance_mismatch(
    tmp_path: Path, field: str, replacement: object, message: str
) -> None:
    prepared, manifest_path, _, manifest = _prepared_fixture(tmp_path)
    manifest[field] = replacement
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_prepared_pool(prepared, manifest_path)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("id", "wrong-id", "stable ID"),
        ("prompt", "", "prompt"),
        ("completion", None, "completion"),
        ("source_row_index", 99, "source_row_index"),
    ],
)
def test_load_prepared_pool_rejects_invalid_rows(
    tmp_path: Path, field: str, replacement: object, message: str
) -> None:
    prepared, manifest_path, rows, manifest = _prepared_fixture(tmp_path)
    rows[1][field] = replacement
    prepared.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    manifest["prepared_jsonl_sha256"] = _sha256(prepared)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_prepared_pool(prepared, manifest_path)


def test_load_prepared_pool_rejects_duplicate_ids(tmp_path: Path) -> None:
    prepared, manifest_path, rows, manifest = _prepared_fixture(tmp_path)
    rows[1]["id"] = rows[0]["id"]
    prepared.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    manifest["prepared_jsonl_sha256"] = _sha256(prepared)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        load_prepared_pool(prepared, manifest_path)


def test_write_or_validate_gradient_manifest_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "gradient.manifest.json"
    expected = {
        "prepared_sha256": "a" * 64,
        "prefix": "deepmath",
        "num_shards": 4,
    }

    first = write_or_validate_gradient_manifest(path, expected)
    second = write_or_validate_gradient_manifest(path, expected)

    assert first == expected
    assert second == expected
    assert path.read_text(encoding="utf-8") == (
        '{"num_shards":4,"prefix":"deepmath","prepared_sha256":"'
        + "a" * 64
        + '"}\n'
    )


def test_write_or_validate_gradient_manifest_rejects_any_mismatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "gradient.manifest.json"
    write_or_validate_gradient_manifest(
        path,
        {"prepared_sha256": "a" * 64, "prefix": "deepmath", "num_shards": 4},
    )

    with pytest.raises(ValueError, match="mismatch"):
        write_or_validate_gradient_manifest(
            path,
            {
                "prepared_sha256": "b" * 64,
                "prefix": "deepmath",
                "num_shards": 4,
            },
        )
    with pytest.raises(ValueError, match="mismatch"):
        write_or_validate_gradient_manifest(
            path,
            {
                "prepared_sha256": "a" * 64,
                "prefix": "deepmath",
                "num_shards": 8,
            },
        )


def test_write_or_validate_gradient_manifest_serializes_concurrent_creators(
    tmp_path: Path,
) -> None:
    path = tmp_path / "gradient.manifest.json"
    expected = {
        "prepared_sha256": "a" * 64,
        "prefix": "deepmath",
        "num_shards": 4,
    }

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(
                lambda _: write_or_validate_gradient_manifest(path, expected),
                range(16),
            )
        )

    assert results == [expected] * 16
    assert json.loads(path.read_text(encoding="utf-8")) == expected


@pytest.mark.parametrize(
    "artifact_name",
    [
        "deepmath.0.txt",
        "deepmath.0.safetensors",
        "deepmath.0.safetensors.tmp",
        "deepmath.unexpected",
    ],
)
def test_manifest_creation_refuses_to_retroactively_bless_prefix_artifacts(
    tmp_path: Path, artifact_name: str
) -> None:
    path = tmp_path / "gradient.manifest.json"
    artifact = tmp_path / artifact_name
    artifact.write_bytes(b"crash artifact")
    expected = {
        "prepared_sha256": "a" * 64,
        "prefix": "deepmath",
        "num_shards": 4,
    }

    with pytest.raises(ValueError, match="manifest.*missing.*artifact"):
        write_or_validate_gradient_manifest(path, expected)

    assert not path.exists()
    assert artifact.read_bytes() == b"crash artifact"


def test_resolve_resume_start_returns_shard_start_without_chunks(
    tmp_path: Path,
) -> None:
    assert resolve_resume_start(_ids(4), tmp_path, "deepmath", 0, 2) == 0


def test_resolve_resume_start_accepts_exact_final_short_chunk(tmp_path: Path) -> None:
    ids = _ids(4)
    _write_chunk(tmp_path, prefix="deepmath", start=0, ids=ids[:2])

    assert resolve_resume_start(ids, tmp_path, "deepmath", 0, 2) == 2


def test_resolve_resume_start_accepts_500_row_intermediate_chunk(
    tmp_path: Path,
) -> None:
    ids = _ids(700)
    _write_chunk(
        tmp_path,
        prefix="deepmath",
        start=0,
        ids=ids[:SAVE_INTERVAL],
    )

    assert resolve_resume_start(ids, tmp_path, "deepmath", 0, 700) == 500


def test_resolve_resume_start_ignores_valid_other_shard_chunks(
    tmp_path: Path,
) -> None:
    ids = _ids(4)
    _write_chunk(tmp_path, prefix="deepmath", start=2, ids=ids[2:])

    assert resolve_resume_start(ids, tmp_path, "deepmath", 0, 2) == 0


@pytest.mark.parametrize("start", [2, 999_999])
@pytest.mark.parametrize("file_kind", ["paired", "txt", "safetensors"])
def test_resolve_resume_start_rejects_chunks_starting_outside_dataset(
    tmp_path: Path,
    start: int,
    file_kind: str,
) -> None:
    ids = _ids(2)
    if file_kind == "paired":
        _write_chunk(tmp_path, prefix="deepmath", start=start, ids=[ids[0]])
    elif file_kind == "txt":
        (tmp_path / f"deepmath.{start}.txt").write_text(
            json.dumps({"id": ids[0]}) + "\n", encoding="utf-8"
        )
    else:
        save_file(
            {ids[0]: torch.ones(PROJECTION_DIM, dtype=torch.float16)},
            tmp_path / f"deepmath.{start}.safetensors",
        )

    with pytest.raises(ValueError, match="outside dataset range"):
        resolve_resume_start(ids, tmp_path, "deepmath", 0, 1)


@pytest.mark.parametrize("suffix", ["txt", "safetensors"])
def test_resolve_resume_start_rejects_orphan_chunk_files(
    tmp_path: Path, suffix: str
) -> None:
    ids = _ids(2)
    if suffix == "txt":
        (tmp_path / "deepmath.0.txt").write_text(
            json.dumps({"id": ids[0]}) + "\n", encoding="utf-8"
        )
    else:
        save_file(
            {ids[0]: torch.ones(PROJECTION_DIM, dtype=torch.float16)},
            tmp_path / "deepmath.0.safetensors",
        )

    with pytest.raises(ValueError, match="paired"):
        resolve_resume_start(ids, tmp_path, "deepmath", 0, 2)


def test_resolve_resume_start_rejects_malformed_prefix_filename(
    tmp_path: Path,
) -> None:
    (tmp_path / "deepmath.not-an-index.txt").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="malformed"):
        resolve_resume_start(_ids(2), tmp_path, "deepmath", 0, 2)


@pytest.mark.parametrize(("start", "message"), [(500, "gap"), (400, "overlap")])
def test_resolve_resume_start_rejects_gaps_and_overlaps(
    tmp_path: Path, start: int, message: str
) -> None:
    ids = _ids(1000)
    if start == 400:
        _write_chunk(
            tmp_path,
            prefix="deepmath",
            start=0,
            ids=ids[:SAVE_INTERVAL],
        )
    _write_chunk(
        tmp_path,
        prefix="deepmath",
        start=start,
        ids=ids[start : start + SAVE_INTERVAL],
    )

    with pytest.raises(ValueError, match=message):
        resolve_resume_start(ids, tmp_path, "deepmath", 0, 1000)


def test_resolve_resume_start_rejects_short_intermediate_chunk(
    tmp_path: Path,
) -> None:
    ids = _ids(10)
    _write_chunk(tmp_path, prefix="deepmath", start=0, ids=ids[:2])

    with pytest.raises(ValueError, match="short"):
        resolve_resume_start(ids, tmp_path, "deepmath", 0, 10)


def test_resolve_resume_start_rejects_sidecar_id_order_mismatch(
    tmp_path: Path,
) -> None:
    ids = _ids(2)
    _write_chunk(tmp_path, prefix="deepmath", start=0, ids=list(reversed(ids)))

    with pytest.raises(ValueError, match="sidecar IDs"):
        resolve_resume_start(ids, tmp_path, "deepmath", 0, 2)


@pytest.mark.parametrize(
    ("dtype", "width", "value", "message"),
    [
        (torch.float16, PROJECTION_DIM, 1.0, "float32"),
        (torch.float32, PROJECTION_DIM - 1, 1.0, "shape"),
        (torch.float32, PROJECTION_DIM, float("nan"), "finite"),
        (torch.float32, PROJECTION_DIM, 0.0, "non-zero"),
    ],
)
def test_resolve_resume_start_rejects_invalid_projected_tensor(
    tmp_path: Path,
    dtype: torch.dtype,
    width: int,
    value: float,
    message: str,
) -> None:
    ids = _ids(1)
    _write_chunk(
        tmp_path,
        prefix="deepmath",
        start=0,
        ids=ids,
        dtype=dtype,
        width=width,
        value=value,
    )

    with pytest.raises(ValueError, match=message):
        resolve_resume_start(ids, tmp_path, "deepmath", 0, 1)


def test_resolve_resume_start_rejects_tensor_key_mismatch(tmp_path: Path) -> None:
    ids = _ids(1)
    _write_chunk(
        tmp_path,
        prefix="deepmath",
        start=0,
        ids=ids,
        tensor_ids=["unknown-id"],
    )

    with pytest.raises(ValueError, match="tensor keys"):
        resolve_resume_start(ids, tmp_path, "deepmath", 0, 1)


def test_pinned_gradient_constants_match_the_approved_design() -> None:
    assert REFERENCE_COMMIT == "d9484cd3b5991030b901ac4a3a9e2472dbfac2ad"
    assert MODEL_NAME == "Qwen/Qwen2.5-0.5B-Instruct"
    assert MODEL_REVISION == "7ae557604adf67be50417f59c2c2f167def9a775"
    assert PROJECTION_DIM == 1024
    assert PROJECTION_SEED == 0
    assert PROJECT_INTERVAL == 4
    assert SAVE_INTERVAL == 500
    assert MAX_CONTEXT_TOKENS == 32768
    assert ASSISTANT_RESPONSE_MARKER == "<|im_start|>assistant"
    assert TRL_VERSION == "0.17.0"


def test_build_gradient_manifest_records_all_binding_provenance(
    tmp_path: Path,
) -> None:
    prepared, prepared_manifest_path, _, prepared_manifest = _prepared_fixture(
        tmp_path
    )
    reference, commit, tree, source = _official_git_repository(tmp_path)
    package_versions = {
        "torch": "test-torch",
        "transformers": "test-transformers",
        "trl": "test-trl",
        "trak": "test-trak",
        "fast_jl": "test-fast-jl",
    }

    manifest = build_gradient_manifest(
        prepared,
        prepared_manifest_path,
        prepared_manifest,
        reference,
        commit,
        tree,
        prefix="deepmath",
        num_shards=4,
        trl_version=TRL_VERSION,
        package_versions=package_versions,
    )

    assert manifest == {
        "manifest_version": 1,
        "prepared_jsonl": str(prepared.resolve()),
        "prepared_jsonl_sha256": _sha256(prepared),
        "prepared_manifest": str(prepared_manifest_path.resolve()),
        "prepared_manifest_sha256": _sha256(prepared_manifest_path),
        "prepared_row_count": 3,
        "source_sha256": prepared_manifest["source_sha256"],
        "dataset_name": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "reference_repo": str(reference.resolve()),
        "reference_commit": commit,
        "reference_tree": tree,
        "official_gradient_module": str(source.resolve()),
        "prefix": "deepmath",
        "num_shards": 4,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "trl_version": TRL_VERSION,
        "package_versions": package_versions,
        "projection": {
            "backend": "CudaProjector",
            "dimension": PROJECTION_DIM,
            "seed": PROJECTION_SEED,
            "type": "rademacher",
            "input_dtype": "float16",
            "output_dtype": "float32",
            "project_interval": PROJECT_INTERVAL,
            "save_interval": SAVE_INTERVAL,
            "completion_only_loss": True,
            "full_parameter_gradients": True,
            "response_marker": ASSISTANT_RESPONSE_MARKER,
        },
    }


@pytest.mark.parametrize(
    ("distribution", "actual", "message"),
    [
        ("traker", "0.3.3", r"traker.*expected 0\.3\.2.*got 0\.3\.3"),
        ("fast-jl", "0.1.4", r"fast-jl.*expected 0\.1\.3.*got 0\.1\.4"),
        ("traker", None, r"traker.*expected 0\.3\.2.*not installed"),
        ("fast-jl", None, r"fast-jl.*expected 0\.1\.3.*not installed"),
    ],
)
def test_installed_package_versions_rejects_unpinned_projection_runtime(
    monkeypatch: pytest.MonkeyPatch,
    distribution: str,
    actual: str | None,
    message: str,
) -> None:
    import math_eval.collect_prismatic_gradients as collection

    versions = {
        "torch": "test-torch",
        "transformers": "test-transformers",
        "trl": TRL_VERSION,
        "traker": "0.3.2",
        "fast-jl": "0.1.3",
    }

    def installed_version(name: str) -> str:
        if name == distribution:
            if actual is None:
                raise collection.importlib.metadata.PackageNotFoundError(name)
            return actual
        return versions[name]

    monkeypatch.setattr(collection.importlib.metadata, "version", installed_version)

    with pytest.raises(RuntimeError, match=message):
        collection._installed_package_versions()


def test_installed_package_versions_accepts_and_records_exact_projection_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import math_eval.collect_prismatic_gradients as collection

    versions = {
        "torch": "test-torch",
        "transformers": "test-transformers",
        "trl": TRL_VERSION,
        "traker": "0.3.2",
        "fast-jl": "0.1.3",
    }
    monkeypatch.setattr(
        collection.importlib.metadata, "version", lambda name: versions[name]
    )

    assert collection._installed_package_versions() == {
        "torch": "test-torch",
        "transformers": "test-transformers",
        "trl": TRL_VERSION,
        "trak": "0.3.2",
        "fast_jl": "0.1.3",
    }


def test_validate_single_cuda_device_accepts_only_process_local_cuda_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    validate_single_cuda_device("cuda:0")


def test_validate_single_cuda_device_rejects_other_device_before_cuda_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_probe() -> int:
        raise AssertionError("device_count must not run for an invalid device")

    monkeypatch.setattr(torch.cuda, "device_count", unexpected_probe)

    with pytest.raises(ValueError, match="cuda:0"):
        validate_single_cuda_device("cuda:1")


@pytest.mark.parametrize("device_count", [0, 2, 4])
def test_validate_single_cuda_device_requires_exactly_one_visible_gpu(
    monkeypatch: pytest.MonkeyPatch, device_count: int
) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: device_count)

    with pytest.raises(RuntimeError, match="exactly one"):
        validate_single_cuda_device("cuda:0")


def test_load_official_gradient_computer_class_resolves_exact_reference_file(
    tmp_path: Path,
) -> None:
    reference, commit, _, source = _official_git_repository(tmp_path)

    gradient_computer = load_official_gradient_computer_class(
        reference, commit
    )

    assert inspect.getsourcefile(gradient_computer) == str(source.resolve())


def test_load_official_gradient_computer_class_rejects_wrong_module_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference, commit, _, _ = _official_git_repository(tmp_path)
    monkeypatch.setattr(inspect, "getfile", lambda _: "/wrong/source.py")

    with pytest.raises(RuntimeError, match="origin"):
        load_official_gradient_computer_class(reference, commit)


def test_load_model_and_tokenizer_uses_pinned_revision_and_official_settings() -> None:
    model = MagicMock()
    model.config.max_position_embeddings = MAX_CONTEXT_TOKENS
    moved_model = MagicMock()
    moved_model.config.max_position_embeddings = MAX_CONTEXT_TOKENS
    model.to.return_value = moved_model
    tokenizer = SimpleNamespace(eos_token_id=151645, pad_token_id=None)

    with patch(
        "transformers.AutoModelForCausalLM.from_pretrained", return_value=model
    ) as model_loader, patch(
        "transformers.AutoTokenizer.from_pretrained", return_value=tokenizer
    ) as tokenizer_loader:
        loaded_model, loaded_tokenizer = load_model_and_tokenizer(
            MODEL_NAME, MODEL_REVISION, "cuda:0"
        )

    model_loader.assert_called_once_with(
        MODEL_NAME, revision=MODEL_REVISION, torch_dtype="auto"
    )
    tokenizer_loader.assert_called_once_with(MODEL_NAME, revision=MODEL_REVISION)
    model.to.assert_called_once_with(torch.device("cuda:0"))
    assert loaded_model is moved_model
    assert loaded_tokenizer is tokenizer
    assert tokenizer.pad_token_id == tokenizer.eos_token_id
    assert tokenizer.padding_side == "left"


def test_construct_strict_collector_refuses_basic_projector_before_allocation() -> None:
    events: list[str] = []

    class FakeCudaProjector:
        pass

    class FakeBasicProjector:
        def __init__(self, **_: object) -> None:
            events.append("basic allocated")

    class FakeGradientComputer:
        @staticmethod
        def get_trak_projector(_: object):
            return FakeBasicProjector

        def __init__(self, model_name: str, model: object, tokenizer: object) -> None:
            projector_class = self.get_trak_projector("cuda:0")
            self.projector = projector_class()

    with pytest.raises(RuntimeError, match="BasicProjector"):
        construct_strict_collector(
            FakeGradientComputer,
            MODEL_NAME,
            object(),
            object(),
            cuda_projector_class=FakeCudaProjector,
        )

    assert events == []
    assert FakeGradientComputer.get_trak_projector("cuda:0") is FakeBasicProjector


def test_construct_strict_collector_preserves_official_class_and_cuda_type() -> None:
    class FakeCudaProjector:
        pass

    class FakeGradientComputer:
        @staticmethod
        def get_trak_projector(_: object):
            return FakeCudaProjector

        def __init__(self, model_name: str, model: object, tokenizer: object) -> None:
            self.projector = self.get_trak_projector("cuda:0")()
            self.proj_dim = PROJECTION_DIM
            self.project_interval = PROJECT_INTERVAL
            self.save_interval = SAVE_INTERVAL

    collector = construct_strict_collector(
        FakeGradientComputer,
        MODEL_NAME,
        object(),
        object(),
        cuda_projector_class=FakeCudaProjector,
    )

    assert type(collector) is FakeGradientComputer
    assert type(collector.projector) is FakeCudaProjector


class _PreflightTokenizer:
    model_max_length = 131_072

    def __init__(self, formatted: dict[str, str] | None = None) -> None:
        self.formatted = formatted or {}
        self.calls: list[dict] = []

    def apply_chat_template(self, messages: list[dict], **kwargs: object) -> str:
        self.calls.append(dict(kwargs))
        prompt = messages[0]["content"]
        return self.formatted.get(
            prompt,
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"{ASSISTANT_RESPONSE_MARKER}\n{messages[1]['content']}<|im_end|>",
        )


class _PreflightCollector:
    def __init__(self, lengths: dict[str, int], supervised: dict[str, int] | None = None):
        self.lengths = lengths
        self.supervised = supervised or {}
        self.seen: list[str] = []
        self.tokenizer = _PreflightTokenizer()

    def prepare_model_input(
        self, prompt: str, completion: str
    ) -> dict[str, torch.Tensor]:
        self.seen.append(prompt)
        length = self.lengths[prompt]
        labels = torch.full((1, length), -100, dtype=torch.long)
        supervised_count = self.supervised.get(prompt, 1)
        if supervised_count:
            labels[0, -supervised_count:] = 1
        return {
            "input_ids": torch.zeros((1, length), dtype=torch.long),
            "labels": labels,
        }


def test_preflight_samples_accepts_exact_model_context_boundary() -> None:
    collector = _PreflightCollector({"boundary": MAX_CONTEXT_TOKENS})
    samples = [{"id": "boundary-id", "prompt": "boundary", "completion": "ok"}]

    summary = preflight_samples(samples, collector.tokenizer, collector)

    assert summary == {
        "sample_count": 1,
        "max_input_tokens": MAX_CONTEXT_TOKENS,
        "min_supervised_tokens": 1,
    }


def test_preflight_samples_reports_all_overlong_ids_deterministically() -> None:
    collector = _PreflightCollector(
        {
            "first": MAX_CONTEXT_TOKENS + 1,
            "valid": 12,
            "second": MAX_CONTEXT_TOKENS + 866,
        }
    )
    samples = [
        {"id": "first-id", "prompt": "first", "completion": "x"},
        {"id": "valid-id", "prompt": "valid", "completion": "x"},
        {"id": "second-id", "prompt": "second", "completion": "x"},
    ]

    with pytest.raises(ValueError) as error:
        preflight_samples(samples, collector.tokenizer, collector)

    assert "first-id=32769" in str(error.value)
    assert "second-id=33634" in str(error.value)
    assert str(error.value).index("first-id") < str(error.value).index("second-id")
    assert collector.seen == ["first", "valid", "second"]


def test_preflight_samples_rejects_empty_completion_only_labels() -> None:
    collector = _PreflightCollector({"bad": 5}, supervised={"bad": 0})
    samples = [{"id": "bad-id", "prompt": "bad", "completion": "x"}]

    with pytest.raises(ValueError, match="bad-id.*no supervised"):
        preflight_samples(samples, collector.tokenizer, collector)


@pytest.mark.parametrize(
    "formatted",
    [
        "<|im_start|>user\nquestion<|im_end|>",
        f"{ASSISTANT_RESPONSE_MARKER}\n{ASSISTANT_RESPONSE_MARKER}\nanswer",
    ],
)
def test_preflight_samples_requires_exactly_one_official_response_marker(
    formatted: str,
) -> None:
    collector = _PreflightCollector({"question": 17})
    collector.tokenizer = _PreflightTokenizer({"question": formatted})
    samples = [{"id": "marker-id", "prompt": "question", "completion": "answer"}]

    with pytest.raises(ValueError, match=r"marker-id.*17 tokens.*exactly once"):
        preflight_samples(samples, collector.tokenizer, collector)


def test_preflight_samples_uses_pinned_limit_not_tokenizer_advertisement() -> None:
    collector = _PreflightCollector({"long": MAX_CONTEXT_TOKENS + 1})
    samples = [{"id": "long-id", "prompt": "long", "completion": "answer"}]

    with pytest.raises(ValueError, match=r"long-id=32769"):
        preflight_samples(samples, collector.tokenizer, collector)

    assert collector.tokenizer.model_max_length == 131_072
    assert collector.tokenizer.calls == [{"tokenize": False}]


@pytest.mark.parametrize(
    "prefix",
    ["", "../escape", "/tmp/x", "a/b", ".", "line\nbreak", "has space"],
)
def test_run_collection_rejects_unsafe_prefix_without_output_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prefix: str,
) -> None:
    import math_eval.collect_prismatic_gradients as collection

    prepared, prepared_manifest_path, _, _ = _prepared_fixture(tmp_path, count=1)
    reference, commit, _, _ = _official_git_repository(tmp_path)
    output_dir = tmp_path / "gradients"
    monkeypatch.setattr(collection, "REFERENCE_COMMIT", commit)
    monkeypatch.setattr(
        collection,
        "validate_single_cuda_device",
        lambda *_: (_ for _ in ()).throw(
            AssertionError("unsafe prefix must be rejected before CUDA")
        ),
    )

    with pytest.raises(ValueError, match="prefix"):
        run_collection(
            reference_repo=reference,
            prepared_jsonl=prepared,
            prepared_manifest=prepared_manifest_path,
            output_dir=output_dir,
            prefix=prefix,
            model_name=MODEL_NAME,
            model_revision=MODEL_REVISION,
            shard_index=0,
            num_shards=1,
            device="cuda:0",
        )

    assert not output_dir.exists()


def test_run_collection_complete_shard_returns_before_cuda_or_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import math_eval.collect_prismatic_gradients as collection

    prepared, prepared_manifest_path, rows, _ = _prepared_fixture(tmp_path, count=2)
    output_dir = tmp_path / "gradients"
    reference, commit, _, _ = _official_git_repository(tmp_path)
    _write_matching_gradient_manifest(
        output_dir,
        prepared,
        prepared_manifest_path,
        reference,
        commit,
        prefix="deepmath",
        num_shards=1,
    )
    _write_chunk(
        output_dir,
        prefix="deepmath",
        start=0,
        ids=[row["id"] for row in rows],
    )
    monkeypatch.setattr(collection, "REFERENCE_COMMIT", commit)
    monkeypatch.setattr(
        collection,
        "validate_single_cuda_device",
        lambda *_: (_ for _ in ()).throw(AssertionError("CUDA must not be queried")),
    )
    monkeypatch.setattr(
        collection,
        "load_model_and_tokenizer",
        lambda *_: (_ for _ in ()).throw(AssertionError("model must not load")),
    )

    result = run_collection(
        reference_repo=reference,
        prepared_jsonl=prepared,
        prepared_manifest=prepared_manifest_path,
        output_dir=output_dir,
        prefix="deepmath",
        model_name=MODEL_NAME,
        model_revision=MODEL_REVISION,
        shard_index=0,
        num_shards=1,
        device="cuda:0",
    )

    assert result["status"] == "complete"
    assert result["resume_start"] == 2
    assert (output_dir / "gradient.manifest.json").is_file()
    assert (output_dir / ".deepmath.shard-00000-of-00001.lock").is_file()


def test_run_collection_calls_official_once_on_only_unresolved_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import math_eval.collect_prismatic_gradients as collection

    prepared, prepared_manifest_path, rows, _ = _prepared_fixture(
        tmp_path, count=SAVE_INTERVAL + 2
    )
    output_dir = tmp_path / "gradients"
    ids = [row["id"] for row in rows]
    reference, commit, _, _ = _official_git_repository(tmp_path)
    _write_matching_gradient_manifest(
        output_dir,
        prepared,
        prepared_manifest_path,
        reference,
        commit,
        prefix="deepmath",
        num_shards=1,
    )
    _write_chunk(
        output_dir,
        prefix="deepmath",
        start=0,
        ids=ids[:SAVE_INTERVAL],
    )
    calls: list[tuple[list[dict], str, Path, int]] = []

    class FakeCollector(_PreflightCollector):
        def compute_project_store_gradients(
            self,
            samples: list[dict],
            prefix: str,
            directory: Path,
            global_start: int,
        ) -> None:
            calls.append((samples, prefix, Path(directory), global_start))
            _write_chunk(
                Path(directory),
                prefix=prefix,
                start=global_start,
                ids=[sample["id"] for sample in samples],
            )

    collector = FakeCollector(
        {row["prompt"]: 10 for row in rows[SAVE_INTERVAL:]}
    )
    model = SimpleNamespace(
        config=SimpleNamespace(max_position_embeddings=MAX_CONTEXT_TOKENS)
    )
    monkeypatch.setattr(collection, "REFERENCE_COMMIT", commit)
    monkeypatch.setattr(collection, "validate_single_cuda_device", lambda *_: None)
    monkeypatch.setattr(
        collection, "load_official_gradient_computer_class", lambda *_: object
    )
    monkeypatch.setattr(
        collection,
        "load_model_and_tokenizer",
        lambda *_: (model, collector.tokenizer),
    )
    monkeypatch.setattr(
        collection, "construct_strict_collector", lambda *_args, **_kwargs: collector
    )

    result = run_collection(
        reference_repo=reference,
        prepared_jsonl=prepared,
        prepared_manifest=prepared_manifest_path,
        output_dir=output_dir,
        prefix="deepmath",
        model_name=MODEL_NAME,
        model_revision=MODEL_REVISION,
        shard_index=0,
        num_shards=1,
        device="cuda:0",
    )

    assert result["status"] == "collected"
    assert len(calls) == 1
    samples, called_prefix, called_directory, global_start = calls[0]
    assert samples == rows[SAVE_INTERVAL:]
    assert called_prefix == "deepmath"
    assert called_directory == output_dir
    assert global_start == SAVE_INTERVAL
    assert result["resume_start"] == SAVE_INTERVAL
    assert result["shard_end"] == SAVE_INTERVAL + 2


def test_run_collection_rejects_nonpinned_model_context_before_collector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import math_eval.collect_prismatic_gradients as collection

    prepared, prepared_manifest_path, _, _ = _prepared_fixture(tmp_path, count=1)
    reference, commit, _, _ = _official_git_repository(tmp_path)
    model = SimpleNamespace(
        config=SimpleNamespace(max_position_embeddings=131072)
    )
    monkeypatch.setattr(collection, "REFERENCE_COMMIT", commit)
    monkeypatch.setattr(collection, "validate_single_cuda_device", lambda *_: None)
    monkeypatch.setattr(
        collection, "load_official_gradient_computer_class", lambda *_: object
    )
    monkeypatch.setattr(
        collection, "load_model_and_tokenizer", lambda *_: (model, object())
    )
    monkeypatch.setattr(
        collection,
        "construct_strict_collector",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("collector must not be constructed")
        ),
    )

    with pytest.raises(ValueError, match="32768"):
        run_collection(
            reference_repo=tmp_path / "reference",
            prepared_jsonl=prepared,
            prepared_manifest=prepared_manifest_path,
            output_dir=tmp_path / "gradients",
            prefix="deepmath",
            model_name=MODEL_NAME,
            model_revision=MODEL_REVISION,
            shard_index=0,
            num_shards=1,
            device="cuda:0",
        )


@pytest.mark.parametrize(
    "sidecar_ids",
    [
        ["deepmath-level6-999999"],
        ["deepmath-level6-000000", "deepmath-level6-000000"],
    ],
)
def test_resolve_resume_start_rejects_unknown_and_duplicate_ids(
    tmp_path: Path, sidecar_ids: list[str]
) -> None:
    expected_ids = _ids(len(sidecar_ids))
    _write_chunk(
        tmp_path,
        prefix="deepmath",
        start=0,
        ids=sidecar_ids,
    )

    with pytest.raises(ValueError, match="sidecar IDs"):
        resolve_resume_start(expected_ids, tmp_path, "deepmath", 0, len(expected_ids))


def test_resolve_resume_start_rejects_unreadable_safetensors(tmp_path: Path) -> None:
    sample_id = _ids(1)[0]
    (tmp_path / "deepmath.0.txt").write_text(
        json.dumps({"id": sample_id}) + "\n", encoding="utf-8"
    )
    (tmp_path / "deepmath.0.safetensors").write_bytes(b"not safetensors")

    with pytest.raises(ValueError, match="unreadable"):
        resolve_resume_start([sample_id], tmp_path, "deepmath", 0, 1)


def test_official_four_way_shard_bounds_are_exact() -> None:
    assert [
        official_shard_bounds(57_046, 4, shard_index)
        for shard_index in range(4)
    ] == [
        (0, 14_262),
        (14_262, 28_524),
        (28_524, 42_786),
        (42_786, 57_046),
    ]


def test_run_collection_complete_shard_skips_official_module_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import math_eval.collect_prismatic_gradients as collection

    prepared, prepared_manifest_path, rows, _ = _prepared_fixture(tmp_path, count=1)
    output_dir = tmp_path / "gradients"
    reference, commit, _, _ = _official_git_repository(tmp_path)
    _write_matching_gradient_manifest(
        output_dir,
        prepared,
        prepared_manifest_path,
        reference,
        commit,
        prefix="deepmath",
        num_shards=1,
    )
    _write_chunk(
        output_dir,
        prefix="deepmath",
        start=0,
        ids=[rows[0]["id"]],
    )
    monkeypatch.setattr(collection, "REFERENCE_COMMIT", commit)
    monkeypatch.setattr(
        collection,
        "load_official_gradient_computer_class",
        lambda *_: (_ for _ in ()).throw(AssertionError("official import must not run")),
    )

    result = run_collection(
        reference_repo=reference,
        prepared_jsonl=prepared,
        prepared_manifest=prepared_manifest_path,
        output_dir=output_dir,
        prefix="deepmath",
        model_name=MODEL_NAME,
        model_revision=MODEL_REVISION,
        shard_index=0,
        num_shards=1,
        device="cuda:0",
    )

    assert result["status"] == "complete"


def test_run_collection_cleanly_skips_one_trailing_empty_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import math_eval.collect_prismatic_gradients as collection

    prepared, prepared_manifest_path, _, _ = _prepared_fixture(tmp_path, count=1)
    reference, commit, _, _ = _official_git_repository(tmp_path)
    monkeypatch.setattr(collection, "REFERENCE_COMMIT", commit)
    monkeypatch.setattr(
        collection,
        "validate_single_cuda_device",
        lambda *_: (_ for _ in ()).throw(AssertionError("CUDA must not be queried")),
    )

    result = run_collection(
        reference_repo=reference,
        prepared_jsonl=prepared,
        prepared_manifest=prepared_manifest_path,
        output_dir=tmp_path / "gradients",
        prefix="deepmath",
        model_name=MODEL_NAME,
        model_revision=MODEL_REVISION,
        shard_index=1,
        num_shards=2,
        device="cuda:0",
    )

    assert result == {
        "status": "empty",
        "shard_start": 1,
        "shard_end": 1,
        "resume_start": 1,
    }
    assert (
        tmp_path
        / "gradients"
        / ".deepmath.shard-00001-of-00002.lock"
    ).is_file()


def test_run_collection_rejects_empty_shard_beyond_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import math_eval.collect_prismatic_gradients as collection

    prepared, prepared_manifest_path, _, _ = _prepared_fixture(tmp_path, count=1)
    reference, commit, _, _ = _official_git_repository(tmp_path)
    output_dir = tmp_path / "gradients"
    monkeypatch.setattr(collection, "REFERENCE_COMMIT", commit)

    with pytest.raises(ValueError, match="empty logical shard"):
        run_collection(
            reference_repo=reference,
            prepared_jsonl=prepared,
            prepared_manifest=prepared_manifest_path,
            output_dir=output_dir,
            prefix="deepmath",
            model_name=MODEL_NAME,
            model_revision=MODEL_REVISION,
            shard_index=2,
            num_shards=3,
            device="cuda:0",
        )

    assert not output_dir.exists()


def test_run_collection_rejects_collectively_impossible_shard_layout_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import math_eval.collect_prismatic_gradients as collection

    prepared, prepared_manifest_path, _, _ = _prepared_fixture(tmp_path, count=1)
    reference, commit, _, _ = _official_git_repository(tmp_path)
    output_dir = tmp_path / "gradients"
    monkeypatch.setattr(collection, "REFERENCE_COMMIT", commit)

    with pytest.raises(ValueError, match="empty logical shard"):
        run_collection(
            reference_repo=reference,
            prepared_jsonl=prepared,
            prepared_manifest=prepared_manifest_path,
            output_dir=output_dir,
            prefix="deepmath",
            model_name=MODEL_NAME,
            model_revision=MODEL_REVISION,
            shard_index=0,
            num_shards=3,
            device="cuda:0",
        )

    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("num_shards", "shard_index"),
    [(0, 0), (2, -1), (2, 2)],
)
def test_run_collection_rejects_invalid_shard_arguments_without_output_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    num_shards: int,
    shard_index: int,
) -> None:
    import math_eval.collect_prismatic_gradients as collection

    prepared, prepared_manifest_path, _, _ = _prepared_fixture(tmp_path, count=1)
    reference, commit, _, _ = _official_git_repository(tmp_path)
    output_dir = tmp_path / "gradients"
    monkeypatch.setattr(collection, "REFERENCE_COMMIT", commit)

    with pytest.raises(ValueError, match="num_shards|shard_index"):
        run_collection(
            reference_repo=reference,
            prepared_jsonl=prepared,
            prepared_manifest=prepared_manifest_path,
            output_dir=output_dir,
            prefix="deepmath",
            model_name=MODEL_NAME,
            model_revision=MODEL_REVISION,
            shard_index=shard_index,
            num_shards=num_shards,
            device="cuda:0",
        )

    assert not output_dir.exists()


def test_run_collection_rejects_invalid_device_even_when_shard_is_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import math_eval.collect_prismatic_gradients as collection

    prepared, prepared_manifest_path, rows, _ = _prepared_fixture(tmp_path, count=1)
    output_dir = tmp_path / "gradients"
    _write_chunk(output_dir, prefix="deepmath", start=0, ids=[rows[0]["id"]])
    reference, commit, _, _ = _official_git_repository(tmp_path)
    monkeypatch.setattr(collection, "REFERENCE_COMMIT", commit)

    with pytest.raises(ValueError, match="cuda:0"):
        run_collection(
            reference_repo=reference,
            prepared_jsonl=prepared,
            prepared_manifest=prepared_manifest_path,
            output_dir=output_dir,
            prefix="deepmath",
            model_name=MODEL_NAME,
            model_revision=MODEL_REVISION,
            shard_index=0,
            num_shards=1,
            device="cuda:1",
        )


def test_run_collection_revalidates_and_rejects_incomplete_official_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import math_eval.collect_prismatic_gradients as collection

    prepared, prepared_manifest_path, _, _ = _prepared_fixture(tmp_path, count=1)
    reference, commit, _, _ = _official_git_repository(tmp_path)
    monkeypatch.setattr(collection, "REFERENCE_COMMIT", commit)
    monkeypatch.setattr(collection, "validate_single_cuda_device", lambda *_: None)
    monkeypatch.setattr(
        collection, "load_official_gradient_computer_class", lambda *_: object
    )
    fake_collector = _PreflightCollector({"Question 0": 8})
    fake_collector.compute_project_store_gradients = lambda *_: None
    model = SimpleNamespace(
        config=SimpleNamespace(max_position_embeddings=MAX_CONTEXT_TOKENS)
    )
    monkeypatch.setattr(
        collection,
        "load_model_and_tokenizer",
        lambda *_: (model, fake_collector.tokenizer),
    )
    monkeypatch.setattr(
        collection,
        "construct_strict_collector",
        lambda *_args, **_kwargs: fake_collector,
    )

    with pytest.raises(RuntimeError, match="complete shard coverage"):
        run_collection(
            reference_repo=reference,
            prepared_jsonl=prepared,
            prepared_manifest=prepared_manifest_path,
            output_dir=tmp_path / "gradients",
            prefix="deepmath",
            model_name=MODEL_NAME,
            model_revision=MODEL_REVISION,
            shard_index=0,
            num_shards=1,
            device="cuda:0",
        )


@pytest.mark.parametrize(
    ("model_name", "model_revision", "message"),
    [
        ("other/model", MODEL_REVISION, "model name"),
        (MODEL_NAME, "moving-main", "model revision"),
    ],
)
def test_load_model_and_tokenizer_rejects_unpinned_inputs_before_loading(
    model_name: str, model_revision: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_model_and_tokenizer(model_name, model_revision, "cuda:0")


def test_direct_cli_help_works_without_pythonpath(tmp_path: Path) -> None:
    script = Path(__file__).with_name("collect_prismatic_gradients.py")
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    for flag in (
        "--reference-repo",
        "--prepared-jsonl",
        "--prepared-manifest",
        "--output-dir",
        "--prefix",
        "--model-name",
        "--model-revision",
        "--shard-index",
        "--num-shards",
        "--device",
        "--validate-global-only",
    ):
        assert flag in result.stdout


def test_shard_locks_fail_fast_for_same_shard_and_allow_distinct_shards(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "gradients"
    output_dir.mkdir()

    with _exclusive_shard_lock(output_dir, "deepmath", 4, 0) as first_path:
        with pytest.raises(RuntimeError, match="already active"):
            with _exclusive_shard_lock(output_dir, "deepmath", 4, 0):
                pass
        with _exclusive_shard_lock(output_dir, "deepmath", 4, 1) as second_path:
            assert second_path != first_path

    assert first_path.is_file()
    assert second_path.is_file()


def test_run_collection_serializes_duplicate_logical_shard_through_postvalidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import math_eval.collect_prismatic_gradients as collection

    prepared, prepared_manifest_path, rows, _ = _prepared_fixture(tmp_path, count=1)
    reference, commit, _, _ = _official_git_repository(tmp_path)
    output_dir = tmp_path / "gradients"
    monkeypatch.setattr(collection, "REFERENCE_COMMIT", commit)
    monkeypatch.setattr(collection, "validate_single_cuda_device", lambda *_: None)
    monkeypatch.setattr(
        collection, "load_official_gradient_computer_class", lambda *_: object
    )
    tokenizer = _PreflightTokenizer()
    model = SimpleNamespace(
        config=SimpleNamespace(max_position_embeddings=MAX_CONTEXT_TOKENS)
    )
    monkeypatch.setattr(
        collection, "load_model_and_tokenizer", lambda *_: (model, tokenizer)
    )

    state_lock = threading.Lock()
    first_entered = threading.Event()
    release_first = threading.Event()
    active = 0
    max_active = 0
    call_count = 0

    class BlockingCollector(_PreflightCollector):
        def compute_project_store_gradients(
            self, samples, prefix, directory, global_start
        ) -> None:
            nonlocal active, max_active, call_count
            with state_lock:
                call_count += 1
                own_call = call_count
                active += 1
                max_active = max(max_active, active)
                first_entered.set()
            try:
                assert release_first.wait(timeout=10)
                if own_call == 1:
                    _write_chunk(
                        Path(directory),
                        prefix=prefix,
                        start=global_start,
                        ids=[sample["id"] for sample in samples],
                    )
            finally:
                with state_lock:
                    active -= 1

    monkeypatch.setattr(
        collection,
        "construct_strict_collector",
        lambda *_args, **_kwargs: BlockingCollector({"Question 0": 8}),
    )
    arguments = {
        "reference_repo": reference,
        "prepared_jsonl": prepared,
        "prepared_manifest": prepared_manifest_path,
        "output_dir": output_dir,
        "prefix": "deepmath",
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "shard_index": 0,
        "num_shards": 1,
        "device": "cuda:0",
    }

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(run_collection, **arguments)
        assert first_entered.wait(timeout=5)
        second = executor.submit(run_collection, **arguments)
        try:
            with pytest.raises(RuntimeError, match="already active"):
                second.result(timeout=5)
        finally:
            release_first.set()
        assert first.result(timeout=10)["status"] == "collected"

    assert max_active == 1
    assert call_count == 1
    assert [row["id"] for row in rows] == _ids(1)


@pytest.mark.parametrize("use_symlink", [False, True])
def test_run_collection_rejects_output_inside_reference_before_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_symlink: bool,
) -> None:
    import math_eval.collect_prismatic_gradients as collection

    prepared, prepared_manifest_path, _, _ = _prepared_fixture(tmp_path, count=1)
    reference, commit, _, _ = _official_git_repository(tmp_path)
    monkeypatch.setattr(collection, "REFERENCE_COMMIT", commit)
    parent = reference
    if use_symlink:
        parent = tmp_path / "reference-alias"
        parent.symlink_to(reference, target_is_directory=True)
    output_dir = parent / "generated-gradients"

    with pytest.raises(ValueError, match="reference repository"):
        run_collection(
            reference_repo=reference,
            prepared_jsonl=prepared,
            prepared_manifest=prepared_manifest_path,
            output_dir=output_dir,
            prefix="deepmath",
            model_name=MODEL_NAME,
            model_revision=MODEL_REVISION,
            shard_index=1,
            num_shards=2,
            device="cuda:0",
        )

    assert not output_dir.exists()
    assert _run_git(reference, "status", "--porcelain") == ""


@pytest.mark.parametrize("input_kind", ["prepared", "manifest", "source"])
def test_run_collection_rejects_output_equal_to_input_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    input_kind: str,
) -> None:
    import math_eval.collect_prismatic_gradients as collection

    prepared, prepared_manifest_path, _, manifest = _prepared_fixture(
        tmp_path, count=1
    )
    reference, commit, _, _ = _official_git_repository(tmp_path)
    monkeypatch.setattr(collection, "REFERENCE_COMMIT", commit)
    output_dir = {
        "prepared": prepared,
        "manifest": prepared_manifest_path,
        "source": Path(manifest["source_parquet"]),
    }[input_kind]

    with pytest.raises(ValueError, match="input artifact"):
        run_collection(
            reference_repo=reference,
            prepared_jsonl=prepared,
            prepared_manifest=prepared_manifest_path,
            output_dir=output_dir,
            prefix="deepmath",
            model_name=MODEL_NAME,
            model_revision=MODEL_REVISION,
            shard_index=0,
            num_shards=1,
            device="cuda:0",
        )


def test_run_collection_rejects_output_containing_input_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import math_eval.collect_prismatic_gradients as collection

    prepared, prepared_manifest_path, _, _ = _prepared_fixture(tmp_path, count=1)
    reference, commit, _, _ = _official_git_repository(tmp_path)
    monkeypatch.setattr(collection, "REFERENCE_COMMIT", commit)

    with pytest.raises(ValueError, match="contains input artifact"):
        run_collection(
            reference_repo=reference,
            prepared_jsonl=prepared,
            prepared_manifest=prepared_manifest_path,
            output_dir=tmp_path,
            prefix="deepmath",
            model_name=MODEL_NAME,
            model_revision=MODEL_REVISION,
            shard_index=0,
            num_shards=1,
            device="cuda:0",
        )


def _write_complete_global_chunks(
    output_dir: Path, ids: list[str], num_shards: int
) -> int:
    chunk_count = 0
    for shard_index in range(num_shards):
        shard_start, shard_end = official_shard_bounds(
            len(ids), num_shards, shard_index
        )
        cursor = shard_start
        while cursor < shard_end:
            chunk_end = min(cursor + SAVE_INTERVAL, shard_end)
            _write_chunk(
                output_dir,
                prefix="deepmath",
                start=cursor,
                ids=ids[cursor:chunk_end],
            )
            cursor = chunk_end
            chunk_count += 1
    return chunk_count


def test_global_gradient_validation_proves_exact_all_shard_coverage(
    tmp_path: Path,
) -> None:
    ids = _ids(4)
    chunk_count = _write_complete_global_chunks(tmp_path, ids, num_shards=2)

    result = validate_global_gradient_coverage(ids, tmp_path, "deepmath", 2)

    assert result == {
        "row_count": 4,
        "chunk_count": chunk_count,
        "shard_count": 2,
    }


def test_global_validation_refuses_to_scan_while_a_collector_is_active(
    tmp_path: Path,
) -> None:
    ids = _ids(4)
    _write_complete_global_chunks(tmp_path, ids, num_shards=2)

    with _exclusive_shard_lock(tmp_path, "deepmath", 2, 0):
        with pytest.raises(RuntimeError, match="active"):
            validate_global_gradient_coverage(ids, tmp_path, "deepmath", 2)


def test_global_validation_rejects_cross_shard_chunk_400_to_900(
    tmp_path: Path,
) -> None:
    ids = _ids(999)
    _write_chunk(
        tmp_path,
        prefix="deepmath",
        start=400,
        ids=ids[400:900],
    )

    with pytest.raises(ValueError, match="crosses logical shard boundary"):
        validate_global_gradient_coverage(ids, tmp_path, "deepmath", 2)


def test_owning_shard_rejects_chunk_extending_past_its_end(tmp_path: Path) -> None:
    ids = _ids(999)
    _write_chunk(
        tmp_path,
        prefix="deepmath",
        start=400,
        ids=ids[400:900],
    )

    with pytest.raises(ValueError, match="beyond logical shard end"):
        resolve_resume_start(ids, tmp_path, "deepmath", 0, 500)


def test_global_validation_rejects_far_out_numeric_start(tmp_path: Path) -> None:
    ids = _ids(1)
    _write_chunk(
        tmp_path,
        prefix="deepmath",
        start=999_999,
        ids=ids,
    )

    with pytest.raises(ValueError, match="outside dataset range"):
        validate_global_gradient_coverage(ids, tmp_path, "deepmath", 1)


def test_global_validation_rejects_prefix_owned_temporary_suffix(
    tmp_path: Path,
) -> None:
    ids = _ids(1)
    _write_complete_global_chunks(tmp_path, ids, num_shards=1)
    partial = tmp_path / "deepmath.0.safetensors.tmp"
    partial.write_bytes(b"partial")

    with pytest.raises(ValueError, match="malformed gradient artifact"):
        validate_global_gradient_coverage(ids, tmp_path, "deepmath", 1)

    assert partial.is_file()


def test_global_validation_preserves_orphan_from_crashed_writer(
    tmp_path: Path,
) -> None:
    orphan = tmp_path / "deepmath.0.safetensors"
    save_file({_ids(1)[0]: torch.ones(PROJECTION_DIM, dtype=torch.float16)}, orphan)

    with pytest.raises(ValueError, match="paired"):
        validate_global_gradient_coverage(_ids(1), tmp_path, "deepmath", 1)

    assert orphan.is_file()


def test_run_global_validation_loads_prepared_ids_and_returns_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import math_eval.collect_prismatic_gradients as collection

    prepared, prepared_manifest_path, rows, _ = _prepared_fixture(tmp_path, count=4)
    output_dir = tmp_path / "gradients"
    reference, commit, _, _ = _official_git_repository(tmp_path)
    monkeypatch.setattr(collection, "REFERENCE_COMMIT", commit)
    _write_matching_gradient_manifest(
        output_dir,
        prepared,
        prepared_manifest_path,
        reference,
        commit,
        prefix="deepmath",
        num_shards=2,
    )
    _write_complete_global_chunks(
        output_dir, [row["id"] for row in rows], num_shards=2
    )

    result = run_global_validation(
        prepared_jsonl=prepared,
        prepared_manifest=prepared_manifest_path,
        output_dir=output_dir,
        prefix="deepmath",
        num_shards=2,
    )

    assert result["status"] == "validated"
    assert result["row_count"] == 4


def test_run_global_validation_requires_existing_manifest_without_mutation(
    tmp_path: Path,
) -> None:
    import math_eval.collect_prismatic_gradients as collection

    prepared, prepared_manifest_path, rows, _ = _prepared_fixture(tmp_path, count=4)
    output_dir = tmp_path / "gradients"
    _write_complete_global_chunks(
        output_dir, [row["id"] for row in rows], num_shards=2
    )
    manifest_path = output_dir / collection.GRADIENT_MANIFEST_NAME
    before = {path.name: path.read_bytes() for path in output_dir.iterdir()}

    with pytest.raises(ValueError, match="gradient manifest.*does not exist"):
        run_global_validation(
            prepared_jsonl=prepared,
            prepared_manifest=prepared_manifest_path,
            output_dir=output_dir,
            prefix="deepmath",
            num_shards=2,
        )

    assert not manifest_path.exists()
    assert {path.name: path.read_bytes() for path in output_dir.iterdir()} == before


@pytest.mark.parametrize(
    ("field_path", "replacement"),
    [
        (("prepared_jsonl_sha256",), "0" * 64),
        (("model_revision",), "moving-main"),
        (("prefix",), "other-prefix"),
        (("num_shards",), 1),
        (("projection", "backend"), "BasicProjector"),
        (("projection", "output_dtype"), "float16"),
        (("package_versions", "trak"), "0.3.3"),
    ],
)
def test_run_global_validation_rejects_any_manifest_provenance_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field_path: tuple[str, ...],
    replacement: object,
) -> None:
    import math_eval.collect_prismatic_gradients as collection

    prepared, prepared_manifest_path, rows, _ = _prepared_fixture(tmp_path, count=4)
    output_dir = tmp_path / "gradients"
    reference, commit, _, _ = _official_git_repository(tmp_path)
    monkeypatch.setattr(collection, "REFERENCE_COMMIT", commit)
    _write_matching_gradient_manifest(
        output_dir,
        prepared,
        prepared_manifest_path,
        reference,
        commit,
        prefix="deepmath",
        num_shards=2,
    )
    _write_complete_global_chunks(
        output_dir, [row["id"] for row in rows], num_shards=2
    )
    manifest_path = output_dir / collection.GRADIENT_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target = manifest
    for field in field_path[:-1]:
        target = target[field]
    target[field_path[-1]] = replacement
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="gradient manifest.*mismatch"):
        run_global_validation(
            prepared_jsonl=prepared,
            prepared_manifest=prepared_manifest_path,
            output_dir=output_dir,
            prefix="deepmath",
            num_shards=2,
        )


def test_global_validation_cli_mode_needs_no_model_reference_or_device(
    tmp_path: Path,
) -> None:
    reference = Path("/home/mchen/prismatic-synthesis-reference")
    verify_reference_repo(reference, REFERENCE_COMMIT)
    prepared, prepared_manifest_path, rows, _ = _prepared_fixture(tmp_path, count=4)
    output_dir = tmp_path / "gradients"
    _write_matching_gradient_manifest(
        output_dir,
        prepared,
        prepared_manifest_path,
        reference,
        REFERENCE_COMMIT,
        prefix="deepmath",
        num_shards=2,
    )
    _write_complete_global_chunks(
        output_dir, [row["id"] for row in rows], num_shards=2
    )
    script = Path(__file__).with_name("collect_prismatic_gradients.py")
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--validate-global-only",
            "--prepared-jsonl",
            str(prepared),
            "--prepared-manifest",
            str(prepared_manifest_path),
            "--output-dir",
            str(output_dir),
            "--prefix",
            "deepmath",
            "--num-shards",
            "2",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "validated"
