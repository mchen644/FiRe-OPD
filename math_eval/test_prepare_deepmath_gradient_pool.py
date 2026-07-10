import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from math_eval import prepare_deepmath_gradient_pool as preparation
from math_eval.deepmath_gradient_diversity import sha256_file


OPD_SUFFIX = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}."
)


def _filtered_row(question: str, answer: str, index: int) -> dict:
    return {
        "prompt": [{"role": "user", "content": question + OPD_SUFFIX}],
        "reward_model": {"ground_truth": answer},
        "extra_info": {"index": index, "split": "train"},
    }


def _original_row(
    question: str, answer: str, completion: str, difficulty: float = 6.0
) -> dict:
    return {
        "question": question,
        "final_answer": answer,
        "r1_solution_1": completion,
        "topic": "Algebra",
        "difficulty": difficulty,
    }


def _write_source(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _fixture_inputs(tmp_path: Path) -> tuple[Path, Path, Path, list[dict]]:
    source = tmp_path / "source" / "filtered.parquet"
    output = tmp_path / "prepared" / "pool.jsonl"
    manifest = tmp_path / "metadata" / "pool.manifest.json"
    _write_source(
        source,
        [
            _filtered_row("Question B", "2", 0),
            _filtered_row("Question A", "1", 1),
        ],
    )
    originals = [
        _original_row("Question B", "2", "解法 B"),
        _original_row("Question A", "1", "Solution A"),
    ]
    return source, output, manifest, originals


def _prepare_fixture(tmp_path: Path) -> tuple[Path, Path, Path, list[dict], dict]:
    source, output, manifest, originals = _fixture_inputs(tmp_path)
    result = preparation.prepare_pool(
        source,
        output,
        manifest,
        originals,
        expected_count=2,
    )
    return source, output, manifest, originals, result


def test_prepare_pool_writes_pinned_rows_in_source_order(tmp_path: Path) -> None:
    source, output, manifest_path, _, manifest = _prepare_fixture(tmp_path)

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert manifest["source_row_count"] == 2
    assert manifest["prepared_row_count"] == 2
    assert manifest["expected_count"] == 2
    assert manifest["dataset_name"] == preparation.DATASET_NAME
    assert manifest["dataset_revision"] == preparation.DATASET_REVISION
    assert manifest["source_sha256"] == sha256_file(source)
    assert manifest["prepared_jsonl_sha256"] == sha256_file(output)
    assert [record["source_row_index"] for record in records] == [0, 1]
    assert [record["prompt"] for record in records] == ["Question B", "Question A"]
    assert json.loads(manifest_path.read_text()) == manifest
    assert output.parent.is_dir()
    assert manifest_path.parent.is_dir()

    lines = output.read_text(encoding="utf-8").splitlines()
    assert "解法 B" in lines[0]
    assert lines == [
        json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        for record in records
    ]


def test_prepare_pool_reuses_a_fully_validated_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output, manifest_path, originals, first = _prepare_fixture(tmp_path)
    output_bytes = output.read_bytes()
    manifest_bytes = manifest_path.read_bytes()

    def unexpected_join(*args, **kwargs):
        raise AssertionError("a valid pinned cache should be reused")

    monkeypatch.setattr(preparation, "join_filtered_rows", unexpected_join)
    second = preparation.prepare_pool(
        source,
        output,
        manifest_path,
        originals,
        expected_count=2,
    )

    assert second == first
    assert output.read_bytes() == output_bytes
    assert manifest_path.read_bytes() == manifest_bytes


def test_prepare_pool_rejects_a_changed_source_without_overwriting_cache(
    tmp_path: Path,
) -> None:
    source, output, manifest_path, originals, _ = _prepare_fixture(tmp_path)
    output_bytes = output.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    _write_source(
        source,
        [
            _filtered_row("Changed B", "2", 0),
            _filtered_row("Question A", "1", 1),
        ],
    )

    with pytest.raises(ValueError, match="source_sha256"):
        preparation.prepare_pool(
            source,
            output,
            manifest_path,
            originals,
            expected_count=2,
        )

    assert output.read_bytes() == output_bytes
    assert manifest_path.read_bytes() == manifest_bytes
    assert not list(tmp_path.rglob("*.tmp"))


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("source_sha256", "bad-source-hash"),
        ("prepared_jsonl_sha256", "bad-output-hash"),
        ("source_row_count", 3),
        ("prepared_row_count", 3),
        ("expected_count", 3),
        ("dataset_name", "some/unpinned-dataset"),
        ("dataset_revision", "main"),
    ],
)
def test_prepare_pool_rejects_every_manifest_provenance_mismatch(
    tmp_path: Path, field: str, bad_value: object
) -> None:
    source, output, manifest_path, originals, manifest = _prepare_fixture(tmp_path)
    manifest[field] = bad_value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        preparation.prepare_pool(
            source,
            output,
            manifest_path,
            originals,
            expected_count=2,
        )


def test_prepare_pool_rejects_corrupted_output_hash(tmp_path: Path) -> None:
    source, output, manifest_path, originals, _ = _prepare_fixture(tmp_path)
    output.write_text(output.read_text() + "{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="prepared_jsonl_sha256"):
        preparation.prepare_pool(
            source,
            output,
            manifest_path,
            originals,
            expected_count=2,
        )


@pytest.mark.parametrize("existing", ["output", "manifest"])
def test_prepare_pool_rejects_partial_existing_cache(
    tmp_path: Path, existing: str
) -> None:
    source, output, manifest_path, originals = _fixture_inputs(tmp_path)
    path = output if existing == "output" else manifest_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="partial cache"):
        preparation.prepare_pool(
            source,
            output,
            manifest_path,
            originals,
            expected_count=2,
        )


def test_prepare_pool_leaves_no_targets_or_temps_when_join_fails(
    tmp_path: Path,
) -> None:
    source, output, manifest_path, originals = _fixture_inputs(tmp_path)
    originals[1]["r1_solution_1"] = ""

    with pytest.raises(ValueError, match="empty r1_solution_1"):
        preparation.prepare_pool(
            source,
            output,
            manifest_path,
            originals,
            expected_count=2,
        )

    assert not output.exists()
    assert not manifest_path.exists()
    assert not list(tmp_path.rglob("*.tmp"))


def test_prepare_pool_rolls_back_output_if_manifest_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output, manifest_path, originals = _fixture_inputs(tmp_path)
    original_replace = Path.replace
    replacements: list[tuple[Path, Path]] = []

    def fail_manifest_replace(self: Path, target: Path) -> Path:
        target = Path(target)
        replacements.append((self, target))
        if target == manifest_path:
            raise OSError("simulated manifest replace failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_manifest_replace)

    with pytest.raises(OSError, match="simulated manifest replace failure"):
        preparation.prepare_pool(
            source,
            output,
            manifest_path,
            originals,
            expected_count=2,
        )

    replacement_parents = [
        (temporary.parent, target.parent) for temporary, target in replacements
    ]
    assert replacement_parents == [
        (output.parent, output.parent),
        (manifest_path.parent, manifest_path.parent),
    ]
    assert all(temporary.suffix == ".tmp" for temporary, _ in replacements)
    assert not output.exists()
    assert not manifest_path.exists()
    assert not list(tmp_path.rglob("*.tmp"))


@pytest.mark.parametrize("failing_object_key", ["id", "manifest_version"])
def test_prepare_pool_cleans_temps_when_serialization_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failing_object_key: str,
) -> None:
    source, output, manifest_path, originals = _fixture_inputs(tmp_path)
    real_dumps = preparation.json.dumps

    def failing_dumps(value, *args, **kwargs):
        if isinstance(value, dict) and failing_object_key in value:
            raise TypeError(f"cannot serialize object containing {failing_object_key}")
        return real_dumps(value, *args, **kwargs)

    monkeypatch.setattr(preparation.json, "dumps", failing_dumps)

    with pytest.raises(TypeError, match="cannot serialize"):
        preparation.prepare_pool(
            source,
            output,
            manifest_path,
            originals,
            expected_count=2,
        )

    assert not output.exists()
    assert not manifest_path.exists()
    assert not list(tmp_path.rglob("*.tmp"))


def test_prepare_pool_rejects_missing_source_index_schema(tmp_path: Path) -> None:
    source, output, manifest_path, originals = _fixture_inputs(tmp_path)
    rows = [
        {key: value for key, value in row.items() if key != "extra_info"}
        for row in pq.read_table(source).to_pylist()
    ]
    _write_source(source, rows)

    with pytest.raises(ValueError, match="extra_info.index"):
        preparation.prepare_pool(
            source,
            output,
            manifest_path,
            originals,
            expected_count=2,
        )


def test_cli_loads_only_the_exact_pinned_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output, manifest_path, originals = _fixture_inputs(tmp_path)
    calls: list[tuple[str, dict]] = []

    def fake_load_dataset(name: str, **kwargs):
        calls.append((name, kwargs))
        return originals

    monkeypatch.setitem(
        sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset)
    )
    preparation.main(
        [
            "--source-parquet",
            str(source),
            "--output-jsonl",
            str(output),
            "--manifest",
            str(manifest_path),
            "--dataset-name",
            preparation.DATASET_NAME,
            "--dataset-revision",
            preparation.DATASET_REVISION,
            "--expected-count",
            "2",
        ]
    )

    assert calls == [
        (
            preparation.DATASET_NAME,
            {"split": "train", "revision": preparation.DATASET_REVISION},
        )
    ]


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--dataset-name", "untrusted/dataset"),
        ("--dataset-revision", "main"),
    ],
)
def test_cli_rejects_unpinned_dataset_inputs(
    tmp_path: Path, flag: str, value: str
) -> None:
    source, output, manifest_path, _ = _fixture_inputs(tmp_path)
    arguments = [
        "--source-parquet",
        str(source),
        "--output-jsonl",
        str(output),
        "--manifest",
        str(manifest_path),
        "--dataset-name",
        preparation.DATASET_NAME,
        "--dataset-revision",
        preparation.DATASET_REVISION,
        "--expected-count",
        "2",
    ]
    arguments[arguments.index(flag) + 1] = value

    with pytest.raises(SystemExit):
        preparation.main(arguments)
