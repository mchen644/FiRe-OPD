import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from math_eval import validate_gradient_diverse_training_data as validation


SCHEMA = pa.schema(
    [
        pa.field("data_source", pa.string()),
        pa.field(
            "prompt",
            pa.list_(
                pa.struct(
                    [pa.field("content", pa.string()), pa.field("role", pa.string())]
                )
            ),
        ),
        pa.field("ability", pa.string()),
        pa.field(
            "reward_model",
            pa.struct(
                [pa.field("ground_truth", pa.string()), pa.field("style", pa.string())]
            ),
        ),
        pa.field(
            "extra_info",
            pa.struct([pa.field("index", pa.int64()), pa.field("split", pa.string())]),
        ),
    ],
    metadata={b"fixture": b"gradient-diverse-ablation"},
)


def _row(index: int, token_count: int | None = None) -> dict:
    count = token_count if token_count is not None else index + 3
    return {
        "data_source": "DeepMath-103K",
        "prompt": [{"role": "user", "content": f"tokens={count}; problem={index}"}],
        "ability": "math",
        "reward_model": {"ground_truth": str(index), "style": "rule"},
        "extra_info": {"index": index, "split": "train"},
    }


class FakeTokenizer:
    def __init__(self):
        self.calls: list[dict[str, object]] = []

    def apply_chat_template(self, conversations, **kwargs):
        self.calls.append(dict(kwargs))
        return [
            list(range(int(chat[0]["content"].split(";", 1)[0].split("=", 1)[1])))
            for chat in conversations
        ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifacts(tmp_path: Path, source_rows: list[dict], selected_indices: list[int]):
    source_path = tmp_path / "source.parquet"
    selected_path = tmp_path / "selected.parquet"
    ids_path = tmp_path / "selected_ids.jsonl"
    manifest_path = tmp_path / "manifest.json"
    source = pa.Table.from_pylist(source_rows, schema=SCHEMA)
    selected = source.take(pa.array(selected_indices, type=pa.int64()))
    pq.write_table(source, source_path)
    pq.write_table(selected, selected_path)
    id_rows = [
        {
            "eligible_position": source_index,
            "id": f"deepmath-level6-{source_index:06d}",
            "original_dataset_index": source_index + 100,
            "source_row_index": source_index,
        }
        for source_index in selected_indices
    ]
    ids_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in id_rows),
        encoding="utf-8",
    )
    id_sequence = "".join(row["id"] + "\n" for row in id_rows).encode("utf-8")
    manifest = {
        "manifest_version": 1,
        "source_parquet": str(source_path.resolve()),
        "source_sha256": _sha256(source_path),
        "source_row_count": len(source_rows),
        "eligible_row_count": len(source_rows),
        "selected_row_count": len(selected_indices),
        "output_parquet": str(selected_path.resolve()),
        "output_parquet_sha256": _sha256(selected_path),
        "selected_ids": str(ids_path.resolve()),
        "selected_ids_sha256": _sha256(ids_path),
        "selected_id_sequence_sha256": hashlib.sha256(id_sequence).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return source_path, selected_path, ids_path, manifest_path, manifest


def test_validate_training_artifact_accepts_exact_source_subset(tmp_path: Path):
    source, selected, ids, manifest_path, manifest = _artifacts(
        tmp_path, [_row(0), _row(1), _row(2)], [2, 0]
    )
    tokenizer = FakeTokenizer()

    report = validation.validate_training_artifact(
        selected,
        source,
        manifest_path,
        ids,
        tokenizer,
        expected_selected_sha256=manifest["output_parquet_sha256"],
        expected_source_sha256=manifest["source_sha256"],
        expected_rows=2,
        expected_source_rows=3,
        expected_eligible_rows=3,
        max_prompt_tokens=10,
    )

    assert report["selected_rows"] == 2
    assert report["unique_prompts"] == 2
    assert report["max_prompt_tokens"] == 5
    assert report["prompts_over_limit"] == 0
    assert report["schema_equal"] is True
    assert report["source_rows_equal"] is True
    assert tokenizer.calls == [
        {
            "tokenize": True,
            "add_generation_prompt": True,
            "padding": False,
            "truncation": False,
            "enable_thinking": False,
        }
    ]


def _validate_fixture(paths, manifest, tokenizer, **overrides):
    source, selected, ids, manifest_path = paths
    arguments = {
        "expected_selected_sha256": manifest["output_parquet_sha256"],
        "expected_source_sha256": manifest["source_sha256"],
        "expected_rows": manifest["selected_row_count"],
        "expected_source_rows": manifest["source_row_count"],
        "expected_eligible_rows": manifest["eligible_row_count"],
        "max_prompt_tokens": 10,
    }
    arguments.update(overrides)
    return validation.validate_training_artifact(
        selected, source, manifest_path, ids, tokenizer, **arguments
    )


def test_validator_rejects_pinned_selected_hash_mismatch(tmp_path: Path):
    source, selected, ids, manifest_path, manifest = _artifacts(
        tmp_path, [_row(0), _row(1)], [0]
    )
    with pytest.raises(ValueError, match="selected parquet SHA-256 mismatch"):
        _validate_fixture(
            (source, selected, ids, manifest_path),
            manifest,
            FakeTokenizer(),
            expected_selected_sha256="0" * 64,
        )


def test_validator_rejects_duplicate_exact_prompts(tmp_path: Path):
    duplicate = _row(0)
    second = _row(1)
    second["prompt"] = duplicate["prompt"]
    source, selected, ids, manifest_path, manifest = _artifacts(
        tmp_path, [duplicate, second], [0, 1]
    )
    with pytest.raises(ValueError, match="exact prompts must be unique"):
        _validate_fixture(
            (source, selected, ids, manifest_path), manifest, FakeTokenizer()
        )


def test_validator_rejects_prompt_over_training_limit(tmp_path: Path):
    source, selected, ids, manifest_path, manifest = _artifacts(
        tmp_path, [_row(0, token_count=11)], [0]
    )
    with pytest.raises(ValueError, match="prompt token limit exceeded"):
        _validate_fixture(
            (source, selected, ids, manifest_path), manifest, FakeTokenizer()
        )


def test_validator_rejects_selected_row_not_named_by_source_index(tmp_path: Path):
    source, selected, ids, manifest_path, manifest = _artifacts(
        tmp_path, [_row(0), _row(1)], [0]
    )
    pq.write_table(pa.Table.from_pylist([_row(1)], schema=SCHEMA), selected)
    manifest["output_parquet_sha256"] = _sha256(selected)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="selected parquet does not equal source.take"):
        _validate_fixture(
            (source, selected, ids, manifest_path), manifest, FakeTokenizer()
        )


def test_validator_rejects_manifest_selected_id_hash_mismatch(tmp_path: Path):
    source, selected, ids, manifest_path, manifest = _artifacts(
        tmp_path, [_row(0), _row(1)], [0]
    )
    manifest["selected_ids_sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="selected IDs SHA-256 mismatch"):
        _validate_fixture(
            (source, selected, ids, manifest_path), manifest, FakeTokenizer()
        )


def test_ensure_empty_checkpoint_dir_accepts_missing_and_empty_but_rejects_nonempty(
    tmp_path: Path,
):
    missing = tmp_path / "missing"
    empty = tmp_path / "empty"
    empty.mkdir()
    validation.ensure_empty_checkpoint_dir(missing)
    validation.ensure_empty_checkpoint_dir(empty)
    (empty / "global_step_1").mkdir()
    with pytest.raises(ValueError, match="checkpoint directory is non-empty"):
        validation.ensure_empty_checkpoint_dir(empty)
