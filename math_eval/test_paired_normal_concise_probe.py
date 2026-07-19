from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from math_eval.paired_normal_concise_probe import (
    classify_compression_sensitive,
    concise_cap,
    concise_messages,
    ensure_generation_destinations,
    normal_messages,
    normalize_request_output,
    prepare_sample,
    quadrant,
    read_jsonl,
    select_probe_rows,
    validate_relaxed_prefix,
)


def _rows(count: int = 8) -> list[dict]:
    return [
        {
            "data_source": "DeepMath-103K",
            "prompt": [
                {
                    "role": "user",
                    "content": (
                        f"Compute {index}+1.\n"
                        "Please reason step by step, and put your final answer within \\boxed{}."
                    ),
                }
            ],
            "ability": "math",
            "reward_model": {"ground_truth": str(index + 1), "style": "rule"},
            "extra_info": {"index": 100 + index, "split": "train"},
        }
        for index in range(count)
    ]


def test_select_probe_rows_is_exact_distinct_and_repeatable() -> None:
    rows = _rows()

    first = select_probe_rows(rows, sample_size=4, seed=42)
    repeated = select_probe_rows(rows, sample_size=4, seed=42)
    changed = select_probe_rows(rows, sample_size=4, seed=43)

    assert first == repeated
    assert len(first) == 4
    assert len({row["source_row_index"] for row in first}) == 4
    assert len({row["question_id"] for row in first}) == 4
    assert [row["source_row_index"] for row in first] != [
        row["source_row_index"] for row in changed
    ]
    assert all(row["request_seed"] == 42 + row["source_row_index"] for row in first)
    assert [row["sample_ordinal"] for row in first] == list(range(4))


def test_select_probe_rows_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="sample_size"):
        select_probe_rows(_rows(), sample_size=9, seed=42)

    duplicate = _rows()
    duplicate[1]["extra_info"]["index"] = duplicate[0]["extra_info"]["index"]
    with pytest.raises(ValueError, match="distinct question"):
        select_probe_rows(duplicate, sample_size=4, seed=42)

    malformed_prompt = _rows()
    malformed_prompt[0]["prompt"] = [{"role": "user", "content": ""}]
    with pytest.raises(ValueError, match="prompt"):
        select_probe_rows(malformed_prompt, sample_size=4, seed=42)

    missing_truth = _rows()
    missing_truth[0]["reward_model"]["ground_truth"] = ""
    with pytest.raises(ValueError, match="ground truth"):
        select_probe_rows(missing_truth, sample_size=4, seed=42)


def test_normal_and_concise_messages_preserve_question_but_change_instruction() -> None:
    row = select_probe_rows(_rows(), sample_size=1, seed=42)[0]
    original_prompt = copy.deepcopy(row["prompt"])

    assert normal_messages(row) == original_prompt
    concise = concise_messages(row)

    assert row["prompt"] == original_prompt
    assert len(concise) == 1
    assert concise[0]["role"] == "user"
    assert "Compute" in concise[0]["content"]
    assert "Solve concisely. Avoid unnecessary explanation." in concise[0]["content"]
    assert "Please reason step by step" not in concise[0]["content"]


def test_concise_cap_and_quadrants_are_exact() -> None:
    assert concise_cap(1) == 1
    assert concise_cap(2) == 1
    assert concise_cap(9) == 4
    with pytest.raises(ValueError, match="positive integer"):
        concise_cap(0)
    with pytest.raises(ValueError, match="positive integer"):
        concise_cap(True)

    assert quadrant(True, True) == "compression_safe"
    assert quadrant(True, False) == "compression_sensitive"
    assert quadrant(False, True) == "concise_rescued"
    assert quadrant(False, False) == "both_wrong"
    with pytest.raises(TypeError, match="boolean"):
        quadrant(1, False)


def test_validate_relaxed_prefix_accepts_only_exact_token_prefix() -> None:
    validate_relaxed_prefix([1, 2], [1, 2, 3])
    validate_relaxed_prefix([1, 2], [1, 2])

    with pytest.raises(ValueError, match="prefix"):
        validate_relaxed_prefix([1, 9], [1, 2, 3])
    with pytest.raises(ValueError, match="nonempty"):
        validate_relaxed_prefix([], [1])


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        (
            {
                "quadrant": "compression_sensitive",
                "concise": {"cap_hit": True, "correct": False, "token_ids": [1, 2]},
                "relaxed_concise": {"correct": True, "token_ids": [1, 2, 3]},
            },
            "budget_limited_recovered",
        ),
        (
            {
                "quadrant": "compression_sensitive",
                "concise": {"cap_hit": True, "correct": False, "token_ids": [1, 2]},
                "relaxed_concise": {"correct": False, "token_ids": [1, 2, 4]},
            },
            "budget_limited_unrecovered",
        ),
        (
            {
                "quadrant": "compression_sensitive",
                "concise": {"cap_hit": False, "correct": False, "token_ids": [1]},
                "relaxed_concise": None,
            },
            "prompt_or_sampling_failure",
        ),
        (
            {
                "quadrant": "compression_safe",
                "concise": {"cap_hit": False, "correct": True, "token_ids": [1]},
                "relaxed_concise": None,
            },
            None,
        ),
    ],
)
def test_classify_compression_sensitive(record: dict, expected: str | None) -> None:
    assert classify_compression_sensitive(record) == expected


def test_classify_compression_sensitive_requires_relaxed_cap_hit_result() -> None:
    record = {
        "quadrant": "compression_sensitive",
        "concise": {"cap_hit": True, "correct": False, "token_ids": [1]},
        "relaxed_concise": None,
    }
    with pytest.raises(ValueError, match="relaxed"):
        classify_compression_sensitive(record)


def _write_parquet(path: Path, rows: list[dict] | None = None) -> str:
    pq.write_table(pa.Table.from_pylist(_rows() if rows is None else rows), path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_prepare_sample_writes_immutable_sample_and_manifest(tmp_path: Path) -> None:
    dataset = tmp_path / "train.parquet"
    dataset_sha = _write_parquet(dataset)
    sample_file = tmp_path / "sample.jsonl"
    manifest_file = tmp_path / "manifest.json"

    manifest = prepare_sample(
        dataset_path=dataset,
        sample_file=sample_file,
        manifest_file=manifest_file,
        sample_size=4,
        seed=42,
        source_commit="abc123",
        expected_dataset_sha256=dataset_sha,
    )

    sample = read_jsonl(sample_file)
    persisted_manifest = json.loads(manifest_file.read_text())
    assert len(sample) == 4
    assert manifest == persisted_manifest
    assert manifest["dataset_sha256"] == dataset_sha
    assert manifest["source_commit"] == "abc123"
    assert manifest["sample_size"] == 4
    assert manifest["sample_sha256"] == hashlib.sha256(sample_file.read_bytes()).hexdigest()
    assert manifest["protocol"]["normal_max_tokens"] == 16384
    assert manifest["protocol"]["concise_cap_ratio"] == 0.5

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare_sample(
            dataset_path=dataset,
            sample_file=sample_file,
            manifest_file=manifest_file,
            sample_size=4,
            seed=42,
            source_commit="abc123",
            expected_dataset_sha256=dataset_sha,
        )


def test_prepare_sample_rejects_wrong_dataset_hash_without_outputs(tmp_path: Path) -> None:
    dataset = tmp_path / "train.parquet"
    _write_parquet(dataset)
    sample_file = tmp_path / "sample.jsonl"
    manifest_file = tmp_path / "manifest.json"

    with pytest.raises(ValueError, match="dataset SHA256"):
        prepare_sample(
            dataset_path=dataset,
            sample_file=sample_file,
            manifest_file=manifest_file,
            sample_size=4,
            seed=42,
            source_commit="abc123",
            expected_dataset_sha256="0" * 64,
        )
    assert not sample_file.exists()
    assert not manifest_file.exists()


def test_normalize_request_output_uses_generated_ids_and_training_math_reward() -> None:
    request_output = SimpleNamespace(
        outputs=[
            SimpleNamespace(
                text=r"Reasoning. Therefore \\boxed{2}",
                token_ids=[11, 12, 13],
                finish_reason="stop",
                stop_reason=151645,
            )
        ]
    )

    response = normalize_request_output(
        request_output,
        ground_truth="2",
        max_tokens=7,
        rendered_prompt="rendered prompt",
    )

    assert response == {
        "text": r"Reasoning. Therefore \\boxed{2}",
        "token_ids": [11, 12, 13],
        "length": 3,
        "finish_reason": "stop",
        "stop_reason": 151645,
        "boxed_answer": "2",
        "parseable": True,
        "correct": True,
        "max_tokens": 7,
        "cap_hit": False,
        "rendered_prompt_sha256": hashlib.sha256(b"rendered prompt").hexdigest(),
    }


def test_normalize_request_output_rejects_multiple_or_over_cap_outputs() -> None:
    completion = SimpleNamespace(
        text=r"\\boxed{1}", token_ids=[1, 2], finish_reason="length", stop_reason=None
    )
    with pytest.raises(ValueError, match="exactly one"):
        normalize_request_output(
            SimpleNamespace(outputs=[completion, completion]),
            ground_truth="1",
            max_tokens=2,
            rendered_prompt="x",
        )
    with pytest.raises(ValueError, match="exceeds"):
        normalize_request_output(
            SimpleNamespace(outputs=[completion]),
            ground_truth="1",
            max_tokens=1,
            rendered_prompt="x",
        )


def test_generation_destinations_fail_closed(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    output = tmp_path / "output"
    ensure_generation_destinations(model_path=model, output_dir=output)

    output.mkdir()
    with pytest.raises(FileExistsError, match="output directory"):
        ensure_generation_destinations(model_path=model, output_dir=output)
    with pytest.raises(FileNotFoundError, match="model directory"):
        ensure_generation_destinations(model_path=tmp_path / "missing", output_dir=tmp_path / "fresh")
