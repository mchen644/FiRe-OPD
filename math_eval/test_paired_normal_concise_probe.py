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
    analyze_run,
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
    summarize_records,
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


def _response(
    *, length: int, correct: bool, max_tokens: int, parseable: bool = True
) -> dict:
    return {
        "text": r"Work. \\boxed{1}" if parseable else "unfinished work",
        "token_ids": list(range(length)),
        "length": length,
        "finish_reason": "length" if length == max_tokens else "stop",
        "stop_reason": None,
        "boxed_answer": "1" if parseable else None,
        "parseable": parseable,
        "correct": correct,
        "max_tokens": max_tokens,
        "cap_hit": length == max_tokens,
        "rendered_prompt_sha256": "a" * 64,
    }


def _summary_records() -> list[dict]:
    records = [
        {
            "sample_ordinal": 0,
            "source_row_index": 10,
            "question_id": 110,
            "request_seed": 52,
            "model_label": "test",
            "normal": _response(length=100, correct=True, max_tokens=16384),
            "concise": _response(length=20, correct=True, max_tokens=50),
            "quadrant": "compression_safe",
            "relaxed_concise": None,
            "compression_sensitive_class": None,
        },
        {
            "sample_ordinal": 1,
            "source_row_index": 11,
            "question_id": 111,
            "request_seed": 53,
            "model_label": "test",
            "normal": _response(length=100, correct=True, max_tokens=16384),
            "concise": _response(length=50, correct=False, max_tokens=50),
            "quadrant": "compression_sensitive",
            "relaxed_concise": _response(length=80, correct=True, max_tokens=100),
            "compression_sensitive_class": "budget_limited_recovered",
        },
        {
            "sample_ordinal": 2,
            "source_row_index": 12,
            "question_id": 112,
            "request_seed": 54,
            "model_label": "test",
            "normal": _response(length=120, correct=False, max_tokens=16384),
            "concise": _response(length=30, correct=True, max_tokens=60),
            "quadrant": "concise_rescued",
            "relaxed_concise": None,
            "compression_sensitive_class": None,
        },
        {
            "sample_ordinal": 3,
            "source_row_index": 13,
            "question_id": 113,
            "request_seed": 55,
            "model_label": "test",
            "normal": _response(length=80, correct=False, max_tokens=16384),
            "concise": _response(
                length=40, correct=False, max_tokens=40, parseable=False
            ),
            "quadrant": "both_wrong",
            "relaxed_concise": None,
            "compression_sensitive_class": None,
        },
    ]
    return records


def test_summarize_records_recomputes_quadrants_lengths_and_failure_classes() -> None:
    summary = summarize_records(_summary_records(), expected_count=4)

    assert summary["model_label"] == "test"
    assert summary["record_count"] == 4
    assert summary["quadrant_counts"] == {
        "compression_safe": 1,
        "compression_sensitive": 1,
        "concise_rescued": 1,
        "both_wrong": 1,
    }
    assert summary["normal_accuracy"] == pytest.approx(0.5)
    assert summary["concise_accuracy"] == pytest.approx(0.5)
    assert summary["concise_minus_normal_accuracy"] == pytest.approx(0.0)
    assert summary["normal_length_mean"] == pytest.approx(100.0)
    assert summary["normal_length_median"] == pytest.approx(100.0)
    assert summary["concise_length_mean"] == pytest.approx(35.0)
    assert summary["concise_length_median"] == pytest.approx(35.0)
    assert summary["concise_to_normal_ratio_mean"] == pytest.approx(0.3625)
    assert summary["concise_cap_hit_count"] == 2
    assert summary["concise_parse_failure_count"] == 1
    assert summary["compression_sensitive_class_counts"] == {
        "budget_limited_recovered": 1,
        "budget_limited_unrecovered": 0,
        "prompt_or_sampling_failure": 0,
    }


def test_summarize_records_rejects_tampered_route_or_count() -> None:
    records = _summary_records()
    records[0]["quadrant"] = "both_wrong"
    with pytest.raises(ValueError, match="quadrant"):
        summarize_records(records, expected_count=4)
    with pytest.raises(ValueError, match="exactly 5"):
        summarize_records(_summary_records(), expected_count=5)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _complete_analyzable_run(run_dir: Path) -> None:
    sample = []
    for ordinal, record in enumerate(_summary_records()):
        sample.append(
            {
                "sample_ordinal": ordinal,
                "source_row_index": record["source_row_index"],
                "question_id": record["question_id"],
                "request_seed": record["request_seed"],
                "prompt": [{"role": "user", "content": f"Question {ordinal}"}],
                "ground_truth": "1",
            }
        )
    _write_jsonl(run_dir / "sample.jsonl", sample)
    manifest = {
        "artifact_type": "paired_normal_concise_compression_sensitivity_probe",
        "dataset_path": "/dataset.parquet",
        "dataset_sha256": "d" * 64,
        "sample_file": str((run_dir / "sample.jsonl").resolve()),
        "sample_sha256": hashlib.sha256((run_dir / "sample.jsonl").read_bytes()).hexdigest(),
        "sample_size": 4,
        "seed": 42,
        "source_commit": "abc123",
        "protocol": {"normal_max_tokens": 16384, "concise_cap_ratio": 0.5},
        "artifacts": {},
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    for label in ("base", "adaptive_step50"):
        records = copy.deepcopy(_summary_records())
        for record, sample_row in zip(records, sample, strict=True):
            record["model_label"] = label
            record["model_path"] = f"/{label}"
            record["prompt"] = sample_row["prompt"]
            record["ground_truth"] = sample_row["ground_truth"]
            record["normal_prompt"] = sample_row["prompt"]
            record["concise_prompt"] = [
                {"role": "user", "content": f"Question {record['sample_ordinal']} concise"}
            ]
        _write_jsonl(run_dir / label / "records.jsonl", records)


def test_analyze_run_writes_model_reports_comparison_and_hash_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _complete_analyzable_run(run_dir)

    comparison = analyze_run(run_dir=run_dir, expected_count=4)

    assert comparison["record_count_per_model"] == 4
    assert comparison["quadrant_count_delta_adaptive_minus_base"] == {
        "compression_safe": 0,
        "compression_sensitive": 0,
        "concise_rescued": 0,
        "both_wrong": 0,
    }
    for relative in (
        "base/summary.json",
        "base/compression_sensitive_cases.md",
        "adaptive_step50/summary.json",
        "adaptive_step50/compression_sensitive_cases.md",
        "comparison.json",
        "comparison.md",
    ):
        assert (run_dir / relative).is_file()
    case_text = (run_dir / "base/compression_sensitive_cases.md").read_text()
    assert "budget_limited_recovered" in case_text
    assert "source_row_index: 11" in case_text

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["artifacts"]["base/records.jsonl"] == hashlib.sha256(
        (run_dir / "base/records.jsonl").read_bytes()
    ).hexdigest()
    assert manifest["artifacts"]["comparison.json"] == hashlib.sha256(
        (run_dir / "comparison.json").read_bytes()
    ).hexdigest()


def test_analyze_run_refuses_to_overwrite_reports(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _complete_analyzable_run(run_dir)
    analyze_run(run_dir=run_dir, expected_count=4)

    with pytest.raises(FileExistsError, match="analysis output"):
        analyze_run(run_dir=run_dir, expected_count=4)
