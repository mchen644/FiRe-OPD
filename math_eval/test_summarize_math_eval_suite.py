import json
from pathlib import Path

import pytest

from math_eval import summarize_math_eval_suite as summary


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_summarize_eval_file_defines_pass1_passk_and_lengths(tmp_path: Path):
    path = tmp_path / "run.jsonl"
    _write(
        path,
        [
            {"acc_list": [True, False, False, False], "response_lengths": [1, 3, 5, 7]},
            {"acc_list": [True, True, False, False], "response_lengths": [2, 4, 6, 8]},
        ],
    )

    result = summary.summarize_eval_file(path, expected_samples=4)

    assert result == {
        "problems": 2,
        "samples": 8,
        "samples_per_problem": 4,
        "pass_at_1": pytest.approx(3 / 8),
        "pass_at_4": 1.0,
        "mean_response_length": 4.5,
        "median_response_length": 4.5,
    }


def test_summarize_eval_file_rejects_wrong_or_misaligned_sample_count(tmp_path: Path):
    wrong_n = tmp_path / "wrong_n.jsonl"
    _write(wrong_n, [{"acc_list": [True], "response_lengths": [1]}])
    with pytest.raises(ValueError, match="expected 4 samples"):
        summary.summarize_eval_file(wrong_n, expected_samples=4)

    misaligned = tmp_path / "misaligned.jsonl"
    _write(misaligned, [{"acc_list": [True, False], "response_lengths": [1]}])
    with pytest.raises(ValueError, match="acc_list/response_lengths length mismatch"):
        summary.summarize_eval_file(misaligned, expected_samples=2)


def test_suite_macro_is_unweighted_and_delta_is_candidate_minus_baseline(tmp_path: Path):
    datasets = ("small", "large")
    candidate = "candidate"
    baseline = "baseline"
    for dataset, candidate_acc, baseline_acc in (
        ("small", [True, False], [False, False]),
        ("large", [True, True], [True, False]),
    ):
        _write(
            tmp_path / dataset / f"{candidate}.jsonl",
            [{"acc_list": candidate_acc, "response_lengths": [2, 4]}],
        )
        _write(
            tmp_path / dataset / f"{baseline}.jsonl",
            [{"acc_list": baseline_acc, "response_lengths": [4, 8]}],
        )

    candidate_summary = summary.summarize_suite(tmp_path, candidate, datasets, 2)
    baseline_summary = summary.summarize_suite(tmp_path, baseline, datasets, 2)
    compared = summary.compare_runs(candidate_summary, {"group_success": baseline_summary})

    assert candidate_summary["macro"]["pass_at_1"] == 0.75
    assert candidate_summary["macro"]["pass_at_2"] == 1.0
    assert compared["deltas"]["group_success"]["pass_at_1"] == 0.5
    assert compared["deltas"]["group_success"]["mean_response_length"] == -3.0
