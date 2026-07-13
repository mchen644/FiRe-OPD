import json
from pathlib import Path

import pytest

from math_eval import summarize_math_eval_suite as summary


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _eval_row(problem, answer, acc_list, response_lengths):
    return {
        "problem": problem,
        "answer": answer,
        "acc_list": acc_list,
        "response_lengths": response_lengths,
    }


def test_summarize_eval_file_defines_pass1_passk_and_lengths(tmp_path: Path):
    path = tmp_path / "run.jsonl"
    _write(
        path,
        [
            _eval_row(
                "problem-1",
                "answer-1",
                [True, False, False, False],
                [1, 3, 5, 7],
            ),
            _eval_row(
                "problem-2",
                "answer-2",
                [True, True, False, False],
                [2, 4, 6, 8],
            ),
        ],
    )

    result = summary.summarize_eval_file(path, expected_samples=4)

    assert result == {
        "problems": 2,
        "samples": 8,
        "samples_per_problem": 4,
        "benchmark_identity_sha256": "2b1529d89f431a06a219a92f7d8bc646877ff8596cdaacdb57807ad130f79298",
        "pass_at_1": pytest.approx(3 / 8),
        "pass_at_4": 1.0,
        "mean_response_length": 4.5,
        "median_response_length": 4.5,
    }


def test_summarize_eval_file_rejects_wrong_or_misaligned_sample_count(tmp_path: Path):
    wrong_n = tmp_path / "wrong_n.jsonl"
    _write(wrong_n, [_eval_row("problem", "answer", [True], [1])])
    with pytest.raises(ValueError, match="expected 4 samples"):
        summary.summarize_eval_file(wrong_n, expected_samples=4)

    misaligned = tmp_path / "misaligned.jsonl"
    _write(
        misaligned,
        [_eval_row("problem", "answer", [True, False], [1])],
    )
    with pytest.raises(ValueError, match="acc_list/response_lengths length mismatch"):
        summary.summarize_eval_file(misaligned, expected_samples=2)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_summarize_eval_file_rejects_non_standard_json_constants(
    tmp_path: Path, constant: str
):
    path = tmp_path / "non_standard.jsonl"
    path.write_text(
        f'{{"problem":"p","answer":"a","acc_list":[true],'
        f'"response_lengths":[1],"unused":{constant}}}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="malformed JSON"):
        summary.summarize_eval_file(path, expected_samples=1)


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
            [
                _eval_row(
                    f"{dataset}-problem",
                    f"{dataset}-answer",
                    candidate_acc,
                    [2, 4],
                )
            ],
        )
        _write(
            tmp_path / dataset / f"{baseline}.jsonl",
            [
                _eval_row(
                    f"{dataset}-problem",
                    f"{dataset}-answer",
                    baseline_acc,
                    [4, 8],
                )
            ],
        )

    candidate_summary = summary.summarize_suite(tmp_path, candidate, datasets, 2)
    baseline_summary = summary.summarize_suite(tmp_path, baseline, datasets, 2)
    compared = summary.compare_runs(candidate_summary, {"group_success": baseline_summary})

    assert candidate_summary["macro"]["pass_at_1"] == 0.75
    assert candidate_summary["macro"]["pass_at_2"] == 1.0
    assert compared["deltas"]["group_success"]["pass_at_1"] == 0.5
    assert compared["deltas"]["group_success"]["mean_response_length"] == -3.0


@pytest.mark.parametrize(
    ("problem", "answer", "message"),
    [
        (None, "answer", "problem must be a nonempty string"),
        ("", "answer", "problem must be a nonempty string"),
        ("problem", None, "answer must be a nonempty string"),
        ("problem", "", "answer must be a nonempty string"),
    ],
)
def test_summarize_eval_file_requires_problem_and_answer_identity_fields(
    tmp_path: Path, problem, answer, message
):
    path = tmp_path / "invalid-identity.jsonl"
    _write(path, [_eval_row(problem, answer, [True], [1])])

    with pytest.raises(ValueError, match=message):
        summary.summarize_eval_file(path, expected_samples=1)


def test_compare_runs_rejects_per_dataset_problem_count_mismatch(tmp_path: Path):
    dataset = "benchmark"
    _write(
        tmp_path / dataset / "candidate.jsonl",
        [_eval_row("problem-1", "answer-1", [True], [1])],
    )
    _write(
        tmp_path / dataset / "baseline.jsonl",
        [
            _eval_row("problem-1", "answer-1", [False], [2]),
            _eval_row("problem-2", "answer-2", [False], [3]),
        ],
    )
    candidate = summary.summarize_suite(tmp_path, "candidate", (dataset,), 1)
    baseline = summary.summarize_suite(tmp_path, "baseline", (dataset,), 1)

    with pytest.raises(ValueError, match="problem count mismatch"):
        summary.compare_runs(candidate, {"baseline": baseline})


def test_compare_runs_rejects_same_count_different_benchmark_identity(tmp_path: Path):
    dataset = "benchmark"
    _write(
        tmp_path / dataset / "candidate.jsonl",
        [_eval_row("candidate-problem", "answer", [True], [1])],
    )
    _write(
        tmp_path / dataset / "baseline.jsonl",
        [_eval_row("baseline-problem", "answer", [False], [2])],
    )
    candidate = summary.summarize_suite(tmp_path, "candidate", (dataset,), 1)
    baseline = summary.summarize_suite(tmp_path, "baseline", (dataset,), 1)

    with pytest.raises(ValueError, match="benchmark identity mismatch"):
        summary.compare_runs(candidate, {"baseline": baseline})


def test_atomic_writer_rejects_dangling_output_symlink(tmp_path: Path):
    output_path = tmp_path / "report.json"
    output_path.symlink_to(tmp_path / "missing-target.json")

    with pytest.raises(ValueError, match="--output-json must be a file path"):
        summary._write_json_atomic(output_path, "{}\n")

    assert output_path.is_symlink()


def test_main_rejects_dangling_output_symlink_before_reading_inputs(tmp_path: Path):
    output_path = tmp_path / "report.json"
    output_path.symlink_to(tmp_path / "missing-target.json")

    with pytest.raises(ValueError, match="--output-json must be a file path"):
        summary.main(
            [
                "--candidate",
                "candidate",
                "--output-root",
                str(tmp_path / "missing-evaluations"),
                "--expected-samples",
                "1",
                "--output-json",
                str(output_path),
            ]
        )

    assert output_path.is_symlink()
