"""Strict summaries for the canonical eight-dataset math evaluation suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path


DATASETS = (
    "aime24",
    "aime25",
    "hmmt25_feb",
    "hmmt25_nov",
    "math500",
    "minervamath",
    "olympiadbench",
    "amc2023",
)


def metric_names(expected_samples: int) -> tuple[str, str, str, str]:
    return (
        "pass_at_1",
        f"pass_at_{expected_samples}",
        "mean_response_length",
        "median_response_length",
    )


def _require_positive_sample_count(expected_samples: int) -> None:
    if (
        isinstance(expected_samples, bool)
        or not isinstance(expected_samples, int)
        or expected_samples <= 0
    ):
        raise ValueError("expected_samples must be a positive integer")


def _reject_non_standard_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _load_eval_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(f"evaluation file is missing or not a regular file: {path}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"unable to read evaluation file {path}: {error}") from error

    rows = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(
                line, parse_constant=_reject_non_standard_json_constant
            )
        except (json.JSONDecodeError, ValueError) as error:
            reason = (
                error.msg if isinstance(error, json.JSONDecodeError) else str(error)
            )
            raise ValueError(
                f"malformed JSON in {path} on line {line_number}: {reason}"
            ) from error
        if not isinstance(row, dict):
            raise ValueError(
                f"evaluation row {line_number} in {path} must be a JSON object"
            )
        rows.append(row)

    if not rows:
        raise ValueError(f"evaluation file is empty: {path}")
    return rows


def summarize_eval_file(
    path: Path, expected_samples: int
) -> dict[str, float | int | str]:
    """Validate one evaluation JSONL file and summarize all sampled responses."""
    _require_positive_sample_count(expected_samples)
    rows = _load_eval_rows(path)
    per_problem_accuracy: list[list[bool]] = []
    flat_accuracy: list[bool] = []
    flat_lengths: list[int | float] = []
    benchmark_identity: list[tuple[str, str]] = []

    for row_number, row in enumerate(rows, start=1):
        problem = row.get("problem")
        answer = row.get("answer")
        if not isinstance(problem, str) or not problem:
            raise ValueError(f"row {row_number} problem must be a nonempty string")
        if not isinstance(answer, str) or not answer:
            raise ValueError(f"row {row_number} answer must be a nonempty string")
        accuracy = row.get("acc_list")
        response_lengths = row.get("response_lengths")
        if not isinstance(accuracy, list):
            raise ValueError(f"row {row_number} acc_list must be a list")
        if not isinstance(response_lengths, list):
            raise ValueError(f"row {row_number} response_lengths must be a list")
        if len(accuracy) != len(response_lengths):
            raise ValueError(
                f"row {row_number} acc_list/response_lengths length mismatch: "
                f"{len(accuracy)} != {len(response_lengths)}"
            )
        if len(accuracy) != expected_samples:
            raise ValueError(
                f"row {row_number} expected {expected_samples} samples, "
                f"got {len(accuracy)}"
            )
        if any(not isinstance(value, bool) for value in accuracy):
            raise ValueError(f"row {row_number} acc_list entries must be boolean")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in response_lengths
        ):
            raise ValueError(
                f"row {row_number} response_lengths entries must be finite, "
                "non-negative numbers"
            )

        per_problem_accuracy.append(accuracy)
        flat_accuracy.extend(accuracy)
        flat_lengths.extend(response_lengths)
        benchmark_identity.append((problem, answer))

    problem_count = len(per_problem_accuracy)
    pass_at_k_name = f"pass_at_{expected_samples}"
    identity_payload = json.dumps(
        benchmark_identity,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "problems": problem_count,
        "samples": len(flat_accuracy),
        "samples_per_problem": expected_samples,
        "benchmark_identity_sha256": hashlib.sha256(identity_payload).hexdigest(),
        "pass_at_1": sum(flat_accuracy) / len(flat_accuracy),
        pass_at_k_name: sum(any(row) for row in per_problem_accuracy)
        / problem_count,
        "mean_response_length": statistics.fmean(flat_lengths),
        "median_response_length": statistics.median(flat_lengths),
    }


def summarize_suite(
    output_root: Path,
    model_name: str,
    datasets: Sequence[str],
    expected_samples: int,
) -> dict[str, object]:
    """Summarize each requested dataset and its unweighted macro metrics."""
    _require_positive_sample_count(expected_samples)
    if isinstance(datasets, (str, bytes)):
        raise ValueError("datasets must be a sequence of dataset names")
    dataset_order = list(datasets)
    if not dataset_order:
        raise ValueError("datasets must contain at least one dataset name")
    if any(not isinstance(name, str) or not name for name in dataset_order):
        raise ValueError("dataset names must be nonempty strings")
    if len(set(dataset_order)) != len(dataset_order):
        raise ValueError("dataset names must be unique")

    per_dataset = {
        name: summarize_eval_file(
            output_root / name / f"{model_name}.jsonl", expected_samples
        )
        for name in dataset_order
    }
    return {
        "model_name": model_name,
        "expected_samples": expected_samples,
        "dataset_order": dataset_order,
        "datasets": per_dataset,
        "macro": {
            metric: statistics.fmean(
                per_dataset[name][metric] for name in dataset_order
            )
            for metric in metric_names(expected_samples)
        },
    }


def compare_runs(
    candidate: dict, baselines: Mapping[str, dict]
) -> dict[str, object]:
    """Return full run summaries and candidate-minus-baseline metric deltas."""
    try:
        expected_samples = candidate["expected_samples"]
        dataset_order = candidate["dataset_order"]
        candidate_datasets = candidate["datasets"]
        candidate_macro = candidate["macro"]
    except KeyError as error:
        raise ValueError(f"candidate summary is missing {error.args[0]!r}") from error

    _require_positive_sample_count(expected_samples)
    if not isinstance(dataset_order, list) or any(
        not isinstance(name, str) for name in dataset_order
    ):
        raise ValueError("candidate dataset_order must be a list of strings")
    metrics = metric_names(expected_samples)
    baseline_summaries = dict(baselines)
    deltas: dict[str, dict[str, object]] = {}

    for label, baseline in baseline_summaries.items():
        if baseline.get("dataset_order") != dataset_order:
            raise ValueError(f"baseline {label!r} dataset order mismatch")
        if baseline.get("expected_samples") != expected_samples:
            raise ValueError(f"baseline {label!r} expected sample count mismatch")
        try:
            baseline_datasets = baseline["datasets"]
            baseline_macro = baseline["macro"]
            for dataset in dataset_order:
                candidate_dataset = candidate_datasets[dataset]
                baseline_dataset = baseline_datasets[dataset]
                if candidate_dataset["problems"] != baseline_dataset["problems"]:
                    raise ValueError(
                        f"baseline {label!r} dataset {dataset!r} problem count mismatch"
                    )
                if (
                    candidate_dataset["benchmark_identity_sha256"]
                    != baseline_dataset["benchmark_identity_sha256"]
                ):
                    raise ValueError(
                        f"baseline {label!r} dataset {dataset!r} "
                        "benchmark identity mismatch"
                    )
            per_dataset_deltas = {
                dataset: {
                    metric: candidate_datasets[dataset][metric]
                    - baseline_datasets[dataset][metric]
                    for metric in metrics
                }
                for dataset in dataset_order
            }
            macro_deltas = {
                metric: candidate_macro[metric] - baseline_macro[metric]
                for metric in metrics
            }
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"candidate/baseline {label!r} summary shape mismatch"
            ) from error
        deltas[label] = {**macro_deltas, "datasets": per_dataset_deltas}

    return {
        "candidate": candidate,
        "baselines": baseline_summaries,
        "deltas": deltas,
    }


def _parse_baselines(values: Sequence[str]) -> dict[str, str]:
    parsed = {}
    for value in values:
        label, separator, model_name = value.partition("=")
        if not separator or not label or not model_name:
            raise ValueError("--baseline must use LABEL=MODEL_NAME")
        if label in parsed:
            raise ValueError(f"duplicate --baseline label: {label}")
        parsed[label] = model_name
    return parsed


def _write_json_atomic(path: Path, payload: str) -> None:
    if (path.exists() or path.is_symlink()) and not path.is_file():
        raise ValueError(f"--output-json must be a file path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(payload)
            temporary_path = Path(temporary.name)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize and compare strict eight-dataset math evaluations."
    )
    parser.add_argument("--candidate", required=True)
    parser.add_argument(
        "--baseline",
        action="append",
        default=[],
        metavar="LABEL=MODEL_NAME",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    baseline_models = _parse_baselines(args.baseline)
    if (
        args.output_json.exists() or args.output_json.is_symlink()
    ) and not args.output_json.is_file():
        raise ValueError(f"--output-json must be a file path: {args.output_json}")

    candidate = summarize_suite(
        args.output_root, args.candidate, DATASETS, args.expected_samples
    )
    baselines = {
        label: summarize_suite(
            args.output_root, model_name, DATASETS, args.expected_samples
        )
        for label, model_name in baseline_models.items()
    }
    report = compare_runs(candidate, baselines)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    _write_json_atomic(args.output_json, payload)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
