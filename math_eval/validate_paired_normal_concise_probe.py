"""Independently validate paired normal/concise probe artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

_ROUTE_NAMES = (
    "compression_safe",
    "compression_sensitive",
    "concise_rescued",
    "both_wrong",
)
_CLASS_NAMES = (
    "budget_limited_recovered",
    "budget_limited_unrecovered",
    "prompt_or_sampling_failure",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    values: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path} line {line_number} is not a JSON object")
            values.append(value)
    return values


def _route(normal_correct: bool, concise_correct: bool) -> str:
    if normal_correct and concise_correct:
        return "compression_safe"
    if normal_correct and not concise_correct:
        return "compression_sensitive"
    if not normal_correct and concise_correct:
        return "concise_rescued"
    return "both_wrong"


def _validate_response(response: Mapping, *, name: str) -> tuple[int, bool, bool]:
    token_ids = response.get("token_ids")
    if not isinstance(token_ids, list) or not token_ids:
        raise ValueError(f"{name} token IDs must be nonempty")
    if not all(isinstance(token_id, int) and not isinstance(token_id, bool) for token_id in token_ids):
        raise ValueError(f"{name} token IDs must be integers")
    length = response.get("length")
    if isinstance(length, bool) or not isinstance(length, int) or length != len(token_ids):
        raise ValueError(f"{name} length does not match token IDs")
    max_tokens = response.get("max_tokens")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError(f"{name} max_tokens must be positive")
    if length > max_tokens:
        raise ValueError(f"{name} length exceeds max_tokens")
    if response.get("cap_hit") is not (length == max_tokens):
        raise ValueError(f"{name} cap-hit flag is inconsistent")
    correct = response.get("correct")
    parseable = response.get("parseable")
    if not isinstance(correct, bool) or not isinstance(parseable, bool):
        raise ValueError(f"{name} correctness and parseability must be boolean")
    if parseable is not (response.get("boxed_answer") is not None):
        raise ValueError(f"{name} parseability disagrees with boxed answer")
    return length, correct, parseable


def _independent_summary(records: Sequence[Mapping], *, label: str, expected_count: int) -> dict:
    route_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    cap_hits: Counter[str] = Counter()
    normal_lengths: list[int] = []
    concise_lengths: list[int] = []
    ratios: list[float] = []
    normal_correct_count = 0
    concise_correct_count = 0
    normal_parse_failures = 0
    concise_parse_failures = 0

    if len(records) != expected_count:
        raise ValueError(f"{label} records must contain exactly {expected_count} rows")
    for ordinal, record in enumerate(records):
        if record.get("sample_ordinal") != ordinal:
            raise ValueError(f"{label} record ordinal mismatch at row {ordinal}")
        if record.get("model_label") != label:
            raise ValueError(f"{label} record has an inconsistent model label")
        normal = record.get("normal")
        concise = record.get("concise")
        if not isinstance(normal, Mapping) or not isinstance(concise, Mapping):
            raise ValueError(f"{label} record {ordinal} lacks response objects")
        normal_length, normal_correct, normal_parseable = _validate_response(
            normal, name=f"{label} normal"
        )
        concise_length, concise_correct, concise_parseable = _validate_response(
            concise, name=f"{label} concise"
        )
        if normal.get("max_tokens") != 16384:
            raise ValueError(f"{label} normal max_tokens must equal 16384")
        expected_cap = max(1, normal_length // 2)
        if concise.get("max_tokens") != expected_cap:
            raise ValueError(f"{label} concise cap does not match half the normal length")
        expected_route = _route(normal_correct, concise_correct)
        if record.get("quadrant") != expected_route:
            raise ValueError(f"{label} persisted quadrant is incorrect")

        relaxed = record.get("relaxed_concise")
        if expected_route == "compression_sensitive" and concise.get("cap_hit"):
            if not isinstance(relaxed, Mapping):
                raise ValueError(f"{label} cap-hit sensitive row lacks relaxed response")
            relaxed_length, relaxed_correct, _ = _validate_response(
                relaxed, name=f"{label} relaxed concise"
            )
            if relaxed.get("max_tokens") != normal_length:
                raise ValueError(f"{label} relaxed max_tokens does not match normal length")
            if relaxed_length < concise_length or relaxed["token_ids"][:concise_length] != concise["token_ids"]:
                raise ValueError(f"{label} relaxed response does not preserve capped prefix")
            expected_class = (
                "budget_limited_recovered"
                if relaxed_correct
                else "budget_limited_unrecovered"
            )
        elif expected_route == "compression_sensitive":
            if relaxed is not None:
                raise ValueError(f"{label} natural-stop sensitive row unexpectedly has relaxed output")
            expected_class = "prompt_or_sampling_failure"
        else:
            if relaxed is not None:
                raise ValueError(f"{label} nonsensitive row unexpectedly has relaxed output")
            expected_class = None
        if record.get("compression_sensitive_class") != expected_class:
            raise ValueError(f"{label} compression-sensitive class is incorrect")

        route_counts[expected_route] += 1
        if expected_class is not None:
            class_counts[expected_class] += 1
        if concise.get("cap_hit"):
            cap_hits[expected_route] += 1
        normal_lengths.append(normal_length)
        concise_lengths.append(concise_length)
        ratios.append(concise_length / normal_length)
        normal_correct_count += int(normal_correct)
        concise_correct_count += int(concise_correct)
        normal_parse_failures += int(not normal_parseable)
        concise_parse_failures += int(not concise_parseable)

    total_cap_hits = sum(cap_hits.values())
    return {
        "model_label": label,
        "record_count": expected_count,
        "quadrant_counts": {name: route_counts[name] for name in _ROUTE_NAMES},
        "quadrant_ratios": {name: route_counts[name] / expected_count for name in _ROUTE_NAMES},
        "normal_accuracy": normal_correct_count / expected_count,
        "concise_accuracy": concise_correct_count / expected_count,
        "concise_minus_normal_accuracy": (
            concise_correct_count - normal_correct_count
        )
        / expected_count,
        "normal_length_mean": statistics.fmean(normal_lengths),
        "normal_length_median": statistics.median(normal_lengths),
        "concise_length_mean": statistics.fmean(concise_lengths),
        "concise_length_median": statistics.median(concise_lengths),
        "concise_to_normal_ratio_mean": statistics.fmean(ratios),
        "concise_to_normal_ratio_median": statistics.median(ratios),
        "concise_cap_hit_count": total_cap_hits,
        "concise_cap_hit_ratio": total_cap_hits / expected_count,
        "concise_cap_hits_by_quadrant": {name: cap_hits[name] for name in _ROUTE_NAMES},
        "normal_parse_failure_count": normal_parse_failures,
        "concise_parse_failure_count": concise_parse_failures,
        "compression_sensitive_class_counts": {
            name: class_counts[name] for name in _CLASS_NAMES
        },
    }


def _assert_equal(actual, expected, *, path: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise ValueError(f"{path} keys differ from independent recomputation")
        for key in expected:
            _assert_equal(actual[key], expected[key], path=f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{path} list differs from independent recomputation")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=True)):
            _assert_equal(actual_item, expected_item, path=f"{path}[{index}]")
        return
    if isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not math.isclose(
            float(actual), expected, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"{path} differs from independent recomputation")
        return
    if actual != expected:
        raise ValueError(f"{path} differs from independent recomputation")


def validate_run(*, run_dir: str | Path, expected_count: int) -> dict:
    """Validate all records, summaries, comparisons, identities, and hashes."""

    root = Path(run_dir)
    manifest_path = root / "manifest.json"
    sample_path = root / "sample.jsonl"
    if not manifest_path.is_file() or not sample_path.is_file():
        raise FileNotFoundError("run directory lacks manifest.json or sample.jsonl")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    if manifest.get("sample_size") != expected_count:
        raise ValueError("manifest sample size does not match expected count")
    if manifest.get("sample_sha256") != _sha256(sample_path):
        raise ValueError("sample SHA256 does not match manifest")

    sample = _read_jsonl(sample_path)
    if len(sample) != expected_count:
        raise ValueError(f"sample must contain exactly {expected_count} rows")
    sample_keys = []
    question_ids = []
    seed = manifest.get("seed")
    for ordinal, row in enumerate(sample):
        if row.get("sample_ordinal") != ordinal:
            raise ValueError("sample ordinal sequence is invalid")
        source_index = row.get("source_row_index")
        if row.get("request_seed") != seed + source_index:
            raise ValueError("sample request seed does not match source identity")
        sample_keys.append(
            (ordinal, source_index, row.get("question_id"), row.get("request_seed"))
        )
        question_ids.append(row.get("question_id"))
    if len(set(question_ids)) != expected_count:
        raise ValueError("sample question identities are not distinct")

    summaries: dict[str, dict] = {}
    for label in ("base", "adaptive_step50"):
        records_path = root / label / "records.jsonl"
        summary_path = root / label / "summary.json"
        if not records_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"missing {label} records or summary")
        records = _read_jsonl(records_path)
        record_keys = [
            (
                record.get("sample_ordinal"),
                record.get("source_row_index"),
                record.get("question_id"),
                record.get("request_seed"),
            )
            for record in records
        ]
        if record_keys != sample_keys:
            raise ValueError(f"{label} records do not match persisted sample identities")
        for record, sample_row in zip(records, sample, strict=True):
            if record.get("prompt") != sample_row.get("prompt") or record.get(
                "ground_truth"
            ) != sample_row.get("ground_truth"):
                raise ValueError(f"{label} record content does not match persisted sample")
        recomputed = _independent_summary(
            records, label=label, expected_count=expected_count
        )
        persisted = json.loads(summary_path.read_text(encoding="utf-8"))
        _assert_equal(persisted, recomputed, path=f"{label} summary")
        summaries[label] = recomputed

    expected_comparison = {
        "record_count_per_model": expected_count,
        "models": summaries,
        "quadrant_count_delta_adaptive_minus_base": {
            name: summaries["adaptive_step50"]["quadrant_counts"][name]
            - summaries["base"]["quadrant_counts"][name]
            for name in _ROUTE_NAMES
        },
        "metric_delta_adaptive_minus_base": {
            name: summaries["adaptive_step50"][name] - summaries["base"][name]
            for name in (
                "normal_accuracy",
                "concise_accuracy",
                "concise_minus_normal_accuracy",
                "normal_length_mean",
                "concise_length_mean",
                "concise_to_normal_ratio_mean",
                "concise_cap_hit_ratio",
            )
        },
    }
    comparison_path = root / "comparison.json"
    persisted_comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    _assert_equal(persisted_comparison, expected_comparison, path="comparison")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("manifest artifact hash mapping is missing")
    for relative, expected_sha in artifacts.items():
        artifact_path = root / relative
        if not artifact_path.is_file():
            raise ValueError(f"hashed artifact is missing: {relative}")
        actual_sha = _sha256(artifact_path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"artifact SHA256 mismatch for {relative}: expected {expected_sha}, got {actual_sha}"
            )

    return {
        "gate": "pass",
        "models": ["base", "adaptive_step50"],
        "record_count_per_model": expected_count,
        "artifact_count": len(artifacts),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--expected-count", type=int, default=128)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    report = validate_run(run_dir=args.run_dir, expected_count=args.expected_count)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("PAIRED_NORMAL_CONCISE_PROBE_GATE=PASS")


if __name__ == "__main__":
    main()
