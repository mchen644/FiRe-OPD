"""Validate the first training step of adaptive concise token-neutral OPD."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Mapping

_STEP_RE = re.compile(r"(?:^|\s)step:([0-9]+)\s+-")
_SOURCE_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_REQUIRED_TIMINGS = (
    "timing_s/normal_rollout",
    "timing_s/normal_reward",
    "timing_s/concise_probe",
    "timing_s/concise_reward",
    "timing_s/old_log_prob",
    "timing_s/ref",
    "timing_s/update_actor",
    "timing_s/step",
)
_REQUIRED_METRICS = (
    "training/global_step",
    "adaptive_concise_opd/total_questions",
    "adaptive_concise_opd/normal_correct_count",
    "adaptive_concise_opd/normal_wrong_count",
    "adaptive_concise_opd/concise_probe_count",
    "adaptive_concise_opd/easy_count",
    "adaptive_concise_opd/learnable_count",
    "adaptive_concise_opd/hard_count",
    "adaptive_concise_opd/concise_teacher_count",
    "adaptive_concise_opd/normal_teacher_count",
    "adaptive_concise_opd/response_budget_residual_tokens",
    "adaptive_concise_opd/response_budget_ratio",
    "actor/pg_loss",
    "actor/grad_norm",
    "rollout_corr/rollout_is_max",
    "rollout_corr/rollout_is_veto_fraction",
    "rollout_corr/rollout_is_catastrophic_token_fraction",
    *_REQUIRED_TIMINGS,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_metric_line(line: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for field in line.split(" - ")[1:]:
        if ":" not in field:
            continue
        key, value_text = field.split(":", 1)
        key = key.strip()
        value_text = value_text.strip()
        try:
            value = float(value_text)
        except ValueError:
            continue
        if key in metrics:
            raise ValueError(f"duplicate metric in step line: {key}")
        metrics[key] = value
    return metrics


def _require_metric(metrics: Mapping[str, float], key: str) -> float:
    if key not in metrics:
        raise ValueError(f"required profile metric is missing: {key}")
    value = float(metrics[key])
    if not math.isfinite(value):
        raise ValueError(f"profile metric must be finite: {key}")
    return value


def _as_nonnegative_int(metrics: Mapping[str, float], key: str) -> int:
    value = _require_metric(metrics, key)
    if value < 0 or not value.is_integer():
        raise ValueError(f"profile count must be a nonnegative integer: {key}")
    return int(value)


def validate_adaptive_concise_profile(
    *,
    log_path: Path,
    expected_step: int,
    expected_questions: int,
    source_commit: str,
) -> dict[str, object]:
    """Require exact routing identities, token conservation, and finite first-step health."""

    if isinstance(expected_step, bool) or not isinstance(expected_step, int) or expected_step <= 0:
        raise ValueError("expected_step must be a positive integer")
    if (
        isinstance(expected_questions, bool)
        or not isinstance(expected_questions, int)
        or expected_questions <= 0
    ):
        raise ValueError("expected_questions must be a positive integer")
    if not isinstance(source_commit, str) or _SOURCE_COMMIT_RE.fullmatch(source_commit) is None:
        raise ValueError("source_commit must be a 40-character lowercase git SHA")

    log = Path(log_path).resolve()
    if not log.is_file() or log.stat().st_size <= 0:
        raise ValueError(f"profile log is missing or empty: {log}")
    lines = log.read_text(encoding="utf-8", errors="strict").splitlines()
    matching_lines: list[tuple[int, str]] = []
    for line_number, line in enumerate(lines, start=1):
        match = _STEP_RE.search(line)
        if match is not None and int(match.group(1)) == expected_step:
            parsed = _parse_metric_line(line)
            if parsed.get("training/global_step") == float(expected_step):
                matching_lines.append((line_number, line))
    if len(matching_lines) != 1:
        raise ValueError(
            f"profile log must contain exactly one step-{expected_step} metric line; found {len(matching_lines)}"
        )

    line_number, metric_line = matching_lines[0]
    metrics = _parse_metric_line(metric_line)
    for key in _REQUIRED_METRICS:
        _require_metric(metrics, key)

    if _as_nonnegative_int(metrics, "training/global_step") != expected_step:
        raise ValueError("training/global_step does not match expected_step")
    total = _as_nonnegative_int(metrics, "adaptive_concise_opd/total_questions")
    if total != expected_questions:
        raise ValueError(
            f"adaptive total_questions mismatch: expected {expected_questions}, got {total}"
        )
    normal_correct = _as_nonnegative_int(metrics, "adaptive_concise_opd/normal_correct_count")
    normal_wrong = _as_nonnegative_int(metrics, "adaptive_concise_opd/normal_wrong_count")
    probe_count = _as_nonnegative_int(metrics, "adaptive_concise_opd/concise_probe_count")
    easy = _as_nonnegative_int(metrics, "adaptive_concise_opd/easy_count")
    learnable = _as_nonnegative_int(metrics, "adaptive_concise_opd/learnable_count")
    hard = _as_nonnegative_int(metrics, "adaptive_concise_opd/hard_count")
    concise_teacher = _as_nonnegative_int(metrics, "adaptive_concise_opd/concise_teacher_count")
    normal_teacher = _as_nonnegative_int(metrics, "adaptive_concise_opd/normal_teacher_count")
    if easy + learnable + hard != total:
        raise ValueError("easy/learnable/hard counts do not partition total_questions")
    if normal_correct != easy + learnable:
        raise ValueError("normal_correct count does not equal easy plus learnable")
    if normal_wrong != hard:
        raise ValueError("normal_wrong count does not equal hard")
    if probe_count != normal_correct:
        raise ValueError("concise probe count does not equal normal_correct")
    if concise_teacher != easy or normal_teacher != learnable + hard:
        raise ValueError("teacher prompt counts do not match adaptive routes")

    residual = _require_metric(metrics, "adaptive_concise_opd/response_budget_residual_tokens")
    ratio = _require_metric(metrics, "adaptive_concise_opd/response_budget_ratio")
    if residual != 0.0:
        raise ValueError(f"response budget residual must be zero, got {residual}")
    if ratio != 1.0:
        raise ValueError(f"response budget ratio must be exactly 1.0, got {ratio}")

    pg_loss = _require_metric(metrics, "actor/pg_loss")
    grad_norm = _require_metric(metrics, "actor/grad_norm")
    if grad_norm <= 0.0:
        raise ValueError(f"actor/grad_norm must be positive, got {grad_norm}")
    rollout_is_max = _require_metric(metrics, "rollout_corr/rollout_is_max")
    if rollout_is_max <= 0.0 or rollout_is_max > 5.0:
        raise ValueError(f"rollout_is_max must be in (0, 5], got {rollout_is_max}")
    veto_fraction = _require_metric(metrics, "rollout_corr/rollout_is_veto_fraction")
    catastrophic_fraction = _require_metric(
        metrics, "rollout_corr/rollout_is_catastrophic_token_fraction"
    )
    if veto_fraction != 0.0:
        raise ValueError(f"rollout IS veto fraction must be zero, got {veto_fraction}")
    if catastrophic_fraction != 0.0:
        raise ValueError(
            "rollout IS catastrophic token fraction must be zero, "
            f"got {catastrophic_fraction}"
        )

    timing = {key: _require_metric(metrics, key) for key in _REQUIRED_TIMINGS}
    for key, value in timing.items():
        if key in {"timing_s/concise_probe", "timing_s/concise_reward"} and probe_count == 0:
            if value < 0.0:
                raise ValueError(f"{key} must be nonnegative when no probes are generated")
        elif value <= 0.0:
            raise ValueError(f"{key} must be positive, got {value}")

    return {
        "decision": "pass",
        "log_path": str(log),
        "log_size": log.stat().st_size,
        "log_sha256": _sha256_file(log),
        "metric_line_number": line_number,
        "expected_step": expected_step,
        "expected_questions": expected_questions,
        "source_commit": source_commit,
        "route_counts": {"easy": easy, "learnable": learnable, "hard": hard},
        "normal_correct_count": normal_correct,
        "normal_wrong_count": normal_wrong,
        "probe_count": probe_count,
        "response_budget_residual_tokens": residual,
        "response_budget_ratio": ratio,
        "actor_pg_loss": pg_loss,
        "actor_grad_norm": grad_norm,
        "rollout_is_max": rollout_is_max,
        "rollout_is_veto_fraction": veto_fraction,
        "rollout_is_catastrophic_token_fraction": catastrophic_fraction,
        "timing_seconds": timing,
    }


def write_report_atomic(path: Path, report: Mapping[str, object]) -> None:
    """Create one JSON report atomically and refuse overwrite."""

    output = Path(path).resolve()
    if output.exists():
        raise FileExistsError(f"acceptance report already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(report), handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _current_source_commit() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--expected-questions", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    source_commit = args.source_commit or _current_source_commit()
    try:
        report = validate_adaptive_concise_profile(
            log_path=args.log,
            expected_step=args.expected_step,
            expected_questions=args.expected_questions,
            source_commit=source_commit,
        )
    except Exception as error:
        report = {
            "decision": "fail",
            "error_type": type(error).__name__,
            "error": str(error),
            "log_path": str(args.log.resolve()),
            "expected_step": args.expected_step,
            "expected_questions": args.expected_questions,
            "source_commit": source_commit,
        }
        write_report_atomic(args.output, report)
        print(json.dumps(report, sort_keys=True))
        return 2
    write_report_atomic(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
