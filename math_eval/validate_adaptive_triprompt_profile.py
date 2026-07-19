"""Validate an immutable one-step endpoint-preserving tri-prompt OPD profile."""

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
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_TOKEN_STATISTIC_STEMS = (
    "old_log_prob_mean",
    "ref_log_prob_mean",
    "actor_entropy_mean",
    "ref_minus_old_log_prob_mean",
)
_TOKEN_STATISTIC_KEYS = (
    *(
        f"adaptive_triprompt_opd/{stem}"
        for stem in _TOKEN_STATISTIC_STEMS
    ),
    *(
        f"adaptive_triprompt_opd/{route}_{stem}"
        for route in ("easy", "sensitive", "hard")
        for stem in _TOKEN_STATISTIC_STEMS
    ),
)
_REQUIRED_TIMINGS = (
    "timing_s/normal_rollout",
    "timing_s/normal_reward",
    "timing_s/concise_probe",
    "timing_s/concise_reward",
    "timing_s/triprompt_routing",
    "timing_s/old_log_prob",
    "timing_s/ref",
    "timing_s/update_actor",
    "timing_s/step",
)
_REQUIRED_METRICS = (
    "training/global_step",
    "adaptive_triprompt_opd/total_questions",
    "adaptive_triprompt_opd/normal_correct_count",
    "adaptive_triprompt_opd/normal_wrong_count",
    "adaptive_triprompt_opd/concise_probe_count",
    "adaptive_triprompt_opd/easy_count",
    "adaptive_triprompt_opd/sensitive_count",
    "adaptive_triprompt_opd/hard_count",
    "adaptive_triprompt_opd/concise_teacher_count",
    "adaptive_triprompt_opd/budget_teacher_count",
    "adaptive_triprompt_opd/normal_teacher_count",
    "adaptive_triprompt_opd/concise_correct_count",
    "adaptive_triprompt_opd/concise_wrong_count",
    "adaptive_triprompt_opd/concise_parse_fail_count",
    "adaptive_triprompt_opd/concise_missing_box_count",
    "adaptive_triprompt_opd/concise_cap_hit_count",
    "adaptive_triprompt_opd/normal_eos_count",
    "adaptive_triprompt_opd/normal_eos_ratio",
    "adaptive_triprompt_opd/normal_cap_hit_count",
    "adaptive_triprompt_opd/normal_cap_hit_ratio",
    "adaptive_triprompt_opd/concise_eos_count",
    "adaptive_triprompt_opd/concise_eos_ratio",
    "adaptive_triprompt_opd/concise_cap_hit_ratio",
    "adaptive_triprompt_opd/normal_response_tokens",
    "adaptive_triprompt_opd/actor_supervised_tokens",
    "adaptive_triprompt_opd/concise_response_tokens",
    "adaptive_triprompt_opd/supervision_token_residual",
    "adaptive_triprompt_opd/full_response_preservation_ratio",
    "adaptive_triprompt_opd/route_weight_min",
    "adaptive_triprompt_opd/route_weight_max",
    "adaptive_triprompt_opd/sensitive_budget_min",
    "adaptive_triprompt_opd/sensitive_budget_mean",
    "adaptive_triprompt_opd/sensitive_budget_max",
    "adaptive_triprompt_opd/total_questions_cumulative",
    "adaptive_triprompt_opd/concise_probe_count_cumulative",
    "adaptive_triprompt_opd/easy_count_cumulative",
    "adaptive_triprompt_opd/sensitive_count_cumulative",
    "adaptive_triprompt_opd/hard_count_cumulative",
    *_TOKEN_STATISTIC_KEYS,
    "actor/entropy",
    "actor/pg_loss",
    "actor/grad_norm",
    "rollout_corr/rollout_is_max",
    "rollout_corr/rollout_is_veto_fraction",
    "rollout_corr/rollout_is_catastrophic_token_fraction",
    "perf/max_memory_allocated_gb",
    "perf/max_memory_reserved_gb",
    *_REQUIRED_TIMINGS,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_metric_line(line: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for field in line.split(" - ")[1:]:
        if ":" not in field:
            continue
        key, value_text = field.split(":", 1)
        try:
            value = float(value_text.strip())
        except ValueError:
            continue
        key = key.strip()
        if key in metrics:
            raise ValueError(f"duplicate metric in step line: {key}")
        metrics[key] = value
    return metrics


def _parse_contract(lines: list[str]) -> dict[str, str]:
    contract: dict[str, str] = {}
    required = {
        "RUN_MODE",
        "TRAIN_DATA",
        "STUDENT_MODEL",
        "TEACHER_MODEL",
        "EXPERIMENT_NAME",
        "SOURCE_COMMIT",
        "triprompt_command",
    }
    for raw_line in lines:
        line = _ANSI_RE.sub("", raw_line)
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key not in required:
            continue
        if key in contract:
            raise ValueError(f"duplicate profile manifest field: {key}")
        contract[key] = value
    missing = sorted(required - contract.keys())
    if missing:
        raise ValueError(f"profile manifest fields are missing: {missing}")
    return contract


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


def _require_close(actual: float, expected: float, *, message: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"{message}: expected {expected}, got {actual}")


def _validate_positive_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_positive_float(value: float, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be positive")
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{name} must be positive")


def validate_adaptive_triprompt_profile(
    *,
    log_path: Path,
    expected_step: int,
    expected_questions: int,
    expected_total_steps: int,
    source_commit: str,
    train_data_path: Path,
    expected_train_data_sha256: str,
    student_model: str,
    teacher_model: str,
    expected_experiment: str,
    max_memory_allocated_gb: float,
    max_step_seconds: float,
    max_projected_train_hours: float,
) -> dict[str, object]:
    """Require immutable inputs, exact route identities, and healthy step-one training."""

    _validate_positive_int(expected_step, name="expected_step")
    _validate_positive_int(expected_questions, name="expected_questions")
    _validate_positive_int(expected_total_steps, name="expected_total_steps")
    _validate_positive_float(max_memory_allocated_gb, name="max_memory_allocated_gb")
    _validate_positive_float(max_step_seconds, name="max_step_seconds")
    _validate_positive_float(max_projected_train_hours, name="max_projected_train_hours")
    if _SOURCE_COMMIT_RE.fullmatch(source_commit) is None:
        raise ValueError("source_commit must be a 40-character lowercase git SHA")
    if _SHA256_RE.fullmatch(expected_train_data_sha256) is None:
        raise ValueError("expected_train_data_sha256 must be a lowercase SHA256")
    for name, value in (
        ("student_model", student_model),
        ("teacher_model", teacher_model),
        ("expected_experiment", expected_experiment),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be nonempty")

    log = Path(log_path).resolve()
    train_data = Path(train_data_path).resolve()
    if not log.is_file() or log.stat().st_size <= 0:
        raise ValueError(f"profile log is missing or empty: {log}")
    if not train_data.is_file() or train_data.stat().st_size <= 0:
        raise ValueError(f"training data is missing or empty: {train_data}")
    actual_train_hash = _sha256_file(train_data)
    if actual_train_hash != expected_train_data_sha256:
        raise ValueError(
            "training data sha256 mismatch: "
            f"expected {expected_train_data_sha256}, got {actual_train_hash}"
        )

    lines = log.read_text(encoding="utf-8", errors="strict").splitlines()
    contract = _parse_contract(lines)
    expected_contract = {
        "RUN_MODE": "profile",
        "TRAIN_DATA": str(train_data),
        "STUDENT_MODEL": student_model,
        "TEACHER_MODEL": teacher_model,
        "EXPERIMENT_NAME": expected_experiment,
        "SOURCE_COMMIT": source_commit,
    }
    for key, expected in expected_contract.items():
        if contract[key] != expected:
            raise ValueError(
                f"profile manifest {key} mismatch: expected {expected!r}, got {contract[key]!r}"
            )
    command = contract["triprompt_command"]
    if "algorithm.adaptive_triprompt_opd.enabled=True" not in command:
        raise ValueError("profile manifest command does not enable adaptive tri-prompt OPD")

    matching_lines: list[tuple[int, str]] = []
    for line_number, line in enumerate(lines, start=1):
        match = _STEP_RE.search(line)
        if match is not None and int(match.group(1)) == expected_step:
            parsed = _parse_metric_line(line)
            if parsed.get("training/global_step") == float(expected_step):
                matching_lines.append((line_number, line))
    if len(matching_lines) != 1:
        raise ValueError(
            f"profile log must contain exactly one step-{expected_step} metric line; "
            f"found {len(matching_lines)}"
        )

    line_number, metric_line = matching_lines[0]
    metrics = _parse_metric_line(metric_line)
    for key in _REQUIRED_METRICS:
        _require_metric(metrics, key)
    if _as_nonnegative_int(metrics, "training/global_step") != expected_step:
        raise ValueError("training/global_step does not match expected_step")

    total = _as_nonnegative_int(metrics, "adaptive_triprompt_opd/total_questions")
    normal_correct = _as_nonnegative_int(
        metrics, "adaptive_triprompt_opd/normal_correct_count"
    )
    normal_wrong = _as_nonnegative_int(
        metrics, "adaptive_triprompt_opd/normal_wrong_count"
    )
    probe_count = _as_nonnegative_int(
        metrics, "adaptive_triprompt_opd/concise_probe_count"
    )
    easy = _as_nonnegative_int(metrics, "adaptive_triprompt_opd/easy_count")
    sensitive = _as_nonnegative_int(metrics, "adaptive_triprompt_opd/sensitive_count")
    hard = _as_nonnegative_int(metrics, "adaptive_triprompt_opd/hard_count")
    concise_teacher = _as_nonnegative_int(
        metrics, "adaptive_triprompt_opd/concise_teacher_count"
    )
    budget_teacher = _as_nonnegative_int(
        metrics, "adaptive_triprompt_opd/budget_teacher_count"
    )
    normal_teacher = _as_nonnegative_int(
        metrics, "adaptive_triprompt_opd/normal_teacher_count"
    )
    concise_correct = _as_nonnegative_int(
        metrics, "adaptive_triprompt_opd/concise_correct_count"
    )
    concise_wrong = _as_nonnegative_int(
        metrics, "adaptive_triprompt_opd/concise_wrong_count"
    )
    if total != expected_questions:
        raise ValueError(
            f"adaptive total_questions mismatch: expected {expected_questions}, got {total}"
        )
    if easy + sensitive + hard != total:
        raise ValueError("easy/sensitive/hard route partition does not equal total questions")
    if normal_correct != easy + sensitive or normal_wrong != hard:
        raise ValueError("normal correctness does not match the route partition")
    if probe_count != normal_correct:
        raise ValueError("concise probe count does not equal normal-correct count")
    if (concise_teacher, budget_teacher, normal_teacher) != (easy, sensitive, hard):
        raise ValueError("teacher prompt counts do not match tri-prompt routes")
    if (concise_correct, concise_wrong) != (easy, sensitive):
        raise ValueError("concise correctness counts do not match easy/sensitive routes")

    cumulative_pairs = {
        "total_questions": total,
        "concise_probe_count": probe_count,
        "easy_count": easy,
        "sensitive_count": sensitive,
        "hard_count": hard,
    }
    for stem, expected in cumulative_pairs.items():
        actual = _as_nonnegative_int(
            metrics, f"adaptive_triprompt_opd/{stem}_cumulative"
        )
        if actual != expected:
            raise ValueError(f"step-one cumulative {stem} does not match its route count")

    normal_tokens = _require_metric(
        metrics, "adaptive_triprompt_opd/normal_response_tokens"
    )
    actor_tokens = _require_metric(
        metrics, "adaptive_triprompt_opd/actor_supervised_tokens"
    )
    residual = _require_metric(
        metrics, "adaptive_triprompt_opd/supervision_token_residual"
    )
    preservation_ratio = _require_metric(
        metrics, "adaptive_triprompt_opd/full_response_preservation_ratio"
    )
    concise_tokens = _require_metric(
        metrics, "adaptive_triprompt_opd/concise_response_tokens"
    )
    if normal_tokens <= 0 or actor_tokens <= 0:
        raise ValueError("normal and actor supervision token totals must be positive")
    if (probe_count and concise_tokens <= 0) or (not probe_count and concise_tokens != 0):
        raise ValueError(
            f"diagnostic tokens must match probe presence, got {concise_tokens} for {probe_count} probes"
        )
    if actor_tokens != normal_tokens or residual != 0.0 or preservation_ratio != 1.0:
        raise ValueError(
            "full-response supervision preservation failed: "
            f"normal={normal_tokens}, actor={actor_tokens}, residual={residual}, "
            f"ratio={preservation_ratio}"
        )
    route_weight_min = _require_metric(
        metrics, "adaptive_triprompt_opd/route_weight_min"
    )
    route_weight_max = _require_metric(
        metrics, "adaptive_triprompt_opd/route_weight_max"
    )
    if route_weight_min != 1.0 or route_weight_max != 1.0:
        raise ValueError(
            f"route weight must remain exactly 1.0, got [{route_weight_min}, {route_weight_max}]"
        )

    budget_min = _require_metric(metrics, "adaptive_triprompt_opd/sensitive_budget_min")
    budget_mean = _require_metric(metrics, "adaptive_triprompt_opd/sensitive_budget_mean")
    budget_max = _require_metric(metrics, "adaptive_triprompt_opd/sensitive_budget_max")
    if sensitive:
        if budget_min <= 0 or not (budget_min <= budget_mean <= budget_max):
            raise ValueError("sensitive B=L_n budget telemetry is invalid")
    elif (budget_min, budget_mean, budget_max) != (0.0, 0.0, 0.0):
        raise ValueError("zero-sensitive profile must have zero budget telemetry")

    parse_fail = _as_nonnegative_int(
        metrics, "adaptive_triprompt_opd/concise_parse_fail_count"
    )
    missing_box = _as_nonnegative_int(
        metrics, "adaptive_triprompt_opd/concise_missing_box_count"
    )
    normal_eos = _as_nonnegative_int(metrics, "adaptive_triprompt_opd/normal_eos_count")
    normal_cap = _as_nonnegative_int(
        metrics, "adaptive_triprompt_opd/normal_cap_hit_count"
    )
    concise_eos = _as_nonnegative_int(
        metrics, "adaptive_triprompt_opd/concise_eos_count"
    )
    concise_cap = _as_nonnegative_int(
        metrics, "adaptive_triprompt_opd/concise_cap_hit_count"
    )
    if parse_fail != missing_box:
        raise ValueError("concise parse-failure and missing-box counts disagree")
    if any(count > total for count in (normal_eos, normal_cap)) or any(
        count > probe_count for count in (concise_eos, concise_cap, parse_fail)
    ):
        raise ValueError("endpoint telemetry count exceeds its source row count")
    _require_close(
        _require_metric(metrics, "adaptive_triprompt_opd/normal_eos_ratio"),
        normal_eos / total,
        message="normal EOS endpoint ratio mismatch",
    )
    _require_close(
        _require_metric(metrics, "adaptive_triprompt_opd/normal_cap_hit_ratio"),
        normal_cap / total,
        message="normal cap endpoint ratio mismatch",
    )
    probe_denom = float(probe_count) if probe_count else 1.0
    _require_close(
        _require_metric(metrics, "adaptive_triprompt_opd/concise_eos_ratio"),
        concise_eos / probe_denom if probe_count else 0.0,
        message="concise EOS endpoint ratio mismatch",
    )
    _require_close(
        _require_metric(metrics, "adaptive_triprompt_opd/concise_cap_hit_ratio"),
        concise_cap / probe_denom if probe_count else 0.0,
        message="concise cap endpoint ratio mismatch",
    )

    token_statistics = {
        key.removeprefix("adaptive_triprompt_opd/"): _require_metric(metrics, key)
        for key in _TOKEN_STATISTIC_KEYS
    }
    actor_entropy = _require_metric(metrics, "actor/entropy")
    if actor_entropy < 0.0:
        raise ValueError(f"actor/entropy must be nonnegative, got {actor_entropy}")
    if not math.isclose(
        actor_entropy,
        token_statistics["actor_entropy_mean"],
        rel_tol=1e-5,
        abs_tol=1e-5,
    ):
        raise ValueError("actor/entropy disagrees with full-response token telemetry")
    for prefix, route_count in (("easy_", easy), ("sensitive_", sensitive), ("hard_", hard)):
        if route_count and token_statistics[f"{prefix}actor_entropy_mean"] < 0.0:
            raise ValueError(f"{prefix}actor entropy must be nonnegative")
    for prefix in ("", "easy_", "sensitive_", "hard_"):
        if prefix and not {"easy_": easy, "sensitive_": sensitive, "hard_": hard}[prefix]:
            continue
        expected_delta = (
            token_statistics[f"{prefix}ref_log_prob_mean"]
            - token_statistics[f"{prefix}old_log_prob_mean"]
        )
        if not math.isclose(
            token_statistics[f"{prefix}ref_minus_old_log_prob_mean"],
            expected_delta,
            rel_tol=1e-5,
            abs_tol=1e-5,
        ):
            raise ValueError(f"{prefix}old/ref log-prob telemetry identity failed")

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
    if veto_fraction != 0.0 or catastrophic_fraction != 0.0:
        raise ValueError("rollout importance-sampling veto/catastrophic fractions must be zero")

    memory_allocated = _require_metric(metrics, "perf/max_memory_allocated_gb")
    memory_reserved = _require_metric(metrics, "perf/max_memory_reserved_gb")
    if (
        memory_allocated <= 0
        or memory_reserved <= 0
        or memory_allocated > max_memory_allocated_gb
        or memory_reserved > max_memory_allocated_gb
    ):
        raise ValueError(
            "profile memory does not fit gate: "
            f"allocated={memory_allocated}, reserved={memory_reserved}, "
            f"limit={max_memory_allocated_gb}"
        )

    timing = {key: _require_metric(metrics, key) for key in _REQUIRED_TIMINGS}
    for key, value in timing.items():
        if probe_count == 0 and key in {
            "timing_s/concise_probe",
            "timing_s/concise_reward",
        }:
            if value < 0:
                raise ValueError(f"{key} must be nonnegative without probes")
        elif value <= 0:
            raise ValueError(f"{key} must be positive, got {value}")
    step_seconds = timing["timing_s/step"]
    if step_seconds > max_step_seconds:
        raise ValueError(
            f"profile step time exceeds gate: {step_seconds} > {max_step_seconds}"
        )
    projected_hours = step_seconds * expected_total_steps / 3600.0
    if projected_hours > max_projected_train_hours:
        raise ValueError(
            "projected training time exceeds gate: "
            f"{projected_hours}h > {max_projected_train_hours}h"
        )

    return {
        "decision": "pass",
        "log_path": str(log),
        "log_size": log.stat().st_size,
        "log_sha256": _sha256_file(log),
        "metric_line_number": line_number,
        "expected_step": expected_step,
        "expected_questions": expected_questions,
        "expected_total_steps": expected_total_steps,
        "source_commit": source_commit,
        "train_data_path": str(train_data),
        "train_data_sha256": actual_train_hash,
        "student_model": student_model,
        "teacher_model": teacher_model,
        "experiment_name": expected_experiment,
        "config_command_sha256": _sha256_text(command),
        "route_counts": {"easy": easy, "sensitive": sensitive, "hard": hard},
        "teacher_prompt_counts": {
            "concise": concise_teacher,
            "budget": budget_teacher,
            "normal": normal_teacher,
        },
        "normal_correct_count": normal_correct,
        "normal_wrong_count": normal_wrong,
        "probe_count": probe_count,
        "normal_response_tokens": normal_tokens,
        "actor_supervised_tokens": actor_tokens,
        "concise_response_tokens": concise_tokens,
        "supervision_token_residual": residual,
        "full_response_preservation_ratio": preservation_ratio,
        "endpoint_counts": {
            "normal_eos": normal_eos,
            "normal_cap_hit": normal_cap,
            "concise_eos": concise_eos,
            "concise_cap_hit": concise_cap,
            "concise_parse_fail": parse_fail,
        },
        "token_statistics": token_statistics,
        "actor_entropy": actor_entropy,
        "actor_pg_loss": pg_loss,
        "actor_grad_norm": grad_norm,
        "rollout_is_max": rollout_is_max,
        "rollout_is_veto_fraction": veto_fraction,
        "rollout_is_catastrophic_token_fraction": catastrophic_fraction,
        "max_memory_allocated_gb": memory_allocated,
        "max_memory_reserved_gb": memory_reserved,
        "timing_seconds": timing,
        "projected_50_step_train_hours": projected_hours,
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
    parser.add_argument("--expected-total-steps", type=int, required=True)
    parser.add_argument("--source-commit", default=None)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--expected-train-data-sha256", required=True)
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--expected-experiment", required=True)
    parser.add_argument("--max-memory-allocated-gb", type=float, required=True)
    parser.add_argument("--max-step-seconds", type=float, required=True)
    parser.add_argument("--max-projected-train-hours", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    source_commit = args.source_commit or _current_source_commit()
    try:
        report = validate_adaptive_triprompt_profile(
            log_path=args.log,
            expected_step=args.expected_step,
            expected_questions=args.expected_questions,
            expected_total_steps=args.expected_total_steps,
            source_commit=source_commit,
            train_data_path=args.train_data,
            expected_train_data_sha256=args.expected_train_data_sha256,
            student_model=args.student_model,
            teacher_model=args.teacher_model,
            expected_experiment=args.expected_experiment,
            max_memory_allocated_gb=args.max_memory_allocated_gb,
            max_step_seconds=args.max_step_seconds,
            max_projected_train_hours=args.max_projected_train_hours,
        )
    except Exception as error:
        report = {
            "decision": "fail",
            "error_type": type(error).__name__,
            "error": str(error),
            "log_path": str(args.log.resolve()),
            "expected_step": args.expected_step,
            "expected_questions": args.expected_questions,
            "expected_total_steps": args.expected_total_steps,
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
