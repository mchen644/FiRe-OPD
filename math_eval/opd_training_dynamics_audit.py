#!/usr/bin/env python3
"""Offline diagnostics for FiRe-OPD training dynamics.

This module intentionally avoids training code. It parses local W&B logs,
computes response-level diagnostics, and optionally scores fixed rollouts with
student/teacher models for token-level OPD alignment metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


METRIC_KEYS = [
    "training/global_step",
    "response_length/mean",
    "response_length/max",
    "response_length/min",
    "response_length/clip_ratio",
    "response_length_non_aborted/mean",
    "response/aborted_ratio",
    "tale_budget/response_length_mean",
    "tale_budget/esr_tokens_mean",
    "tale_budget/esr_supervised_fraction_mean",
    "tale_budget/truncated_token_fraction",
    "critic/score/mean",
    "actor/entropy",
    "actor/grad_norm",
    "rollout_corr/chi2_seq",
    "rollout_corr/log_ppl_abs_diff",
    "rollout_corr/skipped_no_valid_tokens",
]


@dataclass(frozen=True)
class RunSpec:
    name: str
    group: str
    log_path: str


def default_run_specs() -> list[RunSpec]:
    """Return local W&B runs used in the first offline audit."""

    return [
        RunSpec(
            name="raw_opd",
            group="raw",
            log_path="wandb/run-20260621_181403-t3g8daac/files/output.log",
        ),
        RunSpec(
            name="hardtrunc_budget",
            group="budget_teacher_20pct",
            log_path="wandb/run-20260630_050733-bk8qeqt1/files/output.log",
        ),
        RunSpec(
            name="hardtrunc_budget_resume75",
            group="budget_teacher_20pct",
            log_path="wandb/run-20260701_002104-h7aha9v2/files/output.log",
        ),
        RunSpec(
            name="hardtrunc_concise",
            group="concise_teacher_20pct",
            log_path="wandb/run-20260702_063810-6joq0xfl/files/output.log",
        ),
        RunSpec(
            name="hardtrunc_normal",
            group="normal_teacher_20pct",
            log_path="wandb/run-20260702_195307-ecq9x9ki/files/output.log",
        ),
    ]


def _parse_float_metric(line: str, key: str) -> float | None:
    match = re.search(re.escape(key) + r":(-?[0-9.eE+]+)", line)
    if not match:
        return None
    return float(match.group(1))


def parse_metric_line(line: str) -> dict[str, float | str]:
    """Parse one W&B output metric line.

    The local output uses both `step:N` and sometimes `training/global_step:N`.
    Prefer `training/global_step` because W&B record step can restart after
    resumed runs.
    """

    step_match = re.search(r"(?:^|\s)step:(\d+)\s+-", line)
    if not step_match:
        raise ValueError("metric line must contain 'step:N -'")
    wandb_step = int(step_match.group(1))
    record: dict[str, float | str] = {"wandb_step": float(wandb_step), "step": float(wandb_step)}
    for key in METRIC_KEYS:
        value = _parse_float_metric(line, key)
        if value is not None:
            record[key] = value
    if "training/global_step" in record:
        record["step"] = float(record["training/global_step"])
    return record


def normalize_record(record: dict[str, float | str]) -> dict[str, float | str]:
    """Add normalized fields used by plots and tables."""

    out = dict(record)
    response_mean = float(out.get("response_length/mean", float("nan")))
    tale_original = out.get("tale_budget/response_length_mean")
    out["supervised_length_mean"] = response_mean
    if tale_original is None:
        out["original_rollout_length_mean"] = response_mean
        out["has_tale_original_length"] = 0.0
    else:
        out["original_rollout_length_mean"] = float(tale_original)
        out["has_tale_original_length"] = 1.0
    return out


def parse_wandb_output_log(text: str, run_name: str, source_path: str) -> list[dict[str, float | str]]:
    """Parse all metric records from a local W&B output.log string."""

    rows: list[dict[str, float | str]] = []
    for line in text.splitlines():
        if not line.startswith("step:"):
            continue
        if " - " not in line:
            continue
        try:
            record = parse_metric_line(line)
        except ValueError:
            continue
        record["run_name"] = run_name
        record["source_path"] = source_path
        rows.append(normalize_record(record))
    rows.sort(key=lambda row: (str(row.get("run_name", "")), float(row.get("step", 0.0))))
    return rows
