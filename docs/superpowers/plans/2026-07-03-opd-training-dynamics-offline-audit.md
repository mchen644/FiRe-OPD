# OPD Training Dynamics Offline Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an offline audit tool and report that explains raw OPD length inflation, concise-teacher length collapse, and rollout-length-budget stability without launching new training.

**Architecture:** Add one focused analysis module under `math_eval/` with pure-Python log parsing, SVG rendering, response repetition metrics, and lazy heavy-model alignment scoring. Unit tests cover all non-heavy logic; model scoring is exposed by CLI and run on small subsets for the report. The first report uses existing W&B logs, existing eval outputs, and available checkpoints only.

**Tech Stack:** Python stdlib (`argparse`, `csv`, `json`, `math`, `re`, `statistics`, `zlib`, `pathlib`), PyTorch for tensor alignment metrics, optional Transformers for model scoring, pure SVG output instead of matplotlib.

## Global Constraints

- Do not launch new training jobs in this phase.
- Do not change training semantics.
- Do not add new Python dependencies; current `matplotlib` is unavailable, so plots must be pure SVG.
- Do not claim step18/19 token-level mechanism proof if no step18/19 checkpoint exists; label it as log-level evidence plus checkpoint-level consistency.
- For hardtrunc runs, `response_length/mean` is supervised/truncated training width, not original rollout length.
- For hardtrunc runs, use `tale_budget/response_length_mean` as original student rollout length when present.
- Keep student rollout prompt fixed when comparing teacher prompt styles for alignment scoring.
- Default token-level alignment top-k is `k=16`, matching the 2604.13016 diagnostic lens.

---

## File Structure

- Create `math_eval/opd_training_dynamics_audit.py`
  - owns log parsing, CSV writing, pure SVG plots, repetition metrics, tensor-level alignment metrics, optional model scoring, and CLI entry points.
- Create `math_eval/test_opd_training_dynamics_audit.py`
  - unit tests for parser, repetition metrics, tensor alignment metrics, SVG rendering, and prompt-style construction.
- Create generated output directory `math_eval/opd_training_dynamics_audit/`
  - contains `summary.md`, `metrics_by_step.csv`, `repetition_metrics.csv`, optional `alignment_metrics.csv`, and SVGs.
- Modify no training code.

---

### Task 1: Log Parser and Normalized Step Metrics

**Files:**
- Create: `math_eval/opd_training_dynamics_audit.py`
- Create: `math_eval/test_opd_training_dynamics_audit.py`

**Interfaces:**
- Produces: `parse_metric_line(line: str) -> dict[str, float | str]`
- Produces: `parse_wandb_output_log(text: str, run_name: str, source_path: str) -> list[dict[str, float | str]]`
- Produces: `normalize_record(record: dict[str, float | str]) -> dict[str, float | str]`
- Produces: `default_run_specs() -> list[RunSpec]`
- Produces: `RunSpec(name: str, group: str, log_path: str)` dataclass

- [ ] **Step 1: Write failing parser tests**

Add this to `math_eval/test_opd_training_dynamics_audit.py`:

```python
from pathlib import Path

from math_eval.opd_training_dynamics_audit import (
    parse_metric_line,
    parse_wandb_output_log,
    normalize_record,
    default_run_specs,
)


def test_parse_metric_line_prefers_training_global_step():
    line = (
        "step:7 - actor/entropy:0.25 - training/global_step:19 - "
        "response_length/mean:8218.8 - response_length/max:16384.0 - "
        "critic/score/mean:0.711 - actor/grad_norm:2.72"
    )
    record = parse_metric_line(line)
    assert record["step"] == 19
    assert record["wandb_step"] == 7
    assert record["response_length/mean"] == 8218.8
    assert record["critic/score/mean"] == 0.711
    assert record["actor/grad_norm"] == 2.72


def test_parse_metric_line_falls_back_to_wandb_step():
    line = "step:3 - actor/entropy:0.33 - response_length/mean:1500.0"
    record = parse_metric_line(line)
    assert record["step"] == 3
    assert record["wandb_step"] == 3
    assert record["actor/entropy"] == 0.33


def test_normalize_record_distinguishes_hardtrunc_original_rollout_length():
    record = normalize_record(
        {
            "run_name": "normal hardtrunc",
            "step": 50,
            "response_length/mean": 713.5,
            "tale_budget/response_length_mean": 3567.6,
        }
    )
    assert record["supervised_length_mean"] == 713.5
    assert record["original_rollout_length_mean"] == 3567.6
    assert record["has_tale_original_length"] == 1.0


def test_normalize_record_uses_response_length_for_raw_opd():
    record = normalize_record(
        {
            "run_name": "raw OPD",
            "step": 19,
            "response_length/mean": 8218.8,
        }
    )
    assert record["supervised_length_mean"] == 8218.8
    assert record["original_rollout_length_mean"] == 8218.8
    assert record["has_tale_original_length"] == 0.0


def test_parse_wandb_output_log_filters_non_metric_lines():
    text = "hello\nstep:1 - training/global_step:1 - response_length/mean:10.0\nbye\n"
    rows = parse_wandb_output_log(text, run_name="r", source_path="fake.log")
    assert len(rows) == 1
    assert rows[0]["run_name"] == "r"
    assert rows[0]["source_path"] == "fake.log"
    assert rows[0]["step"] == 1
    assert rows[0]["original_rollout_length_mean"] == 10.0


def test_default_run_specs_include_required_runs():
    names = {spec.name for spec in default_run_specs()}
    assert "raw_opd" in names
    assert "hardtrunc_budget" in names
    assert "hardtrunc_concise" in names
    assert "hardtrunc_normal" in names
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /home/mchen/FiRe-OPD
pytest math_eval/test_opd_training_dynamics_audit.py -q
```

Expected: FAIL because `math_eval.opd_training_dynamics_audit` does not exist.

- [ ] **Step 3: Implement parser module**

Create `math_eval/opd_training_dynamics_audit.py` with this initial content:

```python
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
```

- [ ] **Step 4: Run parser tests**

Run:

```bash
cd /home/mchen/FiRe-OPD
pytest math_eval/test_opd_training_dynamics_audit.py -q
```

Expected: PASS for the five parser tests.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
cd /home/mchen/FiRe-OPD
git add math_eval/opd_training_dynamics_audit.py math_eval/test_opd_training_dynamics_audit.py
git commit -m "Add OPD audit log parser"
```

---

### Task 2: Macro CSV, SVG Curves, and Log-Only Summary

**Files:**
- Modify: `math_eval/opd_training_dynamics_audit.py`
- Modify: `math_eval/test_opd_training_dynamics_audit.py`
- Create generated: `math_eval/opd_training_dynamics_audit/metrics_by_step.csv`
- Create generated: `math_eval/opd_training_dynamics_audit/macro_length_score_entropy_grad.svg`
- Create generated: `math_eval/opd_training_dynamics_audit/summary.md`

**Interfaces:**
- Consumes: `parse_wandb_output_log(...)`
- Produces: `load_default_log_metrics(repo_dir: Path) -> list[dict[str, float | str]]`
- Produces: `write_csv(rows: list[dict[str, float | str]], path: Path) -> None`
- Produces: `render_macro_svg(rows: list[dict[str, float | str]], path: Path) -> None`
- Produces: `write_summary(rows: list[dict[str, float | str]], out_dir: Path) -> None`
- Produces CLI: `python math_eval/opd_training_dynamics_audit.py macro --repo-dir /home/mchen/FiRe-OPD --output-dir math_eval/opd_training_dynamics_audit`

- [ ] **Step 1: Add failing tests for CSV and SVG rendering**

Append to `math_eval/test_opd_training_dynamics_audit.py`:

```python
from math_eval.opd_training_dynamics_audit import write_csv, render_macro_svg, summarize_run_lengths


def test_write_csv_includes_normalized_fields(tmp_path):
    rows = [
        {"run_name": "r", "step": 1.0, "supervised_length_mean": 10.0, "original_rollout_length_mean": 50.0},
        {"run_name": "r", "step": 2.0, "supervised_length_mean": 20.0, "original_rollout_length_mean": 60.0},
    ]
    out = tmp_path / "metrics.csv"
    write_csv(rows, out)
    text = out.read_text()
    assert "run_name,step" in text
    assert "original_rollout_length_mean" in text
    assert "60.0" in text


def test_render_macro_svg_writes_series_labels(tmp_path):
    rows = [
        {"run_name": "raw_opd", "step": 1.0, "original_rollout_length_mean": 1500.0, "supervised_length_mean": 1500.0, "critic/score/mean": 0.5, "actor/entropy": 0.3, "actor/grad_norm": 4.0},
        {"run_name": "raw_opd", "step": 2.0, "original_rollout_length_mean": 2500.0, "supervised_length_mean": 2500.0, "critic/score/mean": 0.6, "actor/entropy": 0.2, "actor/grad_norm": 3.0},
        {"run_name": "hardtrunc_concise", "step": 1.0, "original_rollout_length_mean": 1500.0, "supervised_length_mean": 300.0, "critic/score/mean": 0.5, "actor/entropy": 0.3, "actor/grad_norm": 12.0},
    ]
    out = tmp_path / "macro.svg"
    render_macro_svg(rows, out)
    svg = out.read_text()
    assert "AIME-independent training dynamics" in svg
    assert "raw_opd" in svg
    assert "hardtrunc_concise" in svg
    assert "Original rollout length" in svg


def test_summarize_run_lengths_reports_peak_and_final():
    rows = [
        {"run_name": "raw_opd", "step": 1.0, "original_rollout_length_mean": 1500.0},
        {"run_name": "raw_opd", "step": 19.0, "original_rollout_length_mean": 8218.8},
        {"run_name": "raw_opd", "step": 69.0, "original_rollout_length_mean": 4895.8},
    ]
    summary = summarize_run_lengths(rows)
    assert summary["raw_opd"]["peak_step"] == 19.0
    assert summary["raw_opd"]["peak_original_rollout_length"] == 8218.8
    assert summary["raw_opd"]["final_original_rollout_length"] == 4895.8
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /home/mchen/FiRe-OPD
pytest math_eval/test_opd_training_dynamics_audit.py -q
```

Expected: FAIL because `write_csv`, `render_macro_svg`, and `summarize_run_lengths` are not defined.

- [ ] **Step 3: Implement CSV, SVG, summary, and CLI**

Append these functions to `math_eval/opd_training_dynamics_audit.py`:

```python
RUN_COLORS = {
    "raw_opd": "#1f77b4",
    "hardtrunc_budget": "#d62728",
    "hardtrunc_budget_resume75": "#e377c2",
    "hardtrunc_concise": "#ff7f0e",
    "hardtrunc_normal": "#2ca02c",
}


def load_default_log_metrics(repo_dir: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for spec in default_run_specs():
        path = repo_dir / spec.log_path
        if not path.exists():
            continue
        parsed = parse_wandb_output_log(path.read_text(errors="ignore"), run_name=spec.name, source_path=str(path))
        for row in parsed:
            row["run_group"] = spec.group
        rows.extend(parsed)
    rows.sort(key=lambda row: (str(row.get("run_name", "")), float(row.get("step", 0.0))))
    return rows


def write_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "run_name",
        "run_group",
        "step",
        "wandb_step",
        "source_path",
        "original_rollout_length_mean",
        "supervised_length_mean",
        "has_tale_original_length",
        "response_length/max",
        "response_length/clip_ratio",
        "response/aborted_ratio",
        "tale_budget/esr_tokens_mean",
        "tale_budget/esr_supervised_fraction_mean",
        "critic/score/mean",
        "actor/entropy",
        "actor/grad_norm",
        "rollout_corr/chi2_seq",
        "rollout_corr/log_ppl_abs_diff",
        "rollout_corr/skipped_no_valid_tokens",
    ]
    keys = list(preferred)
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _group_rows(rows: list[dict[str, float | str]]) -> dict[str, list[dict[str, float | str]]]:
    grouped: dict[str, list[dict[str, float | str]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("run_name", "unknown")), []).append(row)
    for group_rows in grouped.values():
        group_rows.sort(key=lambda row: float(row.get("step", 0.0)))
    return grouped


def _polyline(points: list[tuple[float, float]], color: str, width: float = 2.0) -> str:
    if not points:
        return ""
    encoded = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{encoded}" fill="none" stroke="{color}" stroke-width="{width}"/>'


def render_macro_svg(rows: list[dict[str, float | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped = _group_rows(rows)
    width, height = 1180, 760
    margin_left, margin_top = 70, 70
    panel_w, panel_h = 470, 245
    gap_x, gap_y = 70, 75
    panels = [
        ("Original rollout length", "original_rollout_length_mean", 0.0, None),
        ("Supervised/training length", "supervised_length_mean", 0.0, None),
        ("Score", "critic/score/mean", 0.0, 1.0),
        ("Actor entropy", "actor/entropy", 0.0, None),
    ]
    max_step = max(float(row.get("step", 0.0)) for row in rows) if rows else 1.0
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933}.title{font-size:22px;font-weight:700}.axis{font-size:12px;fill:#52616b}.label{font-size:13px}</style>',
        '<text x="590" y="34" text-anchor="middle" class="title">AIME-independent training dynamics</text>',
        '<text x="590" y="55" text-anchor="middle" class="axis">Hardtrunc original rollout length uses tale_budget/response_length_mean when available</text>',
    ]
    for idx, (title, key, ymin_fixed, ymax_fixed) in enumerate(panels):
        col = idx % 2
        row_idx = idx // 2
        x0 = margin_left + col * (panel_w + gap_x)
        y0 = margin_top + row_idx * (panel_h + gap_y)
        vals = [float(row[key]) for row in rows if key in row and not math.isnan(float(row[key]))]
        ymin = ymin_fixed
        ymax = ymax_fixed if ymax_fixed is not None else (max(vals) * 1.08 if vals else 1.0)
        if ymax <= ymin:
            ymax = ymin + 1.0
        lines.append(f'<text x="{x0 + panel_w / 2:.1f}" y="{y0 - 14:.1f}" text-anchor="middle" class="label">{title}</text>')
        lines.append(f'<rect x="{x0}" y="{y0}" width="{panel_w}" height="{panel_h}" fill="#fbfcfd" stroke="#d9e2ec"/>')
        for tick in range(0, 6):
            gx = x0 + panel_w * tick / 5
            gy = y0 + panel_h * tick / 5
            lines.append(f'<line x1="{gx:.1f}" y1="{y0}" x2="{gx:.1f}" y2="{y0 + panel_h}" stroke="#edf2f7"/>')
            lines.append(f'<line x1="{x0}" y1="{gy:.1f}" x2="{x0 + panel_w}" y2="{gy:.1f}" stroke="#edf2f7"/>')
        for run_name, group_rows in grouped.items():
            pts: list[tuple[float, float]] = []
            for rec in group_rows:
                if key not in rec:
                    continue
                value = float(rec[key])
                if math.isnan(value):
                    continue
                sx = x0 + panel_w * float(rec.get("step", 0.0)) / max_step
                sy = y0 + panel_h * (ymax - value) / (ymax - ymin)
                pts.append((sx, sy))
            lines.append(_polyline(pts, RUN_COLORS.get(run_name, "#64748b"), width=2.2))
        lines.append(f'<text x="{x0}" y="{y0 + panel_h + 18}" class="axis">step 0</text>')
        lines.append(f'<text x="{x0 + panel_w}" y="{y0 + panel_h + 18}" text-anchor="end" class="axis">step {max_step:.0f}</text>')
        lines.append(f'<text x="{x0 - 8}" y="{y0 + 4}" text-anchor="end" class="axis">{ymax:.2g}</text>')
        lines.append(f'<text x="{x0 - 8}" y="{y0 + panel_h}" text-anchor="end" class="axis">{ymin:.2g}</text>')
    legend_x, legend_y = 920, 645
    lines.append(f'<text x="{legend_x}" y="{legend_y}" class="label">Legend</text>')
    offset = 24
    for run_name in grouped:
        color = RUN_COLORS.get(run_name, "#64748b")
        y = legend_y + offset
        lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 28}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{legend_x + 36}" y="{y + 4}" class="axis">{run_name}</text>')
        offset += 20
    lines.append("</svg>")
    path.write_text("\n".join(lines))


def summarize_run_lengths(rows: list[dict[str, float | str]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for run_name, group_rows in _group_rows(rows).items():
        valid = [row for row in group_rows if "original_rollout_length_mean" in row]
        if not valid:
            continue
        peak = max(valid, key=lambda row: float(row["original_rollout_length_mean"]))
        final = max(valid, key=lambda row: float(row.get("step", 0.0)))
        summary[run_name] = {
            "peak_step": float(peak.get("step", 0.0)),
            "peak_original_rollout_length": float(peak["original_rollout_length_mean"]),
            "final_step": float(final.get("step", 0.0)),
            "final_original_rollout_length": float(final["original_rollout_length_mean"]),
        }
    return summary


def write_summary(rows: list[dict[str, float | str]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    length_summary = summarize_run_lengths(rows)
    lines = [
        "# OPD Training Dynamics Offline Audit",
        "",
        "This report is generated from existing local W&B logs only. It does not launch training.",
        "",
        "## Macro length summary",
        "",
        "| run | peak step | peak original rollout length | final step | final original rollout length |",
        "|---|---:|---:|---:|---:|",
    ]
    for run_name, stats in sorted(length_summary.items()):
        lines.append(
            f"| {run_name} | {stats['peak_step']:.0f} | {stats['peak_original_rollout_length']:.1f} | "
            f"{stats['final_step']:.0f} | {stats['final_original_rollout_length']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Initial interpretation",
            "",
            "- Raw OPD should be inspected around its peak-length step for truncation-repetition inflation.",
            "- Concise-teacher hardtrunc should be inspected for entropy collapse and short-mode attraction.",
            "- Rollout-length-budget hardtrunc should be inspected using original rollout length, not supervised length.",
            "",
            "## Limitations",
            "",
            "- Step18/19 mechanism-level proof requires step18/19 checkpoints. If unavailable, token-level claims are limited to available checkpoints and log-level consistency.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines))


def run_macro(args: argparse.Namespace) -> None:
    repo_dir = Path(args.repo_dir).resolve()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = repo_dir / out_dir
    rows = load_default_log_metrics(repo_dir)
    write_csv(rows, out_dir / "metrics_by_step.csv")
    render_macro_svg(rows, out_dir / "macro_length_score_entropy_grad.svg")
    write_summary(rows, out_dir)
    print(f"wrote {out_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    macro = sub.add_parser("macro", help="parse W&B logs and render macro curves")
    macro.add_argument("--repo-dir", default="/home/mchen/FiRe-OPD")
    macro.add_argument("--output-dir", default="math_eval/opd_training_dynamics_audit")
    macro.set_defaults(func=run_macro)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run:

```bash
cd /home/mchen/FiRe-OPD
pytest math_eval/test_opd_training_dynamics_audit.py -q
```

Expected: PASS.

- [ ] **Step 5: Run log-only macro audit**

Run:

```bash
cd /home/mchen/FiRe-OPD
python math_eval/opd_training_dynamics_audit.py macro \
  --repo-dir /home/mchen/FiRe-OPD \
  --output-dir math_eval/opd_training_dynamics_audit
```

Expected files:

```text
math_eval/opd_training_dynamics_audit/metrics_by_step.csv
math_eval/opd_training_dynamics_audit/macro_length_score_entropy_grad.svg
math_eval/opd_training_dynamics_audit/summary.md
```

- [ ] **Step 6: Commit Task 2**

Run:

```bash
cd /home/mchen/FiRe-OPD
git add math_eval/opd_training_dynamics_audit.py math_eval/test_opd_training_dynamics_audit.py math_eval/opd_training_dynamics_audit/metrics_by_step.csv math_eval/opd_training_dynamics_audit/macro_length_score_entropy_grad.svg math_eval/opd_training_dynamics_audit/summary.md
git commit -m "Add OPD audit macro curves"
```

---

### Task 3: Response-Level Repetition and Truncation Metrics

**Files:**
- Modify: `math_eval/opd_training_dynamics_audit.py`
- Modify: `math_eval/test_opd_training_dynamics_audit.py`
- Create generated: `math_eval/opd_training_dynamics_audit/repetition_metrics.csv`

**Interfaces:**
- Produces: `compression_repetition_ratio(text: str) -> float`
- Produces: `ngram_repetition_rate(tokens: list[str], n: int) -> float`
- Produces: `iter_eval_responses(path: Path, run_name: str) -> Iterable[dict[str, Any]]`
- Produces: `compute_repetition_metrics(rows: Iterable[dict[str, Any]], max_token_length: int | None) -> list[dict[str, float | str]]`
- Extends CLI with `repetition` subcommand.

- [ ] **Step 1: Add failing repetition tests**

Append to `math_eval/test_opd_training_dynamics_audit.py`:

```python
import json

from math_eval.opd_training_dynamics_audit import (
    compression_repetition_ratio,
    ngram_repetition_rate,
    iter_eval_responses,
    compute_repetition_metrics,
)


def test_compression_repetition_ratio_higher_for_repetitive_text():
    repetitive = "wait wait wait wait wait wait wait wait " * 30
    diverse = "We solve by substituting x, simplifying a quadratic, and checking the boundary cases. " * 4
    assert compression_repetition_ratio(repetitive) > compression_repetition_ratio(diverse)


def test_ngram_repetition_rate_counts_repeated_ngrams():
    tokens = "a b c a b c a b c d e".split()
    assert ngram_repetition_rate(tokens, n=3) > 0.0
    assert ngram_repetition_rate(["a", "b"], n=3) == 0.0


def test_iter_eval_responses_reads_problem_level_schema(tmp_path):
    path = tmp_path / "eval.jsonl"
    path.write_text(
        json.dumps(
            {
                "problem": "p0",
                "responses": ["short", "long long long"],
                "response_lengths": [1, 3],
                "acc_list": [True, False],
            }
        )
        + "\n"
    )
    rows = list(iter_eval_responses(path, run_name="r"))
    assert len(rows) == 2
    assert rows[0]["run_name"] == "r"
    assert rows[0]["problem_index"] == 0
    assert rows[1]["sample_index"] == 1
    assert rows[1]["is_correct"] == 0.0


def test_compute_repetition_metrics_flags_truncated_length():
    rows = [
        {"run_name": "r", "response": "a b c", "response_length": 3.0, "is_correct": 1.0},
        {"run_name": "r", "response": "wait " * 100, "response_length": 16384.0, "is_correct": 0.0},
    ]
    metrics = compute_repetition_metrics(rows, max_token_length=16384)
    assert len(metrics) == 1
    assert metrics[0]["run_name"] == "r"
    assert metrics[0]["num_responses"] == 2.0
    assert metrics[0]["truncation_rate"] == 0.5
    assert metrics[0]["mean_compression_repetition_ratio"] > 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /home/mchen/FiRe-OPD
pytest math_eval/test_opd_training_dynamics_audit.py -q
```

Expected: FAIL because repetition functions are not defined.

- [ ] **Step 3: Implement repetition metrics and CLI**

Append to `math_eval/opd_training_dynamics_audit.py`:

```python

def compression_repetition_ratio(text: str) -> float:
    """Return a compressibility-based repetition score in [0, 1].

    Higher values mean the text is easier to compress and therefore more
    repetition-like. This is a transparent proxy for 2604.08527's
    compression-based repetition lens.
    """

    raw = text.encode("utf-8", errors="ignore")
    if not raw:
        return 0.0
    compressed = zlib.compress(raw, level=9)
    return max(0.0, min(1.0, 1.0 - len(compressed) / max(1, len(raw))))


def ngram_repetition_rate(tokens: list[str], n: int) -> float:
    if n <= 0:
        raise ValueError("n must be positive")
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    seen: set[tuple[str, ...]] = set()
    repeated = 0
    for ngram in ngrams:
        if ngram in seen:
            repeated += 1
        else:
            seen.add(ngram)
    return repeated / len(ngrams)


def iter_eval_responses(path: Path, run_name: str) -> Iterable[dict[str, Any]]:
    """Yield one row per response from FiRe-OPD eval JSONL outputs."""

    with path.open() as f:
        for problem_index, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            responses = row.get("responses", [])
            lengths = row.get("response_lengths", [float("nan")] * len(responses))
            acc_list = row.get("acc_list", [False] * len(responses))
            for sample_index, response in enumerate(responses):
                yield {
                    "run_name": run_name,
                    "source_path": str(path),
                    "problem_index": float(problem_index),
                    "sample_index": float(sample_index),
                    "response": str(response),
                    "response_length": float(lengths[sample_index]) if sample_index < len(lengths) else float("nan"),
                    "is_correct": 1.0 if sample_index < len(acc_list) and bool(acc_list[sample_index]) else 0.0,
                }


def compute_repetition_metrics(rows: Iterable[dict[str, Any]], max_token_length: int | None) -> list[dict[str, float | str]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["run_name"]), []).append(row)
    out: list[dict[str, float | str]] = []
    for run_name, group_rows in sorted(grouped.items()):
        compression_scores = [compression_repetition_ratio(str(row.get("response", ""))) for row in group_rows]
        unigram_tokens = [str(row.get("response", "")).split() for row in group_rows]
        rep4 = [ngram_repetition_rate(tokens, 4) for tokens in unigram_tokens]
        lengths = [float(row.get("response_length", float("nan"))) for row in group_rows]
        correct = [float(row.get("is_correct", 0.0)) for row in group_rows]
        if max_token_length is None:
            truncation_rate = float("nan")
        else:
            truncation_rate = sum(1 for length in lengths if length >= max_token_length) / max(1, len(lengths))
        out.append(
            {
                "run_name": run_name,
                "num_responses": float(len(group_rows)),
                "mean_response_length": statistics.fmean(lengths) if lengths else float("nan"),
                "accuracy": statistics.fmean(correct) if correct else float("nan"),
                "truncation_rate": truncation_rate,
                "mean_compression_repetition_ratio": statistics.fmean(compression_scores) if compression_scores else float("nan"),
                "mean_4gram_repetition_rate": statistics.fmean(rep4) if rep4 else float("nan"),
            }
        )
    return out


def run_repetition(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for item in args.eval_jsonl:
        run_name, path_str = item.split("=", 1)
        rows.extend(iter_eval_responses(Path(path_str), run_name=run_name))
    metrics = compute_repetition_metrics(rows, max_token_length=args.max_token_length)
    write_csv(metrics, out_dir / "repetition_metrics.csv")
    print(f"wrote {out_dir / 'repetition_metrics.csv'}")
```

Modify `build_arg_parser()` to include:

```python
    repetition = sub.add_parser("repetition", help="compute response-level repetition metrics from eval JSONL")
    repetition.add_argument("--output-dir", default="math_eval/opd_training_dynamics_audit")
    repetition.add_argument("--max-token-length", type=int, default=16384)
    repetition.add_argument(
        "--eval-jsonl",
        action="append",
        required=True,
        help="Named eval output in the form run_name=/path/to/output.jsonl. Can be repeated.",
    )
    repetition.set_defaults(func=run_repetition)
```

- [ ] **Step 4: Run tests**

Run:

```bash
cd /home/mchen/FiRe-OPD
pytest math_eval/test_opd_training_dynamics_audit.py -q
```

Expected: PASS.

- [ ] **Step 5: Run repetition metrics on existing AIME24 outputs**

Run:

```bash
cd /home/mchen/FiRe-OPD
python math_eval/opd_training_dynamics_audit.py repetition \
  --output-dir math_eval/opd_training_dynamics_audit \
  --max-token-length 16384 \
  --eval-jsonl raw_opd=math_eval/opd_rawprompt_step_eval_outputs/aime24/opd-strong-to-weak-rawprompt-4gpu-tp4-refmb4-rollmb4-step50-baseline-n32-seed42.jsonl \
  --eval-jsonl budget20=math_eval/tale_budget_rolloutlen_hardtrunc_step50_table2_eval_outputs/aime24/rolloutlen-hardtrunc-opd-step50-normalprompt-n32-seed42.jsonl \
  --eval-jsonl normal20=math_eval/tale_budget_rolloutlen_hardtrunc_normalteacher_step50_table2_eval_outputs/aime24/rolloutlen-hardtrunc-normalteacher-opd-step50-normalprompt-n32-seed42.jsonl
```

Expected: `math_eval/opd_training_dynamics_audit/repetition_metrics.csv` exists and includes three runs.

- [ ] **Step 6: Commit Task 3**

Run:

```bash
cd /home/mchen/FiRe-OPD
git add math_eval/opd_training_dynamics_audit.py math_eval/test_opd_training_dynamics_audit.py math_eval/opd_training_dynamics_audit/repetition_metrics.csv
git commit -m "Add OPD audit repetition metrics"
```

---

### Task 4: Tensor-Level Top-k Alignment Metrics

**Files:**
- Modify: `math_eval/opd_training_dynamics_audit.py`
- Modify: `math_eval/test_opd_training_dynamics_audit.py`

**Interfaces:**
- Produces: `compute_topk_alignment_metrics(student_logits, teacher_logits, mask=None, k: int = 16) -> dict[str, float]`
- Produces: `compute_position_binned_alignment(student_logits, teacher_logits, mask=None, k: int = 16, num_bins: int = 8) -> list[dict[str, float]]`

- [ ] **Step 1: Add failing tensor alignment tests**

Append to `math_eval/test_opd_training_dynamics_audit.py`:

```python
import pytest
import torch

from math_eval.opd_training_dynamics_audit import (
    compute_topk_alignment_metrics,
    compute_position_binned_alignment,
)


def test_compute_topk_alignment_metrics_identical_logits_have_full_overlap():
    logits = torch.tensor([[[4.0, 3.0, 1.0, 0.0], [0.0, 1.0, 3.0, 4.0]]])
    metrics = compute_topk_alignment_metrics(logits, logits.clone(), k=2)
    assert metrics["num_positions"] == 2.0
    assert metrics["topk_overlap_ratio"] == 1.0
    assert metrics["entropy_gap"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["overlap_student_mass"] > 0.8
    assert metrics["overlap_teacher_mass"] > 0.8


def test_compute_topk_alignment_metrics_disjoint_top1_has_zero_overlap():
    student = torch.tensor([[[10.0, 0.0, 0.0]]])
    teacher = torch.tensor([[[0.0, 10.0, 0.0]]])
    metrics = compute_topk_alignment_metrics(student, teacher, k=1)
    assert metrics["topk_overlap_ratio"] == 0.0
    assert metrics["overlap_student_mass"] == 0.0
    assert metrics["overlap_teacher_mass"] == 0.0


def test_compute_topk_alignment_metrics_respects_mask():
    student = torch.tensor([[[4.0, 3.0, 0.0], [10.0, 0.0, 0.0]]])
    teacher = torch.tensor([[[4.0, 3.0, 0.0], [0.0, 10.0, 0.0]]])
    mask = torch.tensor([[1, 0]])
    metrics = compute_topk_alignment_metrics(student, teacher, mask=mask, k=1)
    assert metrics["num_positions"] == 1.0
    assert metrics["topk_overlap_ratio"] == 1.0


def test_compute_position_binned_alignment_returns_requested_bins():
    logits = torch.randn(1, 8, 6)
    bins = compute_position_binned_alignment(logits, logits.clone(), k=2, num_bins=4)
    assert len(bins) == 4
    assert bins[0]["position_bin"] == 0.0
    assert bins[-1]["position_bin"] == 3.0
    assert all(row["topk_overlap_ratio"] == pytest.approx(1.0) for row in bins)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /home/mchen/FiRe-OPD
pytest math_eval/test_opd_training_dynamics_audit.py -q
```

Expected: FAIL because alignment functions are not defined.

- [ ] **Step 3: Implement alignment metrics**

Append to `math_eval/opd_training_dynamics_audit.py`:

```python

def compute_topk_alignment_metrics(student_logits, teacher_logits, mask=None, k: int = 16) -> dict[str, float]:
    """Compute 2604.13016-style top-k alignment metrics.

    Inputs are tensors with shape [batch, seq, vocab]. Mask, when provided,
    has shape [batch, seq] with 1 for valid response positions.
    """

    import torch
    import torch.nn.functional as F

    if student_logits.shape != teacher_logits.shape:
        raise ValueError(f"student_logits and teacher_logits must have the same shape, got {student_logits.shape} and {teacher_logits.shape}")
    if student_logits.ndim != 3:
        raise ValueError("logits must have shape [batch, seq, vocab]")
    vocab = student_logits.shape[-1]
    if k <= 0 or k > vocab:
        raise ValueError(f"k must be in [1, vocab], got k={k}, vocab={vocab}")
    if mask is None:
        valid = torch.ones(student_logits.shape[:2], dtype=torch.bool, device=student_logits.device)
    else:
        valid = mask.to(device=student_logits.device).bool()
    if valid.sum().item() == 0:
        return {
            "num_positions": 0.0,
            "topk_overlap_ratio": float("nan"),
            "overlap_student_mass": float("nan"),
            "overlap_teacher_mass": float("nan"),
            "student_entropy": float("nan"),
            "teacher_entropy": float("nan"),
            "entropy_gap": float("nan"),
            "overlap_token_advantage": float("nan"),
        }

    s_logits = student_logits[valid]
    t_logits = teacher_logits[valid]
    s_logp = F.log_softmax(s_logits, dim=-1)
    t_logp = F.log_softmax(t_logits, dim=-1)
    s_prob = s_logp.exp()
    t_prob = t_logp.exp()
    s_top = torch.topk(s_prob, k=k, dim=-1).indices
    t_top = torch.topk(t_prob, k=k, dim=-1).indices
    s_mask = torch.zeros_like(s_prob, dtype=torch.bool).scatter_(dim=-1, index=s_top, value=True)
    t_mask = torch.zeros_like(t_prob, dtype=torch.bool).scatter_(dim=-1, index=t_top, value=True)
    overlap = s_mask & t_mask
    overlap_count = overlap.sum(dim=-1).float()
    overlap_ratio = overlap_count / float(k)
    s_mass = (s_prob * overlap.float()).sum(dim=-1)
    t_mass = (t_prob * overlap.float()).sum(dim=-1)
    s_entropy = -(s_prob * s_logp).sum(dim=-1)
    t_entropy = -(t_prob * t_logp).sum(dim=-1)
    nonempty = overlap_count > 0
    if nonempty.any():
        p_overlap = torch.where(overlap, s_prob, torch.zeros_like(s_prob))[nonempty]
        q_overlap = torch.where(overlap, t_prob, torch.zeros_like(t_prob))[nonempty]
        p_norm = p_overlap / p_overlap.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        q_norm = q_overlap / q_overlap.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        adv = (p_norm * (q_norm.clamp_min(1e-12).log() - p_norm.clamp_min(1e-12).log())).sum(dim=-1)
        overlap_advantage = adv.mean().item()
    else:
        overlap_advantage = float("nan")
    return {
        "num_positions": float(valid.sum().item()),
        "topk_overlap_ratio": overlap_ratio.mean().item(),
        "overlap_student_mass": s_mass.mean().item(),
        "overlap_teacher_mass": t_mass.mean().item(),
        "student_entropy": s_entropy.mean().item(),
        "teacher_entropy": t_entropy.mean().item(),
        "entropy_gap": (t_entropy - s_entropy).abs().mean().item(),
        "overlap_token_advantage": overlap_advantage,
    }


def compute_position_binned_alignment(student_logits, teacher_logits, mask=None, k: int = 16, num_bins: int = 8) -> list[dict[str, float]]:
    import torch

    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    seq_len = student_logits.shape[1]
    if mask is None:
        mask = torch.ones(student_logits.shape[:2], dtype=torch.bool, device=student_logits.device)
    out: list[dict[str, float]] = []
    for bin_idx in range(num_bins):
        start = int(round(seq_len * bin_idx / num_bins))
        end = int(round(seq_len * (bin_idx + 1) / num_bins))
        if end <= start:
            end = min(seq_len, start + 1)
        metrics = compute_topk_alignment_metrics(
            student_logits[:, start:end, :],
            teacher_logits[:, start:end, :],
            mask=mask[:, start:end],
            k=k,
        )
        metrics["position_bin"] = float(bin_idx)
        metrics["position_start"] = float(start)
        metrics["position_end"] = float(end)
        out.append(metrics)
    return out
```

- [ ] **Step 4: Run tests**

Run:

```bash
cd /home/mchen/FiRe-OPD
pytest math_eval/test_opd_training_dynamics_audit.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 4**

Run:

```bash
cd /home/mchen/FiRe-OPD
git add math_eval/opd_training_dynamics_audit.py math_eval/test_opd_training_dynamics_audit.py
git commit -m "Add OPD audit top-k alignment metrics"
```

---

### Task 5: Lazy Model-Scoring CLI for Prompt-Style Alignment

**Files:**
- Modify: `math_eval/opd_training_dynamics_audit.py`
- Modify: `math_eval/test_opd_training_dynamics_audit.py`
- Create generated when run: `math_eval/opd_training_dynamics_audit/alignment_metrics.csv`

**Interfaces:**
- Consumes: `compute_topk_alignment_metrics(...)`
- Produces: `build_teacher_messages_for_audit(question: str, raw_messages: list[dict[str, str]] | None, style: str, budget: int | None) -> list[dict[str, str]]`
- Produces: `score_response_logits(model, tokenizer, messages: list[dict[str, str]], response: str, max_score_tokens: int) -> Any`
- Produces: `run_alignment(args: argparse.Namespace) -> None`
- Extends CLI with `alignment` subcommand.

- [ ] **Step 1: Add failing prompt builder tests**

Append to `math_eval/test_opd_training_dynamics_audit.py`:

```python
from math_eval.opd_training_dynamics_audit import build_teacher_messages_for_audit


def test_build_teacher_messages_for_audit_normal_uses_raw_messages():
    raw = [{"role": "user", "content": "Problem text\nPlease reason step by step, and put your final answer within \\boxed{}."}]
    messages = build_teacher_messages_for_audit("Problem text", raw, style="normal", budget=None)
    assert messages == raw


def test_build_teacher_messages_for_audit_budget_inserts_budget():
    messages = build_teacher_messages_for_audit("Problem text", None, style="budget", budget=512)
    assert "less than 512 tokens" in messages[0]["content"]
    assert "Problem text" in messages[0]["content"]


def test_build_teacher_messages_for_audit_concise_inserts_concise_instruction():
    messages = build_teacher_messages_for_audit("Problem text", None, style="concise", budget=None)
    assert "Solve concisely" in messages[0]["content"]


def test_build_teacher_messages_for_audit_rejects_unknown_style():
    with pytest.raises(ValueError, match="style"):
        build_teacher_messages_for_audit("Problem text", None, style="shortest", budget=None)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /home/mchen/FiRe-OPD
pytest math_eval/test_opd_training_dynamics_audit.py -q
```

Expected: FAIL because `build_teacher_messages_for_audit` is not defined.

- [ ] **Step 3: Implement prompt builder and response scorer**

Append to `math_eval/opd_training_dynamics_audit.py`:

```python

def build_teacher_messages_for_audit(
    question: str,
    raw_messages: list[dict[str, str]] | None,
    style: str,
    budget: int | None,
) -> list[dict[str, str]]:
    if style == "normal":
        if raw_messages is not None:
            return [dict(message) for message in raw_messages]
        return [{"role": "user", "content": question}]
    if style == "budget":
        if budget is None:
            raise ValueError("budget style requires budget")
        from verl.trainer.ppo.tale_budget import build_tale_budget_teacher_messages

        return build_tale_budget_teacher_messages(question=question, budget=int(budget))
    if style == "concise":
        from verl.trainer.ppo.tale_budget import build_concise_teacher_messages

        return build_concise_teacher_messages(question=question)
    raise ValueError(f"style must be one of normal, budget, concise; got {style!r}")


def _apply_chat_template(tokenizer, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def score_response_logits(model, tokenizer, messages: list[dict[str, str]], response: str, max_score_tokens: int):
    """Return logits for response-token positions under prompt+response.

    This is lazy-heavy and intended for small diagnostic subsets. It returns a
    tensor with shape [1, scored_response_tokens, vocab].
    """

    import torch

    prompt_text = _apply_chat_template(tokenizer, messages)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    response_ids = tokenizer(response, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    if max_score_tokens > 0:
        response_ids = response_ids[:, :max_score_tokens]
    input_ids = torch.cat([prompt_ids, response_ids], dim=1)
    with torch.no_grad():
        output = model(input_ids=input_ids)
    prompt_len = prompt_ids.shape[1]
    response_len = response_ids.shape[1]
    if response_len == 0:
        return output.logits[:, :0, :]
    return output.logits[:, prompt_len - 1 : prompt_len - 1 + response_len, :]


def _load_transformers_model(model_path: str, dtype: str, device_map: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"auto": "auto", "bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def _iter_alignment_eval_rows(path: Path, max_prompts: int, max_responses_per_prompt: int):
    with path.open() as f:
        for problem_index, line in enumerate(f):
            if max_prompts >= 0 and problem_index >= max_prompts:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            problem = str(row.get("problem", row.get("question", "")))
            responses = list(row.get("responses", []))[:max_responses_per_prompt]
            lengths = row.get("response_lengths", [])
            for sample_index, response in enumerate(responses):
                yield {
                    "problem_index": problem_index,
                    "sample_index": sample_index,
                    "question": problem,
                    "response": str(response),
                    "budget": int(lengths[sample_index]) if sample_index < len(lengths) else None,
                }


def run_alignment(args: argparse.Namespace) -> None:
    import torch

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    student_model, student_tokenizer = _load_transformers_model(args.student_model, args.dtype, args.device_map)
    teacher_model, teacher_tokenizer = _load_transformers_model(args.teacher_model, args.dtype, args.device_map)
    rows: list[dict[str, float | str]] = []
    for item in _iter_alignment_eval_rows(Path(args.responses_jsonl), args.max_prompts, args.max_responses_per_prompt):
        student_messages = [{"role": "user", "content": item["question"]}]
        student_logits = score_response_logits(
            student_model,
            student_tokenizer,
            student_messages,
            item["response"],
            max_score_tokens=args.max_score_tokens,
        )
        for style in args.teacher_prompt_style:
            teacher_messages = build_teacher_messages_for_audit(
                question=item["question"],
                raw_messages=student_messages,
                style=style,
                budget=item["budget"],
            )
            teacher_logits = score_response_logits(
                teacher_model,
                teacher_tokenizer,
                teacher_messages,
                item["response"],
                max_score_tokens=args.max_score_tokens,
            )
            seq = min(student_logits.shape[1], teacher_logits.shape[1])
            metrics = compute_topk_alignment_metrics(
                student_logits[:, :seq, :].to(dtype=torch.float32),
                teacher_logits[:, :seq, :].to(dtype=torch.float32),
                k=args.top_k,
            )
            metrics.update(
                {
                    "teacher_prompt_style": style,
                    "problem_index": float(item["problem_index"]),
                    "sample_index": float(item["sample_index"]),
                    "scored_tokens": float(seq),
                }
            )
            rows.append(metrics)
    write_csv(rows, out_dir / "alignment_metrics.csv")
    print(f"wrote {out_dir / 'alignment_metrics.csv'}")
```

Modify `build_arg_parser()` to include:

```python
    alignment = sub.add_parser("alignment", help="score fixed responses for student-teacher top-k alignment")
    alignment.add_argument("--output-dir", default="math_eval/opd_training_dynamics_audit")
    alignment.add_argument("--responses-jsonl", required=True)
    alignment.add_argument("--student-model", required=True)
    alignment.add_argument("--teacher-model", required=True)
    alignment.add_argument("--teacher-prompt-style", action="append", choices=["normal", "budget", "concise"], required=True)
    alignment.add_argument("--top-k", type=int, default=16)
    alignment.add_argument("--max-prompts", type=int, default=4)
    alignment.add_argument("--max-responses-per-prompt", type=int, default=1)
    alignment.add_argument("--max-score-tokens", type=int, default=512)
    alignment.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    alignment.add_argument("--device-map", default="auto")
    alignment.set_defaults(func=run_alignment)
```

- [ ] **Step 4: Run tests**

Run:

```bash
cd /home/mchen/FiRe-OPD
pytest math_eval/test_opd_training_dynamics_audit.py -q
```

Expected: PASS. These tests do not load models.

- [ ] **Step 5: Run a tiny alignment smoke if GPUs are free**

Run only if no active training/eval job is using the target GPUs:

```bash
cd /home/mchen/FiRe-OPD
CUDA_VISIBLE_DEVICES=0 python math_eval/opd_training_dynamics_audit.py alignment \
  --output-dir math_eval/opd_training_dynamics_audit \
  --responses-jsonl math_eval/tale_budget_rolloutlen_hardtrunc_normalteacher_step50_table2_eval_outputs/aime24/rolloutlen-hardtrunc-normalteacher-opd-step50-normalprompt-n32-seed42.jsonl \
  --student-model models/Qwen3-4B \
  --teacher-model models/Qwen3-30B-A3B-Instruct-2507 \
  --teacher-prompt-style normal \
  --teacher-prompt-style budget \
  --teacher-prompt-style concise \
  --max-prompts 2 \
  --max-responses-per-prompt 1 \
  --max-score-tokens 256 \
  --top-k 16 \
  --dtype bfloat16 \
  --device-map auto
```

Expected: `math_eval/opd_training_dynamics_audit/alignment_metrics.csv` exists. If memory is insufficient, skip this smoke and record the failure in `summary.md` under limitations; do not start training or kill jobs.

- [ ] **Step 6: Commit Task 5**

Run:

```bash
cd /home/mchen/FiRe-OPD
git add math_eval/opd_training_dynamics_audit.py math_eval/test_opd_training_dynamics_audit.py
git add math_eval/opd_training_dynamics_audit/alignment_metrics.csv 2>/dev/null || true
git commit -m "Add OPD audit alignment scorer"
```

---

### Task 6: Final Report Assembly and Interpretation

**Files:**
- Modify generated: `math_eval/opd_training_dynamics_audit/summary.md`
- Create generated: `math_eval/opd_training_dynamics_audit/raw_opd_inflation_diagnostics.svg`
- Create generated: `math_eval/opd_training_dynamics_audit/prompt_style_alignment_metrics.svg` if alignment metrics exist

**Interfaces:**
- Consumes: `metrics_by_step.csv`, `repetition_metrics.csv`, optional `alignment_metrics.csv`
- Produces: final report with evidence-backed conclusions and limitations.

- [ ] **Step 1: Add focused diagnostic SVG helpers**

Append to `math_eval/opd_training_dynamics_audit.py`:

```python

def render_single_run_svg(rows: list[dict[str, float | str]], run_name: str, path: Path) -> None:
    selected = [row for row in rows if row.get("run_name") == run_name]
    render_macro_svg(selected, path)


def render_alignment_svg(alignment_rows: list[dict[str, float | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 760, 420
    styles = sorted({str(row.get("teacher_prompt_style", "unknown")) for row in alignment_rows})
    metrics = ["topk_overlap_ratio", "overlap_student_mass", "student_entropy", "teacher_entropy", "entropy_gap"]
    means: dict[tuple[str, str], float] = {}
    for style in styles:
        style_rows = [row for row in alignment_rows if row.get("teacher_prompt_style") == style]
        for metric in metrics:
            vals = [float(row[metric]) for row in style_rows if metric in row and not math.isnan(float(row[metric]))]
            means[(style, metric)] = statistics.fmean(vals) if vals else float("nan")
    bar_w = 28
    group_gap = 34
    x0, y0 = 80, 330
    max_val = max([value for value in means.values() if not math.isnan(value)] + [1.0])
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933}.title{font-size:20px;font-weight:700}.axis{font-size:12px;fill:#52616b}</style>',
        '<text x="380" y="32" text-anchor="middle" class="title">Teacher prompt style alignment metrics</text>',
        f'<line x1="{x0}" y1="{y0}" x2="700" y2="{y0}" stroke="#1f2933"/>',
    ]
    colors = {"normal": "#2ca02c", "budget": "#d62728", "concise": "#ff7f0e"}
    x = x0
    for metric in metrics:
        lines.append(f'<text x="{x + 35}" y="{y0 + 42}" text-anchor="middle" class="axis">{metric}</text>')
        for style in styles:
            value = means[(style, metric)]
            if math.isnan(value):
                height_px = 0.0
            else:
                height_px = 250.0 * value / max_val
            lines.append(f'<rect x="{x}" y="{y0 - height_px:.1f}" width="{bar_w}" height="{height_px:.1f}" fill="{colors.get(style, "#64748b")}"/>')
            x += bar_w + 3
        x += group_gap
    lx, ly = 560, 70
    for style in styles:
        lines.append(f'<rect x="{lx}" y="{ly}" width="14" height="14" fill="{colors.get(style, "#64748b")}"/>')
        lines.append(f'<text x="{lx + 20}" y="{ly + 12}" class="axis">{style}</text>')
        ly += 22
    lines.append("</svg>")
    path.write_text("\n".join(lines))
```

- [ ] **Step 2: Generate final artifacts**

Run:

```bash
cd /home/mchen/FiRe-OPD
python math_eval/opd_training_dynamics_audit.py macro \
  --repo-dir /home/mchen/FiRe-OPD \
  --output-dir math_eval/opd_training_dynamics_audit
python math_eval/opd_training_dynamics_audit.py repetition \
  --output-dir math_eval/opd_training_dynamics_audit \
  --max-token-length 16384 \
  --eval-jsonl raw_opd=math_eval/opd_rawprompt_step_eval_outputs/aime24/opd-strong-to-weak-rawprompt-4gpu-tp4-refmb4-rollmb4-step50-baseline-n32-seed42.jsonl \
  --eval-jsonl budget20=math_eval/tale_budget_rolloutlen_hardtrunc_step50_table2_eval_outputs/aime24/rolloutlen-hardtrunc-opd-step50-normalprompt-n32-seed42.jsonl \
  --eval-jsonl normal20=math_eval/tale_budget_rolloutlen_hardtrunc_normalteacher_step50_table2_eval_outputs/aime24/rolloutlen-hardtrunc-normalteacher-opd-step50-normalprompt-n32-seed42.jsonl
```

Expected: macro and repetition outputs are refreshed.

- [ ] **Step 3: Manually update `summary.md` with observed numbers**

Open `math_eval/opd_training_dynamics_audit/metrics_by_step.csv` and `repetition_metrics.csv`, then edit `summary.md` so it contains these sections:

```markdown
## Findings

### Raw OPD length inflation

- Log-level evidence: raw OPD original rollout length rises from about 1.5K at step1 to a peak near step18-20.
- This is consistent with 2604.08527's truncation-repetition inflation mechanism.
- If step18/19 checkpoints are unavailable, this report does not claim direct repeated-token advantage proof at the transition step.

### Concise-teacher collapse

- Log-level evidence: concise-teacher hardtrunc original rollout length falls sharply by step20, while actor entropy also falls.
- This is consistent with 2604.13016's low-entropy prompt-alignment collapse risk.
- Alignment metrics, if present, compare normal, budget, and concise teacher prompts on fixed student responses.

### Rollout-length budget stability

- Log-level evidence: rollout-length-budget hardtrunc original rollout length grows gradually rather than exploding or collapsing.
- The supervised length remains around 20% of original rollout length by construction.
- This supports the interpretation that student-length budgets are anchored to the current student policy rather than acting as a shorter-than-student target.

## Next metrics to log in reruns

- compression-based repetition ratio;
- repeated-token vs regular-token reverse-KL advantage;
- top-k overlap ratio;
- student and teacher entropy gap;
- position-binned entropy.
```

Replace the approximate values with exact values from generated CSVs.

- [ ] **Step 4: Commit Task 6**

Run:

```bash
cd /home/mchen/FiRe-OPD
git add math_eval/opd_training_dynamics_audit.py math_eval/opd_training_dynamics_audit/summary.md math_eval/opd_training_dynamics_audit/*.csv math_eval/opd_training_dynamics_audit/*.svg
git commit -m "Write OPD training dynamics audit report"
```

---

### Task 7: Verification and Handoff to Sweep Design

**Files:**
- Read generated: `math_eval/opd_training_dynamics_audit/summary.md`
- Read generated: `math_eval/opd_training_dynamics_audit/metrics_by_step.csv`
- Read generated: `math_eval/opd_training_dynamics_audit/repetition_metrics.csv`
- Optional read generated: `math_eval/opd_training_dynamics_audit/alignment_metrics.csv`

**Interfaces:**
- Consumes all prior task outputs.
- Produces final user-facing summary and recommendation for next sweep.

- [ ] **Step 1: Run unit tests**

Run:

```bash
cd /home/mchen/FiRe-OPD
pytest math_eval/test_opd_training_dynamics_audit.py -q
```

Expected: PASS.

- [ ] **Step 2: Verify no training process was launched by this audit**

Run:

```bash
ps -u mchen -o pid,etime,cmd | grep -E 'main_ppo|ray::TaskRunner|run_train_' | grep -v grep || true
```

Expected: no new training process attributable to this audit. Existing unrelated processes may appear; do not kill them.

- [ ] **Step 3: Verify generated report exists and names the limitation**

Run:

```bash
grep -n "step18/19" math_eval/opd_training_dynamics_audit/summary.md
grep -n "response_length/mean" math_eval/opd_training_dynamics_audit/summary.md
```

Expected: output lines explain the step18/19 checkpoint limitation and hardtrunc length caveat.

- [ ] **Step 4: Inspect git diff for accidental training-code changes**

Run:

```bash
cd /home/mchen/FiRe-OPD
git diff --stat HEAD
```

Expected: only `math_eval/opd_training_dynamics_audit.py`, `math_eval/test_opd_training_dynamics_audit.py`, and generated `math_eval/opd_training_dynamics_audit/*` should be part of the current audit commits.

- [ ] **Step 5: Final response to user**

Report:

```text
Offline audit complete.
Main artifacts:
- math_eval/opd_training_dynamics_audit/summary.md
- math_eval/opd_training_dynamics_audit/macro_length_score_entropy_grad.svg
- math_eval/opd_training_dynamics_audit/metrics_by_step.csv
- math_eval/opd_training_dynamics_audit/repetition_metrics.csv

Main conclusion:
[one paragraph from summary.md]

Recommended next step:
normal-teacher prefix-supervision sweep x={0.1,0.2,0.4,0.6,0.8,1.0}, step50 fixed, with x=1.0 as raw OPD endpoint.
```

- [ ] **Step 6: Commit any final report adjustments**

Run only if Step 5 revealed wording changes needed:

```bash
cd /home/mchen/FiRe-OPD
git add math_eval/opd_training_dynamics_audit/summary.md
git commit -m "Refine OPD audit summary"
```
