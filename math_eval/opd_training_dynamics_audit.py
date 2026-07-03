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
        "step",
        "run_group",
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
        valid = [
            row
            for row in group_rows
            if "original_rollout_length_mean" in row and not math.isnan(float(row["original_rollout_length_mean"]))
        ]
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


def compute_topk_alignment_metrics(student_logits, teacher_logits, mask=None, k: int = 16) -> dict[str, float]:
    """Compute 2604.13016-style top-k alignment metrics.

    Inputs are tensors with shape [batch, seq, vocab]. Mask, when provided,
    has shape [batch, seq] with 1 for valid response positions.
    """

    import torch
    import torch.nn.functional as F

    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"student_logits and teacher_logits must have the same shape, got {student_logits.shape} and {teacher_logits.shape}"
        )
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
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
