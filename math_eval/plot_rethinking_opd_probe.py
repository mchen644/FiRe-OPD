#!/usr/bin/env python3
"""Plot training-time Rethinking OPD probe CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def load_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open() as f:
            for row in csv.DictReader(f):
                rows.append(row)
    return rows


def write_summary(rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["run_name"], row["chunk_start"])
        grouped.setdefault(key, []).append(row)
    with output.open("w") as f:
        f.write("# Rethinking OPD Training Probe Summary\n\n")
        for (run, chunk), group in sorted(grouped.items()):
            last = sorted(group, key=lambda r: int(float(r["step"])))[-1]
            f.write(
                f"- {run} chunk {chunk}: "
                f"last_step={last['step']}, "
                f"overlap={_float(last['topk_overlap_ratio']):.4f}, "
                f"student_entropy={_float(last['student_entropy']):.4f}, "
                f"entropy_gap={_float(last['entropy_gap']):.4f}, "
                f"valid_tokens={last['valid_token_count']}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--summary", type=Path, default=Path("math_eval/opd_training_dynamics_audit/training_probe_summary.md"))
    args = parser.parse_args()
    rows = load_rows(args.csv)
    write_summary(rows, args.summary)
    print(f"wrote {args.summary}")


if __name__ == "__main__":
    main()
