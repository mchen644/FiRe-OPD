#!/usr/bin/env python3
"""Continuous teacher-signal probe for BAGEN-style OPD budget awareness.

This probe asks a teacher to score whether a *specific* student rollout prefix
will eventually finish correctly. It is intended for the hard-for-student AIME
subset where fixed 20% supervision loses most of its accuracy relative to raw
OPD.

Unlike math_eval/bagen_opd_probe.py, this script uses a continuous probability
answer so we can compute AUROC/AUPRC and within-problem correlations instead of
only sparse possible/impossible labels.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from bagen_opd_probe import (
    ProbeExample,
    _apply_chat_template,
    _format_percent,
    _load_tokenizer,
    _write_json,
    _write_jsonl,
    flatten_eval_outputs,
)

_ANSWER_RE = re.compile(
    r"(?:<|\[)answer(?:>|\])\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*%?\s*(?:(?:<|\[)/answer(?:>|\]))?",
    re.IGNORECASE | re.DOTALL,
)
_THINK_RE = re.compile(r"(?:<|\[)think(?:>|\])\s*(.*?)\s*(?:<|\[)/think(?:>|\])", re.IGNORECASE | re.DOTALL)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_problem_accs(path: str | Path) -> list[float]:
    accs: list[float] = []
    for row in _read_jsonl(path):
        acc_list = [bool(value) for value in row.get("acc_list", [])]
        accs.append(sum(acc_list) / len(acc_list) if acc_list else 0.0)
    return accs


def select_problem_indices(
    accs: Sequence[float],
    *,
    min_acc_exclusive: float,
    max_acc_inclusive: float,
) -> list[int]:
    return [
        idx
        for idx, acc in enumerate(accs)
        if float(acc) > float(min_acc_exclusive) and float(acc) <= float(max_acc_inclusive)
    ]


def build_signal_messages(
    *,
    problem: str,
    prefix_text: str,
    student_budget_tokens: int,
    prefix_tokens: int,
    remaining_budget_tokens: int,
    prefix_fraction: float,
) -> list[dict[str, str]]:
    """Build a BAGEN-like teacher prompt with short reasoning + probability."""
    prefix_percent = _format_percent(prefix_fraction)
    system = (
        "You are an evaluation agent. Read the provided math student rollout context "
        "and estimate whether this same student trajectory will finish with a correct "
        "final answer within the student's total token budget. Follow the required "
        "output format exactly."
    )
    user = f"""Based on the provided rollout context, you are provided below information:
1. This is a single-turn math solution attempt.
2. The student has completed the first {prefix_percent}% of its solution prefix.
3. The student total rollout budget is {int(student_budget_tokens)} tokens.
4. The student has used about {int(prefix_tokens)} tokens so far.
5. The remaining student budget is about {int(remaining_budget_tokens)} tokens.

Problem:
{problem}

Student partial solution prefix:
```text
{prefix_text}
```

Now, estimate whether this exact student trajectory will eventually produce the correct final answer if it continues within the remaining budget.
Do not solve the problem. Do not continue the student's solution. Judge the current direction, mathematical soundness, and whether the remaining budget is enough.

Output a calibrated probability from 0 to 100 that the final answer of this same student trajectory will be correct.
Use 0 for certainly wrong/impossible to recover, and 100 for certainly correct/on track.

Required output format, with no extra prose and no markdown brackets:
<think>one short sentence, less than 30 words</think><answer>NUMBER</answer>

The answer tag must contain only one number from 0 to 100. Do not include a percent sign."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_probability_answer(text: str) -> float | None:
    response = str(text or "").strip()
    match = _ANSWER_RE.search(response)
    raw_value = match.group(1) if match else None
    if raw_value is None:
        # Some models obey the numeric part but omit tags, e.g. end with a bare "10".
        # Only accept a final standalone number to avoid accidentally parsing math in the rationale.
        fallback = re.search(r"(?<![A-Za-z0-9])([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*%?\s*$", response)
        raw_value = fallback.group(1) if fallback else None
    if raw_value is None:
        return None
    try:
        value = float(raw_value)
    except ValueError:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return max(0.0, min(100.0, value))


def parse_think(text: str) -> str | None:
    match = _THINK_RE.search(str(text or ""))
    return match.group(1).strip() if match else None


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) == 0:
        return float("nan")
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy) if vx and vy else float("nan")


def _rankdata(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    pos = [score for label, score in zip(labels, scores) if int(label) == 1]
    neg = [score for label, score in zip(labels, scores) if int(label) == 0]
    if not pos or not neg:
        return float("nan")
    greater = 0
    ties = 0
    for ps in pos:
        for ns in neg:
            if ps > ns:
                greater += 1
            elif ps == ns:
                ties += 1
    return (greater + 0.5 * ties) / (len(pos) * len(neg))


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    n_pos = sum(int(label) == 1 for label in labels)
    if n_pos == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    hits = 0
    precision_sum = 0.0
    for rank, idx in enumerate(order, start=1):
        if int(labels[idx]) == 1:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / n_pos


def compute_signal_metrics(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    all_rows = list(records)
    valid = [row for row in all_rows if row.get("score_0_to_100") is not None]
    labels = [1 if bool(row.get("student_correct")) else 0 for row in valid]
    scores = [float(row["score_0_to_100"]) / 100.0 for row in valid]

    n = len(valid)
    n_pos = sum(labels)
    n_neg = n - n_pos
    metrics: dict[str, Any] = {
        "n": n,
        "n_total_rows": len(all_rows),
        "invalid_prediction_count": len(all_rows) - n,
        "n_correct": n_pos,
        "n_wrong": n_neg,
        "positive_rate": n_pos / n if n else 0.0,
        "auroc": auroc(labels, scores) if n else float("nan"),
        "auprc": average_precision(labels, scores) if n else float("nan"),
        "point_biserial": _pearson([float(x) for x in labels], scores) if n else float("nan"),
        "score_mean": sum(scores) / n if n else float("nan"),
        "score_mean_correct": sum(s for y, s in zip(labels, scores) if y == 1) / n_pos if n_pos else float("nan"),
        "score_mean_wrong": sum(s for y, s in zip(labels, scores) if y == 0) / n_neg if n_neg else float("nan"),
    }

    # Problem-fixed metrics: does the score distinguish successful vs failed
    # rollouts of the same hard problem, rather than merely problem difficulty?
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in valid:
        grouped.setdefault(int(row["problem_index"]), []).append(row)

    per_problem: list[dict[str, Any]] = []
    residual_labels: list[float] = []
    residual_scores: list[float] = []
    for problem_index, rows in sorted(grouped.items()):
        ys = [1 if bool(row.get("student_correct")) else 0 for row in rows]
        ss = [float(row["score_0_to_100"]) / 100.0 for row in rows]
        pos = sum(ys)
        neg = len(ys) - pos
        row_metrics = {
            "problem_index": problem_index,
            "n": len(rows),
            "n_correct": pos,
            "n_wrong": neg,
            "auroc": auroc(ys, ss) if pos and neg else float("nan"),
            "point_biserial": _pearson([float(y) for y in ys], ss) if pos and neg else float("nan"),
            "score_mean_correct": sum(s for y, s in zip(ys, ss) if y == 1) / pos if pos else float("nan"),
            "score_mean_wrong": sum(s for y, s in zip(ys, ss) if y == 0) / neg if neg else float("nan"),
        }
        per_problem.append(row_metrics)
        y_mean = sum(ys) / len(ys)
        s_mean = sum(ss) / len(ss)
        residual_labels.extend([float(y) - y_mean for y in ys])
        residual_scores.extend([s - s_mean for s in ss])

    finite_auc = [float(row["auroc"]) for row in per_problem if not math.isnan(float(row["auroc"]))]
    finite_pb = [float(row["point_biserial"]) for row in per_problem if not math.isnan(float(row["point_biserial"]))]
    metrics.update(
        {
            "n_problems": len(grouped),
            "within_problem_macro_auroc": sum(finite_auc) / len(finite_auc) if finite_auc else float("nan"),
            "within_problem_macro_point_biserial": sum(finite_pb) / len(finite_pb) if finite_pb else float("nan"),
            "within_problem_residual_point_biserial": _pearson(residual_labels, residual_scores) if residual_labels else float("nan"),
            "per_problem": per_problem,
        }
    )
    return metrics


def run_teacher_signal_probe(
    *,
    examples: Sequence[ProbeExample],
    teacher_model: str,
    tensor_parallel_size: int | None,
    max_model_len: int | None,
    max_num_seqs: int,
    gpu_memory_utilization: float,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model)
    prompts: list[str] = []
    messages_list: list[list[dict[str, str]]] = []
    for example in examples:
        messages = build_signal_messages(
            problem=example.problem,
            prefix_text=example.prefix_text,
            student_budget_tokens=example.student_budget_tokens,
            prefix_tokens=example.prefix_tokens,
            remaining_budget_tokens=example.remaining_budget_tokens,
            prefix_fraction=example.prefix_fraction,
        )
        messages_list.append(messages)
        prompts.append(_apply_chat_template(teacher_tokenizer, messages))

    tp_size = int(tensor_parallel_size or torch.cuda.device_count() or 1)
    llm_kwargs: dict[str, Any] = {
        "model": teacher_model,
        "tokenizer": teacher_model,
        "tensor_parallel_size": tp_size,
        "max_num_seqs": int(max_num_seqs),
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "enforce_eager": True,
    }
    if max_model_len is not None and int(max_model_len) > 0:
        llm_kwargs["max_model_len"] = int(max_model_len)

    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        temperature=float(temperature),
        top_p=float(top_p),
        max_tokens=int(max_tokens),
        n=1,
        seed=int(seed),
    )
    generations = llm.generate(prompts, sampling_params=sampling_params)

    records: list[dict[str, Any]] = []
    for example, messages, generation in zip(examples, messages_list, generations):
        text = generation.outputs[0].text.strip() if generation.outputs else ""
        score = parse_probability_answer(text)
        record = example.to_record()
        record.update(
            {
                "messages": messages,
                "prediction_text": text,
                "think": parse_think(text),
                "score_0_to_100": score,
            }
        )
        records.append(record)
    return records


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", required=True, help="Student eval output JSONL with responses/acc_list")
    parser.add_argument("--hard-source-file", required=True, help="Eval JSONL used to select hard problem indices, usually base student")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--dataset", default="aime24")
    parser.add_argument("--tokenizer-path", default="/home/mchen/FiRe-OPD/models/Qwen3-4B")
    parser.add_argument("--teacher-model", default="/home/mchen/FiRe-OPD/models/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--prefix-fraction", type=float, default=0.20)
    parser.add_argument("--min-hard-acc-exclusive", type=float, default=0.0)
    parser.add_argument("--max-hard-acc-inclusive", type=float, default=0.25)
    parser.add_argument("--max-rollouts-per-problem", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    tokenizer = _load_tokenizer(str(args.tokenizer_path))
    hard_accs = load_problem_accs(args.hard_source_file)
    hard_indices = set(
        select_problem_indices(
            hard_accs,
            min_acc_exclusive=float(args.min_hard_acc_exclusive),
            max_acc_inclusive=float(args.max_hard_acc_inclusive),
        )
    )
    max_rollouts = args.max_rollouts_per_problem
    if max_rollouts is not None and int(max_rollouts) <= 0:
        max_rollouts = None

    all_examples = flatten_eval_outputs(
        input_path=args.input_file,
        tokenizer=tokenizer,
        prefix_fractions=[float(args.prefix_fraction)],
        dataset=str(args.dataset),
        max_rollouts_per_problem=max_rollouts,
    )
    examples = [example for example in all_examples if int(example.problem_index) in hard_indices]
    print(f"Selected hard problem indices: {sorted(hard_indices)}")
    print(f"Built {len(examples)} hard-subset signal examples from {args.input_file}")

    if args.dry_run:
        records: list[dict[str, Any]] = []
        for example in examples:
            messages = build_signal_messages(
                problem=example.problem,
                prefix_text=example.prefix_text,
                student_budget_tokens=example.student_budget_tokens,
                prefix_tokens=example.prefix_tokens,
                remaining_budget_tokens=example.remaining_budget_tokens,
                prefix_fraction=example.prefix_fraction,
            )
            record = example.to_record()
            record.update({"messages": messages, "prediction_text": None, "think": None, "score_0_to_100": None})
            records.append(record)
    else:
        records = run_teacher_signal_probe(
            examples=examples,
            teacher_model=str(args.teacher_model),
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )

    _write_jsonl(Path(args.output_jsonl), records)
    summary = {
        "input_file": str(args.input_file),
        "hard_source_file": str(args.hard_source_file),
        "teacher_model": None if args.dry_run else str(args.teacher_model),
        "tokenizer_path": str(args.tokenizer_path),
        "prefix_fraction": float(args.prefix_fraction),
        "min_hard_acc_exclusive": float(args.min_hard_acc_exclusive),
        "max_hard_acc_inclusive": float(args.max_hard_acc_inclusive),
        "hard_problem_indices": sorted(hard_indices),
        "hard_problem_accs": {str(idx): hard_accs[idx] for idx in sorted(hard_indices)},
        "metrics": compute_signal_metrics(records),
    }
    _write_json(Path(args.summary_json), summary)
    print(f"Wrote records: {args.output_jsonl}")
    print(f"Wrote summary: {args.summary_json}")
    if not args.dry_run:
        print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
