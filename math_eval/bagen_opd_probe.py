#!/usr/bin/env python3
"""BAGEN-style budget-awareness probe for OPD student rollouts.

This script replays prefixes of previously generated student math rollouts and
asks the current teacher model whether the student can still finish correctly
within the student's own remaining rollout budget.

The first intended experiment is:
  - dataset: AIME 2024
  - student rollouts: base 4B normal-prompt eval outputs
  - prefixes: 10%, 20%, 40%
  - teacher: Qwen3-30B-A3B-Instruct-2507
  - output: <answer>possible</answer> or <answer>impossible</answer>

The summary follows BAGEN-style feasibility metrics: macro-F1, fail-F1
(impossible-class F1), false-abort rate, false-continue rate, MCC, and token
saving proxies if "impossible" would trigger early stop/drop after the prefix.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

_LABELS = {"possible", "impossible"}
_ANSWER_RE = re.compile(r"<answer>\s*(possible|impossible)\s*</answer>", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class ProbeExample:
    probe_id: str
    dataset: str
    problem_index: int
    sample_index: int
    prefix_fraction: float
    prefix_percent: int
    problem: str
    prefix_text: str
    student_budget_tokens: int
    prefix_tokens: int
    remaining_budget_tokens: int
    student_correct: bool
    gold_label: str
    saved_response_length: int | None = None

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def _format_percent(prefix_fraction: float) -> int:
    return int(round(float(prefix_fraction) * 100))


def _ceil_fraction_count(total: int, fraction: float) -> int:
    total = max(0, int(total))
    if total == 0:
        return 0
    return min(total, max(1, int(math.ceil(float(fraction) * float(total)))))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _tokenize_response(tokenizer: Any, response: str) -> list[int]:
    return list(tokenizer.encode(response, add_special_tokens=False))


def _decode_tokens(tokenizer: Any, token_ids: Sequence[int]) -> str:
    return tokenizer.decode(list(token_ids), skip_special_tokens=True).strip()


def flatten_eval_outputs(
    *,
    input_path: str | Path,
    tokenizer: Any,
    prefix_fractions: Sequence[float],
    dataset: str = "aime24",
    max_rollouts_per_problem: int | None = None,
    max_samples: int | None = None,
) -> list[ProbeExample]:
    """Flatten math eval output JSONL into BAGEN-style prefix probe examples."""
    rows = _read_jsonl(Path(input_path))
    examples: list[ProbeExample] = []

    normalized_prefixes = [float(value) for value in prefix_fractions]
    for value in normalized_prefixes:
        if value <= 0.0 or value >= 1.0:
            raise ValueError(f"prefix fractions must be in (0, 1), got {value}")

    for problem_index, row in enumerate(rows):
        problem = str(row.get("problem") or "")
        responses = list(row.get("responses") or [])
        acc_list = list(row.get("acc_list") or [])
        response_lengths = list(row.get("response_lengths") or [])
        if not responses:
            continue

        n_responses = len(responses)
        if max_rollouts_per_problem is not None and int(max_rollouts_per_problem) > 0:
            n_responses = min(n_responses, int(max_rollouts_per_problem))

        for sample_index in range(n_responses):
            response = str(responses[sample_index] or "")
            token_ids = _tokenize_response(tokenizer, response)
            saved_len = _safe_int(response_lengths[sample_index]) if sample_index < len(response_lengths) else None
            student_budget_tokens = int(saved_len) if saved_len is not None and saved_len > 0 else len(token_ids)
            student_budget_tokens = max(0, student_budget_tokens)
            student_correct = bool(acc_list[sample_index]) if sample_index < len(acc_list) else False
            gold_label = "possible" if student_correct else "impossible"

            for prefix_fraction in normalized_prefixes:
                token_prefix_count = _ceil_fraction_count(len(token_ids), prefix_fraction)
                budget_prefix_count = _ceil_fraction_count(student_budget_tokens, prefix_fraction)
                prefix_text = _decode_tokens(tokenizer, token_ids[:token_prefix_count]) if token_prefix_count else ""
                remaining_budget_tokens = max(0, int(student_budget_tokens) - int(budget_prefix_count))
                prefix_percent = _format_percent(prefix_fraction)
                examples.append(
                    ProbeExample(
                        probe_id=f"{dataset}-p{problem_index:04d}-s{sample_index:02d}-pref{prefix_percent:02d}",
                        dataset=dataset,
                        problem_index=problem_index,
                        sample_index=sample_index,
                        prefix_fraction=float(prefix_fraction),
                        prefix_percent=prefix_percent,
                        problem=problem,
                        prefix_text=prefix_text,
                        student_budget_tokens=int(student_budget_tokens),
                        prefix_tokens=int(budget_prefix_count),
                        remaining_budget_tokens=int(remaining_budget_tokens),
                        student_correct=student_correct,
                        gold_label=gold_label,
                        saved_response_length=saved_len,
                    )
                )
                if max_samples is not None and int(max_samples) > 0 and len(examples) >= int(max_samples):
                    return examples
    return examples


def build_probe_messages(
    *,
    problem: str,
    prefix_text: str,
    student_budget_tokens: int,
    prefix_tokens: int,
    remaining_budget_tokens: int,
    prefix_fraction: float,
) -> list[dict[str, str]]:
    """Build teacher probe messages for one prefix."""
    prefix_percent = _format_percent(prefix_fraction)
    system = (
        "You are a careful math teacher evaluating a student's partial solution. "
        "Your job is not to solve the problem for the student, but to judge whether "
        "the current partial solution can plausibly still lead to a correct final answer "
        "within the student's own remaining token budget."
    )
    user = f"""We are probing budget-aware OPD supervision.

Problem:
{problem}

Student total rollout budget: {int(student_budget_tokens)} tokens
Student has used about {int(prefix_tokens)} tokens ({prefix_percent}%)
Remaining student budget: {int(remaining_budget_tokens)} tokens

Student partial solution prefix:
```text
{prefix_text}
```

Question: Given only this problem, the student's partial solution prefix, and the student's remaining budget, is it plausible that this same student can finish a complete correct solution within the remaining budget?

Use <answer>impossible</answer> if the prefix has a fatal mathematical error, is drifting away from a viable solution, is too incomplete for the remaining budget, or otherwise makes a correct finish unlikely.
Use <answer>possible</answer> if the prefix is compatible with a viable correct solution within the remaining budget.

Output exactly one of:
<answer>possible</answer>
<answer>impossible</answer>

Do not solve the problem. Do not continue the solution. Do not output anything outside the answer tag."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_probe_answer(text: str) -> str | None:
    """Parse a strict possible/impossible answer tag."""
    match = _ANSWER_RE.search(str(text or ""))
    if not match:
        return None
    label = match.group(1).strip().lower()
    return label if label in _LABELS else None


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def compute_bagen_style_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Compute BAGEN-style binary feasibility metrics.

    gold_label='possible' means the student final rollout was correct.
    gold_label='impossible' means the student final rollout was wrong.
    prediction is the teacher's parsed possible/impossible answer.
    """
    all_rows = list(rows)
    valid_rows = [
        row for row in all_rows
        if str(row.get("gold_label", "")).lower() in _LABELS
        and str(row.get("prediction", row.get("parsed_prediction", ""))).lower() in _LABELS
    ]

    tp_possible = sum(1 for row in valid_rows if row["gold_label"] == "possible" and row.get("prediction", row.get("parsed_prediction")) == "possible")
    fp_possible = sum(1 for row in valid_rows if row["gold_label"] == "impossible" and row.get("prediction", row.get("parsed_prediction")) == "possible")
    fn_possible = sum(1 for row in valid_rows if row["gold_label"] == "possible" and row.get("prediction", row.get("parsed_prediction")) == "impossible")

    tp_impossible = sum(1 for row in valid_rows if row["gold_label"] == "impossible" and row.get("prediction", row.get("parsed_prediction")) == "impossible")
    fp_impossible = sum(1 for row in valid_rows if row["gold_label"] == "possible" and row.get("prediction", row.get("parsed_prediction")) == "impossible")
    fn_impossible = sum(1 for row in valid_rows if row["gold_label"] == "impossible" and row.get("prediction", row.get("parsed_prediction")) == "possible")

    precision_possible, recall_possible, possible_f1 = _prf(tp_possible, fp_possible, fn_possible)
    precision_impossible, recall_impossible, fail_f1 = _prf(tp_impossible, fp_impossible, fn_impossible)

    n_valid = len(valid_rows)
    n_possible = sum(1 for row in valid_rows if row["gold_label"] == "possible")
    n_impossible = sum(1 for row in valid_rows if row["gold_label"] == "impossible")
    correct = tp_possible + tp_impossible
    accuracy = correct / n_valid if n_valid else 0.0

    # MCC with impossible as the positive class.
    tn = tp_possible
    tp = tp_impossible
    fp = fp_impossible
    fn = fn_impossible
    denom = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = ((tp * tn - fp * fn) / denom) if denom else 0.0

    def remaining_tokens(row: dict[str, Any]) -> float:
        budget = float(row.get("student_budget_tokens") or 0.0)
        frac = float(row.get("prefix_fraction") or 0.0)
        return max(0.0, budget * (1.0 - frac))

    wrong_total_tokens = sum(float(row.get("student_budget_tokens") or 0.0) for row in valid_rows if row["gold_label"] == "impossible")
    correct_total_tokens = sum(float(row.get("student_budget_tokens") or 0.0) for row in valid_rows if row["gold_label"] == "possible")
    all_total_tokens = sum(float(row.get("student_budget_tokens") or 0.0) for row in valid_rows)
    saved_wrong_tokens = sum(remaining_tokens(row) for row in valid_rows if row["gold_label"] == "impossible" and row.get("prediction", row.get("parsed_prediction")) == "impossible")
    false_abort_correct_tokens = sum(remaining_tokens(row) for row in valid_rows if row["gold_label"] == "possible" and row.get("prediction", row.get("parsed_prediction")) == "impossible")
    saved_all_predicted_impossible_tokens = sum(remaining_tokens(row) for row in valid_rows if row.get("prediction", row.get("parsed_prediction")) == "impossible")

    return {
        "n": n_valid,
        "n_total_rows": len(all_rows),
        "invalid_prediction_count": len(all_rows) - n_valid,
        "n_possible": n_possible,
        "n_impossible": n_impossible,
        "accuracy": accuracy,
        "macro_f1": 0.5 * (possible_f1 + fail_f1),
        "possible_f1": possible_f1,
        "fail_f1": fail_f1,
        "precision_possible": precision_possible,
        "recall_possible": recall_possible,
        "precision_impossible": precision_impossible,
        "recall_impossible": recall_impossible,
        "false_abort_rate": fp_impossible / n_possible if n_possible else 0.0,
        "false_continue_rate": fn_impossible / n_impossible if n_impossible else 0.0,
        "predicted_impossible_rate": (tp_impossible + fp_impossible) / n_valid if n_valid else 0.0,
        "mcc": mcc,
        "wrong_token_saving_rate": saved_wrong_tokens / wrong_total_tokens if wrong_total_tokens else 0.0,
        "correct_token_false_abort_rate": false_abort_correct_tokens / correct_total_tokens if correct_total_tokens else 0.0,
        "overall_token_saving_if_abort_rate": saved_all_predicted_impossible_tokens / all_total_tokens if all_total_tokens else 0.0,
        "confusion": {
            "true_possible_pred_possible": tp_possible,
            "true_possible_pred_impossible": fp_impossible,
            "true_impossible_pred_possible": fn_impossible,
            "true_impossible_pred_impossible": tp_impossible,
        },
    }


def _apply_chat_template(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_tokenizer(path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path)


def _build_records_without_predictions(examples: Sequence[ProbeExample]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for example in examples:
        messages = build_probe_messages(
            problem=example.problem,
            prefix_text=example.prefix_text,
            student_budget_tokens=example.student_budget_tokens,
            prefix_tokens=example.prefix_tokens,
            remaining_budget_tokens=example.remaining_budget_tokens,
            prefix_fraction=example.prefix_fraction,
        )
        record = example.to_record()
        record["messages"] = messages
        record["prediction_text"] = None
        record["prediction"] = None
        records.append(record)
    return records


def run_teacher_probe(
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
    """Run vLLM teacher binary possible/impossible probe."""
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model)
    prompts: list[str] = []
    messages_list: list[list[dict[str, str]]] = []
    for example in examples:
        messages = build_probe_messages(
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
        prediction = parse_probe_answer(text)
        record = example.to_record()
        record.update(
            {
                "messages": messages,
                "prediction_text": text,
                "prediction": prediction,
            }
        )
        records.append(record)
    return records


def _group_metrics_by_prefix(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = f"prefix_{int(record.get('prefix_percent', _format_percent(record.get('prefix_fraction', 0)))):02d}"
        grouped.setdefault(key, []).append(record)
    return {key: compute_bagen_style_metrics(value) for key, value in sorted(grouped.items())}


def _build_summary(records: Sequence[dict[str, Any]], *, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "input_file": str(args.input_file),
        "dataset": str(args.dataset),
        "teacher_model": None if args.dry_run else str(args.teacher_model),
        "tokenizer_path": str(args.tokenizer_path),
        "prefix_fractions": [float(value) for value in args.prefix_fractions],
        "max_rollouts_per_problem": args.max_rollouts_per_problem,
        "max_samples": args.max_samples,
        "dry_run": bool(args.dry_run),
        "overall": compute_bagen_style_metrics(records),
        "by_prefix": _group_metrics_by_prefix(records),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", required=True, help="Student eval output JSONL with responses/response_lengths/acc_list")
    parser.add_argument("--output-jsonl", required=True, help="Path to write probe records JSONL")
    parser.add_argument("--summary-json", required=True, help="Path to write summary metrics JSON")
    parser.add_argument("--dataset", default="aime24")
    parser.add_argument("--tokenizer-path", default="/home/mchen/FiRe-OPD/models/Qwen3-4B")
    parser.add_argument("--teacher-model", default="/home/mchen/FiRe-OPD/models/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--prefix-fractions", nargs="+", type=float, default=[0.10, 0.20, 0.40])
    parser.add_argument("--max-rollouts-per-problem", type=int, default=32, help="Use first K rollouts per problem; <=0 means all")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap after flattening prefix examples")
    parser.add_argument("--dry-run", action="store_true", help="Only build prompt records; do not load/run teacher model")
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    tokenizer = _load_tokenizer(str(args.tokenizer_path))
    max_rollouts = args.max_rollouts_per_problem
    if max_rollouts is not None and int(max_rollouts) <= 0:
        max_rollouts = None

    examples = flatten_eval_outputs(
        input_path=args.input_file,
        tokenizer=tokenizer,
        prefix_fractions=args.prefix_fractions,
        dataset=args.dataset,
        max_rollouts_per_problem=max_rollouts,
        max_samples=args.max_samples,
    )
    print(f"Built {len(examples)} probe examples from {args.input_file}")

    if args.dry_run:
        records = _build_records_without_predictions(examples)
    else:
        records = run_teacher_probe(
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
    summary = _build_summary(records, args=args)
    _write_json(Path(args.summary_json), summary)
    print(f"Wrote records: {args.output_jsonl}")
    print(f"Wrote summary: {args.summary_json}")
    if not args.dry_run:
        print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
