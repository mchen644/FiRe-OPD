"""Prepare TALE-style budget-aware teacher prompts for FiRe-OPD parquet data."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:
    from math_eval.cod_prompt import strip_fire_opd_verbose_instruction
except ImportError:
    from cod_prompt import strip_fire_opd_verbose_instruction

TALE_BUDGET_CONTEXT = (
    "Task: Analyze the given question and estimate the minimum number of tokens required "
    "to generate a complete and accurate response. Please Give the response by strictly "
    "following this format: [[budget]],for example: Budget: [[12]]."
)

_BUDGET_RE = re.compile(r"\[\[(\d+)\]\]")


def _messages_to_list(messages: Any) -> list[dict[str, Any]]:
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    return list(messages)


def extract_single_user_question(messages: Any) -> str:
    message_list = _messages_to_list(messages)
    if len(message_list) != 1:
        raise ValueError(f"Expected exactly one prompt message, got {len(message_list)}")
    message = dict(message_list[0])
    if message.get("role") != "user":
        raise ValueError(f"Expected prompt role 'user', got {message.get('role')!r}")
    return str(message.get("content", ""))


def build_budget_estimation_prompt(question: str) -> str:
    return f'{TALE_BUDGET_CONTEXT}\n\nBelow is the question:\n\nQuestion: "{question}"\n'


def parse_budget(text: str) -> int | None:
    match = _BUDGET_RE.search(str(text))
    if match is None:
        return None
    return int(match.group(1))


def normalize_budget(
    raw_budget: int | None,
    *,
    min_budget: int,
    max_budget: int,
    round_to: int,
    fallback_budget: int,
) -> int:
    if min_budget <= 0:
        raise ValueError("min_budget must be positive")
    if max_budget < min_budget:
        raise ValueError("max_budget must be >= min_budget")
    if round_to <= 0:
        raise ValueError("round_to must be positive")
    value = fallback_budget if raw_budget is None else int(raw_budget)
    value = max(min_budget, min(max_budget, value))
    value = int(round(value / round_to) * round_to)
    value = max(min_budget, min(max_budget, value))
    return value


def build_budgeted_teacher_messages(question: str, budget: int) -> list[dict[str, str]]:
    stripped_question = strip_fire_opd_verbose_instruction(question)
    content = (
        f"{stripped_question}\n"
        f"Let's think step by step and use less than {int(budget)} tokens. "
        r"Put your final answer within \boxed{}."
    )
    return [{"role": "user", "content": content}]


def _records_by_index(budget_records: Iterable[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for record in budget_records:
        index = int(record["index"])
        if index in records:
            raise ValueError(f"Duplicate budget record index: {index}")
        records[index] = dict(record)
    return records


def convert_dataframe_with_budget_estimates(
    dataframe: pd.DataFrame,
    *,
    budget_records: Iterable[dict[str, Any]],
    min_budget: int,
    max_budget: int,
    round_to: int,
    fallback_budget: int,
    prompt_key: str = "prompt",
    output_prompt_key: str = "teacher_prompt",
) -> pd.DataFrame:
    if prompt_key not in dataframe.columns:
        raise ValueError(f"Input dataframe must contain a {prompt_key!r} column")
    records = _records_by_index(budget_records)
    converted = dataframe.copy(deep=True)

    teacher_prompts = []
    normalized_budgets = []
    raw_budgets = []
    estimate_texts = []

    for row_idx, messages in enumerate(converted[prompt_key]):
        record = records.get(row_idx, {})
        raw_budget = record.get("raw_budget")
        estimate_text = str(record.get("estimate_text", ""))
        if raw_budget is None and estimate_text:
            raw_budget = parse_budget(estimate_text)
        budget = normalize_budget(
            raw_budget,
            min_budget=min_budget,
            max_budget=max_budget,
            round_to=round_to,
            fallback_budget=fallback_budget,
        )
        question = extract_single_user_question(messages)
        teacher_prompts.append(build_budgeted_teacher_messages(question, budget))
        normalized_budgets.append(budget)
        raw_budgets.append(raw_budget)
        estimate_texts.append(estimate_text)

    converted[output_prompt_key] = teacher_prompts
    converted["tale_budget"] = normalized_budgets
    converted["tale_budget_raw"] = raw_budgets
    converted["tale_budget_estimate_text"] = estimate_texts
    return converted


def load_budget_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_budget_records(records: list[dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_budget_records_cover_rows(budget_records: Iterable[dict[str, Any]], row_count: int) -> None:
    records = _records_by_index(budget_records)
    missing = [idx for idx in range(row_count) if idx not in records]
    if missing:
        preview = ", ".join(str(idx) for idx in missing[:10])
        raise ValueError(
            f"Budget records missing {len(missing)} row indices for {row_count} rows; "
            f"first missing: {preview}"
        )


def estimate_budgets_with_vllm(
    questions: list[str],
    *,
    model_path: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_num_seqs: int,
    tensor_parallel_size: int | None,
    max_model_len: int | None,
) -> list[dict[str, Any]]:
    import torch
    from vllm import LLM, SamplingParams

    tp_size = tensor_parallel_size or torch.cuda.device_count()
    llm_kwargs = dict(
        model=model_path,
        tokenizer=model_path,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=0.9,
        max_num_seqs=max_num_seqs,
        enforce_eager=True,
    )
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len

    llm = LLM(**llm_kwargs)
    prompts = [build_budget_estimation_prompt(question) for question in questions]
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=1,
    )
    generations = llm.generate(prompts, sampling_params=sampling_params)
    records = []
    for index, generation in enumerate(generations):
        estimate_text = generation.outputs[0].text.strip() if generation.outputs else ""
        records.append(
            {
                "index": index,
                "estimate_text": estimate_text,
                "raw_budget": parse_budget(estimate_text),
            }
        )
    return records


def convert_parquet_file(
    input_file: str | Path,
    output_file: str | Path,
    *,
    budget_records: list[dict[str, Any]],
    min_budget: int,
    max_budget: int,
    round_to: int,
    fallback_budget: int,
    prompt_key: str,
    output_prompt_key: str,
) -> None:
    dataframe = pd.read_parquet(input_file)
    converted = convert_dataframe_with_budget_estimates(
        dataframe,
        budget_records=budget_records,
        min_budget=min_budget,
        max_budget=max_budget,
        round_to=round_to,
        fallback_budget=fallback_budget,
        prompt_key=prompt_key,
        output_prompt_key=output_prompt_key,
    )
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    converted.to_parquet(output_path, index=False)
    print(f"wrote {len(converted)} rows: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create TALE-style budgeted teacher prompts for FiRe-OPD parquet data."
    )
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--student_model", default=None, help="Student model path for vLLM budget estimation")
    parser.add_argument("--budget_records", default=None, help="Existing JSONL budget records to reuse")
    parser.add_argument("--save_budget_records", default=None, help="Where to save generated JSONL budget records")
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--output_prompt_key", default="teacher_prompt")
    parser.add_argument("--min_budget", type=int, default=128)
    parser.add_argument("--max_budget", type=int, default=8192)
    parser.add_argument("--round_to", type=int, default=64)
    parser.add_argument("--fallback_budget", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--estimation_max_tokens", type=int, default=64)
    parser.add_argument("--max_num_seqs", type=int, default=256)
    parser.add_argument("--tensor_parallel_size", type=int, default=None)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--limit_rows", type=int, default=None, help="Debug-only limit for smoke conversion")
    args = parser.parse_args()

    dataframe = pd.read_parquet(args.input_file)
    if args.limit_rows is not None:
        dataframe = dataframe.head(args.limit_rows).copy()

    if args.budget_records is not None:
        budget_records = load_budget_records(args.budget_records)
        validate_budget_records_cover_rows(budget_records, len(dataframe))
    else:
        if args.student_model is None:
            raise ValueError("Set --student_model when --budget_records is not provided")
        questions = [extract_single_user_question(messages) for messages in dataframe[args.prompt_key]]
        budget_records = estimate_budgets_with_vllm(
            questions,
            model_path=args.student_model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.estimation_max_tokens,
            max_num_seqs=args.max_num_seqs,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
        )
        if args.save_budget_records is not None:
            save_budget_records(budget_records, args.save_budget_records)

    converted = convert_dataframe_with_budget_estimates(
        dataframe,
        budget_records=budget_records,
        min_budget=args.min_budget,
        max_budget=args.max_budget,
        round_to=args.round_to,
        fallback_budget=args.fallback_budget,
        prompt_key=args.prompt_key,
        output_prompt_key=args.output_prompt_key,
    )
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    converted.to_parquet(output_path, index=False)
    print(f"wrote {len(converted)} rows: {output_path}")


if __name__ == "__main__":
    main()
