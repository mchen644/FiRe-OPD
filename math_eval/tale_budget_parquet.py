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
