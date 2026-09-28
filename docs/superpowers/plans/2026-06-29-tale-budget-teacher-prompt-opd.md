# TALE Budgeted Teacher Prompt OPD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a TALE-style prompt-layer OPD experiment where the current student policy estimates a per-problem token budget online at each PPO step, the trainer builds an in-memory budget-aware `teacher_prompt`, and OPD trains with normal student prompts plus budget-aware teacher/ref prompts.

**Architecture:** Keep the existing offline parquet converter as an ablation/debugging path, but make the main path online/on-policy: before ref log-prob computation, the trainer asks the current actor/rollout worker to estimate budgets for the original prompts, parses and normalizes those budgets, writes budget-aware prompts into `batch.non_tensor_batch["teacher_prompt"]`, and reuses existing veRL `data.ref_raw_prompt_key=teacher_prompt` support. The first experiment is pure OPD: no length-aware penalty and no candidate-selection loss masking.

**Tech Stack:** Python, pandas/pyarrow parquet for ablation data prep, vLLM/actor rollout generation for online student budget estimation, veRL PPO trainer, pytest, bash wrappers.

## Global Constraints

- Do not run full data prep, training, eval, or live vLLM budget estimation while implementing this plan unless explicitly requested.
- Student rollout must continue to use the normal original prompt.
- Online budgeted teacher/ref prompts must be generated in memory before ref log-prob computation.
- Offline `teacher_prompt` parquet generation is ablation-only after the online pivot.
- Pure OPD remains the first experiment: no length-aware penalty and no candidate-selection loss masking.
- Default online budget settings: `enabled=False`, `min_budget=128`, `max_budget=8192`, `round_to=64`, `fallback_budget=2048`, `estimation_max_tokens=64`, `temperature=0.1`, `top_p=0.9`, `teacher_prompt_key=teacher_prompt`.

---

## File Structure

- Create `math_eval/tale_budget_parquet.py`
  - Pure prompt helpers: extract raw problem, build TALE budget-estimation prompt, parse/normalize budgets, build budgeted teacher messages.
  - DataFrame/parquet conversion helpers for offline ablation data prep.
  - CLI for either loading budget estimates from JSONL or generating estimates with vLLM for ablation data prep.
- Create `math_eval/test_tale_budget_parquet.py`
  - CPU-only unit tests for prompt formatting, parsing, normalization, and parquet conversion.
- Create `math_eval/prepare_g_opd_student_raw_teacher_tale_budget_data.sh`
  - Ablation-only shell wrapper that converts train/AIME parquet files into a new data root.
- Create `verl/examples/fire_opd/run_opd_strong_to_weak_student_raw_teacher_tale_budget30b.sh`
  - Pure OPD training script using `+data.ref_raw_prompt_key=teacher_prompt` and `algorithm.tale_budget.enabled=True`.
- Create `run_train_tale_budget_opd.sh`
  - Top-level convenience launcher.
- Create `math_eval/run_eval_math_tale_budget_step50_table2.sh`
  - Table-2 eval wrapper for the step-50 checkpoint.
- Create `verl/verl/trainer/ppo/tale_budget.py`
  - CPU-testable online TALE budget helpers for building estimation prompts, parsing/normalizing budgets, building teacher prompts, and summarizing metrics.
- Modify `verl/verl/trainer/config/algorithm.py`
  - Add typed `TaleBudgetConfig` defaults.
- Modify `verl/verl/trainer/config/ppo_trainer.yaml`
  - Add `algorithm.tale_budget` defaults.
- Create `verl/tests/trainer/ppo/test_tale_budget.py`
  - CPU-only unit tests for online TALE budget config/helper behavior.

---

## Online Pivot Tasks

### Task 8: Update Docs for Online/On-Policy Budget Main Path

**Files:**
- Modify: `docs/superpowers/specs/2026-06-29-tale-budget-teacher-prompt-opd-design.md`
- Modify: `docs/superpowers/plans/2026-06-29-tale-budget-teacher-prompt-opd.md`

- [x] **Step 1: Mark offline data prep as ablation-only**

Update the design and plan docs so the main path is online/on-policy budget estimation during PPO, and the existing parquet conversion scripts are explicitly ablation-only.

- [x] **Step 2: Add online budget config and helper task entries**

Document the online config keys and the first CPU-testable implementation slice:

```text
algorithm.tale_budget.enabled
algorithm.tale_budget.min_budget
algorithm.tale_budget.max_budget
algorithm.tale_budget.round_to
algorithm.tale_budget.fallback_budget
algorithm.tale_budget.estimation_max_tokens
algorithm.tale_budget.temperature
algorithm.tale_budget.top_p
algorithm.tale_budget.teacher_prompt_key
```

- [x] **Step 3: Self-review docs**

Check that the docs no longer describe offline parquet `teacher_prompt` generation as the main experiment path.

### Task 9: Add Online Budget Config and Pure Helpers

**Files:**
- Modify: `verl/verl/trainer/config/algorithm.py`
- Modify: `verl/verl/trainer/config/ppo_trainer.yaml`
- Create: `verl/verl/trainer/ppo/tale_budget.py`
- Create: `verl/tests/trainer/ppo/test_tale_budget.py`

**Interfaces:**
- Produces: `TaleBudgetConfig`
- Produces: `build_tale_budget_estimation_prompt(question: str) -> str`
- Produces: `parse_tale_budget(text: Any) -> int | None`
- Produces: `normalize_tale_budget(raw_budget: int | None, *, min_budget: int, max_budget: int, round_to: int, fallback_budget: int) -> int`
- Produces: `strip_fire_opd_verbose_instruction(question: str) -> str`
- Produces: `build_tale_budget_teacher_messages(question: str, budget: int) -> list[dict[str, str]]`
- Produces: `summarize_tale_budget_metrics(raw_budgets: Sequence[int | None], budgets: Sequence[int]) -> dict[str, float]`

- [x] **Step 1: Write failing CPU tests**

Create `verl/tests/trainer/ppo/test_tale_budget.py` with tests for:

```python
from verl.trainer.config.algorithm import TaleBudgetConfig
from verl.trainer.ppo.tale_budget import (
    build_tale_budget_estimation_prompt,
    build_tale_budget_teacher_messages,
    normalize_tale_budget,
    parse_tale_budget,
    summarize_tale_budget_metrics,
)


def test_tale_budget_config_defaults_match_online_pivot():
    config = TaleBudgetConfig()

    assert config.enabled is False
    assert config.min_budget == 128
    assert config.max_budget == 8192
    assert config.round_to == 64
    assert config.fallback_budget == 2048
    assert config.estimation_max_tokens == 64
    assert config.temperature == 0.1
    assert config.top_p == 0.9
    assert config.teacher_prompt_key == "teacher_prompt"


def test_build_tale_budget_estimation_prompt_matches_tale_format():
    prompt = build_tale_budget_estimation_prompt("What is 2+2?")

    assert "Task: Analyze the given question and estimate the minimum number of tokens" in prompt
    assert "strictly following this format: [[budget]]" in prompt
    assert "Budget: [[12]]" in prompt
    assert 'Below is the question:\n\nQuestion: "What is 2+2?"' in prompt


def test_parse_and_normalize_tale_budget():
    assert parse_tale_budget("Budget: [[256]]") == 256
    assert parse_tale_budget("Budget: 256") is None
    assert normalize_tale_budget(190, min_budget=128, max_budget=8192, round_to=64, fallback_budget=2048) == 192
    assert normalize_tale_budget(None, min_budget=128, max_budget=8192, round_to=64, fallback_budget=2048) == 2048


def test_build_tale_budget_teacher_messages_strips_old_instruction():
    messages = build_tale_budget_teacher_messages(
        "Find x if x+1=3.\nPlease reason step by step, and put your final answer within \\boxed{}.",
        256,
    )

    assert messages == [
        {
            "role": "user",
            "content": (
                "Find x if x+1=3.\n"
                "Let's think step by step and use less than 256 tokens. "
                "Put your final answer within \\boxed{}."
            ),
        }
    ]


def test_summarize_tale_budget_metrics_reports_parse_fail_ratio():
    metrics = summarize_tale_budget_metrics(raw_budgets=[128, None, 256], budgets=[128, 2048, 256])

    assert metrics["tale_budget/mean"] == (128 + 2048 + 256) / 3
    assert metrics["tale_budget/min"] == 128
    assert metrics["tale_budget/max"] == 2048
    assert metrics["tale_budget/parse_fail_ratio"] == 1 / 3
```

- [x] **Step 2: Run tests to verify RED**

Run:

```bash
cd /home/mchen/FiRe-OPD
/home/mchen/miniconda3/envs/verl/bin/python -m pytest verl/tests/trainer/ppo/test_tale_budget.py -q
```

Expected: FAIL because `TaleBudgetConfig` / `verl.trainer.ppo.tale_budget` do not exist yet.

- [x] **Step 3: Implement config and helpers**

Add `TaleBudgetConfig` to `verl/verl/trainer/config/algorithm.py` and wire `algorithm.tale_budget` defaults in `verl/verl/trainer/config/ppo_trainer.yaml`. Add pure helper functions in `verl/verl/trainer/ppo/tale_budget.py`.

- [x] **Step 4: Run tests to verify GREEN**

Run:

```bash
cd /home/mchen/FiRe-OPD
/home/mchen/miniconda3/envs/verl/bin/python -m pytest verl/tests/trainer/ppo/test_tale_budget.py math_eval/test_tale_budget_parquet.py -q
```

Expected: all tests pass.

### Task 10: Wire Online Budget Prompts Before Ref Log-Prob

**Files:**
- Modify: `verl/verl/trainer/ppo/ray_trainer.py`
- Modify: `verl/verl/workers/fsdp_workers.py`
- Modify: `verl/verl/workers/megatron_workers.py`
- Modify: `verl/examples/fire_opd/run_opd_strong_to_weak_student_raw_teacher_tale_budget30b.sh`
- Test: `verl/tests/trainer/ppo/test_tale_budget.py`

**Status:** Implemented.

This task will call the actor/rollout worker to generate budget estimates for TALE budget-estimation prompts, build `batch.non_tensor_batch[teacher_prompt_key]`, set `data.ref_raw_prompt_key=teacher_prompt`, and add metrics `tale_budget/mean`, `tale_budget/min`, `tale_budget/max`, and `tale_budget/parse_fail_ratio`.

- [x] **Step 1: Add failing trainer wiring tests**

Add CPU tests that fail if `ray_trainer.py` does not generate online TALE budgets, write `teacher_prompt`, and call the helper before reference prompt re-tokenization.

- [x] **Step 2: Implement trainer-side online budget prompt generation**

Build TALE budget-estimation prompts from `raw_prompt`, generate estimates with the actor rollout worker, parse/normalize budgets, write budget-aware teacher messages into `batch.non_tensor_batch[teacher_prompt_key]`, and emit TALE budget metrics.

- [x] **Step 3: Forward per-call rollout sampling kwargs**

Allow FSDP/Megatron actor rollout workers to pass `generation_kwargs` to rollout backends that support keyword sampling overrides, so vLLM/SGLang can honor the budget-estimation `max_tokens`, `temperature`, and `top_p`.

- [x] **Step 4: Update online TALE training wrapper**

Point the wrapper at original `data/g-opd`, set `algorithm.tale_budget.enabled=True`, set `algorithm.tale_budget.teacher_prompt_key=teacher_prompt`, and keep `actor_rollout_ref.actor.policy_loss.length_aware_opd=False`.

---

### Task 1: Add CPU Tests for TALE Budget Prompt Utilities

**Files:**
- Create: `math_eval/test_tale_budget_parquet.py`
- Create later in Task 2: `math_eval/tale_budget_parquet.py`

- [ ] **Step 1: Write failing tests**

Create `math_eval/test_tale_budget_parquet.py` with:

```python
from pathlib import Path

import pandas as pd
import pytest

from math_eval.tale_budget_parquet import (
    build_budget_estimation_prompt,
    build_budgeted_teacher_messages,
    convert_dataframe_with_budget_estimates,
    extract_single_user_question,
    normalize_budget,
    parse_budget,
)


def test_build_budget_estimation_prompt_matches_tale_format():
    prompt = build_budget_estimation_prompt("What is 2+2?")

    assert "Task: Analyze the given question and estimate the minimum number of tokens" in prompt
    assert "strictly following this format: [[budget]]" in prompt
    assert "Budget: [[12]]" in prompt
    assert 'Below is the question:\n\nQuestion: "What is 2+2?"' in prompt


def test_parse_budget_uses_double_square_brackets():
    assert parse_budget("Budget: [[256]]") == 256
    assert parse_budget("I estimate [[1024]] tokens.") == 1024
    assert parse_budget("Budget: 256") is None
    assert parse_budget("[[abc]]") is None


def test_normalize_budget_clamps_and_rounds():
    assert normalize_budget(130, min_budget=128, max_budget=8192, round_to=64, fallback_budget=2048) == 128
    assert normalize_budget(190, min_budget=128, max_budget=8192, round_to=64, fallback_budget=2048) == 192
    assert normalize_budget(9000, min_budget=128, max_budget=8192, round_to=64, fallback_budget=2048) == 8192
    assert normalize_budget(None, min_budget=128, max_budget=8192, round_to=64, fallback_budget=2048) == 2048


def test_extract_single_user_question_accepts_numpy_like_values():
    messages = [{"role": "user", "content": "Problem text"}]
    assert extract_single_user_question(messages) == "Problem text"


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "system", "content": "x"}],
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
    ],
)
def test_extract_single_user_question_rejects_invalid_messages(messages):
    with pytest.raises(ValueError):
        extract_single_user_question(messages)


def test_build_budgeted_teacher_messages_strips_old_instruction_and_adds_budget():
    original = (
        "Find x if x+1=3.\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
    )

    messages = build_budgeted_teacher_messages(original, 256)

    assert messages == [
        {
            "role": "user",
            "content": (
                "Find x if x+1=3.\n"
                "Let's think step by step and use less than 256 tokens. "
                "Put your final answer within \\boxed{}."
            ),
        }
    ]


def test_convert_dataframe_with_budget_estimates_preserves_student_prompt_and_adds_teacher_prompt():
    original_prompt = [
        {
            "role": "user",
            "content": (
                "Find x if x+1=3.\n"
                "Please reason step by step, and put your final answer within \\boxed{}."
            ),
        }
    ]
    df = pd.DataFrame(
        {
            "data_source": ["unit"],
            "prompt": [original_prompt],
            "ability": ["math"],
            "reward_model": [{"style": "rule", "ground_truth": "2"}],
            "extra_info": [{"index": 0}],
        }
    )
    budget_records = [{"index": 0, "raw_budget": 250, "estimate_text": "Budget: [[250]]"}]

    converted = convert_dataframe_with_budget_estimates(
        df,
        budget_records=budget_records,
        min_budget=128,
        max_budget=8192,
        round_to=64,
        fallback_budget=2048,
        prompt_key="prompt",
        output_prompt_key="teacher_prompt",
    )

    assert converted.loc[0, "prompt"] == original_prompt
    assert converted.loc[0, "tale_budget_raw"] == 250
    assert converted.loc[0, "tale_budget"] == 256
    assert converted.loc[0, "tale_budget_estimate_text"] == "Budget: [[250]]"
    teacher_prompt = converted.loc[0, "teacher_prompt"]
    assert teacher_prompt[0]["role"] == "user"
    assert "use less than 256 tokens" in teacher_prompt[0]["content"]
    assert "Please reason step by step" not in teacher_prompt[0]["content"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /home/mchen/FiRe-OPD
/home/mchen/miniconda3/envs/verl/bin/python -m pytest math_eval/test_tale_budget_parquet.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'math_eval.tale_budget_parquet'`.

---

### Task 2: Implement TALE Budget Prompt Utility Core

**Files:**
- Create: `math_eval/tale_budget_parquet.py`
- Test: `math_eval/test_tale_budget_parquet.py`

- [ ] **Step 1: Create the utility module with pure helpers**

Create `math_eval/tale_budget_parquet.py` with these definitions:

```python
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
```

- [ ] **Step 2: Run unit tests**

Run:

```bash
cd /home/mchen/FiRe-OPD
/home/mchen/miniconda3/envs/verl/bin/python -m pytest math_eval/test_tale_budget_parquet.py -q
```

Expected: all tests pass.

---

### Task 3: Add CLI for Budget Estimation and Parquet Conversion

**Files:**
- Modify: `math_eval/tale_budget_parquet.py`
- Test: `math_eval/test_tale_budget_parquet.py`

- [ ] **Step 1: Add JSONL budget-cache helpers and CLI argument parsing**

Append to `math_eval/tale_budget_parquet.py`:

```python

def load_budget_records(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_budget_records(records: list[dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


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
    parser = argparse.ArgumentParser(description="Create TALE-style budgeted teacher prompts for FiRe-OPD parquet data.")
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
```

- [ ] **Step 2: Add tests for JSONL helpers using temp files**

Append to `math_eval/test_tale_budget_parquet.py`:

```python
from math_eval.tale_budget_parquet import load_budget_records, save_budget_records


def test_budget_record_jsonl_roundtrip(tmp_path: Path):
    path = tmp_path / "budgets.jsonl"
    records = [
        {"index": 0, "estimate_text": "Budget: [[128]]", "raw_budget": 128},
        {"index": 1, "estimate_text": "Budget: [[256]]", "raw_budget": 256},
    ]

    save_budget_records(records, path)

    assert load_budget_records(path) == records
```

- [ ] **Step 3: Run tests**

Run:

```bash
cd /home/mchen/FiRe-OPD
/home/mchen/miniconda3/envs/verl/bin/python -m pytest math_eval/test_tale_budget_parquet.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Run a no-GPU smoke conversion using handcrafted budget records**

Run:

```bash
cd /home/mchen/FiRe-OPD
mkdir -p /tmp/tale_budget_smoke
printf '%s\n' \
  '{"index": 0, "estimate_text": "Budget: [[256]]", "raw_budget": 256}' \
  '{"index": 1, "estimate_text": "Budget: [[512]]", "raw_budget": 512}' \
  > /tmp/tale_budget_smoke/budgets.jsonl
/home/mchen/miniconda3/envs/verl/bin/python math_eval/tale_budget_parquet.py \
  --input_file data/g-opd/DeepMath-103K/train_filtered_level6.parquet \
  --output_file /tmp/tale_budget_smoke/train2.parquet \
  --budget_records /tmp/tale_budget_smoke/budgets.jsonl \
  --limit_rows 2
```

Expected: prints `wrote 2 rows: /tmp/tale_budget_smoke/train2.parquet`.

---

### Task 4: Add Dataset Preparation Shell Wrapper

**Files:**
- Create: `math_eval/prepare_g_opd_student_raw_teacher_tale_budget_data.sh`

- [ ] **Step 1: Create shell wrapper**

Create `math_eval/prepare_g_opd_student_raw_teacher_tale_budget_data.sh`:

```bash
#!/bin/bash
set -euo pipefail

# Create FiRe-OPD parquet files where student rollout keeps the raw prompt and
# teacher/ref log-prob uses a TALE-style budget-aware prompt stored in teacher_prompt.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/mchen/miniconda3/envs/verl/bin/python}"
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/verl:${PYTHONPATH:-}"

STUDENT_MODEL="${STUDENT_MODEL:-${REPO_DIR}/models/Qwen3-4B}"
INPUT_ROOT="${INPUT_ROOT:-${REPO_DIR}/data/g-opd}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/data/g-opd-student-raw-teacher-tale-budget}"
BUDGET_CACHE_ROOT="${BUDGET_CACHE_ROOT:-${OUTPUT_ROOT}/budget_records}"
TEACHER_PROMPT_KEY="${TEACHER_PROMPT_KEY:-teacher_prompt}"
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
MIN_BUDGET="${MIN_BUDGET:-128}"
MAX_BUDGET="${MAX_BUDGET:-8192}"
ROUND_TO="${ROUND_TO:-64}"
FALLBACK_BUDGET="${FALLBACK_BUDGET:-2048}"
LIMIT_ROWS="${LIMIT_ROWS:-}"

convert_one() {
  local rel_path="$1"
  local input_file="${INPUT_ROOT}/${rel_path}"
  local output_file="${OUTPUT_ROOT}/${rel_path}"
  local cache_file="${BUDGET_CACHE_ROOT}/${rel_path%.parquet}.jsonl"

  if [ ! -f "${input_file}" ]; then
    echo "ERROR: missing input file: ${input_file}" >&2
    exit 1
  fi

  mkdir -p "$(dirname "${output_file}")" "$(dirname "${cache_file}")"

  local args=(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/tale_budget_parquet.py"
    --input_file "${input_file}"
    --output_file "${output_file}"
    --output_prompt_key "${TEACHER_PROMPT_KEY}"
    --min_budget "${MIN_BUDGET}"
    --max_budget "${MAX_BUDGET}"
    --round_to "${ROUND_TO}"
    --fallback_budget "${FALLBACK_BUDGET}"
  )

  if [ -f "${cache_file}" ]; then
    args+=(--budget_records "${cache_file}")
  else
    args+=(
      --student_model "${STUDENT_MODEL}"
      --save_budget_records "${cache_file}"
      --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}"
    )
  fi

  if [ -n "${LIMIT_ROWS}" ]; then
    args+=(--limit_rows "${LIMIT_ROWS}")
  fi

  echo "Converting ${rel_path}"
  echo "  input : ${input_file}"
  echo "  output: ${output_file}"
  echo "  cache : ${cache_file}"
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${args[@]}"
}

convert_one "DeepMath-103K/train_filtered_level6.parquet"
convert_one "AIME2024/test.parquet"
convert_one "AIME2025/test.parquet"

echo "Done. Output root: ${OUTPUT_ROOT}"
```

- [ ] **Step 2: Make wrapper executable and run shell syntax check**

Run:

```bash
cd /home/mchen/FiRe-OPD
chmod +x math_eval/prepare_g_opd_student_raw_teacher_tale_budget_data.sh
bash -n math_eval/prepare_g_opd_student_raw_teacher_tale_budget_data.sh
```

Expected: no output from `bash -n`.

- [ ] **Step 3: Run a smoke conversion with two rows**

Run on an available GPU if using live estimation:

```bash
cd /home/mchen/FiRe-OPD
LIMIT_ROWS=2 GPU_IDS=0,1,2,3 TENSOR_PARALLEL_SIZE=4 \
bash math_eval/prepare_g_opd_student_raw_teacher_tale_budget_data.sh
```

Expected:

```text
wrote 2 rows: .../data/g-opd-student-raw-teacher-tale-budget/DeepMath-103K/train_filtered_level6.parquet
wrote 2 rows: .../data/g-opd-student-raw-teacher-tale-budget/AIME2024/test.parquet
wrote 2 rows: .../data/g-opd-student-raw-teacher-tale-budget/AIME2025/test.parquet
```

---

### Task 5: Add Pure OPD Training Wrapper for Budgeted Teacher Prompts

**Files:**
- Create: `verl/examples/fire_opd/run_opd_strong_to_weak_student_raw_teacher_tale_budget30b.sh`
- Create: `run_train_tale_budget_opd.sh`

- [ ] **Step 1: Create inner veRL training script**

Create `verl/examples/fire_opd/run_opd_strong_to_weak_student_raw_teacher_tale_budget30b.sh`:

```bash
#!/bin/bash
# OPD strong-to-weak with normal student prompts and TALE-style budgeted teacher prompts.
set -x
set -euo pipefail

export PYTHONUNBUFFERED=1
REPO_DIR="${REPO_DIR:-/home/mchen/FiRe-OPD}"
export PYTHONPATH="${REPO_DIR}/verl:${REPO_DIR}:${PYTHONPATH:-}"

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

if [ -n "${SLURM_JOB_ID:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
else
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
fi

RAY_NUM_CPUS="${RAY_NUM_CPUS:-${SLURM_CPUS_PER_TASK:-16}}"
TEACHER_PROMPT_KEY="${TEACHER_PROMPT_KEY:-teacher_prompt}"
DATA_ROOT="${DATA_ROOT:-${REPO_DIR}/data/g-opd-student-raw-teacher-tale-budget}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-4}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
ROLLOUT_N="${ROLLOUT_N:-1}"

TRAIN_DATA="${TRAIN_DATA:-${DATA_ROOT}/DeepMath-103K/train_filtered_level6.parquet}"
VAL_DATA="${VAL_DATA:-['${DATA_ROOT}/AIME2024/test.parquet', '${DATA_ROOT}/AIME2025/test.parquet']}"
STUDENT_MODEL="${STUDENT_MODEL:-${REPO_DIR}/models/Qwen3-4B}"
TEACHER_MODEL="${TEACHER_MODEL:-${REPO_DIR}/models/Qwen3-30B-A3B-Instruct-2507}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-opd-strong-to-weak-studentraw-teachertale-budget-selectn${ROLLOUT_N}-${N_GPUS_PER_NODE}gpu-tp${ROLLOUT_TP_SIZE}-refmb4-rollmb4}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_DIR}/checkpoints/${EXPERIMENT_NAME}}"

if [ ! -f "${TRAIN_DATA}" ]; then
  echo "ERROR: TRAIN_DATA not found: ${TRAIN_DATA}" >&2
  echo "Run: bash ${REPO_DIR}/math_eval/prepare_g_opd_student_raw_teacher_tale_budget_data.sh" >&2
  exit 1
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=5.0 \
    algorithm.rollout_correction.rollout_rs=null \
    algorithm.rollout_correction.bypass_mode=false \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    data.train_files=${TRAIN_DATA} \
    data.val_files="${VAL_DATA}" \
    data.train_batch_size=1024 \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    data.seed=42 \
    data.return_raw_chat=True \
    +data.ref_raw_prompt_key=${TEACHER_PROMPT_KEY} \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=${STUDENT_MODEL} \
    +actor_rollout_ref.ref.model.path=${TEACHER_MODEL} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=True \
    actor_rollout_ref.actor.policy_loss.length_aware_opd=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=1024 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP_SIZE} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=naive \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.logger='["console","wandb"]' \
    trainer.log_val_generations=10 \
    trainer.project_name='fire-opd' \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.default_local_dir=${CHECKPOINT_DIR} \
    trainer.test_freq=10 \
    trainer.total_epochs=3 \
    trainer.resume_mode=auto \
    ray_kwargs.ray_init.num_cpus=${RAY_NUM_CPUS}
```

- [ ] **Step 2: Create top-level launcher**

Create `run_train_tale_budget_opd.sh`:

```bash
#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/mchen/FiRe-OPD}"
cd "${REPO_DIR}"

ROLLOUT_N="${ROLLOUT_N:-1}" \
bash verl/examples/fire_opd/run_opd_strong_to_weak_student_raw_teacher_tale_budget30b.sh
```

- [ ] **Step 3: Make scripts executable and syntax-check them**

Run:

```bash
cd /home/mchen/FiRe-OPD
chmod +x run_train_tale_budget_opd.sh \
  verl/examples/fire_opd/run_opd_strong_to_weak_student_raw_teacher_tale_budget30b.sh
bash -n run_train_tale_budget_opd.sh
bash -n verl/examples/fire_opd/run_opd_strong_to_weak_student_raw_teacher_tale_budget30b.sh
```

Expected: no syntax errors.

---

### Task 6: Add Step-50 Table-2 Eval Wrapper

**Files:**
- Create: `math_eval/run_eval_math_tale_budget_step50_table2.sh`

- [ ] **Step 1: Copy and adapt the OPD raw eval wrapper**

Create `math_eval/run_eval_math_tale_budget_step50_table2.sh` based on `math_eval/run_eval_math_opd_rawprompt_step50.sh`, with these default changes:

```bash
EXPERIMENT_NAME="${EXPERIMENT_NAME:-opd-strong-to-weak-studentraw-teachertale-budget-selectn1-4gpu-tp4-refmb4-rollmb4}"
MODEL_KEY="${MODEL_KEY:-tale_budget_opd_step${STEP}}"
MODEL_NAME="${MODEL_NAME:-${EXPERIMENT_NAME}-step${STEP}-baseline}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/tale_budget_table2_eval_outputs}"
MISTAKES_ROOT="${MISTAKES_ROOT:-${SCRIPT_DIR}/tale_budget_table2_eval_mistakes}"
LOG_ROOT="${LOG_ROOT:-${SCRIPT_DIR}/tale_budget_table2_eval_logs}"
```

Keep these settings unchanged:

```bash
PROMPT_STYLE=baseline
NO_EXTRA_PROMPT=1
N_SAMPLES=32
MAX_TOKENS=16384
TEMPERATURE=1.0
TOP_P=1.0
SEED=42
DATASETS="aime24 aime25 hmmt25_feb hmmt25_nov math500 minervamath olympiadbench amc2023"
```

- [ ] **Step 2: Make executable and syntax-check**

Run:

```bash
cd /home/mchen/FiRe-OPD
chmod +x math_eval/run_eval_math_tale_budget_step50_table2.sh
bash -n math_eval/run_eval_math_tale_budget_step50_table2.sh
```

Expected: no syntax errors.

---

### Task 7: Full Verification and Runbook

**Files:**
- No new files.

- [ ] **Step 1: Run unit tests**

Run:

```bash
cd /home/mchen/FiRe-OPD
/home/mchen/miniconda3/envs/verl/bin/python -m pytest math_eval/test_tale_budget_parquet.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run smoke data prep**

Run on available GPUs:

```bash
cd /home/mchen/FiRe-OPD
LIMIT_ROWS=2 GPU_IDS=0,1,2,3 TENSOR_PARALLEL_SIZE=4 \
bash math_eval/prepare_g_opd_student_raw_teacher_tale_budget_data.sh
```

Expected: three 2-row parquet files under:

```text
data/g-opd-student-raw-teacher-tale-budget/
```

- [ ] **Step 3: Inspect smoke parquet**

Run:

```bash
cd /home/mchen/FiRe-OPD
/home/mchen/miniconda3/envs/verl/bin/python - <<'PY'
import pandas as pd
p='data/g-opd-student-raw-teacher-tale-budget/DeepMath-103K/train_filtered_level6.parquet'
df=pd.read_parquet(p)
print(df.columns.tolist())
print(df.loc[0,'prompt'][0]['content'])
print('--- teacher prompt ---')
print(df.loc[0,'teacher_prompt'][0]['content'])
print('budget', df.loc[0,'tale_budget'], 'raw', df.loc[0,'tale_budget_raw'])
PY
```

Expected:

- `prompt` still contains the original normal FiRe-OPD instruction.
- `teacher_prompt` contains `Let's think step by step and use less than ... tokens.`
- `teacher_prompt` contains `\boxed{}`.

- [ ] **Step 4: Run full data prep when ready**

Run on available GPUs after smoke succeeds:

```bash
cd /home/mchen/FiRe-OPD
GPU_IDS=0,1,2,3 TENSOR_PARALLEL_SIZE=4 \
bash math_eval/prepare_g_opd_student_raw_teacher_tale_budget_data.sh
```

Expected: full train/AIME parquet files written under:

```text
data/g-opd-student-raw-teacher-tale-budget/
```

- [ ] **Step 5: Launch training**

Run in tmux on intended GPUs:

```bash
cd /home/mchen/FiRe-OPD
ray stop --force || true
HYDRA_FULL_ERROR=1 bash run_train_tale_budget_opd.sh
```

Expected early config evidence:

```text
trainer.experiment_name=opd-strong-to-weak-studentraw-teachertale-budget-selectn1-4gpu-tp4-refmb4-rollmb4
+data.ref_raw_prompt_key=teacher_prompt
actor_rollout_ref.actor.policy_loss.length_aware_opd=False
actor_rollout_ref.rollout.n=1
```

- [ ] **Step 6: After step 50, run eval**

Run on available GPUs:

```bash
cd /home/mchen/FiRe-OPD
GPU_IDS=0,1,2,3 bash math_eval/run_eval_math_tale_budget_step50_table2.sh
```

Expected: summary logs under:

```text
math_eval/tale_budget_table2_eval_logs/
```

Compare against:

```text
math_eval/opd_rawprompt_step_eval_logs/
math_eval/lengthaware_table2_eval_logs/
math_eval/correctcompress_table2_eval_logs/
```

---

## Self-Review

- Spec coverage: The plan implements TALE budget estimation, budgeted teacher prompts, `teacher_prompt` parquet data, pure OPD training, and Table-2 eval.
- Scope check: This is one coherent prompt-layer experiment. It intentionally excludes online budget estimation, length-aware penalties, and candidate-selection loss masking.
- Placeholder scan: No task contains unresolved placeholders. Commands include exact paths and expected outcomes.
- Type consistency: The planned utility uses `teacher_prompt`, `tale_budget`, `tale_budget_raw`, and `tale_budget_estimate_text` consistently across tests, conversion, and scripts.
