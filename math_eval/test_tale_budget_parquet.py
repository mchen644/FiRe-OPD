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
