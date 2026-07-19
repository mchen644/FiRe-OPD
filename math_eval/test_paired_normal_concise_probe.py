from __future__ import annotations

import copy

import pytest

from math_eval.paired_normal_concise_probe import (
    classify_compression_sensitive,
    concise_cap,
    concise_messages,
    normal_messages,
    quadrant,
    select_probe_rows,
    validate_relaxed_prefix,
)


def _rows(count: int = 8) -> list[dict]:
    return [
        {
            "data_source": "DeepMath-103K",
            "prompt": [
                {
                    "role": "user",
                    "content": (
                        f"Compute {index}+1.\n"
                        "Please reason step by step, and put your final answer within \\boxed{}."
                    ),
                }
            ],
            "ability": "math",
            "reward_model": {"ground_truth": str(index + 1), "style": "rule"},
            "extra_info": {"index": 100 + index, "split": "train"},
        }
        for index in range(count)
    ]


def test_select_probe_rows_is_exact_distinct_and_repeatable() -> None:
    rows = _rows()

    first = select_probe_rows(rows, sample_size=4, seed=42)
    repeated = select_probe_rows(rows, sample_size=4, seed=42)
    changed = select_probe_rows(rows, sample_size=4, seed=43)

    assert first == repeated
    assert len(first) == 4
    assert len({row["source_row_index"] for row in first}) == 4
    assert len({row["question_id"] for row in first}) == 4
    assert [row["source_row_index"] for row in first] != [
        row["source_row_index"] for row in changed
    ]
    assert all(row["request_seed"] == 42 + row["source_row_index"] for row in first)
    assert [row["sample_ordinal"] for row in first] == list(range(4))


def test_select_probe_rows_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="sample_size"):
        select_probe_rows(_rows(), sample_size=9, seed=42)

    duplicate = _rows()
    duplicate[1]["extra_info"]["index"] = duplicate[0]["extra_info"]["index"]
    with pytest.raises(ValueError, match="distinct question"):
        select_probe_rows(duplicate, sample_size=4, seed=42)

    malformed_prompt = _rows()
    malformed_prompt[0]["prompt"] = [{"role": "user", "content": ""}]
    with pytest.raises(ValueError, match="prompt"):
        select_probe_rows(malformed_prompt, sample_size=4, seed=42)

    missing_truth = _rows()
    missing_truth[0]["reward_model"]["ground_truth"] = ""
    with pytest.raises(ValueError, match="ground truth"):
        select_probe_rows(missing_truth, sample_size=4, seed=42)


def test_normal_and_concise_messages_preserve_question_but_change_instruction() -> None:
    row = select_probe_rows(_rows(), sample_size=1, seed=42)[0]
    original_prompt = copy.deepcopy(row["prompt"])

    assert normal_messages(row) == original_prompt
    concise = concise_messages(row)

    assert row["prompt"] == original_prompt
    assert len(concise) == 1
    assert concise[0]["role"] == "user"
    assert "Compute" in concise[0]["content"]
    assert "Solve concisely. Avoid unnecessary explanation." in concise[0]["content"]
    assert "Please reason step by step" not in concise[0]["content"]


def test_concise_cap_and_quadrants_are_exact() -> None:
    assert concise_cap(1) == 1
    assert concise_cap(2) == 1
    assert concise_cap(9) == 4
    with pytest.raises(ValueError, match="positive integer"):
        concise_cap(0)
    with pytest.raises(ValueError, match="positive integer"):
        concise_cap(True)

    assert quadrant(True, True) == "compression_safe"
    assert quadrant(True, False) == "compression_sensitive"
    assert quadrant(False, True) == "concise_rescued"
    assert quadrant(False, False) == "both_wrong"
    with pytest.raises(TypeError, match="boolean"):
        quadrant(1, False)


def test_validate_relaxed_prefix_accepts_only_exact_token_prefix() -> None:
    validate_relaxed_prefix([1, 2], [1, 2, 3])
    validate_relaxed_prefix([1, 2], [1, 2])

    with pytest.raises(ValueError, match="prefix"):
        validate_relaxed_prefix([1, 9], [1, 2, 3])
    with pytest.raises(ValueError, match="nonempty"):
        validate_relaxed_prefix([], [1])


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        (
            {
                "quadrant": "compression_sensitive",
                "concise": {"cap_hit": True, "correct": False, "token_ids": [1, 2]},
                "relaxed_concise": {"correct": True, "token_ids": [1, 2, 3]},
            },
            "budget_limited_recovered",
        ),
        (
            {
                "quadrant": "compression_sensitive",
                "concise": {"cap_hit": True, "correct": False, "token_ids": [1, 2]},
                "relaxed_concise": {"correct": False, "token_ids": [1, 2, 4]},
            },
            "budget_limited_unrecovered",
        ),
        (
            {
                "quadrant": "compression_sensitive",
                "concise": {"cap_hit": False, "correct": False, "token_ids": [1]},
                "relaxed_concise": None,
            },
            "prompt_or_sampling_failure",
        ),
        (
            {
                "quadrant": "compression_safe",
                "concise": {"cap_hit": False, "correct": True, "token_ids": [1]},
                "relaxed_concise": None,
            },
            None,
        ),
    ],
)
def test_classify_compression_sensitive(record: dict, expected: str | None) -> None:
    assert classify_compression_sensitive(record) == expected


def test_classify_compression_sensitive_requires_relaxed_cap_hit_result() -> None:
    record = {
        "quadrant": "compression_sensitive",
        "concise": {"cap_hit": True, "correct": False, "token_ids": [1]},
        "relaxed_concise": None,
    }
    with pytest.raises(ValueError, match="relaxed"):
        classify_compression_sensitive(record)
