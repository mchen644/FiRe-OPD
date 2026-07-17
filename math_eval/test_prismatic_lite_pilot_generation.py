from __future__ import annotations

from collections import Counter

import pytest

from math_eval.prismatic_lite_pilot_generation import (
    NearDuplicateIndex,
    candidate_id,
    difficulty_weighted_fewshots,
    majority_group,
    normalized_tokens,
    parse_generated_problems,
    parse_semantic_judgment,
    problem_prompt,
    question_level_completion_rows,
    solution_messages,
    stratified_calibration_sample,
    ten_grams,
)


def _rows() -> list[dict]:
    rows = []
    for index, (topic, difficulty) in enumerate(
        [("A", 6.0)] * 6
        + [("A", 7.0)] * 4
        + [("B", 6.0)] * 3
        + [("C", 8.0)] * 3
    ):
        rows.append(
            {
                "id": f"row-{index:03d}",
                "prompt": f"Problem {index}",
                "completion": f"Solution {index}",
                "topic": topic,
                "difficulty": difficulty,
                "source_row_index": index,
                "original_dataset_index": index + 100,
            }
        )
    return rows


def test_stratified_calibration_sample_is_exact_unique_and_repeatable() -> None:
    rows = _rows()

    first = stratified_calibration_sample(rows, size=8, seed=42)
    repeated = stratified_calibration_sample(rows, size=8, seed=42)
    changed = stratified_calibration_sample(rows, size=8, seed=43)

    assert first == repeated
    assert len(first) == len({row["id"] for row in first}) == 8
    assert [row["id"] for row in first] != [row["id"] for row in changed]
    counts = Counter((row["topic"], row["difficulty"]) for row in first)
    assert counts == {("A", 6.0): 3, ("A", 7.0): 2, ("B", 6.0): 2, ("C", 8.0): 1}


def test_stratified_calibration_sample_rejects_invalid_rows_and_size() -> None:
    with pytest.raises(ValueError, match="size"):
        stratified_calibration_sample(_rows(), size=17, seed=42)
    bad = _rows()
    bad[0]["topic"] = ""
    with pytest.raises(ValueError, match="topic"):
        stratified_calibration_sample(bad, size=8, seed=42)


def test_difficulty_weighted_fewshots_are_distinct_and_deterministic() -> None:
    rows = _rows()

    first = difficulty_weighted_fewshots(rows, requests=6, width=5, seed=42)
    repeated = difficulty_weighted_fewshots(rows, requests=6, width=5, seed=42)
    changed = difficulty_weighted_fewshots(rows, requests=6, width=5, seed=43)

    assert first == repeated
    assert first != changed
    assert len(first) == 6
    assert all(len(group) == len(set(group)) == 5 for group in first)
    assert all(item.startswith("row-") for group in first for item in group)


def test_problem_prompt_and_parser_round_trip_one_or_more_blocks() -> None:
    examples = [
        {"id": f"row-{index}", "prompt": f"Example problem {index}"}
        for index in range(5)
    ]
    prompt = problem_prompt(examples)

    assert "similar or harder" in prompt
    assert all(example["prompt"] in prompt for example in examples)
    assert "non-multiple-choice" in prompt

    generation = """---
[[Problem]]
Find the integer n such that n+1=3.
---
noise
---
[[Problem]]
Prove that 1+1=2.
---"""
    assert parse_generated_problems(generation) == [
        "Find the integer n such that n+1=3.",
        "Prove that 1+1=2.",
    ]
    assert parse_generated_problems("unstructured answer") == []


def test_solution_messages_disable_implicit_answer_context() -> None:
    messages = solution_messages("Compute 2+2.")

    assert messages == [
        {
            "role": "system",
            "content": "Solve the problem rigorously and put the final answer in \\boxed{}.",
        },
        {"role": "user", "content": "Compute 2+2."},
    ]


def test_candidate_id_binds_request_ordinal_and_problem_text() -> None:
    value = candidate_id(12, 0, "  Compute 2+2.  ")

    assert value.startswith("prismatic-qwen3-2k-r000012-p00-")
    assert value == candidate_id(12, 0, "Compute 2+2.")
    assert value != candidate_id(12, 1, "Compute 2+2.")
    assert value != candidate_id(12, 0, "Compute 3+3.")


def test_majority_group_requires_two_mathematically_equivalent_boxed_answers() -> None:
    responses = [
        r"Work. Final answer: \\boxed{\frac{1}{2}}",
        r"Different derivation. Thus \\boxed{0.5}",
        r"Mistake. \\boxed{2}",
    ]

    assert majority_group(responses) == (0, 1)
    assert majority_group(
        [r"\\boxed{1}", r"\\boxed{2}", r"\\boxed{3}"]
    ) == ()
    assert majority_group([r"answer is 1", r"answer is 1", r"\\boxed{2}"]) == ()
    with pytest.raises(ValueError, match="exactly three"):
        majority_group(responses[:2])


def test_question_level_completion_rows_preserve_all_majority_solutions() -> None:
    record = {
        "id": "question-1",
        "prompt": "Compute one half.",
        "responses": [r"\\boxed{1/2}", r"\\boxed{0.5}", r"\\boxed{2}"],
        "majority_indices": [0, 1],
    }

    assert question_level_completion_rows(record) == [
        {
            "id": "question-1.solution-0",
            "question_id": "question-1",
            "solution_index": 0,
            "prompt": "Compute one half.",
            "completion": r"\\boxed{1/2}",
        },
        {
            "id": "question-1.solution-1",
            "question_id": "question-1",
            "solution_index": 1,
            "prompt": "Compute one half.",
            "completion": r"\\boxed{0.5}",
        },
    ]


def test_normalized_tokens_and_ten_grams_are_unicode_and_case_stable() -> None:
    a = "Ａ Quick, BROWN fox jumps over the lazy dog and runs away!"
    b = "a quick brown FOX jumps over the lazy dog and runs away"

    assert normalized_tokens(a) == normalized_tokens(b)
    grams = ten_grams(a)
    assert len(grams) == 3
    assert all(len(gram) == 10 for gram in grams)


def test_near_duplicate_index_finds_exact_and_highest_jaccard_neighbor() -> None:
    index = NearDuplicateIndex(
        [
            ("seed-a", "one two three four five six seven eight nine ten eleven"),
            ("seed-b", "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"),
        ]
    )

    exact = index.query("ONE two three four five six seven eight nine ten eleven")
    assert exact == ("seed-a", 1.0)

    near = index.query("one two three four five six seven eight nine ten twelve")
    assert near[0] == "seed-a"
    assert near[1] == pytest.approx(1 / 3)

    assert index.query("completely unrelated short text") == (None, 0.0)
    index.add("new", "red blue green yellow orange purple black white gray cyan magenta")
    assert index.query(
        "red blue green yellow orange purple black white gray cyan other"
    )[0] == "new"


def test_parse_semantic_judgment_accepts_only_exact_frozen_labels() -> None:
    assert parse_semantic_judgment(" equivalent \n") is True
    assert parse_semantic_judgment("not_equivalent") is False
    for value in ("Equivalent because...", "yes", "not equivalent", ""):
        with pytest.raises(ValueError, match="semantic judgment"):
            parse_semantic_judgment(value)
