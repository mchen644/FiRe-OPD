import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from math_eval.deepmath_gradient_diversity import (
    balanced_round_robin,
    cluster_size_summary,
    join_filtered_rows,
    normalize_math_answer,
    official_shard_bounds,
    sha256_file,
    strip_opd_instruction,
    validate_gradient_matrix,
)


OPD_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
MISSING = object()


def _filtered_row(question: str = "Solve x.", answer: object = "2") -> dict:
    return {
        "prompt": [
            {
                "role": "user",
                "content": f"{question}\n{OPD_INSTRUCTION}",
            }
        ],
        "reward_model": {"ground_truth": answer},
        "extra_info": {"index": 7, "split": "train"},
    }


def _original_row(question: str = "Solve x.", answer: object = "2") -> dict:
    return {
        "question": question,
        "final_answer": answer,
        "difficulty": 6.0,
        "topic": "Algebra",
        "r1_solution_1": "Reasoning... \\boxed{2}",
        "r1_solution_2": "unused",
        "r1_solution_3": "unused",
    }


def test_strip_opd_instruction_removes_only_the_exact_terminal_prompt_suffix():
    prompt = f"Solve x.\n{OPD_INSTRUCTION}"

    assert strip_opd_instruction(prompt) == "Solve x."
    assert strip_opd_instruction(prompt + " ") == prompt + " "
    assert strip_opd_instruction(OPD_INSTRUCTION + "\nSolve x.") == (
        OPD_INSTRUCTION + "\nSolve x."
    )


def test_normalize_math_answer_handles_scalar_objects_and_formatting():
    assert normalize_math_answer(" $1,024$ ") == "1024"
    assert normalize_math_answer(2) == "2"


def test_join_filtered_rows_uses_exact_question_and_r1_solution_one():
    filtered = [_filtered_row()]
    original = [_original_row()]

    result = join_filtered_rows(filtered, original, expected_count=1)

    assert result == [
        {
            "id": "deepmath-level6-000000",
            "prompt": "Solve x.",
            "completion": "Reasoning... \\boxed{2}",
            "source_row_index": 0,
            "original_dataset_index": 0,
            "topic": "Algebra",
            "difficulty": 6.0,
        }
    ]


def test_join_filtered_rows_ids_follow_source_order_not_original_dataset_index():
    originals = [
        _original_row(question="Different question.", answer="99"),
        _original_row(),
    ]

    result = join_filtered_rows([_filtered_row()], originals, expected_count=1)

    assert result[0]["id"] == "deepmath-level6-000000"
    assert result[0]["source_row_index"] == 0
    assert result[0]["original_dataset_index"] == 1


def test_join_filtered_rows_rejects_a_missing_exact_question_match():
    with pytest.raises(ValueError, match="no original question match"):
        join_filtered_rows(
            [_filtered_row(question="Solve y.")],
            [_original_row(question="Solve x.")],
            expected_count=1,
        )


def test_join_filtered_rows_disambiguates_duplicate_keys_by_surrounding_order():
    originals = [
        _original_row(question="Before."),
        _original_row(question="Duplicate."),
        _original_row(question="Left anchor."),
        _original_row(question="Duplicate."),
        _original_row(question="Right anchor."),
    ]
    filtered = [
        _filtered_row(question="Left anchor."),
        _filtered_row(question="Duplicate."),
        _filtered_row(question="Right anchor."),
    ]

    result = join_filtered_rows(filtered, originals, expected_count=3)

    assert [row["original_dataset_index"] for row in result] == [2, 3, 4]
    assert [row["id"] for row in result] == [
        "deepmath-level6-000000",
        "deepmath-level6-000001",
        "deepmath-level6-000002",
    ]


def test_join_filtered_rows_rejects_truly_ambiguous_duplicate_sequence():
    originals = [
        _original_row(question="Left anchor."),
        _original_row(question="Duplicate."),
        _original_row(question="Duplicate."),
        _original_row(question="Right anchor."),
    ]
    filtered = [
        _filtered_row(question="Left anchor."),
        _filtered_row(question="Duplicate."),
        _filtered_row(question="Right anchor."),
    ]

    with pytest.raises(ValueError, match="ambiguous ordered original match"):
        join_filtered_rows(filtered, originals, expected_count=3)


def test_join_filtered_rows_rejects_empty_r1_solution_one():
    original = _original_row()
    original["r1_solution_1"] = "   "

    with pytest.raises(ValueError, match="empty r1_solution_1"):
        join_filtered_rows([_filtered_row()], [original], expected_count=1)


def test_join_filtered_rows_rejects_answer_mismatch():
    with pytest.raises(ValueError, match="answer mismatch"):
        join_filtered_rows(
            [_filtered_row(answer="3")],
            [_original_row(answer="2")],
            expected_count=1,
        )


@pytest.mark.parametrize(
    "answer",
    [pytest.param(MISSING, id="missing"), pytest.param(None, id="none"), pytest.param("   ", id="blank")],
)
def test_join_filtered_rows_rejects_missing_or_empty_ground_truth(answer):
    filtered = _filtered_row()
    if answer is MISSING:
        del filtered["reward_model"]["ground_truth"]
    else:
        filtered["reward_model"]["ground_truth"] = answer

    with pytest.raises(ValueError, match="missing or empty ground_truth"):
        join_filtered_rows([filtered], [_original_row()], expected_count=1)


@pytest.mark.parametrize(
    "answer",
    [pytest.param(MISSING, id="missing"), pytest.param(None, id="none"), pytest.param("   ", id="blank")],
)
def test_join_filtered_rows_rejects_missing_or_empty_original_final_answer(answer):
    original = _original_row()
    if answer is MISSING:
        del original["final_answer"]
    else:
        original["final_answer"] = answer

    with pytest.raises(ValueError, match="missing or empty final_answer"):
        join_filtered_rows([_filtered_row()], [original], expected_count=1)


@pytest.mark.parametrize(
    "answer",
    [pytest.param(MISSING, id="missing"), pytest.param(None, id="none"), pytest.param("   ", id="blank")],
)
def test_join_filtered_rows_ignores_unrelated_incomplete_original_rows(answer):
    incomplete_before = _original_row(question="Unrelated before.")
    incomplete_after = _original_row(question="Unrelated after.")
    if answer is MISSING:
        del incomplete_before["final_answer"]
        del incomplete_after["final_answer"]
    else:
        incomplete_before["final_answer"] = answer
        incomplete_after["final_answer"] = answer

    result = join_filtered_rows(
        [_filtered_row()],
        [incomplete_before, _original_row(), incomplete_after],
        expected_count=1,
    )

    assert result[0]["original_dataset_index"] == 1


def test_join_filtered_rows_rejects_malformed_chat_prompt():
    filtered = _filtered_row()
    filtered["prompt"] = [{"role": "assistant", "content": "Solve x."}]

    with pytest.raises(ValueError, match="malformed chat prompt"):
        join_filtered_rows([filtered], [_original_row()], expected_count=1)


def test_join_filtered_rows_rejects_unexpected_row_count():
    with pytest.raises(ValueError, match="expected 2 filtered rows, got 1"):
        join_filtered_rows(
            [_filtered_row()],
            [_original_row()],
            expected_count=2,
        )


@pytest.mark.parametrize(
    ("shard", "expected"),
    [(0, (0, 7131)), (1, (7131, 14262)), (7, (49917, 57046))],
)
def test_official_shard_bounds_matches_reference_partition(shard, expected):
    assert official_shard_bounds(57046, 8, shard) == expected


@pytest.mark.parametrize(
    ("num_shards", "shard_index"),
    [(0, 0), (-1, 0), (8, -1), (8, 8)],
)
def test_official_shard_bounds_rejects_invalid_shard_count_or_index(
    num_shards, shard_index
):
    with pytest.raises(ValueError):
        official_shard_bounds(57046, num_shards, shard_index)


def test_sha256_file_hash_changes_when_file_bytes_change(tmp_path: Path):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"first payload")

    first = sha256_file(path)
    path.write_bytes(b"second payload")
    second = sha256_file(path)

    assert first == hashlib.sha256(b"first payload").hexdigest()
    assert second == hashlib.sha256(b"second payload").hexdigest()
    assert first != second


def test_validate_gradient_matrix_accepts_exact_finite_nonzero_rows():
    validate_gradient_matrix(
        ["row-a", "row-b"],
        torch.tensor([[1.0, 0.0], [0.0, -2.0]]),
        ["row-a", "row-b"],
    )


def test_validate_gradient_matrix_rejects_id_order_mismatch():
    with pytest.raises(ValueError, match="gradient IDs do not exactly match"):
        validate_gradient_matrix(
            ["row-b", "row-a"],
            torch.ones(2, 2),
            ["row-a", "row-b"],
        )


def test_validate_gradient_matrix_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="duplicate gradient IDs"):
        validate_gradient_matrix(
            ["row-a", "row-a"],
            torch.ones(2, 2),
            ["row-a", "row-a"],
        )


@pytest.mark.parametrize(
    "gradients",
    [
        torch.tensor([[1.0, float("nan")]]),
        torch.tensor([[1.0, float("inf")]]),
    ],
)
def test_validate_gradient_matrix_rejects_non_finite_gradients(gradients):
    with pytest.raises(ValueError, match="non-finite"):
        validate_gradient_matrix(["row-a"], gradients, ["row-a"])


def test_validate_gradient_matrix_rejects_zero_norm_gradients():
    with pytest.raises(ValueError, match="zero-norm"):
        validate_gradient_matrix(
            ["row-a", "row-b"],
            torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
            ["row-a", "row-b"],
        )


@pytest.mark.parametrize(
    "gradients",
    [torch.ones(2), torch.ones(1, 2)],
)
def test_validate_gradient_matrix_rejects_invalid_shape(gradients):
    with pytest.raises(ValueError, match="two-dimensional with one row per ID"):
        validate_gradient_matrix(
            ["row-a", "row-b"],
            gradients,
            ["row-a", "row-b"],
        )


def test_balanced_round_robin_selects_exact_unique_size_and_is_deterministic():
    labels = [0, 0, 0, 1, 1, 2, 2, 2, 2]

    first = balanced_round_robin(labels, target_size=7, seed=42)
    second = balanced_round_robin(labels, target_size=7, seed=42)
    another_seed = balanced_round_robin(labels, target_size=7, seed=43)

    assert first == [3, 7, 2, 4, 0, 5, 1]
    assert first == second
    assert another_seed == [0, 8, 3, 4, 1, 6, 2]
    assert len(first) == 7
    assert len(set(first)) == 7
    assert set(labels[i] for i in first[:3]) == {0, 1, 2}

    full_selection = balanced_round_robin(labels, target_size=len(labels), seed=42)
    round_boundaries = [(0, 3), (3, 6), (6, 8), (8, 9)]
    for start, end in round_boundaries:
        round_labels = [labels[index] for index in full_selection[start:end]]
        assert len(round_labels) == len(set(round_labels))


def test_balanced_round_robin_rejects_insufficient_rows():
    with pytest.raises(ValueError, match="cannot select 4 rows from 3"):
        balanced_round_robin([0, 1, 2], target_size=4, seed=42)


def test_balanced_round_robin_rejects_negative_cluster_labels():
    with pytest.raises(ValueError, match="cluster labels must be non-negative"):
        balanced_round_robin([0, -1, 2], target_size=2, seed=42)


def test_balanced_round_robin_rejects_negative_target_size():
    with pytest.raises(ValueError, match="target_size must be non-negative"):
        balanced_round_robin([0, 1, 2], target_size=-1, seed=42)


def test_cluster_size_summary_includes_empty_requested_clusters():
    summary = cluster_size_summary([0, 0, 1, 2, 2, 2], requested_clusters=4)

    assert summary == pytest.approx(
        {
            "requested_clusters": 4,
            "non_empty_clusters": 3,
            "min_cluster_size": 0,
            "median_cluster_size": 1.5,
            "mean_cluster_size": 1.5,
            "p90_cluster_size": 2.7,
            "p99_cluster_size": 2.97,
            "max_cluster_size": 3,
            "singleton_cluster_ratio": 0.25,
            "largest_cluster_fraction": 0.5,
        }
    )


def test_cluster_size_summary_accepts_numpy_cluster_labels():
    summary = cluster_size_summary(np.array([0, 0, 2]), requested_clusters=4)

    assert summary["non_empty_clusters"] == 2
    assert summary["largest_cluster_fraction"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("labels", [[-1, 0], [0, 4]])
def test_cluster_size_summary_rejects_labels_outside_requested_range(labels):
    with pytest.raises(ValueError, match="cluster labels must be in"):
        cluster_size_summary(labels, requested_clusters=4)
