from __future__ import annotations

import numpy as np
import pytest

from math_eval.prismatic_lite_pilot_analysis import (
    assign_to_centroids,
    calibration_decision,
    candidate_selection,
    final_pilot_decision,
    gradient_vendi,
    paired_calibration_metrics,
    pairing_group_indices,
    question_gradient,
    selection_null,
    selection_null_torch,
    smallest_cluster_ids,
    stratified_pairing_null,
)


def test_question_gradient_normalizes_each_solution_then_mean_and_final() -> None:
    vectors = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32)

    result = question_gradient(vectors)

    expected = np.array([1.0, 1.0]) / np.sqrt(2.0)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, expected, atol=1e-7)
    np.testing.assert_allclose(question_gradient(vectors[::-1]), result)


@pytest.mark.parametrize(
    "vectors",
    [
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        np.array([[np.nan, 0.0], [1.0, 0.0]]),
        np.array([[1.0, 0.0]]),
    ],
)
def test_question_gradient_rejects_invalid_or_insufficient_vectors(vectors) -> None:
    with pytest.raises(ValueError):
        question_gradient(vectors)


def test_smallest_cluster_ids_selects_exact_count_with_id_tie_break() -> None:
    labels = np.array([0, 0, 0, 1, 2, 2, 3, 3], dtype=np.int64)

    assert smallest_cluster_ids(labels, count=2, cluster_count=4) == (1, 2)
    assert smallest_cluster_ids(labels, count=2, cluster_count=5) == (4, 1)


def test_assign_to_centroids_uses_cosine_and_rejects_nonfinite() -> None:
    vectors = np.array([[2.0, 0.0], [0.0, 4.0], [-1.0, 0.0]])
    centroids = np.array([[1.0, 0.0], [0.0, 1.0]])

    np.testing.assert_array_equal(
        assign_to_centroids(vectors, centroids), np.array([0, 1, 1])
    )
    with pytest.raises(ValueError, match="non-finite"):
        assign_to_centroids(np.array([[np.nan, 0.0]]), centroids)


def test_paired_calibration_metrics_reports_cosine_labels_and_transitions() -> None:
    original = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    changed = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    original_labels = np.array([0, 1, 2, 3])
    changed_labels = np.array([0, 0, 2, 1])

    result = paired_calibration_metrics(
        original,
        changed,
        original_labels,
        changed_labels,
        sparse_cluster_ids=(0, 1),
    )

    assert result["paired_cosine_median"] == pytest.approx(1.0)
    assert result["exact_cluster_agreement"] == pytest.approx(0.5)
    assert result["sparse_dense_agreement"] == pytest.approx(0.75)
    assert result["transition_counts"] == {
        "dense_to_dense": 1,
        "dense_to_sparse": 1,
        "sparse_to_dense": 0,
        "sparse_to_sparse": 2,
    }


def test_stratified_pairing_null_is_repeatable_and_preserves_strata() -> None:
    original = np.eye(4, dtype=np.float64)
    changed = np.eye(4, dtype=np.float64)
    metadata = [
        {"topic": "A", "difficulty": 6.0},
        {"topic": "A", "difficulty": 6.0},
        {"topic": "B", "difficulty": 7.0},
        {"topic": "B", "difficulty": 8.0},
    ]
    original_sparse = np.array([True, False, True, False])
    changed_sparse = original_sparse.copy()

    first = stratified_pairing_null(
        original,
        changed,
        metadata,
        original_sparse,
        changed_sparse,
        draws=20,
        seed=42,
    )
    repeated = stratified_pairing_null(
        original,
        changed,
        metadata,
        original_sparse,
        changed_sparse,
        draws=20,
        seed=42,
    )

    assert first == repeated
    assert len(first["paired_cosine_medians"]) == 20
    assert len(first["sparse_dense_agreements"]) == 20
    assert all(0.0 <= value <= 1.0 for value in first["sparse_dense_agreements"])


def test_pairing_groups_collapse_an_entire_topic_when_any_exact_stratum_is_small() -> None:
    metadata = [
        {"topic": "A", "difficulty": 6.0},
        {"topic": "A", "difficulty": 6.0},
        {"topic": "A", "difficulty": 7.0},
        {"topic": "A", "difficulty": 8.0},
        {"topic": "B", "difficulty": 6.0},
        {"topic": "B", "difficulty": 6.0},
    ]

    assert pairing_group_indices(metadata) == [(0, 1, 2, 3), (4, 5)]


def _passing_calibration() -> dict:
    return {
        "qualified_count": 192,
        "finite": True,
        "seeds": {
            "42": {
                "sparse_dense_agreement": 0.65,
                "null_agreement_median": 0.50,
                "paired_cosine_median": 0.71,
                "null_paired_cosine_median_p90": 0.70,
            },
            "43": {
                "sparse_dense_agreement": 0.70,
                "null_agreement_median": 0.50,
                "paired_cosine_median": 0.75,
                "null_paired_cosine_median_p90": 0.70,
            },
        },
    }


def test_calibration_decision_uses_inclusive_primary_boundaries() -> None:
    result = calibration_decision(_passing_calibration())

    assert result == {"decision": "promising", "failed_conditions": []}


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("qualified_count",), 191),
        (("finite",), False),
        (("seeds", "42", "sparse_dense_agreement"), 0.649),
        (("seeds", "42", "null_agreement_median"), 0.501),
        (("seeds", "43", "paired_cosine_median"), 0.70),
    ],
)
def test_calibration_decision_fails_each_primary_condition(path, value) -> None:
    metrics = _passing_calibration()
    target = metrics
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    result = calibration_decision(metrics)

    assert result["decision"] == "no_go"
    assert result["failed_conditions"]


def test_candidate_selection_uses_seed42_and_reports_seed_agreement() -> None:
    result = candidate_selection(
        labels_seed42=np.array([0, 1, 2, 3]),
        labels_seed43=np.array([3, 1, 0, 2]),
        sparse_seed42=(0, 1),
        sparse_seed43=(0, 1),
    )

    np.testing.assert_array_equal(result["accepted_mask"], [True, True, False, False])
    assert result["accepted_indices"] == [0, 1]
    assert result["seed_sparse_dense_agreement"] == pytest.approx(0.5)


def test_gradient_vendi_is_one_for_duplicates_and_two_for_orthogonal_rows() -> None:
    assert gradient_vendi(np.array([[1.0, 0.0], [2.0, 0.0]])) == pytest.approx(1.0)
    assert gradient_vendi(np.eye(2)) == pytest.approx(2.0)


def test_selection_null_is_deterministic_and_counts_observed_percentile() -> None:
    original = np.array([[1.0, 0.0], [1.0, 0.0]])
    quality = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])

    first = selection_null(
        original,
        quality,
        accepted_indices=(0,),
        draws=20,
        seed=42,
    )
    repeated = selection_null(
        original,
        quality,
        accepted_indices=(0,),
        draws=20,
        seed=42,
    )

    assert first == repeated
    assert first["observed_g_vendi"] > 1.0
    assert 0.0 < first["percentile"] <= 1.0


def test_batched_torch_selection_null_matches_scalar_contract_on_cpu() -> None:
    original = np.array([[1.0, 0.0], [1.0, 0.0]])
    quality = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])

    scalar = selection_null(
        original, quality, accepted_indices=(0,), draws=20, seed=42
    )
    batched = selection_null_torch(
        original,
        quality,
        accepted_indices=(0,),
        draws=20,
        seed=42,
        device="cpu",
        batch_size=3,
    )

    assert batched["percentile"] == scalar["percentile"]
    assert batched["observed_g_vendi"] == pytest.approx(
        scalar["observed_g_vendi"], rel=1e-5
    )
    np.testing.assert_allclose(
        batched["null_g_vendi"], scalar["null_g_vendi"], rtol=1e-5
    )


def test_final_pilot_decision_has_exact_inclusive_boundaries() -> None:
    passing = {
        "quality_count": 1000,
        "accepted_count": 400,
        "seed_sparse_dense_agreement": 0.65,
        "g_vendi_percentile": 0.90,
        "unique_prompts": True,
        "benchmark_contamination_count": 0,
    }

    assert final_pilot_decision(passing) == {
        "decision": "promising",
        "failed_conditions": [],
    }
    for key, bad in (
        ("quality_count", 999),
        ("accepted_count", 399),
        ("seed_sparse_dense_agreement", 0.649),
        ("g_vendi_percentile", 0.899),
        ("unique_prompts", False),
        ("benchmark_contamination_count", 1),
    ):
        value = dict(passing)
        value[key] = bad
        assert final_pilot_decision(value)["decision"] == "no_go"
