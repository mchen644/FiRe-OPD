from pathlib import Path

import numpy as np
import pytest
import torch

from math_eval.opd_proxy_gradient_verify_artifacts import sha256_int_rows
from math_eval.opd_proxy_gradient_statistics import (
    TargetSpace,
    build_permutation_schedule,
    debiased_linear_cka,
    evaluate_subset,
    evaluate_subset_table,
    facility_coverage,
    g_vendi_from_gram,
    inclusive_percentile,
    normalize_rows,
    partition_agreement,
    selected_set_overlap,
    target_dependence_permutation_test,
    target_g_vendi,
    unbiased_hsic,
)
from math_eval.select_gradient_diverse_deepmath import (
    load_official_reference_classes,
)


def _target_space() -> TargetSpace:
    candidate = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    held_out = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    return TargetSpace(
        seed=42,
        candidate_ids=("q0", "q1", "q2", "q3"),
        heldout_ids=("h0", "h1"),
        candidate_vectors=candidate,
        heldout_vectors=held_out,
        full_gradient_norm=np.array([1.0, 2.0, 3.0, 4.0]),
        opd_signal_rms=np.array([0.1, 0.2, 0.3, 0.4]),
        valid_token_count=np.array([10.0, 20.0, 30.0, 40.0]),
        sampled_reverse_kl=np.array([-0.1, 0.0, 0.1, 0.2]),
        response_length=np.array([11.0, 21.0, 31.0, 41.0]),
    )


def _cka_fixture() -> tuple[np.ndarray, np.ndarray]:
    left = np.array(
        [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [-1.0, 1.0], [1.0, -1.0]]
    )
    right = np.array(
        [[1.0, 0.2], [0.7, 1.0], [-0.1, 1.0], [-1.0, 0.8], [0.9, -0.8]]
    )
    return left, right


def test_normalize_rows_returns_float64_unit_rows_without_mutating_input():
    source = np.array([[3.0, 4.0], [0.0, -2.0]], dtype=np.float32)
    before = source.copy()
    normalized = normalize_rows(source)
    assert normalized.dtype == np.float64
    assert normalized.flags.c_contiguous
    np.testing.assert_array_equal(source, before)
    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), 1.0)
    with pytest.raises(ValueError, match="zero-norm"):
        normalize_rows(np.array([[0.0, 0.0], [1.0, 0.0]]))


def test_g_vendi_fixed_gram_and_eigenvalue_policy():
    gram = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float64)
    assert g_vendi_from_gram(gram) == pytest.approx(1.7547653506033232)
    assert target_g_vendi(np.eye(2)) == pytest.approx(2.0)
    assert target_g_vendi(np.ones((2, 1))) == pytest.approx(1.0)
    assert g_vendi_from_gram(np.diag([-1e-7, 1.0])) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="negative Gram eigenvalue"):
        g_vendi_from_gram(np.array([[1.0, 2.0], [2.0, 1.0]]))
    with pytest.raises(ValueError, match="positive trace"):
        g_vendi_from_gram(np.zeros((2, 2)))


def test_target_g_vendi_matches_pinned_official_reference_smoke_fixture():
    vectors = np.array(
        [[1.0, 0.0], [1.0 / 3.0, np.sqrt(8.0) / 3.0]], dtype=np.float64
    )
    local = target_g_vendi(vectors)
    _, vendi_class, provenance = load_official_reference_classes(
        Path("/home/mchen/prismatic-synthesis-reference")
    )
    normalized = torch.from_numpy(normalize_rows(vectors))
    official = vendi_class.compute_vendi_score(normalized.T @ normalized, n=2)
    assert provenance["reference_commit"] == (
        "d9484cd3b5991030b901ac4a3a9e2472dbfac2ad"
    )
    assert local == pytest.approx(1.889881574842311)
    assert local == pytest.approx(official, rel=1e-6, abs=1e-6)


def test_facility_coverage_fixed_half_and_exact_fixtures():
    held_out = np.array([[1.0, 0.0]])
    half = np.array([[0.5, np.sqrt(3.0) / 2.0]])
    exact = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert facility_coverage(held_out, half) == pytest.approx(0.5)
    assert facility_coverage(held_out, exact) == pytest.approx(1.0)


def test_evaluate_subset_returns_scalar_means_and_table_matches_rows():
    space = _target_space()
    selected = np.array([0, 2], dtype=np.int32)
    scores = evaluate_subset(space, selected)
    assert scores.g_vendi == pytest.approx(2.0)
    assert scores.coverage == pytest.approx(1.0)
    assert scores.full_gradient_norm == pytest.approx(2.0)
    assert scores.opd_signal_rms == pytest.approx(0.2)
    assert scores.valid_token_count == pytest.approx(20.0)
    assert scores.sampled_reverse_kl == pytest.approx(0.0)
    assert scores.response_length == pytest.approx(21.0)

    subsets = np.array([[0, 2], [1, 3]], dtype=np.int32)
    table = evaluate_subset_table(space, subsets)
    assert set(table) == {
        "g_vendi",
        "coverage",
        "full_gradient_norm",
        "opd_signal_rms",
        "valid_token_count",
        "sampled_reverse_kl",
        "response_length",
    }
    assert all(values.shape == (2,) for values in table.values())
    second = evaluate_subset(space, subsets[1])
    for name, values in table.items():
        assert values[1] == pytest.approx(getattr(second, name))


@pytest.mark.parametrize(
    "positions",
    [
        np.array([2, 0], dtype=np.int32),
        np.array([0, 0], dtype=np.int32),
        np.array([-1, 2], dtype=np.int32),
        np.array([0, 4], dtype=np.int32),
        np.array([0.0, 2.0]),
    ],
)
def test_subset_positions_must_be_sorted_unique_in_range_integers(positions):
    with pytest.raises(ValueError, match="subset positions"):
        evaluate_subset(_target_space(), positions)


def test_target_space_rejects_id_scalar_and_vector_misalignment():
    space = _target_space()
    with pytest.raises(ValueError, match="candidate scalar"):
        TargetSpace(
            **{
                **space.as_init_dict(),
                "full_gradient_norm": np.array([1.0, 2.0]),
            }
        )
    with pytest.raises(ValueError, match="feature dimension"):
        TargetSpace(
            **{
                **space.as_init_dict(),
                "heldout_vectors": np.ones((2, 4)),
            }
        )


def test_inclusive_percentile_counts_ties_and_rejects_empty_or_nonfinite():
    assert inclusive_percentile(np.array([1.0, 2.0, 2.0, 4.0]), 2.0) == 0.75
    with pytest.raises(ValueError, match="nonempty"):
        inclusive_percentile(np.array([]), 1.0)
    with pytest.raises(ValueError, match="finite"):
        inclusive_percentile(np.array([1.0, np.nan]), 1.0)


def test_unbiased_cka_fixed_positive_negative_and_different_dimensions():
    left, right = _cka_fixture()
    assert debiased_linear_cka(left, right) == pytest.approx(0.998739479819789)
    permutation = np.array([0, 2, 1, 4, 3])
    assert debiased_linear_cka(left, right[permutation]) == pytest.approx(
        -0.446182589285757
    )
    wider_right = np.column_stack([right, np.arange(5, dtype=np.float64) + 1.0])
    assert np.isfinite(debiased_linear_cka(left, wider_right))


def test_unbiased_hsic_matches_fixed_fixture_and_rejects_bad_self_hsic():
    left, right = _cka_fixture()
    left_kernel = normalize_rows(left) @ normalize_rows(left).T
    right_kernel = normalize_rows(right) @ normalize_rows(right).T
    assert unbiased_hsic(left_kernel, right_kernel) == pytest.approx(
        0.44262555424286704
    )
    with pytest.raises(ValueError, match="self-HSIC"):
        debiased_linear_cka(np.ones((5, 2)), right)


def test_cka_permutation_schedule_and_hash_are_pinned():
    rows = build_permutation_schedule(n=5, draws=4, seed=2026071403)
    assert rows.dtype.str == "<i4"
    assert rows.flags.c_contiguous
    assert rows.tolist() == [
        [0, 2, 1, 4, 3],
        [1, 2, 0, 3, 4],
        [2, 4, 3, 0, 1],
        [3, 0, 4, 2, 1],
    ]
    assert sha256_int_rows(rows) == (
        "c4c02ab70a6edef9c1050d1ee92bbe05d4b541929687ce3e386677d2767ef0ee"
    )


def test_efficient_permutation_cka_matches_brute_force_for_every_row():
    left, right = _cka_fixture()
    schedule = build_permutation_schedule(n=5, draws=4, seed=2026071403)
    result = target_dependence_permutation_test(
        left,
        right,
        draws=4,
        seed=2026071403,
        permutation_schedule=schedule,
    )
    expected = np.array(
        [debiased_linear_cka(left, right[row]) for row in schedule]
    )
    np.testing.assert_allclose(result["null_cka"], expected, rtol=1e-14, atol=1e-14)
    assert result["observed_cka"] == pytest.approx(0.998739479819789)
    assert result["permutation_schedule_sha256"] == sha256_int_rows(schedule)


def test_permutation_p_value_uses_plus_one_and_counts_ties():
    left, right = _cka_fixture()
    schedule = np.array(
        [[0, 1, 2, 3, 4], [0, 2, 1, 4, 3]], dtype="<i4"
    )
    result = target_dependence_permutation_test(
        left,
        right,
        draws=2,
        permutation_schedule=schedule,
    )
    assert result["exceedance_count"] == 1
    assert result["p_value"] == pytest.approx(2.0 / 3.0)
    assert result["null_cka"][0] == pytest.approx(result["observed_cka"])


def test_partition_agreement_fixed_ami_ari_fixtures():
    left = np.array([0, 0, 1, 1, 2, 2])
    cases = [
        (np.array([1, 1, 0, 0, 2, 2]), 1.0),
        (np.array([0, 1, 0, 1, 2, 2]), 1.0 / 6.0),
        (np.array([0, 1, 2, 0, 1, 2]), -1.0 / 4.0),
    ]
    for right, expected in cases:
        result = partition_agreement(left, right)
        assert result["adjusted_mutual_info"] == pytest.approx(expected)
        assert result["adjusted_rand_index"] == pytest.approx(expected)


def test_selected_overlap_fixture_and_validation():
    result = selected_set_overlap(
        [0, 1, 2, 3], [2, 3, 4, 5], candidate_count=10, selected_size=4
    )
    assert result["intersection_count"] == 2
    assert result["jaccard"] == pytest.approx(1.0 / 3.0)
    assert result["chance_adjusted_overlap"] == pytest.approx(1.0 / 6.0)
    with pytest.raises(ValueError, match="sorted unique"):
        selected_set_overlap(
            [1, 0, 2, 3], [2, 3, 4, 5], candidate_count=10, selected_size=4
        )
