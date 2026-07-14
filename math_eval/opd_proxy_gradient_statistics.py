"""Pure target-space metrics and agreement diagnostics for OPD verification."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, fields

import numpy as np

from math_eval.opd_proxy_gradient_verify_artifacts import sha256_int_rows


EIGENVALUE_TOLERANCE = 1e-7
CKA_PERMUTATION_SEED = 2026071403
CKA_PERMUTATION_DRAWS = 10_000
_SCALAR_FIELDS = (
    "full_gradient_norm",
    "opd_signal_rms",
    "valid_token_count",
    "sampled_reverse_kl",
    "response_length",
)
_SCORE_FIELDS = ("g_vendi", "coverage", *_SCALAR_FIELDS)


def _readonly_float64(value: object, description: str) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True, order="C")
    if not np.isfinite(array).all():
        raise ValueError(f"{description} must contain only finite values")
    array.setflags(write=False)
    return array


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """Return a finite, C-contiguous float64 matrix with unit-norm rows."""
    array = np.array(vectors, dtype=np.float64, copy=True, order="C")
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("row normalization requires a nonempty rank-2 matrix")
    if not np.isfinite(array).all():
        raise ValueError("row normalization requires finite vectors")
    norms = np.linalg.norm(array, axis=1)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError("row normalization encountered a zero-norm or invalid row")
    array /= norms[:, None]
    return np.ascontiguousarray(array)


@dataclass(frozen=True)
class TargetSpace:
    seed: int
    candidate_ids: tuple[str, ...]
    heldout_ids: tuple[str, ...]
    candidate_vectors: np.ndarray
    heldout_vectors: np.ndarray
    full_gradient_norm: np.ndarray
    opd_signal_rms: np.ndarray
    valid_token_count: np.ndarray
    sampled_reverse_kl: np.ndarray
    response_length: np.ndarray

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("target seed must be an integer")
        candidate_ids = tuple(self.candidate_ids)
        heldout_ids = tuple(self.heldout_ids)
        for name, values in (
            ("candidate", candidate_ids),
            ("held-out", heldout_ids),
        ):
            if not values or any(
                not isinstance(value, str) or not value for value in values
            ):
                raise ValueError(f"target {name} IDs must be nonempty strings")
            if len(set(values)) != len(values):
                raise ValueError(f"target {name} IDs must be unique")
        if set(candidate_ids).intersection(heldout_ids):
            raise ValueError("target candidate and held-out IDs must be disjoint")
        candidate_vectors = normalize_rows(self.candidate_vectors)
        heldout_vectors = normalize_rows(self.heldout_vectors)
        if candidate_vectors.shape[0] != len(candidate_ids):
            raise ValueError("target candidate vector/ID count mismatch")
        if heldout_vectors.shape[0] != len(heldout_ids):
            raise ValueError("target held-out vector/ID count mismatch")
        if candidate_vectors.shape[1] != heldout_vectors.shape[1]:
            raise ValueError("target candidate and held-out feature dimensions differ")
        candidate_vectors.setflags(write=False)
        heldout_vectors.setflags(write=False)
        object.__setattr__(self, "candidate_ids", candidate_ids)
        object.__setattr__(self, "heldout_ids", heldout_ids)
        object.__setattr__(self, "candidate_vectors", candidate_vectors)
        object.__setattr__(self, "heldout_vectors", heldout_vectors)

        for name in _SCALAR_FIELDS:
            array = _readonly_float64(getattr(self, name), f"target {name}")
            if array.ndim != 1 or array.shape != (len(candidate_ids),):
                raise ValueError(
                    f"target candidate scalar {name} must have one value per candidate"
                )
            if name != "sampled_reverse_kl" and np.any(array < 0):
                raise ValueError(f"target candidate scalar {name} must be nonnegative")
            object.__setattr__(self, name, array)

    def as_init_dict(self) -> dict[str, object]:
        """Return constructor fields for deterministic test/adapter transformations."""
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True)
class SubsetScores:
    g_vendi: float
    coverage: float
    full_gradient_norm: float
    opd_signal_rms: float
    valid_token_count: float
    sampled_reverse_kl: float
    response_length: float

    def as_dict(self) -> dict[str, float]:
        return {field.name: float(getattr(self, field.name)) for field in fields(self)}


def g_vendi_from_gram(gram: np.ndarray) -> float:
    """Compute effective rank from an explicitly symmetrized float64 Gram matrix."""
    array = np.asarray(gram, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[0] != array.shape[1]:
        raise ValueError("G-Vendi requires a nonempty square Gram matrix")
    if not np.isfinite(array).all():
        raise ValueError("G-Vendi Gram matrix must be finite")
    symmetric = 0.5 * (array + array.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    if eigenvalues.min(initial=0.0) < -EIGENVALUE_TOLERANCE:
        raise ValueError("negative Gram eigenvalue exceeds tolerance")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total = float(eigenvalues.sum())
    if not np.isfinite(eigenvalues).all() or not math.isfinite(total) or total <= 0:
        raise ValueError("Gram spectrum must be finite with positive trace")
    probabilities = eigenvalues / total
    positive = probabilities > 0
    entropy = -float(
        np.sum(probabilities[positive] * np.log(probabilities[positive]))
    )
    result = float(np.exp(entropy))
    if not math.isfinite(result):
        raise ValueError("G-Vendi result must be finite")
    return result


def target_g_vendi(vectors: np.ndarray) -> float:
    normalized = normalize_rows(vectors)
    gram = normalized @ normalized.T
    gram = 0.5 * (gram + gram.T)
    return g_vendi_from_gram(gram)


def facility_coverage(
    heldout_vectors: np.ndarray, selected_vectors: np.ndarray
) -> float:
    heldout = normalize_rows(heldout_vectors)
    selected = normalize_rows(selected_vectors)
    if heldout.shape[1] != selected.shape[1]:
        raise ValueError("coverage vectors must share one feature dimension")
    similarities = heldout @ selected.T
    result = float(np.max(similarities, axis=1).mean(dtype=np.float64))
    if not math.isfinite(result):
        raise ValueError("facility coverage must be finite")
    return result


def _validate_positions(
    positions: object, *, candidate_count: int, description: str = "subset positions"
) -> np.ndarray:
    array = np.asarray(positions)
    if array.ndim != 1 or array.size == 0 or not np.issubdtype(
        array.dtype, np.integer
    ) or np.issubdtype(array.dtype, np.bool_):
        raise ValueError(f"{description} must be a nonempty integer vector")
    normalized = np.asarray(array, dtype=np.int64)
    if (
        np.any(normalized < 0)
        or np.any(normalized >= candidate_count)
        or np.any(np.diff(normalized) <= 0)
    ):
        raise ValueError(
            f"{description} must be sorted, unique, and within candidate range"
        )
    return normalized


def _validate_subset_table(subsets: object, candidate_count: int) -> np.ndarray:
    array = np.asarray(subsets)
    if (
        array.ndim != 2
        or array.shape[0] == 0
        or array.shape[1] == 0
        or not np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise ValueError("subset table must be a nonempty rank-2 integer array")
    normalized = np.asarray(array, dtype=np.int64)
    for row in normalized:
        _validate_positions(row, candidate_count=candidate_count)
    return normalized


def evaluate_subset_table(
    space: TargetSpace, subsets: np.ndarray
) -> dict[str, np.ndarray]:
    """Evaluate a fixed subset table once in one normalized target space."""
    if not isinstance(space, TargetSpace):
        raise TypeError("subset evaluation requires a TargetSpace")
    positions = _validate_subset_table(subsets, len(space.candidate_ids))
    candidate_gram = space.candidate_vectors @ space.candidate_vectors.T
    candidate_gram = 0.5 * (candidate_gram + candidate_gram.T)
    heldout_similarity = space.heldout_vectors @ space.candidate_vectors.T
    row_count = positions.shape[0]
    g_vendi = np.empty(row_count, dtype=np.float64)
    coverage = np.empty(row_count, dtype=np.float64)
    for index, selected in enumerate(positions):
        g_vendi[index] = g_vendi_from_gram(
            candidate_gram[np.ix_(selected, selected)]
        )
        coverage[index] = float(
            np.max(heldout_similarity[:, selected], axis=1).mean(dtype=np.float64)
        )
    result: dict[str, np.ndarray] = {
        "g_vendi": np.ascontiguousarray(g_vendi),
        "coverage": np.ascontiguousarray(coverage),
    }
    for name in _SCALAR_FIELDS:
        values = getattr(space, name)
        result[name] = np.ascontiguousarray(
            values[positions].mean(axis=1, dtype=np.float64)
        )
    for name, values in result.items():
        if values.shape != (row_count,) or not np.isfinite(values).all():
            raise ValueError(f"subset score table {name} is invalid")
    return result


def evaluate_subset(space: TargetSpace, selected_positions: np.ndarray) -> SubsetScores:
    selected = _validate_positions(
        selected_positions, candidate_count=len(space.candidate_ids)
    )
    table = evaluate_subset_table(space, selected.reshape(1, -1))
    return SubsetScores(**{name: float(table[name][0]) for name in _SCORE_FIELDS})


def inclusive_percentile(null_scores: np.ndarray, observed: float) -> float:
    null = np.asarray(null_scores, dtype=np.float64)
    if null.ndim != 1 or null.size == 0:
        raise ValueError("percentile null scores must be a nonempty vector")
    if not np.isfinite(null).all() or not math.isfinite(float(observed)):
        raise ValueError("percentile inputs must be finite")
    return float(np.count_nonzero(null <= float(observed)) / null.size)


def _validated_kernel(kernel: np.ndarray, description: str) -> np.ndarray:
    array = np.array(kernel, dtype=np.float64, copy=True, order="C")
    if (
        array.ndim != 2
        or array.shape[0] < 4
        or array.shape[0] != array.shape[1]
    ):
        raise ValueError(
            "unbiased HSIC requires aligned square kernels with n >= 4"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{description} kernel must be finite")
    if not np.allclose(array, array.T, rtol=1e-12, atol=1e-12):
        raise ValueError(f"{description} kernel must be symmetric")
    np.fill_diagonal(array, 0.0)
    return array


def _unbiased_hsic_zero_diagonal(left: np.ndarray, right: np.ndarray) -> float:
    n = left.shape[0]
    trace_product = float(np.sum(left * right.T, dtype=np.float64))
    sum_product = float(left.sum(axis=0) @ right.sum(axis=1))
    numerator = (
        trace_product
        + float(left.sum() * right.sum()) / ((n - 1) * (n - 2))
        - 2.0 * sum_product / (n - 2)
    )
    result = float(numerator / (n * (n - 3)))
    if not math.isfinite(result):
        raise ValueError("unbiased HSIC result must be finite")
    return result


def unbiased_hsic(left_kernel: np.ndarray, right_kernel: np.ndarray) -> float:
    left = _validated_kernel(left_kernel, "left")
    right = _validated_kernel(right_kernel, "right")
    if left.shape != right.shape:
        raise ValueError("unbiased HSIC requires aligned kernels")
    return _unbiased_hsic_zero_diagonal(left, right)


def _linear_kernel(vectors: np.ndarray) -> np.ndarray:
    normalized = normalize_rows(vectors)
    kernel = normalized @ normalized.T
    return 0.5 * (kernel + kernel.T)


def _cka_from_kernels(left_kernel: np.ndarray, right_kernel: np.ndarray) -> float:
    left = _validated_kernel(left_kernel, "left")
    right = _validated_kernel(right_kernel, "right")
    if left.shape != right.shape:
        raise ValueError("debiased CKA requires aligned question rows")
    left_self = _unbiased_hsic_zero_diagonal(left, left)
    right_self = _unbiased_hsic_zero_diagonal(right, right)
    if left_self <= 0 or right_self <= 0:
        raise ValueError("debiased CKA has non-positive self-HSIC")
    cross = _unbiased_hsic_zero_diagonal(left, right)
    result = float(cross / math.sqrt(left_self * right_self))
    if not math.isfinite(result):
        raise ValueError("debiased CKA result must be finite")
    return result


def debiased_linear_cka(
    left_vectors: np.ndarray, right_vectors: np.ndarray
) -> float:
    left = np.asarray(left_vectors)
    right = np.asarray(right_vectors)
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("debiased CKA requires aligned rank-2 feature matrices")
    if left.shape[0] < 4:
        raise ValueError("debiased CKA requires at least four aligned questions")
    return _cka_from_kernels(_linear_kernel(left), _linear_kernel(right))


def build_permutation_schedule(
    *, n: int, draws: int = CKA_PERMUTATION_DRAWS, seed: int = CKA_PERMUTATION_SEED
) -> np.ndarray:
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("permutation size must be a positive integer")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws <= 0:
        raise ValueError("permutation draws must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("permutation seed must be a nonnegative integer")
    generator = np.random.Generator(np.random.PCG64(seed))
    schedule = np.empty((draws, n), dtype=np.dtype("<i4"))
    for index in range(draws):
        schedule[index] = generator.permutation(n)
    return np.ascontiguousarray(schedule)


def _validate_permutation_schedule(
    schedule: np.ndarray, *, draws: int, n: int
) -> np.ndarray:
    array = np.asarray(schedule)
    if array.ndim != 2 or array.shape != (draws, n) or not np.issubdtype(
        array.dtype, np.integer
    ) or np.issubdtype(array.dtype, np.bool_):
        raise ValueError("permutation schedule shape/dtype mismatch")
    normalized = np.ascontiguousarray(array, dtype=np.dtype("<i4"))
    expected = np.arange(n, dtype=np.int32)
    for row in normalized:
        if not np.array_equal(np.sort(row), expected):
            raise ValueError("permutation schedule row is not a permutation")
    return normalized


def target_dependence_permutation_test(
    left_vectors: np.ndarray,
    right_vectors: np.ndarray,
    *,
    draws: int = CKA_PERMUTATION_DRAWS,
    seed: int = CKA_PERMUTATION_SEED,
    permutation_schedule: np.ndarray | None = None,
) -> dict[str, object]:
    """Run the fixed CKA permutation test with O(n^2) work per replicate."""
    left_values = np.asarray(left_vectors)
    right_values = np.asarray(right_vectors)
    if (
        left_values.ndim != 2
        or right_values.ndim != 2
        or left_values.shape[0] != right_values.shape[0]
        or left_values.shape[0] < 4
    ):
        raise ValueError("target CKA test requires at least four aligned rows")
    n = left_values.shape[0]
    if isinstance(draws, bool) or not isinstance(draws, int) or draws <= 0:
        raise ValueError("target CKA draws must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("target CKA seed must be a nonnegative integer")
    if permutation_schedule is None:
        schedule = build_permutation_schedule(n=n, draws=draws, seed=seed)
    else:
        schedule = _validate_permutation_schedule(
            permutation_schedule, draws=draws, n=n
        )
    left = _validated_kernel(_linear_kernel(left_values), "left")
    right = _validated_kernel(_linear_kernel(right_values), "right")
    left_self = _unbiased_hsic_zero_diagonal(left, left)
    right_self = _unbiased_hsic_zero_diagonal(right, right)
    if left_self <= 0 or right_self <= 0:
        raise ValueError("target CKA test has non-positive self-HSIC")
    denominator = math.sqrt(left_self * right_self)
    observed = _unbiased_hsic_zero_diagonal(left, right) / denominator

    left_column_sums = left.sum(axis=0)
    right_row_sums = right.sum(axis=1)
    total_term = float(left.sum() * right.sum()) / ((n - 1) * (n - 2))
    scale = n * (n - 3)
    null = np.empty(draws, dtype=np.float64)
    for index, permutation in enumerate(schedule):
        permuted = right[np.ix_(permutation, permutation)]
        trace_product = float(np.sum(left * permuted.T, dtype=np.float64))
        sum_product = float(left_column_sums @ right_row_sums[permutation])
        hsic = (
            trace_product + total_term - 2.0 * sum_product / (n - 2)
        ) / scale
        null[index] = hsic / denominator
    if not np.isfinite(null).all() or not math.isfinite(observed):
        raise ValueError("target CKA permutation values must be finite")
    exceedances = int(np.count_nonzero(null >= observed))
    return {
        "observed_cka": float(observed),
        "null_cka": null,
        "p_value": float((1 + exceedances) / (draws + 1)),
        "exceedance_count": exceedances,
        "draws": draws,
        "seed": seed,
        "permutation_schedule_sha256": sha256_int_rows(schedule),
        "tie_rule": "null_cka_greater_than_or_equal_to_observed",
    }


def _validate_labels(labels: object, description: str) -> np.ndarray:
    array = np.asarray(labels)
    if (
        array.ndim != 1
        or array.size < 2
        or not np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise ValueError(f"{description} labels must be a one-dimensional integer array")
    normalized = np.asarray(array, dtype=np.int64)
    if np.any(normalized < 0):
        raise ValueError(f"{description} labels must be nonnegative")
    return normalized


def partition_agreement(
    left_labels: np.ndarray, right_labels: np.ndarray
) -> dict[str, float]:
    left = _validate_labels(left_labels, "left")
    right = _validate_labels(right_labels, "right")
    if left.shape != right.shape:
        raise ValueError("partition labels must have aligned rows")
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    ami = float(
        adjusted_mutual_info_score(left, right, average_method="arithmetic")
    )
    ari = float(adjusted_rand_score(left, right))
    if not math.isfinite(ami) or not math.isfinite(ari):
        raise ValueError("partition agreement must be finite")
    return {"adjusted_mutual_info": ami, "adjusted_rand_index": ari}


def selected_set_overlap(
    left: Sequence[int],
    right: Sequence[int],
    *,
    candidate_count: int,
    selected_size: int,
) -> dict[str, float | int]:
    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or candidate_count <= 1
    ):
        raise ValueError("candidate_count must be an integer greater than one")
    if (
        isinstance(selected_size, bool)
        or not isinstance(selected_size, int)
        or selected_size <= 0
        or selected_size >= candidate_count
    ):
        raise ValueError("selected_size must be in [1, candidate_count)")
    left_positions = _validate_positions(
        left, candidate_count=candidate_count, description="left sorted unique positions"
    )
    right_positions = _validate_positions(
        right,
        candidate_count=candidate_count,
        description="right sorted unique positions",
    )
    if left_positions.size != selected_size or right_positions.size != selected_size:
        raise ValueError("selected overlap inputs must match selected_size")
    intersection = int(np.intersect1d(left_positions, right_positions).size)
    union = 2 * selected_size - intersection
    expected_intersection = selected_size * selected_size / candidate_count
    denominator = selected_size - expected_intersection
    return {
        "intersection_count": intersection,
        "jaccard": float(intersection / union),
        "chance_adjusted_overlap": float(
            (intersection - expected_intersection) / denominator
        ),
    }


__all__ = [
    "SubsetScores",
    "TargetSpace",
    "build_permutation_schedule",
    "debiased_linear_cka",
    "evaluate_subset",
    "evaluate_subset_table",
    "facility_coverage",
    "g_vendi_from_gram",
    "inclusive_percentile",
    "normalize_rows",
    "partition_agreement",
    "selected_set_overlap",
    "target_dependence_permutation_test",
    "target_g_vendi",
    "unbiased_hsic",
]
