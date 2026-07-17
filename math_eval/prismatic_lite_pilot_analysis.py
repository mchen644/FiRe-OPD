"""Gradient aggregation, cluster diagnostics, and frozen pilot decisions."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _matrix(value: object, description: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{description} must be a nonempty rank-2 matrix")
    if not bool(np.isfinite(matrix).all()):
        raise ValueError(f"{description} contains non-finite values")
    return matrix


def _unit_rows(value: object, description: str) -> np.ndarray:
    matrix = _matrix(value, description)
    norms = np.linalg.norm(matrix, axis=1)
    if not bool(np.isfinite(norms).all()) or bool((norms <= 0).any()):
        raise ValueError(f"{description} contains zero or invalid row norms")
    return matrix / norms[:, None]


def question_gradient(solution_vectors: object) -> np.ndarray:
    vectors = _unit_rows(solution_vectors, "solution vectors")
    if vectors.shape[0] < 2:
        raise ValueError("a qualified question requires at least two solution vectors")
    mean = vectors.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("mean question gradient has an invalid norm")
    return np.asarray(mean / norm, dtype=np.float32)


def smallest_cluster_ids(
    labels: object, *, count: int, cluster_count: int | None = None
) -> tuple[int, ...]:
    values = np.asarray(labels)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("labels must be a nonempty rank-1 array")
    if not np.issubdtype(values.dtype, np.integer) or bool((values < 0).any()):
        raise ValueError("labels must be nonnegative integers")
    inferred = int(values.max()) + 1
    if cluster_count is None:
        cluster_count = inferred
    if (
        isinstance(cluster_count, bool)
        or not isinstance(cluster_count, int)
        or cluster_count < inferred
    ):
        raise ValueError("cluster_count must include every label")
    if isinstance(count, bool) or not isinstance(count, int) or not 0 < count <= cluster_count:
        raise ValueError("count must be in [1, cluster_count]")
    sizes = np.bincount(values.astype(np.int64), minlength=cluster_count)
    ranked = sorted(range(cluster_count), key=lambda cluster_id: (sizes[cluster_id], cluster_id))
    return tuple(ranked[:count])


def assign_to_centroids(vectors: object, centroids: object) -> np.ndarray:
    normalized_vectors = _unit_rows(vectors, "vectors")
    normalized_centroids = _unit_rows(centroids, "centroids")
    if normalized_vectors.shape[1] != normalized_centroids.shape[1]:
        raise ValueError("vectors and centroids have different dimensions")
    similarities = normalized_vectors @ normalized_centroids.T
    return np.asarray(np.argmax(similarities, axis=1), dtype=np.int64)


def _labels(value: object, size: int, description: str) -> np.ndarray:
    labels = np.asarray(value)
    if labels.shape != (size,) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"{description} must be an integer vector of length {size}")
    if bool((labels < 0).any()):
        raise ValueError(f"{description} contains negative labels")
    return labels.astype(np.int64, copy=False)


def paired_calibration_metrics(
    original_vectors: object,
    regenerated_vectors: object,
    original_labels: object,
    regenerated_labels: object,
    *,
    sparse_cluster_ids: Sequence[int],
) -> dict[str, object]:
    original = _unit_rows(original_vectors, "original vectors")
    regenerated = _unit_rows(regenerated_vectors, "regenerated vectors")
    if original.shape != regenerated.shape:
        raise ValueError("paired gradient matrices must have identical shape")
    size = original.shape[0]
    old_labels = _labels(original_labels, size, "original labels")
    new_labels = _labels(regenerated_labels, size, "regenerated labels")
    sparse_ids = set(int(value) for value in sparse_cluster_ids)
    if not sparse_ids or any(value < 0 for value in sparse_ids):
        raise ValueError("sparse_cluster_ids must be nonempty and nonnegative")

    paired_cosines = np.sum(original * regenerated, axis=1)
    old_sparse = np.isin(old_labels, tuple(sparse_ids))
    new_sparse = np.isin(new_labels, tuple(sparse_ids))
    transitions = {
        "dense_to_dense": int((~old_sparse & ~new_sparse).sum()),
        "dense_to_sparse": int((~old_sparse & new_sparse).sum()),
        "sparse_to_dense": int((old_sparse & ~new_sparse).sum()),
        "sparse_to_sparse": int((old_sparse & new_sparse).sum()),
    }
    return {
        "count": size,
        "paired_cosine_mean": float(paired_cosines.mean()),
        "paired_cosine_median": float(np.median(paired_cosines)),
        "exact_cluster_agreement": float(np.mean(old_labels == new_labels)),
        "sparse_dense_agreement": float(np.mean(old_sparse == new_sparse)),
        "transition_counts": transitions,
    }


def pairing_group_indices(
    metadata: Sequence[Mapping[str, object]],
) -> list[tuple[int, ...]]:
    """Partition rows for the frozen exact-stratum/topic-fallback null.

    If any exact difficulty stratum in a topic has fewer than two rows, the
    entire topic is collapsed into one topic-only stratum.  This yields a true
    partition rather than overlapping permutation pools.
    """
    topics: dict[str, dict[object, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(metadata):
        topic = row.get("topic")
        difficulty = row.get("difficulty")
        if not isinstance(topic, str) or not topic:
            raise ValueError(f"metadata row {index} has invalid topic")
        if isinstance(difficulty, bool) or not isinstance(difficulty, (int, float)):
            raise ValueError(f"metadata row {index} has invalid difficulty")
        topics[topic][difficulty].append(index)

    groups: list[tuple[int, ...]] = []
    for topic in sorted(topics):
        difficulty_groups = topics[topic]
        if any(len(indices) < 2 for indices in difficulty_groups.values()):
            groups.append(
                tuple(
                    sorted(
                        index
                        for indices in difficulty_groups.values()
                        for index in indices
                    )
                )
            )
        else:
            groups.extend(
                tuple(difficulty_groups[difficulty])
                for difficulty in sorted(difficulty_groups, key=lambda value: float(value))
            )
    flattened = sorted(index for group in groups for index in group)
    if flattened != list(range(len(metadata))):
        raise RuntimeError("stratified permutation groups do not partition rows")
    return groups


def _pairing_groups(metadata: Sequence[Mapping[str, object]]) -> list[np.ndarray]:
    return [np.asarray(group, dtype=np.int64) for group in pairing_group_indices(metadata)]


def stratified_pairing_null(
    original_vectors: object,
    regenerated_vectors: object,
    metadata: Sequence[Mapping[str, object]],
    original_sparse: object,
    regenerated_sparse: object,
    *,
    draws: int,
    seed: int,
) -> dict[str, object]:
    original = _unit_rows(original_vectors, "original vectors")
    regenerated = _unit_rows(regenerated_vectors, "regenerated vectors")
    if original.shape != regenerated.shape:
        raise ValueError("paired gradient matrices must have identical shape")
    size = original.shape[0]
    if len(metadata) != size:
        raise ValueError("metadata length does not match gradients")
    old_sparse = np.asarray(original_sparse, dtype=bool)
    new_sparse = np.asarray(regenerated_sparse, dtype=bool)
    if old_sparse.shape != (size,) or new_sparse.shape != (size,):
        raise ValueError("sparse masks must match paired row count")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws <= 0:
        raise ValueError("draws must be a positive integer")

    groups = _pairing_groups(metadata)
    children = np.random.SeedSequence(seed).spawn(draws)
    cosine_medians: list[float] = []
    agreements: list[float] = []
    for child in children:
        generator = np.random.Generator(np.random.PCG64(child))
        permutation = np.arange(size, dtype=np.int64)
        for group in groups:
            permutation[group] = generator.permutation(group)
        paired_cosines = np.sum(original * regenerated[permutation], axis=1)
        cosine_medians.append(float(np.median(paired_cosines)))
        agreements.append(float(np.mean(old_sparse == new_sparse[permutation])))
    return {
        "draws": draws,
        "seed": seed,
        "paired_cosine_medians": cosine_medians,
        "sparse_dense_agreements": agreements,
        "paired_cosine_median_p90": float(np.quantile(cosine_medians, 0.90)),
        "sparse_dense_agreement_median": float(np.median(agreements)),
    }


def calibration_decision(metrics: Mapping[str, object]) -> dict[str, object]:
    failed: list[str] = []
    qualified = metrics.get("qualified_count")
    if isinstance(qualified, bool) or not isinstance(qualified, int) or qualified < 192:
        failed.append("qualified_count_below_192")
    if metrics.get("finite") is not True:
        failed.append("nonfinite_metric_or_gradient")
    seeds = metrics.get("seeds")
    if not isinstance(seeds, Mapping):
        failed.append("missing_seed_metrics")
    else:
        for seed in ("42", "43"):
            values = seeds.get(seed)
            if not isinstance(values, Mapping):
                failed.append(f"seed_{seed}_metrics_missing")
                continue
            try:
                agreement = float(values["sparse_dense_agreement"])
                null_agreement = float(values["null_agreement_median"])
                paired = float(values["paired_cosine_median"])
                null_p90 = float(values["null_paired_cosine_median_p90"])
            except (KeyError, TypeError, ValueError):
                failed.append(f"seed_{seed}_metric_invalid")
                continue
            if not all(math.isfinite(item) for item in (agreement, null_agreement, paired, null_p90)):
                failed.append(f"seed_{seed}_metric_nonfinite")
                continue
            if agreement < 0.65:
                failed.append(f"seed_{seed}_sparse_dense_agreement_below_0.65")
            if agreement - null_agreement < 0.15:
                failed.append(f"seed_{seed}_agreement_margin_below_0.15")
            if paired <= null_p90:
                failed.append(f"seed_{seed}_paired_cosine_not_above_null_p90")
    return {
        "decision": "promising" if not failed else "no_go",
        "failed_conditions": failed,
    }


def candidate_selection(
    *,
    labels_seed42: object,
    labels_seed43: object,
    sparse_seed42: Sequence[int],
    sparse_seed43: Sequence[int],
) -> dict[str, object]:
    primary = np.asarray(labels_seed42)
    sensitivity = np.asarray(labels_seed43)
    if primary.ndim != 1 or sensitivity.shape != primary.shape:
        raise ValueError("candidate label vectors must have identical rank-1 shape")
    if not np.issubdtype(primary.dtype, np.integer) or not np.issubdtype(
        sensitivity.dtype, np.integer
    ):
        raise ValueError("candidate labels must be integer vectors")
    accepted = np.isin(primary, tuple(int(value) for value in sparse_seed42))
    secondary = np.isin(sensitivity, tuple(int(value) for value in sparse_seed43))
    return {
        "accepted_mask": accepted,
        "accepted_indices": np.flatnonzero(accepted).astype(int).tolist(),
        "seed_sparse_dense_agreement": float(np.mean(accepted == secondary)),
    }


def gradient_vendi(vectors: object) -> float:
    normalized = _unit_rows(vectors, "G-Vendi vectors")
    # The nonzero spectra of X X^T and X^T X are identical.  Use the
    # feature-space covariance so the frozen 57K-row pool never creates a
    # quadratic row-space matrix.
    covariance = normalized.T @ normalized
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = float(eigenvalues.sum())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("G-Vendi spectrum has invalid trace")
    probabilities = eigenvalues[eigenvalues > 1e-15] / total
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return float(math.exp(entropy))


def selection_null(
    original_vectors: object,
    quality_vectors: object,
    *,
    accepted_indices: Sequence[int],
    draws: int,
    seed: int,
) -> dict[str, object]:
    original = _unit_rows(original_vectors, "original vectors")
    quality = _unit_rows(quality_vectors, "quality vectors")
    if original.shape[1] != quality.shape[1]:
        raise ValueError("original and quality vectors have different dimensions")
    accepted = np.asarray(accepted_indices, dtype=np.int64)
    if accepted.ndim != 1 or accepted.size == 0:
        raise ValueError("accepted_indices must be nonempty")
    if len(set(accepted.tolist())) != accepted.size or bool((accepted < 0).any()) or bool(
        (accepted >= quality.shape[0]).any()
    ):
        raise ValueError("accepted_indices are invalid")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws <= 0:
        raise ValueError("draws must be a positive integer")

    observed = gradient_vendi(np.concatenate((original, quality[accepted]), axis=0))
    generator = np.random.Generator(np.random.PCG64(seed))
    null_values: list[float] = []
    for _ in range(draws):
        indices = generator.choice(quality.shape[0], size=accepted.size, replace=False)
        null_values.append(
            gradient_vendi(np.concatenate((original, quality[indices]), axis=0))
        )
    percentile = (1 + sum(value <= observed for value in null_values)) / (draws + 1)
    return {
        "draws": draws,
        "seed": seed,
        "observed_g_vendi": observed,
        "null_g_vendi": null_values,
        "null_median": float(np.median(null_values)),
        "percentile": float(percentile),
    }


def selection_null_torch(
    original_vectors: object,
    quality_vectors: object,
    *,
    accepted_indices: Sequence[int],
    draws: int,
    seed: int,
    device: str,
    batch_size: int = 4,
) -> dict[str, object]:
    """Compute the exact feature-space G-Vendi null in bounded GPU batches."""
    import torch

    if device not in {"cpu", "cuda:0"}:
        raise ValueError("selection null device must be cpu or process-local cuda:0")
    if device == "cuda:0" and torch.cuda.device_count() != 1:
        raise RuntimeError("CUDA selection null requires exactly one visible GPU")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws <= 0:
        raise ValueError("draws must be a positive integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    def normalized(value: object, description: str) -> "torch.Tensor":
        tensor = torch.as_tensor(value, dtype=torch.float32, device="cpu")
        if tensor.ndim != 2 or tensor.shape[0] == 0 or tensor.shape[1] == 0:
            raise ValueError(f"{description} must be a nonempty rank-2 matrix")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{description} contains non-finite values")
        norms = torch.linalg.vector_norm(tensor, dim=1)
        if bool((norms <= 0).any()):
            raise ValueError(f"{description} contains zero row norms")
        return (tensor / norms[:, None]).to(device)

    original = normalized(original_vectors, "original vectors")
    quality = normalized(quality_vectors, "quality vectors")
    if original.shape[1] != quality.shape[1]:
        raise ValueError("original and quality vectors have different dimensions")
    accepted = np.asarray(accepted_indices, dtype=np.int64)
    if accepted.ndim != 1 or accepted.size == 0:
        raise ValueError("accepted_indices must be nonempty")
    if len(set(accepted.tolist())) != accepted.size or bool((accepted < 0).any()) or bool(
        (accepted >= quality.shape[0]).any()
    ):
        raise ValueError("accepted_indices are invalid")

    base_covariance = original.T @ original

    def vendi_from_covariances(covariances: "torch.Tensor") -> "torch.Tensor":
        eigenvalues = torch.linalg.eigvalsh(covariances).clamp_min_(0)
        totals = eigenvalues.sum(dim=-1, keepdim=True)
        if not bool(torch.isfinite(totals).all()) or bool((totals <= 0).any()):
            raise ValueError("G-Vendi spectrum has invalid trace")
        probabilities = eigenvalues / totals
        terms = torch.where(
            probabilities > 1e-15,
            probabilities * torch.log(probabilities),
            torch.zeros_like(probabilities),
        )
        return torch.exp(-terms.sum(dim=-1))

    accepted_tensor = torch.as_tensor(accepted, dtype=torch.int64, device=device)
    observed_covariance = base_covariance + quality.index_select(0, accepted_tensor).T @ quality.index_select(0, accepted_tensor)
    observed = float(vendi_from_covariances(observed_covariance[None])[0].cpu())

    generator = np.random.Generator(np.random.PCG64(seed))
    random_indices = np.stack(
        [
            generator.choice(quality.shape[0], size=accepted.size, replace=False)
            for _ in range(draws)
        ],
        axis=0,
    )
    null_values: list[float] = []
    for start in range(0, draws, batch_size):
        indices = torch.as_tensor(
            random_indices[start : start + batch_size],
            dtype=torch.int64,
            device=device,
        )
        selected = quality[indices]
        updates = torch.einsum("bnd,bne->bde", selected, selected)
        values = vendi_from_covariances(base_covariance[None] + updates)
        null_values.extend(float(value) for value in values.cpu().tolist())
    percentile = (1 + sum(value <= observed for value in null_values)) / (draws + 1)
    return {
        "draws": draws,
        "seed": seed,
        "observed_g_vendi": observed,
        "null_g_vendi": null_values,
        "null_median": float(np.median(null_values)),
        "percentile": float(percentile),
    }


def final_pilot_decision(metrics: Mapping[str, object]) -> dict[str, object]:
    failed: list[str] = []
    numeric_thresholds = (
        ("quality_count", 1000),
        ("accepted_count", 400),
        ("seed_sparse_dense_agreement", 0.65),
        ("g_vendi_percentile", 0.90),
    )
    for key, threshold in numeric_thresholds:
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            failed.append(f"{key}_invalid")
        elif not math.isfinite(float(value)) or float(value) < threshold:
            failed.append(f"{key}_below_{threshold}")
    if metrics.get("unique_prompts") is not True:
        failed.append("accepted_prompts_not_unique")
    contamination = metrics.get("benchmark_contamination_count")
    if isinstance(contamination, bool) or not isinstance(contamination, int):
        failed.append("benchmark_contamination_count_invalid")
    elif contamination != 0:
        failed.append("benchmark_contamination_detected")
    return {
        "decision": "promising" if not failed else "no_go",
        "failed_conditions": failed,
    }
