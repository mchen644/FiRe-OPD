"""Pure helpers for preparing and selecting a gradient-diverse DeepMath pool."""

import hashlib
import random
from bisect import bisect_left, bisect_right
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch


_OPD_SUFFIX = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}."
)


def strip_opd_instruction(prompt: str) -> str:
    """Remove the exact FiRe-OPD instruction when it is the prompt suffix."""
    if prompt.endswith(_OPD_SUFFIX):
        return prompt[: -len(_OPD_SUFFIX)]
    return prompt


def normalize_math_answer(answer: object) -> str:
    """Normalize scalar math answers for exact provenance comparisons."""
    normalized = str(answer).strip().replace("$", "").replace(" ", "")
    if normalized.replace(",", "").isdigit():
        normalized = normalized.replace(",", "")
    return normalized


def _is_missing_answer(answer: object) -> bool:
    return answer is None or (isinstance(answer, str) and not answer.strip())


def _extract_user_prompt(row: Mapping) -> str:
    prompt = row.get("prompt")
    if (
        not isinstance(prompt, Sequence)
        or isinstance(prompt, (str, bytes))
        or len(prompt) != 1
        or not isinstance(prompt[0], Mapping)
        or prompt[0].get("role") != "user"
        or not isinstance(prompt[0].get("content"), str)
    ):
        raise ValueError("malformed chat prompt: expected one user message")
    return prompt[0]["content"]


def join_filtered_rows(
    filtered_rows: Sequence[Mapping],
    original_rows: Sequence[Mapping],
    expected_count: int | None,
) -> list[dict]:
    """Join filtered training rows to original DeepMath rows by exact question."""
    if expected_count is not None and len(filtered_rows) != expected_count:
        raise ValueError(
            f"expected {expected_count} filtered rows, got {len(filtered_rows)}"
        )

    original_indices_by_key: dict[tuple[object, str], list[int]] = {}
    original_questions: set[object] = set()
    questions_with_complete_answers: set[object] = set()
    for original_index, original in enumerate(original_rows):
        question = original.get("question")
        original_questions.add(question)
        if "final_answer" not in original or _is_missing_answer(
            original.get("final_answer")
        ):
            continue
        key = (question, normalize_math_answer(original["final_answer"]))
        questions_with_complete_answers.add(question)
        original_indices_by_key.setdefault(key, []).append(original_index)

    questions: list[str] = []
    filtered_keys: list[tuple[object, str]] = []
    for source_index, filtered in enumerate(filtered_rows):
        question = strip_opd_instruction(_extract_user_prompt(filtered))
        reward_model = filtered.get("reward_model")
        if (
            not isinstance(reward_model, Mapping)
            or "ground_truth" not in reward_model
            or _is_missing_answer(reward_model.get("ground_truth"))
        ):
            raise ValueError(
                f"missing or empty ground_truth for filtered row {source_index}"
            )
        key = (question, normalize_math_answer(reward_model["ground_truth"]))
        if key not in original_indices_by_key:
            if question not in original_questions:
                raise ValueError(
                    f"no original question match for filtered row {source_index}"
                )
            if question not in questions_with_complete_answers:
                raise ValueError(
                    "missing or empty final_answer for original question matched by "
                    f"filtered row {source_index}"
                )
            raise ValueError(f"answer mismatch for filtered row {source_index}")
        questions.append(question)
        filtered_keys.append(key)

    forward_indices: list[int] = []
    previous_index = -1
    for source_index, key in enumerate(filtered_keys):
        candidates = original_indices_by_key[key]
        position = bisect_right(candidates, previous_index)
        if position == len(candidates):
            raise ValueError(f"no ordered original match for filtered row {source_index}")
        previous_index = candidates[position]
        forward_indices.append(previous_index)

    backward_indices = [0] * len(filtered_keys)
    next_index = len(original_rows)
    for source_index in range(len(filtered_keys) - 1, -1, -1):
        candidates = original_indices_by_key[filtered_keys[source_index]]
        position = bisect_left(candidates, next_index) - 1
        if position < 0:
            raise ValueError(f"no ordered original match for filtered row {source_index}")
        next_index = candidates[position]
        backward_indices[source_index] = next_index

    for source_index, (forward_index, backward_index) in enumerate(
        zip(forward_indices, backward_indices)
    ):
        if forward_index != backward_index:
            raise ValueError(
                "ambiguous ordered original match for filtered row "
                f"{source_index}: earliest {forward_index}, latest {backward_index}"
            )

    prepared: list[dict] = []
    for source_index, (question, original_index) in enumerate(
        zip(questions, forward_indices)
    ):
        original = original_rows[original_index]
        completion = original.get("r1_solution_1")
        if not isinstance(completion, str) or not completion.strip():
            raise ValueError(f"empty r1_solution_1 for filtered row {source_index}")

        prepared.append(
            {
                "id": f"deepmath-level6-{source_index:06d}",
                "prompt": question,
                "completion": completion,
                "source_row_index": source_index,
                "original_dataset_index": original_index,
                "topic": original.get("topic"),
                "difficulty": original.get("difficulty"),
            }
        )

    return prepared


def official_shard_bounds(
    total: int, num_shards: int, shard_index: int
) -> tuple[int, int]:
    """Return bounds using the official gradient collector's partition rule."""
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")

    partition_size = int(total / num_shards) + 1
    start = shard_index * partition_size
    return start, min(start + partition_size, total)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_gradient_matrix(
    ids: Sequence[str],
    gradients: torch.Tensor,
    expected_ids: Sequence[str],
) -> None:
    """Validate exact gradient coverage and usable projected-gradient rows."""
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate gradient IDs")
    if list(ids) != list(expected_ids):
        raise ValueError("gradient IDs do not exactly match expected IDs")
    if gradients.ndim != 2 or gradients.shape[0] != len(ids):
        raise ValueError("gradients must be two-dimensional with one row per ID")
    if not bool(torch.isfinite(gradients).all()):
        raise ValueError("gradient matrix contains non-finite values")
    if bool((torch.linalg.vector_norm(gradients.float(), dim=1) == 0).any()):
        raise ValueError("gradient matrix contains zero-norm rows")


def balanced_round_robin(
    labels: Sequence[int], target_size: int, seed: int
) -> list[int]:
    """Select shuffled members in seeded, cluster-balanced rounds."""
    if target_size < 0:
        raise ValueError("target_size must be non-negative")
    if target_size > len(labels):
        raise ValueError(f"cannot select {target_size} rows from {len(labels)}")
    if any(label < 0 for label in labels):
        raise ValueError("cluster labels must be non-negative")

    members: dict[int, list[int]] = {}
    for row_index, cluster_id in enumerate(labels):
        members.setdefault(cluster_id, []).append(row_index)

    rng = random.Random(seed)
    for cluster_id in sorted(members):
        rng.shuffle(members[cluster_id])

    selected: list[int] = []
    while len(selected) < target_size:
        active_clusters = [
            cluster_id for cluster_id in sorted(members) if members[cluster_id]
        ]
        rng.shuffle(active_clusters)
        for cluster_id in active_clusters:
            selected.append(members[cluster_id].pop())
            if len(selected) == target_size:
                break

    return selected


def _linear_quantile(sorted_values: Sequence[int], quantile: float) -> float:
    position = (len(sorted_values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return float(
        sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
    )


def cluster_size_summary(
    labels: Sequence[int], requested_clusters: int
) -> dict[str, float | int]:
    """Summarize cluster sizes, retaining zeros for empty requested clusters."""
    if requested_clusters <= 0:
        raise ValueError("requested_clusters must be positive")
    if any(label < 0 or label >= requested_clusters for label in labels):
        raise ValueError("cluster labels must be in [0, requested_clusters)")

    sizes = [0] * requested_clusters
    for label in labels:
        sizes[label] += 1
    sorted_sizes = sorted(sizes)
    maximum = sorted_sizes[-1]

    return {
        "requested_clusters": requested_clusters,
        "non_empty_clusters": sum(size > 0 for size in sizes),
        "min_cluster_size": sorted_sizes[0],
        "median_cluster_size": _linear_quantile(sorted_sizes, 0.5),
        "mean_cluster_size": float(sum(sizes) / requested_clusters),
        "p90_cluster_size": _linear_quantile(sorted_sizes, 0.9),
        "p99_cluster_size": _linear_quantile(sorted_sizes, 0.99),
        "max_cluster_size": maximum,
        "singleton_cluster_ratio": sum(size == 1 for size in sizes)
        / requested_clusters,
        "largest_cluster_fraction": (
            float(maximum / len(labels)) if len(labels) > 0 else 0.0
        ),
    }
