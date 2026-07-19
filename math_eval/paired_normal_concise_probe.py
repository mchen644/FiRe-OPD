"""Paired normal/concise compression-sensitivity probe utilities."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence

import numpy as np

from verl.trainer.ppo.tale_budget import build_concise_teacher_messages

_QUADRANTS = {
    (True, True): "compression_safe",
    (True, False): "compression_sensitive",
    (False, True): "concise_rescued",
    (False, False): "both_wrong",
}


def _validate_source_row(row: Mapping, *, source_row_index: int) -> tuple[int, list[dict], str]:
    prompt = row.get("prompt")
    if not isinstance(prompt, (list, np.ndarray)) or len(prompt) != 1:
        raise ValueError(f"row {source_row_index} prompt must contain exactly one message")
    message = prompt[0]
    if not isinstance(message, Mapping):
        raise ValueError(f"row {source_row_index} prompt message must be a mapping")
    if message.get("role") != "user" or not isinstance(message.get("content"), str) or not message["content"].strip():
        raise ValueError(f"row {source_row_index} prompt must contain nonempty user content")

    reward_model = row.get("reward_model")
    if not isinstance(reward_model, Mapping):
        raise ValueError(f"row {source_row_index} is missing reward_model")
    ground_truth = reward_model.get("ground_truth")
    if ground_truth is None or not str(ground_truth).strip():
        raise ValueError(f"row {source_row_index} has an empty ground truth")

    extra_info = row.get("extra_info")
    if not isinstance(extra_info, Mapping):
        raise ValueError(f"row {source_row_index} is missing extra_info")
    question_id = extra_info.get("index")
    if isinstance(question_id, bool) or not isinstance(question_id, int):
        raise ValueError(f"row {source_row_index} has an invalid distinct question identity")

    normalized_prompt = [{"role": "user", "content": str(message["content"])}]
    return int(question_id), normalized_prompt, str(ground_truth)


def select_probe_rows(rows: Sequence[Mapping], *, sample_size: int, seed: int) -> list[dict]:
    """Select a deterministic, identity-distinct sample from parquet-style rows."""

    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size <= 0:
        raise ValueError("sample_size must be a positive integer")
    if sample_size > len(rows):
        raise ValueError("sample_size cannot exceed the dataset row count")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    normalized: list[tuple[int, list[dict], str]] = []
    identities: list[int] = []
    for source_row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"row {source_row_index} must be a mapping")
        question_id, prompt, ground_truth = _validate_source_row(
            row, source_row_index=source_row_index
        )
        normalized.append((question_id, prompt, ground_truth))
        identities.append(question_id)
    if len(set(identities)) != len(identities):
        raise ValueError("dataset rows must have distinct question identities")

    rng = np.random.default_rng(seed)
    source_indices = rng.choice(len(rows), size=sample_size, replace=False).tolist()
    selected: list[dict] = []
    for sample_ordinal, raw_index in enumerate(source_indices):
        source_row_index = int(raw_index)
        question_id, prompt, ground_truth = normalized[source_row_index]
        request_seed = seed + source_row_index
        if request_seed >= 2**31:
            raise ValueError("derived request seed exceeds the signed 32-bit range")
        selected.append(
            {
                "sample_ordinal": sample_ordinal,
                "source_row_index": source_row_index,
                "question_id": question_id,
                "request_seed": request_seed,
                "prompt": copy.deepcopy(prompt),
                "ground_truth": ground_truth,
            }
        )
    return selected


def normal_messages(row: Mapping) -> list[dict[str, str]]:
    """Return a defensive copy of one persisted normal prompt."""

    _, prompt, _ = _validate_source_row(
        {"prompt": row.get("prompt"), "reward_model": {"ground_truth": row.get("ground_truth")}, "extra_info": {"index": row.get("question_id")}},
        source_row_index=int(row.get("source_row_index", -1)),
    )
    return copy.deepcopy(prompt)


def concise_messages(row: Mapping) -> list[dict[str, str]]:
    """Build exactly the concise prompt used by adaptive training."""

    messages = normal_messages(row)
    return build_concise_teacher_messages(question=messages[0]["content"])


def concise_cap(normal_length: int) -> int:
    """Return the frozen half-normal per-request concise cap."""

    if isinstance(normal_length, bool) or not isinstance(normal_length, int) or normal_length <= 0:
        raise ValueError("normal_length must be a positive integer")
    return max(1, normal_length // 2)


def quadrant(normal_correct: bool, concise_correct: bool) -> str:
    """Classify a paired correctness event into one of four quadrants."""

    if not isinstance(normal_correct, bool) or not isinstance(concise_correct, bool):
        raise TypeError("quadrant correctness flags must be boolean")
    return _QUADRANTS[(normal_correct, concise_correct)]


def validate_relaxed_prefix(capped_ids: Sequence[int], relaxed_ids: Sequence[int]) -> None:
    """Require a same-seed relaxed response to preserve the capped token prefix."""

    capped = list(capped_ids)
    relaxed = list(relaxed_ids)
    if not capped or not relaxed:
        raise ValueError("capped and relaxed token sequences must be nonempty")
    if len(relaxed) < len(capped) or relaxed[: len(capped)] != capped:
        raise ValueError("relaxed concise response does not preserve the capped token prefix")


def classify_compression_sensitive(record: Mapping) -> str | None:
    """Classify the mechanical failure mode of one paired record."""

    if record.get("quadrant") != "compression_sensitive":
        return None
    concise = record.get("concise")
    if not isinstance(concise, Mapping) or concise.get("correct") is not False:
        raise ValueError("compression-sensitive records require an incorrect concise response")
    if concise.get("cap_hit") is False:
        if record.get("relaxed_concise") is not None:
            raise ValueError("non-cap-hit concise responses must not have a relaxed result")
        return "prompt_or_sampling_failure"
    if concise.get("cap_hit") is not True:
        raise ValueError("concise cap_hit must be boolean")

    relaxed = record.get("relaxed_concise")
    if not isinstance(relaxed, Mapping):
        raise ValueError("cap-hit compression-sensitive records require a relaxed result")
    validate_relaxed_prefix(concise.get("token_ids", []), relaxed.get("token_ids", []))
    if not isinstance(relaxed.get("correct"), bool):
        raise ValueError("relaxed concise correctness must be boolean")
    if relaxed["correct"]:
        return "budget_limited_recovered"
    return "budget_limited_unrecovered"
