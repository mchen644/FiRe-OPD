"""Paired normal/concise compression-sensitivity probe utilities."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from verl.trainer.ppo.tale_budget import build_concise_teacher_messages
from verl.utils.reward_score.math_reward import (
    compute_score,
    last_boxed_only_string,
    remove_boxed,
)

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
    """Require a relaxed response to preserve the capped token prefix."""

    capped = list(capped_ids)
    relaxed = list(relaxed_ids)
    if not capped or not relaxed:
        raise ValueError("capped and relaxed token sequences must be nonempty")
    if len(relaxed) < len(capped) or relaxed[: len(capped)] != capped:
        raise ValueError("relaxed concise response does not preserve the capped token prefix")


def _validated_token_ids(value, *, name: str, allow_empty: bool = False) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be an integer sequence")
    token_ids = list(value)
    if (not allow_empty and not token_ids) or not all(
        isinstance(token_id, int) and not isinstance(token_id, bool) for token_id in token_ids
    ):
        qualifier = "possibly empty " if allow_empty else "nonempty "
        raise ValueError(f"{name} must be a {qualifier}integer sequence")
    return token_ids


def forced_prefix_continuation_prompt(
    request_output, capped_ids: Sequence[int]
) -> dict[str, list[int]]:
    """Build a vLLM token prompt ending in the exact observed capped response."""

    prompt_ids = _validated_token_ids(
        getattr(request_output, "prompt_token_ids", None), name="prompt token IDs"
    )
    capped = _validated_token_ids(capped_ids, name="capped token IDs")
    return {"prompt_token_ids": prompt_ids + capped}


def sha256_file(path: str | Path) -> str:
    """Return the SHA256 digest of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} must be an object")
            rows.append(value)
    return rows


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: Mapping) -> None:
    _atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping]) -> None:
    content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    _atomic_write_text(path, content)


def prepare_sample(
    *,
    dataset_path: str | Path,
    sample_file: str | Path,
    manifest_file: str | Path,
    sample_size: int,
    seed: int,
    source_commit: str,
    expected_dataset_sha256: str,
) -> dict:
    """Verify the dataset and persist one immutable paired-probe sample."""

    dataset = Path(dataset_path)
    sample_path = Path(sample_file)
    manifest_path = Path(manifest_file)
    if not dataset.is_file():
        raise FileNotFoundError(f"dataset does not exist: {dataset}")
    for destination in (sample_path, manifest_path):
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {destination}")
    if not isinstance(source_commit, str) or not source_commit.strip():
        raise ValueError("source_commit must be nonempty")
    actual_dataset_sha256 = sha256_file(dataset)
    if actual_dataset_sha256 != expected_dataset_sha256:
        raise ValueError(
            "dataset SHA256 mismatch: "
            f"expected {expected_dataset_sha256}, got {actual_dataset_sha256}"
        )

    rows = pq.read_table(dataset).to_pylist()
    selected = select_probe_rows(rows, sample_size=sample_size, seed=seed)
    _atomic_write_jsonl(sample_path, selected)
    manifest = {
        "artifact_type": "paired_normal_concise_compression_sensitivity_probe",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset.resolve()),
        "dataset_sha256": actual_dataset_sha256,
        "sample_file": str(sample_path.resolve()),
        "sample_sha256": sha256_file(sample_path),
        "sample_size": sample_size,
        "seed": seed,
        "source_commit": source_commit,
        "protocol": {
            "normal_max_tokens": 16384,
            "concise_cap_ratio": 0.5,
            "relaxed_counterfactual_mode": "forced_prefix_continuation",
            "temperature": 1.0,
            "top_p": 1.0,
            "enable_thinking": False,
            "n": 1,
            "max_model_len": 40960,
            "tensor_parallel_size": 4,
        },
        "artifacts": {},
    }
    _atomic_write_json(manifest_path, manifest)
    return manifest


def _boxed_answer(text: str) -> str | None:
    try:
        boxed = last_boxed_only_string(text)
        if boxed is None:
            return None
        return remove_boxed(boxed)
    except Exception:
        return None


def normalize_request_output(
    request_output,
    *,
    ground_truth: str,
    max_tokens: int,
    rendered_prompt: str,
) -> dict:
    """Normalize one vLLM request result without re-tokenizing its response."""

    outputs = getattr(request_output, "outputs", None)
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise ValueError("each request must return exactly one generation")
    output = outputs[0]
    text = str(getattr(output, "text", ""))
    token_ids = [int(token_id) for token_id in getattr(output, "token_ids", [])]
    if not token_ids:
        raise ValueError("generation must contain at least one token")
    if len(token_ids) > max_tokens:
        raise ValueError("generated response exceeds its max_tokens cap")
    finish_reason = getattr(output, "finish_reason", None)
    if finish_reason is not None:
        finish_reason = str(finish_reason)
    stop_reason = getattr(output, "stop_reason", None)
    if stop_reason is not None and not isinstance(stop_reason, (str, int, float, bool)):
        stop_reason = str(stop_reason)
    boxed_answer = _boxed_answer(text)
    return {
        "text": text,
        "token_ids": token_ids,
        "length": len(token_ids),
        "finish_reason": finish_reason,
        "stop_reason": stop_reason,
        "boxed_answer": boxed_answer,
        "parseable": boxed_answer is not None,
        "correct": bool(compute_score(text, str(ground_truth)) > 0.5),
        "max_tokens": max_tokens,
        "cap_hit": len(token_ids) == max_tokens,
        "rendered_prompt_sha256": hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest(),
    }


def normalize_forced_prefix_continuation(
    request_output,
    *,
    capped_response: Mapping,
    tokenizer,
    ground_truth: str,
    total_max_tokens: int,
    rendered_prompt: str,
    continuation_seed: int,
    continuation_prompt: Mapping,
) -> dict:
    """Normalize a generated suffix combined with an exact capped-response prefix."""

    if isinstance(total_max_tokens, bool) or not isinstance(total_max_tokens, int) or total_max_tokens <= 0:
        raise ValueError("total_max_tokens must be a positive integer")
    if isinstance(continuation_seed, bool) or not isinstance(continuation_seed, int) or continuation_seed < 0:
        raise ValueError("continuation_seed must be a nonnegative integer")
    capped_ids = _validated_token_ids(
        capped_response.get("token_ids"), name="capped token IDs"
    )
    capped_length = capped_response.get("length")
    if (
        isinstance(capped_length, bool)
        or not isinstance(capped_length, int)
        or capped_length != len(capped_ids)
    ):
        raise ValueError("capped response length does not match its token IDs")
    if capped_length > total_max_tokens:
        raise ValueError("capped response exceeds the relaxed total budget")
    if not isinstance(continuation_prompt, Mapping):
        raise ValueError("continuation prompt must be a token-prompt mapping")
    continuation_prompt_ids = _validated_token_ids(
        continuation_prompt.get("prompt_token_ids"), name="continuation prompt token IDs"
    )
    if (
        len(continuation_prompt_ids) <= capped_length
        or continuation_prompt_ids[-capped_length:] != capped_ids
    ):
        raise ValueError("continuation prompt does not preserve the exact capped suffix")

    continuation_max_tokens = total_max_tokens - capped_length
    if continuation_max_tokens == 0:
        if request_output is not None:
            raise ValueError("zero remaining budget must not have a continuation output")
        continuation_ids: list[int] = []
        combined_text = str(capped_response.get("text", ""))
        finish_reason = capped_response.get("finish_reason")
        stop_reason = capped_response.get("stop_reason")
    else:
        actual_prompt_ids = _validated_token_ids(
            getattr(request_output, "prompt_token_ids", None),
            name="generated continuation prompt token IDs",
        )
        if actual_prompt_ids != continuation_prompt_ids:
            raise ValueError("generated continuation prompt token IDs changed")
        outputs = getattr(request_output, "outputs", None)
        if not isinstance(outputs, list) or len(outputs) != 1:
            raise ValueError("each continuation request must return exactly one generation")
        output = outputs[0]
        continuation_ids = _validated_token_ids(
            getattr(output, "token_ids", None), name="continuation token IDs"
        )
        if len(continuation_ids) > continuation_max_tokens:
            raise ValueError("continuation exceeds its remaining token budget")
        combined_ids_for_decode = capped_ids + continuation_ids
        combined_text = str(
            tokenizer.decode(
                combined_ids_for_decode,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )
        finish_reason = getattr(output, "finish_reason", None)
        stop_reason = getattr(output, "stop_reason", None)

    combined_ids = capped_ids + continuation_ids
    if finish_reason is not None:
        finish_reason = str(finish_reason)
    if stop_reason is not None and not isinstance(stop_reason, (str, int, float, bool)):
        stop_reason = str(stop_reason)
    boxed_answer = _boxed_answer(combined_text)
    return {
        "text": combined_text,
        "token_ids": combined_ids,
        "length": len(combined_ids),
        "finish_reason": finish_reason,
        "stop_reason": stop_reason,
        "boxed_answer": boxed_answer,
        "parseable": boxed_answer is not None,
        "correct": bool(compute_score(combined_text, str(ground_truth)) > 0.5),
        "max_tokens": total_max_tokens,
        "cap_hit": len(combined_ids) == total_max_tokens,
        "rendered_prompt_sha256": hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest(),
        "counterfactual_mode": "forced_prefix_continuation",
        "forced_prefix_length": capped_length,
        "continuation_seed": continuation_seed,
        "continuation_max_tokens": continuation_max_tokens,
        "continuation_token_ids": continuation_ids,
        "continuation_length": len(continuation_ids),
        "continuation_prompt_token_ids": continuation_prompt_ids,
        "continuation_prompt_length": len(continuation_prompt_ids),
    }


def ensure_generation_destinations(*, model_path: str | Path, output_dir: str | Path) -> None:
    model = Path(model_path)
    output = Path(output_dir)
    if not model.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {model}")
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")


def _validate_forced_prefix_counterfactual(
    record: Mapping, *, concise: Mapping, relaxed: Mapping
) -> None:
    if relaxed.get("counterfactual_mode") != "forced_prefix_continuation":
        raise ValueError("relaxed concise counterfactual mode is invalid")
    capped_ids = _validated_token_ids(concise.get("token_ids"), name="capped token IDs")
    relaxed_ids = _validated_token_ids(relaxed.get("token_ids"), name="relaxed token IDs")
    validate_relaxed_prefix(capped_ids, relaxed_ids)
    if relaxed.get("forced_prefix_length") != len(capped_ids):
        raise ValueError("relaxed concise forced prefix length is invalid")

    request_seed = record.get("request_seed")
    if (
        isinstance(request_seed, bool)
        or not isinstance(request_seed, int)
        or relaxed.get("continuation_seed") != request_seed
    ):
        raise ValueError("relaxed concise continuation seed is invalid")
    relaxed_max_tokens = relaxed.get("max_tokens")
    if isinstance(relaxed_max_tokens, bool) or not isinstance(relaxed_max_tokens, int):
        raise ValueError("relaxed concise max_tokens must be an integer")
    expected_remaining = relaxed_max_tokens - len(capped_ids)
    if expected_remaining < 0 or relaxed.get("continuation_max_tokens") != expected_remaining:
        raise ValueError("relaxed concise remaining budget is invalid")

    continuation_ids = _validated_token_ids(
        relaxed.get("continuation_token_ids"),
        name="continuation token IDs",
        allow_empty=True,
    )
    if relaxed.get("continuation_length") != len(continuation_ids):
        raise ValueError("relaxed concise continuation length is invalid")
    if len(continuation_ids) > expected_remaining:
        raise ValueError("relaxed concise continuation exceeds its remaining budget")
    if relaxed_ids != capped_ids + continuation_ids:
        raise ValueError("relaxed concise combined token IDs are invalid")

    continuation_prompt_ids = _validated_token_ids(
        relaxed.get("continuation_prompt_token_ids"),
        name="continuation prompt token IDs",
    )
    if (
        len(continuation_prompt_ids) <= len(capped_ids)
        or continuation_prompt_ids[-len(capped_ids):] != capped_ids
    ):
        raise ValueError("relaxed concise continuation prompt capped suffix is invalid")
    if relaxed.get("continuation_prompt_length") != len(continuation_prompt_ids):
        raise ValueError("relaxed concise continuation prompt length is invalid")


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
    _validate_forced_prefix_counterfactual(record, concise=concise, relaxed=relaxed)
    if not isinstance(relaxed.get("correct"), bool):
        raise ValueError("relaxed concise correctness must be boolean")
    if relaxed["correct"]:
        return "budget_limited_recovered"
    return "budget_limited_unrecovered"


def _validate_response_payload(response: Mapping, *, name: str) -> None:
    token_ids = response.get("token_ids")
    length = response.get("length")
    max_tokens = response.get("max_tokens")
    if not isinstance(token_ids, list) or not token_ids or not all(
        isinstance(token_id, int) and not isinstance(token_id, bool) for token_id in token_ids
    ):
        raise ValueError(f"{name} token_ids must be a nonempty integer list")
    if isinstance(length, bool) or not isinstance(length, int) or length != len(token_ids):
        raise ValueError(f"{name} length does not match token_ids")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError(f"{name} max_tokens must be a positive integer")
    if length > max_tokens:
        raise ValueError(f"{name} length exceeds max_tokens")
    if response.get("cap_hit") is not (length == max_tokens):
        raise ValueError(f"{name} cap_hit does not match its length")
    if not isinstance(response.get("correct"), bool):
        raise ValueError(f"{name} correctness must be boolean")
    if not isinstance(response.get("parseable"), bool):
        raise ValueError(f"{name} parseable must be boolean")
    if response.get("parseable") is not (response.get("boxed_answer") is not None):
        raise ValueError(f"{name} parseable flag does not match boxed_answer")


def summarize_records(records: Sequence[Mapping], *, expected_count: int) -> dict:
    """Validate and summarize one model's complete paired probe records."""

    if len(records) != expected_count:
        raise ValueError(f"records must contain exactly {expected_count} rows")
    if expected_count <= 0:
        raise ValueError("expected_count must be positive")
    ordinals = [record.get("sample_ordinal") for record in records]
    if ordinals != list(range(expected_count)):
        raise ValueError("record sample ordinals are incomplete or out of order")
    question_ids = [record.get("question_id") for record in records]
    if len(set(question_ids)) != expected_count:
        raise ValueError("record question identities must be distinct")
    model_labels = {record.get("model_label") for record in records}
    if len(model_labels) != 1 or not isinstance(next(iter(model_labels)), str):
        raise ValueError("records must share one nonempty model label")
    model_label = next(iter(model_labels))
    if not model_label:
        raise ValueError("records must share one nonempty model label")

    route_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    cap_hits_by_quadrant: Counter[str] = Counter()
    normal_lengths: list[int] = []
    concise_lengths: list[int] = []
    ratios: list[float] = []
    normal_correct = 0
    concise_correct = 0
    normal_parse_failures = 0
    concise_parse_failures = 0

    for record in records:
        normal = record.get("normal")
        concise = record.get("concise")
        if not isinstance(normal, Mapping) or not isinstance(concise, Mapping):
            raise ValueError("each record requires normal and concise response objects")
        _validate_response_payload(normal, name="normal")
        _validate_response_payload(concise, name="concise")
        if normal["max_tokens"] != 16384:
            raise ValueError("normal max_tokens must equal 16384")
        expected_cap = concise_cap(normal["length"])
        if concise["max_tokens"] != expected_cap:
            raise ValueError("concise max_tokens does not match the normal-length cap")
        expected_route = quadrant(normal["correct"], concise["correct"])
        if record.get("quadrant") != expected_route:
            raise ValueError("persisted quadrant does not match response correctness")
        relaxed = record.get("relaxed_concise")
        if expected_route != "compression_sensitive" and relaxed is not None:
            raise ValueError("nonsensitive rows must not have a relaxed counterfactual")
        expected_class = classify_compression_sensitive(record)
        if record.get("compression_sensitive_class") != expected_class:
            raise ValueError("persisted compression-sensitive class does not recompute")
        if relaxed is not None:
            if not isinstance(relaxed, Mapping):
                raise ValueError("relaxed concise response must be an object or null")
            _validate_response_payload(relaxed, name="relaxed concise")
            if relaxed["max_tokens"] != normal["length"]:
                raise ValueError("relaxed concise max_tokens must equal normal length")

        route_counts[expected_route] += 1
        if expected_class is not None:
            class_counts[expected_class] += 1
        if concise["cap_hit"]:
            cap_hits_by_quadrant[expected_route] += 1
        normal_lengths.append(normal["length"])
        concise_lengths.append(concise["length"])
        ratios.append(concise["length"] / normal["length"])
        normal_correct += int(normal["correct"])
        concise_correct += int(concise["correct"])
        normal_parse_failures += int(not normal["parseable"])
        concise_parse_failures += int(not concise["parseable"])

    route_names = [
        "compression_safe",
        "compression_sensitive",
        "concise_rescued",
        "both_wrong",
    ]
    class_names = [
        "budget_limited_recovered",
        "budget_limited_unrecovered",
        "prompt_or_sampling_failure",
    ]
    summary = {
        "model_label": model_label,
        "record_count": expected_count,
        "quadrant_counts": {name: route_counts[name] for name in route_names},
        "quadrant_ratios": {name: route_counts[name] / expected_count for name in route_names},
        "normal_accuracy": normal_correct / expected_count,
        "concise_accuracy": concise_correct / expected_count,
        "concise_minus_normal_accuracy": (concise_correct - normal_correct) / expected_count,
        "normal_length_mean": statistics.fmean(normal_lengths),
        "normal_length_median": statistics.median(normal_lengths),
        "concise_length_mean": statistics.fmean(concise_lengths),
        "concise_length_median": statistics.median(concise_lengths),
        "concise_to_normal_ratio_mean": statistics.fmean(ratios),
        "concise_to_normal_ratio_median": statistics.median(ratios),
        "concise_cap_hit_count": sum(cap_hits_by_quadrant.values()),
        "concise_cap_hit_ratio": sum(cap_hits_by_quadrant.values()) / expected_count,
        "concise_cap_hits_by_quadrant": {
            name: cap_hits_by_quadrant[name] for name in route_names
        },
        "normal_parse_failure_count": normal_parse_failures,
        "concise_parse_failure_count": concise_parse_failures,
        "compression_sensitive_class_counts": {
            name: class_counts[name] for name in class_names
        },
    }
    for key, value in summary.items():
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError(f"summary metric {key} is non-finite")
    return summary


def _response_excerpt(text: str, *, head: int = 1200, tail: int = 600) -> str:
    normalized = str(text).strip()
    if len(normalized) <= head + tail:
        return normalized
    return normalized[:head] + "\n...[excerpt truncated]...\n" + normalized[-tail:]


def _compression_sensitive_markdown(*, model_label: str, records: Sequence[Mapping]) -> str:
    cases = [record for record in records if record.get("quadrant") == "compression_sensitive"]
    lines = [
        f"# Compression-sensitive cases: {model_label}",
        "",
        f"Total cases: {len(cases)}",
        "",
    ]
    for case_number, record in enumerate(cases, start=1):
        normal = record["normal"]
        concise = record["concise"]
        relaxed = record.get("relaxed_concise")
        lines.extend(
            [
                f"## Case {case_number}: question_id={record['question_id']}",
                "",
                f"- source_row_index: {record['source_row_index']}",
                f"- request_seed: {record['request_seed']}",
                f"- failure_class: {record['compression_sensitive_class']}",
                f"- normal: length={normal['length']}, finish={normal['finish_reason']}, "
                f"boxed={normal['boxed_answer']!r}, correct={normal['correct']}",
                f"- concise: length={concise['length']}/{concise['max_tokens']}, "
                f"finish={concise['finish_reason']}, cap_hit={concise['cap_hit']}, "
                f"boxed={concise['boxed_answer']!r}, correct={concise['correct']}",
            ]
        )
        if relaxed is not None:
            lines.append(
                f"- relaxed concise: length={relaxed['length']}/{relaxed['max_tokens']}, "
                f"finish={relaxed['finish_reason']}, boxed={relaxed['boxed_answer']!r}, "
                f"correct={relaxed['correct']}"
            )
        lines.extend(
            [
                "",
                "### Normal excerpt",
                "",
                "```text",
                _response_excerpt(normal["text"]),
                "```",
                "",
                "### Capped concise excerpt",
                "",
                "```text",
                _response_excerpt(concise["text"]),
                "```",
                "",
            ]
        )
        if relaxed is not None:
            lines.extend(
                [
                    "### Relaxed concise excerpt",
                    "",
                    "```text",
                    _response_excerpt(relaxed["text"]),
                    "```",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def _comparison_markdown(comparison: Mapping) -> str:
    base = comparison["models"]["base"]
    adaptive = comparison["models"]["adaptive_step50"]
    lines = [
        "# Paired normal/concise probe comparison",
        "",
        "| Model | Normal accuracy | Concise accuracy | Safe | Sensitive | Rescued | Both wrong | Normal mean length | Concise mean length |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, summary in (("base", base), ("adaptive_step50", adaptive)):
        counts = summary["quadrant_counts"]
        lines.append(
            f"| {label} | {summary['normal_accuracy']:.4f} | {summary['concise_accuracy']:.4f} "
            f"| {counts['compression_safe']} | {counts['compression_sensitive']} "
            f"| {counts['concise_rescued']} | {counts['both_wrong']} "
            f"| {summary['normal_length_mean']:.2f} | {summary['concise_length_mean']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Adaptive minus base quadrant-count deltas",
            "",
        ]
    )
    for name, value in comparison["quadrant_count_delta_adaptive_minus_base"].items():
        lines.append(f"- {name}: {value:+d}")
    return "\n".join(lines) + "\n"


def analyze_run(*, run_dir: str | Path, expected_count: int) -> dict:
    """Write immutable per-model summaries and a cross-model comparison."""

    root = Path(run_dir)
    manifest_path = root / "manifest.json"
    sample_path = root / "sample.jsonl"
    if not manifest_path.is_file() or not sample_path.is_file():
        raise FileNotFoundError("run directory is missing manifest.json or sample.jsonl")
    outputs = [
        root / "base" / "summary.json",
        root / "base" / "compression_sensitive_cases.md",
        root / "adaptive_step50" / "summary.json",
        root / "adaptive_step50" / "compression_sensitive_cases.md",
        root / "comparison.json",
        root / "comparison.md",
    ]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing analysis output: {existing[0]}")

    model_records: dict[str, list[dict]] = {}
    summaries: dict[str, dict] = {}
    for label in ("base", "adaptive_step50"):
        records_path = root / label / "records.jsonl"
        if not records_path.is_file():
            raise FileNotFoundError(f"missing model records: {records_path}")
        records = read_jsonl(records_path)
        summary = summarize_records(records, expected_count=expected_count)
        if summary["model_label"] != label:
            raise ValueError(f"model records under {label} have label {summary['model_label']!r}")
        model_records[label] = records
        summaries[label] = summary

    sample = read_jsonl(sample_path)
    if len(sample) != expected_count:
        raise ValueError(f"sample must contain exactly {expected_count} rows")
    expected_keys = [
        (row.get("sample_ordinal"), row.get("source_row_index"), row.get("question_id"), row.get("request_seed"))
        for row in sample
    ]
    for label, records in model_records.items():
        actual_keys = [
            (record.get("sample_ordinal"), record.get("source_row_index"), record.get("question_id"), record.get("request_seed"))
            for record in records
        ]
        if actual_keys != expected_keys:
            raise ValueError(f"{label} record identities do not match the persisted sample")

    route_names = [
        "compression_safe",
        "compression_sensitive",
        "concise_rescued",
        "both_wrong",
    ]
    comparison = {
        "record_count_per_model": expected_count,
        "models": summaries,
        "quadrant_count_delta_adaptive_minus_base": {
            name: summaries["adaptive_step50"]["quadrant_counts"][name]
            - summaries["base"]["quadrant_counts"][name]
            for name in route_names
        },
        "metric_delta_adaptive_minus_base": {
            name: summaries["adaptive_step50"][name] - summaries["base"][name]
            for name in (
                "normal_accuracy",
                "concise_accuracy",
                "concise_minus_normal_accuracy",
                "normal_length_mean",
                "concise_length_mean",
                "concise_to_normal_ratio_mean",
                "concise_cap_hit_ratio",
            )
        },
    }

    for label in ("base", "adaptive_step50"):
        _atomic_write_json(root / label / "summary.json", summaries[label])
        _atomic_write_text(
            root / label / "compression_sensitive_cases.md",
            _compression_sensitive_markdown(model_label=label, records=model_records[label]),
        )
    _atomic_write_json(root / "comparison.json", comparison)
    _atomic_write_text(root / "comparison.md", _comparison_markdown(comparison))

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must contain an object")
    artifact_paths = [sample_path]
    for label in ("base", "adaptive_step50"):
        artifact_paths.extend(
            [
                root / label / "records.jsonl",
                root / label / "summary.json",
                root / label / "compression_sensitive_cases.md",
            ]
        )
        generation_path = root / label / "generation.json"
        if generation_path.is_file():
            artifact_paths.append(generation_path)
    artifact_paths.extend([root / "comparison.json", root / "comparison.md"])
    manifest["artifacts"] = {
        str(path.relative_to(root)): sha256_file(path) for path in artifact_paths
    }
    _atomic_write_json(manifest_path, manifest)
    return comparison


def _render_messages(tokenizer, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )


def _sampling_params(*, seeds: Sequence[int], max_tokens: Sequence[int]):
    from vllm import SamplingParams

    if len(seeds) != len(max_tokens):
        raise ValueError("sampling seeds and max-token caps must align")
    return [
        SamplingParams(
            temperature=1.0,
            top_p=1.0,
            max_tokens=int(cap),
            n=1,
            seed=int(seed),
        )
        for seed, cap in zip(seeds, max_tokens, strict=True)
    ]


def _generate_batch(llm, *, prompts: Sequence[str], seeds: Sequence[int], caps: Sequence[int]):
    if not prompts:
        return []
    if len(prompts) != len(seeds) or len(prompts) != len(caps):
        raise ValueError("generation prompts, seeds, and caps must align")
    return llm.generate(
        list(prompts),
        sampling_params=_sampling_params(seeds=seeds, max_tokens=caps),
        use_tqdm=True,
    )


def generate_model_records(
    *,
    sample_file: str | Path,
    model_path: str | Path,
    model_label: str,
    output_dir: str | Path,
    expected_count: int,
    tensor_parallel_size: int,
    max_model_len: int,
    max_num_seqs: int,
    gpu_memory_utilization: float,
) -> Path:
    """Generate paired and relaxed responses for one read-only student model."""

    ensure_generation_destinations(model_path=model_path, output_dir=output_dir)
    if not isinstance(model_label, str) or not model_label.strip():
        raise ValueError("model_label must be nonempty")
    sample = read_jsonl(sample_file)
    if len(sample) != expected_count:
        raise ValueError(f"sample must contain exactly {expected_count} rows")
    ordinals = [row.get("sample_ordinal") for row in sample]
    question_ids = [row.get("question_id") for row in sample]
    if ordinals != list(range(expected_count)):
        raise ValueError("sample ordinals are incomplete or out of order")
    if len(set(question_ids)) != expected_count:
        raise ValueError("sample question identities must be distinct")

    from transformers import AutoTokenizer
    from vllm import LLM

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    llm = LLM(
        model=str(model_path),
        tokenizer=str(model_path),
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=True,
    )

    normal_prompt_messages = [normal_messages(row) for row in sample]
    rendered_normal = [_render_messages(tokenizer, messages) for messages in normal_prompt_messages]
    seeds = [int(row["request_seed"]) for row in sample]
    normal_caps = [16384] * expected_count
    normal_outputs = _generate_batch(
        llm, prompts=rendered_normal, seeds=seeds, caps=normal_caps
    )
    if len(normal_outputs) != expected_count:
        raise ValueError("normal generation output count mismatch")
    normal_responses = [
        normalize_request_output(
            output,
            ground_truth=str(row["ground_truth"]),
            max_tokens=16384,
            rendered_prompt=prompt,
        )
        for output, row, prompt in zip(normal_outputs, sample, rendered_normal, strict=True)
    ]

    concise_prompt_messages = [concise_messages(row) for row in sample]
    rendered_concise = [_render_messages(tokenizer, messages) for messages in concise_prompt_messages]
    concise_caps = [concise_cap(response["length"]) for response in normal_responses]
    concise_outputs = _generate_batch(
        llm, prompts=rendered_concise, seeds=seeds, caps=concise_caps
    )
    if len(concise_outputs) != expected_count:
        raise ValueError("concise generation output count mismatch")
    concise_responses = [
        normalize_request_output(
            output,
            ground_truth=str(row["ground_truth"]),
            max_tokens=cap,
            rendered_prompt=prompt,
        )
        for output, row, cap, prompt in zip(
            concise_outputs, sample, concise_caps, rendered_concise, strict=True
        )
    ]

    records: list[dict] = []
    for row, normal_prompt, concise_prompt, normal, concise in zip(
        sample,
        normal_prompt_messages,
        concise_prompt_messages,
        normal_responses,
        concise_responses,
        strict=True,
    ):
        records.append(
            {
                **copy.deepcopy(row),
                "model_label": model_label,
                "model_path": str(Path(model_path).resolve()),
                "normal_prompt": normal_prompt,
                "concise_prompt": concise_prompt,
                "normal": normal,
                "concise": concise,
                "quadrant": quadrant(normal["correct"], concise["correct"]),
                "relaxed_concise": None,
                "compression_sensitive_class": None,
            }
        )

    relaxed_indices = [
        index
        for index, record in enumerate(records)
        if record["quadrant"] == "compression_sensitive" and record["concise"]["cap_hit"]
    ]
    if relaxed_indices:
        generated_indices: list[int] = []
        continuation_prompts: list[dict[str, list[int]]] = []
        continuation_prompt_by_index: dict[int, dict[str, list[int]]] = {}
        continuation_seeds: list[int] = []
        continuation_caps: list[int] = []
        for index in relaxed_indices:
            remaining = normal_responses[index]["length"] - concise_responses[index]["length"]
            if remaining < 0:
                raise ValueError("concise response exceeds the relaxed total budget")
            continuation_prompt = forced_prefix_continuation_prompt(
                concise_outputs[index], concise_responses[index]["token_ids"]
            )
            continuation_prompt_by_index[index] = continuation_prompt
            if remaining == 0:
                continue
            generated_indices.append(index)
            continuation_prompts.append(continuation_prompt)
            continuation_seeds.append(seeds[index])
            continuation_caps.append(remaining)

        continuation_outputs = _generate_batch(
            llm,
            prompts=continuation_prompts,
            seeds=continuation_seeds,
            caps=continuation_caps,
        )
        if len(continuation_outputs) != len(generated_indices):
            raise ValueError("forced-prefix continuation output count mismatch")
        output_by_index = dict(zip(generated_indices, continuation_outputs, strict=True))
        for index in relaxed_indices:
            relaxed = normalize_forced_prefix_continuation(
                output_by_index.get(index),
                capped_response=concise_responses[index],
                tokenizer=tokenizer,
                ground_truth=str(sample[index]["ground_truth"]),
                total_max_tokens=normal_responses[index]["length"],
                rendered_prompt=rendered_concise[index],
                continuation_seed=seeds[index],
                continuation_prompt=continuation_prompt_by_index[index],
            )
            records[index]["relaxed_concise"] = relaxed

    for record in records:
        record["compression_sensitive_class"] = classify_compression_sensitive(record)

    records_path = output_path / "records.jsonl"
    _atomic_write_jsonl(records_path, records)
    _atomic_write_json(
        output_path / "generation.json",
        {
            "model_label": model_label,
            "model_path": str(Path(model_path).resolve()),
            "record_count": len(records),
            "sample_file": str(Path(sample_file).resolve()),
            "tensor_parallel_size": tensor_parallel_size,
            "max_model_len": max_model_len,
            "max_num_seqs": max_num_seqs,
            "gpu_memory_utilization": gpu_memory_utilization,
            "relaxed_counterfactual_mode": "forced_prefix_continuation",
        },
    )
    return records_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--dataset", required=True)
    prepare.add_argument("--sample-file", required=True)
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--sample-size", type=int, default=128)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--source-commit", required=True)
    prepare.add_argument("--expected-dataset-sha256", required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--sample-file", required=True)
    generate.add_argument("--model-path", required=True)
    generate.add_argument("--model-label", required=True)
    generate.add_argument("--output-dir", required=True)
    generate.add_argument("--expected-count", type=int, default=128)
    generate.add_argument("--tensor-parallel-size", type=int, default=4)
    generate.add_argument("--max-model-len", type=int, default=40960)
    generate.add_argument("--max-num-seqs", type=int, default=128)
    generate.add_argument("--gpu-memory-utilization", type=float, default=0.90)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--run-dir", required=True)
    analyze.add_argument("--expected-count", type=int, default=128)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "prepare":
        manifest = prepare_sample(
            dataset_path=args.dataset,
            sample_file=args.sample_file,
            manifest_file=args.manifest,
            sample_size=args.sample_size,
            seed=args.seed,
            source_commit=args.source_commit,
            expected_dataset_sha256=args.expected_dataset_sha256,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    if args.command == "generate":
        records_path = generate_model_records(
            sample_file=args.sample_file,
            model_path=args.model_path,
            model_label=args.model_label,
            output_dir=args.output_dir,
            expected_count=args.expected_count,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        print(f"records_file={records_path}")
        return
    if args.command == "analyze":
        comparison = analyze_run(run_dir=args.run_dir, expected_count=args.expected_count)
        print(json.dumps(comparison, indent=2, sort_keys=True))
        return
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()
