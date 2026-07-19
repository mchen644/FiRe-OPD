"""Paired normal/concise compression-sensitivity probe utilities."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
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
    """Require a same-seed relaxed response to preserve the capped token prefix."""

    capped = list(capped_ids)
    relaxed = list(relaxed_ids)
    if not capped or not relaxed:
        raise ValueError("capped and relaxed token sequences must be nonempty")
    if len(relaxed) < len(capped) or relaxed[: len(capped)] != capped:
        raise ValueError("relaxed concise response does not preserve the capped token prefix")


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


def ensure_generation_destinations(*, model_path: str | Path, output_dir: str | Path) -> None:
    model = Path(model_path)
    output = Path(output_dir)
    if not model.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {model}")
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")


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
        relaxed_outputs = _generate_batch(
            llm,
            prompts=[rendered_concise[index] for index in relaxed_indices],
            seeds=[seeds[index] for index in relaxed_indices],
            caps=[normal_responses[index]["length"] for index in relaxed_indices],
        )
        if len(relaxed_outputs) != len(relaxed_indices):
            raise ValueError("relaxed concise generation output count mismatch")
        for index, output in zip(relaxed_indices, relaxed_outputs, strict=True):
            relaxed = normalize_request_output(
                output,
                ground_truth=str(sample[index]["ground_truth"]),
                max_tokens=normal_responses[index]["length"],
                rendered_prompt=rendered_concise[index],
            )
            validate_relaxed_prefix(records[index]["concise"]["token_ids"], relaxed["token_ids"])
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
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()
