"""Raw-prompt Qwen3 embedding baseline for OPD verification."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from math_eval.opd_proxy_gradient_verify_artifacts import (
    VectorSet,
    build_runtime_metadata,
    canonical_json_bytes,
    load_vector_set,
    recursive_file_manifest,
    repository_state,
    sha256_file,
    sha256_id_lines,
)
from math_eval.replay_opd_proxy_gradients import ReplayVector, run_replay_shard


QWEN3_MODEL_NAME = "Qwen/Qwen3-0.6B"
EMBEDDING_REPRESENTATION = "E"
EMBEDDING_DIMENSION = 1024


@dataclass(frozen=True)
class EmbeddingRecord:
    vector: ReplayVector
    prompt_token_count: int
    pre_normalization_norm: torch.Tensor

    @property
    def vector_id(self) -> str:
        return self.vector.vector_id


@dataclass(frozen=True)
class EmbeddingCollector:
    model: torch.nn.Module
    tokenizer: object
    model_manifest_sha256: str
    tokenizer_manifest_sha256: str
    chat_template_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.model, torch.nn.Module):
            raise TypeError("embedding collector model must be a torch module")
        if not callable(getattr(self.tokenizer, "apply_chat_template", None)):
            raise TypeError("embedding collector tokenizer lacks apply_chat_template")
        for field_name in (
            "model_manifest_sha256",
            "tokenizer_manifest_sha256",
            "chat_template_sha256",
        ):
            value = getattr(self, field_name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"embedding collector {field_name} is invalid")
        self.model.eval()

    def collect_one(self, row: Mapping[str, object]) -> EmbeddingRecord:
        stable_id, prompt, split, expected_tokens = _validate_embedding_row(row)
        if any(parameter.grad is not None for parameter in self.model.parameters()):
            raise ValueError("embedding collector started with parameter gradients")
        input_ids = format_embedding_prompt(self.tokenizer, prompt)
        if input_ids.shape[1] != expected_tokens:
            raise ValueError("embedding prompt token count differs from frozen preflight")
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        self.model.eval()
        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        hidden_states = getattr(output, "hidden_states", None)
        if not isinstance(hidden_states, tuple | list) or not hidden_states:
            raise ValueError("embedding model did not return final hidden states")
        final_hidden = hidden_states[-1]
        normalized = mean_pool_last_hidden(final_hidden, attention_mask)
        raw = _raw_mean_pool(final_hidden, attention_mask)
        raw_norm = torch.linalg.vector_norm(raw, dim=1)
        if normalized.shape != (1, EMBEDDING_DIMENSION):
            raise ValueError("embedding model hidden dimension differs from 1024")
        if any(parameter.grad is not None for parameter in self.model.parameters()):
            raise ValueError("embedding collection materialized parameter gradients")
        vector = ReplayVector(
            vector_id=f"E:{stable_id}",
            stable_id=stable_id,
            split=split,
            representation=EMBEDDING_REPRESENTATION,
            engine_seed=0,
            rollout_slot=None,
            aggregation="final_hidden_mean_pool_l2",
            source_capture_sha256="0" * 64,
            projected_gradient=normalized.squeeze(0).detach().cpu().to(torch.float32),
            full_gradient_norm=raw_norm.squeeze(0).detach().cpu().to(torch.float32),
            projected_gradient_norm=torch.tensor(1.0, dtype=torch.float32),
            valid_token_count=torch.tensor(float(expected_tokens)),
            response_length=torch.tensor(float(expected_tokens)),
            sampled_reverse_kl=torch.tensor(0.0),
            opd_signal_rms=torch.tensor(0.0),
            prompt_token_count=expected_tokens,
            full_token_count=expected_tokens,
        )
        return EmbeddingRecord(
            vector=vector,
            prompt_token_count=expected_tokens,
            pre_normalization_norm=raw_norm.squeeze(0).detach().cpu().to(torch.float32),
        )


def _extract_input_ids(value: object) -> torch.Tensor:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if isinstance(value, torch.Tensor):
        tensor = value
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        tensor = torch.as_tensor(value, dtype=torch.long)
    else:
        raise ValueError("embedding chat template did not return token IDs")
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.shape[0] != 1 or tensor.shape[1] <= 0:
        raise ValueError("embedding prompt token IDs must have shape (1, length)")
    return tensor.to(torch.long).contiguous()


def _marker_ids(tokenizer: object) -> list[int]:
    if not callable(tokenizer):
        raise ValueError("embedding tokenizer must tokenize the assistant marker")
    encoded = tokenizer("<|im_start|>assistant", add_special_tokens=False)
    value = encoded.get("input_ids") if isinstance(encoded, Mapping) else encoded
    if isinstance(value, torch.Tensor):
        value = value.reshape(-1).tolist()
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError("assistant marker tokenization is invalid")
    ids = [int(token_id) for token_id in value]
    if not ids:
        raise ValueError("assistant marker tokenization is empty")
    return ids


def format_embedding_prompt(tokenizer: object, prompt: str) -> torch.Tensor:
    """Format one raw user prompt with the disabled-thinking generation prefix."""
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("embedding prompt must be a nonempty string")
    apply = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply):
        raise ValueError("embedding tokenizer lacks apply_chat_template")
    value = apply(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    )
    input_ids = _extract_input_ids(value)
    marker = _marker_ids(tokenizer)
    sequence = input_ids[0].tolist()
    starts = [
        index
        for index in range(len(sequence) - len(marker) + 1)
        if sequence[index : index + len(marker)] == marker
    ]
    if len(starts) != 1:
        raise ValueError("embedding prompt lacks exactly one assistant-generation prefix")
    return input_ids


def _raw_mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
        raise ValueError("embedding hidden states must be three-dimensional")
    if not isinstance(attention_mask, torch.Tensor) or tuple(
        attention_mask.shape
    ) != tuple(hidden.shape[:2]):
        raise ValueError("embedding attention mask shape mismatch")
    if not torch.all((attention_mask == 0) | (attention_mask == 1)):
        raise ValueError("embedding attention mask must be binary")
    mask = attention_mask.bool()
    counts = mask.sum(dim=1)
    if torch.any(counts <= 0):
        raise ValueError("embedding attention mask has zero valid tokens")
    value = hidden.to(torch.float32)
    if not bool(torch.isfinite(value[mask]).all().item()):
        raise ValueError("embedding hidden states must be finite on valid tokens")
    masked = torch.where(mask.unsqueeze(-1), value, torch.zeros_like(value))
    return masked.sum(dim=1) / counts.to(torch.float32).unsqueeze(1)


def mean_pool_last_hidden(
    hidden: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Mean-pool non-padding final states and L2-normalize in float32."""
    pooled = _raw_mean_pool(hidden, attention_mask)
    norms = torch.linalg.vector_norm(pooled, dim=1, keepdim=True)
    if not bool(torch.isfinite(norms).all().item()) or torch.any(norms <= 0):
        raise ValueError("embedding mean pool must be finite and nonzero")
    normalized = pooled / norms
    if not bool(torch.isfinite(normalized).all().item()):
        raise ValueError("normalized embedding must be finite")
    return normalized.contiguous()


def _validate_embedding_row(
    row: Mapping[str, object]
) -> tuple[str, str, str, int]:
    if not isinstance(row, Mapping):
        raise TypeError("embedding row must be mapping-like")
    stable_id = row.get("stable_id")
    prompt = row.get("prompt")
    split = row.get("split")
    token_count = row.get("prompt_token_count_0_6b")
    if not isinstance(stable_id, str) or not stable_id:
        raise ValueError("embedding stable ID must be nonempty")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"embedding prompt is invalid for {stable_id}")
    if not isinstance(split, str) or not split:
        raise ValueError(f"embedding split is invalid for {stable_id}")
    if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count <= 0:
        raise ValueError(f"embedding prompt token count is invalid for {stable_id}")
    return stable_id, prompt, split, token_count


def _validate_candidate_rows(
    rows: Sequence[Mapping[str, object]], expected_candidate_ids: Sequence[str]
) -> tuple[Mapping[str, object], ...]:
    normalized = tuple(rows)
    expected = tuple(expected_candidate_ids)
    if not normalized or len(normalized) != len(expected):
        raise ValueError("embedding rows do not match candidate order")
    actual = []
    for index, row in enumerate(normalized):
        stable_id, _, split, _ = _validate_embedding_row(row)
        if split != "candidate":
            raise ValueError(f"embedding row {stable_id} is not a candidate")
        if row.get("manifest_index") != index:
            raise ValueError("embedding candidate manifest indices are not contiguous")
        actual.append(stable_id)
    if tuple(actual) != expected or len(set(actual)) != len(actual):
        raise ValueError("embedding candidate order differs from frozen manifest")
    return normalized


def collect_embedding_shard(
    *,
    rows: Sequence[Mapping[str, object]],
    expected_candidate_ids: Sequence[str],
    collector,
    output_directory: Path,
    stage: int,
    stage1_vector_directory: Path | None,
    parent_hashes: Mapping[str, str],
    source_snapshot: Mapping[str, object],
    repository: Mapping[str, object],
    runtime: Mapping[str, object],
    chunk_size: int = 16,
) -> VectorSet:
    """Collect deterministic prompt embeddings, appending only new Stage-2 rows."""
    if not callable(getattr(collector, "collect_one", None)):
        raise TypeError("embedding collector must expose collect_one")
    normalized = _validate_candidate_rows(rows, expected_candidate_ids)
    if stage not in {0, 1, 2}:
        raise ValueError("embedding collection stage must be 0, 1, or 2")
    parents = dict(parent_hashes)
    source_hash = parents.get("sample_manifest_sha256")
    if not isinstance(source_hash, str):
        raise ValueError("embedding collection requires sample manifest hash")
    if parents.get("model_manifest_sha256") != collector.model_manifest_sha256:
        raise ValueError("embedding model parent hash mismatch")
    if (
        parents.get("tokenizer_manifest_sha256")
        != collector.tokenizer_manifest_sha256
    ):
        raise ValueError("embedding tokenizer parent hash mismatch")
    source_snapshot_hash = source_snapshot.get("manifest_sha256")
    if not isinstance(source_snapshot_hash, str) or len(source_snapshot_hash) != 64:
        raise ValueError("embedding collection requires source snapshot hash")
    parents["source_snapshot_sha256"] = source_snapshot_hash
    append_start = 0
    if stage == 2:
        if stage1_vector_directory is None:
            raise ValueError("Stage 2 embedding collection requires Stage-1 vectors")
        stage1 = load_vector_set(
            stage1_vector_directory,
            expected_representation=EMBEDDING_REPRESENTATION,
        )
        expected_prefix = tuple(expected_candidate_ids[: len(stage1.vector_ids)])
        if stage1.vector_ids != tuple(f"E:{stable_id}" for stable_id in expected_prefix):
            raise ValueError("Stage-1 embeddings are not the Stage-2 candidate prefix")
        append_start = len(stage1.vector_ids)
        if append_start <= 0 or append_start >= len(normalized):
            raise ValueError("Stage 2 embedding append range is empty or invalid")
        parents["stage1_complete_sha256"] = sha256_file(
            Path(stage1_vector_directory) / "COMPLETE.json"
        )
    elif stage1_vector_directory is not None:
        raise ValueError(f"Stage {stage} embedding collection forbids Stage-1 vectors")
    new_rows = normalized[append_start:]
    expected_vector_ids = tuple(f"E:{row['stable_id']}" for row in new_rows)

    def record_factory(index: int) -> ReplayVector:
        record = collector.collect_one(new_rows[index])
        if not isinstance(record, EmbeddingRecord):
            raise TypeError("embedding collector returned the wrong record type")
        return replace(record.vector, source_capture_sha256=source_hash)

    metadata = {
        "model_name": QWEN3_MODEL_NAME,
        "model_manifest_sha256": collector.model_manifest_sha256,
        "tokenizer_manifest_sha256": collector.tokenizer_manifest_sha256,
        "chat_template_sha256": collector.chat_template_sha256,
        "chat_template_kwargs": {
            "tokenize": True,
            "add_generation_prompt": True,
            "enable_thinking": False,
            "return_tensors": "pt",
        },
        "pooling": "final_hidden_nonpadding_mean_l2",
        "model_mode": "eval_no_grad",
        "model_dtype": "bfloat16",
        "pooling_dtype": "float32",
        "stage": stage,
        "candidate_count": len(normalized),
        "candidate_ids_sha256": sha256_id_lines(tuple(expected_candidate_ids)),
        "append_start": append_start,
        "non_applicable_scalar_fields": [
            "sampled_reverse_kl",
            "opd_signal_rms",
            "response_length",
        ],
        "full_gradient_norm_semantics": "pre_normalization_embedding_norm",
    }
    return run_replay_shard(
        output_directory=output_directory,
        expected_vector_ids=expected_vector_ids,
        record_factory=record_factory,
        representation=EMBEDDING_REPRESENTATION,
        parent_hashes=parents,
        source_snapshot=source_snapshot,
        repository=repository,
        runtime=runtime,
        metadata=metadata,
        chunk_size=chunk_size,
        verifier_status="not_computed",
    )


def validate_embedding_stage_union(
    stage1_vector_directory: Path,
    stage2_vector_directory: Path,
    *,
    expected_candidate_ids: Sequence[str],
) -> tuple[str, ...]:
    stage1 = load_vector_set(
        stage1_vector_directory, expected_representation=EMBEDDING_REPRESENTATION
    )
    stage2 = load_vector_set(
        stage2_vector_directory, expected_representation=EMBEDDING_REPRESENTATION
    )
    combined = stage1.vector_ids + stage2.vector_ids
    expected = tuple(f"E:{stable_id}" for stable_id in expected_candidate_ids)
    if combined != expected or len(set(combined)) != len(combined):
        raise ValueError("Stage-1/Stage-2 embedding vector union coverage mismatch")
    if set(stage1.stable_ids) & set(stage2.stable_ids):
        raise ValueError("Stage-2 embeddings recompute Stage-1 rows")
    return combined


def load_production_embedding_collector(
    *, model_path: Path, expected_model_sha256: str
) -> EmbeddingCollector:
    if torch.cuda.device_count() != 1:
        raise RuntimeError("embedding collection requires exactly one visible CUDA device")
    model_manifest = recursive_file_manifest(model_path)
    if model_manifest["manifest_sha256"] != expected_model_sha256:
        raise ValueError("Qwen3-0.6B embedding model hash mismatch")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
        attn_implementation="flash_attention_2",
    ).to("cuda:0")
    if model.config.hidden_size != EMBEDDING_DIMENSION:
        raise ValueError("Qwen3-0.6B embedding dimension differs from 1024")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str) or not chat_template:
        raise ValueError("Qwen3 tokenizer lacks a chat template")
    chat_template_sha256 = hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
    return EmbeddingCollector(
        model=model,
        tokenizer=tokenizer,
        model_manifest_sha256=str(model_manifest["manifest_sha256"]),
        tokenizer_manifest_sha256=str(model_manifest["manifest_sha256"]),
        chat_template_sha256=chat_template_sha256,
    )


def _load_stage_rows(path: Path) -> tuple[dict[str, object], ...]:
    rows = []
    with Path(path).open("rb") as handle:
        for line in handle:
            value = json.loads(line)
            if not isinstance(value, dict) or line != canonical_json_bytes(value):
                raise ValueError("stage sample manifest must be canonical JSONL")
            if value.get("split") == "candidate":
                rows.append(value)
    if not rows:
        raise ValueError("stage sample manifest contains no embedding candidates")
    return tuple(rows)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect Qwen3 prompt embeddings")
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--stage", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--stage1-vector-directory", type=Path)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--chunk-size", type=int, default=16)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    rows = _load_stage_rows(args.sample_manifest)
    collector = load_production_embedding_collector(
        model_path=args.model_path,
        expected_model_sha256=args.expected_model_sha256,
    )
    source_payload = args.source_snapshot.read_bytes()
    source_snapshot = json.loads(source_payload)
    if not isinstance(source_snapshot, dict) or source_payload != canonical_json_bytes(
        source_snapshot
    ):
        raise ValueError("embedding source snapshot must be canonical JSON")
    repository = repository_state(args.repository_root)
    if repository["status"]:
        raise ValueError("embedding collection requires a clean repository")
    collect_embedding_shard(
        rows=rows,
        expected_candidate_ids=tuple(str(row["stable_id"]) for row in rows),
        collector=collector,
        output_directory=args.output_directory,
        stage=args.stage,
        stage1_vector_directory=args.stage1_vector_directory,
        parent_hashes={
            "sample_manifest_sha256": sha256_file(args.sample_manifest),
            "model_manifest_sha256": collector.model_manifest_sha256,
            "tokenizer_manifest_sha256": collector.tokenizer_manifest_sha256,
        },
        source_snapshot=source_snapshot,
        repository=repository,
        runtime=build_runtime_metadata("gvendi_analysis"),
        chunk_size=args.chunk_size,
    )
    return 0


__all__ = [
    "EmbeddingCollector",
    "EmbeddingRecord",
    "collect_embedding_shard",
    "format_embedding_prompt",
    "load_production_embedding_collector",
    "mean_pool_last_hidden",
    "validate_embedding_stage_union",
]


if __name__ == "__main__":
    raise SystemExit(main())
