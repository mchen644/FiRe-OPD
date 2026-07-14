"""Same-model completion-only SFT-gradient baseline for OPD verification."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from math_eval.collect_prismatic_gradients import (
    construct_strict_collector,
    load_official_gradient_computer_class,
)
from math_eval.opd_proxy_gradient_projection import (
    PRISMATIC_REFERENCE_COMMIT,
    PRISMATIC_REFERENCE_TREE,
    ProjectionConfig,
    ReferenceSnapshot,
    build_projection_manifest,
    full_gradient_l2_norm,
    sha256_parameter_layout,
    verify_prismatic_reference,
)
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
SFT_REPRESENTATION = "S"
IGNORE_INDEX = -100
MAX_SFT_CONTEXT_TOKENS = 40_960


@dataclass(frozen=True)
class SFTGradientRecord:
    vector: ReplayVector
    supervised_label_count: int
    full_token_count: int
    prompt_token_count: int
    completion_token_count: int

    @property
    def vector_id(self) -> str:
        return self.vector.vector_id

    @property
    def projected_gradient(self) -> torch.Tensor:
        return self.vector.projected_gradient

    @property
    def full_gradient_norm(self) -> torch.Tensor:
        return self.vector.full_gradient_norm


@dataclass(frozen=True)
class OfficialSFTCollector:
    gradient_computer: object
    reference: ReferenceSnapshot
    projection_manifest: dict[str, object]

    def collect_one(self, row: Mapping[str, object]) -> SFTGradientRecord:
        example = build_sft_example(row)
        stable_id = example["stable_id"]
        prompt = example["prompt"]
        completion = example["completion"]
        encoding = self.gradient_computer.prepare_model_input(prompt, completion)
        supervised_count, full_count = _validate_completion_only_encoding(
            encoding, row
        )
        full_gradient = self.gradient_computer.obtain_gradient(encoding)
        if (
            not isinstance(full_gradient, torch.Tensor)
            or full_gradient.ndim != 1
            or full_gradient.numel() == 0
            or not full_gradient.is_floating_point()
            or not bool(torch.isfinite(full_gradient).all().item())
        ):
            raise ValueError("official SFT full gradient must be finite and one-dimensional")
        full_float32 = full_gradient.detach().to(torch.float32).contiguous()
        full_norm = full_gradient_l2_norm(full_float32)
        projected_mapping = self.gradient_computer.project_gradients(
            {stable_id: full_gradient}
        )
        if not isinstance(projected_mapping, Mapping) or set(projected_mapping) != {
            stable_id
        }:
            raise ValueError("official projector returned wrong SFT vector IDs")
        projected = projected_mapping[stable_id]
        if not isinstance(projected, torch.Tensor):
            raise ValueError("official SFT projection must be a torch tensor")
        if projected.shape == (1, 1024):
            projected = projected.squeeze(0)
        projected = projected.detach().to(torch.float32).contiguous()
        if (
            projected.shape != (1024,)
            or not bool(torch.isfinite(projected).all().item())
            or float(torch.linalg.vector_norm(projected).item()) <= 0
        ):
            raise ValueError("official SFT projection must be finite nonzero float32[1024]")
        split = row.get("split")
        if not isinstance(split, str) or not split:
            raise ValueError("SFT row split must be a nonempty string")
        completion_count = _positive_row_int(
            row, "r1_completion_token_count_0_6b"
        )
        prompt_count = _positive_row_int(row, "prompt_token_count_0_6b")
        vector = ReplayVector(
            vector_id=f"S:{stable_id}",
            stable_id=stable_id,
            split=split,
            representation=SFT_REPRESENTATION,
            engine_seed=0,
            rollout_slot=None,
            aggregation="completion_only_sft",
            source_capture_sha256="0" * 64,
            projected_gradient=projected,
            full_gradient_norm=full_norm.detach().cpu().to(torch.float32),
            projected_gradient_norm=torch.linalg.vector_norm(projected),
            valid_token_count=torch.tensor(float(supervised_count)),
            response_length=torch.tensor(float(completion_count)),
            sampled_reverse_kl=torch.tensor(0.0),
            opd_signal_rms=torch.tensor(0.0),
            prompt_token_count=prompt_count,
            completion_token_count=completion_count,
            supervised_label_count=supervised_count,
            full_token_count=full_count,
        )
        return SFTGradientRecord(
            vector=vector,
            supervised_label_count=supervised_count,
            full_token_count=full_count,
            prompt_token_count=prompt_count,
            completion_token_count=completion_count,
        )


def _positive_row_int(row: Mapping[str, object], field: str) -> int:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"SFT row {field} must be a positive integer")
    return value


def build_sft_example(row: Mapping[str, object]) -> dict[str, object]:
    """Use the original exact question and frozen R1 completion."""
    if not isinstance(row, Mapping):
        raise TypeError("SFT row must be mapping-like")
    stable_id = row.get("stable_id")
    prompt = row.get("prompt")
    completion = row.get("completion")
    if not isinstance(stable_id, str) or not stable_id:
        raise ValueError("SFT row stable_id must be nonempty")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"SFT prompt is invalid for {stable_id}")
    if not isinstance(completion, str) or not completion:
        raise ValueError(f"SFT completion is invalid for {stable_id}")
    return {
        "stable_id": stable_id,
        "prompt": prompt,
        "completion": completion,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ],
    }


def _validate_completion_only_encoding(
    encoding: object, row: Mapping[str, object]
) -> tuple[int, int]:
    if not isinstance(encoding, Mapping):
        raise ValueError("official prepare_model_input must return a mapping")
    required = {"input_ids", "attention_mask", "labels"}
    missing = required - set(encoding)
    if missing:
        raise ValueError(f"official SFT encoding lacks fields: {sorted(missing)}")
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    labels = encoding["labels"]
    if not all(isinstance(value, torch.Tensor) for value in (input_ids, attention_mask, labels)):
        raise ValueError("official SFT encoding fields must be tensors")
    if (
        input_ids.ndim != 2
        or input_ids.shape[0] != 1
        or tuple(attention_mask.shape) != tuple(input_ids.shape)
        or tuple(labels.shape) != tuple(input_ids.shape)
    ):
        raise ValueError("official SFT encoding tensors have inconsistent shapes")
    full_count = int(attention_mask.bool().sum().item())
    expected_full = _positive_row_int(row, "sft_full_token_count_0_6b")
    if full_count != expected_full or input_ids.shape[1] != expected_full:
        raise ValueError("official SFT context length differs from frozen preflight")
    expected_supervised = _positive_row_int(row, "sft_supervised_label_count")
    supervised = labels.ne(IGNORE_INDEX)
    actual_supervised = int(supervised.sum().item())
    if actual_supervised == 0:
        raise ValueError("official SFT encoding has zero completion labels")
    suffix_start = labels.shape[1] - expected_supervised
    if bool(supervised[:, :suffix_start].any().item()):
        raise ValueError("official SFT encoding contains prompt-supervised labels")
    if not bool(supervised[:, suffix_start:].all().item()):
        raise ValueError("official SFT labels must form one contiguous suffix")
    if actual_supervised != expected_supervised:
        raise ValueError("official SFT supervised label count differs from preflight")
    if not torch.equal(labels[supervised], input_ids[supervised]):
        raise ValueError("official SFT labels differ from completion token IDs")
    return actual_supervised, full_count


def _validate_projection_manifest(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("official SFT collector lacks projection manifest")
    manifest = dict(value)
    constructor = manifest.get("constructor")
    if not isinstance(constructor, Mapping):
        raise ValueError("official SFT projection constructor is missing")
    expected = {
        "proj_dim": 1024,
        "seed": 0,
        "proj_type": "rademacher",
        "device": "cuda:0",
        "dtype": "float16",
        "block_size": 128,
        "max_batch_size": 16,
    }
    for field, expected_value in expected.items():
        if constructor.get(field) != expected_value:
            raise ValueError(f"official SFT projection {field} mismatch")
    grad_dim = constructor.get("grad_dim")
    if isinstance(grad_dim, bool) or not isinstance(grad_dim, int) or grad_dim <= 0:
        raise ValueError("official SFT projection grad_dim mismatch")
    if manifest.get("backend") != "CudaProjector":
        raise ValueError("official SFT projection must use CudaProjector")
    if manifest.get("model_id") != 0:
        raise ValueError("official SFT projection model_id mismatch")
    if manifest.get("output_scale") != "1/sqrt(1024)":
        raise ValueError("official SFT projection output scaling mismatch")
    if manifest.get("native_gradient_dtype") != "float32":
        raise ValueError("official SFT native gradient dtype mismatch")
    if manifest.get("projector_input_dtype") != "float16":
        raise ValueError("official SFT projector input dtype mismatch")
    if manifest.get("stored_dtype") != "float32":
        raise ValueError("official SFT stored projection dtype mismatch")
    layout_hash = manifest.get("parameter_layout_sha256")
    if (
        not isinstance(layout_hash, str)
        or len(layout_hash) != 64
        or any(character not in "0123456789abcdef" for character in layout_hash)
    ):
        raise ValueError("official SFT parameter layout hash mismatch")
    return manifest


def construct_official_sft_collector(
    gradient_computer: object,
    reference: ReferenceSnapshot,
) -> OfficialSFTCollector:
    """Wrap only an exact pinned official GradientComputer instance."""
    if not isinstance(reference, ReferenceSnapshot):
        raise TypeError("SFT reference must be ReferenceSnapshot")
    if reference.commit != PRISMATIC_REFERENCE_COMMIT:
        raise ValueError("official SFT reference commit mismatch")
    if reference.tree != PRISMATIC_REFERENCE_TREE:
        raise ValueError("official SFT reference tree mismatch")
    source = Path(reference.official_gradient_module)
    if not source.is_file() or sha256_file(source) != reference.official_gradient_module_sha256:
        raise ValueError("official SFT reference source hash mismatch")
    for method_name in (
        "prepare_model_input",
        "obtain_gradient",
        "project_gradients",
    ):
        if not callable(getattr(gradient_computer, method_name, None)):
            raise ValueError(f"official GradientComputer lacks {method_name}")
    projection_manifest = _validate_projection_manifest(
        getattr(gradient_computer, "opd_projection_manifest", None)
    )
    return OfficialSFTCollector(
        gradient_computer=gradient_computer,
        reference=reference,
        projection_manifest=projection_manifest,
    )


def _validate_candidate_rows(
    rows: Sequence[Mapping[str, object]], expected_candidate_ids: Sequence[str]
) -> tuple[Mapping[str, object], ...]:
    normalized = tuple(rows)
    expected = tuple(expected_candidate_ids)
    if not normalized or not expected or len(normalized) != len(expected):
        raise ValueError("SFT rows do not match expected candidate order")
    actual = []
    for index, row in enumerate(normalized):
        example = build_sft_example(row)
        stable_id = str(example["stable_id"])
        if row.get("split") != "candidate":
            raise ValueError(f"SFT row {stable_id} is not a candidate")
        if row.get("manifest_index") != index:
            raise ValueError("SFT candidate manifest indices are not contiguous")
        actual.append(stable_id)
    if tuple(actual) != expected or len(set(actual)) != len(actual):
        raise ValueError("SFT candidate order differs from frozen manifest")
    return normalized


def _reject_legacy_qwen25_artifacts(output_directory: Path) -> None:
    root = Path(output_directory)
    if not root.exists():
        return
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid existing SFT vector manifest: {error}") from error
        model_name = (
            manifest.get("metadata", {}).get("model_name")
            if isinstance(manifest, dict)
            and isinstance(manifest.get("metadata"), dict)
            else None
        )
        if model_name != QWEN3_MODEL_NAME:
            raise ValueError("existing Qwen2.5 or unknown-model vector artifact is forbidden")
    for path in root.iterdir():
        if path.name.startswith("vectors_") or path.name.startswith(".vectors_"):
            continue
        if path.name in {
            "SHARD_CONTRACT.json",
            ".SHARD_CONTRACT.json.lock",
            "manifest.json",
            ".manifest.json.lock",
            "COMPLETE.json",
            ".COMPLETE.json.lock",
        }:
            continue
        if path.suffix in {".safetensors", ".txt"} or "gradient" in path.name:
            raise ValueError(
                f"legacy Qwen2.5 gradient artifact is forbidden: {path.name}"
            )


def collect_sft_shard(
    *,
    rows: Sequence[Mapping[str, object]],
    expected_candidate_ids: Sequence[str],
    collector: OfficialSFTCollector,
    output_directory: Path,
    stage: int,
    stage1_vector_directory: Path | None,
    parent_hashes: Mapping[str, str],
    source_snapshot: Mapping[str, object],
    repository: Mapping[str, object],
    runtime: Mapping[str, object],
    metadata: Mapping[str, object],
    chunk_size: int = 16,
) -> VectorSet:
    """Collect all new SFT vectors, with Stage 2 restricted to its append."""
    if not isinstance(collector, OfficialSFTCollector):
        raise TypeError("collector must be an OfficialSFTCollector")
    normalized = _validate_candidate_rows(rows, expected_candidate_ids)
    if stage not in {0, 1, 2}:
        raise ValueError("SFT collection stage must be 0, 1, or 2")
    _reject_legacy_qwen25_artifacts(output_directory)
    model_name = metadata.get("model_name")
    if model_name != QWEN3_MODEL_NAME or "Qwen2.5" in str(model_name):
        raise ValueError("SFT baseline requires exact Qwen3-0.6B, never Qwen2.5")
    parents = dict(parent_hashes)
    source_hash = parents.get("sample_manifest_sha256")
    if not isinstance(source_hash, str):
        raise ValueError("SFT collection requires sample manifest parent hash")
    source_snapshot_hash = source_snapshot.get("manifest_sha256")
    if not isinstance(source_snapshot_hash, str) or len(source_snapshot_hash) != 64:
        raise ValueError("SFT collection requires source snapshot hash")
    parents["source_snapshot_sha256"] = source_snapshot_hash
    append_start = 0
    if stage == 2:
        if stage1_vector_directory is None:
            raise ValueError("Stage 2 SFT collection requires Stage-1 vectors")
        stage1 = load_vector_set(
            stage1_vector_directory, expected_representation=SFT_REPRESENTATION
        )
        expected_prefix = tuple(expected_candidate_ids[: len(stage1.vector_ids)])
        if stage1.vector_ids != tuple(f"S:{stable_id}" for stable_id in expected_prefix):
            raise ValueError("Stage-1 SFT vectors are not the Stage-2 candidate prefix")
        append_start = len(stage1.vector_ids)
        if append_start <= 0 or append_start >= len(normalized):
            raise ValueError("Stage 2 SFT append range is empty or invalid")
        parents["stage1_complete_sha256"] = sha256_file(
            Path(stage1_vector_directory) / "COMPLETE.json"
        )
    elif stage1_vector_directory is not None:
        raise ValueError(f"Stage {stage} SFT collection forbids Stage-1 vectors")
    new_rows = normalized[append_start:]
    expected_vector_ids = tuple(f"S:{row['stable_id']}" for row in new_rows)

    def record_factory(index: int) -> ReplayVector:
        record = collector.collect_one(new_rows[index])
        return replace(record.vector, source_capture_sha256=source_hash)

    output_metadata = {
        **dict(metadata),
        "baseline": "completion_only_sft_gradient",
        "stage": stage,
        "candidate_count": len(normalized),
        "candidate_ids_sha256": sha256_id_lines(tuple(expected_candidate_ids)),
        "append_start": append_start,
        "projection": collector.projection_manifest,
        "non_applicable_scalar_fields": ["sampled_reverse_kl", "opd_signal_rms"],
    }
    return run_replay_shard(
        output_directory=output_directory,
        expected_vector_ids=expected_vector_ids,
        record_factory=record_factory,
        representation=SFT_REPRESENTATION,
        parent_hashes=parents,
        source_snapshot=source_snapshot,
        repository=repository,
        runtime=runtime,
        metadata=output_metadata,
        chunk_size=chunk_size,
        verifier_status="not_computed",
    )


def validate_sft_stage_union(
    stage1_vector_directory: Path,
    stage2_vector_directory: Path,
    *,
    expected_candidate_ids: Sequence[str],
) -> tuple[str, ...]:
    stage1 = load_vector_set(
        stage1_vector_directory, expected_representation=SFT_REPRESENTATION
    )
    stage2 = load_vector_set(
        stage2_vector_directory, expected_representation=SFT_REPRESENTATION
    )
    combined = stage1.vector_ids + stage2.vector_ids
    expected = tuple(f"S:{stable_id}" for stable_id in expected_candidate_ids)
    if combined != expected or len(set(combined)) != len(combined):
        raise ValueError("Stage-1/Stage-2 SFT vector union coverage mismatch")
    if set(stage1.stable_ids) & set(stage2.stable_ids):
        raise ValueError("Stage-2 SFT vectors recompute Stage-1 rows")
    return combined


def load_production_sft_collector(
    *,
    model_path: Path,
    reference_repo: Path,
    expected_model_sha256: str,
) -> tuple[OfficialSFTCollector, dict[str, object], dict[str, object]]:
    """Load exact Qwen3-0.6B and the verified official gradient collector."""
    if torch.cuda.device_count() != 1:
        raise RuntimeError("SFT collection requires exactly one visible CUDA device")
    reference = verify_prismatic_reference(reference_repo)
    model_manifest = recursive_file_manifest(model_path)
    if model_manifest["manifest_sha256"] != expected_model_sha256:
        raise ValueError("Qwen3-0.6B model manifest hash mismatch")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=False,
        attn_implementation="flash_attention_2",
    ).to("cuda:0")
    if model.config.max_position_embeddings != MAX_SFT_CONTEXT_TOKENS:
        raise ValueError("Qwen3-0.6B context length differs from frozen contract")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    gradient_class = load_official_gradient_computer_class(
        reference_repo, PRISMATIC_REFERENCE_COMMIT
    )
    gradient_computer = construct_strict_collector(
        gradient_class,
        QWEN3_MODEL_NAME,
        model,
        tokenizer,
    )
    entries = []
    offset = 0
    for name, parameter in model.named_parameters(remove_duplicate=True):
        if not parameter.requires_grad:
            continue
        entries.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "numel": parameter.numel(),
                "offset": offset,
            }
        )
        offset += parameter.numel()
    layout_hash = sha256_parameter_layout(entries)
    projection_manifest = build_projection_manifest(
        gradient_dimension=offset,
        parameter_layout_sha256=layout_hash,
        config=ProjectionConfig(),
        reference=reference,
    )
    object.__setattr__(gradient_computer, "opd_projection_manifest", projection_manifest)
    # Model and tokenizer are materialized in the same immutable local root.
    tokenizer_manifest = dict(model_manifest)
    return (
        construct_official_sft_collector(gradient_computer, reference),
        model_manifest,
        tokenizer_manifest,
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
        raise ValueError("stage sample manifest contains no candidates")
    return tuple(rows)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect Qwen3 SFT-gradient baseline")
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--reference-repo", type=Path, required=True)
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
    collector, model_manifest, tokenizer_manifest = load_production_sft_collector(
        model_path=args.model_path,
        reference_repo=args.reference_repo,
        expected_model_sha256=args.expected_model_sha256,
    )
    source_snapshot_payload = args.source_snapshot.read_bytes()
    source_snapshot = json.loads(source_snapshot_payload)
    if (
        not isinstance(source_snapshot, dict)
        or source_snapshot_payload != canonical_json_bytes(source_snapshot)
    ):
        raise ValueError("SFT source snapshot must be canonical JSON")
    repository = repository_state(args.repository_root)
    if repository["status"]:
        raise ValueError("SFT collection requires a clean repository")
    collect_sft_shard(
        rows=rows,
        expected_candidate_ids=tuple(str(row["stable_id"]) for row in rows),
        collector=collector,
        output_directory=args.output_directory,
        stage=args.stage,
        stage1_vector_directory=args.stage1_vector_directory,
        parent_hashes={
            "sample_manifest_sha256": sha256_file(args.sample_manifest),
            "model_manifest_sha256": str(model_manifest["manifest_sha256"]),
            "tokenizer_manifest_sha256": str(tokenizer_manifest["manifest_sha256"]),
        },
        source_snapshot=source_snapshot,
        repository=repository,
        runtime=build_runtime_metadata("gvendi_analysis"),
        metadata={
            "model_name": QWEN3_MODEL_NAME,
            "model_manifest_sha256": model_manifest["manifest_sha256"],
            "tokenizer_manifest_sha256": tokenizer_manifest["manifest_sha256"],
        },
        chunk_size=args.chunk_size,
    )
    return 0


__all__ = [
    "OfficialSFTCollector",
    "SFTGradientRecord",
    "build_sft_example",
    "collect_sft_shard",
    "construct_official_sft_collector",
    "load_production_sft_collector",
    "validate_sft_stage_union",
]


if __name__ == "__main__":
    raise SystemExit(main())
