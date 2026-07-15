# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fail-closed capture tensors and resumable artifacts for OPD verification."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from math_eval.opd_proxy_gradient_verify_artifacts import (
    TrajectoryKey,
    atomic_write_bytes,
    canonical_json_bytes,
    sha256_file,
    sha256_id_lines,
    validate_exact_key_coverage,
    write_or_validate_manifest,
)
from safetensors.torch import load as load_safetensors_bytes
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save as save_safetensors

from verl import DataProto
from verl.trainer.ppo.core_algos import get_policy_loss_fn

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CHUNK_RE = re.compile(r"^chunk_(0|[1-9][0-9]*)_(0|[1-9][0-9]*)\.(jsonl|safetensors)$")
_RESERVED_SIDECAR_FIELDS = frozenset(
    {
        "capture_index",
        "tensor_row",
        "chunk_start",
        "chunk_end",
        "tensor_sha256",
        "parent_hashes",
    }
)
_REQUIRED_SEED_MANIFEST_FIELDS = frozenset(
    {
        "vllm_version",
        "engine_args",
        "sampling_args",
        "ordered_prompt_keys_sha256",
        "returned_compound_keys_sha256",
        "model_hashes",
        "tokenizer_hashes",
        "config_hashes",
        "source_hashes",
        "engine_count",
        "generation_call_count",
    }
)


class _DuplicateJsonKey(ValueError):
    pass


def _reject_duplicate_pairs(
    pairs: Sequence[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateJsonKey(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_nonfinite_json(value: str) -> object:
    raise ValueError(f"non-finite JSON value: {value}")


def _strict_json_text(text: str, description: str) -> object:
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_json,
        )
    except (json.JSONDecodeError, _DuplicateJsonKey, ValueError) as error:
        raise ValueError(f"invalid {description}: {error}") from error
    if text.encode("utf-8") != canonical_json_bytes(value):
        raise ValueError(f"invalid {description}: non-canonical JSON bytes")
    return value


@dataclass(frozen=True)
class LoadedCaptureChunks:
    tensors: dict[str, torch.Tensor]
    sidecar_rows: tuple[dict[str, object], ...]
    keys: tuple[TrajectoryKey, ...]
    chunk_records: tuple[dict[str, object], ...]


def _require_nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _require_positive_int(value: object, field: str) -> int:
    parsed = _require_nonnegative_int(value, field)
    if parsed == 0:
        raise ValueError(f"{field} must be positive")
    return parsed


def _normalize_capture_stage(value: object) -> int | str:
    if isinstance(value, bool):
        raise ValueError("capture stage must be 0, 1, 2, or efficacy_pilot")
    if isinstance(value, int) and value in {0, 1, 2}:
        return value
    if isinstance(value, str) and value in {"0", "1", "2"}:
        return int(value)
    if value == "efficacy_pilot":
        return "efficacy_pilot"
    raise ValueError("capture stage must be 0, 1, 2, or efficacy_pilot")


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _normalize_hashes(value: Mapping[str, object], field: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{field} must be a nonempty hash mapping")
    normalized: dict[str, str] = {}
    for name in sorted(value):
        if not isinstance(name, str) or not name:
            raise ValueError(f"{field} names must be nonempty strings")
        normalized[name] = _require_sha256(value[name], f"{field}.{name}")
    return normalized


def _trajectory_key_from_row(row: Mapping[str, object]) -> TrajectoryKey:
    try:
        stable_id = row["stable_id"]
        engine_seed = row["engine_seed"]
        rollout_slot = row["rollout_slot"]
    except KeyError as error:
        raise ValueError(f"capture sidecar lacks key field: {error.args[0]}") from error
    if isinstance(engine_seed, np.integer):
        engine_seed = int(engine_seed)
    if isinstance(rollout_slot, np.integer):
        rollout_slot = int(rollout_slot)
    return TrajectoryKey(stable_id, engine_seed, rollout_slot)  # type: ignore[arg-type]


def trajectory_keys_sha256(keys: Sequence[TrajectoryKey]) -> str:
    normalized = tuple(keys)
    validate_exact_key_coverage(normalized, normalized)
    digest = hashlib.sha256()
    for key in normalized:
        digest.update(key.stable_id.encode("utf-8"))
        digest.update(b"\t")
        digest.update(str(key.engine_seed).encode("ascii"))
        digest.update(b"\t")
        digest.update(str(key.rollout_slot).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def validate_capture_contract(config: Mapping[str, object]) -> dict[str, object]:
    """Resolve and validate the immutable typed work-unit contract."""
    if not isinstance(config, Mapping):
        raise TypeError("capture config must be mapping-like")
    if config.get("enabled") is not True:
        raise ValueError("OPD proxy capture contract is not enabled")
    output_root = config.get("output_root")
    sample_manifest = config.get("sample_manifest")
    if not isinstance(output_root, str) or not output_root:
        raise ValueError("capture output_root must be a nonempty path")
    if not isinstance(sample_manifest, str) or not sample_manifest:
        raise ValueError("capture sample_manifest must be a nonempty path")
    try:
        sample_path = Path(sample_manifest).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"capture sample manifest does not exist: {sample_manifest}") from error
    if not sample_path.is_file():
        raise ValueError("capture sample manifest is not a regular file")
    declared_sample_hash = _require_sha256(
        config.get("sample_manifest_sha256"), "sample_manifest_sha256"
    )
    actual_sample_hash = sha256_file(sample_path)
    if actual_sample_hash != declared_sample_hash:
        raise ValueError(
            "sample manifest hash mismatch: "
            f"expected {declared_sample_hash}, got {actual_sample_hash}"
        )
    source_snapshot = config.get("source_snapshot")
    if not isinstance(source_snapshot, str) or not source_snapshot:
        raise ValueError("capture source_snapshot must be a nonempty path")
    try:
        source_path = Path(source_snapshot).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError(
            f"capture source snapshot does not exist: {source_snapshot}"
        ) from error
    if not source_path.is_file():
        raise ValueError("capture source snapshot is not a regular file")
    declared_source_hash = _require_sha256(
        config.get("source_snapshot_sha256"), "source_snapshot_sha256"
    )
    try:
        source_value = _strict_json_text(
            source_path.read_text(encoding="utf-8"),
            "experiment source snapshot",
        )
    except (OSError, UnicodeError) as error:
        raise ValueError(f"cannot read experiment source snapshot: {error}") from error
    if not isinstance(source_value, dict):
        raise ValueError("experiment source snapshot must be a JSON object")
    source_files = source_value.get("files")
    if not isinstance(source_files, list) or not source_files:
        raise ValueError("experiment source snapshot must contain source files")
    actual_source_hash = hashlib.sha256(
        canonical_json_bytes(source_files)
    ).hexdigest()
    if (
        source_value.get("manifest_sha256") != actual_source_hash
        or actual_source_hash != declared_source_hash
    ):
        raise ValueError(
            "source snapshot hash mismatch: "
            f"expected {declared_source_hash}, got {actual_source_hash}"
        )

    stage = _normalize_capture_stage(config.get("stage"))
    pair = config.get("pair")
    if pair not in {"target", "proxy"}:
        raise ValueError("capture pair must be target or proxy")
    engine_seed = _require_nonnegative_int(config.get("engine_seed"), "engine_seed")
    native_rollouts = _require_positive_int(
        config.get("native_rollouts"), "native_rollouts"
    )
    algorithm_value = config.get("algorithm_contract_sha256")
    algorithm_contract_sha256 = (
        None
        if algorithm_value is None
        else _require_sha256(algorithm_value, "algorithm_contract_sha256")
    )
    if stage == "efficacy_pilot":
        if engine_seed != 42 or native_rollouts != 1:
            raise ValueError(
                "efficacy_pilot capture requires seed 42 and native_rollouts 1"
            )
        if algorithm_contract_sha256 is None:
            raise ValueError(
                "efficacy_pilot capture requires algorithm_contract_sha256"
            )
    elif engine_seed not in {42, 43} or native_rollouts != 4:
        raise ValueError(
            "numbered-stage capture requires seeds 42/43 and native_rollouts 4"
        )
    expected_questions = _require_positive_int(
        config.get("expected_questions"), "expected_questions"
    )
    chunk_size = _require_positive_int(config.get("chunk_size"), "chunk_size")
    schema_version = _require_positive_int(
        config.get("schema_version"), "schema_version"
    )
    if schema_version != 1:
        raise ValueError("unsupported capture schema version")
    return {
        "enabled": True,
        "output_root": str(Path(output_root).expanduser().resolve()),
        "sample_manifest": str(sample_path),
        "sample_manifest_sha256": actual_sample_hash,
        "source_snapshot": str(source_path),
        "source_snapshot_sha256": actual_source_hash,
        "algorithm_contract_sha256": algorithm_contract_sha256,
        "stage": stage,
        "pair": pair,
        "engine_seed": engine_seed,
        "native_rollouts": native_rollouts,
        "expected_questions": expected_questions,
        "chunk_size": chunk_size,
        "schema_version": schema_version,
    }


def _validate_exact_rollout_slots(
    keys: Sequence[TrajectoryKey], native_rollouts: int
) -> None:
    grouped: dict[str, list[int]] = defaultdict(list)
    for key in keys:
        grouped[key.stable_id].append(key.rollout_slot)
    expected_slots = list(range(native_rollouts))
    for stable_id, actual_slots in grouped.items():
        if sorted(actual_slots) != expected_slots:
            raise ValueError(
                f"stable ID {stable_id} does not have exact rollout slots {expected_slots}"
            )


def attach_and_validate_keys(
    batch: DataProto,
    *,
    engine_seed: int,
    native_rollouts: int = 4,
    expected_keys: Sequence[TrajectoryKey] | None = None,
    require_complete_slots: bool = True,
) -> tuple[DataProto, tuple[TrajectoryKey, ...]]:
    """Attach engine seed and validate stable-ID/seed/slot compound keys."""
    seed = _require_nonnegative_int(engine_seed, "engine_seed")
    rollout_count = _require_positive_int(native_rollouts, "native_rollouts")
    if batch.batch is None:
        raise ValueError("capture batch has no tensor batch")
    stable_ids = batch.non_tensor_batch.get("opd_verify_stable_id")
    if not isinstance(stable_ids, np.ndarray) or stable_ids.ndim != 1:
        raise ValueError("capture batch requires one-dimensional opd_verify_stable_id")
    slots = batch.batch.get("opd_proxy_verify_rollout_slot", None)
    if not isinstance(slots, torch.Tensor) or slots.ndim != 1:
        raise ValueError("capture batch requires one-dimensional rollout slot tensor")
    if slots.shape[0] != len(batch) or stable_ids.shape[0] != len(batch):
        raise ValueError("capture key columns do not match batch size")
    if slots.dtype == torch.bool or slots.is_floating_point() or slots.is_complex():
        raise ValueError("capture rollout slot tensor must have integer dtype")

    slot_values = [int(value) for value in slots.detach().cpu().tolist()]
    if any(slot < 0 or slot >= rollout_count for slot in slot_values):
        raise ValueError(f"capture rollout slot must be in [0, {rollout_count})")
    stable_values = stable_ids.tolist()
    if any(not isinstance(value, str) or not value for value in stable_values):
        raise ValueError("capture stable IDs must be nonempty strings")
    keys = tuple(
        TrajectoryKey(stable_id, seed, slot)
        for stable_id, slot in zip(stable_values, slot_values, strict=True)
    )
    validate_exact_key_coverage(keys, keys)

    if require_complete_slots:
        _validate_exact_rollout_slots(keys, rollout_count)
    if expected_keys is not None:
        validate_exact_key_coverage(keys, expected_keys)

    existing_seed = batch.batch.get("opd_proxy_verify_engine_seed", None)
    if existing_seed is not None:
        if (
            not isinstance(existing_seed, torch.Tensor)
            or existing_seed.shape != slots.shape
            or not torch.all(existing_seed == seed)
        ):
            raise ValueError("existing capture engine-seed tensor mismatch")
    else:
        batch.batch["opd_proxy_verify_engine_seed"] = torch.full_like(
            slots, seed, dtype=torch.long
        )
    return batch, keys


def build_response_mask(
    responses: torch.Tensor, *, eos_token_id: int, pad_token_id: int
) -> torch.Tensor:
    """Return the production response mask, including the first EOS token."""
    if not isinstance(responses, torch.Tensor) or responses.ndim != 2:
        raise ValueError("responses must be a two-dimensional tensor")
    eos = _require_nonnegative_int(eos_token_id, "eos_token_id")
    pad = _require_nonnegative_int(pad_token_id, "pad_token_id")
    width = responses.shape[1]
    positions = torch.arange(width, device=responses.device).unsqueeze(0)
    eos_hits = responses.eq(eos)
    sentinel = torch.full_like(positions.expand_as(responses), width)
    first_eos = torch.where(eos_hits, positions.expand_as(responses), sentinel).min(
        dim=1
    ).values
    has_eos = first_eos.lt(width)
    through_eos = positions <= first_eos.unsqueeze(1)
    no_eos_mask = responses.ne(pad)
    mask = torch.where(has_eos.unsqueeze(1), through_eos, no_eos_mask)
    if pad != eos:
        mask = mask & (responses.ne(pad) | responses.eq(eos))
    return mask.bool()


def _validate_loss_tensor_inputs(tensors: Mapping[str, torch.Tensor]) -> tuple[int, ...]:
    shape: tuple[int, ...] | None = None
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch tensor")
        if tensor.ndim != 2:
            raise ValueError(f"{name} must be two-dimensional")
        if shape is None:
            shape = tuple(tensor.shape)
        elif tuple(tensor.shape) != shape:
            raise ValueError(f"{name} shape does not match current_log_prob")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains non-finite values")
    assert shape is not None
    return shape


def compute_authoritative_capture_tensors(
    *,
    current_log_prob: torch.Tensor,
    batch_old_log_prob: torch.Tensor,
    ref_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_is_weights: torch.Tensor | None,
    actor_config: object,
    rollout_log_prob: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Construct the exact stopped OPD objective at the frozen actor point."""
    inputs: dict[str, torch.Tensor] = {
        "current_log_prob": current_log_prob,
        "batch_old_log_prob": batch_old_log_prob,
        "ref_log_prob": ref_log_prob,
        "response_mask": response_mask,
    }
    if rollout_log_prob is not None:
        inputs["rollout_log_prob"] = rollout_log_prob
    if rollout_is_weights is not None:
        inputs["rollout_is_weights"] = rollout_is_weights
    _validate_loss_tensor_inputs(inputs)
    if not current_log_prob.requires_grad:
        raise ValueError("current_log_prob must retain gradients")
    mask = response_mask.detach()
    if not torch.all((mask == 0) | (mask == 1)) or not mask.bool().any():
        raise ValueError("response_mask must be binary with at least one valid token")

    if rollout_log_prob is not None:
        expected_weights = torch.exp(
            torch.clamp(
                batch_old_log_prob.detach() - rollout_log_prob.detach(),
                min=-20.0,
                max=20.0,
            )
        ).clamp(max=5.0)
        expected_weights = expected_weights * mask
        if rollout_is_weights is not None and not torch.allclose(
            rollout_is_weights.detach(), expected_weights, rtol=0.0, atol=1e-6
        ):
            raise ValueError("rollout IS weights mismatch authoritative token ratio")
        authoritative_weights = expected_weights.detach()
    else:
        if rollout_is_weights is None:
            raise ValueError("rollout_is_weights or rollout_log_prob is required")
        authoritative_weights = rollout_is_weights.detach()
        if not torch.all(authoritative_weights[mask == 0] == 0):
            raise ValueError("rollout IS weights must be zero on masked tokens")
        if torch.any(authoritative_weights < 0) or torch.any(authoritative_weights > 5):
            raise ValueError("rollout IS weights must be in [0, 5]")

    local_old = current_log_prob.detach()
    advantages = (ref_log_prob - local_old).detach()
    ratio = torch.exp(torch.clamp(current_log_prob - local_old, -20.0, 20.0))
    policy_loss_fn = get_policy_loss_fn("vanilla")
    policy_loss, policy_metrics = policy_loss_fn(
        old_log_prob=local_old,
        log_prob=current_log_prob,
        advantages=advantages,
        response_mask=mask,
        loss_agg_mode="token-mean",
        config=actor_config,
        rollout_is_weights=authoritative_weights,
    )
    if not isinstance(policy_loss, torch.Tensor) or policy_loss.ndim != 0:
        raise ValueError("registered vanilla policy loss must return a scalar tensor")
    metric_names = {
        "pg_clipfrac": "actor/pg_clipfrac",
        "ppo_kl": "actor/ppo_kl",
        "pg_clipfrac_lower": "actor/pg_clipfrac_lower",
    }
    missing_metrics = set(metric_names.values()) - set(policy_metrics)
    if missing_metrics:
        raise ValueError(f"registered vanilla loss omitted metrics: {sorted(missing_metrics)}")

    result: dict[str, torch.Tensor] = {
        "current_log_prob": current_log_prob,
        "batch_old_log_prob": batch_old_log_prob.detach(),
        "ref_log_prob": ref_log_prob.detach(),
        "local_old_log_prob": local_old,
        "advantages": advantages,
        "ratio": ratio,
        "response_mask": mask,
        "rollout_is_weights": authoritative_weights,
        "policy_loss": policy_loss,
    }
    if rollout_log_prob is not None:
        result["rollout_log_prob"] = rollout_log_prob.detach()
    for output_name, metric_name in metric_names.items():
        metric = torch.as_tensor(
            policy_metrics[metric_name],
            dtype=torch.float32,
            device=current_log_prob.device,
        )
        if metric.ndim != 0 or not torch.isfinite(metric):
            raise ValueError(f"invalid policy metric: {metric_name}")
        result[output_name] = metric.detach()
    return result


def recursive_parameter_sha256(module: torch.nn.Module) -> str:
    """Hash registration-ordered parameter metadata and exact tensor bytes."""
    if not isinstance(module, torch.nn.Module):
        raise TypeError("parameter hash input must be a torch module")
    digest = hashlib.sha256()
    seen: set[str] = set()
    count = 0
    for name, parameter in module.named_parameters(remove_duplicate=True):
        if name in seen:
            raise ValueError(f"duplicate parameter name: {name}")
        seen.add(name)
        if parameter.layout != torch.strided:
            raise ValueError(f"parameter {name} must use strided layout")
        value = parameter.detach().cpu().contiguous()
        metadata = {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "requires_grad": bool(parameter.requires_grad),
            "numel": value.numel(),
        }
        digest.update(canonical_json_bytes(metadata))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
        count += 1
    if count == 0:
        raise ValueError("parameter hash requires at least one parameter")
    return digest.hexdigest()


def flatten_full_parameter_gradients(
    module: torch.nn.Module,
) -> tuple[torch.Tensor, tuple[dict[str, object], ...], str]:
    """Flatten one complete unsharded native gradient in registration order."""
    if not isinstance(module, torch.nn.Module):
        raise TypeError("direct fixture requires a torch module")
    class_name = type(module).__name__.lower()
    if "fullyshard" in class_name or "fully_sharded" in class_name:
        raise ValueError("direct fixture requires a complete unsharded actor")
    records: list[dict[str, object]] = []
    gradients: list[torch.Tensor] = []
    offset = 0
    for name, parameter in module.named_parameters(remove_duplicate=True):
        if not parameter.requires_grad:
            continue
        parameter_type = type(parameter)
        if parameter_type.__name__ == "DTensor" or parameter_type.__module__.startswith(
            "torch.distributed.tensor"
        ):
            raise ValueError("direct fixture rejects local or distributed parameter shards")
        if parameter.grad is None:
            raise ValueError(f"direct fixture lacks gradient for parameter {name}")
        gradient = parameter.grad.detach()
        if gradient.layout != torch.strided or tuple(gradient.shape) != tuple(
            parameter.shape
        ):
            raise ValueError(f"direct fixture gradient layout mismatch for {name}")
        value = gradient.to(torch.float32).reshape(-1)
        if not torch.isfinite(value).all():
            raise ValueError(f"direct fixture gradient is non-finite for {name}")
        records.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "numel": parameter.numel(),
                "offset": offset,
            }
        )
        gradients.append(value)
        offset += parameter.numel()
    if not gradients:
        raise ValueError("direct fixture actor has no trainable parameter gradients")
    flattened = torch.cat(gradients).contiguous()
    from math_eval.opd_proxy_gradient_projection import (
        full_gradient_l2_norm,
        sha256_parameter_layout,
    )

    full_gradient_l2_norm(flattened)
    layout_hash = sha256_parameter_layout(records)
    return flattened, tuple(records), layout_hash


def publish_direct_gradient_fixture(
    output_root: Path,
    *,
    artifact: Mapping[str, object],
    parent_hashes: Mapping[str, object],
) -> dict[str, object]:
    """Publish the one-trajectory smoke gradient as a create-once artifact."""
    root = Path(output_root)
    parents = _normalize_hashes(parent_hashes, "parent_hashes")
    projected = artifact.get("projected_gradient")
    full_norm = artifact.get("full_gradient_norm")
    policy_loss = artifact.get("policy_loss")
    if not isinstance(projected, torch.Tensor) or projected.shape != (1024,):
        raise ValueError("direct fixture projected gradient must have shape (1024,)")
    if not isinstance(full_norm, torch.Tensor) or full_norm.numel() != 1:
        raise ValueError("direct fixture full gradient norm must be scalar")
    if not isinstance(policy_loss, torch.Tensor) or policy_loss.numel() != 1:
        raise ValueError("direct fixture policy loss must be scalar")
    tensor_payload = save_safetensors(
        {
            "projected_gradient": projected.detach().cpu().to(torch.float32).reshape(1, -1),
            "full_gradient_norm": full_norm.detach().cpu().to(torch.float32).reshape(1),
            "projected_gradient_norm": torch.linalg.vector_norm(projected)
            .detach()
            .cpu()
            .to(torch.float32)
            .reshape(1),
            "policy_loss": policy_loss.detach().cpu().to(torch.float32).reshape(1),
        }
    )
    root.mkdir(parents=True, exist_ok=True)
    tensor_path = root / "direct_gradient.safetensors"
    _write_once(tensor_path, tensor_payload)
    manifest_fields = {
        key: value
        for key, value in artifact.items()
        if key not in {"projected_gradient", "full_gradient_norm", "policy_loss"}
    }
    manifest = write_or_validate_manifest(
        root / "manifest.json",
        {
            "schema_version": 1,
            "artifact_type": "opd_proxy_direct_gradient_fixture",
            **manifest_fields,
            "full_gradient_norm": float(full_norm.detach().cpu().item()),
            "policy_loss": float(policy_loss.detach().cpu().item()),
            "tensor_file": tensor_path.name,
            "tensor_sha256": sha256_file(tensor_path),
            "parent_hashes": parents,
        },
    )
    write_or_validate_manifest(
        root / "COMPLETE.json",
        {
            "schema_version": 1,
            "artifact_type": "opd_proxy_direct_gradient_fixture_complete",
            "manifest_sha256": sha256_file(root / "manifest.json"),
            "tensor_sha256": sha256_file(tensor_path),
            "parent_hashes": parents,
        },
    )
    return manifest


def _validate_capture_tensor_shapes(
    tensors: Mapping[str, torch.Tensor],
) -> tuple[int, set[str]]:
    if not isinstance(tensors, Mapping) or not tensors:
        raise ValueError("capture tensor mapping cannot be empty")
    row_count: int | None = None
    names: set[str] = set()
    for name in sorted(tensors):
        value = tensors[name]
        if not isinstance(name, str) or not name:
            raise ValueError("capture tensor names must be nonempty strings")
        if not isinstance(value, torch.Tensor) or value.ndim == 0:
            raise ValueError(f"capture tensor {name} must have a row dimension")
        if row_count is None:
            row_count = value.shape[0]
        elif value.shape[0] != row_count:
            raise ValueError("capture tensors have inconsistent row counts")
        names.add(name)
    assert row_count is not None
    if row_count <= 0:
        raise ValueError("capture tensor chunk cannot be empty")
    return row_count, names


def _safetensors_payload(tensors: Mapping[str, torch.Tensor]) -> tuple[bytes, int, set[str]]:
    row_count, names = _validate_capture_tensor_shapes(tensors)
    normalized: dict[str, torch.Tensor] = {}
    for name in sorted(tensors):
        value = tensors[name]
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"capture tensor {name} contains non-finite values")
        normalized[name] = value.detach().cpu().contiguous()
    try:
        payload = save_safetensors(normalized)
        reloaded = load_safetensors_bytes(payload)
    except Exception as error:
        raise ValueError(f"cannot serialize capture tensors: {error}") from error
    if set(reloaded) != set(normalized):
        raise ValueError("capture safetensors readback key mismatch")
    for name, expected in normalized.items():
        actual = reloaded[name]
        if actual.dtype != expected.dtype or actual.shape != expected.shape:
            raise ValueError(f"capture safetensors readback mismatch for {name}")
    return payload, row_count, names


def _write_once(path: Path, payload: bytes) -> None:
    target = Path(path)
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise ValueError(f"existing capture artifact mismatch: {target}")
        return
    atomic_write_bytes(target, payload)


def write_tensor_chunks_atomic(
    directory: Path,
    *,
    tensors: Mapping[str, torch.Tensor],
    sidecar_rows: Sequence[Mapping[str, object]],
    parent_hashes: Mapping[str, object],
    chunk_size: int,
    start_index: int = 0,
) -> tuple[dict[str, object], ...]:
    """Publish deterministic paired safetensors/JSONL chunks create-once."""
    root = Path(directory)
    size = _require_positive_int(chunk_size, "chunk_size")
    start = _require_nonnegative_int(start_index, "start_index")
    parents = _normalize_hashes(parent_hashes, "parent_hashes")
    tensor_rows, tensor_names = _validate_capture_tensor_shapes(tensors)
    if len(sidecar_rows) != tensor_rows:
        raise ValueError("sidecar row count does not match capture tensors")
    normalized_rows: list[dict[str, object]] = []
    keys: list[TrajectoryKey] = []
    for row_index, source in enumerate(sidecar_rows):
        if not isinstance(source, Mapping):
            raise ValueError(f"sidecar row {row_index} must be a mapping")
        unknown_reserved = _RESERVED_SIDECAR_FIELDS & set(source)
        if unknown_reserved:
            raise ValueError(
                f"sidecar row uses reserved fields: {sorted(unknown_reserved)}"
            )
        row = dict(source)
        key = _trajectory_key_from_row(row)
        row["stable_id"] = key.stable_id
        row["engine_seed"] = key.engine_seed
        row["rollout_slot"] = key.rollout_slot
        if "manifest_index" in row and isinstance(row["manifest_index"], np.integer):
            row["manifest_index"] = int(row["manifest_index"])
        canonical_json_bytes(row)
        normalized_rows.append(row)
        keys.append(key)
    validate_exact_key_coverage(keys, keys)

    root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for local_start in range(0, tensor_rows, size):
        local_end = min(local_start + size, tensor_rows)
        global_start = start + local_start
        global_end = start + local_end
        chunk_tensors = {
            name: value[local_start:local_end]
            for name, value in tensors.items()
        }
        tensor_payload, chunk_count, chunk_names = _safetensors_payload(chunk_tensors)
        if chunk_names != tensor_names:
            raise AssertionError("internal capture tensor-name mismatch")
        tensor_hash = hashlib.sha256(tensor_payload).hexdigest()
        enriched_rows: list[dict[str, object]] = []
        for tensor_row, source_row in enumerate(
            normalized_rows[local_start:local_end]
        ):
            enriched_rows.append(
                {
                    **source_row,
                    "capture_index": global_start + tensor_row,
                    "tensor_row": tensor_row,
                    "chunk_start": global_start,
                    "chunk_end": global_end,
                    "tensor_sha256": tensor_hash,
                    "parent_hashes": parents,
                }
            )
        sidecar_payload = b"".join(
            canonical_json_bytes(row) for row in enriched_rows
        )
        tensor_name = f"chunk_{global_start}_{global_end}.safetensors"
        sidecar_name = f"chunk_{global_start}_{global_end}.jsonl"
        tensor_path = root / tensor_name
        sidecar_path = root / sidecar_name
        if (root / "COMPLETE.json").exists() and (
            not tensor_path.is_file() or not sidecar_path.is_file()
        ):
            raise ValueError("cannot add rows to a completed capture subtree")
        lock_path = root / f".chunk_{global_start}_{global_end}.lock"
        lock_descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
            _write_once(tensor_path, tensor_payload)
            _write_once(sidecar_path, sidecar_payload)
        finally:
            try:
                fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            finally:
                os.close(lock_descriptor)
        records.append(
            {
                "start": global_start,
                "end": global_end,
                "row_count": chunk_count,
                "tensor_file": tensor_name,
                "tensor_sha256": tensor_hash,
                "sidecar_file": sidecar_name,
                "sidecar_sha256": hashlib.sha256(sidecar_payload).hexdigest(),
                "tensor_names": sorted(tensor_names),
            }
        )
    return tuple(records)


def _read_json_object(path: Path, description: str) -> dict[str, object]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ValueError(f"invalid {description} {path}: {error}") from error
    value = _strict_json_text(text, f"{description} {path}")
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(f"blank capture sidecar line {line_number}")
                value = _strict_json_text(
                    line, f"capture sidecar {path} line {line_number}"
                )
                if not isinstance(value, dict):
                    raise ValueError("capture sidecar rows must be JSON objects")
                rows.append(value)
    except (OSError, UnicodeError) as error:
        raise ValueError(f"invalid capture sidecar {path}: {error}") from error
    return rows


def _discover_chunk_ranges(directory: Path) -> list[tuple[int, int]]:
    root = Path(directory)
    if not root.exists():
        return []
    if not root.is_dir():
        raise ValueError(f"capture subtree is not a directory: {root}")
    extensions: dict[tuple[int, int], set[str]] = defaultdict(set)
    for path in root.iterdir():
        match = _CHUNK_RE.fullmatch(path.name)
        if match is None:
            if path.name.startswith("chunk_"):
                raise ValueError(f"malformed capture chunk filename: {path.name}")
            continue
        if not path.is_file():
            raise ValueError(f"capture chunk is not a regular file: {path}")
        start, end, extension = int(match.group(1)), int(match.group(2)), match.group(3)
        if end <= start:
            raise ValueError(f"invalid capture chunk range: {path.name}")
        extensions[(start, end)].add(extension)
    for interval, found in extensions.items():
        if found != {"jsonl", "safetensors"}:
            raise ValueError(f"orphan capture chunk artifact for range {interval}")
    return sorted(extensions)


def _validate_chunk(
    directory: Path,
    *,
    start: int,
    end: int,
    expected_keys: Sequence[TrajectoryKey],
    parent_hashes: Mapping[str, str],
    required_tensor_names: set[str] | None,
) -> tuple[dict[str, torch.Tensor], list[dict[str, object]], dict[str, object]]:
    tensor_path = directory / f"chunk_{start}_{end}.safetensors"
    sidecar_path = directory / f"chunk_{start}_{end}.jsonl"
    expected_count = end - start
    tensor_hash = sha256_file(tensor_path)
    try:
        tensors = load_safetensors_file(tensor_path, device="cpu")
    except Exception as error:
        raise ValueError(f"invalid capture safetensors {tensor_path}: {error}") from error
    if not tensors:
        raise ValueError("capture safetensors cannot be empty")
    if required_tensor_names is not None and set(tensors) != required_tensor_names:
        raise ValueError("capture tensor names mismatch")
    for name, tensor in tensors.items():
        if tensor.ndim == 0 or tensor.shape[0] != expected_count:
            raise ValueError(f"capture tensor row count mismatch for {name}")
    rows = _read_jsonl(sidecar_path)
    if len(rows) != expected_count:
        raise ValueError("capture sidecar row count does not match tensor rows")
    actual_keys: list[TrajectoryKey] = []
    normalized_parents = dict(parent_hashes)
    for tensor_row, row in enumerate(rows):
        if row.get("capture_index") != start + tensor_row:
            raise ValueError("capture sidecar index is not manifest-order contiguous")
        if row.get("tensor_row") != tensor_row:
            raise ValueError("capture sidecar tensor_row mismatch")
        if row.get("chunk_start") != start or row.get("chunk_end") != end:
            raise ValueError("capture sidecar chunk range mismatch")
        if row.get("tensor_sha256") != tensor_hash:
            raise ValueError("capture sidecar tensor hash mismatch")
        if row.get("parent_hashes") != normalized_parents:
            raise ValueError("capture parent hashes mismatch")
        actual_keys.append(_trajectory_key_from_row(row))
    expected_slice = tuple(expected_keys[start:end])
    if tuple(actual_keys) != expected_slice:
        validate_exact_key_coverage(actual_keys, expected_slice)
        raise ValueError("capture chunk key order is not manifest-order contiguous")
    record = {
        "start": start,
        "end": end,
        "row_count": expected_count,
        "tensor_file": tensor_path.name,
        "tensor_sha256": tensor_hash,
        "sidecar_file": sidecar_path.name,
        "sidecar_sha256": sha256_file(sidecar_path),
        "tensor_names": sorted(tensors),
    }
    return tensors, rows, record


_ChunkConsumer = Callable[
    [dict[str, torch.Tensor], list[dict[str, object]], dict[str, object]], None
]


def _scan_capture_chunks(
    directory: Path,
    *,
    expected_keys: Sequence[TrajectoryKey],
    parent_hashes: Mapping[str, str],
    required_tensor_names: set[str] | None,
    allow_prefix: bool,
    consumer: _ChunkConsumer | None = None,
) -> tuple[int, tuple[dict[str, object], ...]]:
    expected = tuple(expected_keys)
    validate_exact_key_coverage(expected, expected)
    root = Path(directory)
    next_index = 0
    records: list[dict[str, object]] = []
    inferred_names = set(required_tensor_names) if required_tensor_names is not None else None
    for start, end in _discover_chunk_ranges(root):
        if start != next_index or end > len(expected):
            raise ValueError("capture chunks are not a contiguous manifest-order prefix")
        tensors, rows, record = _validate_chunk(
            root,
            start=start,
            end=end,
            expected_keys=expected,
            parent_hashes=parent_hashes,
            required_tensor_names=inferred_names,
        )
        if inferred_names is None:
            inferred_names = set(tensors)
        if consumer is not None:
            consumer(tensors, rows, record)
        records.append(record)
        next_index = end
    if not allow_prefix and next_index not in {0, len(expected)}:
        raise ValueError("capture subtree does not permit prefix resume")
    complete_path = root / "COMPLETE.json"
    if complete_path.exists():
        complete = _read_json_object(complete_path, "capture completion marker")
        if next_index != len(expected):
            raise ValueError("capture completion marker exists for an incomplete subtree")
        if complete.get("trajectory_count") != len(expected):
            raise ValueError("capture completion trajectory count mismatch")
        if complete.get("trajectory_keys_sha256") != trajectory_keys_sha256(expected):
            raise ValueError("capture completion key hash mismatch")
        if complete.get("parent_hashes") != dict(parent_hashes):
            raise ValueError("capture completion parent hashes mismatch")
        if complete.get("chunk_count") != len(records) or complete.get(
            "chunks"
        ) != records:
            raise ValueError("capture completion chunk records mismatch")
    return next_index, tuple(records)


def resolve_capture_resume_prefix(
    directory: Path,
    *,
    expected_keys: Sequence[TrajectoryKey],
    parent_hashes: Mapping[str, object],
    required_tensor_names: set[str] | None = None,
    allow_prefix: bool = True,
) -> int:
    """Validate chunks and return the sole legal contiguous resume offset."""
    parents = _normalize_hashes(parent_hashes, "parent_hashes")
    prefix, _ = _scan_capture_chunks(
        directory,
        expected_keys=expected_keys,
        parent_hashes=parents,
        required_tensor_names=required_tensor_names,
        allow_prefix=allow_prefix,
    )
    return prefix


def load_and_join_capture_chunks(
    directory: Path,
    *,
    expected_keys: Sequence[TrajectoryKey],
    parent_hashes: Mapping[str, object],
    required_tensor_names: set[str] | None = None,
    require_complete: bool = True,
) -> LoadedCaptureChunks:
    """Load validated chunks in declared stream order without incidental joins."""
    expected = tuple(expected_keys)
    parents = _normalize_hashes(parent_hashes, "parent_hashes")
    loaded_tensors: dict[str, list[torch.Tensor]] = defaultdict(list)
    sidecar_rows: list[dict[str, object]] = []

    def retain(
        tensors: dict[str, torch.Tensor],
        rows: list[dict[str, object]],
        record: dict[str, object],
    ) -> None:
        del record
        for name, tensor in tensors.items():
            loaded_tensors[name].append(tensor)
        sidecar_rows.extend(rows)

    prefix, records = _scan_capture_chunks(
        directory,
        expected_keys=expected,
        parent_hashes=parents,
        required_tensor_names=required_tensor_names,
        allow_prefix=True,
        consumer=retain,
    )
    if require_complete and prefix != len(expected):
        raise ValueError("capture subtree is incomplete")
    joined = {
        name: torch.cat(parts, dim=0) for name, parts in loaded_tensors.items()
    }
    actual_keys = tuple(_trajectory_key_from_row(row) for row in sidecar_rows)
    if actual_keys != expected[:prefix]:
        raise ValueError("joined capture keys are out of order")
    return LoadedCaptureChunks(
        tensors=joined,
        sidecar_rows=tuple(sidecar_rows),
        keys=actual_keys,
        chunk_records=records,
    )


def _stream_validate_complete_subtree(
    directory: Path,
    *,
    expected_keys: Sequence[TrajectoryKey],
    parent_hashes: Mapping[str, str],
) -> tuple[dict[str, object], ...]:
    prefix, records = _scan_capture_chunks(
        directory,
        expected_keys=expected_keys,
        parent_hashes=parent_hashes,
        required_tensor_names=None,
        allow_prefix=False,
    )
    if prefix != len(expected_keys):
        raise ValueError("capture subtree is incomplete")
    return records


def _complete_subtree(
    directory: Path,
    *,
    expected_keys: Sequence[TrajectoryKey],
    parent_hashes: Mapping[str, str],
    validated_records: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    records = (
        tuple(dict(record) for record in validated_records)
        if validated_records is not None
        else _stream_validate_complete_subtree(
            directory, expected_keys=expected_keys, parent_hashes=parent_hashes
        )
    )
    marker: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "opd_proxy_capture_subtree_complete",
        "trajectory_count": len(expected_keys),
        "trajectory_keys_sha256": trajectory_keys_sha256(expected_keys),
        "parent_hashes": dict(parent_hashes),
        "chunk_count": len(records),
        "chunks": list(records),
    }
    return write_or_validate_manifest(Path(directory) / "COMPLETE.json", marker)


def _validate_hash_group(seed_manifest: Mapping[str, object], field: str) -> dict[str, str]:
    value = seed_manifest.get(field)
    if not isinstance(value, Mapping):
        raise ValueError(f"seed manifest {field} must be a hash mapping")
    return _normalize_hashes(value, f"seed_manifest.{field}")


def finalize_capture_seed(
    seed_root: Path,
    *,
    native_rollouts: int,
    expected_keys: Sequence[TrajectoryKey],
    actor_rank_expected_keys: Mapping[int, Sequence[TrajectoryKey]],
    parent_hashes: Mapping[str, object],
    actor_parameter_sha256_before: str,
    actor_parameter_sha256_after: str,
    seed_manifest: Mapping[str, object],
) -> dict[str, object]:
    """Validate global coverage and publish immutable COMPLETE markers."""
    root = Path(seed_root)
    rollout_count = _require_positive_int(native_rollouts, "native_rollouts")
    expected = tuple(expected_keys)
    validate_exact_key_coverage(expected, expected)
    if not expected:
        raise ValueError("capture seed requires at least one trajectory")
    _validate_exact_rollout_slots(expected, rollout_count)
    parents = _normalize_hashes(parent_hashes, "parent_hashes")
    before = _require_sha256(
        actor_parameter_sha256_before, "actor_parameter_sha256_before"
    )
    after = _require_sha256(
        actor_parameter_sha256_after, "actor_parameter_sha256_after"
    )
    if before != after:
        raise ValueError("actor parameter hash changed during capture")
    if not isinstance(seed_manifest, Mapping):
        raise ValueError("seed_manifest must be a mapping")
    canonical_json_bytes(dict(seed_manifest))
    missing = _REQUIRED_SEED_MANIFEST_FIELDS - set(seed_manifest)
    if missing:
        raise ValueError(f"seed manifest missing fields: {sorted(missing)}")
    vllm_version = seed_manifest.get("vllm_version")
    if not isinstance(vllm_version, str) or not vllm_version:
        raise ValueError("seed manifest requires resolved vLLM version")
    engine_args = seed_manifest.get("engine_args")
    sampling_args = seed_manifest.get("sampling_args")
    if not isinstance(engine_args, Mapping) or not isinstance(sampling_args, Mapping):
        raise ValueError("seed manifest engine/sampling args must be mappings")
    canonical_json_bytes(dict(engine_args))
    canonical_json_bytes(dict(sampling_args))
    if seed_manifest.get("engine_count") != 1:
        raise ValueError("capture seed must use exactly one engine")
    if seed_manifest.get("generation_call_count") != 1:
        raise ValueError("capture seed must use exactly one generation call")

    engine_seeds = {key.engine_seed for key in expected}
    if len(engine_seeds) != 1 or engine_args.get("seed") != next(iter(engine_seeds)):
        raise ValueError("seed manifest engine seed mismatch")
    if sampling_args.get("n") != rollout_count:
        raise ValueError(
            "seed manifest sampling n must equal the native rollout count"
        )
    prompt_ids = list(dict.fromkeys(key.stable_id for key in expected))
    if seed_manifest.get("ordered_prompt_keys_sha256") != sha256_id_lines(prompt_ids):
        raise ValueError("seed manifest ordered prompt-key hash mismatch")
    if seed_manifest.get("returned_compound_keys_sha256") != trajectory_keys_sha256(
        expected
    ):
        raise ValueError("seed manifest returned compound-key hash mismatch")
    for hash_field in (
        "model_hashes",
        "tokenizer_hashes",
        "config_hashes",
        "source_hashes",
    ):
        _validate_hash_group(seed_manifest, hash_field)

    if not isinstance(actor_rank_expected_keys, Mapping) or not actor_rank_expected_keys:
        raise ValueError("capture seed requires actor rank key partitions")
    actor_keys: list[TrajectoryKey] = []
    normalized_rank_keys: dict[int, tuple[TrajectoryKey, ...]] = {}
    for rank in sorted(actor_rank_expected_keys):
        parsed_rank = _require_nonnegative_int(rank, "actor rank")
        local = tuple(actor_rank_expected_keys[rank])
        validate_exact_key_coverage(local, local)
        if not local:
            raise ValueError(f"actor rank {rank} has no expected keys")
        normalized_rank_keys[parsed_rank] = local
        actor_keys.extend(local)
    validate_exact_key_coverage(actor_keys, expected)

    rollout_dir = root / "rollout"
    rollout_staging = root / ".rollout.tmp"
    if rollout_dir.exists() and rollout_staging.exists():
        raise ValueError("both complete and staged rollout subtrees exist")
    active_rollout = rollout_dir if rollout_dir.exists() else rollout_staging
    if not active_rollout.exists():
        raise ValueError("capture rollout subtree is missing")

    # Validate every subtree once with bounded chunk memory before any marker.
    rollout_records = _stream_validate_complete_subtree(
        active_rollout, expected_keys=expected, parent_hashes=parents
    )
    trainer_records = _stream_validate_complete_subtree(
        root / "trainer_boundary", expected_keys=expected, parent_hashes=parents
    )
    rank_records: dict[int, tuple[dict[str, object], ...]] = {}
    for rank, local in normalized_rank_keys.items():
        rank_records[rank] = _stream_validate_complete_subtree(
            root / f"actor/rank_{rank}",
            expected_keys=local,
            parent_hashes=parents,
        )

    rollout_complete = _complete_subtree(
        active_rollout,
        expected_keys=expected,
        parent_hashes=parents,
        validated_records=rollout_records,
    )
    if active_rollout == rollout_staging:
        os.replace(rollout_staging, rollout_dir)
        directory_descriptor = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    trainer_complete = _complete_subtree(
        root / "trainer_boundary",
        expected_keys=expected,
        parent_hashes=parents,
        validated_records=trainer_records,
    )
    rank_complete: dict[str, object] = {}
    for rank, local in normalized_rank_keys.items():
        marker = _complete_subtree(
            root / f"actor/rank_{rank}",
            expected_keys=local,
            parent_hashes=parents,
            validated_records=rank_records[rank],
        )
        marker_path = root / f"actor/rank_{rank}/COMPLETE.json"
        rank_complete[str(rank)] = {
            "trajectory_count": len(local),
            "trajectory_keys_sha256": trajectory_keys_sha256(local),
            "complete_sha256": sha256_file(marker_path),
            "complete": marker,
        }
    actor_complete = write_or_validate_manifest(
        root / "actor/COMPLETE.json",
        {
            "schema_version": 1,
            "artifact_type": "opd_proxy_capture_actor_complete",
            "trajectory_count": len(expected),
            "trajectory_keys_sha256": trajectory_keys_sha256(expected),
            "parent_hashes": parents,
            "ranks": rank_complete,
        },
    )

    manifest: dict[str, object] = {
        **dict(seed_manifest),
        "schema_version": 1,
        "artifact_type": "opd_proxy_capture_seed",
        "expected_trajectory_count": len(expected),
        "expected_trajectory_keys_sha256": trajectory_keys_sha256(expected),
        "parent_hashes": parents,
        "actor_parameter_sha256_before": before,
        "actor_parameter_sha256_after": after,
        "subtree_complete_sha256": {
            "rollout": sha256_file(rollout_dir / "COMPLETE.json"),
            "trainer_boundary": sha256_file(
                root / "trainer_boundary/COMPLETE.json"
            ),
            "actor": sha256_file(root / "actor/COMPLETE.json"),
        },
        "subtree_complete": {
            "rollout": rollout_complete,
            "trainer_boundary": trainer_complete,
            "actor": actor_complete,
        },
    }
    normalized_manifest = write_or_validate_manifest(root / "manifest.json", manifest)
    write_or_validate_manifest(
        root / "COMPLETE.json",
        {
            "schema_version": 1,
            "artifact_type": "opd_proxy_capture_seed_complete",
            "manifest_sha256": sha256_file(root / "manifest.json"),
            "trajectory_count": len(expected),
            "trajectory_keys_sha256": trajectory_keys_sha256(expected),
            "parent_hashes": parents,
        },
    )
    return normalized_manifest


__all__ = [
    "LoadedCaptureChunks",
    "attach_and_validate_keys",
    "build_response_mask",
    "compute_authoritative_capture_tensors",
    "finalize_capture_seed",
    "flatten_full_parameter_gradients",
    "load_and_join_capture_chunks",
    "publish_direct_gradient_fixture",
    "recursive_parameter_sha256",
    "resolve_capture_resume_prefix",
    "trajectory_keys_sha256",
    "validate_capture_contract",
    "write_tensor_chunks_atomic",
]
