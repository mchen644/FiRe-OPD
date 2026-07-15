"""Replay frozen OPD trajectories and publish exact projected gradients."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors

from math_eval.deepmath_gradient_diversity import official_shard_bounds
from math_eval.opd_proxy_gradient_projection import (
    PROJECTION_LINEARITY_ATOL,
    PROJECTION_LINEARITY_RTOL,
    ProjectionConfig,
    build_projection_manifest,
    construct_cuda_projector,
    full_gradient_l2_norm,
    project_full_gradient,
    sha256_parameter_layout,
    verify_prismatic_reference,
)
from math_eval.opd_proxy_gradient_verify_artifacts import (
    TrajectoryKey,
    VectorSet,
    atomic_save_safetensors,
    atomic_write_bytes,
    build_runtime_metadata,
    canonical_json_bytes,
    load_vector_set,
    recursive_file_manifest,
    repository_state,
    sha256_file,
    sha256_id_lines,
    sha256_ordered_id_lines,
    write_or_validate_manifest,
)
from verl.trainer.ppo.opd_proxy_verify_capture import (
    compute_authoritative_capture_tensors,
    load_and_join_capture_chunks,
    trajectory_keys_sha256,
)


_CAPTURE_RTOL = 5e-3
_CAPTURE_ATOL = 5e-3
_VECTOR_CHUNK_RE = re.compile(
    r"vectors_(0|[1-9][0-9]*)_(0|[1-9][0-9]*)\.(jsonl|safetensors)"
)
_CAPTURE_CHUNK_RE = re.compile(
    r"chunk_(0|[1-9][0-9]*)_(0|[1-9][0-9]*)\.(jsonl|safetensors)"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ParameterLayoutEntry:
    name: str
    shape: tuple[int, ...]
    numel: int
    offset: int


@dataclass(frozen=True)
class ParameterLayout:
    entries: tuple[ParameterLayoutEntry, ...]
    total_numel: int
    sha256: str


@dataclass(frozen=True)
class ReplayTrajectory:
    stable_id: str
    split: str
    engine_seed: int
    rollout_slot: int
    input_ids: torch.Tensor
    responses: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    response_mask: torch.Tensor
    rollout_log_prob: torch.Tensor
    batch_old_log_prob: torch.Tensor
    ref_log_prob: torch.Tensor
    rollout_is_weights: torch.Tensor
    captured_current_log_prob: torch.Tensor | None = None
    captured_local_old_log_prob: torch.Tensor | None = None
    captured_advantages: torch.Tensor | None = None
    captured_ratio: torch.Tensor | None = None
    captured_policy_loss: torch.Tensor | None = None
    verifier_correct_count: int | None = None
    verifier_total: int | None = None
    source_capture_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.stable_id, str) or not self.stable_id:
            raise ValueError("replay trajectory stable ID must be nonempty")
        if not isinstance(self.split, str) or not self.split:
            raise ValueError("replay trajectory split must be nonempty")
        for field_name in ("engine_seed", "rollout_slot"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"replay trajectory {field_name} must be nonnegative")
        if _SHA256_RE.fullmatch(self.source_capture_sha256) is None:
            raise ValueError("replay trajectory source capture SHA-256 is invalid")
        response_shape = tuple(self.responses.shape)
        if len(response_shape) != 2 or response_shape[0] != 1:
            raise ValueError("replay trajectory responses must have shape (1, length)")
        response_fields = (
            "response_mask",
            "rollout_log_prob",
            "batch_old_log_prob",
            "ref_log_prob",
            "rollout_is_weights",
        )
        optional_response_fields = (
            "captured_current_log_prob",
            "captured_local_old_log_prob",
            "captured_advantages",
            "captured_ratio",
        )
        for field_name in response_fields + optional_response_fields:
            value = getattr(self, field_name)
            if value is None and field_name in optional_response_fields:
                continue
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != response_shape:
                raise ValueError(
                    f"replay trajectory {field_name} must match response shape"
                )
            if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"replay trajectory {field_name} must be finite")
        if self.response_mask.dtype != torch.bool:
            raise ValueError("replay response mask must be bool")
        if not bool(self.response_mask.any().item()):
            raise ValueError("replay response mask must contain a valid token")
        if self.input_ids.ndim != 2 or self.input_ids.shape[0] != 1:
            raise ValueError("replay input_ids must have one batch row")
        for field_name in ("attention_mask", "position_ids"):
            value = getattr(self, field_name)
            if tuple(value.shape) != tuple(self.input_ids.shape):
                raise ValueError(f"replay {field_name} must match input_ids")
        if self.captured_policy_loss is not None:
            if not isinstance(self.captured_policy_loss, torch.Tensor) or self.captured_policy_loss.numel() != 1:
                raise ValueError("captured policy loss must be scalar")
        if (self.verifier_correct_count is None) != (self.verifier_total is None):
            raise ValueError("verifier correct and total must be present together")
        if self.verifier_total is not None:
            if (
                isinstance(self.verifier_total, bool)
                or not isinstance(self.verifier_total, int)
                or self.verifier_total <= 0
                or isinstance(self.verifier_correct_count, bool)
                or not isinstance(self.verifier_correct_count, int)
                or self.verifier_correct_count < 0
                or self.verifier_correct_count > self.verifier_total
            ):
                raise ValueError("invalid verifier correct/total values")

    def to(self, device: torch.device | str) -> ReplayTrajectory:
        replacements: dict[str, object] = {}
        for field_name in (
            "input_ids",
            "responses",
            "attention_mask",
            "position_ids",
            "response_mask",
            "rollout_log_prob",
            "batch_old_log_prob",
            "ref_log_prob",
            "rollout_is_weights",
            "captured_current_log_prob",
            "captured_local_old_log_prob",
            "captured_advantages",
            "captured_ratio",
            "captured_policy_loss",
        ):
            value = getattr(self, field_name)
            if isinstance(value, torch.Tensor):
                replacements[field_name] = value.to(device)
        return replace(self, **replacements)


@dataclass(frozen=True)
class ReplayLoss:
    loss: torch.Tensor
    current_log_prob: torch.Tensor
    local_old_log_prob: torch.Tensor
    advantages: torch.Tensor
    ratio: torch.Tensor
    response_mask: torch.Tensor
    rollout_is_weights: torch.Tensor


@dataclass(frozen=True)
class ReplayVector:
    vector_id: str
    stable_id: str
    split: str
    representation: str
    engine_seed: int
    rollout_slot: int | None
    aggregation: str
    source_capture_sha256: str
    projected_gradient: torch.Tensor
    full_gradient_norm: torch.Tensor
    projected_gradient_norm: torch.Tensor
    valid_token_count: torch.Tensor
    response_length: torch.Tensor
    sampled_reverse_kl: torch.Tensor
    opd_signal_rms: torch.Tensor
    verifier_correct_count: torch.Tensor | None = None
    verifier_total: torch.Tensor | None = None
    prompt_token_count: int | None = None
    completion_token_count: int | None = None
    supervised_label_count: int | None = None
    full_token_count: int | None = None


@dataclass(frozen=True)
class ProxyGroupReplay:
    individual: tuple[ReplayVector, ...]
    group: ReplayVector

    @property
    def projected_gradient(self) -> torch.Tensor:
        return self.group.projected_gradient

    @property
    def full_gradient_norm(self) -> torch.Tensor:
        return self.group.full_gradient_norm


@dataclass(frozen=True)
class _TrajectoryReplay:
    loss: ReplayLoss
    full_gradient: torch.Tensor
    vector: ReplayVector


def _read_canonical_json(path: Path, description: str) -> dict[str, object]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {description} {path}: {error}") from error
    if not isinstance(value, dict) or payload != canonical_json_bytes(value):
        raise ValueError(f"{description} must be a canonical JSON object")
    return value


def _capture_rank_keys(rank_directory: Path) -> tuple[TrajectoryKey, ...]:
    grouped: dict[tuple[int, int], dict[str, Path]] = {}
    for path in rank_directory.iterdir():
        match = _CAPTURE_CHUNK_RE.fullmatch(path.name)
        if match is None:
            if path.name.startswith("chunk_"):
                raise ValueError(f"malformed actor capture chunk: {path.name}")
            continue
        start, end, extension = int(match.group(1)), int(match.group(2)), match.group(3)
        grouped.setdefault((start, end), {})[extension] = path
    keys: list[TrajectoryKey] = []
    cursor = 0
    for (start, end), files in sorted(grouped.items()):
        if start != cursor or end <= start or set(files) != {"jsonl", "safetensors"}:
            raise ValueError("actor capture chunks are not a paired contiguous prefix")
        with files["jsonl"].open("rb") as handle:
            rows = []
            for line in handle:
                value = json.loads(line)
                if not isinstance(value, dict) or line != canonical_json_bytes(value):
                    raise ValueError("actor capture sidecar is not canonical JSON")
                rows.append(value)
        if len(rows) != end - start:
            raise ValueError("actor capture sidecar row count mismatch")
        for row in rows:
            stable_id = row.get("stable_id")
            engine_seed = row.get("engine_seed")
            rollout_slot = row.get("rollout_slot")
            if not isinstance(stable_id, str) or not stable_id:
                raise ValueError("actor capture sidecar stable ID is invalid")
            for field_name, value in (
                ("engine_seed", engine_seed),
                ("rollout_slot", rollout_slot),
            ):
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(
                        f"actor capture sidecar {field_name} is invalid"
                    )
            keys.append(TrajectoryKey(stable_id, engine_seed, rollout_slot))
        cursor = end
    if not keys:
        raise ValueError("actor capture rank has no trajectory chunks")
    return tuple(keys)


def load_capture_trajectories(capture_root: Path) -> tuple[ReplayTrajectory, ...]:
    """Load an immutable completed actor capture and join every row by compound key."""
    root = Path(capture_root)
    manifest_path = root / "manifest.json"
    complete_path = root / "COMPLETE.json"
    manifest = _read_canonical_json(manifest_path, "capture seed manifest")
    complete = _read_canonical_json(complete_path, "capture seed completion marker")
    if manifest.get("artifact_type") != "opd_proxy_capture_seed":
        raise ValueError("unsupported capture seed manifest")
    if complete.get("artifact_type") != "opd_proxy_capture_seed_complete":
        raise ValueError("unsupported capture seed completion marker")
    if complete.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("capture seed completion manifest hash mismatch")
    parent_hashes = manifest.get("parent_hashes")
    if not isinstance(parent_hashes, dict) or parent_hashes != complete.get(
        "parent_hashes"
    ):
        raise ValueError("capture seed parent hashes mismatch")
    expected_count = manifest.get("expected_trajectory_count")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count <= 0
        or complete.get("trajectory_count") != expected_count
    ):
        raise ValueError("capture seed trajectory count mismatch")
    expected_key_hash = manifest.get("expected_trajectory_keys_sha256")
    if expected_key_hash != complete.get("trajectory_keys_sha256"):
        raise ValueError("capture seed trajectory key hash mismatch")

    actor_complete_path = root / "actor" / "COMPLETE.json"
    actor_complete = _read_canonical_json(
        actor_complete_path, "capture actor completion marker"
    )
    if actor_complete.get("artifact_type") != "opd_proxy_capture_actor_complete":
        raise ValueError("unsupported capture actor completion marker")
    subtree_hashes = manifest.get("subtree_complete_sha256")
    if not isinstance(subtree_hashes, dict) or subtree_hashes.get("actor") != sha256_file(
        actor_complete_path
    ):
        raise ValueError("capture actor completion hash mismatch")
    if actor_complete.get("parent_hashes") != parent_hashes:
        raise ValueError("capture actor parent hashes mismatch")
    ranks = actor_complete.get("ranks")
    if not isinstance(ranks, dict) or not ranks:
        raise ValueError("capture actor completion lacks rank partitions")

    required_tensors = {
        "input_ids",
        "responses",
        "attention_mask",
        "position_ids",
        "response_mask",
        "rollout_log_prob",
        "batch_old_log_prob",
        "ref_log_prob",
        "rollout_is_weights",
        "current_log_prob",
        "local_old_log_prob",
        "advantages",
        "ratio",
        "policy_loss",
    }
    source_capture_sha256 = sha256_file(manifest_path)
    all_keys: list[TrajectoryKey] = []
    trajectories: list[ReplayTrajectory] = []
    for rank_name in sorted(ranks, key=lambda value: int(value)):
        if not rank_name.isdigit() or str(int(rank_name)) != rank_name:
            raise ValueError("capture actor rank names must be canonical integers")
        rank = int(rank_name)
        rank_info = ranks[rank_name]
        if not isinstance(rank_info, dict):
            raise ValueError("capture actor rank metadata must be an object")
        rank_directory = root / "actor" / f"rank_{rank}"
        rank_keys = _capture_rank_keys(rank_directory)
        if rank_info.get("trajectory_count") != len(rank_keys) or rank_info.get(
            "trajectory_keys_sha256"
        ) != trajectory_keys_sha256(rank_keys):
            raise ValueError("capture actor rank key coverage mismatch")
        rank_complete_path = rank_directory / "COMPLETE.json"
        if rank_info.get("complete_sha256") != sha256_file(rank_complete_path):
            raise ValueError("capture actor rank completion hash mismatch")
        loaded = load_and_join_capture_chunks(
            rank_directory,
            expected_keys=rank_keys,
            parent_hashes=parent_hashes,
        )
        missing = required_tensors - set(loaded.tensors)
        if missing:
            raise ValueError(f"capture actor rank lacks replay tensors: {sorted(missing)}")
        for index, (key, sidecar) in enumerate(
            zip(rank_keys, loaded.sidecar_rows, strict=True)
        ):
            if sidecar.get("actor_rank") != rank:
                raise ValueError("capture actor sidecar rank mismatch")
            split = sidecar.get("split")
            if not isinstance(split, str) or not split:
                raise ValueError("capture actor sidecar split is invalid")
            trajectories.append(
                ReplayTrajectory(
                    stable_id=key.stable_id,
                    split=split,
                    engine_seed=key.engine_seed,
                    rollout_slot=key.rollout_slot,
                    input_ids=loaded.tensors["input_ids"][index : index + 1],
                    responses=loaded.tensors["responses"][index : index + 1],
                    attention_mask=loaded.tensors["attention_mask"][index : index + 1],
                    position_ids=loaded.tensors["position_ids"][index : index + 1],
                    response_mask=loaded.tensors["response_mask"][index : index + 1].bool(),
                    rollout_log_prob=loaded.tensors["rollout_log_prob"][index : index + 1],
                    batch_old_log_prob=loaded.tensors["batch_old_log_prob"][index : index + 1],
                    ref_log_prob=loaded.tensors["ref_log_prob"][index : index + 1],
                    rollout_is_weights=loaded.tensors["rollout_is_weights"][index : index + 1],
                    captured_current_log_prob=loaded.tensors["current_log_prob"][index : index + 1],
                    captured_local_old_log_prob=loaded.tensors["local_old_log_prob"][index : index + 1],
                    captured_advantages=loaded.tensors["advantages"][index : index + 1],
                    captured_ratio=loaded.tensors["ratio"][index : index + 1],
                    captured_policy_loss=loaded.tensors["policy_loss"][index : index + 1],
                    source_capture_sha256=source_capture_sha256,
                )
            )
        all_keys.extend(rank_keys)
    if len(all_keys) != expected_count or trajectory_keys_sha256(all_keys) != expected_key_hash:
        raise ValueError("capture actor global trajectory coverage mismatch")
    return tuple(trajectories)


def _is_local_shard_model(model: torch.nn.Module) -> bool:
    if getattr(model, "_opd_fsdp_local_shard", False):
        return True
    class_name = type(model).__name__.lower()
    if "fullyshard" in class_name or "fully_sharded" in class_name:
        return True
    for parameter in model.parameters():
        parameter_type = type(parameter)
        if parameter_type.__name__ == "DTensor" or parameter_type.__module__.startswith(
            "torch.distributed.tensor"
        ):
            return True
    return False


def build_parameter_layout(model: torch.nn.Module) -> ParameterLayout:
    """Freeze trainable full-parameter registration order for flattening."""
    if not isinstance(model, torch.nn.Module):
        raise TypeError("parameter layout requires a torch module")
    if _is_local_shard_model(model):
        raise ValueError("replay requires a full unsharded model, not an FSDP local shard")
    entries: list[ParameterLayoutEntry] = []
    offset = 0
    for name, parameter in model.named_parameters(remove_duplicate=True):
        if not parameter.requires_grad:
            continue
        if not isinstance(name, str) or not name:
            raise ValueError("trainable parameters require nonempty names")
        if parameter.layout != torch.strided or parameter.numel() <= 0:
            raise ValueError(f"parameter {name} has unsupported layout or size")
        entry = ParameterLayoutEntry(
            name=name,
            shape=tuple(parameter.shape),
            numel=parameter.numel(),
            offset=offset,
        )
        entries.append(entry)
        offset += entry.numel
    if not entries:
        raise ValueError("replay actor has no trainable parameters")
    return ParameterLayout(
        entries=tuple(entries),
        total_numel=offset,
        sha256=sha256_parameter_layout(entries),
    )


def flatten_float32_gradients(
    model: torch.nn.Module, layout: ParameterLayout | None = None
) -> torch.Tensor:
    """Flatten complete native gradients in frozen registration order."""
    expected = build_parameter_layout(model) if layout is None else layout
    parameters = dict(model.named_parameters(remove_duplicate=True))
    flattened: list[torch.Tensor] = []
    for entry in expected.entries:
        parameter = parameters.get(entry.name)
        if parameter is None or tuple(parameter.shape) != entry.shape:
            raise ValueError(f"parameter layout mismatch for {entry.name}")
        if parameter.grad is None:
            raise ValueError(f"missing full gradient for parameter {entry.name}")
        gradient = parameter.grad.detach()
        if gradient.layout != torch.strided or tuple(gradient.shape) != entry.shape:
            raise ValueError(f"invalid gradient layout for parameter {entry.name}")
        value = gradient.to(torch.float32).reshape(-1)
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"non-finite gradient for parameter {entry.name}")
        flattened.append(value)
    result = torch.cat(flattened).contiguous()
    if result.numel() != expected.total_numel:
        raise AssertionError("flattened gradient dimension mismatch")
    full_gradient_l2_norm(result)
    return result


def _accumulate_parameter_gradients(
    model: torch.nn.Module,
    layout: ParameterLayout,
    accumulator: torch.Tensor,
    *,
    scale: float,
) -> None:
    parameters = dict(model.named_parameters(remove_duplicate=True))
    for entry in layout.entries:
        parameter = parameters.get(entry.name)
        if parameter is None or parameter.grad is None:
            raise ValueError(f"missing full gradient for parameter {entry.name}")
        gradient = parameter.grad.detach()
        if tuple(gradient.shape) != entry.shape or gradient.layout != torch.strided:
            raise ValueError(f"invalid gradient layout for parameter {entry.name}")
        value = gradient.to(torch.float32).reshape(-1)
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"non-finite gradient for parameter {entry.name}")
        accumulator[entry.offset : entry.offset + entry.numel].add_(
            value, alpha=scale
        )


def _actor_module(actor) -> torch.nn.Module:
    module = getattr(actor, "actor_module", actor)
    if not isinstance(module, torch.nn.Module):
        raise TypeError("replay actor must expose actor_module")
    return module


def _actor_device(actor) -> torch.device:
    return next(_actor_module(actor).parameters()).device


def _forward_log_prob(
    actor, trajectory: ReplayTrajectory, *, require_grad: bool = True
) -> torch.Tensor:
    local = trajectory.to(_actor_device(actor))
    custom = getattr(actor, "replay_forward_log_prob", None)
    if callable(custom):
        current = custom(local)
    else:
        forward = getattr(actor, "_forward_micro_batch", None)
        if not callable(forward):
            raise TypeError("replay actor lacks _forward_micro_batch")
        inputs = {
            "input_ids": local.input_ids,
            "responses": local.responses,
            "attention_mask": local.attention_mask,
            "position_ids": local.position_ids,
        }
        _, current, _ = forward(
            inputs,
            temperature=1.0,
            calculate_entropy=False,
        )
    if not isinstance(current, torch.Tensor) or tuple(current.shape) != tuple(
        local.responses.shape
    ):
        raise ValueError("replay actor returned invalid action log probabilities")
    if require_grad and not current.requires_grad:
        raise ValueError("replay current log probabilities must retain gradients")
    if not bool(torch.isfinite(current).all().item()):
        raise ValueError("replay current log probabilities must be finite")
    return current


def _assert_valid_tokens_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    mask: torch.Tensor,
    description: str,
) -> None:
    expected = expected.to(actual.device)
    mask = mask.to(actual.device).bool()
    if not torch.allclose(
        actual[mask], expected[mask], rtol=_CAPTURE_RTOL, atol=_CAPTURE_ATOL
    ):
        raise ValueError(f"recomputed {description} differs from captured values")


def validate_replay_trajectory_integrity(trajectory: ReplayTrajectory) -> None:
    """Check immutable rollout-correction fields without using them as local old."""
    mask = trajectory.response_mask
    expected_weights = torch.exp(
        torch.clamp(
            trajectory.batch_old_log_prob - trajectory.rollout_log_prob,
            min=-20.0,
            max=20.0,
        )
    ).clamp(max=5.0)
    expected_weights = expected_weights * mask
    if not torch.allclose(
        trajectory.rollout_is_weights,
        expected_weights,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("captured rollout correction weights fail integrity check")
    if torch.any(trajectory.rollout_is_weights[~mask] != 0):
        raise ValueError("rollout correction weights must be zero outside response mask")


def verify_captured_actor_log_prob(actor, trajectory: ReplayTrajectory) -> None:
    """Recompute the captured batch-old role in eval/no-grad mode."""
    module = _actor_module(actor)
    module.eval()
    with torch.no_grad():
        recomputed = _forward_log_prob(actor, trajectory, require_grad=False)
    local = trajectory.to(recomputed.device)
    _assert_valid_tokens_close(
        recomputed,
        local.batch_old_log_prob,
        local.response_mask,
        "batch_old_log_prob",
    )
    module.train()


def compute_replay_loss(actor, trajectory: ReplayTrajectory) -> ReplayLoss:
    """Reconstruct the canonical unscaled stopped reverse-KL actor loss."""
    module = _actor_module(actor)
    module.train()
    current = _forward_log_prob(actor, trajectory)
    local = trajectory.to(current.device)
    if local.captured_current_log_prob is not None:
        _assert_valid_tokens_close(
            current,
            local.captured_current_log_prob,
            local.response_mask,
            "current_log_prob",
        )
    if local.captured_local_old_log_prob is not None:
        _assert_valid_tokens_close(
            current.detach(),
            local.captured_local_old_log_prob,
            local.response_mask,
            "local_old_log_prob",
        )
    actor_config = getattr(actor, "config", None)
    if actor_config is None:
        raise TypeError("replay actor must expose its frozen actor config")
    authoritative = compute_authoritative_capture_tensors(
        current_log_prob=current,
        # Captured batch old is deliberately not the actor-local PPO anchor.
        batch_old_log_prob=current.detach(),
        ref_log_prob=local.ref_log_prob,
        response_mask=local.response_mask,
        rollout_is_weights=local.rollout_is_weights,
        rollout_log_prob=None,
        actor_config=actor_config,
    )
    for field_name, captured in (
        ("advantages", local.captured_advantages),
        ("ratio", local.captured_ratio),
    ):
        if captured is not None:
            _assert_valid_tokens_close(
                authoritative[field_name],
                captured,
                local.response_mask,
                field_name,
            )
    if local.captured_policy_loss is not None and not torch.allclose(
        authoritative["policy_loss"],
        local.captured_policy_loss.reshape(()),
        rtol=_CAPTURE_RTOL,
        atol=_CAPTURE_ATOL,
    ):
        raise ValueError("recomputed policy loss differs from captured value")
    return ReplayLoss(
        loss=authoritative["policy_loss"],
        current_log_prob=current,
        local_old_log_prob=authoritative["local_old_log_prob"],
        advantages=authoritative["advantages"],
        ratio=authoritative["ratio"],
        response_mask=authoritative["response_mask"],
        rollout_is_weights=authoritative["rollout_is_weights"],
    )


def _zero_grad(actor) -> None:
    _actor_module(actor).zero_grad(set_to_none=True)


def _validate_group(
    trajectories: Sequence[ReplayTrajectory],
) -> tuple[ReplayTrajectory, ...]:
    if not isinstance(trajectories, Sequence) or len(trajectories) != 4:
        raise ValueError("replay group requires exactly rollout slots 0,1,2,3")
    ordered = tuple(sorted(trajectories, key=lambda row: row.rollout_slot))
    if tuple(row.rollout_slot for row in ordered) != (0, 1, 2, 3):
        raise ValueError("replay group must contain exact unique rollout slots 0,1,2,3")
    stable_ids = {row.stable_id for row in ordered}
    if len(stable_ids) != 1:
        raise ValueError("replay group stable IDs do not agree")
    if len({row.engine_seed for row in ordered}) != 1:
        raise ValueError("replay group engine seeds do not agree")
    if len({row.split for row in ordered}) != 1:
        raise ValueError("replay group splits do not agree")
    if len({row.source_capture_sha256 for row in ordered}) != 1:
        raise ValueError("replay group source capture hashes do not agree")
    for row in ordered:
        validate_replay_trajectory_integrity(row)
    return ordered


def _trajectory_diagnostics(
    loss: ReplayLoss, trajectory: ReplayTrajectory
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    local = trajectory.to(loss.current_log_prob.device)
    mask = local.response_mask.bool()
    reverse_kl = (
        loss.local_old_log_prob[mask] - local.ref_log_prob[mask]
    ).mean()
    opd_rms = torch.sqrt(
        torch.square(local.ref_log_prob[mask] - loss.local_old_log_prob[mask]).mean()
    )
    valid_count = mask.sum().to(torch.float32)
    response_length = valid_count.clone()
    return (
        reverse_kl.detach().cpu().to(torch.float32),
        opd_rms.detach().cpu().to(torch.float32),
        valid_count.detach().cpu(),
        response_length.detach().cpu(),
    )


def proxy_vector_id(stable_id: str, engine_seed: int, rollout_slot: int) -> str:
    return f"P:{stable_id}:seed={engine_seed}:slot={rollout_slot}:n1"


def proxy_group_vector_id(stable_id: str, engine_seed: int) -> str:
    return f"P:{stable_id}:seed={engine_seed}:n4"


def target_group_vector_id(stable_id: str, engine_seed: int) -> str:
    return f"T:{stable_id}:seed={engine_seed}:n4"


def _make_vector(
    *,
    vector_id: str,
    representation: str,
    aggregation: str,
    trajectory: ReplayTrajectory,
    projected_gradient: torch.Tensor,
    full_gradient_norm: torch.Tensor,
    diagnostics: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    rollout_slot: int | None,
    verifier_correct_count: int | None,
    verifier_total: int | None,
) -> ReplayVector:
    projected = projected_gradient.detach().cpu().to(torch.float32).contiguous()
    if projected.ndim != 1 or projected.numel() != 1024:
        raise ValueError("replay projected gradient must have shape (1024,)")
    reverse_kl, opd_rms, valid_count, response_length = diagnostics
    return ReplayVector(
        vector_id=vector_id,
        stable_id=trajectory.stable_id,
        split=trajectory.split,
        representation=representation,
        engine_seed=trajectory.engine_seed,
        rollout_slot=rollout_slot,
        aggregation=aggregation,
        source_capture_sha256=trajectory.source_capture_sha256,
        projected_gradient=projected,
        full_gradient_norm=full_gradient_norm.detach().cpu().to(torch.float32),
        projected_gradient_norm=torch.linalg.vector_norm(projected),
        valid_token_count=valid_count,
        response_length=response_length,
        sampled_reverse_kl=reverse_kl,
        opd_signal_rms=opd_rms,
        verifier_correct_count=(
            None
            if verifier_correct_count is None
            else torch.tensor(float(verifier_correct_count), dtype=torch.float32)
        ),
        verifier_total=(
            None
            if verifier_total is None
            else torch.tensor(float(verifier_total), dtype=torch.float32)
        ),
    )


def _replay_proxy_one(
    actor,
    trajectory: ReplayTrajectory,
    projector,
    config: ProjectionConfig,
) -> _TrajectoryReplay:
    validate_replay_trajectory_integrity(trajectory)
    verify_captured_actor_log_prob(actor, trajectory)
    layout = build_parameter_layout(_actor_module(actor))
    _zero_grad(actor)
    replay_loss = compute_replay_loss(actor, trajectory)
    replay_loss.loss.backward()
    full_gradient = flatten_float32_gradients(_actor_module(actor), layout)
    norm = full_gradient_l2_norm(full_gradient)
    projected = project_full_gradient(projector, full_gradient, config).squeeze(0)
    diagnostics = _trajectory_diagnostics(replay_loss, trajectory)
    vector = _make_vector(
        vector_id=proxy_vector_id(
            trajectory.stable_id,
            trajectory.engine_seed,
            trajectory.rollout_slot,
        ),
        representation="P",
        aggregation="n1",
        trajectory=trajectory,
        projected_gradient=projected,
        full_gradient_norm=norm,
        diagnostics=diagnostics,
        rollout_slot=trajectory.rollout_slot,
        verifier_correct_count=trajectory.verifier_correct_count,
        verifier_total=trajectory.verifier_total,
    )
    _zero_grad(actor)
    return _TrajectoryReplay(replay_loss, full_gradient, vector)


def replay_proxy_trajectory(
    actor,
    trajectory: ReplayTrajectory,
    projector,
    config: ProjectionConfig = ProjectionConfig(),
) -> ReplayVector:
    return _replay_proxy_one(actor, trajectory, projector, config).vector


def _group_diagnostics(
    individual: Sequence[ReplayVector],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    reverse_kl = torch.stack([row.sampled_reverse_kl for row in individual]).mean()
    opd_rms = torch.sqrt(
        torch.square(torch.stack([row.opd_signal_rms for row in individual])).mean()
    )
    valid_count = torch.stack([row.valid_token_count for row in individual]).mean()
    response_length = torch.stack([row.response_length for row in individual]).mean()
    return reverse_kl, opd_rms, valid_count, response_length


def _group_verifier(
    trajectories: Sequence[ReplayTrajectory],
) -> tuple[int | None, int | None]:
    if all(row.verifier_total is None for row in trajectories):
        return None, None
    if any(row.verifier_total is None for row in trajectories):
        raise ValueError("group verifier metadata must be present for every trajectory")
    return (
        sum(int(row.verifier_correct_count) for row in trajectories),
        sum(int(row.verifier_total) for row in trajectories),
    )


def replay_proxy_group(
    actor,
    trajectories: Sequence[ReplayTrajectory],
    projector,
    config: ProjectionConfig = ProjectionConfig(),
) -> ProxyGroupReplay:
    ordered = _validate_group(trajectories)
    layout = build_parameter_layout(_actor_module(actor))
    accumulator = torch.zeros(
        layout.total_numel,
        dtype=torch.float32,
        device=_actor_device(actor),
    )
    individual: list[ReplayVector] = []
    for trajectory in ordered:
        replay = _replay_proxy_one(actor, trajectory, projector, config)
        accumulator.add_(replay.full_gradient / len(ordered))
        individual.append(replay.vector)
        del replay
    group_norm = full_gradient_l2_norm(accumulator)
    group_projected = project_full_gradient(projector, accumulator, config).squeeze(0)
    projected_mean = torch.stack(
        [row.projected_gradient.to(group_projected.device) for row in individual]
    ).mean(dim=0)
    if not torch.allclose(
        group_projected,
        projected_mean,
        rtol=PROJECTION_LINEARITY_RTOL,
        atol=PROJECTION_LINEARITY_ATOL,
    ):
        raise ValueError("proxy P_n4 direct projection fails mean(P_n1) linearity")
    diagnostics = _group_diagnostics(individual)
    verifier_correct, verifier_total = _group_verifier(ordered)
    group = _make_vector(
        vector_id=proxy_group_vector_id(ordered[0].stable_id, ordered[0].engine_seed),
        representation="P",
        aggregation="n4",
        trajectory=ordered[0],
        projected_gradient=group_projected,
        full_gradient_norm=group_norm,
        diagnostics=diagnostics,
        rollout_slot=None,
        verifier_correct_count=verifier_correct,
        verifier_total=verifier_total,
    )
    _zero_grad(actor)
    return ProxyGroupReplay(individual=tuple(individual), group=group)


def _accumulate_target_with_diagnostics(
    actor, trajectories: Sequence[ReplayTrajectory]
) -> tuple[
    torch.Tensor,
    tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], ...],
    tuple[ReplayTrajectory, ...],
]:
    ordered = _validate_group(trajectories)
    layout = build_parameter_layout(_actor_module(actor))
    accumulator = torch.zeros(
        layout.total_numel,
        dtype=torch.float32,
        device=_actor_device(actor),
    )
    diagnostics = []
    for trajectory in ordered:
        verify_captured_actor_log_prob(actor, trajectory)
        _zero_grad(actor)
        loss = compute_replay_loss(actor, trajectory)
        loss.loss.backward()
        _accumulate_parameter_gradients(
            _actor_module(actor),
            layout,
            accumulator,
            scale=1.0 / len(ordered),
        )
        diagnostics.append(_trajectory_diagnostics(loss, trajectory))
        del loss
    _zero_grad(actor)
    full_gradient_l2_norm(accumulator)
    return accumulator, tuple(diagnostics), ordered


def accumulate_target_group_full_gradient(
    actor, trajectories: Sequence[ReplayTrajectory]
) -> torch.Tensor:
    return _accumulate_target_with_diagnostics(actor, trajectories)[0]


def replay_target_group(
    actor,
    trajectories: Sequence[ReplayTrajectory],
    projector,
    config: ProjectionConfig = ProjectionConfig(),
) -> ReplayVector:
    full_gradient, individual_diagnostics, ordered = (
        _accumulate_target_with_diagnostics(actor, trajectories)
    )
    projected = project_full_gradient(projector, full_gradient, config).squeeze(0)
    diagnostics = (
        torch.stack([value[0] for value in individual_diagnostics]).mean(),
        torch.sqrt(
            torch.square(
                torch.stack([value[1] for value in individual_diagnostics])
            ).mean()
        ),
        torch.stack([value[2] for value in individual_diagnostics]).mean(),
        torch.stack([value[3] for value in individual_diagnostics]).mean(),
    )
    verifier_correct, verifier_total = _group_verifier(ordered)
    return _make_vector(
        vector_id=target_group_vector_id(ordered[0].stable_id, ordered[0].engine_seed),
        representation="T",
        aggregation="n4",
        trajectory=ordered[0],
        projected_gradient=projected,
        full_gradient_norm=full_gradient_l2_norm(full_gradient),
        diagnostics=diagnostics,
        rollout_slot=None,
        verifier_correct_count=verifier_correct,
        verifier_total=verifier_total,
    )


def _validate_replay_vector(record: ReplayVector, representation: str) -> None:
    if not isinstance(record, ReplayVector):
        raise TypeError("record factory must return ReplayVector")
    if record.representation != representation:
        raise ValueError("replay vector representation mismatch")
    for text_field in ("vector_id", "stable_id", "split", "aggregation"):
        value = getattr(record, text_field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"replay vector {text_field} must be nonempty")
    if _SHA256_RE.fullmatch(record.source_capture_sha256) is None:
        raise ValueError("replay vector source capture hash is invalid")
    projected = record.projected_gradient
    if (
        not isinstance(projected, torch.Tensor)
        or projected.dtype != torch.float32
        or projected.shape != (1024,)
        or not bool(torch.isfinite(projected).all().item())
        or float(torch.linalg.vector_norm(projected).item()) <= 0
    ):
        raise ValueError("replay projected gradient must be finite nonzero float32[1024]")
    for field_name in (
        "full_gradient_norm",
        "projected_gradient_norm",
        "valid_token_count",
        "response_length",
        "sampled_reverse_kl",
        "opd_signal_rms",
    ):
        value = getattr(record, field_name)
        if not isinstance(value, torch.Tensor) or value.numel() != 1 or not bool(
            torch.isfinite(value).all().item()
        ):
            raise ValueError(f"replay vector {field_name} must be a finite scalar")
    if float(record.full_gradient_norm.item()) <= 0:
        raise ValueError("replay full gradient norm must be positive")
    actual_projected_norm = torch.linalg.vector_norm(projected)
    if float(record.projected_gradient_norm.item()) <= 0 or not torch.allclose(
        record.projected_gradient_norm.to(torch.float32).reshape(()),
        actual_projected_norm,
        rtol=1e-5,
        atol=1e-7,
    ):
        raise ValueError("replay projected gradient norm mismatch")
    if (
        float(record.valid_token_count.item()) <= 0
        or float(record.response_length.item()) <= 0
        or float(record.opd_signal_rms.item()) < 0
    ):
        raise ValueError("replay vector count/length/RMS scalars are invalid")
    if (record.verifier_correct_count is None) != (record.verifier_total is None):
        raise ValueError("replay verifier tensors must be present together")
    for count_field in (
        "prompt_token_count",
        "completion_token_count",
        "supervised_label_count",
        "full_token_count",
    ):
        count = getattr(record, count_field)
        if count is not None and (
            isinstance(count, bool) or not isinstance(count, int) or count <= 0
        ):
            raise ValueError(f"replay vector {count_field} must be positive")
    if record.verifier_total is not None and (
        float(record.verifier_total.item()) <= 0
        or float(record.verifier_correct_count.item()) < 0
        or float(record.verifier_correct_count.item())
        > float(record.verifier_total.item())
    ):
        raise ValueError("replay verifier correct/total scalars are invalid")


def _chunk_payload(records: Sequence[ReplayVector]) -> tuple[dict[str, torch.Tensor], bytes]:
    tensors = {
        "projected_gradient": torch.stack(
            [row.projected_gradient for row in records]
        ).to(torch.float32)
    }
    for field_name in (
        "full_gradient_norm",
        "projected_gradient_norm",
        "valid_token_count",
        "response_length",
        "sampled_reverse_kl",
        "opd_signal_rms",
    ):
        tensors[field_name] = torch.stack(
            [getattr(row, field_name).reshape(()) for row in records]
        ).to(torch.float32)
    if records[0].verifier_total is not None:
        if any(row.verifier_total is None for row in records):
            raise ValueError("verifier tensor presence differs within replay chunk")
        tensors["verifier_correct_count"] = torch.stack(
            [row.verifier_correct_count.reshape(()) for row in records]
        ).to(torch.float32)
        tensors["verifier_total"] = torch.stack(
            [row.verifier_total.reshape(()) for row in records]
        ).to(torch.float32)
    elif any(row.verifier_total is not None for row in records):
        raise ValueError("verifier tensor presence differs within replay chunk")
    sidecars = []
    for tensor_row, record in enumerate(records):
        sidecar: dict[str, object] = {
            "vector_id": record.vector_id,
            "stable_id": record.stable_id,
            "split": record.split,
            "representation": record.representation,
            "aggregation": record.aggregation,
            "source_capture_sha256": record.source_capture_sha256,
            "tensor_row": tensor_row,
        }
        if record.representation in {"T", "P"}:
            sidecar["engine_seed"] = record.engine_seed
        if record.rollout_slot is not None:
            sidecar["rollout_slot"] = record.rollout_slot
        for count_field in (
            "prompt_token_count",
            "completion_token_count",
            "supervised_label_count",
            "full_token_count",
        ):
            count = getattr(record, count_field)
            if count is not None:
                sidecar[count_field] = count
        sidecars.append(sidecar)
    return tensors, b"".join(canonical_json_bytes(row) for row in sidecars)


def _write_once(path: Path, payload: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise ValueError(f"existing replay artifact mismatch: {path}")
        return
    atomic_write_bytes(path, payload)


def _write_vector_chunk(
    directory: Path,
    *,
    start: int,
    records: Sequence[ReplayVector],
    shard_contract_sha256: str,
) -> None:
    end = start + len(records)
    tensor_path = directory / f"vectors_{start}_{end}.safetensors"
    sidecar_path = directory / f"vectors_{start}_{end}.jsonl"
    lock_path = directory / f".vectors_{start}_{end}.lock"
    tensors, sidecar_payload = _chunk_payload(records)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".vectors_{start}_{end}.", suffix=".safetensors", dir=directory
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        atomic_save_safetensors(temporary, tensors)
        tensor_payload = temporary.read_bytes()
    finally:
        temporary.unlink(missing_ok=True)
    lock_descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        _write_once(tensor_path, tensor_payload)
        _write_once(sidecar_path, sidecar_payload)
        marker = {
            "schema_version": 1,
            "artifact_type": "opd_replay_vector_chunk_complete",
            "start": start,
            "end": end,
            "vector_ids_sha256": sha256_id_lines(
                [record.vector_id for record in records]
            ),
            "tensor_file": tensor_path.name,
            "tensor_sha256": sha256_file(tensor_path),
            "sidecar_file": sidecar_path.name,
            "sidecar_sha256": sha256_file(sidecar_path),
            "shard_contract_sha256": shard_contract_sha256,
        }
        write_or_validate_manifest(
            directory / f".vectors_{start}_{end}.complete.json", marker
        )
    finally:
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        finally:
            os.close(lock_descriptor)


def _discover_chunks(directory: Path) -> list[tuple[int, int, Path, Path]]:
    grouped: dict[tuple[int, int], dict[str, Path]] = {}
    for path in directory.iterdir():
        match = _VECTOR_CHUNK_RE.fullmatch(path.name)
        if match is None:
            if path.name.startswith("vectors_"):
                raise ValueError(f"malformed replay vector chunk: {path.name}")
            continue
        start, end, extension = int(match.group(1)), int(match.group(2)), match.group(3)
        if end <= start:
            raise ValueError("invalid replay vector chunk range")
        grouped.setdefault((start, end), {})[extension] = path
    chunks = []
    cursor = 0
    for (start, end), files in sorted(grouped.items()):
        if start != cursor or set(files) != {"jsonl", "safetensors"}:
            raise ValueError("replay chunks must form a paired contiguous prefix")
        chunks.append((start, end, files["jsonl"], files["safetensors"]))
        cursor = end
    return chunks


def _read_sidecars(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open("rb") as handle:
        for line in handle:
            value = json.loads(line)
            if line != canonical_json_bytes(value) or not isinstance(value, dict):
                raise ValueError("replay sidecar is not canonical JSON")
            rows.append(value)
    return rows


def _chunk_marker(
    directory: Path,
    start: int,
    end: int,
    *,
    expected_vector_ids: Sequence[str],
    shard_contract_sha256: str,
) -> dict[str, object]:
    marker = _read_canonical_json(
        directory / f".vectors_{start}_{end}.complete.json",
        "replay vector chunk completion marker",
    )
    expected = {
        "schema_version": 1,
        "artifact_type": "opd_replay_vector_chunk_complete",
        "start": start,
        "end": end,
        "vector_ids_sha256": sha256_id_lines(expected_vector_ids[start:end]),
        "tensor_file": f"vectors_{start}_{end}.safetensors",
        "sidecar_file": f"vectors_{start}_{end}.jsonl",
        "shard_contract_sha256": shard_contract_sha256,
    }
    for field_name, value in expected.items():
        if marker.get(field_name) != value:
            raise ValueError(f"replay chunk marker mismatch: {field_name}")
    for artifact, hash_field in (
        (directory / expected["tensor_file"], "tensor_sha256"),
        (directory / expected["sidecar_file"], "sidecar_sha256"),
    ):
        if marker.get(hash_field) != sha256_file(artifact):
            raise ValueError(f"replay chunk marker mismatch: {hash_field}")
    return marker


def _resume_prefix(
    directory: Path,
    expected_vector_ids: Sequence[str],
    *,
    shard_contract_sha256: str,
) -> int:
    prefix = 0
    tensor_names: set[str] | None = None
    for start, end, sidecar_path, tensor_path in _discover_chunks(directory):
        if end > len(expected_vector_ids):
            raise ValueError("replay chunk exceeds expected vector coverage")
        _chunk_marker(
            directory,
            start,
            end,
            expected_vector_ids=expected_vector_ids,
            shard_contract_sha256=shard_contract_sha256,
        )
        sidecars = _read_sidecars(sidecar_path)
        if len(sidecars) != end - start:
            raise ValueError("replay sidecar row count mismatch")
        ids = tuple(str(row.get("vector_id")) for row in sidecars)
        if ids != tuple(expected_vector_ids[start:end]):
            raise ValueError("replay chunk vector ID order mismatch")
        tensors = load_safetensors(tensor_path, device="cpu")
        if not tensors or any(value.shape[0] != end - start for value in tensors.values()):
            raise ValueError("replay tensor row count mismatch")
        if tensor_names is None:
            tensor_names = set(tensors)
        elif set(tensors) != tensor_names:
            raise ValueError("replay tensor names differ across chunks")
        prefix = end
    return prefix


def run_replay_shard(
    *,
    output_directory: Path,
    expected_vector_ids: Sequence[str],
    record_factory: Callable[[int], ReplayVector],
    representation: str,
    parent_hashes: Mapping[str, str],
    source_snapshot: Mapping[str, object],
    repository: Mapping[str, object],
    runtime: Mapping[str, object],
    metadata: Mapping[str, object],
    chunk_size: int = 16,
    verifier_status: str = "not_computed",
    exercise_interrupt_after_chunks: int | None = None,
) -> VectorSet:
    """Create or resume one immutable vector shard from a contiguous prefix."""
    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    expected = tuple(expected_vector_ids)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("expected replay vector IDs must be nonempty and unique")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("replay chunk size must be positive")
    if not isinstance(representation, str) or not representation:
        raise ValueError("replay representation must be nonempty")
    if verifier_status not in {"computed", "not_computed"}:
        raise ValueError("invalid verifier status")
    if exercise_interrupt_after_chunks is not None and (
        isinstance(exercise_interrupt_after_chunks, bool)
        or not isinstance(exercise_interrupt_after_chunks, int)
        or exercise_interrupt_after_chunks <= 0
    ):
        raise ValueError("exercise interrupt chunk count must be positive")
    normalized_parents = dict(parent_hashes)
    if not normalized_parents or any(
        not isinstance(name, str)
        or not name
        or not isinstance(value, str)
        or _SHA256_RE.fullmatch(value) is None
        for name, value in normalized_parents.items()
    ):
        raise ValueError("replay parent hashes must be nonempty SHA-256 values")
    contract = {
        "schema_version": 1,
        "artifact_type": "opd_replay_shard_contract",
        "expected_vector_count": len(expected),
        "expected_vector_ids_sha256": sha256_id_lines(expected),
        "representation": representation,
        "vector_dimension": 1024,
        "chunk_size": chunk_size,
        "parent_hashes": normalized_parents,
        "source_snapshot": dict(source_snapshot),
        "repository": dict(repository),
        "runtime": dict(runtime),
        "verifier": {"status": verifier_status},
        "metadata": dict(metadata),
    }
    write_or_validate_manifest(root / "SHARD_CONTRACT.json", contract)
    shard_contract_sha256 = sha256_file(root / "SHARD_CONTRACT.json")
    if (root / "COMPLETE.json").is_file():
        return load_vector_set(
            root,
            expected_vector_ids=expected,
            expected_parent_hashes=normalized_parents,
            expected_representation=representation,
        )
    prefix = _resume_prefix(
        root, expected, shard_contract_sha256=shard_contract_sha256
    )
    written_chunks = 0
    for start in range(prefix, len(expected), chunk_size):
        end = min(start + chunk_size, len(expected))
        records = []
        for index in range(start, end):
            record = record_factory(index)
            _validate_replay_vector(record, representation)
            if record.vector_id != expected[index]:
                raise ValueError("record factory returned unexpected vector ID")
            if record.source_capture_sha256 not in normalized_parents.values():
                raise ValueError("record source capture hash is not a declared parent")
            if (record.verifier_total is not None) != (verifier_status == "computed"):
                raise ValueError("record verifier tensors disagree with shard contract")
            records.append(record)
        _write_vector_chunk(
            root,
            start=start,
            records=records,
            shard_contract_sha256=shard_contract_sha256,
        )
        written_chunks += 1
        if written_chunks == exercise_interrupt_after_chunks:
            raise RuntimeError(
                "intentional replay interruption after immutable chunk publication"
            )

    chunks = _discover_chunks(root)
    if not chunks or chunks[-1][1] != len(expected):
        raise ValueError("replay shard did not produce complete vector coverage")
    sidecars = []
    chunk_records = []
    for start, end, sidecar_path, _ in chunks:
        sidecars.extend(_read_sidecars(sidecar_path))
        chunk_records.append(
            _chunk_marker(
                root,
                start,
                end,
                expected_vector_ids=expected,
                shard_contract_sha256=shard_contract_sha256,
            )
        )
    stable_ids = [str(row["stable_id"]) for row in sidecars]
    manifest = {
        "schema_version": 1,
        "artifact_type": "vector_set",
        "representation": representation,
        "vector_count": len(expected),
        "vector_dimension": 1024,
        "chunk_count": len(chunks),
        "vector_ids_sha256": sha256_id_lines(expected),
        "stable_ids_sha256": sha256_ordered_id_lines(stable_ids),
        "parent_hashes": normalized_parents,
        "source_snapshot": dict(source_snapshot),
        "repository": dict(repository),
        "runtime": dict(runtime),
        "verifier": {"status": verifier_status},
        "metadata": {
            **dict(metadata),
            "shard_contract_sha256": shard_contract_sha256,
            "replay_chunks": chunk_records,
        },
    }
    write_or_validate_manifest(root / "manifest.json", manifest)
    write_or_validate_manifest(
        root / "COMPLETE.json",
        {
            "schema_version": 1,
            "artifact_type": "vector_set_complete",
            "manifest_sha256": sha256_file(root / "manifest.json"),
            "vector_count": len(expected),
            "chunk_count": len(chunks),
        },
    )
    return load_vector_set(
        root,
        expected_vector_ids=expected,
        expected_parent_hashes=normalized_parents,
        expected_representation=representation,
    )


def _concatenate_optional(values: Sequence[np.ndarray | None]) -> np.ndarray | None:
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("replay shards disagree on optional tensor presence")
    return np.ascontiguousarray(np.concatenate(values, axis=0))


def validate_replay_coverage(
    shard_directories: Sequence[Path],
    *,
    expected_vector_ids: Sequence[str],
    expected_parent_hashes: Mapping[str, str],
    expected_representation: str,
) -> VectorSet:
    """Validate and concatenate completed shards in declared shard order."""
    if not shard_directories:
        raise ValueError("replay coverage requires at least one shard")
    loaded = [
        load_vector_set(
            path,
            expected_parent_hashes=expected_parent_hashes,
            expected_representation=expected_representation,
        )
        for path in shard_directories
    ]
    vector_ids = tuple(value for shard in loaded for value in shard.vector_ids)
    expected = tuple(expected_vector_ids)
    if vector_ids != expected or len(set(vector_ids)) != len(vector_ids):
        raise ValueError("replay vector coverage does not match expected vector IDs")
    stable_ids = tuple(value for shard in loaded for value in shard.stable_ids)
    vectors = np.ascontiguousarray(np.concatenate([shard.vectors for shard in loaded]))
    return VectorSet(
        vector_ids=vector_ids,
        stable_ids=stable_ids,
        vectors=vectors,
        full_gradient_norm=_concatenate_optional(
            [shard.full_gradient_norm for shard in loaded]
        ),
        projected_gradient_norm=_concatenate_optional(
            [shard.projected_gradient_norm for shard in loaded]
        ),
        valid_token_count=_concatenate_optional(
            [shard.valid_token_count for shard in loaded]
        ),
        response_length=_concatenate_optional(
            [shard.response_length for shard in loaded]
        ),
        sampled_reverse_kl=_concatenate_optional(
            [shard.sampled_reverse_kl for shard in loaded]
        ),
        opd_signal_rms=_concatenate_optional(
            [shard.opd_signal_rms for shard in loaded]
        ),
        verifier_correct_count=_concatenate_optional(
            [shard.verifier_correct_count for shard in loaded]
        ),
        verifier_total=_concatenate_optional(
            [shard.verifier_total for shard in loaded]
        ),
        manifest={"shards": [shard.manifest for shard in loaded]},
    )


def _default_replay_actor_config():
    from omegaconf import OmegaConf, open_dict

    from verl.workers.config import FSDPActorConfig, PolicyLossConfig

    config = OmegaConf.create(
        asdict(
            FSDPActorConfig(
                strategy="fsdp",
                rollout_n=4,
                ppo_mini_batch_size=1,
                ppo_micro_batch_size_per_gpu=1,
                ppo_epochs=1,
                use_dynamic_bsz=False,
                use_torch_compile=False,
                use_kl_loss=True,
                kl_loss_coef=0.0,
                entropy_coeff=0.0,
                policy_loss=PolicyLossConfig(
                    loss_mode="vanilla", only_reverse_kl_advantages=True
                ),
            )
        )
    )
    with open_dict(config):
        config.use_remove_padding = True
        config.use_fused_kernels = False
    return config


def load_replay_actor(
    model_path: Path | str,
    *,
    device: str = "cuda:0",
    expected_model_sha256: str | None = None,
    actor_config=None,
):
    """Load one complete unsharded BF16 actor with no optimizer."""
    if device != "cuda:0":
        raise ValueError("replay actor requires process-local cuda:0")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("replay actor requires exactly one visible CUDA device")
    model_root = Path(model_path).resolve()
    model_manifest = recursive_file_manifest(model_root)
    if expected_model_sha256 is not None and model_manifest["manifest_sha256"] != expected_model_sha256:
        raise ValueError("replay actor model manifest hash mismatch")
    from transformers import AutoModelForCausalLM

    from verl.models.transformers.monkey_patch import apply_monkey_patch
    from verl.workers.actor.dp_actor import DataParallelPPOActor

    model = AutoModelForCausalLM.from_pretrained(
        model_root,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
        attn_implementation="flash_attention_2",
    ).to(device)
    if _is_local_shard_model(model):
        raise ValueError("replay actor must be a full unsharded model")
    apply_monkey_patch(
        model=model,
        use_remove_padding=True,
        ulysses_sp_size=1,
        use_fused_kernels=False,
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.train()
    if not torch.distributed.is_initialized():
        rendezvous = tempfile.NamedTemporaryFile(prefix="opd-replay-dist-", delete=False)
        rendezvous_path = rendezvous.name
        rendezvous.close()
        os.unlink(rendezvous_path)
        torch.distributed.init_process_group(
            backend="gloo",
            init_method=f"file://{rendezvous_path}",
            rank=0,
            world_size=1,
        )
    config = _default_replay_actor_config() if actor_config is None else actor_config
    actor = DataParallelPPOActor(
        config=config,
        actor_module=model,
        actor_optimizer=None,
    )
    if actor.actor_optimizer is not None:
        raise AssertionError("replay actor unexpectedly constructed an optimizer")
    build_parameter_layout(model)
    return actor


def _group_capture_trajectories(
    trajectories: Sequence[ReplayTrajectory],
) -> tuple[tuple[ReplayTrajectory, ...], ...]:
    groups: list[tuple[ReplayTrajectory, ...]] = []
    current: list[ReplayTrajectory] = []
    current_key: tuple[str, int] | None = None
    for trajectory in trajectories:
        key = (trajectory.stable_id, trajectory.engine_seed)
        if current_key is not None and key != current_key:
            groups.append(_validate_group(current))
            current = []
        current_key = key
        current.append(trajectory)
    if current:
        groups.append(_validate_group(current))
    if not groups:
        raise ValueError("capture contains no replay trajectory groups")
    group_keys = [(group[0].stable_id, group[0].engine_seed) for group in groups]
    if len(set(group_keys)) != len(group_keys):
        raise ValueError("capture trajectory groups are not compound-key contiguous")
    return tuple(groups)


def run_capture_replay_shard(
    *,
    capture_root: Path,
    model_path: Path,
    output_directory: Path,
    pair: str,
    num_shards: int,
    shard_index: int,
    expected_model_sha256: str,
    source_snapshot_path: Path,
    reference_repo: Path,
    repository_root: Path,
    chunk_size: int = 16,
    exercise_interrupt_after_chunks: int | None = None,
) -> VectorSet:
    """Run one production replay shard from a completed capture root."""
    if pair not in {"target", "proxy"}:
        raise ValueError("replay pair must be target or proxy")
    trajectories = load_capture_trajectories(capture_root)
    groups = _group_capture_trajectories(trajectories)
    start, end = official_shard_bounds(len(groups), num_shards, shard_index)
    if start >= end:
        raise ValueError("replay shard has no trajectory groups")
    shard_groups = groups[start:end]
    verifier_presence = {
        trajectory.verifier_total is not None
        for group in shard_groups
        for trajectory in group
    }
    if len(verifier_presence) != 1:
        raise ValueError("replay shard mixes computed and missing verifier metadata")
    verifier_status = "computed" if verifier_presence == {True} else "not_computed"

    source_snapshot = _read_canonical_json(
        Path(source_snapshot_path), "replay source snapshot"
    )
    source_snapshot_hash = source_snapshot.get("manifest_sha256")
    if not isinstance(source_snapshot_hash, str) or _SHA256_RE.fullmatch(
        source_snapshot_hash
    ) is None:
        raise ValueError("replay source snapshot lacks manifest SHA-256")
    actor = load_replay_actor(
        model_path,
        expected_model_sha256=expected_model_sha256,
    )
    layout = build_parameter_layout(_actor_module(actor))
    reference = verify_prismatic_reference(reference_repo)
    projector_config = ProjectionConfig()
    projector = construct_cuda_projector(
        layout.total_numel,
        "cuda:0",
        projector_config,
        reference_repo=reference_repo,
    )
    projection_manifest = build_projection_manifest(
        gradient_dimension=layout.total_numel,
        parameter_layout_sha256=layout.sha256,
        config=projector_config,
        reference=reference,
    )
    capture_manifest_hash = sha256_file(Path(capture_root) / "manifest.json")
    parent_hashes = {
        "capture_manifest_sha256": capture_manifest_hash,
        "model_manifest_sha256": expected_model_sha256,
        "source_snapshot_sha256": source_snapshot_hash,
    }
    layout_metadata = {
        "total_numel": layout.total_numel,
        "sha256": layout.sha256,
        "entries": [
            {
                "name": entry.name,
                "shape": list(entry.shape),
                "numel": entry.numel,
                "offset": entry.offset,
                "parameter_dtype": str(
                    dict(
                        _actor_module(actor).named_parameters(
                            remove_duplicate=True
                        )
                    )[entry.name].dtype
                ),
                "gradient_dtype": "torch.float32",
            }
            for entry in layout.entries
        ],
    }

    cached_group_index: int | None = None
    cached_records: tuple[ReplayVector, ...] = ()
    if pair == "target":
        expected_vector_ids = tuple(
            target_group_vector_id(group[0].stable_id, group[0].engine_seed)
            for group in shard_groups
        )

        def record_factory(index: int) -> ReplayVector:
            return replay_target_group(actor, shard_groups[index], projector)

        representation = "T"
    else:
        expected_vector_ids = tuple(
            vector_id
            for group in shard_groups
            for vector_id in (
                *(
                    proxy_vector_id(
                        group[0].stable_id,
                        group[0].engine_seed,
                        slot,
                    )
                    for slot in range(4)
                ),
                proxy_group_vector_id(group[0].stable_id, group[0].engine_seed),
            )
        )

        def record_factory(index: int) -> ReplayVector:
            nonlocal cached_group_index, cached_records
            group_index, record_index = divmod(index, 5)
            if cached_group_index != group_index:
                result = replay_proxy_group(actor, shard_groups[group_index], projector)
                cached_records = (*result.individual, result.group)
                cached_group_index = group_index
            return cached_records[record_index]

        representation = "P"

    repository = repository_state(repository_root)
    if repository.get("status") != "":
        raise ValueError("replay repository must be clean; source snapshot is immutable")
    return run_replay_shard(
        output_directory=output_directory,
        expected_vector_ids=expected_vector_ids,
        record_factory=record_factory,
        representation=representation,
        parent_hashes=parent_hashes,
        source_snapshot=source_snapshot,
        repository=repository,
        runtime=build_runtime_metadata("gvendi_analysis"),
        metadata={
            "pair": pair,
            "capture_root": str(Path(capture_root).resolve()),
            "model_path": str(Path(model_path).resolve()),
            "num_shards": num_shards,
            "shard_index": shard_index,
            "group_start": start,
            "group_end": end,
            "projection": projection_manifest,
            "parameter_layout": layout_metadata,
            "replay_actor": {
                "model_parameter_dtype": "float32",
                "autocast_dtype": "bfloat16",
                "mode": "train",
                "attention_implementation": "flash_attention_2",
                "remove_padding": True,
                "gradient_checkpointing": True,
                "policy_loss": "vanilla",
                "only_reverse_kl_advantages": True,
                "loss_aggregation": "token-mean",
                "optimizer": None,
                "gradient_scaler": None,
                "gradient_clipping": None,
            },
        },
        chunk_size=chunk_size,
        verifier_status=verifier_status,
        exercise_interrupt_after_chunks=exercise_interrupt_after_chunks,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay exact OPD actor gradients for one immutable shard"
    )
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--pair", choices=("target", "proxy"), required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--reference-repo", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument(
        "--exercise-interrupt-after-chunks",
        type=int,
        help="Stage-0 resume fixture only: stop after N immutable chunks",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run_capture_replay_shard(
        capture_root=args.capture_root,
        model_path=args.model_path,
        output_directory=args.output_directory,
        pair=args.pair,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        expected_model_sha256=args.expected_model_sha256,
        source_snapshot_path=args.source_snapshot,
        reference_repo=args.reference_repo,
        repository_root=args.repository_root,
        chunk_size=args.chunk_size,
        exercise_interrupt_after_chunks=args.exercise_interrupt_after_chunks,
    )
    return 0


__all__ = [
    "ParameterLayout",
    "ParameterLayoutEntry",
    "ProxyGroupReplay",
    "ReplayLoss",
    "ReplayTrajectory",
    "ReplayVector",
    "accumulate_target_group_full_gradient",
    "build_parameter_layout",
    "compute_replay_loss",
    "flatten_float32_gradients",
    "load_capture_trajectories",
    "load_replay_actor",
    "proxy_group_vector_id",
    "proxy_vector_id",
    "replay_proxy_group",
    "replay_proxy_trajectory",
    "replay_target_group",
    "run_capture_replay_shard",
    "run_replay_shard",
    "target_group_vector_id",
    "validate_replay_coverage",
    "validate_replay_trajectory_integrity",
    "verify_captured_actor_log_prob",
]


if __name__ == "__main__":
    raise SystemExit(main())
