"""Pure routing and preservation helpers for endpoint-preserving tri-prompt OPD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch

from verl import DataProto
from verl.utils.reward_score.math_reward import last_boxed_only_string

_CUMULATIVE_KEYS = (
    "total_questions",
    "concise_probe_count",
    "easy_count",
    "sensitive_count",
    "hard_count",
)
_PRIMARY_TENSOR_KEYS = (
    "responses",
    "response_mask",
    "input_ids",
    "attention_mask",
    "position_ids",
)


@dataclass(frozen=True)
class AdaptiveTriPromptProbePlan:
    """Normal-correct rows selected for one full-cap concise diagnostic."""

    normal_correct: torch.Tensor
    hard: torch.Tensor
    normal_lengths: torch.Tensor
    probe_indices: torch.Tensor


@dataclass(frozen=True)
class AdaptiveTriPromptRoutingResult:
    """Three-way teacher-prompt route with unchanged normal supervision."""

    normal_correct: torch.Tensor
    easy: torch.Tensor
    sensitive: torch.Tensor
    hard: torch.Tensor
    normal_lengths: torch.Tensor
    concise_lengths: torch.Tensor
    actor_supervised_lengths: torch.Tensor
    probe_indices: torch.Tensor
    budgets: torch.Tensor
    teacher_prompt_styles: np.ndarray
    concise_parse_fail_count: int
    concise_cap_hit_count: int


@dataclass(frozen=True)
class _TensorDescriptor:
    data_ptr: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    version: int


@dataclass(frozen=True)
class PrimaryTensorContract:
    """Zero-copy descriptors proving routing did not alter primary tensors."""

    tensors: Mapping[str, _TensorDescriptor]
    reward: _TensorDescriptor
    response_token_count: int


def _require_binary_mask(mask: torch.Tensor, *, name: str) -> None:
    if mask.dim() != 2:
        raise ValueError(f"{name} must be a 2D tensor")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError(f"{name} must be binary")
    seen_zero = torch.cumsum((mask == 0).to(dtype=torch.long), dim=-1) > 0
    if torch.any((mask == 1) & seen_zero):
        raise ValueError(f"{name} must be a contiguous valid-token prefix")


def _require_frozen_threshold(correct_reward_threshold: float) -> None:
    if float(correct_reward_threshold) != 0.5:
        raise ValueError("correct reward threshold must remain pinned to 0.5")


def plan_adaptive_triprompt_probes(
    *,
    normal_reward: torch.Tensor,
    normal_response_mask: torch.Tensor,
    correct_reward_threshold: float,
) -> AdaptiveTriPromptProbePlan:
    """Select exactly the normal-correct rows for concise diagnostics."""

    _require_frozen_threshold(correct_reward_threshold)
    if normal_reward.dim() != 2 or normal_reward.shape != normal_response_mask.shape:
        raise ValueError("normal_reward and normal_response_mask must have identical shape and be 2D")
    if not torch.isfinite(normal_reward).all():
        raise ValueError("normal_reward must be finite")
    _require_binary_mask(normal_response_mask, name="normal_response_mask")

    normal_lengths = normal_response_mask.to(dtype=torch.long).sum(dim=-1)
    if torch.any(normal_lengths <= 0):
        raise ValueError("normal responses must have positive valid lengths")
    normal_correct = normal_reward.float().sum(dim=-1) > float(correct_reward_threshold)
    probe_indices = torch.nonzero(normal_correct, as_tuple=False).flatten().to(dtype=torch.long)
    return AdaptiveTriPromptProbePlan(
        normal_correct=normal_correct.detach(),
        hard=(~normal_correct).detach(),
        normal_lengths=normal_lengths.detach(),
        probe_indices=probe_indices.detach(),
    )


def _count_parse_failures(texts: Sequence[str]) -> int:
    failures = 0
    for text in texts:
        try:
            extracted = last_boxed_only_string(str(text))
        except Exception:
            extracted = None
        failures += extracted is None
    return int(failures)


def finalize_adaptive_triprompt_routing(
    *,
    plan: AdaptiveTriPromptProbePlan,
    concise_reward: torch.Tensor,
    concise_response_mask: torch.Tensor,
    concise_original_indices: Sequence[int] | np.ndarray,
    concise_texts: Sequence[str],
    correct_reward_threshold: float,
    max_response_length: int,
) -> AdaptiveTriPromptRoutingResult:
    """Map full-cap diagnostics back to primary rows and select teacher prompts."""

    _require_frozen_threshold(correct_reward_threshold)
    if isinstance(max_response_length, bool) or not isinstance(max_response_length, int):
        raise ValueError("max_response_length must be a positive integer")
    if max_response_length <= 0:
        raise ValueError("max_response_length must be positive")
    if concise_reward.dim() != 2 or concise_reward.shape != concise_response_mask.shape:
        raise ValueError("concise_reward and concise_response_mask must have identical shape and be 2D")
    if not torch.isfinite(concise_reward).all():
        raise ValueError("concise_reward must be finite")
    _require_binary_mask(concise_response_mask, name="concise_response_mask")

    expected_indices = plan.probe_indices.detach().cpu().numpy().astype(np.int64, copy=False)
    actual_indices = np.asarray(concise_original_indices)
    if actual_indices.ndim != 1 or not np.issubdtype(actual_indices.dtype, np.integer):
        raise ValueError("concise diagnostic mapping must be a one-dimensional integer array")
    actual_indices = actual_indices.astype(np.int64, copy=False)
    if not np.array_equal(actual_indices, expected_indices):
        raise ValueError("concise diagnostic mapping must exactly match probe indices")

    probe_count = int(expected_indices.size)
    if concise_reward.shape[0] != probe_count or len(concise_texts) != probe_count:
        raise ValueError("concise diagnostic mapping count does not match probe plan")
    concise_probe_lengths = concise_response_mask.to(dtype=torch.long).sum(dim=-1)
    if probe_count and torch.any(concise_probe_lengths <= 0):
        raise ValueError("concise diagnostics must contain positive valid lengths")
    if torch.any(concise_probe_lengths > int(max_response_length)):
        raise ValueError("concise diagnostic length exceeds max_response_length")

    batch_size = int(plan.normal_lengths.numel())
    device = plan.normal_lengths.device
    concise_lengths = torch.zeros(batch_size, device=device, dtype=torch.long)
    easy = torch.zeros(batch_size, device=device, dtype=torch.bool)
    sensitive = torch.zeros(batch_size, device=device, dtype=torch.bool)
    if probe_count:
        probe_indices = plan.probe_indices.to(device=device, dtype=torch.long)
        concise_lengths[probe_indices] = concise_probe_lengths.to(device=device, dtype=torch.long)
        concise_correct = concise_reward.float().sum(dim=-1) > float(correct_reward_threshold)
        easy[probe_indices] = concise_correct.to(device=device)
        sensitive[probe_indices] = (~concise_correct).to(device=device)

    normal_correct = plan.normal_correct.to(device=device, dtype=torch.bool)
    hard = plan.hard.to(device=device, dtype=torch.bool)
    partition_count = easy.to(torch.long) + sensitive.to(torch.long) + hard.to(torch.long)
    if not torch.all(partition_count == 1):
        raise ValueError("easy, sensitive, and hard must form an exhaustive disjoint partition")
    if not torch.equal(normal_correct, easy | sensitive):
        raise ValueError("normal-correct rows must equal easy plus sensitive rows")

    normal_lengths = plan.normal_lengths.to(device=device, dtype=torch.long)
    actor_supervised_lengths = normal_lengths.clone()
    budgets = torch.zeros(batch_size, device=device, dtype=torch.long)
    budgets[sensitive] = normal_lengths[sensitive]
    teacher_prompt_styles = np.full(batch_size, "normal", dtype=object)
    teacher_prompt_styles[sensitive.detach().cpu().numpy()] = "budget"
    teacher_prompt_styles[easy.detach().cpu().numpy()] = "concise"
    cap_hit_count = int((concise_probe_lengths == int(max_response_length)).sum().item()) if probe_count else 0

    return AdaptiveTriPromptRoutingResult(
        normal_correct=normal_correct.detach(),
        easy=easy.detach(),
        sensitive=sensitive.detach(),
        hard=hard.detach(),
        normal_lengths=normal_lengths.detach(),
        concise_lengths=concise_lengths.detach(),
        actor_supervised_lengths=actor_supervised_lengths.detach(),
        probe_indices=plan.probe_indices.to(device=device, dtype=torch.long).detach(),
        budgets=budgets.detach(),
        teacher_prompt_styles=teacher_prompt_styles,
        concise_parse_fail_count=_count_parse_failures(concise_texts),
        concise_cap_hit_count=cap_hit_count,
    )


def _tensor_count(mask: torch.Tensor) -> float:
    return float(mask.to(dtype=torch.long).sum().item())


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    selected = values[mask]
    return float(selected.float().mean().item()) if selected.numel() else 0.0


def summarize_adaptive_triprompt_routing(
    result: AdaptiveTriPromptRoutingResult,
) -> dict[str, float]:
    """Return complete route, teacher-prompt, and full-supervision metrics."""

    total = int(result.normal_lengths.numel())
    if total <= 0:
        raise ValueError("adaptive tri-prompt OPD requires a nonempty primary batch")
    easy_count = _tensor_count(result.easy)
    sensitive_count = _tensor_count(result.sensitive)
    hard_count = _tensor_count(result.hard)
    normal_correct_count = _tensor_count(result.normal_correct)
    probe_count = float(result.probe_indices.numel())
    if float(total) != easy_count + sensitive_count + hard_count:
        raise ValueError("tri-prompt route counts do not partition total questions")
    if normal_correct_count != easy_count + sensitive_count:
        raise ValueError("normal-correct count does not equal easy plus sensitive")
    if probe_count != normal_correct_count:
        raise ValueError("concise probe count does not equal normal-correct count")

    normal_tokens = float(result.normal_lengths.sum().item())
    actor_tokens = float(result.actor_supervised_lengths.sum().item())
    residual = actor_tokens - normal_tokens
    if residual != 0.0 or not torch.equal(result.actor_supervised_lengths, result.normal_lengths):
        raise ValueError("primary full-response supervision was not preserved")
    if torch.any(result.budgets[~result.sensitive] != 0):
        raise ValueError("only sensitive rows may carry a budget")
    if not torch.equal(result.budgets[result.sensitive], result.normal_lengths[result.sensitive]):
        raise ValueError("sensitive budgets must equal normal response lengths")

    sensitive_budgets = result.budgets[result.sensitive]
    if sensitive_budgets.numel():
        budget_min = float(sensitive_budgets.min().item())
        budget_mean = float(sensitive_budgets.float().mean().item())
        budget_max = float(sensitive_budgets.max().item())
    else:
        budget_min = budget_mean = budget_max = 0.0

    denom = float(total)
    concise_probe_lengths = result.concise_lengths[result.normal_correct]
    return {
        "adaptive_triprompt_opd/total_questions": denom,
        "adaptive_triprompt_opd/normal_correct_count": normal_correct_count,
        "adaptive_triprompt_opd/normal_wrong_count": hard_count,
        "adaptive_triprompt_opd/concise_probe_count": probe_count,
        "adaptive_triprompt_opd/easy_count": easy_count,
        "adaptive_triprompt_opd/sensitive_count": sensitive_count,
        "adaptive_triprompt_opd/hard_count": hard_count,
        "adaptive_triprompt_opd/easy_ratio": easy_count / denom,
        "adaptive_triprompt_opd/sensitive_ratio": sensitive_count / denom,
        "adaptive_triprompt_opd/hard_ratio": hard_count / denom,
        "adaptive_triprompt_opd/concise_teacher_count": easy_count,
        "adaptive_triprompt_opd/budget_teacher_count": sensitive_count,
        "adaptive_triprompt_opd/normal_teacher_count": hard_count,
        "adaptive_triprompt_opd/concise_correct_count": easy_count,
        "adaptive_triprompt_opd/concise_wrong_count": sensitive_count,
        "adaptive_triprompt_opd/concise_parse_fail_count": float(result.concise_parse_fail_count),
        "adaptive_triprompt_opd/concise_missing_box_count": float(result.concise_parse_fail_count),
        "adaptive_triprompt_opd/concise_cap_hit_count": float(result.concise_cap_hit_count),
        "adaptive_triprompt_opd/concise_response_tokens": float(result.concise_lengths.sum().item()),
        "adaptive_triprompt_opd/concise_response_length_mean": (
            float(concise_probe_lengths.float().mean().item()) if concise_probe_lengths.numel() else 0.0
        ),
        "adaptive_triprompt_opd/easy_normal_response_length_mean": _masked_mean(
            result.normal_lengths, result.easy
        ),
        "adaptive_triprompt_opd/sensitive_normal_response_length_mean": _masked_mean(
            result.normal_lengths, result.sensitive
        ),
        "adaptive_triprompt_opd/hard_normal_response_length_mean": _masked_mean(
            result.normal_lengths, result.hard
        ),
        "adaptive_triprompt_opd/easy_concise_response_length_mean": _masked_mean(
            result.concise_lengths, result.easy
        ),
        "adaptive_triprompt_opd/sensitive_concise_response_length_mean": _masked_mean(
            result.concise_lengths, result.sensitive
        ),
        "adaptive_triprompt_opd/sensitive_budget_min": budget_min,
        "adaptive_triprompt_opd/sensitive_budget_mean": budget_mean,
        "adaptive_triprompt_opd/sensitive_budget_max": budget_max,
        "adaptive_triprompt_opd/normal_response_tokens": normal_tokens,
        "adaptive_triprompt_opd/actor_supervised_tokens": actor_tokens,
        "adaptive_triprompt_opd/supervision_token_residual": residual,
        "adaptive_triprompt_opd/full_response_preservation_ratio": actor_tokens / normal_tokens,
        "adaptive_triprompt_opd/route_weight_min": 1.0,
        "adaptive_triprompt_opd/route_weight_max": 1.0,
    }


def update_adaptive_triprompt_cumulative_counts(
    cumulative: Mapping[str, int],
    result: AdaptiveTriPromptRoutingResult,
) -> tuple[dict[str, int], dict[str, float]]:
    """Return updated immutable routing-event totals and logger metrics."""

    current = {key: int(cumulative.get(key, 0)) for key in _CUMULATIVE_KEYS}
    if any(value < 0 for value in current.values()):
        raise ValueError("cumulative tri-prompt counts must be nonnegative")
    if current["total_questions"] != current["easy_count"] + current["sensitive_count"] + current["hard_count"]:
        raise ValueError("existing cumulative routes do not partition total questions")
    if current["concise_probe_count"] != current["easy_count"] + current["sensitive_count"]:
        raise ValueError("existing cumulative probes do not equal easy plus sensitive")

    increments = {
        "total_questions": int(result.normal_lengths.numel()),
        "concise_probe_count": int(result.probe_indices.numel()),
        "easy_count": int(result.easy.sum().item()),
        "sensitive_count": int(result.sensitive.sum().item()),
        "hard_count": int(result.hard.sum().item()),
    }
    updated = {key: current[key] + increments[key] for key in _CUMULATIVE_KEYS}
    if updated["total_questions"] != updated["easy_count"] + updated["sensitive_count"] + updated["hard_count"]:
        raise ValueError("cumulative routes do not partition total questions")
    if updated["concise_probe_count"] != updated["easy_count"] + updated["sensitive_count"]:
        raise ValueError("cumulative probes do not equal easy plus sensitive")
    metrics = {
        f"adaptive_triprompt_opd/{key}_cumulative": float(value)
        for key, value in updated.items()
    }
    return updated, metrics


def summarize_response_endpoints(
    *,
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
    eos_token_id: int,
    max_response_length: int,
    metric_prefix: str,
) -> dict[str, float]:
    """Summarize natural EOS and global-cap termination without changing correctness."""

    if response_ids.dim() != 2 or response_ids.shape != response_mask.shape:
        raise ValueError("response_ids and response_mask must have identical shape and be 2D")
    _require_binary_mask(response_mask, name="response_mask")
    if isinstance(max_response_length, bool) or not isinstance(max_response_length, int) or max_response_length <= 0:
        raise ValueError("max_response_length must be a positive integer")
    if not isinstance(metric_prefix, str) or not metric_prefix:
        raise ValueError("metric_prefix must be nonempty")

    lengths = response_mask.to(dtype=torch.long).sum(dim=-1)
    if torch.any(lengths <= 0) or torch.any(lengths > int(max_response_length)):
        raise ValueError("valid response lengths must be within the global response bound")
    total = int(lengths.numel())
    if total == 0:
        return {
            f"{metric_prefix}_eos_count": 0.0,
            f"{metric_prefix}_eos_ratio": 0.0,
            f"{metric_prefix}_cap_hit_count": 0.0,
            f"{metric_prefix}_cap_hit_ratio": 0.0,
            f"{metric_prefix}_response_length_mean": 0.0,
            f"{metric_prefix}_response_tokens": 0.0,
        }

    valid_eos = (response_ids == int(eos_token_id)) & response_mask.bool()
    eos_per_row = valid_eos.to(dtype=torch.long).sum(dim=-1)
    if torch.any(eos_per_row > 1):
        raise ValueError("a response may contain at most one valid EOS token")
    eos_rows = eos_per_row == 1
    if torch.any(eos_rows):
        positions = torch.arange(response_ids.shape[-1], device=response_ids.device).unsqueeze(0)
        eos_positions = torch.where(valid_eos, positions, torch.full_like(positions, -1)).max(dim=-1).values
        if torch.any(eos_rows & (eos_positions != lengths - 1)):
            raise ValueError("EOS must be the terminal valid response token")

    eos_count = int(eos_rows.sum().item())
    cap_hit_count = int((lengths == int(max_response_length)).sum().item())
    return {
        f"{metric_prefix}_eos_count": float(eos_count),
        f"{metric_prefix}_eos_ratio": eos_count / float(total),
        f"{metric_prefix}_cap_hit_count": float(cap_hit_count),
        f"{metric_prefix}_cap_hit_ratio": cap_hit_count / float(total),
        f"{metric_prefix}_response_length_mean": float(lengths.float().mean().item()),
        f"{metric_prefix}_response_tokens": float(lengths.sum().item()),
    }


def _describe_tensor(tensor: torch.Tensor) -> _TensorDescriptor:
    return _TensorDescriptor(
        data_ptr=int(tensor.data_ptr()),
        shape=tuple(tensor.shape),
        stride=tuple(tensor.stride()),
        dtype=tensor.dtype,
        device=tensor.device,
        version=int(tensor._version),
    )


def capture_primary_tensor_contract(
    *, batch: DataProto, reward_tensor: torch.Tensor
) -> PrimaryTensorContract:
    """Capture zero-copy descriptors for tensors routing must not alter."""

    if batch.batch is None:
        raise ValueError("primary batch tensors are required")
    missing = [key for key in _PRIMARY_TENSOR_KEYS if key not in batch.batch.keys()]
    if missing:
        raise ValueError(f"primary batch is missing required tensors: {missing}")
    tensors = {key: _describe_tensor(batch.batch[key]) for key in _PRIMARY_TENSOR_KEYS}
    response_mask = batch.batch["response_mask"]
    _require_binary_mask(response_mask, name="response_mask")
    if reward_tensor.shape != response_mask.shape:
        raise ValueError("reward tensor must align with the primary response mask")
    return PrimaryTensorContract(
        tensors=tensors,
        reward=_describe_tensor(reward_tensor),
        response_token_count=int(response_mask.to(dtype=torch.long).sum().item()),
    )


def verify_primary_tensor_contract(
    *, snapshot: PrimaryTensorContract, batch: DataProto, reward_tensor: torch.Tensor
) -> None:
    """Fail if routing replaced or mutated any primary tensor or reward."""

    if batch.batch is None:
        raise ValueError("primary batch tensors changed: batch is empty")
    for key, expected in snapshot.tensors.items():
        if key not in batch.batch.keys() or _describe_tensor(batch.batch[key]) != expected:
            raise ValueError(f"primary tensor changed during routing: {key}")
    if _describe_tensor(reward_tensor) != snapshot.reward:
        raise ValueError("primary reward tensor changed during routing")
    token_count = int(batch.batch["response_mask"].to(dtype=torch.long).sum().item())
    if token_count != snapshot.response_token_count:
        raise ValueError("primary supervised token count changed during routing")
