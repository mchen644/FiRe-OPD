"""Pure routing and accounting helpers for adaptive concise-probe OPD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch

from verl import DataProto
from verl.utils.reward_score.math_reward import last_boxed_only_string

_RESPONSE_ALIGNED_KEYS = {
    "responses",
    "response_mask",
    "old_log_probs",
    "ref_log_prob",
    "base_ref_log_prob",
    "rollout_log_probs",
    "rollout_is_weights",
    "advantages",
    "returns",
    "values",
    "token_level_scores",
    "token_level_rewards",
    "student_entropys",
    "ref_entropys",
    "base_ref_entropys",
}
_SEQUENCE_KEYS = {
    "input_ids",
    "attention_mask",
    "position_ids",
    "ref_input_ids",
    "ref_attention_mask",
    "ref_position_ids",
}
_CUMULATIVE_KEYS = (
    "total_questions",
    "concise_probe_count",
    "easy_count",
    "learnable_count",
    "hard_count",
)


@dataclass(frozen=True)
class AdaptiveConciseProbePlan:
    """Rows and per-request caps for the concise diagnostic call."""

    normal_correct: torch.Tensor
    hard: torch.Tensor
    normal_lengths: torch.Tensor
    probe_indices: torch.Tensor
    max_tokens: torch.Tensor


@dataclass(frozen=True)
class AdaptiveConciseRoutingResult:
    """Final route and exact response-token accounting for one normal batch."""

    normal_correct: torch.Tensor
    easy: torch.Tensor
    learnable: torch.Tensor
    hard: torch.Tensor
    normal_lengths: torch.Tensor
    concise_lengths: torch.Tensor
    supervised_lengths: torch.Tensor
    probe_indices: torch.Tensor
    max_tokens: torch.Tensor
    prompt_styles: np.ndarray
    concise_parse_fail_count: int
    concise_cap_hit_count: int


def _require_binary_mask(mask: torch.Tensor, *, name: str) -> None:
    if mask.dim() != 2:
        raise ValueError(f"{name} must be a 2D tensor")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError(f"{name} must be binary")
    seen_zero = torch.cumsum((mask == 0).to(dtype=torch.long), dim=-1) > 0
    if torch.any((mask == 1) & seen_zero):
        raise ValueError(f"{name} must be a contiguous valid-token prefix")


def _require_frozen_threshold_and_ratio(*, correct_reward_threshold: float, concise_cap_ratio: float) -> None:
    if float(correct_reward_threshold) != 0.5:
        raise ValueError("correct reward threshold must remain pinned to 0.5")
    if float(concise_cap_ratio) != 0.5:
        raise ValueError("concise cap ratio must remain pinned to 0.5")


def plan_adaptive_concise_probes(
    *,
    normal_reward: torch.Tensor,
    normal_response_mask: torch.Tensor,
    correct_reward_threshold: float,
    concise_cap_ratio: float,
) -> AdaptiveConciseProbePlan:
    """Select normal-correct rows and derive exact half-length concise caps."""

    _require_frozen_threshold_and_ratio(
        correct_reward_threshold=correct_reward_threshold,
        concise_cap_ratio=concise_cap_ratio,
    )
    if normal_reward.dim() != 2 or normal_reward.shape != normal_response_mask.shape:
        raise ValueError("normal_reward and normal_response_mask must have identical shape and be 2D")
    if not torch.isfinite(normal_reward).all():
        raise ValueError("normal_reward must be finite")
    _require_binary_mask(normal_response_mask, name="normal_response_mask")

    normal_lengths = normal_response_mask.to(dtype=torch.long).sum(dim=-1)
    if torch.any(normal_lengths <= 0):
        raise ValueError("normal responses must have positive valid lengths")
    sequence_reward = normal_reward.float().sum(dim=-1)
    normal_correct = sequence_reward > float(correct_reward_threshold)
    if torch.any(normal_correct & (normal_lengths < 2)):
        raise ValueError("normal-correct responses must contain at least two valid tokens")

    probe_indices = torch.nonzero(normal_correct, as_tuple=False).flatten().to(dtype=torch.long)
    max_tokens = torch.floor(
        normal_lengths[probe_indices].to(dtype=torch.float32) * float(concise_cap_ratio)
    ).to(dtype=torch.long)
    if torch.any(max_tokens <= 0):
        raise ValueError("concise probe max tokens must be positive")

    return AdaptiveConciseProbePlan(
        normal_correct=normal_correct.detach(),
        hard=(~normal_correct).detach(),
        normal_lengths=normal_lengths.detach(),
        probe_indices=probe_indices.detach(),
        max_tokens=max_tokens.detach(),
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


def finalize_adaptive_concise_routing(
    *,
    plan: AdaptiveConciseProbePlan,
    concise_reward: torch.Tensor,
    concise_response_mask: torch.Tensor,
    concise_original_indices: Sequence[int] | np.ndarray,
    concise_texts: Sequence[str],
    correct_reward_threshold: float,
) -> AdaptiveConciseRoutingResult:
    """Map aligned concise diagnostics back to normal rows and finalize routes."""

    _require_frozen_threshold_and_ratio(
        correct_reward_threshold=correct_reward_threshold,
        concise_cap_ratio=0.5,
    )
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
    caps = plan.max_tokens.to(device=concise_probe_lengths.device, dtype=torch.long)
    if concise_probe_lengths.shape != caps.shape or torch.any(concise_probe_lengths > caps):
        raise ValueError("concise diagnostic length exceeds its per-row cap")

    batch_size = int(plan.normal_lengths.numel())
    device = plan.normal_lengths.device
    concise_lengths = torch.zeros(batch_size, device=device, dtype=torch.long)
    easy = torch.zeros(batch_size, device=device, dtype=torch.bool)
    learnable = torch.zeros(batch_size, device=device, dtype=torch.bool)
    if probe_count:
        probe_indices = plan.probe_indices.to(device=device)
        concise_lengths[probe_indices] = concise_probe_lengths.to(device=device)
        concise_correct = concise_reward.float().sum(dim=-1) > float(correct_reward_threshold)
        easy[probe_indices] = concise_correct.to(device=device)
        learnable[probe_indices] = (~concise_correct).to(device=device)

    hard = plan.hard.to(device=device, dtype=torch.bool)
    normal_correct = plan.normal_correct.to(device=device, dtype=torch.bool)
    supervised_lengths = plan.normal_lengths.to(device=device, dtype=torch.long) - concise_lengths
    if torch.any(supervised_lengths <= 0):
        raise ValueError("supervised normal-response lengths must be positive")
    if torch.any(supervised_lengths > plan.normal_lengths.to(device=device)):
        raise ValueError("supervised lengths cannot exceed normal response lengths")
    partition_count = easy.to(torch.long) + learnable.to(torch.long) + hard.to(torch.long)
    if not torch.all(partition_count == 1):
        raise ValueError("easy, learnable, and hard must form an exhaustive disjoint partition")
    if not torch.equal(normal_correct, easy | learnable):
        raise ValueError("normal-correct rows must equal easy plus learnable rows")
    if not torch.equal(concise_lengths + supervised_lengths, plan.normal_lengths.to(device=device)):
        raise ValueError("response-token budget identity failed")

    prompt_styles = np.full(batch_size, "normal", dtype=object)
    prompt_styles[easy.detach().cpu().numpy()] = "concise"
    cap_hit_count = int((concise_probe_lengths == caps).sum().item()) if probe_count else 0

    return AdaptiveConciseRoutingResult(
        normal_correct=normal_correct.detach(),
        easy=easy.detach(),
        learnable=learnable.detach(),
        hard=hard.detach(),
        normal_lengths=plan.normal_lengths.to(device=device, dtype=torch.long).detach(),
        concise_lengths=concise_lengths.detach(),
        supervised_lengths=supervised_lengths.detach(),
        probe_indices=plan.probe_indices.to(device=device, dtype=torch.long).detach(),
        max_tokens=plan.max_tokens.to(device=device, dtype=torch.long).detach(),
        prompt_styles=prompt_styles,
        concise_parse_fail_count=_count_parse_failures(concise_texts),
        concise_cap_hit_count=cap_hit_count,
    )


def _tensor_count(mask: torch.Tensor) -> float:
    return float(mask.to(dtype=torch.long).sum().item())


def summarize_adaptive_concise_routing(result: AdaptiveConciseRoutingResult) -> dict[str, float]:
    """Return complete per-step route, prompt, and response-token metrics."""

    total = int(result.normal_lengths.numel())
    if total <= 0:
        raise ValueError("adaptive concise routing requires a nonempty normal batch")
    easy_count = _tensor_count(result.easy)
    learnable_count = _tensor_count(result.learnable)
    hard_count = _tensor_count(result.hard)
    normal_correct_count = _tensor_count(result.normal_correct)
    probe_count = float(result.probe_indices.numel())
    normal_tokens = float(result.normal_lengths.sum().item())
    concise_tokens = float(result.concise_lengths.sum().item())
    supervised_tokens = float(result.supervised_lengths.sum().item())
    residual = concise_tokens + supervised_tokens - normal_tokens
    if residual != 0.0:
        raise ValueError("response-token budget residual must be zero")
    if normal_correct_count != easy_count + learnable_count or probe_count != normal_correct_count:
        raise ValueError("adaptive concise count identities failed")
    if float(total) != easy_count + learnable_count + hard_count:
        raise ValueError("adaptive concise route partition count failed")

    probed_lengths = result.concise_lengths[result.normal_correct]
    probed_normal_lengths = result.normal_lengths[result.normal_correct]
    concise_length_mean = float(probed_lengths.float().mean().item()) if probed_lengths.numel() else 0.0
    concise_ratio_mean = (
        float((probed_lengths.float() / probed_normal_lengths.float()).mean().item())
        if probed_lengths.numel()
        else 0.0
    )
    supervised_fraction_mean = float(
        (result.supervised_lengths.float() / result.normal_lengths.float()).mean().item()
    )
    denom = float(total)
    return {
        "adaptive_concise_opd/total_questions": denom,
        "adaptive_concise_opd/normal_correct_count": normal_correct_count,
        "adaptive_concise_opd/normal_wrong_count": hard_count,
        "adaptive_concise_opd/concise_probe_count": probe_count,
        "adaptive_concise_opd/easy_count": easy_count,
        "adaptive_concise_opd/learnable_count": learnable_count,
        "adaptive_concise_opd/hard_count": hard_count,
        "adaptive_concise_opd/easy_ratio": easy_count / denom,
        "adaptive_concise_opd/learnable_ratio": learnable_count / denom,
        "adaptive_concise_opd/hard_ratio": hard_count / denom,
        "adaptive_concise_opd/concise_correct_count": easy_count,
        "adaptive_concise_opd/concise_wrong_count": learnable_count,
        "adaptive_concise_opd/concise_parse_fail_count": float(result.concise_parse_fail_count),
        "adaptive_concise_opd/concise_cap_hit_count": float(result.concise_cap_hit_count),
        "adaptive_concise_opd/concise_response_length_mean": concise_length_mean,
        "adaptive_concise_opd/concise_to_normal_length_ratio_mean": concise_ratio_mean,
        "adaptive_concise_opd/concise_teacher_count": easy_count,
        "adaptive_concise_opd/normal_teacher_count": learnable_count + hard_count,
        "adaptive_concise_opd/concise_teacher_ratio": easy_count / denom,
        "adaptive_concise_opd/normal_teacher_ratio": (learnable_count + hard_count) / denom,
        "adaptive_concise_opd/normal_response_tokens": normal_tokens,
        "adaptive_concise_opd/concise_probe_tokens": concise_tokens,
        "adaptive_concise_opd/supervised_normal_tokens": supervised_tokens,
        "adaptive_concise_opd/response_budget_residual_tokens": residual,
        "adaptive_concise_opd/response_budget_ratio": (concise_tokens + supervised_tokens) / normal_tokens,
        "adaptive_concise_opd/supervised_fraction_mean": supervised_fraction_mean,
    }


def update_adaptive_concise_cumulative_counts(
    cumulative: Mapping[str, int],
    result: AdaptiveConciseRoutingResult,
) -> tuple[dict[str, int], dict[str, float]]:
    """Return updated immutable routing-event totals and logger metrics."""

    current = {key: int(cumulative.get(key, 0)) for key in _CUMULATIVE_KEYS}
    if any(value < 0 for value in current.values()):
        raise ValueError("cumulative adaptive concise counts must be nonnegative")
    increments = {
        "total_questions": int(result.normal_lengths.numel()),
        "concise_probe_count": int(result.probe_indices.numel()),
        "easy_count": int(result.easy.sum().item()),
        "learnable_count": int(result.learnable.sum().item()),
        "hard_count": int(result.hard.sum().item()),
    }
    updated = {key: current[key] + increments[key] for key in _CUMULATIVE_KEYS}
    if updated["total_questions"] != updated["easy_count"] + updated["learnable_count"] + updated["hard_count"]:
        raise ValueError("cumulative route counts do not partition total questions")
    if updated["concise_probe_count"] != updated["easy_count"] + updated["learnable_count"]:
        raise ValueError("cumulative probe count does not equal easy plus learnable")
    metrics = {
        f"adaptive_concise_opd/{key}_cumulative": float(value)
        for key, value in updated.items()
    }
    return updated, metrics


def _slice_sequence_suffix(tensor: torch.Tensor, *, response_width: int, keep_width: int) -> torch.Tensor:
    prompt_width = tensor.shape[-1] - response_width
    if prompt_width < 0:
        raise ValueError("sequence tensor is shorter than the response width")
    return tensor[..., : prompt_width + keep_width]


def truncate_to_adaptive_concise_prefix(
    *, batch: DataProto, supervised_lengths: torch.Tensor
) -> DataProto:
    """Physically bound the actor/ref batch to per-row normal-response prefixes."""

    if batch.batch is None or "responses" not in batch.batch.keys() or "response_mask" not in batch.batch.keys():
        raise ValueError("adaptive prefix truncation requires responses and response_mask")
    response_mask = batch.batch["response_mask"]
    _require_binary_mask(response_mask, name="response_mask")
    batch_size, response_width = response_mask.shape
    if supervised_lengths.dim() != 1 or supervised_lengths.shape[0] != batch_size:
        raise ValueError("supervised_lengths must have shape [batch]")
    supervised_lengths = supervised_lengths.to(device=response_mask.device, dtype=torch.long)
    valid_lengths = response_mask.to(dtype=torch.long).sum(dim=-1)
    if torch.any(supervised_lengths <= 0):
        raise ValueError("supervised lengths must be positive")
    if torch.any(supervised_lengths > valid_lengths):
        raise ValueError("supervised lengths cannot exceed valid response lengths")

    positions = torch.arange(response_width, device=response_mask.device).unsqueeze(0)
    prefix_mask = ((positions < supervised_lengths.unsqueeze(1)) & response_mask.bool()).to(
        dtype=response_mask.dtype
    )
    keep_width = int(supervised_lengths.max().item())
    tensors: dict[str, torch.Tensor] = {}
    for key, tensor in batch.batch.items():
        if key in _RESPONSE_ALIGNED_KEYS and tensor.dim() >= 2 and tensor.shape[-1] == response_width:
            tensors[key] = tensor[..., :keep_width]
        elif key in _SEQUENCE_KEYS and tensor.dim() >= 2 and tensor.shape[-1] >= response_width:
            tensors[key] = _slice_sequence_suffix(
                tensor, response_width=response_width, keep_width=keep_width
            )
        else:
            tensors[key] = tensor

    truncated_mask = prefix_mask[..., :keep_width]
    tensors["response_mask"] = truncated_mask
    for attention_key in ("attention_mask", "ref_attention_mask"):
        if attention_key in tensors:
            attention_mask = tensors[attention_key]
            prompt_width = attention_mask.shape[-1] - keep_width
            tensors[attention_key] = torch.cat(
                [attention_mask[..., :prompt_width], truncated_mask.to(dtype=attention_mask.dtype)],
                dim=-1,
            )

    meta_info = dict(batch.meta_info)
    if "attention_mask" in tensors:
        meta_info["global_token_num"] = torch.sum(tensors["attention_mask"], dim=-1).tolist()
    return DataProto.from_dict(
        tensors=tensors,
        non_tensors=dict(batch.non_tensor_batch),
        meta_info=meta_info,
    )
