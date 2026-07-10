"""Pure helpers for single-rollout difficulty-routed Budget20 OPD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

_ALLOWED_PROMPT_STYLES = {"budget", "normal", "concise"}


@dataclass(frozen=True)
class DifficultyAwareRoutingResult:
    correct: torch.Tensor
    confidence_rank: torch.Tensor
    easy: torch.Tensor
    hard: torch.Tensor
    esr_beta: torch.Tensor
    entropy_weight: torch.Tensor
    prompt_styles: np.ndarray
    sequence_rewards: torch.Tensor
    student_logp_mean: torch.Tensor
    easy_prompt_threshold: float


@dataclass(frozen=True)
class GroupSuccessRoutingResult:
    correct: torch.Tensor
    group_correct_count: torch.Tensor
    group_accuracy: torch.Tensor
    easy: torch.Tensor
    learnable: torch.Tensor
    unresolved: torch.Tensor
    esr_beta: torch.Tensor
    prompt_styles: np.ndarray
    sequence_rewards: torch.Tensor
    group_correct_counts: torch.Tensor
    expected_group_size: int


def _config_get(config, name: str, default):
    if hasattr(config, "get"):
        return config.get(name, default)
    return getattr(config, name, default)


def rank_to_unit_interval(values: torch.Tensor) -> torch.Tensor:
    """Return deterministic percentile ranks in [0, 1], using 0.5 for NaNs and singleton batches."""

    flat = values.detach().float().flatten()
    out = torch.full_like(flat, 0.5)
    finite_mask = torch.isfinite(flat)
    finite_values = flat[finite_mask]
    n = int(finite_values.numel())
    if n <= 1:
        return out.view_as(values)

    sorted_values, sorted_order = torch.sort(finite_values)
    finite_ranks = torch.empty_like(finite_values)
    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_values[end] == sorted_values[start]:
            end += 1
        midrank = (start + end - 1) / 2.0
        finite_ranks[sorted_order[start:end]] = midrank / float(n - 1)
        start = end
    out[finite_mask] = finite_ranks
    return out.view_as(values)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=values.device, dtype=values.dtype)
    denom = mask.sum(dim=-1).clamp(min=1.0)
    return (values * mask).sum(dim=-1) / denom


def compute_group_success_routing(
    *,
    token_level_scores: torch.Tensor,
    response_mask: torch.Tensor,
    uids: Sequence[object],
    config,
) -> GroupSuccessRoutingResult:
    """Route every rollout in a question group from the group's verifier success count."""

    if token_level_scores.shape != response_mask.shape:
        raise ValueError("token_level_scores and response_mask must have identical shape")
    if token_level_scores.dim() != 2:
        raise ValueError("token_level_scores and response_mask must be 2D tensors")
    if str(_config_get(config, "method", "group_success_prompt_esr")) != "group_success_prompt_esr":
        raise ValueError("difficulty_aware_opd.method must be 'group_success_prompt_esr'")

    uid_values = np.asarray(uids, dtype=object)
    if uid_values.ndim != 1:
        raise ValueError("uids must be a 1D sequence")
    batch_size = int(token_level_scores.shape[0])
    if len(uid_values) != batch_size:
        raise ValueError(f"uids length must equal batch size {batch_size}, got {len(uid_values)}")

    expected_group_size = int(_config_get(config, "expected_group_size", 4))
    if expected_group_size <= 0:
        raise ValueError("expected_group_size must be positive")
    easy_group_correct_count = int(_config_get(config, "easy_group_correct_count", expected_group_size))
    if not 1 <= easy_group_correct_count <= expected_group_size:
        raise ValueError("easy_group_correct_count must be in [1, expected_group_size]")

    easy_prompt_style = str(_config_get(config, "easy_prompt_style", "concise"))
    default_prompt_style = str(_config_get(config, "default_prompt_style", "normal"))
    if easy_prompt_style not in _ALLOWED_PROMPT_STYLES or default_prompt_style not in _ALLOWED_PROMPT_STYLES:
        raise ValueError("prompt style must be one of 'budget', 'normal', or 'concise'")

    easy_esr_beta = float(_config_get(config, "easy_esr_beta", 0.20))
    non_easy_esr_beta = float(_config_get(config, "non_easy_esr_beta", 0.50))
    if not 0.0 < easy_esr_beta <= 1.0:
        raise ValueError("easy_esr_beta must be in (0, 1]")
    if not 0.0 < non_easy_esr_beta <= 1.0:
        raise ValueError("non_easy_esr_beta must be in (0, 1]")

    groups: dict[object, list[int]] = {}
    for row_index, uid in enumerate(uid_values.tolist()):
        try:
            groups.setdefault(uid, []).append(row_index)
        except TypeError as exc:
            raise ValueError(f"uid values must be hashable, got {uid!r}") from exc

    sequence_rewards = token_level_scores.float().sum(dim=-1)
    correct_reward_threshold = float(_config_get(config, "correct_reward_threshold", 0.5))
    correct = (sequence_rewards > correct_reward_threshold).to(dtype=torch.float32)
    device = token_level_scores.device
    group_correct_count = torch.empty(batch_size, device=device, dtype=torch.float32)
    group_accuracy = torch.empty(batch_size, device=device, dtype=torch.float32)
    easy = torch.zeros(batch_size, device=device, dtype=torch.float32)
    learnable = torch.zeros(batch_size, device=device, dtype=torch.float32)
    unresolved = torch.zeros(batch_size, device=device, dtype=torch.float32)
    esr_beta = torch.full((batch_size,), non_easy_esr_beta, device=device, dtype=torch.float32)
    prompt_styles = np.full(batch_size, default_prompt_style, dtype=object)
    group_correct_counts: list[int] = []

    for uid, row_indices in groups.items():
        if len(row_indices) != expected_group_size:
            raise ValueError(
                f"group-success routing expected {expected_group_size} rows for uid={uid!r}, got {len(row_indices)}"
            )
        index = torch.tensor(row_indices, device=device, dtype=torch.long)
        correct_count = int(correct[index].sum().item())
        group_correct_counts.append(correct_count)
        group_correct_count[index] = float(correct_count)
        group_accuracy[index] = float(correct_count) / float(expected_group_size)
        if correct_count == easy_group_correct_count:
            easy[index] = 1.0
            esr_beta[index] = easy_esr_beta
            prompt_styles[row_indices] = easy_prompt_style
        elif correct_count == 0:
            unresolved[index] = 1.0
        else:
            learnable[index] = 1.0

    return GroupSuccessRoutingResult(
        correct=correct.detach(),
        group_correct_count=group_correct_count.detach(),
        group_accuracy=group_accuracy.detach(),
        easy=easy.detach(),
        learnable=learnable.detach(),
        unresolved=unresolved.detach(),
        esr_beta=esr_beta.detach(),
        prompt_styles=prompt_styles,
        sequence_rewards=sequence_rewards.detach(),
        group_correct_counts=torch.tensor(group_correct_counts, device=device, dtype=torch.long),
        expected_group_size=expected_group_size,
    )


def compute_two_signal_difficulty_routing(
    *,
    token_level_scores: torch.Tensor,
    old_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    config,
    base_esr_beta: float,
) -> DifficultyAwareRoutingResult:
    """Compute correctness-confidence easy/hard gates and per-sample routing decisions."""

    if token_level_scores.shape != old_log_probs.shape or response_mask.shape != old_log_probs.shape:
        raise ValueError("token_level_scores, old_log_probs, and response_mask must have identical shape")
    if str(_config_get(config, "method", "two_signal_prompt_esr_entropy")) != "two_signal_prompt_esr_entropy":
        raise ValueError("difficulty_aware_opd.method must be 'two_signal_prompt_esr_entropy'")

    easy_prompt_style = str(_config_get(config, "easy_prompt_style", "concise"))
    default_prompt_style = str(_config_get(config, "default_prompt_style", "budget"))
    if easy_prompt_style not in _ALLOWED_PROMPT_STYLES:
        raise ValueError(f"Invalid easy_prompt_style: {easy_prompt_style!r}")
    if default_prompt_style not in _ALLOWED_PROMPT_STYLES:
        raise ValueError(f"Invalid default_prompt_style: {default_prompt_style!r}")

    sequence_rewards = token_level_scores.float().sum(dim=-1)
    threshold = float(_config_get(config, "correct_reward_threshold", 0.5))
    correct = (sequence_rewards > threshold).to(dtype=torch.float32)

    student_logp_mean = _masked_mean(old_log_probs.float(), response_mask.float())
    confidence_rank = rank_to_unit_interval(student_logp_mean).to(device=old_log_probs.device, dtype=torch.float32)

    easy = correct * confidence_rank
    hard = (1.0 - correct) * (1.0 - confidence_rank)

    base = torch.full(easy.shape, float(base_esr_beta), device=easy.device, dtype=torch.float64)
    min_beta = float(_config_get(config, "min_easy_esr_beta", 0.10))
    delta = float(_config_get(config, "easy_esr_delta", 0.10))
    esr_beta = torch.clamp(base - delta * easy.to(dtype=torch.float64), min=min_beta, max=float(base_esr_beta))

    entropy_coef = float(_config_get(config, "hard_entropy_coef", 0.001))
    entropy_weight = hard * entropy_coef

    easy_threshold = float(_config_get(config, "easy_prompt_threshold", 0.7))
    prompt_styles = np.array(
        [easy_prompt_style if float(value) >= easy_threshold else default_prompt_style for value in easy.detach().cpu()],
        dtype=object,
    )

    return DifficultyAwareRoutingResult(
        correct=correct.detach(),
        confidence_rank=confidence_rank.detach(),
        easy=easy.detach(),
        hard=hard.detach(),
        esr_beta=esr_beta.detach(),
        entropy_weight=entropy_weight.detach(),
        prompt_styles=prompt_styles,
        sequence_rewards=sequence_rewards.detach(),
        student_logp_mean=student_logp_mean.detach(),
        easy_prompt_threshold=easy_threshold,
    )


def _safe_mean(values: torch.Tensor) -> float:
    return float(values.float().mean().item()) if values.numel() else 0.0


def summarize_group_success_routing(
    result: GroupSuccessRoutingResult,
    original_response_lengths: torch.Tensor | None = None,
) -> dict[str, float]:
    """Summarize group-level difficulty buckets and their broadcast per-row routes."""

    group_count = int(result.group_correct_counts.numel())
    prompt_styles: Sequence[object] = result.prompt_styles.tolist()
    row_count = max(len(prompt_styles), 1)
    metrics = {
        "difficulty_aware_opd/enabled": 1.0,
        "difficulty_aware_opd/group_count": float(group_count),
        "difficulty_aware_opd/expected_group_size": float(result.expected_group_size),
        "difficulty_aware_opd/correct_rate": _safe_mean(result.correct),
        "difficulty_aware_opd/easy_group_ratio": _safe_mean(result.easy),
        "difficulty_aware_opd/learnable_group_ratio": _safe_mean(result.learnable),
        "difficulty_aware_opd/unresolved_group_ratio": _safe_mean(result.unresolved),
        "difficulty_aware_opd/concise_prompt_ratio": sum(style == "concise" for style in prompt_styles) / row_count,
        "difficulty_aware_opd/normal_prompt_ratio": sum(style == "normal" for style in prompt_styles) / row_count,
        "difficulty_aware_opd/esr_beta_mean": _safe_mean(result.esr_beta),
        "difficulty_aware_opd/esr_beta_min": float(result.esr_beta.min().item()) if result.esr_beta.numel() else 0.0,
        "difficulty_aware_opd/esr_beta_max": float(result.esr_beta.max().item()) if result.esr_beta.numel() else 0.0,
    }
    for correct_count in range(result.expected_group_size + 1):
        metrics[f"difficulty_aware_opd/group_correct_count_{correct_count}_ratio"] = _safe_mean(
            (result.group_correct_counts == correct_count).float()
        )

    if original_response_lengths is not None:
        lengths = original_response_lengths.float()
        if lengths.dim() != 1 or lengths.shape[0] != result.correct.shape[0]:
            raise ValueError("original_response_lengths must have shape [batch]")
        metrics["difficulty_aware_opd/orig_response_length_mean"] = _safe_mean(lengths)
        for name, mask in (
            ("easy", result.easy.bool()),
            ("learnable", result.learnable.bool()),
            ("unresolved", result.unresolved.bool()),
        ):
            metrics[f"difficulty_aware_opd/{name}_orig_response_length_mean"] = (
                _safe_mean(lengths[mask]) if mask.any() else 0.0
            )
    return metrics


def summarize_difficulty_routing(
    result: DifficultyAwareRoutingResult,
    original_response_lengths: torch.Tensor | None = None,
) -> dict[str, float]:
    """Summarize routing gates and actions for trainer logging."""

    prompt_styles: Sequence[object] = result.prompt_styles.tolist()
    total = max(len(prompt_styles), 1)
    concise_count = sum(style == "concise" for style in prompt_styles)
    budget_count = sum(style == "budget" for style in prompt_styles)
    wrong = 1.0 - result.correct
    high_conf = result.confidence_rank >= result.easy_prompt_threshold
    low_conf = result.confidence_rank <= (1.0 - result.easy_prompt_threshold)
    metrics = {
        "difficulty_aware_opd/enabled": 1.0,
        "difficulty_aware_opd/correct_rate": _safe_mean(result.correct),
        "difficulty_aware_opd/confidence_rank_mean": _safe_mean(result.confidence_rank),
        "difficulty_aware_opd/confidence_rank_correct_mean": _safe_mean(result.confidence_rank[result.correct.bool()])
        if result.correct.bool().any()
        else 0.0,
        "difficulty_aware_opd/confidence_rank_wrong_mean": _safe_mean(result.confidence_rank[wrong.bool()])
        if wrong.bool().any()
        else 0.0,
        "difficulty_aware_opd/easy_mean": _safe_mean(result.easy),
        "difficulty_aware_opd/easy_max": float(result.easy.float().max().item()) if result.easy.numel() else 0.0,
        "difficulty_aware_opd/hard_mean": _safe_mean(result.hard),
        "difficulty_aware_opd/hard_max": float(result.hard.float().max().item()) if result.hard.numel() else 0.0,
        "difficulty_aware_opd/concise_prompt_ratio": concise_count / total,
        "difficulty_aware_opd/budget_prompt_ratio": budget_count / total,
        "difficulty_aware_opd/esr_beta_mean": _safe_mean(result.esr_beta),
        "difficulty_aware_opd/esr_beta_min": float(result.esr_beta.float().min().item()) if result.esr_beta.numel() else 0.0,
        "difficulty_aware_opd/esr_beta_max": float(result.esr_beta.float().max().item()) if result.esr_beta.numel() else 0.0,
        "difficulty_aware_opd/hard_entropy_weight_mean": _safe_mean(result.entropy_weight),
        "difficulty_aware_opd/wrong_high_conf_ratio": _safe_mean((wrong.bool() & high_conf).float()),
        "difficulty_aware_opd/correct_low_conf_ratio": _safe_mean((result.correct.bool() & low_conf).float()),
    }
    if original_response_lengths is not None:
        metrics["difficulty_aware_opd/orig_response_length_mean"] = _safe_mean(original_response_lengths.float())
    return metrics
