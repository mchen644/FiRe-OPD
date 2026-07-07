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
    hard_prompt_style = str(_config_get(config, "hard_prompt_style", "normal"))
    if easy_prompt_style not in _ALLOWED_PROMPT_STYLES:
        raise ValueError(f"Invalid easy_prompt_style: {easy_prompt_style!r}")
    if default_prompt_style not in _ALLOWED_PROMPT_STYLES:
        raise ValueError(f"Invalid default_prompt_style: {default_prompt_style!r}")
    if hard_prompt_style not in _ALLOWED_PROMPT_STYLES:
        raise ValueError(f"Invalid hard_prompt_style: {hard_prompt_style!r}")

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
    hard_threshold = _config_get(config, "hard_prompt_threshold", None)
    hard_threshold = None if hard_threshold is None else float(hard_threshold)
    prompt_styles_list: list[str] = []
    for easy_value, hard_value in zip(easy.detach().cpu(), hard.detach().cpu(), strict=True):
        if hard_threshold is not None and float(hard_value) >= hard_threshold:
            prompt_styles_list.append(hard_prompt_style)
        elif float(easy_value) >= easy_threshold:
            prompt_styles_list.append(easy_prompt_style)
        else:
            prompt_styles_list.append(default_prompt_style)
    prompt_styles = np.array(prompt_styles_list, dtype=object)

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


def summarize_difficulty_routing(
    result: DifficultyAwareRoutingResult,
    original_response_lengths: torch.Tensor | None = None,
) -> dict[str, float]:
    """Summarize routing gates and actions for trainer logging."""

    prompt_styles: Sequence[object] = result.prompt_styles.tolist()
    total = max(len(prompt_styles), 1)
    concise_count = sum(style == "concise" for style in prompt_styles)
    budget_count = sum(style == "budget" for style in prompt_styles)
    normal_count = sum(style == "normal" for style in prompt_styles)
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
        "difficulty_aware_opd/normal_prompt_ratio": normal_count / total,
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
