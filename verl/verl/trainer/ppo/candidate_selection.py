# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch

from verl import DataProto

_SUPPORTED_METHODS = {"shortest_correct_else_teacher", "quality_gated_correct_compression"}


def _config_get(config: Any, name: str, default: Any) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(name, default)
    return getattr(config, name, default)


def _require_batch_key(batch: DataProto, key: str) -> torch.Tensor:
    if batch.batch is None or key not in batch.batch.keys():
        raise ValueError(f"candidate selection requires batch['{key}']")
    return batch.batch[key]


def _candidate_selection_scores(batch: DataProto, correct_reward_threshold: float):
    response_mask = _require_batch_key(batch, "response_mask").float()
    ref_log_prob = _require_batch_key(batch, "ref_log_prob")
    token_level_scores = _require_batch_key(batch, "token_level_scores")

    response_len = response_mask.sum(dim=-1).clamp(min=1.0)
    seq_reward = (token_level_scores.float() * response_mask).sum(dim=-1)
    correct = seq_reward > float(correct_reward_threshold)
    normalized_teacher_logprob = (ref_log_prob.float() * response_mask).sum(dim=-1) / response_len
    return response_len, correct, normalized_teacher_logprob


def _teacher_accept_mask(normalized_teacher_logprob: torch.Tensor, teacher_reject_percentile: float):
    if teacher_reject_percentile < 0.0 or teacher_reject_percentile > 100.0:
        raise ValueError("candidate_selection.teacher_reject_percentile must be between 0.0 and 100.0")
    if teacher_reject_percentile == 0.0:
        threshold = torch.tensor(float("nan"), device=normalized_teacher_logprob.device)
        return torch.ones_like(normalized_teacher_logprob, dtype=torch.bool), threshold
    threshold = torch.quantile(normalized_teacher_logprob.float(), teacher_reject_percentile / 100.0)
    return normalized_teacher_logprob > threshold, threshold


def _select_shortest_with_teacher_tiebreak(
    candidate_indices: list[int],
    response_len: torch.Tensor,
    normalized_teacher_logprob: torch.Tensor,
) -> int:
    return max(
        candidate_indices,
        key=lambda idx: (-float(response_len[idx].item()), float(normalized_teacher_logprob[idx].item())),
    )


def _select_index_shortest_correct_else_teacher(
    candidate_indices: list[int],
    response_len: torch.Tensor,
    correct: torch.Tensor,
    normalized_teacher_logprob: torch.Tensor,
) -> tuple[int, bool, bool]:
    correct_indices = [idx for idx in candidate_indices if bool(correct[idx].item())]
    if correct_indices:
        return _select_shortest_with_teacher_tiebreak(correct_indices, response_len, normalized_teacher_logprob), False, False

    selected = max(candidate_indices, key=lambda idx: float(normalized_teacher_logprob[idx].item()))
    return selected, True, False


def _select_index_quality_gated_correct_compression(
    candidate_indices: list[int],
    response_len: torch.Tensor,
    correct: torch.Tensor,
    normalized_teacher_logprob: torch.Tensor,
    teacher_accepted: torch.Tensor,
    drop_rejected_no_correct: bool,
) -> tuple[int | None, bool, bool]:
    correct_indices = [idx for idx in candidate_indices if bool(correct[idx].item())]
    if correct_indices:
        selected = _select_shortest_with_teacher_tiebreak(correct_indices, response_len, normalized_teacher_logprob)
        return selected, False, False

    accepted_indices = [idx for idx in candidate_indices if bool(teacher_accepted[idx].item())]
    if accepted_indices:
        selected = _select_shortest_with_teacher_tiebreak(accepted_indices, response_len, normalized_teacher_logprob)
        return selected, True, False

    selected = max(candidate_indices, key=lambda idx: float(normalized_teacher_logprob[idx].item()))
    if drop_rejected_no_correct:
        return selected, False, True

    return selected, True, False


def _safe_mean(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return values.float().mean().item()


def select_short_correct_candidates(batch: DataProto, selection_config) -> tuple[DataProto, dict[str, float]]:
    """Select one candidate per prompt uid for OPD training.

    Supported methods:
    - shortest_correct_else_teacher: legacy behavior; shortest correct if any, otherwise teacher-best.
    - quality_gated_correct_compression: shortest correct if any; otherwise shortest teacher-accepted;
      optionally drop no-correct groups where all candidates are teacher-rejected.
    """
    enabled = bool(_config_get(selection_config, "enabled", False))
    if not enabled:
        return batch, {"candidate_selection/enabled": 0.0}

    method = _config_get(selection_config, "method", "shortest_correct_else_teacher")
    if method not in _SUPPORTED_METHODS:
        raise ValueError(f"Unsupported candidate_selection.method: {method}. Supported: {sorted(_SUPPORTED_METHODS)}")

    keep_per_uid = int(_config_get(selection_config, "keep_per_uid", 1))
    if keep_per_uid != 1:
        raise ValueError("candidate selection currently supports keep_per_uid=1 only")

    if "uid" not in batch.non_tensor_batch:
        raise ValueError("candidate selection requires non_tensor_batch['uid']")

    response_len, correct, normalized_teacher_logprob = _candidate_selection_scores(
        batch=batch,
        correct_reward_threshold=float(_config_get(selection_config, "correct_reward_threshold", 0.5)),
    )
    teacher_accepted, teacher_reject_threshold = _teacher_accept_mask(
        normalized_teacher_logprob=normalized_teacher_logprob,
        teacher_reject_percentile=float(_config_get(selection_config, "teacher_reject_percentile", 20.0)),
    )

    uid_to_indices: OrderedDict[str, list[int]] = OrderedDict()
    for idx, uid in enumerate(batch.non_tensor_batch["uid"]):
        uid_to_indices.setdefault(str(uid), []).append(idx)

    selected_indices: list[int] = []
    selected_loss_masks: list[float] = []
    fallback_teacher_count = 0
    any_correct_count = 0
    no_correct_count = 0
    no_correct_teacher_accept_count = 0
    dropped_uid_count = 0
    drop_rejected_no_correct = bool(_config_get(selection_config, "drop_rejected_no_correct", True))

    for indices in uid_to_indices.values():
        has_correct = any(bool(correct[idx].item()) for idx in indices)
        if has_correct:
            any_correct_count += 1
        else:
            no_correct_count += 1

        if method == "shortest_correct_else_teacher":
            selected, used_teacher_fallback, dropped = _select_index_shortest_correct_else_teacher(
                candidate_indices=indices,
                response_len=response_len,
                correct=correct,
                normalized_teacher_logprob=normalized_teacher_logprob,
            )
        else:
            has_teacher_accept = any(bool(teacher_accepted[idx].item()) for idx in indices)
            if (not has_correct) and has_teacher_accept:
                no_correct_teacher_accept_count += 1
            selected, used_teacher_fallback, dropped = _select_index_quality_gated_correct_compression(
                candidate_indices=indices,
                response_len=response_len,
                correct=correct,
                normalized_teacher_logprob=normalized_teacher_logprob,
                teacher_accepted=teacher_accepted,
                drop_rejected_no_correct=drop_rejected_no_correct,
            )

        fallback_teacher_count += int(used_teacher_fallback)
        dropped_uid_count += int(dropped)
        if selected is not None:
            selected_indices.append(selected)
            selected_loss_masks.append(0.0 if dropped else 1.0)

    selected_index_tensor = torch.tensor(selected_indices, dtype=torch.long)
    selected = batch.select_idxs(selected_index_tensor)
    selected_loss_mask_tensor = torch.tensor(
        selected_loss_masks,
        dtype=batch.batch["response_mask"].dtype,
        device=batch.batch["response_mask"].device,
    )
    selected.batch["candidate_selection_loss_mask"] = selected_loss_mask_tensor

    selected_response_len = response_len[selected_index_tensor]
    selected_correct = correct[selected_index_tensor].float()
    selected_teacher_logprob = normalized_teacher_logprob[selected_index_tensor]
    selected_correct_len = selected_response_len[selected_correct.bool()]
    selected_wrong_len = selected_response_len[~selected_correct.bool()]
    correct_candidate_len = response_len[correct]
    wrong_candidate_len = response_len[~correct]

    groups = len(uid_to_indices)
    selected_groups = len(selected_indices)
    candidates = len(batch)

    metrics = {
        "candidate_selection/enabled": 1.0,
        "candidate_selection/groups": float(groups),
        "candidate_selection/selected_groups": float(selected_groups),
        "candidate_selection/candidates": float(candidates),
        "candidate_selection/keep_ratio": float(selected_groups / max(candidates, 1)),
        "candidate_selection/any_correct_ratio": float(any_correct_count / max(groups, 1)),
        "candidate_selection/no_correct_ratio": float(no_correct_count / max(groups, 1)),
        "candidate_selection/selected_correct_ratio": _safe_mean(selected_correct),
        "candidate_selection/candidate_response_len_mean": response_len.float().mean().item(),
        "candidate_selection/selected_response_len_mean": _safe_mean(selected_response_len),
        "candidate_selection/fallback_teacher_ratio": float(fallback_teacher_count / max(groups, 1)),
        "candidate_selection/selected_teacher_logprob_mean": _safe_mean(selected_teacher_logprob),
        "candidate_selection/teacher_accept_ratio": teacher_accepted.float().mean().item(),
        "candidate_selection/no_correct_teacher_accept_ratio": float(no_correct_teacher_accept_count / max(groups, 1)),
        "candidate_selection/dropped_uid_ratio": float(dropped_uid_count / max(groups, 1)),
        "candidate_selection/loss_mask_mean": _safe_mean(selected_loss_mask_tensor),
        "candidate_selection/selected_correct_len_mean": _safe_mean(selected_correct_len),
        "candidate_selection/selected_wrong_len_mean": _safe_mean(selected_wrong_len),
        "candidate_selection/correct_candidate_len_mean": _safe_mean(correct_candidate_len),
        "candidate_selection/wrong_candidate_len_mean": _safe_mean(wrong_candidate_len),
        "candidate_selection/teacher_reject_threshold": float(teacher_reject_threshold.item()),
    }
    return selected, metrics
