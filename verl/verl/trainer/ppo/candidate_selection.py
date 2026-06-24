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

_SUPPORTED_METHODS = {"shortest_correct_else_teacher"}


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


def _select_index_for_uid(
    candidate_indices: list[int],
    response_len: torch.Tensor,
    correct: torch.Tensor,
    normalized_teacher_logprob: torch.Tensor,
) -> tuple[int, bool]:
    correct_indices = [idx for idx in candidate_indices if bool(correct[idx].item())]
    if correct_indices:
        selected = max(
            correct_indices,
            key=lambda idx: (-float(response_len[idx].item()), float(normalized_teacher_logprob[idx].item())),
        )
        return selected, False

    selected = max(candidate_indices, key=lambda idx: float(normalized_teacher_logprob[idx].item()))
    return selected, True


def select_short_correct_candidates(batch: DataProto, selection_config) -> tuple[DataProto, dict[str, float]]:
    """Select one candidate per prompt uid for OPD training.

    The initial method keeps the shortest correct candidate when a group has any
    correct response, otherwise it falls back to the candidate with highest
    normalized teacher log-probability.
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

    uid_to_indices: OrderedDict[str, list[int]] = OrderedDict()
    for idx, uid in enumerate(batch.non_tensor_batch["uid"]):
        uid_to_indices.setdefault(str(uid), []).append(idx)

    selected_indices: list[int] = []
    fallback_teacher_count = 0
    any_correct_count = 0
    for indices in uid_to_indices.values():
        if any(bool(correct[idx].item()) for idx in indices):
            any_correct_count += 1
        selected, used_teacher_fallback = _select_index_for_uid(
            candidate_indices=indices,
            response_len=response_len,
            correct=correct,
            normalized_teacher_logprob=normalized_teacher_logprob,
        )
        selected_indices.append(selected)
        fallback_teacher_count += int(used_teacher_fallback)

    selected_index_tensor = torch.tensor(selected_indices, dtype=torch.long)
    selected = batch.select_idxs(selected_index_tensor)

    selected_response_len = response_len[selected_index_tensor]
    selected_correct = correct[selected_index_tensor].float()
    selected_teacher_logprob = normalized_teacher_logprob[selected_index_tensor]
    groups = len(selected_indices)
    candidates = len(batch)

    metrics = {
        "candidate_selection/enabled": 1.0,
        "candidate_selection/groups": float(groups),
        "candidate_selection/candidates": float(candidates),
        "candidate_selection/keep_ratio": float(groups / max(candidates, 1)),
        "candidate_selection/any_correct_ratio": float(any_correct_count / max(groups, 1)),
        "candidate_selection/selected_correct_ratio": selected_correct.mean().item() if groups else 0.0,
        "candidate_selection/candidate_response_len_mean": response_len.float().mean().item(),
        "candidate_selection/selected_response_len_mean": selected_response_len.float().mean().item() if groups else 0.0,
        "candidate_selection/fallback_teacher_ratio": float(fallback_teacher_count / max(groups, 1)),
        "candidate_selection/selected_teacher_logprob_mean": selected_teacher_logprob.float().mean().item()
        if groups
        else 0.0,
    }
    return selected, metrics
