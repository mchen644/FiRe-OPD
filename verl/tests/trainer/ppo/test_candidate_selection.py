from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.candidate_selection import select_short_correct_candidates


def _cfg(**overrides):
    values = {
        "enabled": True,
        "method": "shortest_correct_else_teacher",
        "correct_reward_threshold": 0.5,
        "keep_per_uid": 1,
        "teacher_reject_percentile": 20.0,
        "drop_rejected_no_correct": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _batch_for_selection():
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],  # uid-a, correct but long
            [1, 1, 1, 0, 0, 0],  # uid-a, correct and short -> select
            [1, 1, 0, 0, 0, 0],  # uid-b, incorrect, teacher worse
            [1, 1, 1, 1, 0, 0],  # uid-b, incorrect, teacher better -> select
        ],
        dtype=torch.float32,
    )
    ref_log_prob = torch.tensor(
        [
            [-0.2, -0.2, -0.2, -0.2, -0.2, -0.2],
            [-0.5, -0.5, -0.5, 0.0, 0.0, 0.0],
            [-3.0, -3.0, 0.0, 0.0, 0.0, 0.0],
            [-0.3, -0.3, -0.3, -0.3, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    token_level_scores = torch.zeros_like(response_mask)
    token_level_scores[0, 5] = 1.0
    token_level_scores[1, 2] = 1.0
    input_ids = torch.arange(4 * 6, dtype=torch.long).view(4, 6)
    batch = DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "response_mask": response_mask,
            "ref_log_prob": ref_log_prob,
            "token_level_scores": token_level_scores,
        },
        non_tensors={
            "uid": np.array(["uid-a", "uid-a", "uid-b", "uid-b"], dtype=object),
            "source": np.array(["a0", "a1", "b0", "b1"], dtype=object),
        },
    )
    return batch


def _batch_for_quality_gated_selection():
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],  # uid-a, correct but long
            [1, 1, 1, 0, 0, 0],  # uid-a, correct and short -> select
            [1, 1, 0, 0, 0, 0],  # uid-b, wrong, teacher accepted and short -> select
            [1, 1, 1, 1, 1, 0],  # uid-b, wrong, teacher accepted and longer
            [1, 1, 1, 0, 0, 0],  # uid-c, wrong, teacher rejected
            [1, 1, 1, 1, 0, 0],  # uid-c, wrong, teacher rejected -> group dropped
        ],
        dtype=torch.float32,
    )
    ref_log_prob = torch.tensor(
        [
            [-0.2, -0.2, -0.2, -0.2, -0.2, -0.2],
            [-0.5, -0.5, -0.5, 0.0, 0.0, 0.0],
            [-0.2, -0.2, 0.0, 0.0, 0.0, 0.0],
            [-0.1, -0.1, -0.1, -0.1, -0.1, 0.0],
            [-10.0, -10.0, -10.0, 0.0, 0.0, 0.0],
            [-9.0, -9.0, -9.0, -9.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    token_level_scores = torch.zeros_like(response_mask)
    token_level_scores[0, 5] = 1.0
    token_level_scores[1, 2] = 1.0
    input_ids = torch.arange(6 * 6, dtype=torch.long).view(6, 6)
    batch = DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "response_mask": response_mask,
            "ref_log_prob": ref_log_prob,
            "token_level_scores": token_level_scores,
        },
        non_tensors={
            "uid": np.array(["uid-a", "uid-a", "uid-b", "uid-b", "uid-c", "uid-c"], dtype=object),
            "source": np.array(["a0", "a1", "b0", "b1", "c0", "c1"], dtype=object),
        },
    )
    return batch


def test_selects_shortest_correct_candidate_and_teacher_best_fallback():
    batch = _batch_for_selection()

    selected, metrics = select_short_correct_candidates(batch, _cfg())

    assert selected.batch["input_ids"].tolist() == [
        [6, 7, 8, 9, 10, 11],
        [18, 19, 20, 21, 22, 23],
    ]
    assert selected.non_tensor_batch["uid"].tolist() == ["uid-a", "uid-b"]
    assert selected.non_tensor_batch["source"].tolist() == ["a1", "b1"]
    assert metrics["candidate_selection/enabled"] == 1.0
    assert metrics["candidate_selection/groups"] == 2.0
    assert metrics["candidate_selection/candidates"] == 4.0
    assert metrics["candidate_selection/keep_ratio"] == 0.5
    assert metrics["candidate_selection/any_correct_ratio"] == 0.5
    assert metrics["candidate_selection/selected_correct_ratio"] == 0.5
    assert metrics["candidate_selection/fallback_teacher_ratio"] == 0.5
    assert metrics["candidate_selection/selected_response_len_mean"] == 3.5
    assert metrics["candidate_selection/candidate_response_len_mean"] == 3.75


def test_quality_gated_correct_compression_selects_correct_compresses_wrong_fallback_and_drops_rejected():
    batch = _batch_for_quality_gated_selection()

    selected, metrics = select_short_correct_candidates(
        batch,
        _cfg(method="quality_gated_correct_compression", teacher_reject_percentile=20.0),
    )

    assert selected.batch["input_ids"].tolist() == [
        [6, 7, 8, 9, 10, 11],
        [12, 13, 14, 15, 16, 17],
    ]
    assert selected.non_tensor_batch["uid"].tolist() == ["uid-a", "uid-b"]
    assert selected.non_tensor_batch["source"].tolist() == ["a1", "b0"]

    assert metrics["candidate_selection/enabled"] == 1.0
    assert metrics["candidate_selection/groups"] == 3.0
    assert metrics["candidate_selection/selected_groups"] == 2.0
    assert metrics["candidate_selection/candidates"] == 6.0
    assert metrics["candidate_selection/keep_ratio"] == pytest.approx(2.0 / 6.0)
    assert metrics["candidate_selection/any_correct_ratio"] == pytest.approx(1.0 / 3.0)
    assert metrics["candidate_selection/no_correct_ratio"] == pytest.approx(2.0 / 3.0)
    assert metrics["candidate_selection/selected_correct_ratio"] == 0.5
    assert metrics["candidate_selection/fallback_teacher_ratio"] == pytest.approx(1.0 / 3.0)
    assert metrics["candidate_selection/no_correct_teacher_accept_ratio"] == pytest.approx(1.0 / 3.0)
    assert metrics["candidate_selection/dropped_uid_ratio"] == pytest.approx(1.0 / 3.0)
    assert metrics["candidate_selection/selected_response_len_mean"] == 2.5
    assert metrics["candidate_selection/selected_correct_len_mean"] == 3.0
    assert metrics["candidate_selection/selected_wrong_len_mean"] == 2.0
    assert metrics["candidate_selection/correct_candidate_len_mean"] == 4.5
    assert metrics["candidate_selection/wrong_candidate_len_mean"] == 3.5
    assert metrics["candidate_selection/teacher_accept_ratio"] == pytest.approx(4.0 / 6.0)
    assert metrics["candidate_selection/teacher_reject_threshold"] < -1.0


def test_quality_gated_correct_compression_can_keep_teacher_best_when_drop_disabled():
    batch = _batch_for_quality_gated_selection()

    selected, metrics = select_short_correct_candidates(
        batch,
        _cfg(
            method="quality_gated_correct_compression",
            teacher_reject_percentile=20.0,
            drop_rejected_no_correct=False,
        ),
    )

    assert selected.non_tensor_batch["uid"].tolist() == ["uid-a", "uid-b", "uid-c"]
    assert selected.non_tensor_batch["source"].tolist() == ["a1", "b0", "c1"]
    assert metrics["candidate_selection/groups"] == 3.0
    assert metrics["candidate_selection/selected_groups"] == 3.0
    assert metrics["candidate_selection/dropped_uid_ratio"] == 0.0
    assert metrics["candidate_selection/fallback_teacher_ratio"] == pytest.approx(2.0 / 3.0)


def test_disabled_selection_returns_original_batch_and_disabled_metric():
    batch = _batch_for_selection()

    selected, metrics = select_short_correct_candidates(batch, _cfg(enabled=False))

    assert selected is batch
    assert len(selected) == 4
    assert metrics == {"candidate_selection/enabled": 0.0}


def test_selection_rejects_unsupported_keep_per_uid_and_method():
    batch = _batch_for_selection()

    with pytest.raises(ValueError, match="keep_per_uid=1"):
        select_short_correct_candidates(batch, _cfg(keep_per_uid=2))

    with pytest.raises(ValueError, match="Unsupported candidate_selection.method"):
        select_short_correct_candidates(batch, _cfg(method="shortest_only"))


def test_selection_requires_uid_ref_log_prob_response_mask_and_scores():
    batch = _batch_for_selection()

    no_uid = DataProto.from_dict(
        tensors={key: value.clone() for key, value in batch.batch.items()},
        non_tensors={"source": batch.non_tensor_batch["source"].copy()},
    )
    with pytest.raises(ValueError, match="uid"):
        select_short_correct_candidates(no_uid, _cfg())

    missing_ref = _batch_for_selection()
    missing_ref.batch.pop("ref_log_prob")
    with pytest.raises(ValueError, match="ref_log_prob"):
        select_short_correct_candidates(missing_ref, _cfg())

    missing_scores = _batch_for_selection()
    missing_scores.batch.pop("token_level_scores")
    with pytest.raises(ValueError, match="token_level_scores"):
        select_short_correct_candidates(missing_scores, _cfg())


def test_selection_handles_single_candidate_groups():
    response_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.float32)
    ref_log_prob = torch.full_like(response_mask, -0.2)
    token_level_scores = torch.zeros_like(response_mask)
    token_level_scores[0, 1] = 1.0
    batch = DataProto.from_dict(
        tensors={
            "response_mask": response_mask,
            "ref_log_prob": ref_log_prob,
            "token_level_scores": token_level_scores,
        },
        non_tensors={"uid": np.array(["u0", "u1"], dtype=object)},
    )

    selected, metrics = select_short_correct_candidates(batch, _cfg())

    assert len(selected) == 2
    assert metrics["candidate_selection/keep_ratio"] == 1.0
    assert metrics["candidate_selection/groups"] == 2.0


def test_selection_reduces_rollout_n_two_batch_to_one_per_uid():
    batch = _batch_for_selection()
    selected, metrics = select_short_correct_candidates(batch, _cfg())

    assert len(batch) == 4
    assert len(selected) == 2
    assert selected.non_tensor_batch["uid"].tolist() == ["uid-a", "uid-b"]
    assert metrics["candidate_selection/keep_ratio"] == 0.5
