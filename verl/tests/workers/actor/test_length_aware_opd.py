from types import SimpleNamespace

import pytest
import torch

from verl import DataProto
from verl.workers.actor.dp_actor import (
    _add_length_aware_opd_tensors,
    _compute_length_aware_opd_tensors,
)


def _policy_loss_config(**overrides):
    values = {
        "length_aware_opd": True,
        "length_penalty_coef": 0.02,
        "length_penalty_type": "log_batch_median",
        "length_penalty_gate": "incorrect_or_low_teacher",
        "length_correct_reward_threshold": 0.5,
        "length_teacher_reject_percentile": 20.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_length_penalty_penalizes_long_incorrect_response_only():
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    ref_log_prob = torch.full_like(response_mask, -0.2)
    token_level_scores = torch.zeros_like(response_mask)
    token_level_scores[0, 3] = 1.0
    token_level_scores[1, 7] = 0.0
    token_level_scores[2, 1] = 0.0

    tensors, metrics = _compute_length_aware_opd_tensors(
        response_mask=response_mask,
        ref_log_prob=ref_log_prob,
        token_level_scores=token_level_scores,
        policy_loss_config=_policy_loss_config(length_teacher_reject_percentile=0.0),
    )

    expected_base_penalty = torch.log(torch.tensor(8.0 / 4.0))
    assert torch.isclose(tensors["length_aware_opd_base_penalty"][1], expected_base_penalty)
    assert torch.isclose(tensors["length_aware_opd_applied_penalty"][1], expected_base_penalty)
    assert torch.isclose(tensors["length_aware_opd_penalty"][1], expected_base_penalty * 0.02)
    assert tensors["length_aware_opd_applied_penalty"][0].item() == 0.0
    assert tensors["length_aware_opd_applied_penalty"][2].item() == 0.0
    assert metrics["length_aware_opd/median_response_len"] == 4.0
    assert metrics["length_aware_opd/coef"] == 0.02


def test_length_penalty_skips_long_correct_teacher_accepted_response():
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    ref_log_prob = torch.full_like(response_mask, -0.2)
    token_level_scores = torch.zeros_like(response_mask)
    token_level_scores[0, 3] = 1.0
    token_level_scores[1, 7] = 1.0
    token_level_scores[2, 1] = 0.0

    tensors, _ = _compute_length_aware_opd_tensors(
        response_mask=response_mask,
        ref_log_prob=ref_log_prob,
        token_level_scores=token_level_scores,
        policy_loss_config=_policy_loss_config(length_teacher_reject_percentile=0.0),
    )

    assert tensors["length_aware_opd_base_penalty"][1].item() > 0.0
    assert tensors["length_aware_opd_applied_penalty"][1].item() == 0.0
    assert tensors["length_aware_opd_penalty"][1].item() == 0.0
    assert tensors["length_aware_opd_correct_mask"][1].item() == 1.0
    assert tensors["length_aware_opd_teacher_reject"][1].item() == 0.0


def test_length_penalty_penalizes_long_correct_low_teacher_response():
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    ref_log_prob = torch.tensor(
        [
            [-0.2, -0.2, -0.2, -0.2, 0, 0, 0, 0],
            [-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],
            [-0.1, -0.1, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    token_level_scores = torch.zeros_like(response_mask)
    token_level_scores[0, 3] = 1.0
    token_level_scores[1, 7] = 1.0
    token_level_scores[2, 1] = 1.0

    tensors, metrics = _compute_length_aware_opd_tensors(
        response_mask=response_mask,
        ref_log_prob=ref_log_prob,
        token_level_scores=token_level_scores,
        policy_loss_config=_policy_loss_config(length_teacher_reject_percentile=50.0),
    )

    assert tensors["length_aware_opd_teacher_reject"][1].item() == 1.0
    assert tensors["length_aware_opd_applied_penalty"][1].item() > 0.0
    assert metrics["length_aware_opd/teacher_reject_ratio"] > 0.0


def test_length_penalty_rejects_invalid_type_and_gate():
    response_mask = torch.ones((2, 4), dtype=torch.float32)
    ref_log_prob = torch.full_like(response_mask, -0.2)
    token_level_scores = torch.zeros_like(response_mask)

    with pytest.raises(ValueError, match="Invalid length_penalty_type"):
        _compute_length_aware_opd_tensors(
            response_mask=response_mask,
            ref_log_prob=ref_log_prob,
            token_level_scores=token_level_scores,
            policy_loss_config=_policy_loss_config(length_penalty_type="linear"),
        )

    with pytest.raises(ValueError, match="Invalid length_penalty_gate"):
        _compute_length_aware_opd_tensors(
            response_mask=response_mask,
            ref_log_prob=ref_log_prob,
            token_level_scores=token_level_scores,
            policy_loss_config=_policy_loss_config(length_penalty_gate="unknown"),
        )


def test_length_penalty_requires_scores_for_incorrect_gate():
    response_mask = torch.ones((2, 4), dtype=torch.float32)
    ref_log_prob = torch.full_like(response_mask, -0.2)

    with pytest.raises(ValueError, match="token_level_scores is required"):
        _compute_length_aware_opd_tensors(
            response_mask=response_mask,
            ref_log_prob=ref_log_prob,
            token_level_scores=None,
            policy_loss_config=_policy_loss_config(length_penalty_gate="incorrect_or_low_teacher"),
        )


def test_minibatch_precompute_survives_single_sequence_microbatch_split():
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    ref_log_prob = torch.full_like(response_mask, -0.2)
    token_level_scores = torch.zeros_like(response_mask)
    token_level_scores[1, 7] = 0.0
    batch = DataProto.from_dict(
        tensors={
            "response_mask": response_mask,
            "ref_log_prob": ref_log_prob,
            "token_level_scores": token_level_scores,
        }
    )

    metrics = _add_length_aware_opd_tensors(
        mini_batch=batch,
        policy_loss_config=_policy_loss_config(length_teacher_reject_percentile=0.0),
    )
    micro_batches = batch.split(1)

    assert metrics["length_aware_opd/median_response_len"] == 4.0
    assert micro_batches[1].batch["length_aware_opd_applied_penalty"].item() > 0.0
    assert micro_batches[1].batch["length_aware_opd_penalty"].item() > 0.0
