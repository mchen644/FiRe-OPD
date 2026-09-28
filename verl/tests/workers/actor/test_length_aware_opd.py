from types import SimpleNamespace

import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
from verl.workers.actor.dp_actor import (
    _add_length_aware_opd_tensors,
    _apply_candidate_selection_loss_mask,
    _apply_length_aware_opd_penalty,
    _apply_tale_budget_esr_loss_mask,
    _compute_difficulty_aware_entropy_loss,
    _compute_length_aware_opd_tensors,
    _maybe_compute_rollout_corr_metrics,
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


class _ActorLossConfig(SimpleNamespace):
    def get(self, name, default=None):
        return getattr(self, name, default)


def test_difficulty_aware_entropy_loss_is_negative_and_weighted():
    entropy = torch.tensor([[0.5, 1.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float32)
    response_mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.float32)
    entropy_weight = torch.tensor([0.0, 0.01], dtype=torch.float32)

    loss, metrics = _compute_difficulty_aware_entropy_loss(
        entropy=entropy,
        response_mask=response_mask,
        entropy_weight=entropy_weight,
        loss_agg_mode="token-mean",
    )

    assert loss.item() < 0.0
    assert abs(metrics["difficulty_aware_opd/hard_entropy_weight_mean"] - 0.005) < 1e-8
    assert metrics["difficulty_aware_opd/hard_entropy_loss"] == loss.item()


def test_difficulty_aware_entropy_loss_returns_zero_without_valid_weight():
    entropy = torch.ones((2, 3), dtype=torch.float32)
    response_mask = torch.ones((2, 3), dtype=torch.float32)
    entropy_weight = torch.zeros(2, dtype=torch.float32)

    loss, metrics = _compute_difficulty_aware_entropy_loss(
        entropy=entropy,
        response_mask=response_mask,
        entropy_weight=entropy_weight,
        loss_agg_mode="token-mean",
    )

    assert loss.item() == 0.0
    assert metrics["difficulty_aware_opd/hard_entropy_weight_mean"] == 0.0


def test_candidate_selection_loss_mask_zeroes_masked_response_rows():
    response_mask = torch.tensor(
        [[1, 1, 1], [1, 1, 0], [1, 0, 0]],
        dtype=torch.float32,
    )
    model_inputs = {"candidate_selection_loss_mask": torch.tensor([1.0, 0.0, 1.0])}

    masked = _apply_candidate_selection_loss_mask(response_mask, model_inputs)

    assert masked.tolist() == [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]


def test_tale_budget_esr_loss_mask_combines_with_existing_loss_mask():
    response_mask = torch.tensor(
        [[1, 1, 1, 1], [1, 1, 0, 0]],
        dtype=torch.float32,
    )
    model_inputs = {
        "tale_budget_esr_loss_mask": torch.tensor(
            [[1, 1, 0, 0], [1, 0, 0, 0]],
            dtype=torch.float32,
        )
    }

    masked = _apply_tale_budget_esr_loss_mask(response_mask, model_inputs)

    assert masked.tolist() == [[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]


def test_rollout_corr_metrics_skip_fully_masked_candidate_microbatch():
    response_mask = torch.zeros((1, 3), dtype=torch.float32)
    log_prob = torch.zeros_like(response_mask)
    rollout_log_prob = torch.zeros_like(response_mask)

    metrics = _maybe_compute_rollout_corr_metrics(
        loss_mode="vanilla",
        log_prob=log_prob,
        rollout_log_prob=rollout_log_prob,
        response_mask=response_mask,
    )

    assert metrics == {"rollout_corr/skipped_no_valid_tokens": 1.0}


def test_rollout_corr_metrics_emit_skip_key_for_unmasked_microbatch(monkeypatch):
    response_mask = torch.ones((1, 3), dtype=torch.float32)
    log_prob = torch.zeros_like(response_mask)
    rollout_log_prob = torch.zeros_like(response_mask)

    def fake_compute_rollout_corr_metrics_from_logprobs(**kwargs):
        return {"rollout_corr/fake_gap": 0.25}

    import verl.trainer.ppo.rollout_corr_helper as helper

    monkeypatch.setattr(
        helper,
        "compute_rollout_corr_metrics_from_logprobs",
        fake_compute_rollout_corr_metrics_from_logprobs,
    )

    metrics = _maybe_compute_rollout_corr_metrics(
        loss_mode="vanilla",
        log_prob=log_prob,
        rollout_log_prob=rollout_log_prob,
        response_mask=response_mask,
    )

    assert metrics["rollout_corr/skipped_no_valid_tokens"] == 0.0
    assert metrics["rollout_corr/fake_gap"] == 0.25


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


def test_length_penalty_correct_gate_penalizes_long_correct_response_only():
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
    token_level_scores[0, 3] = 0.0
    token_level_scores[1, 7] = 1.0
    token_level_scores[2, 1] = 0.0

    tensors, metrics = _compute_length_aware_opd_tensors(
        response_mask=response_mask,
        ref_log_prob=ref_log_prob,
        token_level_scores=token_level_scores,
        policy_loss_config=_policy_loss_config(
            length_penalty_gate="correct",
            length_teacher_reject_percentile=0.0,
        ),
    )

    expected_base_penalty = torch.log(torch.tensor(8.0 / 4.0))
    assert torch.isclose(tensors["length_aware_opd_base_penalty"][1], expected_base_penalty)
    assert torch.isclose(tensors["length_aware_opd_applied_penalty"][1], expected_base_penalty)
    assert torch.isclose(tensors["length_aware_opd_penalty"][1], expected_base_penalty * 0.02)
    assert tensors["length_aware_opd_applied_penalty"][0].item() == 0.0
    assert tensors["length_aware_opd_applied_penalty"][2].item() == 0.0
    assert metrics["length_aware_opd/penalty_gate_ratio"] == pytest.approx(1.0 / 3.0)
    assert metrics["length_aware_opd/correct_penalty_ratio"] == pytest.approx(1.0 / 3.0)


def test_log_length_teacher_confidence_penalty_increases_with_correct_response_length():
    response_mask = torch.tensor(
        [
            [1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.float32,
    )
    ref_log_prob = torch.full_like(response_mask, -0.2)
    token_level_scores = torch.zeros_like(response_mask)
    token_level_scores[0, 1] = 1.0
    token_level_scores[1, 7] = 1.0
    token_level_scores[2, 7] = 0.0

    tensors, metrics = _compute_length_aware_opd_tensors(
        response_mask=response_mask,
        ref_log_prob=ref_log_prob,
        token_level_scores=token_level_scores,
        policy_loss_config=_policy_loss_config(
            length_penalty_type="log_length_teacher_confidence",
            length_penalty_gate="correct",
            length_teacher_reject_percentile=0.0,
            length_confidence_temperature=0.1,
        ),
    )

    assert tensors["length_aware_opd_base_penalty"][1] > tensors["length_aware_opd_base_penalty"][0]
    assert tensors["length_aware_opd_applied_penalty"][1] > tensors["length_aware_opd_applied_penalty"][0]
    assert tensors["length_aware_opd_applied_penalty"][2].item() == 0.0
    assert metrics["length_aware_opd/confidence_temperature"] == pytest.approx(0.1)


def test_log_length_teacher_confidence_penalty_increases_with_teacher_confidence():
    response_mask = torch.ones((2, 4), dtype=torch.float32)
    ref_log_prob = torch.tensor(
        [
            [-0.1, -0.1, -0.1, -0.1],
            [-1.0, -1.0, -1.0, -1.0],
        ],
        dtype=torch.float32,
    )
    token_level_scores = torch.zeros_like(response_mask)
    token_level_scores[:, -1] = 1.0

    tensors, _ = _compute_length_aware_opd_tensors(
        response_mask=response_mask,
        ref_log_prob=ref_log_prob,
        token_level_scores=token_level_scores,
        policy_loss_config=_policy_loss_config(
            length_penalty_type="log_length_teacher_confidence",
            length_penalty_gate="correct",
            length_teacher_reject_percentile=50.0,
            length_confidence_temperature=0.1,
        ),
    )

    assert torch.isclose(tensors["length_aware_opd_base_penalty"][0], tensors["length_aware_opd_base_penalty"][1])
    assert tensors["length_aware_opd_applied_penalty"][0] > tensors["length_aware_opd_applied_penalty"][1]
    assert tensors["length_aware_opd_confidence_weight"][0] > tensors["length_aware_opd_confidence_weight"][1]


def test_log_length_teacher_confidence_rejects_invalid_temperature():
    response_mask = torch.ones((2, 4), dtype=torch.float32)
    ref_log_prob = torch.full_like(response_mask, -0.2)
    token_level_scores = torch.zeros_like(response_mask)
    token_level_scores[:, -1] = 1.0

    with pytest.raises(ValueError, match="length_confidence_temperature must be positive"):
        _compute_length_aware_opd_tensors(
            response_mask=response_mask,
            ref_log_prob=ref_log_prob,
            token_level_scores=token_level_scores,
            policy_loss_config=_policy_loss_config(
                length_penalty_type="log_length_teacher_confidence",
                length_penalty_gate="correct",
                length_confidence_temperature=0.0,
            ),
        )


def test_length_penalty_correct_gate_requires_scores():
    response_mask = torch.ones((2, 4), dtype=torch.float32)
    ref_log_prob = torch.full_like(response_mask, -0.2)

    with pytest.raises(ValueError, match="token_level_scores is required"):
        _compute_length_aware_opd_tensors(
            response_mask=response_mask,
            ref_log_prob=ref_log_prob,
            token_level_scores=None,
            policy_loss_config=_policy_loss_config(length_penalty_gate="correct"),
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


def test_broadcast_penalty_adds_sequence_margin_under_single_sequence_token_mean():
    response_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.float32)
    old_log_prob = torch.zeros_like(response_mask)
    log_prob = torch.zeros_like(response_mask)
    advantages = torch.zeros_like(response_mask)
    scaled_penalty = torch.tensor([0.03], dtype=torch.float32)

    adjusted_advantages = _apply_length_aware_opd_penalty(
        advantages=advantages,
        model_inputs={"length_aware_opd_penalty": scaled_penalty},
    )
    pg_loss, _ = compute_policy_loss_vanilla(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=adjusted_advantages,
        response_mask=response_mask,
        loss_agg_mode="token-mean",
        config=_ActorLossConfig(clip_ratio=0.2, clip_ratio_low=0.2, clip_ratio_high=0.2, clip_ratio_c=3.0),
    )

    assert torch.isclose(pg_loss, scaled_penalty[0])


def test_apply_length_penalty_requires_precomputed_microbatch_tensor():
    with pytest.raises(ValueError, match="length_aware_opd_penalty missing"):
        _apply_length_aware_opd_penalty(
            advantages=torch.zeros((1, 4), dtype=torch.float32),
            model_inputs={},
        )
