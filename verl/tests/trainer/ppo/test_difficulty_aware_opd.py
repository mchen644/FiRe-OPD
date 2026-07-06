from types import SimpleNamespace

import numpy as np
import torch

from verl import DataProto

from verl.trainer.ppo.difficulty_aware_opd import (
    compute_two_signal_difficulty_routing,
    rank_to_unit_interval,
    summarize_difficulty_routing,
)


def _cfg(**overrides):
    values = {
        "correct_reward_threshold": 0.5,
        "easy_prompt_threshold": 0.7,
        "easy_prompt_style": "concise",
        "default_prompt_style": "budget",
        "min_easy_esr_beta": 0.10,
        "easy_esr_delta": 0.10,
        "hard_entropy_coef": 0.001,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_rank_to_unit_interval_handles_monotonic_ties_nan_and_singleton():
    values = torch.tensor([3.0, 1.0, 1.0, float("nan"), 5.0])

    ranks = rank_to_unit_interval(values)

    assert torch.isfinite(ranks).all()
    assert ranks[1].item() == ranks[2].item()
    assert ranks[3].item() == 0.5
    assert ranks[4].item() == 1.0
    assert rank_to_unit_interval(torch.tensor([7.0])).tolist() == [0.5]


def test_two_signal_routing_marks_correct_high_conf_easy_and_wrong_low_conf_hard():
    response_mask = torch.ones((4, 3), dtype=torch.float32)
    old_log_probs = torch.tensor(
        [
            [-0.05, -0.05, -0.05],  # highest confidence, correct -> easy
            [-3.0, -3.0, -3.0],  # low confidence, wrong -> hard
            [-2.0, -2.0, -2.0],  # low-ish confidence, correct -> not easy
            [-0.1, -0.1, -0.1],  # high confidence, wrong -> overconfident wrong
        ],
        dtype=torch.float32,
    )
    token_level_scores = torch.zeros_like(response_mask)
    token_level_scores[0, -1] = 1.0
    token_level_scores[2, -1] = 1.0

    result = compute_two_signal_difficulty_routing(
        token_level_scores=token_level_scores,
        old_log_probs=old_log_probs,
        response_mask=response_mask,
        config=_cfg(),
        base_esr_beta=0.20,
    )

    assert result.correct.tolist() == [1.0, 0.0, 1.0, 0.0]
    assert result.easy[0].item() > 0.7
    assert result.hard[1].item() > 0.7
    assert result.prompt_styles.tolist()[0] == "concise"
    assert result.prompt_styles.tolist()[1] == "budget"
    assert result.esr_beta[0].item() < 0.20
    assert result.esr_beta[0].item() >= 0.10
    assert result.esr_beta[1].item() == 0.20
    assert result.entropy_weight[1].item() > result.entropy_weight[0].item()
    assert result.entropy_weight[3].item() < result.entropy_weight[1].item()


def test_summarize_difficulty_routing_emits_proxy_diagnostics():
    response_mask = torch.ones((2, 2), dtype=torch.float32)
    old_log_probs = torch.tensor([[-0.1, -0.1], [-2.0, -2.0]], dtype=torch.float32)
    token_level_scores = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.float32)

    result = compute_two_signal_difficulty_routing(
        token_level_scores=token_level_scores,
        old_log_probs=old_log_probs,
        response_mask=response_mask,
        config=_cfg(),
        base_esr_beta=0.20,
    )
    metrics = summarize_difficulty_routing(result, original_response_lengths=torch.tensor([10, 20]))

    assert metrics["difficulty_aware_opd/correct_rate"] == 0.5
    assert metrics["difficulty_aware_opd/concise_prompt_ratio"] == 0.5
    assert metrics["difficulty_aware_opd/budget_prompt_ratio"] == 0.5
    assert metrics["difficulty_aware_opd/orig_response_length_mean"] == 15.0
    assert "difficulty_aware_opd/wrong_high_conf_ratio" in metrics
    assert "difficulty_aware_opd/correct_low_conf_ratio" in metrics


def test_apply_difficulty_aware_opd_routing_adds_batch_tensors_and_metrics():
    from verl.trainer.ppo.ray_trainer import _apply_difficulty_aware_opd_routing

    response_mask = torch.ones((2, 3), dtype=torch.float32)
    old_log_probs = torch.tensor([[-0.1, -0.1, -0.1], [-2.0, -2.0, -2.0]], dtype=torch.float32)
    reward_tensor = torch.zeros_like(response_mask)
    reward_tensor[0, -1] = 1.0
    batch = DataProto.from_dict(
        tensors={
            "response_mask": response_mask,
            "old_log_probs": old_log_probs,
        }
    )
    metrics = {}

    _apply_difficulty_aware_opd_routing(
        batch=batch,
        reward_tensor=reward_tensor,
        difficulty_config=_cfg(),
        base_esr_beta=0.20,
        metrics=metrics,
    )

    assert batch.batch["difficulty_aware_correct"].tolist() == [1.0, 0.0]
    assert batch.non_tensor_batch["difficulty_aware_prompt_style"].tolist() == ["concise", "budget"]
    assert batch.batch["difficulty_aware_esr_beta"][0].item() < 0.20
    assert batch.batch["difficulty_aware_entropy_weight"][1].item() > 0.0
    assert metrics["difficulty_aware_opd/correct_rate"] == 0.5
    assert metrics["difficulty_aware_opd/concise_prompt_ratio"] == 0.5
