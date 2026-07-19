import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.adaptive_triprompt_opd import (
    capture_primary_tensor_contract,
    finalize_adaptive_triprompt_routing,
    plan_adaptive_triprompt_probes,
    summarize_adaptive_triprompt_routing,
    summarize_adaptive_triprompt_token_statistics,
    summarize_response_endpoints,
    update_adaptive_triprompt_cumulative_counts,
    verify_primary_tensor_contract,
)


def _normal_reward(correct: list[bool], width: int) -> torch.Tensor:
    reward = torch.zeros((len(correct), width), dtype=torch.float32)
    for row, is_correct in enumerate(correct):
        if is_correct:
            reward[row, width - 1] = 1.0
    return reward


def _mask(lengths: list[int], width: int) -> torch.Tensor:
    positions = torch.arange(width).unsqueeze(0)
    return (positions < torch.tensor(lengths).unsqueeze(1)).to(dtype=torch.long)


def _three_route_result():
    plan = plan_adaptive_triprompt_probes(
        normal_reward=_normal_reward([False, True, True], width=8),
        normal_response_mask=_mask([8, 7, 6], width=8),
        correct_reward_threshold=0.5,
    )
    concise_reward = torch.zeros((2, 4), dtype=torch.float32)
    concise_reward[1, 1] = 1.0
    return finalize_adaptive_triprompt_routing(
        plan=plan,
        concise_reward=concise_reward,
        concise_response_mask=_mask([3, 2], width=4),
        concise_original_indices=np.array([1, 2], dtype=np.int64),
        concise_texts=["wrong without a box", r"done: \boxed{2}"],
        correct_reward_threshold=0.5,
        max_response_length=4,
    )


def test_probe_plan_selects_only_normal_correct_without_per_row_caps() -> None:
    plan = plan_adaptive_triprompt_probes(
        normal_reward=_normal_reward([True, False, True], width=8),
        normal_response_mask=_mask([8, 7, 6], width=8),
        correct_reward_threshold=0.5,
    )

    assert plan.probe_indices.tolist() == [0, 2]
    assert plan.normal_correct.tolist() == [True, False, True]
    assert plan.hard.tolist() == [False, True, False]
    assert plan.normal_lengths.tolist() == [8, 7, 6]
    assert not hasattr(plan, "max_tokens")


@pytest.mark.parametrize(
    ("reward", "mask", "threshold", "message"),
    [
        (torch.zeros(2, 3), torch.ones(2, 2), 0.5, "identical shape"),
        (torch.tensor([[float("nan")]]), torch.ones(1, 1), 0.5, "finite"),
        (torch.zeros(1, 2), torch.zeros(1, 2), 0.5, "positive"),
        (torch.zeros(1, 2), torch.tensor([[1, 0, 1]]), 0.5, "identical shape"),
        (torch.zeros(1, 3), torch.tensor([[1, 0, 1]]), 0.5, "contiguous"),
        (torch.zeros(1, 2), torch.ones(1, 2), 0.4, "threshold"),
    ],
)
def test_probe_plan_rejects_invalid_inputs(
    reward: torch.Tensor,
    mask: torch.Tensor,
    threshold: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        plan_adaptive_triprompt_probes(
            normal_reward=reward,
            normal_response_mask=mask,
            correct_reward_threshold=threshold,
        )


def test_finalize_routes_hard_sensitive_and_easy_with_full_supervision() -> None:
    result = _three_route_result()

    assert result.hard.tolist() == [True, False, False]
    assert result.sensitive.tolist() == [False, True, False]
    assert result.easy.tolist() == [False, False, True]
    assert result.teacher_prompt_styles.tolist() == ["normal", "budget", "concise"]
    assert result.normal_lengths.tolist() == [8, 7, 6]
    assert result.concise_lengths.tolist() == [0, 3, 2]
    assert result.actor_supervised_lengths.tolist() == [8, 7, 6]
    assert result.budgets.tolist() == [0, 7, 0]
    assert result.concise_parse_fail_count == 1
    assert result.concise_cap_hit_count == 0


def test_finalize_counts_global_cap_hits_without_relative_caps() -> None:
    plan = plan_adaptive_triprompt_probes(
        normal_reward=_normal_reward([True], width=3),
        normal_response_mask=torch.ones(1, 3, dtype=torch.long),
        correct_reward_threshold=0.5,
    )
    result = finalize_adaptive_triprompt_routing(
        plan=plan,
        concise_reward=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        concise_response_mask=torch.ones(1, 4, dtype=torch.long),
        concise_original_indices=np.array([0], dtype=np.int64),
        concise_texts=[r"\boxed{1}"],
        correct_reward_threshold=0.5,
        max_response_length=4,
    )
    assert result.easy.tolist() == [True]
    assert result.concise_lengths.tolist() == [4]
    assert result.concise_cap_hit_count == 1
    assert result.actor_supervised_lengths.tolist() == [3]


def test_finalize_supports_empty_diagnostics_when_all_normal_rows_are_wrong() -> None:
    plan = plan_adaptive_triprompt_probes(
        normal_reward=_normal_reward([False, False], width=3),
        normal_response_mask=_mask([3, 2], width=3),
        correct_reward_threshold=0.5,
    )
    result = finalize_adaptive_triprompt_routing(
        plan=plan,
        concise_reward=torch.zeros(0, 0),
        concise_response_mask=torch.zeros(0, 0, dtype=torch.long),
        concise_original_indices=np.array([], dtype=np.int64),
        concise_texts=[],
        correct_reward_threshold=0.5,
        max_response_length=4,
    )

    assert result.hard.tolist() == [True, True]
    assert result.easy.sum().item() == 0
    assert result.sensitive.sum().item() == 0
    assert result.teacher_prompt_styles.tolist() == ["normal", "normal"]
    assert result.actor_supervised_lengths.tolist() == [3, 2]
    assert result.budgets.tolist() == [0, 0]


@pytest.mark.parametrize(
    ("indices", "lengths", "reward_value", "max_length", "message"),
    [
        (np.array([1, 1]), [2, 2], 0.0, 4, "mapping"),
        (np.array([1]), [2], 0.0, 4, "mapping"),
        (np.array([1, 2]), [5, 2], 0.0, 4, "exceeds"),
        (np.array([1, 2]), [2, 2], float("nan"), 4, "finite"),
        (np.array([1, 2]), [2, 2], 0.0, 0, "positive"),
    ],
)
def test_finalize_rejects_invalid_diagnostics(
    indices: np.ndarray,
    lengths: list[int],
    reward_value: float,
    max_length: int,
    message: str,
) -> None:
    plan = plan_adaptive_triprompt_probes(
        normal_reward=_normal_reward([False, True, True], width=8),
        normal_response_mask=_mask([8, 7, 6], width=8),
        correct_reward_threshold=0.5,
    )
    width = max(lengths)
    reward = torch.zeros((len(lengths), width), dtype=torch.float32)
    reward[0, 0] = reward_value
    with pytest.raises(ValueError, match=message):
        finalize_adaptive_triprompt_routing(
            plan=plan,
            concise_reward=reward,
            concise_response_mask=_mask(lengths, width=width),
            concise_original_indices=indices,
            concise_texts=[r"\boxed{0}"] * len(lengths),
            correct_reward_threshold=0.5,
            max_response_length=max_length,
        )


def test_route_metrics_preserve_all_normal_tokens_and_unit_weights() -> None:
    metrics = summarize_adaptive_triprompt_routing(_three_route_result())

    assert metrics["adaptive_triprompt_opd/total_questions"] == 3.0
    assert metrics["adaptive_triprompt_opd/normal_correct_count"] == 2.0
    assert metrics["adaptive_triprompt_opd/normal_wrong_count"] == 1.0
    assert metrics["adaptive_triprompt_opd/concise_probe_count"] == 2.0
    assert metrics["adaptive_triprompt_opd/easy_count"] == 1.0
    assert metrics["adaptive_triprompt_opd/sensitive_count"] == 1.0
    assert metrics["adaptive_triprompt_opd/hard_count"] == 1.0
    assert metrics["adaptive_triprompt_opd/concise_teacher_count"] == 1.0
    assert metrics["adaptive_triprompt_opd/budget_teacher_count"] == 1.0
    assert metrics["adaptive_triprompt_opd/normal_teacher_count"] == 1.0
    assert metrics["adaptive_triprompt_opd/normal_response_tokens"] == 21.0
    assert metrics["adaptive_triprompt_opd/actor_supervised_tokens"] == 21.0
    assert metrics["adaptive_triprompt_opd/supervision_token_residual"] == 0.0
    assert metrics["adaptive_triprompt_opd/full_response_preservation_ratio"] == 1.0
    assert metrics["adaptive_triprompt_opd/route_weight_min"] == 1.0
    assert metrics["adaptive_triprompt_opd/route_weight_max"] == 1.0
    assert metrics["adaptive_triprompt_opd/sensitive_budget_min"] == 7.0
    assert metrics["adaptive_triprompt_opd/sensitive_budget_mean"] == 7.0
    assert metrics["adaptive_triprompt_opd/sensitive_budget_max"] == 7.0


def test_token_statistics_are_finite_and_route_specific_on_full_responses() -> None:
    result = _three_route_result()
    response_mask = _mask([8, 7, 6], width=8)
    old_log_probs = torch.tensor([[-1.0] * 8, [-2.0] * 8, [-3.0] * 8])
    ref_log_probs = torch.tensor([[-1.5] * 8, [-2.5] * 8, [-3.5] * 8])
    actor_entropies = torch.tensor([[0.1] * 8, [0.2] * 8, [0.3] * 8])

    metrics = summarize_adaptive_triprompt_token_statistics(
        result=result,
        response_mask=response_mask,
        old_log_probs=old_log_probs,
        ref_log_probs=ref_log_probs,
        actor_entropies=actor_entropies,
    )

    assert metrics["adaptive_triprompt_opd/hard_old_log_prob_mean"] == pytest.approx(-1.0)
    assert metrics["adaptive_triprompt_opd/sensitive_old_log_prob_mean"] == pytest.approx(-2.0)
    assert metrics["adaptive_triprompt_opd/easy_old_log_prob_mean"] == pytest.approx(-3.0)
    assert metrics["adaptive_triprompt_opd/easy_ref_log_prob_mean"] == pytest.approx(-3.5)
    assert metrics["adaptive_triprompt_opd/sensitive_actor_entropy_mean"] == pytest.approx(0.2)
    assert metrics["adaptive_triprompt_opd/ref_minus_old_log_prob_mean"] == pytest.approx(-0.5)
    assert metrics["adaptive_triprompt_opd/old_log_prob_mean"] == pytest.approx(-40.0 / 21.0)


def test_token_statistics_reject_nonfinite_valid_tokens_but_ignore_padding() -> None:
    result = _three_route_result()
    response_mask = _mask([8, 7, 6], width=8)
    finite = torch.zeros(3, 8)
    padded_nan = finite.clone()
    padded_nan[1, 7] = torch.nan
    summarize_adaptive_triprompt_token_statistics(
        result=result,
        response_mask=response_mask,
        old_log_probs=padded_nan,
        ref_log_probs=finite,
        actor_entropies=finite,
    )

    valid_nan = finite.clone()
    valid_nan[1, 6] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        summarize_adaptive_triprompt_token_statistics(
            result=result,
            response_mask=response_mask,
            old_log_probs=valid_nan,
            ref_log_probs=finite,
            actor_entropies=finite,
        )


def test_cumulative_counts_are_immutable_and_preserve_route_identities() -> None:
    original = {
        "total_questions": 10,
        "concise_probe_count": 4,
        "easy_count": 2,
        "sensitive_count": 2,
        "hard_count": 6,
    }
    updated, metrics = update_adaptive_triprompt_cumulative_counts(
        original,
        _three_route_result(),
    )

    assert original["total_questions"] == 10
    assert updated == {
        "total_questions": 13,
        "concise_probe_count": 6,
        "easy_count": 3,
        "sensitive_count": 3,
        "hard_count": 7,
    }
    assert metrics["adaptive_triprompt_opd/total_questions_cumulative"] == 13.0
    assert metrics["adaptive_triprompt_opd/sensitive_count_cumulative"] == 3.0


def test_endpoint_metrics_count_natural_eos_and_global_cap_independently() -> None:
    response_ids = torch.tensor([[10, 11, 99, 0], [20, 21, 22, 23]])
    response_mask = _mask([3, 4], width=4)

    metrics = summarize_response_endpoints(
        response_ids=response_ids,
        response_mask=response_mask,
        eos_token_id=99,
        max_response_length=4,
        metric_prefix="adaptive_triprompt_opd/normal",
    )

    assert metrics["adaptive_triprompt_opd/normal_eos_count"] == 1.0
    assert metrics["adaptive_triprompt_opd/normal_eos_ratio"] == 0.5
    assert metrics["adaptive_triprompt_opd/normal_cap_hit_count"] == 1.0
    assert metrics["adaptive_triprompt_opd/normal_cap_hit_ratio"] == 0.5
    assert metrics["adaptive_triprompt_opd/normal_response_length_mean"] == 3.5


def test_endpoint_metrics_accept_all_configured_qwen_eos_ids() -> None:
    metrics = summarize_response_endpoints(
        response_ids=torch.tensor([[10, 151645], [20, 151643]]),
        response_mask=torch.ones(2, 2, dtype=torch.long),
        eos_token_id=[151645, 151643],
        max_response_length=4,
        metric_prefix="qwen",
    )

    assert metrics["qwen_eos_count"] == 2.0
    assert metrics["qwen_eos_ratio"] == 1.0


def test_endpoint_metrics_reject_eos_before_later_valid_tokens() -> None:
    with pytest.raises(ValueError, match="terminal"):
        summarize_response_endpoints(
            response_ids=torch.tensor([[10, 99, 11]]),
            response_mask=torch.ones(1, 3, dtype=torch.long),
            eos_token_id=99,
            max_response_length=4,
            metric_prefix="x",
        )


def _primary_batch() -> DataProto:
    prompts = torch.tensor([[1, 2], [3, 4]])
    responses = torch.tensor([[10, 11, 99], [20, 21, 22]])
    response_mask = torch.ones(2, 3, dtype=torch.long)
    return DataProto.from_dict(
        tensors={
            "prompts": prompts,
            "responses": responses,
            "response_mask": response_mask,
            "input_ids": torch.cat([prompts, responses], dim=-1),
            "attention_mask": torch.ones(2, 5, dtype=torch.long),
            "position_ids": torch.arange(5).repeat(2, 1),
        },
        non_tensors={"uid": np.array(["a", "b"], dtype=object)},
    )


def test_primary_tensor_contract_allows_non_tensor_prompt_addition() -> None:
    batch = _primary_batch()
    reward = torch.zeros(2, 3)
    snapshot = capture_primary_tensor_contract(batch=batch, reward_tensor=reward)
    batch.non_tensor_batch["teacher_prompt"] = np.array(["normal", "budget"], dtype=object)

    verify_primary_tensor_contract(
        snapshot=snapshot,
        batch=batch,
        reward_tensor=reward,
    )


@pytest.mark.parametrize("mutation", ["replace_responses", "mutate_mask", "replace_reward"])
def test_primary_tensor_contract_rejects_tensor_changes(mutation: str) -> None:
    batch = _primary_batch()
    reward = torch.zeros(2, 3)
    snapshot = capture_primary_tensor_contract(batch=batch, reward_tensor=reward)

    if mutation == "replace_responses":
        batch.batch["responses"] = batch.batch["responses"].clone()
    elif mutation == "mutate_mask":
        batch.batch["response_mask"][0, 0] = 0
    else:
        reward = reward.clone()

    with pytest.raises(ValueError, match="changed"):
        verify_primary_tensor_contract(
            snapshot=snapshot,
            batch=batch,
            reward_tensor=reward,
        )
