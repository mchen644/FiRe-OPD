import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.config.algorithm import AdaptiveConciseOpdConfig, AlgoConfig
from verl.trainer.ppo.adaptive_concise_opd import (
    finalize_adaptive_concise_routing,
    plan_adaptive_concise_probes,
    summarize_adaptive_concise_routing,
    truncate_to_adaptive_concise_prefix,
    update_adaptive_concise_cumulative_counts,
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


def test_adaptive_config_defaults_are_disabled_and_frozen() -> None:
    config = AdaptiveConciseOpdConfig()
    assert config.enabled is False
    assert config.correct_reward_threshold == 0.5
    assert config.concise_cap_ratio == 0.5
    assert config.teacher_prompt_key == "teacher_prompt"
    assert config.temperature == 1.0
    assert config.top_p == 1.0
    assert config.expected_questions_per_step == 1024
    assert AlgoConfig().adaptive_concise_opd.enabled is False


def test_probe_plan_only_selects_normal_correct_rows_and_caps_at_half() -> None:
    response_mask = _mask([8, 7, 6], width=8)
    plan = plan_adaptive_concise_probes(
        normal_reward=_normal_reward([True, False, True], width=8),
        normal_response_mask=response_mask,
        correct_reward_threshold=0.5,
        concise_cap_ratio=0.5,
    )

    assert plan.probe_indices.tolist() == [0, 2]
    assert plan.max_tokens.tolist() == [4, 3]
    assert plan.normal_correct.tolist() == [True, False, True]
    assert plan.hard.tolist() == [False, True, False]
    assert plan.normal_lengths.tolist() == [8, 7, 6]


@pytest.mark.parametrize(
    ("reward", "mask", "message"),
    [
        (torch.zeros(2, 3), torch.ones(2, 2), "identical shape"),
        (torch.tensor([[float("nan")]]), torch.ones(1, 1), "finite"),
        (torch.zeros(1, 2), torch.zeros(1, 2), "positive"),
    ],
)
def test_probe_plan_rejects_invalid_normal_inputs(
    reward: torch.Tensor, mask: torch.Tensor, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        plan_adaptive_concise_probes(
            normal_reward=reward,
            normal_response_mask=mask,
            correct_reward_threshold=0.5,
            concise_cap_ratio=0.5,
        )


def test_probe_plan_rejects_a_correct_response_too_short_to_probe() -> None:
    with pytest.raises(ValueError, match="at least two"):
        plan_adaptive_concise_probes(
            normal_reward=torch.tensor([[1.0]]),
            normal_response_mask=torch.tensor([[1]]),
            correct_reward_threshold=0.5,
            concise_cap_ratio=0.5,
        )


@pytest.mark.parametrize(
    ("threshold", "ratio", "message"),
    [(0.4, 0.5, "threshold"), (0.5, 0.4, "cap ratio")],
)
def test_probe_plan_rejects_changed_scientific_contract(
    threshold: float, ratio: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        plan_adaptive_concise_probes(
            normal_reward=_normal_reward([False], width=2),
            normal_response_mask=torch.ones(1, 2),
            correct_reward_threshold=threshold,
            concise_cap_ratio=ratio,
        )


def test_finalize_routes_hard_learnable_and_easy_with_exact_accounting() -> None:
    plan = plan_adaptive_concise_probes(
        normal_reward=_normal_reward([False, True, True], width=8),
        normal_response_mask=_mask([8, 8, 7], width=8),
        correct_reward_threshold=0.5,
        concise_cap_ratio=0.5,
    )
    concise_mask = _mask([3, 2], width=4)
    concise_reward = torch.zeros((2, 4), dtype=torch.float32)
    concise_reward[1, 1] = 1.0

    result = finalize_adaptive_concise_routing(
        plan=plan,
        concise_reward=concise_reward,
        concise_response_mask=concise_mask,
        concise_original_indices=np.array([1, 2]),
        concise_texts=["attempt without a box", r"answer: \boxed{1}"],
        correct_reward_threshold=0.5,
    )

    assert result.hard.tolist() == [True, False, False]
    assert result.learnable.tolist() == [False, True, False]
    assert result.easy.tolist() == [False, False, True]
    assert result.concise_lengths.tolist() == [0, 3, 2]
    assert result.supervised_lengths.tolist() == [8, 5, 5]
    assert result.prompt_styles.tolist() == ["normal", "normal", "concise"]
    assert result.concise_parse_fail_count == 1
    assert result.concise_cap_hit_count == 0
    assert torch.equal(
        result.concise_lengths + result.supervised_lengths,
        result.normal_lengths,
    )

    metrics = summarize_adaptive_concise_routing(result)
    assert metrics["adaptive_concise_opd/total_questions"] == 3.0
    assert metrics["adaptive_concise_opd/normal_correct_count"] == 2.0
    assert metrics["adaptive_concise_opd/normal_wrong_count"] == 1.0
    assert metrics["adaptive_concise_opd/concise_probe_count"] == 2.0
    assert metrics["adaptive_concise_opd/easy_count"] == 1.0
    assert metrics["adaptive_concise_opd/learnable_count"] == 1.0
    assert metrics["adaptive_concise_opd/hard_count"] == 1.0
    assert metrics["adaptive_concise_opd/concise_teacher_count"] == 1.0
    assert metrics["adaptive_concise_opd/normal_teacher_count"] == 2.0
    assert metrics["adaptive_concise_opd/normal_response_tokens"] == 23.0
    assert metrics["adaptive_concise_opd/concise_probe_tokens"] == 5.0
    assert metrics["adaptive_concise_opd/supervised_normal_tokens"] == 18.0
    assert metrics["adaptive_concise_opd/response_budget_residual_tokens"] == 0.0
    assert metrics["adaptive_concise_opd/response_budget_ratio"] == 1.0


def test_finalize_counts_exact_cap_hits() -> None:
    plan = plan_adaptive_concise_probes(
        normal_reward=_normal_reward([True], width=7),
        normal_response_mask=_mask([7], width=7),
        correct_reward_threshold=0.5,
        concise_cap_ratio=0.5,
    )
    result = finalize_adaptive_concise_routing(
        plan=plan,
        concise_reward=torch.tensor([[0.0, 0.0, 1.0]]),
        concise_response_mask=torch.ones(1, 3, dtype=torch.long),
        concise_original_indices=np.array([0]),
        concise_texts=[r"\boxed{1}"],
        correct_reward_threshold=0.5,
    )
    assert plan.max_tokens.tolist() == [3]
    assert result.concise_cap_hit_count == 1
    assert result.supervised_lengths.tolist() == [4]


def test_finalize_supports_an_empty_probe_batch() -> None:
    plan = plan_adaptive_concise_probes(
        normal_reward=_normal_reward([False, False], width=3),
        normal_response_mask=_mask([3, 2], width=3),
        correct_reward_threshold=0.5,
        concise_cap_ratio=0.5,
    )
    result = finalize_adaptive_concise_routing(
        plan=plan,
        concise_reward=torch.zeros(0, 0),
        concise_response_mask=torch.zeros(0, 0, dtype=torch.long),
        concise_original_indices=np.array([], dtype=np.int64),
        concise_texts=[],
        correct_reward_threshold=0.5,
    )
    assert result.hard.tolist() == [True, True]
    assert result.easy.sum().item() == 0
    assert result.learnable.sum().item() == 0
    assert result.concise_lengths.tolist() == [0, 0]
    assert result.supervised_lengths.tolist() == [3, 2]
    assert result.prompt_styles.tolist() == ["normal", "normal"]


@pytest.mark.parametrize(
    ("indices", "lengths", "reward_value", "message"),
    [
        (np.array([1, 1]), [2, 2], 0.0, "mapping"),
        (np.array([1]), [2], 0.0, "mapping"),
        (np.array([1, 2]), [5, 2], 0.0, "cap"),
        (np.array([1, 2]), [2, 2], float("nan"), "finite"),
    ],
)
def test_finalize_rejects_invalid_diagnostics(
    indices: np.ndarray, lengths: list[int], reward_value: float, message: str
) -> None:
    plan = plan_adaptive_concise_probes(
        normal_reward=_normal_reward([False, True, True], width=8),
        normal_response_mask=_mask([8, 8, 7], width=8),
        correct_reward_threshold=0.5,
        concise_cap_ratio=0.5,
    )
    width = max(lengths)
    reward = torch.zeros((len(lengths), width), dtype=torch.float32)
    if reward.numel():
        reward[0, 0] = reward_value
    with pytest.raises(ValueError, match=message):
        finalize_adaptive_concise_routing(
            plan=plan,
            concise_reward=reward,
            concise_response_mask=_mask(lengths, width=width),
            concise_original_indices=indices,
            concise_texts=[r"\boxed{0}"] * len(lengths),
            correct_reward_threshold=0.5,
        )


def test_cumulative_counts_are_immutable_and_add_route_events() -> None:
    plan = plan_adaptive_concise_probes(
        normal_reward=_normal_reward([False, True], width=4),
        normal_response_mask=_mask([4, 4], width=4),
        correct_reward_threshold=0.5,
        concise_cap_ratio=0.5,
    )
    result = finalize_adaptive_concise_routing(
        plan=plan,
        concise_reward=torch.tensor([[0.0, 1.0]]),
        concise_response_mask=torch.ones(1, 2, dtype=torch.long),
        concise_original_indices=np.array([1]),
        concise_texts=[r"\boxed{1}"],
        correct_reward_threshold=0.5,
    )
    original = {"total_questions": 10, "concise_probe_count": 4, "easy_count": 2, "learnable_count": 2, "hard_count": 6}
    updated, metrics = update_adaptive_concise_cumulative_counts(original, result)
    assert original["total_questions"] == 10
    assert updated == {
        "total_questions": 12,
        "concise_probe_count": 5,
        "easy_count": 3,
        "learnable_count": 2,
        "hard_count": 7,
    }
    assert metrics["adaptive_concise_opd/total_questions_cumulative"] == 12.0
    assert metrics["adaptive_concise_opd/easy_count_cumulative"] == 3.0


def test_truncate_to_adaptive_prefix_slices_response_aligned_and_sequence_tensors() -> None:
    batch_size = 2
    prompt_width = 2
    response_width = 8
    prompts = torch.tensor([[10, 11], [20, 21]])
    responses = torch.arange(batch_size * response_width).reshape(batch_size, response_width)
    response_mask = _mask([8, 7], width=response_width)
    input_ids = torch.cat([prompts, responses], dim=-1)
    attention_mask = torch.cat([torch.ones(batch_size, prompt_width, dtype=torch.long), response_mask], dim=-1)
    position_ids = torch.arange(prompt_width + response_width).repeat(batch_size, 1)
    batch = DataProto.from_dict(
        tensors={
            "prompts": prompts,
            "responses": responses,
            "response_mask": response_mask,
            "rollout_log_probs": torch.ones(batch_size, response_width),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "route_code": torch.tensor([1, 2]),
        },
        non_tensors={"uid": np.array(["a", "b"], dtype=object)},
        meta_info={"global_token_num": [10, 9]},
    )

    truncated = truncate_to_adaptive_concise_prefix(
        batch=batch,
        supervised_lengths=torch.tensor([5, 3]),
    )

    assert truncated.batch["responses"].shape == (2, 5)
    assert truncated.batch["rollout_log_probs"].shape == (2, 5)
    assert truncated.batch["input_ids"].shape == (2, prompt_width + 5)
    assert truncated.batch["attention_mask"].shape == (2, prompt_width + 5)
    assert truncated.batch["position_ids"].shape == (2, prompt_width + 5)
    assert truncated.batch["response_mask"].tolist() == [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]
    assert truncated.batch["attention_mask"].tolist() == [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 0],
    ]
    assert torch.equal(truncated.batch["prompts"], prompts)
    assert truncated.batch["route_code"].tolist() == [1, 2]
    assert truncated.non_tensor_batch["uid"].tolist() == ["a", "b"]
    assert truncated.meta_info["global_token_num"] == [7, 5]


@pytest.mark.parametrize(
    ("supervised", "message"),
    [([0, 3], "positive"), ([9, 3], "exceed"), ([5], "shape")],
)
def test_truncate_to_adaptive_prefix_rejects_invalid_lengths(
    supervised: list[int], message: str
) -> None:
    response_mask = _mask([8, 7], width=8)
    batch = DataProto.from_dict(
        tensors={
            "responses": torch.ones(2, 8, dtype=torch.long),
            "response_mask": response_mask,
            "input_ids": torch.ones(2, 10, dtype=torch.long),
            "attention_mask": torch.cat([torch.ones(2, 2, dtype=torch.long), response_mask], dim=-1),
            "position_ids": torch.arange(10).repeat(2, 1),
        }
    )
    with pytest.raises(ValueError, match=message):
        truncate_to_adaptive_concise_prefix(
            batch=batch,
            supervised_lengths=torch.tensor(supervised),
        )
