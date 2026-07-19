import inspect
from dataclasses import replace

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.config.algorithm import AdaptiveTriPromptOpdConfig
from verl.trainer.ppo import ray_trainer
from verl.trainer.ppo.adaptive_triprompt_opd import (
    finalize_adaptive_triprompt_routing,
    plan_adaptive_triprompt_probes,
)
from verl.trainer.ppo.ray_trainer import (
    _apply_adaptive_triprompt_opd,
    _apply_adaptive_triprompt_teacher_prompts,
    _assert_no_adaptive_triprompt_diagnostic_keys,
    _build_adaptive_triprompt_generation_batch,
)


class FakeTriPromptTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def __init__(self) -> None:
        self.seen_messages = []

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kwargs):
        assert add_generation_prompt is True
        assert tokenize is False
        assert kwargs == {"enable_thinking": False}
        self.seen_messages.append(messages)
        return messages[0]["content"] + "\n<GEN>"

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        assert isinstance(text, str)
        assert return_tensors == "pt"
        assert add_special_tokens is False
        ids = [(ord(char) % 89) + 1 for char in text]
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
        }

    def batch_decode(self, responses, skip_special_tokens=True):
        assert skip_special_tokens is True
        assert responses.shape == (2, 4)
        return ["concise but wrong", r"done: \boxed{2}"]


def _object_array(values):
    output = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        output[index] = value
    return output


def _normal_batch() -> DataProto:
    raw_prompts = _object_array(
        [
            [
                {
                    "role": "user",
                    "content": "Problem zero?\nPlease reason step by step, and put your final answer within \\boxed{}.",
                }
            ],
            [{"role": "user", "content": "Problem one?"}],
            [{"role": "user", "content": "Problem two?"}],
            [{"role": "user", "content": "Problem three?"}],
        ]
    )
    reward_model = _object_array(
        [
            {"ground_truth": str(index), "style": "rule"}
            for index in range(4)
        ]
    )
    extra_info = _object_array(
        [{"index": 10 + index, "split": "train"} for index in range(4)]
    )
    prompt_width = 2
    response_width = 4
    prompts = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    responses = torch.tensor(
        [
            [101, 102, 103, 104],
            [111, 112, 99, 0],
            [121, 122, 123, 99],
            [131, 99, 0, 0],
        ]
    )
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 0, 0],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.cat(
        [torch.ones(4, prompt_width, dtype=torch.long), response_mask], dim=-1
    )
    return DataProto.from_dict(
        tensors={
            "prompts": prompts,
            "responses": responses,
            "response_mask": response_mask,
            "rollout_log_probs": torch.zeros(4, response_width),
            "input_ids": torch.cat([prompts, responses], dim=-1),
            "attention_mask": attention_mask,
            "position_ids": torch.arange(prompt_width + response_width).repeat(4, 1),
        },
        non_tensors={
            "raw_prompt": raw_prompts,
            "reward_model": reward_model,
            "extra_info": extra_info,
            "data_source": np.array(["DeepMath-103K"] * 4, dtype=object),
            "uid": np.array(["u0", "u1", "u2", "u3"], dtype=object),
        },
        meta_info={"global_token_num": [6, 5, 6, 4]},
    )


def _normal_reward() -> torch.Tensor:
    reward = torch.zeros(4, 4)
    reward[1, 2] = 1.0
    reward[2, 3] = 1.0
    return reward


def _plan():
    batch = _normal_batch()
    return plan_adaptive_triprompt_probes(
        normal_reward=_normal_reward(),
        normal_response_mask=batch.batch["response_mask"],
        correct_reward_threshold=0.5,
    )


def _routing_result():
    return finalize_adaptive_triprompt_routing(
        plan=_plan(),
        concise_reward=torch.tensor(
            [[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        ),
        concise_response_mask=torch.tensor(
            [[1, 1, 1, 0], [1, 1, 1, 0]], dtype=torch.long
        ),
        concise_original_indices=np.array([1, 2], dtype=np.int64),
        concise_texts=["wrong", r"\boxed{2}"],
        correct_reward_threshold=0.5,
        max_response_length=4,
    )


def test_build_diagnostic_batch_uses_one_global_full_cap_without_row_caps() -> None:
    batch = _normal_batch()
    tokenizer = FakeTriPromptTokenizer()

    diagnostic = _build_adaptive_triprompt_generation_batch(
        batch=batch,
        plan=_plan(),
        tokenizer=tokenizer,
        max_prompt_length=512,
        truncation="error",
        max_response_length=16384,
        temperature=1.0,
        top_p=1.0,
        apply_chat_template_kwargs={"enable_thinking": False},
    )

    assert len(diagnostic) == 2
    assert diagnostic.non_tensor_batch["adaptive_triprompt_original_row"].tolist() == [1, 2]
    assert diagnostic.non_tensor_batch["uid"].tolist() == ["u1", "u2"]
    assert [row["ground_truth"] for row in diagnostic.non_tensor_batch["reward_model"]] == ["1", "2"]
    assert [row["index"] for row in diagnostic.non_tensor_batch["extra_info"]] == [11, 12]
    assert diagnostic.meta_info["response_length"] == 16384
    assert diagnostic.meta_info["generation_kwargs"] == {
        "disable_rollout_log_probs": True,
        "temperature": 1.0,
        "top_p": 1.0,
    }
    assert "max_tokens_by_row" not in diagnostic.meta_info["generation_kwargs"]
    assert len(tokenizer.seen_messages) == 2
    assert "Solve concisely. Avoid unnecessary explanation." in tokenizer.seen_messages[0][0]["content"]
    assert diagnostic.non_tensor_batch["adaptive_triprompt_prompt"].tolist() == tokenizer.seen_messages


def test_build_diagnostic_batch_fails_instead_of_truncating_prompt() -> None:
    with pytest.raises(ValueError, match="longer than"):
        _build_adaptive_triprompt_generation_batch(
            batch=_normal_batch(),
            plan=_plan(),
            tokenizer=FakeTriPromptTokenizer(),
            max_prompt_length=4,
            truncation="error",
            max_response_length=16384,
            temperature=1.0,
            top_p=1.0,
            apply_chat_template_kwargs={"enable_thinking": False},
        )


def test_apply_teacher_prompts_routes_normal_budget_and_concise_only() -> None:
    batch = _normal_batch()
    original_data_ptr = batch.batch["responses"].data_ptr()
    result = _routing_result()

    _apply_adaptive_triprompt_teacher_prompts(
        batch=batch,
        result=result,
        teacher_prompt_key="teacher_prompt",
    )

    prompts = batch.non_tensor_batch["teacher_prompt"].tolist()
    assert prompts[0] == list(batch.non_tensor_batch["raw_prompt"][0])
    assert "use less than 3 tokens" in prompts[1][0]["content"]
    assert "Solve concisely" in prompts[2][0]["content"]
    assert prompts[3] == list(batch.non_tensor_batch["raw_prompt"][3])
    assert "Please reason step by step" not in prompts[1][0]["content"]
    assert "Please reason step by step" not in prompts[2][0]["content"]
    assert batch.batch["responses"].data_ptr() == original_data_ptr


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda result: replace(result, teacher_prompt_styles=np.array(["normal"])), "styles"),
        (lambda result: replace(result, budgets=torch.tensor([0, 2, 0, 0])), "budget"),
        (
            lambda result: replace(
                result,
                teacher_prompt_styles=np.array(["normal", "bad", "concise", "normal"]),
            ),
            "styles",
        ),
    ],
)
def test_apply_teacher_prompts_rejects_invalid_routes(mutator, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _apply_adaptive_triprompt_teacher_prompts(
            batch=_normal_batch(),
            result=mutator(_routing_result()),
            teacher_prompt_key="teacher_prompt",
        )


class FakeTriPromptRolloutWorker:
    def __init__(self) -> None:
        self.calls = []

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        self.calls.append(prompts)
        prompt_ids = prompts.batch["input_ids"]
        prompt_mask = prompts.batch["attention_mask"]
        prompt_positions = prompts.batch["position_ids"]
        responses = torch.tensor([[31, 32, 33, 0], [41, 42, 99, 0]], dtype=torch.long)
        response_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=torch.long)
        response_positions = prompt_positions[:, -1:] + torch.arange(1, 5).unsqueeze(0)
        return DataProto.from_dict(
            tensors={
                "prompts": prompt_ids,
                "responses": responses,
                "response_mask": response_mask,
                "input_ids": torch.cat([prompt_ids, responses], dim=-1),
                "attention_mask": torch.cat([prompt_mask, response_mask], dim=-1),
                "position_ids": torch.cat([prompt_positions, response_positions], dim=-1),
            },
            non_tensors=dict(prompts.non_tensor_batch),
            meta_info={"timing": {"generate_sequences": 0.01}},
        )


class FakeTriPromptRewardManager:
    def __call__(self, data: DataProto, return_dict: bool = False):
        reward = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        for row, info in enumerate(data.non_tensor_batch["extra_info"]):
            if info["index"] == 12:
                reward[row, 2] = 1.0
        result = {"reward_tensor": reward, "reward_extra_info": {}}
        return result if return_dict else reward


def test_apply_triprompt_opd_keeps_full_primary_batch_and_discards_diagnostics() -> None:
    batch = _normal_batch()
    reward = _normal_reward()
    original_tensors = {
        key: (batch.batch[key].data_ptr(), batch.batch[key].clone())
        for key in ("responses", "response_mask", "input_ids", "attention_mask", "position_ids")
    }
    original_reward_ptr = reward.data_ptr()
    worker = FakeTriPromptRolloutWorker()
    timing_raw = {}

    routed_batch, routed_reward, result, metrics = _apply_adaptive_triprompt_opd(
        batch=batch,
        normal_reward=reward,
        actor_rollout_wg=worker,
        reward_fn=FakeTriPromptRewardManager(),
        tokenizer=FakeTriPromptTokenizer(),
        triprompt_config=AdaptiveTriPromptOpdConfig(
            enabled=True,
            diagnostic_max_response_length=4,
            expected_questions_per_step=4,
        ),
        max_prompt_length=512,
        truncation="error",
        apply_chat_template_kwargs={"enable_thinking": False},
        timing_raw=timing_raw,
    )

    assert routed_batch is batch
    assert routed_reward is reward
    assert routed_reward.data_ptr() == original_reward_ptr
    assert len(worker.calls) == 1
    assert worker.calls[0].non_tensor_batch["adaptive_triprompt_original_row"].tolist() == [1, 2]
    for key, (data_ptr, expected) in original_tensors.items():
        assert routed_batch.batch[key].data_ptr() == data_ptr
        assert torch.equal(routed_batch.batch[key], expected)
    assert result.hard.tolist() == [True, False, False, True]
    assert result.sensitive.tolist() == [False, True, False, False]
    assert result.easy.tolist() == [False, False, True, False]
    assert result.actor_supervised_lengths.tolist() == [4, 3, 4, 2]
    assert result.budgets.tolist() == [0, 3, 0, 0]
    assert routed_batch.non_tensor_batch["teacher_prompt"][1][0]["content"].find(
        "use less than 3 tokens"
    ) >= 0
    assert "Solve concisely" in routed_batch.non_tensor_batch["teacher_prompt"][2][0]["content"]
    assert "adaptive_triprompt_original_row" not in routed_batch.non_tensor_batch
    assert metrics["adaptive_triprompt_opd/normal_response_tokens"] == 13.0
    assert metrics["adaptive_triprompt_opd/actor_supervised_tokens"] == 13.0
    assert metrics["adaptive_triprompt_opd/supervision_token_residual"] == 0.0
    assert metrics["adaptive_triprompt_opd/full_response_preservation_ratio"] == 1.0
    assert metrics["adaptive_triprompt_opd/normal_eos_count"] == 3.0
    assert metrics["adaptive_triprompt_opd/normal_cap_hit_count"] == 2.0
    assert metrics["adaptive_triprompt_opd/concise_eos_count"] == 1.0
    assert metrics["adaptive_triprompt_opd/concise_cap_hit_count"] == 0.0
    assert timing_raw["concise_probe"] >= 0.0
    assert timing_raw["concise_reward"] >= 0.0
    assert timing_raw["triprompt_routing"] >= 0.0


def test_apply_triprompt_opd_rejects_duplicate_question_indices() -> None:
    batch = _normal_batch()
    batch.non_tensor_batch["extra_info"][3] = {"index": 12, "split": "train"}
    with pytest.raises(ValueError, match="distinct question"):
        _apply_adaptive_triprompt_opd(
            batch=batch,
            normal_reward=torch.zeros(4, 4),
            actor_rollout_wg=FakeTriPromptRolloutWorker(),
            reward_fn=FakeTriPromptRewardManager(),
            tokenizer=FakeTriPromptTokenizer(),
            triprompt_config=AdaptiveTriPromptOpdConfig(
                enabled=True,
                diagnostic_max_response_length=4,
                expected_questions_per_step=4,
            ),
            max_prompt_length=512,
            truncation="error",
            apply_chat_template_kwargs={"enable_thinking": False},
            timing_raw={},
        )


def test_diagnostic_leak_guard_rejects_tensor_and_non_tensor_keys() -> None:
    tensor_leak = _normal_batch()
    tensor_leak.batch["adaptive_triprompt_diagnostic_scores"] = torch.zeros(4, 4)
    with pytest.raises(ValueError, match="diagnostic leakage"):
        _assert_no_adaptive_triprompt_diagnostic_keys(tensor_leak)

    non_tensor_leak = _normal_batch()
    non_tensor_leak.non_tensor_batch["adaptive_triprompt_original_row"] = np.arange(4)
    with pytest.raises(ValueError, match="diagnostic leakage"):
        _assert_no_adaptive_triprompt_diagnostic_keys(non_tensor_leak)


def test_fit_source_routes_before_post_rollout_model_forwards() -> None:
    source = inspect.getsource(ray_trainer.RayPPOTrainer.fit)
    triprompt_call = "_apply_adaptive_triprompt_opd("
    leak_guard_call = "_assert_no_adaptive_triprompt_diagnostic_keys(batch)"
    old_log_prob_call = "compute_log_prob(batch)"
    ref_prepare_call = "prepare_ref_model_inputs("
    actor_update_call = "update_actor(batch)"

    assert triprompt_call in source
    assert source.index(triprompt_call) < source.index(leak_guard_call)
    assert source.index(leak_guard_call) < source.index(old_log_prob_call)
    assert source.index(triprompt_call) < source.index(ref_prepare_call)
    assert source.index(triprompt_call) < source.index(actor_update_call)
    assert "adaptive_triprompt_cumulative" in source
    assert "triprompt_routing_result" in source
    assert "adaptive_triprompt_seqlen" not in source
