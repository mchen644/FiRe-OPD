import inspect

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.config.algorithm import AdaptiveConciseOpdConfig
from verl.trainer.ppo import ray_trainer
from verl.trainer.ppo.adaptive_concise_opd import plan_adaptive_concise_probes
from verl.trainer.ppo.ray_trainer import (
    _apply_adaptive_concise_opd,
    _apply_adaptive_concise_teacher_prompts,
    _build_adaptive_concise_generation_batch,
    _remap_reward_to_supervised_prefix,
)


class FakeAdaptiveTokenizer:
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
        ]
    )
    reward_model = _object_array(
        [
            {"ground_truth": "0", "style": "rule"},
            {"ground_truth": "1", "style": "rule"},
            {"ground_truth": "2", "style": "rule"},
        ]
    )
    extra_info = _object_array(
        [
            {"index": 10, "split": "train"},
            {"index": 11, "split": "train"},
            {"index": 12, "split": "train"},
        ]
    )
    return DataProto.from_dict(
        tensors={
            "input_ids": torch.ones(3, 5, dtype=torch.long),
            "attention_mask": torch.ones(3, 5, dtype=torch.long),
            "position_ids": torch.arange(5).repeat(3, 1),
        },
        non_tensors={
            "raw_prompt": raw_prompts,
            "reward_model": reward_model,
            "extra_info": extra_info,
            "data_source": np.array(["DeepMath-103K"] * 3, dtype=object),
            "uid": np.array(["u0", "u1", "u2"], dtype=object),
        },
    )


def test_build_adaptive_concise_generation_batch_preserves_reward_metadata_and_caps() -> None:
    batch = _normal_batch()
    normal_reward = torch.zeros(3, 8)
    normal_reward[0, -1] = 1.0
    normal_reward[2, -1] = 1.0
    plan = plan_adaptive_concise_probes(
        normal_reward=normal_reward,
        normal_response_mask=torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 0, 0],
            ]
        ),
        correct_reward_threshold=0.5,
        concise_cap_ratio=0.5,
    )
    tokenizer = FakeAdaptiveTokenizer()

    diagnostic = _build_adaptive_concise_generation_batch(
        batch=batch,
        plan=plan,
        tokenizer=tokenizer,
        max_prompt_length=512,
        truncation="error",
        temperature=1.0,
        top_p=1.0,
        apply_chat_template_kwargs={"enable_thinking": False},
    )

    assert len(diagnostic) == 2
    assert diagnostic.non_tensor_batch["adaptive_concise_original_row"].tolist() == [0, 2]
    assert diagnostic.non_tensor_batch["uid"].tolist() == ["u0", "u2"]
    assert [row["ground_truth"] for row in diagnostic.non_tensor_batch["reward_model"]] == ["0", "2"]
    assert [row["index"] for row in diagnostic.non_tensor_batch["extra_info"]] == [10, 12]
    assert diagnostic.meta_info["generation_kwargs"] == {
        "max_tokens_by_row": [4, 3],
        "disable_rollout_log_probs": True,
        "temperature": 1.0,
        "top_p": 1.0,
    }
    assert diagnostic.meta_info["response_length"] == 4
    assert len(tokenizer.seen_messages) == 2
    assert "Solve concisely. Avoid unnecessary explanation." in tokenizer.seen_messages[0][0]["content"]
    assert "Please reason step by step" not in tokenizer.seen_messages[0][0]["content"]
    assert diagnostic.non_tensor_batch["adaptive_concise_prompt"].tolist() == tokenizer.seen_messages


def test_build_adaptive_concise_generation_batch_fails_instead_of_truncating_prompt() -> None:
    batch = _normal_batch()
    reward = torch.zeros(3, 8)
    reward[0, -1] = 1.0
    plan = plan_adaptive_concise_probes(
        normal_reward=reward,
        normal_response_mask=torch.ones(3, 8, dtype=torch.long),
        correct_reward_threshold=0.5,
        concise_cap_ratio=0.5,
    )
    with pytest.raises(ValueError, match="longer than"):
        _build_adaptive_concise_generation_batch(
            batch=batch,
            plan=plan,
            tokenizer=FakeAdaptiveTokenizer(),
            max_prompt_length=4,
            truncation="error",
            temperature=1.0,
            top_p=1.0,
            apply_chat_template_kwargs={"enable_thinking": False},
        )


def test_apply_teacher_prompts_changes_only_easy_rows() -> None:
    batch = _normal_batch()
    original_inputs = batch.batch["input_ids"].clone()

    _apply_adaptive_concise_teacher_prompts(
        batch=batch,
        prompt_styles=np.array(["normal", "normal", "concise"], dtype=object),
        teacher_prompt_key="teacher_prompt",
    )

    teacher_prompts = batch.non_tensor_batch["teacher_prompt"].tolist()
    assert teacher_prompts[0] == list(batch.non_tensor_batch["raw_prompt"][0])
    assert teacher_prompts[1] == list(batch.non_tensor_batch["raw_prompt"][1])
    assert "Solve concisely. Avoid unnecessary explanation." in teacher_prompts[2][0]["content"]
    assert torch.equal(batch.batch["input_ids"], original_inputs)


@pytest.mark.parametrize("styles", [["normal"], ["normal", "bad", "concise"]])
def test_apply_teacher_prompts_rejects_invalid_style_vectors(styles: list[str]) -> None:
    with pytest.raises(ValueError, match="prompt styles"):
        _apply_adaptive_concise_teacher_prompts(
            batch=_normal_batch(),
            prompt_styles=np.array(styles, dtype=object),
            teacher_prompt_key="teacher_prompt",
        )


def test_remap_reward_preserves_sequence_score_at_last_supervised_token() -> None:
    reward = torch.zeros(3, 8)
    reward[0, 7] = 1.0
    reward[1, 6] = -0.5
    reward[2, 5] = 0.25

    remapped = _remap_reward_to_supervised_prefix(
        reward_tensor=reward,
        supervised_lengths=torch.tensor([8, 5, 3]),
    )

    assert remapped.shape == (3, 8)
    assert remapped.sum(dim=-1).tolist() == pytest.approx([1.0, -0.5, 0.25])
    assert remapped[0, 7].item() == 1.0
    assert remapped[1, 4].item() == -0.5
    assert remapped[2, 2].item() == 0.25
    assert torch.count_nonzero(remapped).item() == 3


def test_remap_reward_rejects_invalid_supervision_lengths() -> None:
    with pytest.raises(ValueError, match="shape"):
        _remap_reward_to_supervised_prefix(
            reward_tensor=torch.zeros(2, 4),
            supervised_lengths=torch.tensor([2]),
        )
    with pytest.raises(ValueError, match="positive"):
        _remap_reward_to_supervised_prefix(
            reward_tensor=torch.zeros(2, 4),
            supervised_lengths=torch.tensor([2, 0]),
        )
    with pytest.raises(ValueError, match="width"):
        _remap_reward_to_supervised_prefix(
            reward_tensor=torch.zeros(2, 4),
            supervised_lengths=torch.tensor([2, 5]),
        )


class FakeAdaptiveRolloutWorker:
    def __init__(self) -> None:
        self.calls = []

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        self.calls.append(prompts)
        prompt_ids = prompts.batch["input_ids"]
        prompt_mask = prompts.batch["attention_mask"]
        prompt_positions = prompts.batch["position_ids"]
        responses = torch.tensor([[31, 0], [41, 42]], dtype=torch.long)
        response_attention = torch.tensor([[1, 0], [1, 1]], dtype=torch.long)
        input_ids = torch.cat([prompt_ids, responses], dim=-1)
        attention_mask = torch.cat([prompt_mask, response_attention], dim=-1)
        response_positions = prompt_positions[:, -1:] + torch.arange(1, 3).unsqueeze(0)
        position_ids = torch.cat([prompt_positions, response_positions], dim=-1)
        return DataProto.from_dict(
            tensors={
                "prompts": prompt_ids,
                "responses": responses,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            non_tensors=dict(prompts.non_tensor_batch),
            meta_info={"timing": {"generate_sequences": 0.01}},
        )


class FakeAdaptiveRewardManager:
    def __call__(self, data: DataProto, return_dict: bool = False):
        reward = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        for row, info in enumerate(data.non_tensor_batch["extra_info"]):
            if info["index"] == 12:
                valid_length = int(data.batch["attention_mask"][row, -reward.shape[-1] :].sum().item())
                reward[row, valid_length - 1] = 1.0
        result = {"reward_tensor": reward, "reward_extra_info": {}}
        return result if return_dict else reward


class FakeAdaptiveDecodeTokenizer(FakeAdaptiveTokenizer):
    def batch_decode(self, responses, skip_special_tokens=True):
        assert skip_special_tokens is True
        assert responses.shape == (2, 2)
        return ["concise but wrong", r"done: \\boxed{2}"]


def _normal_rollout_batch() -> DataProto:
    base = _normal_batch()
    raw_prompts = np.concatenate(
        [
            base.non_tensor_batch["raw_prompt"],
            _object_array([[{"role": "user", "content": "Problem three?"}]]),
        ]
    )
    reward_model = np.concatenate(
        [
            base.non_tensor_batch["reward_model"],
            _object_array([{"ground_truth": "3", "style": "rule"}]),
        ]
    )
    extra_info = np.concatenate(
        [
            base.non_tensor_batch["extra_info"],
            _object_array([{"index": 13, "split": "train"}]),
        ]
    )
    prompt_width = 2
    response_width = 4
    prompts = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    responses = torch.tensor(
        [[101, 102, 103, 104], [111, 112, 113, 114], [121, 122, 123, 124], [131, 132, 133, 134]]
    )
    response_mask = torch.ones(4, response_width, dtype=torch.long)
    attention_mask = torch.cat([torch.ones(4, prompt_width, dtype=torch.long), response_mask], dim=-1)
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
        meta_info={"global_token_num": [6, 6, 6, 6]},
    )


def test_apply_adaptive_concise_opd_generates_only_for_correct_rows_and_keeps_normal_batch() -> None:
    batch = _normal_rollout_batch()
    original_responses = batch.batch["responses"].clone()
    normal_reward = torch.zeros(4, 4)
    normal_reward[1, -1] = 1.0
    normal_reward[2, -1] = 1.0
    worker = FakeAdaptiveRolloutWorker()
    timing_raw = {}

    routed_batch, routed_reward, result, metrics = _apply_adaptive_concise_opd(
        batch=batch,
        normal_reward=normal_reward,
        actor_rollout_wg=worker,
        reward_fn=FakeAdaptiveRewardManager(),
        tokenizer=FakeAdaptiveDecodeTokenizer(),
        adaptive_config=AdaptiveConciseOpdConfig(
            enabled=True,
            expected_questions_per_step=4,
        ),
        max_prompt_length=512,
        truncation="error",
        apply_chat_template_kwargs={"enable_thinking": False},
        timing_raw=timing_raw,
    )

    assert len(worker.calls) == 1
    assert worker.calls[0].non_tensor_batch["adaptive_concise_original_row"].tolist() == [1, 2]
    assert worker.calls[0].meta_info["generation_kwargs"]["max_tokens_by_row"] == [2, 2]
    assert len(routed_batch) == 4
    assert routed_batch.batch["responses"][:, :2].tolist() == original_responses[:, :2].tolist()
    assert routed_batch.batch["response_mask"].tolist() == [
        [1, 1, 1, 1],
        [1, 1, 1, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 1],
    ]
    assert "adaptive_concise_original_row" not in routed_batch.non_tensor_batch
    assert result.hard.tolist() == [True, False, False, True]
    assert result.learnable.tolist() == [False, True, False, False]
    assert result.easy.tolist() == [False, False, True, False]
    assert result.supervised_lengths.tolist() == [4, 3, 2, 4]
    assert routed_reward.sum(dim=-1).tolist() == [0.0, 1.0, 1.0, 0.0]
    assert routed_reward[1, 2].item() == 1.0
    assert routed_reward[2, 1].item() == 1.0
    assert metrics["adaptive_concise_opd/total_questions"] == 4.0
    assert metrics["adaptive_concise_opd/response_budget_residual_tokens"] == 0.0
    assert metrics["adaptive_concise_opd/response_budget_ratio"] == 1.0
    assert timing_raw["concise_probe"] >= 0.0
    assert timing_raw["concise_reward"] >= 0.0


def test_apply_adaptive_concise_opd_rejects_duplicate_question_indices() -> None:
    batch = _normal_rollout_batch()
    batch.non_tensor_batch["extra_info"][3] = {"index": 12, "split": "train"}
    with pytest.raises(ValueError, match="distinct question"):
        _apply_adaptive_concise_opd(
            batch=batch,
            normal_reward=torch.zeros(4, 4),
            actor_rollout_wg=FakeAdaptiveRolloutWorker(),
            reward_fn=FakeAdaptiveRewardManager(),
            tokenizer=FakeAdaptiveDecodeTokenizer(),
            adaptive_config=AdaptiveConciseOpdConfig(enabled=True, expected_questions_per_step=4),
            max_prompt_length=512,
            truncation="error",
            apply_chat_template_kwargs={"enable_thinking": False},
            timing_raw={},
        )


def test_fit_source_routes_before_post_rollout_model_forwards() -> None:
    source = inspect.getsource(ray_trainer.RayPPOTrainer.fit)
    adaptive_call = "_apply_adaptive_concise_opd("
    old_log_prob_call = "compute_log_prob(batch)"
    ref_prepare_call = "prepare_ref_model_inputs("
    actor_update_call = "update_actor(batch)"
    assert adaptive_call in source
    assert source.index(adaptive_call) < source.index(old_log_prob_call)
    assert source.index(adaptive_call) < source.index(ref_prepare_call)
    assert source.index(adaptive_call) < source.index(actor_update_call)
    assert "adaptive_concise_seqlen" in source
