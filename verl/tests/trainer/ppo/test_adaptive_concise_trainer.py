import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.adaptive_concise_opd import plan_adaptive_concise_probes
from verl.trainer.ppo.ray_trainer import (
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
