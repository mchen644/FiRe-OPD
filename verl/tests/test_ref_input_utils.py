import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.ref_input_utils import prepare_ref_model_inputs


class FakeTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.seen_messages = []

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kwargs):
        self.seen_messages.append(messages)
        assert add_generation_prompt is True
        assert tokenize is False
        return "|".join(message["content"] for message in messages) + "|GEN"

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        assert return_tensors == "pt"
        assert add_special_tokens is False
        ids = [(ord(char) % 97) + 1 for char in text]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


def test_prepare_ref_model_inputs_can_use_configured_teacher_prompt_key():
    student_prompt = [{"role": "user", "content": "student normal prompt"}]
    teacher_prompt = [{"role": "user", "content": "teacher cod prompt"}]
    batch = DataProto.from_dict(
        tensors={
            "input_ids": torch.tensor([[11, 12, 13]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            "position_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
            "responses": torch.tensor([[21, 22]], dtype=torch.long),
            "response_mask": torch.tensor([[1, 1]], dtype=torch.long),
        },
        non_tensors={
            "raw_prompt": np.array([student_prompt], dtype=object),
            "teacher_prompt": np.array([teacher_prompt], dtype=object),
        },
    )
    tokenizer = FakeTokenizer()

    prepared = prepare_ref_model_inputs(
        batch=batch,
        ref_tokenizer=tokenizer,
        raw_prompt_key="teacher_prompt",
    )

    assert tokenizer.seen_messages == [teacher_prompt]
    expected_prompt_ids = tokenizer("teacher cod prompt|GEN")["input_ids"][0]
    assert torch.equal(prepared.batch["ref_input_ids"][0, : len(expected_prompt_ids)], expected_prompt_ids)
    assert torch.equal(prepared.batch["ref_input_ids"][0, -2:], torch.tensor([21, 22]))
    assert "ref_attention_mask" in prepared.batch
    assert "ref_position_ids" in prepared.batch
