import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.ref_input_utils import prepare_ref_model_inputs


class _FakeRefTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.seen_messages = []

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kwargs):
        self.seen_messages.append(messages)
        assert add_generation_prompt is True
        assert tokenize is False
        return str(messages[0]["content"])

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        assert return_tensors == "pt"
        assert add_special_tokens is False
        if text == "teacher prompt":
            return {"input_ids": torch.tensor([[11, 12]], dtype=torch.long)}
        if text == "student prompt":
            return {"input_ids": torch.tensor([[21, 22]], dtype=torch.long)}
        raise AssertionError(f"unexpected prompt text: {text!r}")


def test_prepare_ref_model_inputs_uses_requested_raw_prompt_key():
    tokenizer = _FakeRefTokenizer()
    batch = DataProto.from_dict(
        tensors={
            "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
            "position_ids": torch.arange(4, dtype=torch.long).unsqueeze(0),
            "responses": torch.tensor([[101, 102]], dtype=torch.long),
            "response_mask": torch.ones((1, 2), dtype=torch.long),
        },
        non_tensors={
            "raw_prompt": np.array([[{"role": "user", "content": "student prompt"}]], dtype=object),
            "teacher_prompt": np.array([[{"role": "user", "content": "teacher prompt"}]], dtype=object),
        },
    )

    prepared = prepare_ref_model_inputs(
        batch=batch,
        ref_tokenizer=tokenizer,
        apply_chat_template_kwargs={},
        raw_prompt_key="teacher_prompt",
    )

    assert tokenizer.seen_messages == [[{"role": "user", "content": "teacher prompt"}]]
    assert prepared.batch["ref_input_ids"].tolist() == [[11, 12, 101, 102]]
    assert prepared.batch["ref_attention_mask"].tolist() == [[1, 1, 1, 1]]
    assert prepared.batch["ref_position_ids"].tolist() == [[0, 1, 2, 3]]
