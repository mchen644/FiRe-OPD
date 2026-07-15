# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.workers.config import RolloutConfig
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import (
    _expand_native_n_outputs,
    vLLMRollout,
)


@dataclass
class FakeCompletion:
    token_ids: list[int]
    logprobs: list[dict[int, object]]


@dataclass
class FakeRequestOutput:
    outputs: list[FakeCompletion]


class FakeEngine:
    def __init__(self, outputs: list[FakeRequestOutput]):
        self.outputs = outputs
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.outputs


def _request_outputs(native_n: int) -> list[FakeRequestOutput]:
    return [
        FakeRequestOutput(
            [FakeCompletion([10 + slot], []) for slot in range(native_n)]
        ),
        FakeRequestOutput(
            [FakeCompletion([20 + slot], []) for slot in range(native_n)]
        ),
    ]


def _prompts(*, capture_enabled: bool = True, stable_ids=("q0", "q1")) -> DataProto:
    raw_prompt_ids = np.empty(2, dtype=object)
    raw_prompt_ids[0] = [101, 102]
    raw_prompt_ids[1] = [201, 202]
    return DataProto(
        batch=TensorDict(
            {
                "input_ids": torch.tensor([[0, 101, 102], [0, 201, 202]]),
                "attention_mask": torch.tensor([[0, 1, 1], [0, 1, 1]]),
                "position_ids": torch.tensor([[0, 0, 1], [0, 0, 1]]),
            },
            batch_size=2,
        ),
        non_tensor_batch={
            "raw_prompt_ids": raw_prompt_ids,
            "opd_verify_stable_id": np.array(stable_ids, dtype=object),
            "opd_verify_split": np.array(["candidate", "held_out"], dtype=object),
            "opd_verify_manifest_index": np.array([0, 1], dtype=object),
        },
        meta_info={
            "eos_token_id": 99,
            "do_sample": True,
            "validate": False,
            "opd_proxy_verify_capture_enabled": capture_enabled,
        },
    )


def _bare_rollout(outputs: list[FakeRequestOutput]) -> vLLMRollout:
    rollout = object.__new__(vLLMRollout)
    rollout.config = RolloutConfig(
        name="vllm",
        tensor_model_parallel_size=1,
        prompt_length=3,
        response_length=4,
    )
    rollout.pad_token_id = 0
    rollout.sampling_params = SimpleNamespace(n=1)
    rollout.lora_kwargs = {}
    rollout.inference_engine = FakeEngine(outputs)
    rollout._opd_proxy_verify_capture_generated = False
    return rollout


def _generate_without_device_decorators(
    rollout: vLLMRollout, prompts: DataProto, **kwargs
) -> DataProto:
    generate = vLLMRollout.generate_sequences
    while hasattr(generate, "__wrapped__"):
        generate = generate.__wrapped__
    return generate(rollout, prompts, **kwargs)


def test_native_n_expands_prompt_major_and_assigns_slots():
    expanded = _expand_native_n_outputs(
        stable_ids=np.array(["q0", "q1"], dtype=object),
        request_outputs=[
            FakeRequestOutput([FakeCompletion([10], []), FakeCompletion([11], [])]),
            FakeRequestOutput([FakeCompletion([20], []), FakeCompletion([21], [])]),
        ],
        native_n=2,
    )
    assert expanded.stable_ids.tolist() == ["q0", "q0", "q1", "q1"]
    assert expanded.rollout_slots.tolist() == [0, 1, 0, 1]
    assert expanded.token_ids == [[10], [11], [20], [21]]


def test_native_n_calls_one_generation_and_expands_every_prompt_field():
    rollout = _bare_rollout(_request_outputs(4))
    prompts = _prompts()

    result = _generate_without_device_decorators(
        rollout, prompts, opd_proxy_verify_native_n=4
    )

    assert len(rollout.inference_engine.calls) == 1
    call = rollout.inference_engine.calls[0]
    assert len(call["prompts"]) == 2
    assert call["sampling_params"].n == 4
    assert rollout.sampling_params.n == 1
    assert result.non_tensor_batch["opd_verify_stable_id"].tolist() == [
        "q0",
        "q0",
        "q0",
        "q0",
        "q1",
        "q1",
        "q1",
        "q1",
    ]
    assert result.batch["opd_proxy_verify_rollout_slot"].tolist() == [
        0,
        1,
        2,
        3,
        0,
        1,
        2,
        3,
    ]
    assert result.batch["responses"][:, 0].tolist() == [
        10,
        11,
        12,
        13,
        20,
        21,
        22,
        23,
    ]
    assert result.batch["prompts"].tolist() == [
        [0, 101, 102],
        [0, 101, 102],
        [0, 101, 102],
        [0, 101, 102],
        [0, 201, 202],
        [0, 201, 202],
        [0, 201, 202],
        [0, 201, 202],
    ]
    # Native capture must not consume the caller's prompt metadata.
    assert "raw_prompt_ids" in prompts.non_tensor_batch

    with pytest.raises(RuntimeError, match="second native capture generation"):
        _generate_without_device_decorators(
            rollout, prompts, opd_proxy_verify_native_n=4
        )
    assert len(rollout.inference_engine.calls) == 1


def test_pilot_native_n1_uses_one_generation_call_and_assigns_slot_zero():
    rollout = _bare_rollout(_request_outputs(1))
    prompts = _prompts()
    prompts.meta_info["opd_proxy_verify_stage"] = "efficacy_pilot"

    result = _generate_without_device_decorators(
        rollout, prompts, opd_proxy_verify_native_n=1
    )

    assert len(rollout.inference_engine.calls) == 1
    assert rollout.inference_engine.calls[0]["sampling_params"].n == 1
    assert result.non_tensor_batch["opd_verify_stable_id"].tolist() == ["q0", "q1"]
    assert result.batch["opd_proxy_verify_rollout_slot"].tolist() == [0, 0]
    assert result.batch["responses"][:, 0].tolist() == [10, 20]


def test_native_n_requires_capture_mode_stage_rollouts_and_unique_prompt_ids():
    rollout = _bare_rollout(_request_outputs(4))
    with pytest.raises(ValueError, match="capture enabled"):
        _generate_without_device_decorators(
            rollout,
            _prompts(capture_enabled=False),
            opd_proxy_verify_native_n=4,
        )
    with pytest.raises(ValueError, match="numbered stage.*exactly 4"):
        _generate_without_device_decorators(
            rollout, _prompts(), opd_proxy_verify_native_n=2
        )
    pilot_prompts = _prompts()
    pilot_prompts.meta_info["opd_proxy_verify_stage"] = "efficacy_pilot"
    with pytest.raises(ValueError, match="efficacy_pilot.*exactly 1"):
        _generate_without_device_decorators(
            rollout, pilot_prompts, opd_proxy_verify_native_n=4
        )
    with pytest.raises(ValueError, match="unique.*stable"):
        _generate_without_device_decorators(
            rollout,
            _prompts(stable_ids=("q0", "q0")),
            opd_proxy_verify_native_n=4,
        )
    assert rollout.inference_engine.calls == []


def test_default_n1_path_remains_single_output_without_capture_slot():
    rollout = _bare_rollout(_request_outputs(1))
    result = _generate_without_device_decorators(
        rollout, _prompts(capture_enabled=False)
    )

    assert len(rollout.inference_engine.calls) == 1
    assert rollout.inference_engine.calls[0]["sampling_params"].n == 1
    assert result.batch.batch_size == torch.Size([2])
    assert result.batch["responses"][:, 0].tolist() == [10, 20]
    assert "opd_proxy_verify_rollout_slot" not in result.batch
