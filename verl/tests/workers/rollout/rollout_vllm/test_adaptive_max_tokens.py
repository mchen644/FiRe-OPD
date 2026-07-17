import inspect

import pytest
import torch
from vllm import SamplingParams

from verl.workers.rollout.vllm_rollout import vllm_rollout_spmd
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import (
    _build_per_request_sampling_params,
    _build_response_attention_mask,
)


def test_build_per_request_sampling_params_uses_exact_caps_without_mutating_base() -> None:
    base = SamplingParams(max_tokens=128, temperature=1.0, top_p=1.0, logprobs=0)
    rows = _build_per_request_sampling_params(
        base,
        max_tokens_by_row=[10, 31, 64],
        expected_rows=3,
        max_response_length=128,
        disable_rollout_log_probs=True,
    )

    assert [row.max_tokens for row in rows] == [10, 31, 64]
    assert all(row.logprobs is None for row in rows)
    assert len({id(row) for row in rows}) == 3
    assert base.max_tokens == 128
    assert base.logprobs == 0


def test_build_per_request_sampling_params_can_keep_rollout_log_probs() -> None:
    base = SamplingParams(max_tokens=32, logprobs=0)
    rows = _build_per_request_sampling_params(
        base,
        max_tokens_by_row=[3, 5],
        expected_rows=2,
        max_response_length=32,
        disable_rollout_log_probs=False,
    )
    assert [row.logprobs for row in rows] == [0, 0]


@pytest.mark.parametrize(
    ("caps", "expected_rows", "max_response_length", "message"),
    [
        ([1], 2, 128, "row count"),
        ([True, 2], 2, 128, "positive integers"),
        ([0, 2], 2, 128, "positive integers"),
        ([-1, 2], 2, 128, "positive integers"),
        ([129, 2], 2, 128, "configured response length"),
    ],
)
def test_build_per_request_sampling_params_rejects_invalid_caps(
    caps: list[int], expected_rows: int, max_response_length: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _build_per_request_sampling_params(
            SamplingParams(max_tokens=128, logprobs=0),
            max_tokens_by_row=caps,
            expected_rows=expected_rows,
            max_response_length=max_response_length,
            disable_rollout_log_probs=True,
        )


def test_response_attention_mask_excludes_non_eos_padding_after_shorter_per_request_caps() -> None:
    pad_token_id = 151643
    eos_token_id = 151645
    responses = torch.tensor(
        [
            [10, pad_token_id, pad_token_id, pad_token_id],
            [20, 21, 22, 23],
            [30, eos_token_id, pad_token_id, pad_token_id],
        ]
    )

    mask = _build_response_attention_mask(
        responses,
        eos_token_id=eos_token_id,
        generated_lengths=[1, 4, 2],
        dtype=torch.long,
    )

    assert mask.tolist() == [
        [1, 0, 0, 0],
        [1, 1, 1, 1],
        [1, 1, 0, 0],
    ]
    assert mask.sum(dim=-1).tolist() == [1, 4, 2]


def test_vllm_generate_path_consumes_adaptive_kwargs_before_sampling_update() -> None:
    source = inspect.getsource(vllm_rollout_spmd)
    cap_pop = 'kwargs.pop("max_tokens_by_row", None)'
    disable_pop = 'kwargs.pop("disable_rollout_log_probs", False)'
    assert cap_pop in source
    assert disable_pop in source
    assert source.index(cap_pop) < source.index("with self.update_sampling_params")
    assert source.index(disable_pop) < source.index("with self.update_sampling_params")
    assert "generation_sampling_params = _build_per_request_sampling_params(" in source
    assert "output_response_length = max(max_tokens_by_row)" in source
    assert "collect_rollout_log_probs" in source
