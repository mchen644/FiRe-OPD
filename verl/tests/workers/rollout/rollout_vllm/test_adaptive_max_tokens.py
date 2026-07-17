import inspect

import pytest
from vllm import SamplingParams

from verl.workers.rollout.vllm_rollout import vllm_rollout_spmd
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import (
    _build_per_request_sampling_params,
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
