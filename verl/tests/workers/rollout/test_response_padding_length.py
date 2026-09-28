from verl.workers.rollout.base import resolve_response_padding_length


def test_resolve_response_padding_length_prefers_generation_max_tokens():
    assert resolve_response_padding_length(
        config_response_length=16384,
        prompts_meta_info={"response_length": 64},
        generation_kwargs={"max_tokens": 64, "temperature": 0.1},
    ) == 64


def test_resolve_response_padding_length_falls_back_to_prompt_meta_then_config():
    assert resolve_response_padding_length(
        config_response_length=16384,
        prompts_meta_info={"response_length": 128},
        generation_kwargs={},
    ) == 128
    assert resolve_response_padding_length(
        config_response_length=16384,
        prompts_meta_info={},
        generation_kwargs={},
    ) == 16384
