import inspect

import numpy as np
import torch

from verl import DataProto
from verl.trainer.config import AlgoConfig, TaleBudgetConfig
from verl.trainer.ppo import ray_trainer, tale_budget
from verl.trainer.ppo.tale_budget import (
    build_concise_teacher_messages,
    build_tale_budget_estimation_prompt,
    build_tale_budget_teacher_messages,
    compute_rollout_length_tale_budget,
    normalize_tale_budget,
    parse_tale_budget,
    summarize_tale_budget_metrics,
)
from verl.utils.config import omega_conf_to_dataclass


class FakeBudgetTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def __init__(self, decoded_outputs):
        self.decoded_outputs = decoded_outputs
        self.seen_messages = []

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kwargs):
        self.seen_messages.append(messages)
        assert add_generation_prompt is True
        assert tokenize is False
        assert kwargs == {"enable_thinking": False}
        return messages[0]["content"] + "\n<GEN>"

    def __call__(self, texts, return_tensors="pt", add_special_tokens=False, padding=True):
        assert return_tensors == "pt"
        assert add_special_tokens is False
        assert padding is True
        if isinstance(texts, str):
            texts = [texts]
        encoded = [[(ord(char) % 89) + 1 for char in text] for text in texts]
        max_len = max(len(row) for row in encoded)
        input_ids = []
        attention_mask = []
        for row in encoded:
            pad_len = max_len - len(row)
            input_ids.append([self.pad_token_id] * pad_len + row)
            attention_mask.append([0] * pad_len + [1] * len(row))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def batch_decode(self, responses, skip_special_tokens=True):
        assert skip_special_tokens is True
        assert responses.shape[0] == len(self.decoded_outputs)
        return self.decoded_outputs


class FakeBudgetRolloutWorker:
    def __init__(self):
        self.calls = []

    def generate_sequences(self, prompts):
        self.calls.append(prompts)
        return DataProto.from_dict(
            tensors={"responses": torch.tensor([[11, 12, 0], [21, 0, 0]], dtype=torch.long)}
        )


class RightPaddingBudgetTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kwargs):
        assert add_generation_prompt is True
        assert tokenize is False
        return messages[0]["content"]

    def __call__(self, texts, return_tensors="pt", add_special_tokens=False, padding=False):
        assert return_tensors == "pt"
        assert add_special_tokens is False
        if isinstance(texts, str):
            encoded = [[(ord(char) % 89) + 1 for char in texts]]
        else:
            encoded = [[(ord(char) % 89) + 1 for char in text] for text in texts]
            if padding:
                max_len = max(len(row) for row in encoded)
                encoded = [row + [self.pad_token_id] * (max_len - len(row)) for row in encoded]

        max_len = max(len(row) for row in encoded)
        input_ids = []
        attention_mask = []
        for row in encoded:
            pad_len = max_len - len(row)
            input_ids.append(row + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(row) + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def _object_array(values):
    output = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        output[index] = value
    return output


def test_build_online_tale_budget_generation_batch_removes_right_padding_before_vllm_prompt_ids():
    from verl.trainer.ppo.ray_trainer import _build_online_tale_budget_generation_batch

    tokenizer = RightPaddingBudgetTokenizer()
    batch = _build_online_tale_budget_generation_batch(
        questions=["short?", "long? " + "x " * 100],
        tokenizer=tokenizer,
        max_prompt_length=1024,
        truncation="error",
        estimation_max_tokens=64,
        temperature=0.1,
        top_p=0.9,
        apply_chat_template_kwargs={"enable_thinking": False},
    )

    for row in batch.batch["input_ids"]:
        first_non_pad = int((row != tokenizer.pad_token_id).nonzero().flatten()[0])
        token_ids_sent_to_vllm = row[first_non_pad:].tolist()
        assert tokenizer.pad_token_id not in token_ids_sent_to_vllm


def test_apply_online_tale_budget_prompts_generates_teacher_prompt_and_metrics():
    from verl.trainer.ppo.ray_trainer import _apply_online_tale_budget_prompts

    raw_prompts = _object_array(
        [
            [{"role": "user", "content": "First problem?\nPlease reason step by step, and put your final answer within \\boxed{}."}],
            [{"role": "user", "content": "Second problem?"}],
        ]
    )
    batch = DataProto.from_dict(
        tensors={"input_ids": torch.ones((2, 3), dtype=torch.long)},
        non_tensors={"raw_prompt": raw_prompts},
    )
    tokenizer = FakeBudgetTokenizer(["Budget: [[256]]", "Budget: nope"])
    rollout_worker = FakeBudgetRolloutWorker()
    config = TaleBudgetConfig(enabled=True)

    metrics = _apply_online_tale_budget_prompts(
        batch=batch,
        actor_rollout_wg=rollout_worker,
        tokenizer=tokenizer,
        tale_budget_config=config,
        max_prompt_length=512,
        truncation="error",
        apply_chat_template_kwargs={"enable_thinking": False},
    )

    assert len(rollout_worker.calls) == 1
    generation_batch = rollout_worker.calls[0]
    assert generation_batch.meta_info["do_sample"] is True
    assert generation_batch.meta_info["response_length"] == config.estimation_max_tokens
    assert generation_batch.meta_info["temperature"] == config.temperature
    assert generation_batch.meta_info["top_p"] == config.top_p
    assert generation_batch.meta_info["generation_kwargs"] == {
        "max_tokens": config.estimation_max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
    }
    assert "Estimate how many tokens" in tokenizer.seen_messages[0][0]["content"]
    assert "N must be an integer between 128 and 8192" in tokenizer.seen_messages[0][0]["content"]
    assert "2048" not in tokenizer.seen_messages[0][0]["content"]
    assert "Problem:\nFirst problem?" in tokenizer.seen_messages[0][0]["content"]
    assert "Please reason step by step" not in tokenizer.seen_messages[0][0]["content"]

    assert batch.non_tensor_batch["teacher_prompt"].tolist() == [
        [
            {
                "role": "user",
                "content": (
                    "First problem?\n"
                    "Let's think step by step and use less than 256 tokens. "
                    "Put your final answer within \\boxed{}."
                ),
            }
        ],
        [
            {
                "role": "user",
                "content": (
                    "Second problem?\n"
                    "Let's think step by step and use less than 2048 tokens. "
                    "Put your final answer within \\boxed{}."
                ),
            }
        ],
    ]
    assert batch.non_tensor_batch["tale_budget"].tolist() == [256, 2048]
    assert batch.non_tensor_batch["tale_budget_raw"].tolist() == [256, None]
    assert batch.non_tensor_batch["tale_budget_estimate_text"].tolist() == ["Budget: [[256]]", "Budget: nope"]
    assert metrics == {
        "tale_budget/mean": (256 + 2048) / 2,
        "tale_budget/min": 256.0,
        "tale_budget/max": 2048.0,
        "tale_budget/parse_fail_ratio": 0.5,
    }


def test_apply_online_tale_budget_prompts_can_use_rollout_length_without_budget_generation():
    from verl.trainer.ppo.ray_trainer import _apply_online_tale_budget_prompts

    raw_prompts = _object_array(
        [
            [{"role": "user", "content": "First problem?\nPlease reason step by step, and put your final answer within \\boxed{}."}],
            [{"role": "user", "content": "Second problem?"}],
        ]
    )
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )
    batch = DataProto.from_dict(
        tensors={
            "input_ids": torch.ones((2, 8), dtype=torch.long),
            "response_mask": response_mask,
        },
        non_tensors={"raw_prompt": raw_prompts},
    )
    tokenizer = FakeBudgetTokenizer([])
    rollout_worker = FakeBudgetRolloutWorker()
    config = TaleBudgetConfig(
        enabled=True,
        source="rollout_length",
        min_budget=1,
        max_budget=8192,
        round_to=1,
        rollout_length_alpha=0.8,
        esr_beta=1.0,
    )

    metrics = _apply_online_tale_budget_prompts(
        batch=batch,
        actor_rollout_wg=rollout_worker,
        tokenizer=tokenizer,
        tale_budget_config=config,
        max_prompt_length=512,
        truncation="error",
        apply_chat_template_kwargs={"enable_thinking": False},
    )

    assert rollout_worker.calls == []
    assert tokenizer.seen_messages == []
    assert batch.non_tensor_batch["teacher_prompt"].tolist() == [
        [
            {
                "role": "user",
                "content": (
                    "First problem?\n"
                    "Let's think step by step and use less than 4 tokens. "
                    "Put your final answer within \\boxed{}."
                ),
            }
        ],
        [
            {
                "role": "user",
                "content": (
                    "Second problem?\n"
                    "Let's think step by step and use less than 5 tokens. "
                    "Put your final answer within \\boxed{}."
                ),
            }
        ],
    ]
    assert batch.non_tensor_batch["tale_budget"].tolist() == [4, 5]
    assert batch.non_tensor_batch["tale_budget_raw"].tolist() == [4, 5]
    assert batch.non_tensor_batch["tale_budget_estimate_text"].tolist() == [
        "rollout_length=5,alpha=0.8",
        "rollout_length=6,alpha=0.8",
    ]
    assert batch.batch["tale_budget_esr_loss_mask"].tolist() == [
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 0],
    ]
    assert metrics["tale_budget/mean"] == 4.5
    assert metrics["tale_budget/source_rollout_length"] == 1.0
    assert metrics["tale_budget/use_budget_teacher_prompt"] == 1.0
    assert metrics["tale_budget/esr_tokens_mean"] == 4.5
    assert abs(metrics["tale_budget/esr_supervised_fraction_mean"] - ((4 / 5 + 5 / 6) / 2)) < 1e-6


def test_apply_online_tale_budget_prompts_can_keep_normal_teacher_prompt_with_esr_mask():
    from verl.trainer.ppo.ray_trainer import _apply_online_tale_budget_prompts

    raw_prompts = _object_array(
        [
            [{"role": "user", "content": "First problem?\nPlease reason step by step, and put your final answer within \\boxed{}."}],
            [{"role": "user", "content": "Second problem?"}],
        ]
    )
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )
    batch = DataProto.from_dict(
        tensors={
            "input_ids": torch.ones((2, 8), dtype=torch.long),
            "response_mask": response_mask,
        },
        non_tensors={"raw_prompt": raw_prompts},
    )
    tokenizer = FakeBudgetTokenizer([])
    rollout_worker = FakeBudgetRolloutWorker()
    config = TaleBudgetConfig(
        enabled=True,
        source="rollout_length",
        min_budget=1,
        max_budget=8192,
        round_to=1,
        rollout_length_alpha=1.0,
        esr_beta=0.2,
        use_budget_teacher_prompt=False,
    )

    metrics = _apply_online_tale_budget_prompts(
        batch=batch,
        actor_rollout_wg=rollout_worker,
        tokenizer=tokenizer,
        tale_budget_config=config,
        max_prompt_length=512,
        truncation="error",
        apply_chat_template_kwargs={"enable_thinking": False},
    )

    assert rollout_worker.calls == []
    assert tokenizer.seen_messages == []
    assert batch.non_tensor_batch["teacher_prompt"].tolist() == raw_prompts.tolist()
    assert batch.non_tensor_batch["tale_budget"].tolist() == [5, 6]
    assert batch.batch["tale_budget_esr_loss_mask"].tolist() == [
        [1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
    ]
    assert metrics["tale_budget/source_rollout_length"] == 1.0
    assert metrics["tale_budget/use_budget_teacher_prompt"] == 0.0
    assert metrics["tale_budget/esr_tokens_mean"] == 1.0


def test_apply_online_tale_budget_prompts_can_use_concise_teacher_prompt_with_esr_mask():
    from verl.trainer.ppo.ray_trainer import _apply_online_tale_budget_prompts

    raw_prompts = _object_array(
        [
            [{"role": "user", "content": "First problem?\nPlease reason step by step, and put your final answer within \\boxed{}."}],
            [{"role": "user", "content": "Second problem?"}],
        ]
    )
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )
    batch = DataProto.from_dict(
        tensors={
            "input_ids": torch.ones((2, 8), dtype=torch.long),
            "response_mask": response_mask,
        },
        non_tensors={"raw_prompt": raw_prompts},
    )
    tokenizer = FakeBudgetTokenizer([])
    rollout_worker = FakeBudgetRolloutWorker()
    config = TaleBudgetConfig(
        enabled=True,
        source="rollout_length",
        min_budget=1,
        max_budget=8192,
        round_to=1,
        rollout_length_alpha=1.0,
        esr_beta=0.2,
        teacher_prompt_style="concise",
    )

    metrics = _apply_online_tale_budget_prompts(
        batch=batch,
        actor_rollout_wg=rollout_worker,
        tokenizer=tokenizer,
        tale_budget_config=config,
        max_prompt_length=512,
        truncation="error",
        apply_chat_template_kwargs={"enable_thinking": False},
    )

    assert rollout_worker.calls == []
    assert tokenizer.seen_messages == []
    assert batch.non_tensor_batch["teacher_prompt"].tolist() == [
        [
            {
                "role": "user",
                "content": (
                    "First problem?\n"
                    "Solve concisely. Avoid unnecessary explanation. "
                    "Put your final answer within \\boxed{}."
                ),
            }
        ],
        [
            {
                "role": "user",
                "content": (
                    "Second problem?\n"
                    "Solve concisely. Avoid unnecessary explanation. "
                    "Put your final answer within \\boxed{}."
                ),
            }
        ],
    ]
    assert batch.non_tensor_batch["tale_budget"].tolist() == [5, 6]
    assert batch.batch["tale_budget_esr_loss_mask"].tolist() == [
        [1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
    ]
    assert metrics["tale_budget/source_rollout_length"] == 1.0
    assert metrics["tale_budget/use_budget_teacher_prompt"] == 1.0
    assert metrics["tale_budget/esr_tokens_mean"] == 1.0



def test_apply_online_tale_budget_prompts_uses_per_sample_difficulty_routing():
    from verl.trainer.ppo.ray_trainer import _apply_online_tale_budget_prompts

    raw_prompts = _object_array(
        [
            [{"role": "user", "content": "Easy problem?\nPlease reason step by step, and put your final answer within \\boxed{}."}],
            [{"role": "user", "content": "Hard problem?"}],
        ]
    )
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )
    batch = DataProto.from_dict(
        tensors={
            "input_ids": torch.ones((2, 12), dtype=torch.long),
            "response_mask": response_mask,
            "difficulty_aware_esr_beta": torch.tensor([0.1, 0.2], dtype=torch.float32),
        },
        non_tensors={
            "raw_prompt": raw_prompts,
            "difficulty_aware_prompt_style": _object_array(["concise", "budget"]),
        },
    )
    tokenizer = FakeBudgetTokenizer([])
    rollout_worker = FakeBudgetRolloutWorker()
    config = TaleBudgetConfig(
        enabled=True,
        source="rollout_length",
        min_budget=1,
        max_budget=8192,
        round_to=1,
        rollout_length_alpha=1.0,
        esr_beta=0.2,
        teacher_prompt_style="budget",
    )

    metrics = _apply_online_tale_budget_prompts(
        batch=batch,
        actor_rollout_wg=rollout_worker,
        tokenizer=tokenizer,
        tale_budget_config=config,
        max_prompt_length=512,
        truncation="error",
        apply_chat_template_kwargs={"enable_thinking": False},
    )

    prompts = batch.non_tensor_batch["teacher_prompt"].tolist()
    assert "Solve concisely" in prompts[0][0]["content"]
    assert "use less than 10 tokens" in prompts[1][0]["content"]
    assert batch.batch["tale_budget_response_lengths"].tolist() == [10, 10]
    assert batch.batch["tale_budget_esr_loss_mask"].sum(dim=-1).tolist() == [1, 2]
    assert metrics["tale_budget/esr_tokens_mean"] == 1.5
    assert abs(metrics["tale_budget/esr_supervised_fraction_mean"] - 0.15) < 1e-6


def test_truncate_to_tale_budget_esr_physically_keeps_only_prefix_needed_for_training():
    responses = torch.tensor(
        [
            [10, 11, 12, 13, 14, 0],
            [20, 21, 22, 23, 24, 25],
        ],
        dtype=torch.long,
    )
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )
    esr_loss_mask = torch.tensor(
        [
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
        ],
        dtype=torch.long,
    )
    # Three prompt tokens followed by six response tokens.
    input_ids = torch.tensor(
        [
            [1, 2, 3, 10, 11, 12, 13, 14, 0],
            [4, 5, 6, 20, 21, 22, 23, 24, 25],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.cat([torch.ones((2, 3), dtype=torch.long), response_mask], dim=1)
    position_ids = torch.arange(input_ids.shape[1]).repeat(2, 1)
    old_log_probs = torch.arange(12, dtype=torch.float32).view(2, 6)

    batch = DataProto.from_dict(
        tensors={
            "responses": responses,
            "response_mask": response_mask,
            "tale_budget_esr_loss_mask": esr_loss_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "old_log_probs": old_log_probs,
        },
        meta_info={"global_token_num": [8, 9]},
    )

    truncated, metrics = tale_budget.truncate_to_tale_budget_esr(batch)

    assert batch.batch["responses"].shape[-1] == 6
    assert truncated.batch["responses"].tolist() == [[10, 11, 12, 13], [20, 21, 22, 23]]
    assert truncated.batch["response_mask"].tolist() == [[1, 1, 0, 0], [1, 1, 1, 1]]
    assert truncated.batch["tale_budget_esr_loss_mask"].tolist() == [[1, 1, 0, 0], [1, 1, 1, 1]]
    assert truncated.batch["old_log_probs"].tolist() == [[0.0, 1.0, 2.0, 3.0], [6.0, 7.0, 8.0, 9.0]]
    assert truncated.batch["input_ids"].tolist() == [[1, 2, 3, 10, 11, 12, 13], [4, 5, 6, 20, 21, 22, 23]]
    assert truncated.batch["attention_mask"].tolist() == [[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1]]
    assert truncated.batch["position_ids"].shape[-1] == 7
    assert truncated.meta_info["global_token_num"] == [5, 7]
    assert metrics == {
        "tale_budget/truncated_response_length": 4.0,
        "tale_budget/truncated_token_fraction": 6 / 11,
    }


def test_drop_ref_retokenization_tensors_removes_large_temporary_inputs_before_actor_update():
    batch = DataProto.from_dict(
        tensors={
            "responses": torch.ones((2, 3), dtype=torch.long),
            "ref_input_ids": torch.ones((2, 5), dtype=torch.long),
            "ref_attention_mask": torch.ones((2, 5), dtype=torch.long),
            "ref_position_ids": torch.ones((2, 5), dtype=torch.long),
        }
    )

    tale_budget.drop_ref_retokenization_tensors(batch)

    assert "responses" in batch.batch.keys()
    assert "ref_input_ids" not in batch.batch.keys()
    assert "ref_attention_mask" not in batch.batch.keys()
    assert "ref_position_ids" not in batch.batch.keys()


def test_compute_rollout_length_tale_budget_has_no_esr_max_cap():
    response_mask = torch.ones((2, 20000), dtype=torch.long)

    result = compute_rollout_length_tale_budget(
        response_mask=response_mask,
        alpha=0.8,
        beta=1.0,
        min_budget=1,
        round_to=1,
        max_budget=None,
    )

    assert result.budgets.tolist() == [16000, 16000]
    assert result.esr_tokens.tolist() == [16000, 16000]
    assert int(result.esr_loss_mask.sum(dim=-1)[0].item()) == 16000


def test_truncate_response_tensor_to_batch_preserves_sequence_reward_on_last_supervised_token():
    batch = DataProto.from_dict(
        tensors={
            "responses": torch.ones((2, 4), dtype=torch.long),
            "response_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.long),
        }
    )
    full_reward = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
        ]
    )

    truncated_reward = ray_trainer._truncate_response_tensor_to_batch(full_reward, batch)

    assert truncated_reward.tolist() == [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0]]
    assert truncated_reward.sum(dim=-1).tolist() == full_reward.sum(dim=-1).tolist()


def test_rethinking_probe_aggregates_full_mask_before_tale_truncation():
    from verl.trainer.ppo.rethinking_opd_probe import aggregate_rethinking_opd_probe_metrics

    response_mask = torch.ones(1, 6)
    tensors = {
        "student_top_k_log_probs": torch.log(torch.full((1, 6, 2), 0.5)),
        "teacher_on_student_log_probs": torch.log(torch.full((1, 6, 2), 0.5)),
        "overlap_mask": torch.ones(1, 6, 2),
        "student_entropys": torch.ones(1, 6) * 0.1,
        "ref_entropys": torch.ones(1, 6) * 0.2,
    }

    rows = aggregate_rethinking_opd_probe_metrics(tensors, response_mask, top_k=2, chunk_size=4)

    global_row = next(row for row in rows if row["chunk_start"] == -1)
    tail_row = next(row for row in rows if row["chunk_start"] == 4)
    assert global_row["valid_token_count"] == 6
    assert tail_row["valid_token_count"] == 2



def test_ray_trainer_applies_tale_budget_before_ref_retokenization():
    fit_source = inspect.getsource(ray_trainer.RayPPOTrainer.fit)

    assert "_apply_online_tale_budget_prompts(" in fit_source
    assert fit_source.index("_apply_online_tale_budget_prompts(") < fit_source.index("prepare_ref_model_inputs")


def test_ray_trainer_probe_logs_before_tale_truncation_and_drops_ref_inputs_before_actor_update():
    fit_source = inspect.getsource(ray_trainer.RayPPOTrainer.fit)

    assert "_log_rethinking_opd_probe_metrics(" in fit_source
    assert "truncate_to_tale_budget_esr(" in fit_source
    assert fit_source.index("_log_rethinking_opd_probe_metrics(") < fit_source.index("truncate_to_tale_budget_esr(")
    assert "drop_ref_retokenization_tensors(" in fit_source
    assert fit_source.index("drop_ref_retokenization_tensors(") < fit_source.index("update_actor(batch)")


def test_tale_budget_config_defaults_match_online_pivot():
    config = TaleBudgetConfig()

    assert config.enabled is False
    assert config.source == "llm_estimate"
    assert config.min_budget == 128
    assert config.max_budget == 8192
    assert config.round_to == 64
    assert config.fallback_budget == 2048
    assert config.estimation_max_tokens == 64
    assert config.temperature == 0.0
    assert config.top_p == 1.0
    assert config.teacher_prompt_key == "teacher_prompt"
    assert config.rollout_length_alpha == 0.8
    assert config.esr_beta == 1.0
    assert config.rollout_length_max_budget is None
    assert config.truncate_to_esr is False
    assert config.use_budget_teacher_prompt is True
    assert config.teacher_prompt_style == "auto"


def test_tale_budget_config_instantiates_from_algo_target():
    config = omega_conf_to_dataclass(
        {
            "_target_": "verl.trainer.config.AlgoConfig",
            "tale_budget": {
                "_target_": "verl.trainer.config.TaleBudgetConfig",
                "enabled": True,
                "source": "rollout_length",
                "min_budget": 64,
                "max_budget": 4096,
                "round_to": 32,
                "fallback_budget": 1024,
                "estimation_max_tokens": 48,
                "temperature": 0.0,
                "top_p": 1.0,
                "teacher_prompt_key": "online_teacher_prompt",
                "rollout_length_alpha": 0.7,
                "esr_beta": 1.0,
                "rollout_length_max_budget": None,
                "truncate_to_esr": True,
                "use_budget_teacher_prompt": False,
                "teacher_prompt_style": "concise",
            },
        }
    )

    assert isinstance(config, AlgoConfig)
    assert isinstance(config.tale_budget, TaleBudgetConfig)
    assert config.tale_budget.enabled is True
    assert config.tale_budget.min_budget == 64
    assert config.tale_budget.max_budget == 4096
    assert config.tale_budget.source == "rollout_length"
    assert config.tale_budget.teacher_prompt_key == "online_teacher_prompt"
    assert config.tale_budget.rollout_length_alpha == 0.7
    assert config.tale_budget.esr_beta == 1.0
    assert config.tale_budget.rollout_length_max_budget is None
    assert config.tale_budget.truncate_to_esr is True
    assert config.tale_budget.use_budget_teacher_prompt is False
    assert config.tale_budget.teacher_prompt_style == "concise"


def test_build_tale_budget_estimation_prompt_matches_tale_format():
    prompt = build_tale_budget_estimation_prompt(
        "What is 2+2?\nPlease reason step by step, and put your final answer within \\boxed{}."
    )

    assert "Do NOT solve" in prompt
    assert "Budget: [[N]]" in prompt
    assert "N must be an integer between 128 and 8192" in prompt
    assert "2048" not in prompt
    assert "What is 2+2?" in prompt
    assert "Please reason step by step" not in prompt
    assert prompt.rstrip().endswith("What is 2+2?")


def test_parse_and_normalize_tale_budget():
    assert parse_tale_budget("Budget: [[256]]") == 256
    assert parse_tale_budget("Budget: 256") == 256
    assert parse_tale_budget("[[384]]") == 384
    assert parse_tale_budget("512") == 512
    assert parse_tale_budget("To solve this, use theorem 2 and get 5.") is None
    assert normalize_tale_budget(190, min_budget=128, max_budget=8192, round_to=64, fallback_budget=2048) == 192
    assert normalize_tale_budget(None, min_budget=128, max_budget=8192, round_to=64, fallback_budget=2048) == 2048


def test_build_tale_budget_teacher_messages_strips_old_instruction():
    messages = build_tale_budget_teacher_messages(
        "Find x if x+1=3.\nPlease reason step by step, and put your final answer within \\boxed{}.",
        256,
    )

    assert messages == [
        {
            "role": "user",
            "content": (
                "Find x if x+1=3.\n"
                "Let's think step by step and use less than 256 tokens. "
                "Put your final answer within \\boxed{}."
            ),
        }
    ]



def test_build_concise_teacher_messages_strips_old_instruction():
    messages = build_concise_teacher_messages(
        "Find x if x+1=3.\nPlease reason step by step, and put your final answer within \\boxed{}."
    )

    assert messages == [
        {
            "role": "user",
            "content": (
                "Find x if x+1=3.\n"
                "Solve concisely. Avoid unnecessary explanation. "
                "Put your final answer within \\boxed{}."
            ),
        }
    ]


def test_summarize_tale_budget_metrics_reports_parse_fail_ratio():
    metrics = summarize_tale_budget_metrics(raw_budgets=[128, None, 256], budgets=[128, 2048, 256])

    assert metrics["tale_budget/mean"] == (128 + 2048 + 256) / 3
    assert metrics["tale_budget/min"] == 128
    assert metrics["tale_budget/max"] == 2048
    assert metrics["tale_budget/parse_fail_ratio"] == 1 / 3
