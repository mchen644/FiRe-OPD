# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.candidate_selection import select_short_correct_candidates
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.tale_budget import (
    build_concise_teacher_messages,
    build_tale_budget_estimation_prompt,
    build_tale_budget_teacher_messages,
    compute_rollout_length_tale_budget,
    drop_ref_retokenization_tensors,
    normalize_tale_budget,
    parse_tale_budget,
    summarize_tale_budget_metrics,
    truncate_to_tale_budget_esr,
)
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.model import compute_position_id_with_mask
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean, postprocess_data
from verl.utils.tracking import ValidationGenerationsLogger


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def _messages_to_list(messages):
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    return list(messages)


def _object_array(values):
    output = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        output[index] = value
    return output


def _extract_single_user_question(messages) -> str:
    message_list = _messages_to_list(messages)
    if len(message_list) != 1:
        raise ValueError(f"Expected exactly one raw prompt message, got {len(message_list)}")
    message = dict(message_list[0])
    if message.get("role") != "user":
        raise ValueError(f"Expected raw prompt role 'user', got {message.get('role')!r}")
    return str(message.get("content", ""))


def _build_online_tale_budget_generation_batch(
    *,
    questions: list[str],
    tokenizer,
    max_prompt_length: int,
    truncation: str,
    estimation_max_tokens: int,
    temperature: float,
    top_p: float,
    apply_chat_template_kwargs: dict | None,
) -> DataProto:
    if apply_chat_template_kwargs is None:
        apply_chat_template_kwargs = {}

    budget_messages = [
        [{"role": "user", "content": build_tale_budget_estimation_prompt(question)}] for question in questions
    ]
    prompt_texts = [
        tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            **apply_chat_template_kwargs,
        )
        for messages in budget_messages
    ]
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_id_rows = []
    attention_mask_rows = []
    for prompt_text in prompt_texts:
        model_inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        input_ids, attention_mask = postprocess_data(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            max_length=max_prompt_length,
            pad_token_id=pad_token_id,
            left_pad=True,
            truncation=truncation,
        )
        input_id_rows.append(input_ids[0])
        attention_mask_rows.append(attention_mask[0])

    input_ids = torch.stack(input_id_rows, dim=0)
    attention_mask = torch.stack(attention_mask_rows, dim=0)
    position_ids = compute_position_id_with_mask(attention_mask)

    return DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        meta_info={
            "do_sample": True,
            "response_length": int(estimation_max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": pad_token_id,
            "generation_kwargs": {
                "max_tokens": int(estimation_max_tokens),
                "temperature": float(temperature),
                "top_p": float(top_p),
            },
        },
    )


def _resolve_tale_teacher_prompt_style(tale_budget_config) -> str:
    """Resolve legacy budget-prompt bool plus the newer prompt-style selector."""

    style = str(tale_budget_config.get("teacher_prompt_style", "auto"))
    if style == "auto":
        return "budget" if tale_budget_config.get("use_budget_teacher_prompt", True) else "normal"
    if style not in {"budget", "normal", "concise"}:
        raise ValueError(
            "algorithm.tale_budget.teacher_prompt_style must be one of "
            f"'auto', 'budget', 'normal', or 'concise'. Got {style!r}."
        )
    return style



def _apply_online_tale_budget_prompts(
    *,
    batch: DataProto,
    actor_rollout_wg,
    tokenizer,
    tale_budget_config,
    max_prompt_length: int,
    truncation: str,
    apply_chat_template_kwargs: dict | None = None,
) -> dict[str, float]:
    """Generate online TALE budgets and write budget-aware teacher prompts into batch."""

    if not tale_budget_config or not tale_budget_config.get("enabled", False):
        return {}
    if "raw_prompt" not in batch.non_tensor_batch:
        raise ValueError("algorithm.tale_budget.enabled=True requires data.return_raw_chat=True for raw_prompt.")

    questions = [_extract_single_user_question(messages) for messages in batch.non_tensor_batch["raw_prompt"]]
    source = tale_budget_config.get("source", "llm_estimate")
    teacher_prompt_style = _resolve_tale_teacher_prompt_style(tale_budget_config)

    if source == "rollout_length":
        if "response_mask" not in batch.batch.keys():
            raise ValueError("algorithm.tale_budget.source=rollout_length requires response_mask in batch.")
        max_budget = tale_budget_config.get("rollout_length_max_budget", None)
        esr_beta_value = tale_budget_config.get("esr_beta", 1.0)
        if "difficulty_aware_esr_beta" in batch.batch.keys():
            esr_beta_value = batch.batch["difficulty_aware_esr_beta"]
        result = compute_rollout_length_tale_budget(
            response_mask=batch.batch["response_mask"],
            alpha=float(tale_budget_config.get("rollout_length_alpha", 0.8)),
            beta=esr_beta_value,
            min_budget=int(tale_budget_config.min_budget),
            round_to=int(tale_budget_config.round_to),
            max_budget=None if max_budget is None else int(max_budget),
        )
        budgets = [int(value) for value in result.budgets.detach().cpu().tolist()]
        raw_budgets = budgets
        response_lengths = [int(value) for value in result.response_lengths.detach().cpu().tolist()]
        estimate_texts = [
            f"rollout_length={response_length},alpha={float(tale_budget_config.get('rollout_length_alpha', 0.8)):g}"
            for response_length in response_lengths
        ]
        batch.batch["tale_budget_esr_loss_mask"] = result.esr_loss_mask
        batch.batch["tale_budget_response_lengths"] = result.response_lengths.detach().to(dtype=torch.long)
        esr_tokens = result.esr_tokens.float()
        response_lengths_tensor = result.response_lengths.float().clamp(min=1.0)
        metrics = summarize_tale_budget_metrics(raw_budgets=raw_budgets, budgets=budgets)
        metrics.update(
            {
                "tale_budget/source_rollout_length": 1.0,
                "tale_budget/response_length_mean": response_lengths_tensor.mean().item(),
                "tale_budget/esr_tokens_mean": esr_tokens.mean().item(),
                "tale_budget/esr_supervised_fraction_mean": (esr_tokens / response_lengths_tensor).mean().item(),
                "tale_budget/use_budget_teacher_prompt": 0.0 if teacher_prompt_style == "normal" else 1.0,
            }
        )
    elif source == "llm_estimate":
        budget_generation_batch = _build_online_tale_budget_generation_batch(
            questions=questions,
            tokenizer=tokenizer,
            max_prompt_length=max_prompt_length,
            truncation=truncation,
            estimation_max_tokens=tale_budget_config.estimation_max_tokens,
            temperature=tale_budget_config.temperature,
            top_p=tale_budget_config.top_p,
            apply_chat_template_kwargs=apply_chat_template_kwargs,
        )
        budget_output = actor_rollout_wg.generate_sequences(budget_generation_batch)
        estimate_texts = tokenizer.batch_decode(budget_output.batch["responses"], skip_special_tokens=True)
        if len(estimate_texts) != len(questions):
            raise ValueError(f"Expected {len(questions)} TALE budget estimates, got {len(estimate_texts)}")

        raw_budgets = [parse_tale_budget(text) for text in estimate_texts]
        budgets = [
            normalize_tale_budget(
                raw_budget,
                min_budget=tale_budget_config.min_budget,
                max_budget=tale_budget_config.max_budget,
                round_to=tale_budget_config.round_to,
                fallback_budget=tale_budget_config.fallback_budget,
            )
            for raw_budget in raw_budgets
        ]
        metrics = summarize_tale_budget_metrics(raw_budgets=raw_budgets, budgets=budgets)
    else:
        raise ValueError(f"Unsupported algorithm.tale_budget.source={source!r}")

    per_sample_prompt_styles = batch.non_tensor_batch.get("difficulty_aware_prompt_style", None)
    if per_sample_prompt_styles is None:
        per_sample_prompt_styles = np.array([teacher_prompt_style] * len(questions), dtype=object)

    teacher_prompts = []
    for question, budget, row_style, raw_messages in zip(
        questions,
        budgets,
        per_sample_prompt_styles.tolist(),
        batch.non_tensor_batch["raw_prompt"].tolist(),
        strict=True,
    ):
        row_style = str(row_style)
        if row_style == "budget":
            teacher_prompts.append(build_tale_budget_teacher_messages(question=question, budget=budget))
        elif row_style == "normal":
            teacher_prompts.append(deepcopy(raw_messages))
        elif row_style == "concise":
            teacher_prompts.append(build_concise_teacher_messages(question=question))
        else:
            raise ValueError(f"Unsupported per-sample teacher prompt style: {row_style!r}")

    teacher_prompt_key = tale_budget_config.teacher_prompt_key
    batch.non_tensor_batch[teacher_prompt_key] = _object_array(teacher_prompts)
    batch.non_tensor_batch["tale_budget"] = np.array(budgets)
    batch.non_tensor_batch["tale_budget_raw"] = _object_array(raw_budgets)
    batch.non_tensor_batch["tale_budget_estimate_text"] = _object_array(estimate_texts)
    return metrics


def _difficulty_aware_opd_enabled(config) -> bool:
    difficulty_config = config.algorithm.get("difficulty_aware_opd", None)
    return bool(difficulty_config and difficulty_config.get("enabled", False))



def _apply_difficulty_aware_opd_routing(
    *,
    batch: DataProto,
    reward_tensor: torch.Tensor,
    difficulty_config,
    base_esr_beta: float,
    metrics: dict,
) -> None:
    from verl.trainer.ppo.difficulty_aware_opd import (
        compute_two_signal_difficulty_routing,
        summarize_difficulty_routing,
    )

    if "old_log_probs" not in batch.batch.keys():
        raise ValueError("difficulty-aware OPD routing requires old_log_probs")
    if "response_mask" not in batch.batch.keys():
        raise ValueError("difficulty-aware OPD routing requires response_mask")

    result = compute_two_signal_difficulty_routing(
        token_level_scores=reward_tensor,
        old_log_probs=batch.batch["old_log_probs"],
        response_mask=batch.batch["response_mask"],
        config=difficulty_config,
        base_esr_beta=base_esr_beta,
    )
    target_device = batch.batch["response_mask"].device
    batch.batch["difficulty_aware_correct"] = result.correct.to(device=target_device)
    batch.batch["difficulty_aware_confidence_rank"] = result.confidence_rank.to(device=target_device)
    batch.batch["difficulty_aware_easy"] = result.easy.to(device=target_device)
    batch.batch["difficulty_aware_hard"] = result.hard.to(device=target_device)
    batch.batch["difficulty_aware_esr_beta"] = result.esr_beta.to(device=target_device)
    if torch.any(result.entropy_weight > 0):
        batch.batch["difficulty_aware_entropy_weight"] = result.entropy_weight.to(device=target_device)
    batch.non_tensor_batch["difficulty_aware_prompt_style"] = result.prompt_styles

    original_lengths = batch.batch["response_mask"].to(dtype=torch.long).sum(dim=-1)
    metrics.update(summarize_difficulty_routing(result, original_response_lengths=original_lengths))



def _rethinking_probe_enabled(config) -> bool:
    probe_config = config.algorithm.get("rethinking_opd_probe", None)
    return bool(probe_config and probe_config.get("enabled", False))


def _old_log_prob_entropy_required(config) -> bool:
    """Whether old-log-prob recomputation must also materialize per-token entropy."""

    entropy_coeff = float(OmegaConf.select(config, "actor_rollout_ref.actor.entropy_coeff", default=0.0) or 0.0)
    if entropy_coeff != 0.0:
        return True

    entropy_aware_distill = bool(
        OmegaConf.select(config, "actor_rollout_ref.actor.policy_loss.entropy_aware_distill", default=False)
    )
    if entropy_aware_distill:
        return True

    difficulty_aware_enabled = bool(OmegaConf.select(config, "algorithm.difficulty_aware_opd.enabled", default=False))
    hard_entropy_coef = float(
        OmegaConf.select(config, "algorithm.difficulty_aware_opd.hard_entropy_coef", default=0.0) or 0.0
    )
    if difficulty_aware_enabled and hard_entropy_coef != 0.0:
        return True

    return _rethinking_probe_enabled(config)



def _set_rethinking_probe_meta(batch: DataProto, config) -> None:
    probe_config = config.algorithm.get("rethinking_opd_probe", None)
    top_k = int(probe_config.get("top_k", 16)) if probe_config else 0
    batch.meta_info["rethinking_opd_probe_top_k"] = (
        top_k if probe_config and probe_config.get("enabled", False) else 0
    )



def _log_rethinking_opd_probe_metrics(batch: DataProto, *, config, global_step: int, metrics: dict) -> None:
    probe_config = config.algorithm.get("rethinking_opd_probe", None)
    if not probe_config or not probe_config.get("enabled", False):
        return

    from verl.trainer.ppo.rethinking_opd_probe import (
        aggregate_rethinking_opd_probe_metrics,
        append_rethinking_opd_probe_csv,
        decorate_rethinking_probe_rows,
        has_rethinking_opd_probe_tensors,
    )

    log_prefix = str(probe_config.get("log_prefix", "rethinking_opd"))
    if not has_rethinking_opd_probe_tensors(batch.batch):
        if probe_config.get("include_scalar_logger", True):
            metrics[f"{log_prefix}/skipped_missing_tensors"] = 1.0
        return

    top_k = int(probe_config.get("top_k", 16))
    chunk_size = int(probe_config.get("chunk_size", 1024))
    response_mask = batch.batch["response_mask"]
    tensors = {
        key: batch.batch[key]
        for key in (
            "student_top_k_log_probs",
            "teacher_on_student_log_probs",
            "overlap_mask",
            "student_entropys",
            "ref_entropys",
        )
        if key in batch.batch
    }

    rows = aggregate_rethinking_opd_probe_metrics(tensors, response_mask, top_k=top_k, chunk_size=chunk_size)
    run_name = str(config.trainer.get("experiment_name", "unknown"))
    rows = decorate_rethinking_probe_rows(
        rows,
        run_name=run_name,
        step=int(global_step),
        top_k=top_k,
        response_mask=response_mask,
    )

    if probe_config.get("include_scalar_logger", True):
        for row in rows:
            suffix = "global" if row["chunk_start"] == -1 else f"chunk_{row['chunk_start']}_{row['chunk_end']}"
            for key in (
                "topk_overlap_ratio",
                "student_overlap_mass",
                "teacher_overlap_mass",
                "student_entropy",
                "teacher_entropy",
                "entropy_gap",
                "overlap_token_advantage",
                "valid_token_count",
                "valid_sequence_count",
            ):
                metrics[f"{log_prefix}/{key}_{suffix}"] = row[key]

    csv_path = probe_config.get("csv_path", None)
    if csv_path:
        append_rethinking_opd_probe_csv(csv_path, rows)



def _truncate_response_tensor_to_batch(response_tensor: torch.Tensor, batch: DataProto) -> torch.Tensor:
    """Align full-response outcome rewards to a hard-truncated ESR training batch.

    Rule-based math rewards are sequence-level scores stored on a response token.
    If that token is outside the ESR prefix, a plain slice would turn correct
    trajectories into zero-reward trajectories for logging/advantage plumbing.
    Preserve the sequence reward sum by moving it to the last supervised token.
    """

    response_length = batch.batch["responses"].shape[-1]
    if response_tensor.shape[-1] == response_length:
        return response_tensor
    if response_tensor.shape[-1] < response_length:
        raise ValueError(
            f"Cannot align response tensor length {response_tensor.shape[-1]} to batch response length {response_length}"
        )

    response_mask = batch.batch["response_mask"].to(device=response_tensor.device)
    sequence_reward = response_tensor.sum(dim=-1)
    truncated = torch.zeros(
        (*response_tensor.shape[:-1], response_length),
        dtype=response_tensor.dtype,
        device=response_tensor.device,
    )
    last_supervised_index = response_mask.to(dtype=torch.long).sum(dim=-1).clamp(min=1) - 1
    truncated.scatter_(dim=-1, index=last_supervised_index.unsqueeze(-1), src=sequence_reward.unsqueeze(-1))
    return truncated


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
        ref_tokenizer=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
            ref_tokenizer: Optional tokenizer for reference model. If provided and different from tokenizer,
                re-tokenization will be performed before computing ref log probs.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        # Store ref_tokenizer for re-tokenization when ref model uses different tokenizer
        self.ref_tokenizer = ref_tokenizer
        self.use_ref_retokenization = ref_tokenizer is not None
        self.ref_raw_prompt_key = config.data.get("ref_raw_prompt_key", "raw_prompt")

        if self.use_ref_retokenization:
            return_raw_chat = config.data.get("return_raw_chat", False)
            if self.ref_raw_prompt_key == "raw_prompt" and not return_raw_chat:
                raise ValueError(
                    "When using a different tokenizer for ref model (ref_tokenizer is provided) "
                    "you must set data.return_raw_chat=True in config to enable re-tokenization, "
                    "or set data.ref_raw_prompt_key to a parquet prompt column such as teacher_prompt."
                )

        # Multi-teacher: ref's base model is the code teacher
        self.ref_base_model_path = config.actor_rollout_ref.ref.get("model", None)
        if self.ref_base_model_path is not None:
            self.ref_base_model_path = self.ref_base_model_path.get("base_model_path", None)
        self.use_base_models = self.ref_base_model_path is not None

        if self.use_base_models:
            print(f"Multi-teacher mode: code teacher model = {self.ref_base_model_path}")

                
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = (
            config.actor_rollout_ref.model.get("lora_rank", 0) > 0
            or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        )

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role=str(Role.ActorRollout),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.ActorRollout)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(Role.ActorRollout)]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            # NOTE: while there is no checkpoint to load, we still need to offload the model and optimizer to CPU
            self.actor_rollout_wg.load_checkpoint(None)
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                self.actor_rollout_wg.load_checkpoint(None)
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        global_seqlen_lst = calculate_workload(global_seqlen_lst)
        world_size = self.actor_rollout_wg.world_size
        if keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(global_seqlen_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(world_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    global_seqlen_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=world_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(
                global_seqlen_lst, k_partitions=world_size, equal_size=True
            )
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        for idx, partition in enumerate(global_partition_lst):
            partition.sort(key=lambda x: (global_seqlen_lst[x], x))
            ordered_partition = partition[::2] + partition[1::2][::-1]
            global_partition_lst[idx] = ordered_partition
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                rm_scores = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(rm_scores)
                            reward_baseline_tensor, _ = compute_reward(batch, self.reward_fn)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    reward_tensor = None
                    reward_extra_infos_dict = {}
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_rollout_correction

                        apply_rollout_correction(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            _set_rethinking_probe_meta(batch, self.config)
                            batch.meta_info["calculate_entropy"] = _old_log_prob_entropy_required(self.config)
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            # Keep student entropys in batch if entropy-aware distillation is enabled
                            entropy_aware = getattr(
                                self.config.actor_rollout_ref.actor.policy_loss,
                                "entropy_aware_distill", False
                            )
                            if "entropys" in old_log_prob.batch.keys():
                                entropys = old_log_prob.batch["entropys"]
                                response_masks = batch.batch["response_mask"]
                                loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                                entropy_agg = agg_loss(
                                    loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                                )
                                old_log_prob_metrics = {
                                    "actor/entropy": entropy_agg.detach().item(),
                                    "actor/entropy_computed": 1.0,
                                }
                                metrics.update(old_log_prob_metrics)
                                if entropy_aware:
                                    # Rename to student_entropys and keep in batch
                                    old_log_prob.batch["student_entropys"] = old_log_prob.batch.pop("entropys")
                                else:
                                    old_log_prob.batch.pop("entropys")
                            else:
                                metrics["actor/entropy_computed"] = 0.0
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if _difficulty_aware_opd_enabled(self.config):
                        if self.config.reward_model.launch_reward_fn_async and reward_tensor is None:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        difficulty_config = self.config.algorithm.difficulty_aware_opd
                        tale_budget_config_for_da = self.config.algorithm.get("tale_budget", None)
                        configured_base_esr_beta = difficulty_config.get("base_esr_beta", None)
                        if configured_base_esr_beta is None:
                            base_esr_beta = float(tale_budget_config_for_da.get("esr_beta", 0.2))
                        else:
                            base_esr_beta = float(configured_base_esr_beta)
                        _apply_difficulty_aware_opd_routing(
                            batch=batch,
                            reward_tensor=reward_tensor,
                            difficulty_config=difficulty_config,
                            base_esr_beta=base_esr_beta,
                            metrics=metrics,
                        )

                    tale_budget_hard_truncated = False
                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            tale_budget_config = self.config.algorithm.get("tale_budget", None)
                            if tale_budget_config and tale_budget_config.get("enabled", False):
                                teacher_prompt_key = tale_budget_config.get("teacher_prompt_key", "teacher_prompt")
                                if self.ref_raw_prompt_key != teacher_prompt_key:
                                    raise ValueError(
                                        "algorithm.tale_budget.enabled=True requires "
                                        f"data.ref_raw_prompt_key={teacher_prompt_key!r}; "
                                        f"got {self.ref_raw_prompt_key!r}."
                                    )
                                if not self.use_ref_retokenization:
                                    raise ValueError(
                                        "algorithm.tale_budget.enabled=True requires ref re-tokenization so "
                                        "teacher/ref log-prob uses the generated budget-aware prompt."
                                    )
                                with marked_timer("tale_budget", timing_raw, color="orange"):
                                    tale_budget_metrics = _apply_online_tale_budget_prompts(
                                        batch=batch,
                                        actor_rollout_wg=self.actor_rollout_wg,
                                        tokenizer=self.tokenizer,
                                        tale_budget_config=tale_budget_config,
                                        max_prompt_length=self.config.data.max_prompt_length,
                                        truncation=self.config.data.get("truncation", "error"),
                                        apply_chat_template_kwargs=self.config.data.get(
                                            "apply_chat_template_kwargs", {}
                                        ),
                                    )
                                metrics.update(tale_budget_metrics)
                                probe_enabled = _rethinking_probe_enabled(self.config)
                                if probe_enabled and self.use_ref_retokenization:
                                    from verl.trainer.ppo.ref_input_utils import prepare_ref_model_inputs

                                    apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
                                    batch = prepare_ref_model_inputs(
                                        batch=batch,
                                        ref_tokenizer=self.ref_tokenizer,
                                        apply_chat_template_kwargs=apply_chat_template_kwargs,
                                        raw_prompt_key=self.ref_raw_prompt_key,
                                    )
                                    _set_rethinking_probe_meta(batch, self.config)
                                    if not self.ref_in_actor:
                                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                    else:
                                        ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                                    batch = batch.union(ref_log_prob)
                                    _log_rethinking_opd_probe_metrics(
                                        batch,
                                        config=self.config,
                                        global_step=self.global_steps,
                                        metrics=metrics,
                                    )

                                if tale_budget_config.get("truncate_to_esr", False):
                                    candidate_selection_config = self.config.algorithm.get("candidate_selection", None)
                                    if candidate_selection_config is not None and candidate_selection_config.get(
                                        "enabled", False
                                    ):
                                        raise ValueError(
                                            "algorithm.tale_budget.truncate_to_esr=True is not compatible with "
                                            "algorithm.candidate_selection.enabled=True because candidate selection "
                                            "needs full-response teacher scores."
                                        )
                                    batch, truncate_metrics = truncate_to_tale_budget_esr(batch)
                                    metrics.update(truncate_metrics)
                                    tale_budget_hard_truncated = True
                            else:
                                probe_enabled = _rethinking_probe_enabled(self.config)

                            # Get apply_chat_template_kwargs from config if available
                            apply_chat_template_kwargs = self.config.data.get(
                                "apply_chat_template_kwargs", {}
                            )
                            ref_log_prob_already_computed = probe_enabled and "ref_log_prob" in batch.batch.keys()

                            # If ref model uses different tokenizer/prompt template, re-tokenize inputs for ref model
                            if self.use_ref_retokenization:
                                from verl.trainer.ppo.ref_input_utils import prepare_ref_model_inputs

                                if not ref_log_prob_already_computed:
                                    batch = prepare_ref_model_inputs(
                                        batch=batch,
                                        ref_tokenizer=self.ref_tokenizer,
                                        apply_chat_template_kwargs=apply_chat_template_kwargs,
                                        raw_prompt_key=self.ref_raw_prompt_key,
                                    )

                                    if not self.ref_in_actor:
                                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                    else:
                                        ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                                    batch = batch.union(ref_log_prob)
                                    _log_rethinking_opd_probe_metrics(
                                        batch,
                                        config=self.config,
                                        global_step=self.global_steps,
                                        metrics=metrics,
                                    )
                                if tale_budget_hard_truncated:
                                    drop_ref_retokenization_tensors(batch)

                            else:
                                # Standard ref model log prob computation
                                if not self.ref_in_actor:
                                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                else:
                                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)
                                _log_rethinking_opd_probe_metrics(
                                    batch,
                                    config=self.config,
                                    global_step=self.global_steps,
                                    metrics=metrics,
                                )
                                if tale_budget_hard_truncated:
                                    drop_ref_retokenization_tensors(batch)

                    # Compute code teacher log probs (and entropy) for multi-teacher distillation
                    if self.use_base_models:
                        with marked_timer("base_ref_log_prob", timing_raw, color="green"):
                            if not self.ref_in_actor:
                                base_ref_log_prob = self.ref_policy_wg.compute_base_ref_log_prob(batch)
                            else:
                                base_ref_log_prob = self.actor_rollout_wg.compute_base_ref_log_prob(batch)
                            batch = batch.union(base_ref_log_prob)
                    
                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async and reward_tensor is None:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        if tale_budget_hard_truncated:
                            reward_tensor = _truncate_response_tensor_to_batch(reward_tensor, batch)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        candidate_selection_config = self.config.algorithm.get("candidate_selection", None)
                        if candidate_selection_config is not None and candidate_selection_config.get("enabled", False):
                            batch, candidate_selection_metrics = select_short_correct_candidates(
                                batch=batch,
                                selection_config=candidate_selection_config,
                            )
                            metrics.update(candidate_selection_metrics)
                            batch.meta_info["global_token_num"] = torch.sum(
                                batch.batch["attention_mask"], dim=-1
                            ).tolist()

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
