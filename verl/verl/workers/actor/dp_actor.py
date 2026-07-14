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
Single Process Actor
"""

import logging
import os

import numpy as np
import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.trainer.ppo.opd_proxy_verify_capture import (
    attach_and_validate_keys,
    compute_authoritative_capture_tensors,
    recursive_parameter_sha256,
    resolve_capture_resume_prefix,
    write_tensor_chunks_atomic,
)
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


_LENGTH_PENALTY_TYPES = {"log_batch_median"}
_LENGTH_PENALTY_GATES = {"incorrect", "low_teacher", "incorrect_or_low_teacher", "correct"}


def _policy_loss_get(policy_loss_config, name: str, default):
    if hasattr(policy_loss_config, "get"):
        return policy_loss_config.get(name, default)
    return getattr(policy_loss_config, name, default)


def _compute_length_aware_opd_tensors(
    response_mask: torch.Tensor,
    ref_log_prob: torch.Tensor,
    token_level_scores: torch.Tensor | None,
    policy_loss_config,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Compute mini-batch-scoped gated length penalties for original OPD."""
    penalty_type = _policy_loss_get(policy_loss_config, "length_penalty_type", "log_batch_median")
    if penalty_type not in _LENGTH_PENALTY_TYPES:
        raise ValueError(
            f"Invalid length_penalty_type: {penalty_type}. Supported values: {sorted(_LENGTH_PENALTY_TYPES)}"
        )

    penalty_gate = _policy_loss_get(policy_loss_config, "length_penalty_gate", "incorrect_or_low_teacher")
    if penalty_gate not in _LENGTH_PENALTY_GATES:
        raise ValueError(
            f"Invalid length_penalty_gate: {penalty_gate}. Supported values: {sorted(_LENGTH_PENALTY_GATES)}"
        )

    needs_scores = penalty_gate in {"incorrect", "incorrect_or_low_teacher", "correct"}
    if needs_scores and token_level_scores is None:
        raise ValueError("token_level_scores is required when length_penalty_gate uses correctness")

    response_mask = response_mask.float()
    response_len = response_mask.sum(dim=-1).clamp(min=1.0)
    reference_len = response_len.median().clamp(min=1.0)
    base_penalty = torch.log(response_len / reference_len).clamp(min=0.0)

    normalized_teacher_logprob = (ref_log_prob * response_mask).sum(dim=-1) / response_len
    teacher_reject_percentile = float(_policy_loss_get(policy_loss_config, "length_teacher_reject_percentile", 20.0))
    if teacher_reject_percentile < 0.0 or teacher_reject_percentile > 100.0:
        raise ValueError("length_teacher_reject_percentile must be between 0.0 and 100.0")

    if teacher_reject_percentile == 0.0:
        teacher_reject = torch.zeros_like(response_len, dtype=torch.bool)
        teacher_threshold = torch.tensor(float("nan"), device=response_mask.device)
    else:
        teacher_threshold = torch.quantile(normalized_teacher_logprob.float(), teacher_reject_percentile / 100.0)
        teacher_reject = normalized_teacher_logprob <= teacher_threshold

    if token_level_scores is not None:
        seq_reward = (token_level_scores.float() * response_mask).sum(dim=-1)
    else:
        seq_reward = torch.zeros_like(response_len)
    correct_threshold = float(_policy_loss_get(policy_loss_config, "length_correct_reward_threshold", 0.5))
    incorrect = seq_reward <= correct_threshold
    correct = ~incorrect

    if penalty_gate == "incorrect":
        gate = incorrect
    elif penalty_gate == "low_teacher":
        gate = teacher_reject
    elif penalty_gate == "correct":
        gate = correct
    else:
        gate = incorrect | teacher_reject

    applied_penalty = base_penalty * gate.float()
    coef = float(_policy_loss_get(policy_loss_config, "length_penalty_coef", 0.0))
    scaled_penalty = applied_penalty * coef

    tensors = {
        "length_aware_opd_penalty": scaled_penalty.detach(),
        "length_aware_opd_base_penalty": base_penalty.detach(),
        "length_aware_opd_applied_penalty": applied_penalty.detach(),
        "length_aware_opd_penalty_gate": gate.float().detach(),
        "length_aware_opd_correct_mask": correct.float().detach(),
        "length_aware_opd_teacher_reject": teacher_reject.float().detach(),
        "length_aware_opd_response_len": response_len.detach(),
        "length_aware_opd_reference_len": reference_len.detach().expand_as(response_len),
        "length_aware_opd_teacher_threshold": teacher_threshold.detach().expand_as(response_len),
    }
    metrics = _summarize_length_aware_opd_tensors(tensors, coef=coef)
    return tensors, metrics


def _summarize_length_aware_opd_tensors(tensors: dict[str, torch.Tensor], coef: float) -> dict[str, float]:
    response_len = tensors["length_aware_opd_response_len"]
    reference_len = tensors["length_aware_opd_reference_len"]
    base_penalty = tensors["length_aware_opd_base_penalty"]
    applied_penalty = tensors["length_aware_opd_applied_penalty"]
    penalty_gate = tensors["length_aware_opd_penalty_gate"]
    correct_mask = tensors["length_aware_opd_correct_mask"]
    teacher_reject = tensors["length_aware_opd_teacher_reject"]
    return {
        "length_aware_opd/mean_response_len": response_len.float().mean().item(),
        "length_aware_opd/median_response_len": reference_len.float().median().item(),
        "length_aware_opd/mean_base_penalty": base_penalty.float().mean().item(),
        "length_aware_opd/mean_applied_penalty": applied_penalty.float().mean().item(),
        "length_aware_opd/max_applied_penalty": applied_penalty.float().max().item(),
        "length_aware_opd/penalty_gate_ratio": penalty_gate.float().mean().item(),
        "length_aware_opd/correct_skip_ratio": ((1.0 - penalty_gate) * correct_mask).float().mean().item(),
        "length_aware_opd/correct_penalty_ratio": (penalty_gate * correct_mask).float().mean().item(),
        "length_aware_opd/teacher_reject_ratio": teacher_reject.float().mean().item(),
        "length_aware_opd/coef": float(coef),
    }


def _add_length_aware_opd_tensors(mini_batch: DataProto, policy_loss_config) -> dict[str, float]:
    token_level_scores = None
    if "token_level_scores" in mini_batch.batch.keys():
        token_level_scores = mini_batch.batch["token_level_scores"]
    elif "token_level_rewards" in mini_batch.batch.keys():
        token_level_scores = mini_batch.batch["token_level_rewards"]

    tensors, metrics = _compute_length_aware_opd_tensors(
        response_mask=mini_batch.batch["response_mask"],
        ref_log_prob=mini_batch.batch["ref_log_prob"],
        token_level_scores=token_level_scores,
        policy_loss_config=policy_loss_config,
    )
    for key, value in tensors.items():
        mini_batch.batch[key] = value
    return metrics


def _apply_length_aware_opd_penalty(advantages: torch.Tensor, model_inputs: dict) -> torch.Tensor:
    if "length_aware_opd_penalty" not in model_inputs:
        raise ValueError("length_aware_opd_penalty missing from actor micro-batch")
    penalty = model_inputs["length_aware_opd_penalty"].to(device=advantages.device, dtype=advantages.dtype)
    return advantages - penalty.unsqueeze(-1)


def _apply_candidate_selection_loss_mask(response_mask: torch.Tensor, model_inputs: dict) -> torch.Tensor:
    """Zero response-mask rows for selected candidates that should not contribute actor loss."""
    loss_mask = model_inputs.get("candidate_selection_loss_mask", None)
    if loss_mask is None:
        return response_mask
    loss_mask = loss_mask.to(device=response_mask.device, dtype=response_mask.dtype)
    return response_mask * loss_mask.view(-1, *([1] * (response_mask.dim() - 1)))


def _apply_tale_budget_esr_loss_mask(response_mask: torch.Tensor, model_inputs: dict) -> torch.Tensor:
    """Keep only rollout-length adaptive ESR-supervised response tokens for actor loss."""
    esr_loss_mask = model_inputs.get("tale_budget_esr_loss_mask", None)
    if esr_loss_mask is None:
        return response_mask
    esr_loss_mask = esr_loss_mask.to(device=response_mask.device, dtype=response_mask.dtype)
    return response_mask * esr_loss_mask


def _compute_difficulty_aware_entropy_loss(
    *,
    entropy: torch.Tensor,
    response_mask: torch.Tensor,
    entropy_weight: torch.Tensor,
    loss_agg_mode: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    entropy_weight = entropy_weight.to(device=entropy.device, dtype=entropy.dtype)
    if entropy_weight.dim() != 1 or entropy_weight.shape[0] != entropy.shape[0]:
        raise ValueError("difficulty_aware_entropy_weight must have shape [batch]")
    if torch.all(entropy_weight <= 0):
        zero = entropy.sum() * 0.0
        return zero, {"difficulty_aware_opd/hard_entropy_weight_mean": 0.0, "difficulty_aware_opd/hard_entropy_loss": 0.0}

    weighted_entropy = entropy * entropy_weight.unsqueeze(-1)
    entropy_loss = -agg_loss(loss_mat=weighted_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    metrics = {
        "difficulty_aware_opd/hard_entropy_weight_mean": entropy_weight.float().mean().item(),
        "difficulty_aware_opd/hard_entropy_loss": entropy_loss.detach().item(),
    }
    return entropy_loss, metrics



def _maybe_compute_rollout_corr_metrics(
    loss_mode: str,
    log_prob: torch.Tensor,
    rollout_log_prob: torch.Tensor | None,
    response_mask: torch.Tensor,
) -> dict[str, float]:
    """Compute rollout-correction metrics unless candidate masking removed every valid token."""
    if loss_mode == "rollout_correction" or rollout_log_prob is None:
        return {}

    # Always emit this key when rollout-log-prob diagnostics are active so every
    # actor worker returns lists with the same length during cross-worker metric
    # reduction, even if only some microbatches are fully candidate-masked.
    metrics: dict[str, float] = {"rollout_corr/skipped_no_valid_tokens": 0.0}

    if not response_mask.bool().any().item():
        metrics["rollout_corr/skipped_no_valid_tokens"] = 1.0
        return metrics

    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

    metrics.update(
        compute_rollout_corr_metrics_from_logprobs(
            log_prob=log_prob,
            rollout_log_prob=rollout_log_prob,
            response_mask=response_mask,
        )
    )
    return metrics


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

    def _forward_micro_batch(
        self,
        micro_batch,
        temperature,
        calculate_entropy=False,
        top_k: int = 0,
        student_top_k_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor, dict[str, torch.Tensor]]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        probe_tensors: dict[str, torch.Tensor] = {}
        need_logits_for_probe = top_k > 0 or student_top_k_ids is not None
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        def _store_student_topk(logits: torch.Tensor, *, rmpad: bool = False) -> None:
            if top_k <= 0 or student_top_k_ids is not None:
                return
            from verl.trainer.ppo.rethinking_opd_probe import compute_student_topk_from_logits

            topk_ids, topk_log_probs = compute_student_topk_from_logits(logits, top_k=top_k)
            if rmpad:
                if self.use_ulysses_sp:
                    topk_ids = gather_outputs_and_unpad(
                        topk_ids,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    topk_log_probs = gather_outputs_and_unpad(
                        topk_log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                topk_ids = pad_input(
                    hidden_states=topk_ids,
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                topk_log_probs = pad_input(
                    hidden_states=topk_log_probs,
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                topk_ids = topk_ids[:, -response_length - 1 : -1, :]
                topk_log_probs = topk_log_probs[:, -response_length - 1 : -1, :]
            probe_tensors["student_top_k_ids"] = topk_ids
            probe_tensors["student_top_k_log_probs"] = topk_log_probs

        def _store_teacher_topk_overlap(logits: torch.Tensor, *, rmpad: bool = False) -> None:
            if student_top_k_ids is None:
                return
            from verl.trainer.ppo.rethinking_opd_probe import compute_teacher_topk_overlap

            if rmpad:
                full_student_top_k_ids = torch.zeros(
                    (batch_size, seqlen, student_top_k_ids.shape[-1]),
                    dtype=student_top_k_ids.dtype,
                    device=student_top_k_ids.device,
                )
                full_student_top_k_ids[:, -response_length - 1 : -1, :] = student_top_k_ids
                packed_student_top_k_ids = index_first_axis(
                    rearrange(full_student_top_k_ids, "b s k -> (b s) k"),
                    indices,
                )
                if self.use_ulysses_sp:
                    packed_student_top_k_ids, _, _ = ulysses_pad_and_slice_inputs(
                        packed_student_top_k_ids.transpose(0, 1),
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )
                    packed_student_top_k_ids = packed_student_top_k_ids.transpose(0, 1)
                teacher_overlap = compute_teacher_topk_overlap(logits, packed_student_top_k_ids, top_k=top_k)
                for key, value in teacher_overlap.items():
                    if self.use_ulysses_sp:
                        value = gather_outputs_and_unpad(
                            value,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    value = pad_input(
                        hidden_states=value,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    probe_tensors[key] = value[:, -response_length - 1 : -1, :]
                return

            teacher_overlap = compute_teacher_topk_overlap(
                logits.reshape(-1, logits.shape[-1]),
                student_top_k_ids.reshape(-1, student_top_k_ids.shape[-1]),
                top_k=top_k,
            )
            bsz, resp_len, _ = student_top_k_ids.shape
            for key, value in teacher_overlap.items():
                probe_tensors[key] = value.view(bsz, resp_len, value.shape[-1])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            # reset input_ids, attention_mask, position_ids to ref model inputs if ref model input_ids is different from actor input_ids
            if "ref_input_ids" in micro_batch.keys():
                input_ids = micro_batch["ref_input_ids"]
                attention_mask = micro_batch["ref_attention_mask"]
                position_ids = micro_batch["ref_position_ids"]
                batch_size, seqlen = input_ids.shape

            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True
                    if need_logits_for_probe:
                        extra_args["output_hidden_states"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)
                    if need_logits_for_probe:
                        if output.hidden_states is None:
                            raise RuntimeError("student top-k probe requires hidden_states from the fused actor forward")
                        actor_module = getattr(self.actor_module, "module", self.actor_module)
                        logits_rmpad = torch.matmul(output.hidden_states[-1], actor_module.lm_head.weight.t())
                        logits_rmpad.div_(temperature)
                        logits_rmpad = logits_rmpad.squeeze(0)
                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                if need_logits_for_probe:
                    _store_student_topk(logits_rmpad, rmpad=True)
                    _store_teacher_topk_overlap(logits_rmpad, rmpad=True)

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True
                    if need_logits_for_probe:
                        extra_args["output_hidden_states"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)
                    if need_logits_for_probe:
                        if output.hidden_states is None:
                            raise RuntimeError("student top-k probe requires hidden_states from the fused actor forward")
                        actor_module = getattr(self.actor_module, "module", self.actor_module)
                        logits = torch.matmul(output.hidden_states[-1], actor_module.lm_head.weight.t())
                        logits.div_(temperature)
                        logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                        _store_student_topk(logits, rmpad=False)
                        _store_teacher_topk_overlap(logits, rmpad=False)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
                    if need_logits_for_probe:
                        _store_student_topk(logits, rmpad=False)
                        _store_teacher_topk_overlap(logits, rmpad=False)

            return entropy, log_probs, probe_tensors

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            if not torch.isfinite(grad_norm):
                print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
                self.actor_optimizer.zero_grad()
            else:
                self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        top_k = int(data.meta_info.get("rethinking_opd_probe_top_k", 0) or 0)
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        has_ref_input_ids = "ref_input_ids" in data.batch.keys() # handle when ref input_ids is different from actor input_ids
        has_student_top_k_ids = "student_top_k_ids" in data.batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if has_ref_input_ids:
            select_keys.extend(["ref_input_ids", "ref_attention_mask", "ref_position_ids"])
        if has_student_top_k_ids:
            select_keys.append("student_top_k_ids")
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        probe_tensor_lists: dict[str, list[torch.Tensor]] = {}
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                mb_student_top_k_ids = model_inputs.get("student_top_k_ids")
                entropy, log_probs, probe_tensors = self._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    top_k=top_k,
                    student_top_k_ids=mb_student_top_k_ids,
                )
            for key, value in probe_tensors.items():
                probe_tensor_lists.setdefault(key, []).append(value)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        extra_tensors = {}
        for key, values in probe_tensor_lists.items():
            tensor = torch.concat(values, dim=0)
            if use_dynamic_bsz:
                tensor = restore_dynamic_batch(tensor, batch_idx_list)
            extra_tensors[key] = tensor

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        if extra_tensors:
            return log_probs, entropys, extra_tensors
        return log_probs, entropys

    def _compute_entropy_aware_loss(
        self,
        old_log_prob,
        log_prob,
        ref_log_prob,
        response_mask,
        student_entropys,
        ref_entropys,
        loss_agg_mode,
        base_ref_log_prob=None,
        base_ref_entropys=None,
        opd_teacher=None,
    ):
        """FiRe-OPD: Entropy-aware distillation with trajectory filtering and token-level adaptive weighting.

        Supports single-teacher and multi-teacher distillation.

        Args:
            old_log_prob: (bsz, response_length) - student log probs (detached)
            log_prob: (bsz, response_length) - student log probs (with grad)
            ref_log_prob: (bsz, response_length) - teacher log probs (math teacher in multi-teacher mode)
            response_mask: (bsz, response_length) - mask for valid tokens
            student_entropys: (bsz, response_length) - student per-token entropy
            ref_entropys: (bsz, response_length) - teacher per-token entropy (math teacher)
            loss_agg_mode: str - loss aggregation mode
            base_ref_log_prob: (bsz, response_length) - code teacher log probs in multi-teacher mode (optional)
            base_ref_entropys: (bsz, response_length) - code teacher per-token entropy (optional)
            opd_teacher: list/tuple of str - per-sample teacher type ("math"/"code") for multi-teacher (optional)

        Returns:
            loss: scalar loss
            metrics: dict of metrics
        """
        plc = self.config.policy_loss
        bsz, seq_len = old_log_prob.shape
        multi_teacher = getattr(plc, 'multi_teacher_distill', False)

        # ============================================================
        # Step 1: Trajectory-level filtering
        # Skip bottom N% trajectories by normalized teacher log prob.
        # In multi-teacher mode, use the correct teacher's log prob per sample.
        # ============================================================

        seq_lengths = response_mask.sum(dim=-1).clamp(min=1)  # (bsz,)

        if multi_teacher and opd_teacher is not None and base_ref_log_prob is not None:
            normalized_teacher_logprob = torch.zeros(bsz, device=ref_log_prob.device)
            for i in range(bsz):
                teacher_type = opd_teacher[i] if isinstance(opd_teacher, (list, tuple)) else opd_teacher
                if teacher_type == "code":
                    normalized_teacher_logprob[i] = (base_ref_log_prob[i] * response_mask[i]).sum() / seq_lengths[i]
                else:
                    normalized_teacher_logprob[i] = (ref_log_prob[i] * response_mask[i]).sum() / seq_lengths[i]
        else:
            normalized_teacher_logprob = (ref_log_prob * response_mask).sum(dim=-1) / seq_lengths

        logprob_threshold = torch.quantile(
            normalized_teacher_logprob.float(), plc.traj_skip_percentile / 100.0
        )
        traj_keep_mask = normalized_teacher_logprob >= logprob_threshold  # (bsz,)

        # ============================================================
        # Step 2: Compute entropy-based continuous token weights
        # weight = (1 + α·teacher_confidence) × (1 + β·student_confusion)
        # In multi-teacher mode, use the correct teacher's entropy per sample.
        # ============================================================

        # For multi-teacher: combine math and code teacher entropy per sample
        if multi_teacher and opd_teacher is not None and base_ref_entropys is not None:
            combined_teacher_entropys = ref_entropys.clone()
            for i in range(bsz):
                teacher_type = opd_teacher[i] if isinstance(opd_teacher, (list, tuple)) else opd_teacher
                if teacher_type == "code":
                    combined_teacher_entropys[i] = base_ref_entropys[i]
            teacher_entropys = combined_teacher_entropys
        else:
            teacher_entropys = ref_entropys

        # Teacher confidence: normalize entropy to [0,1], invert
        valid_teacher_entropys = teacher_entropys[response_mask.bool()]
        if valid_teacher_entropys.numel() > 0:
            teacher_entropy_max = valid_teacher_entropys.max().clamp(min=1e-6)
        else:
            teacher_entropy_max = torch.tensor(1.0, device=teacher_entropys.device)
        teacher_confidence = (1.0 - teacher_entropys / teacher_entropy_max).clamp(min=0.0, max=1.0)

        # Student confusion: normalize entropy to [0,1]
        valid_student_entropys = student_entropys[response_mask.bool()]
        if valid_student_entropys.numel() > 0:
            student_entropy_max = valid_student_entropys.max().clamp(min=1e-6)
        else:
            student_entropy_max = torch.tensor(1.0, device=student_entropys.device)
        student_confusion = (student_entropys / student_entropy_max).clamp(min=0.0, max=1.0)

        # Continuous weight (detached — no gradient through weights)
        alpha = getattr(plc, 'entropy_alpha', 1.0)
        beta = getattr(plc, 'entropy_beta', 1.0)
        token_weight = ((1.0 + alpha * teacher_confidence) * (1.0 + beta * student_confusion)).detach()

        # Normalize token_weight to mean=1.0 over valid tokens
        valid_weight_sum = (token_weight * response_mask).sum()
        valid_token_count = response_mask.sum().clamp(min=1)
        valid_weight_mean = valid_weight_sum / valid_token_count
        token_weight = token_weight / valid_weight_mean.clamp(min=1e-6)

        # ============================================================
        # Step 3: Weighted advantages + standard PPO policy gradient
        # ============================================================

        # Compute reverse KL advantages: multi-teacher routes to correct teacher per sample
        if multi_teacher and base_ref_log_prob is not None and opd_teacher is not None:
            reverse_kl = torch.zeros_like(old_log_prob)
            for i in range(bsz):
                teacher_type = opd_teacher[i] if isinstance(opd_teacher, (list, tuple)) else opd_teacher
                if teacher_type == "code":
                    reverse_kl[i] = old_log_prob[i] - base_ref_log_prob[i]
                else:
                    reverse_kl[i] = old_log_prob[i] - ref_log_prob[i]
            advantages = -reverse_kl
        else:
            advantages = -(old_log_prob - ref_log_prob)

        # Apply token-level entropy weight to advantages
        weighted_advantages = (token_weight * advantages).detach()

        # Apply trajectory-level filtering
        traj_keep_expanded = traj_keep_mask.unsqueeze(1).expand_as(weighted_advantages).float()
        weighted_advantages = weighted_advantages * traj_keep_expanded

        # PPO ratio
        negative_approx_kl = log_prob - old_log_prob
        negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
        ratio = torch.exp(negative_approx_kl)

        # Clipped PPO loss
        clip_ratio = getattr(self.config, 'clip_ratio', 0.2)
        pg_losses1 = -weighted_advantages * ratio
        pg_losses2 = -weighted_advantages * torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
        pg_loss_mat = torch.maximum(pg_losses1, pg_losses2)

        # Aggregate with mask
        effective_mask = response_mask * traj_keep_expanded
        loss = agg_loss(loss_mat=pg_loss_mat, loss_mask=effective_mask, loss_agg_mode=loss_agg_mode)

        # ============================================================
        # Metrics
        # ============================================================
        metrics = {}
        with torch.no_grad():
            metrics["fire_opd/traj_keep_ratio"] = traj_keep_mask.float().mean().item()
            metrics["fire_opd/logprob_threshold"] = logprob_threshold.item()
            metrics["fire_opd/normalized_teacher_logprob_mean"] = normalized_teacher_logprob.mean().item()

            valid_mask = effective_mask.bool()
            valid_weights = token_weight[valid_mask]
            if valid_weights.numel() > 0:
                metrics["fire_opd/token_weight_mean"] = valid_weights.mean().item()
                metrics["fire_opd/token_weight_max"] = valid_weights.max().item()
                metrics["fire_opd/token_weight_min"] = valid_weights.min().item()

            valid_tc = teacher_confidence[valid_mask]
            valid_sc = student_confusion[valid_mask]
            if valid_tc.numel() > 0:
                metrics["fire_opd/teacher_confidence_mean"] = valid_tc.mean().item()
                metrics["fire_opd/student_confusion_mean"] = valid_sc.mean().item()

            if valid_teacher_entropys.numel() > 0:
                metrics["fire_opd/teacher_entropy_mean"] = valid_teacher_entropys.mean().item()
            if valid_student_entropys.numel() > 0:
                metrics["fire_opd/student_entropy_mean"] = valid_student_entropys.mean().item()

            ppo_kl = verl_F.masked_mean(-negative_approx_kl, effective_mask)
            pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), effective_mask)
            metrics["fire_opd/ppo_kl"] = ppo_kl.item()
            metrics["fire_opd/pg_clipfrac"] = pg_clipfrac.item()
            metrics["fire_opd/loss"] = loss.detach().item()

        return loss, metrics

    @GPUMemoryLogger(role="dp actor OPD proxy capture", logger=logger)
    def capture_opd_proxy_verify(self, data: DataProto) -> DataProto:
        """Capture authoritative actor tensors without backward or optimizer state."""
        if not self.config.get("opd_proxy_verify_capture_only", False):
            raise ValueError("actor is not configured for OPD proxy capture only")
        if self.actor_optimizer is not None:
            raise ValueError("capture-only actor must not have an optimizer")
        if self.config.ppo_epochs != 1:
            raise ValueError("capture actor requires exactly one PPO epoch")
        if self.config.use_dynamic_bsz:
            raise ValueError("capture actor forbids dynamic micro-batching")
        if self.config.ppo_micro_batch_size_per_gpu != 1:
            raise ValueError("capture actor requires micro-batch size one")
        if len(data) != self.config.ppo_mini_batch_size:
            raise ValueError("capture actor requires exactly one local mini-batch")
        frozen_actor_values = {
            "loss_agg_mode": "token-mean",
            "entropy_coeff": 0,
            "use_kl_loss": True,
            "kl_loss_coef": 0,
        }
        for field, expected in frozen_actor_values.items():
            if self.config.get(field) != expected:
                raise ValueError(
                    f"capture actor requires {field}={expected!r}"
                )
        if self.config.policy_loss.get("loss_mode", None) != "vanilla" or not self.config.policy_loss.get(
            "only_reverse_kl_advantages", False
        ):
            raise ValueError("capture actor requires vanilla reverse-KL policy loss")
        forbidden_prefixes = (
            "candidate_selection",
            "difficulty_aware",
            "length_aware_opd",
            "rethinking_opd",
            "tale_budget",
        )
        forbidden_exact = {
            "advantages",
            "returns",
            "token_level_rewards",
            "token_level_scores",
            "difficulty_aware_entropy_weight",
        }
        all_keys = set(data.batch.keys()) | set(data.non_tensor_batch)
        forbidden = sorted(
            key
            for key in all_keys
            if key in forbidden_exact or any(key.startswith(prefix) for prefix in forbidden_prefixes)
        )
        if forbidden:
            raise ValueError(f"forbidden capture batch keys: {forbidden}")
        required_tensors = {
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "ref_log_prob",
            "rollout_log_probs",
            "rollout_is_weights",
            "opd_proxy_verify_rollout_slot",
            "opd_proxy_verify_engine_seed",
        }
        missing = required_tensors - set(data.batch.keys())
        if missing:
            raise ValueError(f"capture actor missing tensors: {sorted(missing)}")
        required_non_tensors = {
            "opd_verify_stable_id",
            "opd_verify_split",
            "opd_verify_manifest_index",
        }
        missing_non_tensors = required_non_tensors - set(data.non_tensor_batch)
        if missing_non_tensors:
            raise ValueError(
                f"capture actor missing metadata: {sorted(missing_non_tensors)}"
            )

        engine_seed = int(data.meta_info["opd_proxy_verify_engine_seed"])
        data, local_keys = attach_and_validate_keys(
            data,
            engine_seed=engine_seed,
            native_rollouts=4,
            require_complete_slots=False,
        )
        output_root = str(data.meta_info["opd_proxy_verify_output_root"])
        parent_hashes = dict(data.meta_info["opd_proxy_verify_parent_hashes"])
        chunk_size = int(data.meta_info["opd_proxy_verify_chunk_size"])
        rank = torch.distributed.get_rank()
        rank_directory = os.path.join(output_root, "actor", f"rank_{rank}")
        resume_prefix = resolve_capture_resume_prefix(
            rank_directory,
            expected_keys=local_keys,
            parent_hashes=parent_hashes,
        )

        self.actor_module.train()
        if any(parameter.grad is not None for parameter in self.actor_module.parameters()):
            raise ValueError("capture actor started with materialized parameter gradients")
        parameter_hash_before = recursive_parameter_sha256(self.actor_module)
        temperature = float(data.meta_info["temperature"])
        tensor_keys = sorted(required_tensors)
        selected = data.select(
            batch_keys=tensor_keys,
            non_tensor_batch_keys=sorted(required_non_tensors),
        )
        for chunk_start in range(resume_prefix, len(selected), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(selected))
            tensor_lists: dict[str, list[torch.Tensor]] = {}
            sidecars: list[dict[str, object]] = []
            for row_index in range(chunk_start, chunk_end):
                micro_batch = selected[row_index : row_index + 1].to(get_device_id())
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                _, current_log_prob, _ = self._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=False,
                )
                authoritative = compute_authoritative_capture_tensors(
                    current_log_prob=current_log_prob,
                    batch_old_log_prob=model_inputs["old_log_probs"],
                    rollout_log_prob=model_inputs["rollout_log_probs"],
                    ref_log_prob=model_inputs["ref_log_prob"],
                    response_mask=model_inputs["response_mask"],
                    rollout_is_weights=model_inputs["rollout_is_weights"],
                    actor_config=self.config,
                )
                row_tensors = {
                    key: model_inputs[key]
                    for key in (
                        "responses",
                        "input_ids",
                        "attention_mask",
                        "position_ids",
                    )
                }
                row_tensors.update(
                    {
                        name: value.reshape(1) if value.ndim == 0 else value
                        for name, value in authoritative.items()
                    }
                )
                for name, value in row_tensors.items():
                    tensor_lists.setdefault(name, []).append(value.detach().cpu())
                sidecars.append(
                    {
                        "stable_id": str(
                            micro_batch.non_tensor_batch["opd_verify_stable_id"][0]
                        ),
                        "engine_seed": engine_seed,
                        "rollout_slot": int(
                            micro_batch.batch["opd_proxy_verify_rollout_slot"][0]
                        ),
                        "split": str(
                            micro_batch.non_tensor_batch["opd_verify_split"][0]
                        ),
                        "manifest_index": int(
                            micro_batch.non_tensor_batch[
                                "opd_verify_manifest_index"
                            ][0]
                        ),
                        "actor_rank": rank,
                    }
                )
                del authoritative, current_log_prob, micro_batch, model_inputs
            chunk_tensors = {
                name: torch.cat(values, dim=0) for name, values in tensor_lists.items()
            }
            write_tensor_chunks_atomic(
                rank_directory,
                tensors=chunk_tensors,
                sidecar_rows=sidecars,
                parent_hashes=parent_hashes,
                chunk_size=chunk_end - chunk_start,
                start_index=chunk_start,
            )

        if any(parameter.grad is not None for parameter in self.actor_module.parameters()):
            raise ValueError("capture actor materialized parameter gradients")
        parameter_hash_after = recursive_parameter_sha256(self.actor_module)
        if parameter_hash_before != parameter_hash_after:
            raise ValueError("actor parameter hash changed during capture")
        count = len(data)
        return DataProto(
            batch=data.batch.select(
                "opd_proxy_verify_rollout_slot",
                "opd_proxy_verify_engine_seed",
            ).update(
                {
                    "opd_proxy_verify_actor_rank": torch.full(
                        (count,), rank, dtype=torch.long
                    )
                }
            ),
            non_tensor_batch={
                "opd_verify_stable_id": data.non_tensor_batch[
                    "opd_verify_stable_id"
                ].copy(),
                "opd_proxy_verify_parameter_sha256_before": np.asarray(
                    [parameter_hash_before] * count, dtype=object
                ),
                "opd_proxy_verify_parameter_sha256_after": np.asarray(
                    [parameter_hash_after] * count, dtype=object
                ),
            },
        ).to("cpu")

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        if self.config.get("opd_proxy_verify_capture_only", False):
            raise RuntimeError("update_policy is forbidden in OPD proxy capture-only mode")
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")
        if "difficulty_aware_entropy_weight" in data.batch.keys():
            select_keys.append("difficulty_aware_entropy_weight")
         # Include code teacher log probs for multi-teacher distillation
        if "base_ref_log_prob" in data.batch.keys():
            select_keys.append("base_ref_log_prob")
        # Include ref_log_prob for only_reverse_kl_advantages mode
        if self.config.policy_loss.only_reverse_kl_advantages and "ref_log_prob" in data.batch.keys():
            if "ref_log_prob" not in select_keys:
                select_keys.append("ref_log_prob")

        # Include entropy tensors for entropy-aware distillation
        entropy_aware = getattr(self.config.policy_loss, "entropy_aware_distill", False)
        difficulty_aware_entropy = "difficulty_aware_entropy_weight" in data.batch.keys()
        length_aware_opd = (
            getattr(self.config.policy_loss, "length_aware_opd", False)
            and self.config.policy_loss.only_reverse_kl_advantages
            and not entropy_aware
        )
        if length_aware_opd:
            if "token_level_scores" in data.batch.keys() and "token_level_scores" not in select_keys:
                select_keys.append("token_level_scores")
            elif "token_level_rewards" in data.batch.keys() and "token_level_rewards" not in select_keys:
                select_keys.append("token_level_rewards")
        if entropy_aware:
            if "student_entropys" in data.batch.keys():
                select_keys.append("student_entropys")
            if "ref_entropys" in data.batch.keys():
                select_keys.append("ref_entropys")
            if "base_ref_entropys" in data.batch.keys():
                select_keys.append("base_ref_entropys")
            if "ref_log_prob" in data.batch.keys() and "ref_log_prob" not in select_keys:
                select_keys.append("ref_log_prob")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        # Include opd_teacher for multi-teacher distillation
        if "opd_teacher" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("opd_teacher")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if length_aware_opd:
                    length_aware_metrics = _add_length_aware_opd_tensors(
                        mini_batch=mini_batch,
                        policy_loss_config=self.config.policy_loss,
                    )
                    append_to_dict(metrics, length_aware_metrics)

                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    # For entropy-aware distillation, always compute current student entropy
                    calculate_entropy = entropy_coeff != 0 or entropy_aware or difficulty_aware_entropy
                    entropy, log_prob, _ = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    # for fully_async_policy recipe
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # ============================================================
                    # Entropy-aware distillation path
                    # ============================================================
                    if entropy_aware and "ref_entropys" in model_inputs and "student_entropys" in model_inputs:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        ref_entropys = model_inputs["ref_entropys"]
                        # Use pre-computed student entropy from compute_log_prob (more accurate, same forward pass)
                        student_entropys = model_inputs["student_entropys"]

                        entropy_loss, entropy_metrics = self._compute_entropy_aware_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            ref_log_prob=ref_log_prob,
                            response_mask=response_mask,
                            student_entropys=student_entropys,
                            ref_entropys=ref_entropys,
                            loss_agg_mode=loss_agg_mode,
                            base_ref_log_prob=model_inputs.get("base_ref_log_prob", None),
                            base_ref_entropys=model_inputs.get("base_ref_entropys", None),
                            opd_teacher=model_inputs.get("opd_teacher", None),
                        )
                        pg_loss = entropy_loss
                        micro_batch_metrics.update(entropy_metrics)

                    else:
                        # Vanilla OPD fallback (no entropy-aware weighting)
                        if self.config.policy_loss.only_reverse_kl_advantages and "ref_log_prob" in model_inputs:
                            advantages = -(old_log_prob - model_inputs["ref_log_prob"])
                            if length_aware_opd:
                                advantages = _apply_length_aware_opd_penalty(advantages, model_inputs)

                        policy_loss_fn = get_policy_loss_fn(loss_mode)
                        pg_loss, pg_metrics = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_is_weights=rollout_is_weights,
                        )
                        micro_batch_metrics.update(pg_metrics)

                    if "difficulty_aware_entropy_weight" in model_inputs:
                        difficulty_entropy_loss, difficulty_entropy_metrics = _compute_difficulty_aware_entropy_loss(
                            entropy=entropy,
                            response_mask=response_mask,
                            entropy_weight=model_inputs["difficulty_aware_entropy_weight"],
                            loss_agg_mode=loss_agg_mode,
                        )
                        pg_loss = pg_loss + difficulty_entropy_loss
                        micro_batch_metrics.update(difficulty_entropy_metrics)

                    # Skip if using pure rollout correction mode (metrics already in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "rollout_correction" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
