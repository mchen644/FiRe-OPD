"""Pure helpers for online TALE token-budget teacher prompts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from verl import DataProto

TALE_BUDGET_CONTEXT = (
    "Estimate how many tokens a strong math solver would need to write a complete, correct solution.\n"
    "Do NOT solve. Output only one line: Budget: [[N]]\n"
    "N must be an integer between 128 and 8192.\n"
    "Use larger N for proofs, multiple cases, advanced theorems, or long derivations."
)

_BRACKETED_BUDGET_RE = re.compile(r"\[\[\s*(\d+)\s*\]\]")
_BUDGET_LABEL_RE = re.compile(r"(?im)^\s*Budget\s*:\s*(?:\[\[\s*)?(\d+)(?:\s*\]\])?\s*$")
_PLAIN_BUDGET_RE = re.compile(r"^\s*(?:\[\[\s*)?(\d+)(?:\s*\]\])?\s*$")
_FIRE_OPD_VERBOSE_INSTRUCTION_RE = re.compile(
    r"\s*Please reason step by step, and put your final answer within \\boxed\{\}\.\s*$",
    flags=re.IGNORECASE,
)

_RESPONSE_ALIGNED_KEYS = {
    "responses",
    "response_mask",
    "tale_budget_esr_loss_mask",
    "old_log_probs",
    "ref_log_prob",
    "base_ref_log_prob",
    "rollout_log_probs",
    "rollout_is_weights",
    "advantages",
    "returns",
    "values",
    "token_level_scores",
    "token_level_rewards",
    "student_entropys",
    "ref_entropys",
    "base_ref_entropys",
}
_SEQUENCE_KEYS = {
    "input_ids",
    "attention_mask",
    "position_ids",
    "ref_input_ids",
    "ref_attention_mask",
    "ref_position_ids",
}
_REF_RETOKENIZATION_KEYS = ("ref_input_ids", "ref_attention_mask", "ref_position_ids")


@dataclass(frozen=True)
class RolloutLengthTaleBudgetResult:
    """Rollout-length-derived budget and ESR supervision mask."""

    budgets: torch.Tensor
    raw_budgets: torch.Tensor
    response_lengths: torch.Tensor
    esr_tokens: torch.Tensor
    esr_loss_mask: torch.Tensor


def strip_fire_opd_verbose_instruction(question: str) -> str:
    """Remove FiRe-OPD's default verbose-answer suffix before adding a budget suffix."""

    return _FIRE_OPD_VERBOSE_INSTRUCTION_RE.sub("", str(question)).strip()


def _slice_sequence_suffix(tensor: torch.Tensor, *, response_length: int, keep_length: int) -> torch.Tensor:
    prefix_length = tensor.shape[-1] - response_length
    if prefix_length < 0:
        raise ValueError("sequence tensor is shorter than response_length")
    return tensor[..., : prefix_length + keep_length]


def truncate_to_tale_budget_esr(batch: DataProto) -> tuple[DataProto, dict[str, float]]:
    """Physically truncate response-aligned tensors to the ESR training prefix.

    The original rollout can remain available to the caller for reward/metrics, while
    the returned batch contains only the max supervised prefix across rows. Rows with
    smaller ESR windows are padded by zeroing their response attention/loss masks.
    """

    if batch.batch is None or "responses" not in batch.batch.keys():
        raise ValueError("truncate_to_tale_budget_esr requires batch['responses']")
    if "response_mask" not in batch.batch.keys():
        raise ValueError("truncate_to_tale_budget_esr requires batch['response_mask']")
    if "tale_budget_esr_loss_mask" not in batch.batch.keys():
        raise ValueError("truncate_to_tale_budget_esr requires batch['tale_budget_esr_loss_mask']")

    response_length = batch.batch["responses"].shape[-1]
    esr_loss_mask = batch.batch["tale_budget_esr_loss_mask"]
    if esr_loss_mask.shape[-1] != response_length:
        raise ValueError("tale_budget_esr_loss_mask must have the same response length as responses")

    keep_length = int(esr_loss_mask.to(dtype=torch.long).sum(dim=-1).max().item())
    if response_length > 0:
        keep_length = max(1, min(keep_length, response_length))

    tensors: dict[str, torch.Tensor] = {}
    for key, tensor in batch.batch.items():
        if key in _RESPONSE_ALIGNED_KEYS and tensor.dim() >= 2 and tensor.shape[-1] == response_length:
            tensors[key] = tensor[..., :keep_length]
        elif key in _SEQUENCE_KEYS and tensor.dim() >= 2 and tensor.shape[-1] >= response_length:
            tensors[key] = _slice_sequence_suffix(tensor, response_length=response_length, keep_length=keep_length)
        else:
            tensors[key] = tensor

    truncated_response_mask = tensors["tale_budget_esr_loss_mask"].to(dtype=tensors["response_mask"].dtype)
    tensors["response_mask"] = truncated_response_mask
    for attention_key in ("attention_mask", "ref_attention_mask"):
        if attention_key in tensors:
            attention_mask = tensors[attention_key]
            prefix_length = attention_mask.shape[-1] - keep_length
            tensors[attention_key] = torch.cat([attention_mask[..., :prefix_length], truncated_response_mask], dim=-1)

    meta_info = dict(batch.meta_info)
    if "attention_mask" in tensors:
        meta_info["global_token_num"] = torch.sum(tensors["attention_mask"], dim=-1).tolist()

    original_tokens = batch.batch["response_mask"].float().sum().item()
    truncated_tokens = truncated_response_mask.float().sum().item()
    metrics = {
        "tale_budget/truncated_response_length": float(keep_length),
        "tale_budget/truncated_token_fraction": float(truncated_tokens / max(original_tokens, 1.0)),
    }

    return DataProto.from_dict(
        tensors=tensors,
        non_tensors=dict(batch.non_tensor_batch),
        meta_info=meta_info,
    ), metrics


def drop_ref_retokenization_tensors(batch: DataProto) -> None:
    """Drop temporary ref-tokenized inputs after ref log-prob computation."""

    keys = [key for key in _REF_RETOKENIZATION_KEYS if batch.batch is not None and key in batch.batch.keys()]
    if keys:
        batch.pop(batch_keys=keys)


def build_tale_budget_estimation_prompt(question: str) -> str:
    """Build the TALE budget-estimation prompt for the current student policy."""

    stripped_question = strip_fire_opd_verbose_instruction(question)
    return f"{TALE_BUDGET_CONTEXT}\n\nProblem:\n{stripped_question}"


def parse_tale_budget(text: Any) -> int | None:
    """Extract a TALE-style budget integer from generated text."""

    normalized_text = str(text).strip()
    for pattern in (_BRACKETED_BUDGET_RE, _BUDGET_LABEL_RE, _PLAIN_BUDGET_RE):
        match = pattern.search(normalized_text)
        if match is not None:
            return int(match.group(1))
    return None


def normalize_tale_budget(
    raw_budget: int | None,
    *,
    min_budget: int,
    max_budget: int,
    round_to: int,
    fallback_budget: int,
) -> int:
    """Clamp and round a raw TALE budget estimate to a trainer-safe token budget."""

    if min_budget <= 0:
        raise ValueError("min_budget must be positive")
    if max_budget < min_budget:
        raise ValueError("max_budget must be >= min_budget")
    if round_to <= 0:
        raise ValueError("round_to must be positive")

    value = fallback_budget if raw_budget is None else int(raw_budget)
    value = max(min_budget, min(max_budget, value))
    value = int(round(value / round_to) * round_to)
    return max(min_budget, min(max_budget, value))


def build_tale_budget_teacher_messages(question: str, budget: int) -> list[dict[str, str]]:
    """Build an in-memory budget-aware teacher prompt for ref log-prob scoring."""

    stripped_question = strip_fire_opd_verbose_instruction(question)
    content = (
        f"{stripped_question}\n"
        f"Let's think step by step and use less than {int(budget)} tokens. "
        r"Put your final answer within \boxed{}."
    )
    return [{"role": "user", "content": content}]



def build_concise_teacher_messages(question: str) -> list[dict[str, str]]:
    """Build a concise teacher prompt without a numeric token budget."""

    stripped_question = strip_fire_opd_verbose_instruction(question)
    content = (
        f"{stripped_question}\n"
        r"Solve concisely. Avoid unnecessary explanation. Put your final answer within \boxed{}."
    )
    return [{"role": "user", "content": content}]


def compute_rollout_length_tale_budget(
    *,
    response_mask: torch.Tensor,
    alpha: float,
    beta: float,
    min_budget: int,
    round_to: int,
    max_budget: int | None = None,
) -> RolloutLengthTaleBudgetResult:
    """Derive teacher budgets and ESR masks from current student rollout lengths.

    Budgets are `round(alpha * response_length)` and ESR supervision keeps the
    first `round(beta * budget)` valid response tokens. `max_budget=None` means
    there is no budget/N max cap; ESR tokens are still capped by the actual
    response length so the mask never extends into padding.
    """

    if response_mask.dim() != 2:
        raise ValueError("response_mask must be a 2D tensor")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    if min_budget <= 0:
        raise ValueError("min_budget must be positive")
    if round_to <= 0:
        raise ValueError("round_to must be positive")
    if max_budget is not None and max_budget < min_budget:
        raise ValueError("max_budget must be >= min_budget when provided")

    response_lengths = response_mask.to(dtype=torch.long).sum(dim=-1)
    raw_budgets = response_lengths.float() * float(alpha)
    budgets = torch.round(raw_budgets / float(round_to)).to(dtype=torch.long) * int(round_to)
    budgets = budgets.clamp(min=int(min_budget))
    if max_budget is not None:
        budgets = budgets.clamp(max=int(max_budget))

    esr_tokens = torch.round(budgets.float() * float(beta)).to(dtype=torch.long)
    esr_tokens = torch.minimum(esr_tokens.clamp(min=0), response_lengths)
    positions = torch.arange(response_mask.shape[-1], device=response_mask.device).unsqueeze(0)
    esr_loss_mask = ((positions < esr_tokens.unsqueeze(1)) & response_mask.bool()).to(dtype=response_mask.dtype)

    return RolloutLengthTaleBudgetResult(
        budgets=budgets,
        raw_budgets=raw_budgets,
        response_lengths=response_lengths,
        esr_tokens=esr_tokens,
        esr_loss_mask=esr_loss_mask,
    )


def summarize_tale_budget_metrics(raw_budgets: Sequence[int | None], budgets: Sequence[int]) -> dict[str, float]:
    """Summarize online TALE budget estimates for trainer metrics."""

    normalized_budgets = [int(budget) for budget in budgets]
    raw_budget_values = list(raw_budgets)
    parse_fail_denom = len(raw_budget_values) if raw_budget_values else len(normalized_budgets)
    parse_fail_count = sum(raw_budget is None for raw_budget in raw_budget_values)
    parse_fail_ratio = parse_fail_count / parse_fail_denom if parse_fail_denom else 0.0

    if not normalized_budgets:
        return {
            "tale_budget/mean": 0.0,
            "tale_budget/min": 0.0,
            "tale_budget/max": 0.0,
            "tale_budget/parse_fail_ratio": parse_fail_ratio,
        }

    return {
        "tale_budget/mean": sum(normalized_budgets) / len(normalized_budgets),
        "tale_budget/min": float(min(normalized_budgets)),
        "tale_budget/max": float(max(normalized_budgets)),
        "tale_budget/parse_fail_ratio": parse_fail_ratio,
    }
