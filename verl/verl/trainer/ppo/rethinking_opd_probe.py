"""Training-time Rethinking OPD probe metrics.

These helpers are pure tensor/CSV utilities. They do not trigger rollout or
model forward passes. Worker/trainer code supplies the tensors from the current
training batch.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch

CSV_FIELDS = [
    "run_name",
    "step",
    "chunk_start",
    "chunk_end",
    "top_k",
    "batch_size",
    "valid_sequence_count",
    "valid_token_count",
    "mean_all_batch_response_length",
    "max_all_batch_response_length",
    "clip_rate_all_batch",
    "topk_overlap_ratio",
    "student_overlap_mass",
    "teacher_overlap_mass",
    "student_entropy",
    "teacher_entropy",
    "entropy_gap",
    "overlap_token_advantage",
]

RETHINKING_OPD_PROBE_REQUIRED_TENSORS = (
    "student_top_k_log_probs",
    "teacher_on_student_log_probs",
    "overlap_mask",
)


def has_rethinking_opd_probe_tensors(tensors_or_batch_keys: Any) -> bool:
    """Return True when the required probe tensors are present."""

    if hasattr(tensors_or_batch_keys, "keys"):
        keys = tensors_or_batch_keys.keys()
    else:
        keys = tensors_or_batch_keys
    key_set = set(keys)
    return all(key in key_set for key in RETHINKING_OPD_PROBE_REQUIRED_TENSORS)


def compute_student_topk_from_logits(logits: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return top-k ids and log-probs from logits with shape `(..., vocab_size)`."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    top_k_logits, top_k_ids = torch.topk(logits, k=top_k, dim=-1)
    logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
    return top_k_ids, top_k_logits - logsumexp


def compute_teacher_topk_overlap(
    logits: torch.Tensor, student_top_k_ids: torch.Tensor, top_k: int
) -> dict[str, torch.Tensor]:
    """Compute teacher top-k tensors and overlap with provided student top-k ids.

    Args:
        logits: Teacher logits with shape `(N, vocab_size)`.
        student_top_k_ids: Student top-k token ids with shape `(N, k)`.
        top_k: Number of teacher top-k tokens. Must match the student k used by the probe.
    """

    if logits.dim() != 2:
        raise ValueError("logits must have shape (N, vocab_size)")
    if student_top_k_ids.dim() != 2:
        raise ValueError("student_top_k_ids must have shape (N, k)")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if student_top_k_ids.shape[-1] != top_k:
        raise ValueError("student_top_k_ids last dimension must equal top_k")

    teacher_top_k_logits, teacher_top_k_ids = torch.topk(logits, k=top_k, dim=-1)
    teacher_logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
    teacher_top_k_log_probs = teacher_top_k_logits - teacher_logsumexp
    teacher_on_student_log_probs = torch.gather(logits, dim=-1, index=student_top_k_ids) - teacher_logsumexp

    matches = student_top_k_ids.unsqueeze(-1) == teacher_top_k_ids.unsqueeze(-2)
    overlap_mask = matches.any(dim=-1).to(dtype=logits.dtype)
    teacher_in_student_mask = matches.any(dim=-2).to(dtype=logits.dtype)

    return {
        "teacher_top_k_ids": teacher_top_k_ids,
        "teacher_top_k_log_probs": teacher_top_k_log_probs,
        "teacher_on_student_log_probs": teacher_on_student_log_probs,
        "overlap_mask": overlap_mask,
        "teacher_in_student_mask": teacher_in_student_mask,
    }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    denom = mask.float().sum().clamp(min=1.0)
    return ((values.float() * mask.float()).sum() / denom).detach().item()


def _aggregate_one_window(
    tensors: dict[str, torch.Tensor],
    response_mask: torch.Tensor,
    *,
    top_k: int,
    chunk_start: int,
    chunk_end: int,
) -> dict[str, float | int]:
    if chunk_start >= 0:
        sl = slice(chunk_start, chunk_end)
        mask = response_mask[:, sl]
        student_log_probs = tensors["student_top_k_log_probs"][:, sl, :]
        teacher_on_student = tensors["teacher_on_student_log_probs"][:, sl, :]
        overlap_mask = tensors["overlap_mask"][:, sl, :]
        student_entropy = tensors.get("student_entropys")
        teacher_entropy = tensors.get("ref_entropys")
        if student_entropy is not None:
            student_entropy = student_entropy[:, sl]
        if teacher_entropy is not None:
            teacher_entropy = teacher_entropy[:, sl]
    else:
        mask = response_mask
        student_log_probs = tensors["student_top_k_log_probs"]
        teacher_on_student = tensors["teacher_on_student_log_probs"]
        overlap_mask = tensors["overlap_mask"]
        student_entropy = tensors.get("student_entropys")
        teacher_entropy = tensors.get("ref_entropys")

    token_mask = mask.bool()
    valid_token_count = int(token_mask.sum().item())
    valid_sequence_count = int(token_mask.any(dim=-1).sum().item())
    if valid_token_count == 0:
        return {
            "chunk_start": chunk_start,
            "chunk_end": chunk_end,
            "valid_sequence_count": 0,
            "valid_token_count": 0,
            "topk_overlap_ratio": float("nan"),
            "student_overlap_mass": float("nan"),
            "teacher_overlap_mass": float("nan"),
            "student_entropy": float("nan"),
            "teacher_entropy": float("nan"),
            "entropy_gap": float("nan"),
            "overlap_token_advantage": float("nan"),
        }

    expanded_token_mask = token_mask.unsqueeze(-1).expand_as(overlap_mask)
    valid_k_mask = expanded_token_mask.float()
    overlap_bool = (overlap_mask > 0.5) & expanded_token_mask
    overlap_float = overlap_bool.float()

    overlap_ratio = (overlap_float.sum() / (valid_k_mask.sum().clamp(min=1.0))).detach().item()

    student_probs = torch.exp(student_log_probs.float())
    teacher_probs = torch.exp(teacher_on_student.float())
    student_mass_per_token = (student_probs * overlap_float).sum(dim=-1)
    teacher_mass_per_token = (teacher_probs * overlap_float).sum(dim=-1)

    student_overlap_mass = _masked_mean(student_mass_per_token, token_mask)
    teacher_overlap_mass = _masked_mean(teacher_mass_per_token, token_mask)

    if student_entropy is None:
        student_entropy_mean = float("nan")
    else:
        student_entropy_mean = _masked_mean(student_entropy, token_mask)
    if teacher_entropy is None:
        teacher_entropy_mean = float("nan")
        entropy_gap = float("nan")
    else:
        teacher_entropy_mean = _masked_mean(teacher_entropy, token_mask)
        if student_entropy is None:
            entropy_gap = float("nan")
        else:
            entropy_gap = _masked_mean((teacher_entropy.float() - student_entropy.float()).abs(), token_mask)

    # Rethinking-style overlap-token advantage over the overlap set.
    # Normalize student/teacher probabilities inside the overlap set per token.
    student_overlap_sum = (student_probs * overlap_float).sum(dim=-1, keepdim=True)
    teacher_overlap_sum = (teacher_probs * overlap_float).sum(dim=-1, keepdim=True)
    has_overlap = (student_overlap_sum.squeeze(-1) > 0) & token_mask
    normalized_student = torch.where(
        overlap_bool,
        student_probs / student_overlap_sum.clamp(min=1e-12),
        torch.zeros_like(student_probs),
    )
    normalized_teacher = torch.where(
        overlap_bool,
        teacher_probs / teacher_overlap_sum.clamp(min=1e-12),
        torch.ones_like(teacher_probs),
    )
    token_adv = (normalized_student * (torch.log(normalized_teacher.clamp(min=1e-12)) - torch.log(normalized_student.clamp(min=1e-12)))).sum(dim=-1)
    if int(has_overlap.sum().item()) == 0:
        overlap_token_advantage = float("nan")
    else:
        overlap_token_advantage = _masked_mean(token_adv, has_overlap)

    return {
        "chunk_start": chunk_start,
        "chunk_end": chunk_end,
        "valid_sequence_count": valid_sequence_count,
        "valid_token_count": valid_token_count,
        "topk_overlap_ratio": overlap_ratio,
        "student_overlap_mass": student_overlap_mass,
        "teacher_overlap_mass": teacher_overlap_mass,
        "student_entropy": student_entropy_mean,
        "teacher_entropy": teacher_entropy_mean,
        "entropy_gap": entropy_gap,
        "overlap_token_advantage": overlap_token_advantage,
    }


def aggregate_rethinking_opd_probe_metrics(
    tensors: dict[str, torch.Tensor],
    response_mask: torch.Tensor,
    *,
    top_k: int,
    chunk_size: int,
) -> list[dict[str, float | int]]:
    """Aggregate global and depth-chunk metrics from current-batch probe tensors."""

    missing = [key for key in RETHINKING_OPD_PROBE_REQUIRED_TENSORS if key not in tensors]
    if missing:
        raise ValueError(f"Missing Rethinking OPD probe tensors: {missing}")
    if response_mask.dim() != 2:
        raise ValueError("response_mask must have shape (batch, response_length)")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    rows = [_aggregate_one_window(tensors, response_mask, top_k=top_k, chunk_start=-1, chunk_end=-1)]
    response_length = response_mask.shape[-1]
    for start in range(0, response_length, chunk_size):
        end = min(start + chunk_size, response_length)
        row = _aggregate_one_window(tensors, response_mask, top_k=top_k, chunk_start=start, chunk_end=end)
        rows.append(row)
    return rows


def decorate_rethinking_probe_rows(
    rows: list[dict[str, Any]],
    *,
    run_name: str,
    step: int,
    top_k: int,
    response_mask: torch.Tensor,
) -> list[dict[str, Any]]:
    response_lengths = response_mask.float().sum(dim=-1)
    max_response_width = float(response_mask.shape[-1])
    context = {
        "run_name": run_name,
        "step": int(step),
        "top_k": int(top_k),
        "batch_size": int(response_mask.shape[0]),
        "mean_all_batch_response_length": response_lengths.mean().detach().item(),
        "max_all_batch_response_length": response_lengths.max().detach().item(),
        "clip_rate_all_batch": (response_lengths == max_response_width).float().mean().detach().item(),
    }
    return [{**context, **row} for row in rows]



def append_rethinking_opd_probe_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Append probe rows to a CSV, writing the header once."""

    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists() and csv_path.stat().st_size > 0
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
