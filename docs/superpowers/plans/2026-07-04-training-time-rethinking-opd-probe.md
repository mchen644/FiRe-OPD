# Training-Time Rethinking OPD Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a training-time current-batch probe that logs Rethinking OPD-style depth-binned top-k overlap, entropy, overlap mass, and overlap-token advantage for raw OPD, budget20 hardtrunc, and concise20 hardtrunc runs.

**Architecture:** The probe follows `thunlp/OPD`: actor log-prob computation emits student top-k ids/log-probs when the probe is enabled; teacher/ref scoring on the same current training batch emits teacher top-k ids/log-probs, teacher-on-student log-probs, entropy, and overlap masks when the probe is enabled; `ray_trainer.py` aggregates global and 1024-token chunk metrics before TALE hard truncation. Pure tensor aggregation lives in a focused helper module and writes scalar metrics plus a CSV sidecar.

**Tech Stack:** Python 3.10, PyTorch, veRL `DataProto`, Hydra/OmegaConf config, pytest.

## Global Constraints

- Do not add extra student rollouts for probing.
- Do not add fixed probe-prompt evaluation.
- Initial runs only: `raw_opd`, `budget20`, `concise20`; do not add `normal20` scripts in this implementation.
- For `budget20` and `concise20`, compute probe metrics after rollout generation and after TALE teacher prompt rewriting, but before `truncate_to_tale_budget_esr` physically truncates tensors.
- Do not use `rollout_corr/chi2_seq` as a primary metric.
- Keep `top_k=16` and `position_chunk_size=1024` defaults.
- Do not store logits by default.
- Preserve existing training behavior when `algorithm.rethinking_opd_probe.enabled=False`.

---

## File Structure

Create:

- `verl/verl/trainer/ppo/rethinking_opd_probe.py`
  - Pure tensor helpers for top-k overlap, overlap mass, entropy gap, chunk aggregation, and CSV appending.
- `verl/tests/trainer/ppo/test_rethinking_opd_probe.py`
  - Unit tests for pure metric helpers and CSV formatting.
- `math_eval/plot_rethinking_opd_probe.py`
  - Lightweight analysis script for heatmaps/curves from the CSV sidecar.
- `run_train_raw_opd_rethinking_probe.sh`
  - Raw OPD probe run script.
- `run_train_tale_budget_rolloutlen_hardtrunc_rethinking_probe_opd.sh`
  - Budget20 hardtrunc probe run script.
- `run_train_tale_budget_rolloutlen_hardtrunc_conciseteacher_rethinking_probe_opd.sh`
  - Concise20 hardtrunc probe run script.

Modify:

- `verl/verl/trainer/config/algorithm.py`
  - Add `RethinkingOpdProbeConfig` and `AlgoConfig.rethinking_opd_probe`.
- `verl/verl/trainer/config/ppo_trainer.yaml`
  - Add YAML defaults for `algorithm.rethinking_opd_probe`.
- `verl/tests/trainer/config/test_algo_config_on_cpu.py`
  - Add config acceptance test for the new keys.
- `verl/verl/workers/actor/dp_actor.py`
  - Add probe-gated top-k/teacher-overlap tensor emission in `DataParallelPPOActor.compute_log_prob`.
- `verl/verl/workers/fsdp_workers.py`
  - Pass probe top-k meta info into actor/ref workers and return probe tensors from `compute_log_prob` / `compute_ref_log_prob`.
- `verl/verl/trainer/ppo/ray_trainer.py`
  - Enable actor/ref probe paths, aggregate/log metrics, write CSV, and reorder the TALE hardtrunc reference path only when the probe is enabled.
- `verl/tests/trainer/ppo/test_tale_budget.py`
  - Add regression test for “probe before hard truncation” using the pure aggregation helper and a full-length response mask.

---

### Task 1: Pure Rethinking OPD Metric Helpers

**Files:**
- Create: `verl/verl/trainer/ppo/rethinking_opd_probe.py`
- Create: `verl/tests/trainer/ppo/test_rethinking_opd_probe.py`

**Interfaces:**
- Produces:
  - `compute_teacher_topk_overlap(logits: torch.Tensor, student_top_k_ids: torch.Tensor, top_k: int) -> dict[str, torch.Tensor]`
  - `aggregate_rethinking_opd_probe_metrics(tensors: dict[str, torch.Tensor], response_mask: torch.Tensor, *, top_k: int, chunk_size: int) -> list[dict[str, float | int]]`
  - `append_rethinking_opd_probe_csv(path: str | pathlib.Path, rows: list[dict[str, float | int | str]]) -> None`
- Consumes: plain tensors only. Later tasks call these helpers from worker/trainer code.

- [ ] **Step 1: Write failing tests for overlap, mass, advantage, and chunks**

Add `verl/tests/trainer/ppo/test_rethinking_opd_probe.py`:

```python
from pathlib import Path

import pytest
import torch

from verl.trainer.ppo.rethinking_opd_probe import (
    aggregate_rethinking_opd_probe_metrics,
    append_rethinking_opd_probe_csv,
    compute_teacher_topk_overlap,
)


def test_compute_teacher_topk_overlap_marks_student_ids_in_teacher_topk():
    logits = torch.tensor(
        [
            [5.0, 4.0, 1.0, 0.0],
            [0.0, 1.0, 4.0, 5.0],
        ]
    )
    student_ids = torch.tensor([[0, 2], [3, 1]])

    out = compute_teacher_topk_overlap(logits, student_ids, top_k=2)

    assert out["teacher_top_k_ids"].tolist() == [[0, 1], [3, 2]]
    assert out["overlap_mask"].tolist() == [[1.0, 0.0], [1.0, 0.0]]
    assert out["teacher_in_student_mask"].tolist() == [[1.0, 0.0], [1.0, 0.0]]
    assert out["teacher_on_student_log_probs"].shape == (2, 2)
    assert out["teacher_top_k_log_probs"].shape == (2, 2)


def test_aggregate_rethinking_opd_probe_metrics_global_and_chunks():
    student_log_probs = torch.log(torch.tensor([[[0.50, 0.25], [0.60, 0.20], [0.70, 0.10], [0.80, 0.05]]]))
    teacher_on_student = torch.log(torch.tensor([[[0.40, 0.30], [0.50, 0.10], [0.60, 0.20], [0.70, 0.10]]]))
    overlap_mask = torch.tensor([[[1.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]])
    response_mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    student_entropy = torch.tensor([[0.10, 0.20, 0.30, 0.40]])
    teacher_entropy = torch.tensor([[0.20, 0.25, 0.35, 0.50]])

    rows = aggregate_rethinking_opd_probe_metrics(
        {
            "student_top_k_log_probs": student_log_probs,
            "teacher_on_student_log_probs": teacher_on_student,
            "overlap_mask": overlap_mask,
            "student_entropys": student_entropy,
            "ref_entropys": teacher_entropy,
        },
        response_mask,
        top_k=2,
        chunk_size=2,
    )

    global_row = next(row for row in rows if row["chunk_start"] == -1)
    chunk0 = next(row for row in rows if row["chunk_start"] == 0)
    chunk1 = next(row for row in rows if row["chunk_start"] == 2)

    assert global_row["valid_token_count"] == 3
    assert global_row["valid_sequence_count"] == 1
    assert global_row["topk_overlap_ratio"] == pytest.approx(4 / 6)
    assert global_row["student_overlap_mass"] == pytest.approx((0.75 + 0.60 + 0.80) / 3)
    assert global_row["teacher_overlap_mass"] == pytest.approx((0.70 + 0.50 + 0.80) / 3)
    assert global_row["entropy_gap"] == pytest.approx(((0.10) + (0.05) + (0.05)) / 3)
    assert "overlap_token_advantage" in global_row

    assert chunk0["valid_token_count"] == 2
    assert chunk0["topk_overlap_ratio"] == pytest.approx(3 / 4)
    assert chunk1["valid_token_count"] == 1
    assert chunk1["topk_overlap_ratio"] == pytest.approx(1.0)


def test_append_rethinking_opd_probe_csv_writes_header_once(tmp_path: Path):
    path = tmp_path / "probe.csv"
    rows = [
        {
            "run_name": "unit",
            "step": 1,
            "chunk_start": -1,
            "chunk_end": -1,
            "top_k": 2,
            "batch_size": 1,
            "valid_sequence_count": 1,
            "valid_token_count": 3,
            "mean_all_batch_response_length": 3.0,
            "max_all_batch_response_length": 4.0,
            "clip_rate_all_batch": 0.0,
            "topk_overlap_ratio": 0.5,
            "student_overlap_mass": 0.7,
            "teacher_overlap_mass": 0.6,
            "student_entropy": 0.2,
            "teacher_entropy": 0.3,
            "entropy_gap": 0.1,
            "overlap_token_advantage": -0.01,
        }
    ]

    append_rethinking_opd_probe_csv(path, rows)
    append_rethinking_opd_probe_csv(path, rows)

    lines = path.read_text().strip().splitlines()
    assert lines[0].startswith("run_name,step,chunk_start")
    assert len(lines) == 3
```

- [ ] **Step 2: Run tests to verify they fail because the module is missing**

Run:

```bash
cd /home/mchen/FiRe-OPD
/home/mchen/miniconda3/envs/verl/bin/python -m pytest verl/tests/trainer/ppo/test_rethinking_opd_probe.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'verl.trainer.ppo.rethinking_opd_probe'`.

- [ ] **Step 3: Implement pure helper module**

Create `verl/verl/trainer/ppo/rethinking_opd_probe.py`:

```python
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


def compute_teacher_topk_overlap(logits: torch.Tensor, student_top_k_ids: torch.Tensor, top_k: int) -> dict[str, torch.Tensor]:
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

    required = ["student_top_k_log_probs", "teacher_on_student_log_probs", "overlap_mask"]
    missing = [key for key in required if key not in tensors]
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
```

- [ ] **Step 4: Run tests and commit**

Run:

```bash
cd /home/mchen/FiRe-OPD
/home/mchen/miniconda3/envs/verl/bin/python -m pytest verl/tests/trainer/ppo/test_rethinking_opd_probe.py -q
```

Expected: PASS, `3 passed`.

Commit:

```bash
git add verl/verl/trainer/ppo/rethinking_opd_probe.py verl/tests/trainer/ppo/test_rethinking_opd_probe.py
git commit -m "Add Rethinking OPD probe metric helpers"
```

---

### Task 2: Config Surface for Training-Time Probe

**Files:**
- Modify: `verl/verl/trainer/config/algorithm.py`
- Modify: `verl/verl/trainer/config/ppo_trainer.yaml`
- Modify: `verl/tests/trainer/config/test_algo_config_on_cpu.py`

**Interfaces:**
- Consumes: no new runtime tensors.
- Produces:
  - `RethinkingOpdProbeConfig`
  - `AlgoConfig.rethinking_opd_probe`
  - Hydra keys under `algorithm.rethinking_opd_probe.*`

- [ ] **Step 1: Write config acceptance test**

Add to `verl/tests/trainer/config/test_algo_config_on_cpu.py`:

```python
def test_yaml_accepts_rethinking_opd_probe_overrides(self):
    config = self._load_config(
        overrides=[
            "algorithm.rethinking_opd_probe.enabled=True",
            "algorithm.rethinking_opd_probe.top_k=16",
            "algorithm.rethinking_opd_probe.chunk_size=1024",
            "algorithm.rethinking_opd_probe.csv_path=/tmp/rethinking_probe.csv",
            "algorithm.rethinking_opd_probe.log_prefix=rethinking_opd",
        ]
    )

    probe = config.algorithm.rethinking_opd_probe
    assert probe.enabled is True
    assert probe.top_k == 16
    assert probe.chunk_size == 1024
    assert probe.csv_path == "/tmp/rethinking_probe.csv"
    assert probe.log_prefix == "rethinking_opd"
```

If this test class does not expose `_load_config`, follow the existing helper style in the same file and keep the assertions identical.

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /home/mchen/FiRe-OPD/verl
/home/mchen/miniconda3/envs/verl/bin/python -m pytest tests/trainer/config/test_algo_config_on_cpu.py::TestAlgoConfig::test_yaml_accepts_rethinking_opd_probe_overrides -q
```

Expected: FAIL with an OmegaConf/Hydra unknown key error for `algorithm.rethinking_opd_probe`.

- [ ] **Step 3: Add dataclass config**

Modify `verl/verl/trainer/config/algorithm.py`:

```python
__all__ = [
    "AlgoConfig",
    "CandidateSelectionConfig",
    "FilterGroupsConfig",
    "KLControlConfig",
    "RethinkingOpdProbeConfig",
    "RolloutCorrectionConfig",
    "TaleBudgetConfig",
]
```

Add after `TaleBudgetConfig`:

```python
@dataclass
class RethinkingOpdProbeConfig(BaseConfig):
    """Training-time current-batch Rethinking OPD diagnostics."""

    enabled: bool = False
    top_k: int = 16
    chunk_size: int = 1024
    csv_path: Optional[str] = None
    log_prefix: str = "rethinking_opd"
    include_scalar_logger: bool = True
```

Add to `AlgoConfig` fields:

```python
    rethinking_opd_probe: RethinkingOpdProbeConfig = field(default_factory=RethinkingOpdProbeConfig)
```

- [ ] **Step 4: Add YAML defaults**

In `verl/verl/trainer/config/ppo_trainer.yaml`, under `algorithm:` after `tale_budget:` block, add:

```yaml
  # Training-time current-batch Rethinking OPD diagnostics
  rethinking_opd_probe:

    # Required when using verl.utils.omega_conf_to_dataclass to instantiate dataclass configs
    _target_: verl.trainer.config.RethinkingOpdProbeConfig

    # Whether to enable current-batch Rethinking OPD diagnostics
    enabled: False

    # Number of top-k tokens used for overlap diagnostics
    top_k: 16

    # Response-token chunk size for depth-binned metrics
    chunk_size: 1024

    # Optional CSV sidecar path; null disables sidecar writing
    csv_path: null

    # Metric prefix for scalar logger keys
    log_prefix: rethinking_opd

    # Whether to add metrics to the normal training logger
    include_scalar_logger: True
```

- [ ] **Step 5: Run config tests and commit**

Run:

```bash
cd /home/mchen/FiRe-OPD/verl
/home/mchen/miniconda3/envs/verl/bin/python -m pytest tests/trainer/config/test_algo_config_on_cpu.py::TestAlgoConfig::test_yaml_accepts_rethinking_opd_probe_overrides -q
```

Expected: PASS.

Commit:

```bash
git add verl/trainer/config/algorithm.py verl/trainer/config/ppo_trainer.yaml tests/trainer/config/test_algo_config_on_cpu.py
git commit -m "Add Rethinking OPD probe config"
```

---

### Task 3: Student Top-K Tensor Emission from Actor Log-Prob

**Files:**
- Modify: `verl/verl/workers/actor/dp_actor.py`
- Modify: `verl/verl/workers/fsdp_workers.py`
- Test: `verl/tests/trainer/ppo/test_rethinking_opd_probe.py`

**Interfaces:**
- Consumes: `data.meta_info["rethinking_opd_probe_top_k"]` set by trainer.
- Produces, when `top_k > 0`:
  - `student_top_k_ids: torch.LongTensor[B, T, K]`
  - `student_top_k_log_probs: torch.Tensor[B, T, K]`
  - `student_entropys: torch.Tensor[B, T]` alias from actor entropy for probe aggregation.

- [ ] **Step 1: Add a pure top-k extraction test**

Append to `verl/tests/trainer/ppo/test_rethinking_opd_probe.py`:

```python
from verl.trainer.ppo.rethinking_opd_probe import compute_student_topk_from_logits


def test_compute_student_topk_from_logits_returns_log_probs():
    logits = torch.tensor([[[5.0, 4.0, 1.0], [0.0, 3.0, 2.0]]])

    ids, log_probs = compute_student_topk_from_logits(logits, top_k=2)

    assert ids.tolist() == [[[0, 1], [1, 2]]]
    expected = torch.log_softmax(logits, dim=-1).gather(-1, ids)
    assert torch.allclose(log_probs, expected)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /home/mchen/FiRe-OPD
/home/mchen/miniconda3/envs/verl/bin/python -m pytest verl/tests/trainer/ppo/test_rethinking_opd_probe.py::test_compute_student_topk_from_logits_returns_log_probs -q
```

Expected: FAIL with `ImportError` for `compute_student_topk_from_logits`.

- [ ] **Step 3: Implement top-k helper**

Add to `verl/verl/trainer/ppo/rethinking_opd_probe.py`:

```python
def compute_student_topk_from_logits(logits: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return top-k ids and log-probs from logits with shape `(..., vocab_size)`."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    top_k_logits, top_k_ids = torch.topk(logits, k=top_k, dim=-1)
    logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
    return top_k_ids, top_k_logits - logsumexp
```

- [ ] **Step 4: Extend `DataParallelPPOActor._forward_micro_batch` signature**

In `verl/verl/workers/actor/dp_actor.py`, change:

```python
def _forward_micro_batch(
    self, micro_batch, temperature, calculate_entropy=False
) -> tuple[torch.Tensor, torch.Tensor]:
```

To:

```python
def _forward_micro_batch(
    self,
    micro_batch,
    temperature,
    calculate_entropy=False,
    top_k: int = 0,
    student_top_k_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor, dict[str, torch.Tensor]]:
```

At the start of the method add:

```python
        probe_tensors: dict[str, torch.Tensor] = {}
        need_logits_for_probe = top_k > 0 or student_top_k_ids is not None
```

In both non-fused logits branches, after slicing response logits to `(bsz, response_length, vocab_size)`, add:

```python
                    if top_k > 0 and student_top_k_ids is None:
                        from verl.trainer.ppo.rethinking_opd_probe import compute_student_topk_from_logits

                        topk_ids, topk_log_probs = compute_student_topk_from_logits(logits, top_k=top_k)
                        probe_tensors["student_top_k_ids"] = topk_ids
                        probe_tensors["student_top_k_log_probs"] = topk_log_probs
```

For remove-padding mode, follow the same pattern on the padded response logits after `full_log_probs` is available. If logits are only available in packed form, use the `thunlp/OPD` pattern: compute top-k on `logits_rmpad`, pad `topk_ids` and `topk_log_probs` back to `(batch, seqlen, top_k)`, then slice `[:, -response_length - 1 : -1, :]`.

Change the return statement from:

```python
            return entropy, log_probs
```

To:

```python
            return entropy, log_probs, probe_tensors
```

- [ ] **Step 5: Preserve backward compatibility in `compute_log_prob`**

In `DataParallelPPOActor.compute_log_prob`, read top-k:

```python
        top_k = int(data.meta_info.get("rethinking_opd_probe_top_k", 0) or 0)
```

Update the microbatch call:

```python
                entropy, log_probs, probe_tensors = self._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    top_k=top_k,
                )
```

Collect tensors:

```python
        student_top_k_ids_lst = []
        student_top_k_log_probs_lst = []
```

Inside the loop:

```python
            if top_k > 0:
                student_top_k_ids_lst.append(probe_tensors["student_top_k_ids"])
                student_top_k_log_probs_lst.append(probe_tensors["student_top_k_log_probs"])
```

After dynamic restore:

```python
        extra_tensors = {}
        if top_k > 0:
            student_top_k_ids = torch.concat(student_top_k_ids_lst, dim=0)
            student_top_k_log_probs = torch.concat(student_top_k_log_probs_lst, dim=0)
            if use_dynamic_bsz:
                student_top_k_ids = restore_dynamic_batch(student_top_k_ids, batch_idx_list)
                student_top_k_log_probs = restore_dynamic_batch(student_top_k_log_probs, batch_idx_list)
            extra_tensors["student_top_k_ids"] = student_top_k_ids
            extra_tensors["student_top_k_log_probs"] = student_top_k_log_probs
```

Change return to:

```python
        return log_probs, entropys, extra_tensors
```

- [ ] **Step 6: Update FSDP worker actor wrapper**

In `verl/verl/workers/fsdp_workers.py::compute_log_prob`, set meta info:

```python
        data.meta_info["rethinking_opd_probe_top_k"] = data.meta_info.get("rethinking_opd_probe_top_k", 0)
```

Change call:

```python
                output, entropys, probe_tensors = self.actor.compute_log_prob(data=data, calculate_entropy=True)
```

Build tensors:

```python
            tensors = {"old_log_probs": output, "entropys": entropys}
            if probe_tensors:
                tensors.update(probe_tensors)
                if entropys is not None:
                    tensors["student_entropys"] = entropys
```

Return `DataProto.from_dict(tensors=tensors, ...)`.

- [ ] **Step 7: Run focused tests and commit**

Run:

```bash
cd /home/mchen/FiRe-OPD
/home/mchen/miniconda3/envs/verl/bin/python -m pytest verl/tests/trainer/ppo/test_rethinking_opd_probe.py -q
```

Expected: PASS.

Commit:

```bash
git add verl/verl/trainer/ppo/rethinking_opd_probe.py verl/tests/trainer/ppo/test_rethinking_opd_probe.py verl/verl/workers/actor/dp_actor.py verl/verl/workers/fsdp_workers.py
git commit -m "Emit student top-k tensors for Rethinking OPD probe"
```

---

### Task 4: Teacher Top-K / Overlap Tensor Emission from Ref Scoring

**Files:**
- Modify: `verl/verl/workers/actor/dp_actor.py`
- Modify: `verl/verl/workers/fsdp_workers.py`
- Test: `verl/tests/trainer/ppo/test_rethinking_opd_probe.py`

**Interfaces:**
- Consumes:
  - `student_top_k_ids: torch.LongTensor[B, T, K]`
  - `data.meta_info["rethinking_opd_probe_top_k"]`
  - existing ref retokenization tensors when teacher prompt differs.
- Produces:
  - `teacher_top_k_ids`
  - `teacher_top_k_log_probs`
  - `teacher_on_student_log_probs`
  - `overlap_mask`
  - `teacher_in_student_mask`
  - `ref_entropys`

- [ ] **Step 1: Add shape validation tests for teacher overlap helper**

Append to `verl/tests/trainer/ppo/test_rethinking_opd_probe.py`:

```python
def test_compute_teacher_topk_overlap_rejects_shape_mismatch():
    logits = torch.zeros(2, 5)
    student_ids = torch.zeros(2, 3, dtype=torch.long)

    with pytest.raises(ValueError, match="last dimension must equal top_k"):
        compute_teacher_topk_overlap(logits, student_ids, top_k=2)
```

Run:

```bash
cd /home/mchen/FiRe-OPD
/home/mchen/miniconda3/envs/verl/bin/python -m pytest verl/tests/trainer/ppo/test_rethinking_opd_probe.py::test_compute_teacher_topk_overlap_rejects_shape_mismatch -q
```

Expected: PASS if Task 1 validation is present.

- [ ] **Step 2: Add teacher-top-k path to `DataParallelPPOActor._forward_micro_batch`**

In both logits branches of `DataParallelPPOActor._forward_micro_batch`, after response logits are available, add:

```python
                    if student_top_k_ids is not None:
                        from verl.trainer.ppo.rethinking_opd_probe import compute_teacher_topk_overlap

                        teacher_overlap = compute_teacher_topk_overlap(
                            logits.reshape(-1, logits.shape[-1]),
                            student_top_k_ids.reshape(-1, student_top_k_ids.shape[-1]),
                            top_k=top_k,
                        )
                        bsz, resp_len, _ = student_top_k_ids.shape
                        for key, value in teacher_overlap.items():
                            if value.dim() == 2:
                                probe_tensors[key] = value.view(bsz, resp_len, value.shape[-1])
                            else:
                                probe_tensors[key] = value
```

For remove-padding mode, use the same official `thunlp/OPD` approach: pad `student_top_k_ids` to full sequence length, unpad to packed positions, call `compute_teacher_topk_overlap` on packed logits and ids, gather/pad back, then slice the response region.

- [ ] **Step 3: Pass `student_top_k_ids` through `compute_log_prob` when present**

In `DataParallelPPOActor.compute_log_prob`, include `student_top_k_ids` in selected batch keys if present:

```python
        has_student_top_k_ids = "student_top_k_ids" in data.batch.keys()
        if has_student_top_k_ids:
            select_keys.append("student_top_k_ids")
```

Inside the microbatch loop:

```python
                mb_student_top_k_ids = model_inputs.get("student_top_k_ids")
                entropy, log_probs, probe_tensors = self._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    top_k=top_k,
                    student_top_k_ids=mb_student_top_k_ids,
                )
```

Collect extra teacher tensors generically:

```python
        probe_tensor_lists: dict[str, list[torch.Tensor]] = {}
```

Inside the loop:

```python
            for key, value in probe_tensors.items():
                probe_tensor_lists.setdefault(key, []).append(value)
```

After the loop:

```python
        extra_tensors = {}
        for key, values in probe_tensor_lists.items():
            tensor = torch.concat(values, dim=0)
            if use_dynamic_bsz:
                tensor = restore_dynamic_batch(tensor, batch_idx_list)
            extra_tensors[key] = tensor
```

- [ ] **Step 4: Update `compute_ref_log_prob` wrapper**

In `verl/verl/workers/fsdp_workers.py::compute_ref_log_prob`, preserve existing entropy-aware behavior but force entropy when the probe is enabled:

```python
        probe_top_k = int(data.meta_info.get("rethinking_opd_probe_top_k", 0) or 0)
        compute_teacher_entropy = compute_teacher_entropy or probe_top_k > 0
```

Change ref call:

```python
            output, ref_entropys, probe_tensors = self.ref_policy.compute_log_prob(
                data=data, calculate_entropy=compute_teacher_entropy
            )
```

Build tensors:

```python
            tensors = {"ref_log_prob": output}
            if compute_teacher_entropy and ref_entropys is not None:
                tensors["ref_entropys"] = ref_entropys
            if probe_top_k > 0:
                tensors.update(probe_tensors)
```

For LoRA ref path, call `self.compute_log_prob(data)` with probe meta info and map `old_log_probs` to `ref_log_prob` while preserving teacher probe tensors:

```python
            tensors = {"ref_log_prob": data.batch["old_log_probs"]}
            for key in ("teacher_top_k_ids", "teacher_top_k_log_probs", "teacher_on_student_log_probs", "overlap_mask", "teacher_in_student_mask"):
                if key in data.batch:
                    tensors[key] = data.batch[key]
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
cd /home/mchen/FiRe-OPD
/home/mchen/miniconda3/envs/verl/bin/python -m pytest verl/tests/trainer/ppo/test_rethinking_opd_probe.py -q
```

Expected: PASS.

Commit:

```bash
git add verl/verl/workers/actor/dp_actor.py verl/verl/workers/fsdp_workers.py verl/tests/trainer/ppo/test_rethinking_opd_probe.py
git commit -m "Emit teacher top-k overlap tensors for Rethinking OPD probe"
```

---

### Task 5: Ray Trainer Hook, Pre-Truncation Ordering, Scalar Metrics, and CSV Sidecar

**Files:**
- Modify: `verl/verl/trainer/ppo/ray_trainer.py`
- Modify: `verl/tests/trainer/ppo/test_tale_budget.py`
- Test: `verl/tests/trainer/ppo/test_rethinking_opd_probe.py`

**Interfaces:**
- Consumes: config `algorithm.rethinking_opd_probe` and tensors emitted by Tasks 3-4.
- Produces:
  - scalar metrics under `rethinking_opd/*`
  - CSV rows at `algorithm.rethinking_opd_probe.csv_path`
  - preserves existing behavior when disabled.

- [ ] **Step 1: Add aggregation row decoration test**

Append to `verl/tests/trainer/ppo/test_rethinking_opd_probe.py`:

```python
from verl.trainer.ppo.rethinking_opd_probe import decorate_rethinking_probe_rows


def test_decorate_rethinking_probe_rows_adds_context_fields():
    rows = [{"chunk_start": -1, "chunk_end": -1, "valid_token_count": 2, "topk_overlap_ratio": 0.5}]
    response_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])

    out = decorate_rethinking_probe_rows(
        rows,
        run_name="probe-run",
        step=7,
        top_k=16,
        response_mask=response_mask,
    )

    assert out[0]["run_name"] == "probe-run"
    assert out[0]["step"] == 7
    assert out[0]["top_k"] == 16
    assert out[0]["batch_size"] == 2
    assert out[0]["mean_all_batch_response_length"] == pytest.approx(1.5)
    assert out[0]["max_all_batch_response_length"] == pytest.approx(2.0)
    assert out[0]["clip_rate_all_batch"] == pytest.approx(0.0)
```

- [ ] **Step 2: Implement row decoration helper**

Add to `rethinking_opd_probe.py`:

```python
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
```

- [ ] **Step 3: Add trainer helper functions**

In `verl/verl/trainer/ppo/ray_trainer.py`, add near other local helper functions:

```python
def _rethinking_probe_enabled(config) -> bool:
    probe_config = config.algorithm.get("rethinking_opd_probe", None)
    return bool(probe_config and probe_config.get("enabled", False))


def _set_rethinking_probe_meta(batch: DataProto, config) -> None:
    probe_config = config.algorithm.get("rethinking_opd_probe", None)
    top_k = int(probe_config.get("top_k", 16)) if probe_config else 0
    batch.meta_info["rethinking_opd_probe_top_k"] = top_k if probe_config and probe_config.get("enabled", False) else 0
```

Add aggregation helper:

```python
def _log_rethinking_opd_probe_metrics(batch: DataProto, *, config, global_step: int, metrics: dict) -> None:
    probe_config = config.algorithm.get("rethinking_opd_probe", None)
    if not probe_config or not probe_config.get("enabled", False):
        return

    from verl.trainer.ppo.rethinking_opd_probe import (
        aggregate_rethinking_opd_probe_metrics,
        append_rethinking_opd_probe_csv,
        decorate_rethinking_probe_rows,
    )

    top_k = int(probe_config.get("top_k", 16))
    chunk_size = int(probe_config.get("chunk_size", 1024))
    log_prefix = str(probe_config.get("log_prefix", "rethinking_opd"))
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
    rows = decorate_rethinking_probe_rows(rows, run_name=run_name, step=int(global_step), top_k=top_k, response_mask=response_mask)

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
```

- [ ] **Step 4: Set probe meta before actor log-prob computation**

In the training loop before:

```python
old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
```

Add:

```python
_set_rethinking_probe_meta(batch, self.config)
```

This ensures `student_top_k_ids`, `student_top_k_log_probs`, and `student_entropys` are emitted from the full current response batch.

- [ ] **Step 5: Reorder TALE ref path only when probe is enabled**

Inside the `if self.use_reference_policy:` block, after `_apply_online_tale_budget_prompts(...)` and before `truncate_to_tale_budget_esr(batch)`, add a probe-enabled branch:

```python
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
```

Then guard the existing later `prepare_ref_model_inputs` / `compute_ref_log_prob` call so it is skipped if `probe_enabled` already computed `ref_log_prob` on the full batch:

```python
                            ref_log_prob_already_computed = probe_enabled and "ref_log_prob" in batch.batch.keys()
```

Use:

```python
                                if not ref_log_prob_already_computed:
                                    batch = prepare_ref_model_inputs(...)
                                    ... compute ref_log_prob as before ...
```

Then keep existing:

```python
                                if tale_budget_hard_truncated:
                                    drop_ref_retokenization_tensors(batch)
```

The intended sequence when probe is enabled is:

```text
full rollout batch
TALE prompt rewrite
prepare full ref inputs
compute full ref/top-k/entropy
log probe metrics
truncate tensors to ESR prefix
continue training with truncated ref_log_prob and response_mask
```

- [ ] **Step 6: Log raw OPD/non-TALE probe metrics after normal ref computation**

For runs without TALE, after `batch = batch.union(ref_log_prob)` add:

```python
                                _log_rethinking_opd_probe_metrics(
                                    batch,
                                    config=self.config,
                                    global_step=self.global_steps,
                                    metrics=metrics,
                                )
```

The helper is a no-op when disabled.

- [ ] **Step 7: Add a lightweight regression test for aggregation before truncation**

In `verl/tests/trainer/ppo/test_tale_budget.py`, add a pure ordering test that exercises the intended helper behavior without Ray:

```python
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
```

- [ ] **Step 8: Run focused tests and commit**

Run:

```bash
cd /home/mchen/FiRe-OPD
/home/mchen/miniconda3/envs/verl/bin/python -m pytest \
  verl/tests/trainer/ppo/test_rethinking_opd_probe.py \
  verl/tests/trainer/ppo/test_tale_budget.py::test_rethinking_probe_aggregates_full_mask_before_tale_truncation -q
```

Expected: PASS.

Commit:

```bash
git add verl/verl/trainer/ppo/ray_trainer.py verl/verl/trainer/ppo/rethinking_opd_probe.py verl/tests/trainer/ppo/test_rethinking_opd_probe.py verl/tests/trainer/ppo/test_tale_budget.py
git commit -m "Log current-batch Rethinking OPD probe metrics"
```

---

### Task 6: Probe Run Scripts and Analysis Plotter

**Files:**
- Create: `run_train_raw_opd_rethinking_probe.sh`
- Create: `run_train_tale_budget_rolloutlen_hardtrunc_rethinking_probe_opd.sh`
- Create: `run_train_tale_budget_rolloutlen_hardtrunc_conciseteacher_rethinking_probe_opd.sh`
- Create: `math_eval/plot_rethinking_opd_probe.py`

**Interfaces:**
- Consumes: CSV sidecars written by Task 5.
- Produces: training scripts and SVG heatmaps/curves.

- [ ] **Step 1: Create raw OPD probe run script**

Create `run_train_raw_opd_rethinking_probe.sh` based on the existing raw OPD script, with these overrides included in the Python/Hydra command:

```bash
algorithm.rethinking_opd_probe.enabled=True \
algorithm.rethinking_opd_probe.top_k=16 \
algorithm.rethinking_opd_probe.chunk_size=1024 \
algorithm.rethinking_opd_probe.csv_path=math_eval/opd_training_dynamics_audit/training_probe_raw_opd.csv \
trainer.experiment_name=opd-raw-rethinking-probe \
trainer.val_before_train=False \
trainer.test_freq=-1 \
trainer.save_freq=-1
```

Ensure the script does not reuse a checkpoint directory shared with active jobs.

- [ ] **Step 2: Create budget20 probe run script**

Create `run_train_tale_budget_rolloutlen_hardtrunc_rethinking_probe_opd.sh` from `run_train_tale_budget_rolloutlen_hardtrunc_opd.sh` or the current budget20 script, with these required overrides:

```bash
algorithm.tale_budget.enabled=True \
algorithm.tale_budget.source=rollout_length \
algorithm.tale_budget.rollout_length_alpha=1.0 \
algorithm.tale_budget.esr_beta=0.2 \
algorithm.tale_budget.rollout_length_max_budget=null \
algorithm.tale_budget.truncate_to_esr=True \
algorithm.tale_budget.use_budget_teacher_prompt=True \
algorithm.tale_budget.teacher_prompt_style=auto \
algorithm.rethinking_opd_probe.enabled=True \
algorithm.rethinking_opd_probe.top_k=16 \
algorithm.rethinking_opd_probe.chunk_size=1024 \
algorithm.rethinking_opd_probe.csv_path=math_eval/opd_training_dynamics_audit/training_probe_budget20.csv \
trainer.experiment_name=opd-budget20-rethinking-probe \
trainer.val_before_train=False \
trainer.test_freq=-1 \
trainer.save_freq=-1
```

- [ ] **Step 3: Create concise20 probe run script**

Create `run_train_tale_budget_rolloutlen_hardtrunc_conciseteacher_rethinking_probe_opd.sh` from the concise teacher script, with these required overrides:

```bash
algorithm.tale_budget.enabled=True \
algorithm.tale_budget.source=rollout_length \
algorithm.tale_budget.rollout_length_alpha=1.0 \
algorithm.tale_budget.esr_beta=0.2 \
algorithm.tale_budget.rollout_length_max_budget=null \
algorithm.tale_budget.truncate_to_esr=True \
algorithm.tale_budget.teacher_prompt_style=concise \
algorithm.rethinking_opd_probe.enabled=True \
algorithm.rethinking_opd_probe.top_k=16 \
algorithm.rethinking_opd_probe.chunk_size=1024 \
algorithm.rethinking_opd_probe.csv_path=math_eval/opd_training_dynamics_audit/training_probe_concise20.csv \
trainer.experiment_name=opd-concise20-rethinking-probe \
trainer.val_before_train=False \
trainer.test_freq=-1 \
trainer.save_freq=-1
```

- [ ] **Step 4: Create analysis plotter**

Create `math_eval/plot_rethinking_opd_probe.py`:

```python
#!/usr/bin/env python3
"""Plot training-time Rethinking OPD probe CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def load_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open() as f:
            for row in csv.DictReader(f):
                rows.append(row)
    return rows


def write_summary(rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["run_name"], row["chunk_start"])
        grouped.setdefault(key, []).append(row)
    with output.open("w") as f:
        f.write("# Rethinking OPD Training Probe Summary\n\n")
        for (run, chunk), group in sorted(grouped.items()):
            last = sorted(group, key=lambda r: int(float(r["step"])))[-1]
            f.write(
                f"- {run} chunk {chunk}: "
                f"last_step={last['step']}, "
                f"overlap={_float(last['topk_overlap_ratio']):.4f}, "
                f"student_entropy={_float(last['student_entropy']):.4f}, "
                f"entropy_gap={_float(last['entropy_gap']):.4f}, "
                f"valid_tokens={last['valid_token_count']}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--summary", type=Path, default=Path("math_eval/opd_training_dynamics_audit/training_probe_summary.md"))
    args = parser.parse_args()
    rows = load_rows(args.csv)
    write_summary(rows, args.summary)
    print(f"wrote {args.summary}")


if __name__ == "__main__":
    main()
```

This first version writes a text summary only. Heatmap SVGs can be added after the first probe CSVs exist and their row shape is verified.

- [ ] **Step 5: Validate scripts are syntactically safe**

Run:

```bash
cd /home/mchen/FiRe-OPD
bash -n run_train_raw_opd_rethinking_probe.sh
bash -n run_train_tale_budget_rolloutlen_hardtrunc_rethinking_probe_opd.sh
bash -n run_train_tale_budget_rolloutlen_hardtrunc_conciseteacher_rethinking_probe_opd.sh
/home/mchen/miniconda3/envs/verl/bin/python -m py_compile math_eval/plot_rethinking_opd_probe.py
```

Expected: all commands exit 0.

- [ ] **Step 6: Commit**

```bash
git add \
  run_train_raw_opd_rethinking_probe.sh \
  run_train_tale_budget_rolloutlen_hardtrunc_rethinking_probe_opd.sh \
  run_train_tale_budget_rolloutlen_hardtrunc_conciseteacher_rethinking_probe_opd.sh \
  math_eval/plot_rethinking_opd_probe.py
git commit -m "Add Rethinking OPD probe run scripts"
```

---

### Task 7: End-to-End Smoke and Verification

**Files:**
- Modify only if smoke test reveals defects in files touched by Tasks 1-6.

**Interfaces:**
- Consumes: all previous tasks.
- Produces: verified probe metrics and a clear runbook for full experiments.

- [ ] **Step 1: Run full focused test suite**

Run:

```bash
cd /home/mchen/FiRe-OPD
/home/mchen/miniconda3/envs/verl/bin/python -m pytest \
  verl/tests/trainer/ppo/test_rethinking_opd_probe.py \
  verl/tests/trainer/ppo/test_tale_budget.py \
  verl/tests/trainer/config/test_algo_config_on_cpu.py::TestAlgoConfig::test_yaml_accepts_rethinking_opd_probe_overrides -q
```

Expected: PASS.

- [ ] **Step 2: Run one short smoke job only if GPUs are available**

Before launching, check GPU/process state:

```bash
ps -u mchen -o pid,etime,cmd | grep -E 'main_ppo|ray::TaskRunner|vllm|sglang' | grep -v grep || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits | head -20
```

If GPU0 is occupied, avoid GPU0. If no safe GPU is available, skip this smoke and report that code-level verification passed but runtime smoke is pending.

Use a one-step reduced config based on the budget20 script with a unique checkpoint directory and:

```bash
trainer.total_training_steps=1 \
trainer.save_freq=-1 \
trainer.test_freq=-1 \
trainer.val_before_train=False \
algorithm.rethinking_opd_probe.enabled=True \
algorithm.rethinking_opd_probe.csv_path=math_eval/opd_training_dynamics_audit/training_probe_smoke.csv
```

Expected: job finishes one step and writes `training_probe_smoke.csv` with at least one global row and one chunk row.

- [ ] **Step 3: Verify CSV summary script**

Run:

```bash
cd /home/mchen/FiRe-OPD
/home/mchen/miniconda3/envs/verl/bin/python math_eval/plot_rethinking_opd_probe.py \
  math_eval/opd_training_dynamics_audit/training_probe_smoke.csv \
  --summary math_eval/opd_training_dynamics_audit/training_probe_smoke_summary.md
```

Expected: prints `wrote math_eval/opd_training_dynamics_audit/training_probe_smoke_summary.md`.

- [ ] **Step 4: Commit smoke fixes or verification note**

If code changed during smoke fixes, inspect the modified paths and add only files touched by this plan:

```bash
git status --short
git add \
  verl/verl/trainer/ppo/rethinking_opd_probe.py \
  verl/verl/trainer/config/algorithm.py \
  verl/verl/trainer/config/ppo_trainer.yaml \
  verl/verl/workers/actor/dp_actor.py \
  verl/verl/workers/fsdp_workers.py \
  verl/verl/trainer/ppo/ray_trainer.py \
  verl/tests/trainer/ppo/test_rethinking_opd_probe.py \
  verl/tests/trainer/ppo/test_tale_budget.py \
  verl/tests/trainer/config/test_algo_config_on_cpu.py \
  math_eval/plot_rethinking_opd_probe.py \
  run_train_raw_opd_rethinking_probe.sh \
  run_train_tale_budget_rolloutlen_hardtrunc_rethinking_probe_opd.sh \
  run_train_tale_budget_rolloutlen_hardtrunc_conciseteacher_rethinking_probe_opd.sh
git commit -m "Fix Rethinking OPD probe smoke issues"
```

If no code changed, do not create an empty commit. Record verification commands and outputs in the final response.

---

## Self-Review Checklist

- Spec coverage:
  - Current training batch only: Tasks 3-5.
  - No extra rollouts: Tasks 3-5 use existing `batch` tensors only.
  - `raw_opd`, `budget20`, `concise20` scripts only: Task 6.
  - Pre-TALE-truncation probe for hardtrunc: Task 5.
  - Rethinking metrics and chunking: Tasks 1 and 5.
  - CSV/scalar logging: Tasks 1 and 5.
  - No dense checkpoint requirement: Task 6 uses `save_freq=-1`; Task 7 smoke also uses `save_freq=-1`.
- Placeholder scan: no `TBD`, `TODO`, or unspecified implementation steps remain.
- Type consistency:
  - `rethinking_opd_probe_top_k` is the meta-info key used by trainer, FSDP worker, and actor worker.
  - Student tensors are named `student_top_k_ids`, `student_top_k_log_probs`, `student_entropys`.
  - Teacher tensors are named `teacher_top_k_ids`, `teacher_top_k_log_probs`, `teacher_on_student_log_probs`, `overlap_mask`, `teacher_in_student_mask`, `ref_entropys`.
  - CSV fields match `CSV_FIELDS` in Task 1.
