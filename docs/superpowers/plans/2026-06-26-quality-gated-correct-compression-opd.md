# Quality-gated Correct-compression OPD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a quality-gated multi-candidate OPD selection mode that compresses correct trajectories, uses teacher-accepted wrong trajectories only for OPD correction, and drops wrong teacher-rejected groups.

**Architecture:** Extend the existing `candidate_selection` helper with a second method, `quality_gated_correct_compression`, and add config fields for teacher rejection/drop behavior. Extend length-aware OPD with `length_penalty_gate=correct` so selected correct trajectories receive the same median-relative `λ=0.02` penalty, while wrong selected fallback trajectories do not. Add a new strong-to-weak training script and wrapper for the controlled experiment.

**Tech Stack:** Python, PyTorch, verl `DataProto`, OmegaConf/Hydra dataclass configs, bash training scripts, pytest.

---

## File Structure

- Modify `verl/tests/trainer/ppo/test_candidate_selection.py`
  - Add failing tests for `quality_gated_correct_compression` selection behavior and new metrics.
- Modify `verl/verl/trainer/ppo/candidate_selection.py`
  - Support both existing `shortest_correct_else_teacher` and new `quality_gated_correct_compression` methods.
  - Compute batch-relative teacher accept/reject masks.
  - Physically drop no-correct teacher-rejected groups.
- Modify `verl/tests/trainer/config/test_algorithm_config.py`
  - Verify new `CandidateSelectionConfig` defaults and overrides.
- Modify `verl/verl/trainer/config/algorithm.py`
  - Add `teacher_reject_percentile` and `drop_rejected_no_correct` fields.
- Modify `verl/verl/trainer/config/ppo_trainer.yaml`
  - Add YAML defaults for the new candidate-selection fields.
- Modify `verl/tests/workers/actor/test_length_aware_opd.py`
  - Add failing tests for `length_penalty_gate=correct`.
- Modify `verl/verl/workers/actor/dp_actor.py`
  - Add `correct` to valid length penalty gates and implement the gate.
  - Add a clearer `length_aware_opd/correct_penalty_ratio` metric.
- Create `verl/examples/fire_opd/run_opd_strong_to_weak_lengthaware_correctcompress_teacher30b.sh`
  - New experiment launcher with `rollout.n=2`, candidate-selection method `quality_gated_correct_compression`, and `length_penalty_gate=correct`.
- Create `run_train_lengthaware_correctcompress_opd.sh`
  - Top-level srun-aware wrapper for the new launcher.

---

### Task 1: Add failing tests for quality-gated candidate selection

**Files:**
- Modify: `verl/tests/trainer/ppo/test_candidate_selection.py`
- Test: `verl/tests/trainer/ppo/test_candidate_selection.py`

- [ ] **Step 1: Extend `_cfg` test helper with new defaults**

In `verl/tests/trainer/ppo/test_candidate_selection.py`, replace `_cfg` with:

```python
def _cfg(**overrides):
    values = {
        "enabled": True,
        "method": "shortest_correct_else_teacher",
        "correct_reward_threshold": 0.5,
        "keep_per_uid": 1,
        "teacher_reject_percentile": 20.0,
        "drop_rejected_no_correct": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)
```

- [ ] **Step 2: Add a quality-gated test batch helper**

Append this helper after `_batch_for_selection()`:

```python
def _batch_for_quality_gated_selection():
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],  # uid-a, correct but long
            [1, 1, 1, 0, 0, 0],  # uid-a, correct and short -> select
            [1, 1, 0, 0, 0, 0],  # uid-b, wrong, teacher accepted and short -> select
            [1, 1, 1, 1, 1, 0],  # uid-b, wrong, teacher accepted and longer
            [1, 1, 1, 0, 0, 0],  # uid-c, wrong, teacher rejected
            [1, 1, 1, 1, 0, 0],  # uid-c, wrong, teacher rejected -> group dropped
        ],
        dtype=torch.float32,
    )
    ref_log_prob = torch.tensor(
        [
            [-0.2, -0.2, -0.2, -0.2, -0.2, -0.2],
            [-0.5, -0.5, -0.5, 0.0, 0.0, 0.0],
            [-0.2, -0.2, 0.0, 0.0, 0.0, 0.0],
            [-0.1, -0.1, -0.1, -0.1, -0.1, 0.0],
            [-10.0, -10.0, -10.0, 0.0, 0.0, 0.0],
            [-9.0, -9.0, -9.0, -9.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    token_level_scores = torch.zeros_like(response_mask)
    token_level_scores[0, 5] = 1.0
    token_level_scores[1, 2] = 1.0
    input_ids = torch.arange(6 * 6, dtype=torch.long).view(6, 6)
    batch = DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "response_mask": response_mask,
            "ref_log_prob": ref_log_prob,
            "token_level_scores": token_level_scores,
        },
        non_tensors={
            "uid": np.array(["uid-a", "uid-a", "uid-b", "uid-b", "uid-c", "uid-c"], dtype=object),
            "source": np.array(["a0", "a1", "b0", "b1", "c0", "c1"], dtype=object),
        },
    )
    return batch
```

- [ ] **Step 3: Add failing behavior test for the new method**

Append this test:

```python
def test_quality_gated_correct_compression_selects_correct_compresses_wrong_fallback_and_drops_rejected():
    batch = _batch_for_quality_gated_selection()

    selected, metrics = select_short_correct_candidates(
        batch,
        _cfg(method="quality_gated_correct_compression", teacher_reject_percentile=20.0),
    )

    assert selected.batch["input_ids"].tolist() == [
        [6, 7, 8, 9, 10, 11],
        [12, 13, 14, 15, 16, 17],
    ]
    assert selected.non_tensor_batch["uid"].tolist() == ["uid-a", "uid-b"]
    assert selected.non_tensor_batch["source"].tolist() == ["a1", "b0"]

    assert metrics["candidate_selection/enabled"] == 1.0
    assert metrics["candidate_selection/groups"] == 3.0
    assert metrics["candidate_selection/selected_groups"] == 2.0
    assert metrics["candidate_selection/candidates"] == 6.0
    assert metrics["candidate_selection/keep_ratio"] == pytest.approx(2.0 / 6.0)
    assert metrics["candidate_selection/any_correct_ratio"] == pytest.approx(1.0 / 3.0)
    assert metrics["candidate_selection/no_correct_ratio"] == pytest.approx(2.0 / 3.0)
    assert metrics["candidate_selection/selected_correct_ratio"] == 0.5
    assert metrics["candidate_selection/fallback_teacher_ratio"] == pytest.approx(1.0 / 3.0)
    assert metrics["candidate_selection/no_correct_teacher_accept_ratio"] == pytest.approx(1.0 / 3.0)
    assert metrics["candidate_selection/dropped_uid_ratio"] == pytest.approx(1.0 / 3.0)
    assert metrics["candidate_selection/selected_response_len_mean"] == 2.5
    assert metrics["candidate_selection/selected_correct_len_mean"] == 3.0
    assert metrics["candidate_selection/selected_wrong_len_mean"] == 2.0
    assert metrics["candidate_selection/correct_candidate_len_mean"] == 4.5
    assert metrics["candidate_selection/wrong_candidate_len_mean"] == 3.5
    assert metrics["candidate_selection/teacher_accept_ratio"] == pytest.approx(4.0 / 6.0)
    assert metrics["candidate_selection/teacher_reject_threshold"] < -1.0
```

- [ ] **Step 4: Add failing test for `drop_rejected_no_correct=False` fallback**

Append this test:

```python
def test_quality_gated_correct_compression_can_keep_teacher_best_when_drop_disabled():
    batch = _batch_for_quality_gated_selection()

    selected, metrics = select_short_correct_candidates(
        batch,
        _cfg(
            method="quality_gated_correct_compression",
            teacher_reject_percentile=20.0,
            drop_rejected_no_correct=False,
        ),
    )

    assert selected.non_tensor_batch["uid"].tolist() == ["uid-a", "uid-b", "uid-c"]
    assert selected.non_tensor_batch["source"].tolist() == ["a1", "b0", "c1"]
    assert metrics["candidate_selection/groups"] == 3.0
    assert metrics["candidate_selection/selected_groups"] == 3.0
    assert metrics["candidate_selection/dropped_uid_ratio"] == 0.0
    assert metrics["candidate_selection/fallback_teacher_ratio"] == pytest.approx(2.0 / 3.0)
```

- [ ] **Step 5: Run tests to verify they fail for the expected reason**

Run:

```bash
cd /home/mchen/FiRe-OPD
PYTHONPATH=/home/mchen/FiRe-OPD/verl /home/mchen/miniconda3/envs/verl/bin/python -m pytest \
  verl/tests/trainer/ppo/test_candidate_selection.py -q
```

Expected: FAIL with `Unsupported candidate_selection.method: quality_gated_correct_compression`.

- [ ] **Step 6: Commit failing tests**

```bash
git add verl/tests/trainer/ppo/test_candidate_selection.py
git commit -m "test: add quality-gated OPD candidate selection tests"
```

---

### Task 2: Implement quality-gated candidate selection

**Files:**
- Modify: `verl/verl/trainer/ppo/candidate_selection.py`
- Test: `verl/tests/trainer/ppo/test_candidate_selection.py`

- [ ] **Step 1: Replace `candidate_selection.py` with quality-gated implementation**

Replace the entire content of `verl/verl/trainer/ppo/candidate_selection.py` with:

```python
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch

from verl import DataProto

_SUPPORTED_METHODS = {"shortest_correct_else_teacher", "quality_gated_correct_compression"}


def _config_get(config: Any, name: str, default: Any) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(name, default)
    return getattr(config, name, default)


def _require_batch_key(batch: DataProto, key: str) -> torch.Tensor:
    if batch.batch is None or key not in batch.batch.keys():
        raise ValueError(f"candidate selection requires batch['{key}']")
    return batch.batch[key]


def _candidate_selection_scores(batch: DataProto, correct_reward_threshold: float):
    response_mask = _require_batch_key(batch, "response_mask").float()
    ref_log_prob = _require_batch_key(batch, "ref_log_prob")
    token_level_scores = _require_batch_key(batch, "token_level_scores")

    response_len = response_mask.sum(dim=-1).clamp(min=1.0)
    seq_reward = (token_level_scores.float() * response_mask).sum(dim=-1)
    correct = seq_reward > float(correct_reward_threshold)
    normalized_teacher_logprob = (ref_log_prob.float() * response_mask).sum(dim=-1) / response_len
    return response_len, correct, normalized_teacher_logprob


def _teacher_accept_mask(normalized_teacher_logprob: torch.Tensor, teacher_reject_percentile: float):
    if teacher_reject_percentile < 0.0 or teacher_reject_percentile > 100.0:
        raise ValueError("candidate_selection.teacher_reject_percentile must be between 0.0 and 100.0")
    if teacher_reject_percentile == 0.0:
        threshold = torch.tensor(float("nan"), device=normalized_teacher_logprob.device)
        return torch.ones_like(normalized_teacher_logprob, dtype=torch.bool), threshold
    threshold = torch.quantile(normalized_teacher_logprob.float(), teacher_reject_percentile / 100.0)
    return normalized_teacher_logprob > threshold, threshold


def _select_shortest_with_teacher_tiebreak(
    candidate_indices: list[int],
    response_len: torch.Tensor,
    normalized_teacher_logprob: torch.Tensor,
) -> int:
    return max(
        candidate_indices,
        key=lambda idx: (-float(response_len[idx].item()), float(normalized_teacher_logprob[idx].item())),
    )


def _select_index_shortest_correct_else_teacher(
    candidate_indices: list[int],
    response_len: torch.Tensor,
    correct: torch.Tensor,
    normalized_teacher_logprob: torch.Tensor,
) -> tuple[int, bool, bool]:
    correct_indices = [idx for idx in candidate_indices if bool(correct[idx].item())]
    if correct_indices:
        return _select_shortest_with_teacher_tiebreak(correct_indices, response_len, normalized_teacher_logprob), False, False

    selected = max(candidate_indices, key=lambda idx: float(normalized_teacher_logprob[idx].item()))
    return selected, True, False


def _select_index_quality_gated_correct_compression(
    candidate_indices: list[int],
    response_len: torch.Tensor,
    correct: torch.Tensor,
    normalized_teacher_logprob: torch.Tensor,
    teacher_accepted: torch.Tensor,
    drop_rejected_no_correct: bool,
) -> tuple[int | None, bool, bool]:
    correct_indices = [idx for idx in candidate_indices if bool(correct[idx].item())]
    if correct_indices:
        selected = _select_shortest_with_teacher_tiebreak(correct_indices, response_len, normalized_teacher_logprob)
        return selected, False, False

    accepted_indices = [idx for idx in candidate_indices if bool(teacher_accepted[idx].item())]
    if accepted_indices:
        selected = _select_shortest_with_teacher_tiebreak(accepted_indices, response_len, normalized_teacher_logprob)
        return selected, True, False

    if drop_rejected_no_correct:
        return None, False, True

    selected = max(candidate_indices, key=lambda idx: float(normalized_teacher_logprob[idx].item()))
    return selected, True, False


def _safe_mean(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return values.float().mean().item()


def select_short_correct_candidates(batch: DataProto, selection_config) -> tuple[DataProto, dict[str, float]]:
    """Select one candidate per prompt uid for OPD training.

    Supported methods:
    - shortest_correct_else_teacher: legacy behavior; shortest correct if any, otherwise teacher-best.
    - quality_gated_correct_compression: shortest correct if any; otherwise shortest teacher-accepted;
      optionally drop no-correct groups where all candidates are teacher-rejected.
    """
    enabled = bool(_config_get(selection_config, "enabled", False))
    if not enabled:
        return batch, {"candidate_selection/enabled": 0.0}

    method = _config_get(selection_config, "method", "shortest_correct_else_teacher")
    if method not in _SUPPORTED_METHODS:
        raise ValueError(f"Unsupported candidate_selection.method: {method}. Supported: {sorted(_SUPPORTED_METHODS)}")

    keep_per_uid = int(_config_get(selection_config, "keep_per_uid", 1))
    if keep_per_uid != 1:
        raise ValueError("candidate selection currently supports keep_per_uid=1 only")

    if "uid" not in batch.non_tensor_batch:
        raise ValueError("candidate selection requires non_tensor_batch['uid']")

    response_len, correct, normalized_teacher_logprob = _candidate_selection_scores(
        batch=batch,
        correct_reward_threshold=float(_config_get(selection_config, "correct_reward_threshold", 0.5)),
    )
    teacher_accepted, teacher_reject_threshold = _teacher_accept_mask(
        normalized_teacher_logprob=normalized_teacher_logprob,
        teacher_reject_percentile=float(_config_get(selection_config, "teacher_reject_percentile", 20.0)),
    )

    uid_to_indices: OrderedDict[str, list[int]] = OrderedDict()
    for idx, uid in enumerate(batch.non_tensor_batch["uid"]):
        uid_to_indices.setdefault(str(uid), []).append(idx)

    selected_indices: list[int] = []
    fallback_teacher_count = 0
    any_correct_count = 0
    no_correct_count = 0
    no_correct_teacher_accept_count = 0
    dropped_uid_count = 0
    drop_rejected_no_correct = bool(_config_get(selection_config, "drop_rejected_no_correct", True))

    for indices in uid_to_indices.values():
        has_correct = any(bool(correct[idx].item()) for idx in indices)
        if has_correct:
            any_correct_count += 1
        else:
            no_correct_count += 1

        if method == "shortest_correct_else_teacher":
            selected, used_teacher_fallback, dropped = _select_index_shortest_correct_else_teacher(
                candidate_indices=indices,
                response_len=response_len,
                correct=correct,
                normalized_teacher_logprob=normalized_teacher_logprob,
            )
        else:
            has_teacher_accept = any(bool(teacher_accepted[idx].item()) for idx in indices)
            if (not has_correct) and has_teacher_accept:
                no_correct_teacher_accept_count += 1
            selected, used_teacher_fallback, dropped = _select_index_quality_gated_correct_compression(
                candidate_indices=indices,
                response_len=response_len,
                correct=correct,
                normalized_teacher_logprob=normalized_teacher_logprob,
                teacher_accepted=teacher_accepted,
                drop_rejected_no_correct=drop_rejected_no_correct,
            )

        fallback_teacher_count += int(used_teacher_fallback)
        dropped_uid_count += int(dropped)
        if selected is not None:
            selected_indices.append(selected)

    selected_index_tensor = torch.tensor(selected_indices, dtype=torch.long)
    selected = batch.select_idxs(selected_index_tensor)

    selected_response_len = response_len[selected_index_tensor]
    selected_correct = correct[selected_index_tensor].float()
    selected_teacher_logprob = normalized_teacher_logprob[selected_index_tensor]
    selected_correct_len = selected_response_len[selected_correct.bool()]
    selected_wrong_len = selected_response_len[~selected_correct.bool()]
    correct_candidate_len = response_len[correct]
    wrong_candidate_len = response_len[~correct]

    groups = len(uid_to_indices)
    selected_groups = len(selected_indices)
    candidates = len(batch)

    metrics = {
        "candidate_selection/enabled": 1.0,
        "candidate_selection/groups": float(groups),
        "candidate_selection/selected_groups": float(selected_groups),
        "candidate_selection/candidates": float(candidates),
        "candidate_selection/keep_ratio": float(selected_groups / max(candidates, 1)),
        "candidate_selection/any_correct_ratio": float(any_correct_count / max(groups, 1)),
        "candidate_selection/no_correct_ratio": float(no_correct_count / max(groups, 1)),
        "candidate_selection/selected_correct_ratio": _safe_mean(selected_correct),
        "candidate_selection/candidate_response_len_mean": response_len.float().mean().item(),
        "candidate_selection/selected_response_len_mean": _safe_mean(selected_response_len),
        "candidate_selection/fallback_teacher_ratio": float(fallback_teacher_count / max(groups, 1)),
        "candidate_selection/selected_teacher_logprob_mean": _safe_mean(selected_teacher_logprob),
        "candidate_selection/teacher_accept_ratio": teacher_accepted.float().mean().item(),
        "candidate_selection/no_correct_teacher_accept_ratio": float(no_correct_teacher_accept_count / max(groups, 1)),
        "candidate_selection/dropped_uid_ratio": float(dropped_uid_count / max(groups, 1)),
        "candidate_selection/selected_correct_len_mean": _safe_mean(selected_correct_len),
        "candidate_selection/selected_wrong_len_mean": _safe_mean(selected_wrong_len),
        "candidate_selection/correct_candidate_len_mean": _safe_mean(correct_candidate_len),
        "candidate_selection/wrong_candidate_len_mean": _safe_mean(wrong_candidate_len),
        "candidate_selection/teacher_reject_threshold": float(teacher_reject_threshold.item()),
    }
    return selected, metrics
```

- [ ] **Step 2: Run candidate selection tests**

Run:

```bash
cd /home/mchen/FiRe-OPD
PYTHONPATH=/home/mchen/FiRe-OPD/verl /home/mchen/miniconda3/envs/verl/bin/python -m pytest \
  verl/tests/trainer/ppo/test_candidate_selection.py -q
```

Expected: PASS, all candidate-selection tests pass.

- [ ] **Step 3: Commit implementation**

```bash
git add verl/verl/trainer/ppo/candidate_selection.py verl/tests/trainer/ppo/test_candidate_selection.py
git commit -m "feat: add quality-gated OPD candidate selection"
```

---

### Task 3: Add candidate-selection config fields

**Files:**
- Modify: `verl/tests/trainer/config/test_algorithm_config.py`
- Modify: `verl/verl/trainer/config/algorithm.py`
- Modify: `verl/verl/trainer/config/ppo_trainer.yaml`
- Test: `verl/tests/trainer/config/test_algorithm_config.py`

- [ ] **Step 1: Update config test to fail on missing fields**

Replace `test_candidate_selection_config_defaults_and_overrides` in `verl/tests/trainer/config/test_algorithm_config.py` with:

```python
def test_candidate_selection_config_defaults_and_overrides():
    default = CandidateSelectionConfig()
    assert default.enabled is False
    assert default.method == "shortest_correct_else_teacher"
    assert default.correct_reward_threshold == 0.5
    assert default.keep_per_uid == 1
    assert default.teacher_reject_percentile == 20.0
    assert default.drop_rejected_no_correct is True

    cfg = omega_conf_to_dataclass(
        {
            "_target_": "verl.trainer.config.AlgoConfig",
            "candidate_selection": {
                "_target_": "verl.trainer.config.CandidateSelectionConfig",
                "enabled": True,
                "method": "quality_gated_correct_compression",
                "correct_reward_threshold": 0.25,
                "keep_per_uid": 1,
                "teacher_reject_percentile": 15.0,
                "drop_rejected_no_correct": False,
            },
        }
    )
    assert isinstance(cfg, AlgoConfig)
    assert isinstance(cfg.candidate_selection, CandidateSelectionConfig)
    assert cfg.candidate_selection.enabled is True
    assert cfg.candidate_selection.method == "quality_gated_correct_compression"
    assert cfg.candidate_selection.correct_reward_threshold == 0.25
    assert cfg.candidate_selection.teacher_reject_percentile == 15.0
    assert cfg.candidate_selection.drop_rejected_no_correct is False
```

- [ ] **Step 2: Run config test to verify it fails**

Run:

```bash
cd /home/mchen/FiRe-OPD
PYTHONPATH=/home/mchen/FiRe-OPD/verl /home/mchen/miniconda3/envs/verl/bin/python -m pytest \
  verl/tests/trainer/config/test_algorithm_config.py -q
```

Expected: FAIL with `AttributeError` for `teacher_reject_percentile` or `drop_rejected_no_correct`.

- [ ] **Step 3: Add dataclass fields**

In `verl/verl/trainer/config/algorithm.py`, replace `CandidateSelectionConfig` with:

```python
@dataclass
class CandidateSelectionConfig(BaseConfig):
    """Configuration for OPD multi-candidate response selection."""

    enabled: bool = False
    method: str = "shortest_correct_else_teacher"
    correct_reward_threshold: float = 0.5
    keep_per_uid: int = 1
    teacher_reject_percentile: float = 20.0
    drop_rejected_no_correct: bool = True
```

- [ ] **Step 4: Add YAML defaults**

In `verl/verl/trainer/config/ppo_trainer.yaml`, under `algorithm.candidate_selection`, after `keep_per_uid: 1`, add:

```yaml

    # Bottom teacher normalized-logprob percentile used to reject low-quality no-correct candidates
    teacher_reject_percentile: 20.0

    # Whether no-correct groups with only teacher-rejected candidates are dropped from actor update
    drop_rejected_no_correct: True
```

- [ ] **Step 5: Run config test**

Run:

```bash
cd /home/mchen/FiRe-OPD
PYTHONPATH=/home/mchen/FiRe-OPD/verl /home/mchen/miniconda3/envs/verl/bin/python -m pytest \
  verl/tests/trainer/config/test_algorithm_config.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit config changes**

```bash
git add verl/tests/trainer/config/test_algorithm_config.py verl/verl/trainer/config/algorithm.py verl/verl/trainer/config/ppo_trainer.yaml
git commit -m "feat: add quality-gated candidate selection config"
```

---

### Task 4: Add `length_penalty_gate=correct`

**Files:**
- Modify: `verl/tests/workers/actor/test_length_aware_opd.py`
- Modify: `verl/verl/workers/actor/dp_actor.py`
- Test: `verl/tests/workers/actor/test_length_aware_opd.py`

- [ ] **Step 1: Add failing tests for correct-only length gate**

Append these tests to `verl/tests/workers/actor/test_length_aware_opd.py`:

```python
def test_length_penalty_correct_gate_penalizes_long_correct_response_only():
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    ref_log_prob = torch.full_like(response_mask, -0.2)
    token_level_scores = torch.zeros_like(response_mask)
    token_level_scores[0, 3] = 0.0
    token_level_scores[1, 7] = 1.0
    token_level_scores[2, 1] = 0.0

    tensors, metrics = _compute_length_aware_opd_tensors(
        response_mask=response_mask,
        ref_log_prob=ref_log_prob,
        token_level_scores=token_level_scores,
        policy_loss_config=_policy_loss_config(
            length_penalty_gate="correct",
            length_teacher_reject_percentile=0.0,
        ),
    )

    expected_base_penalty = torch.log(torch.tensor(8.0 / 4.0))
    assert torch.isclose(tensors["length_aware_opd_base_penalty"][1], expected_base_penalty)
    assert torch.isclose(tensors["length_aware_opd_applied_penalty"][1], expected_base_penalty)
    assert torch.isclose(tensors["length_aware_opd_penalty"][1], expected_base_penalty * 0.02)
    assert tensors["length_aware_opd_applied_penalty"][0].item() == 0.0
    assert tensors["length_aware_opd_applied_penalty"][2].item() == 0.0
    assert metrics["length_aware_opd/penalty_gate_ratio"] == pytest.approx(1.0 / 3.0)
    assert metrics["length_aware_opd/correct_penalty_ratio"] == pytest.approx(1.0 / 3.0)


def test_length_penalty_correct_gate_requires_scores():
    response_mask = torch.ones((2, 4), dtype=torch.float32)
    ref_log_prob = torch.full_like(response_mask, -0.2)

    with pytest.raises(ValueError, match="token_level_scores is required"):
        _compute_length_aware_opd_tensors(
            response_mask=response_mask,
            ref_log_prob=ref_log_prob,
            token_level_scores=None,
            policy_loss_config=_policy_loss_config(length_penalty_gate="correct"),
        )
```

- [ ] **Step 2: Run length-aware tests to verify they fail**

Run:

```bash
cd /home/mchen/FiRe-OPD
PYTHONPATH=/home/mchen/FiRe-OPD/verl /home/mchen/miniconda3/envs/verl/bin/python -m pytest \
  verl/tests/workers/actor/test_length_aware_opd.py -q
```

Expected: FAIL with `Invalid length_penalty_gate: correct`.

- [ ] **Step 3: Add `correct` to supported gates**

In `verl/verl/workers/actor/dp_actor.py`, replace:

```python
_LENGTH_PENALTY_GATES = {"incorrect", "low_teacher", "incorrect_or_low_teacher"}
```

with:

```python
_LENGTH_PENALTY_GATES = {"incorrect", "low_teacher", "incorrect_or_low_teacher", "correct"}
```

- [ ] **Step 4: Require scores for correct gate**

In `_compute_length_aware_opd_tensors`, replace:

```python
needs_scores = penalty_gate in {"incorrect", "incorrect_or_low_teacher"}
```

with:

```python
needs_scores = penalty_gate in {"incorrect", "incorrect_or_low_teacher", "correct"}
```

- [ ] **Step 5: Implement correct gate**

In `_compute_length_aware_opd_tensors`, replace:

```python
    if penalty_gate == "incorrect":
        gate = incorrect
    elif penalty_gate == "low_teacher":
        gate = teacher_reject
    else:
        gate = incorrect | teacher_reject
```

with:

```python
    if penalty_gate == "incorrect":
        gate = incorrect
    elif penalty_gate == "low_teacher":
        gate = teacher_reject
    elif penalty_gate == "correct":
        gate = correct
    else:
        gate = incorrect | teacher_reject
```

- [ ] **Step 6: Add clearer correct-gate metric**

In `_summarize_length_aware_opd_tensors`, replace the returned dict with:

```python
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
```

- [ ] **Step 7: Run length-aware tests**

Run:

```bash
cd /home/mchen/FiRe-OPD
PYTHONPATH=/home/mchen/FiRe-OPD/verl /home/mchen/miniconda3/envs/verl/bin/python -m pytest \
  verl/tests/workers/actor/test_length_aware_opd.py -q
```

Expected: PASS.

- [ ] **Step 8: Run actor config tests to ensure the new string remains accepted in config objects**

Run:

```bash
cd /home/mchen/FiRe-OPD
PYTHONPATH=/home/mchen/FiRe-OPD/verl /home/mchen/miniconda3/envs/verl/bin/python -m pytest \
  verl/tests/workers/config/test_actor_config_on_cpu.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit correct gate implementation**

```bash
git add verl/tests/workers/actor/test_length_aware_opd.py verl/verl/workers/actor/dp_actor.py
git commit -m "feat: add correct-gated length penalty for OPD"
```

---

### Task 5: Add correct-compression training scripts

**Files:**
- Create: `verl/examples/fire_opd/run_opd_strong_to_weak_lengthaware_correctcompress_teacher30b.sh`
- Create: `run_train_lengthaware_correctcompress_opd.sh`
- Test: shell syntax checks

- [ ] **Step 1: Create the lower-level experiment launcher**

Create `verl/examples/fire_opd/run_opd_strong_to_weak_lengthaware_correctcompress_teacher30b.sh` with:

```bash
#!/bin/bash
# Original OPD strong-to-weak with correct-compression length penalty and quality-gated candidate selection.
# Student rollout and teacher/ref log-prob both consume the original OPD prompt column.
#
# Student: Qwen3-4B
# Teacher: Qwen3-30B-A3B-Instruct-2507
set -x
set -euo pipefail

export PYTHONUNBUFFERED=1
REPO_DIR="${REPO_DIR:-/home/mchen/FiRe-OPD}"
export PYTHONPATH="${REPO_DIR}/verl:${REPO_DIR}:${PYTHONPATH:-}"

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

if [ -n "${SLURM_JOB_ID:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
else
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7,8,9}"
fi
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-${SLURM_CPUS_PER_TASK:-16}}"
PYTHON_BIN="${PYTHON_BIN:-/home/mchen/miniconda3/envs/verl/bin/python}"

N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-4}"
ROLLOUT_N="${ROLLOUT_N:-2}"
RAW_DATA_ROOT="${RAW_DATA_ROOT:-${REPO_DIR}/data/g-opd}"

LENGTH_PENALTY_COEF="${LENGTH_PENALTY_COEF:-0.02}"
LENGTH_PENALTY_GATE="${LENGTH_PENALTY_GATE:-correct}"
LENGTH_CORRECT_REWARD_THRESHOLD="${LENGTH_CORRECT_REWARD_THRESHOLD:-0.5}"
LENGTH_TEACHER_REJECT_PERCENTILE="${LENGTH_TEACHER_REJECT_PERCENTILE:-20.0}"

CANDIDATE_SELECTION_ENABLED="${CANDIDATE_SELECTION_ENABLED:-True}"
CANDIDATE_SELECTION_METHOD="${CANDIDATE_SELECTION_METHOD:-quality_gated_correct_compression}"
CANDIDATE_SELECTION_KEEP_PER_UID="${CANDIDATE_SELECTION_KEEP_PER_UID:-1}"
CANDIDATE_SELECTION_CORRECT_REWARD_THRESHOLD="${CANDIDATE_SELECTION_CORRECT_REWARD_THRESHOLD:-0.5}"
CANDIDATE_SELECTION_TEACHER_REJECT_PERCENTILE="${CANDIDATE_SELECTION_TEACHER_REJECT_PERCENTILE:-20.0}"
CANDIDATE_SELECTION_DROP_REJECTED_NO_CORRECT="${CANDIDATE_SELECTION_DROP_REJECTED_NO_CORRECT:-True}"

TRAIN_DATA="${TRAIN_DATA:-${RAW_DATA_ROOT}/DeepMath-103K/train_filtered_level6.parquet}"
VAL_DATA="${VAL_DATA:-['${RAW_DATA_ROOT}/AIME2024/test.parquet', '${RAW_DATA_ROOT}/AIME2025/test.parquet']}"

STUDENT_MODEL="${STUDENT_MODEL:-${REPO_DIR}/models/Qwen3-4B}"
TEACHER_MODEL="${TEACHER_MODEL:-${REPO_DIR}/models/Qwen3-30B-A3B-Instruct-2507}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-opd-strong-to-weak-lengthaware-correctcompress-lambda${LENGTH_PENALTY_COEF}-selectn${ROLLOUT_N}-rawprompt-${N_GPUS_PER_NODE}gpu-tp${ROLLOUT_TP_SIZE}-refmb4-rollmb4}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_DIR}/checkpoints/${EXPERIMENT_NAME}}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"

if [ ! -f "${TRAIN_DATA}" ]; then
  echo "ERROR: TRAIN_DATA not found: ${TRAIN_DATA}" >&2
  exit 1
fi

"${PYTHON_BIN}" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=5.0 \
    algorithm.rollout_correction.rollout_rs=null \
    algorithm.rollout_correction.bypass_mode=false \
    algorithm.candidate_selection.enabled=${CANDIDATE_SELECTION_ENABLED} \
    algorithm.candidate_selection.method=${CANDIDATE_SELECTION_METHOD} \
    algorithm.candidate_selection.keep_per_uid=${CANDIDATE_SELECTION_KEEP_PER_UID} \
    algorithm.candidate_selection.correct_reward_threshold=${CANDIDATE_SELECTION_CORRECT_REWARD_THRESHOLD} \
    algorithm.candidate_selection.teacher_reject_percentile=${CANDIDATE_SELECTION_TEACHER_REJECT_PERCENTILE} \
    algorithm.candidate_selection.drop_rejected_no_correct=${CANDIDATE_SELECTION_DROP_REJECTED_NO_CORRECT} \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    data.train_files=${TRAIN_DATA} \
    data.val_files="${VAL_DATA}" \
    data.train_batch_size=1024 \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    data.seed=42 \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=${STUDENT_MODEL} \
    +actor_rollout_ref.ref.model.path=${TEACHER_MODEL} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=True \
    actor_rollout_ref.actor.policy_loss.length_aware_opd=True \
    actor_rollout_ref.actor.policy_loss.length_penalty_coef=${LENGTH_PENALTY_COEF} \
    actor_rollout_ref.actor.policy_loss.length_penalty_type=log_batch_median \
    actor_rollout_ref.actor.policy_loss.length_penalty_gate=${LENGTH_PENALTY_GATE} \
    actor_rollout_ref.actor.policy_loss.length_correct_reward_threshold=${LENGTH_CORRECT_REWARD_THRESHOLD} \
    actor_rollout_ref.actor.policy_loss.length_teacher_reject_percentile=${LENGTH_TEACHER_REJECT_PERCENTILE} \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.ppo_mini_batch_size=1024 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP_SIZE} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=naive \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.logger='["console","wandb"]' \
    trainer.log_val_generations=10 \
    trainer.project_name='fire-opd' \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.default_local_dir=${CHECKPOINT_DIR} \
    trainer.test_freq=10 \
    trainer.total_epochs=3 \
    trainer.resume_mode=auto \
    ray_kwargs.ray_init.num_cpus=${RAY_NUM_CPUS}
```

- [ ] **Step 2: Create top-level srun-aware wrapper**

Create `run_train_lengthaware_correctcompress_opd.sh` with:

```bash
#!/bin/bash
set -euo pipefail

cd /home/mchen/FiRe-OPD

# srun/sbatch should control visible logical GPU ids. Outside Slurm,
# default to physical GPUs 1-4 to match the local interactive convention.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
else
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4}"
fi

ROLLOUT_N="${ROLLOUT_N:-2}" \
LENGTH_PENALTY_COEF="${LENGTH_PENALTY_COEF:-0.02}" \
LENGTH_PENALTY_GATE="${LENGTH_PENALTY_GATE:-correct}" \
LENGTH_CORRECT_REWARD_THRESHOLD="${LENGTH_CORRECT_REWARD_THRESHOLD:-0.5}" \
LENGTH_TEACHER_REJECT_PERCENTILE="${LENGTH_TEACHER_REJECT_PERCENTILE:-20.0}" \
CANDIDATE_SELECTION_ENABLED="${CANDIDATE_SELECTION_ENABLED:-True}" \
CANDIDATE_SELECTION_METHOD="${CANDIDATE_SELECTION_METHOD:-quality_gated_correct_compression}" \
CANDIDATE_SELECTION_KEEP_PER_UID="${CANDIDATE_SELECTION_KEEP_PER_UID:-1}" \
CANDIDATE_SELECTION_CORRECT_REWARD_THRESHOLD="${CANDIDATE_SELECTION_CORRECT_REWARD_THRESHOLD:-0.5}" \
CANDIDATE_SELECTION_TEACHER_REJECT_PERCENTILE="${CANDIDATE_SELECTION_TEACHER_REJECT_PERCENTILE:-20.0}" \
CANDIDATE_SELECTION_DROP_REJECTED_NO_CORRECT="${CANDIDATE_SELECTION_DROP_REJECTED_NO_CORRECT:-True}" \
bash verl/examples/fire_opd/run_opd_strong_to_weak_lengthaware_correctcompress_teacher30b.sh
```

- [ ] **Step 3: Make scripts executable and syntax-check them**

Run:

```bash
cd /home/mchen/FiRe-OPD
chmod +x \
  verl/examples/fire_opd/run_opd_strong_to_weak_lengthaware_correctcompress_teacher30b.sh \
  run_train_lengthaware_correctcompress_opd.sh
bash -n verl/examples/fire_opd/run_opd_strong_to_weak_lengthaware_correctcompress_teacher30b.sh
bash -n run_train_lengthaware_correctcompress_opd.sh
```

Expected: both `bash -n` commands exit 0.

- [ ] **Step 4: Verify key settings are present**

Run:

```bash
cd /home/mchen/FiRe-OPD
grep -n 'quality_gated_correct_compression\|length_penalty_gate=${LENGTH_PENALTY_GATE}\|actor_rollout_ref.rollout.n=${ROLLOUT_N}\|drop_rejected_no_correct' \
  verl/examples/fire_opd/run_opd_strong_to_weak_lengthaware_correctcompress_teacher30b.sh
grep -n 'LENGTH_PENALTY_GATE="${LENGTH_PENALTY_GATE:-correct}"\|CANDIDATE_SELECTION_METHOD="${CANDIDATE_SELECTION_METHOD:-quality_gated_correct_compression}"' \
  run_train_lengthaware_correctcompress_opd.sh
```

Expected: lines showing the new method, `length_penalty_gate=correct`, `rollout.n=${ROLLOUT_N}`, and `drop_rejected_no_correct`.

- [ ] **Step 5: Commit scripts**

```bash
git add \
  verl/examples/fire_opd/run_opd_strong_to_weak_lengthaware_correctcompress_teacher30b.sh \
  run_train_lengthaware_correctcompress_opd.sh
git commit -m "feat: add correct-compression OPD training scripts"
```

---

### Task 6: Final verification

**Files:**
- Test-only task; no source edits expected.

- [ ] **Step 1: Run candidate-selection tests**

Run:

```bash
cd /home/mchen/FiRe-OPD
PYTHONPATH=/home/mchen/FiRe-OPD/verl /home/mchen/miniconda3/envs/verl/bin/python -m pytest \
  verl/tests/trainer/ppo/test_candidate_selection.py -q
```

Expected: PASS.

- [ ] **Step 2: Run algorithm config tests**

Run:

```bash
cd /home/mchen/FiRe-OPD
PYTHONPATH=/home/mchen/FiRe-OPD/verl /home/mchen/miniconda3/envs/verl/bin/python -m pytest \
  verl/tests/trainer/config/test_algorithm_config.py -q
```

Expected: PASS.

- [ ] **Step 3: Run length-aware OPD actor tests**

Run:

```bash
cd /home/mchen/FiRe-OPD
PYTHONPATH=/home/mchen/FiRe-OPD/verl /home/mchen/miniconda3/envs/verl/bin/python -m pytest \
  verl/tests/workers/actor/test_length_aware_opd.py -q
```

Expected: PASS.

- [ ] **Step 4: Run actor config tests**

Run:

```bash
cd /home/mchen/FiRe-OPD
PYTHONPATH=/home/mchen/FiRe-OPD/verl /home/mchen/miniconda3/envs/verl/bin/python -m pytest \
  verl/tests/workers/config/test_actor_config_on_cpu.py -q
```

Expected: PASS.

- [ ] **Step 5: Syntax-check scripts**

Run:

```bash
cd /home/mchen/FiRe-OPD
bash -n verl/examples/fire_opd/run_opd_strong_to_weak_lengthaware_correctcompress_teacher30b.sh
bash -n run_train_lengthaware_correctcompress_opd.sh
```

Expected: both exit 0.

- [ ] **Step 6: Print run command but do not start training**

Print this command for the user:

```bash
cd /home/mchen/FiRe-OPD
mkdir -p train_logs
bash run_train_lengthaware_correctcompress_opd.sh 2>&1 | tee "train_logs/lengthaware_correctcompress_lambda0.02_selectn2_$(date +%Y%m%d_%H%M%S).log"
```

Expected checkpoint root:

```text
checkpoints/opd-strong-to-weak-lengthaware-correctcompress-lambda0.02-selectn2-rawprompt-4gpu-tp4-refmb4-rollmb4
```

- [ ] **Step 7: Commit no-op verification note only if additional source changes were needed**

If all previous tasks already committed their source changes and no files changed during verification, do not create another commit.

---

## Self-review Checklist

- Spec coverage:
  - Quality-gated selection: Task 1 and Task 2.
  - Teacher-reject percentile/drop config: Task 3.
  - Correct trajectory compression with `λ=0.02`: Task 4 and Task 5.
  - FiRe-OPD-inspired no-correct filtering: Task 2.
  - Training launcher and srun wrapper: Task 5.
  - Verification: Task 6.
- Placeholder scan: no placeholder markers or unspecified implementation steps remain.
- Type consistency:
  - Method name is consistently `quality_gated_correct_compression`.
  - Config fields are consistently `teacher_reject_percentile` and `drop_rejected_no_correct`.
  - Length gate string is consistently `correct`.
  - Main helper remains `select_short_correct_candidates` for compatibility with `ray_trainer.py`.
