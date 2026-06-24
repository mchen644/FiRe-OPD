# Multi-candidate Short-correct Selection for OPD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional OPD candidate-selection module that generates multiple student responses per prompt and keeps one short/correct or teacher-preferred response for OPD update.

**Architecture:** Add a small helper module under `verl.trainer.ppo` that selects candidate indices by `uid` from an already-generated and already-scored `DataProto`. Add typed config under `algorithm.candidate_selection`, invoke selection in `RayPPOTrainer.fit` after reward/ref log-prob tensors exist and before advantage computation, and add a new experiment launcher stacking selection on top of length-aware OPD `lambda=0.02`.

**Tech Stack:** Python, PyTorch, NumPy, verl `DataProto`, Hydra/OmegaConf config dataclasses, pytest, bash.

---

## File Structure

- Create `verl/verl/trainer/ppo/candidate_selection.py`
  - Pure helper functions for selecting one candidate per `uid` and producing metrics.
- Create `verl/tests/trainer/ppo/test_candidate_selection.py`
  - CPU unit tests for selection behavior and field preservation.
- Modify `verl/verl/trainer/config/algorithm.py`
  - Add `CandidateSelectionConfig` and an optional `candidate_selection` field on `AlgoConfig`.
- Modify `verl/verl/trainer/config/ppo_trainer.yaml`
  - Add default `algorithm.candidate_selection` block with `enabled: false`.
- Modify `verl/verl/trainer/ppo/ray_trainer.py`
  - Invoke selection after reward tensors exist and before rollout correction / `compute_advantage(...)`.
- Create `verl/examples/fire_opd/run_opd_strong_to_weak_lengthaware_select_teacher30b.sh`
  - New training script with `rollout.n=2`, length-aware OPD `lambda=0.02`, and candidate selection enabled.

---

### Task 1: Add failing unit tests for candidate selection helper

**Files:**
- Create: `verl/tests/trainer/ppo/test_candidate_selection.py`

- [ ] **Step 1: Write the failing tests**

Create `verl/tests/trainer/ppo/test_candidate_selection.py` with this content:

```python
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.candidate_selection import select_short_correct_candidates


def _cfg(**overrides):
    values = {
        "enabled": True,
        "method": "shortest_correct_else_teacher",
        "correct_reward_threshold": 0.5,
        "keep_per_uid": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _batch_for_selection():
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],  # uid-a, correct but long
            [1, 1, 1, 0, 0, 0],  # uid-a, correct and short -> select
            [1, 1, 0, 0, 0, 0],  # uid-b, incorrect, teacher worse
            [1, 1, 1, 1, 0, 0],  # uid-b, incorrect, teacher better -> select
        ],
        dtype=torch.float32,
    )
    ref_log_prob = torch.tensor(
        [
            [-0.2, -0.2, -0.2, -0.2, -0.2, -0.2],
            [-0.5, -0.5, -0.5, 0.0, 0.0, 0.0],
            [-3.0, -3.0, 0.0, 0.0, 0.0, 0.0],
            [-0.3, -0.3, -0.3, -0.3, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    token_level_scores = torch.zeros_like(response_mask)
    token_level_scores[0, 5] = 1.0
    token_level_scores[1, 2] = 1.0
    input_ids = torch.arange(4 * 6, dtype=torch.long).view(4, 6)
    batch = DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "response_mask": response_mask,
            "ref_log_prob": ref_log_prob,
            "token_level_scores": token_level_scores,
        },
        non_tensors={
            "uid": np.array(["uid-a", "uid-a", "uid-b", "uid-b"], dtype=object),
            "source": np.array(["a0", "a1", "b0", "b1"], dtype=object),
        },
    )
    return batch


def test_selects_shortest_correct_candidate_and_teacher_best_fallback():
    batch = _batch_for_selection()

    selected, metrics = select_short_correct_candidates(batch, _cfg())

    assert selected.batch["input_ids"].tolist() == [
        [6, 7, 8, 9, 10, 11],
        [18, 19, 20, 21, 22, 23],
    ]
    assert selected.non_tensor_batch["uid"].tolist() == ["uid-a", "uid-b"]
    assert selected.non_tensor_batch["source"].tolist() == ["a1", "b1"]
    assert metrics["candidate_selection/enabled"] == 1.0
    assert metrics["candidate_selection/groups"] == 2.0
    assert metrics["candidate_selection/candidates"] == 4.0
    assert metrics["candidate_selection/keep_ratio"] == 0.5
    assert metrics["candidate_selection/any_correct_ratio"] == 0.5
    assert metrics["candidate_selection/selected_correct_ratio"] == 0.5
    assert metrics["candidate_selection/fallback_teacher_ratio"] == 0.5
    assert metrics["candidate_selection/selected_response_len_mean"] == 3.5
    assert metrics["candidate_selection/candidate_response_len_mean"] == 3.75


def test_disabled_selection_returns_original_batch_and_disabled_metric():
    batch = _batch_for_selection()

    selected, metrics = select_short_correct_candidates(batch, _cfg(enabled=False))

    assert selected is batch
    assert len(selected) == 4
    assert metrics == {"candidate_selection/enabled": 0.0}


def test_selection_rejects_unsupported_keep_per_uid_and_method():
    batch = _batch_for_selection()

    with pytest.raises(ValueError, match="keep_per_uid=1"):
        select_short_correct_candidates(batch, _cfg(keep_per_uid=2))

    with pytest.raises(ValueError, match="Unsupported candidate_selection.method"):
        select_short_correct_candidates(batch, _cfg(method="shortest_only"))


def test_selection_requires_uid_ref_log_prob_response_mask_and_scores():
    batch = _batch_for_selection()

    no_uid = DataProto.from_dict(
        tensors={key: value.clone() for key, value in batch.batch.items()},
        non_tensors={"source": batch.non_tensor_batch["source"].copy()},
    )
    with pytest.raises(ValueError, match="uid"):
        select_short_correct_candidates(no_uid, _cfg())

    missing_ref = _batch_for_selection()
    missing_ref.batch.pop("ref_log_prob")
    with pytest.raises(ValueError, match="ref_log_prob"):
        select_short_correct_candidates(missing_ref, _cfg())

    missing_scores = _batch_for_selection()
    missing_scores.batch.pop("token_level_scores")
    with pytest.raises(ValueError, match="token_level_scores"):
        select_short_correct_candidates(missing_scores, _cfg())


def test_selection_handles_single_candidate_groups():
    response_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.float32)
    ref_log_prob = torch.full_like(response_mask, -0.2)
    token_level_scores = torch.zeros_like(response_mask)
    token_level_scores[0, 1] = 1.0
    batch = DataProto.from_dict(
        tensors={
            "response_mask": response_mask,
            "ref_log_prob": ref_log_prob,
            "token_level_scores": token_level_scores,
        },
        non_tensors={"uid": np.array(["u0", "u1"], dtype=object)},
    )

    selected, metrics = select_short_correct_candidates(batch, _cfg())

    assert len(selected) == 2
    assert metrics["candidate_selection/keep_ratio"] == 1.0
    assert metrics["candidate_selection/groups"] == 2.0
```

- [ ] **Step 2: Run the tests and verify they fail because the module does not exist**

Run:

```bash
cd /home/mchen/FiRe-OPD
PYTHONPATH=/home/mchen/FiRe-OPD/verl pytest verl/tests/trainer/ppo/test_candidate_selection.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'verl.trainer.ppo.candidate_selection'`.

- [ ] **Step 3: Commit the failing tests**

Run:

```bash
git add verl/tests/trainer/ppo/test_candidate_selection.py
git commit -m "test: add OPD candidate selection tests"
```

---

### Task 2: Implement candidate selection helper

**Files:**
- Create: `verl/verl/trainer/ppo/candidate_selection.py`
- Test: `verl/tests/trainer/ppo/test_candidate_selection.py`

- [ ] **Step 1: Add the helper module**

Create `verl/verl/trainer/ppo/candidate_selection.py` with this content:

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

import numpy as np
import torch

from verl import DataProto

_SUPPORTED_METHODS = {"shortest_correct_else_teacher"}


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


def _select_index_for_uid(
    candidate_indices: list[int],
    response_len: torch.Tensor,
    correct: torch.Tensor,
    normalized_teacher_logprob: torch.Tensor,
) -> tuple[int, bool]:
    correct_indices = [idx for idx in candidate_indices if bool(correct[idx].item())]
    if correct_indices:
        selected = max(
            correct_indices,
            key=lambda idx: (-float(response_len[idx].item()), float(normalized_teacher_logprob[idx].item())),
        )
        return selected, False

    selected = max(candidate_indices, key=lambda idx: float(normalized_teacher_logprob[idx].item()))
    return selected, True


def select_short_correct_candidates(batch: DataProto, selection_config) -> tuple[DataProto, dict[str, float]]:
    """Select one candidate per prompt uid for OPD training.

    The initial method keeps the shortest correct candidate when a group has any
    correct response, otherwise it falls back to the candidate with highest
    normalized teacher log-probability.
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

    uid_to_indices: OrderedDict[str, list[int]] = OrderedDict()
    for idx, uid in enumerate(batch.non_tensor_batch["uid"]):
        uid_to_indices.setdefault(str(uid), []).append(idx)

    selected_indices: list[int] = []
    fallback_teacher_count = 0
    any_correct_count = 0
    for indices in uid_to_indices.values():
        if any(bool(correct[idx].item()) for idx in indices):
            any_correct_count += 1
        selected, used_teacher_fallback = _select_index_for_uid(
            candidate_indices=indices,
            response_len=response_len,
            correct=correct,
            normalized_teacher_logprob=normalized_teacher_logprob,
        )
        selected_indices.append(selected)
        fallback_teacher_count += int(used_teacher_fallback)

    selected_index_tensor = torch.tensor(selected_indices, dtype=torch.long)
    selected = batch.select_idxs(selected_index_tensor)

    selected_response_len = response_len[selected_index_tensor]
    selected_correct = correct[selected_index_tensor].float()
    selected_teacher_logprob = normalized_teacher_logprob[selected_index_tensor]
    groups = len(selected_indices)
    candidates = len(batch)

    metrics = {
        "candidate_selection/enabled": 1.0,
        "candidate_selection/groups": float(groups),
        "candidate_selection/candidates": float(candidates),
        "candidate_selection/keep_ratio": float(groups / max(candidates, 1)),
        "candidate_selection/any_correct_ratio": float(any_correct_count / max(groups, 1)),
        "candidate_selection/selected_correct_ratio": selected_correct.mean().item() if groups else 0.0,
        "candidate_selection/candidate_response_len_mean": response_len.float().mean().item(),
        "candidate_selection/selected_response_len_mean": selected_response_len.float().mean().item() if groups else 0.0,
        "candidate_selection/fallback_teacher_ratio": float(fallback_teacher_count / max(groups, 1)),
        "candidate_selection/selected_teacher_logprob_mean": selected_teacher_logprob.float().mean().item() if groups else 0.0,
    }
    return selected, metrics
```

- [ ] **Step 2: Run the helper tests**

Run:

```bash
cd /home/mchen/FiRe-OPD
PYTHONPATH=/home/mchen/FiRe-OPD/verl pytest verl/tests/trainer/ppo/test_candidate_selection.py -q
```

Expected: PASS.

- [ ] **Step 3: Commit the helper implementation**

Run:

```bash
git add verl/verl/trainer/ppo/candidate_selection.py
git commit -m "feat: add OPD short-correct candidate selection helper"
```

---

### Task 3: Add typed config for candidate selection

**Files:**
- Modify: `verl/verl/trainer/config/algorithm.py`
- Modify: `verl/verl/trainer/config/ppo_trainer.yaml`
- Create or modify: `verl/tests/trainer/config/test_algorithm_config.py`

- [ ] **Step 1: Add failing config tests**

Create `verl/tests/trainer/config/test_algorithm_config.py` with this content:

```python
from verl.trainer.config import AlgoConfig, CandidateSelectionConfig
from verl.utils.config import omega_conf_to_dataclass


def test_candidate_selection_config_defaults_and_overrides():
    default = CandidateSelectionConfig()
    assert default.enabled is False
    assert default.method == "shortest_correct_else_teacher"
    assert default.correct_reward_threshold == 0.5
    assert default.keep_per_uid == 1

    cfg = omega_conf_to_dataclass(
        {
            "_target_": "verl.trainer.config.AlgoConfig",
            "candidate_selection": {
                "_target_": "verl.trainer.config.CandidateSelectionConfig",
                "enabled": True,
                "method": "shortest_correct_else_teacher",
                "correct_reward_threshold": 0.25,
                "keep_per_uid": 1,
            },
        }
    )
    assert isinstance(cfg, AlgoConfig)
    assert isinstance(cfg.candidate_selection, CandidateSelectionConfig)
    assert cfg.candidate_selection.enabled is True
    assert cfg.candidate_selection.correct_reward_threshold == 0.25
```

- [ ] **Step 2: Run the config test and verify it fails**

Run:

```bash
cd /home/mchen/FiRe-OPD
PYTHONPATH=/home/mchen/FiRe-OPD/verl pytest verl/tests/trainer/config/test_algorithm_config.py -q
```

Expected: FAIL with `ImportError` for `CandidateSelectionConfig`.

- [ ] **Step 3: Add the config dataclass and field**

In `verl/verl/trainer/config/algorithm.py`, update `__all__` from:

```python
__all__ = ["AlgoConfig", "FilterGroupsConfig", "KLControlConfig", "RolloutCorrectionConfig"]
```

to:

```python
__all__ = ["AlgoConfig", "CandidateSelectionConfig", "FilterGroupsConfig", "KLControlConfig", "RolloutCorrectionConfig"]
```

Then insert this dataclass immediately before `class AlgoConfig`:

```python
@dataclass
class CandidateSelectionConfig(BaseConfig):
    """Configuration for OPD multi-candidate response selection."""

    enabled: bool = False
    method: str = "shortest_correct_else_teacher"
    correct_reward_threshold: float = 0.5
    keep_per_uid: int = 1
```

Then add this field to `AlgoConfig` immediately after `filter_groups`:

```python
    candidate_selection: Optional[CandidateSelectionConfig] = field(default_factory=CandidateSelectionConfig)
```

- [ ] **Step 4: Export the config class**

Find `verl/verl/trainer/config/__init__.py`. Add `CandidateSelectionConfig` to the import from `.algorithm` and to `__all__` if the file has an explicit `__all__` list.

The import block should include:

```python
from .algorithm import AlgoConfig, CandidateSelectionConfig, FilterGroupsConfig, KLControlConfig, RolloutCorrectionConfig
```

- [ ] **Step 5: Add YAML defaults**

In `verl/verl/trainer/config/ppo_trainer.yaml`, under `algorithm:` after the existing `filter_groups` or `pf_ppo` block, add:

```yaml
  # Optional OPD multi-candidate response selection after reward/ref scoring
  candidate_selection:
    _target_: verl.trainer.config.CandidateSelectionConfig
    enabled: False
    method: shortest_correct_else_teacher
    correct_reward_threshold: 0.5
    keep_per_uid: 1
```

- [ ] **Step 6: Run config tests**

Run:

```bash
cd /home/mchen/FiRe-OPD
PYTHONPATH=/home/mchen/FiRe-OPD/verl pytest verl/tests/trainer/config/test_algorithm_config.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit config changes**

Run:

```bash
git add verl/verl/trainer/config/algorithm.py verl/verl/trainer/config/__init__.py verl/verl/trainer/config/ppo_trainer.yaml verl/tests/trainer/config/test_algorithm_config.py
git commit -m "feat: add OPD candidate selection config"
```

---

### Task 4: Integrate selection into `RayPPOTrainer.fit`

**Files:**
- Modify: `verl/verl/trainer/ppo/ray_trainer.py`
- Test: `verl/tests/trainer/ppo/test_candidate_selection.py`

- [ ] **Step 1: Add an integration-style failing test for selected batch size**

Append this test to `verl/tests/trainer/ppo/test_candidate_selection.py`:

```python
def test_selection_reduces_rollout_n_two_batch_to_one_per_uid():
    batch = _batch_for_selection()
    selected, metrics = select_short_correct_candidates(batch, _cfg())

    assert len(batch) == 4
    assert len(selected) == 2
    assert selected.non_tensor_batch["uid"].tolist() == ["uid-a", "uid-b"]
    assert metrics["candidate_selection/keep_ratio"] == 0.5
```

- [ ] **Step 2: Run helper tests**

Run:

```bash
cd /home/mchen/FiRe-OPD
PYTHONPATH=/home/mchen/FiRe-OPD/verl pytest verl/tests/trainer/ppo/test_candidate_selection.py -q
```

Expected: PASS. This test documents the behavior that `RayPPOTrainer.fit` will rely on.

- [ ] **Step 3: Import the helper in `ray_trainer.py`**

In `verl/verl/trainer/ppo/ray_trainer.py`, add this import near other PPO helper imports:

```python
from verl.trainer.ppo.candidate_selection import select_short_correct_candidates
```

- [ ] **Step 4: Invoke selection before rollout correction and advantage computation**

In `RayPPOTrainer.fit`, find this block inside `with marked_timer("adv", timing_raw):`:

```python
                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
```

Insert this code immediately before that comment:

```python
                        candidate_selection_config = self.config.algorithm.get("candidate_selection", None)
                        if candidate_selection_config is not None and candidate_selection_config.get("enabled", False):
                            batch, candidate_selection_metrics = select_short_correct_candidates(
                                batch=batch,
                                selection_config=candidate_selection_config,
                            )
                            metrics.update(candidate_selection_metrics)
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
cd /home/mchen/FiRe-OPD
PYTHONPATH=/home/mchen/FiRe-OPD/verl pytest verl/tests/trainer/ppo/test_candidate_selection.py verl/tests/trainer/config/test_algorithm_config.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit trainer integration**

Run:

```bash
git add verl/verl/trainer/ppo/ray_trainer.py verl/tests/trainer/ppo/test_candidate_selection.py
git commit -m "feat: select short-correct OPD candidates before advantage computation"
```

---

### Task 5: Add length-aware + selection OPD training script

**Files:**
- Create: `verl/examples/fire_opd/run_opd_strong_to_weak_lengthaware_select_teacher30b.sh`

- [ ] **Step 1: Create the script**

Create `verl/examples/fire_opd/run_opd_strong_to_weak_lengthaware_select_teacher30b.sh` with this content:

```bash
#!/bin/bash
# Original OPD strong-to-weak with length-aware penalty plus multi-candidate short-correct selection.
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
LENGTH_PENALTY_GATE="${LENGTH_PENALTY_GATE:-incorrect_or_low_teacher}"
LENGTH_CORRECT_REWARD_THRESHOLD="${LENGTH_CORRECT_REWARD_THRESHOLD:-0.5}"
LENGTH_TEACHER_REJECT_PERCENTILE="${LENGTH_TEACHER_REJECT_PERCENTILE:-20.0}"

CANDIDATE_SELECTION_ENABLED="${CANDIDATE_SELECTION_ENABLED:-True}"
CANDIDATE_SELECTION_METHOD="${CANDIDATE_SELECTION_METHOD:-shortest_correct_else_teacher}"
CANDIDATE_SELECTION_KEEP_PER_UID="${CANDIDATE_SELECTION_KEEP_PER_UID:-1}"
CANDIDATE_SELECTION_CORRECT_REWARD_THRESHOLD="${CANDIDATE_SELECTION_CORRECT_REWARD_THRESHOLD:-0.5}"

TRAIN_DATA="${TRAIN_DATA:-${RAW_DATA_ROOT}/DeepMath-103K/train_filtered_level6.parquet}"
VAL_DATA="${VAL_DATA:-['${RAW_DATA_ROOT}/AIME2024/test.parquet', '${RAW_DATA_ROOT}/AIME2025/test.parquet']}"

STUDENT_MODEL="${STUDENT_MODEL:-${REPO_DIR}/models/Qwen3-4B}"
TEACHER_MODEL="${TEACHER_MODEL:-${REPO_DIR}/models/Qwen3-30B-A3B-Instruct-2507}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-opd-strong-to-weak-lengthaware-lambda${LENGTH_PENALTY_COEF}-selectn${ROLLOUT_N}-rawprompt-${N_GPUS_PER_NODE}gpu-tp${ROLLOUT_TP_SIZE}-refmb4-rollmb4}"
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

- [ ] **Step 2: Make the script executable and syntax-check it**

Run:

```bash
chmod +x /home/mchen/FiRe-OPD/verl/examples/fire_opd/run_opd_strong_to_weak_lengthaware_select_teacher30b.sh
bash -n /home/mchen/FiRe-OPD/verl/examples/fire_opd/run_opd_strong_to_weak_lengthaware_select_teacher30b.sh
```

Expected: no output and exit code 0.

- [ ] **Step 3: Confirm script settings**

Run:

```bash
grep -n 'rollout.n=${ROLLOUT_N}\|candidate_selection\|length_penalty_coef=${LENGTH_PENALTY_COEF}' /home/mchen/FiRe-OPD/verl/examples/fire_opd/run_opd_strong_to_weak_lengthaware_select_teacher30b.sh
```

Expected: lines showing candidate selection overrides, `rollout.n=${ROLLOUT_N}`, and `length_penalty_coef=${LENGTH_PENALTY_COEF}`.

- [ ] **Step 4: Commit the script**

Run:

```bash
git add verl/examples/fire_opd/run_opd_strong_to_weak_lengthaware_select_teacher30b.sh
git commit -m "feat: add length-aware OPD candidate selection run script"
```

---

### Task 6: Run verification suite

**Files:**
- Verify only.

- [ ] **Step 1: Run selection tests**

Run:

```bash
cd /home/mchen/FiRe-OPD
PYTHONPATH=/home/mchen/FiRe-OPD/verl pytest verl/tests/trainer/ppo/test_candidate_selection.py -q
```

Expected: PASS.

- [ ] **Step 2: Run algorithm config tests**

Run:

```bash
cd /home/mchen/FiRe-OPD
PYTHONPATH=/home/mchen/FiRe-OPD/verl pytest verl/tests/trainer/config/test_algorithm_config.py -q
```

Expected: PASS.

- [ ] **Step 3: Run existing length-aware tests to guard the previous module**

Run:

```bash
cd /home/mchen/FiRe-OPD
PYTHONPATH=/home/mchen/FiRe-OPD/verl pytest verl/tests/workers/actor/test_length_aware_opd.py -q
```

Expected: PASS.

- [ ] **Step 4: Run actor config tests**

Run:

```bash
cd /home/mchen/FiRe-OPD
PYTHONPATH=/home/mchen/FiRe-OPD/verl pytest verl/tests/workers/config/test_actor_config_on_cpu.py -q
```

Expected: PASS.

- [ ] **Step 5: Check script syntax**

Run:

```bash
bash -n /home/mchen/FiRe-OPD/verl/examples/fire_opd/run_opd_strong_to_weak_lengthaware_select_teacher30b.sh
```

Expected: no output and exit code 0.

---

### Task 7: First experiment command and expected logging

**Files:**
- No code changes.

- [ ] **Step 1: Use this first training command inside the srun allocation**

Run:

```bash
cd /home/mchen/FiRe-OPD
mkdir -p train_logs
ROLLOUT_N=2 \
LENGTH_PENALTY_COEF=0.02 \
CANDIDATE_SELECTION_ENABLED=True \
bash verl/examples/fire_opd/run_opd_strong_to_weak_lengthaware_select_teacher30b.sh \
  2>&1 | tee "train_logs/lengthaware_lambda0.02_selectn2_$(date +%Y%m%d_%H%M%S).log"
```

Expected checkpoint root:

```text
checkpoints/opd-strong-to-weak-lengthaware-lambda0.02-selectn2-rawprompt-4gpu-tp4-refmb4-rollmb4
```

- [ ] **Step 2: Confirm candidate-selection metrics appear**

Look for these metrics in console or wandb logs:

```text
candidate_selection/enabled
candidate_selection/groups
candidate_selection/candidates
candidate_selection/keep_ratio
candidate_selection/any_correct_ratio
candidate_selection/selected_correct_ratio
candidate_selection/candidate_response_len_mean
candidate_selection/selected_response_len_mean
candidate_selection/fallback_teacher_ratio
candidate_selection/selected_teacher_logprob_mean
```

Expected early sanity values for `rollout.n=2` and `keep_per_uid=1`:

```text
candidate_selection/keep_ratio ≈ 0.5
candidate_selection/groups ≈ data.train_batch_size
candidate_selection/candidates ≈ 2 * data.train_batch_size
```

- [ ] **Step 3: Confirm length-aware metrics still appear**

Look for:

```text
length_aware_opd/coef = 0.02
length_aware_opd/mean_applied_penalty
length_aware_opd/penalty_gate_ratio
```

Expected: length-aware metrics are computed after selection on the selected batch.

---

## Self-Review Notes

Spec coverage:

- Multi-candidate generation: covered by Task 5 script setting `actor_rollout_ref.rollout.n=${ROLLOUT_N}`.
- One selected response per prompt: covered by Task 2 helper and Task 4 trainer integration.
- Correct-first, shortest-correct selection: covered by Task 1 and Task 2.
- Teacher fallback when no candidate is correct: covered by Task 1 and Task 2.
- Metrics: covered by Task 2 and Task 7.
- Stack on length-aware lambda 0.02: covered by Task 5 and Task 7.

Placeholder scan: no `TBD`, `TODO`, or unspecified implementation steps remain.

Type consistency: config field names are consistently `candidate_selection.enabled`, `candidate_selection.method`, `candidate_selection.correct_reward_threshold`, and `candidate_selection.keep_per_uid`. Helper name is consistently `select_short_correct_candidates`.
