# Length-aware OPD Strong-to-Weak Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a gated length-aware penalty to original OPD strong-to-weak training so long incorrect or teacher-rejected student rollouts are discouraged while long correct teacher-accepted rollouts are protected.

**Architecture:** Add small, testable helper functions in `dp_actor.py` to precompute per-sequence length penalties at actor mini-batch scope before actor micro-batch splitting. The existing OPD fallback path subtracts the precomputed scaled penalty from reverse-KL advantages. A new run script enables the feature for the raw strong-to-weak Table 2-style setting without enabling FiRe-OPD entropy-aware loss.

**Tech Stack:** Python, PyTorch tensors, verl `DataProto`/actor training loop, pytest, bash training scripts.

---

## File Structure

- Modify `verl/verl/workers/config/actor.py`
  - Add length-aware OPD config fields to `PolicyLossConfig`.
- Modify `verl/verl/workers/actor/dp_actor.py`
  - Add pure helper functions for length-penalty computation and metrics.
  - Retain reward tensors when length-aware OPD is enabled.
  - Precompute penalties at actor mini-batch scope before micro-batch splitting.
  - Subtract the precomputed penalty from OPD reverse-KL advantages.
- Create `verl/tests/workers/actor/test_length_aware_opd.py`
  - Unit tests for helper behavior, gate behavior, mini-batch scope behavior, and aggregation-calibration assumptions.
- Modify `verl/tests/workers/config/test_actor_config_on_cpu.py`
  - Add config field/default coverage.
- Create `verl/examples/fire_opd/run_opd_strong_to_weak_lengthaware_teacher30b.sh`
  - New experiment launcher derived from `run_opd_strong_to_weak_raw_teacher30b.sh`.

---

### Task 1: Add failing tests for length-aware OPD helper behavior

**Files:**
- Create: `verl/tests/workers/actor/test_length_aware_opd.py`

- [ ] **Step 1: Write the failing helper tests**

Create `verl/tests/workers/actor/test_length_aware_opd.py` with this content:

```python
from types import SimpleNamespace

import pytest
import torch

from verl import DataProto
from verl.workers.actor.dp_actor import (
    _add_length_aware_opd_tensors,
    _compute_length_aware_opd_tensors,
)


def _policy_loss_config(**overrides):
    values = {
        "length_aware_opd": True,
        "length_penalty_coef": 0.02,
        "length_penalty_type": "log_batch_median",
        "length_penalty_gate": "incorrect_or_low_teacher",
        "length_correct_reward_threshold": 0.5,
        "length_teacher_reject_percentile": 20.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_length_penalty_penalizes_long_incorrect_response_only():
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
    token_level_scores[0, 3] = 1.0
    token_level_scores[1, 7] = 0.0
    token_level_scores[2, 1] = 0.0

    tensors, metrics = _compute_length_aware_opd_tensors(
        response_mask=response_mask,
        ref_log_prob=ref_log_prob,
        token_level_scores=token_level_scores,
        policy_loss_config=_policy_loss_config(length_teacher_reject_percentile=0.0),
    )

    expected_base_penalty = torch.log(torch.tensor(8.0 / 4.0))
    assert torch.isclose(tensors["length_aware_opd_base_penalty"][1], expected_base_penalty)
    assert torch.isclose(tensors["length_aware_opd_applied_penalty"][1], expected_base_penalty)
    assert torch.isclose(tensors["length_aware_opd_penalty"][1], expected_base_penalty * 0.02)
    assert tensors["length_aware_opd_applied_penalty"][0].item() == 0.0
    assert tensors["length_aware_opd_applied_penalty"][2].item() == 0.0
    assert metrics["length_aware_opd/median_response_len"] == 4.0
    assert metrics["length_aware_opd/coef"] == 0.02


def test_length_penalty_skips_long_correct_teacher_accepted_response():
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
    token_level_scores[0, 3] = 1.0
    token_level_scores[1, 7] = 1.0
    token_level_scores[2, 1] = 0.0

    tensors, _ = _compute_length_aware_opd_tensors(
        response_mask=response_mask,
        ref_log_prob=ref_log_prob,
        token_level_scores=token_level_scores,
        policy_loss_config=_policy_loss_config(length_teacher_reject_percentile=0.0),
    )

    assert tensors["length_aware_opd_base_penalty"][1].item() > 0.0
    assert tensors["length_aware_opd_applied_penalty"][1].item() == 0.0
    assert tensors["length_aware_opd_penalty"][1].item() == 0.0
    assert tensors["length_aware_opd_correct_mask"][1].item() == 1.0
    assert tensors["length_aware_opd_teacher_reject"][1].item() == 0.0


def test_length_penalty_penalizes_long_correct_low_teacher_response():
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    ref_log_prob = torch.tensor(
        [
            [-0.2, -0.2, -0.2, -0.2, 0, 0, 0, 0],
            [-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],
            [-0.1, -0.1, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    token_level_scores = torch.zeros_like(response_mask)
    token_level_scores[0, 3] = 1.0
    token_level_scores[1, 7] = 1.0
    token_level_scores[2, 1] = 1.0

    tensors, metrics = _compute_length_aware_opd_tensors(
        response_mask=response_mask,
        ref_log_prob=ref_log_prob,
        token_level_scores=token_level_scores,
        policy_loss_config=_policy_loss_config(length_teacher_reject_percentile=50.0),
    )

    assert tensors["length_aware_opd_teacher_reject"][1].item() == 1.0
    assert tensors["length_aware_opd_applied_penalty"][1].item() > 0.0
    assert metrics["length_aware_opd/teacher_reject_ratio"] > 0.0


def test_length_penalty_rejects_invalid_type_and_gate():
    response_mask = torch.ones((2, 4), dtype=torch.float32)
    ref_log_prob = torch.full_like(response_mask, -0.2)
    token_level_scores = torch.zeros_like(response_mask)

    with pytest.raises(ValueError, match="Invalid length_penalty_type"):
        _compute_length_aware_opd_tensors(
            response_mask=response_mask,
            ref_log_prob=ref_log_prob,
            token_level_scores=token_level_scores,
            policy_loss_config=_policy_loss_config(length_penalty_type="linear"),
        )

    with pytest.raises(ValueError, match="Invalid length_penalty_gate"):
        _compute_length_aware_opd_tensors(
            response_mask=response_mask,
            ref_log_prob=ref_log_prob,
            token_level_scores=token_level_scores,
            policy_loss_config=_policy_loss_config(length_penalty_gate="unknown"),
        )


def test_length_penalty_requires_scores_for_incorrect_gate():
    response_mask = torch.ones((2, 4), dtype=torch.float32)
    ref_log_prob = torch.full_like(response_mask, -0.2)

    with pytest.raises(ValueError, match="token_level_scores is required"):
        _compute_length_aware_opd_tensors(
            response_mask=response_mask,
            ref_log_prob=ref_log_prob,
            token_level_scores=None,
            policy_loss_config=_policy_loss_config(length_penalty_gate="incorrect_or_low_teacher"),
        )


def test_minibatch_precompute_survives_single_sequence_microbatch_split():
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
    token_level_scores[1, 7] = 0.0
    batch = DataProto.from_dict(
        tensors={
            "response_mask": response_mask,
            "ref_log_prob": ref_log_prob,
            "token_level_scores": token_level_scores,
        }
    )

    metrics = _add_length_aware_opd_tensors(
        mini_batch=batch,
        policy_loss_config=_policy_loss_config(length_teacher_reject_percentile=0.0),
    )
    micro_batches = batch.split(1)

    assert metrics["length_aware_opd/median_response_len"] == 4.0
    assert micro_batches[1].batch["length_aware_opd_applied_penalty"].item() > 0.0
    assert micro_batches[1].batch["length_aware_opd_penalty"].item() > 0.0
```

- [ ] **Step 2: Run the failing helper tests**

Run:

```bash
cd /home/mchen/FiRe-OPD
PYTHONPATH=/home/mchen/FiRe-OPD/verl pytest verl/tests/workers/actor/test_length_aware_opd.py -q
```

Expected: FAIL during import with an error like:

```text
ImportError: cannot import name '_compute_length_aware_opd_tensors'
```

- [ ] **Step 3: Commit the failing tests**

```bash
git add verl/tests/workers/actor/test_length_aware_opd.py
git commit -m "test: add length-aware OPD helper tests"
```

---

### Task 2: Implement pure length-aware OPD helper functions

**Files:**
- Modify: `verl/verl/workers/actor/dp_actor.py`
- Test: `verl/tests/workers/actor/test_length_aware_opd.py`

- [ ] **Step 1: Add helper functions near the top of `dp_actor.py`**

Insert the following code after the logger setup and before `class DataParallelPPOActor`:

```python
_LENGTH_PENALTY_TYPES = {"log_batch_median"}
_LENGTH_PENALTY_GATES = {"incorrect", "low_teacher", "incorrect_or_low_teacher"}


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
    """Compute mini-batch-scoped gated length penalties for original OPD.

    Returns tensors with shape `(batch_size,)`. The returned
    `length_aware_opd_penalty` is already scaled by `length_penalty_coef` and is
    ready to subtract from token-level advantages after unsqueezing.
    """
    penalty_type = _policy_loss_get(policy_loss_config, "length_penalty_type", "log_batch_median")
    if penalty_type not in _LENGTH_PENALTY_TYPES:
        raise ValueError(
            f"Invalid length_penalty_type: {penalty_type}. "
            f"Supported values: {sorted(_LENGTH_PENALTY_TYPES)}"
        )

    penalty_gate = _policy_loss_get(policy_loss_config, "length_penalty_gate", "incorrect_or_low_teacher")
    if penalty_gate not in _LENGTH_PENALTY_GATES:
        raise ValueError(
            f"Invalid length_penalty_gate: {penalty_gate}. "
            f"Supported values: {sorted(_LENGTH_PENALTY_GATES)}"
        )

    needs_scores = penalty_gate in {"incorrect", "incorrect_or_low_teacher"}
    if needs_scores and token_level_scores is None:
        raise ValueError("token_level_scores is required when length_penalty_gate uses correctness")

    response_mask = response_mask.float()
    response_len = response_mask.sum(dim=-1).clamp(min=1.0)
    reference_len = response_len.median().clamp(min=1.0)
    base_penalty = torch.log(response_len / reference_len).clamp(min=0.0)

    normalized_teacher_logprob = (ref_log_prob * response_mask).sum(dim=-1) / response_len
    teacher_reject_percentile = float(
        _policy_loss_get(policy_loss_config, "length_teacher_reject_percentile", 20.0)
    )
    if teacher_reject_percentile < 0.0 or teacher_reject_percentile > 100.0:
        raise ValueError("length_teacher_reject_percentile must be between 0.0 and 100.0")

    if teacher_reject_percentile == 0.0:
        teacher_reject = torch.zeros_like(response_len, dtype=torch.bool)
        teacher_threshold = torch.tensor(float("nan"), device=response_mask.device)
    else:
        teacher_threshold = torch.quantile(
            normalized_teacher_logprob.float(), teacher_reject_percentile / 100.0
        )
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
        "length_aware_opd_reference_len": torch.full_like(response_len, reference_len.detach()),
        "length_aware_opd_teacher_threshold": torch.full_like(response_len, teacher_threshold.detach()),
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
```

- [ ] **Step 2: Run helper tests**

Run:

```bash
cd /home/mchen/FiRe-OPD
PYTHONPATH=/home/mchen/FiRe-OPD/verl pytest verl/tests/workers/actor/test_length_aware_opd.py -q
```

Expected: PASS.

- [ ] **Step 3: Commit helper implementation**

```bash
git add verl/verl/workers/actor/dp_actor.py
git commit -m "feat: add length-aware OPD penalty helpers"
```

---

### Task 3: Add config fields and config tests

**Files:**
- Modify: `verl/verl/workers/config/actor.py`
- Modify: `verl/tests/workers/config/test_actor_config_on_cpu.py`

- [ ] **Step 1: Add config test coverage**

Append this test method inside `TestActorConfig` in `verl/tests/workers/config/test_actor_config_on_cpu.py`:

```python
    def test_policy_loss_length_aware_opd_defaults_and_overrides(self):
        """Test length-aware OPD fields in PolicyLossConfig."""
        config_dict = {
            "_target_": "verl.workers.config.ActorConfig",
            "strategy": "fsdp",
            "ppo_mini_batch_size": 256,
            "ppo_micro_batch_size_per_gpu": 1,
            "policy_loss": {
                "length_aware_opd": True,
                "length_penalty_coef": 0.02,
                "length_penalty_type": "log_batch_median",
                "length_penalty_gate": "incorrect_or_low_teacher",
                "length_correct_reward_threshold": 0.5,
                "length_teacher_reject_percentile": 20.0,
            },
            "optim": {
                "_target_": "verl.workers.config.OptimizerConfig",
                "lr": 0.1,
            },
        }
        config = omega_conf_to_dataclass(config_dict)

        self.assertTrue(config.policy_loss.length_aware_opd)
        self.assertEqual(config.policy_loss.length_penalty_coef, 0.02)
        self.assertEqual(config.policy_loss.length_penalty_type, "log_batch_median")
        self.assertEqual(config.policy_loss.length_penalty_gate, "incorrect_or_low_teacher")
        self.assertEqual(config.policy_loss.length_correct_reward_threshold, 0.5)
        self.assertEqual(config.policy_loss.length_teacher_reject_percentile, 20.0)

        default_config = ActorConfig(
            strategy="fsdp",
            ppo_micro_batch_size_per_gpu=1,
            optim=OptimizerConfig(lr=0.1),
        )
        self.assertFalse(default_config.policy_loss.length_aware_opd)
        self.assertEqual(default_config.policy_loss.length_penalty_coef, 0.0)
        self.assertEqual(default_config.policy_loss.length_penalty_type, "log_batch_median")
        self.assertEqual(default_config.policy_loss.length_penalty_gate, "incorrect_or_low_teacher")
        self.assertEqual(default_config.policy_loss.length_correct_reward_threshold, 0.5)
        self.assertEqual(default_config.policy_loss.length_teacher_reject_percentile, 20.0)
```

- [ ] **Step 2: Run the config test to verify it fails**

Run:

```bash
cd /home/mchen/FiRe-OPD
PYTHONPATH=/home/mchen/FiRe-OPD/verl pytest verl/tests/workers/config/test_actor_config_on_cpu.py::TestActorConfig::test_policy_loss_length_aware_opd_defaults_and_overrides -q
```

Expected: FAIL with an attribute error or config conversion error for `length_aware_opd`.

- [ ] **Step 3: Add config fields**

In `verl/verl/workers/config/actor.py`, add these fields to `PolicyLossConfig` immediately after `only_reverse_kl_advantages`:

```python
    # TokenSqueeze-inspired length-aware OPD config
    length_aware_opd: bool = False
    length_penalty_coef: float = 0.0
    length_penalty_type: str = "log_batch_median"
    length_penalty_gate: str = "incorrect_or_low_teacher"
    length_correct_reward_threshold: float = 0.5
    length_teacher_reject_percentile: float = 20.0
```

- [ ] **Step 4: Run config tests**

Run:

```bash
cd /home/mchen/FiRe-OPD
PYTHONPATH=/home/mchen/FiRe-OPD/verl pytest verl/tests/workers/config/test_actor_config_on_cpu.py::TestActorConfig::test_policy_loss_length_aware_opd_defaults_and_overrides -q
```

Expected: PASS.

- [ ] **Step 5: Run helper tests after config change**

Run:

```bash
cd /home/mchen/FiRe-OPD
PYTHONPATH=/home/mchen/FiRe-OPD/verl pytest verl/tests/workers/actor/test_length_aware_opd.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit config changes**

```bash
git add verl/verl/workers/config/actor.py verl/tests/workers/config/test_actor_config_on_cpu.py
git commit -m "feat: add length-aware OPD policy config"
```

---

### Task 4: Integrate length-aware penalty into original OPD actor update

**Files:**
- Modify: `verl/verl/workers/actor/dp_actor.py`
- Test: `verl/tests/workers/actor/test_length_aware_opd.py`

- [ ] **Step 1: Retain reward tensors when length-aware OPD is enabled**

In `DataParallelPPOActor.update_policy`, after this block:

```python
        if self.config.policy_loss.only_reverse_kl_advantages and "ref_log_prob" in data.batch.keys():
            if "ref_log_prob" not in select_keys:
                select_keys.append("ref_log_prob")
```

insert:

```python
        length_aware_opd = getattr(self.config.policy_loss, "length_aware_opd", False)
        if length_aware_opd:
            if "token_level_scores" in data.batch.keys():
                select_keys.append("token_level_scores")
            elif "token_level_rewards" in data.batch.keys():
                select_keys.append("token_level_rewards")
```

Then remove or avoid the later local-only assignment if you add another `length_aware_opd` variable below. The function should define `length_aware_opd` once before `data.select(...)` and reuse it later.

- [ ] **Step 2: Precompute penalties at mini-batch scope**

Inside the `for batch_idx, mini_batch in enumerate(mini_batches):` loop, immediately before this existing line:

```python
                if self.config.use_dynamic_bsz:
```

insert:

```python
                if length_aware_opd:
                    length_aware_metrics = _add_length_aware_opd_tensors(
                        mini_batch=mini_batch,
                        policy_loss_config=self.config.policy_loss,
                    )
                    append_to_dict(metrics, length_aware_metrics)
```

This placement ensures the median uses the actor mini-batch rather than a one-sample actor micro-batch.

- [ ] **Step 3: Subtract precomputed penalty from OPD advantages**

In the non-entropy-aware OPD fallback path, replace:

```python
                        if self.config.policy_loss.only_reverse_kl_advantages and "ref_log_prob" in model_inputs:
                            advantages = -(old_log_prob - model_inputs["ref_log_prob"])
```

with:

```python
                        if self.config.policy_loss.only_reverse_kl_advantages and "ref_log_prob" in model_inputs:
                            advantages = -(old_log_prob - model_inputs["ref_log_prob"])
                            if length_aware_opd:
                                if "length_aware_opd_penalty" not in model_inputs:
                                    raise ValueError("length_aware_opd_penalty missing from actor micro-batch")
                                advantages = advantages - model_inputs["length_aware_opd_penalty"].unsqueeze(-1)
```

- [ ] **Step 4: Run focused actor tests**

Run:

```bash
cd /home/mchen/FiRe-OPD
PYTHONPATH=/home/mchen/FiRe-OPD/verl pytest verl/tests/workers/actor/test_length_aware_opd.py -q
```

Expected: PASS.

- [ ] **Step 5: Run config tests**

Run:

```bash
cd /home/mchen/FiRe-OPD
PYTHONPATH=/home/mchen/FiRe-OPD/verl pytest verl/tests/workers/config/test_actor_config_on_cpu.py::TestActorConfig::test_policy_loss_length_aware_opd_defaults_and_overrides -q
```

Expected: PASS.

- [ ] **Step 6: Commit actor integration**

```bash
git add verl/verl/workers/actor/dp_actor.py
git commit -m "feat: apply gated length penalty in OPD actor update"
```

---

### Task 5: Add strong-to-weak length-aware OPD run script

**Files:**
- Create: `verl/examples/fire_opd/run_opd_strong_to_weak_lengthaware_teacher30b.sh`

- [ ] **Step 1: Create the new script**

Create `verl/examples/fire_opd/run_opd_strong_to_weak_lengthaware_teacher30b.sh` with this content:

```bash
#!/bin/bash
# Original OPD strong-to-weak with a gated length-aware OPD penalty.
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
RAW_DATA_ROOT="${RAW_DATA_ROOT:-${REPO_DIR}/data/g-opd}"

LENGTH_PENALTY_COEF="${LENGTH_PENALTY_COEF:-0.02}"
LENGTH_PENALTY_GATE="${LENGTH_PENALTY_GATE:-incorrect_or_low_teacher}"
LENGTH_CORRECT_REWARD_THRESHOLD="${LENGTH_CORRECT_REWARD_THRESHOLD:-0.5}"
LENGTH_TEACHER_REJECT_PERCENTILE="${LENGTH_TEACHER_REJECT_PERCENTILE:-20.0}"

TRAIN_DATA="${TRAIN_DATA:-${RAW_DATA_ROOT}/DeepMath-103K/train_filtered_level6.parquet}"
VAL_DATA="${VAL_DATA:-['${RAW_DATA_ROOT}/AIME2024/test.parquet', '${RAW_DATA_ROOT}/AIME2025/test.parquet']}"

STUDENT_MODEL="${STUDENT_MODEL:-${REPO_DIR}/models/Qwen3-4B}"
TEACHER_MODEL="${TEACHER_MODEL:-${REPO_DIR}/models/Qwen3-30B-A3B-Instruct-2507}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-opd-strong-to-weak-lengthaware-lambda${LENGTH_PENALTY_COEF}-rawprompt-${N_GPUS_PER_NODE}gpu-tp${ROLLOUT_TP_SIZE}-refmb4-rollmb4}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_DIR}/checkpoints/${EXPERIMENT_NAME}}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"

echo "Using N_GPUS_PER_NODE=${N_GPUS_PER_NODE}, ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE}"
echo "Using LENGTH_PENALTY_COEF=${LENGTH_PENALTY_COEF}, LENGTH_PENALTY_GATE=${LENGTH_PENALTY_GATE}"
echo "Using LENGTH_CORRECT_REWARD_THRESHOLD=${LENGTH_CORRECT_REWARD_THRESHOLD}, LENGTH_TEACHER_REJECT_PERCENTILE=${LENGTH_TEACHER_REJECT_PERCENTILE}"

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
    actor_rollout_ref.rollout.n=1 \
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

- [ ] **Step 2: Make the script executable**

Run:

```bash
chmod +x /home/mchen/FiRe-OPD/verl/examples/fire_opd/run_opd_strong_to_weak_lengthaware_teacher30b.sh
```

- [ ] **Step 3: Validate script syntax**

Run:

```bash
bash -n /home/mchen/FiRe-OPD/verl/examples/fire_opd/run_opd_strong_to_weak_lengthaware_teacher30b.sh
```

Expected: no output and exit code 0.

- [ ] **Step 4: Confirm script does not enable FiRe-OPD entropy-aware mode**

Run:

```bash
if grep -q 'entropy_aware_distill=True' /home/mchen/FiRe-OPD/verl/examples/fire_opd/run_opd_strong_to_weak_lengthaware_teacher30b.sh; then
  echo 'unexpected FiRe-OPD entropy-aware flag' >&2
  exit 1
fi
grep -n 'length_aware_opd=True' /home/mchen/FiRe-OPD/verl/examples/fire_opd/run_opd_strong_to_weak_lengthaware_teacher30b.sh
```

Expected: one `length_aware_opd=True` line is printed; no error is printed.

- [ ] **Step 5: Commit the run script**

```bash
git add verl/examples/fire_opd/run_opd_strong_to_weak_lengthaware_teacher30b.sh
git commit -m "feat: add length-aware strong-to-weak OPD script"
```

---

### Task 6: Run verification suite before experiment launch

**Files:**
- Verify only; no file changes expected.

- [ ] **Step 1: Run length-aware helper tests**

Run:

```bash
cd /home/mchen/FiRe-OPD
PYTHONPATH=/home/mchen/FiRe-OPD/verl pytest verl/tests/workers/actor/test_length_aware_opd.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run actor config tests**

Run:

```bash
cd /home/mchen/FiRe-OPD
PYTHONPATH=/home/mchen/FiRe-OPD/verl pytest verl/tests/workers/config/test_actor_config_on_cpu.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Run existing ref input utility test to guard unrelated OPD prompt routing changes**

Run:

```bash
cd /home/mchen/FiRe-OPD
PYTHONPATH=/home/mchen/FiRe-OPD/verl pytest verl/tests/test_ref_input_utils.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Validate new script syntax again**

Run:

```bash
bash -n /home/mchen/FiRe-OPD/verl/examples/fire_opd/run_opd_strong_to_weak_lengthaware_teacher30b.sh
```

Expected: no output and exit code 0.

- [ ] **Step 5: Record verification status**

Run:

```bash
git status --short
```

Expected: no uncommitted changes from the length-aware OPD implementation files. Existing unrelated working-tree changes may still be present from prior experiments; do not stage them.

---

### Task 7: Prepare first training command and expected metrics

**Files:**
- Verify only; no file changes expected unless the user asks to launch training in this session.

- [ ] **Step 1: Print the exact command for the first training run**

Use this command for the first length-aware OPD run:

```bash
cd /home/mchen/FiRe-OPD
LENGTH_PENALTY_COEF=0.02 \
LENGTH_PENALTY_GATE=incorrect_or_low_teacher \
LENGTH_CORRECT_REWARD_THRESHOLD=0.5 \
LENGTH_TEACHER_REJECT_PERCENTILE=20.0 \
bash verl/examples/fire_opd/run_opd_strong_to_weak_lengthaware_teacher30b.sh
```

- [ ] **Step 2: During training, confirm length-aware metrics appear**

Look for metrics with these names in console or wandb logs:

```text
length_aware_opd/mean_response_len
length_aware_opd/median_response_len
length_aware_opd/mean_base_penalty
length_aware_opd/mean_applied_penalty
length_aware_opd/max_applied_penalty
length_aware_opd/penalty_gate_ratio
length_aware_opd/correct_skip_ratio
length_aware_opd/teacher_reject_ratio
length_aware_opd/coef
```

Expected: metrics appear after actor update steps. `length_aware_opd/coef` should be `0.02`.

- [ ] **Step 3: Check that the penalty is active but gated**

Expected early-run sanity ranges:

```text
length_aware_opd/mean_base_penalty >= 0.0
length_aware_opd/mean_applied_penalty >= 0.0
length_aware_opd/penalty_gate_ratio > 0.0
length_aware_opd/correct_skip_ratio >= 0.0
length_aware_opd/teacher_reject_ratio >= 0.0
```

If `mean_base_penalty` and `mean_applied_penalty` stay exactly `0.0` for multiple actor updates, stop and inspect whether penalty tensors are being computed before micro-batch splitting.

- [ ] **Step 4: After checkpoint save, merge and evaluate using existing project workflow**

Use the existing model merger pattern from prior notes, replacing paths with the new experiment directory:

```bash
cd /home/mchen/FiRe-OPD
export PYTHONPATH=/home/mchen/FiRe-OPD/verl:${PYTHONPATH:-}
/home/mchen/miniconda3/envs/verl/bin/python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /home/mchen/FiRe-OPD/checkpoints/opd-strong-to-weak-lengthaware-lambda0.02-rawprompt-4gpu-tp4-refmb4-rollmb4/global_step_50/actor \
  --target_dir /home/mchen/FiRe-OPD/checkpoints/opd-strong-to-weak-lengthaware-lambda0.02-rawprompt-4gpu-tp4-refmb4-rollmb4/global_step_50_hf
```

For a quick check, evaluate AIME24/AIME25 first using the existing math eval scripts or a copied script with `MODEL_PATH` set to the merged checkpoint. Compare `Accuracy`, `pass@k`, and `avg_length` against `opd-strong-to-weak-rawprompt-4gpu-tp4-refmb4-rollmb4`.

---

## Self-Review Notes

Spec coverage:

- Original OPD only: covered by Task 4 and Task 5; no FiRe-OPD entropy-aware flag is enabled.
- Gated penalty by correctness and teacher confidence: covered by Task 1 and Task 2.
- Mini-batch-scope median before micro-batch split: covered by Task 1 and Task 4.
- Loss aggregation and λ calibration: covered by Task 1 test, Task 5 script settings, and Task 7 metric checks.
- New script and verification commands: covered by Task 5 and Task 6.

Completeness scan: this plan contains no incomplete implementation steps. Every code-changing step includes exact code or exact replacement instructions.

Type consistency: helper names, tensor keys, config field names, script flags, and metric names are consistent across tasks.
