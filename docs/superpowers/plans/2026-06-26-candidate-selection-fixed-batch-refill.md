# Candidate Selection Fixed-batch Refill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make OPD candidate selection with physical drops produce fixed-size actor-update batches by accumulating selected rows across generation batches.

**Architecture:** Add a small accumulator helper beside `select_short_correct_candidates()` because it operates on selected `DataProto` objects and is easy to unit-test there. Integrate it in `RayPPOTrainer.fit` after candidate selection and before rollout correction/advantage/update; skipped accumulation batches do not increment `global_steps`.

**Tech Stack:** Python, PyTorch, verl `DataProto`, pytest.

---

## File Structure

- Modify `verl/verl/trainer/ppo/candidate_selection.py`
  - Add `accumulate_selected_candidates()` helper.
- Modify `verl/tests/trainer/ppo/test_candidate_selection.py`
  - Add RED tests for fixed-size accumulation/truncation.
- Modify `verl/verl/trainer/ppo/ray_trainer.py`
  - Import the helper and use it after `select_short_correct_candidates()`.
  - Recompute `global_token_num` after final fixed-size selection.

---

### Task 1: Add accumulator unit test

**Files:**
- Modify: `verl/tests/trainer/ppo/test_candidate_selection.py`

- [ ] **Step 1: Add the failing test**

Append this test to `verl/tests/trainer/ppo/test_candidate_selection.py`:

```python
def test_accumulate_selected_candidates_refills_to_fixed_size_and_truncates_extra():
    first = DataProto.from_dict(
        tensors={"input_ids": torch.arange(3, dtype=torch.long).view(3, 1)},
        non_tensors={"uid": np.array(["u0", "u1", "u2"], dtype=object)},
    )
    second = DataProto.from_dict(
        tensors={"input_ids": torch.arange(3, 8, dtype=torch.long).view(5, 1)},
        non_tensors={"uid": np.array(["u3", "u4", "u5", "u6", "u7"], dtype=object)},
    )

    ready, pending, metrics = accumulate_selected_candidates(
        pending_batch=None,
        selected_batch=first,
        target_size=6,
        accumulated_batches=1,
    )

    assert ready is None
    assert len(pending) == 3
    assert metrics["candidate_selection/accumulated_batches"] == 1.0
    assert metrics["candidate_selection/accumulated_selected_size"] == 3.0
    assert metrics["candidate_selection/final_selected_size"] == 0.0
    assert metrics["candidate_selection/truncated_size"] == 0.0

    ready, pending, metrics = accumulate_selected_candidates(
        pending_batch=pending,
        selected_batch=second,
        target_size=6,
        accumulated_batches=2,
    )

    assert pending is None
    assert ready.batch["input_ids"].squeeze(-1).tolist() == [0, 1, 2, 3, 4, 5]
    assert ready.non_tensor_batch["uid"].tolist() == ["u0", "u1", "u2", "u3", "u4", "u5"]
    assert metrics["candidate_selection/accumulated_batches"] == 2.0
    assert metrics["candidate_selection/accumulated_selected_size"] == 8.0
    assert metrics["candidate_selection/final_selected_size"] == 6.0
    assert metrics["candidate_selection/truncated_size"] == 2.0
```

Also extend the import to:

```python
from verl.trainer.ppo.candidate_selection import accumulate_selected_candidates, select_short_correct_candidates
```

- [ ] **Step 2: Verify RED**

Run:

```bash
cd /home/mchen/FiRe-OPD
PYTHONPATH=/home/mchen/FiRe-OPD/verl /home/mchen/miniconda3/envs/verl/bin/python -m pytest \
  verl/tests/trainer/ppo/test_candidate_selection.py::test_accumulate_selected_candidates_refills_to_fixed_size_and_truncates_extra -q
```

Expected: FAIL with `ImportError` or `NameError` for `accumulate_selected_candidates`.

---

### Task 2: Implement accumulator helper

**Files:**
- Modify: `verl/verl/trainer/ppo/candidate_selection.py`
- Test: `verl/tests/trainer/ppo/test_candidate_selection.py`

- [ ] **Step 1: Add helper implementation**

Add this function before `select_short_correct_candidates()`:

```python
def accumulate_selected_candidates(
    pending_batch: DataProto | None,
    selected_batch: DataProto,
    target_size: int,
    accumulated_batches: int,
) -> tuple[DataProto | None, DataProto | None, dict[str, float]]:
    if target_size <= 0:
        raise ValueError("candidate selection accumulation target_size must be positive")

    accumulated = selected_batch if pending_batch is None else DataProto.concat([pending_batch, selected_batch])
    accumulated_size = len(accumulated)
    metrics = {
        "candidate_selection/accumulated_batches": float(accumulated_batches),
        "candidate_selection/accumulated_selected_size": float(accumulated_size),
        "candidate_selection/final_selected_size": 0.0,
        "candidate_selection/truncated_size": 0.0,
    }

    if accumulated_size < target_size:
        return None, accumulated, metrics

    ready = accumulated[:target_size]
    metrics["candidate_selection/final_selected_size"] = float(len(ready))
    metrics["candidate_selection/truncated_size"] = float(accumulated_size - len(ready))
    return ready, None, metrics
```

- [ ] **Step 2: Verify GREEN**

Run the same focused test. Expected: PASS.

- [ ] **Step 3: Run candidate-selection test file**

Run:

```bash
cd /home/mchen/FiRe-OPD
PYTHONPATH=/home/mchen/FiRe-OPD/verl /home/mchen/miniconda3/envs/verl/bin/python -m pytest \
  verl/tests/trainer/ppo/test_candidate_selection.py -q
```

Expected: all tests PASS.

---

### Task 3: Integrate fixed-size accumulation in trainer

**Files:**
- Modify: `verl/verl/trainer/ppo/ray_trainer.py`

- [ ] **Step 1: Import helper**

Change the import to:

```python
from verl.trainer.ppo.candidate_selection import accumulate_selected_candidates, select_short_correct_candidates
```

- [ ] **Step 2: Add pending state before the epoch loop**

Before `for epoch in range(self.config.trainer.total_epochs):`, add:

```python
        pending_candidate_selection_batch = None
        pending_candidate_selection_batches = 0
```

- [ ] **Step 3: Accumulate after candidate selection**

After `select_short_correct_candidates(...)`, compute:

```python
                            pending_candidate_selection_batches += 1
                            target_selected_size = int(self.config.data.train_batch_size) * int(
                                candidate_selection_config.get("keep_per_uid", 1)
                            )
                            batch, pending_candidate_selection_batch, accumulation_metrics = accumulate_selected_candidates(
                                pending_batch=pending_candidate_selection_batch,
                                selected_batch=batch,
                                target_size=target_selected_size,
                                accumulated_batches=pending_candidate_selection_batches,
                            )
                            metrics.update(accumulation_metrics)
                            if batch is None:
                                wait_for_candidate_selection_refill = True
                            else:
                                pending_candidate_selection_batches = 0
                                batch.meta_info["global_token_num"] = torch.sum(
                                    batch.batch["attention_mask"], dim=-1
                                ).tolist()
```

Initialize `wait_for_candidate_selection_refill = False` before the `with marked_timer("adv", ...)` block.

- [ ] **Step 4: Skip update when not enough selected rows**

After the `adv` block, before critic/actor update, add:

```python
                    if wait_for_candidate_selection_refill:
                        continue
```

This leaves `global_steps` unchanged and proceeds to the next dataloader batch.

- [ ] **Step 5: Verify syntax/imports**

Run:

```bash
cd /home/mchen/FiRe-OPD
PYTHONPATH=/home/mchen/FiRe-OPD/verl /home/mchen/miniconda3/envs/verl/bin/python -m py_compile \
  verl/verl/trainer/ppo/candidate_selection.py \
  verl/verl/trainer/ppo/ray_trainer.py
```

Expected: command exits 0.

---

## Self-review

- Spec coverage: Task 1/2 covers fixed-size accumulation; Task 3 integrates before rollout correction/advantage/update and recomputes `global_token_num`.
- Placeholder scan: no TBD/TODO placeholders.
- Type consistency: helper returns `(ready_batch_or_none, pending_batch_or_none, metrics)` and trainer uses the same order.
