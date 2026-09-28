# Candidate Selection Loss-mask Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace slow physical dropping/refill for quality-gated OPD candidate selection with fixed-size selection plus actor loss masking.

**Architecture:** `select_short_correct_candidates()` will always keep one selected row per uid for `quality_gated_correct_compression`; rows that were previously dropped get `candidate_selection_loss_mask=0`. `DataParallelPPOActor.update_policy()` will include that tensor and multiply it into `response_mask`, making masked rows contribute zero policy/KL/length loss while preserving fixed batch size and distributed divisibility.

**Tech Stack:** Python, PyTorch, verl `DataProto`, pytest.

---

## File Structure

- Modify `verl/tests/trainer/ppo/test_candidate_selection.py`
  - Add/adjust tests proving rejected no-correct groups are retained with zero loss mask.
- Modify `verl/verl/trainer/ppo/candidate_selection.py`
  - Add `candidate_selection_loss_mask` to selected output.
  - Stop physical dropping for `quality_gated_correct_compression` when `drop_rejected_no_correct=True`; mark those rows as masked instead.
- Modify `verl/tests/workers/actor/test_length_aware_opd.py`
  - Add a focused test that multiplying the mask into `response_mask` zeroes masked rows.
- Modify `verl/verl/workers/actor/dp_actor.py`
  - Include `candidate_selection_loss_mask` in actor selected keys.
  - Multiply `response_mask` by the mask before policy/KL/length loss.
- Modify `verl/verl/trainer/ppo/ray_trainer.py`
  - Disable/remove fixed-batch refill path because selection is fixed-size again.

---

### Task 1: Candidate-selection RED/GREEN

- [ ] Write failing tests expecting `quality_gated_correct_compression` to retain uid-c with `candidate_selection_loss_mask=0`.
- [ ] Run the focused tests and observe failure.
- [ ] Implement fixed-size masked selection.
- [ ] Run candidate-selection tests and observe pass.

### Task 2: Actor mask RED/GREEN

- [ ] Write a focused test for applying `candidate_selection_loss_mask` to `response_mask`.
- [ ] Run the focused test and observe failure.
- [ ] Implement a small helper in `dp_actor.py` and use it in `update_policy()`.
- [ ] Run actor/candidate tests and observe pass.

### Task 3: Trainer cleanup

- [ ] Remove refill accumulation from `ray_trainer.py` so every selected batch updates immediately.
- [ ] Run py_compile for changed trainer/selection/actor files.
- [ ] Run focused pytest files.

## Self-review

- No refill means no extra generation/ref pass per step.
- Masked rows keep fixed batch shape and zero actor loss.
- Existing metrics keep reporting rejected ratio; selected size stays equal to uid groups.
