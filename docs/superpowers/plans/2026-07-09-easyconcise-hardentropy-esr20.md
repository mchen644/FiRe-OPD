# Easy-Concise Hard-Entropy ESR20 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a difficulty-aware OPD variant that compresses easy samples with a concise teacher prompt while encouraging hard-sample exploration through a current-policy entropy bonus, without widening hard ESR supervision.

**Architecture:** Reuse the existing difficulty-aware prompt routing and hard entropy loss. Change only old-log-prob entropy gating so hard entropy does not materialize old-policy entropy, then add a shell wrapper that configures easy=concise+ESR20, hard=normal+ESR20+entropy, middle=budget+ESR20.

**Tech Stack:** Bash training wrappers, veRL PPO trainer, PyTorch actor loss, pytest.

## Global Constraints

- Student rollout remains normal/raw prompt.
- Easy samples use concise teacher prompt and ESR beta 0.20.
- Hard samples use normal teacher prompt and ESR beta remains base 0.20.
- Hard exploration uses `algorithm.difficulty_aware_opd.hard_entropy_coef`, initially 0.003.
- Disable rethinking probe, validation before train, and test frequency for full runs.
- Preserve entropy gating: old-log-prob entropy is computed only for true old-log-prob entropy consumers.

---

### Task 1: Prevent hard entropy from triggering old-log-prob entropy

**Files:**
- Modify: `verl/tests/trainer/ppo/test_difficulty_aware_opd.py`
- Modify: `verl/verl/trainer/ppo/ray_trainer.py`

**Interfaces:**
- Consumes: `_old_log_prob_entropy_required(config) -> bool`
- Produces: `_old_log_prob_entropy_required` returns `False` when only `algorithm.difficulty_aware_opd.hard_entropy_coef != 0`, because the hard entropy bonus uses current actor-update entropy.

- [ ] **Step 1: Write the failing test**

Change `test_old_log_prob_entropy_is_required_for_probe_or_entropy_distill` so it asserts probe, entropy-aware distill, and global actor entropy still require old-log-prob entropy. Add a separate test where difficulty-aware OPD is enabled with `hard_entropy_coef=0.003` and assert `_old_log_prob_entropy_required(config) is False`.

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=verl pytest -q verl/tests/trainer/ppo/test_difficulty_aware_opd.py::test_old_log_prob_entropy_is_not_required_for_difficulty_aware_hard_entropy`

Expected: FAIL because current `_old_log_prob_entropy_required` returns `True` for difficulty-aware hard entropy.

- [ ] **Step 3: Write minimal implementation**

In `verl/verl/trainer/ppo/ray_trainer.py`, remove the branch that returns `True` solely because `algorithm.difficulty_aware_opd.enabled` and `hard_entropy_coef != 0`. Keep returns for `actor.entropy_coeff != 0`, `policy_loss.entropy_aware_distill=True`, and `rethinking_opd_probe.enabled=True`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=verl pytest -q verl/tests/trainer/ppo/test_difficulty_aware_opd.py::test_old_log_prob_entropy_is_not_required_for_difficulty_aware_hard_entropy verl/tests/trainer/ppo/test_difficulty_aware_opd.py::test_old_log_prob_entropy_is_required_for_probe_or_entropy_distill`

Expected: PASS.

---

### Task 2: Add training wrapper for easy-concise hard-normal entropy ESR20

**Files:**
- Create: `run_train_tale_budget_rolloutlen_hardtrunc_easyconcise_hardnormal_entropy_opd.sh`

**Interfaces:**
- Consumes: existing `run_train_tale_budget_rolloutlen_hardtrunc_hardnormal_opd.sh` wrapper.
- Produces: a runnable wrapper setting easy concise routing, hard normal routing, no hard ESR override, hard entropy coefficient, save frequency, and probe off.

- [ ] **Step 1: Write wrapper**

Create a shell script that sets:

```bash
TALE_ESR_BETA=0.20
DA_EASY_PROMPT_THRESHOLD=0.7
DA_EASY_PROMPT_STYLE=concise
DA_MIN_EASY_ESR_BETA=0.20
DA_EASY_ESR_DELTA=0.0
DA_DEFAULT_PROMPT_STYLE=budget
DA_HARD_PROMPT_THRESHOLD=0.5
DA_HARD_PROMPT_STYLE=normal
DA_HARD_ENTROPY_COEF=0.003
TRAINER_SAVE_FREQ=20
EXPERIMENT_NAME=opd-budget20-easyconcise-hardnormal-entropy003-esr20-noprobe
```

Call `run_train_tale_budget_rolloutlen_hardtrunc_hardnormal_opd.sh` with `algorithm.rethinking_opd_probe.enabled=False`, `trainer.save_freq=${TRAINER_SAVE_FREQ}`, `trainer.val_before_train=False`, and `trainer.test_freq=-1`.

- [ ] **Step 2: Syntax check**

Run: `bash -n run_train_tale_budget_rolloutlen_hardtrunc_easyconcise_hardnormal_entropy_opd.sh`

Expected: no output, exit 0.

---

### Task 3: Verification and launch

**Files:**
- Verify: `verl/tests/trainer/ppo/test_difficulty_aware_opd.py`
- Verify: `run_train_tale_budget_rolloutlen_hardtrunc_easyconcise_hardnormal_entropy_opd.sh`

**Interfaces:**
- Consumes: Task 1 and Task 2 deliverables.
- Produces: launch command and log path for the full step50 experiment.

- [ ] **Step 1: Run focused tests**

Run: `PYTHONPATH=verl pytest -q verl/tests/trainer/ppo/test_difficulty_aware_opd.py verl/tests/workers/test_fsdp_workers.py::test_log_prob_entropy_meta_flag_defaults_true_and_can_disable`

Expected: PASS.

- [ ] **Step 2: Run py_compile and bash syntax checks**

Run: `PYTHONPATH=verl python -m py_compile verl/verl/trainer/ppo/ray_trainer.py verl/verl/trainer/ppo/difficulty_aware_opd.py verl/verl/workers/actor/dp_actor.py verl/verl/workers/fsdp_workers.py`

Run: `bash -n run_train_tale_budget_rolloutlen_hardtrunc_easyconcise_hardnormal_entropy_opd.sh run_train_tale_budget_rolloutlen_hardtrunc_hardnormal_opd.sh`

Expected: both commands exit 0.

- [ ] **Step 3: Launch full run**

Use `srun --gres=gpu:4 --cpus-per-task=16 bash run_train_tale_budget_rolloutlen_hardtrunc_easyconcise_hardnormal_entropy_opd.sh ...` with explicit experiment name `opd-budget20-easyconcise-hardnormal-entropy003-esr20-step50-noprobe-compileoff-<timestamp>` and disabled compile/probe settings matching previous ESR50 runs.

Expected first-step metrics: `concise_prompt_ratio > 0`, `normal_prompt_ratio > 0`, `budget_prompt_ratio > 0`, `esr_beta_max = 0.2`, `actor/entropy_computed = 0.0` during old-log-prob metrics, and `difficulty_aware_opd/hard_entropy_loss < 0` during actor update.
