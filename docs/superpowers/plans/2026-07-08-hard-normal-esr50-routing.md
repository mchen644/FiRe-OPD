# Hard-Normal ESR50 Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a diagnostic variant where hard samples use normal teacher prompts and 50% ESR supervision, while non-hard samples keep budget teacher prompts and 20% ESR supervision.

**Architecture:** Extend the existing pure difficulty-routing helper with an optional hard ESR override separate from prompt routing. Keep the default behavior unchanged unless `hard_esr_threshold` and `hard_esr_beta` are configured. Add a wrapper for the hard-normal-ESR50 experiment.

**Tech Stack:** Python 3.10, PyTorch, NumPy, pytest, bash/Hydra config overrides, veRL/FiRe-OPD trainer.

## Global Constraints

- Work in isolated worktree `/home/mchen/FiRe-OPD/.worktrees/hard-normal-budget20-routing`.
- Do not modify or delete WIP in `/home/mchen/FiRe-OPD` main checkout.
- Preserve hardtrunc semantics: `response_length/mean` is supervised/truncated training width; `tale_budget/response_length_mean` is original rollout length.
- Default behavior must remain compatible with existing difficulty-routed runs.
- New diagnostic defaults: non-hard `esr_beta=0.20`, hard `esr_beta=0.50`, `hard_prompt_threshold=0.5`, `hard_esr_threshold=0.5`, `hard_prompt_style=normal`, `hard_entropy_coef=0`.
- Disable rethinking probe and torch compile for the run.

---

### Task 1: Add hard ESR override to routing helper/config

**Files:**
- Modify: `verl/verl/trainer/config/algorithm.py`
- Modify: `verl/verl/trainer/config/ppo_trainer.yaml`
- Modify: `verl/verl/trainer/ppo/difficulty_aware_opd.py`
- Test: `verl/tests/trainer/ppo/test_difficulty_aware_opd.py`
- Test: `verl/tests/trainer/config/test_algo_config_on_cpu.py`

**Interfaces:**
- Consumes: `compute_two_signal_difficulty_routing(..., config, base_esr_beta)`.
- Produces: optional config fields `hard_esr_threshold: Optional[float] = None`, `hard_esr_beta: Optional[float] = None`.
- Behavior: compute existing easy ESR beta first; then if `hard_esr_threshold` and `hard_esr_beta` are set, replace ESR beta with `hard_esr_beta` for rows where `hard >= hard_esr_threshold`.

- [ ] **Step 1: Write failing test**

Add a test where a wrong low-confidence sample has `hard >= 0.5` and receives `esr_beta=0.5`, while non-hard samples remain at `0.2`. Assert prompt routing uses `normal` for hard and `budget` otherwise.

- [ ] **Step 2: Verify RED**

Run:

```bash
pytest verl/tests/trainer/ppo/test_difficulty_aware_opd.py::test_hard_esr_override_expands_supervision_for_hard_normal_samples -q
```

Expected: FAIL because `hard_esr_threshold` / `hard_esr_beta` are not implemented.

- [ ] **Step 3: Implement config fields and routing logic**

Add fields to `DifficultyAwareOpdConfig` and `ppo_trainer.yaml`. In `compute_two_signal_difficulty_routing`, after the current `esr_beta` calculation, apply:

```python
hard_esr_threshold = _config_get(config, "hard_esr_threshold", None)
hard_esr_beta = _config_get(config, "hard_esr_beta", None)
if hard_esr_threshold is not None and hard_esr_beta is not None:
    hard_esr_threshold = float(hard_esr_threshold)
    hard_esr_beta = float(hard_esr_beta)
    if hard_esr_beta <= 0.0:
        raise ValueError("hard_esr_beta must be positive")
    hard_mask = hard.to(dtype=torch.float64) >= hard_esr_threshold
    esr_beta = torch.where(hard_mask, torch.full_like(esr_beta, hard_esr_beta), esr_beta)
```

- [ ] **Step 4: Verify GREEN**

Run:

```bash
pytest verl/tests/trainer/ppo/test_difficulty_aware_opd.py::test_hard_esr_override_expands_supervision_for_hard_normal_samples verl/tests/trainer/config/test_algo_config_on_cpu.py -q
```

Expected: PASS.

---

### Task 2: Add hard-normal ESR50 run wrapper

**Files:**
- Create: `run_train_tale_budget_rolloutlen_hardtrunc_hardnormal_esr50_opd.sh`
- Test: shell syntax.

**Interfaces:**
- Consumes: `run_train_tale_budget_rolloutlen_hardtrunc_hardnormal_opd.sh`.
- Produces: wrapper that sets hard prompt and hard ESR threshold to 0.5.

- [ ] **Step 1: Write wrapper**

Create a wrapper that exports:

```bash
DA_HARD_PROMPT_THRESHOLD=0.5
DA_HARD_ESR_THRESHOLD=0.5
DA_HARD_ESR_BETA=0.50
DA_HARD_ENTROPY_COEF=0
TALE_ESR_BETA=0.20
```

Then calls `run_train_tale_budget_rolloutlen_hardtrunc_hardnormal_opd.sh` and forwards:

```bash
algorithm.difficulty_aware_opd.hard_esr_threshold=${DA_HARD_ESR_THRESHOLD}
algorithm.difficulty_aware_opd.hard_esr_beta=${DA_HARD_ESR_BETA}
```

- [ ] **Step 2: Verify shell syntax**

```bash
bash -n run_train_tale_budget_rolloutlen_hardtrunc_hardnormal_esr50_opd.sh
```

Expected: exit 0.

---

### Task 3: Regression verification and launch

**Files:**
- Test: `verl/tests/trainer/ppo/test_difficulty_aware_opd.py`
- Test: `verl/tests/trainer/config/test_algo_config_on_cpu.py`
- Test: `verl/tests/trainer/ppo/test_ref_input_utils.py`

**Interfaces:**
- Consumes: Tasks 1-2.
- Produces: verified branch ready for hard-normal ESR50 step50 run.

- [ ] **Step 1: Run targeted tests**

```bash
pytest verl/tests/trainer/ppo/test_difficulty_aware_opd.py verl/tests/trainer/config/test_algo_config_on_cpu.py verl/tests/trainer/ppo/test_ref_input_utils.py -q
```

Expected: pass.

- [ ] **Step 2: Run compile/syntax checks**

```bash
python -m py_compile verl/verl/trainer/ppo/difficulty_aware_opd.py verl/verl/trainer/config/algorithm.py verl/verl/trainer/ppo/ref_input_utils.py
bash -n run_train_tale_budget_rolloutlen_hardtrunc_hardnormal_opd.sh run_train_tale_budget_rolloutlen_hardtrunc_hardnormal_esr50_opd.sh verl/examples/fire_opd/run_opd_strong_to_weak_student_raw_teacher_tale_budget30b.sh
```

Expected: exit 0.

- [ ] **Step 3: Interrupt current hard-normal20 run and launch ESR50**

Stop `opd-budget20-hardnormal-step50-noprobe-compileoff-20260707_165639`, verify GPU0-3 free, then launch:

```bash
EXP=opd-budget20-hardnormal-esr50-step50-noprobe-compileoff-$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME=$EXP \
CHECKPOINT_DIR=/home/mchen/FiRe-OPD/checkpoints/$EXP \
REPO_DIR=/home/mchen/FiRe-OPD/.worktrees/hard-normal-budget20-routing \
DATA_ROOT=/home/mchen/FiRe-OPD/data/g-opd \
STUDENT_MODEL=/home/mchen/FiRe-OPD/models/Qwen3-4B \
TEACHER_MODEL=/home/mchen/FiRe-OPD/models/Qwen3-30B-A3B-Instruct-2507 \
bash run_train_tale_budget_rolloutlen_hardtrunc_hardnormal_esr50_opd.sh \
  trainer.total_training_steps=50 \
  trainer.save_freq=50 \
  trainer.test_freq=-1 \
  trainer.val_before_train=False \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.ref.use_torch_compile=False
```

Expected metrics: `normal_prompt_ratio` materially higher than 15%, `esr_beta_mean` above 0.20, hard rows use normal prompt, non-hard rows remain budget, hard entropy remains 0.
