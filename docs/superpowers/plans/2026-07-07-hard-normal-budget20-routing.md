# Hard-Normal Budget20 Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Budget20 OPD variant where all samples keep 20% ESR supervision, while hard samples use the original/normal teacher prompt and all other samples use the budget teacher prompt.

**Architecture:** Extend the existing pure difficulty-routing helper so prompt style can be selected by a hard gate before the existing easy/default gate. Keep ESR logic unchanged and disable easy ESR compression by configuration for this variant. Add a wrapper script that sets hard-normal routing and no entropy/probe/compile defaults for a clean experiment.

**Tech Stack:** Python 3.10, PyTorch, NumPy, pytest, bash/Hydra config overrides, veRL/FiRe-OPD trainer.

## Global Constraints

- Work in isolated worktree `/home/mchen/FiRe-OPD/.worktrees/hard-normal-budget20-routing`.
- Do not modify or delete WIP in `/home/mchen/FiRe-OPD` main checkout.
- Preserve hardtrunc semantics: `response_length/mean` is supervised/truncated training width; `tale_budget/response_length_mean` is original rollout length.
- This variant uses `TALE_ESR_BETA=0.20`, `DA_EASY_ESR_DELTA=0`, and `DA_MIN_EASY_ESR_BETA=0.20` so all samples supervise 20%.
- This variant uses hard prompt routing: hard samples -> `normal`; all other samples -> `budget`.
- Disable hard entropy for the first diagnostic run: `DA_HARD_ENTROPY_COEF=0`.
- Disable rethinking probe for full training: `algorithm.rethinking_opd_probe.enabled=False`.
- Disable torch compile for retry comparability: `actor_rollout_ref.actor.use_torch_compile=False`, `actor_rollout_ref.ref.use_torch_compile=False`.

---

### Task 1: Add hard-prompt routing to pure helper and config

**Files:**
- Modify: `verl/verl/trainer/config/algorithm.py`
- Modify: `verl/verl/trainer/ppo/difficulty_aware_opd.py`
- Test: `verl/tests/trainer/ppo/test_difficulty_aware_opd.py`
- Test: `verl/tests/trainer/config/test_algo_config_on_cpu.py`

**Interfaces:**
- Consumes: `compute_two_signal_difficulty_routing(..., config, base_esr_beta)`.
- Produces: config fields `hard_prompt_threshold: Optional[float] = None`, `hard_prompt_style: str = "normal"`; prompt selection order `hard gate -> easy gate -> default`.

- [ ] **Step 1: Write failing routing tests**

Add tests to `verl/tests/trainer/ppo/test_difficulty_aware_opd.py` that construct four rows with known correctness/confidence ranks, set `hard_prompt_threshold=0.7`, `hard_prompt_style="normal"`, `easy_prompt_style="budget"`, `default_prompt_style="budget"`, and assert only hard rows get `normal` while ESR beta remains 0.20 when `easy_esr_delta=0` and `min_easy_esr_beta=0.20`.

- [ ] **Step 2: Write failing config test**

Add assertions to `verl/tests/trainer/config/test_algo_config_on_cpu.py` that `DifficultyAwareOpdConfig` exposes `hard_prompt_threshold is None` and `hard_prompt_style == "normal"` by default.

- [ ] **Step 3: Verify RED**

Run:

```bash
pytest verl/tests/trainer/ppo/test_difficulty_aware_opd.py::test_hard_prompt_routing_uses_normal_prompt_without_changing_budget20_esr verl/tests/trainer/config/test_algo_config_on_cpu.py -q
```

Expected: FAIL because the new config fields and hard routing do not exist yet.

- [ ] **Step 4: Implement minimal code**

Add `hard_prompt_threshold: Optional[float] = None` and `hard_prompt_style: str = "normal"` to `DifficultyAwareOpdConfig`. In `compute_two_signal_difficulty_routing`, validate `hard_prompt_style` against `_ALLOWED_PROMPT_STYLES`, then choose prompt style by:

```python
hard_threshold = _config_get(config, "hard_prompt_threshold", None)
hard_prompt_style = str(_config_get(config, "hard_prompt_style", "normal"))
prompt_styles = []
for easy_value, hard_value in zip(easy.detach().cpu(), hard.detach().cpu(), strict=True):
    if hard_threshold is not None and float(hard_value) >= float(hard_threshold):
        prompt_styles.append(hard_prompt_style)
    elif float(easy_value) >= easy_threshold:
        prompt_styles.append(easy_prompt_style)
    else:
        prompt_styles.append(default_prompt_style)
prompt_styles = np.array(prompt_styles, dtype=object)
```

- [ ] **Step 5: Verify GREEN**

Run:

```bash
pytest verl/tests/trainer/ppo/test_difficulty_aware_opd.py::test_hard_prompt_routing_uses_normal_prompt_without_changing_budget20_esr verl/tests/trainer/config/test_algo_config_on_cpu.py -q
```

Expected: PASS.

---

### Task 2: Add run wrapper for hard-normal Budget20 diagnostic

**Files:**
- Create: `run_train_tale_budget_rolloutlen_hardtrunc_hardnormal_opd.sh`
- Test: `run_train_tale_budget_rolloutlen_hardtrunc_hardnormal_opd.sh`

**Interfaces:**
- Consumes: existing wrapper `run_train_tale_budget_rolloutlen_hardtrunc_difficultyrouted_rethinking_probe_opd.sh`.
- Produces: shell wrapper with defaults for hard-normal routing.

- [ ] **Step 1: Write wrapper**

Create a bash wrapper that sets:

```bash
export DA_EASY_PROMPT_STYLE="${DA_EASY_PROMPT_STYLE:-budget}"
export DA_DEFAULT_PROMPT_STYLE="${DA_DEFAULT_PROMPT_STYLE:-budget}"
export DA_EASY_PROMPT_THRESHOLD="${DA_EASY_PROMPT_THRESHOLD:-1.1}"
export DA_MIN_EASY_ESR_BETA="${DA_MIN_EASY_ESR_BETA:-0.20}"
export DA_EASY_ESR_DELTA="${DA_EASY_ESR_DELTA:-0.0}"
export DA_HARD_ENTROPY_COEF="${DA_HARD_ENTROPY_COEF:-0}"
```

Then call the existing difficulty-routed wrapper and append overrides:

```bash
algorithm.difficulty_aware_opd.hard_prompt_threshold=${DA_HARD_PROMPT_THRESHOLD:-0.7}
algorithm.difficulty_aware_opd.hard_prompt_style=${DA_HARD_PROMPT_STYLE:-normal}
algorithm.rethinking_opd_probe.enabled=False
```

- [ ] **Step 2: Verify shell syntax**

Run:

```bash
bash -n run_train_tale_budget_rolloutlen_hardtrunc_hardnormal_opd.sh
```

Expected: exit 0.

---

### Task 3: Regression verification

**Files:**
- Test: `verl/tests/trainer/ppo/test_difficulty_aware_opd.py`
- Test: `verl/tests/trainer/config/test_algo_config_on_cpu.py`
- Test: `run_train_tale_budget_rolloutlen_hardtrunc_hardnormal_opd.sh`

**Interfaces:**
- Consumes: Task 1 helper/config and Task 2 wrapper.
- Produces: verified branch ready to launch a step50 diagnostic run.

- [ ] **Step 1: Run targeted tests**

```bash
pytest verl/tests/trainer/ppo/test_difficulty_aware_opd.py verl/tests/trainer/config/test_algo_config_on_cpu.py -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run syntax checks**

```bash
python -m py_compile verl/verl/trainer/ppo/difficulty_aware_opd.py verl/verl/trainer/config/algorithm.py
bash -n run_train_tale_budget_rolloutlen_hardtrunc_hardnormal_opd.sh
```

Expected: exit 0.

- [ ] **Step 3: Show launch command**

Use:

```bash
EXPERIMENT_NAME=opd-budget20-hardnormal-step50-noprobe-compileoff-$(date +%Y%m%d_%H%M%S) \
CHECKPOINT_DIR=/home/mchen/FiRe-OPD/checkpoints/$EXPERIMENT_NAME \
REPO_DIR=/home/mchen/FiRe-OPD/.worktrees/hard-normal-budget20-routing \
bash run_train_tale_budget_rolloutlen_hardtrunc_hardnormal_opd.sh \
  trainer.total_training_steps=50 \
  trainer.save_freq=50 \
  trainer.test_freq=-1 \
  trainer.val_before_train=False \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.ref.use_torch_compile=False
```

Expected logs should show `hard_entropy_weight_mean: 0.0`, `esr_beta_mean: 0.2`, hard-routed `normal` prompts, and `response_length/mean / tale_budget/response_length_mean` near 0.20.
