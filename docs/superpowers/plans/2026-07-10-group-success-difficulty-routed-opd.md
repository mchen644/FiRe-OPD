# Group-Success Difficulty-Routed OPD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a clean online `n=4` group-success difficulty route and launch a 50-step pure-RKL OPD run with `4/4 -> concise ESR20` and every other group routed to normal ESR50.

**Architecture:** Keep the legacy single-rollout confidence method intact and add an isolated pure helper for grouping shuffled rows by `uid`. Dispatch to that helper in the trainer, reuse the existing per-row TALE prompt and ESR plumbing, and add a dedicated launcher that fixes 256 prompts, four rollouts, and 1024 actor rows.

**Tech Stack:** Python 3.10, PyTorch, NumPy, Hydra/OmegaConf, pytest, Bash, veRL/Ray/vLLM.

## Global Constraints

- Generate exactly `256 * 4 = 1024` trajectories per production step.
- Teacher-score and actor-train all 1024 trajectories; do not filter candidates.
- Route `k=4` groups to concise teacher plus ESR `0.20`.
- Route `k=0..3` groups to normal teacher plus ESR `0.50`.
- Use verifier reward only for group routing; keep the actor objective pure reverse-KL OPD.
- Do not use confidence, direct entropy, length reward, EOPD, peer context, or historical state.
- Preserve the legacy `two_signal_prompt_esr_entropy` method and its tests.
- Fail on missing `uid` or any group whose size differs from four.
- Do not modify or commit unrelated existing worktree changes.

---

## File Structure

- Modify `verl/verl/trainer/ppo/difficulty_aware_opd.py`: pure group-success result type, routing computation, validation, and metrics.
- Modify `verl/tests/trainer/ppo/test_difficulty_aware_opd.py`: unit and trainer-dispatch tests for shuffled groups, all difficulty buckets, ESR masks, and failures.
- Modify `verl/verl/trainer/config/algorithm.py`: typed fields for group size, group threshold, and two ESR values.
- Modify `verl/verl/trainer/config/ppo_trainer.yaml`: Hydra defaults and documentation for the new method.
- Modify `verl/tests/trainer/config/test_algo_config_on_cpu.py`: Hydra/dataclass override coverage.
- Modify `verl/verl/trainer/ppo/ray_trainer.py`: method dispatch and new per-row batch fields.
- Create `run_train_group_success_difficulty_opd.sh`: dedicated validated production launcher with a dry-run mode.
- Create `verl/tests/trainer/ppo/test_group_success_launcher.py`: launcher contract test without starting GPU work.

### Task 1: Pure Group-Success Routing

**Files:**
- Modify: `verl/tests/trainer/ppo/test_difficulty_aware_opd.py`
- Modify: `verl/verl/trainer/ppo/difficulty_aware_opd.py`

**Interfaces:**
- Produces: `GroupSuccessRoutingResult`.
- Produces: `compute_group_success_routing(*, token_level_scores, response_mask, uids, config) -> GroupSuccessRoutingResult`.
- Produces: `summarize_group_success_routing(result, original_response_lengths=None) -> dict[str, float]`.

- [ ] **Step 1: Write failing tests for shuffled groups and all routes**

Add tests that build five four-row groups with `k=0,1,2,3,4`, shuffle row order, and assert:

```python
result = compute_group_success_routing(
    token_level_scores=scores,
    response_mask=torch.ones_like(scores),
    uids=np.asarray(uids, dtype=object),
    config=_group_cfg(),
)

for uid in {"g0", "g1", "g2", "g3"}:
    rows = np.flatnonzero(np.asarray(uids) == uid)
    assert set(result.prompt_styles[rows]) == {"normal"}
    assert set(result.esr_beta[rows].tolist()) == {0.5}

easy_rows = np.flatnonzero(np.asarray(uids) == "g4")
assert set(result.prompt_styles[easy_rows]) == {"concise"}
assert all(value == pytest.approx(0.2) for value in result.esr_beta[easy_rows].tolist())
assert result.group_correct_counts.tolist() == [0, 1, 2, 3, 4]
```

Also assert the broadcast `easy`, `learnable`, and `unresolved` masks for each group.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
PYTHONPATH=/home/mchen/FiRe-OPD/verl pytest -q \
  verl/tests/trainer/ppo/test_difficulty_aware_opd.py -k group_success
```

Expected: collection or import failure because the new result/helper does not exist.

- [ ] **Step 3: Implement the result type and pure helper**

Add this public result shape:

```python
@dataclass(frozen=True)
class GroupSuccessRoutingResult:
    correct: torch.Tensor
    group_correct_count: torch.Tensor
    group_accuracy: torch.Tensor
    easy: torch.Tensor
    learnable: torch.Tensor
    unresolved: torch.Tensor
    esr_beta: torch.Tensor
    prompt_styles: np.ndarray
    sequence_rewards: torch.Tensor
    group_correct_counts: torch.Tensor
    expected_group_size: int
```

Implement the helper with these exact rules:

```python
sequence_rewards = token_level_scores.float().sum(dim=-1)
correct = (sequence_rewards > correct_reward_threshold).float()

for uid in first_seen_uid_order:
    rows = rows_by_uid[uid]
    if len(rows) != expected_group_size:
        raise ValueError(
            f"group-success routing expected {expected_group_size} rows for uid={uid!r}, got {len(rows)}"
        )
    k = int(correct[rows].sum().item())
    is_easy = k == easy_group_correct_count
    is_unresolved = k == 0
    is_learnable = not is_easy and not is_unresolved
```

Broadcast `k`, `k / expected_group_size`, the three masks, prompt style, and ESR fraction back to the original row positions. Validate shapes, prompt styles, group size, threshold, and ESR fractions before grouping.

- [ ] **Step 4: Add metrics and validation tests**

Test metrics for group count, the five `k` histogram bins, bucket ratios, prompt ratios, ESR min/max/mean, and bucket-specific response lengths. Add explicit failure tests for:

```python
with pytest.raises(ValueError, match="uids length"):
    compute_group_success_routing(..., uids=np.array(["too-short"], dtype=object), ...)

with pytest.raises(ValueError, match="expected 4 rows"):
    compute_group_success_routing(..., uids=np.array(["a", "a", "a", "b"], dtype=object), ...)
```

- [ ] **Step 5: Run routing tests and verify GREEN**

Run:

```bash
PYTHONPATH=/home/mchen/FiRe-OPD/verl pytest -q \
  verl/tests/trainer/ppo/test_difficulty_aware_opd.py -k 'group_success or rank_to_unit or two_signal or summarize'
```

Expected: all selected tests pass, including the unchanged legacy tests.

- [ ] **Step 6: Commit the pure helper**

```bash
git add verl/verl/trainer/ppo/difficulty_aware_opd.py \
        verl/tests/trainer/ppo/test_difficulty_aware_opd.py
git commit -m "Add group-success OPD routing helper"
```

### Task 2: Config and Trainer Dispatch

**Files:**
- Modify: `verl/verl/trainer/config/algorithm.py`
- Modify: `verl/verl/trainer/config/ppo_trainer.yaml`
- Modify: `verl/tests/trainer/config/test_algo_config_on_cpu.py`
- Modify: `verl/verl/trainer/ppo/ray_trainer.py`
- Modify: `verl/tests/trainer/ppo/test_difficulty_aware_opd.py`

**Interfaces:**
- Consumes: `compute_group_success_routing` and `summarize_group_success_routing` from Task 1.
- Produces: method dispatch for `group_success_prompt_esr`.
- Produces batch keys: `difficulty_aware_group_correct_count`, `difficulty_aware_group_accuracy`, `difficulty_aware_learnable`, and `difficulty_aware_unresolved`.

- [ ] **Step 1: Add failing Hydra config test**

Compose the trainer config with:

```python
overrides=[
    "algorithm.difficulty_aware_opd.enabled=True",
    "algorithm.difficulty_aware_opd.method=group_success_prompt_esr",
    "algorithm.difficulty_aware_opd.expected_group_size=4",
    "algorithm.difficulty_aware_opd.easy_group_correct_count=4",
    "algorithm.difficulty_aware_opd.easy_esr_beta=0.2",
    "algorithm.difficulty_aware_opd.non_easy_esr_beta=0.5",
]
```

Assert every typed value after `omega_conf_to_dataclass` conversion.

- [ ] **Step 2: Run config test and verify RED**

Run:

```bash
PYTHONPATH=/home/mchen/FiRe-OPD/verl pytest -q \
  verl/tests/trainer/config/test_algo_config_on_cpu.py::TestAlgoConfig::test_yaml_accepts_group_success_difficulty_overrides
```

Expected: Hydra rejects unknown fields or the named test is absent.

- [ ] **Step 3: Add typed config and YAML defaults**

Extend `DifficultyAwareOpdConfig` and the matching YAML block with:

```python
expected_group_size: int = 4
easy_group_correct_count: int = 4
easy_esr_beta: float = 0.20
non_easy_esr_beta: float = 0.50
```

Keep every existing legacy field and default unchanged.

- [ ] **Step 4: Add failing trainer-dispatch test**

Build an eight-row `DataProto` with two shuffled UIDs. Omit `old_log_probs` to prove the new method does not depend on confidence. Call `_apply_difficulty_aware_opd_routing` and assert:

```python
assert batch.batch["difficulty_aware_group_correct_count"].tolist() == [4.0] * 4 + [1.0] * 4
assert "difficulty_aware_confidence_rank" not in batch.batch
assert "difficulty_aware_entropy_weight" not in batch.batch
assert metrics["difficulty_aware_opd/group_count"] == 2.0
```

Use UID-based row selection if the fixture order is shuffled rather than relying on contiguous rows.

- [ ] **Step 5: Run trainer test and verify RED**

Run:

```bash
PYTHONPATH=/home/mchen/FiRe-OPD/verl pytest -q \
  verl/tests/trainer/ppo/test_difficulty_aware_opd.py -k apply_group_success
```

Expected: the current dispatcher rejects missing `old_log_probs` or calls the legacy helper.

- [ ] **Step 6: Implement method dispatch**

At the start of `_apply_difficulty_aware_opd_routing`, read:

```python
method = str(difficulty_config.get("method", "two_signal_prompt_esr_entropy"))
```

For `group_success_prompt_esr`, require `response_mask` and `uid`, call the group helper, write all documented batch fields, set `difficulty_aware_prompt_style`, and update group metrics. Do not create confidence or entropy keys. For `two_signal_prompt_esr_entropy`, retain the existing old-log-prob validation and behavior. Raise for any other method.

- [ ] **Step 7: Verify per-row ESR mask reuse**

In a test, pass the group result's `esr_beta` into `compute_rollout_length_tale_budget` with four 100-token rows and assert:

```python
assert result.esr_tokens.tolist() == [20, 20, 20, 20, 50, 50, 50, 50]
```

- [ ] **Step 8: Run config, routing, and TALE tests**

Run:

```bash
PYTHONPATH=/home/mchen/FiRe-OPD/verl pytest -q \
  verl/tests/trainer/config/test_algo_config_on_cpu.py \
  verl/tests/trainer/ppo/test_difficulty_aware_opd.py \
  verl/tests/trainer/ppo/test_tale_budget.py
```

Expected: all tests pass.

- [ ] **Step 9: Commit config and trainer dispatch**

```bash
git add verl/verl/trainer/config/algorithm.py \
        verl/verl/trainer/config/ppo_trainer.yaml \
        verl/verl/trainer/ppo/ray_trainer.py \
        verl/tests/trainer/config/test_algo_config_on_cpu.py \
        verl/tests/trainer/ppo/test_difficulty_aware_opd.py
git commit -m "Route OPD by rollout-group success"
```

### Task 3: Dedicated Production Launcher

**Files:**
- Create: `run_train_group_success_difficulty_opd.sh`
- Create: `verl/tests/trainer/ppo/test_group_success_launcher.py`

**Interfaces:**
- Produces: `GROUP_SUCCESS_DRY_RUN=1 bash run_train_group_success_difficulty_opd.sh`.
- Produces: the production experiment `opd-n4-easy4of4-concise20-noneasynormal50-purerkl-step50`.

- [ ] **Step 1: Write failing launcher contract test**

Use `subprocess.run` with `GROUP_SUCCESS_DRY_RUN=1` and assert the output contains:

```python
required = [
    "data.train_batch_size=256",
    "actor_rollout_ref.rollout.n=4",
    "actor_rollout_ref.actor.ppo_mini_batch_size=1024",
    "algorithm.difficulty_aware_opd.method=group_success_prompt_esr",
    "algorithm.difficulty_aware_opd.easy_esr_beta=0.20",
    "algorithm.difficulty_aware_opd.non_easy_esr_beta=0.50",
    "algorithm.difficulty_aware_opd.hard_entropy_coef=0.0",
    "actor_rollout_ref.actor.entropy_coeff=0",
    "trainer.total_training_steps=50",
]
for value in required:
    assert value in completed.stdout
```

- [ ] **Step 2: Run launcher test and verify RED**

Run:

```bash
pytest -q verl/tests/trainer/ppo/test_group_success_launcher.py
```

Expected: failure because the launcher does not exist.

- [ ] **Step 3: Implement the launcher**

The Bash launcher must:

```bash
ROLLOUT_N=4
PROMPT_BATCH_SIZE=256
PPO_MINI_BATCH_SIZE=1024
EXPECTED_GROUP_SIZE=4
TOTAL_TRAINING_STEPS=50
EXPERIMENT_NAME=opd-n4-easy4of4-concise20-noneasynormal50-purerkl-step50
```

Validate both equalities before building a Bash argument array:

```bash
if (( ROLLOUT_N != EXPECTED_GROUP_SIZE )); then
  echo "ERROR: ROLLOUT_N must equal EXPECTED_GROUP_SIZE" >&2
  exit 2
fi
if (( PROMPT_BATCH_SIZE * ROLLOUT_N != PPO_MINI_BATCH_SIZE )); then
  echo "ERROR: prompt batch times rollout count must equal PPO mini-batch size" >&2
  exit 2
fi
```

Set normal teacher style, TALE rollout-length source, hard truncation, pure-RKL flags, candidate selection disabled, probe disabled, entropy zero, compile disabled, `trainer.resume_mode=disable`, save frequency 20, and validation disabled. In dry-run mode, print the resolved environment and argument array and exit without importing Python or touching GPUs. Otherwise execute the existing strong-to-weak TALE base script with the array.

- [ ] **Step 4: Run syntax and contract tests**

Run:

```bash
bash -n run_train_group_success_difficulty_opd.sh
pytest -q verl/tests/trainer/ppo/test_group_success_launcher.py
GROUP_SUCCESS_DRY_RUN=1 bash run_train_group_success_difficulty_opd.sh | \
  rg 'train_batch_size=256|rollout.n=4|group_success_prompt_esr|easy_esr_beta=0.20|non_easy_esr_beta=0.50'
```

Expected: shell syntax passes, pytest passes, and all five config fragments are printed.

- [ ] **Step 5: Commit the launcher**

```bash
git add run_train_group_success_difficulty_opd.sh \
        verl/tests/trainer/ppo/test_group_success_launcher.py
git commit -m "Add group-success OPD training launcher"
```

### Task 4: Verification and `opd-CLI` Launch

**Files:**
- Verify only: all files from Tasks 1-3.
- Runtime output: `logs/difficulty_routed/opd-n4-easy4of4-concise20-noneasynormal50-purerkl-step50.log`.

**Interfaces:**
- Consumes: the production launcher from Task 3.
- Produces: an active 50-step training process in the existing `opd-CLI` session.

- [ ] **Step 1: Run the focused CPU suite**

```bash
PYTHONPATH=/home/mchen/FiRe-OPD/verl pytest -q \
  verl/tests/trainer/config/test_algo_config_on_cpu.py \
  verl/tests/trainer/ppo/test_difficulty_aware_opd.py \
  verl/tests/trainer/ppo/test_tale_budget.py \
  verl/tests/trainer/ppo/test_group_success_launcher.py
```

Expected: all selected tests pass.

- [ ] **Step 2: Run static validation**

```bash
python -m py_compile \
  verl/verl/trainer/ppo/difficulty_aware_opd.py \
  verl/verl/trainer/ppo/ray_trainer.py \
  verl/verl/trainer/config/algorithm.py
bash -n run_train_group_success_difficulty_opd.sh
git diff --check HEAD -- \
  verl/verl/trainer/ppo/difficulty_aware_opd.py \
  verl/verl/trainer/ppo/ray_trainer.py \
  verl/verl/trainer/config/algorithm.py \
  verl/verl/trainer/config/ppo_trainer.yaml \
  verl/tests/trainer/ppo/test_difficulty_aware_opd.py \
  verl/tests/trainer/config/test_algo_config_on_cpu.py \
  run_train_group_success_difficulty_opd.sh \
  verl/tests/trainer/ppo/test_group_success_launcher.py
```

Expected: all commands exit zero.

- [ ] **Step 3: Confirm the target session and GPUs are free**

```bash
tmux has-session -t opd-CLI
tmux list-panes -t opd-CLI -F '#{pane_id} #{pane_current_command} #{pane_dead}'
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

Expected: `opd-CLI` exists, its target pane is live, and no stopped-run GPU process remains.

- [ ] **Step 4: Start the production run in `opd-CLI`**

```bash
mkdir -p /home/mchen/FiRe-OPD/logs/difficulty_routed
tmux send-keys -t opd-CLI C-c
tmux send-keys -t opd-CLI \
  "cd /home/mchen/FiRe-OPD && bash run_train_group_success_difficulty_opd.sh 2>&1 | tee logs/difficulty_routed/opd-n4-easy4of4-concise20-noneasynormal50-purerkl-step50.log" C-m
```

Expected: the pane starts the dedicated launcher and the named log file appears.

- [ ] **Step 5: Verify resolved config and first completed step**

Poll the log without stopping the run:

```bash
rg -n 'train_batch_size=256|rollout.n=4|group_success_prompt_esr|easy_esr_beta=0.2|non_easy_esr_beta=0.5|hard_entropy_coef=0' \
  logs/difficulty_routed/opd-n4-easy4of4-concise20-noneasynormal50-purerkl-step50.log
rg -n 'difficulty_aware_opd/group_count|difficulty_aware_opd/easy_group_ratio|difficulty_aware_opd/learnable_group_ratio|difficulty_aware_opd/unresolved_group_ratio|step:1' \
  logs/difficulty_routed/opd-n4-easy4of4-concise20-noneasynormal50-purerkl-step50.log
```

Expected after step 1: group count 256; the three bucket ratios sum to one; concise ratio equals easy ratio; ESR min/max are 0.20/0.50 when both routes occur; no nonzero hard-entropy metric appears; actor update completes.

- [ ] **Step 6: Leave training active and report runtime identifiers**

Report the tmux session, run name, log path, main process PID, current step, and first-step routing metrics. Do not wait for all 50 steps.
