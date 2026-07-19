# Endpoint-Preserving Tri-Prompt Full-Response OPD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement, validate, profile, and launch a 50-step OPD treatment that trains every complete normal student response while routing the 30B teacher to concise, `B=L_n` budget, or normal prompts.

**Architecture:** Add a new `algorithm.adaptive_triprompt_opd` path without changing historical `adaptive_concise_opd` behavior. Pure helpers own routing, accounting, endpoint telemetry, and cumulative identities; `ray_trainer.py` owns the diagnostic generation call and per-row teacher prompt attachment. The primary batch and reward tensor remain unchanged, while the concise diagnostic is local routing-only state discarded before old/ref/actor stages.

**Tech Stack:** Python 3.10, PyTorch, NumPy, OmegaConf/Hydra, VERL `DataProto`, vLLM rollout workers, pytest, Ruff, Bash, tmux/Slurm.

## Global Constraints

- Work only in `/home/mchen/FiRe-OPD/.worktrees/vanilla-sft-gradient-51200-opd`; do not alter the dirty root checkout's tracked files.
- Use `PYTHONPATH="$PWD/verl:$PWD"` for vendored VERL imports.
- Use `/home/mchen/miniconda3/envs/verl/bin/python` for VERL tests and runtime; use `/home/mchen/miniconda3/envs/gvendi-opd/bin/python` for CPU-only artifact analysis.
- Preserve the historical `adaptive_concise_opd` code path, tests, launchers, checkpoints, logs, and semantics.
- The primary actor prompt and target are always the original normal/raw prompt and complete normal response `y_n`.
- The concise response is diagnostic only, generated only for normal-correct rows, with global `max_tokens=16384`, temperature `1.0`, top-p `1.0`, and natural EOS stopping.
- Correctness is the existing DeepMath `math_verify` binary reward; `sequence_reward > 0.5` defines correct.
- Routes are exactly: easy -> concise 30B prompt; sensitive -> budget 30B prompt with `B=L_n`; hard -> normal 30B prompt.
- Teacher is always Qwen3-30B-A3B-Instruct-2507; no self-training or teacher-generated target is allowed.
- All route weights are `1.0`; do not truncate, replace, remap, or synthesize EOS for the primary response.
- Keep 1,024 distinct questions and 1,024 primary actor rows per step, rollout `n=1`, 50 optimizer steps, seed 42, LR `1e-6`, token-mean loss, reverse-KL-only advantages, and token IS threshold 5.0.
- Keep old adaptive, global TALE, difficulty-aware OPD, candidate selection, rethinking probes, and length-aware losses disabled and mutually exclusive with the new path.
- Runtime work uses tmux pane `opd-CLI:0.0`, the existing allocation's visible GPU tokens `0,1,2,3`, and no nested `srun`.
- Every profile/full namespace is immutable; failed outputs are preserved and never resumed or overwritten.
- A full run starts only after all CPU tests and the one-step GPU profile acceptance gate pass.

---

## File Map

### New files

- `verl/verl/trainer/ppo/adaptive_triprompt_opd.py` — pure route planning, finalization, metrics, endpoint telemetry, cumulative counts, and primary-tensor preservation contracts.
- `verl/tests/trainer/ppo/test_adaptive_triprompt_opd.py` — pure helper and accounting tests.
- `verl/tests/trainer/ppo/test_adaptive_triprompt_trainer.py` — diagnostic generation, prompt routing, preservation, and fit-order tests.
- `verl/tests/trainer/ppo/test_adaptive_triprompt_launcher.py` — Hydra runtime contract and launcher dry-run tests.
- `run_train_adaptive_triprompt_full_response_opd.sh` — immutable profile/full launcher.
- `math_eval/validate_adaptive_triprompt_profile.py` — fail-closed one-step profile validator and atomic report writer.
- `math_eval/test_validate_adaptive_triprompt_profile.py` — synthetic profile validator tests.

### Modified files

- `verl/verl/trainer/config/algorithm.py` — add and export `AdaptiveTriPromptOpdConfig`; add it to `AlgoConfig`.
- `verl/verl/trainer/config/ppo_trainer.yaml` — add disabled structured defaults under `algorithm.adaptive_triprompt_opd`.
- `verl/verl/trainer/main_ppo.py` — add `validate_adaptive_triprompt_runtime_config` and invoke it before worker startup.
- `verl/verl/trainer/ppo/ray_trainer.py` — add runtime glue and integrate the new path before old/ref/actor computation.

### Files that must remain behaviorally unchanged

- `verl/verl/trainer/ppo/adaptive_concise_opd.py`
- `run_train_adaptive_concise_token_neutral_opd.sh`
- all historical evaluation and checkpoint artifacts

---

### Task 1: Add the isolated structured configuration and frozen runtime contract

**Files:**
- Modify: `verl/verl/trainer/config/algorithm.py`
- Modify: `verl/verl/trainer/config/ppo_trainer.yaml`
- Modify: `verl/verl/trainer/main_ppo.py`
- Modify: `verl/tests/trainer/config/test_algo_config_on_cpu.py`
- Create: `verl/tests/trainer/ppo/test_adaptive_triprompt_launcher.py`

**Interfaces:**
- Produces: `AdaptiveTriPromptOpdConfig` with fields `enabled`, `correct_reward_threshold`, `diagnostic_max_response_length`, `budget_alpha`, `teacher_prompt_key`, `temperature`, `top_p`, and `expected_questions_per_step`.
- Produces: `validate_adaptive_triprompt_runtime_config(config) -> dict[str, object] | None`.
- Consumed later by: trainer runtime and launcher tasks.

- [ ] **Step 1: Write failing structured-config tests**

Add a config test that imports the new class and composes Hydra overrides:

```python
def test_yaml_accepts_adaptive_triprompt_opd_overrides(self):
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = omega_conf_to_dataclass(
            compose(
                config_name="ppo_trainer",
                overrides=[
                    "algorithm.adaptive_triprompt_opd.enabled=True",
                    "algorithm.adaptive_triprompt_opd.correct_reward_threshold=0.5",
                    "algorithm.adaptive_triprompt_opd.diagnostic_max_response_length=16384",
                    "algorithm.adaptive_triprompt_opd.budget_alpha=1.0",
                    "algorithm.adaptive_triprompt_opd.teacher_prompt_key=teacher_prompt",
                    "algorithm.adaptive_triprompt_opd.temperature=1.0",
                    "algorithm.adaptive_triprompt_opd.top_p=1.0",
                    "algorithm.adaptive_triprompt_opd.expected_questions_per_step=1024",
                ],
            ).algorithm
        )
    routed = config.adaptive_triprompt_opd
    assert routed.enabled is True
    assert routed.diagnostic_max_response_length == 16384
    assert routed.budget_alpha == 1.0
    assert routed.teacher_prompt_key == "teacher_prompt"
```

- [ ] **Step 2: Run the focused config test and verify RED**

Run:

```bash
PYTHONPATH="$PWD/verl:$PWD" /home/mchen/miniconda3/envs/verl/bin/python \
  -m pytest verl/tests/trainer/config/test_algo_config_on_cpu.py \
  -k adaptive_triprompt -q
```

Expected: FAIL because `AdaptiveTriPromptOpdConfig` and the YAML key do not exist.

- [ ] **Step 3: Implement the dataclass and YAML defaults**

Add and export:

```python
@dataclass
class AdaptiveTriPromptOpdConfig(BaseConfig):
    """Endpoint-preserving tri-prompt OPD configuration."""

    enabled: bool = False
    correct_reward_threshold: float = 0.5
    diagnostic_max_response_length: int = 16384
    budget_alpha: float = 1.0
    teacher_prompt_key: str = "teacher_prompt"
    temperature: float = 1.0
    top_p: float = 1.0
    expected_questions_per_step: int = 1024
```

Add `adaptive_triprompt_opd: AdaptiveTriPromptOpdConfig = field(default_factory=AdaptiveTriPromptOpdConfig)` to `AlgoConfig`, add the class to `__all__`, and add matching structured defaults to `ppo_trainer.yaml`.

- [ ] **Step 4: Write failing runtime-contract tests**

In the new launcher test file, compose a valid config and assert the exact returned contract. Parameterize mutations for every scientific pin and forbidden treatment, including simultaneous old adaptive enablement:

```python
def test_triprompt_runtime_contract_accepts_only_frozen_shape():
    contract = validate_adaptive_triprompt_runtime_config(_compose_valid_config())
    assert contract == {
        "correct_reward_threshold": 0.5,
        "diagnostic_max_response_length": 16384,
        "budget_alpha": 1.0,
        "teacher_prompt_key": "teacher_prompt",
        "expected_questions_per_step": 1024,
    }

@pytest.mark.parametrize(("path", "value"), [
    ("algorithm.adaptive_triprompt_opd.budget_alpha", 0.5),
    ("algorithm.adaptive_concise_opd.enabled", True),
    ("algorithm.tale_budget.enabled", True),
    ("data.max_response_length", 8192),
    ("actor_rollout_ref.rollout.n", 2),
    ("actor_rollout_ref.actor.policy_loss.length_aware_opd", True),
    ("trainer.resume_mode", "auto"),
])
def test_triprompt_runtime_contract_rejects_mutation(path, value):
    config = _compose_valid_config()
    OmegaConf.update(config, path, value, merge=True)
    with pytest.raises(ValueError, match=path):
        validate_adaptive_triprompt_runtime_config(config)
```

- [ ] **Step 5: Run runtime-contract tests and verify RED**

Run:

```bash
PYTHONPATH="$PWD/verl:$PWD" /home/mchen/miniconda3/envs/verl/bin/python \
  -m pytest verl/tests/trainer/ppo/test_adaptive_triprompt_launcher.py -q
```

Expected: collection FAIL because `validate_adaptive_triprompt_runtime_config` does not exist.

- [ ] **Step 6: Implement the runtime validator and startup invocation**

Implement a sibling of `validate_adaptive_concise_runtime_config` that pins all fields from the global constraints and rejects these paths when true:

```python
for path in (
    "algorithm.adaptive_concise_opd.enabled",
    "algorithm.tale_budget.enabled",
    "algorithm.difficulty_aware_opd.enabled",
    "algorithm.candidate_selection.enabled",
    "algorithm.rethinking_opd_probe.enabled",
    "algorithm.opd_proxy_verify_capture.enabled",
):
    if OmegaConf.select(config, path, default=False):
        raise ValueError(f"adaptive tri-prompt OPD forbids {path}")
```

Call the validator in `TaskRunner.run` at the same pre-worker validation boundary as the old adaptive validator.

- [ ] **Step 7: Run focused tests and verify GREEN**

Run both focused files. Expected: all selected tests PASS and historical adaptive config tests remain PASS.

- [ ] **Step 8: Commit Task 1**

```bash
git add verl/verl/trainer/config/algorithm.py \
  verl/verl/trainer/config/ppo_trainer.yaml \
  verl/verl/trainer/main_ppo.py \
  verl/tests/trainer/config/test_algo_config_on_cpu.py \
  verl/tests/trainer/ppo/test_adaptive_triprompt_launcher.py
git commit -m "feat: add adaptive tri-prompt OPD config"
```

---

### Task 2: Implement pure tri-route planning, accounting, and preservation contracts

**Files:**
- Create: `verl/verl/trainer/ppo/adaptive_triprompt_opd.py`
- Create: `verl/tests/trainer/ppo/test_adaptive_triprompt_opd.py`

**Interfaces:**
- Produces: `AdaptiveTriPromptProbePlan`.
- Produces: `AdaptiveTriPromptRoutingResult`.
- Produces: `plan_adaptive_triprompt_probes(...)`.
- Produces: `finalize_adaptive_triprompt_routing(...)`.
- Produces: `summarize_adaptive_triprompt_routing(...)`.
- Produces: `summarize_response_endpoints(...)`.
- Produces: `update_adaptive_triprompt_cumulative_counts(...)`.
- Produces: `capture_primary_tensor_contract(...)` and `verify_primary_tensor_contract(...)`.

- [ ] **Step 1: Write failing route-table tests**

Construct normal rewards `[wrong, correct, correct]`, diagnostic rewards `[wrong, correct]`, and assert:

```python
assert result.hard.tolist() == [True, False, False]
assert result.sensitive.tolist() == [False, True, False]
assert result.easy.tolist() == [False, False, True]
assert result.teacher_prompt_styles.tolist() == ["normal", "budget", "concise"]
assert result.budgets.tolist() == [0, normal_lengths[1], 0]
assert result.actor_supervised_lengths.tolist() == normal_lengths
```

Also assert the diagnostic plan selects only normal-correct rows and carries no per-row `max_tokens` field.

- [ ] **Step 2: Run route tests and verify RED**

```bash
PYTHONPATH="$PWD/verl:$PWD" /home/mchen/miniconda3/envs/verl/bin/python \
  -m pytest verl/tests/trainer/ppo/test_adaptive_triprompt_opd.py -q
```

Expected: import/collection failure because the module does not exist.

- [ ] **Step 3: Implement minimal route dataclasses and functions**

Use these shapes:

```python
@dataclass(frozen=True)
class AdaptiveTriPromptProbePlan:
    normal_correct: torch.Tensor
    hard: torch.Tensor
    normal_lengths: torch.Tensor
    probe_indices: torch.Tensor

@dataclass(frozen=True)
class AdaptiveTriPromptRoutingResult:
    normal_correct: torch.Tensor
    easy: torch.Tensor
    sensitive: torch.Tensor
    hard: torch.Tensor
    normal_lengths: torch.Tensor
    concise_lengths: torch.Tensor
    actor_supervised_lengths: torch.Tensor
    probe_indices: torch.Tensor
    budgets: torch.Tensor
    teacher_prompt_styles: np.ndarray
    concise_parse_fail_count: int
    concise_cap_hit_count: int
```

Pin threshold `0.5`, require finite rewards and contiguous masks, require exact diagnostic index equality, allow an empty diagnostic batch, and reject any concise length above 16,384. Set `budgets[sensitive] = normal_lengths[sensitive]` and leave all other budgets zero.

- [ ] **Step 4: Write failing metrics and cumulative tests**

Assert all count identities, teacher prompt counts, diagnostic length metrics, and:

```python
assert metrics["adaptive_triprompt_opd/actor_supervised_tokens"] == metrics[
    "adaptive_triprompt_opd/normal_response_tokens"
]
assert metrics["adaptive_triprompt_opd/supervision_token_residual"] == 0.0
assert metrics["adaptive_triprompt_opd/full_response_preservation_ratio"] == 1.0
assert metrics["adaptive_triprompt_opd/route_weight_min"] == 1.0
assert metrics["adaptive_triprompt_opd/route_weight_max"] == 1.0
```

Cumulative keys are `total_questions`, `concise_probe_count`, `easy_count`, `sensitive_count`, and `hard_count`.

- [ ] **Step 5: Implement metrics and cumulative functions**

Implement exact integer identities before returning floats for logger compatibility. Sensitive budget min/mean/max must be zero when no sensitive rows exist and exactly summarize `L_n` otherwise.

- [ ] **Step 6: Write failing EOS/cap telemetry tests**

Use response IDs containing EOS on one row and a full 16,384-token mask without EOS on another. Require independent counts for EOS and cap hits; correctness remains independent of EOS. Test concise missing-box accounting separately through `finalize_adaptive_triprompt_routing(..., concise_texts=...)`.

- [ ] **Step 7: Implement endpoint telemetry**

Implement:

```python
def summarize_response_endpoints(
    *,
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
    eos_token_id: int,
    max_response_length: int,
    metric_prefix: str,
) -> dict[str, float]:
```

Count EOS only inside valid masks and count cap hits by valid length equality. Reject multiple valid EOS tokens after the first terminal EOS or malformed masks.

- [ ] **Step 8: Write failing primary-preservation tests**

Capture descriptors, mutate/replace one primary tensor, and require verification failure. A no-op non-tensor prompt addition must pass.

- [ ] **Step 9: Implement zero-copy preservation descriptors**

Capture for `responses`, `response_mask`, `input_ids`, `attention_mask`, and `position_ids`: object identity/data pointer, shape, stride, dtype, device, and PyTorch version counter. Capture the reward tensor similarly and verify descriptors plus valid token totals after routing. This avoids cloning hundreds of MB while detecting replacement or in-place mutation.

- [ ] **Step 10: Run pure tests and verify GREEN**

Run:

```bash
PYTHONPATH="$PWD/verl:$PWD" /home/mchen/miniconda3/envs/verl/bin/python \
  -m pytest verl/tests/trainer/ppo/test_adaptive_triprompt_opd.py -q
```

Expected: all tests PASS.

- [ ] **Step 11: Commit Task 2**

```bash
git add verl/verl/trainer/ppo/adaptive_triprompt_opd.py \
  verl/tests/trainer/ppo/test_adaptive_triprompt_opd.py
git commit -m "feat: add adaptive tri-prompt routing"
```

---

### Task 3: Add full-cap diagnostic generation and route-specific 30B prompts

**Files:**
- Modify: `verl/verl/trainer/ppo/ray_trainer.py`
- Create: `verl/tests/trainer/ppo/test_adaptive_triprompt_trainer.py`

**Interfaces:**
- Consumes: Task 2 plan/result/preservation APIs.
- Produces: `_build_adaptive_triprompt_generation_batch(...)`.
- Produces: `_apply_adaptive_triprompt_teacher_prompts(...)`.
- Produces: `_apply_adaptive_triprompt_opd(...)`.

- [ ] **Step 1: Write failing diagnostic-generation tests**

Use the existing fake tokenizer/batch pattern, but assert the new diagnostic request has:

```python
assert diagnostic.meta_info["response_length"] == 16384
assert diagnostic.meta_info["generation_kwargs"] == {
    "disable_rollout_log_probs": True,
    "temperature": 1.0,
    "top_p": 1.0,
}
assert "max_tokens_by_row" not in diagnostic.meta_info["generation_kwargs"]
assert diagnostic.non_tensor_batch["adaptive_triprompt_original_row"].tolist() == [0, 2]
```

Require preserved `reward_model`, `extra_info`, `uid`, and concise prompt text with the verbose normal suffix removed.

- [ ] **Step 2: Run the focused trainer test and verify RED**

```bash
PYTHONPATH="$PWD/verl:$PWD" /home/mchen/miniconda3/envs/verl/bin/python \
  -m pytest verl/tests/trainer/ppo/test_adaptive_triprompt_trainer.py -q
```

Expected: import failure for `_build_adaptive_triprompt_generation_batch`.

- [ ] **Step 3: Implement the diagnostic request builder**

Select `plan.probe_indices`, construct concise messages with `build_concise_teacher_messages`, tokenize with left padding, and return `DataProto` metadata containing global response length 16,384. Fail rather than truncate overlong prompts. Do not request rollout log-probs for diagnostics.

- [ ] **Step 4: Write failing three-prompt attachment tests**

For styles `normal`, `budget`, `concise`, assert exact messages:

```python
assert teacher_prompts[0] == raw_prompt_0
assert "use less than 321 tokens" in teacher_prompts[1][0]["content"]
assert "Solve concisely" in teacher_prompts[2][0]["content"]
assert "Please reason step by step" not in teacher_prompts[1][0]["content"]
assert "Please reason step by step" not in teacher_prompts[2][0]["content"]
```

Reject a non-sensitive nonzero budget, a sensitive budget unequal to `L_n`, invalid styles, and misaligned vectors.

- [ ] **Step 5: Implement route-specific teacher prompt attachment**

Deep-copy raw messages for `normal`, call `build_tale_budget_teacher_messages(question, budget)` for `budget`, and call `build_concise_teacher_messages(question)` for `concise`. Mutate only `batch.non_tensor_batch[teacher_prompt_key]`.

- [ ] **Step 6: Write failing end-to-end helper tests**

Use a fake worker returning one wrong and one correct concise diagnostic. Assert:

- only normal-correct indices are generated;
- output `batch is input_batch` or retains every primary tensor descriptor;
- output `reward_tensor` is the original full-width object;
- primary response IDs/masks are unchanged;
- styles are hard normal, sensitive budget, easy concise;
- diagnostic-only keys are absent from the primary batch;
- all preservation metrics equal their required values.

- [ ] **Step 7: Implement `_apply_adaptive_triprompt_opd`**

Follow this skeleton:

```python
snapshot = capture_primary_tensor_contract(batch=batch, reward_tensor=normal_reward)
plan = plan_adaptive_triprompt_probes(...)
diagnostic_batch = None
if plan.probe_indices.numel():
    diagnostic_prompts = _build_adaptive_triprompt_generation_batch(...)
    diagnostic_batch = actor_rollout_wg.generate_sequences(diagnostic_prompts)
    if "response_mask" not in diagnostic_batch.batch.keys():
        diagnostic_batch.batch["response_mask"] = compute_response_mask(diagnostic_batch)
    concise_reward, _ = compute_reward(diagnostic_batch, reward_fn)
    result = finalize_adaptive_triprompt_routing(...)
else:
    result = finalize_adaptive_triprompt_routing(...empty tensors...)
_apply_adaptive_triprompt_teacher_prompts(batch=batch, result=result, ...)
verify_primary_tensor_contract(snapshot=snapshot, batch=batch, reward_tensor=normal_reward)
metrics = summarize_adaptive_triprompt_routing(result)
if diagnostic_batch is not None:
    del diagnostic_batch
return batch, normal_reward, result, metrics
```

Do not call `truncate_to_adaptive_concise_prefix`, `_remap_reward_to_supervised_prefix`, or global TALE helpers.

- [ ] **Step 8: Run trainer helper tests and verify GREEN**

Expected: all new trainer tests PASS.

- [ ] **Step 9: Commit Task 3**

```bash
git add verl/verl/trainer/ppo/ray_trainer.py \
  verl/tests/trainer/ppo/test_adaptive_triprompt_trainer.py
git commit -m "feat: add full-response tri-prompt runtime"
```

---

### Task 4: Integrate routing into `RayPPOTrainer.fit` before all post-rollout model calls

**Files:**
- Modify: `verl/verl/trainer/ppo/ray_trainer.py`
- Modify: `verl/tests/trainer/ppo/test_adaptive_triprompt_trainer.py`

**Interfaces:**
- Consumes: `_apply_adaptive_triprompt_opd` and Task 2 cumulative updater.
- Produces: complete per-step and cumulative metrics in normal trainer logs.

- [ ] **Step 1: Write failing source-order and metric-lifecycle tests**

Assert the tri-prompt call occurs after full normal reward and before `compute_log_prob(batch)`, `prepare_ref_model_inputs`, and `update_actor(batch)`. Assert the fit source does not mention adaptive truncation inside the tri-prompt branch and contains a separate tri-prompt cumulative state.

- [ ] **Step 2: Run the source-order test and verify RED**

```bash
PYTHONPATH="$PWD/verl:$PWD" /home/mchen/miniconda3/envs/verl/bin/python \
  -m pytest verl/tests/trainer/ppo/test_adaptive_triprompt_trainer.py \
  -k fit_source -q
```

Expected: FAIL because the new helper is not called from `fit`.

- [ ] **Step 3: Integrate the new branch**

Add imports and initialize:

```python
adaptive_triprompt_cumulative: dict[str, int] = {}
triprompt_routing_result: AdaptiveTriPromptRoutingResult | None = None
```

Immediately after normal reward computation, resolve async reward if necessary, record `normal_rollout`/`normal_reward`, call `_apply_adaptive_triprompt_opd`, and merge metrics. Do not rebalance a second time because the primary tensors and lengths did not change. Keep `batch.meta_info["global_token_num"]` unchanged.

After a successful actor update, update cumulative tri-prompt counts independently from historical adaptive counts.

- [ ] **Step 4: Add a diagnostic-leak assertion at the post-route boundary**

Before old log-prob, reject any primary non-tensor or tensor key beginning with `adaptive_triprompt_diagnostic_` or equal to `adaptive_triprompt_original_row`. This is defense in depth in addition to local diagnostic scope.

- [ ] **Step 5: Run new integration tests and historical adaptive trainer tests**

Run:

```bash
PYTHONPATH="$PWD/verl:$PWD" /home/mchen/miniconda3/envs/verl/bin/python -m pytest \
  verl/tests/trainer/ppo/test_adaptive_triprompt_trainer.py \
  verl/tests/trainer/ppo/test_adaptive_concise_trainer.py -q
```

Expected: both suites PASS.

- [ ] **Step 6: Commit Task 4**

```bash
git add verl/verl/trainer/ppo/ray_trainer.py \
  verl/tests/trainer/ppo/test_adaptive_triprompt_trainer.py
git commit -m "feat: integrate tri-prompt OPD trainer path"
```

---

### Task 5: Add the immutable profile/full launcher

**Files:**
- Create: `run_train_adaptive_triprompt_full_response_opd.sh`
- Modify: `verl/tests/trainer/ppo/test_adaptive_triprompt_launcher.py`

**Interfaces:**
- Consumes: Task 1 runtime config.
- Produces: `ADAPTIVE_TRIPROMPT_RUN_MODE=profile|full` launcher contract.

- [ ] **Step 1: Write failing launcher dry-run tests**

Pin these mode-specific values:

```text
profile: steps=1, save_freq=-1, test_freq=-1, val_before=False, logger=["console"]
full:    steps=50, save_freq=50, test_freq=10, val_before=True, logger=["console","wandb"]
```

Require exact experiment names from the spec, `diagnostic_max_response_length=16384`, `budget_alpha=1.0`, all mutual exclusions, entropy chunking, old-log-prob micro-batch 4, resume disablement, and no positional overrides. Parse the printed Hydra command and call `validate_adaptive_triprompt_runtime_config`.

- [ ] **Step 2: Run launcher tests and verify RED**

```bash
PYTHONPATH="$PWD/verl:$PWD" /home/mchen/miniconda3/envs/verl/bin/python \
  -m pytest verl/tests/trainer/ppo/test_adaptive_triprompt_launcher.py \
  -k launcher -q
```

Expected: FAIL because the launcher does not exist.

- [ ] **Step 3: Implement the launcher**

Follow the existing adaptive launcher pattern with new environment names and immutable destinations. Use:

```bash
PROFILE_EXPERIMENT=opd-adaptive-fullconciseprobe-triprompt-fullnormal-rawprompt-4gpu-tp4-profile-step1
FULL_EXPERIMENT=opd-adaptive-fullconciseprobe-triprompt-fullnormal-rawprompt-4gpu-tp4-step50
```

Pin `algorithm.adaptive_triprompt_opd.enabled=True`, disable all conflicting treatments, use the completed adaptive memory settings, require a clean source worktree, reject existing checkpoint/log paths, atomically claim the log, set `PYTHONPATH`, and execute without nested `srun`.

- [ ] **Step 4: Run launcher tests and verify GREEN**

```bash
PYTHONPATH="$PWD/verl:$PWD" /home/mchen/miniconda3/envs/verl/bin/python \
  -m pytest verl/tests/trainer/ppo/test_adaptive_triprompt_launcher.py -q
```

Expected: all dry-run and contract tests PASS.

- [ ] **Step 5: Verify shell syntax**

```bash
bash -n run_train_adaptive_triprompt_full_response_opd.sh
```

Expected: exit 0.

- [ ] **Step 6: Commit Task 5**

```bash
git add run_train_adaptive_triprompt_full_response_opd.sh \
  verl/tests/trainer/ppo/test_adaptive_triprompt_launcher.py
git commit -m "feat: launch endpoint-preserving tri-prompt OPD"
```

---

### Task 6: Add a fail-closed one-step profile validator

**Files:**
- Create: `math_eval/validate_adaptive_triprompt_profile.py`
- Create: `math_eval/test_validate_adaptive_triprompt_profile.py`

**Interfaces:**
- Produces: `validate_adaptive_triprompt_profile(log_path, expected_step, expected_questions, source_commit) -> dict[str, object]`.
- Produces: atomic immutable JSON acceptance report.

- [ ] **Step 1: Write failing synthetic PASS test**

Construct one step line containing all required route counts, prompt counts, full-response metrics, finite actor health, rollout correction metrics, and timing metrics. Require `decision == "pass"` and exact route counts.

- [ ] **Step 2: Run the focused validator test and verify RED**

```bash
/home/mchen/miniconda3/envs/gvendi-opd/bin/python \
  -m pytest math_eval/test_validate_adaptive_triprompt_profile.py -q
```

Expected: import failure because the validator does not exist.

- [ ] **Step 3: Implement parser and PASS validation**

Adapt the historical adaptive validator but require:

```text
normal_correct = easy + sensitive
normal_wrong = hard
probe_count = normal_correct
concise_teacher = easy
budget_teacher = sensitive
normal_teacher = hard
supervision_token_residual = 0
full_response_preservation_ratio = 1
route_weight_min = route_weight_max = 1
```

Require finite `actor/pg_loss`, positive finite `actor/grad_norm`, rollout IS max in `(0,5]`, zero veto/catastrophic fractions, positive timings, exactly one requested step line, and a clean source SHA.

- [ ] **Step 4: Add parameterized FAIL tests**

Mutate each identity, omit each required metric, insert NaN/Inf, duplicate step lines, use zero gradient norm, nonzero supervision residual, preservation ratio below one, wrong route weights, or invalid source SHA. Each case must raise a specific `ValueError`.

- [ ] **Step 5: Implement atomic report CLI behavior**

On success write `decision=pass` and return 0. On validation failure write `decision=fail`, preserve error type/message, and return 2. Refuse to overwrite an existing report.

- [ ] **Step 6: Run validator tests and verify GREEN**

```bash
/home/mchen/miniconda3/envs/gvendi-opd/bin/python \
  -m pytest math_eval/test_validate_adaptive_triprompt_profile.py -q
```

Expected: all tests PASS.

- [ ] **Step 7: Commit Task 6**

```bash
git add math_eval/validate_adaptive_triprompt_profile.py \
  math_eval/test_validate_adaptive_triprompt_profile.py
git commit -m "test: validate adaptive tri-prompt profile"
```

---

### Task 7: Run full static and regression verification

**Files:**
- Verify all files changed in Tasks 1–6.

- [ ] **Step 1: Run focused tri-prompt tests**

```bash
PYTHONPATH="$PWD/verl:$PWD" /home/mchen/miniconda3/envs/verl/bin/python -m pytest \
  verl/tests/trainer/config/test_algo_config_on_cpu.py \
  verl/tests/trainer/ppo/test_adaptive_triprompt_opd.py \
  verl/tests/trainer/ppo/test_adaptive_triprompt_trainer.py \
  verl/tests/trainer/ppo/test_adaptive_triprompt_launcher.py \
  math_eval/test_validate_adaptive_triprompt_profile.py -q
```

Expected: zero failures.

- [ ] **Step 2: Run historical adaptive and legacy regression suites**

```bash
PYTHONPATH="$PWD/verl:$PWD" /home/mchen/miniconda3/envs/verl/bin/python -m pytest \
  verl/tests/trainer/ppo/test_adaptive_concise_opd.py \
  verl/tests/trainer/ppo/test_adaptive_concise_trainer.py \
  verl/tests/trainer/ppo/test_adaptive_concise_launcher.py \
  verl/tests/trainer/ppo/test_tale_budget.py \
  verl/tests/trainer/ppo/test_vanilla_opd_launcher.py \
  verl/tests/workers/rollout/rollout_vllm/test_adaptive_max_tokens.py -q
```

Expected: zero failures.

- [ ] **Step 3: Run formatting, compile, shell, and diff checks**

```bash
/home/mchen/miniconda3/envs/gvendi-opd/bin/python -m ruff check \
  verl/verl/trainer/config/algorithm.py \
  verl/verl/trainer/main_ppo.py \
  verl/verl/trainer/ppo/adaptive_triprompt_opd.py \
  verl/verl/trainer/ppo/ray_trainer.py \
  verl/tests/trainer/ppo/test_adaptive_triprompt_opd.py \
  verl/tests/trainer/ppo/test_adaptive_triprompt_trainer.py \
  verl/tests/trainer/ppo/test_adaptive_triprompt_launcher.py \
  math_eval/validate_adaptive_triprompt_profile.py \
  math_eval/test_validate_adaptive_triprompt_profile.py
PYTHONPYCACHEPREFIX=/tmp/mchen-triprompt-pycache \
  /home/mchen/miniconda3/envs/verl/bin/python -m compileall -q \
  verl/verl/trainer/ppo/adaptive_triprompt_opd.py \
  math_eval/validate_adaptive_triprompt_profile.py
bash -n run_train_adaptive_triprompt_full_response_opd.sh
git diff --check
git status --short
```

Expected: all commands exit 0 and the worktree is clean after task commits.

- [ ] **Step 4: Review the cumulative diff against the design spec**

Verify no historical semantics changed, no diagnostic enters the actor batch, every required metric exists, and the launcher contains no unapproved treatment. Record `git log --oneline` and `git show --stat` as review evidence.

---

### Task 8: Run the one-step GPU profile and launch the gated full training run

**Files and artifacts:**
- Read-only source: committed worktree HEAD.
- Create profile log: `/home/mchen/FiRe-OPD/logs/adaptive_triprompt_opd/opd-adaptive-fullconciseprobe-triprompt-fullnormal-rawprompt-4gpu-tp4-profile-step1.log`
- Create profile report: same basename with `.acceptance.json`.
- Create full log/checkpoint namespace only after profile PASS.

- [ ] **Step 1: Preflight `opd-CLI` and immutable namespaces**

From the controlling shell, capture the pane and verify it is at a prompt inside Slurm, then run inside the pane:

```bash
cd /home/mchen/FiRe-OPD/.worktrees/vanilla-sft-gradient-51200-opd
printf 'SLURM_JOB_ID=%s\nCUDA_VISIBLE_DEVICES=%s\n' "$SLURM_JOB_ID" "$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,uuid,memory.used,memory.free --format=csv,noheader
```

Require four visible tokens `0,1,2,3`, sufficient free memory, clean committed HEAD, and absent profile/full log and checkpoint destinations. Do not use nested `srun`.

- [ ] **Step 2: Launch the profile in `opd-CLI`**

Send exactly:

```bash
ADAPTIVE_TRIPROMPT_RUN_MODE=profile \
  bash run_train_adaptive_triprompt_full_response_opd.sh
```

Monitor the immutable log for startup, route metrics, worker failures, OOM, and process completion. Preserve any failed namespace.

- [ ] **Step 3: Validate the profile independently**

After profile exit, run:

```bash
/home/mchen/miniconda3/envs/gvendi-opd/bin/python \
  math_eval/validate_adaptive_triprompt_profile.py \
  --log /home/mchen/FiRe-OPD/logs/adaptive_triprompt_opd/opd-adaptive-fullconciseprobe-triprompt-fullnormal-rawprompt-4gpu-tp4-profile-step1.log \
  --expected-step 1 \
  --expected-questions 1024 \
  --source-commit "$(git rev-parse HEAD)" \
  --output /home/mchen/FiRe-OPD/logs/adaptive_triprompt_opd/opd-adaptive-fullconciseprobe-triprompt-fullnormal-rawprompt-4gpu-tp4-profile-step1.acceptance.json
```

Require `decision=pass`, then independently read the metric line and recompute all route/prompt/full-response identities.

- [ ] **Step 4: Check projected runtime against Slurm time remaining**

Use profile `timing_s/step`, add conservative validation/checkpoint overhead, and require projected 50-step completion before the allocation deadline. If it does not fit, stop and report without launching.

- [ ] **Step 5: Launch the full run without another user prompt**

Only after Steps 1–4 pass, send in `opd-CLI`:

```bash
ADAPTIVE_TRIPROMPT_RUN_MODE=full \
  bash run_train_adaptive_triprompt_full_response_opd.sh
```

The launcher must claim the formal log and checkpoint namespace atomically, print the source commit and frozen command, initialize W&B only for the full run, and start from the original Qwen3-4B model with resume disabled.

- [ ] **Step 6: Verify formal startup**

Require a live process tree under Slurm job ownership, the expected experiment name, exact source commit, W&B run ID, four visible GPUs, and the first complete step metric line satisfying all route and endpoint identities. If startup or step 1 fails, preserve artifacts and stop; never retry into the same namespace.

- [ ] **Step 7: Record launch status**

Report the formal experiment name, source commit, W&B run ID, Slurm job, log/checkpoint paths, first-step route counts, endpoint metrics, measured step time, and projected completion time. Do not claim final training success until all 50 steps and checkpoint validation complete.
