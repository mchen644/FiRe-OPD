# Adaptive Concise-Probe Token-Neutral OPD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and launch a Vanilla-matched OPD treatment that adaptively probes normal-correct questions under a 50%-length concise prompt, routes only concise-correct questions to a concise teacher, and exactly exchanges every diagnostic response token for one removed normal-response supervision token.

**Architecture:** Add a focused pure routing/accounting module, a vLLM per-request maximum-token facility, and a trainer orchestration path that runs after normal reward but before old/ref model forwards. Keep the final actor batch at 1,024 normal rollouts, physically prefix-mask it before post-rollout model calls, expose fail-closed metrics, and launch through a pinned standalone script. The normal and diagnostic generations use one student snapshot; diagnostics are reward-only and never enter OPD model forwards.

**Tech Stack:** Python 3.10, PyTorch, TensorDict/DataProto, Hydra/OmegaConf dataclass configuration, VERL Ray trainer, vLLM `SamplingParams`, Bash, pytest, Ruff.

## Global Constraints

- Work only in `/home/mchen/FiRe-OPD/.worktrees/vanilla-sft-gradient-51200-opd`; do not modify the dirty root checkout.
- Use `/home/mchen/miniconda3/envs/verl/bin/python` for VERL tests and execution.
- Primary training is exactly 1,024 distinct questions, `rollout.n=1`, PPO mini-batch 1,024, 50 steps, seed 42, and max normal response length 16,384.
- Student is `/home/mchen/FiRe-OPD/models/Qwen3-4B`; teacher is `/home/mchen/FiRe-OPD/models/Qwen3-30B-A3B-Instruct-2507`.
- Training data is `/home/mchen/FiRe-OPD/data/g-opd/DeepMath-103K/train_filtered_level6.parquet`.
- Preserve reverse-KL-only advantages, token-mean aggregation, token IS threshold 5.0, LR `1e-6`, and all frozen Vanilla optimizer settings.
- Disable TALE, legacy difficulty routing, candidate selection, rethinking probes, hard entropy, length penalty, G-Vendi, resume, and overwrite.
- Concise probes are generated only for normal-correct rows with per-request `max_tokens=floor(0.5 * normal_valid_response_tokens)`.
- Only normal rows enter old-log-prob, teacher/ref, advantages, and actor update; diagnostics are reward-only.
- For every row define `L_c=0` when unprobed and `L_s=L_n-L_c`; abort before update unless `sum(L_c)+sum(L_s)-sum(L_n)==0`.
- Learnable uses a normal teacher; easy uses a concise teacher; hard uses a normal teacher with full response supervision.
- GPU work runs in existing tmux session `opd-CLI` on allocation-owned tokens `0,1,2,3`, without nested `srun`.
- Run a distinct one-step profile before the 50-step launch. Archive failures; never resume or overwrite.

---

## File Structure

### New files

- `verl/verl/trainer/ppo/adaptive_concise_opd.py` — pure probe planning, route finalization, token accounting, metrics, teacher styles, and prefix truncation.
- `verl/tests/trainer/ppo/test_adaptive_concise_opd.py` — unit tests for all route/accounting/error cases.
- `verl/tests/trainer/ppo/test_adaptive_concise_trainer.py` — prompt construction and trainer-order integration tests.
- `verl/tests/workers/rollout/rollout_vllm/test_adaptive_max_tokens.py` — CPU tests for per-request vLLM sampling parameters.
- `run_train_adaptive_concise_token_neutral_opd.sh` — pinned profile/full launcher.
- `verl/tests/trainer/ppo/test_adaptive_concise_launcher.py` — launcher/config/override tests.
- `math_eval/validate_adaptive_concise_profile.py` — parse a step line and enforce first-step scientific/runtime gates.
- `math_eval/test_validate_adaptive_concise_profile.py` — profile-validator tests.

### Modified files

- `verl/verl/trainer/config/algorithm.py` — typed `AdaptiveConciseOpdConfig` and `AlgoConfig` field.
- `verl/verl/trainer/config/ppo_trainer.yaml` — disabled-by-default adaptive configuration.
- `verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` — optional per-request max tokens and diagnostic no-log-prob path.
- `verl/verl/trainer/main_ppo.py` — fail-closed runtime contract validation.
- `verl/verl/trainer/ppo/ray_trainer.py` — adaptive prompt generation, reward, routing, pre-forward truncation, cumulative metrics, and timings.

---

### Task 1: Typed configuration and pure adaptive routing

**Files:**
- Create: `verl/verl/trainer/ppo/adaptive_concise_opd.py`
- Create: `verl/tests/trainer/ppo/test_adaptive_concise_opd.py`
- Modify: `verl/verl/trainer/config/algorithm.py`
- Modify: `verl/verl/trainer/config/ppo_trainer.yaml`
- Test: `verl/tests/trainer/config/test_algo_config_on_cpu.py`

**Interfaces:**
- Produces `AdaptiveConciseProbePlan`, `AdaptiveConciseRoutingResult`, `plan_adaptive_concise_probes(...)`, `finalize_adaptive_concise_routing(...)`, `summarize_adaptive_concise_routing(...)`, `update_adaptive_concise_cumulative_counts(...)`, and `truncate_to_adaptive_concise_prefix(...)`.
- Later tasks consume `probe_indices`, `max_tokens`, `prompt_styles`, `supervised_lengths`, and scalar metrics.

- [ ] **Step 1: Add failing tests for typed defaults and probe planning**

```python
from omegaconf import OmegaConf
import torch

from verl.trainer.config.algorithm import AdaptiveConciseOpdConfig, AlgoConfig
from verl.trainer.ppo.adaptive_concise_opd import plan_adaptive_concise_probes


def test_adaptive_config_defaults_are_disabled_and_frozen() -> None:
    config = AdaptiveConciseOpdConfig()
    assert config.enabled is False
    assert config.correct_reward_threshold == 0.5
    assert config.concise_cap_ratio == 0.5
    assert config.teacher_prompt_key == "teacher_prompt"
    assert config.temperature == 1.0
    assert config.top_p == 1.0
    assert AlgoConfig().adaptive_concise_opd.enabled is False


def test_probe_plan_only_selects_normal_correct_rows_and_caps_at_half() -> None:
    reward = torch.tensor([[0.0, 1.0], [0.0, 0.0], [0.0, 1.0]])
    mask = torch.tensor([[1, 1], [1, 1], [1, 1]])
    plan = plan_adaptive_concise_probes(
        normal_reward=reward,
        normal_response_mask=mask,
        correct_reward_threshold=0.5,
        concise_cap_ratio=0.5,
    )
    assert plan.probe_indices.tolist() == [0, 2]
    assert plan.max_tokens.tolist() == [1, 1]
    assert plan.normal_correct.tolist() == [True, False, True]
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
cd /home/mchen/FiRe-OPD/.worktrees/vanilla-sft-gradient-51200-opd
/home/mchen/miniconda3/envs/verl/bin/python -m pytest \
  verl/tests/trainer/ppo/test_adaptive_concise_opd.py \
  verl/tests/trainer/config/test_algo_config_on_cpu.py -q
```

Expected: import/attribute failures for the new config and module.

- [ ] **Step 3: Add the typed config**

Add to `algorithm.py` and export through `__all__`:

```python
@dataclass
class AdaptiveConciseOpdConfig(BaseConfig):
    """Adaptive concise-probe response-token-neutral OPD."""

    enabled: bool = False
    correct_reward_threshold: float = 0.5
    concise_cap_ratio: float = 0.5
    teacher_prompt_key: str = "teacher_prompt"
    temperature: float = 1.0
    top_p: float = 1.0
    expected_questions_per_step: int = 1024


# Add this field next to tale_budget/difficulty_aware_opd inside AlgoConfig.
adaptive_concise_opd: AdaptiveConciseOpdConfig = field(
    default_factory=AdaptiveConciseOpdConfig
)
```

Add the matching disabled-by-default YAML block with `_target_: verl.trainer.config.AdaptiveConciseOpdConfig` and exactly the defaults above.

- [ ] **Step 4: Implement immutable plan/result dataclasses and planning validation**

```python
@dataclass(frozen=True)
class AdaptiveConciseProbePlan:
    normal_correct: torch.Tensor
    hard: torch.Tensor
    normal_lengths: torch.Tensor
    probe_indices: torch.Tensor
    max_tokens: torch.Tensor


@dataclass(frozen=True)
class AdaptiveConciseRoutingResult:
    normal_correct: torch.Tensor
    easy: torch.Tensor
    learnable: torch.Tensor
    hard: torch.Tensor
    normal_lengths: torch.Tensor
    concise_lengths: torch.Tensor
    supervised_lengths: torch.Tensor
    probe_indices: torch.Tensor
    max_tokens: torch.Tensor
    prompt_styles: np.ndarray
    concise_parse_fail_count: int
    concise_cap_hit_count: int
```

`plan_adaptive_concise_probes` must validate 2-D equal-shaped finite reward/mask tensors, positive valid normal lengths, threshold `0.5`, cap ratio `0.5`, and `L_n>=2` for every probed row. It computes sequence correctness from `normal_reward.sum(-1) > threshold`, uses `torch.nonzero`, and returns `floor(0.5*L_n)` caps.

- [ ] **Step 5: Add failing route-finalization and accounting tests**

Cover this exact batch:

```python
# normal rows: lengths [8, 8, 7]
# normal correctness: [wrong, correct, correct]
# concise probe lengths [3, 2], rewards [wrong, correct]
# routes: hard, learnable, easy
# supervision: [8, 5, 5]
# prompt styles: normal, normal, concise
# concise + supervision equals normal for every row
```

Also test odd cap `floor(0.5*7)==3`, empty probe batches, non-finite rewards, duplicate/missing mappings, over-cap concise lengths, non-positive supervised lengths, and concise text without `\boxed` incrementing parse failures.

- [ ] **Step 6: Implement route finalization, metrics, and cumulative counters**

`finalize_adaptive_concise_routing` accepts a probe plan plus aligned concise reward/mask/text arrays. It fills a batch-sized zero concise-length vector, uses concise correctness to split probed rows into easy/learnable, sets hard from the plan, calculates `L_s=L_n-L_c`, and aborts unless the partition and per-row token identity are exact.

`summarize_adaptive_concise_routing` must emit all spec keys and exact identities under `adaptive_concise_opd/`, including zero residual and ratio 1.0. `update_adaptive_concise_cumulative_counts` returns a new dictionary and emits cumulative total/probe/easy/learnable/hard counts without mutating input on failure.

- [ ] **Step 7: Add failing prefix-truncation tests**

Construct a `DataProto` with response width 8, different valid lengths, `rollout_log_probs`, sequence tensors, and non-tensor teacher metadata. Assert truncation:

- slices all response-aligned tensors to max supervised width;
- changes each row's response/attention mask to the first `L_s` valid tokens;
- preserves prompt tensors and row count;
- updates `global_token_num`;
- does not retain any diagnostic response tensor.

- [ ] **Step 8: Implement fail-closed prefix truncation**

Implement `truncate_to_adaptive_concise_prefix(batch, supervised_lengths)` using explicit response-aligned and sequence-key allowlists modeled on `truncate_to_tale_budget_esr`. Build a per-row prefix mask, physically slice to `max(L_s)`, replace response attention masks, and return a fresh `DataProto`.

- [ ] **Step 9: Run Task 1 tests and commit**

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m pytest \
  verl/tests/trainer/ppo/test_adaptive_concise_opd.py \
  verl/tests/trainer/config/test_algo_config_on_cpu.py -q
git add verl/verl/trainer/config/algorithm.py \
  verl/verl/trainer/config/ppo_trainer.yaml \
  verl/verl/trainer/ppo/adaptive_concise_opd.py \
  verl/tests/trainer/ppo/test_adaptive_concise_opd.py \
  verl/tests/trainer/config/test_algo_config_on_cpu.py
git commit -m "feat: add adaptive concise OPD routing"
```

Expected: all focused tests pass.

---

### Task 2: Per-request vLLM diagnostic caps

**Files:**
- Modify: `verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`
- Create: `verl/tests/workers/rollout/rollout_vllm/test_adaptive_max_tokens.py`

**Interfaces:**
- Consumes generation kwargs `max_tokens_by_row: list[int]` and `disable_rollout_log_probs: bool`.
- Produces one `SamplingParams` per request, output padded only to `max(max_tokens_by_row)`, and no `rollout_log_probs` for diagnostic calls.

- [ ] **Step 1: Write failing CPU tests for per-request parameter construction**

```python
from vllm import SamplingParams
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import _build_per_request_sampling_params


def test_build_per_request_sampling_params_uses_exact_caps_without_mutating_base() -> None:
    base = SamplingParams(max_tokens=128, temperature=1.0, top_p=1.0, logprobs=0)
    rows = _build_per_request_sampling_params(
        base, max_tokens_by_row=[10, 31, 64], disable_rollout_log_probs=True
    )
    assert [row.max_tokens for row in rows] == [10, 31, 64]
    assert all(row.logprobs is None for row in rows)
    assert base.max_tokens == 128
    assert base.logprobs == 0
```

Add failures for booleans, zero/negative values, values above configured response length, wrong row count, native-N combination, and no-log-prob output selection.

- [ ] **Step 2: Run the focused test and confirm RED**

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m pytest \
  verl/tests/workers/rollout/rollout_vllm/test_adaptive_max_tokens.py -q
```

Expected: helper import failure.

- [ ] **Step 3: Implement the sampling helper and generation path**

Add a pure helper that deep-copies the base `SamplingParams`, validates each integer cap, assigns `max_tokens`, and sets `logprobs=None` for diagnostics.

In `vLLMRollout.generate_sequences`:

```python
max_tokens_by_row = kwargs.pop("max_tokens_by_row", None)
disable_rollout_log_probs = kwargs.pop("disable_rollout_log_probs", False)
collect_rollout_log_probs = self.config.calculate_log_probs and not disable_rollout_log_probs
```

When caps are present, require one cap per prompt and forbid native-N capture. Pass the list of `SamplingParams` directly to `LLM.generate`, which the installed vLLM API supports. Pad response tensors to `max(max_tokens_by_row)` instead of the configured 16,384 width. Only iterate and return rollout log-probs when `collect_rollout_log_probs` is true.

- [ ] **Step 4: Run vLLM and legacy rollout tests**

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m pytest \
  verl/tests/workers/rollout/rollout_vllm/test_adaptive_max_tokens.py \
  verl/tests/workers/rollout/test_opd_proxy_verify_native_n.py -q
```

Expected: all tests pass and native capture behavior is unchanged.

- [ ] **Step 5: Commit**

```bash
git add verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py \
  verl/tests/workers/rollout/rollout_vllm/test_adaptive_max_tokens.py
git commit -m "feat: support per-request diagnostic rollout caps"
```

---

### Task 3: Concise prompt construction and diagnostic orchestration helpers

**Files:**
- Modify: `verl/verl/trainer/ppo/ray_trainer.py`
- Create: `verl/tests/trainer/ppo/test_adaptive_concise_trainer.py`

**Interfaces:**
- Produces `_build_adaptive_concise_generation_batch(...)`, `_apply_adaptive_concise_teacher_prompts(...)`, and `_remap_reward_to_supervised_prefix(...)`.
- Consumes Task 1's probe plan/result and Task 2's generation kwargs.

- [ ] **Step 1: Add failing prompt-batch tests**

Build a three-row normal `DataProto` with `raw_prompt`, `reward_model`, `data_source`, `extra_info.index`, and probe indices `[0, 2]`. Assert that `_build_adaptive_concise_generation_batch`:

- returns exactly two rows;
- uses `build_concise_teacher_messages` and strips the verbose suffix;
- preserves reward metadata and stable original-row indices;
- puts `[4, 3]` into `meta_info["generation_kwargs"]["max_tokens_by_row"]`;
- sets `disable_rollout_log_probs=True`, temperature 1.0, top-p 1.0;
- does not silently truncate prompts.

- [ ] **Step 2: Run and confirm RED**

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m pytest \
  verl/tests/trainer/ppo/test_adaptive_concise_trainer.py -q
```

Expected: missing helper failures.

- [ ] **Step 3: Implement concise generation batch construction**

Reuse `_extract_single_user_question`, `_object_array`, tokenizer chat templates, `postprocess_data`, and `compute_position_id_with_mask`. Copy selected non-tensor reward metadata and add `adaptive_concise_original_row`. Set generation kwargs exactly as tested. Reject duplicate source indices, empty/mismatched caps, and prompt truncation.

- [ ] **Step 4: Add and implement teacher-prompt routing tests**

For route styles `normal, normal, concise`, assert:

- hard and learnable copy the original raw prompt exactly;
- easy uses the existing concise prompt builder;
- all teacher prompts are stored under the configured `teacher_prompt` key;
- the normal student prompt tensors are unchanged.

- [ ] **Step 5: Add and implement reward remapping tests**

`_remap_reward_to_supervised_prefix` must preserve each sequence reward sum while moving it to `L_s-1` in the shortened response width. This prevents correct normal rewards from disappearing merely because their original terminal reward lies outside the retained prefix.

- [ ] **Step 6: Run focused tests and commit**

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m pytest \
  verl/tests/trainer/ppo/test_adaptive_concise_trainer.py \
  verl/tests/trainer/ppo/test_tale_budget.py -q
git add verl/verl/trainer/ppo/ray_trainer.py \
  verl/tests/trainer/ppo/test_adaptive_concise_trainer.py
git commit -m "feat: build adaptive concise probe batches"
```

---

### Task 4: Integrate adaptive routing before all post-rollout model forwards

**Files:**
- Modify: `verl/verl/trainer/ppo/ray_trainer.py`
- Modify: `verl/tests/trainer/ppo/test_adaptive_concise_trainer.py`

**Interfaces:**
- Consumes all Tasks 1–3 interfaces.
- Produces a final 1,024-row normal batch with routed teacher prompts, prefix-only response masks, route metrics, cumulative metrics, and split timings.

- [ ] **Step 1: Add a source-order integration test**

Use `inspect.getsource(RayPPOTrainer.fit)` to assert this strict ordering:

```text
normal reward completion
< plan_adaptive_concise_probes
< concise generate_sequences
< finalize_adaptive_concise_routing
< truncate_to_adaptive_concise_prefix
< compute_log_prob
< prepare_ref_model_inputs
< update_actor
```

Also assert the concise diagnostic object is never unioned with the normal actor batch.

- [ ] **Step 2: Add a mocked orchestration test**

Create a fake actor-rollout worker whose normal output has four rows and whose diagnostic output has only the two normal-correct rows. Verify:

- diagnostics are called once with two rows and per-request caps;
- final row count remains four;
- route counts partition four;
- old/ref/update see only the truncated normal responses;
- diagnostic tensors and `rollout_log_probs` are absent from the actor input;
- second token balancing happens after prefix masking.

- [ ] **Step 3: Run and confirm RED**

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m pytest \
  verl/tests/trainer/ppo/test_adaptive_concise_trainer.py -q
```

Expected: source-order and orchestration failures.

- [ ] **Step 4: Implement the fit-loop adaptive block**

Immediately after normal reward becomes available and before rollout-correction/old-log-prob:

1. validate 1,024 distinct `extra_info.index` values;
2. create the probe plan;
3. time diagnostic generation as `concise_probe`;
4. compute diagnostic `response_mask`, decode valid concise texts, and time reward as `concise_reward`;
5. finalize routes and teacher prompts;
6. remap normal rewards to supervised prefix positions;
7. truncate the normal batch;
8. temporarily store remapped reward in the batch, rerun `_balance_batch(..., logging_prefix="adaptive_concise_seqlen")`, then pop it back so reward order follows the batch;
9. update `global_token_num`;
10. continue into old-log-prob/ref/update.

Set timing aliases `normal_rollout=gen` and `normal_reward=reward`. The empty-probe case skips the second rollout and records zero timings/counts without failing.

- [ ] **Step 5: Implement cumulative metrics after successful actor update**

Initialize a local cumulative-count dictionary at the start of `fit`. After `update_actor` succeeds, update cumulative route-event counters and merge them into `metrics` before logger output. Do not increment cumulative counts for a failed update.

- [ ] **Step 6: Preserve legacy paths**

Guard all new behavior behind `algorithm.adaptive_concise_opd.enabled`. Add an explicit incompatibility check with TALE, legacy difficulty routing, candidate selection, and rethinking probes. Run source-order tests for existing TALE and proxy-capture paths.

- [ ] **Step 7: Run trainer regressions and commit**

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m pytest \
  verl/tests/trainer/ppo/test_adaptive_concise_opd.py \
  verl/tests/trainer/ppo/test_adaptive_concise_trainer.py \
  verl/tests/trainer/ppo/test_tale_budget.py \
  verl/tests/trainer/ppo/test_difficulty_aware_opd.py \
  verl/tests/trainer/ppo/test_candidate_selection.py \
  verl/tests/trainer/ppo/test_rollout_corr.py \
  verl/tests/trainer/ppo/test_rollout_corr_integration.py -q
git add verl/verl/trainer/ppo/ray_trainer.py \
  verl/tests/trainer/ppo/test_adaptive_concise_trainer.py
git commit -m "feat: integrate token-neutral concise routing"
```

---

### Task 5: Runtime contract and pinned launcher

**Files:**
- Modify: `verl/verl/trainer/main_ppo.py`
- Create: `run_train_adaptive_concise_token_neutral_opd.sh`
- Create: `verl/tests/trainer/ppo/test_adaptive_concise_launcher.py`

**Interfaces:**
- Produces `validate_adaptive_concise_runtime_config(config)` and profile/full dry-run contracts.
- The launcher is the only approved GPU entrypoint.

- [ ] **Step 1: Write failing runtime-validator tests**

Compose Hydra config and assert enabled adaptive mode rejects changes to:

- batch/mini-batch 1,024, primary n=1, sync vLLM, seed 42;
- cap ratio 0.5 and expected questions 1,024;
- teacher prompt key mismatch or missing raw chat;
- async reward;
- non-reverse-KL objective, length penalty, entropy, TALE, legacy routing, candidate selection, rethinking, G-Vendi-related data override, or resume.

- [ ] **Step 2: Implement fail-closed runtime validation**

Add `validate_adaptive_concise_runtime_config(config)` next to the proxy validator and call it in `TaskRunner.run` before worker construction. It returns `None` when disabled and a resolved contract dictionary when enabled. It must reject simultaneous proxy capture.

- [ ] **Step 3: Write the failing launcher tests**

Test both:

```bash
ADAPTIVE_CONCISE_DRY_RUN=1 ADAPTIVE_CONCISE_RUN_MODE=profile bash run_train_adaptive_concise_token_neutral_opd.sh
ADAPTIVE_CONCISE_DRY_RUN=1 ADAPTIVE_CONCISE_RUN_MODE=full bash run_train_adaptive_concise_token_neutral_opd.sh
```

Profile must pin one step, no validation, distinct profile outputs. Full must pin 50 steps, save 50, Vanilla-matched validation, and distinct final outputs. Both must print source worktree, production data/models, adaptive config, cap 0.5, teacher key, disabled competing features, and resume disabled. Override tests must fail with exit 2.

- [ ] **Step 4: Implement the standalone launcher**

Use:

```bash
PRODUCTION_ROOT=/home/mchen/FiRe-OPD
REPO_DIR=/home/mchen/FiRe-OPD/.worktrees/vanilla-sft-gradient-51200-opd
PYTHON_BIN=/home/mchen/miniconda3/envs/verl/bin/python
```

Full experiment name:

```text
opd-adaptive-conciseprobe-cap50-tokenneutral-rawprompt-4gpu-tp4-step50
```

Profile experiment name:

```text
opd-adaptive-conciseprobe-cap50-tokenneutral-rawprompt-4gpu-tp4-profile-step1
```

Before real execution, require tokens `0,1,2,3`, all source/data/model paths, absent checkpoint/log destinations, clean worktree, no positional overrides, and `SLURM_JOB_ID`. Print the fully quoted command in dry-run mode. In real mode, the launcher itself atomically claims its fixed log path, redirects stdout/stderr there, and then `exec`s the VERL Python process; callers must not pre-create the log via shell redirection.

- [ ] **Step 5: Run config and launcher tests**

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m pytest \
  verl/tests/trainer/ppo/test_adaptive_concise_launcher.py \
  verl/tests/trainer/config/test_algo_config_on_cpu.py -q
bash -n run_train_adaptive_concise_token_neutral_opd.sh
```

- [ ] **Step 6: Commit**

```bash
git add verl/verl/trainer/main_ppo.py \
  run_train_adaptive_concise_token_neutral_opd.sh \
  verl/tests/trainer/ppo/test_adaptive_concise_launcher.py
git commit -m "feat: launch adaptive concise token-neutral OPD"
```

---

### Task 6: First-step profile validator

**Files:**
- Create: `math_eval/validate_adaptive_concise_profile.py`
- Create: `math_eval/test_validate_adaptive_concise_profile.py`

**Interfaces:**
- CLI consumes `--log PATH --expected-step 1 --expected-questions 1024 --output PATH`.
- Produces immutable JSON with parsed metrics and `decision: pass|fail`; exits nonzero on failure.

- [ ] **Step 1: Write failing parser/gate tests**

Use synthetic full step lines and assert pass requires:

```text
training/global_step = 1
adaptive_concise_opd/total_questions = 1024
partition and probe identities exact
response_budget_residual_tokens = 0
response_budget_ratio = 1.0
actor/pg_loss finite
actor/grad_norm finite and > 0
rollout-correction maximum IS <= 5.0
all required timing fields finite and > 0 except allowed zero concise timing when probe_count=0
```

Assert missing, duplicate, NaN, overflow, count mismatch, and token mismatch all fail.

- [ ] **Step 2: Implement strict parsing and atomic report creation**

Parse the final matching `step:1 - key:value` line without depending on W&B truncation. Refuse an existing output report. Write via a same-directory temporary file followed by `os.replace`. Include log SHA256 and source commit.

- [ ] **Step 3: Run tests and commit**

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m pytest \
  math_eval/test_validate_adaptive_concise_profile.py -q
git add math_eval/validate_adaptive_concise_profile.py \
  math_eval/test_validate_adaptive_concise_profile.py
git commit -m "feat: validate adaptive concise profile"
```

---

### Task 7: Full CPU verification and source freeze

**Files:**
- Verify all modified/new files.

- [ ] **Step 1: Run focused adaptive suite**

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m pytest \
  verl/tests/trainer/ppo/test_adaptive_concise_opd.py \
  verl/tests/trainer/ppo/test_adaptive_concise_trainer.py \
  verl/tests/trainer/ppo/test_adaptive_concise_launcher.py \
  verl/tests/workers/rollout/rollout_vllm/test_adaptive_max_tokens.py \
  math_eval/test_validate_adaptive_concise_profile.py -q
```

- [ ] **Step 2: Run legacy regression suite**

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m pytest \
  verl/tests/trainer/ppo/test_vanilla_opd_launcher.py \
  verl/tests/trainer/ppo/test_group_success_launcher.py \
  verl/tests/trainer/ppo/test_tale_budget.py \
  verl/tests/trainer/ppo/test_difficulty_aware_opd.py \
  verl/tests/trainer/ppo/test_candidate_selection.py \
  verl/tests/trainer/ppo/test_rollout_corr.py \
  verl/tests/trainer/ppo/test_rollout_corr_integration.py \
  verl/tests/trainer/config/test_algo_config_on_cpu.py \
  verl/tests/workers/rollout/test_opd_proxy_verify_native_n.py -q
```

- [ ] **Step 3: Run static checks**

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m ruff check \
  verl/verl/trainer/ppo/adaptive_concise_opd.py \
  verl/verl/trainer/ppo/ray_trainer.py \
  verl/verl/trainer/main_ppo.py \
  verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py \
  verl/tests/trainer/ppo/test_adaptive_concise_opd.py \
  verl/tests/trainer/ppo/test_adaptive_concise_trainer.py \
  verl/tests/trainer/ppo/test_adaptive_concise_launcher.py \
  verl/tests/workers/rollout/rollout_vllm/test_adaptive_max_tokens.py \
  math_eval/validate_adaptive_concise_profile.py \
  math_eval/test_validate_adaptive_concise_profile.py
/home/mchen/miniconda3/envs/verl/bin/python -m compileall -q \
  verl/verl/trainer/ppo/adaptive_concise_opd.py \
  verl/verl/trainer/ppo/ray_trainer.py \
  verl/verl/trainer/main_ppo.py \
  math_eval/validate_adaptive_concise_profile.py
bash -n run_train_adaptive_concise_token_neutral_opd.sh
git diff --check
git status --porcelain=v1 --untracked-files=all
```

Expected: all tests/checks pass and the worktree is clean.

- [ ] **Step 4: Record the frozen source commit and dry-run contracts**

```bash
ADAPTIVE_CONCISE_DRY_RUN=1 ADAPTIVE_CONCISE_RUN_MODE=profile \
  bash run_train_adaptive_concise_token_neutral_opd.sh > /tmp/adaptive_concise_profile.contract
ADAPTIVE_CONCISE_DRY_RUN=1 ADAPTIVE_CONCISE_RUN_MODE=full \
  bash run_train_adaptive_concise_token_neutral_opd.sh > /tmp/adaptive_concise_full.contract
sha256sum /tmp/adaptive_concise_profile.contract /tmp/adaptive_concise_full.contract
git rev-parse HEAD
git status --porcelain=v1 --untracked-files=all
```

---

### Task 8: GPU profile gate and full launch in `opd-CLI`

**Files:**
- Runtime outputs under `/home/mchen/FiRe-OPD/logs/adaptive_concise_opd/`.
- Checkpoints under `/home/mchen/FiRe-OPD/checkpoints/`.

- [ ] **Step 1: Verify the allocation and tmux gate**

From outside tmux, inspect but do not create a nested allocation:

```bash
tmux has-session -t opd-CLI
tmux list-panes -t opd-CLI -F '#{pane_id} #{pane_current_command} #{pane_dead}'
squeue -j "$SLURM_JOB_ID" -o '%i %T %L %N'
```

Inside `opd-CLI`, require `SLURM_JOB_ID`, `SLURM_STEP_GPUS`/`CUDA_VISIBLE_DEVICES` mapping to four allocation-owned tokens, idle GPU memory, and enough remaining time for profile plus the estimated full run. Abort without launching if the gate fails.

- [ ] **Step 2: Launch the one-step profile without nested `srun`**

Send this command to `opd-CLI`:

```bash
cd /home/mchen/FiRe-OPD/.worktrees/vanilla-sft-gradient-51200-opd
export CUDA_VISIBLE_DEVICES=0,1,2,3
export ADAPTIVE_CONCISE_RUN_MODE=profile
bash run_train_adaptive_concise_token_neutral_opd.sh
```

Wait conditionally for process completion. Do not infer completion from sleeps alone.

- [ ] **Step 3: Validate and archive the profile decision**

```bash
/home/mchen/miniconda3/envs/verl/bin/python \
  math_eval/validate_adaptive_concise_profile.py \
  --log /home/mchen/FiRe-OPD/logs/adaptive_concise_opd/profile-step1.log \
  --expected-step 1 \
  --expected-questions 1024 \
  --output /home/mchen/FiRe-OPD/logs/adaptive_concise_opd/profile-step1.acceptance.json
```

Require `decision=pass`, inspect route counts/timings, verify no output overwrite, and confirm worktree/source commit unchanged.

- [ ] **Step 4: Launch the full 50-step training**

Only after profile pass, send to `opd-CLI`:

```bash
cd /home/mchen/FiRe-OPD/.worktrees/vanilla-sft-gradient-51200-opd
export CUDA_VISIBLE_DEVICES=0,1,2,3
export ADAPTIVE_CONCISE_RUN_MODE=full
bash run_train_adaptive_concise_token_neutral_opd.sh
```

The launcher itself must reject existing log/checkpoint paths.

- [ ] **Step 5: Verify live startup and first full-training step**

Conditionally wait for Ray workers, model initialization, W&B run creation, and the first `step:1` metrics. Verify:

- 1,024 questions and 1,024 final actor rows;
- exact easy/learnable/hard partition;
- probe count equals easy plus learnable;
- zero response-token residual and ratio 1.0;
- finite nonzero loss/gradient;
- token IS remains at or below 5.0;
- `timing_s/concise_probe` and all post-rollout timings are logged;
- source commit and worktree remain frozen.

Leave the accepted full run active in `opd-CLI` and report the experiment name, PID/Ray job, W&B run ID when available, log path, checkpoint path, source commit, first-step route counts, token identity, and timings.
