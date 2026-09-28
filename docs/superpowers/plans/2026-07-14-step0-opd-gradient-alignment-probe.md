# Step-0 OPD Gradient Alignment Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a fail-closed diagnostic that measures the existing gradient-diverse DeepMath coreset and its exact seed-42 random control in the real Step-0 Qwen3-4B OPD gradient space.

**Architecture:** Prepare one deterministic 512-question probe parquet and an eight-question smoke subset. Capture the unchanged production rollout/teacher/ESR tensors in two independent trainer/vLLM launches, replay four trajectories sequentially for each prompt with the pristine Qwen3-4B actor, project each FP32 full group gradient to 1,024 dimensions, then compute the preregistered diversity, reliability, alignment, and signal-strength report. Runtime stages are resumable and atomic, but every provenance or numerical mismatch fails instead of overwriting or falling back.

**Tech Stack:** Python 3.10, PyTorch 2.6.0, transformers 4.51.1, veRL/Ray/vLLM, safetensors, PyArrow, NumPy/SciPy, TRAK 0.3.2, fast-jl 0.1.3, pytest, Bash, Slurm/tmux.

## Global Constraints

- Treat `docs/superpowers/specs/2026-07-14-step0-opd-gradient-alignment-probe-design.md` at commit `8833372` as the scientific contract.
- Use pristine `/home/mchen/FiRe-OPD/models/Qwen3-4B`; never use a step-50 checkpoint.
- Compare exactly 256 selected and 256 exact seed-42 random stable IDs; all 512 IDs must be unique.
- Capture every ID with two separately initialized vLLM engines, seeds `42` and `43`, and exactly four rollout slots per seed.
- Production capture is two 256-prompt steps per seed; smoke capture is one eight-prompt step per seed with four IDs from each arm.
- Keep actor learning rate `0`, parameters unchanged, W&B/checkpointing/validation disabled, and candidate selection plus length-aware OPD disabled.
- Preserve the production reverse-KL-only vanilla PPO loss, token-mean mask, token rollout-IS threshold `5.0`, easy `4/4 -> concise ESR20`, and all other groups `normal ESR50`.
- Replay four trajectories through four sequential forward/backward calls, each loss scaled by `1/4`; never build a batch-of-four graph or retain four full gradients.
- Load FP32 master parameters, run BF16 autocast forward, and accumulate FP32 `.grad` buffers across all four backwards.
- Project the full `4,022,468,096`-parameter gradient with exact `CudaProjector`, Rademacher seed `0`, dimension `1,024`, `model_id=0`, and divide by `sqrt(1024)`; do not use `BasicProjector`, last-block, LoRA, or another proxy.
- The current fast-jl API requires one `(1, 4_022_468_096)` FP32 staging buffer (about 14.985 GiB). Stream-copy parameters into that buffer in registration order; do not claim an equivalent chunk-wise fast-JL operator.
- Capture uses `/home/mchen/miniconda3/envs/verl/bin/python`; replay/projection and analysis use `/home/mchen/miniconda3/envs/gvendi-opd/bin/python`. Record both environments' package versions.
- Run GPU commands only from the existing `opd-CLI` tmux session inside a Slurm allocation with exactly four unique `CUDA_VISIBLE_DEVICES` tokens. Do not invoke `srun` or substitute host physical GPU indices.
- Store runtime outputs only under ignored `data/opd_gradient_alignment/` and `logs/opd_gradient_alignment/`; do not overwrite existing training, gradient, checkpoint, or evaluation artifacts.
- Preserve all unrelated dirty-worktree changes and stage only files belonging to the current task.

---

## File Structure

- Create `math_eval/prepare_opd_gradient_alignment.py`: pinned population reconstruction, sampling, probe parquet, token-length metadata, and preparation CLI.
- Create `math_eval/test_prepare_opd_gradient_alignment.py`: deterministic sampling/hash/schema tests.
- Modify `verl/verl/workers/config/rollout.py`: typed vLLM construction seed with default `0`.
- Modify `verl/verl/trainer/config/rollout/rollout.yaml`: documented rollout seed default.
- Modify `verl/verl/trainer/config/algorithm.py`: typed, default-off capture configuration.
- Modify `verl/verl/trainer/config/ppo_trainer.yaml`: default-off capture block.
- Modify `verl/tests/trainer/config/test_algo_config_on_cpu.py`: capture config composition tests.
- Create `verl/tests/workers/config/test_rollout_config_on_cpu.py`: seed default/override tests.
- Create `verl/verl/trainer/ppo/opd_gradient_alignment_capture.py`: contract validation, diagnostics, atomic capture artifacts, parameter-shard hashing, and finalization.
- Modify `verl/verl/trainer/main_ppo.py`: capture-only ownership and teardown of the local Ray runtime.
- Modify `verl/verl/trainer/ppo/ray_trainer.py`: opt-in deterministic identities, driver capture, actor context, and finalization.
- Modify `verl/verl/workers/actor/dp_actor.py`: opt-in update-time diagnostics and rank-local capture.
- Create `verl/tests/trainer/ppo/test_opd_gradient_alignment_capture.py`: pure diagnostic, identity, atomicity, and disabled-path tests.
- Create `run_capture_step0_opd_gradient_alignment_seed.sh`: one fresh trainer/vLLM capture process for one mode/seed.
- Create `run_capture_step0_opd_gradient_alignment.sh`: ordered seed-42 then seed-43 capture wrapper.
- Create `verl/tests/trainer/ppo/test_opd_gradient_alignment_capture_launcher.py`: dry-run launcher contract tests.
- Create `math_eval/opd_gradient_alignment.py`: replay artifact types, joins, manifests, group validation, parameter layout, and atomic group output.
- Create `math_eval/collect_opd_gradient_sketches.py`: production-compatible forward/loss, sequential backward, full-gradient projection, replay CLI, and global validator.
- Create `math_eval/test_opd_gradient_alignment.py`: shared artifact/sharding/resume tests.
- Create `math_eval/test_collect_opd_gradient_sketches.py`: replay loss/gradient/projector tests.
- Create `math_eval/analyze_opd_gradient_alignment.py`: fixed resampling tables, null-state kernel, Vendi, block CKA, statistical gates, JSON and Markdown reports.
- Create `math_eval/test_analyze_opd_gradient_alignment.py`: synthetic aligned/noisy/zero/degenerate statistical tests.
- Create `math_eval/opd_gradient_alignment_runtime.py`: allocation-aware logical-GPU validation and smoke ETA gate.
- Create `run_step0_opd_gradient_alignment_probe.sh`: top-level, locked, resumable `opd-CLI` orchestrator.
- Create `math_eval/test_opd_gradient_alignment_launcher.py`: fake-runtime stage-order, CVD, failure, and resume tests.

### Task 1: Deterministic Probe Dataset and Provenance

**Files:**
- Create: `math_eval/prepare_opd_gradient_alignment.py`
- Create: `math_eval/test_prepare_opd_gradient_alignment.py`

**Interfaces:**
- Produces: `ProbeRow`, `reconstruct_random_population()`, `sample_probe_rows()`, `write_probe_artifacts()`, and the CLI entry point.
- Produces: `data/opd_gradient_alignment/input/probe_512.parquet`, `probe_smoke_8.parquet`, `sample_manifest.jsonl`, and `preparation.manifest.json`.
- Consumes later: stable IDs, source row indices, arms, capture row indices, deterministic group UIDs, topic/difficulty, and Qwen3-tokenized R1 lengths.

- [ ] **Step 1: Write failing reconstruction and sampling tests**

Test the production PyTorch random permutation and the exact prompt hash contract:

```python
assert Version(torch.__version__).base_version == "2.6.0"
indices = reconstruct_random_population(
    source_rows=57_046,
    seed=42,
    population_size=12_800,
)
assert len(indices) == 12_800
assert indices[:5] == [54434, 23328, 20314, 25544, 20200]
assert sha256_compact_json(indices) == (
    "e32287327ee446433397d0412ecaae81724b0d7391ac42b29a13959e260aaaa5"
)
assert ordered_prompt_sha256(source, indices) == (
    "f16e847834161d68fd590b4bc98356f78740fb8406a85e71ed73e3a4b5d458bc"
)
assert 38_794 not in indices
```

`ordered_prompt_sha256()` must canonicalize each Arrow `prompt` with
`json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`,
hash an unsigned eight-byte big-endian payload length, then the payload bytes,
in random-population order.

Add a collision fixture and assert selected-arm priority, deterministic random
replacement, global uniqueness, `256/256` arm counts, and `S0,R0,S1,R1,...`
interleaving so each 256-row production batch contains `128/128` arms.
For compact-JSON source-index lists, pin the realized sample hashes:

```text
selected 256:    0912cc802d62b0860329d12ca1c8c569f1e2b46e57132a8a3796b8538b72ff84
random 256:      ccb32f0cd0e31f91d85840210e078b5afcc0ec9b4a889c9bf8d4d5f6f657d6ea
interleaved 512: a4d4dc9b797c518a647791041f4aa130a1dd7ce2b09c3f36efdc4de7aaa4cba0
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  math_eval/test_prepare_opd_gradient_alignment.py
```

Expected: import failure because `prepare_opd_gradient_alignment.py` does not exist.

- [ ] **Step 3: Implement the pinned populations and sampling**

Use these public shapes and the exact RNG order:

```python
@dataclass(frozen=True)
class ProbeRow:
    capture_row_index: int
    stable_id: str
    source_row_index: int
    arm: Literal["selected", "random"]
    probe_group_uid: str
    topic: str | None
    difficulty: float | None
    r1_solution_tokens_qwen3_4b: int

def reconstruct_random_population(
    *,
    source_rows: int,
    seed: int,
    population_size: int = 12_800,
) -> list[int]:
    if Version(torch.__version__).base_version != "2.6.0":
        raise RuntimeError(f"frozen random population requires torch 2.6.0, got {torch.__version__}")
    if not 0 < population_size <= source_rows:
        raise ValueError("population_size must be in [1, source_rows]")
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(source_rows, generator=generator)[:population_size].tolist()

def sample_probe_rows(
    selected_population: Sequence[Mapping[str, Any]],
    random_source_indices: Sequence[int],
    prepared_by_source_index: Mapping[int, Mapping[str, Any]],
    *,
    seed: int = 20260714,
    per_arm: int = 256,
    random_population_size: int = 12_800,
) -> list[ProbeRow]:
    if len(random_source_indices) != random_population_size:
        raise ValueError("random control must be the exact frozen population")
    rng = np.random.Generator(np.random.PCG64(seed))
    selected_order = rng.permutation(len(selected_population))
    random_order = rng.permutation(len(random_source_indices))
    # Draw selected first; scan random_order and skip only realized ID collisions.
```

The production path must pass the exact 12,800-entry prefix, validate its
pinned compact-index and ordered-prompt hashes before sampling, and reject a
longer full permutation. Small tests may override `random_population_size` but
must exercise the same length check. Pin a version test that accepts the actual
`2.6.0+cu124` build string, rejects `2.6.1` and `2.7.0`, and records the full
unmodified `torch.__version__` in provenance.

Define R1 length as
`len(Qwen3-4B_tokenizer(completion, add_special_tokens=False).input_ids)` and
record hashes for every tokenizer file used. Set `probe_group_uid` to
`f"opd-gradient-{stable_id}"`; it must be identical across rollout seeds.

- [ ] **Step 4: Write atomic parquet and manifest outputs**

Copy source rows in probe order and append top-level columns
`probe_capture_row_index`, `probe_stable_id`, `probe_arm`, and
`probe_group_uid`. Preserve every original column value. Build the smoke
parquet from the first four selected and first four random rows, interleaved.
Write temporary siblings, fsync, read back and validate, then `os.replace()`.
The manifest must bind source/prepared/selected-ID bytes, repository commit and
dirty-diff hash, torch version, tokenizer hashes, RNG implementation, all
ordered IDs, and every output SHA256.

- [ ] **Step 5: Run tests and the real CPU preparation preflight**

Run:

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  math_eval/test_prepare_opd_gradient_alignment.py

/home/mchen/miniconda3/envs/verl/bin/python -m math_eval.prepare_opd_gradient_alignment \
  --source-parquet data/g-opd/DeepMath-103K/train_filtered_level6.parquet \
  --prepared-jsonl data/gradient_diversity/deepmath_level6_r1_solution1.jsonl \
  --selected-ids data/gradient_diversity/selection/selected_ids.jsonl \
  --tokenizer models/Qwen3-4B \
  --output-dir data/opd_gradient_alignment/input \
  --dry-run
```

Expected: tests pass; dry-run reports `512` unique IDs, `256/256` arms,
`collision_count=0` for the current data, both pinned hashes, and writes nothing.

- [ ] **Step 6: Commit the preparation component**

```bash
git add math_eval/prepare_opd_gradient_alignment.py \
  math_eval/test_prepare_opd_gradient_alignment.py
git commit -m "Add deterministic OPD gradient probe sampling"
```

### Task 2: Typed Seed/Capture Config and Exact Diagnostic Algebra

**Files:**
- Modify: `verl/verl/workers/config/rollout.py`
- Modify: `verl/verl/trainer/config/rollout/rollout.yaml`
- Modify: `verl/verl/trainer/config/algorithm.py`
- Modify: `verl/verl/trainer/config/ppo_trainer.yaml`
- Modify: `verl/tests/trainer/config/test_algo_config_on_cpu.py`
- Create: `verl/tests/workers/config/test_rollout_config_on_cpu.py`
- Create: `verl/verl/trainer/ppo/opd_gradient_alignment_capture.py`
- Create: `verl/tests/trainer/ppo/test_opd_gradient_alignment_capture.py`

**Interfaces:**
- Produces: `OpdGradientAlignmentCaptureConfig` and `RolloutConfig.seed`.
- Produces: `VanillaPpoTokenDiagnostics`, `validate_capture_contract()`, `compute_vanilla_ppo_token_diagnostics()`, and atomic artifact primitives.
- Produces: `validate_capture_seed()` for exact step/rank/trajectory coverage and a seed-level complete marker.
- Preserves: current `compute_policy_loss_vanilla()` as the only loss used for production backward.

- [ ] **Step 1: Write failing config tests**

Assert `RolloutConfig.seed == 0` by default and Hydra override `seed=43` survives
dataclass conversion. Assert the capture block defaults disabled and accepts:

```python
OpdGradientAlignmentCaptureConfig(
    enabled=True,
    mode="production",
    output_dir="/tmp/capture",
    sample_manifest="/tmp/sample.jsonl",
    sample_manifest_sha256="a" * 64,
    rollout_seed=42,
    expected_steps=2,
    expected_groups_per_step=256,
    expected_rollouts_per_group=4,
    expected_world_size=4,
)
```

- [ ] **Step 2: Run config tests and verify RED**

Run:

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  verl/tests/workers/config/test_rollout_config_on_cpu.py \
  verl/tests/trainer/config/test_algo_config_on_cpu.py -k 'gradient_alignment or rollout_seed'
```

Expected: missing field/class failures.

- [ ] **Step 3: Add default-preserving typed configuration**

Add `seed: int = 0` beside rollout sampling fields. Add and export:

```python
@dataclass
class OpdGradientAlignmentCaptureConfig(BaseConfig):
    enabled: bool = False
    mode: str = "production"
    output_dir: Optional[str] = None
    sample_manifest: Optional[str] = None
    sample_manifest_sha256: Optional[str] = None
    rollout_seed: Optional[int] = None
    expected_steps: int = 2
    expected_groups_per_step: int = 256
    expected_rollouts_per_group: int = 4
    expected_world_size: int = 4
    schema_version: int = 1
```

Attach it to `AlgoConfig` as `opd_gradient_alignment_capture`. Mirror both new
defaults in YAML. Do not change the preexisting effective vLLM seed: the sync
engine already used `config.get("seed", 0)`.

- [ ] **Step 4: Write failing loss/mask/atomicity tests**

Cover exact mask multiplication, stopped advantage, token IS, unclipped/PPO
clip/dual-clip branches, zero-mask rejection, finite checks, canonical JSON,
atomic pair publication, provenance-exact resume, and crash-temp recovery. A
published final pair or `COMPLETE.json` mismatch is fatal. A hidden,
schema-named temp owned by the current shard lock is never treated as complete;
after validating its path/ownership it may be removed and recomputed. Unknown,
unlocked, or published partial artifacts remain fatal.
Use a synthetic production batch to assert:

```python
diagnostics = compute_vanilla_ppo_token_diagnostics(
    old_log_prob=current.detach(),
    current_log_prob=current,
    stopped_advantage=(ref - current.detach()).detach(),
    loss_response_mask=response_mask * esr_mask,
    rollout_is_weights=is_weights,
    actor_config=actor_config,
    loss_agg_mode="token-mean",
)
pg_loss, _ = compute_policy_loss_vanilla(...)
torch.testing.assert_close(diagnostics.trajectory_pg_loss, pg_loss.detach())
assert not diagnostics.stopped_advantage.requires_grad
```

- [ ] **Step 5: Implement the pure capture helpers**

Use this result shape:

```python
@dataclass(frozen=True)
class VanillaPpoTokenDiagnostics:
    current_log_prob: torch.Tensor
    stopped_advantage: torch.Tensor
    ratio: torch.Tensor
    clip_branch: torch.Tensor
    weighted_token_loss: torch.Tensor
    trajectory_pg_loss: torch.Tensor
```

Reproduce `core_algos.compute_policy_loss_vanilla()` under `torch.no_grad()`
for diagnostics and assert its scalar against the real function. Keep the real
actor backward call untouched. Implement `atomic_save_tensor_json_pair()` with
safetensors plus strict canonical JSON, file and directory fsync, and a final
`COMPLETE.json` only after all expected files/hash/identity checks pass.

- [ ] **Step 6: Run focused tests and commit**

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  verl/tests/workers/config/test_rollout_config_on_cpu.py \
  verl/tests/trainer/config/test_algo_config_on_cpu.py \
  verl/tests/trainer/ppo/test_opd_gradient_alignment_capture.py \
  -k 'config or diagnostics or mask or atomic or resume'

git add verl/verl/workers/config/rollout.py \
  verl/verl/trainer/config/rollout/rollout.yaml \
  verl/verl/trainer/config/algorithm.py \
  verl/verl/trainer/config/ppo_trainer.yaml \
  verl/verl/trainer/ppo/opd_gradient_alignment_capture.py \
  verl/tests/workers/config/test_rollout_config_on_cpu.py \
  verl/tests/trainer/config/test_algo_config_on_cpu.py \
  verl/tests/trainer/ppo/test_opd_gradient_alignment_capture.py
git commit -m "Add exact OPD gradient capture primitives"
```

### Task 3: Opt-In Driver and Actor Capture Integration

**Files:**
- Modify: `verl/verl/trainer/main_ppo.py`
- Modify: `verl/verl/trainer/ppo/ray_trainer.py`
- Modify: `verl/verl/workers/actor/dp_actor.py`
- Modify: `verl/tests/trainer/ppo/test_opd_gradient_alignment_capture.py`

**Interfaces:**
- Consumes: typed capture config and helpers from Task 2.
- Produces: one driver tensor/JSON pair, four actor-rank pairs, and one complete marker per step.
- Produces: capture-only `ray.shutdown()` for the local runtime created by that driver.
- Leaves disabled training control flow, imports, metrics, and outputs unchanged.

- [ ] **Step 1: Add failing identity and disabled-path tests**

Test deterministic `probe_group_uid`, rollout slots `{0,1,2,3}` after arbitrary
batch reorder, and one-to-one joins over
`stable_id/group_uid/source-index/seed/slot`; missing/duplicate rank/identity/
slot failures are fatal. Include a regression fixture where `index` exists only
as a Python `int` array in `batch.non_tensor_batch`, proving no code reads
`batch.batch["index"]`. Patch Python import and file open functions so the
disabled test proves no capture helper import or write.
Mock Ray and assert a capture driver that created its own local runtime always
calls `ray.shutdown()` in `finally`, on both success and exception; a
preinitialized/external Ray runtime and capture-disabled training are untouched.

- [ ] **Step 2: Run the integration-focused tests and verify RED**

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  verl/tests/trainer/ppo/test_opd_gradient_alignment_capture.py \
  -k 'identity or disabled or finalize or factor'
```

Expected: failures because the trainer and actor have no capture path.

- [ ] **Step 3: Add the driver-side hook immediately before actor update**

When enabled only:

```python
batch.non_tensor_batch["uid"] = np.asarray(
    batch.non_tensor_batch["probe_group_uid"], dtype=object
)
# After n-way interleaved repeat, before any reorder:
batch.batch["opd_capture_rollout_slot"] = torch.arange(len(batch)) % 4
batch.batch["opd_capture_source_index"] = torch.as_tensor(
    np.asarray(batch.non_tensor_batch["index"], dtype=np.int64),
    dtype=torch.long,
    device=batch.batch["responses"].device,
)
```

After teacher scoring, truncation, rollout correction, and advantages, but
before `update_actor(batch)`, lazily import the capture module, validate the
complete frozen contract, write driver artifacts, and set only a small
serializable context in `batch.meta_info["opd_gradient_alignment_capture"]`.
The context includes output step directory, mode, seed, step, expected local
trajectories/gradient accumulation, world size, and sample-manifest hash.

- [ ] **Step 4: Add rank-local update-time actor capture**

Only when the meta context exists, include the two identity tensors in
`select_keys`, construct an `ActorUpdateCapture`, and append after the actual
on-policy anchor and real PPO loss are known:

```python
if on_policy:
    old_log_prob = log_prob.detach()
if self.config.policy_loss.only_reverse_kl_advantages:
    advantages = -(old_log_prob - model_inputs["ref_log_prob"])

pg_loss, pg_metrics = policy_loss_fn(...)
diagnostics = compute_vanilla_ppo_token_diagnostics(...)
torch.testing.assert_close(
    diagnostics.trajectory_pg_loss,
    pg_loss.detach(),
    atol=1e-7,
    rtol=1e-6,
)
actor_capture.append_microbatch(...)
micro_batch_metrics["opd_gradient_alignment/objective_pg_loss"] = (
    pg_loss.detach().item()
)
```

Keep the current scaled metric exactly as-is. After optimizer clipping, save
`clip_factor = min(1.0, grad_clip / (float(grad_norm) + 1e-6))`. Hash local
FSDP parameter shards in deterministic named order before and after each LR-zero
step without calling `full_tensor()`; require equality for every rank/step.

The driver safetensors artifact must contain capture/source/slot identities,
actor `input_ids`, `responses`, `attention_mask`, `position_ids`, final
`response_mask`, ESR loss mask, reference/old/rollout log-probs, rollout-IS
weights, verifier scores, original response lengths, ESR beta, and all routing
masks/counts. Its JSON rows contain stable ID, UID, arm, topic, difficulty, R1
token length, actual teacher prompt messages/style, route, token counts, and
verifier result. Each actor-rank safetensors artifact contains identities,
update-time current log-prob, stopped advantage, exact loss mask, ratio, clip
branch, weighted token loss, trajectory loss, loss scale, and backward loss.
Rank JSON stores local objective/logged loss, pre-clip norm, clip factor,
parameter hashes, model/tokenizer/config hashes, package versions, and file
hashes. Each actor-rank JSON additionally stores one canonical identity row per
local trajectory with `stable_id`, `probe_group_uid`, source index, rollout
seed, rollout slot, and tensor-row offset. Carry the stable-ID/UID arrays into
the actor update alongside the numeric identity tensors, and require exact
row-wise agreement among driver JSON, driver tensors, actor JSON, and actor
tensors during finalization. Tensor artifacts use safetensors; no pickle
serialization is permitted.
The real on-policy capture must additionally assert `ratio == 1` elementwise on
the loss mask and zero PPO/dual-clip branch selections.

In `main_ppo.run_ppo()`, record whether this capture process itself initialized
Ray. Wrap task-runner creation/execution and optional timeline emission in
`try/finally`; when and only when capture is enabled and this process owns that
runtime, call `ray.shutdown()` and wait for the owned runtime to release its
children. Do not shut down a preexisting or external Ray connection.

- [ ] **Step 5: Pin logged versus objective loss aggregation**

Finalization must calculate the local accumulation factor from the capture
contract and verify:

```python
logged = actor_output_metrics["actor/pg_loss"]
objective = actor_output_metrics["opd_gradient_alignment/objective_pg_loss"]
assert math.isclose(
    objective,
    expected_gradient_accumulation * logged,
    rel_tol=1e-6,
    abs_tol=1e-7,
)
```

Production requires factor `256`; smoke requires factor `8`. Persist both
values rather than treating the logged scalar as the objective. Finalize only
after all four actor rank artifacts join the driver rows exactly.

- [ ] **Step 6: Run actor/trainer regressions and commit**

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  verl/tests/trainer/ppo/test_opd_gradient_alignment_capture.py \
  verl/tests/trainer/ppo/test_rollout_corr_integration.py \
  verl/tests/workers/actor/test_length_aware_opd.py

git add verl/verl/trainer/main_ppo.py \
  verl/verl/trainer/ppo/ray_trainer.py \
  verl/verl/workers/actor/dp_actor.py \
  verl/tests/trainer/ppo/test_opd_gradient_alignment_capture.py
git commit -m "Capture production OPD update tensors"
```

### Task 4: Separate-Engine Capture Launchers

**Files:**
- Create: `run_capture_step0_opd_gradient_alignment_seed.sh`
- Create: `run_capture_step0_opd_gradient_alignment.sh`
- Create: `verl/tests/trainer/ppo/test_opd_gradient_alignment_capture_launcher.py`

**Interfaces:**
- Consumes: smoke/production probe parquet plus preparation manifest.
- Produces: two independent seed directories per mode.
- Delegates directly to `verl/examples/fire_opd/run_opd_strong_to_weak_student_raw_teacher_tale_budget30b.sh`.

- [ ] **Step 1: Write failing fake-launcher tests**

Assert dry-run emits two separate trainer commands in order, seed `42` must
complete before seed `43`, and each command pins the matching vLLM and capture
seed. Assert LR zero, sequential data, four rollouts, microbatch one, logger
console-only, no validation/checkpoints/resume, and the exact routing/loss
contract. Assert an incomplete seed-42 marker prevents seed 43. Run the fake
base launcher with a competing `python3` earlier in the incoming `PATH` and
prove the capture launcher still resolves
`/home/mchen/miniconda3/envs/verl/bin/python3`.

- [ ] **Step 2: Run launcher tests and verify RED**

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  verl/tests/trainer/ppo/test_opd_gradient_alignment_capture_launcher.py
```

Expected: missing launcher failure.

- [ ] **Step 3: Implement the one-seed launcher**

Require positional `mode` in `{smoke,production}` and seed in `{42,43}`.
Production sets batch/mini-batch `256`, steps `2`; smoke sets both to `8`,
steps `1`. Pass these fixed overrides after the base launcher's defaults:

```bash
export PATH="/home/mchen/miniconda3/envs/verl/bin:${PATH}"
test "$(command -v python3)" = "/home/mchen/miniconda3/envs/verl/bin/python3"
```

Fail before model loading if that interpreter preflight does not match. Then
pass these fixed overrides after the base launcher's defaults:

```text
actor_rollout_ref.rollout.seed=<42|43>
algorithm.opd_gradient_alignment_capture.enabled=True
algorithm.opd_gradient_alignment_capture.mode=<mode>
algorithm.opd_gradient_alignment_capture.rollout_seed=<seed>
actor_rollout_ref.actor.optim.lr=0
data.shuffle=False
actor_rollout_ref.rollout.n=4
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
trainer.save_freq=-1
trainer.test_freq=-1
trainer.val_before_train=False
trainer.logger=["console"]
trainer.resume_mode=disable
```

Also pin the approved group-success, ESR, reverse-KL, IS, model, precision, and
response-length settings. Do not call `run_train_group_success_difficulty_opd.sh`
because it rejects negative save frequency.

After the trainer exits, invoke the capture module's seed validator. Production
requires two complete step directories and `2 * 256 * 4 = 2,048` trajectory
identities; smoke requires one step and `8 * 4 = 32`. Only that validator may
atomically publish `seed-<seed>/COMPLETE.json`.

- [ ] **Step 4: Implement the two-process wrapper**

```bash
bash run_capture_step0_opd_gradient_alignment_seed.sh "$mode" 42
test -f "$capture_root/seed-42/COMPLETE.json"
bash run_capture_step0_opd_gradient_alignment_seed.sh "$mode" 43
test -f "$capture_root/seed-43/COMPLETE.json"
```

If a complete seed directory has exact provenance, skip its trainer process.
If it exists with mismatched provenance, fail without deletion.

- [ ] **Step 5: Run tests/dry-run and commit**

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  verl/tests/trainer/ppo/test_opd_gradient_alignment_capture_launcher.py

OPD_CAPTURE_DRY_RUN=1 bash run_capture_step0_opd_gradient_alignment.sh production

git add run_capture_step0_opd_gradient_alignment_seed.sh \
  run_capture_step0_opd_gradient_alignment.sh \
  verl/tests/trainer/ppo/test_opd_gradient_alignment_capture_launcher.py
git commit -m "Add independent OPD capture launchers"
```

### Task 5: Exact Standalone Replay Objective

**Files:**
- Create: `math_eval/opd_gradient_alignment.py`
- Create: `math_eval/collect_opd_gradient_sketches.py`
- Create: `math_eval/test_opd_gradient_alignment.py`
- Create: `math_eval/test_collect_opd_gradient_sketches.py`

**Interfaces:**
- Produces: `ReplayGroupKey`, `CapturedTrajectory`, `CapturedGroup`, `ParameterLayout`, artifact loaders, and `replay_group()`.
- Consumes: complete capture step manifests only.
- Uses: existing `compute_policy_loss_vanilla()` for the differentiable replay loss.

- [ ] **Step 1: Write failing artifact and group-validation tests**

Test exactly four distinct slots, matching stable ID/seed/step, exact
`response_mask * ESR mask`, one-to-one driver/actor join, finite tensors,
capture hash binding, and prompt sharding that keeps both seeds together:

```python
assert contiguous_prompt_shard(512, 4, 0) == range(0, 128)
assert contiguous_prompt_shard(512, 4, 3) == range(384, 512)
```

- [ ] **Step 2: Define shared replay types and strict loaders**

Implement the approved dataclasses and these public helpers:

```python
def load_captured_groups(
    capture_root: Path,
    sample_manifest: Path,
) -> list[CapturedGroup]: ...

def build_parameter_layout(model: nn.Module) -> ParameterLayout: ...
def validate_captured_group(group: CapturedGroup) -> None: ...
def contiguous_prompt_shard(total: int, count: int, index: int) -> range: ...
def write_or_validate_replay_manifest(path: Path, expected: Mapping) -> dict: ...
def atomic_save_group_artifact(
    path: Path,
    tensors: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Any],
) -> None: ...
```

Use `model.named_parameters(remove_duplicate=True)` registration order, not a
sorted mapping. Store name, shape, numel, offset, total numel, and canonical
layout SHA256. Reject any capture directory lacking its final complete marker.

- [ ] **Step 3: Write failing replay loss and sequential-gradient tests**

With a toy model, assert ESR-excluded token gradients are zero, reference and
detached advantage have no gradient, token IS changes the gradient, and four
sequential `(loss / 4).backward()` calls equal the arithmetic mean of four
explicit per-trajectory gradients. Count exactly four backward calls and prove
the code does not construct a simultaneous batch-of-four graph.

- [ ] **Step 4: Implement production-compatible forward and loss**

Load Qwen3-4B with FP32 weights and FlashAttention 2, enable non-reentrant
gradient checkpointing, apply the veRL remove-padding monkey patch with Ulysses
size one, set `model.train()`, and use BF16 autocast plus `use_cache=False`.
Reproduce the batch-size-one remove-padding forward from
`DataParallelPPOActor._forward_micro_batch()` using captured IDs, attention,
positions, and response tokens; never re-tokenize text.

Build the differentiable loss exactly as:

```python
old_log_prob = log_prob.detach()
advantage = (ref_log_prob.detach() - old_log_prob).detach()
loss, metrics = compute_policy_loss_vanilla(
    old_log_prob=old_log_prob,
    log_prob=log_prob,
    advantages=advantage,
    response_mask=loss_response_mask,
    loss_agg_mode="token-mean",
    config=policy_config,
    rollout_is_weights=rollout_is_weights.detach(),
)
```

Fix clip ratios to `0.2/0.2/0.2` and dual-clip `3.0`. Validate replay current
log-prob, mask, token count, trajectory loss, and stopped advantage against the
actor artifact before allowing its gradient into a group.

- [ ] **Step 5: Implement four-call group replay**

```python
model.zero_grad(set_to_none=True)
for trajectory in sorted(group.trajectories, key=lambda row: row.rollout_slot):
    result = compute_step0_opd_loss(...)
    (result.loss / 4.0).backward()
    del result
# Project only after the fourth backward; clear only after projection.
```

The stored group gradient is unscaled by production `1/256`. Record four
trajectory losses/effective-token counts and replay-versus-capture errors.

- [ ] **Step 6: Run CPU tests and commit replay core**

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest -q \
  math_eval/test_opd_gradient_alignment.py \
  math_eval/test_collect_opd_gradient_sketches.py \
  -m 'not gpu' -k 'group or mask or loss or backward or layout or artifact'

git add math_eval/opd_gradient_alignment.py \
  math_eval/collect_opd_gradient_sketches.py \
  math_eval/test_opd_gradient_alignment.py \
  math_eval/test_collect_opd_gradient_sketches.py
git commit -m "Add exact Step-0 OPD gradient replay"
```

### Task 6: FP32 Fast-JL Projection, Resumable Collection, and Validation

**Files:**
- Modify: `math_eval/collect_opd_gradient_sketches.py`
- Modify: `math_eval/test_collect_opd_gradient_sketches.py`
- Modify: `math_eval/test_opd_gradient_alignment.py`

**Interfaces:**
- Produces: `FullGradientFastJLProjector`, `run_collection()`, `run_global_validation()`, and `--projection-smoke-only`.
- Produces: `--single-trajectory-smoke`, which deterministically chooses the shortest valid captured smoke trajectory and checks real Qwen3 log-prob/loss before any full-group replay.
- Produces: one atomic safetensors record per prompt/seed realization and a globally validated replay manifest.

- [ ] **Step 1: Write failing projector and resume tests**

With an injected small fake model/projector, assert parameter order, FP32 norm,
FP32 staging input, seed/dimension/scaling, finite/non-finite behavior, exact
zero acceptance, missing gradient failure, per-ID two-seed colocation, partial
temp refusal, exact resume preserving hash/mtime, and manifest mismatch failure.

- [ ] **Step 2: Implement the only approved projection path**

```python
flat = torch.empty(
    (1, layout.total_numel),
    dtype=torch.float32,
    device=device,
)
for entry, (name, parameter) in zip(layout.entries, named_parameters, strict=True):
    gradient = parameter.grad
    if gradient is None or gradient.dtype != torch.float32:
        raise RuntimeError(f"invalid FP32 gradient for {name}")
    if not torch.isfinite(gradient).all():
        raise RuntimeError(f"non-finite gradient for {name}")
    flat[0, entry.offset : entry.offset + entry.numel].copy_(gradient.reshape(-1))

projector = CudaProjector(
    grad_dim=layout.total_numel,
    proj_dim=1024,
    seed=0,
    proj_type=ProjectionType.rademacher,
    device=device,
    max_batch_size=8,
)
sketch = projector.project(flat, model_id=0)[0].float() / math.sqrt(1024)
```

Compute the squared norm from FP32 parameter gradients before projection.
Require exact `CudaProjector`; do not catch its failure and substitute another
implementation. Reuse one current-group flat buffer per process if safe, but
never retain more than one group gradient/staging buffer.

- [ ] **Step 3: Implement atomic per-group collection**

Shard the 512 prompt IDs into four contiguous blocks of 128 so both seeds for
an ID stay on one GPU. Store:

```text
sketch                 float32 [1024]
full_gradient_sq_norm  float32 scalar
full_gradient_norm     float32 scalar
projected_norm         float32 scalar
trajectory_losses      float32 [4]
effective_tokens       int64 [4]
```

Safetensors metadata contains canonical record JSON with all capture hashes,
layout/replay manifest hashes, seed/step/slots, tolerance errors, elapsed time,
and peak CUDA memory. Use temporary file, fsync, replace, and directory fsync.

- [ ] **Step 4: Implement projection-only and global validation modes**

`--projection-smoke-only` must allocate an FP32
`(1, 4_022_468_096)` input and execute real seed-0 fast-JL before any capture.
The global validator requires exact `512 × 2` coverage, four slots per group,
one layout/projection contract, finite sketches, fixed zero-row handling, and
re-aggregated capture/replay losses. For production it independently verifies
`objective_pg_loss == 256 * logged_actor_pg_loss`.

- [ ] **Step 5: Run non-GPU tests and tiny-GPU tests**

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest -q \
  math_eval/test_opd_gradient_alignment.py \
  math_eval/test_collect_opd_gradient_sketches.py -m 'not gpu'

FIRST_ALLOCATED_TOKEN="${CUDA_VISIBLE_DEVICES%%,*}"
CUDA_VISIBLE_DEVICES="$FIRST_ALLOCATED_TOKEN" PYTHONPATH=verl:. \
  /home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest -q \
  math_eval/test_collect_opd_gradient_sketches.py -m gpu \
  -k 'tiny_qwen or projector'
```

Expected: CPU suite passes; tiny Qwen actor/replay log-probs align, parameters
and gradients stay FP32, forward uses BF16, and projected output is deterministic
FP32 shape `(1024,)` with the required scaling.

- [ ] **Step 6: Commit collector/projection changes**

```bash
git add math_eval/collect_opd_gradient_sketches.py \
  math_eval/test_collect_opd_gradient_sketches.py \
  math_eval/test_opd_gradient_alignment.py
git commit -m "Add full-parameter OPD gradient projection"
```

### Task 7: Preregistered Statistical Analysis and Reports

**Files:**
- Create: `math_eval/analyze_opd_gradient_alignment.py`
- Create: `math_eval/test_analyze_opd_gradient_alignment.py`

**Interfaces:**
- Consumes: exactly matched SFT sketches, both OPD seed sketches, full norms, capture metadata, and preparation manifest.
- Produces: `analysis.json`, `analysis.md`, and persisted resampling-index artifacts.
- Produces: one exclusive representation label plus an optional weak-signal/budget-coupling flag.
- Exposes:

```python
def run_analysis(
    *,
    sample_manifest: Path,
    sft_gradient_dir: Path,
    sft_gradient_manifest: Path,
    sft_eligibility_report: Path,
    capture_root: Path,
    replay_root: Path,
    resampling_dir: Path,
    output_json: Path,
    output_markdown: Path,
) -> dict[str, Any]: ...
```

The CLI requires the corresponding nine explicit path arguments; it has no
ambient `input-root` discovery or latest-run fallback.

- [ ] **Step 1: Write failing kernel/Vendi tests**

Test the shared orthogonal null state:

```python
kernel = null_state_cosine_kernel(np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]))
np.testing.assert_allclose(kernel, np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 1.0],
    [0.0, 1.0, 1.0],
]))
assert np.linalg.eigvalsh(kernel).min() >= -1e-7
```

Assert effective-rank eigenvalues are divided by kernel trace, not blindly by
row count; test positive-only Vendi and near-zero sensitivity separately.
Use float64 eigendecomposition. Clip a negative eigenvalue only when its
magnitude is at most `1e-10 * max(1, max_eigenvalue)`; a more negative value is
a fatal non-PSD error. Require unit diagonal and trace equal to sample count.

- [ ] **Step 2: Write failing relational/resampling/gate tests**

Use synthetic spaces to cover known aligned, within-arm shuffled, arm-label-only
aligned, noisy, zero-row, and degenerate Gram cases. Assert block-centered CKA
excludes cross-arm structure, Spearman and neighbors are within arm, undefined
blocks fail reliability, and permutations only shuffle IDs within their arm.
Break nearest-neighbor ties by similarity descending then stable row index
ascending.

Test exact fixed resampling seeds `2026071501` through `2026071505`. For
`CKA_excess`, assert every bootstrap subtracts the one frozen original-sample
`m_null`; no nested permutations are allowed.

Persist these exact index arrays:

```text
reliability permutations:     (2000, 2, 256), seed 2026071501
diversity label permutations: (2000, 512),    seed 2026071502
selected/random bootstraps:   (2000, 256),    shared seed 2026071503
selected/random half-samples: (2000, 128),    shared seed 2026071504
cross-space permutations:     (2000, 2, 256), seed 2026071505
```

- [ ] **Step 3: Implement pure statistical functions**

Provide these stable interfaces:

```python
def null_state_cosine_kernel(vectors: np.ndarray, *, threshold: float = 0.0) -> np.ndarray: ...
def vendi_effective_rank(kernel: np.ndarray) -> float: ...
def block_center(kernel: np.ndarray, arms: np.ndarray) -> np.ndarray: ...
def block_cka(left: np.ndarray, right: np.ndarray, arms: np.ndarray) -> float: ...
def mean_within_arm_spearman(left: np.ndarray, right: np.ndarray, arms: np.ndarray) -> float: ...
def within_arm_neighbor_jaccard(left: np.ndarray, right: np.ndarray, arms: np.ndarray, k: int) -> float: ...
def build_resampling_indices(arms: np.ndarray) -> dict[str, np.ndarray]: ...
def centered_ratio_bootstrap_p_value(samples: np.ndarray, estimate: float, null: float = 0.90) -> float: ...
def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]: ...
def classify_probe(report: Mapping[str, Any]) -> dict[str, Any]: ...
```

Persist RNG implementation, seed, shape, and SHA256 for every resampling table.
Reuse the same 2,000 arm-stratified bootstrap table for diversity, CKA excess,
gradient norm ratio, and token ratio.

- [ ] **Step 4: Implement exact endpoints and interpretation hierarchy**

Load the existing globally validated SFT gradients from
`data/gradient_diversity/gradients/qwen2.5-0.5b-instruct/`, subset to the exact
512 stable IDs, average the two OPD sketches before normalization, and never
compare SFT/OPD vector coordinates directly. Implement every reliability,
diversity, alignment, signal, and routing diagnostic in the approved spec.

Define signal ratios exactly:

```python
n_arm = np.sqrt(np.mean([norm_i_seed ** 2 for every ID and both seeds]))
t_arm = np.mean([effective_tokens for every ID, both seeds, and four slots])
q_norm = n_selected / n_random
q_token = t_selected / t_random
```

Apply the one-sided centered ratio bootstrap and Holm correction. Keep the
primary representation label exclusive; the weak-signal flag is secondary and
may coexist only under the specified OPD-diversity-transfer condition.

Encode the preregistered gates literally:

```text
OPD reliability:
  replicate block-CKA >= 0.50
  mean within-arm Spearman >= 0.30
  both one-sided within-arm permutation p-values <= 0.01

OPD diversity transfers:
  OPD D bootstrap lower bound > 0
  selected-greater arm-label permutation p <= 0.05

support representation mismatch:
  reliable OPD
  SFT D bootstrap lower bound > 0
  OPD D bootstrap upper bound <= 0
  OPD selected-greater arm-label permutation p > 0.05
  CKA_excess bootstrap upper bound < 0.10

reject mismatch as primary explanation:
  reliable OPD
  OPD diversity transfers
  CKA_excess bootstrap lower bound > 0.10

weak-signal/budget-coupling endpoint:
  OPD diversity transfers
  Q upper percentile bound < 0.90
  Holm-adjusted one-sided p <= 0.05
```

An undefined arm Spearman, zero block-centered Frobenius norm, or non-positive
random-arm signal denominator is a structured degeneracy, never a silently
dropped sample. Also report the required non-gating diagnostics: both
arm-specific Spearmans; top-10/top-20 within-arm neighbor Jaccards; per-question
seed cosine distribution; full/projected norm; teacher gap; rollout-IS mean and
clip fraction; route/correct-count/topic/difficulty composition; original and
effective token counts; zero/near-zero fractions; and correlations of norm and
neighbor structure with route, length, ESR tokens, topic, difficulty, and R1
solution length.

- [ ] **Step 5: Implement atomic JSON and Markdown reports**

The JSON includes every point estimate, interval, p-value, gate input/output,
composition diagnostic, zero handling count, artifact SHA, and resampling SHA.
The Markdown begins with the exclusive conclusion, reliability status, OPD
Vendi contrast, CKA excess, norm/token ratios, then a limitations paragraph.
Regenerate the Markdown solely from stored artifacts and JSON inputs.
Write JSON atomically with `allow_nan=False`. Represent undefined values as a
record such as `{"status": "undefined", "reason": "zero_block_frobenius"}`;
never serialize `NaN`, `Infinity`, or an unstructured null.

- [ ] **Step 6: Run tests and commit**

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest -q \
  math_eval/test_analyze_opd_gradient_alignment.py

git add math_eval/analyze_opd_gradient_alignment.py \
  math_eval/test_analyze_opd_gradient_alignment.py
git commit -m "Add OPD gradient alignment analysis"
```

### Task 8: Allocation-Aware End-to-End Orchestrator

**Files:**
- Create: `math_eval/opd_gradient_alignment_runtime.py`
- Create: `run_step0_opd_gradient_alignment_probe.sh`
- Create: `math_eval/test_opd_gradient_alignment_launcher.py`

**Interfaces:**
- Consumes: all CLIs and launchers from Tasks 1–7.
- Produces: one locked/resumable stage manifest and unique log per stage.
- Runs capture with all four allocated tokens; runs four replay processes with one allocated token each and local `cuda:0`.

- [ ] **Step 1: Write failing fake-runtime tests**

Simulate a `CUDA_VISIBLE_DEVICES` list containing four UUID-like tokens. Assert:

- missing Slurm job, missing tmux, wrong session, empty/duplicate/wrong token count, occupied GPU, or excess memory fails before artifacts;
- the original full CVD string reaches each capture subprocess;
- replay shard `i` receives only token `i` and always uses `--device cuda:0`;
- no generated command contains `srun`;
- projection failure prevents capture, smoke failure prevents production, and seed-42 failure prevents seed 43;
- exact complete artifacts are skipped, while mismatched artifacts fail;
- failure cleanup terminates only child PIDs started by this launcher.
- the idle-GPU check runs immediately before every GPU-bearing stage, not only
  once at launcher startup.

- [ ] **Step 2: Run launcher tests and verify RED**

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  math_eval/test_opd_gradient_alignment_launcher.py
```

Expected: missing runtime module/launcher failure.

- [ ] **Step 3: Implement allocation and ETA validation**

Require `SLURM_JOB_ID`, `TMUX`, current `tmux display-message -p '#S'` equal to
`opd-CLI`, and exactly four nonempty unique comma-separated CVD tokens. Map
logical CUDA devices to NVML UUIDs, reject compute processes and memory above
the fixed idle threshold, and record logical index/token/UUID without exposing
or reusing host indices.

After smoke replay, use the slowest-shard per-group rate with a `1.20` safety
factor plus measured capture time and a two-hour reserve. Parse the current
Slurm job end time and stop before production unless the estimate fits.

- [ ] **Step 4: Implement the ordered top-level stages**

The locked launcher runs:

```text
1. prepare/validate 512-row and 8-row inputs
2. verify allocation/logical GPUs
3. tiny-model GPU replay/projection equivalence smoke
4. real 4,022,468,096-D projection-only smoke on one GPU
5. two fresh-engine smoke captures, seeds 42 then 43
6. deterministic shortest captured Qwen3 trajectory single-trajectory replay smoke
7. four parallel eight-question smoke replay shards
8. smoke global validation, restart/skip check, and ETA gate
9. two fresh-engine production captures, seeds 42 then 43
10. four parallel production replay shards
11. production global validation
12. CPU analysis and report regeneration check
```

Every command is printed, logged through `tee` to a timestamped stage path,
and recorded with its resume command. Use exact pinned Python executables. The
script must not install packages, create allocations, nest launchers under
`srun`, or send terminal-control characters. Call `check-gpus --require-idle`
immediately before stages 3, 4, 5, 6, 7, 9, and 10. Before each capture stage,
also require that neither seed-level `COMPLETE.json` exists unless its manifest
matches exactly; a partial seed is resumed only through the capture validator.

- [ ] **Step 5: Run fake-runtime tests and dry-run**

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  math_eval/test_opd_gradient_alignment_launcher.py

OPD_GRADIENT_PROBE_DRY_RUN=1 bash run_step0_opd_gradient_alignment_probe.sh
```

Expected: tests pass; dry-run lists all twelve stages, two distinct capture
processes per mode, four replay shards, and no GPU/model load.

- [ ] **Step 6: Commit the orchestrator**

```bash
git add math_eval/opd_gradient_alignment_runtime.py \
  run_step0_opd_gradient_alignment_probe.sh \
  math_eval/test_opd_gradient_alignment_launcher.py
git commit -m "Add fail-closed OPD gradient probe launcher"
```

### Task 9: Full CPU Regression and Implementation Review

**Files:**
- Verify all files changed in Tasks 1–8.

**Interfaces:**
- Produces: evidence that capture-disabled veRL, probe CLIs, and shell contracts pass together before GPU work.

- [ ] **Step 1: Run all focused CPU suites from a clean command**

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  math_eval/test_prepare_opd_gradient_alignment.py \
  verl/tests/workers/config/test_rollout_config_on_cpu.py \
  verl/tests/trainer/config/test_algo_config_on_cpu.py \
  verl/tests/trainer/ppo/test_opd_gradient_alignment_capture.py \
  verl/tests/trainer/ppo/test_opd_gradient_alignment_capture_launcher.py \
  math_eval/test_opd_gradient_alignment_launcher.py

PYTHONPATH=verl:. /home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest -q \
  math_eval/test_opd_gradient_alignment.py \
  math_eval/test_collect_opd_gradient_sketches.py \
  math_eval/test_analyze_opd_gradient_alignment.py -m 'not gpu'

bash -n run_capture_step0_opd_gradient_alignment_seed.sh \
  run_capture_step0_opd_gradient_alignment.sh \
  run_step0_opd_gradient_alignment_probe.sh
```

Expected: both commands exit `0` with no failed tests.

- [ ] **Step 2: Run relevant existing regressions**

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  verl/tests/trainer/ppo/test_rollout_corr_integration.py \
  verl/tests/trainer/ppo/test_tale_budget.py \
  verl/tests/workers/actor/test_length_aware_opd.py \
  verl/tests/trainer/config/test_legacy_config_on_cpu.py
```

Expected: command exits `0`; capture defaults do not alter existing behavior.

- [ ] **Step 3: Verify plan/spec coverage and working-tree boundaries**

```bash
rg -n 'TO[D]O|FIX[M]E|PLACE[H]OLDER' \
  math_eval/prepare_opd_gradient_alignment.py \
  verl/verl/trainer/ppo/opd_gradient_alignment_capture.py \
  math_eval/opd_gradient_alignment.py \
  math_eval/collect_opd_gradient_sketches.py \
  math_eval/analyze_opd_gradient_alignment.py \
  run_capture_step0_opd_gradient_alignment_seed.sh \
  run_capture_step0_opd_gradient_alignment.sh \
  run_step0_opd_gradient_alignment_probe.sh
git diff --check
git status --short
```

Expected: placeholder search prints nothing, `git diff --check` exits `0`, and
only intended probe changes plus preexisting user changes are present.

- [ ] **Step 4: Request two-stage review**

Review first for exact spec compliance, then for code quality/security. Resolve
every blocker/high finding with a focused regression test and rerun Steps 1–3.

### Task 10: `opd-CLI` GPU Smoke, Production Probe, and Result Handoff

**Files:**
- Runtime outputs only under `data/opd_gradient_alignment/` and `logs/opd_gradient_alignment/`.

**Interfaces:**
- Consumes: reviewed implementation and existing four-GPU Slurm allocation.
- Produces: validated 1,024 OPD group sketches and the final analysis JSON/Markdown.

- [ ] **Step 1: Revalidate the authorized terminal and allocation**

Inside `opd-CLI`, run:

```bash
tmux display-message -p '#S'
printf 'SLURM_JOB_ID=%s\nCUDA_VISIBLE_DEVICES=%s\n' \
  "${SLURM_JOB_ID:?}" "${CUDA_VISIBLE_DEVICES:?}"
/home/mchen/miniconda3/envs/verl/bin/python \
  -m math_eval.opd_gradient_alignment_runtime check-gpus \
  --expected 4 --require-idle
```

Expected: session `opd-CLI`, one active job ID, four owned logical GPUs, and no
foreign compute process. Stop without launching if any check fails.

- [ ] **Step 2: Launch the fail-closed pipeline directly, without `srun`**

```bash
cd /home/mchen/FiRe-OPD
bash run_step0_opd_gradient_alignment_probe.sh
```

Expected first gates: preparation hashes pass; the tiny-model actor/replay and
projection equivalence test passes; the real 4B-dimensional FP32 fast-JL smoke
returns finite nonzero shape `(1024,)`; and both seed-specific smoke captures
complete.

- [ ] **Step 3: Validate the eight-question replay before production**

First replay the deterministically selected shortest real captured Qwen3
trajectory with `--single-trajectory-smoke`; only after its actor/replay
log-probabilities, scalar loss, full-gradient norm, and sketch agree should the
launcher run the four-shard eight-question replay. Require all of the following
from smoke validation:

```text
8 unique IDs × 2 seeds × 4 slots
exact masks and token counts
log-prob atol <= 0.02 and rtol <= 0.01
scalar-loss atol <= 0.001 and rtol <= 0.01
one parameter layout/count across four processes
finite nonzero 1,024-D sketches
second invocation resume-skips every complete group
estimated production completion plus reserve fits allocation
```

If any condition fails, leave artifacts/logs intact and do not start production.

- [ ] **Step 4: Monitor production capture/replay to exact completion**

The launcher must reach four complete capture steps
`2 seeds × 2 steps`, then four replay shards of `128 IDs × 2 seeds`. Poll only
the launcher's own processes and logs. Do not interrupt the allocation or any
unrelated process. A failed shard stops sibling shard processes started by the
launcher and prints exact resume commands.

- [ ] **Step 5: Validate and inspect the final report**

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/gvendi-opd/bin/python \
  -m math_eval.collect_opd_gradient_sketches \
  --validate-global-only \
  --capture-root data/opd_gradient_alignment/production/capture \
  --output-dir data/opd_gradient_alignment/production/replay

PYTHONPATH=verl:. /home/mchen/miniconda3/envs/gvendi-opd/bin/python \
  -m math_eval.analyze_opd_gradient_alignment \
  --sample-manifest data/opd_gradient_alignment/input/preparation.manifest.json \
  --sft-gradient-dir data/gradient_diversity/gradients/qwen2.5-0.5b-instruct \
  --sft-gradient-manifest data/gradient_diversity/gradients/qwen2.5-0.5b-instruct/gradient.manifest.json \
  --sft-eligibility-report data/gradient_diversity/deepmath_level6_r1_solution1.eligibility.json \
  --capture-root data/opd_gradient_alignment/production/capture \
  --replay-root data/opd_gradient_alignment/production/replay \
  --resampling-dir data/opd_gradient_alignment/analysis/resampling \
  --output-json data/opd_gradient_alignment/analysis.json \
  --output-markdown data/opd_gradient_alignment/analysis.md
```

Expected: exact `512 IDs × 2 seeds × 4 slots`, no non-finite value, all
provenance checks pass, and regenerating both reports yields byte-identical
outputs.

- [ ] **Step 6: Report evidence without overstating causality**

Handoff the exclusive representation label, optional weak-signal flag, OPD
reliability gate, selected/random SFT and OPD Vendi contrasts, block-CKA excess,
gradient norm/token ratios, runtime/cost, and links to `analysis.md`,
`analysis.json`, manifests, and logs. Explicitly state that this diagnostic
tests representation transfer and does not itself prove downstream OOD gain.
