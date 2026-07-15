# One-Time OPD Efficacy Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a one-time, CPU-prepared `efficacy_pilot` stage that selects 56 of the first 250 frozen Stage-1 candidates with one proxy OPD gradient per question and reports a preregistered target-gradient-space `go|no_go|borderline` decision without launching GPU work during implementation.

**Architecture:** Add a first-class string stage profile consumed by the existing prepare, capture, replay, selector, analyzer, and orchestrator entry points. Derive the pilot manifest from the immutable Stage-1 manifest by exact hash and ordered prefixes, while binding pilot runtime artifacts to a separate source/algorithm identity on branch `opd-efficacy-pilot`. Keep Stage-0 and Stage-1 bytes untouched and publish all pilot files under an independent `efficacy_pilot/` namespace.

**Tech Stack:** Python 3.10, dataclasses, NumPy/PCG64, PyArrow, PyTorch, OmegaConf/Hydra, VERL, vLLM, pinned Prismatic/TRAK `CudaProjector`, pytest, Ruff, Git SHA-256 provenance.

## Global Constraints

- Work only in `/home/mchen/FiRe-OPD/.worktrees/opd-proxy-gradient-verify-impl` on branch `opd-efficacy-pilot`.
- Preserve original branch `opd-proxy-gradient-verify-impl` at commit `174849613a9c61445765f0b913174d286f3819e4` for future Stage-1 execution.
- Do not alter any file under `data/opd_proxy_gradient_verify/stage_0/` or `data/opd_proxy_gradient_verify/stage_1/`.
- The frozen Stage-1 manifest SHA-256 must remain `6d698d75995c777d6faaf1abd385a758977ca0350ce862284f1e9d8751eddd83` before and after every task.
- Derive exactly the first 250 ordered Stage-1 candidates and first 84 ordered Stage-1 held-out rows; never reshuffle, replace, retokenize, or resample membership.
- Publish pilot artifacts only beneath `data/opd_proxy_gradient_verify/efficacy_pilot/` and logs only beneath `logs/opd_proxy_gradient_verify/efficacy_pilot/`.
- Pilot target and proxy capture use only generation seed 42 and native `rollout.n=1`.
- Every Stage-1 algorithm field other than rollout count and generation-seed set remains exactly equal; maximum response length stays 16,384.
- Produce exactly `P_pilot` and `T_pilot`; do not produce `P_n4`, SFT, embedding, seed 43, or target-oracle selections.
- Use pinned TRAK `CudaProjector`, 1,024 dimensions, Rademacher projection, projection seed 0, and no CPU fallback.
- Use selected size 56, primary K 25, diagnostic K 3, K-means seeds 42/43, and round-robin seed 42.
- Generate 10,000 `R_uniform` and 10,000 `R_stratified` schedules from `SeedSequence(2026071402).spawn(2)`.
- The gate consumes primary-K `R_uniform` percentiles only; diagnostic K and stratified nulls are report-only.
- Every report status is `pilot_only`; no main-hypothesis pass/fail/inconclusive classification is allowed.
- Use `/home/mchen/miniconda3/envs/verl/bin/python` for VERL-focused tests and `/home/mchen/miniconda3/envs/gvendi-opd/bin/python` for preparation, replay, selection, analysis, and CPU tests.
- Do not install packages at runtime.
- Follow red-green-refactor for every behavior change and commit after each independently reviewable task.
- Do not launch any GPU work unit. Only canonical preparation, deterministic dry-run, and `validate-stage --preflight-only` are permitted in the final task.
- Do not publish the canonical pilot manifest until all tracked source, tests, protocol documentation, and this plan are committed and the worktree is clean.
- After canonical pilot publication, do not modify tracked source; record the resulting manifest SHA only in the ignored runtime status document.

---

### Task 1: Add the First-Class Stage Profile and Frozen Algorithm Contract

**Files:**
- Create: `math_eval/opd_proxy_gradient_stage_profiles.py`
- Create: `math_eval/test_opd_proxy_gradient_stage_profiles.py`
- Modify: `math_eval/prepare_opd_proxy_gradient_verify.py`

**Interfaces:**
- Consumes: the approved cardinalities and Stage-1 capture contract.
- Produces: `StageKind`, `StageProfile`, `EFFICACY_PILOT`, `parse_stage_kind`, `stage_profile`, `stage_directory_name`, `capture_algorithm_contract`, and `capture_algorithm_contract_sha256`.
- Later tasks use these interfaces instead of adding independent pilot constants.

- [ ] **Step 1: Write failing profile/cardinality/contract tests**

Create tests with these exact assertions:

```python
from math_eval.opd_proxy_gradient_stage_profiles import (
    EFFICACY_PILOT,
    capture_algorithm_contract,
    capture_algorithm_contract_sha256,
    parse_stage_kind,
    stage_directory_name,
    stage_profile,
)


def test_efficacy_pilot_profile_is_exact():
    profile = stage_profile(EFFICACY_PILOT)
    assert profile.candidate_count == 250
    assert profile.held_out_count == 84
    assert profile.selected_size == 56
    assert profile.primary_k == 25
    assert profile.diagnostic_k == 3
    assert profile.null_draws == 10_000
    assert profile.generation_seeds == (42,)
    assert profile.native_rollouts == 1
    assert profile.selection_representations == ("P_pilot",)
    assert profile.target_representations == ("T_pilot",)
    assert profile.run_sft is False
    assert profile.run_embedding is False
    assert profile.run_target_oracle is False
    assert stage_directory_name(EFFICACY_PILOT) == "efficacy_pilot"


def test_stage_parser_accepts_only_numbered_stages_and_efficacy_pilot():
    assert [parse_stage_kind(value) for value in ("0", "1", "2")] == [0, 1, 2]
    assert parse_stage_kind("efficacy_pilot") == EFFICACY_PILOT
    for value in ("3", "-1", "pilot", "stage_1"):
        with pytest.raises(ValueError, match="unsupported stage"):
            parse_stage_kind(value)


def test_pilot_contract_differs_only_in_approved_fields():
    stage1 = capture_algorithm_contract(1)
    pilot = capture_algorithm_contract(EFFICACY_PILOT)
    assert stage1["rollout_n"] == 4
    assert pilot["rollout_n"] == 1
    assert stage1["generation_seeds"] == [42, 43]
    assert pilot["generation_seeds"] == [42]
    ignored = {"stage_identity", "rollout_n", "generation_seeds"}
    assert {k: v for k, v in stage1.items() if k not in ignored} == {
        k: v for k, v in pilot.items() if k not in ignored
    }
    assert stage1["max_response_length"] == pilot["max_response_length"] == 16_384
    assert capture_algorithm_contract_sha256(1) != capture_algorithm_contract_sha256(
        EFFICACY_PILOT
    )
```

Also extend `test_stage_layouts_are_derived_without_stage1_constant_leakage` to assert that `stage_layout("efficacy_pilot")` exposes counts 250/84/56/25/3/10,000 but is not accepted by `build_stage_rows`, because pilot rows must come from the parent manifest rather than the root permutation builder.

- [ ] **Step 2: Run tests and verify the new module is missing**

Run:

```bash
/home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest \
  math_eval/test_opd_proxy_gradient_stage_profiles.py \
  math_eval/test_prepare_opd_proxy_gradient_verify.py::test_stage_layouts_are_derived_without_stage1_constant_leakage -q
```

Expected: collection fails with `ModuleNotFoundError: math_eval.opd_proxy_gradient_stage_profiles`.

- [ ] **Step 3: Implement the immutable profile module**

Implement this shape, filling the numbered profiles with their existing values and representations:

```python
from dataclasses import dataclass
from typing import Literal, TypeAlias

from math_eval.opd_proxy_gradient_verify_artifacts import canonical_json_bytes

EFFICACY_PILOT = "efficacy_pilot"
StageKind: TypeAlias = Literal[0, 1, 2, "efficacy_pilot"]


@dataclass(frozen=True)
class StageProfile:
    key: StageKind
    directory_name: str
    candidate_count: int
    held_out_count: int
    selected_size: int
    primary_k: int
    diagnostic_k: int
    null_draws: int
    generation_seeds: tuple[int, ...]
    native_rollouts: int
    selection_representations: tuple[str, ...]
    target_representations: tuple[str, ...]
    run_sft: bool
    run_embedding: bool
    run_target_oracle: bool
    run_direct_fixture: bool
    require_resume_exercise: bool


def parse_stage_kind(value: str | int) -> StageKind:
    if value in (0, 1, 2):
        return value
    if isinstance(value, str) and value in {"0", "1", "2"}:
        return int(value)
    if value == EFFICACY_PILOT:
        return EFFICACY_PILOT
    raise ValueError(f"unsupported stage: {value}")
```

Define one `_PROFILES` mapping. The pilot profile must contain exactly the values asserted above. Define `COMMON_CAPTURE_CONTRACT` with every frozen Stage-1 field from the design, then build canonical payloads by adding only `stage_identity`, `rollout_n`, and `generation_seeds`. Hash with `hashlib.sha256(canonical_json_bytes(payload)).hexdigest()`.

Change `StageLayout.stage` to `StageKind`, make `stage_layout` delegate cardinalities to `stage_profile`, and explicitly reject `EFFICACY_PILOT` inside `build_stage_rows` with `ValueError("efficacy_pilot rows require the frozen Stage-1 parent")`.

- [ ] **Step 4: Run focused tests**

Run the command from Step 2.

Expected: all selected tests pass.

- [ ] **Step 5: Verify numbered-stage regression tests**

Run:

```bash
/home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest \
  math_eval/test_prepare_opd_proxy_gradient_verify.py \
  math_eval/test_opd_proxy_gradient_stage_profiles.py -q
```

Expected: all tests pass, including unchanged 0/1/2 layouts.

- [ ] **Step 6: Commit**

```bash
git add math_eval/opd_proxy_gradient_stage_profiles.py \
  math_eval/test_opd_proxy_gradient_stage_profiles.py \
  math_eval/prepare_opd_proxy_gradient_verify.py \
  math_eval/test_prepare_opd_proxy_gradient_verify.py
git commit -m "feat: define efficacy pilot stage contract"
```

---

### Task 2: Derive and Publish the Pilot Manifest from the Frozen Stage-1 Parent

**Files:**
- Modify: `math_eval/prepare_opd_proxy_gradient_verify.py`
- Modify: `math_eval/test_prepare_opd_proxy_gradient_verify.py`

**Interfaces:**
- Consumes: `StageProfile`, the frozen Stage-1 manifest and its sample/capture-input artifacts.
- Produces: `FROZEN_STAGE1_MANIFEST_SHA256`, `EfficacyPilotParent`, `load_efficacy_pilot_parent`, `derive_efficacy_pilot_rows`, and a pilot-aware `prepare_stage`/`write_stage_artifacts` path.
- The pilot manifest contains `parent_stage_manifest`, `algorithm_contract`, and `algorithm_contract_sha256`.

- [ ] **Step 1: Add failing parent-prefix tests**

Add a fixture that writes a canonical synthetic Stage-1 manifest, 768 candidate sample rows followed by 256 held-out rows, and matching target/proxy capture-input parquet files. Test:

```python
def test_pilot_derives_exact_stage1_prefixes_without_mutating_parent(stage1_parent):
    parent_path = stage1_parent / "manifest.json"
    before = {path: sha256_file(path) for path in stage1_parent.rglob("*") if path.is_file()}
    parent = load_efficacy_pilot_parent(parent_path, sha256_file(parent_path))
    rows = derive_efficacy_pilot_rows(parent)
    assert [row["stable_id"] for row in rows[:250]] == [f"c{i:04d}" for i in range(250)]
    assert [row["stable_id"] for row in rows[250:]] == [f"h{i:04d}" for i in range(84)]
    assert [row["manifest_index"] for row in rows] == list(range(334))
    assert [row["parent_manifest_index"] for row in rows[:250]] == list(range(250))
    assert [row["parent_manifest_index"] for row in rows[250:]] == list(range(768, 852))
    assert all(len(row["parent_row_sha256"]) == 64 for row in rows)
    after = {path: sha256_file(path) for path in stage1_parent.rglob("*") if path.is_file()}
    assert after == before
```

Add rejection tests for wrong parent SHA, one reordered candidate, one replacement ID, changed split, noncanonical JSON, and a mismatched referenced artifact hash. Add a publication test asserting:

```python
assert manifest["stage"] == "efficacy_pilot"
assert manifest["candidate_count"] == 250
assert manifest["held_out_count"] == 84
assert manifest["target_capture_count"] == 334
assert manifest["proxy_capture_count"] == 250
assert manifest["parent_stage_manifest"]["sha256"] == parent_sha
assert manifest["algorithm_contract_sha256"] == capture_algorithm_contract_sha256(
    EFFICACY_PILOT
)
assert not any(path.is_relative_to(stage1_parent) for path in pilot_written_paths)
```

- [ ] **Step 2: Run the prefix tests and verify missing interfaces**

Run:

```bash
/home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest \
  math_eval/test_prepare_opd_proxy_gradient_verify.py \
  -k 'efficacy_pilot or pilot_derives or pilot_parent' -q
```

Expected: failures because `load_efficacy_pilot_parent` and `derive_efficacy_pilot_rows` do not exist.

- [ ] **Step 3: Implement strict parent loading and row linkage**

Add:

```python
FROZEN_STAGE1_MANIFEST_SHA256 = (
    "6d698d75995c777d6faaf1abd385a758977ca0350ce862284f1e9d8751eddd83"
)


@dataclass(frozen=True)
class EfficacyPilotParent:
    manifest_path: Path
    manifest_sha256: str
    manifest: dict[str, object]
    sample_path: Path
    sample_sha256: str
    rows: tuple[dict[str, object], ...]
```

`load_efficacy_pilot_parent` must canonical-load the manifest, verify the supplied digest, require Stage 1 and exact Stage-1 cardinalities/hashes, resolve only paths beneath the parent stage directory, hash every referenced input, and load exactly 768 candidate plus 256 held-out sample rows.

`derive_efficacy_pilot_rows` must copy parent rows, add local contiguous indices, preserve a `parent_manifest_index`, compute `parent_row_sha256 = sha256(canonical_json_bytes(parent_row))`, and retain parent exact prompt/message bytes.

- [ ] **Step 4: Extend canonical preparation and artifact publication**

Extend the parser with:

```text
--stage efficacy_pilot
--parent-stage1-manifest PATH
--expected-parent-stage1-manifest-sha256 SHA256
```

For canonical mode, require the supplied expected digest to equal `FROZEN_STAGE1_MANIFEST_SHA256`. For disposable tests, still require exact equality to the synthetic parent's actual digest but do not require the production constant.

Publish to `output_root / "efficacy_pilot"`, not `stage_<value>`. Include parent prefix hashes, parent-row linkage hashes, pilot publication provenance, and pilot algorithm identity. Recreate pilot capture-input parquet bytes from verified child rows; do not copy GPU outputs.

- [ ] **Step 5: Run focused preparation tests**

Run the command from Step 2.

Expected: all selected tests pass.

- [ ] **Step 6: Verify the frozen production parent remains unchanged**

Run:

```bash
sha256sum data/opd_proxy_gradient_verify/stage_1/manifest.json
git status --short data/opd_proxy_gradient_verify/stage_0 data/opd_proxy_gradient_verify/stage_1
```

Expected:

```text
6d698d75995c777d6faaf1abd385a758977ca0350ce862284f1e9d8751eddd83  data/opd_proxy_gradient_verify/stage_1/manifest.json
```

and no status entries for either stage directory.

- [ ] **Step 7: Commit**

```bash
git add math_eval/prepare_opd_proxy_gradient_verify.py \
  math_eval/test_prepare_opd_proxy_gradient_verify.py
git commit -m "feat: derive efficacy pilot from frozen stage1"
```

---

### Task 3: Support Native `n=1` Capture with an Isolated Typed Contract

**Files:**
- Modify: `verl/verl/trainer/config/algorithm.py`
- Modify: `verl/verl/trainer/config/ppo_trainer.yaml`
- Modify: `verl/verl/trainer/ppo/opd_proxy_verify_capture.py`
- Modify: `verl/verl/trainer/ppo/ray_trainer.py`
- Modify: `verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`
- Modify: `verl/tests/trainer/ppo/test_opd_proxy_verify_capture.py`
- Modify: `verl/tests/trainer/ppo/test_opd_proxy_verify_capture_launcher.py`
- Modify: `verl/tests/workers/rollout/test_opd_proxy_verify_native_n.py`
- Modify: `verl/tests/trainer/config/test_algo_config_on_cpu.py`

**Interfaces:**
- Consumes: pilot `stage`, `native_rollouts`, sample/source hashes, and algorithm-contract hash.
- Produces: a validated capture contract that accepts only `(efficacy_pilot, seed 42, n=1)` for pilot while retaining `(numbered stage, seed 42/43, n=4)` behavior.

- [ ] **Step 1: Add failing typed-contract tests**

Add a pilot config fixture based on the existing real source/sample fixture and assert:

```python
contract = validate_capture_contract(
    {
        **valid_capture_config,
        "stage": "efficacy_pilot",
        "engine_seed": 42,
        "native_rollouts": 1,
        "algorithm_contract_sha256": "a" * 64,
    }
)
assert contract["stage"] == "efficacy_pilot"
assert contract["native_rollouts"] == 1
```

Add failures for pilot `n=4`, seed 43, missing algorithm-contract hash, and numbered-stage `n=1`. Add an OmegaConf construction test proving the typed config can represent the string stage without `ValidationError`.

- [ ] **Step 2: Add a failing native-`n=1` one-engine/one-call test**

In `test_opd_proxy_verify_native_n.py`, add:

```python
def test_pilot_native_n1_uses_one_generation_call_and_assigns_slot_zero():
    rollout = _bare_rollout(_request_outputs(1))
    prompts = _prompts()
    prompts.meta_info["opd_proxy_verify_stage"] = "efficacy_pilot"
    result = _generate_without_device_decorators(
        rollout, prompts, opd_proxy_verify_native_n=1
    )
    assert len(rollout.inference_engine.calls) == 1
    assert rollout.inference_engine.calls[0]["sampling_params"].n == 1
    assert result.non_tensor_batch["opd_verify_stable_id"].tolist() == ["q0", "q1"]
    assert result.batch["opd_proxy_verify_rollout_slot"].tolist() == [0, 0]
```

Retain the existing native-`n=4` test unchanged.

- [ ] **Step 3: Run focused VERL tests and observe rejection of `n=1`**

Run:

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/verl/bin/python -m pytest \
  verl/tests/trainer/ppo/test_opd_proxy_verify_capture.py \
  verl/tests/trainer/ppo/test_opd_proxy_verify_capture_launcher.py \
  verl/tests/workers/rollout/test_opd_proxy_verify_native_n.py \
  verl/tests/trainer/config/test_algo_config_on_cpu.py -q
```

Expected: pilot tests fail with the current `native_rollouts must equal 4` / `requires exactly 4 completions` checks.

- [ ] **Step 4: Implement stage-aware capture validation**

Use an OmegaConf-safe field type for `stage` (`Any` with explicit runtime validation is acceptable; an unsupported structured union is not). Add optional `algorithm_contract_sha256` to the dataclass/YAML.

Normalize stage values to `0`, `1`, `2`, or `"efficacy_pilot"`. Enforce:

```python
if stage == "efficacy_pilot":
    if native_rollouts != 1 or engine_seed != 42:
        raise ValueError("efficacy_pilot capture requires seed 42 and native_rollouts 1")
    algorithm_contract_sha256 = _require_sha256(...)
else:
    if native_rollouts != 4 or engine_seed not in {42, 43}:
        raise ValueError("numbered-stage capture requires seeds 42/43 and native_rollouts 4")
```

Include the algorithm-contract hash in capture parent hashes and completion identity.

- [ ] **Step 5: Generalize native capture expansion only to approved counts**

Allow native capture `n in {1, 4}` in vLLM, continue to deep-copy sampling parameters, set `sampling_params.n`, perform exactly one `generate` call, expand prompt-major outputs, and always attach rollout slots. Use `opd_proxy_verify_stage` metadata to require `n=1` only for `efficacy_pilot` and `n=4` only otherwise.

The ray trainer must pass the normalized stage metadata and continue to build expected compound keys with `range(contract["native_rollouts"])`; do not introduce a second generation call.

- [ ] **Step 6: Run focused VERL tests**

Run the command from Step 3.

Expected: all tests pass, including both native `n=1` and native `n=4` one-call paths.

- [ ] **Step 7: Commit**

```bash
git add verl/verl/trainer/config/algorithm.py \
  verl/verl/trainer/config/ppo_trainer.yaml \
  verl/verl/trainer/ppo/opd_proxy_verify_capture.py \
  verl/verl/trainer/ppo/ray_trainer.py \
  verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py \
  verl/tests/trainer/ppo/test_opd_proxy_verify_capture.py \
  verl/tests/trainer/ppo/test_opd_proxy_verify_capture_launcher.py \
  verl/tests/workers/rollout/test_opd_proxy_verify_native_n.py \
  verl/tests/trainer/config/test_algo_config_on_cpu.py
git commit -m "feat: capture efficacy pilot with native n1"
```

---

### Task 4: Replay Single Pilot Trajectories as `P_pilot` and `T_pilot`

**Files:**
- Modify: `math_eval/replay_opd_proxy_gradients.py`
- Modify: `math_eval/test_replay_opd_proxy_gradients.py`

**Interfaces:**
- Consumes: completed pilot capture roots with exact stage/n/source/algorithm identity.
- Produces: one `P_pilot` candidate vector or one `T_pilot` candidate/held-out vector per stable ID, with no group averaging.

- [ ] **Step 1: Add failing single-trajectory replay tests**

Add tests that construct one seed-42/slot-0 trajectory and assert:

```python
proxy = replay_pilot_trajectory(actor, trajectory, projector, pair="proxy")
target = replay_pilot_trajectory(actor, trajectory, projector, pair="target")
assert proxy.representation == "P_pilot"
assert target.representation == "T_pilot"
assert proxy.aggregation == target.aggregation == "single_trajectory"
assert proxy.rollout_slot == target.rollout_slot == 0
assert proxy.vector_id == "P_pilot:q0:seed=42"
assert target.vector_id == "T_pilot:q0:seed=42"
```

Add capture-shard tests that expect exactly one vector per group, reject slot 1, reject a four-slot capture in pilot mode, and prove no vector ID contains `n4` or `P_n4`.

- [ ] **Step 2: Run focused replay tests and verify missing pilot path**

Run:

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest \
  math_eval/test_replay_opd_proxy_gradients.py \
  -k 'pilot or single_trajectory' -q
```

Expected: failures because `replay_pilot_trajectory` and pilot vector IDs do not exist.

- [ ] **Step 3: Implement dynamic exact-slot grouping**

Change `_group_capture_trajectories` to accept `native_rollouts`. For one rollout, require exactly slot `(0,)`; for numbered stages retain exact `(0,1,2,3)`. Keep compound-key contiguity and duplicate-group rejection.

Refactor the existing single trajectory backward into a shared helper that accepts representation, vector ID, and aggregation. Define:

```python
def pilot_vector_id(pair: str, stable_id: str, engine_seed: int) -> str:
    prefix = "P_pilot" if pair == "proxy" else "T_pilot"
    return f"{prefix}:{stable_id}:seed={engine_seed}"


def replay_pilot_trajectory(actor, trajectory, projector, *, pair, config=ProjectionConfig()):
    if trajectory.engine_seed != 42 or trajectory.rollout_slot != 0:
        raise ValueError("efficacy_pilot replay requires seed 42 slot 0")
    representation = "P_pilot" if pair == "proxy" else "T_pilot"
    return _replay_single(..., representation=representation,
                          vector_id=pilot_vector_id(...),
                          aggregation="single_trajectory")
```

The target pilot uses the same per-trajectory token-mean backward and diagnostics; it must not call the target group accumulator.

- [ ] **Step 4: Make `run_capture_replay_shard` identity-aware**

Add an explicit expected stage argument. Read and verify the completed capture manifest's stage, native-rollout count, source snapshot, algorithm-contract hash, and pair. For `efficacy_pilot`, shard question groups and return one record per group through `replay_pilot_trajectory`. Retain existing numbered-stage behavior byte-for-byte.

- [ ] **Step 5: Run focused and full replay CPU tests**

Run:

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest \
  math_eval/test_replay_opd_proxy_gradients.py -q
```

Expected: all non-GPU replay tests pass.

- [ ] **Step 6: Commit**

```bash
git add math_eval/replay_opd_proxy_gradients.py \
  math_eval/test_replay_opd_proxy_gradients.py
git commit -m "feat: replay single efficacy pilot gradients"
```

---

### Task 5: Parameterize the Unified Selector and Pilot Random Schedules

**Files:**
- Modify: `math_eval/select_opd_proxy_gradient_verify.py`
- Modify: `math_eval/test_select_opd_proxy_gradient_verify.py`

**Interfaces:**
- Consumes: `P_pilot` vectors, 250 ordered candidate rows, pilot profile, and parent hashes.
- Produces: two primary-K and two diagnostic-K selected sets plus independent 10,000-draw uniform/stratified schedules.

- [ ] **Step 1: Add failing pilot selector tests**

Add:

```python
def test_pilot_selector_uses_only_proxy_and_exact_fixed_cardinalities():
    rows = _rows(250)
    vectors = {"P_pilot": _vector_set("P_pilot", rows)}
    bundle = run_selection(
        vectors,
        stage="efficacy_pilot",
        candidate_rows=rows,
        fake_cluster_manager=_FakeClusterManager,
    )
    assert bundle.representations == ("P_pilot",)
    assert bundle.k_values == (25, 3)
    assert bundle.kmeans_seeds == (42, 43)
    assert bundle.round_robin_seed == 42
    assert len(bundle.selected_positions) == 4
    assert all(len(value) == 56 for value in bundle.selected_positions.values())
    assert {call[1] for call in _FakeClusterManager.calls} == {25, 3}
```

Add a test passing `T_pilot` into the selector and require an `unexpected representation` failure. Add random-schedule assertions:

```python
schedules = generate_random_schedules(
    rows, selected_size=56, draws=10_000, stage="efficacy_pilot"
)
assert schedules.uniform.shape == schedules.stratified.shape == (10_000, 56)
assert schedules.uniform.dtype.str == schedules.stratified.dtype.str == "<i4"
assert schedules.manifest["paired_target_seeds"] == [42]
assert schedules.manifest["stage"] == "efficacy_pilot"
assert schedules.manifest["uniform_logical_sha256"] != schedules.manifest[
    "stratified_logical_sha256"
]
```

Call generation twice and assert byte-identical arrays and logical hashes.

- [ ] **Step 2: Run focused selector tests and observe integer-stage/hardcoded-representation failures**

Run:

```bash
/home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest \
  math_eval/test_select_opd_proxy_gradient_verify.py \
  -k 'pilot or efficacy' -q
```

Expected: failures because selector stage must currently be an integer and requires all 14 representations.

- [ ] **Step 3: Parameterize expected representations and K values**

Replace global-only usage with:

```python
def expected_representations(stage: StageKind) -> tuple[str, ...]:
    return stage_profile(stage).selection_representations
```

Use the profile's explicit K values. Preserve ratio labels `primary`/`diagnostic`, but skip ratio-derived K equality only for the approved pilot diagnostic K=3; require exact profile K instead. Numbered stages retain the existing ratio-derived assertion.

- [ ] **Step 4: Parameterize random manifest target seeds**

Accept `StageKind`, validate all cardinalities through `stage_profile`, store the string stage, and set `paired_target_seeds` from `profile.generation_seeds`. Preserve the exact `SeedSequence`, spawn, choice, ordering, quartile, and largest-remainder algorithms.

- [ ] **Step 5: Run focused then full selector tests**

Run:

```bash
/home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest \
  math_eval/test_select_opd_proxy_gradient_verify.py -q
```

Expected: all tests pass and numbered-stage frozen schedule fixtures remain unchanged.

- [ ] **Step 6: Commit**

```bash
git add math_eval/select_opd_proxy_gradient_verify.py \
  math_eval/test_select_opd_proxy_gradient_verify.py
git commit -m "feat: select efficacy pilot proxy subsets"
```

---

### Task 6: Implement the Exact Pilot Go/No-Go Gate

**Files:**
- Modify: `math_eval/opd_proxy_gradient_classification.py`
- Modify: `math_eval/test_opd_proxy_gradient_classification.py`

**Interfaces:**
- Consumes: primary-K `R_uniform` percentile records for K-means seeds 42 and 43.
- Produces: `classify_efficacy_pilot(percentiles_by_kmeans_seed) -> dict[str, object]`.

- [ ] **Step 1: Write exhaustive failing boundary tests**

Use a helper:

```python
def _pilot_records(*, overrides=None):
    base = {
        42: {"g_vendi": 0.90, "coverage": 0.90, "gradient_norm": 0.25, "opd_signal": 0.25},
        43: {"g_vendi": 0.90, "coverage": 0.90, "gradient_norm": 0.25, "opd_signal": 0.25},
    }
    for seed, values in (overrides or {}).items():
        base[seed].update(values)
    return base
```

Add exact assertions:

```python
assert classify_efficacy_pilot(_pilot_records())["decision"] == "go"
assert classify_efficacy_pilot(
    _pilot_records(overrides={43: {"g_vendi": 0.60}})
)["decision"] == "no_go"
assert classify_efficacy_pilot(
    _pilot_records(overrides={42: {"coverage": 0.60}})
)["decision"] == "no_go"
assert classify_efficacy_pilot(
    _pilot_records(overrides={42: {"gradient_norm": 0.249999}})
)["decision"] == "borderline"
assert classify_efficacy_pilot(
    _pilot_records(overrides={43: {"g_vendi": 0.899999}})
)["decision"] == "borderline"
```

Add validation failures for missing/extra seed, missing metric, NaN, out-of-range percentile, diagnostic records, and stratified records.

- [ ] **Step 2: Run gate tests and verify the function is absent**

Run:

```bash
/home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest \
  math_eval/test_opd_proxy_gradient_classification.py \
  -k efficacy_pilot -q
```

Expected: import or attribute failure for `classify_efficacy_pilot`.

- [ ] **Step 3: Implement the gate literally**

Implement:

```python
_PILOT_METRICS = ("g_vendi", "coverage", "gradient_norm", "opd_signal")


def classify_efficacy_pilot(percentiles_by_kmeans_seed):
    normalized = _normalize_exact_pilot_records(percentiles_by_kmeans_seed)
    go = all(
        row["g_vendi"] >= 0.90
        and row["coverage"] >= 0.90
        and row["gradient_norm"] >= 0.25
        and row["opd_signal"] >= 0.25
        for row in normalized.values()
    )
    no_go = any(
        row["g_vendi"] <= 0.60 or row["coverage"] <= 0.60
        for row in normalized.values()
    )
    decision = "go" if go else "no_go" if no_go else "borderline"
    return {
        "status": "pilot_only",
        "decision": decision,
        "main_hypothesis": "not_evaluated",
        "stage1_thresholds_modified": False,
        "thresholds": {
            "go": {"g_vendi": 0.90, "coverage": 0.90,
                   "gradient_norm": 0.25, "opd_signal": 0.25},
            "no_go": {"g_vendi_at_or_below": 0.60,
                      "coverage_at_or_below": 0.60},
        },
        "primary_uniform_percentiles_by_kmeans_seed": normalized,
    }
```

Require exact keys `{42, 43}` and exact metric keys. Reuse `_percentile` for finite `[0,1]` validation.

- [ ] **Step 4: Run full classification tests**

Run:

```bash
/home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest \
  math_eval/test_opd_proxy_gradient_classification.py -q
```

Expected: all existing Stage-1/2 gates and new pilot tests pass.

- [ ] **Step 5: Commit**

```bash
git add math_eval/opd_proxy_gradient_classification.py \
  math_eval/test_opd_proxy_gradient_classification.py
git commit -m "feat: classify efficacy pilot go no-go gate"
```

---

### Task 7: Analyze Only `P_pilot` Selections in `T_pilot`

**Files:**
- Modify: `math_eval/analyze_opd_proxy_gradient_verify.py`
- Modify: `math_eval/test_analyze_opd_proxy_gradient_verify.py`

**Interfaces:**
- Consumes: pilot manifest, `P_pilot` candidate vectors, `T_pilot` candidate/held-out vectors, selection bundle, and both random schedules.
- Produces: byte-stable pilot JSON/Markdown reports with `pilot_only` classification and structured unavailable diagnostics.

- [ ] **Step 1: Add a synthetic pilot report fixture and failing report tests**

Extend the existing synthetic-stage fixture to create 250 candidate IDs, 84 held-out IDs, one proxy vector set, one target vector set, pilot selections for K-means seeds 42/43 and K values 25/3, and 10,000 deterministic random rows.

Assert:

```python
report = run_analysis(stage_dir=pilot_dir, reference_repo=reference_repo)
assert report["stage"] == "efficacy_pilot"
assert report["counts"] == {
    "candidate": 250,
    "held_out": 84,
    "selected": 56,
    "primary_k": 25,
    "diagnostic_k": 3,
    "random_draws": 10_000,
    "representations": 2,
}
assert report["classification"]["status"] == "pilot_only"
assert report["classification"]["decision"] in {"go", "no_go", "borderline"}
assert report["classification"]["main_hypothesis"] == "not_evaluated"
assert report["unavailable_diagnostics"]["cross_seed_oracle"]["status"] == "unavailable"
assert report["unavailable_diagnostics"]["target_seed_dependence"]["status"] == "unavailable"
assert "one generation seed and one rollout" in report[
    "unavailable_diagnostics"
]["cross_seed_oracle"]["reason"]
```

Add tests proving that only primary-K uniform percentiles are passed to `classify_efficacy_pilot`: mutate diagnostic and stratified values from 0 to 1 and require the decision to remain byte-identical. Add failures for a main `pass/fail/inconclusive`, a computed oracle, a missing K-means seed, `P_n4`, S/E, T in selector inputs, or seed-43 vectors.

- [ ] **Step 2: Run focused analyzer tests and observe hardcoded 14-representation failures**

Run:

```bash
/home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest \
  math_eval/test_analyze_opd_proxy_gradient_verify.py \
  -k 'pilot or efficacy' -q
```

Expected: failures from hardcoded `EXPECTED_REPRESENTATIONS`, `T:seed=42/43`, and target-dependence paths.

- [ ] **Step 3: Add a pilot-specific loader behind the shared entry point**

Keep `run_analysis` as the public entry. After canonical-loading the stage manifest, dispatch string stage `efficacy_pilot` to `_load_pilot_inputs`; numbered stages retain `_load_inputs` unchanged.

The pilot loader must require exactly:

```text
selection representations: P_pilot
candidate target vectors:  T_pilot, 250 IDs
held-out target vectors:   T_pilot, 84 IDs
```

Verify all vector parent hashes, stable-ID order, dimensions, metrics, source snapshot, reference commit/tree, and selection/random manifest hashes with the same strict loaders.

- [ ] **Step 4: Compute pilot metrics and classification**

Reuse `TargetSpace`, `evaluate_subset`, `evaluate_subset_table`, and `inclusive_percentile`. For each K-means seed, evaluate primary and diagnostic selected IDs against both nulls. Build the gate input only from primary/uniform fields:

```python
{
    seed: {
        "g_vendi": primary_uniform[seed]["g_vendi"],
        "coverage": primary_uniform[seed]["coverage"],
        "gradient_norm": primary_uniform[seed]["full_gradient_norm"],
        "opd_signal": primary_uniform[seed]["opd_signal_rms"],
    }
    for seed in (42, 43)
}
```

Do not call target-dependence or oracle code. Emit the two structured unavailable records and force classification through `classify_efficacy_pilot`.

- [ ] **Step 5: Render explicit pilot Markdown**

Add a branch headed `One-Time Efficacy Pilot` that states:

```text
Classification status: pilot_only
Operational decision: go|no_go|borderline
The main Stage-1 hypothesis was not evaluated.
This pilot does not establish downstream or OOD improvement.
Cross-seed oracle and target-seed dependence are unavailable because the pilot uses one generation seed and one rollout per question.
```

Include both K-means seeds, all four primary uniform percentiles, stratified diagnostics, diagnostic-K results, parent Stage-1 SHA, pilot manifest SHA, and source identity.

- [ ] **Step 6: Run focused then full analyzer tests**

Run:

```bash
/home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest \
  math_eval/test_analyze_opd_proxy_gradient_verify.py -q
```

Expected: all tests pass and repeated report generation is byte-identical.

- [ ] **Step 7: Commit**

```bash
git add math_eval/analyze_opd_proxy_gradient_verify.py \
  math_eval/test_analyze_opd_proxy_gradient_verify.py
git commit -m "feat: analyze target-space efficacy pilot"
```

---

### Task 8: Extend the Shared Orchestrator Without Adding a Pilot Executable

**Files:**
- Modify: `math_eval/run_opd_proxy_gradient_verify.py`
- Modify: `math_eval/test_run_opd_proxy_gradient_verify.py`
- Modify: `verl/examples/fire_opd/run_capture_opd_proxy_verify.sh`
- Modify: `verl/tests/trainer/ppo/test_opd_proxy_verify_capture_launcher.py`

**Interfaces:**
- Consumes: pilot manifest/profile and existing capture/replay/select/analyze modules.
- Produces: stage-aware Hydra overrides, a 14-unit pilot command graph, CPU dry-run, internal vector validation, analysis-input preparation, and pilot report validation.

- [ ] **Step 1: Add failing CLI and exact-command-graph tests**

Extend `_manifest` to accept `"efficacy_pilot"`. Assert CLI parsing accepts the string stage for run, validate, capture-work-unit, and internal commands but rejects `pilot`, `3`, and `-1`.

Add:

```python
def test_efficacy_pilot_plan_has_only_approved_work_units(tmp_path):
    commands = build_stage_commands(
        stage="efficacy_pilot",
        manifest=_manifest("efficacy_pilot"),
        repository_root=tmp_path,
        stage_directory=tmp_path / "efficacy_pilot",
        reference_repo=REFERENCE_REPO,
    )
    names = [command.name for command in commands]
    assert names[:2] == ["capture_target_seed_42", "capture_proxy_seed_42"]
    assert len([name for name in names if name.startswith("replay_target_seed_42")]) == 4
    assert len([name for name in names if name.startswith("replay_proxy_seed_42")]) == 4
    assert names[-4:] == ["validate_vectors", "select", "prepare_analysis_inputs", "analyze"]
    assert len(commands) == 14
    forbidden = ("seed_43", "direct_gradient_fixture", "collect_sft", "collect_embedding")
    assert not any(any(token in name for token in forbidden) for name in names)
```

Inspect the select argv and require one `--vector P_pilot=...` source and no `T_pilot`, P_n4, S, E, or target vector argument.

- [ ] **Step 2: Add failing Hydra contract-equality tests**

Build Stage-1 and pilot overrides, parse them into key/value dictionaries, remove only:

```text
algorithm.opd_proxy_verify_capture.stage
actor_rollout_ref.rollout.n
actor_rollout_ref.rollout.seed
algorithm.opd_proxy_verify_capture.native_rollouts
algorithm.opd_proxy_verify_capture.algorithm_contract_sha256
paths, counts, sample/source hashes
```

Assert every remaining override is equal, including response length 16,384, temperature/top-p, Vanilla loss, token mean, IS threshold 5.0, and every disabled feature. Assert pilot has only seed 42 and both rollout fields equal 1.

- [ ] **Step 3: Run orchestrator tests and observe integer/hardcoded-plan failures**

Run:

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest \
  math_eval/test_run_opd_proxy_gradient_verify.py \
  verl/tests/trainer/ppo/test_opd_proxy_verify_capture_launcher.py \
  -k 'pilot or efficacy or hydra_contract' -q
```

Expected: parser and command builder reject the string stage.

- [ ] **Step 4: Parameterize stage paths, seeds, rollouts, and command families**

Replace `_STAGE_LAYOUTS` and `SEEDS` assumptions at stage-dependent call sites with `stage_profile(stage)`. Keep global K-means seeds `(42,43)` distinct from generation seeds.

`build_capture_hydra_overrides` must use profile values, include `algorithm_contract_sha256`, and preserve every common override. The shell launcher must accept the string stage and pass it unchanged to Python/Hydra without integer arithmetic.

`build_stage_commands` must create the exact 14-unit graph above. Reuse four replay shards, one per allocated GPU in each pair's parallel group. Skip baselines and direct/resume fixtures. The target replay row total is 334 and proxy total is 250.

- [ ] **Step 5: Parameterize internal vector and analysis-input validation**

For pilot mode, validate exactly one proxy selection vector view and one target candidate/held-out representation. Build `analysis_inputs.json` with pilot-specific representation records and parent hashes. Keep Stage-2 sibling/union behavior unchanged.

- [ ] **Step 6: Force pilot report validation**

Extend `_validate_report_pair`/completion checks to require:

```python
report["stage"] == "efficacy_pilot"
report["classification"]["status"] == "pilot_only"
report["classification"]["decision"] in {"go", "no_go", "borderline"}
report["classification"]["main_hypothesis"] == "not_evaluated"
```

Reject Stage-1 classification keys in pilot reports.

- [ ] **Step 7: Prove CPU dry-run has no process-start side effect**

Add a test with `OPD_PROXY_VERIFY_DRY_RUN=1`, monkeypatch `execute_stage_commands` and runtime process start to fail if called, invoke the CLI twice, and assert identical stdout containing 14 command records and no `srun`.

- [ ] **Step 8: Run full orchestrator and launcher tests**

Run:

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest \
  math_eval/test_run_opd_proxy_gradient_verify.py \
  verl/tests/trainer/ppo/test_opd_proxy_verify_capture_launcher.py -q
```

Expected: all tests pass.

- [ ] **Step 9: Commit**

```bash
git add math_eval/run_opd_proxy_gradient_verify.py \
  math_eval/test_run_opd_proxy_gradient_verify.py \
  verl/examples/fire_opd/run_capture_opd_proxy_verify.sh \
  verl/tests/trainer/ppo/test_opd_proxy_verify_capture_launcher.py
git commit -m "feat: orchestrate isolated efficacy pilot stage"
```

---

### Task 9: Enforce Pilot/Stage-1 Provenance and Resume Isolation

**Files:**
- Modify: `math_eval/opd_proxy_gradient_verify_artifacts.py`
- Modify: `math_eval/test_opd_proxy_gradient_verify_artifacts.py`
- Modify: `math_eval/run_opd_proxy_gradient_verify.py`
- Modify: `math_eval/test_run_opd_proxy_gradient_verify.py`
- Modify: `math_eval/test_opd_proxy_gradient_source_audit.py`

**Interfaces:**
- Consumes: stage kind, sample/source/algorithm hashes, capture completion records, and parent Stage-1 SHA.
- Produces: exact work-unit identity validation that rejects cross-stage resume and a source snapshot that includes every pilot source file.

- [ ] **Step 1: Add failing cross-resume tests**

Construct otherwise identical pilot and Stage-1 work-unit contracts. Assert:

```python
assert pilot_contract["algorithm_contract_sha256"] != stage1_contract[
    "algorithm_contract_sha256"
]
with pytest.raises(ValueError, match="work-unit contract differs"):
    validate_existing_work_unit(stage1_output, pilot_contract)
with pytest.raises(ValueError, match="work-unit contract differs"):
    validate_existing_work_unit(pilot_output, stage1_contract)
```

Parameterize independent mismatches for `stage`, `native_rollouts`, generation seed set, sample manifest SHA, source snapshot SHA, and algorithm-contract SHA. Add a test showing that matching output path names do not rescue a mismatched contract.

- [ ] **Step 2: Add failing source-audit and frozen-parent tests**

Require `MAIN_SOURCE_FILES`/the source audit to include:

```text
docs/superpowers/specs/2026-07-15-opd-efficacy-pilot-design.md
docs/superpowers/plans/2026-07-15-opd-efficacy-pilot.md
math_eval/opd_proxy_gradient_stage_profiles.py
math_eval/test_opd_proxy_gradient_stage_profiles.py
```

Add a test that builds a pilot source snapshot at the pilot branch head while retaining the parent Stage-1 manifest's repository head `1748496...` only inside `parent_stage_manifest` provenance. The two source identities must differ and both hashes must validate.

- [ ] **Step 3: Run focused identity tests and observe missing fields/files**

Run:

```bash
/home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest \
  math_eval/test_opd_proxy_gradient_verify_artifacts.py \
  math_eval/test_run_opd_proxy_gradient_verify.py \
  math_eval/test_opd_proxy_gradient_source_audit.py \
  -k 'pilot or efficacy or cross_resume or source_identity' -q
```

Expected: failures because algorithm/stage identity is not yet part of every ledger and source list.

- [ ] **Step 4: Add all identity fields to immutable ledgers**

Ensure capture, replay, selection, analysis-input, report, work-unit contract, timing, and stage completion records carry the pilot source snapshot and algorithm-contract hash. Existing-output validation must compare canonical full contracts before considering completion paths reusable.

Do not weaken numbered-stage live-source validation. Pilot validation accepts the old Stage-1 source only as a hashed membership parent, never as the live pilot source.

- [ ] **Step 5: Update source snapshot inventory and static audit**

Add the new files to the exact source inventory. The audit must fail if any stage-specific `n=1`, seed, response-length, representation, or gate logic exists outside the declared profile/capture/analyzer files or if a pilot path writes under `stage_1`.

- [ ] **Step 6: Run focused identity tests, then all artifact/source tests**

Run:

```bash
/home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest \
  math_eval/test_opd_proxy_gradient_verify_artifacts.py \
  math_eval/test_opd_proxy_gradient_source_audit.py \
  math_eval/test_run_opd_proxy_gradient_verify.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Verify immutable existing artifacts**

Run:

```bash
sha256sum \
  data/opd_proxy_gradient_verify/stage_0/STAGE_COMPLETE.json \
  data/opd_proxy_gradient_verify/stage_1/manifest.json
```

Expected Stage-1 digest remains exactly `6d698d75995c777d6faaf1abd385a758977ca0350ce862284f1e9d8751eddd83`; Stage-0 digest remains `ef1948b4a1b91f74d203e32d9bd4247ea664386cc569113a2b3a5f8079fd6636`.

- [ ] **Step 8: Commit**

```bash
git add math_eval/opd_proxy_gradient_verify_artifacts.py \
  math_eval/test_opd_proxy_gradient_verify_artifacts.py \
  math_eval/run_opd_proxy_gradient_verify.py \
  math_eval/test_run_opd_proxy_gradient_verify.py \
  math_eval/test_opd_proxy_gradient_source_audit.py
git commit -m "feat: isolate efficacy pilot provenance and resume"
```

---

### Task 10: Update the Parent Protocol and Visible Status Documentation

**Files:**
- Modify: `docs/superpowers/specs/2026-07-14-vanilla-opd-proxy-gradient-selection-verify-design.md`
- Modify locally but do not track: `docs/opd_proxy_gradient_verify_implementation.md`
- Test: `math_eval/test_opd_proxy_gradient_source_audit.py`

**Interfaces:**
- Consumes: the implemented pilot profile and approved design amendment.
- Produces: a protocol amendment in the original staged-execution narrative and a visible operational status section.

- [ ] **Step 1: Add the pilot to the tracked protocol**

Insert a section between Stage 0 and Stage 1 containing:

- one-time `efficacy_pilot` purpose and `pilot_only` scope;
- parent Stage-1 manifest SHA and exact 250/84 prefix derivation;
- the only two deviations (`n=1`, seed set `[42]`);
- a table proving every other algorithm field remains frozen;
- exactly `P_pilot` and `T_pilot`;
- selector 56/25/3 and K-means seeds 42/43;
- both 10,000-draw nulls;
- exact GO/NO-GO/borderline inequalities;
- unavailable cross-seed diagnostics;
- independent source/artifact identity and no Stage-1 reuse.

Update the staged table to:

```text
Stage 0 -> efficacy_pilot -> Stage 1 -> conditional Stage 2
```

State that pilot `go/no_go/borderline` does not alter Stage-1 preregistered thresholds or produce the main pass/fail.

- [ ] **Step 2: Update the visible ignored implementation document**

Append a section to `docs/opd_proxy_gradient_verify_implementation.md` with implementation branch, current source commit field, parent manifest SHA, pilot parameters, CPU/GPU status, and a statement that no GPU work unit has started. Do not add this ignored file to Git.

- [ ] **Step 3: Run documentation/source checks**

Run:

```bash
git diff --check
/home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest \
  math_eval/test_opd_proxy_gradient_source_audit.py -q
rg -n 'efficacy_pilot|pilot_only|rollout.n = 1|selected size.*56|primary K.*25|diagnostic K.*3' \
  docs/superpowers/specs/2026-07-14-vanilla-opd-proxy-gradient-selection-verify-design.md \
  docs/opd_proxy_gradient_verify_implementation.md
```

Expected: no whitespace errors, source audit passes, and both documents contain the pilot contract.

- [ ] **Step 4: Verify the ignored status document is not staged**

Run:

```bash
git check-ignore -v docs/opd_proxy_gradient_verify_implementation.md
git status --short
```

Expected: the local exclude rule is shown; only the tracked parent protocol appears before commit.

- [ ] **Step 5: Commit the tracked protocol only**

```bash
git add docs/superpowers/specs/2026-07-14-vanilla-opd-proxy-gradient-selection-verify-design.md
git commit -m "docs: insert one-time efficacy pilot stage"
```

---

### Task 11: Run Full CPU/VERL Verification and Freeze the Pilot Source Commit

**Files:**
- Modify only if a verified regression requires a source correction; commit every correction before proceeding.
- Do not create canonical pilot artifacts in this task until all verification is green and the worktree is clean.

**Interfaces:**
- Consumes: all implemented pilot code/tests/docs.
- Produces: a clean final pilot source commit suitable for canonical provenance.

- [ ] **Step 1: Run all non-GPU `math_eval` tests**

Run:

```bash
cd /home/mchen/FiRe-OPD/.worktrees/opd-proxy-gradient-verify-impl
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest \
  $(find math_eval -maxdepth 1 -name 'test_*.py' -print | sort) \
  -m 'not gpu' -q
```

Expected: zero failures.

- [ ] **Step 2: Run focused VERL regression suites**

Run:

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/verl/bin/python -m pytest \
  verl/tests/trainer/config/test_algo_config_on_cpu.py \
  verl/tests/trainer/ppo/test_opd_proxy_verify_capture.py \
  verl/tests/trainer/ppo/test_opd_proxy_verify_capture_launcher.py \
  verl/tests/workers/rollout/test_opd_proxy_verify_native_n.py \
  verl/tests/workers/config/test_rollout_config_on_cpu.py \
  verl/tests/workers/config/test_actor_config_on_cpu.py \
  verl/tests/trainer/ppo/test_rollout_corr.py \
  verl/tests/trainer/ppo/test_rollout_corr_integration.py -q
```

Expected: zero failures.

- [ ] **Step 3: Run lint, compilation, and diff checks**

Run:

```bash
/home/mchen/miniconda3/envs/gvendi-opd/bin/python -m ruff check \
  math_eval/opd_proxy_gradient_stage_profiles.py \
  math_eval/prepare_opd_proxy_gradient_verify.py \
  math_eval/replay_opd_proxy_gradients.py \
  math_eval/select_opd_proxy_gradient_verify.py \
  math_eval/opd_proxy_gradient_classification.py \
  math_eval/analyze_opd_proxy_gradient_verify.py \
  math_eval/run_opd_proxy_gradient_verify.py \
  math_eval/test_opd_proxy_gradient_stage_profiles.py \
  math_eval/test_prepare_opd_proxy_gradient_verify.py \
  math_eval/test_replay_opd_proxy_gradients.py \
  math_eval/test_select_opd_proxy_gradient_verify.py \
  math_eval/test_opd_proxy_gradient_classification.py \
  math_eval/test_analyze_opd_proxy_gradient_verify.py \
  math_eval/test_run_opd_proxy_gradient_verify.py
/home/mchen/miniconda3/envs/gvendi-opd/bin/python -m py_compile \
  math_eval/opd_proxy_gradient_stage_profiles.py \
  math_eval/prepare_opd_proxy_gradient_verify.py \
  math_eval/replay_opd_proxy_gradients.py \
  math_eval/select_opd_proxy_gradient_verify.py \
  math_eval/opd_proxy_gradient_classification.py \
  math_eval/analyze_opd_proxy_gradient_verify.py \
  math_eval/run_opd_proxy_gradient_verify.py
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 4: Reverify immutable parent bytes and branch isolation**

Run:

```bash
test "$(git branch --show-current)" = opd-efficacy-pilot
test "$(git rev-parse opd-proxy-gradient-verify-impl)" = 174849613a9c61445765f0b913174d286f3819e4
test "$(sha256sum data/opd_proxy_gradient_verify/stage_1/manifest.json | cut -d' ' -f1)" = \
  6d698d75995c777d6faaf1abd385a758977ca0350ce862284f1e9d8751eddd83
test "$(sha256sum data/opd_proxy_gradient_verify/stage_0/STAGE_COMPLETE.json | cut -d' ' -f1)" = \
  ef1948b4a1b91f74d203e32d9bd4247ea664386cc569113a2b3a5f8079fd6636
```

Expected: all commands exit 0.

- [ ] **Step 5: Commit any final verified source correction**

If Steps 1-4 required source changes, commit them with a specific message and rerun Steps 1-4 from the beginning. If no source change was required, do not create an empty commit.

- [ ] **Step 6: Freeze and record the clean source identity**

Run:

```bash
test -z "$(git status --porcelain)"
git rev-parse HEAD
git status --short
```

Expected: a clean worktree and one final pilot source commit hash. No canonical pilot directory exists yet unless it was created by a previous matching, reviewed attempt.

---

### Task 12: Publish the Canonical Pilot Manifest, Run CPU Preflight, Report ETA, and Stop

**Files:**
- Create via canonical prepare only: `data/opd_proxy_gradient_verify/efficacy_pilot/manifest.json`
- Create via canonical prepare only: pilot sample/capture-input/source-snapshot artifacts under `data/opd_proxy_gradient_verify/efficacy_pilot/`
- Modify locally but do not track: `docs/opd_proxy_gradient_verify_implementation.md`
- Do not create any capture/replay work-unit output.

**Interfaces:**
- Consumes: clean final pilot source commit, frozen parent manifest, pinned reference checkout.
- Produces: canonical pilot manifest SHA, byte-stable dry-run hash, CPU preflight evidence, no-GPU evidence, and a conservative GPU-duration estimate.

- [ ] **Step 1: Record preflight no-GPU and immutable-parent evidence**

Run:

```bash
cd /home/mchen/FiRe-OPD/.worktrees/opd-proxy-gradient-verify-impl
PILOT_HEAD=$(git rev-parse HEAD)
test -z "$(git status --porcelain)"
sha256sum data/opd_proxy_gradient_verify/stage_0/STAGE_COMPLETE.json \
  data/opd_proxy_gradient_verify/stage_1/manifest.json
find data/opd_proxy_gradient_verify/efficacy_pilot/capture \
  data/opd_proxy_gradient_verify/efficacy_pilot/replay \
  data/opd_proxy_gradient_verify/efficacy_pilot/work_units \
  -type f -print 2>/dev/null | sort > /tmp/opd_efficacy_pilot_gpu_files.before
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader \
  > /tmp/opd_efficacy_pilot_gpu_processes.before
```

Expected: immutable hashes match the global constraints; the GPU-file list is empty. Existing unrelated GPU processes, if any, are recorded but not killed.

- [ ] **Step 2: Canonically prepare the pilot manifest on CPU**

Run:

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/gvendi-opd/bin/python -m \
  math_eval.prepare_opd_proxy_gradient_verify prepare-stage \
  --stage efficacy_pilot \
  --output-root data/opd_proxy_gradient_verify \
  --parent-stage1-manifest data/opd_proxy_gradient_verify/stage_1/manifest.json \
  --expected-parent-stage1-manifest-sha256 \
    6d698d75995c777d6faaf1abd385a758977ca0350ce862284f1e9d8751eddd83 \
  --expected-source-commit "$PILOT_HEAD" \
  --reference-repo /home/mchen/prismatic-synthesis-reference
```

Expected: create-once pilot inputs are published or matching existing bytes validate. Stage-0/1 bytes do not change.

- [ ] **Step 3: Run CPU-only stage preflight**

Run:

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/gvendi-opd/bin/python -m \
  math_eval.run_opd_proxy_gradient_verify validate-stage \
  --stage efficacy_pilot \
  --preflight-only \
  --reference-repo /home/mchen/prismatic-synthesis-reference
```

Expected: exit 0 after validating manifest, parent, source, model, tokenizer, command graph, and output namespace; no work unit starts.

- [ ] **Step 4: Generate the deterministic dry-run twice**

Run:

```bash
export PYTHONPATH="$PWD/verl:$PWD"
export OPD_PROXY_VERIFY_DRY_RUN=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
/home/mchen/miniconda3/envs/verl/bin/python -m \
  math_eval.run_opd_proxy_gradient_verify \
  --stage efficacy_pilot \
  --reference-repo /home/mchen/prismatic-synthesis-reference \
  > /tmp/opd_efficacy_pilot_dry_run_1.json
/home/mchen/miniconda3/envs/verl/bin/python -m \
  math_eval.run_opd_proxy_gradient_verify \
  --stage efficacy_pilot \
  --reference-repo /home/mchen/prismatic-synthesis-reference \
  > /tmp/opd_efficacy_pilot_dry_run_2.json
cmp /tmp/opd_efficacy_pilot_dry_run_1.json \
  /tmp/opd_efficacy_pilot_dry_run_2.json
sha256sum /tmp/opd_efficacy_pilot_dry_run_1.json
```

Expected: `cmp` exits 0; the plan contains exactly 14 units, seed 42 only, n=1 capture overrides, no baselines, no direct fixture, and no `srun`.

- [ ] **Step 5: Prove no GPU work unit or process was launched**

Run:

```bash
find data/opd_proxy_gradient_verify/efficacy_pilot/capture \
  data/opd_proxy_gradient_verify/efficacy_pilot/replay \
  data/opd_proxy_gradient_verify/efficacy_pilot/work_units \
  -type f -print 2>/dev/null | sort > /tmp/opd_efficacy_pilot_gpu_files.after
cmp /tmp/opd_efficacy_pilot_gpu_files.before \
  /tmp/opd_efficacy_pilot_gpu_files.after
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader \
  > /tmp/opd_efficacy_pilot_gpu_processes.after
cmp /tmp/opd_efficacy_pilot_gpu_processes.before \
  /tmp/opd_efficacy_pilot_gpu_processes.after
```

Expected: both comparisons exit 0. If unrelated GPU occupancy changed independently, do not kill it; inspect ownership and prove instead that no PID or command from the pilot dry-run appeared.

- [ ] **Step 6: Compute and report conservative GPU duration without launching work**

Use Stage-0 per-family timing records and the pilot command graph. For each capture/replay family, multiply the observed Stage-0 seconds per question by pilot row count, deliberately retain the slower Stage-0 `n=4` rate without dividing by four, and take the maximum duration inside each four-shard replay parallel group. Sum sequential groups. Report:

1. the resulting conservative active-GPU upper estimate;
2. the same estimate with safety factor `1.20` plus two hours reserve;
3. the fact that actual `n=1` execution should be lower but is not assumed by the safety gate.

The calculation script must read immutable timing JSON and dry-run row counts rather than typing durations manually. Save its canonical JSON output under `/tmp/opd_efficacy_pilot_eta.json`; do not add a runtime estimate to the tracked protocol.

- [ ] **Step 7: Record final hashes and status locally**

Run:

```bash
sha256sum \
  data/opd_proxy_gradient_verify/efficacy_pilot/manifest.json \
  data/opd_proxy_gradient_verify/stage_1/manifest.json \
  data/opd_proxy_gradient_verify/stage_0/STAGE_COMPLETE.json
git status --short
```

Append the pilot source commit, parent SHA, pilot manifest SHA, dry-run SHA, CPU preflight result, ETA, and `GPU work units started: no` to `docs/opd_proxy_gradient_verify_implementation.md`. Confirm it remains ignored and `git status --porcelain` remains empty.

- [ ] **Step 8: Stop and report**

Report only:

- final pilot source commit;
- frozen Stage-1 parent SHA;
- pilot manifest SHA;
- deterministic dry-run SHA and 14-unit summary;
- CPU preflight result;
- conservative expected GPU duration;
- explicit confirmation that no GPU work unit was started;
- the command that would launch pilot later, clearly marked as not executed.

Do not run the launch command and do not proceed to Stage 1.
