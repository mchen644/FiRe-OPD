# Vanilla-OPD Proxy-Gradient Selection Verify Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a frozen-model verification pipeline that tests whether subsets selected by `Qwen3-4B -> Qwen3-0.6B` vanilla-OPD gradients beat same-size random subsets when evaluated in the true `Qwen3-30B-A3B-Instruct-2507 -> Qwen3-4B` vanilla-OPD gradient space.

**Architecture:** A CPU manifest layer freezes the benchmark-clean population, stage membership, model/tokenizer contracts, and provenance. A default-off VERL capture path performs one native `n=4` vLLM call per pair/seed without updating an actor, after which standalone full-parameter replay produces strict TRAK projections; separate collectors produce SFT and prompt-embedding baselines. A unified selector writes deterministic selected sets and random schedules, and a model-free analyzer computes target-space metrics and the pre-registered classifications.

**Tech Stack:** Python 3.10, PyTorch 2.6.0, VERL/Ray/FSDP, vLLM, Transformers 4.51.1, NumPy 1.26.4, SciPy 1.15.3, scikit-learn 1.6.1, safetensors, TRAK `CudaProjector`, Prismatic Synthesis reference code, pytest, Bash, Slurm `opd-CLI`.

## Global Constraints

- The scientific contract is [the approved design](../specs/2026-07-14-vanilla-opd-proxy-gradient-selection-verify-design.md); do not revive the superseded Step-0/group-success probe.
- This is verification only: never run an optimizer step, update a scheduler, train a checkpoint, or claim downstream OOD improvement.
- Start implementation in an isolated worktree from the commit that contains this implementation plan and whose parent is `3b0d1a2`; the current workspace contains unrelated user changes and must not be staged, overwritten, or committed.
- The primary workspace `/home/mchen/FiRe-OPD` remains the owner of ignored model, large-data, output, and log assets. The isolated worktree uses ignored symlinks to those assets; source-code edits and commits remain isolated.
- Stage sizes are fixed: smoke `24/8/5`, Stage 1 `768/256/172`, and conditional Stage 2 `1,536/512/345` for candidate/held-out/selected counts.
- The root population is the 57,045-ID eligibility contract. The full benchmark audit must reproduce 383 rejected IDs, 56,662 clean IDs, and every hash pinned in the design before model work starts.
- Target is `models/Qwen3-30B-A3B-Instruct-2507 -> models/Qwen3-4B`; proxy is the byte-identical same 4B checkpoint as teacher into `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`, materialized as `models/Qwen3-0.6B`.
- Verify exact vocabulary-to-ID, EOS, and padding-token compatibility independently for 30B/4B and 4B/0.6B; model-family names are not proof.
- Prismatic reference checkout must equal commit `d9484cd3b5991030b901ac4a3a9e2472dbfac2ad` and tree `a0079d8c5e15cb18bb4790f99c43cc19bb9ecd50`.
- Capture uses raw OPD prompts, thinking disabled, one native `n=4` request, temperature/top-p `1.0/1.0`, max prompt/response `2,048/16,384`, engine seeds 42 and 43, and exactly one generate call per stage/pair/seed work unit.
- Vanilla OPD is fixed to one PPO epoch, one mini-batch per actor rank, micro-batch one, dynamic batching off, token-mean loss, reverse-KL advantages only, token IS upper-clamped at 5 with no lower clamp or normalization, explicit KL worker enabled with coefficient zero, and all compression/routing/selection/entropy extensions absent.
- Keep `rollout_log_prob`, trainer `batch_old_log_prob`, actor `current_log_prob`, actor-local `current.detach()`, and teacher `ref_log_prob` semantically distinct. Replay advantage must use actor-local stopped current log-probability.
- Capture is default-off and side-effect-free for normal training. In capture mode no optimizer is constructed; a fallback optimizer step is a fail-fast error.
- Full gradients are native float32 before projection. Project only with the pinned Rademacher `CudaProjector` to 1,024 dimensions using seed 0, block size 128, max batch 16, model ID 0, float16 projector input, division by `sqrt(1024)`, and float32 storage. There is no CPU projector fallback.
- Never compare raw vectors from different parameter spaces. Cross-representation comparisons operate on aligned within-representation Gram matrices, partitions, or selected IDs.
- All joins use `(stable_id, engine_seed, rollout_slot)`, never incidental row order. Every tensor chunk has an ID sidecar, parent hashes, and atomic publication.
- Source snapshots include byte size and SHA-256 for every affecting tracked or untracked file. A git diff hash alone is insufficient.
- CUDA model loading, rollout, backward, projection, and official K-means run only inside the user's four-GPU `opd-CLI` allocation. Login-shell GPU execution is forbidden.
- Stage 1 cannot start until every Stage-0 real-model gate passes. Stage 2 runs only through the predeclared Stage-1 trigger and can report `extended_pass`, `extended_fail`, or `extended_inconclusive` without changing Stage 1.

---

## File Map

### New shared and CPU pipeline files

- `math_eval/opd_proxy_gradient_verify_artifacts.py`: canonical hashes, source snapshots, atomic artifacts, strict vector/capture loaders, and shared dataclasses.
- `math_eval/prepare_opd_proxy_gradient_verify.py`: full-population decontamination, one-permutation sampling contract, stage views, token preflight, tokenizer checks, and capture parquet creation.
- `math_eval/opd_proxy_gradient_projection.py`: strict Prismatic/fast-JL reference verification and the only permitted `CudaProjector` wrapper.
- `math_eval/replay_opd_proxy_gradients.py`: standalone exact actor replay, target group accumulation, proxy per-slot replay, projection, resume, and global validation.
- `math_eval/collect_opd_proxy_sft_gradients.py`: official completion-only Prismatic SFT-gradient baseline over candidate IDs.
- `math_eval/collect_opd_proxy_prompt_embeddings.py`: Qwen3-0.6B raw-prompt mean-pooled embedding baseline.
- `math_eval/select_opd_proxy_gradient_verify.py`: official cosine K-means, fixed round robin, random schedules, and selection artifacts.
- `math_eval/opd_proxy_gradient_statistics.py`: pure target-space G-Vendi, coverage, null, CKA, partition, overlap, and diagnostic statistics.
- `math_eval/opd_proxy_gradient_classification.py`: exact oracle, candidate, Stage-1, and Stage-2 gates.
- `math_eval/analyze_opd_proxy_gradient_verify.py`: strict artifact loading plus byte-stable JSON/Markdown reports.
- `math_eval/run_opd_proxy_gradient_verify.py`: fail-closed stage orchestrator and CPU/GPU boundary checks.
- `pytest.ini`: registration for the allocation-only `gpu` test marker.

### New VERL capture files

- `verl/verl/trainer/ppo/opd_proxy_verify_capture.py`: compound keys, authoritative capture tensors, chunk/resume semantics, and actor-parameter hashing.
- `verl/examples/fire_opd/run_capture_opd_proxy_verify.sh`: one pair/seed native-`n=4` capture launcher.
- `verl/examples/fire_opd/run_direct_opd_proxy_gradient_fixture.sh`: one-trajectory production-backward smoke fixture.

### Existing files modified only for the default-off capture path

- `verl/verl/workers/config/rollout.py` and `verl/verl/trainer/config/rollout/rollout.yaml`: typed rollout seed.
- `verl/verl/workers/config/actor.py` and `verl/verl/trainer/config/actor/actor.yaml`: capture-only actor flag.
- `verl/verl/trainer/config/algorithm.py` and `verl/verl/trainer/config/ppo_trainer.yaml`: typed capture contract.
- `verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`: opt-in one-call native `n=4` expansion.
- `verl/verl/trainer/ppo/ray_trainer.py`: dedicated capture-only orchestration before the normal training loop.
- `verl/verl/workers/fsdp_workers.py`: optimizer-free actor initialization and capture RPC.
- `verl/verl/workers/actor/dp_actor.py`: forward-only authoritative capture plus smoke-only direct-gradient hook.
- `verl/verl/trainer/main_ppo.py`: deterministic capture shutdown and no unnecessary validation/reward initialization.

### New tests

- `math_eval/test_opd_proxy_gradient_verify_artifacts.py`
- `math_eval/test_prepare_opd_proxy_gradient_verify.py`
- `math_eval/test_opd_proxy_gradient_projection.py`
- `math_eval/test_replay_opd_proxy_gradients.py`
- `math_eval/test_collect_opd_proxy_sft_gradients.py`
- `math_eval/test_collect_opd_proxy_prompt_embeddings.py`
- `math_eval/test_select_opd_proxy_gradient_verify.py`
- `math_eval/test_opd_proxy_gradient_statistics.py`
- `math_eval/test_opd_proxy_gradient_classification.py`
- `math_eval/test_analyze_opd_proxy_gradient_verify.py`
- `math_eval/test_run_opd_proxy_gradient_verify.py`
- `verl/tests/workers/rollout/test_opd_proxy_verify_native_n.py`
- `verl/tests/trainer/ppo/test_opd_proxy_verify_capture.py`
- `verl/tests/trainer/ppo/test_opd_proxy_verify_capture_launcher.py`

### Execution Worktree Bootstrap

Before invoking the required `using-git-worktrees` skill, prove that this plan is tracked and identify its immutable plan commit:

```bash
PLAN_FILE=docs/superpowers/plans/2026-07-14-vanilla-opd-proxy-gradient-selection-verify.md
git ls-files --error-unmatch "$PLAN_FILE"
PLAN_COMMIT="$(git log -1 --format=%H -- "$PLAN_FILE")"
test "$(git rev-parse "$PLAN_COMMIT^")" = "$(git rev-parse 3b0d1a2)"
test "$(git rev-parse "$PLAN_COMMIT:$PLAN_FILE")" = "$(git hash-object "$PLAN_FILE")"
```

Create the isolated worktree at `PLAN_COMMIT`, then enter it and connect only ignored asset directories to the primary workspace:

```bash
PRIMARY_REPO=/home/mchen/FiRe-OPD
mkdir -p data "$PRIMARY_REPO/data/opd_proxy_gradient_verify" \
  "$PRIMARY_REPO/logs/opd_proxy_gradient_verify"
ln -s "$PRIMARY_REPO/models" models
ln -s "$PRIMARY_REPO/data/g-opd" data/g-opd
ln -s "$PRIMARY_REPO/data/gradient_diversity" data/gradient_diversity
ln -s "$PRIMARY_REPO/data/opd_proxy_gradient_verify" data/opd_proxy_gradient_verify
ln -s "$PRIMARY_REPO/logs" logs
```

Expected: `git status --short` remains clean because all five paths are ignored; tracked benchmark JSONLs remain ordinary files inside the worktree's existing `data/` directory. Refuse to replace any existing non-symlink path.

### Task 1: Shared Artifact and Provenance Contract

**Files:**
- Create: `math_eval/opd_proxy_gradient_verify_artifacts.py`
- Test: `math_eval/test_opd_proxy_gradient_verify_artifacts.py`

**Interfaces:**
- Consumes: standard-library paths/hashes, NumPy arrays, safetensors tensors.
- Produces: `SourceFileSpec`, `TrajectoryKey`, `VectorSet`, `canonical_json_bytes`, `sha256_id_lines`, `sha256_int_rows`, `sha256_file`, `recursive_file_manifest`, `build_source_snapshot`, `atomic_write_json`, `atomic_write_jsonl`, `atomic_save_npy`, `atomic_save_safetensors`, `write_or_validate_manifest`, `load_vector_set`, and `validate_exact_key_coverage`.

- [ ] **Step 1: Write the failing canonicalization, snapshot, atomicity, and coverage tests**

```python
def test_sha256_id_lines_uses_utf8_terminal_newlines():
    assert sha256_id_lines(["a", "β"]) == (
        "d3c5672deb0c99f78c72cf77d08b03ca61f7685b7fb1689d9861bcfd48afe094"
    )


def test_sha256_int_rows_uses_compact_json_and_terminal_newlines():
    rows = np.array([[0, 2], [1, 3]], dtype=np.int32)
    assert sha256_int_rows(rows) == (
        "74201e550190c3ead9a6c11a336e2d25ed25cb5d36b30166eba198b0596104c2"
    )


def test_source_snapshot_hashes_untracked_file_bytes(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "untracked.py"
    source.write_bytes(b"VALUE = 1\n")
    first = build_source_snapshot(
        {"main": repo}, [SourceFileSpec("main", "untracked.py")]
    )
    source.write_bytes(b"VALUE = 2\n")
    second = build_source_snapshot(
        {"main": repo}, [SourceFileSpec("main", "untracked.py")]
    )
    assert first["files"][0]["sha256"] != second["files"][0]["sha256"]
    assert first["manifest_sha256"] != second["manifest_sha256"]


def test_recursive_model_manifest_is_relative_sorted_and_byte_sensitive(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "b.bin").write_bytes(b"b")
    (model / "a.json").write_bytes(b"a")
    first = recursive_file_manifest(model)
    assert [row["path"] for row in first["files"]] == ["a.json", "b.bin"]
    (model / "b.bin").write_bytes(b"changed")
    assert recursive_file_manifest(model)["manifest_sha256"] != first["manifest_sha256"]


def test_key_coverage_rejects_duplicate_and_missing_slots():
    expected = {
        TrajectoryKey("q0", 42, 0),
        TrajectoryKey("q0", 42, 1),
    }
    with pytest.raises(ValueError, match="duplicate trajectory key"):
        validate_exact_key_coverage(
            [TrajectoryKey("q0", 42, 0), TrajectoryKey("q0", 42, 0)],
            expected,
        )
    with pytest.raises(ValueError, match="missing trajectory keys"):
        validate_exact_key_coverage([TrajectoryKey("q0", 42, 0)], expected)
```

Also test that interrupted temporary files are ignored, a manifest with a changed parent hash is rejected, `.npy` reload enforces little-endian C-contiguous dtype/shape, and vector sidecar keys reorder to the requested stable-ID order rather than trusting tensor row order.

- [ ] **Step 2: Run the focused tests and confirm the missing module failure**

Run:

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest -q \
  math_eval/test_opd_proxy_gradient_verify_artifacts.py
```

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'math_eval.opd_proxy_gradient_verify_artifacts'`.

- [ ] **Step 3: Implement the shared contracts and atomic writers**

Start the module with these exact public records and hash rules:

```python
@dataclass(frozen=True, order=True)
class TrajectoryKey:
    stable_id: str
    engine_seed: int
    rollout_slot: int


@dataclass(frozen=True)
class SourceFileSpec:
    repository: str
    path: str


@dataclass(frozen=True)
class VectorSet:
    vector_ids: tuple[str, ...]
    stable_ids: tuple[str, ...]
    vectors: np.ndarray
    full_gradient_norm: np.ndarray | None
    projected_gradient_norm: np.ndarray | None
    valid_token_count: np.ndarray | None
    response_length: np.ndarray | None
    sampled_reverse_kl: np.ndarray | None
    opd_signal_rms: np.ndarray | None
    verifier_correct_count: np.ndarray | None
    verifier_total: np.ndarray | None
    manifest: dict[str, object]


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_id_lines(ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for stable_id in ids:
        digest.update(stable_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def sha256_int_rows(rows: np.ndarray) -> str:
    digest = hashlib.sha256()
    for row in np.asarray(rows):
        payload = json.dumps(
            [int(value) for value in row],
            ensure_ascii=True,
            separators=(",", ":"),
        )
        digest.update(payload.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
```

Implement all loaders as fail-closed validators: reject unknown fields, duplicate IDs, non-finite vectors, zero vectors, sidecar/tensor row mismatches, non-contiguous resume prefixes, changed parent/source hashes, and incomplete `COMPLETE.json`. `build_source_snapshot` must sort `(repository, path)` and hash actual bytes, regardless of git tracking state. `recursive_file_manifest` walks regular files in sorted repository-relative order, records relative path/size/SHA-256, rejects broken or root-escaping symlinks, and hashes the canonical file list; use it for every model and tokenizer directory.

Every root/stage/vector manifest also records repository HEAD, full porcelain-v1 status text and its SHA-256, the exact Python executable/version, a `runtime_profile`, and one common package-version map. Resolve profile-required packages and fail if any are missing: `verl_capture` requires torch, transformers, vLLM, NumPy, safetensors, and pyarrow; `gvendi_analysis` requires torch, transformers, NumPy, SciPy, scikit-learn, safetensors, pyarrow, traker, and fast-jl. Keep every schema key in both profiles, but record JSON `null` for a non-required unavailable package rather than making the VERL environment depend on analysis-only packages. Tests inject a fake version resolver and cover required-missing failure, optional-missing nulls, and a changed package version.

- [ ] **Step 4: Run the artifact tests and existing manifest regressions**

Run:

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest -q \
  math_eval/test_opd_proxy_gradient_verify_artifacts.py \
  math_eval/test_build_gradient_eligibility.py \
  math_eval/test_deepmath_gradient_diversity.py
```

Expected: all tests PASS; no file is written outside pytest temporary directories.

- [ ] **Step 5: Commit the artifact layer**

```bash
git add math_eval/opd_proxy_gradient_verify_artifacts.py \
  math_eval/test_opd_proxy_gradient_verify_artifacts.py
git commit -m "feat: add OPD proxy artifact contracts"
```

### Task 2: Full-Population Decontamination and Frozen Sampling Contract

**Files:**
- Create: `math_eval/prepare_opd_proxy_gradient_verify.py`
- Test: `math_eval/test_prepare_opd_proxy_gradient_verify.py`
- Reuse: `math_eval/build_gradient_eligibility.py`
- Reuse: `math_eval/deepmath_gradient_diversity.py`

**Interfaces:**
- Consumes: `load_prepared_pool`, `apply_eligibility_report`, `sha256_id_lines`, the eight pinned evaluation JSONLs, and seed `2026071401`.
- Produces: `DecontaminationAudit`, `StageLayout`, `normalize_question`, `ten_token_gram_hashes`, `build_decontamination_audit`, `build_production_contract`, `build_sampling_contract`, `stage_layout`, and root artifacts under `data/opd_proxy_gradient_verify/`.

- [ ] **Step 1: Write failing exact/10-gram, one-permutation, stage-layout, and production-hash tests**

```python
def test_normalized_exact_and_shared_ten_gram_are_rejected():
    evaluation = ["Find x when alpha beta gamma delta epsilon zeta eta theta iota kappa."]
    training = [
        {"id": "q0", "prompt": "  FIND x when alpha beta gamma delta epsilon zeta eta theta iota kappa.  "},
        {"id": "q1", "prompt": "prefix alpha beta gamma delta epsilon zeta eta theta iota kappa suffix"},
        {"id": "q2", "prompt": "a genuinely unrelated problem"},
    ]
    audit = build_decontamination_audit(training, {"eval": evaluation})
    assert audit.clean_mask.tolist() == [False, False, True]
    assert [hit.stable_id for hit in audit.hits] == ["q0", "q1"]


def test_production_clean_population_and_sampling_hashes(production_inputs):
    audit, contract = build_production_contract(**production_inputs)
    assert audit.exact_match_count == 0
    assert len(audit.hits) == 383
    assert int(audit.clean_mask.sum()) == 56_662
    assert audit.clean_ids_sha256 == "88a1a21ddba4a8aa460963bb404c7bc8f3bf7f6373155fba5eebaad8e52606ee"
    assert audit.rejected_ids_sha256 == "ba92df92881df385d9662c3a7305d05fe73f69e10cd5b3b3fb9a18e88f8699a6"
    assert contract.scanned_permutation_stop == 2064
    assert len(contract.skipped_before_cutoff) == 16
    assert contract.first_2048_ids_sha256 == "aebf0d9a4743b3bb6d2ee307b6194d3d95a948f699a4439b8c4d43f498efbfc4"
    assert contract.stage_hashes == {
        "stage0_all": "bf4822f0cab82aa5461b30247ee9149506618e61ea087486d27712b2227709df",
        "stage1_candidate": "bd4b790e7db49cabc85aad41de792cce436ce60ee004be505554fd83a1a55e9e",
        "stage1_held_out": "e867617c9b33e6d8881af7dd8b651f64ea7c27ef990f3153222ff34858e33fc7",
        "stage1_all": "5d84fdbc96212e9915ecb32f0ef5b48d437876ec0c02efc801a43e777274acf6",
        "stage2_added_candidate": "4ef21f73c8306702a414964cc817bfbb501764e9af82e311d89ce16230369753",
        "stage2_added_held_out": "dcca926e2ad4a7b0b9be909446ed398f7339cbd988acb2fd69a782a8865397b8",
        "stage2_candidate": "e9b0fa16c2cbeb8e61e0f9e8033a973dc2ab41ce2d2ff93157d53eb012a47811",
        "stage2_held_out": "7dde4b385a7f01cd410bd0f8d0c43a6874a0b94ead78505b4f3ad93aea112ecf",
        "stage2_all": "047ffab516afafc6bc9820351e497ecbbccd240ca4f1504c6c90d583ca040eff",
    }
```

Add an injected RNG spy that asserts exactly one call to `permutation(57_045)`; the dictionary above is the complete ordered-ID hash fixture.

- [ ] **Step 2: Run the preparation tests and confirm the missing symbols**

Run:

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest -q \
  math_eval/test_prepare_opd_proxy_gradient_verify.py \
  -k 'decontamination or sampling or stage_layout'
```

Expected: FAIL with an import error for `build_decontamination_audit`.

- [ ] **Step 3: Implement normalization, full-pool audit, the single filtered permutation, and stage views**

Use these exact normalization and layout rules:

```python
OPD_SUFFIX = "Please reason step by step, and put your final answer within \\boxed{}."
TEN_TOKEN_RE = re.compile(r"\\[A-Za-z]+|[A-Za-z0-9]+")


@dataclass(frozen=True)
class StageLayout:
    stage: int
    candidate_clean_positions: tuple[int, ...]
    held_out_clean_positions: tuple[int, ...]
    selected_size: int
    primary_k: int
    diagnostic_k: int
    null_draws: int


@dataclass(frozen=True)
class DecontaminationHit:
    stable_id: str
    eligible_position: int
    normalized_exact: bool
    benchmark_path: str
    benchmark_sha256: str
    shared_gram_hash: str | None
    shared_gram_text: str | None


@dataclass(frozen=True)
class DecontaminationAudit:
    clean_mask: np.ndarray
    hits: tuple[DecontaminationHit, ...]
    exact_match_count: int
    clean_ids_sha256: str
    rejected_ids_sha256: str


@dataclass(frozen=True)
class SamplingContract:
    first_2048_ids: tuple[str, ...]
    first_2048_ids_sha256: str
    scanned_permutation_stop: int
    skipped_before_cutoff: tuple[dict[str, object], ...]
    stage_hashes: dict[str, str]


def normalize_question(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    stripped = normalized.rstrip()
    if stripped.endswith(OPD_SUFFIX):
        stripped = stripped[: -len(OPD_SUFFIX)].rstrip()
    return " ".join(stripped.casefold().split())


def ten_token_gram_hashes(text: str) -> frozenset[str]:
    tokens = TEN_TOKEN_RE.findall(normalize_question(text))
    return frozenset(
        hashlib.sha256("\u001f".join(tokens[index : index + 10]).encode("utf-8")).hexdigest()
        for index in range(max(0, len(tokens) - 9))
    )


def stage_layout(stage: int) -> StageLayout:
    if stage == 0:
        return StageLayout(0, tuple(range(24)), tuple(range(768, 776)), 5, 2, 2, 100)
    if stage == 1:
        return StageLayout(1, tuple(range(768)), tuple(range(768, 1024)), 172, 76, 7, 10_000)
    if stage == 2:
        candidates = tuple(range(768)) + tuple(range(1024, 1792))
        held_out = tuple(range(768, 1024)) + tuple(range(1792, 2048))
        return StageLayout(2, candidates, held_out, 345, 153, 15, 10_000)
    raise ValueError(f"unsupported stage: {stage}")
```

`build_sampling_contract` must instantiate `Generator(PCG64(2026071401))`, call `permutation` once, preserve original eligible permutation ranks, filter the immutable clean mask, and freeze the first 2,048 clean IDs. Publish atomically:

```text
data/opd_proxy_gradient_verify/decontamination.mask.npy
data/opd_proxy_gradient_verify/decontamination_hits.jsonl
data/opd_proxy_gradient_verify/decontamination_manifest.json
data/opd_proxy_gradient_verify/sampling_contract.json
```

Every hit row records stable ID, eligible position, normalized-exact status, benchmark path/hash, and lexicographically first shared gram hash/text. Refuse any count or hash that differs from the approved production fixtures.

- [ ] **Step 4: Run the focused and full production CPU audit tests**

Run:

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest -q \
  math_eval/test_prepare_opd_proxy_gradient_verify.py \
  math_eval/test_build_gradient_eligibility.py
```

Expected: all tests PASS; the production fixture reports `57045/383/56662`, stop rank `2063`, and 16 skipped positions.

- [ ] **Step 5: Commit the frozen sampling contract**

```bash
git add math_eval/prepare_opd_proxy_gradient_verify.py \
  math_eval/test_prepare_opd_proxy_gradient_verify.py
git commit -m "feat: freeze benchmark-clean OPD proxy sample"
```

### Task 3: Qwen3 Materialization, Tokenizer Compatibility, Stage Preflight, and Capture Data

**Files:**
- Modify: `math_eval/prepare_opd_proxy_gradient_verify.py`
- Modify: `math_eval/test_prepare_opd_proxy_gradient_verify.py`

**Interfaces:**
- Consumes: frozen `sampling_contract.json`, Qwen3 tokenizers, raw prepared rows, and the approved parent hashes.
- Produces: `build_raw_opd_prompt`, `validate_tokenizer_compatibility`, `preflight_stage_rows`, `build_stage_rows`, `prepare_stage`, `stage_{0,1,2}/sample_manifest.jsonl`, `stage_{0,1,2}/manifest.json`, and target/proxy capture parquets.

- [ ] **Step 1: Add failing tokenizer, chat-template, context, leaf-topic, and stage-parent tests**

```python
def test_tokenizer_compatibility_checks_every_id_and_special_token(fake_tokenizer):
    left = fake_tokenizer({"a": 0, "b": 1}, eos=2, pad=3)
    right = fake_tokenizer({"a": 0, "b": 4}, eos=2, pad=3)
    with pytest.raises(ValueError, match="token ID mismatch for 'b'"):
        validate_tokenizer_compatibility(left, right, pair_name="proxy")


def test_stage_rows_use_raw_prompt_and_completion_only_sft_labels(fake_qwen_tokenizer):
    rows = preflight_stage_rows([PREPARED_ROW], fake_qwen_tokenizer, fake_qwen_tokenizer)
    assert rows[0]["raw_opd_messages"] == [
        {
            "role": "user",
            "content": PREPARED_ROW["prompt"].rstrip() + "\n" + OPD_SUFFIX,
        }
    ]
    assert rows[0]["leaf_topic"] == "Other"
    assert rows[0]["sft_supervised_label_count"] > 0
    assert fake_qwen_tokenizer.embedding_call == {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
```

Test that 4B and 0.6B prompt overflow, 0.6B full-SFT overflow, missing/empty topic components, zero supervised labels, a Stage-2 manifest without the exact Stage-1 report hash, a sampled benchmark collision, a dirty/wrong `--expected-source-commit`, a mismatched `--reference-repo` commit/tree, disposable mode targeting the canonical root, and canonical mode missing either provenance argument all fail before writing a manifest.

- [ ] **Step 2: Run the new preflight tests and verify they fail**

Run:

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest -q \
  math_eval/test_prepare_opd_proxy_gradient_verify.py \
  -k 'tokenizer or preflight or stage_parent or capture_parquet'
```

Expected: FAIL because `validate_tokenizer_compatibility` and `prepare_stage` are not implemented.

- [ ] **Step 3: Implement strict preflight and stage artifacts**

Implement compatibility by comparing the complete `get_vocab()` token-to-ID mappings and then `eos_token_id`, `pad_token_id`, and tokenizer length. Join every prepared row to its pinned source-parquet row by `source_row_index`. Validate that stripping only `OPD_SUFFIX` from the source user content gives the prepared exact question, then construct the raw OPD prompt by appending exactly one newline and `OPD_SUFFIX` to that exact question. Preserve the prepared row's canonical `prompt` (exact question) and `completion` (R1 solution) fields byte-for-byte in every stage manifest, and preserve the source `reward_model` unchanged for optional verifier diagnostics. For every stage row derive `leaf_topic` only from exact delimiter `" -> "`, compute 4B/0.6B raw-OPD prompt counts, 0.6B R1-completion/full-SFT/supervised-label counts, and retain both `eligible_permutation_position` and `clean_sample_position`.

Write target capture parquet in manifest order with both candidate and held-out rows; write proxy capture parquet with candidates only. Include these top-level columns so `RLHFDataset` preserves them as non-tensors:

```python
capture_row = {
    "data_source": "DeepMath-103K",
    "prompt": raw_opd_messages,
    "ability": "math",
    "reward_model": source_reward_model,
    "extra_info": {
        "index": original_dataset_index,
        "split": "train",
    },
    "opd_verify_stable_id": stable_id,
    "opd_verify_split": split,
    "opd_verify_manifest_index": manifest_index,
}
```

Stage 0 is a view over candidate clean positions `0:24` plus held-out positions `768:776`; Stage 1 uses `0:768` plus `768:1024`; Stage 2 concatenates the immutable Stage-1 blocks with the approved appended blocks and binds the exact Stage-1 report hash. Never replace a post-sampling token failure.

Expose `prepare-root` and `prepare-stage` with required `--output-root`, explicit `--disposable-preflight` for Task-3-only temporary validation, and fail-closed `--expected-source-commit` plus `--reference-repo` for canonical publication. `prepare-stage --stage 2` additionally requires `--parent-report` and hashes/validates the complete Stage-1 report before deriving any row. The disposable flag rejects an output beneath the shared canonical root and marks every emitted manifest non-canonical. Without that flag, both provenance arguments are required, status must be clean, and HEAD must equal the expected commit before computing the source snapshot. Canonical paths are create-once; matching existing bytes validate, while any mismatch fails without replacement.

- [ ] **Step 4: Materialize the pinned small model and run CPU preflight**

Run from the implementation worktree, but write this early smoke to a disposable directory because later implementation commits must not stale a canonical source-bound manifest:

```bash
/home/mchen/miniconda3/envs/verl/bin/huggingface-cli download Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --local-dir models/Qwen3-0.6B

PREFLIGHT_ROOT="$(mktemp -d /tmp/opd_proxy_preflight.XXXXXX)"
trap 'rm -rf "$PREFLIGHT_ROOT"' EXIT

CUDA_VISIBLE_DEVICES="" PYTHONPATH=verl:. \
  /home/mchen/miniconda3/envs/verl/bin/python \
  -m math_eval.prepare_opd_proxy_gradient_verify prepare-root \
  --output-root "$PREFLIGHT_ROOT" --disposable-preflight

CUDA_VISIBLE_DEVICES="" PYTHONPATH=verl:. \
  /home/mchen/miniconda3/envs/verl/bin/python \
  -m math_eval.prepare_opd_proxy_gradient_verify prepare-stage \
  --stage 0 --output-root "$PREFLIGHT_ROOT" --disposable-preflight

PYTHONPATH=verl:. /home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest -q \
  math_eval/test_prepare_opd_proxy_gradient_verify.py
```

Expected: model download resolves the pinned revision; both token-pair checks PASS; disposable Stage-0 manifest contains 24 candidate and 8 held-out rows; pytest PASS. The trap removes the disposable tree, and `data/opd_proxy_gradient_verify` remains absent/unmodified until Task 17 freezes the final reviewed source commit.

- [ ] **Step 5: Commit the stage preflight code, not generated data or model weights**

```bash
git add math_eval/prepare_opd_proxy_gradient_verify.py \
  math_eval/test_prepare_opd_proxy_gradient_verify.py
git commit -m "feat: add Qwen3 OPD proxy stage preflight"
```

### Task 4: Typed Capture Configuration and Native One-Call `n=4` Rollout

**Files:**
- Modify: `verl/verl/workers/config/rollout.py:112-166`
- Modify: `verl/verl/trainer/config/rollout/rollout.yaml`
- Modify: `verl/verl/workers/config/actor.py:70-145`
- Modify: `verl/verl/trainer/config/actor/actor.yaml`
- Modify: `verl/verl/trainer/config/algorithm.py:330-450`
- Modify: `verl/verl/trainer/config/ppo_trainer.yaml`
- Modify: `verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py:225-430`
- Create: `verl/tests/workers/rollout/test_opd_proxy_verify_native_n.py`
- Modify or create: `verl/tests/workers/config/test_rollout_config_on_cpu.py`
- Modify or create: `verl/tests/trainer/config/test_algo_config_on_cpu.py`

**Interfaces:**
- Consumes: normal VERL `DataProto`, vLLM request outputs, and the default rollout path.
- Produces: `RolloutConfig.seed`, `ActorConfig.opd_proxy_verify_capture_only`, `OpdProxyVerifyCaptureConfig`, `_expand_native_n_outputs`, and opt-in `opd_proxy_verify_native_n=4` generation.

- [ ] **Step 1: Write failing config-default and prompt-major native-`n` tests**

```python
@dataclass
class FakeCompletion:
    token_ids: list[int]
    logprobs: list[dict[int, object]]


@dataclass
class FakeRequestOutput:
    outputs: list[FakeCompletion]


def test_native_n_expands_prompt_major_and_assigns_slots():
    expanded = _expand_native_n_outputs(
        stable_ids=np.array(["q0", "q1"], dtype=object),
        request_outputs=[
            FakeRequestOutput([FakeCompletion([10], []), FakeCompletion([11], [])]),
            FakeRequestOutput([FakeCompletion([20], []), FakeCompletion([21], [])]),
        ],
        native_n=2,
    )
    assert expanded.stable_ids.tolist() == ["q0", "q0", "q1", "q1"]
    assert expanded.rollout_slots.tolist() == [0, 1, 0, 1]
    assert expanded.token_ids == [[10], [11], [20], [21]]


def test_capture_config_defaults_are_disabled():
    assert RolloutConfig(name="vllm").seed == 0
    actor = ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
    )
    assert actor.opd_proxy_verify_capture_only is False
    assert OpdProxyVerifyCaptureConfig().enabled is False


def test_capture_config_is_publicly_exported_and_hydra_instantiable():
    from verl.trainer.config import OpdProxyVerifyCaptureConfig as PublicConfig

    assert PublicConfig is OpdProxyVerifyCaptureConfig
    instantiated = hydra.utils.instantiate(
        {
            "_target_": "verl.trainer.config.OpdProxyVerifyCaptureConfig",
            "enabled": False,
        }
    )
    assert isinstance(instantiated, OpdProxyVerifyCaptureConfig)
```

Add a fake-engine test that passes two unrepeated prompts with native `n=4`, asserts exactly one `generate` invocation, asserts `sampling_params.n == 4`, and rejects a second invocation on the same capture engine. Add a control test proving disabled/default generation still repeats nothing inside vLLM and remains byte-for-byte compatible with the pre-change `n=1` result.

- [ ] **Step 2: Run the config/native-rollout tests and verify failure**

Run:

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  verl/tests/workers/rollout/test_opd_proxy_verify_native_n.py \
  verl/tests/workers/config/test_rollout_config_on_cpu.py \
  verl/tests/trainer/config/test_algo_config_on_cpu.py \
  -k 'opd_proxy_verify or native_n or rollout_seed'
```

Expected: FAIL because the typed fields and `_expand_native_n_outputs` do not exist.

- [ ] **Step 3: Add typed defaults and the guarded native-`n` branch**

Add this dataclass and field to `algorithm.py`:

```python
@dataclass
class OpdProxyVerifyCaptureConfig(BaseConfig):
    enabled: bool = False
    output_root: Optional[str] = None
    sample_manifest: Optional[str] = None
    sample_manifest_sha256: Optional[str] = None
    stage: int = 0
    pair: str = "target"
    engine_seed: int = 42
    native_rollouts: int = 4
    expected_questions: int = 0
    chunk_size: int = 16
    schema_version: int = 1


@dataclass
class AlgoConfig(BaseConfig):
    opd_proxy_verify_capture: OpdProxyVerifyCaptureConfig = field(
        default_factory=OpdProxyVerifyCaptureConfig
    )
```

Add `OpdProxyVerifyCaptureConfig` to `algorithm.py.__all__` so `verl.trainer.config.__init__` publicly re-exports it, then add matching YAML with `_target_: verl.trainer.config.OpdProxyVerifyCaptureConfig`. Add `seed: int = 0` to `RolloutConfig`/YAML and `opd_proxy_verify_capture_only: bool = False` to `ActorConfig`/YAML.

In the vLLM rollout, pop `opd_proxy_verify_native_n`. When absent, run the existing code unchanged. When present, require capture enabled and value exactly four, require unrepeated unique prompt keys, clone the sampling parameters with `n=4`, call `inference_engine.generate` once, expand prompt tensors and non-tensors in prompt-major/slot-minor order, and emit `opd_proxy_verify_rollout_slot`. Set a per-engine Boolean after the call and reject a second capture call.

- [ ] **Step 4: Run focused tests plus default rollout regressions**

Run:

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  verl/tests/workers/rollout/test_opd_proxy_verify_native_n.py \
  verl/tests/workers/config/test_rollout_config_on_cpu.py \
  verl/tests/trainer/config/test_algo_config_on_cpu.py
```

Expected: all tests PASS; the control test records one completion per prompt in the default path and four ordered slots only in capture mode.

- [ ] **Step 5: Commit only the typed config and native generation files**

```bash
git add verl/verl/workers/config/rollout.py \
  verl/verl/trainer/config/rollout/rollout.yaml \
  verl/verl/workers/config/actor.py \
  verl/verl/trainer/config/actor/actor.yaml \
  verl/verl/trainer/config/algorithm.py \
  verl/verl/trainer/config/ppo_trainer.yaml \
  verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py \
  verl/tests/workers/rollout/test_opd_proxy_verify_native_n.py \
  verl/tests/workers/config/test_rollout_config_on_cpu.py \
  verl/tests/trainer/config/test_algo_config_on_cpu.py
git commit -m "feat: add native n4 OPD proxy capture mode"
```

### Task 5: Authoritative Capture Tensors, Compound Keys, and Atomic Resume

**Files:**
- Create: `verl/verl/trainer/ppo/opd_proxy_verify_capture.py`
- Create: `verl/tests/trainer/ppo/test_opd_proxy_verify_capture.py`

**Interfaces:**
- Consumes: `DataProto`, registered `get_policy_loss_fn("vanilla")`, existing rollout-correction weights, and shared `(stable_id, seed, slot)` semantics.
- Produces: `validate_capture_contract`, `attach_and_validate_keys`, `build_response_mask`, `compute_authoritative_capture_tensors`, `recursive_parameter_sha256`, `write_tensor_chunks_atomic`, `resolve_capture_resume_prefix`, `load_and_join_capture_chunks`, and `finalize_capture_seed`.

- [ ] **Step 1: Write failing tests that distinguish trainer-old from actor-local-old and enforce exact keys**

```python
def test_authoritative_advantage_uses_stopped_current_not_batch_old(monkeypatch):
    seen = {}

    def fake_vanilla(**kwargs):
        seen.update(kwargs)
        return torch.tensor(2.5), {
            "actor/pg_clipfrac": 0.0,
            "actor/ppo_kl": 0.0,
            "actor/pg_clipfrac_lower": 0.0,
        }

    monkeypatch.setattr(capture, "get_policy_loss_fn", lambda name: fake_vanilla)
    result = compute_authoritative_capture_tensors(
        current_log_prob=torch.tensor([[0.2, 0.4]], requires_grad=True),
        batch_old_log_prob=torch.tensor([[-3.0, -3.0]]),
        ref_log_prob=torch.tensor([[0.5, 0.1]]),
        response_mask=torch.tensor([[1, 1]]),
        rollout_is_weights=torch.tensor([[1.0, 0.5]]),
        actor_config=frozen_actor_config(),
    )
    torch.testing.assert_close(result["local_old_log_prob"], torch.tensor([[0.2, 0.4]]))
    torch.testing.assert_close(result["advantages"], torch.tensor([[0.3, -0.3]]))
    assert not result["advantages"].requires_grad
    assert seen["old_log_prob"].data_ptr() == result["local_old_log_prob"].data_ptr()


def test_eos_mask_includes_first_eos_and_excludes_following_tokens():
    assert build_response_mask(
        responses=torch.tensor([[7, 2, 8, 0]]), eos_token_id=2, pad_token_id=0
    ).tolist() == [[True, True, False, False]]
```

Also test that rollout IS equals detached `min(exp(clamp(batch_old-rollout,-20,20)),5)`, local ratio retains a gradient at one, duplicate/missing/wrong-slot keys fail, chunk sidecars cover the same rows as safetensors, changed source hashes reject resume, and only a contiguous manifest-order prefix resumes.

- [ ] **Step 2: Run the capture-helper tests and verify failure**

Run:

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  verl/tests/trainer/ppo/test_opd_proxy_verify_capture.py
```

Expected: FAIL because `opd_proxy_verify_capture.py` is missing.

- [ ] **Step 3: Implement the capture-only validation and artifact schema**

Implement `compute_authoritative_capture_tensors` with this data flow, while calling the registered production function rather than reimplementing its clipping:

```python
local_old = current_log_prob.detach()
advantages = (ref_log_prob - local_old).detach()
ratio = torch.exp(torch.clamp(current_log_prob - local_old, -20.0, 20.0))
policy_loss_fn = get_policy_loss_fn("vanilla")
policy_loss, policy_metrics = policy_loss_fn(
    old_log_prob=local_old,
    log_prob=current_log_prob,
    advantages=advantages,
    response_mask=response_mask,
    loss_agg_mode="token-mean",
    config=actor_config,
    rollout_is_weights=rollout_is_weights.detach(),
)
```

Return current, local old, stopped advantage, ratio, the three values from `policy_metrics`, and the unscaled token-mean scalar. Keep trainer `batch_old_log_prob` in the artifact but never pass it as `old_log_prob` to the policy loss.

Use three capture subtrees:

```text
rollout/chunk_<start>_<end>.safetensors + .jsonl
trainer_boundary/chunk_<start>_<end>.safetensors + .jsonl
actor/rank_<rank>/chunk_<start>_<end>.safetensors + .jsonl
```

Publish every safetensors/JSONL chunk through its own temporary file, fsync, and atomic rename. The all-at-once rollout result is staged in a temporary subtree and renamed only when its full key coverage validates; it has no prefix-resume mode. Trainer-boundary and actor-rank subtrees expose individually atomic chunks, accept only a contiguous manifest-order prefix, and publish `COMPLETE.json` atomically only after exact global compound-key coverage, parent hashes, and actor-before/after parameter hashes validate.

The seed-level manifest additionally binds resolved vLLM version, complete engine/sampling arguments, original ordered prompt-key hash, returned ordered compound-key hash, model/tokenizer/config/source hashes, and counters proving one engine plus one generation call. A process restart may consume an already complete matching rollout subtree but may not issue a second call for the same declared work unit.

- [ ] **Step 4: Run helper tests and rollout-correction regressions**

Run:

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  verl/tests/trainer/ppo/test_opd_proxy_verify_capture.py \
  verl/tests/trainer/ppo/test_rollout_corr.py \
  verl/tests/trainer/ppo/test_rollout_corr_integration.py
```

Expected: all tests PASS and the actor-local-old test proves `batch_old_log_prob` cannot affect the stopped advantage.

- [ ] **Step 5: Commit the standalone capture contract**

```bash
git add verl/verl/trainer/ppo/opd_proxy_verify_capture.py \
  verl/tests/trainer/ppo/test_opd_proxy_verify_capture.py
git commit -m "feat: add authoritative OPD proxy capture artifacts"
```

### Task 6: Optimizer-Free Trainer, Worker, and Actor Capture Integration

**Files:**
- Modify: `verl/verl/trainer/ppo/ray_trainer.py:690-1840`
- Modify: `verl/verl/workers/fsdp_workers.py:281-965`
- Modify: `verl/verl/workers/actor/dp_actor.py:288-1185`
- Modify: `verl/verl/trainer/main_ppo.py`
- Create: `verl/tests/trainer/ppo/test_opd_proxy_verify_capture_launcher.py`

**Interfaces:**
- Consumes: Task 4 native rollout, Task 5 artifact helpers, existing `compute_rollout_correction_and_add_to_batch`, and manifest columns `opd_verify_stable_id`, `opd_verify_split`, `opd_verify_manifest_index`.
- Produces: `RayPPOTrainer._fit_opd_proxy_verify_capture`, worker RPC `capture_opd_proxy_verify`, and actor method `DataParallelPPOActor.capture_opd_proxy_verify`.

- [ ] **Step 1: Write a failing mocked call-order/no-optimizer/no-update integration test**

```python
def test_capture_path_calls_one_generation_and_never_updates(monkeypatch, capture_trainer):
    calls = []
    capture_trainer.actor_rollout_wg.generate_sequences = recorder(calls, "generate", generated_batch())
    capture_trainer.actor_rollout_wg.compute_log_prob = recorder(calls, "batch_old", old_batch())
    capture_trainer.ref_policy_wg.compute_ref_log_prob = recorder(calls, "ref", ref_batch())
    capture_trainer.actor_rollout_wg.capture_opd_proxy_verify = recorder(calls, "actor_capture", DataProto())
    capture_trainer.actor_rollout_wg.update_actor = lambda batch: pytest.fail("update_actor called")
    capture_trainer.fit()
    assert calls == ["generate", "batch_old", "ref", "actor_capture"]
    assert capture_trainer.actor_rollout_wg.generate_sequences.call_count == 1


def test_capture_actor_resolves_optimizer_config_to_none():
    actor_config = OmegaConf.create(
        {
            "opd_proxy_verify_capture_only": True,
            "optim": {"lr": 1e-6},
        }
    )
    assert _resolve_actor_optim_config(actor_config) is None


def test_normal_actor_keeps_optimizer_config():
    actor_config = OmegaConf.create(
        {
            "opd_proxy_verify_capture_only": False,
            "optim": {"lr": 1e-6},
        }
    )
    assert _resolve_actor_optim_config(actor_config) is actor_config.optim
```

Add tests that stable IDs survive native expansion, `_balance_batch` reorder, `DataProto.select`, and rank dispatch; actor capture sees one mini-batch/epoch and micro-batch one; forbidden feature keys cause failure; parameter hashes match; and the disabled path still enters the original `fit()` body.

- [ ] **Step 2: Run the launcher integration test and verify failure**

Run:

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  verl/tests/trainer/ppo/test_opd_proxy_verify_capture_launcher.py \
  verl/tests/trainer/ppo/test_opd_proxy_verify_capture.py
```

Expected: FAIL because `_fit_opd_proxy_verify_capture` and the worker/actor RPC do not exist.

- [ ] **Step 3: Add one isolated capture entry point and leave normal training unchanged**

At the first line of `fit()`, dispatch only when the typed flag is enabled:

```python
def fit(self):
    if self.config.algorithm.opd_proxy_verify_capture.enabled:
        return self._fit_opd_proxy_verify_capture()
    # The existing normal training body remains below this guard unchanged.
```

Do not extract or re-indent the existing training loop. In `_fit_opd_proxy_verify_capture`:

1. Load exactly one ordered capture batch and set `uid` from `opd_verify_stable_id`.
2. Do not call trainer-side `repeat(n=4)` before generation.
3. Call native generation once with `opd_proxy_verify_native_n=4`.
4. Repeat the original batch prompt-major after generation and validate returned slots/keys.
5. Run the existing actor old-log-prob recomputation, teacher ref-log-prob computation, and rollout-correction helper.
6. Write the trainer-boundary artifact, call `capture_opd_proxy_verify`, finalize coverage, and return before reward, advantage, update, validation, checkpoint, or logging loops.

Do not use `extra_info.index` as identity. Extend capture-only actor selection to retain non-tensor keys `opd_verify_stable_id`, `opd_verify_split`, and `opd_verify_manifest_index`, plus tensor `opd_proxy_verify_rollout_slot`; attach the engine seed from the typed work-unit config and validate the compound key before and after every repeat/reorder/dispatch boundary.

Add pure helper `_resolve_actor_optim_config(actor_config)` in `fsdp_workers.py` and use its return value as `_build_model_optimizer(..., optim_config=...)`. When `ActorConfig.opd_proxy_verify_capture_only` is true it returns `None`, so no optimizer or scheduler is constructed. Actor capture must set `actor_module.train()`, use `_forward_micro_batch`, call Task 5's authoritative helper per micro-batch, write rank-local chunks, and never call `backward`, `_optimizer_step`, or `zero_grad`.

- [ ] **Step 4: Run integration and normal-path regression tests**

Run:

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  verl/tests/trainer/ppo/test_opd_proxy_verify_capture.py \
  verl/tests/trainer/ppo/test_opd_proxy_verify_capture_launcher.py \
  verl/tests/workers/actor/test_length_aware_opd.py \
  verl/tests/trainer/ppo/test_rollout_corr_integration.py
```

Expected: all tests PASS; the mocked capture path has the exact four-call order and the normal path produces its pre-change calls.

- [ ] **Step 5: Commit the default-off integration**

```bash
git add verl/verl/trainer/ppo/ray_trainer.py \
  verl/verl/workers/fsdp_workers.py \
  verl/verl/workers/actor/dp_actor.py \
  verl/verl/trainer/main_ppo.py \
  verl/tests/trainer/ppo/test_opd_proxy_verify_capture_launcher.py
git commit -m "feat: integrate optimizer-free OPD proxy capture"
```

### Task 7: Strict Prismatic CUDA Projection Wrapper

**Files:**
- Create: `math_eval/opd_proxy_gradient_projection.py`
- Test: `math_eval/test_opd_proxy_gradient_projection.py`
- Reuse: `math_eval/collect_prismatic_gradients.py`

**Interfaces:**
- Consumes: pinned reference checkout and a full flattened native gradient.
- Produces: `ProjectionConfig`, `verify_prismatic_reference`, `construct_cuda_projector`, `project_full_gradient`, and `projected_mean_is_mean_projection`.

- [ ] **Step 1: Write failing constructor, scale, dtype, no-fallback, and linearity tests**

```python
def test_strict_projector_constructor_and_output_scale():
    fake = FakeCudaProjector(output=torch.full((1, 1024), 32.0))
    projector = construct_cuda_projector(4096, "cuda:0", ProjectionConfig(), cuda_projector_class=fake.factory)
    result = project_full_gradient(projector, torch.ones(4096, dtype=torch.float32), ProjectionConfig())
    assert fake.kwargs == {
        "grad_dim": 4096,
        "proj_dim": 1024,
        "seed": 0,
        "proj_type": ProjectionType.rademacher,
        "device": "cuda:0",
        "dtype": torch.float16,
        "block_size": 128,
        "max_batch_size": 16,
    }
    assert fake.seen_input_dtype == torch.float16
    assert fake.seen_model_id == 0
    torch.testing.assert_close(result, torch.ones((1, 1024)))


def test_missing_cuda_projector_never_falls_back(monkeypatch):
    monkeypatch.setattr(projection, "load_cuda_projector", lambda path: None)
    with pytest.raises(RuntimeError, match="CudaProjector is required"):
        construct_cuda_projector(8, "cuda:0", ProjectionConfig())
```

Test `P((g1+g2)/2)` against `(P(g1)+P(g2))/2`, float32 full-norm measurement before the float16 cast, exact commit/tree rejection, parameter-order hashes, and non-finite/zero output rejection.

- [ ] **Step 2: Run projection tests and verify the missing module failure**

Run:

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest -q \
  math_eval/test_opd_proxy_gradient_projection.py
```

Expected: FAIL because the strict wrapper is missing.

- [ ] **Step 3: Implement the only permitted projector path**

```python
@dataclass(frozen=True)
class ProjectionConfig:
    dimension: int = 1024
    seed: int = 0
    block_size: int = 128
    max_batch_size: int = 16
    model_id: int = 0
    input_dtype: torch.dtype = torch.float16


def project_full_gradient(projector, flat_gradient: torch.Tensor, config: ProjectionConfig) -> torch.Tensor:
    if flat_gradient.dtype != torch.float32 or flat_gradient.ndim != 1:
        raise ValueError("native flat gradient must be one-dimensional float32")
    if not torch.isfinite(flat_gradient).all() or torch.linalg.vector_norm(flat_gradient) == 0:
        raise ValueError("native flat gradient must be finite and nonzero")
    projected = projector.project(
        flat_gradient.to(config.input_dtype).unsqueeze(0), model_id=config.model_id
    ).float() / math.sqrt(config.dimension)
    if projected.shape != (1, config.dimension) or not torch.isfinite(projected).all():
        raise ValueError("invalid projected gradient")
    return projected
```

Load only the official `CudaProjector` and `ProjectionType.rademacher` from the verified checkout. Record every constructor field, input/output dtype, parameter-layout hash, commit, tree, and source snapshot in the vector manifest.

- [ ] **Step 4: Run projection and existing official-collector regressions**

Run:

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest -q \
  math_eval/test_opd_proxy_gradient_projection.py \
  math_eval/test_collect_prismatic_gradients.py
```

Expected: all tests PASS and the fake projector proves the output division is exactly 32.

- [ ] **Step 5: Commit the strict projection layer**

```bash
git add math_eval/opd_proxy_gradient_projection.py \
  math_eval/test_opd_proxy_gradient_projection.py
git commit -m "feat: add strict OPD gradient projection"
```

### Task 8: Standalone Exact OPD Replay and Vector Collection

**Files:**
- Create: `math_eval/replay_opd_proxy_gradients.py`
- Test: `math_eval/test_replay_opd_proxy_gradients.py`
- Modify: `math_eval/opd_proxy_gradient_verify_artifacts.py`
- Modify: `math_eval/test_opd_proxy_gradient_verify_artifacts.py`

**Interfaces:**
- Consumes: completed capture artifacts, full unsharded student checkpoint, Task 5 authoritative-loss helper, and Task 7 projector.
- Produces: `ParameterLayoutEntry`, `ParameterLayout`, `build_parameter_layout`, `load_replay_actor`, `compute_replay_loss`, `replay_target_group`, `replay_proxy_trajectory`, `run_replay_shard`, and `validate_replay_coverage`.

- [ ] **Step 1: Write failing toy-model tests for exact stop-gradients, group averaging, and vector schema**

```python
def test_target_group_is_mean_of_four_canonical_trajectory_gradients(toy_actor, trajectories):
    direct = []
    for trajectory in trajectories:
        toy_actor.zero_grad(set_to_none=True)
        compute_replay_loss(toy_actor, trajectory).loss.backward()
        direct.append(flatten_float32_gradients(toy_actor))
    expected = torch.stack(direct).mean(dim=0)
    actual = accumulate_target_group_full_gradient(toy_actor, trajectories)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)


def test_replay_uses_actor_local_old_not_captured_batch_old(toy_actor, trajectory):
    first = compute_replay_loss(toy_actor, replace(trajectory, batch_old_log_prob=-100.0))
    second = compute_replay_loss(toy_actor, replace(trajectory, batch_old_log_prob=100.0))
    torch.testing.assert_close(first.loss, second.loss)
    torch.testing.assert_close(first.advantages, second.advantages)


def test_proxy_vector_ids_encode_seed_slot_and_aggregation():
    assert proxy_vector_id("q7", 43, rollout_slot=2) == "P:q7:seed=43:slot=2:n1"
    assert proxy_group_vector_id("q7", 43) == "P:q7:seed=43:n4"
    assert target_group_vector_id("q7", 42) == "T:q7:seed=42:n4"


def test_proxy_group_norm_and_projection_use_mean_full_gradient(toy_actor, trajectories, projector):
    individual_full = []
    individual_projected = []
    for trajectory in trajectories:
        toy_actor.zero_grad(set_to_none=True)
        compute_replay_loss(toy_actor, trajectory).loss.backward()
        full = flatten_float32_gradients(toy_actor)
        individual_full.append(full)
        individual_projected.append(projector.project(full))
    expected_full = torch.stack(individual_full).mean(dim=0)
    actual = replay_proxy_group(toy_actor, trajectories, projector)
    torch.testing.assert_close(actual.projected_gradient, projector.project(expected_full))
    torch.testing.assert_close(
        actual.projected_gradient,
        torch.stack(individual_projected).mean(dim=0),
    )
    torch.testing.assert_close(actual.full_gradient_norm, expected_full.norm())
```

Add tests for registration-order `named_parameters(remove_duplicate=True)`, float32 accumulation before projection, zeroing between proxy trajectories, per-trajectory token means before target averaging, group scalar formulas, exact sidecar/tensor coverage, shard resume, and rejection of an FSDP local shard as a full gradient.

- [ ] **Step 2: Run replay tests and verify failure**

Run:

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest -q \
  math_eval/test_replay_opd_proxy_gradients.py \
  math_eval/test_opd_proxy_gradient_verify_artifacts.py
```

Expected: FAIL because the replay module and parameter-layout records do not exist.

- [ ] **Step 3: Implement full-model replay and strict vector chunks**

Load each replay student as a complete Hugging Face model on one GPU in the production actor's train mode and BF16 autocast context. Use `DataParallelPPOActor._forward_micro_batch` to recompute action-token log-probabilities, check captured actor values on valid tokens, and call the registered vanilla loss through Task 5.

Define the parameter layout from registration order:

```python
@dataclass(frozen=True)
class ParameterLayoutEntry:
    name: str
    shape: tuple[int, ...]
    numel: int
    offset: int


@dataclass(frozen=True)
class ParameterLayout:
    entries: tuple[ParameterLayoutEntry, ...]
    total_numel: int
    sha256: str


def build_parameter_layout(model: torch.nn.Module) -> ParameterLayout:
    entries = []
    offset = 0
    for name, parameter in model.named_parameters(remove_duplicate=True):
        if not parameter.requires_grad:
            continue
        entries.append(ParameterLayoutEntry(name, tuple(parameter.shape), parameter.numel(), offset))
        offset += parameter.numel()
    return ParameterLayout(tuple(entries), offset, sha256_parameter_layout(entries))
```

First run the actor recomputation in eval/no-grad mode to verify captured `batch_old_log_prob`. Then run the gradient-bearing actor recomputation in production train mode under BF16 autocast to obtain `current_log_prob`; reconstruct actor-local old as `current_log_prob.detach()`. Treat captured `rollout_log_prob` and `ref_log_prob` as immutable byte/hash-checked inputs rather than claiming to regenerate vLLM or teacher values.

For target groups, perform four separate `(loss / 4).backward()` calls without optimizer scaling, accumulate/gather every full parameter gradient in float32, then project once. For each proxy `(stable_id, engine_seed)` group, process its four slots together: zero gradients before each slot, run one unscaled trajectory backward, compute that slot's exact full-gradient norm, project and store its unnormalized float32 `P_n1`, and add every named full-parameter gradient divided by four to a float32 group accumulator. Only after all four slots are present, compute the exact `P_n4` full-gradient norm from that accumulator and project the accumulator once. Assert that this directly projected `P_n4` equals the arithmetic mean of the four stored unnormalized `P_n1` projections within the pinned numeric tolerance; never infer the full-gradient norm from a projected vector. Release the group accumulator before advancing to the next question. Derive target group diagnostics exactly as:

```python
group_sampled_reverse_kl = np.mean(trajectory_sampled_reverse_kl)
group_opd_signal_rms = np.sqrt(np.mean(np.square(trajectory_opd_signal_rms)))
group_valid_token_count = np.mean(trajectory_valid_token_count)
group_response_length = np.mean(trajectory_response_length)
```

Each trajectory's sampled reverse KL is `mean(local_old_log_prob - ref_log_prob)` over its valid response mask, and its OPD-signal RMS is `sqrt(mean((ref_log_prob - local_old_log_prob) ** 2))` over the same mask.

Write `vectors_<start>_<end>.safetensors` with keys:

```text
projected_gradient, full_gradient_norm, projected_gradient_norm,
valid_token_count, response_length, sampled_reverse_kl, opd_signal_rms
```

When verifier results exist, add `verifier_correct_count` and `verifier_total`; otherwise omit both tensors, record `{"status": "not_computed"}` in the manifest, and make the loader return `None` for both fields. The JSONL sidecar records `vector_id`, stable ID, split, representation, seed, slot, aggregation, source-capture hash, and tensor row.

- [ ] **Step 4: Run replay, artifact, and projection tests**

Run:

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest -q \
  math_eval/test_replay_opd_proxy_gradients.py \
  math_eval/test_opd_proxy_gradient_verify_artifacts.py \
  math_eval/test_opd_proxy_gradient_projection.py \
  -m 'not gpu'
```

Expected: all tests PASS; toy target accumulation matches the explicit four-gradient mean and all vector records have exact IDs.

- [ ] **Step 5: Commit replay and vector schema**

```bash
git add math_eval/replay_opd_proxy_gradients.py \
  math_eval/test_replay_opd_proxy_gradients.py \
  math_eval/opd_proxy_gradient_verify_artifacts.py \
  math_eval/test_opd_proxy_gradient_verify_artifacts.py
git commit -m "feat: replay exact OPD proxy gradients"
```

### Task 9: Direct Production-Backward Smoke Fixture

**Files:**
- Modify: `verl/verl/workers/actor/dp_actor.py:941-1185`
- Modify: `verl/verl/trainer/ppo/opd_proxy_verify_capture.py`
- Modify: `verl/tests/trainer/ppo/test_opd_proxy_verify_capture.py`
- Create: `verl/examples/fire_opd/run_direct_opd_proxy_gradient_fixture.sh`

**Interfaces:**
- Consumes: one frozen Stage-0 capture trajectory and the same parameter layout/projector convention as replay.
- Produces: smoke-only `capture_direct_opd_proxy_gradient_fixture` and a projected direct-backward artifact; it never steps an optimizer.

- [ ] **Step 1: Write a failing test that proves real backward runs but optimizer step cannot run**

```python
def test_direct_fixture_projects_registered_loss_without_optimizer_step(monkeypatch, actor, trajectory):
    backward_calls = []
    original_backward = torch.Tensor.backward

    def backward_spy(tensor, *args, **kwargs):
        backward_calls.append(tensor.detach())
        return original_backward(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "backward", backward_spy)
    monkeypatch.setattr(actor, "_optimizer_step", lambda: pytest.fail("optimizer step called"))
    artifact = actor.capture_direct_opd_proxy_gradient_fixture(trajectory)
    assert len(backward_calls) == 1
    assert artifact["loss_mode"] == "vanilla"
    assert artifact["optimizer_step_called"] is False
```

Add a test that the hook is unavailable unless both capture mode and the explicit smoke fixture flag are true, parameter hashes are unchanged after gradients are cleared, and the stored scalar uses the same unscaled token-mean convention as standalone replay.

- [ ] **Step 2: Run the fixture test and verify failure**

Run:

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  verl/tests/trainer/ppo/test_opd_proxy_verify_capture.py \
  -k direct_fixture
```

Expected: FAIL because `capture_direct_opd_proxy_gradient_fixture` is missing.

- [ ] **Step 3: Implement the isolated one-trajectory production backward hook**

Reuse the exact forward, actor-local old overwrite, reverse advantage, registered loss, and backward statements from `update_policy`. Stop immediately before `_optimizer_step`, gather/project the full gradient under the canonical layout, clear gradients, compare actor hashes, publish the artifact, and return. Do not make the ordinary capture path call backward.

The launcher must require these arguments, reject more than one trajectory, and invoke the production hook with `/home/mchen/miniconda3/envs/gvendi-opd/bin/python` so both VERL and the pinned `traker`/`fast_jl` projector dependencies are importable:

```bash
--stage 0 --pair proxy --engine-seed 42 --rollout-slot 0 \
--stable-id "${PINNED_SMOKE_ID}" --capture-direct-gradient-fixture true
```

- [ ] **Step 4: Run capture regressions and shell syntax validation**

Run:

```bash
bash -n verl/examples/fire_opd/run_direct_opd_proxy_gradient_fixture.sh
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  verl/tests/trainer/ppo/test_opd_proxy_verify_capture.py \
  verl/tests/trainer/ppo/test_opd_proxy_verify_capture_launcher.py
```

Expected: shell syntax PASS and all tests PASS without loading a real model.

- [ ] **Step 5: Commit the smoke-only backward fixture**

```bash
git add verl/verl/workers/actor/dp_actor.py \
  verl/verl/trainer/ppo/opd_proxy_verify_capture.py \
  verl/tests/trainer/ppo/test_opd_proxy_verify_capture.py \
  verl/examples/fire_opd/run_direct_opd_proxy_gradient_fixture.sh
git commit -m "test: add direct OPD gradient smoke fixture"
```

### Task 10: Same-Model Completion-Only SFT-Gradient Baseline

**Files:**
- Create: `math_eval/collect_opd_proxy_sft_gradients.py`
- Test: `math_eval/test_collect_opd_proxy_sft_gradients.py`
- Reuse: `math_eval/collect_prismatic_gradients.py`

**Interfaces:**
- Consumes: candidate rows, `models/Qwen3-0.6B`, R1 completion, official `GradientComputer`, and Task 7 projection.
- Produces: `build_sft_example`, `construct_official_sft_collector`, `collect_sft_shard`, and strict `S:<stable_id>` vector artifacts.

- [ ] **Step 1: Write failing official-method, completion-mask, context, and resume tests**

```python
def test_sft_collector_calls_official_completion_only_methods(fake_gradient_computer, sample_row):
    collector = construct_official_sft_collector(fake_gradient_computer, pinned_reference())
    record = collector.collect_one(sample_row)
    assert fake_gradient_computer.calls == [
        "prepare_model_input",
        "obtain_gradient",
        "project_gradients",
    ]
    assert record.vector_id == f"S:{sample_row['stable_id']}"
    assert record.supervised_label_count > 0


def test_sft_example_uses_original_question_and_r1_solution(sample_row):
    example = build_sft_example(sample_row)
    assert example["messages"] == [
        {"role": "user", "content": sample_row["prompt"]},
        {"role": "assistant", "content": sample_row["completion"]},
    ]
```

Test rejection of prompt-supervised labels, zero completion labels, changed official commit/tree, an existing Qwen2.5 vector artifact, wrong candidate order, non-contiguous resume, and Stage-2 collection that recomputes old Stage-1 rows.

- [ ] **Step 2: Run baseline tests and verify failure**

Run:

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest -q \
  math_eval/test_collect_opd_proxy_sft_gradients.py
```

Expected: FAIL because the Qwen3 SFT collector is missing.

- [ ] **Step 3: Implement the official collector adapter and common vector interface**

Verify the pinned reference before importing it. Delegate formatting/masking and gradient computation to official `GradientComputer.prepare_model_input`, `obtain_gradient`, and `project_gradients`; do not write a local CE loss. Validate that all non-ignored labels lie in the assistant completion and at least one remains.

Use Task 1 chunks/sidecars and require the official collector's projector manifest to equal Task 7's pinned constructor fields and output scaling. Record full/projected norms, supervised tokens, prompt/completion lengths, model/tokenizer recursive hashes, official source snapshot, and vector IDs in candidate manifest order. Stage 2 computes only its 768 appended candidates and validates the union with Stage 1.

- [ ] **Step 4: Run SFT and official collector regressions**

Run:

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest -q \
  math_eval/test_collect_opd_proxy_sft_gradients.py \
  math_eval/test_collect_prismatic_gradients.py
```

Expected: all tests PASS and the fake official call order matches exactly.

- [ ] **Step 5: Commit the SFT baseline**

```bash
git add math_eval/collect_opd_proxy_sft_gradients.py \
  math_eval/test_collect_opd_proxy_sft_gradients.py
git commit -m "feat: add Qwen3 SFT-gradient baseline"
```

### Task 11: Raw-Prompt Embedding Baseline

**Files:**
- Create: `math_eval/collect_opd_proxy_prompt_embeddings.py`
- Test: `math_eval/test_collect_opd_proxy_prompt_embeddings.py`

**Interfaces:**
- Consumes: candidate rows and the pinned Qwen3-0.6B model/tokenizer.
- Produces: `format_embedding_prompt`, `mean_pool_last_hidden`, `collect_embedding_shard`, and strict `E:<stable_id>` vector artifacts.

- [ ] **Step 1: Write failing chat-template, pooling, normalization, and resume tests**

```python
def test_embedding_prompt_uses_generation_prefix_and_disables_thinking(fake_tokenizer):
    format_embedding_prompt(fake_tokenizer, "raw question")
    assert fake_tokenizer.last_messages == [{"role": "user", "content": "raw question"}]
    assert fake_tokenizer.last_kwargs == {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": False,
        "return_tensors": "pt",
    }


def test_mean_pool_ignores_padding_and_l2_normalizes():
    hidden = torch.tensor([[[3.0, 0.0], [0.0, 4.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    pooled = mean_pool_last_hidden(hidden, mask)
    expected = torch.tensor([[1.5, 2.0]])
    expected = expected / torch.linalg.vector_norm(expected, dim=1, keepdim=True)
    torch.testing.assert_close(pooled, expected)
```

Test exact candidate ordering, non-finite/zero embeddings, assistant-prefix inclusion, no response/teacher/backward invocation, Stage-2 append-only behavior, and atomic resume.

- [ ] **Step 2: Run embedding tests and verify failure**

Run:

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest -q \
  math_eval/test_collect_opd_proxy_prompt_embeddings.py
```

Expected: FAIL because the embedding collector is missing.

- [ ] **Step 3: Implement deterministic final-layer mean pooling**

Load the pinned model in eval/no-grad mode, format the single raw OPD user message exactly as tested, obtain the final hidden state, mean over all non-padding tokens including the assistant-generation prefix, L2-normalize in float32, and publish Task 1 vector artifacts with IDs `E:<stable_id>`. Record chat-template hash, kwargs, token counts, model/tokenizer hashes, source snapshot, and append-only Stage-2 parentage.

- [ ] **Step 4: Run embedding and preparation tests**

Run:

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest -q \
  math_eval/test_collect_opd_proxy_prompt_embeddings.py \
  math_eval/test_prepare_opd_proxy_gradient_verify.py \
  -k 'embedding or preflight or chat_template'
```

Expected: all tests PASS with no CUDA requirement.

- [ ] **Step 5: Commit the embedding baseline**

```bash
git add math_eval/collect_opd_proxy_prompt_embeddings.py \
  math_eval/test_collect_opd_proxy_prompt_embeddings.py
git commit -m "feat: add OPD prompt-embedding baseline"
```

### Task 12: Unified Official Selector and Frozen Random Schedules

**Files:**
- Create: `math_eval/select_opd_proxy_gradient_verify.py`
- Test: `math_eval/test_select_opd_proxy_gradient_verify.py`
- Create: `pytest.ini`
- Reuse: `math_eval/select_gradient_diverse_deepmath.py`
- Reuse: `math_eval/deepmath_gradient_diversity.py`

**Interfaces:**
- Consumes: aligned candidate `VectorSet` objects and stage metadata.
- Produces: `RandomSchedules`, `SelectionBundle`, `build_length_quartiles`, `largest_remainder_allocation`, `generate_random_schedules`, `run_selection`, `load_random_schedules`, and `load_selection_bundle`.

- [ ] **Step 1: Write failing K-derivation, seed, balanced-round-robin, and random-schedule tests**

```python
def test_stage1_selector_uses_both_kmeans_seeds_and_fixed_round_robin_seed(fake_cluster):
    bundle = run_selection(synthetic_stage1_vectors(), fake_cluster_manager=fake_cluster)
    assert bundle.k_values == (76, 7)
    assert bundle.kmeans_seeds == (42, 43)
    assert bundle.round_robin_seed == 42
    assert all(len(selected) == 172 for selected in bundle.selected_positions.values())


def test_random_streams_are_spawned_once_and_stored_as_sorted_int32(stage1_rows):
    schedules = generate_random_schedules(stage1_rows, selected_size=172, draws=10_000)
    assert schedules.uniform.dtype == np.dtype("<i4")
    assert schedules.stratified.dtype == np.dtype("<i4")
    assert schedules.uniform.flags.c_contiguous
    assert schedules.uniform.shape == (10_000, 172)
    assert np.all(np.diff(schedules.uniform, axis=1) > 0)
    assert schedules.manifest["seed_sequence"] == 2026071402
    assert schedules.manifest["spawn_count"] == 2
    assert sha256_int_rows(np.array([[0, 2], [1, 3]], dtype="<i4")) == (
        "74201e550190c3ead9a6c11a336e2d25ed25cb5d36b30166eba198b0596104c2"
    )
```

Add tests for rank-based length quartiles, lexicographic largest-remainder ties, capacity redistribution, independent child RNGs, `shuffle=False`, paired schedules across target seeds, official input permutation seeds 42/43, label inverse permutation, selected sampler order preservation, wrong-ID/provenance rejection, and Stage 0/2 derived cardinalities. Mark the real official K-means integration test with `@pytest.mark.gpu`, and register `gpu = requires an authorized CUDA allocation` in the new root `pytest.ini` so CPU deselection is warning-free.

- [ ] **Step 2: Run selector tests and verify failure**

Run:

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest -q \
  math_eval/test_select_opd_proxy_gradient_verify.py
```

Expected: FAIL because the stage-aware selector does not exist.

- [ ] **Step 3: Implement random schedules and wrap the existing official selector primitives**

Generate random schedules with:

```python
children = np.random.SeedSequence(2026071402).spawn(2)
uniform_rng = np.random.Generator(np.random.PCG64(children[0]))
stratified_rng = np.random.Generator(np.random.PCG64(children[1]))
uniform_positions = np.sort(
    uniform_rng.choice(candidate_count, size=selected_size, replace=False, shuffle=False)
)
```

Store `random_uniform.npy` and `random_stratified.npy` as little-endian C-contiguous int32 arrays plus both file SHA and a logical row hash where each row is compact JSON followed by newline. The analyzer must load these files; it must never regenerate them.

Build each stratification key only from `(leaf_topic, qwen3_0_6b_prompt_token_length_quartile)`. Assign quartiles by sorting `(prompt_token_count_0_6b, stable_id)` and using `floor(4 * rank / candidate_count)`. Traverse stratum keys lexicographically, sample stable-ID-sorted member positions with the dedicated child generator and `shuffle=False`, redistribute capacity deficits by the same largest-remainder rule, union positions, and sort globally.

For every one of eight `P_n1`, two `P_n4`, S, E, `T_42`, and `T_43`, L2-normalize rows, call existing `cluster_official` separately for primary/diagnostic K and seeds 42/43, then call `balanced_round_robin(labels, target_size=layout.selected_size, seed=42)`. Preserve both seeds and never choose the better output.

The real official K-means subprocess must see exactly one allocation-owned `CUDA_VISIBLE_DEVICES` token. It may run sequential representations on that one logical GPU; CPU balanced round robin and metrics must not inherit a CUDA requirement.

- [ ] **Step 4: Run selector and legacy selection regressions**

Run:

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest -q \
  math_eval/test_select_opd_proxy_gradient_verify.py \
  math_eval/test_select_gradient_diverse_deepmath.py \
  math_eval/test_deepmath_gradient_diversity.py \
  -m 'not gpu'
```

Expected: all CPU/fake-cluster tests PASS; GPU-marked official K-means tests remain deselected outside `opd-CLI`.

- [ ] **Step 5: Commit the unified selector**

```bash
git add math_eval/select_opd_proxy_gradient_verify.py \
  math_eval/test_select_opd_proxy_gradient_verify.py \
  pytest.ini
git commit -m "feat: add OPD proxy selector and random nulls"
```

### Task 13: Pure Target-Space Metrics and Agreement Diagnostics

**Files:**
- Create: `math_eval/opd_proxy_gradient_statistics.py`
- Test: `math_eval/test_opd_proxy_gradient_statistics.py`

**Interfaces:**
- Consumes: normalized candidate/held-out vectors, selected position arrays, and stored random schedules.
- Produces: `TargetSpace`, `SubsetScores`, `normalize_rows`, `g_vendi_from_gram`, `target_g_vendi`, `facility_coverage`, `evaluate_subset`, `evaluate_subset_table`, `inclusive_percentile`, `unbiased_hsic`, `debiased_linear_cka`, `build_permutation_schedule`, `target_dependence_permutation_test`, `partition_agreement`, and `selected_set_overlap`.

- [ ] **Step 1: Write failing fixed-value G-Vendi, coverage, percentile, CKA, and overlap tests**

```python
def test_g_vendi_fixed_gram_and_eigenvalue_policy():
    gram = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float64)
    assert g_vendi_from_gram(gram) == pytest.approx(1.7547653506033232)
    assert target_g_vendi(np.eye(2)) == pytest.approx(2.0)
    assert target_g_vendi(np.ones((2, 1))) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="negative Gram eigenvalue"):
        g_vendi_from_gram(np.array([[1.0, 2.0], [2.0, 1.0]]))


def test_inclusive_percentile_counts_ties():
    assert inclusive_percentile(np.array([1.0, 2.0, 2.0, 4.0]), 2.0) == 0.75


def test_selected_overlap_fixture():
    result = selected_set_overlap([0, 1, 2, 3], [2, 3, 4, 5], candidate_count=10, selected_size=4)
    assert result["jaccard"] == pytest.approx(1 / 3)
    assert result["chance_adjusted_overlap"] == pytest.approx(1 / 6)


def test_cka_permutation_schedule_and_hash_are_pinned():
    rows = build_permutation_schedule(n=5, draws=4, seed=2026071403)
    assert rows.tolist() == [
        [0, 2, 1, 4, 3],
        [1, 2, 0, 3, 4],
        [2, 4, 3, 0, 1],
        [3, 0, 4, 2, 1],
    ]
    assert sha256_int_rows(rows) == (
        "c4c02ab70a6edef9c1050d1ee92bbe05d4b541929687ce3e386677d2767ef0ee"
    )
```

Add the official-reference Vendi smoke fixture `1.889881574842311`, coverage fixtures `0.5` and `1.0`, `1e-7` negative-eigenvalue clamping, scalar subset means, sorted/unique position validation, fixed unbiased CKA values `0.998739479819789` and `-0.446182589285757`, plus-one/tie permutation p-values, non-positive self-HSIC failure, AMI/ARI fixtures, and different left/right feature dimensions.

- [ ] **Step 2: Run statistics tests and verify failure**

Run:

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest -q \
  math_eval/test_opd_proxy_gradient_statistics.py
```

Expected: FAIL because the statistics module is missing.

- [ ] **Step 3: Implement float64 target metrics and efficient fixed-schedule CKA**

Implement G-Vendi as:

```python
def g_vendi_from_gram(gram: np.ndarray) -> float:
    symmetric = 0.5 * (np.asarray(gram, dtype=np.float64) + np.asarray(gram, dtype=np.float64).T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    if eigenvalues.min(initial=0.0) < -1e-7:
        raise ValueError("negative Gram eigenvalue exceeds tolerance")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    if not np.isfinite(eigenvalues).all() or eigenvalues.sum() <= 0:
        raise ValueError("Gram spectrum must be finite with positive trace")
    probabilities = eigenvalues / eigenvalues.sum()
    positive = probabilities > 0
    return float(np.exp(-np.sum(probabilities[positive] * np.log(probabilities[positive]))))


def inclusive_percentile(null_scores: np.ndarray, observed: float) -> float:
    null = np.asarray(null_scores, dtype=np.float64)
    if null.ndim != 1 or not np.isfinite(null).all() or not np.isfinite(observed):
        raise ValueError("percentile inputs must be finite")
    return float(np.count_nonzero(null <= observed) / null.size)


def unbiased_hsic(left_kernel: np.ndarray, right_kernel: np.ndarray) -> float:
    left = np.array(left_kernel, dtype=np.float64, copy=True)
    right = np.array(right_kernel, dtype=np.float64, copy=True)
    if left.shape != right.shape or left.ndim != 2 or left.shape[0] < 4:
        raise ValueError("unbiased HSIC requires aligned square kernels with n >= 4")
    np.fill_diagonal(left, 0.0)
    np.fill_diagonal(right, 0.0)
    n = left.shape[0]
    trace_product = float(np.sum(left * right.T))
    sum_product = float(left.sum(axis=0) @ right.sum(axis=1))
    numerator = (
        trace_product
        + float(left.sum() * right.sum()) / ((n - 1) * (n - 2))
        - 2.0 * sum_product / (n - 2)
    )
    return numerator / (n * (n - 3))
```

For the CKA null, initialize `Generator(PCG64(2026071403))` once and call `permutation(n)` 10,000 times. Compute `tr(K @ L_perm)` as `sum(K * L_perm.T)` and `sum(K @ L_perm)` from column/row sums, avoiding a cubic matrix product. Test each efficient value against a brute-force small fixture and never clamp negative CKA.

- [ ] **Step 4: Run statistics tests and compare smoke G-Vendi to the official reference**

Run:

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest -q \
  math_eval/test_opd_proxy_gradient_statistics.py
```

Expected: all tests PASS; official and local smoke G-Vendi agree within `rtol=1e-6, atol=1e-6`.

- [ ] **Step 5: Commit the pure statistics layer**

```bash
git add math_eval/opd_proxy_gradient_statistics.py \
  math_eval/test_opd_proxy_gradient_statistics.py
git commit -m "feat: add OPD proxy target-space statistics"
```

### Task 14: Pre-Registered Classification and Conditional Stage-2 Decision

**Files:**
- Create: `math_eval/opd_proxy_gradient_classification.py`
- Test: `math_eval/test_opd_proxy_gradient_classification.py`

**Interfaces:**
- Consumes: per-seed/per-K selected metrics and target CKA p-value.
- Produces: `exact_median_of_eight`, `evaluate_oracle_gate`, `evaluate_pn1_components`, `evaluate_strict_arm_components`, `classify_stage1`, `decide_stage2`, and `classify_stage2_sensitivity`.

- [ ] **Step 1: Write failing exact-boundary, worst-case, oracle-precedence, and Stage-2 tests**

```python
def test_exact_pn1_thresholds_pass():
    components = evaluate_pn1_components(
        worst_by_realization=make_worst_cases(
            g_vendi=[.89, .89, .95, .95, .95, .95, .95, .95],
            coverage=[.09, .10, .50, .50, .50, .50, .50, .50],
            gradient_norm=[0, 0, .10, .10, .10, .10, .10, .10],
            opd_signal=[0, 0, .10, .10, .10, .10, .10, .10],
        )
    )
    assert components["g_vendi"]["pass"] is True
    assert components["coverage"]["pass"] is True
    assert components["gradient_norm"]["pass"] is True
    assert components["opd_signal"]["pass"] is True


@pytest.mark.parametrize(
    ("median", "expected"),
    [(0.899999, False), (0.90, True), (0.949999, True), (0.95, False)],
)
def test_stage2_trigger_uses_half_open_gvendi_interval(median, expected):
    report = stage1_report(oracle_pass=True, pn1_g_vendi_median=median)
    assert decide_stage2(report)["run_stage2"] is expected
```

Test that only five of eight G-Vendi values at least `.90` fails despite a passing median; only six of eight coverage values at least `.10` fails; minimum over both target and K-means seeds is retained; stratified and diagnostic-K values cannot change primary classification; oracle failure makes all candidate arms inconclusive; a coverage failure does not suppress the Stage-2 trigger when the primary G-Vendi median is inside `[.90,.95)`; and expanded oracle/CKA failure yields `extended_inconclusive`.

- [ ] **Step 2: Run classification tests and verify failure**

Run:

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest -q \
  math_eval/test_opd_proxy_gradient_classification.py
```

Expected: FAIL because the classification module is missing.

- [ ] **Step 3: Encode the exact conjunctions without library median defaults**

```python
def exact_median_of_eight(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if len(ordered) != 8 or not all(math.isfinite(value) for value in ordered):
        raise ValueError("median-of-eight requires eight finite values")
    return 0.5 * (ordered[3] + ordered[4])


def decide_stage2(stage1_report: Mapping[str, object]) -> dict[str, object]:
    if stage1_report["oracle"]["classification"] != "pass":
        return {"run_stage2": False, "reason": "oracle_not_pass"}
    median = float(stage1_report["P_n1"]["components"]["g_vendi"]["median"])
    return {
        "run_stage2": 0.90 <= median < 0.95,
        "reason": "borderline_primary_pn1_g_vendi" if 0.90 <= median < 0.95 else "outside_trigger",
    }
```

Implement the classification precedence exactly as the design: correctness abort is outside this module; oracle failure produces candidate `inconclusive`; otherwise `P_n1` requires four component gates, and `P_n4`, S, and E are independent strict conjunctions. Preserve every raw failed component in the result.

- [ ] **Step 4: Run all gate tests**

Run:

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest -q \
  math_eval/test_opd_proxy_gradient_classification.py
```

Expected: all tests PASS at exact `.10/.50/.90/.95/.01` boundaries.

- [ ] **Step 5: Commit the classification contract**

```bash
git add math_eval/opd_proxy_gradient_classification.py \
  math_eval/test_opd_proxy_gradient_classification.py
git commit -m "feat: encode OPD proxy verification gates"
```

### Task 15: Strict Analyzer and Byte-Stable JSON/Markdown Reports

**Files:**
- Create: `math_eval/analyze_opd_proxy_gradient_verify.py`
- Test: `math_eval/test_analyze_opd_proxy_gradient_verify.py`

**Interfaces:**
- Consumes: all vector, selection, random, stage, and provenance manifests plus Tasks 13-14.
- Produces: `run_analysis`, `render_markdown`, and `stage_{0,1,2}/report.json`/`report.md`.

- [ ] **Step 1: Write a failing end-to-end synthetic analyzer fixture**

```python
def test_json_and_markdown_share_one_report_dict(synthetic_stage_dir, tmp_path):
    report = run_analysis(stage_dir=synthetic_stage_dir, reference_repo=REFERENCE_REPO)
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"
    write_reports(report, json_path, md_path)
    loaded = json.loads(json_path.read_text())
    assert render_markdown(loaded) == md_path.read_text()
    assert "NaN" not in json_path.read_text()
    assert "Infinity" not in json_path.read_text()


def test_stage0_is_smoke_only(synthetic_stage0_dir):
    report = run_analysis(stage_dir=synthetic_stage0_dir, reference_repo=REFERENCE_REPO)
    assert report["classification"] == {"status": "smoke_only"}
```

Add failures for any missing Pn1/Pn4/target seed/slot/K-ratio/K-means seed, random dtype/shape/hash mismatch, changed parent hash, differing ID order, same-seed oracle entering the primary gate, and Stage-1 constants leaking into Stage 2. Assert 53 CKA representation pairs, 212 AMI, 212 ARI, and 292 primary-K selected-overlap records.

- [ ] **Step 2: Run analyzer tests and verify failure**

Run:

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest -q \
  math_eval/test_analyze_opd_proxy_gradient_verify.py
```

Expected: FAIL because `run_analysis` is missing.

- [ ] **Step 3: Implement strict loading, paired null evaluation, diagnostics, and rendering**

Load two target group vector sets, eight `P_n1`, two `P_n4`, S, and E in manifest candidate order. Compute each uniform and stratified null table once per target seed, then reuse it for every selected arm. Primary gates use only uniform percentiles. For `P_n1`, take each realization's minimum over two target seeds and two primary K-means seeds; do not pool diagnostic K.

Oracle selection names are `(selection_target_seed, kmeans_seed)` and primary evaluation always uses the opposing target seed. CKA uses candidate IDs only. Use `std(ddof=0)`, `quantile(method="linear")`, structured `{"status": "not_computed"}` for absent verifier diagnostics, and `json.dumps(report, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)`.

Emit all secondary diagnostics without allowing them into gates: aligned CKA/AMI/ARI records for every predeclared pair/seed combination; Jaccard and chance-adjusted overlap for every primary-K pairing; separately labeled diagnostic-K results; topic coverage/entropy; prompt, R1 completion, rollout, supervised-token, and valid-token summaries at quantiles `[0,.25,.5,.75,.9,.95,1]`; verifier counts when present; and sampled reverse-KL/OPD-signal distributions for every selected arm plus both random nulls.

The report writer must publish JSON and Markdown only after full validation, derive Markdown solely from the in-memory report dictionary, fsync/rename both, and delete neither prior valid report nor upstream artifacts after a failure.

- [ ] **Step 4: Run analyzer, selector, statistics, and classification tests together**

Run:

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest -q \
  math_eval/test_select_opd_proxy_gradient_verify.py \
  math_eval/test_opd_proxy_gradient_statistics.py \
  math_eval/test_opd_proxy_gradient_classification.py \
  math_eval/test_analyze_opd_proxy_gradient_verify.py
```

Expected: all tests PASS; two repeated synthetic analyses are byte-identical.

- [ ] **Step 5: Commit the analyzer**

```bash
git add math_eval/analyze_opd_proxy_gradient_verify.py \
  math_eval/test_analyze_opd_proxy_gradient_verify.py
git commit -m "feat: add OPD proxy verification analyzer"
```

### Task 16: Fail-Closed `opd-CLI` Capture Launcher and Stage Orchestrator

**Files:**
- Create: `verl/examples/fire_opd/run_capture_opd_proxy_verify.sh`
- Create: `math_eval/run_opd_proxy_gradient_verify.py`
- Create: `math_eval/test_run_opd_proxy_gradient_verify.py`

**Interfaces:**
- Consumes: all preceding CLIs plus one active four-GPU `opd-CLI` Slurm allocation.
- Produces: dry-run command plans, allocation validation, locked/resumable Stage-0/1/2 execution, and exact log paths.

- [ ] **Step 1: Write failing fake-runtime tests for allocation and ordered stages**

```python
def test_runtime_requires_four_unique_allocated_tokens(fake_runtime):
    for visible in ("", "0,1,2", "0,1,1,3", "0,1,2,3,4"):
        with pytest.raises(RuntimeError):
            validate_opd_cli_runtime(env=fake_runtime(cuda_visible_devices=visible))


def test_dry_run_orders_capture_replay_selection_analysis(stage0_manifest):
    commands = build_stage_commands(stage=0, manifest=stage0_manifest)
    names = [command.name for command in commands]
    assert names[:4] == [
        "capture_target_seed_42",
        "capture_target_seed_43",
        "capture_proxy_seed_42",
        "capture_proxy_seed_43",
    ]
    assert names.index("validate_vectors") < names.index("select") < names.index("analyze")
    assert all("srun" not in command.argv for command in commands)
```

Test missing Slurm job, missing/wrong tmux session, occupied GPUs, a non-idle check not repeated immediately before each GPU stage, duplicate output lock, parent mismatch, complete-artifact skip, replay/actor-chunk contiguous-prefix resume, seed-42 failure preventing seed 43, smoke failure preventing Stage 1, cleanup limited to launcher-owned child PIDs, exact VERL-versus-G-Vendi child-interpreter routing, parser acceptance/forwarding of `--reference-repo` for both stage execution and `validate-stage`, and the fully composed four-GPU Hydra contract. The latter must assert one node/four GPUs, TP/DP/PP `4/1/1`, exactly one rollout engine replica, 32,768 batched tokens, 18,432 model length, and non-null rollout/ref log-prob micro-batches. An incomplete rollout-generation subtree must block rather than trigger a second engine/call; only a fully published matching rollout subtree can be reused after restart.

- [ ] **Step 2: Run runtime and shell syntax tests and verify failure**

Run:

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  math_eval/test_run_opd_proxy_gradient_verify.py
bash -n verl/examples/fire_opd/run_capture_opd_proxy_verify.sh
```

Expected: pytest FAIL because the runtime module is missing; shell syntax passes only after the launcher is created.

- [ ] **Step 3: Implement exact capture arguments and ordered orchestration**

The capture launcher derives pair-specific student/teacher paths and capture parquet, then passes these frozen overrides (angle-bracket values are resolved from the immutable stage manifest, never shell defaults):

```text
data.train_files=<target-or-proxy capture parquet>
data.val_files=[]
data.train_batch_size=<manifest pair question count>
data.max_prompt_length=2048
data.max_response_length=16384
data.filter_overlong_prompts=false
data.truncation=error
data.shuffle=false
data.return_raw_chat=true
+data.apply_chat_template_kwargs.enable_thinking=false
trainer.nnodes=1
trainer.n_gpus_per_node=4
trainer.logger='["console"]'
trainer.val_before_train=false
trainer.save_freq=-1
trainer.test_freq=-1
trainer.total_epochs=1
trainer.resume_mode=disable
actor_rollout_ref.model.path=<target:models/Qwen3-4B|proxy:models/Qwen3-0.6B>
+actor_rollout_ref.ref.model.path=<target:models/Qwen3-30B-A3B-Instruct-2507|proxy:models/Qwen3-4B>
actor_rollout_ref.model.use_remove_padding=true
actor_rollout_ref.rollout.n=4
actor_rollout_ref.rollout.name=vllm
actor_rollout_ref.rollout.temperature=1.0
actor_rollout_ref.rollout.top_p=1.0
actor_rollout_ref.rollout.seed=<42|43>
actor_rollout_ref.rollout.calculate_log_probs=true
actor_rollout_ref.rollout.tensor_model_parallel_size=4
actor_rollout_ref.rollout.data_parallel_size=1
actor_rollout_ref.rollout.pipeline_model_parallel_size=1
actor_rollout_ref.rollout.max_model_len=18432
actor_rollout_ref.rollout.max_num_batched_tokens=32768
actor_rollout_ref.rollout.gpu_memory_utilization=0.6
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
actor_rollout_ref.ref.fsdp_config.param_offload=true
actor_rollout_ref.actor.ppo_epochs=1
actor_rollout_ref.actor.ppo_mini_batch_size=<manifest pair question count>
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=18432
actor_rollout_ref.actor.use_dynamic_bsz=false
actor_rollout_ref.actor.opd_proxy_verify_capture_only=true
actor_rollout_ref.actor.loss_agg_mode=token-mean
actor_rollout_ref.actor.policy_loss.loss_mode=vanilla
actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=true
actor_rollout_ref.actor.policy_loss.length_aware_opd=false
actor_rollout_ref.actor.policy_loss.entropy_aware_distill=false
actor_rollout_ref.actor.use_kl_loss=true
actor_rollout_ref.actor.kl_loss_coef=0
actor_rollout_ref.actor.entropy_coeff=0
algorithm.rollout_correction.rollout_is=token
algorithm.rollout_correction.rollout_is_threshold=5.0
algorithm.rollout_correction.rollout_is_batch_normalize=false
algorithm.rollout_correction.rollout_rs=null
algorithm.rollout_correction.bypass_mode=false
algorithm.use_kl_in_reward=false
algorithm.candidate_selection.enabled=false
algorithm.tale_budget.enabled=false
algorithm.difficulty_aware_opd.enabled=false
algorithm.rethinking_opd_probe.enabled=false
algorithm.opd_proxy_verify_capture.enabled=true
algorithm.opd_proxy_verify_capture.output_root=<stage/pair/seed capture directory>
algorithm.opd_proxy_verify_capture.sample_manifest=<canonical stage sample manifest>
algorithm.opd_proxy_verify_capture.sample_manifest_sha256=<canonical manifest SHA-256>
algorithm.opd_proxy_verify_capture.stage=<0|1|2>
algorithm.opd_proxy_verify_capture.pair=<target|proxy>
algorithm.opd_proxy_verify_capture.engine_seed=<42|43>
algorithm.opd_proxy_verify_capture.native_rollouts=4
algorithm.opd_proxy_verify_capture.expected_questions=<manifest pair question count>
algorithm.opd_proxy_verify_capture.chunk_size=16
algorithm.opd_proxy_verify_capture.schema_version=1
```

Explicitly set every forbidden feature `enabled=false`, and have contract validation reject forbidden keys that reach the capture batch. Before either capture pair starts, recursively hash both uses of `models/Qwen3-4B` and require byte identity between target-student and proxy-teacher manifests. The shell script must not call `srun`, install packages, infer physical GPU indices, or default `CUDA_VISIBLE_DEVICES`.

The Python orchestrator validates `SLURM_JOB_ID`, tmux session `opd-CLI`, four unique visible tokens, and GPU idleness immediately before every GPU subprocess. Keep its imports limited to the standard library and shared dependency-light contracts. Route only forward capture through `/home/mchen/miniconda3/envs/verl/bin/python`; route the direct production-backward/projector fixture, replay, SFT-gradient collection, embedding collection, official K-means, CPU selection/statistics, and report generation through `/home/mchen/miniconda3/envs/gvendi-opd/bin/python`. It runs four capture work units sequentially, four single-token replay shards concurrently with each child seeing one allocated token as local `cuda:0`, baseline collectors, and official K-means with exactly one visible allocation token, followed by CPU selection/statistics and report regeneration. Complete matching work units skip; mismatched work units fail. Fake-runtime tests assert the exact interpreter selected for every child command, prove the direct fixture can import both VERL and `trak.projectors`, and fail if any projector/analysis unit inherits the VERL interpreter.

Expose and test these exact CLI surfaces because later execution tasks call them:

```text
run_opd_proxy_gradient_verify --stage {0,1,2} [--parent-report PATH] [--exercise-resume] [--reference-repo PATH]
run_opd_proxy_gradient_verify check-runtime --expected-gpus 4 --require-idle
run_opd_proxy_gradient_verify validate-stage --stage N [--preflight-only|--require-complete] [--regenerate-report] [--reference-repo PATH]
run_opd_proxy_gradient_verify estimate --from-stage 0 --to-stage 1 --safety-factor 1.20 --reserve-seconds 7200
run_opd_proxy_gradient_verify stage2-decision --stage1-report PATH
```

For Stage 0, insert the direct production-backward fixture after capture and before replay validation, and implement `--exercise-resume` by interrupting only after immutable rollout `COMPLETE.json` exists and during a replay/actor-chunk prefix. The restart must reuse rollout bytes without starting an engine and reproduce uninterrupted downstream hashes. These smoke-only actions never run for Stage 1 or 2.

Build the source snapshot from every code/config/launcher/test file in this plan, the approved design, this implementation plan, and every imported Prismatic reference file. Test that removing one entry or changing an untracked launcher's bytes changes the source manifest hash before resume is considered.

Record wall-clock duration and completed-row count for every Stage-0 work unit. The `estimate` subcommand scales the slowest observed per-row rate to each remaining capture/replay/baseline/selection unit, multiplies their sum by the requested safety factor, adds the reserve, parses the active job `EndTime` from `scontrol show job -o "$SLURM_JOB_ID"`, and fails if completion would cross that boundary. Missing timings, job ID, or parseable end time are errors rather than optimistic defaults.

- [ ] **Step 4: Run dry-run, shell, and fake-runtime validation**

Run:

```bash
bash -n verl/examples/fire_opd/run_capture_opd_proxy_verify.sh \
  verl/examples/fire_opd/run_direct_opd_proxy_gradient_fixture.sh
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  math_eval/test_run_opd_proxy_gradient_verify.py
```

Expected: all tests PASS; the synthetic-manifest dry-run fixture prints all four pair/seed captures, replay shards, two baselines, selection, and analysis, with no model load and no `srun`. Do not invoke the real Stage-0 CLI yet because the canonical manifest is intentionally created only after final code review in Task 17 Step 6.

- [ ] **Step 5: Commit the launch boundary**

```bash
git add verl/examples/fire_opd/run_capture_opd_proxy_verify.sh \
  math_eval/run_opd_proxy_gradient_verify.py \
  math_eval/test_run_opd_proxy_gradient_verify.py
git commit -m "feat: orchestrate OPD proxy verification stages"
```

### Task 17: Full CPU Regression, Source Audit, and Two-Stage Code Review

**Files:**
- Verify: every file created or modified in Tasks 1-16.

**Interfaces:**
- Consumes: the complete implementation.
- Produces: review evidence that default VERL behavior is unchanged and every pre-GPU gate is satisfied.

- [ ] **Step 1: Run the complete VERL-side CPU suite**

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  verl/tests/workers/rollout/test_opd_proxy_verify_native_n.py \
  verl/tests/workers/config/test_rollout_config_on_cpu.py \
  verl/tests/trainer/config/test_algo_config_on_cpu.py \
  verl/tests/trainer/ppo/test_opd_proxy_verify_capture.py \
  verl/tests/trainer/ppo/test_opd_proxy_verify_capture_launcher.py \
  verl/tests/trainer/ppo/test_rollout_corr.py \
  verl/tests/trainer/ppo/test_rollout_corr_integration.py \
  verl/tests/workers/actor/test_length_aware_opd.py
```

Expected: command exits 0 with no failed tests; capture-disabled controls match the original training path.

- [ ] **Step 2: Run the complete math-eval CPU suite**

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=verl:. \
  /home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest -q \
  math_eval/test_opd_proxy_gradient_verify_artifacts.py \
  math_eval/test_prepare_opd_proxy_gradient_verify.py \
  math_eval/test_opd_proxy_gradient_projection.py \
  math_eval/test_replay_opd_proxy_gradients.py \
  math_eval/test_collect_opd_proxy_sft_gradients.py \
  math_eval/test_collect_opd_proxy_prompt_embeddings.py \
  math_eval/test_select_opd_proxy_gradient_verify.py \
  math_eval/test_opd_proxy_gradient_statistics.py \
  math_eval/test_opd_proxy_gradient_classification.py \
  math_eval/test_analyze_opd_proxy_gradient_verify.py \
  math_eval/test_run_opd_proxy_gradient_verify.py \
  math_eval/test_build_gradient_eligibility.py \
  math_eval/test_collect_prismatic_gradients.py \
  math_eval/test_select_gradient_diverse_deepmath.py \
  -m 'not gpu'
```

Expected: command exits 0; production decontamination fixtures, official-reference adapters, null hashes, and classifications all pass.

- [ ] **Step 3: Validate shell files, source boundaries, and absence of unfinished markers**

```bash
bash -n verl/examples/fire_opd/run_capture_opd_proxy_verify.sh \
  verl/examples/fire_opd/run_direct_opd_proxy_gradient_fixture.sh

rg -n 'TO[D]O|FIX[M]E|PLACE[H]OLDER|NotImplementedError' \
  math_eval/opd_proxy_gradient_verify_artifacts.py \
  math_eval/prepare_opd_proxy_gradient_verify.py \
  math_eval/opd_proxy_gradient_projection.py \
  math_eval/replay_opd_proxy_gradients.py \
  math_eval/collect_opd_proxy_sft_gradients.py \
  math_eval/collect_opd_proxy_prompt_embeddings.py \
  math_eval/select_opd_proxy_gradient_verify.py \
  math_eval/opd_proxy_gradient_statistics.py \
  math_eval/opd_proxy_gradient_classification.py \
  math_eval/analyze_opd_proxy_gradient_verify.py \
  math_eval/run_opd_proxy_gradient_verify.py \
  verl/verl/trainer/ppo/opd_proxy_verify_capture.py

git diff --check
git status --short
```

Expected: shell syntax and `git diff --check` exit 0; marker search prints nothing; status contains no generated data/model/log files and no unrelated user files.

- [ ] **Step 4: Verify byte-identical command plans with the synthetic manifest fixture**

```bash
PYTHONPATH=verl:. /home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  math_eval/test_run_opd_proxy_gradient_verify.py \
  -k 'dry_run_is_byte_identical or dry_run_orders_capture_replay_selection_analysis'
```

Expected: synthetic plans are byte-identical and contain one native generation work unit for each target/proxy seed. The real CLI remains deferred until canonical Stage 0 exists.

- [ ] **Step 5: Request spec-compliance review, then code-quality review**

Use the `requesting-code-review` skill. First review every design invariant and artifact count; after all spec blockers are fixed and Steps 1-4 rerun, request a second review for safety, maintainability, default-path regressions, and resume correctness. Each fix must add or strengthen a regression test and be committed with only its affected files.

- [ ] **Step 6: Freeze the final reviewed source commit and create canonical root/Stage-0 manifests once**

After all review fixes are committed, rerun `git status --short` and require empty output. Record `FINAL_SOURCE_COMMIT="$(git rev-parse HEAD)"`; no source/config/test/launcher change is permitted after this point without discarding all runtime artifacts and starting a new output root. Require that `data/opd_proxy_gradient_verify` has no canonical `root_manifest.json` or `stage_0/manifest.json`, then run:

```bash
test -z "$(git status --short)"
FINAL_SOURCE_COMMIT="$(git rev-parse HEAD)"
test ! -e data/opd_proxy_gradient_verify/root_manifest.json
test ! -e data/opd_proxy_gradient_verify/stage_0/manifest.json

CUDA_VISIBLE_DEVICES="" PYTHONPATH=verl:. \
  /home/mchen/miniconda3/envs/verl/bin/python \
  -m math_eval.prepare_opd_proxy_gradient_verify prepare-root \
  --output-root data/opd_proxy_gradient_verify \
  --expected-source-commit "$FINAL_SOURCE_COMMIT" \
  --reference-repo /home/mchen/prismatic-synthesis-reference

CUDA_VISIBLE_DEVICES="" PYTHONPATH=verl:. \
  /home/mchen/miniconda3/envs/verl/bin/python \
  -m math_eval.prepare_opd_proxy_gradient_verify prepare-stage \
  --stage 0 --output-root data/opd_proxy_gradient_verify \
  --expected-source-commit "$FINAL_SOURCE_COMMIT" \
  --reference-repo /home/mchen/prismatic-synthesis-reference
```

Expected: both manifests bind the clean final HEAD, the complete byte-level source snapshot, frozen model/tokenizer/reference hashes, and the approved sampling contract. Publication is create-once: an existing canonical artifact is validated byte-for-byte or rejected, never overwritten.

- [ ] **Step 7: Dry-run the real canonical Stage-0 CLI twice**

```bash
for run in 1 2; do
  CUDA_VISIBLE_DEVICES="GPU-a,GPU-b,GPU-c,GPU-d" \
    OPD_PROXY_VERIFY_DRY_RUN=1 \
    PYTHONPATH=verl:. /home/mchen/miniconda3/envs/verl/bin/python \
    -m math_eval.run_opd_proxy_gradient_verify --stage 0 \
    --reference-repo /home/mchen/prismatic-synthesis-reference \
    > "/tmp/opd_proxy_verify_dry_run_${run}.txt"
done
cmp /tmp/opd_proxy_verify_dry_run_1.txt /tmp/opd_proxy_verify_dry_run_2.txt
```

Expected: `cmp` exits 0; both plans resolve the canonical 24/8 Stage-0 manifest, contain the same four pair/seed capture work units and child interpreters, and perform no model load or GPU call.

### Task 18: Stage-0 Real-Model Smoke in the Authorized `opd-CLI` Allocation

**Files:**
- Runtime outputs only: `data/opd_proxy_gradient_verify/stage_0/` and `logs/opd_proxy_gradient_verify/stage_0/`.

**Interfaces:**
- Consumes: reviewed implementation, frozen Stage-0 manifest, and the active four-GPU allocation.
- Produces: complete real capture/replay/baseline/selection smoke artifacts and a `smoke_only` report.

- [ ] **Step 1: Locate and validate the live allocation without assuming a pane index**

From the implementation worktree shell:

```bash
OPD_PANE="$(tmux list-panes -a -F '#S:#I.#P #{pane_dead}' \
  | awk '$1 ~ /^opd-CLI:/ && $2 == 0 {print $1; exit}')"
test -n "$OPD_PANE"
printf 'pane=%s\n' "$OPD_PANE"
tmux send-keys -t "$OPD_PANE" \
  "printf 'session=%s job=%s cvd=%s\\n' \"\$(tmux display-message -p '#S')\" \"\${SLURM_JOB_ID:-}\" \"\${CUDA_VISIBLE_DEVICES:-}\"" C-m
```

Expected: one live `opd-CLI` pane, nonempty Slurm job ID, and four allocation-owned visible tokens. If absent, stop and report that GPU execution is blocked; do not create an allocation.

- [ ] **Step 2: Run allocation/GPU idleness validation inside the pane**

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
CHECK_CMD="cd $(printf '%q' "$REPO_ROOT") && PYTHONPATH=verl:. /home/mchen/miniconda3/envs/verl/bin/python -m math_eval.run_opd_proxy_gradient_verify check-runtime --expected-gpus 4 --require-idle"
tmux send-keys -t "$OPD_PANE" "$CHECK_CMD" C-m
```

Expected: session is `opd-CLI`, the four visible tokens are unique, and no unexpected compute process occupies them. Stop on failure.

- [ ] **Step 3: Launch Stage 0 directly in `opd-CLI`, without `srun`**

```bash
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_CMD="cd $(printf '%q' "$REPO_ROOT") && mkdir -p logs/opd_proxy_gradient_verify/stage_0 && PYTHONPATH=verl:. /home/mchen/miniconda3/envs/verl/bin/python -m math_eval.run_opd_proxy_gradient_verify --stage 0 --reference-repo /home/mchen/prismatic-synthesis-reference --exercise-resume 2>&1 | tee logs/opd_proxy_gradient_verify/stage_0/orchestrator_${STAMP}.log"
tmux send-keys -t "$OPD_PANE" "$RUN_CMD" C-m
```

Expected: root/sample/model/tokenizer/source hashes pass before the first GPU load; four capture work units each make one native `n=4` call; the orchestrator then runs replay, SFT, embedding, official K-means, resume exercise, and analysis.

- [ ] **Step 4: Verify every real-model smoke gate before Stage 1**

Run the CPU validator after the process exits:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=verl:. \
  /home/mchen/miniconda3/envs/gvendi-opd/bin/python \
  -m math_eval.run_opd_proxy_gradient_verify validate-stage \
  --stage 0 --require-complete \
  --reference-repo /home/mchen/prismatic-synthesis-reference
```

Expected validation includes:

```text
32 sample IDs: 24 candidate + 8 held-out
2 target seeds x 32 questions x 4 exact slots
2 proxy seeds x 24 questions x 4 exact slots
capture/replay masks and token IDs identical
actor-derived log probabilities and losses within rtol=5e-3, atol=5e-3
direct production-backward vs replay cosine >= 0.999
direct production-backward relative norm error <= 0.005
target P(mean) and proxy mean(P) checks within rtol=5e-3, atol=5e-3
actor parameter hashes unchanged
interrupted/resumed artifacts byte-identical to uninterrupted artifacts
official CUDA K-means seeds 42 and 43 complete
report classification status: smoke_only
```

Any missing gate blocks Stage 1; diagnose with systematic debugging rather than weakening tolerance or deleting artifacts.

- [ ] **Step 5: Record smoke evidence without committing runtime artifacts**

```bash
git status --short
sha256sum data/opd_proxy_gradient_verify/stage_0/report.json \
  data/opd_proxy_gradient_verify/stage_0/report.md
```

Expected: runtime paths are ignored/untracked from commits, and both report hashes are recorded in the execution handoff.

### Task 19: Stage-1 Main Probe and Conditional Stage-2 Sensitivity

**Files:**
- Runtime outputs only under `data/opd_proxy_gradient_verify/stage_{1,2}/` and `logs/opd_proxy_gradient_verify/stage_{1,2}/`.

**Interfaces:**
- Consumes: a passing Stage 0, reviewed source snapshot, and enough time remaining in an authorized `opd-CLI` allocation.
- Produces: the Stage-1 primary scientific classification and, only when triggered and authorized, a separate Stage-2 sensitivity classification.

- [ ] **Step 1: Prepare and validate the immutable Stage-1 view on CPU**

```bash
test -z "$(git status --short)"
FINAL_SOURCE_COMMIT="$(git rev-parse HEAD)"

CUDA_VISIBLE_DEVICES="" PYTHONPATH=verl:. \
  /home/mchen/miniconda3/envs/verl/bin/python \
  -m math_eval.prepare_opd_proxy_gradient_verify prepare-stage \
  --stage 1 --output-root data/opd_proxy_gradient_verify \
  --expected-source-commit "$FINAL_SOURCE_COMMIT" \
  --reference-repo /home/mchen/prismatic-synthesis-reference

CUDA_VISIBLE_DEVICES="" PYTHONPATH=verl:. \
  /home/mchen/miniconda3/envs/verl/bin/python \
  -m math_eval.run_opd_proxy_gradient_verify validate-stage \
  --stage 1 --preflight-only \
  --reference-repo /home/mchen/prismatic-synthesis-reference
```

Expected: candidate/held-out counts `768/256`; ordered hashes exactly match the approved design; Stage-0 artifacts are not mutated.

- [ ] **Step 2: Revalidate allocation idleness and remaining wall time**

Repeat Task 18 Steps 1-2, then run the orchestrator's ETA check using measured Stage-0 rates and a 1.20 safety factor plus a two-hour reserve:

```bash
ETA_CMD="cd $(printf '%q' "$REPO_ROOT") && PYTHONPATH=verl:. /home/mchen/miniconda3/envs/verl/bin/python -m math_eval.run_opd_proxy_gradient_verify estimate --from-stage 0 --to-stage 1 --safety-factor 1.20 --reserve-seconds 7200"
tmux send-keys -t "$OPD_PANE" "$ETA_CMD" C-m
```

Expected: estimate fits before Slurm end time. If it does not, stop and request a fresh user-authorized allocation.

- [ ] **Step 3: Launch Stage 1 in the same authorized pane**

```bash
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_CMD="cd $(printf '%q' "$REPO_ROOT") && mkdir -p logs/opd_proxy_gradient_verify/stage_1 && PYTHONPATH=verl:. /home/mchen/miniconda3/envs/verl/bin/python -m math_eval.run_opd_proxy_gradient_verify --stage 1 --reference-repo /home/mchen/prismatic-synthesis-reference 2>&1 | tee logs/opd_proxy_gradient_verify/stage_1/orchestrator_${STAMP}.log"
tmux send-keys -t "$OPD_PANE" "$RUN_CMD" C-m
```

Expected final coverage:

```text
1,024 sample rows: 768 candidate + 256 held-out
2,048 target question-seed vectors from 8,192 trajectories
6,144 proxy per-trajectory vectors and two P_n4 views
768 SFT vectors
768 prompt embeddings
10,000 uniform and 10,000 stratified random subsets
172 IDs in every selected set
all eight P_n1, two P_n4, S, E, and cross-seed oracle records
```

- [ ] **Step 4: Validate and regenerate the Stage-1 report on CPU**

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=verl:. \
  /home/mchen/miniconda3/envs/gvendi-opd/bin/python \
  -m math_eval.run_opd_proxy_gradient_verify validate-stage \
  --stage 1 --require-complete --regenerate-report \
  --reference-repo /home/mchen/prismatic-synthesis-reference
```

Expected: regenerated JSON/Markdown hashes match published files; report contains oracle, `P_n1`, `P_n4`, S, E, uniform, stratified, both target seeds, both K-means seeds, every guardrail, and every failed component. Provenance/correctness failure produces no scientific classification.

- [ ] **Step 5: Evaluate the predeclared extension decision without silently launching it**

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=verl:. \
  /home/mchen/miniconda3/envs/gvendi-opd/bin/python \
  -m math_eval.run_opd_proxy_gradient_verify stage2-decision \
  --stage1-report data/opd_proxy_gradient_verify/stage_1/report.json
```

Expected outcomes:

- `oracle_not_pass`, clear fail below `.90`, clear pass at or above `.95`, or provenance failure: do not create Stage 2.
- Oracle pass and `0.90 <= P_n1 worst-case G-Vendi median < 0.95`: print the exact Stage-2 cost/ETA and stop for confirmation that an authorized allocation has sufficient wall time.

- [ ] **Step 6: If and only if the gate fires and execution is confirmed, prepare and run append-only Stage 2**

Prepare Stage 2 with the exact Stage-1 report as parent and the same frozen source/reference provenance:

```bash
test -z "$(git status --short)"
FINAL_SOURCE_COMMIT="$(git rev-parse HEAD)"

CUDA_VISIBLE_DEVICES="" PYTHONPATH=verl:. \
  /home/mchen/miniconda3/envs/verl/bin/python \
  -m math_eval.prepare_opd_proxy_gradient_verify prepare-stage \
  --stage 2 --output-root data/opd_proxy_gradient_verify \
  --expected-source-commit "$FINAL_SOURCE_COMMIT" \
  --reference-repo /home/mchen/prismatic-synthesis-reference \
  --parent-report data/opd_proxy_gradient_verify/stage_1/report.json

CUDA_VISIBLE_DEVICES="" PYTHONPATH=verl:. \
  /home/mchen/miniconda3/envs/verl/bin/python \
  -m math_eval.run_opd_proxy_gradient_verify validate-stage \
  --stage 2 --preflight-only \
  --reference-repo /home/mchen/prismatic-synthesis-reference
```

Expected: Stage 2 is create-once, validates the exact Stage-1 report SHA from `--parent-report`, and appends only the predeclared 768 candidate/256 held-out rows. Revalidate the allocation and wall-time estimate, then send:

```bash
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_CMD="cd $(printf '%q' "$REPO_ROOT") && mkdir -p logs/opd_proxy_gradient_verify/stage_2 && PYTHONPATH=verl:. /home/mchen/miniconda3/envs/verl/bin/python -m math_eval.run_opd_proxy_gradient_verify --stage 2 --parent-report data/opd_proxy_gradient_verify/stage_1/report.json --reference-repo /home/mchen/prismatic-synthesis-reference 2>&1 | tee logs/opd_proxy_gradient_verify/stage_2/orchestrator_${STAMP}.log"
tmux send-keys -t "$OPD_PANE" "$RUN_CMD" C-m
```

Expected: only 768 appended candidates and 256 appended held-out questions are captured/collected; Stage-1 vectors are reused by verified parent hash; full expanded selection/metrics are recomputed over `1,536/512`, with K `153/15`, selection size 345, and a separate `extended_pass|extended_fail|extended_inconclusive` label that cannot modify Stage 1.

- [ ] **Step 7: Hand off the result with evidence, not interpretation inflation**

Report the Stage-1 oracle status, all four `P_n1` component gates, independent `P_n4`/S/E classifications, raw worst-case percentiles, key diagnostics, report paths/hashes, runtime cost, and whether Stage 2 ran. State explicitly that a pass supports a later full-pool selector experiment but does not establish OOD improvement.

---

## Dependency and Execution Order

```text
Tasks 1-3   immutable data/provenance contract
Task 4      typed native-n rollout
Tasks 5-6   optimizer-free capture
Tasks 7-9   projection, replay, and direct-gradient equivalence
Tasks 10-11 controlled baselines
Task 12     unified selector and random schedules
Tasks 13-15 metrics, gates, and reports
Task 16     fail-closed runtime orchestration
Task 17     CPU regression and review gate
Task 18     Stage-0 real-model gate
Task 19     Stage-1 result and conditional Stage-2 sensitivity
```

Tasks 10 and 11 may be implemented in parallel after Tasks 1-3 and 7. Tasks 13 and 14 may be implemented in parallel after Task 1. All other dependencies are sequential as shown. No GPU task begins before Task 17 passes.
