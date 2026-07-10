# Gradient-Diverse DeepMath Coreset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and launch a reproducible offline pipeline that preserves all 57,046 DeepMath level-6 source identities, excludes only explicitly recorded proxy-context-ineligible rows, and selects exactly 12,800 unique eligible questions using official Prismatic-Synthesis projected gradients while adding no OPD rollout or teacher cost.

**Architecture:** A pure helper module owns exact joins, provenance, sharding, gradient validation, and balanced fixed-pool selection. Thin CLIs prepare the R1-completion pool, write a pinned token-eligibility report, invoke the official `GradientComputer` on eligible stable IDs, and run code-/paper-aligned clustering diagnostics before writing a schema-preserving parquet. A resumable shell launcher orchestrates the stages on explicitly chosen GPUs and is the only entry point used in `opd-CLI`.

**Tech Stack:** Python 3.10, PyArrow, Hugging Face Datasets/Transformers, PyTorch, TRL, TRAK, safetensors, scikit-learn, pytest, Bash, tmux.

## Global Constraints

- Work only in `/home/mchen/FiRe-OPD/.worktrees/gradient-diverse-data-selection` until integration.
- Preserve the source parquet Arrow schema exactly and select exactly 12,800 unique rows from 57,046.
- Use only `r1_solution_1`; do not generate new solutions.
- Pin `zwhe99/DeepMath-103K` to `5cf055d1fe3d7a2eb19719ac020211469736ae44`.
- Pin `Qwen/Qwen2.5-0.5B-Instruct` to `7ae557604adf67be50417f59c2c2f167def9a775`.
- Reuse official `GradientComputer` from `/home/mchen/prismatic-synthesis-reference` at commit `d9484cd3b5991030b901ac4a3a9e2472dbfac2ad`; do not copy its source.
- Keep official projection dimension 1024, Rademacher seed 0, project interval 4, completion-only loss, and full-parameter gradients.
- Pin `traker==0.3.2` and `fast-jl==0.1.3`; validate their observed official-path float32 projected output without adding a cast.
- Use code-aligned cluster ratio 0.10 for the output and paper-aligned ratio 0.01 for diagnostics.
- Do not use any evaluation examples or metrics during selection.
- No production code is written before its corresponding test has failed for the expected reason.
- Do not interrupt existing processes in `opd-CLI`; launch only on GPUs verified free immediately beforehand.

---

### Task 1: Pure preparation, provenance, and balanced-selection helpers

**Files:**
- Create: `math_eval/deepmath_gradient_diversity.py`
- Create: `math_eval/test_deepmath_gradient_diversity.py`

**Interfaces:**
- Produces: `strip_opd_instruction(prompt: str) -> str`
- Produces: `normalize_math_answer(answer: object) -> str`
- Produces: `join_filtered_rows(filtered_rows: Sequence[Mapping], original_rows: Sequence[Mapping], expected_count: int | None) -> list[dict]`
- Produces: `official_shard_bounds(total: int, num_shards: int, shard_index: int) -> tuple[int, int]`
- Produces: `sha256_file(path: Path) -> str`
- Produces: `validate_gradient_matrix(ids: Sequence[str], gradients: torch.Tensor, expected_ids: Sequence[str]) -> None`
- Produces: `balanced_round_robin(labels: Sequence[int], target_size: int, seed: int) -> list[int]`
- Produces: `cluster_size_summary(labels: Sequence[int], requested_clusters: int) -> dict[str, float | int]`

- [ ] **Step 1: Write failing tests for exact prompt normalization and join**

```python
def test_join_filtered_rows_uses_exact_question_and_r1_solution_one():
    filtered = [{
        "prompt": [{"role": "user", "content": "Solve x.\nPlease reason step by step, and put your final answer within \\boxed{}."}],
        "reward_model": {"ground_truth": "2"},
        "extra_info": {"index": 7, "split": "train"},
    }]
    original = [{
        "question": "Solve x.",
        "final_answer": "2",
        "difficulty": 6.0,
        "topic": "Algebra",
        "r1_solution_1": "Reasoning... \\boxed{2}",
        "r1_solution_2": "unused",
        "r1_solution_3": "unused",
    }]

    result = join_filtered_rows(filtered, original, expected_count=1)

    assert result == [{
        "id": "deepmath-level6-000000",
        "prompt": "Solve x.",
        "completion": "Reasoning... \\boxed{2}",
        "source_row_index": 0,
        "original_dataset_index": 0,
        "topic": "Algebra",
        "difficulty": 6.0,
    }]
```

Add separate tests asserting failures for a missing match, a genuinely ambiguous duplicate-key subsequence, empty `r1_solution_1`, answer mismatch, malformed chat prompt, and unexpected row count. Add a positive fixture in which duplicate question/answer keys are uniquely disambiguated by surrounding row order.

- [ ] **Step 2: Run the join tests and verify RED**

Run:

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  math_eval/test_deepmath_gradient_diversity.py -k 'join or prompt'
```

Expected: collection fails because `math_eval.deepmath_gradient_diversity` does not exist.

- [ ] **Step 3: Implement the minimal exact join**

Implement exact terminal-suffix removal and a `(question, normalized_answer) -> sorted original indices` mapping. Recover the filtered rows with both forward-earliest and backward-latest strictly increasing subsequence passes and require identical mappings. Construct stable prepared rows from that unique mapping. Do not add fuzzy question matching or fallback to another R1 solution.

- [ ] **Step 4: Run the join tests and verify GREEN**

Run the Step 2 command. Expected: all selected tests pass.

- [ ] **Step 5: Write failing tests for official sharding and provenance hashing**

```python
@pytest.mark.parametrize(
    ("shard", "expected"),
    [(0, (0, 7131)), (1, (7131, 14262)), (7, (49917, 57046))],
)
def test_official_shard_bounds_matches_reference_partition(shard, expected):
    assert official_shard_bounds(57046, 8, shard) == expected
```

Also test invalid shard counts/indices and verify `sha256_file` changes when file bytes change.

- [ ] **Step 6: Run the new tests and verify RED**

Expected: imports succeed but the new functions are missing.

- [ ] **Step 7: Implement sharding and SHA-256 helpers**

Use the reference partition size `int(total / num_shards) + 1`, clamping the returned end to `total`.

- [ ] **Step 8: Run the tests and verify GREEN**

Expected: all Task 1 tests added so far pass.

- [ ] **Step 9: Write failing tests for gradient validation and exact balanced selection**

```python
def test_balanced_round_robin_selects_exact_unique_size_and_is_deterministic():
    labels = [0, 0, 0, 1, 1, 2, 2, 2, 2]
    first = balanced_round_robin(labels, target_size=7, seed=42)
    second = balanced_round_robin(labels, target_size=7, seed=42)
    assert first == second
    assert len(first) == 7
    assert len(set(first)) == 7
    assert set(labels[i] for i in first[:3]) == {0, 1, 2}
```

Add tests for insufficient rows, invalid negative labels, ID mismatch, duplicate IDs, non-finite gradients, zero-norm gradients, and cluster-size statistics including empty requested clusters.

- [ ] **Step 10: Run selection tests and verify RED**

Expected: failures identify the missing selection and validation helpers.

- [ ] **Step 11: Implement minimal validation and balanced round-robin logic**

Use `random.Random(seed)` and sorted integer cluster IDs. Shuffle each member list once; shuffle active cluster IDs once per round; pop one member per active cluster until the target is reached.

- [ ] **Step 12: Run all Task 1 tests and verify GREEN**

Run:

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  math_eval/test_deepmath_gradient_diversity.py
```

Expected: all tests pass with no warnings from the new module.

- [ ] **Step 13: Commit Task 1**

```bash
git add math_eval/deepmath_gradient_diversity.py \
  math_eval/test_deepmath_gradient_diversity.py
git commit -m "Add gradient-diverse DeepMath selection helpers"
```

---

### Task 2: Pinned DeepMath preparation CLI

**Files:**
- Create: `math_eval/prepare_deepmath_gradient_pool.py`
- Create: `math_eval/test_prepare_deepmath_gradient_pool.py`

**Interfaces:**
- Consumes: `join_filtered_rows`, `sha256_file`
- Produces: `prepare_pool(source_parquet: Path, output_jsonl: Path, manifest_path: Path, original_rows: Sequence[Mapping], expected_count: int) -> dict`
- CLI inputs: `--source-parquet`, `--output-jsonl`, `--manifest`, `--dataset-name`, `--dataset-revision`, `--expected-count`

- [ ] **Step 1: Write a failing end-to-end preparation test using tiny local rows**

Create a two-row PyArrow parquet fixture and two original DeepMath dictionaries. Call `prepare_pool` and assert:

```python
assert manifest["source_row_count"] == 2
assert manifest["prepared_row_count"] == 2
assert manifest["dataset_revision"] == DATASET_REVISION
assert [json.loads(line)["source_row_index"] for line in output.read_text().splitlines()] == [0, 1]
```

Also assert rerunning with identical inputs is idempotent and rerunning over a manifest with a different source hash fails.

- [ ] **Step 2: Run the preparation test and verify RED**

Run:

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  math_eval/test_prepare_deepmath_gradient_pool.py
```

Expected: module import fails because the preparation CLI does not exist.

- [ ] **Step 3: Implement `prepare_pool` and the pinned dataset CLI**

The CLI must call:

```python
load_dataset(
    "zwhe99/DeepMath-103K",
    split="train",
    revision="5cf055d1fe3d7a2eb19719ac020211469736ae44",
)
```

Write JSONL and manifest through temporary sibling files followed by atomic `Path.replace`. Preserve UTF-8 and write one compact JSON object per line.

- [ ] **Step 4: Run preparation tests and verify GREEN**

Expected: all preparation tests pass without network access.

- [ ] **Step 5: Commit Task 2**

```bash
git add math_eval/prepare_deepmath_gradient_pool.py \
  math_eval/test_prepare_deepmath_gradient_pool.py
git commit -m "Add pinned DeepMath R1 pool preparation"
```

---

### Task 3: Official GradientComputer wrapper with explicit logical shards

**Files:**
- Create: `math_eval/collect_prismatic_gradients.py`
- Create: `math_eval/test_collect_prismatic_gradients.py`

**Interfaces:**
- Consumes: prepared JSONL/manifest and `official_shard_bounds`, `sha256_file`
- Produces: `verify_reference_repo(path: Path, expected_commit: str) -> None`
- Produces: `resolve_resume_start(dataset_ids: Sequence[str], output_dir: Path, prefix: str, shard_start: int, shard_end: int) -> int`
- Produces: `write_or_validate_gradient_manifest(path: Path, expected: Mapping) -> dict`
- CLI inputs include `--reference-repo`, `--prepared-jsonl`, `--prepared-manifest`, `--output-dir`, `--prefix`, `--model-name`, `--model-revision`, `--shard-index`, `--num-shards`, `--device`

- [ ] **Step 1: Write failing tests for reference commit verification, logical shard selection, resume, and manifest mismatch**

Use temporary local Git repositories rather than mocking `git`. Create ID-sidecar fixtures such as `deepmath.0.txt` and assert resume starts immediately after the last valid ID in the shard. Assert gaps, unknown IDs, overlapping chunks, changed shard count, and changed prepared hash fail.

- [ ] **Step 2: Run the wrapper tests and verify RED**

Run:

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  math_eval/test_collect_prismatic_gradients.py
```

Expected: import fails because the wrapper does not exist.

- [ ] **Step 3: Implement provenance and resume helpers without loading a model**

Use `git -C <repo> rev-parse HEAD` and `git -C <repo> status --porcelain`; require the pinned commit and a clean tree. Validate `.safetensors`/`.txt` pairs form a contiguous prefix of the logical shard rather than trusting the largest filename alone. Sidecar IDs must match the expected slice exactly; safetensors must be readable, contain the same keys, and store one finite `(1024,)` vector per ID. Intermediate chunks contain exactly 500 rows, while only a shard-final chunk may be shorter.

- [ ] **Step 4: Run wrapper unit tests and verify GREEN**

Expected: all CPU-only wrapper tests pass.

- [ ] **Step 5: Write a failing test for importing the official collector from its pinned path**

Patch only `AutoModelForCausalLM.from_pretrained` and `AutoTokenizer.from_pretrained`; assert the loader receives the pinned model revision and the constructed collector class comes from `prismatic-synthesis/gradient_modules/gradient_computer.py` under the verified reference repository.

- [ ] **Step 6: Run the import test and verify RED**

Expected: fails because model/collector loading is not implemented.

- [ ] **Step 7: Implement the minimal model and official collector loading path**

Add the reference `prismatic-synthesis` directory to `sys.path`, import `GradientComputer`, load the pinned model and tokenizer with `torch_dtype="auto"` and the exact revision, and move the model to the process-local `cuda:0`. Temporarily wrap the official projector factory so a `BasicProjector` fallback raises before allocation, then assert the instantiated collector uses `CudaProjector`. Check that official completion-only labels contain at least one supervised token and that token length does not exceed the pinned model's `max_position_embeddings`. Call `compute_project_store_gradients` only on the unresolved shard suffix and revalidate complete shard coverage after it returns.

- [ ] **Step 8: Run all Task 3 unit tests and verify GREEN**

Expected: CPU tests pass without allocating CUDA memory.

- [ ] **Step 9: Commit Task 3**

```bash
git add math_eval/collect_prismatic_gradients.py \
  math_eval/test_collect_prismatic_gradients.py
git commit -m "Wrap official Prismatic gradient collection safely"
```

---

### Task 3B: Pinned token-eligibility report and eligible-ID collection

**Files:**
- Create: `math_eval/build_gradient_eligibility.py`
- Create: `math_eval/test_build_gradient_eligibility.py`
- Modify: `math_eval/collect_prismatic_gradients.py`
- Modify: `math_eval/test_collect_prismatic_gradients.py`

**Interfaces:**
- Produces: `build_eligibility_report(rows, prepared_manifest, tokenizer, max_context_tokens, max_excluded, expected_excluded_ids) -> dict`
- Produces: `write_or_validate_eligibility_report(path: Path, expected: Mapping) -> dict`
- Produces: `apply_eligibility_report(rows, prepared_manifest, report_path: Path) -> tuple[list[dict], dict]`
- CLI inputs: `--prepared-jsonl`, `--prepared-manifest`, `--output`, `--model-name`, `--model-revision`, `--max-context-tokens`, `--max-excluded`, repeatable `--expected-excluded-id`

- [ ] **Step 1: Write failing tests for deterministic token eligibility**

Use a recording fake tokenizer with batched Qwen chat-template calls. Assert exact token counts, source-order preservation, a stable-ID gap after excluding a middle row, no truncation argument, the pinned 32,768 boundary, at most one excluded row, and deterministic eligible-ID hashing. Assert the report binds prepared hashes, model/tokenizer revision, Transformers version, and chat-template hash.

- [ ] **Step 2: Run eligibility tests and verify RED**

Run:

```bash
/home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest -q \
  math_eval/test_build_gradient_eligibility.py
```

Expected: module import fails because the eligibility builder does not exist.

- [ ] **Step 3: Implement atomic report creation and pinned CLI**

Load the already validated prepared pool, `AutoConfig`, and `AutoTokenizer` at the exact model revision. Require `config.max_position_embeddings == 32768`; apply the exact user/assistant chat template in bounded batches with `truncation=False`; retain no encoded batch after counting. Require the discovered excluded stable-ID set to equal the explicitly configured production set. Write one canonical JSON report through a sibling temporary file and `Path.replace`, using a persistent lock for concurrent/idempotent creation. Existing reports are reused only after full provenance validation.

- [ ] **Step 4: Run eligibility tests and verify GREEN**

Expected: all builder tests pass offline with no model or CUDA allocation.

- [ ] **Step 5: Write failing collector tests for applying the report**

Assert that the collector derives ordered eligible rows from the full prepared list without renumbering IDs, rejects a changed prepared hash/model revision/context boundary/count/excluded ID/token reason/eligible hash, records the eligibility report hash in the gradient manifest, and computes logical shards over the eligible count. Direct collection without a report retains the existing hard-fail behavior.

- [ ] **Step 6: Implement eligible-ID collection and run Task 3 regressions**

Filter only by the validated excluded stable-ID set, then pass the resulting ordered eligible rows to existing sharding, resume, preflight, and official collection logic. Do not interpret eligible positions as source indices. Run Task 3 and Task 3B tests together.

- [ ] **Step 7: Commit Task 3B**

```bash
git add math_eval/build_gradient_eligibility.py \
  math_eval/test_build_gradient_eligibility.py \
  math_eval/collect_prismatic_gradients.py \
  math_eval/test_collect_prismatic_gradients.py
git commit -m "Exclude unsupported proxy-context rows explicitly"
```

---

### Task 4: Code-/paper-aligned clustering diagnostics and parquet writer

**Files:**
- Create: `math_eval/select_gradient_diverse_deepmath.py`
- Create: `math_eval/test_select_gradient_diverse_deepmath.py`

**Interfaces:**
- Consumes: official-format safetensors chunks, prepared JSONL, eligibility report, current source parquet
- Produces: `load_projected_gradients(directory: Path, expected_ids: Sequence[str]) -> tuple[list[str], torch.Tensor]`
- Produces: `cluster_official(gradients: torch.Tensor, ratio: float, iterations: int, seed: int, reference_repo: Path) -> np.ndarray`
- Produces: `compute_selection_diagnostics(...) -> dict`
- Produces: `write_selected_parquet(source: pa.Table, selected_source_indices: Sequence[int], output: Path) -> None`
- CLI produces `diagnostics.json`, `selected_ids.jsonl`, final parquet, and final manifest.

- [ ] **Step 1: Write failing tests for safetensors coverage and schema-preserving output**

Create two safetensors chunks whose keys are deliberately unordered and whose eligible stable IDs contain a gap for an excluded middle source row. Assert loading returns eligibility-report order and rejects missing, extra, duplicate, or excluded IDs. Construct a nested PyArrow table matching the training schema and assert selected eligible positions map through prepared metadata to original source indices; output has the identical schema including metadata, requested order, exact row count, and unique prompts.

- [ ] **Step 2: Run the selector tests and verify RED**

Run:

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  math_eval/test_select_gradient_diverse_deepmath.py
```

Expected: selector module is missing.

- [ ] **Step 3: Implement gradient loading and atomic schema-preserving parquet output**

Use `safetensors.torch.load_file`, validate through `validate_gradient_matrix`, use `pyarrow.Table.take`, compare `selected.schema == source.schema`, write to a temporary path, read it back, validate again, then atomically replace.

- [ ] **Step 4: Run tests and verify GREEN**

Expected: coverage and Arrow tests pass.

- [ ] **Step 5: Write failing deterministic-clustering adapter tests**

Use a fake reference `ClusterManager` injected as a dependency so the CPU test can assert that:

- input is L2-normalized
- the deterministic permutation is inverted correctly
- `K = floor(ratio * N)` with a minimum of two
- exactly 20 iterations are requested
- seed 42 and seed 43 produce separate diagnostics

- [ ] **Step 6: Run clustering tests and verify RED**

Expected: failures identify the missing adapter and diagnostics functions.

- [ ] **Step 7: Implement the official clustering adapter and diagnostics**

Import `ClusterManager` from the verified reference repository. Compute adjusted Rand index with scikit-learn. Compute topic entropy from selected metadata. Compute G-Vendi using the reference `Vendi.compute_vendi_score` on the normalized selected gradient covariance; do not reimplement entropy.

- [ ] **Step 8: Run all selector tests and verify GREEN**

Expected: all Task 4 tests pass.

- [ ] **Step 9: Commit Task 4**

```bash
git add math_eval/select_gradient_diverse_deepmath.py \
  math_eval/test_select_gradient_diverse_deepmath.py
git commit -m "Select exact gradient-diverse DeepMath coreset"
```

---

### Task 5: Resumable production launcher and validation gates

**Files:**
- Create: `run_select_gradient_diverse_deepmath.sh`
- Create: `math_eval/test_gradient_diversity_launcher.py`

**Interfaces:**
- Environment: `GPU_IDS`, `PYTHON_BIN`, `REFERENCE_REPO`, `SOURCE_PARQUET`, `OUTPUT_ROOT`, `EXPECTED_SOURCE_ROWS`, `EXPECTED_ELIGIBLE_ROWS`, `EXPECTED_EXCLUDED_ID`, `MAX_CONTEXT_TOKENS`, `TARGET_ROWS`, `MODEL_NAME`, `MODEL_REVISION`, `DATASET_REVISION`, `PRIMARY_CLUSTER_RATIO`, `SENSITIVITY_CLUSTER_RATIO`, `CLUSTER_SEEDS`
- Produces: one master log and the artifact tree from the spec.

- [ ] **Step 1: Write a failing static launcher contract test**

Parse the shell script text and assert it contains exact pinned defaults, quotes all path variables, runs and validates eligibility before any collector, derives logical shard indices independently from physical `GPU_IDS`, uses `set -euo pipefail`, writes a PID lock, waits for every collector PID, and invokes selection only after the exact eligible-gradient-ID validation command succeeds.

- [ ] **Step 2: Run launcher test and verify RED**

Run:

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  math_eval/test_gradient_diversity_launcher.py
```

Expected: fails because the launcher does not exist.

- [ ] **Step 3: Implement the minimal resumable launcher**

Default outputs:

```text
data/gradient_diversity/deepmath_level6_r1_solution1.jsonl
data/gradient_diversity/deepmath_level6_r1_solution1.eligibility.json
data/gradient_diversity/gradients/qwen2.5-0.5b-instruct
data/gradient_diversity/DeepMath-103K/train_gradient_diverse_12800.parquet
logs/gradient_diversity/deepmath_gradient_diverse_12800.log
```

The launcher must echo every resolved input and revision, reject an empty GPU list, create a lock with `flock -n`, and install no packages automatically. It invokes Python CLIs with `-m math_eval...` from the repository root. Eligibility must report exactly 57,045 rows and only `deepmath-level6-038794` excluded before collection. Each collector command exposes exactly one physical GPU through `CUDA_VISIBLE_DEVICES=<physical_id>`, passes `--device cuda:0`, passes the eligibility report, and passes its separately enumerated logical `--shard-index`.

- [ ] **Step 4: Run launcher test and verify GREEN**

Expected: launcher contract passes.

- [ ] **Step 5: Run the complete CPU test suite for the feature**

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  math_eval/test_deepmath_gradient_diversity.py \
  math_eval/test_prepare_deepmath_gradient_pool.py \
  math_eval/test_build_gradient_eligibility.py \
  math_eval/test_collect_prismatic_gradients.py \
  math_eval/test_select_gradient_diverse_deepmath.py \
  math_eval/test_gradient_diversity_launcher.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 5**

```bash
git add run_select_gradient_diverse_deepmath.sh \
  math_eval/test_gradient_diversity_launcher.py
git commit -m "Add gradient-diverse data selection launcher"
```

---

### Task 6: Real-data preflight, GPU smoke, review, and `opd-CLI` launch

**Files:**
- Runtime only under ignored `data/gradient_diversity/` and `logs/gradient_diversity/`
- No production source edits unless a failing test is added first.

**Interfaces:**
- Consumes all prior CLIs and launcher.
- Produces a live resumable selection process in `opd-CLI`.

- [ ] **Step 1: Run the focused existing OPD regression suite**

```bash
PYTHONPATH="$PWD/verl" /home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  verl/tests/trainer/config/test_algo_config_on_cpu.py \
  verl/tests/trainer/ppo/test_difficulty_aware_opd.py \
  verl/tests/trainer/ppo/test_tale_budget.py \
  verl/tests/trainer/ppo/test_group_success_launcher.py
```

Expected: 51 tests pass.

- [ ] **Step 2: Run real preparation only**

```bash
/home/mchen/miniconda3/envs/gvendi-opd/bin/python \
  -m math_eval.prepare_deepmath_gradient_pool \
  --source-parquet data/g-opd/DeepMath-103K/train_filtered_level6.parquet \
  --output-jsonl data/gradient_diversity/deepmath_level6_r1_solution1.jsonl \
  --manifest data/gradient_diversity/deepmath_level6_r1_solution1.manifest.json \
  --expected-count 57046
```

Expected: exact 57,046/57,046 join, no ambiguity, pinned revision recorded.

- [ ] **Step 3: Run and validate the real eligibility report**

Expected: 57,046 prepared rows, exactly one excluded row (`deepmath-level6-038794`, 33,634 tokens), 57,045 eligible rows, and unchanged stable IDs around the exclusion gap.

- [ ] **Step 4: Run four-sample one-GPU gradient smoke**

Use a free GPU identified immediately before the command. Restrict the smoke input to four copied prepared rows in a temporary ignored JSONL. Expected artifacts: one safetensors chunk with four IDs and tensor shape `(4, 1024)`; all entries finite and non-zero.

- [ ] **Step 5: Verify smoke resume**

Rerun the same four-sample command. Expected: the wrapper detects complete coverage and exits without writing or overwriting another chunk.

- [ ] **Step 6: Run tiny GPU clustering smoke**

Run selection against a fixture target smaller than four and both ratios with the implementation's minimum `K=2`. Expected: diagnostics and schema-preserving parquet are written without traceback.

- [ ] **Step 7: Request code review and address only verified findings through TDD**

Review the committed diff against the spec, official reference behavior, resume safety, and test coverage. Any fix starts with a failing regression test.

- [ ] **Step 8: Run final verification**

Run the feature suite, the 51-test OPD suite, `bash -n run_select_gradient_diverse_deepmath.sh`, `git diff --check`, and confirm the worktree is clean except ignored runtime artifacts.

- [ ] **Step 9: Integrate the reviewed commits into the main branch without touching unrelated dirty files**

Use non-destructive cherry-picks of this branch's commits into `quality-gated-correct-compression-opd`. Stop and report if any path overlaps user modifications.

- [ ] **Step 10: Launch in existing `opd-CLI`**

Inspect the tmux panes and active GPU processes again. Open a new pane/window in `opd-CLI` if necessary, then run:

```bash
GPU_IDS=<verified-free-comma-separated-ids> \
  bash run_select_gradient_diverse_deepmath.sh
```

Expected before handoff: preparation succeeds, reference/model revisions appear in the master log, at least one gradient chunk is written, and all collector processes remain alive without traceback.

- [ ] **Step 11: Record launch evidence**

Report the tmux target, process IDs, physical GPU IDs, master log, prepared row count, gradient chunk count, and exact resume command.
