# Paired Normal/Concise Compression-Sensitivity Probe Implementation Plan

> **Runtime amendment:** Task 2's independent relaxed resampling is superseded by `2026-07-19-paired-normal-concise-forced-prefix-remediation.md` after TP=4 vLLM disproved the assumed same-seed prefix identity.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible paired normal/concise training-distribution probe on the base Qwen3-4B and completed adaptive step-50 checkpoint, including all four correctness quadrants and same-seed relaxed-budget counterfactuals.

**Architecture:** A CPU-testable core module owns sampling, prompt construction, route classification, vLLM record normalization, and summary aggregation. A separate validator independently recomputes artifact invariants. A fail-closed launcher prepares one immutable sample, runs the two TP=4 model jobs sequentially in `opd-CLI`, analyzes their outputs, and writes a completion marker.

**Tech Stack:** Python 3.10, PyArrow, NumPy, Transformers, vLLM, vendored VERL math reward helpers, pytest, Ruff, Bash, tmux, four allocation-owned GPUs.

## Global Constraints

- Work only in `/home/mchen/FiRe-OPD/.worktrees/vanilla-sft-gradient-51200-opd`; do not modify the dirty root checkout.
- Use `/home/mchen/miniconda3/envs/gvendi-opd/bin/python` for CPU preparation/analysis and `/home/mchen/miniconda3/envs/verl/bin/python` for vLLM generation.
- Use `PYTHONPATH="$PWD/verl:$PWD"` whenever importing vendored VERL.
- Dataset is `/home/mchen/FiRe-OPD/data/g-opd/DeepMath-103K/train_filtered_level6.parquet` with SHA256 `de3350fdd00bc0410550098ea65179e2be873da99e4075f80de575fc17670597`.
- Select exactly 128 distinct rows with seed 42 and use the identical persisted sample for both models.
- Use model label/path `base=/home/mchen/FiRe-OPD/models/Qwen3-4B` and `adaptive_step50=/home/mchen/FiRe-OPD/checkpoints/opd-adaptive-conciseprobe-cap50-tokenneutral-rawprompt-4gpu-tp4-step50/global_step_50_hf`, sequentially and read-only.
- Generate one normal and one capped-concise response per row with temperature/top-p `1.0`, `enable_thinking=False`, and stable per-row seeds.
- Normal generation uses `max_tokens=16384`; concise cap is `max(1, floor(0.5 * normal_generated_length))`.
- Generate concise responses for all rows, including normal-wrong rows.
- Relax only capped `normal-correct / concise-wrong` cases to the corresponding normal length with the same seed, and require exact token-prefix preservation.
- GPU work runs in tmux `opd-CLI` on allocation-owned CUDA tokens `0,1,2,3`, TP=4, max model length 40,960, without nested `srun`.
- Never overwrite or resume an existing runtime namespace.
- Do not train, update weights, launch W&B, or run canonical benchmark evaluation.

---

### Task 1: Add deterministic sampling, prompt, routing, and counterfactual contracts

**Files:**
- Create: `math_eval/paired_normal_concise_probe.py`
- Create: `math_eval/test_paired_normal_concise_probe.py`

**Interfaces:**
- Produces: `select_probe_rows(rows, sample_size, seed) -> list[dict]`.
- Produces: `normal_messages(row) -> list[dict[str, str]]`.
- Produces: `concise_messages(row) -> list[dict[str, str]]`.
- Produces: `concise_cap(normal_length) -> int`.
- Produces: `quadrant(normal_correct, concise_correct) -> str`.
- Produces: `classify_compression_sensitive(record) -> str | None`.
- Produces: `validate_relaxed_prefix(capped_ids, relaxed_ids) -> None`.

- [ ] **Step 1: Write failing unit tests for deterministic sampling and prompt construction**

Add tests which create synthetic parquet-style rows and assert:

```python
first = select_probe_rows(rows, sample_size=4, seed=42)
assert first == select_probe_rows(rows, sample_size=4, seed=42)
assert len({row["source_row_index"] for row in first}) == 4
assert [row["source_row_index"] for row in first] != [
    row["source_row_index"] for row in select_probe_rows(rows, sample_size=4, seed=43)
]
assert normal_messages(first[0]) == first[0]["prompt"]
assert "Solve concisely. Avoid unnecessary explanation." in concise_messages(first[0])[0]["content"]
assert "Please reason step by step" not in concise_messages(first[0])[0]["content"]
```

Also assert invalid sizes, duplicate identities, malformed prompt messages, and missing ground truths raise `ValueError`.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
PYTHONPATH="$PWD/verl:$PWD" /home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest \
  math_eval/test_paired_normal_concise_probe.py -q
```

Expected: collection fails because `math_eval.paired_normal_concise_probe` does not exist.

- [ ] **Step 3: Implement deterministic sample and prompt helpers**

Implement validation and sampling with `np.random.default_rng(seed).choice(..., replace=False)`, persist `sample_ordinal`, `source_row_index`, `question_id`, `request_seed=42+source_row_index`, prompt, and ground truth. Reuse `build_concise_teacher_messages` from `verl.trainer.ppo.tale_budget` so the prompt exactly matches adaptive training.

- [ ] **Step 4: Add failing routing and counterfactual tests**

Assert:

```python
assert concise_cap(1) == 1
assert concise_cap(9) == 4
assert quadrant(True, True) == "compression_safe"
assert quadrant(True, False) == "compression_sensitive"
assert quadrant(False, True) == "concise_rescued"
assert quadrant(False, False) == "both_wrong"
validate_relaxed_prefix([1, 2], [1, 2, 3])
with pytest.raises(ValueError, match="prefix"):
    validate_relaxed_prefix([1, 9], [1, 2, 3])
```

Create synthetic records for `budget_limited_recovered`, `budget_limited_unrecovered`, and `prompt_or_sampling_failure`.

- [ ] **Step 5: Run the new tests and verify RED**

Expected: failures identify the missing routing/counterfactual functions.

- [ ] **Step 6: Implement minimal routing and counterfactual helpers**

Require positive integer lengths, boolean correctness flags, exact capped-token prefix, and relaxed generation only when `quadrant == "compression_sensitive"` and `concise.cap_hit is True`.

- [ ] **Step 7: Run the focused tests and verify GREEN**

Run the Task 1 pytest command. Expected: all tests pass.

- [ ] **Step 8: Commit Task 1**

```bash
git add math_eval/paired_normal_concise_probe.py math_eval/test_paired_normal_concise_probe.py
git diff --check
git commit -m "feat: add paired compression probe contracts"
```

---

### Task 2: Add immutable preparation and per-model vLLM generation

**Files:**
- Modify: `math_eval/paired_normal_concise_probe.py`
- Modify: `math_eval/test_paired_normal_concise_probe.py`

**Interfaces:**
- Produces CLI subcommands:
  - `prepare --dataset --sample-file --manifest --sample-size --seed --source-commit`.
  - `generate --sample-file --model-path --model-label --output-dir`.
- Produces: `<model>/records.jsonl` containing exactly 128 complete model records.

- [ ] **Step 1: Write failing tests for preparation and normalized generation records**

Use a temporary parquet file to verify `prepare` writes exactly four deterministic sample rows and a manifest containing dataset hash, source commit, sample hash, and frozen protocol. Use simple fake output objects to verify response normalization records token IDs directly, preserves finish reason, extracts the boxed answer, and calls the training-compatible math reward.

Test that an existing sample/manifest path, wrong dataset hash, missing model directory, or output directory already existing fails before generation.

- [ ] **Step 2: Run focused tests and verify RED**

Expected: failures identify missing preparation, record normalization, and CLI behavior.

- [ ] **Step 3: Implement the `prepare` subcommand and atomic JSON/JSONL writes**

Use PyArrow to read the parquet file, verify its SHA256, select rows, and write via temporary siblings followed by `os.replace`. Refuse any pre-existing destination.

- [ ] **Step 4: Implement lazy vLLM generation**

Import `torch`, `transformers.AutoTokenizer`, `vllm.LLM`, and `vllm.SamplingParams` only inside the generation path. Initialize:

```python
LLM(
    model=model_path,
    tokenizer=model_path,
    tensor_parallel_size=4,
    max_model_len=40960,
    max_num_seqs=128,
    gpu_memory_utilization=0.90,
    enforce_eager=True,
)
```

Render all prompts with `enable_thinking=False`. Use one `SamplingParams` per row so `seed=request_seed` and per-row concise caps are explicit. Require one output per request.

- [ ] **Step 5: Implement relaxed concise counterfactual generation**

After scoring paired records, select only capped `compression_sensitive` cases, regenerate concise prompts with `max_tokens=normal.length`, validate exact capped-token prefixes, and attach the relaxed result and mechanical class. All noneligible records carry `relaxed_concise=null`.

- [ ] **Step 6: Verify GREEN and commit Task 2**

Run:

```bash
PYTHONPATH="$PWD/verl:$PWD" /home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest \
  math_eval/test_paired_normal_concise_probe.py -q
/home/mchen/miniconda3/envs/verl/bin/python -m compileall -q math_eval/paired_normal_concise_probe.py
```

Then commit:

```bash
git add math_eval/paired_normal_concise_probe.py math_eval/test_paired_normal_concise_probe.py
git commit -m "feat: generate paired normal concise probes"
```

---

### Task 3: Add summaries, qualitative case reports, comparison, and independent validation

**Files:**
- Modify: `math_eval/paired_normal_concise_probe.py`
- Create: `math_eval/validate_paired_normal_concise_probe.py`
- Modify: `math_eval/test_paired_normal_concise_probe.py`
- Create: `math_eval/test_validate_paired_normal_concise_probe.py`

**Interfaces:**
- Produces CLI subcommand `analyze --run-dir --expected-count 128`.
- Produces model `summary.json` and `compression_sensitive_cases.md`.
- Produces root `comparison.json` and `comparison.md`.
- Validator consumes the complete run directory and prints `PAIRED_NORMAL_CONCISE_PROBE_GATE=PASS` only after independent recomputation.

- [ ] **Step 1: Write failing aggregation tests**

Construct four synthetic rows, one per quadrant, and assert exact counts, accuracies, paired delta, length statistics, cap-hit counts, parse failures, and failure-class counts. Assert all summary values are finite and quadrant counts sum to the expected row count.

- [ ] **Step 2: Run and verify RED**

Expected: failures identify missing summary and report functions.

- [ ] **Step 3: Implement summary and Markdown report generation**

Use standard-library `statistics` for means/medians. Case Markdown must list every compression-sensitive case with source ID, lengths, cap, finish reasons, boxed answers, relaxed status, mechanical class, and bounded excerpts; full text remains in JSONL.

- [ ] **Step 4: Write failing independent-validator tests**

Create a complete temporary run tree and assert the validator accepts it. Mutate, one at a time, a cap, seed, prefix, route, summary count, sample identity, and cross-model sample ordering; each mutation must raise a specific `ValueError`.

- [ ] **Step 5: Run and verify RED**

Expected: import or missing-function failures for the validator.

- [ ] **Step 6: Implement independent validation**

The validator must read JSONL independently, recompute all caps/routes/counts/metrics, compare persisted summaries, verify both models share the exact sample identities and seeds, verify required relaxed prefixes, and verify SHA256 entries in `manifest.json` after analysis updates them.

- [ ] **Step 7: Run all probe tests and verify GREEN**

```bash
PYTHONPATH="$PWD/verl:$PWD" /home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest -q \
  math_eval/test_paired_normal_concise_probe.py \
  math_eval/test_validate_paired_normal_concise_probe.py
```

- [ ] **Step 8: Commit Task 3**

```bash
git add math_eval/paired_normal_concise_probe.py \
  math_eval/validate_paired_normal_concise_probe.py \
  math_eval/test_paired_normal_concise_probe.py \
  math_eval/test_validate_paired_normal_concise_probe.py
git diff --check
git commit -m "feat: analyze and validate paired compression probes"
```

---

### Task 4: Add a pinned, fail-closed sequential launcher

**Files:**
- Create: `run_paired_normal_concise_probe.sh`
- Create: `math_eval/test_paired_normal_concise_probe_launcher.py`

**Interfaces:**
- Consumes the Python CLIs from Tasks 2-3.
- Produces one immutable root run namespace and a final `PAIRED_NORMAL_CONCISE_PROBE_DONE_<id>:<exit>` marker.
- `PAIRED_PROBE_DRY_RUN=1` prints all commands without creating output or invoking GPUs.

- [ ] **Step 1: Write failing launcher contract tests**

Assert dry-run output pins dataset, dataset hash, both model paths, sample size/seed, TP=4 behavior, max model length 40,960, CUDA tokens `0,1,2,3`, sequential base-before-adaptive ordering, analyzer, validator, and marker. Assert source text contains no `srun`, W&B, resume, or overwrite path.

- [ ] **Step 2: Run launcher tests and verify RED**

```bash
PYTHONPATH="$PWD/verl:$PWD" /home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest \
  math_eval/test_paired_normal_concise_probe_launcher.py -q
```

Expected: failure because the launcher does not exist.

- [ ] **Step 3: Implement the launcher**

The launcher must use `set -euo pipefail`, resolve the worktree path, refuse an existing run/log namespace, validate source/data/model files, require job 429 to be running in non-dry mode, require `CUDA_VISIBLE_DEVICES=0,1,2,3`, run prepare, base generation, adaptive generation, analysis, and validation sequentially, and preserve nonzero exit markers via an `EXIT` trap.

The launcher must not kill GPU processes. GPU ownership/idleness is an external pre-launch gate run from the allocation shell.

- [ ] **Step 4: Verify launcher GREEN**

Run launcher pytest plus:

```bash
bash -n run_paired_normal_concise_probe.sh
PAIRED_PROBE_DRY_RUN=1 PAIRED_PROBE_RUN_ID=test-dry-run \
  bash run_paired_normal_concise_probe.sh
```

Expected: syntax pass, pinned command trace, no created runtime namespace.

- [ ] **Step 5: Commit Task 4**

```bash
git add run_paired_normal_concise_probe.sh \
  math_eval/test_paired_normal_concise_probe_launcher.py
git diff --check
git commit -m "feat: launch paired compression sensitivity probe"
```

---

### Task 5: Run CPU/static gates, launch in `opd-CLI`, and audit results

**Files:**
- Runtime only under `/home/mchen/FiRe-OPD/math_eval/paired_normal_concise_probe_outputs/<run-id>/`.
- Runtime log under `/home/mchen/FiRe-OPD/math_eval/paired_normal_concise_probe_logs/`.

**Interfaces:**
- Consumes committed clean source from Tasks 1-4.
- Produces independently validated base/adaptive records and comparison artifacts.

- [ ] **Step 1: Run complete CPU and static gates**

```bash
export PYTHONPATH="$PWD/verl:$PWD"
/home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest -q \
  math_eval/test_paired_normal_concise_probe.py \
  math_eval/test_validate_paired_normal_concise_probe.py \
  math_eval/test_paired_normal_concise_probe_launcher.py
/home/mchen/miniconda3/envs/verl/bin/python -m ruff check \
  math_eval/paired_normal_concise_probe.py \
  math_eval/validate_paired_normal_concise_probe.py \
  math_eval/test_paired_normal_concise_probe.py \
  math_eval/test_validate_paired_normal_concise_probe.py \
  math_eval/test_paired_normal_concise_probe_launcher.py
/home/mchen/miniconda3/envs/verl/bin/python -m compileall -q \
  math_eval/paired_normal_concise_probe.py \
  math_eval/validate_paired_normal_concise_probe.py
bash -n run_paired_normal_concise_probe.sh
git diff --check
test -z "$(git status --short)"
```

- [ ] **Step 2: Run explicit allocation/GPU gate**

Verify job 429 is running. Map every GPU process to a physical GPU index and owner; require allocation tokens `0,1,2,3` to be idle. Do not disturb processes on other GPU indices.

- [ ] **Step 3: Launch one new namespace in `opd-CLI`**

Generate a UTC run ID containing source SHA, then send one shell command to `opd-CLI` that exports `CUDA_VISIBLE_DEVICES=0,1,2,3`, executes the launcher, and tees the immutable log. Do not use nested `srun`.

- [ ] **Step 4: Monitor conditionally**

Poll for the completion marker, process exit, or fatal log signatures. Do not relaunch unless the marker is nonzero or a fatal failure is proven.

- [ ] **Step 5: Run fresh independent validation and hashes**

After an exit-zero marker, rerun the validator from the analysis environment, count JSONL rows, recompute metrics, and SHA256 all manifests, records, summaries, Markdown reports, and the launch log.

- [ ] **Step 6: Inspect concrete paired cases**

Read all `compression_sensitive` records and representative `concise_rescued` records. Separate cap-recovered, cap-unrecovered, natural-stop wrong, parse failure, arithmetic omission, strategy change, and unfinished calculation patterns using evidence from response text. Do not infer categories that the text does not support.

- [ ] **Step 7: Report results**

Report both four-quadrant tables, model-to-model changes, compression-sensitive counterfactual counts, length/cap statistics, and concrete examples. Explicitly answer whether failures are predominantly consistent with insufficient scratch space and whether `normal-wrong / concise-correct` exists in the sample.
