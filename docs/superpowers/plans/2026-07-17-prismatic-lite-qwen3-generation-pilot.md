# Prismatic-Lite Qwen3 Generation Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a fail-closed 256-question solution-style calibration followed, only on calibration pass, by an immutable 2,000-question Qwen3 Prismatic-lite generation and sparse-cluster filtering pilot.

**Architecture:** Keep all scientific transformations in small pure modules with strict schemas and atomic artifacts. Use a phase-based CLI to separate CPU preparation, TP=4 Qwen3 generation, single-GPU Qwen2.5 projected-gradient shards, and GPU/CPU analysis; a shell launcher enforces clean-source, Slurm, GPU, hash, target-absence, and process-drain contracts. Reuse frozen DeepMath gradients without recomputation and use the pinned official gradient computer only for newly generated completions.

**Tech Stack:** Python 3.10, PyTorch, vLLM 0.8.5, Transformers, `math_verify`, safetensors, NumPy, scikit-learn, Qwen3-30B-A3B-Instruct-2507, cached Qwen2.5-0.5B-Instruct, Bash, tmux `opd-CLI`, Slurm.

## Global Constraints

- Work only in `/home/mchen/FiRe-OPD/.worktrees/vanilla-sft-gradient-51200-opd`; never modify the dirty root source checkout.
- Preserve all 57,046 source rows and all 57,045 frozen projected gradients byte-for-byte.
- Publish only under `/home/mchen/FiRe-OPD/data/prismatic_lite/qwen3_2k_pilot` and `/home/mchen/FiRe-OPD/logs/prismatic_lite/qwen3_2k_pilot`.
- Resume, overwrite, implicit retry, altered membership, and nested `srun` are forbidden.
- Run Qwen3 only as one four-GPU TP=4 engine; run each Qwen2.5 gradient shard with exactly one visible allocation token.
- Use exactly three independent solutions and an integer majority threshold of two.
- Primary clustering is cosine K-means K=570, 20 iterations, seed 42; seed 43 is a stability gate. K=285 and K=1,140 are diagnostics only.
- Calibration failure is a declared stop condition: archive evidence, publish a `no_go` calibration report, launch no 2K generation, and report to the user.
- Any insufficient-wall-time, occupied-GPU, source/hash drift, malformed/partial target, fatal model error, or threshold failure is a declared stop condition.
- No OPD/SFT training and no downstream efficacy claim.

---

### Task 1: Immutable Artifact and Schema Layer

**Files:**
- Create: `math_eval/prismatic_lite_pilot_artifacts.py`
- Create: `math_eval/test_prismatic_lite_pilot_artifacts.py`

**Interfaces:**
- Produces `PilotPaths`, `strict_jsonl`, `atomic_publish_json`, `atomic_publish_jsonl`, `atomic_publish_npz`, `write_or_validate_manifest`, `validate_exact_fields`, and `artifact_record`.
- All later tasks consume these APIs; no later module writes canonical files directly.

- [ ] **Step 1: Write failing tests for strict JSON and collision refusal**

Test duplicate JSON keys, NaN/Infinity, blank JSONL lines, duplicate IDs, missing exact fields, an existing canonical target, and a parent directory containing a partial artifact without its manifest. Require errors to name the offending path/field.

Run:

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m pytest \
  math_eval/test_prismatic_lite_pilot_artifacts.py -q
```

Expected: FAIL because `math_eval.prismatic_lite_pilot_artifacts` does not exist.

- [ ] **Step 2: Implement the minimal immutable artifact API**

Use canonical UTF-8 JSON (`sort_keys=True`, compact separators, `allow_nan=False`), same-directory temporary files, file and directory `fsync`, hard collision refusal, strict object-pair hooks, SHA-256 streaming, exact JSONL row validation, and non-pickle NumPy NPZ publication with sorted exact keys. `PilotPaths.from_root(root)` must expose every path named in the approved design.

- [ ] **Step 3: Run artifact tests and existing artifact regression tests**

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m pytest \
  math_eval/test_prismatic_lite_pilot_artifacts.py \
  math_eval/test_opd_proxy_gradient_verify_artifacts.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add math_eval/prismatic_lite_pilot_artifacts.py \
        math_eval/test_prismatic_lite_pilot_artifacts.py
git commit -m "feat: add Prismatic-lite pilot artifacts"
```

### Task 2: Deterministic Sampling, Prompting, and Quality Logic

**Files:**
- Create: `math_eval/prismatic_lite_pilot_generation.py`
- Create: `math_eval/test_prismatic_lite_pilot_generation.py`

**Interfaces:**
- Produces `stratified_calibration_sample(rows, size, seed)`, `difficulty_weighted_fewshots(rows, requests, width, seed)`, `problem_prompt(examples)`, `solution_messages(problem)`, `parse_generated_problems(text)`, `parse_final_answer(text)`, `majority_group(responses)`, `question_level_completion_rows(record)`, `normalized_tokens(text)`, `ten_grams(text)`, and `NearDuplicateIndex`.
- Runtime generation consumes a backend protocol `generate(prompts, sampling_params) -> list[list[str]]`, allowing CPU tests without importing vLLM.

- [ ] **Step 1: Write failing tests for deterministic membership and prompt parsing**

Cover exact 256 stratified unique rows, repeatability, changed-seed sensitivity, five distinct parents per request, deterministic candidate IDs, rejection of malformed problem boundaries, and exact preservation of parent IDs.

Expected initial command:

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m pytest \
  math_eval/test_prismatic_lite_pilot_generation.py -q
```

Expected: FAIL on missing module.

- [ ] **Step 2: Implement sampling and prompt construction**

Calibration stratification must use exact `(topic, difficulty)` cells, allocate quotas by largest remainder, and choose rows with `numpy.random.Generator(PCG64(seed))`. Few-shot sampling uses probabilities proportional to positive difficulty, no replacement within a request, and one child `SeedSequence` per request. Prompts match the paper's “similar or harder, novel, non-multiple-choice” intent and request one `[[Problem]]...---` block.

- [ ] **Step 3: Write failing tests for answer majority and lexical quality gates**

Use real `math_verify` on equivalent fractions/LaTeX, three-way disagreement, null answers, exact integer threshold two, normalized exact duplicates, 10-gram Jaccard threshold 0.30, and benchmark any-10-gram contamination.

- [ ] **Step 4: Implement answer and quality logic**

Return all majority-matching response indices; never accept three distinct one-vote answers. Unicode-normalize NFKC, lowercase, split words/math tokens deterministically, and use an inverted index rather than all-pairs scans. Semantic judgments must parse only exact `equivalent` or `not_equivalent`.

- [ ] **Step 5: Test and commit**

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m pytest \
  math_eval/test_prismatic_lite_pilot_generation.py -q
git add math_eval/prismatic_lite_pilot_generation.py \
        math_eval/test_prismatic_lite_pilot_generation.py
git commit -m "feat: add Prismatic-lite generation contracts"
```

### Task 3: Sparse-Cluster and Calibration Analysis

**Files:**
- Create: `math_eval/prismatic_lite_pilot_analysis.py`
- Create: `math_eval/test_prismatic_lite_pilot_analysis.py`

**Interfaces:**
- Produces `question_gradient(vectors)`, `smallest_cluster_ids(labels, count)`, `assign_to_centroids(vectors, centroids)`, `paired_calibration_metrics`, `stratified_pairing_null`, `calibration_decision`, `candidate_selection`, `selection_null`, and `final_pilot_decision`.
- Uses the official cosine K-means wrapper already validated in `math_eval.select_gradient_diverse_deepmath.cluster_official`.

- [ ] **Step 1: Write failing tests for question-level gradient averaging**

Require per-solution normalization, arithmetic mean, final normalization, finite/nonzero checks, majority-index coverage, and invariance to majority-solution order.

- [ ] **Step 2: Implement minimal gradient aggregation**

Use float64 accumulation and publish float32 normalized vectors. Reject duplicate/missing solution IDs and vectors whose norm is zero or nonfinite.

- [ ] **Step 3: Write failing tests for exact sparse membership and assignments**

Fixtures must test exactly half the clusters, deterministic `(cluster_size, cluster_id)` tie breaks, cosine nearest-centroid assignment, seed-42 primary membership, and seed-43 diagnostics.

- [ ] **Step 4: Implement clustering helpers and calibration metrics**

Persist labels, normalized centroids, cluster sizes, sparse IDs, K, iterations, seed, input IDs hash, and gradient hash. Stratified null generation uses 10,000 child RNG streams and the exact fallback specified in the design.

- [ ] **Step 5: Write failing tests for decision boundaries**

Test inclusive thresholds at 192 rows, 0.65 agreement, +0.15 null margin, 90th percentile paired cosine, 1,000 quality rows, 400 sparse rows, 0.65 seed agreement, and 90th G-Vendi null percentile.

- [ ] **Step 6: Implement decisions, run tests, and commit**

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m pytest \
  math_eval/test_prismatic_lite_pilot_analysis.py -q
git add math_eval/prismatic_lite_pilot_analysis.py \
        math_eval/test_prismatic_lite_pilot_analysis.py
git commit -m "feat: analyze Prismatic-lite sparse gradients"
```

### Task 4: New-Completion Projected Gradient Collector

**Files:**
- Create: `math_eval/collect_prismatic_pilot_gradients.py`
- Create: `math_eval/test_collect_prismatic_pilot_gradients.py`

**Interfaces:**
- CLI consumes immutable JSONL rows with exact fields `id`, `question_id`, `solution_index`, `prompt`, `completion` and an input manifest SHA.
- Produces four-shard-compatible chunk pairs and `gradient.manifest.json` under a phase-owned directory.
- Reuses pinned constants and `construct_strict_collector` from `collect_prismatic_gradients.py` without changing old collector behavior.

- [ ] **Step 1: Write failing tests for arbitrary stable IDs and manifests**

Require exact input order, unique IDs, one visible `cuda:0`, pinned model/revision/projector, 1,024 float32 vectors, four nonoverlapping shard ranges, global coverage, collision refusal, and rejection of the old frozen gradient directory as an output.

- [ ] **Step 2: Implement the strict collector**

Load the verified official `GradientComputer`, format prompt/completion with the pinned Qwen2.5 chat template, enforce 32,768-token context, call official projection, and write chunks with an isolated prefix. Record input hash, source question IDs, model revision, reference commit/tree, package versions, projection contract, and exact shard bounds.

- [ ] **Step 3: Test compatibility and commit**

```bash
PYTHONPATH="$PWD:$PWD/verl" /home/mchen/miniconda3/envs/verl/bin/python -m pytest \
  math_eval/test_collect_prismatic_pilot_gradients.py \
  math_eval/test_collect_prismatic_gradients.py -q
git add math_eval/collect_prismatic_pilot_gradients.py \
        math_eval/test_collect_prismatic_pilot_gradients.py
git commit -m "feat: collect Prismatic-lite pilot gradients"
```

### Task 5: Phase-Based Pilot CLI

**Files:**
- Create: `math_eval/run_prismatic_lite_qwen3_pilot.py`
- Create: `math_eval/test_run_prismatic_lite_qwen3_pilot.py`

**Interfaces:**
- Subcommands: `prepare`, `generate-calibration-solutions`, `analyze-calibration`, `generate-problems`, `generate-candidate-solutions`, `quality-review`, `analyze-selection`, `validate`, and `estimate`.
- Internal GPU backend is loaded lazily only by generation subcommands.
- Each subcommand consumes completed prior manifests and publishes one phase completion marker.

- [ ] **Step 1: Write failing command-construction and phase-order tests**

Test dry-run command equivalence, forbidden phase skips, missing/partial prior artifacts, target collision, exact Qwen3 model path, TP=4, sampling settings, generation seed ledger, and no model import during CPU subcommands.

- [ ] **Step 2: Implement `prepare` and `estimate`**

Validate all frozen hashes and rows; build the 256-row calibration manifest; validate frozen-gradient coverage without clustering; report storage and wall-time estimates; write no candidate data.

- [ ] **Step 3: Implement Qwen3 generation backend**

Load `/home/mchen/FiRe-OPD/models/Qwen3-30B-A3B-Instruct-2507` with TP=4, bfloat16, prefix caching, and explicit max model length. Use per-request deterministic `SamplingParams`; verify one returned request/result mapping per input; fully close vLLM before phase completion.

- [ ] **Step 4: Implement calibration preparation and analysis**

Generate 3×256 raw solutions, publish majority solution rows for gradient collection, load projected vectors after external shard completion, run K=570 seed-42/43 clustering on allocation token 0, average by question, run null metrics, and publish `promising` or `no_go`. A `no_go` returns a distinct documented exit code and forbids later subcommands.

- [ ] **Step 5: Implement candidate generation and quality phases**

Reach exactly 2,000 unique parsed candidates before 3,000 requests; generate exactly three solutions each; evaluate majority, duplicate, contamination, prompt length, and Qwen3 semantic-review records; prepare majority-solution gradient rows. Preserve every rejection.

- [ ] **Step 6: Implement selection analysis and final validation**

Load new vectors, average by question, assign all K values/seeds, publish accepted/rejected membership, compute the 1,000-draw matched null, make the frozen decision, verify all hashes/counts/process ledgers, and write report JSON/Markdown plus `STAGE_COMPLETE.json` only on complete analysis.

- [ ] **Step 7: Run tests and commit**

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m pytest \
  math_eval/test_run_prismatic_lite_qwen3_pilot.py -q
git add math_eval/run_prismatic_lite_qwen3_pilot.py \
        math_eval/test_run_prismatic_lite_qwen3_pilot.py
git commit -m "feat: orchestrate Prismatic-lite Qwen3 pilot"
```

### Task 6: Fail-Closed Slurm/tmux Launcher

**Files:**
- Create: `run_prismatic_lite_qwen3_2k_pilot.sh`
- Create: `math_eval/test_prismatic_lite_qwen3_launcher.py`

**Interfaces:**
- `DRY_RUN=1` performs all CPU/source/hash/path checks and prints exact phase commands without model construction.
- Real execution requires tmux session `opd-CLI`, one allocation, tokens `0,1,2,3`, and no nested `srun`.

- [ ] **Step 1: Write failing launcher contract tests**

Require shell strict mode, worktree identity, clean source, pinned HEAD/tree manifest, Slurm/tmux check, token parser, idle-GPU gate, wall-time estimate, absent roots, phase-specific `CUDA_VISIBLE_DEVICES`, process drain between Qwen3/proxy/K-means phases, and calibration stop behavior.

- [ ] **Step 2: Implement the launcher**

Sequence:

```text
CPU prepare
Qwen3 calibration solutions on 0,1,2,3
four Qwen2.5 calibration gradient shards on one token each
calibration K-means/analysis on token 0
STOP on no_go
Qwen3 problem generation on 0,1,2,3
Qwen3 candidate solutions on 0,1,2,3
Qwen3 semantic review on 0,1,2,3 only when required
four Qwen2.5 candidate gradient shards
selection K-means/analysis on token 0
final validation and process drain
```

Write one append-only master log and explicit `PRISMATIC_LITE_DONE:<rc>` marker. Archive any failed attempt before a source-changing relaunch.

- [ ] **Step 3: Test, syntax-check, and commit**

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m pytest \
  math_eval/test_prismatic_lite_qwen3_launcher.py -q
bash -n run_prismatic_lite_qwen3_2k_pilot.sh
git add run_prismatic_lite_qwen3_2k_pilot.sh \
        math_eval/test_prismatic_lite_qwen3_launcher.py
git commit -m "feat: launch Prismatic-lite Qwen3 pilot"
```

### Task 7: CPU Verification and Review Gate

**Files:**
- Read/review all Task 1–6 files.
- Update no production source except fixes arising from tests/review.

- [ ] **Step 1: Run focused tests**

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m pytest \
  math_eval/test_prismatic_lite_pilot_artifacts.py \
  math_eval/test_prismatic_lite_pilot_generation.py \
  math_eval/test_prismatic_lite_pilot_analysis.py \
  math_eval/test_collect_prismatic_pilot_gradients.py \
  math_eval/test_run_prismatic_lite_qwen3_pilot.py \
  math_eval/test_prismatic_lite_qwen3_launcher.py -q
```

Expected: all pass.

- [ ] **Step 2: Run relevant regressions and static checks**

Run existing gradient collector/selector tests, non-GPU proxy artifact tests, Ruff on changed Python, compileall, shell syntax, `git diff --check`, dry-run launcher, frozen hash checks, and executable-tree review.

- [ ] **Step 3: Perform whole-branch inline code/science review**

Check membership leakage, mutable defaults, hidden retries, answer threshold, RNG coupling, vLLM teardown, projection identity, K mismatch, null construction, collision behavior, and whether any path can launch 2K generation after calibration `no_go`. Fix findings with TDD and commit separately.

### Task 8: CPU Preflight and Runtime Gate

- [ ] **Step 1: Run final `DRY_RUN=1`**

Expected: exact candidate/calibration counts, all source/model/reference hashes, missing artifact roots, expected phase commands, and no GPU process/model construction.

- [ ] **Step 2: Inspect `opd-CLI`**

Require active Slurm allocation, sufficient estimated wall time plus 20% reserve, idle logical GPUs `0,1,2,3`, and no stale vLLM/Ray/collector process. The user's unattended-execution approval authorizes launch if every gate passes.

### Task 9: Run Calibration and Enforce the Stop Gate

- [ ] **Step 1: Launch the canonical script inside `opd-CLI`**

Run without nested `srun` and monitor startup configuration before first generation.

- [ ] **Step 2: Validate 768 calibration trajectories and projected gradients**

Require exact request IDs/seeds, three responses per question, at least 192 qualified questions, exact majority rows, complete projected-vector coverage, and no fatal markers.

- [ ] **Step 3: Make the calibration decision**

If any primary calibration threshold fails, publish `no_go`, verify drain, and stop/report. If all pass, continue automatically to Task 10.

### Task 10: Generate and Quality-Control 2,000 Candidates

- [ ] **Step 1: Publish exactly 2,000 unique parsed candidates**

Require at most 3,000 requests, five unique parent IDs per request, exact generation configuration, and no source/evaluation contamination publication.

- [ ] **Step 2: Generate exactly 6,000 solution trajectories**

Require three per candidate, immutable request order, parse records, and complete model-process drain.

- [ ] **Step 3: Apply quality and semantic gates**

Publish every pass/rejection reason and prepare only majority-matching completion rows for gradient collection. Do not replace rejected candidates.

### Task 11: Collect New Gradients, Select Sparse Additions, and Report

- [ ] **Step 1: Run four isolated proxy-gradient shards**

Require exact global ID coverage and the frozen Qwen2.5/projector contract.

- [ ] **Step 2: Analyze sparse membership and nulls**

Publish seed-42 accepted membership, seed-43/K sensitivity, G-Vendi null percentile, and the deterministic operational decision.

- [ ] **Step 3: Final verification**

Require all process/GPU drain, unchanged source/frozen hashes, complete artifact inventory, report consistency, and `STAGE_COMPLETE.json`.

- [ ] **Step 4: Report to the user**

Report calibration result, 2K generation and quality yields, sparse accepted count, stability/null metrics, GPU cost, artifact paths/hashes, and whether scaling to 12.8K accepted is scientifically warranted. Explicitly state that no training occurred.
