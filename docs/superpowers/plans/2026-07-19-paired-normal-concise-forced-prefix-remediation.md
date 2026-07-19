# Paired Normal/Concise Forced-Prefix Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace invalid independent relaxed resampling with an auditable forced-prefix continuation, then complete the base/adaptive paired probe in a fresh namespace.

**Architecture:** Keep the frozen normal and capped-concise requests unchanged. Add pure helpers that construct continuation token prompts from vLLM's exact prompt IDs plus the observed capped IDs and normalize the generated suffix into one combined response. Extend both the primary and independent validators with continuation-provenance invariants before profiling and relaunching sequential TP=4 generation.

**Tech Stack:** Python 3.10, Transformers, vLLM 0.8.5.post1, vendored VERL reward helpers, PyArrow, pytest, Ruff, Bash, tmux, four allocation-owned GPUs.

## Global Constraints

- Work only in `/home/mchen/FiRe-OPD/.worktrees/vanilla-sft-gradient-51200-opd`; leave the dirty root checkout untouched.
- Use `/home/mchen/miniconda3/envs/gvendi-opd/bin/python` for CPU tests/analysis and `/home/mchen/miniconda3/envs/verl/bin/python` for vLLM.
- Preserve failed namespaces `20260719T155519Z-f98abd12ebaf` and `20260719T155822Z-92e670dbb302`; never resume or overwrite them.
- Keep the persisted 128-row sampling protocol, both model paths, normal/capped generation, scoring, quadrants, and all GPU settings frozen.
- Counterfactual generation applies only to capped `compression_sensitive` events and conditions on their exact observed capped token IDs.
- Continue using `VLLM_WORKER_MULTIPROC_METHOD=spawn`.
- Do not train, mutate model/checkpoint files, launch W&B, use nested `srun`, or run canonical benchmarks.

---

### Task 1: Add forced-prefix continuation contracts

**Files:**
- Modify: `math_eval/test_paired_normal_concise_probe.py`
- Modify: `math_eval/paired_normal_concise_probe.py`

**Interfaces:**
- Produces: `forced_prefix_continuation_prompt(request_output, capped_ids) -> dict[str, list[int]]`.
- Produces: `normalize_forced_prefix_continuation(request_output, *, capped_response, tokenizer, ground_truth, total_max_tokens, rendered_prompt, continuation_seed) -> dict`.

- [ ] **Step 1: Write failing tests for exact token-prompt construction**

Create a fake request output with `prompt_token_ids=[10, 11]` and capped IDs `[20, 21]`. Assert the helper returns `{"prompt_token_ids": [10, 11, 20, 21]}` without mutating either input. Assert missing/empty/noninteger prompt or capped IDs fail.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
PYTHONPATH="$PWD/verl:$PWD" /home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest \
  math_eval/test_paired_normal_concise_probe.py -q
```

Expected: import failure for the new helper.

- [ ] **Step 3: Implement minimal token-prompt construction**

Read exact `request_output.prompt_token_ids`, defensively copy both integer lists, concatenate them, and return a vLLM `TokensPrompt`-compatible dictionary.

- [ ] **Step 4: Write failing tests for combined continuation normalization**

Use a fake continuation containing suffix IDs `[30, 31]`, a capped response containing `[20, 21]`, and a fake tokenizer whose decode result has a correct boxed answer. Assert combined IDs `[20, 21, 30, 31]`, total `max_tokens`, exact continuation metadata, scoring, parseability, and cap-hit behavior. Add a test for a zero-token continuation budget that makes no request and preserves the capped response.

- [ ] **Step 5: Run and verify RED**

Expected: import failure for the normalization helper.

- [ ] **Step 6: Implement minimal normalization**

Validate `total_max_tokens >= capped_length`, normalize exactly one generated suffix when remaining budget is positive, decode the combined IDs with `skip_special_tokens=True`, score the combined text, and persist all provenance fields from the design. Handle remaining budget zero without requiring an output.

- [ ] **Step 7: Run focused tests and verify GREEN**

Run the Task 1 pytest command. Expected: all tests pass.

---

### Task 2: Route production counterfactuals through forced prefixes

**Files:**
- Modify: `math_eval/test_paired_normal_concise_probe.py`
- Modify: `math_eval/paired_normal_concise_probe.py`

**Interfaces:**
- Consumes the Task 1 helpers.
- Produces required forced-prefix provenance in every non-null `relaxed_concise` record.

- [ ] **Step 1: Write failing integration-level helper tests**

Test that response validation and `classify_compression_sensitive` reject a non-null relaxed response without `counterfactual_mode="forced_prefix_continuation"`, an incorrect continuation seed, and combined IDs inconsistent with `capped + continuation`.

- [ ] **Step 2: Run and verify RED**

Expected: at least one malformed forced-prefix record is incorrectly accepted.

- [ ] **Step 3: Replace independent relaxed generation**

For eligible indices, compute remaining budgets, build token prompts from each original concise `RequestOutput`, call `_generate_batch` only for positive remaining budgets, normalize suffixes with Task 1's helper, and attach combined responses. Do not alter normal or capped-concise calls.

- [ ] **Step 4: Add primary invariant validation**

Require exact mode, prefix length, seed, remaining budget, suffix IDs/length, and combined IDs in `classify_compression_sensitive` and `summarize_records`. Persist the mode in `manifest.json` and `generation.json`.

- [ ] **Step 5: Run probe tests and verify GREEN**

```bash
PYTHONPATH="$PWD/verl:$PWD" /home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest -q \
  math_eval/test_paired_normal_concise_probe.py
```

---

### Task 3: Extend independent validation

**Files:**
- Modify: `math_eval/test_validate_paired_normal_concise_probe.py`
- Modify: `math_eval/validate_paired_normal_concise_probe.py`

**Interfaces:**
- Independently verifies forced-prefix continuation provenance and the manifest protocol.

- [ ] **Step 1: Add failing tamper tests**

Add mutations for `counterfactual_mode`, `forced_prefix_length`, `continuation_seed`, `continuation_max_tokens`, `continuation_length`, and one continuation token. Rehash the changed records and assert each mutation raises a targeted `ValueError` before summary comparison.

- [ ] **Step 2: Run and verify RED**

Expected: at least one continuation-provenance mutation is accepted.

- [ ] **Step 3: Implement independent checks**

Recompute remaining budget and combined IDs directly from JSONL, require the row seed, verify the manifest mode, and reject provenance on ineligible rows.

- [ ] **Step 4: Run all probe tests and static checks**

```bash
PYTHONPATH="$PWD/verl:$PWD" /home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest -q \
  math_eval/test_paired_normal_concise_probe.py \
  math_eval/test_validate_paired_normal_concise_probe.py \
  math_eval/test_paired_normal_concise_probe_launcher.py
/home/mchen/miniconda3/envs/verl/bin/python -m ruff check \
  math_eval/paired_normal_concise_probe.py \
  math_eval/validate_paired_normal_concise_probe.py \
  math_eval/test_paired_normal_concise_probe.py \
  math_eval/test_validate_paired_normal_concise_probe.py
/home/mchen/miniconda3/envs/verl/bin/python -m compileall -q \
  math_eval/paired_normal_concise_probe.py \
  math_eval/validate_paired_normal_concise_probe.py
bash -n run_paired_normal_concise_probe.sh
git diff --check
```

- [ ] **Step 5: Commit implementation**

```bash
git add math_eval/paired_normal_concise_probe.py \
  math_eval/validate_paired_normal_concise_probe.py \
  math_eval/test_paired_normal_concise_probe.py \
  math_eval/test_validate_paired_normal_concise_probe.py
git commit -m "fix: force paired probe counterfactual prefixes"
```

---

### Task 4: Profile, launch fresh, and validate

**Files:**
- Runtime only under new immutable probe output/log namespaces.

**Interfaces:**
- Produces one independently validated two-model probe and final report.

- [ ] **Step 1: Commit the remediation design and plan before code execution**

Commit both remediation documents separately so protocol provenance predates implementation.

- [ ] **Step 2: Run the complete 189-test adaptive regression suite**

Use the same regression command previously used for the adaptive implementation and require all tests to pass.

- [ ] **Step 3: Run allocation and GPU gate**

Require job 429 `RUNNING`, CUDA tokens `0,1,2,3`, no compute processes, valid data hash, and complete model shards.

- [ ] **Step 4: Run a focused GPU profile**

Use a new diagnostic namespace and a small subset that exercises at least one eligible continuation. Require exact capped-prefix preservation and valid continuation metadata. Preserve any failed profile and do not promote it.

- [ ] **Step 5: Launch a fresh formal namespace**

Run the unchanged sequential launcher from clean committed source in `opd-CLI`. Poll completion marker, fatal signatures, and process exit; do not relaunch while status is ambiguous.

- [ ] **Step 6: Independently validate and hash artifacts**

Require exit-zero marker, rerun the independent validator, count 128 rows per checkpoint, recompute metrics from JSONL, and SHA256 the log plus every persisted artifact.

- [ ] **Step 7: Inspect cases and report**

Report four quadrants, counterfactual recovery classes, length/cap statistics, cross-checkpoint differences, and evidence-backed qualitative examples. State the `n=1`, 128-row, conditional-continuation limitations explicitly.
