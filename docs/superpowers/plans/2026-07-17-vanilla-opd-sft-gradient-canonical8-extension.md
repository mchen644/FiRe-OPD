# Vanilla OPD SFT-Gradient Canonical-Eight Evaluation Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After the active AIME 2025 evaluation completes, evaluate the same candidate on the remaining six canonical math datasets and publish an independently validated eight-dataset report.

**Architecture:** Treat the active AIME pair as immutable phase one. At its success boundary, verify process drain and six-dataset target absence, then reuse the complete merged model with `SKIP_MERGE=1` in a sequential four-GPU phase-two launch. Validate each output at dataset boundaries and atomically publish one canonical-eight report after all processes drain.

**Tech Stack:** Bash, tmux `opd-CLI`, Slurm job 410, four H200 NVL GPUs, VERL/vLLM evaluator, Python JSONL validation.

## Global Constraints

- Do not interrupt or rerun the active AIME 2025 evaluation.
- Do not rerun AIME 2024, AIME 2025, or the historical Vanilla OPD baseline.
- Run only `hmmt25_feb hmmt25_nov math500 minervamath olympiadbench amc2023` in phase two, in that order.
- Preserve the approved `n=32`, seed-42, baseline/raw-prompt, max-16,384-token protocol exactly.
- Use all four allocation tokens `0,1,2,3` as one TP=4 engine in `opd-CLI`, without nested `srun`.
- Reuse the complete merged model; set `SKIP_MERGE=1` and prohibit force merge.
- Existing AIME artifacts are immutable. Every phase-two dataset target and final report target must be absent before launch.
- Executable tracked files must remain identical to commit `94b4f85acb9aa38fff2001c4dd102a55e5daa894`; committed `docs/superpowers/` descendants are allowed.

---

### Task 1: Cross the AIME-to-Remaining-Six Boundary

**Files:**
- Read: `/home/mchen/FiRe-OPD/math_eval/opd_sftgrad51200_step_eval.launch.log`
- Read: AIME 2024/2025 candidate response and log files.
- Read: the complete merged model directory.

**Interfaces:**
- Consumes: active AIME phase.
- Produces: `REMAINING6_PREFLIGHT=PASS` with idle GPUs and collision-free targets.

- [ ] **Step 1: Wait for phase-one success**

Poll without GPU work until the launch log contains exactly `AIME51200_EVAL_DONE:0`. If the AIME process exits with another status, stop and diagnose instead of launching phase two.

- [ ] **Step 2: Validate both AIME outputs**

Require 30 unique rows per file and 32 entries in `responses`, `pred_answers`, `response_lengths`, and `acc_list` per row. Require baseline prompt style, `cod_shot=0`, finite recomputed metrics, and matching final log metrics.

- [ ] **Step 3: Verify process drain, allocation, model, source, and hashes**

Require no candidate vLLM/evaluator process, no compute application on allocation devices, job 410 running with enough time, a complete two-shard merged model, a clean worktree, executable-tree identity to `94b4f85`, and all eight benchmark plus selection hashes matching the approved specs.

- [ ] **Step 4: Verify phase-two targets are absent**

For run name `opd-strong-to-weak-sftgrad51200-rawprompt-4gpu-tp4-refmb4-rollmb4-step50-baseline-n32-seed42`, require response, mistakes, and log files to be absent for each of:

```text
hmmt25_feb hmmt25_nov math500 minervamath olympiadbench amc2023
```

Also require these paths absent:

```text
/home/mchen/FiRe-OPD/math_eval/opd_sftgrad51200_step_eval_remaining6.launch.log
/home/mchen/FiRe-OPD/math_eval/opd_sftgrad51200_step_eval_outputs/opd-strong-to-weak-sftgrad51200-rawprompt-4gpu-tp4-refmb4-rollmb4-step50-baseline-n32-seed42-canonical8-evaluation.json
```

Expected: `REMAINING6_PREFLIGHT=PASS`.

### Task 2: Launch the Remaining Six in `opd-CLI`

**Files:**
- Execute: `math_eval/run_eval_math_tale_budget_step50_table2.sh`
- Create: six response files, six mistakes files, six logs, and the phase-two launch log.

**Interfaces:**
- Consumes: Task 1 pass and the existing merged model.
- Produces: marker `REMAINING6_EVAL_DONE:<rc>`.

- [ ] **Step 1: Launch with exact explicit overrides**

Send a fail-closed wrapper to the existing `opd-CLI` pane. Its evaluator command is:

```bash
EXPERIMENT_NAME=opd-strong-to-weak-sftgrad51200-rawprompt-4gpu-tp4-refmb4-rollmb4 \
STEP=50 \
FSDP_CKPT_DIR=/home/mchen/FiRe-OPD/checkpoints/opd-strong-to-weak-sftgrad51200-rawprompt-4gpu-tp4-refmb4-rollmb4/global_step_50 \
FSDP_ACTOR_DIR=/home/mchen/FiRe-OPD/checkpoints/opd-strong-to-weak-sftgrad51200-rawprompt-4gpu-tp4-refmb4-rollmb4/global_step_50/actor \
HF_MODEL_DIR=/home/mchen/FiRe-OPD/checkpoints/opd-strong-to-weak-sftgrad51200-rawprompt-4gpu-tp4-refmb4-rollmb4/global_step_50_hf \
MODEL_PATH=/home/mchen/FiRe-OPD/checkpoints/opd-strong-to-weak-sftgrad51200-rawprompt-4gpu-tp4-refmb4-rollmb4/global_step_50_hf \
MODEL_KEY=opd_strong_to_weak_sftgrad51200_step50 \
MODEL_NAME=opd-strong-to-weak-sftgrad51200-rawprompt-4gpu-tp4-refmb4-rollmb4-step50-baseline \
RUN_NAME=opd-strong-to-weak-sftgrad51200-rawprompt-4gpu-tp4-refmb4-rollmb4-step50-baseline-n32-seed42 \
DATASETS='hmmt25_feb hmmt25_nov math500 minervamath olympiadbench amc2023' \
N_SAMPLES=32 MAX_TOKENS=16384 MAX_MODEL_LEN=40960 \
TEMPERATURE=1.0 TOP_P=1.0 MAX_NUM_SEQS=256 SEED=42 \
ENABLE_THINKING=0 NO_EXTRA_PROMPT=1 PROMPT_STYLE=baseline GPU_IDS=0,1,2,3 \
OUTPUT_ROOT=/home/mchen/FiRe-OPD/math_eval/opd_sftgrad51200_step_eval_outputs \
MISTAKES_ROOT=/home/mchen/FiRe-OPD/math_eval/opd_sftgrad51200_step_eval_mistakes \
LOG_ROOT=/home/mchen/FiRe-OPD/math_eval/opd_sftgrad51200_step_eval_logs \
PYTHON_BIN=/home/mchen/miniconda3/envs/verl/bin/python \
SKIP_MERGE=1 FORCE_MERGE=0 \
bash math_eval/run_eval_math_tale_budget_step50_table2.sh
```

Pipe output to `/home/mchen/FiRe-OPD/math_eval/opd_sftgrad51200_step_eval_remaining6.launch.log` and append `REMAINING6_EVAL_DONE:<rc>`.

- [ ] **Step 2: Gate HMMT February startup**

Require the candidate model path, bfloat16, TP=4, max sequence length 40,960, and `0/960` prompts. Reject fatal markers. Expected: `HMMT25_FEB_STARTUP_GATE=PASS`.

### Task 3: Monitor Six Sequential Dataset Boundaries

**Files:**
- Monitor: phase-two launch log and six dataset logs.
- Validate: each completed response JSONL before accepting the next dataset.

**Interfaces:**
- Consumes: active phase-two launcher.
- Produces: six complete outputs and `REMAINING6_EVAL_DONE:0`.

- [ ] **Step 1: Validate HMMT February and HMMT November**

At each boundary, require final metrics, no fatal markers, the expected row count 30, exactly 32 samples per row, and next-dataset startup. HMMT November startup must show `0/960`.

- [ ] **Step 2: Validate MATH500**

Require 500 unique rows, 16,000 total samples, final metrics, and no fatal markers. Startup must show `0/16000`.

- [ ] **Step 3: Validate MinervaMath**

Require 272 unique rows, 8,704 total samples, final metrics, and no fatal markers. Startup must show `0/8704`.

- [ ] **Step 4: Validate OlympiadBench**

Require 674 unique rows, 21,568 total samples, final metrics, and no fatal markers. Startup must show `0/21568`; monitor this long stage without timeout-based cancellation or another GPU process.

- [ ] **Step 5: Validate AMC2023 and launcher completion**

Require 40 unique rows, 1,280 total samples, final metrics, no fatal markers, and `REMAINING6_EVAL_DONE:0`. Startup must show `0/1280`.

### Task 4: Publish the Canonical-Eight Report

**Files:**
- Read: all eight candidate response/mistakes/log files and both launch logs.
- Read: six complete historical Vanilla outputs.
- Create: `/home/mchen/FiRe-OPD/math_eval/opd_sftgrad51200_step_eval_outputs/opd-strong-to-weak-sftgrad51200-rawprompt-4gpu-tp4-refmb4-rollmb4-step50-baseline-n32-seed42-canonical8-evaluation.json`

**Interfaces:**
- Consumes: eight independently validated candidate outputs.
- Produces: one immutable report with complete provenance and metrics.

- [ ] **Step 1: Revalidate every JSONL row and recompute metrics**

For each dataset, require the frozen problem count, unique problem coverage, 32 samples per row, baseline prompt metadata, boolean correctness values, and retokenized response lengths in `[0, 16385]`. The frozen historical evaluator outputs establish the one-token retokenization allowance above the 16,384 generation cap. Recompute Accuracy, pass@32, and mean length and match the logs.

- [ ] **Step 2: Compute macro metrics and available historical deltas**

Compute unweighted eight-dataset macro Accuracy, pass@32, and mean length. Compute candidate-minus-historical deltas for AIME24, AIME25, HMMT February, HMMT November, MATH500, and MinervaMath only; mark OlympiadBench and AMC2023 baseline comparisons unavailable.

- [ ] **Step 3: Publish atomically and verify final drain**

Write the absent report through a temporary file and atomic rename. Include all protocol values, counts, metrics, deltas, hashes, checkpoint/training/selection provenance, and process state. Require no candidate evaluator/merger process, no allocation GPU application, unchanged frozen hashes, and a clean executable source tree. Expected: `CANONICAL8_FINAL_VERIFICATION=PASS`.

- [ ] **Step 4: Report results**

Report all eight per-dataset metrics, eight-dataset macro metrics, available historical deltas, output/report hashes, and the explicit fact that no baseline or AIME rerun occurred.
