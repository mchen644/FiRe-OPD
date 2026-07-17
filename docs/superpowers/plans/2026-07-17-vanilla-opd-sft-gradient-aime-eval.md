# Vanilla OPD SFT-Gradient 51,200 AIME Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Independently evaluate the completed SFT-gradient-selected strong-to-weak Vanilla OPD step-50 checkpoint on AIME 2024 and AIME 2025 under the historical Vanilla `n=32`, seed-42 raw-prompt protocol and report candidate-minus-historical deltas.

**Architecture:** Reuse the committed four-GPU normal-prompt evaluation launcher with explicit absolute checkpoint and artifact overrides. Gate the run against frozen training, benchmark, source, allocation, GPU-idleness, and non-overwrite contracts; let the launcher merge the FSDP actor once and evaluate both datasets sequentially; then independently validate every response row and produce an atomic comparison report.

**Tech Stack:** Bash, tmux `opd-CLI`, Slurm job 410, four H200 NVL GPUs, `/home/mchen/miniconda3/envs/verl/bin/python`, VERL FSDP model merger, vLLM, JSONL.

## Global Constraints

- Work only from `/home/mchen/FiRe-OPD/.worktrees/vanilla-sft-gradient-51200-opd`; do not modify the dirty root checkout's source tree.
- Frozen executable source is commit `94b4f85acb9aa38fff2001c4dd102a55e5daa894`; committed documents under `docs/superpowers/` may be descendants, but every other tracked path must be identical.
- Evaluate only `aime24` and `aime25`; do not launch the other six math datasets.
- Evaluate only the new candidate; do not rerun the historical Vanilla baseline.
- Use normal/raw prompts: `--prompt_style baseline`, `--no_extra_prompt`, no thinking-template override.
- Use exactly `n=32`, seed 42, temperature 1.0, top-p 1.0, max generated tokens 16,384, max model length 40,960, and max active sequences 256.
- Use all four allocation-owned devices `0,1,2,3` as one tensor-parallel vLLM engine and run AIME 2024 before AIME 2025.
- Run inside tmux session `opd-CLI` on the existing allocation without nested `srun`.
- Resume and overwrite are forbidden. Every candidate response, mistakes, log, launch-log, and report target must be absent before launch.
- Preserve and recheck the frozen 51,200-row selection hashes and the historical Vanilla output files.
- A failure on AIME 2025 must not overwrite or discard a successfully completed AIME 2024 output.

---

### Task 1: Freeze and Verify the Evaluation Contract

**Files:**
- Read: `/home/mchen/FiRe-OPD/logs/gradient_diversity/opd-strong-to-weak-sftgrad51200-rawprompt-4gpu-tp4-refmb4-rollmb4.completion.json`
- Read: `/home/mchen/FiRe-OPD/data/aime24/test.jsonl`
- Read: `/home/mchen/FiRe-OPD/data/aime25/test.jsonl`
- Read: `math_eval/run_eval_math_tale_budget_step50_table2.sh`
- Produce later: no source files

**Interfaces:**
- Consumes: the validated step-50 FSDP actor and the approved evaluation design.
- Produces: one preflight pass proving the exact command can launch without reuse, drift, or collision.

- [ ] **Step 1: Verify the worktree and executable-tree identity**

Run:

```bash
set -euo pipefail
WORKTREE=/home/mchen/FiRe-OPD/.worktrees/vanilla-sft-gradient-51200-opd
cd "$WORKTREE"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
git diff --quiet \
  94b4f85acb9aa38fff2001c4dd102a55e5daa894 HEAD -- \
  . ':(exclude)docs/superpowers/**'
bash -n math_eval/run_eval_math_tale_budget_step50_table2.sh
printf 'SOURCE_AND_SHELL_GATE=PASS\n'
```

Expected: `SOURCE_AND_SHELL_GATE=PASS` and exit status zero.

- [ ] **Step 2: Revalidate training completion**

Run:

```bash
cd /home/mchen/FiRe-OPD/.worktrees/vanilla-sft-gradient-51200-opd
PYTHONPATH="$PWD/verl:$PWD" \
/home/mchen/miniconda3/envs/verl/bin/python \
  -m math_eval.validate_vanilla_opd_checkpoint \
  --checkpoint-root /home/mchen/FiRe-OPD/checkpoints/opd-strong-to-weak-sftgrad51200-rawprompt-4gpu-tp4-refmb4-rollmb4 \
  --log-path /home/mchen/FiRe-OPD/logs/gradient_diversity/opd-strong-to-weak-sftgrad51200-rawprompt-4gpu-tp4-refmb4-rollmb4.log \
  --expected-step 50 \
  --expected-samples 51200 \
  --expected-batches 50 \
  --world-size 4
```

Expected: JSON with `status: "complete"`, `samples_yielded: 51200`, `sampler_batches: 50`, and `world_size: 4`.

- [ ] **Step 3: Verify benchmark and selection hashes**

Run an inline Python hash gate over these exact mappings:

```python
expected = {
    "/home/mchen/FiRe-OPD/data/aime24/test.jsonl": "a3c49569f3d7125aaf4eea5764bd1b868af534ae3aff6aad0843a7da58fe46b8",
    "/home/mchen/FiRe-OPD/data/aime25/test.jsonl": "6012af2a112f5e26d91f1b0cc644b5dde6399173b8b648aec2e6b9899877a2db",
    "/home/mchen/FiRe-OPD/data/gradient_diversity/DeepMath-103K/train_gradient_diverse_51200.parquet": "ffdd06af56361bd49b25b0da5039db4ce70c21e443e3b96687f61376b078ea2c",
    "/home/mchen/FiRe-OPD/data/gradient_diversity/selection_51200/manifest.json": "6b2cff5bf5d9bece95d280ddfc32ccf13377a05f027011a29bb05fc7882943de",
    "/home/mchen/FiRe-OPD/data/gradient_diversity/selection_51200/selected_ids.jsonl": "5d8868538679cd69f3f0bac896575789ff98be6de0e1214bc3441235e13bff2b",
    "/home/mchen/FiRe-OPD/data/gradient_diversity/selection_51200/diagnostics.json": "a64c20bb285fa13c55e774e7367135968385b39ea60141dd74a42c1f26d5010c",
}
```

Also require exactly 30 nonempty JSONL rows in each AIME input. Expected: `FROZEN_HASH_GATE=PASS`.

- [ ] **Step 4: Prove all publication targets are absent**

Require these paths to be absent:

```text
/home/mchen/FiRe-OPD/checkpoints/opd-strong-to-weak-sftgrad51200-rawprompt-4gpu-tp4-refmb4-rollmb4/global_step_50_hf
/home/mchen/FiRe-OPD/math_eval/opd_sftgrad51200_step_eval_outputs
/home/mchen/FiRe-OPD/math_eval/opd_sftgrad51200_step_eval_mistakes
/home/mchen/FiRe-OPD/math_eval/opd_sftgrad51200_step_eval_logs
/home/mchen/FiRe-OPD/math_eval/opd_sftgrad51200_step_eval.launch.log
/home/mchen/FiRe-OPD/math_eval/opd_sftgrad51200_step_eval_outputs/opd-strong-to-weak-sftgrad51200-rawprompt-4gpu-tp4-refmb4-rollmb4-step50-baseline-n32-seed42-aime-comparison.json
```

Expected: `EMPTY_TARGET_GATE=PASS`. Any existing target stops execution.

- [ ] **Step 5: Reconfirm allocation ownership and idle GPUs inside `opd-CLI`**

Run a probe in the pane that prints `SLURM_JOB_ID`, hostname, `CUDA_VISIBLE_DEVICES`, remaining job time from `squeue`, and `nvidia-smi --query-compute-apps`. Expected: job 410 is running, the pane is on `heisenberg`, logical devices `0,1,2,3` are available, and there are no compute processes.

### Task 2: Merge and Launch the Two-Dataset Evaluation

**Files:**
- Execute: `math_eval/run_eval_math_tale_budget_step50_table2.sh`
- Create: `/home/mchen/FiRe-OPD/checkpoints/opd-strong-to-weak-sftgrad51200-rawprompt-4gpu-tp4-refmb4-rollmb4/global_step_50_hf/`
- Create: `/home/mchen/FiRe-OPD/math_eval/opd_sftgrad51200_step_eval.launch.log`
- Create: candidate response, mistakes, and dataset-log trees specified in the design.

**Interfaces:**
- Consumes: Task 1 preflight pass and idle allocation tokens `0,1,2,3`.
- Produces: marker `AIME51200_EVAL_DONE:<rc>` and two independent candidate JSONL outputs when `<rc>` is zero.

- [ ] **Step 1: Send the exact launch command to `opd-CLI`**

From the login-side controller, use `tmux send-keys` to execute this command in the existing pane, with no nested `srun`:

```bash
cd /home/mchen/FiRe-OPD/.worktrees/vanilla-sft-gradient-51200-opd
set -o pipefail
EXPERIMENT_NAME=opd-strong-to-weak-sftgrad51200-rawprompt-4gpu-tp4-refmb4-rollmb4 \
STEP=50 \
FSDP_CKPT_DIR=/home/mchen/FiRe-OPD/checkpoints/opd-strong-to-weak-sftgrad51200-rawprompt-4gpu-tp4-refmb4-rollmb4/global_step_50 \
FSDP_ACTOR_DIR=/home/mchen/FiRe-OPD/checkpoints/opd-strong-to-weak-sftgrad51200-rawprompt-4gpu-tp4-refmb4-rollmb4/global_step_50/actor \
HF_MODEL_DIR=/home/mchen/FiRe-OPD/checkpoints/opd-strong-to-weak-sftgrad51200-rawprompt-4gpu-tp4-refmb4-rollmb4/global_step_50_hf \
MODEL_PATH=/home/mchen/FiRe-OPD/checkpoints/opd-strong-to-weak-sftgrad51200-rawprompt-4gpu-tp4-refmb4-rollmb4/global_step_50_hf \
MODEL_KEY=opd_strong_to_weak_sftgrad51200_step50 \
MODEL_NAME=opd-strong-to-weak-sftgrad51200-rawprompt-4gpu-tp4-refmb4-rollmb4-step50-baseline \
RUN_NAME=opd-strong-to-weak-sftgrad51200-rawprompt-4gpu-tp4-refmb4-rollmb4-step50-baseline-n32-seed42 \
DATASETS='aime24 aime25' \
N_SAMPLES=32 \
MAX_TOKENS=16384 \
MAX_MODEL_LEN=40960 \
TEMPERATURE=1.0 \
TOP_P=1.0 \
MAX_NUM_SEQS=256 \
SEED=42 \
ENABLE_THINKING=0 \
NO_EXTRA_PROMPT=1 \
PROMPT_STYLE=baseline \
GPU_IDS=0,1,2,3 \
OUTPUT_ROOT=/home/mchen/FiRe-OPD/math_eval/opd_sftgrad51200_step_eval_outputs \
MISTAKES_ROOT=/home/mchen/FiRe-OPD/math_eval/opd_sftgrad51200_step_eval_mistakes \
LOG_ROOT=/home/mchen/FiRe-OPD/math_eval/opd_sftgrad51200_step_eval_logs \
PYTHON_BIN=/home/mchen/miniconda3/envs/verl/bin/python \
SKIP_MERGE=0 \
FORCE_MERGE=0 \
bash math_eval/run_eval_math_tale_budget_step50_table2.sh \
  2>&1 | tee /home/mchen/FiRe-OPD/math_eval/opd_sftgrad51200_step_eval.launch.log
rc=${PIPESTATUS[0]}
echo "AIME51200_EVAL_DONE:${rc}" | tee -a /home/mchen/FiRe-OPD/math_eval/opd_sftgrad51200_step_eval.launch.log
```

Expected: merge starts once, followed by AIME 2024 evaluation.

- [ ] **Step 2: Gate the merged model before accepting generation startup**

Require `config.json`, tokenizer files, and either `model.safetensors` or `model.safetensors.index.json` in the merged directory. Parse the index, if present, and require every referenced shard to exist and be nonempty. Expected: `MERGED_MODEL_GATE=PASS`.

- [ ] **Step 3: Gate AIME 2024 engine startup**

Inspect the AIME 2024 log and require the candidate merged path, bfloat16, tensor parallel size 4, max sequence length 40,960, and exactly 960 requested prompts. Reject traceback, CUDA OOM, killed process, or engine failure. Expected: `AIME24_STARTUP_GATE=PASS`.

### Task 3: Monitor to Completion Without Perturbing the Run

**Files:**
- Monitor: `/home/mchen/FiRe-OPD/math_eval/opd_sftgrad51200_step_eval.launch.log`
- Monitor: per-dataset logs under `/home/mchen/FiRe-OPD/math_eval/opd_sftgrad51200_step_eval_logs/`

**Interfaces:**
- Consumes: active Task 2 vLLM process.
- Produces: completed AIME 2024 and AIME 2025 logs and marker `AIME51200_EVAL_DONE:0`.

- [ ] **Step 1: Wait for AIME 2024 completion**

Poll logs and process state without launching another GPU process. Require final `Accuracy:`, `pass@k:`, and `avg_length:` lines and a 30-row candidate output before AIME 2025 startup. Expected: AIME 2024 process exits zero and the launcher proceeds automatically.

- [ ] **Step 2: Gate AIME 2025 engine startup**

Apply the same startup checks as AIME 2024, including candidate path, TP=4, max sequence length 40,960, and 960 prompts. Expected: `AIME25_STARTUP_GATE=PASS`.

- [ ] **Step 3: Wait for launcher completion**

Poll until the pane and launch log contain `AIME51200_EVAL_DONE:0`. If the marker is nonzero or absent after the evaluation process exits, stop and diagnose before any relaunch. Expected: both per-dataset logs end with metrics and no fatal marker.

### Task 4: Independently Validate and Compare Results

**Files:**
- Read: the two new candidate response JSONLs.
- Read: the two historical Vanilla response JSONLs under `/home/mchen/FiRe-OPD/math_eval/opd_rawprompt_step_eval_outputs/`.
- Create: `/home/mchen/FiRe-OPD/math_eval/opd_sftgrad51200_step_eval_outputs/opd-strong-to-weak-sftgrad51200-rawprompt-4gpu-tp4-refmb4-rollmb4-step50-baseline-n32-seed42-aime-comparison.json`

**Interfaces:**
- Consumes: two completed candidate outputs and two immutable historical outputs.
- Produces: a validated atomic JSON report with per-dataset and macro metrics and candidate-minus-historical deltas.

- [ ] **Step 1: Validate every candidate and historical row**

For each dataset/model file, parse all JSONL records and require:

```python
assert len(rows) == 30
assert len({row["problem"] for row in rows}) == 30
for row in rows:
    assert row["prompt_style"] == "baseline"
    assert row["cod_shot"] == 0
    for key in ("responses", "pred_answers", "response_lengths", "acc_list"):
        assert len(row[key]) == 32
```

Also require all response lengths to be integers in `[0, 16385]`, all `acc_list` entries to be booleans, and every expected input problem to occur exactly once. The evaluator measures decoded text by retokenizing it, and the frozen historical Vanilla outputs establish that a 16,384-token generation can retokenize to 16,385 tokens. Expected: `JSONL_COMPLETENESS_GATE=PASS`.

- [ ] **Step 2: Recompute metrics from JSONL rather than trusting logs**

For each file compute:

```python
accuracy = sum(sum(row["acc_list"]) for row in rows) / (30 * 32)
pass_at_32 = sum(any(row["acc_list"]) for row in rows) / 30
mean_length = sum(sum(row["response_lengths"]) for row in rows) / (30 * 32)
```

Require candidate values to match printed log metrics at the log's displayed precision. Compute per-dataset deltas as candidate minus historical and unweighted AIME24/AIME25 means for Accuracy and pass@32.

- [ ] **Step 3: Publish the report atomically**

Write a temporary JSON file beside the report, fsync/close it, and rename it to the final absent target. Include protocol values, source/training/selection provenance, benchmark hashes, candidate and historical file hashes, per-dataset metrics, deltas, macro means, merged-model file hashes, and launch/dataset log hashes. Expected: report parses as JSON and has status `complete`.

- [ ] **Step 4: Perform final drain and immutability checks**

Require no candidate `eval_math.py`, vLLM, Ray, or merger process; no GPU compute process; unchanged AIME and selection hashes; a clean worktree; and executable-tree identity to commit `94b4f85`. Hash every new response, mistakes, log, launch-log, report, and merged-model marker/model file. Expected: `FINAL_EVAL_VERIFICATION=PASS`.

- [ ] **Step 5: Report the narrow scientific result**

Report AIME 2024 and AIME 2025 candidate Accuracy, pass@32, and mean length; historical values; candidate-minus-historical deltas; two-dataset macro Accuracy and pass@32; artifact paths and hashes; and the fact that no other benchmark or baseline rerun was performed. Treat results as directional rather than publication-grade causal attribution.
