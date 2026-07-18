# Adaptive Concise OPD Old-Log-Prob OOM Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable VERL's bounded-memory actor entropy calculation, validate it, preserve the failed run, and start a fresh accepted 50-step adaptive concise OPD run.

**Architecture:** Keep the existing adaptive concise trainer and all scientific settings unchanged. Pin the existing `actor_rollout_ref.actor.entropy_from_logits_with_chunking=True` execution control in the launcher, prove the pin through Hydra-composed launcher tests, then archive old namespaces and gate a fresh formal launch behind a one-step four-GPU profile.

**Tech Stack:** Bash launcher, Hydra/OmegaConf, pytest, VERL FSDP actor, PyTorch/CUDA, Ray, Slurm, tmux, W&B.

## Global Constraints

- Work only in `/home/mchen/FiRe-OPD/.worktrees/vanilla-sft-gradient-51200-opd`; do not modify the dirty root checkout.
- Use `/home/mchen/miniconda3/envs/verl/bin/python` for VERL tests/runtime and `/home/mchen/miniconda3/envs/gvendi-opd/bin/python` for artifact analysis.
- Preserve Qwen3-4B student, Qwen3-30B teacher, 1,024 distinct questions/actor rows per step, 50 steps, seed 42, maximum response 16,384, LR `1e-6`, reverse-KL-only advantages, token-mean loss, and token IS threshold 5.0.
- Keep old-log-prob micro-batch size four and actor update micro-batch size one.
- Chunking may affect only entropy telemetry; entropy-aware distillation remains disabled and `actor.entropy_coeff=0`.
- Preserve and checksum every historical log/acceptance artifact; never overwrite or resume a failed namespace.
- Run GPU work only in tmux `opd-CLI`, on allocation-owned logical tokens `0,1,2,3`, without nested `srun`.
- Run a fresh profile before the formal run; launch the formal run from step 0 only after profile acceptance.
- Do not push, merge, delete the worktree, or clean up the branch while training uses it.

---

### Task 1: Pin bounded-memory actor entropy with TDD

**Files:**
- Modify: `verl/tests/trainer/ppo/test_adaptive_concise_launcher.py`
- Modify: `run_train_adaptive_concise_token_neutral_opd.sh`

**Interfaces:**
- Consumes: the launcher's `adaptive_command=` dry-run contract and Hydra config `actor_rollout_ref.actor.entropy_from_logits_with_chunking`.
- Produces: a launcher command where actor entropy chunking resolves to boolean `True` and `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu` remains integer `4`.

- [ ] **Step 1: Write the failing launcher test**

Add this test after `test_launcher_command_composes_and_passes_runtime_validation`:

```python
def test_launcher_chunks_actor_entropy_without_changing_old_log_prob_micro_batch() -> None:
    completed = _run_launcher("full")
    assert completed.returncode == 0, completed.stderr
    command_line = next(
        line for line in completed.stdout.splitlines() if line.startswith("adaptive_command=")
    )
    argv = shlex.split(command_line.split("=", 1)[1])
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        config = compose(config_name="ppo_trainer", overrides=argv[3:])
    assert config.actor_rollout_ref.actor.entropy_from_logits_with_chunking is True
    assert config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu == 4
```

Also add this exact required token to the `required` list in `test_launcher_pins_profile_and_full_contracts`:

```python
"actor_rollout_ref.actor.entropy_from_logits_with_chunking=True",
```

- [ ] **Step 2: Run the new test and verify RED**

Run:

```bash
cd /home/mchen/FiRe-OPD/.worktrees/vanilla-sft-gradient-51200-opd
PYTHONPATH="$PWD/verl:$PWD" /home/mchen/miniconda3/envs/verl/bin/python -m pytest \
  verl/tests/trainer/ppo/test_adaptive_concise_launcher.py::test_launcher_chunks_actor_entropy_without_changing_old_log_prob_micro_batch \
  -q
```

Expected: FAIL because `config.actor_rollout_ref.actor.entropy_from_logits_with_chunking` is currently `False`.

- [ ] **Step 3: Add the minimal launcher override**

In `fixed_args`, immediately after `actor_rollout_ref.actor.entropy_coeff=0`, add:

```bash
"actor_rollout_ref.actor.entropy_from_logits_with_chunking=True"
```

Do not change the actor/ref micro-batch settings or any other argument.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run:

```bash
PYTHONPATH="$PWD/verl:$PWD" /home/mchen/miniconda3/envs/verl/bin/python -m pytest \
  verl/tests/trainer/ppo/test_adaptive_concise_launcher.py -q
```

Expected: all adaptive launcher tests pass.

- [ ] **Step 5: Review and commit the single production change**

Run:

```bash
git diff -- run_train_adaptive_concise_token_neutral_opd.sh \
  verl/tests/trainer/ppo/test_adaptive_concise_launcher.py
git diff --check
git add run_train_adaptive_concise_token_neutral_opd.sh \
  verl/tests/trainer/ppo/test_adaptive_concise_launcher.py
git commit -m "fix: bound adaptive OPD entropy memory"
```

Expected: one launcher override plus its regression assertions, with no unrelated changes.

---

### Task 2: Run the complete CPU and provenance gate

**Files:**
- Verify: `run_train_adaptive_concise_token_neutral_opd.sh`
- Verify: `verl/tests/trainer/ppo/test_adaptive_concise_launcher.py`
- Verify the existing adaptive and legacy regression suites.

**Interfaces:**
- Consumes: committed launcher source from Task 1.
- Produces: a clean source commit and dry-run contracts suitable for immutable GPU provenance.

- [ ] **Step 1: Run the focused adaptive suite**

```bash
PYTHONPATH="$PWD/verl:$PWD" /home/mchen/miniconda3/envs/verl/bin/python -m pytest \
  verl/tests/trainer/ppo/test_adaptive_concise_opd.py \
  verl/tests/trainer/ppo/test_adaptive_concise_trainer.py \
  verl/tests/trainer/ppo/test_adaptive_concise_launcher.py \
  verl/tests/workers/rollout/rollout_vllm/test_adaptive_max_tokens.py \
  math_eval/test_validate_adaptive_concise_profile.py -q
```

Expected: all focused adaptive tests pass.

- [ ] **Step 2: Run the legacy regression suite**

```bash
PYTHONPATH="$PWD/verl:$PWD" /home/mchen/miniconda3/envs/verl/bin/python -m pytest \
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

Expected: all legacy regression tests pass.

- [ ] **Step 3: Run static checks**

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m ruff check \
  verl/tests/trainer/ppo/test_adaptive_concise_launcher.py
/home/mchen/miniconda3/envs/verl/bin/python -m compileall -q \
  verl/tests/trainer/ppo/test_adaptive_concise_launcher.py
bash -n run_train_adaptive_concise_token_neutral_opd.sh
git diff --check
git status --porcelain=v1 --untracked-files=all
```

Expected: all checks exit zero and the worktree is clean.

- [ ] **Step 4: Freeze and inspect both dry-run contracts**

```bash
ADAPTIVE_CONCISE_DRY_RUN=1 ADAPTIVE_CONCISE_RUN_MODE=profile \
  bash run_train_adaptive_concise_token_neutral_opd.sh > /tmp/adaptive_concise_profile_oomsafe.contract
ADAPTIVE_CONCISE_DRY_RUN=1 ADAPTIVE_CONCISE_RUN_MODE=full \
  bash run_train_adaptive_concise_token_neutral_opd.sh > /tmp/adaptive_concise_full_oomsafe.contract
for contract in /tmp/adaptive_concise_{profile,full}_oomsafe.contract; do
  test "$(grep -o 'actor_rollout_ref.actor.entropy_from_logits_with_chunking=True' "$contract" | wc -l)" -eq 1
  test "$(grep -o 'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4' "$contract" | wc -l)" -eq 1
done
sha256sum /tmp/adaptive_concise_profile_oomsafe.contract \
  /tmp/adaptive_concise_full_oomsafe.contract
git rev-parse HEAD
git status --porcelain=v1 --untracked-files=all
```

Expected: each contract contains exactly one selected chunking flag and one unchanged micro-batch pin; source is clean.

---

### Task 3: Archive the stopped profile and failed formal run

**Files:**
- Move historical runtime files from `/home/mchen/FiRe-OPD/logs/adaptive_concise_opd/` into immutable timestamped archives.
- Create: archive-local `manifest.json` files outside git.

**Interfaces:**
- Consumes: the stopped profile at source `ece7baa` and failed full run W&B `0c69lwj3`/Slurm job `429`.
- Produces: absent canonical log/acceptance paths and checksum manifests preserving every moved artifact.

- [ ] **Step 1: Verify the failed training process is gone and no partial checkpoint exists**

```bash
if ps -eo cmd | grep -E '[v]erl\.trainer\.main_ppo|[r]ay::(TaskRunner|WorkerDict)' | \
    grep '/home/mchen/FiRe-OPD/.worktrees/vanilla-sft-gradient-51200-opd'; then
  echo 'adaptive training workers still exist' >&2
  exit 1
fi
test ! -e /home/mchen/FiRe-OPD/checkpoints/opd-adaptive-conciseprobe-cap50-tokenneutral-rawprompt-4gpu-tp4-step50
test ! -e /home/mchen/FiRe-OPD/checkpoints/opd-adaptive-conciseprobe-cap50-tokenneutral-rawprompt-4gpu-tp4-profile-step1
```

Expected: no matching worker and no checkpoint namespace.

- [ ] **Step 2: Atomically move the historical files into unique directories**

```bash
LOG_ROOT=/home/mchen/FiRe-OPD/logs/adaptive_concise_opd
FAILED_DIR="$LOG_ROOT/failed_attempts/20260718T030748Z_ece7baa_step17_oom"
PROFILE_DIR="$LOG_ROOT/profile_archives/20260717T215903Z_ece7baa"
mkdir -p "$FAILED_DIR" "$PROFILE_DIR"
test ! -e "$FAILED_DIR/opd-adaptive-conciseprobe-cap50-tokenneutral-rawprompt-4gpu-tp4-step50.log"
test ! -e "$PROFILE_DIR/profile-step1.log"
mv "$LOG_ROOT/opd-adaptive-conciseprobe-cap50-tokenneutral-rawprompt-4gpu-tp4-step50.log" "$FAILED_DIR/"
mv "$LOG_ROOT/opd-adaptive-conciseprobe-cap50-tokenneutral-rawprompt-4gpu-tp4-step50.step1.acceptance.json" "$FAILED_DIR/"
mv "$LOG_ROOT/profile-step1.log" "$PROFILE_DIR/"
mv "$LOG_ROOT/profile-step1.acceptance.json" "$PROFILE_DIR/"
```

Expected: all four canonical paths are now absent and all four archive files exist.

- [ ] **Step 3: Write checksum manifests without modifying archived payloads**

Run:

```bash
/home/mchen/miniconda3/envs/gvendi-opd/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path

archives = [
    (
        Path('/home/mchen/FiRe-OPD/logs/adaptive_concise_opd/failed_attempts/20260718T030748Z_ece7baa_step17_oom'),
        {
            'kind': 'failed_formal_run',
            'source_commit': 'ece7baaa1eb063798b867430ccfa533b2d4da777',
            'wandb_run_id': '0c69lwj3',
            'slurm_job_id': '429',
            'completed_steps': 16,
            'failed_attempted_step': 17,
            'failed_stage': 'actor_rollout_compute_log_prob/entropy_from_logits',
            'exception': 'torch.OutOfMemoryError',
            'requested_allocation_gib': 35.79,
            'free_memory_gib': 35.13,
        },
    ),
    (
        Path('/home/mchen/FiRe-OPD/logs/adaptive_concise_opd/profile_archives/20260717T215903Z_ece7baa'),
        {
            'kind': 'superseded_successful_profile',
            'source_commit': 'ece7baaa1eb063798b867430ccfa533b2d4da777',
            'decision': 'pass',
            'superseded_reason': 'fresh profile required after entropy-memory remediation',
        },
    ),
]
for directory, metadata in archives:
    artifacts = []
    for path in sorted(directory.iterdir()):
        if path.name == 'manifest.json' or not path.is_file():
            continue
        payload = path.read_bytes()
        artifacts.append({
            'name': path.name,
            'size': len(payload),
            'sha256': hashlib.sha256(payload).hexdigest(),
        })
    manifest = {'schema_version': 1, **metadata, 'artifacts': artifacts}
    output = directory / 'manifest.json'
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
PY
sha256sum "$FAILED_DIR"/* "$PROFILE_DIR"/*
chmod -R a-w "$FAILED_DIR" "$PROFILE_DIR"
test ! -e "$LOG_ROOT/profile-step1.log"
test ! -e "$LOG_ROOT/profile-step1.acceptance.json"
test ! -e "$LOG_ROOT/opd-adaptive-conciseprobe-cap50-tokenneutral-rawprompt-4gpu-tp4-step50.log"
test ! -e "$LOG_ROOT/opd-adaptive-conciseprobe-cap50-tokenneutral-rawprompt-4gpu-tp4-step50.step1.acceptance.json"
```

Expected: both manifests contain nonempty artifact lists; archives are read-only; canonical paths are free.

---

### Task 4: Pass a fresh four-GPU profile

**Files:**
- Create at runtime: `/home/mchen/FiRe-OPD/logs/adaptive_concise_opd/profile-step1.log`
- Create at runtime: `/home/mchen/FiRe-OPD/logs/adaptive_concise_opd/profile-step1.acceptance.json`

**Interfaces:**
- Consumes: clean committed source, allocation-owned logical GPU tokens `0,1,2,3`, and empty canonical profile paths.
- Produces: a cleanly exited one-step profile and strict acceptance JSON tied to the remediation commit.

- [ ] **Step 1: Verify tmux, allocation, source, namespace, and idle-token gates**

From the control shell:

```bash
tmux has-session -t opd-CLI
tmux list-panes -t opd-CLI -F '#{pane_id} #{pane_current_command} #{pane_dead}'
squeue -j 429 -o '%.18i %.2t %.12L %R'
```

Inside `opd-CLI`, run without `srun`:

```bash
cd /home/mchen/FiRe-OPD/.worktrees/vanilla-sft-gradient-51200-opd
export CUDA_VISIBLE_DEVICES=0,1,2,3
printf 'PROFILE_GATE SLURM_JOB_ID=%s CUDA_VISIBLE_DEVICES=%s PWD=%s\n' "$SLURM_JOB_ID" "$CUDA_VISIBLE_DEVICES" "$PWD"
PYTHONPATH="$PWD/verl:$PWD" /home/mchen/miniconda3/envs/gvendi-opd/bin/python -c \
  'from math_eval.run_opd_proxy_gradient_verify import validate_opd_cli_runtime; print("TOKENS=" + ",".join(validate_opd_cli_runtime(expected_gpus=4, require_idle=True)))'
test -z "$(git status --porcelain=v1 --untracked-files=all)"
test ! -e /home/mchen/FiRe-OPD/logs/adaptive_concise_opd/profile-step1.log
test ! -e /home/mchen/FiRe-OPD/checkpoints/opd-adaptive-conciseprobe-cap50-tokenneutral-rawprompt-4gpu-tp4-profile-step1
```

Expected: Slurm job 429 is running with sufficient time, tokens are exactly `0,1,2,3`, all are idle, source is clean, and paths are absent.

- [ ] **Step 2: Launch the profile in tmux and wait by process/log condition**

Inside `opd-CLI`:

```bash
export ADAPTIVE_CONCISE_RUN_MODE=profile
bash run_train_adaptive_concise_token_neutral_opd.sh
rc=$?
printf 'ADAPTIVE_OOMSAFE_PROFILE_EXIT:%s\n' "$rc"
```

From the control shell, poll for the exit marker and process state. Do not use a fixed sleep as proof of completion. Require exit code zero and reject `Traceback`, `OutOfMemoryError`, or `Error executing job` in the log.

- [ ] **Step 3: Validate the profile and its provenance**

```bash
/home/mchen/miniconda3/envs/verl/bin/python \
  math_eval/validate_adaptive_concise_profile.py \
  --log /home/mchen/FiRe-OPD/logs/adaptive_concise_opd/profile-step1.log \
  --expected-step 1 \
  --expected-questions 1024 \
  --output /home/mchen/FiRe-OPD/logs/adaptive_concise_opd/profile-step1.acceptance.json
PROFILE_LOG=/home/mchen/FiRe-OPD/logs/adaptive_concise_opd/profile-step1.log
SOURCE_COMMIT=$(git rev-parse HEAD)
test "$(grep -m1 '^SOURCE_COMMIT=' "$PROFILE_LOG" | cut -d= -f2)" = "$SOURCE_COMMIT"
test "$(grep -o 'actor_rollout_ref.actor.entropy_from_logits_with_chunking=True' "$PROFILE_LOG" | head -1)" = \
  'actor_rollout_ref.actor.entropy_from_logits_with_chunking=True'
! grep -Eq 'Traceback|OutOfMemoryError|Error executing job' "$PROFILE_LOG"
sha256sum "$PROFILE_LOG" \
  /home/mchen/FiRe-OPD/logs/adaptive_concise_opd/profile-step1.acceptance.json
test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

Expected: validator reports `decision=pass`, all route/token/loss/IS/timing gates pass, and provenance equals the remediation commit.

---

### Task 5: Launch and accept the fresh formal training startup

**Files:**
- Create at runtime: `/home/mchen/FiRe-OPD/logs/adaptive_concise_opd/opd-adaptive-conciseprobe-cap50-tokenneutral-rawprompt-4gpu-tp4-step50.log`
- Create at runtime after step 1: `/home/mchen/FiRe-OPD/logs/adaptive_concise_opd/opd-adaptive-conciseprobe-cap50-tokenneutral-rawprompt-4gpu-tp4-step50.step1.acceptance.json`
- Future checkpoint: `/home/mchen/FiRe-OPD/checkpoints/opd-adaptive-conciseprobe-cap50-tokenneutral-rawprompt-4gpu-tp4-step50`

**Interfaces:**
- Consumes: accepted profile from Task 4, clean frozen source, idle allocation-owned tokens, and empty formal paths.
- Produces: a new active W&B/Ray formal run from step 0, with accepted step-1 metrics and no reuse of failed state.

- [ ] **Step 1: Re-run the formal launch gate**

Inside `opd-CLI`, after the profile exits:

```bash
cd /home/mchen/FiRe-OPD/.worktrees/vanilla-sft-gradient-51200-opd
export CUDA_VISIBLE_DEVICES=0,1,2,3
export ADAPTIVE_CONCISE_RUN_MODE=full
PYTHONPATH="$PWD/verl:$PWD" /home/mchen/miniconda3/envs/gvendi-opd/bin/python -c \
  'from math_eval.run_opd_proxy_gradient_verify import validate_opd_cli_runtime; print("TOKENS=" + ",".join(validate_opd_cli_runtime(expected_gpus=4, require_idle=True)))'
test -z "$(git status --porcelain=v1 --untracked-files=all)"
test ! -e /home/mchen/FiRe-OPD/logs/adaptive_concise_opd/opd-adaptive-conciseprobe-cap50-tokenneutral-rawprompt-4gpu-tp4-step50.log
test ! -e /home/mchen/FiRe-OPD/checkpoints/opd-adaptive-conciseprobe-cap50-tokenneutral-rawprompt-4gpu-tp4-step50
```

Expected: tokens idle, source clean, formal paths absent, and profile acceptance remains `decision=pass`.

- [ ] **Step 2: Launch the full run from step 0 without nested `srun`**

Inside `opd-CLI`:

```bash
bash run_train_adaptive_concise_token_neutral_opd.sh
rc=$?
printf 'ADAPTIVE_OOMSAFE_FULL_EXIT:%s\n' "$rc"
```

The command must remain active. Confirm the log says `RUN_MODE=full`, `trainer.resume_mode=disable`, source commit equals `git rev-parse HEAD`, and chunked actor entropy is enabled. Capture the new W&B run ID and Ray worker PIDs.

- [ ] **Step 3: Wait conditionally for and validate the first completed formal step**

Poll the log until a complete `step:1 - ...` metrics line appears or a fatal error/process exit occurs. Then run:

```bash
/home/mchen/miniconda3/envs/verl/bin/python \
  math_eval/validate_adaptive_concise_profile.py \
  --log /home/mchen/FiRe-OPD/logs/adaptive_concise_opd/opd-adaptive-conciseprobe-cap50-tokenneutral-rawprompt-4gpu-tp4-step50.log \
  --expected-step 1 \
  --expected-questions 1024 \
  --output /home/mchen/FiRe-OPD/logs/adaptive_concise_opd/opd-adaptive-conciseprobe-cap50-tokenneutral-rawprompt-4gpu-tp4-step50.step1.acceptance.json
```

Require `decision=pass`, exactly 1,024 questions, route identities, response residual 0, ratio 1.0, finite loss/gradient, IS max at most 5.0, finite entropy, and all timing fields.

- [ ] **Step 4: Confirm the accepted formal run remains active**

```bash
! grep -Eq 'Traceback|OutOfMemoryError|Error executing job' \
  /home/mchen/FiRe-OPD/logs/adaptive_concise_opd/opd-adaptive-conciseprobe-cap50-tokenneutral-rawprompt-4gpu-tp4-step50.log
ps -eo pid,stat,etime,cmd | grep -E '[v]erl\.trainer\.main_ppo|[r]ay::(TaskRunner|WorkerDict)'
squeue -j 429 -o '%.18i %.2t %.12L %R'
git status --porcelain=v1 --untracked-files=all
```

Expected: the full run is still advancing after accepted step 1, Slurm job 429 is running, source is clean, and no fatal signature is present. Leave the branch, worktree, tmux session, log, and checkpoint namespace untouched for continued monitoring through the explicit step-17 gate and final step-50 validation.
