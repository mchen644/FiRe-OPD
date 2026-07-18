# Adaptive Concise OPD Old-Log-Prob OOM Remediation Design

Date: 2026-07-18

Status: approved conversational design; awaiting review of this written specification

## 1. Objective

Remediate the GPU out-of-memory failure in the adaptive concise token-neutral OPD run without changing the experiment's scientific contract, routing behavior, training examples, teacher prompts, supervision masks, optimizer updates, or response-token accounting.

The remediation must preserve all artifacts from the failed run, must not resume from partial state, and must pass a fresh one-step GPU profile before a new 50-step run starts from step 0.

## 2. Incident and root cause

The failed formal run used source commit `ece7baaa1eb063798b867430ccfa533b2d4da777` and completed optimizer steps 1 through 16. During the attempted step 17, it failed before teacher/ref scoring and before the actor update in:

```text
actor_rollout_compute_log_prob
  -> dp_actor.compute_log_prob(calculate_entropy=True)
  -> _forward_micro_batch
  -> entropy_from_logits
  -> softmax(logits)
```

CUDA attempted to allocate 35.79 GiB while GPU 0 had 35.13 GiB free. The configured old-log-prob micro-batch size was four rows per GPU, padding removal was enabled, and responses could contain up to 16,384 tokens. A static micro-batch containing several long sequences produced a large packed logits tensor with shape proportional to `total_nonpadding_tokens × vocabulary_size`. The unchunked entropy implementation materialized full-vocabulary softmax intermediates for the entire packed tensor at once, causing the peak-memory failure.

This was not a routing, reward, token-budget, Slurm, or checkpoint error. Every completed step through step 16 reported exactly 1,024 questions, an exhaustive easy/learnable/hard partition, response-token residual 0, and response-budget ratio 1.0.

## 3. Considered approaches

### 3.1 Selected: built-in chunked entropy

Set:

```text
actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
```

VERL then computes entropy over bounded 2,048-token slices instead of materializing softmax intermediates for every packed token simultaneously. This directly addresses the allocation in the failing stack frame while retaining old-log-prob micro-batch size four.

The old log-prob values are computed by the unchanged log-prob path. The chunked calculation affects only the `actor/entropy` telemetry tensor. Entropy-aware distillation is disabled and `actor.entropy_coeff=0`, so student entropy does not affect advantages, loss, gradients, or parameter updates. The chunked implementation promotes each slice to float32, so the telemetry value may differ slightly from the previous bfloat16 calculation; this is an expected numerical diagnostic difference, not a treatment change.

### 3.2 Fallback only: old-log-prob micro-batch size one

Reducing `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu` from four to one would put a strict per-forward bound on packed sequence tokens. It is operationally conservative but increases forward-call overhead and changes a frozen throughput setting. It is therefore reserved for a separately tested fallback if chunked entropy fails the GPU profile or encounters another OOM.

### 3.3 Rejected for this retry: dynamic old-log-prob batching

Dynamic batching would balance variable sequence lengths and likely reduce peaks. However, the current partitioner derives a number of balanced micro-batches from the token target rather than enforcing that target as a strict upper bound for every resulting partition. It introduces more scheduling behavior than needed for this failure and is not selected.

## 4. Scope of implementation

The production change is limited to the adaptive concise launcher:

- explicitly enable actor chunked entropy;
- keep old-log-prob micro-batch size at four;
- keep teacher/ref entropy behavior unchanged;
- expose the setting in dry-run provenance;
- pin it with a launcher regression test so future edits cannot silently disable it.

No trainer, actor, routing, masking, reward, model, or entropy-function source code will change. No unrelated refactoring is in scope.

## 5. Preserved scientific contract

The retry retains:

- Qwen3-4B student and Qwen3-30B teacher;
- 1,024 distinct questions and 1,024 actor rows per step;
- one normal rollout per question;
- concise probes only for normal-correct rows;
- 50% per-row concise response cap;
- easy/learnable/hard routing and teacher-prompt rules;
- physical normal-prefix truncation before post-rollout model calls;
- exact per-row and batch response-token identity;
- 50 optimizer steps from step 0, seed 42, and unchanged data order;
- learning rate `1e-6`, reverse-KL-only advantages, token-mean loss, and token IS threshold 5.0;
- rollout old-log-prob micro-batch size four and actor update micro-batch size one;
- disabled TALE, legacy difficulty routing, candidate selection, rethinking probes, length penalty, G-Vendi, resume, and overwrite.

Chunked entropy is an execution-memory control, not an experiment treatment variable.

## 6. Artifact and namespace handling

Before retrying:

1. confirm that no training worker from the failed process remains;
2. preserve the failed log, acceptance record, launch contract, traceback, source commit, W&B run ID, and a checksum manifest under a timestamped `logs/adaptive_concise_opd/failed_attempts/` directory;
3. preserve prior successful profile artifacts in a separate immutable archive if their canonical paths must be reclaimed;
4. verify that no partial checkpoint exists;
5. never overwrite or resume any failed-run artifact.

After archival, the launcher may reclaim the canonical profile and formal filesystem paths atomically. W&B creates a new run ID even when the experiment display name is unchanged. The new log's printed `SOURCE_COMMIT` must identify the remediation commit.

## 7. Test-driven implementation

### 7.1 Red test

Extend the adaptive launcher test to require the exact dry-run override:

```text
actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
```

The test must be run against the current launcher and fail because the override is absent.

### 7.2 Minimal green change

Add only the selected override to the launcher's fixed arguments. Re-run the focused launcher test and confirm that its Hydra-composed config resolves the field to boolean `True`, while old-log-prob micro-batch size remains four.

### 7.3 CPU regression gate

Run:

- the focused adaptive launcher tests;
- all adaptive concise tests;
- the complete previously established CPU suite;
- Ruff on changed Python files;
- `bash -n` on the launcher;
- `compileall` for changed Python modules where applicable;
- `git diff --check`.

## 8. GPU profile gate

Run a fresh one-step profile inside tmux `opd-CLI` on allocation-owned logical GPU tokens `0,1,2,3`, without nested `srun`.

Acceptance requires:

- the launch contract records chunked entropy enabled and source commit exactly;
- old-log-prob completes and `actor/entropy` is finite;
- exactly 1,024 questions and actor rows;
- all route-count and teacher-route identities;
- concise cap compliance;
- response-token residual 0 and ratio 1.0;
- finite policy loss and gradient norm;
- finite rollout-correction metrics with no catastrophic rejection;
- a completed actor update and clean process exit;
- an acceptance JSON and checksum generated from the new profile log.

A one-step profile confirms integration and normal execution but does not reproduce every possible long-sequence grouping. The full retry must therefore receive an explicit monitoring gate at the attempted step 17 and continue to be checked for OOM through step 50.

## 9. Fresh formal run

Only after the profile passes:

1. launch a fresh 50-step run from step 0;
2. keep resume disabled and atomically claim empty output paths;
3. record the new W&B run ID, Slurm job/allocation, source commit, log checksum, and profile acceptance checksum;
4. validate each completed step's question count, route partition, token residual, budget ratio, loss/gradient health, and IS metrics;
5. explicitly inspect the transition through step 17, where the previous attempt failed;
6. at step 50, validate the checkpoint, `samples_yielded=51200`, cumulative routing counts, provenance, and final acceptance report.

If chunked entropy fails, do not stack additional changes onto the run. Stop, preserve evidence, and return to the fallback micro-batch-size-one design under a separate test and commit.

## 10. Claims and non-claims

Passing CPU tests proves only that the launcher contract is correct. Passing the one-step GPU profile proves that chunked entropy integrates with the treatment under that profile batch. Only progressing beyond the prior failing point provides direct runtime evidence that this incident is remediated, and only step-50 validation establishes completion of the experiment.

The retry may not be described as resumed: it is a fresh run from the initial student checkpoint and initial data order.
