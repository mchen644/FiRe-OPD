# Vanilla OPD on a 51,200-Question SFT-Gradient-Diverse Pool

Date: 2026-07-16

## Decision

Run one new Vanilla OPD training experiment whose training set contains exactly
51,200 unique questions selected by the existing SFT-gradient diversity
pipeline. The selected question count matches the 51,200 question samples
consumed by the canonical Vanilla OPD checkpoint at step 50.

The experiment has two phases:

1. reuse the frozen projected SFT-gradient artifacts to publish a new immutable
   51,200-question selection; and
2. train Vanilla OPD for exactly 50 optimizer steps with batch size 1,024 and
   one rollout per question.

Final benchmark evaluation is deliberately deferred until training and the
step-50 checkpoint have completed.

## Scientific question

At the same 51,200-question, 51,200-rollout, and 50-update budget as canonical
Vanilla OPD, does replacing the seed-42 shuffled full-pool sample with a
51,200-question SFT-gradient-diverse sample improve the eventual held-out math
performance?

This stage produces the candidate checkpoint needed to answer that question.
It does not yet evaluate the checkpoint.

## Existing Vanilla reference

The existing comparison checkpoint is:

```text
/home/mchen/FiRe-OPD/checkpoints/opd-strong-to-weak-rawprompt-4gpu-tp4-refmb4-rollmb4/global_step_50
```

Its checkpointed data-loader state records:

```text
samples_yielded:       51,200
sampler batches:       50
training epoch:        0
questions per batch:   1,024
rollouts per question: 1
```

Because the source pool has 57,046 rows and the checkpoint is still in its
first shuffled epoch, these 51,200 samples are distinct question rows. The new
candidate therefore also uses exactly 51,200 distinct rows rather than
repeating the prior 12,800-question coreset four times.

The historical reference was trained from an older repository snapshot whose
complete dirty state was not frozen. A later comparison against it is useful
for directional screening, but it is not a publication-grade data-only causal
contrast. A same-snapshot full-pool rerun would be required for that stronger
claim. No such rerun is part of this task.

## Frozen SFT-gradient inputs

The new selection reuses these existing artifacts without modification:

```text
source parquet:
  /home/mchen/FiRe-OPD/data/g-opd/DeepMath-103K/train_filtered_level6.parquet
source rows:
  57,046
source SHA-256:
  de3350fdd00bc0410550098ea65179e2be873da99e4075f80de575fc17670597

prepared rows:
  57,046
prepared JSONL SHA-256:
  ee8d55c943577888f14052b3953a0a8aa07d26076e50ddffa41a102b60027344
prepared manifest SHA-256:
  5add2e965473d3647f2318c73f957434fbe23344516089be9adfe89db9c6a70e
gradient-eligible rows:
  57,045
eligibility report SHA-256:
  cc8d5b888def37761519e15c6ef722e5a22d91ae2342c6d669f008bcff3dcbb6
excluded stable ID:
  deepmath-level6-038794

projected-gradient directory:
  /home/mchen/FiRe-OPD/data/gradient_diversity/gradients/qwen2.5-0.5b-instruct
gradient manifest SHA-256:
  9a534118a08e736a15e806933d5cf90c2a99822fee28bf18644e5ff344ce3ae2
```

The representation was produced with:

```text
proxy model:                 Qwen/Qwen2.5-0.5B-Instruct
proxy revision:              7ae557604adf67be50417f59c2c2f167def9a775
supervision:                 r1_solution_1 completion-only cross-entropy
full parameter gradients:    yes
projection:                  Rademacher TRAK
projection dimension:        1,024
projection seed:             0
stored dtype:                float32
official reference commit:   d9484cd3b5991030b901ac4a3a9e2472dbfac2ad
official reference tree:     a0079d8c5e15cb18bb4790f99c43cc19bb9ecd50
```

No gradient collection, model forward pass, or backward pass is repeated.

## First-class 51,200 selection profile

The selector gains a named `vanilla_51200` profile while retaining the old
12,800 profile and its defaults unchanged.

The new profile uses the same selection semantics as the prior coreset:

```text
eligible candidates:              57,045
selected unique questions:        51,200
primary cosine K-means ratio:      0.10
primary cluster count:             5,704
primary K-means seed:              42
K-means iterations:                20
balanced round-robin seed:         42
selection replacement:             false
```

The diagnostic runs remain the same four combinations:

```text
ratios: 0.10 and 0.01
K-means seeds: 42 and 43
```

G-Vendi, cluster statistics, topic/difficulty distributions, AMI/ARI where
already defined, and selected-set overlap remain diagnostics. G-Vendi does not
replace K-means plus balanced round-robin as the selection rule.

The full cluster labels were not persisted by the 12,800 run, so clustering is
rerun from the frozen projected gradients on one GPU. Gradient computation is
not rerun.

### Prefix invariant

For fixed labels and seed, balanced round-robin selection is prefix-stable with
respect to the requested target size. The first 12,800 IDs of the new 51,200
selection must therefore exactly equal the existing selected-ID sequence:

```text
/home/mchen/FiRe-OPD/data/gradient_diversity/selection/selected_ids.jsonl
```

A mismatch fails closed before publication. This catches a changed clustering
result, seed, row order, or selection implementation.

## New immutable selection artifacts

The new profile publishes only to a separate namespace:

```text
/home/mchen/FiRe-OPD/data/gradient_diversity/DeepMath-103K/train_gradient_diverse_51200.parquet
/home/mchen/FiRe-OPD/data/gradient_diversity/selection_51200/manifest.json
/home/mchen/FiRe-OPD/data/gradient_diversity/selection_51200/selected_ids.jsonl
/home/mchen/FiRe-OPD/data/gradient_diversity/selection_51200/diagnostics.json
```

Publication is lock-protected, append-only, and atomic. A partial artifact set,
a hash mismatch, or existing bytes that differ from the recomputed result is an
error. The old 12,800 artifacts must remain byte-identical to these frozen hashes:

```text
parquet:      caf303c5d151fdaed2e21eebc257917f13660c589906e3da0e569ffdfd59b059
manifest:     a1a45382ee577e24f9b455386f9f3adcab210ff40a83ea760c2110fb96f8f90a
selected IDs: a78c04cff10c148f45bf9c828e07cc9cf01398b48efeb707253273af359a037e
diagnostics:  d2105179f2ce5a7c796aef07136bbc1dce79b07b5bc2de4d2319b18b47ea761f
```

The new manifest binds:

- source, prepared-pool, eligibility-report, and gradient-manifest hashes;
- exact reference repository commit and tree;
- all projection, clustering, diagnostic, and selection parameters;
- old 12,800 selected-ID hash used by the prefix check;
- exact 51,200 selected-ID sequence hash;
- output parquet, selected-ID, and diagnostics hashes; and
- the committed selector source identity.

The selected parquet must have exactly the source Arrow schema and metadata.
Every row must equal the source row named by its stable ID and source index.
All 51,200 stable IDs, source indices, and exact prompts must be unique.

## Vanilla OPD training contract

The candidate uses a clean, committed execution worktree and a fail-closed
launcher. The launcher renders both a full-pool baseline command and the
candidate command from one canonical Vanilla OPD command builder. After
normalizing only the training-data path, experiment name, checkpoint path, and
log path, the commands must be byte-identical.

The scientific settings are:

```text
student model:                       models/Qwen3-4B
teacher/ref model:                   models/Qwen3-30B-A3B-Instruct-2507
student prompt:                      original raw OPD prompt
teacher/ref prompt:                  same original raw OPD prompt
prompt batch size:                   1,024
rollouts per question:               1
trajectories per step:               1,024
PPO mini-batch size:                 1,024 before worker expansion
optimizer steps:                     50
question/rollout/trajectory rows:    51,200
data shuffle:                        true
data seed:                           42
maximum prompt length:               2,048
maximum response length:             16,384
rollout temperature / top-p:         1.0 / 1.0
advantage estimator:                 GRPO
actor objective:                     Vanilla OPD reverse-KL only
loss aggregation:                    token mean
token-level importance sampling:     enabled
importance-sampling threshold:       5.0
learning rate / warmup:              1e-6 / 0
entropy coefficient:                 0
KL reward coefficient:               0
KL loss coefficient:                 0
length-aware OPD:                    disabled
difficulty-aware routing:            disabled
TALE/ESR budgeting:                  disabled
candidate selection:                 disabled
rethinking probe:                    disabled
actor parameter/optimizer offload:   disabled
reference parameter offload:         enabled
rollout tensor parallelism:          4
total epoch budget:                  3 (with an explicit stop at step 50)
training GPUs:                        4
checkpoint save frequency:           step 50
resume mode:                          disabled
```

The run stops after step 50. The historical checkpoint was saved at step 50
before its longer epoch budget continued; with zero warmup and the fixed
learning rate, stopping at that comparison point avoids unnecessary later
updates.

### Training-time validation

To preserve the canonical Vanilla run's generation/RNG path, retain:

```text
validation before training: true
validation frequency:       every 10 steps
validation datasets:        AIME 2024 and AIME 2025
samples per problem:        8
validation temperature:     1.0
validation top-p:           1.0
```

This training-time validation is part of the matched training contract. It is
not the deferred final benchmark evaluation.

## Production identity and paths

The production run name is:

```text
opd-strong-to-weak-sftgrad51200-rawprompt-4gpu-tp4-refmb4-rollmb4
```

Its outputs are:

```text
checkpoint:
  /home/mchen/FiRe-OPD/checkpoints/opd-strong-to-weak-sftgrad51200-rawprompt-4gpu-tp4-refmb4-rollmb4
training log:
  /home/mchen/FiRe-OPD/logs/gradient_diversity/opd-strong-to-weak-sftgrad51200-rawprompt-4gpu-tp4-refmb4-rollmb4.log
```

A missing or empty checkpoint directory is accepted. Any non-empty checkpoint
directory is rejected; the experiment never resumes or overwrites an earlier
attempt.

## Isolation and provenance

Implementation starts from local `main` at
`19578a8550a2638840df2bb20149f465dc10c5e2` in the clean branch/worktree:

```text
branch:   vanilla-sft-gradient-51200-opd
worktree: /home/mchen/FiRe-OPD/.worktrees/vanilla-sft-gradient-51200-opd
```

The root checkout's unrelated uncommitted changes are never switched, stashed,
or modified. Selection and training execute from an exact committed worktree
HEAD. Runtime data, models, logs, and checkpoints use explicit absolute paths
under `/home/mchen/FiRe-OPD`.

Before each GPU phase, provenance records:

- Git HEAD and tree;
- `git status --short` and textual diff SHA-256;
- launcher and selector source hashes;
- every input artifact hash;
- resolved environment and command;
- Python executable and relevant package versions; and
- Slurm job, tmux session, visible physical GPUs, and CUDA mapping.

No runtime package installation is allowed.

## Preflight and GPU execution

CPU preflight must complete before allocation use. It validates all frozen
input hashes, exact row coverage, old artifact immutability, output isolation,
command equivalence, selected-data cardinality, and an empty checkpoint target.

The selection phase requires exactly one visible allocation-owned GPU because
the pinned official K-means implementation is CUDA-only. It reruns clustering
and publishes the new selection atomically.

The training phase requires four idle allocation-owned GPUs in the authorized
`opd-CLI` Slurm shell. It is launched directly from that shell without nested
`srun`. GPU launch occurs only after the implementation, tests, dry-run
contract, artifact preflight, and explicit execution authorization are all
present.

## First-step and completion gates

The first completed optimizer step must show:

- dataset length exactly 51,200 after prompt filtering;
- batch size 1,024 and exactly 1,024 distinct question IDs;
- one trajectory per question;
- Vanilla OPD reverse-KL advantages only;
- token-mean aggregation and token-level IS threshold 5.0;
- no difficulty, TALE, candidate-selection, length-aware, rethinking, entropy,
  KL-reward, or effective KL-loss contribution;
- a completed actor update with finite loss and gradient norm; and
- the exact production checkpoint and W&B names.

After acceptance, the run continues through step 50. Completion requires:

- one completed `training/global_step:50` record;
- a complete four-rank actor checkpoint at `global_step_50`;
- checkpointed data-loader state with `samples_yielded=51200` and 50 batches;
- no remaining training/Ray process owned by the run; and
- a final provenance and artifact inventory.

Training reward and training-time validation are diagnostics only and are not
treated as the held-out result.

## Error handling

The workflow fails closed when:

- any frozen input or prior 12,800 artifact hash changes;
- gradient coverage is incomplete, duplicated, reordered, non-finite, or zero;
- clustering/reference provenance differs from the pinned contract;
- the 51,200 selection is not unique or violates the 12,800 prefix invariant;
- output artifacts are partial, pre-existing with different bytes, or overlap
  an input namespace;
- source-row equality, Arrow schema, prompt uniqueness, or token limits fail;
- normalized baseline/candidate training commands differ outside allowed paths
  and names;
- the checkpoint directory is non-empty;
- the worktree is dirty or its HEAD differs from recorded provenance;
- GPU ownership/visibility/idleness checks fail; or
- first-step acceptance does not pass.

A failed training attempt is archived with metadata and logs and is never
silently resumed. Selection and training artifacts from a failed source
identity cannot satisfy a later run.

## Testing strategy

Implementation follows TDD. Focused CPU tests cover:

- the named 12,800 and 51,200 selection profiles;
- unchanged default behavior and frozen old-profile contract;
- deterministic balanced-round-robin prefix behavior;
- exact 51,200 cardinality, uniqueness, source identity, and schema;
- rejection of altered input hashes and partial publication;
- prefix mismatch rejection;
- dry-run Vanilla command construction;
- normalized one-variable command comparison;
- rejection of every mutable scientific override;
- empty-checkpoint enforcement; and
- provenance/report parsing.

Regression verification uses `gvendi-opd` for preparation/selection tests and
`verl` for training-launcher and trainer tests. Shell syntax, Python
compilation, Ruff, focused tests, and the relevant broader suites must pass
before GPU work. A one-GPU clustering preflight precedes full selection, and a
first-step gate precedes unattended training.

## Deferred work and non-goals

This task does not:

- recompute SFT gradients;
- modify or replace the old 12,800 coreset;
- repeat selected questions to reach 51,200 slots;
- alter Vanilla OPD behavior or add a new loss/routing method;
- rerun the historical full-pool Vanilla baseline;
- run the final n=32 benchmark suite;
- claim improved downstream performance from training metrics; or
- claim publication-grade selector causality from one candidate run and a
  historical control.

Final evaluation design and execution begin only after the new step-50
checkpoint passes all completion gates.
