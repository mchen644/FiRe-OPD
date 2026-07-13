# Gradient-Diverse Data-Only Group-Success OPD Ablation

## Decision

Run one 50-step group-success OPD experiment in which the only scientific
variable is the training parquet:

```text
existing run: full DeepMath pool, shuffled seed 42, first 12,800 question slots
new run:      fixed gradient-diverse DeepMath coreset, all 12,800 questions once
```

The group-success algorithm, models, rollout count, optimization settings,
teacher routing, supervision horizons, random seed, and evaluation protocol
remain unchanged.

This is a fast go/no-go experiment. It tests whether offline gradient-diverse
question selection can recover pass@1 and out-of-domain performance when the
`n=4` difficulty estimator reduces the number of unique training questions by
four relative to vanilla `n=1` OPD.

## Scientific question

At a fixed budget of 51,200 student rollouts, teacher scores, and actor
trajectories, does replacing the 12,800 questions encountered by group-success
OPD with the fixed gradient-diverse 12,800-question coreset improve held-out
strong-to-weak reasoning accuracy without materially worsening response length?

The primary contrast is:

```text
gradient-diverse 12.8k + group-success n=4
minus
seed-42 shuffled full pool + group-success n=4
```

The canonical vanilla OPD run remains a system-level fixed-compute anchor. It
is not a pure data-selection control because vanilla `n=1` sees approximately
51,200 unique question slots rather than 12,800.

## Fixed input artifact

The new training data is:

```text
/home/mchen/FiRe-OPD/data/gradient_diversity/DeepMath-103K/train_gradient_diverse_12800.parquet
```

Pinned properties:

```text
row count:          12,800
SHA-256:            caf303c5d151fdaed2e21eebc257917f13660c589906e3da0e569ffdfd59b059
unique stable IDs:  12,800
exact prompt count: 12,800
maximum Qwen3-4B prompt length with thinking disabled: 668 tokens
prompts over the 2,048-token training limit: 0
```

Its Arrow schema is exactly equal to the original training parquet schema, and
every output row is byte-logically equal to the source row identified by the
selection manifest. The authoritative provenance files are:

```text
data/gradient_diversity/selection/manifest.json
data/gradient_diversity/selection/selected_ids.jsonl
data/gradient_diversity/selection/diagnostics.json
```

The selector used Qwen2.5-0.5B gradients over R1 solutions only to choose source
rows. The proxy model and R1 solutions do not become student or teacher inputs
in this training experiment.

## One-variable training contract

The following values are identical to the completed group-success run:

```text
student model:                 models/Qwen3-4B
teacher/ref model:             models/Qwen3-30B-A3B-Instruct-2507
prompt batch size:             256 unique questions per step
rollouts per question:         4
trajectories per step:         1,024
PPO mini-batch size:           256 before rollout expansion
training steps:                50
question slots:                12,800
rollout/teacher/actor rows:     51,200
data shuffle seed:             42
rollout temperature/top-p:     1.0 / 1.0
maximum response length:       16,384
verifier success threshold:    reward > 0.5
easy route:                    exactly 4/4 correct
easy teacher / ESR:            concise / 0.20
non-easy teacher / ESR:        normal / 0.50
actor objective:               reverse-KL OPD only
entropy coefficient:           0
KL reward and KL loss:         0
candidate selection:           disabled
length-aware loss:             disabled
rethinking probe:              disabled
learning rate / warmup:        1e-6 / 0
resume mode:                   disable
checkpoint save frequency:     20 steps
```

The selected parquet has exactly `256 * 50` eligible rows. With the existing
seeded random sampler and `drop_last=True`, one 50-step run consumes every
coreset row exactly once and does not cross an epoch boundary.

Allowed differences are limited to:

- `data.train_files` / `TRAIN_DATA`;
- experiment, checkpoint, and log names;
- artifact provenance emitted before launch.

No training implementation or algorithm configuration is changed for this
ablation.

## Production entry point

Add a thin fail-closed wrapper around
`run_train_group_success_difficulty_opd.sh`. The wrapper must:

1. require the selected parquet at the pinned absolute path;
2. verify its SHA-256 and row count before allocating training work;
3. pin the one-variable training contract above;
4. use the production name
   `opd-n4-graddiv12800-easy4of4-concise20-noneasynormal50-purerkl-step50`;
5. refuse an existing non-empty checkpoint directory;
6. expose a dry-run mode that prints the fully resolved downstream command;
7. delegate to the existing group-success launcher without adding training
   logic.

Before launch, produce resolved commands for the original full-pool setting
and the new coreset setting under the same current worktree. After normalizing
the train path, experiment name, checkpoint path, and log path, the commands
must be identical.

Because the repository currently contains unrelated uncommitted training
changes, the run record must include:

```text
git HEAD
git status --short
SHA-256 of the current textual git diff
SHA-256 of the base and ablation launchers
selected parquet SHA-256
fully resolved launch command
```

If the current dirty snapshot cannot be shown to match the snapshot used for
the completed group-success baseline, the old run may be used for exploratory
go/no-go comparison only. A publication-grade causal claim then requires a
same-snapshot random-data control.

## Preflight and launch

The preflight must verify:

- the parquet has 12,800 rows and no nulls in required columns;
- all 12,800 exact prompts are unique;
- all prompts survive the production Qwen3-4B 2,048-token filter;
- the resolved command changes only the allowed fields;
- all group-success unit and launcher tests pass;
- the requested four GPUs belong to the active `opd-CLI` Slurm allocation and
  are idle.

Launch the production command from the existing `opd-CLI` session. Do not run
it from the login shell or silently resume any prior checkpoint.

## First-step acceptance gates

Monitor through the first completed optimizer step and require:

- dataset length after filtering is exactly 12,800;
- 256 distinct `uid` groups and 1,024 trajectories are present;
- every group contains exactly four rows;
- every group receives one shared route;
- easy, learnable, and unresolved bucket ratios sum to one;
- concise rows use ESR 0.20 and all other rows use ESR 0.50;
- the actor loss contains only reverse-KL OPD advantages;
- entropy, KL reward/loss, candidate selection, and length-aware terms remain
  absent or zero;
- the actor update completes without shape, divisibility, or OOM errors;
- checkpoint and W&B names use the ablation production name.

After these gates pass, leave the run active through step 50. Training-batch
reward is diagnostic only and is not evidence of generalization.

## Evaluation contract

Merge or expose the step-50 Hugging Face checkpoint using the same process as
the completed group-success baseline. Evaluate with the same strong-to-weak
protocol:

```text
samples per problem: 32
temperature:         1.0
top-p:               1.0
maximum tokens:      16,384
seed:                42
extra prompt:        disabled
```

Use the same benchmark files and scoring code as the canonical step-50
strong-to-weak evaluation. The predeclared suite is exactly:

```text
AIME 2024
AIME 2025
HMMT February 2025
HMMT November 2025
MATH500
MinervaMath
OlympiadBench
AMC 2023
```

The aggregate is the unweighted mean of the eight per-dataset metrics. Report:

- pass@1, computed as the mean correctness over all 32 sampled responses;
- pass@32;
- mean and median response length;
- per-dataset values and the predeclared macro average;
- deltas against the completed group-success run and canonical vanilla OPD.

The primary success signal is a consistent increase in pass@1 on held-out
datasets relative to the existing group-success run. Pass@32 alone is not
sufficient because the motivating failure was poor one-sample reliability
despite similar high-sample coverage.

## Interpretation

Possible outcomes are interpreted narrowly:

- **Improves over group-success and narrows the vanilla gap:** offline
  selection helps compensate for fewer unique questions.
- **Improves only pass@32:** selection adds coverage but does not fix policy
  concentration or one-sample reliability.
- **No meaningful improvement:** the proxy-gradient coreset does not solve the
  diversity bottleneck under this algorithm, or proxy mismatch dominates.
- **Worse:** sparse-cluster prioritization may emphasize outliers/noise or
  distort the useful training distribution.

This single run does not establish that a gain is caused specifically by
gradient-space diversity. The selected set changes leaf-topic composition,
and the two clustering seeds have low label/selection stability. A positive
go/no-go result motivates a later same-snapshot matrix with uniform-random and
metadata-matched 12,800-question controls; it does not replace those controls.

## Non-goals

This experiment does not:

- change the group-success threshold or difficulty estimator;
- introduce adaptive rollout allocation;
- change teacher prompts or ESR fractions;
- add entropy, length rewards, candidate filtering, or peer conditioning;
- retrain the selector or select a second coreset;
- claim publication-grade selector causality from one training seed;
- reuse training score as held-out evaluation.
