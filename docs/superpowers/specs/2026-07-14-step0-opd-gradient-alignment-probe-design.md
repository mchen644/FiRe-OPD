# Step-0 OPD Gradient Alignment Probe

## Decision

Build and run a diagnostic that compares the existing gradient-diverse
DeepMath coreset with the exact seed-42 random questions used by the previous
group-success control, but measure both sets in a new gradient space derived
from the real Step-0 OPD objective.

The probe uses the pristine `models/Qwen3-4B` student. It does not use either
trained step-50 checkpoint. Student parameters remain unchanged throughout the
capture run. This prevents the selected-data training run from defining the
space in which its own data are evaluated.

The chosen implementation is a two-stage full-parameter replay with two
independent rollout realizations per question:

1. run the production rollout, verifier, difficulty routing, teacher, ESR, and
   rollout-correction path with actor learning rate zero, and capture the exact
   tensors immediately before the actor update;
2. replay the production OPD loss from those frozen tensors with the pristine
   Qwen3-4B actor, accumulate four sequential trajectory backwards per prompt
   group, and project the full-parameter group gradient to a
   1,024-dimensional sketch.

This is preferred over a last-block proxy because the scientific question is
whether the original SFT-gradient representation transfers to the actual OPD
update. It is preferred over modifying FSDP communication hooks because a
standalone replay is much easier to validate and cannot perturb the training
communication path. A small real-data smoke test must demonstrate that the
full gradient and projection fit on one H200 before the production probe is
allowed to start. There is no silent fallback to a last-layer or LoRA proxy.

## Scientific question

The existing selector represents question `i` by a projected full-parameter
completion-only SFT gradient:

```text
Qwen2.5-0.5B + fixed r1_solution_1 + completion CE
```

The training algorithm updates a different model using student-generated
tokens, teacher likelihoods, adaptive prompts, adaptive ESR horizons, and the
PPO reverse-KL surrogate:

```text
Qwen3-4B student rollout + Qwen3-30B teacher log-prob + OPD/PPO
```

The primary question is:

> Does the selected coreset remain more diverse than the seed-42 random
> control when diversity is measured with Step-0 full-parameter OPD gradients?

The secondary question is:

> Do the SFT-gradient and OPD-gradient spaces preserve the same pairwise
> relationships among questions?

The diagnostic separates three possible explanations for the failed accuracy
ablation:

1. **representation mismatch:** SFT diversity does not transfer to OPD;
2. **weak OPD signal:** directions transfer, but selected examples have lower
   gradient norms, smaller teacher-student gaps, or fewer supervised tokens;
3. **downstream mismatch:** OPD gradients transfer, so data quality, density,
   benchmark relevance, or evaluation power is the more likely bottleneck.

## Frozen training contract

The capture path must inherit the scientific settings of
`opd-n4-easy4of4-concise20-noneasynormal50-purerkl-step50`:

```text
student:                         models/Qwen3-4B
teacher:                         models/Qwen3-30B-A3B-Instruct-2507
rollouts per question:           4
rollout temperature / top-p:     1.0 / 1.0
maximum response length:         16,384
verifier success threshold:      reward > 0.5
easy route:                      exactly 4/4 correct
easy prompt / ESR:               concise / 0.20
non-easy prompt / ESR:           normal / 0.50
actor objective:                 reverse-KL OPD PPO surrogate
loss aggregation in production:  token-mean with micro-batch size 1
rollout importance sampling:     token-level, upper threshold 5.0
entropy coefficient:             0
KL loss coefficient:             0
candidate selection:             disabled
length-aware OPD:                disabled
```

Probe-only changes are:

```text
actor learning rate:             0
unique prompt groups:            512
independent vLLM engine seeds:    42 and 43
captured group realizations:     1,024
capture lifecycle:               2 separate trainer/vLLM launches
steps per launch:                2 × 256 prompt groups
data order:                      deterministic, arm-balanced
actor checkpoint saving:         disabled
W&B logging:                     disabled
```

The actor parameter hash must be identical before and after capture. Optimizer
state may change, but it is not an input to replay and is not saved.

The two rollout seeds cannot share one initialized vLLM engine: the synchronous
rollout implementation consumes `rollout.seed` only in the `LLM(...)`
constructor, not in per-call sampling parameters. The top-level capture
launcher therefore starts two separate trainer processes, each with a freshly
initialized vLLM engine and a seed-specific output namespace. Seed 42 runs its
two arm-balanced batches and exits before seed 43 is launched. Both launches
must start from the same pristine actor hash, sample-manifest hash, data order,
and frozen configuration; relabeling later steps from one engine is forbidden.

## Prompt populations and sampling

Define the two full populations before drawing any probe sample:

- `S`: the 12,800 stable IDs in
  `data/gradient_diversity/selection/selected_ids.jsonl`;
- `R`: the first 12,800 source rows from the production seed-42 PyTorch
  `RandomSampler`, reconstructed as
  `torch.randperm(57046, generator=Generator().manual_seed(42))[:12800]`.

The random-control reconstruction is accepted only under `torch==2.6.0`, with
all of these pinned checks:

```text
first five source indices:       54434, 23328, 20314, 25544, 20200
compact-JSON index-list SHA256:  e32287327ee446433397d0412ecaae81724b0d7391ac42b29a13959e260aaaa5
ordered-prompt SHA256:           f16e847834161d68fd590b4bc98356f78740fb8406a85e71ed73e3a4b5d458bc
```

The reconstruction manifest also records the source parquet hash and verifies
that the sole proxy-ineligible source row, index 38,794, is not in `R`.

Draw 256 IDs uniformly without replacement from `S` and 256 IDs uniformly
without replacement from `R`, using probe seed `20260714`. If an ID is drawn
for both arms, keep it in the selected arm and deterministically draw the next
unused random-arm ID. This makes the 512 capture rows unique while changing at
most the few realized cross-arm collisions rather than excluding the entire
`S ∩ R` population.

Interleave selected and random rows so each 256-question capture batch contains
128 rows from each arm. Sampling membership, source row index, stable ID,
topic, difficulty, R1-solution length, and all source hashes are written before
GPU work. The primary comparison is the unmodified arm contrast. Topic,
difficulty, and R1-solution length are reported as pre-treatment composition
diagnostics but do not define an alternative adjusted endpoint. Post-treatment
variables such as group success, rollout length, ESR tokens, loss, or gradient
norm must not be matched away.

The independent statistical unit is one prompt group. The four rollouts of one
question are never treated as four independent observations. Each of the 512
question IDs is captured twice, once with vLLM seed 42 and once with seed 43,
using identical arm-balanced order and an unchanged actor. Replicate seeds are
an OPD-gradient reliability measurement, not additional independent questions.

## Capture data flow

An opt-in driver-side capture hook is placed after teacher log-probability and
rollout importance weights have been computed and immediately before
`update_actor(batch)`. A second opt-in actor-side diagnostic records the tensors
constructed by the update-time forward. With capture disabled, neither helper
is imported or called and the existing trainer tensors, control flow, and
outputs remain unchanged.

For every trajectory, capture:

- stable source ID, prompt-group UID, arm, capture step, and rollout slot;
- actor `input_ids`, `responses`, `attention_mask`, and `position_ids`;
- final `response_mask` and `tale_budget_esr_loss_mask`;
- original response length, ESR beta, ESR token count, route, prompt style,
  correct count, and verifier result;
- the actual teacher prompt messages used for reference scoring;
- `ref_log_prob`, recomputed actor `old_log_probs`, `rollout_log_probs`, and
  `rollout_is_weights`;
- actor/ref tokenizer hashes and all loss/configuration fields needed to replay
  the objective.

The actor-side diagnostic is keyed by stable ID, rollout seed, and rollout slot
and records, without modifying the loss:

- update-time `log_prob.detach()` and the resulting stopped reverse-KL
  advantage;
- the exact `loss_response_mask` after multiplying `response_mask` by
  `tale_budget_esr_loss_mask`;
- PPO ratio, clip branch, rollout-IS-weighted token loss, per-trajectory masked
  token-mean loss, and loss scale factor;
- the existing reduced/logged actor `pg_loss`, an explicit objective-scale
  batch `pg_loss`, gradient norm, and clip factor returned by the unmodified
  update.

For one 256-group/1,024-trajectory capture step, let `ell_j` be the unscaled
masked token-mean `pg_loss` of trajectory `j`. Each rank has 256 microbatches,
so production backwards `ell_j / 256`; FSDP then averages four ranks. Pin these
three scalar definitions in the artifact:

```text
logged_actor_pg_loss = mean_over_1024(ell_j / 256)
objective_pg_loss    = mean_over_4_ranks(sum_local_256(ell_j / 256))
                     = mean_over_1024(ell_j)
objective_pg_loss    = 256 * logged_actor_pg_loss
```

The first quantity preserves the current veRL metric-reduction behavior. The
second is a new capture-only diagnostic computed without changing backward or
the public metric. The equality, including its factor of 256, is an acceptance
check; replay is never compared to the logged scalar without this conversion.

Driver and actor artifacts must join one-to-one on all 4,096 trajectories:
512 IDs × 2 rollout seeds × 4 rollout slots. Replay validation uses the
actor-side values as its real production reference rather than comparing only
with the earlier `old_log_probs` tensor.

The production trainer physically truncates response-aligned tensors to the
maximum ESR prefix before teacher scoring. The capture therefore stores the
already-truncated tensors and row masks, rather than attempting to reconstruct
them from decoded text.

Each seed/step capture is written atomically as a tensor artifact plus a JSON
sidecar. A manifest binds the artifacts to the repository snapshot, dirty diff
hash, models, input parquet, sample manifest, launcher, configuration, package
versions, and ordered IDs. Existing artifacts are resumed only when every
provenance field matches exactly.

## Exact replay objective

For response token `t` of trajectory `j`, define:

```text
m_jt = response_mask_jt * tale_budget_esr_loss_mask_jt
A_jt = stopgrad(ref_log_prob_jt - current_actor_log_prob_jt)
r_jt = exp(clamp(current_actor_log_prob_jt
                  - stopgrad(current_actor_log_prob_jt), -20, 20)) = 1
w_jt = captured rollout importance weight
```

The replayed `m_jt` must be elementwise equal to the actor-side captured
`loss_response_mask`. Any mismatch is a failed replay, not a recoverable sample.

The replay must call the existing vanilla PPO policy-loss implementation or an
independently tested algebraic equivalent. It must not use
`(student_logp - teacher_logp).mean()`, because that incorrectly allows the
advantage to participate in backpropagation.

Production uses micro-batch size one. Replay therefore runs four sequential
forward/backward calls, clearing activations after each trajectory while
accumulating parameter gradients. Each trajectory loss is multiplied by
`1/4`. It must not put all four trajectories in one simultaneous computation
graph. The resulting gradient is the arithmetic mean of four individual
masked token-mean gradients:

```text
g_group = (1/4) * sum_j gradient(
              masked_token_mean(PPO_loss_jt, m_jt)),  j = 1..4
```

Each trajectory contributes with global scale `1/1024` in the 1,024-trajectory
production batch. After the four trajectories are averaged into `g_group`, the
corresponding prompt-group contribution has scale `1/256`. This positive group
scale is omitted from the stored gradient and does not affect cosine geometry.
All norm comparisons use the same unscaled group definition and label it
explicitly; they are not reported as post-optimizer AdamW contributions.
For replay-equivalence validation only, the mean of the 256 replayed group
losses is the production 1,024-trajectory `objective_pg_loss`. Compare it to
the capture-only objective diagnostic and independently verify that it equals
`256 * logged_actor_pg_loss`; do not compare it directly to the existing
logged scalar.

The pristine model is reloaded from the pinned base-model path for replay.
Forward precision, tokenizer/template, response token IDs, position IDs, and
masks must match capture. No optimizer step is performed.

The standalone replay mirrors the production FSDP precision contract rather
than merely matching its scalar loss: model master parameters and `.grad`
buffers are FP32, forward computation uses BF16 autocast, and reductions and
the accumulation across all four backwards remain FP32. Gradients must never
be cast to BF16 before norm computation or projection. No gradient scaler is
used. The manifest records parameter dtype, autocast dtype, gradient dtype,
and accumulation dtype; a mismatch across shards is fatal.

## Full-gradient projection

After all four sequential trajectory backwards for one group:

1. traverse trainable parameters in a deterministic named-parameter order;
2. validate that every expected parameter has a finite gradient;
3. accumulate the FP32 squared norm;
4. stream or flatten only the current group's gradient;
5. apply a seed-0 Rademacher/fast-JL projection to dimension 1,024;
6. divide by `sqrt(1024)` consistently with the existing Prismatic artifacts;
7. save the FP32 sketch and metadata, then clear gradients before the next
   group.

The collector must use projection interval one. It must never retain four
full 4B gradients as the reference 0.5B collector does. Four independent
processes handle deterministic ID shards, one process per allocated H200.

The production run is blocked until an eight-question smoke test (four IDs per
arm, two rollout seeds, four rollouts per seed) proves:

- one full group backward and projection fits on one H200;
- projected shape is exactly `(1024,)`, finite, and non-zero;
- every group contains exactly four rollout slots;
- parameter order and parameter count are identical across all processes;
- every replayed update-time log-probability, final loss mask, per-trajectory
  scalar loss, aggregate step loss, and token count matches the actor-side
  diagnostic: masks/tokens exactly, log-probabilities with
  `atol=0.02, rtol=0.01`, and scalar losses with
  `atol=0.001, rtol=0.01`;
- restarting skips only complete, provenance-matching group artifacts;
- the measured rate yields a completion estimate within the active allocation.

If full projection OOMs or fast-JL cannot process one group, stop and report the
failure. Switching to a last-block, LoRA, or FSDP communication-hook method
requires a new explicit decision.

## Analysis

Load the existing 1,024-dimensional Qwen2.5 SFT sketches and both new
1,024-dimensional Qwen3 OPD replicate sketches for exactly the same 512 stable
IDs. Let `g_i^(42)` and `g_i^(43)` be the two OPD group sketches. The primary
OPD representation is their arithmetic mean before normalization:

```text
g_i_bar = (g_i^(42) + g_i^(43)) / 2
```

Both OPD replicates use the same projection seed and identical parameter-order
manifest, so averaging their sketches is the linear projection of the averaged
full gradients. A parameter-order or projection-contract mismatch is fatal.

Vectors from different parameter spaces are never compared coordinate by
coordinate.

Generate and persist all statistical resampling index tables before computing
metrics, using NumPy `Generator(PCG64(seed))`:

```text
OPD-replicate within-arm permutations:  2026071501
diversity arm-label permutations:       2026071502
shared arm-stratified ID bootstraps:     2026071503
within-arm half-samples:                 2026071504
SFT-to-OPD within-arm permutations:      2026071505
```

The shared 2,000 bootstrap index table is reused for directional diversity,
`CKA_excess`, and both signal ratios so metric differences are not caused by
different resamples. Each resampling artifact records its RNG implementation,
seed, shape, and SHA256.

### OPD reliability gate

Before interpreting SFT-to-OPD alignment, measure whether OPD itself is stable
across the two independent rollout realizations. Form one cosine Gram matrix
per OPD seed. To prevent the selected/random arm label from creating apparent
alignment, the primary relational statistic is arm-conditioned. For a Gram
matrix `K`, define:

```text
B(K) = blockdiag(H_S K_SS H_S, H_R K_RR H_R)
H_a  = I_256 - (1/256) 11^T
block_CKA(K, L) = <B(K), B(L)>_F /
                  (||B(K)||_F ||B(L)||_F)
```

Thus cross-arm entries are excluded and each within-arm block is centered.
The primary Spearman statistic is the arithmetic mean of the two arm-specific
upper-triangle Spearman correlations. Neighbor sets are also constructed only
among IDs in the same arm. Report:

- replicate-to-replicate block-centered CKA;
- the two arm-specific pairwise Spearman correlations and their unweighted
  mean;
- top-10 and top-20 neighbor Jaccard;
- distribution of per-question cosine between `g_i^(42)` and `g_i^(43)`;
- 2,000 permutations that independently shuffle the row/column ID mapping
  within each arm in one replicate for CKA and Spearman.

Whole-sample CKA/Spearman and an unrestricted permutation null may be reported
as descriptive diagnostics only; they cannot pass a gate.

Alignment interpretation is enabled only if replicate block-centered CKA is
at least `0.50`, the mean within-arm pairwise Spearman is at least `0.30`, and
both corresponding within-arm-permutation one-sided p-values are at most
`0.01`. If any reliability gate fails, the probe may still describe OPD
diversity and signal strength, but it must label the representation-mismatch
question inconclusive. These thresholds are fixed before viewing any
production OPD sketch.

If either block-centered Gram has zero Frobenius norm, or either arm-specific
rank correlation is undefined, the reliability gate fails rather than
dropping an arm or substituting a whole-sample metric.

### Directional diversity

For every positive finite vector, use L2 normalization. An exactly zero finite
gradient is retained; it is never dropped, replaced, or resampled. Because a
zero vector has no direction, embed every zero row as the same shared null
state orthogonal to every positive direction. Equivalently, its kernel values
are `K(zero, zero)=1`, including between two distinct zero rows, and
`K(zero, positive)=0`. Positive-positive entries remain cosine similarities.
This is a PSD, unit-diagonal kernel: it is the Gram matrix of positive unit
vectors augmented with one orthogonal null basis vector. G-Vendi eigenvalues
are clipped only for numerical negative noise and normalized by the kernel
trace (equal to the arm sample count) before entropy, so the reported value is
a standard effective rank. Also report positive-gradient-conditional Vendi as
a sensitivity and the null-state fraction as a separate endpoint.

A non-finite full gradient or sketch is an implementation failure that stops
the run. A sensitivity report additionally maps norms below `1e-8` times the
pooled median positive norm to the shared null state, without changing the
primary analysis.

Report for selected and random arms separately, using `g_i_bar` for OPD:

- G-Vendi effective rank;
- mean and quantiles of pairwise cosine similarity;
- mean and p95 nearest-neighbor cosine distance;
- log G-Vendi contrast
  `D = log(G-Vendi(selected) / G-Vendi(random))`.

Use 2,000 prompt-cluster bootstrap replicates. Report the point estimate and
95% percentile interval. Independently resample exactly 256 prompt IDs with
replacement inside each arm; carry both rollout replicates with an ID and
average them before computing the statistic. Never resample rollout slots,
capture steps, or rollout seeds as independent observations.

Because ordinary bootstrap duplicates can bias a spectral effective-rank
statistic, also report two fixed sensitivities:

- 2,000 pooled-label permutations that split the 512 unique IDs into two groups
  of 256 and test `D` under exchangeability;
- 2,000 without-replacement half-samples of 128 IDs per arm, reporting the
  distribution of the same log ratio.

The full-sample point estimate plus the arm-label permutation p-value is the
primary diversity result. Bootstrap and half-sample intervals quantify
sampling stability; they are not relabeled as independent experiments.

### Cross-space alignment

For the same ordered IDs, form cosine Gram matrices from SFT sketches and
`g_i_bar`. Report:

- block-centered CKA using `B(K)` defined by the two sampling arms;
- each arm's upper-triangle Spearman correlation and their arithmetic mean;
- mean Jaccard overlap of within-arm top-10 and top-20 nearest-neighbor sets;
- a 2,000-permutation null that independently shuffles the OPD ID mapping
  within each arm, with empirical one-sided p-values for block-centered CKA
  and the mean within-arm Spearman.

Define `CKA_excess` as observed SFT-to-OPD block-centered CKA minus the median
within-arm-permuted block-centered CKA. Generate the 2,000 original-sample
within-arm permutations once with NumPy `PCG64(2026071505)`, persist their
index arrays, and freeze their median as `m_null`. The point statistic is:

```text
CKA_excess_hat = block_CKA_hat - m_null
```

Use 2,000 arm-stratified prompt-ID bootstrap samples from
`PCG64(2026071503)` to report a 95% percentile interval. For bootstrap sample
`b`, retain duplicate IDs and their paired rows/columns, recompute only its
observed block-centered CKA, and define:

```text
CKA_excess_b = block_CKA_b - m_null
```

There is no nested permutation inside a bootstrap and no recomputation of
`m_null`; the interval is explicitly conditional on the frozen original-sample
null baseline. The original 2,000 permutations alone define the empirical CKA
p-value. CKA is the primary cross-space endpoint. Spearman and neighbor overlap
are required consistency diagnostics but cannot override the CKA decision.
Whole-sample relational metrics are descriptive only.

These relational metrics are valid across unequal model parameter spaces;
direct SFT-vector-to-OPD-vector cosine is not.

### Signal strength and routing

Report by arm:

- per-replicate full-gradient norm, projected norm, and the precisely defined
  root-mean-square full-gradient norm by arm;
- mean absolute teacher-student log-probability gap;
- rollout-IS mean and clipping fraction;
- correct-count and easy/learnable/unresolved distributions;
- original rollout tokens and effective ESR-supervised tokens;
- fraction of exact-zero and sensitivity-defined near-zero sketches.

Correlate gradient norm and OPD nearest-neighbor structure with rollout length,
ESR tokens, route, topic, difficulty, and offline R1-solution length. These are
mechanism diagnostics, not alternative primary endpoints.

For the weak-signal endpoint, let `g_is` be the unscaled prompt-group gradient
for question `i` and rollout seed `s`, already averaged across its four
trajectories, and let `e_isj` be the effective supervised-token count of
trajectory `j`. Define, for arm `a` with 256 question IDs:

```text
N_a = sqrt((1 / (256 * 2)) * sum_i_in_a sum_s ||g_is||_2^2)
T_a = (1 / (256 * 2 * 4)) * sum_i_in_a sum_s sum_j e_isj
Q_norm = N_selected / N_random
Q_token = T_selected / T_random
```

The random-arm denominator must be strictly positive in the point estimate and
every bootstrap replicate; otherwise that endpoint is degenerate and cannot
trigger a weak-signal label. Obtain each ratio's 95% percentile interval from
the same 2,000 arm-stratified prompt bootstraps: resample 256 IDs with
replacement independently in each arm and carry both seeds and all four
rollout slots with each ID. For a one-sided test of `H0: Q >= 0.90` against
`H1: Q < 0.90`, use the centered ratio-scale bootstrap p-value (which remains
defined when a selected-arm numerator is exactly zero):

```text
p = (1 + count[(Q_b - Q_hat) <= (Q_hat - 0.90)]) / 2001
```

Apply Holm-Bonferroni at familywise alpha `0.05` to the two raw p-values
`Q_norm` and `Q_token`. An endpoint is low only when its unadjusted one-sided
95% upper percentile bound is below `0.90` and its Holm-adjusted p-value is at
most `0.05`.

## Interpretation gates

Interpretation has one exclusive primary representation label and an optional
secondary mechanism flag. Define **OPD diversity transfers** exactly as:

```text
averaged-OPD D bootstrap lower bound > 0
and selected-greater arm-label permutation p <= 0.05
```

Choose the primary representation label by this fixed hierarchy:

1. If the OPD reliability gate fails, report **inconclusive representation
   alignment**. Do not use low SFT-to-OPD CKA as mismatch evidence.
2. Otherwise, **support representation mismatch** only when all three primary
   conditions hold:
   - sampled SFT `D` has bootstrap lower bound above `0`;
   - averaged-OPD `D` has bootstrap upper bound at or below `0` and its
     arm-label permutation p-value is greater than `0.05` in the selected-greater
     direction;
   - `CKA_excess` has bootstrap upper bound below `0.10`.
3. **Reject mismatch as the primary explanation** when OPD diversity transfers
   and `CKA_excess` has bootstrap lower bound above `0.10`.
4. All other patterns are **inconclusive or mixed** and are reported without
   choosing a favorable secondary metric. Scaling beyond 512 IDs or adding a
   third rollout seed requires a new compute decision.

Independently of that exclusive label, add the secondary
**weak-signal/budget-coupling** flag only if OPD diversity transfers and at
least one of `Q_norm` or `Q_token` satisfies both its upper-bound and
Holm-adjusted-p-value criteria above. The report names which endpoint(s)
triggered it. This flag may coexist with **reject mismatch** and never replaces
the primary label; if the reliability gate failed, the headline remains
**inconclusive representation alignment** followed by the secondary flag.

Teacher-gap, rollout-IS, Spearman, and neighbor-overlap results explain a gate
but do not change its label. This prevents post-result metric selection.

This is a diagnostic, not publication-grade proof that a coreset improves OOD
accuracy. It tests the missing link between the selector representation and the
actual optimization signal.

## Components

The implementation plan should keep responsibilities separate:

1. a pure helper module for population reconstruction, deterministic sampling,
   manifests, replay-loss construction, group aggregation, and analysis;
2. opt-in driver-side and actor-side capture helpers with no behavior change
   when disabled;
3. a capture launcher that pins the Step-0, learning-rate-zero contract;
4. a resumable per-GPU full-gradient replay collector;
5. an analysis CLI that joins existing SFT sketches with OPD sketches and
   produces one machine-readable report plus a concise Markdown summary;
6. one top-level fail-closed launcher used from `opd-CLI`.

Runtime outputs use new ignored directories under
`data/opd_gradient_alignment/` and `logs/opd_gradient_alignment/`. No existing
gradient, training, checkpoint, or evaluation artifact is overwritten.

## Testing and acceptance

CPU/unit tests must cover:

- exact reconstruction of the seed-42 random 12,800 IDs;
- deterministic, unique, arm-balanced probe sampling;
- two deterministic rollout replicates with distinct vLLM seeds and identical
  Step-0 actor hashes;
- manifest mismatch and atomic-resume behavior;
- group validation for exactly four rollout slots;
- exact construction and actor-side capture of
  `loss_response_mask = response_mask * tale_budget_esr_loss_mask`;
- ESR-masked tokens contributing zero gradient;
- stop-gradient behavior of the reverse-KL advantage;
- replay loss matching four separate calls to the existing PPO loss at ratio
  one, including rollout-IS weights;
- group gradient equaling the mean of four explicit trajectory gradients in a
  toy model;
- four sequential backwards equaling the gradient of the mathematical
  four-trajectory mean without constructing a simultaneous four-trajectory
  graph;
- projected-gradient validation and deterministic parameter ordering;
- exact-zero retention, non-finite fail-fast behavior, and near-zero
  sensitivity handling;
- G-Vendi, replicate reliability, CKA, Spearman, top-k overlap, bootstrap,
  half-sampling, permutation, and interpretation gates on synthetic
  known-aligned, noisy, zero-row, and shuffled spaces;
- capture-disabled trainer behavior remaining unchanged.

GPU acceptance proceeds in order:

1. tiny-model gradient/replay equivalence;
2. one short real Qwen3 trajectory replay;
3. eight-question, two-rollout-seed end-to-end smoke;
4. four-GPU production replay only after all earlier gates pass.

Completion requires exact coverage of all 512 IDs under both rollout seeds,
exactly four rollout slots for each of the 1,024 group realizations, no
non-finite full gradients or sketches, fixed handling of every exact-zero row,
complete provenance, and a successfully regenerated analysis report from
stored artifacts alone.

## Runtime policy

GPU commands run only inside the existing user-authorized `opd-CLI` Slurm
allocation. Immediately before each GPU stage, verify that the allocation owns
four visible logical GPUs and that they are idle. Do not nest another `srun`, do
not send a blind interrupt to the pane, and do not reuse physical GPU IDs from
the host namespace.

Every stage logs through `tee` to a unique path and records its exact resume
command. A failed stage leaves the allocation and all unrelated processes
untouched.

## Non-goals

This probe does not:

- train a new selector or a new OPD model;
- change the completed coreset;
- use a step-50 checkpoint;
- resume the interrupted HMMT evaluation;
- equate training score with OOD generalization;
- silently replace full gradients with last-layer, LoRA, activation, or
  embedding features;
- claim that one 512-group diagnostic establishes downstream causal benefit.
