# Vanilla-OPD Proxy-Gradient Data-Selection Verify Design

Date: 2026-07-14

## Decision

Run a frozen-model, offline verification of whether gradients from a cheap,
adjacent-scale vanilla on-policy distillation pair can identify a question
subset that is more diverse than random in the real strong-to-weak OPD update
space.

The primary hypothesis is:

> A subset selected from `Qwen3-4B -> Qwen3-0.6B` single-rollout vanilla-OPD
> gradients has higher G-Vendi than same-size uniform-random subsets when all
> subsets are measured with frozen `Qwen3-30B-A3B-Instruct-2507 -> Qwen3-4B`
> vanilla-OPD gradients.

This is a verification experiment only. It does not update model parameters,
train a selected-data checkpoint, or claim an OOD accuracy improvement.

This design supersedes
`docs/superpowers/specs/2026-07-14-step0-opd-gradient-alignment-probe-design.md`
for future implementation. The superseded design measured the already selected
SFT-gradient coreset under the previous `n=4` group-success routing and ESR
algorithm. The new experiment removes compression, difficulty routing, ESR,
and entropy interventions and changes the null hypothesis from “OPD proxy is
closer to target than SFT” to “OPD proxy selection beats random in the target
OPD gradient space.” Existing artifacts from the superseded probe are not
accepted as inputs to this experiment.

## Scientific questions

The primary question is:

> Does a practical `n=1` proxy-OPD selector beat the conditional distribution
> of same-size uniform-random subsets in the real target-OPD gradient space?

Secondary questions are:

1. Does averaging four proxy rollouts succeed when the practical `n=1` proxy
   does not?
2. Does the controlled SFT-gradient representation also beat random after model
   family and candidate pool are controlled?
3. Is prompt-embedding diversity already sufficient?
4. Can the fixed cluster-balanced selector beat random when given the true
   target gradients as an oracle?
5. Are failures caused by representation mismatch, rollout noise, an unstable
   clustering rule, outlier seeking, or weak OPD learning signal?

The experiment does **not** require proxy OPD to outperform SFT gradients. SFT
and proxy OPD are independent candidate representations. Either, both, or
neither may pass the random-null test.

## Non-goals

This experiment does not:

- launch OPD, SFT, RLVR, or coreset training;
- update an actor, optimizer state, scheduler, or teacher;
- select the final 12,800-question production coreset;
- use compression, concise teacher prompts, difficulty routing, ESR, direct
  entropy optimization, length penalties, peer conditioning, candidate
  filtering, or verifier advantages in the actor objective;
- infer question difficulty or prioritize success rate near 0.5;
- use AIME, HMMT, MATH500, MinervaMath, OlympiadBench, or AMC examples as a
  validation gradient or selection target;
- assert that gradient-space diversity necessarily improves downstream OOD
  accuracy;
- treat trajectories from the same question as independent statistical units;
- compare raw gradient vectors across different parameter spaces;
- choose the better of two clustering seeds after seeing target-space scores.

## Compared representations

Use the following names throughout artifacts and reports.

### Target OPD: `T`

```text
teacher:  models/Qwen3-30B-A3B-Instruct-2507
student:  models/Qwen3-4B
rollout:  Qwen3-4B
gradient: full-parameter Qwen3-4B vanilla-OPD gradient
```

`T` is the space in which every selected subset is evaluated. It is not a
scalable production selector.

The arrow notation in this document is `teacher -> student`. The production
signal is nevertheless a student-on-policy sampled reverse-KL gradient,
`KL(student || teacher)`: the teacher scores action tokens sampled by the
student. It is not a full-vocabulary `KL(teacher || student)` gradient.

### Proxy OPD: `P`

```text
teacher:  models/Qwen3-4B
student:  Qwen/Qwen3-0.6B
rollout:  Qwen3-0.6B
gradient: full-parameter Qwen3-0.6B vanilla-OPD gradient
```

The 4B checkpoint and tokenizer used as the target student must be byte-identical
to those used as the proxy teacher. The small model is pinned to:

```text
repository: Qwen/Qwen3-0.6B
revision:   c1899de289a04d12100db370d81485cdf75e47ca
```

It is materialized under `models/Qwen3-0.6B` during implementation. This
same-family ladder avoids conflating `OPD vs. SFT` with `Qwen3 vs. Qwen2.5`.
Because each teacher scores student-generated token IDs directly, every
teacher/student pair must also pass a vocabulary-to-ID, EOS, and padding-token
compatibility check before rollout. Sharing a model family is not accepted as
proof of token compatibility.

### SFT-gradient baseline: `S`

```text
model:      the same pinned Qwen3-0.6B used by P
prompt:     prepared original DeepMath question
completion: r1_solution_1
loss:       completion-only cross entropy
gradient:   full-parameter Qwen3-0.6B gradient
```

The existing Qwen2.5-0.5B gradient vectors are not reused because doing so
would introduce a model-family and parameterization confound. The 768 Stage-1
candidate rows are recomputed for this baseline; Stage 2 appends its 768 new
candidate rows if and only if the extension gate fires.

SFT formatting and gradient collection reuse the official Prismatic
`GradientComputer`, not a new local loss implementation, from the clean
reference checkout pinned to:

```text
commit: d9484cd3b5991030b901ac4a3a9e2472dbfac2ad
tree:   a0079d8c5e15cb18bb4790f99c43cc19bb9ecd50
```

The exact Qwen chat template is applied to the original question and
`r1_solution_1`; prompt labels are masked by the official
`DataCollatorForCompletionOnlyLM`, and at least one completion label must remain.

### Prompt-embedding baseline: `E`

Use the same Qwen3-0.6B checkpoint without a completion. Apply its chat
template to the one user message containing the same raw OPD prompt with
`tokenize=true`, `add_generation_prompt=true`, and `enable_thinking=false`.
Thus the assistant-generation prefix and the template's disabled-thinking
markers are part of the embedded token sequence. Take final-layer hidden states
over every non-padding token, mean-pool them, and L2-normalize the result. No
response, teacher, or backward pass is used.

### Random baselines

- `R_uniform`: a uniform subset sampled without replacement from the candidate
  pool;
- `R_stratified`: a subset sampled without replacement while preserving leaf
  topic and prompt-token-length-quartile counts as closely as integer rounding
  permits.

Random baselines use no model output.

## Pinned source data

The source and prepared artifacts are:

```text
source parquet:
  data/g-opd/DeepMath-103K/train_filtered_level6.parquet
  rows:   57,046
  sha256: de3350fdd00bc0410550098ea65179e2be873da99e4075f80de575fc17670597

prepared R1 JSONL:
  data/gradient_diversity/deepmath_level6_r1_solution1.jsonl
  sha256: ee8d55c943577888f14052b3953a0a8aa07d26076e50ddffa41a102b60027344

prepared manifest:
  data/gradient_diversity/deepmath_level6_r1_solution1.manifest.json
  sha256: 5add2e965473d3647f2318c73f957434fbe23344516089be9adfe89db9c6a70e

eligibility report:
  data/gradient_diversity/deepmath_level6_r1_solution1.eligibility.json
  sha256: cc8d5b888def37761519e15c6ef722e5a22d91ae2342c6d669f008bcff3dcbb6
```

The eligibility report defines exactly 57,045 stable IDs and excludes the one
prepared row that exceeds the pinned 32,768-token SFT proxy context boundary.
Its ordered eligible-ID SHA-256 is
`5dbb267fed334219978d3d74610793fc8d146add15c488bcc59992779e22d8cf`.
This report is explicitly a legacy Qwen2.5 eligibility contract, not evidence
that a row fits Qwen3-0.6B. The new probe retains its 57,045 IDs only to keep the
already established population and production coreset denominator fixed, then
runs a fresh Qwen3-0.6B/Qwen3-4B token preflight on every sampled row. It must
not reinterpret or expand the population after sampling; a sampled row
unsupported by either new model is a fail-closed error rather than an implicit
replacement.

## Deterministic probe population

Before GPU work, construct one immutable sample manifest.

```text
eligible population:       57,045 questions
sample seed:               2026071401 (NumPy PCG64)
main sample:                1,024 questions without replacement
candidate prefix:            768 questions
held-out reference suffix:   256 questions
selection size:               172 questions
```

The selected size is the rounded production coreset fraction:

```text
round(768 * 12,800 / 57,045) = 172
```

Construct `numpy.random.Generator(numpy.random.PCG64(2026071401))` and call
`permutation(57_045)` exactly once on the ordered eligible-ID positions. The
first 768 IDs are candidates and the next 256 are held-out reference questions.
The split is not stratified, rebalanced, or regenerated after inspecting
metadata or model outputs.

Each manifest row records:

```text
stable_id
source_row_index
original_dataset_index
split: candidate | held_out
sample_position
exact question
raw OPD prompt messages
topic and leaf topic
difficulty
prompt token counts for 4B and 0.6B
R1 completion token count for 0.6B
full Qwen3-0.6B SFT chat token count and supervised-label count
source/prepared/eligibility hashes
```

The manifest additionally records the ordered hashes for all 1,024 IDs, the
candidate IDs, and the held-out IDs.

`leaf_topic` is derived only by splitting the prepared row's hierarchical
`topic` string on the exact delimiter `" -> "`, stripping components, and
taking the final non-empty component. A missing topic, empty component, or
non-string value fails manifest construction.

### Benchmark exclusion audit

The sample builder loads these exact benchmark files used by the strong-to-weak
evaluation suite and binds their hashes into the sample manifest:

| Path | Rows | SHA-256 |
|---|---:|---|
| `data/aime24/test.jsonl` | 30 | `a3c49569f3d7125aaf4eea5764bd1b868af534ae3aff6aad0843a7da58fe46b8` |
| `data/aime25/test.jsonl` | 30 | `6012af2a112f5e26d91f1b0cc644b5dde6399173b8b648aec2e6b9899877a2db` |
| `data/hmmt25_feb/test.jsonl` | 30 | `58d889b807a87fd189562d9aeaf2bddf342b2e956843ebbbd8259fc93195cc3f` |
| `data/hmmt25_nov/test.jsonl` | 30 | `37be20ee4d01044638f1d790d938138c2ce6f26fc9e60f6987db5c220783f226` |
| `data/math500/test.jsonl` | 500 | `5b126695a919f8fb40edffaf2e748ae08f7fb6ce0721211671c760b4f0b7c5de` |
| `data/minervamath/test.jsonl` | 272 | `ff6e07ac93af4e43885fe7710ded51c8db04543d627326b855d903075d59e872` |
| `data/olympiadbench/test.jsonl` | 674 | `6c3c658145c21dd76f70eef1456dc5de9bbb38342f2ffdc4a803d8e6e3005ecc` |
| `data/amc2023/test.jsonl` | 40 | `b443425b035d98fec3da4de7e347ac43cebcf7b721e96ba8bf9e0120d24dd61d` |

A read-only raw/exact audit currently finds zero overlaps, but that does not
replace the reproducible normalized audit. The builder rejects a sampled
question on either of these conditions:

1. normalized exact-question equality after Unicode NFKC, case-folding,
   whitespace collapse, and removal of only the fixed terminal OPD instruction;
2. a shared normalized contiguous 10-token gram between the mathematical
   question bodies.

For the second condition, apply the same NFKC/case-fold/suffix removal and then
tokenize with the fixed regex `\\[A-Za-z]+|[A-Za-z0-9]+`; punctuation and
whitespace are separators. Hash every contiguous sequence of ten resulting
tokens and reject any train/eval hash intersection. Questions with fewer than
ten tokens are still covered by exact equality.

The builder does not silently draw replacements. Any collision fails before
GPU work so the sampling contract can be explicitly revised rather than
conditioned on benchmark content.

## Staged execution

### Stage 0: 32-question smoke

Use the first 24 candidate and first 8 held-out IDs from the frozen manifest.
The smoke validates capture, replay, projection, artifact resume, selection,
and analysis. Its test-only selected size is
`round(24*12,800/57,045)=5`; both cluster-ratio paths clamp to `K=2`, and each
random-null generator emits 100 subsets. These smoke constants never enter a
scientific report. The smoke does not produce a pass/fail result for the
hypothesis.

### Stage 1: 1,024-question main probe

Run the complete design below. This is the default stopping point.

### Conditional Stage 2: 2,048-question extension

An extension is permitted only when the target oracle passes and the practical
proxy result is borderline:

```text
0.90 <= median n=1 worst-case G-Vendi percentile < 0.95
```

Append the next 1,024 IDs from the original frozen permutation without changing
any Stage-1 membership: permutation positions `[1024, 1792)` add 768 candidates
and positions `[1792, 2048)` add 256 held-out references. Thus the expanded
split is the ordered concatenation of the old and new blocks, with 1,536
candidate and 512 held-out IDs, and the selected size is:

```text
round(1,536 * 12,800 / 57,045) = 345
```

All cardinality-dependent selector values are recomputed for the expanded
candidate pool: primary `K=floor(0.10*1,536)=153`, diagnostic
`K=floor(0.01*1,536)=15`, and every selected/random subset has 345 IDs. Main-run
constants such as 768, 256, 76, 7, and 172 must never leak into Stage 2.

Stage 2 writes a new manifest parented by the immutable Stage-1 manifest and
report; it never mutates Stage-1 artifacts. It may reuse verified Stage-1
vectors by parent hash and computes only the appended rows, but final Stage-2
selection, random subsets, metrics, and classifications are regenerated over
the full expanded pool. Expected expanded totals are 4,096 target
question-seed vectors, 16,384 target trajectories, 12,288 proxy trajectories,
and 1,536 vectors for each deterministic candidate baseline.

No extension is run for a clear pass, a clear fail below 0.90, a failed oracle,
or a provenance/correctness failure.

Stage 1 remains the sole primary classification. Because Stage 2 is triggered
by a borderline Stage-1 result and reuses its rows, its expanded-pool null does
not calibrate the full adaptive stopping rule. Stage 2 therefore reports a
separate `extended_pass|extended_fail` sensitivity result and may diagnose
finite-pool uncertainty, but it cannot upgrade or downgrade the Stage-1
classification in the interpretation matrix.

## Frozen vanilla-OPD contract

Both target and proxy capture use the same algorithmic contract:

```text
student prompt:                    normal/raw OPD prompt
chat template thinking:            disabled
capture rollout.n:                  4
rollout temperature / top-p:       1.0 / 1.0
maximum prompt length:             2,048
maximum response length:           16,384
policy loss mode:                  vanilla
only reverse-KL advantages:        true
PPO epochs:                        1
mini-batches per actor rank:       exactly 1
micro-batch size per GPU:          1
dynamic micro-batching:            false
loss aggregation:                  token-mean
rollout correction:                token-level IS
rollout IS upper threshold:        5.0
rollout IS batch normalization:    false
entropy coefficient:               0
use explicit KL loss / teacher:     true
explicit KL-loss coefficient:      0
KL in reward:                      false
candidate selection:               disabled
length-aware OPD:                  disabled
difficulty-aware OPD:              disabled
TALE / ESR:                        disabled
teacher prompt routing:            disabled
rethinking probe:                  disabled
actor optimizer step:              forbidden
```

`rollout.n=4` is a capture-only Monte Carlo setting; `P_n1` means one
trajectory/gradient slot from that four-completion request, not an engine run
configured with `n=1`. For each `(stage, target-or-proxy pair, engine seed)`,
create exactly one vLLM engine. Submit all ordered stage prompts for target and
candidate-only ordered prompts for proxy in one `generate_sequences` call with
`n=4`, record returned completion order as slots `0..3`, and destroy the engine
only after the full call is materialized. Do not restart or reseed between
questions and do not emulate four slots with four repeated `n=1` calls. Stage 2
creates a new engine and calls only its deterministic appended prompt block; it
never regenerates Stage-1 slots. The resolved vLLM version, engine arguments,
prompt batch order, and returned compound-key order are hashed as provenance.

Verifier correctness may be computed and stored as a diagnostic, but it cannot
alter the prompt, mask, loss, sampling, or selected subset.

### Exact Step-0 loss

Four log-probabilities must remain distinct:

```text
rollout_log_prob:       vLLM log p for the sampled action token
batch_old_log_prob:     pre-update FSDP actor recomputation in the trainer
current_log_prob:       gradient-bearing actor micro-batch recomputation
ref_log_prob:           teacher log p on the same sampled action token
```

All are action-token log-probabilities under the same response-token alignment
and rollout-temperature scaling; `ref_log_prob` is not a full teacher
distribution. Before actor update, production rollout correction constructs:

```text
delta_t = clamp(batch_old_log_prob_t - rollout_log_prob_t, -20, 20)
w_t = stopgrad(min(exp(delta_t), 5.0))
```

The response mask sets padding weights to zero. The `-20` bound is a numerical
log-ratio safety clamp; there is no configured lower IS-threshold clamp such as
`1/5`, and there is no batch normalization. Because the frozen configuration
has one PPO mini-batch and one PPO epoch, the actor then takes its on-policy
shortcut:

```text
local_old_log_prob_t = stopgrad(current_log_prob_t)
A_t = stopgrad(ref_log_prob_t - local_old_log_prob_t)
r_t = exp(clamp(current_log_prob_t - local_old_log_prob_t, -20, 20))
```

Production passes `A_t`, `r_t`, the response mask, and `w_t` through its exact
dual-clipped `compute_policy_loss_vanilla` path. At the frozen point, `r_t` is
numerically one but still has a derivative through `current_log_prob_t`; it
must not be replaced by a constant tensor. Replay must call the registered
production policy-loss function. Replacing it by a generic full-vocabulary KL
or a hand-written surrogate is forbidden even when the latter has the same
first derivative at this one point.

`use_kl_loss=true` is retained solely because current trainer orchestration uses
it to create the teacher/ref worker. Its coefficient is zero, so the separate
explicit KL-loss term contributes no gradient. Turning it off would remove the
teacher path and is a correctness failure, not an optimization.

The normal response mask is the only loss mask. Rollout correction weights are
captured from the production helper and detached exactly as in training. The
mask includes the first EOS token, excludes every token after EOS, and retains
the full generated response when EOS is absent. Actor train/eval mode, BF16
autocast, temperature division, and response-token alignment are pinned to the
production path rather than chosen independently by replay.

## Rollout repetitions and statistical units

### Target repetitions

For every one of the 1,024 questions:

```text
vLLM engine seeds:       42 and 43
rollouts per seed:       4
target trajectories:     1,024 * 2 * 4 = 8,192
30B teacher-scored rows: 8,192
```

For each question and engine seed, compute four trajectory gradients and take
their arithmetic mean before target-space normalization. This gives two
question-level target representations, `T_42` and `T_43`. Equal trajectory
weighting matches the production actor path with micro-batch size one and
token-mean aggregation; concatenating all response tokens and taking one
global token mean would incorrectly length-weight the four trajectories.

### Proxy repetitions

For each of the 768 candidate questions:

```text
vLLM engine seeds:      42 and 43
rollouts per seed:      4
proxy trajectories:    768 * 2 * 4 = 6,144
4B teacher-scored rows: 6,144
```

Store all eight individual trajectory gradients per question. The eight
engine-seed/slot pairs define eight Monte Carlo practical `P_n1`
representations. Within each engine seed, average the four projected gradients
to define two `P_n4` sensitivity representations. Projection is linear, so
this average must equal the projection of the corresponding averaged full
gradient within numeric tolerance.

The independent statistical unit is always a stable question ID. Rollout slots
and engine seeds are Monte Carlo reliability replicates, not additional
questions.

## Capture and replay data flow

### Capture boundaries

Capture has two explicit boundaries. At the trainer boundary, immediately
before `update_actor`, freeze the full batch after rollout correction weights
and teacher log probabilities exist. At the actor boundary, after the current
forward, on-policy old-log-prob overwrite, and reverse-advantage construction
but before policy-loss backward, return the authoritative loss tensors in
capture mode. Capture mode is opt-in, disabled by default, and must have no
import-time or runtime effect on normal training.

The trainer may reorder rows by response length and the actor's normal tensor
selection drops non-tensor metadata. Capture mode must therefore propagate and
validate an explicit compound key `(stable_id, engine_seed, rollout_slot)` at
both boundaries. Original row order is never used to join artifacts.

For every trajectory, store:

```text
stable ID, split, engine seed, and rollout slot
input_ids, responses, attention_mask, position_ids
response_mask
rollout_log_probs
batch_old_log_probs used by rollout IS
current_log_probs and overwritten local_old_log_probs used by policy loss
ref_log_prob
detached rollout_is_weights
stopped reverse-KL advantages
PPO ratio and clip-branch diagnostics
per-trajectory token-mean policy loss
teacher and student prompt messages
response length and valid-token count
verifier correctness, if computed
all model/tokenizer/config hashes
```

Artifacts contain token IDs and tensors, not only decoded text. Each tensor
chunk has an ID sidecar and is written atomically.

### Frozen actor invariant

No optimizer is constructed or stepped in the preferred capture entry point.
If existing trainer orchestration requires an optimizer object, its learning
rate is zero and `optimizer.step` is replaced by a fail-fast sentinel. The
recursive hash of every actor parameter must be identical before and after
each capture process.

### Replay gradients

Replay loads the pinned actor and captured tensors, recomputes current actor
log probabilities with gradients enabled, reconstructs the exact stopped
advantage and PPO loss, and performs full-parameter backward only on the
student. Teachers are always inference-only. Replay verifies the four captured
log-probability roles separately: it recomputes the actor-derived roles and
integrity-checks the immutable vLLM/teacher roles. It must not substitute
`batch_old_log_probs` for the actor-local stopped current log-probability.

For target `n=4`, trajectory gradients may be accumulated sequentially to fit
memory, but each trajectory first receives its own token-mean normalization and
each parameter's four-trajectory sum is accumulated and divided in float32
before the projection-only float16 cast. This may be streamed parameter by
parameter; it need not materialize four flat vectors. For proxy `P_n1` slots,
zero gradients between trajectories and project each gradient independently.

The canonical per-trajectory vector is the gradient of the unscaled
token-mean scalar returned by `compute_policy_loss_vanilla`, before the actor's
uniform `1 / gradient_accumulation` multiplier. That multiplier is recorded and
checked against production but removed from every question vector; otherwise
the same trajectory would change norm merely because capture batch size or
data-parallel degree changed. The frozen production batch gradient is the
arithmetic mean of these canonical question gradients, so removing this common
factor changes neither directions nor subset ranking. Smoke comparisons apply
the same convention to both sides.

Store for every projected vector:

```text
unnormalized float32 projected gradient
full-gradient L2 norm before projection
projected-gradient L2 norm
valid response-token count
sampled reverse-KL estimate: mean(local_old_log_prob - ref_log_prob)
OPD signal RMS: sqrt(mean((ref_log_prob - local_old_log_prob)^2))
```

Target guardrails operate on one question/engine-seed record, not on four
independent rows. For trajectories `j=1..4`, define:

```text
group gradient norm = norm((1/4) * sum_j g_j)
group sampled reverse KL = (1/4) * sum_j mean_t(local_old_jt - ref_jt)
group OPD signal RMS = sqrt((1/4) * sum_j mean_t((ref_jt-local_old_jt)^2))
group valid-token count = (1/4) * sum_j valid_tokens_j
group response length = (1/4) * sum_j response_length_j
```

Thus every trajectory is weighted equally before subset means are computed;
long responses do not receive extra weight. Per-trajectory values are retained
for diagnostics, but only these group values enter target-space guardrails.

## Gradient projection

Use the same projection family as the existing Prismatic-aligned collector:

```text
projection:        Rademacher / TRAK CudaProjector
dimension:         1,024
projection seed:   0
projection type:   ProjectionType.rademacher
block size:        128
maximum batch:     16
model_id:          0
projection input:  float16 flattened native gradients
output scaling:    divide by sqrt(1,024)
stored dtype:      float32
selector input:    L2-normalized stored projection
reference commit:  d9484cd3b5991030b901ac4a3a9e2472dbfac2ad
```

Each representation has its own source parameter space and projector instance.
Cross-representation vector cosine is undefined and forbidden. Comparisons
between `T`, `P`, and `S` use only their within-representation question Gram
matrices.

The collector must refuse fallback from `CudaProjector` to a basic CPU
projector, matching the current full-gradient collection policy. Parameter
ordering and projector identity are frozen per representation. A sharded actor
gradient must be gathered or projected under one globally equivalent parameter
ordering; a single rank's local FSDP shard is never a valid question gradient.
Before the projection-only float16 cast, compute full-gradient norms from a
float32 view of the native pre-clipping gradients. The actor runs in the exact
production train mode and BF16 autocast state; SFT retains the official
collector's eval mode. Gradient clipping, optimizer scaling, AMP loss scaling,
and optimizer state are absent. Manifests record parameter dtype, gradient
dtype, flatten order, and all projector constructor fields.

## Unified fixed-pool selector

Every non-random representation uses the same official-reference selector:

```text
distance:                  cosine on L2-normalized vectors
primary cluster ratio:     0.10
primary K:                 floor(0.10 * 768) = 76
diagnostic cluster ratio:  0.01
diagnostic K:              floor(0.01 * 768) = 7
K-means iterations:        20
K-means/permutation seeds: 42 and 43
selected rows:             172
fixed-pool sampler:        cluster-balanced round robin
round-robin seed:          42 for both K-means seeds
```

The primary result uses only `K=76`. `K=7` is a sensitivity result. The two
primary K-means seeds both remain visible; the report cannot select the better
seed. The fixed round-robin seed matches the existing code-aligned selector;
changing it together with K-means seed would introduce a second difference.

Official cosine K-means requires exactly one visible CUDA device. The tiny
selection job therefore runs inside `opd-CLI`; balanced round robin and all
metric computation remain CPU-only.

Apply the selector independently to:

```text
P_n1: eight rollout realizations
P_n4: two rollout-seed averages
S:    one deterministic SFT representation
E:    one deterministic prompt-embedding representation
T_42 and T_43: target-oracle representations
```

For the oracle, select with `T_42` and evaluate with `T_43`, then reverse the
roles. Evaluating an oracle subset in the same noisy target realization used
to select it is diagnostic only and cannot satisfy the oracle gate.

## Random null distributions

Let `B=100` for Stage 0 and `B=10,000` for Stages 1 and 2. Generate `B`
uniform subsets and `B` stratified subsets of the manifest-selected cardinality
(5, 172, or 345) without replacement. Create
`numpy.random.SeedSequence(2026071402)`, call `spawn(2)` once, and initialize
the uniform PCG64 generator from child 0 and the stratified PCG64 generator
from child 1. The two nulls never share or continue one another's RNG stream.
Use exactly the same ordered subset lists for `T_42` and `T_43`, so target-seed
comparisons are paired.

For uniform replicate indices `0..B-1`, call `Generator.choice` on integer
candidate positions with `size=m`, `replace=false`, and `shuffle=false`, then
sort the positions ascending for storage. For stratified replicates, traverse
lexicographically sorted stratum keys; within each stratum call the stratified
generator on stable-ID-sorted member positions with `size` equal to that
stratum's allocated count, `replace=false`, and `shuffle=false`; union the
results and sort by global candidate position. Duplicate subsets across
replicate indices are permitted; duplicate IDs inside a subset are not. Pin
the NumPy version and exact ordered-subset-list SHA-256 in every report, with
small hash fixtures in unit tests.

For stratified random, define strata from pre-rollout, non-gradient metadata
only:

```text
leaf topic x Qwen3-0.6B prompt-token-length quartile
```

Length quartiles are rank-based: sort candidate rows by
`(prompt_token_count, stable_id)` and assign quartile
`floor(4 * zero_based_rank / candidate_count)`. This fixes ties without using
model outputs. Largest-remainder allocation ties are resolved by the
lexicographic stratum key.

Allocate the manifest-selected number of slots with deterministic
largest-remainder rounding from candidate-pool stratum proportions. When a
stratum has fewer rows than its allocation, take all rows and redistribute the
deficit by the same rule among remaining strata. Model loss, correctness,
rollout length, KL, and gradient statistics cannot affect strata.

## Primary and guardrail metrics

The thresholds below are predeclared engineering validation criteria over a
fixed candidate pool, not claims of an exact family-wise frequentist error
rate. Rollout realizations share questions and are treated as robustness
replicates rather than independent hypothesis-test samples.

Whenever a rule says “median of 8,” sort float64 values ascending and use the
arithmetic mean of the fourth and fifth values. Do not delegate this convention
to a library default.

### Target-space G-Vendi

For a subset `S`, use the normalized target gradient vectors to build its
float64, explicitly symmetrized cosine Gram matrix. Normalize its non-negative
eigenvalues to sum to one and compute:

```text
D_T(S) = exp(-sum_j lambda_j * log(lambda_j))
```

with zero eigenvalues contributing zero. Numerical negative eigenvalues within
`1e-7` of zero are clamped to zero; a more negative eigenvalue is an error.
The result must match the pinned official-reference `Vendi.compute_vendi_score`
on smoke fixtures within `rtol=1e-6, atol=1e-6`; the Gram formulation and the
official feature-covariance formulation share the same non-zero eigenvalues.

For target seed `s`, convert a selected-set score into its conditional random
percentile:

```text
P_D(S, T_s) = count(D_Ts(R_uniform) <= D_Ts(S)) / B
```

For each proxy rollout realization, define its worst-case percentile as the
minimum over both target seeds and both primary K-means seeds.

The practical `P_n1` primary gate passes when:

```text
median of 8 worst-case percentiles >= 0.95
and
at least 6 of 8 worst-case percentiles >= 0.90
```

### Held-out target-space coverage

For held-out target vectors `H_s` and selected candidate vectors `S_s`, compute:

```text
C_Ts(S) = mean_{h in H_s} max_{j in S_s} cosine(h, j)
```

Convert coverage to a uniform-random percentile with the same inclusive rule.
For `P_n1`, take the minimum over target and K-means seeds for each rollout
realization. The coverage guardrail passes when:

```text
median of 8 worst-case coverage percentiles >= 0.50
and
at least 7 of 8 worst-case coverage percentiles >= 0.10
```

This prevents high G-Vendi obtained solely by selecting target-space outliers.

### Learning-signal guardrails

For every subset and target seed, compute subset means of:

```text
full target-gradient L2 norm
target OPD signal RMS
valid response-token count
sampled reverse-KL estimate
rollout response length
```

Construct the corresponding uniform-random distributions. For each `P_n1`
rollout realization, take the minimum percentile over target and K-means seeds;
gradient norm and OPD signal RMS each pass when the median of those eight
worst-case percentiles is at least 0.10. Token counts, sampled reverse KL, and
length are reported but are not pass/fail gates.

### Oracle and target-dependence gates

Before interpreting proxy results, require:

1. both `T_42 -> evaluate T_43` and `T_43 -> evaluate T_42` oracle selections,
   for both primary K-means seeds, have G-Vendi percentile at least 0.95 and
   held-out coverage percentile at least 0.50;
2. debiased linear CKA between the `T_42` and `T_43` candidate Gram matrices has
   the permutation p-value defined below at most 0.01.

For CKA, L2-normalize every representation row, construct each linear
kernel only within its own representation as `K = X X^T` over the same ordered
stable IDs, and use the unbiased/debiased HSIC estimator; never multiply
vectors from different parameter spaces. For `n` aligned questions, set kernel
diagonals to zero and define:

```text
HSIC_u(K,L) = [tr(KL)
               + sum(K)*sum(L)/((n-1)*(n-2))
               - 2*sum(KL)/(n-2)] / (n*(n-3))
CKA_u(K,L) = HSIC_u(K,L) / sqrt(HSIC_u(K,K)*HSIC_u(L,L))
```

Here `sum(KL)` in the third term means the sum of all entries of the matrix
product `K @ L`, while `sum(K)` means the sum of all entries of `K`. A
non-positive self-HSIC denominator is a correctness failure. The permutation
null initializes `Generator(PCG64(2026071403))` once and calls
`permutation(n)` for replicate indices `0..9,999`, applying each same
permutation to the second kernel's rows and columns. Report
`p=(1 + count(null_CKA >= observed_CKA)) / 10,001`; the dependence gate
requires `p <= 0.01`.

This CKA condition establishes non-random seed dependence, not a standalone
effect-size claim. The cross-seed oracle G-Vendi and coverage conditions provide
the selection-level stability requirement.

If either condition fails, the proxy result is `inconclusive`, not `fail`.

### `P_n4`, SFT, and embedding classifications

- `P_n4` passes only if each of its two engine-seed averages has minimum
  G-Vendi percentile at least 0.95, minimum coverage percentile at least 0.50,
  and minimum gradient-norm and OPD-signal percentiles at least 0.10 over both
  target seeds and both K-means seeds.
- deterministic `S` and `E` pass under those same four minima: 0.95 G-Vendi,
  0.50 coverage, 0.10 target gradient norm, and 0.10 target OPD-signal RMS.
- stratified random is reported as a stronger null; it does not replace the
  predeclared uniform-random primary null.

### Overall classification order

1. A provenance, coverage, tensor-equivalence, or numerical correctness error
   aborts the run and produces no scientific classification.
2. If either oracle/target-dependence gate fails, all candidate
   representations are `inconclusive`.
3. Otherwise `P_n1=pass` if and only if its G-Vendi gate, held-out coverage
   gate, target-gradient-norm guardrail, and target-OPD-signal guardrail all
   pass. If any one fails, `P_n1=fail`, with every failed component reported.
4. Under the same valid oracle, classify `P_n4`, `S`, and `E` independently by
   their exact conjunctions above. One candidate's result never overrides
   another's.

## Secondary diagnostics

Secondary results cannot change the primary classification:

- raw debiased CKA for `T_42` versus `T_43`, each target seed versus every
  `P_n1`, `P_n4`, `S`, and `E` representation, and all pairwise `P_n1`
  replicate comparisons; only the predeclared `T_42` versus `T_43` reliability
  gate runs the 10,000-permutation test;
- AMI and ARI at primary `K` for the same cross-representation pairs, reporting
  all four Cartesian combinations of K-means seeds rather than choosing one;
- selected-set Jaccard and chance-adjusted overlap for `P_n1` replicate pairs,
  `P_n4` pairs, each proxy versus `S` and `E`, and each representation versus
  both target-selected oracle sets, again reporting every Cartesian K-means-seed
  pairing rather than matching or choosing seeds;
- topic coverage, topic entropy, prompt/completion/rollout/supervised-token
  length quantiles, verifier-correctness counts, sampled reverse-KL, and OPD
  signal distributions for every selected arm and both random nulls;
- primary `K=0.10` versus diagnostic `K=0.01` sensitivity, shown separately and
  never pooled.

For two selected sets `A` and `B` of size `m` from `N` candidates, report
`J=|A∩B|/|A∪B|` and chance-adjusted overlap
`(|A∩B|-m^2/N)/(m-m^2/N)`. Topic entropy is
`-sum_c p_c log(p_c)` over the pinned leaf topics. Length summaries use count,
mean, standard deviation, and fixed quantiles `[0, .25, .5, .75, .9, .95, 1]`.
Every CKA/AMI/ARI comparison uses the exact aligned stable-ID order. Oracle
selection identities are named by `(target_seed_used_for_selection,
K-means_seed)`; evaluation target seed never changes the selected-ID set.

Raw subset overlap is not a success metric. Two disjoint subsets may cover the
same target gradient modes.

## Interpretation matrix

This matrix classifies Stage 1 only; any conditional Stage-2 sensitivity is
shown in a separate report block.

In this table, `Oracle=pass` means both the cross-seed oracle and target-CKA
dependence gates pass; otherwise it is `fail` and the scientific result is
inconclusive.

| Oracle | `P_n1` | `P_n4` | SFT | Embedding | Interpretation |
|---|---|---|---|---|---|
| fail | any | any | any | any | Inconclusive: target representation, selector, or probe power failed |
| pass | pass | any | any | any | Practical single-rollout OPD proxy is supported for a later full-pool selector experiment; classify SFT and embedding independently |
| pass | fail | pass | any | any | OPD proxy works only with fourfold rollout/teacher cost; classify cheap baselines independently |
| pass | fail | fail | pass | any | SFT-gradient representation remains a supported cheap selector |
| pass | fail | fail | fail | pass | Prompt embedding is the only supported candidate representation |
| pass | fail | fail | fail | fail | None of the candidate representations beats random under this selector |

“Supported” here means only that the selected subset beats the conditional
random null in target gradient diversity while satisfying coverage and signal
guardrails. It does not mean downstream OOD improvement is established.

## Components and artifact interfaces

Implementation is divided into five bounded components.

### Manifest builder

Owns source validation, deterministic sampling, benchmark exclusion audit,
token-length preflight, and immutable manifest hashes. It has no GPU or model
gradient dependency.

### OPD capture driver

Owns frozen rollout and teacher scoring for one pair and one engine seed. It
writes capture tensors and sidecars, never projected gradients or final
statistics.

### Gradient replay/projector

Owns exact actor forward/backward, production-loss equivalence checks, gradient
accumulation, full-norm computation, projection, and per-vector manifests. It
does not select questions.

### Baseline collectors

Own SFT gradients and prompt embeddings for candidate IDs. They consume the
same sample manifest and produce the same stable-ID/vector interface as replay.

### Selector/analyzer

Owns official GPU cosine K-means, CPU fixed-size selection, random subsets,
metrics, gates, and reports. It is deterministic from frozen vector artifacts,
cannot invoke a model, and cannot mutate upstream artifacts.

## Runtime artifacts

Place generated artifacts under stage-isolated ignored data and log paths:

```text
data/opd_proxy_gradient_verify/sampling_contract.json
data/opd_proxy_gradient_verify/stage_{0,1,2}/sample_manifest.jsonl
data/opd_proxy_gradient_verify/stage_{0,1,2}/manifest.json
data/opd_proxy_gradient_verify/stage_{0,1,2}/capture/target/seed_{42,43}/
data/opd_proxy_gradient_verify/stage_{0,1,2}/capture/proxy/seed_{42,43}/
data/opd_proxy_gradient_verify/stage_{0,1,2}/gradients/target/seed_{42,43}/
data/opd_proxy_gradient_verify/stage_{0,1,2}/gradients/proxy/seed_{42,43}/
data/opd_proxy_gradient_verify/stage_{0,1,2}/gradients/sft/
data/opd_proxy_gradient_verify/stage_{0,1,2}/embeddings/prompt/
data/opd_proxy_gradient_verify/stage_{0,1,2}/selection/
data/opd_proxy_gradient_verify/stage_{0,1,2}/report.{json,md}
logs/opd_proxy_gradient_verify/stage_{0,1,2}/
```

The root sampling contract binds the eligible-ID order, PCG64 algorithm and
seed, and the first 2,048 permutation positions before any GPU output exists.
Stage 0 and Stage 1 are immutable views of that contract. A Stage-2 manifest is
created only when the predeclared extension gate fires and includes the exact
Stage-1 report hash and the deterministic added-ID blocks as parents.

Every directory has a manifest binding:

```text
git HEAD and git status
sorted source-snapshot manifest and SHA-256
source/prepared/eligibility/sample hashes
ordered stable-ID hash
model and tokenizer recursive hashes/revisions
chat-template hash
resolved configuration
package versions
engine, projection, selector, and random seeds
expected and completed question/seed/slot coverage
parent artifact hashes
```

The source snapshot enumerates every code, config, launcher, design, and
reference file that can affect the run as `(repository-relative path, byte
size, SHA-256)`, including untracked files, then hashes the canonical sorted
JSON manifest. Git diff hashes are optional diagnostics and are never treated
as sufficient provenance for an untracked file.

## Resume and failure policy

Chunks are written to temporary names, fsynced, and atomically renamed with an
ID sidecar. Within each deterministic `(stage, pair, engine_seed, logical_shard)`
work unit, resume accepts only a contiguous, duplicate-free prefix of that
unit's manifest order. Global validation then requires disjoint shard keys and
their exact union to equal expected coverage. A mismatch starts no work and
does not delete prior artifacts.

Fail before or during GPU work when:

- a source, prepared, eligibility, sample, model, tokenizer, template, code,
  config, or parent-artifact hash differs;
- target-student and proxy-teacher 4B hashes differ;
- either teacher/student pair fails exact token-ID compatibility;
- any sampled ID is missing, duplicated, in the wrong split, over context, or
  collides with an evaluation question under the pinned audit;
- any question lacks exactly four slots for an expected engine seed;
- capture uses multiple generation calls/engine restarts for one declared work
  unit or a prompt/slot order differs from its manifest;
- actor parameters change during capture;
- capture is not on the one-mini-batch/one-epoch shortcut, lacks the ref worker,
  or substitutes batch `old_log_probs` for actor-local stopped current log-probs;
- capture includes a non-vanilla mask or forbidden algorithm feature; forbidden
  candidate-selection, ESR, difficulty-entropy, or routing keys must be absent,
  not merely numerically zero;
- replay does not use the production policy-loss registration;
- a stored vector is zero, NaN, or infinite;
- projection falls back from the pinned CUDA backend;
- tensor and ID sidecar coverage differs;
- a selector output has the wrong manifest-derived size or contains
  duplicate/unknown IDs;
- a random subset contains duplicate IDs or has the wrong manifest-derived
  cardinality;
- a metric is non-finite or violates its mathematical range.

## Verification tests and acceptance

### CPU unit tests

Cover:

- deterministic eligibility sampling and 768/256 splitting;
- deterministic Stage-2 block membership, parent manifests, and expanded
  1,536/512 cardinalities;
- fixed hashes for a small manifest fixture;
- fixed `n=4` prompt/slot compound-key ordering and rejection of repeated
  `n=1` emulation;
- prompt-embedding chat-template arguments, including the assistant generation
  prefix with thinking disabled;
- benchmark exact/10-token-gram collision rejection;
- largest-remainder stratified allocation;
- exact-size uniform and stratified random subsets, independent spawned RNG
  streams, fixed draw-order hashes, and `B=100/10,000` percentiles;
- seeded input-permutation behavior for seeds 42 and 43 through an injected
  CPU fake cluster manager, and balanced-round-robin determinism for the
  experiment's fixed seed 42;
- projection linearity: `P(mean(g_i)) == mean(P(g_i))` within tolerance;
- pinned projector dtype, constructor fields, and `1/sqrt(1,024)` output scale;
- hand-computed G-Vendi and held-out facility-coverage fixtures;
- inclusive random-percentile calculation;
- worst-case seed aggregation, four-gate `P_n1` conjunction, and all
  pass/fail/inconclusive classifications;
- CKA ID-permutation behavior;
- unbiased CKA formula, fixed permutation schedule, and plus-one p-value ties;
- atomic shard-local chunk/resume validation, exact global coverage, untracked
  source snapshot hashing, and provenance rejection.

### Real-model smoke acceptance

On the 32-question smoke, require:

- exact pinned official-reference CUDA K-means import and deterministic
  selector outputs for seeds 42 and 43;
- exact ID/seed/slot coverage;
- exact vocabulary-to-ID compatibility for each teacher/student pair;
- exact equality of masks and token IDs between capture and replay;
- replay-recomputed `batch_old_log_prob`, `current_log_prob`, and
  `local_old_log_prob` close to their captured actor values with
  `rtol=5e-3, atol=5e-3` on valid tokens;
- `rollout_log_prob` and `ref_log_prob` tensor bytes, masks, and parent hashes
  survive capture serialization/replay loading exactly; standalone replay does
  not claim to regenerate either vLLM sampling probabilities or teacher scores,
  and never compares them to FSDP actor probabilities;
- rollout IS equals detached, mask-aware
  `min(exp(clamp(batch_old_log_prob-rollout_log_prob,-20,20)),5)` with no
  configured lower IS-threshold clamp or normalization;
- capture/replay stopped advantages and per-trajectory token-mean losses close
  with the same tolerance;
- projected gradient cosine at least `0.999` and relative norm error at most
  `0.005` between the direct production-loss backward fixture and standalone
  replay under the same unscaled per-trajectory convention;
- actor parameter hashes identical before and after capture;
- target four-gradient average equals the accumulated-and-divided group
  gradient projection within `rtol=5e-3, atol=5e-3`;
- proxy projected-gradient mean equals its `P_n4` artifact within the same
  tolerance;
- a stop-and-resume exercise reproduces uninterrupted artifact hashes.

The main probe cannot start until every smoke gate passes.

### Main-run completion acceptance

Require:

```text
sample rows:                       1,024
candidate / held-out:              768 / 256
target question-seed vectors:      2,048
target captured trajectories:      8,192
proxy per-trajectory vectors:      6,144
SFT vectors:                         768
embedding vectors:                   768
uniform / stratified random sets: 10,000 / 10,000
selected size for every arm:         172
```

If Stage 2 fires, its corresponding completion values are 2,048 sample rows,
1,536/512 candidate/held-out rows, 4,096 target question-seed vectors, 16,384
target trajectories, 12,288 proxy vectors, 1,536 SFT vectors, 1,536 embedding
vectors, 10,000 subsets per random null, and 345 selected IDs per arm.

The final JSON and Markdown reports must agree on every scalar and
classification and must include results for both target seeds, both K-means
seeds, all eight `P_n1` realizations, both `P_n4` realizations, SFT, embedding,
oracle, uniform random, and stratified random.

## Launch boundary

Implementation may run CPU preparation and unit tests locally. Any task that
loads a CUDA model, performs model rollout, computes GPU gradients, or invokes
the official CUDA-only K-means must be launched from the user-provided
four-GPU `opd-CLI` allocation. Login-shell GPU execution is forbidden.

No GPU launch is authorized by approval of this design alone. GPU work begins
only after the implementation plan is approved and the implementation passes
its CPU/preflight gates.

## Related experimental precedents

The comparison structure is informed by:

- Prismatic Synthesis / G-Vendi, which evaluates gradient diversity against
  embedding, lexical, perplexity, and skill-diversity baselines:
  https://arxiv.org/html/2505.20161v1
- LIMR, which compares a selected RL subset against a same-size random subset
  and the full dataset:
  https://arxiv.org/html/2502.11886
- GradAlign, which compares policy-gradient selection with random,
  accuracy-near-0.5, and within-training-gradient-alignment baselines:
  https://arxiv.org/html/2602.21492
- Neuron-OPSD, a direct OPD data-selection/context-curation study whose own
  analysis finds that its neuron-count selection signal is not sufficient by
  itself:
  https://arxiv.org/html/2607.02460

None of these establishes that adjacent-scale proxy-OPD gradient diversity
beats random for strong-to-weak vanilla OPD. That is the narrow question this
verify is designed to answer.
