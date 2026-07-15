# One-Time Vanilla-OPD Proxy-Gradient Efficacy Pilot Design

**Status:** approved design amendment for implementation

**Parent experiment:** `2026-07-14-vanilla-opd-proxy-gradient-selection-verify-design.md`

**Base source commit:** `174849613a9c61445765f0b913174d286f3819e4`

**Implementation branch:** `opd-efficacy-pilot`

## 1. Purpose and scope

Insert one one-time `efficacy_pilot` stage after the completed Stage 0 smoke and before any Stage-1 GPU work. The pilot answers one narrow question:

> Does selection with one proxy Vanilla-OPD gradient per question produce a subset that is clearly better than random subsets in one target Vanilla-OPD gradient realization?

The pilot is a compute-allocation gate, not the main scientific experiment. It does not establish downstream training improvement, OOD improvement, cross-seed robustness, or the preregistered Stage-1 hypothesis. Its report always has status `pilot_only` and emits only an operational `go`, `no_go`, or `borderline` decision.

The pilot may decide whether it is worthwhile to spend GPU time on Stage 1. It must not modify Stage-1 memberships, artifacts, thresholds, interpretation, or classification logic.

## 2. Immutable parents and branch isolation

The following existing objects are immutable:

```text
Stage-0 published artifacts: unchanged
Stage-1 manifest:            data/opd_proxy_gradient_verify/stage_1/manifest.json
Stage-1 manifest SHA-256:    6d698d75995c777d6faaf1abd385a758977ca0350ce862284f1e9d8751eddd83
Stage-1 source commit:       174849613a9c61445765f0b913174d286f3819e4
clean population:            56,662 stable IDs
sampling permutation:        unchanged
```

The Stage-1 manifest binds source files that must be modified to add a first-class pilot profile. Therefore pilot implementation occurs on branch `opd-efficacy-pilot`, forked from the Stage-1 source commit. Pilot artifacts bind a new clean pilot source commit and pilot source snapshot. Future Stage-1 execution must switch the same isolated worktree back to branch `opd-proxy-gradient-verify-impl` at commit `174849613a9c61445765f0b913174d286f3819e4`.

This is deliberate dual identity:

- Stage 1 retains its original source identity and manifest bytes;
- the pilot has its own source identity and manifest bytes;
- the pilot records the Stage-1 manifest as an immutable data-membership parent;
- neither source identity is presented as equivalent to the other.

A pilot source correction invalidates pilot runtime artifacts only. It does not authorize deleting, replacing, or republishing Stage-0 or Stage-1 artifacts.

## 3. First-class stage profile

The shared preparation and orchestration entry points gain a first-class string stage kind:

```text
efficacy_pilot
```

The CLI spelling is:

```text
--stage efficacy_pilot
```

The canonical namespace is separate from every numbered stage:

```text
data/opd_proxy_gradient_verify/efficacy_pilot/
logs/opd_proxy_gradient_verify/efficacy_pilot/
```

The implementation must not encode the pilot as Stage 3, a negative integer, or an alias of Stage 0. Integer ordering and `stage - 1` prerequisite logic remain exclusive to numbered stages.

A shared immutable profile supplies all cardinality- and execution-dependent fields:

| Field | `efficacy_pilot` |
|---|---:|
| candidate rows | 250 |
| held-out rows | 84 |
| selected rows | 56 |
| primary K | 25 |
| diagnostic K | 3 |
| random draws per null | 10,000 |
| generation seeds | `[42]` |
| native rollouts per question | 1 |
| proxy representations | `P_pilot` |
| target representations | `T_pilot` |
| SFT baseline | disabled |
| embedding baseline | disabled |
| target-oracle selection | disabled |
| direct fixture | disabled |
| Stage-0 resume exercise | disabled |

The profile is consumed by existing prepare, run, selector, analyzer, validation, and dry-run paths. No separate pilot-only executable is introduced.

## 4. Parent-derived membership

### 4.1 Required parent arguments

Canonical pilot preparation requires:

```text
--parent-stage1-manifest <path>
--expected-parent-stage1-manifest-sha256 <64-hex digest>
```

The expected digest for this one-time pilot is the frozen value:

```text
6d698d75995c777d6faaf1abd385a758977ca0350ce862284f1e9d8751eddd83
```

Preparation fails before writing if:

- the parent path is missing;
- the parent digest differs;
- canonical JSON validation fails;
- `stage != 1`;
- any frozen Stage-1 cardinality or ordered-ID hash differs;
- any parent-referenced sample or capture-input artifact is absent or has the wrong hash;
- the parent source/data/model provenance is malformed.

### 4.2 Exact prefix rule

Read the Stage-1 sample manifest in its existing order. Derive:

```text
candidate rows: first 250 rows whose Stage-1 split is candidate
held-out rows:  first 84 rows whose Stage-1 split is held_out
```

The pilot output order is those 250 candidate rows followed by those 84 held-out rows. Within each split, order must be byte-for-byte parent order. There is no shuffle, resampling, replacement, stratification, or fresh tokenization decision.

Each pilot sample row records:

- a local contiguous `manifest_index` in `0..333`;
- the immutable `parent_manifest_index`;
- the parent stable ID and split;
- the parent row logical hash;
- all existing exact prompt, metadata, and source fields.

The child records new local indices only to satisfy existing contiguous artifact contracts. It never rewrites the parent row or claims the local index was the Stage-1 index.

### 4.3 Parent-child hashes

The pilot manifest records and validates:

- parent Stage-1 manifest absolute path and SHA-256;
- parent sample manifest path and SHA-256;
- ordered parent candidate and held-out ID hashes;
- ordered pilot candidate-prefix ID hash;
- ordered pilot held-out-prefix ID hash;
- ordered pilot candidate-then-held-out ID hash;
- logical hashes linking every pilot row to its parent row.

Pilot capture-input parquet files are newly published in the pilot namespace from the verified parent rows. This reuses prepared question identity and metadata, not GPU computation. No pilot file is written beneath `stage_1/`.

The derivation is append-only in the artifact sense: the child namespace is create-once and points to an immutable parent; the parent is never mutated.

## 5. Algorithm contract

Except for the two explicitly approved deviations below, the pilot capture and replay contract must equal the Stage-1 contract field for field.

### 5.1 Only approved deviations

```text
rollout.n:        Stage 1 = 4; pilot = 1
generation seeds: Stage 1 = [42, 43]; pilot = [42]
```

These deviations apply to both target and proxy capture.

### 5.2 Fields that remain frozen

```text
temperature / top-p:                    1.0 / 1.0
maximum prompt length:                  2,048
maximum response length:                16,384
chat-template thinking:                 disabled
policy loss mode:                       vanilla
only reverse-KL advantages:             true
PPO epochs:                             1
mini-batches per actor rank:            exactly 1
micro-batch size per GPU:               1
dynamic micro-batching:                 false
loss aggregation:                       token mean
rollout correction:                     token-level IS
rollout IS upper threshold:             5.0
rollout IS batch normalization:         false
entropy coefficient:                    0
use explicit KL-loss teacher path:      true
explicit KL-loss coefficient:           0
KL in reward:                           false
candidate selection:                    disabled
length-aware OPD:                       disabled
difficulty-aware OPD:                   disabled
TALE / ESR:                             disabled
teacher prompt routing:                 disabled
rethinking probe:                       disabled
actor optimizer construction/step:      forbidden
gradient clipping:                      forbidden
parameter update:                       forbidden
```

The maximum response length must not be reduced for the pilot.

A contract-projection test removes only stage identity, `rollout.n`, and generation-seed fields from Stage-1 and pilot contracts and requires exact equality of every remaining field. This prevents an accidental unlisted pilot shortcut.

### 5.3 Systems-only FSDP dispatch padding

The exact target and proxy trajectory counts, 334 and 250, are not divisible by four FSDP ranks when native `n=1`. Internal actor and reference RPC dispatch therefore pads to 336 and 252 rows, respectively, using marked duplicate prefix rows. This is transport padding, not an algorithm deviation or additional trajectory:

- generation still returns exactly one trajectory for each of the 334 or 250 ordered questions;
- rollout correction and the published trainer boundary consume only the exact unpadded rows;
- every FSDP rank executes an equal number of forwards, including on resume;
- marked padding rows are never written to actor chunks, rank provenance, compound-key coverage, or completion manifests;
- controller outputs are unpadded back to exact order and cardinality before finalization; and
- any non-suffix marker, non-minimal dispatch count, leaked row, missing row, duplicate persisted key, or order change fails closed.

The actor PPO mini-batch field records the internal dispatch count (336 for target and 252 for proxy), while `expected_questions` remains the scientific count (334 and 250). There is still exactly one mini-batch per actor rank and micro-batch size one.

## 6. Representations and replay

The pilot produces exactly two representations.

### 6.1 Proxy `P_pilot`

```text
teacher:    Qwen3-4B
student:    Qwen3-0.6B
questions:  250 candidates
trajectory: one seed-42 rollout per question
gradient:   one full-parameter Qwen3-0.6B Vanilla-OPD gradient
projection: one projected vector per question
```

There is no rollout slot choice and no four-trajectory average. `P_n1`/`P_n4` names are not used in pilot artifacts.

### 6.2 Target `T_pilot`

```text
teacher:    Qwen3-30B-A3B-Instruct-2507
student:    Qwen3-4B
questions:  250 candidates + 84 held-out
trajectory: one seed-42 rollout per question
gradient:   one full-parameter Qwen3-4B Vanilla-OPD gradient
projection: one projected vector per question
```

The target vector is the single trajectory gradient. There is no slot average or group average.

### 6.3 Replay and projection

Replay retains Stage-1 semantics:

- actor-local old log-probability is detached current-policy log-probability;
- captured batch-old remains an integrity check only;
- the exact stopped advantage and Vanilla-OPD loss are reconstructed;
- only the student receives gradients;
- the canonical unscaled token-mean gradient is used;
- optimizer scaling is removed;
- no clipping, optimizer, or update occurs.

Projection remains:

```text
projector:        pinned TRAK CudaProjector
projection type: Rademacher
dimension:        1,024
projection seed: 0
block size:       128
maximum batch:    16
model_id:         0
input cast:       float16 only at projection boundary
stored output:    float32 divided by sqrt(1,024)
```

Each vector retains full-gradient norm, projected norm, valid-token count, sampled reverse KL, OPD signal RMS, and response length.

## 7. Capture and resume identity

Pilot capture contracts include at least:

```text
stage_type = efficacy_pilot
native_rollouts = 1
generation_seeds = [42]
engine_seed = 42
pair = target | proxy
sample_manifest_sha256
source_snapshot_sha256
algorithm_contract_sha256
model_manifest_sha256
ordered stable-ID hash
```

Stage-1 capture contracts bind `stage=1`, `native_rollouts=4`, and seeds 42/43. Contract comparison and resume validation require exact equality for all identity fields. Consequently:

- a pilot capture cannot satisfy a Stage-1 capture contract;
- a Stage-1 capture cannot satisfy a pilot contract;
- copying or symlinking a completion marker across namespaces is rejected;
- pilot replay cannot consume Stage-1 capture output;
- Stage-1 replay cannot consume pilot capture output.

Namespace separation is an additional guard, not the sole guard.

## 8. Random null schedules

Generate two independent schedules over the 250 pilot candidates. Every subset has 56 positions and draws without replacement.

```text
R_uniform draws:    10,000
R_stratified draws: 10,000
root SeedSequence:  2026071402
child streams:      spawn(2) exactly once
```

Use the existing schedule algorithm and pinned NumPy semantics. The pilot count and selected size naturally produce pilot-specific schedule bytes and hashes. The schedules are independent from Stage-0/1 schedules and are not copied from them.

`R_stratified` uses only:

```text
leaf_topic x rank-based Qwen3-0.6B prompt-token-length quartile
```

Both schedules are evaluated only in `T_pilot`. `R_stratified` is reported as a robustness diagnostic; the go/no-go gate uses `R_uniform` only.

## 9. Selector

The selector receives exactly the 250 candidate vectors from `P_pilot`.

```text
selected size:        56
primary K:            25
diagnostic K:          3
K-means iterations:   20
K-means seeds:        42 and 43
round-robin seed:     42
metric:               cosine after row L2 normalization
sampler:              cluster-balanced round robin
```

Both K-means seeds remain in the report. The implementation cannot select the better realization after evaluation.

Only primary-K selections enter the go/no-go gate. Diagnostic-K selections and their metrics remain visible but cannot change the decision.

`T_pilot` does not enter the selector. It is an evaluation representation only, so the pilot contains no target oracle selection.

## 10. Target-space evaluation

For each primary-K K-means seed, evaluate the selected 56 IDs in `T_pilot` against the paired `R_uniform` and `R_stratified` schedules.

Required metrics are:

1. target-space G-Vendi;
2. held-out target-space coverage over the 84 held-out rows;
3. mean full target-gradient norm;
4. mean target OPD signal RMS.

The analyzer may retain the existing valid-token-count, sampled-reverse-KL, and response-length diagnostics, but they do not enter the pilot gate.

Percentiles use the existing inclusive rule and are represented internally in `[0, 1]`. Thus the written gate values 90, 60, and 25 mean `0.90`, `0.60`, and `0.25` in canonical JSON.

## 11. Unavailable diagnostics

Cross-seed oracle selection and target-seed dependence cannot be computed because the pilot has one target generation seed and one rollout per question. The report must not synthesize placeholder values, reuse Stage-0 values, or run same-realization target selection as a substitute.

It records structured unavailable entries such as:

```json
{
  "cross_seed_oracle": {
    "status": "unavailable",
    "reason": "efficacy_pilot has one generation seed and one rollout per question"
  },
  "target_seed_dependence": {
    "status": "unavailable",
    "reason": "efficacy_pilot has one generation seed and one rollout per question"
  }
}
```

These unavailable diagnostics do not make the pilot decision inconclusive because the pilot decision is intentionally narrower than Stage 1. They do prohibit translating the pilot decision into a Stage-1 pass/fail classification.

## 12. Pre-registered go/no-go gate

Let each primary-K K-means realization `k in {42, 43}` have four `R_uniform` percentiles:

```text
G_k = target G-Vendi percentile
C_k = held-out coverage percentile
N_k = full-gradient-norm percentile
S_k = OPD-signal-RMS percentile
```

Evaluate the decision in this fixed order:

### GO

Return `go` if and only if both K-means seeds satisfy all four conditions:

```text
G_k >= 0.90
C_k >= 0.90
N_k >= 0.25
S_k >= 0.25
```

### NO-GO

Otherwise return `no_go` if either K-means seed satisfies either condition:

```text
G_k <= 0.60
or
C_k <= 0.60
```

### Borderline

Return `borderline` for every remaining case.

The inequalities are inclusive at every stated boundary. Gradient norm or OPD signal below `0.25` alone produces `borderline`, not `no_go`, unless a diversity or coverage no-go condition also holds.

The gate consumes only primary-K, `R_uniform` values. Diagnostic K and `R_stratified` cannot affect it.

## 13. Report contract and `pilot_only` semantics

The report remains an OPD proxy-gradient verification report but identifies the string stage type and pilot mode. Its classification has the form:

```json
{
  "status": "pilot_only",
  "decision": "go",
  "main_hypothesis": "not_evaluated",
  "stage1_thresholds_modified": false
}
```

`decision` is exactly one of `go`, `no_go`, or `borderline`.

The report also includes:

- the four primary metrics and both null percentiles for each K-means seed;
- diagnostic-K metrics;
- both random schedule hashes;
- unavailable-diagnostic reasons;
- parent Stage-1 manifest SHA;
- pilot source snapshot and algorithm-contract hashes;
- exact model and reference provenance;
- counts and representation list.

The analyzer and stage validator reject a pilot report that:

- reports Stage-1 `pass`, `fail`, `classified`, or `inconclusive` status;
- omits `pilot_only`;
- omits either K-means realization;
- uses diagnostic K or `R_stratified` in the decision;
- claims an oracle or target-dependence result;
- changes or restates Stage-1 thresholds as pilot-derived thresholds.

The Markdown report states prominently that the pilot does not establish downstream or OOD improvement and does not classify the main hypothesis.

## 14. Shared orchestrator plan

The shared orchestrator creates only these logical GPU/analysis groups:

1. `capture_target_seed_42` for 334 rows, native `n=1`;
2. `capture_proxy_seed_42` for 250 rows, native `n=1`;
3. target seed-42 replay shards;
4. proxy seed-42 replay shards;
5. strict vector validation for `P_pilot` and `T_pilot`;
6. selector and both random schedules;
7. analysis-input preparation;
8. analyzer and byte-stable JSON/Markdown reports.

It creates no seed-43 capture, no baseline collector, no direct fixture, and no Stage-0 resume exercise.

The pilot prerequisite is:

- immutable Stage-1 manifest present at the exact parent hash;
- Stage-0 completion marker remains valid;
- pilot source checkout matches the pilot manifest source identity;
- all existing Slurm/tmux/four-GPU/idle/interpreter/source checks pass before any real GPU execution.

Preparation, dry-run, plan validation, and CPU preflight do not require or launch GPU processes.

## 15. Failure and publication behavior

Canonical pilot artifacts are create-once. Matching existing bytes validate; mismatched bytes fail without replacement. Incomplete staging blocks silent regeneration under a changed identity.

Preparation fails closed on:

- wrong parent path/hash or changed parent bytes;
- wrong prefix membership/order/split;
- any non-approved algorithm-contract difference;
- wrong `n`, extra seed, reduced response limit, or baseline command;
- source/repository/reference dirtiness;
- source identity collision with Stage 1;
- any output path beneath `stage_1/`;
- an existing incompatible pilot artifact.

Owned-process cleanup, detached-worker draining, immutable work-unit ledgers, and output validation reuse the existing orchestration behavior.

## 16. TDD requirements

Implementation follows red-green-refactor. Required regression coverage includes:

### Manifest derivation

- exact first-250 candidate and first-84 held-out prefixes;
- preserved within-split parent order;
- parent index and row-hash linkage;
- wrong parent SHA rejection;
- changed parent bytes, reordered rows, replacement IDs, wrong split, or wrong artifact hash rejection;
- no writes beneath Stage 1;
- byte-stable repeated preparation.

### Algorithm contract

- pilot native `rollout.n=1` for target and proxy;
- only generation seed 42 exists;
- max response remains 16,384;
- contract equality after removing only the two approved deviations and stage identity;
- optimizer/clipping/update remain forbidden.

### Source and resume isolation

- pilot algorithm-contract/source identity differs from Stage 1;
- pilot capture completion cannot satisfy Stage-1 resume;
- Stage-1 capture completion cannot satisfy pilot resume;
- wrong stage kind, native-rollout count, source snapshot, or sample hash is rejected.

### Representation and orchestration

- exactly `P_pilot` and `T_pilot` vector contracts;
- no `P_n4`, SFT, embedding, second target seed, oracle selector, direct fixture, or resume exercise;
- target count 334 and proxy count 250;
- dry-run contains no GPU launch side effect.

### Selector and random nulls

- selected size 56, primary K 25, diagnostic K 3;
- both K-means seeds and fixed round-robin seed;
- 10,000 independent uniform and stratified draws;
- pilot-specific deterministic schedule hashes;
- target vectors never enter selection.

### Analyzer and decision

- forced `pilot_only` status;
- structured unavailable diagnostics;
- GO boundary and all-realization conjunction;
- NO-GO boundary and any-realization disjunction;
- borderline cases, including low norm/signal alone;
- primary/uniform-only gate input;
- rejection of main-hypothesis pass/fail labels.

Existing numbered-stage CPU tests must remain green. Focused VERL capture tests must prove that native `n=1` still uses one engine and one generation call.

## 17. Documentation and execution deliverables

Implementation updates:

1. the parent protocol's staged-execution section with the inserted one-time pilot and explicit deviation table;
2. the visible implementation/status document with pilot implementation and runtime status;
3. the shared prepare/run/select/analyze code and tests;
4. a canonical pilot manifest and CPU preflight evidence.

Because canonical preparation requires a clean source commit, the tracked protocol and implementation code are committed before publishing the pilot manifest. The resulting manifest SHA is then recorded in an ignored runtime status document rather than changing the source commit that the manifest binds.

No GPU work unit is launched as part of this task. After CPU preflight, report:

- pilot source commit;
- parent Stage-1 manifest SHA;
- pilot manifest SHA;
- deterministic dry-run/plan evidence;
- conservative expected GPU duration and estimation method;
- explicit confirmation that no GPU work unit started.

Then stop for user authorization.
