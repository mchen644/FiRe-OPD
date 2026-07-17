# Prismatic-Lite Qwen3 Generation Pilot Design

**Date:** 2026-07-17
**Status:** Approved for unattended execution; stop only on a declared fail-closed condition

## Objective

Test whether a low-cost adaptation of Prismatic Synthesis can generate genuinely novel, quality-controlled math questions that fill sparse regions of the existing DeepMath gradient space. The pilot first checks whether replacing the original R1 solution with a local Qwen3 solution changes gradient-space membership so strongly that clustering would measure solution style rather than problem reasoning. Only if that calibration passes does it generate and curate exactly 2,000 novel candidate questions.

The pilot performs no OPD training, does not modify or remove any of the 57,046 source rows, does not overwrite the frozen 57,045 projected gradients, and does not claim downstream improvement.

## Relationship to the Original Paper

The design preserves the central Prismatic Synthesis loop:

1. cluster the current pool in projected gradient space;
2. generate new problems from five examples in the current pool;
3. generate multiple solutions and apply a majority-answer quality gate;
4. compute gradients for quality-passed new samples;
5. retain only new samples assigned to sparse existing clusters;
6. append retained samples conceptually to the pool before a future iteration.

The pilot deliberately remains smaller than the paper and uses local models:

| Component | Paper | Pilot |
|---|---|---|
| Seed math pool | 94K OpenR1-Math | 57,045 eligible DeepMath rows |
| Problem generator | Qwen2.5-72B-Instruct | local Qwen3-30B-A3B-Instruct-2507 |
| Solution generator | R1-Distill-Qwen-32B | local Qwen3-30B-A3B-Instruct-2507 |
| Solutions per problem | 3 | 3 |
| Majority threshold | exactly 2 of 3 | exactly 2 of 3 |
| Gradient proxy | Qwen2.5-0.5B-Instruct | same cached model and revision |
| Projection | 1,024-d Rademacher | same frozen configuration |
| Primary cluster count | 1% of current pool | K=570 |
| Sparse region | smallest 50% of clusters | exactly 285 clusters |
| Scale | iterative growth to 1M | calibration plus 2K candidates only |
| Downstream objective | SFT | none in this pilot; future target is OPD |

The paper text specifies K as 1% of the current pool. Its public single-iteration example uses 10%; this pilot follows the paper text for primary membership and treats 0.5% and 2% only as sensitivity diagnostics.

## Frozen Inputs

- Source parquet: `/home/mchen/FiRe-OPD/data/g-opd/DeepMath-103K/train_filtered_level6.parquet`
- Source SHA-256: `de3350fdd00bc0410550098ea65179e2be873da99e4075f80de575fc17670597`
- Prepared problem/solution JSONL: `/home/mchen/FiRe-OPD/data/gradient_diversity/deepmath_level6_r1_solution1.jsonl`
- Prepared JSONL SHA-256: `ee8d55c943577888f14052b3953a0a8aa07d26076e50ddffa41a102b60027344`
- Eligible rows: 57,045
- Frozen gradient manifest SHA-256: `9a534118a08e736a15e806933d5cf90c2a99822fee28bf18644e5ff344ce3ae2`
- Frozen gradient model: `Qwen/Qwen2.5-0.5B-Instruct` at revision `7ae557604adf67be50417f59c2c2f167def9a775`
- Frozen projection: full-parameter completion-only gradients, 1,024 dimensions, Rademacher seed 0
- Local generator/solver: `/home/mchen/FiRe-OPD/models/Qwen3-30B-A3B-Instruct-2507`
- Benchmark inputs: the same frozen canonical-eight JSONLs and hashes in the approved canonical-eight evaluation design

All source and gradient hashes are rechecked before every publish boundary.

## Artifact Namespace

Use a new append-only namespace:

```text
/home/mchen/FiRe-OPD/data/prismatic_lite/qwen3_2k_pilot/
/home/mchen/FiRe-OPD/logs/prismatic_lite/qwen3_2k_pilot/
```

Planned data artifacts:

```text
manifest.json
calibration/sample.jsonl
calibration/solutions.jsonl
calibration/paired_gradients/
calibration/report.json
candidates/problems.jsonl
candidates/solutions.jsonl
candidates/quality_passed.jsonl
candidates/new_gradients/
selection/cluster_state_seed42.npz
selection/cluster_state_seed43.npz
selection/accepted.jsonl
selection/rejected.jsonl
report.json
report.md
STAGE_COMPLETE.json
```

Every JSONL record has a stable ID, parent demonstration IDs, model revision, generation seed, sampling parameters, and prior-stage provenance. Partial artifacts remain noncanonical and cannot satisfy completion.

## Stage A: Solution-Style Calibration

### Sampling

Select exactly 256 unique eligible DeepMath rows with seed 42, stratified by topic and difficulty. Preserve their original R1 solutions and frozen projected gradients.

### Qwen3 solutions

For each calibration problem, generate three independent solutions with:

- Qwen3-30B-A3B-Instruct-2507;
- temperature 0.75;
- top-p 0.95;
- maximum response length 16,384;
- explicit per-sample generation seeds;
- no hidden or implicit retry that changes membership.

Parse final answers with `math_verify`. A problem is calibration-qualified only when at least two non-null answers are mathematically equivalent. Require at least 192 of 256 rows to qualify; otherwise classify the solver protocol as `no_go`.

### Paired gradients

For every qualified problem, compute a projected completion-only gradient for each majority-matching Qwen3 solution with the exact frozen Qwen2.5 proxy model and projection contract. Normalize each solution gradient, average the majority-matching vectors, and normalize the resulting question-level vector. Do not alter or recompute the original R1-solution gradient.

### Cluster and null analysis

Cluster all 57,045 frozen original gradients with cosine K-means, K=570, 20 iterations, for seeds 42 and 43. Define sparse clusters as exactly the 285 smallest clusters, sorting ties by cluster ID. Assign each paired Qwen3-solution gradient to the nearest frozen centroid.

For the qualified calibration rows, report:

- paired cosine similarity between original and Qwen3-solution gradients;
- exact cluster-label agreement;
- sparse-versus-dense classification agreement;
- sparse/dense transition matrix;
- corresponding statistics under 10,000 topic-and-difficulty-stratified random pairings, each formed by independently permuting the qualified Qwen3 vectors within the exact `(topic, difficulty)` strata and falling back to topic-only strata when a stratum has fewer than two rows.

Calibration passes only if all conditions hold for both clustering seeds:

1. at least 192 qualified rows;
2. sparse/dense agreement is at least 0.65;
3. sparse/dense agreement exceeds the median stratified-null agreement by at least 0.15;
4. paired cosine median is above the 90th percentile of stratified-null paired-cosine medians;
5. all gradients and centroids are finite and nonzero.

Exact cluster-label agreement is diagnostic because nearby centroids may exchange labels without changing sparse/dense status. Any failed primary condition terminates the pilot before 2K problem generation.

## Stage B: Generate Exactly 2,000 Candidate Problems

### Demonstration sampling

For every generation request, sample five unique examples from the eligible DeepMath pool using seed-derived deterministic streams. Sampling probabilities are proportional to the frozen DeepMath difficulty values, matching the public paper implementation. Save all five parent IDs.

### Problem generation

Use Qwen3-30B-A3B-Instruct-2507 with:

- normal chat template with thinking override disabled;
- temperature 1.0;
- top-p 0.95;
- maximum generated tokens 8,192;
- one generation per request;
- a prompt requiring a novel, non-multiple-choice problem of similar or greater difficulty.

Parse candidate boundaries deterministically. Continue until exactly 2,000 unique, syntactically valid candidate problems are published, with a hard cap of 3,000 generation requests. Failure to reach 2,000 before the cap is `no_go`.

## Stage C: Candidate Quality Control

Generate three Qwen3 solutions per candidate under the calibration solution protocol. Preserve all raw responses. A candidate passes answer consistency only when at least two non-null parsed answers are mathematically equivalent. The implementation must use an integer threshold of two, not `int(3 * 0.5)`.

Apply the following additional gates:

1. normalized exact duplicate rejection against the eligible pool and earlier candidates;
2. tokenize lowercase Unicode-normalized text on whitespace and punctuation, retrieve seed/candidate neighbors through a 10-gram inverted index, and send the highest-overlap pair to semantic review when 10-gram Jaccard is at least 0.30;
3. reject benchmark contamination immediately when any normalized contiguous 10-gram is shared with an evaluation prompt;
4. for only the candidate/seed or candidate/candidate pairs surfaced by item 2, ask the same frozen Qwen3 model for a deterministic `equivalent`/`not_equivalent` judgment and reject `equivalent` candidates;
5. maximum Qwen3 prompt length 2,048 tokens;
6. nonempty problem, completion, and majority answer;
7. immutable audit record for every rejection reason.

Do not silently regenerate a failed problem. Quality-passed membership is determined only by the frozen 2,000 candidates.

For every quality-passed question, create its question-level projected gradient by averaging normalized majority-solution gradients as in calibration.

## Stage D: Sparse-Cluster Filtering

Use the Stage-A K=570 cluster states. Assign every quality-passed question-level gradient to the nearest centroid for seeds 42 and 43.

- Primary accepted membership: assigned to a seed-42 sparse cluster.
- Seed-43 assignment: diagnostic and stability gate only.
- K=285 and K=1,140: sensitivity diagnostics only.

Publish accepted and rejected records in original candidate order. Never select old DeepMath rows and never remove them; accepted records are novel additions.

The generation pilot is operationally promising only if:

1. at least 1,000 of 2,000 candidates pass all quality gates;
2. at least 400 candidates pass the primary sparse-cluster filter;
3. seed-42/43 sparse/dense assignment agreement on quality-passed candidates is at least 0.65;
4. the G-Vendi of `original pool + accepted additions` is above the 90th percentile of 1,000 equal-size additions sampled from the same quality-passed candidate pool;
5. accepted prompts are unique and benchmark contamination is zero.

These thresholds determine only whether a larger generation stage is worth considering. They do not establish OPD efficacy.

## Runtime and Safety

All model work runs in the existing or newly authorized `opd-CLI` Slurm allocation, without nested `srun`. CPU preflight must complete first. Immediately before each GPU phase require:

- active allocation and sufficient wall time;
- exact allocation-owned tokens `0,1,2,3` idle;
- missing output targets;
- clean executable source tree;
- frozen input hashes;
- no stale Qwen3, vLLM, proxy-gradient, or K-means process.

Qwen3 generation uses all four GPUs as one TP=4 engine. Qwen2.5 proxy-gradient collection may use the allocation in four deterministic shards, but must reproduce the frozen projector and model identity. Qwen3 and Qwen2.5 phases are sequential and must fully drain between model swaps.

Resume and overwrite are forbidden. A failed phase is archived under a timestamped failed-attempt directory before any source correction or relaunch. No accepted data is mixed into production training data during this pilot.

## Validation and Reporting

The final report records:

- exact source, model, code, and environment identities;
- generation throughput and GPU hours;
- parse, majority, duplicate, contamination, and sparse acceptance rates;
- calibration paired/null metrics;
- K-means cluster-size and seed-stability diagnostics;
- accepted-set G-Vendi and random-null percentile;
- all artifact hashes and row counts;
- a deterministic decision: `promising`, `no_go`, or `incomplete`.

`STAGE_COMPLETE.json` is published only after process drain, artifact validation, source cleanliness, and report consistency checks. The original 57K pool and all prior selection/evaluation artifacts remain byte-identical.

## Explicit Non-Goals

- No OPD or SFT training.
- No claim that generated questions improve downstream accuracy.
- No replacement or deletion of original DeepMath rows.
- No generation beyond 2,000 candidates without a new approved design.
- No relaxation of majority, contamination, calibration, or sparse-acceptance thresholds after observing results.
- No reuse of the interrupted canonical-eight evaluation as evidence for this pilot.
