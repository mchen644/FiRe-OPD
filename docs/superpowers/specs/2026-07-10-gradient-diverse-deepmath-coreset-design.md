# Gradient-Diverse DeepMath Coreset Design

Date: 2026-07-10

## Context

The current group-success OPD run keeps the per-step trajectory budget fixed at 1024 by changing the sampling layout from 1024 independent questions with one rollout each to 256 independent questions with four rollouts each. Across 50 optimizer steps, the number of question slots therefore falls from 51,200 to 12,800. Its AIME24 pass@32 remains comparable to vanilla OPD, while sample accuracy and Avg@4 are lower. Reduced question coverage is a plausible contributor, although concise teacher prompts, ESR truncation, and correlated gradients among same-question rollouts remain separate confounders.

This experiment isolates the coverage hypothesis. It selects exactly 12,800 unique questions from the 57,046-row `DeepMath-103K/train_filtered_level6.parquet` pool using gradients from a small proxy model, without changing the OPD rollout, teacher, or optimizer budget.

The implementation is grounded in the authors' official supplementary repository for *Prismatic Synthesis: Gradient-based Data Diversification Boosts Generalization in LLM Reasoning*:

```text
repository: https://github.com/jaehunjung1/prismatic-synthesis
commit:     d9484cd3b5991030b901ac4a3a9e2472dbfac2ad
```

The reference repository is cloned outside the FiRe-OPD worktree at:

```text
/home/mchen/prismatic-synthesis-reference
```

## Objective

Produce a reproducible, gradient-diverse DeepMath coreset with:

```text
source questions:                 57,046
selected unique questions:       12,800
intended OPD prompt batch size:      256
intended optimizer steps:             50
rollouts per selected question:        4
trajectories per optimizer step:    1,024
```

The selected parquet must preserve the exact Arrow schema of the current filtered training parquet and contain no duplicate source rows.

## Non-goals

This stage does not:

- modify the group-success difficulty route
- launch a new OPD training run
- use AIME, HMMT, or any evaluation example to target selection
- synthesize new questions or solutions
- compute influence against a target validation set
- train a sparse autoencoder
- apply token cleaning
- use student rollout correctness, confidence, or length for offline selection
- copy the unlicensed reference source into FiRe-OPD
- claim that the fixed-pool selector is an official Prismatic Synthesis implementation

## Inputs and pinned provenance

### Current OPD pool

```text
data/g-opd/DeepMath-103K/train_filtered_level6.parquet
```

Expected row count: `57,046`.

### Original DeepMath metadata and solutions

Hugging Face dataset:

```text
zwhe99/DeepMath-103K
revision: 5cf055d1fe3d7a2eb19719ac020211469736ae44
```

Required columns:

```text
question
final_answer
difficulty
topic
r1_solution_1
```

Only `r1_solution_1` is used in this first experiment. The two other R1 solutions remain unused to keep preprocessing cost to one forward/backward pass per question.

### Gradient proxy

```text
Qwen/Qwen2.5-0.5B-Instruct
revision: 7ae557604adf67be50417f59c2c2f167def9a775
```

## Data join and validation

The current parquet stores a chat prompt whose user content is the original question plus the fixed OPD instruction:

```text
Please reason step by step, and put your final answer within \boxed{}.
```

Preparation removes only this exact terminal instruction and pairs the remaining exact question with the normalized ground-truth answer. It must not use fuzzy question matching.

The pinned original dataset contains duplicate question/answer keys, so row-local lookup is insufficient. The filtered parquet was produced by order-preserving filtering and must be recovered as a unique strictly increasing subsequence of original rows. Preparation computes both:

- the forward greedy mapping that chooses the earliest matching original index after the previous match
- the backward greedy mapping that chooses the latest matching original index before the next match

The two mappings must exist and be identical at every filtered row. This admits duplicated original questions only when surrounding row order identifies one unique source row.

Preparation fails before GPU work when:

- the current parquet row count is not 57,046
- the current Arrow schema lacks the expected prompt, reward, or index fields
- a filtered `(exact question, normalized answer)` key has no original match
- forward and backward ordered-subsequence mappings disagree, indicating genuine ambiguity
- `r1_solution_1` is empty
- the current ground-truth answer and original `final_answer` disagree after the existing math-answer normalization
- prepared IDs are not unique

Each prepared JSONL row has:

```json
{
  "id": "deepmath-level6-000000",
  "prompt": "<original question>",
  "completion": "<r1_solution_1>",
  "source_row_index": 0,
  "original_dataset_index": 0,
  "topic": "<hierarchical topic>",
  "difficulty": 6.0
}
```

The output ID is derived from the current parquet row index so it remains stable and maps directly back to the training row.

## Official-aligned gradient representation

The gradient collector reuses the official `GradientComputer` class rather than reimplementing gradient semantics. Before importing it, the wrapper verifies that the reference repository is clean and at the pinned commit.

The inherited official behavior is:

- Qwen chat template
- completion-only cross-entropy loss
- prompt labels masked with `DataCollatorForCompletionOnlyLM`
- full backward through every trainable proxy-model parameter
- Rademacher projection through TRAK
- projection dimension `1024`
- projection seed `0`
- projection in groups of four examples
- float16 projected vectors
- output split into resumable `.safetensors` files with ID sidecars

FiRe-OPD adds only orchestration and validation:

- explicit `--shard-index` independent of physical CUDA device ID
- one physical GPU exposed per collector process, with the process-local device fixed to `cuda:0`
- pinned model revision
- exact shard-bound computation compatible with the reference implementation
- manifest validation preventing resume with a different dataset hash, proxy revision, reference commit, projection dimension, or shard count
- strict chunk-pair, sidecar-ID, tensor-shape, and contiguous-prefix validation before resume
- fail-fast refusal when the official fast-JL probe would fall back to TRAK `BasicProjector`
- preflight tokenization to ensure every prompt/completion fits the proxy context window
- a non-empty completion-label check after the official completion-only collator runs
- final check that the stored gradient IDs exactly equal the prepared dataset IDs

No truncation is silently applied. The hard context boundary comes from the pinned model config (`max_position_embeddings=32768`), not the tokenizer's larger advertised limit. If a sample exceeds it, preparation stops and reports its ID and token count before gradient collection begins.

## Paper/code discrepancy policy

The paper states that Prismatic Synthesis dynamically uses a cluster count equal to 1% of the current pool. The official released `cluster_filter.py` uses 10%. Both are evaluated offline from the same stored gradients:

```text
primary code-aligned ratio:    0.10
sensitivity paper ratio:       0.01
K-means iterations:              20
distance:                 cosine
```

The selected training parquet uses the code-aligned `K = floor(0.10 * N)`. The paper-aligned result is diagnostic only and does not produce a second training dataset in Stage 1.

For each ratio, clustering runs twice with deterministic input permutations controlled by seeds 42 and 43. This is necessary because the released implementation initializes centroids from the first `K` rows. Diagnostics include:

- requested and non-empty cluster counts
- min, median, mean, p90, p99, and max cluster sizes
- singleton-cluster ratio
- largest-cluster fraction
- adjusted Rand index between the two seeded runs
- selected-set Jaccard overlap between the two seeded runs
- selected-set G-Vendi
- selected topic coverage and entropy
- selected difficulty distribution

The primary coreset uses ratio 0.10 and seed 42.

## Exact fixed-pool selection

The official repository implements sparse-cluster rejection for newly generated samples, not exact-size selection from a fixed pool. The new fixed-pool selector is therefore an explicit adaptation of its gradient representation and cosine K-means, not copied official behavior.

Given cluster labels, the selector performs deterministic cluster-balanced round-robin sampling:

1. Group sample indices by cluster.
2. Shuffle members within each cluster with seed 42.
3. Shuffle the active cluster order with seed 42 at the start of each round.
4. Take at most one remaining member from every active cluster in a round.
5. Continue until exactly 12,800 unique rows are selected.

This gives sparse clusters proportionally more representation than random sampling while avoiding a top-outlier-only rule. It is the deterministic analogue of the paper's cluster-guided balanced sampling.

The selector fails when:

- fewer than 12,800 unique gradients are available
- a prepared ID has no gradient or an unknown gradient ID is present
- a projected gradient contains NaN or infinity
- any gradient has zero norm
- clustering returns a label outside `[0, K)`
- selection contains duplicate IDs
- selected source indices are out of range
- output schema differs from the input schema

## Outputs

Runtime artifacts live under ignored `data/` and `logs/` paths:

```text
data/gradient_diversity/deepmath_level6_r1_solution1.jsonl
data/gradient_diversity/deepmath_level6_r1_solution1.manifest.json
data/gradient_diversity/gradients/qwen2.5-0.5b-instruct/
data/gradient_diversity/selection/diagnostics.json
data/gradient_diversity/selection/selected_ids.jsonl
data/gradient_diversity/DeepMath-103K/train_gradient_diverse_12800.parquet
logs/gradient_diversity/deepmath_gradient_diverse_12800.log
```

The final manifest records:

- SHA-256 hashes of the source parquet and prepared JSONL
- source and selected row counts
- Hugging Face dataset/model revisions
- official reference repository commit
- all projection, clustering, and selection hyperparameters
- selected ID hash
- output parquet hash
- diagnostic results for both cluster ratios and both seeds

## Components

### Pure preparation and selection helpers

`math_eval/deepmath_gradient_diversity.py` contains deterministic, CPU-testable helpers for prompt normalization, exact joining, shard bounds, resume validation, balanced selection, manifest hashing, and Arrow-row reconstruction.

### Preparation CLI

`math_eval/prepare_deepmath_gradient_pool.py` loads the pinned original dataset, validates the exact join, writes the prepared JSONL, and writes its provenance manifest.

### Gradient collection CLI

`math_eval/collect_prismatic_gradients.py` validates provenance, imports the pinned official `GradientComputer`, loads one explicit data shard, and writes official-format projected-gradient chunks.

### Selection CLI

`math_eval/select_gradient_diverse_deepmath.py` loads and validates all stored gradients, runs official cosine K-means for both ratios/seeds, computes diagnostics, selects the primary 12,800 rows, and writes the final parquet and manifest.

### Production launcher

`run_select_gradient_diverse_deepmath.sh` performs preparation, parallel gradient collection on configured physical GPU IDs, exact gradient coverage validation, and final selection. It is resumable and fails fast on provenance mismatches.

## Runtime and tmux policy

The production job starts in the existing `opd-CLI` tmux session without interrupting any running evaluation process. It uses only GPUs verified free immediately before launch. The launcher logs the physical GPU IDs and refuses duplicate concurrent launches for the same output directory.

If fewer GPUs are free than requested, the number of data shards follows the number of selected GPUs; physical CUDA IDs and logical shard IDs remain independent. Relaunching a partially completed job must use the same shard count recorded in the gradient manifest.

## Validation

Before the full job starts:

1. CPU unit tests cover exact join failures, sharding, manifest mismatch, balanced selection, exact row count, uniqueness, deterministic output, and schema preservation.
2. A tiny synthetic safetensors fixture validates gradient ID loading and finite/non-zero checks.
3. A one-GPU smoke test computes official projected gradients for four prepared samples and verifies shape `(4, 1024)`, finite values, expected IDs, and resume behavior.
4. A tiny GPU clustering smoke test runs both cluster ratios on fixture gradients.

The full job is considered successfully launched only after preparation completes, the pinned reference/model metadata is logged, at least one gradient safetensors chunk is written, and the process remains alive without traceback.
