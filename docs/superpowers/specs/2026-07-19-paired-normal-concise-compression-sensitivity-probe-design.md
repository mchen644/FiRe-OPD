# Paired Normal/Concise Compression-Sensitivity Probe Design

> **Runtime amendment:** The independent same-seed relaxed resampling in Section 6 proved not to preserve token prefixes under dynamically batched TP=4 vLLM. It is superseded by `2026-07-19-paired-normal-concise-forced-prefix-remediation-design.md`, which continues from the exact observed capped token prefix. All other frozen protocol sections remain in force.

## 1. Goal

Run a small, reproducible, inference-only audit on training-distribution questions to determine why a student can solve a question under the normal prompt but fail under the concise prompt. Compare both:

1. the original Qwen3-4B student; and
2. the completed adaptive-concise step-50 student.

The probe must also measure the previously unobservable `normal incorrect / concise correct` quadrant.

## 2. Non-goals

This probe does not:

- update model parameters;
- launch another training run;
- change the completed adaptive experiment or its artifacts;
- claim that a single stochastic paired outcome is an intrinsic property of a question;
- launch any benchmark evaluation beyond this training-distribution diagnostic;
- overwrite an existing probe namespace.

## 3. Frozen inputs

### 3.1 Dataset

Use:

```text
/home/mchen/FiRe-OPD/data/g-opd/DeepMath-103K/train_filtered_level6.parquet
```

The expected SHA256 is:

```text
de3350fdd00bc0410550098ea65179e2be873da99e4075f80de575fc17670597
```

Select exactly 128 distinct rows without replacement using NumPy `default_rng(seed=42)`. Persist the selected source-row indices, `extra_info.index` identities, prompts, and ground truths before GPU generation. Both models must use the identical persisted sample.

### 3.2 Models

Run sequentially on the same four allocation-owned GPUs:

```text
base:
/home/mchen/FiRe-OPD/models/Qwen3-4B

adaptive_step50:
/home/mchen/FiRe-OPD/checkpoints/opd-adaptive-conciseprobe-cap50-tokenneutral-rawprompt-4gpu-tp4-step50/global_step_50_hf
```

Models are read-only.

## 4. Paired generation protocol

For every selected question and model, generate exactly one response under each prompt style.

### 4.1 Normal response

- Preserve the dataset prompt exactly, including its existing normal reasoning instruction.
- Apply the model tokenizer chat template with `add_generation_prompt=True` and `enable_thinking=False`.
- Use temperature `1.0`, top-p `1.0`, and `max_tokens=16,384`.

### 4.2 Concise response

- Strip the existing FiRe-OPD verbose instruction with the same helper used by adaptive training.
- Append exactly:

```text
Solve concisely. Avoid unnecessary explanation. Put your final answer within \boxed{}.
```

- Apply the same chat-template settings.
- Set the per-question cap to:

\[
C_i=\max(1,\lfloor 0.5L_{n,i}\rfloor),
\]

where `L_n` is the valid generated-token count of that question's normal response.
- Use temperature `1.0` and top-p `1.0`.

### 4.3 Reproducible sampling

Assign each selected source row a stable request seed derived from global seed 42 and its source-row index. Use the same row seed across normal, concise-capped, and concise-relaxed calls. Persist every seed.

The probe diagnoses paired stochastic events under one reproducible sample. It does not estimate per-question success probabilities.

## 5. Routing quadrants

Score responses using the repository's training-time math reward implementation and the parquet ground truth. Assign one of four event-level quadrants:

| Normal | Concise | Label |
|---|---|---|
| correct | correct | `compression_safe` |
| correct | incorrect | `compression_sensitive` |
| incorrect | correct | `concise_rescued` |
| incorrect | incorrect | `both_wrong` |

The previous adaptive run could observe only the first two quadrants because it never probed normal-wrong rows. This audit must generate concise responses for all 128 rows.

## 6. Compression-sensitive counterfactual

For each `compression_sensitive` row whose capped concise generation ended exactly at its per-row cap, rerun the same concise prompt with:

```text
max_tokens = normal_response_length
```

Use the identical request seed. The relaxed response must have the capped response token IDs as an exact prefix; otherwise validation fails.

Classify each `compression_sensitive` event as:

- `budget_limited_recovered`: capped concise hit its cap and relaxed concise becomes correct;
- `budget_limited_unrecovered`: capped concise hit its cap and relaxed concise remains incorrect;
- `prompt_or_sampling_failure`: capped concise stopped before its cap and was incorrect.

Also record whether each response has a parseable boxed answer. These mechanical classes support, but do not replace, qualitative inspection of the text.

## 7. Persisted artifacts

Use a new immutable run directory under:

```text
/home/mchen/FiRe-OPD/math_eval/paired_normal_concise_probe_outputs/<run-id>/
```

The run directory contains:

```text
manifest.json
sample.jsonl
base/records.jsonl
base/summary.json
base/compression_sensitive_cases.md
adaptive_step50/records.jsonl
adaptive_step50/summary.json
adaptive_step50/compression_sensitive_cases.md
comparison.json
comparison.md
```

Each response record stores:

- dataset/source identity and request seed;
- problem and ground truth;
- prompt messages and rendered prompt hash;
- response text, token IDs, token count, finish reason, extracted boxed answer, and correctness;
- concise cap and cap-hit status;
- quadrant;
- relaxed-concise result when applicable;
- mechanical failure class.

Do not store model weights or mutate model directories.

## 8. Analysis

For each model report:

- exact count and ratio of all four quadrants;
- normal and concise accuracy;
- paired accuracy delta;
- mean/median normal and concise lengths;
- paired concise-to-normal length ratio;
- concise cap-hit rate overall and by quadrant;
- counts of all three compression-sensitive failure classes;
- parse-failure counts;
- representative paired cases with full response text available in JSONL and compact excerpts in Markdown.

The cross-model comparison reports count changes from base to adaptive step 50. Since generation distributions differ across checkpoints even with the same seed, do not interpret row-level quadrant transitions as deterministic causal labels.

## 9. Runtime and GPU gate

Run in tmux session `opd-CLI` on allocation-owned CUDA tokens `0,1,2,3`, without nested `srun`. Before launch:

1. verify Slurm job 429 is `RUNNING`;
2. verify no process is using allocation-owned GPU tokens `0,1,2,3`;
3. verify both model directories and the dataset exist;
4. verify the adaptive merged model has two nonempty safetensor shards;
5. verify the output run directory does not exist.

Use tensor parallelism 4, max model length 40,960, and bounded active sequences. Run the base and adaptive model processes sequentially so only one vLLM engine occupies the four GPUs at a time.

The launcher writes a unique completion marker with the final exit code. A failed namespace is preserved and never resumed or overwritten.

## 10. Validation and failure handling

Fail closed if:

- the dataset hash differs;
- the sample is not exactly 128 distinct rows/identities;
- either model lacks exactly 128 normal and 128 capped-concise responses;
- a concise cap differs from `max(1, floor(0.5 * normal_length))`;
- any capped concise response exceeds its cap;
- paired source identities or seeds differ between models;
- a required relaxed response is missing;
- a relaxed response does not preserve the capped token prefix;
- quadrant counts do not sum to 128;
- summary counts do not recompute from `records.jsonl`;
- non-finite numeric metrics appear.

## 11. Testing

Use TDD for all CPU-testable behavior:

- deterministic distinct sampling;
- normal and concise prompt construction;
- cap derivation;
- training-reward-compatible correctness scoring;
- four-quadrant classification;
- relaxed-counterfactual classification and prefix validation;
- summary aggregation and invariant checks;
- launcher dry-run command and immutable namespace checks.

Before GPU launch run targeted pytest, Ruff, compileall, shell syntax, and `git diff --check`. After generation, run the independent artifact validator and recompute all reported metrics from JSONL.

## 12. Interpretation constraints

The final report may say that a paired event is consistent with insufficient scratch space only when a capped concise response hits its cap and the same-seed relaxed concise continuation becomes correct. It may say a response is consistent with prompt/trajectory failure when it stops naturally before the cap while remaining wrong.

The report must not infer intrinsic compressibility from one rollout, claim statistical significance from 128 `n=1` pairs, or conflate prompt effects with stochastic checkpoint differences.
