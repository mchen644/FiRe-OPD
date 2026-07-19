# Paired Normal/Concise Forced-Prefix Counterfactual Remediation Design

## 1. Problem and evidence

The paired probe originally assumed that two independent vLLM requests with the same prompt and per-request seed, differing only in `max_tokens`, would produce identical token prefixes. The formal retry `20260719T155822Z-92e670dbb302` disproved that assumption and stopped fail-closed at:

```text
ValueError: relaxed concise response does not preserve the capped token prefix
```

A follow-up TP=4 diagnostic on the same persisted 128-row sample completed 21 eligible base-model relaxed requests. Only 2/21 preserved the capped prefix; 19/21 diverged, with some first divergences at token 5. vLLM does instantiate per-request seeded generators, but separate dynamically batched TP=4 requests are not guaranteed to produce token-identical trajectories. Therefore, a separately regenerated relaxed answer cannot be classified causally as recovery from the observed capped trajectory.

The CUDA/fork issue from the earlier failed namespace remains independently fixed by `VLLM_WORKER_MULTIPROC_METHOD=spawn` at source commit `92e670dbb302163fdc16f7715c58a162230a2184`.

## 2. Chosen remediation

Keep normal and capped-concise generation unchanged. For each capped `compression_sensitive` event, construct a continuation request from:

```text
exact vLLM concise prompt token IDs + exact observed capped concise token IDs
```

Generate only the remaining response budget:

\[
R_i = L_{n,i} - L_{c,i}.
\]

Use the same persisted row seed, temperature 1.0, and top-p 1.0 for the continuation sampler. The sampler is newly initialized for the continuation; this is a reproducible conditional continuation from the observed prefix, not a claim that hidden RNG state from the first request was resumed.

Combine the fixed prefix and continuation token IDs, decode and score the complete response, and persist it under the existing `relaxed_concise` field. Exact prefix preservation is now guaranteed structurally and remains independently validated.

### Alternatives rejected

1. **Accept an independent relaxed resample.** This confounds additional budget with a different sampled trajectory and cannot support `budget_limited_recovered`.
2. **Generate the full concise trajectory first and derive the capped output by truncation.** This couples the counterfactual cleanly but changes the frozen production-like capped generation for all 128 rows.
3. **Force identical batch shapes or `max_num_seqs=1`.** This is expensive and still does not establish a supported bitwise-determinism guarantee across separate requests.

## 3. Frozen behavior

The remediation does not change:

- the persisted dataset sample, sample size 128, or seed 42;
- either read-only model checkpoint;
- normal generation (`max_tokens=16,384`);
- capped concise generation (`max(1, floor(0.5 * normal_length))`);
- scoring, four correctness quadrants, or selection of counterfactual rows;
- sequential TP=4 execution in `opd-CLI` on CUDA tokens `0,1,2,3`;
- immutable namespaces and fail-closed behavior.

No training, checkpoint mutation, W&B run, or canonical benchmark evaluation is introduced.

## 4. Persisted continuation provenance

Every non-null `relaxed_concise` response must contain:

- `counterfactual_mode="forced_prefix_continuation"`;
- `forced_prefix_length`, equal to capped concise length;
- `continuation_seed`, equal to the row request seed;
- `continuation_max_tokens`, equal to normal length minus capped length;
- `continuation_token_ids` and `continuation_length`;
- `continuation_prompt_token_ids` and `continuation_prompt_length`, whose exact suffix is the capped response;
- combined `token_ids`, `length`, decoded text, correctness, parseability, finish reason, and cap-hit status.

The combined token IDs must equal:

```text
capped_concise.token_ids + continuation_token_ids
```

and `relaxed_concise.max_tokens` must equal normal response length. If the remaining budget is zero, no vLLM call is made; the empty continuation is persisted and the capped answer is classified as unrecovered.

The run manifest and each model's `generation.json` record `relaxed_counterfactual_mode="forced_prefix_continuation"`.

## 5. Validation

CPU tests and the independent artifact validator must reject:

- a continuation prompt that omits or changes any capped token;
- combined token IDs that do not preserve the exact capped prefix;
- incorrect continuation length, token IDs, seed, or remaining budget;
- missing forced-prefix provenance on a required relaxed response;
- forced-prefix provenance on an ineligible response;
- any existing cap, route, identity, count, summary, or artifact-hash violation.

A focused GPU profile must demonstrate at least one forced-prefix continuation before another formal two-model launch. The formal run uses a fresh namespace and never reuses either failed namespace.

## 6. Interpretation

`budget_limited_recovered` means that the model reached a correct answer when allowed to continue from the exact observed capped failure prefix up to the normal response budget. It is evidence consistent with insufficient remaining scratch space for that sampled prefix.

It does not prove intrinsic question compressibility, restore the original request's hidden RNG state, or estimate per-question success probability. `n=1` and the 128-row sample remain descriptive diagnostics only.
