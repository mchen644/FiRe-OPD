# Training-Time Rethinking OPD Probe Design

## Goal

Measure whether OPD length inflation and concise-teacher shortening are depth-specific **training-time** phenomena using Rethinking OPD-style metrics, rather than relying only on response length or truncation statistics.

The probe should answer three questions:

1. Is raw OPD length inflation driven by late-depth/tail-state degradation rather than early-prefix mismatch?
2. Does budget20 hardtrunc avoid the raw OPD late-depth failure mode without introducing an early-prefix collapse?
3. Does concise-teacher failure arise from early-prefix target/confidence distortion rather than late-tail degradation?

## Scope

Initial runs:

- `raw_opd`: vanilla OPD inflation baseline.
- `budget20`: rollout-length budget, 20% ESR hardtrunc, budget-aware teacher prompt; this is the target method.
- `concise20`: rollout-length budget, 20% ESR hardtrunc, concise teacher prompt; this is the shortening/collapse failure mode.

Out of scope for the initial implementation:

- `normal20` hardtrunc. It can be added after the main three-run story is clear.
- Extra student rollouts solely for probing.
- Fixed probe-prompt evaluation. The primary protocol should use the current training batch, matching the Rethinking OPD code path.
- Dense checkpoint saving.
- Using `rollout_corr/chi2_seq` as a primary claim. It may be kept as appendix context because it is a sequence-level variance amplifier and is not a Rethinking OPD metric.

## Literature and Code Alignment

### Rethinking OPD

Use Rethinking OPD-style distributional diagnostics:

- top-k overlap ratio
- overlap probability mass
- overlap-token advantage
- student entropy
- teacher entropy
- entropy gap
- position/depth-binned curves

The official Rethinking OPD paper lists code at:

```text
https://github.com/thunlp/OPD
```

The official implementation computes the diagnostics on the **current training batch of student-generated rollouts**, not on a separate fixed probe-prompt eval set. The relevant code paths are:

```text
verl/verl/workers/fsdp_workers.py
  _compute_teacher_top_k_log_probs(...)
  compute_rm_score(...)

verl/verl/trainer/ppo/ray_trainer.py
  Top-K Metrics Analysis
  overlap_ratio_chunk_* aggregation

verl/verl/workers/actor/dp_actor.py
  compute_distillation_reward(...)
```

Key implementation pattern from `thunlp/OPD`:

1. During actor log-prob computation, record current-student top-k ids/log-probs on the current response batch:

   ```text
   student_top_k_ids
   student_top_k_log_probs
   student_valid_counts
   ```

2. During teacher/ref scoring on the same batch, compute teacher top-k ids/log-probs and the overlap mask:

   ```text
   teacher_top_k_ids
   teacher_top_k_log_probs
   teacher_on_student_log_probs
   overlap_mask = student_top_k_id in teacher_top_k_ids
   teacher_in_student_mask
   ```

3. Aggregate global and position-chunk metrics from the same batch:

   ```text
   overlap_ratio = sum(overlap_mask * response_mask) / sum(response_mask expanded over k)
   overlap_ratio_chunk_{start}_{end}
   adv_intersection / overlap-token advantage metrics
   ```

Our implementation should follow this protocol as closely as possible. Do not vendor their code into FiRe-OPD unless explicitly approved; use it as a formula and architecture reference.

### Less is More: Early Stopping Rollout

arXiv:2605.27028 argues that late-position OPD teacher reward is ill-posed because the teacher is conditioned on an off-policy student-generated prefix. It calls this Off-policy Teacher Decay and proposes Early Stopping Rollout. This supports our hypothesis that tail supervision can become unreliable.

Important distinctions:

- Less is More emphasizes teacher recoverability/accuracy decay and position-based token selection.
- This probe emphasizes Rethinking OPD-style local distribution geometry across training depth.
- We can cite Less is More as motivation for late-position decay, while using Rethinking metrics to show where the degradation appears during our training runs.

## Probe Timing and Data Source

The primary probe uses the **existing just-generated training batch** at each logged training step.

It must not:

- generate additional student rollouts;
- use a separate fixed prompt eval set;
- require dense checkpoint saving.

At each training step:

1. Generate the normal on-policy rollout batch.
2. Compute student top-k ids/log-probs and student entropy on that same batch.
3. Compute teacher top-k ids/log-probs, teacher entropy, and overlap masks on that same batch.
4. Aggregate global and position-chunk metrics.
5. Log only aggregate metrics and lightweight context metadata.

For `budget20` and `concise20`, the probe must run **after rollout generation and before TALE hard truncation**, so it can observe the full original rollout tail while training still uses only the ESR prefix.

For `raw_opd`, the same conceptual hook point should score the full rollout batch before actor update.

## Batch and Position Aggregation Policy

Default paper-facing protocol:

```text
trajectory source = full current training batch
extra student rollouts = 0
probe frequency = every training step
top_k = 16
position_chunk_size = 1024 response tokens
selection policy = none; aggregate over the available batch tokens
```

This intentionally mirrors the Rethinking OPD code path, which aggregates from the current batch and reports both global and chunk-level metrics.

Position chunks:

```text
chunk_0_1024
chunk_1024_2048
chunk_2048_3072
...
```

For a chunk, only tokens with `response_mask=1` inside that chunk contribute. Chunks with no valid tokens are skipped or logged with `valid_token_count=0`.

Implementation escape hatch, not the primary protocol:

- If full-batch every-step probing is too expensive, allow config options to reduce frequency or subsample trajectories for debugging.
- Any reduced setting must be clearly labeled and should not be presented as the main Rethinking-aligned result.

## Metrics

For each response token position `t`, define:

```text
S_p(t) = TopK(student distribution at token t, k)
S_q(t) = TopK(teacher distribution at token t, k)
O(t) = S_p(t) ∩ S_q(t)
```

### Primary metric

```text
topk_overlap_ratio = mean_t |O(t)| / k
```

This is the main metric for the length-inflation story.

Log globally and by position chunk:

```text
rethinking_opd/topk_overlap_ratio
rethinking_opd/topk_overlap_ratio_chunk_0_1024
rethinking_opd/topk_overlap_ratio_chunk_1024_2048
...
```

Derived summaries for analysis:

```text
late_alignment_drop = overlap(early_chunk) - overlap(late_chunk)
tail_mismatch_ratio = (1 - overlap_late) / max(1 - overlap_early, eps)
```

### Secondary Rethinking-style metrics

Log globally and by chunk where feasible:

```text
student_overlap_mass = mean_t sum_{v in O(t)} p_t(v)
teacher_overlap_mass = mean_t sum_{v in O(t)} q_t(v)
student_entropy = mean_t H(p_t)
teacher_entropy = mean_t H(q_t)
entropy_gap = mean_t |H(q_t) - H(p_t)|
```

Overlap-token advantage should align with the Rethinking OPD definition and the official code's intersection-advantage diagnostics. Conceptually, on the overlap set with renormalized distributions:

```text
overlap_token_advantage = mean_t mean_{v in O(t)} pbar_t(v) * (log qbar_t(v) - log pbar_t(v))
```

Interpretation:

- Values closer to zero indicate better within-overlap agreement.
- More negative values indicate the student is overconfident or misweighted relative to the teacher inside shared support.

### Context-only metrics

Log these for alignment with Length Inflation and Less is More, but do not make them the primary mechanism claim:

```text
mean_response_length
max_response_length
truncation_or_clip_rate
valid_token_count_global
valid_token_count_by_chunk
actor_entropy_global
rollout_corr/chi2_seq, if already available
```

## Expected Evidence Patterns

### Raw OPD length inflation

Expected pattern:

- Early chunks remain relatively stable.
- Late chunks, especially 4K/8K/12K, show top-k overlap drop near the inflation transition.
- Entropy gap or overlap-token advantage may worsen in late chunks.
- The transition should align temporally with the known response-length jump around raw OPD step 15-18.

Claim supported if observed:

```text
Raw OPD inflation is not explained by early-prefix mismatch; it is associated with late-depth distributional degradation on student-induced prefixes.
```

### Budget20 hardtrunc

Expected pattern:

- Early-chunk geometry remains normal-like, not concise-collapse-like.
- Late chunks show smaller or delayed degradation than raw OPD, if enough long rollouts exist.
- Rollout length grows more smoothly than raw OPD.

Claim supported if observed:

```text
Budget20 avoids raw OPD's late-depth instability while preserving early-prefix alignment geometry.
```

This claim should remain conditional on sufficient late-chunk valid token counts.

### Concise20 collapse

Expected pattern:

- Early-chunk metrics change quickly: lower overlap, more negative overlap-token advantage, lower student entropy, or larger confidence mismatch.
- The effect appears before or during the response-length collapse around steps 15-20.
- The failure is not primarily a late-tail effect.

Claim supported if observed:

```text
Concise-teacher collapse is an early-prefix target/confidence distortion rather than the late-depth tail failure seen in raw OPD.
```

## Logging Format

Log scalar metrics to the normal training logger and write a CSV sidecar for analysis.

One CSV row per `(run_name, step, chunk_start, chunk_end)`:

```text
run_name
step
chunk_start
chunk_end
top_k
batch_size
valid_sequence_count
valid_token_count
mean_all_batch_response_length
max_all_batch_response_length
clip_rate_all_batch
topk_overlap_ratio
student_overlap_mass
teacher_overlap_mass
student_entropy
teacher_entropy
entropy_gap
overlap_token_advantage
```

Also write one global row per step with:

```text
chunk_start = -1
chunk_end = -1
```

Optional debug artifacts:

- response length histogram summary;
- selected prompt IDs only if already present in the batch metadata;
- no logits by default.

## Compute and Storage

Storage:

- CSV metrics only: negligible, MB-scale.
- Optional small metadata summaries.
- No dense checkpoints.

Compute:

- No additional generation cost.
- Student side: actor log-prob computation can be extended to return top-k ids/log-probs and entropy, following `thunlp/OPD`.
- Teacher side: requires teacher logits/top-k on the current batch before TALE hard truncation for budget20/concise20.
- Full-batch every-step probing may be expensive because late raw OPD batches can reach 8K-16K response tokens and the teacher is 30B.

Cost controls:

- Keep `top_k=16` by default.
- Use chunk aggregation without storing logits.
- Skip empty chunks and report valid counts.
- Provide debug-only frequency/subsampling config if overhead is intolerable, but label reduced runs clearly.

## Risks and Mitigations

### Risk: full-batch every-step probing slows training too much

Mitigation:

- Implement config gates for reduced frequency or trajectory subsampling as debugging fallbacks.
- Keep paper-facing runs on the current-batch protocol when feasible.
- Record probe timing so overhead is measurable.

### Risk: late-chunk valid counts are low for budget20/concise20

Mitigation:

- Always report `valid_token_count` and `valid_sequence_count` per chunk.
- Avoid claims about late-depth superiority when valid counts are too small.

### Risk: post-truncation probe accidentally hides tails

Mitigation:

- Put the teacher/student top-k probe before `truncate_to_tale_budget_esr`.
- Add a regression test or assertion that budget20/concise20 probe length context matches the original rollout length before hard truncation.

### Risk: metrics diverge from Rethinking OPD formulas

Mitigation:

- Cross-check formulas against `thunlp/OPD` code paths listed above.
- Keep naming and aggregation close to official diagnostics.
- Reuse existing offline audit functions where possible, but adapt them to current-batch training tensors.

## Testing Plan

Unit tests:

- top-k overlap and overlap mass on toy logits.
- overlap-token advantage / intersection advantage on toy distributions.
- entropy and entropy gap on toy logits.
- chunk aggregation uses `response_mask` correctly.
- current-batch probe does not request new rollouts.
- TALE hardtrunc runs call the probe before physical truncation.

Smoke test:

- Run a tiny training/probe job with a small local model or reduced batch.
- Verify scalar logger and CSV rows are written.
- Verify empty chunks are skipped or logged with zero valid counts.
- Verify budget20/concise20 rows reflect original rollout length context, not only ESR supervised width.

Analysis test:

- Reproduce current offline n=16 AIME24 depth metrics through the shared metric implementation where practical.
- Compare metric signs/ranges against `thunlp/OPD` expectations: overlap ratio in `[0,1]`, overlap mass near high-probability support, entropy gap nonnegative, overlap-token advantage typically nonpositive and closer to zero when alignment improves.

## Deliverables

1. Training-time Rethinking probe config keys.
2. Current-batch probe implementation aligned with `thunlp/OPD` top-k tensor flow.
3. Scalar logger and CSV sidecar for per-step global/chunk aggregates.
4. Run scripts for:
   - raw OPD probe
   - budget20 hardtrunc probe
   - concise20 hardtrunc probe
5. Analysis script/notebook to produce:
   - top-k mismatch heatmaps by step/depth
   - entropy heatmaps by step/depth
   - late alignment drop curves
   - concise early-collapse curves
