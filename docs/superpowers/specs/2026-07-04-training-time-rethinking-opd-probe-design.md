# Training-Time Rethinking OPD Probe Design

## Goal

Measure whether OPD length inflation and concise-teacher shortening are depth-specific training-time phenomena using Rethinking OPD-style metrics, not only response length or truncation statistics.

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
- Dense checkpoint saving.
- Using `rollout_corr/chi2_seq` as a primary claim. It may be kept as appendix context because it is a sequence-level variance amplifier and is not a Rethinking OPD metric.

## Literature Alignment

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

The implementation should use that repository as a formula/reference check only. Do not vendor code into FiRe-OPD unless explicitly approved.

### Less is More: Early Stopping Rollout

arXiv:2605.27028 argues that late-position OPD teacher reward is ill-posed because the teacher is conditioned on an off-policy student-generated prefix. It calls this Off-policy Teacher Decay and proposes Early Stopping Rollout. This supports our hypothesis that tail supervision can become unreliable.

Important distinctions:

- Less is More emphasizes teacher recoverability/accuracy decay and position-based token selection.
- This probe emphasizes Rethinking OPD-style local distribution geometry across training depth.
- We can cite Less is More as motivation for late-position decay, while using Rethinking metrics to show where the degradation appears during our training runs.

## Probe Timing and Data Source

The probe must not generate additional student rollouts.

At each selected training step:

1. Use the existing just-generated on-policy rollout batch.
2. Select a small subset of trajectories from that batch.
3. Score selected response windows with the current student and teacher.
4. Log only aggregate metrics and optional lightweight metadata.

For `budget20` and `concise20`, the probe must run after rollout generation and before TALE hard truncation, so it can observe the full original rollout tail while training still uses only the ESR prefix.

For `raw_opd`, the same hook point should score the full rollout batch before actor update.

## Sampling Policy

`probe_examples_per_step` means existing trajectories selected from the current batch, not new rollouts.

Default lite probe:

```text
probe_examples_per_step = 4
selection = 2 longest trajectories + 2 median/random trajectories
window_len = 128
windows = 0, 4096, 8192 if available
top_k = 16
frequency = every training step, if compute budget allows
```

Sparse full probe:

```text
probe_examples_per_step = 8
selection = 4 longest trajectories + 4 median/random trajectories
window_len = 256
windows = 0, 1024, 2048, 4096, 8192, 12288 if available
top_k = 16
steps = 1, 5, 10, 12, 14, 15, 16, 17, 18, 19, 20, 25, 50
```

If every-step lite probing is too expensive, fall back to every 2-5 steps plus the sparse full probe near transition steps.

## Metrics

For each selected response window and each scored token position, compute student and teacher next-token distributions.

Let:

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

Derived summary:

```text
late_alignment_drop = overlap(early_window) - overlap(late_window)
tail_mismatch_ratio = (1 - overlap_late) / max(1 - overlap_early, eps)
```

### Secondary metrics

```text
student_overlap_mass = mean_t sum_{v in O(t)} p_t(v)
teacher_overlap_mass = mean_t sum_{v in O(t)} q_t(v)
student_entropy = mean_t H(p_t)
teacher_entropy = mean_t H(q_t)
entropy_gap = mean_t |H(q_t) - H(p_t)|
```

Overlap-token advantage follows Rethinking OPD's definition over the overlap set, using distributions renormalized on `O(t)`:

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
valid_window_count
actor_entropy_global
rollout_corr/chi2_seq, if already available
```

## Expected Evidence Patterns

### Raw OPD length inflation

Expected pattern:

- Early windows remain relatively stable.
- Late windows, especially 4K/8K/12K, show top-k overlap drop near the inflation transition.
- Entropy gap or overlap-token advantage may worsen in late windows.
- The transition should align temporally with the known response-length jump around raw OPD step 15-18.

Claim supported if observed:

```text
Raw OPD inflation is not explained by early-prefix mismatch; it is associated with late-depth distributional degradation on student-induced prefixes.
```

### Budget20 hardtrunc

Expected pattern:

- Early window geometry remains normal-like, not concise-collapse-like.
- Late windows show smaller or delayed degradation than raw OPD, if enough long rollouts exist.
- Rollout length grows more smoothly than raw OPD.

Claim supported if observed:

```text
Budget20 avoids raw OPD's late-depth instability while preserving early-prefix alignment geometry.
```

This claim should remain conditional on sufficient late-window valid counts.

### Concise20 collapse

Expected pattern:

- Early window metrics change quickly: lower overlap, more negative overlap-token advantage, lower student entropy, or larger confidence mismatch.
- The effect appears before or during the response-length collapse around steps 15-20.
- The failure is not primarily a late-tail effect.

Claim supported if observed:

```text
Concise-teacher collapse is an early-prefix target/confidence distortion rather than the late-depth tail failure seen in raw OPD.
```

## Logging Format

Write one row per `(run_name, step, probe_kind, window_start)`:

```text
run_name
step
probe_kind                 # lite or full
window_start
window_len
top_k
selected_count
valid_count
selection_policy
mean_selected_response_length
mean_all_batch_response_length
clip_rate_all_batch
topk_overlap_ratio
student_overlap_mass
teacher_overlap_mass
student_entropy
teacher_entropy
entropy_gap
overlap_token_advantage
late_alignment_drop        # optional derived row/column after aggregation
tail_mismatch_ratio        # optional derived row/column after aggregation
```

Optional debug artifacts:

- selected prompt IDs or dataset indices
- selected response lengths
- optional short text snippets for qualitative inspection

Do not store logits by default.

## Compute and Storage

Storage:

- CSV metrics only: negligible, MB-scale.
- Optional selected text snippets: small.
- No dense checkpoints.

Compute:

- No additional generation cost.
- Additional cost comes from student and teacher forward passes over selected response windows.
- Late windows require full prefix context up to `window_start + window_len`, so 8K/12K windows are the expensive part.

Cost controls:

- Keep every-step lite probe small.
- Run full probe only on selected steps.
- Skip windows with insufficient response length.
- Track `valid_count` so missing late windows are explicit.
- Allow disabling teacher scoring or reducing windows if training slowdown is too high.

## Risks and Mitigations

### Risk: every-step probing slows training too much

Mitigation:

- Start with sparse full probe only, then enable lite probe if overhead is acceptable.
- Reduce `probe_examples_per_step`, `window_len`, or late windows.

### Risk: late-window valid counts are low for budget20/concise20

Mitigation:

- Select longest trajectories first.
- Always report `valid_count`.
- Avoid claims about late-depth superiority when valid counts are too small.

### Risk: post-truncation probe accidentally hides tails

Mitigation:

- Put hook before `truncate_to_tale_budget_esr`.
- Add a regression test or assertion that probe sees `tale_budget/response_length_mean` full rollout lengths where applicable.

### Risk: metrics diverge from Rethinking OPD formulas

Mitigation:

- Cross-check formulas against `https://github.com/thunlp/OPD` once the repository is accessible.
- Keep our implementation small and tested.
- Reuse existing offline audit functions where possible.

## Testing Plan

Unit tests:

- top-k overlap and overlap mass on toy logits.
- overlap-token advantage on toy distributions.
- entropy and entropy gap on toy logits.
- window slicing preserves full prefix context while scoring only the requested response window.
- probe selection uses existing trajectories and does not request new rollouts.
- TALE hardtrunc runs call the probe before physical truncation.

Smoke test:

- Run a tiny training/probe job with a small local model or reduced batch.
- Verify CSV rows are written.
- Verify `valid_count` behaves correctly for windows longer than available responses.

Analysis test:

- Reproduce current offline n=16 AIME24 depth metrics through the shared metric implementation to ensure consistency.

## Deliverables

1. Training-time probe config keys.
2. Probe implementation reusing existing Rethinking-style metric functions where possible.
3. CSV logger for per-step/window aggregates.
4. Run scripts for:
   - raw OPD probe
   - budget20 hardtrunc probe
   - concise20 hardtrunc probe
5. Analysis script/notebook to produce:
   - top-k mismatch heatmaps by step/depth
   - entropy heatmaps by step/depth
   - late alignment drop curves
   - concise early-collapse curves
