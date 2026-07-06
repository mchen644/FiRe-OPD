# Single-Rollout Difficulty-Aware Budget20 OPD Design

## Context

We want to improve the existing OPD variant:

- student rollout prompt: normal prompt
- teacher/ref prompt: budget-aware prompt
- teacher budget source: current student rollout length
- ESR supervision: first 20% of the rollout (`esr_beta=0.2`)
- rollout count: `ROLLOUT_N=1`

The goal is to add difficulty awareness without extra student rollouts and without changing the hardtrunc semantics:

- `response_length/mean` remains the supervised/truncated training width.
- `tale_budget/response_length_mean` remains the original rollout length.
- ESR still supervises exactly the first 20% of each rollout.

This design is motivated by two papers:

1. CoDaPO / confidence-difficulty adaptive optimization (`2606.07950`):
   - confidence = `exp(mean token log probability)`
   - difficulty = group error rate, `1 - mean(reward)`
   - high-value questions are in the learnable band, not the easiest or hardest extremes.
2. CEEH / Compress Easy, Explore Hard (`2602.22642`):
   - dynamic difficulty = historical per-question accuracy EMA
   - hard questions receive extra entropy support; easy questions can be compressed more aggressively.

Because `ROLLOUT_N=1`, we cannot estimate true group difficulty in the CoDaPO sense. Instead, we estimate a single-rollout mastery proxy from signals already present in OPD training.

## Non-goals

This version will not:

- increase `ROLLOUT_N`
- add extra student rollouts
- add a fixed probe prompt evaluation set
- change the 20% ESR supervision width
- make the teacher budget adaptive in v1
- add CEEH-style entropy regularization in v1
- add shortest-correct historical length penalties in v1
- store logits by default

## Design summary

Add an optional per-sample difficulty-aware OPD weight. The trainer computes one scalar weight per generated response after reward and ref/student log-probs are available. The actor multiplies OPD token advantages by this scalar before the PPO loss.

Conceptually:

```text
budget teacher prompt unchanged
ESR mask unchanged
OPD reverse-KL advantages unchanged except for a per-sample scalar multiplier
```

The weight emphasizes samples that appear learnable under a single rollout and downweights samples that appear already mastered or currently hopeless.

## Single-rollout difficulty proxy

### Available signals

For each generated response, use only current-batch tensors already computed during training:

- `seq_reward`: full-response sequence reward, after hardtrunc-safe relocation into the supervised prefix
- `original_response_len`: the original rollout length from TALE rollout-length budgeting
- `student_logp_mean`: mean `old_log_probs` over the supervised prefix
- `teacher_logp_mean`: mean `ref_log_prob` over the supervised prefix
- `teacher_student_gap`: mean `ref_log_prob - old_log_probs` over the supervised prefix

Default v1 uses the supervised prefix for log-prob signals because budget20 trains only that prefix. Original rollout length is still used so the difficulty proxy can distinguish genuinely long/hard samples from short-mode collapse.

### Batch-rank normalization

Raw log-prob and length scales vary across runs and checkpoints. Convert continuous features to batch-relative ranks in `[0, 1]`:

```text
student_conf_rank  = percentile_rank(student_logp_mean)       # higher = more confident/easier
teacher_accept_rank = percentile_rank(teacher_logp_mean)      # higher = teacher accepts trajectory/prefix
length_easy_rank   = 1 - percentile_rank(log1p(original_response_len))
```

Ties should be handled deterministically and NaNs should be replaced by neutral rank `0.5`.

### Mastery score

Define a mastery proxy `m_hat` in `[0, 1]`:

```text
correct = 1[seq_reward > correct_reward_threshold]

m_hat = clamp(
    w_correct * correct
  + w_student_conf * student_conf_rank
  + w_teacher_accept * teacher_accept_rank
  + w_length_easy * length_easy_rank,
  0,
  1,
)
```

Default coefficients:

```text
w_correct = 0.40
w_student_conf = 0.20
w_teacher_accept = 0.25
w_length_easy = 0.15
correct_reward_threshold = 0.5
```

Interpretation:

- correct + high confidence + teacher-accepted + short => high mastery, likely easy/already learned
- wrong + low confidence + long + teacher-rejected => low mastery, likely too hard/noisy now
- mixed signals => mid mastery, likely learnable
- wrong + high confidence + short + low teacher acceptance => overconfident failure; should land near the middle rather than be discarded, because reverse-KL OPD can suppress bad sampled tokens when the teacher assigns lower probability than the student

### Learnable-band weight

Use CoDaPO's learnable-band idea on `m_hat`:

```text
band = 4 * m_hat * (1 - m_hat)
raw_weight = min_weight + (1 - min_weight) * band
```

Default:

```text
min_weight = 0.25
max_weight = 2.0
normalize_to_batch_mean = true
```

After computing `raw_weight`, normalize over valid samples to mean 1, then clamp:

```text
weight = clamp(raw_weight / mean(raw_weight), min_weight, max_weight)
```

This keeps total gradient scale stable while reallocating gradient mass within the batch.

## Training integration

### Trainer-side computation

Add a helper, tentatively:

```text
verl/verl/trainer/ppo/difficulty_aware_opd.py
```

with pure functions:

```python
compute_single_rollout_difficulty_weights(...)
rank_to_unit_interval(...)
summarize_difficulty_aware_weights(...)
```

The trainer calls the helper after:

- reward is available
- TALE budget metrics/masks are available
- `old_log_probs` are available
- `ref_log_prob` is available
- hardtrunc has happened, if enabled

For budget20 hardtrunc, `seq_reward` remains valid because `_truncate_response_tensor_to_batch` preserves the sequence reward by moving it into the last supervised token.

For original rollout length, the TALE rollout-length path should attach a per-sample tensor before truncation, e.g.:

```text
tale_budget_response_lengths: shape [batch]
```

This tensor is not response-aligned and therefore survives `truncate_to_tale_budget_esr` unchanged.

The trainer adds:

```text
difficulty_aware_weight: shape [batch]
difficulty_aware_mastery: shape [batch]
```

to `batch.batch` only when enabled.

### Actor-side application

In `DataParallelPPOActor.update_policy`, include `difficulty_aware_weight` in selected tensors when present.

When using normal OPD path:

```python
advantages = -(old_log_prob - ref_log_prob)
advantages = advantages * difficulty_aware_weight[:, None]
```

Apply the weight after optional length-aware penalty and before calling the policy loss function.

Do not alter `response_mask` or `tale_budget_esr_loss_mask`. This preserves the 20% ESR supervision semantics and avoids changing loss denominators in surprising ways.

### Compatibility

- Disabled path: no tensor is added, actor behavior is unchanged.
- Rethinking OPD probe: probe should continue to run before hardtrunc for probe-enabled budget20; difficulty weights are independent and logged separately.
- Candidate selection: v1 should either be disabled with candidate selection or applied after candidate-selection loss mask. Since hardtrunc is already incompatible with candidate selection, budget20 use is unaffected.
- Entropy-aware distillation path: v1 targets the vanilla OPD reverse-KL path only. If entropy-aware distillation is enabled, difficulty weights should be ignored unless explicitly extended later.

## Configuration

Add a new algorithm config section:

```yaml
algorithm:
  difficulty_aware_opd:
    enabled: false
    method: single_rollout_learnable_band
    correct_reward_threshold: 0.5
    min_weight: 0.25
    max_weight: 2.0
    normalize_to_batch_mean: true
    w_correct: 0.40
    w_student_conf: 0.20
    w_teacher_accept: 0.25
    w_length_easy: 0.15
    use_teacher_student_gap_metric: true
```

`teacher_student_gap` is logged in v1 but not included in the default mastery formula. It is useful for diagnosing overconfident wrong short responses and may become a formula term after inspection.

## Metrics

Log scalar metrics under `difficulty_aware_opd/`:

```text
enabled
weight_mean
weight_min
weight_max
weight_std
mastery_mean
mastery_min
mastery_max
student_conf_rank_mean
teacher_accept_rank_mean
length_easy_rank_mean
teacher_student_gap_mean
correct_rate
hard_bin_ratio          # m_hat < 0.25
learnable_bin_ratio     # 0.25 <= m_hat <= 0.75
easy_bin_ratio          # m_hat > 0.75
weight_hard_mean
weight_learnable_mean
weight_easy_mean
score_hard_mean
score_learnable_mean
score_easy_mean
orig_len_hard_mean
orig_len_learnable_mean
orig_len_easy_mean
```

These metrics are required before interpreting training results. In particular, the first smoke run should verify that learnable-bin samples receive larger average weights than easy/hard-bin samples.

## Experiment plan

Primary comparison, fixed at `ROLLOUT_N=1`:

1. Baseline budget20:
   ```text
   budget teacher prompt + rollout-length budget + supervise 20% + no difficulty weight
   ```
2. Difficulty-aware budget20:
   ```text
   same as baseline + single_rollout_learnable_band weight
   ```

Use the same settings as prior probe runs:

```text
trainer.val_before_train=False
trainer.test_freq=-1
trainer.save_freq=-1 for probe/smoke
```

For final comparison, save step50 checkpoints and evaluate on AIME24 first.

Optional later ablations:

- correctness-only mastery
- no length feature
- no teacher-accept feature
- include teacher-student gap as a signed term
- CEEH-style entropy bonus on low-mastery hard samples
- difficulty-aware teacher budget multiplier while keeping ESR at 20%

## Tests

Add unit tests for the pure helper:

1. rank normalization handles monotonic values, ties, NaNs, and single-element batches.
2. easy samples receive low weights:
   - correct, high student confidence, high teacher accept, short length
3. hopeless-hard samples receive low weights:
   - wrong, low confidence, low teacher accept, long length
4. learnable/mixed samples receive high weights:
   - correct but long/low confidence, or wrong but overconfident with teacher rejection
5. weights are finite, positive, normalized to mean approximately 1 when enabled.
6. disabled config produces no added tensors and no actor behavior change.

Add actor-side focused tests verifying that `difficulty_aware_weight` multiplies OPD advantages when present and leaves the loss unchanged when absent.

## Risks and mitigations

### Risk: single-rollout difficulty is noisy

Mitigation: use a soft weight, not hard filtering; keep `min_weight > 0`; normalize to mean 1.

### Risk: downweighting very hard samples prevents discovery

With `ROLLOUT_N=1`, a very hard wrong sample has no reliable positive learning signal. Downweighting prevents noisy overtraining. If this hurts, a later CEEH-style entropy bonus can support hard samples without imitating their wrong tokens.

### Risk: overconfident wrong samples get too much weight

This is intentional but should be monitored. OPD reverse-KL advantages can suppress sampled tokens when the teacher assigns lower probability than the student. Metrics should specifically track wrong/high-confidence/low-teacher-accept samples.

### Risk: feature scales drift over training

Batch-rank normalization avoids fixed-scale thresholds and keeps the proxy adaptive.

## Acceptance criteria

Implementation is acceptable if:

1. `algorithm.difficulty_aware_opd.enabled=False` preserves existing training/API behavior.
2. Budget20 hardtrunc semantics remain unchanged.
3. No extra student rollouts are introduced.
4. The helper emits finite weights for realistic batches.
5. Metrics show learnable-bin weights are higher than easy/hard-bin weights in smoke runs.
6. The first training probe can run with:
   ```text
   budget prompt + supervise20 + ROLLOUT_N=1 + difficulty_aware_opd.enabled=True
   ```
   without changing rethinking probe behavior.
