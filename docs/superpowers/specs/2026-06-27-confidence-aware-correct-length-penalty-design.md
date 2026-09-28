# Confidence-Aware Correct Length Penalty Design

## Goal

Replace the batch-median length reference with a teacher-confidence-weighted log-length penalty for OPD runs where we want to compress correct answers without penalizing wrong answers.

## Motivation

The current `log_batch_median` penalty compares all selected responses in a batch. This can unfairly penalize difficult problems whose correct solutions naturally require longer reasoning. A fixed target length has a similar issue: different problems may require different amounts of reasoning. Instead, use a continuous log-length penalty whose strength is scaled by teacher confidence.

## Training Semantics

Use with `ROLLOUT_N=1` by default.

For each selected trajectory:

- correct trajectory: learn it and apply a confidence-weighted length penalty.
- wrong + teacher accepted trajectory: learn it, but do not apply length penalty.
- wrong + teacher rejected trajectory: keep it in the fixed-size batch, but set actor loss mask to zero.

This preserves the current fixed-batch loss-mask design and the “do not learn teacher-rejected wrong trajectories” semantics.

## Penalty Formula

Add a new length penalty type:

```text
log_length_teacher_confidence
```

For response length `L`, max response length `M`, normalized teacher log probability `lp_teacher`, and teacher accept threshold `tau`:

```text
base_penalty = log1p(L) / log1p(M)
confidence_weight = sigmoid((lp_teacher - tau) / confidence_temperature)
penalty = length_penalty_coef * base_penalty * confidence_weight
```

The penalty is still gated by `length_penalty_gate=correct`, so wrong answers get zero length penalty.

`M` is only a normalization scale from `data.max_response_length` / response tensor length; it is not a fixed target and does not create a hard threshold.

## Defaults

New config field under `actor_rollout_ref.actor.policy_loss`:

```yaml
length_confidence_temperature: 0.1
```

Recommended experiment override:

```bash
ROLLOUT_N=1 \
LENGTH_PENALTY_TYPE=log_length_teacher_confidence \
LENGTH_PENALTY_COEF=0.05 \
HYDRA_FULL_ERROR=1 \
bash run_train_lengthaware_correctcompress_opd.sh
```

## Metrics

Add metrics:

- `length_aware_opd/confidence_temperature`
- `length_aware_opd/confidence_weight_mean`
- `length_aware_opd/confidence_weight_correct_mean`
- `length_aware_opd/confidence_weight_penalized_mean`

Existing metrics remain unchanged.

## Testing

Add unit tests for:

1. wrong answers receive zero penalty when gate is `correct`.
2. correct answers receive a positive penalty that increases with response length.
3. correct answers with higher teacher confidence receive a larger penalty than equal-length lower-confidence correct answers.
4. invalid confidence temperature raises a clear error.
