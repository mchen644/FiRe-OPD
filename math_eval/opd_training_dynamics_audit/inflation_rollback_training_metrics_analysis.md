# Training-Time Analysis: Why Raw OPD Length Inflates and Then Rolls Back

Source run:

- Run: `opd-raw-rethinking-probe`
- Log: `logs/rethinking_probe/raw_step50_retry_mem768_20260704_161722.log`
- Probe CSV: `math_eval/opd_training_dynamics_audit/training_probe_raw_opd.csv`
- W&B run: `gz84r9m8`

This note focuses on the transient length pattern in raw OPD: response length rises sharply around step 17--19, then falls back while training score stays similar or improves.

## Key caveat

The current training-time probe does **not** directly record EOS probability or EOS-vs-continuation advantage. Therefore, the causal explanation below is a mechanism hypothesis supported by proxies, not a direct proof.

The missing direct metrics are:

- student EOS probability by depth;
- teacher EOS probability by depth;
- `ref_log_prob - old_log_prob` by depth;
- positive-advantage fraction by depth;
- whether EOS is in student/teacher top-k;
- teacher-vs-student advantage on EOS compared with non-EOS continuation tokens.

## Observed phenomenon

Raw OPD length has a sharp transient spike:

| step | mean length | score | clip rate | reach 8k seq | reach 12k seq | reach 15k seq |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 3332 | 0.669 | 0.0049 | 70 | 25 | 7 |
| 17 | 6682 | 0.749 | 0.0908 | 316 | 161 | 105 |
| 18 | 7935 | 0.721 | 0.1436 | 425 | 252 | 171 |
| 19 | 8493 | 0.701 | 0.1895 | 480 | 298 | 209 |
| 20 | 8177 | 0.727 | 0.1592 | 448 | 273 | 184 |
| 25 | 5991 | 0.730 | 0.0684 | 270 | 137 | 81 |
| 30 | 4590 | 0.763 | 0.0273 | 164 | 74 | 38 |

The length increase is not a small uniform shift. It is a survival jump: many more samples cross deep continuation thresholds, especially 8k/12k/15k.

From step 16 to 17:

- mean length: `3332 -> 6682`
- reach 8k: `70 -> 316`
- reach 12k: `25 -> 161`
- reach 15k: `7 -> 105`
- clip rate: `0.0049 -> 0.0908`

Then from step 19 to 30:

- mean length: `8493 -> 4590`
- reach 8k: `480 -> 164`
- reach 12k: `298 -> 74`
- reach 15k: `209 -> 38`
- score: `0.701 -> 0.763`

Thus the rollback mainly removes long continuation mass without hurting the current-batch score.

## Metrics around the spike

Around the spike, optimization/probe metrics also show a distributional shift:

| step | length | score | top-k overlap | student entropy | teacher entropy | entropy gap | grad norm | chi2_seq |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 3332 | 0.669 | 0.6646 | 0.3011 | 0.3237 | 0.1506 | 1.56 | 4.66 |
| 17 | 6682 | 0.749 | 0.6605 | 0.3037 | 0.3518 | 0.1651 | 2.91 | 84.21 |
| 18 | 7935 | 0.721 | 0.6563 | 0.2866 | 0.3430 | 0.1649 | 3.11 | 22.51 |
| 19 | 8493 | 0.701 | 0.6545 | 0.2845 | 0.3398 | 0.1635 | 2.74 | 21.92 |
| 25 | 5991 | 0.730 | 0.6641 | 0.3061 | 0.3416 | 0.1547 | 1.21 | 2.69 |
| 30 | 4590 | 0.763 | 0.6662 | 0.3154 | 0.3369 | 0.1484 | 0.78 | 3.46 |

The spike coincides with:

- higher grad norm;
- higher teacher entropy;
- larger entropy gap;
- lower top-k overlap;
- much larger sequence-level correction variance proxy (`chi2_seq`).

After the rollback, these metrics partially recover while score does not degrade.

## Mechanism hypothesis

### 1. Length is controlled by EOS hazard

Response length is a survival process:

```text
P(length > t) = product_{i < t} (1 - P(EOS | prefix_i))
```

Small decreases in per-position EOS probability can compound over thousands of decoding steps. Therefore, an apparently modest change in local next-token probabilities can produce a large, sudden increase in observed response length.

This explains why the transition at step 17 is abrupt: the model likely enters a higher-continuation mode, where EOS hazard is lower across many intermediate positions.

### 2. Raw OPD can induce continuation bias

In this setup, the raw OPD actor update uses token-level reverse-KL-style advantages:

```text
A_t = ref_log_prob_t - old_student_log_prob_t
```

A sampled token receives positive pressure when the teacher assigns it higher probability than the old student. If, around step 16, teacher probabilities favor intermediate reasoning continuation tokens over the student's current distribution, OPD will increase those continuation tokens.

This can indirectly suppress EOS probability because probability mass is reallocated toward continuation tokens. Once EOS hazard drops, sequence survival increases nonlinearly, producing the step-17 length jump.

### 3. Rollback happens after the long tail becomes visited

The spike causes many more trajectories to enter 8k+/12k+/15k+ regions. These prefixes are further from the teacher's typical distribution. In those regions, teacher/student alignment worsens:

- lower overlap at deep positions;
- larger entropy gap;
- higher teacher entropy;
- unstable update proxies.

Once these long-tail continuation tokens are actually sampled and scored by the teacher, many of them likely receive weaker or negative relative advantage:

```text
ref_log_prob_t - old_student_log_prob_t <= 0
```

That update pressure reduces continuation probability in the tail and restores EOS/stop probability. The result is rollback: fewer sequences survive past 8k/12k/15k, but the answer-quality score is preserved because the removed tokens are mostly unnecessary continuation mass.

## What this explanation does and does not claim

This explanation does **not** claim that tail degradation is new; Rethinking OPD already shows late-depth entropy degradation. The distinct point here is about the transient dynamics:

> raw OPD length inflation appears as a continuation-survival spike, followed by a rollback when long-tail continuations are exposed to teacher scoring and become less supported.

The current evidence supports this through reach-depth, clip-rate, overlap, entropy, grad-norm, and chi2 proxies. It does not directly prove the EOS mechanism because EOS hazard is not logged yet.

## Direct metrics to add next

To directly test the mechanism, add a lightweight EOS/advantage probe during training:

1. `student_eos_prob_global` and by depth chunk.
2. `teacher_eos_prob_global` and by depth chunk.
3. `eos_advantage = teacher_logprob(EOS) - student_logprob(EOS)` by depth.
4. `sampled_token_advantage = ref_log_prob - old_log_prob` by depth.
5. Positive-advantage fraction by depth.
6. Continuation-vs-EOS advantage gap:

```text
mean_adv(sampled_non_eos_continuation) - adv(EOS)
```

7. Survival diagnostics:

```text
P(length > 4k), P(length > 8k), P(length > 12k), P(length > 15k)
```

These metrics would distinguish two possibilities:

- **EOS-hazard collapse:** EOS advantage becomes negative or EOS probability drops before length spikes.
- **Batch difficulty artifact:** prompt/score distribution changes without systematic EOS/continuation advantage shift.

Current prompt-length metrics do not support a simple prompt-length artifact: prompt length stays around 100 tokens through step 16--20.

## Bottom line

The most plausible reading of the raw training metrics is:

```text
step16 -> step19:
  teacher-induced continuation bias lowers EOS hazard,
  causing a nonlinear survival jump and length inflation.

step19 -> step30:
  newly exposed long-tail continuations receive weaker teacher support,
  continuation survival shrinks,
  length rolls back while score is maintained or improves.
```

To make this claim rigorous, we need direct EOS and token-advantage logging around the inflation window.
