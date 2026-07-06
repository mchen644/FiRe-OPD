# Single-Rollout Difficulty-Routed Budget20 OPD Design

## Context

We want to improve the existing OPD variant while keeping `ROLLOUT_N=1`:

- student rollout prompt: normal prompt
- default teacher/ref prompt: budget-aware prompt
- default teacher budget source: current student rollout length
- default ESR supervision: first 20% of the rollout (`esr_beta=0.2`)
- hardtrunc semantics: `response_length/mean` is the supervised/truncated training width, while `tale_budget/response_length_mean` is the original rollout length

The previous concise-teacher experiment showed that using a concise teacher prompt for every sample can collapse into a very short, low-entropy mode. The new goal is therefore:

```text
Compress easy samples more directly.
Preserve exploration on hard samples.
Do not add extra student rollouts.
Keep the default hard/uncertain path equivalent to budget20.
```

This design is motivated by two papers:

1. CoDaPO / confidence-difficulty adaptive optimization (`2606.07950`):
   - confidence = `exp(mean token log probability)`
   - difficulty = group error rate, `1 - mean(reward)`
   - value uses multiplication: `confidence * learnable_band(difficulty)`
   - CoDaPO does not add an explicit length penalty or entropy regularizer.
2. CEEH / Compress Easy, Explore Hard (`2602.22642`):
   - difficulty = low historical per-question accuracy EMA
   - hard questions receive stronger entropy regularization
   - easy/mastered questions receive compression pressure through a dynamic length penalty anchored to the shortest historical correct response

Because `ROLLOUT_N=1`, we cannot estimate true group difficulty. We use a two-signal single-rollout proxy instead.

## Non-goals

This version will not:

- increase `ROLLOUT_N`
- add extra student rollouts
- add a fixed probe-prompt evaluation set
- store logits by default
- maintain per-question historical accuracy or shortest-correct length in v1
- implement CoDaPO resampling or two-stage updates in v1
- implement CEEH entropy-based advantage in v1

## Paper formulas to reuse

### CoDaPO

For a question with `G` rollouts:

```text
d_q = 1 - (1/G) * sum_i r_i
c_q = exp((1/G) * sum_i mean_t log pi(o_i,t | q, o_i,<t))
v_q = c_q * (1 - 4(d_q - 0.5)^2)
```

`1 - 4(d - 0.5)^2` is high for mid-difficulty questions and low for very easy or very hard questions.

### CEEH

For each question:

```text
Acc_h(x) <- (1 - eta) Acc_h(x) + eta Acc(x)
eta = 0.2 if Acc(x) > Acc_h(x), else 0.05
hard if Acc_h(x) < global_average(Acc_h)
```

Maximum-entropy regularization:

```text
L_ent = -lambda(x,t) * H(pi_theta)
lambda(x,t) = 5 * lambda_0 * schedule(t), if x is hard
              1 * lambda_0 * schedule(t), otherwise
```

Entropy-based advantage, not used in v1:

```text
psi(H) = min(alpha * H, detach(|A|) / kappa)
A_shaped = A + psi(H)
```

CEEH length penalty, not used in v1:

```text
L_x = historically shortest correct response length for question x
R_len(y_i | x) = ((L_{x,y_i} - L_x) / L_x) * 1[y_i correct]
R_total = R - beta * R_len
```

We avoid this in v1 because the DeepMath training run is short and questions rarely repeat within step50.

## Two-signal single-rollout difficulty proxy

Use only two signals:

1. correctness
2. student confidence

For each generated response `i`:

```text
r_i = 1[sequence_reward_i > correct_reward_threshold]
logp_i = mean_t old_log_probs_i,t over the full valid rollout
c_i = percentile_rank_batch(logp_i) in [0, 1]
```

`c_i` is a rank, not a calibrated probability. Higher means the student was more confident than other samples in the current batch.

Define multiplicative gates:

```text
easy_i = r_i * c_i
hard_i = (1 - r_i) * (1 - c_i)
```

Interpretation:

```text
easy_i high: correct and high-confidence -> safe to compress more
hard_i high: wrong and low-confidence -> preserve exploration
correct but low-confidence: not easy yet, keep normal budget20
wrong but high-confidence: overconfident wrong, do not treat as hard-exploration; keep normal budget20 and monitor
```

This deliberately avoids a large multi-feature score.

## How compression is applied

Prompt-only token budgets are not reliable enough by themselves. v1 therefore uses two compression mechanisms for easy samples.

### 1. Prompt routing

Default path:

```text
teacher prompt = budget-aware prompt
```

Easy path:

```text
if easy_i >= easy_prompt_threshold:
    teacher prompt = concise prompt
else:
    teacher prompt = budget-aware prompt
```

Default:

```text
easy_prompt_threshold = 0.7
```

This avoids using concise prompt on all samples, which previously caused short-mode collapse.

### 2. Direct ESR compression

The direct compression signal is the supervision mask, not the numeric token budget prompt.

Let the base ESR fraction be the existing budget20 value:

```text
base_esr_beta = 0.20
```

For easy samples:

```text
esr_beta_i = max(min_easy_esr_beta, base_esr_beta - easy_esr_delta * easy_i)
```

Default:

```text
base_esr_beta = 0.20
min_easy_esr_beta = 0.10
easy_esr_delta = 0.10
```

So:

```text
easy_i = 0.0 -> supervise first 20%
easy_i = 1.0 -> supervise first 10%
```

Hard, uncertain, and overconfident-wrong samples keep the normal 20% ESR unless they also become easy by the formula, which cannot happen for wrong samples because `r_i=0`.

The original rollout length remains logged as `tale_budget/response_length_mean`. The truncated `response_length/mean` becomes the actual average supervised width after difficulty routing.

## How hard-sample exploration is preserved

Use CEEH-style maximum-entropy regularization, gated by `hard_i`.

For the actor update, compute student entropy on the tokens participating in the actor loss. Add:

```text
L_entropy = - hard_entropy_coef * mean_i,t [ hard_i * H_i,t ]
L_total = L_OPD + L_entropy
```

Because training minimizes loss, the negative entropy term encourages the student to keep higher entropy on hard samples.

Default:

```text
hard_entropy_coef = 1e-3
```

This is intentionally small. It should be swept after smoke verification, e.g. `0`, `1e-3`, `3e-3`.

v1 applies entropy regularization only on the supervised/truncated prefix. This keeps compute unchanged and respects hardtrunc, but it means the regularizer protects early reasoning exploration rather than the full rollout tail.

## Training integration

### Where gates are computed

The trainer computes difficulty gates after full-rollout reward and full-rollout student log-probs are available, and before teacher/ref prompt construction:

1. generate student rollout
2. compute full-response reward
3. compute `old_log_probs` on the full rollout
4. compute `r_i`, `c_i`, `easy_i`, `hard_i`
5. build per-sample teacher prompts:
   - easy -> concise
   - otherwise -> budget-aware
6. build per-sample ESR mask using `esr_beta_i`
7. compute teacher/ref log-probs
8. run rethinking probe before hardtrunc when enabled
9. hardtrunc to the max supervised prefix
10. update actor with hard-gated entropy regularization

If reward computation is asynchronous, the trainer must resolve the reward future before step 4 when difficulty routing is enabled.

### Data added to batch

When enabled, add sequence-level tensors:

```text
difficulty_aware_correct: [batch]
difficulty_aware_confidence_rank: [batch]
difficulty_aware_easy: [batch]
difficulty_aware_hard: [batch]
difficulty_aware_esr_beta: [batch]
difficulty_aware_entropy_weight: [batch]
tale_budget_response_lengths: [batch]  # original rollout lengths, survives hardtrunc
```

`difficulty_aware_hard` or `difficulty_aware_entropy_weight` must be selected into actor micro-batches so the entropy term can be applied during `update_policy`.

### TALE / teacher prompt integration

Extend the existing TALE rollout-length path to support per-sample ESR beta and prompt style.

For each row:

```text
budget_i = rollout_length_i * rollout_length_alpha
prompt_style_i = concise if easy_i >= easy_prompt_threshold else budget
esr_tokens_i = round(budget_i * esr_beta_i)
```

The default non-easy path is unchanged from budget20:

```text
prompt_style_i = budget
esr_beta_i = 0.20
```

## Configuration

Add a new algorithm config section:

```yaml
algorithm:
  difficulty_aware_opd:
    enabled: false
    method: two_signal_prompt_esr_entropy
    correct_reward_threshold: 0.5
    confidence_rank_scope: batch

    # easy compression
    easy_prompt_threshold: 0.7
    easy_prompt_style: concise
    default_prompt_style: budget
    base_esr_beta: null        # null means use algorithm.tale_budget.esr_beta
    min_easy_esr_beta: 0.10
    easy_esr_delta: 0.10

    # hard exploration
    hard_entropy_coef: 0.001
```

Disabled path must preserve existing behavior exactly.

## Metrics

Log scalar metrics under `difficulty_aware_opd/`:

```text
enabled
correct_rate
confidence_rank_mean
confidence_rank_correct_mean
confidence_rank_wrong_mean
easy_mean
easy_max
hard_mean
hard_max
concise_prompt_ratio
budget_prompt_ratio
esr_beta_mean
esr_beta_min
esr_beta_max
orig_response_length_mean
supervised_esr_tokens_mean
hard_entropy_weight_mean
hard_entropy_loss
wrong_high_conf_ratio       # wrong and c_i >= easy_prompt_threshold
correct_low_conf_ratio      # correct and c_i <= 1 - easy_prompt_threshold
```

Also keep existing TALE metrics:

```text
tale_budget/response_length_mean      # original rollout length
tale_budget/esr_tokens_mean           # supervised tokens after per-sample ESR beta
tale_budget/esr_supervised_fraction_mean
```

## Experiment plan

Primary comparison, fixed at `ROLLOUT_N=1`:

1. Baseline budget20:
   ```text
   budget teacher prompt + rollout-length budget + fixed 20% ESR + no entropy bonus
   ```
2. Prompt routing only:
   ```text
   easy -> concise teacher, otherwise budget teacher; fixed 20% ESR; no entropy bonus
   ```
3. Prompt routing + direct easy ESR compression:
   ```text
   easy -> concise teacher + ESR down to 10-20%; otherwise budget20
   ```
4. Full difficulty-routed version:
   ```text
   easy -> concise + shorter ESR
   hard -> budget20 + hard-gated entropy
   otherwise -> budget20
   ```

Use smoke/probe settings unless saving a final checkpoint:

```text
trainer.val_before_train=False
trainer.test_freq=-1
trainer.save_freq=-1 for smoke/probe
```

For final comparison, save step50 checkpoints and evaluate AIME24 first.

## Tests

Add unit tests for pure helpers:

1. confidence rank handles monotonic values, ties, NaNs, and single-sample batches.
2. correct/high-confidence samples produce high `easy`, low `hard`, concise prompt, and shorter ESR beta.
3. wrong/low-confidence samples produce high `hard`, low `easy`, budget prompt, base ESR beta, and nonzero entropy weight.
4. correct/low-confidence samples remain on budget prompt and near-base ESR.
5. wrong/high-confidence samples remain on budget prompt and near-base ESR, and are counted in `wrong_high_conf_ratio`.
6. per-sample ESR masks match `round(budget_i * esr_beta_i)` and never supervise padding.
7. disabled config produces identical prompts, masks, and actor loss to existing budget20.

Add actor-side focused tests verifying:

1. hard-gated entropy changes loss only when `difficulty_aware_entropy_weight` is present and `hard_entropy_coef > 0`.
2. existing OPD reverse-KL advantage behavior is unchanged when difficulty-aware OPD is disabled.

## Risks and mitigations

### Risk: concise prompt on easy samples still causes collapse

Mitigation: concise prompt is gated by both correctness and confidence; track `concise_prompt_ratio`; keep easy ESR compression bounded by `min_easy_esr_beta`.

### Risk: prompt routing is less direct than expected

Mitigation: direct compression is applied through the ESR loss mask. Prompt routing is an auxiliary teacher-distribution change, not the only length control.

### Risk: hard entropy increases length

Mitigation: entropy applies only to wrong/low-confidence samples and only on the supervised prefix; keep coefficient small and sweep `0`, `1e-3`, `3e-3`.

### Risk: rollout_n=1 difficulty is noisy

Mitigation: use soft multiplicative scores, not hard difficulty labels for ESR/entropy; only prompt routing uses a conservative threshold.

### Risk: async reward timing changes training flow

Mitigation: only resolve reward before TALE prompt construction when difficulty-aware routing is enabled. Disabled path remains unchanged.

## Acceptance criteria

Implementation is acceptable if:

1. `algorithm.difficulty_aware_opd.enabled=False` preserves existing training/API behavior.
2. No extra student rollouts are introduced.
3. Non-easy samples preserve the existing budget20 behavior: budget prompt and 20% ESR.
4. Easy samples can receive concise prompt and shorter ESR according to `easy_i = r_i * c_i`.
5. Hard samples receive a nonzero entropy bonus according to `hard_i = (1-r_i)(1-c_i)`.
6. Rethinking OPD probe behavior is unchanged and still runs before hardtrunc for probe-enabled budget runs.
7. Metrics expose concise routing ratio, ESR beta distribution, and hard entropy contribution.
