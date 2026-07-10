# Group-Success Difficulty-Routed OPD Design

Date: 2026-07-10

## Context

The current difficulty-routed OPD experiment uses one rollout per question and combines verifier correctness with a batch-relative student-confidence rank. That confidence signal is not an identifiable estimate of question difficulty: it can reflect calibration error, response length, or confidence inflation rather than the probability that the current policy can solve the question.

This design replaces the confidence proxy with an online rollout-group success statistic. It is a deliberately clean Stage 1 experiment. The experiment changes only the difficulty estimate and the resulting teacher-prompt/supervision-horizon route.

The stopped predecessor run was:

```text
opd-budget20-easyconcise-noneasynormal-entropy003-esr20-step50-noprobe-compileoff-20260709_164816
```

The replacement run starts from the same base student checkpoint rather than resuming that run.

## Objective

Test whether a question-level success estimate from four current-policy rollouts gives a safer compression route than single-rollout correctness plus confidence:

```text
reliably solved question -> concise teacher + 20% prefix supervision
otherwise                -> normal teacher  + 50% prefix supervision
```

The experiment keeps the number of generated and teacher-scored trajectories per step equal to the current `1024 x n=1` setup.

## Non-goals

Stage 1 does not implement or enable:

- student-confidence routing
- a learned offline difficulty predictor
- per-question historical accuracy or length state
- a length reward, shortest-rollout reward, or pairwise length preference
- GRPO/verifier advantages in the actor objective
- direct student-entropy maximization
- entropy-weighted trajectory routing
- EOPD forward KL
- MOPD peer-conditioned teacher contexts
- candidate selection or rejection sampling
- selecting only correct or short rollouts for training
- rollout resampling or CoDaPO-style curriculum sampling

## Fixed training volume

Each optimizer step uses:

```text
unique questions:                 256
student rollouts per question:      4
generated trajectories:          1024
teacher-scored trajectories:      1024
actor training trajectories:      1024
```

All four rollouts remain in the batch through verifier scoring, teacher scoring, advantage construction, and actor update. There is no candidate filtering.

`data.train_batch_size` is the number of unique prompts, so the production run sets:

```text
data.train_batch_size=256
actor_rollout_ref.rollout.n=4
actor_rollout_ref.actor.ppo_mini_batch_size=256
```

veRL validates `ppo_mini_batch_size` against the prompt-level `train_batch_size`, then multiplies the actor value by `rollout.n` inside the worker. The effective actor mini-batch therefore still contains all `256 * 4 = 1024` trajectories.

## Difficulty computation

The trainer already assigns one `uid` before repeating each prompt for rollout generation. The four repeated rows therefore share a `uid`, even if batch balancing later changes row order.

For question group `g` with four rollouts, compute:

```text
c_gi = 1[sequence_reward_gi > correct_reward_threshold]
k_g  = sum_i c_gi
a_g  = k_g / 4
```

The route is:

```text
k_g = 4      -> easy / reliably solved
k_g in 1..3  -> learnable
k_g = 0      -> unresolved
```

The `learnable` and `unresolved` buckets use the same Stage 1 optimization route, but remain separate metric buckets. This preserves the distinction needed for a later EOPD experiment without introducing a second intervention now.

The group label is broadcast to all four rows. All rows with the same `uid` must receive identical teacher-prompt styles and ESR fractions.

## Teacher prompt and supervision route

All student rollouts use the existing normal/raw student prompt. Only the teacher/ref prompt is routed.

| Group | Teacher prompt | ESR fraction |
|---|---|---:|
| `k_g = 4` | concise | `0.20` |
| `k_g in 1..3` | normal | `0.50` |
| `k_g = 0` | normal | `0.50` |

The ESR token count is computed independently for every trajectory from its own valid response length:

```text
esr_tokens_gi = round(esr_beta_g * response_length_gi)
```

The existing hard-truncation path retains only the selected prefix for teacher scoring and actor training. The four trajectories in one group share `esr_beta_g`, not an absolute token count.

The teacher scores each student trajectory independently. The teacher input contains the routed teacher prompt and that trajectory's own prefix. It does not contain any peer rollout from the same group.

## Optimization objective

The actor uses the existing reverse-KL OPD objective only:

```text
actor.policy_loss.only_reverse_kl_advantages=True
actor.policy_loss.length_aware_opd=False
actor.entropy_coeff=0
actor.kl_loss_coef=0
algorithm.use_kl_in_reward=False
```

Verifier rewards are required to compute `k_g`, but verifier/GRPO advantages do not directly contribute to the actor loss. The new method must not add `difficulty_aware_entropy_weight` to the batch.

## Configuration interface

Add a new method without changing the behavior of the existing single-rollout method:

```yaml
algorithm:
  difficulty_aware_opd:
    enabled: true
    method: group_success_prompt_esr
    correct_reward_threshold: 0.5
    expected_group_size: 4
    easy_group_correct_count: 4
    easy_prompt_style: concise
    default_prompt_style: normal
    easy_esr_beta: 0.20
    non_easy_esr_beta: 0.50
    hard_entropy_coef: 0.0
```

The legacy `two_signal_prompt_esr_entropy` method and its existing configuration fields remain available for old runs and tests.

The production launcher also sets:

```text
algorithm.tale_budget.enabled=True
algorithm.tale_budget.source=rollout_length
algorithm.tale_budget.rollout_length_alpha=1.0
algorithm.tale_budget.truncate_to_esr=True
algorithm.tale_budget.teacher_prompt_style=normal
algorithm.rethinking_opd_probe.enabled=False
algorithm.candidate_selection.enabled=False
actor_rollout_ref.actor.use_torch_compile=False
actor_rollout_ref.ref.use_torch_compile=False
trainer.total_training_steps=50
```

## Components and data flow

### Pure routing helper

Add a group-success helper in `verl/verl/trainer/ppo/difficulty_aware_opd.py`. It accepts sequence rewards, response masks, `uid` values, and the routing config. It returns per-row tensors/arrays plus group-level metrics. It must not depend on old student log probabilities.

### Trainer dispatch

`_apply_difficulty_aware_opd_routing` dispatches on `difficulty_aware_opd.method`:

- `two_signal_prompt_esr_entropy`: existing behavior
- `group_success_prompt_esr`: new group-success behavior

For the new method, the trainer writes:

```text
difficulty_aware_correct
difficulty_aware_group_correct_count
difficulty_aware_group_accuracy
difficulty_aware_easy
difficulty_aware_learnable
difficulty_aware_unresolved
difficulty_aware_esr_beta
difficulty_aware_prompt_style
```

It does not write `difficulty_aware_confidence_rank` or `difficulty_aware_entropy_weight`.

### TALE/ESR reuse

The existing TALE rollout-length helper consumes the per-row `difficulty_aware_esr_beta` tensor. No new ESR implementation is needed. Existing per-row teacher-prompt routing consumes `difficulty_aware_prompt_style`.

### Production launcher

Add a dedicated launcher/wrapper for this experiment rather than overloading the old confidence-based command. The launcher exposes at least `ROLLOUT_N`, prompt batch size, total steps, run name, and log path, while validating the fixed relationship:

```text
ROLLOUT_N == expected_group_size == 4
PROMPT_BATCH_SIZE * ROLLOUT_N == TOTAL_TRAJECTORIES == 1024
PPO_MINI_BATCH_SIZE == PROMPT_BATCH_SIZE == 256
```

## Metrics

Log the following under `difficulty_aware_opd/`:

```text
enabled
group_count
expected_group_size
correct_rate
easy_group_ratio
learnable_group_ratio
unresolved_group_ratio
group_correct_count_0_ratio
group_correct_count_1_ratio
group_correct_count_2_ratio
group_correct_count_3_ratio
group_correct_count_4_ratio
concise_prompt_ratio
normal_prompt_ratio
esr_beta_mean
esr_beta_min
esr_beta_max
orig_response_length_mean
easy_orig_response_length_mean
learnable_orig_response_length_mean
unresolved_orig_response_length_mean
```

Required invariants in every logged step:

```text
group_count == 256
sum(group_correct_count_j_ratio for j in 0..4) == 1
easy_group_ratio + learnable_group_ratio + unresolved_group_ratio == 1
concise_prompt_ratio == easy_group_ratio
esr_beta values are exactly 0.20 or 0.50
```

Because every group contains four trajectories, group ratios and broadcast trajectory ratios are numerically equal.

## Validation and error handling

The new method fails fast when:

- `uid` is missing
- tensor and `uid` batch lengths differ
- any `uid` has a row count other than `expected_group_size`
- `expected_group_size != 4` for the production configuration
- `easy_group_correct_count` is outside `[1, expected_group_size]`
- ESR fractions are not in `(0, 1]`
- prompt styles are unsupported
- any row in one group would receive a route inconsistent with its peers

The implementation must not silently fall back to single-rollout confidence routing for malformed groups.

## Testing

### Unit tests

Add tests that verify:

1. Shuffled rows are grouped by `uid`, not adjacency.
2. `4/4` broadcasts concise prompt and ESR20 to all four rows.
3. `1/4`, `2/4`, and `3/4` broadcast normal prompt and ESR50.
4. `0/4` broadcasts normal prompt and ESR50.
5. Confidence/log-prob changes cannot change group-success routing.
6. No entropy-weight tensor is produced by the new method.
7. Missing `uid`, mixed group sizes, and invalid config values raise clear errors.
8. Existing single-rollout routing tests remain unchanged and pass.

### Trainer integration tests

Verify that trainer dispatch writes the documented batch fields and metrics, and that the existing TALE helper produces per-row masks with approximately 20% or 50% supervised fractions according to the route.

### Launcher checks

Before GPU launch, verify the resolved command contains:

```text
data.train_batch_size=256
actor_rollout_ref.rollout.n=4
actor_rollout_ref.actor.ppo_mini_batch_size=256
algorithm.difficulty_aware_opd.method=group_success_prompt_esr
algorithm.difficulty_aware_opd.easy_esr_beta=0.20
algorithm.difficulty_aware_opd.non_easy_esr_beta=0.50
algorithm.difficulty_aware_opd.hard_entropy_coef=0.0
actor_rollout_ref.actor.entropy_coeff=0
```

## Launch and acceptance criteria

The production run name is:

```text
opd-n4-easy4of4-concise20-noneasynormal50-purerkl-step50
```

Launch it in the existing `opd-CLI` session after unit/integration tests and a resolved-config check pass. Do not resume the stopped confidence-based run.

Monitor the first completed step and confirm:

- exactly 256 groups and 1024 trajectories are present
- all group/route invariants hold
- concise rows use ESR20 and non-easy rows use ESR50
- hard-entropy loss/weight is absent or zero
- actor update completes without shape or divisibility errors
- the log and checkpoint paths use the new run name

After the first-step checks pass, leave the 50-step run active. Model quality and length are evaluated separately after checkpoints are available; training-batch reward is not treated as a held-out quality result.
