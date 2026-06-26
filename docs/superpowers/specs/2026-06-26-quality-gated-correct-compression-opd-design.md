# Quality-gated Correct-compression Selection for OPD

Date: 2026-06-26
Repository: `/home/mchen/FiRe-OPD`

## Goal

Replace the first multi-candidate selection rule with a TokenSqueeze-style rule that explicitly compresses correct trajectories and avoids learning low-quality wrong trajectories.

The previous `shortest_correct_else_teacher` module improved the training-batch selected length relative to its candidate pool, but Table-2-style eval on the first three completed datasets showed longer average generations than both OPD baseline and length-aware OPD. The key observed failure mode was that wrong eval samples became longer, while correct sample length stayed similar or slightly shorter.

This design keeps the experiment in the original strong-to-weak OPD setting and stacks on the stable length-aware coefficient `length_penalty_coef=0.02`, but changes the semantics:

```text
correct trajectories: learn them and compress them
wrong but teacher-accepted trajectories: use them only as OPD correction/imitation signal, without length compression
wrong and teacher-rejected trajectories: do not learn them
```

## Current Behavior and Problem

The current selection rule is:

```text
if any candidate is correct:
    select the shortest correct candidate
else:
    select the candidate with highest normalized teacher log-probability
```

The current length-aware OPD gate is:

```text
penalty_gate = incorrect OR low_teacher
```

Therefore selected correct trajectories are usually not length-penalized unless the teacher assigns them very low normalized likelihood. With multi-candidate selection, the selected batch becomes more correct, which reduces the fraction of samples covered by the length penalty. At step 50, the selected run had approximately:

```text
candidate_len ≈ 5343
selected_len  ≈ 4633
penalty_gate_ratio ≈ 0.328
correct_skip_ratio ≈ 0.672
```

This means the module selects more successful trajectories but does not strongly teach the model to make successful trajectories shorter. In no-correct groups, it still forces one fallback trajectory into the update, even if both candidates are low quality; the chosen fallback may be long and wrong.

## Proposed Method

Generate multiple student candidates per prompt, initially `rollout.n=2`, and group candidates by `uid` after reward and teacher/ref log-prob computation.

For each prompt group:

1. **Correct case:** If any candidate is correct, select the shortest correct candidate.
2. **No-correct, teacher-accepted case:** If no candidate is correct but one or more candidates pass a teacher-quality threshold, select the shortest teacher-accepted candidate.
3. **No-correct, teacher-rejected case:** If no candidate is correct and all candidates are teacher-rejected, drop the group from the actor update.

Correctness and teacher quality are defined as:

```text
response_len_i = sum(response_mask_i)
seq_reward_i = sum(token_level_scores_i * response_mask_i)
correct_i = seq_reward_i > correct_reward_threshold
normalized_teacher_logprob_i = sum(ref_log_prob_i * response_mask_i) / response_len_i
teacher_accepted_i = normalized_teacher_logprob_i > teacher_reject_threshold
```

The teacher reject threshold should be batch-relative, matching the FiRe-OPD insight:

```text
teacher_reject_threshold = quantile(normalized_teacher_logprob, teacher_reject_percentile / 100)
```

Initial setting:

```text
teacher_reject_percentile = 20.0
correct_reward_threshold = 0.5
keep_per_uid = 1 for selected groups
```

## Length Penalty Semantics

Change the length-compression gate for this module from the current conservative gate:

```text
incorrect OR low_teacher
```

to a correct-compression gate:

```text
selected_correct
```

The base penalty remains the previously working length-aware penalty:

```text
reference_len = median(response_len over selected actor mini-batch)
base_length_penalty_i = max(log(response_len_i / reference_len), 0)
length_penalty_i = base_length_penalty_i * selected_correct_i
advantage'_i,t = advantage_i,t - λ * length_penalty_i
```

Initial coefficient remains:

```text
λ = 0.02
```

Rationale:

- Correct trajectories are the right place to apply compression pressure: the model should first solve the problem, then learn to solve it more concisely.
- Wrong trajectories should not be encouraged to become short. If a trajectory is wrong but teacher-accepted, it can still provide teacher imitation / correction signal through OPD, but it should not receive length compression pressure.
- Wrong and teacher-rejected trajectories should not influence the update, following FiRe-OPD's trajectory filtering principle.

This is intentionally not a weaker `mild` penalty. It reuses the working `0.02` coefficient because the penalty is still median-relative and one-sided:

```text
0.02 * max(log(response_len / median_len), 0)
```

If this hurts correctness substantially, a later sweep can reduce the coefficient, but the first diagnostic run should test the clean hypothesis.

## Relationship to FiRe-OPD

FiRe-OPD contributes one key insight:

```text
not every student-generated trajectory should produce a gradient
```

FiRe-OPD filters bottom-percentile trajectories by normalized teacher log-probability and zeroes their contribution to the policy loss. This design borrows that idea for no-correct groups: when the model cannot produce a correct candidate and the teacher also assigns low normalized likelihood to all candidates, the group is excluded from learning.

Difference from FiRe-OPD:

- FiRe-OPD filters by teacher confidence and reweights tokens by entropy.
- This module filters only the no-correct fallback path and uses correctness to decide when length compression is active.
- It stays in original OPD strong-to-weak mode, without enabling entropy-aware distillation.

## Configuration Interface

Extend `algorithm.candidate_selection` with a new method:

```text
algorithm.candidate_selection.enabled: bool = False
algorithm.candidate_selection.method: str = "quality_gated_correct_compression"
algorithm.candidate_selection.correct_reward_threshold: float = 0.5
algorithm.candidate_selection.keep_per_uid: int = 1
algorithm.candidate_selection.teacher_reject_percentile: float = 20.0
algorithm.candidate_selection.drop_rejected_no_correct: bool = True
```

Add a policy-loss gate option for the actor:

```text
actor_rollout_ref.actor.policy_loss.length_penalty_gate=correct
```

The existing gate options remain unchanged for previous experiments:

```text
incorrect
low_teacher
incorrect_or_low_teacher
```

The new `correct` gate means:

```text
penalize selected correct trajectories only
```

For this module, the run script should set:

```text
actor_rollout_ref.actor.policy_loss.length_aware_opd=True
actor_rollout_ref.actor.policy_loss.length_penalty_coef=0.02
actor_rollout_ref.actor.policy_loss.length_penalty_gate=correct
algorithm.candidate_selection.enabled=True
algorithm.candidate_selection.method=quality_gated_correct_compression
algorithm.candidate_selection.teacher_reject_percentile=20.0
actor_rollout_ref.rollout.n=2
```

## Integration Point

Run candidate selection in `RayPPOTrainer.fit` after these tensors exist:

- `uid`
- `response_mask`
- `token_level_scores`
- `old_log_probs`
- `ref_log_prob`

and before:

- rollout correction metrics / IS tensors
- advantage computation
- actor update

The selected batch may contain fewer rows than the original prompt batch if some no-correct groups are dropped. This mirrors FiRe-OPD's effective reduction in learning signal for filtered trajectories. The training script keeps the same nominal `ppo_mini_batch_size=1024`; if `drop_rejected_no_correct=True`, effective gradient magnitude naturally scales with the selected keep ratio.

If physical dropping causes distributed shape issues, the fallback implementation is to keep one row per dropped group but attach a per-sequence loss mask that zeroes the actor loss. The first implementation should prefer physical dropping because it avoids wasted actor forward/backward compute on no-learning samples.

## Metrics

Keep existing metrics:

- `candidate_selection/enabled`
- `candidate_selection/groups`
- `candidate_selection/candidates`
- `candidate_selection/keep_ratio`
- `candidate_selection/any_correct_ratio`
- `candidate_selection/selected_correct_ratio`
- `candidate_selection/candidate_response_len_mean`
- `candidate_selection/selected_response_len_mean`
- `candidate_selection/fallback_teacher_ratio`
- `candidate_selection/selected_teacher_logprob_mean`

Add metrics needed to diagnose this design:

- `candidate_selection/no_correct_ratio`
- `candidate_selection/teacher_accept_ratio`
- `candidate_selection/no_correct_teacher_accept_ratio`
- `candidate_selection/dropped_uid_ratio`
- `candidate_selection/selected_correct_len_mean`
- `candidate_selection/selected_wrong_len_mean`
- `candidate_selection/correct_candidate_len_mean`
- `candidate_selection/wrong_candidate_len_mean`
- `candidate_selection/teacher_reject_threshold`

Length-aware metrics should remain available and should now show the correct-compression gate:

- `length_aware_opd/penalty_gate_ratio`
- `length_aware_opd/correct_skip_ratio`
- `length_aware_opd/mean_applied_penalty`
- `length_aware_opd/coef`

For the new `correct` gate, `correct_skip_ratio` should be interpreted carefully or renamed in a later cleanup; it will no longer mean "correct samples skipped by penalty".

## Expected Behavior

Compared with the failed `shortest_correct_else_teacher` run:

- selected correct trajectories should receive direct length pressure;
- no-correct wrong trajectories should no longer be selected purely by highest teacher log-probability if all candidates are teacher-rejected;
- wrong eval generations should be less likely to drift longer;
- if the model over-compresses and loses correctness, selected-correct ratio should fall and the training signal will naturally shift back toward longer correct trajectories when those are the only correct candidates.

Compared with length-aware OPD `λ=0.02`:

- average length should decrease more clearly if correct-compression works;
- accuracy/pass should remain within roughly 1-2 absolute points on macro Table-2 metrics;
- if accuracy drops more than that, the next sweep should reduce `λ` or make the correct gate apply only above a higher length percentile.

## Initial Experiment

Create a new run script derived from the current length-aware + selection script:

```text
verl/examples/fire_opd/run_opd_strong_to_weak_lengthaware_correctcompress_teacher30b.sh
```

Initial defaults:

```text
ROLLOUT_N=2
LENGTH_PENALTY_COEF=0.02
LENGTH_PENALTY_GATE=correct
CANDIDATE_SELECTION_ENABLED=True
CANDIDATE_SELECTION_METHOD=quality_gated_correct_compression
CANDIDATE_SELECTION_KEEP_PER_UID=1
CANDIDATE_SELECTION_CORRECT_REWARD_THRESHOLD=0.5
CANDIDATE_SELECTION_TEACHER_REJECT_PERCENTILE=20.0
CANDIDATE_SELECTION_DROP_REJECTED_NO_CORRECT=True
```

Experiment name:

```text
opd-strong-to-weak-lengthaware-correctcompress-lambda0.02-selectn2-rawprompt-4gpu-tp4-refmb4-rollmb4
```

Evaluate with the same FiRe-OPD Table-2 strong-to-weak setting used for OPD baseline and length-aware OPD.

## Risks

- Applying length pressure to correct trajectories can hurt accuracy if `λ=0.02` is too strong under the new gate.
- Dropping no-correct teacher-rejected groups reduces the effective batch size and may lower update magnitude. This is acceptable for the first diagnostic run and matches FiRe-OPD's filtering philosophy.
- With `rollout.n=2`, some prompts may have no correct candidate for many steps. Metrics must report dropped and fallback ratios so we know whether the method is starving the actor update.
- The existing `correct_skip_ratio` metric name becomes misleading under a `correct` gate; the implementation should add clearer metrics rather than relying on that name.

## Success Criteria

This module is worth keeping if, compared with length-aware OPD `λ=0.02` under the same Table-2 evaluation:

1. macro average response length decreases materially, and
2. macro Acc@32 or pass@32 does not drop by more than roughly 1-2 absolute points.

A stronger result would be simultaneous length reduction and pass@32 recovery/improvement relative to the failed `shortest_correct_else_teacher` run.
