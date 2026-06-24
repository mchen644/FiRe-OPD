# Multi-candidate Short-correct Selection for OPD

Date: 2026-06-24
Repository: `/home/mchen/FiRe-OPD`

## Goal

Add the next TokenSqueeze-inspired module to the OPD compression stack: generate multiple student candidates for each prompt, then train OPD only on one selected candidate per prompt. The selected candidate should favor correctness and teacher acceptance first, then shorter reasoning length. This is intended to improve student-generated reasoning compression while preserving OPD's teacher-on-student-response data flow.

## Context

The first module, length-aware OPD, has already been implemented and tested with `length_penalty_coef=0.02` and `0.05`. Early results suggest `0.02` is more stable than `0.05`, but the direct length penalty alone gives only a small average length reduction.

This module should be stacked incrementally on top of original OPD, initially with the stable length-aware setting:

```text
actor_rollout_ref.actor.policy_loss.length_aware_opd=True
actor_rollout_ref.actor.policy_loss.length_penalty_coef=0.02
actor_rollout_ref.rollout.n=2
```

## Proposed Method

Use `rollout.n=2` to generate two student responses per prompt. The existing trainer already repeats each prompt by `rollout.n` and assigns the same `uid` to repeated candidates. After generation, reward computation, old student log-prob computation, and teacher/ref log-prob computation, select one candidate per `uid` before advantage computation and actor update.

Selection rule for each prompt group:

1. If at least one candidate is correct, select the shortest correct candidate.
2. If multiple correct candidates share the same response length, select the one with the highest normalized teacher log-probability.
3. If no candidate is correct, select the candidate with the highest normalized teacher log-probability.
4. Keep exactly one candidate per prompt for the initial implementation.

Definitions:

```text
response_len_i = sum(response_mask_i)
seq_reward_i = sum(token_level_scores_i * response_mask_i)
correct_i = seq_reward_i > correct_reward_threshold
normalized_teacher_logprob_i = sum(ref_log_prob_i * response_mask_i) / response_len_i
```

## Why This Fits OPD

OPD trains on student-generated responses scored by the teacher. This module does not introduce offline rewrites, new teacher prompts, or auxiliary SFT losses. It only changes which student-generated response is used for the OPD update. That makes it easier to attribute gains or failures.

## Configuration Interface

Add `CandidateSelectionConfig` under `algorithm`:

```text
algorithm.candidate_selection.enabled: bool = False
algorithm.candidate_selection.method: str = "shortest_correct_else_teacher"
algorithm.candidate_selection.correct_reward_threshold: float = 0.5
algorithm.candidate_selection.keep_per_uid: int = 1
```

Only `keep_per_uid=1` is supported initially. The method must be explicit so future selection methods can be added without changing the first experiment.

## Integration Point

In `RayPPOTrainer.fit`, run selection after these tensors exist:

- `uid` in `non_tensor_batch`
- `response_mask`
- `token_level_scores`
- `old_log_probs`
- `ref_log_prob`

and before these steps:

- rollout correction metrics / IS tensors
- `compute_advantage(...)`
- actor update

This keeps the actor update batch size equal to the original prompt batch size when `rollout.n=2` and `keep_per_uid=1`.

## Metrics

Log metrics with prefix `candidate_selection/`:

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

## Initial Experiment

Create a new run script derived from the length-aware OPD script:

```text
verl/examples/fire_opd/run_opd_strong_to_weak_lengthaware_select_teacher30b.sh
```

Initial defaults:

```text
ROLLOUT_N=2
LENGTH_PENALTY_COEF=0.02
CANDIDATE_SELECTION_ENABLED=True
CANDIDATE_SELECTION_METHOD=shortest_correct_else_teacher
CANDIDATE_SELECTION_KEEP_PER_UID=1
CANDIDATE_SELECTION_CORRECT_REWARD_THRESHOLD=0.5
```

Experiment name should include both modules:

```text
opd-strong-to-weak-lengthaware-lambda0.02-selectn2-rawprompt-4gpu-tp4-refmb4-rollmb4
```

## Testing Strategy

Use TDD:

1. Unit-test selection helper on small `DataProto` objects.
2. Unit-test config defaults and overrides.
3. Integration-test that selection reduces a repeated `uid` batch from `batch_size * n` to `batch_size` and preserves all tensor/non-tensor fields.
4. Shell-check the new training script.

## Risks

- Selecting only correct short candidates may reduce diversity. The first version keeps one candidate per prompt for simple attribution.
- `rollout.n=2` doubles generation and log-prob cost before selection. This is acceptable for the first controlled experiment.
- If most groups have no correct candidate, selection falls back to teacher preference and may not compress length much. Metrics should reveal this via `fallback_teacher_ratio` and selected length.

## Success Criteria

Compared with length-aware OPD `lambda=0.02`, the module is worth keeping if it improves either:

- response length reduction while holding macro performance within roughly 1-2 points, or
- performance/pass@k while not increasing average length materially.
