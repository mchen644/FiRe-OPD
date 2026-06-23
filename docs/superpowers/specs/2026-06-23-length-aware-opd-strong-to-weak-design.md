# Length-aware OPD for Strong-to-Weak Distillation

Date: 2026-06-23
Repository: `/home/mchen/FiRe-OPD`

## 1. Goal

Implement the first TokenSqueeze-inspired module in the original OPD strong-to-weak setting, before modifying FiRe-OPD. The goal is to train a Qwen3-4B student with a Qwen3-30B teacher so that the student generates shorter reasoning traces while preserving math benchmark performance.

This spec covers only the first incremental module: a length-aware penalty applied to the original OPD reverse-KL advantage. It does not include FiRe-OPD trajectory filtering, FiRe-OPD entropy-aware reweighting, adaptive multi-sample depth selection, DPO, SFT auxiliary loss, or intra-step rewrite/refinement.

## 2. Baseline Setting

Use the existing strong-to-weak original OPD setting:

- Student: `/home/mchen/FiRe-OPD/models/Qwen3-4B`
- Teacher: `/home/mchen/FiRe-OPD/models/Qwen3-30B-A3B-Instruct-2507`
- Train data: `/home/mchen/FiRe-OPD/data/g-opd/DeepMath-103K/train_filtered_level6.parquet`
- Validation data: AIME2024 and AIME2025 parquet files under `/home/mchen/FiRe-OPD/data/g-opd/`
- Existing baseline script: `verl/examples/fire_opd/run_opd_strong_to_weak_raw_teacher30b.sh`
- Original OPD flags: `only_reverse_kl_advantages=True`, `entropy_aware_distill=False`

The initial comparison target is original OPD strong-to-weak with raw prompts. Existing CoD/prompt-conversion experiments are out of scope for this module.

## 3. Method

Current original OPD computes token-level advantages from the teacher/student log-prob gap:

```text
advantage_t = -(old_log_prob_t - ref_log_prob_t)
            = ref_log_prob_t - old_log_prob_t
```

Add an optional gated sequence-level length penalty before the PPO policy loss:

```text
response_len_i = sum(response_mask_i)
reference_len = median(response_len over the actor micro-batch)
base_length_penalty_i = max(log(response_len_i / reference_len), 0)

seq_reward_i = sum(token_level_scores_i)
normalized_teacher_logprob_i = sum(ref_log_prob_i * response_mask_i) / response_len_i
teacher_reject_i = normalized_teacher_logprob_i <= bottom_percentile(normalized_teacher_logprob, teacher_reject_percentile)
incorrect_i = seq_reward_i <= correct_reward_threshold
penalty_gate_i = incorrect_i OR teacher_reject_i

length_penalty_i = base_length_penalty_i * penalty_gate_i
advantage'_i,t = advantage_i,t - λ * length_penalty_i
```

The penalty is broadcast across valid response tokens for each trajectory.

Design choices:

- Use `log_batch_median` as the first penalty type.
- Clamp at zero so responses shorter than or equal to the batch median are not rewarded directly; only long responses are candidates for penalty.
- Gate the penalty so long correct responses are protected unless the teacher gives the trajectory very low normalized likelihood.
- Apply only when `policy_loss.only_reverse_kl_advantages=True` and `policy_loss.length_aware_opd=True`.
- Keep the implementation detached from gradients because it depends only on response masks, rule-based reward, and teacher log-probabilities.
- Keep FiRe-OPD entropy-aware loss unchanged.

Initial settings:

```text
length_penalty_coef = 0.02
length_penalty_gate = "incorrect_or_low_teacher"
length_correct_reward_threshold = 0.5
length_teacher_reject_percentile = 20.0
```

Follow-up sweep if needed:

```text
0.01 if performance drops too much
0.05 if length reduction is too weak
```

## 4. Configuration Interface

Add fields to `PolicyLossConfig`:

```text
length_aware_opd: bool = False
length_penalty_coef: float = 0.0
length_penalty_type: str = "log_batch_median"
length_penalty_gate: str = "incorrect_or_low_teacher"
length_correct_reward_threshold: float = 0.5
length_teacher_reject_percentile: float = 20.0
```

Expected script flags for the first experiment:

```text
actor_rollout_ref.actor.policy_loss.length_aware_opd=True
actor_rollout_ref.actor.policy_loss.length_penalty_coef=0.02
actor_rollout_ref.actor.policy_loss.length_penalty_type=log_batch_median
actor_rollout_ref.actor.policy_loss.length_penalty_gate=incorrect_or_low_teacher
actor_rollout_ref.actor.policy_loss.length_correct_reward_threshold=0.5
actor_rollout_ref.actor.policy_loss.length_teacher_reject_percentile=20.0
```

When `length_aware_opd=False`, current OPD behavior must be unchanged.

## 5. Implementation Touch Points

### `verl/verl/workers/config/actor.py`

Add the length-aware OPD config fields to `PolicyLossConfig`.

### `verl/verl/workers/actor/dp_actor.py`

When length-aware OPD is enabled, retain `token_level_scores` or `token_level_rewards` in the actor input selection so the actor can identify correct vs incorrect trajectories.

In the non-entropy-aware OPD fallback path, after computing:

```python
advantages = -(old_log_prob - model_inputs["ref_log_prob"])
```

optionally apply the gated length-aware adjustment. Add actor metrics such as:

- `length_aware_opd/mean_response_len`
- `length_aware_opd/median_response_len`
- `length_aware_opd/mean_base_penalty`
- `length_aware_opd/mean_applied_penalty`
- `length_aware_opd/max_applied_penalty`
- `length_aware_opd/penalty_gate_ratio`
- `length_aware_opd/correct_skip_ratio`
- `length_aware_opd/teacher_reject_ratio`
- `length_aware_opd/coef`

### New run script

Create a new script based on the raw strong-to-weak OPD baseline:

```text
verl/examples/fire_opd/run_opd_strong_to_weak_lengthaware_teacher30b.sh
```

The script should default to:

```text
EXPERIMENT_NAME=opd-strong-to-weak-lengthaware-lambda${LENGTH_PENALTY_COEF}-rawprompt-${N_GPUS_PER_NODE}gpu-tp${ROLLOUT_TP_SIZE}-refmb4-rollmb4
LENGTH_PENALTY_COEF=0.02
```

It should retain the original OPD settings and not enable FiRe-OPD entropy-aware distillation.

## 6. Data Flow

1. Student generates on-policy responses with vLLM.
2. Trainer computes `response_mask` from generated responses.
3. Actor computes old student log-probs.
4. Teacher computes `ref_log_prob` on the same generated response.
5. Original OPD computes reverse-KL advantages.
6. If enabled, length-aware OPD computes each trajectory's valid response length from `response_mask`, computes a batch-median-relative base penalty, then applies it only when the trajectory is incorrect or in the teacher low-confidence set.
7. Existing PPO policy loss consumes the adjusted advantages.

## 7. Testing and Verification

Unit-level verification:

- Add a CPU test for the length-penalty helper or path.
- Verify no change when `length_aware_opd=False` or `length_penalty_coef=0`.
- Verify long incorrect responses get a positive applied penalty and short/batch-median responses get zero applied penalty.
- Verify long correct responses get zero applied penalty when teacher confidence is not in the low-confidence set.
- Verify long correct responses can still be penalized if `length_penalty_gate=incorrect_or_low_teacher` and teacher normalized log-probability is in the bottom configured percentile.
- Verify invalid `length_penalty_type` or `length_penalty_gate` raises a clear error.

Integration smoke test:

- Run a short Python/pytest target covering config import and length adjustment.
- Run the new shell script only far enough to validate config parsing if a full train is not being launched immediately.

Experiment verification:

- Train length-aware OPD strong-to-weak for the same step budget used by the original OPD baseline, preferably 100 steps for Table 2 alignment.
- Merge the checkpoint to HuggingFace format.
- Evaluate at minimum AIME24/AIME25 first, then all eight math datasets if the quick check looks promising.
- Compare `avg_length`, Avg@8 proxy, Avg@32, and pass@32 against original OPD strong-to-weak rawprompt.

## 8. Success Criteria

The first module is considered worth keeping if it achieves one of the following:

- At least 10% average response-length reduction with no more than about 2 absolute macro Avg@8 points degradation.
- Or a stronger length reduction with an acceptable performance tradeoff that motivates tuning `λ`.

If length does not decrease meaningfully at `λ=0.02`, try `λ=0.05`. If performance drops too much, try `λ=0.01`.

## 9. Risks

- Penalizing length at the sequence level may suppress useful long reasoning on hard problems. This is why the first penalty is median-relative, one-sided, and gated by correctness/teacher confidence rather than applied to every long response.
- Batch-median length can be noisy for small micro-batches. The initial implementation uses the actor micro-batch for simplicity because that is where advantages are adjusted; if unstable, later versions can compute the reference length at mini-batch level before splitting.
- Rule-based correctness is available for the math setting, but it may be imperfect due to answer parsing. The teacher low-confidence branch provides a second quality signal, and metrics should report how often each branch activates.

## 10. Future Modules After This Works

1. Adaptive reasoning-depth selection with `rollout.n > 1`.
2. Selected-short SFT auxiliary loss.
3. Offline intra-step linguistic refinement with KL preservation.
4. Optional FiRe-OPD integration after original OPD behavior is understood.
