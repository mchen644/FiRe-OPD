# Endpoint-Preserving Tri-Prompt Full-Response OPD Design

Date: 2026-07-19

Status: conversationally approved; written review pending

## 1. Objective

Build a new adaptive OPD treatment that preserves every token of the primary normal-prompt student rollout while routing the 30B teacher among three prompt styles:

1. concise teacher prompt for event-level compression-safe rows;
2. rollout-length budget teacher prompt for compression-sensitive rows;
3. unchanged normal teacher prompt for normal-wrong rows.

The treatment is designed to remove the endpoint asymmetry in the completed adaptive-concise experiment. It must not remove successful answer/EOS suffixes, exchange diagnostic tokens for supervised tokens, use a student response as a self-training target, or modify the Vanilla treatment on normal-wrong rows.

## 2. Motivation and evidence

The completed adaptive-concise treatment routed the teacher correctly but trained only on a shortened prefix of the normal student rollout for every normal-correct row. It therefore removed successful answer/EOS suffixes while retaining full normal-wrong tails. Held-out AIME24+AIME25 evaluation showed lower accuracy and pass@32 and longer failure tails than matched Vanilla OPD.

A paired normal/concise training-distribution probe then showed that:

- only 37.0% of base-model normal-correct events remained correct under a half-length concise cap;
- many apparent concise failures were caused by the half-length cap;
- concise correctness alone is not sufficient to justify copying the concise reasoning trajectory;
- normal-wrong events are heterogeneous, but the selected treatment deliberately leaves them identical to Vanilla.

A historical all-row concise-teacher plus ESR20 checkpoint was subsequently evaluated on AIME24. It reached 20.94% accuracy with mean length 2,079 tokens, compared with 56.15% and 9,220 tokens for Vanilla OPD. This establishes that indiscriminate concise prompting combined with endpoint truncation causes severe over-compression. The new treatment differs by routing concise prompting only to double-correct events and supervising the complete normal response on every route.

## 3. Frozen comparison contract

Except for the additional concise diagnostic and route-specific 30B prompt, the treatment inherits the completed Vanilla OPD contract:

- student: `/home/mchen/FiRe-OPD/models/Qwen3-4B`;
- teacher/ref: `/home/mchen/FiRe-OPD/models/Qwen3-30B-A3B-Instruct-2507`;
- training data: `/home/mchen/FiRe-OPD/data/g-opd/DeepMath-103K/train_filtered_level6.parquet`;
- one normal/raw student rollout per question;
- 1,024 distinct questions per optimizer step;
- 1,024 primary actor rows per optimizer step;
- rollout `n=1`;
- 50 optimizer steps and 51,200 question slots;
- data seed 42 and the same shuffle/order contract as Vanilla;
- maximum normal response length 16,384;
- learning rate `1e-6` with zero warmup;
- reverse-KL-only advantages;
- token-mean actor loss;
- token-level rollout importance sampling with threshold 5.0;
- actor and rollout micro-batch settings matched to the completed adaptive run, including chunked entropy;
- no candidate selection, G-Vendi selection, Prismatic selection, difficulty-aware entropy, length penalty, rethinking probe, or self-training;
- resume disabled and output overwrite forbidden.

The full concise diagnostic adds generation cost. This treatment is not response-token-neutral, FLOP-neutral, or wall-clock-neutral.

## 4. Definitions

For each primary question, define:

- `y_n`: the student response generated from the unchanged normal/raw prompt;
- `L_n`: the number of valid tokens in `y_n`, computed as `response_mask.sum()`;
- `r_n`: the binary exact-answer reward for `y_n`;
- `y_c`: a diagnostic student response generated from the concise prompt, only when `r_n = 1`;
- `L_c`: the number of valid tokens in `y_c`;
- `r_c`: the binary exact-answer reward for `y_c`;
- `B`: the sensitive-route teacher budget, fixed to `round(1.0 * L_n) = L_n`.

`y_c` is a routing diagnostic only. It is never an actor target and never reaches old-log-prob, 30B reference scoring, advantage computation, rollout correction, or actor update.

## 5. Correctness rule

Both normal and concise correctness use the existing DeepMath reward path:

1. decode the valid student response tokens;
2. extract the final `\boxed{...}` answer;
3. truncate the extracted answer to 300 characters when necessary;
4. compare it with the dataset ground truth using `math_verify.parse` and `math_verify.verify`;
5. return `1.0` for a verified equivalent answer and `0.0` otherwise.

Missing or malformed boxed answers and parser exceptions receive zero reward. Routing converts the binary sequence reward to a Boolean using `sequence_reward > 0.5`. This threshold is not a confidence estimate.

EOS is telemetry, not part of correctness. A response that reaches the 16,384-token cap after already producing a verified final box may still be correct. A mathematically correct but unboxed answer is incorrect under the frozen reward contract.

## 6. Prompt definitions

### 6.1 Primary actor prompt

Every `y_n` is sampled and trained under the unchanged normal/raw prompt. The actor prompt never changes by route.

### 6.2 Concise diagnostic and easy-route teacher prompt

The concise prompt reuses the existing builder:

```text
{problem}
Solve concisely. Avoid unnecessary explanation. Put your final answer within \boxed{}.
```

It is used for:

- generating `y_c` on normal-correct rows;
- 30B teacher/reference scoring on easy rows.

The diagnostic has no per-row cap derived from `L_n`. Its only response limit is the global 16,384-token safety bound, and it stops early on natural EOS.

### 6.3 Sensitive-route budget teacher prompt

The sensitive route reuses the existing rollout-length TALE teacher prompt with `alpha=1.0`:

```text
{problem}
Let's think step by step and use less than {B} tokens. Put your final answer within \boxed{}.
```

where `B = L_n`. There is no online budget-estimation call and no ESR mask. The old global TALE training path remains disabled because it also owns truncation behavior that is outside this treatment.

### 6.4 Hard-route teacher prompt

The hard route gives the 30B teacher the unchanged normal/raw prompt, exactly as Vanilla OPD.

## 7. Route table

| Normal reward | Concise diagnostic | Route | 30B teacher prompt | Actor trajectory | Route weight |
|---|---|---|---|---|---:|
| correct | correct | `easy` | concise | complete `y_n` | 1.0 |
| correct | incorrect | `sensitive` | budget, `B=L_n` | complete `y_n` | 1.0 |
| incorrect | not generated | `hard` | normal | complete `y_n` | 1.0 |

The partition is event-level and stochastic. The route names do not claim intrinsic or permanent question difficulty.

## 8. Endpoint-preserving supervision

All routes train on the original complete `y_n` tensor and original response mask. The routing stage must not:

- replace `y_n` with `y_c`;
- truncate `y_n` physically or logically;
- remove response tokens to pay for diagnostic generation;
- remap the sequence reward to an earlier token;
- change the actor prompt;
- apply a route-specific loss multiplier;
- append a synthetic EOS.

If `y_n` naturally emits EOS, that token remains in the response and supervised mask. If `y_n` reaches the global cap without EOS, the complete capped response is retained exactly as Vanilla would retain it.

The pre-routing and post-routing primary batch must have identical response IDs, response masks, valid response lengths, attention masks, and sequence rewards. Only the per-row 30B teacher prompt and routing metadata may change.

## 9. OPD objective and ordering

For every row, old log-prob is computed on the complete `y_n` under the actor's normal/raw prompt. The 30B teacher/reference log-prob is computed on the same complete response tokens under the route-specific teacher prompt. The existing reverse-KL-only OPD advantage and actor update are then applied with unit route weight.

The training step order is:

1. load 1,024 distinct questions;
2. generate 1,024 normal/raw responses `y_n`;
3. construct the full normal response mask and compute normal rewards;
4. select exactly the normal-correct rows;
5. generate one full-cap concise diagnostic `y_c` for every selected row;
6. compute concise rewards and map diagnostics back to original row indices;
7. create the exhaustive `easy`/`sensitive`/`hard` partition;
8. attach one route-specific 30B teacher prompt to every primary row;
9. verify that the complete primary actor batch and rewards are unchanged;
10. discard the diagnostic batch;
11. compute full-response old log-prob;
12. retokenize route-specific teacher inputs and compute full-response 30B reference log-prob;
13. compute advantages, rollout correction, and the actor update on all original response tokens.

Route construction must happen before teacher retokenization. The diagnostic batch must be unreachable after step 10.

## 10. Isolation from historical treatments

The implementation must use a new configuration namespace, for example `algorithm.adaptive_triprompt_opd`, rather than changing the semantics of `algorithm.adaptive_concise_opd`. The completed adaptive treatment, launchers, tests, checkpoints, and metrics remain historical and reproducible.

The new path may reuse pure builders and validation utilities, but it must not enable the old adaptive prefix truncation or the global TALE ESR flow. Configuration validation must reject simultaneous enablement of:

- old adaptive-concise OPD;
- new tri-prompt OPD;
- global TALE budgeting;
- difficulty-aware OPD;
- candidate selection;
- rethinking probes;
- length-aware actor losses.

## 11. Metrics

All treatment metrics use a new prefix such as `adaptive_triprompt_opd/`.

### 11.1 Route counts

```text
adaptive_triprompt_opd/total_questions
adaptive_triprompt_opd/normal_correct_count
adaptive_triprompt_opd/normal_wrong_count
adaptive_triprompt_opd/concise_probe_count
adaptive_triprompt_opd/easy_count
adaptive_triprompt_opd/sensitive_count
adaptive_triprompt_opd/hard_count
adaptive_triprompt_opd/easy_ratio
adaptive_triprompt_opd/sensitive_ratio
adaptive_triprompt_opd/hard_ratio
```

Required per-step identities:

```text
total_questions      = 1024
normal_correct_count = easy_count + sensitive_count
normal_wrong_count   = hard_count
concise_probe_count  = normal_correct_count
total_questions      = easy_count + sensitive_count + hard_count
```

The same route counts are accumulated across completed steps as routing-event counts.

### 11.2 Teacher prompt counts

```text
adaptive_triprompt_opd/concise_teacher_count
adaptive_triprompt_opd/budget_teacher_count
adaptive_triprompt_opd/normal_teacher_count
```

Required identities:

```text
concise_teacher_count = easy_count
budget_teacher_count  = sensitive_count
normal_teacher_count  = hard_count
```

### 11.3 Length, EOS, and cap telemetry

Record normal and concise response token totals, means, EOS counts/ratios, and global-cap hit counts. Record normal response lengths by route and concise diagnostic lengths by resulting route. Record concise missing-box counts separately from reward failures.

For sensitive rows, record budget min/mean/max and require each per-row budget to equal its paired `L_n`.

### 11.4 Supervision preservation

```text
adaptive_triprompt_opd/normal_response_tokens
adaptive_triprompt_opd/actor_supervised_tokens
adaptive_triprompt_opd/supervision_token_residual
adaptive_triprompt_opd/full_response_preservation_ratio
adaptive_triprompt_opd/route_weight_min
adaptive_triprompt_opd/route_weight_max
```

Every successful step must report:

```text
actor_supervised_tokens           = normal_response_tokens
supervision_token_residual        = 0
full_response_preservation_ratio  = 1.0
route_weight_min                  = 1.0
route_weight_max                  = 1.0
```

### 11.5 Timing

Keep existing timing metrics and add separate normal rollout, normal reward, concise diagnostic, concise reward, routing, old-log-prob, 30B reference, actor update, and total-step durations. Runtime measurements must not be described as compute neutrality.

## 12. Fail-closed invariants

Abort before any actor update if:

- the primary batch is not exactly 1,024 rows or does not contain 1,024 distinct question identities;
- a normal-wrong row receives a concise diagnostic;
- a normal-correct row lacks exactly one aligned concise diagnostic;
- any normal or concise reward is non-finite;
- any response mask is non-binary or not a contiguous valid-token prefix;
- any normal or concise valid response length is outside `[1, 16384]`;
- the route masks are not mutually exclusive and exhaustive;
- route and teacher-prompt count identities fail;
- any sensitive budget is not exactly the corresponding `L_n`;
- any primary response ID, response mask, valid length, attention mask, or sequence reward changes during routing;
- an existing natural normal EOS is removed;
- a synthetic EOS is added to a capped normal response;
- any diagnostic row or tensor reaches old-log-prob, reference scoring, advantage computation, or actor update;
- any route weight differs from 1.0;
- any forbidden treatment is simultaneously enabled.

An empty diagnostic sub-batch is valid when every normal response is wrong. Then every row is hard and exactly follows Vanilla.

Failed runtime namespaces are preserved and never resumed or overwritten.

## 13. Testing strategy

Implementation follows test-driven development.

### 13.1 Pure unit tests

Tests cover:

- all three route-table outcomes;
- full-cap concise diagnostics and natural early EOS;
- normal and concise global-cap hits without synthetic EOS;
- exact-answer, missing-box, and parser-failure routing behavior;
- sensitive `B=L_n` calculation and exact prompt text;
- empty probe sub-batches;
- invalid or duplicate mappings;
- non-finite rewards and malformed masks;
- partition, prompt-count, and cumulative-count identities;
- unchanged full-response token/reward accounting.

### 13.2 Trainer integration tests

A synthetic trainer fixture verifies:

- one normal rollout per original question;
- concise generation only for normal-correct rows;
- the concise call uses a global 16,384 response limit rather than a per-row half-length cap;
- diagnostics are discarded before post-routing model calls;
- the primary response tensors and rewards are byte-for-byte unchanged;
- teacher prompts align with the original rows after batch balancing;
- old/ref/actor stages see exactly 1,024 complete primary responses;
- 30B reference scoring receives concise, budget, and normal prompts on the correct routes;
- all required metrics are emitted.

### 13.3 Regression and launcher tests

The focused new suite, existing adaptive suite, and legacy OPD suite must all pass. Launcher dry-run tests pin all frozen comparison fields, the new experiment namespace, full-response behavior, full concise diagnostic cap, mutual exclusions, resume disablement, and immutable output paths.

## 14. Runtime gates and launch policy

Before a full run, execute a fresh one-step profile in a new namespace on the four allocation-owned GPUs in `opd-CLI`. The profile must show:

- all route and prompt-count identities;
- exactly 1,024 complete actor rows;
- zero supervision residual and preservation ratio 1.0;
- finite old/ref log-probs, loss, entropy, and gradient norm;
- no diagnostic leakage;
- no OOM, worker failure, or catastrophic rollout-correction rejection;
- measured step time and a conservative projection that fits the remaining allocation.

The user has approved launching the full 50-step treatment after implementation and profile gates pass. If any gate fails or projected runtime exceeds the available allocation, stop fail-closed and report instead of launching.

Suggested immutable namespaces:

```text
profile: opd-adaptive-fullconciseprobe-triprompt-fullnormal-rawprompt-4gpu-tp4-profile-step1
full:    opd-adaptive-fullconciseprobe-triprompt-fullnormal-rawprompt-4gpu-tp4-step50
```

No W&B run, checkpoint directory, or formal log may reuse a historical namespace.

## 15. Evaluation policy

Training-time score is diagnostic only. Held-out evaluation remains authoritative. After a successful full run and checkpoint validation, use the same normal-prompt, `n=32`, seed-42 AIME24/AIME25 protocol used for Vanilla and the completed adaptive treatment. Evaluation launch remains separately gated by successful training artifacts and available resources; stopped canonical-eight work is not implicitly resumed.

## 16. Expected comparisons and non-claims

The primary comparison is against matched Vanilla OPD. Historical adaptive-concise, rollout-length TALE, normal-teacher ESR20, and concise-teacher ESR20 results provide mechanism context but do not replace the Vanilla control.

If all invariants pass, the treatment may claim:

- matched Vanilla question and update exposure;
- event-level three-prompt routing using normal and full-cap concise correctness;
- complete preservation of the normal actor trajectory and natural endpoint;
- 30B teacher supervision on every route;
- no concise self-training.

It must not claim:

- response-token, FLOP, or wall-clock neutrality;
- that easy, sensitive, or hard is an intrinsic question property;
- that final-answer reward certifies concise reasoning quality;
- that the treatment fixes normal-wrong failure tails;
- superiority before matched held-out evaluation.

## 17. Concise method statement

> For each question, we train on the complete normal-prompt student trajectory. A full-cap concise diagnostic is generated only when the normal response is correct and is used solely to route the 30B teacher: double-correct events receive a concise teacher prompt, normal-correct/concise-wrong events receive a rollout-length budget prompt with `B=L_n`, and normal-wrong events retain Vanilla's normal teacher prompt. Every route preserves the original response, reward, natural EOS, and unit OPD weight.
