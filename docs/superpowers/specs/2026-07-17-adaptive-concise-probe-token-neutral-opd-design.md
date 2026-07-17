# Adaptive Concise-Probe Token-Neutral OPD Design

Date: 2026-07-17

Status: approved conversational design; awaiting review of this written specification

## 1. Objective

Build a compression-focused OPD treatment that preserves Vanilla OPD's question coverage while using an adaptive student probe to decide whether the teacher should receive a concise or normal prompt.

The treatment must preserve these Vanilla exposure invariants:

- 1,024 distinct training questions per optimizer step;
- one normal student rollout per question;
- exactly 1,024 normal-rollout rows entering OPD training per step;
- 50 optimizer steps and therefore 51,200 question slots;
- no replacement of question coverage by repeated same-question rollouts.

The treatment measures **compression safety**, not absolute mathematical difficulty. G-Vendi, Prismatic synthesis, gradient selection, candidate selection, and multi-rollout best-of-N selection are outside this experiment.

## 2. Selected approach

The selected method is an **adaptive asymmetric two-prompt probe**:

1. Generate one normal-prompt student rollout for every question.
2. Score every normal rollout with the existing exact-answer reward.
3. Generate one additional concise-prompt student rollout only for normal-correct questions.
4. Limit each concise rollout to at most 50% of its corresponding normal rollout's valid response length.
5. Use concise correctness only to choose the teacher prompt.
6. Train only on the original normal rollout.
7. Pay for every concise diagnostic response token by removing one token from the end of the corresponding normal-rollout supervision window.

This design was selected over:

- **fixed two-rollout routing**, rejected because it probes normal-wrong questions even though both learnable and hard questions receive the same normal teacher route;
- **256 questions with four rollouts**, rejected because it reduces distinct-question exposure to one quarter of Vanilla;
- **batch-global water-filling**, rejected because per-example accounting is simpler and becomes exactly response-token neutral once learnable probes also reduce their own supervision width.

## 3. Frozen comparison contract

Except for the adaptive diagnostic, teacher-prompt routing, and response supervision width described here, the treatment inherits the Vanilla OPD launcher contract:

- student: Qwen3-4B;
- teacher/ref: Qwen3-30B-A3B-Instruct-2507;
- training data: `/home/mchen/FiRe-OPD/data/g-opd/DeepMath-103K/train_filtered_level6.parquet`, with the same shuffle/order contract as Vanilla;
- raw normal student prompts;
- normal rollout `n=1`;
- global question batch 1,024;
- PPO mini-batch 1,024;
- 50 optimizer steps;
- seed 42;
- maximum normal response length 16,384;
- learning rate `1e-6`, zero warmup;
- reverse-KL-only advantages;
- token-mean loss aggregation;
- token-level rollout importance sampling with threshold 5.0;
- no TALE budget estimation, confidence routing, hard entropy, candidate selection, rethinking probe, length penalty, or G-Vendi selection;
- resume disabled and output overwrite forbidden.

The adaptive probe is diagnostic-only. It does not increase the actor training row count above 1,024.

## 4. Prompt definitions

### 4.1 Normal student and teacher prompt

The normal prompt is the unchanged raw Vanilla OPD math prompt. It retains the existing normal reasoning instruction and is used for:

- every primary student rollout;
- teacher scoring for learnable and hard samples.

### 4.2 Concise student probe and teacher prompt

The concise prompt uses the existing concise message builder and wording:

```text
{problem}
Solve concisely. Avoid unnecessary explanation. Put your final answer within \boxed{}.
```

It is used for:

- the adaptive diagnostic student rollout on normal-correct questions;
- teacher scoring for easy samples.

The concise teacher never generates a target response. It scores a prefix of the **normal student rollout** under the concise prompt.

## 5. Per-question data flow

Let:

- \(L_n\) be the number of valid response tokens in the normal student rollout;
- \(C=\lfloor 0.5L_n\rfloor\) be the concise diagnostic maximum response length;
- \(L_c\) be the concise diagnostic's actual valid response length;
- \(L_s\) be the number of normal-rollout tokens retained for OPD supervision.

Lengths are counted from response masks, excluding padding. Every normal rollout must have \(L_n\ge1\). For every normal-correct row, the implementation must require \(L_n\ge2\), which guarantees \(C\ge1\).

### 5.1 Hard route

If the normal rollout is incorrect:

- do not generate a concise diagnostic;
- define \(L_c=0\);
- classify the question as `hard`;
- use the normal teacher prompt;
- retain the full normal rollout, so \(L_s=L_n\).

This is a conservative route: some normal-wrong questions might succeed under another stochastic sample, but they are never incorrectly sent to the concise teacher.

### 5.2 Learnable route

If the normal rollout is correct and the capped concise rollout is incorrect:

- classify the question as `learnable`;
- use the normal teacher prompt;
- train on the first \(L_s=L_n-L_c\) tokens of the normal rollout.

Incorrect includes wrong extracted answers, missing/invalid extracted answers, and responses truncated by the concise cap before producing a correct verified answer.

### 5.3 Easy route

If both the normal rollout and capped concise rollout are correct:

- classify the question as `easy`;
- use the concise teacher prompt;
- train on the first \(L_s=L_n-L_c\) tokens of the normal rollout.

Because \(L_c\le\lfloor0.5L_n\rfloor\), both easy and learnable rows retain at least \(\lceil0.5L_n\rceil\) normal-response tokens for supervision.

### 5.4 Route table

| Normal result | Concise probe | Category | Teacher prompt | Supervised normal tokens |
|---|---|---|---|---:|
| incorrect | not generated | hard | normal | \(L_n\) |
| correct | incorrect | learnable | normal | \(L_n-L_c\) |
| correct | correct | easy | concise | \(L_n-L_c\) |

Easy and learnable rows differ only in teacher prompt. Both pay for the diagnostic using the same per-example supervision rule.

## 6. Response-token budget identity

For every row, including hard rows where \(L_c=0\):

\[
L_c + L_s = L_n.
\]

Therefore, for every batch:

\[
\sum_i L_{c,i} + \sum_i L_{s,i} = \sum_i L_{n,i}.
\]

Example: if \(L_n=350\) and \(L_c=50\), the concise diagnostic consumes 50 response tokens and OPD retains the first 300 normal-response tokens. The total remains 350 response tokens, matching Vanilla's full-response supervision width under the selected accounting.

This identity permits the claim **response-token-budget neutral**. It does not by itself permit a claim of identical FLOPs or wall-clock time because autoregressive student decoding and teacher/actor sequence processing have different per-token costs and fixed overheads.

Prompt tokens are also outside the response-token identity. Their overhead and the second generation call must be measured explicitly.

## 7. Ordering and compute realization

The training step must execute in this order:

1. sample 1,024 distinct questions;
2. generate 1,024 normal rollouts;
3. compute normal rewards;
4. build the sub-batch of normal-correct questions;
5. generate capped concise diagnostics for only that sub-batch;
6. compute concise rewards and route every original question;
7. construct per-row teacher prompts and supervision masks;
8. physically truncate or remove-padding-mask the normal responses to \(L_s\);
9. recompute token balancing for the final 1,024-row actor batch;
10. compute old log-prob, teacher/ref log-prob, advantages, and actor update only on the retained normal prefixes.

The routing and truncation must happen **before** old-log-prob and teacher/ref computation. Applying only a loss mask after full-sequence forward passes would preserve the mathematical loss but would not realize the intended post-rollout compute savings.

Concise diagnostic rows must never be unioned into the OPD actor batch and must never receive old-log-prob, teacher/ref log-prob, advantages, or gradients.

## 8. Training-time metrics

All metrics are logged per step under `adaptive_concise_opd/`. Category counts are also logged cumulatively as routing-event counts through the latest completed step.

### 8.1 Question and route counts

```text
adaptive_concise_opd/total_questions
adaptive_concise_opd/normal_correct_count
adaptive_concise_opd/normal_wrong_count
adaptive_concise_opd/concise_probe_count
adaptive_concise_opd/easy_count
adaptive_concise_opd/learnable_count
adaptive_concise_opd/hard_count
adaptive_concise_opd/easy_ratio
adaptive_concise_opd/learnable_ratio
adaptive_concise_opd/hard_ratio
```

The following identities must hold every step:

```text
normal_correct_count = easy_count + learnable_count
normal_wrong_count   = hard_count
concise_probe_count  = easy_count + learnable_count
total_questions      = easy_count + learnable_count + hard_count = 1024
```

### 8.2 Cumulative counts

```text
adaptive_concise_opd/total_questions_cumulative
adaptive_concise_opd/concise_probe_count_cumulative
adaptive_concise_opd/easy_count_cumulative
adaptive_concise_opd/learnable_count_cumulative
adaptive_concise_opd/hard_count_cumulative
```

These are counts of routing events, not a claim that a repeated dataset record would be a new unique mathematical problem. The frozen data-order contract should independently ensure the intended 51,200 question slots.

### 8.3 Probe diagnostics

```text
adaptive_concise_opd/concise_correct_count
adaptive_concise_opd/concise_wrong_count
adaptive_concise_opd/concise_parse_fail_count
adaptive_concise_opd/concise_cap_hit_count
adaptive_concise_opd/concise_response_length_mean
adaptive_concise_opd/concise_to_normal_length_ratio_mean
```

Required identities:

```text
concise_correct_count = easy_count
concise_wrong_count   = learnable_count
```

A cap hit means the generated valid response length equals the per-row 50% cap. A parse failure is a concise response for which the answer extractor returns no valid answer.

### 8.4 Teacher routing

```text
adaptive_concise_opd/concise_teacher_count
adaptive_concise_opd/normal_teacher_count
adaptive_concise_opd/concise_teacher_ratio
adaptive_concise_opd/normal_teacher_ratio
```

Required identities:

```text
concise_teacher_count = easy_count
normal_teacher_count  = learnable_count + hard_count
```

### 8.5 Token accounting

```text
adaptive_concise_opd/normal_response_tokens
adaptive_concise_opd/concise_probe_tokens
adaptive_concise_opd/supervised_normal_tokens
adaptive_concise_opd/response_budget_residual_tokens
adaptive_concise_opd/response_budget_ratio
adaptive_concise_opd/supervised_fraction_mean
```

Definitions:

\[
\text{residual}=\sum_i L_{c,i}+\sum_iL_{s,i}-\sum_iL_{n,i},
\]

\[
\text{ratio}=\frac{\sum_i L_{c,i}+\sum_iL_{s,i}}{\sum_iL_{n,i}}.
\]

Every successful step must report:

```text
response_budget_residual_tokens = 0
response_budget_ratio           = 1.0
```

A nonzero residual is a fatal contract violation; the actor update must not run.

### 8.6 Timing

Retain existing timing metrics and add:

```text
timing_s/normal_rollout
timing_s/normal_reward
timing_s/concise_probe
timing_s/concise_reward
timing_s/old_log_prob
timing_s/ref
timing_s/update_actor
timing_s/step
```

Category counts and token identities establish the routing and response-token contract. Timing metrics determine whether the treatment is also approximately wall-clock neutral.

## 9. Error handling and fail-closed invariants

Before an actor update, the trainer must abort on any of the following:

- the primary batch does not contain exactly 1,024 rows or 1,024 distinct question identities;
- a concise diagnostic is generated for a normal-wrong row;
- a normal-correct row lacks exactly one mapped concise diagnostic;
- a concise diagnostic cap exceeds `floor(0.5 * normal_length)`;
- any normal response has zero valid tokens, or a probed normal response has fewer than two valid tokens;
- any normal or concise reward used for routing is non-finite;
- easy, learnable, and hard do not form a disjoint exhaustive partition;
- a diagnostic row reaches old-log-prob, teacher/ref, advantages, or actor update;
- any supervised prefix length is non-positive or exceeds its normal response length;
- the response-token budget residual is nonzero;
- teacher-prompt counts do not match route counts.

An empty probe sub-batch is valid when all normal rollouts are wrong. In that case every row is hard, all rows use the normal teacher, and the response-token identity still holds with zero diagnostic tokens.

Failed launches must use a new output namespace or be archived before retry. Resume and overwrite remain disabled.

## 10. Testing strategy

### 10.1 Pure routing tests

Unit tests cover:

- normal wrong -> hard, no probe, normal teacher, full supervision;
- normal correct + concise wrong -> learnable, normal teacher, `L_n-L_c` supervision;
- normal correct + concise correct -> easy, concise teacher, `L_n-L_c` supervision;
- odd and even normal lengths under the 50% cap;
- early EOS and exact-cap concise lengths;
- concise parse failure;
- empty probe sub-batch;
- exact per-row and batch token identities;
- non-finite rewards and invalid mappings fail closed.

### 10.2 Trainer integration tests

A synthetic trainer fixture verifies:

- exactly one normal rollout is generated per original question;
- only normal-correct rows enter the concise generation call;
- dynamic concise caps map back to the correct original rows;
- diagnostic rows are discarded before old/ref/actor stages;
- physical prefix truncation occurs before post-rollout model calls;
- final actor batch remains 1,024 rows;
- all count, ratio, cumulative, token, and timing metric keys are emitted.

### 10.3 Launcher tests

Dry-run tests pin the Vanilla comparison fields and reject:

- `rollout.n != 1` for the primary rollout;
- question batch sizes other than 1,024;
- actor mini-batches other than 1,024;
- total training steps other than 50;
- adaptive cap ratios other than 0.5;
- enabled candidate selection, G-Vendi selection, TALE estimation, hard entropy, length penalty, or resume.

### 10.4 Runtime profile gate

Before any full 50-step launch, run a fresh 1–3 step GPU profile using the allocation-owned four GPUs. The profile must report:

- all route-count identities;
- exactly 1,024 final actor rows;
- zero response-token residual;
- finite reverse-KL loss and gradient norm;
- no catastrophic rollout-correction rejection;
- separate normal and concise generation times;
- old/ref/update and total step times compared with the frozen Vanilla logs.

The profile supports an empirical wall-clock statement but does not alter the response-token contract. A full launch requires a separate explicit approval and sufficient allocation time.

## 11. Claims and non-claims

If all invariants pass, the experiment may claim:

- Vanilla-matched distinct-question and optimizer-update exposure;
- adaptive probing only on normal-correct questions;
- exact response-token accounting by construction;
- easy/learnable/hard routing-event counts observed during training;
- concise teacher routing only when the student solves within a 50%-length concise cap.

Without profiling evidence, it must not claim:

- identical GPU FLOPs;
- identical wall-clock training time;
- that `hard` is an intrinsic or permanent property of a question;
- that one capped stochastic concise failure proves a question is fundamentally uncompressible.

Recommended wording before runtime validation:

> We reallocate one normal-response supervision token for each adaptive concise diagnostic response token, preserving the per-batch response-token budget while retaining Vanilla OPD's question coverage. Easy and learnable examples use the same supervision-width accounting and differ only in whether the teacher receives a concise or normal prompt.
