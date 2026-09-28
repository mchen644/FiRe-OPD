# TALE-Style Budgeted Teacher Prompt OPD Design

## Goal

Build an on-policy prompt-layer OPD variant inspired by `Token-Budget-Aware LLM Reasoning` / TALE: at each PPO step, the current student policy estimates the response token budget needed for each prompt, that budget is used to build an in-memory teacher-only prompt, and OPD trains with normal student rollout prompts but budget-aware teacher/ref log-prob prompts.

## Paper / Code Findings

The official TALE repository is cloned at `/home/mchen/TALE`.

Relevant implementation details:

- Budget estimation prompt lives in `/home/mchen/TALE/utils/__init__.py:create_zero_shot_context()`:
  ```text
  Task: Analyze the given question and estimate the minimum number of tokens required to generate a complete and accurate response. Please Give the response by strictly following this format: [[budget]],for example: Budget: [[12]].
  ```
- TALE-EP combines that context with:
  ```text
  Below is the question:

  Question: "{question}"
  ```
- Budget parsing uses `[[digits]]`.
- Budget injection lives in `/home/mchen/TALE/utils/__init__.py:add_budget()` and replaces:
  ```text
  Let's think step by step:
  ```
  with:
  ```text
  Let's think step by step and use less than {budget} tokens:
  ```
- TALE-EP uses one budget-estimation call, then one budget-aware reasoning call. The budget-search scripts are separate analysis/data-generation tools that need ground-truth evaluation.

## Proposed OPD Variant

Use online/on-policy budget estimation during training. The offline parquet conversion path remains useful as an ablation and debugging tool, but it is not the main experiment path because it estimates budgets once with a fixed student checkpoint.

For each PPO training batch:

1. Keep the original `prompt` column unchanged for student rollout.
2. Extract the problem text from `prompt[0]["content"]`.
3. Remove FiRe-OPD's trailing verbose instruction:
   ```text
   Please reason step by step, and put your final answer within \boxed{}.
   ```
4. Ask the current actor/rollout worker to estimate a budget using TALE's estimation prompt.
5. Parse, clamp, and round the budget.
6. Build an in-memory `teacher_prompt` as a single user message:
   ```text
   {problem}
   Let's think step by step and use less than {budget} tokens. Put your final answer within \boxed{}.
   ```
7. Store the generated teacher prompts in `batch.non_tensor_batch["teacher_prompt"]`.
8. Compute teacher/ref log-probs with `data.ref_raw_prompt_key=teacher_prompt`.

The OPD trainer then uses:

```bash
data.return_raw_chat=True \
+data.ref_raw_prompt_key=teacher_prompt
```

This reuses existing FiRe-OPD support for separate student and teacher/ref prompts while keeping budgets tied to the current policy rather than a precomputed cache.

## First Experiment Scope

This first experiment is prompt-layer only:

- Use pure OPD loss (`only_reverse_kl_advantages=True`).
- Do not enable length-aware penalty.
- Do not enable correctcompress candidate-selection loss masking.
- Use `rollout.n=1`.
- Student rollout prompt remains normal/raw.
- Teacher/ref prompt is budget-aware and generated on-policy before ref log-prob computation.

This isolates whether a budget-aware teacher prompt can create shorter reasoning preference through teacher log-prob without changing actor loss or candidate selection.

## Budget Normalization

Raw student estimates can be malformed or too small/large. The online trainer config should support:

- `algorithm.tale_budget.enabled`, default `False`
- `algorithm.tale_budget.min_budget`, default `128`
- `algorithm.tale_budget.max_budget`, default `8192`
- `algorithm.tale_budget.round_to`, default `64`
- `algorithm.tale_budget.fallback_budget`, default `2048`
- `algorithm.tale_budget.estimation_max_tokens`, default `64`
- `algorithm.tale_budget.temperature`, default `0.1`
- `algorithm.tale_budget.top_p`, default `0.9`
- `algorithm.tale_budget.teacher_prompt_key`, default `teacher_prompt`

Normalization rule:

1. If parsing fails, use fallback.
2. Clamp to `[min_budget, max_budget]`.
3. Round to nearest `round_to`, with final value still clamped.

## Data Flow

```text
original train parquet
  prompt: normal user problem + boxed instruction
      |
      | student rollout with normal prompt
      v
rollout responses
      |
      | current actor/rollout worker estimates [[budget]] for each original prompt
      v
training batch non_tensor_batch
  prompt/raw_prompt: unchanged normal student prompt
  teacher_prompt: in-memory budget-aware teacher prompt
  tale_budget: normalized integer budget
  tale_budget_raw: parsed raw integer or null
  tale_budget_estimate_text: raw estimator response
      |
      | OPD training with +data.ref_raw_prompt_key=teacher_prompt
      v
student samples normal prompt responses
teacher/ref scores responses under budget-aware prompt
```

## Evaluation

Evaluate checkpoints with the existing Table-2 normal-prompt settings:

- baseline/normal prompt
- `--no_extra_prompt`
- `n=32`
- `max_tokens=16384`
- same math datasets as OPD raw / length-aware comparisons

This checks whether the compressed teacher prompt transfers back to normal-prompt inference.

## Risks

- Student budget estimates may be too low for hard problems, causing teacher to reject correct but necessarily long reasoning.
- Online budget estimation adds per-step generation overhead and may increase rollout/ref-logprob synchronization complexity.
- The budget prompt may affect teacher log-prob in a way that penalizes all long responses, including high-quality long solutions.
- Training stays normal-prompt for the student, so the model must internalize the teacher preference without seeing the budget prompt at inference.

## Mitigations

- Start with conservative budget bounds (`min=128`, `max=8192`, `fallback=2048`).
- Log `tale_budget/mean`, `tale_budget/min`, `tale_budget/max`, and `tale_budget/parse_fail_ratio` for auditing.
- Keep offline data-prep scripts as ablation-only tools for comparing fixed-budget and on-policy-budget behavior.
- Compare against OPD raw and direct length-aware at step 50 using the same eval pipeline.
