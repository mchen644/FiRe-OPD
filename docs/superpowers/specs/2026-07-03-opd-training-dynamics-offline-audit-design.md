# OPD Training Dynamics Offline Audit Design

## Goal

Build an offline diagnostic report that explains three observed OPD training dynamics without launching new training jobs:

1. **Raw OPD length inflation** around steps 18-20.
2. **Concise-teacher + 20% prefix supervision length collapse**.
3. **Student-rollout-length budget + 20% prefix supervision stability**.

The report should turn current qualitative observations into evidence-backed plots and tables, then inform the next training sweep where prefix supervision fraction becomes the x-axis for acc-length frontiers.

## References and diagnostic lenses

### 2604.08527: OPD length inflation

Use this paper's truncation-repetition inflation lens for raw OPD. Key offline diagnostics:

- response length over training;
- max-length clip / truncation ratio;
- compression-based repetition ratio;
- student log-prob, teacher log-prob, and reverse-KL advantage on student rollouts;
- repeated-token vs regular-token reverse-KL advantage.

Hypothesis for raw OPD: around step 18-20, repetitive or low-information tails receive larger reverse-KL advantage; as on-policy sampling visits those tails more often, frequency and advantage reinforce each other, causing sudden length inflation.

### 2604.13016: token-level OPD alignment

Use this paper's high-probability-token alignment lens for prompt and supervision-depth effects. Key offline diagnostics:

- student-teacher top-k overlap ratio, default `k=16`;
- overlap-token advantage;
- student entropy and teacher entropy on student-visited states;
- absolute entropy gap;
- student/teacher probability mass assigned to overlap tokens;
- optional position-wise entropy to detect suffix-to-prefix instability.

Hypothesis for concise-teacher collapse: the concise teacher prompt gives a low-entropy short-answer target on normal student prefixes. Prefix-only reverse-KL supervision then rapidly concentrates student probability mass into a short/concise mode, reducing rollout length and actor entropy.

Hypothesis for rollout-length budget stability: the teacher budget is anchored to current student rollout length, so it is not a strong shorter-than-student attractor. Prefix-only supervision also avoids late noisy/repetitive regions, preventing both raw OPD-style inflation and concise-prompt collapse.

## Input data

### Existing logs

Use W&B local output logs under:

- `wandb/run-*/files/output.log`

Initial runs of interest:

- raw OPD: `wandb/run-20260621_181403-t3g8daac/files/output.log`;
- budget-teacher 20% hardtrunc first segment: `wandb/run-20260630_050733-bk8qeqt1/files/output.log`;
- budget-teacher 20% hardtrunc resume: `wandb/run-20260701_002104-h7aha9v2/files/output.log`;
- concise-teacher 20% hardtrunc: `wandb/run-20260702_063810-6joq0xfl/files/output.log`;
- normal-teacher 20% hardtrunc: `wandb/run-20260702_195307-ecq9x9ki/files/output.log`.

Extract at least:

- `training/global_step` or record `step`;
- `response_length/mean`, `response_length/max`, `response_length/clip_ratio`;
- `tale_budget/response_length_mean`, when present, for original rollout length under hard truncation;
- `critic/score/mean`;
- `actor/entropy`;
- `actor/grad_norm`;
- `rollout_corr/chi2_seq`;
- `rollout_corr/log_ppl_abs_diff`;
- `rollout_corr/skipped_no_valid_tokens`;
- `response/aborted_ratio`.

### Existing checkpoints

Use available checkpoints where possible:

- raw OPD step50;
- budget-teacher hardtrunc step50 and step75;
- normal-teacher hardtrunc step50;
- concise-teacher hardtrunc step50, if present.

If intermediate raw OPD step18/19 checkpoints are unavailable, the first report should state this limitation. Do not start a rerun merely to fill this gap unless a later review decides the offline evidence is insufficient.

### Existing eval rollouts

Use existing eval JSONL outputs where useful for response-level analyses such as answer position, trailing length, and repetition. Training-state token-level diagnostics should prefer fresh sampled rollouts from the target checkpoint if existing eval outputs lack logits or prompt metadata.

## Offline measurements

### 1. Macro training curves

Produce plots comparing raw OPD, budget-teacher hardtrunc, normal-teacher hardtrunc, and concise-teacher hardtrunc:

- original rollout length vs step;
- supervised/training response length vs step;
- score vs step;
- actor entropy vs step;
- grad norm vs step;
- clip/aborted/skipped-valid-token ratios vs step.

For hardtrunc runs, explicitly distinguish:

- `response_length/mean`: truncated supervised/training width;
- `tale_budget/response_length_mean`: original student rollout length.

### 2. Repetition and truncation audit

For sampled rollouts or existing response text, compute:

- compression ratio or repetition ratio following the spirit of 2604.08527;
- n-gram repetition rates as a transparent secondary metric;
- max-token truncation / clip rate;
- repeated-token fraction if token-level alignment scoring is available.

Then compare repeated-token advantage vs regular-token advantage using teacher/student log-prob differences on the same sampled tokens.

### 3. Token-level alignment audit

On a fixed prompt subset and selected checkpoints, score student rollouts with both student and teacher policies and compute:

- top-k overlap ratio (`k=16` by default; optionally `k=4,32` for robustness);
- overlap-token advantage;
- student entropy;
- teacher entropy under the selected teacher prompt style;
- entropy gap;
- probability mass on overlap tokens;
- position-binned versions of these metrics.

Teacher prompt styles to compare:

- normal baseline teacher prompt;
- numeric/student-rollout budget prompt;
- concise teacher prompt.

The concise analysis should keep student rollouts fixed when comparing teacher prompts, so the audit separates teacher-target shift from student-policy shift.

### 4. Depth/reliability probe, optional in first report

If runtime permits, implement a small version of the 2604.13016 prefix-depth probe:

- cut student rollouts at several prefix lengths;
- let the teacher continue or score from each prefix;
- measure whether teacher continuation advantage or teacher-student alignment degrades with depth.

This is optional for the first report because it is more expensive than log parsing and logit scoring.

## Outputs

Create an analysis output directory, e.g.:

- `math_eval/opd_training_dynamics_audit/`

Expected artifacts:

- `summary.md`: concise narrative report with hypotheses, plots, and limitations;
- `metrics_by_step.csv`: parsed log metrics;
- `alignment_metrics.csv`: checkpoint/prompt-style token-level diagnostics;
- `repetition_metrics.csv`: repetition/truncation diagnostics;
- plots:
  - `macro_length_score_entropy_grad.svg`;
  - `raw_opd_inflation_diagnostics.svg`;
  - `prompt_style_alignment_metrics.svg`;
  - optional `positionwise_entropy.svg`.

## Non-goals

- Do not launch new training jobs in this phase.
- Do not change training semantics.
- Do not claim step18/19 mechanism-level proof if no step18/19 checkpoint exists; label those claims as consistent evidence from logs plus later checkpoints.
- Do not use hardtrunc `response_length/mean` as actual rollout length; use `tale_budget/response_length_mean` where available.

## Success criteria

The audit is successful if it answers, with plots/tables:

1. whether raw OPD's step18-20 length jump coincides with increased repetition/truncation signals and anomalous advantage/log-prob dynamics;
2. whether concise-teacher collapse coincides with low entropy, high concentration on overlap tokens, or degraded prompt-style alignment;
3. whether student-rollout budget stability is visible in original rollout length and entropy curves;
4. which mechanism metrics should be logged in future short reruns or supervise-fraction sweeps.

## Follow-up experiment design enabled by this audit

After the offline audit, run a normal-teacher prefix-supervision sweep at fixed step50:

- student prompt: normal;
- teacher prompt: normal;
- train steps: 50;
- vary supervised prefix fraction `x in {0.1, 0.2, 0.4, 0.6, 0.8, 1.0}`;
- evaluate pass@1/pass@4 vs average response length.

Here `x=1.0` is the raw OPD endpoint. This isolates supervision depth from teacher prompt effects before any budget/concise prompt sweep.
