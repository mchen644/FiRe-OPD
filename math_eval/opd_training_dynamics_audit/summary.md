# OPD Training Dynamics Offline Audit

This report is generated from existing local W&B logs and AIME24 eval outputs only. It does **not** launch new training.

## Artifacts

- `metrics_by_step.csv`: parsed W&B training metrics.
- `macro_length_score_entropy_grad.svg`: cross-run training curves.
- `raw_opd_inflation_diagnostics.svg`: focused raw OPD curves.
- `repetition_metrics.csv`: AIME24 response-level repetition/truncation metrics.
- `alignment_metrics.csv`: not generated in this pass; the optional 30B model-scoring smoke was skipped to avoid GPU contention.

## Macro length summary

| run | peak step | peak original rollout length | final step | final original rollout length |
|---|---:|---:|---:|---:|
| raw_opd | 19 | 8218.8 | 69 | 4895.8 |
| hardtrunc_budget | 64 | 3209.1 | 64 | 3209.1 |
| hardtrunc_budget_resume75 | 73 | 3555.2 | 75 | 3264.3 |
| hardtrunc_normal | 80 | 4280.1 | 87 | 4149.4 |
| hardtrunc_concise | 2 | 1541.1 | 51 | 570.7 |

For hardtrunc runs, `response_length/mean` is the supervised/truncated training width. Original rollout length is `tale_budget/response_length_mean`.

## Findings

### 1. Raw OPD length inflation

Log-level evidence matches the 2604.08527 truncation-repetition inflation picture:

| step | original rollout length | score | actor entropy | grad norm | clip ratio |
|---:|---:|---:|---:|---:|---:|
| 1 | 1544.5 | 0.5078 | 0.3481 | 4.40 | 0.0000 |
| 15 | 3084.3 | 0.6416 | 0.2999 | 1.80 | 0.0039 |
| 18 | 8085.0 | 0.7295 | 0.2965 | 3.02 | 0.1416 |
| 19 | 8218.8 | 0.7109 | 0.2884 | 2.72 | 0.1709 |
| 64 | 4819.1 | 0.7393 | 0.3057 | 0.46 | 0.0176 |

Interpretation: around steps 18-19, rollout length and clip ratio jump abruptly. This is consistent with OPD entering a truncation-prone long-tail regime. Direct repeated-token advantage proof at the transition requires step18/19 checkpoints, which are not currently available.

AIME24 response-level eval repetition also moves in the expected direction:

| run | avg length | accuracy | truncation rate | compression repetition | 4-gram repetition |
|---|---:|---:|---:|---:|---:|
| raw_opd | 9219.6 | 0.5615 | 0.1938 | 0.7105 | 0.1902 |
| normal20 | 8106.2 | 0.5115 | 0.1344 | 0.7034 | 0.1785 |
| budget20 | 7245.3 | 0.4917 | 0.0813 | 0.6988 | 0.1763 |

Raw OPD has the longest AIME24 responses and highest truncation/repetition proxies among these three AIME24 eval outputs.

### 2. Concise-teacher + 20% prefix supervision collapse

Concise-teacher hardtrunc shows a different failure mode: length collapses rather than inflates.

| step | original rollout length | supervised length | score | actor entropy | grad norm |
|---:|---:|---:|---:|---:|---:|
| 1 | 1510.9 | 302.2 | 0.5391 | 0.3494 | 11.84 |
| 10 | 1204.0 | 240.8 | 0.5762 | 0.2544 | 8.16 |
| 18 | 626.8 | 125.4 | 0.6172 | 0.1215 | 14.01 |
| 20 | 547.9 | 109.6 | 0.6377 | 0.1074 | 14.27 |
| 50 | 531.5 | 106.3 | 0.5811 | 0.1511 | 18.40 |

Interpretation: the concise teacher prompt likely creates a low-entropy short-solution target on normal student prefixes. Under reverse-KL prefix supervision, the student is pulled into a short mode. This is consistent with 2604.13016's warning that prompt-aligned/over-concentrated supervision can reduce entropy and cause collapse.

### 3. Student-rollout-length budget stability

Budget-teacher hardtrunc is stable relative to both raw OPD and concise teacher.

| run/step | original rollout length | supervised length | score | actor entropy | grad norm |
|---|---:|---:|---:|---:|---:|
| budget step1 | 1510.9 | 302.2 | 0.5391 | 0.3494 | 8.45 |
| budget step20 | 1580.2 | 316.0 | 0.6582 | 0.2054 | 5.04 |
| budget step50 | 2735.2 | 547.0 | 0.7070 | 0.3141 | 1.81 |
| budget step75 | 3264.3 | 652.9 | 0.7412 | 0.3398 | 1.50 |
| normal step50 | 3567.6 | 713.5 | 0.7021 | 0.3369 | 1.46 |
| normal step87 | 4149.4 | 829.9 | 0.7246 | 0.3444 | 1.15 |

Interpretation: using the student rollout length as the teacher budget anchors the teacher target near the current student policy rather than forcing a much shorter mode. Prefix-only supervision also avoids the late-depth regions where OPD token rewards are more likely to become noisy or repetitive.

## Limitations

- Some W&B output lines are truncated at 4096 characters and therefore miss `response_length/mean` for certain steps; conclusions use steps where the relevant fields are present.
- Step18/19 mechanism-level proof requires step18/19 checkpoints. Without them, raw OPD conclusions are log-level evidence plus eval-output consistency.
- The optional top-k alignment scorer is implemented but not run in this pass because loading the 30B teacher with Transformers may contend with active GPU work. Running it later would test the 2604.13016 metrics directly: top-k overlap, overlap mass, student/teacher entropy, entropy gap, and overlap-token advantage.

## Next metrics to log in reruns

- compression-based repetition ratio;
- repeated-token vs regular-token reverse-KL advantage;
- top-k overlap ratio;
- overlap token probability mass;
- student and teacher entropy gap;
- position-binned entropy.

## Recommended next experiment

Run the normal-teacher prefix-supervision sweep at fixed step50:

```text
student prompt: normal
teacher prompt: normal
x in {0.1, 0.2, 0.4, 0.6, 0.8, 1.0}
```

Here `x=1.0` is the raw OPD endpoint. This isolates supervision depth from teacher prompt effects before sweeping budget or concise prompts.
