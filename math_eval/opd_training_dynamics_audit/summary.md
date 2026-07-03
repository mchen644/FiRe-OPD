# OPD Training Dynamics Offline Audit

This report is generated from existing local W&B logs only. It does not launch training.

## Macro length summary

| run | peak step | peak original rollout length | final step | final original rollout length |
|---|---:|---:|---:|---:|
| hardtrunc_budget | 64 | 3209.1 | 64 | 3209.1 |
| hardtrunc_budget_resume75 | 73 | 3555.2 | 75 | 3264.3 |
| hardtrunc_concise | 2 | 1541.1 | 51 | 570.7 |
| hardtrunc_normal | 80 | 4280.1 | 87 | 4149.4 |
| raw_opd | 19 | 8218.8 | 69 | 4895.8 |

## Initial interpretation

- Raw OPD should be inspected around its peak-length step for truncation-repetition inflation.
- Concise-teacher hardtrunc should be inspected for entropy collapse and short-mode attraction.
- Rollout-length-budget hardtrunc should be inspected using original rollout length, not supervised length.

## Limitations

- Step18/19 mechanism-level proof requires step18/19 checkpoints. If unavailable, token-level claims are limited to available checkpoints and log-level consistency.