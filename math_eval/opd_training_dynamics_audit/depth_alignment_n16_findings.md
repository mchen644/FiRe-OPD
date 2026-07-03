# Depth-binned Rethinking OPD Alignment: AIME24 n=16 t=256

Outputs:

- `depth_raw_opd_aime24_n16_t256/alignment_metrics.csv`
- `depth_budget20_aime24_n16_t256/alignment_metrics.csv`
- `depth_normal20_aime24_n16_t256/alignment_metrics.csv`
- aggregate: `depth_alignment_aime24_n16_t256_summary.csv`

Setup:

- AIME24 eval outputs, first 16 problems, first response per problem.
- Teacher prompt style: normal baseline only.
- top-k = 16.
- Window length = 256 response tokens.
- Windows start at response tokens: 0, 1024, 2048, 4096, 8192.

## Mean top-k overlap by depth

| window start | raw_opd | budget20 | normal20 |
|---:|---:|---:|---:|
| 0 | 0.6693 | 0.6671 | 0.6701 |
| 1024 | 0.6798 | 0.6703 | 0.6725 |
| 2048 | 0.6798 | 0.6811 | 0.6849 |
| 4096 | 0.6955 | 0.6882 | 0.6948 |
| 8192 | 0.6402 | 0.6483 | 0.6427 |

Interpretation:

- Early windows remain comparable across methods.
- Overlap improves through mid-depth, peaking around 4096.
- The notable degradation is from 4096 to 8192.
- At 8192, budget20 is slightly higher than raw/normal20, but the cross-run difference is not statistically reliable in this n=16 / first-response sample.

## Paired depth changes

### 8192 - 0

| run | n | top-k overlap change | bootstrap 95% CI |
|---|---:|---:|---:|
| raw_opd | 10 | -0.0301 | [-0.0577, -0.0010] |
| budget20 | 6 | -0.0218 | [-0.0500, +0.0081] |
| normal20 | 6 | -0.0292 | [-0.0645, +0.0072] |

### 8192 - 4096

| run | n | top-k overlap change | bootstrap 95% CI |
|---|---:|---:|---:|
| raw_opd | 10 | -0.0586 | [-0.0881, -0.0306] |
| budget20 | 6 | -0.0380 | [-0.0652, -0.0103] |
| normal20 | 6 | -0.0477 | [-0.0773, -0.0150] |

Interpretation:

- The late-depth drop from 4096 to 8192 is visible for all three methods.
- raw_opd has the largest mean drop; budget20 has the smallest mean drop; normal20 is in between.
- However, between-method comparisons at 8192 are weak because the runs use different responses and only 4-6 problem-aligned late-window examples overlap.

## What this supports

Supported:

1. Early-prefix top-k alignment is not obviously worse for raw OPD.
2. There is a real late-depth top-k overlap degradation from 4096 to 8192.
3. This is consistent with Rethinking OPD's claim that long-horizon OPD instability originates at later tokens.

Not yet strongly supported:

1. budget20 is definitively better than raw OPD on late-depth alignment.
2. budget20's slightly higher 8192 top-k overlap is statistically reliable.

Recommended next step:

- Score more late-window samples by selecting responses that actually reach 8192 tokens.
- Use multiple responses per problem rather than only the first response.
- Keep the same problem indices across methods, but report that responses are still method-specific.
- Add windows at 6144 and 12288 to identify where degradation begins and whether it continues.
