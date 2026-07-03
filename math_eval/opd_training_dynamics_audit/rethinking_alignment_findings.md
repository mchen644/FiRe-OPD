# Rethinking OPD Alignment Metrics: AIME24 n=8 t=512

Command outputs:

- `alignment_raw_opd_aime24_n8_t512/alignment_metrics.csv`
- `alignment_budget20_aime24_n8_t512/alignment_metrics.csv`
- `alignment_normal20_aime24_n8_t512/alignment_metrics.csv`
- aggregate: `alignment_aime24_n8_t512_summary.csv`

Setup:

- AIME24 eval outputs, first 8 problems, first response per problem.
- Score first 512 response tokens.
- Student model = matching step50 checkpoint for each run.
- Teacher model = `models/Qwen3-30B-A3B-Instruct-2507`.
- Teacher prompt styles: normal, budget, concise.
- top-k = 16.

## Baseline teacher prompt alignment

| run | top-k overlap | student overlap mass | teacher overlap mass | student entropy | teacher entropy | entropy gap | overlap-token advantage |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw_opd | 0.6679 | 0.9968 | 0.9956 | 0.2280 | 0.2675 | 0.1317 | -0.2353 |
| budget20 | 0.6620 | 0.9981 | 0.9968 | 0.1797 | 0.2192 | 0.1210 | -0.2544 |
| normal20 | 0.6676 | 0.9978 | 0.9969 | 0.2119 | 0.2404 | 0.1311 | -0.2644 |

Observation:

- All three runs have high overlap mass, consistent with Rethinking OPD's claim that the shared high-probability token set carries most probability mass.
- budget20 has the smallest entropy gap under the normal teacher prompt, but also lower student entropy than raw/normal20.
- raw_opd does not look locally misaligned in the first 512 tokens; its failure mode is likely late-depth length/repetition/truncation rather than early-token top-k mismatch.

## Teacher prompt style effect

| run | teacher prompt | top-k overlap | entropy gap | overlap-token advantage |
|---|---|---:|---:|---:|
| raw_opd | normal | 0.6679 | 0.1317 | -0.2353 |
| raw_opd | budget | 0.6651 | 0.1309 | -0.2536 |
| raw_opd | concise | 0.6529 | 0.1415 | -0.3493 |
| budget20 | normal | 0.6620 | 0.1210 | -0.2544 |
| budget20 | budget | 0.6594 | 0.1176 | -0.2671 |
| budget20 | concise | 0.6503 | 0.1264 | -0.3427 |
| normal20 | normal | 0.6676 | 0.1311 | -0.2644 |
| normal20 | budget | 0.6636 | 0.1325 | -0.2934 |
| normal20 | concise | 0.6521 | 0.1453 | -0.3800 |

Observation:

- Concise teacher prompt consistently reduces top-k overlap and makes overlap-token advantage much more negative.
- Budget teacher prompt is usually intermediate between normal and concise.
- This supports the hypothesis that concise teacher prompt changes local teacher-student geometry, not just response length.

## Caveat and next metric

This pass only scores the first 512 tokens. It is enough to show prompt-style target shift, but not enough to prove raw OPD's late-depth degradation. The next Rethinking-style diagnostic should be position/depth-binned alignment windows, e.g. score windows at response positions 0, 1K, 2K, 4K, 8K and plot top-k overlap, entropy gap, and overlap-token advantage by depth.
