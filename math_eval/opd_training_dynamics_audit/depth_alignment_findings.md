# Depth-binned Rethinking OPD Alignment: AIME24 n=4 t=256

Command outputs:

- `depth_raw_opd_aime24_n4_t256/alignment_metrics.csv`
- `depth_budget20_aime24_n4_t256/alignment_metrics.csv`
- `depth_normal20_aime24_n4_t256/alignment_metrics.csv`
- aggregate: `depth_alignment_aime24_n4_t256_summary.csv`

Setup:

- AIME24 eval outputs, first 4 problems, first response per problem.
- Teacher prompt style: normal baseline only.
- top-k = 16.
- Window length = 256 response tokens.
- Windows start at response tokens: 0, 1024, 2048, 4096, 8192.

## Top-k overlap by depth

| window start | raw_opd | budget20 | normal20 |
|---:|---:|---:|---:|
| 0 | 0.6761 | 0.6671 | 0.6703 |
| 1024 | 0.6833 | 0.6758 | 0.6781 |
| 2048 | 0.6829 | 0.6920 | 0.6941 |
| 4096 | 0.6930 | 0.6956 | 0.6868 |
| 8192 | 0.6513 | 0.6935 | 0.6464 |

Observation:

- Early windows do not show raw OPD as locally worse. This supports the interpretation that raw OPD's failure is not an early thinking-pattern mismatch.
- At 8192 tokens, raw OPD and normal20 overlap drop sharply, while budget20 remains high in this small sample.
- This is consistent with a late-depth degradation story, but sample size is small and window availability depends on response length.

## Entropy gap by depth

| window start | raw_opd | budget20 | normal20 |
|---:|---:|---:|---:|
| 0 | 0.1402 | 0.1432 | 0.1414 |
| 1024 | 0.1236 | 0.0858 | 0.1006 |
| 2048 | 0.0952 | 0.1275 | 0.1219 |
| 4096 | 0.1127 | 0.1384 | 0.1282 |
| 8192 | 0.1356 | 0.1591 | 0.1827 |

Observation:

- Entropy gap is not monotonically worse for raw OPD in this n=4 sample.
- normal20 has the largest late 8192 entropy gap.
- budget20 preserves top-k overlap late, but its entropy gap still grows with depth.

## Overlap-token advantage by depth

| window start | raw_opd | budget20 | normal20 |
|---:|---:|---:|---:|
| 0 | -0.2659 | -0.3173 | -0.2788 |
| 1024 | -0.1641 | -0.1303 | -0.1296 |
| 2048 | -0.1302 | -0.1481 | -0.1720 |
| 4096 | -0.1591 | -0.1995 | -0.1669 |
| 8192 | -0.1914 | -0.2105 | -0.2577 |

Observation:

- The late 8192 window worsens for all methods relative to middle-depth windows.
- normal20 is worst on late overlap-token advantage in this small sample.
- budget20 is not best on advantage, but it keeps late top-k overlap higher.

## Relation to concise collapse

Separate prompt-style alignment on first 512 tokens showed that changing the teacher prompt to concise consistently changes early local geometry:

- lower top-k overlap;
- larger entropy gap;
- much more negative overlap-token advantage.

This supports a two-failure-mode picture:

1. **Length inflation / raw OPD:** early alignment looks acceptable; bad behavior likely emerges from late-depth rollout states and truncation/repetition tails.
2. **Concise collapse:** local teacher target shifts already in the early prefix, so the model is pulled into a short low-entropy mode before late-depth effects matter.
3. **Budget20:** early geometry is close to normal teacher, and late top-k overlap is preserved better in this small sample, though entropy gap and advantage still require larger-n confirmation.

## Caveats

- n=4 is exploratory.
- Only the first response per problem was scored.
- Late windows have fewer examples because short responses do not reach all depths.
- The current evidence is strongest for the qualitative early-vs-late distinction, not for a final ranking of methods.
