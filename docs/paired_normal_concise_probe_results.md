# Paired Normal/Concise Training-Set Probe Results

## Executive summary

The 128-question paired probe completed for both the original Qwen3-4B and the adaptive-concise step-50 checkpoint. Every question received one normal-prompt rollout and one concise-prompt rollout capped at half the corresponding normal response length. Normal-wrong rows were deliberately probed, unlike production adaptive training.

The main findings are:

1. **Most normal-correct events were not compression-safe at the 50% cap.** Among normal-correct events, concise remained correct for 17/46 (37.0%) on the base model and 24/55 (43.6%) on adaptive step 50. The majority were `normal correct / concise wrong`.
2. **Insufficient remaining scratch space explains many, but not all, compression-sensitive events.** Of cap-hit sensitive events, an exact-prefix continuation recovered 15/20 (75.0%) for base and 12/21 (57.1%) for adaptive step 50. Other events remained wrong after more budget or had already stopped naturally with a wrong answer.
3. **`normal wrong / concise correct` exists.** It occurred 6/128 times for base and 18/128 times for adaptive step 50. However, strict answer-format effects explain 1/6 and approximately 10/18 of these raw events; the remainder still includes substantive answer corrections and normal-rollout nontermination.
4. **Adaptive step 50 was much more verbose under the normal training prompt than the original base model.** Mean normal length was 5,885 versus 1,594 tokens, with 10 versus 0 normal cap hits. This comparison does not isolate adaptive routing from OPD training generally; held-out comparisons against the matched Vanilla OPD checkpoint remain authoritative for treatment effects.
5. **Correct final answers do not guarantee useful concise reasoning targets.** Some concise-rescued outputs had malformed or unsupported reasoning while receiving final-answer reward. Verified self-training should therefore not copy every reward-correct concise trajectory without an additional quality check.

The evidence argues against treating `normal correct / concise wrong` as compressible. Those rows should retain full endpoint-bearing supervision and strong-teacher guidance in the next routing design.

## Protocol and provenance

- Dataset: `/home/mchen/FiRe-OPD/data/g-opd/DeepMath-103K/train_filtered_level6.parquet`
- Dataset SHA256: `de3350fdd00bc0410550098ea65179e2be873da99e4075f80de575fc17670597`
- Sample: 128 distinct rows selected with NumPy seed 42
- Sample SHA256: `f6f327c74f6156cd4f64e23a0add922569417c9b13f769dd092ede5bab822466`
- Generation: `n=1`, temperature/top-p 1.0, normal cap 16,384, concise cap `max(1, floor(normal_length / 2))`
- Models: original Qwen3-4B and adaptive-concise step 50
- Formal source commit: `eaf9e285b6ec2781184b2ef9d2753976be89d41d`
- Formal run: `20260719T170414Z-eaf9e285b6ec`
- Formal output root: `/home/mchen/FiRe-OPD/math_eval/paired_normal_concise_probe_outputs/20260719T170414Z-eaf9e285b6ec`
- Formal log: `/home/mchen/FiRe-OPD/math_eval/paired_normal_concise_probe_logs/20260719T170414Z-eaf9e285b6ec.log`

The original independent same-seed relaxed resampling assumption was invalid: a diagnostic found exact capped-prefix preservation in only 2/21 requests. The remediated counterfactual instead supplies the exact observed capped token IDs as a fixed prompt suffix and generates only the remaining budget. It does not claim to resume the hidden RNG state of the original request.

## Validation

The formal launcher exited with:

```text
PAIRED_NORMAL_CONCISE_PROBE_GATE=PASS
PAIRED_NORMAL_CONCISE_PROBE_DONE_20260719T170414Z-eaf9e285b6ec:0
```

A fresh independent validator recomputed routes, caps, counterfactual provenance, summaries, identities, and 11 artifact hashes successfully. Counts were exactly 128 sample rows, 128 base records, and 128 adaptive records. The formal log contained no traceback, engine failure, or OOM signature.

Pre-launch gates also passed:

- 55 paired-probe tests;
- 87 focused adaptive tests;
- 102 legacy regression tests;
- Ruff, compileall, shell syntax, and `git diff --check`;
- a 21-row GPU profile containing 11 valid forced-prefix continuations.

## Four correctness quadrants

| Model | Normal ✓ / Concise ✓ | Normal ✓ / Concise ✗ | Normal ✗ / Concise ✓ | Normal ✗ / Concise ✗ | Normal accuracy | Concise accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Base Qwen3-4B | 17 (13.3%) | 29 (22.7%) | 6 (4.7%) | 76 (59.4%) | 46/128 (35.9%) | 23/128 (18.0%) |
| Adaptive step 50 | 24 (18.8%) | 31 (24.2%) | 18 (14.1%) | 55 (43.0%) | 55/128 (43.0%) | 42/128 (32.8%) |

Descriptively, adaptive minus base changed counts by `+7` safe, `+2` sensitive, `+12` rescued, and `-21` both-wrong. These are not deterministic row-level causal transitions: the checkpoints differ, and TP=4 stochastic generation is not bitwise reproducible across separate runs even with persisted per-row seeds.

Conditional rates are more informative for routing:

| Model | Safe among normal-correct | Sensitive among normal-correct | Rescued among normal-wrong |
|---|---:|---:|---:|
| Base | 17/46 (37.0%) | 29/46 (63.0%) | 6/82 (7.3%) |
| Adaptive step 50 | 24/55 (43.6%) | 31/55 (56.4%) | 18/73 (24.7%) |

Thus, a normal-correct result alone is weak evidence of safe compression at a 50% cap.

## Compression-sensitive counterfactuals

| Model | Sensitive total | Cap-hit, recovered | Cap-hit, unrecovered | Natural-stop wrong | Recovery among cap-hit |
|---|---:|---:|---:|---:|---:|
| Base | 29 | 15 | 5 | 9 | 15/20 (75.0%) |
| Adaptive step 50 | 31 | 12 | 9 | 10 | 12/21 (57.1%) |

As a fraction of all sensitive events, forced-prefix continuation recovered 51.7% for base and 38.7% for adaptive step 50. The adaptive continuations were also substantially longer on average (1,087 versus 230 generated suffix tokens), hit the relaxed total cap more often (6/21 versus 1/20), and still had more parse failures (4 versus 1).

These results separate three mechanisms:

1. **Budget-limited:** the capped response is unfinished and continuation from that exact prefix reaches the correct answer.
2. **Trajectory/termination failure:** additional budget does not recover the response, sometimes because it loops or changes a correct intermediate conclusion.
3. **Prompt/sampling failure:** the concise response stops naturally before its cap with a wrong answer, so the cap is not the cause.

Concrete examples include:

- Base question 28836 stopped mid-calculation at 609 tokens and recovered `6\sqrt{5}` after exact-prefix continuation to 965 tokens: a clear budget-limited event.
- Adaptive question 28836 reached the relaxed total cap at 3,614 tokens and truncated inside a final box after repeatedly deriving the correct result: a strong termination/repetition failure.
- Adaptive question 37913 derived `-6` but changed to `+6` after extended contradictory reasoning: extra scratch space did not help.
- Question 39902 naturally stopped with the wrong answer `0` under the concise prompt for both checkpoints because of an asymptotic exponent error. This is prompt/trajectory failure, not budget pressure.
- Question 11469 was unrecovered for base (`-2\pi`) but recovered for adaptive (`-\pi`), illustrating why one stochastic event is not an intrinsic question label.

## Length and termination

| Model | Normal mean / median | Concise mean / median | Mean concise/normal ratio | Normal cap hits | Concise cap hits |
|---|---:|---:|---:|---:|---:|
| Base | 1,594.2 / 1,365.5 | 496.0 / 410.0 | 30.8% | 0 | 36 |
| Adaptive step 50 | 5,885.4 / 3,787.5 | 1,217.3 / 461.5 | 25.7% | 10 | 36 |

Adaptive normal-wrong trajectories averaged 6,530.9 tokens, compared with 5,028.6 for normal-correct trajectories. This reproduces the previously observed failure-tail pattern on training-distribution questions: wrong trajectories are longer and less likely to terminate. The concise prompt shortens responses strongly, but adaptive concise accuracy remains 10.2 percentage points below adaptive normal accuracy.

The base-versus-adaptive length difference includes all effects of OPD training and is not a matched adaptive-versus-Vanilla treatment estimate. The completed AIME24/AIME25 comparison against matched Vanilla OPD remains the correct evidence that the current adaptive objective worsened held-out failure-tail termination.

## Concise-rescued audit

The raw `normal wrong / concise correct` quadrant is real but heterogeneous.

- Base: 6 events. Manual inspection found one answer-granularity/format event and five substantive changed answers.
- Adaptive: 18 events. Manual inspection found approximately ten strict-reward formatting or answer-granularity events, four normal responses that hit 16,384 without a boxed answer, and four substantive changed answers.

Examples:

- Question 2125: the normal response repeatedly reversed itself and ended at `1`; a nine-token concise response gave the rewarded answer `0`.
- Adaptive questions 7294, 29220, 5361, and 10797 hit the normal 16,384-token cap without a parseable answer but produced a rewarded concise answer.
- Several adaptive events were operational reward artifacts: `\text{Yes}` versus ground truth `Yes`, `J(R)=pR` versus `pR`, or a correct inequality instead of the expected selected expression.
- The concise response for question 7294 received the correct final-answer reward but contained an invalid claim that a countable union of finite sets is finite. Final-answer verification alone is therefore insufficient to certify a concise rollout as a high-quality actor target.

Production routing currently labels every normal-wrong event hard without probing it. The probe shows that this merges true hard rows, nonterminating normal rows, answer-format mismatches, and genuinely concise-rescued rows.

## Implications for the next objective

The evidence supports the following redesign, but no new training run has been launched:

1. **Compression-safe (`normal ✓ / concise ✓`):** this is the only event-level candidate for lower weight or concise self-training. If the concise rollout becomes the actor target, include its final answer and EOS and add a reasoning-quality check beyond final-answer reward.
2. **Compression-sensitive, cap-recovered:** preserve the full normal trajectory and endpoint under the strong teacher. Do not exchange away its suffix; the exact-prefix result indicates that additional sequential computation was useful.
3. **Compression-sensitive, cap-unrecovered:** also avoid treating it as compressible. These rows need strategy correction, termination supervision, or a verified teacher target rather than merely more concise-conditioned KL on the same prefix.
4. **Compression-sensitive, natural-stop wrong:** the concise prompt itself changed the trajectory or answer. Concise-conditioned teacher scoring should not be assumed beneficial for these rows.
5. **Concise-rescued:** do not route directly from the raw reward bit. Normalize equivalent answer formats and, ideally, require repeated or teacher-verified evidence before using the concise rollout as supervision.
6. **Both wrong:** retain a separate hard route.

The probe does not contain route-specific entropy or token log-prob telemetry. It therefore does not establish the hypothesis that compression-sensitive trajectories have low per-token entropy while requiring more sequential steps.

## Limitations

- The sample has 128 questions and one stochastic response per prompt/checkpoint.
- Quadrants are event-level observations, not intrinsic question classes.
- Per-row seeds are persisted, but TP=4 dynamic batching does not provide bitwise replay across independent invocations.
- Forced-prefix continuation conditions on the exact observed capped prefix but initializes a new continuation sampler; it does not resume hidden RNG state.
- The training-time reward is intentionally reused and is sensitive to answer formatting and granularity.
- The original base versus adaptive comparison is not a matched Vanilla OPD ablation and does not replace held-out AIME evaluation.
- No entropy measurement was collected.

## Artifact hashes

Key hashes:

- Manifest: `374f7f4ebacacfb8a5cfe2c842f0ba1e68c571ad93b37f77b98c49a0b46e2cf0`
- Base records: `04698153fca24aff5c9d8ea28bff02f341d0204b9a95de50c1a93ff6eaa8c492`
- Adaptive records: `131434a4b63cc82d9742d99f057fce282a2286110e7349e9a053641cd15f0016`
- Comparison: `30b916d099dba8ded6e33a51f50251e091843d4f8d41341e3f6a259091365a4b`
- Formal log: `90a7a232fc8982b431df8bee46db8ff3832db2d07ac7c9cde3ea27060a8313bd`
- Prefix-root-cause diagnostic report: `a29e475d86f43e520567825467771d1b3677f00a36fa0dd29ae51ba0172f2ad2`
- Forced-prefix GPU profile report: `d05cd50cac9421669c2aaea31c2405fc5d307508ed6fef59e77e35e0d4a9a2a4`

The two failed formal namespaces and their logs remain preserved and were not resumed or overwritten.
