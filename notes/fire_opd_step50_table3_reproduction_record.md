# FiRe-OPD Table 3 Single-Teacher Step50 Reproduction Record

Generated: 2026-06-15  
Repository: `/home/mchen/FiRe-OPD`

## 1. Purpose

This note records our local reproduction of the FiRe-OPD paper Table 3 single-teacher math setting using a 50-step training checkpoint. The main goal is to preserve the experimental setup, completed evaluation status, numerical results, and interpretation for later paper writing.

The experiment checks whether a locally trained FiRe-OPD single-teacher checkpoint can reproduce the reported Table 3 math benchmark results on:

- AIME 2024
- AIME 2025
- MATH-500
- AMC 2023
- OlympiadBench
- MinervaMATH
- HMMT 2025 Feb
- HMMT 2025 Nov

## 2. Experimental Setup

### Training setting

- Setting: FiRe-OPD Table 3 single-teacher math distillation
- Student: `Qwen3-4B`
- Teacher: `Qwen3-4B-Non-Thinking-RL-Math-Step500`
- Local experiment name: `fire-opd-table3-single-teacher-4b-math`
- Checkpoint used for evaluation: `global_step_50`

Checkpoint directory:

```text
/home/mchen/FiRe-OPD/checkpoints/fire-opd-table3-single-teacher-4b-math/global_step_50
```

Merged HuggingFace checkpoint:

```text
/home/mchen/FiRe-OPD/checkpoints/fire-opd-table3-single-teacher-4b-math/global_step_50_hf
```

The FSDP checkpoint was merged into HuggingFace format before evaluation.

### Evaluation setting

Evaluation script:

```text
/home/mchen/FiRe-OPD/math_eval/run_eval_math_step50_last4gpus.sh
```

Output pattern:

```text
/home/mchen/FiRe-OPD/math_eval/eval_outputs/<dataset>/fire-opd-table3-single-teacher-4b-math-step50.jsonl
```

Sampling/evaluation parameters:

```text
N_SAMPLES=32
MAX_TOKENS=16384
TEMPERATURE=1.0
TOP_P=1.0
MAX_NUM_SEQS=256
SEED=42
--no_extra_prompt
```

Important metric note: the paper reports Avg@8. Our evaluation generated `n=32` samples per problem. The reported local `Avg@8` below is computed post hoc from the first 8 samples for each problem. `Avg@32` and `pass@32` are also retained as auxiliary statistics.

## 3. Completion Status

All eight math benchmarks finished successfully. The number of output rows matches the number of test examples for every dataset.

| Dataset | Output rows | Expected rows | Output file size |
|---|---:|---:|---:|
| AIME24 | 30 | 30 | 32M |
| AIME25 | 30 | 30 | 34M |
| HMMT-Feb | 30 | 30 | 37M |
| HMMT-Nov | 30 | 30 | 35M |
| MATH500 | 500 | 500 | 155M |
| MinervaMATH | 272 | 272 | 105M |
| OlympiadBench | 674 | 674 | 466M |
| AMC2023 | 40 | 40 | 22M |

No evidence of incomplete output was found.

## 4. Results vs FiRe-OPD Table 3

The FiRe-OPD Table 3 single-teacher row reports:

```text
AIME24 61.25, AIME25 55.00, MATH 94.69, AMC 95.15,
Olymp. 48.07, Miner. 68.49, HMMT-Feb 33.75, HMMT-Nov 37.50,
Avg 61.74
```

However, our local dataset identities and results strongly suggest that the printed Table 3 values for `Olymp.` and `Miner.` are swapped. In the comparison below, `Table 3 corrected` uses the suspected corrected mapping:

- OlympiadBench: `68.49`
- MinervaMATH: `48.07`

| Dataset | Rows | Local Avg@8 | Local Avg@32 | Local pass@32 | Table 3 corrected | Delta |
|---|---:|---:|---:|---:|---:|---:|
| AIME24 | 30 | 58.75 | 57.71 | 83.33 | 61.25 | -2.50 |
| AIME25 | 30 | 55.83 | 55.62 | 80.00 | 55.00 | +0.83 |
| MATH500 | 500 | 95.38 | 95.37 | 99.00 | 94.69 | +0.69 |
| AMC2023 | 40 | 95.00 | 94.38 | 97.50 | 95.15 | -0.15 |
| OlympiadBench | 674 | 68.73 | 68.86 | 82.64 | 68.49 | +0.24 |
| MinervaMATH | 272 | 47.47 | 47.90 | 58.82 | 48.07 | -0.60 |
| HMMT-Feb | 30 | 31.67 | 33.23 | 70.00 | 33.75 | -2.08 |
| HMMT-Nov | 30 | 38.33 | 37.29 | 63.33 | 37.50 | +0.83 |

Macro averages:

| Metric | Value |
|---|---:|
| Local macro Avg@8 | 61.40 |
| Local macro Avg@32 | 61.29 |
| Local macro pass@32 | 79.33 |
| Paper Table 3 macro Avg@8 | 61.74 |
| Local Avg@8 - Paper Avg@8 | -0.34 |

## 5. Key Observations

1. Overall reproduction is very close to Table 3. The local macro Avg@8 is `61.40`, compared with the paper's `61.74`, a difference of only `-0.34` points.
2. Six of the eight benchmarks match Table 3 within roughly 2.5 points or less using the printed table values directly.
3. The apparent large discrepancy on MinervaMATH/OlympiadBench disappears if the Table 3 `Olymp.` and `Miner.` columns are swapped:
   - Local OlympiadBench: `68.73`, matching corrected Table 3 `68.49`.
   - Local MinervaMATH: `47.47`, matching corrected Table 3 `48.07`.
4. This suggests a likely column swap or labeling typo in the paper's Table 3 for the single-teacher row, rather than an evaluation failure.
5. The result indicates that our 50-step FiRe-OPD single-teacher run reproduces the reported Table 3 performance well under the corrected benchmark mapping.

## 6. Notes for Future Paper Writing

Potential wording for later use:

> We reproduced the FiRe-OPD single-teacher math setting using a locally trained 50-step checkpoint. Across eight math benchmarks, our reproduction obtains a macro Avg@8 of 61.40, closely matching the reported Table 3 average of 61.74. We observed that the reported OlympiadBench and MinervaMATH entries in Table 3 appear to be swapped; after correcting this likely table-labeling issue, all benchmark-level numbers align closely with the reported results.

Caveats to preserve:

- Our eval generated 32 samples per problem; Avg@8 was computed from the first 8 samples. This is not a separately generated `n=8` run.
- The evaluated checkpoint is `global_step_50`.
- The Table 3 OlympiadBench/MinervaMATH correction should be described carefully as a suspected paper table-labeling issue, not as an official erratum.

## 7. Raw Artifacts

Checkpoint artifacts:

```text
/home/mchen/FiRe-OPD/checkpoints/fire-opd-table3-single-teacher-4b-math/global_step_50
/home/mchen/FiRe-OPD/checkpoints/fire-opd-table3-single-teacher-4b-math/global_step_50_hf
```

Evaluation script:

```text
/home/mchen/FiRe-OPD/math_eval/run_eval_math_step50_last4gpus.sh
```

Evaluation outputs:

```text
/home/mchen/FiRe-OPD/math_eval/eval_outputs/aime24/fire-opd-table3-single-teacher-4b-math-step50.jsonl
/home/mchen/FiRe-OPD/math_eval/eval_outputs/aime25/fire-opd-table3-single-teacher-4b-math-step50.jsonl
/home/mchen/FiRe-OPD/math_eval/eval_outputs/math500/fire-opd-table3-single-teacher-4b-math-step50.jsonl
/home/mchen/FiRe-OPD/math_eval/eval_outputs/amc2023/fire-opd-table3-single-teacher-4b-math-step50.jsonl
/home/mchen/FiRe-OPD/math_eval/eval_outputs/olympiadbench/fire-opd-table3-single-teacher-4b-math-step50.jsonl
/home/mchen/FiRe-OPD/math_eval/eval_outputs/minervamath/fire-opd-table3-single-teacher-4b-math-step50.jsonl
/home/mchen/FiRe-OPD/math_eval/eval_outputs/hmmt25_feb/fire-opd-table3-single-teacher-4b-math-step50.jsonl
/home/mchen/FiRe-OPD/math_eval/eval_outputs/hmmt25_nov/fire-opd-table3-single-teacher-4b-math-step50.jsonl
```

Related context note:

```text
/home/mchen/FiRe-OPD/notes/eval_step50_context_summary.md
```
