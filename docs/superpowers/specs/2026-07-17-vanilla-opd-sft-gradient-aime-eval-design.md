# Vanilla OPD SFT-Gradient 51,200 AIME Evaluation Design

**Date:** 2026-07-17
**Status:** Approved in conversation; pending written-spec review
**Scope:** Independent step-50 evaluation on AIME 2024 and AIME 2025 only

## Objective

Evaluate the completed strong-to-weak Vanilla OPD candidate trained on the frozen 51,200-row SFT-gradient selection using the same independent AIME protocol previously used for the historical Vanilla OPD checkpoint. Report the new candidate's sample-level accuracy, pass@32, and mean response length, together with direct aggregate deltas from the historical Vanilla OPD results.

This evaluation does not rerun training, rerun the historical baseline, evaluate the other six math benchmarks, or modify the frozen selection artifacts.

## Frozen Candidate

- Experiment: `opd-strong-to-weak-sftgrad51200-rawprompt-4gpu-tp4-refmb4-rollmb4`
- FSDP actor: `/home/mchen/FiRe-OPD/checkpoints/opd-strong-to-weak-sftgrad51200-rawprompt-4gpu-tp4-refmb4-rollmb4/global_step_50/actor`
- Merged Hugging Face target: `/home/mchen/FiRe-OPD/checkpoints/opd-strong-to-weak-sftgrad51200-rawprompt-4gpu-tp4-refmb4-rollmb4/global_step_50_hf`
- Training completion report: `/home/mchen/FiRe-OPD/logs/gradient_diversity/opd-strong-to-weak-sftgrad51200-rawprompt-4gpu-tp4-refmb4-rollmb4.completion.json`
- Source commit: `94b4f85acb9aa38fff2001c4dd102a55e5daa894`
- Selected training parquet SHA-256: `ffdd06af56361bd49b25b0da5039db4ce70c21e443e3b96687f61376b078ea2c`

The completion report must still establish step 50, 51,200 yielded samples, 50 sampler batches, and world size 4 before merge or evaluation begins.

## Evaluation Protocol

The protocol is frozen to the historical strong-to-weak Vanilla OPD raw-prompt evaluation:

| Setting | Value |
|---|---|
| Datasets | `aime24 aime25` |
| Problems per dataset | 30 |
| Samples per problem | 32 |
| Prompt style | `baseline` |
| Extra prompt | disabled (`--no_extra_prompt`) |
| Thinking-template override | disabled |
| Maximum generated tokens | 16,384 |
| Maximum model length | 40,960 |
| Temperature | 1.0 |
| Top-p | 1.0 |
| Seed | 42 |
| Maximum active sequences | 256 |
| GPU topology | all four allocation-owned GPUs, tensor parallel size 4 |
| Dataset scheduling | sequential: AIME 2024, then AIME 2025 |

Frozen benchmark inputs:

- `/home/mchen/FiRe-OPD/data/aime24/test.jsonl`: 30 rows, SHA-256 `a3c49569f3d7125aaf4eea5764bd1b868af534ae3aff6aad0843a7da58fe46b8`
- `/home/mchen/FiRe-OPD/data/aime25/test.jsonl`: 30 rows, SHA-256 `6012af2a112f5e26d91f1b0cc644b5dde6399173b8b648aec2e6b9899877a2db`

## Execution Design

Run only inside the active `opd-CLI` Slurm allocation, without nested `srun`. Immediately before launch, require:

1. the allocation to be active with enough remaining wall time;
2. allocation-owned logical devices `0,1,2,3` to be idle;
3. no stale candidate evaluation or merge process;
4. a clean active worktree at the frozen source commit;
5. the merged target to be absent or complete, never partial;
6. the candidate output, mistakes, and log targets to be absent, preventing silent overwrite or resume.

Reuse the committed normal-prompt evaluator `math_eval/run_eval_math_tale_budget_step50_table2.sh` from the clean worktree. Its executable evaluation path is equivalent to the prior untracked `math_eval/run_eval_math_opd_rawprompt_step50.sh`; only model-specific defaults and output-directory defaults differ. Supply every candidate-specific path and protocol value explicitly, including absolute production checkpoint and output roots. No source-code change is needed for the evaluation itself.

If the Hugging Face model is absent, merge the validated FSDP actor with `python -m verl.model_merger merge --backend fsdp`. An existing complete merge may be reused only after checking its model files and provenance; a partial merge fails closed.

## Immutable Evaluation Artifacts

Use run name:

`opd-strong-to-weak-sftgrad51200-rawprompt-4gpu-tp4-refmb4-rollmb4-step50-baseline-n32-seed42`

Write candidate artifacts under distinct production directories:

- Responses: `/home/mchen/FiRe-OPD/math_eval/opd_sftgrad51200_step_eval_outputs/<dataset>/<run-name>.jsonl`
- Mistakes: `/home/mchen/FiRe-OPD/math_eval/opd_sftgrad51200_step_eval_mistakes/<dataset>/<run-name>.jsonl`
- Logs: `/home/mchen/FiRe-OPD/math_eval/opd_sftgrad51200_step_eval_logs/<dataset>-<run-name>.log`

The historical Vanilla outputs remain read-only and are not regenerated.

## Validation and Failure Handling

For each candidate dataset, require:

- launcher exit status zero;
- no traceback, CUDA OOM, killed process, or engine failure in the log;
- exactly 30 JSONL records with unique problem texts;
- exactly 32 entries in each of `responses`, `pred_answers`, `response_lengths`, and `acc_list` for every record;
- `prompt_style == "baseline"` and `cod_shot == 0` for every record;
- finite aggregate Accuracy, pass@32, and average response length;
- a published mistakes file (which may legitimately be empty) and final metric lines in the dataset log.

After both datasets, require all vLLM and merge processes to drain, hash the generated artifacts, recheck the two benchmark hashes, recheck selection artifact hashes, and verify that the source worktree remains clean.

Any preflight mismatch, existing partial output, process failure, malformed response row, or artifact drift stops the evaluation. A completed AIME 2024 output may be retained if AIME 2025 subsequently fails, but it must not be silently rerun or overwritten; recovery requires an explicit diagnosis and relaunch decision.

## Historical Comparison

Compare against the existing historical Vanilla OPD run with the identical sampling protocol:

| Dataset | Historical Accuracy | Historical pass@32 | Historical mean length |
|---|---:|---:|---:|
| AIME 2024 | 0.5615 | 0.8000 | 9219.6052 |
| AIME 2025 | 0.4802 | 0.8333 | 9318.3917 |

Report candidate values and candidate-minus-historical deltas separately for each dataset. Also report the unweighted two-dataset mean Accuracy and pass@32 as a compact summary, while retaining per-dataset results as primary evidence. No claim of publication-grade causal attribution is made because the historical baseline's full training snapshot was not frozen to the same standard as the new candidate.

## Explicit Non-Goals

- No OlympiadBench, AMC2023, HMMT, MATH500, or MinervaMath evaluation.
- No historical Vanilla rerun.
- No additional seeds or sample counts.
- No training-time `n=8` validation substituted for independent `n=32` evaluation.
- No checkpoint resume, overwrite, or post-training model modification.
- No remote push or Stage-1 proxy-gradient work.
