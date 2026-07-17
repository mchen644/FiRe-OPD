# Vanilla OPD SFT-Gradient 51,200 Canonical-Eight Evaluation Extension Design

**Date:** 2026-07-17
**Status:** Approved for execution by the user's eight-dataset scope extension

## Objective

Extend the running independent evaluation of `opd-strong-to-weak-sftgrad51200-rawprompt-4gpu-tp4-refmb4-rollmb4` from AIME 2024/2025 to the full canonical eight-dataset math suite. Let the active AIME 2025 run finish first, preserve both AIME outputs, and then evaluate the remaining six datasets without rerunning either AIME dataset.

This extension supersedes only the two-dataset scope limit in `2026-07-17-vanilla-opd-sft-gradient-aime-eval-design.md`; all checkpoint, prompt, sampling, provenance, non-overwrite, and validation contracts remain active.

## Protocol

Use the already-approved protocol unchanged:

- normal/raw prompt (`prompt_style=baseline`, `--no_extra_prompt`, thinking override disabled);
- 32 samples per problem, seed 42;
- temperature 1.0, top-p 1.0;
- maximum generation length 16,384 and model length 40,960;
- maximum active sequences 256;
- all four allocation-owned GPUs `0,1,2,3` in one tensor-parallel vLLM engine;
- sequential dataset execution inside `opd-CLI`, without nested `srun`.

The second phase runs exactly this order:

1. `hmmt25_feb`
2. `hmmt25_nov`
3. `math500`
4. `minervamath`
5. `olympiadbench`
6. `amc2023`

Frozen inputs:

| Dataset | Problems | SHA-256 |
|---|---:|---|
| `hmmt25_feb` | 30 | `58d889b807a87fd189562d9aeaf2bddf342b2e956843ebbbd8259fc93195cc3f` |
| `hmmt25_nov` | 30 | `37be20ee4d01044638f1d790d938138c2ce6f26fc9e60f6987db5c220783f226` |
| `math500` | 500 | `5b126695a919f8fb40edffaf2e748ae08f7fb6ce0721211671c760b4f0b7c5de` |
| `minervamath` | 272 | `ff6e07ac93af4e43885fe7710ded51c8db04543d627326b855d903075d59e872` |
| `olympiadbench` | 674 | `6c3c658145c21dd76f70eef1456dc5de9bbb38342f2ffdc4a803d8e6e3005ecc` |
| `amc2023` | 40 | `b443425b035d98fec3da4de7e347ac43cebcf7b721e96ba8bf9e0120d24dd61d` |

## Phase Boundary and Launch Gates

Do not launch phase two until:

1. the current AIME launcher publishes `AIME51200_EVAL_DONE:0`;
2. both AIME files pass row-level `n=32` validation;
3. all current vLLM processes drain and allocation GPUs become idle;
4. Slurm job 410 remains active with sufficient wall time;
5. the merged Hugging Face checkpoint is complete;
6. every response, mistakes, and log target for the six remaining datasets is absent;
7. the dedicated phase-two launch log and final canonical-eight report are absent;
8. benchmark and selection hashes remain frozen and the executable source tree remains identical to commit `94b4f85acb9aa38fff2001c4dd102a55e5daa894` outside committed `docs/superpowers/` files.

Reuse `math_eval/run_eval_math_tale_budget_step50_table2.sh` with explicit overrides. Set `SKIP_MERGE=1` and `FORCE_MERGE=0`; the second phase must not recreate or alter the merged model.

## Artifacts

Keep the same run name and artifact roots as the AIME phase:

- Run name: `opd-strong-to-weak-sftgrad51200-rawprompt-4gpu-tp4-refmb4-rollmb4-step50-baseline-n32-seed42`
- Responses: `/home/mchen/FiRe-OPD/math_eval/opd_sftgrad51200_step_eval_outputs/<dataset>/<run-name>.jsonl`
- Mistakes: `/home/mchen/FiRe-OPD/math_eval/opd_sftgrad51200_step_eval_mistakes/<dataset>/<run-name>.jsonl`
- Dataset logs: `/home/mchen/FiRe-OPD/math_eval/opd_sftgrad51200_step_eval_logs/<dataset>-<run-name>.log`
- Phase-two launch log: `/home/mchen/FiRe-OPD/math_eval/opd_sftgrad51200_step_eval_remaining6.launch.log`
- Final report: `/home/mchen/FiRe-OPD/math_eval/opd_sftgrad51200_step_eval_outputs/<run-name>-canonical8-evaluation.json`

## Validation and Reporting

For every dataset, require exactly the expected number of unique problems and exactly 32 `responses`, `pred_answers`, `response_lengths`, and `acc_list` values per problem. Recompute Accuracy, pass@32, and mean response length from JSONL and match each dataset log. Reject tracebacks, OOMs, killed processes, malformed outputs, target collisions, and hash drift.

After all eight datasets complete, report per-dataset metrics and unweighted eight-dataset macro Accuracy/pass@32/mean length. Compare against historical Vanilla OPD only for datasets with complete historical outputs (AIME24, AIME25, HMMT February, HMMT November, MATH500, and MinervaMath). Do not rerun the baseline or fabricate comparisons for OlympiadBench and AMC2023.

The final report records hashes for inputs, responses, mistakes, logs, launch logs, merged-model files, completion report, and selection artifacts. Final acceptance also requires process/GPU drain and a clean executable source tree.
