# Task 6 Report

## Status
Implemented Task 6.

## Commits created
- `2a7d343` — `Add Rethinking OPD probe run scripts`

## Files added
- `run_train_raw_opd_rethinking_probe.sh`
- `run_train_tale_budget_rolloutlen_hardtrunc_rethinking_probe_opd.sh`
- `run_train_tale_budget_rolloutlen_hardtrunc_conciseteacher_rethinking_probe_opd.sh`
- `math_eval/plot_rethinking_opd_probe.py`

## Validation
- `cd /home/mchen/FiRe-OPD && bash -n run_train_raw_opd_rethinking_probe.sh && bash -n run_train_tale_budget_rolloutlen_hardtrunc_rethinking_probe_opd.sh && bash -n run_train_tale_budget_rolloutlen_hardtrunc_conciseteacher_rethinking_probe_opd.sh && /home/mchen/miniconda3/envs/verl/bin/python -m py_compile math_eval/plot_rethinking_opd_probe.py`

## Concerns
- None.

## Reviewer Findings Requiring Fix
- High: raw probe script shells into base raw script that does not forward `"$@"`, so probe Hydra overrides are dropped.
- Medium: summary plotter sorts chunk_start lexicographically as strings instead of numerically.

## Fix report
- Rewrote `run_train_raw_opd_rethinking_probe.sh` to inline the raw base training command so the probe Hydra overrides are part of the final `python -m verl.trainer.main_ppo` invocation.
- Updated `math_eval/plot_rethinking_opd_probe.py` to sort `chunk_start` numerically within each run.
- Checked the budget and concise probe wrappers; their base script already forwards `"$@"`, so no changes were needed there.

## Test output
- `cd /home/mchen/FiRe-OPD && bash -n run_train_raw_opd_rethinking_probe.sh && bash -n run_train_tale_budget_rolloutlen_hardtrunc_rethinking_probe_opd.sh && bash -n run_train_tale_budget_rolloutlen_hardtrunc_conciseteacher_rethinking_probe_opd.sh && python -m py_compile math_eval/plot_rethinking_opd_probe.py` → exit 0
- Synthetic summary check:
  ```
  # Rethinking OPD Training Probe Summary

  - runA chunk 2: last_step=1, overlap=0.2000, student_entropy=2.0000, entropy_gap=0.3000, valid_tokens=6
  - runA chunk 10: last_step=1, overlap=0.1000, student_entropy=1.0000, entropy_gap=0.2000, valid_tokens=5
  - runB chunk 1: last_step=1, overlap=0.3000, student_entropy=3.0000, entropy_gap=0.4000, valid_tokens=7
  ```
