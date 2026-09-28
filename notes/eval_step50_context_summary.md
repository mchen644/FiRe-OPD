# FiRe-OPD Step50 Evaluation Context Summary

Generated: 2026-06-14
Repo: `/home/mchen/FiRe-OPD`

## What happened in this session

1. Confirmed the training checkpoint at step 50 was saved correctly.
2. Stopped the running training job safely.
3. Merged the FSDP checkpoint into HuggingFace format.
4. Created a math evaluation shell script that uses the last four physical GPUs.
5. Started the Table 3 math evaluation from the merged step50 model.
6. Compared completed results against FiRe-OPD paper Table 3.

## Training/checkpoint status

Checkpoint directory:

```text
/home/mchen/FiRe-OPD/checkpoints/fire-opd-table3-single-teacher-4b-math/global_step_50
```

Confirmed files:

- `latest_checkpointed_iteration.txt = 50`
- FSDP actor checkpoint complete:
  - `model_world_size_4_rank_0.pt` ... `rank_3.pt`
  - `optim_world_size_4_rank_0.pt` ... `rank_3.pt`
  - `extra_state_world_size_4_rank_0.pt` ... `rank_3.pt`
  - `data.pt`
  - `actor/huggingface/*`
- Training process was stopped and Ray workers were cleaned up.

Merged HuggingFace model directory:

```text
/home/mchen/FiRe-OPD/checkpoints/fire-opd-table3-single-teacher-4b-math/global_step_50_hf
```

Contains:

- `model-00001-of-00002.safetensors`
- `model-00002-of-00002.safetensors`
- `model.safetensors.index.json`
- tokenizer/config/generation config files

Merge command used:

```bash
cd /home/mchen/FiRe-OPD
export PYTHONPATH=/home/mchen/FiRe-OPD/verl:${PYTHONPATH:-}
/home/mchen/miniconda3/envs/verl/bin/python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /home/mchen/FiRe-OPD/checkpoints/fire-opd-table3-single-teacher-4b-math/global_step_50/actor \
  --target_dir /home/mchen/FiRe-OPD/checkpoints/fire-opd-table3-single-teacher-4b-math/global_step_50_hf
```

## Evaluation script created

Script path:

```text
/home/mchen/FiRe-OPD/math_eval/run_eval_math_step50_last4gpus.sh
```

Purpose: run all Table 3 math benchmarks using merged step50 HF model and physical GPUs `6,7,8,9`.

Default model:

```text
/home/mchen/FiRe-OPD/checkpoints/fire-opd-table3-single-teacher-4b-math/global_step_50_hf
```

Default settings:

```text
N_SAMPLES=32
MAX_TOKENS=16384
TEMPERATURE=1.0
TOP_P=1.0
MAX_NUM_SEQS=256
SEED=42
--no_extra_prompt
```

GPU scheduling:

- Pair A: `6,7`
- Pair B: `8,9`
- Runs two datasets at a time and waits for both before starting next pair.

Output file pattern:

```text
/home/mchen/FiRe-OPD/math_eval/eval_outputs/<dataset>/fire-opd-table3-single-teacher-4b-math-step50.jsonl
```

## Current eval status at last check

Running processes still present:

```text
bash math_eval/run_eval_math_step50_last4gpus.sh
python eval_math.py --input_file /home/mchen/FiRe-OPD/data/olympiadbench/test.jsonl ...
```

Completed datasets: 7/8

- `aime24`
- `aime25`
- `hmmt25_feb`
- `hmmt25_nov`
- `math500`
- `minervamath`
- `amc2023`

Currently running:

- `olympiadbench`

Latest file completion snapshot:

```text
aime24         lines=30  expected=30
aime25         lines=30  expected=30
hmmt25_feb     lines=30  expected=30
hmmt25_nov     lines=30  expected=30
math500        lines=500 expected=500
minervamath    lines=272 expected=272
olympiadbench  no output yet expected=674
amc2023        lines=40  expected=40
```

## Results so far vs FiRe-OPD paper Table 3

Paper numbers are Table 3, Single-Teacher FiRe-OPD (Ours), `Avg@8`.

Important: our eval script generated `n=32`. The `Avg@8 proxy` below is computed by taking the first 8 samples from each problem in the `n=32` output. `Avg@32` is the full per-sample mean over all 32 samples.

| Dataset | Table 3 Avg@8 | Ours Avg@8 proxy | Delta | Ours Avg@32 | Ours pass@32 |
|---|---:|---:|---:|---:|---:|
| AIME24 | 61.25 | 58.75 | -2.50 | 57.71 | 83.33 |
| AIME25 | 55.00 | 55.83 | +0.83 | 55.62 | 80.00 |
| HMMT-Feb | 33.75 | 31.67 | -2.08 | 33.23 | 70.00 |
| HMMT-Nov | 37.50 | 38.33 | +0.83 | 37.29 | 63.33 |
| MATH500 | 94.69 | 95.38 | +0.69 | 95.37 | 99.00 |
| MinervaMATH | 68.49 | 47.47 | -21.02 | 47.90 | 58.82 |
| AMC2023 | 95.15 | 95.00 | -0.15 | 94.38 | 97.50 |
| OlympiadBench | 48.07 | pending | pending | pending | pending |

Observation:

- Most completed datasets are close to Table 3.
- `MinervaMATH` is a major anomaly: about `-21` Avg@8 points vs Table 3.
- Minerva output is complete (`272/272`), so it is not an incomplete-output issue.
- Later work should investigate Minerva eval/data/answer parsing/prompt differences.

## Paper/training-step clarification

G-OPD / “Learning beyond Teacher” Appendix B says:

- same-size teacher-student G-OPD experiments use 50 optimization steps
- strong-to-weak uses 100 optimization steps

FiRe-OPD paper says:

- training uses 3 epochs, about 165 steps total

Local train data:

```text
/home/mchen/FiRe-OPD/data/g-opd/DeepMath-103K/train_filtered_level6.parquet
rows = 57046
batch_size = 1024
drop_last=True -> 55 steps/epoch
3 epochs -> 165 steps
```

Interpretation:

- Current `global_step_50` aligns with G-OPD same-size 50-step protocol.
- It may not be the final FiRe-OPD Table 3 checkpoint if Table 3 used the paper-stated 3 epochs / 165 steps.
- Still, step50 results are already close on most datasets except MinervaMATH.

## G-OPD/ExOPD code status in this repo

Active scripts found:

```text
verl/examples/fire_opd/run_opd_baseline.sh
verl/examples/fire_opd/run_fire_opd_single_teacher.sh
verl/examples/fire_opd/run_fire_opd_table3_single_teacher_4b_math.sh
verl/examples/fire_opd/run_fire_opd_multi_teacher.sh
```

Current repo can run:

- OPD baseline
- FiRe-OPD single-teacher
- FiRe-OPD multi-teacher

But active repo does **not** appear to include full official G-OPD/ExOPD training support:

- No active `verl/examples/g_opd/run_qwen3-4b-g-opd.sh` found locally.
- Active `verl/verl/workers/config/actor.py` has:
  - `only_reverse_kl_advantages`
  - `multi_teacher_distill`
  - `entropy_aware_distill`
  - `traj_skip_percentile`
  - `entropy_alpha`
  - `entropy_beta`
- Active config does not have `lambda_vals`.
- A backup file contains old G-OPD/ExOPD logic:
  - `verl/verl/workers/actor/dp_actor.py.bak_0409`
  - It has comments like “Original OPD/G-OPD path” and `lambda_vals` logic.

Conclusion:

- Current repo can run OPD baseline and FiRe-OPD.
- To run official G-OPD/ExOPD, likely need to port scripts and `lambda_vals` active implementation from the G-OPD repo or restore/adapt backup code.

## Useful commands for next session

Check eval processes:

```bash
pgrep -af -u mchen 'run_eval_math_step50_last4gpus|eval_math.py|global_step_50_hf|VllmWorker' || true
```

Check GPU usage:

```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
```

Check output completion:

```bash
for ds in aime24 aime25 hmmt25_feb hmmt25_nov math500 minervamath olympiadbench amc2023; do
  p=/home/mchen/FiRe-OPD/math_eval/eval_outputs/$ds/fire-opd-table3-single-teacher-4b-math-step50.jsonl
  exp=$(wc -l < /home/mchen/FiRe-OPD/data/$ds/test.jsonl 2>/dev/null || echo '?')
  if [ -f "$p" ]; then
    printf '%-14s lines=%s expected=%s size=%s mtime=%s\n' "$ds" "$(wc -l < "$p")" "$exp" "$(du -h "$p" | cut -f1)" "$(stat -c '%y' "$p")"
  else
    printf '%-14s no output yet expected=%s\n' "$ds" "$exp"
  fi
done
```

Recompute metrics:

```bash
python - <<'PY'
import json, os
base='/home/mchen/FiRe-OPD/math_eval/eval_outputs'
model='fire-opd-table3-single-teacher-4b-math-step50.jsonl'
expected={'aime24':30,'aime25':30,'hmmt25_feb':30,'hmmt25_nov':30,'math500':500,'minervamath':272,'olympiadbench':674,'amc2023':40}
paper={'aime24':61.25,'aime25':55.00,'math500':94.69,'amc2023':95.15,'olympiadbench':48.07,'minervamath':68.49,'hmmt25_feb':33.75,'hmmt25_nov':37.50}
for ds in ['aime24','aime25','hmmt25_feb','hmmt25_nov','math500','minervamath','olympiadbench','amc2023']:
 p=os.path.join(base,ds,model)
 if not os.path.exists(p):
  print(f'{ds}: no output')
  continue
 rows=[]
 for line in open(p):
  try: rows.append(json.loads(line))
  except Exception: pass
 total32=sum(len(r.get('acc_list',[])) for r in rows)
 corr32=sum(sum(bool(x) for x in r.get('acc_list',[])) for r in rows)
 total8=sum(min(8,len(r.get('acc_list',[]))) for r in rows)
 corr8=sum(sum(bool(x) for x in r.get('acc_list',[])[:8]) for r in rows)
 pass32=sum(any(r.get('acc_list',[])) for r in rows)
 avg8=corr8/total8*100 if total8 else float('nan')
 avg32=corr32/total32*100 if total32 else float('nan')
 p32=pass32/len(rows)*100 if rows else float('nan')
 print(f'{ds}: rows={len(rows)}/{expected[ds]}, avg8={avg8:.2f}, avg32={avg32:.2f}, pass32={p32:.2f}, table3={paper[ds]:.2f}, delta_avg8={avg8-paper[ds]:+.2f}')
PY
```

Suggested next-session prompt:

```text
请先读 /home/mchen/FiRe-OPD/notes/eval_step50_context_summary.md，然后继续检查 eval 是否完成，并重点排查 MinervaMATH 为什么比 Table 3 低很多。
```
