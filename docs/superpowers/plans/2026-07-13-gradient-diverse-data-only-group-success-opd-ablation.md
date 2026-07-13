# Gradient-Diverse Data-Only Group-Success OPD Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, verify, launch, and evaluate a 50-step group-success OPD ablation whose only scientific change is replacing the shuffled full DeepMath pool with the pinned 12,800-row gradient-diverse coreset.

**Architecture:** Keep all training and evaluation implementations unchanged. Extend the existing group-success launcher only so its inherited environment is visible in dry-run output, add a standalone fail-closed artifact/launch preflight, and put a thin production wrapper around the existing launcher that pins the contract, proves the baseline and ablation contracts differ only in allowed names/paths, records provenance, and delegates. Reuse the existing eight-dataset evaluation launcher and add a read-only summarizer for pass@1, pass@32, and response length.

**Tech Stack:** Bash, Python 3.10, PyArrow, Hugging Face Transformers, pytest, tmux, Slurm, veRL, vLLM.

## Global Constraints

- Use `superpowers:using-git-worktrees` before implementation; make code changes in an isolated worktree, then integrate only the ablation commits into `/home/mchen/FiRe-OPD` without altering unrelated dirty files.
- Launch production only from `/home/mchen/FiRe-OPD` in the existing `opd-CLI` Slurm shell, never from the implementation worktree or login shell.
- Pin the selected parquet to `/home/mchen/FiRe-OPD/data/gradient_diversity/DeepMath-103K/train_gradient_diverse_12800.parquet` with SHA-256 `caf303c5d151fdaed2e21eebc257917f13660c589906e3da0e569ffdfd59b059` and exactly 12,800 rows.
- Pin the source parquet to `/home/mchen/FiRe-OPD/data/g-opd/DeepMath-103K/train_filtered_level6.parquet` with SHA-256 `de3350fdd00bc0410550098ea65179e2be873da99e4075f80de575fc17670597` and exactly 57,046 rows.
- Pin the selection manifest and selected-ID list to `data/gradient_diversity/selection/manifest.json` and `data/gradient_diversity/selection/selected_ids.jsonl`; require `eligible_row_count=57045` and `selected_row_count=12800`.
- Pin the student to `models/Qwen3-4B` and the teacher/ref to `models/Qwen3-30B-A3B-Instruct-2507`.
- Keep 256 unique prompts per step, four rollouts per prompt, 1,024 trajectories per step, prompt-level PPO mini-batch 256, 50 optimizer steps, shuffle seed 42, and checkpoint frequency 20.
- Keep easy routing at exactly 4/4 correct with concise teacher plus ESR 0.20; keep every non-easy group on normal teacher plus ESR 0.50.
- Keep reverse-KL OPD only: entropy coefficient 0, KL reward/loss coefficient 0, candidate selection disabled, length-aware loss disabled, and rethinking probe disabled.
- Allowed training-command differences are only `TRAIN_DATA`, experiment name, checkpoint path, log path, and emitted provenance.
- Use production experiment name `opd-n4-graddiv12800-easy4of4-concise20-noneasynormal50-purerkl-step50` and never resume an existing checkpoint.
- Do not modify `verl/verl/trainer/ppo/difficulty_aware_opd.py`, TALE/ESR logic, actor loss code, `math_eval/eval_math.py`, or benchmark inputs for this ablation.
- Evaluate exactly AIME 2024, AIME 2025, HMMT February 2025, HMMT November 2025, MATH500, MinervaMath, OlympiadBench, and AMC 2023 with `n=32`, temperature 1.0, top-p 1.0, maximum tokens 16,384, seed 42, and no extra prompt.
- No production code is written before its corresponding focused test has failed for the expected reason.
- Every implementation commit contains only files named in its task.

---

## File Map

- Modify `run_train_group_success_difficulty_opd.sh`: expose every inherited scientific path/value needed to audit a dry run; do not change training defaults or Hydra overrides.
- Modify `verl/tests/trainer/ppo/test_group_success_launcher.py`: lock the expanded dry-run contract and environment override behavior.
- Create `math_eval/validate_gradient_diverse_training_data.py`: validate the selected artifact, source-row identity, prompt shape/uniqueness/token lengths, manifest/ID provenance, and empty checkpoint target.
- Create `math_eval/test_validate_gradient_diverse_training_data.py`: test the validator entirely with temporary PyArrow artifacts and a fake tokenizer.
- Create `run_train_group_success_gradient_diverse_ablation.sh`: pin the experiment, compare baseline/ablation dry contracts, run preflight, record provenance, and delegate to the existing launcher.
- Create `verl/tests/trainer/ppo/test_group_success_gradient_diverse_ablation_launcher.py`: test dry-run output, pin rejection, delegation, and the one-variable comparison without loading a model.
- Create `math_eval/summarize_math_eval_suite.py`: read existing eval JSONL files and compute strict per-dataset/macro pass@1, pass@32, mean length, median length, and candidate deltas.
- Create `math_eval/test_summarize_math_eval_suite.py`: lock metric definitions, sample-count validation, dataset completeness, and delta direction.
- Reuse `math_eval/run_eval_math_opd_step50.sh` unchanged for checkpoint merge and all eight evaluations.

---

### Task 1: Make the Existing Group-Success Dry Run Fully Auditable

**Files:**
- Modify: `run_train_group_success_difficulty_opd.sh:5-55,96-106`
- Modify: `verl/tests/trainer/ppo/test_group_success_launcher.py`

**Interfaces:**
- Consumes: current environment overrides accepted by the base OPD launcher.
- Produces: newline-delimited `KEY=value` fields followed by one shell-escaped `bash <base-launcher> ...` line when `GROUP_SUCCESS_DRY_RUN=1`.
- Preserves: the non-dry execution command and all Hydra arguments byte-for-byte apart from exporting resolved default environment variables.

- [ ] **Step 1: Add failing tests for inherited path/model fields**

Add these tests to `verl/tests/trainer/ppo/test_group_success_launcher.py`:

```python
def test_group_success_launcher_dry_run_prints_inherited_scientific_environment():
    completed = _run_launcher(
        TRAIN_DATA="/tmp/selected.parquet",
        CHECKPOINT_DIR="/tmp/checkpoints/run",
        STUDENT_MODEL="/tmp/models/student",
        TEACHER_MODEL="/tmp/models/teacher",
        MAX_PROMPT_LENGTH="2048",
        TEACHER_PROMPT_KEY="teacher_prompt",
    )

    assert completed.returncode == 0, completed.stderr
    assert "TRAIN_DATA=/tmp/selected.parquet\n" in completed.stdout
    assert "CHECKPOINT_DIR=/tmp/checkpoints/run\n" in completed.stdout
    assert "STUDENT_MODEL=/tmp/models/student\n" in completed.stdout
    assert "TEACHER_MODEL=/tmp/models/teacher\n" in completed.stdout
    assert "MAX_PROMPT_LENGTH=2048\n" in completed.stdout
    assert "TEACHER_PROMPT_KEY=teacher_prompt\n" in completed.stdout
    assert "N_GPUS_PER_NODE=4\n" in completed.stdout
    assert "ROLLOUT_TP_SIZE=4\n" in completed.stdout


def test_group_success_launcher_dry_run_reports_validation_data_without_changing_train_data():
    completed = _run_launcher(
        DATA_ROOT="/tmp/g-opd",
        TRAIN_DATA="/tmp/coreset.parquet",
    )

    assert completed.returncode == 0, completed.stderr
    assert "TRAIN_DATA=/tmp/coreset.parquet\n" in completed.stdout
    assert (
        "VAL_DATA=['/tmp/g-opd/AIME2024/test.parquet', "
        "'/tmp/g-opd/AIME2025/test.parquet']\n"
    ) in completed.stdout
```

- [ ] **Step 2: Run the two new tests and verify RED**

Run:

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  verl/tests/trainer/ppo/test_group_success_launcher.py \
  -k 'inherited_scientific_environment or validation_data'
```

Expected: both tests fail because the current dry run omits `TRAIN_DATA`, `CHECKPOINT_DIR`, model paths, and validation data.

- [ ] **Step 3: Resolve and export inherited defaults in the group launcher**

Immediately after `TRAIN_DATA` is resolved, add exactly these behavior-preserving defaults:

```bash
export VAL_DATA="${VAL_DATA:-['${DATA_ROOT}/AIME2024/test.parquet', '${DATA_ROOT}/AIME2025/test.parquet']}"
export STUDENT_MODEL="${STUDENT_MODEL:-${REPO_DIR}/models/Qwen3-4B}"
export TEACHER_MODEL="${TEACHER_MODEL:-${REPO_DIR}/models/Qwen3-30B-A3B-Instruct-2507}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
export TEACHER_PROMPT_KEY="${TEACHER_PROMPT_KEY:-teacher_prompt}"
```

Keep the existing defaults and exports for `TRAIN_DATA`, `N_GPUS_PER_NODE`, `ROLLOUT_TP_SIZE`, `EXPERIMENT_NAME`, and `CHECKPOINT_DIR`. Add these lines before the dry-run command line:

```bash
printf 'DATA_ROOT=%s\n' "${DATA_ROOT}"
printf 'TRAIN_DATA=%s\n' "${TRAIN_DATA}"
printf 'VAL_DATA=%s\n' "${VAL_DATA}"
printf 'STUDENT_MODEL=%s\n' "${STUDENT_MODEL}"
printf 'TEACHER_MODEL=%s\n' "${TEACHER_MODEL}"
printf 'MAX_PROMPT_LENGTH=%s\n' "${MAX_PROMPT_LENGTH}"
printf 'TEACHER_PROMPT_KEY=%s\n' "${TEACHER_PROMPT_KEY}"
printf 'N_GPUS_PER_NODE=%s\n' "${N_GPUS_PER_NODE}"
printf 'ROLLOUT_TP_SIZE=%s\n' "${ROLLOUT_TP_SIZE}"
printf 'CHECKPOINT_DIR=%s\n' "${CHECKPOINT_DIR}"
```

Do not reorder or edit `fixed_args`.

- [ ] **Step 4: Run all launcher tests and verify GREEN**

Run:

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  verl/tests/trainer/ppo/test_group_success_launcher.py
bash -n run_train_group_success_difficulty_opd.sh
```

Expected: all launcher tests pass and `bash -n` exits 0.

- [ ] **Step 5: Prove the non-dry Hydra suffix did not change**

Run:

```bash
git diff -- run_train_group_success_difficulty_opd.sh
```

Expected: the diff contains only five resolved exports and ten dry-run `printf` lines; `fixed_args` and the final `exec bash` line are unchanged.

- [ ] **Step 6: Commit Task 1**

```bash
git add run_train_group_success_difficulty_opd.sh \
  verl/tests/trainer/ppo/test_group_success_launcher.py
git commit -m "Expose group-success dry-run contract"
```

---

### Task 2: Add a Fail-Closed Gradient-Diverse Training Artifact Validator

**Files:**
- Create: `math_eval/validate_gradient_diverse_training_data.py`
- Create: `math_eval/test_validate_gradient_diverse_training_data.py`

**Interfaces:**
- Consumes:

```python
validate_training_artifact(
    selected_parquet: Path,
    source_parquet: Path,
    selection_manifest: Path,
    selected_ids_path: Path,
    tokenizer: object,
    *,
    expected_selected_sha256: str,
    expected_source_sha256: str,
    expected_rows: int,
    expected_source_rows: int,
    expected_eligible_rows: int,
    max_prompt_tokens: int,
) -> dict[str, object]
```

- Produces: a JSON-serializable report containing artifact hashes/counts, schema/source equality, prompt uniqueness, and prompt-token statistics.
- Produces: `ensure_empty_checkpoint_dir(checkpoint_dir: Path) -> None`, which accepts a missing or empty directory and raises `ValueError` for a non-empty directory.
- CLI: `python -m math_eval.validate_gradient_diverse_training_data` with explicit artifact, tokenizer, expected hash/count, token-limit, and checkpoint arguments.

- [ ] **Step 1: Write the temporary artifact fixture and happy-path test**

Create `math_eval/test_validate_gradient_diverse_training_data.py` with this fixture shape and test:

```python
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from math_eval import validate_gradient_diverse_training_data as validation


SCHEMA = pa.schema(
    [
        pa.field("data_source", pa.string()),
        pa.field(
            "prompt",
            pa.list_(
                pa.struct(
                    [pa.field("content", pa.string()), pa.field("role", pa.string())]
                )
            ),
        ),
        pa.field("ability", pa.string()),
        pa.field(
            "reward_model",
            pa.struct(
                [pa.field("ground_truth", pa.string()), pa.field("style", pa.string())]
            ),
        ),
        pa.field(
            "extra_info",
            pa.struct([pa.field("index", pa.int64()), pa.field("split", pa.string())]),
        ),
    ],
    metadata={b"fixture": b"gradient-diverse-ablation"},
)


def _row(index: int, token_count: int | None = None) -> dict:
    count = token_count if token_count is not None else index + 3
    return {
        "data_source": "DeepMath-103K",
        "prompt": [{"role": "user", "content": f"tokens={count}; problem={index}"}],
        "ability": "math",
        "reward_model": {"ground_truth": str(index), "style": "rule"},
        "extra_info": {"index": index, "split": "train"},
    }


class FakeTokenizer:
    def __init__(self):
        self.calls: list[dict[str, object]] = []

    def apply_chat_template(self, conversations, **kwargs):
        self.calls.append(dict(kwargs))
        return [
            list(range(int(chat[0]["content"].split(";", 1)[0].split("=", 1)[1])))
            for chat in conversations
        ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifacts(tmp_path: Path, source_rows: list[dict], selected_indices: list[int]):
    source_path = tmp_path / "source.parquet"
    selected_path = tmp_path / "selected.parquet"
    ids_path = tmp_path / "selected_ids.jsonl"
    manifest_path = tmp_path / "manifest.json"
    source = pa.Table.from_pylist(source_rows, schema=SCHEMA)
    selected = source.take(pa.array(selected_indices, type=pa.int64()))
    pq.write_table(source, source_path)
    pq.write_table(selected, selected_path)
    id_rows = [
        {
            "eligible_position": source_index,
            "id": f"deepmath-level6-{source_index:06d}",
            "original_dataset_index": source_index + 100,
            "source_row_index": source_index,
        }
        for source_index in selected_indices
    ]
    ids_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in id_rows),
        encoding="utf-8",
    )
    id_sequence = "".join(row["id"] + "\n" for row in id_rows).encode("utf-8")
    manifest = {
        "manifest_version": 1,
        "source_parquet": str(source_path.resolve()),
        "source_sha256": _sha256(source_path),
        "source_row_count": len(source_rows),
        "eligible_row_count": len(source_rows),
        "selected_row_count": len(selected_indices),
        "output_parquet": str(selected_path.resolve()),
        "output_parquet_sha256": _sha256(selected_path),
        "selected_ids": str(ids_path.resolve()),
        "selected_ids_sha256": _sha256(ids_path),
        "selected_id_sequence_sha256": hashlib.sha256(id_sequence).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return source_path, selected_path, ids_path, manifest_path, manifest


def test_validate_training_artifact_accepts_exact_source_subset(tmp_path: Path):
    source, selected, ids, manifest_path, manifest = _artifacts(
        tmp_path, [_row(0), _row(1), _row(2)], [2, 0]
    )
    tokenizer = FakeTokenizer()

    report = validation.validate_training_artifact(
        selected,
        source,
        manifest_path,
        ids,
        tokenizer,
        expected_selected_sha256=manifest["output_parquet_sha256"],
        expected_source_sha256=manifest["source_sha256"],
        expected_rows=2,
        expected_source_rows=3,
        expected_eligible_rows=3,
        max_prompt_tokens=10,
    )

    assert report["selected_rows"] == 2
    assert report["unique_prompts"] == 2
    assert report["max_prompt_tokens"] == 5
    assert report["prompts_over_limit"] == 0
    assert report["schema_equal"] is True
    assert report["source_rows_equal"] is True
    assert tokenizer.calls == [
        {
            "tokenize": True,
            "add_generation_prompt": True,
            "padding": False,
            "truncation": False,
            "enable_thinking": False,
        }
    ]
```

- [ ] **Step 2: Run the happy-path test and verify RED**

Run:

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  math_eval/test_validate_gradient_diverse_training_data.py \
  -k exact_source_subset
```

Expected: collection fails because `math_eval.validate_gradient_diverse_training_data` does not exist.

- [ ] **Step 3: Implement strict JSON, hash, ID, schema, row, and prompt validation**

Create `math_eval/validate_gradient_diverse_training_data.py` with these constants and validation flow:

```python
REQUIRED_COLUMNS = ("data_source", "prompt", "ability", "reward_model", "extra_info")
TOKENIZATION_BATCH_SIZE = 256


def _id_sequence_sha256(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row["id"].encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _serialized_prompt(prompt: list[dict[str, str]]) -> str:
    return json.dumps(
        prompt, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _validate_prompt(prompt: object, row_index: int) -> list[dict[str, str]]:
    if not isinstance(prompt, list) or not prompt:
        raise ValueError(f"row {row_index} prompt must be a nonempty message list")
    normalized = []
    for message_index, message in enumerate(prompt):
        if not isinstance(message, dict):
            raise ValueError(f"row {row_index} message {message_index} must be an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role or not isinstance(content, str) or not content:
            raise ValueError(
                f"row {row_index} message {message_index} requires nonempty role/content strings"
            )
        normalized.append({"role": role, "content": content})
    return normalized
```

Implement `validate_training_artifact` in this exact order so a corrupt input fails before tokenization:

1. Require all four artifact paths to be regular files.
2. Compute selected/source SHA-256 with `math_eval.deepmath_gradient_diversity.sha256_file` and compare to both pinned arguments and manifest fields.
3. Require manifest version 1, exact resolved artifact paths, source/selected row counts, eligible row count, selected-ID hash, and selected-ID sequence hash.
4. Parse every nonempty selected-ID line as one JSON object; require exactly `expected_rows` rows, nonempty unique string IDs, unique integer `source_row_index` values in `[0, expected_source_rows)`, and unique integer `eligible_position` values.
5. Read both parquet tables; require source/selected counts, all required columns, exact schema equality with metadata, and zero top-level nulls in every required column.
6. Build `expected = source.take(pa.array(source_indices, type=pa.int64()))` and require `selected.equals(expected, check_metadata=True)`.
7. Convert selected rows to Python, validate every message list plus the nested `reward_model.ground_truth`, `reward_model.style`, `extra_info.index`, and `extra_info.split` values, serialize prompts with `_serialized_prompt`, and require exactly `expected_rows` unique strings.
8. Tokenize message lists in batches of 256 using exactly:

```python
token_ids = tokenizer.apply_chat_template(
    batch,
    tokenize=True,
    add_generation_prompt=True,
    padding=False,
    truncation=False,
    enable_thinking=False,
)
```

9. Require one token list per prompt and no length above `max_prompt_tokens`; report count, min, median, p95, p99, max, and over-limit count.

Return this stable report shape:

```python
return {
    "selected_parquet": str(selected_parquet.resolve()),
    "selected_sha256": selected_sha256,
    "selected_rows": selected.num_rows,
    "source_parquet": str(source_parquet.resolve()),
    "source_sha256": source_sha256,
    "source_rows": source.num_rows,
    "eligible_rows": manifest["eligible_row_count"],
    "selected_ids": len(id_rows),
    "unique_prompts": len(set(serialized_prompts)),
    "schema_equal": True,
    "source_rows_equal": True,
    "min_prompt_tokens": min(token_lengths),
    "median_prompt_tokens": statistics.median(token_lengths),
    "p95_prompt_tokens": _nearest_rank(token_lengths, 0.95),
    "p99_prompt_tokens": _nearest_rank(token_lengths, 0.99),
    "max_prompt_tokens": max(token_lengths),
    "prompt_token_limit": max_prompt_tokens,
    "prompts_over_limit": 0,
}
```

Use nearest-rank `sorted_lengths[math.ceil(q * len(sorted_lengths)) - 1]` for p95/p99.

- [ ] **Step 4: Run the happy-path test and verify GREEN**

Run the Step 2 command. Expected: one test passes.

- [ ] **Step 5: Add failing corruption and checkpoint-target tests**

Add these explicit cases:

```python
import pytest


def _validate_fixture(paths, manifest, tokenizer, **overrides):
    source, selected, ids, manifest_path = paths
    arguments = {
        "expected_selected_sha256": manifest["output_parquet_sha256"],
        "expected_source_sha256": manifest["source_sha256"],
        "expected_rows": manifest["selected_row_count"],
        "expected_source_rows": manifest["source_row_count"],
        "expected_eligible_rows": manifest["eligible_row_count"],
        "max_prompt_tokens": 10,
    }
    arguments.update(overrides)
    return validation.validate_training_artifact(
        selected, source, manifest_path, ids, tokenizer, **arguments
    )


def test_validator_rejects_pinned_selected_hash_mismatch(tmp_path: Path):
    source, selected, ids, manifest_path, manifest = _artifacts(
        tmp_path, [_row(0), _row(1)], [0]
    )
    with pytest.raises(ValueError, match="selected parquet SHA-256 mismatch"):
        _validate_fixture(
            (source, selected, ids, manifest_path),
            manifest,
            FakeTokenizer(),
            expected_selected_sha256="0" * 64,
        )


def test_validator_rejects_duplicate_exact_prompts(tmp_path: Path):
    duplicate = _row(0)
    second = _row(1)
    second["prompt"] = duplicate["prompt"]
    source, selected, ids, manifest_path, manifest = _artifacts(
        tmp_path, [duplicate, second], [0, 1]
    )
    with pytest.raises(ValueError, match="exact prompts must be unique"):
        _validate_fixture(
            (source, selected, ids, manifest_path), manifest, FakeTokenizer()
        )


def test_validator_rejects_prompt_over_training_limit(tmp_path: Path):
    source, selected, ids, manifest_path, manifest = _artifacts(
        tmp_path, [_row(0, token_count=11)], [0]
    )
    with pytest.raises(ValueError, match="prompt token limit exceeded"):
        _validate_fixture(
            (source, selected, ids, manifest_path), manifest, FakeTokenizer()
        )


def test_validator_rejects_selected_row_not_named_by_source_index(tmp_path: Path):
    source, selected, ids, manifest_path, manifest = _artifacts(
        tmp_path, [_row(0), _row(1)], [0]
    )
    pq.write_table(pa.Table.from_pylist([_row(1)], schema=SCHEMA), selected)
    manifest["output_parquet_sha256"] = _sha256(selected)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="selected parquet does not equal source.take"):
        _validate_fixture(
            (source, selected, ids, manifest_path), manifest, FakeTokenizer()
        )


def test_validator_rejects_manifest_selected_id_hash_mismatch(tmp_path: Path):
    source, selected, ids, manifest_path, manifest = _artifacts(
        tmp_path, [_row(0), _row(1)], [0]
    )
    manifest["selected_ids_sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="selected IDs SHA-256 mismatch"):
        _validate_fixture(
            (source, selected, ids, manifest_path), manifest, FakeTokenizer()
        )


def test_ensure_empty_checkpoint_dir_accepts_missing_and_empty_but_rejects_nonempty(
    tmp_path: Path,
):
    missing = tmp_path / "missing"
    empty = tmp_path / "empty"
    empty.mkdir()
    validation.ensure_empty_checkpoint_dir(missing)
    validation.ensure_empty_checkpoint_dir(empty)
    (empty / "global_step_1").mkdir()
    with pytest.raises(ValueError, match="checkpoint directory is non-empty"):
        validation.ensure_empty_checkpoint_dir(empty)
```

- [ ] **Step 6: Run the new cases and verify RED**

Run:

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  math_eval/test_validate_gradient_diverse_training_data.py
```

Expected: the happy path passes and the new failure/launch-target behaviors fail until implemented.

- [ ] **Step 7: Implement remaining failures and the CLI**

Add `ensure_empty_checkpoint_dir` and an argparse entry point. The CLI first rejects a non-empty checkpoint target, then loads the tokenizer using:

```python
tokenizer = AutoTokenizer.from_pretrained(
    args.tokenizer_path,
    trust_remote_code=True,
)
```

The CLI arguments are:

```text
--selected-parquet
--source-parquet
--selection-manifest
--selected-ids
--tokenizer-path
--expected-selected-sha256
--expected-source-sha256
--expected-rows
--expected-source-rows
--expected-eligible-rows
--max-prompt-tokens
--checkpoint-dir
```

Call `ensure_empty_checkpoint_dir` before printing `json.dumps(report, sort_keys=True)`. Any mismatch exits nonzero through a concise `ValueError` message.

- [ ] **Step 8: Run validator tests, syntax checks, and the real artifact preflight**

Run:

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  math_eval/test_validate_gradient_diverse_training_data.py
/home/mchen/miniconda3/envs/verl/bin/python -m compileall -q \
  math_eval/validate_gradient_diverse_training_data.py
/home/mchen/miniconda3/envs/verl/bin/python -m math_eval.validate_gradient_diverse_training_data \
  --selected-parquet /home/mchen/FiRe-OPD/data/gradient_diversity/DeepMath-103K/train_gradient_diverse_12800.parquet \
  --source-parquet /home/mchen/FiRe-OPD/data/g-opd/DeepMath-103K/train_filtered_level6.parquet \
  --selection-manifest /home/mchen/FiRe-OPD/data/gradient_diversity/selection/manifest.json \
  --selected-ids /home/mchen/FiRe-OPD/data/gradient_diversity/selection/selected_ids.jsonl \
  --tokenizer-path /home/mchen/FiRe-OPD/models/Qwen3-4B \
  --expected-selected-sha256 caf303c5d151fdaed2e21eebc257917f13660c589906e3da0e569ffdfd59b059 \
  --expected-source-sha256 de3350fdd00bc0410550098ea65179e2be873da99e4075f80de575fc17670597 \
  --expected-rows 12800 \
  --expected-source-rows 57046 \
  --expected-eligible-rows 57045 \
  --max-prompt-tokens 2048 \
  --checkpoint-dir /home/mchen/FiRe-OPD/checkpoints/opd-n4-graddiv12800-easy4of4-concise20-noneasynormal50-purerkl-step50
```

Expected: tests pass; compileall is silent; the final command prints JSON with `selected_rows=12800`, `unique_prompts=12800`, `max_prompt_tokens=668`, `prompts_over_limit=0`, `schema_equal=true`, and `source_rows_equal=true`.

- [ ] **Step 9: Commit Task 2**

```bash
git add math_eval/validate_gradient_diverse_training_data.py \
  math_eval/test_validate_gradient_diverse_training_data.py
git commit -m "Validate gradient-diverse OPD training artifact"
```

---

### Task 3: Add the Pinned Data-Only Production Wrapper

**Files:**
- Create: `run_train_group_success_gradient_diverse_ablation.sh`
- Create: `verl/tests/trainer/ppo/test_group_success_gradient_diverse_ablation_launcher.py`

**Interfaces:**
- `ABLATION_DRY_RUN=1`: compare contracts and print the pinned candidate contract/provenance fields without reading the production parquet, tokenizer, checkpoint, or GPUs.
- `ABLATION_PREFLIGHT_ONLY=1`: compare contracts, run the real artifact/checkpoint validator, print provenance, and exit before training.
- Default: perform the same checks, append stdout/stderr to the pinned log, then `exec bash run_train_group_success_difficulty_opd.sh`.
- Reject any environment override that changes a pinned scientific field.

- [ ] **Step 1: Write failing wrapper tests**

Create `verl/tests/trainer/ppo/test_group_success_gradient_diverse_ablation_launcher.py`:

```python
import os
from pathlib import Path
import subprocess


REPO_DIR = Path(__file__).resolve().parents[4]
LAUNCHER = REPO_DIR / "run_train_group_success_gradient_diverse_ablation.sh"
RUN_NAME = "opd-n4-graddiv12800-easy4of4-concise20-noneasynormal50-purerkl-step50"


def _run(**overrides):
    env = os.environ.copy()
    env.update(
        {
            "ABLATION_DRY_RUN": "1",
            "PYTHON_BIN": "/path/not/used/in/dry-run",
            "REPO_DIR": str(REPO_DIR),
            **overrides,
        }
    )
    return subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_DIR,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_ablation_dry_run_is_artifact_independent_and_data_only():
    completed = _run()

    assert completed.returncode == 0, completed.stderr
    assert "artifact_validation=SKIPPED_DRY_RUN\n" in completed.stdout
    assert "single_variable_contract=PASS\n" in completed.stdout
    assert f"EXPERIMENT_NAME={RUN_NAME}\n" in completed.stdout
    assert "TRAIN_DATA=/home/mchen/FiRe-OPD/data/gradient_diversity/DeepMath-103K/train_gradient_diverse_12800.parquet\n" in completed.stdout
    assert "ROLLOUT_N=4\n" in completed.stdout
    assert "PROMPT_BATCH_SIZE=256\n" in completed.stdout
    assert "PPO_MINI_BATCH_SIZE=256\n" in completed.stdout
    assert "TOTAL_TRAJECTORIES=1024\n" in completed.stdout
    assert "trainer.total_training_steps=50" in completed.stdout
    assert "trainer.resume_mode=disable" in completed.stdout


def test_ablation_rejects_changed_scientific_volume():
    completed = _run(ROLLOUT_N="8")

    assert completed.returncode == 2
    assert "ROLLOUT_N must remain pinned to 4 (got 8)" in completed.stderr


def test_ablation_rejects_changed_training_data():
    completed = _run(TRAIN_DATA="/tmp/random.parquet")

    assert completed.returncode == 2
    assert "TRAIN_DATA must remain pinned" in completed.stderr


def test_ablation_delegates_to_existing_group_success_launcher():
    completed = _run()

    assert completed.returncode == 0, completed.stderr
    expected = str(REPO_DIR / "run_train_group_success_difficulty_opd.sh")
    assert f"bash {expected}" in completed.stdout
    assert "algorithm.difficulty_aware_opd.method=group_success_prompt_esr" in completed.stdout
    assert "actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=True" in completed.stdout
```

- [ ] **Step 2: Run wrapper tests and verify RED**

Run:

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  verl/tests/trainer/ppo/test_group_success_gradient_diverse_ablation_launcher.py
```

Expected: collection succeeds but all tests fail because the wrapper file is absent.

- [ ] **Step 3: Implement constants, pin checks, and dry contract generation**

Create `run_train_group_success_gradient_diverse_ablation.sh` beginning with:

```bash
#!/usr/bin/env bash
set -euo pipefail

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

require_pinned_value() {
  local name="$1"
  local actual="$2"
  local expected="$3"
  [[ "$actual" == "$expected" ]] || die "$name must remain pinned to $expected (got $actual)"
}

PRODUCTION_ROOT="/home/mchen/FiRe-OPD"
REPO_DIR="${REPO_DIR:-${PRODUCTION_ROOT}}"
GROUP_LAUNCHER="${REPO_DIR}/run_train_group_success_difficulty_opd.sh"
BASE_LAUNCHER="${REPO_DIR}/verl/examples/fire_opd/run_opd_strong_to_weak_student_raw_teacher_tale_budget30b.sh"
PYTHON_BIN="${PYTHON_BIN:-/home/mchen/miniconda3/envs/verl/bin/python}"

SOURCE_DATA="${PRODUCTION_ROOT}/data/g-opd/DeepMath-103K/train_filtered_level6.parquet"
SELECTED_DATA="${PRODUCTION_ROOT}/data/gradient_diversity/DeepMath-103K/train_gradient_diverse_12800.parquet"
SELECTION_MANIFEST="${PRODUCTION_ROOT}/data/gradient_diversity/selection/manifest.json"
SELECTED_IDS="${PRODUCTION_ROOT}/data/gradient_diversity/selection/selected_ids.jsonl"
SELECTED_SHA256="caf303c5d151fdaed2e21eebc257917f13660c589906e3da0e569ffdfd59b059"
SOURCE_SHA256="de3350fdd00bc0410550098ea65179e2be873da99e4075f80de575fc17670597"

BASELINE_EXPERIMENT="opd-n4-easy4of4-concise20-noneasynormal50-purerkl-step50"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-opd-n4-graddiv12800-easy4of4-concise20-noneasynormal50-purerkl-step50}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PRODUCTION_ROOT}/checkpoints/${EXPERIMENT_NAME}}"
LOG_FILE="${LOG_FILE:-${PRODUCTION_ROOT}/logs/difficulty_routed/${EXPERIMENT_NAME}.log}"
TRAIN_DATA="${TRAIN_DATA:-${SELECTED_DATA}}"
DATA_ROOT="${DATA_ROOT:-${PRODUCTION_ROOT}/data/g-opd}"
STUDENT_MODEL="${STUDENT_MODEL:-${PRODUCTION_ROOT}/models/Qwen3-4B}"
TEACHER_MODEL="${TEACHER_MODEL:-${PRODUCTION_ROOT}/models/Qwen3-30B-A3B-Instruct-2507}"
VAL_DATA="${VAL_DATA:-['${DATA_ROOT}/AIME2024/test.parquet', '${DATA_ROOT}/AIME2025/test.parquet']}"
TEACHER_PROMPT_KEY="${TEACHER_PROMPT_KEY:-teacher_prompt}"
ROLLOUT_N="${ROLLOUT_N:-4}"
PROMPT_BATCH_SIZE="${PROMPT_BATCH_SIZE:-256}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-256}"
TOTAL_TRAJECTORIES="${TOTAL_TRAJECTORIES:-1024}"
EXPECTED_GROUP_SIZE="${EXPECTED_GROUP_SIZE:-4}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-50}"
SAVE_FREQ="${SAVE_FREQ:-20}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-4}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
```

Require the pins with these exact calls:

```bash
require_pinned_value PRODUCTION_ROOT "${PRODUCTION_ROOT}" /home/mchen/FiRe-OPD
if [[ "${ABLATION_DRY_RUN:-0}" != "1" ]]; then
  require_pinned_value REPO_DIR "${REPO_DIR}" "${PRODUCTION_ROOT}"
fi
require_pinned_value TRAIN_DATA "${TRAIN_DATA}" "${SELECTED_DATA}"
require_pinned_value EXPERIMENT_NAME "${EXPERIMENT_NAME}" \
  opd-n4-graddiv12800-easy4of4-concise20-noneasynormal50-purerkl-step50
require_pinned_value CHECKPOINT_DIR "${CHECKPOINT_DIR}" \
  "${PRODUCTION_ROOT}/checkpoints/${EXPERIMENT_NAME}"
require_pinned_value LOG_FILE "${LOG_FILE}" \
  "${PRODUCTION_ROOT}/logs/difficulty_routed/${EXPERIMENT_NAME}.log"
require_pinned_value DATA_ROOT "${DATA_ROOT}" "${PRODUCTION_ROOT}/data/g-opd"
require_pinned_value STUDENT_MODEL "${STUDENT_MODEL}" "${PRODUCTION_ROOT}/models/Qwen3-4B"
require_pinned_value TEACHER_MODEL "${TEACHER_MODEL}" \
  "${PRODUCTION_ROOT}/models/Qwen3-30B-A3B-Instruct-2507"
require_pinned_value VAL_DATA "${VAL_DATA}" \
  "['${PRODUCTION_ROOT}/data/g-opd/AIME2024/test.parquet', '${PRODUCTION_ROOT}/data/g-opd/AIME2025/test.parquet']"
require_pinned_value TEACHER_PROMPT_KEY "${TEACHER_PROMPT_KEY}" teacher_prompt
require_pinned_value ROLLOUT_N "${ROLLOUT_N}" 4
require_pinned_value PROMPT_BATCH_SIZE "${PROMPT_BATCH_SIZE}" 256
require_pinned_value PPO_MINI_BATCH_SIZE "${PPO_MINI_BATCH_SIZE}" 256
require_pinned_value TOTAL_TRAJECTORIES "${TOTAL_TRAJECTORIES}" 1024
require_pinned_value EXPECTED_GROUP_SIZE "${EXPECTED_GROUP_SIZE}" 4
require_pinned_value TOTAL_TRAINING_STEPS "${TOTAL_TRAINING_STEPS}" 50
require_pinned_value SAVE_FREQ "${SAVE_FREQ}" 20
require_pinned_value N_GPUS_PER_NODE "${N_GPUS_PER_NODE}" 4
require_pinned_value ROLLOUT_TP_SIZE "${ROLLOUT_TP_SIZE}" 4
require_pinned_value MAX_PROMPT_LENGTH "${MAX_PROMPT_LENGTH}" 2048
```

This permits `REPO_DIR` to point at the isolated worktree only when `ABLATION_DRY_RUN=1`, so tests exercise the code under review while data/model/checkpoint/log paths remain the pinned absolute production paths. For `ABLATION_PREFLIGHT_ONLY=1` and default launch, `REPO_DIR` must be `/home/mchen/FiRe-OPD`. `PYTHON_BIN` remains operationally overridable.

Require both delegated launchers to be regular files, then export the inherited contract exactly:

```bash
[[ -f "${GROUP_LAUNCHER}" ]] || die "group-success launcher does not exist: ${GROUP_LAUNCHER}"
[[ -f "${BASE_LAUNCHER}" ]] || die "base launcher does not exist: ${BASE_LAUNCHER}"
export REPO_DIR DATA_ROOT TRAIN_DATA VAL_DATA STUDENT_MODEL TEACHER_MODEL
export TEACHER_PROMPT_KEY ROLLOUT_N PROMPT_BATCH_SIZE PPO_MINI_BATCH_SIZE
export TOTAL_TRAJECTORIES EXPECTED_GROUP_SIZE TOTAL_TRAINING_STEPS SAVE_FREQ
export N_GPUS_PER_NODE ROLLOUT_TP_SIZE MAX_PROMPT_LENGTH EXPERIMENT_NAME CHECKPOINT_DIR
```

Then render two contracts with the same current `GROUP_LAUNCHER`:

```bash
tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

BASELINE_CHECKPOINT="${PRODUCTION_ROOT}/checkpoints/${BASELINE_EXPERIMENT}"
GROUP_SUCCESS_DRY_RUN=1 \
TRAIN_DATA="${SOURCE_DATA}" \
EXPERIMENT_NAME="${BASELINE_EXPERIMENT}" \
CHECKPOINT_DIR="${BASELINE_CHECKPOINT}" \
bash "${GROUP_LAUNCHER}" >"${tmpdir}/baseline.contract"

GROUP_SUCCESS_DRY_RUN=1 \
TRAIN_DATA="${SELECTED_DATA}" \
EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
CHECKPOINT_DIR="${CHECKPOINT_DIR}" \
bash "${GROUP_LAUNCHER}" >"${tmpdir}/candidate.contract"
```

Compare the two files with this exact normalization; any other difference exits 2 and prints a unified diff:

```bash
python3 - \
  "${tmpdir}/baseline.contract" "${tmpdir}/candidate.contract" \
  "${SOURCE_DATA}" "${SELECTED_DATA}" \
  "${BASELINE_EXPERIMENT}" "${EXPERIMENT_NAME}" \
  "${BASELINE_CHECKPOINT}" "${CHECKPOINT_DIR}" <<'PY'
import difflib
from pathlib import Path
import sys

baseline_path, candidate_path = map(Path, sys.argv[1:3])
source_data, selected_data, baseline_name, candidate_name, baseline_ckpt, candidate_ckpt = sys.argv[3:]

def normalize(text, replacements):
    for actual, marker in sorted(replacements, key=lambda pair: len(pair[0]), reverse=True):
        text = text.replace(actual, marker)
    return text

baseline = normalize(
    baseline_path.read_text(encoding="utf-8"),
    [(baseline_ckpt, "<CHECKPOINT_DIR>"), (source_data, "<TRAIN_DATA>"), (baseline_name, "<EXPERIMENT_NAME>")],
)
candidate = normalize(
    candidate_path.read_text(encoding="utf-8"),
    [(candidate_ckpt, "<CHECKPOINT_DIR>"), (selected_data, "<TRAIN_DATA>"), (candidate_name, "<EXPERIMENT_NAME>")],
)
if baseline != candidate:
    sys.stderr.writelines(
        difflib.unified_diff(
            baseline.splitlines(keepends=True),
            candidate.splitlines(keepends=True),
            fromfile="normalized-baseline",
            tofile="normalized-candidate",
        )
    )
    raise SystemExit("data-only contract comparison failed")
PY
printf 'single_variable_contract=PASS\n'
```

For `ABLATION_DRY_RUN=1`, print `artifact_validation=SKIPPED_DRY_RUN`, `cat "${tmpdir}/candidate.contract"`, print the selected hash and log path, then exit 0. Do not test `PYTHON_BIN`, artifact existence, checkpoint contents, or GPU state in this branch.

- [ ] **Step 4: Run dry tests and verify GREEN**

Run the Step 2 command. Expected: all four tests pass.

- [ ] **Step 5: Add actual preflight, provenance, and delegation**

After the dry branch, require an executable `PYTHON_BIN` and capture the strict validator JSON with this exact call:

```bash
artifact_report="$(
  "${PYTHON_BIN}" -m math_eval.validate_gradient_diverse_training_data \
    --selected-parquet "${SELECTED_DATA}" \
    --source-parquet "${SOURCE_DATA}" \
    --selection-manifest "${SELECTION_MANIFEST}" \
    --selected-ids "${SELECTED_IDS}" \
    --tokenizer-path "${STUDENT_MODEL}" \
    --expected-selected-sha256 "${SELECTED_SHA256}" \
    --expected-source-sha256 "${SOURCE_SHA256}" \
    --expected-rows 12800 \
    --expected-source-rows 57046 \
    --expected-eligible-rows 57045 \
    --max-prompt-tokens 2048 \
    --checkpoint-dir "${CHECKPOINT_DIR}"
)"
```

For `ABLATION_PREFLIGHT_ONLY=1`, print the artifact report plus the provenance block and exit 0. For both preflight-only and actual launch, compute provenance with:

```bash
GIT_HEAD="$(git -C "${REPO_DIR}" rev-parse HEAD)"
GIT_DIFF_SHA256="$(git -C "${REPO_DIR}" diff --no-ext-diff --binary HEAD -- | sha256sum | awk '{print $1}')"
GROUP_LAUNCHER_SHA256="$(sha256sum "${GROUP_LAUNCHER}" | awk '{print $1}')"
BASE_LAUNCHER_SHA256="$(sha256sum "${BASE_LAUNCHER}" | awk '{print $1}')"
ABLATION_LAUNCHER_SHA256="$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')"
```

Emit the complete block with exact shell operations:

```bash
print_provenance() {
  printf 'git_head=%s\n' "${GIT_HEAD}"
  printf 'git_status_begin\n'
  git -C "${REPO_DIR}" status --short
  printf 'git_status_end\n'
  printf 'git_diff_sha256=%s\n' "${GIT_DIFF_SHA256}"
  printf 'group_launcher_sha256=%s\n' "${GROUP_LAUNCHER_SHA256}"
  printf 'base_launcher_sha256=%s\n' "${BASE_LAUNCHER_SHA256}"
  printf 'ablation_launcher_sha256=%s\n' "${ABLATION_LAUNCHER_SHA256}"
  printf 'selected_parquet_sha256=%s\n' "${SELECTED_SHA256}"
  printf 'artifact_report=%s\n' "${artifact_report}"
  printf 'resolved_contract_begin\n'
  cat "${tmpdir}/candidate.contract"
  printf 'resolved_contract_end\n'
}
```

For actual launch only, create the log parent, redirect through `tee -a "${LOG_FILE}"`, print the provenance block, remove the temporary directory and its trap, then delegate:

```bash
mkdir -p "$(dirname -- "${LOG_FILE}")"
exec > >(tee -a "${LOG_FILE}") 2>&1
print_provenance
rm -rf "${tmpdir}"
trap - EXIT
cd "${REPO_DIR}"
exec bash "${GROUP_LAUNCHER}"
```

Do not pass extra Hydra arguments here; the pinned exported environment plus the existing group launcher are the complete execution interface.

- [ ] **Step 6: Add tests that all pinned values and forbidden methods are visible**

Extend the dry-run test with:

```python
    required = [
        "algorithm.difficulty_aware_opd.easy_group_correct_count=4",
        "algorithm.difficulty_aware_opd.easy_prompt_style=concise",
        "algorithm.difficulty_aware_opd.default_prompt_style=normal",
        "algorithm.difficulty_aware_opd.easy_esr_beta=0.20",
        "algorithm.difficulty_aware_opd.non_easy_esr_beta=0.50",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.kl_loss_coef=0",
        "algorithm.use_kl_in_reward=False",
        "algorithm.candidate_selection.enabled=False",
        "actor_rollout_ref.actor.policy_loss.length_aware_opd=False",
        "algorithm.rethinking_opd_probe.enabled=False",
        "trainer.save_freq=20",
        "trainer.total_training_steps=50",
    ]
    for value in required:
        assert value in completed.stdout
```

Add one parametrized rejection test for `PROMPT_BATCH_SIZE=128`, `PPO_MINI_BATCH_SIZE=1024`, `TOTAL_TRAJECTORIES=2048`, `EXPECTED_GROUP_SIZE=8`, `TOTAL_TRAINING_STEPS=49`, `SAVE_FREQ=50`, `STUDENT_MODEL=/tmp/student`, and `TEACHER_MODEL=/tmp/teacher`; assert exit 2 and `<NAME> must remain pinned`.

- [ ] **Step 7: Run wrapper and existing launcher verification**

Run:

```bash
bash -n run_train_group_success_difficulty_opd.sh
bash -n run_train_group_success_gradient_diverse_ablation.sh
/home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  verl/tests/trainer/ppo/test_group_success_launcher.py \
  verl/tests/trainer/ppo/test_group_success_gradient_diverse_ablation_launcher.py
ABLATION_DRY_RUN=1 bash run_train_group_success_gradient_diverse_ablation.sh
```

Expected: syntax checks and tests pass; the final output includes `single_variable_contract=PASS`, the selected parquet path/hash, the exact production name, and the existing group-success launcher command.

- [ ] **Step 8: Commit Task 3**

```bash
git add run_train_group_success_gradient_diverse_ablation.sh \
  verl/tests/trainer/ppo/test_group_success_gradient_diverse_ablation_launcher.py
git commit -m "Add gradient-diverse OPD ablation launcher"
```

---

### Task 4: Add a Strict Eight-Dataset Evaluation Summarizer

**Files:**
- Create: `math_eval/summarize_math_eval_suite.py`
- Create: `math_eval/test_summarize_math_eval_suite.py`

**Interfaces:**
- Produces: `summarize_eval_file(path: Path, expected_samples: int) -> dict[str, float | int]`.
- Produces: `summarize_suite(output_root: Path, model_name: str, datasets: Sequence[str], expected_samples: int) -> dict[str, object]`.
- Produces: `compare_runs(candidate: dict, baselines: Mapping[str, dict]) -> dict[str, object]` with deltas defined as candidate minus baseline.
- CLI accepts one `--candidate`, repeated `--baseline LABEL=MODEL_NAME`, `--output-root`, `--expected-samples`, and `--output-json`.

- [ ] **Step 1: Write failing metric-definition tests**

Create `math_eval/test_summarize_math_eval_suite.py`:

```python
import json
from pathlib import Path

import pytest

from math_eval import summarize_math_eval_suite as summary


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_summarize_eval_file_defines_pass1_passk_and_lengths(tmp_path: Path):
    path = tmp_path / "run.jsonl"
    _write(
        path,
        [
            {"acc_list": [True, False, False, False], "response_lengths": [1, 3, 5, 7]},
            {"acc_list": [True, True, False, False], "response_lengths": [2, 4, 6, 8]},
        ],
    )

    result = summary.summarize_eval_file(path, expected_samples=4)

    assert result == {
        "problems": 2,
        "samples": 8,
        "samples_per_problem": 4,
        "pass_at_1": pytest.approx(3 / 8),
        "pass_at_4": 1.0,
        "mean_response_length": 4.5,
        "median_response_length": 4.5,
    }


def test_summarize_eval_file_rejects_wrong_or_misaligned_sample_count(tmp_path: Path):
    wrong_n = tmp_path / "wrong_n.jsonl"
    _write(wrong_n, [{"acc_list": [True], "response_lengths": [1]}])
    with pytest.raises(ValueError, match="expected 4 samples"):
        summary.summarize_eval_file(wrong_n, expected_samples=4)

    misaligned = tmp_path / "misaligned.jsonl"
    _write(misaligned, [{"acc_list": [True, False], "response_lengths": [1]}])
    with pytest.raises(ValueError, match="acc_list/response_lengths length mismatch"):
        summary.summarize_eval_file(misaligned, expected_samples=2)


def test_suite_macro_is_unweighted_and_delta_is_candidate_minus_baseline(tmp_path: Path):
    datasets = ("small", "large")
    candidate = "candidate"
    baseline = "baseline"
    for dataset, candidate_acc, baseline_acc in (
        ("small", [True, False], [False, False]),
        ("large", [True, True], [True, False]),
    ):
        _write(
            tmp_path / dataset / f"{candidate}.jsonl",
            [{"acc_list": candidate_acc, "response_lengths": [2, 4]}],
        )
        _write(
            tmp_path / dataset / f"{baseline}.jsonl",
            [{"acc_list": baseline_acc, "response_lengths": [4, 8]}],
        )

    candidate_summary = summary.summarize_suite(tmp_path, candidate, datasets, 2)
    baseline_summary = summary.summarize_suite(tmp_path, baseline, datasets, 2)
    compared = summary.compare_runs(candidate_summary, {"group_success": baseline_summary})

    assert candidate_summary["macro"]["pass_at_1"] == 0.75
    assert candidate_summary["macro"]["pass_at_2"] == 1.0
    assert compared["deltas"]["group_success"]["pass_at_1"] == 0.5
    assert compared["deltas"]["group_success"]["mean_response_length"] == -3.0
```

- [ ] **Step 2: Run summarizer tests and verify RED**

Run:

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  math_eval/test_summarize_math_eval_suite.py
```

Expected: collection fails because the summarizer module does not exist.

- [ ] **Step 3: Implement strict file and suite summaries**

Use these exact dataset identifiers and metric rules:

```python
DATASETS = (
    "aime24",
    "aime25",
    "hmmt25_feb",
    "hmmt25_nov",
    "math500",
    "minervamath",
    "olympiadbench",
    "amc2023",
)


def metric_names(expected_samples: int) -> tuple[str, str, str, str]:
    return (
        "pass_at_1",
        f"pass_at_{expected_samples}",
        "mean_response_length",
        "median_response_length",
    )
```

`summarize_eval_file` must reject a missing/empty file, malformed JSON, non-list arrays, non-boolean accuracy entries, non-numeric/negative lengths, any row whose sample count differs from `expected_samples`, and array-length mismatch. Flatten all responses across problems and compute:

```python
pass_at_1 = sum(flat_accuracy) / len(flat_accuracy)
pass_at_k = sum(any(row_accuracy) for row_accuracy in per_problem_accuracy) / problem_count
mean_response_length = statistics.fmean(flat_lengths)
median_response_length = statistics.median(flat_lengths)
```

Because every problem is required to have exactly 32 responses, `pass_at_1` is also the mean per-problem correctness. `summarize_suite` must require every `${output_root}/${dataset}/${model_name}.jsonl`, preserve the fixed dataset order, and compute each macro value with `statistics.fmean` over the eight per-dataset values. Name the high-sample field dynamically as `pass_at_{expected_samples}` so the unit fixture uses `pass_at_2` and production uses `pass_at_32`.

Return this stable suite shape:

```python
return {
    "model_name": model_name,
    "expected_samples": expected_samples,
    "dataset_order": list(datasets),
    "datasets": per_dataset,
    "macro": {
        metric: statistics.fmean(per_dataset[name][metric] for name in datasets)
        for metric in metric_names(expected_samples)
    },
}
```

`compare_runs` must require matching dataset order and expected sample count, preserve full candidate/baseline summaries, and compute per-dataset plus macro deltas using `metric_names(candidate["expected_samples"])`.

- [ ] **Step 4: Implement the CLI and atomic JSON output**

Parse each baseline with:

```python
label, separator, model_name = value.partition("=")
if not separator or not label or not model_name:
    raise ValueError("--baseline must use LABEL=MODEL_NAME")
```

Call `summarize_suite` for candidate and each baseline, write a pretty sorted JSON report to a temporary sibling followed by `Path.replace`, and print the same JSON to stdout. Reject duplicate baseline labels and a pre-existing non-file output path.

- [ ] **Step 5: Run summarizer tests and syntax checks**

Run:

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  math_eval/test_summarize_math_eval_suite.py
/home/mchen/miniconda3/envs/verl/bin/python -m compileall -q \
  math_eval/summarize_math_eval_suite.py
```

Expected: all tests pass and compileall is silent.

- [ ] **Step 6: Commit Task 4**

```bash
git add math_eval/summarize_math_eval_suite.py \
  math_eval/test_summarize_math_eval_suite.py
git commit -m "Summarize strong-to-weak math evaluation suite"
```

---

### Task 5: Review, Verify, and Integrate Without Disturbing the Dirty Main Worktree

**Files:**
- Verify: all Task 1-4 files.
- Preserve: every unrelated modified/untracked file in `/home/mchen/FiRe-OPD`.

**Interfaces:**
- Consumes: four isolated implementation commits based on `e9b8d26`.
- Produces: the same four commits applied to the primary worktree, with unrelated status entries unchanged.

- [ ] **Step 1: Run focused and regression tests in the isolated worktree**

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  math_eval/test_validate_gradient_diverse_training_data.py \
  math_eval/test_summarize_math_eval_suite.py \
  verl/tests/trainer/ppo/test_group_success_launcher.py \
  verl/tests/trainer/ppo/test_group_success_gradient_diverse_ablation_launcher.py \
  verl/tests/trainer/ppo/test_difficulty_aware_opd.py \
  verl/tests/trainer/ppo/test_tale_budget.py
bash -n run_train_group_success_difficulty_opd.sh
bash -n run_train_group_success_gradient_diverse_ablation.sh
/home/mchen/miniconda3/envs/verl/bin/python -m compileall -q \
  math_eval/validate_gradient_diverse_training_data.py \
  math_eval/summarize_math_eval_suite.py
```

Expected: every pytest target passes; Bash and compile checks are silent. If either named existing regression test path differs in this checkout, locate the exact file with `rg --files verl/tests | rg '(difficulty_aware_opd|tale_budget)'` and run every matching test file.

- [ ] **Step 2: Run structural contract checks**

```bash
ABLATION_DRY_RUN=1 bash run_train_group_success_gradient_diverse_ablation.sh \
  > /tmp/gradient-diverse-ablation.dry-run.txt
rg -n 'single_variable_contract=PASS|TRAIN_DATA=|EXPERIMENT_NAME=|trainer.total_training_steps=50|trainer.resume_mode=disable' \
  /tmp/gradient-diverse-ablation.dry-run.txt
git diff e9b8d26..HEAD -- \
  run_train_group_success_difficulty_opd.sh \
  math_eval/validate_gradient_diverse_training_data.py \
  run_train_group_success_gradient_diverse_ablation.sh \
  math_eval/summarize_math_eval_suite.py
```

Expected: the dry-run contains the pinned contract; the diff contains no trainer, loss, TALE, scorer, dataset, or eval-launcher implementation change.

- [ ] **Step 3: Request code review and resolve only concrete findings**

Invoke `superpowers:requesting-code-review` against `e9b8d26..HEAD`. Require the reviewer to check scientific one-variable isolation, fail-closed behavior, shell quoting, exact-source validation, tokenizer parity with veRL, non-empty checkpoint refusal, and metric definitions. Re-run the exact commands from Steps 1-2 after any change and commit each accepted correction with only its affected files.

- [ ] **Step 4: Snapshot primary-worktree status before integration**

Run in `/home/mchen/FiRe-OPD`:

```bash
git status --short > /tmp/fire-opd.status.before-ablation-integration
git diff --no-ext-diff --binary HEAD -- | sha256sum \
  > /tmp/fire-opd.diff.before-ablation-integration.sha256
```

Expected: the snapshot records the user's existing dirty state without modifying it.

- [ ] **Step 5: Integrate the isolated commits in chronological order**

From the isolated worktree, record:

```bash
git rev-list --reverse e9b8d26..HEAD \
  > /tmp/gradient-diverse-ablation.commit-list
```

Then in `/home/mchen/FiRe-OPD`:

```bash
while IFS= read -r commit; do
  git cherry-pick "$commit" || exit 1
done < /tmp/gradient-diverse-ablation.commit-list
```

Expected: all commits apply without touching unrelated dirty paths. If Git reports overlap, abort only the active cherry-pick with `git cherry-pick --abort`, keep the dirty files intact, inspect the overlap, and integrate the known ablation hunks with `apply_patch` followed by equivalent commits.

- [ ] **Step 6: Re-run verification in the primary worktree**

Run the complete Step 1 suite plus:

```bash
git status --short > /tmp/fire-opd.status.after-ablation-integration
comm -3 \
  <(sort /tmp/fire-opd.status.before-ablation-integration) \
  <(sort /tmp/fire-opd.status.after-ablation-integration)
```

Expected: tests pass; status differences are limited to paths created/modified by this plan becoming committed. No unrelated dirty entry disappears or changes state.

---

### Task 6: Run Production Preflight and Launch from `opd-CLI`

**Files:**
- Execute: `run_train_group_success_gradient_diverse_ablation.sh`.
- Produce: `logs/difficulty_routed/opd-n4-graddiv12800-easy4of4-concise20-noneasynormal50-purerkl-step50.log`.
- Produce checkpoints under: `checkpoints/opd-n4-graddiv12800-easy4of4-concise20-noneasynormal50-purerkl-step50/`.

**Interfaces:**
- Consumes: one active four-GPU `opd-CLI` Slurm allocation with logical CUDA devices 0-3 idle.
- Produces: a non-resumed 50-step training process whose first optimizer step satisfies all group-success invariants.

- [ ] **Step 1: Locate the live `opd-CLI` pane without assuming its window index**

Run from the login shell:

```bash
OPD_PANE="$(tmux list-panes -a -F '#{session_name}:#{window_index}.#{pane_index}' \
  | awk '/^opd-CLI:/{print; exit}')"
test -n "${OPD_PANE}"
printf 'OPD_PANE=%s\n' "${OPD_PANE}"
```

Expected: one concrete target such as `opd-CLI:0.0` is printed. Do not create a new allocation if the pane is absent; report that launch authority is blocked.

- [ ] **Step 2: Inspect allocation identity and GPU idleness inside that pane**

```bash
tmux send-keys -t "${OPD_PANE}" \
  "printf 'SLURM_JOB_ID=%s CUDA_VISIBLE_DEVICES=%s\\n' \"\${SLURM_JOB_ID:-}\" \"\${CUDA_VISIBLE_DEVICES:-}\"; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader; nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv,noheader" C-m
sleep 2
tmux capture-pane -p -t "${OPD_PANE}" -S -80
```

Expected: `SLURM_JOB_ID` is nonempty, the allocation exposes four logical devices, and there are no unexpected compute processes. Use `scontrol show job "$SLURM_JOB_ID"` inside the pane if the mapping is ambiguous.

- [ ] **Step 3: Run dry-run and full preflight inside `opd-CLI`**

```bash
tmux send-keys -t "${OPD_PANE}" \
  "cd /home/mchen/FiRe-OPD && ABLATION_DRY_RUN=1 bash run_train_group_success_gradient_diverse_ablation.sh | tee /tmp/gradient-diverse-ablation.dry-run.txt && ABLATION_PREFLIGHT_ONLY=1 bash run_train_group_success_gradient_diverse_ablation.sh | tee /tmp/gradient-diverse-ablation.preflight.txt" C-m
```

Poll with `tmux capture-pane -p -t "${OPD_PANE}" -S -120` while continuing to send concise user updates. Expected: both commands exit 0; dry output says `single_variable_contract=PASS`; preflight JSON reports 12,800 rows, 12,800 unique prompts, max 668 tokens, zero over-limit prompts, exact schema/source equality, and an empty/missing candidate checkpoint directory.

- [ ] **Step 4: Launch the exact production wrapper in the same pane**

```bash
tmux send-keys -t "${OPD_PANE}" \
  "cd /home/mchen/FiRe-OPD && bash run_train_group_success_gradient_diverse_ablation.sh" C-m
```

Expected: the wrapper records provenance, delegates to the existing launcher, W&B uses the pinned candidate name, and no resume checkpoint is loaded.

- [ ] **Step 5: Verify data loading before waiting for step 1**

Run from the login shell:

```bash
LOG=/home/mchen/FiRe-OPD/logs/difficulty_routed/opd-n4-graddiv12800-easy4of4-concise20-noneasynormal50-purerkl-step50.log
rg -n 'dataset len: 12800|filter dataset len: 12800|Size of train dataloader: 50|resume_mode=disable|train_gradient_diverse_12800.parquet' "$LOG"
```

Expected: both pre- and post-filter dataset lengths are 12,800, the train dataloader has exactly 50 batches, the selected parquet is active, and resume is disabled. If post-filter length is not 12,800, stop this new process before any optimizer step and diagnose the mismatch.

- [ ] **Step 6: Validate the first completed optimizer step mechanically**

After the log contains `training/global_step:1`, run:

```bash
/home/mchen/miniconda3/envs/verl/bin/python - "$LOG" <<'PY'
import math
from pathlib import Path
import re
import sys

text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
lines = [line for line in text.splitlines() if re.search(r"\bstep:1 -", line) and "training/global_step:1" in line]
if len(lines) != 1:
    raise SystemExit(f"expected exactly one completed step-1 metric line, found {len(lines)}")
line = lines[0]
metrics = {
    key: float(value)
    for key, value in re.findall(r"(?:^| - )([^:]+):([-+0-9.eE]+)", line)
}

def close(name, expected, tolerance=1e-6):
    actual = metrics.get(name)
    if actual is None or not math.isclose(actual, expected, abs_tol=tolerance, rel_tol=0.0):
        raise SystemExit(f"{name}={actual!r}, expected {expected}")

close("difficulty_aware_opd/enabled", 1.0)
close("difficulty_aware_opd/group_count", 256.0)
close("difficulty_aware_opd/expected_group_size", 4.0)
close("difficulty_aware_opd/esr_beta_min", 0.2)
close("difficulty_aware_opd/esr_beta_max", 0.5)
close("difficulty_aware_opd/concise_prompt_ratio", metrics["difficulty_aware_opd/easy_group_ratio"])
close(
    "difficulty_aware_opd/normal_prompt_ratio",
    1.0 - metrics["difficulty_aware_opd/easy_group_ratio"],
)
bucket_sum = sum(
    metrics[name]
    for name in (
        "difficulty_aware_opd/easy_group_ratio",
        "difficulty_aware_opd/learnable_group_ratio",
        "difficulty_aware_opd/unresolved_group_ratio",
    )
)
if not math.isclose(bucket_sum, 1.0, abs_tol=1e-6, rel_tol=0.0):
    raise SystemExit(f"difficulty bucket ratios sum to {bucket_sum}")
correct_count_sum = sum(
    metrics[f"difficulty_aware_opd/group_correct_count_{index}_ratio"]
    for index in range(5)
)
if not math.isclose(correct_count_sum, 1.0, abs_tol=1e-6, rel_tol=0.0):
    raise SystemExit(f"group-correct-count ratios sum to {correct_count_sum}")
close("actor/kl_coef", 0.0)
if "actor/pg_loss" not in metrics or "actor/grad_norm" not in metrics:
    raise SystemExit("actor update metrics are missing")
print("FIRST_STEP_ACCEPTANCE=PASS")
PY
```

Also require the resolved contract in the log to contain entropy 0, KL reward/loss 0, candidate selection false, length-aware false, rethinking probe false, 256 prompts, `n=4`, mini-batch 256, and 50 total steps. Expected: the script prints `FIRST_STEP_ACCEPTANCE=PASS`.

- [ ] **Step 7: Leave the accepted run active through step 50**

Do not stop or restart after first-step acceptance. Report the log path, W&B run name, Slurm job ID, observed first-step wall time, and an ETA based on the rolling `timing_s/step` values. Monitor with short non-blocking checks; training-batch `critic/score/mean` remains diagnostic and is not used as a generalization result.

---

### Task 7: Verify Step 50, Evaluate All Eight Datasets, and Compare Baselines

**Files:**
- Reuse: `math_eval/run_eval_math_opd_step50.sh`.
- Read: candidate, completed group-success, and vanilla OPD step-50 checkpoints.
- Produce: eight JSONL files per evaluated model under `math_eval/eval_outputs/<dataset>/`.
- Produce: `math_eval/eval_outputs/opd-n4-graddiv12800-step50-comparison.json`.

**Interfaces:**
- Candidate model name: `opd-n4-graddiv12800-easy4of4-concise20-noneasynormal50-purerkl-step50-canonical8-n32-seed42`.
- Group-success control name: `opd-n4-easy4of4-concise20-noneasynormal50-purerkl-step50-canonical8-n32-seed42`.
- Vanilla control name: `opd-strong-to-weak-rawprompt-4gpu-tp4-refmb4-rollmb4-step50-canonical8-n32-seed42`.

- [ ] **Step 1: Verify training completion and checkpoint topology**

```bash
EXPERIMENT=opd-n4-graddiv12800-easy4of4-concise20-noneasynormal50-purerkl-step50
ROOT=/home/mchen/FiRe-OPD/checkpoints/${EXPERIMENT}
test -d "${ROOT}/global_step_20/actor"
test -d "${ROOT}/global_step_40/actor"
test -d "${ROOT}/global_step_50/actor"
rg -n 'training/global_step:50|step:50 -' \
  "/home/mchen/FiRe-OPD/logs/difficulty_routed/${EXPERIMENT}.log"
```

Expected: checkpoints 20, 40, and 50 exist and the log has one completed step-50 optimizer line. A missing final checkpoint is not treated as completion.

- [ ] **Step 2: Reconfirm the `opd-CLI` allocation and idle GPUs**

Repeat Task 6 Steps 1-2. Expected: the active allocation still has four available logical GPUs. If the training allocation expired, use a newly user-authorized `opd-CLI` allocation; do not launch eval from the login shell.

- [ ] **Step 3: Merge and evaluate the candidate with the canonical launcher**

Send this exact command to the `opd-CLI` pane:

```bash
cd /home/mchen/FiRe-OPD && \
EXPERIMENT_NAME=opd-n4-graddiv12800-easy4of4-concise20-noneasynormal50-purerkl-step50 \
MODEL_NAME=opd-n4-graddiv12800-easy4of4-concise20-noneasynormal50-purerkl-step50-canonical8-n32-seed42 \
N_SAMPLES=32 MAX_TOKENS=16384 TEMPERATURE=1.0 TOP_P=1.0 SEED=42 \
bash math_eval/run_eval_math_opd_step50.sh
```

Expected: the launcher merges `global_step_50/actor` into `global_step_50_hf` if needed and writes one candidate JSONL for each of the eight fixed datasets.

- [ ] **Step 4: Evaluate the prior group-success control with the identical launcher**

The earlier control currently has only a completed AIME24 output in its old output tree, so run the canonical eight-dataset suite rather than mixing protocols:

```bash
cd /home/mchen/FiRe-OPD && \
EXPERIMENT_NAME=opd-n4-easy4of4-concise20-noneasynormal50-purerkl-step50 \
MODEL_NAME=opd-n4-easy4of4-concise20-noneasynormal50-purerkl-step50-canonical8-n32-seed42 \
N_SAMPLES=32 MAX_TOKENS=16384 TEMPERATURE=1.0 TOP_P=1.0 SEED=42 \
bash math_eval/run_eval_math_opd_step50.sh
```

Expected: eight fresh control files use the same evaluator, sampling settings, GPU-pair layout, and benchmark inputs as the candidate.

- [ ] **Step 5: Evaluate the canonical strong-to-weak vanilla OPD control identically**

```bash
cd /home/mchen/FiRe-OPD && \
EXPERIMENT_NAME=opd-strong-to-weak-rawprompt-4gpu-tp4-refmb4-rollmb4 \
MODEL_NAME=opd-strong-to-weak-rawprompt-4gpu-tp4-refmb4-rollmb4-step50-canonical8-n32-seed42 \
N_SAMPLES=32 MAX_TOKENS=16384 TEMPERATURE=1.0 TOP_P=1.0 SEED=42 \
bash math_eval/run_eval_math_opd_step50.sh
```

Expected: eight vanilla files are regenerated under the exact same canonical protocol; older six-dataset Table-2 outputs are not mixed into the comparison.

- [ ] **Step 6: Validate file completeness before computing metrics**

```bash
for dataset in aime24 aime25 hmmt25_feb hmmt25_nov math500 minervamath olympiadbench amc2023; do
  for model in \
    opd-n4-graddiv12800-easy4of4-concise20-noneasynormal50-purerkl-step50-canonical8-n32-seed42 \
    opd-n4-easy4of4-concise20-noneasynormal50-purerkl-step50-canonical8-n32-seed42 \
    opd-strong-to-weak-rawprompt-4gpu-tp4-refmb4-rollmb4-step50-canonical8-n32-seed42; do
    test -s "/home/mchen/FiRe-OPD/math_eval/eval_outputs/${dataset}/${model}.jsonl" || exit 1
  done
done
```

Expected: all 24 JSONL files exist and are nonempty.

- [ ] **Step 7: Compute the predeclared report and deltas**

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m math_eval.summarize_math_eval_suite \
  --output-root /home/mchen/FiRe-OPD/math_eval/eval_outputs \
  --candidate opd-n4-graddiv12800-easy4of4-concise20-noneasynormal50-purerkl-step50-canonical8-n32-seed42 \
  --baseline group_success=opd-n4-easy4of4-concise20-noneasynormal50-purerkl-step50-canonical8-n32-seed42 \
  --baseline vanilla_opd=opd-strong-to-weak-rawprompt-4gpu-tp4-refmb4-rollmb4-step50-canonical8-n32-seed42 \
  --expected-samples 32 \
  --output-json /home/mchen/FiRe-OPD/math_eval/eval_outputs/opd-n4-graddiv12800-step50-comparison.json
```

Expected: the command validates exactly 32 samples per problem, prints per-dataset and unweighted eight-dataset macro pass@1/pass@32/mean/median length, and writes candidate-minus-control deltas.

- [ ] **Step 8: Hand off a narrow scientific interpretation**

Report:

1. Candidate, group-success control, and vanilla OPD per-dataset values.
2. Unweighted eight-dataset macro pass@1 as the primary outcome.
3. Macro pass@32 and response-length metrics as secondary outcomes.
4. Candidate minus group-success and candidate minus vanilla deltas.
5. The frozen run provenance, selected artifact hash, and whether the current dirty snapshot can be proven identical to the old group-success training snapshot.

Interpret a pass@32-only gain as coverage without improved one-sample reliability. Treat a pass@1 gain over group-success as evidence that the selected data helped this system under the fixed 51,200-trajectory budget, but not as proof that gradient-space diversity caused the gain. Publication-grade attribution still requires same-snapshot uniform-random and metadata-matched 12,800-question controls.
