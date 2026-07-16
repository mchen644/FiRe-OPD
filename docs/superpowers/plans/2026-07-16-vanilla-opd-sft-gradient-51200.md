# Vanilla OPD SFT-Gradient 51,200 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reuse the frozen Qwen2.5 SFT-gradient artifacts to select exactly 51,200 unique DeepMath questions, then train one 50-step canonical Vanilla OPD candidate whose only scientific input difference from the historical Vanilla command is the training data.

**Architecture:** Add an explicit `vanilla_51200` profile to the existing selector without changing the frozen 12,800 profile, enforce the old selected sequence as an exact prefix, and publish to a new append-only namespace. Build one auditable canonical Vanilla OPD launcher and a thin fail-closed candidate wrapper that validates the new artifact, proves command equivalence, records provenance, and delegates without adding training logic.

**Tech Stack:** Python 3.10, PyTorch, PyArrow, safetensors, scikit-learn, official CUDA cosine K-means, Bash, pytest, Hydra/VERL, Ray, vLLM, Slurm, tmux.

## Global Constraints

- Work only in `/home/mchen/FiRe-OPD/.worktrees/vanilla-sft-gradient-51200-opd` on branch `vanilla-sft-gradient-51200-opd`; never switch, stash, or edit the dirty root checkout.
- Base source is local `main@19578a8550a2638840df2bb20149f465dc10c5e2`; every GPU phase must use an exact committed, clean worktree HEAD.
- Preserve the old 12,800 artifacts byte-for-byte:
  - parquet `caf303c5d151fdaed2e21eebc257917f13660c589906e3da0e569ffdfd59b059`;
  - manifest `a1a45382ee577e24f9b455386f9f3adcab210ff40a83ea760c2110fb96f8f90a`;
  - selected IDs `a78c04cff10c148f45bf9c828e07cc9cf01398b48efeb707253273af359a037e`;
  - diagnostics `d2105179f2ce5a7c796aef07136bbc1dce79b07b5bc2de4d2319b18b47ea761f`.
- Reuse, never recompute, the 57,045 projected-gradient rows under `data/gradient_diversity/gradients/qwen2.5-0.5b-instruct`; gradient manifest SHA-256 is `9a534118a08e736a15e806933d5cf90c2a99822fee28bf18644e5ff344ce3ae2`.
- Keep the source parquet at 57,046 rows and SHA-256 `de3350fdd00bc0410550098ea65179e2be873da99e4075f80de575fc17670597`; the only excluded proxy row is `deepmath-level6-038794`.
- Pin the prepared JSONL, prepared manifest, and eligibility report to SHA-256 `ee8d55c943577888f14052b3953a0a8aa07d26076e50ddffa41a102b60027344`, `5add2e965473d3647f2318c73f957434fbe23344516089be9adfe89db9c6a70e`, and `cc8d5b888def37761519e15c6ef722e5a22d91ae2342c6d669f008bcff3dcbb6`.
- Pin the official reference repository to commit `d9484cd3b5991030b901ac4a3a9e2472dbfac2ad` and tree `a0079d8c5e15cb18bb4790f99c43cc19bb9ecd50`; reject a dirty reference checkout.
- The primary selector remains cosine K-means ratio `0.10`, K=5,704, 20 iterations, K-means seed 42, balanced round-robin seed 42, without replacement. Ratios `0.10/0.01` and seeds `42/43` remain the diagnostic grid.
- Select exactly 51,200 unique rows. The first 12,800 selected-ID records must exactly equal the frozen old selected-ID file.
- Publish only under `data/gradient_diversity/selection_51200/` and `data/gradient_diversity/DeepMath-103K/train_gradient_diverse_51200.parquet`; never overwrite or adopt partial artifacts.
- Use `/home/mchen/miniconda3/envs/gvendi-opd/bin/python` for selection and `/home/mchen/miniconda3/envs/verl/bin/python` for training and VERL tests. Do not install packages at runtime.
- Pin Vanilla OPD to Qwen3-4B, Qwen3-30B-A3B-Instruct-2507, raw prompts on both sides, `n=1`, batch/mini-batch 1,024, 50 steps, 51,200 trajectories, seed 42, response length 16,384, temperature/top-p 1.0/1.0, LR `1e-6`, zero warmup, reverse-KL-only advantages, token mean, token-level IS threshold 5.0, four GPUs, and TP=4.
- Explicitly disable length-aware OPD, difficulty routing, TALE/ESR, candidate selection, rethinking, entropy contribution, KL reward, and effective KL loss.
- Retain canonical training-time validation before training and every 10 steps on AIME24/AIME25 with `n=8`; defer the independent n=32 benchmark suite.
- Use experiment name `opd-strong-to-weak-sftgrad51200-rawprompt-4gpu-tp4-refmb4-rollmb4`, save only the step-50 comparison checkpoint, and set `trainer.resume_mode=disable`.
- GPU work requires the active `opd-CLI` tmux session, a uniquely running Slurm allocation, four unique allocation-owned idle CUDA tokens, no nested `srun`, and explicit authorization immediately before launch.
- Follow TDD: each production behavior starts with a focused failing test, then the minimal implementation, verification, and a focused commit.

---

## File Map

- Modify `math_eval/select_gradient_diverse_deepmath.py`: define selection profiles, load/validate the frozen prefix, bind clean source identity, and keep old-profile manifest bytes compatible.
- Modify `math_eval/test_select_gradient_diverse_deepmath.py`: lock profile parsing, prefix validation, source provenance, and old/new manifest differences.
- Create `run_select_gradient_diverse_deepmath_51200.sh`: run selection only from frozen gradients with dry-run, CPU preflight, Slurm/tmux/GPU gates, locks, hashes, and provenance.
- Create `math_eval/test_gradient_diverse_51200_launcher.py`: test the selection-only shell contract without GPUs or production artifacts.
- Modify `math_eval/validate_gradient_diverse_training_data.py`: optionally enforce the named 51,200 profile and independently recheck its frozen prefix.
- Modify `math_eval/test_validate_gradient_diverse_training_data.py`: test profile/prefix acceptance and corruption rejection while preserving the old report shape.
- Create `run_train_vanilla_opd.sh`: provide one canonical auditable Vanilla OPD command with no selection-specific logic.
- Create `verl/tests/trainer/ppo/test_vanilla_opd_launcher.py`: lock all Vanilla settings, raw-prompt behavior, dry-run output, and override rejection.
- Create `run_train_vanilla_sft_gradient_51200.sh`: pin the selected artifact and output identity, compare normalized baseline/candidate commands, validate artifacts, record provenance, gate GPUs, and delegate.
- Create `verl/tests/trainer/ppo/test_vanilla_sft_gradient_51200_launcher.py`: test candidate dry-run equivalence, pin rejection, and delegation.
- Create `math_eval/validate_vanilla_opd_checkpoint.py`: validate the step-50 four-rank checkpoint and exact data-loader consumption.
- Create `math_eval/test_validate_vanilla_opd_checkpoint.py`: lock checkpoint topology and `samples_yielded=51200` semantics.

---

### Task 1: Add an Immutable `vanilla_51200` Selection Profile

**Files:**
- Modify: `math_eval/select_gradient_diverse_deepmath.py`
- Modify: `math_eval/test_select_gradient_diverse_deepmath.py`

**Interfaces:**
- Produces `SelectionProfile`, `selection_profile(name)`, `load_frozen_selection_prefix(...)`, `validate_frozen_selection_prefix(...)`, and `build_selector_source_provenance(...)`.
- Extends `run_selection(..., selection_profile_name: str = "coreset_12800", frozen_prefix_selected_ids_path: Path | None = None)` without changing the old default output contract.
- Extends CLI with `--selection-profile` and `--frozen-prefix-selected-ids`.

- [ ] **Step 1: Add failing profile and target-size tests**

Extend the existing selector import block only with the profile symbols needed by this first test cycle:

```python
from math_eval.select_gradient_diverse_deepmath import (
    CORESET_12800_PROFILE,
    VANILLA_51200_PROFILE,
    selection_profile,
)


def test_selection_profiles_pin_old_and_vanilla_target_sizes() -> None:
    old = selection_profile("coreset_12800")
    vanilla = selection_profile("vanilla_51200")

    assert old == CORESET_12800_PROFILE
    assert old.target_size == 12_800
    assert old.frozen_prefix_size == 0
    assert vanilla == VANILLA_51200_PROFILE
    assert vanilla.target_size == 51_200
    assert vanilla.frozen_prefix_size == 12_800


def test_selection_profile_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unsupported selection profile"):
        selection_profile("almost_vanilla")
```

- [ ] **Step 2: Run the profile tests and verify RED**

Run:

```bash
cd /home/mchen/FiRe-OPD/.worktrees/vanilla-sft-gradient-51200-opd
/home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest -q \
  math_eval/test_select_gradient_diverse_deepmath.py \
  -k 'selection_profiles or selection_profile_rejects'
```

Expected: collection fails because the profile symbols do not exist.

- [ ] **Step 3: Implement the typed profiles**

Add `dataclass` to imports and these definitions near the selector constants:

```python
from dataclasses import dataclass

CORESET_12800_PROFILE_NAME = "coreset_12800"
VANILLA_51200_PROFILE_NAME = "vanilla_51200"
VANILLA_TARGET_SIZE = 51_200
FROZEN_PREFIX_SIZE = 12_800
FROZEN_PREFIX_SELECTED_IDS_SHA256 = (
    "a78c04cff10c148f45bf9c828e07cc9cf01398b48efeb707253273af359a037e"
)
FROZEN_PREFIX_LOGICAL_SHA256 = (
    "e59e9fd7270f5b0a533bd5e4374733a4f68edbbae1edb1ffea4bf21cbb91ecaf"
)


@dataclass(frozen=True)
class SelectionProfile:
    name: str
    target_size: int
    frozen_prefix_size: int


CORESET_12800_PROFILE = SelectionProfile(
    name=CORESET_12800_PROFILE_NAME,
    target_size=TARGET_SIZE,
    frozen_prefix_size=0,
)
VANILLA_51200_PROFILE = SelectionProfile(
    name=VANILLA_51200_PROFILE_NAME,
    target_size=VANILLA_TARGET_SIZE,
    frozen_prefix_size=FROZEN_PREFIX_SIZE,
)
_SELECTION_PROFILES = {
    profile.name: profile
    for profile in (CORESET_12800_PROFILE, VANILLA_51200_PROFILE)
}


def selection_profile(name: str) -> SelectionProfile:
    try:
        return _SELECTION_PROFILES[name]
    except (KeyError, TypeError) as error:
        raise ValueError(f"unsupported selection profile: {name!r}") from error
```

Run the Step 2 command. Expected: four assertions pass.

- [ ] **Step 4: Add failing strict-prefix tests**

Add tests using canonical compact JSONL rows:

```python
def _write_prefix(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def test_frozen_selection_prefix_is_strict_and_exact(tmp_path: Path) -> None:
    rows = [
        {
            "eligible_position": 2,
            "id": "deepmath-level6-000002",
            "original_dataset_index": 12,
            "source_row_index": 2,
        },
        {
            "eligible_position": 0,
            "id": "deepmath-level6-000000",
            "original_dataset_index": 10,
            "source_row_index": 0,
        },
    ]
    path = tmp_path / "selected_ids.jsonl"
    _write_prefix(path, rows)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    loaded, provenance = selection.load_frozen_selection_prefix(
        path, expected_sha256=digest, expected_rows=2
    )
    selection.validate_frozen_selection_prefix(
        rows + [{"id": "later"}], loaded
    )

    assert loaded == rows
    assert provenance == {
        "path": str(path.resolve()),
        "row_count": 2,
        "sha256": digest,
        "selected_id_sequence_sha256": selection._selected_id_sequence_sha256(rows),
    }


def test_frozen_selection_prefix_rejects_hash_and_sequence_mismatch(
    tmp_path: Path,
) -> None:
    row = {
        "eligible_position": 0,
        "id": "deepmath-level6-000000",
        "original_dataset_index": 10,
        "source_row_index": 0,
    }
    path = tmp_path / "selected_ids.jsonl"
    _write_prefix(path, [row])

    with pytest.raises(ValueError, match="SHA-256"):
        selection.load_frozen_selection_prefix(
            path, expected_sha256="0" * 64, expected_rows=1
        )
    with pytest.raises(ValueError, match="exact prefix"):
        selection.validate_frozen_selection_prefix(
            [{**row, "source_row_index": 1}], [row]
        )
```

- [ ] **Step 5: Run the strict-prefix tests and verify RED**

Run:

```bash
/home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest -q \
  math_eval/test_select_gradient_diverse_deepmath.py \
  -k 'frozen_selection_prefix'
```

Expected: import/name failures for the two new helpers.

- [ ] **Step 6: Implement strict prefix loading and comparison**

Implement the public helpers beside `_selected_id_sequence_sha256`:

```python
def load_frozen_selection_prefix(
    path: Path, *, expected_sha256: str, expected_rows: int
) -> tuple[list[dict], dict[str, object]]:
    prefix_path = Path(path).resolve()
    if not prefix_path.is_file():
        raise ValueError(f"frozen selected-ID prefix is missing: {prefix_path}")
    actual_sha256 = sha256_file(prefix_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "frozen selected-ID prefix SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    raw_lines = prefix_path.read_bytes().splitlines(keepends=True)
    if len(raw_lines) != expected_rows or any(not line.endswith(b"\n") for line in raw_lines):
        raise ValueError("frozen selected-ID prefix row count/newline mismatch")
    rows: list[dict] = []
    required = {
        "eligible_position",
        "id",
        "original_dataset_index",
        "source_row_index",
    }
    for line_number, raw_line in enumerate(raw_lines, start=1):
        text = raw_line[:-1].decode("utf-8")
        row = _strict_json_loads(text, f"frozen prefix row {line_number}")
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError(f"frozen prefix row {line_number} has invalid fields")
        canonical = (
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
        if raw_line != canonical:
            raise ValueError(f"frozen prefix row {line_number} is not canonical")
        rows.append(row)
    ids = [row["id"] for row in rows]
    if len(set(ids)) != expected_rows:
        raise ValueError("frozen selected-ID prefix contains duplicate IDs")
    return rows, {
        "path": str(prefix_path),
        "row_count": expected_rows,
        "sha256": actual_sha256,
        "selected_id_sequence_sha256": _selected_id_sequence_sha256(rows),
    }


def validate_frozen_selection_prefix(
    selected_rows: Sequence[Mapping], frozen_rows: Sequence[Mapping]
) -> None:
    prefix_size = len(frozen_rows)
    if [dict(row) for row in selected_rows[:prefix_size]] != [
        dict(row) for row in frozen_rows
    ]:
        raise ValueError("new selection does not preserve the exact frozen prefix")
```

Run the Step 5 command. Expected: both tests pass.

- [ ] **Step 7: Add failing clean-source provenance tests**

Add `import subprocess`, then use a temporary Git repository:

```python
import subprocess


def test_selector_source_provenance_requires_clean_commit(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", repository], check=True)
    subprocess.run(["git", "-C", repository, "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", repository, "config", "user.name", "Test"], check=True)
    source = repository / "selector.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", repository, "add", "selector.py"], check=True)
    subprocess.run(["git", "-C", repository, "commit", "-qm", "fixture"], check=True)

    result = selection.build_selector_source_provenance(
        repository, ("selector.py",)
    )
    assert result["head"] == subprocess.run(
        ["git", "-C", repository, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert result["files"]["selector.py"] == hashlib.sha256(source.read_bytes()).hexdigest()

    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clean"):
        selection.build_selector_source_provenance(repository, ("selector.py",))
```

Run it and verify RED because `build_selector_source_provenance` is absent.

- [ ] **Step 8: Implement clean source provenance**

Add:

```python
SELECTION_SOURCE_FILES = (
    "math_eval/deepmath_gradient_diversity.py",
    "math_eval/select_gradient_diverse_deepmath.py",
)


def build_selector_source_provenance(
    repository: Path, source_files: Sequence[str] = SELECTION_SOURCE_FILES
) -> dict[str, object]:
    root = Path(repository).resolve()
    status = _git_output(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise ValueError("selector repository must be clean")
    head = _git_output(root, "rev-parse", "HEAD")
    tree = _git_output(root, "rev-parse", "HEAD^{tree}")
    files: dict[str, str] = {}
    for relative in source_files:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"selector source file is missing: {relative}")
        files[relative] = sha256_file(path)
    return {
        "repository": str(root),
        "head": head,
        "tree": tree,
        "files": files,
    }
```

Run the focused test. Expected: pass.

- [ ] **Step 9: Add failing old/new manifest-contract tests**

Factor the selection-specific record through a new helper and test that old bytes remain structurally compatible:

```python
def test_selection_contract_preserves_old_shape_and_binds_new_prefix() -> None:
    old = selection.build_selection_contract(
        CORESET_12800_PROFILE,
        prefix_provenance=None,
        source_provenance=None,
    )
    assert old == {
        "selection": {
            "method": "balanced_round_robin",
            "seed": 42,
            "target_size": 12_800,
        }
    }

    prefix = {
        "path": "/frozen/selected_ids.jsonl",
        "row_count": 12_800,
        "sha256": selection.FROZEN_PREFIX_SELECTED_IDS_SHA256,
        "selected_id_sequence_sha256": selection.FROZEN_PREFIX_LOGICAL_SHA256,
    }
    source = {"head": "a" * 40, "tree": "b" * 40, "files": {}}
    new = selection.build_selection_contract(
        VANILLA_51200_PROFILE,
        prefix_provenance=prefix,
        source_provenance=source,
    )
    assert new["selection"]["profile"] == "vanilla_51200"
    assert new["selection"]["frozen_prefix"] == prefix
    assert new["selection"]["target_size"] == 51_200
    assert new["selection_source"] == source
```

Run and verify RED.

- [ ] **Step 10: Implement profile-aware manifest construction and wire `run_selection`**

Add:

```python
def build_selection_contract(
    profile: SelectionProfile,
    *,
    prefix_provenance: Mapping[str, object] | None,
    source_provenance: Mapping[str, object] | None,
) -> dict[str, object]:
    selection = {
        "method": "balanced_round_robin",
        "seed": SELECTION_SEED,
        "target_size": profile.target_size,
    }
    if profile == CORESET_12800_PROFILE:
        if prefix_provenance is not None or source_provenance is not None:
            raise ValueError("old selection profile forbids new provenance fields")
        return {"selection": selection}
    if prefix_provenance is None or source_provenance is None:
        raise ValueError("vanilla_51200 requires prefix and source provenance")
    selection.update(
        {
            "profile": profile.name,
            "frozen_prefix": dict(prefix_provenance),
        }
    )
    return {
        "selection": selection,
        "selection_source": dict(source_provenance),
    }
```

Change `run_selection` so it:

1. resolves `profile = selection_profile(selection_profile_name)`;
2. requires `target_size == profile.target_size`;
3. forbids a prefix path for `coreset_12800`;
4. for `vanilla_51200`, loads the prefix with the two frozen SHA constants before moving gradients to CUDA;
5. includes the prefix file in `_validate_selection_output_isolation` inputs;
6. validates `selected_rows[:12_800]` before publication;
7. obtains clean source provenance from `Path(__file__).resolve().parents[1]`; and
8. merges `build_selection_contract(...)` into `base_manifest` instead of writing the old inline `selection` block.

Extend `_build_parser()` with:

```python
parser.add_argument(
    "--selection-profile",
    choices=tuple(_SELECTION_PROFILES),
    default=CORESET_12800_PROFILE_NAME,
)
parser.add_argument("--frozen-prefix-selected-ids", type=Path)
```

Pass both arguments through `main()`. Keep `--target-size` defaulted to 12,800 so the existing launcher and old manifest validation remain unchanged.

- [ ] **Step 11: Run selector regression tests**

Run:

```bash
/home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest -q \
  math_eval/test_deepmath_gradient_diversity.py \
  math_eval/test_select_gradient_diverse_deepmath.py
/home/mchen/miniconda3/envs/gvendi-opd/bin/python -m compileall -q \
  math_eval/select_gradient_diverse_deepmath.py
```

Expected: all tests pass; compileall is silent.

- [ ] **Step 12: Commit Task 1**

```bash
git add math_eval/select_gradient_diverse_deepmath.py \
  math_eval/test_select_gradient_diverse_deepmath.py
git commit -m "feat: add 51200 gradient selection profile"
```

---

### Task 2: Add the Selection-Only Production Launcher

**Files:**
- Create: `run_select_gradient_diverse_deepmath_51200.sh`
- Create: `math_eval/test_gradient_diverse_51200_launcher.py`

**Interfaces:**
- `SFTGRAD51200_DRY_RUN=1`: print the exact immutable contract without requiring artifacts, Slurm, tmux, or GPUs.
- `SFTGRAD51200_PREFLIGHT_ONLY=1`: validate hashes, coverage, output state, and clean source, then stop before GPU work.
- Default: validate an active four-token `opd-CLI` allocation, expose only its first token to official CUDA K-means, publish the four new artifacts, and print their hashes.

- [ ] **Step 1: Write failing dry-run and pin-rejection tests**

Create `math_eval/test_gradient_diverse_51200_launcher.py`:

```python
import os
from pathlib import Path
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "run_select_gradient_diverse_deepmath_51200.sh"


def _run(**overrides):
    env = os.environ.copy()
    env.update(
        {
            "SFTGRAD51200_DRY_RUN": "1",
            "PYTHON_BIN": "/not/used/in/dry-run",
            **overrides,
        }
    )
    return subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_51200_selection_dry_run_uses_only_frozen_gradients() -> None:
    completed = _run()
    assert completed.returncode == 0, completed.stderr
    required = [
        "selection_profile=vanilla_51200",
        "target_rows=51200",
        "expected_source_rows=57046",
        "expected_eligible_rows=57045",
        "selection_51200/selected_ids.jsonl",
        "train_gradient_diverse_51200.parquet",
        "--selection-profile vanilla_51200",
        "--target-size 51200",
        "--frozen-prefix-selected-ids",
        "math_eval.select_gradient_diverse_deepmath",
    ]
    for value in required:
        assert value in completed.stdout
    assert "collect_prismatic_gradients" not in completed.stdout
    assert "prepare_deepmath_gradient_pool" not in completed.stdout


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("TARGET_ROWS", "12800"),
        ("SELECTION_PROFILE", "coreset_12800"),
        ("PRIMARY_CLUSTER_RATIO", "0.01"),
        ("CLUSTER_SEEDS", "42"),
        ("SOURCE_SHA256", "0" * 64),
        ("PREPARED_JSONL_SHA256", "0" * 64),
        ("REFERENCE_COMMIT", "0" * 40),
    ],
)
def test_51200_selection_rejects_changed_pin(name: str, value: str) -> None:
    completed = _run(**{name: value})
    assert completed.returncode == 2
    assert f"{name} must remain pinned" in completed.stderr
```

- [ ] **Step 2: Run tests and verify RED**

```bash
/home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest -q \
  math_eval/test_gradient_diverse_51200_launcher.py
```

Expected: tests fail because the launcher is absent.

- [ ] **Step 3: Implement constants, dry-run, and exact command rendering**

Create `run_select_gradient_diverse_deepmath_51200.sh` with `set -euo pipefail`, `die`, and `require_pinned_value`. Pin these values:

```bash
CODE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PRODUCTION_ROOT="/home/mchen/FiRe-OPD"
PYTHON_BIN="${PYTHON_BIN:-/home/mchen/miniconda3/envs/gvendi-opd/bin/python}"
REFERENCE_REPO="${REFERENCE_REPO:-/home/mchen/prismatic-synthesis-reference}"
SELECTION_PROFILE="${SELECTION_PROFILE:-vanilla_51200}"
TARGET_ROWS="${TARGET_ROWS:-51200}"
EXPECTED_SOURCE_ROWS="${EXPECTED_SOURCE_ROWS:-57046}"
EXPECTED_ELIGIBLE_ROWS="${EXPECTED_ELIGIBLE_ROWS:-57045}"
SOURCE_SHA256="${SOURCE_SHA256:-de3350fdd00bc0410550098ea65179e2be873da99e4075f80de575fc17670597}"
PREPARED_JSONL_SHA256="${PREPARED_JSONL_SHA256:-ee8d55c943577888f14052b3953a0a8aa07d26076e50ddffa41a102b60027344}"
PREPARED_MANIFEST_SHA256="${PREPARED_MANIFEST_SHA256:-5add2e965473d3647f2318c73f957434fbe23344516089be9adfe89db9c6a70e}"
ELIGIBILITY_REPORT_SHA256="${ELIGIBILITY_REPORT_SHA256:-cc8d5b888def37761519e15c6ef722e5a22d91ae2342c6d669f008bcff3dcbb6}"
GRADIENT_MANIFEST_SHA256="${GRADIENT_MANIFEST_SHA256:-9a534118a08e736a15e806933d5cf90c2a99822fee28bf18644e5ff344ce3ae2}"
REFERENCE_COMMIT="${REFERENCE_COMMIT:-d9484cd3b5991030b901ac4a3a9e2472dbfac2ad}"
REFERENCE_TREE="${REFERENCE_TREE:-a0079d8c5e15cb18bb4790f99c43cc19bb9ecd50}"
OLD_PARQUET_SHA256="${OLD_PARQUET_SHA256:-caf303c5d151fdaed2e21eebc257917f13660c589906e3da0e569ffdfd59b059}"
OLD_MANIFEST_SHA256="${OLD_MANIFEST_SHA256:-a1a45382ee577e24f9b455386f9f3adcab210ff40a83ea760c2110fb96f8f90a}"
OLD_SELECTED_IDS_SHA256="${OLD_SELECTED_IDS_SHA256:-a78c04cff10c148f45bf9c828e07cc9cf01398b48efeb707253273af359a037e}"
OLD_DIAGNOSTICS_SHA256="${OLD_DIAGNOSTICS_SHA256:-d2105179f2ce5a7c796aef07136bbc1dce79b07b5bc2de4d2319b18b47ea761f}"
PRIMARY_CLUSTER_RATIO="${PRIMARY_CLUSTER_RATIO:-0.10}"
SENSITIVITY_CLUSTER_RATIO="${SENSITIVITY_CLUSTER_RATIO:-0.01}"
CLUSTER_SEEDS="${CLUSTER_SEEDS:-42,43}"
```

Derive all absolute input/output paths under `PRODUCTION_ROOT`. Render the command as a Bash array containing exactly:

```bash
selection_command=(
  "${PYTHON_BIN}" -m math_eval.select_gradient_diverse_deepmath
  --source-parquet "${SOURCE_PARQUET}"
  --prepared-jsonl "${PREPARED_JSONL}"
  --prepared-manifest "${PREPARED_MANIFEST}"
  --eligibility-report "${ELIGIBILITY_REPORT}"
  --gradient-dir "${GRADIENT_DIR}"
  --reference-repo "${REFERENCE_REPO}"
  --device cuda:0
  --output-parquet "${OUTPUT_PARQUET}"
  --selected-ids "${SELECTED_IDS}"
  --diagnostics "${DIAGNOSTICS}"
  --manifest "${SELECTION_MANIFEST}"
  --expected-source-count 57046
  --expected-eligible-count 57045
  --target-size 51200
  --selection-profile vanilla_51200
  --frozen-prefix-selected-ids "${OLD_SELECTED_IDS}"
)
```

For dry-run, print all paths/pins and the shell-escaped command, then exit before file or runtime checks. Run the Step 2 command. Expected: all dry-run tests pass.

- [ ] **Step 4: Add static tests for non-dry gates**

Add:

```python
def test_51200_selection_launcher_contains_fail_closed_runtime_gates() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    required = [
        "SFTGRAD51200_PREFLIGHT_ONLY",
        "sha256sum",
        "flock -n",
        "validate_opd_cli_runtime",
        "expected_gpus=4",
        "tmux",
        "SLURM_JOB_ID",
        "CUDA_VISIBLE_DEVICES",
        "git status --porcelain=v1",
        "--validate-global-only",
    ]
    for value in required:
        assert value in text
    assert "pip install" not in text
    assert "conda install" not in text
    assert "srun " not in text
```

Run and verify RED because the gates are not implemented.

- [ ] **Step 5: Implement CPU preflight, lock, provenance, and GPU gate**

In non-dry mode:

1. require executable Python and all frozen input files/directories;
2. require a clean `git -C "$CODE_ROOT" status --porcelain=v1 --untracked-files=all`;
3. verify source, prepared JSONL, prepared manifest, eligibility report, gradient manifest, and all four old artifacts with `require_sha256 PATH EXPECTED`;
4. require the reference checkout to be clean and require `git rev-parse HEAD` / `HEAD^{tree}` to equal `REFERENCE_COMMIT` / `REFERENCE_TREE`;
5. require the four new output files to be either all missing or all present;
6. acquire `/home/mchen/FiRe-OPD/data/gradient_diversity/.deepmath_gradient_diverse_51200.launch.lock` with `flock -n`;
7. invoke the existing global coverage validator with `--validate-global-only`, exact prepared/eligibility paths, prefix `deepmath`, and four shards;
8. print Git HEAD/tree/status hash, every frozen input hash, reference HEAD/tree, Python/package versions, and the resolved command;
9. exit zero at `SFTGRAD51200_PREFLIGHT_ONLY=1`;
10. otherwise require nonempty `SLURM_JOB_ID`, call `validate_opd_cli_runtime(expected_gpus=4, require_idle=True)`, and require tmux session `opd-CLI` through that function;
11. select the first inherited `CUDA_VISIBLE_DEVICES` token, remove `ROCR_VISIBLE_DEVICES`/`HIP_VISIBLE_DEVICES`, and invoke `selection_command` with only that token visible;
12. recheck all frozen inputs and all four old artifact hashes after selection; and
13. print SHA-256 for all four new outputs.

Use this runtime gate without adding another dependency:

```bash
"${PYTHON_BIN}" - <<'PY'
from math_eval.run_opd_proxy_gradient_verify import validate_opd_cli_runtime
print(",".join(validate_opd_cli_runtime(expected_gpus=4, require_idle=True)))
PY
```

Use the log:

```text
/home/mchen/FiRe-OPD/logs/gradient_diversity/deepmath_gradient_diverse_51200.log
```

Run the static test and `bash -n`; both must pass.

- [ ] **Step 6: Run selection launcher regressions**

```bash
/home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest -q \
  math_eval/test_gradient_diversity_launcher.py \
  math_eval/test_gradient_diverse_51200_launcher.py
SFTGRAD51200_DRY_RUN=1 bash run_select_gradient_diverse_deepmath_51200.sh
bash -n run_select_gradient_diverse_deepmath_51200.sh
```

Expected: tests pass; dry-run contains no collection/preparation command.

- [ ] **Step 7: Commit Task 2**

```bash
git add run_select_gradient_diverse_deepmath_51200.sh \
  math_eval/test_gradient_diverse_51200_launcher.py
git commit -m "feat: launch frozen-gradient 51200 selection"
```

---

### Task 3: Extend Artifact Validation for the New Profile and Prefix

**Files:**
- Modify: `math_eval/validate_gradient_diverse_training_data.py`
- Modify: `math_eval/test_validate_gradient_diverse_training_data.py`

**Interfaces:**
- Extends `validate_training_artifact` with optional profile/prefix arguments while leaving existing callers and report fields unchanged when those arguments are omitted.
- Adds CLI flags `--expected-selection-profile`, `--frozen-prefix-selected-ids`, `--expected-frozen-prefix-sha256`, and `--expected-frozen-prefix-rows`.

- [ ] **Step 1: Add a failing profile/prefix happy-path test**

Add this profile fixture and happy-path test after `_validate_fixture`:

```python
def _vanilla_profile_fixture(tmp_path: Path):
    source, selected, ids, diagnostics, manifest_path, manifest = _artifacts(
        tmp_path, [_row(0), _row(1), _row(2)], [2, 0]
    )
    prefix_path = tmp_path / "prefix.jsonl"
    prefix_path.write_text(
        ids.read_text(encoding="utf-8").splitlines()[0] + "\n",
        encoding="utf-8",
    )
    prefix_sha = _sha256(prefix_path)
    prefix_rows = [json.loads(prefix_path.read_text(encoding="utf-8"))]
    manifest["selection"] = {
        "method": "balanced_round_robin",
        "seed": 42,
        "target_size": 2,
        "profile": "vanilla_51200",
        "frozen_prefix": {
            "path": str(prefix_path.resolve()),
            "row_count": 1,
            "sha256": prefix_sha,
            "selected_id_sequence_sha256": validation._id_sequence_sha256(
                prefix_rows
            ),
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    paths = (source, selected, ids, diagnostics, manifest_path)
    profile_arguments = {
        "expected_selection_profile": "vanilla_51200",
        "frozen_prefix_selected_ids_path": prefix_path,
        "expected_frozen_prefix_sha256": prefix_sha,
        "expected_frozen_prefix_rows": 1,
    }
    return paths, manifest, prefix_path, profile_arguments


def test_validator_accepts_vanilla_profile_and_rechecks_exact_prefix(
    tmp_path: Path,
) -> None:
    paths, manifest, prefix_path, profile_arguments = _vanilla_profile_fixture(
        tmp_path
    )

    report = _validate_fixture(
        paths,
        manifest,
        FakeTokenizer(),
        **profile_arguments,
    )

    assert report["selection_profile"] == "vanilla_51200"
    assert report["frozen_prefix_rows"] == 1
    assert report["frozen_prefix_path"] == str(prefix_path.resolve())
    assert report["frozen_prefix_sha256"] == _sha256(prefix_path)
```

- [ ] **Step 2: Run the new test and verify RED**

```bash
/home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest -q \
  math_eval/test_validate_gradient_diverse_training_data.py \
  -k 'vanilla_profile'
```

Expected: `TypeError` for unexpected keyword arguments.

- [ ] **Step 3: Specify the optional profile/prefix acceptance contract**

Before editing production code, use these exact optional keyword arguments as the acceptance contract for the remaining failing tests:

```python
expected_selection_profile: str | None = None,
frozen_prefix_selected_ids_path: Path | None = None,
expected_frozen_prefix_sha256: str | None = None,
expected_frozen_prefix_rows: int | None = None,
```

Require either all four are absent or all four are present. In the present case:

1. require the prefix path is a regular file and its SHA matches;
2. load it through `_load_selected_ids` and require the exact row count;
3. compare `id_rows[:expected_frozen_prefix_rows]` to the prefix records exactly;
4. require `manifest["selection"]` is an object with matching `profile` and `target_size`;
5. require `selection["frozen_prefix"]` exactly matches path, row count, file SHA, and `_id_sequence_sha256(prefix_rows)`; and
6. append `selection_profile`, `frozen_prefix_rows`, `frozen_prefix_path`, and `frozen_prefix_sha256` to the report only in this mode.

The implementation will also add corresponding optional CLI arguments and pass them through `main()`. Do not edit production code until Step 5.

- [ ] **Step 4: Add failing corruption tests**

Add these three exact cases:

```python
def test_validator_rejects_frozen_prefix_file_hash_mismatch(tmp_path: Path) -> None:
    paths, manifest, _, arguments = _vanilla_profile_fixture(tmp_path)
    arguments["expected_frozen_prefix_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="prefix SHA-256"):
        _validate_fixture(paths, manifest, FakeTokenizer(), **arguments)


def test_validator_rejects_selected_ids_that_do_not_start_with_prefix(
    tmp_path: Path,
) -> None:
    paths, manifest, _, arguments = _vanilla_profile_fixture(tmp_path)
    source, selected, ids, diagnostics, manifest_path = paths
    rows = [
        json.loads(line)
        for line in ids.read_text(encoding="utf-8").splitlines()
    ]
    rows.reverse()
    ids.write_text(
        "".join(
            json.dumps(row, separators=(",", ":")) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    manifest["selected_ids_sha256"] = _sha256(ids)
    manifest["selected_id_sequence_sha256"] = validation._id_sequence_sha256(rows)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="exact prefix"):
        _validate_fixture(
            (source, selected, ids, diagnostics, manifest_path),
            manifest,
            FakeTokenizer(),
            **arguments,
        )


def test_validator_rejects_selection_profile_mismatch(tmp_path: Path) -> None:
    paths, manifest, _, arguments = _vanilla_profile_fixture(tmp_path)
    manifest_path = paths[-1]
    manifest["selection"]["profile"] = "coreset_12800"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="selection profile"):
        _validate_fixture(paths, manifest, FakeTokenizer(), **arguments)
```

Run all four new profile tests and verify RED because the production function does not yet accept the optional arguments:

```bash
/home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest -q \
  math_eval/test_validate_gradient_diverse_training_data.py \
  -k 'vanilla_profile or frozen_prefix or exact_prefix or selection_profile'
```

Expected: every selected case fails for the absent profile API, not because the fixtures crash.

- [ ] **Step 5: Implement the profile/prefix acceptance contract**

Now add the four keyword arguments, all six validation/report requirements, and the four CLI flags specified in Step 3. Run the Step 4 command. Expected: all four profile/prefix tests pass.

- [ ] **Step 6: Prove old report compatibility**

Run the entire existing validator test file and assert the original happy-path report has no `selection_profile` or `frozen_prefix_*` fields:

```python
assert "selection_profile" not in report
assert not any(key.startswith("frozen_prefix") for key in report)
```

Run:

```bash
/home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest -q \
  math_eval/test_validate_gradient_diverse_training_data.py
/home/mchen/miniconda3/envs/gvendi-opd/bin/python -m compileall -q \
  math_eval/validate_gradient_diverse_training_data.py
```

Expected: all pass.

- [ ] **Step 7: Commit Task 3**

```bash
git add math_eval/validate_gradient_diverse_training_data.py \
  math_eval/test_validate_gradient_diverse_training_data.py
git commit -m "feat: validate 51200 selection prefix contract"
```

---

### Task 4: Add the Canonical Vanilla OPD Launcher

**Files:**
- Create: `run_train_vanilla_opd.sh`
- Create: `verl/tests/trainer/ppo/test_vanilla_opd_launcher.py`

**Interfaces:**
- `VANILLA_OPD_DRY_RUN=1`: print all resolved scientific fields and one shell-escaped `python -m verl.trainer.main_ppo ...` command.
- Default: execute exactly that command.
- Accepts data/name/checkpoint paths from the candidate wrapper; rejects changes to scientific volume or algorithm settings.

- [ ] **Step 1: Write failing canonical-contract tests**

Create `verl/tests/trainer/ppo/test_vanilla_opd_launcher.py`:

```python
import os
from pathlib import Path
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
LAUNCHER = REPO_ROOT / "run_train_vanilla_opd.sh"


def _run(**overrides):
    env = os.environ.copy()
    env.update({"VANILLA_OPD_DRY_RUN": "1", **overrides})
    return subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_vanilla_launcher_pins_canonical_step50_contract() -> None:
    completed = _run()
    assert completed.returncode == 0, completed.stderr
    required = [
        "ROLLOUT_N=1",
        "PROMPT_BATCH_SIZE=1024",
        "PPO_MINI_BATCH_SIZE=1024",
        "TOTAL_TRAJECTORIES=1024",
        "TOTAL_TRAINING_STEPS=50",
        "data.train_batch_size=1024",
        "data.seed=42",
        "data.max_response_length=16384",
        "actor_rollout_ref.rollout.n=1",
        "actor_rollout_ref.actor.ppo_mini_batch_size=1024",
        "actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=True",
        "actor_rollout_ref.actor.loss_agg_mode=token-mean",
        "algorithm.rollout_correction.rollout_is=token",
        "algorithm.rollout_correction.rollout_is_threshold=5.0",
        "algorithm.tale_budget.enabled=False",
        "algorithm.difficulty_aware_opd.enabled=False",
        "algorithm.candidate_selection.enabled=False",
        "algorithm.rethinking_opd_probe.enabled=False",
        "actor_rollout_ref.actor.policy_loss.length_aware_opd=False",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.kl_loss_coef=0",
        "algorithm.use_kl_in_reward=False",
        "trainer.val_before_train=True",
        "trainer.test_freq=10",
        "trainer.save_freq=50",
        "trainer.total_training_steps=50",
        "trainer.total_epochs=3",
        "trainer.resume_mode=disable",
    ]
    for value in required:
        assert value in completed.stdout
    assert "ref_raw_prompt_key" not in completed.stdout


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("ROLLOUT_N", "4"),
        ("PROMPT_BATCH_SIZE", "256"),
        ("PPO_MINI_BATCH_SIZE", "256"),
        ("TOTAL_TRAJECTORIES", "2048"),
        ("TOTAL_TRAINING_STEPS", "51"),
        ("SAVE_FREQ", "20"),
        ("TEST_FREQ", "-1"),
        ("N_GPUS_PER_NODE", "8"),
        ("ROLLOUT_TP_SIZE", "2"),
    ],
)
def test_vanilla_launcher_rejects_changed_scientific_pin(name: str, value: str) -> None:
    completed = _run(**{name: value})
    assert completed.returncode == 2
    assert f"{name} must remain pinned" in completed.stderr
```

- [ ] **Step 2: Run tests and verify RED**

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  verl/tests/trainer/ppo/test_vanilla_opd_launcher.py
```

Expected: failures because the launcher is absent.

- [ ] **Step 3: Implement environment validation and dry-run fields**

Create `run_train_vanilla_opd.sh` with `set -euo pipefail`, `die`, and `require_pinned_value`. Resolve and pin:

```bash
REPO_DIR="${REPO_DIR:-/home/mchen/FiRe-OPD}"
PYTHON_BIN="${PYTHON_BIN:-/home/mchen/miniconda3/envs/verl/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/mchen/FiRe-OPD/data/g-opd}"
TRAIN_DATA="${TRAIN_DATA:-${DATA_ROOT}/DeepMath-103K/train_filtered_level6.parquet}"
VAL_DATA="${VAL_DATA:-['${DATA_ROOT}/AIME2024/test.parquet', '${DATA_ROOT}/AIME2025/test.parquet']}"
STUDENT_MODEL="${STUDENT_MODEL:-/home/mchen/FiRe-OPD/models/Qwen3-4B}"
TEACHER_MODEL="${TEACHER_MODEL:-/home/mchen/FiRe-OPD/models/Qwen3-30B-A3B-Instruct-2507}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-opd-strong-to-weak-rawprompt-4gpu-tp4-refmb4-rollmb4}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/home/mchen/FiRe-OPD/checkpoints/${EXPERIMENT_NAME}}"
ROLLOUT_N="${ROLLOUT_N:-1}"
PROMPT_BATCH_SIZE="${PROMPT_BATCH_SIZE:-1024}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-1024}"
TOTAL_TRAJECTORIES="${TOTAL_TRAJECTORIES:-1024}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-50}"
SAVE_FREQ="${SAVE_FREQ:-50}"
TEST_FREQ="${TEST_FREQ:-10}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-4}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
```

Require every numeric scientific value to equal the approved constant and verify `PROMPT_BATCH_SIZE * ROLLOUT_N == TOTAL_TRAJECTORIES`. Export `PYTHONPATH="${REPO_DIR}/verl:${REPO_DIR}:${PYTHONPATH:-}"`. Dry-run must occur before checking model/data existence so tests remain artifact-independent.

- [ ] **Step 4: Implement the exact Hydra argument array**

Use a Bash array whose scientific entries are exactly:

```bash
fixed_args=(
  "algorithm.adv_estimator=grpo"
  "algorithm.rollout_correction.rollout_is=token"
  "algorithm.rollout_correction.rollout_is_threshold=5.0"
  "algorithm.rollout_correction.rollout_rs=null"
  "algorithm.rollout_correction.bypass_mode=false"
  "algorithm.tale_budget.enabled=False"
  "algorithm.difficulty_aware_opd.enabled=False"
  "algorithm.candidate_selection.enabled=False"
  "algorithm.rethinking_opd_probe.enabled=False"
  "algorithm.use_kl_in_reward=False"
  "actor_rollout_ref.rollout.calculate_log_probs=true"
  "data.train_files=${TRAIN_DATA}"
  "data.val_files=${VAL_DATA}"
  "data.train_batch_size=1024"
  "data.max_prompt_length=2048"
  "data.max_response_length=16384"
  "data.filter_overlong_prompts=True"
  "data.truncation=error"
  "data.shuffle=True"
  "data.seed=42"
  "data.return_raw_chat=True"
  "+data.apply_chat_template_kwargs.enable_thinking=False"
  "actor_rollout_ref.model.path=${STUDENT_MODEL}"
  "+actor_rollout_ref.ref.model.path=${TEACHER_MODEL}"
  "actor_rollout_ref.actor.optim.lr=1e-6"
  "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0"
  "actor_rollout_ref.model.use_remove_padding=True"
  "actor_rollout_ref.model.enable_gradient_checkpointing=True"
  "actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=True"
  "actor_rollout_ref.actor.policy_loss.length_aware_opd=False"
  "actor_rollout_ref.actor.loss_agg_mode=token-mean"
  "actor_rollout_ref.actor.ppo_mini_batch_size=1024"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.actor.use_kl_loss=True"
  "actor_rollout_ref.actor.kl_loss_coef=0"
  "actor_rollout_ref.actor.kl_loss_type=low_var_kl"
  "actor_rollout_ref.actor.entropy_coeff=0"
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768"
  "actor_rollout_ref.actor.fsdp_config.param_offload=False"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=4"
  "actor_rollout_ref.rollout.name=vllm"
  "actor_rollout_ref.rollout.gpu_memory_utilization=0.6"
  "actor_rollout_ref.rollout.n=1"
  "actor_rollout_ref.rollout.max_num_batched_tokens=32768"
  "actor_rollout_ref.rollout.temperature=1.0"
  "actor_rollout_ref.rollout.top_p=1.0"
  "actor_rollout_ref.rollout.val_kwargs.do_sample=True"
  "actor_rollout_ref.rollout.val_kwargs.temperature=1.0"
  "actor_rollout_ref.rollout.val_kwargs.top_p=1.0"
  "actor_rollout_ref.rollout.val_kwargs.n=8"
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4"
  "actor_rollout_ref.ref.fsdp_config.param_offload=True"
  "reward_model.reward_manager=naive"
  "trainer.critic_warmup=0"
  "trainer.val_before_train=True"
  "trainer.logger=[\"console\",\"wandb\"]"
  "trainer.log_val_generations=10"
  "trainer.project_name=fire-opd"
  "trainer.experiment_name=${EXPERIMENT_NAME}"
  "trainer.n_gpus_per_node=4"
  "trainer.nnodes=1"
  "trainer.save_freq=50"
  "trainer.default_local_dir=${CHECKPOINT_DIR}"
  "trainer.test_freq=10"
  "trainer.total_epochs=3"
  "trainer.total_training_steps=50"
  "trainer.resume_mode=disable"
  "ray_kwargs.ray_init.num_cpus=${RAY_NUM_CPUS:-${SLURM_CPUS_PER_TASK:-16}}"
)
```

Dry-run prints resolved fields followed by:

```bash
printf '%q ' "${PYTHON_BIN}" -m verl.trainer.main_ppo "${fixed_args[@]}"
printf '\n'
```

Non-dry mode verifies files/executables, removes ROCm visibility aliases, changes to `REPO_DIR`, and `exec`s the same array. Do not append `"$@"`; the canonical launcher rejects unreviewed Hydra overrides.

- [ ] **Step 5: Run launcher tests and syntax checks**

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  verl/tests/trainer/ppo/test_vanilla_opd_launcher.py
VANILLA_OPD_DRY_RUN=1 bash run_train_vanilla_opd.sh
bash -n run_train_vanilla_opd.sh
```

Expected: all pass; dry-run contains no `ref_raw_prompt_key`.

- [ ] **Step 6: Commit Task 4**

```bash
git add run_train_vanilla_opd.sh \
  verl/tests/trainer/ppo/test_vanilla_opd_launcher.py
git commit -m "feat: add canonical Vanilla OPD launcher"
```

---

### Task 5: Add the Fail-Closed 51,200-Data Training Wrapper

**Files:**
- Create: `run_train_vanilla_sft_gradient_51200.sh`
- Create: `verl/tests/trainer/ppo/test_vanilla_sft_gradient_51200_launcher.py`

**Interfaces:**
- `VANILLA_SFTGRAD_DRY_RUN=1`: prove baseline/candidate command equivalence without reading artifacts.
- `VANILLA_SFTGRAD_PREFLIGHT_ONLY=1`: inspect manifest hashes, run the strict tokenizer/artifact validator, print provenance, and stop before GPUs.
- Default: additionally gate the four-token `opd-CLI` runtime, lock the experiment, tee the log, and delegate to `run_train_vanilla_opd.sh`.

- [ ] **Step 1: Write failing dry-run equivalence and pin tests**

Create `verl/tests/trainer/ppo/test_vanilla_sft_gradient_51200_launcher.py`:

```python
import os
from pathlib import Path
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
LAUNCHER = REPO_ROOT / "run_train_vanilla_sft_gradient_51200.sh"
RUN_NAME = "opd-strong-to-weak-sftgrad51200-rawprompt-4gpu-tp4-refmb4-rollmb4"


def _run(**overrides):
    env = os.environ.copy()
    env.update(
        {
            "VANILLA_SFTGRAD_DRY_RUN": "1",
            "PYTHON_BIN": "/not/used/in/dry-run",
            "REPO_DIR": str(REPO_ROOT),
            **overrides,
        }
    )
    return subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_candidate_dry_run_is_a_data_only_vanilla_contract() -> None:
    completed = _run()
    assert completed.returncode == 0, completed.stderr
    required = [
        "single_variable_contract=PASS",
        "artifact_validation=SKIPPED_DRY_RUN",
        f"EXPERIMENT_NAME={RUN_NAME}",
        "train_gradient_diverse_51200.parquet",
        "ROLLOUT_N=1",
        "PROMPT_BATCH_SIZE=1024",
        "TOTAL_TRAINING_STEPS=50",
        "trainer.val_before_train=True",
        "trainer.test_freq=10",
        "trainer.resume_mode=disable",
        "bash " + str(REPO_ROOT / "run_train_vanilla_opd.sh"),
    ]
    for value in required:
        assert value in completed.stdout


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("TRAIN_DATA", "/tmp/other.parquet"),
        ("EXPERIMENT_NAME", "other"),
        ("ROLLOUT_N", "4"),
        ("PROMPT_BATCH_SIZE", "256"),
        ("TOTAL_TRAINING_STEPS", "49"),
        ("STUDENT_MODEL", "/tmp/student"),
        ("TEACHER_MODEL", "/tmp/teacher"),
    ],
)
def test_candidate_rejects_changed_pin(name: str, value: str) -> None:
    completed = _run(**{name: value})
    assert completed.returncode == 2
    assert f"{name} must remain pinned" in completed.stderr
```

- [ ] **Step 2: Run tests and verify RED**

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  verl/tests/trainer/ppo/test_vanilla_sft_gradient_51200_launcher.py
```

Expected: failures because the wrapper is absent.

- [ ] **Step 3: Implement pins and normalized command comparison**

Create the wrapper with these identities:

```bash
PRODUCTION_ROOT="/home/mchen/FiRe-OPD"
REPO_DIR="${REPO_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)}"
BASE_LAUNCHER="${REPO_DIR}/run_train_vanilla_opd.sh"
SOURCE_DATA="${PRODUCTION_ROOT}/data/g-opd/DeepMath-103K/train_filtered_level6.parquet"
TRAIN_DATA="${TRAIN_DATA:-${PRODUCTION_ROOT}/data/gradient_diversity/DeepMath-103K/train_gradient_diverse_51200.parquet}"
SELECTION_MANIFEST="${PRODUCTION_ROOT}/data/gradient_diversity/selection_51200/manifest.json"
SELECTED_IDS="${PRODUCTION_ROOT}/data/gradient_diversity/selection_51200/selected_ids.jsonl"
DIAGNOSTICS="${PRODUCTION_ROOT}/data/gradient_diversity/selection_51200/diagnostics.json"
FROZEN_PREFIX="${PRODUCTION_ROOT}/data/gradient_diversity/selection/selected_ids.jsonl"
BASELINE_EXPERIMENT="opd-strong-to-weak-rawprompt-4gpu-tp4-refmb4-rollmb4"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-opd-strong-to-weak-sftgrad51200-rawprompt-4gpu-tp4-refmb4-rollmb4}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PRODUCTION_ROOT}/checkpoints/${EXPERIMENT_NAME}}"
LOG_FILE="${LOG_FILE:-${PRODUCTION_ROOT}/logs/gradient_diversity/${EXPERIMENT_NAME}.log}"
```

Pin all scientific values from Task 4 and all paths/names above. Render baseline and candidate contracts by invoking `VANILLA_OPD_DRY_RUN=1 bash "$BASE_LAUNCHER"` twice. Normalize only source/selected data paths, baseline/candidate names, and checkpoint paths to literal markers; reject any remaining diff. Dry-run prints the candidate contract and delegation line, then exits before artifacts or GPUs.

Run the Step 2 tests. Expected: all pass.

- [ ] **Step 4: Add static non-dry gate tests**

Add one source inspection test requiring:

```python
def test_candidate_wrapper_contains_artifact_provenance_and_gpu_gates() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    required = [
        "validate_gradient_diverse_training_data",
        "--expected-selection-profile",
        "vanilla_51200",
        "--frozen-prefix-selected-ids",
        "--expected-frozen-prefix-rows",
        "12800",
        "--checkpoint-dir",
        "git status --porcelain=v1",
        "validate_opd_cli_runtime",
        "expected_gpus=4",
        "flock -n",
        "VANILLA_SFTGRAD_PREFLIGHT_ONLY",
    ]
    for value in required:
        assert value in text
    assert "pip install" not in text
    assert "conda install" not in text
    assert "srun " not in text
```

Run and verify RED.

- [ ] **Step 5: Implement dynamic immutable hash extraction and strict validation**

In non-dry mode:

1. require a clean committed `REPO_DIR`;
2. require the four selection artifacts and frozen prefix;
3. independently pin source SHA and frozen-prefix SHA;
4. use `/home/mchen/miniconda3/envs/verl/bin/python` with duplicate-key/nonfinite rejection to load `manifest.json` and require:
   - `manifest_version == 1`;
   - source/selected/eligible counts `57046/51200/57045`;
   - exact absolute artifact paths;
   - `selection.profile == "vanilla_51200"`;
   - `selection.target_size == 51200`;
   - frozen prefix path/rows/SHA equal the approved constants;
   - `selection_source.head` / `selection_source.tree` equal the current clean execution HEAD/tree;
   - every `selection_source.files` hash equals both the current file bytes and the corresponding committed blob; and
   - every declared artifact hash equals the actual file hash;
5. emit one compact JSON object containing the actual manifest, selected parquet, selected IDs, and diagnostics SHA-256 values;
6. pass those exact values into `math_eval.validate_gradient_diverse_training_data` together with the four new profile/prefix flags;
7. strictly parse the validator report and require counts, paths, hashes, `selection_profile`, prefix fields, schema/source equality, 51,200 unique prompts, token maximum at most 2,048, and monotone token quantiles; and
8. call `ensure_empty_checkpoint_dir` both before and after tokenization.

The profile flags are:

```bash
--expected-selection-profile vanilla_51200 \
--frozen-prefix-selected-ids "${FROZEN_PREFIX}" \
--expected-frozen-prefix-sha256 a78c04cff10c148f45bf9c828e07cc9cf01398b48efeb707253273af359a037e \
--expected-frozen-prefix-rows 12800
```

Do not hard-code runtime-generated output hashes in source; bind their actual values through the immutable, prefix-gated manifest and print them in provenance.

- [ ] **Step 6: Implement preflight/launch separation, lock, provenance, and delegation**

Print:

- Git HEAD/tree/status and diff SHA;
- base/candidate launcher hashes;
- selection artifact hashes and validated report;
- normalized command comparison and exact candidate command; and
- experiment/checkpoint/log paths.

At `VANILLA_SFTGRAD_PREFLIGHT_ONLY=1`, stop after this block. For actual launch:

1. acquire `/tmp/fire-opd-${UID}-${EXPERIMENT_NAME}.launch.lock`;
2. call `validate_opd_cli_runtime(expected_gpus=4, require_idle=True)` with the VERL Python;
3. create the log parent and tee stdout/stderr to the fixed log;
4. unset inherited dry-run flags and ROCm aliases; and
5. `exec bash "$BASE_LAUNCHER"` without extra Hydra arguments.

Run the static test and `bash -n`; both must pass.

- [ ] **Step 7: Run wrapper regressions**

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  verl/tests/trainer/ppo/test_vanilla_opd_launcher.py \
  verl/tests/trainer/ppo/test_vanilla_sft_gradient_51200_launcher.py
VANILLA_SFTGRAD_DRY_RUN=1 bash run_train_vanilla_sft_gradient_51200.sh
bash -n run_train_vanilla_sft_gradient_51200.sh
```

Expected: all pass and `single_variable_contract=PASS` is printed.

- [ ] **Step 8: Commit Task 5**

```bash
git add run_train_vanilla_sft_gradient_51200.sh \
  verl/tests/trainer/ppo/test_vanilla_sft_gradient_51200_launcher.py
git commit -m "feat: launch Vanilla OPD on SFT-gradient data"
```

---

### Task 6: Add a Step-50 Completion Validator

**Files:**
- Create: `math_eval/validate_vanilla_opd_checkpoint.py`
- Create: `math_eval/test_validate_vanilla_opd_checkpoint.py`

**Interfaces:**
- `validate_checkpoint(checkpoint_root: Path, log_path: Path, *, expected_step: int, expected_samples: int, expected_batches: int, world_size: int) -> dict[str, object]`.
- CLI takes the same paths/counts and prints a sorted JSON report.

- [ ] **Step 1: Write failing complete-checkpoint test**

Create `math_eval/test_validate_vanilla_opd_checkpoint.py` with imports, one reusable fixture, and the happy path:

```python
from pathlib import Path

import pytest
import torch

from math_eval import validate_vanilla_opd_checkpoint as validation


def _checkpoint_fixture(
    tmp_path: Path,
    *,
    pointer_step: int = 50,
    samples_yielded: int = 51_200,
    sampler_batches: int = 50,
    missing_rank: int | None = None,
    include_log_marker: bool = True,
) -> tuple[Path, Path]:
    root = tmp_path / "checkpoint"
    actor = root / "global_step_50" / "actor"
    hf = actor / "huggingface"
    hf.mkdir(parents=True)
    for rank in range(4):
        if rank == missing_rank:
            continue
        for kind in ("model", "optim", "extra_state"):
            (actor / f"{kind}_world_size_4_rank_{rank}.pt").write_bytes(b"fixture")
    (hf / "config.json").write_text("{}\n", encoding="utf-8")
    (hf / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (root / "latest_checkpointed_iteration.txt").write_text(
        str(pointer_step), encoding="utf-8"
    )
    torch.save(
        {
            "_snapshot": {
                "_snapshot_step": 50,
                "_main_snapshot": {
                    "_sampler_iter_state": {
                        "samples_yielded": samples_yielded
                    },
                    "_sampler_iter_yielded": sampler_batches,
                },
            },
            "_steps_since_snapshot": 0,
            "_iterator_finished": False,
        },
        root / "global_step_50" / "data.pt",
    )
    log = tmp_path / "run.log"
    marker = (
        "step:50 - training/global_step:50 - actor/pg_loss:0.1\n"
        if include_log_marker
        else "step:49 - training/global_step:49 - actor/pg_loss:0.1\n"
    )
    log.write_text(marker, encoding="utf-8")
    return root, log


def test_validate_checkpoint_accepts_exact_step50_consumption(tmp_path: Path) -> None:
    root, log = _checkpoint_fixture(tmp_path)

    report = validation.validate_checkpoint(
        root,
        log,
        expected_step=50,
        expected_samples=51_200,
        expected_batches=50,
        world_size=4,
    )

    assert report["status"] == "complete"
    assert report["samples_yielded"] == 51_200
    assert report["sampler_batches"] == 50
```

- [ ] **Step 2: Run test and verify RED**

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  math_eval/test_validate_vanilla_opd_checkpoint.py
```

Expected: import failure because the module is absent.

- [ ] **Step 3: Add failing topology and state-corruption tests**

Append these cases before creating the production module:

```python
@pytest.mark.parametrize(
    ("fixture_overrides", "error_match"),
    [
        ({"missing_rank": 3}, "rank 3"),
        ({"samples_yielded": 50_176}, "samples_yielded"),
        ({"sampler_batches": 49}, "sampler batches"),
        ({"pointer_step": 40}, "latest checkpoint"),
        ({"include_log_marker": False}, "step-50 log marker"),
    ],
)
def test_validate_checkpoint_rejects_incomplete_or_wrong_state(
    tmp_path: Path,
    fixture_overrides: dict[str, object],
    error_match: str,
) -> None:
    root, log = _checkpoint_fixture(tmp_path, **fixture_overrides)

    with pytest.raises(ValueError, match=error_match):
        validation.validate_checkpoint(
            root,
            log,
            expected_step=50,
            expected_samples=51_200,
            expected_batches=50,
            world_size=4,
        )
```

Run the Step 2 command. Expected: collection remains RED because the production module is absent.

- [ ] **Step 4: Implement exact topology and data-state validation**

Create the production module and CLI. It must:

- reject missing/non-file pointer, log, data state, rank shards, or HF markers;
- require pointer bytes equal `str(expected_step).encode()`;
- require exactly the expected rank-numbered model/optim/extra-state files with no missing rank;
- load `data.pt` on CPU with `weights_only=False`;
- require `_snapshot_step`, `samples_yielded`, and sampler batches equal expected values;
- require `_steps_since_snapshot == 0`;
- require one log line contains both `step:50` and `training/global_step:50`; and
- return paths, counts, file sizes, and SHA-256 of the pointer/data/log files.

Run the full test file. Expected: the happy path and all five corruption cases pass.

- [ ] **Step 5: Compile and exercise CLI help**

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m compileall -q \
  math_eval/validate_vanilla_opd_checkpoint.py
/home/mchen/miniconda3/envs/verl/bin/python -m \
  math_eval.validate_vanilla_opd_checkpoint --help >/dev/null
```

Expected: both commands exit zero.

- [ ] **Step 6: Commit Task 6**

```bash
git add math_eval/validate_vanilla_opd_checkpoint.py \
  math_eval/test_validate_vanilla_opd_checkpoint.py
git commit -m "feat: validate Vanilla OPD step50 checkpoint"
```

---

### Task 7: Review and Verify the Complete CPU Implementation

**Files:**
- Verify every Task 1-6 file.
- Preserve all ignored production artifacts and the dirty root checkout.

**Interfaces:**
- Produces one clean committed source identity ready for selection and training.

- [ ] **Step 1: Run focused selection tests in the correct environment**

```bash
/home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest -q \
  math_eval/test_deepmath_gradient_diversity.py \
  math_eval/test_select_gradient_diverse_deepmath.py \
  math_eval/test_gradient_diversity_launcher.py \
  math_eval/test_gradient_diverse_51200_launcher.py \
  math_eval/test_validate_gradient_diverse_training_data.py
```

Expected: all pass.

- [ ] **Step 2: Run focused VERL/launcher tests**

```bash
/home/mchen/miniconda3/envs/verl/bin/python -m pytest -q \
  verl/tests/trainer/ppo/test_vanilla_opd_launcher.py \
  verl/tests/trainer/ppo/test_vanilla_sft_gradient_51200_launcher.py \
  verl/tests/trainer/ppo/test_rollout_corr.py \
  verl/tests/trainer/ppo/test_opd_proxy_verify_capture.py \
  math_eval/test_validate_vanilla_opd_checkpoint.py
```

Expected: all pass.

- [ ] **Step 3: Run broader non-GPU regressions**

```bash
/home/mchen/miniconda3/envs/gvendi-opd/bin/python -m pytest -q math_eval \
  --ignore=math_eval/test_collect_prismatic_gradients.py
```

Then run the explicitly environment-sensitive collector tests in `gvendi-opd` if their markers permit CPU collection. Expected: no new failure; any pre-existing environment skip is reported exactly.

- [ ] **Step 4: Run static checks**

```bash
/home/mchen/miniconda3/envs/gvendi-opd/bin/python -m ruff check \
  math_eval/select_gradient_diverse_deepmath.py \
  math_eval/validate_gradient_diverse_training_data.py \
  math_eval/validate_vanilla_opd_checkpoint.py \
  math_eval/test_select_gradient_diverse_deepmath.py \
  math_eval/test_gradient_diverse_51200_launcher.py \
  math_eval/test_validate_gradient_diverse_training_data.py \
  math_eval/test_validate_vanilla_opd_checkpoint.py
/home/mchen/miniconda3/envs/gvendi-opd/bin/python -m compileall -q math_eval
bash -n run_select_gradient_diverse_deepmath_51200.sh
bash -n run_train_vanilla_opd.sh
bash -n run_train_vanilla_sft_gradient_51200.sh
git diff --check
git status --short
```

Expected: checks pass and status is clean.

- [ ] **Step 5: Recheck every frozen old artifact hash**

```bash
sha256sum \
  /home/mchen/FiRe-OPD/data/g-opd/DeepMath-103K/train_filtered_level6.parquet \
  /home/mchen/FiRe-OPD/data/gradient_diversity/deepmath_level6_r1_solution1.jsonl \
  /home/mchen/FiRe-OPD/data/gradient_diversity/deepmath_level6_r1_solution1.manifest.json \
  /home/mchen/FiRe-OPD/data/gradient_diversity/deepmath_level6_r1_solution1.eligibility.json \
  /home/mchen/FiRe-OPD/data/gradient_diversity/DeepMath-103K/train_gradient_diverse_12800.parquet \
  /home/mchen/FiRe-OPD/data/gradient_diversity/selection/manifest.json \
  /home/mchen/FiRe-OPD/data/gradient_diversity/selection/selected_ids.jsonl \
  /home/mchen/FiRe-OPD/data/gradient_diversity/selection/diagnostics.json \
  /home/mchen/FiRe-OPD/data/gradient_diversity/gradients/qwen2.5-0.5b-instruct/gradient.manifest.json
git -C /home/mchen/prismatic-synthesis-reference status --porcelain=v1 --untracked-files=all
git -C /home/mchen/prismatic-synthesis-reference rev-parse HEAD HEAD^{tree}
```

Expected: exactly the nine hashes plus clean reference commit/tree pinned in Global Constraints.

- [ ] **Step 6: Request code review and apply only verified findings**

Use `superpowers:requesting-code-review`. Review scope is the spec commit plus Task 1-6 commits. If feedback arrives, use `superpowers:receiving-code-review`, reproduce each issue, fix under TDD, rerun affected suites, and commit each accepted fix separately.

- [ ] **Step 7: Record the final source identity**

```bash
git rev-parse HEAD
git rev-parse HEAD^{tree}
git status --porcelain=v1 --untracked-files=all
git log --oneline 19578a8..HEAD
```

Expected: status output is empty. Save HEAD/tree and command output in the selection log at execution time.

---

### Task 8: Publish the 51,200-Question Selection

**Files:**
- Read code from the clean implementation worktree.
- Create ignored runtime artifacts only at the four approved production paths.
- Write log `logs/gradient_diversity/deepmath_gradient_diverse_51200.log`.

**Interfaces:**
- Consumes frozen projected gradients and old prefix.
- Produces the selected parquet, selected IDs, diagnostics, and manifest.

- [ ] **Step 1: Run CPU-only selection preflight**

```bash
cd /home/mchen/FiRe-OPD/.worktrees/vanilla-sft-gradient-51200-opd
SFTGRAD51200_PREFLIGHT_ONLY=1 \
  bash run_select_gradient_diverse_deepmath_51200.sh
```

Expected: frozen hashes and global gradient coverage pass; no GPU process starts and no output artifact is written.

- [ ] **Step 2: Obtain explicit GPU execution authorization**

Report the clean source HEAD/tree, preflight result, expected one-GPU clustering-only work, output paths, and the fact that no gradients will be recomputed. Do not continue until the user authorizes this new GPU phase.

- [ ] **Step 3: Verify `opd-CLI`, Slurm, and four idle tokens**

Inside `opd-CLI`, run:

```bash
echo "tmux=$(tmux display-message -p '#S') job=${SLURM_JOB_ID:-} cuda=${CUDA_VISIBLE_DEVICES:-}"
scontrol show job "${SLURM_JOB_ID}" -o
nvidia-smi
```

Expected: tmux is `opd-CLI`, one RUNNING job is shown, and all four inherited CUDA tokens have no unexpected compute process. Do not create a nested `srun`.

- [ ] **Step 4: Launch selection in the existing tmux shell**

Send or enter:

```bash
cd /home/mchen/FiRe-OPD/.worktrees/vanilla-sft-gradient-51200-opd && \
bash run_select_gradient_diverse_deepmath_51200.sh
```

Expected: global coverage passes, only the first allocation token is exposed to the selector, four K-means diagnostic runs complete, and all four outputs publish atomically.

- [ ] **Step 5: Verify exact prefix and artifact counts independently**

Run:

```bash
/home/mchen/miniconda3/envs/gvendi-opd/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path
import pyarrow.parquet as pq

root = Path('/home/mchen/FiRe-OPD/data/gradient_diversity')
old_path = root / 'selection/selected_ids.jsonl'
new_path = root / 'selection_51200/selected_ids.jsonl'
old = [json.loads(line) for line in old_path.read_text().splitlines()]
new = [json.loads(line) for line in new_path.read_text().splitlines()]
manifest = json.loads((root / 'selection_51200/manifest.json').read_text())
assert len(old) == 12_800
assert len(new) == 51_200
assert new[:12_800] == old
assert len({row['id'] for row in new}) == 51_200
assert pq.read_table(root / 'DeepMath-103K/train_gradient_diverse_51200.parquet').num_rows == 51_200
assert manifest['selection']['profile'] == 'vanilla_51200'
assert manifest['selection']['target_size'] == 51_200
for relative in (
    'DeepMath-103K/train_gradient_diverse_51200.parquet',
    'selection_51200/selected_ids.jsonl',
    'selection_51200/diagnostics.json',
    'selection_51200/manifest.json',
):
    path = root / relative
    print(hashlib.sha256(path.read_bytes()).hexdigest(), path)
print('SELECTION_51200_VALID=1')
PY
```

Expected: `SELECTION_51200_VALID=1`.

- [ ] **Step 6: Recheck old hashes and GPU drain**

Repeat Task 7 Step 5 and inspect `nvidia-smi`. Expected: old hashes unchanged and no selector process remains.

---

### Task 9: Complete the Real Training Preflight

**Files:**
- Read the newly published artifacts.
- Write no checkpoint or training log content beyond preflight provenance.

**Interfaces:**
- Produces a validated artifact report and exact candidate command ready for authorization.

- [ ] **Step 1: Run the real 51,200 artifact validator**

Compute actual hashes from the manifest and invoke:

```bash
cd /home/mchen/FiRe-OPD/.worktrees/vanilla-sft-gradient-51200-opd
VANILLA_SFTGRAD_PREFLIGHT_ONLY=1 \
  bash run_train_vanilla_sft_gradient_51200.sh
```

Expected report fields:

```text
selected_rows=51200
selected_ids=51200
unique_prompts=51200
source_rows=57046
eligible_rows=57045
selection_profile=vanilla_51200
frozen_prefix_rows=12800
schema_equal=true
source_rows_equal=true
prompts_over_limit=0
single_variable_contract=PASS
```

- [ ] **Step 2: Inspect the normalized command proof**

Save baseline/candidate dry-run outputs and independently normalize only train data, experiment name, and checkpoint path. Expected: no remaining diff. Confirm raw prompts are used on both sides and no `ref_raw_prompt_key` appears.

- [ ] **Step 3: Confirm checkpoint/log targets and source cleanliness**

```bash
test ! -e /home/mchen/FiRe-OPD/checkpoints/opd-strong-to-weak-sftgrad51200-rawprompt-4gpu-tp4-refmb4-rollmb4 \
  || test -z "$(find /home/mchen/FiRe-OPD/checkpoints/opd-strong-to-weak-sftgrad51200-rawprompt-4gpu-tp4-refmb4-rollmb4 -mindepth 1 -print -quit)"
git status --porcelain=v1 --untracked-files=all
```

Expected: checkpoint target is missing/empty and Git status is empty.

- [ ] **Step 4: Report cost and request final training authorization**

Report that the run uses four H200 NVL GPUs, 50 updates, 51,200 unique questions, canonical validation at steps 0/10/20/30/40/50, and likely has wall time comparable to the historical Vanilla run. Include exact command, source HEAD/tree, selection hashes, log/checkpoint paths, and no-resume policy. Wait for explicit authorization.

---

### Task 10: Launch, Gate, and Complete Vanilla OPD Training

**Files:**
- Produce checkpoint under the fixed production path.
- Produce training log under the fixed production path.
- Read-only validate selection artifacts.

**Interfaces:**
- Produces one complete step-50 checkpoint; final independent evaluation remains deferred.

- [ ] **Step 1: Reconfirm active allocation and idle GPUs immediately before launch**

Inside `opd-CLI`:

```bash
echo "tmux=$(tmux display-message -p '#S') job=${SLURM_JOB_ID:-} cuda=${CUDA_VISIBLE_DEVICES:-}"
scontrol show job "${SLURM_JOB_ID}" -o
nvidia-smi
```

Expected: exact same four unique idle allocation tokens and sufficient remaining wall time. If not, stop and obtain a new authorized allocation.

- [ ] **Step 2: Launch from the clean committed worktree without nested `srun`**

```bash
cd /home/mchen/FiRe-OPD/.worktrees/vanilla-sft-gradient-51200-opd && \
bash run_train_vanilla_sft_gradient_51200.sh
```

The wrapper writes provenance before delegating. Record Slurm job ID, tmux pane, PID/Ray job, source HEAD/tree, and start time.

- [ ] **Step 3: Gate the resolved startup contract before step 1**

Inspect the log and require all approved fields, dataset length 51,200 after filtering, experiment/checkpoint identity, `n=1`, batch/mini-batch 1,024, validation enabled, and every forbidden algorithm disabled. Terminate the launcher-owned process group on any mismatch.

- [ ] **Step 4: Validate the first completed optimizer step**

Parse the first `step:1 -` line with a strict Python script. Require:

- `training/global_step == 1`;
- finite `actor/pg_loss` and `actor/grad_norm`;
- finite rollout-correction metrics and no catastrophic/veto fraction;
- `actor/kl_coef == 0`;
- no `difficulty_aware_opd/`, `tale_budget/`, candidate-selection, length-aware, or rethinking metrics indicating activation;
- 1,024 training rewards/trajectories implied by the fixed command; and
- a completed actor update without OOM, traceback, or shape error.

Print `FIRST_STEP_ACCEPTANCE=PASS`. If any check fails, stop the run and archive its metadata/log; do not resume it.

- [ ] **Step 5: Leave the accepted run active through step 50**

Monitor non-invasively at validation/checkpoint boundaries. Report rolling ETA from `timing_s/step`, but do not interpret training reward or AIME validation as the final result. Do not alter source, data, environment, or command while running.

- [ ] **Step 6: Verify the final checkpoint with the committed validator**

After the process exits zero:

```bash
cd /home/mchen/FiRe-OPD/.worktrees/vanilla-sft-gradient-51200-opd
/home/mchen/miniconda3/envs/verl/bin/python -m math_eval.validate_vanilla_opd_checkpoint \
  --checkpoint-root /home/mchen/FiRe-OPD/checkpoints/opd-strong-to-weak-sftgrad51200-rawprompt-4gpu-tp4-refmb4-rollmb4 \
  --log-path /home/mchen/FiRe-OPD/logs/gradient_diversity/opd-strong-to-weak-sftgrad51200-rawprompt-4gpu-tp4-refmb4-rollmb4.log \
  --expected-step 50 \
  --expected-samples 51200 \
  --expected-batches 50 \
  --world-size 4
```

Expected: JSON status `complete`, `samples_yielded=51200`, `sampler_batches=50`, and all four rank shards present.

- [ ] **Step 7: Verify process drain and artifact immutability**

Check `nvidia-smi`, Ray processes, exact selection hashes, old 12,800 hashes, worktree status, and checkpoint inventory. Expected: no run-owned process remains, all data hashes are unchanged, and source status is clean.

- [ ] **Step 8: Hand off training completion without running final eval**

Report:

1. source HEAD/tree and Slurm/tmux identity;
2. selected parquet/manifest/IDs/diagnostics SHA-256 values;
3. first-step acceptance evidence;
4. step-50 checkpoint and log paths;
5. exact 51,200 data-loader consumption evidence;
6. wall time and GPU count; and
7. the explicit statement that independent n=32 evaluation has not started.

Do not claim the selected data performs better until the separately approved evaluation is complete.
