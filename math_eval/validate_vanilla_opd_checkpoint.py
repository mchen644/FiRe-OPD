"""Validate an exact completed Vanilla OPD step checkpoint."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch

from math_eval.deepmath_gradient_diversity import sha256_file


_SHARD_PATTERN = re.compile(
    r"^(model|optim|extra_state)_world_size_([1-9][0-9]*)_rank_(0|[1-9][0-9]*)\.pt$"
)
_SHARD_KINDS = ("model", "optim", "extra_state")
_HF_MARKERS = ("config.json", "tokenizer.json")


def _require_positive_int(value: object, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{description} must be a positive integer")
    return value


def _require_mapping(value: object, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint data state {description} must be a mapping")
    return value


def _require_nonempty_file(path: Path, description: str) -> int:
    if not path.is_file():
        raise ValueError(f"{description} is missing or not a file: {path}")
    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f"{description} is empty: {path}")
    return size


def _validate_rank_shards(actor_dir: Path, world_size: int) -> dict[str, int]:
    expected_names = {
        f"{kind}_world_size_{world_size}_rank_{rank}.pt"
        for kind in _SHARD_KINDS
        for rank in range(world_size)
    }
    observed_names: set[str] = set()
    for path in actor_dir.iterdir():
        if not path.is_file() or _SHARD_PATTERN.fullmatch(path.name) is None:
            continue
        observed_names.add(path.name)
    missing = expected_names - observed_names
    extra = observed_names - expected_names
    if missing:
        missing_name = sorted(missing)[0]
        rank = _SHARD_PATTERN.fullmatch(missing_name).group(3)
        raise ValueError(f"checkpoint actor shard for rank {rank} is missing: {missing_name}")
    if extra:
        raise ValueError(f"checkpoint contains unexpected actor shards: {sorted(extra)}")

    sizes: dict[str, int] = {}
    for name in sorted(expected_names):
        sizes[name] = _require_nonempty_file(
            actor_dir / name, f"checkpoint actor shard {name}"
        )
    return sizes


def _load_data_state(path: Path) -> dict[str, Any]:
    _require_nonempty_file(path, "checkpoint data-loader state")
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError(f"unable to load checkpoint data-loader state: {error}") from error
    return _require_mapping(value, "root")


def validate_checkpoint(
    checkpoint_root: Path,
    log_path: Path,
    *,
    expected_step: int,
    expected_samples: int,
    expected_batches: int,
    world_size: int,
) -> dict[str, object]:
    """Require exact step, rank topology, and data-loader consumption."""
    expected_step = _require_positive_int(expected_step, "expected_step")
    expected_samples = _require_positive_int(expected_samples, "expected_samples")
    expected_batches = _require_positive_int(expected_batches, "expected_batches")
    world_size = _require_positive_int(world_size, "world_size")

    root = Path(checkpoint_root).resolve()
    log = Path(log_path).resolve()
    if not root.is_dir():
        raise ValueError(f"checkpoint root is missing or not a directory: {root}")
    _require_nonempty_file(log, "training log")

    pointer = root / "latest_checkpointed_iteration.txt"
    _require_nonempty_file(pointer, "latest checkpoint pointer")
    expected_pointer = str(expected_step).encode("ascii")
    pointer_bytes = pointer.read_bytes()
    if pointer_bytes != expected_pointer:
        raise ValueError(
            "latest checkpoint pointer mismatch: "
            f"expected {expected_pointer!r}, got {pointer_bytes!r}"
        )

    step_dir = root / f"global_step_{expected_step}"
    actor_dir = step_dir / "actor"
    if not actor_dir.is_dir():
        raise ValueError(f"step-{expected_step} actor directory is missing: {actor_dir}")
    shard_sizes = _validate_rank_shards(actor_dir, world_size)

    huggingface_dir = actor_dir / "huggingface"
    if not huggingface_dir.is_dir():
        raise ValueError("checkpoint Hugging Face export directory is missing")
    hf_sizes = {
        name: _require_nonempty_file(
            huggingface_dir / name, f"checkpoint Hugging Face marker {name}"
        )
        for name in _HF_MARKERS
    }

    data_path = step_dir / "data.pt"
    state = _load_data_state(data_path)
    snapshot = _require_mapping(state.get("_snapshot"), "_snapshot")
    snapshot_step = snapshot.get("_snapshot_step")
    if snapshot_step != expected_step:
        raise ValueError(
            f"checkpoint snapshot step mismatch: expected {expected_step}, got {snapshot_step!r}"
        )
    main_snapshot = _require_mapping(
        snapshot.get("_main_snapshot"), "_snapshot._main_snapshot"
    )
    sampler_state = _require_mapping(
        main_snapshot.get("_sampler_iter_state"),
        "_snapshot._main_snapshot._sampler_iter_state",
    )
    samples_yielded = sampler_state.get("samples_yielded")
    if samples_yielded != expected_samples:
        raise ValueError(
            "checkpoint samples_yielded mismatch: "
            f"expected {expected_samples}, got {samples_yielded!r}"
        )
    sampler_batches = main_snapshot.get("_sampler_iter_yielded")
    if sampler_batches != expected_batches:
        raise ValueError(
            "checkpoint sampler batches mismatch: "
            f"expected {expected_batches}, got {sampler_batches!r}"
        )
    steps_since_snapshot = state.get("_steps_since_snapshot")
    if steps_since_snapshot != 0:
        raise ValueError(
            "checkpoint _steps_since_snapshot mismatch: "
            f"expected 0, got {steps_since_snapshot!r}"
        )
    iterator_finished = state.get("_iterator_finished")
    if not isinstance(iterator_finished, bool):
        raise ValueError("checkpoint _iterator_finished must be boolean")

    try:
        log_lines = log.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"unable to read training log: {error}") from error
    step_marker = f"step:{expected_step}"
    global_step_marker = f"training/global_step:{expected_step}"
    if not any(
        step_marker in line and global_step_marker in line for line in log_lines
    ):
        raise ValueError(
            f"step-{expected_step} log marker is missing from training log"
        )

    return {
        "status": "complete",
        "checkpoint_root": str(root),
        "step_directory": str(step_dir),
        "actor_directory": str(actor_dir),
        "huggingface_directory": str(huggingface_dir),
        "log_path": str(log),
        "expected_step": expected_step,
        "snapshot_step": snapshot_step,
        "samples_yielded": samples_yielded,
        "sampler_batches": sampler_batches,
        "steps_since_snapshot": steps_since_snapshot,
        "iterator_finished": iterator_finished,
        "world_size": world_size,
        "actor_shard_count": len(shard_sizes),
        "actor_shard_sizes": shard_sizes,
        "huggingface_marker_sizes": hf_sizes,
        "pointer_size": len(pointer_bytes),
        "pointer_sha256": sha256_file(pointer),
        "data_state_size": data_path.stat().st_size,
        "data_state_sha256": sha256_file(data_path),
        "log_size": log.stat().st_size,
        "log_sha256": sha256_file(log),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--expected-samples", type=int, required=True)
    parser.add_argument("--expected-batches", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = validate_checkpoint(
        args.checkpoint_root,
        args.log_path,
        expected_step=args.expected_step,
        expected_samples=args.expected_samples,
        expected_batches=args.expected_batches,
        world_size=args.world_size,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
