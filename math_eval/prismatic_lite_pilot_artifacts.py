"""Immutable artifact helpers for the Prismatic-lite generation pilot."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PilotPaths:
    root: Path
    manifest: Path
    calibration_sample: Path
    calibration_solutions: Path
    calibration_gradient_input: Path
    calibration_gradients: Path
    calibration_report: Path
    candidate_problems: Path
    candidate_solutions: Path
    quality_passed: Path
    candidate_gradient_input: Path
    candidate_gradients: Path
    cluster_seed42: Path
    cluster_seed43: Path
    accepted: Path
    rejected: Path
    report_json: Path
    report_md: Path
    stage_complete: Path

    @classmethod
    def from_root(cls, root: Path) -> "PilotPaths":
        root = Path(root)
        return cls(
            root=root,
            manifest=root / "manifest.json",
            calibration_sample=root / "calibration/sample.jsonl",
            calibration_solutions=root / "calibration/solutions.jsonl",
            calibration_gradient_input=root / "calibration/gradient_input.jsonl",
            calibration_gradients=root / "calibration/paired_gradients",
            calibration_report=root / "calibration/report.json",
            candidate_problems=root / "candidates/problems.jsonl",
            candidate_solutions=root / "candidates/solutions.jsonl",
            quality_passed=root / "candidates/quality_passed.jsonl",
            candidate_gradient_input=root / "candidates/gradient_input.jsonl",
            candidate_gradients=root / "candidates/new_gradients",
            cluster_seed42=root / "selection/cluster_state_seed42.npz",
            cluster_seed43=root / "selection/cluster_state_seed43.npz",
            accepted=root / "selection/accepted.jsonl",
            rejected=root / "selection/rejected.jsonl",
            report_json=root / "report.json",
            report_md=root / "report.md",
            stage_complete=root / "STAGE_COMPLETE.json",
        )


class _DuplicateKey(ValueError):
    pass


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateKey(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"non-finite JSON value {value!r}")


def _strict_json_text(payload: str, description: str) -> object:
    try:
        return json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (json.JSONDecodeError, _DuplicateKey, ValueError) as error:
        raise ValueError(f"invalid {description}: {error}") from error


def strict_json_file(path: Path, description: str) -> object:
    path = Path(path)
    try:
        payload = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ValueError(f"cannot read {description} {path}: {error}") from error
    return _strict_json_text(payload, f"{description} {path}")


def validate_exact_fields(
    value: Mapping[str, object], expected: Sequence[str], description: str
) -> None:
    expected_set = set(expected)
    actual_set = set(value)
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        raise ValueError(
            f"{description} must have exact fields; missing={missing}, extra={extra}"
        )


def strict_jsonl(
    path: Path,
    *,
    exact_fields: Sequence[str] | None = None,
    id_field: str = "id",
) -> list[dict[str, object]]:
    path = Path(path)
    rows: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(f"{path} contains blank line {line_number}")
                row = _strict_json_text(line, f"JSONL row {line_number} in {path}")
                if not isinstance(row, dict):
                    raise ValueError(
                        f"JSONL row {line_number} in {path} must be an object"
                    )
                if exact_fields is not None:
                    validate_exact_fields(
                        row, exact_fields, f"JSONL row {line_number} in {path}"
                    )
                sample_id = row.get(id_field)
                if sample_id is not None:
                    if not isinstance(sample_id, str) or not sample_id:
                        raise ValueError(
                            f"JSONL row {line_number} in {path} has invalid {id_field}"
                        )
                    if sample_id in seen_ids:
                        raise ValueError(f"duplicate id {sample_id!r} in {path}")
                    seen_ids.add(sample_id)
                rows.append(row)
    except (OSError, UnicodeError) as error:
        raise ValueError(f"cannot read JSONL {path}: {error}") from error
    return rows


def _canonical_json_bytes(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"value is not canonical JSON: {error}") from error
    return (text + "\n").encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_publish_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"canonical target already exists: {path}") from error
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_publish_json(path: Path, value: object) -> None:
    _atomic_publish_bytes(path, _canonical_json_bytes(value))


def atomic_publish_jsonl(path: Path, rows: Iterable[object]) -> None:
    payload = b"".join(_canonical_json_bytes(row) for row in rows)
    _atomic_publish_bytes(path, payload)


def atomic_publish_npz(path: Path, values: Mapping[str, object]) -> None:
    if not values:
        raise ValueError("NPZ values must be nonempty")
    ordered: dict[str, np.ndarray[Any, Any]] = {}
    for key in sorted(values):
        if not isinstance(key, str) or not key:
            raise ValueError("NPZ keys must be nonempty strings")
        array = np.asarray(values[key])
        if array.dtype.hasobject:
            raise ValueError(f"NPZ field {key} has object dtype")
        if np.issubdtype(array.dtype, np.inexact) and not bool(np.isfinite(array).all()):
            raise ValueError(f"NPZ field {key} contains non-finite values")
        ordered[key] = array

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            np.savez(handle, **ordered)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"canonical target already exists: {path}") from error
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_or_validate_manifest(
    path: Path,
    expected: Mapping[str, object],
    *,
    owned_paths: Sequence[Path] = (),
) -> dict[str, object]:
    path = Path(path)
    expected_value = dict(expected)
    lock_path = path.with_name(f".{path.name}.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        if path.exists():
            actual = strict_json_file(path, "manifest")
            if actual != expected_value:
                raise ValueError(f"manifest mismatch at {path}")
            return expected_value
        for owned in owned_paths:
            owned = Path(owned)
            if owned.exists() or owned.is_symlink():
                raise ValueError(
                    f"manifest is missing while owned artifact exists: {owned}"
                )
        atomic_publish_json(path, expected_value)
        return expected_value
    finally:
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        finally:
            os.close(lock_descriptor)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: Path) -> dict[str, object]:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"artifact is not a file: {path}")
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }
