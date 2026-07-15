"""Shared, fail-closed artifacts for the OPD proxy-gradient verification.

This module intentionally has no VERL, Ray, or CUDA dependency.  It owns the
byte-level contracts used to join CPU preparation, capture, replay, selection,
and analysis artifacts.
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from safetensors.numpy import load_file as load_safetensors
from safetensors.numpy import save_file as save_safetensors


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_VECTOR_CHUNK_RE = re.compile(
    r"^vectors_(0|[1-9][0-9]*)_(0|[1-9][0-9]*)\.(jsonl|safetensors)$"
)
_RUNTIME_PACKAGES = (
    "torch",
    "transformers",
    "vllm",
    "numpy",
    "scipy",
    "scikit-learn",
    "safetensors",
    "pyarrow",
    "traker",
    "fast-jl",
)
_REQUIRED_RUNTIME_PACKAGES = {
    "verl_capture": frozenset(
        {"torch", "transformers", "vllm", "numpy", "safetensors", "pyarrow"}
    ),
    "gvendi_analysis": frozenset(_RUNTIME_PACKAGES),
}
_VECTOR_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "representation",
        "vector_count",
        "vector_dimension",
        "chunk_count",
        "vector_ids_sha256",
        "stable_ids_sha256",
        "parent_hashes",
        "source_snapshot",
        "repository",
        "runtime",
        "verifier",
        "metadata",
    }
)
_VECTOR_SIDECAR_REQUIRED_FIELDS = frozenset({"vector_id", "stable_id", "tensor_row"})
_VECTOR_SIDECAR_OPTIONAL_FIELDS = frozenset(
    {
        "split",
        "representation",
        "engine_seed",
        "rollout_slot",
        "aggregation",
        "source_capture_sha256",
        "prompt_token_count",
        "completion_token_count",
        "supervised_label_count",
        "full_token_count",
    }
)
_VECTOR_TENSOR_FIELDS = frozenset(
    {
        "projected_gradient",
        "full_gradient_norm",
        "projected_gradient_norm",
        "valid_token_count",
        "response_length",
        "sampled_reverse_kl",
        "opd_signal_rms",
        "verifier_correct_count",
        "verifier_total",
    }
)
_VECTOR_SCALAR_FIELDS = tuple(
    field for field in _VECTOR_TENSOR_FIELDS if field != "projected_gradient"
)
_COMPLETE_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "manifest_sha256",
        "vector_count",
        "chunk_count",
    }
)


@dataclass(frozen=True, order=True)
class TrajectoryKey:
    stable_id: str
    engine_seed: int
    rollout_slot: int

    def __post_init__(self) -> None:
        if not isinstance(self.stable_id, str) or not self.stable_id:
            raise ValueError("trajectory stable_id must be a nonempty string")
        for field_name, value in (
            ("engine_seed", self.engine_seed),
            ("rollout_slot", self.rollout_slot),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"trajectory {field_name} must be a nonnegative integer"
                )


@dataclass(frozen=True, order=True)
class SourceFileSpec:
    repository: str
    path: str

    def __post_init__(self) -> None:
        if not isinstance(self.repository, str) or not self.repository:
            raise ValueError("source repository name must be a nonempty string")
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("source path must be a nonempty relative path")
        normalized = PurePosixPath(self.path.replace(os.sep, "/"))
        if normalized.is_absolute() or ".." in normalized.parts:
            raise ValueError(
                "source path must be relative and cannot escape its repository"
            )
        if "." in normalized.parts:
            raise ValueError("source path must be normalized")


@dataclass(frozen=True)
class VectorSet:
    vector_ids: tuple[str, ...]
    stable_ids: tuple[str, ...]
    vectors: np.ndarray
    full_gradient_norm: np.ndarray | None
    projected_gradient_norm: np.ndarray | None
    valid_token_count: np.ndarray | None
    response_length: np.ndarray | None
    sampled_reverse_kl: np.ndarray | None
    opd_signal_rms: np.ndarray | None
    verifier_correct_count: np.ndarray | None
    verifier_total: np.ndarray | None
    manifest: dict[str, object]


class _DuplicateJsonKey(ValueError):
    pass


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> object:
    raise ValueError(f"non-finite JSON value: {value}")


def _strict_json_bytes(payload: bytes, description: str) -> object:
    try:
        text = payload.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid {description}: {error}") from error


def _strict_json_file(path: Path, description: str) -> object:
    try:
        payload = Path(path).read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read {description} {path}: {error}") from error
    return _strict_json_bytes(payload, description)


def canonical_json_bytes(value: object) -> bytes:
    """Return the sole canonical JSON encoding used by this experiment."""
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"JSON serialization failed: {error}") from error
    return (serialized + "\n").encode("utf-8")


def sha256_id_lines(ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    seen: set[str] = set()
    for stable_id in ids:
        if not isinstance(stable_id, str) or not stable_id:
            raise ValueError("ID hash input must contain nonempty strings")
        if stable_id in seen:
            raise ValueError(f"ID hash input contains duplicate ID: {stable_id}")
        seen.add(stable_id)
        digest.update(stable_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def sha256_ordered_id_lines(ids: Sequence[str]) -> str:
    """Hash an ordered ID sequence while permitting repeated stable IDs."""
    digest = hashlib.sha256()
    for stable_id in ids:
        if not isinstance(stable_id, str) or not stable_id:
            raise ValueError("ordered ID hash input must contain nonempty strings")
        digest.update(stable_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def sha256_int_rows(rows: np.ndarray) -> str:
    array = np.asarray(rows)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError("row hash input must be a two-dimensional integer array")
    digest = hashlib.sha256()
    for row in array:
        payload = json.dumps(
            [int(value) for value in row],
            ensure_ascii=True,
            separators=(",", ":"),
        )
        digest.update(payload.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"cannot hash file {path}: {error}") from error
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    target = Path(path)
    if not isinstance(payload, bytes):
        raise TypeError("atomic payload must be bytes")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: object) -> None:
    atomic_write_bytes(Path(path), canonical_json_bytes(value))


def atomic_write_jsonl(path: Path, rows: Iterable[object]) -> None:
    payload = b"".join(canonical_json_bytes(row) for row in rows)
    atomic_write_bytes(Path(path), payload)


def _little_endian_dtype(dtype: np.dtype[Any]) -> np.dtype[Any]:
    if dtype.hasobject:
        raise ValueError("object arrays are forbidden")
    if dtype.byteorder == "|":
        return dtype
    return dtype.newbyteorder("<")


def _is_little_endian(dtype: np.dtype[Any]) -> bool:
    return dtype.byteorder in {"<", "|"} or (
        dtype.byteorder == "=" and sys.byteorder == "little"
    )


def load_npy_strict(
    path: Path,
    *,
    dtype: np.dtype[Any] | str,
    shape: tuple[int, ...],
) -> np.ndarray:
    expected_dtype = _little_endian_dtype(np.dtype(dtype))
    try:
        array = np.load(Path(path), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid NumPy artifact {path}: {error}") from error
    if array.dtype != expected_dtype or not _is_little_endian(array.dtype):
        raise ValueError(
            f"NumPy artifact dtype mismatch: expected {expected_dtype}, got {array.dtype}"
        )
    if tuple(array.shape) != tuple(shape):
        raise ValueError(
            f"NumPy artifact shape mismatch: expected {shape}, got {tuple(array.shape)}"
        )
    if not array.flags.c_contiguous:
        raise ValueError("NumPy artifact must be C-contiguous")
    return array


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    target = Path(path)
    source = np.asarray(array)
    normalized_dtype = _little_endian_dtype(source.dtype)
    normalized = np.ascontiguousarray(source, dtype=normalized_dtype)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, normalized, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        load_npy_strict(
            temporary, dtype=normalized.dtype, shape=tuple(normalized.shape)
        )
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _as_safetensors_array(value: object, field: str) -> np.ndarray:
    if isinstance(value, np.ndarray):
        array = value
    else:
        try:
            import torch
        except ImportError:
            torch = None  # type: ignore[assignment]
        if torch is None or not isinstance(value, torch.Tensor):
            raise TypeError(f"safetensors value {field!r} must be NumPy or torch")
        tensor = value.detach().cpu()
        if tensor.dtype == torch.bfloat16:
            raise ValueError(
                "bfloat16 tensors require a torch-specific artifact writer"
            )
        array = tensor.numpy()
    if array.dtype.hasobject:
        raise ValueError(f"safetensors value {field!r} cannot have object dtype")
    if array.dtype.byteorder not in {"|", "="}:
        array = array.astype(array.dtype.newbyteorder("="), copy=False)
    return np.ascontiguousarray(array)


def atomic_save_safetensors(path: Path, tensors: Mapping[str, object]) -> None:
    target = Path(path)
    if not tensors:
        raise ValueError("safetensors artifact cannot be empty")
    normalized: dict[str, np.ndarray] = {}
    for name in sorted(tensors):
        if not isinstance(name, str) or not name:
            raise ValueError("safetensors keys must be nonempty strings")
        normalized[name] = _as_safetensors_array(tensors[name], name)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_safetensors(normalized, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        reloaded = load_safetensors(temporary)
        if set(reloaded) != set(normalized):
            raise ValueError("safetensors readback key mismatch")
        for name, expected in normalized.items():
            actual = reloaded[name]
            if actual.dtype != expected.dtype or actual.shape != expected.shape:
                raise ValueError(f"safetensors readback mismatch for {name}")
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def write_or_validate_manifest(
    path: Path, expected: Mapping[str, object]
) -> dict[str, object]:
    """Create a canonical manifest once, or validate the existing exact object."""
    target = Path(path)
    normalized_value = _strict_json_bytes(
        canonical_json_bytes(dict(expected)), "expected manifest"
    )
    if not isinstance(normalized_value, dict):
        raise ValueError("expected manifest must be a JSON object")
    normalized = dict(normalized_value)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_name(f".{target.name}.lock")
    lock_descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        if target.exists():
            actual = _strict_json_file(target, "manifest")
            if actual != normalized or target.read_bytes() != canonical_json_bytes(
                normalized
            ):
                raise ValueError("manifest mismatch: existing bytes or values differ")
            return normalized
        atomic_write_bytes(target, canonical_json_bytes(normalized))
        return normalized
    finally:
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        finally:
            os.close(lock_descriptor)


def validate_existing_work_unit(
    path: Path, expected_contract: Mapping[str, object]
) -> dict[str, object]:
    """Validate an existing immutable work-unit contract by full canonical value."""
    target = Path(path)
    normalized_value = _strict_json_bytes(
        canonical_json_bytes(dict(expected_contract)), "expected work-unit contract"
    )
    if not isinstance(normalized_value, dict):
        raise ValueError("expected work-unit contract must be a JSON object")
    if not target.is_file():
        raise ValueError(f"work-unit contract does not exist: {target}")
    actual = _strict_json_file(target, "work-unit contract")
    if actual != normalized_value or target.read_bytes() != canonical_json_bytes(
        normalized_value
    ):
        raise ValueError("work-unit contract differs from the immutable identity")
    return dict(normalized_value)


def _stable_file_record(path: Path, logical_path: str) -> dict[str, object]:
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"source file changed while hashing: {path}")
    return {"path": logical_path, "size": after.st_size, "sha256": digest}


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def recursive_file_manifest(root: Path) -> dict[str, object]:
    """Hash a directory tree by logical relative path and actual file bytes."""
    requested_root = Path(root)
    try:
        physical_root = requested_root.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError(
            f"model/tokenizer root does not exist: {requested_root}"
        ) from error
    if not physical_root.is_dir():
        raise ValueError(f"model/tokenizer root is not a directory: {requested_root}")

    records: list[dict[str, object]] = []

    def visit(
        logical: PurePosixPath, physical: Path, ancestors: frozenset[tuple[int, int]]
    ) -> None:
        stat = physical.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity in ancestors:
            raise ValueError(f"symlink cycle beneath {requested_root}")
        next_ancestors = ancestors | {identity}
        try:
            entries = sorted(os.scandir(physical), key=lambda entry: entry.name)
        except OSError as error:
            raise ValueError(
                f"cannot scan model/tokenizer directory {physical}: {error}"
            ) from error
        for entry in entries:
            entry_path = Path(entry.path)
            logical_path = logical / entry.name
            if entry.is_symlink():
                try:
                    target = entry_path.resolve(strict=True)
                except (OSError, RuntimeError) as error:
                    raise ValueError(
                        f"broken symlink in model/tokenizer tree: {logical_path}"
                    ) from error
                if not _is_within(target, physical_root):
                    raise ValueError(f"symlink escapes root: {logical_path}")
                if target.is_dir():
                    visit(logical_path, target, next_ancestors)
                elif target.is_file():
                    records.append(_stable_file_record(target, logical_path.as_posix()))
                else:
                    raise ValueError(f"unsupported symlink target: {logical_path}")
            elif entry.is_dir(follow_symlinks=False):
                visit(logical_path, entry_path, next_ancestors)
            elif entry.is_file(follow_symlinks=False):
                records.append(_stable_file_record(entry_path, logical_path.as_posix()))
            else:
                raise ValueError(f"unsupported filesystem entry: {logical_path}")

    visit(PurePosixPath(), physical_root, frozenset())
    records.sort(key=lambda row: str(row["path"]))
    if not records:
        raise ValueError(
            f"model/tokenizer tree contains no regular files: {requested_root}"
        )
    return {
        "root": str(physical_root),
        "file_count": len(records),
        "files": records,
        "manifest_sha256": hashlib.sha256(canonical_json_bytes(records)).hexdigest(),
    }


def build_source_snapshot(
    repositories: Mapping[str, Path], specs: Sequence[SourceFileSpec]
) -> dict[str, object]:
    if not repositories:
        raise ValueError("source snapshot requires at least one repository")
    roots: dict[str, Path] = {}
    for name, path in repositories.items():
        if not isinstance(name, str) or not name:
            raise ValueError("repository names must be nonempty strings")
        try:
            root = Path(path).resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise ValueError(f"source repository does not exist: {path}") from error
        if not root.is_dir():
            raise ValueError(f"source repository is not a directory: {path}")
        roots[name] = root

    ordered_specs = sorted(specs)
    if len(set(ordered_specs)) != len(ordered_specs):
        raise ValueError("source snapshot contains duplicate file specifications")
    records: list[dict[str, object]] = []
    for spec in ordered_specs:
        if spec.repository not in roots:
            raise ValueError(f"unknown source repository: {spec.repository}")
        root = roots[spec.repository]
        requested = root.joinpath(*PurePosixPath(spec.path).parts)
        try:
            resolved = requested.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise ValueError(
                f"source file does not exist: {spec.repository}:{spec.path}"
            ) from error
        if not _is_within(resolved, root):
            raise ValueError(
                f"source path escapes repository: {spec.repository}:{spec.path}"
            )
        if not resolved.is_file():
            raise ValueError(
                f"source path is not a regular file: {spec.repository}:{spec.path}"
            )
        record = _stable_file_record(resolved, spec.path)
        records.append({"repository": spec.repository, **record})
    return {
        "repositories": {name: str(roots[name]) for name in sorted(roots)},
        "files": records,
        "manifest_sha256": hashlib.sha256(canonical_json_bytes(records)).hexdigest(),
    }


def repository_state(repository: Path) -> dict[str, str]:
    root = Path(repository).resolve()

    def git(*arguments: str) -> str:
        try:
            result = subprocess.run(
                ["git", "-C", str(root), *arguments],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise ValueError(f"invalid git repository {root}: {error}") from error
        return result.stdout

    head = git("rev-parse", "HEAD").strip()
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "head": head,
        "status": status,
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def build_runtime_metadata(
    runtime_profile: str,
    *,
    version_resolver: Callable[[str], str] = importlib.metadata.version,
    python_executable: str | None = None,
    python_version: str | None = None,
) -> dict[str, object]:
    if runtime_profile not in _REQUIRED_RUNTIME_PACKAGES:
        raise ValueError(f"unknown runtime profile: {runtime_profile}")
    required = _REQUIRED_RUNTIME_PACKAGES[runtime_profile]
    versions: dict[str, str | None] = {}
    for package in _RUNTIME_PACKAGES:
        try:
            version = version_resolver(package)
        except (importlib.metadata.PackageNotFoundError, LookupError):
            version = None
        if version is not None and (not isinstance(version, str) or not version):
            raise ValueError(f"invalid package version for {package}")
        if package in required and version is None:
            raise ValueError(
                f"required package {package} is unavailable for {runtime_profile}"
            )
        versions[package] = version
    metadata: dict[str, object] = {
        "runtime_profile": runtime_profile,
        "python_executable": python_executable or sys.executable,
        "python_version": python_version or platform.python_version(),
        "package_versions": versions,
    }
    metadata["runtime_sha256"] = hashlib.sha256(
        canonical_json_bytes(metadata)
    ).hexdigest()
    return metadata


def validate_exact_key_coverage(
    actual_keys: Iterable[TrajectoryKey], expected_keys: Iterable[TrajectoryKey]
) -> None:
    actual = list(actual_keys)
    expected = list(expected_keys)
    actual_counts = Counter(actual)
    duplicates = sorted(key for key, count in actual_counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate trajectory key: {duplicates[0]}")
    expected_counts = Counter(expected)
    expected_duplicates = sorted(
        key for key, count in expected_counts.items() if count > 1
    )
    if expected_duplicates:
        raise ValueError(f"duplicate expected trajectory key: {expected_duplicates[0]}")
    actual_set = set(actual)
    expected_set = set(expected)
    missing = sorted(expected_set - actual_set)
    if missing:
        raise ValueError(f"missing trajectory keys: {missing[:5]}")
    unexpected = sorted(actual_set - expected_set)
    if unexpected:
        raise ValueError(f"unexpected trajectory keys: {unexpected[:5]}")


def _require_exact_fields(
    value: Mapping[str, object],
    allowed: frozenset[str],
    required: frozenset[str],
    description: str,
) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown {description} fields: {sorted(unknown)}")
    missing = required - set(value)
    if missing:
        raise ValueError(f"missing {description} fields: {sorted(missing)}")


def _require_int(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _load_sidecar(path: Path, expected_rows: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        with path.open("rb") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(f"blank vector sidecar line {line_number}")
                value = _strict_json_bytes(line, f"vector sidecar line {line_number}")
                if not isinstance(value, dict):
                    raise ValueError("vector sidecar rows must be JSON objects")
                if line != canonical_json_bytes(value):
                    raise ValueError("vector sidecar rows must use canonical JSON")
                _require_exact_fields(
                    value,
                    _VECTOR_SIDECAR_REQUIRED_FIELDS | _VECTOR_SIDECAR_OPTIONAL_FIELDS,
                    _VECTOR_SIDECAR_REQUIRED_FIELDS,
                    "vector sidecar",
                )
                vector_id = value["vector_id"]
                stable_id = value["stable_id"]
                tensor_row = value["tensor_row"]
                if not isinstance(vector_id, str) or not vector_id:
                    raise ValueError("vector_id must be a nonempty string")
                if not isinstance(stable_id, str) or not stable_id:
                    raise ValueError("stable_id must be a nonempty string")
                if (
                    isinstance(tensor_row, bool)
                    or not isinstance(tensor_row, int)
                    or tensor_row != len(rows)
                ):
                    raise ValueError(
                        "vector sidecar tensor_row does not match row order"
                    )
                for integer_field in ("engine_seed", "rollout_slot"):
                    if integer_field in value:
                        _require_int(value[integer_field], integer_field)
                for count_field in (
                    "prompt_token_count",
                    "completion_token_count",
                    "supervised_label_count",
                    "full_token_count",
                ):
                    if count_field in value:
                        _require_int(value[count_field], count_field, minimum=1)
                for text_field in ("split", "representation", "aggregation"):
                    if text_field in value and (
                        not isinstance(value[text_field], str) or not value[text_field]
                    ):
                        raise ValueError(
                            f"vector sidecar {text_field} must be a nonempty string"
                        )
                if "source_capture_sha256" in value:
                    _require_sha256(
                        value["source_capture_sha256"], "source_capture_sha256"
                    )
                rows.append(dict(value))
    except OSError as error:
        raise ValueError(f"cannot read vector sidecar {path}: {error}") from error
    if len(rows) != expected_rows:
        raise ValueError(
            f"vector sidecar row count mismatch: expected {expected_rows}, got {len(rows)}"
        )
    return rows


def _discover_vector_chunks(directory: Path) -> list[tuple[int, int, Path, Path]]:
    grouped: dict[tuple[int, int], dict[str, Path]] = {}
    for path in directory.iterdir():
        if not path.is_file():
            continue
        if path.name.startswith(".") and ".tmp" in path.name:
            continue
        if not path.name.startswith("vectors_"):
            continue
        match = _VECTOR_CHUNK_RE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"malformed vector chunk artifact: {path.name}")
        start, end, kind = int(match.group(1)), int(match.group(2)), match.group(3)
        if end <= start:
            raise ValueError(f"invalid vector chunk bounds: {path.name}")
        files = grouped.setdefault((start, end), {})
        if kind in files:
            raise ValueError(f"duplicate vector chunk artifact: {path.name}")
        files[kind] = path
    chunks: list[tuple[int, int, Path, Path]] = []
    cursor = 0
    for (start, end), files in sorted(grouped.items()):
        if set(files) != {"jsonl", "safetensors"}:
            raise ValueError(f"vector chunk {start}:{end} must have paired artifacts")
        if start != cursor:
            raise ValueError(
                f"non-contiguous vector chunks: expected start {cursor}, got {start}"
            )
        chunks.append((start, end, files["jsonl"], files["safetensors"]))
        cursor = end
    return chunks


def _validate_scalar_field(name: str, value: np.ndarray, rows: int) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (rows,):
        raise ValueError(f"vector tensor {name} must have shape ({rows},)")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"vector tensor {name} must be finite numeric values")
    if name in {
        "full_gradient_norm",
        "projected_gradient_norm",
        "valid_token_count",
        "response_length",
        "opd_signal_rms",
        "verifier_correct_count",
        "verifier_total",
    } and np.any(array < 0):
        raise ValueError(f"vector tensor {name} cannot be negative")
    return np.ascontiguousarray(array)


def _reorder_optional(value: np.ndarray | None, order: np.ndarray) -> np.ndarray | None:
    if value is None:
        return None
    return np.ascontiguousarray(value[order])


def load_vector_set(
    directory: Path,
    *,
    expected_stable_ids: Sequence[str] | None = None,
    expected_vector_ids: Sequence[str] | None = None,
    expected_parent_hashes: Mapping[str, str] | None = None,
    expected_representation: str | None = None,
) -> VectorSet:
    """Load a complete vector artifact and join by explicit IDs, never row order."""
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"vector directory does not exist: {root}")
    manifest_path = root / "manifest.json"
    complete_path = root / "COMPLETE.json"
    if not manifest_path.is_file():
        raise ValueError(f"vector manifest does not exist: {manifest_path}")
    if not complete_path.is_file():
        raise ValueError(f"vector artifact lacks COMPLETE.json: {root}")

    manifest_value = _strict_json_file(manifest_path, "vector manifest")
    if not isinstance(manifest_value, dict):
        raise ValueError("vector manifest must be a JSON object")
    if manifest_path.read_bytes() != canonical_json_bytes(manifest_value):
        raise ValueError("vector manifest must use canonical JSON")
    _require_exact_fields(
        manifest_value,
        _VECTOR_MANIFEST_FIELDS,
        _VECTOR_MANIFEST_FIELDS,
        "vector manifest",
    )
    manifest = dict(manifest_value)
    if manifest["schema_version"] != 1 or manifest["artifact_type"] != "vector_set":
        raise ValueError("unsupported vector manifest schema or artifact type")
    representation = manifest["representation"]
    if not isinstance(representation, str) or not representation:
        raise ValueError("vector representation must be a nonempty string")
    if (
        expected_representation is not None
        and representation != expected_representation
    ):
        raise ValueError(
            f"vector representation mismatch: expected {expected_representation}, got {representation}"
        )
    vector_count = _require_int(manifest["vector_count"], "vector_count", minimum=1)
    vector_dimension = _require_int(
        manifest["vector_dimension"], "vector_dimension", minimum=1
    )
    chunk_count = _require_int(manifest["chunk_count"], "chunk_count", minimum=1)
    _require_sha256(manifest["vector_ids_sha256"], "vector_ids_sha256")
    _require_sha256(manifest["stable_ids_sha256"], "stable_ids_sha256")
    parent_hashes = manifest["parent_hashes"]
    if not isinstance(parent_hashes, dict):
        raise ValueError("vector parent_hashes must be a JSON object")
    for name, digest in parent_hashes.items():
        if not isinstance(name, str) or not name:
            raise ValueError("vector parent hash names must be nonempty strings")
        _require_sha256(digest, f"parent_hashes.{name}")
    if expected_parent_hashes is not None:
        for name, digest in expected_parent_hashes.items():
            if parent_hashes.get(name) != digest:
                raise ValueError(f"vector parent hash mismatch for {name}")
    for object_field in (
        "source_snapshot",
        "repository",
        "runtime",
        "verifier",
        "metadata",
    ):
        if not isinstance(manifest[object_field], dict):
            raise ValueError(f"vector manifest {object_field} must be a JSON object")
    verifier_status = manifest["verifier"].get("status")
    if verifier_status not in {"computed", "not_computed"}:
        raise ValueError("vector verifier status must be computed or not_computed")

    complete_value = _strict_json_file(complete_path, "vector completion marker")
    if not isinstance(complete_value, dict):
        raise ValueError("vector COMPLETE.json must be a JSON object")
    if complete_path.read_bytes() != canonical_json_bytes(complete_value):
        raise ValueError("vector COMPLETE.json must use canonical JSON")
    _require_exact_fields(
        complete_value, _COMPLETE_FIELDS, _COMPLETE_FIELDS, "vector completion marker"
    )
    if (
        complete_value["schema_version"] != 1
        or complete_value["artifact_type"] != "vector_set_complete"
    ):
        raise ValueError("unsupported vector completion marker")
    if complete_value["manifest_sha256"] != sha256_file(manifest_path):
        raise ValueError("vector completion marker manifest hash mismatch")
    if complete_value["vector_count"] != vector_count:
        raise ValueError("vector completion marker count mismatch")
    if complete_value["chunk_count"] != chunk_count:
        raise ValueError("vector completion marker chunk count mismatch")

    chunks = _discover_vector_chunks(root)
    if len(chunks) != chunk_count:
        raise ValueError(
            f"vector chunk count mismatch: expected {chunk_count}, got {len(chunks)}"
        )
    if not chunks or chunks[-1][1] != vector_count:
        covered = chunks[-1][1] if chunks else 0
        raise ValueError(
            f"incomplete vector chunk coverage: expected {vector_count}, got {covered}"
        )
    replay_chunk_records = manifest["metadata"].get("replay_chunks")
    if replay_chunk_records is not None:
        if not isinstance(replay_chunk_records, list) or len(
            replay_chunk_records
        ) != len(chunks):
            raise ValueError("vector replay chunk records do not match chunks")
        required_record_fields = frozenset(
            {
                "schema_version",
                "artifact_type",
                "start",
                "end",
                "vector_ids_sha256",
                "tensor_file",
                "tensor_sha256",
                "sidecar_file",
                "sidecar_sha256",
                "shard_contract_sha256",
            }
        )
        contract_hash = manifest["metadata"].get("shard_contract_sha256")
        _require_sha256(contract_hash, "metadata.shard_contract_sha256")
        shard_contract_path = root / "SHARD_CONTRACT.json"
        if not shard_contract_path.is_file() or sha256_file(
            shard_contract_path
        ) != contract_hash:
            raise ValueError("vector shard contract hash mismatch")
        for record, (start, end, sidecar_path, tensor_path) in zip(
            replay_chunk_records, chunks, strict=True
        ):
            if not isinstance(record, dict):
                raise ValueError("vector replay chunk record must be an object")
            _require_exact_fields(
                record,
                required_record_fields,
                required_record_fields,
                "vector replay chunk record",
            )
            expected_values = {
                "schema_version": 1,
                "artifact_type": "opd_replay_vector_chunk_complete",
                "start": start,
                "end": end,
                "tensor_file": tensor_path.name,
                "sidecar_file": sidecar_path.name,
                "shard_contract_sha256": contract_hash,
            }
            for field, expected in expected_values.items():
                if record[field] != expected:
                    raise ValueError(f"vector replay chunk {field} mismatch")
            for path, hash_field in (
                (tensor_path, "tensor_sha256"),
                (sidecar_path, "sidecar_sha256"),
            ):
                _require_sha256(record[hash_field], hash_field)
                if record[hash_field] != sha256_file(path):
                    raise ValueError(f"vector replay chunk {hash_field} mismatch")
            _require_sha256(record["vector_ids_sha256"], "vector_ids_sha256")

    all_vector_ids: list[str] = []
    all_stable_ids: list[str] = []
    projected_chunks: list[np.ndarray] = []
    scalar_chunks: dict[str, list[np.ndarray]] = {
        name: [] for name in _VECTOR_SCALAR_FIELDS
    }
    scalar_presence: dict[str, bool | None] = {
        name: None for name in _VECTOR_SCALAR_FIELDS
    }

    for chunk_index, (start, end, sidecar_path, tensor_path) in enumerate(chunks):
        rows = end - start
        sidecars = _load_sidecar(sidecar_path, rows)
        if replay_chunk_records is not None and sha256_id_lines(
            [str(sidecar["vector_id"]) for sidecar in sidecars]
        ) != replay_chunk_records[chunk_index]["vector_ids_sha256"]:
            raise ValueError("vector replay chunk vector ID hash mismatch")
        try:
            tensors = load_safetensors(tensor_path)
        except Exception as error:
            raise ValueError(
                f"invalid vector safetensors {tensor_path}: {error}"
            ) from error
        unknown_tensors = set(tensors) - _VECTOR_TENSOR_FIELDS
        if unknown_tensors:
            raise ValueError(f"unknown vector tensor fields: {sorted(unknown_tensors)}")
        if "projected_gradient" not in tensors:
            raise ValueError("missing vector tensor field: projected_gradient")
        projected = np.asarray(tensors["projected_gradient"])
        if projected.dtype != np.dtype("<f4") or projected.shape != (
            rows,
            vector_dimension,
        ):
            raise ValueError(
                "projected_gradient must be float32 with the manifest dimension"
            )
        if not projected.flags.c_contiguous:
            raise ValueError("projected_gradient must be C-contiguous")
        if not np.isfinite(projected).all():
            raise ValueError("projected gradients must be finite")
        norms = np.linalg.norm(projected.astype(np.float64), axis=1)
        if np.any(norms == 0):
            raise ValueError("projected gradients contain a zero vector")
        projected_chunks.append(projected)

        for field in _VECTOR_SCALAR_FIELDS:
            present = field in tensors
            previous = scalar_presence[field]
            if previous is not None and previous != present:
                raise ValueError(
                    f"vector tensor {field} has inconsistent chunk presence"
                )
            scalar_presence[field] = present
            if present:
                scalar_chunks[field].append(
                    _validate_scalar_field(field, tensors[field], rows)
                )
        if ("verifier_correct_count" in tensors) != ("verifier_total" in tensors):
            raise ValueError("verifier count tensors must be present together")

        for sidecar in sidecars:
            if (
                "representation" in sidecar
                and sidecar["representation"] != representation
            ):
                raise ValueError("vector sidecar representation mismatch")
            all_vector_ids.append(str(sidecar["vector_id"]))
            all_stable_ids.append(str(sidecar["stable_id"]))

    if len(set(all_vector_ids)) != len(all_vector_ids):
        raise ValueError("duplicate vector IDs")
    if sha256_id_lines(all_vector_ids) != manifest["vector_ids_sha256"]:
        raise ValueError("vector ID sequence hash mismatch")
    if sha256_ordered_id_lines(all_stable_ids) != manifest["stable_ids_sha256"]:
        raise ValueError("stable ID sequence hash mismatch")

    core_scalar_fields = set(_VECTOR_SCALAR_FIELDS) - {
        "verifier_correct_count",
        "verifier_total",
    }
    missing_core = sorted(
        field for field in core_scalar_fields if scalar_presence[field] is not True
    )
    if missing_core:
        raise ValueError(f"missing vector tensor fields: {missing_core}")
    verifier_present = scalar_presence["verifier_correct_count"] is True
    if verifier_present != (scalar_presence["verifier_total"] is True):
        raise ValueError("verifier count tensors must be present together")
    if verifier_status == "computed" and not verifier_present:
        raise ValueError("computed verifier status requires verifier count tensors")
    if verifier_status == "not_computed" and verifier_present:
        raise ValueError("not_computed verifier status forbids verifier count tensors")

    vectors = np.ascontiguousarray(np.concatenate(projected_chunks, axis=0))
    scalar_values: dict[str, np.ndarray | None] = {}
    for field in _VECTOR_SCALAR_FIELDS:
        if scalar_presence[field]:
            scalar_values[field] = np.ascontiguousarray(
                np.concatenate(scalar_chunks[field], axis=0)
            )
        else:
            scalar_values[field] = None
    projected_norm = scalar_values["projected_gradient_norm"]
    for positive_field in (
        "full_gradient_norm",
        "projected_gradient_norm",
        "valid_token_count",
        "response_length",
    ):
        value = scalar_values[positive_field]
        if value is None or np.any(value <= 0):
            raise ValueError(f"vector tensor {positive_field} must be positive")
    if verifier_present:
        correct = scalar_values["verifier_correct_count"]
        total = scalar_values["verifier_total"]
        assert correct is not None and total is not None
        if np.any(total <= 0) or np.any(correct > total):
            raise ValueError("invalid verifier correct/total counts")
    if projected_norm is not None and not np.allclose(
        projected_norm.astype(np.float64),
        np.linalg.norm(vectors.astype(np.float64), axis=1),
        rtol=1e-5,
        atol=1e-7,
    ):
        raise ValueError("stored projected_gradient_norm does not match vectors")

    order = np.arange(vector_count, dtype=np.int64)
    if expected_vector_ids is not None:
        requested_vector_ids = tuple(expected_vector_ids)
        if len(set(requested_vector_ids)) != len(requested_vector_ids):
            raise ValueError("expected vector IDs must be unique")
        if set(requested_vector_ids) != set(all_vector_ids):
            raise ValueError("expected vector IDs do not match artifact coverage")
        positions = {vector_id: index for index, vector_id in enumerate(all_vector_ids)}
        order = np.array([positions[vector_id] for vector_id in requested_vector_ids])
    if expected_stable_ids is not None:
        requested_stable_ids = tuple(expected_stable_ids)
        if any(
            not isinstance(stable_id, str) or not stable_id
            for stable_id in requested_stable_ids
        ):
            raise ValueError("expected stable IDs must be nonempty strings")
        if len(requested_stable_ids) != vector_count:
            raise ValueError("expected stable IDs do not match artifact coverage")
        if expected_vector_ids is not None:
            reordered_stable_ids = tuple(all_stable_ids[index] for index in order)
            if reordered_stable_ids != requested_stable_ids:
                raise ValueError("requested vector and stable ID orders disagree")
        elif len(set(all_stable_ids)) != len(all_stable_ids):
            if requested_stable_ids != tuple(all_stable_ids):
                raise ValueError(
                    "duplicate stable IDs require vector IDs for unambiguous reordering"
                )
        else:
            if set(requested_stable_ids) != set(all_stable_ids):
                raise ValueError("expected stable IDs do not match artifact coverage")
            positions = {
                stable_id: index for index, stable_id in enumerate(all_stable_ids)
            }
            order = np.array(
                [positions[stable_id] for stable_id in requested_stable_ids]
            )

    ordered_vector_ids = tuple(all_vector_ids[index] for index in order)
    ordered_stable_ids = tuple(all_stable_ids[index] for index in order)
    return VectorSet(
        vector_ids=ordered_vector_ids,
        stable_ids=ordered_stable_ids,
        vectors=np.ascontiguousarray(vectors[order]),
        full_gradient_norm=_reorder_optional(
            scalar_values["full_gradient_norm"], order
        ),
        projected_gradient_norm=_reorder_optional(projected_norm, order),
        valid_token_count=_reorder_optional(scalar_values["valid_token_count"], order),
        response_length=_reorder_optional(scalar_values["response_length"], order),
        sampled_reverse_kl=_reorder_optional(
            scalar_values["sampled_reverse_kl"], order
        ),
        opd_signal_rms=_reorder_optional(scalar_values["opd_signal_rms"], order),
        verifier_correct_count=_reorder_optional(
            scalar_values["verifier_correct_count"], order
        ),
        verifier_total=_reorder_optional(scalar_values["verifier_total"], order),
        manifest=manifest,
    )


__all__ = [
    "SourceFileSpec",
    "TrajectoryKey",
    "VectorSet",
    "atomic_save_npy",
    "atomic_save_safetensors",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_jsonl",
    "build_runtime_metadata",
    "build_source_snapshot",
    "canonical_json_bytes",
    "load_npy_strict",
    "load_vector_set",
    "recursive_file_manifest",
    "repository_state",
    "sha256_file",
    "sha256_id_lines",
    "sha256_ordered_id_lines",
    "sha256_int_rows",
    "validate_exact_key_coverage",
    "validate_existing_work_unit",
    "write_or_validate_manifest",
]
