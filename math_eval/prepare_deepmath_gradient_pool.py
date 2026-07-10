"""Prepare the pinned DeepMath R1 pool for projected-gradient collection."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from math_eval.deepmath_gradient_diversity import (  # noqa: E402
    join_filtered_rows,
    sha256_file,
)


DATASET_NAME = "zwhe99/DeepMath-103K"
DATASET_REVISION = "5cf055d1fe3d7a2eb19719ac020211469736ae44"
DATASET_SPLIT = "train"
MANIFEST_VERSION = 1


def _validate_source_table(table: pa.Table, expected_count: int) -> None:
    if table.num_rows != expected_count:
        raise ValueError(
            "expected_count mismatch: "
            f"expected {expected_count} source rows, got {table.num_rows}"
        )

    missing_columns = {
        "prompt",
        "reward_model",
        "extra_info",
    }.difference(table.schema.names)
    if missing_columns:
        if "extra_info" in missing_columns:
            missing_columns.remove("extra_info")
            missing_columns.add("extra_info.index")
        raise ValueError(
            "source parquet is missing columns: " + ", ".join(sorted(missing_columns))
        )

    extra_info_type = table.schema.field("extra_info").type
    if not pa.types.is_struct(extra_info_type) or extra_info_type.get_field_index(
        "index"
    ) < 0:
        raise ValueError("source parquet is missing extra_info.index")


def _validate_source_indices(rows: Sequence[Mapping]) -> None:
    for source_row_index, row in enumerate(rows):
        extra_info = row.get("extra_info")
        if not isinstance(extra_info, Mapping) or extra_info.get(
            "index"
        ) != source_row_index:
            raise ValueError(
                "source parquet extra_info.index mismatch at row "
                f"{source_row_index}"
            )


def _require_manifest_value(manifest: Mapping, field: str, expected: object) -> None:
    if manifest.get(field) != expected:
        raise ValueError(
            f"cached manifest {field} mismatch: "
            f"expected {expected!r}, got {manifest.get(field)!r}"
        )


def _read_manifest(path: Path) -> dict:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid cached manifest {path}: {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError(f"invalid cached manifest {path}: expected a JSON object")
    return manifest


def _validate_prepared_jsonl(path: Path, expected_count: int) -> None:
    seen_ids: set[str] = set()
    row_count = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for row_count, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(
                        f"cached prepared JSONL contains a blank line at {row_count}"
                    )
                record = json.loads(line)
                source_row_index = row_count - 1
                if not isinstance(record, dict):
                    raise ValueError(
                        "cached prepared JSONL rows must be JSON objects"
                    )
                if record.get("source_row_index") != source_row_index:
                    raise ValueError(
                        "cached prepared JSONL source_row_index mismatch at row "
                        f"{source_row_index}"
                    )
                expected_id = f"deepmath-level6-{source_row_index:06d}"
                if record.get("id") != expected_id:
                    raise ValueError(
                        f"cached prepared JSONL id mismatch at row {source_row_index}"
                    )
                if expected_id in seen_ids:
                    raise ValueError(f"duplicate prepared ID: {expected_id}")
                seen_ids.add(expected_id)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid cached prepared JSONL {path}: {error}") from error

    if row_count != expected_count:
        raise ValueError(
            "cached prepared JSONL row count mismatch: "
            f"expected {expected_count}, got {row_count}"
        )


def _validate_cached_artifacts(
    *,
    source_parquet: Path,
    source_sha256: str,
    source_row_count: int,
    output_jsonl: Path,
    manifest_path: Path,
    expected_count: int,
) -> dict:
    manifest = _read_manifest(manifest_path)
    expected_values = {
        "manifest_version": MANIFEST_VERSION,
        "dataset_name": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "dataset_split": DATASET_SPLIT,
        "source_parquet": str(source_parquet.resolve()),
        "source_sha256": source_sha256,
        "source_row_count": source_row_count,
        "prepared_jsonl": str(output_jsonl.resolve()),
        "prepared_row_count": expected_count,
        "expected_count": expected_count,
    }
    for field, expected in expected_values.items():
        _require_manifest_value(manifest, field, expected)

    actual_output_hash = sha256_file(output_jsonl)
    _require_manifest_value(
        manifest, "prepared_jsonl_sha256", actual_output_hash
    )
    _validate_prepared_jsonl(output_jsonl, expected_count)
    return manifest


def _temporary_sibling(path: Path):
    return tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )


def _write_jsonl_temporary(path: Path, rows: Sequence[Mapping]) -> Path:
    temporary_path: Path | None = None
    try:
        with _temporary_sibling(path) as handle:
            temporary_path = Path(handle.name)
            for row in rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        return temporary_path
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _write_manifest_temporary(path: Path, manifest: Mapping) -> Path:
    temporary_path: Path | None = None
    try:
        with _temporary_sibling(path) as handle:
            temporary_path = Path(handle.name)
            handle.write(
                json.dumps(manifest, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        return temporary_path
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


@contextmanager
def _exclusive_target_locks(paths: Sequence[Path]):
    """Hold persistent per-target locks in a process-independent order."""
    targets = sorted({path.resolve() for path in paths}, key=os.fspath)
    locked_file_descriptors: list[int] = []
    try:
        for target in targets:
            lock_path = target.with_name(f".{target.name}.lock")
            file_descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_RDWR,
                0o600,
            )
            try:
                fcntl.flock(file_descriptor, fcntl.LOCK_EX)
            except BaseException:
                os.close(file_descriptor)
                raise
            locked_file_descriptors.append(file_descriptor)
        yield
    finally:
        for file_descriptor in reversed(locked_file_descriptors):
            try:
                fcntl.flock(file_descriptor, fcntl.LOCK_UN)
            finally:
                os.close(file_descriptor)


def prepare_pool(
    source_parquet: Path,
    output_jsonl: Path,
    manifest_path: Path,
    original_rows: Sequence[Mapping],
    expected_count: int,
) -> dict:
    """Join the filtered parquet to pinned R1 solutions and atomically cache it."""
    source_parquet = Path(source_parquet)
    output_jsonl = Path(output_jsonl)
    manifest_path = Path(manifest_path)
    if output_jsonl.resolve() == manifest_path.resolve():
        raise ValueError("output_jsonl and manifest_path must be distinct paths")
    if expected_count < 0:
        raise ValueError("expected_count must be non-negative")

    source_table = pq.read_table(source_parquet)
    _validate_source_table(source_table, expected_count)
    source_rows = source_table.to_pylist()
    _validate_source_indices(source_rows)
    source_hash = sha256_file(source_parquet)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with _exclusive_target_locks((output_jsonl, manifest_path)):
        output_exists = output_jsonl.exists()
        manifest_exists = manifest_path.exists()
        if output_exists != manifest_exists:
            raise ValueError(
                "partial cache: prepared JSONL and manifest must either both exist or "
                "both be absent"
            )
        if output_exists:
            return _validate_cached_artifacts(
                source_parquet=source_parquet,
                source_sha256=source_hash,
                source_row_count=source_table.num_rows,
                output_jsonl=output_jsonl,
                manifest_path=manifest_path,
                expected_count=expected_count,
            )

        prepared_rows = join_filtered_rows(
            source_rows, original_rows, expected_count=expected_count
        )
        if len(prepared_rows) != expected_count:
            raise ValueError(
                "prepared row count mismatch: "
                f"expected {expected_count}, got {len(prepared_rows)}"
            )
        prepared_ids = [row.get("id") for row in prepared_rows]
        if len(set(prepared_ids)) != expected_count:
            raise ValueError("prepared IDs are not unique")

        output_temporary: Path | None = None
        manifest_temporary: Path | None = None
        output_replaced = False
        try:
            output_temporary = _write_jsonl_temporary(output_jsonl, prepared_rows)
            manifest = {
                "manifest_version": MANIFEST_VERSION,
                "dataset_name": DATASET_NAME,
                "dataset_revision": DATASET_REVISION,
                "dataset_split": DATASET_SPLIT,
                "source_parquet": str(source_parquet.resolve()),
                "source_sha256": source_hash,
                "source_row_count": source_table.num_rows,
                "prepared_jsonl": str(output_jsonl.resolve()),
                "prepared_jsonl_sha256": sha256_file(output_temporary),
                "prepared_row_count": len(prepared_rows),
                "expected_count": expected_count,
            }
            manifest_temporary = _write_manifest_temporary(manifest_path, manifest)

            if output_jsonl.exists() or manifest_path.exists():
                raise FileExistsError(
                    "cache targets appeared during preparation; refusing to overwrite"
                )
            output_temporary.replace(output_jsonl)
            output_replaced = True
            manifest_temporary.replace(manifest_path)
            return manifest
        except Exception:
            if output_replaced:
                output_jsonl.unlink(missing_ok=True)
            raise
        finally:
            if output_temporary is not None:
                output_temporary.unlink(missing_ok=True)
            if manifest_temporary is not None:
                manifest_temporary.unlink(missing_ok=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the pinned DeepMath R1 gradient-selection pool."
    )
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--dataset-name", choices=(DATASET_NAME,), default=DATASET_NAME
    )
    parser.add_argument(
        "--dataset-revision", choices=(DATASET_REVISION,), default=DATASET_REVISION
    )
    parser.add_argument("--expected-count", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> dict:
    args = _build_parser().parse_args(argv)
    from datasets import load_dataset

    original_rows = load_dataset(
        DATASET_NAME,
        split=DATASET_SPLIT,
        revision=DATASET_REVISION,
    )
    manifest = prepare_pool(
        args.source_parquet,
        args.output_jsonl,
        args.manifest,
        original_rows,
        expected_count=args.expected_count,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return manifest


if __name__ == "__main__":
    main()
