"""Build and validate the pinned proxy-model token-eligibility report."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from math_eval.deepmath_gradient_diversity import sha256_file


DATASET_NAME = "zwhe99/DeepMath-103K"
DATASET_REVISION = "5cf055d1fe3d7a2eb19719ac020211469736ae44"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
MAX_CONTEXT_TOKENS = 32768
REPORT_VERSION = 1
TOKENIZATION_BATCH_SIZE = 16
CONTEXT_EXCLUSION_REASON = "token_count_exceeds_max_context"
PRODUCTION_EXPECTED_EXCLUDED_IDS = ("deepmath-level6-038794",)
PRODUCTION_SOURCE_ROW_COUNT = 57046
PRODUCTION_ELIGIBLE_ROW_COUNT = 57045
PRODUCTION_EXCLUDED_ROW_COUNT = 1
PRODUCTION_MAX_EXCLUDED = 1
PRODUCTION_SOURCE_SHA256 = (
    "de3350fdd00bc0410550098ea65179e2be873da99e4075f80de575fc17670597"
)
PRODUCTION_PREPARED_JSONL_SHA256 = (
    "ee8d55c943577888f14052b3953a0a8aa07d26076e50ddffa41a102b60027344"
)
PRODUCTION_ELIGIBLE_IDS_SHA256 = (
    "5dbb267fed334219978d3d74610793fc8d146add15c488bcc59992779e22d8cf"
)
PRODUCTION_EXCLUDED_ROWS = (
    {
        "id": "deepmath-level6-038794",
        "source_row_index": 38794,
        "original_dataset_index": 62435,
        "token_count": 33634,
        "reason": CONTEXT_EXCLUSION_REASON,
    },
)
PRODUCTION_TOKEN_LENGTH_SUMMARY = {
    "count": 57046,
    "min": 564,
    "mean": 5805.74837850156,
    "p50": 4781.0,
    "p90": 10804.5,
    "p95": 13336.5,
    "p99": 18111.550000000003,
    "p999": 23639.245000000068,
    "max": 33634,
}
PINNED_CHAT_TEMPLATE_SHA256 = (
    "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"
)
TRANSFORMERS_VERSION = importlib.metadata.version("transformers")
REPORT_FIELDS = frozenset(
    {
        "manifest_version",
        "prepared_jsonl",
        "prepared_jsonl_sha256",
        "prepared_manifest",
        "prepared_manifest_sha256",
        "source_sha256",
        "source_row_count",
        "dataset_name",
        "dataset_revision",
        "model_name",
        "model_revision",
        "tokenizer_name",
        "tokenizer_revision",
        "transformers_version",
        "chat_template_sha256",
        "max_context_tokens",
        "max_excluded",
        "expected_excluded_ids",
        "eligible_row_count",
        "excluded_row_count",
        "eligible_ids_sha256",
        "excluded_rows",
        "token_length_summary",
    }
)
TOKEN_LENGTH_SUMMARY_FIELDS = frozenset(
    {"count", "min", "mean", "p50", "p90", "p95", "p99", "p999", "max"}
)


def _sha256_lines(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_equal(
    actual: object, expected: object, field: str, *, description: str
) -> None:
    if actual != expected:
        raise ValueError(
            f"{description} {field} mismatch: expected {expected!r}, got {actual!r}"
        )


def _validate_prepared_contract(
    rows: Sequence[Mapping], prepared_manifest: Mapping
) -> None:
    expected_count = len(rows)
    required_manifest_values = {
        "manifest_version": 1,
        "dataset_name": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "dataset_split": "train",
        "source_row_count": expected_count,
        "prepared_row_count": expected_count,
        "expected_count": expected_count,
    }
    for field, expected in required_manifest_values.items():
        _require_equal(
            prepared_manifest.get(field),
            expected,
            field,
            description="prepared manifest",
        )
    for field in ("source_sha256", "prepared_jsonl_sha256"):
        if not _is_sha256(prepared_manifest.get(field)):
            raise ValueError(f"prepared manifest {field} must be a lowercase SHA-256")
    for field in ("source_parquet", "prepared_jsonl"):
        value = prepared_manifest.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"prepared manifest {field} must be a nonempty path")

    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"prepared row {row_index} must be a mapping")
        expected_id = f"deepmath-level6-{row_index:06d}"
        if row.get("id") != expected_id:
            raise ValueError(
                f"prepared stable ID mismatch at row {row_index}: "
                f"expected {expected_id!r}, got {row.get('id')!r}"
            )
        if row.get("source_row_index") != row_index:
            raise ValueError(
                f"prepared source_row_index mismatch at row {row_index}"
            )
        if not isinstance(row.get("original_dataset_index"), int):
            raise ValueError(
                "prepared original_dataset_index must be an integer at row "
                f"{row_index}"
            )
        for text_field in ("prompt", "completion"):
            value = row.get(text_field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"prepared row {row_index} {text_field} must be nonempty text"
                )


def _linear_quantile(sorted_values: Sequence[int], quantile: float) -> float:
    position = (len(sorted_values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return float(
        sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
    )


def _token_length_summary(token_counts: Sequence[int]) -> dict[str, int | float]:
    if not token_counts:
        raise ValueError("prepared pool must contain at least one row")
    sorted_counts = sorted(token_counts)
    return {
        "count": len(sorted_counts),
        "min": sorted_counts[0],
        "mean": float(sum(sorted_counts) / len(sorted_counts)),
        "p50": _linear_quantile(sorted_counts, 0.50),
        "p90": _linear_quantile(sorted_counts, 0.90),
        "p95": _linear_quantile(sorted_counts, 0.95),
        "p99": _linear_quantile(sorted_counts, 0.99),
        "p999": _linear_quantile(sorted_counts, 0.999),
        "max": sorted_counts[-1],
    }


def _extract_input_ids(encoded: object, expected_batch_size: int) -> Sequence:
    if isinstance(encoded, Mapping):
        encoded = encoded.get("input_ids")
    if (
        not isinstance(encoded, Sequence)
        or isinstance(encoded, (str, bytes))
        or len(encoded) != expected_batch_size
    ):
        raise ValueError(
            "tokenizer chat-template result must contain one input_ids sequence "
            "per prepared row"
        )
    return encoded


def build_eligibility_report(
    rows: Sequence[Mapping],
    prepared_manifest: Mapping,
    tokenizer,
    max_context_tokens: int,
    max_excluded: int,
    expected_excluded_ids: Sequence[str],
) -> dict:
    """Tokenize the complete prepared pool and return its immutable report."""
    if max_context_tokens != MAX_CONTEXT_TOKENS:
        raise ValueError(
            "max_context_tokens must equal the pinned model boundary "
            f"{MAX_CONTEXT_TOKENS}, got {max_context_tokens}"
        )
    if (
        not isinstance(max_excluded, int)
        or isinstance(max_excluded, bool)
        or max_excluded < 0
    ):
        raise ValueError("max_excluded must be a non-negative integer")
    expected_ids = list(expected_excluded_ids)
    if (
        any(
            not isinstance(sample_id, str) or not sample_id
            for sample_id in expected_ids
        )
        or len(set(expected_ids)) != len(expected_ids)
    ):
        raise ValueError(
            "expected_excluded_ids must contain unique nonempty strings"
        )

    _validate_prepared_contract(rows, prepared_manifest)
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str) or not chat_template:
        raise ValueError("tokenizer chat_template must be nonempty text")
    tokenizer_name = getattr(tokenizer, "name_or_path", None)
    if tokenizer_name != MODEL_NAME:
        raise ValueError(
            f"tokenizer name mismatch: expected {MODEL_NAME!r}, got {tokenizer_name!r}"
        )

    token_counts: list[int] = []
    for batch_start in range(0, len(rows), TOKENIZATION_BATCH_SIZE):
        batch = rows[batch_start : batch_start + TOKENIZATION_BATCH_SIZE]
        conversations = [
            [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["completion"]},
            ]
            for row in batch
        ]
        encoded = tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=False,
            padding=False,
            truncation=False,
        )
        input_ids = _extract_input_ids(encoded, len(batch))
        for row_offset, tokens in enumerate(input_ids):
            try:
                token_count = len(tokens)
            except TypeError as error:
                raise ValueError(
                    "tokenizer input_ids must be sized sequences at prepared row "
                    f"{batch_start + row_offset}"
                ) from error
            if token_count <= 0:
                raise ValueError(
                    "tokenizer returned no tokens at prepared row "
                    f"{batch_start + row_offset}"
                )
            token_counts.append(token_count)

    excluded_rows: list[dict] = []
    eligible_ids: list[str] = []
    for row, token_count in zip(rows, token_counts, strict=True):
        if token_count > max_context_tokens:
            excluded_rows.append(
                {
                    "id": row["id"],
                    "source_row_index": row["source_row_index"],
                    "original_dataset_index": row["original_dataset_index"],
                    "token_count": token_count,
                    "reason": CONTEXT_EXCLUSION_REASON,
                }
            )
        else:
            eligible_ids.append(row["id"])

    excluded_ids = [entry["id"] for entry in excluded_rows]
    if len(excluded_rows) > max_excluded:
        raise ValueError(
            f"discovered {len(excluded_rows)} exclusions exceeds max_excluded "
            f"{max_excluded}: {excluded_ids!r}"
        )
    if set(excluded_ids) != set(expected_ids):
        raise ValueError(
            "excluded stable IDs mismatch: "
            f"expected {sorted(expected_ids)!r}, got {sorted(excluded_ids)!r}"
        )

    return {
        "manifest_version": REPORT_VERSION,
        "prepared_jsonl": prepared_manifest["prepared_jsonl"],
        "prepared_jsonl_sha256": prepared_manifest["prepared_jsonl_sha256"],
        "prepared_manifest": None,
        "prepared_manifest_sha256": None,
        "source_sha256": prepared_manifest["source_sha256"],
        "source_row_count": len(rows),
        "dataset_name": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "tokenizer_name": MODEL_NAME,
        "tokenizer_revision": MODEL_REVISION,
        "transformers_version": TRANSFORMERS_VERSION,
        "chat_template_sha256": hashlib.sha256(
            chat_template.encode("utf-8")
        ).hexdigest(),
        "max_context_tokens": max_context_tokens,
        "max_excluded": max_excluded,
        "expected_excluded_ids": sorted(expected_ids),
        "eligible_row_count": len(eligible_ids),
        "excluded_row_count": len(excluded_rows),
        "eligible_ids_sha256": _sha256_lines(eligible_ids),
        "excluded_rows": excluded_rows,
        "token_length_summary": _token_length_summary(token_counts),
    }


def _canonical_json(value: Mapping) -> tuple[dict, bytes]:
    try:
        serialized = json.dumps(
            dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        normalized = json.loads(serialized)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(
            f"eligibility report is not JSON serializable: {error}"
        ) from error
    if not isinstance(normalized, dict):
        raise ValueError("eligibility report must be a JSON object")
    return normalized, (serialized + "\n").encode("utf-8")


def _read_report(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid eligibility report {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(
            f"invalid eligibility report {path}: expected a JSON object"
        )
    return value


def write_or_validate_eligibility_report(path: Path, expected: Mapping) -> dict:
    """Atomically create a canonical report or validate an existing one."""
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    normalized, payload = _canonical_json(expected)
    lock_path = report_path.with_name(f".{report_path.name}.lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    temporary_path: Path | None = None
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        if report_path.exists():
            actual = _read_report(report_path)
            if actual != normalized:
                raise ValueError(
                    "eligibility report mismatch: existing provenance or "
                    "eligibility differs from the requested report"
                )
            return actual

        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=report_path.parent,
            prefix=f".{report_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if report_path.exists():
            actual = _read_report(report_path)
            if actual != normalized:
                raise ValueError(
                    "eligibility report mismatch: target appeared with "
                    "different content"
                )
            return actual
        temporary_path.replace(report_path)
        temporary_path = None
        return normalized
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _validate_token_summary(summary: object, source_row_count: int) -> None:
    if not isinstance(summary, Mapping):
        raise ValueError(
            "eligibility report token_length_summary must be an object"
        )
    if set(summary) != TOKEN_LENGTH_SUMMARY_FIELDS:
        raise ValueError(
            "eligibility report token_length_summary fields mismatch: "
            f"expected {sorted(TOKEN_LENGTH_SUMMARY_FIELDS)!r}, "
            f"got {sorted(summary)!r}"
        )
    _require_equal(
        summary.get("count"),
        source_row_count,
        "token_length_summary.count",
        description="eligibility report",
    )
    fields = ("min", "mean", "p50", "p90", "p95", "p99", "p999", "max")
    for field in fields:
        value = summary.get(field)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(
                "eligibility report token_length_summary."
                f"{field} must be finite and positive"
            )
    quantiles = [
        float(summary[field])
        for field in ("min", "p50", "p90", "p95", "p99", "p999", "max")
    ]
    if quantiles != sorted(quantiles):
        raise ValueError(
            "eligibility report token-length quantiles are not monotonic"
        )
    mean = float(summary["mean"])
    if mean < float(summary["min"]) or mean > float(summary["max"]):
        raise ValueError(
            "eligibility report token-length mean is outside [min, max]"
        )


def _validate_production_semantics(report: Mapping) -> None:
    expected_scalars = {
        "source_row_count": PRODUCTION_SOURCE_ROW_COUNT,
        "eligible_row_count": PRODUCTION_ELIGIBLE_ROW_COUNT,
        "excluded_row_count": PRODUCTION_EXCLUDED_ROW_COUNT,
        "max_excluded": PRODUCTION_MAX_EXCLUDED,
        "source_sha256": PRODUCTION_SOURCE_SHA256,
        "prepared_jsonl_sha256": PRODUCTION_PREPARED_JSONL_SHA256,
        "eligible_ids_sha256": PRODUCTION_ELIGIBLE_IDS_SHA256,
        "expected_excluded_ids": list(PRODUCTION_EXPECTED_EXCLUDED_IDS),
    }
    for field, expected in expected_scalars.items():
        _require_equal(
            report.get(field), expected, field, description="production"
        )
    expected_excluded_rows = [
        dict(entry) for entry in PRODUCTION_EXCLUDED_ROWS
    ]
    if report.get("excluded_rows") != expected_excluded_rows:
        raise ValueError(
            "production excluded_rows mismatch: expected "
            f"{expected_excluded_rows!r}, got {report.get('excluded_rows')!r}"
        )
    if report.get("token_length_summary") != PRODUCTION_TOKEN_LENGTH_SUMMARY:
        raise ValueError(
            "production token_length_summary mismatch: expected "
            f"{PRODUCTION_TOKEN_LENGTH_SUMMARY!r}, "
            f"got {report.get('token_length_summary')!r}"
        )


def apply_eligibility_report(
    rows: Sequence[Mapping],
    prepared_manifest: Mapping,
    report_path: Path,
    *,
    prepared_manifest_path: Path | None = None,
    expected_excluded_ids: Sequence[str] | None = None,
) -> tuple[list[dict], dict]:
    """Validate a report and return eligible rows in unchanged prepared order."""
    _validate_prepared_contract(rows, prepared_manifest)
    report = _read_report(Path(report_path))
    if set(report) != REPORT_FIELDS:
        raise ValueError(
            "eligibility report top-level fields mismatch: "
            f"expected {sorted(REPORT_FIELDS)!r}, got {sorted(report)!r}"
        )
    expected_values = {
        "manifest_version": REPORT_VERSION,
        "prepared_jsonl": prepared_manifest["prepared_jsonl"],
        "prepared_jsonl_sha256": prepared_manifest["prepared_jsonl_sha256"],
        "source_sha256": prepared_manifest["source_sha256"],
        "source_row_count": len(rows),
        "dataset_name": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "tokenizer_name": MODEL_NAME,
        "tokenizer_revision": MODEL_REVISION,
        "transformers_version": TRANSFORMERS_VERSION,
        "chat_template_sha256": PINNED_CHAT_TEMPLATE_SHA256,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
    }
    for field, expected in expected_values.items():
        _require_equal(
            report.get(field), expected, field, description="eligibility report"
        )
    production_mode = expected_excluded_ids is None
    if production_mode:
        _validate_production_semantics(report)

    manifest_path_value = report.get("prepared_manifest")
    manifest_hash_value = report.get("prepared_manifest_sha256")
    if manifest_path_value is None:
        if manifest_hash_value is not None:
            raise ValueError(
                "eligibility report prepared_manifest_sha256 requires "
                "prepared_manifest"
            )
    else:
        if not isinstance(manifest_path_value, str) or not manifest_path_value:
            raise ValueError(
                "eligibility report prepared_manifest must be a path or null"
            )
        manifest_path = Path(manifest_path_value)
        if not manifest_path.is_absolute() or not manifest_path.is_file():
            raise ValueError(
                "eligibility report prepared_manifest does not exist: "
                f"{manifest_path_value!r}"
            )
        _require_equal(
            manifest_hash_value,
            sha256_file(manifest_path),
            "prepared_manifest_sha256",
            description="eligibility report",
        )
    if prepared_manifest_path is not None:
        caller_manifest_path = Path(prepared_manifest_path).resolve()
        _require_equal(
            manifest_path_value,
            str(caller_manifest_path),
            "prepared_manifest path",
            description="eligibility report",
        )
        _require_equal(
            manifest_hash_value,
            sha256_file(caller_manifest_path),
            "prepared_manifest_sha256",
            description="eligibility report",
        )
        actual_manifest = _read_prepared_manifest(caller_manifest_path)
        if actual_manifest != dict(prepared_manifest):
            raise ValueError(
                "caller's prepared manifest content differs from the "
                "validated manifest mapping"
            )

    excluded_rows = report.get("excluded_rows")
    if not isinstance(excluded_rows, list):
        raise ValueError("eligibility report excluded_rows must be a list")
    excluded_ids: list[str] = []
    excluded_source_indices: list[int] = []
    for excluded_index, entry in enumerate(excluded_rows):
        if not isinstance(entry, Mapping):
            raise ValueError(
                "eligibility report excluded_rows"
                f"[{excluded_index}] must be an object"
            )
        required_keys = {
            "id",
            "source_row_index",
            "original_dataset_index",
            "token_count",
            "reason",
        }
        if set(entry) != required_keys:
            raise ValueError(
                "eligibility report excluded_rows"
                f"[{excluded_index}] fields mismatch"
            )
        source_index = entry.get("source_row_index")
        if (
            not isinstance(source_index, int)
            or isinstance(source_index, bool)
            or source_index < 0
            or source_index >= len(rows)
        ):
            raise ValueError(
                "eligibility report excluded source_row_index is invalid"
            )
        source_row = rows[source_index]
        for field in ("id", "source_row_index", "original_dataset_index"):
            _require_equal(
                entry.get(field),
                source_row.get(field),
                field,
                description=(
                    "eligibility report "
                    f"excluded_rows[{excluded_index}]"
                ),
            )
        if entry.get("reason") != CONTEXT_EXCLUSION_REASON:
            raise ValueError(
                "eligibility report exclusion reason mismatch: expected "
                f"{CONTEXT_EXCLUSION_REASON!r}, got {entry.get('reason')!r}"
            )
        token_count = entry.get("token_count")
        if (
            not isinstance(token_count, int)
            or isinstance(token_count, bool)
            or token_count <= MAX_CONTEXT_TOKENS
        ):
            raise ValueError(
                "eligibility report excluded token_count must exceed "
                "max_context_tokens"
            )
        excluded_ids.append(entry["id"])
        excluded_source_indices.append(source_index)

    if len(set(excluded_ids)) != len(excluded_ids):
        raise ValueError("eligibility report contains duplicate excluded IDs")
    if excluded_source_indices != sorted(excluded_source_indices):
        raise ValueError(
            "eligibility report excluded rows are not in prepared order"
        )
    required_excluded_ids = list(
        PRODUCTION_EXPECTED_EXCLUDED_IDS
        if production_mode
        else expected_excluded_ids
    )
    if (
        any(
            not isinstance(sample_id, str) or not sample_id
            for sample_id in required_excluded_ids
        )
        or len(set(required_excluded_ids)) != len(required_excluded_ids)
    ):
        raise ValueError(
            "expected_excluded_ids must contain unique nonempty strings"
        )
    if sorted(excluded_ids) != sorted(required_excluded_ids):
        raise ValueError(
            "production excluded stable IDs mismatch: expected "
            f"{sorted(required_excluded_ids)!r}, got {sorted(excluded_ids)!r}"
        )
    max_excluded = report.get("max_excluded")
    if (
        not isinstance(max_excluded, int)
        or isinstance(max_excluded, bool)
        or max_excluded != len(required_excluded_ids)
    ):
        raise ValueError(
            "eligibility report max_excluded mismatch: expected "
            f"{len(required_excluded_ids)}, got {max_excluded!r}"
        )
    expected_excluded_ids = report.get("expected_excluded_ids")
    if expected_excluded_ids != sorted(required_excluded_ids):
        raise ValueError(
            "eligibility report expected_excluded_ids mismatch"
        )
    _require_equal(
        report.get("excluded_row_count"),
        len(excluded_ids),
        "excluded_row_count",
        description="eligibility report",
    )

    excluded_id_set = set(excluded_ids)
    eligible_rows = [
        dict(row) for row in rows if row["id"] not in excluded_id_set
    ]
    eligible_ids = [row["id"] for row in eligible_rows]
    _require_equal(
        report.get("eligible_row_count"),
        len(eligible_rows),
        "eligible_row_count",
        description="eligibility report",
    )
    _require_equal(
        report.get("eligible_ids_sha256"),
        _sha256_lines(eligible_ids),
        "eligible_ids_sha256",
        description="eligibility report",
    )
    _validate_token_summary(report.get("token_length_summary"), len(rows))
    return eligible_rows, report


def _read_prepared_manifest(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid prepared manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(
            f"invalid prepared manifest {path}: expected a JSON object"
        )
    return value


def load_prepared_pool(
    prepared_jsonl: Path, prepared_manifest_path: Path
) -> tuple[list[dict], dict]:
    """Load the complete pool and validate its file-level provenance."""
    prepared_path = Path(prepared_jsonl)
    manifest_path = Path(prepared_manifest_path)
    manifest = _read_prepared_manifest(manifest_path)
    _require_equal(
        manifest.get("prepared_jsonl"),
        str(prepared_path.resolve()),
        "prepared_jsonl",
        description="prepared manifest",
    )
    if not prepared_path.is_file():
        raise ValueError(f"prepared JSONL does not exist: {prepared_path}")
    _require_equal(
        manifest.get("prepared_jsonl_sha256"),
        sha256_file(prepared_path),
        "prepared_jsonl_sha256",
        description="prepared manifest",
    )

    source_value = manifest.get("source_parquet")
    if not isinstance(source_value, str) or not source_value:
        raise ValueError(
            "prepared manifest source_parquet must be a nonempty path"
        )
    source_path = Path(source_value)
    if not source_path.is_absolute() or not source_path.is_file():
        raise ValueError(
            f"prepared manifest source_parquet does not exist: {source_value!r}"
        )
    _require_equal(
        manifest.get("source_sha256"),
        sha256_file(source_path),
        "source_sha256",
        description="prepared manifest",
    )

    rows: list[dict] = []
    try:
        with prepared_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(
                        f"prepared JSONL contains blank line {line_number}"
                    )
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(
                        f"prepared JSONL row {line_number} must be an object"
                    )
                rows.append(row)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid prepared JSONL {prepared_path}: {error}") from error
    _validate_prepared_contract(rows, manifest)
    return rows, manifest


def load_pinned_tokenizer(model_name: str, model_revision: str):
    """Load the tokenizer only after validating the pinned model config."""
    if model_name != MODEL_NAME:
        raise ValueError(
            f"pinned model name mismatch: expected {MODEL_NAME}, got {model_name}"
        )
    if model_revision != MODEL_REVISION:
        raise ValueError(
            "pinned model revision mismatch: "
            f"expected {MODEL_REVISION}, got {model_revision}"
        )
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(model_name, revision=model_revision)
    configured_context = getattr(config, "max_position_embeddings", None)
    if configured_context != MAX_CONTEXT_TOKENS:
        raise ValueError(
            "pinned model max_position_embeddings mismatch: expected "
            f"{MAX_CONTEXT_TOKENS}, got {configured_context!r}"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=model_revision
    )
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str) or not chat_template:
        raise ValueError("pinned tokenizer chat_template must be nonempty text")
    actual_template_hash = hashlib.sha256(
        chat_template.encode("utf-8")
    ).hexdigest()
    if actual_template_hash != PINNED_CHAT_TEMPLATE_SHA256:
        raise ValueError(
            "pinned tokenizer chat_template hash mismatch: expected "
            f"{PINNED_CHAT_TEMPLATE_SHA256}, got {actual_template_hash}"
        )
    return tokenizer


def run_eligibility_build(
    *,
    prepared_jsonl: Path,
    prepared_manifest_path: Path,
    output_path: Path,
    model_name: str,
    model_revision: str,
    max_context_tokens: int,
    max_excluded: int,
    expected_excluded_ids: Sequence[str],
) -> dict:
    """Build or validate one eligibility report from prepared artifacts."""
    if sorted(expected_excluded_ids) != sorted(
        PRODUCTION_EXPECTED_EXCLUDED_IDS
    ):
        raise ValueError(
            "production expected excluded stable IDs mismatch: expected "
            f"{list(PRODUCTION_EXPECTED_EXCLUDED_IDS)!r}, got "
            f"{sorted(expected_excluded_ids)!r}"
        )
    if max_excluded != len(PRODUCTION_EXPECTED_EXCLUDED_IDS):
        raise ValueError(
            "production max_excluded mismatch: expected "
            f"{len(PRODUCTION_EXPECTED_EXCLUDED_IDS)}, got {max_excluded}"
        )
    prepared_path = Path(prepared_jsonl).resolve()
    manifest_path = Path(prepared_manifest_path).resolve()
    report_path = Path(output_path).resolve()
    if len({prepared_path, manifest_path, report_path}) != 3:
        raise ValueError(
            "prepared JSONL, prepared manifest, and eligibility output "
            "must be distinct paths"
        )
    rows, manifest = load_prepared_pool(prepared_path, manifest_path)
    if report_path == Path(manifest["source_parquet"]).resolve():
        raise ValueError(
            "eligibility output overlaps source parquet: "
            f"{report_path}"
        )
    tokenizer = load_pinned_tokenizer(model_name, model_revision)
    report = build_eligibility_report(
        rows,
        manifest,
        tokenizer,
        max_context_tokens=max_context_tokens,
        max_excluded=max_excluded,
        expected_excluded_ids=expected_excluded_ids,
    )
    report["prepared_manifest"] = str(manifest_path)
    report["prepared_manifest_sha256"] = sha256_file(manifest_path)
    written = write_or_validate_eligibility_report(report_path, report)
    apply_eligibility_report(
        rows,
        manifest,
        report_path,
        prepared_manifest_path=manifest_path,
    )
    return written


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the pinned Qwen proxy-context eligibility report for "
            "the prepared DeepMath pool."
        )
    )
    parser.add_argument("--prepared-jsonl", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument(
        "--max-context-tokens", type=int, default=MAX_CONTEXT_TOKENS
    )
    parser.add_argument("--max-excluded", type=int, default=1)
    parser.add_argument(
        "--expected-excluded-id",
        action="append",
        default=[],
        dest="expected_excluded_ids",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_eligibility_build(
        prepared_jsonl=args.prepared_jsonl,
        prepared_manifest_path=args.prepared_manifest,
        output_path=args.output,
        model_name=args.model_name,
        model_revision=args.model_revision,
        max_context_tokens=args.max_context_tokens,
        max_excluded=args.max_excluded,
        expected_excluded_ids=args.expected_excluded_ids,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "source_row_count": report["source_row_count"],
                "eligible_row_count": report["eligible_row_count"],
                "excluded_row_count": report["excluded_row_count"],
                "excluded_ids": [
                    entry["id"] for entry in report["excluded_rows"]
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
