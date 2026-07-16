"""Fail-closed validation for the gradient-diverse OPD training artifact."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from math_eval.deepmath_gradient_diversity import sha256_file


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
        if (
            not isinstance(role, str)
            or not role
            or not isinstance(content, str)
            or not content
        ):
            raise ValueError(
                f"row {row_index} message {message_index} requires nonempty "
                "role/content strings"
            )
        normalized.append({"role": role, "content": content})
    return normalized


def _load_manifest(path: Path) -> dict:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid selection manifest JSON: {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError("selection manifest must be a JSON object")
    return manifest


def _require_manifest_count(manifest: dict, key: str, expected: int) -> None:
    value = manifest.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ValueError(f"manifest {key} mismatch: expected {expected}, got {value!r}")


def _load_selected_ids(path: Path) -> list[dict]:
    rows = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"invalid selected IDs file: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid selected ID JSON on line {line_number}: {error.msg}"
            ) from error
        if not isinstance(row, dict):
            raise ValueError(f"selected ID line {line_number} must be a JSON object")
        rows.append(row)
    return rows


def _nearest_rank(lengths: list[int], quantile: float) -> int:
    sorted_lengths = sorted(lengths)
    return sorted_lengths[math.ceil(quantile * len(sorted_lengths)) - 1]


def _validate_nested_row(row: dict, row_index: int) -> list[dict[str, str]]:
    prompt = _validate_prompt(row.get("prompt"), row_index)

    reward_model = row.get("reward_model")
    if not isinstance(reward_model, dict):
        raise ValueError(f"row {row_index} reward_model must be an object")
    for key in ("ground_truth", "style"):
        value = reward_model.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"row {row_index} reward_model.{key} must be a nonempty string"
            )

    extra_info = row.get("extra_info")
    if not isinstance(extra_info, dict):
        raise ValueError(f"row {row_index} extra_info must be an object")
    extra_index = extra_info.get("index")
    if isinstance(extra_index, bool) or not isinstance(extra_index, int):
        raise ValueError(f"row {row_index} extra_info.index must be an integer")
    split = extra_info.get("split")
    if not isinstance(split, str) or not split:
        raise ValueError(f"row {row_index} extra_info.split must be a nonempty string")

    return prompt


def ensure_empty_checkpoint_dir(checkpoint_dir: Path) -> None:
    """Reject an existing checkpoint target unless it is an empty directory."""
    if not checkpoint_dir.exists():
        return
    if not checkpoint_dir.is_dir():
        raise ValueError(f"checkpoint path is not a directory: {checkpoint_dir}")
    if any(checkpoint_dir.iterdir()):
        raise ValueError(f"checkpoint directory is non-empty: {checkpoint_dir}")


def validate_training_artifact(
    selected_parquet: Path,
    source_parquet: Path,
    selection_manifest: Path,
    selected_ids_path: Path,
    diagnostics_path: Path,
    tokenizer: object,
    *,
    expected_selected_sha256: str,
    expected_source_sha256: str,
    expected_manifest_sha256: str,
    expected_selected_ids_sha256: str,
    expected_diagnostics_sha256: str,
    expected_rows: int,
    expected_source_rows: int,
    expected_eligible_rows: int,
    max_prompt_tokens: int,
    expected_selection_profile: str | None = None,
    frozen_prefix_selected_ids_path: Path | None = None,
    expected_frozen_prefix_sha256: str | None = None,
    expected_frozen_prefix_rows: int | None = None,
) -> dict[str, object]:
    """Validate exact training data provenance before a training launch."""
    profile_values = (
        expected_selection_profile,
        frozen_prefix_selected_ids_path,
        expected_frozen_prefix_sha256,
        expected_frozen_prefix_rows,
    )
    profile_enabled = all(value is not None for value in profile_values)
    if any(value is not None for value in profile_values) and not profile_enabled:
        raise ValueError(
            "selection profile and frozen prefix arguments must be provided together"
        )
    if profile_enabled:
        if not isinstance(expected_selection_profile, str) or not expected_selection_profile:
            raise ValueError("expected selection profile must be a nonempty string")
        if (
            isinstance(expected_frozen_prefix_rows, bool)
            or not isinstance(expected_frozen_prefix_rows, int)
            or expected_frozen_prefix_rows <= 0
        ):
            raise ValueError("expected frozen prefix rows must be positive")

    artifact_paths = {
        "selected parquet": selected_parquet,
        "source parquet": source_parquet,
        "selection manifest": selection_manifest,
        "selected IDs": selected_ids_path,
        "diagnostics": diagnostics_path,
    }
    if profile_enabled:
        artifact_paths["frozen prefix selected IDs"] = frozen_prefix_selected_ids_path
    for label, path in artifact_paths.items():
        if not path.is_file():
            raise ValueError(f"{label} is not a regular file: {path}")

    selected_sha256 = sha256_file(selected_parquet)
    source_sha256 = sha256_file(source_parquet)
    manifest_sha256 = sha256_file(selection_manifest)
    selected_ids_sha256 = sha256_file(selected_ids_path)
    diagnostics_sha256 = sha256_file(diagnostics_path)
    if selected_sha256 != expected_selected_sha256:
        raise ValueError(
            "selected parquet SHA-256 mismatch: "
            f"expected {expected_selected_sha256}, got {selected_sha256}"
        )
    if source_sha256 != expected_source_sha256:
        raise ValueError(
            "source parquet SHA-256 mismatch: "
            f"expected {expected_source_sha256}, got {source_sha256}"
        )
    if manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            "selection manifest SHA-256 mismatch: "
            f"expected {expected_manifest_sha256}, got {manifest_sha256}"
        )
    if selected_ids_sha256 != expected_selected_ids_sha256:
        raise ValueError(
            "selected IDs SHA-256 mismatch: "
            f"expected {expected_selected_ids_sha256}, got {selected_ids_sha256}"
        )
    if diagnostics_sha256 != expected_diagnostics_sha256:
        raise ValueError(
            "diagnostics SHA-256 mismatch: "
            f"expected {expected_diagnostics_sha256}, got {diagnostics_sha256}"
        )

    manifest = _load_manifest(selection_manifest)
    version = manifest.get("manifest_version")
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise ValueError(f"manifest_version mismatch: expected 1, got {version!r}")
    if manifest.get("output_parquet_sha256") != selected_sha256:
        raise ValueError("manifest selected parquet SHA-256 mismatch")
    if manifest.get("source_sha256") != source_sha256:
        raise ValueError("manifest source parquet SHA-256 mismatch")

    expected_paths = {
        "output_parquet": selected_parquet,
        "source_parquet": source_parquet,
        "selected_ids": selected_ids_path,
        "diagnostics": diagnostics_path,
    }
    for key, path in expected_paths.items():
        expected_path = str(path.resolve())
        if manifest.get(key) != expected_path:
            raise ValueError(
                f"manifest {key} path mismatch: expected {expected_path}, "
                f"got {manifest.get(key)!r}"
            )

    _require_manifest_count(manifest, "selected_row_count", expected_rows)
    _require_manifest_count(manifest, "source_row_count", expected_source_rows)
    _require_manifest_count(manifest, "eligible_row_count", expected_eligible_rows)

    if manifest.get("selected_ids_sha256") != selected_ids_sha256:
        raise ValueError("manifest selected IDs SHA-256 mismatch")
    if manifest.get("diagnostics_sha256") != diagnostics_sha256:
        raise ValueError("manifest diagnostics SHA-256 mismatch")

    id_rows = _load_selected_ids(selected_ids_path)
    if len(id_rows) != expected_rows:
        raise ValueError(
            f"selected ID row count mismatch: expected {expected_rows}, got {len(id_rows)}"
        )

    prefix_rows: list[dict] | None = None
    prefix_path: Path | None = None
    if profile_enabled:
        prefix_path = Path(frozen_prefix_selected_ids_path).resolve()
        actual_prefix_sha256 = sha256_file(prefix_path)
        if actual_prefix_sha256 != expected_frozen_prefix_sha256:
            raise ValueError(
                "frozen prefix SHA-256 mismatch: "
                f"expected {expected_frozen_prefix_sha256}, "
                f"got {actual_prefix_sha256}"
            )
        prefix_rows = _load_selected_ids(prefix_path)
        if len(prefix_rows) != expected_frozen_prefix_rows:
            raise ValueError(
                "frozen prefix row count mismatch: "
                f"expected {expected_frozen_prefix_rows}, got {len(prefix_rows)}"
            )
        if id_rows[:expected_frozen_prefix_rows] != prefix_rows:
            raise ValueError("selected IDs do not preserve the exact prefix")

        selection = manifest.get("selection")
        if not isinstance(selection, dict):
            raise ValueError("manifest selection profile contract is missing")
        if selection.get("profile") != expected_selection_profile:
            raise ValueError(
                "manifest selection profile mismatch: "
                f"expected {expected_selection_profile!r}, "
                f"got {selection.get('profile')!r}"
            )
        if selection.get("target_size") != expected_rows:
            raise ValueError("manifest selection profile target_size mismatch")
        expected_prefix_contract = {
            "path": str(prefix_path),
            "row_count": expected_frozen_prefix_rows,
            "sha256": expected_frozen_prefix_sha256,
            "selected_id_sequence_sha256": _id_sequence_sha256(prefix_rows),
        }
        if selection.get("frozen_prefix") != expected_prefix_contract:
            raise ValueError("manifest frozen prefix contract mismatch")

    ids = []
    source_indices = []
    eligible_positions = []
    for row_index, row in enumerate(id_rows):
        selected_id = row.get("id")
        if not isinstance(selected_id, str) or not selected_id:
            raise ValueError(f"selected ID row {row_index} requires a nonempty string id")
        source_index = row.get("source_row_index")
        if isinstance(source_index, bool) or not isinstance(source_index, int):
            raise ValueError(
                f"selected ID row {row_index} source_row_index must be an integer"
            )
        if source_index < 0 or source_index >= expected_source_rows:
            raise ValueError(
                f"selected ID row {row_index} source_row_index is out of range"
            )
        expected_id = f"deepmath-level6-{source_index:06d}"
        if selected_id != expected_id:
            raise ValueError(
                f"selected ID row {row_index} stable ID mismatch: "
                f"expected {expected_id!r}, got {selected_id!r}"
            )
        eligible_position = row.get("eligible_position")
        if isinstance(eligible_position, bool) or not isinstance(eligible_position, int):
            raise ValueError(
                f"selected ID row {row_index} eligible_position must be an integer"
            )
        ids.append(selected_id)
        source_indices.append(source_index)
        eligible_positions.append(eligible_position)

    if len(set(ids)) != len(ids):
        raise ValueError("selected IDs must be unique")
    if len(set(source_indices)) != len(source_indices):
        raise ValueError("selected source_row_index values must be unique")
    if len(set(eligible_positions)) != len(eligible_positions):
        raise ValueError("selected eligible_position values must be unique")
    if manifest.get("selected_id_sequence_sha256") != _id_sequence_sha256(id_rows):
        raise ValueError("selected ID sequence SHA-256 mismatch")

    try:
        source = pq.read_table(source_parquet)
        selected = pq.read_table(selected_parquet)
    except Exception as error:
        raise ValueError(f"unable to read parquet artifacts: {error}") from error

    if source.num_rows != expected_source_rows:
        raise ValueError(
            f"source parquet row count mismatch: expected {expected_source_rows}, "
            f"got {source.num_rows}"
        )
    if selected.num_rows != expected_rows:
        raise ValueError(
            f"selected parquet row count mismatch: expected {expected_rows}, "
            f"got {selected.num_rows}"
        )

    for label, table in (("source", source), ("selected", selected)):
        missing = [column for column in REQUIRED_COLUMNS if column not in table.column_names]
        if missing:
            raise ValueError(f"{label} parquet is missing required columns: {missing}")
    if not selected.schema.equals(source.schema, check_metadata=True):
        raise ValueError("selected and source parquet schemas are not exactly equal")
    for label, table in (("source", source), ("selected", selected)):
        for column in REQUIRED_COLUMNS:
            if table[column].null_count:
                raise ValueError(f"{label} parquet column {column} contains nulls")

    expected = source.take(pa.array(source_indices, type=pa.int64()))
    if not selected.equals(expected, check_metadata=True):
        raise ValueError("selected parquet does not equal source.take(source_row_index)")

    selected_rows = selected.to_pylist()
    prompts = []
    serialized_prompts = []
    for row_index, row in enumerate(selected_rows):
        prompt = _validate_nested_row(row, row_index)
        prompts.append(prompt)
        serialized_prompts.append(_serialized_prompt(prompt))
    if len(set(serialized_prompts)) != expected_rows:
        raise ValueError("exact prompts must be unique")

    if max_prompt_tokens <= 0:
        raise ValueError("max_prompt_tokens must be positive")
    token_lengths = []
    for start in range(0, len(prompts), TOKENIZATION_BATCH_SIZE):
        batch = prompts[start : start + TOKENIZATION_BATCH_SIZE]
        token_ids = tokenizer.apply_chat_template(
            batch,
            tokenize=True,
            add_generation_prompt=True,
            padding=False,
            truncation=False,
            enable_thinking=False,
        )
        if not isinstance(token_ids, list) or len(token_ids) != len(batch):
            raise ValueError("tokenizer must return one token list per prompt")
        for batch_index, tokens in enumerate(token_ids):
            if not isinstance(tokens, list):
                raise ValueError(
                    "tokenizer must return one token list per prompt: "
                    f"batch item {batch_index} is not a list"
                )
            token_lengths.append(len(tokens))

    prompts_over_limit = sum(length > max_prompt_tokens for length in token_lengths)
    if prompts_over_limit:
        raise ValueError(
            "prompt token limit exceeded: "
            f"{prompts_over_limit} prompt(s) exceed {max_prompt_tokens} tokens"
        )

    report: dict[str, object] = {
        "selected_parquet": str(selected_parquet.resolve()),
        "selected_sha256": selected_sha256,
        "selected_rows": selected.num_rows,
        "source_parquet": str(source_parquet.resolve()),
        "source_sha256": source_sha256,
        "source_rows": source.num_rows,
        "selection_manifest": str(selection_manifest.resolve()),
        "selection_manifest_sha256": manifest_sha256,
        "selected_ids_path": str(selected_ids_path.resolve()),
        "selected_ids_sha256": selected_ids_sha256,
        "diagnostics": str(diagnostics_path.resolve()),
        "diagnostics_sha256": diagnostics_sha256,
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
    if profile_enabled:
        report.update(
            {
                "selection_profile": expected_selection_profile,
                "frozen_prefix_rows": expected_frozen_prefix_rows,
                "frozen_prefix_path": str(prefix_path),
                "frozen_prefix_sha256": expected_frozen_prefix_sha256,
            }
        )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-parquet", type=Path, required=True)
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--selected-ids", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--expected-selected-sha256", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-selected-ids-sha256", required=True)
    parser.add_argument("--expected-diagnostics-sha256", required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--expected-source-rows", type=int, required=True)
    parser.add_argument("--expected-eligible-rows", type=int, required=True)
    parser.add_argument("--max-prompt-tokens", type=int, required=True)
    parser.add_argument("--expected-selection-profile")
    parser.add_argument("--frozen-prefix-selected-ids", type=Path)
    parser.add_argument("--expected-frozen-prefix-sha256")
    parser.add_argument("--expected-frozen-prefix-rows", type=int)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    ensure_empty_checkpoint_dir(args.checkpoint_dir)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=True,
    )
    report = validate_training_artifact(
        args.selected_parquet,
        args.source_parquet,
        args.selection_manifest,
        args.selected_ids,
        args.diagnostics,
        tokenizer,
        expected_selected_sha256=args.expected_selected_sha256,
        expected_source_sha256=args.expected_source_sha256,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_selected_ids_sha256=args.expected_selected_ids_sha256,
        expected_diagnostics_sha256=args.expected_diagnostics_sha256,
        expected_rows=args.expected_rows,
        expected_source_rows=args.expected_source_rows,
        expected_eligible_rows=args.expected_eligible_rows,
        max_prompt_tokens=args.max_prompt_tokens,
        expected_selection_profile=args.expected_selection_profile,
        frozen_prefix_selected_ids_path=args.frozen_prefix_selected_ids,
        expected_frozen_prefix_sha256=args.expected_frozen_prefix_sha256,
        expected_frozen_prefix_rows=args.expected_frozen_prefix_rows,
    )
    ensure_empty_checkpoint_dir(args.checkpoint_dir)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
