"""Collect immutable completion-only gradients for Prismatic-lite additions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

if __package__ in (None, ""):
    repository_root = str(Path(__file__).resolve().parents[1])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)

from math_eval.collect_prismatic_gradients import (
    ASSISTANT_RESPONSE_MARKER,
    FAST_JL_VERSION,
    GRADIENT_MANIFEST_NAME,
    MAX_CONTEXT_TOKENS,
    MODEL_NAME,
    MODEL_REVISION,
    PROJECTED_DTYPE_NAME,
    PROJECTOR_BACKEND,
    PROJECTION_DIM,
    PROJECTION_INPUT_DTYPE_NAME,
    PROJECTION_SEED,
    PROJECT_INTERVAL,
    REFERENCE_COMMIT,
    SAVE_INTERVAL,
    TRAKER_VERSION,
    _exclusive_shard_lock,
    _installed_package_versions,
    _installed_trl_version,
    _official_gradient_module_path,
    _reference_tree,
    _validate_gradient_prefix,
    construct_strict_collector,
    load_model_and_tokenizer,
    load_official_gradient_computer_class,
    preflight_samples,
    resolve_resume_start,
    validate_global_gradient_coverage,
    validate_single_cuda_device,
    verify_reference_repo,
    write_or_validate_gradient_manifest,
)
from math_eval.deepmath_gradient_diversity import official_shard_bounds
from math_eval.prismatic_lite_pilot_artifacts import (
    sha256_file,
    strict_json_file,
    strict_jsonl,
)


PILOT_INPUT_FIELDS = (
    "id",
    "question_id",
    "solution_index",
    "prompt",
    "completion",
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_CHUNK_PATTERN_TEMPLATE = r"^{prefix}\.(0|[1-9][0-9]*)\.(txt|safetensors)$"


def _sha256(value: str, description: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{description} must be a lowercase SHA-256 digest")
    return value


def load_pilot_gradient_input(path: Path, expected_sha256: str) -> list[dict]:
    input_path = Path(path).resolve()
    expected = _sha256(expected_sha256, "input SHA-256")
    if not input_path.is_file():
        raise ValueError(f"pilot gradient input does not exist: {input_path}")
    actual = sha256_file(input_path)
    if actual != expected:
        raise ValueError(
            f"pilot gradient input SHA-256 mismatch: expected {expected}, got {actual}"
        )
    rows = strict_jsonl(input_path, exact_fields=PILOT_INPUT_FIELDS)
    question_rows: dict[str, list[tuple[int, str]]] = defaultdict(list)
    question_prompts: dict[str, str] = {}
    for row_number, row in enumerate(rows, start=1):
        question_id = row["question_id"]
        solution_index = row["solution_index"]
        prompt = row["prompt"]
        completion = row["completion"]
        if not isinstance(question_id, str) or not question_id:
            raise ValueError(f"input row {row_number} has invalid question_id")
        if (
            isinstance(solution_index, bool)
            or not isinstance(solution_index, int)
            or not 0 <= solution_index < 3
        ):
            raise ValueError(f"input row {row_number} has invalid solution_index")
        expected_id = f"{question_id}.solution-{solution_index}"
        if row["id"] != expected_id:
            raise ValueError(
                f"input row {row_number} derived ID mismatch: expected {expected_id}, "
                f"got {row['id']}"
            )
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"input row {row_number} has empty prompt")
        if not isinstance(completion, str) or not completion.strip():
            raise ValueError(f"input row {row_number} has empty completion")
        previous_prompt = question_prompts.setdefault(question_id, prompt)
        if previous_prompt != prompt:
            raise ValueError(f"question {question_id} has inconsistent prompts")
        question_rows[question_id].append((solution_index, str(row["id"])))
    if not rows:
        raise ValueError("pilot gradient input must be nonempty")
    for question_id, values in question_rows.items():
        indices = [index for index, _ in values]
        if len(values) not in (2, 3) or len(set(indices)) != len(indices):
            raise ValueError(
                f"question {question_id} must have two or three unique majority solutions"
            )
    return rows


def validate_pilot_manifest(path: Path, expected_sha256: str) -> dict:
    manifest_path = Path(path).resolve()
    expected = _sha256(expected_sha256, "pilot manifest SHA-256")
    if not manifest_path.is_file():
        raise ValueError(f"pilot manifest does not exist: {manifest_path}")
    actual = sha256_file(manifest_path)
    if actual != expected:
        raise ValueError(
            f"pilot manifest SHA-256 mismatch: expected {expected}, got {actual}"
        )
    value = strict_json_file(manifest_path, "pilot manifest")
    if not isinstance(value, dict):
        raise ValueError("pilot manifest must be a JSON object")
    return value


def _ordered_ids_sha256(rows: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["id"]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_pilot_gradient_manifest(
    *,
    input_jsonl: Path,
    input_sha256: str,
    rows: Sequence[Mapping[str, object]],
    pilot_manifest: Path,
    pilot_manifest_sha256: str,
    reference_repo: Path,
    reference_tree: str,
    prefix: str,
    num_shards: int,
    trl_version: str,
    package_versions: Mapping[str, str | None],
) -> dict:
    prefix = _validate_gradient_prefix(prefix)
    if isinstance(num_shards, bool) or not isinstance(num_shards, int) or num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not rows:
        raise ValueError("gradient manifest requires nonempty input rows")
    return {
        "manifest_version": 1,
        "pilot": "prismatic_lite_qwen3_2k",
        "input_jsonl": str(Path(input_jsonl).resolve()),
        "input_jsonl_sha256": _sha256(input_sha256, "input SHA-256"),
        "input_fields": list(PILOT_INPUT_FIELDS),
        "input_row_count": len(rows),
        "input_ordered_ids_sha256": _ordered_ids_sha256(rows),
        "pilot_manifest": str(Path(pilot_manifest).resolve()),
        "pilot_manifest_sha256": _sha256(
            pilot_manifest_sha256, "pilot manifest SHA-256"
        ),
        "reference_repo": str(Path(reference_repo).resolve()),
        "reference_commit": REFERENCE_COMMIT,
        "reference_tree": reference_tree,
        "official_gradient_module": str(
            _official_gradient_module_path(Path(reference_repo))
        ),
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "prefix": prefix,
        "num_shards": num_shards,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "trl_version": trl_version,
        "package_versions": dict(package_versions),
        "resume_allowed": False,
        "projection": {
            "backend": PROJECTOR_BACKEND,
            "dimension": PROJECTION_DIM,
            "seed": PROJECTION_SEED,
            "type": "rademacher",
            "input_dtype": PROJECTION_INPUT_DTYPE_NAME,
            "output_dtype": PROJECTED_DTYPE_NAME,
            "project_interval": PROJECT_INTERVAL,
            "save_interval": SAVE_INTERVAL,
            "completion_only_loss": True,
            "full_parameter_gradients": True,
            "response_marker": ASSISTANT_RESPONSE_MARKER,
        },
    }


def _all_shard_bounds(total: int, num_shards: int) -> list[tuple[int, int]]:
    bounds = [
        official_shard_bounds(total, num_shards, index) for index in range(num_shards)
    ]
    if any(start > total or end < start for start, end in bounds):
        raise ValueError("num_shards creates an invalid empty logical shard")
    return bounds


def assert_fresh_shard(
    output_dir: Path,
    prefix: str,
    *,
    total: int,
    num_shards: int,
    shard_index: int,
) -> None:
    prefix = _validate_gradient_prefix(prefix)
    bounds = _all_shard_bounds(total, num_shards)
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    shard_start, shard_end = bounds[shard_index]
    pattern = re.compile(_CHUNK_PATTERN_TEMPLATE.format(prefix=re.escape(prefix)))
    directory = Path(output_dir)
    if not directory.is_dir():
        raise ValueError(f"gradient output directory does not exist: {directory}")
    for path in sorted(directory.iterdir()):
        if not path.name.startswith(f"{prefix}."):
            continue
        match = pattern.fullmatch(path.name)
        if not path.is_file() or match is None:
            raise ValueError(f"malformed gradient artifact: {path.name}")
        start = int(match.group(1))
        if not 0 <= start < total:
            raise ValueError(f"gradient chunk start is outside input range: {path.name}")
        if shard_start <= start < shard_end:
            raise ValueError(
                f"resume is forbidden; archive prior shard artifact before relaunch: {path.name}"
            )


def _validate_output_isolation(
    output_dir: Path, input_jsonl: Path, pilot_manifest: Path, reference_repo: Path
) -> Path:
    output = Path(output_dir).resolve()
    for value, description in (
        (Path(input_jsonl).resolve(), "input JSONL"),
        (Path(pilot_manifest).resolve(), "pilot manifest"),
        (Path(reference_repo).resolve(), "reference repository"),
    ):
        if output == value or value in output.parents or output in value.parents:
            raise ValueError(f"gradient output overlaps {description}: {value}")
    return output


def _expected_manifest(
    *,
    reference_repo: Path,
    input_jsonl: Path,
    input_sha256: str,
    rows: Sequence[Mapping[str, object]],
    pilot_manifest: Path,
    pilot_manifest_sha256: str,
    prefix: str,
    num_shards: int,
) -> dict:
    return build_pilot_gradient_manifest(
        input_jsonl=input_jsonl,
        input_sha256=input_sha256,
        rows=rows,
        pilot_manifest=pilot_manifest,
        pilot_manifest_sha256=pilot_manifest_sha256,
        reference_repo=reference_repo,
        reference_tree=_reference_tree(reference_repo),
        prefix=prefix,
        num_shards=num_shards,
        trl_version=_installed_trl_version(),
        package_versions=_installed_package_versions(),
    )


def run_collection(
    *,
    reference_repo: Path,
    input_jsonl: Path,
    input_sha256: str,
    pilot_manifest: Path,
    pilot_manifest_sha256: str,
    output_dir: Path,
    prefix: str,
    model_name: str,
    model_revision: str,
    shard_index: int,
    num_shards: int,
    device: str,
    validate_after_collection: bool = True,
) -> dict[str, object]:
    if model_name != MODEL_NAME or model_revision != MODEL_REVISION:
        raise ValueError("pilot gradient collector model pin mismatch")
    if device != "cuda:0":
        raise ValueError("collector device must be process-local cuda:0")
    rows = load_pilot_gradient_input(input_jsonl, input_sha256)
    validate_pilot_manifest(pilot_manifest, pilot_manifest_sha256)
    repository = Path(reference_repo).resolve()
    verify_reference_repo(repository, REFERENCE_COMMIT)
    module_path = _official_gradient_module_path(repository)
    if not module_path.is_file():
        raise ValueError(f"official GradientComputer module does not exist: {module_path}")
    bounds = _all_shard_bounds(len(rows), num_shards)
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    shard_start, shard_end = bounds[shard_index]
    output = _validate_output_isolation(
        output_dir, input_jsonl, pilot_manifest, repository
    )
    output.mkdir(parents=True, exist_ok=True)
    expected = _expected_manifest(
        reference_repo=repository,
        input_jsonl=Path(input_jsonl),
        input_sha256=input_sha256,
        rows=rows,
        pilot_manifest=Path(pilot_manifest),
        pilot_manifest_sha256=pilot_manifest_sha256,
        prefix=prefix,
        num_shards=num_shards,
    )
    write_or_validate_gradient_manifest(output / GRADIENT_MANIFEST_NAME, expected)

    with _exclusive_shard_lock(output, prefix, num_shards, shard_index):
        assert_fresh_shard(
            output,
            prefix,
            total=len(rows),
            num_shards=num_shards,
            shard_index=shard_index,
        )
        if shard_start == shard_end:
            return {
                "status": "empty",
                "shard_start": shard_start,
                "shard_end": shard_end,
                "sample_count": 0,
            }
        validate_single_cuda_device(device)
        gradient_class = load_official_gradient_computer_class(
            repository, REFERENCE_COMMIT
        )
        model, tokenizer = load_model_and_tokenizer(
            model_name, model_revision, device
        )
        configured_context = getattr(model.config, "max_position_embeddings", None)
        if configured_context != MAX_CONTEXT_TOKENS:
            raise ValueError("pinned model hard context boundary mismatch")
        collector = construct_strict_collector(
            gradient_class, model_name, model, tokenizer
        )
        shard_rows = rows[shard_start:shard_end]
        preflight_samples(shard_rows, tokenizer, collector)
        collector.compute_project_store_gradients(
            shard_rows, prefix, output, shard_start
        )
        if validate_after_collection:
            completed = resolve_resume_start(
                [str(row["id"]) for row in rows],
                output,
                prefix,
                shard_start,
                shard_end,
            )
            if completed != shard_end:
                raise RuntimeError(
                    f"collector returned incomplete shard: expected {shard_end}, got {completed}"
                )
    return {
        "status": "collected",
        "shard_start": shard_start,
        "shard_end": shard_end,
        "sample_count": shard_end - shard_start,
    }


def run_global_validation(
    *,
    reference_repo: Path,
    input_jsonl: Path,
    input_sha256: str,
    pilot_manifest: Path,
    pilot_manifest_sha256: str,
    output_dir: Path,
    prefix: str,
    num_shards: int,
) -> dict[str, object]:
    rows = load_pilot_gradient_input(input_jsonl, input_sha256)
    validate_pilot_manifest(pilot_manifest, pilot_manifest_sha256)
    repository = Path(reference_repo).resolve()
    verify_reference_repo(repository, REFERENCE_COMMIT)
    output = _validate_output_isolation(
        output_dir, input_jsonl, pilot_manifest, repository
    )
    expected = _expected_manifest(
        reference_repo=repository,
        input_jsonl=Path(input_jsonl),
        input_sha256=input_sha256,
        rows=rows,
        pilot_manifest=Path(pilot_manifest),
        pilot_manifest_sha256=pilot_manifest_sha256,
        prefix=prefix,
        num_shards=num_shards,
    )
    manifest_path = output / GRADIENT_MANIFEST_NAME
    actual = strict_json_file(manifest_path, "gradient manifest")
    if actual != expected:
        raise ValueError("gradient manifest mismatch")
    coverage = validate_global_gradient_coverage(
        [str(row["id"]) for row in rows], output, prefix, num_shards
    )
    return {"status": "validated", **coverage}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect immutable Prismatic-lite projected gradients"
    )
    parser.add_argument("--validate-global-only", action="store_true")
    parser.add_argument("--reference-repo", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--pilot-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--device")
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.validate_global_only:
        missing = [
            flag
            for flag, value in (
                ("--shard-index", args.shard_index),
                ("--device", args.device),
            )
            if value is None
        ]
        if missing:
            parser.error("required for collection: " + ", ".join(missing))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    common = {
        "reference_repo": args.reference_repo,
        "input_jsonl": args.input_jsonl,
        "input_sha256": args.input_sha256,
        "pilot_manifest": args.pilot_manifest,
        "pilot_manifest_sha256": args.pilot_manifest_sha256,
        "output_dir": args.output_dir,
        "prefix": args.prefix,
        "num_shards": args.num_shards,
    }
    if args.validate_global_only:
        result = run_global_validation(**common)
    else:
        result = run_collection(
            **common,
            model_name=args.model_name,
            model_revision=args.model_revision,
            shard_index=args.shard_index,
            device=args.device,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
