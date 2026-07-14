"""Strict model-free analyzer for the OPD proxy-gradient verification probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path, PurePosixPath

import numpy as np

from math_eval.opd_proxy_gradient_classification import (
    classify_stage1,
    classify_stage2_sensitivity,
    evaluate_oracle_gate,
    evaluate_pn1_components,
    evaluate_strict_arm_components,
)
from math_eval.opd_proxy_gradient_statistics import (
    TargetSpace,
    evaluate_subset,
    evaluate_subset_table,
    inclusive_percentile,
    normalize_rows,
    partition_agreement,
    selected_set_overlap,
    target_dependence_permutation_test,
    unbiased_hsic,
)
from math_eval.opd_proxy_gradient_verify_artifacts import (
    VectorSet,
    canonical_json_bytes,
    load_vector_set,
    sha256_file,
    sha256_id_lines,
    write_or_validate_manifest,
)
from math_eval.prepare_opd_proxy_gradient_verify import StageLayout, stage_layout
from math_eval.select_gradient_diverse_deepmath import (
    REFERENCE_COMMIT,
    REFERENCE_TREE,
    load_official_reference_classes,
)
from math_eval.select_opd_proxy_gradient_verify import (
    DIAGNOSTIC_RATIO,
    EXPECTED_REPRESENTATIONS,
    KMEANS_SEEDS,
    PRIMARY_RATIO,
    RandomSchedules,
    SelectionBundle,
    build_selection_vector_view,
    load_random_schedules,
    load_selection_bundle,
    selection_vector_id,
)


ANALYSIS_INPUTS_FILE = "analysis_inputs.json"
REPORT_SCHEMA_VERSION = 1
SUMMARY_QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)
_SCORE_FIELDS = (
    "g_vendi",
    "coverage",
    "full_gradient_norm",
    "opd_signal_rms",
    "valid_token_count",
    "sampled_reverse_kl",
    "response_length",
)
_METADATA_FIELDS = (
    "prompt_token_count_4b",
    "prompt_token_count_0_6b",
    "r1_completion_token_count_0_6b",
    "sft_full_token_count_0_6b",
    "sft_supervised_label_count",
)
_PN1 = EXPECTED_REPRESENTATIONS[:8]
_PN4 = EXPECTED_REPRESENTATIONS[8:10]
_NON_TARGET = EXPECTED_REPRESENTATIONS[:12]
_TARGETS = EXPECTED_REPRESENTATIONS[12:]


@dataclass(frozen=True)
class _LoadedInputs:
    stage_root: Path
    stage_manifest: dict[str, object]
    stage_manifest_sha256: str
    rows: tuple[dict[str, object], ...]
    candidate_rows: tuple[dict[str, object], ...]
    heldout_rows: tuple[dict[str, object], ...]
    candidate_ids: tuple[str, ...]
    heldout_ids: tuple[str, ...]
    layout: StageLayout
    input_manifest: dict[str, object]
    input_manifest_sha256: str
    vectors: dict[str, VectorSet]
    target_heldout: dict[int, VectorSet]
    selection: SelectionBundle
    random: RandomSchedules
    reference: dict[str, object]


def _reject_pairs(pairs):
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str):
    raise ValueError(f"non-finite JSON constant: {value}")


def _load_canonical_json(path: Path, description: str) -> dict[str, object]:
    try:
        payload = Path(path).read_bytes()
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_pairs,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid {description}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    if payload != canonical_json_bytes(value):
        raise ValueError(f"{description} must use canonical JSON bytes")
    return value


def _load_canonical_jsonl(path: Path, description: str) -> tuple[dict, ...]:
    rows: list[dict] = []
    try:
        with Path(path).open("rb") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = json.loads(
                    line.decode("utf-8"),
                    object_pairs_hook=_reject_pairs,
                    parse_constant=_reject_constant,
                )
                if not isinstance(value, dict) or line != canonical_json_bytes(value):
                    raise ValueError(
                        f"{description} row {line_number} is not canonical"
                    )
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid {description}: {error}") from error
    if not rows:
        raise ValueError(f"{description} must not be empty")
    return tuple(rows)


def _relative_path(root: Path, value: object, description: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} must be a nonempty relative path")
    logical = PurePosixPath(value)
    if logical.is_absolute() or ".." in logical.parts or "." in logical.parts:
        raise ValueError(f"{description} must be a normalized relative path")
    resolved_root = root.resolve()
    resolved = (root / Path(*logical.parts)).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"{description} escapes the stage directory") from error
    return resolved


def _require_sha(value: object, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{description} must be a lowercase SHA-256 digest")
    return value


def _validate_stage(
    stage_root: Path,
) -> tuple[
    dict[str, object],
    str,
    tuple[dict, ...],
    tuple[str, ...],
    tuple[str, ...],
    StageLayout,
]:
    manifest_path = stage_root / "manifest.json"
    manifest = _load_canonical_json(manifest_path, "stage manifest")
    if manifest.get("schema_version") != 1 or manifest.get("artifact_type") != (
        "opd_proxy_stage"
    ):
        raise ValueError("invalid OPD proxy stage manifest contract")
    stage = manifest.get("stage")
    if isinstance(stage, bool) or not isinstance(stage, int):
        raise ValueError("stage manifest stage must be an integer")
    layout = stage_layout(stage)
    expected_fields = {
        "candidate_count": len(layout.candidate_clean_positions),
        "held_out_count": len(layout.held_out_clean_positions),
        "selected_size": layout.selected_size,
        "primary_k": layout.primary_k,
        "diagnostic_k": layout.diagnostic_k,
        "null_draws": layout.null_draws,
    }
    if any(manifest.get(field) != value for field, value in expected_fields.items()):
        raise ValueError("stage manifest cardinalities differ from derived stage layout")
    sample_path = _relative_path(
        stage_root, manifest.get("sample_manifest"), "stage sample manifest"
    )
    sample_sha = _require_sha(
        manifest.get("sample_manifest_sha256"), "stage sample manifest SHA"
    )
    if sha256_file(sample_path) != sample_sha:
        raise ValueError("stage sample manifest SHA mismatch")
    rows = _load_canonical_jsonl(sample_path, "stage sample manifest")
    candidate_count = expected_fields["candidate_count"]
    heldout_count = expected_fields["held_out_count"]
    if len(rows) != candidate_count + heldout_count:
        raise ValueError("stage sample rows differ from stage layout")
    candidate_ids: list[str] = []
    heldout_ids: list[str] = []
    for index, row in enumerate(rows):
        stable_id = row.get("stable_id")
        expected_split = "candidate" if index < candidate_count else "held_out"
        if (
            not isinstance(stable_id, str)
            or not stable_id
            or row.get("split") != expected_split
            or row.get("manifest_index") != index
        ):
            raise ValueError("stage row stable ID, split, or manifest index mismatch")
        leaf_topic = row.get("leaf_topic")
        if not isinstance(leaf_topic, str) or not leaf_topic:
            raise ValueError("stage row leaf_topic must be nonempty")
        for field in _METADATA_FIELDS:
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"stage row {field} must be a positive integer")
        (candidate_ids if expected_split == "candidate" else heldout_ids).append(
            stable_id
        )
    all_ids = candidate_ids + heldout_ids
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("stage stable IDs must be globally unique")
    hash_fields = {
        "candidate_ids_sha256": candidate_ids,
        "held_out_ids_sha256": heldout_ids,
        "all_ids_sha256": all_ids,
    }
    for field, values in hash_fields.items():
        if manifest.get(field) != sha256_id_lines(values):
            raise ValueError(f"stage {field} mismatch")
    if stage in (0, 1) and manifest.get("parent_report") is not None:
        raise ValueError("Stage 0/1 cannot declare a parent report")
    if stage == 2 and not isinstance(manifest.get("parent_report"), dict):
        raise ValueError("Stage 2 requires its frozen Stage-1 parent report")
    return (
        manifest,
        sha256_file(manifest_path),
        rows,
        tuple(candidate_ids),
        tuple(heldout_ids),
        layout,
    )


def _source_representation(name: str) -> str:
    if name.startswith("P_"):
        return "P"
    if name.startswith("T:"):
        return "T"
    return name


def _load_vector_record(
    stage_root: Path,
    record: object,
    *,
    expected_ids: Sequence[str],
    expected_vector_ids: Sequence[str],
    expected_representation: str,
    selection_name: str,
    description: str,
) -> tuple[VectorSet, str]:
    if not isinstance(record, dict):
        raise ValueError(f"{description} record must be an object")
    singular = record.get("vector_directory")
    plural = record.get("vector_directories")
    hash_value = record.get("vector_manifest_sha256")
    if singular is not None and plural is not None:
        raise ValueError(f"{description} cannot mix singular and sharded sources")
    if singular is not None:
        path_values = [singular]
        hash_values = [hash_value]
    else:
        if (
            not isinstance(plural, list)
            or not plural
            or not isinstance(hash_value, list)
            or len(hash_value) != len(plural)
        ):
            raise ValueError(f"{description} sharded source records are invalid")
        path_values = plural
        hash_values = hash_value
    sources: list[VectorSet] = []
    for source_index, (path_value, expected_hash_value) in enumerate(
        zip(path_values, hash_values)
    ):
        directory = _relative_path(
            stage_root,
            path_value,
            f"{description} source {source_index} directory",
        )
        expected_manifest_sha = _require_sha(
            expected_hash_value, f"{description} source {source_index} manifest SHA"
        )
        manifest_path = directory / "manifest.json"
        if sha256_file(manifest_path) != expected_manifest_sha:
            raise ValueError(f"{description} vector manifest SHA mismatch")
        sources.append(
            load_vector_set(
                directory, expected_representation=expected_representation
            )
        )
        if sha256_file(manifest_path) != expected_manifest_sha:
            raise ValueError(f"{description} vector manifest changed while loading")
    loaded = build_selection_vector_view(sources, selection_name, expected_ids)
    if loaded.vector_ids != tuple(expected_vector_ids):
        raise ValueError(f"{description} vector ID coverage mismatch")
    if loaded.stable_ids != tuple(expected_ids):
        raise ValueError(f"{description} stable ID order mismatch")
    view_hash = (
        str(hash_values[0])
        if len(sources) == 1 and loaded is sources[0]
        else hashlib.sha256(canonical_json_bytes(loaded.manifest)).hexdigest()
    )
    return loaded, view_hash


def write_analysis_input_manifest(
    *,
    stage_dir: Path,
    selection_directory: Path,
    representation_sources: Mapping[str, Sequence[Path]],
    target_heldout_sources: Mapping[int, Sequence[Path]],
) -> dict[str, object]:
    """Bind declared source shards into the analyzer's immutable input manifest."""
    stage_root = Path(stage_dir).resolve()
    if set(representation_sources) != set(EXPECTED_REPRESENTATIONS):
        raise ValueError("analysis input representations are incomplete or unexpected")
    if set(target_heldout_sources) != {42, 43}:
        raise ValueError("analysis target held-out sources must cover seeds 42/43")
    stage_manifest_path = stage_root / "manifest.json"
    _load_canonical_json(stage_manifest_path, "stage manifest")

    def record(paths: Sequence[Path], description: str) -> dict[str, object]:
        declared = tuple(Path(path).resolve() for path in paths)
        if not declared:
            raise ValueError(f"{description} requires at least one source directory")
        logical_paths: list[str] = []
        hashes: list[str] = []
        for directory in declared:
            try:
                logical = directory.relative_to(stage_root)
            except ValueError as error:
                raise ValueError(f"{description} source escapes stage directory") from error
            if not directory.is_dir():
                raise ValueError(f"{description} source directory does not exist")
            logical_paths.append(PurePosixPath(*logical.parts).as_posix())
            hashes.append(sha256_file(directory / "manifest.json"))
        if len(declared) == 1:
            return {
                "vector_directory": logical_paths[0],
                "vector_manifest_sha256": hashes[0],
            }
        return {
            "vector_directories": logical_paths,
            "vector_manifest_sha256": hashes,
        }

    selection_root = Path(selection_directory).resolve()
    try:
        selection_logical = selection_root.relative_to(stage_root)
    except ValueError as error:
        raise ValueError("selection directory escapes stage directory") from error
    manifest = {
        "schema_version": 1,
        "artifact_type": "opd_proxy_analysis_inputs",
        "stage_manifest_sha256": sha256_file(stage_manifest_path),
        "selection_directory": PurePosixPath(*selection_logical.parts).as_posix(),
        "selection_manifest_sha256": sha256_file(
            selection_root / "selection.manifest.json"
        ),
        "random_manifest_sha256": sha256_file(
            selection_root / "random.manifest.json"
        ),
        "representations": {
            name: record(representation_sources[name], name)
            for name in EXPECTED_REPRESENTATIONS
        },
        "target_heldout": {
            str(seed): record(
                target_heldout_sources[seed], f"target held-out seed {seed}"
            )
            for seed in (42, 43)
        },
    }
    write_or_validate_manifest(stage_root / ANALYSIS_INPUTS_FILE, manifest)
    return manifest


def _load_inputs(stage_dir: Path, reference_repo: Path) -> _LoadedInputs:
    stage_root = Path(stage_dir).resolve()
    if not stage_root.is_dir():
        raise ValueError(f"stage directory does not exist: {stage_root}")
    (
        stage_manifest,
        stage_manifest_sha,
        rows,
        candidate_ids,
        heldout_ids,
        layout,
    ) = _validate_stage(stage_root)
    inputs_path = stage_root / ANALYSIS_INPUTS_FILE
    inputs = _load_canonical_json(inputs_path, "analysis input manifest")
    if inputs.get("schema_version") != 1 or inputs.get("artifact_type") != (
        "opd_proxy_analysis_inputs"
    ):
        raise ValueError("invalid OPD proxy analysis input manifest contract")
    if inputs.get("stage_manifest_sha256") != stage_manifest_sha:
        raise ValueError("analysis input stage manifest SHA mismatch")

    representation_records = inputs.get("representations")
    if not isinstance(representation_records, dict) or set(
        representation_records
    ) != set(EXPECTED_REPRESENTATIONS):
        raise ValueError("analysis input representations are incomplete or unexpected")
    vectors: dict[str, VectorSet] = {}
    vector_hashes: dict[str, str] = {}
    for name in EXPECTED_REPRESENTATIONS:
        expected_vector_ids = tuple(
            selection_vector_id(name, stable_id) for stable_id in candidate_ids
        )
        vector, manifest_sha = _load_vector_record(
            stage_root,
            representation_records[name],
            expected_ids=candidate_ids,
            expected_vector_ids=expected_vector_ids,
            expected_representation=_source_representation(name),
            selection_name=name,
            description=name,
        )
        vectors[name] = vector
        vector_hashes[name] = manifest_sha

    heldout_records = inputs.get("target_heldout")
    if not isinstance(heldout_records, dict) or set(heldout_records) != {"42", "43"}:
        raise ValueError("analysis input target held-out records must cover seeds 42/43")
    target_heldout: dict[int, VectorSet] = {}
    for seed in (42, 43):
        name = f"T:seed={seed}"
        expected_vector_ids = tuple(
            selection_vector_id(name, stable_id) for stable_id in heldout_ids
        )
        target_heldout[seed], _ = _load_vector_record(
            stage_root,
            heldout_records[str(seed)],
            expected_ids=heldout_ids,
            expected_vector_ids=expected_vector_ids,
            expected_representation="T",
            selection_name=name,
            description=f"target held-out seed {seed}",
        )

    selection_directory = _relative_path(
        stage_root, inputs.get("selection_directory"), "selection directory"
    )
    selection_manifest_path = selection_directory / "selection.manifest.json"
    random_manifest_path = selection_directory / "random.manifest.json"
    if sha256_file(selection_manifest_path) != _require_sha(
        inputs.get("selection_manifest_sha256"), "selection manifest SHA"
    ):
        raise ValueError("analysis input selection manifest SHA mismatch")
    if sha256_file(random_manifest_path) != _require_sha(
        inputs.get("random_manifest_sha256"), "random manifest SHA"
    ):
        raise ValueError("analysis input random manifest SHA mismatch")
    parent_hashes = {"stage_manifest_sha256": stage_manifest_sha}
    selection = load_selection_bundle(
        selection_directory,
        expected_candidate_ids=candidate_ids,
        expected_vector_manifest_hashes=vector_hashes,
        expected_parent_hashes=parent_hashes,
    )
    random = load_random_schedules(
        selection_directory,
        expected_candidate_ids=candidate_ids,
        expected_candidate_rows=rows[: len(candidate_ids)],
        expected_parent_hashes=parent_hashes,
    )
    random_stage = random.manifest.get("stage")
    if random_stage != layout.stage:
        raise ValueError("random schedule stage differs from stage manifest")

    _, _, reference = load_official_reference_classes(Path(reference_repo))
    reference_record = selection.manifest.get("official_reference")
    if not isinstance(reference_record, dict):
        raise ValueError("selection lacks official reference provenance")
    if (
        reference_record.get("reference_commit") != reference["reference_commit"]
        or reference_record.get("reference_tree") != reference["reference_tree"]
        or reference["reference_commit"] != REFERENCE_COMMIT
        or reference["reference_tree"] != REFERENCE_TREE
    ):
        raise ValueError("analysis official reference provenance mismatch")
    synthetic = bool(
        isinstance(stage_manifest.get("provenance"), dict)
        and stage_manifest["provenance"].get("synthetic_test_fixture") is True
    )
    mode = reference_record.get("mode")
    if mode != "pinned_official_cuda" and not (
        synthetic and mode == "injected_test_cluster_manager"
    ):
        raise ValueError("scientific analysis rejects a non-official selector")

    return _LoadedInputs(
        stage_root=stage_root,
        stage_manifest=stage_manifest,
        stage_manifest_sha256=stage_manifest_sha,
        rows=rows,
        candidate_rows=rows[: len(candidate_ids)],
        heldout_rows=rows[len(candidate_ids) :],
        candidate_ids=candidate_ids,
        heldout_ids=heldout_ids,
        layout=layout,
        input_manifest=inputs,
        input_manifest_sha256=sha256_file(inputs_path),
        vectors=vectors,
        target_heldout=target_heldout,
        selection=selection,
        random=random,
        reference=reference,
    )


def _required_scalar(vector_set: VectorSet, name: str) -> np.ndarray:
    value = getattr(vector_set, name)
    if value is None:
        raise ValueError(f"target vector set lacks required scalar {name}")
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (len(vector_set.stable_ids),) or not np.isfinite(array).all():
        raise ValueError(f"target vector scalar {name} is invalid")
    return array


def _target_spaces(inputs: _LoadedInputs) -> dict[int, TargetSpace]:
    result: dict[int, TargetSpace] = {}
    for seed in (42, 43):
        candidate = inputs.vectors[f"T:seed={seed}"]
        heldout = inputs.target_heldout[seed]
        result[seed] = TargetSpace(
            seed=seed,
            candidate_ids=inputs.candidate_ids,
            heldout_ids=inputs.heldout_ids,
            candidate_vectors=candidate.vectors,
            heldout_vectors=heldout.vectors,
            full_gradient_norm=_required_scalar(candidate, "full_gradient_norm"),
            opd_signal_rms=_required_scalar(candidate, "opd_signal_rms"),
            valid_token_count=_required_scalar(candidate, "valid_token_count"),
            sampled_reverse_kl=_required_scalar(candidate, "sampled_reverse_kl"),
            response_length=_required_scalar(candidate, "response_length"),
        )
    return result


def _summary(values: object) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("numeric summary requires a nonempty finite vector")
    quantiles = np.quantile(array, SUMMARY_QUANTILES, method="linear")
    return {
        "count": int(array.size),
        "mean": float(array.mean(dtype=np.float64)),
        "std": float(array.std(ddof=0, dtype=np.float64)),
        "quantiles": {
            format(level, "g"): float(value)
            for level, value in zip(SUMMARY_QUANTILES, quantiles)
        },
    }


def _score_dict(scores) -> dict[str, float]:
    return {name: float(getattr(scores, name)) for name in _SCORE_FIELDS}


def _score_table_summary(table: Mapping[str, np.ndarray]) -> dict[str, object]:
    return {name: _summary(table[name]) for name in _SCORE_FIELDS}


def _topic_record(
    candidate_rows: Sequence[Mapping[str, object]], positions: Sequence[int]
) -> dict[str, object]:
    counts: dict[str, int] = {}
    for position in positions:
        topic = str(candidate_rows[int(position)]["leaf_topic"])
        counts[topic] = counts.get(topic, 0) + 1
    probabilities = [count / len(positions) for count in counts.values()]
    entropy = -sum(probability * math.log(probability) for probability in probabilities)
    return {
        "coverage": len(counts),
        "entropy": float(entropy),
        "distribution": dict(sorted(counts.items())),
    }


def _metadata_record(
    candidate_rows: Sequence[Mapping[str, object]], positions: Sequence[int]
) -> dict[str, object]:
    return {
        field: _summary(
            [float(candidate_rows[int(position)][field]) for position in positions]
        )
        for field in _METADATA_FIELDS
    }


def _optional_selected_diagnostics(
    vector_set: VectorSet, positions: np.ndarray
) -> dict[str, object]:
    result: dict[str, object] = {}
    for field in (
        "full_gradient_norm",
        "valid_token_count",
        "response_length",
        "sampled_reverse_kl",
        "opd_signal_rms",
    ):
        value = getattr(vector_set, field)
        result[field] = (
            {"status": "not_computed"}
            if value is None
            else {"status": "computed", "summary": _summary(value[positions])}
        )
    correct = vector_set.verifier_correct_count
    total = vector_set.verifier_total
    if correct is None and total is None:
        result["verifier"] = {"status": "not_computed"}
    elif correct is None or total is None:
        raise ValueError("vector verifier diagnostics are only partially present")
    else:
        selected_correct = float(np.asarray(correct)[positions].sum(dtype=np.float64))
        selected_total = float(np.asarray(total)[positions].sum(dtype=np.float64))
        if selected_total <= 0:
            raise ValueError("selected verifier total must be positive")
        result["verifier"] = {
            "status": "computed",
            "correct_count": selected_correct,
            "total": selected_total,
            "rate": selected_correct / selected_total,
        }
    return result


def _verifier_subset(
    vector_set: VectorSet, positions: np.ndarray
) -> dict[str, object]:
    correct = vector_set.verifier_correct_count
    total = vector_set.verifier_total
    if correct is None and total is None:
        return {"status": "not_computed"}
    if correct is None or total is None:
        raise ValueError("target verifier diagnostics are only partially present")
    correct_sum = float(np.asarray(correct)[positions].sum(dtype=np.float64))
    total_sum = float(np.asarray(total)[positions].sum(dtype=np.float64))
    if total_sum <= 0 or correct_sum < 0 or correct_sum > total_sum:
        raise ValueError("target verifier selected counts are invalid")
    return {
        "status": "computed",
        "correct_count": correct_sum,
        "total": total_sum,
        "rate": correct_sum / total_sum,
    }


def _random_verifier(
    vector_set: VectorSet, schedules: np.ndarray
) -> dict[str, object]:
    correct = vector_set.verifier_correct_count
    total = vector_set.verifier_total
    if correct is None and total is None:
        return {"status": "not_computed"}
    if correct is None or total is None:
        raise ValueError("target verifier diagnostics are only partially present")
    correct_sums = np.asarray(correct, dtype=np.float64)[schedules].sum(
        axis=1, dtype=np.float64
    )
    total_sums = np.asarray(total, dtype=np.float64)[schedules].sum(
        axis=1, dtype=np.float64
    )
    if np.any(total_sums <= 0) or np.any(correct_sums < 0) or np.any(
        correct_sums > total_sums
    ):
        raise ValueError("target verifier random-subset counts are invalid")
    return {
        "status": "computed",
        "correct_count": _summary(correct_sums),
        "total": _summary(total_sums),
        "rate": _summary(correct_sums / total_sums),
    }


def _random_metadata_diagnostics(
    candidate_rows: Sequence[Mapping[str, object]], schedules: np.ndarray
) -> dict[str, object]:
    topic_coverage = np.empty(schedules.shape[0], dtype=np.float64)
    topic_entropy = np.empty(schedules.shape[0], dtype=np.float64)
    for index, positions in enumerate(schedules):
        topic = _topic_record(candidate_rows, positions)
        topic_coverage[index] = float(topic["coverage"])
        topic_entropy[index] = float(topic["entropy"])
    metadata: dict[str, object] = {
        "topic_coverage": _summary(topic_coverage),
        "topic_entropy": _summary(topic_entropy),
    }
    for field in _METADATA_FIELDS:
        values = np.asarray([float(row[field]) for row in candidate_rows])
        metadata[field] = _summary(values[schedules].mean(axis=1, dtype=np.float64))
    return metadata


def _selected_and_null_metrics(
    inputs: _LoadedInputs, spaces: Mapping[int, TargetSpace]
) -> tuple[
    list[dict[str, object]],
    dict[tuple[str, int, int], dict[str, object]],
    dict[int, dict[str, dict[str, np.ndarray]]],
    dict[str, object],
    list[dict[str, object]],
]:
    null_tables: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    random_report: dict[str, object] = {
        "schedule": {
            field: inputs.random.manifest[field]
            for field in (
                "candidate_count",
                "selected_size",
                "draws",
                "seed_sequence",
                "spawn_count",
                "bit_generator",
                "numpy_version",
                "paired_target_seeds",
                "uniform_logical_sha256",
                "stratified_logical_sha256",
                "files",
            )
        },
        "target_seeds": {},
        "metadata": {},
    }
    for kind, schedules in (
        ("uniform", inputs.random.uniform),
        ("stratified", inputs.random.stratified),
    ):
        random_report["metadata"][kind] = _random_metadata_diagnostics(  # type: ignore[index]
            inputs.candidate_rows, schedules
        )
    for seed in (42, 43):
        null_tables[seed] = {
            "uniform": evaluate_subset_table(spaces[seed], inputs.random.uniform),
            "stratified": evaluate_subset_table(
                spaces[seed], inputs.random.stratified
            ),
        }
        random_report["target_seeds"][str(seed)] = {  # type: ignore[index]
            kind: {
                **_score_table_summary(table),
                "verifier": _random_verifier(
                    inputs.vectors[f"T:seed={seed}"],
                    inputs.random.uniform
                    if kind == "uniform"
                    else inputs.random.stratified,
                ),
            }
            for kind, table in null_tables[seed].items()
        }

    records: list[dict[str, object]] = []
    index: dict[tuple[str, int, int], dict[str, object]] = {}
    selected_diagnostics: list[dict[str, object]] = []
    runs = inputs.selection.manifest["runs"]
    for run in runs:
        key = str(run["key"])
        representation = str(run["representation"])
        kmeans_seed = int(run["kmeans_seed"])
        positions_sampler_order = inputs.selection.selected_positions[key]
        positions = np.sort(positions_sampler_order.astype(np.int64))
        selected_ids = [
            inputs.candidate_ids[int(position)]
            for position in positions_sampler_order.tolist()
        ]
        selected_diagnostics.append(
            {
                "selection_key": key,
                "representation": representation,
                "ratio_name": run["ratio_name"],
                "kmeans_seed": kmeans_seed,
                "selected_ids_sampler_order": selected_ids,
                "topic": _topic_record(inputs.candidate_rows, positions),
                "metadata": _metadata_record(inputs.candidate_rows, positions),
                "source_vector": _optional_selected_diagnostics(
                    inputs.vectors[representation], positions
                ),
            }
        )
        for target_seed in (42, 43):
            raw = _score_dict(evaluate_subset(spaces[target_seed], positions))
            uniform_percentiles = {
                name: inclusive_percentile(
                    null_tables[target_seed]["uniform"][name], value
                )
                for name, value in raw.items()
            }
            stratified_percentiles = {
                name: inclusive_percentile(
                    null_tables[target_seed]["stratified"][name], value
                )
                for name, value in raw.items()
            }
            record = {
                "selection_key": key,
                "representation": representation,
                "ratio_name": run["ratio_name"],
                "ratio": float(run["ratio"]),
                "k": int(run["k"]),
                "kmeans_seed": kmeans_seed,
                "target_seed": target_seed,
                "selected_logical_sha256": run["selected_logical_sha256"],
                "raw_scores": raw,
                "uniform_percentiles": uniform_percentiles,
                "stratified_percentiles": stratified_percentiles,
                "target_verifier": _verifier_subset(
                    inputs.vectors[f"T:seed={target_seed}"], positions
                ),
            }
            records.append(record)
            if run["ratio_name"] == "primary":
                primary_key = (representation, kmeans_seed, target_seed)
                if primary_key in index:
                    raise RuntimeError("duplicate primary selected metric record")
                index[primary_key] = record
    expected_records = len(EXPECTED_REPRESENTATIONS) * 2 * 2 * 2
    if len(records) != expected_records:
        raise RuntimeError("selected metric record coverage is incomplete")
    if len(index) != len(EXPECTED_REPRESENTATIONS) * 2 * 2:
        raise RuntimeError("primary selected metric index coverage is incomplete")
    return records, index, null_tables, random_report, selected_diagnostics


def _corner_record(record: Mapping[str, object]) -> dict[str, object]:
    uniform = record["uniform_percentiles"]
    return {
        "target_seed": int(record["target_seed"]),
        "kmeans_seed": int(record["kmeans_seed"]),
        "g_vendi": float(uniform["g_vendi"]),
        "coverage": float(uniform["coverage"]),
        "gradient_norm": float(uniform["full_gradient_norm"]),
        "opd_signal": float(uniform["opd_signal_rms"]),
    }


def _gate_diagnostics(
    metric_index: Mapping[tuple[str, int, int], Mapping[str, object]],
    target_cka: Mapping[str, object],
) -> dict[str, object]:
    def realization(name: str) -> dict[str, object]:
        return {
            "primary_corners": [
                _corner_record(metric_index[(name, kmeans_seed, target_seed)])
                for target_seed in (42, 43)
                for kmeans_seed in KMEANS_SEEDS
            ]
        }

    oracle_runs: list[dict[str, object]] = []
    for selection_seed, evaluation_seed in ((42, 43), (43, 42)):
        name = f"T:seed={selection_seed}"
        for kmeans_seed in KMEANS_SEEDS:
            record = metric_index[(name, kmeans_seed, evaluation_seed)]
            uniform = record["uniform_percentiles"]
            oracle_runs.append(
                {
                    "selection_target_seed": selection_seed,
                    "evaluation_target_seed": evaluation_seed,
                    "kmeans_seed": kmeans_seed,
                    "g_vendi_percentile": float(uniform["g_vendi"]),
                    "coverage_percentile": float(uniform["coverage"]),
                }
            )
    oracle = evaluate_oracle_gate(
        oracle_runs, float(target_cka["p_value"])
    )
    pn1 = evaluate_pn1_components({name: realization(name) for name in _PN1})
    pn4 = evaluate_strict_arm_components(
        {name: realization(name) for name in _PN4}, expected_realizations=2
    )
    sft = evaluate_strict_arm_components(
        {"S": realization("S")}, expected_realizations=1
    )
    embedding = evaluate_strict_arm_components(
        {"E": realization("E")}, expected_realizations=1
    )
    return classify_stage1(
        oracle=oracle,
        pn1_components=pn1,
        pn4_components=pn4,
        sft_components=sft,
        embedding_components=embedding,
    )


def _cka_pairs() -> list[tuple[str, str, str]]:
    pairs: list[tuple[str, str, str]] = [
        ("target_cross_seed", _TARGETS[0], _TARGETS[1])
    ]
    pairs.extend(
        ("target_vs_candidate", target, candidate)
        for target in _TARGETS
        for candidate in _NON_TARGET
    )
    pairs.extend(
        ("pn1_pair", left, right) for left, right in combinations(_PN1, 2)
    )
    if len(pairs) != 53:
        raise RuntimeError("predeclared CKA pair count is not 53")
    return pairs


def _overlap_pairs() -> list[tuple[str, str, str]]:
    pairs: list[tuple[str, str, str]] = []
    pairs.extend(("pn1_pair", left, right) for left, right in combinations(_PN1, 2))
    pairs.extend(("pn4_pair", left, right) for left, right in combinations(_PN4, 2))
    pairs.extend(
        ("proxy_vs_baseline", proxy, baseline)
        for proxy in (*_PN1, *_PN4)
        for baseline in ("S", "E")
    )
    pairs.extend(
        ("candidate_vs_target_oracle", candidate, target)
        for candidate in _NON_TARGET
        for target in _TARGETS
    )
    if len(pairs) != 73:
        raise RuntimeError("predeclared selected-overlap pair count is not 73")
    return pairs


def _secondary_diagnostics(
    inputs: _LoadedInputs, target_cka_result: Mapping[str, object]
) -> dict[str, object]:
    pairs = _cka_pairs()
    kernels: dict[str, np.ndarray] = {}
    self_hsic: dict[str, float] = {}
    for representation in EXPECTED_REPRESENTATIONS:
        normalized = normalize_rows(inputs.vectors[representation].vectors)
        kernel = normalized @ normalized.T
        kernel = 0.5 * (kernel + kernel.T)
        kernels[representation] = kernel
        self_hsic[representation] = unbiased_hsic(kernel, kernel)
        if self_hsic[representation] <= 0:
            raise ValueError(
                f"representation {representation} has non-positive self-HSIC"
            )
    cka_records: list[dict[str, object]] = []
    for category, left, right in pairs:
        cross = unbiased_hsic(kernels[left], kernels[right])
        cka = cross / math.sqrt(self_hsic[left] * self_hsic[right])
        if not math.isfinite(cka):
            raise ValueError("secondary debiased CKA must be finite")
        cka_records.append(
            {"category": category, "left": left, "right": right, "cka": cka}
        )
    ami_records: list[dict[str, object]] = []
    ari_records: list[dict[str, object]] = []
    for category, left, right in pairs:
        for left_seed, right_seed in product(KMEANS_SEEDS, repeat=2):
            left_key = inputs.selection.key(
                left,
                k=inputs.layout.primary_k,
                kmeans_seed=left_seed,
                ratio=PRIMARY_RATIO,
            )
            right_key = inputs.selection.key(
                right,
                k=inputs.layout.primary_k,
                kmeans_seed=right_seed,
                ratio=PRIMARY_RATIO,
            )
            agreement = partition_agreement(
                inputs.selection.labels[left_key], inputs.selection.labels[right_key]
            )
            common = {
                "category": category,
                "left": left,
                "right": right,
                "left_kmeans_seed": left_seed,
                "right_kmeans_seed": right_seed,
            }
            ami_records.append(
                {**common, "adjusted_mutual_info": agreement["adjusted_mutual_info"]}
            )
            ari_records.append(
                {**common, "adjusted_rand_index": agreement["adjusted_rand_index"]}
            )

    overlap_records: list[dict[str, object]] = []
    for category, left, right in _overlap_pairs():
        for left_seed, right_seed in product(KMEANS_SEEDS, repeat=2):
            left_key = inputs.selection.key(
                left,
                k=inputs.layout.primary_k,
                kmeans_seed=left_seed,
                ratio=PRIMARY_RATIO,
            )
            right_key = inputs.selection.key(
                right,
                k=inputs.layout.primary_k,
                kmeans_seed=right_seed,
                ratio=PRIMARY_RATIO,
            )
            overlap = selected_set_overlap(
                np.sort(inputs.selection.selected_positions[left_key]).tolist(),
                np.sort(inputs.selection.selected_positions[right_key]).tolist(),
                candidate_count=len(inputs.candidate_ids),
                selected_size=inputs.layout.selected_size,
            )
            overlap_records.append(
                {
                    "category": category,
                    "left": left,
                    "right": right,
                    "left_kmeans_seed": left_seed,
                    "right_kmeans_seed": right_seed,
                    **overlap,
                }
            )

    sensitivity: list[dict[str, object]] = []
    for representation in EXPECTED_REPRESENTATIONS:
        for primary_seed, diagnostic_seed in product(KMEANS_SEEDS, repeat=2):
            primary_key = inputs.selection.key(
                representation,
                k=inputs.layout.primary_k,
                kmeans_seed=primary_seed,
                ratio=PRIMARY_RATIO,
            )
            diagnostic_key = inputs.selection.key(
                representation,
                k=inputs.layout.diagnostic_k,
                kmeans_seed=diagnostic_seed,
                ratio=DIAGNOSTIC_RATIO,
            )
            overlap = selected_set_overlap(
                np.sort(inputs.selection.selected_positions[primary_key]).tolist(),
                np.sort(inputs.selection.selected_positions[diagnostic_key]).tolist(),
                candidate_count=len(inputs.candidate_ids),
                selected_size=inputs.layout.selected_size,
            )
            agreement = partition_agreement(
                inputs.selection.labels[primary_key],
                inputs.selection.labels[diagnostic_key],
            )
            sensitivity.append(
                {
                    "representation": representation,
                    "primary_kmeans_seed": primary_seed,
                    "diagnostic_kmeans_seed": diagnostic_seed,
                    **overlap,
                    **agreement,
                }
            )
    if len(ami_records) != 212 or len(ari_records) != 212:
        raise RuntimeError("predeclared AMI/ARI record coverage mismatch")
    if len(overlap_records) != 292:
        raise RuntimeError("predeclared selected-overlap record coverage mismatch")
    return {
        "target_dependence_permutation": {
            "observed_cka": float(target_cka_result["observed_cka"]),
            "p_value": float(target_cka_result["p_value"]),
            "exceedance_count": int(target_cka_result["exceedance_count"]),
            "draws": int(target_cka_result["draws"]),
            "seed": int(target_cka_result["seed"]),
            "permutation_schedule_sha256": target_cka_result[
                "permutation_schedule_sha256"
            ],
            "tie_rule": target_cka_result["tie_rule"],
            "null_summary": _summary(target_cka_result["null_cka"]),
        },
        "cka_pairs": cka_records,
        "ami_records": ami_records,
        "ari_records": ari_records,
        "primary_selected_overlap": overlap_records,
        "diagnostic_k_sensitivity": sensitivity,
    }


def _load_parent_stage1_report(inputs: _LoadedInputs) -> tuple[dict, str]:
    parent = inputs.stage_manifest.get("parent_report")
    if not isinstance(parent, dict):
        raise ValueError("Stage 2 lacks its parent report record")
    path_value = parent.get("path")
    expected_sha = _require_sha(parent.get("sha256"), "Stage-1 parent report SHA")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("Stage-1 parent report path is invalid")
    path = Path(path_value).resolve()
    if sha256_file(path) != expected_sha:
        raise ValueError("Stage-1 parent report SHA mismatch")
    report = _load_canonical_json(path, "Stage-1 parent report")
    if report.get("stage") != 1 or not isinstance(report.get("classification"), dict):
        raise ValueError("Stage-1 parent report has the wrong stage/classification")
    return report, expected_sha


def run_analysis(
    *,
    stage_dir: Path,
    reference_repo: Path,
    output_json: Path | None = None,
    output_markdown: Path | None = None,
) -> dict[str, object]:
    """Load every frozen artifact, compute metrics, and optionally publish reports."""
    if (output_json is None) != (output_markdown is None):
        raise ValueError("report JSON and Markdown outputs must be provided together")
    inputs = _load_inputs(stage_dir, reference_repo)
    spaces = _target_spaces(inputs)
    (
        selected_metrics,
        metric_index,
        _,
        random_report,
        selected_diagnostics,
    ) = _selected_and_null_metrics(inputs, spaces)
    target_cka = target_dependence_permutation_test(
        inputs.vectors["T:seed=42"].vectors,
        inputs.vectors["T:seed=43"].vectors,
    )
    gate_diagnostics = _gate_diagnostics(metric_index, target_cka)
    secondary = _secondary_diagnostics(inputs, target_cka)
    if inputs.layout.stage == 0:
        classification: dict[str, object] = {"status": "smoke_only"}
    elif inputs.layout.stage == 1:
        classification = gate_diagnostics
    else:
        parent_report, parent_sha = _load_parent_stage1_report(inputs)
        sensitivity = classify_stage2_sensitivity(
            stage1_report=parent_report["classification"],
            expanded_oracle=gate_diagnostics["oracle"],
            expanded_pn1_components=gate_diagnostics["P_n1"]["components"],
        )
        classification = {
            "status": sensitivity["classification"],
            "stage1_report_sha256": parent_sha,
            "sensitivity": sensitivity,
        }

    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_type": "opd_proxy_gradient_verification_report",
        "stage": inputs.layout.stage,
        "counts": {
            "candidate": len(inputs.candidate_ids),
            "held_out": len(inputs.heldout_ids),
            "selected": inputs.layout.selected_size,
            "primary_k": inputs.layout.primary_k,
            "diagnostic_k": inputs.layout.diagnostic_k,
            "random_draws": inputs.layout.null_draws,
            "representations": len(EXPECTED_REPRESENTATIONS),
        },
        "provenance": {
            "stage_manifest_sha256": inputs.stage_manifest_sha256,
            "sample_manifest_sha256": inputs.stage_manifest[
                "sample_manifest_sha256"
            ],
            "analysis_inputs_sha256": inputs.input_manifest_sha256,
            "selection_manifest_sha256": inputs.input_manifest[
                "selection_manifest_sha256"
            ],
            "random_manifest_sha256": inputs.input_manifest[
                "random_manifest_sha256"
            ],
            "representation_vector_manifest_sha256": inputs.selection.manifest[
                "representation_vector_manifest_sha256"
            ],
            "target_heldout_vector_manifest_sha256": {
                seed: inputs.input_manifest["target_heldout"][seed][
                    "vector_manifest_sha256"
                ]
                for seed in ("42", "43")
            },
            "reference_commit": inputs.reference["reference_commit"],
            "reference_tree": inputs.reference["reference_tree"],
        },
        "random_nulls": random_report,
        "selected_metrics": selected_metrics,
        "selected_diagnostics": selected_diagnostics,
        "pre_registered_gate_diagnostics": gate_diagnostics,
        "secondary_diagnostics": secondary,
        "classification": classification,
    }
    canonical_json_bytes(report)
    if output_json is not None and output_markdown is not None:
        write_reports(report, output_json, output_markdown)
    return report


def render_markdown(report: Mapping[str, object]) -> str:
    """Render Markdown solely from a validated report dictionary."""
    if report.get("artifact_type") != "opd_proxy_gradient_verification_report":
        raise ValueError("cannot render a non-OPD verification report")
    stage = int(report["stage"])
    counts = report["counts"]
    classification = report["classification"]
    lines = [
        f"# Vanilla-OPD Proxy-Gradient Verification — Stage {stage}",
        "",
    ]
    if classification == {"status": "smoke_only"}:
        lines.extend(
            [
                "**Classification:** smoke only (no scientific pass/fail).",
                "",
            ]
        )
    elif "status" in classification:
        lines.extend(
            [f"**Stage-2 sensitivity:** `{classification['status']}`.", ""]
        )
    else:
        lines.extend(
            [
                f"**Classification status:** `{classification['classification']}`.",
                f"**Interpretation:** `{classification['interpretation']}`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Frozen cardinalities",
            "",
            f"- Candidates: {counts['candidate']}",
            f"- Held out: {counts['held_out']}",
            f"- Selected per arm: {counts['selected']}",
            f"- Primary / diagnostic K: {counts['primary_k']} / {counts['diagnostic_k']}",
            f"- Random draws per null: {counts['random_draws']}",
            "",
            "## Pre-registered gate diagnostics",
            "",
        ]
    )
    gates = report["pre_registered_gate_diagnostics"]
    lines.append(f"- Oracle: `{gates['oracle']['classification']}`")
    for arm in ("P_n1", "P_n4", "S", "E"):
        lines.append(f"- {arm}: `{gates[arm]['classification']}`")
    target = report["secondary_diagnostics"]["target_dependence_permutation"]
    lines.extend(
        [
            f"- Target cross-seed CKA: {target['observed_cka']:.8g}",
            f"- Target CKA permutation p-value: {target['p_value']:.8g}",
            "",
            "## Secondary diagnostics",
            "",
            f"- CKA representation pairs: {len(report['secondary_diagnostics']['cka_pairs'])}",
            f"- AMI / ARI records: {len(report['secondary_diagnostics']['ami_records'])} / {len(report['secondary_diagnostics']['ari_records'])}",
            f"- Primary selected-overlap records: {len(report['secondary_diagnostics']['primary_selected_overlap'])}",
            "",
            "The probe evaluates frozen gradient-space selection only; it does not establish downstream OOD improvement.",
            "",
        ]
    )
    return "\n".join(lines)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_reports(
    report: Mapping[str, object], json_path: Path, markdown_path: Path
) -> None:
    """Validate both report forms, then publish the pair with rollback on error."""
    json_target = Path(json_path)
    markdown_target = Path(markdown_path)
    if json_target.resolve() == markdown_target.resolve():
        raise ValueError("JSON and Markdown report paths must differ")
    json_payload = canonical_json_bytes(dict(report))
    markdown_payload = render_markdown(report).encode("utf-8")
    # Round-trip before touching either prior report.
    loaded = json.loads(json_payload)
    if render_markdown(loaded).encode("utf-8") != markdown_payload:
        raise ValueError("JSON and Markdown report views disagree")
    targets = ((json_target, json_payload), (markdown_target, markdown_payload))
    temporary: dict[Path, Path] = {}
    prior: dict[Path, bytes | None] = {}
    published: set[Path] = set()
    try:
        for target, payload in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            prior[target] = target.read_bytes() if target.exists() else None
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
            )
            temporary_path = Path(temporary_name)
            temporary[target] = temporary_path
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        for target, _ in targets:
            os.replace(temporary[target], target)
            published.add(target)
            _fsync_directory(target.parent)
    except BaseException:
        for target, old in prior.items():
            if old is None:
                if target in published:
                    target.unlink(missing_ok=True)
                continue
            current = target.read_bytes() if target.exists() else None
            if current != old:
                descriptor, restore_name = tempfile.mkstemp(
                    prefix=f".{target.name}.", suffix=".restore", dir=target.parent
                )
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(old)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(restore_name, target)
                _fsync_directory(target.parent)
        raise
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--reference-repo", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    args = parser.parse_args(argv)
    output_json = args.output_json or args.stage_dir / "report.json"
    output_markdown = args.output_markdown or args.stage_dir / "report.md"
    run_analysis(
        stage_dir=args.stage_dir,
        reference_repo=args.reference_repo,
        output_json=output_json,
        output_markdown=output_markdown,
    )
    return 0


__all__ = [
    "render_markdown",
    "run_analysis",
    "write_analysis_input_manifest",
    "write_reports",
]


if __name__ == "__main__":
    raise SystemExit(main())
