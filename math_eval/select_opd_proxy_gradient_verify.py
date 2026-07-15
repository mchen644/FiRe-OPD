"""Unified official selector and frozen random schedules for OPD verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from math_eval.deepmath_gradient_diversity import balanced_round_robin
from math_eval.opd_proxy_gradient_verify_artifacts import (
    VectorSet,
    atomic_save_npy,
    canonical_json_bytes,
    load_npy_strict,
    load_vector_set,
    sha256_file,
    sha256_id_lines,
    sha256_int_rows,
    write_or_validate_manifest,
)
from math_eval.prepare_opd_proxy_gradient_verify import stage_layout
from math_eval.select_gradient_diverse_deepmath import (
    CLUSTER_ITERATIONS,
    REFERENCE_COMMIT,
    REFERENCE_TREE,
    cluster_official,
    load_official_reference_classes,
)


PRIMARY_RATIO = 0.10
DIAGNOSTIC_RATIO = 0.01
KMEANS_SEEDS = (42, 43)
ROUND_ROBIN_SEED = 42
RANDOM_SEED_SEQUENCE = 2026071402
RANDOM_SPAWN_COUNT = 2
RANDOM_UNIFORM_FILE = "random_uniform.npy"
RANDOM_STRATIFIED_FILE = "random_stratified.npy"
RANDOM_MANIFEST_FILE = "random.manifest.json"
SELECTION_MANIFEST_FILE = "selection.manifest.json"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

EXPECTED_REPRESENTATIONS = tuple(
    [
        f"P_n1:seed={seed}:slot={slot}"
        for seed in KMEANS_SEEDS
        for slot in range(4)
    ]
    + [f"P_n4:seed={seed}" for seed in KMEANS_SEEDS]
    + ["S", "E"]
    + [f"T:seed={seed}" for seed in KMEANS_SEEDS]
)


@dataclass(frozen=True)
class RandomSchedules:
    uniform: np.ndarray
    stratified: np.ndarray
    manifest: dict[str, object]


@dataclass(frozen=True)
class SelectionBundle:
    representations: tuple[str, ...]
    stable_ids: tuple[str, ...]
    selected_positions: dict[str, np.ndarray]
    labels: dict[str, np.ndarray]
    k_values: tuple[int, int]
    kmeans_seeds: tuple[int, int]
    round_robin_seed: int
    manifest: dict[str, object]

    @staticmethod
    def key(
        representation: str,
        *,
        k: int,
        kmeans_seed: int,
        ratio: float = PRIMARY_RATIO,
    ) -> str:
        return _selection_key(representation, ratio, k, kmeans_seed)


def _selection_key(representation: str, ratio: float, k: int, seed: int) -> str:
    return f"{representation}|ratio={ratio:.2f}|k={k}|seed={seed}"


def _require_sha256(value: object, description: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{description} must be a lowercase SHA-256 digest")
    return value


def _normalize_parent_hashes(
    parent_hashes: Mapping[str, str] | None,
) -> dict[str, str]:
    if parent_hashes is None:
        return {}
    normalized = dict(parent_hashes)
    for name, value in normalized.items():
        if not isinstance(name, str) or not name:
            raise ValueError("parent hash names must be nonempty strings")
        _require_sha256(value, f"parent hash {name}")
    return dict(sorted(normalized.items()))


def _load_canonical_json(path: Path, description: str) -> dict[str, object]:
    try:
        payload = Path(path).read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read {description}: {error}") from error

    def reject_pairs(pairs):
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str):
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid {description}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    if payload != canonical_json_bytes(value):
        raise ValueError(f"{description} must use canonical JSON bytes")
    return value


def _validate_candidate_rows(
    rows: Sequence[Mapping[str, object]], expected_count: int
) -> tuple[tuple[str, ...], list[dict[str, object]]]:
    normalized = [dict(row) for row in rows]
    if len(normalized) != expected_count:
        raise ValueError(
            f"candidate row count mismatch: expected {expected_count}, got {len(normalized)}"
        )
    stable_ids: list[str] = []
    for index, row in enumerate(normalized):
        stable_id = row.get("stable_id")
        if not isinstance(stable_id, str) or not stable_id:
            raise ValueError(f"candidate row {index} requires a nonempty stable ID")
        if row.get("split", "candidate") != "candidate":
            raise ValueError("selector candidate rows must all have candidate split")
        stable_ids.append(stable_id)
    if len(set(stable_ids)) != len(stable_ids):
        raise ValueError("candidate stable IDs must be unique")
    return tuple(stable_ids), normalized


def build_length_quartiles(
    candidate_rows: Sequence[Mapping[str, object]],
) -> np.ndarray:
    """Assign rank quartiles using only ``(token count, stable ID)``."""
    if not candidate_rows:
        raise ValueError("length quartiles require at least one candidate row")
    sortable: list[tuple[int, str, int]] = []
    seen: set[str] = set()
    for position, row in enumerate(candidate_rows):
        stable_id = row.get("stable_id")
        token_count = row.get("prompt_token_count_0_6b")
        if not isinstance(stable_id, str) or not stable_id:
            raise ValueError("length quartile rows require nonempty stable IDs")
        if stable_id in seen:
            raise ValueError("length quartile stable IDs must be unique")
        if (
            isinstance(token_count, bool)
            or not isinstance(token_count, int)
            or token_count < 0
        ):
            raise ValueError("prompt_token_count_0_6b must be a nonnegative integer")
        seen.add(stable_id)
        sortable.append((token_count, stable_id, position))
    quartiles = np.empty(len(sortable), dtype=np.int8)
    for rank, (_, _, position) in enumerate(sorted(sortable)):
        quartiles[position] = math.floor(4 * rank / len(sortable))
    return np.ascontiguousarray(quartiles)


def largest_remainder_allocation(
    weights: Mapping[Hashable, int | float],
    *,
    total: int,
    capacities: Mapping[Hashable, int] | None = None,
) -> dict[Hashable, int]:
    """Allocate an exact total with lexicographic ties and capacity recovery."""
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise ValueError("allocation total must be a nonnegative integer")
    if not weights:
        if total:
            raise ValueError("cannot allocate a positive total without strata")
        return {}
    try:
        keys = sorted(weights)
    except TypeError as error:
        raise ValueError("allocation keys must share a lexicographic ordering") from error
    normalized_weights: dict[Hashable, float] = {}
    for key in keys:
        weight = weights[key]
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError("allocation weights must be finite nonnegative numbers")
        numeric = float(weight)
        if not math.isfinite(numeric) or numeric < 0:
            raise ValueError("allocation weights must be finite nonnegative numbers")
        normalized_weights[key] = numeric
    if capacities is None:
        normalized_capacities = {
            key: int(weights[key])
            if isinstance(weights[key], int) and not isinstance(weights[key], bool)
            else total
            for key in keys
        }
    else:
        if set(capacities) != set(keys):
            raise ValueError("capacity keys must exactly match allocation weight keys")
        normalized_capacities = {}
        for key in keys:
            capacity = capacities[key]
            if (
                isinstance(capacity, bool)
                or not isinstance(capacity, int)
                or capacity < 0
            ):
                raise ValueError("allocation capacities must be nonnegative integers")
            normalized_capacities[key] = capacity
    if total > sum(normalized_capacities.values()):
        raise ValueError("allocation total exceeds aggregate capacity")

    allocated = {key: 0 for key in keys}
    remaining = total
    while remaining:
        active = [
            key for key in keys if allocated[key] < normalized_capacities[key]
        ]
        if not active:
            raise RuntimeError("capacity redistribution made no progress")
        active_weights = {key: normalized_weights[key] for key in active}
        weight_sum = sum(active_weights.values())
        if weight_sum <= 0:
            active_weights = {
                key: float(normalized_capacities[key] - allocated[key])
                for key in active
            }
            weight_sum = sum(active_weights.values())
        quotas = {
            key: remaining * active_weights[key] / weight_sum for key in active
        }
        before = remaining
        for key in active:
            base = min(
                math.floor(quotas[key]),
                normalized_capacities[key] - allocated[key],
            )
            allocated[key] += base
            remaining -= base
        if not remaining:
            break
        remainder_order = sorted(
            active,
            key=lambda key: (-(quotas[key] - math.floor(quotas[key])), key),
        )
        for key in remainder_order:
            if remaining == 0:
                break
            if allocated[key] < normalized_capacities[key]:
                allocated[key] += 1
                remaining -= 1
        if remaining == before:
            raise RuntimeError("capacity redistribution made no progress")
    return allocated


def _strata(
    rows: Sequence[Mapping[str, object]], quartiles: np.ndarray
) -> dict[tuple[str, int], list[int]]:
    strata: dict[tuple[str, int], list[int]] = {}
    for position, (row, quartile) in enumerate(zip(rows, quartiles.tolist())):
        leaf_topic = row.get("leaf_topic")
        stable_id = row.get("stable_id")
        if not isinstance(leaf_topic, str) or not leaf_topic:
            raise ValueError("random strata require a nonempty leaf_topic")
        if not isinstance(stable_id, str) or not stable_id:
            raise ValueError("random strata require a nonempty stable_id")
        strata.setdefault((leaf_topic, int(quartile)), []).append(position)
    for key in strata:
        strata[key].sort(key=lambda position: str(rows[position]["stable_id"]))
    return dict(sorted(strata.items()))


def _int32_c(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=np.dtype("<i4"))


def _validate_schedule_array(
    array: np.ndarray,
    *,
    draws: int,
    selected_size: int,
    candidate_count: int,
    description: str,
) -> None:
    if array.dtype.str != "<i4" or not array.flags.c_contiguous:
        raise ValueError(f"{description} schedule must be little-endian C int32")
    if array.shape != (draws, selected_size):
        raise ValueError(f"{description} schedule shape mismatch")
    if selected_size and (
        np.any(array < 0)
        or np.any(array >= candidate_count)
        or np.any(np.diff(array, axis=1) <= 0)
    ):
        raise ValueError(
            f"{description} schedule rows must be sorted unique candidate positions"
        )


def _write_or_validate_npy(path: Path, array: np.ndarray) -> None:
    target = Path(path)
    if target.exists():
        existing = load_npy_strict(
            target, dtype=array.dtype, shape=tuple(array.shape)
        )
        if not np.array_equal(existing, array):
            raise ValueError(f"existing NumPy artifact differs: {target}")
        return
    atomic_save_npy(target, array)


def generate_random_schedules(
    candidate_rows: Sequence[Mapping[str, object]],
    *,
    selected_size: int,
    draws: int,
    output_directory: Path | None = None,
    parent_hashes: Mapping[str, str] | None = None,
    stage: int | None = None,
) -> RandomSchedules:
    """Generate the two pre-registered independent random subset streams."""
    stable_ids, rows = _validate_candidate_rows(candidate_rows, len(candidate_rows))
    candidate_count = len(rows)
    if candidate_count == 0:
        raise ValueError("random schedules require at least one candidate")
    if (
        isinstance(selected_size, bool)
        or not isinstance(selected_size, int)
        or selected_size <= 0
        or selected_size > candidate_count
    ):
        raise ValueError("selected_size must be in [1, candidate_count]")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws <= 0:
        raise ValueError("draws must be a positive integer")
    if stage is not None:
        if isinstance(stage, bool) or not isinstance(stage, int):
            raise ValueError("random schedule stage must be an integer")
        layout = stage_layout(stage)
        if (
            candidate_count != len(layout.candidate_clean_positions)
            or selected_size != layout.selected_size
            or draws != layout.null_draws
        ):
            raise ValueError("random schedule cardinalities differ from stage layout")

    children = np.random.SeedSequence(RANDOM_SEED_SEQUENCE).spawn(
        RANDOM_SPAWN_COUNT
    )
    uniform_rng = np.random.Generator(np.random.PCG64(children[0]))
    stratified_rng = np.random.Generator(np.random.PCG64(children[1]))
    uniform = np.empty((draws, selected_size), dtype=np.dtype("<i4"))
    for replicate in range(draws):
        chosen = uniform_rng.choice(
            candidate_count,
            size=selected_size,
            replace=False,
            shuffle=False,
        )
        uniform[replicate] = np.sort(chosen)

    quartiles = build_length_quartiles(rows)
    strata = _strata(rows, quartiles)
    allocation = largest_remainder_allocation(
        {key: len(members) for key, members in strata.items()},
        total=selected_size,
        capacities={key: len(members) for key, members in strata.items()},
    )
    stratified = np.empty((draws, selected_size), dtype=np.dtype("<i4"))
    for replicate in range(draws):
        pieces: list[np.ndarray] = []
        for key in sorted(strata):
            count = allocation[key]
            if count:
                members = np.asarray(strata[key], dtype=np.int64)
                pieces.append(
                    np.asarray(
                        stratified_rng.choice(
                            members,
                            size=count,
                            replace=False,
                            shuffle=False,
                        ),
                        dtype=np.int64,
                    )
                )
        chosen = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)
        stratified[replicate] = np.sort(chosen)

    uniform = _int32_c(uniform)
    stratified = _int32_c(stratified)
    _validate_schedule_array(
        uniform,
        draws=draws,
        selected_size=selected_size,
        candidate_count=candidate_count,
        description="uniform",
    )
    _validate_schedule_array(
        stratified,
        draws=draws,
        selected_size=selected_size,
        candidate_count=candidate_count,
        description="stratified",
    )
    normalized_parents = _normalize_parent_hashes(parent_hashes)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "opd_proxy_random_schedules",
        "candidate_count": candidate_count,
        "selected_size": selected_size,
        "draws": draws,
        "candidate_ids": list(stable_ids),
        "candidate_ids_sha256": sha256_id_lines(stable_ids),
        "seed_sequence": RANDOM_SEED_SEQUENCE,
        "spawn_count": RANDOM_SPAWN_COUNT,
        "bit_generator": "PCG64",
        "numpy_version": np.__version__,
        "choice_replace": False,
        "choice_shuffle": False,
        "stored_order": "ascending_candidate_position",
        "paired_target_seeds": list(KMEANS_SEEDS),
        "length_quartile_rule": "floor(4 * rank / candidate_count)",
        "stratification_fields": [
            "leaf_topic",
            "qwen3_0_6b_prompt_token_length_quartile",
        ],
        "strata": [
            {
                "leaf_topic": key[0],
                "length_quartile": key[1],
                "capacity": len(strata[key]),
                "allocation": allocation[key],
                "member_positions": strata[key],
            }
            for key in sorted(strata)
        ],
        "uniform_logical_sha256": sha256_int_rows(uniform),
        "stratified_logical_sha256": sha256_int_rows(stratified),
        "parent_hashes": normalized_parents,
    }
    if stage is not None:
        manifest["stage"] = stage
    if output_directory is not None:
        root = Path(output_directory)
        root.mkdir(parents=True, exist_ok=True)
        uniform_path = root / RANDOM_UNIFORM_FILE
        stratified_path = root / RANDOM_STRATIFIED_FILE
        _write_or_validate_npy(uniform_path, uniform)
        _write_or_validate_npy(stratified_path, stratified)
        manifest["files"] = {
            "uniform": {
                "path": RANDOM_UNIFORM_FILE,
                "sha256": sha256_file(uniform_path),
            },
            "stratified": {
                "path": RANDOM_STRATIFIED_FILE,
                "sha256": sha256_file(stratified_path),
            },
        }
        write_or_validate_manifest(root / RANDOM_MANIFEST_FILE, manifest)
    return RandomSchedules(uniform, stratified, manifest)


def _manifest_root(path: Path, filename: str) -> tuple[Path, Path]:
    supplied = Path(path)
    if supplied.is_dir():
        return supplied, supplied / filename
    if supplied.name != filename:
        raise ValueError(f"expected {filename}, got {supplied.name}")
    return supplied.parent, supplied


def load_random_schedules(
    path: Path,
    *,
    expected_candidate_ids: Sequence[str] | None = None,
    expected_candidate_rows: Sequence[Mapping[str, object]] | None = None,
    expected_parent_hashes: Mapping[str, str] | None = None,
) -> RandomSchedules:
    root, manifest_path = _manifest_root(path, RANDOM_MANIFEST_FILE)
    manifest = _load_canonical_json(manifest_path, "random schedule manifest")
    if manifest.get("schema_version") != 1 or manifest.get("artifact_type") != (
        "opd_proxy_random_schedules"
    ):
        raise ValueError("invalid random schedule manifest contract")
    for field in ("candidate_count", "selected_size", "draws"):
        value = manifest.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"random manifest {field} must be positive")
    candidate_count = int(manifest["candidate_count"])
    selected_size = int(manifest["selected_size"])
    draws = int(manifest["draws"])
    stored_stage = manifest.get("stage")
    if stored_stage is not None:
        if isinstance(stored_stage, bool) or not isinstance(stored_stage, int):
            raise ValueError("random schedule stage must be an integer")
        layout = stage_layout(stored_stage)
        if (
            candidate_count != len(layout.candidate_clean_positions)
            or selected_size != layout.selected_size
            or draws != layout.null_draws
        ):
            raise ValueError("random schedule cardinalities differ from stage layout")
    candidate_ids_value = manifest.get("candidate_ids")
    if not isinstance(candidate_ids_value, list) or any(
        not isinstance(value, str) or not value for value in candidate_ids_value
    ):
        raise ValueError("random manifest candidate IDs are invalid")
    candidate_ids = tuple(candidate_ids_value)
    if len(candidate_ids) != candidate_count or len(set(candidate_ids)) != candidate_count:
        raise ValueError("random manifest candidate ID coverage mismatch")
    if sha256_id_lines(candidate_ids) != manifest.get("candidate_ids_sha256"):
        raise ValueError("random manifest candidate ID hash mismatch")
    if expected_candidate_ids is not None and tuple(expected_candidate_ids) != candidate_ids:
        raise ValueError("random schedule candidate IDs differ from expected order")
    if expected_candidate_rows is not None:
        expected_row_ids, expected_rows = _validate_candidate_rows(
            expected_candidate_rows, candidate_count
        )
        if expected_row_ids != candidate_ids:
            raise ValueError("random schedule candidate rows differ from stored IDs")
    else:
        expected_rows = None
    manifest_parents = manifest.get("parent_hashes")
    if not isinstance(manifest_parents, dict):
        raise ValueError("random schedule parent hashes must be an object")
    normalized_manifest_parents = _normalize_parent_hashes(manifest_parents)
    if normalized_manifest_parents != manifest_parents:
        raise ValueError("random schedule parent hashes are not canonical")
    expected_parents = (
        _normalize_parent_hashes(expected_parent_hashes)
        if expected_parent_hashes is not None
        else None
    )
    if expected_parents is not None and manifest_parents != expected_parents:
        raise ValueError("random schedule parent hashes differ from expected provenance")
    if manifest.get("seed_sequence") != RANDOM_SEED_SEQUENCE or manifest.get(
        "spawn_count"
    ) != RANDOM_SPAWN_COUNT:
        raise ValueError("random schedule seed/spawn contract mismatch")
    fixed_generator_fields = {
        "bit_generator": "PCG64",
        "numpy_version": np.__version__,
        "choice_replace": False,
        "choice_shuffle": False,
        "stored_order": "ascending_candidate_position",
        "paired_target_seeds": list(KMEANS_SEEDS),
        "length_quartile_rule": "floor(4 * rank / candidate_count)",
        "stratification_fields": [
            "leaf_topic",
            "qwen3_0_6b_prompt_token_length_quartile",
        ],
    }
    if any(manifest.get(name) != value for name, value in fixed_generator_fields.items()):
        raise ValueError("random schedule generator/stratification contract mismatch")

    stratum_records = manifest.get("strata")
    if not isinstance(stratum_records, list) or not stratum_records:
        raise ValueError("random schedule strata must be a nonempty list")
    stored_strata: dict[tuple[str, int], list[int]] = {}
    stored_allocation: dict[tuple[str, int], int] = {}
    previous_key: tuple[str, int] | None = None
    covered_positions: list[int] = []
    for record in stratum_records:
        if not isinstance(record, dict):
            raise ValueError("random schedule stratum record must be an object")
        leaf_topic = record.get("leaf_topic")
        quartile = record.get("length_quartile")
        capacity = record.get("capacity")
        allocated = record.get("allocation")
        members = record.get("member_positions")
        if not isinstance(leaf_topic, str) or not leaf_topic:
            raise ValueError("random schedule stratum leaf topic is invalid")
        if (
            isinstance(quartile, bool)
            or not isinstance(quartile, int)
            or quartile not in range(4)
        ):
            raise ValueError("random schedule stratum quartile is invalid")
        key = (leaf_topic, quartile)
        if previous_key is not None and key <= previous_key:
            raise ValueError("random schedule strata are not lexicographically ordered")
        previous_key = key
        if not isinstance(members, list) or any(
            isinstance(position, bool) or not isinstance(position, int)
            for position in members
        ):
            raise ValueError("random schedule stratum members are invalid")
        if any(position < 0 or position >= candidate_count for position in members):
            raise ValueError("random schedule stratum member is out of range")
        if members != sorted(members, key=lambda position: candidate_ids[position]):
            raise ValueError("random schedule stratum member order is invalid")
        if (
            isinstance(capacity, bool)
            or not isinstance(capacity, int)
            or capacity != len(members)
            or isinstance(allocated, bool)
            or not isinstance(allocated, int)
        ):
            raise ValueError("random schedule stratum capacity/allocation is invalid")
        stored_strata[key] = members
        stored_allocation[key] = allocated
        covered_positions.extend(members)
    if sorted(covered_positions) != list(range(candidate_count)):
        raise ValueError("random schedule strata do not partition candidate positions")
    expected_allocation = largest_remainder_allocation(
        {key: len(members) for key, members in stored_strata.items()},
        total=selected_size,
        capacities={key: len(members) for key, members in stored_strata.items()},
    )
    if stored_allocation != expected_allocation:
        raise ValueError("random schedule stratum allocation is not largest-remainder")
    if expected_rows is not None:
        quartiles = build_length_quartiles(expected_rows)
        recomputed_strata = _strata(expected_rows, quartiles)
        if recomputed_strata != stored_strata:
            raise ValueError("random schedule strata differ from candidate metadata")

    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("random schedule manifest lacks file records")

    arrays: dict[str, np.ndarray] = {}
    for name, filename in (
        ("uniform", RANDOM_UNIFORM_FILE),
        ("stratified", RANDOM_STRATIFIED_FILE),
    ):
        record = files.get(name)
        if not isinstance(record, dict) or record.get("path") != filename:
            raise ValueError(f"invalid {name} random file record")
        expected_sha = _require_sha256(record.get("sha256"), f"{name} file SHA")
        artifact_path = root / filename
        if sha256_file(artifact_path) != expected_sha:
            raise ValueError(f"{name} random file SHA mismatch")
        array = load_npy_strict(
            artifact_path,
            dtype=np.dtype("<i4"),
            shape=(draws, selected_size),
        )
        _validate_schedule_array(
            array,
            draws=draws,
            selected_size=selected_size,
            candidate_count=candidate_count,
            description=name,
        )
        logical_field = f"{name}_logical_sha256"
        if sha256_int_rows(array) != manifest.get(logical_field):
            raise ValueError(f"{name} random logical row hash mismatch")
        arrays[name] = array
    position_to_stratum = np.empty(candidate_count, dtype=np.int32)
    ordered_strata = list(stored_strata)
    for stratum_index, key in enumerate(ordered_strata):
        position_to_stratum[np.asarray(stored_strata[key], dtype=np.int64)] = (
            stratum_index
        )
    expected_counts = np.asarray(
        [stored_allocation[key] for key in ordered_strata], dtype=np.int64
    )
    for row in arrays["stratified"]:
        actual_counts = np.bincount(
            position_to_stratum[row], minlength=len(ordered_strata)
        )
        if not np.array_equal(actual_counts, expected_counts):
            raise ValueError("stratified random row violates stored allocation")
    return RandomSchedules(arrays["uniform"], arrays["stratified"], manifest)


def _infer_stage(candidate_count: int) -> int:
    matches = [
        stage
        for stage in (0, 1, 2)
        if len(stage_layout(stage).candidate_clean_positions) == candidate_count
    ]
    if len(matches) != 1:
        raise ValueError(f"cannot infer stage from {candidate_count} candidate rows")
    return matches[0]


def _validate_vector_sets(
    vector_sets: Mapping[str, VectorSet],
    *,
    candidate_ids: tuple[str, ...] | None,
    candidate_count: int,
    expected_vector_manifest_hashes: Mapping[str, str] | None,
) -> tuple[tuple[str, ...], dict[str, str]]:
    if tuple(vector_sets.keys()) != EXPECTED_REPRESENTATIONS:
        if set(vector_sets) != set(EXPECTED_REPRESENTATIONS):
            missing = sorted(set(EXPECTED_REPRESENTATIONS) - set(vector_sets))
            extra = sorted(set(vector_sets) - set(EXPECTED_REPRESENTATIONS))
            raise ValueError(
                f"selector representations mismatch; missing={missing}, extra={extra}"
            )
        vector_sets = {name: vector_sets[name] for name in EXPECTED_REPRESENTATIONS}
    first_ids: tuple[str, ...] | None = candidate_ids
    actual_manifest_hashes: dict[str, str] = {}
    for name in EXPECTED_REPRESENTATIONS:
        vector_set = vector_sets[name]
        if not isinstance(vector_set, VectorSet):
            raise TypeError(f"selector input {name} must be a VectorSet")
        stable_ids = tuple(vector_set.stable_ids)
        if len(stable_ids) != candidate_count or len(set(stable_ids)) != candidate_count:
            raise ValueError(f"{name} stable ID coverage is not one-to-one")
        if first_ids is None:
            first_ids = stable_ids
        if stable_ids != first_ids:
            raise ValueError(f"{name} stable ID order differs from selector candidates")
        if len(vector_set.vector_ids) != candidate_count or len(
            set(vector_set.vector_ids)
        ) != candidate_count:
            raise ValueError(f"{name} vector IDs must be unique and complete")
        vectors = np.asarray(vector_set.vectors)
        if vectors.ndim != 2 or vectors.shape[0] != candidate_count:
            raise ValueError(f"{name} vector matrix shape mismatch")
        if vectors.dtype != np.dtype(np.float32) or not np.isfinite(vectors).all():
            raise ValueError(f"{name} vector matrix must be finite float32")
        norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
        if not np.isfinite(norms).all() or np.any(norms == 0):
            raise ValueError(f"{name} vector matrix contains an invalid norm")
        manifest = vector_set.manifest
        if not isinstance(manifest, dict) or manifest.get("artifact_type") not in {
            "vector_set",
            "opd_proxy_selection_vector_view",
        }:
            raise ValueError(f"{name} vector manifest has an invalid artifact type")
        expected_manifest_fields = {
            "vector_count": candidate_count,
            "vector_dimension": vectors.shape[1],
            "vector_ids_sha256": sha256_id_lines(vector_set.vector_ids),
            "stable_ids_sha256": sha256_id_lines(stable_ids),
        }
        if any(manifest.get(field) != value for field, value in expected_manifest_fields.items()):
            raise ValueError(f"{name} vector manifest does not match loaded vectors")
        manifest_hash = hashlib.sha256(
            canonical_json_bytes(vector_set.manifest)
        ).hexdigest()
        actual_manifest_hashes[name] = manifest_hash
    assert first_ids is not None
    if expected_vector_manifest_hashes is not None:
        expected = dict(expected_vector_manifest_hashes)
        if set(expected) != set(EXPECTED_REPRESENTATIONS):
            raise ValueError("expected vector manifest hash keys are incomplete")
        for name in EXPECTED_REPRESENTATIONS:
            _require_sha256(expected[name], f"expected {name} vector manifest hash")
            if expected[name] != actual_manifest_hashes[name]:
                raise ValueError(f"{name} vector manifest hash mismatch")
    return first_ids, actual_manifest_hashes


def _validate_single_visible_cuda(env: Mapping[str, str] | None = None) -> None:
    runtime = os.environ if env is None else env
    visible = runtime.get("CUDA_VISIBLE_DEVICES", "")
    tokens = [token.strip() for token in visible.split(",") if token.strip()]
    if len(tokens) != 1 or len(set(tokens)) != 1:
        raise RuntimeError(
            "official selector requires exactly one allocation-owned visible CUDA device"
        )


def _selection_file_record(path: Path, logical_hash: str) -> dict[str, object]:
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "logical_sha256": logical_hash,
    }


def run_selection(
    vector_sets: Mapping[str, VectorSet],
    *,
    stage: int | None = None,
    candidate_rows: Sequence[Mapping[str, object]] | None = None,
    reference_repo: Path = Path("/home/mchen/prismatic-synthesis-reference"),
    output_directory: Path | None = None,
    parent_hashes: Mapping[str, str] | None = None,
    expected_vector_manifest_hashes: Mapping[str, str] | None = None,
    fake_cluster_manager=None,
) -> SelectionBundle:
    """Run both fixed official K-means seeds for every declared representation."""
    if not isinstance(vector_sets, Mapping) or not vector_sets:
        raise ValueError("selector requires vector sets")
    first_vector = next(iter(vector_sets.values()))
    if not isinstance(first_vector, VectorSet):
        raise TypeError("selector inputs must be VectorSet objects")
    candidate_count = len(first_vector.stable_ids)
    if stage is not None and (isinstance(stage, bool) or not isinstance(stage, int)):
        raise ValueError("selector stage must be an integer")
    selected_stage = _infer_stage(candidate_count) if stage is None else stage
    layout = stage_layout(selected_stage)
    if candidate_count != len(layout.candidate_clean_positions):
        raise ValueError("vector candidate count differs from the selected stage layout")
    if candidate_rows is None:
        candidate_ids: tuple[str, ...] | None = None
        normalized_rows: list[dict[str, object]] | None = None
    else:
        candidate_ids, normalized_rows = _validate_candidate_rows(
            candidate_rows, candidate_count
        )
    stable_ids, actual_manifest_hashes = _validate_vector_sets(
        vector_sets,
        candidate_ids=candidate_ids,
        candidate_count=candidate_count,
        expected_vector_manifest_hashes=expected_vector_manifest_hashes,
    )
    vector_sets = {name: vector_sets[name] for name in EXPECTED_REPRESENTATIONS}
    normalized_parents = _normalize_parent_hashes(parent_hashes)

    if fake_cluster_manager is None:
        _validate_single_visible_cuda()
        manager, _, reference_provenance = load_official_reference_classes(
            Path(reference_repo)
        )
        cluster_manager = manager
        cluster_device = "cuda:0"
        reference_record: dict[str, object] = dict(reference_provenance)
        reference_record["mode"] = "pinned_official_cuda"
    else:
        cluster_manager = fake_cluster_manager
        cluster_device = None
        reference_record = {
            "mode": "injected_test_cluster_manager",
            "reference_commit": REFERENCE_COMMIT,
            "reference_tree": REFERENCE_TREE,
        }

    selected_positions: dict[str, np.ndarray] = {}
    labels_by_key: dict[str, np.ndarray] = {}
    run_records: list[dict[str, object]] = []
    ratios_and_k = (
        (PRIMARY_RATIO, layout.primary_k, "primary"),
        (DIAGNOSTIC_RATIO, layout.diagnostic_k, "diagnostic"),
    )
    root = Path(output_directory) if output_directory is not None else None
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
    run_index = 0
    for representation in EXPECTED_REPRESENTATIONS:
        vectors = np.ascontiguousarray(
            np.asarray(vector_sets[representation].vectors), dtype=np.float32
        )
        tensor = torch.from_numpy(vectors)
        for ratio, expected_k, ratio_name in ratios_and_k:
            derived_k = max(2, min(candidate_count, math.floor(ratio * candidate_count)))
            if derived_k != expected_k:
                raise ValueError("stage K does not match ratio-derived cardinality")
            for kmeans_seed in KMEANS_SEEDS:
                labels64 = cluster_official(
                    tensor,
                    ratio=ratio,
                    iterations=CLUSTER_ITERATIONS,
                    seed=kmeans_seed,
                    reference_repo=Path(reference_repo),
                    device=cluster_device,
                    cluster_manager_class=cluster_manager,
                )
                labels = _int32_c(np.asarray(labels64).reshape(candidate_count))
                if np.any(labels < 0) or np.any(labels >= expected_k):
                    raise ValueError("official selector labels are outside [0, K)")
                selected = _int32_c(
                    np.asarray(
                        balanced_round_robin(
                            labels.tolist(),
                            target_size=layout.selected_size,
                            seed=ROUND_ROBIN_SEED,
                        )
                    )
                )
                if (
                    selected.shape != (layout.selected_size,)
                    or len(set(selected.tolist())) != layout.selected_size
                    or np.any(selected < 0)
                    or np.any(selected >= candidate_count)
                ):
                    raise ValueError("balanced round robin returned invalid positions")
                key = _selection_key(
                    representation, ratio, expected_k, kmeans_seed
                )
                if key in selected_positions:
                    raise RuntimeError("duplicate selector run key")
                selected_positions[key] = selected
                labels_by_key[key] = labels
                record: dict[str, object] = {
                    "run_index": run_index,
                    "key": key,
                    "representation": representation,
                    "ratio_name": ratio_name,
                    "ratio": ratio,
                    "k": expected_k,
                    "kmeans_seed": kmeans_seed,
                    "round_robin_seed": ROUND_ROBIN_SEED,
                    "labels_logical_sha256": sha256_int_rows(
                        labels.reshape(1, -1)
                    ),
                    "selected_logical_sha256": sha256_int_rows(
                        selected.reshape(1, -1)
                    ),
                }
                if root is not None:
                    labels_path = root / f"labels.{run_index:03d}.npy"
                    selected_path = root / f"selected.{run_index:03d}.npy"
                    _write_or_validate_npy(labels_path, labels)
                    _write_or_validate_npy(selected_path, selected)
                    record["labels_file"] = _selection_file_record(
                        labels_path, str(record["labels_logical_sha256"])
                    )
                    record["selected_file"] = _selection_file_record(
                        selected_path, str(record["selected_logical_sha256"])
                    )
                run_records.append(record)
                run_index += 1

    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "opd_proxy_selection_bundle",
        "stage": selected_stage,
        "candidate_count": candidate_count,
        "selected_size": layout.selected_size,
        "candidate_ids": list(stable_ids),
        "candidate_ids_sha256": sha256_id_lines(stable_ids),
        "representations": list(EXPECTED_REPRESENTATIONS),
        "representation_vector_manifest_sha256": actual_manifest_hashes,
        "cluster_ratios": [PRIMARY_RATIO, DIAGNOSTIC_RATIO],
        "k_values": [layout.primary_k, layout.diagnostic_k],
        "kmeans_seeds": list(KMEANS_SEEDS),
        "round_robin_seed": ROUND_ROBIN_SEED,
        "cluster_iterations": CLUSTER_ITERATIONS,
        "distance": "cosine_on_l2_normalized_rows",
        "sampler": "cluster_balanced_round_robin",
        "parent_hashes": normalized_parents,
        "official_reference": reference_record,
        "runs": run_records,
    }
    if normalized_rows is not None:
        metadata_rows = [
            {
                "stable_id": row["stable_id"],
                "leaf_topic": row.get("leaf_topic"),
                "prompt_token_count_0_6b": row.get(
                    "prompt_token_count_0_6b"
                ),
            }
            for row in normalized_rows
        ]
        manifest["candidate_metadata_sha256"] = hashlib.sha256(
            b"".join(canonical_json_bytes(row) for row in metadata_rows)
        ).hexdigest()
    if root is not None:
        write_or_validate_manifest(root / SELECTION_MANIFEST_FILE, manifest)
    return SelectionBundle(
        representations=EXPECTED_REPRESENTATIONS,
        stable_ids=stable_ids,
        selected_positions=selected_positions,
        labels=labels_by_key,
        k_values=(layout.primary_k, layout.diagnostic_k),
        kmeans_seeds=KMEANS_SEEDS,
        round_robin_seed=ROUND_ROBIN_SEED,
        manifest=manifest,
    )


def _validate_selection_manifest_header(
    manifest: Mapping[str, object],
) -> tuple[int, int, int, tuple[str, ...]]:
    if manifest.get("schema_version") != 1 or manifest.get("artifact_type") != (
        "opd_proxy_selection_bundle"
    ):
        raise ValueError("invalid selection bundle manifest contract")
    stage = manifest.get("stage")
    if isinstance(stage, bool) or not isinstance(stage, int):
        raise ValueError("selection stage must be an integer")
    layout = stage_layout(stage)
    candidate_count = len(layout.candidate_clean_positions)
    if manifest.get("candidate_count") != candidate_count:
        raise ValueError("selection candidate count differs from stage layout")
    if manifest.get("selected_size") != layout.selected_size:
        raise ValueError("selection size differs from stage layout")
    if manifest.get("k_values") != [layout.primary_k, layout.diagnostic_k]:
        raise ValueError("selection K values differ from stage layout")
    if manifest.get("cluster_ratios") != [PRIMARY_RATIO, DIAGNOSTIC_RATIO]:
        raise ValueError("selection cluster ratios differ from fixed contract")
    if manifest.get("kmeans_seeds") != list(KMEANS_SEEDS):
        raise ValueError("selection K-means seeds differ from fixed contract")
    if manifest.get("round_robin_seed") != ROUND_ROBIN_SEED:
        raise ValueError("selection round-robin seed differs from fixed contract")
    if manifest.get("cluster_iterations") != CLUSTER_ITERATIONS:
        raise ValueError("selection iteration count differs from official contract")
    if manifest.get("representations") != list(EXPECTED_REPRESENTATIONS):
        raise ValueError("selection representations differ from fixed contract")
    candidate_ids_value = manifest.get("candidate_ids")
    if not isinstance(candidate_ids_value, list) or any(
        not isinstance(value, str) or not value for value in candidate_ids_value
    ):
        raise ValueError("selection candidate IDs are invalid")
    candidate_ids = tuple(candidate_ids_value)
    if len(candidate_ids) != candidate_count or len(set(candidate_ids)) != candidate_count:
        raise ValueError("selection candidate ID coverage mismatch")
    if sha256_id_lines(candidate_ids) != manifest.get("candidate_ids_sha256"):
        raise ValueError("selection candidate ID hash mismatch")
    return stage, candidate_count, layout.selected_size, candidate_ids


def load_selection_bundle(
    path: Path,
    *,
    expected_candidate_ids: Sequence[str] | None = None,
    expected_vector_manifest_hashes: Mapping[str, str] | None = None,
    expected_parent_hashes: Mapping[str, str] | None = None,
) -> SelectionBundle:
    root, manifest_path = _manifest_root(path, SELECTION_MANIFEST_FILE)
    manifest = _load_canonical_json(manifest_path, "selection bundle manifest")
    stage, candidate_count, selected_size, candidate_ids = (
        _validate_selection_manifest_header(manifest)
    )
    del stage
    if expected_candidate_ids is not None and tuple(expected_candidate_ids) != candidate_ids:
        raise ValueError("selection candidate IDs differ from expected order")
    vector_hashes = manifest.get("representation_vector_manifest_sha256")
    if not isinstance(vector_hashes, dict) or set(vector_hashes) != set(
        EXPECTED_REPRESENTATIONS
    ):
        raise ValueError("selection vector manifest hash coverage is incomplete")
    for name in EXPECTED_REPRESENTATIONS:
        _require_sha256(vector_hashes[name], f"selection {name} vector manifest hash")
    if expected_vector_manifest_hashes is not None and dict(
        expected_vector_manifest_hashes
    ) != vector_hashes:
        raise ValueError("selection vector manifest hashes differ from expected provenance")
    manifest_parents = manifest.get("parent_hashes")
    if not isinstance(manifest_parents, dict) or _normalize_parent_hashes(
        manifest_parents
    ) != manifest_parents:
        raise ValueError("selection parent hashes are invalid or noncanonical")
    if expected_parent_hashes is not None and manifest_parents != _normalize_parent_hashes(
        expected_parent_hashes
    ):
        raise ValueError("selection parent hashes differ from expected provenance")
    if manifest.get("distance") != "cosine_on_l2_normalized_rows" or manifest.get(
        "sampler"
    ) != "cluster_balanced_round_robin":
        raise ValueError("selection distance/sampler contract mismatch")
    reference = manifest.get("official_reference")
    if (
        not isinstance(reference, dict)
        or reference.get("reference_commit") != REFERENCE_COMMIT
        or reference.get("reference_tree") != REFERENCE_TREE
        or reference.get("mode")
        not in {"pinned_official_cuda", "injected_test_cluster_manager"}
    ):
        raise ValueError("selection official reference provenance mismatch")

    records = manifest.get("runs")
    expected_run_count = len(EXPECTED_REPRESENTATIONS) * 2 * len(KMEANS_SEEDS)
    if not isinstance(records, list) or len(records) != expected_run_count:
        raise ValueError("selection run coverage is incomplete")
    layout = stage_layout(int(manifest["stage"]))
    expected_specs = [
        (representation, ratio, k, ratio_name, seed)
        for representation in EXPECTED_REPRESENTATIONS
        for ratio, k, ratio_name in (
            (PRIMARY_RATIO, layout.primary_k, "primary"),
            (DIAGNOSTIC_RATIO, layout.diagnostic_k, "diagnostic"),
        )
        for seed in KMEANS_SEEDS
    ]
    selected_positions: dict[str, np.ndarray] = {}
    labels_by_key: dict[str, np.ndarray] = {}
    listed_files: set[str] = set()
    for run_index, (record, expected_spec) in enumerate(zip(records, expected_specs)):
        if not isinstance(record, dict):
            raise ValueError("selection run record must be an object")
        representation, ratio, k, ratio_name, seed = expected_spec
        key = _selection_key(representation, ratio, k, seed)
        expected_fields = {
            "run_index": run_index,
            "key": key,
            "representation": representation,
            "ratio_name": ratio_name,
            "ratio": ratio,
            "k": k,
            "kmeans_seed": seed,
            "round_robin_seed": ROUND_ROBIN_SEED,
        }
        if any(record.get(name) != value for name, value in expected_fields.items()):
            raise ValueError("selection run ordering or fixed fields mismatch")
        arrays: dict[str, np.ndarray] = {}
        for array_name, filename, shape in (
            ("labels", f"labels.{run_index:03d}.npy", (candidate_count,)),
            ("selected", f"selected.{run_index:03d}.npy", (selected_size,)),
        ):
            file_record = record.get(f"{array_name}_file")
            if not isinstance(file_record, dict) or file_record.get("path") != filename:
                raise ValueError(f"selection {array_name} file record mismatch")
            listed_files.add(filename)
            expected_sha = _require_sha256(
                file_record.get("sha256"), f"selection {array_name} file SHA"
            )
            artifact_path = root / filename
            if sha256_file(artifact_path) != expected_sha:
                raise ValueError(f"selection {array_name} file SHA mismatch")
            array = load_npy_strict(
                artifact_path, dtype=np.dtype("<i4"), shape=shape
            )
            logical = sha256_int_rows(array.reshape(1, -1))
            if logical != record.get(f"{array_name}_logical_sha256") or logical != (
                file_record.get("logical_sha256")
            ):
                raise ValueError(f"selection {array_name} logical hash mismatch")
            arrays[array_name] = array
        labels = arrays["labels"]
        selected = arrays["selected"]
        if np.any(labels < 0) or np.any(labels >= k):
            raise ValueError("selection labels are outside [0, K)")
        if (
            len(set(selected.tolist())) != selected_size
            or np.any(selected < 0)
            or np.any(selected >= candidate_count)
        ):
            raise ValueError("selection positions are not a valid unique subset")
        reproduced = np.asarray(
            balanced_round_robin(
                labels.tolist(), target_size=selected_size, seed=ROUND_ROBIN_SEED
            ),
            dtype=np.int32,
        )
        if not np.array_equal(selected, reproduced):
            raise ValueError("selection sampler order differs from balanced round robin")
        labels_by_key[key] = labels
        selected_positions[key] = selected
    discovered = {
        path.name
        for pattern in ("labels.*.npy", "selected.*.npy")
        for path in root.glob(pattern)
    }
    if discovered != listed_files:
        raise ValueError("selection directory has missing or unlisted array artifacts")
    return SelectionBundle(
        representations=EXPECTED_REPRESENTATIONS,
        stable_ids=candidate_ids,
        selected_positions=selected_positions,
        labels=labels_by_key,
        k_values=(layout.primary_k, layout.diagnostic_k),
        kmeans_seeds=KMEANS_SEEDS,
        round_robin_seed=ROUND_ROBIN_SEED,
        manifest=manifest,
    )


def selection_vector_id(representation: str, stable_id: str) -> str:
    if representation.startswith("P_n1:"):
        match = re.fullmatch(r"P_n1:seed=(42|43):slot=([0-3])", representation)
        if match is None:
            raise ValueError("invalid P_n1 representation name")
        return f"P:{stable_id}:seed={match.group(1)}:slot={match.group(2)}:n1"
    if representation.startswith("P_n4:"):
        seed = representation.rsplit("=", 1)[1]
        return f"P:{stable_id}:seed={seed}:n4"
    if representation.startswith("T:"):
        seed = representation.rsplit("=", 1)[1]
        return f"T:{stable_id}:seed={seed}:n4"
    return f"{representation}:{stable_id}"


def build_selection_vector_view(
    sources: Sequence[VectorSet],
    representation: str,
    stable_ids: Sequence[str],
) -> VectorSet:
    """Join immutable source shards and extract one aligned representation view."""
    source_sets = tuple(sources)
    ordered_stable_ids = tuple(stable_ids)
    if not source_sets:
        raise ValueError("selection vector view requires at least one source")
    if (
        not ordered_stable_ids
        or len(set(ordered_stable_ids)) != len(ordered_stable_ids)
        or any(not isinstance(value, str) or not value for value in ordered_stable_ids)
    ):
        raise ValueError("selection vector view stable IDs must be nonempty and unique")
    expected_ids = tuple(
        selection_vector_id(representation, stable_id)
        for stable_id in ordered_stable_ids
    )
    if (
        len(source_sets) == 1
        and source_sets[0].vector_ids == expected_ids
        and source_sets[0].stable_ids == ordered_stable_ids
    ):
        return source_sets[0]

    locations: dict[str, tuple[int, int]] = {}
    source_hashes: list[str] = []
    vector_dimension: int | None = None
    for source_index, source in enumerate(source_sets):
        if not isinstance(source, VectorSet):
            raise TypeError("selection vector sources must be VectorSet objects")
        source_hashes.append(
            hashlib.sha256(canonical_json_bytes(source.manifest)).hexdigest()
        )
        vectors = np.asarray(source.vectors)
        if vectors.ndim != 2 or vectors.shape[0] != len(source.vector_ids):
            raise ValueError("selection vector source shape mismatch")
        if vector_dimension is None:
            vector_dimension = vectors.shape[1]
        elif vectors.shape[1] != vector_dimension:
            raise ValueError("selection vector source dimensions differ")
        if len(source.vector_ids) != len(source.stable_ids):
            raise ValueError("selection vector source ID arrays differ in length")
        for row_index, vector_id in enumerate(source.vector_ids):
            if vector_id in locations:
                raise ValueError(f"duplicate vector ID across source shards: {vector_id}")
            locations[vector_id] = (source_index, row_index)
    missing = [vector_id for vector_id in expected_ids if vector_id not in locations]
    if missing:
        raise ValueError(
            f"source vector sets lack exact {representation} candidate coverage"
        )
    selected_locations = [locations[vector_id] for vector_id in expected_ids]
    for stable_id, (source_index, row_index) in zip(
        ordered_stable_ids, selected_locations
    ):
        if source_sets[source_index].stable_ids[row_index] != stable_id:
            raise ValueError("selection vector source stable ID/vector ID mismatch")

    vectors = np.ascontiguousarray(
        np.stack(
            [
                source_sets[source_index].vectors[row_index]
                for source_index, row_index in selected_locations
            ]
        ),
        dtype=np.float32,
    )

    def optional(field: str) -> np.ndarray | None:
        selected_values = []
        presence = []
        for source_index, row_index in selected_locations:
            value = getattr(source_sets[source_index], field)
            presence.append(value is not None)
            if value is not None:
                selected_values.append(value[row_index])
        if not any(presence):
            return None
        if not all(presence):
            raise ValueError(
                f"selection vector sources disagree on optional field {field}"
            )
        return np.ascontiguousarray(np.asarray(selected_values))

    manifest = {
        "schema_version": 1,
        "artifact_type": "opd_proxy_selection_vector_view",
        "representation": representation,
        "vector_count": len(expected_ids),
        "vector_dimension": vectors.shape[1],
        "vector_ids_sha256": sha256_id_lines(expected_ids),
        "stable_ids_sha256": sha256_id_lines(ordered_stable_ids),
        "source_vector_manifest_sha256": source_hashes,
    }
    return VectorSet(
        vector_ids=expected_ids,
        stable_ids=ordered_stable_ids,
        vectors=vectors,
        full_gradient_norm=optional("full_gradient_norm"),
        projected_gradient_norm=optional("projected_gradient_norm"),
        valid_token_count=optional("valid_token_count"),
        response_length=optional("response_length"),
        sampled_reverse_kl=optional("sampled_reverse_kl"),
        opd_signal_rms=optional("opd_signal_rms"),
        verifier_correct_count=optional("verifier_correct_count"),
        verifier_total=optional("verifier_total"),
        manifest=manifest,
    )


def _read_stage_inputs(stage_directory: Path) -> tuple[dict[str, object], list[dict]]:
    root = Path(stage_directory)
    manifest = _load_canonical_json(root / "manifest.json", "stage manifest")
    if manifest.get("artifact_type") != "opd_proxy_stage":
        raise ValueError("selector CLI requires an OPD proxy stage manifest")
    sample_name = manifest.get("sample_manifest")
    if not isinstance(sample_name, str) or Path(sample_name).name != sample_name:
        raise ValueError("stage sample manifest path is invalid")
    sample_path = root / sample_name
    if sha256_file(sample_path) != manifest.get("sample_manifest_sha256"):
        raise ValueError("stage sample manifest SHA mismatch")
    rows: list[dict] = []
    for line_number, line in enumerate(sample_path.read_bytes().splitlines(keepends=True), 1):
        try:
            row = json.loads(line)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid stage sample row {line_number}: {error}") from error
        if not isinstance(row, dict) or line != canonical_json_bytes(row):
            raise ValueError(f"stage sample row {line_number} is not canonical")
        rows.append(row)
    return manifest, rows


def _parse_vector_specs(values: Sequence[str]) -> dict[str, list[Path]]:
    specs: dict[str, list[Path]] = {}
    for value in values:
        name, separator, path = value.rpartition("=")
        if separator != "=" or name not in EXPECTED_REPRESENTATIONS or not path:
            raise ValueError("--vector must be REPRESENTATION=VECTOR_DIRECTORY")
        specs.setdefault(name, []).append(Path(path))
    if set(specs) != set(EXPECTED_REPRESENTATIONS):
        raise ValueError("--vector specifications must cover all representations")
    return specs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--reference-repo", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument(
        "--vector",
        action="append",
        default=[],
        metavar="REPRESENTATION=PATH",
        help="repeat for every representation/source shard; paths may be shared",
    )
    args = parser.parse_args(argv)
    stage_manifest, rows = _read_stage_inputs(args.stage_directory)
    stage = int(stage_manifest["stage"])
    candidate_count = int(stage_manifest["candidate_count"])
    candidate_rows = rows[:candidate_count]
    candidate_ids = [str(row["stable_id"]) for row in candidate_rows]
    specs = _parse_vector_specs(args.vector)
    source_cache: dict[Path, VectorSet] = {}
    vectors: dict[str, VectorSet] = {}
    for name in EXPECTED_REPRESENTATIONS:
        sources: list[VectorSet] = []
        for declared_path in specs[name]:
            source_path = declared_path.resolve()
            if source_path not in source_cache:
                source_cache[source_path] = load_vector_set(source_path)
            sources.append(source_cache[source_path])
        vectors[name] = build_selection_vector_view(sources, name, candidate_ids)
    vector_hashes = {
        name: hashlib.sha256(canonical_json_bytes(vector.manifest)).hexdigest()
        for name, vector in vectors.items()
    }
    source_snapshot = _load_canonical_json(
        args.source_snapshot.resolve(strict=True), "experiment source snapshot"
    )
    source_files = source_snapshot.get("files")
    if not isinstance(source_files, list) or not source_files:
        raise ValueError("experiment source snapshot has no source files")
    source_hash = hashlib.sha256(canonical_json_bytes(source_files)).hexdigest()
    if source_snapshot.get("manifest_sha256") != source_hash:
        raise ValueError("experiment source snapshot logical hash mismatch")
    parent_hashes = {
        "source_snapshot_sha256": source_hash,
        "stage_manifest_sha256": sha256_file(
            Path(args.stage_directory) / "manifest.json"
        ),
    }
    generate_random_schedules(
        candidate_rows,
        selected_size=int(stage_manifest["selected_size"]),
        draws=int(stage_manifest["null_draws"]),
        output_directory=args.output_directory,
        parent_hashes=parent_hashes,
        stage=stage,
    )
    run_selection(
        vectors,
        stage=stage,
        candidate_rows=candidate_rows,
        reference_repo=args.reference_repo,
        output_directory=args.output_directory,
        parent_hashes=parent_hashes,
        expected_vector_manifest_hashes=vector_hashes,
    )
    return 0


__all__ = [
    "DIAGNOSTIC_RATIO",
    "EXPECTED_REPRESENTATIONS",
    "KMEANS_SEEDS",
    "PRIMARY_RATIO",
    "ROUND_ROBIN_SEED",
    "RandomSchedules",
    "SelectionBundle",
    "build_length_quartiles",
    "build_selection_vector_view",
    "generate_random_schedules",
    "largest_remainder_allocation",
    "load_random_schedules",
    "load_selection_bundle",
    "run_selection",
    "selection_vector_id",
]


if __name__ == "__main__":
    raise SystemExit(main())
