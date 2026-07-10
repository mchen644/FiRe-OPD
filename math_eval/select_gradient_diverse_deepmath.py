"""Select an exact DeepMath coreset from pinned Prismatic projected gradients."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import inspect
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from math_eval.deepmath_gradient_diversity import (
    balanced_round_robin,
    cluster_size_summary,
    sha256_file,
    validate_gradient_matrix,
)


REFERENCE_COMMIT = "d9484cd3b5991030b901ac4a3a9e2472dbfac2ad"
REFERENCE_TREE = "a0079d8c5e15cb18bb4790f99c43cc19bb9ecd50"
PROJECTION_DIM = 1024
PROJECTED_DTYPE = torch.float32
CLUSTER_ITERATIONS = 20
CLUSTER_SEEDS = (42, 43)
CLUSTER_RATIOS = (0.10, 0.01)
PRIMARY_RATIO = 0.10
PRIMARY_SEED = 42
SELECTION_SEED = 42
TARGET_SIZE = 12_800
SOURCE_ROW_COUNT = 57_046
ELIGIBLE_ROW_COUNT = 57_045
EXPECTED_EXCLUDED_IDS = ("deepmath-level6-038794",)
DATASET_NAME = "zwhe99/DeepMath-103K"
DATASET_REVISION = "5cf055d1fe3d7a2eb19719ac020211469736ae44"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
GRADIENT_MANIFEST_NAME = "gradient.manifest.json"

_CHUNK_PATTERN = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9_-]*)\.(0|[1-9][0-9]*)\.(txt|safetensors)$"
)


def _reject_duplicate_json_pairs(pairs):
    value: dict = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_nonfinite_json_constant(value: str):
    raise ValueError(f"non-finite JSON constant: {value}")


def _strict_json_loads(payload: str, description: str):
    try:
        return json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_json_pairs,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid {description}: {error}") from error


def _read_sidecar(path: Path) -> list[str]:
    ids: list[str] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(f"blank gradient sidecar line {line_number}")
                row = _strict_json_loads(
                    line, f"gradient sidecar row {line_number} in {path.name}"
                )
                sample_id = row.get("id") if isinstance(row, Mapping) else None
                if not isinstance(sample_id, str) or not sample_id:
                    raise ValueError(
                        f"invalid gradient sidecar ID at line {line_number}"
                    )
                ids.append(sample_id)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid gradient sidecar {path}: {error}") from error
    if not ids:
        raise ValueError(f"gradient sidecar is empty: {path}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate gradient IDs in sidecar {path.name}")
    return ids


def _discover_chunk_pairs(
    directory: Path, expected_prefix: str | None
) -> list[tuple[int, Path, Path]]:
    files_by_start: dict[int, dict[str, Path]] = {}
    prefixes: set[str] = set()
    for path in Path(directory).iterdir():
        if not path.is_file():
            continue
        looks_like_chunk = ".txt" in path.name or ".safetensors" in path.name
        if not looks_like_chunk:
            continue
        match = _CHUNK_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"malformed gradient artifact: {path.name}")
        prefix, start_text, kind = match.groups()
        if expected_prefix is not None and prefix != expected_prefix:
            raise ValueError(
                f"unexpected gradient prefix {prefix!r}; expected {expected_prefix!r}"
            )
        prefixes.add(prefix)
        start = int(start_text)
        if kind in files_by_start.setdefault(start, {}):
            raise ValueError(f"duplicate gradient chunk artifact: {path.name}")
        files_by_start[start][kind] = path

    if len(prefixes) > 1:
        raise ValueError(f"multiple gradient prefixes found: {sorted(prefixes)}")

    pairs: list[tuple[int, Path, Path]] = []
    for start in sorted(files_by_start):
        files = files_by_start[start]
        if set(files) != {"txt", "safetensors"}:
            raise ValueError(
                f"gradient chunk at {start} must have paired .txt and .safetensors files"
            )
        pairs.append((start, files["txt"], files["safetensors"]))
    return pairs


def load_projected_gradients(
    directory: Path,
    expected_ids: Sequence[str],
    *,
    prefix: str | None = None,
) -> tuple[list[str], torch.Tensor]:
    """Load exact official chunk coverage into eligibility-report order on CPU."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"gradient directory does not exist: {directory}")
    ordered_ids = list(expected_ids)
    if not ordered_ids or any(
        not isinstance(sample_id, str) or not sample_id for sample_id in ordered_ids
    ):
        raise ValueError("expected gradient IDs must be nonempty strings")
    if len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError("expected gradient IDs must be unique")
    if prefix is not None and not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*", prefix
    ):
        raise ValueError("expected gradient prefix is invalid")

    gradients = torch.empty(
        (len(ordered_ids), PROJECTION_DIM), dtype=PROJECTED_DTYPE, device="cpu"
    )
    cursor = 0
    seen_ids: set[str] = set()
    for start, sidecar_path, tensor_path in _discover_chunk_pairs(directory, prefix):
        if start != cursor:
            raise ValueError(
                f"gradient chunk coverage gap or overlap: expected start {cursor}, "
                f"found {start}"
            )
        sidecar_ids = _read_sidecar(sidecar_path)
        end = start + len(sidecar_ids)
        if end > len(ordered_ids):
            raise ValueError("gradient chunk extends beyond expected eligible IDs")
        if sidecar_ids != ordered_ids[start:end]:
            raise ValueError(
                "gradient sidecar IDs do not match exact eligibility-report order"
            )
        duplicates = seen_ids.intersection(sidecar_ids)
        if duplicates:
            raise ValueError(f"duplicate gradient IDs across chunks: {sorted(duplicates)}")

        try:
            tensors = load_file(tensor_path, device="cpu")
        except Exception as error:
            raise ValueError(
                f"unreadable gradient safetensors {tensor_path}: {error}"
            ) from error
        if len(tensors) != len(sidecar_ids) or set(tensors) != set(sidecar_ids):
            raise ValueError(
                f"gradient tensor keys do not match paired sidecar {sidecar_path.name}"
            )
        for offset, sample_id in enumerate(sidecar_ids):
            tensor = tensors[sample_id]
            if tensor.dtype != PROJECTED_DTYPE:
                raise ValueError(f"gradient for {sample_id} must have float32 dtype")
            if tuple(tensor.shape) != (PROJECTION_DIM,):
                raise ValueError(
                    f"gradient for {sample_id} has invalid shape {tuple(tensor.shape)}"
                )
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"gradient for {sample_id} must contain finite values")
            if int(torch.count_nonzero(tensor)) == 0:
                raise ValueError(f"gradient for {sample_id} has zero norm")
            gradients[start + offset].copy_(tensor)
        seen_ids.update(sidecar_ids)
        cursor = end

    if cursor != len(ordered_ids):
        raise ValueError(
            f"missing gradient coverage: expected {len(ordered_ids)} rows, got {cursor}"
        )
    validate_gradient_matrix(ordered_ids, gradients, ordered_ids)
    return ordered_ids, gradients


def load_globally_validated_gradients(
    directory: Path,
    expected_ids: Sequence[str],
    gradient_manifest: Mapping,
    *,
    coverage_validator=None,
) -> tuple[list[str], torch.Tensor, dict]:
    """Run the collector's exact all-shard gate before loading one prefix."""
    prefix = gradient_manifest.get("prefix")
    if not isinstance(prefix, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*", prefix
    ):
        raise ValueError("gradient manifest prefix is invalid")
    num_shards = gradient_manifest.get("num_shards")
    if (
        isinstance(num_shards, bool)
        or not isinstance(num_shards, int)
        or num_shards <= 0
    ):
        raise ValueError("gradient manifest num_shards must be positive")
    if coverage_validator is None:
        from math_eval.collect_prismatic_gradients import (
            validate_global_gradient_coverage,
        )

        coverage_validator = validate_global_gradient_coverage
    coverage = coverage_validator(
        expected_ids, Path(directory), prefix, num_shards
    )
    expected_coverage = {
        "row_count": len(expected_ids),
        "shard_count": num_shards,
    }
    if not isinstance(coverage, Mapping) or any(
        coverage.get(field) != expected
        for field, expected in expected_coverage.items()
    ):
        raise RuntimeError("global gradient coverage summary is inconsistent")
    chunk_count = coverage.get("chunk_count")
    if isinstance(chunk_count, bool) or not isinstance(chunk_count, int) or chunk_count <= 0:
        raise RuntimeError("global gradient coverage must contain positive chunk_count")
    ids, gradients = load_projected_gradients(
        directory, expected_ids, prefix=prefix
    )
    return ids, gradients, dict(coverage)


def _validate_source_indices(source: pa.Table, indices: Sequence[int]) -> list[int]:
    selected_indices = list(indices)
    if any(isinstance(index, bool) or not isinstance(index, int) for index in selected_indices):
        raise ValueError("selected source indices must be integers")
    if len(set(selected_indices)) != len(selected_indices):
        raise ValueError("selected source indices must be unique")
    if any(index < 0 or index >= source.num_rows for index in selected_indices):
        raise ValueError("selected source index is out of range")
    return selected_indices


def _validate_unique_prompts(selected: pa.Table) -> None:
    if "prompt" not in selected.schema.names:
        raise ValueError("source parquet is missing prompt column")
    prompt_keys = [
        json.dumps(prompt, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for prompt in selected.column("prompt").to_pylist()
    ]
    if len(set(prompt_keys)) != len(prompt_keys):
        raise ValueError("selected rows must have unique prompts")


def _fsync_file(path: Path) -> None:
    with Path(path).open("rb") as handle:
        os.fsync(handle.fileno())


def write_selected_parquet(
    source: pa.Table, selected_source_indices: Sequence[int], output: Path
) -> None:
    """Write selected source rows atomically without altering Arrow schema or order."""
    indices = _validate_source_indices(source, selected_source_indices)
    selected = source.take(pa.array(indices, type=pa.int64()))
    if not selected.schema.equals(source.schema, check_metadata=True):
        raise ValueError("selected Arrow schema differs from source schema")
    _validate_unique_prompts(selected)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp.parquet"
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        pq.write_table(selected, temporary_path)
        _fsync_file(temporary_path)
        readback = pq.read_table(temporary_path)
        if not readback.schema.equals(source.schema, check_metadata=True):
            raise ValueError("temporary parquet schema differs from source schema")
        if not readback.equals(selected, check_metadata=True):
            raise ValueError("temporary parquet row content or order mismatch")
        temporary_path.replace(output)
    finally:
        temporary_path.unlink(missing_ok=True)


def _git_output(repository: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"invalid reference repository {repository}: {error}") from error
    return completed.stdout.strip()


def _import_exact_class(module_path: Path, class_name: str, module_tag: str):
    if not module_path.is_file():
        raise ValueError(f"official reference module does not exist: {module_path}")
    module_name = f"_fire_opd_{module_tag}_{REFERENCE_COMMIT}_{abs(hash(str(module_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import official reference module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    imported_path = Path(getattr(module, "__file__", "")).resolve()
    if imported_path != module_path.resolve():
        raise RuntimeError(
            f"official module origin mismatch: expected {module_path}, got {imported_path}"
        )
    selected_class = getattr(module, class_name, None)
    if not isinstance(selected_class, type):
        raise ValueError(f"official class {class_name} is missing from {module_path}")
    if Path(inspect.getfile(selected_class)).resolve() != module_path.resolve():
        raise RuntimeError(f"official class {class_name} has an unexpected origin")
    return selected_class


def load_official_reference_classes(reference_repo: Path):
    """Verify and import ClusterManager and Vendi from the exact pinned files."""
    repository = Path(reference_repo).resolve()
    actual_commit = _git_output(repository, "rev-parse", "HEAD")
    actual_tree = _git_output(repository, "rev-parse", "HEAD^{tree}")
    status = _git_output(repository, "status", "--porcelain=v1", "--untracked-files=all")
    if actual_commit != REFERENCE_COMMIT:
        raise ValueError(
            f"reference commit mismatch: expected {REFERENCE_COMMIT}, got {actual_commit}"
        )
    if actual_tree != REFERENCE_TREE:
        raise ValueError(
            f"reference tree mismatch: expected {REFERENCE_TREE}, got {actual_tree}"
        )
    if status:
        raise ValueError("reference repository must have a clean working tree")

    cluster_path = (
        repository
        / "prismatic-synthesis"
        / "cluster_modules"
        / "cluster_manager.py"
    )
    vendi_path = repository / "g-vendi" / "gradient_vendi.py"
    cluster_manager = _import_exact_class(
        cluster_path.resolve(), "ClusterManager", "cluster_manager"
    )
    vendi = _import_exact_class(vendi_path.resolve(), "Vendi", "vendi")
    return cluster_manager, vendi, {
        "reference_repo": str(repository),
        "reference_commit": actual_commit,
        "reference_tree": actual_tree,
        "cluster_manager_module": str(cluster_path.resolve()),
        "vendi_module": str(vendi_path.resolve()),
    }


def cluster_official(
    gradients: torch.Tensor,
    ratio: float,
    iterations: int,
    seed: int,
    reference_repo: Path,
    *,
    device: str | None = None,
    cluster_manager_class=None,
) -> np.ndarray:
    """Run seeded-permutation cosine K-means through official ClusterManager."""
    if not 0 < ratio <= 1:
        raise ValueError("clustering ratio must be in (0, 1]")
    if iterations != CLUSTER_ITERATIONS:
        raise ValueError(f"official clustering requires exactly {CLUSTER_ITERATIONS} iterations")
    if gradients.ndim != 2 or gradients.shape[0] < 2:
        raise ValueError("clustering requires a rank-2 matrix with at least two rows")
    if not bool(torch.isfinite(gradients).all()):
        raise ValueError("clustering gradients must be finite")
    if bool((torch.linalg.vector_norm(gradients.float(), dim=1) == 0).any()):
        raise ValueError("clustering gradients must have non-zero rows")

    if cluster_manager_class is None and device is None:
        raise ValueError("official clustering requires explicit device='cuda:0'")
    if device is not None:
        if device != "cuda:0":
            raise ValueError("official clustering requires explicit device='cuda:0'")
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                "official clustering requires exactly one visible CUDA device"
            )
    if cluster_manager_class is None:
        cluster_manager_class, _, _ = load_official_reference_classes(reference_repo)

    row_count = gradients.shape[0]
    cluster_count = max(2, min(row_count, math.floor(ratio * row_count)))
    working_gradients = gradients if device is None else gradients.to(torch.device(device))
    normalized = F.normalize(working_gradients.float(), dim=1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation_cpu = torch.randperm(row_count, generator=generator)
    permutation = permutation_cpu.to(normalized.device)
    permuted = normalized.index_select(0, permutation)
    labels, _ = cluster_manager_class.cluster_kmeans(
        permuted,
        cluster_count,
        iterations,
        use_tqdm=row_count > 100_000,
    )
    if not isinstance(labels, torch.Tensor) or tuple(labels.shape) != (row_count,):
        raise ValueError("official clustering returned invalid label shape")
    labels = labels.to(dtype=torch.int64)
    if bool(((labels < 0) | (labels >= cluster_count)).any()):
        raise ValueError("official clustering returned labels outside [0, K)")
    restored = torch.empty_like(labels)
    restored[permutation] = labels
    return restored.cpu().numpy()


def _selected_metadata(rows: Sequence[Mapping], selected: Sequence[int]) -> dict:
    topics = [str(rows[index].get("topic")) for index in selected]
    topic_counts: dict[str, int] = {}
    for topic in topics:
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    topic_probabilities = [count / len(topics) for count in topic_counts.values()]
    topic_entropy = -sum(
        probability * math.log(probability) for probability in topic_probabilities
    )

    difficulty_counts: dict[str, int] = {}
    for index in selected:
        difficulty = str(rows[index].get("difficulty"))
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
    return {
        "selected_topic_coverage": len(topic_counts),
        "selected_topic_entropy": topic_entropy,
        "topic_distribution": dict(sorted(topic_counts.items())),
        "difficulty_distribution": dict(sorted(difficulty_counts.items())),
    }


def _selection_vendi(
    gradients: torch.Tensor, selected: Sequence[int], vendi_class
) -> float:
    selected_indices = torch.as_tensor(
        list(selected), dtype=torch.int64, device=gradients.device
    )
    normalized = F.normalize(
        gradients.index_select(0, selected_indices).float(), dim=1
    )
    covariance = normalized.T @ normalized
    return float(vendi_class.compute_vendi_score(covariance, n=len(selected)))


def compute_selection_diagnostics(
    gradients: torch.Tensor,
    label_runs: Mapping[tuple[float, int], np.ndarray],
    prepared_rows: Sequence[Mapping],
    *,
    target_size: int,
    selection_seed: int,
    vendi_class,
) -> dict:
    """Compute cluster stability and selected-set diagnostics for all runs."""
    from sklearn.metrics import adjusted_rand_score

    row_count = gradients.shape[0]
    if len(prepared_rows) != row_count:
        raise ValueError("prepared metadata and gradients must have equal row counts")
    if target_size <= 0 or target_size > row_count:
        raise ValueError("target_size must be in [1, row_count]")

    ratios = sorted({ratio for ratio, _ in label_runs}, reverse=True)
    diagnostics: dict = {
        "row_count": row_count,
        "target_size": target_size,
        "selection_seed": selection_seed,
        "ratios": {},
    }
    for ratio in ratios:
        seeds = sorted(seed for run_ratio, seed in label_runs if run_ratio == ratio)
        if seeds != list(CLUSTER_SEEDS):
            raise ValueError(
                f"ratio {ratio} requires exact clustering seeds {CLUSTER_SEEDS}"
            )
        requested_clusters = max(2, min(row_count, math.floor(ratio * row_count)))
        run_diagnostics: dict[str, dict] = {}
        selections: dict[int, list[int]] = {}
        labels_for_seed: dict[int, np.ndarray] = {}
        for seed in seeds:
            labels = np.asarray(label_runs[(ratio, seed)], dtype=np.int64)
            if labels.shape != (row_count,):
                raise ValueError("cluster labels must have one value per gradient")
            if np.any(labels < 0) or np.any(labels >= requested_clusters):
                raise ValueError("cluster labels must be in [0, K)")
            selected = balanced_round_robin(
                labels.tolist(), target_size=target_size, seed=selection_seed
            )
            selections[seed] = selected
            labels_for_seed[seed] = labels
            run_diagnostics[str(seed)] = {
                **cluster_size_summary(labels.tolist(), requested_clusters),
                "selected_g_vendi": _selection_vendi(
                    gradients, selected, vendi_class
                ),
                **_selected_metadata(prepared_rows, selected),
            }

        selected_42 = set(selections[CLUSTER_SEEDS[0]])
        selected_43 = set(selections[CLUSTER_SEEDS[1]])
        diagnostics["ratios"][format(ratio, "g")] = {
            "requested_clusters": requested_clusters,
            "adjusted_rand_index": float(
                adjusted_rand_score(
                    labels_for_seed[CLUSTER_SEEDS[0]],
                    labels_for_seed[CLUSTER_SEEDS[1]],
                )
            ),
            "selected_set_jaccard": len(selected_42.intersection(selected_43))
            / len(selected_42.union(selected_43)),
            "runs": run_diagnostics,
        }
    return diagnostics


def _write_json(path: Path, value: object) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl(path: Path, rows: Sequence[Mapping]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())


def _temporary_sibling(path: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=f".tmp{path.suffix}"
    )
    os.close(descriptor)
    temporary = Path(name)
    temporary.unlink()
    return temporary


@contextmanager
def _selection_locks(paths: Sequence[Path]):
    descriptors: list[int] = []
    try:
        for target in sorted({Path(path).resolve() for path in paths}, key=os.fspath):
            lock_path = target.with_name(f".{target.name}.lock")
            descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
            except BaseException:
                os.close(descriptor)
                raise
            descriptors.append(descriptor)
        yield
    finally:
        for descriptor in reversed(descriptors):
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


def _validate_existing_artifacts(
    *,
    source: pa.Table,
    selected_rows: Sequence[Mapping],
    diagnostics: Mapping,
    base_manifest: Mapping,
    output_parquet: Path,
    selected_ids_path: Path,
    diagnostics_path: Path,
    manifest_path: Path,
) -> dict:
    try:
        manifest = _strict_json_loads(
            manifest_path.read_text(encoding="utf-8"), "selection manifest"
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid existing selection manifest: {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError("existing selection manifest must be a JSON object")
    for field, expected in base_manifest.items():
        if manifest.get(field) != expected:
            raise ValueError(f"selection manifest {field} mismatch")
    expected_paths = {
        "output_parquet": output_parquet,
        "selected_ids": selected_ids_path,
        "diagnostics": diagnostics_path,
    }
    for field, path in expected_paths.items():
        if manifest.get(field) != str(path.resolve()):
            raise ValueError(f"selection manifest {field} path mismatch")
        if manifest.get(f"{field}_sha256") != sha256_file(path):
            raise ValueError(f"selection artifact hash mismatch for {field}")
    if manifest.get("selected_row_count") != len(selected_rows):
        raise ValueError("selection manifest selected_row_count mismatch")

    expected_indices = [int(row["source_row_index"]) for row in selected_rows]
    selected = pq.read_table(output_parquet)
    expected_table = source.take(pa.array(expected_indices, type=pa.int64()))
    if not selected.equals(expected_table, check_metadata=True):
        raise ValueError("existing selected parquet content mismatch")
    try:
        stored_diagnostics = _strict_json_loads(
            diagnostics_path.read_text(encoding="utf-8"),
            "selection diagnostics",
        )
        stored_rows = [
            _strict_json_loads(line, f"selected IDs row {line_number}")
            for line_number, line in enumerate(
                selected_ids_path.read_text(encoding="utf-8").splitlines(),
                start=1,
            )
        ]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid existing selection artifacts: {error}") from error
    if stored_diagnostics != diagnostics or stored_rows != list(selected_rows):
        raise ValueError("existing selection artifact content mismatch")
    return manifest


def publish_selection_artifacts(
    *,
    source: pa.Table,
    selected_rows: Sequence[Mapping],
    diagnostics: Mapping,
    base_manifest: Mapping,
    output_parquet: Path,
    selected_ids_path: Path,
    diagnostics_path: Path,
    manifest_path: Path,
) -> dict:
    """Publish or exactly validate the lock-protected four-file artifact set."""
    paths = [
        Path(output_parquet).resolve(),
        Path(selected_ids_path).resolve(),
        Path(diagnostics_path).resolve(),
        Path(manifest_path).resolve(),
    ]
    if len(set(paths)) != len(paths):
        raise ValueError("selection artifact paths must be distinct")
    output_parquet, selected_ids_path, diagnostics_path, manifest_path = paths
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    normalized_rows = [dict(row) for row in selected_rows]
    ids = [row.get("id") for row in normalized_rows]
    indices = [row.get("source_row_index") for row in normalized_rows]
    if any(not isinstance(sample_id, str) or not sample_id for sample_id in ids):
        raise ValueError("selected rows require nonempty IDs")
    if len(set(ids)) != len(ids):
        raise ValueError("selected IDs must be unique")
    validated_indices = _validate_source_indices(source, indices)

    with _selection_locks(paths):
        existence = [path.exists() for path in paths]
        if any(existence) and not all(existence):
            raise ValueError("partial selection artifact set exists")
        if all(existence):
            return _validate_existing_artifacts(
                source=source,
                selected_rows=normalized_rows,
                diagnostics=diagnostics,
                base_manifest=base_manifest,
                output_parquet=output_parquet,
                selected_ids_path=selected_ids_path,
                diagnostics_path=diagnostics_path,
                manifest_path=manifest_path,
            )

        temporary_paths = [_temporary_sibling(path) for path in paths]
        temporary_parquet, temporary_ids, temporary_diagnostics, temporary_manifest = (
            temporary_paths
        )
        try:
            write_selected_parquet(source, validated_indices, temporary_parquet)
            _write_jsonl(temporary_ids, normalized_rows)
            _write_json(temporary_diagnostics, diagnostics)
            manifest = {
                **dict(base_manifest),
                "selected_row_count": len(normalized_rows),
                "output_parquet": str(output_parquet),
                "output_parquet_sha256": sha256_file(temporary_parquet),
                "selected_ids": str(selected_ids_path),
                "selected_ids_sha256": sha256_file(temporary_ids),
                "diagnostics": str(diagnostics_path),
                "diagnostics_sha256": sha256_file(temporary_diagnostics),
            }
            _write_json(temporary_manifest, manifest)
            if any(path.exists() for path in paths):
                raise FileExistsError(
                    "selection artifacts appeared while lock was held; refusing overwrite"
                )
            temporary_parquet.replace(output_parquet)
            temporary_ids.replace(selected_ids_path)
            temporary_diagnostics.replace(diagnostics_path)
            temporary_manifest.replace(manifest_path)
            return manifest
        finally:
            for temporary_path in temporary_paths:
                temporary_path.unlink(missing_ok=True)


def _read_json_object(path: Path, description: str) -> dict:
    try:
        value = _strict_json_loads(
            Path(path).read_text(encoding="utf-8"), description
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"invalid {description} {path}: expected a JSON object")
    return value


def load_validated_gradient_manifest(
    gradient_manifest_path: Path,
    *,
    prepared_manifest_path: Path,
    prepared_manifest: Mapping,
    eligibility_report_path: Path,
    eligibility_report: Mapping,
    reference_repo: Path,
    expected_prefix: str | None = None,
    expected_num_shards: int | None = None,
    manifest_validator=None,
) -> dict:
    """Strict-read then delegate the complete contract to the collector."""
    gradient_manifest_path = Path(gradient_manifest_path).resolve()
    prepared_manifest_path = Path(prepared_manifest_path).resolve()
    eligibility_report_path = Path(eligibility_report_path).resolve()
    reference_repo = Path(reference_repo).resolve()
    strict_manifest = _read_json_object(
        gradient_manifest_path, "gradient manifest"
    )
    if manifest_validator is None:
        from math_eval.collect_prismatic_gradients import (
            load_validated_gradient_manifest as collector_manifest_validator,
        )

        manifest_validator = collector_manifest_validator
    validated = manifest_validator(
        gradient_manifest_path,
        prepared_manifest_path=prepared_manifest_path,
        prepared_manifest=prepared_manifest,
        eligibility_report_path=eligibility_report_path,
        eligibility_report=eligibility_report,
        reference_repo=reference_repo,
        expected_prefix=expected_prefix,
        expected_num_shards=expected_num_shards,
    )
    if strict_manifest != validated:
        raise ValueError(
            "gradient manifest changed or normalized during exact validation"
        )
    return dict(validated)


def build_selection_gradient_provenance(
    gradient_manifest_path: Path, gradient_manifest: Mapping
) -> dict:
    """Embed the complete validated gradient contract plus lookup fields."""
    path = Path(gradient_manifest_path).resolve()
    return {
        "gradient_manifest": str(path),
        "gradient_manifest_sha256": sha256_file(path),
        "gradient_prefix": gradient_manifest["prefix"],
        "gradient_num_shards": gradient_manifest["num_shards"],
        "gradient_runtime": {
            "trl_version": gradient_manifest["trl_version"],
            "package_versions": dict(gradient_manifest["package_versions"]),
        },
        "gradient_provenance": dict(gradient_manifest),
    }


def _selected_id_sequence_sha256(rows: Sequence[Mapping]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["id"]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_selection_output_isolation(
    outputs: Sequence[Path], inputs: Sequence[Path], gradient_dir: Path
) -> None:
    resolved_outputs = [Path(path).resolve() for path in outputs]
    resolved_inputs = {Path(path).resolve() for path in inputs}
    if len(set(resolved_outputs)) != len(resolved_outputs):
        raise ValueError("selection output paths must be distinct")
    gradient_root = Path(gradient_dir).resolve()
    for output in resolved_outputs:
        if output in resolved_inputs:
            raise ValueError(f"selection output overlaps an input artifact: {output}")
        if output == gradient_root or gradient_root in output.parents:
            raise ValueError("selection outputs must be outside the gradient directory")


def _validate_eligible_rows(
    prepared_rows: Sequence[Mapping], eligible_rows: Sequence[Mapping]
) -> None:
    prepared_by_id = {row.get("id"): row for row in prepared_rows}
    if len(prepared_by_id) != len(prepared_rows):
        raise ValueError("prepared IDs must be unique")
    seen: set[str] = set()
    previous_source_index = -1
    for row in eligible_rows:
        sample_id = row.get("id")
        source_index = row.get("source_row_index")
        if sample_id in seen or sample_id not in prepared_by_id:
            raise ValueError("eligible rows contain duplicate or unknown stable IDs")
        if (
            isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or source_index <= previous_source_index
        ):
            raise ValueError("eligible source indices must preserve strict source order")
        if prepared_by_id[sample_id] != row:
            raise ValueError("eligible row content differs from prepared pool")
        seen.add(sample_id)
        previous_source_index = source_index


def apply_production_eligibility_report(
    rows: Sequence[Mapping],
    prepared_manifest: Mapping,
    report_path: Path,
    *,
    prepared_manifest_path: Path,
    apply_report=None,
) -> tuple[list[dict], dict]:
    """Apply production semantics while binding the caller's manifest file."""
    if apply_report is None:
        from math_eval.build_gradient_eligibility import apply_eligibility_report

        apply_report = apply_eligibility_report
    return apply_report(
        rows,
        prepared_manifest,
        Path(report_path).resolve(),
        prepared_manifest_path=Path(prepared_manifest_path).resolve(),
    )


def run_selection(
    *,
    source_parquet: Path,
    prepared_jsonl: Path,
    prepared_manifest_path: Path,
    eligibility_report_path: Path,
    gradient_dir: Path,
    reference_repo: Path,
    device: str,
    output_parquet: Path,
    selected_ids_path: Path,
    diagnostics_path: Path,
    manifest_path: Path,
    expected_source_count: int = SOURCE_ROW_COUNT,
    expected_eligible_count: int = ELIGIBLE_ROW_COUNT,
    target_size: int = TARGET_SIZE,
) -> dict:
    """Run the pinned four-clustering selection and publish its exact artifacts."""
    if expected_source_count != SOURCE_ROW_COUNT:
        raise ValueError(f"expected_source_count must be pinned to {SOURCE_ROW_COUNT}")
    if expected_eligible_count != ELIGIBLE_ROW_COUNT:
        raise ValueError(f"expected_eligible_count must be pinned to {ELIGIBLE_ROW_COUNT}")
    if target_size != TARGET_SIZE:
        raise ValueError(f"target_size must be pinned to {TARGET_SIZE}")
    if device != "cuda:0":
        raise ValueError("selection device must be explicit cuda:0")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "selection requires exactly one visible CUDA device; "
            f"found {torch.cuda.device_count()}"
        )

    source_parquet = Path(source_parquet).resolve()
    prepared_jsonl = Path(prepared_jsonl).resolve()
    prepared_manifest_path = Path(prepared_manifest_path).resolve()
    eligibility_report_path = Path(eligibility_report_path).resolve()
    gradient_dir = Path(gradient_dir).resolve()
    reference_repo = Path(reference_repo).resolve()
    output_paths = [
        Path(output_parquet),
        Path(selected_ids_path),
        Path(diagnostics_path),
        Path(manifest_path),
    ]
    gradient_manifest_path = gradient_dir / GRADIENT_MANIFEST_NAME
    _validate_selection_output_isolation(
        output_paths,
        [
            source_parquet,
            prepared_jsonl,
            prepared_manifest_path,
            eligibility_report_path,
            gradient_manifest_path,
        ],
        gradient_dir,
    )

    # Imported lazily so pure selector helpers remain usable in CPU test envs.
    from math_eval.collect_prismatic_gradients import load_prepared_pool

    prepared_rows, prepared_manifest = load_prepared_pool(
        prepared_jsonl, prepared_manifest_path
    )
    if _read_json_object(
        prepared_manifest_path, "prepared manifest"
    ) != prepared_manifest:
        raise ValueError("prepared manifest changed or normalized during validation")
    if len(prepared_rows) != expected_source_count:
        raise ValueError(
            f"prepared source count mismatch: expected {expected_source_count}, "
            f"got {len(prepared_rows)}"
        )
    manifest_source = Path(str(prepared_manifest["source_parquet"])).resolve()
    if manifest_source != source_parquet:
        raise ValueError("source parquet path differs from prepared manifest")
    source = pq.read_table(source_parquet)
    if source.num_rows != expected_source_count:
        raise ValueError("source parquet row count mismatch")

    eligible_rows, eligibility_report = apply_production_eligibility_report(
        prepared_rows,
        prepared_manifest,
        eligibility_report_path,
        prepared_manifest_path=prepared_manifest_path,
    )
    if _read_json_object(
        eligibility_report_path, "eligibility report"
    ) != eligibility_report:
        raise ValueError(
            "eligibility report changed or normalized during validation"
        )
    _validate_eligible_rows(prepared_rows, eligible_rows)
    if len(eligible_rows) != expected_eligible_count:
        raise ValueError(
            f"eligible row count mismatch: expected {expected_eligible_count}, "
            f"got {len(eligible_rows)}"
        )
    excluded_ids = tuple(
        row.get("id") for row in eligibility_report.get("excluded_rows", [])
    )
    if excluded_ids != EXPECTED_EXCLUDED_IDS:
        raise ValueError(
            f"excluded stable IDs mismatch: expected {EXPECTED_EXCLUDED_IDS}, "
            f"got {excluded_ids}"
        )

    gradient_manifest = load_validated_gradient_manifest(
        gradient_manifest_path,
        prepared_manifest_path=prepared_manifest_path,
        prepared_manifest=prepared_manifest,
        eligibility_report_path=eligibility_report_path,
        eligibility_report=eligibility_report,
        reference_repo=reference_repo,
    )
    expected_ids = [str(row["id"]) for row in eligible_rows]
    loaded_ids, gradients, gradient_coverage = load_globally_validated_gradients(
        gradient_dir, expected_ids, gradient_manifest
    )
    if loaded_ids != expected_ids:
        raise RuntimeError("gradient loader changed eligibility-report ID order")

    cluster_manager, vendi, reference_provenance = load_official_reference_classes(
        reference_repo
    )
    gradients_on_device = gradients.to(torch.device(device))
    label_runs: dict[tuple[float, int], np.ndarray] = {}
    for ratio in CLUSTER_RATIOS:
        for seed in CLUSTER_SEEDS:
            label_runs[(ratio, seed)] = cluster_official(
                gradients_on_device,
                ratio,
                CLUSTER_ITERATIONS,
                seed,
                reference_repo,
                device=device,
                cluster_manager_class=cluster_manager,
            )
    diagnostics = compute_selection_diagnostics(
        gradients_on_device,
        label_runs,
        eligible_rows,
        target_size=target_size,
        selection_seed=SELECTION_SEED,
        vendi_class=vendi,
    )
    diagnostics.update(
        {
            "primary_ratio": PRIMARY_RATIO,
            "primary_seed": PRIMARY_SEED,
            "cluster_iterations": CLUSTER_ITERATIONS,
            "cluster_seeds": list(CLUSTER_SEEDS),
        }
    )
    primary_positions = balanced_round_robin(
        label_runs[(PRIMARY_RATIO, PRIMARY_SEED)].tolist(),
        target_size=target_size,
        seed=SELECTION_SEED,
    )
    selected_rows = [
        {
            "id": eligible_rows[position]["id"],
            "eligible_position": position,
            "source_row_index": eligible_rows[position]["source_row_index"],
            "original_dataset_index": eligible_rows[position][
                "original_dataset_index"
            ],
        }
        for position in primary_positions
    ]
    if len(selected_rows) != target_size or len(
        {row["id"] for row in selected_rows}
    ) != target_size:
        raise RuntimeError("primary selection is not the exact unique target size")

    base_manifest = {
        "manifest_version": 1,
        "source_parquet": str(source_parquet),
        "source_sha256": prepared_manifest["source_sha256"],
        "source_row_count": len(prepared_rows),
        "prepared_jsonl": str(prepared_jsonl),
        "prepared_jsonl_sha256": prepared_manifest["prepared_jsonl_sha256"],
        "prepared_manifest": str(prepared_manifest_path),
        "prepared_manifest_sha256": sha256_file(prepared_manifest_path),
        "eligibility_report": str(eligibility_report_path),
        "eligibility_report_sha256": sha256_file(eligibility_report_path),
        "excluded_row_count": eligibility_report["excluded_row_count"],
        "eligible_row_count": eligibility_report["eligible_row_count"],
        "eligible_ids_sha256": eligibility_report["eligible_ids_sha256"],
        "gradient_directory": str(gradient_dir),
        "gradient_coverage": gradient_coverage,
        **build_selection_gradient_provenance(
            gradient_manifest_path, gradient_manifest
        ),
        "dataset_name": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        **reference_provenance,
        "clustering": {
            "ratios": list(CLUSTER_RATIOS),
            "seeds": list(CLUSTER_SEEDS),
            "iterations": CLUSTER_ITERATIONS,
            "distance": "cosine",
            "primary_ratio": PRIMARY_RATIO,
            "primary_seed": PRIMARY_SEED,
        },
        "selection": {
            "method": "balanced_round_robin",
            "seed": SELECTION_SEED,
            "target_size": target_size,
        },
        "selected_id_sequence_sha256": _selected_id_sequence_sha256(selected_rows),
        "diagnostic_results": diagnostics,
    }
    return publish_selection_artifacts(
        source=source,
        selected_rows=selected_rows,
        diagnostics=diagnostics,
        base_manifest=base_manifest,
        output_parquet=output_paths[0],
        selected_ids_path=output_paths[1],
        diagnostics_path=output_paths[2],
        manifest_path=output_paths[3],
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select the pinned gradient-diverse DeepMath 12,800-row coreset"
    )
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--prepared-jsonl", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--eligibility-report", type=Path, required=True)
    parser.add_argument("--gradient-dir", type=Path, required=True)
    parser.add_argument("--reference-repo", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--selected-ids", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-source-count", type=int, default=SOURCE_ROW_COUNT)
    parser.add_argument(
        "--expected-eligible-count", type=int, default=ELIGIBLE_ROW_COUNT
    )
    parser.add_argument("--target-size", type=int, default=TARGET_SIZE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = run_selection(
        source_parquet=args.source_parquet,
        prepared_jsonl=args.prepared_jsonl,
        prepared_manifest_path=args.prepared_manifest,
        eligibility_report_path=args.eligibility_report,
        gradient_dir=args.gradient_dir,
        reference_repo=args.reference_repo,
        device=args.device,
        output_parquet=args.output_parquet,
        selected_ids_path=args.selected_ids,
        diagnostics_path=args.diagnostics,
        manifest_path=args.manifest,
        expected_source_count=args.expected_source_count,
        expected_eligible_count=args.expected_eligible_count,
        target_size=args.target_size,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
