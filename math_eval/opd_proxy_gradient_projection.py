"""Strict Prismatic-compatible CUDA projection for OPD gradient vectors."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


PRISMATIC_REFERENCE_COMMIT = "d9484cd3b5991030b901ac4a3a9e2472dbfac2ad"
PRISMATIC_REFERENCE_TREE = "a0079d8c5e15cb18bb4790f99c43cc19bb9ecd50"
DEFAULT_REFERENCE_REPO = Path("/home/mchen/prismatic-synthesis-reference")
OFFICIAL_GRADIENT_MODULE = Path(
    "prismatic-synthesis/gradient_modules/gradient_computer.py"
)
TRAKER_VERSION = "0.3.2"
FAST_JL_VERSION = "0.1.3"
PROJECTION_LINEARITY_RTOL = 5e-3
PROJECTION_LINEARITY_ATOL = 5e-3
_HASH_40 = re.compile(r"[0-9a-f]{40}")
_HASH_64 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ProjectionConfig:
    """The only accepted OPD projection configuration."""

    dimension: int = 1024
    seed: int = 0
    block_size: int = 128
    max_batch_size: int = 16
    model_id: int = 0
    input_dtype: torch.dtype = torch.float16


@dataclass(frozen=True)
class ReferenceSnapshot:
    """Exact reference checkout and imported source identity."""

    repository: str
    commit: str
    tree: str
    official_gradient_module: str
    official_gradient_module_sha256: str

    def as_manifest(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class OfficialProjectorSymbols:
    """Symbols imported through the verified official Prismatic module."""

    cuda_projector_class: type
    projection_type: Any
    reference: ReferenceSnapshot


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(f"value is not canonical-JSON serializable: {error}") from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_output(repository: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(
            f"invalid Prismatic reference repository {repository}: {error}"
        ) from error
    return completed.stdout.strip()


def _require_git_hash(value: str, *, field: str) -> str:
    if not isinstance(value, str) or _HASH_40.fullmatch(value) is None:
        raise ValueError(f"expected {field} must be a lowercase 40-character git hash")
    return value


def verify_prismatic_reference(
    reference_repo: Path | str,
    *,
    expected_commit: str = PRISMATIC_REFERENCE_COMMIT,
    expected_tree: str = PRISMATIC_REFERENCE_TREE,
) -> ReferenceSnapshot:
    """Require an exact clean Prismatic checkout and hash its official module."""
    expected_commit = _require_git_hash(expected_commit, field="commit")
    expected_tree = _require_git_hash(expected_tree, field="tree")
    repository = Path(reference_repo).expanduser().resolve()
    if not repository.is_dir():
        raise ValueError(f"Prismatic reference repository does not exist: {repository}")
    actual_commit = _git_output(repository, "rev-parse", "HEAD")
    if actual_commit != expected_commit:
        raise ValueError(
            "Prismatic reference commit mismatch: "
            f"expected {expected_commit}, got {actual_commit}"
        )
    actual_tree = _git_output(repository, "rev-parse", "HEAD^{tree}")
    if actual_tree != expected_tree:
        raise ValueError(
            "Prismatic reference tree mismatch: "
            f"expected {expected_tree}, got {actual_tree}"
        )
    status = _git_output(
        repository, "status", "--porcelain=v1", "--untracked-files=all"
    )
    if status:
        raise ValueError("Prismatic reference repository must have a clean working tree")

    source = (repository / OFFICIAL_GRADIENT_MODULE).resolve()
    try:
        source.relative_to(repository)
    except ValueError as error:
        raise ValueError("official Prismatic gradient module escapes repository") from error
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"official Prismatic gradient module is missing: {source}")
    relative_source = source.relative_to(repository).as_posix()
    _git_output(repository, "ls-files", "--error-unmatch", relative_source)
    return ReferenceSnapshot(
        repository=str(repository),
        commit=actual_commit,
        tree=actual_tree,
        official_gradient_module=str(source),
        official_gradient_module_sha256=_sha256_file(source),
    )


def _require_distribution_version(distribution: str, expected: str) -> None:
    try:
        actual = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            f"{distribution} version mismatch: expected {expected}, package missing"
        ) from error
    if actual != expected:
        raise RuntimeError(
            f"{distribution} version mismatch: expected {expected}, got {actual}"
        )


def load_cuda_projector(
    reference_repo: Path | str = DEFAULT_REFERENCE_REPO,
) -> OfficialProjectorSymbols | None:
    """Import CudaProjector and ProjectionType only through verified official code."""
    snapshot = verify_prismatic_reference(reference_repo)
    _require_distribution_version("traker", TRAKER_VERSION)
    _require_distribution_version("fast-jl", FAST_JL_VERSION)
    source = Path(snapshot.official_gradient_module)
    module_name = (
        "_fire_opd_verified_prismatic_gradient_"
        f"{snapshot.official_gradient_module_sha256}"
    )
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import official Prismatic module: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    imported_path = Path(getattr(module, "__file__", "")).resolve()
    if imported_path != source:
        raise RuntimeError(
            f"official Prismatic import path mismatch: expected {source}, got {imported_path}"
        )

    cuda_projector_class = getattr(module, "CudaProjector", None)
    projection_type = getattr(module, "ProjectionType", None)
    if not isinstance(cuda_projector_class, type) or cuda_projector_class.__name__ != "CudaProjector":
        raise RuntimeError("CudaProjector is required; CPU projector fallback is forbidden")
    if projection_type is None or not hasattr(projection_type, "rademacher"):
        raise RuntimeError("official ProjectionType.rademacher is unavailable")
    try:
        from trak.projectors import CudaProjector, ProjectionType
    except ImportError as error:
        raise RuntimeError("CudaProjector is required; traker import failed") from error
    if cuda_projector_class is not CudaProjector or projection_type is not ProjectionType:
        raise RuntimeError("official Prismatic projector symbols have unexpected origins")
    class_source = Path(inspect.getfile(cuda_projector_class)).resolve()
    if "trak" not in class_source.parts:
        raise RuntimeError("CudaProjector class origin is not the pinned traker package")
    return OfficialProjectorSymbols(
        cuda_projector_class=cuda_projector_class,
        projection_type=projection_type,
        reference=snapshot,
    )


def _validate_projection_config(config: ProjectionConfig) -> None:
    if not isinstance(config, ProjectionConfig):
        raise TypeError("projection config must be ProjectionConfig")
    frozen = {
        "dimension": 1024,
        "seed": 0,
        "block_size": 128,
        "max_batch_size": 16,
        "model_id": 0,
        "input_dtype": torch.float16,
    }
    for field, expected in frozen.items():
        actual = getattr(config, field)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(
                f"projection {field} mismatch: expected {expected!r}, got {actual!r}"
            )


def construct_cuda_projector(
    gradient_dimension: int,
    device: str,
    config: ProjectionConfig,
    *,
    reference_repo: Path | str = DEFAULT_REFERENCE_REPO,
    cuda_projector_class=None,
):
    """Construct the exact Rademacher CudaProjector without fallback."""
    _validate_projection_config(config)
    if (
        isinstance(gradient_dimension, bool)
        or not isinstance(gradient_dimension, int)
        or gradient_dimension <= 0
    ):
        raise ValueError("gradient dimension must be a positive integer")
    if device != "cuda:0":
        raise ValueError("projector device must be process-local cuda:0")
    symbols = load_cuda_projector(reference_repo)
    if symbols is None or not isinstance(symbols, OfficialProjectorSymbols):
        raise RuntimeError("CudaProjector is required; CPU projector fallback is forbidden")
    if not hasattr(symbols.projection_type, "rademacher"):
        raise RuntimeError("official ProjectionType.rademacher is unavailable")
    if cuda_projector_class is None:
        cuda_projector_class = symbols.cuda_projector_class
        if (
            not isinstance(cuda_projector_class, type)
            or cuda_projector_class.__name__ != "CudaProjector"
        ):
            raise RuntimeError(
                "CudaProjector is required; CPU projector fallback is forbidden"
            )
    if not callable(cuda_projector_class):
        raise TypeError("cuda_projector_class must be callable")
    projector = cuda_projector_class(
        grad_dim=gradient_dimension,
        proj_dim=config.dimension,
        seed=config.seed,
        proj_type=symbols.projection_type.rademacher,
        device=device,
        dtype=config.input_dtype,
        block_size=config.block_size,
        max_batch_size=config.max_batch_size,
    )
    if not callable(getattr(projector, "project", None)):
        raise RuntimeError("constructed CudaProjector has no project method")
    return projector


def full_gradient_l2_norm(flat_gradient: torch.Tensor) -> torch.Tensor:
    """Measure a native one-dimensional float32 full gradient before projection."""
    if (
        not isinstance(flat_gradient, torch.Tensor)
        or flat_gradient.dtype != torch.float32
        or flat_gradient.ndim != 1
        or flat_gradient.numel() == 0
        or flat_gradient.layout != torch.strided
    ):
        raise ValueError("native full gradient must be nonempty one-dimensional float32")
    if not bool(torch.isfinite(flat_gradient).all().item()):
        raise ValueError("native full gradient must be finite and nonzero")
    norm = torch.linalg.vector_norm(flat_gradient)
    if not bool(torch.isfinite(norm).item()) or float(norm.item()) <= 0.0:
        raise ValueError("native full gradient must be finite and nonzero")
    return norm


def project_full_gradient(
    projector,
    flat_gradient: torch.Tensor,
    config: ProjectionConfig,
) -> torch.Tensor:
    """Project one native float32 full gradient and return scaled float32 output."""
    _validate_projection_config(config)
    # This call intentionally precedes the sole float16 cast.
    full_gradient_l2_norm(flat_gradient)
    projector_input = flat_gradient.detach().contiguous().to(config.input_dtype).unsqueeze(0)
    if not bool(torch.isfinite(projector_input).all().item()):
        raise ValueError("native full gradient overflows the float16 projector input")
    project = getattr(projector, "project", None)
    if not callable(project):
        raise RuntimeError("CudaProjector is required and must expose project")
    raw_projected = project(projector_input, model_id=config.model_id)
    if (
        not isinstance(raw_projected, torch.Tensor)
        or raw_projected.shape != (1, config.dimension)
        or raw_projected.layout != torch.strided
    ):
        raise ValueError("invalid projected gradient shape or layout")
    projected = raw_projected.detach().to(torch.float32) / math.sqrt(config.dimension)
    if not bool(torch.isfinite(projected).all().item()):
        raise ValueError("invalid projected gradient: values must be finite")
    projected_norm = torch.linalg.vector_norm(projected)
    if not bool(torch.isfinite(projected_norm).item()) or float(projected_norm.item()) <= 0.0:
        raise ValueError("invalid projected gradient: norm must be finite and nonzero")
    return projected


def projected_mean_is_mean_projection(
    projector,
    full_gradients: Sequence[torch.Tensor],
    config: ProjectionConfig,
    *,
    rtol: float = PROJECTION_LINEARITY_RTOL,
    atol: float = PROJECTION_LINEARITY_ATOL,
) -> bool:
    """Check direct projection of a float32 mean against mean projections."""
    if not isinstance(full_gradients, Sequence) or len(full_gradients) < 2:
        raise ValueError("projection linearity check requires at least two gradients")
    if not math.isfinite(rtol) or not math.isfinite(atol) or rtol < 0 or atol < 0:
        raise ValueError("projection linearity tolerances must be finite and nonnegative")
    first_shape = full_gradients[0].shape
    first_device = full_gradients[0].device
    for gradient in full_gradients:
        full_gradient_l2_norm(gradient)
        if gradient.shape != first_shape or gradient.device != first_device:
            raise ValueError("full gradients must have identical shapes and devices")
    mean_full_gradient = torch.stack(tuple(full_gradients), dim=0).mean(dim=0)
    direct = project_full_gradient(projector, mean_full_gradient, config)
    projected = torch.stack(
        tuple(project_full_gradient(projector, value, config) for value in full_gradients),
        dim=0,
    ).mean(dim=0)
    return bool(torch.allclose(direct, projected, rtol=rtol, atol=atol))


def _layout_value(entry: object, field: str) -> object:
    if isinstance(entry, Mapping):
        if field not in entry:
            raise ValueError(f"parameter layout entry lacks {field}")
        return entry[field]
    if not hasattr(entry, field):
        raise ValueError(f"parameter layout entry lacks {field}")
    return getattr(entry, field)


def sha256_parameter_layout(entries: Sequence[object]) -> str:
    """Hash validated registration-order parameter layout records."""
    if not isinstance(entries, Sequence) or not entries:
        raise ValueError("parameter layout must contain at least one entry")
    normalized: list[dict[str, object]] = []
    expected_offset = 0
    seen_names: set[str] = set()
    for index, entry in enumerate(entries):
        name = _layout_value(entry, "name")
        shape_value = _layout_value(entry, "shape")
        numel = _layout_value(entry, "numel")
        offset = _layout_value(entry, "offset")
        if not isinstance(name, str) or not name or name in seen_names:
            raise ValueError(f"invalid or duplicate parameter name at layout row {index}")
        if not isinstance(shape_value, Sequence) or isinstance(shape_value, str | bytes):
            raise ValueError(f"invalid parameter shape at layout row {index}")
        shape = []
        for dimension in shape_value:
            if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0:
                raise ValueError(f"invalid parameter shape at layout row {index}")
            shape.append(dimension)
        expected_numel = math.prod(shape)
        if (
            isinstance(numel, bool)
            or not isinstance(numel, int)
            or numel <= 0
            or numel != expected_numel
        ):
            raise ValueError(f"invalid parameter numel at layout row {index}")
        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or offset != expected_offset
        ):
            raise ValueError("parameter layout offsets must be contiguous")
        normalized.append(
            {"name": name, "shape": shape, "numel": numel, "offset": offset}
        )
        seen_names.add(name)
        expected_offset += numel
    return hashlib.sha256(_canonical_json_bytes(normalized)).hexdigest()


def build_projection_manifest(
    *,
    gradient_dimension: int,
    parameter_layout_sha256: str,
    config: ProjectionConfig,
    reference: ReferenceSnapshot,
    device: str = "cuda:0",
) -> dict[str, object]:
    """Build the projection provenance embedded in every vector manifest."""
    _validate_projection_config(config)
    if (
        isinstance(gradient_dimension, bool)
        or not isinstance(gradient_dimension, int)
        or gradient_dimension <= 0
    ):
        raise ValueError("gradient dimension must be a positive integer")
    if not isinstance(parameter_layout_sha256, str) or _HASH_64.fullmatch(
        parameter_layout_sha256
    ) is None:
        raise ValueError("parameter layout SHA-256 must be 64 lowercase hex characters")
    if not isinstance(reference, ReferenceSnapshot):
        raise TypeError("reference must be a ReferenceSnapshot")
    if device != "cuda:0":
        raise ValueError("projector device must be process-local cuda:0")
    for field in ("commit", "tree"):
        _require_git_hash(getattr(reference, field), field=field)
    if _HASH_64.fullmatch(reference.official_gradient_module_sha256) is None:
        raise ValueError("official module SHA-256 must be 64 lowercase hex characters")
    repository = Path(reference.repository)
    source = Path(reference.official_gradient_module)
    if not repository.is_absolute() or not source.is_absolute():
        raise ValueError("reference repository and source paths must be absolute")
    try:
        source.resolve().relative_to(repository.resolve())
    except ValueError as error:
        raise ValueError("official reference source escapes its repository") from error
    if not source.is_file() or _sha256_file(source) != reference.official_gradient_module_sha256:
        raise ValueError("official reference source hash mismatch")
    _require_distribution_version("traker", TRAKER_VERSION)
    _require_distribution_version("fast-jl", FAST_JL_VERSION)
    manifest: dict[str, object] = {
        "backend": "CudaProjector",
        "constructor": {
            "grad_dim": gradient_dimension,
            "proj_dim": config.dimension,
            "seed": config.seed,
            "proj_type": "rademacher",
            "device": device,
            "dtype": "float16",
            "block_size": config.block_size,
            "max_batch_size": config.max_batch_size,
        },
        "model_id": config.model_id,
        "native_gradient_dtype": "float32",
        "projector_input_dtype": "float16",
        "stored_dtype": "float32",
        "output_scale": f"1/sqrt({config.dimension})",
        "linearity_tolerance": {
            "rtol": PROJECTION_LINEARITY_RTOL,
            "atol": PROJECTION_LINEARITY_ATOL,
        },
        "flatten_order": "named_parameters(remove_duplicate=True):registration_order",
        "parameter_layout_sha256": parameter_layout_sha256,
        "reference": reference.as_manifest(),
        "package_versions": {
            "traker": TRAKER_VERSION,
            "fast-jl": FAST_JL_VERSION,
        },
    }
    _canonical_json_bytes(manifest)
    return manifest
