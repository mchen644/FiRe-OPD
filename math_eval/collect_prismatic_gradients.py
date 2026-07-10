"""Safely orchestrate the pinned official Prismatic gradient collector."""

from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import importlib.util
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

if __package__ in (None, ""):
    repository_root = str(Path(__file__).resolve().parents[1])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)

from math_eval.deepmath_gradient_diversity import official_shard_bounds, sha256_file


REFERENCE_COMMIT = "d9484cd3b5991030b901ac4a3a9e2472dbfac2ad"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
DATASET_NAME = "zwhe99/DeepMath-103K"
DATASET_REVISION = "5cf055d1fe3d7a2eb19719ac020211469736ae44"
PROJECTION_DIM = 1024
PROJECTION_SEED = 0
PROJECT_INTERVAL = 4
SAVE_INTERVAL = 500
MAX_CONTEXT_TOKENS = 32_768
ASSISTANT_RESPONSE_MARKER = "<|im_start|>assistant"
TRL_VERSION = "0.17.0"
GRADIENT_MANIFEST_NAME = "gradient.manifest.json"


def _git_output(repository: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"invalid reference repository {repository}: {error}") from error
    return result.stdout.strip()


def verify_reference_repo(path: Path, expected_commit: str) -> None:
    """Require the reference repository to be clean and at an exact commit."""
    repository = Path(path)
    actual_commit = _git_output(repository, "rev-parse", "HEAD")
    if actual_commit != expected_commit:
        raise ValueError(
            f"reference repository HEAD mismatch: expected {expected_commit}, "
            f"got {actual_commit}"
        )
    status = _git_output(
        repository, "status", "--porcelain=v1", "--untracked-files=all"
    )
    if status:
        raise ValueError("reference repository must have a clean working tree")


def _reference_tree(repository: Path) -> str:
    return _git_output(Path(repository), "rev-parse", "HEAD^{tree}")


def _official_gradient_module_path(reference_repo: Path) -> Path:
    return (
        Path(reference_repo)
        / "prismatic-synthesis"
        / "gradient_modules"
        / "gradient_computer.py"
    ).resolve()


def load_official_gradient_computer_class(
    reference_repo: Path, expected_commit: str
):
    """Import the verified official class from its one pinned source file."""
    repository = Path(reference_repo).resolve()
    verify_reference_repo(repository, expected_commit)
    module_path = _official_gradient_module_path(repository)
    if not module_path.is_file():
        raise ValueError(
            f"official GradientComputer module does not exist: {module_path}"
        )

    module_name = (
        "_fire_opd_prismatic_gradient_computer_"
        f"{expected_commit}_{abs(hash(str(module_path)))}"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import official GradientComputer module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise

    imported_path = Path(getattr(module, "__file__", "")).resolve()
    if imported_path != module_path:
        raise ValueError(
            "official GradientComputer module path mismatch: "
            f"expected {module_path}, got {imported_path}"
        )
    gradient_computer = getattr(module, "GradientComputer", None)
    if not isinstance(gradient_computer, type):
        raise ValueError(
            f"official GradientComputer class is missing from {module_path}"
        )
    class_origin = Path(inspect.getfile(gradient_computer)).resolve()
    if class_origin != module_path:
        raise RuntimeError(
            "official GradientComputer class origin mismatch: "
            f"expected {module_path}, got {class_origin}"
        )
    return gradient_computer


def validate_single_cuda_device(device: str) -> None:
    """Require one process-local GPU and prohibit physical device indices."""
    if device != "cuda:0":
        raise ValueError("collector device must be the process-local cuda:0")
    visible_count = torch.cuda.device_count()
    if visible_count != 1:
        raise RuntimeError(
            "gradient collector requires exactly one visible CUDA device; "
            f"found {visible_count}"
        )


def load_model_and_tokenizer(
    model_name: str, model_revision: str, device: str
):
    """Load the exactly pinned proxy and configure official-compatible padding."""
    if model_name != MODEL_NAME:
        raise ValueError(
            f"pinned model name mismatch: expected {MODEL_NAME}, got {model_name}"
        )
    if model_revision != MODEL_REVISION:
        raise ValueError(
            "pinned model revision mismatch: "
            f"expected {MODEL_REVISION}, got {model_revision}"
        )
    if device != "cuda:0":
        raise ValueError("model device must be the process-local cuda:0")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=model_revision,
        torch_dtype="auto",
    )
    configured_context = getattr(model.config, "max_position_embeddings", None)
    if configured_context != MAX_CONTEXT_TOKENS:
        raise ValueError(
            "pinned model max_position_embeddings mismatch: "
            f"expected {MAX_CONTEXT_TOKENS}, got {configured_context}"
        )
    model = model.to(torch.device(device))

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=model_revision,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    return model, tokenizer


def construct_strict_collector(
    gradient_computer_class,
    model_name: str,
    model,
    tokenizer,
    *,
    cuda_projector_class=None,
):
    """Construct the official collector while forbidding BasicProjector fallback."""
    if cuda_projector_class is None:
        from trak.projectors import CudaProjector

        cuda_projector_class = CudaProjector

    descriptor = gradient_computer_class.__dict__.get("get_trak_projector")
    if descriptor is None:
        raise ValueError("official GradientComputer has no get_trak_projector factory")
    original_factory = getattr(gradient_computer_class, "get_trak_projector")

    def require_cuda_projector(device):
        projector_class = original_factory(device)
        if projector_class is not cuda_projector_class:
            raise RuntimeError(
                "official projector probe did not return exact CudaProjector; "
                "refusing unsafe BasicProjector fallback"
            )
        return projector_class

    setattr(
        gradient_computer_class,
        "get_trak_projector",
        staticmethod(require_cuda_projector),
    )
    try:
        collector = gradient_computer_class(
            model_name=model_name,
            model=model,
            tokenizer=tokenizer,
        )
    finally:
        setattr(gradient_computer_class, "get_trak_projector", descriptor)

    if type(collector.projector) is not cuda_projector_class:
        raise RuntimeError("official collector projector is not exact CudaProjector")
    for field, expected in (
        ("proj_dim", PROJECTION_DIM),
        ("project_interval", PROJECT_INTERVAL),
        ("save_interval", SAVE_INTERVAL),
    ):
        actual = getattr(collector, field, None)
        if actual != expected:
            raise RuntimeError(
                f"official collector {field} mismatch: expected {expected}, got {actual}"
            )
    return collector


def _read_json_object(path: Path, description: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"invalid {description} {path}: expected a JSON object")
    return value


def _require_manifest_value(
    manifest: Mapping, field: str, expected: object, *, description: str = "manifest"
) -> None:
    actual = manifest.get(field)
    if actual != expected:
        raise ValueError(
            f"{description} {field} mismatch: expected {expected!r}, got {actual!r}"
        )


def load_prepared_pool(
    prepared_jsonl: Path, prepared_manifest: Path
) -> tuple[list[dict], dict]:
    """Load and fully validate the pinned prepared gradient pool."""
    prepared_path = Path(prepared_jsonl)
    manifest_path = Path(prepared_manifest)
    manifest = _read_json_object(manifest_path, "prepared manifest")

    required_provenance = {
        "manifest_version": 1,
        "dataset_name": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "dataset_split": "train",
        "prepared_jsonl": str(prepared_path.resolve()),
    }
    for field, expected in required_provenance.items():
        _require_manifest_value(manifest, field, expected, description="prepared manifest")

    source_value = manifest.get("source_parquet")
    if not isinstance(source_value, str) or not source_value:
        raise ValueError("prepared manifest source_parquet must be a nonempty path")
    source_path = Path(source_value)
    if not source_path.is_absolute() or not source_path.is_file():
        raise ValueError(
            f"prepared manifest source_parquet does not exist: {source_value!r}"
        )
    _require_manifest_value(
        manifest,
        "source_sha256",
        sha256_file(source_path),
        description="prepared manifest",
    )

    if not prepared_path.is_file():
        raise ValueError(f"prepared JSONL does not exist: {prepared_path}")
    _require_manifest_value(
        manifest,
        "prepared_jsonl_sha256",
        sha256_file(prepared_path),
        description="prepared manifest",
    )

    counts = {
        manifest.get("source_row_count"),
        manifest.get("prepared_row_count"),
        manifest.get("expected_count"),
    }
    if len(counts) != 1 or not counts or not isinstance(next(iter(counts)), int):
        raise ValueError("prepared manifest row count fields do not agree")
    expected_count = next(iter(counts))
    if expected_count < 0:
        raise ValueError("prepared manifest row count must be non-negative")

    rows: list[dict] = []
    seen_ids: set[str] = set()
    try:
        with prepared_path.open("r", encoding="utf-8") as handle:
            for row_index, line in enumerate(handle):
                if not line.strip():
                    raise ValueError(f"prepared JSONL contains blank line {row_index + 1}")
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(
                        f"prepared JSONL row {row_index} must be a JSON object"
                    )

                sample_id = row.get("id")
                if not isinstance(sample_id, str) or not sample_id:
                    raise ValueError(
                        f"prepared JSONL row {row_index} id must be a nonempty string"
                    )
                if sample_id in seen_ids:
                    raise ValueError(f"duplicate prepared ID at row {row_index}: {sample_id}")
                seen_ids.add(sample_id)
                expected_id = f"deepmath-level6-{row_index:06d}"
                if sample_id != expected_id:
                    raise ValueError(
                        f"prepared stable ID mismatch at row {row_index}: "
                        f"expected {expected_id}, got {sample_id}"
                    )

                for text_field in ("prompt", "completion"):
                    value = row.get(text_field)
                    if not isinstance(value, str) or not value.strip():
                        raise ValueError(
                            f"prepared row {row_index} {text_field} must be a "
                            "nonempty string"
                        )
                if row.get("source_row_index") != row_index:
                    raise ValueError(
                        f"prepared source_row_index mismatch at row {row_index}"
                    )
                rows.append(row)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid prepared JSONL {prepared_path}: {error}") from error

    if len(rows) != expected_count:
        raise ValueError(
            f"prepared row count mismatch: expected {expected_count}, got {len(rows)}"
        )
    return rows, manifest


def _canonical_json_bytes(value: Mapping) -> tuple[dict, bytes]:
    try:
        serialized = json.dumps(
            dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        normalized = json.loads(serialized)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"gradient manifest is not JSON serializable: {error}") from error
    if not isinstance(normalized, dict):
        raise ValueError("gradient manifest must be a JSON object")
    return normalized, (serialized + "\n").encode("utf-8")


def write_or_validate_gradient_manifest(path: Path, expected: Mapping) -> dict:
    """Create one canonical manifest or validate it under a process lock."""
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    expected_dict, payload = _canonical_json_bytes(expected)
    lock_path = manifest_path.with_name(f".{manifest_path.name}.lock")
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        if manifest_path.exists():
            actual = _read_json_object(manifest_path, "gradient manifest")
            if actual != expected_dict:
                raise ValueError(
                    "gradient manifest mismatch: existing provenance or "
                    "hyperparameters differ"
                )
            return actual

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=manifest_path.parent,
                prefix=f".{manifest_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary_path, manifest_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return expected_dict
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)


def _read_sidecar(path: Path) -> list[str]:
    sample_ids: list[str] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(f"blank sidecar line {line_number} in {path.name}")
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"invalid sidecar row in {path.name}")
                sample_id = record.get("id")
                if not isinstance(sample_id, str) or not sample_id:
                    raise ValueError(f"invalid sidecar ID in {path.name}")
                sample_ids.append(sample_id)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid gradient sidecar {path}: {error}") from error
    if not sample_ids:
        raise ValueError(f"gradient sidecar {path.name} is empty")
    return sample_ids


def _validate_tensor_file(path: Path, sidecar_ids: Sequence[str]) -> None:
    try:
        tensors = load_file(path, device="cpu")
    except Exception as error:
        raise ValueError(f"unreadable gradient safetensors {path}: {error}") from error
    if set(tensors) != set(sidecar_ids) or len(tensors) != len(sidecar_ids):
        raise ValueError(f"gradient tensor keys do not match sidecar IDs in {path.name}")
    for sample_id in sidecar_ids:
        tensor = tensors[sample_id]
        if tensor.dtype != torch.float16:
            raise ValueError(f"gradient for {sample_id} must have float16 dtype")
        if tuple(tensor.shape) != (PROJECTION_DIM,):
            raise ValueError(
                f"gradient for {sample_id} has invalid shape {tuple(tensor.shape)}"
            )
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"gradient for {sample_id} must contain only finite values")
        if int(torch.count_nonzero(tensor)) == 0:
            raise ValueError(f"gradient for {sample_id} must be non-zero")


def resolve_resume_start(
    dataset_ids: Sequence[str],
    output_dir: Path,
    prefix: str,
    shard_start: int,
    shard_end: int,
) -> int:
    """Validate official chunk files and return the first unresolved index."""
    if not isinstance(prefix, str) or not prefix:
        raise ValueError("gradient prefix must be a nonempty string")
    if shard_start < 0 or shard_end < shard_start or shard_end > len(dataset_ids):
        raise ValueError("invalid logical shard bounds")

    directory = Path(output_dir)
    if not directory.exists():
        return shard_start
    exact_pattern = re.compile(
        rf"^{re.escape(prefix)}\.(0|[1-9][0-9]*)\.(txt|safetensors)$"
    )
    files_by_start: dict[int, dict[str, Path]] = {}
    for path in directory.iterdir():
        if not path.is_file():
            continue
        name = path.name
        if not name.startswith(f"{prefix}."):
            continue
        if not (name.endswith(".txt") or name.endswith(".safetensors")):
            continue
        match = exact_pattern.fullmatch(name)
        if match is None:
            raise ValueError(f"malformed gradient chunk filename: {name}")
        start = int(match.group(1))
        if shard_start <= start < shard_end:
            files_by_start.setdefault(start, {})[match.group(2)] = path

    cursor = shard_start
    for chunk_start in sorted(files_by_start):
        files = files_by_start[chunk_start]
        if set(files) != {"txt", "safetensors"}:
            raise ValueError(
                f"gradient chunk {prefix}.{chunk_start} must have paired "
                ".txt and .safetensors files"
            )
        if chunk_start > cursor:
            raise ValueError(
                f"gradient chunk gap: expected absolute start {cursor}, "
                f"found {chunk_start}"
            )
        if chunk_start < cursor:
            raise ValueError(
                f"gradient chunk overlap: expected absolute start {cursor}, "
                f"found {chunk_start}"
            )

        sidecar_ids = _read_sidecar(files["txt"])
        chunk_end = chunk_start + len(sidecar_ids)
        if chunk_end > shard_end:
            raise ValueError(
                f"gradient chunk {chunk_start} extends beyond logical shard end"
            )
        expected_ids = list(dataset_ids[chunk_start:chunk_end])
        if sidecar_ids != expected_ids:
            raise ValueError(
                f"gradient sidecar IDs do not match expected ordered slice at "
                f"absolute start {chunk_start}"
            )
        if len(sidecar_ids) > SAVE_INTERVAL:
            raise ValueError(
                f"gradient chunk {chunk_start} exceeds save interval {SAVE_INTERVAL}"
            )
        if len(sidecar_ids) < SAVE_INTERVAL and chunk_end != shard_end:
            raise ValueError(
                f"short gradient chunk {chunk_start} is only valid at shard end"
            )
        _validate_tensor_file(files["safetensors"], sidecar_ids)
        cursor = chunk_end

    return cursor


def _token_count(encoding: Mapping, sample_id: str) -> tuple[int, torch.Tensor]:
    input_ids = encoding.get("input_ids")
    labels = encoding.get("labels")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise ValueError(
            f"official encoding for {sample_id} must contain rank-2 input_ids"
        )
    if not isinstance(labels, torch.Tensor) or labels.shape != input_ids.shape:
        raise ValueError(
            f"official encoding for {sample_id} must contain labels matching input_ids"
        )
    if input_ids.shape[0] != 1:
        raise ValueError(f"official encoding for {sample_id} must contain one sample")
    return int(input_ids.shape[-1]), labels


def preflight_samples(
    samples: Sequence[Mapping],
    tokenizer,
    collector,
) -> dict[str, int]:
    """Validate official Qwen formatting and completion-only labels without truncation."""
    problems: list[str] = []
    overlong: list[str] = []
    max_input_tokens = 0
    min_supervised_tokens: int | None = None
    for sample in samples:
        sample_id = sample["id"]
        prompt = sample["prompt"]
        completion = sample["completion"]
        encoding = collector.prepare_model_input(prompt, completion)
        token_count, labels = _token_count(encoding, sample_id)
        max_input_tokens = max(max_input_tokens, token_count)

        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False)
        if not isinstance(formatted, str):
            problems.append(
                f"official chat formatting for {sample_id} did not return text"
            )
        elif formatted.count(ASSISTANT_RESPONSE_MARKER) != 1:
            problems.append(
                f"sample {sample_id} has {token_count} tokens but assistant response "
                f"marker {ASSISTANT_RESPONSE_MARKER!r} does not occur exactly once"
            )
        if token_count > MAX_CONTEXT_TOKENS:
            overlong.append(f"{sample_id}={token_count}")
        supervised_tokens = int(torch.count_nonzero(labels != -100))
        if min_supervised_tokens is None:
            min_supervised_tokens = supervised_tokens
        else:
            min_supervised_tokens = min(
                min_supervised_tokens, supervised_tokens
            )
        if supervised_tokens == 0:
            problems.append(
                f"sample {sample_id} has {token_count} tokens but no supervised "
                "completion-only labels"
            )

    if overlong:
        problems.append(
            f"samples exceed pinned context boundary {MAX_CONTEXT_TOKENS}: "
            + ", ".join(overlong)
        )
    if problems:
        raise ValueError("; ".join(problems))
    return {
        "sample_count": len(samples),
        "max_input_tokens": max_input_tokens,
        "min_supervised_tokens": min_supervised_tokens or 0,
    }


def _installed_trl_version() -> str:
    try:
        version = importlib.metadata.version("trl")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("trl is not installed") from error
    if version != TRL_VERSION:
        raise RuntimeError(
            f"trl version mismatch: expected {TRL_VERSION}, got {version}"
        )
    return version


def _installed_package_versions() -> dict[str, str | None]:
    distributions = {
        "torch": "torch",
        "transformers": "transformers",
        "trl": "trl",
        "trak": "traker",
        "fast_jl": "fast-jl",
    }
    versions: dict[str, str | None] = {}
    for name, distribution in distributions.items():
        try:
            versions[name] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def build_gradient_manifest(
    prepared_jsonl: Path,
    prepared_manifest_path: Path,
    prepared_manifest: Mapping,
    reference_repo: Path,
    reference_commit: str,
    reference_tree: str,
    *,
    prefix: str,
    num_shards: int,
    trl_version: str,
    package_versions: Mapping[str, str | None],
) -> dict:
    """Build the complete immutable provenance contract for all collectors."""
    return {
        "manifest_version": 1,
        "prepared_jsonl": str(Path(prepared_jsonl).resolve()),
        "prepared_jsonl_sha256": prepared_manifest["prepared_jsonl_sha256"],
        "prepared_manifest": str(Path(prepared_manifest_path).resolve()),
        "prepared_manifest_sha256": sha256_file(Path(prepared_manifest_path)),
        "prepared_row_count": prepared_manifest["prepared_row_count"],
        "source_sha256": prepared_manifest["source_sha256"],
        "dataset_name": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "reference_repo": str(Path(reference_repo).resolve()),
        "reference_commit": reference_commit,
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
        "projection": {
            "dimension": PROJECTION_DIM,
            "seed": PROJECTION_SEED,
            "type": "rademacher",
            "dtype": "float16",
            "project_interval": PROJECT_INTERVAL,
            "save_interval": SAVE_INTERVAL,
            "completion_only_loss": True,
            "full_parameter_gradients": True,
            "response_marker": ASSISTANT_RESPONSE_MARKER,
        },
    }


def _validated_shard_bounds(
    total: int, num_shards: int, shard_index: int
) -> tuple[int, int]:
    start, end = official_shard_bounds(total, num_shards, shard_index)
    if start > total or end < start:
        raise ValueError(
            "empty logical shard starts beyond prepared dataset: "
            f"shard {shard_index}/{num_shards}, bounds {start}:{end}, total {total}"
        )
    return start, end


def _validate_model_context(model) -> None:
    configured_context = getattr(model.config, "max_position_embeddings", None)
    if configured_context != MAX_CONTEXT_TOKENS:
        raise ValueError(
            "pinned model hard context boundary mismatch: "
            f"expected {MAX_CONTEXT_TOKENS}, got {configured_context}"
        )


def run_collection(
    *,
    reference_repo: Path,
    prepared_jsonl: Path,
    prepared_manifest: Path,
    output_dir: Path,
    prefix: str,
    model_name: str,
    model_revision: str,
    shard_index: int,
    num_shards: int,
    device: str,
) -> dict[str, int | str]:
    """Validate, resume, and execute one explicit official logical shard."""
    if model_name != MODEL_NAME:
        raise ValueError(
            f"pinned model name mismatch: expected {MODEL_NAME}, got {model_name}"
        )
    if model_revision != MODEL_REVISION:
        raise ValueError(
            "pinned model revision mismatch: "
            f"expected {MODEL_REVISION}, got {model_revision}"
        )
    if device != "cuda:0":
        raise ValueError("collector device must be the process-local cuda:0")

    rows, source_manifest = load_prepared_pool(
        Path(prepared_jsonl), Path(prepared_manifest)
    )
    repository = Path(reference_repo).resolve()
    verify_reference_repo(repository, REFERENCE_COMMIT)
    reference_tree = _reference_tree(repository)
    official_module = _official_gradient_module_path(repository)
    if not official_module.is_file():
        raise ValueError(
            f"official GradientComputer module does not exist: {official_module}"
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    expected_manifest = build_gradient_manifest(
        Path(prepared_jsonl),
        Path(prepared_manifest),
        source_manifest,
        repository,
        REFERENCE_COMMIT,
        reference_tree,
        prefix=prefix,
        num_shards=num_shards,
        trl_version=_installed_trl_version(),
        package_versions=_installed_package_versions(),
    )
    write_or_validate_gradient_manifest(
        output_path / GRADIENT_MANIFEST_NAME,
        expected_manifest,
    )

    shard_start, shard_end = _validated_shard_bounds(
        len(rows), num_shards, shard_index
    )
    if shard_start == shard_end:
        return {
            "status": "empty",
            "shard_start": shard_start,
            "shard_end": shard_end,
            "resume_start": shard_start,
        }

    dataset_ids = [row["id"] for row in rows]
    resume_start = resolve_resume_start(
        dataset_ids,
        output_path,
        prefix,
        shard_start,
        shard_end,
    )
    if resume_start == shard_end:
        return {
            "status": "complete",
            "shard_start": shard_start,
            "shard_end": shard_end,
            "resume_start": resume_start,
        }

    validate_single_cuda_device(device)
    gradient_computer_class = load_official_gradient_computer_class(
        repository, REFERENCE_COMMIT
    )
    model, tokenizer = load_model_and_tokenizer(
        model_name, model_revision, device
    )
    _validate_model_context(model)
    collector = construct_strict_collector(
        gradient_computer_class,
        model_name,
        model,
        tokenizer,
    )
    unresolved = rows[resume_start:shard_end]
    preflight_samples(unresolved, tokenizer, collector)
    collector.compute_project_store_gradients(
        unresolved,
        prefix,
        output_path,
        resume_start,
    )

    completed_at = resolve_resume_start(
        dataset_ids,
        output_path,
        prefix,
        shard_start,
        shard_end,
    )
    if completed_at != shard_end:
        raise RuntimeError(
            "official gradient collection returned without complete shard coverage: "
            f"expected {shard_end}, got {completed_at}"
        )
    return {
        "status": "collected",
        "shard_start": shard_start,
        "shard_end": shard_end,
        "resume_start": resume_start,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect pinned official Prismatic projected gradients"
    )
    parser.add_argument("--reference-repo", type=Path, required=True)
    parser.add_argument("--prepared-jsonl", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--device", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_collection(
        reference_repo=args.reference_repo,
        prepared_jsonl=args.prepared_jsonl,
        prepared_manifest=args.prepared_manifest,
        output_dir=args.output_dir,
        prefix=args.prefix,
        model_name=args.model_name,
        model_revision=args.model_revision,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        device=args.device,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
