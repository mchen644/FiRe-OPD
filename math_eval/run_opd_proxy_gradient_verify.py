"""Fail-closed orchestration for the staged OPD proxy-gradient verification."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path, PurePosixPath


VERL_PYTHON = "/home/mchen/miniconda3/envs/verl/bin/python"
GVENDI_PYTHON = "/home/mchen/miniconda3/envs/gvendi-opd/bin/python"
DEFAULT_REFERENCE_REPO = Path("/home/mchen/prismatic-synthesis-reference")
DEFAULT_OUTPUT_ROOT = Path("data/opd_proxy_gradient_verify")
CAPTURE_SCRIPT = Path("verl/examples/fire_opd/run_capture_opd_proxy_verify.sh")
DIRECT_FIXTURE_SCRIPT = Path(
    "verl/examples/fire_opd/run_direct_opd_proxy_gradient_fixture.sh"
)
SEEDS = (42, 43)
REPLAY_SHARDS = 4
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
REFERENCE_COMMIT = "d9484cd3b5991030b901ac4a3a9e2472dbfac2ad"
REFERENCE_TREE = "a0079d8c5e15cb18bb4790f99c43cc19bb9ecd50"
_STAGE_LAYOUTS = {
    0: (24, 8, 5, 2, 2, 100),
    1: (768, 256, 172, 76, 7, 10_000),
    2: (1536, 512, 345, 153, 15, 10_000),
}

MAIN_SOURCE_FILES = (
    "docs/superpowers/specs/2026-07-14-vanilla-opd-proxy-gradient-selection-verify-design.md",
    "docs/superpowers/plans/2026-07-14-vanilla-opd-proxy-gradient-selection-verify.md",
    "pytest.ini",
    "math_eval/build_gradient_eligibility.py",
    "math_eval/deepmath_gradient_diversity.py",
    "math_eval/select_gradient_diverse_deepmath.py",
    "math_eval/collect_prismatic_gradients.py",
    "math_eval/test_collect_prismatic_gradients.py",
    "math_eval/opd_proxy_gradient_verify_artifacts.py",
    "math_eval/test_opd_proxy_gradient_verify_artifacts.py",
    "math_eval/prepare_opd_proxy_gradient_verify.py",
    "math_eval/test_prepare_opd_proxy_gradient_verify.py",
    "math_eval/opd_proxy_gradient_projection.py",
    "math_eval/test_opd_proxy_gradient_projection.py",
    "math_eval/replay_opd_proxy_gradients.py",
    "math_eval/test_replay_opd_proxy_gradients.py",
    "math_eval/collect_opd_proxy_sft_gradients.py",
    "math_eval/test_collect_opd_proxy_sft_gradients.py",
    "math_eval/collect_opd_proxy_prompt_embeddings.py",
    "math_eval/test_collect_opd_proxy_prompt_embeddings.py",
    "math_eval/select_opd_proxy_gradient_verify.py",
    "math_eval/test_select_opd_proxy_gradient_verify.py",
    "math_eval/opd_proxy_gradient_statistics.py",
    "math_eval/test_opd_proxy_gradient_statistics.py",
    "math_eval/opd_proxy_gradient_classification.py",
    "math_eval/test_opd_proxy_gradient_classification.py",
    "math_eval/test_opd_proxy_gradient_source_audit.py",
    "math_eval/analyze_opd_proxy_gradient_verify.py",
    "math_eval/test_analyze_opd_proxy_gradient_verify.py",
    "math_eval/run_opd_proxy_gradient_verify.py",
    "math_eval/test_run_opd_proxy_gradient_verify.py",
    "verl/examples/fire_opd/run_capture_opd_proxy_verify.sh",
    "verl/examples/fire_opd/run_direct_opd_proxy_gradient_fixture.sh",
    "verl/verl/trainer/ppo/opd_proxy_verify_capture.py",
    "verl/verl/trainer/ppo/core_algos.py",
    "verl/verl/trainer/ppo/ref_input_utils.py",
    "verl/verl/trainer/ppo/rollout_corr_helper.py",
    "verl/verl/trainer/ppo/ray_trainer.py",
    "verl/verl/trainer/main_ppo.py",
    "verl/verl/workers/fsdp_workers.py",
    "verl/verl/workers/actor/dp_actor.py",
    "verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py",
    "verl/verl/trainer/config/algorithm.py",
    "verl/verl/trainer/config/ppo_trainer.yaml",
    "verl/verl/trainer/config/actor/actor.yaml",
    "verl/verl/trainer/config/rollout/rollout.yaml",
    "verl/verl/workers/config/actor.py",
    "verl/verl/workers/config/rollout.py",
    "verl/verl/workers/config/__init__.py",
    "verl/verl/trainer/config/__init__.py",
    "verl/verl/base_config.py",
    "verl/verl/protocol.py",
    "verl/verl/utils/config.py",
    "verl/verl/utils/dataset/rl_dataset.py",
    "verl/verl/models/transformers/monkey_patch.py",
    "verl/tests/workers/rollout/test_opd_proxy_verify_native_n.py",
    "verl/tests/workers/config/test_rollout_config_on_cpu.py",
    "verl/tests/trainer/config/test_algo_config_on_cpu.py",
    "verl/tests/trainer/ppo/test_opd_proxy_verify_capture.py",
    "verl/tests/trainer/ppo/test_opd_proxy_verify_capture_launcher.py",
    "verl/tests/trainer/ppo/test_rollout_corr.py",
    "verl/tests/trainer/ppo/test_rollout_corr_integration.py",
    "verl/tests/workers/actor/test_length_aware_opd.py",
)
REFERENCE_SOURCE_FILES = (
    "prismatic-synthesis/gradient_modules/gradient_computer.py",
    "prismatic-synthesis/cluster_modules/cluster_manager.py",
    "g-vendi/gradient_vendi.py",
)


def canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(f"value is not canonical JSON: {error}") from error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"cannot hash file {path}: {error}") from error
    return digest.hexdigest()


def _reject_pairs(pairs):
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str):
    raise ValueError(f"non-finite JSON constant: {value}")


def load_canonical_json(path: Path, description: str) -> dict[str, object]:
    try:
        payload = Path(path).read_bytes()
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_pairs,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid {description}: {error}") from error
    if not isinstance(value, dict) or payload != canonical_json_bytes(value):
        raise ValueError(f"{description} must be a canonical JSON object")
    return value


def _atomic_write(path: Path, payload: bytes) -> None:
    target = Path(path)
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
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def write_or_validate_json(path: Path, value: Mapping[str, object]) -> None:
    payload = canonical_json_bytes(dict(value))
    target = Path(path)
    if target.exists():
        actual = load_canonical_json(target, target.name)
        if actual != dict(value) or target.read_bytes() != payload:
            raise RuntimeError(f"existing immutable contract differs: {target}")
        return
    _atomic_write(target, payload)


@dataclass(frozen=True)
class StageCommand:
    name: str
    argv: tuple[str, ...]
    interpreter: str
    completion_paths: tuple[Path, ...]
    output_root: Path
    log_path: Path
    row_count: int
    gpu_tokens: tuple[str, ...] = ()
    parallel_group: str | None = None
    output_kind: str = "generic"
    extra_env: tuple[tuple[str, str], ...] = ()
    input_hashes: tuple[tuple[str, str], ...] = ()
    validate_output: bool = True

    def __post_init__(self) -> None:
        if not self.name or not re.fullmatch(r"[a-z0-9][a-z0-9_]*", self.name):
            raise ValueError("stage command name must be normalized snake case")
        if not self.argv or any(not isinstance(value, str) or not value for value in self.argv):
            raise ValueError("stage command argv must contain nonempty strings")
        if self.interpreter not in {"verl_capture", "gvendi_analysis"}:
            raise ValueError("stage command interpreter profile is invalid")
        if not self.completion_paths:
            raise ValueError("stage command requires at least one completion path")
        if isinstance(self.row_count, bool) or not isinstance(self.row_count, int) or self.row_count <= 0:
            raise ValueError("stage command row_count must be positive")
        if len(set(self.gpu_tokens)) != len(self.gpu_tokens):
            raise ValueError("stage command GPU tokens must be unique")
        for name, value in self.input_hashes:
            if not name or _SHA256_RE.fullmatch(value) is None:
                raise ValueError("stage command input hashes must be named SHA-256 values")

    def as_dict(self) -> dict[str, object]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def contract(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "artifact_type": "opd_proxy_work_unit_contract",
            "name": self.name,
            "argv": list(self.argv),
            "interpreter": self.interpreter,
            "gpu_tokens": list(self.gpu_tokens),
            "parallel_group": self.parallel_group,
            "completion_paths": [str(path.resolve()) for path in self.completion_paths],
            "output_root": str(self.output_root.resolve()),
            "output_kind": self.output_kind,
            "row_count": self.row_count,
            "log_path": str(self.log_path.resolve()),
            "extra_env": {key: value for key, value in self.extra_env},
            "input_hashes": {key: value for key, value in self.input_hashes},
            "validate_output": self.validate_output,
        }

    @property
    def contract_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.contract())).hexdigest()


class SystemRuntime:
    """The only real subprocess and allocation-query boundary."""

    def slurm_job_info(self, job_id: str) -> str:
        try:
            result = subprocess.run(
                ["scontrol", "show", "job", "-o", job_id],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError(f"cannot query Slurm allocation: {error}") from error
        return result.stdout

    def tmux_session_name(self) -> str:
        try:
            result = subprocess.run(
                ["tmux", "display-message", "-p", "#S"],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError(f"cannot query tmux session: {error}") from error
        return result.stdout.strip()

    def gpu_processes(self, token: str) -> set[int]:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    token,
                    "--query-compute-apps=pid",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError(f"cannot query allocated GPU token {token}: {error}") from error
        pids: set[int] = set()
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if stripped and stripped != "[N/A]":
                try:
                    pids.add(int(stripped))
                except ValueError as error:
                    raise RuntimeError(f"invalid nvidia-smi PID row: {stripped}") from error
        return pids

    def start(self, command: StageCommand, *, env: Mapping[str, str], log_path: Path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("ab", buffering=0)
        process = subprocess.Popen(
            list(command.argv),
            cwd=Path.cwd(),
            env=dict(env),
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        process._opd_log_handle = handle  # type: ignore[attr-defined]
        return process

    def terminate_owned(self, process) -> None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass

    def kill_owned(self, process) -> None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass

    def finalize_owned(self, process) -> None:
        self.terminate_owned(process)
        time.sleep(0.05)
        self.kill_owned(process)

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


def _visible_tokens(env: Mapping[str, str], expected_gpus: int) -> tuple[str, ...]:
    visible = env.get("CUDA_VISIBLE_DEVICES", "")
    raw = visible.split(",") if visible else []
    tokens = tuple(value.strip() for value in raw if value.strip())
    if (
        len(tokens) != expected_gpus
        or len(set(tokens)) != expected_gpus
        or any(value != value.strip() for value in raw)
    ):
        raise RuntimeError(
            f"opd-CLI requires exactly {expected_gpus} unique allocated CUDA tokens"
        )
    return tokens


def validate_opd_cli_runtime(
    *,
    env: Mapping[str, str] | None = None,
    runtime=None,
    expected_gpus: int = 4,
    require_idle: bool = True,
    owned_pids: set[int] | None = None,
) -> tuple[str, ...]:
    runtime_env = os.environ if env is None else env
    if not runtime_env.get("SLURM_JOB_ID"):
        raise RuntimeError("opd-CLI runtime requires nonempty SLURM_JOB_ID")
    if isinstance(expected_gpus, bool) or not isinstance(expected_gpus, int) or expected_gpus <= 0:
        raise ValueError("expected_gpus must be positive")
    tokens = _visible_tokens(runtime_env, expected_gpus)
    active_runtime = SystemRuntime() if runtime is None else runtime
    if hasattr(active_runtime, "slurm_job_info"):
        job_id = runtime_env["SLURM_JOB_ID"]
        rows = [
            row.strip()
            for row in active_runtime.slurm_job_info(job_id).splitlines()
            if row.strip()
        ]
        if (
            len(rows) != 1
            or f"JobId={job_id}" not in rows[0]
            or "JobState=RUNNING" not in rows[0]
        ):
            raise RuntimeError("the declared Slurm allocation is not uniquely active")
    if active_runtime.tmux_session_name() != "opd-CLI":
        raise RuntimeError("GPU execution must run inside the opd-CLI tmux session")
    if require_idle:
        allowed = set() if owned_pids is None else set(owned_pids)
        for token in tokens:
            unexpected = set(active_runtime.gpu_processes(token)) - allowed
            if unexpected:
                raise RuntimeError(
                    f"allocated GPU token {token} is occupied by unexpected PIDs {sorted(unexpected)}"
                )
    return tokens


def _require_sha(value: object, description: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{description} must be a lowercase SHA-256 digest")
    return value


def validate_stage_manifest(manifest: Mapping[str, object], stage: int) -> None:
    if stage not in _STAGE_LAYOUTS:
        raise ValueError(f"unsupported stage: {stage}")
    if manifest.get("schema_version") != 1 or manifest.get("artifact_type") != "opd_proxy_stage":
        raise ValueError("invalid OPD proxy stage manifest contract")
    if manifest.get("stage") != stage:
        raise ValueError("stage manifest stage mismatch")
    candidate, heldout, selected, primary, diagnostic, draws = _STAGE_LAYOUTS[stage]
    expected = {
        "candidate_count": candidate,
        "held_out_count": heldout,
        "selected_size": selected,
        "primary_k": primary,
        "diagnostic_k": diagnostic,
        "null_draws": draws,
    }
    if any(manifest.get(name) != value for name, value in expected.items()):
        raise ValueError("stage manifest cardinality contract mismatch")
    for field in (
        "sample_manifest_sha256",
        "target_capture_parquet_sha256",
        "proxy_capture_parquet_sha256",
    ):
        _require_sha(manifest.get(field), field)
    target_expected = candidate + heldout if stage != 2 else 1024
    proxy_expected = candidate if stage != 2 else 768
    if manifest.get("target_capture_count") != target_expected or manifest.get(
        "proxy_capture_count"
    ) != proxy_expected:
        raise ValueError("stage capture counts differ from append-only contract")
    provenance = manifest.get("provenance")
    root = provenance.get("root_manifest") if isinstance(provenance, Mapping) else None
    models = root.get("model_manifests") if isinstance(root, Mapping) else None
    if not isinstance(models, Mapping):
        raise ValueError("stage manifest lacks root model provenance")
    required_roles = {
        "target_teacher",
        "target_student",
        "proxy_teacher",
        "proxy_student",
    }
    if set(models) != required_roles:
        raise ValueError("stage model role provenance is incomplete")
    hashes = {
        role: _require_sha(models[role].get("manifest_sha256"), f"{role} model hash")
        for role in sorted(required_roles)
        if isinstance(models[role], Mapping)
    }
    if len(hashes) != 4:
        raise ValueError("stage model manifests are malformed")
    if hashes["target_student"] != hashes["proxy_teacher"]:
        raise ValueError("target-student and proxy-teacher Qwen3-4B bytes differ")


def _model_hashes(manifest: Mapping[str, object]) -> dict[str, str]:
    models = manifest["provenance"]["root_manifest"]["model_manifests"]  # type: ignore[index]
    return {
        role: str(models[role]["manifest_sha256"])
        for role in (
            "target_teacher",
            "target_student",
            "proxy_teacher",
            "proxy_student",
        )
    }


def validate_stage_parent(stage: int, manifest: Mapping[str, object], parent_report: Path | None) -> None:
    stored = manifest.get("parent_report")
    if stage in (0, 1):
        if parent_report is not None or stored is not None:
            raise ValueError(f"Stage {stage} forbids a parent report")
        return
    if stage != 2:
        raise ValueError(f"unsupported stage: {stage}")
    if parent_report is None or not isinstance(stored, Mapping):
        raise ValueError("Stage 2 requires its exact Stage-1 parent report")
    path = Path(parent_report).resolve()
    stored_path = stored.get("path")
    if not isinstance(stored_path, str) or Path(stored_path).resolve() != path:
        raise ValueError("Stage-2 parent report path mismatch")
    expected = _require_sha(stored.get("sha256"), "Stage-2 parent report SHA")
    if sha256_file(path) != expected:
        raise ValueError("Stage-2 parent report hash mismatch")


def _path_value(path: Path) -> str:
    return str(Path(path).resolve())


def _source_snapshot_manifest_sha256(path: Path) -> str:
    value = load_canonical_json(path, "experiment source snapshot")
    files = value.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("experiment source snapshot has no source files")
    actual = hashlib.sha256(canonical_json_bytes(files)).hexdigest()
    if value.get("manifest_sha256") != actual:
        raise ValueError("experiment source snapshot logical hash mismatch")
    return actual


def build_capture_hydra_overrides(
    *,
    stage: int,
    manifest: Mapping[str, object],
    stage_directory: Path,
    pair: str,
    seed: int,
    output_root: Path,
    repository_root: Path,
) -> tuple[str, ...]:
    validate_stage_manifest(manifest, stage)
    if pair not in {"target", "proxy"}:
        raise ValueError("capture pair must be target or proxy")
    if seed not in SEEDS:
        raise ValueError("capture seed must be 42 or 43")
    stage_root = Path(stage_directory).resolve()
    repository = Path(repository_root).resolve()
    if pair == "target":
        parquet_name = manifest["target_capture_parquet"]
        question_count = int(manifest["target_capture_count"])
        student = repository / "models/Qwen3-4B"
        teacher = repository / "models/Qwen3-30B-A3B-Instruct-2507"
    else:
        parquet_name = manifest["proxy_capture_parquet"]
        question_count = int(manifest["proxy_capture_count"])
        student = repository / "models/Qwen3-0.6B"
        teacher = repository / "models/Qwen3-4B"
    sample_path = stage_root / str(manifest["sample_manifest"])
    source_path = stage_root / "source_snapshot.json"
    source_hash = (
        _source_snapshot_manifest_sha256(source_path)
        if source_path.is_file()
        else "0" * 64
    )
    train_path = stage_root / str(parquet_name)
    values = (
        ("data.train_files", _path_value(train_path)),
        ("data.val_files", "[]"),
        ("data.train_batch_size", str(question_count)),
        ("data.max_prompt_length", "2048"),
        ("data.max_response_length", "16384"),
        ("data.filter_overlong_prompts", "false"),
        ("data.truncation", "error"),
        ("data.shuffle", "false"),
        ("data.return_raw_chat", "true"),
        ("+data.apply_chat_template_kwargs.enable_thinking", "false"),
        ("trainer.nnodes", "1"),
        ("trainer.n_gpus_per_node", "4"),
        ("trainer.logger", '["console"]'),
        ("trainer.val_before_train", "false"),
        ("trainer.save_freq", "-1"),
        ("trainer.test_freq", "-1"),
        ("trainer.total_epochs", "1"),
        ("trainer.resume_mode", "disable"),
        ("actor_rollout_ref.model.path", _path_value(student)),
        ("+actor_rollout_ref.ref.model.path", _path_value(teacher)),
        ("actor_rollout_ref.model.use_remove_padding", "true"),
        ("actor_rollout_ref.rollout.n", "4"),
        ("actor_rollout_ref.rollout.name", "vllm"),
        ("actor_rollout_ref.rollout.temperature", "1.0"),
        ("actor_rollout_ref.rollout.top_p", "1.0"),
        ("actor_rollout_ref.rollout.seed", str(seed)),
        ("actor_rollout_ref.rollout.calculate_log_probs", "true"),
        ("actor_rollout_ref.rollout.tensor_model_parallel_size", "4"),
        ("actor_rollout_ref.rollout.data_parallel_size", "1"),
        ("actor_rollout_ref.rollout.pipeline_model_parallel_size", "1"),
        ("actor_rollout_ref.rollout.max_model_len", "18432"),
        ("actor_rollout_ref.rollout.max_num_batched_tokens", "32768"),
        ("actor_rollout_ref.rollout.gpu_memory_utilization", "0.6"),
        ("actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu", "1"),
        ("actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu", "1"),
        ("actor_rollout_ref.ref.fsdp_config.param_offload", "true"),
        ("actor_rollout_ref.actor.ppo_epochs", "1"),
        ("actor_rollout_ref.actor.ppo_mini_batch_size", str(question_count)),
        ("actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu", "1"),
        ("actor_rollout_ref.actor.ppo_max_token_len_per_gpu", "18432"),
        ("actor_rollout_ref.actor.use_dynamic_bsz", "false"),
        ("actor_rollout_ref.actor.opd_proxy_verify_capture_only", "true"),
        ("actor_rollout_ref.actor.loss_agg_mode", "token-mean"),
        ("actor_rollout_ref.actor.policy_loss.loss_mode", "vanilla"),
        ("actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages", "true"),
        ("actor_rollout_ref.actor.policy_loss.length_aware_opd", "false"),
        ("actor_rollout_ref.actor.policy_loss.entropy_aware_distill", "false"),
        ("actor_rollout_ref.actor.use_kl_loss", "true"),
        ("actor_rollout_ref.actor.kl_loss_coef", "0"),
        ("actor_rollout_ref.actor.entropy_coeff", "0"),
        ("algorithm.rollout_correction.rollout_is", "token"),
        ("algorithm.rollout_correction.rollout_is_threshold", "5.0"),
        ("algorithm.rollout_correction.rollout_is_batch_normalize", "false"),
        ("algorithm.rollout_correction.rollout_rs", "null"),
        ("algorithm.rollout_correction.bypass_mode", "false"),
        ("algorithm.use_kl_in_reward", "false"),
        ("algorithm.candidate_selection.enabled", "false"),
        ("algorithm.tale_budget.enabled", "false"),
        ("algorithm.difficulty_aware_opd.enabled", "false"),
        ("algorithm.rethinking_opd_probe.enabled", "false"),
        ("algorithm.opd_proxy_verify_capture.enabled", "true"),
        ("algorithm.opd_proxy_verify_capture.output_root", _path_value(output_root)),
        ("algorithm.opd_proxy_verify_capture.sample_manifest", _path_value(sample_path)),
        ("algorithm.opd_proxy_verify_capture.sample_manifest_sha256", str(manifest["sample_manifest_sha256"])),
        ("algorithm.opd_proxy_verify_capture.source_snapshot", _path_value(source_path)),
        ("algorithm.opd_proxy_verify_capture.source_snapshot_sha256", source_hash),
        ("algorithm.opd_proxy_verify_capture.stage", str(stage)),
        ("algorithm.opd_proxy_verify_capture.pair", pair),
        ("algorithm.opd_proxy_verify_capture.engine_seed", str(seed)),
        ("algorithm.opd_proxy_verify_capture.native_rollouts", "4"),
        ("algorithm.opd_proxy_verify_capture.expected_questions", str(question_count)),
        ("algorithm.opd_proxy_verify_capture.chunk_size", "16"),
        ("algorithm.opd_proxy_verify_capture.schema_version", "1"),
    )
    if len({key for key, _ in values}) != len(values):
        raise RuntimeError("capture Hydra override keys must be unique")
    return tuple(f"{key}={value}" for key, value in values)


def _official_shard_bounds(total: int, shard_index: int) -> tuple[int, int]:
    partition = int(total / REPLAY_SHARDS) + 1
    start = shard_index * partition
    return start, min(start + partition, total)


def _capture_root(stage_directory: Path, pair: str, seed: int) -> Path:
    return stage_directory / "capture" / pair / f"seed_{seed}"


def _replay_dirs(stage_directory: Path, pair: str, seed: int) -> tuple[Path, ...]:
    return tuple(
        stage_directory / "replay" / pair / f"seed_{seed}" / f"shard_{index}"
        for index in range(REPLAY_SHARDS)
    )


def _source_dirs_for_stage(
    stage: int, stage_directory: Path, representation: str
) -> tuple[Path, ...]:
    roots = [stage_directory]
    if stage == 2:
        roots.insert(0, stage_directory.parent / "stage_1")
    if representation.startswith("P_"):
        seed = int(representation.split(":")[1].split("=")[1])
        return tuple(
            path
            for root in roots
            for path in _replay_dirs(root, "proxy", seed)
        )
    if representation.startswith("T:"):
        seed = int(representation.rsplit("=", 1)[1])
        return tuple(
            path
            for root in roots
            for path in _replay_dirs(root, "target", seed)
        )
    baseline = "sft" if representation == "S" else "embedding"
    # Stage-2 baseline collectors publish an append-only Stage-1+Stage-2 union;
    # adding the Stage-1 directory again would duplicate every prior vector ID.
    return (stage_directory / "baselines" / baseline,)


def _first_candidate_id(stage_directory: Path, manifest: Mapping[str, object]) -> str:
    path = stage_directory / str(manifest["sample_manifest"])
    if not path.is_file():
        return "candidate_0"
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("split") == "candidate" and isinstance(row.get("stable_id"), str):
                    return row["stable_id"]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read first candidate ID: {error}") from error
    raise ValueError("stage sample manifest has no candidate row")


def build_stage_commands(
    *,
    stage: int,
    manifest: Mapping[str, object],
    repository_root: Path | None = None,
    stage_directory: Path | None = None,
    reference_repo: Path = DEFAULT_REFERENCE_REPO,
) -> tuple[StageCommand, ...]:
    validate_stage_manifest(manifest, stage)
    repository = Path.cwd().resolve() if repository_root is None else Path(repository_root).resolve()
    stage_root = (
        (repository / DEFAULT_OUTPUT_ROOT / f"stage_{stage}").resolve()
        if stage_directory is None
        else Path(stage_directory).resolve()
    )
    reference = Path(reference_repo).resolve()
    logs = repository / "logs/opd_proxy_gradient_verify" / f"stage_{stage}"
    source_snapshot = stage_root / "source_snapshot.json"
    hashes = _model_hashes(manifest)
    commands: list[StageCommand] = []

    all_slots = tuple(f"slot:{index}" for index in range(4))
    for pair in ("target", "proxy"):
        count = int(manifest[f"{pair}_capture_count"])
        for seed in SEEDS:
            output = _capture_root(stage_root, pair, seed)
            name = f"capture_{pair}_seed_{seed}"
            commands.append(
                StageCommand(
                    name=name,
                    argv=(
                        str((repository / CAPTURE_SCRIPT).resolve()),
                        "--stage-directory",
                        str(stage_root),
                        "--stage",
                        str(stage),
                        "--pair",
                        pair,
                        "--engine-seed",
                        str(seed),
                        "--output-root",
                        str(output),
                    ),
                    interpreter="verl_capture",
                    gpu_tokens=all_slots,
                    completion_paths=(output / "COMPLETE.json",),
                    output_root=output,
                    output_kind="capture",
                    row_count=count,
                    log_path=logs / f"{name}.log",
                )
            )

    if stage == 0:
        output = stage_root / "direct_gradient_fixture"
        commands.append(
            StageCommand(
                name="direct_gradient_fixture",
                argv=(
                    str((repository / DIRECT_FIXTURE_SCRIPT).resolve()),
                    "--stage",
                    "0",
                    "--pair",
                    "proxy",
                    "--engine-seed",
                    "42",
                    "--rollout-slot",
                    "0",
                    "--stable-id",
                    _first_candidate_id(stage_root, manifest),
                    "--capture-direct-gradient-fixture",
                    "true",
                    "--capture-root",
                    str(_capture_root(stage_root, "proxy", 42)),
                    "--model-path",
                    str(repository / "models/Qwen3-0.6B"),
                    "--output-root",
                    str(output),
                    "--expected-model-sha256",
                    hashes["proxy_student"],
                    "--reference-repo",
                    str(reference),
                ),
                interpreter="gvendi_analysis",
                gpu_tokens=("slot:0",),
                completion_paths=(output / "COMPLETE.json",),
                output_root=output,
                output_kind="direct_fixture",
                row_count=1,
                log_path=logs / "direct_gradient_fixture.log",
            )
        )

    for pair in ("target", "proxy"):
        model_role = "target_student" if pair == "target" else "proxy_student"
        model_path = repository / (
            "models/Qwen3-4B" if pair == "target" else "models/Qwen3-0.6B"
        )
        group_total = int(manifest[f"{pair}_capture_count"])
        for seed in SEEDS:
            group = f"replay_{pair}_seed_{seed}"
            for shard_index, output in enumerate(_replay_dirs(stage_root, pair, seed)):
                start, end = _official_shard_bounds(group_total, shard_index)
                if start >= end:
                    raise ValueError("replay shard layout produced an empty shard")
                name = f"{group}_shard_{shard_index}"
                commands.append(
                    StageCommand(
                        name=name,
                        argv=(
                            GVENDI_PYTHON,
                            "-m",
                            "math_eval.replay_opd_proxy_gradients",
                            "--capture-root",
                            str(_capture_root(stage_root, pair, seed)),
                            "--model-path",
                            str(model_path),
                            "--output-directory",
                            str(output),
                            "--pair",
                            pair,
                            "--num-shards",
                            str(REPLAY_SHARDS),
                            "--shard-index",
                            str(shard_index),
                            "--expected-model-sha256",
                            hashes[model_role],
                            "--source-snapshot",
                            str(source_snapshot),
                            "--reference-repo",
                            str(reference),
                            "--repository-root",
                            str(repository),
                            "--chunk-size",
                            "16",
                        ),
                        interpreter="gvendi_analysis",
                        gpu_tokens=(f"slot:{shard_index}",),
                        parallel_group=group,
                        completion_paths=(output / "COMPLETE.json",),
                        output_root=output,
                        output_kind="replay",
                        row_count=end - start,
                        log_path=logs / f"{name}.log",
                    )
                )

    for baseline, module in (
        ("sft", "math_eval.collect_opd_proxy_sft_gradients"),
        ("embedding", "math_eval.collect_opd_proxy_prompt_embeddings"),
    ):
        output = stage_root / "baselines" / baseline
        argv = [
            GVENDI_PYTHON,
            "-m",
            module,
            "--sample-manifest",
            str(stage_root / str(manifest["sample_manifest"])),
            "--model-path",
            str(repository / "models/Qwen3-0.6B"),
        ]
        if baseline == "sft":
            argv.extend(["--reference-repo", str(reference)])
        argv.extend(
            [
                "--expected-model-sha256",
                hashes["proxy_student"],
                "--output-directory",
                str(output),
                "--stage",
                str(stage),
            ]
        )
        if stage == 2:
            argv.extend(
                [
                    "--stage1-vector-directory",
                    str(stage_root.parent / "stage_1" / "baselines" / baseline),
                ]
            )
        argv.extend(
            [
                "--source-snapshot",
                str(source_snapshot),
                "--repository-root",
                str(repository),
                "--chunk-size",
                "16",
            ]
        )
        rows = 768 if stage == 2 else int(manifest["candidate_count"])
        commands.append(
            StageCommand(
                name=f"collect_{baseline}",
                argv=tuple(argv),
                interpreter="gvendi_analysis",
                gpu_tokens=("slot:0",),
                completion_paths=(output / "COMPLETE.json",),
                output_root=output,
                output_kind="baseline",
                row_count=rows,
                log_path=logs / f"collect_{baseline}.log",
            )
        )

    validated = stage_root / "vectors" / "VALIDATED.json"
    commands.append(
        StageCommand(
            name="validate_vectors",
            argv=(
                GVENDI_PYTHON,
                "-m",
                "math_eval.run_opd_proxy_gradient_verify",
                "internal-validate-vectors",
                "--stage",
                str(stage),
                "--stage-directory",
                str(stage_root),
            ),
            interpreter="gvendi_analysis",
            completion_paths=(validated,),
            output_root=validated.parent,
            output_kind="validation",
            row_count=int(manifest["candidate_count"]),
            log_path=logs / "validate_vectors.log",
        )
    )

    selection_root = stage_root / "selection"
    select_argv = [
        GVENDI_PYTHON,
        "-m",
        "math_eval.select_opd_proxy_gradient_verify",
        "--stage-directory",
        str(stage_root),
        "--output-directory",
        str(selection_root),
        "--reference-repo",
        str(reference),
        "--source-snapshot",
        str(source_snapshot),
    ]
    representations = tuple(
        [f"P_n1:seed={seed}:slot={slot}" for seed in SEEDS for slot in range(4)]
        + [f"P_n4:seed={seed}" for seed in SEEDS]
        + ["S", "E"]
        + [f"T:seed={seed}" for seed in SEEDS]
    )
    for representation in representations:
        for source in _source_dirs_for_stage(stage, stage_root, representation):
            select_argv.extend(["--vector", f"{representation}={source}"])
    commands.append(
        StageCommand(
            name="select",
            argv=tuple(select_argv),
            interpreter="gvendi_analysis",
            gpu_tokens=("slot:0",),
            completion_paths=(
                selection_root / "selection.manifest.json",
                selection_root / "random.manifest.json",
            ),
            output_root=selection_root,
            output_kind="selection",
            row_count=int(manifest["candidate_count"]),
            log_path=logs / "select.log",
        )
    )

    inputs_path = stage_root / "analysis_inputs.json"
    commands.append(
        StageCommand(
            name="prepare_analysis_inputs",
            argv=(
                GVENDI_PYTHON,
                "-m",
                "math_eval.run_opd_proxy_gradient_verify",
                "internal-prepare-analysis-inputs",
                "--stage",
                str(stage),
                "--stage-directory",
                str(stage_root),
            ),
            interpreter="gvendi_analysis",
            completion_paths=(inputs_path,),
            output_root=stage_root,
            output_kind="analysis_inputs",
            row_count=int(manifest["candidate_count"]),
            log_path=logs / "prepare_analysis_inputs.log",
        )
    )
    commands.append(
        StageCommand(
            name="analyze",
            argv=(
                GVENDI_PYTHON,
                "-m",
                "math_eval.analyze_opd_proxy_gradient_verify",
                "--stage-dir",
                str(stage_root),
                "--reference-repo",
                str(reference),
                "--output-json",
                str(stage_root / "report.json"),
                "--output-markdown",
                str(stage_root / "report.md"),
            ),
            interpreter="gvendi_analysis",
            completion_paths=(stage_root / "report.json", stage_root / "report.md"),
            output_root=stage_root,
            output_kind="analysis",
            row_count=int(manifest["candidate_count"]),
            log_path=logs / "analyze.log",
        )
    )
    stage_manifest_sha256 = hashlib.sha256(
        canonical_json_bytes(dict(manifest))
    ).hexdigest()
    source_snapshot_sha256 = (
        _source_snapshot_manifest_sha256(source_snapshot)
        if source_snapshot.is_file()
        else "0" * 64
    )
    return tuple(
        replace(
            command,
            input_hashes=(
                ("source_snapshot_sha256", source_snapshot_sha256),
                ("stage_manifest_sha256", stage_manifest_sha256),
            ),
        )
        for command in commands
    )


def _validate_live_source_snapshot(
    stage_directory: Path, command: StageCommand
) -> None:
    expected = dict(command.input_hashes).get("source_snapshot_sha256")
    if expected is None or expected == "0" * 64:
        return
    snapshot_path = Path(stage_directory) / "source_snapshot.json"
    snapshot = load_canonical_json(snapshot_path, "experiment source snapshot")
    if _source_snapshot_manifest_sha256(snapshot_path) != expected:
        raise RuntimeError("work-unit source snapshot contract changed")
    repositories = snapshot.get("repositories")
    records = snapshot.get("files")
    if not isinstance(repositories, Mapping) or not isinstance(records, list):
        raise RuntimeError("experiment source snapshot is malformed")
    roots = {
        name: Path(path).resolve()
        for name, path in repositories.items()
        if isinstance(name, str) and isinstance(path, str)
    }
    if set(roots) != set(repositories):
        raise RuntimeError("experiment source repository map is malformed")
    for record in records:
        if not isinstance(record, Mapping):
            raise RuntimeError("experiment source file record is malformed")
        repository = record.get("repository")
        logical_path = record.get("path")
        if repository not in roots or not isinstance(logical_path, str):
            raise RuntimeError("experiment source file record has invalid identity")
        logical = PurePosixPath(logical_path)
        if logical.is_absolute() or ".." in logical.parts:
            raise RuntimeError("experiment source file path is unsafe")
        path = (roots[repository] / Path(*logical.parts)).resolve()
        try:
            path.relative_to(roots[repository])
        except ValueError as error:
            raise RuntimeError("experiment source file escaped its repository") from error
        if (
            not path.is_file()
            or path.stat().st_size != record.get("size")
            or sha256_file(path) != record.get("sha256")
        ):
            raise RuntimeError(
                "experiment source file changed during the run: "
                f"{repository}:{logical_path}"
            )


def _resolve_tokens(command: StageCommand, allocated: tuple[str, ...]) -> tuple[str, ...]:
    result: list[str] = []
    for token in command.gpu_tokens:
        if token.startswith("slot:"):
            index = int(token.split(":", 1)[1])
            if index < 0 or index >= len(allocated):
                raise RuntimeError(f"GPU slot outside allocation: {token}")
            result.append(allocated[index])
        else:
            if token not in allocated:
                raise RuntimeError(f"command requests non-allocated GPU token: {token}")
            result.append(token)
    return tuple(result)


def _validate_capture_subtree(directory: Path, marker: Mapping[str, object]) -> None:
    chunks = marker.get("chunks")
    if not isinstance(chunks, list) or marker.get("chunk_count") != len(chunks):
        raise RuntimeError(f"capture subtree marker is malformed: {directory}")
    for record in chunks:
        if not isinstance(record, Mapping):
            raise RuntimeError("capture chunk record must be an object")
        for path_field, hash_field in (
            ("tensor_file", "tensor_sha256"),
            ("sidecar_file", "sidecar_sha256"),
        ):
            filename = record.get(path_field)
            expected = record.get(hash_field)
            if (
                not isinstance(filename, str)
                or Path(filename).name != filename
                or _SHA256_RE.fullmatch(str(expected)) is None
                or sha256_file(directory / filename) != expected
            ):
                raise RuntimeError("capture chunk file/hash mismatch")


def _validate_capture_output(root: Path) -> None:
    if (root / ".rollout.tmp").exists():
        raise RuntimeError("completed capture retains an unfinished rollout subtree")
    manifest = load_canonical_json(root / "manifest.json", "capture manifest")
    complete = load_canonical_json(root / "COMPLETE.json", "capture completion")
    if (
        manifest.get("artifact_type") != "opd_proxy_capture_seed"
        or complete.get("artifact_type") != "opd_proxy_capture_seed_complete"
        or complete.get("manifest_sha256") != sha256_file(root / "manifest.json")
        or manifest.get("actor_parameter_sha256_before")
        != manifest.get("actor_parameter_sha256_after")
    ):
        raise RuntimeError("capture seed completion contract mismatch")
    subtree_hashes = manifest.get("subtree_complete_sha256")
    subtree_values = manifest.get("subtree_complete")
    if not isinstance(subtree_hashes, Mapping) or not isinstance(
        subtree_values, Mapping
    ):
        raise RuntimeError("capture manifest lacks subtree completion records")
    for name, directory in (
        ("rollout", root / "rollout"),
        ("trainer_boundary", root / "trainer_boundary"),
    ):
        marker_path = directory / "COMPLETE.json"
        marker = load_canonical_json(marker_path, f"capture {name} completion")
        if (
            subtree_hashes.get(name) != sha256_file(marker_path)
            or subtree_values.get(name) != marker
        ):
            raise RuntimeError(f"capture {name} completion mismatch")
        _validate_capture_subtree(directory, marker)
    actor_path = root / "actor/COMPLETE.json"
    actor = load_canonical_json(actor_path, "capture actor completion")
    if (
        subtree_hashes.get("actor") != sha256_file(actor_path)
        or subtree_values.get("actor") != actor
    ):
        raise RuntimeError("capture actor completion mismatch")
    ranks = actor.get("ranks")
    if not isinstance(ranks, Mapping) or not ranks:
        raise RuntimeError("capture actor completion lacks rank records")
    for rank, record in ranks.items():
        if not isinstance(record, Mapping):
            raise RuntimeError("capture actor rank record is malformed")
        directory = root / "actor" / f"rank_{rank}"
        marker_path = directory / "COMPLETE.json"
        marker = load_canonical_json(marker_path, "capture actor rank completion")
        if (
            record.get("complete_sha256") != sha256_file(marker_path)
            or record.get("complete") != marker
        ):
            raise RuntimeError("capture actor rank completion mismatch")
        _validate_capture_subtree(directory, marker)


def _require_output_source_parent(
    command: StageCommand, manifest: Mapping[str, object]
) -> None:
    expected = dict(command.input_hashes).get("source_snapshot_sha256")
    if expected is None or expected == "0" * 64:
        return
    parents = manifest.get("parent_hashes")
    if not isinstance(parents, Mapping) or parents.get(
        "source_snapshot_sha256"
    ) != expected:
        raise RuntimeError(
            f"completed work unit {command.name} differs from its source snapshot"
        )


def _validate_completed_output(command: StageCommand) -> None:
    if not command.validate_output:
        return
    if command.output_kind == "capture":
        _validate_capture_output(command.output_root)
        _require_output_source_parent(
            command,
            load_canonical_json(
                command.output_root / "manifest.json", "capture manifest"
            ),
        )
        return
    if command.output_kind in {"replay", "baseline"}:
        from math_eval.opd_proxy_gradient_verify_artifacts import load_vector_set

        vector_set = load_vector_set(command.output_root)
        _require_output_source_parent(command, vector_set.manifest)
        return
    if command.output_kind == "direct_fixture":
        manifest = load_canonical_json(
            command.output_root / "manifest.json", "direct fixture manifest"
        )
        complete = load_canonical_json(
            command.output_root / "COMPLETE.json", "direct fixture completion"
        )
        tensor = command.output_root / str(manifest.get("tensor_file"))
        if (
            complete.get("manifest_sha256")
            != sha256_file(command.output_root / "manifest.json")
            or manifest.get("tensor_sha256") != sha256_file(tensor)
        ):
            raise RuntimeError("direct fixture completion mismatch")
        _require_output_source_parent(command, manifest)
        return
    if command.output_kind == "selection":
        from math_eval.select_opd_proxy_gradient_verify import (
            load_random_schedules,
            load_selection_bundle,
        )

        selection = load_selection_bundle(command.output_root)
        random = load_random_schedules(command.output_root)
        _require_output_source_parent(command, selection.manifest)
        _require_output_source_parent(command, random.manifest)
        return
    if command.output_kind == "analysis":
        stage = int(command.output_root.name.rsplit("_", 1)[-1])
        report = _validate_report_pair(stage, command.output_root)
        expected = dict(command.input_hashes).get("source_snapshot_sha256")
        if report.get("provenance", {}).get("source_snapshot_sha256") != expected:
            raise RuntimeError("analysis report source snapshot mismatch")
        return
    if command.output_kind == "analysis_inputs":
        inputs = load_canonical_json(
            command.completion_paths[0], "analysis input manifest"
        )
        expected = dict(command.input_hashes).get("source_snapshot_sha256")
        if inputs.get("source_snapshot_sha256") != expected:
            raise RuntimeError("analysis inputs source snapshot mismatch")
        return
    for path in command.completion_paths:
        if path.suffix == ".json":
            load_canonical_json(path, f"{command.name} completion")


def _completion_records(command: StageCommand) -> list[dict[str, object]]:
    records = []
    for path in command.completion_paths:
        if not path.is_file():
            raise RuntimeError(
                f"work unit {command.name} exited without completion artifact {path}"
            )
        records.append(
            {
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def _guard_capture_resume(command: StageCommand) -> None:
    if command.output_kind != "capture" or all(
        path.exists() for path in command.completion_paths
    ):
        return
    root = command.output_root
    if not root.exists():
        return
    unfinished = list(root.rglob("*.rollout.tmp"))
    rollout = root / "rollout"
    if unfinished or (rollout.exists() and not (rollout / "COMPLETE.json").is_file()):
        raise RuntimeError(
            f"capture work unit {command.name} has incomplete rollout generation"
        )


def _work_paths(stage_directory: Path, command: StageCommand) -> tuple[Path, Path]:
    root = stage_directory / "work_units"
    return root / f"{command.name}.contract.json", root / f"{command.name}.complete.json"


def _prepare_unit(stage_directory: Path, command: StageCommand) -> bool:
    contract_path, complete_path = _work_paths(stage_directory, command)
    write_or_validate_json(contract_path, command.contract())
    _guard_capture_resume(command)
    completions_exist = [path.is_file() for path in command.completion_paths]
    if complete_path.exists():
        _validate_completed_output(command)
        complete = load_canonical_json(complete_path, f"{command.name} completion")
        if complete.get("contract_sha256") != command.contract_sha256:
            raise RuntimeError(f"work unit {command.name} completion contract mismatch")
        actual = _completion_records(command)
        if complete.get("completion_artifacts") != actual:
            raise RuntimeError(f"work unit {command.name} completion artifact mismatch")
        return True
    if any(completions_exist):
        if not all(completions_exist):
            raise RuntimeError(f"work unit {command.name} has partial completion markers")
        _validate_completed_output(command)
        # Adopt a fully published immutable output after an orchestrator-only
        # interruption between child completion and ledger publication.
        complete = {
            "schema_version": 1,
            "artifact_type": "opd_proxy_work_unit_complete",
            "name": command.name,
            "contract_sha256": command.contract_sha256,
            "completion_artifacts": _completion_records(command),
            "duration_seconds": 0.0,
            "completed_rows": command.row_count,
            "adopted_after_orchestrator_interruption": True,
        }
        write_or_validate_json(complete_path, complete)
        return True
    return False


def _unit_environment(
    base: Mapping[str, str], command: StageCommand, tokens: tuple[str, ...]
) -> dict[str, str]:
    env = dict(base)
    env.update({key: value for key, value in command.extra_env})
    # This NVIDIA allocation exports ROCm aliases too. VERL/Ray correctly
    # rejects simultaneous CUDA and ROCR/HIP visibility because their index
    # spaces compose differently. Child work units use only the allocated CUDA
    # tokens resolved above; remove the irrelevant aliases at this boundary.
    env.pop("ROCR_VISIBLE_DEVICES", None)
    env.pop("HIP_VISIBLE_DEVICES", None)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(tokens)
    env["PYTHONPATH"] = "verl:." + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
    )
    return env


def _wait_for_gpu_idle(
    tokens: Sequence[str],
    *,
    runtime,
    allowed_pids: set[int] | frozenset[int] = frozenset(),
    timeout_seconds: float = 120.0,
    poll_seconds: float = 1.0,
) -> None:
    if timeout_seconds <= 0 or poll_seconds <= 0:
        raise ValueError("GPU drain timeout and poll interval must be positive")
    unique_tokens = tuple(dict.fromkeys(tokens))
    if not unique_tokens:
        return
    deadline = runtime.monotonic() + timeout_seconds
    while True:
        occupied = {
            token: sorted(set(runtime.gpu_processes(token)) - set(allowed_pids))
            for token in unique_tokens
        }
        occupied = {token: pids for token, pids in occupied.items() if pids}
        if not occupied:
            return
        if runtime.monotonic() >= deadline:
            raise RuntimeError(
                f"launcher-owned GPU processes did not drain before timeout: {occupied}"
            )
        if hasattr(runtime, "sleep"):
            runtime.sleep(poll_seconds)
        else:
            time.sleep(poll_seconds)


def _finalize_process_group(process, runtime) -> None:
    if hasattr(runtime, "finalize_owned"):
        runtime.finalize_owned(process)
    elif hasattr(runtime, "terminate_owned"):
        runtime.terminate_owned(process)


def _close_process_log(process) -> None:
    handle = getattr(process, "_opd_log_handle", None)
    if handle is not None:
        handle.close()


def _publish_unit_completion(
    stage_directory: Path,
    command: StageCommand,
    *,
    duration: float,
) -> None:
    _, complete_path = _work_paths(stage_directory, command)
    _validate_completed_output(command)
    complete = {
        "schema_version": 1,
        "artifact_type": "opd_proxy_work_unit_complete",
        "name": command.name,
        "contract_sha256": command.contract_sha256,
        "completion_artifacts": _completion_records(command),
        "duration_seconds": float(duration),
        "completed_rows": command.row_count,
        "adopted_after_orchestrator_interruption": False,
    }
    write_or_validate_json(complete_path, complete)


def _terminate_processes(processes: Sequence[object], runtime=None) -> None:
    for process in processes:
        try:
            if hasattr(runtime, "terminate_owned"):
                runtime.terminate_owned(process)
            elif process.poll() is None:
                process.terminate()
        except (OSError, ProcessLookupError):
            pass
    for process in processes:
        try:
            if hasattr(runtime, "kill_owned"):
                runtime.kill_owned(process)
            elif process.poll() is None:
                process.kill()
        except (OSError, ProcessLookupError):
            pass
        _close_process_log(process)


def _capture_rollout_hashes(stage_root: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for pair in ("target", "proxy"):
        for seed in SEEDS:
            path = _capture_root(stage_root, pair, seed) / "rollout/COMPLETE.json"
            if not path.is_file():
                raise RuntimeError(
                    "resume exercise requires every immutable rollout completion marker"
                )
            records[f"{pair}_seed_{seed}"] = sha256_file(path)
    return records


def _exercise_replay_interruption(
    *,
    stage_root: Path,
    command: StageCommand,
    allocated: tuple[str, ...],
    runtime_env: Mapping[str, str],
    active_runtime,
    owned_pids: set[int],
) -> None:
    evidence_path = stage_root / "resume_exercise.interrupted.json"
    capture_hashes = _capture_rollout_hashes(stage_root)
    if evidence_path.exists():
        evidence = load_canonical_json(evidence_path, "resume interruption evidence")
        if (
            evidence.get("capture_rollout_sha256") != capture_hashes
            or evidence.get("replay_work_unit") != command.name
            or (command.output_root / "COMPLETE.json").exists()
        ):
            raise RuntimeError("resume interruption evidence no longer matches artifacts")
        return
    if (command.output_root / "COMPLETE.json").exists():
        raise RuntimeError(
            "cannot exercise replay resume after the selected shard is already complete"
        )
    tokens = _resolve_tokens(command, allocated)
    for token in tokens:
        unexpected = set(active_runtime.gpu_processes(token)) - owned_pids
        if unexpected:
            raise RuntimeError(
                f"GPU token {token} is occupied before resume exercise: {sorted(unexpected)}"
            )
    interrupt = StageCommand(
        **{
            **command.as_dict(),
            "name": f"{command.name}_intentional_interrupt",
            "argv": (*command.argv, "--exercise-interrupt-after-chunks", "1"),
            "log_path": command.log_path.with_name(
                f"{command.name}.resume_interrupt.log"
            ),
        }
    )
    process = active_runtime.start(
        interrupt,
        env=_unit_environment(runtime_env, interrupt, tokens),
        log_path=interrupt.log_path,
    )
    owned_pids.add(process.pid)
    returncode = process.wait()
    owned_pids.discard(process.pid)
    _finalize_process_group(process, active_runtime)
    _close_process_log(process)
    _wait_for_gpu_idle(
        tokens,
        runtime=active_runtime,
        allowed_pids=owned_pids,
    )
    if returncode == 0:
        raise RuntimeError("resume exercise unexpectedly completed the replay shard")
    if (command.output_root / "COMPLETE.json").exists():
        raise RuntimeError("resume exercise published a full replay completion")
    markers = sorted(command.output_root.glob(".vectors_*_*.complete.json"))
    if not markers:
        raise RuntimeError("resume exercise stopped before an immutable chunk was published")
    if _capture_rollout_hashes(stage_root) != capture_hashes:
        raise RuntimeError("resume exercise changed immutable rollout bytes")
    write_or_validate_json(
        evidence_path,
        {
            "schema_version": 1,
            "artifact_type": "opd_proxy_resume_interruption",
            "replay_work_unit": command.name,
            "capture_rollout_sha256": capture_hashes,
            "published_chunk_markers": [
                {"path": path.name, "sha256": sha256_file(path)} for path in markers
            ],
        },
    )


def execute_stage_commands(
    commands: Sequence[StageCommand],
    *,
    stage_directory: Path,
    env: Mapping[str, str] | None = None,
    runtime=None,
    initially_owned_pids: set[int] | None = None,
    exercise_resume: bool = False,
) -> dict[str, object]:
    """Execute immutable work units, parallelizing only declared replay groups."""
    stage_root = Path(stage_directory).resolve()
    stage_root.mkdir(parents=True, exist_ok=True)
    runtime_env = dict(os.environ if env is None else env)
    active_runtime = SystemRuntime() if runtime is None else runtime
    owned_pids = set() if initially_owned_pids is None else set(initially_owned_pids)
    allocated = validate_opd_cli_runtime(
        env=runtime_env,
        runtime=active_runtime,
        expected_gpus=4,
        require_idle=True,
        owned_pids=owned_pids,
    )
    lock_path = stage_root / ".orchestrator.lock"
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"stage output lock is already held: {lock_path}") from error
        index = 0
        completed_names: list[str] = []
        resume_exercised = False
        while index < len(commands):
            command = commands[index]
            if command.parallel_group is None:
                if _prepare_unit(stage_root, command):
                    completed_names.append(command.name)
                    index += 1
                    continue
                _validate_live_source_snapshot(stage_root, command)
                tokens = _resolve_tokens(command, allocated)
                if tokens:
                    for token in tokens:
                        unexpected = set(active_runtime.gpu_processes(token)) - owned_pids
                        if unexpected:
                            raise RuntimeError(
                                f"GPU token {token} is occupied immediately before {command.name}: {sorted(unexpected)}"
                            )
                start = active_runtime.monotonic()
                process = active_runtime.start(
                    command,
                    env=_unit_environment(runtime_env, command, tokens),
                    log_path=command.log_path,
                )
                owned_pids.add(process.pid)
                returncode = process.wait()
                owned_pids.discard(process.pid)
                _finalize_process_group(process, active_runtime)
                _close_process_log(process)
                if returncode != 0:
                    _terminate_processes([process], active_runtime)
                    _wait_for_gpu_idle(
                        tokens,
                        runtime=active_runtime,
                        allowed_pids=owned_pids,
                    )
                    raise RuntimeError(
                        f"work unit {command.name} failed with exit code {returncode}"
                    )
                _wait_for_gpu_idle(
                    tokens,
                    runtime=active_runtime,
                    allowed_pids=owned_pids,
                )
                duration = active_runtime.monotonic() - start
                _validate_live_source_snapshot(stage_root, command)
                _publish_unit_completion(stage_root, command, duration=duration)
                completed_names.append(command.name)
                index += 1
                continue

            group_name = command.parallel_group
            group: list[StageCommand] = []
            while index < len(commands) and commands[index].parallel_group == group_name:
                group.append(commands[index])
                index += 1
            pending = [item for item in group if not _prepare_unit(stage_root, item)]
            completed_names.extend(item.name for item in group if item not in pending)
            evidence_path = stage_root / "resume_exercise.interrupted.json"
            if exercise_resume and not resume_exercised and evidence_path.is_file():
                evidence = load_canonical_json(
                    evidence_path, "resume interruption evidence"
                )
                if any(
                    item.name == evidence.get("replay_work_unit") for item in group
                ):
                    resume_exercised = True
            selected_resume = next(
                (
                    item
                    for item in pending
                    if item.name == "replay_proxy_seed_42_shard_0"
                ),
                None,
            )
            if exercise_resume and not resume_exercised and selected_resume is not None:
                _validate_live_source_snapshot(stage_root, selected_resume)
                _exercise_replay_interruption(
                    stage_root=stage_root,
                    command=selected_resume,
                    allocated=allocated,
                    runtime_env=runtime_env,
                    active_runtime=active_runtime,
                    owned_pids=owned_pids,
                )
                resume_exercised = True
            processes: list[tuple[StageCommand, object, float]] = []
            try:
                for item in pending:
                    _validate_live_source_snapshot(stage_root, item)
                    tokens = _resolve_tokens(item, allocated)
                    for token in tokens:
                        unexpected = set(active_runtime.gpu_processes(token)) - owned_pids
                        if unexpected:
                            raise RuntimeError(
                                f"GPU token {token} is occupied immediately before {item.name}: {sorted(unexpected)}"
                            )
                    started = active_runtime.monotonic()
                    process = active_runtime.start(
                        item,
                        env=_unit_environment(runtime_env, item, tokens),
                        log_path=item.log_path,
                    )
                    owned_pids.add(process.pid)
                    processes.append((item, process, started))
                for item, process, started in processes:
                    returncode = process.wait()
                    owned_pids.discard(process.pid)
                    _finalize_process_group(process, active_runtime)
                    _close_process_log(process)
                    if returncode != 0:
                        raise RuntimeError(
                            f"work unit {item.name} failed with exit code {returncode}"
                        )
                    _wait_for_gpu_idle(
                        _resolve_tokens(item, allocated),
                        runtime=active_runtime,
                        allowed_pids=owned_pids,
                    )
                    _validate_live_source_snapshot(stage_root, item)
                    _publish_unit_completion(
                        stage_root,
                        item,
                        duration=active_runtime.monotonic() - started,
                    )
                    completed_names.append(item.name)
            except BaseException:
                _terminate_processes(
                    [process for _, process, _ in processes], active_runtime
                )
                _wait_for_gpu_idle(
                    tuple(
                        token
                        for item, _, _ in processes
                        for token in _resolve_tokens(item, allocated)
                    ),
                    runtime=active_runtime,
                    allowed_pids=owned_pids,
                )
                raise
        for command in commands:
            _validate_live_source_snapshot(stage_root, command)
        if exercise_resume:
            evidence_path = stage_root / "resume_exercise.interrupted.json"
            if not resume_exercised or not evidence_path.is_file():
                raise RuntimeError("requested replay resume exercise was not performed")
            evidence = load_canonical_json(evidence_path, "resume interruption evidence")
            replay_name = evidence.get("replay_work_unit")
            replay = next(
                (command for command in commands if command.name == replay_name), None
            )
            if replay is None or not (replay.output_root / "COMPLETE.json").is_file():
                raise RuntimeError("resume exercise replay shard did not complete after restart")
            capture_hashes = _capture_rollout_hashes(stage_root)
            if evidence.get("capture_rollout_sha256") != capture_hashes:
                raise RuntimeError("resumed execution changed immutable rollout hashes")
            write_or_validate_json(
                stage_root / "resume_exercise.json",
                {
                    "schema_version": 1,
                    "artifact_type": "opd_proxy_resume_exercise_complete",
                    "interruption_evidence_sha256": sha256_file(evidence_path),
                    "replay_complete_sha256": sha256_file(
                        replay.output_root / "COMPLETE.json"
                    ),
                    "capture_rollout_sha256": capture_hashes,
                },
            )
        return {"completed_or_skipped": completed_names}
    finally:
        os.close(lock_fd)


def build_source_snapshot(
    repositories: Mapping[str, Path], specs: Sequence[tuple[str, str]]
) -> dict[str, object]:
    if not repositories:
        raise ValueError("source snapshot requires repositories")
    roots = {name: Path(path).resolve() for name, path in repositories.items()}
    if any(not name or not root.is_dir() for name, root in roots.items()):
        raise ValueError("source snapshot repositories must be named directories")
    ordered = sorted(tuple(spec) for spec in specs)
    if len(set(ordered)) != len(ordered):
        raise ValueError("source snapshot contains duplicate specifications")
    records: list[dict[str, object]] = []
    for repository, logical_path in ordered:
        if repository not in roots:
            raise ValueError(f"unknown source repository: {repository}")
        logical = PurePosixPath(logical_path)
        if logical.is_absolute() or ".." in logical.parts:
            raise ValueError("source snapshot path must be relative")
        path = (roots[repository] / Path(*logical.parts)).resolve()
        try:
            path.relative_to(roots[repository])
        except ValueError as error:
            raise ValueError(f"source path escapes repository: {logical_path}") from error
        if not path.is_file():
            raise ValueError(f"source file does not exist: {repository}:{logical_path}")
        before = path.stat()
        digest = sha256_file(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"source file changed while hashing: {logical_path}")
        records.append(
            {
                "repository": repository,
                "path": logical.as_posix(),
                "size": after.st_size,
                "sha256": digest,
            }
        )
    return {
        "repositories": {name: str(roots[name]) for name in sorted(roots)},
        "files": records,
        "manifest_sha256": hashlib.sha256(canonical_json_bytes(records)).hexdigest(),
    }


def validate_main_repository(
    repository_root: Path, manifest: Mapping[str, object]
) -> dict[str, str]:
    repository = Path(repository_root).resolve()

    def git(*arguments: str) -> str:
        try:
            result = subprocess.run(
                ["git", "-C", str(repository), *arguments],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise ValueError(f"invalid source repository: {error}") from error
        return result.stdout

    head = git("rev-parse", "HEAD").strip()
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    publication = manifest.get("provenance")
    publication = publication.get("publication") if isinstance(publication, Mapping) else None
    frozen_repository = (
        publication.get("repository") if isinstance(publication, Mapping) else None
    )
    if not isinstance(frozen_repository, Mapping):
        raise ValueError("stage manifest lacks canonical source publication provenance")
    if frozen_repository.get("head") != head or status:
        raise ValueError("current source repository differs from clean frozen publication")
    return {"head": head, "status": status}


def validate_reference_repository(reference_repo: Path) -> dict[str, str]:
    repository = Path(reference_repo).resolve()

    def git(*arguments: str) -> str:
        try:
            result = subprocess.run(
                ["git", "-C", str(repository), *arguments],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise ValueError(f"invalid Prismatic reference repository: {error}") from error
        return result.stdout.strip()

    commit = git("rev-parse", "HEAD")
    tree = git("rev-parse", "HEAD^{tree}")
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    if commit != REFERENCE_COMMIT or tree != REFERENCE_TREE or status:
        raise ValueError("Prismatic reference commit/tree/cleanliness mismatch")
    return {"commit": commit, "tree": tree}


def build_experiment_source_snapshot(
    repository_root: Path, reference_repo: Path
) -> dict[str, object]:
    specs = tuple(("main", path) for path in MAIN_SOURCE_FILES) + tuple(
        ("reference", path) for path in REFERENCE_SOURCE_FILES
    )
    return build_source_snapshot(
        {"main": Path(repository_root), "reference": Path(reference_repo)}, specs
    )


def estimate_remaining_seconds(
    *,
    timing_records: Sequence[Mapping[str, object]],
    commands: Sequence[StageCommand],
    safety_factor: float,
    reserve_seconds: int,
) -> dict[str, float]:
    if not timing_records:
        raise ValueError("estimate requires Stage-0 timing records")
    if not math.isfinite(float(safety_factor)) or safety_factor <= 0:
        raise ValueError("safety_factor must be positive")
    if isinstance(reserve_seconds, bool) or not isinstance(reserve_seconds, int) or reserve_seconds < 0:
        raise ValueError("reserve_seconds must be nonnegative")
    rates: list[float] = []
    for record in timing_records:
        duration = record.get("duration_seconds")
        rows = record.get("completed_rows")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or float(duration) <= 0
            or isinstance(rows, bool)
            or not isinstance(rows, int)
            or rows <= 0
        ):
            raise ValueError("invalid Stage-0 timing record")
        rates.append(float(duration) / rows)
    slowest = max(rates)
    work_seconds = slowest * sum(command.row_count for command in commands)
    return {
        "slowest_seconds_per_row": slowest,
        "work_seconds": work_seconds,
        "safety_factor": float(safety_factor),
        "reserve_seconds": float(reserve_seconds),
        "total_seconds": work_seconds * float(safety_factor) + reserve_seconds,
    }


def _read_stage_manifest(stage_directory: Path, stage: int, *, require_files: bool) -> dict[str, object]:
    manifest = load_canonical_json(stage_directory / "manifest.json", "stage manifest")
    validate_stage_manifest(manifest, stage)
    if require_files:
        for path_field, hash_field in (
            ("sample_manifest", "sample_manifest_sha256"),
            ("target_capture_parquet", "target_capture_parquet_sha256"),
            ("proxy_capture_parquet", "proxy_capture_parquet_sha256"),
        ):
            path = stage_directory / str(manifest[path_field])
            if sha256_file(path) != manifest[hash_field]:
                raise ValueError(f"stage artifact hash mismatch: {path_field}")
    return manifest


def _representation_names() -> tuple[str, ...]:
    return tuple(
        [f"P_n1:seed={seed}:slot={slot}" for seed in SEEDS for slot in range(4)]
        + [f"P_n4:seed={seed}" for seed in SEEDS]
        + ["S", "E"]
        + [f"T:seed={seed}" for seed in SEEDS]
    )


def _load_stage_ids(stage_directory: Path, manifest: Mapping[str, object]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    candidates: list[str] = []
    heldout: list[str] = []
    with (stage_directory / str(manifest["sample_manifest"])).open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            stable_id = row.get("stable_id")
            split = row.get("split")
            if not isinstance(stable_id, str) or split not in {"candidate", "held_out"}:
                raise ValueError("invalid stage sample row")
            (candidates if split == "candidate" else heldout).append(stable_id)
    if len(candidates) != int(manifest["candidate_count"]) or len(heldout) != int(
        manifest["held_out_count"]
    ):
        raise ValueError("stage sample ID counts mismatch")
    return tuple(candidates), tuple(heldout)


def _internal_validate_vectors(stage: int, stage_directory: Path) -> None:
    from math_eval.opd_proxy_gradient_verify_artifacts import (
        canonical_json_bytes as artifact_json_bytes,
        load_vector_set,
    )
    from math_eval.select_opd_proxy_gradient_verify import (
        build_selection_vector_view,
    )

    stage_root = Path(stage_directory).resolve()
    manifest = _read_stage_manifest(stage_root, stage, require_files=True)
    candidate_ids, heldout_ids = _load_stage_ids(stage_root, manifest)
    expected_source_hash = _source_snapshot_manifest_sha256(
        stage_root / "source_snapshot.json"
    )

    def validate_source(vector_set, description: str) -> None:
        parents = vector_set.manifest.get("parent_hashes")
        if not isinstance(parents, Mapping) or parents.get(
            "source_snapshot_sha256"
        ) != expected_source_hash:
            raise ValueError(
                f"{description} vector source differs from experiment source snapshot"
            )

    view_hashes: dict[str, str] = {}
    for name in _representation_names():
        expected_representation = (
            "P" if name.startswith("P_") else "T" if name.startswith("T:") else name
        )
        sources = [
            load_vector_set(path, expected_representation=expected_representation)
            for path in _source_dirs_for_stage(stage, stage_root, name)
        ]
        for source in sources:
            validate_source(source, name)
        view = build_selection_vector_view(sources, name, candidate_ids)
        view_hashes[name] = hashlib.sha256(
            artifact_json_bytes(view.manifest)
        ).hexdigest()
    for seed in SEEDS:
        name = f"T:seed={seed}"
        sources = [
            load_vector_set(path, expected_representation="T")
            for path in _source_dirs_for_stage(stage, stage_root, name)
        ]
        for source in sources:
            validate_source(source, f"{name} held-out")
        build_selection_vector_view(sources, name, heldout_ids)
    marker = {
        "schema_version": 1,
        "artifact_type": "opd_proxy_vectors_validated",
        "stage": stage,
        "candidate_count": len(candidate_ids),
        "held_out_count": len(heldout_ids),
        "representation_view_manifest_sha256": view_hashes,
    }
    write_or_validate_json(stage_root / "vectors/VALIDATED.json", marker)


def _internal_prepare_analysis_inputs(stage: int, stage_directory: Path) -> None:
    from math_eval.analyze_opd_proxy_gradient_verify import (
        write_analysis_input_manifest,
    )

    stage_root = Path(stage_directory).resolve()
    write_analysis_input_manifest(
        stage_dir=stage_root,
        selection_directory=stage_root / "selection",
        representation_sources={
            name: _source_dirs_for_stage(stage, stage_root, name)
            for name in _representation_names()
        },
        target_heldout_sources={
            seed: _source_dirs_for_stage(stage, stage_root, f"T:seed={seed}")
            for seed in SEEDS
        },
    )


def _capture_work_unit(args) -> int:
    stage_root = Path(args.stage_directory).resolve()
    manifest = _read_stage_manifest(stage_root, args.stage, require_files=True)
    expected_output = _capture_root(stage_root, args.pair, args.engine_seed).resolve()
    if Path(args.output_root).resolve() != expected_output:
        raise ValueError("capture output root differs from canonical pair/seed path")
    _source_snapshot_manifest_sha256(stage_root / "source_snapshot.json")
    overrides = build_capture_hydra_overrides(
        stage=args.stage,
        manifest=manifest,
        stage_directory=stage_root,
        pair=args.pair,
        seed=args.engine_seed,
        output_root=Path(args.output_root),
        repository_root=Path.cwd(),
    )
    argv = [VERL_PYTHON, "-m", "verl.trainer.main_ppo", *overrides]
    os.execv(VERL_PYTHON, argv)
    raise AssertionError("unreachable")


def _validate_model_contract(repository: Path, manifest: Mapping[str, object]) -> None:
    from math_eval.opd_proxy_gradient_verify_artifacts import recursive_file_manifest

    hashes = _model_hashes(manifest)
    paths = {
        "target_teacher": repository / "models/Qwen3-30B-A3B-Instruct-2507",
        "target_student": repository / "models/Qwen3-4B",
        "proxy_teacher": repository / "models/Qwen3-4B",
        "proxy_student": repository / "models/Qwen3-0.6B",
    }
    cache: dict[Path, str] = {}
    actual: dict[str, str] = {}
    for role, path in paths.items():
        resolved = path.resolve()
        if resolved not in cache:
            cache[resolved] = str(
                recursive_file_manifest(resolved)["manifest_sha256"]
            )
        actual[role] = cache[resolved]
    if actual != hashes:
        raise ValueError("materialized model bytes differ from frozen root manifests")
    if actual["target_student"] != actual["proxy_teacher"]:
        raise ValueError("the two Qwen3-4B roles are not byte-identical")


def validate_stage_prerequisite(stage: int, output_root: Path) -> None:
    if stage == 0:
        return
    previous_root = output_root / f"stage_{stage - 1}"
    previous = previous_root / "STAGE_COMPLETE.json"
    if not previous.is_file():
        raise RuntimeError(f"Stage {stage} requires completed Stage {stage - 1}")
    marker = load_canonical_json(previous, "previous stage completion")
    if marker.get("stage") != stage - 1:
        raise RuntimeError("previous stage completion marker has wrong stage")
    if marker.get("report_sha256") != sha256_file(previous_root / "report.json"):
        raise RuntimeError("previous stage report differs from completion marker")
    previous_source = previous_root / "source_snapshot.json"
    if marker.get(
        "source_snapshot_sha256"
    ) != _source_snapshot_manifest_sha256(previous_source) or marker.get(
        "source_snapshot_file_sha256"
    ) != sha256_file(previous_source):
        raise RuntimeError("previous stage source snapshot differs from completion marker")
    if stage == 1:
        resume_path = previous_root / "resume_exercise.json"
        direct_path = previous_root / "direct_gradient_fixture/COMPLETE.json"
        if (
            marker.get("resume_exercise_sha256") != sha256_file(resume_path)
            or marker.get("direct_fixture_complete_sha256")
            != sha256_file(direct_path)
        ):
            raise RuntimeError("Stage 0 smoke/resume evidence is incomplete")


def validate_cross_stage_source(
    stage: int, output_root: Path, current_source_sha256: str
) -> None:
    if stage == 0:
        return
    _require_sha(current_source_sha256, "current source snapshot SHA")
    previous = load_canonical_json(
        Path(output_root) / f"stage_{stage - 1}" / "STAGE_COMPLETE.json",
        "previous stage completion",
    )
    if previous.get("source_snapshot_sha256") != current_source_sha256:
        raise RuntimeError("scientific source bytes differ across stages")


def validate_stage2_trigger(parent_report: Path) -> None:
    report = load_canonical_json(parent_report, "Stage-1 parent report")
    classification = report.get("classification")
    if not isinstance(classification, Mapping):
        raise RuntimeError("Stage-1 parent report lacks classification")
    oracle = classification.get("oracle")
    if not isinstance(oracle, Mapping) or oracle.get("classification") != "pass":
        raise RuntimeError("Stage 2 is forbidden because the Stage-1 oracle did not pass")
    try:
        median = float(
            classification["P_n1"]["components"]["g_vendi"]["median"]  # type: ignore[index]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("Stage-1 parent report lacks the P_n1 median") from error
    if not math.isfinite(median) or not 0.90 <= median < 0.95:
        raise RuntimeError("Stage-1 report does not satisfy the Stage-2 trigger")


def _validate_report_pair(stage: int, stage_root: Path) -> dict[str, object]:
    from math_eval.analyze_opd_proxy_gradient_verify import render_markdown

    report_path = stage_root / "report.json"
    markdown_path = stage_root / "report.md"
    report = load_canonical_json(report_path, "analysis report")
    if (
        report.get("artifact_type")
        != "opd_proxy_gradient_verification_report"
        or report.get("stage") != stage
    ):
        raise RuntimeError("analysis report stage/contract mismatch")
    if markdown_path.read_text(encoding="utf-8") != render_markdown(report):
        raise RuntimeError("analysis JSON and Markdown reports disagree")
    if stage == 0 and report.get("classification") != {"status": "smoke_only"}:
        raise RuntimeError("Stage-0 report must remain smoke-only")
    return report


def _validate_stage0_direct_fixture(
    stage_root: Path, manifest: Mapping[str, object]
) -> None:
    import numpy as np
    from safetensors.torch import load_file

    from math_eval.opd_proxy_gradient_verify_artifacts import load_vector_set
    from math_eval.select_opd_proxy_gradient_verify import (
        build_selection_vector_view,
        selection_vector_id,
    )

    direct_root = stage_root / "direct_gradient_fixture"
    direct = load_canonical_json(direct_root / "manifest.json", "direct fixture")
    complete = load_canonical_json(
        direct_root / "COMPLETE.json", "direct fixture completion"
    )
    if (
        direct.get("artifact_type") != "opd_proxy_direct_gradient_fixture"
        or complete.get("manifest_sha256")
        != sha256_file(direct_root / "manifest.json")
        or direct.get("optimizer_step_called") is not False
        or direct.get("actor_parameter_sha256_before")
        != direct.get("actor_parameter_sha256_after")
        or direct.get("loss_mode") != "vanilla"
        or direct.get("loss_agg_mode") != "token-mean"
    ):
        raise RuntimeError("direct gradient fixture semantic contract failed")
    tensor_path = direct_root / str(direct["tensor_file"])
    tensor_sha = sha256_file(tensor_path)
    if direct.get("tensor_sha256") != tensor_sha or complete.get(
        "tensor_sha256"
    ) != tensor_sha:
        raise RuntimeError("direct fixture tensor hashes do not match")
    tensors = load_file(tensor_path, device="cpu")
    candidates, _ = _load_stage_ids(stage_root, manifest)
    name = "P_n1:seed=42:slot=0"
    source_paths = _source_dirs_for_stage(0, stage_root, name)
    sources = [load_vector_set(path, expected_representation="P") for path in source_paths]
    proxy = build_selection_vector_view(sources, name, candidates)
    expected_id = selection_vector_id(name, candidates[0])
    if direct.get("stable_id") != candidates[0] or proxy.vector_ids[0] != expected_id:
        raise RuntimeError("direct fixture identity differs from proxy replay identity")
    if not np.allclose(
        tensors["projected_gradient"].numpy().reshape(-1),
        proxy.vectors[0],
        rtol=5e-3,
        atol=5e-3,
    ):
        raise RuntimeError("direct fixture and replay projected gradients differ")
    if proxy.full_gradient_norm is None or not np.allclose(
        float(tensors["full_gradient_norm"].item()),
        float(proxy.full_gradient_norm[0]),
        rtol=5e-3,
        atol=5e-3,
    ):
        raise RuntimeError("direct fixture and replay full gradient norms differ")
    source = next(
        value for value in sources if expected_id in set(value.vector_ids)
    )
    metadata = source.manifest.get("metadata")
    if not isinstance(metadata, Mapping):
        raise RuntimeError("proxy replay manifest lacks metadata")
    parameter_layout = metadata.get("parameter_layout")
    if (
        not isinstance(parameter_layout, Mapping)
        or parameter_layout.get("sha256") != direct.get("parameter_layout_sha256")
        or metadata.get("projection") != direct.get("projection")
    ):
        raise RuntimeError("direct fixture and replay layout/projector provenance differ")
    if not (stage_root / "resume_exercise.json").is_file():
        raise RuntimeError("Stage-0 completion requires replay resume evidence")


def _validate_scientific_outputs(
    stage: int,
    stage_root: Path,
    manifest: Mapping[str, object],
) -> dict[str, object]:
    report = _validate_report_pair(stage, stage_root)
    if report.get("provenance", {}).get("stage_manifest_sha256") != sha256_file(
        stage_root / "manifest.json"
    ):
        raise RuntimeError("analysis report is not bound to the stage manifest")
    if report.get("provenance", {}).get(
        "source_snapshot_sha256"
    ) != _source_snapshot_manifest_sha256(stage_root / "source_snapshot.json"):
        raise RuntimeError("analysis report is not bound to the source snapshot")
    if stage == 0:
        _validate_stage0_direct_fixture(stage_root, manifest)
    return report


def _write_stage_complete(stage: int, stage_root: Path, commands: Sequence[StageCommand]) -> None:
    report_path = stage_root / "report.json"
    marker = {
        "schema_version": 1,
        "artifact_type": "opd_proxy_stage_complete",
        "stage": stage,
        "report_sha256": sha256_file(report_path),
        "source_snapshot_sha256": _source_snapshot_manifest_sha256(
            stage_root / "source_snapshot.json"
        ),
        "source_snapshot_file_sha256": sha256_file(
            stage_root / "source_snapshot.json"
        ),
        "work_units": [command.name for command in commands],
    }
    if stage == 0:
        marker["resume_exercise_sha256"] = sha256_file(
            stage_root / "resume_exercise.json"
        )
        marker["direct_fixture_complete_sha256"] = sha256_file(
            stage_root / "direct_gradient_fixture/COMPLETE.json"
        )
    write_or_validate_json(stage_root / "STAGE_COMPLETE.json", marker)


def _run_stage(args) -> int:
    repository = Path.cwd().resolve()
    output_root = (repository / DEFAULT_OUTPUT_ROOT).resolve()
    stage_root = output_root / f"stage_{args.stage}"
    manifest = _read_stage_manifest(stage_root, args.stage, require_files=True)
    validate_stage_parent(args.stage, manifest, args.parent_report)
    commands = build_stage_commands(
        stage=args.stage,
        manifest=manifest,
        repository_root=repository,
        stage_directory=stage_root,
        reference_repo=args.reference_repo,
    )
    if os.environ.get("OPD_PROXY_VERIFY_DRY_RUN") == "1":
        for command in commands:
            sys.stdout.buffer.write(canonical_json_bytes(command.contract()))
        return 0
    if args.stage == 0 and not args.exercise_resume:
        raise RuntimeError("real Stage 0 requires --exercise-resume")
    if args.stage != 0 and args.exercise_resume:
        raise RuntimeError("--exercise-resume is smoke-only and forbidden after Stage 0")
    validate_stage_prerequisite(args.stage, output_root)
    if args.stage == 2:
        assert args.parent_report is not None
        validate_stage2_trigger(args.parent_report)
    validate_opd_cli_runtime(env=os.environ, expected_gpus=4, require_idle=True)
    validate_main_repository(repository, manifest)
    validate_reference_repository(args.reference_repo)
    source = build_experiment_source_snapshot(repository, args.reference_repo)
    write_or_validate_json(stage_root / "source_snapshot.json", source)
    validate_cross_stage_source(
        args.stage, output_root, str(source["manifest_sha256"])
    )
    # Freeze the source snapshot hash into every real work-unit contract. Dry-run
    # plans use the all-zero sentinel because no stage bytes may be written.
    commands = build_stage_commands(
        stage=args.stage,
        manifest=manifest,
        repository_root=repository,
        stage_directory=stage_root,
        reference_repo=args.reference_repo,
    )
    _validate_model_contract(repository, manifest)
    execute_stage_commands(
        commands,
        stage_directory=stage_root,
        env=os.environ,
        exercise_resume=args.exercise_resume,
    )
    _validate_scientific_outputs(args.stage, stage_root, manifest)
    _write_stage_complete(args.stage, stage_root, commands)
    return 0


def _validate_stage_command(args) -> int:
    repository = Path.cwd().resolve()
    output_root = (repository / DEFAULT_OUTPUT_ROOT).resolve()
    stage_root = (output_root / f"stage_{args.stage}").resolve()
    validate_stage_prerequisite(args.stage, output_root)
    manifest = _read_stage_manifest(stage_root, args.stage, require_files=True)
    parent = None if args.stage < 2 else Path(manifest["parent_report"]["path"])  # type: ignore[index]
    validate_stage_parent(args.stage, manifest, parent)
    if parent is not None:
        validate_stage2_trigger(parent)
    validate_main_repository(repository, manifest)
    validate_reference_repository(args.reference_repo)
    current_source = build_experiment_source_snapshot(repository, args.reference_repo)
    source_path = stage_root / "source_snapshot.json"
    if source_path.is_file() and load_canonical_json(
        source_path, "stage source snapshot"
    ) != current_source:
        raise RuntimeError("stage source snapshot differs from current source bytes")
    validate_cross_stage_source(
        args.stage, output_root, str(current_source["manifest_sha256"])
    )
    if args.require_complete and not source_path.is_file():
        raise RuntimeError("complete-stage validation requires the source snapshot")
    _validate_model_contract(repository, manifest)
    if args.preflight_only:
        return 0
    commands = build_stage_commands(
        stage=args.stage,
        manifest=manifest,
        repository_root=repository,
        stage_directory=stage_root,
        reference_repo=args.reference_repo,
    )
    for command in commands:
        if args.regenerate_report and command.name == "analyze":
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = ""
            subprocess.run(command.argv, check=True, env=env)
        complete = _prepare_unit(stage_root, command)
        if args.require_complete and not complete:
            raise RuntimeError(f"stage work unit is incomplete: {command.name}")
    if args.require_complete:
        marker = load_canonical_json(stage_root / "STAGE_COMPLETE.json", "stage completion")
        if marker.get("stage") != args.stage:
            raise RuntimeError("stage completion marker mismatch")
        _validate_scientific_outputs(args.stage, stage_root, manifest)
        if marker.get("report_sha256") != sha256_file(stage_root / "report.json"):
            raise RuntimeError("stage completion report hash mismatch")
        if marker.get(
            "source_snapshot_sha256"
        ) != _source_snapshot_manifest_sha256(source_path) or marker.get(
            "source_snapshot_file_sha256"
        ) != sha256_file(source_path):
            raise RuntimeError("stage completion source snapshot hash mismatch")
    return 0


def _stage2_decision(args) -> int:
    report = load_canonical_json(args.stage1_report, "Stage-1 report")
    classification = report.get("classification")
    if not isinstance(classification, Mapping):
        raise ValueError("Stage-1 report lacks classification")
    oracle = classification.get("oracle")
    run = False
    reason = "oracle_not_pass"
    if isinstance(oracle, Mapping) and oracle.get("classification") == "pass":
        pn1 = classification.get("P_n1")
        median = pn1["components"]["g_vendi"]["median"]  # type: ignore[index]
        if not isinstance(median, (int, float)) or not math.isfinite(float(median)):
            raise ValueError("Stage-1 P_n1 median is invalid")
        run = 0.90 <= float(median) < 0.95
        reason = "borderline_primary_pn1_g_vendi" if run else "outside_trigger"
    sys.stdout.buffer.write(canonical_json_bytes({"run_stage2": run, "reason": reason}))
    return 0


def parse_slurm_end_time(output: str) -> dt.datetime:
    match = re.search(r"(?:^|\s)EndTime=(\S+)", output)
    if match is None or match.group(1) in {"Unknown", "N/A"}:
        raise RuntimeError("cannot parse active Slurm job EndTime")
    try:
        parsed = dt.datetime.fromisoformat(match.group(1))
    except ValueError as error:
        raise RuntimeError("cannot parse active Slurm job EndTime") from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.datetime.now().astimezone().tzinfo)
    return parsed.astimezone(dt.timezone.utc)


def _estimate(args) -> int:
    repository = Path.cwd().resolve()
    output_root = repository / DEFAULT_OUTPUT_ROOT
    stage0_root = output_root / "stage_0"
    stage0_manifest = _read_stage_manifest(stage0_root, 0, require_files=True)
    stage0_commands = build_stage_commands(
        stage=0,
        manifest=stage0_manifest,
        repository_root=repository,
        stage_directory=stage0_root,
        reference_repo=args.reference_repo,
    )
    timing_paths = sorted((stage0_root / "work_units").glob("*.complete.json"))
    timings = [load_canonical_json(path, "Stage-0 timing") for path in timing_paths]
    if {record.get("name") for record in timings} != {
        command.name for command in stage0_commands
    }:
        raise RuntimeError("estimate requires timing records for every Stage-0 work unit")
    target_root = output_root / f"stage_{args.to_stage}"
    manifest = _read_stage_manifest(target_root, args.to_stage, require_files=True)
    commands = build_stage_commands(
        stage=args.to_stage,
        manifest=manifest,
        repository_root=repository,
        stage_directory=target_root,
        reference_repo=args.reference_repo,
    )
    estimate = estimate_remaining_seconds(
        timing_records=timings,
        commands=commands,
        safety_factor=args.safety_factor,
        reserve_seconds=args.reserve_seconds,
    )
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("estimate requires active SLURM_JOB_ID")
    result = subprocess.run(
        ["scontrol", "show", "job", "-o", job_id],
        check=True,
        capture_output=True,
        text=True,
    )
    end = parse_slurm_end_time(result.stdout)
    completion = dt.datetime.now(dt.timezone.utc) + dt.timedelta(
        seconds=estimate["total_seconds"]
    )
    estimate["estimated_completion_epoch"] = completion.timestamp()
    estimate["job_end_epoch"] = end.timestamp()
    if completion >= end:
        raise RuntimeError("estimated stage completion crosses active job EndTime")
    sys.stdout.buffer.write(canonical_json_bytes(estimate))
    return 0


def _add_reference(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--reference-repo", type=Path, default=DEFAULT_REFERENCE_REPO
    )


def parse_cli_args(argv: Sequence[str] | None = None):
    values = list(sys.argv[1:] if argv is None else argv)
    subcommands = {
        "check-runtime",
        "validate-stage",
        "estimate",
        "stage2-decision",
        "capture-work-unit",
        "internal-validate-vectors",
        "internal-prepare-analysis-inputs",
    }
    if not values or values[0] not in subcommands:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.set_defaults(command="run-stage")
        parser.add_argument("--stage", type=int, choices=(0, 1, 2), required=True)
        parser.add_argument("--parent-report", type=Path)
        parser.add_argument("--exercise-resume", action="store_true")
        _add_reference(parser)
        return parser.parse_args(values)

    parser = argparse.ArgumentParser(description=__doc__)
    children = parser.add_subparsers(dest="command", required=True)
    runtime = children.add_parser("check-runtime")
    runtime.add_argument("--expected-gpus", type=int, default=4)
    runtime.add_argument("--require-idle", action="store_true")
    validate = children.add_parser("validate-stage")
    validate.add_argument("--stage", type=int, choices=(0, 1, 2), required=True)
    validate.add_argument("--preflight-only", action="store_true")
    validate.add_argument("--require-complete", action="store_true")
    validate.add_argument("--regenerate-report", action="store_true")
    _add_reference(validate)
    estimate = children.add_parser("estimate")
    estimate.add_argument("--from-stage", type=int, choices=(0,), required=True)
    estimate.add_argument("--to-stage", type=int, choices=(1, 2), required=True)
    estimate.add_argument("--safety-factor", type=float, default=1.20)
    estimate.add_argument("--reserve-seconds", type=int, default=7200)
    _add_reference(estimate)
    decision = children.add_parser("stage2-decision")
    decision.add_argument("--stage1-report", type=Path, required=True)
    capture = children.add_parser("capture-work-unit")
    capture.add_argument("--stage-directory", type=Path, required=True)
    capture.add_argument("--stage", type=int, choices=(0, 1, 2), required=True)
    capture.add_argument("--pair", choices=("target", "proxy"), required=True)
    capture.add_argument("--engine-seed", type=int, choices=SEEDS, required=True)
    capture.add_argument("--output-root", type=Path, required=True)
    for name in ("internal-validate-vectors", "internal-prepare-analysis-inputs"):
        child = children.add_parser(name)
        child.add_argument("--stage", type=int, choices=(0, 1, 2), required=True)
        child.add_argument("--stage-directory", type=Path, required=True)
    return parser.parse_args(values)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_cli_args(argv)
    if args.command == "run-stage":
        return _run_stage(args)
    if args.command == "check-runtime":
        validate_opd_cli_runtime(
            env=os.environ,
            expected_gpus=args.expected_gpus,
            require_idle=args.require_idle,
        )
        return 0
    if args.command == "validate-stage":
        return _validate_stage_command(args)
    if args.command == "estimate":
        return _estimate(args)
    if args.command == "stage2-decision":
        return _stage2_decision(args)
    if args.command == "capture-work-unit":
        return _capture_work_unit(args)
    if args.command == "internal-validate-vectors":
        _internal_validate_vectors(args.stage, args.stage_directory)
        return 0
    if args.command == "internal-prepare-analysis-inputs":
        _internal_prepare_analysis_inputs(args.stage, args.stage_directory)
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


__all__ = [
    "GVENDI_PYTHON",
    "StageCommand",
    "VERL_PYTHON",
    "build_capture_hydra_overrides",
    "build_experiment_source_snapshot",
    "build_source_snapshot",
    "build_stage_commands",
    "estimate_remaining_seconds",
    "execute_stage_commands",
    "parse_cli_args",
    "parse_slurm_end_time",
    "validate_cross_stage_source",
    "validate_main_repository",
    "validate_opd_cli_runtime",
    "validate_stage_parent",
    "validate_stage_prerequisite",
    "validate_stage2_trigger",
    "validate_reference_repository",
]


if __name__ == "__main__":
    raise SystemExit(main())
