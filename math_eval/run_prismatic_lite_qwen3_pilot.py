"""Phase-based, fail-closed Prismatic-lite Qwen3 generation pilot."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

if __package__ in (None, ""):
    repository_root = str(Path(__file__).resolve().parents[1])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)

from math_eval.prismatic_lite_pilot_analysis import (
    assign_to_centroids,
    calibration_decision,
    candidate_selection,
    final_pilot_decision,
    paired_calibration_metrics,
    question_gradient,
    selection_null_torch,
    smallest_cluster_ids,
    stratified_pairing_null,
)
from math_eval.prismatic_lite_pilot_artifacts import (
    PilotPaths,
    artifact_record,
    atomic_publish_json,
    atomic_publish_jsonl,
    atomic_publish_npz,
    sha256_file,
    strict_json_file,
    strict_jsonl,
    write_or_validate_manifest,
)
from math_eval.prismatic_lite_pilot_generation import (
    NearDuplicateIndex,
    candidate_id,
    difficulty_weighted_fewshots,
    majority_group,
    normalized_tokens,
    parse_generated_problems,
    parse_semantic_judgment,
    problem_prompt,
    question_level_completion_rows,
    solution_messages,
    stratified_calibration_sample,
    ten_grams,
)


CANONICAL_REPOSITORY = Path("/home/mchen/FiRe-OPD")
QWEN_MODEL_PATH = CANONICAL_REPOSITORY / "models/Qwen3-30B-A3B-Instruct-2507"
REFERENCE_REPO = Path("/home/mchen/prismatic-synthesis-reference")
REFERENCE_COMMIT = "d9484cd3b5991030b901ac4a3a9e2472dbfac2ad"
PILOT_DATA_ROOT = CANONICAL_REPOSITORY / "data/prismatic_lite/qwen3_2k_pilot"
PILOT_LOG_ROOT = CANONICAL_REPOSITORY / "logs/prismatic_lite/qwen3_2k_pilot"
SOURCE_JSONL = (
    CANONICAL_REPOSITORY
    / "data/gradient_diversity/deepmath_level6_r1_solution1.jsonl"
)
SOURCE_MANIFEST = SOURCE_JSONL.with_suffix(".manifest.json")
ELIGIBILITY_REPORT = SOURCE_JSONL.with_suffix(".eligibility.json")
ORIGINAL_GRADIENT_DIR = (
    CANONICAL_REPOSITORY
    / "data/gradient_diversity/gradients/qwen2.5-0.5b-instruct"
)
ORIGINAL_GRADIENT_MANIFEST = ORIGINAL_GRADIENT_DIR / "gradient.manifest.json"
BENCHMARK_PATHS = tuple(
    CANONICAL_REPOSITORY / f"data/{name}/test.jsonl"
    for name in (
        "aime24",
        "aime25",
        "amc2023",
        "hmmt25_feb",
        "hmmt25_nov",
        "math500",
        "minervamath",
        "olympiadbench",
    )
)

SOURCE_JSONL_SHA256 = "ee8d55c943577888f14052b3953a0a8aa07d26076e50ddffa41a102b60027344"
SOURCE_MANIFEST_SHA256 = "5add2e965473d3647f2318c73f957434fbe23344516089be9adfe89db9c6a70e"
ELIGIBILITY_SHA256 = "cc8d5b888def37761519e15c6ef722e5a22d91ae2342c6d669f008bcff3dcbb6"
ORIGINAL_GRADIENT_MANIFEST_SHA256 = "9a534118a08e736a15e806933d5cf90c2a99822fee28bf18644e5ff344ce3ae2"
EXPECTED_SOURCE_ROWS = 57_046
EXPECTED_ELIGIBLE_ROWS = 57_045
EXPECTED_EXCLUDED_ID = "deepmath-level6-038794"

TENSOR_PARALLEL_SIZE = 4
MAX_MODEL_LEN = 32_768
CALIBRATION_COUNT = 256
CANDIDATE_COUNT = 2_000
PROBLEM_REQUEST_CAP = 3_000
SOLUTION_COUNT = 3
PROBLEM_MAX_TOKENS = 8_192
SOLUTION_MAX_TOKENS = 16_384
CALIBRATION_NULL_DRAWS = 10_000
SELECTION_NULL_DRAWS = 1_000
PRIMARY_K = 570
SPARSE_CLUSTER_COUNT = 285
K_SENSITIVITY = (285, 1_140)
CLUSTER_ITERATIONS = 20
NO_GO_EXIT_CODE = 20


class PilotStopped(RuntimeError):
    """A frozen, predeclared scientific or operational stop condition."""

    def __init__(self, message: str, *, details: object | None = None) -> None:
        super().__init__(message)
        self.details = details


@dataclass(frozen=True)
class PilotConfig:
    data_root: Path = PILOT_DATA_ROOT
    log_root: Path = PILOT_LOG_ROOT
    source_jsonl: Path = SOURCE_JSONL
    source_manifest: Path = SOURCE_MANIFEST
    eligibility_report: Path = ELIGIBILITY_REPORT
    original_gradient_dir: Path = ORIGINAL_GRADIENT_DIR
    original_gradient_manifest: Path = ORIGINAL_GRADIENT_MANIFEST
    model_path: Path = QWEN_MODEL_PATH
    reference_repo: Path = REFERENCE_REPO
    benchmark_paths: tuple[Path, ...] = BENCHMARK_PATHS


class GenerationBackend(Protocol):
    def generate(
        self,
        messages: Sequence[Sequence[Mapping[str, str]]],
        *,
        seeds: Sequence[int],
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> list[dict[str, object]]: ...

    def close(self) -> None: ...


def generation_seed(phase: str, request_index: int, sample_index: int = 0) -> int:
    if not isinstance(phase, str) or not phase:
        raise ValueError("generation phase must be nonempty")
    if request_index < 0 or sample_index < 0:
        raise ValueError("generation indices must be nonnegative")
    payload = f"prismatic-lite-v1\0{phase}\0{request_index}\0{sample_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def _phase_marker(root: Path, phase: str) -> Path:
    if not phase or any(character not in "abcdefghijklmnopqrstuvwxyz-" for character in phase):
        raise ValueError(f"invalid phase name {phase!r}")
    return Path(root) / "phases" / f"{phase}.json"


def publish_phase(root: Path, phase: str, value: Mapping[str, object]) -> Path:
    marker = _phase_marker(root, phase)
    payload = {
        "phase": phase,
        "status": "complete",
        **dict(value),
    }
    atomic_publish_json(marker, payload)
    return marker


def _validate_bound_artifacts(value: object) -> None:
    if isinstance(value, Mapping):
        if set(value) == {"path", "size", "sha256"}:
            path_value = value.get("path")
            if not isinstance(path_value, str) or not Path(path_value).is_absolute():
                raise ValueError("phase artifact has an invalid absolute path")
            try:
                actual = artifact_record(Path(path_value))
            except ValueError as error:
                raise ValueError(f"phase artifact is missing or invalid: {path_value}") from error
            if actual != dict(value):
                raise ValueError(f"phase artifact has changed: {path_value}")
            return
        for nested in value.values():
            _validate_bound_artifacts(nested)
    elif isinstance(value, list):
        for nested in value:
            _validate_bound_artifacts(nested)


def require_phase(root: Path, phase: str) -> dict[str, object]:
    marker = _phase_marker(root, phase)
    if not marker.is_file():
        raise ValueError(f"required phase {phase} is incomplete: {marker}")
    value = strict_json_file(marker, f"phase {phase}")
    if not isinstance(value, dict) or value.get("phase") != phase or value.get("status") != "complete":
        raise ValueError(f"invalid required phase {phase}: {marker}")
    _validate_bound_artifacts(value)
    return value


def require_promising_calibration(paths: PilotPaths) -> dict[str, object]:
    if not paths.calibration_report.is_file():
        raise ValueError(f"calibration report is missing: {paths.calibration_report}")
    report = strict_json_file(paths.calibration_report, "calibration report")
    if not isinstance(report, dict):
        raise ValueError("calibration report must be an object")
    if report.get("decision") != "promising":
        raise PilotStopped(f"calibration decision is {report.get('decision', 'invalid')}")
    return report


def hash_directory(path: Path) -> list[dict[str, object]]:
    root = Path(path).resolve()
    if not root.is_dir():
        raise ValueError(f"model directory does not exist: {root}")
    records: list[dict[str, object]] = []
    for candidate in sorted(root.rglob("*")):
        if candidate.is_symlink():
            raise ValueError(f"model directory contains a symlink: {candidate}")
        if candidate.is_file():
            records.append(
                {
                    "path": candidate.relative_to(root).as_posix(),
                    "size": candidate.stat().st_size,
                    "sha256": sha256_file(candidate),
                }
            )
    if not records:
        raise ValueError(f"model directory is empty: {root}")
    return records


def build_pilot_manifest(
    *,
    source_jsonl: Path,
    eligibility_report: Path,
    original_gradient_manifest: Path,
    model_path: Path,
    model_files: Sequence[Mapping[str, object]],
    code_commit: str,
    code_tree: str,
    reference_commit: str,
    reference_tree: str,
    benchmark_paths: Sequence[Path] = (),
) -> dict[str, object]:
    return {
        "manifest_version": 1,
        "pilot": "prismatic_lite_qwen3_2k",
        "resume_allowed": False,
        "training_allowed": False,
        "source": artifact_record(source_jsonl),
        "eligibility": artifact_record(eligibility_report),
        "original_gradients": {
            "manifest": str(Path(original_gradient_manifest).resolve()),
            "manifest_sha256": sha256_file(original_gradient_manifest),
            "projection_dimension": 1_024,
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "model_revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        },
        "qwen3": {
            "path": str(Path(model_path).resolve()),
            "files": [dict(record) for record in model_files],
            "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
            "dtype": "bfloat16",
            "max_model_len": MAX_MODEL_LEN,
            "enable_prefix_caching": True,
            "enable_thinking": False,
        },
        "benchmarks": [artifact_record(path) for path in benchmark_paths],
        "code": {"commit": code_commit, "tree": code_tree},
        "reference": {
            "path": str(REFERENCE_REPO.resolve()),
            "commit": reference_commit,
            "tree": reference_tree,
        },
        "sampling": {
            "calibration_solutions": {
                "n": 3,
                "temperature": 0.75,
                "top_p": 0.95,
                "max_tokens": SOLUTION_MAX_TOKENS,
            },
            "candidate_problems": {
                "n": 1,
                "temperature": 1.0,
                "top_p": 0.95,
                "max_tokens": PROBLEM_MAX_TOKENS,
                "request_cap": PROBLEM_REQUEST_CAP,
            },
            "candidate_solutions": {
                "n": 3,
                "temperature": 0.75,
                "top_p": 0.95,
                "max_tokens": SOLUTION_MAX_TOKENS,
            },
            "semantic_review": {
                "n": 1,
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 64,
            },
        },
        "clustering": {
            "metric": "cosine",
            "primary_k": PRIMARY_K,
            "iterations": CLUSTER_ITERATIONS,
            "seeds": [42, 43],
            "sparse_cluster_count": SPARSE_CLUSTER_COUNT,
            "sensitivity_k": list(K_SENSITIVITY),
        },
        "decisions": {
            "calibration": {
                "minimum_qualified": 192,
                "minimum_sparse_dense_agreement": 0.65,
                "minimum_null_margin": 0.15,
                "paired_cosine_null_percentile": 0.90,
                "null_draws": CALIBRATION_NULL_DRAWS,
            },
            "final": {
                "minimum_quality_passed": 1_000,
                "minimum_accepted": 400,
                "minimum_seed_agreement": 0.65,
                "minimum_g_vendi_percentile": 0.90,
                "null_draws": SELECTION_NULL_DRAWS,
            },
        },
    }


def _calibration_gradient_rows(record: Mapping[str, object]) -> list[dict[str, object]]:
    return question_level_completion_rows(
        {
            "id": record["id"],
            "prompt": record["prompt"],
            "responses": record["responses"],
            "majority_indices": record["majority_indices"],
        }
    )


def generate_calibration_records(
    sample: Sequence[Mapping[str, object]], backend: GenerationBackend
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    messages: list[list[dict[str, str]]] = []
    seeds: list[int] = []
    for question_index, row in enumerate(sample):
        for solution_index in range(SOLUTION_COUNT):
            messages.append(solution_messages(str(row["prompt"])))
            seeds.append(
                generation_seed("calibration_solution", question_index, solution_index)
            )
    results = backend.generate(
        messages,
        seeds=seeds,
        temperature=0.75,
        top_p=0.95,
        max_tokens=SOLUTION_MAX_TOKENS,
    )
    if len(results) != len(messages):
        raise RuntimeError("calibration backend returned the wrong result count")

    records: list[dict[str, object]] = []
    gradient_rows: list[dict[str, object]] = []
    for question_index, row in enumerate(sample):
        offset = question_index * SOLUTION_COUNT
        responses = [str(results[offset + index]["text"]) for index in range(3)]
        majority = list(majority_group(responses))
        record = {
            **dict(row),
            "responses": responses,
            "generation_seeds": seeds[offset : offset + 3],
            "output_tokens": [
                int(results[offset + index]["output_tokens"]) for index in range(3)
            ],
            "prompt_tokens": int(results[offset]["prompt_tokens"]),
            "majority_indices": majority,
            "qualified": len(majority) >= 2,
            "sampling": {
                "temperature": 0.75,
                "top_p": 0.95,
                "max_tokens": SOLUTION_MAX_TOKENS,
            },
        }
        records.append(record)
        if record["qualified"]:
            gradient_rows.extend(_calibration_gradient_rows(record))
    return records, gradient_rows


def generate_problem_records(
    eligible_rows: Sequence[Mapping[str, object]],
    backend: GenerationBackend,
    *,
    target_count: int = CANDIDATE_COUNT,
    request_cap: int = PROBLEM_REQUEST_CAP,
    batch_size: int = 64,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not 0 < target_count <= request_cap:
        raise ValueError("target_count must be in [1, request_cap]")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    row_by_id = {str(row["id"]): row for row in eligible_rows}
    fewshots = difficulty_weighted_fewshots(
        eligible_rows, requests=request_cap, width=5, seed=42
    )
    problems: list[dict[str, object]] = []
    attempts: list[dict[str, object]] = []
    seen: set[tuple[str, ...]] = set()
    request_index = 0
    while request_index < request_cap and len(problems) < target_count:
        end = min(request_index + batch_size, request_cap)
        request_indices = list(range(request_index, end))
        messages = [
            [
                {
                    "role": "user",
                    "content": problem_prompt(
                        [row_by_id[parent_id] for parent_id in fewshots[index]]
                    ),
                }
            ]
            for index in request_indices
        ]
        seeds = [generation_seed("candidate_problem", index) for index in request_indices]
        results = backend.generate(
            messages,
            seeds=seeds,
            temperature=1.0,
            top_p=0.95,
            max_tokens=PROBLEM_MAX_TOKENS,
        )
        if len(results) != len(request_indices):
            raise RuntimeError("problem backend returned the wrong result count")
        for index, seed, result in zip(request_indices, seeds, results, strict=True):
            raw = str(result["text"])
            parsed = parse_generated_problems(raw)
            accepted_id: str | None = None
            reason: str | None = None
            if len(problems) >= target_count:
                reason = "target_already_reached"
            elif len(parsed) != 1 or not normalized_tokens(parsed[0]):
                reason = "invalid_problem_boundary"
            elif normalized_tokens(parsed[0]) in seen:
                reason = "normalized_exact_duplicate_generation"
            else:
                prompt = parsed[0].strip()
                accepted_id = candidate_id(index, 0, prompt)
                seen.add(normalized_tokens(prompt))
                problems.append(
                    {
                        "id": accepted_id,
                        "prompt": prompt,
                        "request_index": index,
                        "parent_ids": fewshots[index],
                        "generation_seed": seed,
                        "output_tokens": int(result["output_tokens"]),
                        "prompt_tokens": int(result["prompt_tokens"]),
                        "sampling": {
                            "temperature": 1.0,
                            "top_p": 0.95,
                            "max_tokens": PROBLEM_MAX_TOKENS,
                        },
                    }
                )
            attempts.append(
                {
                    "id": f"problem-request-{index:06d}",
                    "request_index": index,
                    "parent_ids": fewshots[index],
                    "generation_seed": seed,
                    "response": raw,
                    "parsed_problem_count": len(parsed),
                    "accepted_candidate_id": accepted_id,
                    "rejection_reason": reason,
                }
            )
        request_index = end
    if len(problems) != target_count:
        raise PilotStopped(
            f"generated {len(problems)} valid problems before {request_cap} requests",
            details={"problems": problems, "attempts": attempts},
        )
    return problems, attempts


def _benchmark_matches(
    text: str, benchmark_grams: Mapping[tuple[str, ...], set[str]]
) -> list[str]:
    matches: set[str] = set()
    for gram in ten_grams(text):
        matches.update(benchmark_grams.get(gram, ()))
    return sorted(matches)


def review_quality_records(
    candidates: Sequence[Mapping[str, object]],
    solutions: Sequence[Mapping[str, object]],
    *,
    original_prompts: Sequence[tuple[str, str]],
    benchmark_prompts: Sequence[tuple[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    if len(candidates) != len(solutions):
        raise ValueError("candidate and solution counts differ")
    solution_by_id = {str(row["id"]): row for row in solutions}
    if len(solution_by_id) != len(solutions):
        raise ValueError("duplicate candidate solution IDs")
    duplicate_index = NearDuplicateIndex(original_prompts)
    benchmark_grams: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for benchmark_id, prompt in benchmark_prompts:
        for gram in ten_grams(prompt):
            benchmark_grams[gram].add(benchmark_id)

    passed: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    gradient_rows: list[dict[str, object]] = []
    for candidate in candidates:
        candidate_id_value = str(candidate["id"])
        if candidate_id_value not in solution_by_id:
            raise ValueError(f"missing solutions for candidate {candidate_id_value}")
        solution = solution_by_id[candidate_id_value]
        prompt = str(candidate.get("prompt", ""))
        reasons: list[str] = []
        if not normalized_tokens(prompt):
            reasons.append("empty_problem")
        responses = solution.get("responses")
        if not isinstance(responses, list) or len(responses) != 3:
            raise ValueError(f"candidate {candidate_id_value} has invalid responses")
        computed_majority = list(majority_group([str(value) for value in responses]))
        if computed_majority != solution.get("majority_indices"):
            raise ValueError(f"candidate {candidate_id_value} majority audit mismatch")
        if len(computed_majority) < 2:
            reasons.append("answer_consistency")
        prompt_tokens = solution.get("prompt_tokens")
        if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int):
            raise ValueError(f"candidate {candidate_id_value} has invalid prompt_tokens")
        if prompt_tokens > 2_048:
            reasons.append("prompt_too_long")

        neighbor_id, score = duplicate_index.query(prompt)
        exact_duplicate = score == 1.0
        if exact_duplicate:
            reasons.append("normalized_exact_duplicate")
        if solution.get("duplicate_neighbor_id") != neighbor_id or not math.isclose(
            float(solution.get("duplicate_jaccard", -1)), score, abs_tol=1e-12
        ):
            raise ValueError(f"candidate {candidate_id_value} duplicate audit mismatch")
        matches = _benchmark_matches(prompt, benchmark_grams)
        if solution.get("benchmark_matches") != matches:
            raise ValueError(f"candidate {candidate_id_value} benchmark audit mismatch")
        if matches:
            reasons.append("benchmark_contamination")
        if score >= 0.30:
            judgment = solution.get("semantic_judgment")
            try:
                equivalent = parse_semantic_judgment(str(judgment))
            except ValueError:
                reasons.append("semantic_review_invalid")
            else:
                if equivalent:
                    reasons.append("semantic_equivalent")
        elif solution.get("semantic_judgment") is not None:
            raise ValueError(f"candidate {candidate_id_value} has spurious semantic review")
        duplicate_index.add(candidate_id_value, prompt)

        audit = {
            **dict(candidate),
            "majority_indices": computed_majority,
            "duplicate_neighbor_id": neighbor_id,
            "duplicate_jaccard": score,
            "benchmark_matches": matches,
            "semantic_judgment": solution.get("semantic_judgment"),
        }
        if reasons:
            rejected.append({**audit, "rejection_reasons": reasons})
        else:
            passed.append(audit)
            gradient_rows.extend(
                question_level_completion_rows(
                    {
                        "id": candidate_id_value,
                        "prompt": prompt,
                        "responses": responses,
                        "majority_indices": computed_majority,
                    }
                )
            )
    return passed, rejected, gradient_rows


def run_estimate(root: Path | None = None) -> dict[str, object]:
    del root
    return {
        "calibration_question_count": CALIBRATION_COUNT,
        "calibration_solution_count": CALIBRATION_COUNT * SOLUTION_COUNT,
        "candidate_problem_count": CANDIDATE_COUNT,
        "candidate_problem_request_cap": PROBLEM_REQUEST_CAP,
        "candidate_solution_count": CANDIDATE_COUNT * SOLUTION_COUNT,
        "proxy_gradient_shards": 4,
        "calibration_null_draws": CALIBRATION_NULL_DRAWS,
        "selection_null_draws": SELECTION_NULL_DRAWS,
        "estimated_wall_hours": 40,
        "required_wall_hours_with_reserve": 48,
    }


class QwenVllmBackend:
    """Lazy vLLM backend; importing this module never imports vLLM."""

    def __init__(self, model_path: Path = QWEN_MODEL_PATH) -> None:
        import torch
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        model_path = Path(model_path).resolve()
        if model_path != QWEN_MODEL_PATH.resolve():
            raise ValueError(f"Qwen3 model path mismatch: {model_path}")
        if torch.cuda.device_count() != TENSOR_PARALLEL_SIZE:
            raise RuntimeError(
                f"Qwen3 generation requires exactly {TENSOR_PARALLEL_SIZE} visible GPUs"
            )
        self._sampling_params_class = SamplingParams
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True
        )
        self._llm = LLM(
            model=str(model_path),
            tokenizer=str(model_path),
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            dtype="bfloat16",
            max_model_len=MAX_MODEL_LEN,
            enable_prefix_caching=True,
            gpu_memory_utilization=0.92,
            max_num_seqs=64,
            enforce_eager=True,
        )
        self._closed = False

    def _tokenize(self, messages: Sequence[Mapping[str, str]]) -> list[int]:
        tokens = self._tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if not isinstance(tokens, list) or not tokens:
            raise RuntimeError("Qwen3 chat template returned invalid token IDs")
        if len(tokens) >= MAX_MODEL_LEN:
            raise ValueError(f"Qwen3 prompt has {len(tokens)} tokens")
        return tokens

    def generate(
        self,
        messages: Sequence[Sequence[Mapping[str, str]]],
        *,
        seeds: Sequence[int],
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> list[dict[str, object]]:
        if self._closed:
            raise RuntimeError("Qwen3 backend is closed")
        if len(messages) != len(seeds):
            raise ValueError("message and seed counts differ")
        prompt_token_ids = [self._tokenize(value) for value in messages]
        parameters = [
            self._sampling_params_class(
                n=1,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                seed=int(seed),
            )
            for seed in seeds
        ]
        outputs = self._llm.generate(
            prompt_token_ids,
            sampling_params=parameters,
            use_tqdm=True,
        )
        if len(outputs) != len(messages):
            raise RuntimeError("vLLM returned the wrong request count")
        results: list[dict[str, object]] = []
        for index, output in enumerate(outputs):
            if len(output.outputs) != 1:
                raise RuntimeError(f"vLLM request {index} did not return exactly one output")
            sample = output.outputs[0]
            results.append(
                {
                    "text": sample.text,
                    "output_tokens": len(sample.token_ids),
                    "prompt_tokens": len(prompt_token_ids[index]),
                    "finish_reason": sample.finish_reason,
                }
            )
        return results

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        llm = self._llm
        self._llm = None
        try:
            executor = getattr(getattr(llm, "llm_engine", None), "model_executor", None)
            shutdown = getattr(executor, "shutdown", None)
            if callable(shutdown):
                shutdown()
        finally:
            del llm
            gc.collect()
            try:
                from vllm.distributed.parallel_state import (
                    destroy_distributed_environment,
                    destroy_model_parallel,
                )

                destroy_model_parallel()
                destroy_distributed_environment()
            except Exception:
                pass
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass


def _git(repository: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"git identity check failed for {repository}: {error}") from error
    return result.stdout.strip()


def _clean_git_identity(repository: Path) -> tuple[str, str]:
    repository = Path(repository).resolve()
    status = _git(repository, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise ValueError(f"executable repository is not clean: {repository}")
    return _git(repository, "rev-parse", "HEAD"), _git(
        repository, "rev-parse", "HEAD^{tree}"
    )


def _require_file_hash(path: Path, expected: str, description: str) -> None:
    if not Path(path).is_file():
        raise ValueError(f"{description} is missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"{description} hash mismatch: expected {expected}, got {actual}"
        )


def _load_eligible_rows(
    config: PilotConfig, *, enforce_canonical_hashes: bool = True
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if enforce_canonical_hashes:
        _require_file_hash(config.source_jsonl, SOURCE_JSONL_SHA256, "prepared source")
        _require_file_hash(
            config.source_manifest, SOURCE_MANIFEST_SHA256, "prepared manifest"
        )
        _require_file_hash(
            config.eligibility_report, ELIGIBILITY_SHA256, "eligibility report"
        )
    source_rows = strict_jsonl(config.source_jsonl)
    eligibility = strict_json_file(config.eligibility_report, "eligibility report")
    if not isinstance(eligibility, dict):
        raise ValueError("eligibility report must be an object")
    excluded = eligibility.get("expected_excluded_ids")
    if not isinstance(excluded, list) or not all(
        isinstance(value, str) for value in excluded
    ):
        raise ValueError("eligibility report has invalid expected_excluded_ids")
    excluded_set = set(excluded)
    eligible = [row for row in source_rows if row.get("id") not in excluded_set]
    if enforce_canonical_hashes:
        if len(source_rows) != EXPECTED_SOURCE_ROWS:
            raise ValueError("prepared source row count drift")
        if excluded != [EXPECTED_EXCLUDED_ID]:
            raise ValueError("eligibility exclusion drift")
        if len(eligible) != EXPECTED_ELIGIBLE_ROWS:
            raise ValueError("eligible row count drift")
        if eligibility.get("eligible_row_count") != EXPECTED_ELIGIBLE_ROWS:
            raise ValueError("eligibility report row count drift")
    seen: set[str] = set()
    for gradient_index, row in enumerate(eligible):
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise ValueError(f"invalid or duplicate eligible ID at {gradient_index}")
        seen.add(sample_id)
        row["original_gradient_index"] = gradient_index
        for field in ("prompt", "completion", "topic"):
            if not isinstance(row.get(field), str) or not str(row[field]).strip():
                raise ValueError(f"eligible row {sample_id} has invalid {field}")
        difficulty = row.get("difficulty")
        if isinstance(difficulty, bool) or not isinstance(difficulty, (int, float)):
            raise ValueError(f"eligible row {sample_id} has invalid difficulty")
    return eligible, eligibility


def _load_benchmark_prompts(config: PilotConfig) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    for path in config.benchmark_paths:
        dataset = Path(path).parent.name
        rows = strict_jsonl(path)
        for index, row in enumerate(rows):
            problem = row.get("problem")
            if not isinstance(problem, str) or not problem.strip():
                raise ValueError(f"benchmark prompt is invalid: {path}:{index + 1}")
            records.append((f"{dataset}-{index:04d}", problem))
    return records


def _load_root_manifest(root: Path) -> dict[str, object]:
    path = PilotPaths.from_root(root).manifest
    if not path.is_file():
        raise ValueError(f"pilot manifest is missing: {path}")
    value = strict_json_file(path, "pilot manifest")
    if (
        not isinstance(value, dict)
        or value.get("manifest_version") != 1
        or value.get("pilot") != "prismatic_lite_qwen3_2k"
        or value.get("resume_allowed") is not False
        or value.get("training_allowed") is not False
    ):
        raise ValueError("pilot manifest contract is invalid")
    return value


def _extra_path(root: Path, relative: str) -> Path:
    return Path(root) / relative


def _publish_no_go(
    paths: PilotPaths,
    *,
    stop_stage: str,
    reason: str,
    report: Mapping[str, object],
    calibration: bool = False,
) -> None:
    target = paths.calibration_report if calibration else paths.report_json
    payload = {
        "pilot": "prismatic_lite_qwen3_2k",
        "decision": "no_go",
        "stop_stage": stop_stage,
        "reason": reason,
        "training_performed": False,
        **dict(report),
    }
    atomic_publish_json(target, payload)
    atomic_publish_json(
        paths.stage_complete,
        {
            "pilot": "prismatic_lite_qwen3_2k",
            "status": "complete",
            "decision": "no_go",
            "stop_stage": stop_stage,
            "report": artifact_record(target),
            "training_performed": False,
        },
    )


def run_prepare(config: PilotConfig) -> dict[str, object]:
    root = Path(config.data_root)
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"pilot root already exists; resume is forbidden: {root}")
    if Path(config.model_path).resolve() != QWEN_MODEL_PATH.resolve():
        raise ValueError("canonical Qwen3 model path drift")
    if Path(config.reference_repo).resolve() != REFERENCE_REPO.resolve():
        raise ValueError("canonical reference repository path drift")
    _require_file_hash(
        config.original_gradient_manifest,
        ORIGINAL_GRADIENT_MANIFEST_SHA256,
        "original gradient manifest",
    )
    eligible, _ = _load_eligible_rows(config)
    gradient_manifest = strict_json_file(
        config.original_gradient_manifest, "original gradient manifest"
    )
    if not isinstance(gradient_manifest, dict):
        raise ValueError("original gradient manifest must be an object")
    projection = gradient_manifest.get("projection")
    if not isinstance(projection, dict) or projection.get("dimension") != 1_024:
        raise ValueError("original gradient projection contract drift")
    from math_eval.collect_prismatic_gradients import validate_global_gradient_coverage

    coverage = validate_global_gradient_coverage(
        [str(row["id"]) for row in eligible],
        config.original_gradient_dir,
        str(gradient_manifest["prefix"]),
        int(gradient_manifest["num_shards"]),
    )
    if coverage.get("row_count") != EXPECTED_ELIGIBLE_ROWS:
        raise ValueError("original gradient coverage drift")

    code_repository = Path(__file__).resolve().parents[1]
    code_commit, code_tree = _clean_git_identity(code_repository)
    reference_commit, reference_tree = _clean_git_identity(config.reference_repo)
    if reference_commit != REFERENCE_COMMIT:
        raise ValueError(
            f"reference commit mismatch: expected {REFERENCE_COMMIT}, got {reference_commit}"
        )
    model_files = hash_directory(config.model_path)
    manifest = build_pilot_manifest(
        source_jsonl=config.source_jsonl,
        eligibility_report=config.eligibility_report,
        original_gradient_manifest=config.original_gradient_manifest,
        model_path=config.model_path,
        model_files=model_files,
        code_commit=code_commit,
        code_tree=code_tree,
        reference_commit=reference_commit,
        reference_tree=reference_tree,
        benchmark_paths=config.benchmark_paths,
    )
    paths = PilotPaths.from_root(root)
    sample = stratified_calibration_sample(eligible, size=CALIBRATION_COUNT, seed=42)
    sample_records = [
        {
            "id": row["id"],
            "prompt": row["prompt"],
            "original_completion": row["completion"],
            "source_row_index": row["source_row_index"],
            "original_dataset_index": row["original_dataset_index"],
            "original_gradient_index": row["original_gradient_index"],
            "topic": row["topic"],
            "difficulty": row["difficulty"],
        }
        for row in sample
    ]
    write_or_validate_manifest(paths.manifest, manifest)
    atomic_publish_jsonl(paths.calibration_sample, sample_records)
    publish_phase(
        root,
        "prepare",
        {
            "manifest": artifact_record(paths.manifest),
            "calibration_sample": artifact_record(paths.calibration_sample),
            "calibration_count": CALIBRATION_COUNT,
            "eligible_count": len(eligible),
            "gradient_coverage": coverage,
        },
    )
    return {
        "status": "prepared",
        "root": str(root),
        "calibration_count": len(sample_records),
        "eligible_count": len(eligible),
    }


def run_generate_calibration_solutions(config: PilotConfig) -> dict[str, object]:
    require_phase(config.data_root, "prepare")
    _load_root_manifest(config.data_root)
    paths = PilotPaths.from_root(config.data_root)
    sample = strict_jsonl(paths.calibration_sample)
    if len(sample) != CALIBRATION_COUNT:
        raise ValueError("calibration sample must contain exactly 256 rows")
    backend = QwenVllmBackend(config.model_path)
    started = time.monotonic()
    try:
        records, gradient_rows = generate_calibration_records(sample, backend)
    finally:
        backend.close()
    atomic_publish_jsonl(paths.calibration_solutions, records)
    atomic_publish_jsonl(paths.calibration_gradient_input, gradient_rows)
    qualified = sum(bool(row["qualified"]) for row in records)
    publish_phase(
        config.data_root,
        "generate-calibration-solutions",
        {
            "solutions": artifact_record(paths.calibration_solutions),
            "gradient_input": artifact_record(paths.calibration_gradient_input),
            "question_count": len(records),
            "trajectory_count": len(records) * 3,
            "qualified_count": qualified,
            "gradient_row_count": len(gradient_rows),
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    if qualified < 192:
        reason = f"only {qualified} of 256 calibration questions had a 2-of-3 majority"
        _publish_no_go(
            paths,
            stop_stage="calibration_qualification",
            reason=reason,
            report={
                "qualified_count": qualified,
                "minimum_qualified": 192,
                "failed_conditions": ["qualified_count_below_192"],
            },
            calibration=True,
        )
        raise PilotStopped(reason)
    return {
        "status": "generated",
        "question_count": len(records),
        "trajectory_count": len(records) * 3,
        "qualified_count": qualified,
        "gradient_row_count": len(gradient_rows),
    }


def _load_original_gradients(
    config: PilotConfig, eligible: Sequence[Mapping[str, object]]
):
    from math_eval.select_gradient_diverse_deepmath import load_projected_gradients

    manifest = strict_json_file(
        config.original_gradient_manifest, "original gradient manifest"
    )
    if not isinstance(manifest, dict):
        raise ValueError("original gradient manifest is invalid")
    ids, gradients = load_projected_gradients(
        config.original_gradient_dir,
        [str(row["id"]) for row in eligible],
        prefix=str(manifest["prefix"]),
    )
    return ids, gradients


def _ids_sha256(ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in ids:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _gradient_matrix_sha256(gradients: object) -> str:
    import torch

    tensor = torch.as_tensor(gradients, dtype=torch.float32, device="cpu").contiguous()
    array = tensor.numpy()
    digest = hashlib.sha256()
    for start in range(0, array.shape[0], 1_024):
        digest.update(array[start : start + 1_024].tobytes(order="C"))
    return digest.hexdigest()


def _cluster_state_path(root: Path, k: int, seed: int) -> Path:
    paths = PilotPaths.from_root(root)
    if k == PRIMARY_K and seed == 42:
        return paths.cluster_seed42
    if k == PRIMARY_K and seed == 43:
        return paths.cluster_seed43
    return _extra_path(root, f"selection/cluster_state_k{k}_seed{seed}.npz")


def _run_cluster_state(
    *,
    root: Path,
    gradients: object,
    ids: Sequence[str],
    k: int,
    seed: int,
    reference_repo: Path,
) -> dict[str, object]:
    import torch
    import torch.nn.functional as functional

    from math_eval.select_gradient_diverse_deepmath import cluster_official

    if k not in {PRIMARY_K, *K_SENSITIVITY}:
        raise ValueError(f"unsupported frozen K={k}")
    ratio = {PRIMARY_K: 0.01, 285: 0.005, 1_140: 0.02}[k]
    tensor = torch.as_tensor(gradients, dtype=torch.float32, device="cpu")
    labels = cluster_official(
        tensor,
        ratio=ratio,
        iterations=CLUSTER_ITERATIONS,
        seed=seed,
        reference_repo=reference_repo,
        device="cuda:0",
    )
    actual_k = max(2, min(tensor.shape[0], math.floor(ratio * tensor.shape[0])))
    if actual_k != k:
        raise RuntimeError(f"cluster ratio produced K={actual_k}, expected {k}")
    normalized = functional.normalize(tensor, dim=1)
    label_tensor = torch.as_tensor(labels, dtype=torch.int64)
    sums = torch.zeros((k, tensor.shape[1]), dtype=torch.float64)
    sums.index_add_(0, label_tensor, normalized.to(torch.float64))
    sizes = torch.bincount(label_tensor, minlength=k)
    if bool((sizes <= 0).any()):
        raise PilotStopped(f"K={k} seed={seed} produced an empty cluster")
    norms = torch.linalg.vector_norm(sums, dim=1)
    if not bool(torch.isfinite(norms).all()) or bool((norms <= 0).any()):
        raise PilotStopped(f"K={k} seed={seed} produced invalid centroids")
    centroids = (sums / norms[:, None]).to(torch.float32).numpy()
    sparse_count = k // 2
    sparse_ids = smallest_cluster_ids(
        labels, count=sparse_count, cluster_count=k
    )
    values = {
        "labels": np.asarray(labels, dtype=np.int64),
        "centroids": centroids,
        "cluster_sizes": sizes.numpy().astype(np.int64),
        "sparse_cluster_ids": np.asarray(sparse_ids, dtype=np.int64),
        "k": np.asarray(k, dtype=np.int64),
        "iterations": np.asarray(CLUSTER_ITERATIONS, dtype=np.int64),
        "seed": np.asarray(seed, dtype=np.int64),
        "input_ids_sha256": np.asarray(_ids_sha256(ids)),
        "gradient_sha256": np.asarray(_gradient_matrix_sha256(tensor)),
    }
    path = _cluster_state_path(root, k, seed)
    atomic_publish_npz(path, values)
    return {key: value for key, value in values.items()}


def _load_cluster_state(path: Path, *, expected_k: int, expected_seed: int) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            values = {name: archive[name].copy() for name in archive.files}
    except Exception as error:
        raise ValueError(f"invalid cluster state {path}: {error}") from error
    required = {
        "labels",
        "centroids",
        "cluster_sizes",
        "sparse_cluster_ids",
        "k",
        "iterations",
        "seed",
        "input_ids_sha256",
        "gradient_sha256",
    }
    if set(values) != required:
        raise ValueError(f"cluster state fields mismatch: {path}")
    if int(values["k"]) != expected_k or int(values["seed"]) != expected_seed:
        raise ValueError(f"cluster state identity mismatch: {path}")
    if int(values["iterations"]) != CLUSTER_ITERATIONS:
        raise ValueError(f"cluster state iteration mismatch: {path}")
    if values["centroids"].shape != (expected_k, 1_024):
        raise ValueError(f"cluster state centroid shape mismatch: {path}")
    if not bool(np.isfinite(values["centroids"]).all()):
        raise ValueError(f"cluster state has nonfinite centroids: {path}")
    return values


def _load_pilot_gradients(directory: Path, rows: Sequence[Mapping[str, object]]):
    from math_eval.select_gradient_diverse_deepmath import load_projected_gradients

    manifest = strict_json_file(directory / "gradient.manifest.json", "pilot gradient manifest")
    if not isinstance(manifest, dict):
        raise ValueError("pilot gradient manifest is invalid")
    return load_projected_gradients(
        directory,
        [str(row["id"]) for row in rows],
        prefix=str(manifest["prefix"]),
    )


def run_analyze_calibration(config: PilotConfig, *, device: str = "cuda:0") -> dict[str, object]:
    if device != "cuda:0":
        raise ValueError("calibration analysis requires process-local cuda:0")
    require_phase(config.data_root, "generate-calibration-solutions")
    paths = PilotPaths.from_root(config.data_root)
    if paths.calibration_report.exists():
        report = strict_json_file(paths.calibration_report, "calibration report")
        if isinstance(report, dict) and report.get("decision") == "no_go":
            raise PilotStopped("calibration is already no_go")
        raise FileExistsError(f"calibration report already exists: {paths.calibration_report}")
    eligible, _ = _load_eligible_rows(config)
    original_ids, original_gradients = _load_original_gradients(config, eligible)
    state42 = _run_cluster_state(
        root=config.data_root,
        gradients=original_gradients,
        ids=original_ids,
        k=PRIMARY_K,
        seed=42,
        reference_repo=config.reference_repo,
    )
    state43 = _run_cluster_state(
        root=config.data_root,
        gradients=original_gradients,
        ids=original_ids,
        k=PRIMARY_K,
        seed=43,
        reference_repo=config.reference_repo,
    )
    sample = strict_jsonl(paths.calibration_sample)
    solutions = strict_jsonl(paths.calibration_solutions)
    solution_by_id = {str(row["id"]): row for row in solutions}
    gradient_input = strict_jsonl(paths.calibration_gradient_input)
    new_ids, new_gradients = _load_pilot_gradients(
        paths.calibration_gradients, gradient_input
    )
    vector_by_id = {
        sample_id: new_gradients[index].numpy()
        for index, sample_id in enumerate(new_ids)
    }
    question_vectors: list[np.ndarray] = []
    original_vectors: list[np.ndarray] = []
    qualified_rows: list[Mapping[str, object]] = []
    for row in sample:
        record = solution_by_id.get(str(row["id"]))
        if record is None or record.get("qualified") is not True:
            continue
        majority = record.get("majority_indices")
        if not isinstance(majority, list) or len(majority) < 2:
            raise ValueError(f"qualified row {row['id']} has invalid majority")
        vectors = [
            vector_by_id[f"{row['id']}.solution-{index}"] for index in majority
        ]
        question_vectors.append(question_gradient(np.stack(vectors)))
        original_index = int(row["original_gradient_index"])
        original_vectors.append(original_gradients[original_index].numpy())
        qualified_rows.append(row)
    regenerated = np.stack(question_vectors)
    paired_original = np.stack(original_vectors)
    metrics_by_seed: dict[str, object] = {}
    for seed, state in ((42, state42), (43, state43)):
        original_labels = np.asarray(
            [state["labels"][int(row["original_gradient_index"])] for row in qualified_rows]
        )
        regenerated_labels = assign_to_centroids(regenerated, state["centroids"])
        sparse = tuple(int(value) for value in state["sparse_cluster_ids"])
        paired = paired_calibration_metrics(
            paired_original,
            regenerated,
            original_labels,
            regenerated_labels,
            sparse_cluster_ids=sparse,
        )
        old_sparse = np.isin(original_labels, sparse)
        new_sparse = np.isin(regenerated_labels, sparse)
        null = stratified_pairing_null(
            paired_original,
            regenerated,
            qualified_rows,
            old_sparse,
            new_sparse,
            draws=CALIBRATION_NULL_DRAWS,
            seed=generation_seed("calibration_null", seed),
        )
        metrics_by_seed[str(seed)] = {
            **paired,
            "null_agreement_median": null["sparse_dense_agreement_median"],
            "null_paired_cosine_median_p90": null["paired_cosine_median_p90"],
            "null": null,
        }
    decision_input = {
        "qualified_count": len(qualified_rows),
        "finite": bool(
            np.isfinite(regenerated).all()
            and np.isfinite(paired_original).all()
            and np.isfinite(state42["centroids"]).all()
            and np.isfinite(state43["centroids"]).all()
        ),
        "seeds": metrics_by_seed,
    }
    decision = calibration_decision(decision_input)
    report = {
        "pilot": "prismatic_lite_qwen3_2k",
        **decision_input,
        **decision,
        "cluster_states": {
            "42": artifact_record(paths.cluster_seed42),
            "43": artifact_record(paths.cluster_seed43),
        },
        "training_performed": False,
    }
    atomic_publish_json(paths.calibration_report, report)
    publish_phase(
        config.data_root,
        "analyze-calibration",
        {
            "decision": decision["decision"],
            "qualified_count": len(qualified_rows),
            "report": artifact_record(paths.calibration_report),
        },
    )
    if decision["decision"] == "no_go":
        atomic_publish_json(
            paths.stage_complete,
            {
                "pilot": "prismatic_lite_qwen3_2k",
                "status": "complete",
                "decision": "no_go",
                "stop_stage": "calibration_analysis",
                "report": artifact_record(paths.calibration_report),
                "training_performed": False,
            },
        )
        raise PilotStopped("calibration analysis failed frozen thresholds")
    return {
        "status": "analyzed",
        "decision": "promising",
        "qualified_count": len(qualified_rows),
    }


def run_generate_problems(config: PilotConfig) -> dict[str, object]:
    require_phase(config.data_root, "analyze-calibration")
    paths = PilotPaths.from_root(config.data_root)
    require_promising_calibration(paths)
    eligible, _ = _load_eligible_rows(config)
    backend = QwenVllmBackend(config.model_path)
    started = time.monotonic()
    try:
        try:
            problems, attempts = generate_problem_records(eligible, backend)
        except PilotStopped as error:
            details = error.details if isinstance(error.details, dict) else {}
            attempts = details.get("attempts", [])
            atomic_publish_jsonl(
                _extra_path(config.data_root, "candidates/problem_attempts.jsonl"),
                attempts,
            )
            _publish_no_go(
                paths,
                stop_stage="candidate_problem_generation",
                reason=str(error),
                report={
                    "accepted_problem_count": len(details.get("problems", [])),
                    "request_count": len(attempts),
                    "request_cap": PROBLEM_REQUEST_CAP,
                },
            )
            raise
    finally:
        backend.close()
    attempts_path = _extra_path(config.data_root, "candidates/problem_attempts.jsonl")
    atomic_publish_jsonl(attempts_path, attempts)
    atomic_publish_jsonl(paths.candidate_problems, problems)
    publish_phase(
        config.data_root,
        "generate-problems",
        {
            "problems": artifact_record(paths.candidate_problems),
            "attempts": artifact_record(attempts_path),
            "problem_count": len(problems),
            "request_count": len(attempts),
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    return {
        "status": "generated",
        "problem_count": len(problems),
        "request_count": len(attempts),
    }


def _semantic_review_messages(left: str, right: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Judge whether two math problems are semantically equivalent or "
                "mere paraphrases/numeric substitutions. Output exactly equivalent "
                "or not_equivalent and nothing else."
            ),
        },
        {"role": "user", "content": f"[[Problem A]]\n{left}\n[[Problem B]]\n{right}"},
    ]


def generate_candidate_solution_records(
    candidates: Sequence[Mapping[str, object]],
    eligible_rows: Sequence[Mapping[str, object]],
    benchmark_prompts: Sequence[tuple[str, str]],
    backend: GenerationBackend,
    *,
    batch_size: int = 128,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if len(candidates) != CANDIDATE_COUNT:
        raise ValueError("candidate solution generation requires exactly 2,000 problems")
    flat_messages: list[list[dict[str, str]]] = []
    flat_seeds: list[int] = []
    for candidate_index, candidate in enumerate(candidates):
        for solution_index in range(3):
            flat_messages.append(solution_messages(str(candidate["prompt"])))
            flat_seeds.append(
                generation_seed("candidate_solution", candidate_index, solution_index)
            )
    flat_results: list[dict[str, object]] = []
    for start in range(0, len(flat_messages), batch_size):
        end = min(start + batch_size, len(flat_messages))
        flat_results.extend(
            backend.generate(
                flat_messages[start:end],
                seeds=flat_seeds[start:end],
                temperature=0.75,
                top_p=0.95,
                max_tokens=SOLUTION_MAX_TOKENS,
            )
        )
    if len(flat_results) != CANDIDATE_COUNT * 3:
        raise RuntimeError("candidate solution backend returned the wrong count")

    original_prompts = [(str(row["id"]), str(row["prompt"])) for row in eligible_rows]
    text_by_id = {sample_id: text for sample_id, text in original_prompts}
    duplicate_index = NearDuplicateIndex(original_prompts)
    benchmark_grams: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for benchmark_id, prompt in benchmark_prompts:
        for gram in ten_grams(prompt):
            benchmark_grams[gram].add(benchmark_id)
    records: list[dict[str, object]] = []
    review_queue: list[tuple[int, str, str, str]] = []
    for candidate_index, candidate in enumerate(candidates):
        offset = candidate_index * 3
        prompt = str(candidate["prompt"])
        responses = [str(flat_results[offset + index]["text"]) for index in range(3)]
        majority = list(majority_group(responses))
        prompt_counts = [
            int(flat_results[offset + index]["prompt_tokens"]) for index in range(3)
        ]
        if len(set(prompt_counts)) != 1:
            raise RuntimeError(f"candidate {candidate['id']} had inconsistent prompt tokens")
        neighbor_id, score = duplicate_index.query(prompt)
        matches = _benchmark_matches(prompt, benchmark_grams)
        record = {
            "id": candidate["id"],
            "prompt": prompt,
            "responses": responses,
            "generation_seeds": flat_seeds[offset : offset + 3],
            "output_tokens": [
                int(flat_results[offset + index]["output_tokens"]) for index in range(3)
            ],
            "prompt_tokens": prompt_counts[0],
            "majority_indices": majority,
            "duplicate_neighbor_id": neighbor_id,
            "duplicate_jaccard": score,
            "benchmark_matches": matches,
            "semantic_judgment": None,
            "sampling": {
                "temperature": 0.75,
                "top_p": 0.95,
                "max_tokens": SOLUTION_MAX_TOKENS,
            },
        }
        records.append(record)
        if score >= 0.30 and neighbor_id is not None:
            review_queue.append((candidate_index, str(candidate["id"]), neighbor_id, text_by_id[neighbor_id]))
        duplicate_index.add(str(candidate["id"]), prompt)
        text_by_id[str(candidate["id"])] = prompt

    reviews: list[dict[str, object]] = []
    for start in range(0, len(review_queue), batch_size):
        batch = review_queue[start : start + batch_size]
        messages = [
            _semantic_review_messages(str(candidates[index]["prompt"]), neighbor_text)
            for index, _, _, neighbor_text in batch
        ]
        seeds = [generation_seed("semantic_review", index) for index, _, _, _ in batch]
        results = backend.generate(
            messages,
            seeds=seeds,
            temperature=0.0,
            top_p=1.0,
            max_tokens=64,
        )
        if len(results) != len(batch):
            raise RuntimeError("semantic review backend returned the wrong count")
        for (candidate_index, sample_id, neighbor_id, _), seed, result in zip(
            batch, seeds, results, strict=True
        ):
            response = str(result["text"]).strip()
            records[candidate_index]["semantic_judgment"] = response
            reviews.append(
                {
                    "id": f"semantic-review-{candidate_index:06d}",
                    "candidate_id": sample_id,
                    "neighbor_id": neighbor_id,
                    "generation_seed": seed,
                    "response": response,
                    "output_tokens": int(result["output_tokens"]),
                    "prompt_tokens": int(result["prompt_tokens"]),
                }
            )
    return records, reviews


def run_generate_candidate_solutions(config: PilotConfig) -> dict[str, object]:
    require_phase(config.data_root, "generate-problems")
    paths = PilotPaths.from_root(config.data_root)
    require_promising_calibration(paths)
    candidates = strict_jsonl(paths.candidate_problems)
    if len(candidates) != CANDIDATE_COUNT:
        raise ValueError("candidate problem artifact must have exactly 2,000 rows")
    eligible, _ = _load_eligible_rows(config)
    benchmarks = _load_benchmark_prompts(config)
    backend = QwenVllmBackend(config.model_path)
    started = time.monotonic()
    try:
        records, reviews = generate_candidate_solution_records(
            candidates, eligible, benchmarks, backend
        )
    finally:
        backend.close()
    reviews_path = _extra_path(config.data_root, "candidates/semantic_reviews.jsonl")
    atomic_publish_jsonl(paths.candidate_solutions, records)
    atomic_publish_jsonl(reviews_path, reviews)
    publish_phase(
        config.data_root,
        "generate-candidate-solutions",
        {
            "solutions": artifact_record(paths.candidate_solutions),
            "semantic_reviews": artifact_record(reviews_path),
            "question_count": len(records),
            "trajectory_count": len(records) * 3,
            "semantic_review_count": len(reviews),
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    return {
        "status": "generated",
        "question_count": len(records),
        "trajectory_count": len(records) * 3,
        "semantic_review_count": len(reviews),
    }


def run_quality_review(config: PilotConfig) -> dict[str, object]:
    require_phase(config.data_root, "generate-candidate-solutions")
    paths = PilotPaths.from_root(config.data_root)
    candidates = strict_jsonl(paths.candidate_problems)
    solutions = strict_jsonl(paths.candidate_solutions)
    eligible, _ = _load_eligible_rows(config)
    benchmarks = _load_benchmark_prompts(config)
    passed, rejected, gradient_rows = review_quality_records(
        candidates,
        solutions,
        original_prompts=[(str(row["id"]), str(row["prompt"])) for row in eligible],
        benchmark_prompts=benchmarks,
    )
    rejected_path = _extra_path(config.data_root, "candidates/quality_rejected.jsonl")
    atomic_publish_jsonl(paths.quality_passed, passed)
    atomic_publish_jsonl(rejected_path, rejected)
    atomic_publish_jsonl(paths.candidate_gradient_input, gradient_rows)
    publish_phase(
        config.data_root,
        "quality-review",
        {
            "quality_passed": artifact_record(paths.quality_passed),
            "quality_rejected": artifact_record(rejected_path),
            "gradient_input": artifact_record(paths.candidate_gradient_input),
            "quality_count": len(passed),
            "rejected_count": len(rejected),
            "gradient_row_count": len(gradient_rows),
        },
    )
    if len(passed) < 1_000:
        reason = f"only {len(passed)} of 2,000 candidates passed frozen quality gates"
        _publish_no_go(
            paths,
            stop_stage="candidate_quality",
            reason=reason,
            report={
                "quality_count": len(passed),
                "minimum_quality_count": 1_000,
                "rejected_count": len(rejected),
            },
        )
        raise PilotStopped(reason)
    return {
        "status": "reviewed",
        "quality_count": len(passed),
        "rejected_count": len(rejected),
        "gradient_row_count": len(gradient_rows),
    }


def _aggregate_candidate_gradients(
    quality_rows: Sequence[Mapping[str, object]],
    gradient_rows: Sequence[Mapping[str, object]],
    gradient_ids: Sequence[str],
    gradients: object,
) -> np.ndarray:
    vectors = {
        sample_id: gradients[index].numpy()
        for index, sample_id in enumerate(gradient_ids)
    }
    rows_by_question: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in gradient_rows:
        rows_by_question[str(row["question_id"])].append(row)
    output: list[np.ndarray] = []
    for quality in quality_rows:
        question_id = str(quality["id"])
        rows = sorted(
            rows_by_question.get(question_id, []),
            key=lambda row: int(row["solution_index"]),
        )
        expected_indices = quality.get("majority_indices")
        if [row["solution_index"] for row in rows] != expected_indices:
            raise ValueError(f"gradient solution coverage mismatch for {question_id}")
        output.append(
            question_gradient(np.stack([vectors[str(row["id"])] for row in rows]))
        )
    return np.stack(output)


def _write_report_markdown(path: Path, report: Mapping[str, object]) -> None:
    text = (
        "# Prismatic-lite Qwen3 2K pilot\n\n"
        f"- Decision: **{report['decision']}**\n"
        f"- Quality-passed candidates: {report['quality_count']} / 2000\n"
        f"- Sparse accepted additions: {report['accepted_count']}\n"
        f"- Seed-42/43 sparse/dense agreement: {report['seed_sparse_dense_agreement']:.6f}\n"
        f"- G-Vendi null percentile: {report['g_vendi']['percentile']:.6f}\n"
        "- OPD/SFT training performed: no\n"
        "- Original DeepMath rows deleted or modified: no\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise FileExistsError(f"canonical target already exists: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def run_analyze_selection(config: PilotConfig, *, device: str = "cuda:0") -> dict[str, object]:
    if device != "cuda:0":
        raise ValueError("selection analysis requires process-local cuda:0")
    require_phase(config.data_root, "quality-review")
    paths = PilotPaths.from_root(config.data_root)
    quality = strict_jsonl(paths.quality_passed)
    if len(quality) < 1_000:
        raise PilotStopped("quality threshold already failed")
    gradient_rows = strict_jsonl(paths.candidate_gradient_input)
    gradient_ids, gradients = _load_pilot_gradients(
        paths.candidate_gradients, gradient_rows
    )
    question_vectors = _aggregate_candidate_gradients(
        quality, gradient_rows, gradient_ids, gradients
    )
    eligible, _ = _load_eligible_rows(config)
    original_ids, original_gradients = _load_original_gradients(config, eligible)
    state42 = _load_cluster_state(
        paths.cluster_seed42, expected_k=PRIMARY_K, expected_seed=42
    )
    state43 = _load_cluster_state(
        paths.cluster_seed43, expected_k=PRIMARY_K, expected_seed=43
    )
    if str(state42["input_ids_sha256"]) != _ids_sha256(original_ids):
        raise ValueError("seed-42 cluster state input ID hash mismatch")
    labels42 = assign_to_centroids(question_vectors, state42["centroids"])
    labels43 = assign_to_centroids(question_vectors, state43["centroids"])
    selection = candidate_selection(
        labels_seed42=labels42,
        labels_seed43=labels43,
        sparse_seed42=state42["sparse_cluster_ids"].tolist(),
        sparse_seed43=state43["sparse_cluster_ids"].tolist(),
    )
    accepted_indices = selection["accepted_indices"]

    sensitivity: dict[str, object] = {}
    for k in K_SENSITIVITY:
        states: dict[int, dict[str, object]] = {}
        assignments: dict[int, np.ndarray] = {}
        for seed in (42, 43):
            state = _run_cluster_state(
                root=config.data_root,
                gradients=original_gradients,
                ids=original_ids,
                k=k,
                seed=seed,
                reference_repo=config.reference_repo,
            )
            states[seed] = state
            assignments[seed] = assign_to_centroids(
                question_vectors, state["centroids"]
            )
        sparse42 = np.isin(assignments[42], states[42]["sparse_cluster_ids"])
        sparse43 = np.isin(assignments[43], states[43]["sparse_cluster_ids"])
        sensitivity[str(k)] = {
            "seed42_sparse_count": int(sparse42.sum()),
            "seed43_sparse_count": int(sparse43.sum()),
            "seed_sparse_dense_agreement": float(np.mean(sparse42 == sparse43)),
            "states": {
                str(seed): artifact_record(_cluster_state_path(config.data_root, k, seed))
                for seed in (42, 43)
            },
        }

    if accepted_indices:
        g_vendi = selection_null_torch(
            original_gradients,
            question_vectors,
            accepted_indices=accepted_indices,
            draws=SELECTION_NULL_DRAWS,
            seed=42,
            device=device,
            batch_size=4,
        )
    else:
        g_vendi = {
            "draws": 0,
            "seed": 42,
            "observed_g_vendi": 0.0,
            "null_g_vendi": [],
            "null_median": 0.0,
            "percentile": 0.0,
            "not_computed_reason": "accepted_set_empty",
        }

    quality_by_id = {str(row["id"]): row for row in quality}
    candidates = strict_jsonl(paths.candidate_problems)
    quality_rejected = strict_jsonl(
        _extra_path(config.data_root, "candidates/quality_rejected.jsonl")
    )
    rejected_by_id = {str(row["id"]): row for row in quality_rejected}
    quality_position = {str(row["id"]): index for index, row in enumerate(quality)}
    accepted_set = {str(quality[index]["id"]) for index in accepted_indices}
    accepted_records: list[dict[str, object]] = []
    rejected_records: list[dict[str, object]] = []
    for candidate in candidates:
        sample_id = str(candidate["id"])
        if sample_id in rejected_by_id:
            rejected_records.append(
                {
                    **dict(candidate),
                    "stage": "quality",
                    "rejection_reasons": rejected_by_id[sample_id]["rejection_reasons"],
                }
            )
            continue
        position = quality_position[sample_id]
        record = {
            **dict(quality_by_id[sample_id]),
            "seed42_cluster": int(labels42[position]),
            "seed43_cluster": int(labels43[position]),
            "seed42_sparse": sample_id in accepted_set,
            "seed43_sparse": bool(
                labels43[position] in set(state43["sparse_cluster_ids"].tolist())
            ),
        }
        if sample_id in accepted_set:
            accepted_records.append(record)
        else:
            rejected_records.append(
                {
                    **record,
                    "stage": "sparse_filter",
                    "rejection_reasons": ["seed42_dense_cluster"],
                }
            )
    atomic_publish_jsonl(paths.accepted, accepted_records)
    atomic_publish_jsonl(paths.rejected, rejected_records)
    normalized_accepted = [normalized_tokens(str(row["prompt"])) for row in accepted_records]
    contamination_count = sum(
        bool(row.get("benchmark_matches")) for row in accepted_records
    )
    decision_metrics = {
        "quality_count": len(quality),
        "accepted_count": len(accepted_records),
        "seed_sparse_dense_agreement": selection["seed_sparse_dense_agreement"],
        "g_vendi_percentile": g_vendi["percentile"],
        "unique_prompts": len(normalized_accepted) == len(set(normalized_accepted)),
        "benchmark_contamination_count": contamination_count,
    }
    decision = final_pilot_decision(decision_metrics)
    report = {
        "pilot": "prismatic_lite_qwen3_2k",
        **decision_metrics,
        **decision,
        "calibration": strict_json_file(paths.calibration_report, "calibration report"),
        "g_vendi": g_vendi,
        "sensitivity": sensitivity,
        "candidate_count": len(candidates),
        "quality_rejected_count": len(quality_rejected),
        "sparse_rejected_count": len(rejected_records) - len(quality_rejected),
        "accepted_artifact": artifact_record(paths.accepted),
        "rejected_artifact": artifact_record(paths.rejected),
        "training_performed": False,
        "original_rows_deleted_or_modified": False,
    }
    atomic_publish_json(paths.report_json, report)
    _write_report_markdown(paths.report_md, report)
    publish_phase(
        config.data_root,
        "analyze-selection",
        {
            "decision": decision["decision"],
            "quality_count": len(quality),
            "accepted_count": len(accepted_records),
            "report": artifact_record(paths.report_json),
        },
    )
    if decision["decision"] == "no_go":
        raise PilotStopped("final selection failed frozen thresholds")
    return {
        "status": "analyzed",
        "decision": decision["decision"],
        "quality_count": len(quality),
        "accepted_count": len(accepted_records),
    }


def run_validate(config: PilotConfig) -> dict[str, object]:
    _load_root_manifest(config.data_root)
    paths = PilotPaths.from_root(config.data_root)
    if paths.stage_complete.is_file():
        stage = strict_json_file(paths.stage_complete, "stage completion")
        if not isinstance(stage, dict) or stage.get("status") != "complete":
            raise ValueError("invalid stage completion marker")
        return {
            "status": "validated",
            "decision": stage.get("decision"),
            "stop_stage": stage.get("stop_stage"),
        }
    require_phase(config.data_root, "analyze-selection")
    report = strict_json_file(paths.report_json, "final report")
    if not isinstance(report, dict) or report.get("decision") not in {"promising", "no_go"}:
        raise ValueError("final report decision is invalid")
    if report.get("training_performed") is not False:
        raise ValueError("final report training contract is invalid")
    accepted = strict_jsonl(paths.accepted)
    rejected = strict_jsonl(paths.rejected)
    if len(accepted) != report.get("accepted_count"):
        raise ValueError("accepted count does not match final report")
    if len(accepted) + len(rejected) != CANDIDATE_COUNT:
        raise ValueError("selection artifacts do not partition 2,000 candidates")
    if len({str(row["id"]) for row in accepted + rejected}) != CANDIDATE_COUNT:
        raise ValueError("selection artifacts have duplicate or overlapping IDs")
    _require_file_hash(config.source_jsonl, SOURCE_JSONL_SHA256, "prepared source")
    _require_file_hash(
        config.original_gradient_manifest,
        ORIGINAL_GRADIENT_MANIFEST_SHA256,
        "original gradient manifest",
    )
    manifest = _load_root_manifest(config.data_root)
    if hash_directory(config.model_path) != manifest["qwen3"]["files"]:
        raise ValueError("Qwen3 model directory changed during pilot")
    publish_phase(
        config.data_root,
        "validate",
        {
            "decision": report["decision"],
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
        },
    )
    atomic_publish_json(
        paths.stage_complete,
        {
            "pilot": "prismatic_lite_qwen3_2k",
            "status": "complete",
            "decision": report["decision"],
            "stop_stage": None,
            "report": artifact_record(paths.report_json),
            "report_markdown": artifact_record(paths.report_md),
            "accepted": artifact_record(paths.accepted),
            "rejected": artifact_record(paths.rejected),
            "training_performed": False,
        },
    )
    return {
        "status": "validated",
        "decision": report["decision"],
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PILOT_DATA_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in (
        "prepare",
        "generate-calibration-solutions",
        "analyze-calibration",
        "generate-problems",
        "generate-candidate-solutions",
        "quality-review",
        "analyze-selection",
        "validate",
        "estimate",
    ):
        phase = subparsers.add_parser(command)
        if command in {"analyze-calibration", "analyze-selection"}:
            phase.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = PilotConfig(data_root=args.root)
    try:
        if args.command == "estimate":
            result = run_estimate(args.root)
        elif args.command == "prepare":
            result = run_prepare(config)
        elif args.command == "generate-calibration-solutions":
            result = run_generate_calibration_solutions(config)
        elif args.command == "analyze-calibration":
            result = run_analyze_calibration(config, device=args.device)
        elif args.command == "generate-problems":
            result = run_generate_problems(config)
        elif args.command == "generate-candidate-solutions":
            result = run_generate_candidate_solutions(config)
        elif args.command == "quality-review":
            result = run_quality_review(config)
        elif args.command == "analyze-selection":
            result = run_analyze_selection(config, device=args.device)
        elif args.command == "validate":
            result = run_validate(config)
        else:
            parser.error(f"unknown command: {args.command}")
    except PilotStopped as error:
        print(json.dumps({"status": "no_go", "reason": str(error)}, sort_keys=True))
        return NO_GO_EXIT_CODE
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
