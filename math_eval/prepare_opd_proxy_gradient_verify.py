"""Freeze the benchmark-clean population for OPD proxy-gradient verification."""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import io
import json
import os
import re
import subprocess
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from math_eval.build_gradient_eligibility import (
    apply_eligibility_report,
    load_prepared_pool,
)
from math_eval.opd_proxy_gradient_verify_artifacts import (
    atomic_write_bytes,
    build_runtime_metadata,
    build_source_snapshot,
    canonical_json_bytes,
    recursive_file_manifest,
    repository_state,
    sha256_file,
    sha256_id_lines,
    SourceFileSpec,
)


OPD_SUFFIX = "Please reason step by step, and put your final answer within \\boxed{}."
TEN_TOKEN_RE = re.compile(r"\\[A-Za-z]+|[A-Za-z0-9]+")
SAMPLE_SEED = 2026071401
ELIGIBLE_POPULATION = 57_045
CLEAN_SAMPLE_SIZE = 2_048
REFERENCE_COMMIT = "d9484cd3b5991030b901ac4a3a9e2472dbfac2ad"
REFERENCE_TREE = "a0079d8c5e15cb18bb4790f99c43cc19bb9ecd50"
SMALL_MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
CANONICAL_OUTPUT_ROOT = Path("/home/mchen/FiRe-OPD/data/opd_proxy_gradient_verify")
DEFAULT_MODEL_PATHS = {
    "target_teacher": Path("models/Qwen3-30B-A3B-Instruct-2507"),
    "target_student": Path("models/Qwen3-4B"),
    "proxy_teacher": Path("models/Qwen3-4B"),
    "proxy_student": Path("models/Qwen3-0.6B"),
}

BENCHMARK_CONTRACT: dict[str, tuple[int, str]] = {
    "data/aime24/test.jsonl": (
        30,
        "a3c49569f3d7125aaf4eea5764bd1b868af534ae3aff6aad0843a7da58fe46b8",
    ),
    "data/aime25/test.jsonl": (
        30,
        "6012af2a112f5e26d91f1b0cc644b5dde6399173b8b648aec2e6b9899877a2db",
    ),
    "data/hmmt25_feb/test.jsonl": (
        30,
        "58d889b807a87fd189562d9aeaf2bddf342b2e956843ebbbd8259fc93195cc3f",
    ),
    "data/hmmt25_nov/test.jsonl": (
        30,
        "37be20ee4d01044638f1d790d938138c2ce6f26fc9e60f6987db5c220783f226",
    ),
    "data/math500/test.jsonl": (
        500,
        "5b126695a919f8fb40edffaf2e748ae08f7fb6ce0721211671c760b4f0b7c5de",
    ),
    "data/minervamath/test.jsonl": (
        272,
        "ff6e07ac93af4e43885fe7710ded51c8db04543d627326b855d903075d59e872",
    ),
    "data/olympiadbench/test.jsonl": (
        674,
        "6c3c658145c21dd76f70eef1456dc5de9bbb38342f2ffdc4a803d8e6e3005ecc",
    ),
    "data/amc2023/test.jsonl": (
        40,
        "b443425b035d98fec3da4de7e347ac43cebcf7b721e96ba8bf9e0120d24dd61d",
    ),
}
EXPECTED_ELIGIBLE_IDS_SHA256 = (
    "5dbb267fed334219978d3d74610793fc8d146add15c488bcc59992779e22d8cf"
)
EXPECTED_CLEAN_IDS_SHA256 = (
    "88a1a21ddba4a8aa460963bb404c7bc8f3bf7f6373155fba5eebaad8e52606ee"
)
EXPECTED_REJECTED_IDS_SHA256 = (
    "ba92df92881df385d9662c3a7305d05fe73f69e10cd5b3b3fb9a18e88f8699a6"
)
EXPECTED_FIRST_2048_SHA256 = (
    "aebf0d9a4743b3bb6d2ee307b6194d3d95a948f699a4439b8c4d43f498efbfc4"
)
EXPECTED_STAGE_HASHES = {
    "stage0_all": "bf4822f0cab82aa5461b30247ee9149506618e61ea087486d27712b2227709df",
    "stage1_candidate": "bd4b790e7db49cabc85aad41de792cce436ce60ee004be505554fd83a1a55e9e",
    "stage1_held_out": "e867617c9b33e6d8881af7dd8b651f64ea7c27ef990f3153222ff34858e33fc7",
    "stage1_all": "5d84fdbc96212e9915ecb32f0ef5b48d437876ec0c02efc801a43e777274acf6",
    "stage2_added_candidate": "4ef21f73c8306702a414964cc817bfbb501764e9af82e311d89ce16230369753",
    "stage2_added_held_out": "dcca926e2ad4a7b0b9be909446ed398f7339cbd988acb2fd69a782a8865397b8",
    "stage2_candidate": "e9b0fa16c2cbeb8e61e0f9e8033a973dc2ab41ce2d2ff93157d53eb012a47811",
    "stage2_held_out": "7dde4b385a7f01cd410bd0f8d0c43a6874a0b94ead78505b4f3ad93aea112ecf",
    "stage2_all": "047ffab516afafc6bc9820351e497ecbbccd240ca4f1504c6c90d583ca040eff",
}


@dataclass(frozen=True)
class StageLayout:
    stage: int
    candidate_clean_positions: tuple[int, ...]
    held_out_clean_positions: tuple[int, ...]
    selected_size: int
    primary_k: int
    diagnostic_k: int
    null_draws: int


@dataclass(frozen=True)
class DecontaminationHit:
    stable_id: str
    eligible_position: int
    normalized_exact: bool
    benchmark_path: str
    benchmark_sha256: str
    shared_gram_hash: str | None
    shared_gram_text: str | None


@dataclass(frozen=True)
class DecontaminationAudit:
    clean_mask: np.ndarray
    hits: tuple[DecontaminationHit, ...]
    exact_match_count: int
    clean_ids_sha256: str
    rejected_ids_sha256: str
    eligible_ids_sha256: str = ""
    benchmark_hashes: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class SamplingContract:
    first_2048_ids: tuple[str, ...]
    first_2048_ids_sha256: str
    first_2048_eligible_positions: tuple[int, ...]
    first_2048_permutation_positions: tuple[int, ...]
    scanned_permutation_stop: int
    skipped_before_cutoff: tuple[dict[str, object], ...]
    stage_hashes: dict[str, str]


@dataclass(frozen=True)
class _BenchmarkCorpus:
    path: str
    sha256: str
    questions: tuple[str, ...]


class _PermutationGenerator(Protocol):
    def permutation(self, count: int) -> np.ndarray: ...


def normalize_question(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("question must be a string")
    normalized = unicodedata.normalize("NFKC", text)
    stripped = normalized.rstrip()
    if stripped.endswith(OPD_SUFFIX):
        stripped = stripped[: -len(OPD_SUFFIX)].rstrip()
    return " ".join(stripped.casefold().split())


def _normalized_tokens(text: str) -> tuple[str, ...]:
    return tuple(TEN_TOKEN_RE.findall(normalize_question(text)))


def _gram_rows(tokens: Sequence[str]) -> tuple[tuple[str, str], ...]:
    rows: list[tuple[str, str]] = []
    for index in range(max(0, len(tokens) - 9)):
        gram_tokens = tokens[index : index + 10]
        gram_text = " ".join(gram_tokens)
        digest = hashlib.sha256("\u001f".join(gram_tokens).encode("utf-8")).hexdigest()
        rows.append((digest, gram_text))
    return tuple(rows)


def ten_token_gram_hashes(text: str) -> frozenset[str]:
    return frozenset(digest for digest, _ in _gram_rows(_normalized_tokens(text)))


def stage_layout(stage: int) -> StageLayout:
    if stage == 0:
        return StageLayout(0, tuple(range(24)), tuple(range(768, 776)), 5, 2, 2, 100)
    if stage == 1:
        return StageLayout(
            1,
            tuple(range(768)),
            tuple(range(768, 1024)),
            172,
            76,
            7,
            10_000,
        )
    if stage == 2:
        candidates = tuple(range(768)) + tuple(range(1024, 1792))
        held_out = tuple(range(768, 1024)) + tuple(range(1792, 2048))
        return StageLayout(2, candidates, held_out, 345, 153, 15, 10_000)
    raise ValueError(f"unsupported stage: {stage}")


def _coerce_benchmarks(
    evaluations: Mapping[str, Sequence[str] | _BenchmarkCorpus],
) -> tuple[_BenchmarkCorpus, ...]:
    corpora: list[_BenchmarkCorpus] = []
    for path in sorted(evaluations):
        value = evaluations[path]
        if isinstance(value, _BenchmarkCorpus):
            corpus = value
            if corpus.path != path:
                raise ValueError("benchmark mapping key/path mismatch")
        else:
            questions = tuple(value)
            if any(
                not isinstance(question, str) or not question for question in questions
            ):
                raise ValueError(f"benchmark {path} contains an invalid question")
            digest = hashlib.sha256(canonical_json_bytes(list(questions))).hexdigest()
            corpus = _BenchmarkCorpus(path, digest, questions)
        corpora.append(corpus)
    return tuple(corpora)


def build_decontamination_audit(
    training_rows: Sequence[Mapping[str, object]],
    evaluations: Mapping[str, Sequence[str] | _BenchmarkCorpus],
) -> DecontaminationAudit:
    """Audit every training row against exact and contiguous ten-token matches."""
    corpora = _coerce_benchmarks(evaluations)
    exact_index: dict[str, list[tuple[str, str]]] = {}
    gram_index: dict[str, list[tuple[str, str, str]]] = {}
    for corpus in corpora:
        for question in corpus.questions:
            normalized = normalize_question(question)
            exact_index.setdefault(normalized, []).append((corpus.path, corpus.sha256))
            for digest, text in _gram_rows(_normalized_tokens(question)):
                gram_index.setdefault(digest, []).append(
                    (corpus.path, corpus.sha256, text)
                )
    for values in exact_index.values():
        values[:] = sorted(set(values))
    for values in gram_index.values():
        values[:] = sorted(set(values))

    clean_mask = np.ones(len(training_rows), dtype=np.bool_)
    hits: list[DecontaminationHit] = []
    stable_ids: list[str] = []
    seen_stable_ids: set[str] = set()
    exact_match_count = 0
    for eligible_position, row in enumerate(training_rows):
        stable_id = row.get("id")
        prompt = row.get("prompt")
        if not isinstance(stable_id, str) or not stable_id:
            raise ValueError(f"training row {eligible_position} has invalid id")
        if stable_id in seen_stable_ids:
            raise ValueError(f"duplicate training stable ID: {stable_id}")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"training row {eligible_position} has invalid prompt")
        stable_ids.append(stable_id)
        seen_stable_ids.add(stable_id)

        normalized = normalize_question(prompt)
        exact_matches = exact_index.get(normalized, [])
        exact = bool(exact_matches)
        if exact:
            exact_match_count += 1
        shared: list[tuple[str, str, str, str]] = []
        for digest, training_text in _gram_rows(_normalized_tokens(prompt)):
            for benchmark_path, benchmark_sha256, benchmark_text in gram_index.get(
                digest, ()
            ):
                # The normalized token sequence is identical by construction.  Use
                # the training rendering to keep one deterministic diagnostic text.
                shared.append(
                    (
                        benchmark_path,
                        benchmark_sha256,
                        digest,
                        min(training_text, benchmark_text),
                    )
                )
        if not exact and not shared:
            continue

        clean_mask[eligible_position] = False
        first_shared = min(set(shared)) if shared else None
        if exact_matches:
            benchmark_path, benchmark_sha256 = min(exact_matches)
        else:
            assert first_shared is not None
            benchmark_path, benchmark_sha256 = first_shared[:2]
        hits.append(
            DecontaminationHit(
                stable_id=stable_id,
                eligible_position=eligible_position,
                normalized_exact=exact,
                benchmark_path=benchmark_path,
                benchmark_sha256=benchmark_sha256,
                shared_gram_hash=first_shared[2] if first_shared else None,
                shared_gram_text=first_shared[3] if first_shared else None,
            )
        )

    clean_ids = [
        stable_id for stable_id, clean in zip(stable_ids, clean_mask.tolist()) if clean
    ]
    rejected_ids = [hit.stable_id for hit in hits]
    return DecontaminationAudit(
        clean_mask=np.ascontiguousarray(clean_mask),
        hits=tuple(hits),
        exact_match_count=exact_match_count,
        clean_ids_sha256=sha256_id_lines(clean_ids),
        rejected_ids_sha256=sha256_id_lines(rejected_ids),
        eligible_ids_sha256=sha256_id_lines(stable_ids),
        benchmark_hashes=tuple((corpus.path, corpus.sha256) for corpus in corpora),
    )


def _stage_hashes(first_ids: Sequence[str]) -> dict[str, str]:
    if len(first_ids) != CLEAN_SAMPLE_SIZE:
        raise ValueError("stage hashes require exactly 2,048 clean IDs")
    stage0 = tuple(first_ids[:24]) + tuple(first_ids[768:776])
    stage1_candidate = tuple(first_ids[:768])
    stage1_held_out = tuple(first_ids[768:1024])
    stage2_added_candidate = tuple(first_ids[1024:1792])
    stage2_added_held_out = tuple(first_ids[1792:2048])
    stage2_candidate = stage1_candidate + stage2_added_candidate
    stage2_held_out = stage1_held_out + stage2_added_held_out
    return {
        "stage0_all": sha256_id_lines(stage0),
        "stage1_candidate": sha256_id_lines(stage1_candidate),
        "stage1_held_out": sha256_id_lines(stage1_held_out),
        "stage1_all": sha256_id_lines(stage1_candidate + stage1_held_out),
        "stage2_added_candidate": sha256_id_lines(stage2_added_candidate),
        "stage2_added_held_out": sha256_id_lines(stage2_added_held_out),
        "stage2_candidate": sha256_id_lines(stage2_candidate),
        "stage2_held_out": sha256_id_lines(stage2_held_out),
        "stage2_all": sha256_id_lines(stage2_candidate + stage2_held_out),
    }


def build_sampling_contract(
    eligible_rows: Sequence[Mapping[str, object]],
    audit: DecontaminationAudit,
    *,
    rng: _PermutationGenerator | None = None,
) -> SamplingContract:
    if len(eligible_rows) != ELIGIBLE_POPULATION:
        raise ValueError("sampling contract requires exactly 57,045 eligible rows")
    if audit.clean_mask.dtype != np.bool_ or audit.clean_mask.shape != (
        ELIGIBLE_POPULATION,
    ):
        raise ValueError("clean mask must be a 57,045-entry Boolean array")
    stable_ids: list[str] = []
    for position, row in enumerate(eligible_rows):
        stable_id = row.get("id")
        if not isinstance(stable_id, str) or not stable_id:
            raise ValueError(f"eligible row {position} has invalid stable ID")
        stable_ids.append(stable_id)
    if len(set(stable_ids)) != len(stable_ids):
        raise ValueError("eligible stable IDs must be unique")
    if int(audit.clean_mask.sum()) < CLEAN_SAMPLE_SIZE:
        raise ValueError("sampling contract requires at least 2,048 clean rows")

    generator: _PermutationGenerator
    if rng is None:
        generator = np.random.Generator(np.random.PCG64(SAMPLE_SEED))
    else:
        generator = rng
    permutation = np.asarray(generator.permutation(ELIGIBLE_POPULATION))
    if permutation.shape != (ELIGIBLE_POPULATION,) or not np.issubdtype(
        permutation.dtype, np.integer
    ):
        raise ValueError("sampling RNG returned an invalid permutation")
    if len(np.unique(permutation)) != ELIGIBLE_POPULATION or (
        permutation.min() != 0 or permutation.max() != ELIGIBLE_POPULATION - 1
    ):
        raise ValueError("sampling RNG did not return a complete permutation")

    hit_by_position = {hit.eligible_position: hit for hit in audit.hits}
    selected_ids: list[str] = []
    selected_eligible_positions: list[int] = []
    selected_permutation_positions: list[int] = []
    skipped: list[dict[str, object]] = []
    scanned_stop: int | None = None
    for permutation_position, raw_eligible_position in enumerate(permutation):
        eligible_position = int(raw_eligible_position)
        if bool(audit.clean_mask[eligible_position]):
            selected_ids.append(stable_ids[eligible_position])
            selected_eligible_positions.append(eligible_position)
            selected_permutation_positions.append(permutation_position)
            if len(selected_ids) == CLEAN_SAMPLE_SIZE:
                scanned_stop = permutation_position + 1
                break
        else:
            hit = hit_by_position.get(eligible_position)
            row: dict[str, object] = {
                "stable_id": stable_ids[eligible_position],
                "eligible_position": eligible_position,
                "eligible_permutation_position": permutation_position,
            }
            if hit is not None:
                row["decontamination_hit"] = dataclasses.asdict(hit)
            skipped.append(row)
    if scanned_stop is None:
        raise RuntimeError("permutation ended before 2,048 clean IDs were found")

    first_ids = tuple(selected_ids)
    return SamplingContract(
        first_2048_ids=first_ids,
        first_2048_ids_sha256=sha256_id_lines(first_ids),
        first_2048_eligible_positions=tuple(selected_eligible_positions),
        first_2048_permutation_positions=tuple(selected_permutation_positions),
        scanned_permutation_stop=scanned_stop,
        skipped_before_cutoff=tuple(skipped),
        stage_hashes=_stage_hashes(first_ids),
    )


def _load_benchmark_corpus(relative_path: str, path: Path) -> _BenchmarkCorpus:
    expected_rows, expected_sha256 = BENCHMARK_CONTRACT[relative_path]
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"benchmark hash mismatch for {relative_path}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    questions: list[str] = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(
                        f"benchmark {relative_path} contains blank line {line_number}"
                    )
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"benchmark {relative_path} row must be an object")
                question = row.get("problem")
                if not isinstance(question, str) or not question:
                    raise ValueError(
                        f"benchmark {relative_path} row {line_number} lacks problem"
                    )
                questions.append(question)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid benchmark {relative_path}: {error}") from error
    if len(questions) != expected_rows:
        raise ValueError(
            f"benchmark row count mismatch for {relative_path}: "
            f"expected {expected_rows}, got {len(questions)}"
        )
    return _BenchmarkCorpus(relative_path, actual_sha256, tuple(questions))


def _validate_production_result(
    audit: DecontaminationAudit, contract: SamplingContract
) -> None:
    expected_values = {
        "exact match count": (audit.exact_match_count, 0),
        "rejected row count": (len(audit.hits), 383),
        "clean row count": (int(audit.clean_mask.sum()), 56_662),
        "eligible ID hash": (
            audit.eligible_ids_sha256,
            EXPECTED_ELIGIBLE_IDS_SHA256,
        ),
        "benchmark hashes": (
            dict(audit.benchmark_hashes),
            {path: digest for path, (_, digest) in BENCHMARK_CONTRACT.items()},
        ),
        "clean ID hash": (audit.clean_ids_sha256, EXPECTED_CLEAN_IDS_SHA256),
        "rejected ID hash": (
            audit.rejected_ids_sha256,
            EXPECTED_REJECTED_IDS_SHA256,
        ),
        "sampling stop": (contract.scanned_permutation_stop, 2064),
        "skipped row count": (len(contract.skipped_before_cutoff), 16),
        "first 2048 hash": (
            contract.first_2048_ids_sha256,
            EXPECTED_FIRST_2048_SHA256,
        ),
        "stage hashes": (contract.stage_hashes, EXPECTED_STAGE_HASHES),
    }
    for description, (actual, expected) in expected_values.items():
        if actual != expected:
            raise ValueError(
                f"production {description} mismatch: expected {expected!r}, got {actual!r}"
            )
    hit_positions = {hit.eligible_position for hit in audit.hits}
    rejected_positions = set(np.flatnonzero(~audit.clean_mask).tolist())
    if hit_positions != rejected_positions:
        raise ValueError("production clean mask and hit rows disagree")


def build_production_contract(
    *,
    prepared_jsonl: Path,
    prepared_manifest: Path,
    eligibility_report: Path,
    benchmark_paths: Mapping[str, Path],
) -> tuple[DecontaminationAudit, SamplingContract]:
    if set(benchmark_paths) != set(BENCHMARK_CONTRACT):
        raise ValueError(
            "production benchmark paths mismatch: "
            f"expected {sorted(BENCHMARK_CONTRACT)}, got {sorted(benchmark_paths)}"
        )
    prepared_rows, prepared_metadata = load_prepared_pool(
        Path(prepared_jsonl), Path(prepared_manifest)
    )
    eligible_rows, _ = apply_eligibility_report(
        prepared_rows,
        prepared_metadata,
        Path(eligibility_report),
        prepared_manifest_path=Path(prepared_manifest).resolve(),
    )
    corpora = {
        relative_path: _load_benchmark_corpus(
            relative_path, benchmark_paths[relative_path]
        )
        for relative_path in sorted(BENCHMARK_CONTRACT)
    }
    audit = build_decontamination_audit(eligible_rows, corpora)
    contract = build_sampling_contract(eligible_rows, audit)
    _validate_production_result(audit, contract)
    return audit, contract


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(array), allow_pickle=False)
    return buffer.getvalue()


def _write_or_validate_bytes(path: Path, payload: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_name(f".{target.name}.lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        if target.exists():
            if not target.is_file() or target.read_bytes() != payload:
                raise ValueError(f"existing artifact mismatch: {target}")
            return
        atomic_write_bytes(target, payload)
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def publish_root_contract(
    output_root: Path,
    audit: DecontaminationAudit,
    contract: SamplingContract,
) -> dict[str, object]:
    """Publish the immutable decontamination and one-permutation root artifacts."""
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    mask_payload = _npy_bytes(audit.clean_mask.astype(np.bool_, copy=False))
    hits_payload = b"".join(
        canonical_json_bytes(dataclasses.asdict(hit)) for hit in audit.hits
    )
    mask_path = root / "decontamination.mask.npy"
    hits_path = root / "decontamination_hits.jsonl"
    _write_or_validate_bytes(mask_path, mask_payload)
    _write_or_validate_bytes(hits_path, hits_payload)

    decontamination_manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "opd_proxy_decontamination",
        "eligible_row_count": int(audit.clean_mask.size),
        "clean_row_count": int(audit.clean_mask.sum()),
        "rejected_row_count": len(audit.hits),
        "normalized_exact_match_count": audit.exact_match_count,
        "eligible_ids_sha256": audit.eligible_ids_sha256,
        "benchmark_hashes": dict(audit.benchmark_hashes),
        "clean_ids_sha256": audit.clean_ids_sha256,
        "rejected_ids_sha256": audit.rejected_ids_sha256,
        "mask": mask_path.name,
        "mask_sha256": hashlib.sha256(mask_payload).hexdigest(),
        "mask_dtype": "bool",
        "mask_shape": [int(audit.clean_mask.size)],
        "hits": hits_path.name,
        "hits_sha256": hashlib.sha256(hits_payload).hexdigest(),
    }
    decontamination_manifest_payload = canonical_json_bytes(decontamination_manifest)
    _write_or_validate_bytes(
        root / "decontamination_manifest.json", decontamination_manifest_payload
    )

    sampling_value: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "opd_proxy_sampling_contract",
        "population_size": ELIGIBLE_POPULATION,
        "sample_seed": SAMPLE_SEED,
        "bit_generator": "PCG64",
        "permutation_call_count": 1,
        "first_2048_ids": list(contract.first_2048_ids),
        "first_2048_ids_sha256": contract.first_2048_ids_sha256,
        "first_2048_eligible_positions": list(contract.first_2048_eligible_positions),
        "first_2048_permutation_positions": list(
            contract.first_2048_permutation_positions
        ),
        "scanned_permutation_stop": contract.scanned_permutation_stop,
        "skipped_before_cutoff": list(contract.skipped_before_cutoff),
        "stage_hashes": dict(contract.stage_hashes),
        "decontamination_manifest_sha256": hashlib.sha256(
            decontamination_manifest_payload
        ).hexdigest(),
    }
    _write_or_validate_bytes(
        root / "sampling_contract.json", canonical_json_bytes(sampling_value)
    )
    return {
        "decontamination": decontamination_manifest,
        "sampling": sampling_value,
    }


def build_raw_opd_prompt(question: str) -> list[dict[str, str]]:
    if not isinstance(question, str) or not question.rstrip():
        raise ValueError("raw OPD question must be a nonempty string")
    return [
        {
            "role": "user",
            "content": question + "\n" + OPD_SUFFIX,
        }
    ]


def _tokenizer_length(tokenizer: object) -> int:
    try:
        length = len(tokenizer)  # type: ignore[arg-type]
    except (TypeError, AttributeError) as error:
        raise ValueError("tokenizer must expose __len__") from error
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise ValueError("tokenizer length must be a positive integer")
    return length


def validate_tokenizer_compatibility(
    left: object, right: object, *, pair_name: str
) -> dict[str, object]:
    if not isinstance(pair_name, str) or not pair_name:
        raise ValueError("tokenizer pair_name must be nonempty")
    for tokenizer, side in ((left, "left"), (right, "right")):
        if not callable(getattr(tokenizer, "get_vocab", None)):
            raise ValueError(f"{pair_name} {side} tokenizer lacks get_vocab")
    left_vocab = left.get_vocab()  # type: ignore[attr-defined]
    right_vocab = right.get_vocab()  # type: ignore[attr-defined]
    if not isinstance(left_vocab, dict) or not isinstance(right_vocab, dict):
        raise ValueError(f"{pair_name} tokenizer vocabularies must be dictionaries")
    for token in sorted(set(left_vocab) | set(right_vocab)):
        left_id = left_vocab.get(token)
        right_id = right_vocab.get(token)
        if left_id != right_id:
            raise ValueError(
                f"{pair_name} token ID mismatch for {token!r}: "
                f"left={left_id!r}, right={right_id!r}"
            )
    left_length = _tokenizer_length(left)
    right_length = _tokenizer_length(right)
    if left_length != right_length:
        raise ValueError(
            f"{pair_name} tokenizer length mismatch: {left_length} != {right_length}"
        )
    left_eos = getattr(left, "eos_token_id", None)
    right_eos = getattr(right, "eos_token_id", None)
    if left_eos != right_eos:
        raise ValueError(
            f"{pair_name} EOS token ID mismatch: {left_eos!r} != {right_eos!r}"
        )
    left_pad = getattr(left, "pad_token_id", None)
    right_pad = getattr(right, "pad_token_id", None)
    if left_pad != right_pad:
        raise ValueError(
            f"{pair_name} padding token ID mismatch: {left_pad!r} != {right_pad!r}"
        )
    ordered_vocab = [[token, int(left_vocab[token])] for token in sorted(left_vocab)]
    return {
        "pair_name": pair_name,
        "tokenizer_length": left_length,
        "eos_token_id": left_eos,
        "pad_token_id": left_pad,
        "vocab_sha256": hashlib.sha256(canonical_json_bytes(ordered_vocab)).hexdigest(),
    }


def _extract_token_ids(value: object, description: str) -> list[int]:
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise ValueError(f"{description} lacks input_ids")
        value = value["input_ids"]
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if (
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], (list, tuple))
    ):
        value = list(value[0])
    if not isinstance(value, list) or any(
        isinstance(token_id, bool) or not isinstance(token_id, int)
        for token_id in value
    ):
        raise ValueError(f"{description} must resolve to one integer token sequence")
    return value


def _apply_chat_tokens(
    tokenizer: object,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    apply = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply):
        raise ValueError("tokenizer lacks apply_chat_template")
    encoded = apply(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )
    return _extract_token_ids(encoded, "chat template output")


def _completion_token_ids(tokenizer: object, completion: str) -> list[int]:
    if not callable(tokenizer):
        raise ValueError("tokenizer is not callable")
    encoded = tokenizer(completion, add_special_tokens=False)  # type: ignore[operator]
    return _extract_token_ids(encoded, "completion tokenization")


def _completion_only_label_count(
    tokenizer: object, full_ids: Sequence[int], *, stable_id: str
) -> int:
    response_marker = _completion_token_ids(
        tokenizer, "<|im_start|>assistant"
    )
    if not response_marker:
        raise ValueError("assistant response marker tokenization is empty")
    starts = [
        index
        for index in range(len(full_ids) - len(response_marker) + 1)
        if list(full_ids[index : index + len(response_marker)]) == response_marker
    ]
    if len(starts) != 1:
        raise ValueError(
            f"SFT sequence for {stable_id} must contain exactly one assistant response marker"
        )
    supervised_start = starts[0] + len(response_marker)
    supervised = len(full_ids) - supervised_start
    if supervised <= 0:
        raise ValueError(f"zero supervised label tokens for {stable_id}")
    return supervised


def _leaf_topic(topic: object) -> str:
    if not isinstance(topic, str) or not topic:
        raise ValueError("topic must be a nonempty string")
    components = [component.strip() for component in topic.split(" -> ")]
    if not components or any(not component for component in components):
        raise ValueError("topic contains a missing or empty component")
    return components[-1]


def preflight_stage_rows(
    rows: Sequence[Mapping[str, object]],
    tokenizer_4b: object,
    tokenizer_0_6b: object,
    *,
    max_prompt_length: int = 2_048,
    max_sft_tokens: int = 40_960,
) -> list[dict[str, object]]:
    if max_prompt_length <= 0 or max_sft_tokens <= 0:
        raise ValueError("token limits must be positive")
    output: list[dict[str, object]] = []
    seen: set[str] = set()
    for row_index, row in enumerate(rows):
        stable_id = row.get("stable_id", row.get("id"))
        question = row.get("prompt")
        completion = row.get("completion")
        if not isinstance(stable_id, str) or not stable_id:
            raise ValueError(f"stage row {row_index} has invalid stable ID")
        if stable_id in seen:
            raise ValueError(f"duplicate stage stable ID: {stable_id}")
        seen.add(stable_id)
        if not isinstance(question, str) or not question:
            raise ValueError(f"stage row {stable_id} has invalid prompt")
        if not isinstance(completion, str) or not completion:
            raise ValueError(f"stage row {stable_id} has invalid completion")
        raw_messages = build_raw_opd_prompt(question)
        prompt_4b = _apply_chat_tokens(
            tokenizer_4b, raw_messages, add_generation_prompt=True
        )
        prompt_small = _apply_chat_tokens(
            tokenizer_0_6b, raw_messages, add_generation_prompt=True
        )
        if len(prompt_4b) > max_prompt_length:
            raise ValueError(
                f"4B prompt exceeds {max_prompt_length} tokens for {stable_id}"
            )
        if len(prompt_small) > max_prompt_length:
            raise ValueError(
                f"0.6B prompt exceeds {max_prompt_length} tokens for {stable_id}"
            )

        sft_messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": completion},
        ]
        sft_full = _apply_chat_tokens(
            tokenizer_0_6b, sft_messages, add_generation_prompt=False
        )
        if len(sft_full) > max_sft_tokens:
            raise ValueError(
                f"SFT context exceeds {max_sft_tokens} tokens for {stable_id}"
            )
        supervised = _completion_only_label_count(
            tokenizer_0_6b, sft_full, stable_id=stable_id
        )
        completion_ids = _completion_token_ids(tokenizer_0_6b, completion)
        if not completion_ids:
            raise ValueError(f"completion tokenization is empty for {stable_id}")

        prepared = dict(row)
        prepared.update(
            {
                "stable_id": stable_id,
                "raw_opd_messages": raw_messages,
                "leaf_topic": _leaf_topic(row.get("topic")),
                "prompt_token_count_4b": len(prompt_4b),
                "prompt_token_count_0_6b": len(prompt_small),
                "r1_completion_token_count_0_6b": len(completion_ids),
                "sft_full_token_count_0_6b": len(sft_full),
                "sft_supervised_label_count": supervised,
            }
        )
        output.append(prepared)
    return output


def _source_prompt_content(source_row: Mapping[str, object], source_index: int) -> str:
    prompt = source_row.get("prompt")
    if (
        not isinstance(prompt, Sequence)
        or isinstance(prompt, (str, bytes))
        or len(prompt) != 1
        or not isinstance(prompt[0], Mapping)
        or prompt[0].get("role") != "user"
        or not isinstance(prompt[0].get("content"), str)
    ):
        raise ValueError(f"source prompt at row {source_index} is malformed")
    return str(prompt[0]["content"])


def build_stage_rows(
    *,
    stage: int,
    eligible_rows: Sequence[Mapping[str, object]],
    source_rows: Sequence[Mapping[str, object]],
    sampling_contract: SamplingContract,
    tokenizer_4b: object,
    tokenizer_0_6b: object,
    rejected_ids: set[str] | frozenset[str] = frozenset(),
    max_prompt_length: int = 2_048,
    max_sft_tokens: int = 40_960,
) -> list[dict[str, object]]:
    if len(sampling_contract.first_2048_ids) != CLEAN_SAMPLE_SIZE:
        raise ValueError("sampling contract does not contain 2,048 IDs")
    if len(sampling_contract.first_2048_eligible_positions) != CLEAN_SAMPLE_SIZE:
        raise ValueError("sampling contract eligible positions are incomplete")
    if len(sampling_contract.first_2048_permutation_positions) != CLEAN_SAMPLE_SIZE:
        raise ValueError("sampling contract permutation positions are incomplete")
    layout = stage_layout(stage)
    clean_positions = layout.candidate_clean_positions + layout.held_out_clean_positions
    joined: list[dict[str, object]] = []
    for manifest_index, clean_position in enumerate(clean_positions):
        eligible_position = sampling_contract.first_2048_eligible_positions[
            clean_position
        ]
        if eligible_position < 0 or eligible_position >= len(eligible_rows):
            raise ValueError("sampling contract eligible position is out of range")
        prepared = eligible_rows[eligible_position]
        stable_id = prepared.get("id")
        if stable_id != sampling_contract.first_2048_ids[clean_position]:
            raise ValueError("sampling contract stable ID/eligible position mismatch")
        if stable_id in rejected_ids:
            raise ValueError(f"sampled benchmark collision for {stable_id}")
        source_index = prepared.get("source_row_index")
        if (
            isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or source_index < 0
            or source_index >= len(source_rows)
        ):
            raise ValueError(f"invalid source row index for {stable_id}")
        source_row = source_rows[source_index]
        question = prepared.get("prompt")
        if not isinstance(question, str) or not question:
            raise ValueError(f"invalid prepared prompt for {stable_id}")
        raw_messages = build_raw_opd_prompt(question)
        if (
            _source_prompt_content(source_row, source_index)
            != raw_messages[0]["content"]
        ):
            raise ValueError(
                f"source prompt differs from prepared exact question for {stable_id}"
            )
        reward_model = source_row.get("reward_model")
        if not isinstance(reward_model, Mapping):
            raise ValueError(f"source reward_model is malformed for {stable_id}")
        split = (
            "candidate"
            if manifest_index < len(layout.candidate_clean_positions)
            else "held_out"
        )
        joined.append(
            {
                "id": stable_id,
                "stable_id": stable_id,
                "source_row_index": source_index,
                "original_dataset_index": prepared.get("original_dataset_index"),
                "split": split,
                "manifest_index": manifest_index,
                "eligible_position": eligible_position,
                "eligible_permutation_position": sampling_contract.first_2048_permutation_positions[
                    clean_position
                ],
                "clean_sample_position": clean_position,
                "prompt": question,
                "completion": prepared.get("completion"),
                "topic": prepared.get("topic"),
                "difficulty": prepared.get("difficulty"),
                "raw_opd_messages": raw_messages,
                "reward_model": dict(reward_model),
            }
        )
    preflighted = preflight_stage_rows(
        joined,
        tokenizer_4b,
        tokenizer_0_6b,
        max_prompt_length=max_prompt_length,
        max_sft_tokens=max_sft_tokens,
    )
    expected_count = len(layout.candidate_clean_positions) + len(
        layout.held_out_clean_positions
    )
    if len(preflighted) != expected_count:
        raise RuntimeError("stage preflight changed row count")
    return preflighted


def _git_output(repository: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"invalid git repository {repository}: {error}") from error
    return result.stdout.strip()


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_publication_provenance(
    *,
    output_root: Path,
    repository_root: Path,
    expected_source_commit: str | None,
    reference_repo: Path | None,
    disposable_preflight: bool,
    canonical_output_root: Path = CANONICAL_OUTPUT_ROOT,
) -> dict[str, object]:
    output = Path(output_root).resolve(strict=False)
    repository = Path(repository_root).resolve(strict=True)
    canonical = Path(canonical_output_root).resolve(strict=False)
    source_state = repository_state(repository)
    if disposable_preflight:
        if output == canonical or _path_is_within(output, canonical):
            raise ValueError(
                "disposable preflight cannot target the canonical output root"
            )
        return {
            "canonical": False,
            "mode": "disposable_preflight",
            "repository": source_state,
            "reference": None,
        }
    if expected_source_commit is None or reference_repo is None:
        raise ValueError(
            "expected_source_commit and reference_repo are required for canonical publication"
        )
    if source_state["head"] != expected_source_commit:
        raise ValueError(
            "source commit mismatch: "
            f"expected {expected_source_commit}, got {source_state['head']}"
        )
    if source_state["status"]:
        raise ValueError("canonical publication requires a clean source repository")
    reference = Path(reference_repo).resolve(strict=True)
    reference_state = repository_state(reference)
    reference_head = reference_state["head"]
    reference_tree = _git_output(reference, "rev-parse", "HEAD^{tree}")
    reference_status = reference_state["status"]
    if reference_head != REFERENCE_COMMIT:
        raise ValueError(
            f"reference commit mismatch: expected {REFERENCE_COMMIT}, got {reference_head}"
        )
    if reference_tree != REFERENCE_TREE:
        raise ValueError(
            f"reference tree mismatch: expected {REFERENCE_TREE}, got {reference_tree}"
        )
    if reference_status:
        raise ValueError("canonical publication requires a clean reference repository")
    return {
        "canonical": True,
        "mode": "canonical",
        "repository": source_state,
        "reference": {
            "path": str(reference),
            "commit": reference_head,
            "tree": reference_tree,
            "status": reference_status,
            "status_sha256": reference_state["status_sha256"],
        },
    }


def validate_stage_parent(
    stage: int,
    parent_report: Path | None,
    expected_parent_report_sha256: str | None,
) -> dict[str, object] | None:
    if stage in {0, 1}:
        if parent_report is not None or expected_parent_report_sha256 is not None:
            raise ValueError(f"stage {stage} does not accept a parent report")
        return None
    if stage != 2:
        raise ValueError(f"unsupported stage: {stage}")
    if parent_report is None or expected_parent_report_sha256 is None:
        raise ValueError("stage 2 requires a parent report and exact hash")
    path = Path(parent_report).resolve(strict=True)
    actual = sha256_file(path)
    if actual != expected_parent_report_sha256:
        raise ValueError(
            "parent report hash mismatch: "
            f"expected {expected_parent_report_sha256}, got {actual}"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid Stage-1 parent report: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("Stage-1 parent report must be a JSON object")
    return {"path": str(path), "sha256": actual}


def _parquet_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    if not rows:
        raise ValueError("capture parquet cannot be empty")
    table = pa.Table.from_pylist([dict(row) for row in rows])
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    return sink.getvalue().to_pybytes()


def _capture_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    capture: list[dict[str, object]] = []
    for row in rows:
        capture.append(
            {
                "data_source": "DeepMath-103K",
                "prompt": row["raw_opd_messages"],
                "ability": "math",
                "reward_model": row["reward_model"],
                "extra_info": {
                    "index": row["original_dataset_index"],
                    "split": "train",
                },
                "opd_verify_stable_id": row["stable_id"],
                "opd_verify_split": row["split"],
                "opd_verify_manifest_index": row["manifest_index"],
            }
        )
    return capture


def _validate_stage1_manifest_parent(
    output_root: Path, sampling_contract: SamplingContract
) -> dict[str, object]:
    path = Path(output_root) / "stage_1/manifest.json"
    if not path.is_file():
        raise ValueError(f"Stage-1 manifest is required before Stage 2: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid Stage-1 manifest: {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError("Stage-1 manifest must be a JSON object")
    expected = {
        "stage": 1,
        "candidate_count": 768,
        "held_out_count": 256,
        "candidate_ids_sha256": sampling_contract.stage_hashes.get(
            "stage1_candidate"
        ),
        "held_out_ids_sha256": sampling_contract.stage_hashes.get(
            "stage1_held_out"
        ),
        "all_ids_sha256": sampling_contract.stage_hashes.get("stage1_all"),
        "sampling_population_ids_sha256": (
            sampling_contract.first_2048_ids_sha256
        ),
    }
    for field, expected_value in expected.items():
        if manifest.get(field) != expected_value:
            raise ValueError(f"Stage-1 manifest field mismatch: {field}")
    artifact_fields = (
        ("sample_manifest", "sample_manifest_sha256"),
        ("target_capture_parquet", "target_capture_parquet_sha256"),
        ("proxy_capture_parquet", "proxy_capture_parquet_sha256"),
    )
    for path_field, hash_field in artifact_fields:
        relative = manifest.get(path_field)
        expected_hash = manifest.get(hash_field)
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise ValueError(f"Stage-1 manifest lacks {path_field}/{hash_field}")
        artifact_path = path.parent / relative
        if not artifact_path.is_file() or sha256_file(artifact_path) != expected_hash:
            raise ValueError(f"Stage-1 manifest artifact hash mismatch: {path_field}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def write_stage_artifacts(
    *,
    output_root: Path,
    stage: int,
    rows: Sequence[Mapping[str, object]],
    sampling_contract: SamplingContract,
    provenance: Mapping[str, object],
    parent_report: Path | None = None,
    expected_parent_report_sha256: str | None = None,
) -> dict[str, object]:
    layout = stage_layout(stage)
    expected_candidates = len(layout.candidate_clean_positions)
    expected_held_out = len(layout.held_out_clean_positions)
    normalized_rows = [dict(row) for row in rows]
    if len(normalized_rows) != expected_candidates + expected_held_out:
        raise ValueError("stage row count does not match its derived layout")
    stable_ids = [row.get("stable_id") for row in normalized_rows]
    if any(not isinstance(stable_id, str) or not stable_id for stable_id in stable_ids):
        raise ValueError("stage rows require nonempty stable IDs")
    if len(set(stable_ids)) != len(stable_ids):
        raise ValueError("stage rows contain duplicate stable IDs")
    splits = [row.get("split") for row in normalized_rows]
    if splits != ["candidate"] * expected_candidates + ["held_out"] * expected_held_out:
        raise ValueError("stage row split/order mismatch")
    for index, row in enumerate(normalized_rows):
        if row.get("manifest_index") != index:
            raise ValueError("stage manifest_index must match row order")
    parent = validate_stage_parent(stage, parent_report, expected_parent_report_sha256)
    parent_stage_manifest = (
        _validate_stage1_manifest_parent(output_root, sampling_contract)
        if stage == 2
        else None
    )
    candidate_ids = [str(value) for value in stable_ids[:expected_candidates]]
    held_out_ids = [str(value) for value in stable_ids[expected_candidates:]]
    all_ids = candidate_ids + held_out_ids
    hash_checks = {
        0: {"stage0_all": all_ids},
        1: {
            "stage1_candidate": candidate_ids,
            "stage1_held_out": held_out_ids,
            "stage1_all": all_ids,
        },
        2: {
            "stage2_added_candidate": candidate_ids[768:],
            "stage2_added_held_out": held_out_ids[256:],
            "stage2_candidate": candidate_ids,
            "stage2_held_out": held_out_ids,
            "stage2_all": all_ids,
        },
    }[stage]
    for hash_name, ids in hash_checks.items():
        expected_hash = sampling_contract.stage_hashes.get(hash_name)
        if expected_hash is not None and sha256_id_lines(ids) != expected_hash:
            raise ValueError(f"{hash_name} ordered-ID hash mismatch")

    stage_dir = Path(output_root) / f"stage_{stage}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    sample_payload = b"".join(canonical_json_bytes(row) for row in normalized_rows)
    if stage == 2:
        target_source_rows = [
            row
            for row in normalized_rows
            if int(row["clean_sample_position"]) >= 1024
        ]
        proxy_source_rows = [
            row for row in target_source_rows if row["split"] == "candidate"
        ]
        if len(target_source_rows) != 1024 or len(proxy_source_rows) != 768:
            raise ValueError("Stage-2 capture inputs must contain only appended rows")
    else:
        target_source_rows = normalized_rows
        proxy_source_rows = normalized_rows[:expected_candidates]
    target_rows = _capture_rows(target_source_rows)
    proxy_rows = _capture_rows(proxy_source_rows)
    target_payload = _parquet_bytes(target_rows)
    proxy_payload = _parquet_bytes(proxy_rows)
    sample_path = stage_dir / "sample_manifest.jsonl"
    target_path = stage_dir / "capture_target.parquet"
    proxy_path = stage_dir / "capture_proxy.parquet"
    _write_or_validate_bytes(sample_path, sample_payload)
    _write_or_validate_bytes(target_path, target_payload)
    _write_or_validate_bytes(proxy_path, proxy_payload)

    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "opd_proxy_stage",
        "stage": stage,
        "candidate_count": expected_candidates,
        "held_out_count": expected_held_out,
        "selected_size": layout.selected_size,
        "primary_k": layout.primary_k,
        "diagnostic_k": layout.diagnostic_k,
        "null_draws": layout.null_draws,
        "candidate_ids_sha256": sha256_id_lines(candidate_ids),
        "held_out_ids_sha256": sha256_id_lines(held_out_ids),
        "all_ids_sha256": sha256_id_lines(all_ids),
        "sampling_population_ids_sha256": sampling_contract.first_2048_ids_sha256,
        "sample_manifest": sample_path.name,
        "sample_manifest_sha256": hashlib.sha256(sample_payload).hexdigest(),
        "target_capture_count": len(target_rows),
        "target_capture_ids_sha256": sha256_id_lines(
            [str(row["opd_verify_stable_id"]) for row in target_rows]
        ),
        "proxy_capture_count": len(proxy_rows),
        "proxy_capture_ids_sha256": sha256_id_lines(
            [str(row["opd_verify_stable_id"]) for row in proxy_rows]
        ),
        "target_capture_parquet": target_path.name,
        "target_capture_parquet_sha256": hashlib.sha256(target_payload).hexdigest(),
        "proxy_capture_parquet": proxy_path.name,
        "proxy_capture_parquet_sha256": hashlib.sha256(proxy_payload).hexdigest(),
        "parent_report": parent,
        "parent_stage_manifest": parent_stage_manifest,
        "provenance": dict(provenance),
    }
    manifest_path = stage_dir / "manifest.json"
    _write_or_validate_bytes(manifest_path, canonical_json_bytes(manifest))
    return manifest


def _read_json_object(path: Path, description: str) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _sampling_contract_from_json(path: Path) -> SamplingContract:
    value = _read_json_object(path, "sampling contract")
    required = {
        "first_2048_ids",
        "first_2048_ids_sha256",
        "first_2048_eligible_positions",
        "first_2048_permutation_positions",
        "scanned_permutation_stop",
        "skipped_before_cutoff",
        "stage_hashes",
    }
    if not required.issubset(value):
        raise ValueError("sampling contract is missing required fields")
    first_ids = tuple(value["first_2048_ids"])
    if sha256_id_lines(first_ids) != value["first_2048_ids_sha256"]:
        raise ValueError("sampling contract ID hash mismatch")
    return SamplingContract(
        first_2048_ids=first_ids,
        first_2048_ids_sha256=str(value["first_2048_ids_sha256"]),
        first_2048_eligible_positions=tuple(
            int(item) for item in value["first_2048_eligible_positions"]
        ),
        first_2048_permutation_positions=tuple(
            int(item) for item in value["first_2048_permutation_positions"]
        ),
        scanned_permutation_stop=int(value["scanned_permutation_stop"]),
        skipped_before_cutoff=tuple(
            dict(item) for item in value["skipped_before_cutoff"]
        ),
        stage_hashes={
            str(key): str(item) for key, item in value["stage_hashes"].items()
        },
    )


def _default_data_paths(repository_root: Path) -> dict[str, object]:
    return {
        "prepared_jsonl": repository_root
        / "data/gradient_diversity/deepmath_level6_r1_solution1.jsonl",
        "prepared_manifest": repository_root
        / "data/gradient_diversity/deepmath_level6_r1_solution1.manifest.json",
        "eligibility_report": repository_root
        / "data/gradient_diversity/deepmath_level6_r1_solution1.eligibility.json",
        "benchmark_paths": {
            relative: repository_root / relative for relative in BENCHMARK_CONTRACT
        },
    }


def _resolved_model_paths(repository_root: Path) -> dict[str, Path]:
    return {
        name: (repository_root / relative).resolve(strict=True)
        for name, relative in DEFAULT_MODEL_PATHS.items()
    }


def validate_materialized_revision(
    model_path: Path, expected_revision: str
) -> dict[str, object]:
    root = Path(model_path).resolve(strict=True)
    metadata_root = root / ".cache/huggingface/download"
    if not metadata_root.is_dir():
        raise ValueError(f"materialized model lacks download metadata: {root}")
    metadata_files = sorted(metadata_root.glob("*.metadata"), key=lambda path: path.name)
    required_names = {"config.json.metadata", "tokenizer_config.json.metadata"}
    names = {path.name for path in metadata_files}
    if not required_names.issubset(names) or not any(
        name.startswith("model") and ".safetensors" in name for name in names
    ):
        raise ValueError("materialized model metadata is incomplete")
    rows: list[dict[str, object]] = []
    for path in metadata_files:
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines or lines[0] != expected_revision:
            actual = lines[0] if lines else None
            raise ValueError(
                "materialized revision mismatch for "
                f"{path.name}: expected {expected_revision}, got {actual}"
            )
        rows.append(
            {
                "path": path.name,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "revision": expected_revision,
        "metadata_file_count": len(rows),
        "metadata": rows,
        "metadata_manifest_sha256": hashlib.sha256(
            canonical_json_bytes(rows)
        ).hexdigest(),
    }


def _source_snapshot(repository_root: Path) -> dict[str, object]:
    paths = [
        "math_eval/opd_proxy_gradient_verify_artifacts.py",
        "math_eval/prepare_opd_proxy_gradient_verify.py",
        "math_eval/test_prepare_opd_proxy_gradient_verify.py",
        "math_eval/build_gradient_eligibility.py",
        "math_eval/deepmath_gradient_diversity.py",
        "docs/superpowers/specs/2026-07-14-vanilla-opd-proxy-gradient-selection-verify-design.md",
        "docs/superpowers/plans/2026-07-14-vanilla-opd-proxy-gradient-selection-verify.md",
    ]
    return build_source_snapshot(
        {"main": repository_root},
        [SourceFileSpec("main", path) for path in paths],
    )


def prepare_root(
    *,
    output_root: Path,
    repository_root: Path | None = None,
    expected_source_commit: str | None = None,
    reference_repo: Path | None = None,
    disposable_preflight: bool = False,
) -> dict[str, object]:
    repository = (
        Path(repository_root).resolve(strict=True)
        if repository_root is not None
        else Path(__file__).resolve().parents[1]
    )
    provenance = validate_publication_provenance(
        output_root=output_root,
        repository_root=repository,
        expected_source_commit=expected_source_commit,
        reference_repo=reference_repo,
        disposable_preflight=disposable_preflight,
    )
    paths = _default_data_paths(repository)
    audit, sampling = build_production_contract(**paths)
    model_paths = _resolved_model_paths(repository)
    manifest_cache: dict[Path, dict[str, object]] = {}
    model_manifests: dict[str, dict[str, object]] = {}
    for name, path in model_paths.items():
        if path not in manifest_cache:
            manifest_cache[path] = recursive_file_manifest(path)
        model_manifests[name] = manifest_cache[path]
    small_model_materialization = validate_materialized_revision(
        model_paths["proxy_student"], SMALL_MODEL_REVISION
    )
    if (
        model_manifests["target_student"]["manifest_sha256"]
        != model_manifests["proxy_teacher"]["manifest_sha256"]
    ):
        raise ValueError("target-student and proxy-teacher 4B model bytes differ")
    # Validate every expensive/model-bound input before publishing any root bytes.
    publish_root_contract(output_root, audit, sampling)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "opd_proxy_root",
        "provenance": provenance,
        "source_snapshot": _source_snapshot(repository),
        "runtime": build_runtime_metadata("verl_capture"),
        "model_manifests": model_manifests,
        "small_model_revision": SMALL_MODEL_REVISION,
        "small_model_materialization": small_model_materialization,
        "prepared_jsonl_sha256": sha256_file(paths["prepared_jsonl"]),
        "prepared_manifest_sha256": sha256_file(paths["prepared_manifest"]),
        "eligibility_report_sha256": sha256_file(paths["eligibility_report"]),
        "decontamination_manifest_sha256": sha256_file(
            Path(output_root) / "decontamination_manifest.json"
        ),
        "sampling_contract_sha256": sha256_file(
            Path(output_root) / "sampling_contract.json"
        ),
    }
    _write_or_validate_bytes(
        Path(output_root) / "root_manifest.json", canonical_json_bytes(manifest)
    )
    return manifest


def _load_tokenizer(path: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=False,
    )


def _small_model_context(path: Path) -> int:
    config = _read_json_object(path / "config.json", "small-model config")
    value = config.get("max_position_embeddings")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("small-model max_position_embeddings is invalid")
    return value


def prepare_stage(
    *,
    stage: int,
    output_root: Path,
    repository_root: Path | None = None,
    expected_source_commit: str | None = None,
    reference_repo: Path | None = None,
    disposable_preflight: bool = False,
    parent_report: Path | None = None,
) -> dict[str, object]:
    repository = (
        Path(repository_root).resolve(strict=True)
        if repository_root is not None
        else Path(__file__).resolve().parents[1]
    )
    publication = validate_publication_provenance(
        output_root=output_root,
        repository_root=repository,
        expected_source_commit=expected_source_commit,
        reference_repo=reference_repo,
        disposable_preflight=disposable_preflight,
    )
    if stage == 2:
        if parent_report is None:
            raise ValueError("stage 2 requires --parent-report")
        parent_sha256 = sha256_file(parent_report)
    else:
        parent_sha256 = None
    # Parent validation is deliberately before tokenization or row derivation.
    validate_stage_parent(stage, parent_report, parent_sha256)

    root_manifest_path = Path(output_root) / "root_manifest.json"
    root_manifest = _read_json_object(root_manifest_path, "root manifest")
    sampling_path = Path(output_root) / "sampling_contract.json"
    decontamination_path = Path(output_root) / "decontamination_manifest.json"
    if root_manifest.get("sampling_contract_sha256") != sha256_file(sampling_path):
        raise ValueError("root manifest sampling-contract hash mismatch")
    if root_manifest.get("decontamination_manifest_sha256") != sha256_file(
        decontamination_path
    ):
        raise ValueError("root manifest decontamination hash mismatch")
    current_snapshot = _source_snapshot(repository)
    root_snapshot = root_manifest.get("source_snapshot")
    if not isinstance(root_snapshot, Mapping) or root_snapshot.get(
        "manifest_sha256"
    ) != current_snapshot.get("manifest_sha256"):
        raise ValueError("root manifest source snapshot mismatch")
    sampling = _sampling_contract_from_json(sampling_path)
    if stage == 2:
        _validate_stage1_manifest_parent(output_root, sampling)
    paths = _default_data_paths(repository)
    prepared_rows, prepared_metadata = load_prepared_pool(
        paths["prepared_jsonl"], paths["prepared_manifest"]
    )
    eligible_rows, _ = apply_eligibility_report(
        prepared_rows,
        prepared_metadata,
        paths["eligibility_report"],
        prepared_manifest_path=Path(paths["prepared_manifest"]).resolve(),
    )
    source_rows = pq.read_table(prepared_metadata["source_parquet"]).to_pylist()
    model_paths = _resolved_model_paths(repository)
    tokenizer_30b = _load_tokenizer(model_paths["target_teacher"])
    tokenizer_4b = _load_tokenizer(model_paths["target_student"])
    tokenizer_small = _load_tokenizer(model_paths["proxy_student"])
    compatibility = {
        "target": validate_tokenizer_compatibility(
            tokenizer_30b, tokenizer_4b, pair_name="target"
        ),
        "proxy": validate_tokenizer_compatibility(
            tokenizer_4b, tokenizer_small, pair_name="proxy"
        ),
    }
    hits_path = Path(output_root) / "decontamination_hits.jsonl"
    rejected_ids: set[str] = set()
    with hits_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rejected_ids.add(str(json.loads(line)["stable_id"]))
    rows = build_stage_rows(
        stage=stage,
        eligible_rows=eligible_rows,
        source_rows=source_rows,
        sampling_contract=sampling,
        tokenizer_4b=tokenizer_4b,
        tokenizer_0_6b=tokenizer_small,
        rejected_ids=rejected_ids,
        max_sft_tokens=_small_model_context(model_paths["proxy_student"]),
    )
    provenance = {
        "publication": publication,
        "root_manifest_path": str(root_manifest_path.resolve()),
        "root_manifest_sha256": sha256_file(root_manifest_path),
        "root_manifest": root_manifest,
        "tokenizer_compatibility": compatibility,
        "source_snapshot": current_snapshot,
        "runtime": build_runtime_metadata("verl_capture"),
    }
    return write_stage_artifacts(
        output_root=output_root,
        stage=stage,
        rows=rows,
        sampling_contract=sampling,
        provenance=provenance,
        parent_report=parent_report,
        expected_parent_report_sha256=parent_sha256,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare immutable OPD proxy-gradient verification artifacts"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare-root", "prepare-stage"):
        child = subparsers.add_parser(command)
        child.add_argument("--output-root", type=Path, required=True)
        child.add_argument("--expected-source-commit")
        child.add_argument("--reference-repo", type=Path)
        child.add_argument("--disposable-preflight", action="store_true")
    stage_parser = subparsers.choices["prepare-stage"]
    stage_parser.add_argument("--stage", type=int, choices=(0, 1, 2), required=True)
    stage_parser.add_argument("--parent-report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    common = {
        "output_root": args.output_root,
        "expected_source_commit": args.expected_source_commit,
        "reference_repo": args.reference_repo,
        "disposable_preflight": args.disposable_preflight,
    }
    if args.command == "prepare-root":
        result = prepare_root(**common)
    else:
        result = prepare_stage(
            stage=args.stage,
            parent_report=args.parent_report,
            **common,
        )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


__all__ = [
    "BENCHMARK_CONTRACT",
    "DecontaminationAudit",
    "DecontaminationHit",
    "OPD_SUFFIX",
    "SamplingContract",
    "StageLayout",
    "build_decontamination_audit",
    "build_production_contract",
    "build_raw_opd_prompt",
    "build_sampling_contract",
    "build_stage_rows",
    "main",
    "normalize_question",
    "preflight_stage_rows",
    "prepare_root",
    "prepare_stage",
    "publish_root_contract",
    "stage_layout",
    "ten_token_gram_hashes",
    "validate_materialized_revision",
    "validate_publication_provenance",
    "validate_stage_parent",
    "validate_tokenizer_compatibility",
    "write_stage_artifacts",
]


if __name__ == "__main__":
    raise SystemExit(main())
