"""Freeze the benchmark-clean population for OPD proxy-gradient verification."""

from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import io
import json
import os
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from math_eval.build_gradient_eligibility import (
    apply_eligibility_report,
    load_prepared_pool,
)
from math_eval.opd_proxy_gradient_verify_artifacts import (
    atomic_write_bytes,
    canonical_json_bytes,
    sha256_file,
    sha256_id_lines,
)


OPD_SUFFIX = "Please reason step by step, and put your final answer within \\boxed{}."
TEN_TOKEN_RE = re.compile(r"\\[A-Za-z]+|[A-Za-z0-9]+")
SAMPLE_SEED = 2026071401
ELIGIBLE_POPULATION = 57_045
CLEAN_SAMPLE_SIZE = 2_048

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


__all__ = [
    "BENCHMARK_CONTRACT",
    "DecontaminationAudit",
    "DecontaminationHit",
    "OPD_SUFFIX",
    "SamplingContract",
    "StageLayout",
    "build_decontamination_audit",
    "build_production_contract",
    "build_sampling_contract",
    "normalize_question",
    "publish_root_contract",
    "stage_layout",
    "ten_token_gram_hashes",
]
