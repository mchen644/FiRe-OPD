from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from math_eval.opd_proxy_gradient_stage_profiles import (
    EFFICACY_PILOT,
    capture_algorithm_contract_sha256,
)
from math_eval.opd_proxy_gradient_verify_artifacts import (
    canonical_json_bytes,
    sha256_file,
    sha256_id_lines,
)
from math_eval.prepare_opd_proxy_gradient_verify import (
    FROZEN_STAGE1_MANIFEST_SHA256,
    OPD_SUFFIX,
    SamplingContract,
    build_decontamination_audit,
    build_production_contract,
    build_raw_opd_prompt,
    build_sampling_contract,
    build_stage_rows,
    derive_efficacy_pilot_rows,
    load_efficacy_pilot_parent,
    normalize_question,
    preflight_stage_rows,
    publish_root_contract,
    stage_layout,
    validate_materialized_revision,
    validate_publication_provenance,
    validate_stage_parent,
    validate_tokenizer_compatibility,
    write_stage_artifacts,
    ten_token_gram_hashes,
)


BENCHMARKS = (
    "aime24",
    "aime25",
    "hmmt25_feb",
    "hmmt25_nov",
    "math500",
    "minervamath",
    "olympiadbench",
    "amc2023",
)


@pytest.fixture(scope="module")
def production_inputs() -> dict[str, object]:
    repository = Path(__file__).resolve().parents[1]
    return {
        "prepared_jsonl": repository
        / "data/gradient_diversity/deepmath_level6_r1_solution1.jsonl",
        "prepared_manifest": repository
        / "data/gradient_diversity/deepmath_level6_r1_solution1.manifest.json",
        "eligibility_report": repository
        / "data/gradient_diversity/deepmath_level6_r1_solution1.eligibility.json",
        "benchmark_paths": {
            f"data/{name}/test.jsonl": repository / f"data/{name}/test.jsonl"
            for name in BENCHMARKS
        },
    }


@pytest.fixture(scope="module")
def production_contract(production_inputs):
    return build_production_contract(**production_inputs)


def test_normalize_question_uses_nfkc_casefold_whitespace_and_only_terminal_suffix():
    assert normalize_question("  ＦＩＮＤ   X  \n" + OPD_SUFFIX + "  ") == "find x"
    assert normalize_question(OPD_SUFFIX + " remains") == (
        "please reason step by step, and put your final answer within \\boxed{}. remains"
    )


def test_ten_token_gram_hashes_use_the_pinned_tokenizer_and_unit_separator():
    text = r"\alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
    grams = ten_token_gram_hashes(text)
    tokens = [
        r"\alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "zeta",
        "eta",
        "theta",
        "iota",
        "kappa",
    ]
    expected = hashlib.sha256("\u001f".join(tokens).encode("utf-8")).hexdigest()
    assert expected in grams
    assert len(grams) == 2


def test_normalized_exact_and_shared_ten_gram_are_rejected():
    evaluation = [
        "Find x when alpha beta gamma delta epsilon zeta eta theta iota kappa."
    ]
    training = [
        {
            "id": "q0",
            "prompt": "  FIND x when alpha beta gamma delta epsilon zeta eta theta iota kappa.  ",
        },
        {
            "id": "q1",
            "prompt": "prefix alpha beta gamma delta epsilon zeta eta theta iota kappa suffix",
        },
        {"id": "q2", "prompt": "a genuinely unrelated problem"},
    ]
    audit = build_decontamination_audit(training, {"eval": evaluation})
    assert audit.clean_mask.tolist() == [False, False, True]
    assert audit.exact_match_count == 1
    assert [hit.stable_id for hit in audit.hits] == ["q0", "q1"]
    assert audit.hits[0].normalized_exact is True
    assert audit.hits[1].normalized_exact is False
    assert audit.hits[1].shared_gram_hash is not None
    assert audit.hits[1].shared_gram_text == (
        "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    )


def test_audit_uses_lexicographically_first_benchmark_and_shared_gram():
    training = [
        {
            "id": "q0",
            "prompt": (
                "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
            ),
        }
    ]
    evaluations = {
        "z.jsonl": ["beta gamma delta epsilon zeta eta theta iota kappa lambda"],
        "a.jsonl": ["alpha beta gamma delta epsilon zeta eta theta iota kappa"],
    }
    audit = build_decontamination_audit(training, evaluations)
    assert audit.hits[0].benchmark_path == "a.jsonl"
    assert audit.hits[0].shared_gram_text == (
        "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    )


def test_stage_layouts_are_derived_without_stage1_constant_leakage():
    smoke = stage_layout(0)
    assert smoke.candidate_clean_positions == tuple(range(24))
    assert smoke.held_out_clean_positions == tuple(range(768, 776))
    assert (
        smoke.selected_size,
        smoke.primary_k,
        smoke.diagnostic_k,
        smoke.null_draws,
    ) == (
        5,
        2,
        2,
        100,
    )

    stage1 = stage_layout(1)
    assert len(stage1.candidate_clean_positions) == 768
    assert len(stage1.held_out_clean_positions) == 256
    assert (stage1.selected_size, stage1.primary_k, stage1.diagnostic_k) == (
        172,
        76,
        7,
    )

    stage2 = stage_layout(2)
    assert len(stage2.candidate_clean_positions) == 1536
    assert len(stage2.held_out_clean_positions) == 512
    assert stage2.candidate_clean_positions[:768] == tuple(range(768))
    assert stage2.candidate_clean_positions[768:] == tuple(range(1024, 1792))
    assert (stage2.selected_size, stage2.primary_k, stage2.diagnostic_k) == (
        345,
        153,
        15,
    )
    pilot = stage_layout(EFFICACY_PILOT)
    assert len(pilot.candidate_clean_positions) == 250
    assert len(pilot.held_out_clean_positions) == 84
    assert pilot.candidate_clean_positions == tuple(range(250))
    assert pilot.held_out_clean_positions == tuple(range(768, 852))
    assert (
        pilot.selected_size,
        pilot.primary_k,
        pilot.diagnostic_k,
        pilot.null_draws,
    ) == (56, 25, 3, 10_000)

    with pytest.raises(ValueError, match="unsupported stage"):
        stage_layout(3)


def test_root_stage_builder_rejects_efficacy_pilot_parent_bypass():
    contract = SamplingContract(
        first_2048_ids=tuple(f"q{index}" for index in range(2048)),
        first_2048_ids_sha256="a" * 64,
        first_2048_eligible_positions=tuple(range(2048)),
        first_2048_permutation_positions=tuple(range(2048)),
        scanned_permutation_stop=2048,
        skipped_before_cutoff=(),
        stage_hashes={},
    )
    with pytest.raises(ValueError, match="require the frozen Stage-1 parent"):
        build_stage_rows(
            stage=EFFICACY_PILOT,
            eligible_rows=(),
            source_rows=(),
            sampling_contract=contract,
            tokenizer_4b=object(),
            tokenizer_0_6b=object(),
        )


class _PermutationSpy:
    def __init__(self, permutation: np.ndarray):
        self._permutation = permutation
        self.calls: list[int] = []

    def permutation(self, count: int) -> np.ndarray:
        self.calls.append(count)
        return self._permutation.copy()


def test_sampling_contract_calls_permutation_exactly_once_and_filters_in_order():
    row_count = 57_045
    rows = [
        {"id": f"q{index:05d}", "prompt": f"question {index}"}
        for index in range(row_count)
    ]
    clean_mask = np.ones(row_count, dtype=np.bool_)
    clean_mask[[0, 3, 100]] = False
    audit = build_decontamination_audit(rows, {})
    audit = audit.__class__(
        clean_mask=clean_mask,
        hits=audit.hits,
        exact_match_count=audit.exact_match_count,
        clean_ids_sha256=audit.clean_ids_sha256,
        rejected_ids_sha256=audit.rejected_ids_sha256,
    )
    spy = _PermutationSpy(np.arange(row_count, dtype=np.int64))
    contract = build_sampling_contract(rows, audit, rng=spy)
    assert spy.calls == [57_045]
    assert contract.first_2048_eligible_positions[:4] == (1, 2, 4, 5)
    assert contract.scanned_permutation_stop == 2051
    assert [
        row["eligible_permutation_position"] for row in contract.skipped_before_cutoff
    ] == [
        0,
        3,
        100,
    ]


def test_sampling_contract_rejects_wrong_population_and_insufficient_clean_rows():
    rows = [{"id": f"q{index}", "prompt": f"question {index}"} for index in range(10)]
    audit = build_decontamination_audit(rows, {})
    with pytest.raises(ValueError, match="57,045"):
        build_sampling_contract(rows, audit)

    production_rows = [
        {"id": f"q{index}", "prompt": f"question {index}"} for index in range(57_045)
    ]
    production_audit = build_decontamination_audit(production_rows, {})
    too_few = production_audit.__class__(
        clean_mask=np.zeros(57_045, dtype=np.bool_),
        hits=production_audit.hits,
        exact_match_count=production_audit.exact_match_count,
        clean_ids_sha256=production_audit.clean_ids_sha256,
        rejected_ids_sha256=production_audit.rejected_ids_sha256,
    )
    with pytest.raises(ValueError, match="2,048 clean"):
        build_sampling_contract(production_rows, too_few)


def test_production_clean_population_and_sampling_hashes(production_contract):
    audit, contract = production_contract
    assert audit.exact_match_count == 0
    assert len(audit.hits) == 383
    assert int(audit.clean_mask.sum()) == 56_662
    assert audit.clean_ids_sha256 == (
        "88a1a21ddba4a8aa460963bb404c7bc8f3bf7f6373155fba5eebaad8e52606ee"
    )
    assert audit.rejected_ids_sha256 == (
        "ba92df92881df385d9662c3a7305d05fe73f69e10cd5b3b3fb9a18e88f8699a6"
    )
    assert contract.scanned_permutation_stop == 2064
    assert len(contract.skipped_before_cutoff) == 16
    assert contract.first_2048_ids_sha256 == (
        "aebf0d9a4743b3bb6d2ee307b6194d3d95a948f699a4439b8c4d43f498efbfc4"
    )
    assert contract.stage_hashes == {
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


def test_production_contract_binds_every_benchmark_hash(
    production_inputs, production_contract
):
    audit, _ = production_contract
    expected = {
        relative: sha256_file(path)
        for relative, path in production_inputs["benchmark_paths"].items()
    }
    assert dict(audit.benchmark_hashes) == expected
    assert audit.eligible_ids_sha256 == (
        "5dbb267fed334219978d3d74610793fc8d146add15c488bcc59992779e22d8cf"
    )
    assert {hit.benchmark_path for hit in audit.hits}.issubset(expected)
    assert all(
        expected[hit.benchmark_path] == hit.benchmark_sha256 for hit in audit.hits
    )


def test_root_contract_publication_is_create_once_and_detects_changed_bytes(
    tmp_path: Path, production_contract
):
    audit, contract = production_contract
    publish_root_contract(tmp_path, audit, contract)
    expected = {
        "decontamination.mask.npy",
        "decontamination_hits.jsonl",
        "decontamination_manifest.json",
        "sampling_contract.json",
    }
    assert {
        path.name for path in tmp_path.iterdir() if not path.name.startswith(".")
    } == expected
    first_hashes = {name: sha256_file(tmp_path / name) for name in expected}
    decontamination_manifest = json.loads(
        (tmp_path / "decontamination_manifest.json").read_text(encoding="utf-8")
    )
    assert decontamination_manifest["eligible_ids_sha256"] == audit.eligible_ids_sha256
    assert decontamination_manifest["benchmark_hashes"] == dict(audit.benchmark_hashes)
    publish_root_contract(tmp_path, audit, contract)
    assert {name: sha256_file(tmp_path / name) for name in expected} == first_hashes

    hits = tmp_path / "decontamination_hits.jsonl"
    hits.write_text("corrupted\n", encoding="utf-8")
    with pytest.raises(ValueError, match="existing artifact mismatch"):
        publish_root_contract(tmp_path, audit, contract)
    assert hits.read_text(encoding="utf-8") == "corrupted\n"


class _FakeQwenTokenizer:
    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        *,
        eos: int = 2,
        pad: int = 3,
        raw_length: int = 5,
        sft_extra: int = 3,
        completion_length: int = 2,
    ):
        self._vocab = dict(vocab or {"a": 0, "b": 1, "<eos>": eos, "<pad>": pad})
        self.eos_token_id = eos
        self.pad_token_id = pad
        self.raw_length = raw_length
        self.sft_extra = sft_extra
        self.completion_length = completion_length
        self.embedding_call: dict[str, object] | None = None
        self.chat_template = "fake-qwen-template"

    def get_vocab(self):
        return dict(self._vocab)

    def __len__(self):
        return max(self._vocab.values(), default=-1) + 1

    def apply_chat_template(self, messages, **kwargs):
        if len(messages) == 1 and kwargs.get("add_generation_prompt") is True:
            self.embedding_call = dict(kwargs)
            # IDs 3,4 are the assistant response marker. Any later IDs model
            # Qwen3's disabled-thinking generation-only suffix.
            return [0, 1, 2, 3, 4] + list(range(90, 90 + self.raw_length - 5))
        if len(messages) == 2 and kwargs.get("add_generation_prompt") is False:
            # A supplied assistant response starts immediately after the marker;
            # it does not contain the generation-only disabled-thinking suffix.
            return [0, 1, 2, 3, 4] + list(range(10, 10 + self.sft_extra))
        raise AssertionError((messages, kwargs))

    def __call__(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False}
        if text == "<|im_start|>assistant":
            return {"input_ids": [3, 4]}
        return {"input_ids": list(range(self.completion_length))}


PREPARED_ROW = {
    "id": "deepmath-level6-000000",
    "prompt": "What is one plus one?",
    "completion": "It is two.",
    "source_row_index": 0,
    "original_dataset_index": 9,
    "topic": "Mathematics -> Arithmetic -> Other",
    "difficulty": 1.0,
    "reward_model": {"ground_truth": "2", "style": "rule"},
}


def test_tokenizer_compatibility_checks_every_id_and_special_token():
    left = _FakeQwenTokenizer({"a": 0, "b": 1, "<eos>": 2, "<pad>": 3})
    right = _FakeQwenTokenizer({"a": 0, "b": 4, "<eos>": 2, "<pad>": 3})
    with pytest.raises(ValueError, match="token ID mismatch for 'b'"):
        validate_tokenizer_compatibility(left, right, pair_name="proxy")

    right = _FakeQwenTokenizer({"a": 0, "b": 1, "<eos>": 2, "<pad>": 3}, eos=5)
    with pytest.raises(ValueError, match="EOS"):
        validate_tokenizer_compatibility(left, right, pair_name="target")


def test_stage_rows_use_raw_prompt_and_completion_only_sft_labels():
    tokenizer = _FakeQwenTokenizer()
    rows = preflight_stage_rows([PREPARED_ROW], tokenizer, tokenizer)
    assert rows[0]["raw_opd_messages"] == [
        {
            "role": "user",
            "content": PREPARED_ROW["prompt"] + "\n" + OPD_SUFFIX,
        }
    ]
    assert rows[0]["leaf_topic"] == "Other"
    assert rows[0]["sft_supervised_label_count"] == 3
    assert rows[0]["sft_full_token_count_0_6b"] == 8
    assert tokenizer.embedding_call == {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }


def test_completion_mask_uses_response_marker_not_generation_only_thinking_suffix():
    tokenizer = _FakeQwenTokenizer(raw_length=7, sft_extra=3)
    rows = preflight_stage_rows(
        [PREPARED_ROW], tokenizer, tokenizer, max_prompt_length=10
    )
    assert rows[0]["prompt_token_count_0_6b"] == 7
    assert rows[0]["sft_full_token_count_0_6b"] == 8
    assert rows[0]["sft_supervised_label_count"] == 3


def test_raw_prompt_builder_preserves_exact_question_before_suffix():
    assert build_raw_opd_prompt("question   \n")[0]["content"] == (
        "question   \n\n" + OPD_SUFFIX
    )


@pytest.mark.parametrize(
    ("tokenizer_4b", "tokenizer_small", "max_sft_tokens", "message"),
    [
        (_FakeQwenTokenizer(raw_length=7), _FakeQwenTokenizer(), 20, "4B prompt"),
        (_FakeQwenTokenizer(), _FakeQwenTokenizer(raw_length=7), 20, "0.6B prompt"),
        (
            _FakeQwenTokenizer(),
            _FakeQwenTokenizer(raw_length=5, sft_extra=6),
            10,
            "SFT context",
        ),
    ],
)
def test_stage_preflight_rejects_prompt_and_sft_overflow(
    tokenizer_4b, tokenizer_small, max_sft_tokens, message
):
    with pytest.raises(ValueError, match=message):
        preflight_stage_rows(
            [PREPARED_ROW],
            tokenizer_4b,
            tokenizer_small,
            max_prompt_length=6,
            max_sft_tokens=max_sft_tokens,
        )


def test_stage_preflight_rejects_empty_topic_component_and_zero_supervised_labels():
    malformed = {**PREPARED_ROW, "topic": "Mathematics ->  -> Other"}
    with pytest.raises(ValueError, match="topic"):
        preflight_stage_rows([malformed], _FakeQwenTokenizer(), _FakeQwenTokenizer())
    with pytest.raises(ValueError, match="supervised label"):
        preflight_stage_rows(
            [PREPARED_ROW],
            _FakeQwenTokenizer(),
            _FakeQwenTokenizer(sft_extra=0),
        )


def _stage_contract() -> SamplingContract:
    ids = tuple(f"q{index:04d}" for index in range(2048))
    stage0 = ids[:24] + ids[768:776]
    stage1_candidate = ids[:768]
    stage1_held_out = ids[768:1024]
    stage2_candidate = stage1_candidate + ids[1024:1792]
    stage2_held_out = stage1_held_out + ids[1792:2048]
    return SamplingContract(
        first_2048_ids=ids,
        first_2048_ids_sha256=sha256_id_lines(ids),
        first_2048_eligible_positions=tuple(range(2048)),
        first_2048_permutation_positions=tuple(range(2048)),
        scanned_permutation_stop=2048,
        skipped_before_cutoff=(),
        stage_hashes={
            "stage0_all": sha256_id_lines(stage0),
            "stage1_candidate": sha256_id_lines(stage1_candidate),
            "stage1_held_out": sha256_id_lines(stage1_held_out),
            "stage1_all": sha256_id_lines(stage1_candidate + stage1_held_out),
            "stage2_candidate": sha256_id_lines(stage2_candidate),
            "stage2_held_out": sha256_id_lines(stage2_held_out),
            "stage2_all": sha256_id_lines(stage2_candidate + stage2_held_out),
        },
    )


def _eligible_and_source_rows():
    eligible = []
    source = []
    for index in range(2048):
        question = f"question {index}"
        eligible.append(
            {
                "id": f"q{index:04d}",
                "prompt": question,
                "completion": f"solution {index}",
                "source_row_index": index,
                "original_dataset_index": index + 10_000,
                "topic": "Mathematics -> Other",
                "difficulty": 1.0,
            }
        )
        source.append(
            {
                "prompt": [{"role": "user", "content": question + "\n" + OPD_SUFFIX}],
                "reward_model": {"ground_truth": str(index), "style": "rule"},
            }
        )
    return eligible, source


def test_build_stage_rows_joins_source_by_stable_contract_and_rejects_collision():
    eligible, source = _eligible_and_source_rows()
    contract = _stage_contract()
    rows = build_stage_rows(
        stage=0,
        eligible_rows=eligible,
        source_rows=source,
        sampling_contract=contract,
        tokenizer_4b=_FakeQwenTokenizer(),
        tokenizer_0_6b=_FakeQwenTokenizer(),
    )
    assert len(rows) == 32
    assert [row["split"] for row in rows].count("candidate") == 24
    assert [row["split"] for row in rows].count("held_out") == 8
    assert rows[24]["clean_sample_position"] == 768
    assert rows[0]["reward_model"] == source[0]["reward_model"]

    with pytest.raises(ValueError, match="benchmark collision"):
        build_stage_rows(
            stage=0,
            eligible_rows=eligible,
            source_rows=source,
            sampling_contract=contract,
            tokenizer_4b=_FakeQwenTokenizer(),
            tokenizer_0_6b=_FakeQwenTokenizer(),
            rejected_ids={"q0000"},
        )


def test_build_stage_rows_rejects_source_prompt_drift():
    eligible, source = _eligible_and_source_rows()
    source[0]["prompt"][0]["content"] = "different\n" + OPD_SUFFIX
    with pytest.raises(ValueError, match="source prompt"):
        build_stage_rows(
            stage=0,
            eligible_rows=eligible,
            source_rows=source,
            sampling_contract=_stage_contract(),
            tokenizer_4b=_FakeQwenTokenizer(),
            tokenizer_0_6b=_FakeQwenTokenizer(),
        )


def _run_git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "source"
    repository.mkdir(parents=True)
    _run_git(repository, "init", "--quiet")
    _run_git(repository, "config", "user.email", "test@example.com")
    _run_git(repository, "config", "user.name", "Test User")
    (repository / "tracked.py").write_text("VALUE = 1\n", encoding="utf-8")
    _run_git(repository, "add", ".")
    _run_git(repository, "commit", "--quiet", "-m", "fixture")
    return repository, _run_git(repository, "rev-parse", "HEAD")


def test_publication_provenance_rejects_missing_dirty_wrong_and_bad_reference(
    tmp_path: Path,
):
    source, commit = _git_repository(tmp_path)
    reference = Path("/home/mchen/prismatic-synthesis-reference")
    output = tmp_path / "output"
    canonical = tmp_path / "canonical"

    with pytest.raises(ValueError, match="required"):
        validate_publication_provenance(
            output_root=output,
            repository_root=source,
            expected_source_commit=None,
            reference_repo=None,
            disposable_preflight=False,
            canonical_output_root=canonical,
        )
    with pytest.raises(ValueError, match="source commit"):
        validate_publication_provenance(
            output_root=output,
            repository_root=source,
            expected_source_commit="0" * 40,
            reference_repo=reference,
            disposable_preflight=False,
            canonical_output_root=canonical,
        )

    (source / "untracked.py").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clean"):
        validate_publication_provenance(
            output_root=output,
            repository_root=source,
            expected_source_commit=commit,
            reference_repo=reference,
            disposable_preflight=False,
            canonical_output_root=canonical,
        )
    (source / "untracked.py").unlink()

    bad_reference, _ = _git_repository(tmp_path / "bad")
    with pytest.raises(ValueError, match="reference commit"):
        validate_publication_provenance(
            output_root=output,
            repository_root=source,
            expected_source_commit=commit,
            reference_repo=bad_reference,
            disposable_preflight=False,
            canonical_output_root=canonical,
        )


def test_disposable_preflight_rejects_canonical_output_root(tmp_path: Path):
    source, _ = _git_repository(tmp_path)
    canonical = tmp_path / "canonical"
    with pytest.raises(ValueError, match="disposable.*canonical"):
        validate_publication_provenance(
            output_root=canonical / "nested",
            repository_root=source,
            expected_source_commit=None,
            reference_repo=None,
            disposable_preflight=True,
            canonical_output_root=canonical,
        )


def test_stage2_parent_requires_exact_complete_report_hash(tmp_path: Path):
    report = tmp_path / "report.json"
    report.write_text('{"classification":"pass"}\n', encoding="utf-8")
    digest = sha256_file(report)
    assert validate_stage_parent(2, report, digest)["sha256"] == digest
    with pytest.raises(ValueError, match="parent report hash"):
        validate_stage_parent(2, report, "0" * 64)
    with pytest.raises(ValueError, match="requires.*parent report"):
        validate_stage_parent(2, None, None)
    with pytest.raises(ValueError, match="not accept"):
        validate_stage_parent(1, report, digest)


def test_stage_artifacts_write_manifest_and_target_proxy_capture_parquets(
    tmp_path: Path,
):
    eligible, source = _eligible_and_source_rows()
    contract = _stage_contract()
    rows = build_stage_rows(
        stage=0,
        eligible_rows=eligible,
        source_rows=source,
        sampling_contract=contract,
        tokenizer_4b=_FakeQwenTokenizer(),
        tokenizer_0_6b=_FakeQwenTokenizer(),
    )
    manifest = write_stage_artifacts(
        output_root=tmp_path,
        stage=0,
        rows=rows,
        sampling_contract=contract,
        provenance={"canonical": False, "mode": "disposable_preflight"},
    )
    stage_dir = tmp_path / "stage_0"
    target = pq.read_table(stage_dir / "capture_target.parquet").to_pylist()
    proxy = pq.read_table(stage_dir / "capture_proxy.parquet").to_pylist()
    assert len(target) == 32
    assert len(proxy) == 24
    assert target[0]["opd_verify_stable_id"] == "q0000"
    assert target[0]["opd_verify_split"] == "candidate"
    assert target[0]["extra_info"]["index"] == 10_000
    assert manifest["candidate_count"] == 24
    assert manifest["held_out_count"] == 8
    assert manifest["selected_size"] == 5
    assert json.loads((stage_dir / "manifest.json").read_text()) == manifest

    # Matching create-once publication validates; changed rows cannot overwrite it.
    assert (
        write_stage_artifacts(
            output_root=tmp_path,
            stage=0,
            rows=rows,
            sampling_contract=contract,
            provenance={"canonical": False, "mode": "disposable_preflight"},
        )
        == manifest
    )
    changed = [dict(row) for row in rows]
    changed[0] = {**changed[0], "difficulty": 9.0}
    with pytest.raises(
        ValueError, match="existing artifact mismatch|manifest mismatch"
    ):
        write_stage_artifacts(
            output_root=tmp_path,
            stage=0,
            rows=changed,
            sampling_contract=contract,
            provenance={"canonical": False, "mode": "disposable_preflight"},
        )


def _write_synthetic_stage1_parent(output_root: Path) -> tuple[Path, SamplingContract]:
    eligible, source = _eligible_and_source_rows()
    contract = _stage_contract()
    rows = build_stage_rows(
        stage=1,
        eligible_rows=eligible,
        source_rows=source,
        sampling_contract=contract,
        tokenizer_4b=_FakeQwenTokenizer(),
        tokenizer_0_6b=_FakeQwenTokenizer(),
    )
    write_stage_artifacts(
        output_root=output_root,
        stage=1,
        rows=rows,
        sampling_contract=contract,
        provenance={"canonical": False, "mode": "disposable_preflight"},
    )
    return output_root / "stage_1/manifest.json", contract


def _file_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_frozen_stage1_manifest_digest_is_the_approved_parent():
    assert FROZEN_STAGE1_MANIFEST_SHA256 == (
        "6d698d75995c777d6faaf1abd385a758977ca0350ce862284f1e9d8751eddd83"
    )


def test_pilot_derives_exact_stage1_prefixes_without_mutating_parent(tmp_path: Path):
    parent_path, _ = _write_synthetic_stage1_parent(tmp_path)
    parent_root = parent_path.parent
    before = _file_hashes(parent_root)
    parent = load_efficacy_pilot_parent(parent_path, sha256_file(parent_path))
    rows = derive_efficacy_pilot_rows(parent)

    assert [row["stable_id"] for row in rows[:250]] == [
        f"q{index:04d}" for index in range(250)
    ]
    assert [row["stable_id"] for row in rows[250:]] == [
        f"q{index:04d}" for index in range(768, 852)
    ]
    assert [row["manifest_index"] for row in rows] == list(range(334))
    assert [row["parent_manifest_index"] for row in rows[:250]] == list(range(250))
    assert [row["parent_manifest_index"] for row in rows[250:]] == list(
        range(768, 852)
    )
    assert all(len(str(row["parent_row_sha256"])) == 64 for row in rows)
    assert _file_hashes(parent_root) == before


def test_pilot_parent_rejects_wrong_hash_noncanonical_bytes_and_path_escape(
    tmp_path: Path,
):
    parent_path, _ = _write_synthetic_stage1_parent(tmp_path)
    with pytest.raises(ValueError, match="parent Stage-1 manifest hash"):
        load_efficacy_pilot_parent(parent_path, "0" * 64)

    manifest = json.loads(parent_path.read_text(encoding="utf-8"))
    parent_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical"):
        load_efficacy_pilot_parent(parent_path, sha256_file(parent_path))

    parent_path.write_bytes(canonical_json_bytes(manifest))
    manifest["sample_manifest"] = "../outside.jsonl"
    parent_path.write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(ValueError, match="beneath.*Stage-1"):
        load_efficacy_pilot_parent(parent_path, sha256_file(parent_path))


def test_pilot_parent_rejects_reordered_rows_even_with_updated_sample_hash(
    tmp_path: Path,
):
    parent_path, _ = _write_synthetic_stage1_parent(tmp_path)
    manifest = json.loads(parent_path.read_text(encoding="utf-8"))
    sample_path = parent_path.parent / str(manifest["sample_manifest"])
    rows = [json.loads(line) for line in sample_path.read_text(encoding="utf-8").splitlines()]
    rows[0], rows[1] = rows[1], rows[0]
    sample_path.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))
    manifest["sample_manifest_sha256"] = sha256_file(sample_path)
    manifest["candidate_ids_sha256"] = sha256_id_lines(
        [str(row["stable_id"]) for row in rows[:768]]
    )
    manifest["all_ids_sha256"] = sha256_id_lines(
        [str(row["stable_id"]) for row in rows]
    )
    parent_path.write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(ValueError, match="manifest_index|capture.*order"):
        load_efficacy_pilot_parent(parent_path, sha256_file(parent_path))


def test_pilot_artifacts_publish_in_separate_namespace_with_parent_identity(
    tmp_path: Path,
):
    parent_path, contract = _write_synthetic_stage1_parent(tmp_path)
    parent_before = _file_hashes(parent_path.parent)
    parent_sha = sha256_file(parent_path)
    parent = load_efficacy_pilot_parent(parent_path, parent_sha)
    rows = derive_efficacy_pilot_rows(parent)

    manifest = write_stage_artifacts(
        output_root=tmp_path,
        stage=EFFICACY_PILOT,
        rows=rows,
        sampling_contract=contract,
        provenance={"canonical": False, "mode": "disposable_preflight"},
        parent_stage1_manifest=parent_path,
        expected_parent_stage1_manifest_sha256=parent_sha,
    )

    pilot_root = tmp_path / EFFICACY_PILOT
    assert manifest["stage"] == EFFICACY_PILOT
    assert manifest["candidate_count"] == 250
    assert manifest["held_out_count"] == 84
    assert manifest["target_capture_count"] == 334
    assert manifest["proxy_capture_count"] == 250
    assert manifest["parent_stage_manifest"]["sha256"] == parent_sha
    assert manifest["algorithm_contract_sha256"] == (
        capture_algorithm_contract_sha256(EFFICACY_PILOT)
    )
    assert manifest["candidate_ids_sha256"] == sha256_id_lines(
        [f"q{index:04d}" for index in range(250)]
    )
    assert manifest["held_out_ids_sha256"] == sha256_id_lines(
        [f"q{index:04d}" for index in range(768, 852)]
    )
    target = pq.read_table(pilot_root / "capture_target.parquet").to_pylist()
    proxy = pq.read_table(pilot_root / "capture_proxy.parquet").to_pylist()
    assert len(target) == 334
    assert len(proxy) == 250
    assert [row["opd_verify_stable_id"] for row in target[:250]] == [
        f"q{index:04d}" for index in range(250)
    ]
    assert [row["opd_verify_stable_id"] for row in target[250:]] == [
        f"q{index:04d}" for index in range(768, 852)
    ]
    assert _file_hashes(parent_path.parent) == parent_before
    assert not (parent_path.parent / EFFICACY_PILOT).exists()


def test_stage_artifacts_reject_a_wrong_frozen_stage_hash(tmp_path: Path):
    eligible, source = _eligible_and_source_rows()
    contract = _stage_contract()
    rows = build_stage_rows(
        stage=0,
        eligible_rows=eligible,
        source_rows=source,
        sampling_contract=contract,
        tokenizer_4b=_FakeQwenTokenizer(),
        tokenizer_0_6b=_FakeQwenTokenizer(),
    )
    bad_contract = replace(
        contract,
        stage_hashes={**contract.stage_hashes, "stage0_all": "0" * 64},
    )
    with pytest.raises(ValueError, match="stage0_all.*hash"):
        write_stage_artifacts(
            output_root=tmp_path,
            stage=0,
            rows=rows,
            sampling_contract=bad_contract,
            provenance={"canonical": False},
        )
    assert not (tmp_path / "stage_0/sample_manifest.jsonl").exists()


def test_stage2_capture_parquets_contain_only_appended_rows(tmp_path: Path):
    eligible, source = _eligible_and_source_rows()
    contract = _stage_contract()
    rows = build_stage_rows(
        stage=2,
        eligible_rows=eligible,
        source_rows=source,
        sampling_contract=contract,
        tokenizer_4b=_FakeQwenTokenizer(),
        tokenizer_0_6b=_FakeQwenTokenizer(),
    )
    report = tmp_path / "stage1_report.json"
    report.write_text('{"stage":1,"classification":"pass"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Stage-1 manifest"):
        write_stage_artifacts(
            output_root=tmp_path,
            stage=2,
            rows=rows,
            sampling_contract=contract,
            provenance={"canonical": False},
            parent_report=report,
            expected_parent_report_sha256=sha256_file(report),
        )
    assert not (tmp_path / "stage_2/sample_manifest.jsonl").exists()

    stage1_rows = build_stage_rows(
        stage=1,
        eligible_rows=eligible,
        source_rows=source,
        sampling_contract=contract,
        tokenizer_4b=_FakeQwenTokenizer(),
        tokenizer_0_6b=_FakeQwenTokenizer(),
    )
    write_stage_artifacts(
        output_root=tmp_path,
        stage=1,
        rows=stage1_rows,
        sampling_contract=contract,
        provenance={"canonical": False},
    )
    manifest = write_stage_artifacts(
        output_root=tmp_path,
        stage=2,
        rows=rows,
        sampling_contract=contract,
        provenance={"canonical": False},
        parent_report=report,
        expected_parent_report_sha256=sha256_file(report),
    )
    assert manifest["target_capture_count"] == 1024
    assert manifest["proxy_capture_count"] == 768
    stage_dir = tmp_path / "stage_2"
    target = pq.read_table(stage_dir / "capture_target.parquet").to_pylist()
    proxy = pq.read_table(stage_dir / "capture_proxy.parquet").to_pylist()
    assert len(target) == 1024
    assert len(proxy) == 768
    assert {row["opd_verify_stable_id"] for row in target}.isdisjoint(
        {f"q{index:04d}" for index in range(1024)}
    )


def test_materialized_revision_requires_matching_download_metadata(tmp_path: Path):
    model = tmp_path / "model"
    metadata = model / ".cache/huggingface/download"
    metadata.mkdir(parents=True)
    revision = "c1899de289a04d12100db370d81485cdf75e47ca"
    for name in ("config.json", "tokenizer_config.json", "model.safetensors"):
        (metadata / f"{name}.metadata").write_text(
            revision + "\nblob\ntimestamp\n", encoding="utf-8"
        )
    result = validate_materialized_revision(model, revision)
    assert result["revision"] == revision
    assert result["metadata_file_count"] == 3
    (metadata / "config.json.metadata").write_text(
        "0" * 40 + "\nblob\ntimestamp\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="materialized revision mismatch"):
        validate_materialized_revision(model, revision)
