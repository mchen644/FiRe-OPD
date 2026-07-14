from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from math_eval.opd_proxy_gradient_verify_artifacts import sha256_file
from math_eval.prepare_opd_proxy_gradient_verify import (
    OPD_SUFFIX,
    build_decontamination_audit,
    build_production_contract,
    build_sampling_contract,
    normalize_question,
    publish_root_contract,
    stage_layout,
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
    with pytest.raises(ValueError, match="unsupported stage"):
        stage_layout(3)


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
