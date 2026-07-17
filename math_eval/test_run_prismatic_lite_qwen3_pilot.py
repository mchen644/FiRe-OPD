from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from math_eval.prismatic_lite_pilot_artifacts import (
    PilotPaths,
    atomic_publish_json,
    sha256_file,
)


@pytest.fixture
def pilot_module():
    sys.modules.pop("math_eval.run_prismatic_lite_qwen3_pilot", None)
    module = importlib.import_module("math_eval.run_prismatic_lite_qwen3_pilot")
    return module


def test_import_is_cpu_safe_and_does_not_import_vllm(pilot_module) -> None:
    assert "vllm" not in sys.modules
    assert pilot_module.QWEN_MODEL_PATH == Path(
        "/home/mchen/FiRe-OPD/models/Qwen3-30B-A3B-Instruct-2507"
    )
    assert pilot_module.TENSOR_PARALLEL_SIZE == 4
    assert pilot_module.MAX_MODEL_LEN == 32_768


def test_parser_exposes_every_frozen_phase(pilot_module) -> None:
    parser = pilot_module.build_parser()
    subparsers = next(
        action for action in parser._actions if action.dest == "command"
    )

    assert set(subparsers.choices) == {
        "prepare",
        "generate-calibration-solutions",
        "analyze-calibration",
        "generate-problems",
        "generate-candidate-solutions",
        "quality-review",
        "analyze-selection",
        "validate",
        "estimate",
    }


def test_seed_ledger_is_phase_separated_repeatable_and_unique(pilot_module) -> None:
    values = [
        pilot_module.generation_seed("calibration_solution", index, sample)
        for index in range(5)
        for sample in range(3)
    ]

    assert values == [
        pilot_module.generation_seed("calibration_solution", index, sample)
        for index in range(5)
        for sample in range(3)
    ]
    assert len(values) == len(set(values))
    assert values[0] != pilot_module.generation_seed("candidate_solution", 0, 0)
    assert all(0 <= value < 2**31 for value in values)


def test_phase_markers_enforce_order_and_collision_refusal(
    tmp_path: Path, pilot_module
) -> None:
    with pytest.raises(ValueError, match="required phase prepare"):
        pilot_module.require_phase(tmp_path, "prepare")

    marker = pilot_module.publish_phase(
        tmp_path, "prepare", {"status": "complete", "row_count": 256}
    )
    assert marker == tmp_path / "phases/prepare.json"
    assert pilot_module.require_phase(tmp_path, "prepare")["row_count"] == 256
    with pytest.raises(FileExistsError):
        pilot_module.publish_phase(tmp_path, "prepare", {"status": "changed"})


def test_require_phase_rejects_tampered_bound_artifact(tmp_path: Path, pilot_module) -> None:
    artifact = tmp_path / "value.json"
    atomic_publish_json(artifact, {"value": 1})
    pilot_module.publish_phase(
        tmp_path,
        "prepare",
        {"artifact": pilot_module.artifact_record(artifact)},
    )
    artifact.write_text('{"value":2}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="phase artifact.*changed"):
        pilot_module.require_phase(tmp_path, "prepare")


def test_require_promising_calibration_rejects_no_go_and_missing_report(
    tmp_path: Path, pilot_module
) -> None:
    paths = PilotPaths.from_root(tmp_path)
    with pytest.raises(ValueError, match="calibration report"):
        pilot_module.require_promising_calibration(paths)

    atomic_publish_json(paths.calibration_report, {"decision": "no_go"})
    with pytest.raises(pilot_module.PilotStopped, match="calibration.*no_go"):
        pilot_module.require_promising_calibration(paths)


def _calibration_sample(count: int = 2) -> list[dict]:
    return [
        {
            "id": f"deepmath-{index}",
            "prompt": f"Compute {index}+1.",
            "original_completion": rf"\\boxed{{{index + 1}}}",
            "source_row_index": index,
            "original_dataset_index": index,
            "original_gradient_index": index,
            "topic": "Algebra",
            "difficulty": 6.0,
        }
        for index in range(count)
    ]


class _FakeBackend:
    def __init__(self, batches: list[list[str]]) -> None:
        self.batches = list(batches)
        self.calls: list[dict] = []
        self.closed = False

    def generate(self, messages, *, seeds, temperature, top_p, max_tokens):
        self.calls.append(
            {
                "messages": messages,
                "seeds": list(seeds),
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            }
        )
        values = self.batches.pop(0)
        assert len(values) == len(messages)
        return [
            {"text": value, "output_tokens": 4, "prompt_tokens": 20}
            for value in values
        ]

    def close(self):
        self.closed = True


def test_calibration_generation_uses_exact_three_independent_solutions(
    pilot_module,
) -> None:
    backend = _FakeBackend(
        [
            [
                r"work \\boxed{1}",
                r"other \\boxed{1}",
                r"wrong \\boxed{2}",
                r"work \\boxed{2}",
                r"other \\boxed{2}",
                r"wrong \\boxed{3}",
            ]
        ]
    )

    records, gradient_rows = pilot_module.generate_calibration_records(
        _calibration_sample(), backend
    )

    assert len(records) == 2
    assert all(record["majority_indices"] == [0, 1] for record in records)
    assert all(
        record["model_revision"] == "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
        for record in records
    )
    assert len(gradient_rows) == 4
    assert backend.calls[0]["temperature"] == 0.75
    assert backend.calls[0]["top_p"] == 0.95
    assert backend.calls[0]["max_tokens"] == 16_384
    assert len(backend.calls[0]["seeds"]) == len(set(backend.calls[0]["seeds"])) == 6


def _eligible_rows(count: int = 10) -> list[dict]:
    return [
        {
            "id": f"seed-{index}",
            "prompt": f"Seed problem {index}",
            "completion": rf"\\boxed{{{index}}}",
            "source_row_index": index,
            "original_dataset_index": index,
            "topic": "Algebra",
            "difficulty": float(6 + index % 3),
        }
        for index in range(count)
    ]


def test_problem_generation_stops_at_exact_target_without_hidden_retry(
    pilot_module,
) -> None:
    backend = _FakeBackend(
        [
            [
                "---\n[[Problem]]\nNovel A\n---",
                "malformed",
                "---\n[[Problem]]\nNovel B\n---",
            ],
            [
                "---\n[[Problem]]\nNovel C\n---",
                "---\n[[Problem]]\nNovel D\n---",
            ],
        ]
    )

    problems, attempts = pilot_module.generate_problem_records(
        _eligible_rows(),
        backend,
        target_count=4,
        request_cap=5,
        batch_size=3,
    )

    assert len(problems) == 4
    assert all(
        row["model_revision"] == "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
        for row in problems
    )
    assert [row["prompt"] for row in problems] == [
        "Novel A",
        "Novel B",
        "Novel C",
        "Novel D",
    ]
    assert len(attempts) == 5
    assert attempts[1]["rejection_reason"] == "invalid_problem_boundary"
    assert backend.calls[0]["max_tokens"] == 8_192
    assert backend.calls[0]["temperature"] == 1.0


def test_problem_generation_cap_is_a_declared_stop(pilot_module) -> None:
    backend = _FakeBackend([["bad"], ["still bad"]])

    with pytest.raises(pilot_module.PilotStopped, match="before 2 requests"):
        pilot_module.generate_problem_records(
            _eligible_rows(),
            backend,
            target_count=1,
            request_cap=2,
            batch_size=2,
        )


def test_quality_review_enforces_majority_duplicates_contamination_and_semantics(
    pilot_module,
) -> None:
    candidates = [
        {"id": "c0", "prompt": "A fresh standalone problem", "request_index": 0},
        {
            "id": "c1",
            "prompt": "one two three four five six seven eight nine ten eleven",
            "request_index": 1,
        },
        {
            "id": "c2",
            "prompt": "benchmark alpha beta gamma delta epsilon zeta eta theta iota kappa",
            "request_index": 2,
        },
        {"id": "c3", "prompt": "Another fresh problem", "request_index": 3},
    ]
    solutions = []
    for candidate in candidates:
        solutions.append(
            {
                "id": candidate["id"],
                "prompt": candidate["prompt"],
                "responses": [r"\\boxed{1}", r"\\boxed{1}", r"\\boxed{2}"],
                "majority_indices": [0, 1],
                "prompt_tokens": 100,
                "duplicate_neighbor_id": "seed" if candidate["id"] == "c1" else None,
                "duplicate_jaccard": 1.0 if candidate["id"] == "c1" else 0.0,
                "benchmark_matches": ["bench-0"] if candidate["id"] == "c2" else [],
                "semantic_judgment": "equivalent" if candidate["id"] == "c1" else None,
            }
        )

    passed, rejected, gradient_rows = pilot_module.review_quality_records(
        candidates,
        solutions,
        original_prompts=[("seed", "one two three four five six seven eight nine ten eleven")],
        benchmark_prompts=[
            (
                "bench-0",
                "benchmark alpha beta gamma delta epsilon zeta eta theta iota kappa",
            )
        ],
    )

    assert [row["id"] for row in passed] == ["c0", "c3"]
    reasons = {row["id"]: row["rejection_reasons"] for row in rejected}
    assert "semantic_equivalent" in reasons["c1"]
    assert "benchmark_contamination" in reasons["c2"]
    assert len(gradient_rows) == 4


def test_prepare_manifest_binds_every_frozen_input_hash(tmp_path: Path, pilot_module) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text('{"id":"x"}\n', encoding="utf-8")
    eligibility = tmp_path / "eligibility.json"
    eligibility.write_text('{"eligible_row_count":1}\n', encoding="utf-8")
    gradients = tmp_path / "gradient.manifest.json"
    gradients.write_text('{"projection":{"dimension":1024}}\n', encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")

    value = pilot_module.build_pilot_manifest(
        source_jsonl=source,
        eligibility_report=eligibility,
        original_gradient_manifest=gradients,
        model_path=model,
        model_files=pilot_module.hash_directory(model),
        code_commit="a" * 40,
        code_tree="b" * 40,
        reference_commit="c" * 40,
        reference_tree="d" * 40,
    )

    assert value["source"]["sha256"] == sha256_file(source)
    assert value["eligibility"]["sha256"] == sha256_file(eligibility)
    assert value["original_gradients"]["manifest_sha256"] == sha256_file(gradients)
    assert value["qwen3"]["files"][0]["path"] == "config.json"
    assert value["qwen3"]["revision"] == "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
    assert value["decisions"]["calibration"]["minimum_qualified"] == 192
    assert value["decisions"]["final"]["minimum_accepted"] == 400


def test_estimate_is_read_only(tmp_path: Path, pilot_module, capsys) -> None:
    absent = tmp_path / "absent"
    result = pilot_module.run_estimate(absent)

    assert result["candidate_problem_count"] == 2_000
    assert result["candidate_solution_count"] == 6_000
    assert not absent.exists()
    assert "candidate_solution_count" not in capsys.readouterr().out
