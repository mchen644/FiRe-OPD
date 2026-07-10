import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from math_eval import build_gradient_eligibility as eligibility


def _row(index: int) -> dict:
    return {
        "id": f"deepmath-level6-{index:06d}",
        "prompt": f"Question {index}",
        "completion": f"Reasoning {index} \\boxed{{{index}}}",
        "source_row_index": index,
        "original_dataset_index": index + 100,
        "topic": "Algebra",
        "difficulty": 6.0,
    }


def _manifest(tmp_path: Path, row_count: int) -> dict:
    return {
        "manifest_version": 1,
        "dataset_name": eligibility.DATASET_NAME,
        "dataset_revision": eligibility.DATASET_REVISION,
        "dataset_split": "train",
        "source_parquet": str((tmp_path / "source.parquet").resolve()),
        "source_sha256": "1" * 64,
        "source_row_count": row_count,
        "prepared_jsonl": str((tmp_path / "pool.jsonl").resolve()),
        "prepared_jsonl_sha256": "2" * 64,
        "prepared_row_count": row_count,
        "expected_count": row_count,
    }


class RecordingTokenizer:
    def __init__(self, token_counts: dict[str, int], chat_template: str = "qwen-template"):
        self.token_counts = token_counts
        self.chat_template = chat_template
        self.name_or_path = eligibility.MODEL_NAME
        self.calls: list[tuple[list[list[dict]], dict]] = []

    def apply_chat_template(self, conversations, **kwargs):
        copied = [[dict(message) for message in chat] for chat in conversations]
        self.calls.append((copied, dict(kwargs)))
        return [
            list(range(self.token_counts[chat[0]["content"]]))
            for chat in conversations
        ]


def _build_three_row_report(tmp_path: Path, monkeypatch) -> tuple[list[dict], dict, dict, RecordingTokenizer]:
    rows = [_row(index) for index in range(3)]
    manifest = _manifest(tmp_path, len(rows))
    tokenizer = RecordingTokenizer(
        {
            "Question 0": eligibility.MAX_CONTEXT_TOKENS,
            "Question 1": eligibility.MAX_CONTEXT_TOKENS + 1,
            "Question 2": 7,
        }
    )
    monkeypatch.setattr(eligibility, "TOKENIZATION_BATCH_SIZE", 2)
    monkeypatch.setattr(
        eligibility,
        "PINNED_CHAT_TEMPLATE_SHA256",
        hashlib.sha256(tokenizer.chat_template.encode("utf-8")).hexdigest(),
    )
    report = eligibility.build_eligibility_report(
        rows,
        manifest,
        tokenizer,
        max_context_tokens=eligibility.MAX_CONTEXT_TOKENS,
        max_excluded=1,
        expected_excluded_ids=["deepmath-level6-000001"],
    )
    return rows, manifest, report, tokenizer


def test_build_report_tokenizes_qwen_chats_in_bounded_batches_without_truncation(
    tmp_path: Path, monkeypatch
) -> None:
    rows, manifest, report, tokenizer = _build_three_row_report(tmp_path, monkeypatch)

    assert len(tokenizer.calls) == 2
    assert [len(conversations) for conversations, _ in tokenizer.calls] == [2, 1]
    assert tokenizer.calls[0][0][0] == [
        {"role": "user", "content": rows[0]["prompt"]},
        {"role": "assistant", "content": rows[0]["completion"]},
    ]
    for _, kwargs in tokenizer.calls:
        assert kwargs == {
            "tokenize": True,
            "add_generation_prompt": False,
            "padding": False,
            "truncation": False,
        }

    assert report["manifest_version"] == eligibility.REPORT_VERSION
    assert report["prepared_jsonl"] == manifest["prepared_jsonl"]
    assert report["prepared_jsonl_sha256"] == manifest["prepared_jsonl_sha256"]
    assert report["prepared_manifest_sha256"] is None
    assert report["source_sha256"] == manifest["source_sha256"]
    assert report["source_row_count"] == 3
    assert report["eligible_row_count"] == 2
    assert report["excluded_row_count"] == 1
    assert report["max_context_tokens"] == 32768
    assert report["model_name"] == eligibility.MODEL_NAME
    assert report["model_revision"] == eligibility.MODEL_REVISION
    assert report["tokenizer_name"] == eligibility.MODEL_NAME
    assert report["tokenizer_revision"] == eligibility.MODEL_REVISION
    assert report["transformers_version"] == eligibility.TRANSFORMERS_VERSION
    assert report["chat_template_sha256"] == hashlib.sha256(
        tokenizer.chat_template.encode("utf-8")
    ).hexdigest()
    assert report["excluded_rows"] == [
        {
            "id": "deepmath-level6-000001",
            "source_row_index": 1,
            "original_dataset_index": 101,
            "token_count": 32769,
            "reason": eligibility.CONTEXT_EXCLUSION_REASON,
        }
    ]
    assert report["eligible_ids_sha256"] == hashlib.sha256(
        b"deepmath-level6-000000\ndeepmath-level6-000002\n"
    ).hexdigest()
    assert report["token_length_summary"] == {
        "count": 3,
        "min": 7,
        "mean": pytest.approx((32768 + 32769 + 7) / 3),
        "p50": 32768.0,
        "p90": pytest.approx(32768.8),
        "p95": pytest.approx(32768.9),
        "p99": pytest.approx(32768.98),
        "p999": pytest.approx(32768.998),
        "max": 32769,
    }


def test_build_report_is_deterministic_and_preserves_stable_id_gap(
    tmp_path: Path, monkeypatch
) -> None:
    rows, manifest, first, _ = _build_three_row_report(tmp_path, monkeypatch)
    second = eligibility.build_eligibility_report(
        rows,
        manifest,
        RecordingTokenizer(
            {
                "Question 0": eligibility.MAX_CONTEXT_TOKENS,
                "Question 1": eligibility.MAX_CONTEXT_TOKENS + 1,
                "Question 2": 7,
            }
        ),
        max_context_tokens=eligibility.MAX_CONTEXT_TOKENS,
        max_excluded=1,
        expected_excluded_ids=["deepmath-level6-000001"],
    )
    assert first == second

    report_path = tmp_path / "pool.eligibility.json"
    eligibility.write_or_validate_eligibility_report(report_path, first)
    eligible_rows, loaded = eligibility.apply_eligibility_report(
        rows,
        manifest,
        report_path,
        expected_excluded_ids=["deepmath-level6-000001"],
    )
    assert loaded == first
    assert [row["id"] for row in eligible_rows] == [
        "deepmath-level6-000000",
        "deepmath-level6-000002",
    ]
    assert [row["source_row_index"] for row in eligible_rows] == [0, 2]


@pytest.mark.parametrize(
    ("max_excluded", "expected_ids", "match"),
    [
        (0, ["deepmath-level6-000001"], "exceeds max_excluded"),
        (1, [], "excluded stable IDs mismatch"),
        (1, ["deepmath-level6-000002"], "excluded stable IDs mismatch"),
    ],
)
def test_build_report_rejects_unexpected_context_exclusions(
    tmp_path: Path,
    max_excluded: int,
    expected_ids: list[str],
    match: str,
) -> None:
    rows = [_row(index) for index in range(3)]
    tokenizer = RecordingTokenizer(
        {"Question 0": 2, "Question 1": 32769, "Question 2": 3}
    )
    with pytest.raises(ValueError, match=match):
        eligibility.build_eligibility_report(
            rows,
            _manifest(tmp_path, 3),
            tokenizer,
            max_context_tokens=32768,
            max_excluded=max_excluded,
            expected_excluded_ids=expected_ids,
        )


def test_build_report_requires_pinned_context_boundary(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_context_tokens.*32768"):
        eligibility.build_eligibility_report(
            [_row(0)],
            _manifest(tmp_path, 1),
            RecordingTokenizer({"Question 0": 3}),
            max_context_tokens=8192,
            max_excluded=0,
            expected_excluded_ids=[],
        )


def test_build_report_rejects_noncontiguous_or_duplicate_prepared_identity(
    tmp_path: Path,
) -> None:
    rows = [_row(0), _row(2)]
    with pytest.raises(ValueError, match="stable ID mismatch at row 1"):
        eligibility.build_eligibility_report(
            rows,
            _manifest(tmp_path, 2),
            RecordingTokenizer({"Question 0": 3, "Question 2": 4}),
            max_context_tokens=32768,
            max_excluded=0,
            expected_excluded_ids=[],
        )


def test_write_or_validate_report_is_canonical_idempotent_and_fail_closed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "nested" / "eligibility.json"
    expected = {"z": [2, 1], "a": {"unicode": "解"}}

    assert eligibility.write_or_validate_eligibility_report(path, expected) == expected
    assert path.read_bytes() == (
        json.dumps(expected, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    assert eligibility.write_or_validate_eligibility_report(path, expected) == expected
    assert path.with_name(f".{path.name}.lock").is_file()

    with pytest.raises(ValueError, match="eligibility report mismatch"):
        eligibility.write_or_validate_eligibility_report(path, {"a": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == expected


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda report: report.update(prepared_jsonl_sha256="f" * 64), "prepared_jsonl_sha256 mismatch"),
        (lambda report: report.update(model_revision="wrong"), "model_revision mismatch"),
        (lambda report: report.update(max_context_tokens=4096), "max_context_tokens mismatch"),
        (lambda report: report.update(eligible_row_count=1), "eligible_row_count mismatch"),
        (
            lambda report: report["excluded_rows"][0].update(reason="manual"),
            "exclusion reason mismatch",
        ),
        (lambda report: report.update(eligible_ids_sha256="0" * 64), "eligible_ids_sha256 mismatch"),
    ],
)
def test_apply_report_rejects_provenance_or_eligibility_mutation(
    tmp_path: Path, monkeypatch, mutation, match: str
) -> None:
    rows, manifest, report, _ = _build_three_row_report(tmp_path, monkeypatch)
    mutation(report)
    path = tmp_path / "eligibility.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        eligibility.apply_eligibility_report(
            rows,
            manifest,
            path,
            expected_excluded_ids=["deepmath-level6-000001"],
        )


def test_apply_report_binds_optional_prepared_manifest_file_hash(
    tmp_path: Path, monkeypatch
) -> None:
    rows, manifest, report, _ = _build_three_row_report(tmp_path, monkeypatch)
    manifest_path = tmp_path / "pool.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    report["prepared_manifest"] = str(manifest_path.resolve())
    report["prepared_manifest_sha256"] = eligibility.sha256_file(manifest_path)
    path = tmp_path / "eligibility.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    eligible_rows, _ = eligibility.apply_eligibility_report(
        rows,
        manifest,
        path,
        expected_excluded_ids=["deepmath-level6-000001"],
    )
    assert len(eligible_rows) == 2

    manifest_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prepared_manifest_sha256 mismatch"):
        eligibility.apply_eligibility_report(
            rows,
            manifest,
            path,
            expected_excluded_ids=["deepmath-level6-000001"],
        )


def test_apply_report_does_not_treat_mean_as_a_quantile(
    tmp_path: Path, monkeypatch
) -> None:
    rows = [_row(index) for index in range(3)]
    manifest = _manifest(tmp_path, 3)
    tokenizer = RecordingTokenizer(
        {"Question 0": 1, "Question 1": 1, "Question 2": 100}
    )
    monkeypatch.setattr(
        eligibility,
        "PINNED_CHAT_TEMPLATE_SHA256",
        hashlib.sha256(tokenizer.chat_template.encode("utf-8")).hexdigest(),
    )
    report = eligibility.build_eligibility_report(
        rows,
        manifest,
        tokenizer,
        max_context_tokens=32768,
        max_excluded=0,
        expected_excluded_ids=[],
    )
    assert report["token_length_summary"]["mean"] > report["token_length_summary"]["p50"]
    path = tmp_path / "eligibility.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    eligible_rows, _ = eligibility.apply_eligibility_report(
        rows, manifest, path, expected_excluded_ids=[]
    )
    assert len(eligible_rows) == 3


def test_run_builder_loads_validated_pool_and_binds_manifest_file(
    tmp_path: Path, monkeypatch
) -> None:
    rows = [_row(index) for index in range(3)]
    source_path = tmp_path / "source.parquet"
    source_path.write_bytes(b"source")
    prepared_path = tmp_path / "pool.jsonl"
    prepared_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    manifest = _manifest(tmp_path, 3)
    manifest["source_sha256"] = eligibility.sha256_file(source_path)
    manifest["prepared_jsonl_sha256"] = eligibility.sha256_file(prepared_path)
    manifest_path = tmp_path / "pool.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    tokenizer = RecordingTokenizer(
        {"Question 0": 3, "Question 1": 32769, "Question 2": 4},
        chat_template="pinned-test-template",
    )
    template_hash = hashlib.sha256(
        tokenizer.chat_template.encode("utf-8")
    ).hexdigest()
    monkeypatch.setattr(
        eligibility, "PINNED_CHAT_TEMPLATE_SHA256", template_hash
    )
    monkeypatch.setattr(
        eligibility,
        "PRODUCTION_EXPECTED_EXCLUDED_IDS",
        ("deepmath-level6-000001",),
    )
    monkeypatch.setattr(eligibility, "PRODUCTION_SOURCE_ROW_COUNT", 3)
    monkeypatch.setattr(eligibility, "PRODUCTION_ELIGIBLE_ROW_COUNT", 2)
    monkeypatch.setattr(eligibility, "PRODUCTION_EXCLUDED_ROW_COUNT", 1)
    monkeypatch.setattr(eligibility, "PRODUCTION_MAX_EXCLUDED", 1)
    monkeypatch.setattr(
        eligibility,
        "PRODUCTION_SOURCE_SHA256",
        manifest["source_sha256"],
    )
    monkeypatch.setattr(
        eligibility,
        "PRODUCTION_PREPARED_JSONL_SHA256",
        manifest["prepared_jsonl_sha256"],
    )
    monkeypatch.setattr(
        eligibility,
        "PRODUCTION_ELIGIBLE_IDS_SHA256",
        hashlib.sha256(
            b"deepmath-level6-000000\ndeepmath-level6-000002\n"
        ).hexdigest(),
    )
    monkeypatch.setattr(
        eligibility,
        "PRODUCTION_EXCLUDED_ROWS",
        (
            {
                "id": "deepmath-level6-000001",
                "source_row_index": 1,
                "original_dataset_index": 101,
                "token_count": 32769,
                "reason": eligibility.CONTEXT_EXCLUSION_REASON,
            },
        ),
    )
    monkeypatch.setattr(
        eligibility,
        "PRODUCTION_TOKEN_LENGTH_SUMMARY",
        {
            "count": 3,
            "min": 3,
            "mean": (3 + 32769 + 4) / 3,
            "p50": 4.0,
            "p90": 26216.0,
            "p95": 29492.5,
            "p99": 32113.7,
            "p999": 32703.47,
            "max": 32769,
        },
    )
    monkeypatch.setattr(
        eligibility,
        "load_pinned_tokenizer",
        lambda model_name, model_revision: tokenizer,
    )
    output_path = tmp_path / "reports" / "pool.eligibility.json"

    report = eligibility.run_eligibility_build(
        prepared_jsonl=prepared_path,
        prepared_manifest_path=manifest_path,
        output_path=output_path,
        model_name=eligibility.MODEL_NAME,
        model_revision=eligibility.MODEL_REVISION,
        max_context_tokens=32768,
        max_excluded=1,
        expected_excluded_ids=["deepmath-level6-000001"],
    )

    assert report["prepared_manifest"] == str(manifest_path.resolve())
    assert report["prepared_manifest_sha256"] == eligibility.sha256_file(
        manifest_path
    )
    assert json.loads(output_path.read_text(encoding="utf-8")) == report
    assert report["eligible_row_count"] == 2

    with pytest.raises(ValueError, match="eligibility output overlaps source parquet"):
        eligibility.run_eligibility_build(
            prepared_jsonl=prepared_path,
            prepared_manifest_path=manifest_path,
            output_path=source_path,
            model_name=eligibility.MODEL_NAME,
            model_revision=eligibility.MODEL_REVISION,
            max_context_tokens=32768,
            max_excluded=1,
            expected_excluded_ids=["deepmath-level6-000001"],
        )


def test_load_pinned_tokenizer_uses_exact_revision_and_model_context(
    monkeypatch,
) -> None:
    calls: list[tuple[str, str, dict]] = []
    tokenizer = RecordingTokenizer({"unused": 1})

    class FakeAutoConfig:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            calls.append(("config", name, kwargs))
            return type("Config", (), {"max_position_embeddings": 32768})()

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            calls.append(("tokenizer", name, kwargs))
            return tokenizer

    import transformers

    monkeypatch.setattr(transformers, "AutoConfig", FakeAutoConfig)
    monkeypatch.setattr(transformers, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(
        eligibility,
        "PINNED_CHAT_TEMPLATE_SHA256",
        hashlib.sha256(tokenizer.chat_template.encode("utf-8")).hexdigest(),
    )

    assert (
        eligibility.load_pinned_tokenizer(
            eligibility.MODEL_NAME, eligibility.MODEL_REVISION
        )
        is tokenizer
    )
    assert calls == [
        (
            "config",
            eligibility.MODEL_NAME,
            {"revision": eligibility.MODEL_REVISION},
        ),
        (
            "tokenizer",
            eligibility.MODEL_NAME,
            {"revision": eligibility.MODEL_REVISION},
        ),
    ]


def test_load_pinned_tokenizer_rejects_config_context_mismatch(monkeypatch) -> None:
    class FakeAutoConfig:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            return type("Config", (), {"max_position_embeddings": 131072})()

    import transformers

    monkeypatch.setattr(transformers, "AutoConfig", FakeAutoConfig)
    with pytest.raises(ValueError, match="max_position_embeddings mismatch"):
        eligibility.load_pinned_tokenizer(
            eligibility.MODEL_NAME, eligibility.MODEL_REVISION
        )


def test_apply_report_rejects_generic_fixture_in_production_default_mode(
    tmp_path: Path, monkeypatch
) -> None:
    rows = [_row(index) for index in range(3)]
    manifest = _manifest(tmp_path, 3)
    tokenizer = RecordingTokenizer(
        {"Question 0": 2, "Question 1": 3, "Question 2": 32769}
    )
    monkeypatch.setattr(
        eligibility,
        "PINNED_CHAT_TEMPLATE_SHA256",
        hashlib.sha256(tokenizer.chat_template.encode("utf-8")).hexdigest(),
    )
    report = eligibility.build_eligibility_report(
        rows,
        manifest,
        tokenizer,
        max_context_tokens=32768,
        max_excluded=1,
        expected_excluded_ids=["deepmath-level6-000002"],
    )
    path = tmp_path / "eligibility.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="production source_row_count mismatch"):
        eligibility.apply_eligibility_report(rows, manifest, path)


def test_apply_report_binds_the_callers_actual_prepared_manifest_path(
    tmp_path: Path, monkeypatch
) -> None:
    rows, manifest, report, _ = _build_three_row_report(tmp_path, monkeypatch)
    claimed_path = tmp_path / "claimed.manifest.json"
    actual_path = tmp_path / "actual.manifest.json"
    serialized = json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    claimed_path.write_text(serialized, encoding="utf-8")
    actual_path.write_text(serialized, encoding="utf-8")
    report["prepared_manifest"] = str(claimed_path.resolve())
    report["prepared_manifest_sha256"] = eligibility.sha256_file(claimed_path)
    report_path = tmp_path / "eligibility.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="prepared_manifest path mismatch"):
        eligibility.apply_eligibility_report(
            rows,
            manifest,
            report_path,
            prepared_manifest_path=actual_path,
            expected_excluded_ids=["deepmath-level6-000001"],
        )


def test_direct_file_cli_help_bootstraps_without_pythonpath() -> None:
    script = Path(eligibility.__file__).resolve()
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=script.parents[1],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--prepared-jsonl" in result.stdout


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda report: report.update(unexpected=True), "top-level fields mismatch"),
        (lambda report: report.pop("dataset_name"), "top-level fields mismatch"),
        (
            lambda report: report["token_length_summary"].update(unexpected=True),
            "token_length_summary fields mismatch",
        ),
        (
            lambda report: report["token_length_summary"].pop("p999"),
            "token_length_summary fields mismatch",
        ),
    ],
)
def test_apply_report_rejects_nonexact_report_schema(
    tmp_path: Path, monkeypatch, mutation, match: str
) -> None:
    rows, manifest, report, _ = _build_three_row_report(tmp_path, monkeypatch)
    mutation(report)
    path = tmp_path / "eligibility.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        eligibility.apply_eligibility_report(
            rows,
            manifest,
            path,
            expected_excluded_ids=["deepmath-level6-000001"],
        )


def _install_tiny_production_contract(monkeypatch, manifest: dict, report: dict) -> None:
    monkeypatch.setattr(
        eligibility,
        "PRODUCTION_EXPECTED_EXCLUDED_IDS",
        ("deepmath-level6-000001",),
    )
    monkeypatch.setattr(eligibility, "PRODUCTION_SOURCE_ROW_COUNT", 3)
    monkeypatch.setattr(eligibility, "PRODUCTION_ELIGIBLE_ROW_COUNT", 2)
    monkeypatch.setattr(eligibility, "PRODUCTION_EXCLUDED_ROW_COUNT", 1)
    monkeypatch.setattr(eligibility, "PRODUCTION_MAX_EXCLUDED", 1)
    monkeypatch.setattr(
        eligibility,
        "PRODUCTION_SOURCE_SHA256",
        manifest["source_sha256"],
    )
    monkeypatch.setattr(
        eligibility,
        "PRODUCTION_PREPARED_JSONL_SHA256",
        manifest["prepared_jsonl_sha256"],
    )
    monkeypatch.setattr(
        eligibility,
        "PRODUCTION_ELIGIBLE_IDS_SHA256",
        report["eligible_ids_sha256"],
    )
    monkeypatch.setattr(
        eligibility,
        "PRODUCTION_EXCLUDED_ROWS",
        tuple(dict(entry) for entry in report["excluded_rows"]),
    )
    monkeypatch.setattr(
        eligibility,
        "PRODUCTION_TOKEN_LENGTH_SUMMARY",
        dict(report["token_length_summary"]),
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda report: report["excluded_rows"][0].update(token_count=32770),
            "production excluded_rows mismatch",
        ),
        (
            lambda report: report["token_length_summary"].update(p99=32768.97),
            "production token_length_summary mismatch",
        ),
        (
            lambda report: report.update(source_row_count=4),
            "source_row_count mismatch",
        ),
    ],
)
def test_apply_production_default_rejects_semantic_tampering(
    tmp_path: Path, monkeypatch, mutation, match: str
) -> None:
    rows, manifest, report, _ = _build_three_row_report(tmp_path, monkeypatch)
    _install_tiny_production_contract(monkeypatch, manifest, report)
    mutation(report)
    path = tmp_path / "eligibility.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        eligibility.apply_eligibility_report(rows, manifest, path)
