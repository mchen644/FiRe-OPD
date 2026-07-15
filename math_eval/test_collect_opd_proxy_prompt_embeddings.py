from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from math_eval.collect_opd_proxy_prompt_embeddings import (
    EmbeddingCollector,
    collect_embedding_shard,
    format_embedding_prompt,
    mean_pool_last_hidden,
    validate_embedding_stage_union,
)


class FakeTokenizer:
    def __init__(self, *, include_assistant=True):
        self.last_messages = None
        self.last_kwargs = None
        self.include_assistant = include_assistant

    def apply_chat_template(self, messages, **kwargs):
        self.last_messages = messages
        self.last_kwargs = kwargs
        ids = [10, 11, 12, 99] if self.include_assistant else [10, 11, 12]
        return torch.tensor([ids], dtype=torch.long)

    def __call__(self, text, *, add_special_tokens):
        assert text == "<|im_start|>assistant"
        assert add_special_tokens is False
        return {"input_ids": [99]}


class FakeEmbeddingModel(torch.nn.Module):
    def __init__(self, *, nonfinite=False, zero=False):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.ones(1))
        self.calls = []
        self.teacher_calls = 0
        self.nonfinite = nonfinite
        self.zero = zero

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        length = kwargs["input_ids"].shape[1]
        hidden = torch.arange(length * 1024, dtype=torch.float32).reshape(
            1, length, 1024
        )
        if self.zero:
            hidden.zero_()
        if self.nonfinite:
            hidden[0, 0, 0] = float("nan")
        return type("Output", (), {"hidden_states": (hidden,)})()

    def score_teacher(self, *args, **kwargs):
        self.teacher_calls += 1
        raise AssertionError("teacher invoked")


def _row(index: int):
    return {
        "stable_id": f"q{index}",
        "split": "candidate",
        "manifest_index": index,
        "prompt": f"Question {index}",
        "raw_opd_messages": [
            {"role": "user", "content": f"Question {index} [OPD suffix]"}
        ],
        "completion": f"must not be embedded {index}",
        "prompt_token_count_0_6b": 4,
    }


def _collector(*, tokenizer=None, model=None):
    return EmbeddingCollector(
        model=FakeEmbeddingModel() if model is None else model,
        tokenizer=FakeTokenizer() if tokenizer is None else tokenizer,
        model_manifest_sha256="1" * 64,
        tokenizer_manifest_sha256="2" * 64,
        chat_template_sha256="3" * 64,
    )


def _collect_kwargs(tmp_path, collector, rows, *, output="vectors", stage=1, stage1=None):
    return {
        "rows": rows,
        "expected_candidate_ids": tuple(row["stable_id"] for row in rows),
        "collector": collector,
        "output_directory": tmp_path / output,
        "stage": stage,
        "stage1_vector_directory": stage1,
        "parent_hashes": {
            "sample_manifest_sha256": "4" * 64,
            "model_manifest_sha256": "1" * 64,
            "tokenizer_manifest_sha256": "2" * 64,
        },
        "source_snapshot": {"manifest_sha256": "5" * 64},
        "repository": {
            "head": "6" * 40,
            "status": "",
            "status_sha256": "7" * 64,
        },
        "runtime": {"runtime_profile": "gvendi_analysis"},
        "chunk_size": 1,
    }


def test_embedding_prompt_uses_generation_prefix_and_disables_thinking():
    tokenizer = FakeTokenizer()
    ids = format_embedding_prompt(tokenizer, "raw question")
    assert tokenizer.last_messages == [{"role": "user", "content": "raw question"}]
    assert tokenizer.last_kwargs == {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": False,
        "return_tensors": "pt",
    }
    assert ids.tolist() == [[10, 11, 12, 99]]


def test_embedding_prompt_rejects_missing_assistant_prefix():
    with pytest.raises(ValueError, match="assistant-generation prefix"):
        format_embedding_prompt(
            FakeTokenizer(include_assistant=False), "raw question"
        )


def test_mean_pool_ignores_padding_and_l2_normalizes():
    hidden = torch.tensor([[[3.0, 0.0], [0.0, 4.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    pooled = mean_pool_last_hidden(hidden, mask)
    expected = torch.tensor([[1.5, 2.0]])
    expected = expected / torch.linalg.vector_norm(expected, dim=1, keepdim=True)
    torch.testing.assert_close(pooled, expected)


@pytest.mark.parametrize(
    "hidden",
    [
        torch.zeros((1, 2, 3)),
        torch.tensor([[[float("nan"), 1.0], [1.0, 1.0]]]),
    ],
)
def test_mean_pool_rejects_zero_or_nonfinite_embeddings(hidden):
    with pytest.raises(ValueError, match="finite|zero"):
        mean_pool_last_hidden(hidden, torch.ones(hidden.shape[:2], dtype=torch.long))


def test_embedding_collector_uses_only_raw_prompt_and_no_backward(monkeypatch):
    model = FakeEmbeddingModel()
    tokenizer = FakeTokenizer()
    collector = _collector(model=model, tokenizer=tokenizer)
    monkeypatch.setattr(
        torch.Tensor,
        "backward",
        lambda *args, **kwargs: pytest.fail("backward invoked"),
    )

    record = collector.collect_one(_row(0))

    assert record.vector_id == "E:q0"
    assert tokenizer.last_messages == [
        {"role": "user", "content": "Question 0 [OPD suffix]"}
    ]
    assert model.training is False
    assert model.teacher_calls == 0
    assert len(model.calls) == 1
    assert model.calls[0]["output_hidden_states"] is True
    assert model.calls[0]["use_cache"] is False
    assert all(parameter.grad is None for parameter in model.parameters())
    torch.testing.assert_close(record.vector.projected_gradient.norm(), torch.tensor(1.0))


@pytest.mark.parametrize("kind", ["nonfinite", "zero"])
def test_embedding_collector_rejects_invalid_model_output(kind):
    model = FakeEmbeddingModel(
        nonfinite=kind == "nonfinite", zero=kind == "zero"
    )
    with pytest.raises(ValueError, match="finite|zero"):
        _collector(model=model).collect_one(_row(0))


def test_embedding_collection_requires_exact_candidate_order(tmp_path: Path):
    rows = [_row(0), _row(1)]
    kwargs = _collect_kwargs(tmp_path, _collector(), rows)
    kwargs["expected_candidate_ids"] = ("q1", "q0")
    with pytest.raises(ValueError, match="candidate order"):
        collect_embedding_shard(**kwargs)


def test_embedding_stage2_is_append_only_and_union_is_exact(tmp_path: Path):
    first = [_row(0), _row(1)]
    stage1_collector = _collector()
    stage1 = collect_embedding_shard(
        **_collect_kwargs(tmp_path, stage1_collector, first, output="stage1")
    )
    assert stage1.vector_ids == ("E:q0", "E:q1")
    sidecar = json.loads(
        (tmp_path / "stage1/vectors_0_1.jsonl").read_text(encoding="utf-8")
    )
    assert sidecar["prompt_token_count"] == 4

    all_rows = first + [_row(2), _row(3)]
    stage2_collector = _collector()
    stage2 = collect_embedding_shard(
        **_collect_kwargs(
            tmp_path,
            stage2_collector,
            all_rows,
            output="stage2",
            stage=2,
            stage1=tmp_path / "stage1",
        )
    )
    assert stage2.vector_ids == ("E:q2", "E:q3")
    assert [call["input_ids"][0, 1].item() for call in stage2_collector.model.calls] == [
        11,
        11,
    ]
    assert validate_embedding_stage_union(
        tmp_path / "stage1",
        tmp_path / "stage2",
        expected_candidate_ids=("q0", "q1", "q2", "q3"),
    ) == ("E:q0", "E:q1", "E:q2", "E:q3")


def test_embedding_collection_resumes_atomically(tmp_path: Path):
    rows = [_row(0), _row(1), _row(2)]

    class InterruptingCollector:
        def __init__(self):
            self.base = _collector()
            self.calls = []
            self.model_manifest_sha256 = self.base.model_manifest_sha256
            self.tokenizer_manifest_sha256 = self.base.tokenizer_manifest_sha256
            self.chat_template_sha256 = self.base.chat_template_sha256

        def collect_one(self, row):
            self.calls.append(row["stable_id"])
            if row["stable_id"] == "q1":
                raise RuntimeError("synthetic interruption")
            return self.base.collect_one(row)

    interrupted = InterruptingCollector()
    with pytest.raises(RuntimeError, match="interruption"):
        collect_embedding_shard(**_collect_kwargs(tmp_path, interrupted, rows))
    assert interrupted.calls == ["q0", "q1"]

    resumed = _collector()
    loaded = collect_embedding_shard(**_collect_kwargs(tmp_path, resumed, rows))
    assert loaded.vector_ids == ("E:q0", "E:q1", "E:q2")
    assert len(resumed.model.calls) == 2
