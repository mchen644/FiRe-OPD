from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from math_eval.collect_opd_proxy_sft_gradients import (
    build_sft_example,
    collect_sft_shard,
    construct_official_sft_collector,
    validate_sft_stage_union,
)
from math_eval.opd_proxy_gradient_projection import (
    PRISMATIC_REFERENCE_COMMIT,
    PRISMATIC_REFERENCE_TREE,
    ReferenceSnapshot,
)


def _reference(tmp_path: Path, *, commit=PRISMATIC_REFERENCE_COMMIT):
    source = tmp_path / "gradient_computer.py"
    source.write_text("official\n", encoding="utf-8")
    return ReferenceSnapshot(
        repository=str(tmp_path.resolve()),
        commit=commit,
        tree=PRISMATIC_REFERENCE_TREE,
        official_gradient_module=str(source.resolve()),
        official_gradient_module_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )


def _row(index: int) -> dict[str, object]:
    return {
        "stable_id": f"q{index}",
        "split": "candidate",
        "manifest_index": index,
        "prompt": f"Question {index}",
        "completion": f"Solution {index}",
        "sft_full_token_count_0_6b": 6,
        "sft_supervised_label_count": 3,
        "r1_completion_token_count_0_6b": 2,
        "prompt_token_count_0_6b": 2,
    }


class FakeGradientComputer:
    def __init__(self, *, labels=None, fail_on_id: str | None = None):
        self.calls = []
        self.collected_ids = []
        self.labels = (
            torch.tensor([[-100, -100, -100, 11, 12, 13]])
            if labels is None
            else labels
        )
        self.fail_on_id = fail_on_id
        self.current_id = None
        self.opd_projection_manifest = {
            "backend": "CudaProjector",
            "constructor": {
                "grad_dim": 2,
                "proj_dim": 1024,
                "seed": 0,
                "proj_type": "rademacher",
                "device": "cuda:0",
                "dtype": "float16",
                "block_size": 128,
                "max_batch_size": 16,
            },
            "model_id": 0,
            "native_gradient_dtype": "float32",
            "projector_input_dtype": "float16",
            "stored_dtype": "float32",
            "output_scale": "1/sqrt(1024)",
            "parameter_layout_sha256": "b" * 64,
        }

    def prepare_model_input(self, prompt, completion):
        self.calls.append("prepare_model_input")
        self.current_id = prompt.split()[-1]
        return {
            "input_ids": torch.tensor([[1, 2, 3, 11, 12, 13]]),
            "attention_mask": torch.ones((1, 6), dtype=torch.long),
            "labels": self.labels.clone(),
        }

    def obtain_gradient(self, encoding):
        self.calls.append("obtain_gradient")
        stable_id = f"q{self.current_id}"
        self.collected_ids.append(stable_id)
        if stable_id == self.fail_on_id:
            raise RuntimeError("synthetic interruption")
        value = float(int(self.current_id) + 1)
        return torch.tensor([value, value + 1.0], dtype=torch.float32)

    def project_gradients(self, gradients):
        self.calls.append("project_gradients")
        stable_id, gradient = next(iter(gradients.items()))
        return {stable_id: gradient.sum().repeat(1024).to(torch.float32) / 32.0}


def _collect_kwargs(tmp_path: Path, collector, rows, *, output="vectors", stage=1, stage1=None):
    return {
        "rows": rows,
        "expected_candidate_ids": tuple(row["stable_id"] for row in rows),
        "collector": collector,
        "output_directory": tmp_path / output,
        "stage": stage,
        "stage1_vector_directory": stage1,
        "parent_hashes": {"sample_manifest_sha256": "c" * 64},
        "source_snapshot": {"manifest_sha256": "d" * 64},
        "repository": {
            "head": "e" * 40,
            "status": "",
            "status_sha256": "f" * 64,
        },
        "runtime": {"runtime_profile": "gvendi_analysis"},
        "metadata": {
            "model_name": "Qwen/Qwen3-0.6B",
            "model_manifest_sha256": "1" * 64,
            "tokenizer_manifest_sha256": "2" * 64,
        },
        "chunk_size": 1,
    }


def test_sft_collector_calls_official_completion_only_methods(tmp_path: Path):
    fake = FakeGradientComputer()
    collector = construct_official_sft_collector(fake, _reference(tmp_path))

    record = collector.collect_one(_row(0))

    assert fake.calls == [
        "prepare_model_input",
        "obtain_gradient",
        "project_gradients",
    ]
    assert record.vector_id == "S:q0"
    assert record.supervised_label_count == 3
    assert record.full_gradient_norm.item() > 0
    assert record.projected_gradient.dtype == torch.float32


def test_sft_example_uses_original_question_and_r1_solution():
    row = _row(0)
    example = build_sft_example(row)
    assert example["messages"] == [
        {"role": "user", "content": row["prompt"]},
        {"role": "assistant", "content": row["completion"]},
    ]


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        (torch.tensor([[1, -100, -100, 11, 12, 13]]), "prompt-supervised"),
        (torch.full((1, 6), -100), "zero completion labels"),
        (torch.tensor([[-100, -100, -100, 11, -100, 13]]), "contiguous suffix"),
    ],
)
def test_sft_collector_rejects_invalid_completion_only_labels(
    tmp_path: Path, labels: torch.Tensor, message: str
):
    collector = construct_official_sft_collector(
        FakeGradientComputer(labels=labels), _reference(tmp_path)
    )
    with pytest.raises(ValueError, match=message):
        collector.collect_one(_row(0))


def test_sft_collector_rejects_changed_reference_commit_or_tree(tmp_path: Path):
    fake = FakeGradientComputer()
    with pytest.raises(ValueError, match="commit"):
        construct_official_sft_collector(
            fake, _reference(tmp_path, commit="0" * 40)
        )
    wrong_tree = replace(_reference(tmp_path), tree="0" * 40)
    with pytest.raises(ValueError, match="tree"):
        construct_official_sft_collector(fake, wrong_tree)


def test_collect_sft_shard_requires_exact_candidate_order(tmp_path: Path):
    rows = [_row(0), _row(1)]
    collector = construct_official_sft_collector(
        FakeGradientComputer(), _reference(tmp_path)
    )
    kwargs = _collect_kwargs(tmp_path, collector, rows)
    kwargs["expected_candidate_ids"] = ("q1", "q0")
    with pytest.raises(ValueError, match="candidate order"):
        collect_sft_shard(**kwargs)


def test_stage2_collects_only_appended_rows_and_validates_union(tmp_path: Path):
    stage1_rows = [_row(0), _row(1)]
    stage1_fake = FakeGradientComputer()
    stage1_collector = construct_official_sft_collector(
        stage1_fake, _reference(tmp_path)
    )
    stage1 = collect_sft_shard(
        **_collect_kwargs(tmp_path, stage1_collector, stage1_rows, output="stage1")
    )
    assert stage1.vector_ids == ("S:q0", "S:q1")
    first_sidecar = json.loads(
        (tmp_path / "stage1/vectors_0_1.jsonl").read_text(encoding="utf-8")
    )
    assert first_sidecar["prompt_token_count"] == 2
    assert first_sidecar["completion_token_count"] == 2
    assert first_sidecar["supervised_label_count"] == 3
    assert first_sidecar["full_token_count"] == 6

    all_rows = stage1_rows + [_row(2), _row(3)]
    stage2_fake = FakeGradientComputer()
    stage2_collector = construct_official_sft_collector(
        stage2_fake, _reference(tmp_path)
    )
    stage2 = collect_sft_shard(
        **_collect_kwargs(
            tmp_path,
            stage2_collector,
            all_rows,
            output="stage2",
            stage=2,
            stage1=tmp_path / "stage1",
        )
    )

    assert stage2.vector_ids == ("S:q2", "S:q3")
    assert stage2_fake.collected_ids == ["q2", "q3"]
    union = validate_sft_stage_union(
        tmp_path / "stage1",
        tmp_path / "stage2",
        expected_candidate_ids=("q0", "q1", "q2", "q3"),
    )
    assert union == ("S:q0", "S:q1", "S:q2", "S:q3")


def test_sft_collection_resumes_without_recomputing_completed_prefix(tmp_path: Path):
    rows = [_row(0), _row(1), _row(2)]
    interrupted_fake = FakeGradientComputer(fail_on_id="q1")
    interrupted = construct_official_sft_collector(
        interrupted_fake, _reference(tmp_path)
    )
    with pytest.raises(RuntimeError, match="interruption"):
        collect_sft_shard(**_collect_kwargs(tmp_path, interrupted, rows))
    assert interrupted_fake.collected_ids == ["q0", "q1"]

    resumed_fake = FakeGradientComputer()
    resumed = construct_official_sft_collector(resumed_fake, _reference(tmp_path))
    loaded = collect_sft_shard(**_collect_kwargs(tmp_path, resumed, rows))

    assert loaded.vector_ids == ("S:q0", "S:q1", "S:q2")
    assert resumed_fake.collected_ids == ["q1", "q2"]


def test_sft_collection_rejects_noncontiguous_resume_prefix(tmp_path: Path):
    rows = [_row(0), _row(1)]
    interrupted = construct_official_sft_collector(
        FakeGradientComputer(fail_on_id="q1"), _reference(tmp_path)
    )
    with pytest.raises(RuntimeError, match="interruption"):
        collect_sft_shard(**_collect_kwargs(tmp_path, interrupted, rows))
    output = tmp_path / "vectors"
    (output / "vectors_0_1.safetensors").rename(
        output / "vectors_2_3.safetensors"
    )
    (output / "vectors_0_1.jsonl").rename(output / "vectors_2_3.jsonl")
    resumed = construct_official_sft_collector(
        FakeGradientComputer(), _reference(tmp_path)
    )
    with pytest.raises(ValueError, match="contiguous"):
        collect_sft_shard(**_collect_kwargs(tmp_path, resumed, rows))


def test_sft_collection_rejects_existing_qwen25_vector_artifact(tmp_path: Path):
    output = tmp_path / "vectors"
    output.mkdir()
    (output / "deepmath-gradient.0.safetensors").write_bytes(b"old qwen2.5")
    collector = construct_official_sft_collector(
        FakeGradientComputer(), _reference(tmp_path)
    )
    with pytest.raises(ValueError, match="Qwen2.5|legacy"):
        collect_sft_shard(**_collect_kwargs(tmp_path, collector, [_row(0)]))
