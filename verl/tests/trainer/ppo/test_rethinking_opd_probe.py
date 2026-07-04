from pathlib import Path

import pytest
import torch

from verl.trainer.ppo.rethinking_opd_probe import (
    aggregate_rethinking_opd_probe_metrics,
    append_rethinking_opd_probe_csv,
    compute_student_topk_from_logits,
    compute_teacher_topk_overlap,
    decorate_rethinking_probe_rows,
    has_rethinking_opd_probe_tensors,
)


def test_compute_student_topk_from_logits_returns_log_probs():
    logits = torch.tensor([[[5.0, 4.0, 1.0], [0.0, 3.0, 2.0]]])

    ids, log_probs = compute_student_topk_from_logits(logits, top_k=2)

    assert ids.tolist() == [[[0, 1], [1, 2]]]
    expected = torch.log_softmax(logits, dim=-1).gather(-1, ids)
    assert torch.allclose(log_probs, expected)


def test_compute_teacher_topk_overlap_marks_student_ids_in_teacher_topk():
    logits = torch.tensor(
        [
            [5.0, 4.0, 1.0, 0.0],
            [0.0, 1.0, 4.0, 5.0],
        ]
    )
    student_ids = torch.tensor([[0, 2], [3, 1]])

    out = compute_teacher_topk_overlap(logits, student_ids, top_k=2)

    assert out["teacher_top_k_ids"].tolist() == [[0, 1], [3, 2]]
    assert out["overlap_mask"].tolist() == [[1.0, 0.0], [1.0, 0.0]]
    assert out["teacher_in_student_mask"].tolist() == [[1.0, 0.0], [1.0, 0.0]]
    assert out["teacher_on_student_log_probs"].shape == (2, 2)
    assert out["teacher_top_k_log_probs"].shape == (2, 2)


def test_compute_teacher_topk_overlap_rejects_shape_mismatch():
    logits = torch.zeros(2, 5)
    student_ids = torch.zeros(2, 3, dtype=torch.long)

    with pytest.raises(ValueError, match="last dimension must equal top_k"):
        compute_teacher_topk_overlap(logits, student_ids, top_k=2)


def test_has_rethinking_opd_probe_tensors_detects_required_keys():
    required = {
        "student_top_k_log_probs": torch.zeros(1, 1, 1),
        "teacher_on_student_log_probs": torch.zeros(1, 1, 1),
        "overlap_mask": torch.zeros(1, 1, 1),
    }
    missing_student = {
        "teacher_on_student_log_probs": torch.zeros(1, 1, 1),
        "overlap_mask": torch.zeros(1, 1, 1),
    }

    assert has_rethinking_opd_probe_tensors(required)
    assert not has_rethinking_opd_probe_tensors(missing_student)


def test_aggregate_rethinking_opd_probe_metrics_global_and_chunks():
    student_log_probs = torch.log(torch.tensor([[[0.50, 0.25], [0.60, 0.20], [0.70, 0.10], [0.80, 0.05]]]))
    teacher_on_student = torch.log(torch.tensor([[[0.40, 0.30], [0.50, 0.10], [0.60, 0.20], [0.70, 0.10]]]))
    overlap_mask = torch.tensor([[[1.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]])
    response_mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    student_entropy = torch.tensor([[0.10, 0.20, 0.30, 0.40]])
    teacher_entropy = torch.tensor([[0.20, 0.25, 0.35, 0.50]])

    rows = aggregate_rethinking_opd_probe_metrics(
        {
            "student_top_k_log_probs": student_log_probs,
            "teacher_on_student_log_probs": teacher_on_student,
            "overlap_mask": overlap_mask,
            "student_entropys": student_entropy,
            "ref_entropys": teacher_entropy,
        },
        response_mask,
        top_k=2,
        chunk_size=2,
    )

    global_row = next(row for row in rows if row["chunk_start"] == -1)
    chunk0 = next(row for row in rows if row["chunk_start"] == 0)
    chunk1 = next(row for row in rows if row["chunk_start"] == 2)

    assert global_row["valid_token_count"] == 3
    assert global_row["valid_sequence_count"] == 1
    assert global_row["topk_overlap_ratio"] == pytest.approx(5 / 6)
    assert global_row["student_overlap_mass"] == pytest.approx((0.75 + 0.60 + 0.80) / 3)
    assert global_row["teacher_overlap_mass"] == pytest.approx((0.70 + 0.50 + 0.80) / 3)
    assert global_row["entropy_gap"] == pytest.approx(((0.10) + (0.05) + (0.05)) / 3)
    assert "overlap_token_advantage" in global_row

    assert chunk0["valid_token_count"] == 2
    assert chunk0["topk_overlap_ratio"] == pytest.approx(3 / 4)
    assert chunk1["valid_token_count"] == 1
    assert chunk1["topk_overlap_ratio"] == pytest.approx(1.0)


def test_aggregate_rethinking_opd_probe_metrics_no_overlap_advantage_nan():
    student_log_probs = torch.log(torch.tensor([[[0.50, 0.50], [0.20, 0.80], [0.60, 0.40]]])
    )
    teacher_on_student = torch.log(
        torch.tensor([[[0.70, 0.30], [0.90, 0.10], [0.55, 0.45]]])
    )
    overlap_mask = torch.zeros((1, 3, 2))
    response_mask = torch.tensor([[1.0, 1.0, 1.0]])

    rows = aggregate_rethinking_opd_probe_metrics(
        {
            "student_top_k_log_probs": student_log_probs,
            "teacher_on_student_log_probs": teacher_on_student,
            "overlap_mask": overlap_mask,
            "student_entropys": torch.tensor([[0.1, 0.2, 0.3]]),
            "ref_entropys": torch.tensor([[0.2, 0.3, 0.4]]),
        },
        response_mask,
        top_k=2,
        chunk_size=2,
    )

    global_row = next(row for row in rows if row["chunk_start"] == -1)
    chunk0 = next(row for row in rows if row["chunk_start"] == 0)
    chunk1 = next(row for row in rows if row["chunk_start"] == 2)

    assert global_row["valid_token_count"] == 3
    assert chunk0["valid_token_count"] == 2
    assert chunk1["valid_token_count"] == 1
    assert all(row["topk_overlap_ratio"] == 0.0 for row in rows)
    assert all(torch.isnan(torch.tensor(row["overlap_token_advantage"])) for row in rows)


def test_decorate_rethinking_probe_rows_adds_context_fields():
    rows = [{"chunk_start": -1, "chunk_end": -1, "valid_token_count": 2, "topk_overlap_ratio": 0.5}]
    response_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])

    out = decorate_rethinking_probe_rows(
        rows,
        run_name="probe-run",
        step=7,
        top_k=16,
        response_mask=response_mask,
    )

    assert out[0]["run_name"] == "probe-run"
    assert out[0]["step"] == 7
    assert out[0]["top_k"] == 16
    assert out[0]["batch_size"] == 2
    assert out[0]["mean_all_batch_response_length"] == pytest.approx(1.5)
    assert out[0]["max_all_batch_response_length"] == pytest.approx(2.0)
    assert out[0]["clip_rate_all_batch"] == pytest.approx(0.0)


def test_log_rethinking_opd_probe_metrics_skips_when_required_tensors_missing():
    from types import SimpleNamespace

    from verl.trainer.ppo import ray_trainer

    batch = SimpleNamespace(
        batch={
            "teacher_on_student_log_probs": torch.zeros((1, 2, 2)),
            "overlap_mask": torch.ones((1, 2, 2)),
        }
    )
    metrics = {}
    config = SimpleNamespace(
        algorithm={
            "rethinking_opd_probe": {
                "enabled": True,
                "include_scalar_logger": True,
                "log_prefix": "rethinking_opd",
            }
        },
        trainer={"experiment_name": "probe-run"},
    )

    ray_trainer._log_rethinking_opd_probe_metrics(batch, config=config, global_step=3, metrics=metrics)

    assert metrics == {"rethinking_opd/skipped_missing_tensors": 1.0}


def test_append_rethinking_opd_probe_csv_writes_header_once(tmp_path: Path):
    path = tmp_path / "probe.csv"
    rows = [
        {
            "run_name": "unit",
            "step": 1,
            "chunk_start": -1,
            "chunk_end": -1,
            "top_k": 2,
            "batch_size": 1,
            "valid_sequence_count": 1,
            "valid_token_count": 3,
            "mean_all_batch_response_length": 3.0,
            "max_all_batch_response_length": 4.0,
            "clip_rate_all_batch": 0.0,
            "topk_overlap_ratio": 0.5,
            "student_overlap_mass": 0.7,
            "teacher_overlap_mass": 0.6,
            "student_entropy": 0.2,
            "teacher_entropy": 0.3,
            "entropy_gap": 0.1,
            "overlap_token_advantage": -0.01,
        }
    ]

    append_rethinking_opd_probe_csv(path, rows)
    append_rethinking_opd_probe_csv(path, rows)

    lines = path.read_text().strip().splitlines()
    assert lines[0].startswith("run_name,step,chunk_start")
    assert len(lines) == 3
