import json
from pathlib import Path

import pytest
import torch

from math_eval.opd_training_dynamics_audit import (
    parse_metric_line,
    parse_wandb_output_log,
    normalize_record,
    default_run_specs,
)


def test_parse_metric_line_prefers_training_global_step():
    line = (
        "step:7 - actor/entropy:0.25 - training/global_step:19 - "
        "response_length/mean:8218.8 - response_length/max:16384.0 - "
        "critic/score/mean:0.711 - actor/grad_norm:2.72"
    )
    record = parse_metric_line(line)
    assert record["step"] == 19
    assert record["wandb_step"] == 7
    assert record["response_length/mean"] == 8218.8
    assert record["critic/score/mean"] == 0.711
    assert record["actor/grad_norm"] == 2.72


def test_parse_metric_line_falls_back_to_wandb_step():
    line = "step:3 - actor/entropy:0.33 - response_length/mean:1500.0"
    record = parse_metric_line(line)
    assert record["step"] == 3
    assert record["wandb_step"] == 3
    assert record["actor/entropy"] == 0.33


def test_normalize_record_distinguishes_hardtrunc_original_rollout_length():
    record = normalize_record(
        {
            "run_name": "normal hardtrunc",
            "step": 50,
            "response_length/mean": 713.5,
            "tale_budget/response_length_mean": 3567.6,
        }
    )
    assert record["supervised_length_mean"] == 713.5
    assert record["original_rollout_length_mean"] == 3567.6
    assert record["has_tale_original_length"] == 1.0


def test_normalize_record_uses_response_length_for_raw_opd():
    record = normalize_record(
        {
            "run_name": "raw OPD",
            "step": 19,
            "response_length/mean": 8218.8,
        }
    )
    assert record["supervised_length_mean"] == 8218.8
    assert record["original_rollout_length_mean"] == 8218.8
    assert record["has_tale_original_length"] == 0.0


def test_parse_wandb_output_log_filters_non_metric_lines():
    text = "hello\nstep:1 - training/global_step:1 - response_length/mean:10.0\nbye\n"
    rows = parse_wandb_output_log(text, run_name="r", source_path="fake.log")
    assert len(rows) == 1
    assert rows[0]["run_name"] == "r"
    assert rows[0]["source_path"] == "fake.log"
    assert rows[0]["step"] == 1
    assert rows[0]["original_rollout_length_mean"] == 10.0


def test_default_run_specs_include_required_runs():
    names = {spec.name for spec in default_run_specs()}
    assert "raw_opd" in names
    assert "hardtrunc_budget" in names
    assert "hardtrunc_concise" in names
    assert "hardtrunc_normal" in names


from math_eval.opd_training_dynamics_audit import write_csv, render_macro_svg, summarize_run_lengths


def test_write_csv_includes_normalized_fields(tmp_path):
    rows = [
        {"run_name": "r", "step": 1.0, "supervised_length_mean": 10.0, "original_rollout_length_mean": 50.0},
        {"run_name": "r", "step": 2.0, "supervised_length_mean": 20.0, "original_rollout_length_mean": 60.0},
    ]
    out = tmp_path / "metrics.csv"
    write_csv(rows, out)
    text = out.read_text()
    assert "run_name,step" in text
    assert "original_rollout_length_mean" in text
    assert "60.0" in text


def test_render_macro_svg_writes_series_labels(tmp_path):
    rows = [
        {"run_name": "raw_opd", "step": 1.0, "original_rollout_length_mean": 1500.0, "supervised_length_mean": 1500.0, "critic/score/mean": 0.5, "actor/entropy": 0.3, "actor/grad_norm": 4.0},
        {"run_name": "raw_opd", "step": 2.0, "original_rollout_length_mean": 2500.0, "supervised_length_mean": 2500.0, "critic/score/mean": 0.6, "actor/entropy": 0.2, "actor/grad_norm": 3.0},
        {"run_name": "hardtrunc_concise", "step": 1.0, "original_rollout_length_mean": 1500.0, "supervised_length_mean": 300.0, "critic/score/mean": 0.5, "actor/entropy": 0.3, "actor/grad_norm": 12.0},
    ]
    out = tmp_path / "macro.svg"
    render_macro_svg(rows, out)
    svg = out.read_text()
    assert "AIME-independent training dynamics" in svg
    assert "raw_opd" in svg
    assert "hardtrunc_concise" in svg
    assert "Original rollout length" in svg


def test_summarize_run_lengths_reports_peak_and_final():
    rows = [
        {"run_name": "raw_opd", "step": 1.0, "original_rollout_length_mean": 1500.0},
        {"run_name": "raw_opd", "step": 19.0, "original_rollout_length_mean": 8218.8},
        {"run_name": "raw_opd", "step": 69.0, "original_rollout_length_mean": 4895.8},
    ]
    summary = summarize_run_lengths(rows)
    assert summary["raw_opd"]["peak_step"] == 19.0
    assert summary["raw_opd"]["peak_original_rollout_length"] == 8218.8
    assert summary["raw_opd"]["final_original_rollout_length"] == 4895.8


def test_summarize_run_lengths_ignores_nan_lengths():
    rows = [
        {"run_name": "raw_opd", "step": 0.0, "original_rollout_length_mean": float("nan")},
        {"run_name": "raw_opd", "step": 19.0, "original_rollout_length_mean": 8218.8},
        {"run_name": "raw_opd", "step": 69.0, "original_rollout_length_mean": 4895.8},
    ]
    summary = summarize_run_lengths(rows)
    assert summary["raw_opd"]["peak_step"] == 19.0
    assert summary["raw_opd"]["final_step"] == 69.0


from math_eval.opd_training_dynamics_audit import (
    compression_repetition_ratio,
    ngram_repetition_rate,
    iter_eval_responses,
    compute_repetition_metrics,
)


def test_compression_repetition_ratio_higher_for_repetitive_text():
    repetitive = "wait wait wait wait wait wait wait wait " * 30
    diverse = "We solve by substituting x, simplifying a quadratic, and checking the boundary cases. " * 4
    assert compression_repetition_ratio(repetitive) > compression_repetition_ratio(diverse)


def test_ngram_repetition_rate_counts_repeated_ngrams():
    tokens = "a b c a b c a b c d e".split()
    assert ngram_repetition_rate(tokens, n=3) > 0.0
    assert ngram_repetition_rate(["a", "b"], n=3) == 0.0


def test_iter_eval_responses_reads_problem_level_schema(tmp_path):
    path = tmp_path / "eval.jsonl"
    path.write_text(
        json.dumps(
            {
                "problem": "p0",
                "responses": ["short", "long long long"],
                "response_lengths": [1, 3],
                "acc_list": [True, False],
            }
        )
        + "\n"
    )
    rows = list(iter_eval_responses(path, run_name="r"))
    assert len(rows) == 2
    assert rows[0]["run_name"] == "r"
    assert rows[0]["problem_index"] == 0
    assert rows[1]["sample_index"] == 1
    assert rows[1]["is_correct"] == 0.0


def test_compute_repetition_metrics_flags_truncated_length():
    rows = [
        {"run_name": "r", "response": "a b c", "response_length": 3.0, "is_correct": 1.0},
        {"run_name": "r", "response": "wait " * 100, "response_length": 16384.0, "is_correct": 0.0},
    ]
    metrics = compute_repetition_metrics(rows, max_token_length=16384)
    assert len(metrics) == 1
    assert metrics[0]["run_name"] == "r"
    assert metrics[0]["num_responses"] == 2.0
    assert metrics[0]["truncation_rate"] == 0.5
    assert metrics[0]["mean_compression_repetition_ratio"] > 0.0


from math_eval.opd_training_dynamics_audit import (
    compute_topk_alignment_metrics,
    compute_position_binned_alignment,
)


def test_compute_topk_alignment_metrics_identical_logits_have_full_overlap():
    logits = torch.tensor([[[4.0, 3.0, 1.0, 0.0], [0.0, 1.0, 3.0, 4.0]]])
    metrics = compute_topk_alignment_metrics(logits, logits.clone(), k=2)
    assert metrics["num_positions"] == 2.0
    assert metrics["topk_overlap_ratio"] == 1.0
    assert metrics["entropy_gap"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["overlap_student_mass"] > 0.8
    assert metrics["overlap_teacher_mass"] > 0.8


def test_compute_topk_alignment_metrics_disjoint_top1_has_zero_overlap():
    student = torch.tensor([[[10.0, 0.0, 0.0]]])
    teacher = torch.tensor([[[0.0, 10.0, 0.0]]])
    metrics = compute_topk_alignment_metrics(student, teacher, k=1)
    assert metrics["topk_overlap_ratio"] == 0.0
    assert metrics["overlap_student_mass"] == 0.0
    assert metrics["overlap_teacher_mass"] == 0.0


def test_compute_topk_alignment_metrics_respects_mask():
    student = torch.tensor([[[4.0, 3.0, 0.0], [10.0, 0.0, 0.0]]])
    teacher = torch.tensor([[[4.0, 3.0, 0.0], [0.0, 10.0, 0.0]]])
    mask = torch.tensor([[1, 0]])
    metrics = compute_topk_alignment_metrics(student, teacher, mask=mask, k=1)
    assert metrics["num_positions"] == 1.0
    assert metrics["topk_overlap_ratio"] == 1.0


def test_compute_position_binned_alignment_returns_requested_bins():
    logits = torch.randn(1, 8, 6)
    bins = compute_position_binned_alignment(logits, logits.clone(), k=2, num_bins=4)
    assert len(bins) == 4
    assert bins[0]["position_bin"] == 0.0
    assert bins[-1]["position_bin"] == 3.0
    assert all(row["topk_overlap_ratio"] == pytest.approx(1.0) for row in bins)


from math_eval.opd_training_dynamics_audit import build_teacher_messages_for_audit, score_response_logits


def test_build_teacher_messages_for_audit_normal_uses_raw_messages():
    raw = [{"role": "user", "content": "Problem text\nPlease reason step by step, and put your final answer within \\boxed{}."}]
    messages = build_teacher_messages_for_audit("Problem text", raw, style="normal", budget=None)
    assert messages == raw


def test_build_teacher_messages_for_audit_budget_inserts_budget():
    messages = build_teacher_messages_for_audit("Problem text", None, style="budget", budget=512)
    assert "less than 512 tokens" in messages[0]["content"]
    assert "Problem text" in messages[0]["content"]


def test_build_teacher_messages_for_audit_budget_strips_verbose_instruction():
    question = "Problem text\nPlease reason step by step, and put your final answer within \\boxed{}."
    messages = build_teacher_messages_for_audit(question, None, style="budget", budget=512)
    assert messages[0]["content"].count("Please reason step by step") == 0
    assert messages[0]["content"].startswith("Problem text\nLet's think step by step")


def test_build_teacher_messages_for_audit_concise_inserts_concise_instruction():
    messages = build_teacher_messages_for_audit("Problem text", None, style="concise", budget=None)
    assert "Solve concisely" in messages[0]["content"]


def test_build_teacher_messages_for_audit_rejects_unknown_style():
    with pytest.raises(ValueError, match="style"):
        build_teacher_messages_for_audit("Problem text", None, style="shortest", budget=None)


from math_eval.opd_training_dynamics_audit import render_alignment_svg, render_single_run_svg


def test_score_response_logits_supports_response_window():
    class Encoded:
        def __init__(self, ids):
            self.input_ids = torch.tensor([ids])

    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
            return "P0 P1"

        def __call__(self, text, return_tensors="pt", add_special_tokens=False):
            ids = list(range(len(text.split())))
            return Encoded(ids)

    class FakeOutput:
        def __init__(self, logits):
            self.logits = logits

    class FakeModel:
        device = torch.device("cpu")

        def __call__(self, input_ids):
            seq_len = input_ids.shape[1]
            logits = torch.arange(seq_len, dtype=torch.float32).view(1, seq_len, 1).expand(1, seq_len, 3)
            return FakeOutput(logits)

    logits = score_response_logits(
        FakeModel(),
        FakeTokenizer(),
        [{"role": "user", "content": "ignored"}],
        "R0 R1 R2 R3",
        max_score_tokens=2,
        window_start=1,
    )
    assert logits.shape == (1, 2, 3)
    # prompt length is 2. Response token 1 is predicted at absolute position 2.
    assert logits[0, :, 0].tolist() == [2.0, 3.0]


def test_render_alignment_svg_writes_prompt_styles(tmp_path):
    rows = [
        {"teacher_prompt_style": "normal", "topk_overlap_ratio": 0.8, "overlap_student_mass": 0.9, "student_entropy": 1.2, "teacher_entropy": 1.1, "entropy_gap": 0.1},
        {"teacher_prompt_style": "concise", "topk_overlap_ratio": 0.6, "overlap_student_mass": 0.95, "student_entropy": 0.5, "teacher_entropy": 0.4, "entropy_gap": 0.1},
    ]
    out = tmp_path / "alignment.svg"
    render_alignment_svg(rows, out)
    svg = out.read_text()
    assert "Teacher prompt style alignment metrics" in svg
    assert "normal" in svg
    assert "concise" in svg


def test_render_single_run_svg_filters_other_runs(tmp_path):
    rows = [
        {"run_name": "raw_opd", "step": 1.0, "original_rollout_length_mean": 1500.0, "supervised_length_mean": 1500.0, "critic/score/mean": 0.5, "actor/entropy": 0.3},
        {"run_name": "hardtrunc_normal", "step": 1.0, "original_rollout_length_mean": 1000.0, "supervised_length_mean": 200.0, "critic/score/mean": 0.4, "actor/entropy": 0.2},
    ]
    out = tmp_path / "single.svg"
    render_single_run_svg(rows, "raw_opd", out)
    svg = out.read_text()
    assert "raw_opd" in svg
    assert "hardtrunc_normal" not in svg
