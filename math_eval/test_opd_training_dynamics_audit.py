import json
from pathlib import Path

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
