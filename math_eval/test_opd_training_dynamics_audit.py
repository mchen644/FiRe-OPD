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
