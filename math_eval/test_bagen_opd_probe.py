import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from math_eval.bagen_opd_probe import (
    build_probe_messages,
    compute_bagen_style_metrics,
    flatten_eval_outputs,
    parse_probe_answer,
)


class DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(token_ids)


def test_flatten_eval_outputs_builds_one_probe_per_prefix_and_response(tmp_path: Path):
    input_path = tmp_path / "eval.jsonl"
    record = {
        "problem": "Compute 1+1.",
        "answer": "2",
        "responses": ["a b c d e f g h i j", "x y z w"],
        "response_lengths": [10, 4],
        "acc_list": [True, False],
    }
    input_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    examples = flatten_eval_outputs(
        input_path=input_path,
        tokenizer=DummyTokenizer(),
        prefix_fractions=[0.1, 0.2, 0.4],
        max_rollouts_per_problem=2,
    )

    assert len(examples) == 6
    assert examples[0].problem_index == 0
    assert examples[0].sample_index == 0
    assert examples[0].prefix_fraction == pytest.approx(0.1)
    assert examples[0].student_budget_tokens == 10
    assert examples[0].prefix_tokens == 1
    assert examples[0].remaining_budget_tokens == 9
    assert examples[0].student_correct is True
    assert examples[0].gold_label == "possible"
    assert examples[0].prefix_text == "a"

    wrong_40 = [ex for ex in examples if ex.sample_index == 1 and math.isclose(ex.prefix_fraction, 0.4)][0]
    assert wrong_40.student_budget_tokens == 4
    assert wrong_40.prefix_tokens == 2
    assert wrong_40.remaining_budget_tokens == 2
    assert wrong_40.student_correct is False
    assert wrong_40.gold_label == "impossible"
    assert wrong_40.prefix_text == "x y"


def test_build_probe_messages_contains_budget_prefix_and_strict_output_format():
    messages = build_probe_messages(
        problem="Find x.",
        prefix_text="I start by setting x=1.",
        student_budget_tokens=100,
        prefix_tokens=20,
        remaining_budget_tokens=80,
        prefix_fraction=0.2,
    )

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    user = messages[1]["content"]
    assert "Student total rollout budget: 100 tokens" in user
    assert "Student has used about 20 tokens (20%)" in user
    assert "Remaining student budget: 80 tokens" in user
    assert "<answer>possible</answer>" in user
    assert "<answer>impossible</answer>" in user
    assert "Do not solve the problem" in user


def test_parse_probe_answer_requires_possible_or_impossible_answer_tag():
    assert parse_probe_answer("<think>x</think><answer>possible</answer>") == "possible"
    assert parse_probe_answer("noise <answer> impossible </answer>") == "impossible"
    assert parse_probe_answer("I think impossible") is None
    assert parse_probe_answer("<answer>maybe</answer>") is None


def test_compute_bagen_style_metrics_reports_fail_f1_mcc_and_token_saving():
    rows = [
        {"gold_label": "possible", "prediction": "possible", "student_budget_tokens": 100, "prefix_fraction": 0.2},
        {"gold_label": "possible", "prediction": "impossible", "student_budget_tokens": 200, "prefix_fraction": 0.2},
        {"gold_label": "impossible", "prediction": "impossible", "student_budget_tokens": 300, "prefix_fraction": 0.2},
        {"gold_label": "impossible", "prediction": "possible", "student_budget_tokens": 400, "prefix_fraction": 0.2},
    ]

    metrics = compute_bagen_style_metrics(rows)

    assert metrics["n"] == 4
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["possible_f1"] == pytest.approx(0.5)
    assert metrics["fail_f1"] == pytest.approx(0.5)
    assert metrics["macro_f1"] == pytest.approx(0.5)
    assert metrics["precision_impossible"] == pytest.approx(0.5)
    assert metrics["recall_impossible"] == pytest.approx(0.5)
    assert metrics["false_abort_rate"] == pytest.approx(0.5)
    assert metrics["false_continue_rate"] == pytest.approx(0.5)
    assert metrics["mcc"] == pytest.approx(0.0)
    # Wrong-token saving: only the 300-token failed rollout is stopped after 20%.
    assert metrics["wrong_token_saving_rate"] == pytest.approx((0.8 * 300) / (300 + 400))
    # Correct-token false abort: only the 200-token correct rollout is falsely stopped.
    assert metrics["correct_token_false_abort_rate"] == pytest.approx((0.8 * 200) / (100 + 200))
