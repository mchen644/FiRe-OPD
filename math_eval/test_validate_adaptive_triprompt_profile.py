import hashlib
import json
from pathlib import Path

import pytest

from math_eval.validate_adaptive_triprompt_profile import (
    main,
    validate_adaptive_triprompt_profile,
    write_report_atomic,
)

SOURCE_COMMIT = "a" * 40
EXPERIMENT = (
    "opd-adaptive-fullconciseprobe-triprompt-fullnormal-rawprompt-4gpu-tp4-profile-step1"
)
STUDENT_MODEL = "/home/mchen/FiRe-OPD/models/Qwen3-4B"
TEACHER_MODEL = "/home/mchen/FiRe-OPD/models/Qwen3-30B-A3B-Instruct-2507"


def _metrics() -> dict[str, float]:
    return {
        "training/global_step": 1.0,
        "adaptive_triprompt_opd/total_questions": 10.0,
        "adaptive_triprompt_opd/normal_correct_count": 6.0,
        "adaptive_triprompt_opd/normal_wrong_count": 4.0,
        "adaptive_triprompt_opd/concise_probe_count": 6.0,
        "adaptive_triprompt_opd/easy_count": 4.0,
        "adaptive_triprompt_opd/sensitive_count": 2.0,
        "adaptive_triprompt_opd/hard_count": 4.0,
        "adaptive_triprompt_opd/concise_teacher_count": 4.0,
        "adaptive_triprompt_opd/budget_teacher_count": 2.0,
        "adaptive_triprompt_opd/normal_teacher_count": 4.0,
        "adaptive_triprompt_opd/concise_correct_count": 4.0,
        "adaptive_triprompt_opd/concise_wrong_count": 2.0,
        "adaptive_triprompt_opd/concise_parse_fail_count": 1.0,
        "adaptive_triprompt_opd/concise_missing_box_count": 1.0,
        "adaptive_triprompt_opd/concise_cap_hit_count": 1.0,
        "adaptive_triprompt_opd/normal_eos_count": 7.0,
        "adaptive_triprompt_opd/normal_eos_ratio": 0.7,
        "adaptive_triprompt_opd/normal_cap_hit_count": 2.0,
        "adaptive_triprompt_opd/normal_cap_hit_ratio": 0.2,
        "adaptive_triprompt_opd/concise_eos_count": 5.0,
        "adaptive_triprompt_opd/concise_eos_ratio": 5.0 / 6.0,
        "adaptive_triprompt_opd/concise_cap_hit_ratio": 1.0 / 6.0,
        "adaptive_triprompt_opd/normal_response_tokens": 100.0,
        "adaptive_triprompt_opd/actor_supervised_tokens": 100.0,
        "adaptive_triprompt_opd/concise_response_tokens": 40.0,
        "adaptive_triprompt_opd/supervision_token_residual": 0.0,
        "adaptive_triprompt_opd/full_response_preservation_ratio": 1.0,
        "adaptive_triprompt_opd/route_weight_min": 1.0,
        "adaptive_triprompt_opd/route_weight_max": 1.0,
        "adaptive_triprompt_opd/sensitive_budget_min": 8.0,
        "adaptive_triprompt_opd/sensitive_budget_mean": 10.0,
        "adaptive_triprompt_opd/sensitive_budget_max": 12.0,
        "adaptive_triprompt_opd/total_questions_cumulative": 10.0,
        "adaptive_triprompt_opd/concise_probe_count_cumulative": 6.0,
        "adaptive_triprompt_opd/easy_count_cumulative": 4.0,
        "adaptive_triprompt_opd/sensitive_count_cumulative": 2.0,
        "adaptive_triprompt_opd/hard_count_cumulative": 4.0,
        "actor/pg_loss": 0.01,
        "actor/grad_norm": 2.0,
        "rollout_corr/rollout_is_max": 2.5,
        "rollout_corr/rollout_is_veto_fraction": 0.0,
        "rollout_corr/rollout_is_catastrophic_token_fraction": 0.0,
        "perf/max_memory_allocated_gb": 100.0,
        "perf/max_memory_reserved_gb": 110.0,
        "timing_s/normal_rollout": 100.0,
        "timing_s/normal_reward": 10.0,
        "timing_s/concise_probe": 50.0,
        "timing_s/concise_reward": 5.0,
        "timing_s/triprompt_routing": 1.0,
        "timing_s/old_log_prob": 30.0,
        "timing_s/ref": 300.0,
        "timing_s/update_actor": 400.0,
        "timing_s/step": 1000.0,
    }


def _write_fixture(
    tmp_path: Path, *, overrides: dict[str, float] | None = None, source_commit: str = SOURCE_COMMIT
) -> tuple[Path, Path, str]:
    dataset = tmp_path / "train.parquet"
    dataset.write_bytes(b"immutable training fixture\n")
    dataset_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
    metrics = _metrics()
    metrics.update(overrides or {})
    metric_fields = " - ".join(f"{key}:{value}" for key, value in metrics.items())
    log = tmp_path / "profile.log"
    log.write_text(
        "\n".join(
            [
                "RUN_MODE=profile",
                f"TRAIN_DATA={dataset}",
                f"STUDENT_MODEL={STUDENT_MODEL}",
                f"TEACHER_MODEL={TEACHER_MODEL}",
                f"EXPERIMENT_NAME={EXPERIMENT}",
                f"SOURCE_COMMIT={source_commit}",
                "triprompt_command=/opt/python -m verl.trainer.main_ppo algorithm.adaptive_triprompt_opd.enabled=True",
                f"step:1 - {metric_fields}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return log, dataset, dataset_hash


def _validate(log: Path, dataset: Path, dataset_hash: str):
    return validate_adaptive_triprompt_profile(
        log_path=log,
        expected_step=1,
        expected_questions=10,
        expected_total_steps=50,
        source_commit=SOURCE_COMMIT,
        train_data_path=dataset,
        expected_train_data_sha256=dataset_hash,
        student_model=STUDENT_MODEL,
        teacher_model=TEACHER_MODEL,
        expected_experiment=EXPERIMENT,
        max_memory_allocated_gb=140.0,
        max_step_seconds=1800.0,
        max_projected_train_hours=20.0,
    )


def test_validator_accepts_route_endpoint_supervision_and_health_contract(tmp_path: Path) -> None:
    log, dataset, dataset_hash = _write_fixture(tmp_path)
    report = _validate(log, dataset, dataset_hash)

    assert report["decision"] == "pass"
    assert report["route_counts"] == {"easy": 4, "sensitive": 2, "hard": 4}
    assert report["teacher_prompt_counts"] == {
        "concise": 4,
        "budget": 2,
        "normal": 4,
    }
    assert report["normal_response_tokens"] == 100.0
    assert report["actor_supervised_tokens"] == 100.0
    assert report["supervision_token_residual"] == 0.0
    assert report["full_response_preservation_ratio"] == 1.0
    assert report["endpoint_counts"] == {
        "normal_eos": 7,
        "normal_cap_hit": 2,
        "concise_eos": 5,
        "concise_cap_hit": 1,
        "concise_parse_fail": 1,
    }
    assert report["projected_50_step_train_hours"] == pytest.approx(1000 * 50 / 3600)
    assert report["train_data_sha256"] == dataset_hash
    assert len(report["log_sha256"]) == 64
    assert len(report["config_command_sha256"]) == 64


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"adaptive_triprompt_opd/hard_count": 3.0}, "partition"),
        ({"adaptive_triprompt_opd/concise_probe_count": 5.0}, "probe"),
        ({"adaptive_triprompt_opd/budget_teacher_count": 3.0}, "teacher"),
        ({"adaptive_triprompt_opd/actor_supervised_tokens": 99.0}, "supervision"),
        ({"adaptive_triprompt_opd/concise_response_tokens": -1.0}, "diagnostic tokens"),
        ({"adaptive_triprompt_opd/route_weight_min": 0.5}, "route weight"),
        ({"adaptive_triprompt_opd/normal_eos_count": 11.0}, "endpoint"),
        ({"adaptive_triprompt_opd/concise_missing_box_count": 2.0}, "parse"),
        ({"actor/grad_norm": 0.0}, "grad_norm"),
        ({"rollout_corr/rollout_is_max": 5.1}, "rollout_is_max"),
        ({"perf/max_memory_allocated_gb": 141.0}, "memory"),
        ({"timing_s/step": 1801.0}, "step time"),
    ],
)
def test_validator_fails_closed_on_gate_violation(
    tmp_path: Path, overrides: dict[str, float], message: str
) -> None:
    log, dataset, dataset_hash = _write_fixture(tmp_path, overrides=overrides)
    with pytest.raises(ValueError, match=message):
        _validate(log, dataset, dataset_hash)


def test_validator_rejects_source_and_dataset_manifest_mismatch(tmp_path: Path) -> None:
    log, dataset, dataset_hash = _write_fixture(tmp_path, source_commit="b" * 40)
    with pytest.raises(ValueError, match="SOURCE_COMMIT"):
        _validate(log, dataset, dataset_hash)

    log, dataset, dataset_hash = _write_fixture(tmp_path)
    with pytest.raises(ValueError, match="sha256"):
        _validate(log, dataset, "0" * 64)


def test_write_report_is_atomic_and_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "acceptance.json"
    write_report_atomic(output, {"decision": "pass"})
    assert json.loads(output.read_text()) == {"decision": "pass"}
    with pytest.raises(FileExistsError):
        write_report_atomic(output, {"decision": "fail"})


def test_cli_writes_immutable_failure_report(tmp_path: Path) -> None:
    log, dataset, dataset_hash = _write_fixture(
        tmp_path, overrides={"adaptive_triprompt_opd/supervision_token_residual": 1.0}
    )
    output = tmp_path / "failure.json"
    exit_code = main(
        [
            "--log",
            str(log),
            "--expected-step",
            "1",
            "--expected-questions",
            "10",
            "--expected-total-steps",
            "50",
            "--source-commit",
            SOURCE_COMMIT,
            "--train-data",
            str(dataset),
            "--expected-train-data-sha256",
            dataset_hash,
            "--student-model",
            STUDENT_MODEL,
            "--teacher-model",
            TEACHER_MODEL,
            "--expected-experiment",
            EXPERIMENT,
            "--max-memory-allocated-gb",
            "140",
            "--max-step-seconds",
            "1800",
            "--max-projected-train-hours",
            "20",
            "--output",
            str(output),
        ]
    )
    assert exit_code == 2
    report = json.loads(output.read_text())
    assert report["decision"] == "fail"
    assert "supervision" in report["error"]
