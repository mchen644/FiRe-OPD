import json
from pathlib import Path

import pytest

from math_eval.validate_adaptive_concise_profile import (
    validate_adaptive_concise_profile,
    write_report_atomic,
)


def _metrics(**updates) -> dict[str, float]:
    metrics = {
        "training/global_step": 1.0,
        "adaptive_concise_opd/total_questions": 1024.0,
        "adaptive_concise_opd/normal_correct_count": 700.0,
        "adaptive_concise_opd/normal_wrong_count": 324.0,
        "adaptive_concise_opd/concise_probe_count": 700.0,
        "adaptive_concise_opd/easy_count": 500.0,
        "adaptive_concise_opd/learnable_count": 200.0,
        "adaptive_concise_opd/hard_count": 324.0,
        "adaptive_concise_opd/concise_teacher_count": 500.0,
        "adaptive_concise_opd/normal_teacher_count": 524.0,
        "adaptive_concise_opd/response_budget_residual_tokens": 0.0,
        "adaptive_concise_opd/response_budget_ratio": 1.0,
        "actor/pg_loss": 0.001,
        "actor/grad_norm": 2.5,
        "rollout_corr/rollout_is_max": 3.2,
        "rollout_corr/rollout_is_veto_fraction": 0.0,
        "rollout_corr/rollout_is_catastrophic_token_fraction": 0.0,
        "timing_s/normal_rollout": 100.0,
        "timing_s/normal_reward": 2.0,
        "timing_s/concise_probe": 30.0,
        "timing_s/concise_reward": 1.0,
        "timing_s/old_log_prob": 10.0,
        "timing_s/ref": 20.0,
        "timing_s/update_actor": 25.0,
        "timing_s/step": 190.0,
    }
    metrics.update(updates)
    return metrics


def _line(metrics: dict[str, float], *, step: int = 1) -> str:
    fields = " - ".join(f"{key}:{value}" for key, value in metrics.items())
    return f"\x1b[36m(TaskRunner pid=123)\x1b[0m step:{step} - {fields}\n"


def _write_log(path: Path, metrics: dict[str, float], *, duplicate: bool = False) -> None:
    line = _line(metrics)
    path.write_text("startup\n" + line + (line if duplicate else ""), encoding="utf-8")


def test_validate_profile_accepts_exact_count_token_and_runtime_contract(tmp_path: Path) -> None:
    log = tmp_path / "profile.log"
    _write_log(log, _metrics())

    report = validate_adaptive_concise_profile(
        log_path=log,
        expected_step=1,
        expected_questions=1024,
        source_commit="a" * 40,
    )

    assert report["decision"] == "pass"
    assert report["route_counts"] == {"easy": 500, "learnable": 200, "hard": 324}
    assert report["probe_count"] == 700
    assert report["response_budget_residual_tokens"] == 0.0
    assert report["response_budget_ratio"] == 1.0
    assert report["rollout_is_max"] == 3.2
    assert report["source_commit"] == "a" * 40
    assert len(report["log_sha256"]) == 64


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"adaptive_concise_opd/total_questions": 1023.0}, "total_questions"),
        ({"adaptive_concise_opd/normal_correct_count": 699.0}, "normal_correct"),
        ({"adaptive_concise_opd/normal_wrong_count": 323.0}, "normal_wrong"),
        ({"adaptive_concise_opd/concise_probe_count": 699.0}, "probe"),
        ({"adaptive_concise_opd/response_budget_residual_tokens": 1.0}, "residual"),
        ({"adaptive_concise_opd/response_budget_ratio": 0.99}, "ratio"),
        ({"actor/pg_loss": float("nan")}, "finite"),
        ({"actor/grad_norm": 0.0}, "grad_norm"),
        ({"rollout_corr/rollout_is_max": 5.01}, "rollout_is_max"),
        ({"rollout_corr/rollout_is_veto_fraction": 0.01}, "veto"),
        ({"rollout_corr/rollout_is_catastrophic_token_fraction": 0.01}, "catastrophic"),
        ({"timing_s/ref": 0.0}, "timing_s/ref"),
    ],
)
def test_validate_profile_rejects_contract_violations(
    tmp_path: Path, updates: dict[str, float], message: str
) -> None:
    log = tmp_path / "profile.log"
    _write_log(log, _metrics(**updates))
    with pytest.raises(ValueError, match=message):
        validate_adaptive_concise_profile(
            log_path=log,
            expected_step=1,
            expected_questions=1024,
            source_commit="b" * 40,
        )


def test_validate_profile_allows_zero_concise_timing_only_for_zero_probes(tmp_path: Path) -> None:
    log = tmp_path / "profile.log"
    _write_log(
        log,
        _metrics(
            **{
                "adaptive_concise_opd/normal_correct_count": 0.0,
                "adaptive_concise_opd/normal_wrong_count": 1024.0,
                "adaptive_concise_opd/concise_probe_count": 0.0,
                "adaptive_concise_opd/easy_count": 0.0,
                "adaptive_concise_opd/learnable_count": 0.0,
                "adaptive_concise_opd/hard_count": 1024.0,
                "adaptive_concise_opd/concise_teacher_count": 0.0,
                "adaptive_concise_opd/normal_teacher_count": 1024.0,
                "timing_s/concise_probe": 0.0,
                "timing_s/concise_reward": 0.0,
            }
        ),
    )
    report = validate_adaptive_concise_profile(
        log_path=log,
        expected_step=1,
        expected_questions=1024,
        source_commit="c" * 40,
    )
    assert report["decision"] == "pass"
    assert report["probe_count"] == 0


def test_validate_profile_rejects_missing_or_duplicate_step_lines(tmp_path: Path) -> None:
    missing = tmp_path / "missing.log"
    missing.write_text(_line(_metrics(), step=2), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one"):
        validate_adaptive_concise_profile(
            log_path=missing,
            expected_step=1,
            expected_questions=1024,
            source_commit="d" * 40,
        )

    duplicate = tmp_path / "duplicate.log"
    _write_log(duplicate, _metrics(), duplicate=True)
    with pytest.raises(ValueError, match="exactly one"):
        validate_adaptive_concise_profile(
            log_path=duplicate,
            expected_step=1,
            expected_questions=1024,
            source_commit="d" * 40,
        )


def test_validate_profile_rejects_missing_required_metric(tmp_path: Path) -> None:
    log = tmp_path / "profile.log"
    metrics = _metrics()
    del metrics["actor/pg_loss"]
    _write_log(log, metrics)
    with pytest.raises(ValueError, match="actor/pg_loss"):
        validate_adaptive_concise_profile(
            log_path=log,
            expected_step=1,
            expected_questions=1024,
            source_commit="e" * 40,
        )


def test_write_report_atomic_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "acceptance.json"
    write_report_atomic(output, {"decision": "pass"})
    assert json.loads(output.read_text()) == {"decision": "pass"}
    with pytest.raises(FileExistsError, match="already exists"):
        write_report_atomic(output, {"decision": "fail"})
