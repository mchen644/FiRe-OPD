from __future__ import annotations

import pytest

from math_eval.opd_proxy_gradient_stage_profiles import (
    EFFICACY_PILOT,
    capture_algorithm_contract,
    capture_algorithm_contract_sha256,
    parse_stage_kind,
    stage_directory_name,
    stage_profile,
)


def test_efficacy_pilot_profile_is_exact():
    profile = stage_profile(EFFICACY_PILOT)
    assert profile.candidate_count == 250
    assert profile.held_out_count == 84
    assert profile.selected_size == 56
    assert profile.primary_k == 25
    assert profile.diagnostic_k == 3
    assert profile.null_draws == 10_000
    assert profile.generation_seeds == (42,)
    assert profile.native_rollouts == 1
    assert profile.selection_representations == ("P_pilot",)
    assert profile.target_representations == ("T_pilot",)
    assert profile.run_sft is False
    assert profile.run_embedding is False
    assert profile.run_target_oracle is False
    assert profile.run_direct_fixture is False
    assert profile.require_resume_exercise is False
    assert stage_directory_name(EFFICACY_PILOT) == "efficacy_pilot"


def test_stage_parser_accepts_only_numbered_stages_and_efficacy_pilot():
    assert [parse_stage_kind(value) for value in ("0", "1", "2")] == [0, 1, 2]
    assert parse_stage_kind(EFFICACY_PILOT) == EFFICACY_PILOT
    assert parse_stage_kind(0) == 0
    for value in ("3", "-1", "pilot", "stage_1", 3, -1, True):
        with pytest.raises(ValueError, match="unsupported stage"):
            parse_stage_kind(value)


def test_numbered_stage_profiles_preserve_frozen_cardinalities():
    expected = {
        0: (24, 8, 5, 2, 2, 100),
        1: (768, 256, 172, 76, 7, 10_000),
        2: (1536, 512, 345, 153, 15, 10_000),
    }
    for stage, values in expected.items():
        profile = stage_profile(stage)
        assert (
            profile.candidate_count,
            profile.held_out_count,
            profile.selected_size,
            profile.primary_k,
            profile.diagnostic_k,
            profile.null_draws,
        ) == values
        assert profile.generation_seeds == (42, 43)
        assert profile.native_rollouts == 4
        assert stage_directory_name(stage) == f"stage_{stage}"


def test_pilot_contract_differs_only_in_approved_fields():
    stage1 = capture_algorithm_contract(1)
    pilot = capture_algorithm_contract(EFFICACY_PILOT)
    assert stage1["rollout_n"] == 4
    assert pilot["rollout_n"] == 1
    assert stage1["generation_seeds"] == [42, 43]
    assert pilot["generation_seeds"] == [42]
    ignored = {"stage_identity", "rollout_n", "generation_seeds"}
    assert {key: value for key, value in stage1.items() if key not in ignored} == {
        key: value for key, value in pilot.items() if key not in ignored
    }
    assert stage1["max_response_length"] == pilot["max_response_length"] == 16_384
    assert stage1["max_prompt_length"] == pilot["max_prompt_length"] == 2_048
    assert stage1["temperature"] == pilot["temperature"] == 1.0
    assert stage1["top_p"] == pilot["top_p"] == 1.0
    assert stage1["optimizer_step"] == pilot["optimizer_step"] == "forbidden"
    assert capture_algorithm_contract_sha256(1) != capture_algorithm_contract_sha256(
        EFFICACY_PILOT
    )
    assert capture_algorithm_contract_sha256(1) == capture_algorithm_contract_sha256(
        "1"
    )
