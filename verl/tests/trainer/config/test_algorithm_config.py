from verl.trainer.config import AlgoConfig, CandidateSelectionConfig
from verl.utils.config import omega_conf_to_dataclass


def test_candidate_selection_config_defaults_and_overrides():
    default = CandidateSelectionConfig()
    assert default.enabled is False
    assert default.method == "shortest_correct_else_teacher"
    assert default.correct_reward_threshold == 0.5
    assert default.keep_per_uid == 1
    assert default.teacher_reject_percentile == 20.0
    assert default.drop_rejected_no_correct is True

    cfg = omega_conf_to_dataclass(
        {
            "_target_": "verl.trainer.config.AlgoConfig",
            "candidate_selection": {
                "_target_": "verl.trainer.config.CandidateSelectionConfig",
                "enabled": True,
                "method": "quality_gated_correct_compression",
                "correct_reward_threshold": 0.25,
                "keep_per_uid": 1,
                "teacher_reject_percentile": 15.0,
                "drop_rejected_no_correct": False,
            },
        }
    )
    assert isinstance(cfg, AlgoConfig)
    assert isinstance(cfg.candidate_selection, CandidateSelectionConfig)
    assert cfg.candidate_selection.enabled is True
    assert cfg.candidate_selection.method == "quality_gated_correct_compression"
    assert cfg.candidate_selection.correct_reward_threshold == 0.25
    assert cfg.candidate_selection.teacher_reject_percentile == 15.0
    assert cfg.candidate_selection.drop_rejected_no_correct is False
