from verl.trainer.config import AlgoConfig, CandidateSelectionConfig
from verl.utils.config import omega_conf_to_dataclass


def test_candidate_selection_config_defaults_and_overrides():
    default = CandidateSelectionConfig()
    assert default.enabled is False
    assert default.method == "shortest_correct_else_teacher"
    assert default.correct_reward_threshold == 0.5
    assert default.keep_per_uid == 1

    cfg = omega_conf_to_dataclass(
        {
            "_target_": "verl.trainer.config.AlgoConfig",
            "candidate_selection": {
                "_target_": "verl.trainer.config.CandidateSelectionConfig",
                "enabled": True,
                "method": "shortest_correct_else_teacher",
                "correct_reward_threshold": 0.25,
                "keep_per_uid": 1,
            },
        }
    )
    assert isinstance(cfg, AlgoConfig)
    assert isinstance(cfg.candidate_selection, CandidateSelectionConfig)
    assert cfg.candidate_selection.enabled is True
    assert cfg.candidate_selection.correct_reward_threshold == 0.25
