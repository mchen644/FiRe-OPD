"""First-class stage profiles for OPD proxy-gradient verification."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

from math_eval.opd_proxy_gradient_verify_artifacts import canonical_json_bytes

EFFICACY_PILOT = "efficacy_pilot"
StageKind: TypeAlias = Literal[0, 1, 2, "efficacy_pilot"]

_NUMBERED_REPRESENTATIONS = tuple(
    [
        f"P_n1:seed={seed}:slot={slot}"
        for seed in (42, 43)
        for slot in range(4)
    ]
    + [f"P_n4:seed={seed}" for seed in (42, 43)]
    + ["S", "E"]
    + [f"T:seed={seed}" for seed in (42, 43)]
)
_NUMBERED_TARGET_REPRESENTATIONS = ("T:seed=42", "T:seed=43")


@dataclass(frozen=True)
class StageProfile:
    """Immutable cardinality and execution contract for one experiment stage."""

    key: StageKind
    directory_name: str
    candidate_count: int
    held_out_count: int
    selected_size: int
    primary_k: int
    diagnostic_k: int
    null_draws: int
    generation_seeds: tuple[int, ...]
    native_rollouts: int
    selection_representations: tuple[str, ...]
    target_representations: tuple[str, ...]
    run_sft: bool
    run_embedding: bool
    run_target_oracle: bool
    run_direct_fixture: bool
    require_resume_exercise: bool


_PROFILES: dict[StageKind, StageProfile] = {
    0: StageProfile(
        key=0,
        directory_name="stage_0",
        candidate_count=24,
        held_out_count=8,
        selected_size=5,
        primary_k=2,
        diagnostic_k=2,
        null_draws=100,
        generation_seeds=(42, 43),
        native_rollouts=4,
        selection_representations=_NUMBERED_REPRESENTATIONS,
        target_representations=_NUMBERED_TARGET_REPRESENTATIONS,
        run_sft=True,
        run_embedding=True,
        run_target_oracle=True,
        run_direct_fixture=True,
        require_resume_exercise=True,
    ),
    1: StageProfile(
        key=1,
        directory_name="stage_1",
        candidate_count=768,
        held_out_count=256,
        selected_size=172,
        primary_k=76,
        diagnostic_k=7,
        null_draws=10_000,
        generation_seeds=(42, 43),
        native_rollouts=4,
        selection_representations=_NUMBERED_REPRESENTATIONS,
        target_representations=_NUMBERED_TARGET_REPRESENTATIONS,
        run_sft=True,
        run_embedding=True,
        run_target_oracle=True,
        run_direct_fixture=False,
        require_resume_exercise=False,
    ),
    2: StageProfile(
        key=2,
        directory_name="stage_2",
        candidate_count=1_536,
        held_out_count=512,
        selected_size=345,
        primary_k=153,
        diagnostic_k=15,
        null_draws=10_000,
        generation_seeds=(42, 43),
        native_rollouts=4,
        selection_representations=_NUMBERED_REPRESENTATIONS,
        target_representations=_NUMBERED_TARGET_REPRESENTATIONS,
        run_sft=True,
        run_embedding=True,
        run_target_oracle=True,
        run_direct_fixture=False,
        require_resume_exercise=False,
    ),
    EFFICACY_PILOT: StageProfile(
        key=EFFICACY_PILOT,
        directory_name=EFFICACY_PILOT,
        candidate_count=250,
        held_out_count=84,
        selected_size=56,
        primary_k=25,
        diagnostic_k=3,
        null_draws=10_000,
        generation_seeds=(42,),
        native_rollouts=1,
        selection_representations=("P_pilot",),
        target_representations=("T_pilot",),
        run_sft=False,
        run_embedding=False,
        run_target_oracle=False,
        run_direct_fixture=False,
        require_resume_exercise=False,
    ),
}

# Every field below is identical between Stage 1 and the one-time pilot. The
# only sanctioned differences are added by capture_algorithm_contract().
COMMON_CAPTURE_CONTRACT: dict[str, object] = {
    "temperature": 1.0,
    "top_p": 1.0,
    "max_prompt_length": 2_048,
    "max_response_length": 16_384,
    "thinking": False,
    "policy_loss_mode": "vanilla",
    "only_reverse_kl_advantages": True,
    "ppo_epochs": 1,
    "mini_batches_per_actor_rank": 1,
    "micro_batch_size_per_gpu": 1,
    "dynamic_micro_batching": False,
    "loss_aggregation": "token-mean",
    "rollout_correction": "token-level-is",
    "rollout_is_upper_threshold": 5.0,
    "rollout_is_batch_normalization": False,
    "entropy_coefficient": 0.0,
    "use_explicit_kl_teacher_path": True,
    "explicit_kl_loss_coefficient": 0.0,
    "kl_in_reward": False,
    "candidate_selection": False,
    "length_aware_opd": False,
    "difficulty_aware_opd": False,
    "tale_esr": False,
    "teacher_prompt_routing": False,
    "rethinking_probe": False,
    "optimizer_step": "forbidden",
    "gradient_clipping": "forbidden",
    "parameter_update": "forbidden",
    "projection_family": "TRAK CudaProjector",
    "projection_type": "rademacher",
    "projection_dimension": 1_024,
    "projection_seed": 0,
    "projection_block_size": 128,
    "projection_max_batch_size": 16,
    "projection_model_id": 0,
}


def parse_stage_kind(value: str | int) -> StageKind:
    """Parse a CLI/artifact stage value without integer aliasing surprises."""
    if isinstance(value, bool):
        raise ValueError(f"unsupported stage: {value}")
    if isinstance(value, int) and value in (0, 1, 2):
        return cast(StageKind, value)
    if isinstance(value, str) and value in {"0", "1", "2"}:
        return cast(StageKind, int(value))
    if value == EFFICACY_PILOT:
        return EFFICACY_PILOT
    raise ValueError(f"unsupported stage: {value}")


def stage_profile(stage: str | int) -> StageProfile:
    """Return the exact immutable profile for a supported stage."""
    return _PROFILES[parse_stage_kind(stage)]


def stage_directory_name(stage: str | int) -> str:
    return stage_profile(stage).directory_name


def capture_algorithm_contract(stage: str | int) -> dict[str, object]:
    """Build the canonical cross-stage algorithm contract payload."""
    profile = stage_profile(stage)
    return {
        **COMMON_CAPTURE_CONTRACT,
        "stage_identity": profile.key,
        "rollout_n": profile.native_rollouts,
        "generation_seeds": list(profile.generation_seeds),
    }


def capture_algorithm_contract_sha256(stage: str | int) -> str:
    return hashlib.sha256(canonical_json_bytes(capture_algorithm_contract(stage))).hexdigest()


__all__ = [
    "COMMON_CAPTURE_CONTRACT",
    "EFFICACY_PILOT",
    "StageKind",
    "StageProfile",
    "capture_algorithm_contract",
    "capture_algorithm_contract_sha256",
    "parse_stage_kind",
    "stage_directory_name",
    "stage_profile",
]
