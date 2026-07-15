from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from math_eval.opd_proxy_gradient_projection import (
    ProjectionConfig,
    project_full_gradient,
)
from math_eval.opd_proxy_gradient_verify_artifacts import (
    TrajectoryKey,
    canonical_json_bytes,
    load_vector_set,
    sha256_file,
    sha256_id_lines,
)
from math_eval import replay_opd_proxy_gradients as replay_module
from math_eval.replay_opd_proxy_gradients import (
    ReplayTrajectory,
    ReplayVector,
    accumulate_target_group_full_gradient,
    build_parameter_layout,
    compute_replay_loss,
    flatten_float32_gradients,
    load_capture_trajectories,
    proxy_group_vector_id,
    proxy_vector_id,
    replay_proxy_group,
    replay_proxy_trajectory,
    replay_target_group,
    run_capture_replay_shard,
    run_replay_shard,
    target_group_vector_id,
    validate_replay_coverage,
)
from verl.trainer.ppo.opd_proxy_verify_capture import (
    finalize_capture_seed,
    trajectory_keys_sha256,
    write_tensor_chunks_atomic,
)
from verl.workers.config import ActorConfig, PolicyLossConfig


def test_load_replay_actor_uses_checkpoint_bfloat16_precision(tmp_path, monkeypatch):
    model_root = tmp_path / "model"
    model_root.mkdir()
    (model_root / "config.json").write_text("{}", encoding="utf-8")
    seen = {}

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def to(self, device):
            seen["device"] = device
            return self

        def gradient_checkpointing_enable(self, **kwargs):
            seen["gradient_checkpointing"] = kwargs

    class Actor:
        def __init__(self, *, config, actor_module, actor_optimizer):
            self.config = config
            self.actor_module = actor_module
            self.actor_optimizer = actor_optimizer

    def from_pretrained(cls, path, **kwargs):
        seen["model_path"] = Path(path)
        seen["torch_dtype"] = kwargs["torch_dtype"]
        return Model()

    from transformers import AutoModelForCausalLM
    from verl.models.transformers import monkey_patch as monkey_patch_module
    from verl.workers.actor import dp_actor as dp_actor_module

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        AutoModelForCausalLM,
        "from_pretrained",
        classmethod(from_pretrained),
    )
    monkeypatch.setattr(monkey_patch_module, "apply_monkey_patch", lambda **kwargs: None)
    monkeypatch.setattr(dp_actor_module, "DataParallelPPOActor", Actor)

    actor = replay_module.load_replay_actor(model_root)

    assert seen["model_path"] == model_root.resolve()
    assert seen["device"] == "cuda:0"
    assert seen["torch_dtype"] is torch.bfloat16
    assert actor.actor_optimizer is None


def test_default_replay_actor_config_constructs_without_typed_model_config():
    config = replay_module._default_replay_actor_config()
    assert config.strategy == "fsdp"
    assert config.policy_loss.loss_mode == "vanilla"
    assert config.policy_loss.only_reverse_kl_advantages is True
    assert config.use_remove_padding is True
    assert config.use_fused_kernels is False


class ToyReplayActor:
    def __init__(self):
        self.actor_module = torch.nn.Linear(1, 1, bias=True)
        with torch.no_grad():
            self.actor_module.weight.fill_(0.4)
            self.actor_module.bias.fill_(-0.1)
        self.config = ActorConfig(
            strategy="fsdp",
            rollout_n=4,
            ppo_mini_batch_size=1,
            ppo_micro_batch_size_per_gpu=1,
            ppo_epochs=1,
            use_dynamic_bsz=False,
            use_torch_compile=False,
            use_kl_loss=True,
            kl_loss_coef=0.0,
            entropy_coeff=0.0,
            policy_loss=PolicyLossConfig(
                loss_mode="vanilla", only_reverse_kl_advantages=True
            ),
        )

    def replay_forward_log_prob(self, trajectory: ReplayTrajectory) -> torch.Tensor:
        features = trajectory.responses.to(torch.float32).unsqueeze(-1)
        return self.actor_module(features).squeeze(-1)


def _trajectory(
    slot: int,
    *,
    stable_id: str = "q0",
    seed: int = 42,
    response_values: tuple[int, int, int] = (1, 2, 3),
    mask: tuple[int, int, int] = (1, 1, 1),
    ref_delta: float = 0.5,
) -> ReplayTrajectory:
    actor = ToyReplayActor()
    responses = torch.tensor([response_values], dtype=torch.long)
    with torch.no_grad():
        current = actor.actor_module(responses.float().unsqueeze(-1)).squeeze(-1)
    response_mask = torch.tensor([mask], dtype=torch.bool)
    rollout_log_prob = current - 0.2
    weights = response_mask.float() * torch.exp(torch.tensor(0.2))
    ref = current + ref_delta * (slot + 1)
    return ReplayTrajectory(
        stable_id=stable_id,
        split="candidate",
        engine_seed=seed,
        rollout_slot=slot,
        input_ids=torch.cat([torch.zeros((1, 1), dtype=torch.long), responses], dim=1),
        responses=responses,
        attention_mask=torch.ones((1, 4), dtype=torch.long),
        position_ids=torch.arange(4).unsqueeze(0),
        response_mask=response_mask,
        rollout_log_prob=rollout_log_prob,
        batch_old_log_prob=current,
        ref_log_prob=ref,
        rollout_is_weights=weights,
        captured_current_log_prob=current,
        captured_local_old_log_prob=current,
        source_capture_sha256="a" * 64,
    )


def _trajectories() -> tuple[ReplayTrajectory, ...]:
    return (
        _trajectory(0, response_values=(1, 2, 3), mask=(1, 1, 1)),
        _trajectory(1, response_values=(2, 1, 4), mask=(1, 1, 0)),
        _trajectory(2, response_values=(3, 2, 1), mask=(1, 0, 0)),
        _trajectory(3, response_values=(4, 3, 2), mask=(1, 1, 1)),
    )


class LinearProjector:
    def __init__(self, gradient_dimension: int):
        values = torch.arange(1, gradient_dimension * 1024 + 1, dtype=torch.float32)
        self.matrix = values.reshape(gradient_dimension, 1024) / 1000.0

    def project(self, value: torch.Tensor, *, model_id: int):
        assert model_id == 0
        return value.float() @ self.matrix.to(value.device)


def _direct_gradient(actor: ToyReplayActor, trajectory: ReplayTrajectory) -> torch.Tensor:
    actor.actor_module.zero_grad(set_to_none=True)
    result = compute_replay_loss(actor, trajectory)
    result.loss.backward()
    gradient = flatten_float32_gradients(actor.actor_module)
    actor.actor_module.zero_grad(set_to_none=True)
    return gradient


def test_target_group_is_mean_of_four_canonical_trajectory_gradients():
    actor = ToyReplayActor()
    trajectories = _trajectories()
    direct = [_direct_gradient(actor, trajectory) for trajectory in trajectories]
    expected = torch.stack(direct).mean(dim=0)

    actual = accumulate_target_group_full_gradient(actor, trajectories)

    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)


def test_replay_uses_actor_local_old_not_captured_batch_old():
    actor = ToyReplayActor()
    trajectory = _trajectory(0)
    first = compute_replay_loss(
        actor,
        replace(
            trajectory,
            batch_old_log_prob=torch.full_like(
                trajectory.batch_old_log_prob, -100.0
            ),
        ),
    )
    second = compute_replay_loss(
        actor,
        replace(
            trajectory,
            batch_old_log_prob=torch.full_like(
                trajectory.batch_old_log_prob, 100.0
            ),
        ),
    )

    torch.testing.assert_close(first.loss, second.loss)
    torch.testing.assert_close(first.advantages, second.advantages)
    torch.testing.assert_close(first.local_old_log_prob, first.current_log_prob.detach())


def test_proxy_vector_ids_encode_seed_slot_and_aggregation():
    assert proxy_vector_id("q7", 43, rollout_slot=2) == "P:q7:seed=43:slot=2:n1"
    assert proxy_group_vector_id("q7", 43) == "P:q7:seed=43:n4"
    assert target_group_vector_id("q7", 42) == "T:q7:seed=42:n4"


def test_proxy_group_norm_and_projection_use_mean_full_gradient():
    actor = ToyReplayActor()
    trajectories = _trajectories()
    layout = build_parameter_layout(actor.actor_module)
    projector = LinearProjector(layout.total_numel)
    individual_full = [_direct_gradient(actor, row) for row in trajectories]
    individual_projected = [
        project_full_gradient(projector, full, ProjectionConfig()).squeeze(0)
        for full in individual_full
    ]
    expected_full = torch.stack(individual_full).mean(dim=0)

    actual = replay_proxy_group(actor, trajectories, projector)

    torch.testing.assert_close(
        actual.projected_gradient,
        project_full_gradient(projector, expected_full, ProjectionConfig()).squeeze(0),
    )
    torch.testing.assert_close(
        actual.projected_gradient,
        torch.stack(individual_projected).mean(dim=0),
        rtol=5e-3,
        atol=5e-3,
    )
    torch.testing.assert_close(actual.full_gradient_norm, expected_full.norm())
    assert [record.vector_id for record in actual.individual] == [
        proxy_vector_id("q0", 42, slot) for slot in range(4)
    ]


def test_proxy_trajectory_zeros_existing_gradients_and_uses_unscaled_loss():
    actor = ToyReplayActor()
    trajectory = _trajectory(0)
    expected = _direct_gradient(actor, trajectory)
    for parameter in actor.actor_module.parameters():
        parameter.grad = torch.full_like(parameter, 1000.0)
    projector = LinearProjector(expected.numel())

    record = replay_proxy_trajectory(actor, trajectory, projector)

    expected_projected = project_full_gradient(
        projector, expected, ProjectionConfig()
    ).squeeze(0)
    torch.testing.assert_close(record.projected_gradient, expected_projected)
    torch.testing.assert_close(record.full_gradient_norm, expected.norm())
    assert all(parameter.grad is None for parameter in actor.actor_module.parameters())


def test_target_uses_per_trajectory_token_means_before_group_average():
    actor = ToyReplayActor()
    trajectories = _trajectories()
    per_trajectory = [_direct_gradient(actor, row) for row in trajectories]
    expected_equal_weight = torch.stack(per_trajectory).mean(dim=0)
    token_counts = torch.tensor(
        [row.response_mask.sum().item() for row in trajectories], dtype=torch.float32
    )
    length_weighted = (
        torch.stack(per_trajectory) * token_counts.unsqueeze(1)
    ).sum(dim=0) / token_counts.sum()

    actual = accumulate_target_group_full_gradient(actor, trajectories)

    torch.testing.assert_close(actual, expected_equal_weight, rtol=0, atol=1e-7)
    assert not torch.allclose(actual, length_weighted)


def test_group_diagnostics_follow_equal_trajectory_formulas():
    actor = ToyReplayActor()
    trajectories = _trajectories()
    projector = LinearProjector(build_parameter_layout(actor.actor_module).total_numel)

    target = replay_target_group(actor, trajectories, projector)

    reverse_kls = []
    rms_values = []
    valid_counts = []
    for trajectory in trajectories:
        current = compute_replay_loss(actor, trajectory).local_old_log_prob
        mask = trajectory.response_mask.bool()
        reverse_kls.append((current - trajectory.ref_log_prob)[mask].mean())
        rms_values.append(
            torch.sqrt(
                torch.square(trajectory.ref_log_prob - current)[mask].mean()
            )
        )
        valid_counts.append(float(mask.sum()))
    torch.testing.assert_close(
        target.sampled_reverse_kl, torch.stack(reverse_kls).mean()
    )
    torch.testing.assert_close(
        target.opd_signal_rms,
        torch.sqrt(torch.square(torch.stack(rms_values)).mean()),
    )
    assert target.valid_token_count.item() == pytest.approx(np.mean(valid_counts))
    assert target.response_length.item() == pytest.approx(np.mean(valid_counts))


def test_parameter_layout_uses_registration_order_and_rejects_local_shard():
    class TiedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Linear(2, 2, bias=False)
            self.second = torch.nn.Linear(2, 1, bias=False)
            self.tied = self.first

    model = TiedModel()
    layout = build_parameter_layout(model)

    assert [entry.name for entry in layout.entries] == [
        "first.weight",
        "second.weight",
    ]
    assert [entry.offset for entry in layout.entries] == [0, 4]
    assert layout.total_numel == 6
    assert len(layout.sha256) == 64
    model._opd_fsdp_local_shard = True
    with pytest.raises(ValueError, match="full unsharded"):
        build_parameter_layout(model)


def test_completed_capture_loader_joins_actor_tensors_by_compound_key(tmp_path):
    root = tmp_path / "capture"
    keys = tuple(TrajectoryKey("q0", 42, slot) for slot in range(4))
    parent_hashes = {"sample": "1" * 64, "config": "2" * 64}
    current = torch.tensor(
        [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]],
        dtype=torch.float32,
    )
    mask = torch.ones((4, 2), dtype=torch.bool)
    tensors = {
        "input_ids": torch.arange(12).reshape(4, 3),
        "responses": torch.arange(8).reshape(4, 2),
        "attention_mask": torch.ones((4, 3), dtype=torch.long),
        "position_ids": torch.arange(3).repeat(4, 1),
        "response_mask": mask,
        "rollout_log_prob": current - 0.2,
        "batch_old_log_prob": current.clone(),
        "ref_log_prob": current + 0.5,
        "rollout_is_weights": mask.float() * torch.exp(torch.tensor(0.2)),
        "current_log_prob": current.clone(),
        "local_old_log_prob": current.clone(),
        "advantages": torch.full_like(current, 0.5),
        "ratio": torch.ones_like(current),
        "policy_loss": torch.arange(1, 5, dtype=torch.float32),
    }
    sidecars = [
        {
            "stable_id": key.stable_id,
            "engine_seed": key.engine_seed,
            "rollout_slot": key.rollout_slot,
            "split": "candidate",
            "manifest_index": 0,
            "actor_rank": 0,
        }
        for key in keys
    ]
    for directory in (
        root / "rollout",
        root / "trainer_boundary",
        root / "actor/rank_0",
    ):
        write_tensor_chunks_atomic(
            directory,
            tensors=tensors,
            sidecar_rows=sidecars,
            parent_hashes=parent_hashes,
            chunk_size=2,
        )
    finalize_capture_seed(
        root,
        expected_keys=keys,
        actor_rank_expected_keys={0: keys},
        parent_hashes=parent_hashes,
        actor_parameter_sha256_before="3" * 64,
        actor_parameter_sha256_after="3" * 64,
        seed_manifest={
            "vllm_version": "test",
            "engine_args": {"seed": 42},
            "sampling_args": {"n": 4},
            "ordered_prompt_keys_sha256": sha256_id_lines(["q0"]),
            "returned_compound_keys_sha256": trajectory_keys_sha256(keys),
            "model_hashes": {"student": "4" * 64},
            "tokenizer_hashes": {"student": "5" * 64},
            "config_hashes": {"resolved": "6" * 64},
            "source_hashes": {"snapshot": "7" * 64},
            "engine_count": 1,
            "generation_call_count": 1,
        },
    )

    loaded = load_capture_trajectories(root)

    assert [(row.stable_id, row.engine_seed, row.rollout_slot) for row in loaded] == [
        ("q0", 42, slot) for slot in range(4)
    ]
    torch.testing.assert_close(loaded[2].ref_log_prob, current[2:3] + 0.5)
    assert loaded[0].source_capture_sha256 == hashlib.sha256(
        (root / "manifest.json").read_bytes()
    ).hexdigest()


def test_production_replay_shard_wires_capture_group_actor_and_projector(
    monkeypatch, tmp_path: Path
):
    capture_root = tmp_path / "capture"
    capture_root.mkdir()
    (capture_root / "manifest.json").write_bytes(canonical_json_bytes({"capture": 1}))
    capture_hash = sha256_file(capture_root / "manifest.json")
    trajectories = tuple(
        replace(row, source_capture_sha256=capture_hash) for row in _trajectories()
    )
    actor = ToyReplayActor()
    projector = LinearProjector(build_parameter_layout(actor.actor_module).total_numel)
    source_snapshot_path = tmp_path / "source_snapshot.json"
    source_snapshot_path.write_bytes(
        canonical_json_bytes({"manifest_sha256": "2" * 64})
    )
    monkeypatch.setattr(
        replay_module, "load_capture_trajectories", lambda path: trajectories
    )
    monkeypatch.setattr(replay_module, "load_replay_actor", lambda *args, **kwargs: actor)
    monkeypatch.setattr(
        replay_module, "verify_prismatic_reference", lambda path: object()
    )
    monkeypatch.setattr(
        replay_module, "construct_cuda_projector", lambda *args, **kwargs: projector
    )
    monkeypatch.setattr(
        replay_module,
        "build_projection_manifest",
        lambda **kwargs: {"dimension": 1024},
    )
    monkeypatch.setattr(
        replay_module,
        "repository_state",
        lambda path: {
            "head": "3" * 40,
            "status": "",
            "status_sha256": "4" * 64,
        },
    )
    monkeypatch.setattr(
        replay_module,
        "build_runtime_metadata",
        lambda profile: {"runtime_profile": profile},
    )

    loaded = run_capture_replay_shard(
        capture_root=capture_root,
        model_path=tmp_path / "model",
        output_directory=tmp_path / "vectors",
        pair="target",
        num_shards=1,
        shard_index=0,
        expected_model_sha256="1" * 64,
        source_snapshot_path=source_snapshot_path,
        reference_repo=tmp_path / "reference",
        repository_root=tmp_path,
        chunk_size=1,
    )

    assert loaded.vector_ids == (target_group_vector_id("q0", 42),)
    assert loaded.manifest["metadata"]["pair"] == "target"
    assert loaded.manifest["metadata"]["parameter_layout"]["total_numel"] == 2


def _vector(index: int, *, stable_id: str = "q0") -> ReplayVector:
    projected = torch.zeros(1024, dtype=torch.float32)
    projected[index] = float(index + 1)
    return ReplayVector(
        vector_id=f"P:{stable_id}:seed=42:slot={index}:n1",
        stable_id=stable_id,
        split="candidate",
        representation="P",
        engine_seed=42,
        rollout_slot=index,
        aggregation="n1",
        source_capture_sha256="a" * 64,
        projected_gradient=projected,
        full_gradient_norm=torch.tensor(float(index + 1)),
        projected_gradient_norm=projected.norm(),
        valid_token_count=torch.tensor(float(index + 2)),
        response_length=torch.tensor(float(index + 2)),
        sampled_reverse_kl=torch.tensor(0.1 * (index + 1)),
        opd_signal_rms=torch.tensor(0.2 * (index + 1)),
    )


def _run_kwargs(tmp_path: Path, factory):
    vector_ids = tuple(_vector(index).vector_id for index in range(2))
    return {
        "output_directory": tmp_path,
        "expected_vector_ids": vector_ids,
        "record_factory": factory,
        "representation": "P",
        "parent_hashes": {"capture": "a" * 64},
        "source_snapshot": {"manifest_sha256": "b" * 64},
        "repository": {
            "head": "c" * 40,
            "status": "",
            "status_sha256": "d" * 64,
        },
        "runtime": {"runtime_profile": "gvendi_analysis"},
        "metadata": {"projection": {"dimension": 1024}},
        "chunk_size": 1,
    }


def test_replay_shard_resumes_only_a_contiguous_vector_prefix(tmp_path: Path):
    first_calls = []

    def interrupted(index):
        first_calls.append(index)
        if index == 1:
            raise RuntimeError("synthetic interruption")
        return _vector(index)

    with pytest.raises(RuntimeError, match="interruption"):
        run_replay_shard(**_run_kwargs(tmp_path, interrupted))
    assert first_calls == [0, 1]
    assert (tmp_path / "vectors_0_1.safetensors").is_file()
    assert not (tmp_path / "COMPLETE.json").exists()

    resumed_calls = []

    def resumed(index):
        resumed_calls.append(index)
        return _vector(index)

    loaded = run_replay_shard(**_run_kwargs(tmp_path, resumed))

    assert resumed_calls == [1]
    assert loaded.vector_ids == tuple(_vector(index).vector_id for index in range(2))
    assert loaded.stable_ids == ("q0", "q0")
    assert loaded.verifier_correct_count is None
    assert loaded.verifier_total is None
    assert loaded.manifest["verifier"] == {"status": "not_computed"}


def test_explicit_resume_exercise_interrupts_after_immutable_chunk(tmp_path: Path):
    first_calls = []
    with pytest.raises(RuntimeError, match="intentional replay interruption"):
        run_replay_shard(
            **_run_kwargs(
                tmp_path,
                lambda index: first_calls.append(index) or _vector(index),
            ),
            exercise_interrupt_after_chunks=1,
        )
    assert first_calls == [0]
    assert (tmp_path / ".vectors_0_1.complete.json").is_file()
    assert not (tmp_path / "COMPLETE.json").exists()

    resumed_calls = []
    loaded = run_replay_shard(
        **_run_kwargs(
            tmp_path,
            lambda index: resumed_calls.append(index) or _vector(index),
        )
    )
    assert resumed_calls == [1]
    assert loaded.vector_ids == tuple(_vector(index).vector_id for index in range(2))

    uninterrupted = tmp_path.parent / f"{tmp_path.name}-uninterrupted"
    run_replay_shard(**_run_kwargs(uninterrupted, _vector))
    resumed_hashes = {
        path.name: sha256_file(path)
        for path in tmp_path.iterdir()
        if path.is_file() and not path.name.endswith(".lock")
    }
    uninterrupted_hashes = {
        path.name: sha256_file(path)
        for path in uninterrupted.iterdir()
        if path.is_file() and not path.name.endswith(".lock")
    }
    assert resumed_hashes == uninterrupted_hashes


def test_replay_resume_rejects_tampered_completed_prefix(tmp_path: Path):
    def interrupted(index):
        if index == 1:
            raise RuntimeError("stop")
        return _vector(index)

    with pytest.raises(RuntimeError, match="stop"):
        run_replay_shard(**_run_kwargs(tmp_path, interrupted))
    sidecar = tmp_path / "vectors_0_1.jsonl"
    sidecar.write_bytes(sidecar.read_bytes() + b" ")
    resumed_calls = []

    with pytest.raises(ValueError, match="sidecar_sha256"):
        run_replay_shard(
            **_run_kwargs(
                tmp_path,
                lambda index: resumed_calls.append(index) or _vector(index),
            )
        )
    assert resumed_calls == []


def test_validate_replay_coverage_requires_exact_vector_ids_and_parent(tmp_path):
    run_replay_shard(**_run_kwargs(tmp_path, _vector))
    expected = tuple(_vector(index).vector_id for index in range(2))

    loaded = validate_replay_coverage(
        [tmp_path],
        expected_vector_ids=expected,
        expected_parent_hashes={"capture": "a" * 64},
        expected_representation="P",
    )

    assert loaded.vector_ids == expected
    with pytest.raises(ValueError, match="coverage"):
        validate_replay_coverage(
            [tmp_path],
            expected_vector_ids=(expected[0], "missing"),
            expected_parent_hashes={"capture": "a" * 64},
            expected_representation="P",
        )


def test_replay_group_rejects_missing_duplicate_or_cross_question_slots():
    actor = ToyReplayActor()
    projector = LinearProjector(build_parameter_layout(actor.actor_module).total_numel)
    rows = list(_trajectories())
    with pytest.raises(ValueError, match="slots"):
        replay_target_group(actor, rows[:3], projector)
    with pytest.raises(ValueError, match="slots"):
        replay_target_group(actor, [rows[0], rows[0], rows[2], rows[3]], projector)
    with pytest.raises(ValueError, match="stable ID"):
        replay_target_group(
            actor,
            [rows[0], rows[1], replace(rows[2], stable_id="other"), rows[3]],
            projector,
        )


def test_vector_loader_rejects_partial_verifier_fields(tmp_path):
    run_replay_shard(**_run_kwargs(tmp_path, _vector))
    assert load_vector_set(tmp_path).verifier_total is None
