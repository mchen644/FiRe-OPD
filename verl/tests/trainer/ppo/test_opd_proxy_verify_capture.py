# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from math_eval.opd_proxy_gradient_verify_artifacts import (
    TrajectoryKey,
    canonical_json_bytes,
    sha256_file,
    sha256_id_lines,
)
from safetensors.torch import load_file
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.config import OpdProxyVerifyCaptureConfig
from verl.trainer.ppo import opd_proxy_verify_capture as capture
from verl.trainer.ppo.opd_proxy_verify_capture import (
    attach_and_validate_keys,
    build_response_mask,
    compute_authoritative_capture_tensors,
    finalize_capture_seed,
    load_and_join_capture_chunks,
    recursive_parameter_sha256,
    resolve_capture_resume_prefix,
    trajectory_keys_sha256,
    validate_capture_contract,
    write_tensor_chunks_atomic,
)
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.config import ActorConfig, PolicyLossConfig


def _direct_fixture_actor_and_trajectory(tmp_path: Path):
    actor = object.__new__(DataParallelPPOActor)
    actor.config = ActorConfig(
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
        opd_proxy_verify_capture_only=True,
        policy_loss=PolicyLossConfig(
            loss_mode="vanilla", only_reverse_kl_advantages=True
        ),
    )
    actor.actor_module = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        actor.actor_module.weight.fill_(0.25)
    actor.actor_optimizer = None

    def fake_forward(model_inputs, temperature, calculate_entropy=False):
        assert temperature == 1.0
        assert calculate_entropy is False
        current = actor.actor_module.weight.sum() * torch.ones_like(
            model_inputs["old_log_probs"]
        )
        return None, current, {}

    actor._forward_micro_batch = fake_forward
    current = torch.full((1, 2), 0.25)
    rollout = torch.zeros((1, 2))
    old = torch.full((1, 2), 0.2)
    mask = torch.ones((1, 2), dtype=torch.bool)
    data = DataProto(
        batch=TensorDict(
            {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "responses": torch.tensor([[2, 3]]),
                "attention_mask": torch.ones((1, 3), dtype=torch.long),
                "position_ids": torch.arange(3).unsqueeze(0),
                "response_mask": mask,
                "rollout_log_probs": rollout,
                "old_log_probs": old,
                "ref_log_prob": current + 0.5,
                "rollout_is_weights": torch.exp(old - rollout) * mask,
                "opd_proxy_verify_rollout_slot": torch.tensor([0]),
                "opd_proxy_verify_engine_seed": torch.tensor([42]),
            },
            batch_size=1,
        ),
        non_tensor_batch={
            "opd_verify_stable_id": np.asarray(["q-smoke"], dtype=object),
            "opd_verify_split": np.asarray(["candidate"], dtype=object),
            "opd_verify_manifest_index": np.asarray([0], dtype=object),
        },
        meta_info={
            "temperature": 1.0,
            "opd_proxy_verify_capture_enabled": True,
            "opd_proxy_verify_direct_gradient_fixture": True,
            "opd_proxy_verify_direct_output_root": str(tmp_path / "direct"),
            "opd_proxy_verify_parent_hashes": {"capture": "a" * 64},
        },
    )
    return actor, data


class _DirectFixtureProjector:
    def project(self, value, *, model_id):
        assert model_id == 0
        return value.float().sum(dim=1, keepdim=True).repeat(1, 1024)


def test_direct_fixture_projects_registered_loss_without_optimizer_step(
    monkeypatch, tmp_path: Path
):
    import verl.workers.actor.dp_actor as actor_module

    actor, trajectory = _direct_fixture_actor_and_trajectory(tmp_path)
    backward_calls = []
    original_backward = torch.Tensor.backward

    def backward_spy(tensor, *args, **kwargs):
        backward_calls.append(tensor.detach().clone())
        return original_backward(tensor, *args, **kwargs)

    monkeypatch.setattr(actor_module, "get_device_id", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.Tensor, "backward", backward_spy)
    actor._optimizer_step = lambda: pytest.fail("optimizer step called")
    before = recursive_parameter_sha256(actor.actor_module)
    expected_current = actor.actor_module.weight.sum() * torch.ones_like(
        trajectory.batch["old_log_probs"]
    )
    expected_loss = compute_authoritative_capture_tensors(
        current_log_prob=expected_current,
        batch_old_log_prob=trajectory.batch["old_log_probs"],
        rollout_log_prob=trajectory.batch["rollout_log_probs"],
        ref_log_prob=trajectory.batch["ref_log_prob"],
        response_mask=trajectory.batch["response_mask"],
        rollout_is_weights=trajectory.batch["rollout_is_weights"],
        actor_config=actor.config,
    )["policy_loss"].detach()
    method = DataParallelPPOActor.capture_direct_opd_proxy_gradient_fixture
    while hasattr(method, "__wrapped__"):
        method = method.__wrapped__

    artifact = method(actor, trajectory, projector=_DirectFixtureProjector())

    assert len(backward_calls) == 1
    assert artifact["loss_mode"] == "vanilla"
    assert artifact["loss_agg_mode"] == "token-mean"
    assert artifact["optimizer_step_called"] is False
    assert artifact["gradient_accumulation_scale_removed"] == 1.0
    torch.testing.assert_close(artifact["policy_loss"], expected_loss)
    assert artifact["actor_parameter_sha256_before"] == before
    assert artifact["actor_parameter_sha256_after"] == before
    assert recursive_parameter_sha256(actor.actor_module) == before
    assert all(parameter.grad is None for parameter in actor.actor_module.parameters())
    assert artifact["projected_gradient"].shape == (1024,)
    assert artifact["projected_gradient"].dtype == torch.float32
    assert (tmp_path / "direct/manifest.json").is_file()
    assert (tmp_path / "direct/direct_gradient.safetensors").is_file()
    assert (tmp_path / "direct/COMPLETE.json").is_file()


def test_direct_fixture_requires_capture_and_explicit_smoke_flags(tmp_path: Path):
    actor, trajectory = _direct_fixture_actor_and_trajectory(tmp_path)
    method = DataParallelPPOActor.capture_direct_opd_proxy_gradient_fixture
    while hasattr(method, "__wrapped__"):
        method = method.__wrapped__
    actor.config = ActorConfig(
        strategy="fsdp",
        rollout_n=4,
        ppo_micro_batch_size_per_gpu=1,
        opd_proxy_verify_capture_only=False,
    )
    with pytest.raises(ValueError, match="capture-only"):
        method(actor, trajectory, projector=_DirectFixtureProjector())

    actor, trajectory = _direct_fixture_actor_and_trajectory(tmp_path)
    trajectory.meta_info["opd_proxy_verify_direct_gradient_fixture"] = False
    with pytest.raises(ValueError, match="smoke fixture flag"):
        method(actor, trajectory, projector=_DirectFixtureProjector())

    actor, trajectory = _direct_fixture_actor_and_trajectory(tmp_path)
    with pytest.raises(ValueError, match="exactly one trajectory"):
        method(
            actor,
            DataProto.concat([trajectory, trajectory]),
            projector=_DirectFixtureProjector(),
        )


def frozen_actor_config() -> ActorConfig:
    return ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=3.0,
        loss_agg_mode="token-mean",
    )


def test_authoritative_advantage_uses_stopped_current_not_batch_old(monkeypatch):
    seen = {}

    def fake_vanilla(**kwargs):
        seen.update(kwargs)
        return torch.tensor(2.5), {
            "actor/pg_clipfrac": 0.0,
            "actor/ppo_kl": 0.0,
            "actor/pg_clipfrac_lower": 0.0,
        }

    monkeypatch.setattr(capture, "get_policy_loss_fn", lambda name: fake_vanilla)
    current = torch.tensor([[0.2, 0.4]], requires_grad=True)
    result = compute_authoritative_capture_tensors(
        current_log_prob=current,
        batch_old_log_prob=torch.tensor([[-3.0, -3.0]]),
        ref_log_prob=torch.tensor([[0.5, 0.1]]),
        response_mask=torch.tensor([[1, 1]]),
        rollout_is_weights=torch.tensor([[1.0, 0.5]], requires_grad=True),
        actor_config=frozen_actor_config(),
    )
    torch.testing.assert_close(
        result["local_old_log_prob"], torch.tensor([[0.2, 0.4]])
    )
    torch.testing.assert_close(result["advantages"], torch.tensor([[0.3, -0.3]]))
    assert not result["advantages"].requires_grad
    assert result["batch_old_log_prob"].tolist() == [[-3.0, -3.0]]
    assert seen["old_log_prob"].data_ptr() == result["local_old_log_prob"].data_ptr()
    assert seen["rollout_is_weights"].requires_grad is False
    assert seen["loss_agg_mode"] == "token-mean"
    assert result["policy_loss"].item() == 2.5

    ratio_gradient = torch.autograd.grad(result["ratio"].sum(), current)[0]
    torch.testing.assert_close(result["ratio"], torch.ones_like(current))
    torch.testing.assert_close(ratio_gradient, torch.ones_like(current))


def test_rollout_is_is_exact_detached_token_ratio_with_upper_only_truncation():
    current = torch.tensor([[0.1, 0.2, 0.3]], requires_grad=True)
    batch_old = torch.tensor([[30.0, -30.0, 1.0]], requires_grad=True)
    rollout = torch.zeros_like(batch_old, requires_grad=True)
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    result = compute_authoritative_capture_tensors(
        current_log_prob=current,
        batch_old_log_prob=batch_old,
        rollout_log_prob=rollout,
        ref_log_prob=torch.tensor([[0.4, 0.1, 0.3]]),
        response_mask=mask,
        rollout_is_weights=None,
        actor_config=frozen_actor_config(),
    )
    expected = torch.tensor([[5.0, torch.exp(torch.tensor(-20.0)), 0.0]])
    torch.testing.assert_close(result["rollout_is_weights"], expected)
    assert result["rollout_is_weights"].requires_grad is False

    with pytest.raises(ValueError, match="rollout IS weights mismatch"):
        compute_authoritative_capture_tensors(
            current_log_prob=current,
            batch_old_log_prob=batch_old,
            rollout_log_prob=rollout,
            ref_log_prob=torch.tensor([[0.4, 0.1, 0.3]]),
            response_mask=mask,
            rollout_is_weights=torch.ones_like(mask),
            actor_config=frozen_actor_config(),
        )


def test_eos_mask_includes_first_eos_and_excludes_following_tokens():
    mask = build_response_mask(
        responses=torch.tensor([[7, 2, 8, 0], [7, 8, 0, 0]]),
        eos_token_id=2,
        pad_token_id=0,
    )
    assert mask.tolist() == [
        [True, True, False, False],
        [True, True, False, False],
    ]


def _key_batch(stable_ids, slots) -> DataProto:
    size = len(stable_ids)
    return DataProto(
        batch=TensorDict(
            {
                "opd_proxy_verify_rollout_slot": torch.tensor(slots),
                "responses": torch.ones((size, 2), dtype=torch.long),
            },
            batch_size=size,
        ),
        non_tensor_batch={
            "opd_verify_stable_id": np.asarray(stable_ids, dtype=object),
            "opd_verify_split": np.asarray(["candidate"] * size, dtype=object),
            "opd_verify_manifest_index": np.asarray(
                list(range(size)), dtype=object
            ),
        },
    )


def test_compound_keys_attach_seed_and_reject_duplicate_missing_or_wrong_slots():
    expected = tuple(
        TrajectoryKey(stable_id, 42, slot)
        for stable_id in ("q0", "q1")
        for slot in range(4)
    )
    batch, keys = attach_and_validate_keys(
        _key_batch(
            ["q0"] * 4 + ["q1"] * 4,
            [0, 1, 2, 3, 0, 1, 2, 3],
        ),
        engine_seed=42,
        expected_keys=expected,
        native_rollouts=4,
    )
    assert keys == expected
    assert batch.batch["opd_proxy_verify_engine_seed"].tolist() == [42] * 8

    with pytest.raises(ValueError, match="duplicate trajectory key"):
        attach_and_validate_keys(
            _key_batch(["q0"] * 4, [0, 0, 2, 3]),
            engine_seed=42,
            native_rollouts=4,
        )
    missing = _key_batch(["q0"] * 4, [0, 1, 2, 3])
    del missing.non_tensor_batch["opd_verify_stable_id"]
    with pytest.raises(ValueError, match="opd_verify_stable_id"):
        attach_and_validate_keys(missing, engine_seed=42, native_rollouts=4)
    missing_slot = _key_batch(["q0"] * 4, [0, 1, 2, 3])
    del missing_slot.batch["opd_proxy_verify_rollout_slot"]
    with pytest.raises(ValueError, match="rollout slot"):
        attach_and_validate_keys(missing_slot, engine_seed=42, native_rollouts=4)
    with pytest.raises(ValueError, match="rollout slot"):
        attach_and_validate_keys(
            _key_batch(["q0"] * 4, [0, 1, 2, 4]),
            engine_seed=42,
            native_rollouts=4,
        )
    with pytest.raises(ValueError, match="missing trajectory keys"):
        attach_and_validate_keys(
            _key_batch(["q0"] * 4, [0, 1, 2, 3]),
            engine_seed=42,
            native_rollouts=4,
            expected_keys=expected,
        )


def test_capture_contract_hashes_the_declared_sample_manifest(tmp_path: Path):
    sample_manifest = tmp_path / "samples.jsonl"
    sample_manifest.write_text('{"stable_id":"q0"}\n', encoding="utf-8")
    config = OpdProxyVerifyCaptureConfig(
        enabled=True,
        output_root=str(tmp_path / "capture"),
        sample_manifest=str(sample_manifest),
        sample_manifest_sha256=sha256_file(sample_manifest),
        stage=0,
        pair="proxy",
        engine_seed=42,
        native_rollouts=4,
        expected_questions=1,
        chunk_size=1,
        schema_version=1,
    )
    normalized = validate_capture_contract(config)
    assert normalized["sample_manifest_sha256"] == sha256_file(sample_manifest)
    assert normalized["native_rollouts"] == 4

    wrong = OpdProxyVerifyCaptureConfig(
        enabled=True,
        output_root=str(tmp_path / "capture"),
        sample_manifest=str(sample_manifest),
        sample_manifest_sha256="0" * 64,
        expected_questions=1,
    )
    with pytest.raises(ValueError, match="sample manifest hash"):
        validate_capture_contract(wrong)


def test_recursive_parameter_hash_changes_with_any_parameter_byte():
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    before = recursive_parameter_sha256(model)
    with torch.no_grad():
        model[0].weight[0, 0].add_(1)
    after = recursive_parameter_sha256(model)
    assert len(before) == 64
    assert before != after


def _keys(count: int = 5) -> tuple[TrajectoryKey, ...]:
    return tuple(TrajectoryKey(f"q{index // 4}", 42, index % 4) for index in range(count))


def _rows(keys: tuple[TrajectoryKey, ...], offset: int = 0):
    return [
        {
            "stable_id": key.stable_id,
            "engine_seed": key.engine_seed,
            "rollout_slot": key.rollout_slot,
            "split": "candidate",
            "manifest_index": offset + index,
        }
        for index, key in enumerate(keys)
    ]


def _tensors(count: int, offset: int = 0):
    return {
        "input_ids": torch.arange(offset * 3, (offset + count) * 3).reshape(count, 3),
        "current_log_prob": torch.arange(
            offset * 2, (offset + count) * 2, dtype=torch.float32
        ).reshape(count, 2),
        "bf16_value": torch.ones((count, 1), dtype=torch.bfloat16),
    }


PARENT_HASHES = {
    "sample_manifest_sha256": "a" * 64,
    "source_snapshot_sha256": "b" * 64,
}


def test_tensor_writer_serializes_only_bounded_chunks(monkeypatch, tmp_path: Path):
    calls = []
    original = capture.save_safetensors

    def recording_save(tensors):
        calls.append(next(iter(tensors.values())).shape[0])
        return original(tensors)

    monkeypatch.setattr(capture, "save_safetensors", recording_save)
    keys = _keys()
    write_tensor_chunks_atomic(
        tmp_path,
        tensors=_tensors(len(keys)),
        sidecar_rows=_rows(keys),
        parent_hashes=PARENT_HASHES,
        chunk_size=2,
    )
    assert calls == [2, 2, 1]


def test_tensor_chunks_have_exact_sidecar_rows_and_round_trip(tmp_path: Path):
    keys = _keys()
    records = write_tensor_chunks_atomic(
        tmp_path,
        tensors=_tensors(len(keys)),
        sidecar_rows=_rows(keys),
        parent_hashes=PARENT_HASHES,
        chunk_size=2,
    )
    assert [(row["start"], row["end"]) for row in records] == [
        (0, 2),
        (2, 4),
        (4, 5),
    ]
    for record in records:
        tensor_path = tmp_path / record["tensor_file"]
        sidecar_path = tmp_path / record["sidecar_file"]
        tensors = load_file(tensor_path)
        rows = [json.loads(line) for line in sidecar_path.read_text().splitlines()]
        assert len(rows) == record["end"] - record["start"]
        assert all(value.shape[0] == len(rows) for value in tensors.values())
        assert all(row["tensor_sha256"] == sha256_file(tensor_path) for row in rows)

    assert resolve_capture_resume_prefix(
        tmp_path,
        expected_keys=keys,
        parent_hashes=PARENT_HASHES,
        required_tensor_names=set(_tensors(1)),
    ) == len(keys)
    loaded = load_and_join_capture_chunks(
        tmp_path,
        expected_keys=keys,
        parent_hashes=PARENT_HASHES,
        required_tensor_names=set(_tensors(1)),
    )
    assert loaded.keys == keys
    torch.testing.assert_close(loaded.tensors["input_ids"], _tensors(5)["input_ids"])
    assert loaded.tensors["bf16_value"].dtype == torch.bfloat16


def test_complete_marker_binds_the_exact_validated_chunk_records(tmp_path: Path):
    keys = _keys(2)
    write_tensor_chunks_atomic(
        tmp_path,
        tensors=_tensors(2),
        sidecar_rows=_rows(keys),
        parent_hashes=PARENT_HASHES,
        chunk_size=2,
    )
    (tmp_path / "COMPLETE.json").write_bytes(
        canonical_json_bytes(
            {
                "schema_version": 1,
                "artifact_type": "opd_proxy_capture_subtree_complete",
                "trajectory_count": 2,
                "trajectory_keys_sha256": trajectory_keys_sha256(keys),
                "parent_hashes": PARENT_HASHES,
                "chunk_count": 0,
                "chunks": [],
            }
        )
    )
    with pytest.raises(ValueError, match="completion chunk records"):
        resolve_capture_resume_prefix(
            tmp_path, expected_keys=keys, parent_hashes=PARENT_HASHES
        )


def test_resume_rejects_changed_source_hash_and_noncontiguous_chunks(tmp_path: Path):
    keys = _keys(4)
    prefix_dir = tmp_path / "prefix"
    write_tensor_chunks_atomic(
        prefix_dir,
        tensors=_tensors(2),
        sidecar_rows=_rows(keys[:2]),
        parent_hashes=PARENT_HASHES,
        chunk_size=2,
    )
    assert resolve_capture_resume_prefix(
        prefix_dir, expected_keys=keys, parent_hashes=PARENT_HASHES
    ) == 2
    with pytest.raises(ValueError, match="parent hashes"):
        resolve_capture_resume_prefix(
            prefix_dir,
            expected_keys=keys,
            parent_hashes={**PARENT_HASHES, "source_snapshot_sha256": "c" * 64},
        )

    write_tensor_chunks_atomic(
        prefix_dir,
        tensors=_tensors(1, offset=3),
        sidecar_rows=_rows(keys[3:], offset=3),
        parent_hashes=PARENT_HASHES,
        chunk_size=1,
        start_index=3,
    )
    with pytest.raises(ValueError, match="contiguous manifest-order prefix"):
        resolve_capture_resume_prefix(
            prefix_dir, expected_keys=keys, parent_hashes=PARENT_HASHES
        )


def test_resume_rejects_sidecar_tensor_row_count_mismatch(tmp_path: Path):
    keys = _keys(2)
    write_tensor_chunks_atomic(
        tmp_path,
        tensors=_tensors(2),
        sidecar_rows=_rows(keys),
        parent_hashes=PARENT_HASHES,
        chunk_size=2,
    )
    sidecar = tmp_path / "chunk_0_2.jsonl"
    original_lines = sidecar.read_text().splitlines()
    sidecar.write_text(original_lines[0] + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar row count"):
        resolve_capture_resume_prefix(
            tmp_path, expected_keys=keys, parent_hashes=PARENT_HASHES
        )

    duplicate_key_line = '{"stable_id":"duplicate",' + original_lines[0][1:]
    sidecar.write_text(
        duplicate_key_line + "\n" + original_lines[1] + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        resolve_capture_resume_prefix(
            tmp_path, expected_keys=keys, parent_hashes=PARENT_HASHES
        )


def _seed_manifest(keys: tuple[TrajectoryKey, ...]):
    prompt_ids = list(dict.fromkeys(key.stable_id for key in keys))
    return {
        "vllm_version": "0.8.5",
        "engine_args": {"seed": 42, "tensor_parallel_size": 1},
        "sampling_args": {"n": 4, "temperature": 1.0},
        "ordered_prompt_keys_sha256": sha256_id_lines(prompt_ids),
        "returned_compound_keys_sha256": trajectory_keys_sha256(keys),
        "model_hashes": {"student": "1" * 64, "teacher": "2" * 64},
        "tokenizer_hashes": {"student": "3" * 64, "teacher": "4" * 64},
        "config_hashes": {"capture": "5" * 64},
        "source_hashes": {"snapshot": "6" * 64},
        "engine_count": 1,
        "generation_call_count": 1,
    }


def test_finalize_rejects_compound_keys_without_four_slots_per_question(
    tmp_path: Path,
):
    malformed = (
        TrajectoryKey("q0", 42, 0),
        TrajectoryKey("q0", 42, 1),
        TrajectoryKey("q1", 42, 2),
        TrajectoryKey("q1", 42, 3),
    )
    with pytest.raises(ValueError, match="exact rollout slots"):
        finalize_capture_seed(
            tmp_path,
            expected_keys=malformed,
            actor_rank_expected_keys={0: malformed},
            parent_hashes=PARENT_HASHES,
            actor_parameter_sha256_before="d" * 64,
            actor_parameter_sha256_after="d" * 64,
            seed_manifest=_seed_manifest(malformed),
        )


def test_finalize_requires_exact_coverage_one_call_and_unchanged_actor(
    monkeypatch, tmp_path: Path
):
    keys = _keys(4)
    for subtree in ("rollout", "trainer_boundary"):
        write_tensor_chunks_atomic(
            tmp_path / subtree,
            tensors=_tensors(4),
            sidecar_rows=_rows(keys),
            parent_hashes=PARENT_HASHES,
            chunk_size=2,
        )
    rank_keys = {0: keys[:2], 1: keys[2:]}
    for rank, local_keys in rank_keys.items():
        write_tensor_chunks_atomic(
            tmp_path / f"actor/rank_{rank}",
            tensors=_tensors(len(local_keys), offset=rank * 2),
            sidecar_rows=_rows(local_keys, offset=rank * 2),
            parent_hashes=PARENT_HASHES,
            chunk_size=2,
        )

    def forbidden_full_join(*args, **kwargs):
        pytest.fail("finalization loaded and joined a full capture subtree")

    monkeypatch.setattr(capture, "load_and_join_capture_chunks", forbidden_full_join)
    digest = "d" * 64
    manifest = finalize_capture_seed(
        tmp_path,
        expected_keys=keys,
        actor_rank_expected_keys=rank_keys,
        parent_hashes=PARENT_HASHES,
        actor_parameter_sha256_before=digest,
        actor_parameter_sha256_after=digest,
        seed_manifest=_seed_manifest(keys),
    )
    assert manifest["generation_call_count"] == 1
    assert manifest["expected_trajectory_count"] == 4
    assert (tmp_path / "rollout/COMPLETE.json").is_file()
    assert (tmp_path / "trainer_boundary/COMPLETE.json").is_file()
    assert (tmp_path / "actor/rank_0/COMPLETE.json").is_file()
    assert (tmp_path / "COMPLETE.json").is_file()

    with pytest.raises(ValueError, match="completed capture subtree"):
        write_tensor_chunks_atomic(
            tmp_path / "rollout",
            tensors=_tensors(1, offset=4),
            sidecar_rows=_rows((TrajectoryKey("q1", 42, 0),), offset=4),
            parent_hashes=PARENT_HASHES,
            chunk_size=1,
            start_index=4,
        )

    with pytest.raises(ValueError, match="actor parameter hash changed"):
        finalize_capture_seed(
            tmp_path,
            expected_keys=keys,
            actor_rank_expected_keys=rank_keys,
            parent_hashes=PARENT_HASHES,
            actor_parameter_sha256_before=digest,
            actor_parameter_sha256_after="e" * 64,
            seed_manifest=_seed_manifest(keys),
        )
    bad_manifest = {**_seed_manifest(keys), "generation_call_count": 2}
    with pytest.raises(ValueError, match="exactly one generation call"):
        finalize_capture_seed(
            tmp_path,
            expected_keys=keys,
            actor_rank_expected_keys=rank_keys,
            parent_hashes=PARENT_HASHES,
            actor_parameter_sha256_before=digest,
            actor_parameter_sha256_after=digest,
            seed_manifest=bad_manifest,
        )
