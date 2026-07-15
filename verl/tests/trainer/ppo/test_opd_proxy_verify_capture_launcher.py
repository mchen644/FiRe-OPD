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

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from math_eval.opd_proxy_gradient_verify_artifacts import (
    TrajectoryKey,
    canonical_json_bytes,
    sha256_file,
)
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.main_ppo import validate_opd_proxy_capture_runtime_config
from verl.trainer.ppo import ray_trainer as trainer_module
from verl.trainer.ppo.opd_proxy_verify_capture import (
    attach_and_validate_keys,
    recursive_parameter_sha256,
)
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    _validate_capture_batch_forbidden_keys,
)
from verl.trainer.ppo.ref_input_utils import prepare_ref_model_inputs
from verl.utils import config as config_module
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.config import ActorConfig, PolicyLossConfig
from verl.workers.fsdp_workers import (
    ActorRolloutRefWorker,
    _resolve_actor_optim_config,
)


def test_capture_shell_launcher_declares_first_class_efficacy_pilot_stage():
    script = Path("verl/examples/fire_opd/run_capture_opd_proxy_verify.sh")
    text = script.read_text(encoding="utf-8")
    assert "efficacy_pilot" in text
    assert '"${stage}" == "efficacy_pilot"' in text
    assert "srun" not in text


PARENTS = {
    "sample_manifest_sha256": "a" * 64,
    "resolved_config_sha256": "b" * 64,
}


def _capture_config(tmp_path, *, enabled=True):
    sample_manifest = tmp_path / "sample_manifest.jsonl"
    sample_manifest.write_text(
        '{"stable_id":"q0","split":"candidate"}\n'
        '{"stable_id":"q1","split":"held_out"}\n',
        encoding="utf-8",
    )
    source_snapshot = tmp_path / "source_snapshot.json"
    source_files = [{"path": "module.py", "sha256": "9" * 64}]
    source_hash = hashlib.sha256(canonical_json_bytes(source_files)).hexdigest()
    source_snapshot.write_bytes(
        canonical_json_bytes(
            {"files": source_files, "manifest_sha256": source_hash}
        )
    )
    return OmegaConf.create(
        {
            "data": {
                "max_prompt_length": 2048,
                "max_response_length": 16384,
                "truncation": "error",
                "shuffle": False,
            },
            "trainer": {
                "balance_batch": False,
                "n_gpus_per_node": 4,
                "nnodes": 1,
            },
            "actor_rollout_ref": {
                "rollout": {
                    "name": "vllm",
                    "mode": "sync",
                    "n": 4,
                    "seed": 42,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "calculate_log_probs": True,
                },
                "actor": {
                    "opd_proxy_verify_capture_only": True,
                    "ppo_epochs": 1,
                    "ppo_mini_batch_size": 2,
                    "ppo_micro_batch_size_per_gpu": 1,
                    "use_dynamic_bsz": False,
                    "loss_agg_mode": "token-mean",
                    "entropy_coeff": 0.0,
                    "use_kl_loss": True,
                    "kl_loss_coef": 0.0,
                    "policy_loss": {
                        "loss_mode": "vanilla",
                        "only_reverse_kl_advantages": True,
                        "length_aware_opd": False,
                        "entropy_aware_distill": False,
                    },
                },
            },
            "algorithm": {
                "use_kl_in_reward": False,
                "candidate_selection": {"enabled": False},
                "tale_budget": {"enabled": False},
                "difficulty_aware_opd": {"enabled": False},
                "rethinking_opd_probe": {"enabled": False},
                "rollout_correction": {
                    "rollout_is": "token",
                    "rollout_is_threshold": 5.0,
                    "rollout_is_batch_normalize": False,
                    "rollout_rs": None,
                    "bypass_mode": False,
                },
                "opd_proxy_verify_capture": {
                    "enabled": enabled,
                    "output_root": str(tmp_path / "capture"),
                    "sample_manifest": str(sample_manifest),
                    "sample_manifest_sha256": sha256_file(sample_manifest),
                    "source_snapshot": str(source_snapshot),
                    "source_snapshot_sha256": source_hash,
                    "stage": 0,
                    "pair": "target",
                    "engine_seed": 42,
                    "native_rollouts": 4,
                    "expected_questions": 2,
                    "chunk_size": 2,
                    "schema_version": 1,
                },
            },
        }
    )


def _base_batch() -> DataProto:
    raw = np.empty(2, dtype=object)
    raw[0] = [{"role": "user", "content": "question zero"}]
    raw[1] = [{"role": "user", "content": "question one"}]
    return DataProto(
        batch=TensorDict(
            {
                "input_ids": torch.tensor([[0, 10], [0, 20]]),
                "attention_mask": torch.tensor([[0, 1], [0, 1]]),
                "position_ids": torch.tensor([[0, 0], [0, 0]]),
            },
            batch_size=2,
        ),
        non_tensor_batch={
            "opd_verify_stable_id": np.array(["q0", "q1"], dtype=object),
            "opd_verify_split": np.array(["candidate", "held_out"], dtype=object),
            "opd_verify_manifest_index": np.array([0, 1], dtype=object),
            "raw_prompt": raw,
        },
    )


def _generated_batch() -> DataProto:
    stable_ids = np.repeat(np.array(["q0", "q1"], dtype=object), 4)
    splits = np.repeat(np.array(["candidate", "held_out"], dtype=object), 4)
    indices = np.repeat(np.array([0, 1], dtype=object), 4)
    prompts = torch.tensor([[0, 10]] * 4 + [[0, 20]] * 4)
    responses = torch.tensor(
        [[30, 2], [31, 2], [32, 2], [33, 2], [40, 2], [41, 2], [42, 2], [43, 2]]
    )
    return DataProto(
        batch=TensorDict(
            {
                "prompts": prompts,
                "responses": responses,
                "input_ids": torch.cat([prompts, responses], dim=1),
                "attention_mask": torch.ones((8, 4), dtype=torch.long),
                "position_ids": torch.arange(4).repeat(8, 1),
                "rollout_log_probs": torch.full((8, 2), -1.0),
                "opd_proxy_verify_rollout_slot": torch.tensor([0, 1, 2, 3] * 2),
            },
            batch_size=8,
        ),
        non_tensor_batch={
            "opd_verify_stable_id": stable_ids,
            "opd_verify_split": splits,
            "opd_verify_manifest_index": indices,
        },
    )


class Recorder:
    def __init__(self, calls, name, result):
        self.calls = calls
        self.name = name
        self.result = result
        self.call_count = 0

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        self.calls.append(self.name)
        return self.result


def test_ref_retokenization_uses_the_declared_raw_prompt_key():
    messages = np.empty(1, dtype=object)
    messages[0] = [{"role": "user", "content": "question"}]
    batch = DataProto(
        batch=TensorDict(
            {
                "input_ids": torch.tensor([[10, 20, 30, 0]]),
                "responses": torch.tensor([[30, 0]]),
                "attention_mask": torch.tensor([[1, 1, 1, 0]]),
                "position_ids": torch.tensor([[0, 1, 2, 0]]),
                "response_mask": torch.tensor([[1, 0]]),
            },
            batch_size=1,
        ),
        non_tensor_batch={"teacher_prompt": messages},
    )

    class Tokenizer:
        pad_token_id = 0

        def apply_chat_template(self, value, **kwargs):
            assert value == messages[0]
            assert kwargs["add_generation_prompt"] is True
            return "rendered prompt"

        def __call__(self, value, **kwargs):
            assert value == "rendered prompt"
            return {"input_ids": torch.tensor([[7, 8]])}

    result = prepare_ref_model_inputs(
        batch,
        Tokenizer(),
        apply_chat_template_kwargs={"enable_thinking": False},
        raw_prompt_key="teacher_prompt",
    )
    assert result.batch["ref_input_ids"].tolist() == [[7, 8, 30, 0]]
    assert result.batch["ref_attention_mask"].tolist() == [[1, 1, 1, 0]]


def test_completed_rollout_subtree_is_reused_and_incomplete_staging_blocks(tmp_path):
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = _capture_config(tmp_path)
    batch = _generated_batch()
    batch.batch["response_mask"] = torch.ones_like(
        batch.batch["responses"], dtype=torch.bool
    )
    batch, keys = attach_and_validate_keys(
        batch,
        engine_seed=42,
        native_rollouts=4,
        expected_keys=tuple(
            TrajectoryKey(stable_id, 42, slot)
            for stable_id in ("q0", "q1")
            for slot in range(4)
        ),
    )
    contract = {
        "output_root": str(tmp_path / "capture"),
        "chunk_size": 3,
        "native_rollouts": 4,
        "engine_seed": 42,
    }
    parent_hashes = {
        "sample_manifest_sha256": "a" * 64,
        "resolved_config_sha256": "b" * 64,
    }
    provenance = trainer._publish_opd_proxy_rollout_capture(
        batch, keys, keys, contract, parent_hashes
    )

    loaded = trainer._load_completed_opd_proxy_rollout(
        contract, parent_hashes, keys
    )
    assert loaded is not None
    loaded_batch, loaded_keys, loaded_provenance = loaded
    assert loaded_keys == keys
    assert loaded_provenance == provenance
    torch.testing.assert_close(
        loaded_batch.batch["rollout_log_probs"], batch.batch["rollout_log_probs"]
    )
    assert loaded_batch.non_tensor_batch["uid"].tolist() == [
        "q0",
        "q0",
        "q0",
        "q0",
        "q1",
        "q1",
        "q1",
        "q1",
    ]

    blocked_contract = {
        **contract,
        "output_root": str(tmp_path / "blocked"),
    }
    (tmp_path / "blocked/.rollout.tmp").mkdir(parents=True)
    with pytest.raises(ValueError, match="incomplete rollout-generation subtree"):
        trainer._load_completed_opd_proxy_rollout(
            blocked_contract, parent_hashes, keys
        )


def test_capture_parent_hashes_resolve_stage_model_tokenizer_and_source(tmp_path):
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = _capture_config(tmp_path)
    root_manifest = {
        "model_manifests": {
            "target_student": {"manifest_sha256": "1" * 64},
            "target_teacher": {"manifest_sha256": "2" * 64},
        }
    }
    stage_manifest = {
        "provenance": {
            "root_manifest": root_manifest,
            "source_snapshot": {"manifest_sha256": "3" * 64},
            "tokenizer_compatibility": {
                "target": {"vocab_sha256": "4" * 64}
            },
        }
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps(stage_manifest), encoding="utf-8"
    )
    contract = trainer.config.algorithm.opd_proxy_verify_capture

    parents = trainer._opd_proxy_capture_parent_hashes(contract)

    assert parents["student_model_sha256"] == "1" * 64
    assert parents["teacher_model_sha256"] == "2" * 64
    assert parents["source_snapshot_sha256"] == (
        trainer.config.algorithm.opd_proxy_verify_capture.source_snapshot_sha256
    )
    assert parents["preparation_source_snapshot_sha256"] == "3" * 64
    assert parents["tokenizer_vocab_sha256"] == "4" * 64
    assert trainer._opd_proxy_capture_provenance_hashes["model_hashes"] == {
        "student": "1" * 64,
        "teacher": "2" * 64,
    }


def test_capture_finalizer_binds_rank_hashes_and_generation_provenance(
    monkeypatch, tmp_path
):
    import verl.trainer.ppo.ray_trainer as trainer_module

    trainer = object.__new__(RayPPOTrainer)
    trainer.config = _capture_config(tmp_path)
    trainer._opd_proxy_capture_provenance_hashes = {
        "model_hashes": {"student": "1" * 64, "teacher": "2" * 64},
        "tokenizer_hashes": {"student": "3" * 64, "teacher": "3" * 64},
        "source_hashes": {"snapshot": "4" * 64},
    }
    keys = tuple(
        TrajectoryKey(stable_id, 42, slot)
        for stable_id in ("q0", "q1")
        for slot in range(4)
    )
    before = ["a" * 64] * 4 + ["b" * 64] * 4
    actor_output = DataProto(
        batch=TensorDict(
            {
                "opd_proxy_verify_rollout_slot": torch.tensor([0, 1, 2, 3] * 2),
                "opd_proxy_verify_engine_seed": torch.full((8,), 42),
                "opd_proxy_verify_actor_rank": torch.tensor([0] * 4 + [1] * 4),
            },
            batch_size=8,
        ),
        non_tensor_batch={
            "opd_verify_stable_id": np.repeat(
                np.array(["q0", "q1"], dtype=object), 4
            ),
            "opd_proxy_verify_parameter_sha256_before": np.asarray(
                before, dtype=object
            ),
            "opd_proxy_verify_parameter_sha256_after": np.asarray(
                before, dtype=object
            ),
        },
    )
    seen = {}

    def fake_finalize(root, **kwargs):
        seen["root"] = root
        seen.update(kwargs)
        return {"complete": True}

    monkeypatch.setattr(trainer_module, "finalize_capture_seed", fake_finalize)
    parent_hashes = {
        "sample_manifest_sha256": "5" * 64,
        "resolved_config_sha256": "6" * 64,
    }
    rollout_provenance = {
        "vllm_version": "test-vllm",
        "engine_args": {"seed": 42},
        "sampling_args": {"n": 4, "temperature": 1.0, "top_p": 1.0},
        "ordered_prompt_keys_sha256": "7" * 64,
        "returned_compound_keys_sha256": "8" * 64,
        "engine_count": 1,
        "generation_call_count": 1,
    }

    result = trainer._finalize_opd_proxy_capture(
        actor_output,
        keys,
        _capture_config(tmp_path).algorithm.opd_proxy_verify_capture,
        parent_hashes,
        rollout_provenance,
    )

    assert result == {"complete": True}
    assert seen["native_rollouts"] == 4
    assert seen["expected_keys"] == keys
    assert seen["actor_rank_expected_keys"] == {
        0: list(keys[:4]),
        1: list(keys[4:]),
    }
    assert seen["actor_parameter_sha256_before"] == seen[
        "actor_parameter_sha256_after"
    ]
    assert seen["seed_manifest"]["vllm_version"] == "test-vllm"
    assert seen["seed_manifest"]["model_hashes"] == {
        "student": "1" * 64,
        "teacher": "2" * 64,
    }


def test_capture_only_config_validates_exact_nondivisible_batch_via_internal_padding(
    monkeypatch,
):
    config = OmegaConf.create(
        {
            "trainer": {"n_gpus_per_node": 4, "nnodes": 1},
            "data": {"train_batch_size": 334, "val_batch_size": None},
            "actor_rollout_ref": {
                "model": {},
                "actor": {
                    "use_dynamic_bsz": False,
                    "strategy": "fsdp",
                    "ppo_mini_batch_size": 336,
                    "opd_proxy_verify_capture_only": True,
                },
                "rollout": {
                    "n": 1,
                    "log_prob_micro_batch_size": None,
                    "log_prob_micro_batch_size_per_gpu": 1,
                    "val_kwargs": {"do_sample": False},
                },
                "ref": {
                    "log_prob_micro_batch_size": None,
                    "log_prob_micro_batch_size_per_gpu": 1,
                },
            },
            "reward_model": {"enable": False},
            "algorithm": {"use_kl_in_reward": False},
        }
    )
    seen = {}

    class Actor:
        use_kl_loss = True

        def validate(self, n_gpus, train_batch_size, model_config):
            seen.update(
                n_gpus=n_gpus,
                train_batch_size=train_batch_size,
                model_config=model_config,
            )

    monkeypatch.setattr(config_module, "omega_conf_to_dataclass", lambda *args, **kwargs: Actor())
    config_module.validate_config(
        config, use_reference_policy=False, use_critic=False
    )
    assert seen["n_gpus"] == 4
    assert seen["train_batch_size"] == 336

    config.actor_rollout_ref.actor.opd_proxy_verify_capture_only = False
    with pytest.raises(AssertionError, match="334.*divisible.*4"):
        config_module.validate_config(
            config, use_reference_policy=False, use_critic=False
        )


def test_capture_worker_padding_is_internal_and_restores_exact_order():
    data = DataProto.from_dict(
        tensors={"value": torch.arange(5).reshape(5, 1)},
        non_tensors={"stable_id": np.array([f"q{i}" for i in range(5)], dtype=object)},
    )

    class WorkerGroup:
        world_size = 4

        @staticmethod
        def compute(batch):
            assert len(batch) == 8
            assert batch.non_tensor_batch["opd_proxy_verify_padding"].tolist() == (
                [False] * 5 + [True] * 3
            )
            return batch

    output = trainer_module._call_capture_worker_with_padding(
        WorkerGroup(), "compute", data
    )
    assert len(data) == 5
    assert len(output) == 5
    assert output.batch["value"].flatten().tolist() == list(range(5))
    assert output.non_tensor_batch["stable_id"].tolist() == [f"q{i}" for i in range(5)]
    assert "opd_proxy_verify_padding" not in output.non_tensor_batch


def test_capture_path_calls_one_generation_and_never_updates(monkeypatch, tmp_path):
    config = _capture_config(tmp_path)
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = config
    trainer.tokenizer = SimpleNamespace(eos_token_id=2, pad_token_id=0)
    trainer.use_reference_policy = True
    trainer.ref_in_actor = False
    trainer.use_ref_retokenization = False
    trainer.async_rollout_mode = False
    trainer._load_ordered_opd_proxy_capture_batch = lambda contract: _base_batch()

    calls = []
    generated = _generated_batch()
    old = DataProto.from_dict(
        tensors={
            "old_log_probs": torch.full((8, 2), -0.8),
            "entropys": torch.zeros((8, 2)),
        }
    )
    ref = DataProto.from_dict(tensors={"ref_log_prob": torch.full((8, 2), -0.5)})
    actor_result = DataProto(
        batch=TensorDict(
            {"opd_proxy_verify_actor_rank": torch.zeros(8, dtype=torch.long)},
            batch_size=8,
        ),
        non_tensor_batch={
            "opd_verify_stable_id": generated.non_tensor_batch[
                "opd_verify_stable_id"
            ],
            "opd_proxy_verify_parameter_sha256_before": np.array(
                ["d" * 64] * 8, dtype=object
            ),
            "opd_proxy_verify_parameter_sha256_after": np.array(
                ["d" * 64] * 8, dtype=object
            ),
        },
    )
    trainer.actor_rollout_wg = SimpleNamespace(
        generate_sequences=Recorder(calls, "generate", generated),
        compute_log_prob=Recorder(calls, "batch_old", old),
        capture_opd_proxy_verify=Recorder(calls, "actor_capture", actor_result),
        update_actor=lambda batch: pytest.fail("update_actor called"),
        world_size=1,
    )
    trainer.ref_policy_wg = SimpleNamespace(
        compute_ref_log_prob=Recorder(calls, "ref", ref),
        world_size=1,
    )
    monkeypatch.setattr(
        trainer, "_publish_opd_proxy_rollout_capture", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(
        trainer, "_publish_opd_proxy_trainer_boundary", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        trainer, "_finalize_opd_proxy_capture", lambda *args, **kwargs: {"done": True}
    )

    assert trainer.fit() == {"done": True}
    assert calls == ["generate", "batch_old", "ref", "actor_capture"]
    assert trainer.actor_rollout_wg.generate_sequences.call_count == 1


def test_capture_actor_resolves_optimizer_config_to_none():
    actor_config = OmegaConf.create(
        {"opd_proxy_verify_capture_only": True, "optim": {"lr": 1e-6}}
    )
    assert _resolve_actor_optim_config(actor_config) is None


def test_normal_actor_keeps_optimizer_config():
    actor_config = OmegaConf.create(
        {"opd_proxy_verify_capture_only": False, "optim": {"lr": 1e-6}}
    )
    assert _resolve_actor_optim_config(actor_config) is actor_config.optim


def test_capture_keys_survive_balance_select_and_rank_dispatch(monkeypatch):
    generated = _generated_batch()
    generated.batch["response_mask"] = torch.ones((8, 2), dtype=torch.bool)
    generated.batch["opd_proxy_verify_engine_seed"] = torch.full(
        (8,), 42, dtype=torch.long
    )
    expected = tuple(
        TrajectoryKey(stable_id, 42, slot)
        for stable_id in ("q0", "q1")
        for slot in range(4)
    )
    _, before = attach_and_validate_keys(
        generated,
        engine_seed=42,
        native_rollouts=4,
        expected_keys=expected,
    )

    trainer = object.__new__(RayPPOTrainer)
    trainer.actor_rollout_wg = SimpleNamespace(world_size=2)
    trainer.config = OmegaConf.create(
        {"actor_rollout_ref": {"actor": {"ppo_mini_batch_size": 8}}}
    )
    monkeypatch.setattr(
        "verl.trainer.ppo.ray_trainer.log_seqlen_unbalance", lambda **kwargs: {}
    )
    trainer._balance_batch(generated, metrics={})
    _, after = attach_and_validate_keys(
        generated,
        engine_seed=42,
        native_rollouts=4,
        expected_keys=expected,
    )
    assert set(after) == set(before)

    selected = generated.select(
        batch_keys=list(generated.batch.keys()),
        non_tensor_batch_keys=[
            "opd_verify_stable_id",
            "opd_verify_split",
            "opd_verify_manifest_index",
        ],
    )
    dispatched_keys = []
    for rank_batch in (selected[:4], selected[4:]):
        _, local = attach_and_validate_keys(
            rank_batch,
            engine_seed=42,
            native_rollouts=4,
            require_complete_slots=False,
        )
        dispatched_keys.extend(local)
    assert set(dispatched_keys) == set(expected)


def _actor_capture_data(tmp_path):
    generated = _generated_batch()[:2]
    generated.batch["response_mask"] = torch.ones((2, 2))
    generated.batch["old_log_probs"] = torch.full((2, 2), -0.8)
    generated.batch["ref_log_prob"] = torch.full((2, 2), -0.5)
    generated.batch["rollout_is_weights"] = torch.full(
        (2, 2), float(torch.exp(torch.tensor(0.2)))
    )
    generated.batch["opd_proxy_verify_engine_seed"] = torch.full(
        (2,), 42, dtype=torch.long
    )
    generated.meta_info.update(
        {
            "temperature": 1.0,
            "opd_proxy_verify_output_root": str(tmp_path),
            "opd_proxy_verify_parent_hashes": PARENTS,
            "opd_proxy_verify_chunk_size": 1,
            "opd_proxy_verify_engine_seed": 42,
        }
    )
    return generated


def test_actor_capture_uses_micro_batch_one_and_never_backward_or_updates(
    monkeypatch, tmp_path
):
    import verl.workers.actor.dp_actor as actor_module

    actor = object.__new__(DataParallelPPOActor)
    actor.config = ActorConfig(
        strategy="fsdp",
        rollout_n=4,
        ppo_mini_batch_size=2,
        ppo_micro_batch_size_per_gpu=1,
        ppo_epochs=1,
        use_dynamic_bsz=False,
        opd_proxy_verify_capture_only=True,
        use_torch_compile=False,
        use_kl_loss=True,
        kl_loss_coef=0.0,
        entropy_coeff=0.0,
        policy_loss=PolicyLossConfig(
            loss_mode="vanilla", only_reverse_kl_advantages=True
        ),
    )
    actor.actor_module = torch.nn.Linear(1, 1, bias=False)
    actor.actor_optimizer = None
    calls = []

    def fake_forward(model_inputs, temperature, calculate_entropy=False):
        calls.append(model_inputs["responses"].shape[0])
        current = actor.actor_module.weight.sum() * torch.ones_like(
            model_inputs["old_log_probs"]
        )
        return None, current, {}

    actor._forward_micro_batch = fake_forward
    monkeypatch.setattr(actor_module, "get_device_id", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(
        torch.Tensor,
        "backward",
        lambda *args, **kwargs: pytest.fail("backward called"),
    )
    before = recursive_parameter_sha256(actor.actor_module)

    method = DataParallelPPOActor.capture_opd_proxy_verify
    while hasattr(method, "__wrapped__"):
        method = method.__wrapped__
    output = method(actor, _actor_capture_data(tmp_path))

    assert calls == [1, 1]
    assert actor.actor_module.training is True
    assert recursive_parameter_sha256(actor.actor_module) == before
    assert set(output.batch["opd_proxy_verify_actor_rank"].tolist()) == {0}
    assert set(
        output.non_tensor_batch[
            "opd_proxy_verify_parameter_sha256_before"
        ].tolist()
    ) == {before}
    assert (tmp_path / "actor/rank_0/chunk_0_1.safetensors").is_file()
    assert (tmp_path / "actor/rank_0/chunk_1_2.safetensors").is_file()

    calls.clear()
    method(actor, _actor_capture_data(tmp_path))
    assert calls == []


def test_actor_capture_computes_but_never_publishes_internal_padding(
    monkeypatch, tmp_path
):
    import verl.workers.actor.dp_actor as actor_module

    actor = object.__new__(DataParallelPPOActor)
    actor.config = ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_mini_batch_size=2,
        ppo_micro_batch_size_per_gpu=1,
        ppo_epochs=1,
        use_dynamic_bsz=False,
        opd_proxy_verify_capture_only=True,
        use_torch_compile=False,
        use_kl_loss=True,
        kl_loss_coef=0.0,
        entropy_coeff=0.0,
        policy_loss=PolicyLossConfig(
            loss_mode="vanilla", only_reverse_kl_advantages=True
        ),
    )
    actor.actor_module = torch.nn.Linear(1, 1, bias=False)
    actor.actor_optimizer = None
    calls = []

    def fake_forward(model_inputs, temperature, calculate_entropy=False):
        del temperature, calculate_entropy
        calls.append(str(model_inputs["opd_verify_stable_id"][0]))
        current = actor.actor_module.weight.sum() * torch.ones_like(
            model_inputs["old_log_probs"]
        )
        return None, current, {}

    actor._forward_micro_batch = fake_forward
    observed_native_rollouts = []
    original_attach_and_validate_keys = actor_module.attach_and_validate_keys

    def recording_attach_and_validate_keys(*args, **kwargs):
        observed_native_rollouts.append(kwargs["native_rollouts"])
        return original_attach_and_validate_keys(*args, **kwargs)

    monkeypatch.setattr(
        actor_module,
        "attach_and_validate_keys",
        recording_attach_and_validate_keys,
    )
    monkeypatch.setattr(actor_module, "get_device_id", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 3)
    data = _actor_capture_data(tmp_path)
    data.batch["opd_proxy_verify_rollout_slot"][1] = 0
    data.non_tensor_batch["opd_verify_stable_id"][1] = "padding-q0"
    data.non_tensor_batch["opd_proxy_verify_padding"] = np.array(
        [False, True], dtype=np.bool_
    )

    method = DataParallelPPOActor.capture_opd_proxy_verify
    while hasattr(method, "__wrapped__"):
        method = method.__wrapped__
    output = method(actor, data)

    assert calls == ["q0", "padding-q0"]
    assert len(output) == 2
    assert (tmp_path / "actor/rank_3/chunk_0_1.safetensors").is_file()
    assert not (tmp_path / "actor/rank_3/chunk_1_2.safetensors").exists()
    sidecars = [
        json.loads(line)
        for line in (tmp_path / "actor/rank_3/chunk_0_1.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [row["stable_id"] for row in sidecars] == ["q0"]

    calls.clear()
    method(actor, data)
    assert calls == ["q0", "padding-q0"]
    assert observed_native_rollouts == [1, 1]
    assert not (tmp_path / "actor/rank_3/chunk_1_2.safetensors").exists()


def test_efficacy_pilot_runtime_contract_uses_native_n1(tmp_path):
    config = _capture_config(tmp_path)
    config.actor_rollout_ref.rollout.n = 1
    config.actor_rollout_ref.actor.ppo_mini_batch_size = 4
    capture = config.algorithm.opd_proxy_verify_capture
    capture.stage = "efficacy_pilot"
    capture.native_rollouts = 1
    capture.algorithm_contract_sha256 = "a" * 64

    contract = validate_opd_proxy_capture_runtime_config(config)

    assert contract["stage"] == "efficacy_pilot"
    assert contract["native_rollouts"] == 1
    assert contract["algorithm_contract_sha256"] == "a" * 64


def test_capture_contract_rejects_forbidden_config_and_batch_keys(tmp_path):
    config = _capture_config(tmp_path)
    validate_opd_proxy_capture_runtime_config(config)
    config.algorithm.candidate_selection.enabled = True
    with pytest.raises(ValueError, match="candidate_selection"):
        validate_opd_proxy_capture_runtime_config(config)

    batch = _actor_capture_data(tmp_path)
    batch.batch["candidate_selection_loss_mask"] = torch.ones(2)
    with pytest.raises(ValueError, match="forbidden capture batch keys"):
        _validate_capture_batch_forbidden_keys(batch)


def test_worker_exposes_capture_rpc_and_normal_fit_guard_is_default_off(
    monkeypatch, tmp_path
):
    assert hasattr(ActorRolloutRefWorker, "capture_opd_proxy_verify")
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = _capture_config(tmp_path, enabled=False)
    trainer.config.trainer.update(
        {
            "project_name": "test",
            "experiment_name": "test",
            "logger": ["console"],
        }
    )
    trainer._fit_opd_proxy_verify_capture = lambda: pytest.fail(
        "capture path entered while disabled"
    )
    monkeypatch.setattr(
        "verl.utils.tracking.Tracking", lambda **kwargs: SimpleNamespace(log=lambda **kwargs: None)
    )
    trainer._load_checkpoint = lambda: pytest.fail("normal fit body entered")
    with pytest.raises(pytest.fail.Exception, match="normal fit body entered"):
        trainer.fit()
