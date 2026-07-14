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

from pathlib import Path

from hydra import compose, initialize_config_dir

from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import ActorConfig, RolloutConfig

_CONFIG_DIR = Path(__file__).resolve().parents[3] / "verl" / "trainer" / "config"


def test_capture_config_defaults_are_disabled_and_rollout_seed_is_fixed():
    assert RolloutConfig(name="vllm").seed == 0
    actor = ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
    )
    assert actor.opd_proxy_verify_capture_only is False


def test_rollout_yaml_materializes_seed_default():
    with initialize_config_dir(config_dir=str(_CONFIG_DIR / "rollout")):
        cfg = compose(config_name="rollout", overrides=["name=vllm"])
    rollout = omega_conf_to_dataclass(cfg)
    assert isinstance(rollout, RolloutConfig)
    assert rollout.seed == 0


def test_actor_yaml_materializes_capture_only_default():
    with initialize_config_dir(config_dir=str(_CONFIG_DIR / "actor")):
        cfg = compose(
            config_name="actor",
            overrides=["strategy=fsdp", "ppo_micro_batch_size_per_gpu=1"],
        )
    actor = omega_conf_to_dataclass(cfg)
    assert isinstance(actor, ActorConfig)
    assert actor.opd_proxy_verify_capture_only is False
