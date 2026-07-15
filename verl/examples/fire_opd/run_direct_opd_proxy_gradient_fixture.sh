#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/home/mchen/miniconda3/envs/gvendi-opd/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "missing required interpreter: ${PYTHON_BIN}" >&2
  exit 2
fi

stage=""
pair=""
engine_seed=""
rollout_slot=""
stable_id=""
direct_flag=""
capture_root=""
model_path=""
output_root=""
model_sha256=""
reference_repo=""
declare -A seen=()

while (($#)); do
  option="$1"
  shift
  case "${option}" in
    --stage|--pair|--engine-seed|--rollout-slot|--stable-id|\
    --capture-direct-gradient-fixture|--capture-root|--model-path|\
    --output-root|--expected-model-sha256|--reference-repo)
      if [[ -n "${seen[${option}]:-}" ]]; then
        echo "duplicate argument: ${option}" >&2
        exit 2
      fi
      seen["${option}"]=1
      if (($# == 0)); then
        echo "missing value for ${option}" >&2
        exit 2
      fi
      value="$1"
      shift
      case "${option}" in
        --stage) stage="${value}" ;;
        --pair) pair="${value}" ;;
        --engine-seed) engine_seed="${value}" ;;
        --rollout-slot) rollout_slot="${value}" ;;
        --stable-id) stable_id="${value}" ;;
        --capture-direct-gradient-fixture) direct_flag="${value}" ;;
        --capture-root) capture_root="${value}" ;;
        --model-path) model_path="${value}" ;;
        --output-root) output_root="${value}" ;;
        --expected-model-sha256) model_sha256="${value}" ;;
        --reference-repo) reference_repo="${value}" ;;
      esac
      ;;
    *)
      echo "unknown argument: ${option}" >&2
      exit 2
      ;;
  esac
done

for required in stage pair engine_seed rollout_slot stable_id direct_flag \
  capture_root model_path output_root model_sha256 reference_repo; do
  if [[ -z "${!required}" ]]; then
    echo "missing required direct-fixture argument: ${required}" >&2
    exit 2
  fi
done
if [[ "${stage}" != "0" || "${pair}" != "proxy" || \
      "${engine_seed}" != "42" || "${rollout_slot}" != "0" || \
      "${direct_flag}" != "true" ]]; then
  echo "direct fixture is frozen to stage=0 pair=proxy seed=42 slot=0 and explicit true flag" >&2
  exit 2
fi
if [[ ! "${model_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "expected model SHA-256 must be 64 lowercase hex characters" >&2
  exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "direct fixture requires an active Slurm job" >&2
  exit 2
fi
if [[ "$(tmux display-message -p '#S')" != "opd-CLI" ]]; then
  echo "direct fixture must run inside the opd-CLI tmux session" >&2
  exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
  echo "direct fixture requires exactly one process-local CUDA_VISIBLE_DEVICES token" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"
export PYTHONPATH="verl:.${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON_BIN}" - \
  "${capture_root}" "${model_path}" "${output_root}" "${model_sha256}" \
  "${reference_repo}" "${stable_id}" "${engine_seed}" "${rollout_slot}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import open_dict
from tensordict import TensorDict

from math_eval.opd_proxy_gradient_verify_artifacts import sha256_file
from math_eval.replay_opd_proxy_gradients import (
    load_capture_trajectories,
    load_replay_actor,
)
from verl import DataProto

(
    capture_root,
    model_path,
    output_root,
    expected_model_sha256,
    reference_repo,
    stable_id,
    engine_seed_text,
    rollout_slot_text,
) = sys.argv[1:]
engine_seed = int(engine_seed_text)
rollout_slot = int(rollout_slot_text)
root = Path(capture_root).resolve()
trajectories = load_capture_trajectories(root)
selected = [
    row
    for row in trajectories
    if row.stable_id == stable_id
    and row.engine_seed == engine_seed
    and row.rollout_slot == rollout_slot
]
if len(selected) != 1:
    raise ValueError(
        f"direct fixture requires exactly one matching trajectory, found {len(selected)}"
    )
trajectory = selected[0]
actor = load_replay_actor(
    Path(model_path),
    expected_model_sha256=expected_model_sha256,
)
with open_dict(actor.config):
    actor.config.opd_proxy_verify_capture_only = True
manifest_path = root / "manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
parent_hashes = dict(manifest["parent_hashes"])
parent_hashes["capture_manifest_sha256"] = sha256_file(manifest_path)
batch = DataProto(
    batch=TensorDict(
        {
            "input_ids": trajectory.input_ids,
            "responses": trajectory.responses,
            "attention_mask": trajectory.attention_mask,
            "position_ids": trajectory.position_ids,
            "response_mask": trajectory.response_mask,
            "rollout_log_probs": trajectory.rollout_log_prob,
            "old_log_probs": trajectory.batch_old_log_prob,
            "ref_log_prob": trajectory.ref_log_prob,
            "rollout_is_weights": trajectory.rollout_is_weights,
            "opd_proxy_verify_rollout_slot": torch.tensor([rollout_slot]),
            "opd_proxy_verify_engine_seed": torch.tensor([engine_seed]),
        },
        batch_size=1,
    ),
    non_tensor_batch={
        "opd_verify_stable_id": np.asarray([stable_id], dtype=object),
        "opd_verify_split": np.asarray([trajectory.split], dtype=object),
        "opd_verify_manifest_index": np.asarray([0], dtype=object),
    },
    meta_info={
        "temperature": 1.0,
        "opd_proxy_verify_capture_enabled": True,
        "opd_proxy_verify_direct_gradient_fixture": True,
        "opd_proxy_verify_direct_output_root": str(Path(output_root).resolve()),
        "opd_proxy_verify_parent_hashes": parent_hashes,
        "opd_proxy_verify_reference_repo": str(Path(reference_repo).resolve()),
    },
)
artifact = actor.capture_direct_opd_proxy_gradient_fixture(batch)
print(artifact["published_manifest"])
PY
