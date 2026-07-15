#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/home/mchen/miniconda3/envs/verl/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "missing required VERL interpreter: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "capture requires an active Slurm job" >&2
  exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "capture requires allocation-owned visible GPU tokens" >&2
  exit 2
fi
IFS=',' read -r -a visible_tokens <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#visible_tokens[@]}" -ne 4 ]]; then
  echo "capture requires exactly four visible allocation tokens" >&2
  exit 2
fi
declare -A unique_tokens=()
for token in "${visible_tokens[@]}"; do
  if [[ -z "${token}" || -n "${unique_tokens[${token}]:-}" ]]; then
    echo "capture visible allocation tokens must be nonempty and unique" >&2
    exit 2
  fi
  unique_tokens["${token}"]=1
done
if [[ "$(tmux display-message -p '#S')" != "opd-CLI" ]]; then
  echo "capture must run inside the opd-CLI tmux session" >&2
  exit 2
fi

stage_directory=""
stage=""
pair=""
engine_seed=""
output_root=""
declare -A seen=()
while (($#)); do
  option="$1"
  shift
  case "${option}" in
    --stage-directory|--stage|--pair|--engine-seed|--output-root)
      if [[ -n "${seen[${option}]:-}" ]]; then
        echo "duplicate capture argument: ${option}" >&2
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
        --stage-directory) stage_directory="${value}" ;;
        --stage) stage="${value}" ;;
        --pair) pair="${value}" ;;
        --engine-seed) engine_seed="${value}" ;;
        --output-root) output_root="${value}" ;;
      esac
      ;;
    *)
      echo "unknown capture argument: ${option}" >&2
      exit 2
      ;;
  esac
done

for required in stage_directory stage pair engine_seed output_root; do
  if [[ -z "${!required}" ]]; then
    echo "missing required capture argument: ${required}" >&2
    exit 2
  fi
done
if [[ "${stage}" == "efficacy_pilot" ]]; then
  :
elif [[ ! "${stage}" =~ ^[012]$ ]]; then
  echo "capture stage must be 0, 1, 2, or efficacy_pilot" >&2
  exit 2
fi
if [[ "${pair}" != "target" && "${pair}" != "proxy" ]]; then
  echo "capture pair must be target or proxy" >&2
  exit 2
fi
if [[ "${engine_seed}" != "42" && "${engine_seed}" != "43" ]]; then
  echo "capture engine seed must be 42 or 43" >&2
  exit 2
fi
if [[ "${stage}" == "efficacy_pilot" && "${engine_seed}" != "42" ]]; then
  echo "efficacy_pilot capture engine seed must be 42" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"
export PYTHONPATH="verl:.${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" -m math_eval.run_opd_proxy_gradient_verify \
  capture-work-unit \
  --stage-directory "${stage_directory}" \
  --stage "${stage}" \
  --pair "${pair}" \
  --engine-seed "${engine_seed}" \
  --output-root "${output_root}"
