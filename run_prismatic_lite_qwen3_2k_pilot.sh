#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PRODUCTION_ROOT="/home/mchen/FiRe-OPD"
DATA_ROOT="${PRODUCTION_ROOT}/data/prismatic_lite/qwen3_2k_pilot"
LOG_ROOT="${PRODUCTION_ROOT}/logs/prismatic_lite/qwen3_2k_pilot"
FAILED_ATTEMPTS_ROOT="${PRODUCTION_ROOT}/logs/prismatic_lite/qwen3_2k_pilot.failed_attempts"
MASTER_LOG="${LOG_ROOT}/pilot.log"
MASTER_LOCK="${PRODUCTION_ROOT}/logs/prismatic_lite/.qwen3_2k_pilot.launch.lock"
GVENDI_PYTHON="/home/mchen/miniconda3/envs/gvendi-opd/bin/python"
VERL_PYTHON="/home/mchen/miniconda3/envs/verl/bin/python"
QWEN_MODEL="/home/mchen/FiRe-OPD/models/Qwen3-30B-A3B-Instruct-2507"
QWEN_MODEL_REVISION="0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
REFERENCE_REPO="/home/mchen/prismatic-synthesis-reference"
EXPECTED_TMUX_SESSION="opd-CLI"
REFERENCE_COMMIT="d9484cd3b5991030b901ac4a3a9e2472dbfac2ad"
SOURCE_JSONL="${PRODUCTION_ROOT}/data/gradient_diversity/deepmath_level6_r1_solution1.jsonl"
SOURCE_MANIFEST="${PRODUCTION_ROOT}/data/gradient_diversity/deepmath_level6_r1_solution1.manifest.json"
ELIGIBILITY_REPORT="${PRODUCTION_ROOT}/data/gradient_diversity/deepmath_level6_r1_solution1.eligibility.json"
ORIGINAL_GRADIENT_MANIFEST="${PRODUCTION_ROOT}/data/gradient_diversity/gradients/qwen2.5-0.5b-instruct/gradient.manifest.json"
SOURCE_JSONL_SHA256="ee8d55c943577888f14052b3953a0a8aa07d26076e50ddffa41a102b60027344"
SOURCE_MANIFEST_SHA256="5add2e965473d3647f2318c73f957434fbe23344516089be9adfe89db9c6a70e"
ELIGIBILITY_SHA256="cc8d5b888def37761519e15c6ef722e5a22d91ae2342c6d669f008bcff3dcbb6"
ORIGINAL_GRADIENT_MANIFEST_SHA256="9a534118a08e736a15e806933d5cf90c2a99822fee28bf18644e5ff344ce3ae2"
NO_GO_EXIT_CODE=20
REQUIRED_REMAINING_SECONDS=172800
ALLOCATION_TOKENS="0,1,2,3"

CALIBRATION_INPUT="${DATA_ROOT}/calibration/gradient_input.jsonl"
CALIBRATION_GRADIENT_DIR="${DATA_ROOT}/calibration/paired_gradients"
CANDIDATE_INPUT="${DATA_ROOT}/candidates/gradient_input.jsonl"
CANDIDATE_GRADIENT_DIR="${DATA_ROOT}/candidates/new_gradients"
PILOT_MANIFEST="${DATA_ROOT}/manifest.json"
STAGE_MARKER="${DATA_ROOT}/STAGE_COMPLETE.json"


die() {
  printf 'ERROR: %s\n' "$*" >&2
  printf 'A failed attempt must be archived under %s before any relaunch.\n' \
    "$FAILED_ATTEMPTS_ROOT" >&2
  exit 2
}

require_sha256() {
  local path="$1"
  local expected="$2"
  [[ -f "$path" ]] || die "required frozen file is missing: $path"
  local actual
  actual="$(sha256sum -- "$path" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || \
    die "SHA-256 mismatch for $path: expected $expected, got $actual"
}

require_clean_repository() {
  local repository="$1"
  local description="$2"
  [[ -d "$repository/.git" || -f "$repository/.git" ]] || \
    die "$description is not a Git worktree: $repository"
  local status
  status="$(git -C "$repository" status --porcelain=v1 --untracked-files=all)"
  [[ -z "$status" ]] || die "$description must be clean: $status"
}

verify_frozen_inputs() {
  [[ "$CODE_ROOT" == "/home/mchen/FiRe-OPD/.worktrees/vanilla-sft-gradient-51200-opd" ]] || \
    die "launcher must execute from the approved isolated worktree"
  [[ -x "$GVENDI_PYTHON" ]] || die "missing gvendi Python: $GVENDI_PYTHON"
  [[ -x "$VERL_PYTHON" ]] || die "missing VERL Python: $VERL_PYTHON"
  [[ -d "$QWEN_MODEL" ]] || die "missing Qwen3 model: $QWEN_MODEL"
  [[ -f "$QWEN_MODEL/config.json" ]] || die "missing Qwen3 config.json"
  [[ -f "$QWEN_MODEL/model.safetensors.index.json" ]] || \
    die "missing Qwen3 model index"
  require_sha256 "$SOURCE_JSONL" "$SOURCE_JSONL_SHA256"
  require_sha256 "$SOURCE_MANIFEST" "$SOURCE_MANIFEST_SHA256"
  require_sha256 "$ELIGIBILITY_REPORT" "$ELIGIBILITY_SHA256"
  require_sha256 "$ORIGINAL_GRADIENT_MANIFEST" \
    "$ORIGINAL_GRADIENT_MANIFEST_SHA256"
  require_clean_repository "$CODE_ROOT" "pilot code worktree"
  require_clean_repository "$REFERENCE_REPO" "Prismatic reference repository"
  local reference_head
  reference_head="$(git -C "$REFERENCE_REPO" rev-parse HEAD)"
  [[ "$reference_head" == "$REFERENCE_COMMIT" ]] || \
    die "reference commit mismatch: $reference_head"
  git -C "$CODE_ROOT" rev-parse HEAD >/dev/null
  git -C "$CODE_ROOT" rev-parse 'HEAD^{tree}' >/dev/null
  git -C "$REFERENCE_REPO" rev-parse 'HEAD^{tree}' >/dev/null
}

verify_model_tree() {
  "$GVENDI_PYTHON" - "$QWEN_MODEL" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
from math_eval.run_prismatic_lite_qwen3_pilot import (
    QWEN_MODEL_REVISION,
    _verify_qwen_model_revision,
    hash_directory,
)

root = Path(sys.argv[1])
_verify_qwen_model_revision(root)
records = hash_directory(root)
payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
print(json.dumps({
    "model_file_count": len(records),
    "model_revision": QWEN_MODEL_REVISION,
    "model_total_bytes": sum(int(row["size"]) for row in records),
    "model_tree_sha256": hashlib.sha256(payload).hexdigest(),
}, sort_keys=True))
PY
}

require_absent() {
  local path="$1"
  [[ ! -e "$path" && ! -L "$path" ]] || \
    die "canonical target already exists; overwrite is forbidden: $path"
}

require_absent_initial_targets() {
  require_absent "$DATA_ROOT"
  require_absent "$LOG_ROOT"
  require_absent "$MASTER_LOG"
}

pilot_command() {
  local python_bin="$1"
  shift
  "$python_bin" -m math_eval.run_prismatic_lite_qwen3_pilot \
    --root "$DATA_ROOT" "$@"
}

print_phase_commands() {
  local commands=(
    "$GVENDI_PYTHON -m math_eval.run_prismatic_lite_qwen3_pilot --root $DATA_ROOT prepare"
    "CUDA_VISIBLE_DEVICES=0,1,2,3 $VERL_PYTHON -m math_eval.run_prismatic_lite_qwen3_pilot --root $DATA_ROOT generate-calibration-solutions"
    "four process-local Qwen2.5 collectors for $CALIBRATION_INPUT -> $CALIBRATION_GRADIENT_DIR"
    "CUDA_VISIBLE_DEVICES=0 $GVENDI_PYTHON -m math_eval.run_prismatic_lite_qwen3_pilot --root $DATA_ROOT analyze-calibration --device cuda:0"
    "CUDA_VISIBLE_DEVICES=0,1,2,3 $VERL_PYTHON -m math_eval.run_prismatic_lite_qwen3_pilot --root $DATA_ROOT generate-problems"
    "CUDA_VISIBLE_DEVICES=0,1,2,3 $VERL_PYTHON -m math_eval.run_prismatic_lite_qwen3_pilot --root $DATA_ROOT generate-candidate-solutions"
    "$GVENDI_PYTHON -m math_eval.run_prismatic_lite_qwen3_pilot --root $DATA_ROOT quality-review"
    "four process-local Qwen2.5 collectors for $CANDIDATE_INPUT -> $CANDIDATE_GRADIENT_DIR"
    "CUDA_VISIBLE_DEVICES=0 $GVENDI_PYTHON -m math_eval.run_prismatic_lite_qwen3_pilot --root $DATA_ROOT analyze-selection --device cuda:0"
    "$GVENDI_PYTHON -m math_eval.run_prismatic_lite_qwen3_pilot --root $DATA_ROOT validate"
  )
  local command
  for command in "${commands[@]}"; do
    printf 'phase_command=%s\n' "$command"
  done
}

print_provenance() {
  printf 'code_root=%s\n' "$CODE_ROOT"
  printf 'code_head=%s\n' "$(git -C "$CODE_ROOT" rev-parse HEAD)"
  printf 'code_tree=%s\n' "$(git -C "$CODE_ROOT" rev-parse 'HEAD^{tree}')"
  printf 'reference_head=%s\n' "$(git -C "$REFERENCE_REPO" rev-parse HEAD)"
  printf 'reference_tree=%s\n' \
    "$(git -C "$REFERENCE_REPO" rev-parse 'HEAD^{tree}')"
  printf 'source_jsonl_sha256=%s\n' "$SOURCE_JSONL_SHA256"
  printf 'eligibility_sha256=%s\n' "$ELIGIBILITY_SHA256"
  printf 'original_gradient_manifest_sha256=%s\n' \
    "$ORIGINAL_GRADIENT_MANIFEST_SHA256"
  printf 'qwen_model_revision=%s\n' "$QWEN_MODEL_REVISION"
  printf 'required_tmux_session=%s\n' "$EXPECTED_TMUX_SESSION"
  printf 'data_root=%s\n' "$DATA_ROOT"
  printf 'log_root=%s\n' "$LOG_ROOT"
  printf 'failed_attempts_root=%s\n' "$FAILED_ATTEMPTS_ROOT"
  pilot_command "$GVENDI_PYTHON" estimate
}

validate_runtime_gate() {
  local allocated_tokens
  allocated_tokens="$($GVENDI_PYTHON - <<'PY'
from math_eval.run_opd_proxy_gradient_verify import validate_opd_cli_runtime
print(",".join(validate_opd_cli_runtime(expected_gpus=4, require_idle=True)))
PY
)"
  [[ "$allocated_tokens" == "0,1,2,3" ]] || \
    die "allocation tokens must be exactly 0,1,2,3 (got $allocated_tokens)"
  printf 'validated_allocation_tokens=%s\n' "$allocated_tokens"
}

require_remaining_walltime() {
  [[ -n "${SLURM_JOB_ID:-}" ]] || die "SLURM_JOB_ID must be nonempty"
  local remaining
  remaining="$(squeue -h -j "$SLURM_JOB_ID" -o '%L')"
  [[ -n "$remaining" ]] || die "cannot determine remaining allocation wall time"
  "$GVENDI_PYTHON" - "$remaining" "$REQUIRED_REMAINING_SECONDS" <<'PY'
import re
import sys

value = sys.argv[1].strip()
required = int(sys.argv[2])
match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d+):(\d+)", value)
if match is None:
    raise SystemExit(f"invalid Slurm remaining time: {value}")
days, hours, minutes, seconds = (int(part or 0) for part in match.groups())
total = (((days * 24) + hours) * 60 + minutes) * 60 + seconds
if total < required:
    raise SystemExit(
        f"remaining wall time {total}s is below required reserve {required}s"
    )
print(f"remaining_walltime_seconds={total}")
PY
}

require_no_stale_processes() {
  local pattern='[p]ython.*([v]llm|[c]ollect_prismatic_pilot_gradients|[r]un_prismatic_lite_qwen3_pilot)'
  local stale
  stale="$(pgrep -u "$(id -u)" -f "$pattern" || true)"
  [[ -z "$stale" ]] || die "stale pilot/model processes found: $stale"
}

wait_for_gpu_drain() {
  local deadline=$((SECONDS + 180))
  while (( SECONDS < deadline )); do
    if "$GVENDI_PYTHON" - <<'PY' >/dev/null 2>&1
from math_eval.run_opd_proxy_gradient_verify import validate_opd_cli_runtime
validate_opd_cli_runtime(expected_gpus=4, require_idle=True)
PY
    then
      require_no_stale_processes
      printf '%s\n' 'gpu_process_drain=PASS'
      return 0
    fi
    sleep 5
  done
  die "GPU processes did not drain within 180 seconds"
}

immediate_gpu_gate() {
  verify_frozen_inputs
  require_remaining_walltime
  require_no_stale_processes
  validate_runtime_gate
}

run_prepare() {
  pilot_command "$GVENDI_PYTHON" prepare
}

run_calibration_generation() {
  immediate_gpu_gate
  CUDA_VISIBLE_DEVICES="0,1,2,3" \
    pilot_command "$VERL_PYTHON" generate-calibration-solutions
}

run_proxy_gradient_shards() {
  local input_jsonl="$1"
  local output_dir="$2"
  local prefix="$3"
  require_absent "$output_dir"
  [[ -f "$input_jsonl" ]] || die "gradient input is missing: $input_jsonl"
  [[ -f "$PILOT_MANIFEST" ]] || die "pilot manifest is missing"
  local input_sha manifest_sha
  input_sha="$(sha256sum -- "$input_jsonl" | awk '{print $1}')"
  manifest_sha="$(sha256sum -- "$PILOT_MANIFEST" | awk '{print $1}')"
  local pids=()
  local shard_index
  for shard_index in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="$shard_index" "$GVENDI_PYTHON" \
      -m math_eval.collect_prismatic_pilot_gradients \
      --reference-repo "$REFERENCE_REPO" \
      --input-jsonl "$input_jsonl" \
      --input-sha256 "$input_sha" \
      --pilot-manifest "$PILOT_MANIFEST" \
      --pilot-manifest-sha256 "$manifest_sha" \
      --output-dir "$output_dir" \
      --prefix "$prefix" \
      --model-name Qwen/Qwen2.5-0.5B-Instruct \
      --model-revision 7ae557604adf67be50417f59c2c2f167def9a775 \
      --shard-index "$shard_index" \
      --num-shards 4 \
      --device cuda:0 &
    pids+=("$!")
  done
  local rc=0
  local pid
  set +e
  for pid in "${pids[@]}"; do
    wait "$pid"
    if (( $? != 0 )); then
      rc=1
    fi
  done
  set -e
  (( rc == 0 )) || die "one or more projected-gradient shards failed"
  CUDA_VISIBLE_DEVICES="" "$GVENDI_PYTHON" \
    -m math_eval.collect_prismatic_pilot_gradients \
    --validate-global-only \
    --reference-repo "$REFERENCE_REPO" \
    --input-jsonl "$input_jsonl" \
    --input-sha256 "$input_sha" \
    --pilot-manifest "$PILOT_MANIFEST" \
    --pilot-manifest-sha256 "$manifest_sha" \
    --output-dir "$output_dir" \
    --prefix "$prefix" \
    --num-shards 4
}

run_calibration_gradients() {
  immediate_gpu_gate
  run_proxy_gradient_shards \
    "$CALIBRATION_INPUT" "$CALIBRATION_GRADIENT_DIR" calibration
}

run_calibration_analysis() {
  immediate_gpu_gate
  CUDA_VISIBLE_DEVICES="0" \
    pilot_command "$GVENDI_PYTHON" analyze-calibration --device cuda:0
}

run_problem_generation() {
  immediate_gpu_gate
  CUDA_VISIBLE_DEVICES="0,1,2,3" \
    pilot_command "$VERL_PYTHON" generate-problems
}

run_candidate_solution_generation() {
  immediate_gpu_gate
  CUDA_VISIBLE_DEVICES="0,1,2,3" \
    pilot_command "$VERL_PYTHON" generate-candidate-solutions
}

run_quality_review() {
  pilot_command "$GVENDI_PYTHON" quality-review
}

run_candidate_gradients() {
  immediate_gpu_gate
  run_proxy_gradient_shards \
    "$CANDIDATE_INPUT" "$CANDIDATE_GRADIENT_DIR" candidates
}

run_selection_analysis() {
  immediate_gpu_gate
  CUDA_VISIBLE_DEVICES="0" \
    pilot_command "$GVENDI_PYTHON" analyze-selection --device cuda:0
}

run_final_validation() {
  verify_frozen_inputs
  pilot_command "$GVENDI_PYTHON" validate
  [[ -f "$STAGE_MARKER" ]] || die "final STAGE_COMPLETE.json is missing"
}

PHASE_RC=0
run_allow_no_go() {
  set +e
  "$@"
  PHASE_RC=$?
  set -e
  if (( PHASE_RC != 0 && PHASE_RC != NO_GO_EXIT_CODE )); then
    return "$PHASE_RC"
  fi
  return 0
}

handle_early_no_go() {
  local phase="$1"
  if (( PHASE_RC == NO_GO_EXIT_CODE )); then
    wait_for_gpu_drain
    [[ -f "$STAGE_MARKER" ]] || die "$phase no_go lacks STAGE_COMPLETE.json"
    run_final_validation
    printf 'declared_stop_phase=%s\n' "$phase"
    exit "$NO_GO_EXIT_CODE"
  fi
}

case "${DRY_RUN:-0}" in
  0|1) ;;
  *) die "DRY_RUN must be 0 or 1" ;;
esac

export PYTHONPATH="${CODE_ROOT}:${CODE_ROOT}/verl${PYTHONPATH:+:${PYTHONPATH}}"
cd "$CODE_ROOT"
for command in awk flock git id pgrep sha256sum squeue tee; do
  command -v "$command" >/dev/null 2>&1 || die "missing command: $command"
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  verify_frozen_inputs
  require_absent_initial_targets
  verify_model_tree
  print_provenance
  print_phase_commands
  printf '%s\n' 'DRY_RUN=PASS'
  exit 0
fi

require_absent_initial_targets
verify_frozen_inputs
[[ "${CUDA_VISIBLE_DEVICES:-}" == "$ALLOCATION_TOKENS" ]] || \
  die "CUDA_VISIBLE_DEVICES must be exactly $ALLOCATION_TOKENS"
require_remaining_walltime
require_no_stale_processes
validate_runtime_gate
mkdir -p "$(dirname -- "$MASTER_LOCK")" "$LOG_ROOT"
exec 9<>"$MASTER_LOCK"
flock -n 9 || die "another pilot launcher holds $MASTER_LOCK"
printf '%s\n' "$$" >&9
exec > >(tee -a "$MASTER_LOG") 2>&1
FINAL_MARKER_ENABLED=1
on_exit() {
  local rc=$?
  if [[ "${FINAL_MARKER_ENABLED:-0}" == "1" ]]; then
    printf 'PRISMATIC_LITE_DONE:%s\n' "$rc"
  fi
}
trap on_exit EXIT

printf '%s\n' '=== Prismatic-lite Qwen3 2K pilot ==='
print_provenance
printf '%s\n' '=== canonical execution ==='
run_prepare
run_allow_no_go run_calibration_generation
handle_early_no_go calibration_qualification
wait_for_gpu_drain
run_calibration_gradients
wait_for_gpu_drain
run_allow_no_go run_calibration_analysis
handle_early_no_go calibration_analysis
wait_for_gpu_drain
run_allow_no_go run_problem_generation
handle_early_no_go candidate_problem_generation
wait_for_gpu_drain
run_candidate_solution_generation
wait_for_gpu_drain
run_allow_no_go run_quality_review
handle_early_no_go candidate_quality
run_candidate_gradients
wait_for_gpu_drain
run_allow_no_go run_selection_analysis
selection_rc="$PHASE_RC"
wait_for_gpu_drain
run_final_validation
if (( selection_rc == NO_GO_EXIT_CODE )); then
  exit "$NO_GO_EXIT_CODE"
fi
printf '%s\n' 'pilot_publication=PASS'
