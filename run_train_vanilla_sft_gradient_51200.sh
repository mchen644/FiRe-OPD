#!/usr/bin/env bash
set -euo pipefail

PRODUCTION_ROOT="/home/mchen/FiRe-OPD"
REPO_DIR="${REPO_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)}"
PYTHON_BIN="${PYTHON_BIN:-/home/mchen/miniconda3/envs/verl/bin/python}"
BASE_LAUNCHER="${REPO_DIR}/run_train_vanilla_opd.sh"
DATA_ROOT="${DATA_ROOT:-${PRODUCTION_ROOT}/data/g-opd}"
SOURCE_DATA="${PRODUCTION_ROOT}/data/g-opd/DeepMath-103K/train_filtered_level6.parquet"
TRAIN_DATA="${TRAIN_DATA:-${PRODUCTION_ROOT}/data/gradient_diversity/DeepMath-103K/train_gradient_diverse_51200.parquet}"
VAL_DATA="${VAL_DATA:-['${PRODUCTION_ROOT}/data/g-opd/AIME2024/test.parquet', '${PRODUCTION_ROOT}/data/g-opd/AIME2025/test.parquet']}"
STUDENT_MODEL="${STUDENT_MODEL:-${PRODUCTION_ROOT}/models/Qwen3-4B}"
TEACHER_MODEL="${TEACHER_MODEL:-${PRODUCTION_ROOT}/models/Qwen3-30B-A3B-Instruct-2507}"
SELECTION_MANIFEST="${PRODUCTION_ROOT}/data/gradient_diversity/selection_51200/manifest.json"
SELECTED_IDS="${PRODUCTION_ROOT}/data/gradient_diversity/selection_51200/selected_ids.jsonl"
DIAGNOSTICS="${PRODUCTION_ROOT}/data/gradient_diversity/selection_51200/diagnostics.json"
FROZEN_PREFIX="${PRODUCTION_ROOT}/data/gradient_diversity/selection/selected_ids.jsonl"
OLD_PARQUET="${PRODUCTION_ROOT}/data/gradient_diversity/DeepMath-103K/train_gradient_diverse_12800.parquet"
OLD_MANIFEST="${PRODUCTION_ROOT}/data/gradient_diversity/selection/manifest.json"
OLD_DIAGNOSTICS="${PRODUCTION_ROOT}/data/gradient_diversity/selection/diagnostics.json"
BASELINE_EXPERIMENT="opd-strong-to-weak-rawprompt-4gpu-tp4-refmb4-rollmb4"
BASELINE_CHECKPOINT="${PRODUCTION_ROOT}/checkpoints/${BASELINE_EXPERIMENT}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-opd-strong-to-weak-sftgrad51200-rawprompt-4gpu-tp4-refmb4-rollmb4}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PRODUCTION_ROOT}/checkpoints/${EXPERIMENT_NAME}}"
LOG_FILE="${LOG_FILE:-${PRODUCTION_ROOT}/logs/gradient_diversity/${EXPERIMENT_NAME}.log}"
SOURCE_SHA256="de3350fdd00bc0410550098ea65179e2be873da99e4075f80de575fc17670597"
FROZEN_PREFIX_SHA256="a78c04cff10c148f45bf9c828e07cc9cf01398b48efeb707253273af359a037e"
OLD_PARQUET_SHA256="caf303c5d151fdaed2e21eebc257917f13660c589906e3da0e569ffdfd59b059"
OLD_MANIFEST_SHA256="a1a45382ee577e24f9b455386f9f3adcab210ff40a83ea760c2110fb96f8f90a"
OLD_DIAGNOSTICS_SHA256="d2105179f2ce5a7c796aef07136bbc1dce79b07b5bc2de4d2319b18b47ea761f"
EXPECTED_ROWS="51200"
EXPECTED_SOURCE_ROWS="57046"
EXPECTED_ELIGIBLE_ROWS="57045"
MAX_PROMPT_TOKENS="2048"
EXPECTED_SELECTION_PROFILE="vanilla_51200"
EXPECTED_FROZEN_PREFIX_ROWS="12800"
ROLLOUT_N="${ROLLOUT_N:-1}"
PROMPT_BATCH_SIZE="${PROMPT_BATCH_SIZE:-1024}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-1024}"
TOTAL_TRAJECTORIES="${TOTAL_TRAJECTORIES:-1024}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-50}"
SAVE_FREQ="${SAVE_FREQ:-50}"
TEST_FREQ="${TEST_FREQ:-10}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-4}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
DATA_SEED="${DATA_SEED:-42}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
WARMUP_RATIO="${WARMUP_RATIO:-0.0}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
VALIDATION_N="${VALIDATION_N:-8}"
VALIDATION_TEMPERATURE="${VALIDATION_TEMPERATURE:-1.0}"
VALIDATION_TOP_P="${VALIDATION_TOP_P:-1.0}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-${SLURM_CPUS_PER_TASK:-16}}"


die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

require_pinned_value() {
  local name="$1"
  local actual="$2"
  local expected="$3"
  [[ "$actual" == "$expected" ]] || \
    die "${name} must remain pinned to ${expected} (got ${actual})"
}

require_pinned_value REPO_DIR "$REPO_DIR" /home/mchen/FiRe-OPD/.worktrees/vanilla-sft-gradient-51200-opd
require_pinned_value DATA_ROOT "$DATA_ROOT" /home/mchen/FiRe-OPD/data/g-opd
require_pinned_value SOURCE_DATA "$SOURCE_DATA" /home/mchen/FiRe-OPD/data/g-opd/DeepMath-103K/train_filtered_level6.parquet
require_pinned_value TRAIN_DATA "$TRAIN_DATA" /home/mchen/FiRe-OPD/data/gradient_diversity/DeepMath-103K/train_gradient_diverse_51200.parquet
require_pinned_value VAL_DATA "$VAL_DATA" "['/home/mchen/FiRe-OPD/data/g-opd/AIME2024/test.parquet', '/home/mchen/FiRe-OPD/data/g-opd/AIME2025/test.parquet']"
require_pinned_value STUDENT_MODEL "$STUDENT_MODEL" /home/mchen/FiRe-OPD/models/Qwen3-4B
require_pinned_value TEACHER_MODEL "$TEACHER_MODEL" /home/mchen/FiRe-OPD/models/Qwen3-30B-A3B-Instruct-2507
require_pinned_value EXPERIMENT_NAME "$EXPERIMENT_NAME" opd-strong-to-weak-sftgrad51200-rawprompt-4gpu-tp4-refmb4-rollmb4
require_pinned_value CHECKPOINT_DIR "$CHECKPOINT_DIR" /home/mchen/FiRe-OPD/checkpoints/opd-strong-to-weak-sftgrad51200-rawprompt-4gpu-tp4-refmb4-rollmb4
require_pinned_value LOG_FILE "$LOG_FILE" /home/mchen/FiRe-OPD/logs/gradient_diversity/opd-strong-to-weak-sftgrad51200-rawprompt-4gpu-tp4-refmb4-rollmb4.log
require_pinned_value ROLLOUT_N "$ROLLOUT_N" 1
require_pinned_value PROMPT_BATCH_SIZE "$PROMPT_BATCH_SIZE" 1024
require_pinned_value PPO_MINI_BATCH_SIZE "$PPO_MINI_BATCH_SIZE" 1024
require_pinned_value TOTAL_TRAJECTORIES "$TOTAL_TRAJECTORIES" 1024
require_pinned_value TOTAL_TRAINING_STEPS "$TOTAL_TRAINING_STEPS" 50
require_pinned_value SAVE_FREQ "$SAVE_FREQ" 50
require_pinned_value TEST_FREQ "$TEST_FREQ" 10
require_pinned_value N_GPUS_PER_NODE "$N_GPUS_PER_NODE" 4
require_pinned_value ROLLOUT_TP_SIZE "$ROLLOUT_TP_SIZE" 4
require_pinned_value MAX_PROMPT_LENGTH "$MAX_PROMPT_LENGTH" 2048
require_pinned_value MAX_RESPONSE_LENGTH "$MAX_RESPONSE_LENGTH" 16384
require_pinned_value DATA_SEED "$DATA_SEED" 42
require_pinned_value LEARNING_RATE "$LEARNING_RATE" 1e-6
require_pinned_value WARMUP_RATIO "$WARMUP_RATIO" 0.0
require_pinned_value ROLLOUT_TEMPERATURE "$ROLLOUT_TEMPERATURE" 1.0
require_pinned_value ROLLOUT_TOP_P "$ROLLOUT_TOP_P" 1.0
require_pinned_value VALIDATION_N "$VALIDATION_N" 8
require_pinned_value VALIDATION_TEMPERATURE "$VALIDATION_TEMPERATURE" 1.0
require_pinned_value VALIDATION_TOP_P "$VALIDATION_TOP_P" 1.0
require_pinned_value TOTAL_EPOCHS "$TOTAL_EPOCHS" 3
[[ "$RAY_NUM_CPUS" =~ ^[1-9][0-9]*$ ]] || die "RAY_NUM_CPUS must be positive"
(( $# == 0 )) || die "unreviewed command-line overrides are forbidden"
[[ -f "$BASE_LAUNCHER" ]] || die "canonical Vanilla launcher is missing: $BASE_LAUNCHER"

render_vanilla_contract() {
  local train_data="$1"
  local experiment="$2"
  local checkpoint="$3"
  env \
    VANILLA_OPD_DRY_RUN=1 \
    REPO_DIR="$REPO_DIR" \
    PYTHON_BIN="$PYTHON_BIN" \
    DATA_ROOT="$DATA_ROOT" \
    TRAIN_DATA="$train_data" \
    VAL_DATA="$VAL_DATA" \
    STUDENT_MODEL="$STUDENT_MODEL" \
    TEACHER_MODEL="$TEACHER_MODEL" \
    EXPERIMENT_NAME="$experiment" \
    CHECKPOINT_DIR="$checkpoint" \
    ROLLOUT_N="$ROLLOUT_N" \
    PROMPT_BATCH_SIZE="$PROMPT_BATCH_SIZE" \
    PPO_MINI_BATCH_SIZE="$PPO_MINI_BATCH_SIZE" \
    TOTAL_TRAJECTORIES="$TOTAL_TRAJECTORIES" \
    TOTAL_TRAINING_STEPS="$TOTAL_TRAINING_STEPS" \
    SAVE_FREQ="$SAVE_FREQ" \
    TEST_FREQ="$TEST_FREQ" \
    N_GPUS_PER_NODE="$N_GPUS_PER_NODE" \
    ROLLOUT_TP_SIZE="$ROLLOUT_TP_SIZE" \
    MAX_PROMPT_LENGTH="$MAX_PROMPT_LENGTH" \
    MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH" \
    DATA_SEED="$DATA_SEED" \
    LEARNING_RATE="$LEARNING_RATE" \
    WARMUP_RATIO="$WARMUP_RATIO" \
    ROLLOUT_TEMPERATURE="$ROLLOUT_TEMPERATURE" \
    ROLLOUT_TOP_P="$ROLLOUT_TOP_P" \
    VALIDATION_N="$VALIDATION_N" \
    VALIDATION_TEMPERATURE="$VALIDATION_TEMPERATURE" \
    VALIDATION_TOP_P="$VALIDATION_TOP_P" \
    TOTAL_EPOCHS="$TOTAL_EPOCHS" \
    RAY_NUM_CPUS="$RAY_NUM_CPUS" \
    bash "$BASE_LAUNCHER"
}

normalize_contract() {
  local contract="$1"
  local train_data="$2"
  local experiment="$3"
  local checkpoint="$4"
  contract="${contract//"$checkpoint"/__CHECKPOINT_DIR__}"
  contract="${contract//"$train_data"/__TRAIN_DATA__}"
  contract="${contract//"$experiment"/__EXPERIMENT_NAME__}"
  printf '%s' "$contract"
}

baseline_contract="$(render_vanilla_contract "$SOURCE_DATA" "$BASELINE_EXPERIMENT" "$BASELINE_CHECKPOINT")"
candidate_contract="$(render_vanilla_contract "$TRAIN_DATA" "$EXPERIMENT_NAME" "$CHECKPOINT_DIR")"
baseline_normalized="$(normalize_contract "$baseline_contract" "$SOURCE_DATA" "$BASELINE_EXPERIMENT" "$BASELINE_CHECKPOINT")"
candidate_normalized="$(normalize_contract "$candidate_contract" "$TRAIN_DATA" "$EXPERIMENT_NAME" "$CHECKPOINT_DIR")"
[[ "$baseline_normalized" == "$candidate_normalized" ]] || \
  die "normalized baseline and candidate Vanilla commands differ"

case "${VANILLA_SFTGRAD_DRY_RUN:-0}" in
  0|1) ;;
  *) die "VANILLA_SFTGRAD_DRY_RUN must be 0 or 1" ;;
esac
case "${VANILLA_SFTGRAD_PREFLIGHT_ONLY:-0}" in
  0|1) ;;
  *) die "VANILLA_SFTGRAD_PREFLIGHT_ONLY must be 0 or 1" ;;
esac

printf '%s\n' 'single_variable_contract=PASS'
if [[ "${VANILLA_SFTGRAD_DRY_RUN:-0}" == "1" ]]; then
  printf '%s\n' 'artifact_validation=SKIPPED_DRY_RUN'
  printf '%s\n' "$candidate_contract"
  printf 'delegation_command=bash %q\n' "$BASE_LAUNCHER"
  exit 0
fi

require_sha256() {
  local path="$1"
  local expected="$2"
  [[ -f "$path" ]] || die "required artifact is not a regular file: $path"
  local actual
  actual="$(sha256sum -- "$path" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || \
    die "SHA-256 mismatch for $path: expected $expected, got $actual"
}

ensure_checkpoint_empty() {
  "$PYTHON_BIN" - "$CHECKPOINT_DIR" <<'PY'
import sys
from pathlib import Path

from math_eval.validate_gradient_diverse_training_data import ensure_empty_checkpoint_dir

ensure_empty_checkpoint_dir(Path(sys.argv[1]))
PY
}

[[ -x "$PYTHON_BIN" ]] || die "PYTHON_BIN is not executable: $PYTHON_BIN"
require_pinned_value PYTHON_BIN "$PYTHON_BIN" /home/mchen/miniconda3/envs/verl/bin/python
for command in awk flock git sha256sum tee; do
  command -v "$command" >/dev/null 2>&1 || die "required command is missing: $command"
done
for artifact in \
  "$TRAIN_DATA" "$SELECTION_MANIFEST" "$SELECTED_IDS" "$DIAGNOSTICS" \
  "$FROZEN_PREFIX" "$OLD_PARQUET" "$OLD_MANIFEST" "$OLD_DIAGNOSTICS" \
  "$SOURCE_DATA" "$DATA_ROOT/AIME2024/test.parquet" "$DATA_ROOT/AIME2025/test.parquet"; do
  [[ -f "$artifact" ]] || die "required artifact is not a regular file: $artifact"
done
[[ -d "$STUDENT_MODEL" ]] || die "student model directory is missing: $STUDENT_MODEL"
[[ -d "$TEACHER_MODEL" ]] || die "teacher model directory is missing: $TEACHER_MODEL"
[[ ! -e "$LOG_FILE" ]] || die "training log already exists: $LOG_FILE"

export PYTHONPATH="${REPO_DIR}/verl:${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
cd "$REPO_DIR"
code_status="$(git status --porcelain=v1 --untracked-files=all)"
[[ -z "$code_status" ]] || die "training code worktree must be clean: $code_status"
code_head="$(git rev-parse HEAD)"
code_tree="$(git rev-parse 'HEAD^{tree}')"
code_status_sha256="$(printf '%s' "$code_status" | sha256sum | awk '{print $1}')"
code_diff_sha256="$(git diff --binary HEAD | sha256sum | awk '{print $1}')"
require_sha256 "$SOURCE_DATA" "$SOURCE_SHA256"
require_sha256 "$FROZEN_PREFIX" "$FROZEN_PREFIX_SHA256"
require_sha256 "$OLD_PARQUET" "$OLD_PARQUET_SHA256"
require_sha256 "$OLD_MANIFEST" "$OLD_MANIFEST_SHA256"
require_sha256 "$OLD_DIAGNOSTICS" "$OLD_DIAGNOSTICS_SHA256"
ensure_checkpoint_empty

artifact_contract_json="$($PYTHON_BIN - \
  "$SELECTION_MANIFEST" \
  "$TRAIN_DATA" \
  "$SELECTED_IDS" \
  "$DIAGNOSTICS" \
  "$SOURCE_DATA" \
  "$FROZEN_PREFIX" \
  "$REPO_DIR" \
  "$code_head" \
  "$code_tree" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def reject_duplicate_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_nonfinite(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def strict_load(path):
    value = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_pairs,
        parse_constant=reject_nonfinite,
    )
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


(
    manifest_path,
    selected_path,
    selected_ids_path,
    diagnostics_path,
    source_path,
    frozen_prefix_path,
    repository_path,
    code_head,
    code_tree,
) = (
    Path(sys.argv[1]),
    Path(sys.argv[2]),
    Path(sys.argv[3]),
    Path(sys.argv[4]),
    Path(sys.argv[5]),
    Path(sys.argv[6]),
    Path(sys.argv[7]),
    sys.argv[8],
    sys.argv[9],
)
manifest = strict_load(manifest_path)
diagnostics = strict_load(diagnostics_path)
require(manifest.get("manifest_version") == 1, "manifest_version mismatch")
for field, expected in (
    ("source_row_count", 57_046),
    ("eligible_row_count", 57_045),
    ("selected_row_count", 51_200),
):
    require(manifest.get(field) == expected, f"manifest {field} mismatch")
for field, expected in (
    ("source_parquet", source_path.resolve()),
    ("output_parquet", selected_path.resolve()),
    ("selected_ids", selected_ids_path.resolve()),
    ("diagnostics", diagnostics_path.resolve()),
):
    require(manifest.get(field) == str(expected), f"manifest {field} path mismatch")

actual = {
    "diagnostics_sha256": sha256(diagnostics_path),
    "manifest_sha256": sha256(manifest_path),
    "selected_ids_sha256": sha256(selected_ids_path),
    "selected_sha256": sha256(selected_path),
    "source_sha256": sha256(source_path),
}
require(
    actual["source_sha256"]
    == "de3350fdd00bc0410550098ea65179e2be873da99e4075f80de575fc17670597",
    "source SHA-256 mismatch",
)
for field, actual_key in (
    ("source_sha256", "source_sha256"),
    ("output_parquet_sha256", "selected_sha256"),
    ("selected_ids_sha256", "selected_ids_sha256"),
    ("diagnostics_sha256", "diagnostics_sha256"),
):
    require(manifest.get(field) == actual[actual_key], f"manifest {field} mismatch")
require(manifest.get("diagnostic_results") == diagnostics, "diagnostics content mismatch")

for path_field, hash_field, expected_path, expected_hash in (
    (
        "prepared_jsonl",
        "prepared_jsonl_sha256",
        "/home/mchen/FiRe-OPD/data/gradient_diversity/deepmath_level6_r1_solution1.jsonl",
        "ee8d55c943577888f14052b3953a0a8aa07d26076e50ddffa41a102b60027344",
    ),
    (
        "prepared_manifest",
        "prepared_manifest_sha256",
        "/home/mchen/FiRe-OPD/data/gradient_diversity/deepmath_level6_r1_solution1.manifest.json",
        "5add2e965473d3647f2318c73f957434fbe23344516089be9adfe89db9c6a70e",
    ),
    (
        "eligibility_report",
        "eligibility_report_sha256",
        "/home/mchen/FiRe-OPD/data/gradient_diversity/deepmath_level6_r1_solution1.eligibility.json",
        "cc8d5b888def37761519e15c6ef722e5a22d91ae2342c6d669f008bcff3dcbb6",
    ),
    (
        "gradient_manifest",
        "gradient_manifest_sha256",
        "/home/mchen/FiRe-OPD/data/gradient_diversity/gradients/qwen2.5-0.5b-instruct/gradient.manifest.json",
        "9a534118a08e736a15e806933d5cf90c2a99822fee28bf18644e5ff344ce3ae2",
    ),
):
    declared_path = manifest.get(path_field)
    require(declared_path == expected_path, f"manifest {path_field} path mismatch")
    require(manifest.get(hash_field) == expected_hash, f"manifest {hash_field} pin mismatch")
    require(sha256(Path(declared_path)) == expected_hash, f"manifest {hash_field} mismatch")

for field, expected in (
    ("dataset_name", "zwhe99/DeepMath-103K"),
    ("dataset_revision", "5cf055d1fe3d7a2eb19719ac020211469736ae44"),
    ("model_name", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("model_revision", "7ae557604adf67be50417f59c2c2f167def9a775"),
    ("gradient_prefix", "deepmath"),
    ("gradient_num_shards", 4),
):
    require(manifest.get(field) == expected, f"manifest {field} mismatch")
require(
    manifest.get("reference_repo") == "/home/mchen/prismatic-synthesis-reference",
    "reference repository mismatch",
)
require(
    manifest.get("reference_commit")
    == "d9484cd3b5991030b901ac4a3a9e2472dbfac2ad",
    "reference commit mismatch",
)
require(
    manifest.get("reference_tree")
    == "a0079d8c5e15cb18bb4790f99c43cc19bb9ecd50",
    "reference tree mismatch",
)
clustering = manifest.get("clustering")
require(
    clustering
    == {
        "distance": "cosine",
        "iterations": 20,
        "primary_ratio": 0.1,
        "primary_seed": 42,
        "ratios": [0.1, 0.01],
        "seeds": [42, 43],
    },
    "clustering contract mismatch",
)
selection = manifest.get("selection")
require(isinstance(selection, dict), "selection contract missing")
expected_prefix = {
    "path": str(frozen_prefix_path.resolve()),
    "row_count": 12_800,
    "sha256": "a78c04cff10c148f45bf9c828e07cc9cf01398b48efeb707253273af359a037e",
    "selected_id_sequence_sha256": "e59e9fd7270f5b0a533bd5e4374733a4f68edbbae1edb1ffea4bf21cbb91ecaf",
}
require(
    selection
    == {
        "method": "balanced_round_robin",
        "seed": 42,
        "target_size": 51_200,
        "profile": "vanilla_51200",
        "frozen_prefix": expected_prefix,
    },
    "selection profile or frozen prefix contract mismatch",
)
require(sha256(frozen_prefix_path) == expected_prefix["sha256"], "frozen prefix hash mismatch")

selection_source = manifest.get("selection_source")
require(isinstance(selection_source, dict), "selection source provenance missing")
require(
    selection_source.get("repository") == str(repository_path.resolve()),
    "selection source repository mismatch",
)
require(selection_source.get("head") == code_head, "selection source HEAD mismatch")
require(selection_source.get("tree") == code_tree, "selection source tree mismatch")
expected_source_files = {
    "math_eval/deepmath_gradient_diversity.py",
    "math_eval/select_gradient_diverse_deepmath.py",
}
source_files = selection_source.get("files")
require(
    isinstance(source_files, dict) and set(source_files) == expected_source_files,
    "selection source file inventory mismatch",
)
for relative, expected_hash in source_files.items():
    current = repository_path / relative
    require(sha256(current) == expected_hash, f"selection source hash mismatch: {relative}")
    committed = subprocess.run(
        ["git", "-C", str(repository_path), "show", f"{code_head}:{relative}"],
        check=True,
        capture_output=True,
    ).stdout
    require(
        hashlib.sha256(committed).hexdigest() == expected_hash,
        f"committed selection source hash mismatch: {relative}",
    )
print(json.dumps(actual, sort_keys=True, separators=(",", ":")))
PY
)"
printf 'artifact_contract=%s\n' "$artifact_contract_json"
hash_line="$($PYTHON_BIN - "$artifact_contract_json" <<'PY'
import json
import sys

value = json.loads(sys.argv[1])
print("\t".join(value[key] for key in (
    "manifest_sha256",
    "selected_sha256",
    "selected_ids_sha256",
    "diagnostics_sha256",
)))
PY
)"
IFS=$'\t' read -r manifest_sha256 selected_sha256 selected_ids_sha256 diagnostics_sha256 <<<"$hash_line"
for value in "$manifest_sha256" "$selected_sha256" "$selected_ids_sha256" "$diagnostics_sha256"; do
  [[ "$value" =~ ^[0-9a-f]{64}$ ]] || die "artifact contract returned an invalid SHA-256"
done

validator_command=(
  "$PYTHON_BIN" -m math_eval.validate_gradient_diverse_training_data
  --selected-parquet "$TRAIN_DATA"
  --source-parquet "$SOURCE_DATA"
  --selection-manifest "$SELECTION_MANIFEST"
  --selected-ids "$SELECTED_IDS"
  --diagnostics "$DIAGNOSTICS"
  --tokenizer-path "$STUDENT_MODEL"
  --expected-selected-sha256 "$selected_sha256"
  --expected-source-sha256 "$SOURCE_SHA256"
  --expected-manifest-sha256 "$manifest_sha256"
  --expected-selected-ids-sha256 "$selected_ids_sha256"
  --expected-diagnostics-sha256 "$diagnostics_sha256"
  --expected-rows "$EXPECTED_ROWS"
  --expected-source-rows "$EXPECTED_SOURCE_ROWS"
  --expected-eligible-rows "$EXPECTED_ELIGIBLE_ROWS"
  --max-prompt-tokens "$MAX_PROMPT_TOKENS"
  --expected-selection-profile "$EXPECTED_SELECTION_PROFILE"
  --frozen-prefix-selected-ids "$FROZEN_PREFIX"
  --expected-frozen-prefix-sha256 "$FROZEN_PREFIX_SHA256"
  --expected-frozen-prefix-rows "$EXPECTED_FROZEN_PREFIX_ROWS"
  --checkpoint-dir "$CHECKPOINT_DIR"
)
validation_report="$("${validator_command[@]}")"
$PYTHON_BIN - "$validation_report" \
  "$TRAIN_DATA" "$SOURCE_DATA" "$SELECTION_MANIFEST" "$SELECTED_IDS" "$DIAGNOSTICS" \
  "$selected_sha256" "$SOURCE_SHA256" "$manifest_sha256" "$selected_ids_sha256" "$diagnostics_sha256" <<'PY'
import json
from pathlib import Path
import sys


def reject_duplicate_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_nonfinite(value):
    raise ValueError(f"non-finite JSON constant: {value}")


report = json.loads(
    sys.argv[1],
    object_pairs_hook=reject_duplicate_pairs,
    parse_constant=reject_nonfinite,
)
expected_paths = {
    "selected_parquet": str(Path(sys.argv[2]).resolve()),
    "source_parquet": str(Path(sys.argv[3]).resolve()),
    "selection_manifest": str(Path(sys.argv[4]).resolve()),
    "selected_ids_path": str(Path(sys.argv[5]).resolve()),
    "diagnostics": str(Path(sys.argv[6]).resolve()),
}
for key, expected in expected_paths.items():
    if report.get(key) != expected:
        raise ValueError(f"validator report {key} mismatch")
expected_hashes = {
    "selected_sha256": sys.argv[7],
    "source_sha256": sys.argv[8],
    "selection_manifest_sha256": sys.argv[9],
    "selected_ids_sha256": sys.argv[10],
    "diagnostics_sha256": sys.argv[11],
}
for key, expected in expected_hashes.items():
    if report.get(key) != expected:
        raise ValueError(f"validator report {key} mismatch")
expected_values = {
    "selected_rows": 51_200,
    "source_rows": 57_046,
    "eligible_rows": 57_045,
    "selected_ids": 51_200,
    "unique_prompts": 51_200,
    "schema_equal": True,
    "source_rows_equal": True,
    "prompts_over_limit": 0,
    "selection_profile": "vanilla_51200",
    "frozen_prefix_rows": 12_800,
    "frozen_prefix_sha256": "a78c04cff10c148f45bf9c828e07cc9cf01398b48efeb707253273af359a037e",
    "frozen_prefix_path": "/home/mchen/FiRe-OPD/data/gradient_diversity/selection/selected_ids.jsonl",
}
for key, expected in expected_values.items():
    if report.get(key) != expected:
        raise ValueError(f"validator report {key} mismatch")
lengths = [
    report.get("min_prompt_tokens"),
    report.get("median_prompt_tokens"),
    report.get("p95_prompt_tokens"),
    report.get("p99_prompt_tokens"),
    report.get("max_prompt_tokens"),
]
if not all(isinstance(value, (int, float)) for value in lengths):
    raise ValueError("validator report token quantiles are invalid")
if lengths != sorted(lengths) or lengths[-1] > 2_048:
    raise ValueError("validator report token quantiles violate training limit")
if report.get("prompt_token_limit") != 2_048:
    raise ValueError("validator report prompt_token_limit mismatch")
print("ARTIFACT_VALIDATION_REPORT=PASS")
PY
ensure_checkpoint_empty
printf 'artifact_validation=PASS\n'
printf 'validator_report=%s\n' "$validation_report"
printf 'code_head=%s\n' "$code_head"
printf 'code_tree=%s\n' "$code_tree"
printf 'code_status_sha256=%s\n' "$code_status_sha256"
printf 'code_diff_sha256=%s\n' "$code_diff_sha256"
printf 'base_launcher_sha256=%s\n' "$(sha256sum -- "$BASE_LAUNCHER" | awk '{print $1}')"
printf 'candidate_launcher_sha256=%s\n' "$(sha256sum -- "${BASH_SOURCE[0]}" | awk '{print $1}')"
printf '%s\n' "$candidate_contract"
printf 'delegation_command=bash %q\n' "$BASE_LAUNCHER"

if [[ "${VANILLA_SFTGRAD_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  printf '%s\n' 'training_gpu_work=SKIPPED_PREFLIGHT_ONLY'
  exit 0
fi

launch_lock="/tmp/fire-opd-${UID}-${EXPERIMENT_NAME}.launch.lock"
exec 9<>"$launch_lock"
flock -n 9 || die "another launcher holds $launch_lock"
printf '%s\n' "$$" >&9
[[ -n "${SLURM_JOB_ID:-}" ]] || die "SLURM_JOB_ID must be nonempty"
allocated_tokens="$($PYTHON_BIN - <<'PY'
from math_eval.run_opd_proxy_gradient_verify import validate_opd_cli_runtime

print(",".join(validate_opd_cli_runtime(expected_gpus=4, require_idle=True)))
PY
)"
[[ -n "$allocated_tokens" ]] || die "runtime validation returned no CUDA tokens"
ensure_checkpoint_empty
mkdir -p "$(dirname -- "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1
printf '%s\n' '=== Vanilla OPD SFT-gradient 51,200 authorized launch ==='
printf 'started_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'slurm_job_id=%s\n' "$SLURM_JOB_ID"
printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-}"
printf 'validated_allocation_tokens=%s\n' "$allocated_tokens"
printf 'artifact_contract=%s\n' "$artifact_contract_json"
printf 'validator_report=%s\n' "$validation_report"
printf 'code_head=%s\n' "$code_head"
printf 'code_tree=%s\n' "$code_tree"
printf 'base_launcher_sha256=%s\n' "$(sha256sum -- "$BASE_LAUNCHER" | awk '{print $1}')"
printf 'candidate_launcher_sha256=%s\n' "$(sha256sum -- "${BASH_SOURCE[0]}" | awk '{print $1}')"
printf '%s\n' "$candidate_contract"
printf '%s\n' 'single_variable_contract=PASS'

unset VANILLA_SFTGRAD_DRY_RUN VANILLA_SFTGRAD_PREFLIGHT_ONLY VANILLA_OPD_DRY_RUN
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export REPO_DIR PYTHON_BIN DATA_ROOT TRAIN_DATA VAL_DATA STUDENT_MODEL TEACHER_MODEL
export EXPERIMENT_NAME CHECKPOINT_DIR ROLLOUT_N PROMPT_BATCH_SIZE PPO_MINI_BATCH_SIZE
export TOTAL_TRAJECTORIES TOTAL_TRAINING_STEPS SAVE_FREQ TEST_FREQ N_GPUS_PER_NODE
export ROLLOUT_TP_SIZE MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH DATA_SEED LEARNING_RATE
export WARMUP_RATIO ROLLOUT_TEMPERATURE ROLLOUT_TOP_P VALIDATION_N
export VALIDATION_TEMPERATURE VALIDATION_TOP_P TOTAL_EPOCHS RAY_NUM_CPUS
exec bash "$BASE_LAUNCHER"
