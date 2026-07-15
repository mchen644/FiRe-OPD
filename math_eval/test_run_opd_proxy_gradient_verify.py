import fcntl
import hashlib
import os
import subprocess
from pathlib import Path

import pytest

from math_eval.run_opd_proxy_gradient_verify import (
    GVENDI_PYTHON,
    VERL_PYTHON,
    StageCommand,
    _unit_environment,
    _wait_for_gpu_idle,
    build_capture_hydra_overrides,
    build_experiment_source_snapshot,
    build_source_snapshot,
    build_stage_commands,
    canonical_json_bytes,
    estimate_remaining_seconds,
    execute_stage_commands,
    parse_cli_args,
    parse_slurm_end_time,
    validate_cross_stage_source,
    validate_opd_cli_runtime,
    validate_stage_parent,
    validate_stage_prerequisite,
    validate_stage2_trigger,
)


REFERENCE_REPO = Path("/home/mchen/prismatic-synthesis-reference")


def _manifest(stage: int = 0) -> dict:
    sizes = {
        0: (24, 8, 5, 2, 2, 100),
        1: (768, 256, 172, 76, 7, 10_000),
        2: (1536, 512, 345, 153, 15, 10_000),
    }
    candidate, heldout, selected, primary, diagnostic, draws = sizes[stage]
    model_manifests = {
        "target_teacher": {"manifest_sha256": "1" * 64},
        "target_student": {"manifest_sha256": "2" * 64},
        "proxy_teacher": {"manifest_sha256": "2" * 64},
        "proxy_student": {"manifest_sha256": "3" * 64},
    }
    return {
        "schema_version": 1,
        "artifact_type": "opd_proxy_stage",
        "stage": stage,
        "candidate_count": candidate,
        "held_out_count": heldout,
        "selected_size": selected,
        "primary_k": primary,
        "diagnostic_k": diagnostic,
        "null_draws": draws,
        "sample_manifest": "sample_manifest.jsonl",
        "sample_manifest_sha256": "4" * 64,
        "target_capture_count": candidate + heldout if stage != 2 else 1024,
        "proxy_capture_count": candidate if stage != 2 else 768,
        "target_capture_parquet": "capture_target.parquet",
        "target_capture_parquet_sha256": "5" * 64,
        "proxy_capture_parquet": "capture_proxy.parquet",
        "proxy_capture_parquet_sha256": "6" * 64,
        "parent_report": (
            {"path": "/tmp/stage1-report.json", "sha256": "7" * 64}
            if stage == 2
            else None
        ),
        "provenance": {"root_manifest": {"model_manifests": model_manifests}},
    }


def _runtime_env(visible="GPU-a,GPU-b,GPU-c,GPU-d", **updates):
    env = {
        "SLURM_JOB_ID": "1234",
        "CUDA_VISIBLE_DEVICES": visible,
    }
    env.update(updates)
    return env


class _FakeProcess:
    def __init__(self, runtime, command, returncode, pid):
        self.runtime = runtime
        self.command = command
        self.returncode = returncode
        self.pid = pid
        self.terminated = False
        self.killed = False
        self._done = False

    def wait(self):
        self._done = True
        if self.returncode == 0:
            for path in self.command.completion_paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(
                    canonical_json_bytes(
                        {"status": "complete", "name": self.command.name}
                    )
                )
        return self.returncode

    def poll(self):
        return self.returncode if self._done else None

    def terminate(self):
        self.terminated = True
        self._done = True

    def kill(self):
        self.killed = True
        self._done = True


class _FakeRuntime:
    def __init__(
        self,
        *,
        session="opd-CLI",
        occupied=None,
        failures=None,
        create_completions=True,
    ):
        self.session = session
        self.occupied = occupied or {}
        self.failures = failures or {}
        self.create_completions = create_completions
        self.started = []
        self.processes = []
        self.idle_checks = []
        self.clock = 100.0
        self.next_pid = 2000

    def tmux_session_name(self):
        return self.session

    def gpu_processes(self, token):
        self.idle_checks.append((token, tuple(command.name for command in self.started)))
        return set(self.occupied.get(token, set()))

    def start(self, command, *, env, log_path):
        self.started.append(command)
        self.next_pid += 1
        process = _FakeProcess(
            self,
            command,
            self.failures.get(command.name, 0),
            self.next_pid,
        )
        if not self.create_completions:
            process.returncode = self.failures.get(command.name, 0)
            original_wait = process.wait

            def wait_without_completion():
                process._done = True
                return process.returncode

            process.wait = wait_without_completion
            process.original_wait = original_wait
        self.processes.append(process)
        return process

    def monotonic(self):
        self.clock += 1.0
        return self.clock


def _command(
    tmp_path: Path,
    name: str,
    *,
    gpu_tokens=(),
    group=None,
    output_kind="generic",
    rows=1,
):
    output = tmp_path / name
    return StageCommand(
        name=name,
        argv=("echo", name),
        interpreter="gvendi_analysis",
        gpu_tokens=tuple(gpu_tokens),
        parallel_group=group,
        completion_paths=(output / "COMPLETE.json",),
        output_root=output,
        output_kind=output_kind,
        row_count=rows,
        log_path=tmp_path / "logs" / f"{name}.log",
        validate_output=False,
    )


def test_child_environment_removes_conflicting_rocm_visibility_aliases(tmp_path):
    command = _command(tmp_path, "environment")
    env = _unit_environment(
        {
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "ROCR_VISIBLE_DEVICES": "0,1,2,3",
            "HIP_VISIBLE_DEVICES": "0,1,2,3",
        },
        command,
        ("0", "1", "2", "3"),
    )
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert "ROCR_VISIBLE_DEVICES" not in env
    assert "HIP_VISIBLE_DEVICES" not in env


def test_gpu_cleanup_wait_is_condition_based_and_bounded():
    class Runtime:
        def __init__(self, responses):
            self.responses = list(responses)
            self.clock = 0.0
            self.sleeps = 0

        def gpu_processes(self, token):
            assert token == "0"
            if len(self.responses) > 1:
                return self.responses.pop(0)
            return self.responses[0]

        def monotonic(self):
            return self.clock

        def sleep(self, seconds):
            self.sleeps += 1
            self.clock += seconds

    draining = Runtime([{1257647}, {1257647}, set()])
    _wait_for_gpu_idle(("0",), runtime=draining, timeout_seconds=5)
    assert draining.sleeps == 2

    stuck = Runtime([{1257647}])
    with pytest.raises(RuntimeError, match="did not drain"):
        _wait_for_gpu_idle(("0",), runtime=stuck, timeout_seconds=2)


def test_runtime_requires_four_unique_allocated_tokens():
    runtime = _FakeRuntime()
    for visible in ("", "0,1,2", "0,1,1,3", "0,1,2,3,4"):
        with pytest.raises(RuntimeError):
            validate_opd_cli_runtime(
                env=_runtime_env(visible), runtime=runtime, expected_gpus=4
            )


def test_runtime_requires_slurm_job_and_exact_tmux_session():
    with pytest.raises(RuntimeError, match="SLURM_JOB_ID"):
        validate_opd_cli_runtime(
            env={"CUDA_VISIBLE_DEVICES": "0,1,2,3"}, runtime=_FakeRuntime()
        )
    with pytest.raises(RuntimeError, match="opd-CLI"):
        validate_opd_cli_runtime(
            env=_runtime_env(), runtime=_FakeRuntime(session="other")
        )


def test_runtime_rejects_inactive_or_ambiguous_slurm_job():
    class _InactiveSlurmRuntime(_FakeRuntime):
        def slurm_job_info(self, job_id):
            return f"JobId={job_id} JobState=COMPLETED\n"

    with pytest.raises(RuntimeError, match="uniquely active"):
        validate_opd_cli_runtime(env=_runtime_env(), runtime=_InactiveSlurmRuntime())


def test_runtime_rejects_occupied_gpu_but_ignores_declared_owned_child():
    runtime = _FakeRuntime(occupied={"GPU-a": {999}})
    with pytest.raises(RuntimeError, match="occupied"):
        validate_opd_cli_runtime(env=_runtime_env(), runtime=runtime)
    assert validate_opd_cli_runtime(
        env=_runtime_env(), runtime=runtime, owned_pids={999}
    ) == ("GPU-a", "GPU-b", "GPU-c", "GPU-d")


def test_dry_run_orders_capture_replay_selection_analysis(tmp_path):
    commands = build_stage_commands(
        stage=0,
        manifest=_manifest(0),
        repository_root=tmp_path,
        stage_directory=tmp_path / "stage_0",
        reference_repo=REFERENCE_REPO,
    )
    names = [command.name for command in commands]
    assert names[:4] == [
        "capture_target_seed_42",
        "capture_target_seed_43",
        "capture_proxy_seed_42",
        "capture_proxy_seed_43",
    ]
    assert names.index("direct_gradient_fixture") > 3
    assert names.index("validate_vectors") < names.index("select") < names.index(
        "analyze"
    )
    assert all("srun" not in command.argv for command in commands)
    assert all(command.interpreter == "verl_capture" for command in commands[:4])
    assert all(
        command.interpreter == "gvendi_analysis" for command in commands[4:]
    )
    assert all(VERL_PYTHON not in command.argv for command in commands[4:])
    replay = [command for command in commands if command.name.startswith("replay_")]
    assert len(replay) == 16
    assert all(len(command.gpu_tokens) == 1 for command in replay)
    assert all(command.argv[0] == GVENDI_PYTHON for command in replay)
    assert next(command for command in commands if command.name == "select").argv[0] == (
        GVENDI_PYTHON
    )


def test_cli_dry_run_is_byte_identical_and_loads_no_models(tmp_path):
    stage = tmp_path / "data/opd_proxy_gradient_verify/stage_0"
    stage.mkdir(parents=True)
    sample = stage / "sample_manifest.jsonl"
    target = stage / "capture_target.parquet"
    proxy = stage / "capture_proxy.parquet"
    sample.write_text(
        '{"split":"candidate","stable_id":"q0"}\n', encoding="utf-8"
    )
    target.write_bytes(b"target")
    proxy.write_bytes(b"proxy")
    manifest = _manifest(0)
    manifest["sample_manifest_sha256"] = hashlib.sha256(sample.read_bytes()).hexdigest()
    manifest["target_capture_parquet_sha256"] = hashlib.sha256(
        target.read_bytes()
    ).hexdigest()
    manifest["proxy_capture_parquet_sha256"] = hashlib.sha256(
        proxy.read_bytes()
    ).hexdigest()
    (stage / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    repository = Path.cwd()
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b,GPU-c,GPU-d",
        "OPD_PROXY_VERIFY_DRY_RUN": "1",
        "PYTHONPATH": f"{repository / 'verl'}:{repository}",
    }
    command = [
        VERL_PYTHON,
        "-m",
        "math_eval.run_opd_proxy_gradient_verify",
        "--stage",
        "0",
        "--reference-repo",
        str(REFERENCE_REPO),
    ]
    first = subprocess.run(
        command, cwd=tmp_path, env=env, check=True, capture_output=True
    ).stdout
    second = subprocess.run(
        command, cwd=tmp_path, env=env, check=True, capture_output=True
    ).stdout
    assert first == second
    assert b'"name":"capture_target_seed_42"' in first
    assert b'"name":"analyze"' in first
    assert b"srun" not in first


def test_non_smoke_plan_omits_direct_fixture_and_stage2_uses_stage1_sources(tmp_path):
    commands = build_stage_commands(
        stage=2,
        manifest=_manifest(2),
        repository_root=tmp_path,
        stage_directory=tmp_path / "stage_2",
        reference_repo=REFERENCE_REPO,
    )
    names = [command.name for command in commands]
    assert "direct_gradient_fixture" not in names
    select = next(command for command in commands if command.name == "select")
    assert any("stage_1" in argument for argument in select.argv)
    vectors = [
        select.argv[index + 1]
        for index, value in enumerate(select.argv[:-1])
        if value == "--vector"
    ]
    sft_sources = [value for value in vectors if value.startswith("S=")]
    embedding_sources = [value for value in vectors if value.startswith("E=")]
    assert len(sft_sources) == len(embedding_sources) == 1
    assert all("stage_2" in value and "stage_1" not in value for value in sft_sources)
    assert all(
        "stage_2" in value and "stage_1" not in value
        for value in embedding_sources
    )


def test_capture_hydra_contract_is_fully_explicit(tmp_path):
    stage_dir = tmp_path / "stage_0"
    overrides = build_capture_hydra_overrides(
        stage=0,
        manifest=_manifest(0),
        stage_directory=stage_dir,
        pair="target",
        seed=42,
        output_root=stage_dir / "capture/target/seed_42",
        repository_root=tmp_path,
    )
    values = dict(override.split("=", 1) for override in overrides)
    assert values["trainer.nnodes"] == "1"
    assert values["trainer.n_gpus_per_node"] == "4"
    assert values["actor_rollout_ref.rollout.tensor_model_parallel_size"] == "4"
    assert values["actor_rollout_ref.rollout.data_parallel_size"] == "1"
    assert values["actor_rollout_ref.rollout.pipeline_model_parallel_size"] == "1"
    assert values["actor_rollout_ref.rollout.n"] == "4"
    assert values["actor_rollout_ref.rollout.max_num_batched_tokens"] == "32768"
    assert values["actor_rollout_ref.rollout.max_model_len"] == "18432"
    assert values["actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu"] == "1"
    assert values["actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu"] == "1"
    assert values["data.filter_overlong_prompts"] == "false"
    assert values["data.truncation"] == "error"
    assert values["data.shuffle"] == "false"
    assert values["algorithm.candidate_selection.enabled"] == "false"
    assert values["algorithm.tale_budget.enabled"] == "false"
    assert values["algorithm.difficulty_aware_opd.enabled"] == "false"
    assert values["algorithm.rethinking_opd_probe.enabled"] == "false"
    assert values["algorithm.opd_proxy_verify_capture.expected_questions"] == "32"
    assert values["algorithm.opd_proxy_verify_capture.source_snapshot"] == str(
        (stage_dir / "source_snapshot.json").resolve()
    )
    assert (
        values["algorithm.opd_proxy_verify_capture.source_snapshot_sha256"]
        == "0" * 64
    )
    assert values["actor_rollout_ref.model.path"].endswith("models/Qwen3-4B")
    assert values["+actor_rollout_ref.ref.model.path"].endswith(
        "models/Qwen3-30B-A3B-Instruct-2507"
    )


def test_real_work_contracts_bind_logical_source_snapshot_hash(tmp_path):
    stage_dir = tmp_path / "stage_0"
    stage_dir.mkdir()
    source_files = [{"path": "module.py", "sha256": "1" * 64}]
    source_hash = hashlib.sha256(canonical_json_bytes(source_files)).hexdigest()
    (stage_dir / "source_snapshot.json").write_bytes(
        canonical_json_bytes(
            {"files": source_files, "manifest_sha256": source_hash}
        )
    )
    overrides = build_capture_hydra_overrides(
        stage=0,
        manifest=_manifest(0),
        stage_directory=stage_dir,
        pair="target",
        seed=42,
        output_root=stage_dir / "capture/target/seed_42",
        repository_root=tmp_path,
    )
    values = dict(value.split("=", 1) for value in overrides)
    assert values["algorithm.opd_proxy_verify_capture.source_snapshot_sha256"] == (
        source_hash
    )
    commands = build_stage_commands(
        stage=0,
        manifest=_manifest(0),
        repository_root=tmp_path,
        stage_directory=stage_dir,
        reference_repo=REFERENCE_REPO,
    )
    assert all(
        dict(command.input_hashes)["source_snapshot_sha256"] == source_hash
        for command in commands
    )
    select = next(command for command in commands if command.name == "select")
    assert select.argv[select.argv.index("--source-snapshot") + 1] == str(
        stage_dir / "source_snapshot.json"
    )


def test_capture_hydra_overrides_compose_without_model_loading(tmp_path):
    overrides = build_capture_hydra_overrides(
        stage=0,
        manifest=_manifest(0),
        stage_directory=tmp_path / "stage_0",
        pair="target",
        seed=42,
        output_root=tmp_path / "capture",
        repository_root=Path.cwd(),
    )
    result = subprocess.run(
        [VERL_PYTHON, "-m", "verl.trainer.main_ppo", "--cfg", "job", *overrides],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "verl:."},
    )
    assert "tensor_model_parallel_size: 4" in result.stdout
    assert "max_num_batched_tokens: 32768" in result.stdout
    assert "opd_proxy_verify_capture_only: true" in result.stdout


def test_proxy_capture_uses_small_student_and_byte_identical_4b_teacher(tmp_path):
    values = dict(
        value.split("=", 1)
        for value in build_capture_hydra_overrides(
            stage=0,
            manifest=_manifest(0),
            stage_directory=tmp_path / "stage_0",
            pair="proxy",
            seed=43,
            output_root=tmp_path / "capture",
            repository_root=tmp_path,
        )
    )
    assert values["actor_rollout_ref.model.path"].endswith("models/Qwen3-0.6B")
    assert values["+actor_rollout_ref.ref.model.path"].endswith("models/Qwen3-4B")
    models = _manifest(0)["provenance"]["root_manifest"]["model_manifests"]
    assert models["target_student"]["manifest_sha256"] == models["proxy_teacher"][
        "manifest_sha256"
    ]


def test_stage_parent_contract_rejects_wrong_or_unexpected_parent(tmp_path):
    parent = tmp_path / "report.json"
    parent.write_text("{}\n", encoding="utf-8")
    manifest = _manifest(2)
    manifest["parent_report"] = {
        "path": str(parent),
        "sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="parent"):
        validate_stage_parent(2, manifest, parent)
    with pytest.raises(ValueError, match="forbids"):
        validate_stage_parent(1, _manifest(1), parent)


def test_stage1_cannot_start_without_completed_stage0_smoke(tmp_path):
    with pytest.raises(RuntimeError, match="completed Stage 0"):
        validate_stage_prerequisite(1, tmp_path)


def test_stage_prerequisite_binds_report_source_and_smoke_evidence(tmp_path):
    stage0 = tmp_path / "stage_0"
    (stage0 / "direct_gradient_fixture").mkdir(parents=True)
    report = stage0 / "report.json"
    resume = stage0 / "resume_exercise.json"
    direct = stage0 / "direct_gradient_fixture/COMPLETE.json"
    report.write_bytes(canonical_json_bytes({"stage": 0}))
    resume.write_bytes(canonical_json_bytes({"status": "complete"}))
    direct.write_bytes(canonical_json_bytes({"status": "complete"}))
    files = [{"path": "module.py", "sha256": "1" * 64}]
    source_hash = hashlib.sha256(canonical_json_bytes(files)).hexdigest()
    source = stage0 / "source_snapshot.json"
    source.write_bytes(
        canonical_json_bytes({"files": files, "manifest_sha256": source_hash})
    )
    marker = {
        "stage": 0,
        "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "source_snapshot_sha256": source_hash,
        "source_snapshot_file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "resume_exercise_sha256": hashlib.sha256(resume.read_bytes()).hexdigest(),
        "direct_fixture_complete_sha256": hashlib.sha256(direct.read_bytes()).hexdigest(),
    }
    marker_path = stage0 / "STAGE_COMPLETE.json"
    marker_path.write_bytes(canonical_json_bytes(marker))
    validate_stage_prerequisite(1, tmp_path)
    validate_cross_stage_source(1, tmp_path, source_hash)
    with pytest.raises(RuntimeError, match="differ across stages"):
        validate_cross_stage_source(1, tmp_path, "2" * 64)

    marker["source_snapshot_sha256"] = "0" * 64
    marker_path.write_bytes(canonical_json_bytes(marker))
    with pytest.raises(RuntimeError, match="source snapshot"):
        validate_stage_prerequisite(1, tmp_path)


def test_stage2_trigger_requires_oracle_and_half_open_median(tmp_path):
    path = tmp_path / "report.json"

    def write(oracle, median):
        path.write_bytes(
            canonical_json_bytes(
                {
                    "classification": {
                        "oracle": {"classification": oracle},
                        "P_n1": {
                            "components": {"g_vendi": {"median": median}}
                        },
                    }
                }
            )
        )

    write("pass", 0.90)
    validate_stage2_trigger(path)
    write("pass", 0.949999)
    validate_stage2_trigger(path)
    write("pass", 0.95)
    with pytest.raises(RuntimeError, match="trigger"):
        validate_stage2_trigger(path)
    write("fail", 0.92)
    with pytest.raises(RuntimeError, match="oracle"):
        validate_stage2_trigger(path)


def test_executor_rechecks_idleness_immediately_before_every_gpu_start(tmp_path):
    commands = (
        _command(tmp_path, "gpu_a", gpu_tokens=("GPU-a",)),
        _command(tmp_path, "cpu"),
        _command(tmp_path, "gpu_b", gpu_tokens=("GPU-b",)),
    )
    runtime = _FakeRuntime()
    execute_stage_commands(
        commands,
        stage_directory=tmp_path,
        env=_runtime_env(),
        runtime=runtime,
    )
    checked_tokens = [token for token, _ in runtime.idle_checks]
    assert checked_tokens.count("GPU-a") >= 2  # initial runtime gate + pre-start
    assert checked_tokens.count("GPU-b") >= 2
    assert [command.name for command in runtime.started] == ["gpu_a", "cpu", "gpu_b"]


def test_executor_skips_complete_matching_units_and_rejects_changed_contract(tmp_path):
    first = _command(tmp_path, "one", rows=2)
    runtime = _FakeRuntime()
    execute_stage_commands(
        (first,), stage_directory=tmp_path, env=_runtime_env(), runtime=runtime
    )
    assert [command.name for command in runtime.started] == ["one"]

    resumed = _FakeRuntime()
    execute_stage_commands(
        (first,), stage_directory=tmp_path, env=_runtime_env(), runtime=resumed
    )
    assert resumed.started == []

    first.completion_paths[0].write_bytes(b"tampered\n")
    with pytest.raises(RuntimeError, match="completion artifact"):
        execute_stage_commands(
            (first,),
            stage_directory=tmp_path,
            env=_runtime_env(),
            runtime=_FakeRuntime(),
        )
    first.completion_paths[0].write_bytes(
        canonical_json_bytes({"status": "complete", "name": "one"})
    )

    changed = StageCommand(**{**first.as_dict(), "row_count": 3})
    with pytest.raises(RuntimeError, match="contract"):
        execute_stage_commands(
            (changed,),
            stage_directory=tmp_path,
            env=_runtime_env(),
            runtime=_FakeRuntime(),
        )


def test_capture_incomplete_rollout_blocks_but_complete_rollout_resumes(tmp_path):
    capture = _command(tmp_path, "capture", gpu_tokens=("GPU-a",), output_kind="capture")
    rollout = capture.output_root / "rollout"
    rollout.mkdir(parents=True)
    (rollout / "partial.rollout.tmp").write_text("partial", encoding="utf-8")
    with pytest.raises(RuntimeError, match="incomplete rollout"):
        execute_stage_commands(
            (capture,),
            stage_directory=tmp_path,
            env=_runtime_env(),
            runtime=_FakeRuntime(),
        )
    assert not (capture.output_root / "COMPLETE.json").exists()

    (rollout / "partial.rollout.tmp").unlink()
    (rollout / "COMPLETE.json").write_text("complete\n", encoding="utf-8")
    runtime = _FakeRuntime()
    execute_stage_commands(
        (capture,), stage_directory=tmp_path, env=_runtime_env(), runtime=runtime
    )
    assert [command.name for command in runtime.started] == ["capture"]


def test_replay_contiguous_prefix_is_left_for_native_resume(tmp_path):
    replay = _command(tmp_path, "replay", gpu_tokens=("GPU-a",), output_kind="replay")
    replay.output_root.mkdir(parents=True)
    prefix = replay.output_root / "vectors_0_1.safetensors"
    prefix.write_bytes(b"immutable-prefix")
    runtime = _FakeRuntime()
    execute_stage_commands(
        (replay,), stage_directory=tmp_path, env=_runtime_env(), runtime=runtime
    )
    assert prefix.read_bytes() == b"immutable-prefix"


def test_seed42_failure_prevents_seed43_and_cleanup_is_owned_only(tmp_path):
    commands = (
        _command(tmp_path, "capture_target_seed_42", gpu_tokens=("GPU-a",)),
        _command(tmp_path, "capture_target_seed_43", gpu_tokens=("GPU-a",)),
    )
    runtime = _FakeRuntime(
        failures={"capture_target_seed_42": 9}, occupied={"GPU-a": {999}}
    )
    # The unrelated PID is declared occupied only after initial validation in this
    # focused cleanup test, so treat it as an already-owned external observation.
    with pytest.raises(RuntimeError, match="failed"):
        execute_stage_commands(
            commands,
            stage_directory=tmp_path,
            env=_runtime_env(),
            runtime=runtime,
            initially_owned_pids={999},
        )
    assert [command.name for command in runtime.started] == [
        "capture_target_seed_42"
    ]
    assert all(process.pid != 999 for process in runtime.processes)


def test_executor_waits_for_detached_gpu_child_to_drain(tmp_path):
    class DrainingRuntime(_FakeRuntime):
        def __init__(self):
            super().__init__()
            self.post_exit_queries = 0

        def gpu_processes(self, token):
            if self.processes and self.processes[-1]._done:
                self.post_exit_queries += 1
                if self.post_exit_queries <= 2:
                    return {90_001}
            return set()

        def sleep(self, seconds):
            self.clock += seconds

    runtime = DrainingRuntime()
    command = _command(tmp_path, "draining", gpu_tokens=("slot:0",))
    execute_stage_commands(
        (command,),
        stage_directory=tmp_path,
        env=_runtime_env(),
        runtime=runtime,
    )
    assert runtime.post_exit_queries == 3
    assert command.completion_paths[0].is_file()


def test_parallel_failure_terminates_only_sibling_processes(tmp_path):
    commands = (
        _command(
            tmp_path,
            "replay_0",
            gpu_tokens=("GPU-a",),
            group="replay_group",
        ),
        _command(
            tmp_path,
            "replay_1",
            gpu_tokens=("GPU-b",),
            group="replay_group",
        ),
    )
    runtime = _FakeRuntime(failures={"replay_0": 1})
    with pytest.raises(RuntimeError, match="replay_0"):
        execute_stage_commands(
            commands,
            stage_directory=tmp_path,
            env=_runtime_env(),
            runtime=runtime,
        )
    sibling = next(process for process in runtime.processes if process.command.name == "replay_1")
    assert sibling.terminated or sibling.poll() is not None


def test_duplicate_output_lock_is_rejected(tmp_path):
    lock = tmp_path / ".orchestrator.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="lock"):
            execute_stage_commands(
                (_command(tmp_path, "one"),),
                stage_directory=tmp_path,
                env=_runtime_env(),
                runtime=_FakeRuntime(),
            )
    finally:
        os.close(descriptor)


def test_executor_rechecks_every_live_source_file_before_launch(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    source_file = repository / "module.py"
    source_file.write_text("value = 1\n", encoding="utf-8")
    records = [
        {
            "repository": "main",
            "path": "module.py",
            "size": source_file.stat().st_size,
            "sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
        }
    ]
    source_hash = hashlib.sha256(canonical_json_bytes(records)).hexdigest()
    (tmp_path / "source_snapshot.json").write_bytes(
        canonical_json_bytes(
            {
                "repositories": {"main": str(repository.resolve())},
                "files": records,
                "manifest_sha256": source_hash,
            }
        )
    )
    command = _command(tmp_path, "source_bound")
    command = StageCommand(
        **{
            **command.as_dict(),
            "input_hashes": (("source_snapshot_sha256", source_hash),),
        }
    )
    execute_stage_commands(
        (command,),
        stage_directory=tmp_path,
        env=_runtime_env(),
        runtime=_FakeRuntime(),
    )

    source_file.write_text("value = 2\n", encoding="utf-8")
    changed = _command(tmp_path, "source_changed")
    changed = StageCommand(
        **{
            **changed.as_dict(),
            "input_hashes": (("source_snapshot_sha256", source_hash),),
        }
    )
    with pytest.raises(RuntimeError, match="source file changed"):
        execute_stage_commands(
            (changed,),
            stage_directory=tmp_path,
            env=_runtime_env(),
            runtime=_FakeRuntime(),
        )


def test_source_snapshot_is_ordered_complete_and_byte_sensitive(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "a.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repository / "b.sh").write_text("echo b\n", encoding="utf-8")
    first = build_source_snapshot(
        {"main": repository}, (("main", "b.sh"), ("main", "a.py"))
    )
    assert [record["path"] for record in first["files"]] == ["a.py", "b.sh"]
    (repository / "b.sh").write_text("echo changed\n", encoding="utf-8")
    second = build_source_snapshot(
        {"main": repository}, (("main", "b.sh"), ("main", "a.py"))
    )
    assert first["manifest_sha256"] != second["manifest_sha256"]
    removed = build_source_snapshot(
        {"main": repository}, (("main", "a.py"),)
    )
    assert removed["manifest_sha256"] != second["manifest_sha256"]
    with pytest.raises(ValueError, match="does not exist"):
        build_source_snapshot({"main": repository}, (("main", "missing.py"),))


def test_full_experiment_source_snapshot_covers_main_and_reference_sources():
    snapshot = build_experiment_source_snapshot(Path.cwd(), REFERENCE_REPO)
    assert set(snapshot["repositories"]) == {"main", "reference"}
    paths = {(row["repository"], row["path"]) for row in snapshot["files"]}
    assert ("main", "math_eval/run_opd_proxy_gradient_verify.py") in paths
    assert (
        "reference",
        "prismatic-synthesis/gradient_modules/gradient_computer.py",
    ) in paths


def test_gvendi_interpreter_imports_both_verl_and_cuda_projector_symbol():
    subprocess.run(
        [
            GVENDI_PYTHON,
            "-c",
            "import verl; from trak.projectors import CudaProjector; assert CudaProjector",
        ],
        check=True,
        env={**os.environ, "PYTHONPATH": "verl:."},
    )


def test_estimate_uses_slowest_stage0_per_row_rate_and_reserve(tmp_path):
    commands = (
        _command(tmp_path, "a", rows=10),
        _command(tmp_path, "b", rows=20),
    )
    result = estimate_remaining_seconds(
        timing_records=[
            {"duration_seconds": 5.0, "completed_rows": 10},
            {"duration_seconds": 20.0, "completed_rows": 10},
        ],
        commands=commands,
        safety_factor=1.2,
        reserve_seconds=100,
    )
    # Slowest = 2 sec/row; target rows = 30.
    assert result["work_seconds"] == pytest.approx(60.0)
    assert result["total_seconds"] == pytest.approx(172.0)


def test_slurm_end_time_parser_fails_closed():
    parsed = parse_slurm_end_time(
        "JobId=123 JobState=RUNNING EndTime=2026-07-20T12:34:56+00:00"
    )
    assert parsed.isoformat() == "2026-07-20T12:34:56+00:00"
    for output in ("JobId=123 EndTime=Unknown", "JobId=123", "EndTime=not-a-time"):
        with pytest.raises(RuntimeError, match="EndTime"):
            parse_slurm_end_time(output)


def test_parser_accepts_and_forwards_reference_repo_for_run_and_validate():
    run = parse_cli_args(
        ["--stage", "0", "--reference-repo", str(REFERENCE_REPO)]
    )
    assert run.stage == 0
    assert run.reference_repo == REFERENCE_REPO
    validate = parse_cli_args(
        [
            "validate-stage",
            "--stage",
            "0",
            "--preflight-only",
            "--reference-repo",
            str(REFERENCE_REPO),
        ]
    )
    assert validate.command == "validate-stage"
    assert validate.reference_repo == REFERENCE_REPO


def test_capture_launcher_has_no_forbidden_allocation_or_install_behavior():
    script = Path("verl/examples/fire_opd/run_capture_opd_proxy_verify.sh")
    text = script.read_text(encoding="utf-8")
    assert VERL_PYTHON in text
    assert "srun" not in text
    assert "pip install" not in text
    assert "conda install" not in text
    assert "CUDA_VISIBLE_DEVICES=" not in text
    subprocess.run(["bash", "-n", str(script)], check=True)
    direct = Path("verl/examples/fire_opd/run_direct_opd_proxy_gradient_fixture.sh")
    direct_text = direct.read_text(encoding="utf-8")
    assert GVENDI_PYTHON in direct_text
    assert "srun" not in direct_text
    subprocess.run(["bash", "-n", str(direct)], check=True)
