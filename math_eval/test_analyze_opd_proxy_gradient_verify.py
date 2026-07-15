import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from math_eval import analyze_opd_proxy_gradient_verify as analyzer_module
from math_eval.analyze_opd_proxy_gradient_verify import (
    render_markdown,
    run_analysis,
    write_analysis_input_manifest,
    write_reports,
)
from math_eval.opd_proxy_gradient_classification import classify_efficacy_pilot
from math_eval.opd_proxy_gradient_stage_profiles import EFFICACY_PILOT
from math_eval.opd_proxy_gradient_statistics import SubsetScores
from math_eval.opd_proxy_gradient_verify_artifacts import (
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json_bytes,
    load_vector_set,
    sha256_file,
    sha256_id_lines,
)
from math_eval.prepare_opd_proxy_gradient_verify import stage_layout
from math_eval.replay_opd_proxy_gradients import ReplayVector, run_replay_shard
from math_eval.select_opd_proxy_gradient_verify import (
    EXPECTED_REPRESENTATIONS,
    KMEANS_SEEDS,
    generate_random_schedules,
    run_selection,
)


REFERENCE_REPO = Path("/home/mchen/prismatic-synthesis-reference")
TEST_SOURCE_FILES = [{"path": "module.py", "sha256": "a" * 64}]
TEST_SOURCE_HASH = hashlib.sha256(
    canonical_json_bytes(TEST_SOURCE_FILES)
).hexdigest()


class _FakeClusterManager:
    @staticmethod
    def cluster_kmeans(data, k, num_iter, use_tqdm=False):
        del num_iter, use_tqdm
        labels = torch.arange(data.shape[0], device=data.device) % k
        return labels, torch.empty((k, data.shape[1]), device=data.device)


def _selection_vector_id(name: str, stable_id: str) -> str:
    if name == "P_pilot":
        return f"P_pilot:{stable_id}:seed=42"
    if name == "T_pilot":
        return f"T_pilot:{stable_id}:seed=42"
    if name.startswith("P_n1:"):
        seed = name.split(":")[1].split("=")[1]
        slot = name.split(":")[2].split("=")[1]
        return f"P:{stable_id}:seed={seed}:slot={slot}:n1"
    if name.startswith("P_n4:"):
        seed = name.rsplit("=", 1)[1]
        return f"P:{stable_id}:seed={seed}:n4"
    if name.startswith("T:"):
        seed = name.rsplit("=", 1)[1]
        return f"T:{stable_id}:seed={seed}:n4"
    return f"{name}:{stable_id}"


def _source_representation(name: str) -> str:
    if name in {"P_pilot", "T_pilot"}:
        return name
    if name.startswith("P_"):
        return "P"
    if name.startswith("T:"):
        return "T"
    return name


def _vectors(name: str, count: int, *, salt: int) -> np.ndarray:
    generator = np.random.Generator(np.random.PCG64(10_000 + salt))
    values = generator.normal(size=(count, 1024)).astype(np.float32)
    values[:, salt % 1024] += np.linspace(0.5, 2.0, count, dtype=np.float32)
    return values


def _write_vector_directory(
    directory: Path,
    *,
    name: str,
    stable_ids: list[str],
    vectors: np.ndarray,
    parent_hash: str,
    split: str,
    verifier: bool,
) -> None:
    representation = _source_representation(name)
    vector_ids = tuple(_selection_vector_id(name, stable_id) for stable_id in stable_ids)
    seed = int(name.rsplit("=", 1)[1]) if name.startswith("T:") else 0
    if name in {"P_pilot", "T_pilot"}:
        seed = 42
    elif name.startswith("P_"):
        seed = int(name.split(":")[1].split("=")[1])

    def factory(index: int) -> ReplayVector:
        vector = torch.from_numpy(vectors[index].copy())
        norm = torch.linalg.vector_norm(vector)
        return ReplayVector(
            vector_id=vector_ids[index],
            stable_id=stable_ids[index],
            split=split,
            representation=representation,
            engine_seed=seed,
            rollout_slot=None,
            aggregation="synthetic",
            source_capture_sha256=parent_hash,
            projected_gradient=vector,
            full_gradient_norm=torch.tensor(float(index + 1)),
            projected_gradient_norm=norm,
            valid_token_count=torch.tensor(float(10 + index)),
            response_length=torch.tensor(float(20 + index)),
            sampled_reverse_kl=torch.tensor(float(index) / 100.0 - 0.05),
            opd_signal_rms=torch.tensor(float(index + 1) / 10.0),
            verifier_correct_count=(
                torch.tensor(float(index % 5)) if verifier else None
            ),
            verifier_total=torch.tensor(4.0) if verifier else None,
        )

    run_replay_shard(
        output_directory=directory,
        expected_vector_ids=vector_ids,
        record_factory=factory,
        representation=representation,
        parent_hashes={
            "sample_manifest_sha256": parent_hash,
            "source_snapshot_sha256": TEST_SOURCE_HASH,
        },
        source_snapshot={"manifest_sha256": TEST_SOURCE_HASH},
        repository={"head": "c" * 40, "status": "", "status_sha256": "d" * 64},
        runtime={"runtime_profile": "synthetic_test"},
        metadata={"selection_name": name},
        chunk_size=32,
        verifier_status="computed" if verifier else "not_computed",
    )


def _sample_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(24):
        rows.append(
            {
                "stable_id": f"q{index:03d}",
                "split": "candidate",
                "manifest_index": index,
                "leaf_topic": f"topic-{index % 4}",
                "prompt_token_count_4b": 20 + index,
                "prompt_token_count_0_6b": 18 + index,
                "r1_completion_token_count_0_6b": 30 + index,
                "sft_full_token_count_0_6b": 50 + index,
                "sft_supervised_label_count": 29 + index,
            }
        )
    for index in range(8):
        rows.append(
            {
                "stable_id": f"h{index:03d}",
                "split": "held_out",
                "manifest_index": 24 + index,
                "leaf_topic": f"held-{index % 2}",
                "prompt_token_count_4b": 25 + index,
                "prompt_token_count_0_6b": 22 + index,
                "r1_completion_token_count_0_6b": 35 + index,
                "sft_full_token_count_0_6b": 55 + index,
                "sft_supervised_label_count": 34 + index,
            }
        )
    return rows


def _build_synthetic_stage(root: Path) -> Path:
    root.mkdir(parents=True)
    rows = _sample_rows()
    sample_path = root / "sample_manifest.jsonl"
    atomic_write_jsonl(sample_path, rows)
    candidate_ids = [str(row["stable_id"]) for row in rows[:24]]
    heldout_ids = [str(row["stable_id"]) for row in rows[24:]]
    layout = stage_layout(0)
    stage_manifest = {
        "schema_version": 1,
        "artifact_type": "opd_proxy_stage",
        "stage": 0,
        "candidate_count": 24,
        "held_out_count": 8,
        "selected_size": layout.selected_size,
        "primary_k": layout.primary_k,
        "diagnostic_k": layout.diagnostic_k,
        "null_draws": layout.null_draws,
        "candidate_ids_sha256": sha256_id_lines(candidate_ids),
        "held_out_ids_sha256": sha256_id_lines(heldout_ids),
        "all_ids_sha256": sha256_id_lines(candidate_ids + heldout_ids),
        "sample_manifest": sample_path.name,
        "sample_manifest_sha256": sha256_file(sample_path),
        "parent_report": None,
        "provenance": {"synthetic_test_fixture": True},
    }
    atomic_write_json(root / "manifest.json", stage_manifest)
    atomic_write_json(
        root / "source_snapshot.json",
        {"files": TEST_SOURCE_FILES, "manifest_sha256": TEST_SOURCE_HASH},
    )
    stage_hash = sha256_file(root / "manifest.json")
    sample_hash = sha256_file(sample_path)

    vector_root = root / "analysis_vectors"
    representations: dict[str, dict[str, str]] = {}
    loaded = {}
    for salt, name in enumerate(EXPECTED_REPRESENTATIONS):
        directory = vector_root / f"representation_{salt:02d}"
        values = _vectors(name, 24, salt=salt + 1)
        _write_vector_directory(
            directory,
            name=name,
            stable_ids=candidate_ids,
            vectors=values,
            parent_hash=sample_hash,
            split="candidate",
            verifier=name.startswith("T:"),
        )
        manifest_hash = sha256_file(directory / "manifest.json")
        representations[name] = {
            "vector_directory": directory.relative_to(root).as_posix(),
            "vector_manifest_sha256": manifest_hash,
        }
        loaded[name] = load_vector_set(directory)

    heldout_records: dict[str, dict[str, object]] = {}
    for seed in (42, 43):
        name = f"T:seed={seed}"
        values = _vectors(name, 8, salt=100 + seed)
        if seed == 42:
            directories = []
            hashes = []
            for shard_index, bounds in enumerate(((0, 4), (4, 8))):
                start, end = bounds
                directory = vector_root / f"heldout_seed_{seed}_shard_{shard_index}"
                _write_vector_directory(
                    directory,
                    name=name,
                    stable_ids=heldout_ids[start:end],
                    vectors=values[start:end],
                    parent_hash=sample_hash,
                    split="held_out",
                    verifier=True,
                )
                directories.append(directory.relative_to(root).as_posix())
                hashes.append(sha256_file(directory / "manifest.json"))
            heldout_records[str(seed)] = {
                "vector_directories": directories,
                "vector_manifest_sha256": hashes,
            }
        else:
            directory = vector_root / f"heldout_seed_{seed}"
            _write_vector_directory(
                directory,
                name=name,
                stable_ids=heldout_ids,
                vectors=values,
                parent_hash=sample_hash,
                split="held_out",
                verifier=True,
            )
            heldout_records[str(seed)] = {
                "vector_directory": directory.relative_to(root).as_posix(),
                "vector_manifest_sha256": sha256_file(directory / "manifest.json"),
            }

    selection_dir = root / "selection"
    vector_hashes = {
        name: hashlib.sha256(canonical_json_bytes(vector.manifest)).hexdigest()
        for name, vector in loaded.items()
    }
    run_selection(
        loaded,
        stage=0,
        candidate_rows=rows[:24],
        output_directory=selection_dir,
        parent_hashes={
            "source_snapshot_sha256": TEST_SOURCE_HASH,
            "stage_manifest_sha256": stage_hash,
        },
        expected_vector_manifest_hashes=vector_hashes,
        fake_cluster_manager=_FakeClusterManager,
    )
    generate_random_schedules(
        rows[:24],
        selected_size=5,
        draws=100,
        output_directory=selection_dir,
        parent_hashes={
            "source_snapshot_sha256": TEST_SOURCE_HASH,
            "stage_manifest_sha256": stage_hash,
        },
        stage=0,
    )
    inputs = {
        "schema_version": 1,
        "artifact_type": "opd_proxy_analysis_inputs",
        "stage_manifest_sha256": stage_hash,
        "source_snapshot_sha256": TEST_SOURCE_HASH,
        "selection_directory": "selection",
        "selection_manifest_sha256": sha256_file(
            selection_dir / "selection.manifest.json"
        ),
        "random_manifest_sha256": sha256_file(
            selection_dir / "random.manifest.json"
        ),
        "representations": representations,
        "target_heldout": heldout_records,
    }
    atomic_write_json(root / "analysis_inputs.json", inputs)
    return root


def _pilot_sample_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(250):
        rows.append(
            {
                "stable_id": f"pc{index:03d}",
                "split": "candidate",
                "manifest_index": index,
                "parent_manifest_index": index,
                "parent_row_sha256": hashlib.sha256(
                    f"candidate-{index}".encode()
                ).hexdigest(),
                "leaf_topic": f"topic-{index % 5}",
                "prompt_token_count_4b": 20 + index,
                "prompt_token_count_0_6b": 18 + index,
                "r1_completion_token_count_0_6b": 30 + index,
                "sft_full_token_count_0_6b": 50 + index,
                "sft_supervised_label_count": 29 + index,
            }
        )
    for index in range(84):
        rows.append(
            {
                "stable_id": f"ph{index:03d}",
                "split": "held_out",
                "manifest_index": 250 + index,
                "parent_manifest_index": 768 + index,
                "parent_row_sha256": hashlib.sha256(
                    f"heldout-{index}".encode()
                ).hexdigest(),
                "leaf_topic": f"held-{index % 3}",
                "prompt_token_count_4b": 25 + index,
                "prompt_token_count_0_6b": 22 + index,
                "r1_completion_token_count_0_6b": 35 + index,
                "sft_full_token_count_0_6b": 55 + index,
                "sft_supervised_label_count": 34 + index,
            }
        )
    return rows


def _build_synthetic_pilot(root: Path) -> Path:
    root.mkdir(parents=True)
    rows = _pilot_sample_rows()
    sample_path = root / "sample_manifest.jsonl"
    atomic_write_jsonl(sample_path, rows)
    candidate_ids = [str(row["stable_id"]) for row in rows[:250]]
    heldout_ids = [str(row["stable_id"]) for row in rows[250:]]
    layout = stage_layout(EFFICACY_PILOT)
    parent_sha = "e" * 64
    stage_manifest = {
        "schema_version": 1,
        "artifact_type": "opd_proxy_stage",
        "stage": EFFICACY_PILOT,
        "candidate_count": 250,
        "held_out_count": 84,
        "selected_size": layout.selected_size,
        "primary_k": layout.primary_k,
        "diagnostic_k": layout.diagnostic_k,
        "null_draws": layout.null_draws,
        "candidate_ids_sha256": sha256_id_lines(candidate_ids),
        "held_out_ids_sha256": sha256_id_lines(heldout_ids),
        "all_ids_sha256": sha256_id_lines(candidate_ids + heldout_ids),
        "sample_manifest": sample_path.name,
        "sample_manifest_sha256": sha256_file(sample_path),
        "parent_report": None,
        "parent_stage_manifest": {
            "path": "/synthetic/stage_1/manifest.json",
            "sha256": parent_sha,
        },
        "algorithm_contract_sha256": "f" * 64,
        "provenance": {"synthetic_test_fixture": True},
    }
    atomic_write_json(root / "manifest.json", stage_manifest)
    atomic_write_json(
        root / "source_snapshot.json",
        {"files": TEST_SOURCE_FILES, "manifest_sha256": TEST_SOURCE_HASH},
    )
    stage_hash = sha256_file(root / "manifest.json")
    sample_hash = sha256_file(sample_path)

    vector_root = root / "analysis_vectors"
    representations: dict[str, dict[str, str]] = {}
    loaded: dict[str, object] = {}
    for salt, name in enumerate(("P_pilot", "T_pilot"), start=1):
        directory = vector_root / name
        _write_vector_directory(
            directory,
            name=name,
            stable_ids=candidate_ids,
            vectors=_vectors(name, 250, salt=200 + salt),
            parent_hash=sample_hash,
            split="candidate",
            verifier=False,
        )
        representations[name] = {
            "vector_directory": directory.relative_to(root).as_posix(),
            "vector_manifest_sha256": sha256_file(directory / "manifest.json"),
        }
        loaded[name] = load_vector_set(directory)

    heldout_directory = vector_root / "T_pilot_heldout"
    _write_vector_directory(
        heldout_directory,
        name="T_pilot",
        stable_ids=heldout_ids,
        vectors=_vectors("T_pilot", 84, salt=242),
        parent_hash=sample_hash,
        split="held_out",
        verifier=False,
    )
    selection_dir = root / "selection"
    proxy = loaded["P_pilot"]
    proxy_hash = hashlib.sha256(canonical_json_bytes(proxy.manifest)).hexdigest()
    run_selection(
        {"P_pilot": proxy},
        stage=EFFICACY_PILOT,
        candidate_rows=rows[:250],
        output_directory=selection_dir,
        parent_hashes={
            "source_snapshot_sha256": TEST_SOURCE_HASH,
            "stage_manifest_sha256": stage_hash,
        },
        expected_vector_manifest_hashes={"P_pilot": proxy_hash},
        fake_cluster_manager=_FakeClusterManager,
    )
    generate_random_schedules(
        rows[:250],
        selected_size=56,
        draws=10_000,
        output_directory=selection_dir,
        parent_hashes={
            "source_snapshot_sha256": TEST_SOURCE_HASH,
            "stage_manifest_sha256": stage_hash,
        },
        stage=EFFICACY_PILOT,
    )
    inputs = {
        "schema_version": 1,
        "artifact_type": "opd_proxy_analysis_inputs",
        "stage_manifest_sha256": stage_hash,
        "source_snapshot_sha256": TEST_SOURCE_HASH,
        "selection_directory": "selection",
        "selection_manifest_sha256": sha256_file(
            selection_dir / "selection.manifest.json"
        ),
        "random_manifest_sha256": sha256_file(
            selection_dir / "random.manifest.json"
        ),
        "representations": representations,
        "target_heldout": {
            "42": {
                "vector_directory": heldout_directory.relative_to(root).as_posix(),
                "vector_manifest_sha256": sha256_file(
                    heldout_directory / "manifest.json"
                ),
            }
        },
    }
    atomic_write_json(root / "analysis_inputs.json", inputs)
    return root


@pytest.fixture(scope="module")
def synthetic_stage0_dir(tmp_path_factory):
    return _build_synthetic_stage(tmp_path_factory.mktemp("opd-analyzer") / "stage_0")


@pytest.fixture(scope="module")
def synthetic_report(synthetic_stage0_dir):
    return run_analysis(stage_dir=synthetic_stage0_dir, reference_repo=REFERENCE_REPO)


@pytest.fixture(scope="module")
def synthetic_pilot_dir(tmp_path_factory):
    return _build_synthetic_pilot(
        tmp_path_factory.mktemp("opd-pilot-analyzer") / EFFICACY_PILOT
    )


def _patch_pilot_statistics(monkeypatch):
    table_calls = 0
    subset_calls = 0

    def fake_table(space, subsets):
        nonlocal table_calls
        del space
        table_calls += 1
        # Uniform is the first table; make the stratified diagnostic maximally bad.
        values = (
            np.linspace(0.0, 1.0, len(subsets), dtype=np.float64)
            if table_calls == 1
            else np.ones(len(subsets), dtype=np.float64)
        )
        return {
            "g_vendi": values,
            "coverage": values,
            "full_gradient_norm": values,
            "opd_signal_rms": values,
            "valid_token_count": values,
            "sampled_reverse_kl": values,
            "response_length": values,
        }

    def fake_subset(space, positions):
        nonlocal subset_calls
        del space, positions
        subset_calls += 1
        # Primary K runs are first. Diagnostic-K runs deliberately fail the gate.
        quality = 0.95 if subset_calls <= len(KMEANS_SEEDS) else 0.10
        signal = 0.50 if subset_calls <= len(KMEANS_SEEDS) else 0.10
        return SubsetScores(
            g_vendi=quality,
            coverage=quality,
            full_gradient_norm=signal,
            opd_signal_rms=signal,
            valid_token_count=signal,
            sampled_reverse_kl=signal,
            response_length=signal,
        )

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("single-seed pilot must not compute cross-seed diagnostics")

    monkeypatch.setattr(analyzer_module, "evaluate_subset_table", fake_table)
    monkeypatch.setattr(analyzer_module, "evaluate_subset", fake_subset)
    monkeypatch.setattr(analyzer_module, "target_dependence_permutation_test", forbidden)
    monkeypatch.setattr(analyzer_module, "_secondary_diagnostics", forbidden)


def test_pilot_report_is_forced_pilot_only_with_unavailable_cross_seed_diagnostics(
    synthetic_pilot_dir, monkeypatch, tmp_path
):
    _patch_pilot_statistics(monkeypatch)
    seen = {}

    def gate(records):
        seen.update(records)
        return classify_efficacy_pilot(records)

    monkeypatch.setattr(analyzer_module, "classify_efficacy_pilot", gate)
    report = run_analysis(
        stage_dir=synthetic_pilot_dir, reference_repo=REFERENCE_REPO
    )

    assert report["stage"] == EFFICACY_PILOT
    assert report["counts"] == {
        "candidate": 250,
        "held_out": 84,
        "selected": 56,
        "primary_k": 25,
        "diagnostic_k": 3,
        "random_draws": 10_000,
        "representations": 2,
    }
    assert report["classification"]["status"] == "pilot_only"
    assert report["classification"]["decision"] == "go"
    assert report["classification"]["main_hypothesis"] == "not_evaluated"
    assert set(seen) == {42, 43}
    assert all(set(row) == {
        "g_vendi", "coverage", "gradient_norm", "opd_signal"
    } for row in seen.values())
    unavailable = report["unavailable_diagnostics"]
    assert unavailable["cross_seed_oracle"]["status"] == "unavailable"
    assert unavailable["target_seed_dependence"]["status"] == "unavailable"
    assert "one generation seed and one rollout" in unavailable[
        "cross_seed_oracle"
    ]["reason"]
    assert report["provenance"]["parent_stage1_manifest_sha256"] == "e" * 64

    json_path = tmp_path / "pilot-report.json"
    markdown_path = tmp_path / "pilot-report.md"
    write_reports(report, json_path, markdown_path)
    assert json.loads(json_path.read_text()) == report
    markdown = markdown_path.read_text()
    assert "One-Time Efficacy Pilot" in markdown
    assert "does not establish downstream or OOD improvement" in markdown
    assert "main Stage-1 hypothesis was not evaluated" in markdown


def test_pilot_analyzer_rejects_any_non_pilot_classification(
    synthetic_pilot_dir, monkeypatch
):
    _patch_pilot_statistics(monkeypatch)
    monkeypatch.setattr(
        analyzer_module,
        "classify_efficacy_pilot",
        lambda records: {"status": "pass", "decision": "go"},
    )
    with pytest.raises(ValueError, match="pilot_only"):
        run_analysis(stage_dir=synthetic_pilot_dir, reference_repo=REFERENCE_REPO)

    monkeypatch.setattr(
        analyzer_module,
        "classify_efficacy_pilot",
        lambda records: {
            **classify_efficacy_pilot(records),
            "main_hypothesis_result": "pass",
        },
    )
    with pytest.raises(ValueError, match="pilot_only"):
        run_analysis(stage_dir=synthetic_pilot_dir, reference_repo=REFERENCE_REPO)


@pytest.mark.parametrize("mutation", ["extra_representation", "extra_target_seed"])
def test_pilot_analyzer_rejects_non_pilot_analysis_inputs(
    synthetic_pilot_dir, monkeypatch, tmp_path, mutation
):
    _patch_pilot_statistics(monkeypatch)
    bad = _copy_stage(synthetic_pilot_dir, tmp_path / mutation)
    inputs_path = bad / "analysis_inputs.json"
    inputs = json.loads(inputs_path.read_text())
    if mutation == "extra_representation":
        inputs["representations"]["S"] = dict(inputs["representations"]["P_pilot"])
    else:
        inputs["target_heldout"]["43"] = dict(inputs["target_heldout"]["42"])
    atomic_write_json(inputs_path, inputs)
    with pytest.raises(ValueError, match="incomplete|unexpected|stage seeds"):
        run_analysis(stage_dir=bad, reference_repo=REFERENCE_REPO)


def _copy_stage(source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination)
    return destination


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_analysis_input_manifest_builder_binds_direct_and_sharded_sources(
    synthetic_stage0_dir, tmp_path
):
    stage = _copy_stage(synthetic_stage0_dir, tmp_path / "rebuilt-inputs")
    original_path = stage / "analysis_inputs.json"
    original = _load(original_path)
    original_path.unlink()

    def paths(record):
        if "vector_directory" in record:
            return [stage / record["vector_directory"]]
        return [stage / value for value in record["vector_directories"]]

    rebuilt = write_analysis_input_manifest(
        stage_dir=stage,
        selection_directory=stage / original["selection_directory"],
        representation_sources={
            name: paths(original["representations"][name])
            for name in EXPECTED_REPRESENTATIONS
        },
        target_heldout_sources={
            seed: paths(original["target_heldout"][str(seed)])
            for seed in (42, 43)
        },
    )
    assert rebuilt == original
    assert (stage / "analysis_inputs.json").read_bytes() == canonical_json_bytes(
        original
    )


def test_analysis_input_manifest_allows_only_exact_stage1_sibling_sources(
    synthetic_stage0_dir, tmp_path
):
    experiment = tmp_path / "experiment"
    stage2 = _copy_stage(synthetic_stage0_dir, experiment / "stage_2")
    original = _load(stage2 / "analysis_inputs.json")
    (stage2 / "analysis_inputs.json").unlink()
    first_name = EXPECTED_REPRESENTATIONS[0]
    source = stage2 / original["representations"][first_name]["vector_directory"]
    stage1_source = experiment / "stage_1" / "prior_vector"
    stage1_source.parent.mkdir(parents=True)
    shutil.copytree(source, stage1_source)

    def paths(record):
        if "vector_directory" in record:
            return [stage2 / record["vector_directory"]]
        return [stage2 / value for value in record["vector_directories"]]

    representation_sources = {
        name: paths(original["representations"][name])
        for name in EXPECTED_REPRESENTATIONS
    }
    representation_sources[first_name] = [stage1_source]
    rebuilt = write_analysis_input_manifest(
        stage_dir=stage2,
        selection_directory=stage2 / original["selection_directory"],
        representation_sources=representation_sources,
        target_heldout_sources={
            seed: paths(original["target_heldout"][str(seed)])
            for seed in (42, 43)
        },
    )
    assert rebuilt["representations"][first_name]["vector_directory"] == (
        "../stage_1/prior_vector"
    )

    outside = tmp_path / "outside"
    shutil.copytree(source, outside)
    (stage2 / "analysis_inputs.json").unlink()
    representation_sources[first_name] = [outside]
    with pytest.raises(ValueError, match="Stage-1 sibling"):
        write_analysis_input_manifest(
            stage_dir=stage2,
            selection_directory=stage2 / original["selection_directory"],
            representation_sources=representation_sources,
            target_heldout_sources={
                seed: paths(original["target_heldout"][str(seed)])
                for seed in (42, 43)
            },
        )


def test_json_and_markdown_share_one_report_dict(synthetic_report, tmp_path):
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"
    write_reports(synthetic_report, json_path, md_path)
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert render_markdown(loaded) == md_path.read_text(encoding="utf-8")
    assert "NaN" not in json_path.read_text(encoding="utf-8")
    assert "Infinity" not in json_path.read_text(encoding="utf-8")


def test_report_pair_rollback_preserves_both_prior_files(
    synthetic_report, tmp_path, monkeypatch
):
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"
    json_path.write_bytes(b"old-json\n")
    md_path.write_bytes(b"old-markdown\n")
    real_replace = analyzer_module.os.replace
    calls = 0

    def fail_second_replace(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic second-publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr(analyzer_module.os, "replace", fail_second_replace)
    with pytest.raises(OSError, match="second-publication"):
        write_reports(synthetic_report, json_path, md_path)
    assert json_path.read_bytes() == b"old-json\n"
    assert md_path.read_bytes() == b"old-markdown\n"


def test_stage0_is_smoke_only(synthetic_report):
    assert synthetic_report["classification"] == {"status": "smoke_only"}
    assert "pre_registered_gate_diagnostics" in synthetic_report


def test_analysis_is_byte_stable_and_has_all_predeclared_secondary_records(
    synthetic_stage0_dir,
):
    first = run_analysis(stage_dir=synthetic_stage0_dir, reference_repo=REFERENCE_REPO)
    second = run_analysis(stage_dir=synthetic_stage0_dir, reference_repo=REFERENCE_REPO)
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    secondary = first["secondary_diagnostics"]
    assert len(secondary["cka_pairs"]) == 53
    assert len(secondary["ami_records"]) == 212
    assert len(secondary["ari_records"]) == 212
    assert len(secondary["primary_selected_overlap"]) == 292


def test_oracle_gate_uses_only_opposing_target_seed(synthetic_report):
    runs = synthetic_report["pre_registered_gate_diagnostics"]["oracle"][
        "cross_seed_runs"
    ]
    assert len(runs) == 4
    assert all(
        run["selection_target_seed"] != run["evaluation_target_seed"]
        for run in runs
    )
    assert {
        (run["selection_target_seed"], run["evaluation_target_seed"])
        for run in runs
    } == {(42, 43), (43, 42)}


def test_optional_verifier_diagnostics_are_structured(synthetic_report):
    embedding = next(
        record
        for record in synthetic_report["selected_diagnostics"]
        if record["representation"] == "E"
    )
    assert embedding["source_vector"]["verifier"] == {
        "status": "not_computed"
    }
    target_random = synthetic_report["random_nulls"]["target_seeds"]["42"][
        "uniform"
    ]["verifier"]
    assert target_random["status"] == "computed"
    assert target_random["rate"]["std"] >= 0.0


def test_report_contains_paired_uniform_and_stratified_nulls_for_both_targets(
    synthetic_report,
):
    random_nulls = synthetic_report["random_nulls"]
    assert set(random_nulls["target_seeds"]) == {"42", "43"}
    assert random_nulls["schedule"]["paired_target_seeds"] == [42, 43]
    for seed in ("42", "43"):
        assert set(random_nulls["target_seeds"][seed]) == {"uniform", "stratified"}


def test_missing_representation_is_rejected(synthetic_stage0_dir, tmp_path):
    stage = _copy_stage(synthetic_stage0_dir, tmp_path / "missing-representation")
    inputs_path = stage / "analysis_inputs.json"
    inputs = _load(inputs_path)
    inputs["representations"].pop("E")
    inputs_path.write_bytes(canonical_json_bytes(inputs))
    with pytest.raises(ValueError, match="representations"):
        run_analysis(stage_dir=stage, reference_repo=REFERENCE_REPO)


def test_analysis_failure_preserves_both_prior_reports(synthetic_stage0_dir, tmp_path):
    stage = _copy_stage(synthetic_stage0_dir, tmp_path / "failed-analysis")
    inputs_path = stage / "analysis_inputs.json"
    inputs = _load(inputs_path)
    inputs["stage_manifest_sha256"] = "0" * 64
    inputs_path.write_bytes(canonical_json_bytes(inputs))
    json_path = tmp_path / "prior-report.json"
    markdown_path = tmp_path / "prior-report.md"
    json_path.write_bytes(b"prior-json\n")
    markdown_path.write_bytes(b"prior-markdown\n")
    with pytest.raises(ValueError, match="stage manifest SHA"):
        run_analysis(
            stage_dir=stage,
            reference_repo=REFERENCE_REPO,
            output_json=json_path,
            output_markdown=markdown_path,
        )
    assert json_path.read_bytes() == b"prior-json\n"
    assert markdown_path.read_bytes() == b"prior-markdown\n"


def test_changed_parent_hash_is_rejected(synthetic_stage0_dir, tmp_path):
    stage = _copy_stage(synthetic_stage0_dir, tmp_path / "wrong-parent")
    inputs_path = stage / "analysis_inputs.json"
    inputs = _load(inputs_path)
    inputs["stage_manifest_sha256"] = "0" * 64
    inputs_path.write_bytes(canonical_json_bytes(inputs))
    with pytest.raises(ValueError, match="stage manifest SHA"):
        run_analysis(stage_dir=stage, reference_repo=REFERENCE_REPO)


def test_analysis_rejects_source_snapshot_parent_mismatch(
    synthetic_stage0_dir, tmp_path
):
    stage = _copy_stage(synthetic_stage0_dir, tmp_path / "wrong-source")
    inputs_path = stage / "analysis_inputs.json"
    inputs = _load(inputs_path)
    inputs["source_snapshot_sha256"] = "0" * 64
    inputs_path.write_bytes(canonical_json_bytes(inputs))
    with pytest.raises(ValueError, match="source snapshot"):
        run_analysis(stage_dir=stage, reference_repo=REFERENCE_REPO)


def test_missing_k_ratio_or_seed_run_is_rejected(synthetic_stage0_dir, tmp_path):
    stage = _copy_stage(synthetic_stage0_dir, tmp_path / "missing-run")
    selection_path = stage / "selection" / "selection.manifest.json"
    selection = _load(selection_path)
    selection["runs"].pop()
    selection_path.write_bytes(canonical_json_bytes(selection))
    inputs_path = stage / "analysis_inputs.json"
    inputs = _load(inputs_path)
    inputs["selection_manifest_sha256"] = sha256_file(selection_path)
    inputs_path.write_bytes(canonical_json_bytes(inputs))
    with pytest.raises(ValueError, match="run coverage"):
        run_analysis(stage_dir=stage, reference_repo=REFERENCE_REPO)


def test_random_dtype_shape_and_hash_are_strict(synthetic_stage0_dir, tmp_path):
    stage = _copy_stage(synthetic_stage0_dir, tmp_path / "wrong-random")
    random_path = stage / "selection" / "random_uniform.npy"
    original = np.load(random_path, allow_pickle=False)
    with random_path.open("wb") as handle:
        np.save(handle, original.astype(np.int64), allow_pickle=False)
    random_manifest_path = stage / "selection" / "random.manifest.json"
    random_manifest = _load(random_manifest_path)
    random_manifest["files"]["uniform"]["sha256"] = sha256_file(random_path)
    random_manifest_path.write_bytes(canonical_json_bytes(random_manifest))
    inputs_path = stage / "analysis_inputs.json"
    inputs = _load(inputs_path)
    inputs["random_manifest_sha256"] = sha256_file(random_manifest_path)
    inputs_path.write_bytes(canonical_json_bytes(inputs))
    with pytest.raises(ValueError, match="dtype"):
        run_analysis(stage_dir=stage, reference_repo=REFERENCE_REPO)


def test_differing_candidate_id_order_is_rejected(synthetic_stage0_dir, tmp_path):
    stage = _copy_stage(synthetic_stage0_dir, tmp_path / "wrong-id-order")
    sample_path = stage / "sample_manifest.jsonl"
    rows = [json.loads(line) for line in sample_path.read_text().splitlines()]
    rows[0]["stable_id"], rows[1]["stable_id"] = rows[1]["stable_id"], rows[0]["stable_id"]
    sample_path.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))
    stage_manifest_path = stage / "manifest.json"
    stage_manifest = _load(stage_manifest_path)
    stage_manifest["sample_manifest_sha256"] = sha256_file(sample_path)
    stage_manifest["candidate_ids_sha256"] = sha256_id_lines(
        [str(row["stable_id"]) for row in rows[:24]]
    )
    stage_manifest["all_ids_sha256"] = sha256_id_lines(
        [str(row["stable_id"]) for row in rows]
    )
    stage_manifest_path.write_bytes(canonical_json_bytes(stage_manifest))
    inputs_path = stage / "analysis_inputs.json"
    inputs = _load(inputs_path)
    inputs["stage_manifest_sha256"] = sha256_file(stage_manifest_path)
    inputs_path.write_bytes(canonical_json_bytes(inputs))
    with pytest.raises(ValueError, match="parent|candidate|stable ID"):
        run_analysis(stage_dir=stage, reference_repo=REFERENCE_REPO)


def test_stage2_rejects_stage1_cardinality_leak_before_loading_vectors(
    synthetic_stage0_dir, tmp_path
):
    stage = _copy_stage(synthetic_stage0_dir, tmp_path / "stage2-leak")
    stage_manifest_path = stage / "manifest.json"
    manifest = _load(stage_manifest_path)
    manifest["stage"] = 2
    # Deliberately retain the Stage-0/1-like cardinalities.
    stage_manifest_path.write_bytes(canonical_json_bytes(manifest))
    inputs_path = stage / "analysis_inputs.json"
    inputs = _load(inputs_path)
    inputs["stage_manifest_sha256"] = sha256_file(stage_manifest_path)
    inputs_path.write_bytes(canonical_json_bytes(inputs))
    with pytest.raises(ValueError, match="stage layout"):
        run_analysis(stage_dir=stage, reference_repo=REFERENCE_REPO)
