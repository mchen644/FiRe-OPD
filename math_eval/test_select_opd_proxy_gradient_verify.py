import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from math_eval.opd_proxy_gradient_verify_artifacts import (
    VectorSet,
    canonical_json_bytes,
    sha256_id_lines,
    sha256_int_rows,
)
from math_eval.select_opd_proxy_gradient_verify import (
    EXPECTED_REPRESENTATIONS,
    KMEANS_SEEDS,
    ROUND_ROBIN_SEED,
    _parse_vector_specs,
    build_length_quartiles,
    build_selection_vector_view,
    generate_random_schedules,
    largest_remainder_allocation,
    load_random_schedules,
    load_selection_bundle,
    run_selection,
    selection_vector_id,
)
from math_eval.select_gradient_diverse_deepmath import cluster_official


def test_vector_spec_parser_preserves_equals_in_representation_names(tmp_path):
    values = [
        f"{name}={tmp_path / f'vector-{index}'}"
        for index, name in enumerate(EXPECTED_REPRESENTATIONS)
    ]
    specs = _parse_vector_specs(values)
    assert tuple(specs) == EXPECTED_REPRESENTATIONS
    assert specs["P_n1:seed=42:slot=0"] == [tmp_path / "vector-0"]
    assert specs["T:seed=43"] == [
        tmp_path / f"vector-{len(EXPECTED_REPRESENTATIONS) - 1}"
    ]


def _rows(count: int) -> list[dict[str, object]]:
    return [
        {
            "stable_id": f"q{index:04d}",
            "split": "candidate",
            "leaf_topic": f"topic-{index % 3}",
            "prompt_token_count_0_6b": 10 + (index % 11),
        }
        for index in range(count)
    ]


def _vector_set(
    name: str,
    rows: list[dict[str, object]],
    *,
    width: int = 4,
    stage_parent: str = "a" * 64,
) -> VectorSet:
    stable_ids = tuple(str(row["stable_id"]) for row in rows)
    offset = sum(name.encode("utf-8")) % 13 + 1
    base = np.arange(1, len(rows) * width + 1, dtype=np.float32).reshape(
        len(rows), width
    )
    vectors = np.ascontiguousarray(base + np.float32(offset))
    vector_ids = tuple(f"{name}:{stable_id}" for stable_id in stable_ids)
    return VectorSet(
        vector_ids=vector_ids,
        stable_ids=stable_ids,
        vectors=vectors,
        full_gradient_norm=None,
        projected_gradient_norm=None,
        valid_token_count=None,
        response_length=None,
        sampled_reverse_kl=None,
        opd_signal_rms=None,
        verifier_correct_count=None,
        verifier_total=None,
        manifest={
            "schema_version": 1,
            "artifact_type": "vector_set",
            "representation": name,
            "vector_count": len(rows),
            "vector_dimension": width,
            "vector_ids_sha256": sha256_id_lines(vector_ids),
            "stable_ids_sha256": sha256_id_lines(stable_ids),
            "parent_hashes": {"stage_manifest_sha256": stage_parent},
        },
    )


def _vectors(count: int) -> tuple[list[dict[str, object]], dict[str, VectorSet]]:
    rows = _rows(count)
    return rows, {
        name: _vector_set(name, rows) for name in EXPECTED_REPRESENTATIONS
    }


class _FakeClusterManager:
    calls: list[tuple[torch.Tensor, int, int, bool]] = []

    @staticmethod
    def cluster_kmeans(data, k, num_iter, use_tqdm=False):
        _FakeClusterManager.calls.append((data.clone(), k, num_iter, use_tqdm))
        labels = torch.arange(data.shape[0], device=data.device) % k
        return labels, torch.empty((k, data.shape[1]), device=data.device)


def test_stage1_selector_uses_both_kmeans_seeds_and_fixed_round_robin_seed():
    rows, vectors = _vectors(768)
    _FakeClusterManager.calls.clear()

    bundle = run_selection(
        vectors,
        stage=1,
        candidate_rows=rows,
        fake_cluster_manager=_FakeClusterManager,
    )

    assert bundle.k_values == (76, 7)
    assert bundle.kmeans_seeds == KMEANS_SEEDS == (42, 43)
    assert bundle.round_robin_seed == ROUND_ROBIN_SEED == 42
    assert tuple(bundle.representations) == EXPECTED_REPRESENTATIONS
    assert len(bundle.selected_positions) == 14 * 2 * 2
    assert all(len(selected) == 172 for selected in bundle.selected_positions.values())
    assert len(_FakeClusterManager.calls) == 14 * 2 * 2
    assert {call[1] for call in _FakeClusterManager.calls} == {76, 7}
    assert all(call[2:] == (20, False) for call in _FakeClusterManager.calls)


def test_selection_vector_view_joins_shards_and_restores_candidate_order():
    rows = _rows(24)
    name = "P_n1:seed=42:slot=0"

    def shard(positions):
        stable_ids = tuple(str(rows[position]["stable_id"]) for position in positions)
        vector_ids = tuple(selection_vector_id(name, stable_id) for stable_id in stable_ids)
        vectors = np.stack(
            [np.full(4, position + 1, dtype=np.float32) for position in positions]
        )
        manifest = {
            "schema_version": 1,
            "artifact_type": "vector_set",
            "representation": "P",
            "vector_count": len(positions),
            "vector_dimension": 4,
            "vector_ids_sha256": sha256_id_lines(vector_ids),
            "stable_ids_sha256": sha256_id_lines(stable_ids),
        }
        return VectorSet(
            vector_ids=vector_ids,
            stable_ids=stable_ids,
            vectors=vectors,
            full_gradient_norm=np.asarray(positions, dtype=np.float32) + 1,
            projected_gradient_norm=None,
            valid_token_count=None,
            response_length=None,
            sampled_reverse_kl=None,
            opd_signal_rms=None,
            verifier_correct_count=None,
            verifier_total=None,
            manifest=manifest,
        )

    view = build_selection_vector_view(
        (shard(range(12, 24)), shard(range(12))),
        name,
        [str(row["stable_id"]) for row in rows],
    )
    assert view.vector_ids == tuple(
        selection_vector_id(name, str(row["stable_id"])) for row in rows
    )
    assert view.stable_ids == tuple(str(row["stable_id"]) for row in rows)
    assert view.vectors[:, 0].tolist() == list(range(1, 25))
    assert view.full_gradient_norm.tolist() == list(range(1, 25))
    assert view.manifest["artifact_type"] == "opd_proxy_selection_vector_view"
    assert len(view.manifest["source_vector_manifest_sha256"]) == 2


def test_selector_passes_normalized_seeded_permutations_and_restores_labels():
    rows, vectors = _vectors(24)
    _FakeClusterManager.calls.clear()

    bundle = run_selection(
        vectors,
        stage=0,
        candidate_rows=rows,
        fake_cluster_manager=_FakeClusterManager,
    )

    representation = EXPECTED_REPRESENTATIONS[0]
    normalized = torch.nn.functional.normalize(
        torch.from_numpy(vectors[representation].vectors).float(), dim=1
    )
    first_call = _FakeClusterManager.calls[0][0]
    second_call = _FakeClusterManager.calls[1][0]
    permutation_42 = torch.randperm(24, generator=torch.Generator().manual_seed(42))
    permutation_43 = torch.randperm(24, generator=torch.Generator().manual_seed(43))
    assert torch.equal(first_call, normalized[permutation_42])
    assert torch.equal(second_call, normalized[permutation_43])

    expected_labels = torch.empty(24, dtype=torch.int64)
    expected_labels[permutation_42] = torch.arange(24) % 2
    key = bundle.key(representation, k=2, kmeans_seed=42)
    assert np.array_equal(bundle.labels[key], expected_labels.numpy())


def test_stage0_and_stage2_cardinalities_are_derived_not_stage1_constants():
    for stage, count, selected_size, k_values in (
        (0, 24, 5, (2, 2)),
        (2, 1536, 345, (153, 15)),
    ):
        rows, vectors = _vectors(count)
        bundle = run_selection(
            vectors,
            stage=stage,
            candidate_rows=rows,
            fake_cluster_manager=_FakeClusterManager,
        )
        assert bundle.k_values == k_values
        assert all(
            selected.shape == (selected_size,)
            for selected in bundle.selected_positions.values()
        )


def test_selector_rejects_missing_representation_wrong_id_and_wrong_provenance():
    rows, vectors = _vectors(24)
    missing = dict(vectors)
    missing.pop("E")
    with pytest.raises(ValueError, match="representations"):
        run_selection(
            missing,
            stage=0,
            candidate_rows=rows,
            fake_cluster_manager=_FakeClusterManager,
        )

    wrong_id = dict(vectors)
    altered = list(rows)
    altered[3] = {**altered[3], "stable_id": "wrong"}
    wrong_id["E"] = _vector_set("E", altered)
    with pytest.raises(ValueError, match="stable ID"):
        run_selection(
            wrong_id,
            stage=0,
            candidate_rows=rows,
            fake_cluster_manager=_FakeClusterManager,
        )

    wrong_parent = dict(vectors)
    wrong_parent["E"] = _vector_set("E", rows, stage_parent="b" * 64)
    expected_manifest_hashes = {
        name: hashlib.sha256(canonical_json_bytes(vector.manifest)).hexdigest()
        for name, vector in vectors.items()
    }
    with pytest.raises(ValueError, match="manifest hash"):
        run_selection(
            wrong_parent,
            stage=0,
            candidate_rows=rows,
            expected_vector_manifest_hashes=expected_manifest_hashes,
            fake_cluster_manager=_FakeClusterManager,
        )


def test_build_length_quartiles_breaks_ties_by_stable_id_and_restores_row_order():
    rows = [
        {"stable_id": "d", "prompt_token_count_0_6b": 4},
        {"stable_id": "b", "prompt_token_count_0_6b": 1},
        {"stable_id": "c", "prompt_token_count_0_6b": 1},
        {"stable_id": "a", "prompt_token_count_0_6b": 1},
        {"stable_id": "e", "prompt_token_count_0_6b": 5},
        {"stable_id": "f", "prompt_token_count_0_6b": 6},
        {"stable_id": "g", "prompt_token_count_0_6b": 7},
        {"stable_id": "h", "prompt_token_count_0_6b": 8},
    ]
    # Sorted ranks are a,b,c,d,e,f,g,h -> quartiles 0,0,1,1,2,2,3,3.
    assert build_length_quartiles(rows).tolist() == [1, 0, 1, 0, 2, 2, 3, 3]


@pytest.mark.parametrize(
    ("weights", "total", "capacities", "expected"),
    [
        ({"b": 1, "a": 1, "c": 1}, 2, None, {"a": 1, "b": 1, "c": 0}),
        (
            {"a": 100, "b": 1},
            5,
            {"a": 1, "b": 10},
            {"a": 1, "b": 4},
        ),
        (
            {"a": 0, "b": 0},
            3,
            {"a": 1, "b": 4},
            {"a": 1, "b": 2},
        ),
    ],
)
def test_largest_remainder_has_lexicographic_ties_and_capacity_redistribution(
    weights, total, capacities, expected
):
    assert largest_remainder_allocation(
        weights, total=total, capacities=capacities
    ) == expected


def test_random_streams_are_spawned_once_sorted_int32_and_hash_pinned():
    rows = _rows(768)
    schedules = generate_random_schedules(rows, selected_size=172, draws=10_000)

    assert schedules.uniform.dtype.str == "<i4"
    assert schedules.stratified.dtype.str == "<i4"
    assert schedules.uniform.flags.c_contiguous
    assert schedules.stratified.flags.c_contiguous
    assert schedules.uniform.shape == (10_000, 172)
    assert schedules.stratified.shape == (10_000, 172)
    assert np.all(np.diff(schedules.uniform, axis=1) > 0)
    assert np.all(np.diff(schedules.stratified, axis=1) > 0)
    assert schedules.manifest["seed_sequence"] == 2026071402
    assert schedules.manifest["spawn_count"] == 2
    assert schedules.manifest["paired_target_seeds"] == [42, 43]
    assert sha256_int_rows(np.array([[0, 2], [1, 3]], dtype="<i4")) == (
        "74201e550190c3ead9a6c11a336e2d25ed25cb5d36b30166eba198b0596104c2"
    )


def test_random_generation_matches_direct_child_rng_with_shuffle_false():
    rows = _rows(24)
    actual = generate_random_schedules(rows, selected_size=5, draws=3)
    children = np.random.SeedSequence(2026071402).spawn(2)
    rng = np.random.Generator(np.random.PCG64(children[0]))
    expected = np.stack(
        [
            np.sort(rng.choice(24, size=5, replace=False, shuffle=False))
            for _ in range(3)
        ]
    ).astype("<i4")
    assert np.array_equal(actual.uniform, expected)

    # Changing only strata cannot continue or perturb the independent uniform stream.
    changed = [{**row, "leaf_topic": "one-topic"} for row in rows]
    assert np.array_equal(
        generate_random_schedules(changed, selected_size=5, draws=3).uniform,
        actual.uniform,
    )


def test_stratified_schedule_uses_static_allocation_and_global_sorted_positions():
    rows = _rows(24)
    schedules = generate_random_schedules(rows, selected_size=5, draws=10)
    quartiles = build_length_quartiles(rows)
    strata: dict[tuple[str, int], list[int]] = {}
    for position, (row, quartile) in enumerate(zip(rows, quartiles.tolist())):
        strata.setdefault((str(row["leaf_topic"]), quartile), []).append(position)
    expected = largest_remainder_allocation(
        {key: len(value) for key, value in strata.items()}, total=5
    )
    for subset in schedules.stratified:
        actual = {key: 0 for key in strata}
        for position in subset.tolist():
            key = (str(rows[position]["leaf_topic"]), int(quartiles[position]))
            actual[key] += 1
        assert actual == expected


def test_random_artifact_round_trip_rejects_tamper_ids_and_parent(tmp_path: Path):
    rows = _rows(24)
    parents = {"stage_manifest_sha256": "a" * 64}
    written = generate_random_schedules(
        rows,
        selected_size=5,
        draws=100,
        output_directory=tmp_path,
        parent_hashes=parents,
        stage=0,
    )
    loaded = load_random_schedules(
        tmp_path,
        expected_candidate_ids=[str(row["stable_id"]) for row in rows],
        expected_candidate_rows=rows,
        expected_parent_hashes=parents,
    )
    assert np.array_equal(loaded.uniform, written.uniform)
    assert np.array_equal(loaded.stratified, written.stratified)

    with pytest.raises(ValueError, match="candidate ID"):
        load_random_schedules(
            tmp_path,
            expected_candidate_ids=["wrong", *[str(r["stable_id"]) for r in rows[1:]]],
            expected_parent_hashes=parents,
        )
    with pytest.raises(ValueError, match="parent"):
        load_random_schedules(
            tmp_path,
            expected_candidate_ids=[str(row["stable_id"]) for row in rows],
            expected_parent_hashes={"stage_manifest_sha256": "b" * 64},
        )

    path = tmp_path / "random_uniform.npy"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="file SHA"):
        load_random_schedules(tmp_path)


def test_selection_artifact_round_trip_preserves_sampler_order_and_rejects_tamper(
    tmp_path: Path,
):
    rows, vectors = _vectors(24)
    expected_hashes = {
        name: hashlib.sha256(canonical_json_bytes(vector.manifest)).hexdigest()
        for name, vector in vectors.items()
    }
    written = run_selection(
        vectors,
        stage=0,
        candidate_rows=rows,
        expected_vector_manifest_hashes=expected_hashes,
        parent_hashes={"stage_manifest_sha256": "a" * 64},
        output_directory=tmp_path,
        fake_cluster_manager=_FakeClusterManager,
    )
    loaded = load_selection_bundle(
        tmp_path,
        expected_candidate_ids=[str(row["stable_id"]) for row in rows],
        expected_vector_manifest_hashes=expected_hashes,
        expected_parent_hashes={"stage_manifest_sha256": "a" * 64},
    )
    assert loaded.selected_positions.keys() == written.selected_positions.keys()
    for key in written.selected_positions:
        assert np.array_equal(
            loaded.selected_positions[key], written.selected_positions[key]
        )

    manifest_path = tmp_path / "selection.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["round_robin_seed"] = 43
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(ValueError, match="round-robin"):
        load_selection_bundle(tmp_path)


def test_real_official_selector_requires_authorized_single_visible_gpu(monkeypatch):
    rows, vectors = _vectors(24)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(RuntimeError, match="exactly one"):
        run_selection(vectors, stage=0, candidate_rows=rows)


@pytest.mark.gpu
def test_real_official_kmeans_cuda_integration():
    labels = cluster_official(
        torch.arange(1, 97, dtype=torch.float32).reshape(24, 4),
        ratio=0.1,
        iterations=20,
        seed=42,
        reference_repo=Path("/home/mchen/prismatic-synthesis-reference"),
        device="cuda:0",
    )
    assert labels.shape == (24,)
    assert labels.dtype == np.int64
    assert set(labels.tolist()).issubset({0, 1})
