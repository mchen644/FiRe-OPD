from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import torch
from trak.projectors import ProjectionType

from math_eval import opd_proxy_gradient_projection as projection
from math_eval.opd_proxy_gradient_projection import (
    OfficialProjectorSymbols,
    ProjectionConfig,
    ReferenceSnapshot,
    build_projection_manifest,
    construct_cuda_projector,
    full_gradient_l2_norm,
    project_full_gradient,
    projected_mean_is_mean_projection,
    sha256_parameter_layout,
    verify_prismatic_reference,
)


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _reference_repository(tmp_path: Path) -> tuple[Path, str, str, Path]:
    repository = tmp_path / "reference"
    source = (
        repository
        / "prismatic-synthesis"
        / "gradient_modules"
        / "gradient_computer.py"
    )
    source.parent.mkdir(parents=True)
    source.write_text("# exact official source fixture\n", encoding="utf-8")
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.email", "test@example.com")
    _git(repository, "config", "user.name", "Test User")
    _git(repository, "add", ".")
    _git(repository, "commit", "--quiet", "-m", "fixture")
    return (
        repository,
        _git(repository, "rev-parse", "HEAD"),
        _git(repository, "rev-parse", "HEAD^{tree}"),
        source,
    )


def _snapshot(tmp_path: Path) -> ReferenceSnapshot:
    source = tmp_path / "gradient_computer.py"
    source.write_bytes(b"official source\n")
    return ReferenceSnapshot(
        repository=str(tmp_path.resolve()),
        commit="1" * 40,
        tree="2" * 40,
        official_gradient_module=str(source.resolve()),
        official_gradient_module_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )


def _symbols(tmp_path: Path, cuda_class=object) -> OfficialProjectorSymbols:
    return OfficialProjectorSymbols(
        cuda_projector_class=cuda_class,
        projection_type=ProjectionType,
        reference=_snapshot(tmp_path),
    )


class FakeCudaProjector:
    def __init__(self, output: torch.Tensor):
        self.output = output
        self.kwargs = None
        self.seen_input_dtype = None
        self.seen_model_id = None

    def factory(self, **kwargs):
        self.kwargs = kwargs
        return self

    def project(self, value: torch.Tensor, *, model_id: int):
        self.seen_input_dtype = value.dtype
        self.seen_model_id = model_id
        return self.output.clone()


def test_strict_projector_constructor_and_output_scale(monkeypatch, tmp_path):
    monkeypatch.setattr(
        projection, "load_cuda_projector", lambda path: _symbols(tmp_path)
    )
    fake = FakeCudaProjector(torch.full((1, 1024), 32.0))

    projector = construct_cuda_projector(
        4096,
        "cuda:0",
        ProjectionConfig(),
        cuda_projector_class=fake.factory,
    )
    result = project_full_gradient(
        projector, torch.ones(4096, dtype=torch.float32), ProjectionConfig()
    )

    assert fake.kwargs == {
        "grad_dim": 4096,
        "proj_dim": 1024,
        "seed": 0,
        "proj_type": ProjectionType.rademacher,
        "device": "cuda:0",
        "dtype": torch.float16,
        "block_size": 128,
        "max_batch_size": 16,
    }
    assert fake.seen_input_dtype == torch.float16
    assert fake.seen_model_id == 0
    torch.testing.assert_close(result, torch.ones((1, 1024)))
    assert result.dtype == torch.float32


def test_missing_cuda_projector_never_falls_back(monkeypatch):
    monkeypatch.setattr(projection, "load_cuda_projector", lambda path: None)
    with pytest.raises(RuntimeError, match="CudaProjector is required"):
        construct_cuda_projector(8, "cuda:0", ProjectionConfig())


def test_basic_projector_symbol_is_rejected(monkeypatch, tmp_path):
    class BasicProjector:
        pass

    monkeypatch.setattr(
        projection,
        "load_cuda_projector",
        lambda path: _symbols(tmp_path, cuda_class=BasicProjector),
    )
    with pytest.raises(RuntimeError, match="CudaProjector is required"):
        construct_cuda_projector(8, "cuda:0", ProjectionConfig())


def test_full_norm_is_measured_from_float32_before_projection_cast(
    monkeypatch, tmp_path
):
    events = []

    class Recorder:
        def project(self, value, *, model_id):
            events.append(("project", value.dtype, model_id))
            return torch.ones((1, 1024), dtype=torch.float16)

    original = projection.full_gradient_l2_norm

    def record_norm(value):
        events.append(("norm", value.dtype))
        return original(value)

    monkeypatch.setattr(projection, "full_gradient_l2_norm", record_norm)
    gradient = torch.tensor([65504.0, 1.0], dtype=torch.float32)

    project_full_gradient(Recorder(), gradient, ProjectionConfig())

    assert events == [
        ("norm", torch.float32),
        ("project", torch.float16, 0),
    ]
    torch.testing.assert_close(
        full_gradient_l2_norm(gradient), torch.linalg.vector_norm(gradient)
    )


def test_projection_is_linear_for_mean_full_gradient():
    class LinearProjector:
        def project(self, value, *, model_id):
            assert model_id == 0
            summed = value.float().sum(dim=1, keepdim=True)
            return summed.repeat(1, 1024)

    gradients = (
        torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
        torch.tensor([3.0, 4.0, 5.0], dtype=torch.float32),
        torch.tensor([5.0, 6.0, 7.0], dtype=torch.float32),
        torch.tensor([7.0, 8.0, 9.0], dtype=torch.float32),
    )

    assert projected_mean_is_mean_projection(
        LinearProjector(), gradients, ProjectionConfig(), rtol=0.0, atol=0.0
    )
    direct = project_full_gradient(
        LinearProjector(), torch.stack(gradients).mean(dim=0), ProjectionConfig()
    )
    individual_mean = torch.stack(
        [
            project_full_gradient(LinearProjector(), value, ProjectionConfig())
            for value in gradients
        ]
    ).mean(dim=0)
    torch.testing.assert_close(direct, individual_mean, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "gradient",
    [
        torch.zeros(4, dtype=torch.float32),
        torch.tensor([1.0, float("nan")], dtype=torch.float32),
        torch.tensor([1.0, float("inf")], dtype=torch.float32),
        torch.ones((2, 2), dtype=torch.float32),
        torch.ones(4, dtype=torch.float64),
    ],
)
def test_invalid_native_full_gradient_is_rejected(gradient):
    with pytest.raises(ValueError, match="full gradient"):
        project_full_gradient(object(), gradient, ProjectionConfig())


@pytest.mark.parametrize(
    "output",
    [
        torch.zeros((1, 1024)),
        torch.full((1, 1024), float("nan")),
        torch.ones((1024,)),
        torch.ones((1, 8)),
    ],
)
def test_invalid_projector_output_is_rejected(output):
    fake = FakeCudaProjector(output)
    with pytest.raises(ValueError, match="projected gradient"):
        project_full_gradient(
            fake, torch.ones(8, dtype=torch.float32), ProjectionConfig()
        )


def test_noncanonical_projection_configuration_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(
        projection, "load_cuda_projector", lambda path: _symbols(tmp_path)
    )
    with pytest.raises(ValueError, match="dimension"):
        construct_cuda_projector(
            8,
            "cuda:0",
            ProjectionConfig(dimension=8),
            cuda_projector_class=FakeCudaProjector(torch.ones((1, 8))).factory,
        )
    with pytest.raises(ValueError, match="seed"):
        construct_cuda_projector(
            8,
            "cuda:0",
            ProjectionConfig(seed=False),
            cuda_projector_class=FakeCudaProjector(torch.ones((1, 1024))).factory,
        )
    with pytest.raises(ValueError, match="cuda:0"):
        construct_cuda_projector(
            8,
            "cuda:1",
            ProjectionConfig(),
            cuda_projector_class=FakeCudaProjector(torch.ones((1, 1024))).factory,
        )


def test_verify_prismatic_reference_binds_commit_tree_cleanliness_and_source(
    tmp_path,
):
    repository, commit, tree, source = _reference_repository(tmp_path)

    snapshot = verify_prismatic_reference(
        repository, expected_commit=commit, expected_tree=tree
    )

    assert snapshot.commit == commit
    assert snapshot.tree == tree
    assert snapshot.official_gradient_module == str(source.resolve())
    assert snapshot.official_gradient_module_sha256 == hashlib.sha256(
        source.read_bytes()
    ).hexdigest()

    with pytest.raises(ValueError, match="commit"):
        verify_prismatic_reference(
            repository, expected_commit="0" * 40, expected_tree=tree
        )
    with pytest.raises(ValueError, match="tree"):
        verify_prismatic_reference(
            repository, expected_commit=commit, expected_tree="0" * 40
        )
    (repository / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clean"):
        verify_prismatic_reference(
            repository, expected_commit=commit, expected_tree=tree
        )


def test_parameter_layout_hash_is_registration_order_sensitive_and_validated():
    first = [
        {"name": "a", "shape": [2, 3], "numel": 6, "offset": 0},
        {"name": "b", "shape": [2], "numel": 2, "offset": 6},
    ]
    second = [
        {"name": "b", "shape": [2], "numel": 2, "offset": 0},
        {"name": "a", "shape": [2, 3], "numel": 6, "offset": 2},
    ]

    assert sha256_parameter_layout(first) != sha256_parameter_layout(second)
    assert sha256_parameter_layout(first) == sha256_parameter_layout(first)
    with pytest.raises(ValueError, match="contiguous"):
        sha256_parameter_layout(
            [{"name": "a", "shape": [2], "numel": 2, "offset": 1}]
        )


def test_projection_manifest_records_every_frozen_constructor_and_source_field(
    tmp_path,
):
    layout_hash = sha256_parameter_layout(
        [{"name": "weight", "shape": [2, 2], "numel": 4, "offset": 0}]
    )
    manifest = build_projection_manifest(
        gradient_dimension=4,
        parameter_layout_sha256=layout_hash,
        config=ProjectionConfig(),
        reference=_snapshot(tmp_path),
    )

    assert manifest == json.loads(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    )
    assert manifest["backend"] == "CudaProjector"
    assert manifest["constructor"] == {
        "grad_dim": 4,
        "proj_dim": 1024,
        "seed": 0,
        "proj_type": "rademacher",
        "device": "cuda:0",
        "dtype": "float16",
        "block_size": 128,
        "max_batch_size": 16,
    }
    assert manifest["model_id"] == 0
    assert manifest["native_gradient_dtype"] == "float32"
    assert manifest["stored_dtype"] == "float32"
    assert manifest["output_scale"] == "1/sqrt(1024)"
    assert manifest["linearity_tolerance"] == {"rtol": 5e-3, "atol": 5e-3}
    assert manifest["parameter_layout_sha256"] == layout_hash
    assert manifest["reference"]["commit"] == "1" * 40
    assert manifest["reference"]["tree"] == "2" * 40
