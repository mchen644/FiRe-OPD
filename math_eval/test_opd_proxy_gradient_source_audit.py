from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _function(path: str, name: str, *, class_name: str | None = None) -> ast.FunctionDef:
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"), filename=path)
    body = tree.body
    if class_name is not None:
        classes = [
            node
            for node in body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ]
        assert len(classes) == 1
        body = classes[0].body
    matches = [
        node
        for node in body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    assert len(matches) == 1
    node = matches[0]
    assert isinstance(node, ast.FunctionDef)
    return node


def _called_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if isinstance(child.func, ast.Attribute):
            names.add(child.func.attr)
        elif isinstance(child.func, ast.Name):
            names.add(child.func.id)
    return names


def test_capture_actor_source_has_no_backward_optimizer_or_update_call():
    node = _function(
        "verl/verl/workers/actor/dp_actor.py",
        "capture_opd_proxy_verify",
        class_name="DataParallelPPOActor",
    )
    assert _called_names(node).isdisjoint(
        {
            "backward",
            "step",
            "zero_grad",
            "_optimizer_step",
            "update_actor",
            "update_critic",
        }
    )


def test_capture_trainer_source_bypasses_reward_critic_balancing_and_updates():
    node = _function(
        "verl/verl/trainer/ppo/ray_trainer.py",
        "_fit_opd_proxy_verify_capture",
        class_name="RayPPOTrainer",
    )
    assert _called_names(node).isdisjoint(
        {
            "_balance_batch",
            "compute_reward",
            "compute_reward_async",
            "update_actor",
            "update_critic",
            "compute_values",
            "_validate",
        }
    )


def test_capture_optimizer_resolver_is_fail_closed():
    node = _function(
        "verl/verl/workers/fsdp_workers.py", "_resolve_actor_optim_config"
    )
    source = ast.unparse(node)
    assert "opd_proxy_verify_capture_only" in source
    assert "return None" in source
    assert "return actor_config.optim" in source


def test_launchers_have_no_runtime_install_srun_or_interpreter_fallback():
    production_files = (
        "math_eval/run_opd_proxy_gradient_verify.py",
        "verl/examples/fire_opd/run_capture_opd_proxy_verify.sh",
        "verl/examples/fire_opd/run_direct_opd_proxy_gradient_fixture.sh",
    )
    forbidden = ("pip install", "conda install", "uv pip", "srun ")
    for relative in production_files:
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert all(token not in text for token in forbidden), relative
    capture = (ROOT / production_files[1]).read_text(encoding="utf-8")
    direct = (ROOT / production_files[2]).read_text(encoding="utf-8")
    assert '/home/mchen/miniconda3/envs/verl/bin/python' in capture
    assert '/home/mchen/miniconda3/envs/gvendi-opd/bin/python' in direct


def test_projection_source_exposes_only_pinned_cuda_projector_backend():
    text = (ROOT / "math_eval/opd_proxy_gradient_projection.py").read_text(
        encoding="utf-8"
    )
    assert "CudaProjector" in text
    assert "BasicProjector" not in text
    assert "device != \"cuda:0\"" in text
    assert "flat_gradient.detach().contiguous().to(config.input_dtype)" in text
