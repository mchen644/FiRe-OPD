# Single-Rollout Difficulty-Routed Budget20 OPD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add rollout_n=1 difficulty-routed Budget20 OPD: correctness×confidence gates route easy samples to concise/shorter ESR and hard samples to entropy-preserving training, with metrics to verify the proxy.

**Architecture:** Add a small pure helper module for two-signal routing, extend TALE rollout-length prompt/mask construction to accept per-sample prompt styles and ESR fractions, wire the helper into the trainer before teacher/ref scoring, and add a small hard-gated entropy loss in the actor. The disabled path remains byte-for-byte behaviorally equivalent: no new tensors, no routing, no entropy term.

**Tech Stack:** Python 3.10, PyTorch, veRL `DataProto`, Hydra/OmegaConf dataclass config, pytest, bash wrappers.

## Global Constraints

- Keep `ROLLOUT_N=1`; do not add extra student rollouts.
- Student rollout prompt remains normal prompt.
- Default non-easy path remains Budget20: budget-aware teacher prompt, rollout-length budget, 20% ESR.
- Easy gate formula: `easy_i = r_i * c_i`.
- Hard gate formula: `hard_i = (1 - r_i) * (1 - c_i)`.
- `r_i = 1[sequence_reward_i > correct_reward_threshold]`.
- `c_i = percentile_rank_batch(mean_t old_log_probs_i,t)`.
- Easy compression: route to concise teacher prompt and reduce ESR beta from 0.20 down to no less than 0.10.
- Hard exploration: keep budget teacher prompt, keep 20% ESR, add hard-gated max-entropy loss.
- Rethinking OPD probe behavior must remain unchanged and run before hardtrunc for probe-enabled budget runs.
- Disabled path must preserve existing training/API behavior.
- No logits stored by default.
- Disable eval for smoke/probe wrappers unless explicitly overridden: `trainer.val_before_train=False`, `trainer.test_freq=-1`.

---

## File structure

- Create `verl/verl/trainer/ppo/difficulty_aware_opd.py`
  - Pure tensor helpers for confidence ranks, gates, per-sample ESR beta, prompt style routing, and metric summaries.
- Create `verl/tests/trainer/ppo/test_difficulty_aware_opd.py`
  - CPU tests for helper behavior.
- Modify `verl/verl/trainer/config/algorithm.py`
  - Add `DifficultyAwareOpdConfig` and `AlgoConfig.difficulty_aware_opd`.
- Modify `verl/tests/trainer/config/test_algo_config_on_cpu.py`
  - Add Hydra override/config tests.
- Modify `verl/verl/trainer/ppo/tale_budget.py`
  - Allow `compute_rollout_length_tale_budget(..., beta=...)` to accept a per-sample tensor.
- Modify `verl/verl/trainer/ppo/ray_trainer.py`
  - Resolve reward early only when routing is enabled, compute gates, attach tensors/non-tensors, and pass per-sample routing to TALE.
- Modify `verl/tests/trainer/ppo/test_tale_budget.py`
  - Add per-sample ESR beta + per-sample prompt style tests.
- Modify `verl/verl/workers/actor/dp_actor.py`
  - Select `difficulty_aware_entropy_weight`, calculate entropy when present, and add hard-gated entropy loss.
- Modify `verl/tests/workers/actor/test_length_aware_opd.py`
  - Add focused tests for hard-gated entropy helper.
- Create `run_train_tale_budget_rolloutlen_hardtrunc_difficultyrouted_rethinking_probe_opd.sh`
  - Training/probe wrapper for the active difficulty-routed Budget20 run.

---

### Task 1: Config and Pure Difficulty-Routing Helper

**Files:**
- Create: `verl/verl/trainer/ppo/difficulty_aware_opd.py`
- Create: `verl/tests/trainer/ppo/test_difficulty_aware_opd.py`
- Modify: `verl/verl/trainer/config/algorithm.py`
- Modify: `verl/tests/trainer/config/test_algo_config_on_cpu.py`

**Interfaces:**
- Produces: `DifficultyAwareOpdConfig` in `verl.trainer.config.algorithm`.
- Produces: `DifficultyAwareRoutingResult` dataclass in `verl.trainer.ppo.difficulty_aware_opd`.
- Produces: `rank_to_unit_interval(values: torch.Tensor) -> torch.Tensor`.
- Produces: `compute_two_signal_difficulty_routing(token_level_scores: torch.Tensor, old_log_probs: torch.Tensor, response_mask: torch.Tensor, config, base_esr_beta: float) -> DifficultyAwareRoutingResult`.
- Produces: `summarize_difficulty_routing(result: DifficultyAwareRoutingResult, original_response_lengths: torch.Tensor | None = None) -> dict[str, float]`.
- Consumes later: trainer will call the helper after full reward and `old_log_probs` exist.

- [ ] **Step 1: Write failing helper tests**

Create `verl/tests/trainer/ppo/test_difficulty_aware_opd.py` with:

```python
from types import SimpleNamespace

import torch

from verl.trainer.ppo.difficulty_aware_opd import (
    compute_two_signal_difficulty_routing,
    rank_to_unit_interval,
    summarize_difficulty_routing,
)


def _cfg(**overrides):
    values = {
        "correct_reward_threshold": 0.5,
        "easy_prompt_threshold": 0.7,
        "easy_prompt_style": "concise",
        "default_prompt_style": "budget",
        "min_easy_esr_beta": 0.10,
        "easy_esr_delta": 0.10,
        "hard_entropy_coef": 0.001,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_rank_to_unit_interval_handles_monotonic_ties_nan_and_singleton():
    values = torch.tensor([3.0, 1.0, 1.0, float("nan"), 5.0])

    ranks = rank_to_unit_interval(values)

    assert torch.isfinite(ranks).all()
    assert ranks[1].item() == ranks[2].item()
    assert ranks[3].item() == 0.5
    assert ranks[4].item() == 1.0
    assert rank_to_unit_interval(torch.tensor([7.0])).tolist() == [0.5]


def test_two_signal_routing_marks_correct_high_conf_easy_and_wrong_low_conf_hard():
    response_mask = torch.ones((4, 3), dtype=torch.float32)
    old_log_probs = torch.tensor(
        [
            [-0.1, -0.1, -0.1],  # high confidence, correct -> easy
            [-3.0, -3.0, -3.0],  # low confidence, wrong -> hard
            [-2.0, -2.0, -2.0],  # low-ish confidence, correct -> not easy
            [-0.2, -0.2, -0.2],  # high confidence, wrong -> overconfident wrong
        ],
        dtype=torch.float32,
    )
    token_level_scores = torch.zeros_like(response_mask)
    token_level_scores[0, -1] = 1.0
    token_level_scores[2, -1] = 1.0

    result = compute_two_signal_difficulty_routing(
        token_level_scores=token_level_scores,
        old_log_probs=old_log_probs,
        response_mask=response_mask,
        config=_cfg(),
        base_esr_beta=0.20,
    )

    assert result.correct.tolist() == [1.0, 0.0, 1.0, 0.0]
    assert result.easy[0].item() > 0.7
    assert result.hard[1].item() > 0.7
    assert result.prompt_styles.tolist()[0] == "concise"
    assert result.prompt_styles.tolist()[1] == "budget"
    assert result.esr_beta[0].item() < 0.20
    assert result.esr_beta[0].item() >= 0.10
    assert result.esr_beta[1].item() == 0.20
    assert result.entropy_weight[1].item() > result.entropy_weight[0].item()
    assert result.entropy_weight[3].item() == 0.0


def test_summarize_difficulty_routing_emits_proxy_diagnostics():
    response_mask = torch.ones((2, 2), dtype=torch.float32)
    old_log_probs = torch.tensor([[-0.1, -0.1], [-2.0, -2.0]], dtype=torch.float32)
    token_level_scores = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.float32)

    result = compute_two_signal_difficulty_routing(
        token_level_scores=token_level_scores,
        old_log_probs=old_log_probs,
        response_mask=response_mask,
        config=_cfg(),
        base_esr_beta=0.20,
    )
    metrics = summarize_difficulty_routing(result, original_response_lengths=torch.tensor([10, 20]))

    assert metrics["difficulty_aware_opd/correct_rate"] == 0.5
    assert metrics["difficulty_aware_opd/concise_prompt_ratio"] == 0.5
    assert metrics["difficulty_aware_opd/budget_prompt_ratio"] == 0.5
    assert metrics["difficulty_aware_opd/orig_response_length_mean"] == 15.0
    assert "difficulty_aware_opd/wrong_high_conf_ratio" in metrics
    assert "difficulty_aware_opd/correct_low_conf_ratio" in metrics
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=/home/mchen/FiRe-OPD/verl pytest -q verl/tests/trainer/ppo/test_difficulty_aware_opd.py
```

Expected: FAIL with `ModuleNotFoundError: No module named 'verl.trainer.ppo.difficulty_aware_opd'`.

- [ ] **Step 3: Add config dataclass**

Modify `verl/verl/trainer/config/algorithm.py`:

```python
__all__ = [
    "AlgoConfig",
    "CandidateSelectionConfig",
    "DifficultyAwareOpdConfig",
    "FilterGroupsConfig",
    "KLControlConfig",
    "RethinkingOpdProbeConfig",
    "RolloutCorrectionConfig",
    "TaleBudgetConfig",
]
```

Add after `RethinkingOpdProbeConfig`:

```python
@dataclass
class DifficultyAwareOpdConfig(BaseConfig):
    """Single-rollout difficulty routing for Budget20 OPD."""

    enabled: bool = False
    method: str = "two_signal_prompt_esr_entropy"
    correct_reward_threshold: float = 0.5
    confidence_rank_scope: str = "batch"
    easy_prompt_threshold: float = 0.7
    easy_prompt_style: str = "concise"
    default_prompt_style: str = "budget"
    base_esr_beta: Optional[float] = None
    min_easy_esr_beta: float = 0.10
    easy_esr_delta: float = 0.10
    hard_entropy_coef: float = 0.001
```

Add to `AlgoConfig` fields:

```python
    difficulty_aware_opd: DifficultyAwareOpdConfig = field(default_factory=DifficultyAwareOpdConfig)
```

- [ ] **Step 4: Add pure helper implementation**

Create `verl/verl/trainer/ppo/difficulty_aware_opd.py`:

```python
"""Pure helpers for single-rollout difficulty-routed Budget20 OPD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


_ALLOWED_PROMPT_STYLES = {"budget", "normal", "concise"}


@dataclass(frozen=True)
class DifficultyAwareRoutingResult:
    correct: torch.Tensor
    confidence_rank: torch.Tensor
    easy: torch.Tensor
    hard: torch.Tensor
    esr_beta: torch.Tensor
    entropy_weight: torch.Tensor
    prompt_styles: np.ndarray
    sequence_rewards: torch.Tensor
    student_logp_mean: torch.Tensor


def _config_get(config, name: str, default):
    if hasattr(config, "get"):
        return config.get(name, default)
    return getattr(config, name, default)


def rank_to_unit_interval(values: torch.Tensor) -> torch.Tensor:
    """Return deterministic percentile ranks in [0, 1], using 0.5 for NaNs and singleton batches."""

    flat = values.detach().float().flatten()
    out = torch.full_like(flat, 0.5)
    finite_mask = torch.isfinite(flat)
    finite_values = flat[finite_mask]
    n = int(finite_values.numel())
    if n <= 1:
        return out.view_as(values)

    sorted_values, sorted_order = torch.sort(finite_values)
    finite_ranks = torch.empty_like(finite_values)
    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_values[end] == sorted_values[start]:
            end += 1
        midrank = (start + end - 1) / 2.0
        finite_ranks[sorted_order[start:end]] = midrank / float(n - 1)
        start = end
    out[finite_mask] = finite_ranks
    return out.view_as(values)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=values.device, dtype=values.dtype)
    denom = mask.sum(dim=-1).clamp(min=1.0)
    return (values * mask).sum(dim=-1) / denom


def compute_two_signal_difficulty_routing(
    *,
    token_level_scores: torch.Tensor,
    old_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    config,
    base_esr_beta: float,
) -> DifficultyAwareRoutingResult:
    """Compute correctness-confidence easy/hard gates and per-sample routing decisions."""

    if token_level_scores.shape != old_log_probs.shape or response_mask.shape != old_log_probs.shape:
        raise ValueError("token_level_scores, old_log_probs, and response_mask must have identical shape")
    if str(_config_get(config, "method", "two_signal_prompt_esr_entropy")) != "two_signal_prompt_esr_entropy":
        raise ValueError("difficulty_aware_opd.method must be 'two_signal_prompt_esr_entropy'")

    easy_prompt_style = str(_config_get(config, "easy_prompt_style", "concise"))
    default_prompt_style = str(_config_get(config, "default_prompt_style", "budget"))
    if easy_prompt_style not in _ALLOWED_PROMPT_STYLES:
        raise ValueError(f"Invalid easy_prompt_style: {easy_prompt_style!r}")
    if default_prompt_style not in _ALLOWED_PROMPT_STYLES:
        raise ValueError(f"Invalid default_prompt_style: {default_prompt_style!r}")

    sequence_rewards = token_level_scores.float().sum(dim=-1)
    threshold = float(_config_get(config, "correct_reward_threshold", 0.5))
    correct = (sequence_rewards > threshold).to(dtype=torch.float32)

    student_logp_mean = _masked_mean(old_log_probs.float(), response_mask.float())
    confidence_rank = rank_to_unit_interval(student_logp_mean).to(device=old_log_probs.device, dtype=torch.float32)

    easy = correct * confidence_rank
    hard = (1.0 - correct) * (1.0 - confidence_rank)

    base = torch.full_like(easy, float(base_esr_beta))
    min_beta = float(_config_get(config, "min_easy_esr_beta", 0.10))
    delta = float(_config_get(config, "easy_esr_delta", 0.10))
    esr_beta = torch.clamp(base - delta * easy, min=min_beta, max=float(base_esr_beta))

    entropy_coef = float(_config_get(config, "hard_entropy_coef", 0.001))
    entropy_weight = hard * entropy_coef

    easy_threshold = float(_config_get(config, "easy_prompt_threshold", 0.7))
    prompt_styles = np.array(
        [easy_prompt_style if float(value) >= easy_threshold else default_prompt_style for value in easy.detach().cpu()],
        dtype=object,
    )

    return DifficultyAwareRoutingResult(
        correct=correct.detach(),
        confidence_rank=confidence_rank.detach(),
        easy=easy.detach(),
        hard=hard.detach(),
        esr_beta=esr_beta.detach(),
        entropy_weight=entropy_weight.detach(),
        prompt_styles=prompt_styles,
        sequence_rewards=sequence_rewards.detach(),
        student_logp_mean=student_logp_mean.detach(),
    )


def _safe_mean(values: torch.Tensor) -> float:
    return float(values.float().mean().item()) if values.numel() else 0.0


def summarize_difficulty_routing(
    result: DifficultyAwareRoutingResult,
    original_response_lengths: torch.Tensor | None = None,
) -> dict[str, float]:
    """Summarize routing gates and actions for trainer logging."""

    prompt_styles: Sequence[object] = result.prompt_styles.tolist()
    total = max(len(prompt_styles), 1)
    concise_count = sum(style == "concise" for style in prompt_styles)
    budget_count = sum(style == "budget" for style in prompt_styles)
    wrong = 1.0 - result.correct
    high_conf = result.confidence_rank >= 0.7
    low_conf = result.confidence_rank <= 0.3
    metrics = {
        "difficulty_aware_opd/enabled": 1.0,
        "difficulty_aware_opd/correct_rate": _safe_mean(result.correct),
        "difficulty_aware_opd/confidence_rank_mean": _safe_mean(result.confidence_rank),
        "difficulty_aware_opd/confidence_rank_correct_mean": _safe_mean(result.confidence_rank[result.correct.bool()])
        if result.correct.bool().any()
        else 0.0,
        "difficulty_aware_opd/confidence_rank_wrong_mean": _safe_mean(result.confidence_rank[wrong.bool()])
        if wrong.bool().any()
        else 0.0,
        "difficulty_aware_opd/easy_mean": _safe_mean(result.easy),
        "difficulty_aware_opd/easy_max": float(result.easy.float().max().item()) if result.easy.numel() else 0.0,
        "difficulty_aware_opd/hard_mean": _safe_mean(result.hard),
        "difficulty_aware_opd/hard_max": float(result.hard.float().max().item()) if result.hard.numel() else 0.0,
        "difficulty_aware_opd/concise_prompt_ratio": concise_count / total,
        "difficulty_aware_opd/budget_prompt_ratio": budget_count / total,
        "difficulty_aware_opd/esr_beta_mean": _safe_mean(result.esr_beta),
        "difficulty_aware_opd/esr_beta_min": float(result.esr_beta.float().min().item()) if result.esr_beta.numel() else 0.0,
        "difficulty_aware_opd/esr_beta_max": float(result.esr_beta.float().max().item()) if result.esr_beta.numel() else 0.0,
        "difficulty_aware_opd/hard_entropy_weight_mean": _safe_mean(result.entropy_weight),
        "difficulty_aware_opd/wrong_high_conf_ratio": _safe_mean((wrong.bool() & high_conf).float()),
        "difficulty_aware_opd/correct_low_conf_ratio": _safe_mean((result.correct.bool() & low_conf).float()),
    }
    if original_response_lengths is not None:
        metrics["difficulty_aware_opd/orig_response_length_mean"] = _safe_mean(original_response_lengths.float())
    return metrics
```

- [ ] **Step 5: Add config tests**

Append to `TestAlgoConfig` in `verl/tests/trainer/config/test_algo_config_on_cpu.py`:

```python
    def test_yaml_accepts_difficulty_aware_opd_overrides(self):
        config = None

        import os

        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        GlobalHydra.instance().clear()
        try:
            config_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "verl", "trainer", "config")
            )
            with initialize_config_dir(config_dir=config_dir):
                config = omega_conf_to_dataclass(
                    compose(
                        config_name="ppo_trainer",
                        overrides=[
                            "algorithm.difficulty_aware_opd.enabled=True",
                            "algorithm.difficulty_aware_opd.easy_prompt_threshold=0.75",
                            "algorithm.difficulty_aware_opd.min_easy_esr_beta=0.1",
                            "algorithm.difficulty_aware_opd.hard_entropy_coef=0.003",
                        ],
                    ).algorithm
                )
        finally:
            GlobalHydra.instance().clear()

        da = config.difficulty_aware_opd
        assert da.enabled is True
        assert da.method == "two_signal_prompt_esr_entropy"
        assert da.easy_prompt_threshold == 0.75
        assert da.min_easy_esr_beta == 0.1
        assert da.hard_entropy_coef == 0.003
```

- [ ] **Step 6: Run tests to verify Task 1 passes**

Run:

```bash
PYTHONPATH=/home/mchen/FiRe-OPD/verl pytest -q \
  verl/tests/trainer/ppo/test_difficulty_aware_opd.py \
  verl/tests/trainer/config/test_algo_config_on_cpu.py::TestAlgoConfig::test_yaml_accepts_difficulty_aware_opd_overrides
```

Expected: PASS.

- [ ] **Step 7: Commit Task 1**

```bash
git add \
  verl/verl/trainer/ppo/difficulty_aware_opd.py \
  verl/tests/trainer/ppo/test_difficulty_aware_opd.py \
  verl/verl/trainer/config/algorithm.py \
  verl/tests/trainer/config/test_algo_config_on_cpu.py
git commit -m "Add difficulty-aware OPD routing helpers"
```

---

### Task 2: Per-Sample ESR Beta and Prompt Style in TALE Budgeting

**Files:**
- Modify: `verl/verl/trainer/ppo/tale_budget.py`
- Modify: `verl/verl/trainer/ppo/ray_trainer.py`
- Modify: `verl/tests/trainer/ppo/test_tale_budget.py`

**Interfaces:**
- Consumes: `batch.batch["difficulty_aware_esr_beta"]`, shape `[batch]`, optional.
- Consumes: `batch.non_tensor_batch["difficulty_aware_prompt_style"]`, object array of `"budget"`, `"normal"`, or `"concise"`, optional.
- Produces: `batch.batch["tale_budget_response_lengths"]`, shape `[batch]`, original rollout lengths.
- Produces unchanged: `batch.batch["tale_budget_esr_loss_mask"]`, response-aligned ESR mask.

- [ ] **Step 1: Add failing TALE test for per-sample routing**

Append to `verl/tests/trainer/ppo/test_tale_budget.py`:

```python
def test_apply_online_tale_budget_prompts_uses_per_sample_difficulty_routing():
    from verl.trainer.ppo.ray_trainer import _apply_online_tale_budget_prompts

    raw_prompts = _object_array(
        [
            [{"role": "user", "content": "Easy problem?\nPlease reason step by step, and put your final answer within \\boxed{}."}],
            [{"role": "user", "content": "Hard problem?"}],
        ]
    )
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )
    batch = DataProto.from_dict(
        tensors={
            "input_ids": torch.ones((2, 12), dtype=torch.long),
            "response_mask": response_mask,
            "difficulty_aware_esr_beta": torch.tensor([0.1, 0.2], dtype=torch.float32),
        },
        non_tensors={
            "raw_prompt": raw_prompts,
            "difficulty_aware_prompt_style": _object_array(["concise", "budget"]),
        },
    )
    tokenizer = FakeBudgetTokenizer([])
    rollout_worker = FakeBudgetRolloutWorker()
    config = TaleBudgetConfig(
        enabled=True,
        source="rollout_length",
        min_budget=1,
        max_budget=8192,
        round_to=1,
        rollout_length_alpha=1.0,
        esr_beta=0.2,
        teacher_prompt_style="budget",
    )

    metrics = _apply_online_tale_budget_prompts(
        batch=batch,
        actor_rollout_wg=rollout_worker,
        tokenizer=tokenizer,
        tale_budget_config=config,
        max_prompt_length=512,
        truncation="error",
        apply_chat_template_kwargs={"enable_thinking": False},
    )

    prompts = batch.non_tensor_batch["teacher_prompt"].tolist()
    assert "Solve concisely" in prompts[0][0]["content"]
    assert "use less than 10 tokens" in prompts[1][0]["content"]
    assert batch.batch["tale_budget_response_lengths"].tolist() == [10, 10]
    assert batch.batch["tale_budget_esr_loss_mask"].sum(dim=-1).tolist() == [1, 2]
    assert metrics["tale_budget/esr_tokens_mean"] == 1.5
    assert metrics["tale_budget/esr_supervised_fraction_mean"] == 0.15
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
PYTHONPATH=/home/mchen/FiRe-OPD/verl pytest -q \
  verl/tests/trainer/ppo/test_tale_budget.py::test_apply_online_tale_budget_prompts_uses_per_sample_difficulty_routing
```

Expected: FAIL because `difficulty_aware_esr_beta` and per-sample prompt styles are ignored.

- [ ] **Step 3: Update `compute_rollout_length_tale_budget` to accept tensor beta**

Modify `verl/verl/trainer/ppo/tale_budget.py` inside `compute_rollout_length_tale_budget`:

```python
    if isinstance(beta, torch.Tensor):
        beta_values = beta.to(device=response_mask.device, dtype=torch.float32)
        if beta_values.dim() != 1 or beta_values.shape[0] != response_mask.shape[0]:
            raise ValueError("beta tensor must have shape [batch]")
        if torch.any(beta_values <= 0.0):
            raise ValueError("all beta values must be positive")
    else:
        if beta <= 0.0:
            raise ValueError("beta must be positive")
        beta_values = torch.full((response_mask.shape[0],), float(beta), device=response_mask.device)
```

Replace the scalar ESR computation:

```python
    esr_tokens = torch.round(budgets.float() * float(beta)).to(dtype=torch.long)
```

with:

```python
    esr_tokens = torch.round(budgets.float() * beta_values).to(dtype=torch.long)
```

Keep the existing `alpha`, `min_budget`, `round_to`, and `max_budget` validation unchanged.

- [ ] **Step 4: Update `_apply_online_tale_budget_prompts` for per-sample beta and styles**

Modify `verl/verl/trainer/ppo/ray_trainer.py` in `_apply_online_tale_budget_prompts` rollout-length branch.

Before calling `compute_rollout_length_tale_budget`, add:

```python
        esr_beta_value = tale_budget_config.get("esr_beta", 1.0)
        if "difficulty_aware_esr_beta" in batch.batch.keys():
            esr_beta_value = batch.batch["difficulty_aware_esr_beta"]
```

Pass `beta=esr_beta_value` instead of `beta=float(tale_budget_config.get("esr_beta", 1.0))`.

After `response_lengths = ...`, add:

```python
        batch.batch["tale_budget_response_lengths"] = result.response_lengths.detach().to(dtype=torch.long)
```

Replace teacher prompt construction with per-row style support:

```python
    per_sample_prompt_styles = batch.non_tensor_batch.get("difficulty_aware_prompt_style", None)
    if per_sample_prompt_styles is None:
        per_sample_prompt_styles = np.array([teacher_prompt_style] * len(questions), dtype=object)

    teacher_prompts = []
    for question, budget, row_style, raw_messages in zip(
        questions,
        budgets,
        per_sample_prompt_styles.tolist(),
        batch.non_tensor_batch["raw_prompt"].tolist(),
        strict=True,
    ):
        row_style = str(row_style)
        if row_style == "budget":
            teacher_prompts.append(build_tale_budget_teacher_messages(question=question, budget=budget))
        elif row_style == "normal":
            teacher_prompts.append(deepcopy(raw_messages))
        elif row_style == "concise":
            teacher_prompts.append(build_concise_teacher_messages(question=question))
        else:
            raise ValueError(f"Unsupported per-sample teacher prompt style: {row_style!r}")
```

Remove the old single-style `if teacher_prompt_style == ...` block.

- [ ] **Step 5: Run TALE tests**

Run:

```bash
PYTHONPATH=/home/mchen/FiRe-OPD/verl pytest -q verl/tests/trainer/ppo/test_tale_budget.py
```

Expected: PASS.

- [ ] **Step 6: Commit Task 2**

```bash
git add \
  verl/verl/trainer/ppo/tale_budget.py \
  verl/verl/trainer/ppo/ray_trainer.py \
  verl/tests/trainer/ppo/test_tale_budget.py
git commit -m "Support per-sample difficulty routing in TALE budget prompts"
```

---

### Task 3: Trainer Plumbing and Difficulty Metrics

**Files:**
- Modify: `verl/verl/trainer/ppo/ray_trainer.py`
- Modify: `verl/tests/trainer/ppo/test_difficulty_aware_opd.py`

**Interfaces:**
- Consumes: Task 1 helper `compute_two_signal_difficulty_routing`.
- Consumes: Task 2 TALE per-sample routing fields.
- Produces: batch tensors `difficulty_aware_correct`, `difficulty_aware_confidence_rank`, `difficulty_aware_easy`, `difficulty_aware_hard`, `difficulty_aware_esr_beta`, and optionally `difficulty_aware_entropy_weight`.
- Produces: non-tensor `difficulty_aware_prompt_style`.
- Produces: scalar metrics under `difficulty_aware_opd/`.

- [ ] **Step 1: Add failing trainer helper test**

Append to `verl/tests/trainer/ppo/test_difficulty_aware_opd.py`:

```python
import numpy as np

from verl import DataProto
from verl.trainer.ppo.ray_trainer import _apply_difficulty_aware_opd_routing


def test_apply_difficulty_aware_opd_routing_adds_batch_tensors_and_metrics():
    response_mask = torch.ones((2, 3), dtype=torch.float32)
    old_log_probs = torch.tensor([[-0.1, -0.1, -0.1], [-2.0, -2.0, -2.0]], dtype=torch.float32)
    reward_tensor = torch.zeros_like(response_mask)
    reward_tensor[0, -1] = 1.0
    batch = DataProto.from_dict(
        tensors={
            "response_mask": response_mask,
            "old_log_probs": old_log_probs,
        }
    )
    metrics = {}

    _apply_difficulty_aware_opd_routing(
        batch=batch,
        reward_tensor=reward_tensor,
        difficulty_config=_cfg(),
        base_esr_beta=0.20,
        metrics=metrics,
    )

    assert batch.batch["difficulty_aware_correct"].tolist() == [1.0, 0.0]
    assert batch.non_tensor_batch["difficulty_aware_prompt_style"].tolist() == ["concise", "budget"]
    assert batch.batch["difficulty_aware_esr_beta"][0].item() < 0.20
    assert batch.batch["difficulty_aware_entropy_weight"][1].item() > 0.0
    assert metrics["difficulty_aware_opd/correct_rate"] == 0.5
    assert metrics["difficulty_aware_opd/concise_prompt_ratio"] == 0.5
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
PYTHONPATH=/home/mchen/FiRe-OPD/verl pytest -q \
  verl/tests/trainer/ppo/test_difficulty_aware_opd.py::test_apply_difficulty_aware_opd_routing_adds_batch_tensors_and_metrics
```

Expected: FAIL because `_apply_difficulty_aware_opd_routing` is missing.

- [ ] **Step 3: Add trainer helper functions**

Modify `verl/verl/trainer/ppo/ray_trainer.py` near `_rethinking_probe_enabled`:

```python
def _difficulty_aware_opd_enabled(config) -> bool:
    difficulty_config = config.algorithm.get("difficulty_aware_opd", None)
    return bool(difficulty_config and difficulty_config.get("enabled", False))


def _apply_difficulty_aware_opd_routing(
    *,
    batch: DataProto,
    reward_tensor: torch.Tensor,
    difficulty_config,
    base_esr_beta: float,
    metrics: dict,
) -> None:
    from verl.trainer.ppo.difficulty_aware_opd import (
        compute_two_signal_difficulty_routing,
        summarize_difficulty_routing,
    )

    if "old_log_probs" not in batch.batch.keys():
        raise ValueError("difficulty-aware OPD routing requires old_log_probs")
    if "response_mask" not in batch.batch.keys():
        raise ValueError("difficulty-aware OPD routing requires response_mask")

    result = compute_two_signal_difficulty_routing(
        token_level_scores=reward_tensor,
        old_log_probs=batch.batch["old_log_probs"],
        response_mask=batch.batch["response_mask"],
        config=difficulty_config,
        base_esr_beta=base_esr_beta,
    )
    batch.batch["difficulty_aware_correct"] = result.correct.to(device=batch.batch["response_mask"].device)
    batch.batch["difficulty_aware_confidence_rank"] = result.confidence_rank.to(device=batch.batch["response_mask"].device)
    batch.batch["difficulty_aware_easy"] = result.easy.to(device=batch.batch["response_mask"].device)
    batch.batch["difficulty_aware_hard"] = result.hard.to(device=batch.batch["response_mask"].device)
    batch.batch["difficulty_aware_esr_beta"] = result.esr_beta.to(device=batch.batch["response_mask"].device)
    if torch.any(result.entropy_weight > 0):
        batch.batch["difficulty_aware_entropy_weight"] = result.entropy_weight.to(device=batch.batch["response_mask"].device)
    batch.non_tensor_batch["difficulty_aware_prompt_style"] = result.prompt_styles

    original_lengths = batch.batch["response_mask"].to(dtype=torch.long).sum(dim=-1)
    metrics.update(summarize_difficulty_routing(result, original_response_lengths=original_lengths))
```

Ensure `torch` is already imported in `ray_trainer.py`; it is used elsewhere in the file.

- [ ] **Step 4: Wire helper into `fit` before TALE prompt construction**

In `RayPPOTrainer.fit`, after `batch = batch.union(old_log_prob)` and before `if self.use_reference_policy:` computes TALE/ref log-probs, initialize reward state near the reward computation block:

```python
                    reward_tensor = None
                    reward_extra_infos_dict = {}
```

When sync reward runs, keep existing assignment:

```python
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
```

After old log-prob computation and before `tale_budget_hard_truncated = False`, add:

```python
                    if _difficulty_aware_opd_enabled(self.config):
                        if self.config.reward_model.launch_reward_fn_async and reward_tensor is None:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        difficulty_config = self.config.algorithm.difficulty_aware_opd
                        tale_budget_config_for_da = self.config.algorithm.get("tale_budget", None)
                        base_esr_beta = float(
                            difficulty_config.get(
                                "base_esr_beta",
                                None,
                            )
                            if difficulty_config.get("base_esr_beta", None) is not None
                            else tale_budget_config_for_da.get("esr_beta", 0.2)
                        )
                        _apply_difficulty_aware_opd_routing(
                            batch=batch,
                            reward_tensor=reward_tensor,
                            difficulty_config=difficulty_config,
                            base_esr_beta=base_esr_beta,
                            metrics=metrics,
                        )
```

In the later advantage block, replace:

```python
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
```

with:

```python
                        if self.config.reward_model.launch_reward_fn_async and reward_tensor is None:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
```

This prevents resolving the same Ray future twice when difficulty routing is enabled.

- [ ] **Step 5: Run focused tests**

Run:

```bash
PYTHONPATH=/home/mchen/FiRe-OPD/verl pytest -q \
  verl/tests/trainer/ppo/test_difficulty_aware_opd.py \
  verl/tests/trainer/ppo/test_tale_budget.py
```

Expected: PASS.

- [ ] **Step 6: Commit Task 3**

```bash
git add \
  verl/verl/trainer/ppo/ray_trainer.py \
  verl/tests/trainer/ppo/test_difficulty_aware_opd.py
git commit -m "Wire difficulty-aware OPD routing into trainer"
```

---

### Task 4: Hard-Gated Entropy Loss in Actor

**Files:**
- Modify: `verl/verl/workers/actor/dp_actor.py`
- Modify: `verl/tests/workers/actor/test_length_aware_opd.py`

**Interfaces:**
- Consumes: `difficulty_aware_entropy_weight`, shape `[batch]`, where values already include `hard_entropy_coef`.
- Produces: actor metrics `difficulty_aware_opd/hard_entropy_loss` and `difficulty_aware_opd/hard_entropy_weight_mean`.
- Leaves existing `actor.entropy_coeff` behavior unchanged.

- [ ] **Step 1: Add failing entropy helper tests**

Append to `verl/tests/workers/actor/test_length_aware_opd.py`:

```python
from verl.workers.actor.dp_actor import _compute_difficulty_aware_entropy_loss


def test_difficulty_aware_entropy_loss_is_negative_and_weighted():
    entropy = torch.tensor([[0.5, 1.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float32)
    response_mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.float32)
    entropy_weight = torch.tensor([0.0, 0.01], dtype=torch.float32)

    loss, metrics = _compute_difficulty_aware_entropy_loss(
        entropy=entropy,
        response_mask=response_mask,
        entropy_weight=entropy_weight,
        loss_agg_mode="token-mean",
    )

    assert loss.item() < 0.0
    assert metrics["difficulty_aware_opd/hard_entropy_weight_mean"] == 0.005
    assert metrics["difficulty_aware_opd/hard_entropy_loss"] == loss.item()


def test_difficulty_aware_entropy_loss_returns_zero_without_valid_weight():
    entropy = torch.ones((2, 3), dtype=torch.float32)
    response_mask = torch.ones((2, 3), dtype=torch.float32)
    entropy_weight = torch.zeros(2, dtype=torch.float32)

    loss, metrics = _compute_difficulty_aware_entropy_loss(
        entropy=entropy,
        response_mask=response_mask,
        entropy_weight=entropy_weight,
        loss_agg_mode="token-mean",
    )

    assert loss.item() == 0.0
    assert metrics["difficulty_aware_opd/hard_entropy_weight_mean"] == 0.0
```

- [ ] **Step 2: Run failing tests**

Run:

```bash
PYTHONPATH=/home/mchen/FiRe-OPD/verl pytest -q \
  verl/tests/workers/actor/test_length_aware_opd.py::test_difficulty_aware_entropy_loss_is_negative_and_weighted \
  verl/tests/workers/actor/test_length_aware_opd.py::test_difficulty_aware_entropy_loss_returns_zero_without_valid_weight
```

Expected: FAIL because `_compute_difficulty_aware_entropy_loss` is missing.

- [ ] **Step 3: Add entropy helper**

Modify `verl/verl/workers/actor/dp_actor.py` near other module-level helper functions:

```python
def _compute_difficulty_aware_entropy_loss(
    *,
    entropy: torch.Tensor,
    response_mask: torch.Tensor,
    entropy_weight: torch.Tensor,
    loss_agg_mode: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    entropy_weight = entropy_weight.to(device=entropy.device, dtype=entropy.dtype)
    if entropy_weight.dim() != 1 or entropy_weight.shape[0] != entropy.shape[0]:
        raise ValueError("difficulty_aware_entropy_weight must have shape [batch]")
    if torch.all(entropy_weight <= 0):
        zero = entropy.sum() * 0.0
        return zero, {"difficulty_aware_opd/hard_entropy_weight_mean": 0.0, "difficulty_aware_opd/hard_entropy_loss": 0.0}

    weighted_entropy = entropy * entropy_weight.unsqueeze(-1)
    entropy_loss = -agg_loss(loss_mat=weighted_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    metrics = {
        "difficulty_aware_opd/hard_entropy_weight_mean": entropy_weight.float().mean().item(),
        "difficulty_aware_opd/hard_entropy_loss": entropy_loss.detach().item(),
    }
    return entropy_loss, metrics
```

- [ ] **Step 4: Wire helper into actor update**

In `DataParallelPPOActor.update_policy`, add selection:

```python
        if "difficulty_aware_entropy_weight" in data.batch.keys():
            select_keys.append("difficulty_aware_entropy_weight")
```

After `entropy_aware = ...`, add:

```python
        difficulty_aware_entropy = "difficulty_aware_entropy_weight" in data.batch.keys()
```

Replace:

```python
                    calculate_entropy = entropy_coeff != 0 or entropy_aware
```

with:

```python
                    calculate_entropy = entropy_coeff != 0 or entropy_aware or difficulty_aware_entropy
```

After the vanilla/entropy-aware OPD block assigns `pg_loss`, add:

```python
                    if "difficulty_aware_entropy_weight" in model_inputs:
                        difficulty_entropy_loss, difficulty_entropy_metrics = _compute_difficulty_aware_entropy_loss(
                            entropy=entropy,
                            response_mask=loss_response_mask,
                            entropy_weight=model_inputs["difficulty_aware_entropy_weight"],
                            loss_agg_mode=loss_agg_mode,
                        )
                        pg_loss = pg_loss + difficulty_entropy_loss
                        micro_batch_metrics.update(difficulty_entropy_metrics)
```

Keep the existing separate `actor.entropy_coeff` block unchanged.

- [ ] **Step 5: Run actor helper tests**

Run:

```bash
PYTHONPATH=/home/mchen/FiRe-OPD/verl pytest -q \
  verl/tests/workers/actor/test_length_aware_opd.py::test_difficulty_aware_entropy_loss_is_negative_and_weighted \
  verl/tests/workers/actor/test_length_aware_opd.py::test_difficulty_aware_entropy_loss_returns_zero_without_valid_weight
```

Expected: PASS.

- [ ] **Step 6: Commit Task 4**

```bash
git add \
  verl/verl/workers/actor/dp_actor.py \
  verl/tests/workers/actor/test_length_aware_opd.py
git commit -m "Add hard-gated entropy loss for difficulty-aware OPD"
```

---

### Task 5: Difficulty-Routed Training Wrapper

**Files:**
- Create: `run_train_tale_budget_rolloutlen_hardtrunc_difficultyrouted_rethinking_probe_opd.sh`

**Interfaces:**
- Consumes: existing `verl/examples/fire_opd/run_opd_strong_to_weak_student_raw_teacher_tale_budget30b.sh`.
- Produces: a reproducible wrapper for Budget20 + difficulty routing + Rethinking OPD probe.

- [ ] **Step 1: Write wrapper script**

Create `run_train_tale_budget_rolloutlen_hardtrunc_difficultyrouted_rethinking_probe_opd.sh`:

```bash
#!/usr/bin/env bash
# Rollout-length ESR / hard-truncation ablation with difficulty-routed Budget20 OPD.
#
# Default behavior:
#   easy = correct * confidence_rank -> concise teacher + shorter ESR
#   hard = (1-correct) * (1-confidence_rank) -> budget teacher + 20% ESR + entropy bonus
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/mchen/FiRe-OPD}"
cd "${REPO_DIR}"

export PATH="/home/mchen/miniconda3/envs/verl/bin:${PATH}"

export ROLLOUT_N=1
export TALE_BUDGET_SOURCE=rollout_length
export TALE_ROLLOUT_ALPHA=1.0
export TALE_ESR_BETA=0.2
export TALE_ROLLOUT_MAX_BUDGET=null
export TALE_MIN_BUDGET=1
export TALE_ROUND_TO=1
export TALE_TRUNCATE_TO_ESR=True
export TALE_USE_BUDGET_TEACHER_PROMPT=True
export DATA_ROOT="${DATA_ROOT:-${REPO_DIR}/data/g-opd}"
export TRAIN_DATA="${TRAIN_DATA:-${DATA_ROOT}/DeepMath-103K/train_filtered_level6.parquet}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
export ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-4}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-opd-budget20-difficulty-routed-rethinking-probe}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_DIR}/checkpoints/${EXPERIMENT_NAME}}"

export DA_EASY_PROMPT_THRESHOLD="${DA_EASY_PROMPT_THRESHOLD:-0.7}"
export DA_MIN_EASY_ESR_BETA="${DA_MIN_EASY_ESR_BETA:-0.10}"
export DA_EASY_ESR_DELTA="${DA_EASY_ESR_DELTA:-0.10}"
export DA_HARD_ENTROPY_COEF="${DA_HARD_ENTROPY_COEF:-0.001}"
export DA_EASY_PROMPT_STYLE="${DA_EASY_PROMPT_STYLE:-concise}"
export DA_DEFAULT_PROMPT_STYLE="${DA_DEFAULT_PROMPT_STYLE:-budget}"

bash verl/examples/fire_opd/run_opd_strong_to_weak_student_raw_teacher_tale_budget30b.sh \
  algorithm.tale_budget.enabled=True \
  algorithm.tale_budget.source=${TALE_BUDGET_SOURCE} \
  algorithm.tale_budget.rollout_length_alpha=${TALE_ROLLOUT_ALPHA} \
  algorithm.tale_budget.esr_beta=${TALE_ESR_BETA} \
  algorithm.tale_budget.rollout_length_max_budget=${TALE_ROLLOUT_MAX_BUDGET} \
  algorithm.tale_budget.truncate_to_esr=${TALE_TRUNCATE_TO_ESR} \
  algorithm.tale_budget.use_budget_teacher_prompt=${TALE_USE_BUDGET_TEACHER_PROMPT} \
  algorithm.tale_budget.min_budget=${TALE_MIN_BUDGET} \
  algorithm.tale_budget.round_to=${TALE_ROUND_TO} \
  algorithm.tale_budget.teacher_prompt_style=auto \
  algorithm.difficulty_aware_opd.enabled=True \
  algorithm.difficulty_aware_opd.method=two_signal_prompt_esr_entropy \
  algorithm.difficulty_aware_opd.easy_prompt_threshold=${DA_EASY_PROMPT_THRESHOLD} \
  algorithm.difficulty_aware_opd.easy_prompt_style=${DA_EASY_PROMPT_STYLE} \
  algorithm.difficulty_aware_opd.default_prompt_style=${DA_DEFAULT_PROMPT_STYLE} \
  algorithm.difficulty_aware_opd.min_easy_esr_beta=${DA_MIN_EASY_ESR_BETA} \
  algorithm.difficulty_aware_opd.easy_esr_delta=${DA_EASY_ESR_DELTA} \
  algorithm.difficulty_aware_opd.hard_entropy_coef=${DA_HARD_ENTROPY_COEF} \
  algorithm.rethinking_opd_probe.enabled=True \
  algorithm.rethinking_opd_probe.top_k=16 \
  algorithm.rethinking_opd_probe.chunk_size=1024 \
  algorithm.rethinking_opd_probe.csv_path=math_eval/opd_training_dynamics_audit/training_probe_budget20_difficulty_routed.csv \
  trainer.experiment_name=${EXPERIMENT_NAME} \
  trainer.val_before_train=False \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  "$@"
```

- [ ] **Step 2: Make executable and run syntax check**

Run:

```bash
chmod +x run_train_tale_budget_rolloutlen_hardtrunc_difficultyrouted_rethinking_probe_opd.sh
bash -n run_train_tale_budget_rolloutlen_hardtrunc_difficultyrouted_rethinking_probe_opd.sh
```

Expected: exit code 0.

- [ ] **Step 3: Commit Task 5**

```bash
git add run_train_tale_budget_rolloutlen_hardtrunc_difficultyrouted_rethinking_probe_opd.sh
git commit -m "Add difficulty-routed Budget20 probe wrapper"
```

---

### Task 6: Focused Verification and Smoke Commands

**Files:**
- Modify only if tests reveal defects in files from Tasks 1-5.

**Interfaces:**
- Consumes: all previous task outputs.
- Produces: verified local test suite and smoke command instructions.

- [ ] **Step 1: Run focused CPU tests**

Run:

```bash
PYTHONPATH=/home/mchen/FiRe-OPD/verl pytest -q \
  verl/tests/trainer/ppo/test_difficulty_aware_opd.py \
  verl/tests/trainer/ppo/test_tale_budget.py \
  verl/tests/trainer/config/test_algo_config_on_cpu.py::TestAlgoConfig::test_yaml_accepts_difficulty_aware_opd_overrides \
  verl/tests/workers/actor/test_length_aware_opd.py::test_difficulty_aware_entropy_loss_is_negative_and_weighted \
  verl/tests/workers/actor/test_length_aware_opd.py::test_difficulty_aware_entropy_loss_returns_zero_without_valid_weight
```

Expected: PASS.

- [ ] **Step 2: Run syntax checks**

Run:

```bash
python -m py_compile verl/verl/trainer/ppo/difficulty_aware_opd.py
bash -n run_train_tale_budget_rolloutlen_hardtrunc_difficultyrouted_rethinking_probe_opd.sh
```

Expected: both commands exit 0.

- [ ] **Step 3: Document metrics-only dry-run overrides**

Use this command shape for a metrics-only first run; it computes/logs gates but keeps prompt, ESR, and entropy behavior equivalent to Budget20:

```bash
EXP="opd-budget20-da-metrics-only" \
EXPERIMENT_NAME="$EXP" \
CHECKPOINT_DIR="/home/mchen/FiRe-OPD/checkpoints/$EXP" \
DA_EASY_PROMPT_STYLE=budget \
DA_EASY_ESR_DELTA=0 \
DA_HARD_ENTROPY_COEF=0 \
bash run_train_tale_budget_rolloutlen_hardtrunc_difficultyrouted_rethinking_probe_opd.sh \
  trainer.total_training_steps=2
```

Expected after implementation: console/W&B metrics include `difficulty_aware_opd/correct_rate`, `difficulty_aware_opd/easy_mean`, `difficulty_aware_opd/hard_mean`, `difficulty_aware_opd/wrong_high_conf_ratio`, and existing Budget20 metrics remain present.

- [ ] **Step 4: Document active smoke command**

Use this command shape for active routing smoke:

```bash
EXP="opd-budget20-da-active-smoke" \
EXPERIMENT_NAME="$EXP" \
CHECKPOINT_DIR="/home/mchen/FiRe-OPD/checkpoints/$EXP" \
bash run_train_tale_budget_rolloutlen_hardtrunc_difficultyrouted_rethinking_probe_opd.sh \
  trainer.total_training_steps=2
```

Expected after implementation: metrics include nonzero `difficulty_aware_opd/concise_prompt_ratio` if any correct/high-confidence samples appear, `tale_budget/esr_supervised_fraction_mean <= 0.2`, and `difficulty_aware_opd/hard_entropy_loss <= 0` when hard samples appear.

- [ ] **Step 5: Commit verification doc updates if any commands/scripts changed**

If Task 6 required edits, commit the changed implementation/test/script files from this plan:

```bash
git add \
  verl/verl/trainer/ppo/difficulty_aware_opd.py \
  verl/verl/trainer/config/algorithm.py \
  verl/verl/trainer/ppo/tale_budget.py \
  verl/verl/trainer/ppo/ray_trainer.py \
  verl/verl/workers/actor/dp_actor.py \
  verl/tests/trainer/ppo/test_difficulty_aware_opd.py \
  verl/tests/trainer/ppo/test_tale_budget.py \
  verl/tests/trainer/config/test_algo_config_on_cpu.py \
  verl/tests/workers/actor/test_length_aware_opd.py \
  run_train_tale_budget_rolloutlen_hardtrunc_difficultyrouted_rethinking_probe_opd.sh
git commit -m "Verify difficulty-routed Budget20 OPD plumbing"
```

If Task 6 required no edits, do not create an empty commit.

---

## Plan self-review

- Spec coverage: Tasks 1 and 3 implement correctness×confidence gates and metrics; Task 2 implements concise prompt routing and direct ESR compression; Task 4 implements CEEH-style hard-gated entropy; Task 5 provides the requested Budget20 rollout_n=1 wrapper; Task 6 verifies disabled-equivalent metrics-only and active smoke commands.
- Disabled path: all new behavior is gated by `algorithm.difficulty_aware_opd.enabled`; TALE per-sample routing only activates when routing tensors/non-tensors are present; actor entropy only activates when `difficulty_aware_entropy_weight` is selected.
- Probe compatibility: trainer routing happens before TALE/ref scoring; the existing Rethinking probe remains in the same ref-log-prob-before-hardtrunc location.
- Type consistency: helper result tensor names match trainer-added batch keys; actor consumes only `difficulty_aware_entropy_weight`; TALE consumes `difficulty_aware_esr_beta` and `difficulty_aware_prompt_style`.
