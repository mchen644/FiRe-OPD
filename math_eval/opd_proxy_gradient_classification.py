"""Pre-registered classification gates for OPD proxy-gradient verification."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from numbers import Real


_METRICS = ("g_vendi", "coverage", "gradient_norm", "opd_signal")
_PRIMARY_CORNERS = {
    (target_seed, kmeans_seed)
    for target_seed in (42, 43)
    for kmeans_seed in (42, 43)
}
_ORACLE_CORNERS = {
    (selection_seed, 43 if selection_seed == 42 else 42, kmeans_seed)
    for selection_seed in (42, 43)
    for kmeans_seed in (42, 43)
}
_STRICT_THRESHOLDS = {
    "g_vendi": 0.95,
    "coverage": 0.50,
    "gradient_norm": 0.10,
    "opd_signal": 0.10,
}
_PILOT_METRICS = ("g_vendi", "coverage", "gradient_norm", "opd_signal")


def _finite_number(value: object, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{description} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{description} must be a finite number")
    return result


def _percentile(value: object, description: str) -> float:
    result = _finite_number(value, description)
    if result < 0 or result > 1:
        raise ValueError(f"{description} must be in [0, 1]")
    return result


def classify_efficacy_pilot(
    percentiles_by_kmeans_seed: Mapping[int, Mapping[str, object]],
) -> dict[str, object]:
    """Apply the exact one-time primary/uniform efficacy-pilot gate."""
    if not isinstance(percentiles_by_kmeans_seed, Mapping) or set(
        percentiles_by_kmeans_seed
    ) != {42, 43}:
        raise ValueError("efficacy pilot requires exact K-means seeds 42 and 43")
    normalized: dict[int, dict[str, float]] = {}
    for seed in (42, 43):
        row = percentiles_by_kmeans_seed[seed]
        if not isinstance(row, Mapping) or set(row) != set(_PILOT_METRICS):
            raise ValueError("efficacy pilot record has incorrect metric keys")
        normalized[seed] = {
            metric: _percentile(row[metric], f"pilot seed {seed} {metric}")
            for metric in _PILOT_METRICS
        }
    go = all(
        row["g_vendi"] >= 0.90
        and row["coverage"] >= 0.90
        and row["gradient_norm"] >= 0.25
        and row["opd_signal"] >= 0.25
        for row in normalized.values()
    )
    no_go = any(
        row["g_vendi"] <= 0.60 or row["coverage"] <= 0.60
        for row in normalized.values()
    )
    decision = "go" if go else "no_go" if no_go else "borderline"
    return {
        "status": "pilot_only",
        "decision": decision,
        "main_hypothesis": "not_evaluated",
        "stage1_thresholds_modified": False,
        "thresholds": {
            "go": {
                "g_vendi": 0.90,
                "coverage": 0.90,
                "gradient_norm": 0.25,
                "opd_signal": 0.25,
            },
            "no_go": {
                "g_vendi_at_or_below": 0.60,
                "coverage_at_or_below": 0.60,
            },
        },
        "primary_uniform_percentiles_by_kmeans_seed": normalized,
    }


def exact_median_of_eight(values: Sequence[float]) -> float:
    if isinstance(values, (str, bytes)):
        raise ValueError("median-of-eight requires eight finite values")
    try:
        raw = list(values)
    except TypeError as error:
        raise ValueError("median-of-eight requires eight finite values") from error
    if len(raw) != 8:
        raise ValueError("median-of-eight requires eight finite values")
    try:
        ordered = sorted(
            _finite_number(value, "median value") for value in raw
        )
    except ValueError as error:
        raise ValueError("median-of-eight requires eight finite values") from error
    return 0.5 * (ordered[3] + ordered[4])


def _corner_worst(record: Mapping[str, object], realization: str) -> dict[str, float]:
    corners = record.get("primary_corners")
    if not isinstance(corners, Sequence) or isinstance(corners, (str, bytes)):
        raise ValueError(
            f"realization {realization} primary_corners must contain four corners"
        )
    if len(corners) != 4:
        raise ValueError(
            f"realization {realization} primary_corners must contain four corners"
        )
    by_corner: dict[tuple[int, int], dict[str, float]] = {}
    for corner in corners:
        if not isinstance(corner, Mapping):
            raise ValueError("primary corner must be a mapping")
        target_seed = corner.get("target_seed")
        kmeans_seed = corner.get("kmeans_seed")
        if (
            isinstance(target_seed, bool)
            or not isinstance(target_seed, int)
            or isinstance(kmeans_seed, bool)
            or not isinstance(kmeans_seed, int)
        ):
            raise ValueError("primary corner seeds must be integers")
        key = (target_seed, kmeans_seed)
        if key in by_corner:
            raise ValueError(f"duplicate primary corner {key}")
        by_corner[key] = {
            metric: _percentile(
                corner.get(metric), f"{realization} {key} {metric} percentile"
            )
            for metric in _METRICS
        }
    if set(by_corner) != _PRIMARY_CORNERS:
        raise ValueError(
            "primary corners must cover both target and K-means seeds exactly"
        )
    return {
        metric: min(values[metric] for values in by_corner.values())
        for metric in _METRICS
    }


def _normalize_worst_by_realization(
    worst_by_realization: Mapping[str, Mapping[str, object]],
    *,
    expected_realizations: int,
) -> dict[str, dict[str, float]]:
    if not isinstance(worst_by_realization, Mapping):
        raise ValueError("worst-case realizations must be a mapping")
    if (
        isinstance(expected_realizations, bool)
        or not isinstance(expected_realizations, int)
        or expected_realizations <= 0
    ):
        raise ValueError("expected realizations must be a positive integer")
    if len(worst_by_realization) != expected_realizations:
        raise ValueError(
            f"expected {expected_realizations} realizations, got {len(worst_by_realization)}"
        )
    normalized: dict[str, dict[str, float]] = {}
    for realization in sorted(worst_by_realization):
        if not isinstance(realization, str) or not realization:
            raise ValueError("realization names must be nonempty strings")
        record = worst_by_realization[realization]
        if not isinstance(record, Mapping):
            raise ValueError(f"realization {realization} must be a mapping")
        if "primary_corners" in record:
            normalized[realization] = _corner_worst(record, realization)
        else:
            normalized[realization] = {
                metric: _percentile(
                    record.get(metric), f"{realization} {metric} percentile"
                )
                for metric in _METRICS
            }
    return normalized


def _values_record(
    normalized: Mapping[str, Mapping[str, float]], metric: str
) -> tuple[dict[str, float], list[float]]:
    by_realization = {
        realization: float(normalized[realization][metric])
        for realization in sorted(normalized)
    }
    return by_realization, list(by_realization.values())


def evaluate_pn1_components(
    worst_by_realization: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Evaluate exact median/count gates over eight primary worst cases."""
    normalized = _normalize_worst_by_realization(
        worst_by_realization, expected_realizations=8
    )
    result: dict[str, object] = {
        "realization_order": list(normalized),
        "worst_by_realization": normalized,
    }

    g_by, g_values = _values_record(normalized, "g_vendi")
    g_median = exact_median_of_eight(g_values)
    g_count = sum(value >= 0.90 for value in g_values)
    result["g_vendi"] = {
        "values_by_realization": g_by,
        "median": g_median,
        "median_threshold": 0.95,
        "secondary_threshold": 0.90,
        "minimum_secondary_count": 6,
        "count_at_or_above_secondary": g_count,
        "pass": g_median >= 0.95 and g_count >= 6,
    }

    coverage_by, coverage_values = _values_record(normalized, "coverage")
    coverage_median = exact_median_of_eight(coverage_values)
    coverage_count = sum(value >= 0.10 for value in coverage_values)
    result["coverage"] = {
        "values_by_realization": coverage_by,
        "median": coverage_median,
        "median_threshold": 0.50,
        "secondary_threshold": 0.10,
        "minimum_secondary_count": 7,
        "count_at_or_above_secondary": coverage_count,
        "pass": coverage_median >= 0.50 and coverage_count >= 7,
    }

    for metric in ("gradient_norm", "opd_signal"):
        values_by_realization, values = _values_record(normalized, metric)
        median = exact_median_of_eight(values)
        result[metric] = {
            "values_by_realization": values_by_realization,
            "median": median,
            "median_threshold": 0.10,
            "pass": median >= 0.10,
        }
    result["all_pass"] = all(
        bool(result[metric]["pass"])  # type: ignore[index]
        for metric in _METRICS
    )
    return result


def evaluate_strict_arm_components(
    worst_by_realization: Mapping[str, Mapping[str, object]],
    *,
    expected_realizations: int,
) -> dict[str, object]:
    """Require every realization to clear every strict-arm threshold."""
    normalized = _normalize_worst_by_realization(
        worst_by_realization, expected_realizations=expected_realizations
    )
    result: dict[str, object] = {
        "realization_order": list(normalized),
        "worst_by_realization": normalized,
    }
    for metric, threshold in _STRICT_THRESHOLDS.items():
        values_by_realization, values = _values_record(normalized, metric)
        minimum = min(values)
        passing_count = sum(value >= threshold for value in values)
        result[metric] = {
            "values_by_realization": values_by_realization,
            "minimum": minimum,
            "threshold": threshold,
            "passing_count": passing_count,
            "required_count": expected_realizations,
            "pass": passing_count == expected_realizations,
        }
    result["all_pass"] = all(
        bool(result[metric]["pass"])  # type: ignore[index]
        for metric in _METRICS
    )
    return result


def evaluate_oracle_gate(
    cross_seed_runs: Sequence[Mapping[str, object]],
    target_cka_p_value: float,
) -> dict[str, object]:
    """Evaluate four opposing-target oracle runs and the target CKA gate."""
    if isinstance(cross_seed_runs, (str, bytes)) or len(cross_seed_runs) != 4:
        raise ValueError("oracle gate requires exactly four cross-seed runs")
    normalized_by_key: dict[tuple[int, int, int], dict[str, object]] = {}
    for run in cross_seed_runs:
        if not isinstance(run, Mapping):
            raise ValueError("oracle run must be a mapping")
        selection_seed = run.get("selection_target_seed")
        evaluation_seed = run.get("evaluation_target_seed")
        kmeans_seed = run.get("kmeans_seed")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (selection_seed, evaluation_seed, kmeans_seed)
        ):
            raise ValueError("oracle run seeds must be integers")
        key = (selection_seed, evaluation_seed, kmeans_seed)
        if key in normalized_by_key:
            raise ValueError(f"duplicate oracle run {key}")
        g_vendi = _percentile(
            run.get("g_vendi_percentile"), "oracle G-Vendi percentile"
        )
        coverage = _percentile(
            run.get("coverage_percentile"), "oracle coverage percentile"
        )
        normalized_by_key[key] = {
            "selection_target_seed": selection_seed,
            "evaluation_target_seed": evaluation_seed,
            "kmeans_seed": kmeans_seed,
            "g_vendi_percentile": g_vendi,
            "coverage_percentile": coverage,
            "pass": g_vendi >= 0.95 and coverage >= 0.50,
        }
    if set(normalized_by_key) != _ORACLE_CORNERS:
        raise ValueError(
            "oracle runs must provide all four opposing cross-seed/K-means corners"
        )
    cka_p_value = _percentile(target_cka_p_value, "target CKA p-value")
    cross_seed_pass = all(
        bool(run["pass"]) for run in normalized_by_key.values()
    )
    target_dependence_pass = cka_p_value <= 0.01
    failed_components: list[str] = []
    if not cross_seed_pass:
        failed_components.append("cross_seed_oracle_selection")
    if not target_dependence_pass:
        failed_components.append("target_dependence_cka")
    return {
        "classification": "pass" if not failed_components else "fail",
        "cross_seed_selection_pass": cross_seed_pass,
        "target_dependence_pass": target_dependence_pass,
        "target_cka_p_value": cka_p_value,
        "target_cka_p_value_threshold": 0.01,
        "g_vendi_percentile_threshold": 0.95,
        "coverage_percentile_threshold": 0.50,
        "cross_seed_runs": [
            normalized_by_key[key] for key in sorted(normalized_by_key)
        ],
        "failed_components": failed_components,
    }


def _oracle_gate_pass(oracle: Mapping[str, object], description: str) -> bool:
    if not isinstance(oracle, Mapping) or oracle.get("classification") not in {
        "pass",
        "fail",
    }:
        raise ValueError(f"{description} gate must have pass/fail classification")
    cross_seed = oracle.get("cross_seed_selection_pass")
    target_dependence = oracle.get("target_dependence_pass")
    if not isinstance(cross_seed, bool) or not isinstance(target_dependence, bool):
        raise ValueError(f"{description} gate lacks Boolean component decisions")
    expected = cross_seed and target_dependence
    if (oracle["classification"] == "pass") != expected:
        raise ValueError(f"{description} classification is inconsistent with components")
    return expected


def _component_gate_pass(components: Mapping[str, object], description: str) -> bool:
    if not isinstance(components, Mapping):
        raise ValueError(f"{description} components must be a mapping")
    passes: list[bool] = []
    for metric in _METRICS:
        record = components.get(metric)
        if not isinstance(record, Mapping) or not isinstance(record.get("pass"), bool):
            raise ValueError(f"{description} {metric} component lacks a Boolean pass")
        passes.append(bool(record["pass"]))
    all_pass = components.get("all_pass")
    if not isinstance(all_pass, bool) or all_pass != all(passes):
        raise ValueError(f"{description} all_pass is inconsistent with components")
    return all_pass


def _arm_result(
    components: Mapping[str, object], *, classification: str, description: str
) -> dict[str, object]:
    failed = [
        metric
        for metric in _METRICS
        if not bool(components[metric]["pass"])  # type: ignore[index]
    ]
    return {
        "classification": classification,
        "components": copy.deepcopy(dict(components)),
        "failed_components": failed,
        "description": description,
    }


def _interpretation(arm_classes: Mapping[str, str]) -> str:
    if arm_classes["P_n1"] == "pass":
        return "practical_single_rollout_proxy_supported"
    if arm_classes["P_n4"] == "pass":
        return "four_rollout_proxy_only_supported"
    if arm_classes["S"] == "pass":
        return "sft_gradient_only_supported"
    if arm_classes["E"] == "pass":
        return "prompt_embedding_only_supported"
    return "no_candidate_representation_supported"


def classify_stage1(
    *,
    oracle: Mapping[str, object],
    pn1_components: Mapping[str, object],
    pn4_components: Mapping[str, object],
    sft_components: Mapping[str, object],
    embedding_components: Mapping[str, object],
) -> dict[str, object]:
    """Apply oracle precedence, then classify all candidate arms independently."""
    oracle_pass = _oracle_gate_pass(oracle, "oracle")
    component_inputs = {
        "P_n1": pn1_components,
        "P_n4": pn4_components,
        "S": sft_components,
        "E": embedding_components,
    }
    component_pass = {
        arm: _component_gate_pass(components, arm)
        for arm, components in component_inputs.items()
    }
    if not oracle_pass:
        arms = {
            arm: _arm_result(
                components,
                classification="inconclusive",
                description="oracle_or_target_dependence_gate_failed",
            )
            for arm, components in component_inputs.items()
        }
        return {
            "classification": "inconclusive",
            "oracle": copy.deepcopy(dict(oracle)),
            **arms,
            "interpretation": "inconclusive_oracle_failure",
        }

    arm_classes = {
        arm: "pass" if component_pass[arm] else "fail" for arm in component_inputs
    }
    arms = {
        arm: _arm_result(
            components,
            classification=arm_classes[arm],
            description="independent_pre_registered_component_conjunction",
        )
        for arm, components in component_inputs.items()
    }
    return {
        "classification": "classified",
        "oracle": copy.deepcopy(dict(oracle)),
        **arms,
        "interpretation": _interpretation(arm_classes),
    }


def decide_stage2(stage1_report: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(stage1_report, Mapping):
        raise ValueError("Stage-1 report must be a mapping")
    oracle = stage1_report.get("oracle")
    if not isinstance(oracle, Mapping) or oracle.get("classification") != "pass":
        return {"run_stage2": False, "reason": "oracle_not_pass"}
    pn1 = stage1_report.get("P_n1")
    if not isinstance(pn1, Mapping):
        raise ValueError("Stage-1 report lacks P_n1")
    components = pn1.get("components")
    if not isinstance(components, Mapping):
        raise ValueError("Stage-1 P_n1 report lacks components")
    g_vendi = components.get("g_vendi")
    if not isinstance(g_vendi, Mapping):
        raise ValueError("Stage-1 P_n1 report lacks G-Vendi component")
    median = _percentile(g_vendi.get("median"), "Stage-1 P_n1 G-Vendi median")
    should_run = 0.90 <= median < 0.95
    return {
        "run_stage2": should_run,
        "reason": (
            "borderline_primary_pn1_g_vendi" if should_run else "outside_trigger"
        ),
    }


def classify_stage2_sensitivity(
    *,
    stage1_report: Mapping[str, object],
    expanded_oracle: Mapping[str, object],
    expanded_pn1_components: Mapping[str, object],
) -> dict[str, object]:
    """Classify the expanded pool without modifying the Stage-1 conclusion."""
    decision = decide_stage2(stage1_report)
    if decision["run_stage2"] is not True:
        raise ValueError("Stage 2 sensitivity requires the pre-registered trigger")
    expanded_oracle_pass = _oracle_gate_pass(expanded_oracle, "expanded oracle")
    components_pass = _component_gate_pass(
        expanded_pn1_components, "expanded P_n1"
    )
    if not expanded_oracle_pass:
        classification = "extended_inconclusive"
        reason = "expanded_oracle_or_target_dependence_gate_failed"
    elif components_pass:
        classification = "extended_pass"
        reason = "expanded_primary_pn1_components_pass"
    else:
        classification = "extended_fail"
        reason = "expanded_primary_pn1_component_failure"
    return {
        "classification": classification,
        "reason": reason,
        "trigger": decision,
        "stage1_classification": stage1_report.get("classification"),
        "stage1_pn1_classification": (
            stage1_report.get("P_n1", {}).get("classification")
            if isinstance(stage1_report.get("P_n1"), Mapping)
            else None
        ),
        "expanded_oracle": copy.deepcopy(dict(expanded_oracle)),
        "expanded_P_n1": {
            "components": copy.deepcopy(dict(expanded_pn1_components)),
            "failed_components": [
                metric
                for metric in _METRICS
                if not bool(expanded_pn1_components[metric]["pass"])  # type: ignore[index]
            ],
        },
    }


__all__ = [
    "classify_efficacy_pilot",
    "classify_stage1",
    "classify_stage2_sensitivity",
    "decide_stage2",
    "evaluate_oracle_gate",
    "evaluate_pn1_components",
    "evaluate_strict_arm_components",
    "exact_median_of_eight",
]
