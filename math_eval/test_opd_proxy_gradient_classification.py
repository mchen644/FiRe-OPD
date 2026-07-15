import copy

import pytest

from math_eval.opd_proxy_gradient_classification import (
    classify_efficacy_pilot,
    classify_stage1,
    classify_stage2_sensitivity,
    decide_stage2,
    evaluate_oracle_gate,
    evaluate_pn1_components,
    evaluate_strict_arm_components,
    exact_median_of_eight,
)


def _worst_cases(
    *,
    g_vendi,
    coverage,
    gradient_norm,
    opd_signal,
):
    return {
        f"r{index}": {
            "g_vendi": g_vendi[index],
            "coverage": coverage[index],
            "gradient_norm": gradient_norm[index],
            "opd_signal": opd_signal[index],
        }
        for index in range(len(g_vendi))
    }


def _passing_pn1():
    return evaluate_pn1_components(
        _worst_cases(
            g_vendi=[0.95] * 8,
            coverage=[0.50] * 8,
            gradient_norm=[0.10] * 8,
            opd_signal=[0.10] * 8,
        )
    )


def _passing_strict(count: int):
    return evaluate_strict_arm_components(
        _worst_cases(
            g_vendi=[0.95] * count,
            coverage=[0.50] * count,
            gradient_norm=[0.10] * count,
            opd_signal=[0.10] * count,
        ),
        expected_realizations=count,
    )


def _oracle_runs(value: float = 1.0):
    return [
        {
            "selection_target_seed": selection_seed,
            "evaluation_target_seed": 43 if selection_seed == 42 else 42,
            "kmeans_seed": kmeans_seed,
            "g_vendi_percentile": value,
            "coverage_percentile": value,
        }
        for selection_seed in (42, 43)
        for kmeans_seed in (42, 43)
    ]


def _passing_oracle():
    return evaluate_oracle_gate(_oracle_runs(), target_cka_p_value=0.01)


def _stage1_report(*, oracle_pass=True, pn1_g_vendi_median=0.95):
    oracle = _passing_oracle()
    if not oracle_pass:
        oracle = evaluate_oracle_gate(_oracle_runs(0.0), target_cka_p_value=0.01)
    pn1 = _passing_pn1()
    pn1 = copy.deepcopy(pn1)
    pn1["g_vendi"]["median"] = pn1_g_vendi_median
    return classify_stage1(
        oracle=oracle,
        pn1_components=pn1,
        pn4_components=_passing_strict(2),
        sft_components=_passing_strict(1),
        embedding_components=_passing_strict(1),
    )


def _pilot_records(*, overrides=None):
    base = {
        42: {
            "g_vendi": 0.90,
            "coverage": 0.90,
            "gradient_norm": 0.25,
            "opd_signal": 0.25,
        },
        43: {
            "g_vendi": 0.90,
            "coverage": 0.90,
            "gradient_norm": 0.25,
            "opd_signal": 0.25,
        },
    }
    for seed, values in (overrides or {}).items():
        base[seed].update(values)
    return base


def test_efficacy_pilot_go_gate_passes_both_exact_boundaries():
    result = classify_efficacy_pilot(_pilot_records())
    assert result["status"] == "pilot_only"
    assert result["decision"] == "go"
    assert result["main_hypothesis"] == "not_evaluated"
    assert result["stage1_thresholds_modified"] is False


@pytest.mark.parametrize(
    "overrides",
    (
        {43: {"g_vendi": 0.60}},
        {42: {"coverage": 0.60}},
        {42: {"g_vendi": 0.0}, 43: {"coverage": 1.0}},
    ),
)
def test_efficacy_pilot_no_go_uses_any_seed_diversity_or_coverage(overrides):
    assert classify_efficacy_pilot(
        _pilot_records(overrides=overrides)
    )["decision"] == "no_go"


@pytest.mark.parametrize(
    "overrides",
    (
        {42: {"gradient_norm": 0.249999}},
        {43: {"opd_signal": 0.249999}},
        {43: {"g_vendi": 0.899999}},
        {42: {"coverage": 0.600001}},
    ),
)
def test_efficacy_pilot_remaining_cases_are_borderline(overrides):
    assert classify_efficacy_pilot(
        _pilot_records(overrides=overrides)
    )["decision"] == "borderline"


def test_efficacy_pilot_gate_requires_exact_seeds_metrics_and_percentiles():
    missing_seed = _pilot_records()
    del missing_seed[43]
    with pytest.raises(ValueError, match="seeds 42 and 43"):
        classify_efficacy_pilot(missing_seed)
    extra_seed = {**_pilot_records(), 44: _pilot_records()[42]}
    with pytest.raises(ValueError, match="seeds 42 and 43"):
        classify_efficacy_pilot(extra_seed)

    for bad_record, message in (
        ({"diagnostic": 1.0}, "metric keys"),
        ({"stratified": 1.0}, "metric keys"),
        ({"g_vendi": float("nan")}, "finite"),
        ({"coverage": 1.1}, r"\[0, 1\]"),
        ({"gradient_norm": True}, "finite"),
    ):
        records = _pilot_records()
        records[42].update(bad_record)
        with pytest.raises(ValueError, match=message):
            classify_efficacy_pilot(records)


def test_exact_median_of_eight_uses_fourth_and_fifth_sorted_values():
    assert exact_median_of_eight([8, 1, 7, 2, 6, 3, 5, 4]) == 4.5
    with pytest.raises(ValueError, match="eight finite"):
        exact_median_of_eight([1.0] * 7)
    with pytest.raises(ValueError, match="eight finite"):
        exact_median_of_eight([1.0] * 7 + [float("nan")])


def test_exact_pn1_thresholds_pass():
    components = evaluate_pn1_components(
        _worst_cases(
            g_vendi=[0.89, 0.89, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95],
            coverage=[0.09, 0.10, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50],
            gradient_norm=[0, 0, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
            opd_signal=[0, 0, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
        )
    )
    assert components["g_vendi"]["pass"] is True
    assert components["g_vendi"]["count_at_or_above_secondary"] == 6
    assert components["coverage"]["pass"] is True
    assert components["coverage"]["count_at_or_above_secondary"] == 7
    assert components["gradient_norm"]["pass"] is True
    assert components["opd_signal"]["pass"] is True
    assert components["all_pass"] is True


def test_pn1_g_vendi_requires_six_of_eight_even_when_median_passes():
    components = evaluate_pn1_components(
        _worst_cases(
            g_vendi=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            coverage=[1.0] * 8,
            gradient_norm=[1.0] * 8,
            opd_signal=[1.0] * 8,
        )
    )
    assert components["g_vendi"]["median"] == 1.0
    assert components["g_vendi"]["count_at_or_above_secondary"] == 5
    assert components["g_vendi"]["pass"] is False


def test_pn1_coverage_requires_seven_of_eight_even_when_median_passes():
    components = evaluate_pn1_components(
        _worst_cases(
            g_vendi=[1.0] * 8,
            coverage=[0.0, 0.0, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50],
            gradient_norm=[1.0] * 8,
            opd_signal=[1.0] * 8,
        )
    )
    assert components["coverage"]["median"] == 0.50
    assert components["coverage"]["count_at_or_above_secondary"] == 6
    assert components["coverage"]["pass"] is False


def test_primary_corner_minimum_is_retained_and_diagnostics_cannot_rescue_it():
    records = _worst_cases(
        g_vendi=[1.0] * 8,
        coverage=[1.0] * 8,
        gradient_norm=[1.0] * 8,
        opd_signal=[1.0] * 8,
    )
    records["r0"] = {
        "primary_corners": [
            {
                "target_seed": target_seed,
                "kmeans_seed": kmeans_seed,
                "g_vendi": 0.0 if (target_seed, kmeans_seed) == (43, 43) else 1.0,
                "coverage": 1.0,
                "gradient_norm": 1.0,
                "opd_signal": 1.0,
            }
            for target_seed in (42, 43)
            for kmeans_seed in (42, 43)
        ],
        "diagnostic_corners": [{"g_vendi": 1.0}],
        "stratified": {"g_vendi": 1.0},
    }
    components = evaluate_pn1_components(records)
    assert components["worst_by_realization"]["r0"]["g_vendi"] == 0.0
    assert components["g_vendi"]["values_by_realization"]["r0"] == 0.0


def test_low_stratified_and_diagnostic_values_do_not_change_primary_components():
    records = _worst_cases(
        g_vendi=[1.0] * 8,
        coverage=[1.0] * 8,
        gradient_norm=[1.0] * 8,
        opd_signal=[1.0] * 8,
    )
    for record in records.values():
        record["diagnostic_corners"] = [{"g_vendi": 0.0, "coverage": 0.0}]
        record["stratified"] = {
            "g_vendi": 0.0,
            "coverage": 0.0,
            "gradient_norm": 0.0,
            "opd_signal": 0.0,
        }
    assert evaluate_pn1_components(records)["all_pass"] is True


@pytest.mark.parametrize(
    ("metric", "threshold"),
    [("g_vendi", 0.95), ("coverage", 0.50), ("gradient_norm", 0.10), ("opd_signal", 0.10)],
)
def test_strict_arm_exact_boundaries_pass_and_just_below_fails(metric, threshold):
    exact = _worst_cases(
        g_vendi=[0.95],
        coverage=[0.50],
        gradient_norm=[0.10],
        opd_signal=[0.10],
    )
    assert evaluate_strict_arm_components(
        exact, expected_realizations=1
    )[metric]["pass"] is True
    below = copy.deepcopy(exact)
    below["r0"][metric] = threshold - 1e-6
    assert evaluate_strict_arm_components(
        below, expected_realizations=1
    )[metric]["pass"] is False


def test_strict_arms_require_every_realization_at_all_four_thresholds():
    passing = _passing_strict(2)
    assert passing["all_pass"] is True
    records = _worst_cases(
        g_vendi=[0.95, 0.949999],
        coverage=[0.50, 0.50],
        gradient_norm=[0.10, 0.10],
        opd_signal=[0.10, 0.10],
    )
    failing = evaluate_strict_arm_components(records, expected_realizations=2)
    assert failing["g_vendi"]["pass"] is False
    assert failing["all_pass"] is False
    with pytest.raises(ValueError, match="realizations"):
        evaluate_strict_arm_components(records, expected_realizations=1)


def test_oracle_gate_requires_four_cross_seed_runs_and_exact_boundaries():
    exact_runs = _oracle_runs()
    for run in exact_runs:
        run["g_vendi_percentile"] = 0.95
        run["coverage_percentile"] = 0.50
    gate = evaluate_oracle_gate(exact_runs, target_cka_p_value=0.01)
    assert gate["classification"] == "pass"
    assert gate["cross_seed_selection_pass"] is True
    assert gate["target_dependence_pass"] is True

    cka_failure = evaluate_oracle_gate(_oracle_runs(), target_cka_p_value=0.010001)
    assert cka_failure["classification"] == "fail"
    assert cka_failure["failed_components"] == ["target_dependence_cka"]

    run_failure = _oracle_runs()
    run_failure[0]["g_vendi_percentile"] = 0.949999
    gate = evaluate_oracle_gate(run_failure, target_cka_p_value=0.01)
    assert gate["classification"] == "fail"
    assert "cross_seed_oracle_selection" in gate["failed_components"]


def test_oracle_rejects_same_seed_or_missing_seed_corner():
    runs = _oracle_runs()
    runs[0]["evaluation_target_seed"] = runs[0]["selection_target_seed"]
    with pytest.raises(ValueError, match="cross-seed"):
        evaluate_oracle_gate(runs, target_cka_p_value=0.01)
    with pytest.raises(ValueError, match="four"):
        evaluate_oracle_gate(_oracle_runs()[:3], target_cka_p_value=0.01)


def test_oracle_failure_makes_all_arms_inconclusive_but_preserves_components():
    failing_oracle = evaluate_oracle_gate(_oracle_runs(0.0), target_cka_p_value=0.01)
    failing_pn1 = _passing_pn1()
    failing_pn1 = copy.deepcopy(failing_pn1)
    failing_pn1["coverage"]["pass"] = False
    failing_pn1["all_pass"] = False
    report = classify_stage1(
        oracle=failing_oracle,
        pn1_components=failing_pn1,
        pn4_components=_passing_strict(2),
        sft_components=_passing_strict(1),
        embedding_components=_passing_strict(1),
    )
    assert report["classification"] == "inconclusive"
    for arm in ("P_n1", "P_n4", "S", "E"):
        assert report[arm]["classification"] == "inconclusive"
        assert "components" in report[arm]
    assert report["P_n1"]["components"]["coverage"]["pass"] is False


def test_valid_oracle_classifies_each_arm_independently():
    pn1 = _passing_pn1()
    pn4 = evaluate_strict_arm_components(
        _worst_cases(
            g_vendi=[1.0, 1.0],
            coverage=[1.0, 0.49],
            gradient_norm=[1.0, 1.0],
            opd_signal=[1.0, 1.0],
        ),
        expected_realizations=2,
    )
    embedding = evaluate_strict_arm_components(
        _worst_cases(
            g_vendi=[0.94],
            coverage=[1.0],
            gradient_norm=[1.0],
            opd_signal=[1.0],
        ),
        expected_realizations=1,
    )
    report = classify_stage1(
        oracle=_passing_oracle(),
        pn1_components=pn1,
        pn4_components=pn4,
        sft_components=_passing_strict(1),
        embedding_components=embedding,
    )
    assert report["classification"] == "classified"
    assert report["P_n1"]["classification"] == "pass"
    assert report["P_n4"]["classification"] == "fail"
    assert report["S"]["classification"] == "pass"
    assert report["E"]["classification"] == "fail"


@pytest.mark.parametrize(
    ("median", "expected"),
    [(0.899999, False), (0.90, True), (0.949999, True), (0.95, False)],
)
def test_stage2_trigger_uses_half_open_gvendi_interval(median, expected):
    report = _stage1_report(oracle_pass=True, pn1_g_vendi_median=median)
    assert decide_stage2(report)["run_stage2"] is expected


def test_stage2_trigger_ignores_non_gvendi_candidate_failure_but_requires_oracle():
    report = _stage1_report(oracle_pass=True, pn1_g_vendi_median=0.92)
    report["P_n1"]["components"]["coverage"]["pass"] = False
    report["P_n1"]["classification"] = "fail"
    assert decide_stage2(report) == {
        "run_stage2": True,
        "reason": "borderline_primary_pn1_g_vendi",
    }
    failed_oracle = _stage1_report(oracle_pass=False, pn1_g_vendi_median=0.92)
    assert decide_stage2(failed_oracle) == {
        "run_stage2": False,
        "reason": "oracle_not_pass",
    }


def test_stage2_sensitivity_has_separate_labels_and_never_mutates_stage1():
    stage1 = _stage1_report(oracle_pass=True, pn1_g_vendi_median=0.92)
    before = copy.deepcopy(stage1)
    passing = classify_stage2_sensitivity(
        stage1_report=stage1,
        expanded_oracle=_passing_oracle(),
        expanded_pn1_components=_passing_pn1(),
    )
    assert passing["classification"] == "extended_pass"

    failed_components = copy.deepcopy(_passing_pn1())
    failed_components["coverage"]["pass"] = False
    failed_components["all_pass"] = False
    failing = classify_stage2_sensitivity(
        stage1_report=stage1,
        expanded_oracle=_passing_oracle(),
        expanded_pn1_components=failed_components,
    )
    assert failing["classification"] == "extended_fail"

    inconclusive = classify_stage2_sensitivity(
        stage1_report=stage1,
        expanded_oracle=evaluate_oracle_gate(
            _oracle_runs(), target_cka_p_value=0.02
        ),
        expanded_pn1_components=_passing_pn1(),
    )
    assert inconclusive["classification"] == "extended_inconclusive"
    assert stage1 == before
