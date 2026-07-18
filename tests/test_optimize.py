import math
from itertools import product

import numpy as np
import pandas as pd
import pulp
import pytest

import tailguard.optimize as optimize_module
from tailguard.bom import SupplierOption
from tailguard.optimize import (
    OptimizationError,
    default_wasserstein_radius,
    solve_benders_mean_cvar,
    solve_mean_cvar_milp,
    solve_spectral_cvar_mixture,
    solve_wasserstein_cvar,
)
from tailguard.risk import normalized_weights, weighted_cvar, weighted_mean


def _option(component, supplier, base, kappa, lead_time=1, source_type="offshore"):
    return SupplierOption(component, supplier, base, kappa, lead_time, source_type)


def _small_bom():
    return {
        "A": (
            _option("A", "A_low_base", 20, 18, 8, "offshore"),
            _option("A", "A_low_kappa", 38, 2, 2, "domestic"),
        ),
        "B": (
            _option("B", "B_low_base", 15, 15, 6, "offshore"),
            _option("B", "B_low_kappa", 31, 1, 3, "domestic"),
        ),
    }


@pytest.mark.parametrize(
    "available,expected",
    [
        (["PULP_CBC_CMD", "COIN_CMD", "HiGHS", "HiGHS_CMD"], "HiGHS_CMD"),
        (["PULP_CBC_CMD", "COIN_CMD", "HiGHS"], "HiGHS"),
        (["PULP_CBC_CMD", "COIN_CMD"], "COIN_CMD"),
        (["PULP_CBC_CMD"], "PULP_CBC_CMD"),
    ],
)
def test_solver_selection_prefers_native_integrations(monkeypatch, available, expected):
    selected = []
    sentinel = object()
    monkeypatch.setattr(optimize_module.pulp, "listSolvers", lambda onlyAvailable: available)
    monkeypatch.setattr(
        optimize_module.pulp,
        "getSolver",
        lambda name, **options: selected.append((name, options)) or sentinel,
    )
    assert optimize_module._solver() is sentinel
    expected_options = {"msg": False}
    if (
        expected == "PULP_CBC_CMD"
        and "_skip_v4_deprecation"
        in optimize_module.inspect.signature(
            optimize_module.pulp.PULP_CBC_CMD
        ).parameters
    ):
        expected_options["_skip_v4_deprecation"] = True
    assert selected == [(expected, expected_options)]


def test_bundled_cbc_constructor_suppresses_the_pulp_332_deprecation(monkeypatch):
    class FutureBundledCBC:
        def __init__(self, msg=True, _skip_v4_deprecation=False):
            pass

    selected = []
    sentinel = object()
    monkeypatch.setattr(optimize_module.pulp, "listSolvers", lambda onlyAvailable: ["PULP_CBC_CMD"])
    monkeypatch.setattr(optimize_module.pulp, "PULP_CBC_CMD", FutureBundledCBC)
    monkeypatch.setattr(
        optimize_module.pulp,
        "getSolver",
        lambda name, **options: selected.append((name, options)) or sentinel,
    )
    assert optimize_module._solver() is sentinel
    assert selected == [
        ("PULP_CBC_CMD", {"msg": False, "_skip_v4_deprecation": True})
    ]


def test_solver_start_failures_are_reported_without_exposing_executable_paths(monkeypatch):
    class BrokenProblem:
        def solve(self, _solver):
            raise pulp.PulpSolverError("failure at /opt/example/solver/path")

    monkeypatch.setattr(optimize_module, "_solver", lambda: object())
    with pytest.raises(OptimizationError, match="compatible native") as error:
        optimize_module._solve_or_raise(BrokenProblem())
    assert "/opt/example" not in str(error.value)


def _enumerate_objective(bom, events, alpha, risk_weight, weights=None):
    probabilities = normalized_weights(weights, len(events))
    candidates = list(product(*bom.values()))
    scored = []
    for policy in candidates:
        losses = np.sum([option.base_cost + option.kappa * np.asarray(events) for option in policy], axis=0)
        objective = weighted_mean(losses, probabilities) + risk_weight * weighted_cvar(losses, alpha, probabilities)
        scored.append((objective, tuple(option.supplier for option in policy)))
    return min(scored)


def test_nominal_milp_matches_exhaustive_enumeration():
    bom = _small_bom()
    events = np.array([0, 0, 1, 2, 4, 0, 3], dtype=float)
    expected_objective, expected_policy = _enumerate_objective(bom, events, 0.8, 1.3)
    result = solve_mean_cvar_milp(bom, events, alpha=0.8, risk_weight=1.3)
    assert result.objective == pytest.approx(expected_objective)
    assert tuple(option.supplier for option in result.selected) == expected_policy


def test_uniform_scenario_permutations_cannot_flip_an_exact_certified_policy():
    uniform_mean = 1.0 / 49
    bom = {
        "A": (
            _option("A", "constant", np.nextafter(uniform_mean, np.inf), 0.0),
            _option("A", "shock", 0.0, 1.0),
        )
    }
    for position in (0, 1, 24, 48):
        events = np.zeros(49)
        events[position] = 1.0
        result = solve_mean_cvar_milp(bom, events, alpha=0.8, risk_weight=0.0)
        assert result.selected[0].supplier == "shock"
        assert result.details["componentwise_optimality_certified"] is True


def test_weighted_stratified_objective_matches_weighted_enumeration():
    bom = _small_bom()
    events = np.array([0, 1, 4, 7], dtype=float)
    weights = np.array([0.75, 0.15, 0.08, 0.02])
    expected_objective, expected_policy = _enumerate_objective(bom, events, 0.8, 1.0, weights)
    result = solve_mean_cvar_milp(bom, events, alpha=0.8, risk_weight=1.0, scenario_weights=weights)
    assert result.objective == pytest.approx(expected_objective)
    assert tuple(option.supplier for option in result.selected) == expected_policy
    assert result.mean_cost == pytest.approx(weighted_mean(result.losses, weights))


def test_benders_matches_nominal_with_discrete_var_ties():
    events = np.r_[np.zeros(480), np.ones(20)]
    bom = {
        "A": (
            _option("A", "A", 51, 28),
            _option("A", "B", 74, 3),
            _option("A", "C", 60, 1),
            _option("A", "D", 52, 19),
        )
    }
    nominal = solve_mean_cvar_milp(bom, events, alpha=0.95)
    benders = solve_benders_mean_cvar(bom, events, alpha=0.95, relative_tolerance=1e-10)
    assert nominal.objective == pytest.approx(119.96)
    assert benders.objective == pytest.approx(nominal.objective)
    assert benders.selected[0].supplier == "D"
    assert benders.details["converged"] is True
    assert benders.details["lower_bound"] <= benders.details["upper_bound"] + 1e-7
    assert benders.details["gap"] == (
        benders.details["upper_bound"] - benders.details["lower_bound"]
    )
    assert benders.details["relative_gap"] == (
        benders.details["gap"] / max(1.0, abs(benders.details["upper_bound"]))
    )


@pytest.mark.parametrize("seed", range(5))
def test_benders_matches_nominal_on_seeded_small_instances(seed):
    events = np.random.default_rng(seed).integers(0, 5, size=17)
    nominal = solve_mean_cvar_milp(_small_bom(), events, alpha=0.77, risk_weight=0.6)
    benders = solve_benders_mean_cvar(_small_bom(), events, alpha=0.77, risk_weight=0.6, relative_tolerance=1e-10)
    assert benders.objective == pytest.approx(nominal.objective)


@pytest.mark.parametrize("seed", range(8))
def test_benders_matches_nominal_on_weighted_randomized_instances(seed):
    rng = np.random.default_rng(seed + 100)
    events = rng.integers(0, 10, size=13)
    weights = rng.uniform(0.001, 1.0, size=len(events))
    alpha = float(rng.uniform(0.1, 0.9))
    risk_weight = float(rng.uniform(0.1, 4.0))
    lead_time_cost_per_day = float(rng.uniform(0.0, 5.0))
    nominal = solve_mean_cvar_milp(
        _small_bom(),
        events,
        alpha=alpha,
        risk_weight=risk_weight,
        scenario_weights=weights,
        lead_time_cost_per_day=lead_time_cost_per_day,
    )
    benders = solve_benders_mean_cvar(
        _small_bom(),
        events,
        alpha=alpha,
        risk_weight=risk_weight,
        scenario_weights=weights,
        lead_time_cost_per_day=lead_time_cost_per_day,
    )
    assert benders.objective == pytest.approx(nominal.objective)
    assert tuple(option.supplier for option in benders.selected) == tuple(
        option.supplier for option in nominal.selected
    )


def test_nested_probability_reuse_cannot_flip_a_near_tied_benders_reference():
    weights = [
        0.12167778692709874,
        0.47588257255962463,
        0.5852880835480021,
        0.4174560432581012,
        0.23470441637430708,
        0.770462881943687,
        0.8204337324439215,
        0.77208358198521,
        0.471993265203288,
        0.1010732080384683,
        0.8142804184386284,
    ]
    events = [3, 1, 0, 3, 3, 0, 1, 3, 2, 2, 0]
    bom = {
        "A": (
            _option("A", "lower_exact_score", 1789988196030.1853, 0),
            _option("A", "higher_exact_score", 0, 1e12),
        )
    }
    nominal = solve_mean_cvar_milp(
        bom,
        events,
        alpha=0.5,
        risk_weight=1,
        scenario_weights=weights,
    )
    benders = solve_benders_mean_cvar(
        bom,
        events,
        alpha=0.5,
        risk_weight=1,
        scenario_weights=weights,
    )
    assert nominal.selected[0].supplier == "lower_exact_score"
    assert benders.selected == nominal.selected


def test_wasserstein_zero_radius_equals_nominal_and_positive_radius_links_tau():
    bom = _small_bom()
    events = [0, 0, 1, 2, 4, 0, 3]
    nominal = solve_mean_cvar_milp(bom, events, alpha=0.8)
    zero = solve_wasserstein_cvar(bom, events, alpha=0.8, epsilon=0.0)
    robust = solve_wasserstein_cvar(bom, events, alpha=0.8, epsilon=0.25)
    assert zero.objective == pytest.approx(nominal.objective)
    assert robust.objective >= zero.objective
    assert robust.details["tau"] == pytest.approx(robust.details["expected_tau"])
    assert robust.details["tau"] > 0


@pytest.mark.parametrize("risk_weight", [0.0, 1e-308])
def test_wasserstein_adds_the_exact_increment_before_final_float_rounding(
    risk_weight,
):
    maximum = np.finfo(float).max
    penultimate = np.nextafter(maximum, 0.0)
    bom = {
        "A": (
            _option("A", "only", penultimate, 1.2474001934592e291, 0.0),
        )
    }

    result = solve_wasserstein_cvar(
        bom,
        [0.0, 1.0, 10.0],
        alpha=0.3,
        epsilon=2.0,
        risk_weight=risk_weight,
    )

    assert result.details["nominal_cvar"] == penultimate
    assert result.risk_cost == maximum


def test_zero_wasserstein_radius_does_not_require_an_irrelevant_representable_tau():
    bom = {"A": (_option("A", "huge_exposure", 0, 1e308),)}
    nominal = solve_mean_cvar_milp(bom, [0], alpha=0.5)
    zero = solve_wasserstein_cvar(bom, [0], epsilon=0, alpha=0.5)
    assert zero.selected == nominal.selected
    assert zero.objective == nominal.objective == 0
    assert zero.details["bypassed_for_zero_radius"] is True
    assert zero.details["tau"] is None
    assert zero.details["tau_not_representable"] is True


def test_zero_risk_weight_does_not_require_an_unused_representable_w1_tau():
    bom = {"A": (_option("A", "huge_exposure", 0, 1e308),)}
    result = solve_wasserstein_cvar(bom, [0], epsilon=0.1, alpha=0.5, risk_weight=0)
    assert result.objective == 0
    assert result.risk_cost == pytest.approx(2e307)
    assert result.details["bypassed_for_zero_risk_weight"] is True
    assert result.details["tau"] is None
    assert result.details["tau_not_representable"] is True


def test_w1_bypasses_do_not_sum_irrelevant_multi_component_exposure_as_a_float():
    bom = {
        "A": (_option("A", "huge_a", 0, 1e308),),
        "B": (_option("B", "huge_b", 0, 1e308),),
    }
    nominal = solve_mean_cvar_milp(bom, [0], alpha=0.5)
    zero_radius = solve_wasserstein_cvar(bom, [0], epsilon=0, alpha=0.5)
    zero_weight = solve_wasserstein_cvar(
        bom,
        [0],
        epsilon=1e-308,
        alpha=0.5,
        risk_weight=0,
    )

    assert nominal.objective == zero_radius.objective == zero_weight.objective == 0
    assert zero_radius.details["bypassed_for_zero_radius"] is True
    assert zero_weight.risk_cost == pytest.approx(4.0)
    assert zero_weight.details["bypassed_for_zero_risk_weight"] is True
    assert zero_weight.details["shock_exposure"] is None
    assert zero_weight.details["tau"] is None


def test_finite_cvar_mixture_matches_direct_evaluation():
    bom = _small_bom()
    events = [0, 0, 1, 2, 4, 0, 3]
    levels = [0.6, 0.8]
    mix = [0.4, 0.6]
    result = solve_spectral_cvar_mixture(bom, events, levels=levels, mixture_weights=mix, risk_weight=0.75)
    expected_risk = sum(weight * weighted_cvar(result.losses, level) for level, weight in zip(levels, mix))
    assert result.risk_cost == pytest.approx(expected_risk)
    assert result.objective == pytest.approx(result.mean_cost + 0.75 * expected_risk)


@pytest.mark.parametrize("risk_weight", [0.0, 1e-308])
def test_spectral_mixture_combines_exact_cvars_before_final_float_rounding(
    risk_weight,
):
    maximum = np.finfo(float).max
    penultimate = np.nextafter(maximum, 0.0)
    bom = {
        "A": (
            _option("A", "only", penultimate, 1.2474001934592e291, 0.0),
        )
    }

    result = solve_spectral_cvar_mixture(
        bom,
        [0.0, 1.0, 10.0],
        levels=(0.2, 0.6),
        mixture_weights=(0.5, 0.5),
        risk_weight=risk_weight,
    )

    assert result.details["per_level_cvar"].tolist() == [penultimate, maximum]
    assert result.risk_cost == maximum


@pytest.mark.parametrize(
    "levels,mixture_weights",
    [
        (0.5, [1.0]),
        ([0.5], 1.0),
        (np.asarray([[0.5]]), [1.0]),
    ],
)
def test_spectral_inputs_reject_noniterable_or_wrong_dimensional_levels_cleanly(
    levels, mixture_weights
):
    with pytest.raises(OptimizationError):
        solve_spectral_cvar_mixture(
            _small_bom(),
            [0, 1],
            levels=levels,
            mixture_weights=mixture_weights,
        )


def test_spectral_and_w1_variants_match_exhaustive_policy_selection():
    bom = _small_bom()
    events = np.array([0, 1, 3, 6], dtype=float)
    weights = np.array([0.55, 0.2, 0.15, 0.1])
    probabilities = normalized_weights(weights, len(events))
    alpha = 0.8
    risk_weight = 0.7
    levels = [0.6, 0.8]
    mixture_weights = [0.25, 0.75]
    epsilon = 0.3
    spectral_scores = []
    robust_scores = []
    for policy in product(*bom.values()):
        losses = np.sum([option.base_cost + option.kappa * events for option in policy], axis=0)
        mean = weighted_mean(losses, probabilities)
        spectral_risk = sum(
            weight * weighted_cvar(losses, level, probabilities) for level, weight in zip(levels, mixture_weights)
        )
        robust_risk = weighted_cvar(losses, alpha, probabilities) + epsilon * sum(option.kappa for option in policy) / (
            1.0 - alpha
        )
        suppliers = tuple(option.supplier for option in policy)
        spectral_scores.append((mean + risk_weight * spectral_risk, suppliers))
        robust_scores.append((mean + risk_weight * robust_risk, suppliers))

    spectral = solve_spectral_cvar_mixture(
        bom,
        events,
        levels=levels,
        mixture_weights=mixture_weights,
        risk_weight=risk_weight,
        scenario_weights=weights,
    )
    robust = solve_wasserstein_cvar(
        bom,
        events,
        epsilon=epsilon,
        alpha=alpha,
        risk_weight=risk_weight,
        scenario_weights=weights,
    )
    expected_spectral = min(spectral_scores)
    expected_robust = min(robust_scores)
    assert spectral.objective == pytest.approx(expected_spectral[0])
    assert tuple(option.supplier for option in spectral.selected) == expected_spectral[1]
    assert robust.objective == pytest.approx(expected_robust[0])
    assert tuple(option.supplier for option in robust.selected) == expected_robust[1]


def test_lead_time_limit_and_explicit_penalty_change_feasible_costs():
    bom = _small_bom()
    events = [0, 1]
    constrained = solve_mean_cvar_milp(bom, events, max_lead_time_days=3)
    assert all(option.lead_time <= 3 for option in constrained.selected)
    base = solve_mean_cvar_milp(bom, events)
    penalized = solve_mean_cvar_milp(bom, events, lead_time_cost_per_day=10)
    assert penalized.objective >= base.objective
    with pytest.raises(OptimizationError):
        solve_mean_cvar_milp(bom, events, max_lead_time_days=0)


def test_zero_risk_weight_is_supported_by_decomposed_and_robust_variants():
    bom = _small_bom()
    events = [0, 1, 2]
    nominal = solve_mean_cvar_milp(bom, events, risk_weight=0)
    benders = solve_benders_mean_cvar(bom, events, risk_weight=0)
    robust = solve_wasserstein_cvar(bom, events, epsilon=0.2, risk_weight=0)
    assert benders.objective == pytest.approx(nominal.objective)
    assert robust.objective == pytest.approx(nominal.objective)
    assert benders.details["bypassed_for_zero_risk_weight"] is True
    assert robust.details["bypassed_for_zero_risk_weight"] is True


@pytest.mark.parametrize(
    "bom,expected",
    [
        (
            {
                "A": (
                    _option("A", "zero_one", 0, 0),
                    _option("A", "zero_two", 0, 0),
                )
            },
            0.0,
        ),
        ({"A": (_option("A", "only_common_floor", 7, 3),)}, 8.5),
    ],
)
def test_solver_variants_handle_constant_zero_conditioned_objectives(bom, expected):
    events = [0, 1]
    results = (
        solve_mean_cvar_milp(bom, events, alpha=0.5, risk_weight=0),
        solve_benders_mean_cvar(bom, events, alpha=0.5, risk_weight=0),
        solve_spectral_cvar_mixture(
            bom,
            events,
            levels=[0.5],
            mixture_weights=[1],
            risk_weight=0,
        ),
        solve_wasserstein_cvar(bom, events, epsilon=0.2, alpha=0.5, risk_weight=0),
    )
    for result in results:
        assert result.objective == pytest.approx(expected)
        if expected == 0.0:
            assert result.selected[0].supplier == "zero_one"


def test_solver_variants_ignore_zero_probability_extremes_without_losing_small_costs():
    bom = {"A": (_option("A", "only", 0, 1),)}
    events = [1e308, 0.0, 1e-20]
    weights = [0.0, 0.0, 1.0]
    results = (
        solve_mean_cvar_milp(bom, events, alpha=0.5, risk_weight=0.5, scenario_weights=weights),
        solve_benders_mean_cvar(bom, events, alpha=0.5, risk_weight=0.5, scenario_weights=weights),
        solve_spectral_cvar_mixture(
            bom,
            events,
            levels=[0.5],
            mixture_weights=[1],
            risk_weight=0.5,
            scenario_weights=weights,
        ),
        solve_wasserstein_cvar(
            bom,
            events,
            epsilon=0,
            alpha=0.5,
            risk_weight=0.5,
            scenario_weights=weights,
        ),
    )
    for result in results:
        assert result.mean_cost == 1e-20
        assert result.risk_cost == 1e-20
        assert result.objective == pytest.approx(1.5e-20, rel=1e-12, abs=0)
        assert np.array_equal(result.losses, np.asarray(events, dtype=float))

    full_risk_weight = solve_mean_cvar_milp(
        bom,
        events,
        alpha=0.5,
        risk_weight=1.0,
        scenario_weights=weights,
    )
    assert full_risk_weight.objective == pytest.approx(2e-20, rel=1e-12, abs=0)
    assert np.array_equal(full_risk_weight.losses, np.asarray(events, dtype=float))


def test_benders_falls_back_to_the_certified_nominal_reference_when_limited():
    bom = _small_bom()
    events = [0, 0, 1, 2, 4, 0, 3]
    nominal = solve_mean_cvar_milp(bom, events, alpha=0.8)
    result = solve_benders_mean_cvar(bom, events, alpha=0.8, max_iterations=1)
    assert result.objective == pytest.approx(nominal.objective)
    assert result.details["fallback_to_nominal"] is True
    assert result.details["gap"] == (
        result.details["upper_bound"] - result.details["lower_bound"]
    )
    assert result.details["relative_gap"] == (
        result.details["gap"] / max(1.0, abs(result.details["upper_bound"]))
    )


def test_benders_uses_nominal_reference_when_float_dual_bounds_are_unrepresentable():
    bom = {"A": (_option("A", "only", 0.0, 1.0),)}
    weights = [np.nextafter(0.0, 1.0), 1.0]
    nominal = solve_mean_cvar_milp(bom, [1.0, 0.0], alpha=0.5, scenario_weights=weights)
    result = solve_benders_mean_cvar(bom, [1.0, 0.0], alpha=0.5, scenario_weights=weights)
    assert result.objective == nominal.objective
    assert result.selected == nominal.selected
    assert result.details["fallback_to_nominal"] is True
    assert result.details["fallback_reason"] == "subnormal_probability_dual_not_representable"


def test_benders_tolerates_solver_serialization_rounding_and_verifies_against_reference():
    bom = {"A": (_option("A", "only", 1, 16),)}
    nominal = solve_mean_cvar_milp(bom, [0, 1], alpha=0.85, risk_weight=8, scenario_weights=[0.9, 0.1])
    result = solve_benders_mean_cvar(bom, [0, 1], alpha=0.85, risk_weight=8, scenario_weights=[0.9, 0.1])
    assert result.objective == pytest.approx(nominal.objective)
    assert result.details.get("reference_verified", False) or result.details.get("fallback_to_nominal", False)


def test_component_names_that_collide_after_pulp_sanitization_are_valid_input():
    bom = {
        "A-B": (_option("A-B", "one", 1, 1),),
        "A_B": (_option("A_B", "two", 2, 1),),
    }
    result = solve_mean_cvar_milp(bom, [0, 1])
    assert {option.component for option in result.selected} == {"A-B", "A_B"}


def test_milp_variants_reject_numerically_unsafe_tail_probabilities():
    near_one = np.nextafter(1.0, 0.0)
    near_zero = np.nextafter(0.0, 1.0)
    bom = _small_bom()
    with pytest.raises(OptimizationError, match="too close to 1"):
        solve_mean_cvar_milp(bom, [0, 1], alpha=near_one)
    with pytest.raises(OptimizationError, match="too close to 1"):
        solve_benders_mean_cvar(bom, [0, 1], alpha=near_one)
    with pytest.raises(OptimizationError, match="too close to 1"):
        solve_spectral_cvar_mixture(bom, [0, 1], levels=[near_one], mixture_weights=[1])
    with pytest.raises(OptimizationError, match="too close to 1"):
        solve_wasserstein_cvar(bom, [0, 1], alpha=near_one, epsilon=0)
    with pytest.raises(OptimizationError, match="too close to 0"):
        solve_mean_cvar_milp(bom, [0, 1], alpha=near_zero)


def test_cost_overflow_is_reported_before_constructing_a_solver_model():
    bom = {"A": (_option("A", "huge", 1e308, 1e308),)}
    with pytest.raises(OptimizationError, match="rescale monetary inputs"):
        solve_mean_cvar_milp(bom, [2])


def test_zero_mass_extreme_is_not_evaluated_for_unselected_eligible_options():
    bom = {
        "A": (
            _option("A", "safe", 1e-20, 0.0),
            _option("A", "irrelevant_overflow", 0.0, 2.0),
        )
    }
    events = [1e-20, 1e308, 0.0]
    weights = [1.0, 0.0, 0.0]
    results = (
        solve_mean_cvar_milp(
            bom, events, alpha=0.5, risk_weight=1.0, scenario_weights=weights
        ),
        solve_benders_mean_cvar(
            bom, events, alpha=0.5, risk_weight=1.0, scenario_weights=weights
        ),
        solve_spectral_cvar_mixture(
            bom,
            events,
            levels=[0.5],
            mixture_weights=[1.0],
            risk_weight=1.0,
            scenario_weights=weights,
        ),
        solve_wasserstein_cvar(
            bom,
            events,
            epsilon=0.1,
            alpha=0.5,
            risk_weight=1.0,
            scenario_weights=weights,
        ),
    )
    for result in results:
        assert result.selected[0].supplier == "safe"
        assert result.losses.tolist() == [1e-20, 1e-20, 1e-20]
        assert result.objective == pytest.approx(2e-20, rel=1e-12, abs=0)
        assert result.details["componentwise_optimality_certified"] is True
        assert np.array_equal(result.details["scenario_weights"], np.asarray(weights))


def test_selected_policy_with_unrepresentable_zero_mass_loss_is_rejected_cleanly():
    bom = {"A": (_option("A", "only", 0.0, 2.0),)}
    calls = (
        lambda: solve_mean_cvar_milp(
            bom, [1e-20, 1e308], alpha=0.5, scenario_weights=[1.0, 0.0]
        ),
        lambda: solve_benders_mean_cvar(
            bom, [1e-20, 1e308], alpha=0.5, scenario_weights=[1.0, 0.0]
        ),
        lambda: solve_spectral_cvar_mixture(
            bom,
            [1e-20, 1e308],
            levels=[0.5],
            mixture_weights=[1.0],
            scenario_weights=[1.0, 0.0],
        ),
        lambda: solve_wasserstein_cvar(
            bom,
            [1e-20, 1e308],
            epsilon=0.1,
            alpha=0.5,
            scenario_weights=[1.0, 0.0],
        ),
    )
    for call in calls:
        with pytest.raises(OptimizationError, match="selected policy produces non-finite"):
            call()


def test_solver_variants_do_not_evaluate_numerically_extreme_ineligible_options():
    bom = {
        "A": (
            _option("A", "eligible", 1.0, 0.0, lead_time=1.0),
            _option("A", "excluded_extreme", 1e308, 1e308, lead_time=100.0),
        )
    }
    results = (
        solve_mean_cvar_milp(bom, [2.0], alpha=0.5, max_lead_time_days=10.0),
        solve_benders_mean_cvar(bom, [2.0], alpha=0.5, max_lead_time_days=10.0),
        solve_spectral_cvar_mixture(
            bom,
            [2.0],
            levels=[0.5],
            mixture_weights=[1.0],
            max_lead_time_days=10.0,
        ),
        solve_wasserstein_cvar(
            bom,
            [2.0],
            epsilon=0.1,
            alpha=0.5,
            max_lead_time_days=10.0,
        ),
    )
    for result in results:
        assert result.selected[0].supplier == "eligible"
        assert result.losses.tolist() == [1.0]
        assert result.objective == 2.0


def test_common_large_cost_offset_cannot_hide_the_strictly_cheaper_policy():
    bom = {
        "A": (
            _option("A", "more_expensive", 1e13 + 1, 0),
            _option("A", "less_expensive", 1e13, 0),
        )
    }
    expected_objective = 2e13
    results = (
        solve_mean_cvar_milp(bom, [0, 1], alpha=0.5, risk_weight=1),
        solve_benders_mean_cvar(bom, [0, 1], alpha=0.5, risk_weight=1),
        solve_spectral_cvar_mixture(
            bom,
            [0, 1],
            levels=[0.5],
            mixture_weights=[1],
            risk_weight=1,
        ),
        solve_wasserstein_cvar(bom, [0, 1], epsilon=0, alpha=0.5, risk_weight=1),
    )
    for result in results:
        assert result.selected[0].supplier == "less_expensive"
        assert result.objective == expected_objective
        assert result.details["componentwise_optimality_certified"] is True


@pytest.mark.parametrize("second_base", [0.0, 1.0])
def test_exact_certificate_resolves_tradeoffs_hidden_by_float_objective_rounding(second_base):
    bom = {
        "A": (
            _option("A", "lower_exact_score", 1e16, 0),
            _option("A", "higher_exact_score", second_base, 1.3333333333333334e16),
        )
    }
    results = (
        solve_mean_cvar_milp(bom, [0, 1], alpha=0.5, risk_weight=1),
        solve_benders_mean_cvar(bom, [0, 1], alpha=0.5, risk_weight=1),
        solve_spectral_cvar_mixture(
            bom,
            [0, 1],
            levels=[0.5],
            mixture_weights=[1],
            risk_weight=1,
        ),
        solve_wasserstein_cvar(bom, [0, 1], epsilon=0, alpha=0.5, risk_weight=1),
    )
    for result in results:
        assert result.selected[0].supplier == "lower_exact_score"
        assert result.objective == 2e16
        assert result.details["componentwise_optimality_certified"] is True


def test_public_numeric_apis_reject_unrepresentable_python_integers_cleanly():
    too_large = 10**10_000
    with pytest.raises(OptimizationError, match="event_counts"):
        solve_mean_cvar_milp(_small_bom(), [too_large])
    with pytest.raises(OptimizationError, match="not text"):
        solve_mean_cvar_milp(_small_bom(), "12")
    with pytest.raises(OptimizationError, match="bytes-like"):
        solve_mean_cvar_milp(_small_bom(), memoryview(b"12"))
    with pytest.raises(OptimizationError, match="iterable"):
        solve_mean_cvar_milp(_small_bom(), 5)
    with pytest.raises(OptimizationError, match="risk_weight"):
        solve_mean_cvar_milp(_small_bom(), [0, 1], risk_weight=np.asarray([0.5]))
    with pytest.raises(OptimizationError, match="one-dimensional"):
        solve_mean_cvar_milp(
            _small_bom(), pd.DataFrame([[99, 88]], columns=[0.0, 2.0])
        )
    with pytest.raises(OptimizationError, match="ordered"):
        solve_mean_cvar_milp(_small_bom(), {0.0, 2.0})
    with pytest.raises(OptimizationError, match="risk_weight"):
        solve_mean_cvar_milp(
            _small_bom(), [0, 1], risk_weight=pd.Series([0.5])
        )
    with pytest.raises(OptimizationError, match="complex"):
        solve_mean_cvar_milp(_small_bom(), [np.complex128(1 + 2j)])
    with pytest.raises(OptimizationError, match="risk_weight"):
        solve_mean_cvar_milp(
            _small_bom(), [0, 1], risk_weight=np.complex128(1 + 2j)
        )
    with pytest.raises(OptimizationError, match="risk_weight"):
        solve_mean_cvar_milp(_small_bom(), [0, 1], risk_weight=np.array(True))
    with pytest.raises(OptimizationError, match="numeric scalar"):
        solve_mean_cvar_milp(_small_bom(), [np.array("1")])


def test_milp_variants_reject_boolean_event_counts():
    with pytest.raises(OptimizationError, match="not boolean"):
        solve_mean_cvar_milp(_small_bom(), [False, True])
    with pytest.raises(OptimizationError, match="not boolean"):
        solve_benders_mean_cvar(_small_bom(), [False, True])
    with pytest.raises(OptimizationError, match="not boolean"):
        solve_spectral_cvar_mixture(_small_bom(), [False, True], levels=[0.8], mixture_weights=[1.0])
    with pytest.raises(OptimizationError, match="not boolean"):
        solve_wasserstein_cvar(_small_bom(), [False, True], epsilon=0.1)


def test_wasserstein_radius_uses_a_scaled_standard_deviation_and_zero_shortcut():
    maximum = np.finfo(float).max
    radius = default_wasserstein_radius([0.0, maximum])
    assert math.isfinite(radius)
    assert radius == pytest.approx((maximum / 2) / math.sqrt(2))
    assert default_wasserstein_radius([0.0, maximum], multiplier=0) == 0.0


def test_solver_scaling_does_not_reject_zero_or_dominated_extreme_coefficients():
    maximum = np.finfo(float).max
    zero_bom = {"A": (_option("A", "first", 0, 0), _option("A", "second", 0, 0))}
    dominated_bom = {
        "A": (
            _option("A", "safe", 0, 0),
            _option("A", "dominated", 1e50, 0),
        )
    }
    calls = (
        lambda bom, weight: solve_mean_cvar_milp(bom, [0, 1], alpha=0.5, risk_weight=weight),
        lambda bom, weight: solve_benders_mean_cvar(bom, [0, 1], alpha=0.5, risk_weight=weight),
        lambda bom, weight: solve_spectral_cvar_mixture(
            bom,
            [0, 1],
            levels=[0.5],
            mixture_weights=[1],
            risk_weight=weight,
        ),
        lambda bom, weight: solve_wasserstein_cvar(
            bom, [0, 1], epsilon=0.1, alpha=0.5, risk_weight=weight
        ),
    )
    for call in calls:
        zero = call(zero_bom, maximum)
        dominated = call(dominated_bom, 1.0)
        assert zero.objective == 0.0
        assert dominated.selected[0].supplier == "safe"
        assert dominated.objective == 0.0


def test_exact_certificate_can_replace_an_unreportable_raw_solver_policy():
    bom = {
        "A": (
            _option("A", "huge", 1e308, 0),
            _option("A", "middle", 1e100, 0),
            _option("A", "finite_optimum", 1e-320, 0),
        )
    }
    spectral = solve_spectral_cvar_mixture(
        bom,
        [0, 0, 0],
        levels=[0.5],
        mixture_weights=[1],
        risk_weight=1e300,
        scenario_weights=[0.2, 0.3, 0.5],
    )
    robust = solve_wasserstein_cvar(
        bom,
        [0, 0, 0],
        alpha=0.5,
        epsilon=1,
        risk_weight=1e300,
        scenario_weights=[0.2, 0.3, 0.5],
    )
    for result in (spectral, robust):
        assert result.selected[0].supplier == "finite_optimum"
        assert math.isfinite(result.objective)


def test_benders_returns_the_nominal_reference_when_an_intermediate_policy_overflows():
    bom = {
        "A": (
            _option("A", "safe", 1, 0),
            _option("A", "risky", 0, 1),
        )
    }
    result = solve_benders_mean_cvar(
        bom,
        [0, 2],
        alpha=0.5,
        risk_weight=1e308,
        scenario_weights=[0.55, 0.45],
    )
    assert result.selected[0].supplier == "safe"
    assert result.objective == 1e308
    assert result.details["fallback_to_nominal"] is True


def test_zero_weight_w1_tie_prefers_a_finite_robust_risk_report():
    bom = {
        "A": (
            _option("A", "unreportable_tie", 0, 1e308),
            _option("A", "safe_tie", 0, 0),
        )
    }
    result = solve_wasserstein_cvar(
        bom, [0], alpha=0.5, epsilon=1, risk_weight=0
    )
    assert result.selected[0].supplier == "safe_tie"
    assert result.objective == result.risk_cost == 0.0
    assert result.details["bypassed_for_zero_risk_weight"] is True


def test_exact_certificate_fallback_is_supported_when_the_milp_solver_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        optimize_module,
        "_solve_or_raise",
        lambda _problem: (_ for _ in ()).throw(OptimizationError("solver unavailable")),
    )
    results = (
        solve_mean_cvar_milp(_small_bom(), [0, 1], alpha=0.5),
        solve_benders_mean_cvar(_small_bom(), [0, 1], alpha=0.5),
        solve_spectral_cvar_mixture(
            _small_bom(),
            [0, 1],
            levels=[0.5],
            mixture_weights=[1],
        ),
        solve_wasserstein_cvar(_small_bom(), [0, 1], alpha=0.5, epsilon=0.1),
    )
    assert all(result.status == "ExactOptimal" for result in results)
    assert all(result.details["componentwise_optimality_certified"] for result in results)


def test_wasserstein_radius_avoids_multiply_before_divide_overflow():
    maximum = np.finfo(float).max
    events = [0.0] * 50 + [maximum] * 50
    assert default_wasserstein_radius(events, multiplier=10) == maximum / 2


def test_optimizer_aggregates_exposure_before_subnormal_multiplication():
    bom = {
        str(index): (
            _option(str(index), "only", 0.0, 1e-201),
        )
        for index in range(10)
    }
    result = solve_mean_cvar_milp(bom, [0.0, 1e-123], alpha=0.5)
    assert result.losses.tolist() == [0.0, 1e-323]
    assert result.mean_cost > 0
    assert result.risk_cost == 1e-323
    assert result.objective > 0


def test_minimum_subnormal_common_floors_preserve_full_solver_diagnostics():
    minimum = np.nextafter(0.0, 1.0)
    lower_bound, upper_bound, gap, relative_gap = (
        optimize_module._reported_benders_bounds(
            minimum,
            minimum,
            minimum + minimum + minimum,
        )
    )
    assert lower_bound == minimum + minimum
    assert upper_bound == minimum + minimum + minimum
    assert gap == upper_bound - lower_bound == minimum
    assert relative_gap == gap / max(1.0, abs(upper_bound))
    bom = {
        component: (_option(component, f"{component}_only", 0.0, minimum),)
        for component in ("A", "B")
    }
    nominal = solve_mean_cvar_milp(bom, [0.5], alpha=0.5)
    benders = solve_benders_mean_cvar(bom, [0.5], alpha=0.5)
    spectral = solve_spectral_cvar_mixture(
        bom,
        [0.5],
        levels=[0.5],
        mixture_weights=[1.0],
    )
    robust = solve_wasserstein_cvar(
        bom,
        [0.5],
        alpha=0.5,
        epsilon=0.0,
    )
    expected_objective = minimum + minimum
    for result in (nominal, benders, spectral, robust):
        assert result.losses.tolist() == [minimum]
        assert result.mean_cost == minimum
        assert result.risk_cost == minimum
        assert result.objective == expected_objective
    for result in (nominal, spectral):
        assert result.details["baseline_objective"] == expected_objective
        assert result.details["conditioned_objective"] == 0.0
        assert result.details["solver_conditioned_objective"] == 0.0
    assert benders.details["lower_bound"] == expected_objective
    assert benders.details["upper_bound"] == expected_objective
    assert benders.details["gap"] == (
        benders.details["upper_bound"] - benders.details["lower_bound"]
    )
    assert benders.details["relative_gap"] == 0.0


def test_unrepresentable_conditioned_residual_uses_exact_certificates():
    minimum = np.nextafter(0.0, 1.0)
    bom = {
        "A": (
            _option("A", "safe", 0.0, 0.0),
            _option("A", "unrepresentable_residual", 0.0, minimum),
        )
    }
    nominal = solve_mean_cvar_milp(bom, [0.5], alpha=0.5)
    benders = solve_benders_mean_cvar(bom, [0.5], alpha=0.5)
    spectral = solve_spectral_cvar_mixture(
        bom,
        [0.5],
        levels=[0.5],
        mixture_weights=[1.0],
    )
    robust = solve_wasserstein_cvar(
        bom,
        [0.5],
        alpha=0.5,
        epsilon=1.0,
    )
    for result in (nominal, benders, spectral, robust):
        assert result.status == "ExactOptimal"
        assert result.selected[0].supplier == "safe"
        assert result.objective == 0.0
        assert result.details["componentwise_optimality_certified"] is True
    assert nominal.details["solver_fallback_reason"] == (
        "conditioned_solver_coefficients_not_representable"
    )
    assert benders.details["fallback_reason"] == (
        "conditioned_solver_coefficients_not_representable"
    )
    assert spectral.details["solver_fallback_reason"] == (
        "conditioned_solver_coefficients_not_representable"
    )
    assert robust.details["solver_fallback_reason"] == (
        "conditioned_solver_coefficients_not_representable"
    )


def test_w1_rejects_a_nonzero_robust_risk_that_cannot_be_reported_as_float():
    minimum = np.nextafter(0.0, 1.0)
    bom = {"A": (_option("A", "only", 0.0, minimum),)}
    with pytest.raises(OptimizationError, match="W1 risk scale"):
        solve_wasserstein_cvar(
            bom, [0], alpha=0.5, epsilon=1e-300, risk_weight=0
        )
