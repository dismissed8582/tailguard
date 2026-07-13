import math

import numpy as np
import pytest

import tailguard.validation as validation_module
from tailguard.bom import SupplierOption
from tailguard.validation import evaluate_policy


def _option(component, supplier, base_cost=1.0):
    return SupplierOption(component, supplier, base_cost, 1.0, 1.0, "offshore")


@pytest.mark.parametrize(
    "name,value",
    [("risk_weight", math.nan), ("risk_weight", math.inf), ("lead_time_cost_per_day", math.nan)],
)
def test_policy_evaluation_rejects_nonfinite_tuning_values(name, value):
    kwargs = {name: value}
    with pytest.raises(ValueError, match="finite non-negative"):
        evaluate_policy([_option("A", "one")], [0, 1], **kwargs)


def test_policy_evaluation_requires_one_supplier_option_per_component():
    with pytest.raises(ValueError, match="multiple options"):
        evaluate_policy([_option("A", "one"), _option("A", "two", 100)], [0, 1])
    with pytest.raises(TypeError, match="SupplierOption"):
        evaluate_policy([object()], [0, 1])


def test_policy_evaluation_rejects_boolean_events_and_reports_overflow_at_its_boundary():
    with pytest.raises(ValueError, match="not boolean"):
        evaluate_policy([_option("A", "one")], [False, True])
    with pytest.raises(ValueError, match="non-finite scenario losses"):
        evaluate_policy([SupplierOption("A", "huge", 1e308, 1e308, 0.0, "offshore")], [2])
    with pytest.raises(ValueError, match="objective overflows"):
        evaluate_policy([SupplierOption("A", "huge", 1e308, 0.0, 0.0, "offshore")], [0], risk_weight=1e308)


def test_policy_evaluation_rejects_unrepresentable_python_integers_cleanly():
    too_large = 10**10_000
    with pytest.raises(ValueError, match="event_counts"):
        evaluate_policy([_option("A", "one")], [too_large])
    with pytest.raises(ValueError, match="alpha"):
        evaluate_policy([_option("A", "one")], [0], alpha=too_large)
    with pytest.raises(ValueError, match="risk_weight"):
        evaluate_policy([_option("A", "one")], [0], risk_weight=too_large)
    with pytest.raises(ValueError, match="not text"):
        evaluate_policy([_option("A", "one")], "12")
    with pytest.raises(ValueError, match="not text"):
        evaluate_policy([_option("A", "one")], b"12")
    with pytest.raises(ValueError, match="bytes-like"):
        evaluate_policy([_option("A", "one")], memoryview(b"12"))
    with pytest.raises(ValueError, match="iterable"):
        evaluate_policy([_option("A", "one")], 5)
    with pytest.raises(ValueError, match="risk_weight"):
        evaluate_policy([_option("A", "one")], [0], risk_weight=np.asarray([1.0]))
    with pytest.raises(ValueError, match="complex"):
        evaluate_policy([_option("A", "one")], [np.complex128(1 + 2j)])
    with pytest.raises(ValueError, match="risk_weight"):
        evaluate_policy(
            [_option("A", "one")], [0], risk_weight=np.complex128(1 + 2j)
        )
    with pytest.raises(ValueError, match="numeric scalar"):
        evaluate_policy([_option("A", "one")], [np.array(True)])


def test_policy_evaluation_ignores_zero_probability_extremes_without_losing_small_costs():
    result = evaluate_policy(
        [SupplierOption("A", "only", 0.0, 1.0, 0.0, "offshore")],
        [1e308, 0.0, 1e-20],
        alpha=0.5,
        risk_weight=0.5,
        scenario_weights=[0.0, 0.0, 1.0],
    )
    assert result.mean_cost == 1e-20
    assert result.cvar == 1e-20
    assert result.objective == pytest.approx(1.5e-20, rel=1e-12, abs=0)


def test_policy_evaluation_aggregates_exposure_before_subnormal_multiplication():
    selected = [
        SupplierOption(str(index), "only", 0.0, 1e-201, 0.0, "offshore")
        for index in range(10)
    ]
    result = evaluate_policy(selected, [0.0, 1e-123], alpha=0.5)
    assert result.losses.tolist() == [0.0, 1e-323]
    assert result.mean_cost > 0
    assert result.cvar == 1e-323
    assert result.objective > 0


def test_policy_evaluation_rejects_an_exact_nonzero_objective_that_underflows(
    monkeypatch,
):
    minimum = np.nextafter(0.0, 1.0)
    monkeypatch.setattr(validation_module, "weighted_mean", lambda *_args: 0.0)
    monkeypatch.setattr(
        validation_module,
        "weighted_cvar",
        lambda *_args: minimum,
    )
    with pytest.raises(ValueError, match="objective overflows or underflows"):
        evaluate_policy(
            [SupplierOption("A", "only", 0.0, 0.0, 0.0, "offshore")],
            [0.0],
            alpha=0.5,
            risk_weight=minimum,
        )
