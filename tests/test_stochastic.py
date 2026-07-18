import math
import warnings
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

import tailguard.stochastic as stochastic_module
from tailguard.data import DataValidationError, load_series_csv, load_series_csv_bytes, synthetic_series
from tailguard.stochastic import (
    CIRParameters,
    ScenarioValidationError,
    fit_cir_proxy,
    simulate_cox,
    simulate_stratified_cox,
)


def test_synthetic_series_is_deterministic_and_explicitly_synthetic():
    first = synthetic_series("FREIGHT_PROXY", start="2019-01-01", end="2020-12-01")
    second = synthetic_series("FREIGHT_PROXY", start="2019-01-01", end="2020-12-01")
    assert first.equals(second)
    assert len(first) == 24
    with pytest.raises(DataValidationError, match="non-empty"):
        synthetic_series("   ")
    with pytest.raises(DataValidationError, match="UTF-8"):
        synthetic_series("\ud800")


def test_proxy_fit_and_cox_simulation_are_seed_reproducible():
    series = synthetic_series(start="2017-01-01", end="2019-12-01")
    fit = fit_cir_proxy(series)
    first_counts, first_integrals = simulate_cox(fit.parameters, scenarios=100, seed=99, steps=30)
    second_counts, second_integrals = simulate_cox(fit.parameters, scenarios=100, seed=99, steps=30)
    assert np.array_equal(first_counts, second_counts)
    assert np.allclose(first_integrals, second_integrals)
    assert np.all(first_counts >= 0)
    assert np.all(first_integrals >= 0)

    numpy_counts, _ = simulate_cox(
        fit.parameters,
        scenarios=np.int64(4),
        steps=np.int64(2),
        seed=np.int64(1),
    )
    assert len(numpy_counts) == 4


def test_cox_simulation_reports_unrepresentable_parameter_scale():
    with pytest.raises(ScenarioValidationError, match="rescale"):
        simulate_cox(CIRParameters(1e308, 1e308, 0.0, 0.0), scenarios=1, steps=1)


def test_cox_simulation_rejects_a_positive_trapezoid_increment_that_underflows():
    minimum = np.nextafter(0.0, 1.0)
    parameters = CIRParameters(1.0, minimum, 0.0, minimum)
    _, one_step_integral = simulate_cox(parameters, scenarios=1, steps=1, seed=1)
    assert one_step_integral.tolist() == [minimum]
    with pytest.raises(ScenarioValidationError, match="integration increment underflows"):
        simulate_cox(parameters, scenarios=1, steps=2, seed=1)


def test_cox_simulation_rejects_hidden_state_underflow_and_preprojection_overflow():
    with pytest.raises(ScenarioValidationError, match="drift increment"):
        simulate_cox(
            CIRParameters(1e-200, 1e-121, 0.0, 0.0),
            horizon_years=1.0,
            scenarios=1,
            steps=1_000,
            seed=1,
        )
    _, finite_boundary_integral = simulate_cox(
        CIRParameters(1.0, 4.0, 1e308, 4.0),
        horizon_years=1.0,
        scenarios=1,
        steps=1,
        seed=4,
    )
    assert finite_boundary_integral.tolist() == [2.0]
    with pytest.raises(ScenarioValidationError, match="diffusion increment"):
        simulate_cox(
            CIRParameters(1.0, 4.0, np.finfo(float).max, 4.0),
            horizon_years=1.0,
            scenarios=1,
            steps=1,
            seed=4,
        )


def test_cox_simulation_forms_a_representable_mean_reversion_step_before_drift():
    descending = CIRParameters(1e308, 1.0, 0.0, 3.0)
    ascending = CIRParameters(1e308, 3.0, 0.0, 1.0)
    _, descending_integral = simulate_cox(
        descending, horizon_years=1e-308, scenarios=1, steps=1, seed=1
    )
    _, ascending_integral = simulate_cox(
        ascending, horizon_years=1e-308, scenarios=1, steps=1, seed=1
    )
    assert descending_integral[0] == pytest.approx(2e-308, rel=1e-12, abs=0.0)
    assert ascending_integral[0] == pytest.approx(2e-308, rel=1e-12, abs=0.0)


def test_cox_integration_preserves_collectively_representable_increments():
    _, integrated = simulate_cox(
        CIRParameters(1_000.0, 1e-14, 0.0, 2_000.0),
        horizon_years=1.0,
        scenarios=1,
        steps=1_000,
        seed=1,
    )
    assert integrated[0] == 1.00000000000001
    assert integrated[0] > 1.0


def test_scaled_cir_products_preserve_the_minimum_subnormal_boundary():
    left = math.ldexp((2**52 + 1) / 2**52, -500)
    right = math.ldexp((2**53 - 1) / 2**52, -576)
    minimum = math.ulp(0.0)
    assert left * right == minimum
    assert stochastic_module._scaled_nonnegative_product(
        left, np.asarray([right]), "test product"
    ).tolist() == [minimum]
    assert stochastic_module._scaled_diffusion_term(
        left,
        np.asarray([right]),
        1.0,
        np.asarray([1.0]),
    ).tolist() == [minimum]


def test_scaled_diffusion_classifies_exact_range_boundaries_after_mantissa_rounding():
    denominator = 2**53

    underflow_factors = [
        8640531598592716,
        6674571894452240,
        7341360969184081,
        7772999297079854,
    ]
    underflow_inputs = (
        math.ldexp(underflow_factors[0] / denominator, -500),
        math.ldexp(underflow_factors[1] / denominator, -574),
        underflow_factors[2] / denominator,
        underflow_factors[3] / denominator,
    )
    assert float(math.prod(Fraction(value) for value in underflow_inputs)) == 0.0
    with pytest.raises(ScenarioValidationError, match="diffusion increment"):
        stochastic_module._scaled_diffusion_term(
            underflow_inputs[0],
            np.asarray([underflow_inputs[2]]),
            underflow_inputs[1],
            np.asarray([underflow_inputs[3]]),
        )

    overflow_factors = [
        8318692298130026,
        8160465420823440,
        8031587183278780,
        6036117953564618,
    ]
    overflow_inputs = (
        math.ldexp(overflow_factors[0] / denominator, 500),
        math.ldexp(overflow_factors[1] / denominator, 500),
        math.ldexp(overflow_factors[2] / denominator, 25),
        overflow_factors[3] / denominator,
    )
    with pytest.raises(OverflowError):
        float(math.prod(Fraction(value) for value in overflow_inputs))
    with pytest.raises(ScenarioValidationError, match="diffusion increment"):
        stochastic_module._scaled_diffusion_term(
            overflow_inputs[0],
            np.asarray([overflow_inputs[2]]),
            overflow_inputs[1],
            np.asarray([overflow_inputs[3]]),
        )


def test_mean_reversion_grid_uses_the_exact_user_input_ratio():
    with pytest.raises(ScenarioValidationError, match="non-overshooting"):
        simulate_cox(
            CIRParameters(np.nextafter(1.0, np.inf), 1.0, 0.0, 2.0),
            horizon_years=np.nextafter(1.0, 0.0),
            scenarios=1,
            steps=1,
            seed=1,
        )
    _, exact_unit_step = simulate_cox(
        CIRParameters(1_000.0, 1e-14, 0.0, 2_000.0),
        horizon_years=1.0,
        scenarios=1,
        steps=1_000,
        seed=1,
    )
    assert exact_unit_step[0] == 1.00000000000001


def test_state_rounding_does_not_create_an_eta_dependent_rejection_boundary():
    integrals = []
    for volatility in (0.0, 1e-200):
        _, integrated = simulate_cox(
            CIRParameters(1e-13, 2.0, volatility, 1.0),
            horizon_years=1.0,
            scenarios=1,
            steps=1_000,
            seed=1,
        )
        integrals.append(integrated[0])
    assert integrals == [1.0, 1.0]


def test_tiny_random_draw_rounding_does_not_make_simulation_sample_size_dependent():
    parameters = CIRParameters(1.0, 1.0, 1e-10, 1.0)
    small_counts, small_integrals = simulate_cox(
        parameters, scenarios=2, steps=252, seed=0
    )
    large_counts, large_integrals = simulate_cox(
        parameters, scenarios=500, steps=252, seed=42
    )
    assert len(small_counts) == len(small_integrals) == 2
    assert len(large_counts) == len(large_integrals) == 500
    assert np.isfinite(small_integrals).all()
    assert np.isfinite(large_integrals).all()


def test_cox_counts_remain_int64_above_windows_native_int_range():
    counts, _ = simulate_cox(
        CIRParameters(1.0, 3e9, 0.0, 3e9),
        scenarios=4,
        steps=1,
        seed=7,
    )
    assert counts.dtype == np.int64
    assert np.all(counts > np.iinfo(np.int32).max)


def test_cir_diagnostic_and_seed_validation_reject_invalid_public_inputs():
    assert CIRParameters(float("inf"), 1.0, 1.0, 0.0).feller_satisfied is False
    assert CIRParameters(-1.0, 1.0, 0.0, 0.0).feller_satisfied is False
    with pytest.raises(ScenarioValidationError, match="seed"):
        simulate_cox(CIRParameters(1.0, 1.0, 0.0, 1.0), scenarios=1, steps=1, seed="invalid")
    with pytest.raises(ScenarioValidationError, match="time step"):
        simulate_cox(CIRParameters(1.0, 1.0, 0.0, 1.0), scenarios=1, steps=10**400)
    with pytest.raises(ScenarioValidationError, match="CIRParameters"):
        simulate_cox(None)
    with pytest.raises(ScenarioValidationError, match="horizon_years"):
        simulate_cox(CIRParameters(1.0, 1.0, 0.0, 1.0), horizon_years=np.asarray([1.0]))
    with pytest.raises(ScenarioValidationError, match="horizon_years"):
        simulate_cox(
            CIRParameters(1.0, 1.0, 0.0, 1.0),
            horizon_years=np.complex128(1 + 2j),
        )
    with pytest.raises(ScenarioValidationError, match="horizon_years"):
        simulate_cox(
            CIRParameters(1.0, 1.0, 0.0, 1.0),
            horizon_years=np.array(True),
        )


@pytest.mark.parametrize(
    "parameters,expected",
    [
        (
            CIRParameters(3.9920068259118517, 0.3561411167112906, 1.6862489548695598, 0.0),
            True,
        ),
        (
            CIRParameters(3.211851286690545, 18.523416116216982, 10.908204058118452, 0.0),
            False,
        ),
    ],
)
def test_feller_diagnostic_is_exact_at_binary64_rounding_boundaries(parameters, expected):
    assert parameters.feller_satisfied is expected


@pytest.mark.parametrize(
    "invalid_value",
    [
        "1",
        b"1",
        np.complex128(1 + 2j),
        np.array(1.0),
        pd.Series([1.0]),
    ],
)
def test_feller_diagnostic_is_total_for_malformed_scalar_values(invalid_value):
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        satisfied = CIRParameters(invalid_value, 1.0, 1.0, 0.0).feller_satisfied
    assert not captured
    assert satisfied is False


def test_feller_diagnostic_suppresses_arbitrary_float_conversion_failures():
    class BrokenScalar:
        def __float__(self):
            raise RuntimeError("custom scalar conversion failed")

    assert CIRParameters(BrokenScalar(), 1.0, 1.0, 0.0).feller_satisfied is False


def test_stratified_resampling_returns_normalized_weights_without_hanging_on_degenerate_cir():
    params = CIRParameters(1.0, 0.5, 0.0, 0.5)
    counts, integrals, weights = simulate_stratified_cox(params, scenarios=20, strata=4, pilot_size=40, seed=3, steps=5)
    assert len(counts) == len(integrals) == len(weights) == 20
    assert weights.sum() == pytest.approx(1.0)
    assert np.all(weights > 0)


def test_invalid_stratified_budget_is_rejected():
    params = CIRParameters(1.0, 0.5, 0.2, 0.5)
    with pytest.raises(ScenarioValidationError):
        simulate_stratified_cox(params, scenarios=3, strata=4)


def test_proxy_fit_uses_each_irregular_date_gap_and_requires_explicit_date_opt_out():
    irregular = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02", "2020-04-01", "2021-01-01", "2021-01-15"],
            "value": [0.0, 1.0, -0.5, 0.75, 0.2],
        }
    )
    fit = fit_cir_proxy(irregular)
    assert fit.time_steps_years == pytest.approx([1 / 365.25, 90 / 365.25, 275 / 365.25, 14 / 365.25])
    assert fit.time_step_years == pytest.approx(np.median(fit.time_steps_years))

    without_dates = irregular[["value"]]
    with pytest.raises(ScenarioValidationError, match="missing"):
        fit_cir_proxy(without_dates)
    assumed_monthly = fit_cir_proxy(without_dates, date_column=None)
    assert assumed_monthly.time_steps_years == pytest.approx([1 / 12] * 4)

    centuries = pd.DataFrame(
        {"date": ["1700", "1800", "1900", "2200"], "value": [0.0, 1.0, 0.5, 2.0]}
    )
    long_span_fit = fit_cir_proxy(centuries)
    assert long_span_fit.time_steps_years == pytest.approx([100.0, 100.0, 300.0], rel=0.01)

    nanosecond_dates = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2020-01-01 00:00:00.000000000",
                    "2020-01-01 00:00:00.000000100",
                    "2020-01-01 00:00:00.000000300",
                    "2020-01-01 00:00:00.000000600",
                ],
                utc=True,
            ),
            "value": [0.0, 1.0, 0.5, 2.0],
        }
    )
    nanosecond_fit = fit_cir_proxy(nanosecond_dates)
    expected_nanosecond_years = np.asarray([100.0, 200.0, 300.0]) / (1e9 * 60 * 60 * 24 * 365.25)
    assert nanosecond_fit.time_steps_years == pytest.approx(expected_nanosecond_years)

    wide_microsecond_dates = pd.DataFrame(
        {
            "date": pd.Series(
                np.asarray(["2500", "2600", "2700", "2800"], dtype="datetime64[us]")
            ),
            "value": [0.0, 1.0, 0.5, 2.0],
        }
    )
    wide_date_fit = fit_cir_proxy(wide_microsecond_dates)
    assert wide_date_fit.time_steps_years == pytest.approx([100.0, 100.0, 100.0], rel=0.01)


def test_proxy_fit_rejects_unidentified_or_underdetermined_series_and_reports_all_regularization():
    with pytest.raises(ScenarioValidationError, match="DataFrame"):
        fit_cir_proxy([0.0, 1.0, 0.0, 1.0])

    boolean_values = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=4, freq="MS"),
            "value": [False, True, False, True],
        }
    )
    with pytest.raises(ScenarioValidationError, match="not boolean"):
        fit_cir_proxy(boolean_values)

    hidden_value = boolean_values.copy()
    hidden_value["value"] = pd.Series(
        [np.array(True), 1.0, 0.5, 2.0], dtype=object
    )
    with pytest.raises(ScenarioValidationError, match="numeric scalar"):
        fit_cir_proxy(hidden_value)

    duplicate_base = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=4, freq="MS"),
            "value": [0.0, 1.0, 0.0, 1.0],
        }
    )
    duplicate_value_columns = pd.concat(
        [duplicate_base["date"], duplicate_base["value"], duplicate_base["value"]],
        axis=1,
    )
    with pytest.raises(ScenarioValidationError, match="duplicate value"):
        fit_cir_proxy(duplicate_value_columns)

    duplicate_date_columns = pd.concat(
        [duplicate_base["date"], duplicate_base["date"], duplicate_base["value"]],
        axis=1,
    )
    with pytest.raises(ScenarioValidationError, match="duplicate date"):
        fit_cir_proxy(duplicate_date_columns)
    with pytest.raises(ScenarioValidationError, match="distinct"):
        fit_cir_proxy(duplicate_base, date_column="value")

    unusual_columns = pd.DataFrame(
        [[0, 0.0], [1, 1.0], [2, 0.5], [3, 2.0]],
        columns=pd.Index([{}, "value"], dtype=object),
    )
    fit_cir_proxy(unusual_columns, date_column=None)

    complex_values = duplicate_base.copy()
    complex_values["value"] = complex_values["value"].astype(object)
    complex_values.loc[0, "value"] = 1 + 1j
    with pytest.raises(ScenarioValidationError, match="not complex"):
        fit_cir_proxy(complex_values)

    nonfinite_date = duplicate_base.copy()
    nonfinite_date["date"] = nonfinite_date["date"].astype(object)
    nonfinite_date.loc[0, "date"] = np.inf
    with pytest.raises(ScenarioValidationError, match="dates must be valid"):
        fit_cir_proxy(nonfinite_date)

    hidden_date = duplicate_base.copy()
    hidden_date["date"] = hidden_date["date"].astype(object)
    hidden_date.at[0, "date"] = np.array(np.datetime64("2020-01-01"))
    with pytest.raises(ScenarioValidationError, match="date-like"):
        fit_cir_proxy(hidden_date)

    numeric_dates = pd.DataFrame(
        {
            "date": [20200101, 20200201, 20200301, 20200401],
            "value": [0.0, 1.0, 0.5, 2.0],
        }
    )
    with pytest.raises(ScenarioValidationError, match="not numbers"):
        fit_cir_proxy(numeric_dates)

    for invalid_values in (
        pd.date_range("2020-01-01", periods=4, freq="MS"),
        pd.to_timedelta([1, 2, 3, 4], unit="D"),
    ):
        swapped_or_duration = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=4, freq="MS"),
                "value": invalid_values,
            }
        )
        with pytest.raises(ScenarioValidationError, match="not dates or durations"):
            fit_cir_proxy(swapped_or_duration)

    three_observations = pd.DataFrame(
        {"date": pd.date_range("2020-01-01", periods=3, freq="MS"), "value": [0.0, 1.0, 0.0]}
    )
    with pytest.raises(ScenarioValidationError, match="at least four"):
        fit_cir_proxy(three_observations)

    constant = pd.DataFrame(
        {"date": pd.date_range("2020-01-01", periods=4, freq="MS"), "value": [1.0, 1.0, 1.0, 1.0]}
    )
    with pytest.raises(ScenarioValidationError, match="does not identify"):
        fit_cir_proxy(constant)

    decreasing = pd.DataFrame(
        {"date": pd.date_range("2020-01-01", periods=4, freq="MS"), "value": [10.0, 5.0, 2.0, 1.0]}
    )
    decreasing_fit = fit_cir_proxy(decreasing)
    assert decreasing_fit.used_regularization is True
    assert decreasing_fit.mean_reversion_regularized or decreasing_fit.long_run_mean_regularized

    extreme = pd.DataFrame(
        {"date": pd.date_range("2020-01-01", periods=4, freq="MS"), "value": [-1e308, 1e308, 0.0, 1.0]}
    )
    with pytest.raises(ScenarioValidationError, match="rescale"):
        fit_cir_proxy(extreme)

    representable_extreme = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=4, freq="MS"),
            "value": [0.0, np.finfo(float).max, 0.0, np.finfo(float).max],
        }
    )
    representable_fit = fit_cir_proxy(representable_extreme)
    assert np.isfinite(representable_fit.transformed_intensity).all()
    assert np.isfinite(representable_fit.parameters.volatility)

    unrepresentable = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=4, freq="MS")})
    unrepresentable["value"] = pd.Series([10**10_000, 1, 2, 3], dtype=object)
    with pytest.raises(ScenarioValidationError, match="finite numbers"):
        fit_cir_proxy(unrepresentable)

    alternating = pd.DataFrame(
        {"date": pd.date_range("2020-01-01", periods=4, freq="MS"), "value": [0.0, 1.0, 0.0, 1.0]}
    )
    with pytest.raises(ScenarioValidationError, match="rescale"):
        fit_cir_proxy(alternating, minimum_mean_reversion=1e155)

    huge_scale = pd.DataFrame(
        {"date": ["1700", "1800", "2000", "2200"], "value": [0.0, 1.0, 0.5, 2.0]}
    )
    with pytest.raises(ScenarioValidationError, match="rescale"):
        fit_cir_proxy(huge_scale, target_mean_events=1e307)


def test_local_series_rejects_invalid_or_duplicate_observations(tmp_path):
    invalid_path = tmp_path / "invalid.csv"
    pd.DataFrame(
        {
            "date": ["2020-01-01", "not-a-date", "2020-03-01"],
            "value": [1.0, 2.0, "bad"],
        }
    ).to_csv(invalid_path, index=False)
    with pytest.raises(DataValidationError, match="invalid"):
        load_series_csv(invalid_path)

    duplicate_path = tmp_path / "duplicate.csv"
    pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-01", "2020-03-01"],
            "value": [1.0, 2.0, 3.0],
        }
    ).to_csv(duplicate_path, index=False)
    with pytest.raises(DataValidationError, match="duplicate"):
        load_series_csv(duplicate_path)

    infinity_path = tmp_path / "infinity.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=4, freq="MS"),
            "value": [1.0, float("inf"), 3.0, 4.0],
        }
    ).to_csv(infinity_path, index=False)
    with pytest.raises(DataValidationError, match="invalid"):
        load_series_csv(infinity_path)

    duplicate_header_path = tmp_path / "duplicate-header.csv"
    duplicate_header_path.write_text(
        "date,value,value\n2020-01-01,1,99\n2020-02-01,2,99\n2020-03-01,3,99\n2020-04-01,4,99\n",
        encoding="utf-8",
    )
    with pytest.raises(DataValidationError, match="duplicate column"):
        load_series_csv(duplicate_header_path)


def test_local_series_normalizes_utc_bounds_to_its_naive_utc_dates(tmp_path):
    path = tmp_path / "valid.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=5, freq="MS"),
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    ).to_csv(path, index=False)
    result = load_series_csv(path, start="2020-01-01T00:00:00Z")
    assert len(result) == 5


def test_in_memory_series_upload_normalizes_friendly_headers_without_a_file():
    content = (
        " Date , VALUE \n"
        "2020-01-01,1\n"
        "2020-02-01,2\n"
        "2020-03-01,3\n"
        "2020-04-01,4\n"
    ).encode("utf-8")
    result = load_series_csv_bytes(bytearray(content))
    assert list(result.columns) == ["date", "value"]
    assert result["value"].tolist() == [1, 2, 3, 4]


def test_series_inputs_accept_mixed_unambiguous_iso8601_date_forms():
    content = (
        "date,value\n"
        "2020-01-01,1\n"
        "2020-02-01T12:00:00Z,2\n"
        "20200301,3\n"
        "2020-04-01T00:00:00+00:00,4\n"
    ).encode("utf-8")
    result = load_series_csv_bytes(content)
    assert result["date"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist() == [
        "2020-01-01T00:00:00",
        "2020-02-01T12:00:00",
        "2020-03-01T00:00:00",
        "2020-04-01T00:00:00",
    ]
    fit = fit_cir_proxy(
        pd.DataFrame(
            {
                "date": [
                    "2020-01-01",
                    "2020-02-01T12:00:00Z",
                    "20200301",
                    "2020-04-01T00:00:00+00:00",
                ],
                "value": [1, 2, 3, 4],
            }
        )
    )
    assert len(fit.time_steps_years) == 3


@pytest.mark.parametrize("underflowed_value", ["1e-9999", "-1e-9999"])
def test_series_csv_rejects_nonzero_values_that_underflow_float(underflowed_value):
    content = (
        "date,value\n"
        f"2020-01-01,{underflowed_value}\n"
        "2020-02-01,1\n"
        "2020-03-01,2\n"
        "2020-04-01,3\n"
    ).encode("utf-8")
    with pytest.raises(DataValidationError, match="non-zero numeric value"):
        load_series_csv_bytes(content)


def test_series_csv_preserves_signed_and_exponentiated_zero_literals():
    content = (
        "date,value\n"
        "2020-01-01,-0\n"
        "2020-02-01,0e999\n"
        "2020-03-01,-0e999\n"
        "2020-04-01,1\n"
    ).encode("utf-8")
    result = load_series_csv_bytes(content)
    assert result["value"].tolist() == [0.0, 0.0, -0.0, 1.0]


def test_numeric_looking_csv_dates_are_parsed_as_dates_not_unix_nanoseconds():
    content = (
        "date,value\n"
        "20200101,1\n"
        "20200201,2\n"
        "20200301,3\n"
        "20200401,4\n"
    ).encode("utf-8")
    result = load_series_csv_bytes(content)
    assert result["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2020-01-01",
        "2020-02-01",
        "2020-03-01",
        "2020-04-01",
    ]


def test_in_memory_series_upload_rejects_invalid_encoding_and_normalized_duplicates():
    with pytest.raises(DataValidationError, match="UTF-8"):
        load_series_csv_bytes(b"\xff\xfe\x00")
    duplicate = b"date, value ,VALUE\n2020-01-01,1,2\n"
    with pytest.raises(DataValidationError, match="duplicate"):
        load_series_csv_bytes(duplicate)
    duplicate_after_whitespace = (
        b"   \n"
        b"date,date,value\n"
        b"2024-01-01,ignored,1\n"
        b"2024-02-01,ignored,2\n"
        b"2024-03-01,ignored,3\n"
        b"2024-04-01,ignored,4\n"
    )
    with pytest.raises(DataValidationError, match="duplicate"):
        load_series_csv_bytes(duplicate_after_whitespace)
    with pytest.raises(TypeError, match="bytes-like"):
        load_series_csv_bytes("date,value")
    released = memoryview(b"released")
    released.release()
    with pytest.raises(DataValidationError, match="could not read"):
        load_series_csv_bytes(released)
    with pytest.raises(DataValidationError, match="NUL"):
        load_series_csv_bytes(
            b"date,value\n2020-01-01,1\x0099\n2020-02-01,2\n2020-03-01,3\n2020-04-01,4\n"
        )
    with pytest.raises(DataValidationError, match="same number of fields"):
        load_series_csv_bytes(
            b"date,value\nEXTRA,2020-01-01,1\n2020-02-01,2\n2020-03-01,3\n2020-04-01,4\n"
        )
    ambiguous = b"date,value\n01-02-03,1\n02-03-04,1\n03-04-05,1\n04-05-06,1\n"
    with pytest.raises(DataValidationError, match="ambiguous or invalid dates"):
        load_series_csv_bytes(ambiguous)
    with pytest.raises(DataValidationError, match="valid date"):
        synthetic_series(start="Jan 1 2020 12:00 EST")
    with pytest.raises(DataValidationError, match="not a number"):
        synthetic_series(start=20200101)


def test_local_series_loader_translates_invalid_filesystem_paths():
    with pytest.raises(DataValidationError, match="could not read"):
        load_series_csv("bad\0path")
