import math
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from tailguard.risk import (
    RiskValidationError,
    normalized_weights,
    rockafellar_uryasev_value,
    tail_conditional_weights,
    weighted_cvar,
    weighted_mean,
)


def test_fractional_uniform_tail_uses_exact_tail_mass():
    losses = np.arange(10, dtype=float)
    assert weighted_cvar(losses, 0.85) == pytest.approx(8.666666666666666)
    value, zeta = rockafellar_uryasev_value(losses, 0.85)
    assert value == pytest.approx(weighted_cvar(losses, 0.85))
    assert zeta == 8.0


def test_weighted_cvar_allocates_only_required_var_mass():
    losses = [0.0, 100.0]
    weights = [0.96, 0.04]
    assert weighted_cvar(losses, 0.95, weights) == pytest.approx(80.0)
    conditional = tail_conditional_weights(losses, 0.95, weights)
    assert conditional.sum() == pytest.approx(1.0)
    assert conditional @ np.asarray(losses) == pytest.approx(80.0)


def test_zero_probability_extremes_do_not_erase_the_weighted_support():
    losses = [1e308, 0.0, 1e-20]
    weights = [0.0, 0.0, 1.0]
    assert weighted_mean(losses, weights) == 1e-20
    assert weighted_cvar(losses, 0.5, weights) == 1e-20
    value, _ = rockafellar_uryasev_value(losses, 0.5, weights)
    assert value == 1e-20


def test_supported_negative_extreme_does_not_erase_a_small_upper_tail():
    losses = [-1e308, 0.0, 1e-20]
    weights = [0.1, 0.4, 0.5]
    expected = weighted_cvar(losses, 0.5, weights)
    value, zeta = rockafellar_uryasev_value(losses, 0.5, weights)
    assert expected == 1e-20
    assert value == expected
    assert zeta == 0.0


def test_tail_conditional_weights_respect_cvar_dual_bounds_with_ties():
    losses = [0.0, 1.0, 1.0, 1.0]
    alpha = 0.5
    conditional = tail_conditional_weights(losses, alpha)
    assert conditional.sum() == pytest.approx(1.0)
    assert np.all(conditional <= 1.0 / (len(losses) * (1.0 - alpha)) + 1e-12)
    assert weighted_cvar(losses, alpha) == pytest.approx(1.0)


def test_extreme_valid_alphas_do_not_drop_tail_mass_or_raise_from_full_mass_rounding():
    near_one = np.nextafter(1.0, 0.0)
    conditional = tail_conditional_weights([0.0, 100.0], near_one)
    assert conditional.sum() == pytest.approx(1.0)
    assert weighted_cvar([0.0, 100.0], near_one) == pytest.approx(100.0)
    assert rockafellar_uryasev_value([0.0, 100.0], near_one)[0] == pytest.approx(100.0)

    near_zero = np.nextafter(0.0, 1.0)
    losses = [0.0, 1.0]
    weights = [0.2616121342493164, 0.2984911434141233]
    assert weighted_cvar(losses, near_zero, weights) == pytest.approx(weighted_mean(losses, weights))


def test_tiny_alpha_uses_exact_cvar_when_float_dual_weights_are_not_representable():
    maximum = np.finfo(float).max
    alpha = 1e-17
    assert weighted_cvar([-maximum, maximum], alpha) == pytest.approx(alpha * maximum, rel=1e-12)
    with pytest.raises(RiskValidationError, match="float dual weights"):
        tail_conditional_weights([-maximum, maximum], alpha)


def test_subnormal_probability_preserves_a_representable_tail_contribution():
    value = weighted_cvar([1e239, 0.0], 0.1, [1e-44, 1e278])
    assert value == 1.0979236574249923e-83
    with pytest.raises(RiskValidationError, match="subnormal atom"):
        tail_conditional_weights(
            [6.76035099e166, 0.0],
            0.7201570510208468,
            [9.30972837e-319, 1.0],
        )


def test_weight_normalization_avoids_finite_sum_overflow():
    weights = normalized_weights([1e308, 1e308], 2)
    assert weights == pytest.approx([0.5, 0.5])
    assert weighted_mean([1.0, 3.0], [1e308, 1e308]) == pytest.approx(2.0)


def test_weight_normalization_rejects_an_unrepresentable_positive_atom():
    with pytest.raises(RiskValidationError, match="deleting positive probability mass"):
        normalized_weights([1e-323, 1e308], 2)
    with pytest.raises(RiskValidationError, match="deleting positive probability mass"):
        weighted_mean([1e308, 0.0], [1e-323, 1e308])
    with pytest.raises(RiskValidationError, match="deleting positive probability mass"):
        normalized_weights([5e-324, 1.0, 1.0], 3)


def test_normalized_relative_probability_partition_is_permutation_safe_and_idempotent():
    probabilities = normalized_weights([0.12167778692709874, 0.47588257255962463, 0.5852880835480021], 3)
    assert np.array_equal(normalized_weights(probabilities, len(probabilities)), probabilities)
    uniform = normalized_weights(None, 49)
    assert np.unique(uniform).size == 1
    assert abs(math.fsum(float(value) for value in uniform) - 1.0) <= math.ulp(1.0)
    assert np.array_equal(normalized_weights(uniform, len(uniform)), uniform)
    uniform_means = []
    for position in (0, 1, 24, 48):
        losses = np.zeros(49)
        losses[position] = 1.0
        uniform_means.append(weighted_mean(losses))
    assert len(set(uniform_means)) == 1

    rng = np.random.default_rng(7007)
    for _ in range(30):
        size = int(rng.integers(1, 50))
        raw = 10 ** rng.uniform(-150, 150, size=size)
        probabilities = normalized_weights(raw, size)
        assert abs(math.fsum(float(value) for value in probabilities) - 1.0) <= math.ulp(1.0)
        assert np.array_equal(normalized_weights(probabilities, size), probabilities)


def test_boolean_losses_and_weights_are_not_silently_coerced_to_numbers():
    with pytest.raises(RiskValidationError, match="not boolean"):
        weighted_cvar([False, True], 0.5)
    with pytest.raises(RiskValidationError, match="not boolean"):
        normalized_weights([False, True], 2)


def test_finite_endpoint_losses_do_not_overflow_a_convex_risk_sum():
    maximum = np.finfo(float).max
    losses = [maximum] * 10
    assert weighted_mean(losses) == maximum
    assert weighted_cvar(losses, 0.5) == maximum
    assert rockafellar_uryasev_value(losses, 0.5)[0] == maximum


def test_weighted_sum_preserves_collectively_representable_underflowed_products():
    losses = [1e-124] * 10 + [0.0]
    weights = [1e-200] * 10 + [1.0]
    assert weighted_mean(losses, weights) == 1e-323

    maximum = np.finfo(float).max
    cancelling_losses = [maximum] + [1e-124] * 10 + [-maximum]
    cancelling_weights = [0.5] + [1e-200] * 10 + [0.5]
    assert weighted_mean(cancelling_losses, cancelling_weights) == 1e-323
    assert weighted_mean([1e-123] * 10 + [0.0], [3e-201] * 10 + [1.0]) == 3e-323


def test_exact_nonzero_risk_results_that_cannot_be_reported_are_rejected():
    minimum = np.nextafter(0.0, 1.0)
    with pytest.raises(RiskValidationError, match="weighted mean is non-zero"):
        weighted_mean([minimum, 0.0], [minimum, 1.0])
    with pytest.raises(RiskValidationError, match="weighted CVaR is non-zero"):
        weighted_cvar([minimum, 0.0], 0.5, [minimum, 1.0])


def test_signed_weighted_mean_uses_the_exact_input_float_dot_product():
    assert weighted_mean([7e15, -3e15], [0.3, 0.7]) == 0.05551115123125783
    assert weighted_mean(
        [5.720798063598169e307, -3.086197330576119e307, 0.0],
        [0.25588716910072784, 0.4743309207703158, 0.26978191012895636],
    ) == -9.862453732262133e290


def test_cvar_allocates_the_tail_exactly_before_signed_cancellation():
    value = weighted_cvar([4e15, -1e16], 0.3)
    cvar, zeta = rockafellar_uryasev_value([4e15, -1e16], 0.3)
    assert value == -0.1586032892321652
    assert cvar == value
    assert cvar >= zeta


def test_cvar_exactly_normalizes_tiny_atoms_at_a_tail_boundary():
    maximum = 1e300
    losses = [maximum] + [1e-124] * 10 + [-maximum, -np.finfo(float).max]
    weights = [0.25] + [1e-200] * 10 + [0.25, 0.5]
    cvar, zeta = rockafellar_uryasev_value(losses, 0.5, weights)
    assert weighted_cvar(losses, 0.5, weights) == 1e101
    assert cvar == 1e101
    assert zeta == -maximum
    assert cvar >= zeta

    boundary_cvar, boundary_var = rockafellar_uryasev_value(
        [-maximum, 0.0], 1e-300, [1.0, 1e300]
    )
    assert boundary_cvar == boundary_var == 0.0


def test_weighted_cvar_matches_rockafellar_uryasev_over_random_ties_and_weights():
    rng = np.random.default_rng(20260710)
    for _ in range(100):
        losses = rng.integers(0, 6, size=11).astype(float)
        weights = rng.uniform(0.001, 1.0, size=len(losses))
        alpha = float(rng.uniform(0.01, 0.99))
        cvar, _ = rockafellar_uryasev_value(losses, alpha, weights)
        conditional = tail_conditional_weights(losses, alpha, weights)
        assert conditional.sum() == pytest.approx(1.0)
        assert weighted_cvar(losses, alpha, weights) == pytest.approx(cvar)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.1])
def test_invalid_alpha_is_rejected(alpha):
    with pytest.raises(RiskValidationError):
        weighted_cvar([1.0], alpha)


def test_risk_inputs_reject_unrepresentable_python_integers_cleanly():
    too_large = 10**10_000
    with pytest.raises(RiskValidationError, match="alpha"):
        weighted_cvar([1.0], too_large)
    with pytest.raises(RiskValidationError, match="losses"):
        weighted_cvar([too_large], 0.5)
    with pytest.raises(RiskValidationError, match="weights"):
        normalized_weights([too_large], 1)
    with pytest.raises(RiskValidationError, match="not text"):
        weighted_cvar("12", 0.5)
    with pytest.raises(RiskValidationError, match="not text"):
        normalized_weights("12", 2)
    with pytest.raises(RiskValidationError, match="bytes-like"):
        weighted_cvar(memoryview(b"12"), 0.5)
    with pytest.raises(RiskValidationError, match="bytes-like"):
        normalized_weights(memoryview(b"12"), 2)
    with pytest.raises(RiskValidationError, match="iterable"):
        weighted_cvar(5, 0.5)
    with pytest.raises(RiskValidationError, match="iterable"):
        normalized_weights(5, 1)
    with pytest.raises(RiskValidationError, match="at least one"):
        normalized_weights(None, True)
    with pytest.raises(RiskValidationError, match="scalar"):
        weighted_cvar([1.0, 2.0], np.asarray([0.5]))


def test_risk_vectors_reject_tables_unordered_inputs_and_underflowed_atoms():
    frame = pd.DataFrame([[999.0, 888.0]], columns=[1.0, 2.0])
    with pytest.raises(RiskValidationError, match="one-dimensional"):
        weighted_mean(frame)
    with pytest.raises(RiskValidationError, match="one-dimensional"):
        normalized_weights(frame, 2)
    with pytest.raises(RiskValidationError, match="ordered"):
        weighted_mean({1.0, 2.0})
    with pytest.raises(RiskValidationError, match="too small"):
        weighted_mean([Fraction(1, 10**1000)])
    with pytest.raises(RiskValidationError, match="too small"):
        normalized_weights([Fraction(-1, 10**1000), 1], 2)
    with pytest.raises(RiskValidationError, match="complex"):
        weighted_mean([np.complex128(1 + 2j)])
    with pytest.raises(RiskValidationError, match="complex"):
        normalized_weights([np.complex128(1 + 2j)], 1)
    with pytest.raises(RiskValidationError, match="scalar"):
        weighted_cvar([1.0], np.complex128(0.5 + 1j))
    with pytest.raises(RiskValidationError, match="scalar"):
        weighted_cvar([1.0], "0.5")
    with pytest.raises(RiskValidationError, match="numeric scalar"):
        weighted_mean([np.array(True)])
    with pytest.raises(RiskValidationError, match="numeric scalar"):
        normalized_weights([np.array("1")], 1)
    with pytest.raises(RiskValidationError, match="scalar"):
        weighted_cvar([1.0], np.array("0.5"))
