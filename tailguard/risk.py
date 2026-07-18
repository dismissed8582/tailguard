"""Risk-measure primitives shared by every Tailguard solver.

The implementation uses the Rockafellar--Uryasev empirical definition rather
than a rounded percentile slice.  In particular, it handles non-integral tail
mass and ties at VaR consistently with the MILP linearisation.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Set
from fractions import Fraction
from typing import Iterable, Optional

import numpy as np


class RiskValidationError(ValueError):
    """Raised for invalid losses, weights, or CVaR confidence levels."""


_MIN_FLOAT_DUAL_ALPHA = 1e-8


def validate_alpha(alpha: float) -> float:
    if isinstance(
        alpha,
        (
            bool,
            np.bool_,
            str,
            bytes,
            bytearray,
            memoryview,
            complex,
            np.complexfloating,
            np.ndarray,
        ),
    ) or getattr(alpha, "ndim", 0) != 0:
        raise RiskValidationError("alpha must be a scalar number strictly between 0 and 1")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            alpha = float(alpha)
    except (TypeError, ValueError, OverflowError, Warning) as exc:
        raise RiskValidationError("alpha must be a number strictly between 0 and 1") from exc
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise RiskValidationError("alpha must be strictly between 0 and 1")
    return alpha


def normalized_weights(weights: Optional[Iterable[float]], size: int) -> np.ndarray:
    """Return validated relative probability weights of length ``size``.

    Binary64 cannot represent an exactly summing, exactly uniform vector for
    every length.  The returned values are therefore interpreted by their
    ratios; they may sum to one with a one-ULP rounding error rather than
    assigning that error to an arbitrary scenario.
    """

    if isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)) or size < 1:
        raise RiskValidationError("at least one scenario is required")
    size = int(size)
    if weights is None:
        values = np.full(size, 1.0 / size, dtype=float)
    else:
        if isinstance(weights, (str, bytes, bytearray, memoryview)):
            raise RiskValidationError("scenario weights must be a numeric iterable, not text or bytes-like data")
        if isinstance(weights, (Mapping, Set)) or getattr(weights, "ndim", 1) != 1:
            raise RiskValidationError(
                "scenario weights must be an ordered one-dimensional numeric iterable"
            )
        try:
            raw_values = list(weights)
        except TypeError as exc:
            raise RiskValidationError("scenario weights must be a numeric iterable") from exc
        if any(
            isinstance(
                value,
                (
                    bool,
                    np.bool_,
                    str,
                    bytes,
                    bytearray,
                    memoryview,
                    complex,
                    np.complexfloating,
                    np.ndarray,
                ),
            )
            or getattr(value, "ndim", 0) != 0
            for value in raw_values
        ):
            raise RiskValidationError(
                "scenario weights must contain numeric scalar values, not boolean, complex, text, or bytes-like values"
            )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                converted_values = [float(value) for value in raw_values]
        except (TypeError, ValueError, OverflowError, Warning) as exc:
            raise RiskValidationError("scenario weights must be finite non-negative numbers") from exc
        if any(
            converted == 0.0 and value != 0
            for value, converted in zip(raw_values, converted_values)
        ):
            raise RiskValidationError(
                "scenario weights contain a non-zero value too small for floating-point representation"
            )
        if any(value < 0 for value in converted_values):
            raise RiskValidationError("scenario weights must be finite and non-negative")
        values = np.asarray(converted_values, dtype=float)
    if values.ndim != 1 or len(values) != size:
        raise RiskValidationError("scenario weights must be a one-dimensional array matching losses")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise RiskValidationError("scenario weights must be finite and non-negative")
    try:
        direct_total = math.fsum(float(value) for value in values)
    except OverflowError:
        direct_total = math.inf
    # Public solvers and risk helpers deliberately pass validated probability
    # vectors to one another. Preserve an existing binary64 partition
    # bit-for-bit so nested evaluation cannot perturb a near-tied policy by
    # normalizing twice. Some exactly uniform vectors can only sum to within
    # one ULP of one without breaking permutation invariance.
    if float(np.max(values)) <= 1.0 and abs(direct_total - 1.0) <= math.ulp(1.0):
        return values.copy()
    scale = float(np.max(values))
    if scale <= 0:
        raise RiskValidationError("scenario weights must have positive total mass")
    # Dividing first prevents an otherwise valid vector such as
    # [1e308, 1e308] from overflowing while its probabilities are computed.
    scaled = values / scale
    if np.any((values > 0) & (scaled == 0)):
        raise RiskValidationError(
            "relative scenario-weight scale cannot be represented without deleting positive probability mass"
        )
    total = math.fsum(float(value) for value in scaled)
    if not math.isfinite(total) or total <= 0:  # Defensive after finite input validation.
        raise RiskValidationError("scenario weights must have positive finite total mass")
    probabilities = scaled / total
    if np.any((values > 0) & (probabilities == 0)):
        raise RiskValidationError(
            "relative scenario-weight scale cannot be represented without deleting positive probability mass"
        )
    # Do not force a residual onto one atom. For tied weights (including a
    # uniform default), doing so makes the result depend on scenario order.
    # Every consumer below treats these floats as exact relative masses.
    return probabilities


def _loss_array(losses: Iterable[float]) -> np.ndarray:
    if isinstance(losses, (str, bytes, bytearray, memoryview)):
        raise RiskValidationError("losses must be a numeric iterable, not text or bytes-like data")
    if isinstance(losses, (Mapping, Set)) or getattr(losses, "ndim", 1) != 1:
        raise RiskValidationError("losses must be an ordered one-dimensional numeric iterable")
    try:
        raw_values = list(losses)
    except TypeError as exc:
        raise RiskValidationError("losses must be a numeric iterable") from exc
    if any(
        isinstance(
            value,
            (
                bool,
                np.bool_,
                str,
                bytes,
                bytearray,
                memoryview,
                complex,
                np.complexfloating,
                np.ndarray,
            ),
        )
        or getattr(value, "ndim", 0) != 0
        for value in raw_values
    ):
        raise RiskValidationError(
            "losses must contain numeric scalar values, not boolean, complex, text, or bytes-like values"
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            converted_values = [float(value) for value in raw_values]
    except (TypeError, ValueError, OverflowError, Warning) as exc:
        raise RiskValidationError("losses must be finite numbers") from exc
    if any(
        converted == 0.0 and value != 0
        for value, converted in zip(raw_values, converted_values)
    ):
        raise RiskValidationError(
            "losses contain a non-zero value too small for floating-point representation"
        )
    values = np.asarray(converted_values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise RiskValidationError("losses must be a non-empty one-dimensional array")
    if not np.isfinite(values).all():
        raise RiskValidationError("losses must be finite")
    return values


def _finite_exact_result(value: Fraction, label: str) -> float:
    """Convert an exact result only when binary64 can report it faithfully."""

    try:
        converted = float(value)
    except OverflowError as exc:  # pragma: no cover - convex risk results are bounded by finite losses.
        raise RiskValidationError(f"{label} is outside floating-point range") from exc
    if not math.isfinite(converted):  # pragma: no cover - Fraction conversion handles overflow.
        raise RiskValidationError(f"{label} is outside floating-point range")
    if converted == 0.0 and value != 0:
        raise RiskValidationError(
            f"{label} is non-zero but too small for floating-point representation"
        )
    return converted


def _stable_weighted_sum(values: np.ndarray, probabilities: np.ndarray) -> float:
    """Return a finite convex combination without overflowing its accumulation."""

    support = probabilities > 0
    supported_values = values[support]
    supported_probabilities = probabilities[support]
    products = [
        float(probability) * float(value)
        for probability, value in zip(supported_probabilities, supported_values)
    ]
    exact_probabilities = tuple(
        Fraction.from_float(float(probability)) for probability in supported_probabilities
    )
    exact_probability_total = sum(exact_probabilities, Fraction(0))
    needs_exact = exact_probability_total != 1 or any(
        (product == 0.0 and probability != 0.0 and value != 0.0)
        or (0.0 < abs(product) < np.finfo(float).tiny)
        for product, probability, value in zip(
            products, supported_probabilities, supported_values
        )
    )
    try:
        result = math.fsum(products) if not needs_exact else 0.0
    except OverflowError:
        needs_exact = True
        result = 0.0
    if not needs_exact and any(product < 0 for product in products) and any(
        product > 0 for product in products
    ):
        # ``fsum`` accurately adds the *rounded* products, but multiplication
        # round-off can still materially perturb any mixed-sign dot product.
        # Public loss helpers permit signed values, so do not guess at a safe
        # cancellation threshold: evaluate those operands exactly.
        needs_exact = True
    if needs_exact:
        # Float multiplication can underflow each term even when their sum is
        # representable; fsum can also overflow a last-bit same-sign partial.
        # Exact rational arithmetic is rare here and preserves every bit in the
        # already-normalized public float inputs, even after huge cancellation.
        exact_result = sum(
            (
                probability * Fraction.from_float(float(value))
                for probability, value in zip(exact_probabilities, supported_values)
            ),
            Fraction(0),
        ) / exact_probability_total
        exact_result = min(
            max(exact_result, Fraction.from_float(float(np.min(supported_values)))),
            Fraction.from_float(float(np.max(supported_values))),
        )
        result = _finite_exact_result(exact_result, "weighted mean")
    # A probability-weighted sum lies in this interval.  Clamping only removes
    # a last-bit round-off that could otherwise turn a finite endpoint into inf.
    return float(
        min(
            max(result, float(np.min(supported_values))),
            float(np.max(supported_values)),
        )
    )


def _fraction_probabilities(probabilities: np.ndarray) -> tuple[Fraction, ...]:
    """Interpret a public float probability vector as exact relative weights."""

    values = tuple(Fraction.from_float(float(probability)) for probability in probabilities)
    total = sum(values, Fraction(0))
    return tuple(value / total for value in values)


def _exact_weighted_cvar(values: np.ndarray, alpha: float, probabilities: np.ndarray) -> float:
    """Evaluate the empirical upper tail from the exact input-float ratios."""

    exact_probabilities = _fraction_probabilities(probabilities)
    tail_mass = Fraction(1) - Fraction.from_float(float(alpha))
    remaining = tail_mass
    weighted_loss = Fraction(0)
    order = np.argsort(-values, kind="mergesort")
    for index in order:
        if remaining <= 0:
            break
        allocated = min(exact_probabilities[index], remaining)
        weighted_loss += allocated * Fraction.from_float(float(values[index]))
        remaining -= allocated
    if remaining > 0:  # pragma: no cover - exact normalized probabilities cover one.
        raise RiskValidationError("scenario probabilities do not cover the requested tail mass")
    return _finite_exact_result(weighted_loss / tail_mass, "weighted CVaR")


def _exact_empirical_var(values: np.ndarray, alpha: float, probabilities: np.ndarray) -> float:
    """Return a deterministic lower empirical alpha-quantile exactly."""

    target = Fraction.from_float(float(alpha))
    cumulative = Fraction(0)
    exact_probabilities = _fraction_probabilities(probabilities)
    order = np.argsort(values, kind="mergesort")
    for index in order:
        cumulative += exact_probabilities[index]
        if cumulative >= target:
            return float(values[index])
    return float(values[order[-1]])  # pragma: no cover - exact probabilities sum to one.


def weighted_mean(losses: Iterable[float], weights: Optional[Iterable[float]] = None) -> float:
    """Return the probability-weighted mean loss."""

    values = _loss_array(losses)
    probabilities = normalized_weights(weights, len(values))
    return _stable_weighted_sum(values, probabilities)


def tail_conditional_weights(
    losses: Iterable[float], alpha: float, weights: Optional[Iterable[float]] = None
) -> np.ndarray:
    """Return one valid dual/subgradient weight vector for empirical CVaR.

    In exact arithmetic its entries sum to one and no entry exceeds
    ``p_i / (1-alpha)``; the returned float vector satisfies those identities
    to ordinary float rounding. Ties at VaR are resolved deterministically by
    stable descending order. A float vector cannot represent the required
    complement weights for extremely small alpha or the exact bounds induced
    by a subnormal probability atom, so those cases are rejected;
    :func:`weighted_cvar` still evaluates them using exact rational allocation.
    """

    values = _loss_array(losses)
    alpha = validate_alpha(alpha)
    if alpha < _MIN_FLOAT_DUAL_ALPHA:
        raise RiskValidationError(
            f"alpha must be at least {_MIN_FLOAT_DUAL_ALPHA:g} for float dual weights; use weighted_cvar directly"
        )
    probabilities = normalized_weights(weights, len(values))
    if np.any((probabilities > 0) & (probabilities < np.finfo(float).tiny)):
        raise RiskValidationError(
            "scenario probabilities contain a subnormal atom whose float dual bound is not representable; "
            "use weighted_cvar directly"
        )
    # Construct the mathematical dual exactly, then expose it as float only if
    # no positive mass disappears. This does not rely on ``np.longdouble``,
    # which is only binary64 on some supported platforms (including Apple ARM).
    tail_mass = Fraction(1) - Fraction.from_float(float(alpha))
    remaining = tail_mass
    probability_mass = _fraction_probabilities(probabilities)
    exact_conditional = [Fraction(0) for _ in probabilities]
    order = np.argsort(-values, kind="mergesort")

    for index in order:
        if remaining <= 0:
            break
        allocated = min(probability_mass[index], remaining)
        exact_conditional[index] = allocated / tail_mass
        remaining -= allocated

    if remaining > 0:  # pragma: no cover - exact normalized probabilities cover one.
        raise RiskValidationError("scenario probabilities do not cover the requested tail mass")
    conditional = np.asarray([float(value) for value in exact_conditional], dtype=float)
    if any(value > 0 and converted == 0 for value, converted in zip(exact_conditional, conditional)):
        raise RiskValidationError(
            "CVaR dual weights cannot be represented without deleting positive tail mass; "
            "use weighted_cvar directly"
        )
    return conditional


def weighted_cvar(losses: Iterable[float], alpha: float, weights: Optional[Iterable[float]] = None) -> float:
    """Compute empirical CVaR / expected shortfall using exact tail mass.

    For weighted scenarios this is equivalent to minimizing the
    Rockafellar--Uryasev variational expression.  It does not include an entire
    tied VaR atom when only a fraction of that atom belongs to the tail.
    """

    values = _loss_array(losses)
    alpha = validate_alpha(alpha)
    probabilities = normalized_weights(weights, len(values))
    # Computing through rounded float dual weights can catastrophically cancel,
    # and any fixed-precision fallback can misclassify a probability/alpha boundary.
    # Exact rational arithmetic over the already-validated input floats avoids
    # both failures; only the final, necessarily bounded CVaR is converted back.
    return _exact_weighted_cvar(values, alpha, probabilities)


def rockafellar_uryasev_value(
    losses: Iterable[float], alpha: float, weights: Optional[Iterable[float]] = None
) -> tuple[float, float]:
    """Return ``(cvar, zeta)`` from exact empirical tail allocation.

    This small reference implementation is useful in tests and diagnostics.
    The solver itself uses the standard linear formulation.
    """

    values = _loss_array(losses)
    alpha = validate_alpha(alpha)
    probabilities = normalized_weights(weights, len(values))
    # The lower empirical alpha-quantile is a valid RU minimizer. Evaluating
    # every candidate after one global rescaling can erase a small upper tail
    # when an unrelated, very large negative loss is also in the support.
    # Exact tail allocation and quantile accumulation avoid that coupling.
    cvar = weighted_cvar(values, alpha, probabilities)
    zeta = _exact_empirical_var(values, alpha, probabilities)
    return cvar, zeta
