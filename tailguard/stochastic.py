"""Transparent proxy calibration and Cox-process scenario simulation.

An index such as GSCPI is not itself an observed disruption intensity.  The
helpers here therefore expose the positive transformation and target event
rate explicitly and label the result a proxy calibration.  They must not be
interpreted as an empirical calibration from a historical event-count series.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Set
from dataclasses import dataclass
from datetime import date, datetime
from fractions import Fraction
from numbers import Integral, Number
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd


class ScenarioValidationError(ValueError):
    """Raised when calibration or simulation input is not meaningful."""


@dataclass(frozen=True)
class CIRParameters:
    mean_reversion: float
    long_run_mean: float
    volatility: float
    initial_intensity: float

    @property
    def feller_satisfied(self) -> bool:
        """Whether finite, physically valid drift parameters meet the Feller condition."""

        values = (self.mean_reversion, self.long_run_mean, self.volatility)
        try:
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
                for value in values
            ):
                return False
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                mean_reversion, long_run_mean, volatility = (
                    float(value) for value in values
                )
        except Exception:
            # This property is a diagnostic for possibly hand-built parameter
            # objects. Malformed values must make the diagnostic false rather
            # than leaking a conversion warning or an implementation-specific
            # scalar exception to callers.
            return False
        if (
            not all(math.isfinite(value) for value in (mean_reversion, long_run_mean, volatility))
            or mean_reversion <= 0
            or long_run_mean <= 0
            or volatility < 0
        ):
            return False
        if volatility == 0:
            return True
        # Compare the already-validated binary64 inputs exactly. A direct
        # product can overflow, while rounded logarithms can return either
        # answer incorrectly at the equality boundary.
        return (
            2
            * Fraction.from_float(mean_reversion)
            * Fraction.from_float(long_run_mean)
            >= Fraction.from_float(volatility) ** 2
        )


@dataclass(frozen=True)
class CIRProxyFit:
    """Result of the documented Euler-style proxy fit.

    ``time_step_years`` is the median transition length for concise display;
    ``time_steps_years`` contains every transition actually used by the fit.
    The two regularization flags identify which fitted CIR quantity was floored.
    """

    parameters: CIRParameters
    transformed_intensity: np.ndarray
    time_step_years: float
    used_regularization: bool
    time_steps_years: np.ndarray
    mean_reversion_regularized: bool = False
    long_run_mean_regularized: bool = False


def _finite_positive(value: float, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(
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
    ) or getattr(value, "ndim", 0) != 0:
        raise ScenarioValidationError(f"{name} must be a finite number")
    original_value = value
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            value = float(value)
    except (TypeError, ValueError, OverflowError, Warning) as exc:
        raise ScenarioValidationError(f"{name} must be a finite number") from exc
    valid = value >= 0 if allow_zero else value > 0
    if not math.isfinite(value) or not valid:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ScenarioValidationError(f"{name} must be finite and {qualifier}")
    if value == 0.0 and original_value != 0:
        raise ScenarioValidationError(
            f"{name} is non-zero but too small for floating-point representation"
        )
    return value


def _validated_seed(seed: Optional[int]) -> Optional[int]:
    """Return a public integer seed or ``None`` without leaking NumPy errors."""

    if seed is None:
        return None
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ScenarioValidationError("seed must be a non-negative integer or None")
    return int(seed)


def _positive_integer(value: int, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        qualifier = "positive" if minimum == 1 else f"at least {minimum}"
        raise ScenarioValidationError(f"{name} must be an integer {qualifier}")
    return int(value)


def _proxy_intensity(values: Iterable[float], target_mean_events: float) -> np.ndarray:
    if isinstance(values, (str, bytes, bytearray, memoryview)):
        raise ScenarioValidationError("proxy observations must be a numeric iterable, not text or bytes-like data")
    if isinstance(values, (Mapping, Set)) or getattr(values, "ndim", 1) != 1:
        raise ScenarioValidationError(
            "proxy observations must be an ordered one-dimensional numeric iterable"
        )
    try:
        raw_values = list(values)
    except TypeError as exc:
        raise ScenarioValidationError("proxy observations must be a numeric iterable") from exc
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
        raise ScenarioValidationError(
            "proxy observations must contain real numeric scalar values, not boolean, "
            "complex, text, or bytes-like values"
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            converted_values = [float(value) for value in raw_values]
    except (TypeError, ValueError, OverflowError, Warning) as exc:
        raise ScenarioValidationError("proxy observations must be finite numbers") from exc
    if any(
        converted == 0.0 and value != 0
        for value, converted in zip(raw_values, converted_values)
    ):
        raise ScenarioValidationError(
            "proxy observations contain a non-zero value too small for floating-point representation"
        )
    raw = np.asarray(converted_values, dtype=float)
    if raw.ndim != 1 or len(raw) < 4:
        raise ScenarioValidationError("at least four proxy observations are required")
    if not np.isfinite(raw).all():
        raise ScenarioValidationError("proxy observations must be finite")
    target_mean_events = _finite_positive(target_mean_events, "target_mean_events")

    # A simple non-negative affine proxy.  The small floor is explicit and
    # prevents division by zero; it is not a claim that the source index is a
    # measured event intensity.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        shifted = raw - raw.min()
        shifted_scale = float(np.max(np.abs(shifted)))
        standard_deviation = (
            0.0
            if shifted_scale == 0
            else shifted_scale * float(np.std(shifted / shifted_scale, ddof=0))
        )
        scale = max(standard_deviation, 1.0)
        intensity = shifted / scale + 0.05
        intensity *= target_mean_events / float(np.mean(intensity))
    if not np.isfinite(intensity).all() or np.any(intensity <= 0):
        raise ScenarioValidationError("proxy values cannot be transformed safely; rescale the input index")
    return intensity


def fit_cir_proxy(
    series: pd.DataFrame,
    *,
    value_column: str = "value",
    date_column: Optional[str] = "date",
    target_mean_events: float = 0.5,
    minimum_mean_reversion: float = 1e-4,
) -> CIRProxyFit:
    """Fit an Euler-style CIR proxy to a positive transformed index.

    If the unconstrained linear regression is not mean reverting, the function
    applies the documented ``minimum_mean_reversion`` regularization and marks
    the fit accordingly.  This keeps the simulator stable while surfacing the
    modelling limitation instead of silently flooring parameters.
    """

    if not isinstance(series, pd.DataFrame):
        raise ScenarioValidationError("series must be a pandas DataFrame")
    if not isinstance(value_column, str) or not value_column:
        raise ScenarioValidationError("value_column must be a non-empty string")
    if date_column is not None and (not isinstance(date_column, str) or not date_column):
        raise ScenarioValidationError("date_column must be a non-empty string or None")
    if date_column == value_column:
        raise ScenarioValidationError("date_column and value_column must name distinct columns")

    def unique_column(label: str, role: str) -> pd.Series:
        positions = [
            index
            for index, column in enumerate(series.columns)
            if isinstance(column, str) and column == label
        ]
        if len(positions) == 0:
            raise ScenarioValidationError(f"series is missing {label!r}")
        if len(positions) > 1:
            raise ScenarioValidationError(f"series has duplicate {role} columns named {label!r}")
        return series.iloc[:, positions[0]]

    value_series = unique_column(value_column, "value")
    if pd.api.types.is_datetime64_any_dtype(value_series.dtype) or pd.api.types.is_timedelta64_dtype(
        value_series.dtype
    ):
        raise ScenarioValidationError(
            "series values must be numeric proxy observations, not dates or durations"
        )
    if any(
        isinstance(
            value,
            (date, datetime, np.datetime64, pd.Timestamp, np.timedelta64, pd.Timedelta),
        )
        for value in value_series.array
    ):
        raise ScenarioValidationError(
            "series values must be numeric proxy observations, not dates or durations"
        )
    if any(isinstance(value, (bool, np.bool_)) for value in value_series.array):
        raise ScenarioValidationError("series values must be numeric finite numbers, not boolean")
    if any(isinstance(value, (complex, np.complexfloating)) for value in value_series.array):
        raise ScenarioValidationError("series values must be real numeric finite numbers, not complex")
    intensity = _proxy_intensity(value_series.array, target_mean_events)
    minimum_mean_reversion = _finite_positive(minimum_mean_reversion, "minimum_mean_reversion")

    if date_column is None:
        time_steps = np.full(len(intensity) - 1, 1.0 / 12.0, dtype=float)
    else:
        date_series = unique_column(date_column, "date")
        if pd.api.types.is_numeric_dtype(date_series.dtype) or pd.api.types.is_timedelta64_dtype(
            date_series.dtype
        ) or any(
            (
                isinstance(value, np.ndarray)
                or isinstance(value, (Number, np.bool_))
            )
            and not isinstance(value, (date, datetime, np.datetime64, pd.Timestamp))
            for value in date_series.array
        ):
            raise ScenarioValidationError(
                "dates must be valid date-like values, not numbers; use explicit strings or datetimes"
            )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                dates = pd.to_datetime(
                    date_series, format="ISO8601", errors="coerce", utc=True
                )
        except (TypeError, ValueError, OverflowError, Warning):
            raise ScenarioValidationError("dates must be valid and strictly increasing") from None
        if dates.isna().any() or not dates.is_monotonic_increasing:
            raise ScenarioValidationError("dates must be valid and strictly increasing")
        # Subtract Python integers rather than NumPy int64 ticks. Respect the
        # DatetimeArray's native unit: forcing every Timestamp to nanoseconds
        # overflows for valid lower-resolution dates outside the ns epoch range.
        seconds_per_tick = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
        date_unit = dates.array.unit
        if date_unit not in seconds_per_tick:  # pragma: no cover - pandas 2.x uses one of the units above.
            raise ScenarioValidationError(f"unsupported timestamp resolution: {date_unit}")
        date_ticks = tuple(int(value) for value in dates.array.asi8)
        elapsed_days = np.asarray(
            [
                (right - left) * seconds_per_tick[date_unit] / (24 * 60 * 60)
                for left, right in zip(date_ticks, date_ticks[1:])
            ],
            dtype=float,
        )
        if np.any(elapsed_days <= 0):
            raise ScenarioValidationError("dates must be strictly increasing")
        time_steps = elapsed_days / 365.25
    if not np.isfinite(time_steps).all() or np.any(time_steps <= 0):
        raise ScenarioValidationError("time steps must be finite and positive")

    lag = intensity[:-1]
    differences = np.diff(intensity)
    # With nonuniform observations, fit the Euler increment directly:
    # d lambda = (a*b)*dt - a*lambda*dt + eta*sqrt(lambda*dt)*epsilon.
    # Weighted least squares acknowledges the Euler residual scale while using
    # the actual transition length rather than silently substituting a median.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        design = np.column_stack([time_steps, -lag * time_steps])
        residual_scale = np.maximum(lag * time_steps, 1e-12)
        sqrt_weights = 1.0 / np.sqrt(residual_scale)
        weighted_design = design * sqrt_weights[:, None]
        weighted_differences = differences * sqrt_weights
    if not all(
        np.isfinite(values).all()
        for values in (design, residual_scale, sqrt_weights, weighted_design, weighted_differences)
    ):
        raise ScenarioValidationError("CIR proxy fit overflows numerical arithmetic; rescale the input index")
    try:
        if np.linalg.matrix_rank(weighted_design) < 2:
            raise ScenarioValidationError("proxy series does not identify both CIR drift parameters")
        coefficients, *_ = np.linalg.lstsq(weighted_design, weighted_differences, rcond=None)
    except np.linalg.LinAlgError as exc:
        raise ScenarioValidationError("CIR proxy fit could not be solved stably; rescale the input index") from exc
    intercept, unconstrained_a = (float(coefficients[0]), float(coefficients[1]))
    if not math.isfinite(intercept) or not math.isfinite(unconstrained_a):
        raise ScenarioValidationError("CIR proxy fit overflows numerical arithmetic; rescale the input index")
    mean_reversion_regularized = unconstrained_a < minimum_mean_reversion
    mean_reversion = max(unconstrained_a, minimum_mean_reversion)
    unconstrained_long_run_mean = intercept / mean_reversion
    minimum_long_run_mean = 1e-8
    long_run_mean_regularized = unconstrained_long_run_mean < minimum_long_run_mean
    long_run_mean = max(unconstrained_long_run_mean, minimum_long_run_mean)
    regularized = mean_reversion_regularized or long_run_mean_regularized

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        fitted_difference = mean_reversion * (long_run_mean - lag) * time_steps
        residual = differences - fitted_difference
        variance_terms = residual**2 / residual_scale
    if not all(np.isfinite(values).all() for values in (fitted_difference, residual, variance_terms)):
        raise ScenarioValidationError("CIR proxy fit overflows numerical arithmetic; rescale the input index")
    variance_scale = float(np.max(variance_terms))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mean_variance = (
            0.0
            if variance_scale == 0
            else variance_scale * float(np.mean(variance_terms / variance_scale))
        )
        volatility = float(np.sqrt(max(mean_variance, 0.0)))
    if not math.isfinite(volatility):
        raise ScenarioValidationError("CIR proxy fit overflows numerical arithmetic; rescale the input index")
    params = CIRParameters(
        mean_reversion=mean_reversion,
        long_run_mean=long_run_mean,
        volatility=volatility,
        initial_intensity=float(intensity[-1]),
    )
    return CIRProxyFit(
        parameters=params,
        transformed_intensity=intensity,
        time_step_years=float(np.median(time_steps)),
        used_regularization=regularized,
        time_steps_years=time_steps.copy(),
        mean_reversion_regularized=mean_reversion_regularized,
        long_run_mean_regularized=long_run_mean_regularized,
    )


def _scaled_nonnegative_product(
    scalar: float, values: np.ndarray, label: str
) -> np.ndarray:
    """Multiply two non-negative factors and reject true range loss."""

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        product = scalar * values
    nonzero_product = (scalar > 0) & (values > 0)
    if np.any(nonzero_product & ((product == 0) | ~np.isfinite(product))):
        raise ScenarioValidationError(
            f"{label} is outside floating-point range; rescale the model"
        )
    return product


def _scaled_diffusion_term(
    volatility: float,
    sqrt_intensity: np.ndarray,
    sqrt_time_step: float,
    shock: np.ndarray,
) -> np.ndarray:
    """Multiply CIR diffusion factors without avoidable range loss."""

    if not np.isfinite(shock).all():  # pragma: no cover - NumPy normals are finite.
        raise ScenarioValidationError("random generator returned a non-finite shock")
    volatility_mantissa, volatility_exponent = math.frexp(volatility)
    time_mantissa, time_exponent = math.frexp(sqrt_time_step)
    intensity_mantissa, intensity_exponent = np.frexp(sqrt_intensity)
    shock_mantissa, shock_exponent = np.frexp(np.abs(shock))
    mantissa = (
        volatility_mantissa
        * time_mantissa
        * intensity_mantissa
        * shock_mantissa
    )
    mantissa, adjustment = np.frexp(mantissa)
    exponent = (
        np.asarray(intensity_exponent, dtype=np.int64)
        + np.asarray(shock_exponent, dtype=np.int64)
        + int(volatility_exponent)
        + int(time_exponent)
        + np.asarray(adjustment, dtype=np.int64)
    )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        magnitude = np.ldexp(mantissa, exponent)
    nonzero_product = (
        volatility > 0
    ) & (sqrt_intensity > 0) & (sqrt_time_step > 0) & (shock != 0)
    # The scaled mantissa product avoids intermediate range loss, but its three
    # rounded multiplications can still move an exact product across either
    # binary64 endpoint.  Recompute only the tiny boundary bins exactly.  The
    # wider upper guard comfortably covers the standard three-operation
    # relative-error bound while keeping the ordinary simulation path fast.
    maximum = np.finfo(float).max
    upper_boundary_guard = maximum * (1.0 - 16.0 * np.finfo(float).eps)
    range_candidates = nonzero_product & (
        (magnitude <= math.ulp(0.0))
        | (magnitude >= upper_boundary_guard)
        | ~np.isfinite(magnitude)
    )
    for index in np.flatnonzero(range_candidates):
        exact_magnitude = (
            Fraction.from_float(volatility)
            * Fraction.from_float(float(sqrt_intensity.flat[index]))
            * Fraction.from_float(sqrt_time_step)
            * Fraction.from_float(abs(float(shock.flat[index])))
        )
        try:
            exact_float = float(exact_magnitude)
        except OverflowError:
            exact_float = math.inf
        if math.isfinite(exact_float) and exact_float > 0:
            magnitude.flat[index] = exact_float
            continue
        raise ScenarioValidationError(
            "CIR diffusion increment is outside floating-point range; rescale the model"
        )
    return np.copysign(magnitude, shock)


def simulate_cox(
    parameters: CIRParameters,
    *,
    horizon_years: float = 1.0,
    scenarios: int = 1_000,
    steps: int = 252,
    seed: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate a shared-count Cox process using non-negative projected Euler paths.

    The returned tuple is ``(event_counts, integrated_intensities)``.  Each
    scenario draws one Poisson count conditional on the simulated integral,
    which makes the Cox conditioning explicit.

    The projected-Euler grid requires
    ``mean_reversion * horizon_years / steps <= 1``; increase ``steps`` when
    a coarser grid falls outside that supported non-overshooting range.
    """

    if not isinstance(parameters, CIRParameters):
        raise ScenarioValidationError("parameters must be a CIRParameters instance")
    a = _finite_positive(parameters.mean_reversion, "mean_reversion")
    b = _finite_positive(parameters.long_run_mean, "long_run_mean")
    eta = _finite_positive(parameters.volatility, "volatility", allow_zero=True)
    lam0 = _finite_positive(parameters.initial_intensity, "initial_intensity", allow_zero=True)
    horizon_years = _finite_positive(horizon_years, "horizon_years")
    scenarios = _positive_integer(scenarios, "scenarios")
    steps = _positive_integer(steps, "steps")

    seed = _validated_seed(seed)
    rng = np.random.default_rng(seed)
    try:
        dt = horizon_years / steps
    except OverflowError as exc:
        raise ScenarioValidationError("steps is too large to form a finite simulation time step") from exc
    if not math.isfinite(dt) or dt <= 0:
        raise ScenarioValidationError("steps is too large to form a positive finite simulation time step")
    exact_mean_reversion_step = (
        Fraction.from_float(a) * Fraction.from_float(horizon_years) / steps
    )
    if exact_mean_reversion_step > 1:
        raise ScenarioValidationError(
            "mean-reversion time step exceeds the supported non-overshooting "
            "projected-Euler range; increase steps or rescale the model"
        )
    try:
        mean_reversion_step = float(exact_mean_reversion_step)
    except OverflowError as exc:  # pragma: no cover - Fraction normally returns inf below.
        raise ScenarioValidationError(
            "mean-reversion time step is outside floating-point range; rescale the model"
        ) from exc
    if not math.isfinite(mean_reversion_step) or (
        mean_reversion_step == 0 and exact_mean_reversion_step != 0
    ):
        raise ScenarioValidationError(
            "mean-reversion time step is outside floating-point range; rescale the model"
        )
    exact_remaining_mean_reversion = Fraction(1) - exact_mean_reversion_step
    remaining_mean_reversion = float(exact_remaining_mean_reversion)
    if (
        remaining_mean_reversion == 0
        and exact_remaining_mean_reversion != 0
    ):
        raise ScenarioValidationError(
            "mean-reversion complement is outside floating-point range; rescale the model"
        )
    sqrt_dt = math.sqrt(dt)
    intensity = np.full(scenarios, lam0, dtype=float)
    integrated = np.zeros(scenarios, dtype=float)
    integration_compensation = np.zeros(scenarios, dtype=float)
    for _ in range(steps):
        positive = np.maximum(intensity, 0.0)
        shock = rng.standard_normal(scenarios)
        deterministic_level = np.empty_like(positive)
        increasing = positive <= b
        drift_difference = np.abs(b - positive)
        # Interpolate from the nearer endpoint.  This reduces cancellation and
        # makes an exact step just below one use its representable complement,
        # even when the step itself rounds to one in binary64.
        if exact_mean_reversion_step <= Fraction(1, 2):
            drift_offset = _scaled_nonnegative_product(
                mean_reversion_step, drift_difference, "CIR drift increment"
            )
            interpolation_anchor = positive
            increasing_sign = 1.0
        else:
            drift_offset = _scaled_nonnegative_product(
                remaining_mean_reversion,
                drift_difference,
                "CIR drift complement",
            )
            interpolation_anchor = np.full_like(positive, b)
            increasing_sign = -1.0
        with np.errstate(over="ignore", invalid="ignore"):
            deterministic_level[increasing] = (
                interpolation_anchor[increasing]
                + increasing_sign * drift_offset[increasing]
            )
            deterministic_level[~increasing] = (
                interpolation_anchor[~increasing]
                - increasing_sign * drift_offset[~increasing]
            )
        # A representable drift can still round away when added to a much
        # larger state.  That is ordinary per-step binary64 Euler rounding.
        # Only a drift product outside range is rejected above; rejecting an
        # absorbed state addition would make success depend on sampled paths.
        if not np.isfinite(deterministic_level).all():  # pragma: no cover - convex interpolation is bounded.
            raise ScenarioValidationError("CIR parameter scale overflows simulation arithmetic; rescale the model")
        diffusion = _scaled_diffusion_term(
            eta, np.sqrt(positive), sqrt_dt, shock
        )
        with np.errstate(over="ignore", invalid="ignore"):
            unprojected_intensity = deterministic_level + diffusion
        if not np.isfinite(unprojected_intensity).all():
            raise ScenarioValidationError(
                "CIR parameter scale overflows simulation arithmetic; rescale the model"
            )
        # A sufficiently small random draw may correctly round back to the
        # deterministic level. Unlike a persistent deterministic drift, one
        # such signed stochastic perturbation is not evidence that cumulative
        # positive mass has been deleted, so it must not make success depend on
        # the number of sampled scenarios.
        next_intensity = np.maximum(unprojected_intensity, 0.0)
        # Form the trapezoid average relative to its larger endpoint.  The
        # direct ``0.5 * (left + right)`` can overflow for two large finite
        # intensities, while halving each endpoint first deletes a valid
        # minimum-subnormal constant path.
        maximum_intensity = np.maximum(positive, next_intensity)
        minimum_intensity = np.minimum(positive, next_intensity)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            endpoint_ratio = np.divide(
                minimum_intensity,
                maximum_intensity,
                out=np.zeros_like(maximum_intensity),
                where=maximum_intensity > 0,
            )
            trapezoid_average = maximum_intensity * (0.5 + 0.5 * endpoint_ratio)
            increment = trapezoid_average * dt
        positive_trapezoid = maximum_intensity > 0
        if np.any(positive_trapezoid & ((trapezoid_average == 0) | (increment == 0))):
            raise ScenarioValidationError(
                "CIR integration increment underflows floating-point arithmetic; "
                "rescale the model or use fewer steps"
            )
        # Kahan accumulation prevents a sequence of representable positive
        # increments from disappearing next to an earlier, larger integral.
        with np.errstate(over="ignore", invalid="ignore"):
            adjusted_increment = increment - integration_compensation
            next_integrated = integrated + adjusted_increment
            next_compensation = (
                next_integrated - integrated
            ) - adjusted_increment
        if not np.isfinite(next_integrated).all() or not np.isfinite(next_compensation).all():
            raise ScenarioValidationError("CIR parameter scale overflows simulation arithmetic; rescale the model")
        integrated = next_integrated
        integration_compensation = next_compensation
        intensity = next_intensity
    try:
        counts = rng.poisson(integrated).astype(np.int64, copy=False)
    except ValueError as exc:
        raise ScenarioValidationError("CIR integrated intensity is outside the Poisson sampler range") from exc
    return counts, integrated


def _largest_remainder_allocation(raw: np.ndarray, total: int) -> np.ndarray:
    """Allocate exactly ``total`` integers while preserving at least one/stratum."""

    count = len(raw)
    if total < count:
        raise ScenarioValidationError("scenarios must be at least the number of strata")
    base = np.ones(count, dtype=int)
    remaining = total - count
    if remaining == 0:
        return base
    if not np.isfinite(raw).all() or np.max(raw) <= 0:
        raw = np.ones(count, dtype=float)
    scaled = raw / np.max(raw)
    total_weight = math.fsum(float(value) for value in scaled)
    shares = scaled / total_weight * remaining
    extras = np.floor(shares).astype(int)
    leftovers = remaining - int(extras.sum())
    if leftovers:
        order = np.argsort(-(shares - extras), kind="mergesort")
        extras[order[:leftovers]] += 1
    return base + extras


def simulate_stratified_cox(
    parameters: CIRParameters,
    *,
    horizon_years: float = 1.0,
    scenarios: int = 1_000,
    strata: int = 10,
    pilot_size: Optional[int] = None,
    steps: int = 252,
    seed: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a safely weighted stratified resample from a Cox pilot sample.

    This is intentionally described as *pilot-resampling stratification*, not
    an exact conditional-path sampler or a guaranteed Neyman allocation.  It
    avoids unbounded rejection loops and returns probability weights that sum
    to one to floating-point precision.
    """

    seed = _validated_seed(seed)
    strata = _positive_integer(strata, "strata", minimum=2)
    scenarios = _positive_integer(scenarios, "scenarios", minimum=strata)
    if pilot_size is None:
        pilot_size = max(10_000, scenarios * 10)
    pilot_size = _positive_integer(pilot_size, "pilot_size", minimum=strata)

    pilot_counts, pilot_integrals = simulate_cox(
        parameters,
        horizon_years=horizon_years,
        scenarios=pilot_size,
        steps=steps,
        seed=seed,
    )
    # Sorting by rank guarantees non-empty strata even when integrated
    # intensities tie (for example a deterministic CIR path).
    order = np.argsort(pilot_integrals, kind="mergesort")
    groups = [group for group in np.array_split(order, strata) if len(group)]
    probabilities = np.asarray([len(group) / pilot_size for group in groups], dtype=float)
    standard_deviations = np.asarray([np.std(pilot_counts[group], ddof=0) for group in groups], dtype=float)
    allocation = _largest_remainder_allocation(probabilities * standard_deviations, scenarios)
    rng = np.random.default_rng(None if seed is None else seed + 1)

    count_parts: list[np.ndarray] = []
    integral_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    for group, probability, requested in zip(groups, probabilities, allocation):
        sample = rng.choice(group, size=int(requested), replace=True)
        count_parts.append(pilot_counts[sample])
        integral_parts.append(pilot_integrals[sample])
        weight_parts.append(np.full(int(requested), probability / requested, dtype=float))
    counts = np.concatenate(count_parts)
    integrals = np.concatenate(integral_parts)
    weights = np.concatenate(weight_parts)
    weights /= weights.sum()
    return counts, integrals, weights
