"""Honest out-of-sample policy evaluation helpers.

These functions report empirical performance on an independent scenario set.
They intentionally do not present a dimensionally invalid 'optimality bound'
or silently claim a Mak--Morton--Wood confidence interval.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Set
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Optional, Sequence

import numpy as np

from .bom import SupplierOption
from .risk import normalized_weights, validate_alpha, weighted_cvar, weighted_mean


@dataclass(frozen=True)
class PolicyEvaluation:
    """Independent empirical performance of a fixed supplier policy."""

    losses: np.ndarray
    mean_cost: float
    cvar: float
    objective: float
    alpha: float
    risk_weight: float


def _finite_nonnegative(value: float, name: str) -> float:
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
        raise ValueError(f"{name} must be a finite non-negative number")
    original_value = value
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = float(value)
    except (TypeError, ValueError, OverflowError, Warning) as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc
    if not math.isfinite(result) or result < 0 or original_value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    if result == 0.0 and original_value != 0:
        raise ValueError(
            f"{name} is non-zero but too small for floating-point representation"
        )
    return result


def evaluate_policy(
    selected: Sequence[SupplierOption],
    event_counts: Iterable[float],
    *,
    alpha: float = 0.95,
    risk_weight: float = 1.0,
    scenario_weights: Optional[Iterable[float]] = None,
    lead_time_cost_per_day: float = 0.0,
) -> PolicyEvaluation:
    """Evaluate a fixed policy without re-optimizing it on the evaluation set."""

    if isinstance(selected, (str, bytes, bytearray, memoryview, Mapping, Set)) or getattr(
        selected, "ndim", 1
    ) != 1:
        raise ValueError(
            "selected policy must be an ordered one-dimensional iterable of SupplierOption instances"
        )
    try:
        selected_options = tuple(selected)
    except TypeError as exc:
        raise ValueError("selected policy must be an iterable of SupplierOption instances") from exc
    if not selected_options:
        raise ValueError("selected policy must contain at least one supplier option")
    seen_components: set[str] = set()
    for option in selected_options:
        if not isinstance(option, SupplierOption):
            raise TypeError("selected policy entries must be SupplierOption instances")
        if option.component in seen_components:
            raise ValueError(f"selected policy contains multiple options for component {option.component!r}")
        seen_components.add(option.component)

    alpha = validate_alpha(alpha)
    if isinstance(event_counts, (str, bytes, bytearray, memoryview)):
        raise ValueError("event_counts must be a numeric iterable, not text or bytes-like data")
    if isinstance(event_counts, (Mapping, Set)) or getattr(event_counts, "ndim", 1) != 1:
        raise ValueError("event_counts must be an ordered one-dimensional numeric iterable")
    try:
        raw_events = list(event_counts)
    except TypeError as exc:
        raise ValueError("event_counts must be a numeric iterable") from exc
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
        for value in raw_events
    ):
        raise ValueError(
            "event_counts must contain numeric scalar values, not boolean, complex, text, or bytes-like values"
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            converted_events = [float(value) for value in raw_events]
    except (TypeError, ValueError, OverflowError, Warning) as exc:
        raise ValueError("event_counts must be a non-empty finite non-negative vector") from exc
    if any(
        converted == 0.0 and value != 0
        for value, converted in zip(raw_events, converted_events)
    ):
        raise ValueError(
            "event_counts contain a non-zero value too small for floating-point representation"
        )
    events = np.asarray(converted_events, dtype=float)
    if events.ndim != 1 or len(events) == 0 or not np.isfinite(events).all() or np.any(events < 0):
        raise ValueError("event_counts must be a non-empty finite non-negative vector")
    risk_weight = _finite_nonnegative(risk_weight, "risk_weight")
    lead_time_cost_per_day = _finite_nonnegative(lead_time_cost_per_day, "lead_time_cost_per_day")
    probabilities = normalized_weights(scenario_weights, len(events))
    exact_rate = Fraction.from_float(lead_time_cost_per_day)
    exact_deterministic = sum(
        (
            Fraction.from_float(float(option.base_cost))
            + exact_rate * Fraction.from_float(float(option.lead_time))
            for option in selected_options
        ),
        Fraction(0),
    )
    exact_exposure = sum(
        (Fraction.from_float(float(option.kappa)) for option in selected_options),
        Fraction(0),
    )
    loss_values: list[float] = []
    for event in events:
        exact_loss = exact_deterministic + exact_exposure * Fraction.from_float(
            float(event)
        )
        try:
            loss = float(exact_loss)
        except OverflowError:
            raise ValueError(
                "selected policy produces non-finite scenario losses or unrepresentable values; "
                "rescale monetary inputs"
            ) from None
        if not math.isfinite(loss) or (loss == 0.0 and exact_loss != 0):
            raise ValueError(
                "selected policy produces non-finite scenario losses or unrepresentable values; "
                "rescale monetary inputs"
            )
        loss_values.append(loss)
    losses = np.asarray(loss_values, dtype=float)
    mean_cost = weighted_mean(losses, probabilities)
    cvar = weighted_cvar(losses, alpha, probabilities)
    exact_objective = Fraction.from_float(mean_cost) + Fraction.from_float(
        risk_weight
    ) * Fraction.from_float(cvar)
    try:
        objective = float(exact_objective)
    except OverflowError:
        raise ValueError(
            "policy objective overflows or underflows floating-point arithmetic; "
            "rescale monetary inputs"
        ) from None
    if not math.isfinite(objective) or (objective == 0.0 and exact_objective != 0):
        raise ValueError(
            "policy objective overflows or underflows floating-point arithmetic; "
            "rescale monetary inputs"
        )
    return PolicyEvaluation(
        losses=losses,
        mean_cost=mean_cost,
        cvar=cvar,
        objective=objective,
        alpha=alpha,
        risk_weight=risk_weight,
    )
