"""Validated MILP formulations for the flat Tailguard sourcing model.

Every function in this module chooses one supplier option per component.  The
nominal formulation is the reference implementation; the weighted,
decomposed, spectral, and Wasserstein variants are all evaluated against the
same empirical risk primitives in :mod:`tailguard.risk`.
"""

from __future__ import annotations

import inspect
import math
import warnings
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pulp

from .bom import BOM, SupplierOption, flatten_bom
from .risk import (
    RiskValidationError,
    normalized_weights,
    validate_alpha,
)
from .risk import (
    rockafellar_uryasev_value as _risk_rockafellar_uryasev_value,
)
from .risk import (
    weighted_cvar as _risk_weighted_cvar,
)
from .risk import (
    weighted_mean as _risk_weighted_mean,
)


class OptimizationError(RuntimeError):
    """Raised when a model is invalid, infeasible, or not solved optimally."""


_MIN_SOLVER_ALPHA = 1e-8
_MIN_SOLVER_TAIL_MASS = 1e-8
_SOLVER_RELATIVE_TOLERANCE_FLOOR = 1e-7


@dataclass(frozen=True)
class OptimizationResult:
    """A solver result with enough information to audit a reported policy."""

    selected: Tuple[SupplierOption, ...]
    losses: np.ndarray
    mean_cost: float
    risk_cost: float
    objective: float
    status: str
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def labelled_domestic_share(self) -> float:
        """Share carrying the literal input label ``domestic``.

        This is a category count, not a geographic or policy claim. Users who
        need a geographic domestic-share metric must supply a home market and
        model that classification outside this flat prototype.
        """

        if not self.selected:
            return 0.0
        return sum(option.source_type == "domestic" for option in self.selected) / len(self.selected)


@dataclass(frozen=True)
class _ConditionedCosts:
    """A lower-dynamic-range representation supplied to the MILP solver.

    In exact arithmetic, every feasible policy loss is ``baseline_losses +
    conditioned policy loss``. Both terms are non-decreasing affine functions
    of the same scalar shock count, so the supported risk measures are additive
    across them. Floating subtraction can itself round, however, so this
    representation improves solver conditioning but is never the final
    optimality certificate; :func:`_componentwise_certified_selection` is.
    """

    costs: np.ndarray
    exposures: np.ndarray
    baseline_losses: np.ndarray
    baseline_exposure: Optional[float]


def _validate_nonnegative(value: float, name: str) -> float:
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
        raise OptimizationError(f"{name} must be a finite non-negative number")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = float(value)
    except (TypeError, ValueError, OverflowError, Warning) as exc:
        raise OptimizationError(f"{name} must be a finite non-negative number") from exc
    if not math.isfinite(result) or result < 0 or value < 0:
        raise OptimizationError(f"{name} must be a finite non-negative number")
    if result == 0.0 and value != 0:
        raise OptimizationError(
            f"{name} is non-zero but too small for floating-point representation"
        )
    return result


def _validate_solver_alpha(alpha: float) -> float:
    """Validate an alpha that will appear in floating-point MILP coefficients."""

    try:
        alpha = validate_alpha(alpha)
    except RiskValidationError as exc:
        raise OptimizationError(str(exc)) from exc
    if alpha < _MIN_SOLVER_ALPHA:
        raise OptimizationError(
            "alpha is too close to 0 for stable floating-point MILP dual weights; "
            f"alpha must be at least {_MIN_SOLVER_ALPHA:g}"
        )
    if 1.0 - alpha < _MIN_SOLVER_TAIL_MASS:
        raise OptimizationError(
            "alpha is too close to 1 for a stable floating-point MILP; "
            f"1 - alpha must be at least {_MIN_SOLVER_TAIL_MASS:g}"
        )
    return alpha


def _solver_tolerance(scale: float, absolute_tolerance: float = 1e-6, relative_tolerance: float = 1e-8) -> float:
    """Return a conservative tolerance for values parsed from a MILP solver."""

    return max(
        absolute_tolerance,
        relative_tolerance * max(1.0, scale),
        _SOLVER_RELATIVE_TOLERANCE_FLOOR * max(1.0, scale),
    )


def _reported_benders_bounds(
    baseline_objective: float,
    conditioned_lower_bound: float,
    upper_bound: float,
) -> tuple[float, float, float, float]:
    """Return internally consistent full-objective Benders diagnostics.

    Benders solves a conditioned objective, but callers see bounds for the
    original objective.  Add the baseline using the exact ratios of the two
    reported floats, clamp solver round-off to the independently certified
    upper bound, and derive both gaps from the floats actually returned.
    """

    if not all(
        math.isfinite(value)
        for value in (baseline_objective, conditioned_lower_bound, upper_bound)
    ):
        raise OptimizationError("Benders produced non-finite bound diagnostics")
    exact_lower_bound = Fraction.from_float(
        baseline_objective
    ) + Fraction.from_float(conditioned_lower_bound)
    try:
        reported_lower_bound = float(exact_lower_bound)
    except OverflowError:
        # A positive overflow is necessarily above the finite certified upper
        # bound and is clamped below. Negative overflow is not expected for
        # this non-negative-cost model, but retain a finite conservative bound.
        reported_lower_bound = (
            upper_bound if exact_lower_bound >= 0 else -np.finfo(float).max
        )
    reported_lower_bound = min(reported_lower_bound, upper_bound)
    gap = upper_bound - reported_lower_bound
    relative_gap = gap / max(1.0, abs(upper_bound))
    return reported_lower_bound, upper_bound, gap, relative_gap


def _objectives_match(left: float, right: float) -> bool:
    return abs(left - right) <= _solver_tolerance(max(abs(left), abs(right)))


def _event_array(event_counts: Iterable[float]) -> np.ndarray:
    if isinstance(event_counts, (str, bytes, bytearray, memoryview)):
        raise OptimizationError("event_counts must be a numeric iterable, not text or bytes-like data")
    if isinstance(event_counts, (Mapping, AbstractSet)) or getattr(event_counts, "ndim", 1) != 1:
        raise OptimizationError(
            "event_counts must be an ordered one-dimensional numeric iterable"
        )
    try:
        raw_values = list(event_counts)
    except TypeError as exc:
        raise OptimizationError("event_counts must be a numeric iterable") from exc
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
        raise OptimizationError(
            "event_counts must contain numeric scalar values, not boolean, complex, text, or bytes-like values"
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            converted_values = [float(value) for value in raw_values]
    except (TypeError, ValueError, OverflowError, Warning) as exc:
        raise OptimizationError("event_counts must be finite non-negative numbers") from exc
    if any(
        converted == 0.0 and value != 0
        for value, converted in zip(raw_values, converted_values)
    ):
        raise OptimizationError(
            "event_counts contain a non-zero value too small for floating-point representation"
        )
    values = np.asarray(converted_values, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise OptimizationError("event_counts must be a non-empty one-dimensional array")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise OptimizationError("event_counts must be finite and non-negative")
    return values


def _positive_probability_scenarios(
    events: np.ndarray, probabilities: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the probability support used by solver coefficients and bounds."""

    support = probabilities > 0
    supported_probabilities = normalized_weights(probabilities[support], int(np.count_nonzero(support)))
    return events[support], supported_probabilities


def _normalized_weights_or_error(
    weights: Optional[Iterable[float]], size: int, name: str = "scenario weights"
) -> np.ndarray:
    """Translate the public risk-layer validation error at an optimizer boundary."""

    try:
        return normalized_weights(weights, size)
    except RiskValidationError as exc:
        raise OptimizationError(f"invalid {name}: {exc}") from exc


def weighted_mean(
    losses: Iterable[float], weights: Optional[Iterable[float]] = None
) -> float:
    """Translate risk-result representation failures at optimizer boundaries."""

    try:
        return _risk_weighted_mean(losses, weights)
    except RiskValidationError as exc:
        raise OptimizationError(f"mean-risk evaluation failed: {exc}") from exc


def weighted_cvar(
    losses: Iterable[float], alpha: float, weights: Optional[Iterable[float]] = None
) -> float:
    """Translate risk-result representation failures at optimizer boundaries."""

    try:
        return _risk_weighted_cvar(losses, alpha, weights)
    except RiskValidationError as exc:
        raise OptimizationError(f"mean-risk evaluation failed: {exc}") from exc


def rockafellar_uryasev_value(
    losses: Iterable[float], alpha: float, weights: Optional[Iterable[float]] = None
) -> tuple[float, float]:
    """Translate reference-risk failures at optimizer boundaries."""

    try:
        return _risk_rockafellar_uryasev_value(losses, alpha, weights)
    except RiskValidationError as exc:
        raise OptimizationError(f"mean-risk evaluation failed: {exc}") from exc


def _solver() -> pulp.LpSolver:
    available = set(pulp.listSolvers(onlyAvailable=True))
    # Prefer native command/API integrations over PuLP's bundled CBC. Some
    # macOS wheels bundle an Intel CBC executable that cannot start on Apple
    # Silicon without Rosetta, while Homebrew HiGHS/CBC are native.
    for solver_name in ("HiGHS_CMD", "HiGHS", "COIN_CMD", "PULP_CBC_CMD"):
        if solver_name in available:
            options: dict[str, object] = {"msg": False}
            # PuLP 3.3.2 added an opt-in compatibility flag while deprecating
            # direct use of its bundled CBC wrapper. Without the flag, merely
            # constructing the solver emits a warning, which correctly becomes
            # an error in Tailguard's warning-clean verification runs. Older
            # supported PuLP releases do not accept the keyword.
            if (
                solver_name == "PULP_CBC_CMD"
                and "_skip_v4_deprecation"
                in inspect.signature(pulp.PULP_CBC_CMD).parameters
            ):
                options["_skip_v4_deprecation"] = True
            return pulp.getSolver(solver_name, **options)
    raise OptimizationError("no supported MILP solver is available (install CBC or HiGHS)")


def _solve_or_raise(problem: pulp.LpProblem) -> str:
    try:
        problem.solve(_solver())
    except (pulp.PulpSolverError, OSError):
        raise OptimizationError(
            "the MILP solver could not be started or completed; install a compatible native CBC or HiGHS solver"
        ) from None
    status = pulp.LpStatus.get(problem.status, str(problem.status))
    if status != "Optimal":
        raise OptimizationError(f"MILP did not solve to optimality: {status}")
    return status


def _solved_objective_value(problem: pulp.LpProblem) -> float:
    """Return a finite objective, including PuLP's constant-zero edge case."""

    objective = problem.objective
    if objective is None:  # pragma: no cover - every model in this module sets one.
        raise OptimizationError("solver model has no objective")
    value = pulp.value(objective)
    if value is None:
        # PuLP represents a constant-zero objective as ``0*__dummy + 0``.
        # The dummy variable has no solution value, so ``pulp.value`` returns
        # None even after an optimal solve. Ignore only genuinely zero terms;
        # a missing value for any real objective term remains an error.
        if any(float(coefficient) != 0.0 for coefficient in objective.values()):
            raise OptimizationError("solver returned no value for the objective")
        value = objective.constant
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:  # pragma: no cover - defensive solver boundary.
        raise OptimizationError("solver returned an invalid objective value") from exc
    if not math.isfinite(result):
        raise OptimizationError("solver returned a non-finite objective value")
    return result


def _lp_variable(
    problem: pulp.LpProblem,
    name: str,
    low_bound: Optional[float] = None,
    up_bound: Optional[float] = None,
    category: str = "Continuous",
) -> pulp.LpVariable:
    """Create a variable across PuLP 2.x and 3.x without noisy deprecations."""

    if hasattr(problem, "add_variable"):
        return problem.add_variable(name, lowBound=low_bound, upBound=up_bound, cat=category)
    return pulp.LpVariable(name, lowBound=low_bound, upBound=up_bound, cat=category)


def _eligible_indices(options: Sequence[SupplierOption], max_lead_time_days: Optional[float]) -> dict[str, list[int]]:
    if max_lead_time_days is not None:
        max_lead_time_days = _validate_nonnegative(max_lead_time_days, "max_lead_time_days")
    by_component: dict[str, list[int]] = {}
    for index, option in enumerate(options):
        if max_lead_time_days is None or option.lead_time <= max_lead_time_days:
            by_component.setdefault(option.component, []).append(index)
        else:
            by_component.setdefault(option.component, [])
    missing = [component for component, indices in by_component.items() if not indices]
    if missing:
        raise OptimizationError("lead-time limit leaves no feasible supplier for: " + ", ".join(sorted(missing)))
    return by_component


def _condition_costs(
    options: Sequence[SupplierOption],
    event_counts: np.ndarray,
    lead_time_cost_per_day: float,
    eligible: Mapping[str, Sequence[int]],
) -> _ConditionedCosts:
    """Remove policy-independent base and exposure floors before solving.

    Large common monetary offsets can make a one-unit policy improvement
    invisible to a command-line MILP solver's relative tolerances.  For each
    component we subtract the smallest eligible deterministic cost and shock
    exposure from every eligible option. In real arithmetic the removed loss
    is policy-independent and comonotone with every non-negative residual loss.
    The exact rational certificate still determines the returned policy because
    subtraction in this solver-only float representation can lose a tiny gap.
    """

    eligible_indices = sorted(index for indices in eligible.values() for index in indices)
    exact_rate = Fraction.from_float(float(lead_time_cost_per_day))
    exact_deterministic = [Fraction(0) for _ in options]
    exact_exposures = [Fraction(0) for _ in options]
    for index in eligible_indices:
        option = options[index]
        exact_deterministic[index] = Fraction.from_float(
            float(option.base_cost)
        ) + exact_rate * Fraction.from_float(float(option.lead_time))
        exact_exposures[index] = Fraction.from_float(float(option.kappa))

    conditioned_base = np.zeros(len(options), dtype=float)
    conditioned_exposure = np.zeros(len(options), dtype=float)
    exact_conditioned_base = [Fraction(0) for _ in options]
    exact_conditioned_exposure = [Fraction(0) for _ in options]
    base_floors: list[Fraction] = []
    exposure_floors: list[Fraction] = []

    for indices in eligible.values():
        base_floor = min(exact_deterministic[index] for index in indices)
        exposure_floor = min(exact_exposures[index] for index in indices)
        base_floors.append(base_floor)
        exposure_floors.append(exposure_floor)
        for index in indices:
            exact_conditioned_base[index] = exact_deterministic[index] - base_floor
            exact_conditioned_exposure[index] = exact_exposures[index] - exposure_floor
            conditioned_base[index] = _finite_fraction(
                exact_conditioned_base[index],
                "conditioned deterministic-cost coefficient is outside floating-point range",
            )
            conditioned_exposure[index] = _finite_fraction(
                exact_conditioned_exposure[index],
                "conditioned exposure coefficient is outside floating-point range",
            )

    exact_baseline_deterministic = sum(base_floors, Fraction(0))
    exact_baseline_exposure = sum(exposure_floors, Fraction(0))
    try:
        converted_baseline_exposure = float(exact_baseline_exposure)
    except OverflowError:
        baseline_exposure = None
    else:
        # The aggregate exposure can exceed one float while every scenario loss
        # remains finite (for example, enormous exposures multiplied by zero or
        # a very small shock count). Nominal models do not need that aggregate.
        baseline_exposure = (
            converted_baseline_exposure
            if math.isfinite(converted_baseline_exposure)
            else None
        )

    exact_events = tuple(
        Fraction.from_float(float(event_count)) for event_count in event_counts
    )
    conditioned_costs = np.zeros((len(options), len(event_counts)), dtype=float)
    for index in eligible_indices:
        for scenario, exact_event in enumerate(exact_events):
            conditioned_costs[index, scenario] = _finite_fraction(
                exact_conditioned_base[index]
                + exact_conditioned_exposure[index] * exact_event,
                "conditioned solver loss coefficient is outside floating-point range",
            )
    baseline_losses = np.asarray(
        [
            _finite_fraction(
                exact_baseline_deterministic
                + exact_baseline_exposure * exact_event,
                "conditioned baseline loss is outside floating-point range",
            )
            for exact_event in exact_events
        ],
        dtype=float,
    )
    return _ConditionedCosts(
        costs=conditioned_costs,
        exposures=conditioned_exposure,
        baseline_losses=baseline_losses,
        baseline_exposure=baseline_exposure,
    )


def _baseline_mean_cvar_objective(
    conditioning: _ConditionedCosts,
    alpha: float,
    risk_weight: float,
    probabilities: np.ndarray,
) -> float:
    return _finite_objective(
        weighted_mean(conditioning.baseline_losses, probabilities),
        weighted_cvar(conditioning.baseline_losses, alpha, probabilities),
        risk_weight,
    )


def _finite_objective(mean_cost: float, risk_cost: float, risk_weight: float) -> float:
    """Combine finite reported values without intermediate overflow or underflow."""

    exact = Fraction.from_float(float(mean_cost)) + Fraction.from_float(
        float(risk_weight)
    ) * Fraction.from_float(float(risk_cost))
    try:
        objective = float(exact)
    except OverflowError as exc:
        raise OptimizationError(
            "selected policy objective overflows or underflows floating-point arithmetic; rescale monetary inputs"
        ) from exc
    if not math.isfinite(objective) or (objective == 0.0 and exact != 0):
        raise OptimizationError(
            "selected policy objective overflows or underflows floating-point arithmetic; rescale monetary inputs"
        )
    return objective


def _finite_fraction(value: Fraction, message: str) -> float:
    """Convert an exact diagnostic or risk term only when float can report it."""

    try:
        converted = float(value)
    except OverflowError as exc:
        raise OptimizationError(message) from exc
    if not math.isfinite(converted) or (converted == 0.0 and value != 0):
        raise OptimizationError(message)
    return converted


def _solver_cost_scale(costs: np.ndarray, eligible: Mapping[str, Sequence[int]]) -> float:
    """Scale every eligible MILP loss coefficient into the interval ``[0, 1]``."""

    indices = np.asarray(
        sorted(index for component in eligible.values() for index in component), dtype=int
    )
    maximum = float(np.max(np.abs(costs[indices])))
    if not math.isfinite(maximum):  # pragma: no cover - callers validate conditioned costs.
        raise OptimizationError("solver cost scale is not finite")
    return max(1.0, maximum)


def _fraction_probabilities(probabilities: np.ndarray) -> tuple[Fraction, ...]:
    values = [Fraction.from_float(float(probability)) for probability in probabilities]
    total = sum(values, Fraction(0))
    return tuple(value / total for value in values)


def _exact_scalar_mean_and_cvars(
    event_counts: np.ndarray,
    alphas: Sequence[float],
    probabilities: np.ndarray,
) -> tuple[Fraction, tuple[Fraction, ...]]:
    """Evaluate scalar-shock moments without discarding sub-ULP policy gaps."""

    events = tuple(Fraction.from_float(float(value)) for value in event_counts)
    exact_probabilities = _fraction_probabilities(probabilities)
    mean = sum(
        (probability * value for probability, value in zip(exact_probabilities, events)),
        Fraction(0),
    )
    order = sorted(range(len(events)), key=lambda index: (-events[index], index))
    cvars: list[Fraction] = []
    for alpha in alphas:
        tail_mass = Fraction(1) - Fraction.from_float(float(alpha))
        remaining = tail_mass
        tail_total = Fraction(0)
        for index in order:
            if remaining <= 0:
                break
            allocated = min(exact_probabilities[index], remaining)
            tail_total += allocated * events[index]
            remaining -= allocated
        if remaining > 0:  # pragma: no cover - exact normalized probabilities cover one.
            raise OptimizationError("scenario probabilities do not cover the requested tail mass")
        cvars.append(tail_total / tail_mass)
    return mean, tuple(cvars)


def _exact_cvar_mixture(
    losses: np.ndarray,
    alphas: Sequence[float],
    mixture_weights: np.ndarray,
    probabilities: np.ndarray,
) -> Fraction:
    """Combine empirical CVaRs exactly before one final float conversion."""

    _, cvars = _exact_scalar_mean_and_cvars(losses, alphas, probabilities)
    mixture = _fraction_probabilities(mixture_weights)
    return sum(
        (weight * cvar for weight, cvar in zip(mixture, cvars)),
        Fraction(0),
    )


def _exact_wasserstein_risk(
    losses: np.ndarray,
    alpha: float,
    probabilities: np.ndarray,
    exact_increment: Fraction,
) -> Fraction:
    """Add a W1 increment to empirical CVaR without an intermediate rounding."""

    _, (nominal_cvar,) = _exact_scalar_mean_and_cvars(
        losses, (alpha,), probabilities
    )
    return nominal_cvar + exact_increment


def _exact_option_scores(
    options: Sequence[SupplierOption],
    lead_time_cost_per_day: float,
    deterministic_coefficient: Fraction,
    exposure_coefficient: Fraction,
) -> tuple[Fraction, ...]:
    rate = Fraction.from_float(float(lead_time_cost_per_day))
    scores = []
    for option in options:
        deterministic = Fraction.from_float(float(option.base_cost)) + rate * Fraction.from_float(
            float(option.lead_time)
        )
        exposure = Fraction.from_float(float(option.kappa))
        scores.append(deterministic_coefficient * deterministic + exposure_coefficient * exposure)
    return tuple(scores)


def _exact_mean_cvar_option_scores(
    options: Sequence[SupplierOption],
    event_counts: np.ndarray,
    alpha: float,
    risk_weight: float,
    probabilities: np.ndarray,
    lead_time_cost_per_day: float,
) -> tuple[Fraction, ...]:
    mean, (cvar,) = _exact_scalar_mean_and_cvars(event_counts, (alpha,), probabilities)
    risk = Fraction.from_float(float(risk_weight))
    return _exact_option_scores(
        options,
        lead_time_cost_per_day,
        Fraction(1) + risk,
        mean + risk * cvar,
    )


def _exact_spectral_option_scores(
    options: Sequence[SupplierOption],
    event_counts: np.ndarray,
    alphas: Sequence[float],
    mixture_weights: np.ndarray,
    risk_weight: float,
    probabilities: np.ndarray,
    lead_time_cost_per_day: float,
) -> tuple[Fraction, ...]:
    mean, cvars = _exact_scalar_mean_and_cvars(event_counts, alphas, probabilities)
    mixture = _fraction_probabilities(mixture_weights)
    spectral = sum((weight * cvar for weight, cvar in zip(mixture, cvars)), Fraction(0))
    risk = Fraction.from_float(float(risk_weight))
    return _exact_option_scores(
        options,
        lead_time_cost_per_day,
        Fraction(1) + risk,
        mean + risk * spectral,
    )


def _exact_wasserstein_option_scores(
    options: Sequence[SupplierOption],
    event_counts: np.ndarray,
    alpha: float,
    epsilon: float,
    risk_weight: float,
    probabilities: np.ndarray,
    lead_time_cost_per_day: float,
) -> tuple[Fraction, ...]:
    mean, (cvar,) = _exact_scalar_mean_and_cvars(event_counts, (alpha,), probabilities)
    exact_alpha = Fraction.from_float(float(alpha))
    risk = Fraction.from_float(float(risk_weight))
    robust_slope = Fraction.from_float(float(epsilon)) / (Fraction(1) - exact_alpha)
    return _exact_option_scores(
        options,
        lead_time_cost_per_day,
        Fraction(1) + risk,
        mean + risk * (cvar + robust_slope),
    )


def _exact_wasserstein_reporting_option_scores(
    options: Sequence[SupplierOption],
    event_counts: np.ndarray,
    alpha: float,
    epsilon: float,
    lead_time_cost_per_day: float,
) -> tuple[Fraction, ...]:
    """Break W1 objective ties by the smallest reportable robust loss."""

    maximum_event = Fraction.from_float(float(np.max(event_counts)))
    robust_slope = Fraction.from_float(float(epsilon)) / (
        Fraction(1) - Fraction.from_float(float(alpha))
    )
    return _exact_option_scores(
        options,
        lead_time_cost_per_day,
        Fraction(1),
        maximum_event + robust_slope,
    )


def _exact_reporting_option_scores(
    options: Sequence[SupplierOption],
    event_counts: np.ndarray,
    lead_time_cost_per_day: float,
) -> tuple[Fraction, ...]:
    """Rank objective ties by their maximum full-scenario reporting loss."""

    maximum_event = Fraction.from_float(float(np.max(event_counts)))
    return _exact_option_scores(
        options,
        lead_time_cost_per_day,
        Fraction(1),
        maximum_event,
    )


def _selected_exposure_diagnostics(
    selected: Sequence[SupplierOption], alpha: float
) -> tuple[Fraction, Optional[float], Optional[float]]:
    """Return exact exposure plus finite display values when representable."""

    exact_exposure = sum(
        (Fraction.from_float(float(option.kappa)) for option in selected),
        Fraction(0),
    )
    exact_tau = exact_exposure / (Fraction(1) - Fraction.from_float(float(alpha)))

    def finite_float(value: Fraction) -> Optional[float]:
        try:
            converted = float(value)
        except OverflowError:
            return None
        return converted if math.isfinite(converted) else None

    return exact_exposure, finite_float(exact_exposure), finite_float(exact_tau)


def _canonical_exact_selection(
    eligible: Mapping[str, Sequence[int]],
    exact_option_scores: Sequence[Fraction],
    reporting_option_scores: Sequence[Fraction],
) -> tuple[list[int], Fraction]:
    """Return the deterministic exact optimum for this separable flat model."""

    certified: list[int] = []
    for indices in eligible.values():
        best_score = min(exact_option_scores[index] for index in indices)
        tied = [index for index in indices if exact_option_scores[index] == best_score]
        minimum_reporting_score = min(reporting_option_scores[index] for index in tied)
        canonical = next(
            index for index in tied if reporting_option_scores[index] == minimum_reporting_score
        )
        certified.append(canonical)
    certified.sort()
    exact_objective = sum((exact_option_scores[index] for index in certified), Fraction(0))
    return certified, exact_objective


def _componentwise_certified_selection(
    solver_selected_indices: Sequence[int],
    eligible: Mapping[str, Sequence[int]],
    exact_option_scores: Sequence[Fraction],
    reporting_option_scores: Sequence[Fraction],
) -> tuple[list[int], bool, Fraction]:
    """Certify a solver policy against the exact global optimum.

    With one option per component and a shared non-negative scalar shock count,
    every implemented objective is the sum of per-option scores. If solver
    tolerances hide a strict difference, replace only that component with its
    exact rational optimum instead of reporting a false ``Optimal`` policy.
    """

    selected_set = set(solver_selected_indices)
    eligible_set = {index for indices in eligible.values() for index in indices}
    if not selected_set.issubset(eligible_set) or len(selected_set) != len(solver_selected_indices):
        raise OptimizationError("solver selected an invalid supplier policy")

    for indices in eligible.values():
        if len([index for index in indices if index in selected_set]) != 1:
            raise OptimizationError("solver did not select exactly one eligible option per component")
    certified, exact_objective = _canonical_exact_selection(
        eligible, exact_option_scores, reporting_option_scores
    )
    replaced = set(certified) != selected_set
    return certified, replaced, exact_objective


def _make_decision_variables(
    problem: pulp.LpProblem,
    options: Sequence[SupplierOption],
    eligible: Mapping[str, Sequence[int]],
) -> dict[int, pulp.LpVariable]:
    variables = {
        index: _lp_variable(problem, f"x_{index}", low_bound=0, up_bound=1, category="Binary")
        for index in range(len(options))
    }
    eligible_set = {index for indices in eligible.values() for index in indices}
    for index in range(len(options)):
        if index not in eligible_set:
            problem += variables[index] == 0, f"ineligible_{index}"
    # Components are user input and can contain characters PuLP normalizes in
    # constraint names (for example ``A-B`` and ``A_B``).  Use the stable
    # insertion index instead of exposing user text to the solver namespace.
    for component_index, indices in enumerate(eligible.values()):
        problem += pulp.lpSum(variables[index] for index in indices) == 1, f"choose_{component_index}"
    return variables


def _selected_indices(variables: Mapping[int, pulp.LpVariable]) -> list[int]:
    values: list[int] = []
    for index, variable in variables.items():
        value = variable.value()
        if value is None:
            raise OptimizationError("solver returned no value for a decision variable")
        value = float(value)
        if not math.isfinite(value) or min(abs(value), abs(value - 1.0)) > 1e-6:
            raise OptimizationError("solver returned a non-binary value for a supplier decision")
        if value > 0.5:
            values.append(index)
    return values


def _policy_losses(costs: np.ndarray, selected_indices: Sequence[int]) -> np.ndarray:
    """Sum selected option costs while rejecting floating-point overflow."""

    with np.errstate(over="ignore", invalid="ignore"):
        losses = costs[np.asarray(selected_indices, dtype=int)].sum(axis=0)
    if not np.isfinite(losses).all():
        raise OptimizationError("selected policy produces non-finite scenario losses; rescale monetary inputs")
    return losses


def _selected_policy_losses(
    options: Sequence[SupplierOption],
    selected_indices: Sequence[int],
    event_counts: np.ndarray,
    lead_time_cost_per_day: float,
) -> np.ndarray:
    """Evaluate only a certified policy across the caller's full scenario vector."""

    if not selected_indices:
        raise OptimizationError("solver selected no supplier options")
    selected = [options[index] for index in selected_indices]
    exact_rate = Fraction.from_float(float(lead_time_cost_per_day))
    exact_deterministic = sum(
        (
            Fraction.from_float(float(option.base_cost))
            + exact_rate * Fraction.from_float(float(option.lead_time))
            for option in selected
        ),
        Fraction(0),
    )
    exact_exposure = sum(
        (Fraction.from_float(float(option.kappa)) for option in selected),
        Fraction(0),
    )
    losses: list[float] = []
    for event_count in event_counts:
        exact_loss = exact_deterministic + exact_exposure * Fraction.from_float(
            float(event_count)
        )
        try:
            loss = float(exact_loss)
        except OverflowError:
            raise OptimizationError(
                "selected policy produces non-finite scenario losses or unrepresentable values; "
                "rescale monetary inputs"
            ) from None
        if not math.isfinite(loss) or (loss == 0.0 and exact_loss != 0):
            raise OptimizationError(
                "selected policy produces non-finite scenario losses or unrepresentable values; "
                "rescale monetary inputs"
            )
        losses.append(loss)
    return np.asarray(losses, dtype=float)


def _evaluate_selected_policy(
    options: Sequence[SupplierOption],
    event_counts: np.ndarray,
    selected_indices: Sequence[int],
    lead_time_cost_per_day: float,
    alpha: float,
    risk_weight: float,
    probabilities: np.ndarray,
) -> tuple[tuple[SupplierOption, ...], np.ndarray, float, float, float]:
    """Evaluate a certified selection on the original scenario vector."""

    selected = tuple(options[index] for index in selected_indices)
    losses = _selected_policy_losses(
        options, selected_indices, event_counts, lead_time_cost_per_day
    )
    mean_cost = weighted_mean(losses, probabilities)
    risk_cost = weighted_cvar(losses, alpha, probabilities)
    objective = _finite_objective(mean_cost, risk_cost, risk_weight)
    return selected, losses, mean_cost, risk_cost, objective


def _evaluate_indices(
    options: Sequence[SupplierOption],
    costs: np.ndarray,
    selected_indices: Sequence[int],
    alpha: float,
    risk_weight: float,
    probabilities: np.ndarray,
) -> tuple[tuple[SupplierOption, ...], np.ndarray, float, float, float]:
    selected = tuple(options[index] for index in selected_indices)
    if not selected:
        raise OptimizationError("solver selected no supplier options")
    losses = _policy_losses(costs, selected_indices)
    mean_cost = weighted_mean(losses, probabilities)
    risk_cost = weighted_cvar(losses, alpha, probabilities)
    objective = _finite_objective(mean_cost, risk_cost, risk_weight)
    return selected, losses, mean_cost, risk_cost, objective


def solve_mean_cvar_milp(
    bom: BOM,
    event_counts: Iterable[float],
    *,
    alpha: float = 0.95,
    risk_weight: float = 1.0,
    scenario_weights: Optional[Iterable[float]] = None,
    lead_time_cost_per_day: float = 0.0,
    max_lead_time_days: Optional[float] = None,
) -> OptimizationResult:
    """Solve the nominal (or scenario-weighted) mean-plus-CVaR MILP.

    ``scenario_weights`` enables a correctly weighted stratified estimator.
    ``lead_time_cost_per_day`` and ``max_lead_time_days`` make time effects an
    explicit user decision rather than a hidden assumption.
    """

    alpha = _validate_solver_alpha(alpha)
    risk_weight = _validate_nonnegative(risk_weight, "risk_weight")
    lead_time_cost_per_day = _validate_nonnegative(lead_time_cost_per_day, "lead_time_cost_per_day")
    events = _event_array(event_counts)
    probabilities = _normalized_weights_or_error(scenario_weights, len(events))
    model_events, model_probabilities = _positive_probability_scenarios(
        events, probabilities
    )
    options = flatten_bom(bom)
    eligible = _eligible_indices(options, max_lead_time_days)
    exact_option_scores = _exact_mean_cvar_option_scores(
        options,
        model_events,
        alpha,
        risk_weight,
        model_probabilities,
        lead_time_cost_per_day,
    )
    reporting_option_scores = _exact_reporting_option_scores(
        options, events, lead_time_cost_per_day
    )
    exact_selected_indices, exact_objective = _canonical_exact_selection(
        eligible, exact_option_scores, reporting_option_scores
    )

    def exact_fallback(
        reason: str, conditioning: Optional[_ConditionedCosts] = None
    ) -> OptimizationResult:
        selected, losses, mean_cost, risk_cost, objective = _evaluate_selected_policy(
            options,
            events,
            exact_selected_indices,
            lead_time_cost_per_day,
            alpha,
            risk_weight,
            probabilities,
        )
        _, reported_zeta = rockafellar_uryasev_value(losses, alpha, probabilities)
        conditioned_objective: Optional[float] = None
        baseline_objective: Optional[float] = None
        if conditioning is not None:
            _, _, _, _, conditioned_objective = _evaluate_indices(
                options,
                conditioning.costs,
                exact_selected_indices,
                alpha,
                risk_weight,
                model_probabilities,
            )
            baseline_objective = _baseline_mean_cvar_objective(
                conditioning, alpha, risk_weight, model_probabilities
            )
        return OptimizationResult(
            selected=selected,
            losses=losses,
            mean_cost=mean_cost,
            risk_cost=risk_cost,
            objective=objective,
            status="ExactOptimal",
            details={
                "method": "nominal_mean_cvar",
                "zeta": reported_zeta,
                "solver_conditioned_zeta": None,
                "solver_conditioned_objective": None,
                "conditioned_objective": conditioned_objective,
                "baseline_objective": baseline_objective,
                "componentwise_optimality_certified": True,
                "solver_policy_replaced_by_certificate": False,
                "solver_bypassed_for_exact_certificate": True,
                "solver_fallback_reason": reason,
                "exact_certificate_objective": str(exact_objective),
                "scenario_weights": probabilities.copy(),
                "lead_time_cost_per_day": lead_time_cost_per_day,
                "max_lead_time_days": max_lead_time_days,
            },
        )

    try:
        conditioning = _condition_costs(
            options, model_events, lead_time_cost_per_day, eligible
        )
    except OptimizationError:
        return exact_fallback("conditioned_solver_coefficients_not_representable")
    model_costs = conditioning.costs
    cost_scale = _solver_cost_scale(model_costs, eligible)
    solver_costs = model_costs / cost_scale
    objective_scale = max(1.0, risk_weight)
    mean_factor = 1.0 / objective_scale
    risk_factor = risk_weight / objective_scale

    problem = pulp.LpProblem("tailguard_mean_cvar", pulp.LpMinimize)
    x = _make_decision_variables(problem, options, eligible)
    zeta = _lp_variable(problem, "zeta")
    slacks = {
        scenario: _lp_variable(problem, f"slack_{scenario}", low_bound=0)
        for scenario in range(len(model_events))
    }
    scenario_losses = {
        scenario: pulp.lpSum(
            x[index] * float(solver_costs[index, scenario]) for index in range(len(options))
        )
        for scenario in range(len(model_events))
    }
    for scenario, loss in scenario_losses.items():
        problem += slacks[scenario] >= loss - zeta, f"tail_{scenario}"

    mean_expression = pulp.lpSum(
        x[index] * weighted_mean(solver_costs[index], model_probabilities)
        for index in range(len(options))
    )
    cvar_expression = zeta + pulp.lpSum(
        float(model_probabilities[scenario]) * slacks[scenario]
        for scenario in range(len(model_events))
    ) / ((1.0 - alpha) * math.fsum(float(value) for value in model_probabilities))
    problem += mean_factor * mean_expression + risk_factor * cvar_expression

    try:
        status = _solve_or_raise(problem)
        solver_selected_indices = _selected_indices(x)
        solver_scaled_losses = _policy_losses(solver_costs, solver_selected_indices)
        solver_scaled_mean = weighted_mean(solver_scaled_losses, model_probabilities)
        solver_scaled_risk = weighted_cvar(
            solver_scaled_losses, alpha, model_probabilities
        )
        _, _, _, _, solver_conditioned_objective = _evaluate_indices(
            options,
            model_costs,
            solver_selected_indices,
            alpha,
            risk_weight,
            model_probabilities,
        )
        solver_normalized_objective = math.fsum(
            (mean_factor * solver_scaled_mean, risk_factor * solver_scaled_risk)
        )
        model_objective = _solved_objective_value(problem)
        if not _objectives_match(model_objective, solver_normalized_objective):
            raise OptimizationError(
                "solver objective does not match independent normalized mean-CVaR evaluation"
            )
        selected_indices, solver_policy_replaced, certified_objective = (
            _componentwise_certified_selection(
                solver_selected_indices,
                eligible,
                exact_option_scores,
                reporting_option_scores,
            )
        )
        if certified_objective != exact_objective:  # pragma: no cover - same certificate path.
            raise OptimizationError("exact optimality certificates disagree")
    except OptimizationError as exc:
        return exact_fallback(str(exc), conditioning)

    selected, losses, mean_cost, risk_cost, objective = _evaluate_selected_policy(
        options,
        events,
        selected_indices,
        lead_time_cost_per_day,
        alpha,
        risk_weight,
        probabilities,
    )
    _, _, _, _, conditioned_objective = _evaluate_indices(
        options, model_costs, selected_indices, alpha, risk_weight, model_probabilities
    )
    _, reported_zeta = rockafellar_uryasev_value(losses, alpha, probabilities)
    zeta_value = zeta.value()
    solver_conditioned_zeta: Optional[float] = None
    if zeta_value is not None:
        with np.errstate(over="ignore", invalid="ignore"):
            candidate_zeta = float(zeta_value) * cost_scale
        if math.isfinite(candidate_zeta):
            solver_conditioned_zeta = candidate_zeta
    return OptimizationResult(
        selected=selected,
        losses=losses,
        mean_cost=mean_cost,
        risk_cost=risk_cost,
        objective=objective,
        status=status,
        details={
            "method": "nominal_mean_cvar",
            "zeta": reported_zeta,
            "solver_conditioned_zeta": solver_conditioned_zeta,
            "solver_conditioned_objective": solver_conditioned_objective,
            "solver_normalized_objective": solver_normalized_objective,
            "solver_cost_scale": cost_scale,
            "solver_objective_scale": objective_scale,
            "conditioned_objective": conditioned_objective,
            "baseline_objective": _baseline_mean_cvar_objective(
                conditioning, alpha, risk_weight, model_probabilities
            ),
            "componentwise_optimality_certified": True,
            "solver_policy_replaced_by_certificate": solver_policy_replaced,
            "exact_certificate_objective": str(exact_objective),
            "scenario_weights": probabilities.copy(),
            "lead_time_cost_per_day": lead_time_cost_per_day,
            "max_lead_time_days": max_lead_time_days,
        },
    )


def solve_benders_mean_cvar(
    bom: BOM,
    event_counts: Iterable[float],
    *,
    alpha: float = 0.95,
    risk_weight: float = 1.0,
    scenario_weights: Optional[Iterable[float]] = None,
    lead_time_cost_per_day: float = 0.0,
    max_lead_time_days: Optional[float] = None,
    max_iterations: int = 200,
    relative_tolerance: float = 1e-8,
    absolute_tolerance: float = 1e-6,
) -> OptimizationResult:
    """Solve mean-plus-CVaR using valid outer-approximation/Benders cuts.

    In this flat scalar-shock model all eligible losses are comonotone affine
    functions of the same event count. The cut can therefore use each option's
    independently evaluated empirical CVaR as a globally valid linear risk
    coefficient, including fractional probability mass at VaR.
    """

    alpha = _validate_solver_alpha(alpha)
    risk_weight = _validate_nonnegative(risk_weight, "risk_weight")
    lead_time_cost_per_day = _validate_nonnegative(lead_time_cost_per_day, "lead_time_cost_per_day")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
        raise OptimizationError("max_iterations must be a positive integer")
    relative_tolerance = _validate_nonnegative(relative_tolerance, "relative_tolerance")
    absolute_tolerance = _validate_nonnegative(absolute_tolerance, "absolute_tolerance")

    events = _event_array(event_counts)
    probabilities = _normalized_weights_or_error(scenario_weights, len(events))
    if risk_weight == 0:
        nominal = solve_mean_cvar_milp(
            bom,
            events,
            alpha=alpha,
            risk_weight=0.0,
            scenario_weights=probabilities,
            lead_time_cost_per_day=lead_time_cost_per_day,
            max_lead_time_days=max_lead_time_days,
        )
        return OptimizationResult(
            selected=nominal.selected,
            losses=nominal.losses,
            mean_cost=nominal.mean_cost,
            risk_cost=nominal.risk_cost,
            objective=nominal.objective,
            status=nominal.status,
            details={
                "method": "benders_mean_cvar",
                "iterations": 0,
                "lower_bound": nominal.objective,
                "upper_bound": nominal.objective,
                "gap": 0.0,
                "relative_gap": 0.0,
                "cuts": 0,
                "converged": True,
                "bypassed_for_zero_risk_weight": True,
                "componentwise_optimality_certified": nominal.details[
                    "componentwise_optimality_certified"
                ],
                "solver_policy_replaced_by_certificate": nominal.details[
                    "solver_policy_replaced_by_certificate"
                ],
                "scenario_weights": probabilities.copy(),
            },
        )
    reference = solve_mean_cvar_milp(
        bom,
        events,
        alpha=alpha,
        risk_weight=risk_weight,
        scenario_weights=probabilities,
        lead_time_cost_per_day=lead_time_cost_per_day,
        max_lead_time_days=max_lead_time_days,
    )

    def nominal_fallback(reason: str, iterations: int = 0, cut_count: int = 0) -> OptimizationResult:
        return OptimizationResult(
            selected=reference.selected,
            losses=reference.losses,
            mean_cost=reference.mean_cost,
            risk_cost=reference.risk_cost,
            objective=reference.objective,
            status=reference.status,
            details={
                "method": "benders_mean_cvar",
                "iterations": iterations,
                "lower_bound": reference.objective,
                "upper_bound": reference.objective,
                "gap": 0.0,
                "relative_gap": 0.0,
                "cuts": cut_count,
                "converged": False,
                "fallback_to_nominal": True,
                "fallback_reason": reason,
                "reference_verified": True,
                "returned_nominal_reference": True,
                "componentwise_optimality_certified": reference.details[
                    "componentwise_optimality_certified"
                ],
                "solver_policy_replaced_by_certificate": reference.details[
                    "solver_policy_replaced_by_certificate"
                ],
                "scenario_weights": probabilities.copy(),
            },
        )

    if np.any((probabilities > 0) & (probabilities < np.finfo(float).tiny)):
        return nominal_fallback("subnormal_probability_dual_not_representable")
    options = flatten_bom(bom)
    model_events, model_probabilities = _positive_probability_scenarios(
        events, probabilities
    )
    eligible = _eligible_indices(options, max_lead_time_days)
    try:
        conditioning = _condition_costs(
            options, model_events, lead_time_cost_per_day, eligible
        )
    except OptimizationError:
        return nominal_fallback("conditioned_solver_coefficients_not_representable")
    costs = conditioning.costs
    baseline_objective = _baseline_mean_cvar_objective(
        conditioning, alpha, risk_weight, model_probabilities
    )
    mean_coefficients = np.asarray(
        [weighted_mean(costs[index], model_probabilities) for index in range(len(options))]
    )
    cuts: list[np.ndarray] = []
    incumbent: Optional[tuple[list[int], tuple[SupplierOption, ...], np.ndarray, float, float, float]] = None
    lower_bound = -math.inf
    effective_lower_bound = -math.inf
    gap = math.inf
    solver_tolerance = _solver_tolerance(1.0, absolute_tolerance, relative_tolerance)
    seen_policies: set[tuple[int, ...]] = set()
    completed_iterations = 0
    converged_by_benders = False
    termination_reason = "iteration_limit"

    for iteration in range(1, max_iterations + 1):
        completed_iterations = iteration
        master = pulp.LpProblem(f"tailguard_benders_{iteration}", pulp.LpMinimize)
        x = _make_decision_variables(master, options, eligible)
        theta = _lp_variable(master, "theta", low_bound=0)
        for cut_index, gradient in enumerate(cuts):
            master += (
                theta >= pulp.lpSum(float(gradient[index]) * x[index] for index in range(len(options))),
                f"cvar_cut_{cut_index}",
            )
        master += (
            pulp.lpSum(float(mean_coefficients[index]) * x[index] for index in range(len(options)))
            + risk_weight * theta
        )
        try:
            _solve_or_raise(master)
        except OptimizationError as exc:
            return nominal_fallback(str(exc), iteration, len(cuts))
        try:
            lower_bound = _solved_objective_value(master)
            selected_indices = _selected_indices(x)
            selected, losses, mean_cost, risk_cost, objective = _evaluate_indices(
                options,
                costs,
                selected_indices,
                alpha,
                risk_weight,
                model_probabilities,
            )
        except OptimizationError as exc:
            return nominal_fallback(str(exc), iteration, len(cuts))
        if incumbent is None or objective < incumbent[5]:
            incumbent = (selected_indices, selected, losses, mean_cost, risk_cost, objective)

        scale = max(1.0, abs(incumbent[5]), abs(lower_bound))
        # CBC/HiGHS can report a master objective a few units in the last
        # decimal place above a feasible recomputation.  Bound comparisons are
        # therefore made to an explicit solver-scale tolerance rather than
        # treating print-rounding as a mathematical invalidity.
        solver_tolerance = _solver_tolerance(scale, absolute_tolerance, relative_tolerance)
        if incumbent[5] + solver_tolerance < lower_bound:
            # Never return an uncertified decomposed result.  A monolithic
            # reference solve below is safe even if this came from solver I/O
            # rounding rather than a defective cut.
            termination_reason = "master_lower_bound_exceeded_feasible_objective"
            break
        effective_lower_bound = min(lower_bound, incumbent[5])
        gap = incumbent[5] - effective_lower_bound
        objective_tolerance = solver_tolerance

        master_theta = theta.value()
        if master_theta is None:
            raise OptimizationError("solver returned no value for Benders theta")
        violation = risk_cost - float(master_theta)
        policy_key = tuple(selected_indices)

        if gap <= objective_tolerance and violation <= solver_tolerance:
            converged_by_benders = True
            termination_reason = "gap_closed"
            break

        if policy_key in seen_policies and violation <= solver_tolerance:
            # Re-solving an already supported policy means bounds, not a missing
            # subgradient, determine convergence.  The reference result below
            # remains the safe result if the numerical gap is not closed.
            termination_reason = "repeated_supported_policy"
            break
        seen_policies.add(policy_key)

        gradient = np.asarray(
            [
                weighted_cvar(costs[index], alpha, model_probabilities)
                for index in range(len(options))
            ]
        )
        # Every option cost is B_i + K_i*N with K_i >= 0, so all option losses
        # have the same scenario ordering. Empirical CVaR is additive for these
        # comonotone losses, making this zero-intercept cut globally exact for
        # the documented model rather than an approximate float dual at one
        # incumbent.
        cuts.append(gradient)

    if incumbent is None:  # pragma: no cover - defensive for unexpected solver behaviour
        return nominal_fallback("Benders did not evaluate a feasible policy", completed_iterations, len(cuts))
    exact_option_scores = _exact_mean_cvar_option_scores(
        options,
        model_events,
        alpha,
        risk_weight,
        model_probabilities,
        lead_time_cost_per_day,
    )
    reporting_option_scores = _exact_reporting_option_scores(
        options, events, lead_time_cost_per_day
    )
    _, benders_incumbent_replaced, _ = _componentwise_certified_selection(
        incumbent[0], eligible, exact_option_scores, reporting_option_scores
    )
    if benders_incumbent_replaced:
        converged_by_benders = False
        termination_reason = "benders_incumbent_failed_exact_certificate"
    # Independently validate every decomposed answer. The nominal result is
    # itself protected by the exact componentwise certificate; a Benders
    # candidate is only reported as converged after its exact certificate and
    # conditioned objective agree with that reference.
    raw_reference_conditioned_objective = reference.details["conditioned_objective"]
    if raw_reference_conditioned_objective is None:
        return nominal_fallback(
            "nominal conditioned objective was not representable",
            completed_iterations,
            len(cuts),
        )
    reference_conditioned_objective = float(raw_reference_conditioned_objective)
    reference_tolerance = _solver_tolerance(
        max(abs(reference_conditioned_objective), abs(incumbent[5])), absolute_tolerance, relative_tolerance
    )
    if converged_by_benders and abs(incumbent[5] - reference_conditioned_objective) <= reference_tolerance:
        verified_lower_bound = min(effective_lower_bound, reference_conditioned_objective)
        (
            reported_lower_bound,
            reported_upper_bound,
            reported_gap,
            reported_relative_gap,
        ) = _reported_benders_bounds(
            baseline_objective, verified_lower_bound, reference.objective
        )
        return OptimizationResult(
            # Return the independently evaluated reference policy rather than
            # the solver-serialized Benders incumbent.  The latter can differ
            # by display precision even after a valid gap closure.
            selected=reference.selected,
            losses=reference.losses,
            mean_cost=reference.mean_cost,
            risk_cost=reference.risk_cost,
            objective=reference.objective,
            status=reference.status,
            details={
                "method": "benders_mean_cvar",
                "iterations": completed_iterations,
                "lower_bound": reported_lower_bound,
                "upper_bound": reported_upper_bound,
                "gap": reported_gap,
                "relative_gap": reported_relative_gap,
                "cuts": len(cuts),
                "converged": True,
                "reference_verified": True,
                "returned_nominal_reference": True,
                "fallback_to_nominal": False,
                "benders_incumbent_objective": baseline_objective + incumbent[5],
                "conditioned_benders_incumbent_objective": incumbent[5],
                "componentwise_optimality_certified": reference.details[
                    "componentwise_optimality_certified"
                ],
                "solver_policy_replaced_by_certificate": reference.details[
                    "solver_policy_replaced_by_certificate"
                ],
                "benders_incumbent_passed_exact_certificate": True,
                "absolute_tolerance": absolute_tolerance,
                "relative_tolerance": relative_tolerance,
                "solver_tolerance": solver_tolerance,
                "scenario_weights": probabilities.copy(),
            },
        )

    if converged_by_benders:
        termination_reason = "benders_candidate_did_not_match_nominal_reference"
    elif lower_bound > reference_conditioned_objective + reference_tolerance:
        termination_reason = "master_lower_bound_exceeded_nominal_reference"
    (
        reported_lower_bound,
        reported_upper_bound,
        reported_gap,
        reported_relative_gap,
    ) = _reported_benders_bounds(
        baseline_objective,
        min(lower_bound, reference_conditioned_objective),
        reference.objective,
    )
    return OptimizationResult(
        selected=reference.selected,
        losses=reference.losses,
        mean_cost=reference.mean_cost,
        risk_cost=reference.risk_cost,
        objective=reference.objective,
        status=reference.status,
        details={
            "method": "benders_mean_cvar",
            "iterations": completed_iterations,
            "lower_bound": reported_lower_bound,
            "upper_bound": reported_upper_bound,
            "gap": reported_gap,
            "relative_gap": reported_relative_gap,
            "cuts": len(cuts),
            "converged": False,
            "fallback_to_nominal": True,
            "fallback_reason": termination_reason,
            "reference_verified": False,
            "benders_incumbent_objective": baseline_objective + incumbent[5],
            "conditioned_benders_incumbent_objective": incumbent[5],
            "componentwise_optimality_certified": reference.details[
                "componentwise_optimality_certified"
            ],
            "solver_policy_replaced_by_certificate": reference.details[
                "solver_policy_replaced_by_certificate"
            ],
            "benders_incumbent_passed_exact_certificate": not benders_incumbent_replaced,
            "absolute_tolerance": absolute_tolerance,
            "relative_tolerance": relative_tolerance,
            "solver_tolerance": reference_tolerance,
            "scenario_weights": probabilities.copy(),
        },
    )


def solve_spectral_cvar_mixture(
    bom: BOM,
    event_counts: Iterable[float],
    *,
    levels: Sequence[float],
    mixture_weights: Sequence[float],
    risk_weight: float = 1.0,
    scenario_weights: Optional[Iterable[float]] = None,
    lead_time_cost_per_day: float = 0.0,
    max_lead_time_days: Optional[float] = None,
) -> OptimizationResult:
    """Solve a documented finite convex mixture of empirical CVaRs.

    This deliberately does not claim to be a continuous spectral-density or a
    Wang transform; users supply the finite levels and convex weights actually
    optimized by the MILP.
    """

    if isinstance(levels, (str, bytes, bytearray, memoryview)) or isinstance(
        mixture_weights, (str, bytes, bytearray, memoryview)
    ):
        raise OptimizationError("levels and mixture_weights must be numeric iterables")
    if (
        isinstance(levels, (Mapping, AbstractSet))
        or isinstance(mixture_weights, (Mapping, AbstractSet))
        or getattr(levels, "ndim", 1) != 1
        or getattr(mixture_weights, "ndim", 1) != 1
    ):
        raise OptimizationError(
            "levels and mixture_weights must be ordered one-dimensional numeric iterables"
        )
    try:
        levels = tuple(levels)
        mixture_weights = tuple(mixture_weights)
    except TypeError as exc:
        raise OptimizationError("levels and mixture_weights must be numeric iterables") from exc
    if len(levels) == 0 or len(levels) != len(mixture_weights):
        raise OptimizationError("levels and mixture_weights must have equal non-zero length")
    alphas = [_validate_solver_alpha(level) for level in levels]
    mix = _normalized_weights_or_error(
        mixture_weights, len(levels), name="mixture weights"
    )
    risk_weight = _validate_nonnegative(risk_weight, "risk_weight")
    lead_time_cost_per_day = _validate_nonnegative(lead_time_cost_per_day, "lead_time_cost_per_day")
    events = _event_array(event_counts)
    probabilities = _normalized_weights_or_error(scenario_weights, len(events))
    model_events, model_probabilities = _positive_probability_scenarios(
        events, probabilities
    )
    options = flatten_bom(bom)
    eligible = _eligible_indices(options, max_lead_time_days)
    exact_option_scores = _exact_spectral_option_scores(
        options,
        model_events,
        alphas,
        mix,
        risk_weight,
        model_probabilities,
        lead_time_cost_per_day,
    )
    reporting_option_scores = _exact_reporting_option_scores(
        options, events, lead_time_cost_per_day
    )
    exact_selected_indices, exact_objective = _canonical_exact_selection(
        eligible, exact_option_scores, reporting_option_scores
    )

    def exact_fallback(
        reason: str, conditioning: Optional[_ConditionedCosts] = None
    ) -> OptimizationResult:
        selected = tuple(options[index] for index in exact_selected_indices)
        losses = _selected_policy_losses(
            options, exact_selected_indices, events, lead_time_cost_per_day
        )
        mean_cost = weighted_mean(losses, probabilities)
        per_level = np.asarray(
            [weighted_cvar(losses, level, probabilities) for level in alphas],
            dtype=float,
        )
        risk_cost = _finite_fraction(
            _exact_cvar_mixture(losses, alphas, mix, probabilities),
            "spectral risk is outside floating-point range; rescale monetary inputs",
        )
        objective = _finite_objective(mean_cost, risk_cost, risk_weight)
        conditioned_objective: Optional[float] = None
        baseline_objective: Optional[float] = None
        if conditioning is not None:
            conditioned_losses = _policy_losses(
                conditioning.costs, exact_selected_indices
            )
            conditioned_mean = weighted_mean(
                conditioned_losses, model_probabilities
            )
            conditioned_objective = _finite_objective(
                conditioned_mean,
                _finite_fraction(
                    _exact_cvar_mixture(
                        conditioned_losses, alphas, mix, model_probabilities
                    ),
                    "spectral risk is outside floating-point range; rescale monetary inputs",
                ),
                risk_weight,
            )
            baseline_objective = _finite_objective(
                weighted_mean(conditioning.baseline_losses, model_probabilities),
                _finite_fraction(
                    _exact_cvar_mixture(
                        conditioning.baseline_losses,
                        alphas,
                        mix,
                        model_probabilities,
                    ),
                    "spectral risk is outside floating-point range; rescale monetary inputs",
                ),
                risk_weight,
            )
        return OptimizationResult(
            selected=selected,
            losses=losses,
            mean_cost=mean_cost,
            risk_cost=risk_cost,
            objective=objective,
            status="ExactOptimal",
            details={
                "method": "finite_cvar_mixture",
                "levels": tuple(alphas),
                "mixture_weights": mix.copy(),
                "per_level_cvar": per_level,
                "solver_conditioned_objective": None,
                "conditioned_objective": conditioned_objective,
                "baseline_objective": baseline_objective,
                "componentwise_optimality_certified": True,
                "solver_policy_replaced_by_certificate": False,
                "solver_bypassed_for_exact_certificate": True,
                "solver_fallback_reason": reason,
                "exact_certificate_objective": str(exact_objective),
                "scenario_weights": probabilities.copy(),
            },
        )

    try:
        conditioning = _condition_costs(
            options, model_events, lead_time_cost_per_day, eligible
        )
    except OptimizationError:
        return exact_fallback("conditioned_solver_coefficients_not_representable")
    model_costs = conditioning.costs
    cost_scale = _solver_cost_scale(model_costs, eligible)
    solver_costs = model_costs / cost_scale
    objective_scale = max(1.0, risk_weight)
    mean_factor = 1.0 / objective_scale
    risk_factor = risk_weight / objective_scale

    problem = pulp.LpProblem("tailguard_spectral_cvar", pulp.LpMinimize)
    x = _make_decision_variables(problem, options, eligible)
    zeta = {index: _lp_variable(problem, f"zeta_{index}") for index in range(len(alphas))}
    slacks = {
        (scenario, level): _lp_variable(problem, f"slack_{scenario}_{level}", low_bound=0)
        for scenario in range(len(model_events))
        for level in range(len(alphas))
    }
    scenario_losses = {
        scenario: pulp.lpSum(
            x[index] * float(solver_costs[index, scenario])
            for index in range(len(options))
        )
        for scenario in range(len(model_events))
    }
    for level in range(len(alphas)):
        for scenario, loss in scenario_losses.items():
            problem += slacks[(scenario, level)] >= loss - zeta[level], f"tail_{scenario}_{level}"

    mean_expression = pulp.lpSum(
        x[index] * weighted_mean(solver_costs[index], model_probabilities)
        for index in range(len(options))
    )
    probability_total = math.fsum(float(value) for value in model_probabilities)
    mixture_total = math.fsum(float(value) for value in mix)
    spectral_expression = (
        pulp.lpSum(
            float(mix[level])
            * (
                zeta[level]
                + pulp.lpSum(
                    float(model_probabilities[scenario]) * slacks[(scenario, level)]
                    for scenario in range(len(model_events))
                )
                / ((1.0 - alphas[level]) * probability_total)
            )
            for level in range(len(alphas))
        )
        / mixture_total
    )
    problem += mean_factor * mean_expression + risk_factor * spectral_expression
    try:
        status = _solve_or_raise(problem)
        solver_selected_indices = _selected_indices(x)
        solver_scaled_losses = _policy_losses(
            solver_costs, solver_selected_indices
        )
        solver_scaled_mean = weighted_mean(
            solver_scaled_losses, model_probabilities
        )
        solver_normalized_objective = math.fsum(
            (
                mean_factor * solver_scaled_mean,
                risk_factor
                * _finite_fraction(
                    _exact_cvar_mixture(
                        solver_scaled_losses, alphas, mix, model_probabilities
                    ),
                    "normalized spectral risk is outside floating-point range",
                ),
            )
        )
        model_objective = _solved_objective_value(problem)
        if not _objectives_match(model_objective, solver_normalized_objective):
            raise OptimizationError(
                "solver objective does not match independent normalized spectral-risk evaluation"
            )
        selected_indices, solver_policy_replaced, certified_objective = (
            _componentwise_certified_selection(
                solver_selected_indices,
                eligible,
                exact_option_scores,
                reporting_option_scores,
            )
        )
        if certified_objective != exact_objective:  # pragma: no cover - same certificate path.
            raise OptimizationError("exact optimality certificates disagree")
    except OptimizationError as exc:
        return exact_fallback(str(exc), conditioning)

    solver_conditioned_objective: Optional[float]
    try:
        solver_conditioned_losses = _policy_losses(
            model_costs, solver_selected_indices
        )
        solver_conditioned_mean = weighted_mean(
            solver_conditioned_losses, model_probabilities
        )
        solver_conditioned_risk = _finite_fraction(
            _exact_cvar_mixture(
                solver_conditioned_losses, alphas, mix, model_probabilities
            ),
            "spectral risk is outside floating-point range; rescale monetary inputs",
        )
        solver_conditioned_objective = _finite_objective(
            solver_conditioned_mean, solver_conditioned_risk, risk_weight
        )
    except OptimizationError:
        # This is only a diagnostic for the raw solver policy. The exact
        # componentwise certificate may already have replaced that policy with
        # a finite, reportable optimum.
        solver_conditioned_objective = None
    selected = tuple(options[index] for index in selected_indices)
    losses = _selected_policy_losses(
        options, selected_indices, events, lead_time_cost_per_day
    )
    mean_cost = weighted_mean(losses, probabilities)
    per_level = np.asarray([weighted_cvar(losses, level, probabilities) for level in alphas], dtype=float)
    risk_cost = _finite_fraction(
        _exact_cvar_mixture(losses, alphas, mix, probabilities),
        "spectral risk is outside floating-point range; rescale monetary inputs",
    )
    objective = _finite_objective(mean_cost, risk_cost, risk_weight)
    conditioned_losses = _policy_losses(model_costs, selected_indices)
    conditioned_mean = weighted_mean(conditioned_losses, model_probabilities)
    conditioned_risk = _finite_fraction(
        _exact_cvar_mixture(
            conditioned_losses, alphas, mix, model_probabilities
        ),
        "spectral risk is outside floating-point range; rescale monetary inputs",
    )
    conditioned_objective = _finite_objective(
        conditioned_mean, conditioned_risk, risk_weight
    )
    baseline_objective = _finite_objective(
        weighted_mean(conditioning.baseline_losses, model_probabilities),
        _finite_fraction(
            _exact_cvar_mixture(
                conditioning.baseline_losses,
                alphas,
                mix,
                model_probabilities,
            ),
            "spectral risk is outside floating-point range; rescale monetary inputs",
        ),
        risk_weight,
    )
    return OptimizationResult(
        selected=selected,
        losses=losses,
        mean_cost=mean_cost,
        risk_cost=risk_cost,
        objective=objective,
        status=status,
        details={
            "method": "finite_cvar_mixture",
            "levels": tuple(alphas),
            "mixture_weights": mix.copy(),
            "per_level_cvar": per_level,
            "solver_conditioned_objective": solver_conditioned_objective,
            "solver_normalized_objective": solver_normalized_objective,
            "solver_cost_scale": cost_scale,
            "solver_objective_scale": objective_scale,
            "conditioned_objective": conditioned_objective,
            "baseline_objective": baseline_objective,
            "componentwise_optimality_certified": True,
            "solver_policy_replaced_by_certificate": solver_policy_replaced,
            "exact_certificate_objective": str(exact_objective),
            "scenario_weights": probabilities.copy(),
        },
    )


def solve_wasserstein_cvar(
    bom: BOM,
    event_counts: Iterable[float],
    *,
    epsilon: float,
    alpha: float = 0.95,
    risk_weight: float = 1.0,
    scenario_weights: Optional[Iterable[float]] = None,
    lead_time_cost_per_day: float = 0.0,
    max_lead_time_days: Optional[float] = None,
) -> OptimizationResult:
    """Solve W1-robust CVaR over the scalar shock-count distribution.

    The ambiguity set is a type-1 Wasserstein ball of radius ``epsilon`` about
    the empirical distribution of ``event_counts``, with ground metric
    ``|n-n'|``.  For ``Z(x,n)=B(x)+K(x)n`` and non-negative
    ``K(x)=sum(kappa_i x_i)``, the exact robust empirical CVaR is

    ``empirical_CVaR + epsilon * K(x) / (1-alpha)``.

    This is intentionally *not* a claim about uncertainty in fitted CIR
    parameters.  ``epsilon`` is measured in shock-count units.
    """

    alpha = _validate_solver_alpha(alpha)
    epsilon = _validate_nonnegative(epsilon, "epsilon")
    risk_weight = _validate_nonnegative(risk_weight, "risk_weight")
    lead_time_cost_per_day = _validate_nonnegative(lead_time_cost_per_day, "lead_time_cost_per_day")
    events = _event_array(event_counts)
    probabilities = _normalized_weights_or_error(scenario_weights, len(events))
    if epsilon == 0:
        nominal = solve_mean_cvar_milp(
            bom,
            events,
            alpha=alpha,
            risk_weight=risk_weight,
            scenario_weights=probabilities,
            lead_time_cost_per_day=lead_time_cost_per_day,
            max_lead_time_days=max_lead_time_days,
        )
        _, total_exposure, finite_tau = _selected_exposure_diagnostics(nominal.selected, alpha)
        return OptimizationResult(
            selected=nominal.selected,
            losses=nominal.losses,
            mean_cost=nominal.mean_cost,
            risk_cost=nominal.risk_cost,
            objective=nominal.objective,
            status=nominal.status,
            details={
                "method": "w1_robust_cvar_over_shock_counts",
                "epsilon": 0.0,
                "nominal_cvar": nominal.risk_cost,
                "shock_exposure": total_exposure,
                "tau": finite_tau,
                "expected_tau": finite_tau,
                "bypassed_for_zero_radius": True,
                "tau_not_representable": finite_tau is None,
                "componentwise_optimality_certified": nominal.details[
                    "componentwise_optimality_certified"
                ],
                "solver_policy_replaced_by_certificate": nominal.details[
                    "solver_policy_replaced_by_certificate"
                ],
                "scenario_weights": probabilities.copy(),
            },
        )
    if risk_weight == 0:
        model_events, model_probabilities = _positive_probability_scenarios(
            events, probabilities
        )
        options = flatten_bom(bom)
        eligible = _eligible_indices(options, max_lead_time_days)
        exact_scores = _exact_wasserstein_option_scores(
            options,
            model_events,
            alpha,
            epsilon,
            0.0,
            model_probabilities,
            lead_time_cost_per_day,
        )
        reporting_scores = _exact_wasserstein_reporting_option_scores(
            options, events, alpha, epsilon, lead_time_cost_per_day
        )
        selected_indices, exact_objective = _canonical_exact_selection(
            eligible, exact_scores, reporting_scores
        )
        selected = tuple(options[index] for index in selected_indices)
        losses = _selected_policy_losses(
            options, selected_indices, events, lead_time_cost_per_day
        )
        mean_cost = weighted_mean(losses, probabilities)
        nominal_cvar = weighted_cvar(losses, alpha, probabilities)
        exact_exposure, total_exposure, finite_tau = _selected_exposure_diagnostics(
            selected, alpha
        )
        exact_increment = Fraction.from_float(float(epsilon)) * exact_exposure / (
            Fraction(1) - Fraction.from_float(float(alpha))
        )
        robust_risk = _finite_fraction(
            _exact_wasserstein_risk(
                losses, alpha, probabilities, exact_increment
            ),
            "W1 risk scale overflows floating-point arithmetic; rescale monetary inputs",
        )
        return OptimizationResult(
            selected=selected,
            losses=losses,
            mean_cost=mean_cost,
            risk_cost=robust_risk,
            objective=mean_cost,
            status="ExactOptimal",
            details={
                "method": "w1_robust_cvar_over_shock_counts",
                "epsilon": epsilon,
                "nominal_cvar": nominal_cvar,
                "shock_exposure": total_exposure,
                "tau": finite_tau,
                "expected_tau": finite_tau,
                "bypassed_for_zero_risk_weight": True,
                "tau_not_representable": finite_tau is None,
                "componentwise_optimality_certified": True,
                "solver_policy_replaced_by_certificate": False,
                "solver_bypassed_for_exact_certificate": True,
                "exact_certificate_objective": str(exact_objective),
                "scenario_weights": probabilities.copy(),
            },
        )
    model_events, model_probabilities = _positive_probability_scenarios(
        events, probabilities
    )
    options = flatten_bom(bom)
    eligible = _eligible_indices(options, max_lead_time_days)
    exact_option_scores = _exact_wasserstein_option_scores(
        options,
        model_events,
        alpha,
        epsilon,
        risk_weight,
        model_probabilities,
        lead_time_cost_per_day,
    )
    reporting_option_scores = _exact_wasserstein_reporting_option_scores(
        options, events, alpha, epsilon, lead_time_cost_per_day
    )
    exact_selected_indices, exact_objective = _canonical_exact_selection(
        eligible, exact_option_scores, reporting_option_scores
    )
    exact_epsilon = Fraction.from_float(float(epsilon))
    exact_tail_mass = Fraction(1) - Fraction.from_float(float(alpha))

    def optional_float(value: Fraction) -> Optional[float]:
        try:
            converted = float(value)
        except OverflowError:
            return None
        return converted if math.isfinite(converted) else None

    def evaluate_exact_selection(
        selected_indices: Sequence[int],
    ) -> tuple[
        tuple[SupplierOption, ...],
        np.ndarray,
        float,
        float,
        float,
        Optional[float],
        Optional[float],
        float,
    ]:
        selected = tuple(options[index] for index in selected_indices)
        losses = _selected_policy_losses(
            options, selected_indices, events, lead_time_cost_per_day
        )
        mean_cost = weighted_mean(losses, probabilities)
        nominal_cvar = weighted_cvar(losses, alpha, probabilities)
        exact_exposure = sum(
            (Fraction.from_float(float(option.kappa)) for option in selected),
            Fraction(0),
        )
        exact_tau = exact_exposure / exact_tail_mass
        exact_increment = exact_epsilon * exact_tau
        risk_cost = _finite_fraction(
            _exact_wasserstein_risk(
                losses, alpha, probabilities, exact_increment
            ),
            "W1 risk scale overflows floating-point arithmetic; rescale monetary inputs",
        )
        objective = _finite_objective(mean_cost, risk_cost, risk_weight)
        return (
            selected,
            losses,
            mean_cost,
            nominal_cvar,
            risk_cost,
            optional_float(exact_exposure),
            optional_float(exact_tau),
            objective,
        )

    def exact_fallback(
        reason: str, conditioning: Optional[_ConditionedCosts] = None
    ) -> OptimizationResult:
        (
            selected,
            losses,
            mean_cost,
            nominal_cvar,
            risk_cost,
            total_exposure,
            analytical_tau,
            objective,
        ) = evaluate_exact_selection(exact_selected_indices)
        conditioned_objective: Optional[float] = None
        baseline_objective: Optional[float] = None
        if conditioning is not None:
            conditioned_losses = _policy_losses(
                conditioning.costs, exact_selected_indices
            )
            conditioned_exposure = sum(
                (
                    Fraction.from_float(float(conditioning.exposures[index]))
                    for index in exact_selected_indices
                ),
                Fraction(0),
            )
            conditioned_risk = _finite_fraction(
                _exact_wasserstein_risk(
                    conditioned_losses,
                    alpha,
                    model_probabilities,
                    exact_epsilon * conditioned_exposure / exact_tail_mass,
                ),
                "W1 risk scale overflows floating-point arithmetic; rescale monetary inputs",
            )
            conditioned_objective = _finite_objective(
                weighted_mean(conditioned_losses, model_probabilities),
                conditioned_risk,
                risk_weight,
            )
            baseline_exposure = sum(
                (
                    min(Fraction.from_float(float(options[index].kappa)) for index in indices)
                    for indices in eligible.values()
                ),
                Fraction(0),
            )
            baseline_risk = _finite_fraction(
                _exact_wasserstein_risk(
                    conditioning.baseline_losses,
                    alpha,
                    model_probabilities,
                    exact_epsilon * baseline_exposure / exact_tail_mass,
                ),
                "W1 risk scale overflows floating-point arithmetic; rescale monetary inputs",
            )
            baseline_objective = _finite_objective(
                weighted_mean(conditioning.baseline_losses, model_probabilities),
                baseline_risk,
                risk_weight,
            )
        return OptimizationResult(
            selected=selected,
            losses=losses,
            mean_cost=mean_cost,
            risk_cost=risk_cost,
            objective=objective,
            status="ExactOptimal",
            details={
                "method": "w1_robust_cvar_over_shock_counts",
                "epsilon": epsilon,
                "nominal_cvar": nominal_cvar,
                "shock_exposure": total_exposure,
                "tau": analytical_tau,
                "expected_tau": analytical_tau,
                "tau_not_representable": analytical_tau is None,
                "solver_tau": None,
                "conditioned_solver_tau": None,
                "solver_conditioned_objective": None,
                "conditioned_objective": conditioned_objective,
                "baseline_objective": baseline_objective,
                "componentwise_optimality_certified": True,
                "solver_policy_replaced_by_certificate": False,
                "solver_bypassed_for_exact_certificate": True,
                "solver_fallback_reason": reason,
                "exact_certificate_objective": str(exact_objective),
                "scenario_weights": probabilities.copy(),
            },
        )

    try:
        conditioning = _condition_costs(
            options, model_events, lead_time_cost_per_day, eligible
        )
    except OptimizationError:
        return exact_fallback("conditioned_solver_coefficients_not_representable")
    model_costs = conditioning.costs
    conditioned_increments = np.zeros(len(options), dtype=float)
    try:
        for indices in eligible.values():
            for index in indices:
                conditioned_increments[index] = _finite_fraction(
                    exact_epsilon
                    * Fraction.from_float(float(conditioning.exposures[index]))
                    / exact_tail_mass,
                    "W1 solver coefficient is not representable",
                )
    except OptimizationError:
        return exact_fallback("conditioned_W1_solver_coefficient_not_representable", conditioning)
    eligible_indices = np.asarray(
        sorted(index for indices in eligible.values() for index in indices), dtype=int
    )
    solver_scale = max(
        _solver_cost_scale(model_costs, eligible),
        float(np.max(conditioned_increments[eligible_indices])),
    )
    solver_costs = model_costs / solver_scale
    solver_increments = conditioned_increments / solver_scale
    objective_scale = max(1.0, risk_weight)
    mean_factor = 1.0 / objective_scale
    risk_factor = risk_weight / objective_scale

    problem = pulp.LpProblem("tailguard_wasserstein_cvar", pulp.LpMinimize)
    x = _make_decision_variables(problem, options, eligible)
    zeta = _lp_variable(problem, "zeta")
    slacks = {
        scenario: _lp_variable(problem, f"slack_{scenario}", low_bound=0)
        for scenario in range(len(model_events))
    }
    scenario_losses = {
        scenario: pulp.lpSum(
            x[index] * float(solver_costs[index, scenario])
            for index in range(len(options))
        )
        for scenario in range(len(model_events))
    }
    for scenario, loss in scenario_losses.items():
        problem += slacks[scenario] >= loss - zeta, f"tail_{scenario}"
    mean_expression = pulp.lpSum(
        x[index] * weighted_mean(solver_costs[index], model_probabilities)
        for index in range(len(options))
    )
    robust_cvar = (
        zeta
        + pulp.lpSum(
            x[index] * float(solver_increments[index]) for index in range(len(options))
        )
        + pulp.lpSum(
            float(model_probabilities[scenario]) * slacks[scenario]
            for scenario in range(len(model_events))
        )
        / (
            (1.0 - alpha)
            * math.fsum(float(value) for value in model_probabilities)
        )
    )
    problem += mean_factor * mean_expression + risk_factor * robust_cvar

    try:
        status = _solve_or_raise(problem)
        solver_selected_indices = _selected_indices(x)
        solver_scaled_losses = _policy_losses(
            solver_costs, solver_selected_indices
        )
        solver_scaled_mean = weighted_mean(
            solver_scaled_losses, model_probabilities
        )
        solver_scaled_increment = sum(
            (
                Fraction.from_float(float(solver_increments[index]))
                for index in solver_selected_indices
            ),
            Fraction(0),
        )
        solver_scaled_risk = _finite_fraction(
            _exact_wasserstein_risk(
                solver_scaled_losses,
                alpha,
                model_probabilities,
                solver_scaled_increment,
            ),
            "normalized W1 risk is outside floating-point range",
        )
        solver_normalized_objective = math.fsum(
            (
                mean_factor * solver_scaled_mean,
                risk_factor * solver_scaled_risk,
            )
        )
        if not _objectives_match(
            _solved_objective_value(problem), solver_normalized_objective
        ):
            raise OptimizationError(
                "solver objective does not match independent normalized W1 evaluation"
            )
        selected_indices, solver_policy_replaced, certified_objective = (
            _componentwise_certified_selection(
                solver_selected_indices,
                eligible,
                exact_option_scores,
                reporting_option_scores,
            )
        )
        if certified_objective != exact_objective:  # pragma: no cover - same certificate path.
            raise OptimizationError("exact optimality certificates disagree")
    except OptimizationError as exc:
        return exact_fallback(str(exc), conditioning)

    (
        selected,
        losses,
        mean_cost,
        nominal_cvar,
        risk_cost,
        total_exposure,
        analytical_tau,
        objective,
    ) = evaluate_exact_selection(selected_indices)
    conditioned_losses = _policy_losses(model_costs, selected_indices)
    conditioned_mean = weighted_mean(conditioned_losses, model_probabilities)
    conditioned_exposure = sum(
        (
            Fraction.from_float(float(conditioning.exposures[index]))
            for index in selected_indices
        ),
        Fraction(0),
    )
    conditioned_increment = exact_epsilon * conditioned_exposure / exact_tail_mass
    conditioned_risk = _finite_fraction(
        _exact_wasserstein_risk(
            conditioned_losses,
            alpha,
            model_probabilities,
            conditioned_increment,
        ),
        "W1 risk scale overflows floating-point arithmetic; rescale monetary inputs",
    )
    conditioned_objective = _finite_objective(
        conditioned_mean, conditioned_risk, risk_weight
    )
    solver_conditioned_exposure = sum(
        (
            Fraction.from_float(float(conditioning.exposures[index]))
            for index in solver_selected_indices
        ),
        Fraction(0),
    )
    solver_conditioned_objective: Optional[float]
    try:
        solver_conditioned_losses = _policy_losses(
            model_costs, solver_selected_indices
        )
        solver_conditioned_risk = _finite_fraction(
            _exact_wasserstein_risk(
                solver_conditioned_losses,
                alpha,
                model_probabilities,
                exact_epsilon * solver_conditioned_exposure / exact_tail_mass,
            ),
            "W1 risk scale overflows floating-point arithmetic; rescale monetary inputs",
        )
        solver_conditioned_objective = _finite_objective(
            weighted_mean(solver_conditioned_losses, model_probabilities),
            solver_conditioned_risk,
            risk_weight,
        )
    except OptimizationError:
        # The raw solver policy is diagnostic-only after exact certification.
        solver_conditioned_objective = None
    baseline_exposure = sum(
        (
            min(Fraction.from_float(float(options[index].kappa)) for index in indices)
            for indices in eligible.values()
        ),
        Fraction(0),
    )
    baseline_risk = _finite_fraction(
        _exact_wasserstein_risk(
            conditioning.baseline_losses,
            alpha,
            model_probabilities,
            exact_epsilon * baseline_exposure / exact_tail_mass,
        ),
        "W1 risk scale overflows floating-point arithmetic; rescale monetary inputs",
    )
    baseline_objective = _finite_objective(
        weighted_mean(conditioning.baseline_losses, model_probabilities),
        baseline_risk,
        risk_weight,
    )
    return OptimizationResult(
        selected=selected,
        losses=losses,
        mean_cost=mean_cost,
        risk_cost=risk_cost,
        objective=objective,
        status=status,
        details={
            "method": "w1_robust_cvar_over_shock_counts",
            "epsilon": epsilon,
            "nominal_cvar": nominal_cvar,
            "shock_exposure": total_exposure,
            "tau": analytical_tau,
            "expected_tau": analytical_tau,
            "tau_not_representable": analytical_tau is None,
            "solver_tau": None,
            "conditioned_solver_tau": None,
            "conditioned_expected_tau": optional_float(
                conditioned_exposure / exact_tail_mass
            ),
            "solver_conditioned_expected_tau": optional_float(
                solver_conditioned_exposure / exact_tail_mass
            ),
            "solver_conditioned_objective": solver_conditioned_objective,
            "solver_normalized_objective": solver_normalized_objective,
            "solver_coefficient_scale": solver_scale,
            "solver_objective_scale": objective_scale,
            "conditioned_objective": conditioned_objective,
            "baseline_objective": baseline_objective,
            "analytical_W1_increment_used": True,
            "componentwise_optimality_certified": True,
            "solver_policy_replaced_by_certificate": solver_policy_replaced,
            "exact_certificate_objective": str(exact_objective),
            "scenario_weights": probabilities.copy(),
        },
    )


def default_wasserstein_radius(event_counts: Iterable[float], multiplier: float = 1.0) -> float:
    """Return a scale-aware heuristic W1 radius in shock-count units.

    This is a convenience for sensitivity analysis, not a statistical coverage
    guarantee.  Production use should choose and justify ``epsilon`` directly.
    """

    events = _event_array(event_counts)
    multiplier = _validate_nonnegative(multiplier, "multiplier")
    if multiplier == 0:
        return 0.0
    event_scale = float(np.max(np.abs(events)))
    with np.errstate(over="ignore", invalid="ignore"):
        standard_deviation = 0.0 if event_scale == 0 else event_scale * float(np.std(events / event_scale, ddof=0))
        scale = max(standard_deviation, 1.0)
    return _finite_fraction(
        Fraction.from_float(float(multiplier))
        * Fraction.from_float(float(scale))
        / Fraction.from_float(math.sqrt(len(events))),
        "shock-count scale overflows Wasserstein-radius arithmetic",
    )
