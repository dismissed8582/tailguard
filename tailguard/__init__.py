"""Tailguard: a validated flat sourcing mean-plus-CVaR research prototype."""

from .bom import BOMValidationError, SupplierOption, bom_from_dataframe, load_bom_csv, load_bom_csv_bytes
from .optimize import (
    OptimizationError,
    OptimizationResult,
    default_wasserstein_radius,
    solve_benders_mean_cvar,
    solve_mean_cvar_milp,
    solve_spectral_cvar_mixture,
    solve_wasserstein_cvar,
)
from .risk import weighted_cvar, weighted_mean

__all__ = [
    "BOMValidationError",
    "OptimizationError",
    "OptimizationResult",
    "SupplierOption",
    "bom_from_dataframe",
    "default_wasserstein_radius",
    "load_bom_csv",
    "load_bom_csv_bytes",
    "solve_benders_mean_cvar",
    "solve_mean_cvar_milp",
    "solve_spectral_cvar_mixture",
    "solve_wasserstein_cvar",
    "weighted_cvar",
    "weighted_mean",
]
