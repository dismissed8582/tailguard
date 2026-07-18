"""Dependency-light compatibility smoke test for the supported runtime API."""

from __future__ import annotations

import math

from tailguard import (
    SupplierOption,
    load_bom_csv_bytes,
    solve_benders_mean_cvar,
    solve_mean_cvar_milp,
    solve_spectral_cvar_mixture,
    solve_wasserstein_cvar,
    weighted_cvar,
)


def main() -> None:
    bom = {
        "A": (
            SupplierOption("A", "low_base", 10.0, 5.0, 4.0, "offshore"),
            SupplierOption("A", "low_tail", 14.0, 1.0, 2.0, "domestic"),
        )
    }
    events = [0.0, 0.0, 1.0, 3.0]
    nominal = solve_mean_cvar_milp(bom, events, alpha=0.75)
    benders = solve_benders_mean_cvar(bom, events, alpha=0.75)
    spectral = solve_spectral_cvar_mixture(
        bom,
        events,
        levels=[0.5, 0.75],
        mixture_weights=[0.25, 0.75],
    )
    robust = solve_wasserstein_cvar(bom, events, epsilon=0.1, alpha=0.75)

    assert all(
        result.status in {"Optimal", "ExactOptimal"}
        for result in (nominal, benders, spectral, robust)
    )
    assert math.isclose(nominal.objective, benders.objective, rel_tol=1e-12, abs_tol=1e-12)
    assert all(result.details["componentwise_optimality_certified"] for result in (nominal, benders, spectral, robust))
    assert weighted_cvar([0.0, 10.0], 0.75) == 10.0

    parsed = load_bom_csv_bytes(
        b"# synthetic smoke fixture\n"
        b"component,supplier,base_cost,kappa,lead_time,type\n"
        b"001,0007,1.5,0.25,2,domestic\n"
    )
    option = parsed["001"][0]
    assert option.supplier == "0007" and option.base_cost == 1.5
    print("Tailguard runtime smoke test passed.")


if __name__ == "__main__":
    main()
