#!/usr/bin/env python3
"""Build the checked-in Tailguard notebook from reviewable cell source."""

from __future__ import annotations

import argparse
import hashlib
import tempfile
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "tailguard.ipynb"


def markdown(source):
    cell = nbformat.v4.new_markdown_cell(source)
    cell["id"] = hashlib.sha256(("markdown\0" + source).encode("utf-8")).hexdigest()[:16]
    return cell


def code(source):
    cell = nbformat.v4.new_code_cell(source)
    cell["id"] = hashlib.sha256(("code\0" + source).encode("utf-8")).hexdigest()[:16]
    return cell


def build_notebook():
    cells = [
        markdown(
            """# Tailguard: validated mean-plus-CVaR sourcing prototype

This notebook is a front end for the tested `tailguard` package. The default
data are synthetic, the model is a flat one-supplier-per-component MILP, and
the output is a research diagnostic rather than sourcing advice."""
        ),
        markdown(
            """## Start here

1. Use **Kernel → Restart Kernel and Run All Cells**.
2. For a first run, keep the bundled synthetic inputs and default controls.
3. Optionally choose your own BOM or time-series CSV in the upload panel.
4. Select **Run validated analysis** and review the comparison and policy tables.

Use a trusted local kernel for sensitive data. Tailguard does not intentionally
save a standalone copy of an uploaded CSV, and it clears the upload control
after processing. Input values and results can still enter notebook outputs or
saved widget state, and the command-line solver may use short-lived operating-
system temporary files. Before sharing, clear every cell output and saved
widget state, save, restart the kernel and refresh without rerunning cells,
then save and inspect the notebook."""
        ),
        markdown(
            """## Scope

The input is a flat sourcing-option table, not a BOM-DAG. Costs and shock
exposures must be normalized to one finished unit. `type` is a user-supplied
reporting label, not verified geography. Lead time affects optimization only if
you explicitly choose a cost per day or a hard maximum.

The default data series is deterministic synthetic data. Tailguard performs no
live fetches. To use real data, load a local `date,value` CSV deliberately;
invalid or unavailable requested inputs are rejected rather than substituted."""
        ),
        code(
            """import os
from pathlib import Path
import sys

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import clear_output, display


def find_project_root():
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "tailguard").is_dir() and (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError(
        "Could not find the Tailguard repository. Clone it locally, install "
        "the notebook dependencies, and start Jupyter from the repository."
    )


def configured_local_csv(environment_name):
    value = os.environ.get(environment_name)
    return Path(value) if value else None


class PathSafeConfigurationError(RuntimeError):
    def _render_traceback_(self):
        # IPython's verbose traceback mode normally displays frame arguments,
        # which can include a configured local path. Render only static text.
        return [str(self)]


def load_configured_csv(loader, path, label):
    try:
        # Expand user-relative paths inside the error-handling boundary too;
        # a failed expansion can otherwise surface the configured value.
        return loader(path.expanduser())
    except Exception:
        # Suppress the chained loader error too: it may contain the configured
        # absolute path, which a user could otherwise save in an output cell.
        message = "Could not load the configured local {} CSV. Check its format and configuration.".format(label)
        path = None
        loader = None
        raise PathSafeConfigurationError(message) from None


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tailguard.bom import BOMValidationError, flatten_bom, load_bom_csv, load_bom_csv_bytes
from tailguard.data import DataValidationError, load_series_csv, load_series_csv_bytes, synthetic_series
from tailguard.optimize import (
    OptimizationError,
    default_wasserstein_radius,
    solve_benders_mean_cvar,
    solve_mean_cvar_milp,
    solve_spectral_cvar_mixture,
    solve_wasserstein_cvar,
)
from tailguard.stochastic import ScenarioValidationError, fit_cir_proxy, simulate_cox, simulate_stratified_cox
from tailguard.validation import evaluate_policy"""
        ),
        markdown(
            """## Load a validated sourcing-option table

The included automotive table is synthetic and illustrative. The upload panel
below is the easiest way to use your own data. For a repeatable automated
setup, you may instead set `TAILGUARD_BOM_CSV` before starting Jupyter; its
path is omitted from Tailguard's status and handled-error output. Leading
documentation lines beginning with `#` are allowed.
Invalid types, duplicates, missing values, and negative values fail fast."""
        ),
        code(
            """LOCAL_BOM_PATH = configured_local_csv("TAILGUARD_BOM_CSV")
DEFAULT_BOM_PATH = PROJECT_ROOT / "data" / "bom" / "bom_auto.csv"
BOM_PATH = LOCAL_BOM_PATH or DEFAULT_BOM_PATH
BOM_SOURCE = "configured local CSV" if LOCAL_BOM_PATH else "bundled synthetic fixture"
if LOCAL_BOM_PATH is None:
    BOM = load_bom_csv(BOM_PATH)
else:
    BOM = load_configured_csv(load_bom_csv, LOCAL_BOM_PATH, "BOM")

def show_bom_input():
    bom_rows = [
        {
            "component": option.component,
            "supplier": option.supplier,
            "base_cost": option.base_cost,
            "kappa": option.kappa,
            "lead_time_days": option.lead_time,
            "type": option.source_type,
        }
        for option in flatten_bom(BOM)
    ]
    display(pd.DataFrame(bom_rows))
    component_label = "component" if len(BOM) == 1 else "components"
    option_label = "option" if len(bom_rows) == 1 else "options"
    print(
        "BOM source: {} | {} {} | {} {}".format(
            BOM_SOURCE, len(BOM), component_label, len(bom_rows), option_label
        )
    )


show_bom_input()"""
        ),
        markdown(
            """## Scenario input

`fit_cir_proxy` maps a generic index into a positive proxy intensity; it is not
an event-data calibration. The shared count below is a common-shock stress
model, not empirical evidence of tail dependence. Upload a `date,value` CSV
below, or set `TAILGUARD_SERIES_CSV` before starting Jupyter for a repeatable
automated setup. Tailguard's status and handled-error output omit local paths."""
        ),
        code(
            """LOCAL_SERIES_PATH = configured_local_csv("TAILGUARD_SERIES_CSV")
DEFAULT_TRAINING_SERIES = synthetic_series("GSCPI_PROXY", start="2017-01-01", end="2019-12-01")

if LOCAL_SERIES_PATH is None:
    TRAINING_SERIES = DEFAULT_TRAINING_SERIES.copy()
    DATA_SOURCE = "deterministic synthetic proxy"
else:
    TRAINING_SERIES = load_configured_csv(load_series_csv, LOCAL_SERIES_PATH, "series")
    DATA_SOURCE = "configured local CSV"

def show_series_input():
    display(TRAINING_SERIES.tail())
    print(
        "Data source: {} | {} observations | {} to {}".format(
            DATA_SOURCE,
            len(TRAINING_SERIES),
            TRAINING_SERIES["date"].min().date(),
            TRAINING_SERIES["date"].max().date(),
        )
    )


show_series_input()"""
        ),
        markdown(
            """## Use your own CSVs (optional)

Choose a file to replace the corresponding input for this kernel session. The
validated data are activated only after the whole file passes validation; a
bad upload leaves the current input unchanged. Use the reset buttons to return
to the bundled examples.

- BOM CSV columns: `component,supplier,base_cost,kappa,lead_time,type`
- Series CSV columns: `date,value` with at least four finite observations at distinct dates
- Validation cap: 5 MiB per file, checked after the browser has transferred
  the selection to the kernel; it is not a transport or transient-memory cap

After validation, the upload control is cleared so it does not retain the raw
file or browser-supplied filename. Parsed values and results can still appear
in cell outputs or saved widget state; use a trusted local kernel and the
notebook cleanup steps above."""
        ),
        code(
            """MAX_UPLOAD_BYTES = 5 * 1024 * 1024


class UploadValidationError(ValueError):
    pass


def uploaded_csv_bytes(upload_widget):
    value = upload_widget.value
    if not value:
        raise UploadValidationError("Choose a CSV file first")
    item = next(iter(value.values())) if isinstance(value, dict) else value[0]
    content = item.get("content") if hasattr(item, "get") else getattr(item, "content", None)
    if content is None:
        raise UploadValidationError("The browser did not provide file content; choose the CSV again")
    if not isinstance(content, (bytes, bytearray, memoryview)):
        raise UploadValidationError("The browser provided an unsupported file-content representation")
    content_size = content.nbytes if isinstance(content, memoryview) else len(content)
    if content_size == 0:
        raise UploadValidationError("The selected CSV is empty")
    if content_size > MAX_UPLOAD_BYTES:
        raise UploadValidationError("The selected CSV exceeds the 5 MiB validation limit")
    return bytes(content)


def safe_error_message(exc):
    safe_types = (
        UploadValidationError,
        BOMValidationError,
        DataValidationError,
        ScenarioValidationError,
        OptimizationError,
    )
    if not isinstance(exc, safe_types):
        return type(exc).__name__ + ": operation failed; diagnostic details omitted"
    message = str(exc)
    for path_text, placeholder in (
        (str(PROJECT_ROOT), "<repository>"),
        (str(Path.cwd()), "<working directory>"),
        (str(Path.home()), "<home>"),
    ):
        if path_text:
            message = message.replace(path_text, placeholder)
    for local_path in (LOCAL_BOM_PATH, LOCAL_SERIES_PATH):
        if local_path is not None:
            message = message.replace(str(local_path), "<local CSV>")
    # Domain exceptions use slash-free static wording, so any remaining path
    # separator is treated as filesystem content and the detail is omitted.
    contains_unrecognized_path = "/" in message or "\\\\" in message
    if contains_unrecognized_path:
        return type(exc).__name__ + ": operation failed; path details omitted"
    return type(exc).__name__ + ": " + message


def forget_uploaded_file(upload_widget, callback):
    if not hasattr(upload_widget, "unobserve") or not hasattr(upload_widget, "observe"):
        return
    empty_value = {} if isinstance(upload_widget.value, dict) else ()
    upload_widget.unobserve(callback, names="value")
    try:
        upload_widget.value = empty_value
    finally:
        upload_widget.observe(callback, names="value")


def invalidate_analysis_results():
    callback = globals().get("mark_results_stale")
    if callback is not None:
        callback()


bom_upload = widgets.FileUpload(accept=".csv,text/csv", multiple=False, description="Choose BOM CSV")
bom_reset = widgets.Button(description="Use bundled BOM")
bom_status = widgets.HTML(value="<b>Current:</b> " + BOM_SOURCE)
bom_upload_output = widgets.Output()

series_upload = widgets.FileUpload(accept=".csv,text/csv", multiple=False, description="Choose series CSV")
series_reset = widgets.Button(description="Use synthetic series")
series_status = widgets.HTML(value="<b>Current:</b> " + DATA_SOURCE)
series_upload_output = widgets.Output()


def on_bom_upload(_):
    global BOM, BOM_SOURCE
    with bom_upload_output:
        clear_output(wait=True)
        try:
            candidate = load_bom_csv_bytes(uploaded_csv_bytes(bom_upload))
        except Exception as exc:
            bom_status.value = '<b style="color:#b91c1c">Upload rejected; current BOM unchanged.</b>'
            print(safe_error_message(exc))
        else:
            BOM = candidate
            BOM_SOURCE = "uploaded CSV"
            invalidate_analysis_results()
            bom_status.value = '<b style="color:#047857">Uploaded BOM is active.</b>'
            show_bom_input()
        finally:
            forget_uploaded_file(bom_upload, on_bom_upload)


def on_bom_reset(_):
    global BOM, BOM_SOURCE
    with bom_upload_output:
        clear_output(wait=True)
        BOM = load_bom_csv(DEFAULT_BOM_PATH)
        BOM_SOURCE = "bundled synthetic fixture"
        invalidate_analysis_results()
        bom_status.value = '<b style="color:#1d4ed8">Bundled BOM is active.</b>'
        show_bom_input()


def on_series_upload(_):
    global TRAINING_SERIES, DATA_SOURCE
    with series_upload_output:
        clear_output(wait=True)
        try:
            candidate = load_series_csv_bytes(uploaded_csv_bytes(series_upload))
            candidate_fit = fit_cir_proxy(candidate)
            # Exercise the notebook's fixed annual grid once so common numeric
            # incompatibilities are reported before activation. The full run
            # still performs its own validation for every sampled scenario.
            simulate_cox(candidate_fit.parameters, scenarios=2, steps=252, seed=0)
        except Exception as exc:
            series_status.value = '<b style="color:#b91c1c">Upload rejected; current series unchanged.</b>'
            print(safe_error_message(exc))
        else:
            TRAINING_SERIES = candidate
            DATA_SOURCE = "uploaded CSV"
            invalidate_analysis_results()
            series_status.value = '<b style="color:#047857">Uploaded series is active.</b>'
            show_series_input()
        finally:
            forget_uploaded_file(series_upload, on_series_upload)


def on_series_reset(_):
    global TRAINING_SERIES, DATA_SOURCE
    with series_upload_output:
        clear_output(wait=True)
        TRAINING_SERIES = DEFAULT_TRAINING_SERIES.copy()
        DATA_SOURCE = "deterministic synthetic proxy"
        invalidate_analysis_results()
        series_status.value = '<b style="color:#1d4ed8">Synthetic series is active.</b>'
        show_series_input()


bom_upload.observe(on_bom_upload, names="value")
bom_reset.on_click(on_bom_reset)
series_upload.observe(on_series_upload, names="value")
series_reset.on_click(on_series_reset)

display(
    widgets.VBox(
        [
            widgets.HTML("<b>Sourcing-option table</b>"),
            widgets.HBox([bom_upload, bom_reset]),
            bom_status,
            bom_upload_output,
            widgets.HTML("<b>Shock-proxy time series</b>"),
            widgets.HBox([series_upload, series_reset]),
            series_status,
            series_upload_output,
        ]
    )
)"""
        ),
        code(
            """def policy_table(result, lead_time_cost_per_day):
    return pd.DataFrame(
        [
            {
                "Component": option.component,
                "Selected supplier": option.supplier,
                "Input label": option.source_type,
                "Base cost": option.base_cost,
                "Shock exposure": option.kappa,
                "Lead time (days)": option.lead_time,
                "Cost before shocks": option.deterministic_cost(lead_time_cost_per_day),
            }
            for option in result.selected
        ]
    )


def run_analysis(alpha, risk_weight, scenarios, lead_time_cost_per_day, max_lead_time_days, epsilon_multiplier):
    fit = fit_cir_proxy(TRAINING_SERIES)
    counts, _ = simulate_cox(fit.parameters, scenarios=scenarios, seed=42)
    common = {
        "alpha": alpha,
        "risk_weight": risk_weight,
        "lead_time_cost_per_day": lead_time_cost_per_day,
        "max_lead_time_days": max_lead_time_days,
    }
    nominal = solve_mean_cvar_milp(BOM, counts, **common)
    benders = solve_benders_mean_cvar(BOM, counts, **common)

    stratified_counts, _, stratified_weights = simulate_stratified_cox(
        fit.parameters, scenarios=scenarios, strata=10, seed=43
    )
    stratified = solve_mean_cvar_milp(
        BOM, stratified_counts, scenario_weights=stratified_weights, **common
    )
    mixture_levels = [max(0.50, alpha - 0.15), max(0.60, alpha - 0.05), alpha]
    mixture = solve_spectral_cvar_mixture(
        BOM,
        counts,
        levels=mixture_levels,
        mixture_weights=[0.20, 0.30, 0.50],
        risk_weight=risk_weight,
        lead_time_cost_per_day=lead_time_cost_per_day,
        max_lead_time_days=max_lead_time_days,
    )
    epsilon = default_wasserstein_radius(counts, epsilon_multiplier)
    robust = solve_wasserstein_cvar(BOM, counts, epsilon=epsilon, **common)

    evaluation_counts, _ = simulate_cox(
        fit.parameters, scenarios=min(scenarios * 10, 5000), seed=404
    )
    evaluation = evaluate_policy(
        nominal.selected,
        evaluation_counts,
        alpha=alpha,
        risk_weight=risk_weight,
        lead_time_cost_per_day=lead_time_cost_per_day,
    )
    return fit, counts, nominal, benders, stratified, mixture, robust, epsilon, evaluation


def w1_tau_text(tau, details):
    if tau is not None:
        return f"{tau:.3f}"
    if details.get("bypassed_for_zero_radius"):
        return "not representable (irrelevant at zero radius)"
    if details.get("bypassed_for_zero_risk_weight"):
        return "not representable (irrelevant to the objective at zero risk weight)"
    return "not representable"


def count_histogram_spec(counts, maximum_unit_bins=100):
    '''Return bounded histogram values, bins, and an honest x-axis label.'''

    values = np.asarray(counts, dtype=np.int64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("count histogram requires a non-empty one-dimensional sample")
    minimum = int(values.min())
    maximum = int(values.max())
    span = maximum - minimum
    # Remove a common integer offset before any float conversion so adjacent
    # large int64 counts cannot collapse onto the same binary64 value.
    centered = values - minimum
    label = "event count" if minimum == 0 else "event count minus {:,}".format(minimum)
    if span + 1 <= maximum_unit_bins:
        return centered, range(span + 2), label
    bounded_bins = min(50, max(10, int(np.sqrt(len(values)))))
    return centered.astype(float), bounded_bins, label


def loss_histogram_spec(losses, bin_count=30):
    '''Return finite, strictly increasing cost bins and any displayed offset.'''

    values = np.asarray(losses, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("loss histogram requires a non-empty finite one-dimensional sample")
    minimum = float(values.min())
    maximum = float(values.max())
    lower = minimum if minimum != maximum else minimum - 0.5
    upper = maximum if minimum != maximum else maximum + 0.5
    with np.errstate(over="ignore", invalid="ignore"):
        edges = np.linspace(lower, upper, bin_count + 1)
    if np.isfinite(edges).all() and np.all(np.diff(edges) > 0):
        return values, edges, "policy cost", 0.0
    # A large common offset can make distinct or padded bin edges collapse.
    # Center only this plotting copy; reported costs remain unchanged.
    centered = values - minimum
    return centered, bin_count, "policy cost minus " + repr(minimum), minimum


def show_analysis(results, run_context):
    fit, counts, nominal, benders, stratified, mixture, robust, epsilon, evaluation = results
    print("Run snapshot")
    print("  BOM source: " + run_context["bom_source"])
    print("  Series source: " + run_context["data_source"])
    print(
        "  alpha={alpha:.2f} | risk weight={risk_weight:.2f} | scenarios={scenarios} | "
        "lead-time cost/day={lead_time_cost_per_day:.2f} | maximum lead time={max_lead_time_days} | "
        "robustness multiplier={epsilon_multiplier:.2f}".format(**run_context)
    )
    lead_time_cost_per_day = run_context["lead_time_cost_per_day"]
    print("CIR proxy fit")
    print(
        "  a={:.4f}, b={:.4f}, eta={:.4f}, lambda0={:.4f}, Feller={}, regularized(a={}, b={})".format(
            fit.parameters.mean_reversion,
            fit.parameters.long_run_mean,
            fit.parameters.volatility,
            fit.parameters.initial_intensity,
            fit.parameters.feller_satisfied,
            fit.mean_reversion_regularized,
            fit.long_run_mean_regularized,
        )
    )
    print("  scenarios={} | mean shared-shock count={:.3f}".format(len(counts), counts.mean()))
    variants = [
        ("Standard mean + CVaR", nominal),
        ("Benders mean + CVaR", benders),
        ("Stratified scenarios", stratified),
        ("Finite CVaR mixture", mixture),
        ("W1 shock-count robustness", robust),
    ]
    rows = []
    policy_rows = []
    for label, result in variants:
        rows.append(
            {
                "Method": label,
                "Average cost": result.mean_cost,
                "Risk measure": result.risk_cost,
                "Combined objective": result.objective,
                "Labelled domestic (%)": 100.0 * result.labelled_domestic_share,
                "Solver status": result.status,
            }
        )
    print("\\nMethod comparison")
    display(pd.DataFrame(rows).set_index("Method").round(3))
    for label, result in variants:
        table = policy_table(result, lead_time_cost_per_day)
        table.insert(0, "Method", label)
        policy_rows.append(table)
    print("Selected options by method")
    display(pd.concat(policy_rows, ignore_index=True).round(3))
    replaced = [
        label
        for label, result in variants
        if result.details.get("solver_policy_replaced_by_certificate", False)
    ]
    if replaced:
        print(
            "Numerical safeguard: the exact rational global certificate replaced a solver-tolerance "
            "choice for " + ", ".join(replaced)
        )
    exact_fallbacks = [
        label for label, result in variants if result.status == "ExactOptimal"
    ]
    if exact_fallbacks:
        print(
            "ExactOptimal means no compatible numerical MILP result was used for "
            + ", ".join(exact_fallbacks)
            + "; the independent exact componentwise certificate supplied the global optimum "
            "for this documented flat scalar-shock model."
        )
    if benders.details.get("bypassed_for_zero_risk_weight", False):
        print("Benders was bypassed because risk weight is zero; the nominal mean-cost solve is exact.")
    elif benders.details.get("fallback_to_nominal", False):
        print(
            "Benders returned the certified nominal reference: "
            + benders.details.get("fallback_reason", "unspecified")
        )
    else:
        print("Benders reference verification: " + str(benders.details.get("reference_verified", False)))
    print(
        "Independent simulation evaluation of the nominal policy (not a real time-series holdout): "
        "mean={:.3f}, CVaR={:.3f}, objective={:.3f}".format(
            evaluation.mean_cost, evaluation.cvar, evaluation.objective
        )
    )
    exposure = robust.details.get("shock_exposure")
    tau = robust.details.get("tau")
    exposure_text = "not representable" if exposure is None else f"{exposure:.3f}"
    tau_text = w1_tau_text(tau, robust.details)
    print(f"W1 shock-count radius epsilon={epsilon:.5f}; exposure={exposure_text}; tau={tau_text}")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    count_values, count_bins, count_label = count_histogram_spec(counts)
    axes[0].hist(count_values, bins=count_bins, color="#2563eb", edgecolor="white")
    axes[0].set_title("Shared shock-count scenarios")
    axes[0].set_xlabel(count_label)
    loss_values, loss_bins, loss_label, loss_offset = loss_histogram_spec(nominal.losses)
    axes[1].hist(loss_values, bins=loss_bins, color="#059669", edgecolor="white")
    axes[1].axvline(nominal.mean_cost - loss_offset, color="#2563eb", linestyle="--", label="mean")
    axes[1].axvline(nominal.risk_cost - loss_offset, color="#dc2626", linestyle="--", label="CVaR")
    axes[1].set_title("Nominal policy costs")
    axes[1].set_xlabel(loss_label)
    axes[1].legend()
    plt.tight_layout()
    display(fig)
    plt.close(fig)"""
        ),
        markdown(
            """## Run a reproducible analysis

The defaults run entirely on the bundled synthetic examples. `CVaR confidence`
controls how far into the upper-cost tail the risk measure looks; `risk weight`
controls how strongly that tail affects the objective. Leave lead-time cost at
zero unless you have a defensible currency-per-day value.

The finite mixture below is exactly the listed CVaR levels and weights; it is
not a continuous spectral density. The W1 model is robust only to an ambiguity
set over scalar shock counts, and `epsilon` is in shock-count units. It is not
parameter uncertainty for the fitted CIR process. Each successful run prints a
path-safe input and parameter snapshot. Changing any input or control clears
the old result and requires a new run."""
        ),
        code(
            """CONTROL_STYLE = {"description_width": "initial"}
alpha_widget = widgets.FloatSlider(
    value=0.95,
    min=0.80,
    max=0.99,
    step=0.01,
    description="CVaR confidence",
    readout_format=".2f",
    continuous_update=False,
    style=CONTROL_STYLE,
)
risk_widget = widgets.FloatSlider(
    value=1.0,
    min=0.0,
    max=10.0,
    step=0.25,
    description="risk weight",
    continuous_update=False,
    style=CONTROL_STYLE,
)
scenario_widget = widgets.Dropdown(
    options=[("500", 500), ("1,000", 1000), ("2,000", 2000)],
    value=500,
    description="scenarios",
    style=CONTROL_STYLE,
)
lead_cost_widget = widgets.FloatSlider(
    value=0.0,
    min=0.0,
    max=25.0,
    step=0.5,
    description="lead-time cost/day",
    continuous_update=False,
    style=CONTROL_STYLE,
)
max_lead_widget = widgets.Dropdown(
    options=[("No limit", None), ("7 days", 7.0), ("14 days", 14.0), ("30 days", 30.0)],
    value=None,
    description="maximum lead time",
    style=CONTROL_STYLE,
)
epsilon_widget = widgets.FloatSlider(
    value=1.0,
    min=0.0,
    max=5.0,
    step=0.25,
    description="robustness multiplier",
    continuous_update=False,
    style=CONTROL_STYLE,
)
run_button = widgets.Button(description="Run validated analysis", button_style="success")
run_output = widgets.Output()
results_status = widgets.HTML(value='<b style="color:#475569">No analysis has been run yet.</b>')
analysis_has_run = False
analysis_is_current = False


def mark_results_stale(_=None):
    global analysis_is_current
    if not analysis_has_run or not analysis_is_current:
        return
    analysis_is_current = False
    results_status.value = (
        '<b style="color:#b91c1c">Inputs or controls changed; run validated analysis again.</b>'
    )
    with run_output:
        clear_output(wait=True)
        print("Previous results were cleared because inputs or controls changed. Run the analysis again.")


def on_run(_):
    global analysis_has_run, analysis_is_current
    run_button.disabled = True
    run_context = {
        "bom_source": BOM_SOURCE,
        "data_source": DATA_SOURCE,
        "alpha": alpha_widget.value,
        "risk_weight": risk_widget.value,
        "scenarios": scenario_widget.value,
        "lead_time_cost_per_day": lead_cost_widget.value,
        "max_lead_time_days": max_lead_widget.value,
        "epsilon_multiplier": epsilon_widget.value,
    }
    with run_output:
        clear_output(wait=True)
        print("Running validated models…")
        try:
            results = run_analysis(
                alpha_widget.value,
                risk_widget.value,
                scenario_widget.value,
                lead_cost_widget.value,
                max_lead_widget.value,
                epsilon_widget.value,
            )
            clear_output(wait=True)
            show_analysis(results, run_context)
            analysis_has_run = True
            analysis_is_current = True
            results_status.value = '<b style="color:#047857">Displayed results match the run snapshot below.</b>'
        except Exception as exc:
            clear_output(wait=True)
            print(safe_error_message(exc))
            analysis_has_run = True
            analysis_is_current = False
            results_status.value = '<b style="color:#b91c1c">Analysis failed; no current results.</b>'
        finally:
            run_button.disabled = False


run_button.on_click(on_run)
for control in (
    alpha_widget,
    risk_widget,
    scenario_widget,
    lead_cost_widget,
    max_lead_widget,
    epsilon_widget,
):
    control.observe(mark_results_stale, names="value")
display(
    widgets.VBox(
        [
            widgets.HTML("<b>Risk and scenario controls</b>"),
            widgets.HBox([alpha_widget, risk_widget, scenario_widget]),
            widgets.HTML("<b>Optional operational and robustness controls</b>"),
            widgets.HBox([lead_cost_widget, max_lead_widget, epsilon_widget]),
            run_button,
            results_status,
            run_output,
        ]
    )
)"""
        ),
        markdown(
            """## Interpretation checklist

- Verify that costs and `kappa` share one finished-unit basis and currency.
- Treat `type` as an input label, not a verified supplier geography.
- Choose lead-time economics from a real service-level model; zero cost per day
  means lead time is reported but not monetized.
- Treat synthetic proxy scenarios as stress-test inputs.
- The independent evaluation is a fresh simulation from the same fitted proxy,
  not real-world out-of-sample evidence."""
        ),
    ]
    return nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.9"},
        },
    )


def rendered_notebook():
    return nbformat.writes(build_notebook(), version=4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the checked-in notebook is stale")
    args = parser.parse_args()
    rendered = rendered_notebook()
    if args.check:
        if not TARGET.exists() or TARGET.read_text(encoding="utf-8") != rendered:
            raise SystemExit("notebooks/tailguard.ipynb is stale; run scripts/build_notebook.py")
        print("notebook is up to date")
    else:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=TARGET.parent,
                prefix=f".{TARGET.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                stream.write(rendered)
            temporary.replace(TARGET)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        print("wrote", TARGET.relative_to(ROOT))


if __name__ == "__main__":
    main()
