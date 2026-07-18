#!/usr/bin/env python3
"""Execute the notebook's analysis path with a temporary local series.

This exercises the analysis logic without saving an executed notebook or
modifying the tracked notebook source.
"""

from __future__ import annotations

import io
import os
import tempfile
import traceback
import warnings
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

import nbformat
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "tailguard.ipynb"
NOTEBOOK_LABEL = NOTEBOOK.relative_to(ROOT).as_posix()


def _execute_through(notebook, marker: str) -> None:
    """Run setup cells through a marker without running interactive widgets."""

    namespace: dict[str, object] = {"__name__": "__main__"}
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        exec(compile(cell.source, NOTEBOOK_LABEL, "exec"), namespace)
        if marker in cell.source:
            return
    raise RuntimeError(f"notebook did not contain setup marker {marker!r}")


def _assert_path_safe_startup_error(
    notebook, environment_name: str, missing_path: object, marker: str
) -> None:
    """Ensure an invalid configured path cannot survive traceback formatting."""

    previous_value = os.environ.get(environment_name)
    os.environ[environment_name] = str(missing_path)
    try:
        _execute_through(notebook, marker)
    except RuntimeError as exc:
        rendered = traceback.format_exc()
        path_safe_renderer = getattr(exc, "_render_traceback_", None)
        if not callable(path_safe_renderer):
            raise RuntimeError("notebook setup error lacks path-safe IPython rendering") from None
        ipython_rendered = "\n".join(path_safe_renderer())
        if (
            str(missing_path) in rendered
            or str(missing_path) in str(exc)
            or str(missing_path) in ipython_rendered
        ):
            raise RuntimeError("notebook setup error exposed a configured local path") from None
        if "Check its format and configuration" not in ipython_rendered:
            raise RuntimeError("notebook setup error omitted its actionable guidance") from None
    else:
        raise RuntimeError("notebook unexpectedly loaded a missing configured local CSV")
    finally:
        if previous_value is None:
            os.environ.pop(environment_name, None)
        else:
            os.environ[environment_name] = previous_value


def _exercise_in_memory_uploads(namespace: dict[str, object]) -> None:
    """Exercise notebook upload callbacks without opening a Jupyter socket."""

    def upload_value(name: str, content: bytes) -> tuple[dict[str, object], ...]:
        return (
            {
                "name": name,
                "type": "text/csv",
                "size": len(content),
                "content": memoryview(content),
                "last_modified": datetime.now(timezone.utc),
            },
        )

    bom_content = (
        "component,supplier,base_cost,kappa,lead_time,type\n"
        "A,offshore-option,10,4,12,offshore\n"
        "A,domestic-option,12,1,2,domestic\n"
    ).encode("utf-8")
    namespace["bom_upload"].value = upload_value("uploaded-bom.csv", bom_content)
    if namespace["BOM_SOURCE"] != "uploaded CSV":
        raise RuntimeError("notebook did not activate an uploaded BOM")
    if len(namespace["BOM"]["A"]) != 2:
        raise RuntimeError("notebook uploaded BOM did not reach the active analysis input")
    if namespace["bom_upload"].value:
        raise RuntimeError("notebook retained BOM upload bytes or filename in widget state")

    active_bom = namespace["BOM"]
    namespace["bom_upload"].value = upload_value("invalid-bom.csv", b"invalid\n")
    if namespace["BOM"] is not active_bom or "unchanged" not in namespace["bom_status"].value:
        raise RuntimeError("an invalid BOM upload replaced the last valid input")
    if namespace["bom_upload"].value:
        raise RuntimeError("notebook retained an invalid BOM upload in widget state")

    series_content = (
        "date,value\n"
        "2024-01-01,0.0\n"
        "2024-02-01,1.0\n"
        "2024-03-01,-0.5\n"
        "2024-04-01,0.75\n"
        "2024-05-01,0.2\n"
        "2024-06-01,0.4\n"
    ).encode("utf-8")
    namespace["series_upload"].value = upload_value("uploaded-series.csv", series_content)
    if namespace["DATA_SOURCE"] != "uploaded CSV":
        raise RuntimeError("notebook did not activate an uploaded series")
    if len(namespace["TRAINING_SERIES"]) != 6:
        raise RuntimeError("notebook uploaded series did not reach the active analysis input")
    if namespace["series_upload"].value:
        raise RuntimeError("notebook retained series upload bytes or filename in widget state")

    active_series = namespace["TRAINING_SERIES"]
    constant_series = (
        "date,value\n"
        "2024-01-01,1\n"
        "2024-02-01,1\n"
        "2024-03-01,1\n"
        "2024-04-01,1\n"
    ).encode("utf-8")
    namespace["series_upload"].value = upload_value("constant-series.csv", constant_series)
    if namespace["TRAINING_SERIES"] is not active_series or "unchanged" not in namespace["series_status"].value:
        raise RuntimeError("an unusable constant series replaced the last valid input")
    if namespace["series_upload"].value:
        raise RuntimeError("notebook retained a rejected series upload in widget state")

    ambiguous_dates = (
        "date,value\n"
        "01-02-03,1\n"
        "02-03-04,1\n"
        "03-04-05,1\n"
        "04-05-06,1\n"
    ).encode("utf-8")
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        namespace["series_upload"].value = upload_value(
            "ambiguous-series.csv", ambiguous_dates
        )
    if captured_warnings:
        raise RuntimeError("notebook ambiguous-date upload emitted a path-bearing warning")
    if namespace["TRAINING_SERIES"] is not active_series or "unchanged" not in namespace[
        "series_status"
    ].value:
        raise RuntimeError("an ambiguous-date series replaced the last valid input")
    if namespace["series_upload"].value:
        raise RuntimeError("notebook retained an ambiguous-date upload in widget state")

    safe_error_message = namespace["safe_error_message"]
    upload_error = namespace["UploadValidationError"]
    path_samples = (
        "/example",
        "/var/example/project/input.csv",
        "/opt/example/My Files/input.csv",
        "path:/srv/example/project/input.csv",
        "/srv/example/project/input.csv",
        "relative/example/project/input.csv",
        "project/input.csv",
        r"C:\example\project\input.csv",
        r"project\input.csv",
        r"\\server\sample-share\input.csv",
    )
    for path_sample in path_samples:
        rendered = safe_error_message(upload_error("could not read " + path_sample))
        if path_sample in rendered or "input.csv" in rendered:
            raise RuntimeError("notebook upload error exposed an arbitrary local path")
    unexpected = safe_error_message(RuntimeError("failure at /srv/example/project/input.csv"))
    if (
        "/srv/" in unexpected
        or "input.csv" in unexpected
        or "diagnostic details omitted" not in unexpected
    ):
        raise RuntimeError("notebook did not suppress an unexpected path-bearing error")

    oversized_content = memoryview(bytearray(namespace["MAX_UPLOAD_BYTES"] + 1))
    try:
        namespace["uploaded_csv_bytes"](
            type("Upload", (), {"value": ({"content": oversized_content},)})()
        )
    except namespace["UploadValidationError"] as exc:
        if "validation limit" not in str(exc):
            raise RuntimeError("notebook did not label its post-transfer upload cap precisely") from None
    else:
        raise RuntimeError("notebook accepted content over its validation cap")

    namespace["on_bom_reset"](None)
    namespace["on_series_reset"](None)
    if namespace["BOM_SOURCE"] != "bundled synthetic fixture":
        raise RuntimeError("BOM reset did not restore the bundled fixture")
    if namespace["DATA_SOURCE"] != "deterministic synthetic proxy":
        raise RuntimeError("series reset did not restore the synthetic fixture")


def main() -> None:
    os.chdir(ROOT)
    with tempfile.TemporaryDirectory() as temporary_directory:
        series_path = Path(temporary_directory) / "local-series.csv"
        pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=6, freq="MS"),
                "value": [0.0, 1.0, -0.5, 0.75, 0.2, 0.4],
            }
        ).to_csv(series_path, index=False)

        previous_bom_path = os.environ.get("TAILGUARD_BOM_CSV")
        previous_series_path = os.environ.get("TAILGUARD_SERIES_CSV")
        previous_matplotlib_dir = os.environ.get("MPLCONFIGDIR")
        os.environ.setdefault("MPLCONFIGDIR", str(Path(temporary_directory) / "matplotlib"))
        try:
            notebook = nbformat.read(NOTEBOOK, as_version=4)
            _assert_path_safe_startup_error(
                notebook,
                "TAILGUARD_BOM_CSV",
                Path(temporary_directory) / "missing-bom-path.csv",
                "LOCAL_BOM_PATH",
            )
            _assert_path_safe_startup_error(
                notebook,
                "TAILGUARD_SERIES_CSV",
                Path(temporary_directory) / "missing-series-path.csv",
                "LOCAL_SERIES_PATH",
            )
            _assert_path_safe_startup_error(
                notebook,
                "TAILGUARD_BOM_CSV",
                "~missing_notebook_account/input.csv",
                "LOCAL_BOM_PATH",
            )
            os.environ["TAILGUARD_SERIES_CSV"] = str(series_path)
            namespace: dict[str, object] = {"__name__": "__main__"}
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    exec(compile(cell.source, NOTEBOOK_LABEL, "exec"), namespace)
            if namespace["DATA_SOURCE"] != "configured local CSV":
                raise RuntimeError("notebook did not select the configured local series")
            _exercise_in_memory_uploads(namespace)
            results = namespace["run_analysis"](0.9, 1.0, 500, 0.0, None, 1.0)
            nominal = results[2]
            benders = results[3]
            if nominal.status not in {"Optimal", "ExactOptimal"}:
                raise RuntimeError("notebook nominal analysis did not solve to optimality")
            if not (benders.details.get("reference_verified") or benders.details.get("fallback_to_nominal")):
                raise RuntimeError("notebook Benders result was not independently certified")
            format_tau = namespace["w1_tau_text"]
            if format_tau(None, {"bypassed_for_zero_radius": True}) != (
                "not representable (irrelevant at zero radius)"
            ):
                raise RuntimeError("notebook mislabeled an unrepresentable zero-radius W1 tau")
            if format_tau(None, {"bypassed_for_zero_risk_weight": True}) != (
                "not representable (irrelevant to the objective at zero risk weight)"
            ):
                raise RuntimeError("notebook mislabeled an unrepresentable zero-risk-weight W1 tau")
            if format_tau(2.5, {}) != "2.500":
                raise RuntimeError("notebook did not format a representable W1 tau")
            histogram_spec = namespace["count_histogram_spec"]
            centered, bins, label = histogram_spec(
                namespace["np"].asarray(
                    [
                        namespace["np"].iinfo(namespace["np"].int64).max - 1,
                        namespace["np"].iinfo(namespace["np"].int64).max,
                    ],
                    dtype=namespace["np"].int64,
                )
            )
            if centered.tolist() != [0, 1] or list(bins) != [0, 1, 2] or "minus" not in label:
                raise RuntimeError("notebook histogram did not safely center a narrow large-count range")
            wide_values, wide_bins, wide_label = histogram_spec(
                namespace["np"].asarray([0, 100_000_000], dtype=namespace["np"].int64)
            )
            if len(wide_values) != 2 or not isinstance(wide_bins, int) or not 10 <= wide_bins <= 50:
                raise RuntimeError("notebook histogram did not bound a wide count range")
            if wide_label != "event count":
                raise RuntimeError("notebook histogram mislabeled uncentered counts")
            high_wide_values, high_wide_bins, high_wide_label = histogram_spec(
                namespace["np"].asarray(
                    [
                        namespace["np"].iinfo(namespace["np"].int64).max - 101,
                        namespace["np"].iinfo(namespace["np"].int64).max,
                    ],
                    dtype=namespace["np"].int64,
                )
            )
            if high_wide_values.tolist() != [0.0, 101.0] or not isinstance(high_wide_bins, int):
                raise RuntimeError("notebook histogram lost resolution in a wide large-count range")
            if "minus" not in high_wide_label:
                raise RuntimeError("notebook histogram mislabeled centered large counts")
            loss_histogram_spec = namespace["loss_histogram_spec"]
            centered_losses, loss_bins, loss_label, loss_offset = loss_histogram_spec(
                namespace["np"].asarray([1e308, 1e308], dtype=float)
            )
            if centered_losses.tolist() != [0.0, 0.0] or loss_bins != 30 or loss_offset != 1e308:
                raise RuntimeError("notebook histogram did not safely center a huge constant loss")
            if "minus" not in loss_label:
                raise RuntimeError("notebook histogram mislabeled centered policy costs")
            precise_offset = 1.0000000000000002e16
            _, _, precise_label, returned_offset = loss_histogram_spec(
                namespace["np"].asarray([precise_offset, precise_offset], dtype=float)
            )
            displayed_offset = float(precise_label.removeprefix("policy cost minus "))
            if returned_offset != precise_offset or displayed_offset != returned_offset:
                raise RuntimeError("notebook histogram rounded its displayed policy-cost offset")
            figure, axis = namespace["plt"].subplots()
            try:
                axis.hist(centered_losses, bins=loss_bins)
            finally:
                namespace["plt"].close(figure)
            snapshot_output = io.StringIO()
            with redirect_stdout(snapshot_output):
                namespace["show_analysis"](
                    results,
                    {
                        "bom_source": namespace["BOM_SOURCE"],
                        "data_source": namespace["DATA_SOURCE"],
                        "alpha": 0.9,
                        "risk_weight": 1.0,
                        "scenarios": 500,
                        "lead_time_cost_per_day": 0.0,
                        "max_lead_time_days": None,
                        "epsilon_multiplier": 1.0,
                    },
                )
            rendered_snapshot = snapshot_output.getvalue()
            for expected in (
                "BOM source: bundled synthetic fixture",
                "Series source: deterministic synthetic proxy",
                "alpha=0.90 | risk weight=1.00 | scenarios=500",
                "maximum lead time=None | robustness multiplier=1.00",
            ):
                if expected not in rendered_snapshot:
                    raise RuntimeError("notebook result omitted its path-safe run snapshot")
            namespace["analysis_has_run"] = True
            namespace["analysis_is_current"] = True
            namespace["risk_widget"].value = 1.25
            if namespace["analysis_is_current"] or "run validated analysis again" not in namespace[
                "results_status"
            ].value:
                raise RuntimeError("a changed control did not invalidate displayed analysis results")
            namespace["analysis_has_run"] = True
            namespace["analysis_is_current"] = True
            namespace["on_bom_reset"](None)
            if namespace["analysis_is_current"] or "run validated analysis again" not in namespace[
                "results_status"
            ].value:
                raise RuntimeError("a reset input did not invalidate displayed analysis results")
        finally:
            if previous_bom_path is None:
                os.environ.pop("TAILGUARD_BOM_CSV", None)
            else:
                os.environ["TAILGUARD_BOM_CSV"] = previous_bom_path
            if previous_series_path is None:
                os.environ.pop("TAILGUARD_SERIES_CSV", None)
            else:
                os.environ["TAILGUARD_SERIES_CSV"] = previous_series_path
            if previous_matplotlib_dir is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous_matplotlib_dir
    print("notebook analysis smoke test passed")


if __name__ == "__main__":
    main()
