import subprocess
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def test_generated_notebook_is_current_and_valid():
    subprocess.run(
        [sys.executable, "scripts/build_notebook.py", "--check"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    notebook = nbformat.read(ROOT / "notebooks" / "tailguard.ipynb", as_version=4)
    nbformat.validate(notebook)
    assert all(cell.execution_count is None for cell in notebook.cells if cell.cell_type == "code")
    assert all(not cell.outputs for cell in notebook.cells if cell.cell_type == "code")
    assert all(not cell.get("attachments", {}) for cell in notebook.cells)
    assert "widgets" not in notebook.metadata


def test_notebook_has_no_runtime_installer_or_automatic_source_fallback():
    notebook = nbformat.read(ROOT / "notebooks" / "tailguard.ipynb", as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assert "!pip" not in source
    assert "FRED_API_KEY" not in source
    assert "requests.get" not in source
    assert "TAILGUARD_BOM_CSV" in source
    assert "TAILGUARD_SERIES_CSV" in source
    assert "def load_configured_csv" in source
    assert "class PathSafeConfigurationError" in source
    assert "def _render_traceback_" in source
    assert "Check its format and configuration" in source
    assert "from None" in source
    assert 'LOCAL_SERIES_PATH, start="2017-01-01"' not in source
    assert 'print(f"Project root:' not in source
    assert "/absolute/path" not in source
    assert "run_analysis" in source
    assert "fallback_to_nominal" in source
    assert "policy_rows" in source
    assert "widgets.FileUpload" in source
    assert "load_bom_csv_bytes" in source
    assert "load_series_csv_bytes" in source
    assert 'BOM_SOURCE = "uploaded CSV"' in source
    assert 'DATA_SOURCE = "uploaded CSV"' in source
    assert "MAX_UPLOAD_BYTES" in source
    assert "forget_uploaded_file" in source
    assert "fit_cir_proxy(candidate)" in source
    assert "simulate_cox(candidate_fit.parameters" in source
    assert "diagnostic details omitted" in source
    assert "path details omitted" in source
    assert "def safe_error_message" in source
    assert "def mark_results_stale" in source
    assert "Displayed results match the run snapshot" in source
    assert '"bom_source": BOM_SOURCE' in source
    assert '"data_source": DATA_SOURCE' in source
    assert "not representable (irrelevant at zero radius)" in source
    assert "not representable (irrelevant to the objective at zero risk weight)" in source
    assert "def count_histogram_spec" in source
    assert "def loss_histogram_spec" in source
    assert "bins=range(int(counts.max()) + 2)" not in source
    assert "Upload rejected; current BOM unchanged" in source
    assert 'item.get("name")' not in source
    assert 'item["name"]' not in source
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, "notebooks/tailguard.ipynb", "exec")
