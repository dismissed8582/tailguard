import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_relative_links_resolve():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    targets = re.findall(r"\]\(([^)]+)\)", readme)
    for target in targets:
        if "://" in target or target.startswith("#"):
            continue
        path = target.split("#", maxsplit=1)[0]
        assert (ROOT / path).exists(), target


def test_archived_result_artifacts_are_explicitly_marked_unreproducible():
    archive_note = (ROOT / "results" / "README.md").read_text(encoding="utf-8")
    assert "historical artifacts" in archive_note
    assert "must not be used" in archive_note
    audit = (ROOT / "LEGACY_ARTIFACT_AUDIT.md").read_text(encoding="utf-8")
    assert "No independently identifiable source was found" in audit
    assert "do not follow from the model stated in Equation (26)" in audit


def test_readme_leads_with_a_runnable_and_safe_local_input_flow():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert readme.index("## Synthetic quickstart") < readme.index("## What the model actually does")
    assert 'python -m pip install -e ".[notebook]"' in readme
    assert "trusted local kernel" in readme
    assert "Clear Notebook Widget State" in readme
    assert "short-lived model and solution files" in readme
