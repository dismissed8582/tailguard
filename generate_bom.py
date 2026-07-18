#!/usr/bin/env python3
"""
generate_bom.py  --  Synthetic sourcing-option generator for Tailguard
====================================================================

Generates illustrative synthetic sourcing-option tables for different industries.
Output: CSV files ready for local selection with ``TAILGUARD_BOM_CSV`` without
embedding their paths in notebook source.
Existing files are not overwritten unless --force is supplied.

Usage:
    python generate_bom.py                    # all industries -> generated/bom/
    python generate_bom.py --industry auto    # only automotive
    python generate_bom.py --industry chem    # only chemicals
    python generate_bom.py --industry semi    # only semiconductors
    python generate_bom.py --industry pharma  # only pharma
    python generate_bom.py --list             # show available industries
    python generate_bom.py --custom 8 3       # random BOM: 8 components, 3 options each

Columns generated:
    component   -- part group name
    supplier    -- synthetic option label
    base_cost   -- unit cost in normal times (one user-defined currency)
    kappa       -- illustrative cost shock per disruption event (same basis)
    lead_time   -- delivery time in days
    type        -- abstract offshore | domestic source-class label

All names, costs, and classifications are synthetic examples. They are not
supplier facts, sourcing recommendations, or evidence of affiliation.
"""

import argparse
import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "generated" / "bom"

# ── Industry templates ────────────────────────────────────────────────────────
# Each option tuple contains only synthetic numerical parameters
# (base_cost, kappa, lead_time).  There are deliberately no embedded supplier,
# country, or route identifiers for users to mistake for source data.

INDUSTRIES = {
    "auto": {
        "name": "Automotive components (illustrative)",
        "desc": "Illustrative EV powertrain and body components with varied shock exposure.",
        "components": {
            "Battery_Cell": {
                "offshore": [
                    (85, 62, 28),
                    (95, 22, 21),
                    (110, 18, 8),
                ],
                "domestic": [
                    (128, 4, 1),
                    (135, 5, 1),
                ],
            },
            "Semiconductor": {
                "offshore": [
                    (70, 85, 35),
                    (100, 28, 7),
                    (95, 30, 9),
                ],
                "domestic": [
                    (140, 5, 1),
                    (155, 6, 1),
                ],
            },
            "Structural_Steel": {
                "offshore": [
                    (40, 35, 32),
                    (45, 20, 25),
                ],
                "domestic": [
                    (65, 3, 1),
                ],
            },
            "Drive_Unit": {
                "offshore": [
                    (120, 55, 30),
                    (145, 15, 5),
                ],
                "domestic": [
                    (175, 4, 1),
                ],
            },
            "Power_Electronics": {
                "offshore": [
                    (42, 60, 33),
                    (55, 22, 8),
                ],
                "domestic": [
                    (80, 3, 1),
                ],
            },
        },
    },
    "chem": {
        "name": "Chemicals / specialty chemicals (illustrative)",
        "desc": "Precursors and specialty inputs with long offshore lead times.",
        "components": {
            "Ethylene_Oxide": {
                "offshore": [
                    (210, 140, 40),
                    (185, 90, 35),
                ],
                "domestic": [
                    (280, 12, 1),
                    (295, 10, 1),
                ],
            },
            "Rare_Earth_Oxides": {
                "offshore": [
                    (320, 280, 45),
                    (290, 200, 38),
                ],
                "domestic": [
                    (480, 25, 1),
                ],
            },
            "Specialty_Solvents": {
                "offshore": [
                    (95, 55, 30),
                    (110, 40, 28),
                ],
                "domestic": [
                    (145, 8, 1),
                    (150, 6, 1),
                ],
            },
            "Polymer_Pellets": {
                "offshore": [
                    (78, 48, 38),
                    (72, 52, 36),
                ],
                "domestic": [
                    (105, 5, 1),
                ],
            },
            "Catalyst_Package": {
                "offshore": [
                    (850, 120, 7),
                    (780, 200, 30),
                ],
                "domestic": [
                    (1100, 30, 1),
                    (1050, 25, 1),
                ],
            },
        },
    },
    "semi": {
        "name": "Semiconductor supply (illustrative)",
        "desc": "Advanced node wafers, substrates, and photomask inputs.",
        "components": {
            "300mm_Wafer": {
                "offshore": [
                    (185, 95, 30),
                    (175, 80, 28),
                    (170, 85, 32),
                ],
                "domestic": [
                    (250, 15, 1),
                ],
            },
            "Photoresist_EUV": {
                "offshore": [
                    (620, 350, 7),
                    (590, 320, 8),
                ],
                "domestic": [
                    (850, 40, 1),
                ],
            },
            "CMP_Slurry": {
                "offshore": [
                    (310, 120, 25),
                    (290, 110, 28),
                ],
                "domestic": [
                    (420, 18, 1),
                ],
            },
            "Deposition_Gas": {
                "offshore": [
                    (125, 80, 30),
                    (115, 75, 28),
                ],
                "domestic": [
                    (180, 10, 1),
                    (175, 8, 1),
                ],
            },
            "Substrate_PKG": {
                "offshore": [
                    (420, 200, 35),
                    (390, 220, 38),
                ],
                "domestic": [
                    (580, 20, 1),
                ],
            },
        },
    },
    "pharma": {
        "name": "Pharmaceutical API supply (illustrative)",
        "desc": "Active pharmaceutical ingredients with deliberately high long-distance shock exposure.",
        "components": {
            "API_Core": {
                "offshore": [
                    (45, 38, 55),
                    (50, 32, 48),
                    (48, 35, 50),
                ],
                "domestic": [
                    (120, 8, 1),
                    (135, 6, 1),
                ],
            },
            "Excipients": {
                "offshore": [
                    (22, 15, 42),
                    (24, 12, 40),
                ],
                "domestic": [
                    (38, 3, 1),
                    (40, 4, 1),
                ],
            },
            "Solvent_GMP": {
                "offshore": [
                    (88, 45, 38),
                    (82, 50, 40),
                ],
                "domestic": [
                    (130, 5, 1),
                ],
            },
            "Packaging_Primary": {
                "offshore": [
                    (65, 40, 35),
                    (60, 38, 38),
                ],
                "domestic": [
                    (95, 4, 1),
                    (98, 5, 1),
                ],
            },
            "Cold_Chain_Logistics": {
                "offshore": [
                    (180, 90, 3),
                    (195, 100, 2),
                ],
                "domestic": [
                    (240, 12, 1),
                    (255, 10, 1),
                ],
            },
        },
    },
}


# ── Random BOM generator ──────────────────────────────────────────────────────


def generate_random_bom(n_components=6, n_options_per=3, seed=42):
    """
    Generate a fully randomised synthetic BOM.
    Useful for stress-testing the optimizer with arbitrary problem sizes.
    """
    if isinstance(n_components, bool) or not isinstance(n_components, int) or n_components < 1:
        raise ValueError("n_components must be at least 1")
    if isinstance(n_options_per, bool) or not isinstance(n_options_per, int) or n_options_per < 2:
        raise ValueError("n_options_per must be at least 2")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    component_names = ["Component_{:02d}".format(i + 1) for i in range(n_components)]
    routes = ["sea", "air", "truck"]

    rows = []
    for comp in component_names:
        # base cost: uniform $30-$300
        base = int(rng.integers(30, 300))
        # offshore options
        for j in range(n_options_per - 1):
            route = rng.choice(routes)
            b_off = int(base * rng.uniform(0.6, 1.0))
            kappa = int(b_off * rng.uniform(0.3, 1.2))
            lt = int(rng.integers(5, 45)) if route != "truck" else int(rng.integers(2, 10))
            supplier = "{}_offshore_{}_{}".format(comp, route, j + 1)
            rows.append([comp, supplier, b_off, kappa, lt, "offshore"])
        # domestic option (1 per component)
        b_dom = int(base * rng.uniform(1.1, 1.8))
        kappa_d = int(b_dom * rng.uniform(0.02, 0.08))
        lt_d = int(rng.integers(1, 7))
        supplier_d = "{}_domestic_1".format(comp)
        rows.append([comp, supplier_d, b_dom, kappa_d, lt_d, "domestic"])

    df = pd.DataFrame(rows, columns=["component", "supplier", "base_cost", "kappa", "lead_time", "type"])
    return df


# ── Industry BOM to DataFrame ─────────────────────────────────────────────────


def _synthetic_supplier_label(component, source_type, ordinal):
    slug = "".join(character.lower() if character.isalnum() else "_" for character in component)
    slug = slug.strip("_")
    return "{}_{}_{}".format(slug, source_type, ordinal)


def industry_to_df(industry_key):
    ind = INDUSTRIES[industry_key]
    rows = []
    for comp, d in ind["components"].items():
        for ordinal, (base, kappa, lt) in enumerate(d["offshore"], start=1):
            rows.append([comp, _synthetic_supplier_label(comp, "offshore", ordinal), base, kappa, lt, "offshore"])
        for ordinal, (base, kappa, lt) in enumerate(d["domestic"], start=1):
            rows.append([comp, _synthetic_supplier_label(comp, "domestic", ordinal), base, kappa, lt, "domestic"])
    return pd.DataFrame(rows, columns=["component", "supplier", "base_cost", "kappa", "lead_time", "type"])


# ── CLI ───────────────────────────────────────────────────────────────────────


def _preflight_output_paths(paths, force=False):
    collisions = [Path(path) for path in paths if Path(path).exists()]
    if collisions and not force:
        listed = ", ".join(str(path) for path in collisions)
        raise FileExistsError(f"output already exists: {listed}; choose another --out or pass --force")


def _seed_filename_token(seed):
    """Keep ordinary seeds readable while bounding generated filename length."""

    text = str(seed)
    if len(text) <= 64:
        return text
    digest = hashlib.sha256(text.encode("ascii")).hexdigest()
    return "sha256_{}".format(digest)


def _output_error(parser, exc):
    """Report filesystem failures without echoing a configured output path."""

    if isinstance(exc, FileExistsError):
        parser.error("output already exists; choose another --out or pass --force")
    parser.error("could not access the output location; check --out and its permissions")


def _publish_temporary(temporary, path, force=False):
    """Publish a complete file atomically, without clobbering unless requested."""

    if force:
        temporary.replace(path)
    else:
        # The temporary file is created in path.parent, so this is one
        # filesystem. Creating the destination hard link is atomic and fails
        # if another process creates any entry at path after preflight.
        os.link(temporary, path)


def _write_csv(df, path, force=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError("{} already exists; choose another --out or pass --force".format(path))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write("# Synthetic Tailguard sourcing-option fixture; not supplier data or sourcing advice.\n")
            df.to_csv(stream, index=False)
        _publish_temporary(temporary, path, force=force)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _write_template(path, force=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError("{} already exists; choose another --out or pass --force".format(path))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(
                "# Synthetic Tailguard sourcing-option template; not supplier data or sourcing advice.\n"
                "# tailguard flat sourcing-option template\n"
                "# Costs and kappa must be normalized to one finished unit.\n"
                "# type is an input label: offshore or domestic.\n"
                "component,supplier,base_cost,kappa,lead_time,type\n"
                "Component_A,Supplier_Offshore_1,100,60,30,offshore\n"
                "Component_A,Supplier_Domestic_1,140,5,2,domestic\n"
            )
        _publish_temporary(temporary, path, force=force)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate synthetic BOM CSVs for tailguard CVaR notebook",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--industry", "-i", choices=list(INDUSTRIES.keys()) + ["all"], help="Which industry to generate (default: all)"
    )
    mode.add_argument("--list", "-l", action="store_true", help="List available industries and exit")
    mode.add_argument(
        "--custom",
        "-c",
        nargs=2,
        type=int,
        metavar=("N_COMP", "N_OPT"),
        help="Generate random BOM with N_COMP components and N_OPT options each",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for --custom (default: 42)")
    parser.add_argument(
        "--out", "-o", default=str(DEFAULT_OUTPUT_DIR), help="Output directory (default: repository/generated/bom)"
    )
    parser.add_argument("--force", action="store_true", help="Allow replacing an existing generated file")
    args = parser.parse_args(argv)

    if args.list:
        print("\nAvailable industries:")
        for k, v in INDUSTRIES.items():
            print("  {:8s}  {}".format(k, v["name"]))
            print("           {}".format(v["desc"]))
        return

    if args.custom:
        n_comp, n_opt = args.custom
        if n_comp < 1:
            parser.error("--custom N_COMP must be at least 1")
        if n_opt < 2:
            parser.error("--custom N_OPT must be at least 2 so every component has a choice")
        if args.seed < 0:
            parser.error("--seed must be non-negative")
        print("Generating random BOM: {} components x {} options/each (seed={})".format(n_comp, n_opt, args.seed))
        df = generate_random_bom(n_components=n_comp, n_options_per=n_opt, seed=args.seed)
        seed_token = _seed_filename_token(args.seed)
        path = Path(args.out) / "bom_random_{}_comp_{}_opt_seed_{}.csv".format(n_comp, n_opt, seed_token)
        try:
            _preflight_output_paths([path], force=args.force)
            _write_csv(df, path, force=args.force)
        except (OSError, ValueError) as exc:
            _output_error(parser, exc)
        _print_summary(df, path)
        return

    industry = args.industry or "all"
    targets = list(INDUSTRIES.keys()) if industry == "all" else [industry]
    plans = [(key, industry_to_df(key), Path(args.out) / "bom_{}.csv".format(key)) for key in targets]
    template_path = Path(args.out) / "bom_template.csv"
    try:
        _preflight_output_paths([path for _, _, path in plans] + [template_path], force=args.force)
    except (OSError, ValueError) as exc:
        _output_error(parser, exc)

    for key, df, path in plans:
        ind = INDUSTRIES[key]
        try:
            _write_csv(df, path, force=args.force)
        except (OSError, ValueError) as exc:
            _output_error(parser, exc)
        _print_summary(df, path, title=ind["name"])

    try:
        _write_template(template_path, force=args.force)
    except (OSError, ValueError) as exc:
        _output_error(parser, exc)
    print("\nTemplate written to: {}".format(template_path))
    print(
        "\nSet TAILGUARD_BOM_CSV in the Jupyter environment to select one of these CSVs "
        "without embedding its path in notebook source."
    )


def _print_summary(df, path, title=None):
    if title:
        print("\n[ {} ]".format(title))
    print("  Written: {}  ({} rows, {} components)".format(path, len(df), df["component"].nunique()))
    print(
        "  {:<24} {:>8} {:>8} {:>10}  Offshore-labeled options".format(
            "Component", "base_off", "kappa_off", "lead_off"
        )
    )
    print("  " + "-" * 70)
    for comp, grp in df.groupby("component"):
        off = grp[grp["type"] == "offshore"]
        dom = grp[grp["type"] == "domestic"]
        if len(off):
            print(
                "  {:<24} {:>8.0f} {:>8.0f} {:>9.0f}d  {} (domestic-labeled: {})".format(
                    comp,
                    off["base_cost"].mean(),
                    off["kappa"].mean(),
                    off["lead_time"].mean(),
                    ", ".join(off["supplier"].tolist()),
                    ", ".join(dom["supplier"].tolist()),
                )
            )


if __name__ == "__main__":
    main()
