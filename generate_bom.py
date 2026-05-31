#!/usr/bin/env python3
"""
generate_bom.py  --  Synthetic BOM generator for tailguard CVaR Reshoring Optimizer
=====================================================================================

Generates realistic example BOMs for different industries.
Output: CSV files ready to upload into the tailguard notebook.

Usage:
    python generate_bom.py                    # all industries -> data/bom/
    python generate_bom.py --industry auto    # only automotive
    python generate_bom.py --industry chem    # only chemicals (BASF-style)
    python generate_bom.py --industry semi    # only semiconductors
    python generate_bom.py --industry pharma  # only pharma
    python generate_bom.py --list             # show available industries
    python generate_bom.py --custom 8 3       # random BOM: 8 components, 3 options each

Columns generated:
    component   -- part group name
    supplier    -- supplier / route label
    base_cost   -- unit cost in normal times ($)
    kappa       -- cost shock per disruption event ($)
                   Rule of thumb: kappa ~ 0.3-1.5x base_cost for offshore
                                  kappa ~ 0.03-0.1x base_cost for domestic
    lead_time   -- delivery time in days
    type        -- offshore | domestic
"""

import argparse
import os
import numpy as np
import pandas as pd

# ── Industry templates ────────────────────────────────────────────────────────
# Each entry: (supplier_label, base_cost, kappa, lead_time, type)

INDUSTRIES = {

    "auto": {
        "name": "Automotive (Tesla / BMW style)",
        "desc": "EV powertrain + body components. High kappa on battery cells and semiconductors.",
        "components": {
            "Battery_Cell": {
                "offshore": [
                    ("CATL_CN_sea",      85,  62, 28),
                    ("Panasonic_JP_sea", 95,  22, 21),
                    ("LG_KR_air",       110,  18,  8),
                ],
                "domestic": [
                    ("Panasonic_NV",    128,   4),
                    ("Ultium_OH",       135,   5),
                ],
            },
            "Semiconductor": {
                "offshore": [
                    ("TSMC_TW_sea",     70,  85, 35),
                    ("Samsung_KR_air", 100,  28,  7),
                    ("Renesas_JP_air",  95,  30,  9),
                ],
                "domestic": [
                    ("TSMC_AZ",        140,   5),
                    ("Intel_OH",       155,   6),
                ],
            },
            "Structural_Steel": {
                "offshore": [
                    ("Baosteel_CN_sea",  40,  35, 32),
                    ("POSCO_KR_sea",     45,  20, 25),
                ],
                "domestic": [
                    ("Nucor_US",         65,   3),
                ],
            },
            "Drive_Unit": {
                "offshore": [
                    ("Nidec_CN_sea",   120,  55, 30),
                    ("Bosch_DE_air",   145,  15,  5),
                ],
                "domestic": [
                    ("Tesla_Fremont",  175,   4),
                ],
            },
            "Power_Electronics": {
                "offshore": [
                    ("Infineon_MY_sea",  42,  60, 33),
                    ("Rohm_JP_air",      55,  22,  8),
                ],
                "domestic": [
                    ("Wolfspeed_NC",     80,   3),
                ],
            },
        }
    },

    "chem": {
        "name": "Chemicals / Specialty Chemicals (BASF style)",
        "desc": "Precursors and specialty inputs with long offshore lead times.",
        "components": {
            "Ethylene_Oxide": {
                "offshore": [
                    ("SINOPEC_CN_sea",  210,  140, 40),
                    ("Saudi_Kayan_sea", 185,   90, 35),
                ],
                "domestic": [
                    ("BASF_Ludwigshafen", 280,  12),
                    ("LyondellBasell_DE", 295,  10),
                ],
            },
            "Rare_Earth_Oxides": {
                "offshore": [
                    ("China_Minmetals_sea", 320, 280, 45),
                    ("MP_Materials_sea",    290, 200, 38),
                ],
                "domestic": [
                    ("MP_Materials_CA",  480,  25),
                ],
            },
            "Specialty_Solvents": {
                "offshore": [
                    ("Wanhua_CN_sea",  95,  55, 30),
                    ("Eastman_SG_sea", 110,  40, 28),
                ],
                "domestic": [
                    ("Eastman_TN",    145,   8),
                    ("Evonik_DE",     150,   6),
                ],
            },
            "Polymer_Pellets": {
                "offshore": [
                    ("Sabic_SA_sea",   78,  48, 38),
                    ("Formosa_TW_sea", 72,  52, 36),
                ],
                "domestic": [
                    ("Dow_TX",        105,   5),
                ],
            },
            "Catalyst_Package": {
                "offshore": [
                    ("Umicore_BE_air",  850,  120,  7),
                    ("Heraeus_CN_sea",  780,  200, 30),
                ],
                "domestic": [
                    ("BASF_Catalysts", 1100,  30),
                    ("Clariant_US",    1050,  25),
                ],
            },
        }
    },

    "semi": {
        "name": "Semiconductor Equipment / Fabless Design",
        "desc": "Advanced node wafers, substrates, and photomask inputs.",
        "components": {
            "300mm_Wafer": {
                "offshore": [
                    ("Siltronic_SG_sea",  185,  95, 30),
                    ("Sumco_JP_sea",      175,  80, 28),
                    ("SK_Siltron_KR_sea", 170,  85, 32),
                ],
                "domestic": [
                    ("GlobalWafers_TX", 250,  15),
                ],
            },
            "Photoresist_EUV": {
                "offshore": [
                    ("JSR_JP_air",       620, 350,  7),
                    ("TOK_JP_air",       590, 320,  8),
                ],
                "domestic": [
                    ("DuPont_US",        850,  40),
                ],
            },
            "CMP_Slurry": {
                "offshore": [
                    ("Fujimi_JP_sea",   310, 120, 25),
                    ("CMC_TW_sea",      290, 110, 28),
                ],
                "domestic": [
                    ("Entegris_US",     420,  18),
                ],
            },
            "Deposition_Gas": {
                "offshore": [
                    ("Air_Liquide_JP_sea", 125,  80, 30),
                    ("Showa_Denko_sea",    115,  75, 28),
                ],
                "domestic": [
                    ("Linde_US",         180,  10),
                    ("Air_Products_US",  175,   8),
                ],
            },
            "Substrate_PKG": {
                "offshore": [
                    ("Ibiden_JP_sea",   420, 200, 35),
                    ("Unimicron_TW_sea",390, 220, 38),
                ],
                "domestic": [
                    ("TTM_Tech_US",     580,  20),
                ],
            },
        }
    },

    "pharma": {
        "name": "Pharmaceuticals / API supply",
        "desc": "Active Pharmaceutical Ingredients with heavy China/India offshore dependency.",
        "components": {
            "API_Core": {
                "offshore": [
                    ("Zhejiang_Huahai_CN_sea",  45,  38, 55),
                    ("Sun_Pharma_IN_sea",        50,  32, 48),
                    ("Aurobindo_IN_sea",         48,  35, 50),
                ],
                "domestic": [
                    ("Pfizer_Ringaskiddy",  120,   8),
                    ("Lonza_US",            135,   6),
                ],
            },
            "Excipients": {
                "offshore": [
                    ("Roquette_CN_sea",   22,  15, 42),
                    ("Ashland_IN_sea",    24,  12, 40),
                ],
                "domestic": [
                    ("Roquette_FR",  38,   3),
                    ("IMCD_NL",      40,   4),
                ],
            },
            "Solvent_GMP": {
                "offshore": [
                    ("Merck_CN_sea",  88,  45, 38),
                    ("Avantor_IN_sea", 82,  50, 40),
                ],
                "domestic": [
                    ("Merck_DE",      130,   5),
                ],
            },
            "Packaging_Primary": {
                "offshore": [
                    ("Gerresheimer_CN_sea", 65,  40, 35),
                    ("SGD_IN_sea",          60,  38, 38),
                ],
                "domestic": [
                    ("Gerresheimer_DE",  95,   4),
                    ("Schott_DE",        98,   5),
                ],
            },
            "Cold_Chain_Logistics": {
                "offshore": [
                    ("Kerry_Logistics_HK_air", 180,  90, 3),
                    ("DHL_Supply_CN_air",      195, 100, 2),
                ],
                "domestic": [
                    ("Cryoport_US",    240,  12),
                    ("World_Courier",  255,  10),
                ],
            },
        }
    },
}


# ── Random BOM generator ──────────────────────────────────────────────────────

def generate_random_bom(n_components=6, n_options_per=3, seed=42):
    """
    Generate a fully randomised synthetic BOM.
    Useful for stress-testing the optimizer with arbitrary problem sizes.
    """
    rng = np.random.default_rng(seed)
    component_names = [
        "Component_{:02d}".format(i+1) for i in range(n_components)
    ]
    regions_off = ["CN", "TW", "KR", "JP", "SG", "MY", "MX", "IN"]
    regions_dom = ["US", "DE", "FR", "UK", "CA"]
    routes      = ["sea", "air", "truck"]

    rows = []
    for comp in component_names:
        # base cost: uniform $30-$300
        base = int(rng.integers(30, 300))
        # offshore options
        for j in range(n_options_per - 1):
            region  = rng.choice(regions_off)
            route   = rng.choice(routes)
            b_off   = int(base * rng.uniform(0.6, 1.0))
            kappa   = int(b_off * rng.uniform(0.3, 1.2))
            lt      = int(rng.integers(5, 45)) if route != "truck" else int(rng.integers(2, 10))
            supplier = "{}_{}_{}_{}".format(comp[:4], region, route, j+1)
            rows.append([comp, supplier, b_off, kappa, lt, "offshore"])
        # domestic option (1 per component)
        region_d = rng.choice(regions_dom)
        b_dom    = int(base * rng.uniform(1.1, 1.8))
        kappa_d  = int(b_dom * rng.uniform(0.02, 0.08))
        lt_d     = int(rng.integers(1, 7))
        supplier_d = "{}_{}_dom".format(comp[:4], region_d)
        rows.append([comp, supplier_d, b_dom, kappa_d, lt_d, "domestic"])

    df = pd.DataFrame(rows, columns=["component","supplier","base_cost","kappa","lead_time","type"])
    return df


# ── Industry BOM to DataFrame ─────────────────────────────────────────────────

def industry_to_df(industry_key):
    ind  = INDUSTRIES[industry_key]
    rows = []
    for comp, d in ind["components"].items():
        for sup, base, kappa, lt in d["offshore"]:
            rows.append([comp, sup, base, kappa, lt, "offshore"])
        for item in d["domestic"]:
            sup, base, kappa = item
            rows.append([comp, sup, base, kappa, 1, "domestic"])
    return pd.DataFrame(rows, columns=["component","supplier","base_cost","kappa","lead_time","type"])


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic BOM CSVs for tailguard CVaR notebook",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--industry", "-i", choices=list(INDUSTRIES.keys()) + ["all"],
                        default="all", help="Which industry to generate")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List available industries and exit")
    parser.add_argument("--custom", "-c", nargs=2, type=int, metavar=("N_COMP","N_OPT"),
                        help="Generate random BOM with N_COMP components and N_OPT options each")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for --custom (default: 42)")
    parser.add_argument("--out", "-o", default="data/bom",
                        help="Output directory (default: data/bom/)")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable industries:")
        for k, v in INDUSTRIES.items():
            print("  {:8s}  {}".format(k, v["name"]))
            print("           {}".format(v["desc"]))
        return

    os.makedirs(args.out, exist_ok=True)

    if args.custom:
        n_comp, n_opt = args.custom
        print("Generating random BOM: {} components x {} options/each (seed={})".format(
            n_comp, n_opt, args.seed))
        df = generate_random_bom(n_components=n_comp, n_options_per=n_opt, seed=args.seed)
        path = os.path.join(args.out, "bom_random_{}_comp_{}_opt.csv".format(n_comp, n_opt))
        df.to_csv(path, index=False)
        _print_summary(df, path)
        return

    targets = list(INDUSTRIES.keys()) if args.industry == "all" else [args.industry]
    for key in targets:
        ind  = INDUSTRIES[key]
        df   = industry_to_df(key)
        path = os.path.join(args.out, "bom_{}.csv".format(key))
        df.to_csv(path, index=False)
        _print_summary(df, path, title=ind["name"])

    # Also write a template CSV with comments
    template_path = os.path.join(args.out, "bom_template.csv")
    with open(template_path, "w") as f:
        f.write("# tailguard BOM template\n")
        f.write("# kappa = extra cost per disruption event\n")
        f.write("# type  = offshore | domestic\n")
        f.write("component,supplier,base_cost,kappa,lead_time,type\n")
        f.write("Component_A,Supplier_Offshore_1,100,60,30,offshore\n")
        f.write("Component_A,Supplier_Domestic_1,140,5,2,domestic\n")
    print("\nTemplate written to: {}".format(template_path))
    print("\nUpload any of these CSVs into the BOM widget in the notebook.")


def _print_summary(df, path, title=None):
    if title:
        print("\n[ {} ]".format(title))
    print("  Written: {}  ({} rows, {} components)".format(
        path, len(df), df["component"].nunique()))
    print("  {:<24} {:>8} {:>8} {:>10}  Offshore suppliers".format(
        "Component", "base_off", "kappa_off", "lead_off"))
    print("  " + "-" * 70)
    for comp, grp in df.groupby("component"):
        off = grp[grp["type"] == "offshore"]
        dom = grp[grp["type"] == "domestic"]
        if len(off):
            print("  {:<24} {:>8.0f} {:>8.0f} {:>9.0f}d  {} (dom: {})".format(
                comp,
                off["base_cost"].mean(),
                off["kappa"].mean(),
                off["lead_time"].mean(),
                ", ".join(off["supplier"].tolist()),
                ", ".join(dom["supplier"].tolist()),
            ))


if __name__ == "__main__":
    main()
