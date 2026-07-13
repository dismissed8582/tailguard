# tailguard

Tailguard is a research prototype for a flat sourcing-choice problem with a
mean-plus-CVaR objective. It is designed to make the assumptions visible and
to provide a tested reference implementation for small-to-medium scenario
sets. It is not production sourcing software, financial advice, or evidence
that a particular supplier, region, or route is preferable.

## Synthetic quickstart

Tailguard includes a ready-to-run synthetic example. You do not need your own
data for the first run. You need Git and Python 3.9 or newer.

### macOS or Linux

```zsh
git clone https://github.com/dismissed8582/tailguard.git
cd tailguard
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[notebook]"
python -m jupyter notebook notebooks/tailguard.ipynb
```

On Apple Silicon, some PuLP wheels bundle an Intel-only CBC executable. Tailguard
prefers a native HiGHS or CBC installation when one is available. A missing or
incompatible MILP executable does not block this documented flat model: the
independent exact componentwise certificate returns the same optimum with
status `ExactOptimal`. To exercise the MILP and Benders implementations too,
install `brew install highs` (recommended) or `brew install cbc`, then restart
the notebook kernel. Rosetta is not required with a native solver.

### Windows PowerShell

```powershell
git clone https://github.com/dismissed8582/tailguard.git
cd tailguard
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[notebook]"
.\.venv\Scripts\python.exe -m jupyter notebook notebooks/tailguard.ipynb
```

In the notebook:

1. Select **Kernel → Restart Kernel and Run All Cells**.
2. Keep the bundled inputs for the first run, or choose your own CSVs in the
   optional upload panel.
3. Keep the default controls and select **Run validated analysis**.
4. Review the method-comparison table and the selected supplier options.

Each successful run prints path-safe input labels and a complete control
snapshot. Changing an input or control clears the previous results and marks
the analysis for rerun, so an old policy is not left looking current.

If you plan to use sensitive inputs, use a trusted local kernel and review
[Local file handling](#local-file-handling) before uploading them. Notebook
state, cell outputs, and the solver's short-lived model and solution files
require the ordinary cleanup precautions documented there.

## What the model actually does

For each component, the model chooses exactly one supplier option from a flat
CSV table. Given shared disruption-count scenarios $N^\omega$, an option has
scenario cost

\[
c_i + h\ell_i + \kappa_i N^\omega,
\]

where `c_i` is base cost, `ell_i` is lead time, `h` is an explicitly chosen
cost per lead-time day, and `kappa_i` is shock exposure. The default is `h=0`,
so lead time is reported but not monetized unless the user supplies
a defensible conversion. A hard maximum lead time can also be selected.

The nominal objective is

\[
\min_x\;\mathbb{E}[Z(x)] + \lambda\operatorname{CVaR}_\alpha[Z(x)],
\]

with one binary supplier choice per component. The normal path uses the
Rockafellar-Uryasev linearization and checks the MILP solver status. Because the
documented model is separable by component and every option is an affine,
non-decreasing function of one shared scalar shock count, an independent exact
rational certificate is authoritative. If the MILP cannot run or its numeric
result is unsafe, Tailguard returns the certified optimum with status
`ExactOptimal` and records the fallback reason in result details.

## Important limits

- This is a **flat sourcing-option table**, not a BOM-DAG. It does not model
  quantities, parent-child dependencies, capacities, assembly locations,
  currencies, duties, service-level probabilities, or multi-stage decisions.
- Every `base_cost` and `kappa` value must already be normalized to one finished
  unit in one currency. Tailguard does not infer units or quantities.
- `type` is an abstract source-class label (`offshore` or `domestic`). The
  bundled data define no home market, so their `domestic` label has no
  geographic, legal, or policy meaning. The UI reports it only as a labelled
  category count.
- The bundled names, costs, classifications, and scenario data are synthetic
  and illustrative. They are not supplier facts, affiliations, or sourcing
  recommendations.
- The default data series is deterministic synthetic proxy data. It is not NY
  Fed data or an observed disruption-event history. A proxy CIR fit must not be
  described as an empirical event-intensity calibration.
- The included companion PDF is a preprint/research note, not an executable
  specification or a source of validated performance claims. Its incomplete
  citation, unreproduced numerical claims, and a contradiction in its worked
  example are recorded in [the legacy-artifact audit](LEGACY_ARTIFACT_AUDIT.md).
  The tested code and this README define the supported behavior.
- `weighted_cvar` accepts every representable `0 < alpha < 1`. Float
  dual/subgradient weights require `alpha >= 1e-8`; the MILP formulations
  additionally require `alpha <= 1 - 1e-8`. These bounds keep the respective
  complement weights and solver coefficients representable without a
  misleading loss of tail mass.
- Projected-Euler simulation requires
  `mean_reversion * horizon_years / steps <= 1`. Increase `steps` when a
  coarser grid falls outside this supported non-overshooting range;
  drift and diffusion products or integration increments that fall outside
  binary64 range are rejected with actionable errors rather than silently
  discarded.

## Implemented, with explicit scope

- Validated BOM parsing that preserves identifier text, decimal costs, lead
  time, and source type; leading CSV documentation lines beginning with `#`
  are accepted.
- Nominal and scenario-weighted mean-plus-CVaR MILPs.
- Common per-component cost and exposure floors are removed before each MILP
  solve to prevent a large policy-independent offset from hiding a smaller
  real policy difference. Because this flat model uses one shared non-negative
  scalar shock count, every returned policy is also checked in exact rational
  arithmetic against an independent component-by-component global-optimality
  certificate. If solver tolerances hide a strict difference, the certified
  policy is returned and that replacement is recorded in result details. If a
  compatible solver cannot run, the same certificate supplies an explicit
  `ExactOptimal` fallback rather than failing an otherwise solvable flat model.
- A Benders/outer-approximation implementation whose globally valid risk cut
  uses empirical-CVaR coefficients evaluated by exact rational tail allocation
  for the comonotone per-option losses, including fractional mass at VaR. The
  cut is used only when those coefficients are safely representable. Every
  decomposed candidate is cross-checked against the independently certified
  nominal reference; an unclosed or numerically suspect decomposition returns
  an explicitly labelled reference fallback rather than an uncertified policy.
- Pilot-resampling stratification with correctly weighted mean and CVaR. This
  is a sampling convenience, not a claim of a fixed variance-reduction factor
  or an exact conditional-path/Neyman sampler.
- Finite, documented mixtures of CVaR levels. These are not represented as
  continuous spectral kernels or Wang transforms.
- A type-1 Wasserstein robust CVaR model **only over the scalar shock-count
  distribution**, using ground metric |(n-n')|. With nonnegative total
  shock exposure `K(x)`, its risk term is

  \[
  \widehat{\operatorname{CVaR}}_\alpha + \frac{\varepsilon K(x)}{1-\alpha}.
  \]

  `epsilon` is in shock-count units. This is not uncertainty over fitted CIR
  parameters.
- Independent simulation evaluation of a fixed chosen policy. It is not a
  real-world time-series holdout or a certified SAA confidence interval.

## Developer installation and verification

The quickstart above installs only what is needed to use the notebook. The
runtime package supports Python 3.9+, while the full development test suite
requires Python 3.10+ so it can use a currently maintained pytest release. CI
separately compiles and smoke-tests the runtime on Python 3.9.
To run the development checks from the repository root, install the additional
tools:

Ruff is the fast Python linter used by CI. In this repository it reports common
Python errors, undefined or unused names and imports, and import-order problems.
The command below only reports issues; it does not rewrite files.

```zsh
python -m pip install -e ".[notebook,dev]"
python -m ruff check tailguard tests scripts generate_bom.py
python -m pytest -q
```

The notebook is generated from reviewable source. Check that it has not drifted
after editing the builder:

```zsh
python3 scripts/build_notebook.py --check
```

Regenerate it intentionally with:

```zsh
python3 scripts/build_notebook.py
```

## Local file handling

The checked-in notebook has no machine-specific paths, saved outputs,
attachments, or saved widget state. The easiest way to use sensitive data is
its upload panel. It accepts a BOM CSV and a `date,value` series CSV and
validates each input before activation (including a trial proxy fit for the
series). It does not display the browser-supplied filename or path in analysis
output, clears the upload control after processing, and provides a reset button
for each input.

The notebook rejects a selected file larger than 5 MiB before making its own
bytes copy, but that check runs only after the browser has sent the selection
to the active kernel. It is a validation cap, not a browser/kernel transport,
storage, or transient-memory guarantee.

For repeatable automated use, set `TAILGUARD_BOM_CSV` or
`TAILGUARD_SERIES_CSV` in the Jupyter kernel environment before opening the
notebook. Tailguard's status and handled-error output label a configured local
file without printing its path, including under IPython's verbose traceback
mode. Do not type an absolute or machine-specific path into notebook source.
This protection is not a boundary against a user deliberately inspecting the
kernel environment, globals, or process state. A browser upload is sent to
whichever kernel is active, so do not use a remote or shared kernel for
sensitive data unless you trust its operator and storage.

Before sharing or committing a notebook that processed sensitive data:

1. Clear every cell output.
2. Use **Widgets → Clear Notebook Widget State** when that command is
   available; otherwise verify that the notebook's top-level metadata has no
   `widgets` entry.
3. Save the notebook, then restart the kernel and refresh the page without
   rerunning its cells.
4. Save again, close it, and inspect the file/diff for input values,
   outputs, filenames, paths, and widget metadata.

These steps reduce accidental disclosure; they are not a secure-erasure
guarantee for browser, kernel, autosave, backup, swap, or temporary-file data.

Build artifacts under `dist/` are ignored and this repository's workflow does
not upload them. Release archives should be generated from a tagged commit in
CI, with their headers and contents checked before publication.

## BOM input format

The required CSV columns are:

| Column | Meaning |
| --- | --- |
| `component` | Component identifier; one option is selected per component. |
| `supplier` | Unique option identifier within a component. |
| `base_cost` | Non-negative cost per finished unit, in one user-defined currency. |
| `kappa` | Non-negative extra cost per disruption count, on the same basis. |
| `lead_time` | Non-negative days. It matters only with an explicit cost/day or limit. |
| `type` | Exactly `offshore` or `domestic`; abstract source-class label only. |

`data/bom/bom_template.csv` is a parseable example template. Leading lines
beginning with `#` may document a file. Invalid types, duplicate
component/supplier pairs, empty names, non-finite values, and negative costs
fail validation.

Notebook uploads and local proxy-series CSV loaders require `date,value` with
at least four finite observations at distinct, unambiguous ISO-8601 dates or
timestamps. Every actual date interval is used in the Euler proxy fit. Only
callers using the programmatic
`fit_cir_proxy` DataFrame API may omit dates, and they must explicitly pass
`date_column=None` to request monthly spacing.

## Generate illustrative examples

The generator writes outside tracked fixture data by default and refuses to
overwrite existing output unless `--force` is supplied.

```zsh
python3 generate_bom.py --list
python3 generate_bom.py --industry auto
python3 generate_bom.py --custom 10 3 --seed 42
```

Generated files go to `generated/bom/` by default. The custom filename includes
the seed for provenance. All templates remain synthetic examples.

## Continuous verification

GitHub Actions runs linting and the regression suite on Python 3.10, 3.12, and
3.14, compile- and smoke-tests the runtime on Python 3.9, checks the declared
minimum core dependencies, validates the Windows quickstart path, checks
generated-notebook drift, executes the dependency-backed notebook, and installs
the built wheel and source distribution into separate clean virtual
environments before checking their dependencies and importing them.

## Repository layout

```text
tailguard/
├── tailguard/                 # validated model, risk, scenario, and evaluation code
├── tests/                     # mathematical and input-regression tests
├── notebooks/tailguard.ipynb  # generated thin interactive front end
├── scripts/build_notebook.py  # canonical notebook source and drift check
├── data/bom/                  # synthetic input fixtures
└── generate_bom.py            # safe synthetic fixture generator
```

## Verification philosophy

The test suite includes fractional-tail and tied-VaR CVaR cases, extreme valid
confidence levels, weighted stratified objectives, exhaustive-enumeration
comparisons for the nominal solver, randomized weighted Benders comparisons,
Wasserstein radius checks, CSV-header and BOM validation, deterministic
scenario generation, and generator CLI validation.

Before treating a result as decision support, supply a validated cost basis,
actual service/capacity constraints, documented data provenance, and an
independent domain review.

## References

- R. T. Rockafellar and S. Uryasev, “Optimization of Conditional Value-at-Risk,”
  *The Journal of Risk* 2(3), 21–41 (2000),
  [DOI: 10.21314/JOR.2000.038](https://doi.org/10.21314/JOR.2000.038).
- P. Mohajerin Esfahani and D. Kuhn, “Data-Driven Distributionally Robust
  Optimization Using the Wasserstein Metric: Performance Guarantees and
  Tractable Reformulations,” *Mathematical Programming* 171, 115–166 (2018),
  [DOI: 10.1007/s10107-017-1172-1](https://doi.org/10.1007/s10107-017-1172-1).

Licensed under the [MIT License](LICENSE).
