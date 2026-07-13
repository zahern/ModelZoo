# ModelZoo

Streamlit GUI for configuring and running [SearchLibrium](https://pypi.org/project/SearchLibrium/)
and [metacountregressor](https://github.com/zahern/MetaCount) searches — with one-click local
runs and HPC (PBS) job generation.

## Architecture

The app itself only needs `streamlit` + `pandas` (installed in `.venv/`). All actual model
fitting/search runs happen in a **separate Python interpreter** — the "engine" — that has
SearchLibrium, metacountregressor, and JAX installed (defaults to
`Z:\test_runs_tours\code\.venv\Scripts\python.exe`, configurable in the app sidebar).

The app never imports SearchLibrium/metacountregressor/JAX directly: for every run it generates
a standalone `.py` script and shells out to the engine interpreter, streaming its stdout back
into a console panel. The same generated script is what gets bundled into the HPC job export
(paired with a PBS job file matching this project's cluster conventions).

## Getting started

### Prerequisites

- **Python 3.10+** to run the GUI itself (only needs `streamlit` + `pandas` — see
  [requirements.txt](requirements.txt)).
- **An "engine" Python interpreter** with `SearchLibrium`, `metacountregressor`, and `jax`
  installed. This is a *separate* interpreter from the one running the GUI — see
  [Architecture](#architecture) above. If you don't have one yet:

  ```powershell
  python -m venv engine-venv
  engine-venv\Scripts\pip install SearchLibrium metacountregressor jax jaxlib jaxopt
  ```

  If you're working from the QUT SEQ ABM pipeline, the engine venv already exists at
  `Z:\test_runs_tours\code\.venv` (or the equivalent path on your machine) — that's the
  default the app sidebar starts with.
- **The ABM pipeline code directory** (only needed for the ABM page) — defaults to
  `Z:\test_runs_tours\code`; point the page at a different path if yours lives elsewhere.

### 1. Clone and set up the GUI's own environment

```powershell
git clone https://github.com/zahern/ModelZoo.git
cd ModelZoo
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

### 2. Launch

```powershell
.venv\Scripts\python.exe -m streamlit run app\Home.py
```

Streamlit prints a local URL (defaults to `http://localhost:8501`) — open it in a browser.
Running headless on a remote box (no browser to auto-open)? Add
`--server.headless true --server.port 8501` to the command above.

**Or just double-click `RunModelZoo.exe`** in the repo root — it starts Streamlit using the
`.venv` next to it and opens the app in your default browser automatically. It's a thin
launcher (~8 MB, doesn't bundle Streamlit/pandas itself), so `.venv` must already be set up
per step 1 first. If you change `launcher.py`, rebuild it with:

```powershell
.venv\Scripts\python.exe build_exe.py
```

### 3. Point it at your engine interpreter

On the **Home** page, the sidebar has an "Engine Python interpreter" field, prefilled with the
default engine venv path. Change it if yours lives elsewhere, then click **Check engine** — it
imports SearchLibrium/metacountregressor/JAX in that interpreter (first check can take
30-90s, mostly JAX) and shows a pass/fail grid per package.

Below the status grid, **Update engine packages** runs `pip` inside that same interpreter — pick
SearchLibrium / metacountregressor / the jax stack, either "PyPI (upgrade to latest release)" or
"Local source (editable install)" (prefilled with this machine's dev checkouts under
`C:\Users\ahernz\source\...`, override the path for a different setup), and **Run update** streams
the pip output live. Re-run **Check engine** afterwards to confirm the new version landed.

### 4. Try a page

Each of the three tool pages (**SearchLibrium**, **MetaCountRegressor**, **ABM Pipeline**) follows
the same shape: configure → preview the generated script/command → **Run locally now** (streams
console output live) or export an HPC job. The fastest way to see it work end-to-end without
any of your own data: open the **SearchLibrium** or **MetaCountRegressor** page and pick
"Bundled example dataset" / "Bundled Example 16-3 dataset" as the data source, leave the
defaults, and click **Run locally now**.

### Stopping the app

Ctrl+C in the terminal running `streamlit run`, or close the terminal/process.

## Layout

- `app/Home.py` — landing page, engine interpreter configuration + smoke-test status
- `app/pages/1_SearchLibrium.py` — discrete-choice model search runner
- `app/pages/2_MetaCountRegressor.py` — count/CMF/duration/linear model search runner
- `app/pages/3_ABM.py` — ABM pipeline runner (`Z:\test_runs_tours\code`): modes, strategies, HPC commands
- `app/lib/env.py` — engine interpreter discovery + package probing
- `app/lib/abm.py` — ABM mode/strategy metadata + local/qsub command builders
- `app/lib/script_gen.py` — generates the standalone Python run-scripts
- `app/lib/pbs_gen.py` — generates matching PBS job files
- `app/lib/runner.py` — subprocess execution with streamed output
- `app/lib/ui_common.py` — shared preview/run/export UI
- `generated_jobs/` — local run outputs and uploaded data (gitignored)

## ABM page

The ABM page drives `pipeline_logger.py` in the ABM code directory (default
`Z:\test_runs_tours\code`, configurable on the page). It covers every pipeline mode —
`main`, `ga`, `ga_staged`, `ga_stage5_smoke`, `mcr_search`, `hhts_core`, `hhts_search`,
`safety_baseline`/`safety_nosafety`/`safety_compare`, `list_searches` — plus the strategy
knobs for each:

- **GA modes**: estimator (`GA_ESTIMATOR`: default/metacount/searchlibrium/both/hybrid/sa_bandit),
  budget (`GA_BUDGET`: smoke → thorough), restarts, bandit-guided SA (`GA_USE_BANDIT_SA`).
- **HHTS modes**: `--search` preset (core_fixed, nested_fast, nested_standard, mnl_fast,
  selection_screen, survival_screen) with `--sa-iter` / `--sa-temp` / `--sa-model` overrides.
- **All modes**: zone selector, run tag (`PIPELINE_RUN_TAG`), CPU/GPU accelerator, thread caps.

Local runs stream the pipeline console into the page. The HPC tab renders ready-to-paste
`qsub` / `submit_pipeline_runs.sh` commands (single mode, sequential batch, safety chain, and
the GA parallel stage fan-out chain) matching PBS_RUN_GUIDE.md in the code directory.

## Constraints dashboards

Both the **SearchLibrium** and **MetaCountRegressor** pages have a "Constraints" section that
maps onto each package's real constraint-builder API — including **mutually-exclusive groups**
(pick 2+ variables, at most one may appear in the search's final structure):

- **SearchLibrium** → `ConstraintBuilder`: force include / never include, mutually-exclusive
  groups, minimum-behavioural-content pools, force/exclude random parameters (with a
  distribution picker).
- **MetaCountRegressor** → `ModelConstraints`: force include/fixed, never random, never
  zero-inflation, exclude, membership-only/allow-membership/outcome-only, allow-random with a
  restricted distribution set, and mutual-exclusion groups. Note: due to a verified upstream
  limitation, `ModelConstraints` is only merged into the search for `model_family='count'` —
  the page shows a warning if you pick `duration`/`linear` with constraints set.

Group-based constraints (mutually-exclusive groups, minimum-behavioural pools) use an "add
group" widget — pick 2+ variables and click Add; each group appears as its own removable row.

## Notes on package API surface

Both packages' READMEs have occasionally been ahead of what's actually importable in a given
installed version — e.g. `ExperimentBuilder.run()` returns a plain `dict` (not an object with
`.best_score`), so the generated metacountregressor scripts go through the documented
`extract_search_best()` / `extract_summary()` / `evaluator.build_spec()` helpers instead.
SearchLibrium's bundled example datasets (`load_electricity_data()`, `load_travel_mode_data()`,
`load_swiss_metro_data()`) were added to the SearchLibrium source but, as of this writing, not
yet published to PyPI — install from source (`pip install -e path/to/SearchLibrium`) in the
engine venv to use the "Bundled example dataset" option on that page; otherwise use "Upload CSV"
/ "Path on disk" instead until a release ships it. If you upgrade either package, re-check
`app/lib/script_gen.py` against the new signatures before trusting generated scripts.

**SearchLibrium `pre_spec_constraints` bug (fixed in source, not yet on PyPI):** while building
the constraints dashboard we found `ConstraintBuilder`-based constraints (`force_include`,
`mutually_exclusive`, `min_behavioral`, `force_random`, `never_random`) had **zero effect** on
any search prior to this fix — `Search.apply_constraints()` and its helpers checked
`self.pres_spec_constr`, an attribute that only ever exists on the `Parameters` object
(`self.param.pres_spec_constr`), never on the solver itself, so `hasattr()` was always `False`
and every constraint silently no-op'd. Also `apply_constraints()` was never called at all from
the plain multinomial/mixed-logit/RRM evaluation path (only latent-class/nested/mixed-nested
routed through it). Both are fixed in `search.py` in the local source checkout; verified with a
real search where a `mutually_exclusive(['COST','HEADWAY'])` group is now actually enforced
(previously both appeared together in the result regardless). Until a new PyPI release ships
this, constraints only work with an editable/source install of SearchLibrium.
