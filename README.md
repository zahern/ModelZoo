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

## Run

```powershell
.venv\Scripts\python.exe -m streamlit run app\Home.py
```

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

## Notes on package API surface

Both packages ship READMEs that are ahead of what's actually importable in the installed
versions pinned in the engine venv (SearchLibrium 0.0.128, metacountregressor 1.0.88) — e.g.
SearchLibrium's "bundled datasets" aren't in the installed wheel, and
`ExperimentBuilder.run()` returns a plain `dict` (not an object with `.best_score`), so the
generated metacountregressor scripts go through the documented `extract_search_best()` /
`extract_summary()` / `evaluator.build_spec()` helpers instead. If you upgrade either package,
re-check `app/lib/script_gen.py` against the new signatures before trusting generated scripts.
