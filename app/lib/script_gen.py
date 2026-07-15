"""Generate standalone, runnable Python scripts that drive SearchLibrium and
metacountregressor via their documented public APIs. The same generated
script is used for a local subprocess run and for an HPC/PBS submission —
only the data-path handling differs (see `for_hpc`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


def _pylist(items: list[str]) -> str:
    return "[" + ", ".join(repr(i) for i in items) + "]"


# ─────────────────────────────────────────────────────────────────────────────
# SearchLibrium
# ─────────────────────────────────────────────────────────────────────────────

# Bundled example datasets shipped inside SearchLibrium itself (>=0.0.140) —
# see SearchLibrium/sample_data.py. No CSV needs to travel with the job: the
# generated script just imports the loader, on HPC included.
SEARCHLIBRIUM_BUNDLED_PRESETS: dict[str, dict] = {
    "swiss_metro": {
        "loader": "load_swiss_metro_data",
        "columns": ["custom_id", "alt", "FIRST", "PURPOSE", "LUGGAGE", "DEST", "CHOICE",
                    "MALE", "GROUP", "SURVEY", "TICKET", "AGE", "ID", "SP", "GA", "WHO",
                    "INCOME", "ORIGIN", "TIME", "COST", "HEADWAY", "SEATS", "AV"],
        "choice_col": "CHOICE", "alt_col": "alt", "choice_id_col": "custom_id",
        "ind_id_col": "ID", "base_alt": "SM",
        "choice_expr": 'df["CHOICE"].astype(int).values',
        "default_vars": ["TIME", "COST", "HEADWAY", "SEATS"],
    },
    "electricity": {
        "loader": "load_electricity_data",
        "columns": ["choice", "id", "alt", "pf", "cl", "loc", "wk", "tod", "seas", "chid"],
        "choice_col": "choice", "alt_col": "alt", "choice_id_col": "chid",
        "ind_id_col": "id", "base_alt": "1",
        "choice_expr": 'df["choice"].astype(int).values',
        "default_vars": ["pf", "cl", "loc", "wk", "tod", "seas"],
    },
    "travel_mode": {
        "loader": "load_travel_mode_data",
        "columns": ["individual", "mode", "choice", "wait", "vcost", "travel", "gcost", "income", "size"],
        "choice_col": "choice", "alt_col": "mode", "choice_id_col": "individual",
        "ind_id_col": None, "base_alt": "car",
        "choice_expr": '(df["choice"] == "yes").astype(int).values',
        "default_vars": ["wait", "vcost", "travel", "gcost"],
    },
}


SEARCHLIBRIUM_DISTRIBUTIONS: dict[str, str] = {
    "n": "Normal", "ln": "Log-normal", "t": "Triangular", "tn": "Truncated normal", "u": "Uniform",
}


@dataclass
class SearchLibriumConstraintsConfig:
    force_include: list[str] = field(default_factory=list)
    never_include: list[str] = field(default_factory=list)
    mutually_exclusive_groups: list[list[str]] = field(default_factory=list)
    min_behavioral_rules: list[dict] = field(default_factory=list)   # [{"min": int, "pool": [...]}]
    force_random_vars: list[str] = field(default_factory=list)
    force_random_distribution: str = "n"
    exclude_random: list[str] = field(default_factory=list)


def _sl_constraints_code(c: SearchLibriumConstraintsConfig) -> str:
    calls = []
    if c.force_include:
        calls.append(f".force_include({', '.join(repr(v) for v in c.force_include)})")
    if c.never_include:
        calls.append(f".never_include({', '.join(repr(v) for v in c.never_include)})")
    for group in c.mutually_exclusive_groups:
        calls.append(f".mutually_exclusive({', '.join(repr(v) for v in group)})")
    for rule in c.min_behavioral_rules:
        pool_args = ", ".join(repr(v) for v in rule["pool"])
        calls.append(f".min_behavioral({rule['min']!r}, {pool_args})")
    for var in c.force_random_vars:
        calls.append(f".force_random({var!r}, distribution={c.force_random_distribution!r})")
    if c.exclude_random:
        calls.append(f".exclude_random({', '.join(repr(v) for v in c.exclude_random)})")
    if not calls:
        return ""
    return "ConstraintBuilder()\n    " + "\n    ".join(calls)


@dataclass
class SearchLibriumConfig:
    data_path: str                 # local absolute path to the CSV
    hpc_data_filename: str         # filename expected alongside the PBS job on HPC
    choice_col: str
    alt_col: str
    choice_id_col: str
    ind_id_col: Optional[str]
    base_alt: str
    asvarnames: list[str]
    isvarnames: list[str] = field(default_factory=list)
    models: list[str] = field(default_factory=lambda: ["multinomial"])
    allow_random: bool = False
    allow_bcvars: bool = False
    allow_corvars: bool = False
    p_val: float = 0.05
    n_draws: int = 500
    maxiter: int = 2000
    criterion: str = "bic"         # bic | aic | loglik
    mae_enabled: bool = False      # add ("mae", -1) as a second criterion
    val_share: float = 0.25        # held-out share for the auto train/test split when mae_enabled
    algorithm: str = "sa"          # sa | hs | sapbil | banditsa | hspbil | parsa | parcopsa
    nthrds: int = 4                # only used by parsa/parcopsa
    seed: int = 1
    nests_json: Optional[str] = None      # raw JSON text, only for nested_logit/mixed_nested
    lambdas_json: Optional[str] = None
    latent_class: bool = False
    num_classes: int = 2
    output_dir: str = "results"
    experiment_name: str = "searchlibrium_run"
    use_bundled: Optional[str] = None     # key into SEARCHLIBRIUM_BUNDLED_PRESETS, or None
    constraints: SearchLibriumConstraintsConfig = field(default_factory=SearchLibriumConstraintsConfig)


def generate_searchlibrium_script(cfg: SearchLibriumConfig, for_hpc: bool) -> str:
    data_expr = repr(cfg.hpc_data_filename) if for_hpc else repr(cfg.data_path)
    varnames = cfg.asvarnames + cfg.isvarnames

    nests_block = ""
    needs_nests = "nested_logit" in cfg.models or "mixed_nested" in cfg.models
    if needs_nests:
        nests_block = f"""
nests = {cfg.nests_json or '{}'}
lambdas = {cfg.lambdas_json or '{}'}
"""

    ind_id_line = f"ind_id      = df[{cfg.ind_id_col!r}].values," if cfg.ind_id_col else "ind_id      = None,"

    if cfg.use_bundled:
        preset = SEARCHLIBRIUM_BUNDLED_PRESETS[cfg.use_bundled]
        load_block = f"from SearchLibrium import {preset['loader']}\ndf = {preset['loader']}()"
        choices_expr = preset["choice_expr"]
    else:
        load_block = f"df = pd.read_csv({data_expr})"
        choices_expr = f"df[{cfg.choice_col!r}].values"

    constraints_code = _sl_constraints_code(cfg.constraints)
    constraints_import = ", ConstraintBuilder" if constraints_code else ""
    constraints_block = ""
    constraints_kwarg = ""
    if constraints_code:
        constraints_block = f"\nconstraints = (\n    {constraints_code}\n)\nprint(constraints.summary())\n"
        constraints_kwarg = "pre_spec_constraints = constraints.dict(),"

    if cfg.mae_enabled:
        criterions_expr = f"[({cfg.criterion!r}, -1), ('mae', -1)]"
        val_share_kw = f"val_share    = {cfg.val_share!r},"
    else:
        criterions_expr = f"[({cfg.criterion!r}, -1)]"
        val_share_kw = ""

    lc_kwargs = f"latent_class = True,\n    num_classes  = {cfg.num_classes!r}," if cfg.latent_class else ""

    is_parallel_sa = cfg.algorithm in ("parsa", "parcopsa")
    if is_parallel_sa:
        # call_parsa/call_parcopsa don't return anything (unlike every other
        # call_* wrapper) -- retrieve the best solution manually. Verified
        # against SearchLibrium source (call_meta.py, siman.py PARSA/PARCOPSA).
        if cfg.algorithm == "parsa":
            sign_expr = "-1" if cfg.criterion in ("bic", "aic", "mae") else "1"
            run_block = f'''from SearchLibrium.siman import PARSA
from SearchLibrium import estimate_ctrl
ctrl = estimate_ctrl(params, algorithm="sa")
parsa = PARSA(params, None, ctrl, nthrds={cfg.nthrds!r})
parsa.run()
_pick = min if {sign_expr} == -1 else max
candidates = [s.return_best() for s in parsa.solvers]
best = _pick(candidates, key=lambda s: s[{cfg.criterion!r}])'''
        else:
            run_block = f'''from SearchLibrium.siman import PARCOPSA
from SearchLibrium import estimate_ctrl
ctrl = estimate_ctrl(params, algorithm="sa")
parcopsa = PARCOPSA(params, None, ctrl, nthrds={cfg.nthrds!r})
parcopsa.run()
best = parcopsa.solvers[parcopsa.get_best()].best_sol'''
    else:
        run_block = f"best = call_search(params, algorithm={cfg.algorithm!r}, id_num={cfg.seed!r})"

    parallel_summary_block = ""
    if is_parallel_sa:
        # call_parsa/call_parcopsa print no dashboard (unlike call_search's
        # algorithms) -- summarise the retrieved best solution manually.
        parallel_summary_block = (
            f'print("\\nBest solution (retrieved from {cfg.algorithm.upper()}):")\n'
            'for _k in ("asvars", "isvars", "randvars", "bic", "aic", "loglik", "converged"):\n'
            '    if _k in best:\n'
            '        print(f"  {_k:>10}: {best[_k]}")\n'
        )

    return f'''"""Auto-generated by ModelZoo GUI — SearchLibrium search run: {cfg.experiment_name}

Run locally:
    python run_searchlibrium.py

The dashboard SearchLibrium prints at the end of the run is the primary
report; this script also writes a small JSON summary to {cfg.output_dir}/.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
# PYTHONIOENCODING only takes effect if set before the interpreter starts, so
# also reconfigure stdout/stderr directly — SearchLibrium's dashboard prints
# box-drawing Unicode that crashes on Windows' default cp1252 console.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from SearchLibrium import Parameters, call_search{constraints_import}

print("=" * 70)
print("SearchLibrium run: {cfg.experiment_name}")
print("=" * 70)

t0 = time.time()
{load_block}
print(f"Loaded data: {{df.shape[0]}} rows x {{df.shape[1]}} cols")

choice_set = np.unique(df[{cfg.alt_col!r}]).tolist()
{nests_block}{constraints_block}
params = Parameters(
    criterions   = {criterions_expr},
    df           = df,
    varnames     = {_pylist(varnames)},
    asvarnames   = {_pylist(cfg.asvarnames)},
    isvarnames   = {_pylist(cfg.isvarnames)},
    choice_set   = choice_set,
    choices      = {choices_expr},
    alt_var      = df[{cfg.alt_col!r}].values,
    choice_id    = df[{cfg.choice_id_col!r}].values,
    {ind_id_line}
    base_alt     = {cfg.base_alt!r},
    models       = {_pylist(cfg.models)},
    allow_random = {cfg.allow_random!r},
    allow_bcvars = {cfg.allow_bcvars!r},
    allow_corvars= {cfg.allow_corvars!r},
    {constraints_kwarg}
    {val_share_kw}
    {lc_kwargs}
    n_draws      = {cfg.n_draws!r},
    maxiter      = {cfg.maxiter!r},
    p_val        = {cfg.p_val!r},
    {"nests=nests, lambdas=lambdas," if needs_nests else ""}
)

print("\\nRunning search (algorithm={cfg.algorithm!r})...")
{run_block}

elapsed = time.time() - t0
print(f"\\nSearch finished in {{elapsed:.1f}}s")
{parallel_summary_block}

out_dir = Path({cfg.output_dir!r})
out_dir.mkdir(parents=True, exist_ok=True)
summary = {{
    "experiment_name": {cfg.experiment_name!r},
    "algorithm": {cfg.algorithm!r},
    "models": {_pylist(cfg.models)},
    "elapsed_seconds": elapsed,
    "best_repr": repr(best),
}}
out_path = out_dir / ({cfg.experiment_name!r} + "_summary.json")
out_path.write_text(json.dumps(summary, indent=2, default=str))
print(f"Saved run summary to {{out_path}}")
'''


def generate_searchlibrium_ctrl_preview(cfg: SearchLibriumConfig) -> str:
    """A small script that loads the data, builds Parameters, and prints the
    auto-estimated hyperparameters for the chosen algorithm WITHOUT running a
    search — lets a user sanity-check problem complexity/ctrl before
    committing to a (possibly long) real run.
    """
    varnames = cfg.asvarnames + cfg.isvarnames
    if cfg.use_bundled:
        preset = SEARCHLIBRIUM_BUNDLED_PRESETS[cfg.use_bundled]
        load_block = f"from SearchLibrium import {preset['loader']}\ndf = {preset['loader']}()"
        choices_expr = preset["choice_expr"]
    else:
        load_block = f"df = pd.read_csv({cfg.data_path!r})"
        choices_expr = f"df[{cfg.choice_col!r}].values"
    ind_id_line = f"ind_id      = df[{cfg.ind_id_col!r}].values," if cfg.ind_id_col else "ind_id      = None,"
    # ctrl tuple shape depends on the underlying family: sa/sapbil/banditsa/
    # parsa/parcopsa all use the 4-tuple SA-style estimate; hs/hspbil use the
    # 6-tuple HS-style estimate. Verified directly against call_meta.py
    # docstrings (call_sapbil/call_banditsa document the 4-tuple SA shape;
    # call_harmony_pbil documents the 6-tuple HS shape) rather than assumed.
    preview_algo = "hs" if cfg.algorithm in ("hs", "hspbil") else "sa"

    return f'''import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from SearchLibrium import Parameters, estimate_ctrl
from SearchLibrium.call_meta import _describe_ctrl

{load_block}
choice_set = np.unique(df[{cfg.alt_col!r}]).tolist()
params = Parameters(
    criterions   = [({cfg.criterion!r}, -1)],
    df           = df,
    varnames     = {_pylist(varnames)},
    asvarnames   = {_pylist(cfg.asvarnames)},
    isvarnames   = {_pylist(cfg.isvarnames)},
    choice_set   = choice_set,
    choices      = {choices_expr},
    alt_var      = df[{cfg.alt_col!r}].values,
    choice_id    = df[{cfg.choice_id_col!r}].values,
    {ind_id_line}
    base_alt     = {cfg.base_alt!r},
    models       = {_pylist(cfg.models)},
    allow_random = {cfg.allow_random!r},
)

ctrl = estimate_ctrl(params, algorithm={preview_algo!r})
print(f"Auto-estimated ctrl for algorithm={preview_algo!r}:")
print(_describe_ctrl(ctrl, {preview_algo!r}))
print()
print("Raw tuple:", ctrl)
print()
print("Note: sapbil/banditsa/parsa/parcopsa reuse the 'sa'-style estimate; hspbil reuses the 'hs'-style estimate (verified against call_meta.py docstrings).")
'''


# ─────────────────────────────────────────────────────────────────────────────
# SearchLibrium — Standalone fit (direct model.setup()/model.fit(), no search)
# ─────────────────────────────────────────────────────────────────────────────

# The 12 standalone model classes do NOT share one uniform setup()/fit() API.
# Verified directly against source (not the README) -- three distinct
# families, plus two outright special cases:
#   Family A: empty ctor -> .setup(X, y, varnames, alts, isvars, ids, ...) ->
#             .fit() with no required args, returns None.
#             MultinomialLogit, MixedLogit, NestedLogit, MultiLayerNestedLogit,
#             MixedNested, MultinomialProbit.
#   Family B: ctor itself calls setup(**kwargs) (X/y/J/distr passed at
#             construction); OrderedLogit, OrderedLogitLong.
#   Family C: fluent .setup() returns self, .fit() returns a value (not
#             None), hard JAX dependency; BinaryProbit, HeckmanTwoStep.
#   Special: RandomRegret/MixedRandomRegret use RandomRegret's *generic*
#            .setup(X, y, varnames, alts, isvars, ids, ...) path (confirmed
#            it calls self.initialise(), NOT the df=/short= constructor path
#            which requires rigid literal 'id'/'alt'/'choice' column names).
#            LatentClassMixedLogit takes its hyperparameters (n_classes,
#            maxiter, ...) at construction time, unlike anything else.
#
# All 9 classes below were run end-to-end against the real engine and real
# data this session; along the way this surfaced (and fixed, in the
# SearchLibrium source) 8 genuine upstream bugs -- e.g. NestedLogit.setup()
# was defined twice in the same class body with the second (X_nest-requiring)
# definition silently shadowing the first; MixedRandomRegret/MixedNested/
# MultiLayerNestedLogit never initialise attributes their own base classes'
# __init__ chains are supposed to set up. Three classes are deliberately
# OMITTED from this dict (and so unavailable in the GUI) because they hit
# bugs deep enough that a full fix was out of scope for this pass:
#   - MixedRandomRegret: RandomRegret's setup() never calls MixedLogit's, so
#     several attributes generate_draws() needs (rvdist, randvars, rvidx,
#     ...) are never initialised -- would need MixedLogit's random-variable
#     processing re-implemented inside RandomRegret's setup.
#   - MixedNested: its MixedLogit.__init__(self, _jax=_jax) call targets the
#     *other* MixedLogit (mixed_logit.py's zero-arg version), raising
#     TypeError immediately.
#   - MultiLayerNestedLogit: self.np (normally set inside NestedLogit's own
#     __init__, not inherited generically) is never initialised, so setup()
#     crashes with AttributeError as soon as it needs self.np.
#   - OrderedLogitLong: its own __init__/self.np bug IS fixed (see
#     ordered_logit.py), but it additionally requires the same
#     misc.wide_to_long-expanded long-format data (one row per individual per
#     ordinal category, not one row per observation) that MixedNested-style
#     alt-based models need -- out of scope for this simple column-mapping UI.
STANDALONE_MODEL_FAMILIES: dict[str, str] = {
    "MultinomialLogit": "mnl_family",
    "MultinomialProbit": "mnl_family",
    "MixedLogit": "mixed_logit",
    "NestedLogit": "nested",
    "RandomRegret": "rrm",
    "OrderedLogit": "ordered",
    "BinaryProbit": "binary_probit",
    "HeckmanTwoStep": "heckman",
    "LatentClassMixedLogit": "latent_class_mxl",
}


@dataclass
class StandaloneFitConfig:
    data_path: str
    hpc_data_filename: str
    model_class: str
    use_bundled: Optional[str] = None
    # Discrete-choice (long-format) column mapping -- used by most families
    choice_col: str = "CHOICE"
    alt_col: str = "alt"
    choice_id_col: str = "custom_id"
    ind_id_col: Optional[str] = None       # panels, MixedLogit only
    avail_col: Optional[str] = None
    base_alt: str = ""
    asvarnames: list[str] = field(default_factory=list)
    isvarnames: list[str] = field(default_factory=list)
    fit_intercept: bool = False
    maxiter: int = 2000
    # MixedLogit
    randvars: dict[str, str] = field(default_factory=dict)   # var -> n/ln/t/tn/u
    correlated_vars: list[str] = field(default_factory=list)
    n_draws: int = 1000
    # NestedLogit
    nests_json: Optional[str] = None
    lambdas_json: Optional[str] = None
    # OrderedLogit
    ordinal_y_col: str = ""
    n_categories: int = 3
    ordered_distr: str = "logit"          # probit | logit
    normalize: bool = False
    # BinaryProbit
    binary_y_col: str = ""
    # HeckmanTwoStep
    selection_y_col: str = ""
    selection_varnames: list[str] = field(default_factory=list)
    outcome_y_col: str = ""
    outcome_varnames: list[str] = field(default_factory=list)
    # LatentClassMixedLogit
    n_classes: int = 2
    membership_vars: list[str] = field(default_factory=list)
    output_dir: str = "results"
    experiment_name: str = "standalone_fit"


def generate_standalone_fit_script(cfg: StandaloneFitConfig, for_hpc: bool) -> str:
    if cfg.use_bundled:
        preset = SEARCHLIBRIUM_BUNDLED_PRESETS[cfg.use_bundled]
        load_block = f"from SearchLibrium import {preset['loader']}\ndf = {preset['loader']}()"
    else:
        data_expr = repr(cfg.hpc_data_filename) if for_hpc else repr(cfg.data_path)
        load_block = f"df = pd.read_csv({data_expr})"

    family = STANDALONE_MODEL_FAMILIES[cfg.model_class]
    varnames = cfg.asvarnames + cfg.isvarnames
    avail_kw = f"avail=df[{cfg.avail_col!r}].values," if cfg.avail_col else ""

    import_line = f"from SearchLibrium import {cfg.model_class}"
    prelude = f"varnames = {_pylist(varnames)}\n"
    report_lines = ""

    if family == "mnl_family":
        prelude += ""
        body = f'''model = {cfg.model_class}()
model.setup(X=df[varnames], y=df[{cfg.choice_col!r}], varnames=varnames, isvars={_pylist(cfg.isvarnames)},
            fit_intercept={cfg.fit_intercept!r}, alts=df[{cfg.alt_col!r}], ids=df[{cfg.choice_id_col!r}],
            {avail_kw} base_alt={cfg.base_alt!r}, maxiter={cfg.maxiter!r})
model.fit()'''
        report_lines = 'model.get_loglik_null()\nmodel.summarise()'

    elif family == "mixed_logit":
        panels_kw = f"panels=df[{cfg.ind_id_col!r}].values," if cfg.ind_id_col else ""
        correlated_expr = _pylist(cfg.correlated_vars) if cfg.correlated_vars else "None"
        prelude += "X = df[varnames].values\ny = df[choice_col].values\n".replace("choice_col", repr(cfg.choice_col))
        body = f'''model = MixedLogit()
model.setup(X, y, ids=df[{cfg.choice_id_col!r}].values, {panels_kw} varnames=varnames,
            isvars={_pylist(cfg.isvarnames)}, transvars=[], correlated_vars={correlated_expr},
            randvars={cfg.randvars!r}, fit_intercept={cfg.fit_intercept!r}, alts=df[{cfg.alt_col!r}],
            n_draws={cfg.n_draws!r}, mnl_init=True)
model.fit()'''
        report_lines = 'model.get_loglik_null()\nmodel.summarise()'

    elif family == "nested":
        # nests dict values are 0-based positional indices into the *sorted
        # unique* alt values (not the raw alt labels) -- confirmed directly
        # against compute_probabilities()'s integer array indexing, and by
        # a real end-to-end run against swiss_metro (alts CAR/SM/TRAIN sort
        # to positions 0/1/2).
        prelude += f"nests = {cfg.nests_json or '{}'}\nlambdas = {cfg.lambdas_json or '{}'}\n"
        # NestedLogit.setup() is defined TWICE in multinomial_nested.py; the
        # second definition (requiring X_nest) silently shadows the first
        # and is the one actually reachable -- confirmed directly (the
        # "simple" no-X_nest call demonstrated in main.py raises TypeError:
        # missing 1 required positional argument: 'X_nest'). search.py's own
        # metaheuristic nested_logit path already passes X_nest, so this is
        # the class's real, current live signature.
        body = f'''model = NestedLogit()
model.setup(X=df[varnames], X_nest=None, y=df[{cfg.choice_col!r}], varnames=varnames,
            isvars={_pylist(cfg.isvarnames)}, fit_intercept={cfg.fit_intercept!r}, alts=df[{cfg.alt_col!r}],
            ids=df[{cfg.choice_id_col!r}], nests=nests, lambdas=lambdas)
model.fit()'''
        report_lines = 'try:\n    model.summarise()\nexcept Exception as e:\n    print(f"[warn] summarise() failed: {e}")'

    elif family == "rrm":
        body = f'''model = RandomRegret()
model.setup(X=df[varnames], y=df[{cfg.choice_col!r}], varnames=varnames, alts=df[{cfg.alt_col!r}],
            isvars={_pylist(cfg.isvarnames)}, ids=df[{cfg.choice_id_col!r}], base_alt={cfg.base_alt!r},
            fit_intercept={cfg.fit_intercept!r})
model.fit()'''
        report_lines = 'model.report()'

    elif family == "ordered":
        prelude = f"asvarnames = {_pylist(cfg.asvarnames)}\n"
        body = f'''model = OrderedLogit(X=df[asvarnames], y=df[{cfg.ordinal_y_col!r}], J={cfg.n_categories!r},
                     distr={cfg.ordered_distr!r}, start=None, normalize={cfg.normalize!r},
                     fit_intercept={cfg.fit_intercept!r})
model.fit()'''
        report_lines = 'model.report()'

    elif family == "binary_probit":
        prelude = f"asvarnames = {_pylist(cfg.asvarnames)}\n"
        body = f'''model = BinaryProbit()
model.setup(X=df[asvarnames], y=df[{cfg.binary_y_col!r}], varnames=asvarnames,
            fit_intercept={cfg.fit_intercept!r})
res = model.fit()'''
        report_lines = 'print(res)\nprint("Coefficients:", dict(zip(asvarnames, model.coeff_est)))'

    elif family == "heckman":
        prelude = (
            f"selection_varnames = {_pylist(cfg.selection_varnames)}\n"
            f"outcome_varnames = {_pylist(cfg.outcome_varnames)}\n"
        )
        body = f'''model = HeckmanTwoStep()
model.setup(selection_X=df[selection_varnames], selection_y=df[{cfg.selection_y_col!r}],
            outcome_X=df[outcome_varnames], outcome_y=df[{cfg.outcome_y_col!r}],
            selection_varnames=selection_varnames, outcome_varnames=outcome_varnames,
            fit_intercept={cfg.fit_intercept!r})
result = model.fit()'''
        report_lines = 'print(model.params_table)\nprint("Selection-stage params:", result["probit"].coeff_est)'

    else:  # latent_class_mxl
        avail_kw2 = f"avail=df[{cfg.avail_col!r}].values," if cfg.avail_col else ""
        membership_kw = f"membership_vars={_pylist(cfg.membership_vars)}," if cfg.membership_vars else ""
        body = f'''model = LatentClassMixedLogit(n_classes={cfg.n_classes!r}, maxiter={cfg.maxiter!r},
                              class_maxiter=50, tol=1e-6, random_state=42)
model.setup(X=df[varnames].values, y=df[{cfg.choice_col!r}].values, varnames=varnames,
            ids=df[{cfg.choice_id_col!r}].values, alts=df[{cfg.alt_col!r}].values,
            {avail_kw2} {membership_kw})
model.fit()'''
        report_lines = 'try:\n    model.summarise()\nexcept Exception as e:\n    print(f"[warn] summarise() failed: {e}")\n    print("class_betas:", getattr(model, \'class_betas\', None))'

    return f'''"""Auto-generated by ModelZoo GUI — SearchLibrium standalone fit: {cfg.experiment_name}
Model class: {cfg.model_class}

Run locally:
    python run_standalone_fit.py

Fits a single, pre-specified {cfg.model_class} directly (no metaheuristic
search over structures) -- for exploring/reporting on one specification.
"""
import os
import sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import time
from pathlib import Path
import numpy as np
import pandas as pd
{import_line}

print("=" * 70)
print("SearchLibrium standalone fit: {cfg.experiment_name} ({cfg.model_class})")
print("=" * 70)

t0 = time.time()
{load_block}
print(f"Loaded data: {{df.shape[0]}} rows x {{df.shape[1]}} cols")

{prelude}
{body}

elapsed = time.time() - t0
print(f"\\nFit finished in {{elapsed:.1f}}s")
{report_lines}
'''


# ─────────────────────────────────────────────────────────────────────────────
# SearchLibrium — MDCEV budget allocation (MDCEVModel, verified against
# src/SearchLibrium/mdcev.py directly)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MDCEVConfig:
    data_path: str
    hpc_data_filename: str
    allocation_cols: list[str]
    outside_good_col: Optional[str] = None   # one of allocation_cols, or None
    alpha_floor: float = 0.05
    alpha_cap: float = 0.95
    gamma_floor: float = 1e-3
    tol: float = 1e-9
    fit_mode: str = "heuristic"        # heuristic | mle
    mle_maxiter: int = 400
    mle_l2_penalty: float = 1e-4
    predict_budgets: list[float] = field(default_factory=lambda: [100.0])
    run_simulation: bool = False
    n_draws: int = 100
    sim_seed: int = 42
    output_dir: str = "results"
    experiment_name: str = "mdcev_run"


def generate_mdcev_script(cfg: MDCEVConfig, for_hpc: bool) -> str:
    data_expr = repr(cfg.hpc_data_filename) if for_hpc else repr(cfg.data_path)
    outside_good_idx = repr(cfg.allocation_cols.index(cfg.outside_good_col)) if cfg.outside_good_col else "None"

    if cfg.fit_mode == "mle":
        fit_call = (
            f"model.fit_mle(allocations, labels=labels, maxiter={cfg.mle_maxiter!r}, "
            f"l2_penalty={cfg.mle_l2_penalty!r})"
        )
    else:
        fit_call = "model.fit(allocations, labels=labels)"

    sim_block = ""
    if cfg.run_simulation:
        sim_block = f'''
print("\\nSimulating stochastic allocations (n_draws={cfg.n_draws!r})...")
sims = model.simulate(budgets, n_draws={cfg.n_draws!r}, random_state={cfg.sim_seed!r})
sim_mean = sims.mean(axis=0)
sim_df = pd.DataFrame(sim_mean, columns=labels)
sim_df.insert(0, "budget", budgets)
print(sim_df.to_string(index=False))
sim_df.to_csv(out_dir / ({cfg.experiment_name!r} + "_simulated_mean.csv"), index=False)
'''

    return f'''"""Auto-generated by ModelZoo GUI — MDCEV budget allocation run: {cfg.experiment_name}

Run locally:
    python run_mdcev.py

Fits SearchLibrium's translated-utility MDCEV allocator (MDCEVModel) to
observed budget-allocation data (rows = observations, columns = the
alternatives a fixed budget is split across), then predicts/simulates
allocations for one or more budget levels.
"""
import os
import sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import time
from pathlib import Path
import pandas as pd
from SearchLibrium import MDCEVModel

print("=" * 70)
print("MDCEV budget allocation run: {cfg.experiment_name}")
print("=" * 70)

t0 = time.time()
df = pd.read_csv({data_expr})
print(f"Loaded data: {{df.shape[0]}} rows x {{df.shape[1]}} cols")

labels = {_pylist(cfg.allocation_cols)}
allocations = df[labels].values

model = MDCEVModel(
    outside_good={outside_good_idx}, alpha_floor={cfg.alpha_floor!r}, alpha_cap={cfg.alpha_cap!r},
    gamma_floor={cfg.gamma_floor!r}, tol={cfg.tol!r},
)
print("\\nFitting (mode={cfg.fit_mode!r})...")
{fit_call}

elapsed = time.time() - t0
print(f"\\nFit finished in {{elapsed:.1f}}s")

summary = model.summary()
print("\\nModel summary:")
print(summary.to_string(index=False))

out_dir = Path({cfg.output_dir!r})
out_dir.mkdir(parents=True, exist_ok=True)
summary.to_csv(out_dir / ({cfg.experiment_name!r} + "_summary.csv"), index=False)

budgets = {cfg.predict_budgets!r}
print("\\nPredicting deterministic allocations for budgets:", budgets)
preds = model.predict(budgets)
pred_df = pd.DataFrame(preds, columns=labels)
pred_df.insert(0, "budget", budgets)
print(pred_df.to_string(index=False))
pred_df.to_csv(out_dir / ({cfg.experiment_name!r} + "_predictions.csv"), index=False)
{sim_block}
print(f"\\nSaved outputs to {{out_dir}}")
'''


# ─────────────────────────────────────────────────────────────────────────────
# SearchLibrium — Destination/flow prediction (DestinationPredictor,
# verified against src/SearchLibrium/predict.py directly)
# ─────────────────────────────────────────────────────────────────────────────

# DestinationPredictor.compute_probabilities() reuses the fitted model's
# cached ind_pred_prob/choice_pred_prob attributes whenever their shape
# matches (n_trips, n_dests) -- if you fit a model on one dataset and then
# build a DestinationPredictor over a *different* idca dataframe that
# happens to have the same trip/destination counts, it will silently return
# stale probabilities from the original fit instead of recomputing. This
# generator always fits the model and runs the predictor over the exact
# same idca dataframe in one script, which sidesteps that trap by
# construction rather than trying to fix the caching logic itself.
#
# RandomRegret is deliberately excluded: it has neither ind_pred_prob/
# choice_pred_prob nor a compute_probabilities() method (confirmed directly
# -- DestinationPredictor raises "Model does not support predict()" for it),
# so it's fundamentally incompatible with this tool by design, not a bug.
DESTINATION_MODEL_FAMILIES: dict[str, str] = {
    "MultinomialLogit": "mnl_family",
    "MixedLogit": "mixed_logit",
    "NestedLogit": "nested",
}


@dataclass
class DestinationPredictionConfig:
    data_path: str
    hpc_data_filename: str
    model_class: str                   # one of DESTINATION_MODEL_FAMILIES
    trip_id_col: str = "trip_id"
    dest_col: str = "dest_code"
    choice_col: str = "chosen"
    dest_name_col: Optional[str] = None
    avail_col: Optional[str] = None
    varnames: list[str] = field(default_factory=list)
    isvarnames: list[str] = field(default_factory=list)
    base_alt: str = ""
    fit_intercept: bool = False
    maxiter: int = 2000
    randvars: dict[str, str] = field(default_factory=dict)
    n_draws: int = 200
    nests_json: Optional[str] = None
    lambdas_json: Optional[str] = None
    output_dir: str = "results"
    experiment_name: str = "destination_prediction"


def generate_destination_prediction_script(cfg: DestinationPredictionConfig, for_hpc: bool) -> str:
    data_expr = repr(cfg.hpc_data_filename) if for_hpc else repr(cfg.data_path)
    family = DESTINATION_MODEL_FAMILIES[cfg.model_class]
    varnames = cfg.varnames + cfg.isvarnames
    avail_kw = f"avail=idca[{cfg.avail_col!r}].values," if cfg.avail_col else ""
    dest_name_kw = f"dest_name_col={cfg.dest_name_col!r}," if cfg.dest_name_col else ""

    prelude = f"varnames = {_pylist(varnames)}\n"

    if family == "mnl_family":
        body = f'''model = MultinomialLogit()
model.setup(X=idca[varnames], y=idca[{cfg.choice_col!r}], varnames=varnames, isvars={_pylist(cfg.isvarnames)},
            fit_intercept={cfg.fit_intercept!r}, alts=idca[{cfg.dest_col!r}], ids=idca[{cfg.trip_id_col!r}],
            {avail_kw} base_alt={cfg.base_alt!r}, maxiter={cfg.maxiter!r})
model.fit()'''
    elif family == "mixed_logit":
        prelude += "X = idca[varnames].values\ny = idca[choice_col].values\n".replace("choice_col", repr(cfg.choice_col))
        body = f'''model = MixedLogit()
model.setup(X, y, ids=idca[{cfg.trip_id_col!r}].values, varnames=varnames, isvars={_pylist(cfg.isvarnames)},
            transvars=[], randvars={cfg.randvars!r}, fit_intercept={cfg.fit_intercept!r},
            alts=idca[{cfg.dest_col!r}], n_draws={cfg.n_draws!r}, mnl_init=True)
model.fit()'''
    else:  # nested
        prelude += f"nests = {cfg.nests_json or '{}'}\nlambdas = {cfg.lambdas_json or '{}'}\n"
        # See STANDALONE_MODEL_FAMILIES comment re: NestedLogit's duplicated
        # setup() -- X_nest=None matches the class's real, live signature.
        body = f'''model = NestedLogit()
model.setup(X=idca[varnames], X_nest=None, y=idca[{cfg.choice_col!r}], varnames=varnames,
            isvars={_pylist(cfg.isvarnames)}, fit_intercept={cfg.fit_intercept!r}, alts=idca[{cfg.dest_col!r}],
            ids=idca[{cfg.trip_id_col!r}], nests=nests, lambdas=lambdas)
model.fit()'''

    return f'''"""Auto-generated by ModelZoo GUI — Destination/flow prediction run: {cfg.experiment_name}
Model class: {cfg.model_class}

Run locally:
    python run_destination_prediction.py

Fits {cfg.model_class} on trip-level destination-choice data (idca, long
format), then uses DestinationPredictor to predict the most likely
destination per trip and aggregate flows per destination, compared
against the observed data. Deliberately fits and predicts over the exact
same idca dataframe in one script (see the comment above
DESTINATION_MODEL_FAMILIES in script_gen.py for why).
"""
import os
import sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import time
from pathlib import Path
import pandas as pd
from SearchLibrium import {cfg.model_class}
from SearchLibrium.predict import DestinationPredictor

print("=" * 70)
print("Destination/flow prediction run: {cfg.experiment_name} ({cfg.model_class})")
print("=" * 70)

t0 = time.time()
idca = pd.read_csv({data_expr})
print(f"Loaded data: {{idca.shape[0]}} rows x {{idca.shape[1]}} cols")

{prelude}
{body}

elapsed = time.time() - t0
print(f"\\nFit finished in {{elapsed:.1f}}s")

predictor = DestinationPredictor(
    model, idca, varnames, dest_col={cfg.dest_col!r}, trip_id_col={cfg.trip_id_col!r},
    choice_col={cfg.choice_col!r}, {dest_name_kw}
)

print("\\nPredicting per-trip destinations...")
dest_preds = predictor.predict_destinations()
print(dest_preds.head(20).to_string(index=False))

print("\\nPredicting aggregate flows per destination...")
flow_preds = predictor.predict_aggregate_flows()
print(flow_preds.to_string(index=False))

metrics = predictor.compute_metrics()
print("\\nValidation metrics:")
for k, v in metrics.items():
    print(f"  {{k:>10}}: {{v}}")

out_dir = Path({cfg.output_dir!r})
out_dir.mkdir(parents=True, exist_ok=True)
dest_preds.to_csv(out_dir / ({cfg.experiment_name!r} + "_destination_predictions.csv"), index=False)
flow_preds.to_csv(out_dir / ({cfg.experiment_name!r} + "_flow_predictions.csv"), index=False)
print(f"\\nSaved outputs to {{out_dir}}")
'''


# ─────────────────────────────────────────────────────────────────────────────
# metacountregressor
# ─────────────────────────────────────────────────────────────────────────────

METACOUNT_DISTRIBUTIONS = ["normal", "lognormal", "triangular", "uniform"]


@dataclass
class ConstraintsConfig:
    force_include: list[str] = field(default_factory=list)
    force_fixed: list[str] = field(default_factory=list)
    no_random: list[str] = field(default_factory=list)
    no_zi: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    membership_only: list[str] = field(default_factory=list)
    allow_membership: list[str] = field(default_factory=list)
    outcome_only: list[str] = field(default_factory=list)
    allow_random_vars: list[str] = field(default_factory=list)
    allow_random_distributions: list[str] = field(default_factory=lambda: list(METACOUNT_DISTRIBUTIONS))
    mutual_exclusion_groups: list[list[str]] = field(default_factory=list)


@dataclass
class MetaCountConfig:
    data_path: str
    hpc_data_filename: str
    use_bundled_example: bool
    id_col: str
    y_col: str
    offset_col: Optional[str]
    group_id_col: Optional[str]
    variables: list[str]
    constraints: ConstraintsConfig
    model_family: str = "count"        # count | cmf | duration | linear
    default_roles: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 5])
    max_latent_classes: int = 1
    r_draws: int = 200
    algo: str = "sa"                   # sa | de | hs
    max_iter: int = 1000
    seed: int = 42
    fit_model_families: list[str] = field(default_factory=lambda: ["nb"])  # e.g. ["nb", "poisson"]
    final_r_draws: int = 500
    output_dir: str = "results"
    experiment_name: str = "metacount_run"
    search_description: str = ""


def _constraints_code(c: ConstraintsConfig) -> str:
    calls = []
    if c.force_include:
        calls.append(f".force_include({', '.join(repr(v) for v in c.force_include)})")
    if c.force_fixed:
        calls.append(f".force_fixed({', '.join(repr(v) for v in c.force_fixed)})")
    if c.no_random:
        calls.append(f".no_random({', '.join(repr(v) for v in c.no_random)})")
    if c.no_zi:
        calls.append(f".no_zi({', '.join(repr(v) for v in c.no_zi)})")
    if c.exclude:
        calls.append(f".exclude({', '.join(repr(v) for v in c.exclude)})")
    if c.membership_only:
        calls.append(f".membership_only({', '.join(repr(v) for v in c.membership_only)})")
    if c.allow_membership:
        calls.append(f".allow_membership({', '.join(repr(v) for v in c.allow_membership)})")
    if c.outcome_only:
        calls.append(f".outcome_only({', '.join(repr(v) for v in c.outcome_only)})")
    if c.allow_random_vars:
        var_args = ", ".join(repr(v) for v in c.allow_random_vars)
        calls.append(f".allow_random({var_args}, distributions={c.allow_random_distributions!r})")
    if c.mutual_exclusion_groups:
        group_args = ", ".join(repr(g) for g in c.mutual_exclusion_groups)
        calls.append(f".mutual_exclusion({group_args})")
    if not calls:
        return "ModelConstraints()"
    return "ModelConstraints()\n    " + "\n    ".join(calls)


def generate_metacount_script(cfg: MetaCountConfig, for_hpc: bool) -> str:
    data_expr = repr(cfg.hpc_data_filename) if for_hpc else repr(cfg.data_path)

    if cfg.use_bundled_example:
        load_block = "from metacountregressor import load_example16_3_model_data\ndf = load_example16_3_model_data()"
    else:
        load_block = f"df = pd.read_csv({data_expr})"

    offset_kw = f"offset_col={cfg.offset_col!r}," if cfg.offset_col else ""
    group_kw = f"group_id_col={cfg.group_id_col!r}," if cfg.group_id_col else ""

    build_method = "build_count_evaluator" if cfg.model_family == "count" else "build_evaluator"
    family_kw = "" if cfg.model_family == "count" else f"model_family={cfg.model_family!r},"

    constraints_code = _constraints_code(cfg.constraints)
    has_constraints = constraints_code != "ModelConstraints()"
    constraints_warning = ""
    if has_constraints and cfg.model_family != "count":
        warn_msg = (
            f"[warn] model_family={cfg.model_family!r} - build_evaluator() only merges "
            "ModelConstraints for the count family; constraints above (including "
            "mutual_exclusion) may be silently ignored. Verified against source, not "
            "upstream-documented."
        )
        constraints_warning = f"print({warn_msg!r})\n"

    return f'''"""Auto-generated by ModelZoo GUI — metacountregressor search run: {cfg.experiment_name}

Run locally:
    python run_metacount.py
"""
import os
import sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import time
from pathlib import Path
import pandas as pd
from metacountregressor import (
    ExperimentBuilder,
    ModelConstraints,
    SearchOutputConfig,
    extract_search_best,
    extract_summary,
    compare_models,
)

print("=" * 70)
print("metacountregressor run: {cfg.experiment_name}")
print("=" * 70)

t0 = time.time()
{load_block}
print(f"Loaded data: {{df.shape[0]}} rows x {{df.shape[1]}} cols")

constraints = (
    {constraints_code}
)
print(constraints)
{constraints_warning}
builder = ExperimentBuilder(
    df=df,
    id_col={cfg.id_col!r},
    y_col={cfg.y_col!r},
    {offset_kw}
    {group_kw}
)
try:
    builder.describe()
except Exception as e:
    # describe() is diagnostic-only; some upstream metacountregressor versions
    # crash on non-numeric columns outside `variables` (e.g. label columns).
    print(f"[warn] builder.describe() failed, continuing without it: {{e}}")

evaluator = builder.{build_method}(
    variables={_pylist(cfg.variables)},
    constraints=constraints,
    {family_kw}
    default_roles={cfg.default_roles!r},
    max_latent_classes={cfg.max_latent_classes!r},
    R={cfg.r_draws!r},
)

output_config = SearchOutputConfig(
    output_dir={cfg.output_dir!r},
    experiment_name={cfg.experiment_name!r},
    search_description={cfg.search_description!r},
    save_json=True,
)

print("\\nRunning search (algo={cfg.algo!r})...")
result = builder.run(
    evaluator,
    algo={cfg.algo!r},
    max_iter={cfg.max_iter!r},
    seed={cfg.seed!r},
    output_config=output_config,
)

elapsed = time.time() - t0
best = extract_search_best(result)
print(f"\\nSearch finished in {{elapsed:.1f}}s")
print("Best BIC:", best["best_bic"])
print("Saved to:", result.get("saved_to"))

best_spec = evaluator.build_spec(best["best_decision"])
out_dir = Path({cfg.output_dir!r})
out_dir.mkdir(parents=True, exist_ok=True)

fit_families = {_pylist(cfg.fit_model_families)}
fits = {{}}
for family in fit_families:
    print(f"\\nRefitting best structure as {{family!r}} with R={cfg.final_r_draws}...")
    fits[family] = builder.fit_manual_model(manual_spec=best_spec, model=family, R={cfg.final_r_draws!r})
    summary = extract_summary(fits[family])
    print(f"  {{family}} summary:")
    for k, v in summary.items():
        print(f"    {{k:>10}}: {{v}}")

if len(fits) > 1:
    print("\\nModel family comparison (sorted by BIC):")
    comparison = compare_models(fits)
    print(comparison.to_string(index=False))
    comparison.to_csv(out_dir / ({cfg.experiment_name!r} + "_family_comparison.csv"), index=False)
'''


# ─────────────────────────────────────────────────────────────────────────────
# metacountregressor — CMF (Crash Modification Function)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CMFConfig:
    data_path: str
    hpc_data_filename: str
    y_col: str
    aadt_col: str
    baseline_vars: list[str]
    local_vars: list[str]
    search_mode: str = "ga"            # ga | jax
    # GA mode (original AADT-specific GA search + CMF interpretation table)
    ga_R: int = 200
    ga_final_R: int = 500
    # JAX flexible mode (full ModelConstraints + latent-class support)
    id_col: str = "id"
    offset_col: Optional[str] = None
    group_id_col: Optional[str] = None
    variables: list[str] = field(default_factory=list)   # extra auxiliary vars, beyond baseline/local
    constraints: ConstraintsConfig = field(default_factory=ConstraintsConfig)
    default_roles: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 5])
    max_latent_classes: int = 1
    r_draws: int = 200
    force_aadt_term: bool = True
    algo: str = "sa"                   # sa | de | hs
    max_iter: int = 1000
    seed: int = 42
    fit_model_families: list[str] = field(default_factory=lambda: ["nb"])
    final_r_draws: int = 500
    output_dir: str = "results"
    experiment_name: str = "cmf_run"
    search_description: str = ""


def generate_cmf_script(cfg: CMFConfig, for_hpc: bool) -> str:
    data_expr = repr(cfg.hpc_data_filename) if for_hpc else repr(cfg.data_path)
    load_block = f"df = pd.read_csv({data_expr})"

    if cfg.search_mode == "ga":
        return f'''"""Auto-generated by ModelZoo GUI — CMF (GA search) run: {cfg.experiment_name}

Run locally:
    python run_cmf.py

Uses metacountregressor's original AADT-specific GA/JAX CMF search
(CMFExperimentBuilder.run_search/fit_best_model) and prints a CMF
interpretation table (CMF = exp(beta * delta), HSM-style).
"""
import os
import sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import time
from pathlib import Path
import pandas as pd
from metacountregressor import CMFExperimentBuilder

print("=" * 70)
print("CMF (GA search) run: {cfg.experiment_name}")
print("=" * 70)

t0 = time.time()
{load_block}
print(f"Loaded data: {{df.shape[0]}} rows x {{df.shape[1]}} cols")

cmf = CMFExperimentBuilder(
    df=df,
    y_col={cfg.y_col!r},
    aadt_col={cfg.aadt_col!r},
    baseline_vars={_pylist(cfg.baseline_vars)},
    local_vars={_pylist(cfg.local_vars)},
)

print("\\nRunning GA structure search (R={cfg.ga_R!r})...")
search_result = cmf.run_search(R={cfg.ga_R!r})
print("Selected baseline:", search_result.selected_baseline)
print("Selected local:", search_result.selected_local)
print("Model:", search_result.model, "| Fitness:", search_result.fitness)

print("\\nRefitting best structure (final_R={cfg.ga_final_R!r})...")
fit_result = cmf.fit_best_model(search_result, final_R={cfg.ga_final_R!r})
cmf.print_report(search_result, fit_result)
cmf_table = cmf.print_cmf_interpretation(fit_result)

elapsed = time.time() - t0
print(f"\\nFinished in {{elapsed:.1f}}s")

out_dir = Path({cfg.output_dir!r})
out_dir.mkdir(parents=True, exist_ok=True)
cmf_table.to_csv(out_dir / ({cfg.experiment_name!r} + "_cmf_interpretation.csv"), index=False)
print(f"Saved CMF interpretation table to {{out_dir}}")
'''

    # JAX flexible mode: builds a normal ExperimentBuilder/evaluator under the
    # hood (via build_jax_count_evaluator), so it gets full ModelConstraints
    # support, latent classes, and the same run()/fit_manual_model()/
    # compare_models() flow as the count-family search above.
    constraints_code = _constraints_code(cfg.constraints)
    aux_vars = [v for v in cfg.variables if v not in (*cfg.baseline_vars, *cfg.local_vars, cfg.aadt_col)]
    offset_kw = f"offset_col={cfg.offset_col!r}," if cfg.offset_col else ""
    group_kw = f"group_id_col={cfg.group_id_col!r}," if cfg.group_id_col else ""

    return f'''"""Auto-generated by ModelZoo GUI — CMF (JAX flexible search) run: {cfg.experiment_name}

Run locally:
    python run_cmf.py

Uses CMFExperimentBuilder.build_jax_count_evaluator(), which bridges the
AADT baseline/local-interaction structure into a regular ExperimentBuilder
evaluator -- this is the route that supports full ModelConstraints
(force_include/force_fixed/exclude/mutual_exclusion/...) and latent classes.
"""
import os
import sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import time
from pathlib import Path
import pandas as pd
from metacountregressor import (
    CMFExperimentBuilder,
    ModelConstraints,
    SearchOutputConfig,
    extract_search_best,
    extract_summary,
    compare_models,
)

print("=" * 70)
print("CMF (JAX flexible search) run: {cfg.experiment_name}")
print("=" * 70)

t0 = time.time()
{load_block}
print(f"Loaded data: {{df.shape[0]}} rows x {{df.shape[1]}} cols")

constraints = (
    {constraints_code}
)
print(constraints)

cmf = CMFExperimentBuilder(
    df=df,
    y_col={cfg.y_col!r},
    aadt_col={cfg.aadt_col!r},
    baseline_vars={_pylist(cfg.baseline_vars)},
    local_vars={_pylist(cfg.local_vars)},
)

builder, evaluator, metadata = cmf.build_jax_count_evaluator(
    id_col={cfg.id_col!r},
    {offset_kw}
    {group_kw}
    variables={_pylist(aux_vars)},
    constraints=constraints,
    max_latent_classes={cfg.max_latent_classes!r},
    R={cfg.r_draws!r},
    default_roles={cfg.default_roles!r},
    force_aadt_term={cfg.force_aadt_term!r},
)
print("CMF term map (baseline/local -> internal names):", metadata["term_map"])

output_config = SearchOutputConfig(
    output_dir={cfg.output_dir!r},
    experiment_name={cfg.experiment_name!r},
    search_description={cfg.search_description!r},
    save_json=True,
)

print("\\nRunning search (algo={cfg.algo!r})...")
result = builder.run(
    evaluator,
    algo={cfg.algo!r},
    max_iter={cfg.max_iter!r},
    seed={cfg.seed!r},
    output_config=output_config,
)

elapsed = time.time() - t0
best = extract_search_best(result)
print(f"\\nSearch finished in {{elapsed:.1f}}s")
print("Best BIC:", best["best_bic"])
print("Saved to:", result.get("saved_to"))

best_spec = evaluator.build_spec(best["best_decision"])
out_dir = Path({cfg.output_dir!r})
out_dir.mkdir(parents=True, exist_ok=True)

fit_families = {_pylist(cfg.fit_model_families)}
fits = {{}}
for family in fit_families:
    print(f"\\nRefitting best structure as {{family!r}} with R={cfg.final_r_draws}...")
    fits[family] = builder.fit_manual_model(manual_spec=best_spec, model=family, R={cfg.final_r_draws!r})
    summary = extract_summary(fits[family])
    print(f"  {{family}} summary:")
    for k, v in summary.items():
        print(f"    {{k:>10}}: {{v}}")

if len(fits) > 1:
    print("\\nModel family comparison (sorted by BIC):")
    comparison = compare_models(fits)
    print(comparison.to_string(index=False))
    comparison.to_csv(out_dir / ({cfg.experiment_name!r} + "_family_comparison.csv"), index=False)
'''


# ─────────────────────────────────────────────────────────────────────────────
# metacountregressor — Pavement deterioration (clusterwise log-log search)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PavementConfig:
    data_path: str
    hpc_data_filename: str
    psi_col: str
    segment_col: str
    variable_names: list[str]
    categorical_vars: list[str]
    continuous_cols: list[str] = field(default_factory=list)
    min_observations: int = 300
    level_of_significance: float = 0.05
    max_vif: float = 10.0
    temp_init: float = 10.0
    cooling_rate: float = 0.97
    boltzmann: float = 80.0
    n_changes: int = 80
    n_neighbors: int = 3
    cluster_mode: str = "fixed"        # fixed | search_k
    n_clusters: int = 2
    k_min: int = 2
    k_max: int = 6
    seed: int = 42
    max_iterations: int = 1000
    run_temporal_comparison: bool = True
    output_dir: str = "results"
    experiment_name: str = "pavement_run"


def generate_pavement_script(cfg: PavementConfig, for_hpc: bool) -> str:
    data_expr = repr(cfg.hpc_data_filename) if for_hpc else repr(cfg.data_path)
    load_block = f"df = pd.read_csv({data_expr})"

    if cfg.cluster_mode == "search_k":
        search_block = f'''search = opt.search_k(
    df_log, k_min={cfg.k_min!r}, k_max={cfg.k_max!r}, seed={cfg.seed!r},
    max_iterations={cfg.max_iterations!r}, verbose=True,
)
result = search["best_result"]
n_clusters_used = search["best_K"]
print("\\nBest K:", search["best_K"])
print("BIC by K:", search["bic_by_K"])'''
    else:
        search_block = f'''result = opt.fit(
    df_log, n_clusters={cfg.n_clusters!r}, seed={cfg.seed!r},
    max_iterations={cfg.max_iterations!r}, verbose=True,
)
n_clusters_used = {cfg.n_clusters!r}'''

    temporal_block = ""
    if cfg.run_temporal_comparison:
        temporal_block = f'''
print("\\nFitting temporal-error-structure comparison (OLS/AR1/Random Walk/Near-Unit-Root) per cluster...")
cmp = PavementTemporalComparison(
    {_pylist(cfg.variable_names)}, {{{", ".join(repr(v) for v in cfg.categorical_vars)}}},
    psi_col={cfg.psi_col!r}, segment_col={cfg.segment_col!r},
)
df_cmp = cmp.compare(df_log, result["clusters"], n_clusters=n_clusters_used)
print(df_cmp.to_string(index=False))
df_cmp.to_csv(out_dir / ({cfg.experiment_name!r} + "_temporal_comparison.csv"), index=False)
'''

    return f'''"""Auto-generated by ModelZoo GUI — Pavement deterioration run: {cfg.experiment_name}

Run locally:
    python run_pavement.py

Clusterwise log-log (power-law) regression of pavement serviceability (PSI)
on age/traffic/condition variables, with a simulated-annealing joint
cluster+variable-selection search (PavementCLROptimizer), optionally
followed by a comparison of alternative temporal error structures
(PavementTemporalComparison: OLS, AR(1), Random Walk w/ drift, Near-Unit-Root).
"""
import os
import sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import time
from pathlib import Path
import pandas as pd
from metacountregressor import (
    PavementCLROptimizer,
    PavementTemporalComparison,
    log_transform_pavement,
)

print("=" * 70)
print("Pavement deterioration run: {cfg.experiment_name}")
print("=" * 70)

t0 = time.time()
{load_block}
print(f"Loaded data: {{df.shape[0]}} rows x {{df.shape[1]}} cols")

df_log = log_transform_pavement(
    df, psi_col={cfg.psi_col!r}, continuous_cols={_pylist(cfg.continuous_cols)},
)

opt = PavementCLROptimizer(
    variable_names={_pylist(cfg.variable_names)},
    categorical_vars={{{", ".join(repr(v) for v in cfg.categorical_vars)}}},
    psi_col={cfg.psi_col!r},
    segment_col={cfg.segment_col!r},
    min_observations={cfg.min_observations!r},
    level_of_significance={cfg.level_of_significance!r},
    max_vif={cfg.max_vif!r},
    temp_init={cfg.temp_init!r},
    cooling_rate={cfg.cooling_rate!r},
    boltzmann={cfg.boltzmann!r},
    n_changes={cfg.n_changes!r},
    n_neighbors={cfg.n_neighbors!r},
)

print("\\nRunning clusterwise search (mode={cfg.cluster_mode!r})...")
{search_block}
print("BIC:", result["bic"], "| iterations:", result["iterations"], "| converged:", result.get("convergence"))

out_dir = Path({cfg.output_dir!r})
out_dir.mkdir(parents=True, exist_ok=True)
{temporal_block}
preds = opt.predict(df, result["fits"], result["clusters"])
df_out = df.copy()
df_out["psi_pred"] = preds
df_out.to_csv(out_dir / ({cfg.experiment_name!r} + "_predictions.csv"), index=False)

elapsed = time.time() - t0
print(f"\\nFinished in {{elapsed:.1f}}s")
print(f"Saved outputs to {{out_dir}}")
'''
