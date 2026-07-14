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
