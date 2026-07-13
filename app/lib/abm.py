"""ABM pipeline integration for Z:\\test_runs_tours\\code.

Everything here mirrors the CLI surface of ``pipeline_logger.py`` and the PBS
launchers (``pbs_pipeline_mode.pbs``, ``submit_pipeline_runs.sh``,
``job_submit_ga_parallel_chain.pbs``) in that directory — see PBS_RUN_GUIDE.md
there. If the pipeline grows a new mode or env var, update the tables below.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_ABM_CODE_DIR = r"Z:\test_runs_tours\code"
PIPELINE_ENTRY = "pipeline_logger.py"

# ── Run modes (pipeline_logger.py positional `mode`) ─────────────────────────
# second_arg: what the second positional CLI arg means for this mode —
#   "zone"      — zone number, comma list ("40,124,200"), or an order_N.csv path
#   "data_path" — optional path to an external CSV (mcr_search only; falls back
#                 to the bundled Example 16-3 dataset when blank/not a file)
#   None        — second positional arg is unused by this mode
# ga: mode honours the GA_* environment variables (see GA_ESTIMATORS/GA_BUDGETS)
# hhts: mode accepts --search / --sa-iter / --sa-temp / --sa-model
# Sourced directly from pipeline_logger.py's mode dispatch (~line 8544) and each
# target function's docstring — see detail[] for the exact function + line.
ABM_MODES: dict[str, dict] = {
    "main": {
        "label": "Main pipeline",
        "help": "Full estimation + synthetic-population pipeline, all zones, stages 1-6.",
        "detail": (
            "Runs `main()` (pipeline_logger.py:5836). Phase 1 estimates every stage model "
            "from HHTS survey data only; Phase 2 applies those models to the full synthetic "
            "population (every order_N.csv under SYNTH_DIR, or a single zone/list if given) "
            "through stages 1-6 to produce final activity plans. This is the reference/default "
            "production run — no GA feature search, no HHTS spec search. Runs data_checks.py "
            "first and logs (non-fatal) warnings before committing hours of compute. "
            "Output: runs/ directory (PKL model files, HTML reports, synthetic plan tables)."
        ),
        "second_arg": "zone", "ga": False, "hhts": False,
    },
    "ga": {
        "label": "GA feature selection",
        "help": "GA feature search (HHTS only) then apply results to the full pipeline.",
        "detail": (
            "Runs `GA_main()` (pipeline_logger.py:6321). Runs genetic-algorithm feature "
            "selection across all stages using HHTS survey data ONLY (no synth population "
            "loaded yet — keeps memory low during search). Once GA converges, loads the "
            "synthetic population and applies the GA-selected feature set per stage via "
            "apply_ga_results_to_pipeline(). Choose the estimator/budget below — this is the "
            "main entry point for the GA_ESTIMATOR / GA_BUDGET knobs. "
            "Output: runs/.../ga_results.pkl + the applied pipeline output."
        ),
        "second_arg": "zone", "ga": True, "hhts": False,
    },
    "ga_staged": {
        "label": "GA staged + prediction",
        "help": "GA stage-by-stage, then full population prediction — the production GA run.",
        "detail": (
            "Runs `GA_STAGED()` (pipeline_logger.py:6413). Same idea as 'ga' but runs GA "
            "one stage at a time (lower peak memory) and defers loading the synthetic "
            "population until ALL GA stages finish, then runs the full stage 1-6 pipeline "
            "on it once. Documented walltime on a 1-CPU/94GB PBS node: GA stages 1-5 ≈ 3.5h, "
            "synth load ≈ 6min, pipeline stages ≈ 4-5h — total ≈ 9-10h. Set "
            "GA_STAGED_REUSE_ESTIMATOR=1 (via run tag / a prior 'ga' run) to skip "
            "re-estimation and only pay the full-memory cost for the prediction phase."
        ),
        "second_arg": "zone", "ga": True, "hhts": False,
    },
    "ga_stage5_smoke": {
        "label": "GA stage-5 smoke test",
        "help": "Fast stage-5-only GA sanity check — does NOT run the full population.",
        "detail": (
            "Runs `GA_STAGE5_SMOKE()` (pipeline_logger.py:6657). Forces GA_BUDGET='smoke' "
            "and runs ONLY the stage-5 GA search, to verify wiring/checkpoint/report "
            "generation before paying for a full 'ga' or 'ga_staged' run. Intentionally does "
            "not run full synth prediction. Note: the second positional arg is accepted by "
            "the CLI but unused by this function's implementation — leave the zone field "
            "blank, it has no effect here."
        ),
        "second_arg": None, "ga": True, "hhts": False,
    },
    "mcr_search": {
        "label": "MetaCount structure search",
        "help": "Standalone MetaCountRegressor structure search — not a full pipeline run.",
        "detail": (
            "Runs `run_mcr_search()` (pipeline_logger.py:8244). Runs a single MetaCountRegressor "
            "structure search job, configured entirely via MCR_FAMILY / MCR_ALGO / MCR_MAX_ITER "
            "/ MCR_R env vars (below) — this mode does NOT touch the ABM pipeline stages at all. "
            "The second CLI arg here is an optional path to an external CSV to search over; if "
            "omitted (or not a real file) it falls back to the bundled Example 16-3 crash-frequency "
            "dataset. This is unrelated to zone selection despite sharing the same CLI position."
        ),
        "second_arg": "data_path", "ga": False, "hhts": False,
    },
    "hhts_core": {
        "label": "HHTS core models (fixed specs)",
        "help": "HHTS stages 0-4 only, fixed historical specs — validation, not production.",
        "detail": (
            "Runs `run_hhts_optional_pipeline(force_mode='core')` (pipeline_logger.py:8165), "
            "which delegates to the legacy `main_2.py: run_core()`. Runs HHTS stages 0-4 with "
            "the historically-fixed model specifications (no SA search) — use this to validate "
            "that a preset's fixed specs still fit before running a search variant. Does not "
            "touch the synthetic population."
        ),
        "second_arg": None, "ga": False, "hhts": True,
    },
    "hhts_search": {
        "label": "HHTS specification search",
        "help": "HHTS stages 0-4 with an SA specification search (pick a preset below).",
        "detail": (
            "Runs `run_hhts_optional_pipeline(force_mode='search')` (pipeline_logger.py:8165), "
            "delegating to `main_2.py: run_search()`. Runs HHTS stages 0-4 with a simulated-"
            "annealing search over model specifications, using the chosen preset's search_model "
            "(nested/mnl), n_iter, and temperature (overridable below). Does not touch the "
            "synthetic population."
        ),
        "second_arg": None, "ga": False, "hhts": True,
    },
    "safety_baseline": {
        "label": "Safety experiment — baseline",
        "help": "Main pipeline WITH safety skims (CRASH/CRASHPT) included in stage 5.",
        "detail": (
            "Sets SAFETY_EXPERIMENT_MODE=baseline then runs the same `main()` as the 'Main "
            "pipeline' mode (pipeline_logger.py:8577-8583) — full stages 1-6, safety skims "
            "included in stage 5 destination choice. Run this AND 'safety_nosafety' before "
            "using 'safety_compare'. Output lands under runs/safety_baseline/."
        ),
        "second_arg": "zone", "ga": False, "hhts": False,
    },
    "safety_nosafety": {
        "label": "Safety experiment — no safety",
        "help": "Main pipeline WITHOUT safety skims — the counterfactual for comparison.",
        "detail": (
            "Sets SAFETY_EXPERIMENT_MODE=nosafety then runs the same `main()` pipeline with "
            "CRASH/CRASHPT skims excluded from stage 5 estimation (pipeline_logger.py:8577-8583). "
            "Output lands under runs/safety_nosafety/. Run alongside 'safety_baseline' — the "
            "pair is the whole point of the experiment."
        ),
        "second_arg": "zone", "ga": False, "hhts": False,
    },
    "safety_compare": {
        "label": "Safety experiment — compare",
        "help": "Compare completed baseline vs no-safety runs and write a report.",
        "detail": (
            "Runs `compare_safety_runs()` (pipeline_logger.py:8584-8589), reading "
            "runs/safety_baseline/ and runs/safety_nosafety/ and writing a comparison report. "
            "Only meaningful AFTER both of those runs have completed — running this first will "
            "fail or compare stale/missing data."
        ),
        "second_arg": None, "ga": False, "hhts": False,
    },
    "list_searches": {
        "label": "List HHTS search presets",
        "help": "Print the available --search preset names and exit — no compute, seconds.",
        "detail": (
            "Runs `_print_model_zoo_searches()` (pipeline_logger.py:8576), printing the same "
            "preset table shown below (HHTS_SEARCH_PRESETS) as read live from model_zoo.py. "
            "Useful as a fast sanity check that the code directory and presets are wired up "
            "correctly before committing to a real hhts_core/hhts_search run."
        ),
        "second_arg": None, "ga": False, "hhts": False,
    },
}

# ── HHTS strategy presets (model_zoo.py SEARCH_PRESETS) ──────────────────────
HHTS_SEARCH_PRESETS: dict[str, str] = {
    "core_fixed": "HHTS stages 0-4 core run with fixed historical specs.",
    "nested_fast": "Fast nested-logit SA search for stage-4 timing cells (n_iter=150).",
    "nested_standard": "Standard nested-logit SA search budget (n_iter=400).",
    "mnl_fast": "Fast multinomial-logit SA search budget (n_iter=150).",
    "selection_screen": "Binary selection screen using SearchLibrium probit + Heckman.",
    "survival_screen": "Duration screen using MetaCount random-effects AFT models.",
}

# ── GA strategy knobs (environment variables) ────────────────────────────────
GA_ESTIMATORS: dict[str, str] = {
    "default": "Ordered-logit / OLS / MNL statsmodels estimators (fastest).",
    "metacount": "MetaCount NB2 for stage 1, Tobit/lognormal for stages 2-3.",
    "searchlibrium": "SearchLibrium MNL for stage 4; default estimators for 1-3.",
    "both": "MetaCount for stages 1-3, SearchLibrium for stage 4.",
    "hybrid": "Same as 'both' but with no statsmodels fallback.",
    "sa_bandit": "Blended: MetaCount 1-3, SearchLibrium 4, Larch 5; bandit-guided SA outer loop.",
}

GA_BUDGETS: dict[str, str] = {
    "smoke": "pop=8, gens=2 — sanity check only.",
    "ultrafast": "pop=25, gens=30 — ~45 min/purpose for MetaCount.",
    "fast": "pop=40, gens=75, restarts=1.",
    "standard": "pop=60, gens=263, restarts=1 (default).",
    "thorough": "pop=100, gens=540, restarts=2.",
}

# ── mcr_search-only knobs (env vars read by run_mcr_search(), pipeline_logger.py:8270-8283) ──
MCR_FAMILIES: dict[str, str] = {
    "count": "Crash-frequency count family (Poisson/NB). Uses bundled Example 16-3 data if no CSV given.",
    "duration": "Survival/AFT-style positive-duration family.",
    "linear": "Gaussian continuous-outcome family.",
    "cmf": "Crash Modification Function family.",
}
MCR_ALGOS = ["sa", "de", "hs"]


@dataclass
class AbmRunConfig:
    code_dir: str = DEFAULT_ABM_CODE_DIR
    mode: str = "main"
    zone: str = ""                     # zone number, comma list, or file path (second_arg == "zone")
    mcr_data_path: str = ""            # optional external CSV (second_arg == "data_path", mcr_search only)
    # HHTS search strategy
    search_preset: str = ""            # --search
    sa_iter: int | None = None         # --sa-iter
    sa_temp: float | None = None       # --sa-temp
    sa_model: str = ""                 # --sa-model nested|mnl
    # GA strategy (env vars)
    ga_estimator: str = "default"
    ga_budget: str = "standard"
    ga_n_restarts: int | None = None
    ga_use_bandit_sa: bool = False
    # mcr_search-only strategy (env vars)
    mcr_family: str = "count"          # MCR_FAMILY
    mcr_algo: str = "sa"               # MCR_ALGO
    mcr_max_iter: int = 2000           # MCR_MAX_ITER
    mcr_r: int = 200                   # MCR_R
    # Run identity / hardware
    run_tag: str = ""                  # PIPELINE_RUN_TAG
    accelerator: str = "cpu"           # PIPELINE_ACCELERATOR
    threads: int | None = None         # OMP/MKL/... thread caps
    # HPC resource shape
    ncpus: int = 8
    mem: str = "250GB"
    walltime: str = "23:00:00"

    def _info(self) -> dict:
        return ABM_MODES.get(self.mode, {})

    def _second_arg_value(self) -> str:
        kind = self._info().get("second_arg")
        if kind == "zone":
            return self.zone.strip()
        if kind == "data_path":
            return self.mcr_data_path.strip()
        return ""

    # ── local command ────────────────────────────────────────────────────
    def build_argv(self, python_exe: str) -> list[str]:
        argv = [python_exe, "-u", PIPELINE_ENTRY, self.mode]
        second_arg = self._second_arg_value()
        if second_arg:
            argv.append(second_arg)
        if self._info().get("hhts"):
            if self.search_preset:
                argv += ["--search", self.search_preset]
            if self.sa_iter:
                argv += ["--sa-iter", str(self.sa_iter)]
            if self.sa_temp:
                argv += ["--sa-temp", str(self.sa_temp)]
            if self.sa_model:
                argv += ["--sa-model", self.sa_model]
        return argv

    def build_env(self) -> dict[str, str]:
        env: dict[str, str] = {"PIPELINE_ACCELERATOR": self.accelerator}
        if self.run_tag.strip():
            env["PIPELINE_RUN_TAG"] = self.run_tag.strip()
        if self._info().get("ga"):
            env["GA_ESTIMATOR"] = self.ga_estimator
            env["GA_BUDGET"] = self.ga_budget
            if self.ga_n_restarts is not None:
                env["GA_N_RESTARTS"] = str(self.ga_n_restarts)
            if self.ga_use_bandit_sa:
                env["GA_USE_BANDIT_SA"] = "1"
        if self.mode == "mcr_search":
            env["MCR_FAMILY"] = self.mcr_family
            env["MCR_ALGO"] = self.mcr_algo
            env["MCR_MAX_ITER"] = str(self.mcr_max_iter)
            env["MCR_R"] = str(self.mcr_r)
        if self.threads:
            for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                env[var] = str(self.threads)
        return env

    def command_preview(self, python_exe: str) -> str:
        env = self.build_env()
        env_prefix = " ".join(f"{k}={v}" for k, v in env.items())
        cmd = " ".join(self.build_argv(python_exe))
        return f"cd {self.code_dir}\n{env_prefix + '  # env' if env_prefix else ''}\n{cmd}".strip()

    # ── HPC commands (run from the code dir on the cluster) ──────────────
    def build_qsub_command(self) -> str:
        """Single-mode submit via the generic pbs_pipeline_mode.pbs worker."""
        pairs = [f"MODE={self.mode}"]
        second_arg = self._second_arg_value()
        if second_arg:
            # pbs_pipeline_mode.pbs forwards ZONE_ARG verbatim as the pipeline's second
            # positional CLI arg — for mcr_search that's the optional data-path, not a zone.
            pairs.append(f"ZONE_ARG={second_arg}")
        pairs.append(f"PIPELINE_ACCELERATOR={self.accelerator}")
        if self.run_tag.strip():
            pairs.append(f"PIPELINE_RUN_TAG={self.run_tag.strip()}")
        if self._info().get("ga"):
            pairs.append(f"GA_ESTIMATOR={self.ga_estimator}")
            pairs.append(f"GA_BUDGET={self.ga_budget}")
            if self.ga_n_restarts is not None:
                pairs.append(f"GA_N_RESTARTS={self.ga_n_restarts}")
        if self.mode == "mcr_search":
            pairs.append(f"MCR_FAMILY={self.mcr_family}")
            pairs.append(f"MCR_ALGO={self.mcr_algo}")
            pairs.append(f"MCR_MAX_ITER={self.mcr_max_iter}")
            pairs.append(f"MCR_R={self.mcr_r}")
            if self.ga_use_bandit_sa:
                pairs.append("GA_USE_BANDIT_SA=1")
        if self.threads:
            pairs.append(f"THREADS={self.threads}")
        gpu_clause = ":ngpus=1" if self.accelerator == "gpu" else ""
        resources = f"-l select=1:ncpus={self.ncpus}:mem={self.mem}{gpu_clause} -l walltime={self.walltime}"
        return f"qsub -v {','.join(pairs)} {resources} pbs_pipeline_mode.pbs"

    def build_submit_sh_command(self, modes: list[str] | None = None,
                                sequential: bool = False,
                                include_safety: bool = False) -> str:
        """Multi-mode submit via submit_pipeline_runs.sh."""
        parts = ["./submit_pipeline_runs.sh"]
        if modes:
            parts.append(f'--modes "{" ".join(modes)}"')
        if self.zone.strip():
            parts.append(f"--zone {self.zone.strip()}")
        parts.append(f"--cpus {self.ncpus}")
        parts.append(f"--mem {self.mem}")
        parts.append(f"--walltime {self.walltime}")
        if self.threads:
            parts.append(f"--threads {self.threads}")
        if self.accelerator == "gpu":
            parts.append("--gpu 1 --accelerator gpu")
        else:
            # submit_pipeline_runs.sh defaults to ACCELERATOR=gpu/GPU_COUNT=1 — must
            # override explicitly or a "cpu" selection here silently runs on GPU.
            parts.append("--gpu 0 --accelerator cpu")
        if sequential:
            parts.append("--sequential")
        if include_safety:
            parts.append("--include-safety")
        return " \\\n    ".join(parts)

    def build_ga_parallel_chain_command(self) -> str:
        """Stage 1-5 fan-out + merge + ga_staged prediction chain."""
        tag = self.run_tag.strip() or "exp_parallel_001"
        pairs = [
            f"RUN_TAG={tag}",
            f"GA_ESTIMATOR={self.ga_estimator}",
            f"GA_BUDGET={self.ga_budget}",
            f"GA_N_RESTARTS={self.ga_n_restarts if self.ga_n_restarts is not None else 0}",
            # NOT "STAGE_SELECT=1:ncpus=...:mem=..." — job_submit_ga_parallel_chain.pbs
            # explicitly documents that colon-containing values break qsub's -v parsing.
            # Pass the atomic pieces and let the script assemble STAGE_SELECT itself.
            f"STAGE_NCPUS={self.ncpus}",
            f"STAGE_MEM={self.mem}",
            f"STAGE_WALLTIME={self.walltime}",
            f"GA_USE_SA={1 if self.ga_use_bandit_sa else 0}",
            f"GA_USE_BANDIT_SA={1 if self.ga_use_bandit_sa else 0}",
        ]
        return f"qsub -v {','.join(pairs)} job_submit_ga_parallel_chain.pbs"


def check_code_dir(code_dir: str) -> dict[str, bool]:
    """Smoke-check that the ABM code directory has the expected entry points."""
    d = Path(code_dir)
    return {
        PIPELINE_ENTRY: (d / PIPELINE_ENTRY).exists(),
        "model_zoo.py": (d / "model_zoo.py").exists(),
        "pbs_pipeline_mode.pbs": (d / "pbs_pipeline_mode.pbs").exists(),
        "submit_pipeline_runs.sh": (d / "submit_pipeline_runs.sh").exists(),
        "job_submit_ga_parallel_chain.pbs": (d / "job_submit_ga_parallel_chain.pbs").exists(),
    }
