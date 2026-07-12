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
# takes_zone: mode accepts the second positional arg (zone number / list / file)
# ga: mode honours the GA_* environment variables
# hhts: mode accepts --search / --sa-iter / --sa-temp / --sa-model
ABM_MODES: dict[str, dict] = {
    "main": {
        "label": "Main pipeline",
        "help": "Full estimation + synthetic-population pipeline (stages 1-6).",
        "takes_zone": True, "ga": False, "hhts": False,
    },
    "ga": {
        "label": "GA feature selection",
        "help": "GA-only feature-selection workflow across stages.",
        "takes_zone": True, "ga": True, "hhts": False,
    },
    "ga_staged": {
        "label": "GA staged + prediction",
        "help": "Staged GA plus full population application/validation.",
        "takes_zone": True, "ga": True, "hhts": False,
    },
    "ga_stage5_smoke": {
        "label": "GA stage-5 smoke test",
        "help": "Quick stage-5-only GA sanity check.",
        "takes_zone": True, "ga": True, "hhts": False,
    },
    "mcr_search": {
        "label": "MetaCount structure search",
        "help": "MetaCountRegressor structure search over pipeline data.",
        "takes_zone": True, "ga": False, "hhts": False,
    },
    "hhts_core": {
        "label": "HHTS core models (fixed specs)",
        "help": "HHTS stages 0-4 with fixed historical model specs (validation only).",
        "takes_zone": False, "ga": False, "hhts": True,
    },
    "hhts_search": {
        "label": "HHTS specification search",
        "help": "HHTS stages 0-4 with SA specification search (pick a strategy preset).",
        "takes_zone": False, "ga": False, "hhts": True,
    },
    "safety_baseline": {
        "label": "Safety experiment — baseline",
        "help": "Main pipeline WITH safety skims (CRASH/CRASHPT) in stage 5.",
        "takes_zone": True, "ga": False, "hhts": False,
    },
    "safety_nosafety": {
        "label": "Safety experiment — no safety",
        "help": "Main pipeline WITHOUT safety skims in stage 5.",
        "takes_zone": True, "ga": False, "hhts": False,
    },
    "safety_compare": {
        "label": "Safety experiment — compare",
        "help": "Compare safety_baseline vs safety_nosafety runs and write a report. "
                "Run after both experiment runs have completed.",
        "takes_zone": False, "ga": False, "hhts": False,
    },
    "list_searches": {
        "label": "List HHTS search presets",
        "help": "Print the available --search preset names and exit.",
        "takes_zone": False, "ga": False, "hhts": False,
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


@dataclass
class AbmRunConfig:
    code_dir: str = DEFAULT_ABM_CODE_DIR
    mode: str = "main"
    zone: str = ""                     # zone number, comma list, or file path
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

    # ── local command ────────────────────────────────────────────────────
    def build_argv(self, python_exe: str) -> list[str]:
        argv = [python_exe, "-u", PIPELINE_ENTRY, self.mode]
        if self._info().get("takes_zone") and self.zone.strip():
            argv.append(self.zone.strip())
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
        if self._info().get("takes_zone") and self.zone.strip():
            pairs.append(f"ZONE_ARG={self.zone.strip()}")
        pairs.append(f"PIPELINE_ACCELERATOR={self.accelerator}")
        if self.run_tag.strip():
            pairs.append(f"PIPELINE_RUN_TAG={self.run_tag.strip()}")
        if self._info().get("ga"):
            pairs.append(f"GA_ESTIMATOR={self.ga_estimator}")
            pairs.append(f"GA_BUDGET={self.ga_budget}")
            if self.ga_n_restarts is not None:
                pairs.append(f"GA_N_RESTARTS={self.ga_n_restarts}")
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
