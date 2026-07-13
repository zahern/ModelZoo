from pathlib import Path

import streamlit as st

from lib.abm import (
    ABM_MODES,
    AbmRunConfig,
    DEFAULT_ABM_CODE_DIR,
    GA_BUDGETS,
    GA_ESTIMATORS,
    HHTS_SEARCH_PRESETS,
    MCR_ALGOS,
    MCR_FAMILIES,
    check_code_dir,
)
from lib.env import DEFAULT_ENGINE_PYTHON
from lib.runner import stream_cmd

st.set_page_config(page_title="ABM Pipeline — ModelZoo", page_icon="🏙️", layout="wide")
st.title("🏙️ ABM pipeline runner")
st.caption(
    "Run the SEQ activity-based model pipeline (Z:\\test_runs_tours\\code) — "
    "main estimation, GA/SA feature-selection strategies, HHTS search presets, "
    "and the safety-skim experiment — locally or as HPC PBS jobs."
)

engine_python = st.session_state.get("engine_python", DEFAULT_ENGINE_PYTHON)

# ── 1. Pipeline location ─────────────────────────────────────────────────────
st.subheader("1. Pipeline code directory")
code_dir = st.text_input("ABM code directory", value=DEFAULT_ABM_CODE_DIR)

checks = check_code_dir(code_dir)
cols = st.columns(len(checks))
for col, (name, ok) in zip(cols, checks.items()):
    col.metric(name, "found" if ok else "missing",
               delta="OK" if ok else "FAIL",
               delta_color="normal" if ok else "inverse")
if not checks.get("pipeline_logger.py"):
    st.error("pipeline_logger.py not found — set the correct code directory above.")
    st.stop()

# ── 2. Run mode ──────────────────────────────────────────────────────────────
st.subheader("2. Run mode")
mode = st.selectbox(
    "Pipeline mode",
    list(ABM_MODES.keys()),
    format_func=lambda m: f"{m} — {ABM_MODES[m]['label']}",
)
info = ABM_MODES[mode]
st.info(info["help"])
with st.expander("What does this run mode actually do?", expanded=True):
    st.markdown(info["detail"])

# ── 3. Strategy & options ────────────────────────────────────────────────────
st.subheader("3. Strategy & options")

zone = ""
mcr_data_path = ""
mcr_family, mcr_algo, mcr_max_iter, mcr_r = "count", "sa", 2000, 200
search_preset = ""
sa_iter = sa_temp = None
sa_model = ""
ga_estimator, ga_budget = "default", "standard"
ga_n_restarts = None
ga_use_bandit_sa = False

if info["second_arg"] == "zone":
    zone = st.text_input(
        "Zone selector (optional)",
        value="",
        help="A zone number (e.g. 40), a comma-separated list (40,124,200), "
             "or a path to a zone file. Leave blank for the full region.",
    )
elif info["second_arg"] == "data_path":
    mcr_data_path = st.text_input(
        "External dataset CSV (optional)",
        value="",
        help="Path to a CSV to search over. Leave blank to use the bundled Example 16-3 "
             "crash-frequency dataset — this is NOT a zone selector, mcr_search doesn't touch "
             "the ABM pipeline or synthetic population at all.",
    )
    c1, c2 = st.columns(2)
    with c1:
        mcr_family = st.selectbox("Model family (MCR_FAMILY)", list(MCR_FAMILIES.keys()))
        st.caption(MCR_FAMILIES[mcr_family])
        mcr_algo = st.selectbox("Search algorithm (MCR_ALGO)", MCR_ALGOS)
    with c2:
        mcr_max_iter = st.number_input("Max iterations (MCR_MAX_ITER)", 50, 100000, 2000, step=50)
        mcr_r = st.number_input("Halton draws (MCR_R)", 25, 2000, 200, step=25)

if info["hhts"]:
    c1, c2 = st.columns(2)
    with c1:
        default_preset = "core_fixed" if mode == "hhts_core" else "nested_standard"
        search_preset = st.selectbox(
            "Search strategy preset (--search)",
            list(HHTS_SEARCH_PRESETS.keys()),
            index=list(HHTS_SEARCH_PRESETS.keys()).index(default_preset),
        )
        st.caption(HHTS_SEARCH_PRESETS[search_preset])
    with c2:
        sa_model = st.selectbox("Stage-4 search model override (--sa-model)",
                                ["(preset default)", "nested", "mnl"])
        sa_model = "" if sa_model == "(preset default)" else sa_model
        sa_iter = st.number_input("SA iterations override (--sa-iter, 0 = preset default)",
                                  0, 10000, 0, step=50) or None
        sa_temp = st.number_input("SA temperature override (--sa-temp, 0 = preset default)",
                                  0.0, 100.0, 0.0, step=0.25) or None

if info["ga"]:
    c1, c2 = st.columns(2)
    with c1:
        ga_estimator = st.selectbox("GA estimator (GA_ESTIMATOR)", list(GA_ESTIMATORS.keys()),
                                    index=list(GA_ESTIMATORS.keys()).index("sa_bandit"))
        st.caption(GA_ESTIMATORS[ga_estimator])
        ga_use_bandit_sa = st.checkbox(
            "Bandit-guided SA inner estimators (GA_USE_BANDIT_SA=1)",
            value=ga_estimator == "sa_bandit",
        )
    with c2:
        ga_budget = st.selectbox("Search budget (GA_BUDGET)", list(GA_BUDGETS.keys()),
                                 index=list(GA_BUDGETS.keys()).index("standard"))
        st.caption(GA_BUDGETS[ga_budget])
        restarts = st.number_input("GA restarts override (GA_N_RESTARTS, -1 = budget default)",
                                   -1, 10, -1)
        ga_n_restarts = None if restarts < 0 else int(restarts)

c1, c2, c3 = st.columns(3)
with c1:
    run_tag = st.text_input("Run tag (PIPELINE_RUN_TAG, optional)", value="",
                            help="Keeps repeated runs of the same mode in separate output folders.")
with c2:
    accelerator = st.selectbox("Accelerator (PIPELINE_ACCELERATOR)", ["cpu", "gpu"])
with c3:
    threads = st.number_input("Math threads per process (0 = library default)", 0, 32, 0)

st.subheader("4. HPC resources (for PBS submission)")
c1, c2, c3 = st.columns(3)
with c1:
    ncpus = st.number_input("ncpus", 1, 64, 8)
with c2:
    mem = st.text_input("Memory", value="250GB")
with c3:
    walltime = st.text_input("Walltime", value="23:00:00")

cfg = AbmRunConfig(
    code_dir=code_dir, mode=mode, zone=zone, mcr_data_path=mcr_data_path,
    search_preset=search_preset if info["hhts"] else "",
    sa_iter=sa_iter, sa_temp=sa_temp, sa_model=sa_model,
    ga_estimator=ga_estimator, ga_budget=ga_budget,
    ga_n_restarts=ga_n_restarts, ga_use_bandit_sa=ga_use_bandit_sa,
    mcr_family=mcr_family, mcr_algo=mcr_algo,
    mcr_max_iter=int(mcr_max_iter), mcr_r=int(mcr_r),
    run_tag=run_tag, accelerator=accelerator,
    threads=int(threads) or None,
    ncpus=int(ncpus), mem=mem, walltime=walltime,
)

st.divider()

# ── 5. Preview, run, submit ──────────────────────────────────────────────────
tab_cmd, tab_hpc = st.tabs(["Local command", "HPC submission commands"])

with tab_cmd:
    st.code(cfg.command_preview(engine_python), language="bash")

with tab_hpc:
    st.markdown("**Single mode** — generic PBS worker (`pbs_pipeline_mode.pbs`):")
    st.code(cfg.build_qsub_command(), language="bash")
    st.markdown("**Batch submit** — `submit_pipeline_runs.sh` (main → ga → ga_staged):")
    st.code(cfg.build_submit_sh_command(modes=["main", "ga", "ga_staged"], sequential=True),
            language="bash")
    st.markdown("**Safety experiment chain** (baseline → nosafety → compare):")
    st.code(cfg.build_submit_sh_command(modes=["main"], include_safety=True, sequential=True),
            language="bash")
    if info["ga"]:
        st.markdown("**GA parallel stage chain** — stages 1-5 fan-out, merge, then `ga_staged` prediction:")
        st.code(cfg.build_ga_parallel_chain_command(), language="bash")
    st.caption("Run these from the code directory on the cluster. See PBS_RUN_GUIDE.md there for details.")

if st.button("Run locally now", type="primary", use_container_width=True):
    argv = cfg.build_argv(engine_python)
    env = cfg.build_env()
    st.info(f"Working directory: `{code_dir}`")
    console = st.empty()
    lines: list[str] = []
    with st.spinner(f"Running pipeline_logger.py {mode} ..."):
        for line in stream_cmd(argv, cwd=code_dir, env=env):
            lines.append(line)
            console.code("\n".join(lines[-400:]), language="text")
    st.success("Run finished — outputs are under the pipeline's runs/ directory in the code folder.")
