from pathlib import Path

import pandas as pd
import streamlit as st

from lib.env import DEFAULT_ENGINE_PYTHON
from lib.pbs_gen import PbsConfig, generate_pbs_script
from lib.script_gen import SearchLibriumConfig, generate_searchlibrium_script
from lib.ui_common import render_run_and_export

st.set_page_config(page_title="SearchLibrium — ModelZoo", page_icon="🔎", layout="wide")
st.title("🔎 SearchLibrium runner")
st.caption("Discrete-choice model structure search: Multinomial/Mixed/Nested Logit, Random Regret, and more.")

engine_python = st.session_state.get("engine_python", DEFAULT_ENGINE_PYTHON)

st.subheader("1. Data")
st.caption("Long-format choice data — one row per alternative per observation.")
data_choice = st.radio("Data source", ["Upload CSV", "Path on disk"], horizontal=True)

df: pd.DataFrame | None = None
data_path = ""
uploaded_path: Path | None = None

if data_choice == "Upload CSV":
    up = st.file_uploader("CSV file", type=["csv"])
    if up is not None:
        job_dir = Path(__file__).resolve().parent.parent.parent / "generated_jobs" / "_uploads"
        job_dir.mkdir(parents=True, exist_ok=True)
        uploaded_path = job_dir / up.name
        uploaded_path.write_bytes(up.getvalue())
        data_path = str(uploaded_path)
        df = pd.read_csv(uploaded_path)
else:
    data_path = st.text_input("CSV path (must be readable by the engine interpreter)", value="")
    if data_path and Path(data_path).exists():
        try:
            df = pd.read_csv(data_path, nrows=5000)
        except Exception as e:
            st.warning(f"Could not preview file: {e}")

if df is not None:
    st.dataframe(df.head(20), use_container_width=True, height=200)
    columns = list(df.columns)
else:
    columns = []
    st.info("Provide data to continue.")

st.subheader("2. Column mapping")
c1, c2, c3 = st.columns(3)
with c1:
    choice_col = st.selectbox("Choice column (0/1)", columns, index=columns.index("CHOICE") if "CHOICE" in columns else 0) if columns else st.text_input("Choice column", "CHOICE")
    alt_col = st.selectbox("Alternative column", columns, index=columns.index("alt") if "alt" in columns else 0) if columns else st.text_input("Alternative column", "alt")
with c2:
    choice_id_col = st.selectbox("Observation/choice-set ID column", columns, index=columns.index("custom_id") if "custom_id" in columns else 0) if columns else st.text_input("Choice ID column", "custom_id")
    ind_id_options = ["(none)"] + columns
    ind_id_col = st.selectbox("Individual ID column (panel data)", ind_id_options)
    ind_id_col = None if ind_id_col == "(none)" else ind_id_col
with c3:
    base_alt = st.text_input("Base alternative", value="")

st.subheader("3. Candidate variables")
asvarnames = st.multiselect("Alternative-specific variables (varnames)", [c for c in columns if c not in {choice_col, alt_col, choice_id_col}])
isvarnames = st.multiselect("Individual-specific variables (optional)", [c for c in columns if c not in {choice_col, alt_col, choice_id_col, *asvarnames}])

st.subheader("4. Model & search settings")
c1, c2, c3 = st.columns(3)
with c1:
    models = st.multiselect(
        "Model types to search over",
        ["multinomial", "mixed_logit", "random_regret", "mixed_random_regret", "nested_logit", "ordered_logit"],
        default=["multinomial"],
    )
    criterion = st.selectbox("Objective (minimise)", ["bic", "aic", "loglik"])
with c2:
    allow_random = st.checkbox("Allow random parameters", value="mixed_logit" in models or "mixed_random_regret" in models)
    allow_bcvars = st.checkbox("Allow Box-Cox transforms")
    allow_corvars = st.checkbox("Allow correlated random parameters")
with c3:
    algorithm = st.selectbox("Search algorithm", ["sa", "hs", "sapbil", "banditsa", "hspbil"])
    p_val = st.number_input("Significance threshold (p_val)", 0.001, 0.5, 0.05, step=0.01)
    seed = st.number_input("Run ID / seed", 1, 999999, 1)

c1, c2 = st.columns(2)
with c1:
    n_draws = st.number_input("Halton draws (n_draws, for mixed models)", 50, 5000, 500, step=50)
with c2:
    maxiter = st.number_input("Max MLE iterations per fit (maxiter)", 50, 10000, 2000, step=50)

nests_json = lambdas_json = None
if "nested_logit" in models:
    st.caption("Nested logit requires nest structure, e.g. `{\"PublicTransport\": [0, 1], \"Private\": [2, 3]}`")
    nests_json = st.text_area("nests (JSON)", value='{"nest_1": [0, 1]}')
    lambdas_json = st.text_area("lambdas (JSON)", value='{"nest_1": 0.8}')

st.subheader("5. Job naming & output")
c1, c2, c3 = st.columns(3)
with c1:
    experiment_name = st.text_input("Experiment name", value="searchlibrium_run")
with c2:
    output_dir = st.text_input("Output directory", value="results")
with c3:
    ncpus = st.number_input("HPC ncpus", 1, 64, 4)
c1, c2 = st.columns(2)
with c1:
    mem_gb = st.number_input("HPC mem (GB)", 4, 512, 32)
with c2:
    walltime = st.text_input("HPC walltime", value="24:00:00")

st.divider()

ready = bool(models and (asvarnames or isvarnames) and df is not None)
if not ready:
    st.warning("Select data, at least one model type, and at least one variable to generate a script.")
else:
    hpc_data_filename = Path(data_path).name

    cfg = SearchLibriumConfig(
        data_path=data_path,
        hpc_data_filename=hpc_data_filename,
        choice_col=choice_col, alt_col=alt_col, choice_id_col=choice_id_col,
        ind_id_col=ind_id_col, base_alt=base_alt,
        asvarnames=asvarnames, isvarnames=isvarnames,
        models=models, allow_random=allow_random, allow_bcvars=allow_bcvars,
        allow_corvars=allow_corvars, p_val=p_val, n_draws=int(n_draws), maxiter=int(maxiter),
        criterion=criterion, algorithm=algorithm, seed=int(seed),
        nests_json=nests_json, lambdas_json=lambdas_json,
        output_dir=output_dir, experiment_name=experiment_name,
    )

    local_script = generate_searchlibrium_script(cfg, for_hpc=False)
    hpc_script = generate_searchlibrium_script(cfg, for_hpc=True)
    pbs_script = generate_pbs_script(PbsConfig(
        job_name=experiment_name, script_filename="run_searchlibrium.py",
        ncpus=int(ncpus), mem_gb=int(mem_gb), walltime=walltime,
    ))

    render_run_and_export(
        key_prefix="searchlibrium",
        local_script=local_script, local_script_name="run_searchlibrium.py",
        hpc_script=hpc_script, hpc_script_name="run_searchlibrium.py",
        pbs_script=pbs_script, pbs_script_name=f"{experiment_name}.pbs",
        engine_python=engine_python,
        data_file_to_bundle=uploaded_path,
    )
