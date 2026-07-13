from pathlib import Path

import pandas as pd
import streamlit as st

from lib.env import DEFAULT_ENGINE_PYTHON
from lib.pbs_gen import PbsConfig, generate_pbs_script
from lib.script_gen import ConstraintsConfig, MetaCountConfig, generate_metacount_script
from lib.ui_common import render_run_and_export

st.set_page_config(page_title="MetaCountRegressor — ModelZoo", page_icon="📈", layout="wide")
st.title("📈 MetaCountRegressor runner")
st.caption("Structure search + estimation for count, CMF, duration, and linear models.")

engine_python = st.session_state.get("engine_python", DEFAULT_ENGINE_PYTHON)

st.subheader("1. Data")
data_choice = st.radio("Data source", ["Bundled Example 16-3 dataset", "Upload CSV", "Path on disk"], horizontal=True)

df: pd.DataFrame | None = None
data_path = ""
uploaded_path: Path | None = None
use_bundled_example = False

if data_choice == "Bundled Example 16-3 dataset":
    use_bundled_example = True
    st.info("Will load via `metacountregressor.load_example16_3_model_data()` at run time (crash-frequency data, 31 columns).")
    # Best-effort local preview only; not required for script generation.
    example_cols = ["ID", "FREQ", "LENGTH", "WIDTH", "AADT", "SPEED", "CURVES", "URB",
                     "ACCESS", "GRADEBR", "AVEPRE", "FC_ENCODED", "FC_LABEL", "OFFSET"]
    columns = example_cols
elif data_choice == "Upload CSV":
    up = st.file_uploader("CSV file", type=["csv"])
    if up is not None:
        job_dir = Path(__file__).resolve().parent.parent.parent / "generated_jobs" / "_uploads"
        job_dir.mkdir(parents=True, exist_ok=True)
        uploaded_path = job_dir / up.name
        uploaded_path.write_bytes(up.getvalue())
        data_path = str(uploaded_path)
        df = pd.read_csv(uploaded_path)
        st.dataframe(df.head(20), use_container_width=True, height=200)
    columns = list(df.columns) if df is not None else []
else:
    data_path = st.text_input("CSV path (must be readable by the engine interpreter)", value="")
    if data_path and Path(data_path).exists():
        try:
            df = pd.read_csv(data_path, nrows=5000)
            st.dataframe(df.head(20), use_container_width=True, height=200)
        except Exception as e:
            st.warning(f"Could not preview file: {e}")
    columns = list(df.columns) if df is not None else []

if not columns:
    st.info("Provide data to continue.")

st.subheader("2. Column mapping")
c1, c2, c3 = st.columns(3)
with c1:
    id_col = st.selectbox("ID column", columns, index=columns.index("ID") if "ID" in columns else 0) if columns else st.text_input("ID column", "ID")
with c2:
    y_col = st.selectbox("Outcome (Y) column", columns, index=columns.index("FREQ") if "FREQ" in columns else 0) if columns else st.text_input("Outcome column", "FREQ")
with c3:
    offset_options = ["(none)"] + columns
    offset_default = offset_options.index("OFFSET") if "OFFSET" in offset_options else 0
    offset_col = st.selectbox("Offset column (optional, count models)", offset_options, index=offset_default)
    offset_col = None if offset_col == "(none)" else offset_col

group_options = ["(none)"] + columns
group_id_col = st.selectbox("Group/panel ID column (optional)", group_options)
group_id_col = None if group_id_col == "(none)" else group_id_col

st.subheader("3. Candidate variables")
default_vars = [c for c in columns if c not in {id_col, y_col, offset_col, group_id_col}][:6]
variables = st.multiselect(
    "Variables the search may include",
    [c for c in columns if c not in {id_col, y_col, offset_col, group_id_col}],
    default=default_vars,
)

st.subheader("4. Constraints")
c1, c2 = st.columns(2)
with c1:
    force_include = st.multiselect("Force include", variables)
    no_random = st.multiselect("Never random", variables)
with c2:
    no_zi = st.multiselect("Never zero-inflation term", variables)
    exclude = st.multiselect("Exclude from search", variables)
constraints_cfg = ConstraintsConfig(force_include=force_include, no_random=no_random, no_zi=no_zi, exclude=exclude)

st.subheader("5. Model family & search structure")
c1, c2, c3 = st.columns(3)
with c1:
    model_family = st.selectbox("Model family", ["count", "duration", "linear"])
    fit_family_options = {
        "count": ["nb", "poisson"],
        "duration": ["lognormal", "tobit"],
        "linear": ["gaussian"],
    }[model_family]
    fit_model_families = st.multiselect(
        "Refit best structure as (pick 2+ to compare, e.g. Poisson vs NB)",
        fit_family_options, default=fit_family_options,
        help="Each selected family is refit on the search's best structure and, if you pick "
             "more than one, compared with metacountregressor's compare_models() (BIC/AIC/loglik).",
    )
with c2:
    role_labels = {
        0: "0 Excluded", 1: "1 Fixed", 2: "2 Random (ind.)", 3: "3 Random (corr.)",
        4: "4 Grouped", 5: "5 Heterogeneity", 6: "6 Zero Inflation",
    }
    default_roles_sel = st.multiselect(
        "Roles the search may assign", list(role_labels.values()),
        default=[role_labels[k] for k in (0, 1, 2, 3, 5)],
    )
    default_roles = sorted(int(v.split()[0]) for v in default_roles_sel)
with c3:
    max_latent_classes = st.number_input("Max latent classes", 1, 4, 1)
    r_draws = st.number_input("Halton draws (R)", 25, 2000, 200, step=25)

st.subheader("6. Search algorithm")
c1, c2, c3 = st.columns(3)
with c1:
    algo = st.selectbox("Algorithm", ["sa", "de", "hs"], help="sa=Simulated Annealing, de=Differential Evolution, hs=Harmony Search")
with c2:
    max_iter = st.number_input("Max iterations", 50, 100000, 1000, step=50)
with c3:
    seed = st.number_input("Seed", 0, 999999, 42)
final_r_draws = st.number_input("Final refit Halton draws (R)", 50, 5000, 500, step=50)

st.subheader("7. Job naming & output")
c1, c2, c3 = st.columns(3)
with c1:
    experiment_name = st.text_input("Experiment name", value="metacount_run")
with c2:
    output_dir = st.text_input("Output directory", value="results")
with c3:
    search_description = st.text_input("Description", value="")

c1, c2, c3 = st.columns(3)
with c1:
    ncpus = st.number_input("HPC ncpus", 1, 64, 4)
with c2:
    mem_gb = st.number_input("HPC mem (GB)", 4, 512, 32)
with c3:
    walltime = st.text_input("HPC walltime", value="24:00:00")

st.divider()

ready = bool(variables and (df is not None or use_bundled_example))
if not ready:
    st.warning("Select data and at least one candidate variable to generate a script.")
else:
    hpc_data_filename = Path(data_path).name if data_path else "data.csv"

    cfg = MetaCountConfig(
        data_path=data_path, hpc_data_filename=hpc_data_filename,
        use_bundled_example=use_bundled_example,
        id_col=id_col, y_col=y_col, offset_col=offset_col, group_id_col=group_id_col,
        variables=variables, constraints=constraints_cfg,
        model_family=model_family, default_roles=default_roles or [0, 1],
        max_latent_classes=int(max_latent_classes), r_draws=int(r_draws),
        algo=algo, max_iter=int(max_iter), seed=int(seed),
        fit_model_families=fit_model_families or [fit_family_options[0]], final_r_draws=int(final_r_draws),
        output_dir=output_dir, experiment_name=experiment_name,
        search_description=search_description,
    )

    local_script = generate_metacount_script(cfg, for_hpc=False)
    hpc_script = generate_metacount_script(cfg, for_hpc=True)
    pbs_script = generate_pbs_script(PbsConfig(
        job_name=experiment_name, script_filename="run_metacount.py",
        ncpus=int(ncpus), mem_gb=int(mem_gb), walltime=walltime,
    ))

    render_run_and_export(
        key_prefix="metacount",
        local_script=local_script, local_script_name="run_metacount.py",
        hpc_script=hpc_script, hpc_script_name="run_metacount.py",
        pbs_script=pbs_script, pbs_script_name=f"{experiment_name}.pbs",
        engine_python=engine_python,
        data_file_to_bundle=uploaded_path,
    )
