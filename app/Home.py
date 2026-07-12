import streamlit as st

from lib.env import DEFAULT_ENGINE_PYTHON, probe_engine

st.set_page_config(page_title="ModelZoo", page_icon="🔬", layout="wide")

st.title("ModelZoo")
st.caption("GUI runner for SearchLibrium and metacountregressor searches, with one-click HPC job generation.")

st.session_state.setdefault("engine_python", DEFAULT_ENGINE_PYTHON)

with st.sidebar:
    st.header("Engine environment")
    st.caption(
        "The app itself only needs Streamlit + pandas. All model fitting/search "
        "runs in a separate Python interpreter that has SearchLibrium, "
        "metacountregressor, and JAX installed."
    )
    engine_python = st.text_input("Engine Python interpreter", key="engine_python")
    check = st.button("Check engine", use_container_width=True)

st.subheader("Engine status")

if "engine_status" not in st.session_state or check:
    with st.spinner("Probing engine interpreter (imports SearchLibrium + metacountregressor + JAX, can take up to ~60-90s)..."):
        st.session_state["engine_status"] = probe_engine(engine_python)

status = st.session_state["engine_status"]

if not status.exists:
    st.error(f"Interpreter not found at `{status.python_exe}`. Set the correct path in the sidebar.")
elif status.error:
    st.error(f"Could not probe interpreter: {status.error}")
else:
    st.success(f"Interpreter OK — {status.python_version.splitlines()[0]}")
    cols = st.columns(len(status.packages))
    for col, (pkg, info) in zip(cols, status.packages.items()):
        with col:
            if info.get("ok"):
                st.metric(pkg, info.get("version", "?"), delta="OK", delta_color="normal")
            else:
                st.metric(pkg, "missing", delta="FAIL", delta_color="inverse")
                with st.expander("error"):
                    st.code(info.get("error", ""))

st.divider()
st.markdown(
    """
### What's here

- **SearchLibrium** — configure and run discrete-choice model searches (MNL, Mixed Logit,
  Nested Logit, Random Regret, ...), watch a live console, and generate a matching HPC job.
- **MetaCountRegressor** — configure and run structure search + estimation for count,
  CMF, duration, and linear models, watch a live console, and generate a matching HPC job.
- **ABM Pipeline** — run the SEQ activity-based model pipeline (`Z:\\test_runs_tours\\code`):
  main estimation, GA/SA feature-selection strategies (estimator + budget), HHTS search
  presets, and the safety-skim experiment — with ready-made `qsub` commands for the cluster.

Use the sidebar to open a tool page. Every run can be launched locally (subprocess against the
engine interpreter above) or exported as a ready-to-submit PBS job for the HPC cluster.
"""
)
