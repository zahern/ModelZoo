import streamlit as st

from lib.env import (
    DEFAULT_ENGINE_PYTHON,
    KNOWN_SOURCE_PATHS,
    UPDATABLE_PACKAGES,
    build_pip_editable_cmd,
    build_pip_upgrade_cmd,
    probe_engine,
)
from lib.runner import stream_cmd

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
st.subheader("Update engine packages")
st.caption("Runs pip inside the engine interpreter above. Re-run **Check engine** afterwards to confirm the new versions.")

c1, c2 = st.columns(2)
with c1:
    update_targets = st.multiselect("Packages to update", list(UPDATABLE_PACKAGES.keys()))
with c2:
    update_method = st.radio(
        "Update from",
        ["PyPI (upgrade to latest release)", "Local source (editable install)"],
        horizontal=True,
    )

source_paths: dict[str, str] = {}
if update_method.startswith("Local"):
    for pkg in update_targets:
        if pkg not in KNOWN_SOURCE_PATHS:
            st.caption(f"{pkg} has no local-source option — will upgrade from PyPI instead.")
            continue
        source_paths[pkg] = st.text_input(
            f"{pkg} source path", value=KNOWN_SOURCE_PATHS.get(pkg, ""), key=f"src_path_{pkg}"
        )

if st.button("Run update", disabled=not update_targets, type="primary"):
    console = st.empty()
    lines: list[str] = []

    def _log(line: str) -> None:
        lines.append(line)
        console.code("\n".join(lines[-500:]), language="text")

    for pkg in update_targets:
        if update_method.startswith("Local") and pkg in source_paths:
            path = source_paths[pkg].strip()
            if not path:
                _log(f"[skip] {pkg}: no source path given")
                continue
            argv = build_pip_editable_cmd(engine_python, path)
        else:
            argv = build_pip_upgrade_cmd(engine_python, UPDATABLE_PACKAGES[pkg])
        _log(f"\n$ {' '.join(argv)}")
        with st.spinner(f"Updating {pkg}..."):
            for line in stream_cmd(argv):
                _log(line)
    st.success("Update finished. Click **Check engine** above to refresh the version grid.")

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
