"""Shared Streamlit UI: script preview, local run console, HPC bundle download."""
from __future__ import annotations

import io
import zipfile
from datetime import datetime
from pathlib import Path

import streamlit as st

from lib.runner import stream_run

JOBS_ROOT = Path(__file__).resolve().parent.parent.parent / "generated_jobs"


def _job_dir(prefix: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = JOBS_ROOT / f"{prefix}_{stamp}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def render_run_and_export(
    *,
    key_prefix: str,
    local_script: str,
    local_script_name: str,
    hpc_script: str,
    hpc_script_name: str,
    pbs_script: str,
    pbs_script_name: str,
    engine_python: str,
    data_file_to_bundle: Path | None = None,
) -> None:
    """Render preview tabs + 'run locally' console + HPC bundle download button."""

    tab_local, tab_hpc, tab_pbs = st.tabs(["Local script", "HPC script", "PBS job file"])
    with tab_local:
        st.code(local_script, language="python", line_numbers=True)
    with tab_hpc:
        st.code(hpc_script, language="python", line_numbers=True)
    with tab_pbs:
        st.code(pbs_script, language="bash", line_numbers=True)

    col_run, col_export = st.columns(2)

    with col_run:
        if st.button("Run locally now", key=f"{key_prefix}_run", type="primary", use_container_width=True):
            job_dir = _job_dir(key_prefix)
            script_path = job_dir / local_script_name
            script_path.write_text(local_script, encoding="utf-8")
            st.info(f"Job folder: `{job_dir}`")
            console = st.empty()
            lines: list[str] = []
            with st.spinner("Running..."):
                for line in stream_run(engine_python, str(script_path), cwd=str(job_dir)):
                    lines.append(line)
                    console.code("\n".join(lines[-400:]), language="text")
            st.success("Run finished — see console output above and the job folder for saved results.")

    with col_export:
        if st.button("Build HPC job bundle (.zip)", key=f"{key_prefix}_export", use_container_width=True):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(hpc_script_name, hpc_script)
                zf.writestr(pbs_script_name, pbs_script)
                readme = (
                    f"# {key_prefix} HPC job bundle\n\n"
                    f"1. Copy this folder to the HPC cluster (e.g. into your project working directory).\n"
                    f"2. Place your data CSV in the same directory"
                    + (f", named `{data_file_to_bundle.name}`.\n" if data_file_to_bundle else ".\n")
                    + f"3. Submit with: `qsub {pbs_script_name}`\n"
                    f"4. Monitor with: `qstat -u $USER`\n"
                )
                zf.writestr("README.md", readme)
                if data_file_to_bundle and data_file_to_bundle.exists():
                    zf.write(data_file_to_bundle, data_file_to_bundle.name)
            st.download_button(
                "Download bundle",
                data=buf.getvalue(),
                file_name=f"{key_prefix}_hpc_job.zip",
                mime="application/zip",
                key=f"{key_prefix}_download",
                use_container_width=True,
            )
