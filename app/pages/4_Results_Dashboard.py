from pathlib import Path

import pandas as pd
import streamlit as st

from lib.results_parser import (
    discover_sa_runs,
    parse_progress_csv,
    parse_best_txt,
    read_results_txt,
    read_archive_txt,
    parse_solution_blocks,
    run_header,
    discover_metacount_json,
    parse_metacount_json,
)

st.set_page_config(page_title="Results Dashboard — ModelZoo", page_icon="📊", layout="wide")
st.title("📊 Results dashboard")
st.caption("Browse a completed SearchLibrium or metacountregressor run's output directory.")

tab_sl, tab_mc = st.tabs(["SearchLibrium (sa_runs)", "metacountregressor (JSON)"])

with tab_sl:
    st.caption(
        "Point at a SearchLibrium search's output directory — either the `sa_runs/` base folder "
        "(the newest run is picked automatically, or choose from the list) or a specific "
        "`sa_runs/sa_<id>_<timestamp>/` run directory directly."
    )
    sl_dir = st.text_input(
        "Output directory", value="", key="sl_dash_dir",
        placeholder=r"e.g. results\sa_runs or results\sa_runs\sa_1_20260101_120000",
    )

    if not sl_dir:
        st.info("Enter a directory to load a run.")
    else:
        runs = discover_sa_runs(sl_dir)
        if not runs:
            st.warning("No sa_runs-style output found at that path (expected progress.csv/best.txt/results.txt).")
        else:
            if len(runs) > 1:
                run_labels = [r.name for r in runs]
                chosen_label = st.selectbox("Run", run_labels, key="sl_dash_run_pick")
                run_dir = next(r for r in runs if r.name == chosen_label)
            else:
                run_dir = runs[0]
                st.caption(f"Run: `{run_dir}`")

            header = run_header(run_dir)
            if header:
                cols = st.columns(min(len(header), 5))
                for (k, v), col in zip(header.items(), cols * (len(header) // len(cols) + 1)):
                    if k == "_line":
                        continue
                    col.metric(k, v)

            progress_df = parse_progress_csv(run_dir)
            if progress_df is not None and not progress_df.empty:
                st.subheader("Convergence")
                c1, c2 = st.columns(2)
                with c1:
                    st.line_chart(progress_df.set_index("step")[["current_obj", "best_obj"]])
                with c2:
                    st.line_chart(progress_df.set_index("step")[["temperature"]])
                with st.expander("Raw progress.csv"):
                    st.dataframe(progress_df, use_container_width=True)
            else:
                st.info("No progress.csv found (or it was empty) in this run directory.")

            st.subheader("Best specification")
            best = parse_best_txt(run_dir)
            if best:
                st.dataframe(
                    pd.DataFrame({"field": list(best.keys()), "value": list(best.values())}),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.info("No best.txt found (multi-objective runs write archive.txt instead — see below).")

            archive_text = read_archive_txt(run_dir)
            if archive_text.strip():
                st.subheader("Pareto archive (multi-objective run)")
                archive_blocks = parse_solution_blocks(archive_text)
                if archive_blocks:
                    rows = []
                    for b in archive_blocks:
                        row = {"solution": b.title, **b.objectives}
                        rows.append(row)
                    archive_df = pd.DataFrame(rows)
                    st.dataframe(archive_df, use_container_width=True, hide_index=True)
                    obj_cols = [c for c in archive_df.columns if c != "solution"]
                    if len(obj_cols) == 2:
                        st.caption("Trade-off between the two objectives across non-dominated solutions:")
                        st.scatter_chart(archive_df, x=obj_cols[0], y=obj_cols[1])
                    with st.expander("Full specifications per archive solution"):
                        for b in archive_blocks:
                            st.markdown(f"**{b.title}** — {b.objectives}")
                            st.json(b.specification)
                else:
                    st.code(archive_text, language="text")

            with st.expander("Full results.txt (narrative log)"):
                results_text = read_results_txt(run_dir)
                st.code(results_text, language="text")
                blocks = parse_solution_blocks(results_text)
                initial = next((b for b in blocks if b.title == "Initial Solution"), None)
                final = next((b for b in blocks if b.title == "Final Solution"), None)
                if initial and final:
                    st.markdown("**Model Statistics — Final Solution**")
                    st.code(final.model_statistics, language="text")
                    improvement = {
                        crit: (initial.objectives.get(crit), final.objectives.get(crit))
                        for crit in final.objectives
                    }
                    st.markdown("**Initial → Final objective(s):**")
                    for crit, (start, end) in improvement.items():
                        if start is not None and end is not None:
                            st.metric(crit, f"{end:.4f}", delta=f"{end - start:+.4f}", delta_color="inverse")

with tab_mc:
    st.caption(
        "Point at a metacountregressor run's JSON summary (from `SearchOutputConfig(save_json=True)`) "
        "— either the exact `.json` file, or the `results/` directory to pick from all JSON files there."
    )
    mc_path = st.text_input(
        "JSON file or directory", value="", key="mc_dash_path",
        placeholder=r"e.g. results or results\myexperiment_count_sa_20260101_120000.json",
    )

    if not mc_path:
        st.info("Enter a path to load a run.")
    else:
        json_paths = discover_metacount_json(mc_path)
        if not json_paths:
            st.warning("No .json result files found at that path.")
        else:
            if len(json_paths) > 1:
                labels = [p.name for p in json_paths]
                chosen = st.selectbox("Result file", labels, key="mc_dash_pick")
                json_path = next(p for p in json_paths if p.name == chosen)
            else:
                json_path = json_paths[0]
                st.caption(f"File: `{json_path}`")

            data = parse_metacount_json(json_path)
            config = data.get("config", {})
            result = data.get("result", {})

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Experiment", config.get("experiment_name", "—"))
            c2.metric("Family", data.get("family", "—"))
            c3.metric("Algorithm", data.get("algorithm", result.get("algorithm", "—")))
            c4.metric("Best score", f"{result.get('best_score', float('nan')):.4f}" if "best_score" in result else "—")

            scores = result.get("scores")
            if scores:
                st.subheader("Scores explored")
                st.caption(
                    "Score achieved by each accepted/evaluated solution, in the order the search visited "
                    "them (not a strict per-iteration running-best trace)."
                )
                scores_df = pd.DataFrame({"solution_index": range(len(scores)), "score": scores})
                st.line_chart(scores_df.set_index("solution_index"))

            best_solution = result.get("best_solution")
            if best_solution is not None:
                st.subheader("Best decision vector")
                st.code(str(best_solution), language="text")

            with st.expander("Raw JSON"):
                st.json(data)
