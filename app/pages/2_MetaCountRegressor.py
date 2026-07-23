from pathlib import Path

import pandas as pd
import streamlit as st

from lib.env import DEFAULT_ENGINE_PYTHON
from lib.pbs_gen import PbsConfig, generate_pbs_script
from lib.script_gen import (
    ConstraintsConfig,
    MetaCountConfig,
    METACOUNT_DISTRIBUTIONS,
    generate_metacount_script,
    CMFConfig,
    generate_cmf_script,
    PavementConfig,
    generate_pavement_script,
)
from lib.ui_common import render_exclusive_groups, render_run_and_export

st.set_page_config(page_title="MetaCountRegressor — ModelZoo", page_icon="📈", layout="wide")
st.title("📈 MetaCountRegressor runner")
st.caption("Structure search + estimation for count, CMF, duration, linear, and pavement-deterioration models.")

engine_python = st.session_state.get("engine_python", DEFAULT_ENGINE_PYTHON)

tab_count, tab_cmf, tab_pavement = st.tabs([
    "Count / Duration / Linear", "CMF (Crash Modification Function)", "Pavement deterioration",
])

with tab_count:
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
    st.caption("Maps directly onto metacountregressor's `ModelConstraints` — see `get_help('constraints')` in the package for the full reference.")
    c1, c2 = st.columns(2)
    with c1:
        force_include = st.multiselect("Force include (never excluded)", variables)
        force_fixed = st.multiselect("Force fixed (excluded or fixed only, never random)", variables)
        no_random = st.multiselect("Never random", variables)
        no_zi = st.multiselect("Never zero-inflation term", variables)
        exclude = st.multiselect("Exclude from search entirely", variables)
    with c2:
        membership_only = st.multiselect("Membership only (class-membership eq., no outcome effect)", variables)
        allow_membership = st.multiselect("Allow membership (may enter both membership + outcome)", variables)
        outcome_only = st.multiselect("Outcome only (never drives class membership)", variables)
        allow_random_vars = st.multiselect("Allow random with restricted distributions", variables)
        allow_random_distributions = st.multiselect(
            "...restricted to these distributions", METACOUNT_DISTRIBUTIONS,
            default=METACOUNT_DISTRIBUTIONS, disabled=not allow_random_vars,
        )

    st.markdown("**Mutually exclusive groups** — at most one variable per group may appear in the search (multicollinearity/redundancy guard).")
    mutual_exclusion_groups = render_exclusive_groups(
        "mc_mutex", variables,
        help_text="e.g. competing definitions of the same measure (SPEED vs SPEED_50).",
    )

    constraints_cfg = ConstraintsConfig(
        force_include=force_include, force_fixed=force_fixed, no_random=no_random,
        no_zi=no_zi, exclude=exclude, membership_only=membership_only,
        allow_membership=allow_membership, outcome_only=outcome_only,
        allow_random_vars=allow_random_vars,
        allow_random_distributions=allow_random_distributions or list(METACOUNT_DISTRIBUTIONS),
        mutual_exclusion_groups=mutual_exclusion_groups,
    )

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

    st.subheader("6b. Train/test split & stopping criteria")
    c1, c2 = st.columns(2)
    with c1:
        test_split_enabled = st.checkbox("Enable train/test split", key="mc_split_on")
        test_share = st.number_input(
            "Test share", 0.05, 0.5, 0.2, step=0.05, disabled=not test_split_enabled, key="mc_test_share",
        )
        st.caption(
            "metacountregressor has no out-of-sample scoring API (confirmed against source — no "
            "predict()/score() taking fixed coefficients + new data), so this splits the data and "
            "independently refits the *same* discovered structure on train vs. test, as a "
            "stability check — not true out-of-sample log-likelihood."
        )
    with c2:
        max_time_enabled = st.checkbox("Limit by wall-clock time", disabled=algo != "sa", key="mc_maxtime_on")
        max_time = st.number_input(
            "Max seconds", 10, 86400, 3600, step=60, disabled=not max_time_enabled or algo != "sa", key="mc_maxtime",
        ) if max_time_enabled else None
        patience_enabled = st.checkbox("Stop after N iterations with no improvement", disabled=algo != "sa", key="mc_patience_on")
        patience = st.number_input(
            "Patience (iterations)", 10, 100000, 400, step=10, disabled=not patience_enabled or algo != "sa", key="mc_patience",
        ) if patience_enabled else None
        if algo != "sa":
            st.caption("Stopping criteria only apply to algo='sa' — de/hs would crash on these extra kwargs (confirmed against source).")

    st.subheader("6c. Algorithm hyperparameters")
    algo_hyperparams: dict = {}
    if algo == "sa":
        st.caption("Forwarded to AdvancedSimulatedAnnealing — verified against Solvers_METAJAX.py.")
        c1, c2, c3 = st.columns(3)
        with c1:
            mc_t0_enabled = st.checkbox("Set initial temperature (T0)", key="mc_t0_on")
            mc_t0 = st.number_input("T0", 0.001, 1e7, 100.0, key="mc_t0", disabled=not mc_t0_enabled)
            mc_alpha = st.number_input("Cooling rate (alpha)", 0.5, 0.9999, 0.995, format="%.4f", key="mc_alpha")
        with c2:
            mc_n_starts = st.number_input("Parallel restarts (n_starts)", 1, 50, 1, key="mc_n_starts")
            mc_mutation_rate = st.number_input("Mutation rate", 0.01, 1.0, 0.3, key="mc_mut_rate")
        with c3:
            mc_min_changes = st.number_input("Min changes per move", 1, 20, 1, key="mc_min_changes")
            mc_max_changes = st.number_input("Max changes per move", 1, 20, 3, key="mc_max_changes")
        algo_hyperparams = {
            "alpha": float(mc_alpha), "n_starts": int(mc_n_starts),
            "mutation_rate": float(mc_mutation_rate),
            "min_changes": int(mc_min_changes), "max_changes": int(mc_max_changes),
        }
        if mc_t0_enabled:
            algo_hyperparams["T0"] = float(mc_t0)
    elif algo == "de":
        st.caption("Forwarded to AdaptiveDE — verified against Solvers_METAJAX.py.")
        c1, c2, c3 = st.columns(3)
        with c1:
            mc_pop_size = st.number_input("Population size", 4, 500, 20, key="mc_de_pop")
        with c2:
            mc_F = st.number_input("Mutation factor (F)", 0.0, 2.0, 0.5, key="mc_de_F")
        with c3:
            mc_CR = st.number_input("Crossover rate (CR)", 0.0, 1.0, 0.7, key="mc_de_CR")
        algo_hyperparams = {"population_size": int(mc_pop_size), "F": float(mc_F), "CR": float(mc_CR)}
    else:  # hs
        st.caption("Forwarded to DynamicHarmony — verified against Solvers_METAJAX.py.")
        c1, c2, c3 = st.columns(3)
        with c1:
            mc_hs_pop = st.number_input("Harmony memory size (population)", 4, 500, 20, key="mc_hs_pop")
            mc_hmcr = st.number_input("Harmony memory consideration rate (hmcr)", 0.0, 1.0, 0.9, key="mc_hmcr")
        with c2:
            mc_par_min = st.number_input("Min pitch adjustment rate", 0.0, 1.0, 0.1, key="mc_par_min")
            mc_par_max = st.number_input("Max pitch adjustment rate", 0.0, 1.0, 0.9, key="mc_par_max")
        with c3:
            mc_bw_min = st.number_input("Min bandwidth", 0.0, 100.0, 1.0, key="mc_bw_min")
            mc_bw_max = st.number_input("Max bandwidth", 0.0, 100.0, 3.0, key="mc_bw_max")
        algo_hyperparams = {
            "population_size": int(mc_hs_pop), "hmcr": float(mc_hmcr),
            "par_min": float(mc_par_min), "par_max": float(mc_par_max),
            "bw_min": float(mc_bw_min), "bw_max": float(mc_bw_max),
        }

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

    has_any_constraint = any([
        force_include, force_fixed, no_random, no_zi, exclude, membership_only,
        allow_membership, outcome_only, allow_random_vars, mutual_exclusion_groups,
    ])
    if has_any_constraint and model_family != "count":
        st.warning(
            f"model_family={model_family!r}: metacountregressor's `build_evaluator()` only merges "
            "`ModelConstraints` for the 'count' family (verified against source) — the constraints "
            "above, including mutually-exclusive groups, will likely be silently ignored for this "
            "search. Switch to 'count' if the constraints must be enforced."
        )

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
            test_split_enabled=bool(test_split_enabled), test_share=float(test_share),
            max_time=int(max_time) if max_time else None,
            patience=int(patience) if patience else None,
            algo_hyperparams=algo_hyperparams,
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

with tab_cmf:
    st.caption(
        "Crash Modification Function search: an AADT baseline/local-interaction structure "
        "(`CMFExperimentBuilder`), fit either via the original GA/JAX search with a CMF "
        "interpretation table, or via the JAX-flexible route that supports full "
        "`ModelConstraints` and latent classes."
    )

    st.subheader("1. Data")
    cmf_data_choice = st.radio(
        "Data source", ["Bundled example dataset", "Upload CSV", "Path on disk"], horizontal=True, key="cmf_data_choice",
    )
    cmf_df: pd.DataFrame | None = None
    cmf_data_path = ""
    cmf_uploaded_path: Path | None = None
    cmf_use_bundled = False
    if cmf_data_choice == "Bundled example dataset":
        cmf_use_bundled = True
        st.info(
            "Will load via `metacountregressor.load_example_crash_data()` at run time — crash-frequency "
            "data with an AADT column (34 columns, y_col='FREQ', aadt_col='AADT', id_col='ID')."
        )
        cmf_columns = [
            "ID", "FREQ", "LENGTH", "INCLANES", "DECLANES", "WIDTH", "MIMEDSH", "MXMEDSH", "SPEED", "URB",
            "FC", "AADT", "SINGLE", "DOUBLE", "TRAIN", "PEAKHR", "GRADEBR", "MIGRADE", "MXGRADE", "MXGRDIFF",
            "TANGENT", "CURVES", "MINRAD", "ACCESS", "MEDWIDTH", "FRICTION", "ADTLANE", "SLOPE", "INTECHAG",
            "AVEPRE", "AVESNOW", "OFFSET", "FC_ENCODED", "FC_LABEL",
        ]
    elif cmf_data_choice == "Upload CSV":
        cmf_up = st.file_uploader("CSV file", type=["csv"], key="cmf_upload")
        if cmf_up is not None:
            cmf_job_dir = Path(__file__).resolve().parent.parent.parent / "generated_jobs" / "_uploads"
            cmf_job_dir.mkdir(parents=True, exist_ok=True)
            cmf_uploaded_path = cmf_job_dir / cmf_up.name
            cmf_uploaded_path.write_bytes(cmf_up.getvalue())
            cmf_data_path = str(cmf_uploaded_path)
            cmf_df = pd.read_csv(cmf_uploaded_path)
            st.dataframe(cmf_df.head(20), use_container_width=True, height=200)
        cmf_columns = list(cmf_df.columns) if cmf_df is not None else []
    else:
        cmf_data_path = st.text_input("CSV path (must be readable by the engine interpreter)", value="", key="cmf_path")
        if cmf_data_path and Path(cmf_data_path).exists():
            try:
                cmf_df = pd.read_csv(cmf_data_path, nrows=5000)
                st.dataframe(cmf_df.head(20), use_container_width=True, height=200)
            except Exception as e:
                st.warning(f"Could not preview file: {e}")
        cmf_columns = list(cmf_df.columns) if cmf_df is not None else []

    if not cmf_columns:
        st.info("Provide data to continue.")

    st.subheader("2. Column mapping")
    c1, c2 = st.columns(2)
    with c1:
        cmf_y_default = cmf_columns.index("FREQ") if "FREQ" in cmf_columns else 0
        cmf_y_col = st.selectbox("Outcome (crash count) column", cmf_columns, index=cmf_y_default, key="cmf_y") if cmf_columns else st.text_input("Outcome column", "FREQ", key="cmf_y_text")
    with c2:
        cmf_aadt_default = cmf_columns.index("AADT") if "AADT" in cmf_columns else 0
        cmf_aadt_col = st.selectbox("AADT (traffic volume) column", cmf_columns, index=cmf_aadt_default, key="cmf_aadt") if cmf_columns else st.text_input("AADT column", "AADT", key="cmf_aadt_text")

    st.subheader("3. Baseline / local variable structure")
    st.caption(
        "Baseline vars enter the inherent-risk term directly; local vars enter interacted "
        "with log(AADT) (traffic-sensitivity terms) — this is the standard CMF hierarchical form."
    )
    remaining_cols = [c for c in cmf_columns if c not in {cmf_y_col, cmf_aadt_col}]
    c1, c2 = st.columns(2)
    with c1:
        cmf_baseline_vars = st.multiselect("Baseline variables", remaining_cols, key="cmf_baseline")
    with c2:
        cmf_local_vars = st.multiselect(
            "Local (AADT-interacted) variables",
            [c for c in remaining_cols if c not in cmf_baseline_vars], key="cmf_local",
        )

    st.subheader("4. Search mode")
    cmf_search_mode_label = st.radio(
        "Search mode",
        ["GA search (original CMF method + interpretation table)", "JAX flexible search (full constraints + latent classes)"],
        index=1, key="cmf_mode",
    )
    cmf_search_mode = "ga" if cmf_search_mode_label.startswith("GA") else "jax"

    cmf_constraints_cfg = ConstraintsConfig()
    cmf_variables: list[str] = []
    cmf_id_col = cmf_columns[0] if cmf_columns else "id"
    cmf_offset_col = None
    cmf_group_id_col = None
    cmf_default_roles = [0, 1, 2, 3, 5]
    cmf_max_latent_classes = 1
    cmf_r_draws = 200
    cmf_force_aadt_term = True
    cmf_algo = "sa"
    cmf_max_iter = 1000
    cmf_seed = 42
    cmf_fit_families = ["nb"]
    cmf_final_r_draws = 500
    cmf_ga_R = 200
    cmf_ga_final_R = 500
    cmf_test_split_enabled = False
    cmf_test_share = 0.2
    cmf_max_time = None
    cmf_patience = None
    cmf_algo_hyperparams: dict = {}

    if cmf_search_mode == "ga":
        c1, c2 = st.columns(2)
        with c1:
            cmf_ga_R = st.number_input("Search draws (R)", 25, 2000, 200, step=25, key="cmf_ga_r")
        with c2:
            cmf_ga_final_R = st.number_input("Final refit draws (R)", 50, 5000, 500, step=50, key="cmf_ga_final_r")
        st.info(
            "This mode uses the original AADT-specific GA/JAX search and prints a CMF "
            "interpretation table (CMF = exp(β·Δ)); it does not take `ModelConstraints` — "
            "use the baseline/local variable lists above to control which variables are considered."
        )
    else:
        st.markdown("**Constraints** (maps onto `ModelConstraints`, merged in via `build_jax_count_evaluator(constraints=...)`)")
        cmf_constraint_vars = cmf_baseline_vars + cmf_local_vars
        c1, c2 = st.columns(2)
        with c1:
            cmf_force_include = st.multiselect("Force include", cmf_constraint_vars, key="cmf_force_include")
            cmf_force_fixed = st.multiselect("Force fixed", cmf_constraint_vars, key="cmf_force_fixed")
            cmf_no_random = st.multiselect("Never random", cmf_constraint_vars, key="cmf_no_random")
            cmf_exclude = st.multiselect("Exclude entirely", cmf_constraint_vars, key="cmf_exclude")
        with c2:
            cmf_membership_only = st.multiselect("Membership only", cmf_constraint_vars, key="cmf_membership_only")
            cmf_allow_membership = st.multiselect("Allow membership", cmf_constraint_vars, key="cmf_allow_membership")
            cmf_outcome_only = st.multiselect("Outcome only", cmf_constraint_vars, key="cmf_outcome_only")
            cmf_allow_random_vars = st.multiselect("Allow random (restricted distributions)", cmf_constraint_vars, key="cmf_allow_random_vars")
        cmf_mutual_exclusion_groups = render_exclusive_groups(
            "cmf_mutex", cmf_constraint_vars,
            help_text="e.g. competing definitions of the same road/traffic measure.",
        )
        cmf_constraints_cfg = ConstraintsConfig(
            force_include=cmf_force_include, force_fixed=cmf_force_fixed, no_random=cmf_no_random,
            exclude=cmf_exclude, membership_only=cmf_membership_only,
            allow_membership=cmf_allow_membership, outcome_only=cmf_outcome_only,
            allow_random_vars=cmf_allow_random_vars,
            mutual_exclusion_groups=cmf_mutual_exclusion_groups,
        )

        st.markdown("**Search structure**")
        c1, c2, c3 = st.columns(3)
        with c1:
            cmf_id_default = cmf_columns.index("ID") if "ID" in cmf_columns else 0
            cmf_id_col = st.selectbox("ID column", cmf_columns, index=cmf_id_default, key="cmf_id_col") if cmf_columns else st.text_input("ID column", "id", key="cmf_id_col_text")
            offset_opts = ["(none)"] + cmf_columns
            cmf_offset_col = st.selectbox("Offset column (optional)", offset_opts, key="cmf_offset_col")
            cmf_offset_col = None if cmf_offset_col == "(none)" else cmf_offset_col
        with c2:
            group_opts = ["(none)"] + cmf_columns
            cmf_group_id_col = st.selectbox("Group/panel ID (optional)", group_opts, key="cmf_group_col")
            cmf_group_id_col = None if cmf_group_id_col == "(none)" else cmf_group_id_col
            cmf_variables = st.multiselect(
                "Extra auxiliary variables (beyond baseline/local)",
                [c for c in remaining_cols if c not in cmf_baseline_vars and c not in cmf_local_vars],
                key="cmf_aux_vars",
            )
        with c3:
            cmf_max_latent_classes = st.number_input("Max latent classes", 1, 4, 1, key="cmf_max_lc")
            cmf_r_draws = st.number_input("Halton draws (R)", 25, 2000, 200, step=25, key="cmf_r_draws")
        cmf_force_aadt_term = st.checkbox("Force AADT term to always be fixed/active", value=True, key="cmf_force_aadt")

        st.markdown("**Search algorithm**")
        c1, c2, c3 = st.columns(3)
        with c1:
            cmf_algo = st.selectbox("Algorithm", ["sa", "de", "hs"], key="cmf_algo")
        with c2:
            cmf_max_iter = st.number_input("Max iterations", 50, 100000, 1000, step=50, key="cmf_max_iter")
        with c3:
            cmf_seed = st.number_input("Seed", 0, 999999, 42, key="cmf_seed")
        cmf_fit_families = st.multiselect(
            "Refit best structure as (pick 2+ to compare)", ["nb", "poisson"], default=["nb"], key="cmf_fit_families",
        )
        cmf_final_r_draws = st.number_input("Final refit Halton draws (R)", 50, 5000, 500, step=50, key="cmf_final_r")

        st.markdown("**Train/test split & stopping criteria**")
        c1, c2 = st.columns(2)
        with c1:
            cmf_test_split_enabled = st.checkbox("Enable train/test split", key="cmf_split_on")
            cmf_test_share = st.number_input(
                "Test share", 0.05, 0.5, 0.2, step=0.05, disabled=not cmf_test_split_enabled, key="cmf_test_share",
            )
        with c2:
            cmf_max_time_enabled = st.checkbox("Limit by wall-clock time", disabled=cmf_algo != "sa", key="cmf_maxtime_on")
            cmf_max_time = st.number_input(
                "Max seconds", 10, 86400, 3600, step=60, disabled=not cmf_max_time_enabled or cmf_algo != "sa", key="cmf_maxtime",
            ) if cmf_max_time_enabled else None
            cmf_patience_enabled = st.checkbox("Stop after N iterations with no improvement", disabled=cmf_algo != "sa", key="cmf_patience_on")
            cmf_patience = st.number_input(
                "Patience (iterations)", 10, 100000, 400, step=10, disabled=not cmf_patience_enabled or cmf_algo != "sa", key="cmf_patience",
            ) if cmf_patience_enabled else None
        if cmf_algo != "sa":
            st.caption("Stopping criteria only apply to algo='sa' — de/hs would crash on these extra kwargs (confirmed against source).")

        st.markdown("**Algorithm hyperparameters**")
        if cmf_algo == "sa":
            st.caption("Forwarded to AdvancedSimulatedAnnealing — verified against Solvers_METAJAX.py.")
            c1, c2, c3 = st.columns(3)
            with c1:
                cmf_t0_enabled = st.checkbox("Set initial temperature (T0)", key="cmf_t0_on")
                cmf_t0 = st.number_input("T0", 0.001, 1e7, 100.0, key="cmf_t0", disabled=not cmf_t0_enabled)
                cmf_alpha = st.number_input("Cooling rate (alpha)", 0.5, 0.9999, 0.995, format="%.4f", key="cmf_alpha")
            with c2:
                cmf_n_starts = st.number_input("Parallel restarts (n_starts)", 1, 50, 1, key="cmf_n_starts")
                cmf_mutation_rate = st.number_input("Mutation rate", 0.01, 1.0, 0.3, key="cmf_mut_rate")
            with c3:
                cmf_min_changes = st.number_input("Min changes per move", 1, 20, 1, key="cmf_min_changes")
                cmf_max_changes = st.number_input("Max changes per move", 1, 20, 3, key="cmf_max_changes")
            cmf_algo_hyperparams = {
                "alpha": float(cmf_alpha), "n_starts": int(cmf_n_starts),
                "mutation_rate": float(cmf_mutation_rate),
                "min_changes": int(cmf_min_changes), "max_changes": int(cmf_max_changes),
            }
            if cmf_t0_enabled:
                cmf_algo_hyperparams["T0"] = float(cmf_t0)
        elif cmf_algo == "de":
            st.caption("Forwarded to AdaptiveDE — verified against Solvers_METAJAX.py.")
            c1, c2, c3 = st.columns(3)
            with c1:
                cmf_de_pop = st.number_input("Population size", 4, 500, 20, key="cmf_de_pop")
            with c2:
                cmf_de_F = st.number_input("Mutation factor (F)", 0.0, 2.0, 0.5, key="cmf_de_F")
            with c3:
                cmf_de_CR = st.number_input("Crossover rate (CR)", 0.0, 1.0, 0.7, key="cmf_de_CR")
            cmf_algo_hyperparams = {"population_size": int(cmf_de_pop), "F": float(cmf_de_F), "CR": float(cmf_de_CR)}
        else:  # hs
            st.caption("Forwarded to DynamicHarmony — verified against Solvers_METAJAX.py.")
            c1, c2, c3 = st.columns(3)
            with c1:
                cmf_hs_pop = st.number_input("Harmony memory size (population)", 4, 500, 20, key="cmf_hs_pop")
                cmf_hmcr = st.number_input("Harmony memory consideration rate (hmcr)", 0.0, 1.0, 0.9, key="cmf_hmcr")
            with c2:
                cmf_par_min = st.number_input("Min pitch adjustment rate", 0.0, 1.0, 0.1, key="cmf_par_min")
                cmf_par_max = st.number_input("Max pitch adjustment rate", 0.0, 1.0, 0.9, key="cmf_par_max")
            with c3:
                cmf_bw_min = st.number_input("Min bandwidth", 0.0, 100.0, 1.0, key="cmf_bw_min")
                cmf_bw_max = st.number_input("Max bandwidth", 0.0, 100.0, 3.0, key="cmf_bw_max")
            cmf_algo_hyperparams = {
                "population_size": int(cmf_hs_pop), "hmcr": float(cmf_hmcr),
                "par_min": float(cmf_par_min), "par_max": float(cmf_par_max),
                "bw_min": float(cmf_bw_min), "bw_max": float(cmf_bw_max),
            }

    st.subheader("5. Job naming & output")
    c1, c2, c3 = st.columns(3)
    with c1:
        cmf_experiment_name = st.text_input("Experiment name", value="cmf_run", key="cmf_exp_name")
    with c2:
        cmf_output_dir = st.text_input("Output directory", value="results", key="cmf_output_dir")
    with c3:
        cmf_search_description = st.text_input("Description", value="", key="cmf_desc")

    c1, c2, c3 = st.columns(3)
    with c1:
        cmf_ncpus = st.number_input("HPC ncpus", 1, 64, 4, key="cmf_ncpus")
    with c2:
        cmf_mem_gb = st.number_input("HPC mem (GB)", 4, 512, 32, key="cmf_mem")
    with c3:
        cmf_walltime = st.text_input("HPC walltime", value="24:00:00", key="cmf_walltime")

    st.divider()

    cmf_has_data = cmf_use_bundled or cmf_df is not None
    cmf_ready = bool(cmf_y_col and cmf_aadt_col and (cmf_baseline_vars or cmf_local_vars) and cmf_has_data)
    if not cmf_ready:
        st.warning("Select data, the AADT/outcome columns, and at least one baseline or local variable to generate a script.")
    else:
        cmf_hpc_data_filename = Path(cmf_data_path).name if cmf_data_path else "data.csv"
        cmf_cfg = CMFConfig(
            data_path=cmf_data_path, hpc_data_filename=cmf_hpc_data_filename,
            y_col=cmf_y_col, aadt_col=cmf_aadt_col,
            baseline_vars=cmf_baseline_vars, local_vars=cmf_local_vars,
            search_mode=cmf_search_mode, use_bundled=cmf_use_bundled,
            ga_R=int(cmf_ga_R), ga_final_R=int(cmf_ga_final_R),
            id_col=cmf_id_col, offset_col=cmf_offset_col, group_id_col=cmf_group_id_col,
            variables=cmf_variables, constraints=cmf_constraints_cfg,
            default_roles=cmf_default_roles, max_latent_classes=int(cmf_max_latent_classes),
            r_draws=int(cmf_r_draws), force_aadt_term=bool(cmf_force_aadt_term),
            algo=cmf_algo, max_iter=int(cmf_max_iter), seed=int(cmf_seed),
            fit_model_families=cmf_fit_families or ["nb"], final_r_draws=int(cmf_final_r_draws),
            output_dir=cmf_output_dir, experiment_name=cmf_experiment_name,
            search_description=cmf_search_description,
            test_split_enabled=bool(cmf_test_split_enabled), test_share=float(cmf_test_share),
            max_time=int(cmf_max_time) if cmf_max_time else None,
            patience=int(cmf_patience) if cmf_patience else None,
            algo_hyperparams=cmf_algo_hyperparams,
        )

        cmf_local_script = generate_cmf_script(cmf_cfg, for_hpc=False)
        cmf_hpc_script = generate_cmf_script(cmf_cfg, for_hpc=True)
        cmf_pbs_script = generate_pbs_script(PbsConfig(
            job_name=cmf_experiment_name, script_filename="run_cmf.py",
            ncpus=int(cmf_ncpus), mem_gb=int(cmf_mem_gb), walltime=cmf_walltime,
        ))

        render_run_and_export(
            key_prefix="cmf",
            local_script=cmf_local_script, local_script_name="run_cmf.py",
            hpc_script=cmf_hpc_script, hpc_script_name="run_cmf.py",
            pbs_script=cmf_pbs_script, pbs_script_name=f"{cmf_experiment_name}.pbs",
            engine_python=engine_python,
            data_file_to_bundle=cmf_uploaded_path,
        )

with tab_pavement:
    st.caption(
        "Clusterwise log-log (power-law) regression of pavement serviceability (PSI) on "
        "age/traffic/condition variables, via a joint cluster+variable simulated-annealing "
        "search (`PavementCLROptimizer`), with an optional comparison of alternative temporal "
        "error structures (`PavementTemporalComparison`: OLS, AR(1), Random Walk w/ drift, Near-Unit-Root)."
    )

    st.subheader("1. Data")
    pav_data_choice = st.radio("Data source", ["Upload CSV", "Path on disk"], horizontal=True, key="pav_data_choice")
    pav_df: pd.DataFrame | None = None
    pav_data_path = ""
    pav_uploaded_path: Path | None = None
    if pav_data_choice == "Upload CSV":
        pav_up = st.file_uploader("CSV file", type=["csv"], key="pav_upload")
        if pav_up is not None:
            pav_job_dir = Path(__file__).resolve().parent.parent.parent / "generated_jobs" / "_uploads"
            pav_job_dir.mkdir(parents=True, exist_ok=True)
            pav_uploaded_path = pav_job_dir / pav_up.name
            pav_uploaded_path.write_bytes(pav_up.getvalue())
            pav_data_path = str(pav_uploaded_path)
            pav_df = pd.read_csv(pav_uploaded_path)
            st.dataframe(pav_df.head(20), use_container_width=True, height=200)
        pav_columns = list(pav_df.columns) if pav_df is not None else []
    else:
        pav_data_path = st.text_input("CSV path (must be readable by the engine interpreter)", value="", key="pav_path")
        if pav_data_path and Path(pav_data_path).exists():
            try:
                pav_df = pd.read_csv(pav_data_path, nrows=5000)
                st.dataframe(pav_df.head(20), use_container_width=True, height=200)
            except Exception as e:
                st.warning(f"Could not preview file: {e}")
        pav_columns = list(pav_df.columns) if pav_df is not None else []

    if not pav_columns:
        st.info("Provide data to continue (e.g. a panel of segment_id/age/aadt/rut_depth/... /psi rows).")

    st.subheader("2. Column mapping")
    c1, c2 = st.columns(2)
    with c1:
        pav_psi_col = st.selectbox(
            "PSI (serviceability) column", pav_columns,
            index=pav_columns.index("psi") if "psi" in pav_columns else 0,
            key="pav_psi",
        ) if pav_columns else st.text_input("PSI column", "psi", key="pav_psi_text")
    with c2:
        pav_segment_col = st.selectbox(
            "Segment/panel ID column", pav_columns,
            index=pav_columns.index("sample_id") if "sample_id" in pav_columns else 0,
            key="pav_segment",
        ) if pav_columns else st.text_input("Segment column", "sample_id", key="pav_segment_text")

    st.subheader("3. Predictor variables")
    pav_remaining = [c for c in pav_columns if c not in {pav_psi_col, pav_segment_col}]
    pav_variable_names = st.multiselect(
        "Variables the clusterwise search may include", pav_remaining,
        default=[c for c in pav_remaining if c.lower() in ("age", "aadt", "rut_depth")] or pav_remaining[:4],
        key="pav_vars",
    )
    pav_categorical_vars = st.multiselect(
        "...of which, treat as categorical (e.g. functional class)",
        pav_variable_names,
        default=[c for c in pav_variable_names if c.lower() in ("f_class", "category", "sys_id")],
        key="pav_cat_vars",
    )
    pav_continuous_cols = [c for c in pav_variable_names if c not in pav_categorical_vars]
    st.caption(f"Log-transformed continuous predictors: {pav_continuous_cols or '(none)'}")

    st.subheader("4. Clustering")
    pav_cluster_mode_label = st.radio(
        "Cluster count", ["Fixed K", "Search over a K range (BIC-selected)"], key="pav_cluster_mode",
    )
    pav_cluster_mode = "fixed" if pav_cluster_mode_label == "Fixed K" else "search_k"
    c1, c2, c3 = st.columns(3)
    if pav_cluster_mode == "fixed":
        with c1:
            pav_n_clusters = st.number_input("Number of clusters", 1, 20, 2, key="pav_n_clusters")
        pav_k_min, pav_k_max = 2, 6
    else:
        with c1:
            pav_k_min = st.number_input("K min", 2, 20, 2, key="pav_k_min")
        with c2:
            pav_k_max = st.number_input("K max", 2, 20, 6, key="pav_k_max")
        pav_n_clusters = 2

    with st.expander("Advanced search parameters (simulated annealing + model checks)"):
        c1, c2, c3 = st.columns(3)
        with c1:
            pav_min_observations = st.number_input("Min observations per cluster", 10, 10000, 300, key="pav_min_obs")
            pav_level_of_significance = st.number_input("Significance level", 0.001, 0.5, 0.05, step=0.005, format="%.3f", key="pav_alpha")
        with c2:
            pav_max_vif = st.number_input("Max VIF", 1.0, 100.0, 10.0, key="pav_max_vif")
            pav_temp_init = st.number_input("SA initial temperature", 0.1, 1000.0, 10.0, key="pav_temp_init")
        with c3:
            pav_cooling_rate = st.number_input("SA cooling rate", 0.5, 0.999, 0.97, step=0.005, format="%.3f", key="pav_cooling")
            pav_boltzmann = st.number_input("Boltzmann constant", 1.0, 1000.0, 80.0, key="pav_boltzmann")
        c1, c2, c3 = st.columns(3)
        with c1:
            pav_n_changes = st.number_input("Neighbor moves per iteration", 1, 1000, 80, key="pav_n_changes")
        with c2:
            pav_n_neighbors = st.number_input("Neighbors evaluated per move", 1, 20, 3, key="pav_n_neighbors")
        with c3:
            pav_max_iterations = st.number_input("Max SA iterations", 10, 100000, 1000, step=10, key="pav_max_iter")
        pav_seed = st.number_input("Seed", 0, 999999, 42, key="pav_seed")

    pav_run_temporal_comparison = st.checkbox(
        "Also compare temporal error structures per cluster (OLS/AR(1)/Random Walk/Near-Unit-Root)",
        value=True, key="pav_temporal",
    )

    st.subheader("5. Job naming & output")
    c1, c2, c3 = st.columns(3)
    with c1:
        pav_experiment_name = st.text_input("Experiment name", value="pavement_run", key="pav_exp_name")
    with c2:
        pav_output_dir = st.text_input("Output directory", value="results", key="pav_output_dir")
    with c3:
        pass

    c1, c2, c3 = st.columns(3)
    with c1:
        pav_ncpus = st.number_input("HPC ncpus", 1, 64, 4, key="pav_ncpus")
    with c2:
        pav_mem_gb = st.number_input("HPC mem (GB)", 4, 512, 32, key="pav_mem")
    with c3:
        pav_walltime = st.text_input("HPC walltime", value="24:00:00", key="pav_walltime")

    st.divider()

    pav_ready = bool(pav_variable_names and pav_psi_col and pav_segment_col and pav_df is not None)
    if not pav_ready:
        st.warning("Select data and at least one predictor variable to generate a script.")
    else:
        pav_hpc_data_filename = Path(pav_data_path).name if pav_data_path else "data.csv"
        pav_cfg = PavementConfig(
            data_path=pav_data_path, hpc_data_filename=pav_hpc_data_filename,
            psi_col=pav_psi_col, segment_col=pav_segment_col,
            variable_names=pav_variable_names, categorical_vars=pav_categorical_vars,
            continuous_cols=pav_continuous_cols,
            min_observations=int(pav_min_observations), level_of_significance=float(pav_level_of_significance),
            max_vif=float(pav_max_vif), temp_init=float(pav_temp_init), cooling_rate=float(pav_cooling_rate),
            boltzmann=float(pav_boltzmann), n_changes=int(pav_n_changes), n_neighbors=int(pav_n_neighbors),
            cluster_mode=pav_cluster_mode, n_clusters=int(pav_n_clusters),
            k_min=int(pav_k_min), k_max=int(pav_k_max),
            seed=int(pav_seed), max_iterations=int(pav_max_iterations),
            run_temporal_comparison=bool(pav_run_temporal_comparison),
            output_dir=pav_output_dir, experiment_name=pav_experiment_name,
        )

        pav_local_script = generate_pavement_script(pav_cfg, for_hpc=False)
        pav_hpc_script = generate_pavement_script(pav_cfg, for_hpc=True)
        pav_pbs_script = generate_pbs_script(PbsConfig(
            job_name=pav_experiment_name, script_filename="run_pavement.py",
            ncpus=int(pav_ncpus), mem_gb=int(pav_mem_gb), walltime=pav_walltime,
        ))

        render_run_and_export(
            key_prefix="pavement",
            local_script=pav_local_script, local_script_name="run_pavement.py",
            hpc_script=pav_hpc_script, hpc_script_name="run_pavement.py",
            pbs_script=pav_pbs_script, pbs_script_name=f"{pav_experiment_name}.pbs",
            engine_python=engine_python,
            data_file_to_bundle=pav_uploaded_path,
        )
