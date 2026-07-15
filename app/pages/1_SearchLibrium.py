from pathlib import Path

import pandas as pd
import streamlit as st

from lib.env import DEFAULT_ENGINE_PYTHON
from lib.pbs_gen import PbsConfig, generate_pbs_script
from lib.script_gen import (
    SearchLibriumConfig,
    SearchLibriumConstraintsConfig,
    SEARCHLIBRIUM_BUNDLED_PRESETS,
    SEARCHLIBRIUM_DISTRIBUTIONS,
    generate_searchlibrium_script,
    generate_searchlibrium_ctrl_preview,
    StandaloneFitConfig,
    STANDALONE_MODEL_FAMILIES,
    generate_standalone_fit_script,
    MDCEVConfig,
    generate_mdcev_script,
    DestinationPredictionConfig,
    DESTINATION_MODEL_FAMILIES,
    generate_destination_prediction_script,
)
from lib.ui_common import render_exclusive_groups, render_pool_rules, render_run_and_export
from lib.runner import stream_run

st.set_page_config(page_title="SearchLibrium — ModelZoo", page_icon="🔎", layout="wide")
st.title("🔎 SearchLibrium runner")
st.caption("Discrete-choice model structure search: Multinomial/Mixed/Nested Logit, Random Regret, and more.")

engine_python = st.session_state.get("engine_python", DEFAULT_ENGINE_PYTHON)

BUNDLED_LABELS = {
    "swiss_metro": "swiss_metro — SP study, car/train/SM, 3 alts (README quick start)",
    "electricity": "electricity — stated-preference electricity plan choice, 4 alts",
    "travel_mode": "travel_mode — air/train/bus/car mode choice, 4 alts",
}

tab_search, tab_standalone, tab_mdcev, tab_destination = st.tabs(
    ["Structure search", "Standalone fit", "MDCEV budget allocation", "Destination/flow prediction"]
)

with tab_search:
    st.subheader("1. Data")
    st.caption("Long-format choice data — one row per alternative per observation.")
    data_choice = st.radio("Data source", ["Bundled example dataset", "Upload CSV", "Path on disk"], horizontal=True)

    df: pd.DataFrame | None = None
    data_path = ""
    uploaded_path: Path | None = None
    use_bundled: str | None = None
    preset: dict | None = None

    if data_choice == "Bundled example dataset":
        bundled_label = st.selectbox("Dataset", list(BUNDLED_LABELS.values()))
        use_bundled = next(k for k, v in BUNDLED_LABELS.items() if v == bundled_label)
        preset = SEARCHLIBRIUM_BUNDLED_PRESETS[use_bundled]
        st.info(
            f"Shipped inside SearchLibrium (`{preset['loader']}()`) — no file to upload, and none "
            f"needs to travel with an HPC job. Columns: {', '.join(preset['columns'])}"
        )
        columns = preset["columns"]
    elif data_choice == "Upload CSV":
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
    elif use_bundled is None:
        columns = []
        st.info("Provide data to continue.")

    st.subheader("2. Column mapping")
    if preset is not None:
        st.caption("Fixed by the bundled dataset's known schema (see the loader's docstring for details).")
        choice_col, alt_col, choice_id_col = preset["choice_col"], preset["alt_col"], preset["choice_id_col"]
        ind_id_col, base_alt = preset["ind_id_col"], preset["base_alt"]
        st.code(
            f"choice_col={choice_col!r}  alt_col={alt_col!r}  choice_id_col={choice_id_col!r}  "
            f"ind_id_col={ind_id_col!r}  base_alt={base_alt!r}",
            language="text",
        )
    else:
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
    default_asvars = preset["default_vars"] if preset is not None else []
    asvarnames = st.multiselect(
        "Alternative-specific variables (varnames)",
        [c for c in columns if c not in {choice_col, alt_col, choice_id_col}],
        default=default_asvars,
    )
    isvarnames = st.multiselect("Individual-specific variables (optional)", [c for c in columns if c not in {choice_col, alt_col, choice_id_col, *asvarnames}])

    all_candidate_vars = asvarnames + isvarnames

    st.subheader("4. Constraints")
    st.caption("Maps onto SearchLibrium's `ConstraintBuilder` — see the README's Constraints section for the full reference.")
    c1, c2 = st.columns(2)
    with c1:
        sl_force_include = st.multiselect("Force include (must always appear)", all_candidate_vars)
        sl_never_include = st.multiselect("Never include (must never appear)", all_candidate_vars)
    with c2:
        sl_force_random_vars = st.multiselect("Force random (must have a random parameter)", all_candidate_vars)
        sl_force_random_dist = st.selectbox(
            "...with distribution", list(SEARCHLIBRIUM_DISTRIBUTIONS.keys()),
            format_func=lambda k: f"{k} — {SEARCHLIBRIUM_DISTRIBUTIONS[k]}",
            disabled=not sl_force_random_vars,
        )
        sl_exclude_random = st.multiselect("Exclude random (must never be random)", all_candidate_vars)

    st.markdown("**Mutually exclusive groups** — at most one variable per group may appear in any solution.")
    sl_mutex_groups = render_exclusive_groups(
        "sl_mutex", all_candidate_vars,
        help_text="e.g. alternative speed definitions — never both in the model.",
    )

    st.markdown("**Minimum behavioural content** — require at least N variables from a pool, without locking in which ones.")
    sl_pool_rules = render_pool_rules(
        "sl_pool", all_candidate_vars,
        help_text="e.g. at least 2 of {PRICE, BIKELANE, DIST6, RECRE} must be present.",
    )

    constraints_cfg = SearchLibriumConstraintsConfig(
        force_include=sl_force_include, never_include=sl_never_include,
        mutually_exclusive_groups=sl_mutex_groups, min_behavioral_rules=sl_pool_rules,
        force_random_vars=sl_force_random_vars, force_random_distribution=sl_force_random_dist,
        exclude_random=sl_exclude_random,
    )

    st.subheader("5. Model & search settings")
    c1, c2, c3 = st.columns(3)
    with c1:
        models = st.multiselect(
            "Model types to search over",
            ["multinomial", "mixed_logit", "random_regret", "mixed_random_regret",
             "nested_logit", "mixed_nested", "ordered_logit"],
            default=["multinomial"],
            help="mixed_nested needs allow_random + nests/lambdas below, same as nested_logit.",
        )
        criterion = st.selectbox("Objective (minimise)", ["bic", "aic", "loglik"])
        mae_enabled = st.checkbox(
            "Add MAE as a second objective (multi-objective)",
            help="Auto-splits the data into train/test by individual (val_share below) unless "
                 "you're already passing your own held-out set — see SearchLibrium's Parameters(df_test=...).",
        )
        val_share = st.number_input("Held-out share for MAE", 0.05, 0.5, 0.25, step=0.05, disabled=not mae_enabled)
    with c2:
        allow_random = st.checkbox("Allow random parameters", value="mixed_logit" in models or "mixed_random_regret" in models or "mixed_nested" in models)
        allow_bcvars = st.checkbox("Allow Box-Cox transforms")
        allow_corvars = st.checkbox("Allow correlated random parameters")
        latent_class = st.checkbox("Latent class model", help="Every solution in the search is fit as latent-class, regardless of the model types picked above.")
        num_classes = st.number_input("Number of classes", 2, 6, 2, disabled=not latent_class)
    with c3:
        algorithm = st.selectbox(
            "Search algorithm",
            ["sa", "hs", "sapbil", "banditsa", "hspbil", "parsa", "parcopsa"],
            help="parsa/parcopsa run nthrds independent SA solvers in parallel threads; "
                 "parcopsa additionally shares the best solution across solvers periodically.",
        )
        nthrds = st.number_input("Parallel threads (nthrds)", 2, 16, 4, disabled=algorithm not in ("parsa", "parcopsa"))
        p_val = st.number_input("Significance threshold (p_val)", 0.001, 0.5, 0.05, step=0.01)
        seed = st.number_input("Run ID / seed", 1, 999999, 1)

    c1, c2 = st.columns(2)
    with c1:
        n_draws = st.number_input("Halton draws (n_draws, for mixed models)", 50, 5000, 500, step=50)
    with c2:
        maxiter = st.number_input("Max MLE iterations per fit (maxiter)", 50, 10000, 2000, step=50)

    nests_json = lambdas_json = None
    if "nested_logit" in models or "mixed_nested" in models:
        st.caption("Nested/mixed-nested logit require nest structure, e.g. `{\"PublicTransport\": [0, 1], \"Private\": [2, 3]}`")
        nests_json = st.text_area("nests (JSON)", value='{"nest_1": [0, 1]}')
        lambdas_json = st.text_area("lambdas (JSON)", value='{"nest_1": 0.8}')

    st.subheader("6. Job naming & output")
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

    has_data = use_bundled is not None or df is not None
    ready = bool(models and (asvarnames or isvarnames) and has_data)
    if not ready:
        st.warning("Select data, at least one model type, and at least one variable to generate a script.")
    else:
        hpc_data_filename = Path(data_path).name if data_path else ""

        cfg = SearchLibriumConfig(
            data_path=data_path,
            hpc_data_filename=hpc_data_filename,
            choice_col=choice_col, alt_col=alt_col, choice_id_col=choice_id_col,
            ind_id_col=ind_id_col, base_alt=base_alt,
            asvarnames=asvarnames, isvarnames=isvarnames,
            models=models, allow_random=allow_random, allow_bcvars=allow_bcvars,
            allow_corvars=allow_corvars, p_val=p_val, n_draws=int(n_draws), maxiter=int(maxiter),
            criterion=criterion, mae_enabled=mae_enabled, val_share=val_share,
            algorithm=algorithm, nthrds=int(nthrds), seed=int(seed),
            nests_json=nests_json, lambdas_json=lambdas_json,
            latent_class=latent_class, num_classes=int(num_classes),
            output_dir=output_dir, experiment_name=experiment_name,
            use_bundled=use_bundled,
            constraints=constraints_cfg,
        )

        with st.expander("Preview auto-estimated hyperparameters (before running)"):
            st.caption("Runs `estimate_ctrl()` against your data/spec only — no search, seconds not minutes.")
            if st.button("Preview hyperparameters", key="sl_ctrl_preview"):
                preview_script = generate_searchlibrium_ctrl_preview(cfg)
                job_dir = Path(__file__).resolve().parent.parent.parent / "generated_jobs" / "_ctrl_preview"
                job_dir.mkdir(parents=True, exist_ok=True)
                script_path = job_dir / "ctrl_preview.py"
                script_path.write_text(preview_script, encoding="utf-8")
                lines: list[str] = []
                console = st.empty()
                with st.spinner("Estimating..."):
                    for line in stream_run(engine_python, str(script_path), cwd=str(job_dir)):
                        lines.append(line)
                        console.code("\n".join(lines[-100:]), language="text")

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

with tab_standalone:
    st.caption(
        "Fit a single, pre-specified model directly — no metaheuristic search over structures. "
        "Useful for exploring/reporting on one specification you already have in mind."
    )
    st.info(
        "9 of the 12 documented standalone model classes are supported here, each verified end-to-end "
        "against the real engine this session (which surfaced and fixed 8 genuine upstream bugs in "
        "SearchLibrium along the way). **MixedRandomRegret**, **MixedNested**, and **MultiLayerNestedLogit** "
        "are omitted — each hits a deeper upstream bug (missing attribute initialization in their "
        "constructor chains) that needs more substantial source surgery than a quick fix. "
        "**OrderedLogitLong** is also omitted: it additionally requires long/expanded panel data "
        "(one row per individual per ordinal category via `misc.wide_to_long`), which doesn't fit this "
        "simple column-mapping UI — use plain `OrderedLogit` for standard one-row-per-observation data."
    )

    model_class = st.selectbox("Model class", list(STANDALONE_MODEL_FAMILIES.keys()), key="sf_model_class")
    family = STANDALONE_MODEL_FAMILIES[model_class]

    st.subheader("1. Data")
    sf_data_choice = st.radio(
        "Data source", ["Bundled example dataset", "Upload CSV", "Path on disk"], horizontal=True, key="sf_data_choice",
    )
    sf_df: pd.DataFrame | None = None
    sf_data_path = ""
    sf_uploaded_path: Path | None = None
    sf_use_bundled: str | None = None
    sf_preset: dict | None = None

    if sf_data_choice == "Bundled example dataset":
        sf_bundled_label = st.selectbox("Dataset", list(BUNDLED_LABELS.values()), key="sf_bundled")
        sf_use_bundled = next(k for k, v in BUNDLED_LABELS.items() if v == sf_bundled_label)
        sf_preset = SEARCHLIBRIUM_BUNDLED_PRESETS[sf_use_bundled]
        st.info(f"Shipped inside SearchLibrium (`{sf_preset['loader']}()`). Columns: {', '.join(sf_preset['columns'])}")
        sf_columns = sf_preset["columns"]
    elif sf_data_choice == "Upload CSV":
        sf_up = st.file_uploader("CSV file", type=["csv"], key="sf_upload")
        if sf_up is not None:
            sf_job_dir = Path(__file__).resolve().parent.parent.parent / "generated_jobs" / "_uploads"
            sf_job_dir.mkdir(parents=True, exist_ok=True)
            sf_uploaded_path = sf_job_dir / sf_up.name
            sf_uploaded_path.write_bytes(sf_up.getvalue())
            sf_data_path = str(sf_uploaded_path)
            sf_df = pd.read_csv(sf_uploaded_path)
            st.dataframe(sf_df.head(20), use_container_width=True, height=200)
        sf_columns = list(sf_df.columns) if sf_df is not None else []
    else:
        sf_data_path = st.text_input("CSV path (must be readable by the engine interpreter)", value="", key="sf_path")
        if sf_data_path and Path(sf_data_path).exists():
            try:
                sf_df = pd.read_csv(sf_data_path, nrows=5000)
                st.dataframe(sf_df.head(20), use_container_width=True, height=200)
            except Exception as e:
                st.warning(f"Could not preview file: {e}")
        sf_columns = list(sf_df.columns) if sf_df is not None else []

    if not sf_columns:
        st.info("Provide data to continue.")

    # Defaults shared across families
    sf_choice_col = sf_alt_col = sf_choice_id_col = ""
    sf_ind_id_col = sf_avail_col = None
    sf_base_alt = ""
    sf_asvarnames: list[str] = []
    sf_isvarnames: list[str] = []
    sf_randvars: dict[str, str] = {}
    sf_correlated_vars: list[str] = []
    sf_n_draws = 1000
    sf_nests_json = sf_lambdas_json = None
    sf_ordinal_y_col = ""
    sf_n_categories = 3
    sf_ordered_distr = "logit"
    sf_normalize = False
    sf_binary_y_col = ""
    sf_selection_y_col = ""
    sf_selection_varnames: list[str] = []
    sf_outcome_y_col = ""
    sf_outcome_varnames: list[str] = []
    sf_n_classes = 2
    sf_membership_vars: list[str] = []
    sf_fit_intercept = False
    sf_maxiter = 2000

    is_choice_family = family in ("mnl_family", "mixed_logit", "nested", "rrm", "latent_class_mxl")

    if is_choice_family:
        st.subheader("2. Column mapping")
        if sf_preset is not None:
            sf_choice_col, sf_alt_col, sf_choice_id_col = sf_preset["choice_col"], sf_preset["alt_col"], sf_preset["choice_id_col"]
            sf_ind_id_col, sf_base_alt = sf_preset["ind_id_col"], sf_preset["base_alt"]
            st.code(
                f"choice_col={sf_choice_col!r}  alt_col={sf_alt_col!r}  choice_id_col={sf_choice_id_col!r}  "
                f"ind_id_col={sf_ind_id_col!r}  base_alt={sf_base_alt!r}",
                language="text",
            )
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                sf_choice_col = st.selectbox("Choice column", sf_columns, key="sf_choice_col") if sf_columns else st.text_input("Choice column", "CHOICE", key="sf_choice_col_t")
                sf_alt_col = st.selectbox("Alternative column", sf_columns, key="sf_alt_col") if sf_columns else st.text_input("Alternative column", "alt", key="sf_alt_col_t")
            with c2:
                sf_choice_id_col = st.selectbox("Choice-set ID column", sf_columns, key="sf_choice_id") if sf_columns else st.text_input("Choice ID column", "custom_id", key="sf_choice_id_t")
                if family in ("mixed_logit", "latent_class_mxl"):
                    ind_opts = ["(none)"] + sf_columns
                    sf_ind_id_col = st.selectbox("Individual ID column (panel)", ind_opts, key="sf_ind_id")
                    sf_ind_id_col = None if sf_ind_id_col == "(none)" else sf_ind_id_col
            with c3:
                sf_base_alt = st.text_input("Base alternative", value="", key="sf_base_alt")
        avail_opts = ["(none)"] + sf_columns
        sf_avail_col = st.selectbox("Availability column (optional)", avail_opts, key="sf_avail_col")
        sf_avail_col = None if sf_avail_col == "(none)" else sf_avail_col

        st.subheader("3. Candidate variables")
        exclude_set = {sf_choice_col, sf_alt_col, sf_choice_id_col, sf_ind_id_col, sf_avail_col}
        sf_asvarnames = st.multiselect(
            "Alternative-specific variables (varnames)",
            [c for c in sf_columns if c not in exclude_set], key="sf_asvars",
        )
        sf_isvarnames = st.multiselect(
            "Individual-specific variables (optional)",
            [c for c in sf_columns if c not in exclude_set and c not in sf_asvarnames], key="sf_isvars",
        )
        sf_fit_intercept = st.checkbox("Fit intercept", value=(family != "rrm"), key="sf_fit_intercept")

        if family == "mixed_logit":
            st.subheader("4. Random parameters")
            sf_random_vars_sel = st.multiselect("Random variables", sf_asvarnames, key="sf_random_vars")
            c1, c2 = st.columns(2)
            with c1:
                sf_dist_choice = st.selectbox(
                    "Distribution", list(SEARCHLIBRIUM_DISTRIBUTIONS.keys()),
                    format_func=lambda k: f"{k} — {SEARCHLIBRIUM_DISTRIBUTIONS[k]}",
                    disabled=not sf_random_vars_sel, key="sf_dist",
                )
            with c2:
                sf_n_draws = st.number_input("Halton draws (n_draws)", 50, 5000, 200, step=50, key="sf_ndraws")
            sf_randvars = {v: sf_dist_choice for v in sf_random_vars_sel}
            sf_correlated_vars = st.multiselect("Correlated random variables (optional)", sf_random_vars_sel, key="sf_corvars")

        if family == "nested":
            st.subheader("4. Nest structure")
            st.caption(
                "nests values are 0-based positions into the *sorted unique* alt values (not the raw "
                "alt labels) — e.g. for alts ['CAR','SM','TRAIN'] (sorted), positions are CAR=0, SM=1, TRAIN=2."
            )
            sf_nests_json = st.text_area("nests (JSON)", value='{"Car": [0], "Transit": [1, 2]}', key="sf_nests")
            sf_lambdas_json = st.text_area("lambdas (JSON, optional)", value='{"Car": 1, "Transit": 1}', key="sf_lambdas")

        if family == "latent_class_mxl":
            st.subheader("4. Latent class settings")
            c1, c2 = st.columns(2)
            with c1:
                sf_n_classes = st.number_input("Number of classes", 2, 6, 2, key="sf_n_classes")
            with c2:
                sf_maxiter = st.number_input("Max EM iterations", 5, 500, 50, key="sf_lc_maxiter")
            sf_membership_vars = st.multiselect(
                "Class-membership variables (optional — defaults to all variables)",
                sf_asvarnames, key="sf_membership_vars",
            )

    elif family == "ordered":
        st.subheader("2. Column mapping")
        c1, c2 = st.columns(2)
        with c1:
            sf_ordinal_y_col = st.selectbox("Ordinal outcome column", sf_columns, key="sf_ord_y") if sf_columns else st.text_input("Ordinal outcome column", "y", key="sf_ord_y_t")
        with c2:
            sf_n_categories = st.number_input("Number of ordinal categories (J)", 3, 20, 3, key="sf_n_cat")
        c1, c2, c3 = st.columns(3)
        with c1:
            sf_ordered_distr = st.selectbox("Link", ["logit", "probit"], key="sf_ord_distr")
        with c2:
            sf_normalize = st.checkbox("Normalize predictors", key="sf_ord_norm")
        with c3:
            sf_fit_intercept = st.checkbox("Fit intercept", key="sf_ord_intercept")

        st.subheader("3. Candidate variables")
        sf_asvarnames = st.multiselect(
            "Predictor variables", [c for c in sf_columns if c != sf_ordinal_y_col], key="sf_ord_vars",
        )

    elif family == "binary_probit":
        st.subheader("2. Column mapping")
        c1, c2 = st.columns(2)
        with c1:
            sf_binary_y_col = st.selectbox("Binary outcome column (0/1)", sf_columns, key="sf_bin_y") if sf_columns else st.text_input("Binary outcome column", "y", key="sf_bin_y_t")
        with c2:
            sf_fit_intercept = st.checkbox("Fit intercept", value=True, key="sf_bin_intercept")
        st.subheader("3. Candidate variables")
        sf_asvarnames = st.multiselect(
            "Predictor variables", [c for c in sf_columns if c != sf_binary_y_col], key="sf_bin_vars",
        )

    else:  # heckman
        st.subheader("2. Column mapping")
        c1, c2 = st.columns(2)
        with c1:
            sf_selection_y_col = st.selectbox("Selection outcome column (0/1)", sf_columns, key="sf_heck_sely") if sf_columns else st.text_input("Selection outcome column", "selected", key="sf_heck_sely_t")
            sf_selection_varnames = st.multiselect(
                "Selection-stage predictors", [c for c in sf_columns if c != sf_selection_y_col], key="sf_heck_selvars",
            )
        with c2:
            outcome_opts = [c for c in sf_columns if c not in (sf_selection_y_col, *sf_selection_varnames)]
            sf_outcome_y_col = st.selectbox("Outcome column (continuous)", outcome_opts, key="sf_heck_outy") if outcome_opts else st.text_input("Outcome column", "y", key="sf_heck_outy_t")
            sf_outcome_varnames = st.multiselect(
                "Outcome-stage predictors", [c for c in outcome_opts if c != sf_outcome_y_col], key="sf_heck_outvars",
            )
        sf_fit_intercept = st.checkbox("Fit intercept", value=True, key="sf_heck_intercept")

    st.subheader("5. Job naming & output")
    c1, c2, c3 = st.columns(3)
    with c1:
        sf_experiment_name = st.text_input("Experiment name", value="standalone_fit", key="sf_exp_name")
    with c2:
        sf_output_dir = st.text_input("Output directory", value="results", key="sf_output_dir")
    with c3:
        sf_ncpus = st.number_input("HPC ncpus", 1, 64, 2, key="sf_ncpus")
    c1, c2 = st.columns(2)
    with c1:
        sf_mem_gb = st.number_input("HPC mem (GB)", 4, 512, 16, key="sf_mem")
    with c2:
        sf_walltime = st.text_input("HPC walltime", value="4:00:00", key="sf_walltime")

    st.divider()

    sf_has_data = sf_use_bundled is not None or sf_df is not None
    if is_choice_family:
        sf_ready = bool(sf_has_data and (sf_asvarnames or sf_isvarnames) and sf_choice_col)
    elif family in ("ordered", "binary_probit"):
        sf_ready = bool(sf_has_data and sf_asvarnames and (sf_ordinal_y_col if family == "ordered" else sf_binary_y_col))
    else:  # heckman
        sf_ready = bool(sf_has_data and sf_selection_varnames and sf_outcome_varnames and sf_selection_y_col and sf_outcome_y_col)

    if not sf_ready:
        st.warning("Select data and the required columns/variables above to generate a script.")
    else:
        sf_hpc_data_filename = Path(sf_data_path).name if sf_data_path else "data.csv"
        sf_cfg = StandaloneFitConfig(
            data_path=sf_data_path, hpc_data_filename=sf_hpc_data_filename, model_class=model_class,
            use_bundled=sf_use_bundled,
            choice_col=sf_choice_col, alt_col=sf_alt_col, choice_id_col=sf_choice_id_col,
            ind_id_col=sf_ind_id_col, avail_col=sf_avail_col, base_alt=sf_base_alt,
            asvarnames=sf_asvarnames, isvarnames=sf_isvarnames,
            fit_intercept=bool(sf_fit_intercept), maxiter=int(sf_maxiter),
            randvars=sf_randvars, correlated_vars=sf_correlated_vars, n_draws=int(sf_n_draws),
            nests_json=sf_nests_json, lambdas_json=sf_lambdas_json,
            ordinal_y_col=sf_ordinal_y_col, n_categories=int(sf_n_categories),
            ordered_distr=sf_ordered_distr, normalize=bool(sf_normalize),
            binary_y_col=sf_binary_y_col,
            selection_y_col=sf_selection_y_col, selection_varnames=sf_selection_varnames,
            outcome_y_col=sf_outcome_y_col, outcome_varnames=sf_outcome_varnames,
            n_classes=int(sf_n_classes), membership_vars=sf_membership_vars,
            output_dir=sf_output_dir, experiment_name=sf_experiment_name,
        )

        sf_local_script = generate_standalone_fit_script(sf_cfg, for_hpc=False)
        sf_hpc_script = generate_standalone_fit_script(sf_cfg, for_hpc=True)
        sf_pbs_script = generate_pbs_script(PbsConfig(
            job_name=sf_experiment_name, script_filename="run_standalone_fit.py",
            ncpus=int(sf_ncpus), mem_gb=int(sf_mem_gb), walltime=sf_walltime,
        ))

        render_run_and_export(
            key_prefix="standalone",
            local_script=sf_local_script, local_script_name="run_standalone_fit.py",
            hpc_script=sf_hpc_script, hpc_script_name="run_standalone_fit.py",
            pbs_script=sf_pbs_script, pbs_script_name=f"{sf_experiment_name}.pbs",
            engine_python=engine_python,
            data_file_to_bundle=sf_uploaded_path,
        )

with tab_mdcev:
    st.caption(
        "Translated-utility MDCEV allocator (`MDCEVModel`) for continuous budget splits — e.g. daily "
        "time-use minutes or discretionary spend split across activities/categories. Data should have "
        "one row per observation, with one column per alternative holding its allocated amount "
        "(these should sum to each observation's total budget)."
    )

    st.subheader("1. Data")
    md_data_choice = st.radio("Data source", ["Upload CSV", "Path on disk"], horizontal=True, key="md_data_choice")
    md_df: pd.DataFrame | None = None
    md_data_path = ""
    md_uploaded_path: Path | None = None
    if md_data_choice == "Upload CSV":
        md_up = st.file_uploader("CSV file", type=["csv"], key="md_upload")
        if md_up is not None:
            md_job_dir = Path(__file__).resolve().parent.parent.parent / "generated_jobs" / "_uploads"
            md_job_dir.mkdir(parents=True, exist_ok=True)
            md_uploaded_path = md_job_dir / md_up.name
            md_uploaded_path.write_bytes(md_up.getvalue())
            md_data_path = str(md_uploaded_path)
            md_df = pd.read_csv(md_uploaded_path)
            st.dataframe(md_df.head(20), use_container_width=True, height=200)
        md_columns = list(md_df.columns) if md_df is not None else []
    else:
        md_data_path = st.text_input("CSV path (must be readable by the engine interpreter)", value="", key="md_path")
        if md_data_path and Path(md_data_path).exists():
            try:
                md_df = pd.read_csv(md_data_path, nrows=5000)
                st.dataframe(md_df.head(20), use_container_width=True, height=200)
            except Exception as e:
                st.warning(f"Could not preview file: {e}")
        md_columns = list(md_df.columns) if md_df is not None else []

    if not md_columns:
        st.info("Provide data to continue.")

    st.subheader("2. Allocation columns")
    md_allocation_cols = st.multiselect(
        "Columns to allocate the budget across (each row's total = its budget)",
        md_columns, key="md_alloc_cols",
    )
    md_outside_good_opts = ["(none)"] + md_allocation_cols
    md_outside_good_col = st.selectbox(
        "Outside good / numeraire (optional — modeled with near-zero satiation)",
        md_outside_good_opts, key="md_outside_good",
    )
    md_outside_good_col = None if md_outside_good_col == "(none)" else md_outside_good_col

    with st.expander("Advanced model parameters"):
        c1, c2 = st.columns(2)
        with c1:
            md_alpha_floor = st.number_input("Alpha floor", 0.0, 0.5, 0.05, step=0.01, key="md_alpha_floor")
            md_alpha_cap = st.number_input("Alpha cap", 0.5, 0.999, 0.95, step=0.01, key="md_alpha_cap")
        with c2:
            md_gamma_floor = st.number_input("Gamma floor", 1e-6, 1.0, 1e-3, step=1e-3, format="%.6f", key="md_gamma_floor")
            md_tol = st.number_input("Numerical tolerance", 1e-12, 1e-3, 1e-9, step=1e-9, format="%.1e", key="md_tol")

    st.subheader("3. Fit mode")
    md_fit_mode_label = st.radio(
        "Fit mode",
        ["Heuristic (fast, moment-based)", "MLE refinement (slower, JAX autodiff quasi-MLE)"],
        key="md_fit_mode",
    )
    md_fit_mode = "heuristic" if md_fit_mode_label.startswith("Heuristic") else "mle"
    md_mle_maxiter, md_mle_l2 = 400, 1e-4
    if md_fit_mode == "mle":
        c1, c2 = st.columns(2)
        with c1:
            md_mle_maxiter = st.number_input("Max iterations", 20, 5000, 400, step=20, key="md_mle_maxiter")
        with c2:
            md_mle_l2 = st.number_input("L2 penalty", 0.0, 1.0, 1e-4, step=1e-4, format="%.5f", key="md_mle_l2")

    st.subheader("4. Prediction / simulation")
    st.caption(
        "Deterministic `predict()` can show corner solutions (100% to one alternative) when its "
        "gamma is small relative to others — this is expected translated-utility MDCEV behavior, "
        "not a bug. `simulate()` adds Gumbel utility shocks across draws for realistic diversified "
        "predictions."
    )
    md_budgets_text = st.text_input("Budget levels to predict for (comma-separated)", value="100, 300, 500", key="md_budgets")
    try:
        md_predict_budgets = [float(x.strip()) for x in md_budgets_text.split(",") if x.strip()]
    except ValueError:
        md_predict_budgets = []
        st.warning("Could not parse budget levels — use comma-separated numbers, e.g. `100, 300, 500`.")
    md_run_simulation = st.checkbox("Also run stochastic simulation", key="md_run_sim")
    md_n_draws, md_sim_seed = 100, 42
    if md_run_simulation:
        c1, c2 = st.columns(2)
        with c1:
            md_n_draws = st.number_input("Simulation draws", 10, 5000, 100, step=10, key="md_n_draws")
        with c2:
            md_sim_seed = st.number_input("Random seed", 0, 999999, 42, key="md_sim_seed")

    st.subheader("5. Job naming & output")
    c1, c2, c3 = st.columns(3)
    with c1:
        md_experiment_name = st.text_input("Experiment name", value="mdcev_run", key="md_exp_name")
    with c2:
        md_output_dir = st.text_input("Output directory", value="results", key="md_output_dir")
    with c3:
        md_ncpus = st.number_input("HPC ncpus", 1, 64, 2, key="md_ncpus")
    c1, c2 = st.columns(2)
    with c1:
        md_mem_gb = st.number_input("HPC mem (GB)", 4, 512, 16, key="md_mem")
    with c2:
        md_walltime = st.text_input("HPC walltime", value="4:00:00", key="md_walltime")

    st.divider()

    md_ready = bool(md_df is not None and len(md_allocation_cols) >= 2 and md_predict_budgets)
    if not md_ready:
        st.warning("Select data, at least two allocation columns, and valid budget levels to generate a script.")
    else:
        md_hpc_data_filename = Path(md_data_path).name if md_data_path else "data.csv"
        md_cfg = MDCEVConfig(
            data_path=md_data_path, hpc_data_filename=md_hpc_data_filename,
            allocation_cols=md_allocation_cols, outside_good_col=md_outside_good_col,
            alpha_floor=float(md_alpha_floor), alpha_cap=float(md_alpha_cap),
            gamma_floor=float(md_gamma_floor), tol=float(md_tol),
            fit_mode=md_fit_mode, mle_maxiter=int(md_mle_maxiter), mle_l2_penalty=float(md_mle_l2),
            predict_budgets=md_predict_budgets,
            run_simulation=bool(md_run_simulation), n_draws=int(md_n_draws), sim_seed=int(md_sim_seed),
            output_dir=md_output_dir, experiment_name=md_experiment_name,
        )

        md_local_script = generate_mdcev_script(md_cfg, for_hpc=False)
        md_hpc_script = generate_mdcev_script(md_cfg, for_hpc=True)
        md_pbs_script = generate_pbs_script(PbsConfig(
            job_name=md_experiment_name, script_filename="run_mdcev.py",
            ncpus=int(md_ncpus), mem_gb=int(md_mem_gb), walltime=md_walltime,
        ))

        render_run_and_export(
            key_prefix="mdcev",
            local_script=md_local_script, local_script_name="run_mdcev.py",
            hpc_script=md_hpc_script, hpc_script_name="run_mdcev.py",
            pbs_script=md_pbs_script, pbs_script_name=f"{md_experiment_name}.pbs",
            engine_python=engine_python,
            data_file_to_bundle=md_uploaded_path,
        )

with tab_destination:
    st.caption(
        "Fit a discrete-choice model on trip-level destination-choice data (long format: one row per "
        "trip per candidate destination), then predict the most likely destination per trip and "
        "aggregate flows per destination, compared against the observed data."
    )
    st.info(
        "The fit and the prediction always run over the exact same dataset in one script. "
        "SearchLibrium's `DestinationPredictor` reuses the fitted model's cached probability arrays "
        "whenever their shape matches, so predicting on a *different* dataset than the model was fit "
        "on can silently return stale results from the original fit — fitting and predicting together "
        "avoids that by construction. If you need true out-of-sample prediction, refit the model on "
        "the exact evaluation dataset first."
    )

    dp_model_class = st.selectbox("Model class", list(DESTINATION_MODEL_FAMILIES.keys()), key="dp_model_class")
    dp_family = DESTINATION_MODEL_FAMILIES[dp_model_class]

    st.subheader("1. Data")
    dp_data_choice = st.radio("Data source", ["Upload CSV", "Path on disk"], horizontal=True, key="dp_data_choice")
    dp_df: pd.DataFrame | None = None
    dp_data_path = ""
    dp_uploaded_path: Path | None = None
    if dp_data_choice == "Upload CSV":
        dp_up = st.file_uploader("CSV file", type=["csv"], key="dp_upload")
        if dp_up is not None:
            dp_job_dir = Path(__file__).resolve().parent.parent.parent / "generated_jobs" / "_uploads"
            dp_job_dir.mkdir(parents=True, exist_ok=True)
            dp_uploaded_path = dp_job_dir / dp_up.name
            dp_uploaded_path.write_bytes(dp_up.getvalue())
            dp_data_path = str(dp_uploaded_path)
            dp_df = pd.read_csv(dp_uploaded_path)
            st.dataframe(dp_df.head(20), use_container_width=True, height=200)
        dp_columns = list(dp_df.columns) if dp_df is not None else []
    else:
        dp_data_path = st.text_input("CSV path (must be readable by the engine interpreter)", value="", key="dp_path")
        if dp_data_path and Path(dp_data_path).exists():
            try:
                dp_df = pd.read_csv(dp_data_path, nrows=5000)
                st.dataframe(dp_df.head(20), use_container_width=True, height=200)
            except Exception as e:
                st.warning(f"Could not preview file: {e}")
        dp_columns = list(dp_df.columns) if dp_df is not None else []

    if not dp_columns:
        st.info("Provide data to continue.")

    st.subheader("2. Column mapping")
    c1, c2, c3 = st.columns(3)
    with c1:
        dp_trip_id_col = st.selectbox("Trip ID column", dp_columns, key="dp_trip_id") if dp_columns else st.text_input("Trip ID column", "trip_id", key="dp_trip_id_t")
        dp_dest_col = st.selectbox("Destination column", dp_columns, key="dp_dest_col") if dp_columns else st.text_input("Destination column", "dest_code", key="dp_dest_col_t")
    with c2:
        dp_choice_col = st.selectbox("Chosen indicator column (0/1)", dp_columns, key="dp_choice_col") if dp_columns else st.text_input("Chosen column", "chosen", key="dp_choice_col_t")
        dest_name_opts = ["(none)"] + dp_columns
        dp_dest_name_col = st.selectbox("Destination name column (optional)", dest_name_opts, key="dp_dest_name_col")
        dp_dest_name_col = None if dp_dest_name_col == "(none)" else dp_dest_name_col
    with c3:
        dp_base_alt = st.text_input("Base alternative (for the fit)", value="", key="dp_base_alt")
        avail_opts = ["(none)"] + dp_columns
        dp_avail_col = st.selectbox("Availability column (optional)", avail_opts, key="dp_avail_col")
        dp_avail_col = None if dp_avail_col == "(none)" else dp_avail_col

    st.subheader("3. Candidate variables")
    dp_exclude = {dp_trip_id_col, dp_dest_col, dp_choice_col, dp_dest_name_col, dp_avail_col}
    dp_varnames = st.multiselect(
        "Alternative-specific variables used in the utility function",
        [c for c in dp_columns if c not in dp_exclude], key="dp_varnames",
    )
    dp_isvarnames = st.multiselect(
        "Individual-specific variables (optional)",
        [c for c in dp_columns if c not in dp_exclude and c not in dp_varnames], key="dp_isvarnames",
    )
    dp_fit_intercept = st.checkbox("Fit intercept", value=True, key="dp_fit_intercept")

    dp_randvars: dict[str, str] = {}
    dp_n_draws = 200
    dp_nests_json = dp_lambdas_json = None
    if dp_family == "mixed_logit":
        st.subheader("4. Random parameters")
        dp_random_vars_sel = st.multiselect("Random variables", dp_varnames, key="dp_random_vars")
        c1, c2 = st.columns(2)
        with c1:
            dp_dist_choice = st.selectbox(
                "Distribution", list(SEARCHLIBRIUM_DISTRIBUTIONS.keys()),
                format_func=lambda k: f"{k} — {SEARCHLIBRIUM_DISTRIBUTIONS[k]}",
                disabled=not dp_random_vars_sel, key="dp_dist",
            )
        with c2:
            dp_n_draws = st.number_input("Halton draws (n_draws)", 50, 5000, 200, step=50, key="dp_ndraws")
        dp_randvars = {v: dp_dist_choice for v in dp_random_vars_sel}

    if dp_family == "nested":
        st.subheader("4. Nest structure")
        st.caption(
            "nests values are 0-based positions into the *sorted unique* destination values (not the "
            "raw labels) — same convention as the Standalone fit tab's NestedLogit."
        )
        dp_nests_json = st.text_area("nests (JSON)", value='{"Car": [0], "Transit": [1, 2]}', key="dp_nests")
        dp_lambdas_json = st.text_area("lambdas (JSON, optional)", value='{"Car": 1, "Transit": 1}', key="dp_lambdas")

    st.subheader("5. Job naming & output")
    c1, c2, c3 = st.columns(3)
    with c1:
        dp_experiment_name = st.text_input("Experiment name", value="destination_prediction", key="dp_exp_name")
    with c2:
        dp_output_dir = st.text_input("Output directory", value="results", key="dp_output_dir")
    with c3:
        dp_ncpus = st.number_input("HPC ncpus", 1, 64, 2, key="dp_ncpus")
    c1, c2 = st.columns(2)
    with c1:
        dp_mem_gb = st.number_input("HPC mem (GB)", 4, 512, 16, key="dp_mem")
    with c2:
        dp_walltime = st.text_input("HPC walltime", value="4:00:00", key="dp_walltime")

    st.divider()

    dp_ready = bool(dp_df is not None and dp_varnames and dp_trip_id_col and dp_dest_col and dp_choice_col)
    if not dp_ready:
        st.warning("Select data, column mapping, and at least one candidate variable to generate a script.")
    else:
        dp_hpc_data_filename = Path(dp_data_path).name if dp_data_path else "data.csv"
        dp_cfg = DestinationPredictionConfig(
            data_path=dp_data_path, hpc_data_filename=dp_hpc_data_filename, model_class=dp_model_class,
            trip_id_col=dp_trip_id_col, dest_col=dp_dest_col, choice_col=dp_choice_col,
            dest_name_col=dp_dest_name_col, avail_col=dp_avail_col,
            varnames=dp_varnames, isvarnames=dp_isvarnames, base_alt=dp_base_alt,
            fit_intercept=bool(dp_fit_intercept), randvars=dp_randvars, n_draws=int(dp_n_draws),
            nests_json=dp_nests_json, lambdas_json=dp_lambdas_json,
            output_dir=dp_output_dir, experiment_name=dp_experiment_name,
        )

        dp_local_script = generate_destination_prediction_script(dp_cfg, for_hpc=False)
        dp_hpc_script = generate_destination_prediction_script(dp_cfg, for_hpc=True)
        dp_pbs_script = generate_pbs_script(PbsConfig(
            job_name=dp_experiment_name, script_filename="run_destination_prediction.py",
            ncpus=int(dp_ncpus), mem_gb=int(dp_mem_gb), walltime=dp_walltime,
        ))

        render_run_and_export(
            key_prefix="destination",
            local_script=dp_local_script, local_script_name="run_destination_prediction.py",
            hpc_script=dp_hpc_script, hpc_script_name="run_destination_prediction.py",
            pbs_script=dp_pbs_script, pbs_script_name=f"{dp_experiment_name}.pbs",
            engine_python=engine_python,
            data_file_to_bundle=dp_uploaded_path,
        )
