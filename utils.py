import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold


ACTIVITY_INDICATORS = ["s_mw", "ha", "na", "ca", "rhk"]
LINE_INDICATORS = ["vrad", "true_vrad", "fwhm", "bis_span", "contrast"]
VALID_NORMALIZATION_METHODS = {"subtract_mean", "fractional_mean", "zscore", "none"}


def normalize_star_name(star_name):
    return re.sub(r"[^a-z0-9]", "", str(star_name).lower())


def observation_name_from_corrected_path(obs_path):
    return Path(obs_path).name.replace("_linear_corrected.csv", ".rdb")


def star_name_from_corrected_path(obs_path):
    stem = Path(obs_path).name.replace("_linear_corrected.csv", "")
    parts = stem.split("_")

    if len(parts) >= 3 and parts[-1].isdigit() and re.search(r"[a-z]", parts[-2], flags=re.IGNORECASE):
        return "_".join(parts[:-2])

    return stem


def select_corrected_files(corrected_files, include_observations=None, exclude_observations=None):
    include_observations = set(include_observations or [])
    exclude_observations = set(exclude_observations or [])

    selected_files = []
    for obs_path in corrected_files:
        obs_name = observation_name_from_corrected_path(obs_path)
        if include_observations and obs_name not in include_observations:
            continue
        if obs_name in exclude_observations:
            continue
        selected_files.append(obs_path)

    return selected_files


def _build_observations_df(corrected_files):
    return pd.DataFrame(
        [
            {
                "corrected_file": obs_path.name,
                "observation_file": observation_name_from_corrected_path(obs_path),
                "star_key": normalize_star_name(star_name_from_corrected_path(obs_path)),
            }
            for obs_path in corrected_files
        ]
    )


def _sample_catalog_with_keys(sample_catalog_df):
    return sample_catalog_df.assign(star_key=sample_catalog_df["star"].map(normalize_star_name))


def _group_stat_series(df, column, stat, group_column):
    if group_column in df.columns:
        return df.groupby(group_column)[column].transform(stat)

    value = getattr(df[column], stat)(skipna=True)
    return pd.Series(value, index=df.index, dtype="float64")


def _normalize_series(values, mean_values, std_values, method):
    if method == "subtract_mean":
        return values - mean_values

    if method == "fractional_mean":
        denominator = mean_values.where(mean_values != 0)
        return (values - mean_values) / denominator

    if method == "zscore":
        denominator = std_values.where(std_values != 0)
        return (values - mean_values) / denominator

    if method == "none":
        return values.copy()

    raise ValueError(
        "method must be one of 'subtract_mean', 'fractional_mean', 'zscore', or 'none'"
    )


def _validate_normalization_method(method):
    if method not in VALID_NORMALIZATION_METHODS:
        raise ValueError(
            f"Unsupported normalization method {method!r}. "
            f"Expected one of {sorted(VALID_NORMALIZATION_METHODS)}."
        )


def add_group_stat_columns(
    df,
    columns,
    group_column="observation_file",
    mean_suffix="_mean",
    std_suffix="_std",
    overwrite=False,
):
    """
    Add per-group mean/std columns for the requested source columns.

    Example:
    `add_group_stat_columns(df, ['s_mw'])` adds `s_mw_mean` and `s_mw_std`.
    """
    result_df = df.copy()

    for column in columns:
        if column not in result_df.columns:
            continue

        mean_column = f"{column}{mean_suffix}"
        std_column = f"{column}{std_suffix}"

        if not overwrite and mean_column in result_df.columns:
            raise ValueError(f"Column {mean_column!r} already exists.")
        if not overwrite and std_column in result_df.columns:
            raise ValueError(f"Column {std_column!r} already exists.")

        result_df[mean_column] = _group_stat_series(result_df, column, "mean", group_column)
        result_df[std_column] = _group_stat_series(result_df, column, "std", group_column)

    return result_df


def add_normalized_columns(
    df,
    columns,
    method,
    group_column="observation_file",
    round_decimals=None,
    suffix=None,
    add_activity_stats=True,
    activity_columns=None,
    add_group_stats=False,
    stat_columns=None,
    overwrite=False,
):
    """
    Append normalized versions of existing columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    columns : list[str]
        Columns to normalize.
    method : str
        One of `subtract_mean`, `fractional_mean`, `zscore`, or `none`.
    group_column : str
        Grouping column used to compute per-observation statistics.
    round_decimals : int | None
        Optional rounding for the newly created normalized columns.
    suffix : str | None
        Output suffix. Defaults to `_{method}`.
    add_activity_stats : bool
        If true, add `<column>_mean` and `<column>_std` for requested activity columns.
    activity_columns : list[str] | None
        Which columns should receive mean/std helper columns.
    add_group_stats : bool
        If true, add `<column>_mean` and `<column>_std` for requested columns present
        in `stat_columns`, regardless of whether they are activity indicators.
    stat_columns : list[str] | None
        Which columns are eligible for mean/std helper columns when
        `add_group_stats=True`. Defaults to all requested `columns`.
    overwrite : bool
        Whether existing output columns may be replaced.
    """
    _validate_normalization_method(method)

    result_df = df.copy()
    output_suffix = suffix or f"_{method}"
    activity_columns = set(activity_columns or ACTIVITY_INDICATORS)
    stat_columns = set(stat_columns or columns)

    if add_activity_stats:
        requested_activity_columns = [column for column in columns if column in activity_columns]
        missing_stat_columns = [
            column
            for column in requested_activity_columns
            if f"{column}_mean" not in result_df.columns or f"{column}_std" not in result_df.columns
        ]
        if missing_stat_columns:
            result_df = add_group_stat_columns(
                result_df,
                missing_stat_columns,
                group_column=group_column,
                overwrite=overwrite,
            )

    if add_group_stats:
        requested_stat_columns = [column for column in columns if column in stat_columns]
        missing_stat_columns = [
            column
            for column in requested_stat_columns
            if f"{column}_mean" not in result_df.columns or f"{column}_std" not in result_df.columns
        ]
        if missing_stat_columns:
            result_df = add_group_stat_columns(
                result_df,
                missing_stat_columns,
                group_column=group_column,
                overwrite=overwrite,
            )

    for column in columns:
        if column not in result_df.columns:
            continue

        output_column = f"{column}{output_suffix}"
        if output_column in result_df.columns and not overwrite:
            raise ValueError(f"Column {output_column!r} already exists.")

        mean_values = _group_stat_series(result_df, column, "mean", group_column)
        std_values = _group_stat_series(result_df, column, "std", group_column)
        normalized_values = _normalize_series(result_df[column], mean_values, std_values, method)

        if round_decimals is not None:
            normalized_values = normalized_values.round(round_decimals)

        result_df[output_column] = normalized_values

    return result_df


def apply_normalization_specs(
    df,
    normalization_specs,
    group_column="observation_file",
    add_activity_stats=True,
    activity_columns=None,
    add_group_stats=False,
    stat_columns=None,
    overwrite=False,
):
    """
    Apply multiple normalization requests in sequence.

    Example
    -------
    normalization_specs = [
        {"columns": LINE_INDICATORS, "method": "fractional_mean"},
        {"columns": LINE_INDICATORS, "method": "subtract_mean"},
        {"columns": ACTIVITY_INDICATORS, "method": "fractional_mean"},
    ]
    """
    result_df = df.copy()

    for spec in normalization_specs:
        spec = dict(spec)
        result_df = add_normalized_columns(
            result_df,
            columns=spec["columns"],
            method=spec["method"],
            group_column=spec.get("group_column", group_column),
            round_decimals=spec.get("round_decimals"),
            suffix=spec.get("suffix"),
            add_activity_stats=spec.get("add_activity_stats", add_activity_stats),
            activity_columns=spec.get("activity_columns", activity_columns),
            add_group_stats=spec.get("add_group_stats", add_group_stats),
            stat_columns=spec.get("stat_columns", stat_columns),
            overwrite=spec.get("overwrite", overwrite),
        )

    return result_df


def add_column_normalizations(
    df,
    columns,
    methods,
    group_column="observation_file",
    round_decimals=None,
    suffix_map=None,
    add_activity_stats=True,
    activity_columns=None,
    add_group_stats=False,
    stat_columns=None,
    overwrite=False,
):
    """
    Apply one or more normalization methods to the same list of columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    columns : list[str]
        Columns to normalize.
    methods : str | list[str]
        One method or a list such as `["subtract_mean", "zscore"]`.
    group_column : str
        Grouping column used to compute per-observation statistics.
    round_decimals : int | dict[str, int] | None
        Optional rounding for created columns. May be one integer for all methods
        or a dict keyed by method name.
    suffix_map : dict[str, str] | None
        Optional custom suffix per method, e.g. `{"zscore": "_z"}`.
    add_activity_stats, activity_columns, add_group_stats, stat_columns, overwrite
        Passed through to `add_normalized_columns`.
    """
    if isinstance(methods, str):
        methods = [methods]

    result_df = df.copy()
    suffix_map = suffix_map or {}

    for method in methods:
        method_round_decimals = round_decimals
        if isinstance(round_decimals, dict):
            method_round_decimals = round_decimals.get(method)

        result_df = add_normalized_columns(
            result_df,
            columns=columns,
            method=method,
            group_column=group_column,
            round_decimals=method_round_decimals,
            suffix=suffix_map.get(method),
            add_activity_stats=add_activity_stats,
            activity_columns=activity_columns,
            add_group_stats=add_group_stats,
            stat_columns=stat_columns,
            overwrite=overwrite,
        )

    return result_df


def add_normalizations_from_requests(
    df,
    requests,
    group_column="observation_file",
    add_activity_stats=True,
    activity_columns=None,
    add_group_stats=False,
    stat_columns=None,
    overwrite=False,
):
    """
    Notebook-friendly wrapper for adding multiple normalization requests directly
    to an existing dataframe.

    Example
    -------
    requests = [
        {"columns": ["vrad", "fwhm"], "methods": ["subtract_mean", "zscore"]},
        {"columns": ["s_mw", "ha"], "methods": "fractional_mean"},
        {
            "columns": ["iccf_rv", "iccf_bis"],
            "methods": ["subtract_mean"],
            "group_column": "observation_file",
        },
    ]
    """
    result_df = df.copy()

    for request in requests:
        request = dict(request)
        result_df = add_column_normalizations(
            result_df,
            columns=request["columns"],
            methods=request["methods"],
            group_column=request.get("group_column", group_column),
            round_decimals=request.get("round_decimals"),
            suffix_map=request.get("suffix_map"),
            add_activity_stats=request.get("add_activity_stats", add_activity_stats),
            activity_columns=request.get("activity_columns", activity_columns),
            add_group_stats=request.get("add_group_stats", add_group_stats),
            stat_columns=request.get("stat_columns", stat_columns),
            overwrite=request.get("overwrite", overwrite),
        )

    return result_df


def normalize_observation_columns(
    obs_df,
    columns,
    method,
    group_column="observation_file",
    round_decimals=None,
    suffix=None,
    add_activity_stats=True,
    add_group_stats=False,
    stat_columns=None,
    overwrite=False,
):
    """
    Backward-compatible wrapper around `add_normalized_columns`.
    """
    return add_normalized_columns(
        obs_df,
        columns=columns,
        method=method,
        group_column=group_column,
        round_decimals=round_decimals,
        suffix=suffix,
        add_activity_stats=add_activity_stats,
        add_group_stats=add_group_stats,
        stat_columns=stat_columns,
        overwrite=overwrite,
    )


def _unique_existing_columns(df, columns):
    seen = set()
    unique_columns = []
    for column in columns:
        if column in df.columns and column not in seen:
            unique_columns.append(column)
            seen.add(column)
    return unique_columns


def _regression_metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "r2": r2_score(y_true, y_pred),
    }


def evaluate_feature_sets_with_group_cv_and_holdout(
    df,
    feature_sets,
    target_column,
    holdout_observation,
    model_factory,
    group_column="observation_file",
    baseline_column=None,
    sort_column="rjd",
    n_splits=5,
    cv_mode="grouped",
    shuffle=True,
    random_state=42,
    holdout_values=None,
    holdout_column=None,
):
    """
    Evaluate feature sets with group-based CV on the training observations and a
    final score on a held-out observation or held-out subset.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing features, target, and observation labels.
    feature_sets : dict[str, list[str]]
        Mapping from model/feature-set name to candidate feature columns.
    target_column : str
        Regression target.
    holdout_observation : str
        Value in `group_column` that should be excluded from CV training and used
        only for final holdout evaluation. Ignored when `holdout_values` is
        provided.
    model_factory : callable
        Zero-argument callable returning a fresh sklearn-style regressor.
    group_column : str
        Observation/group identifier column. Default: observation_file.
    holdout_column : str | None
        Column used to define the holdout subset. Defaults to `group_column`.
    baseline_column : str | None
        Optional baseline RV-like column to copy into prediction tables.
    sort_column : str
        Sort order for prediction tables. Default: rjd.
    n_splits : int
        Requested number of GroupKFold splits.
    """
    model_df = df.copy()
    holdout_column = holdout_column or group_column

    candidate_columns = {target_column, group_column, holdout_column, sort_column}
    if baseline_column is not None:
        candidate_columns.add(baseline_column)
    for columns in feature_sets.values():
        candidate_columns.update(columns)

    for column in candidate_columns:
        if column in model_df.columns and column not in {group_column, holdout_column}:
            model_df[column] = pd.to_numeric(model_df[column], errors="coerce")

    if holdout_values is None:
        if holdout_observation is None:
            raise ValueError(
                "Provide `holdout_observation` or `holdout_values` to define the holdout set."
            )
        holdout_values = [holdout_observation]
    elif isinstance(holdout_values, str):
        holdout_values = [holdout_values]
    else:
        holdout_values = list(holdout_values)

    holdout_mask = model_df[holdout_column].isin(holdout_values)

    train_pool_df = model_df.loc[~holdout_mask].dropna(subset=[target_column]).copy()
    holdout_pool_df = model_df.loc[holdout_mask].dropna(subset=[target_column]).copy()

    if holdout_pool_df.empty:
        raise ValueError(
            f"No rows found for holdout values {holdout_values!r} in column {holdout_column!r}."
        )

    n_train_groups = train_pool_df[group_column].nunique()
    if n_train_groups < 2:
        raise ValueError("Need at least 2 non-holdout observations for group CV.")

    if cv_mode not in {"grouped", "standard"}:
        raise ValueError("cv_mode must be either 'grouped' or 'standard'.")

    if cv_mode == "grouped":
        splitter = GroupKFold(n_splits=min(n_splits, n_train_groups))
    else:
        splitter = KFold(
            n_splits=min(n_splits, len(train_pool_df)),
            shuffle=shuffle,
            random_state=random_state,
        )

    fitted_models = {}
    holdout_predictions = {}
    cv_prediction_tables = {}
    metric_rows = []

    for model_name, feature_columns in feature_sets.items():
        feature_columns = _unique_existing_columns(model_df, feature_columns)
        required_columns = [*feature_columns, target_column]

        train_df = train_pool_df.dropna(subset=required_columns).copy()
        holdout_df = holdout_pool_df.dropna(subset=feature_columns).copy()

        if train_df.empty:
            raise ValueError(f"No training rows remain for feature set {model_name!r}.")
        if holdout_df.empty:
            raise ValueError(f"No holdout rows remain for feature set {model_name!r}.")

        cv_rows = []
        fold_metric_rows = []

        X_train_all = train_df[feature_columns]
        y_train_all = train_df[target_column]
        groups = train_df[group_column]

        if cv_mode == "grouped":
            split_iterator = splitter.split(X_train_all, y_train_all, groups)
        else:
            split_iterator = splitter.split(X_train_all, y_train_all)

        for fold_index, (fit_idx, val_idx) in enumerate(split_iterator, start=1):
            fold_fit_df = train_df.iloc[fit_idx].copy()
            fold_val_df = train_df.iloc[val_idx].copy()

            model = model_factory()
            model.fit(fold_fit_df[feature_columns], fold_fit_df[target_column])

            fold_val_df = fold_val_df.sort_values(sort_column).copy()
            fold_val_df["predicted_true_vrad"] = model.predict(fold_val_df[feature_columns])
            fold_val_df["cv_fold"] = fold_index
            fold_val_df["model_name"] = model_name

            metrics = _regression_metrics(
                fold_val_df[target_column],
                fold_val_df["predicted_true_vrad"],
            )
            metrics["cv_fold"] = fold_index
            fold_metric_rows.append(metrics)
            cv_rows.append(fold_val_df)

        cv_prediction_df = pd.concat(cv_rows, ignore_index=True).sort_values(
            [group_column, sort_column]
        ).reset_index(drop=True)
        cv_metrics_df = pd.DataFrame(fold_metric_rows)

        final_model = model_factory()
        final_model.fit(train_df[feature_columns], train_df[target_column])

        holdout_prediction_df = holdout_df.sort_values(sort_column).copy()
        holdout_prediction_df["predicted_true_vrad"] = final_model.predict(
            holdout_prediction_df[feature_columns]
        )
        holdout_prediction_df["model_name"] = model_name

        cv_summary = {
            "cv_mae_mean": cv_metrics_df["mae"].mean(),
            "cv_mae_std": cv_metrics_df["mae"].std(ddof=0),
            "cv_rmse_mean": cv_metrics_df["rmse"].mean(),
            "cv_rmse_std": cv_metrics_df["rmse"].std(ddof=0),
            "cv_r2_mean": cv_metrics_df["r2"].mean(),
            "cv_r2_std": cv_metrics_df["r2"].std(ddof=0),
            "n_cv_folds": len(cv_metrics_df),
        }
        holdout_metrics = _regression_metrics(
            holdout_prediction_df[target_column],
            holdout_prediction_df["predicted_true_vrad"],
        )

        metric_rows.append(
            {
                "model_name": model_name,
                "n_features": len(feature_columns),
                "features": feature_columns,
                "cv_mode": cv_mode,
                **cv_summary,
                "holdout_mae": holdout_metrics["mae"],
                "holdout_rmse": holdout_metrics["rmse"],
                "holdout_r2": holdout_metrics["r2"],
            }
        )

        fitted_models[model_name] = final_model
        holdout_predictions[model_name] = holdout_prediction_df
        cv_prediction_tables[model_name] = {
            "cv_predictions": cv_prediction_df,
            "cv_fold_metrics": cv_metrics_df,
        }

    metrics_df = pd.DataFrame(metric_rows).sort_values(
        ["cv_rmse_mean", "holdout_rmse", "cv_mae_mean"]
    ).reset_index(drop=True)

    return {
        "metrics_df": metrics_df,
        "models": fitted_models,
        "holdout_predictions": holdout_predictions,
        "cv_tables": cv_prediction_tables,
        "train_pool_df": train_pool_df,
        "holdout_pool_df": holdout_pool_df,
    }


def greedy_stepwise_feature_search(
    df,
    base_features,
    candidate_features,
    target_column,
    holdout_observation,
    model_factory,
    group_column="observation_file",
    baseline_column=None,
    sort_column="rjd",
    n_splits=5,
    cv_mode="grouped",
    shuffle=True,
    random_state=42,
    optimize_metric="cv_rmse_mean",
    minimize=True,
    verbose=False,
):
    """
    Greedy forward-backward feature search using group-CV on non-holdout
    observations plus a final holdout score.

    Strategy
    --------
    1. Start from `base_features`.
    2. Try adding each remaining candidate one-by-one; keep the best improvement.
    3. After an addition, try removing each current base feature one-by-one; if one
       removal improves the score, remove the best such feature.
    4. Repeat until no add/remove step improves the chosen metric.

    Parameters
    ----------
    optimize_metric : str
        Column from `metrics_df` returned by
        `evaluate_feature_sets_with_group_cv_and_holdout`, e.g. `cv_rmse_mean`,
        `holdout_rmse`, `cv_mae_mean`, `holdout_r2`.
    minimize : bool
        If true, lower metric is better. Use false for metrics like R^2.
    """
    available_columns = set(df.columns)
    current_features = [
        column for column in dict.fromkeys(base_features)
        if column in available_columns
    ]
    testing_features = [
        column for column in dict.fromkeys(candidate_features)
        if column in available_columns and column not in current_features
    ]

    if not current_features:
        raise ValueError("No valid base features remain after filtering against dataframe columns.")

    def is_better(new_value, best_value):
        return new_value < best_value if minimize else new_value > best_value

    def evaluate_features(feature_columns):
        result = evaluate_feature_sets_with_group_cv_and_holdout(
            df=df,
            feature_sets={"current": feature_columns},
            target_column=target_column,
            holdout_observation=holdout_observation,
            model_factory=model_factory,
            group_column=group_column,
            baseline_column=baseline_column,
            sort_column=sort_column,
            n_splits=n_splits,
            cv_mode=cv_mode,
            shuffle=shuffle,
            random_state=random_state,
        )
        metrics = result["metrics_df"].iloc[0].to_dict()
        return metrics, result

    best_metrics, best_result = evaluate_features(current_features)
    history = [{
        "step": 0,
        "action": "initial",
        "feature": None,
        "features": list(current_features),
        **best_metrics,
    }]

    if optimize_metric not in best_metrics:
        raise KeyError(f"Unknown optimize_metric {optimize_metric!r}. Available: {sorted(best_metrics)}")

    step = 1
    improved = True

    while improved:
        improved = False

        # Forward step: add one testing feature at a time
        best_add_feature = None
        best_add_metrics = None
        best_add_result = None

        for feature in testing_features:
            trial_features = [*current_features, feature]
            metrics, result = evaluate_features(trial_features)
            if best_add_metrics is None or is_better(metrics[optimize_metric], best_add_metrics[optimize_metric]):
                best_add_feature = feature
                best_add_metrics = metrics
                best_add_result = result

        if best_add_metrics is not None and is_better(best_add_metrics[optimize_metric], best_metrics[optimize_metric]):
            current_features.append(best_add_feature)
            testing_features.remove(best_add_feature)
            best_metrics = best_add_metrics
            best_result = best_add_result
            improved = True
            history.append({
                "step": step,
                "action": "add",
                "feature": best_add_feature,
                "features": list(current_features),
                **best_metrics,
            })
            if verbose:
                print(f"[step {step}] add {best_add_feature}: {optimize_metric}={best_metrics[optimize_metric]:.6f}")
            step += 1

        # Backward step: remove one current base feature at a time
        best_remove_feature = None
        best_remove_metrics = None
        best_remove_result = None

        if len(current_features) > 1:
            for feature in list(current_features):
                trial_features = [col for col in current_features if col != feature]
                metrics, result = evaluate_features(trial_features)
                if best_remove_metrics is None or is_better(metrics[optimize_metric], best_remove_metrics[optimize_metric]):
                    best_remove_feature = feature
                    best_remove_metrics = metrics
                    best_remove_result = result

        if best_remove_metrics is not None and is_better(best_remove_metrics[optimize_metric], best_metrics[optimize_metric]):
            current_features.remove(best_remove_feature)
            testing_features.append(best_remove_feature)
            best_metrics = best_remove_metrics
            best_result = best_remove_result
            improved = True
            history.append({
                "step": step,
                "action": "remove",
                "feature": best_remove_feature,
                "features": list(current_features),
                **best_metrics,
            })
            if verbose:
                print(f"[step {step}] remove {best_remove_feature}: {optimize_metric}={best_metrics[optimize_metric]:.6f}")
            step += 1

    return {
        "selected_features": list(current_features),
        "remaining_features": list(testing_features),
        "best_metrics": best_metrics,
        "best_result": best_result,
        "history_df": pd.DataFrame(history),
        "optimize_metric": optimize_metric,
        "minimize": minimize,
    }


def create_rm_df(corrected_path, sample_catalog_df, include_observations=None, exclude_observations=None):
    """
    Load corrected observation files and merge them with the stellar catalog.
    """
    corrected_path = Path(corrected_path)
    all_corrected_files = sorted(corrected_path.glob("*_linear_corrected.csv"))
    corrected_files = select_corrected_files(
        all_corrected_files,
        include_observations=include_observations,
        exclude_observations=exclude_observations,
    )

    observations_df = _build_observations_df(corrected_files)
    sample_catalog_with_keys = _sample_catalog_with_keys(sample_catalog_df)

    observation_sample_df = observations_df.merge(
        sample_catalog_with_keys,
        on="star_key",
        how="left",
    ).drop(columns="star_key")
    observation_sample_df = observation_sample_df[
        ["corrected_file", "observation_file", *sample_catalog_df.columns]
    ]

    corrected_tables = []
    for obs_path in corrected_files:
        obs_df = pd.read_csv(obs_path)
        obs_df["corrected_file"] = obs_path.name
        obs_df["observation_file"] = observation_name_from_corrected_path(obs_path)
        obs_df["star_key"] = normalize_star_name(star_name_from_corrected_path(obs_path))
        corrected_tables.append(obs_df)

    rm_df = pd.concat(corrected_tables, ignore_index=True)
    rm_df = rm_df.merge(sample_catalog_with_keys, on="star_key", how="left").drop(columns="star_key")

    return rm_df, observation_sample_df


def create_rm_analysis_df(
    corrected_path,
    sample_catalog_df,
    include_observations=None,
    exclude_observations=None,
    normalization_specs=None,
):
    """
    Convenience helper to build the base dataframe and optionally append derived columns.
    """
    rm_df, observation_sample_df = create_rm_df(
        corrected_path=corrected_path,
        sample_catalog_df=sample_catalog_df,
        include_observations=include_observations,
        exclude_observations=exclude_observations,
    )

    if normalization_specs:
        rm_df = apply_normalization_specs(rm_df, normalization_specs)

    return rm_df, observation_sample_df
