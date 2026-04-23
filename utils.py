import re
from pathlib import Path

import numpy as np
import pandas as pd


ACTIVITY_INDICATORS = ["s_mw", "ha", "na", "ca", "rhk"]
LINE_INDICATORS = ["vrad", "true_vrad", "fwhm", "bis_span", "contrast"]
VALID_NORMALIZATION_METHODS = {"subtract_mean", "fractional_mean", "zscore", "none"}


def normalize_star_name(star_name):
    return re.sub(r"[^a-z0-9]", "", str(star_name).lower())


def observation_name_from_corrected_path(obs_path):
    return Path(obs_path).name.replace("_linear_corrected.csv", ".rdb")


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
                "star_key": normalize_star_name(
                    re.sub(r"_esp.*$", "", obs_path.name.replace("_linear_corrected.csv", ""), flags=re.IGNORECASE)
                ),
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
    overwrite : bool
        Whether existing output columns may be replaced.
    """
    _validate_normalization_method(method)

    result_df = df.copy()
    output_suffix = suffix or f"_{method}"
    activity_columns = set(activity_columns or ACTIVITY_INDICATORS)

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
            overwrite=spec.get("overwrite", overwrite),
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
        overwrite=overwrite,
    )


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
        obs_df["star_key"] = normalize_star_name(
            re.sub(r"_esp.*$", "", obs_path.name.replace("_linear_corrected.csv", ""), flags=re.IGNORECASE)
        )
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
