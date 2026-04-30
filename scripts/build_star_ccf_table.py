#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.process_downloaded_ccf import (
    build_prediction_table_from_local_ccf_directory,
    load_sample_catalog,
)


DEFAULT_CATALOG_PATH = Path("sample_sweetcat.csv")
DEFAULT_COLUMNS_TO_MS = [
    "iccf_bis",
    "iccf_bis_error",
    "iccf_fwhm",
    "iccf_fwhm_error",
    "iccf_rv",
    "iccf_rv_error",
    "iccf_vspan",
    "iccf_wspan",
    "iccf_contrast",
    "iccf_contrast_error",
]


def create_star_ccf_table(
    star_name: str,
    rdb_path: str | Path,
    ccf_root: str | Path,
    observation_name: str | None = None,
    sample_catalog_df: pd.DataFrame | None = None,
    catalog_path: str | Path | None = DEFAULT_CATALOG_PATH,
    stellar_parameters: dict | None = None,
    convert_to_mps: bool = False,
    columns_to_mps: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build an iCCF-enriched dataframe for one star from a local RDB file and a
    directory containing extracted CCF_A FITS files.

    Parameters
    ----------
    star_name : str
        User-facing star name, e.g. "proxima" or "HD22496".
    rdb_path : str | Path
        Path to the .rdb file.
    ccf_root : str | Path
        Directory containing *_CCF_A.fits files, recursively.
    observation_name : str | None
        Optional observation label. Defaults to the RDB filename stem.
    sample_catalog_df : pandas.DataFrame | None
        Optional already-loaded stellar catalog dataframe.
    catalog_path : str | Path | None
        Catalog path used when `sample_catalog_df` is not supplied. Pass None to
        skip loading any catalog.
    stellar_parameters : dict | None
        Optional manual stellar parameters to attach/override.
    convert_to_mps : bool
        If true, convert velocity-like iCCF columns from km/s to m/s in place.
    columns_to_mps : list[str] | None
        Which columns to convert when `convert_to_mps=True`.
    """
    rdb_path = Path(rdb_path)
    observation_name = observation_name or rdb_path.stem

    if sample_catalog_df is None and catalog_path is not None and Path(catalog_path).exists():
        sample_catalog_df = load_sample_catalog(catalog_path)

    star_df = build_prediction_table_from_local_ccf_directory(
        star_name=star_name,
        observation_name=observation_name,
        ccf_root=ccf_root,
        rdb_path=rdb_path,
        sample_catalog_df=sample_catalog_df,
        stellar_parameters=stellar_parameters,
    )

    if convert_to_mps:
        columns_to_mps = columns_to_mps or DEFAULT_COLUMNS_TO_MS
        existing_columns = [column for column in columns_to_mps if column in star_df.columns]
        star_df[existing_columns] = star_df[existing_columns] * 1000.0

    return star_df


def save_star_ccf_table(
    star_df: pd.DataFrame,
    output_base: str | Path,
    drop_array_columns_in_csv: bool = True,
) -> tuple[Path, Path]:
    """
    Save a star CCF table to pickle and CSV.
    """
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    pickle_path = output_base.with_suffix(".pkl")
    csv_path = output_base.with_suffix(".csv")

    star_df.to_pickle(pickle_path)

    csv_df = star_df.copy()
    if drop_array_columns_in_csv:
        for column in ["iccf_ccf", "iccf_rv_grid"]:
            if column in csv_df.columns:
                csv_df = csv_df.drop(columns=column)

    csv_df.to_csv(csv_path, index=False)
    return pickle_path, csv_path


def load_manual_parameters(json_path: str | Path | None = None) -> dict | None:
    if json_path is None:
        return None
    with Path(json_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
