#!/usr/bin/env python3
from __future__ import annotations

import tarfile
from pathlib import Path

import pandas as pd

import utils


DEFAULT_DOWNLOAD_ROOT = Path("downloads/dace_ccf_a")
DEFAULT_EXTRACT_ROOT = Path("downloads/dace_ccf_a_extracted")
DEFAULT_OBSERVATION_ROOT = Path("observations/Best_RM")
DEFAULT_CORRECTED_ROOT = Path("observations/Best_RM/linear_corrected_with_uncertainties")
DEFAULT_CATALOG_PATH = Path("sample_sweetcat.csv")
DEFAULT_FILE_TYPE_SUFFIX = "_CCF_A.fits"

ICCF_SCALAR_COLUMN_MAP = {
    "BIS": "iccf_bis",
    "BISerror": "iccf_bis_error",
    "FWHM": "iccf_fwhm",
    "FWHMerror": "iccf_fwhm_error",
    "RV": "iccf_rv",
    "RVerror": "iccf_rv_error",
    "Vspan": "iccf_vspan",
    "Wspan": "iccf_wspan",
    "contrast": "iccf_contrast",
    "contrast_error": "iccf_contrast_error",
}

ICCF_TEXT_COLUMN_MAP = {
    "OBJECT": "iccf_object",
}

ICCF_ARRAY_COLUMN_MAP = {
    "ccf": "iccf_ccf",
    "rv": "iccf_rv_grid",
}

def extract_spectroscopy_archives(
    download_root: str | Path = DEFAULT_DOWNLOAD_ROOT,
    extract_root: str | Path = DEFAULT_EXTRACT_ROOT,
    overwrite: bool = False,
) -> list[Path]:
    download_root = Path(download_root)
    extract_root = Path(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    extracted_roots = []
    for archive_path in sorted(download_root.glob("*/spectroscopy_download.tar")):
        observation_name = archive_path.parent.name
        destination = extract_root / observation_name
        if overwrite and destination.exists():
            for child in sorted(destination.rglob("*"), reverse=True):
                if child.is_file() or child.is_symlink():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            if destination.exists():
                destination.rmdir()

        if not destination.exists():
            destination.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive_path) as tar:
                tar.extractall(path=destination)

        extracted_roots.append(destination)

    return extracted_roots


def ccf_fits_to_file_rootpath(fits_path: str | Path, suffix: str = DEFAULT_FILE_TYPE_SUFFIX) -> str:
    fits_name = Path(fits_path).name
    if fits_name.endswith(suffix):
        fits_name = fits_name[: -len(suffix)] + ".fits"
    if fits_name.startswith("r."):
        fits_name = fits_name[2:]
    return fits_name


def find_ccf_fits_files(
    extracted_observation_root: str | Path,
    suffix: str = DEFAULT_FILE_TYPE_SUFFIX,
) -> list[Path]:
    extracted_observation_root = Path(extracted_observation_root)
    return sorted(extracted_observation_root.rglob(f"*{suffix}"))


def add_stellar_parameters_to_table(
    df: pd.DataFrame,
    star_name: str,
    sample_catalog_df: pd.DataFrame | None = None,
    stellar_parameters: dict | None = None,
) -> pd.DataFrame:
    result_df = df.copy()

    if sample_catalog_df is not None and not sample_catalog_df.empty:
        catalog_with_keys = sample_catalog_df.assign(
            star_key=sample_catalog_df["star"].map(utils.normalize_star_name)
        )
        star_key = utils.normalize_star_name(star_name)
        result_df["star_key"] = star_key
        result_df = result_df.merge(catalog_with_keys, on="star_key", how="left").drop(columns="star_key")

    if stellar_parameters:
        for column, value in stellar_parameters.items():
            result_df[column] = value

    if "star" not in result_df.columns or result_df["star"].isna().all():
        result_df["star"] = star_name

    return result_df


def build_iccf_table_for_observation(
    observation_name: str,
    extracted_root: str | Path = DEFAULT_EXTRACT_ROOT,
    relevant_columns: list[str] | None = None,
) -> pd.DataFrame:
    import iCCF

    relevant_columns = relevant_columns or [
        "BIS",
        "BISerror",
        "FWHM",
        "FWHMerror",
        "OBJECT",
        "RV",
        "RVerror",
        "Vspan",
        "Wspan",
        "ccf",
        "rv",
        'contrast',
        'contrast_error',
    ]

    observation_stem = Path(observation_name).stem
    extracted_observation_root = Path(extracted_root) / observation_stem
    fits_files = find_ccf_fits_files(extracted_observation_root)
    if not fits_files:
        raise FileNotFoundError(f"No CCF FITS files found under {extracted_observation_root}")

    rows = []
    for fits_path in fits_files:
        indicators = iCCF.from_file(str(fits_path))
        row = {
            "observation_name": observation_name,
            "ccf_fits_path": str(fits_path),
            "ccf_file_name": fits_path.name,
            "file_rootpath": ccf_fits_to_file_rootpath(fits_path),
        }

        for source_name in relevant_columns:
            if source_name in ICCF_SCALAR_COLUMN_MAP:
                value = getattr(indicators, source_name)
                row[ICCF_SCALAR_COLUMN_MAP[source_name]] = float(value)
            elif source_name in ICCF_TEXT_COLUMN_MAP:
                row[ICCF_TEXT_COLUMN_MAP[source_name]] = str(getattr(indicators, source_name))
            elif source_name in ICCF_ARRAY_COLUMN_MAP:
                row[ICCF_ARRAY_COLUMN_MAP[source_name]] = getattr(indicators, source_name)
            else:
                row[f"iccf_{source_name.lower()}"] = getattr(indicators, source_name)

        rows.append(row)

    return pd.DataFrame(rows).sort_values("file_rootpath").reset_index(drop=True)


def build_iccf_table_from_directory(
    ccf_root: str | Path,
    observation_name: str,
    relevant_columns: list[str] | None = None,
) -> pd.DataFrame:
    import iCCF

    relevant_columns = relevant_columns or [
        "BIS",
        "BISerror",
        "FWHM",
        "FWHMerror",
        "OBJECT",
        "RV",
        "RVerror",
        "Vspan",
        "Wspan",
        "ccf",
        "rv",
        "contrast",
        "contrast_error",
    ]

    fits_files = find_ccf_fits_files(ccf_root)
    if not fits_files:
        raise FileNotFoundError(f"No CCF FITS files found under {ccf_root}")

    rows = []
    for fits_path in fits_files:
        indicators = iCCF.from_file(str(fits_path))
        row = {
            "observation_name": observation_name,
            "ccf_fits_path": str(fits_path),
            "ccf_file_name": fits_path.name,
            "file_rootpath": ccf_fits_to_file_rootpath(fits_path),
        }

        for source_name in relevant_columns:
            if source_name in ICCF_SCALAR_COLUMN_MAP:
                row[ICCF_SCALAR_COLUMN_MAP[source_name]] = float(getattr(indicators, source_name))
            elif source_name in ICCF_TEXT_COLUMN_MAP:
                row[ICCF_TEXT_COLUMN_MAP[source_name]] = str(getattr(indicators, source_name))
            elif source_name in ICCF_ARRAY_COLUMN_MAP:
                row[ICCF_ARRAY_COLUMN_MAP[source_name]] = getattr(indicators, source_name)
            else:
                row[f"iccf_{source_name.lower()}"] = getattr(indicators, source_name)

        rows.append(row)

    return pd.DataFrame(rows).sort_values("file_rootpath").reset_index(drop=True)


def load_observation_reference_table(
    observation_name: str,
    observation_root: str | Path = DEFAULT_OBSERVATION_ROOT,
    corrected_root: str | Path = DEFAULT_CORRECTED_ROOT,
) -> pd.DataFrame:
    observation_root = Path(observation_root)
    corrected_root = Path(corrected_root)

    observation_name = Path(observation_name).name
    rdb_path = observation_root / observation_name
    if not rdb_path.exists():
        raise FileNotFoundError(f"Observation RDB not found: {rdb_path}")

    corrected_name = observation_name.replace(".rdb", "_linear_corrected.csv")
    corrected_path = corrected_root / corrected_name

    rdb_df = pd.read_csv(rdb_path, sep="\t", skiprows=[1]).copy()
    rdb_df["observation_file"] = observation_name

    if corrected_path.exists():
        corrected_df = pd.read_csv(corrected_path).copy()
        keep_columns = ["file_rootpath", "true_vrad1", "true_vrad2"]
        corrected_subset = corrected_df[keep_columns]
        reference_df = rdb_df.merge(corrected_subset, on="file_rootpath", how="left")
    else:
        reference_df = rdb_df.copy()
        reference_df["true_vrad1"] = pd.NA
        reference_df["true_vrad2"] = pd.NA

    return reference_df


def build_prediction_table_from_local_ccf_directory(
    star_name: str,
    observation_name: str,
    ccf_root: str | Path,
    rdb_path: str | Path,
    sample_catalog_df: pd.DataFrame | None = None,
    stellar_parameters: dict | None = None,
    relevant_columns: list[str] | None = None,
) -> pd.DataFrame:
    rdb_path = Path(rdb_path)
    observation_file = rdb_path.name
    reference_df = pd.read_csv(rdb_path, sep="\t", skiprows=[1]).copy()
    reference_df["observation_file"] = observation_file
    reference_df["observation_name"] = observation_name

    iccf_df = build_iccf_table_from_directory(
        ccf_root=ccf_root,
        observation_name=observation_name,
        relevant_columns=relevant_columns,
    )

    merged_df = reference_df.merge(
        iccf_df.drop(columns=["observation_name"], errors="ignore"),
        on="file_rootpath",
        how="left",
        validate="one_to_one",
    )
    merged_df.insert(0, "star_from_file", star_name)
    merged_df = add_stellar_parameters_to_table(
        merged_df,
        star_name=star_name,
        sample_catalog_df=sample_catalog_df,
        stellar_parameters=stellar_parameters,
    )

    return merged_df.sort_values("rjd").reset_index(drop=True)


def build_merged_ccf_table_for_observation(
    observation_name: str,
    extracted_root: str | Path = DEFAULT_EXTRACT_ROOT,
    observation_root: str | Path = DEFAULT_OBSERVATION_ROOT,
    corrected_root: str | Path = DEFAULT_CORRECTED_ROOT,
    sample_catalog_df: pd.DataFrame | None = None,
    relevant_columns: list[str] | None = None,
) -> pd.DataFrame:
    observation_name = Path(observation_name).name

    iccf_df = build_iccf_table_for_observation(
        observation_name=observation_name,
        extracted_root=extracted_root,
        relevant_columns=relevant_columns,
    )
    reference_df = load_observation_reference_table(
        observation_name=observation_name,
        observation_root=observation_root,
        corrected_root=corrected_root,
    )

    merged_df = reference_df.merge(
        iccf_df,
        on="file_rootpath",
        how="left",
        validate="one_to_one",
    )

    if sample_catalog_df is not None:
        catalog_with_keys = sample_catalog_df.assign(
            star_key=sample_catalog_df["star"].map(utils.normalize_star_name)
        )
        merged_df["star_key"] = utils.normalize_star_name(
            utils.star_name_from_corrected_path(observation_name.replace(".rdb", "_linear_corrected.csv"))
        )
        merged_df = merged_df.merge(
            catalog_with_keys,
            on="star_key",
            how="left",
        ).drop(columns="star_key")

    return merged_df.sort_values("rjd").reset_index(drop=True)


def build_merged_ccf_table_for_downloads(
    download_root: str | Path = DEFAULT_DOWNLOAD_ROOT,
    extract_root: str | Path = DEFAULT_EXTRACT_ROOT,
    observation_root: str | Path = DEFAULT_OBSERVATION_ROOT,
    corrected_root: str | Path = DEFAULT_CORRECTED_ROOT,
    sample_catalog_df: pd.DataFrame | None = None,
    relevant_columns: list[str] | None = None,
    extract_archives: bool = True,
    overwrite_extract: bool = False,
) -> pd.DataFrame:
    if extract_archives:
        extract_spectroscopy_archives(
            download_root=download_root,
            extract_root=extract_root,
            overwrite=overwrite_extract,
        )

    observation_names = sorted(path.name for path in Path(download_root).glob("*") if path.is_dir())
    tables = []
    for observation_name in observation_names:
        if observation_name == "spectroscopy_download":
            continue
        rdb_name = f"{observation_name}.rdb"
        try:
            merged_df = build_merged_ccf_table_for_observation(
                observation_name=rdb_name,
                extracted_root=extract_root,
                observation_root=observation_root,
                corrected_root=corrected_root,
                sample_catalog_df=sample_catalog_df,
                relevant_columns=relevant_columns,
            )
        except FileNotFoundError:
            continue

        if "observation_name" not in merged_df.columns:
            merged_df.insert(0, "observation_name", observation_name)
        if "star_from_file" not in merged_df.columns:
            merged_df.insert(
                0,
                "star_from_file",
                utils.star_name_from_corrected_path(f"{observation_name}_linear_corrected.csv"),
            )
        tables.append(merged_df)

    if not tables:
        return pd.DataFrame()

    return pd.concat(tables, ignore_index=True)


def load_sample_catalog(catalog_path: str | Path = DEFAULT_CATALOG_PATH) -> pd.DataFrame:
    return pd.read_csv(catalog_path)


def save_merged_ccf_outputs(
    merged_df: pd.DataFrame,
    output_base: str | Path,
) -> tuple[Path, Path]:
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    pickle_path = output_base.with_suffix(".pkl")
    csv_path = output_base.with_suffix(".csv")

    merged_df.to_pickle(pickle_path)

    csv_df = merged_df.copy()
    csv_df.to_csv(csv_path, index=False)
    return pickle_path, csv_path
