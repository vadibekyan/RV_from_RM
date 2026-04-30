#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply precomputed linear RV correction parameters to RM observation "
            "RDB files and save corrected CSV files while preserving uncertainty columns."
        )
    )
    parser.add_argument(
        "--parameters-csv",
        type=Path,
        default=Path("linear_model_parameters.csv"),
        help="CSV containing linear fit parameters such as rjd_ref, intercept, and slope.",
    )
    parser.add_argument(
        "--observations-dir",
        type=Path,
        default=Path("observations/Best_RM"),
        help="Directory containing the source .rdb observation files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("observations/Best_RM/linear_corrected_with_uncertainties"),
        help="Directory where corrected CSV files will be written.",
    )
    parser.add_argument(
        "--round-decimals",
        type=int,
        default=2,
        help="Number of decimals used for the mean-centered true_vrad column.",
    )
    parser.add_argument(
        "--copy-parameters-csv",
        action="store_true",
        help="Also copy the parameter table into the output directory.",
    )
    return parser.parse_args()


def read_rdb_file(file_path: Path) -> pd.DataFrame:
    rdb_df = pd.read_csv(file_path, sep="\t", skiprows=[1])

    for column in ["rjd", "vrad", "svrad"]:
        if column in rdb_df.columns:
            rdb_df[column] = pd.to_numeric(rdb_df[column], errors="coerce")

    return rdb_df


def build_linear_model(
    obs_df: pd.DataFrame,
    rjd_ref: float,
    intercept: float,
    slope: float,
 ) -> pd.Series:
    return intercept + slope * (obs_df["rjd"] - rjd_ref)


def build_true_vrad_variants(
    linear_model: pd.Series,
    vrad: pd.Series,
    decimals: int = 2,
) -> tuple[pd.Series, pd.Series]:
    true_vrad1 = np.round(linear_model - linear_model.mean(), decimals=decimals)
    true_vrad2 = np.round(linear_model - vrad.mean(), decimals=decimals)
    return true_vrad1, true_vrad2


def insert_true_vrad_columns(
    obs_df: pd.DataFrame,
    true_vrad1: pd.Series,
    true_vrad2: pd.Series,
) -> pd.DataFrame:
    corrected_df = obs_df.copy()

    for column in ["true_vrad", "true_vrad1", "true_vrad2"]:
        if column in corrected_df.columns:
            corrected_df = corrected_df.drop(columns=[column])

    insert_at = corrected_df.columns.get_loc("vrad") + 1 if "vrad" in corrected_df.columns else 2
    corrected_df.insert(insert_at, "true_vrad1", true_vrad1)
    corrected_df.insert(insert_at + 1, "true_vrad2", true_vrad2)
    return corrected_df


def corrected_filename_for_row(row: pd.Series) -> str:
    corrected_name = row.get("corrected_file")
    if pd.notna(corrected_name):
        return str(corrected_name)
    return f"{Path(row['observation_file']).stem}_linear_corrected.csv"


def apply_linear_corrections(
    parameters_csv: Path,
    observations_dir: Path,
    output_dir: Path,
    round_decimals: int = 2,
    copy_parameters_csv: bool = False,
) -> pd.DataFrame:
    params_df = pd.read_csv(parameters_csv)
    required_columns = {"observation_file", "rjd_ref", "intercept", "slope"}
    missing_columns = required_columns - set(params_df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns in {parameters_csv}: {missing_text}")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []

    for _, row in params_df.iterrows():
        observation_file = str(row["observation_file"])
        obs_path = observations_dir / observation_file
        if not obs_path.exists():
            raise FileNotFoundError(f"Observation file not found: {obs_path}")

        obs_df = read_rdb_file(obs_path)
        if "rjd" not in obs_df.columns:
            raise ValueError(f"'rjd' column is missing in {obs_path}")

        linear_model = build_linear_model(
            obs_df,
            rjd_ref=row["rjd_ref"],
            intercept=row["intercept"],
            slope=row["slope"],
        )
        true_vrad1, true_vrad2 = build_true_vrad_variants(
            linear_model,
            obs_df["vrad"],
            decimals=round_decimals,
        )
        corrected_df = insert_true_vrad_columns(obs_df, true_vrad1, true_vrad2)

        corrected_file = corrected_filename_for_row(row)
        corrected_path = output_dir / corrected_file
        corrected_df.to_csv(corrected_path, index=False)

        summary_rows.append(
            {
                "observation_file": observation_file,
                "corrected_file": corrected_file,
                "n_rows": len(corrected_df),
                "true_vrad1_mean": float(corrected_df["true_vrad1"].mean()),
                "true_vrad2_mean": float(corrected_df["true_vrad2"].mean()),
                "output_path": str(corrected_path),
            }
        )

    if copy_parameters_csv:
        params_df.to_csv(output_dir / parameters_csv.name, index=False)

    return pd.DataFrame(summary_rows)


def main() -> None:
    args = parse_args()
    summary_df = apply_linear_corrections(
        parameters_csv=args.parameters_csv,
        observations_dir=args.observations_dir,
        output_dir=args.output_dir,
        round_decimals=args.round_decimals,
        copy_parameters_csv=args.copy_parameters_csv,
    )

    if summary_df.empty:
        print("No rows found in parameter CSV.")
        return

    print(f"Wrote {len(summary_df)} corrected files to {args.output_dir}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
