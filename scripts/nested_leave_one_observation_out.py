import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

"""
Nested leave-one-observation-out workflow:
1) Baseline LOO on full dataset.
2) For each observation/group removed entirely, rerun LOO on the remaining data.

Example:
python scripts/nested_leave_one_observation_out.py \
  --input-csv model_tables/rm_df_full.csv \
  --output-dir results/xgb_nested_loo_run_01 \
  --features vrad_subtract_mean fwhm_fractional_mean bis_span_fractional_mean contrast_fractional_mean \
  --target true_vrad \
  --model xgboost \
  --model-params '{"n_estimators": 500, "max_depth": 5, "learning_rate": 0.03, "subsample": 0.9, "colsample_bytree": 0.9, "objective": "reg:squarederror", "random_state": 42}'
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run baseline leave-one-observation-out (LOO), then repeat LOO after "
            "removing each observation/group once, and save comparison tables."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Path to the training table, e.g. model_tables/rm_df_full.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where nested-LOO metrics, predictions, and summaries will be saved.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        required=True,
        help="Feature columns used to train the model.",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Target column to predict, e.g. true_vrad.",
    )
    parser.add_argument(
        "--group-column",
        default="corrected_file",
        help="Column defining observation groups/holdouts. Default: corrected_file.",
    )
    parser.add_argument(
        "--baseline-column",
        default="vrad",
        help="Baseline column used for true-baseline residual comparison. Default: vrad.",
    )
    parser.add_argument(
        "--model",
        default="random_forest",
        choices=["linear_regression", "ridge", "elastic_net", "random_forest", "extra_trees", "xgboost"],
        help="Regression model to train for each holdout split.",
    )
    parser.add_argument(
        "--model-params",
        default="{}",
        help=(
            "JSON string with model hyperparameters, "
            "for example '{\"n_estimators\": 500, \"max_depth\": 6, \"random_state\": 42}'."
        ),
    )
    parser.add_argument(
        "--dropna-subset",
        nargs="*",
        default=None,
        help=(
            "Optional extra columns that must be non-null before splitting. "
            "Features and target are always required."
        ),
    )
    parser.add_argument(
        "--min-groups-after-removal",
        type=int,
        default=2,
        help=(
            "Minimum number of non-null groups required to run inner LOO after removing one group. "
            "Default: 2."
        ),
    )
    return parser.parse_args()


def parse_model_params(raw_value):
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse --model-params as JSON: {raw_value}") from exc


def rmse(y_true, y_pred):
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(np.square(residuals))))


def build_model(model_name, model_params):
    model_params = dict(model_params)

    if model_name == "linear_regression":
        estimator = LinearRegression(**model_params)
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )

    if model_name == "ridge":
        estimator = Ridge(**model_params)
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )

    if model_name == "elastic_net":
        estimator = ElasticNet(**model_params)
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )

    if model_name == "random_forest":
        estimator = RandomForestRegressor(**model_params)
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", estimator),
            ]
        )

    if model_name == "extra_trees":
        estimator = ExtraTreesRegressor(**model_params)
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", estimator),
            ]
        )

    if model_name == "xgboost":
        estimator = XGBRegressor(**model_params)
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", estimator),
            ]
        )

    raise ValueError(f"Unsupported model: {model_name}")


def validate_columns(df, required_columns):
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")


def safe_group_slug(group_value):
    text = str(group_value)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return slug[:80] or "group"


def run_loo(
    df,
    group_column,
    features,
    target,
    baseline_column,
    model_name,
    model_params,
    metadata_columns,
):
    holdout_groups = [value for value in df[group_column].dropna().unique()]
    metric_rows = []
    prediction_tables = []

    for holdout_group in holdout_groups:
        train_df = df.loc[df[group_column] != holdout_group].copy()
        holdout_df = df.loc[df[group_column] == holdout_group].copy()

        if train_df.empty or holdout_df.empty:
            continue

        model = build_model(model_name, model_params)
        model.fit(train_df[features], train_df[target])
        holdout_predictions = model.predict(holdout_df[features])

        prediction_df = holdout_df[metadata_columns].copy()
        prediction_df["model_name"] = model_name
        prediction_df["inner_holdout_group"] = holdout_group
        prediction_df["prediction"] = holdout_predictions
        prediction_df["model_residual"] = prediction_df[target] - prediction_df["prediction"]
        prediction_df["baseline_residual"] = prediction_df[target] - prediction_df[baseline_column]
        prediction_tables.append(prediction_df)

        metric_rows.append(
            {
                "inner_holdout_group": holdout_group,
                "model_name": model_name,
                "n_train_rows": len(train_df),
                "n_holdout_rows": len(holdout_df),
                "n_train_groups": train_df[group_column].nunique(),
                "n_holdout_groups": holdout_df[group_column].nunique(),
                "target": target,
                "baseline_column": baseline_column,
                "features": json.dumps(features),
                "model_rmse": rmse(prediction_df[target], prediction_df["prediction"]),
                "model_mae": float(mean_absolute_error(prediction_df[target], prediction_df["prediction"])),
                "model_r2": float(r2_score(prediction_df[target], prediction_df["prediction"])),
                "baseline_rmse": rmse(prediction_df[target], prediction_df[baseline_column]),
                "baseline_mae": float(mean_absolute_error(prediction_df[target], prediction_df[baseline_column])),
            }
        )

    if not metric_rows:
        raise ValueError("No valid holdout splits were evaluated in run_loo().")

    metrics_df = pd.DataFrame(metric_rows).sort_values("model_rmse").reset_index(drop=True)
    predictions_df = pd.concat(prediction_tables, ignore_index=True)
    summary = {
        "n_holdout_groups_evaluated": int(len(metrics_df)),
        "mean_model_rmse": float(metrics_df["model_rmse"].mean()),
        "median_model_rmse": float(metrics_df["model_rmse"].median()),
        "mean_baseline_rmse": float(metrics_df["baseline_rmse"].mean()),
        "median_baseline_rmse": float(metrics_df["baseline_rmse"].median()),
    }
    return metrics_df, predictions_df, summary


def save_loo_outputs(output_dir, metrics_df, predictions_df, summary):
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "holdout_metrics.csv"
    predictions_path = output_dir / "holdout_predictions.csv"
    summary_path = output_dir / "summary.json"

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2))


def main():
    args = parse_args()
    model_params = parse_model_params(args.model_params)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input_csv)

    required_columns = [args.group_column, args.target, args.baseline_column, *args.features]
    validate_columns(df, required_columns)

    required_non_null_columns = list(dict.fromkeys([*args.features, args.target, *(args.dropna_subset or [])]))
    model_df = df.dropna(subset=required_non_null_columns).copy()

    all_groups = [value for value in model_df[args.group_column].dropna().unique()]
    if len(all_groups) < args.min_groups_after_removal:
        raise ValueError(
            f"Need at least {args.min_groups_after_removal} non-null groups in {args.group_column!r}; "
            f"found {len(all_groups)}."
        )

    metadata_columns = [
        column
        for column in [args.group_column, "observation_file", "corrected_file", "rjd", args.target, args.baseline_column]
        if column in model_df.columns
    ]

    run_config = {
        "input_csv": str(args.input_csv),
        "output_dir": str(args.output_dir),
        "group_column": args.group_column,
        "target": args.target,
        "baseline_column": args.baseline_column,
        "model": args.model,
        "model_params": model_params,
        "features": args.features,
        "n_rows_after_dropna": int(len(model_df)),
        "n_groups_after_dropna": int(len(all_groups)),
        "min_groups_after_removal": int(args.min_groups_after_removal),
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    baseline_dir = args.output_dir / "baseline"
    baseline_metrics_df, baseline_predictions_df, baseline_summary = run_loo(
        df=model_df,
        group_column=args.group_column,
        features=args.features,
        target=args.target,
        baseline_column=args.baseline_column,
        model_name=args.model,
        model_params=model_params,
        metadata_columns=metadata_columns,
    )
    baseline_summary["outer_removed_group"] = None
    save_loo_outputs(baseline_dir, baseline_metrics_df, baseline_predictions_df, baseline_summary)

    baseline_mean_rmse = float(baseline_summary["mean_model_rmse"])
    baseline_median_rmse = float(baseline_summary["median_model_rmse"])

    nested_dir = args.output_dir / "outer_removed"
    nested_dir.mkdir(parents=True, exist_ok=True)

    influence_rows = []
    all_inner_metrics = []
    all_inner_predictions = []

    for index, removed_group in enumerate(all_groups, start=1):
        reduced_df = model_df.loc[model_df[args.group_column] != removed_group].copy()
        remaining_groups = [value for value in reduced_df[args.group_column].dropna().unique()]

        if len(remaining_groups) < args.min_groups_after_removal:
            continue

        inner_metrics_df, inner_predictions_df, inner_summary = run_loo(
            df=reduced_df,
            group_column=args.group_column,
            features=args.features,
            target=args.target,
            baseline_column=args.baseline_column,
            model_name=args.model,
            model_params=model_params,
            metadata_columns=metadata_columns,
        )

        safe_slug = safe_group_slug(removed_group)
        case_dir = nested_dir / f"{index:03d}_{safe_slug}"

        inner_summary["outer_removed_group"] = removed_group
        inner_summary["n_rows_after_outer_removal"] = int(len(reduced_df))
        inner_summary["n_groups_after_outer_removal"] = int(len(remaining_groups))
        save_loo_outputs(case_dir, inner_metrics_df, inner_predictions_df, inner_summary)

        inner_metrics_tagged = inner_metrics_df.copy()
        inner_metrics_tagged["outer_removed_group"] = removed_group
        inner_predictions_tagged = inner_predictions_df.copy()
        inner_predictions_tagged["outer_removed_group"] = removed_group

        all_inner_metrics.append(inner_metrics_tagged)
        all_inner_predictions.append(inner_predictions_tagged)

        mean_rmse = float(inner_summary["mean_model_rmse"])
        median_rmse = float(inner_summary["median_model_rmse"])
        influence_rows.append(
            {
                "outer_removed_group": removed_group,
                "n_rows_after_outer_removal": int(len(reduced_df)),
                "n_groups_after_outer_removal": int(len(remaining_groups)),
                "inner_loo_mean_model_rmse": mean_rmse,
                "inner_loo_median_model_rmse": median_rmse,
                "delta_mean_rmse_vs_baseline": mean_rmse - baseline_mean_rmse,
                "delta_median_rmse_vs_baseline": median_rmse - baseline_median_rmse,
                "improves_vs_baseline_mean": bool(mean_rmse < baseline_mean_rmse),
            }
        )

    if influence_rows:
        influence_df = pd.DataFrame(influence_rows).sort_values("delta_mean_rmse_vs_baseline").reset_index(drop=True)
    else:
        influence_df = pd.DataFrame(
            columns=[
                "outer_removed_group",
                "n_rows_after_outer_removal",
                "n_groups_after_outer_removal",
                "inner_loo_mean_model_rmse",
                "inner_loo_median_model_rmse",
                "delta_mean_rmse_vs_baseline",
                "delta_median_rmse_vs_baseline",
                "improves_vs_baseline_mean",
            ]
        )

    influence_df.to_csv(args.output_dir / "influence_summary.csv", index=False)

    if all_inner_metrics:
        pd.concat(all_inner_metrics, ignore_index=True).to_csv(args.output_dir / "all_inner_holdout_metrics.csv", index=False)
    if all_inner_predictions:
        pd.concat(all_inner_predictions, ignore_index=True).to_csv(
            args.output_dir / "all_inner_holdout_predictions.csv", index=False
        )

    final_summary = {
        "baseline_mean_model_rmse": baseline_mean_rmse,
        "baseline_median_model_rmse": baseline_median_rmse,
        "n_outer_removed_cases_evaluated": int(len(influence_df)),
        "best_outer_removed_group_by_mean_rmse": (
            None if influence_df.empty else influence_df.iloc[0]["outer_removed_group"]
        ),
        "best_delta_mean_rmse_vs_baseline": (
            None if influence_df.empty else float(influence_df.iloc[0]["delta_mean_rmse_vs_baseline"])
        ),
    }
    (args.output_dir / "nested_summary.json").write_text(json.dumps(final_summary, indent=2))

    print(f"Saved baseline LOO in: {baseline_dir}")
    print(f"Saved outer-removal cases in: {nested_dir}")
    print(f"Saved influence summary: {args.output_dir / 'influence_summary.csv'}")
    print(f"Saved nested summary: {args.output_dir / 'nested_summary.json'}")
    if not influence_df.empty:
        print()
        print("Top 5 outer removals by mean RMSE improvement:")
        print(influence_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
