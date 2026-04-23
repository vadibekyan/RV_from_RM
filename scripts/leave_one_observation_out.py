import argparse
import json
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
python scripts/leave_one_observation_out.py \
  --input-csv model_tables/rm_df_full.csv \
  --output-dir results/xgb_run_01 \
  --features vrad_subtract_mean fwhm_fractional_mean bis_span_fractional_mean contrast_fractional_mean \
  --target true_vrad \
  --model xgboost \
  --model-params '{"n_estimators": 500, "max_depth": 5, "learning_rate": 0.03, "subsample": 0.9, "colsample_bytree": 0.9, "objective": "reg:squarederror", "random_state": 42}'


"""


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a regression model by leaving one observation run out at a time, "
            "predict on the holdout run, and save metrics plus per-row predictions."
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
        help="Directory where metrics, predictions, and run metadata will be saved.",
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
        help="Column defining the holdout split. Default: corrected_file.",
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


def main():
    args = parse_args()
    model_params = parse_model_params(args.model_params)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)

    required_columns = [args.group_column, args.target, args.baseline_column, *args.features]
    validate_columns(df, required_columns)

    required_non_null_columns = list(dict.fromkeys([*args.features, args.target, *(args.dropna_subset or [])]))
    model_df = df.dropna(subset=required_non_null_columns).copy()

    holdout_groups = [value for value in model_df[args.group_column].dropna().unique()]
    if not holdout_groups:
        raise ValueError(f"No non-null groups found in column {args.group_column!r}.")

    metric_rows = []
    prediction_tables = []
    metadata_columns = [
        column
        for column in [args.group_column, "observation_file", "corrected_file", "rjd", args.target, args.baseline_column]
        if column in model_df.columns
    ]

    for holdout_group in holdout_groups:
        train_df = model_df.loc[model_df[args.group_column] != holdout_group].copy()
        holdout_df = model_df.loc[model_df[args.group_column] == holdout_group].copy()

        if train_df.empty or holdout_df.empty:
            continue

        model = build_model(args.model, model_params)
        model.fit(train_df[args.features], train_df[args.target])

        holdout_predictions = model.predict(holdout_df[args.features])

        prediction_df = holdout_df[metadata_columns].copy()
        prediction_df["model_name"] = args.model
        prediction_df["prediction"] = holdout_predictions
        prediction_df["model_residual"] = prediction_df[args.target] - prediction_df["prediction"]
        prediction_df["baseline_residual"] = prediction_df[args.target] - prediction_df[args.baseline_column]
        prediction_tables.append(prediction_df)

        metric_rows.append(
            {
                "holdout_group": holdout_group,
                "model_name": args.model,
                "n_train_rows": len(train_df),
                "n_holdout_rows": len(holdout_df),
                "n_train_groups": train_df[args.group_column].nunique(),
                "n_holdout_groups": holdout_df[args.group_column].nunique(),
                "target": args.target,
                "baseline_column": args.baseline_column,
                "features": json.dumps(args.features),
                "model_rmse": rmse(prediction_df[args.target], prediction_df["prediction"]),
                "model_mae": float(mean_absolute_error(prediction_df[args.target], prediction_df["prediction"])),
                "model_r2": float(r2_score(prediction_df[args.target], prediction_df["prediction"])),
                "baseline_rmse": rmse(prediction_df[args.target], prediction_df[args.baseline_column]),
                "baseline_mae": float(
                    mean_absolute_error(prediction_df[args.target], prediction_df[args.baseline_column])
                ),
            }
        )

    if not metric_rows:
        raise ValueError("No valid holdout splits were evaluated.")

    metrics_df = pd.DataFrame(metric_rows).sort_values("model_rmse").reset_index(drop=True)
    predictions_df = pd.concat(prediction_tables, ignore_index=True)

    summary = {
        "input_csv": str(args.input_csv),
        "output_dir": str(args.output_dir),
        "model": args.model,
        "model_params": model_params,
        "group_column": args.group_column,
        "target": args.target,
        "baseline_column": args.baseline_column,
        "features": args.features,
        "n_rows_after_dropna": int(len(model_df)),
        "n_holdout_groups": int(len(metric_rows)),
        "mean_model_rmse": float(metrics_df["model_rmse"].mean()),
        "median_model_rmse": float(metrics_df["model_rmse"].median()),
        "mean_baseline_rmse": float(metrics_df["baseline_rmse"].mean()),
        "median_baseline_rmse": float(metrics_df["baseline_rmse"].median()),
    }

    metrics_path = args.output_dir / "holdout_metrics.csv"
    predictions_path = args.output_dir / "holdout_predictions.csv"
    summary_path = args.output_dir / "summary.json"

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Saved metrics to {metrics_path}")
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved summary to {summary_path}")
    print()
    print(metrics_df.head().to_string(index=False))


if __name__ == "__main__":
    main()
