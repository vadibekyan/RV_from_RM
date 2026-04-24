import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

"""
Example:
python scripts/tune_xgb_feature_sets_single_holdout.py \
  --input-csv model_tables/rm_df_full.csv \
  --output-dir results/xgb_single_holdout_hd189733_esp19_3 \
  --target true_vrad \
  --holdout-value HD189733_esp19_3.rdb \
  --holdout-column observation_file \
  --feature-groups-json config/rm_feature_groups_staged_v1.json \
  --n-trials 200 \
  --base-model-params '{"objective":"reg:squarederror","random_state":42,"tree_method":"hist"}'
"""


DEFAULT_OPTUNA_SPACE = {
    "n_estimators": {"type": "int", "low": 200, "high": 2000, "step": 20},
    "max_depth": {"type": "int", "low": 2, "high": 10},
    "learning_rate": {"type": "float", "low": 0.001, "high": 0.3, "log": True},
    "subsample": {"type": "float", "low": 0.5, "high": 1.0},
    "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
    "min_child_weight": {"type": "int", "low": 1, "high": 10},
    "gamma": {"type": "float", "low": 1.0e-8, "high": 10.0, "log": True},
    "reg_alpha": {"type": "float", "low": 1.0e-8, "high": 10.0, "log": True},
    "reg_lambda": {"type": "float", "low": 1.0e-3, "high": 20.0, "log": True},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Tune XGBoost hyperparameters for one or more feature sets using a single fixed held-out "
            "observation/group, and save per-trial and best-model performance on that holdout."
        )
    )
    parser.add_argument("--input-csv", type=Path, required=True, help="Path to the training table.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where outputs will be saved.")
    parser.add_argument("--target", required=True, help="Target column to predict, e.g. true_vrad.")
    parser.add_argument(
        "--holdout-column",
        default="observation_file",
        help="Column used to identify the held-out observation/group. Default: observation_file.",
    )
    parser.add_argument(
        "--holdout-value",
        required=True,
        help="Exact value in --holdout-column to keep fully held out, e.g. HD189733_esp19_3.rdb",
    )
    parser.add_argument(
        "--baseline-column",
        default="vrad",
        help="Optional baseline column for comparison on the held-out observation. Default: vrad.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        help="Optional single unnamed feature set. Saved as feature_group='features_cli'.",
    )
    parser.add_argument(
        "--feature-group",
        action="append",
        default=None,
        help="Named feature set in the form name=feature1,feature2,feature3. Can be repeated.",
    )
    parser.add_argument(
        "--feature-groups-json",
        type=Path,
        default=None,
        help="Path to a JSON file mapping feature-group names to lists of feature names.",
    )
    parser.add_argument(
        "--dropna-subset",
        nargs="*",
        default=None,
        help=(
            "Optional extra columns that must be non-null. To keep comparison identical across feature sets, "
            "rows are filtered using the union of all requested features plus target."
        ),
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=200,
        help="Number of Optuna trials per feature group. Default: 200.",
    )
    parser.add_argument(
        "--optuna-space-json",
        default=None,
        help="JSON string describing the Optuna search space. If omitted, a built-in space is used.",
    )
    parser.add_argument(
        "--optuna-space-file",
        type=Path,
        default=None,
        help="Path to a JSON file describing the Optuna search space.",
    )
    parser.add_argument(
        "--base-model-params",
        default='{"objective": "reg:squarederror", "random_state": 42}',
        help="JSON string with fixed XGBoost parameters added to every Optuna trial.",
    )
    return parser.parse_args()


def parse_json_string(raw_value, label):
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse {label} as JSON: {raw_value}") from exc


def load_json_file(path):
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse JSON file: {path}") from exc


def parse_cli_feature_group(raw_value):
    if "=" not in raw_value:
        raise ValueError(
            f"Invalid --feature-group value {raw_value!r}. Expected name=feature1,feature2,..."
        )
    name, raw_features = raw_value.split("=", 1)
    features = [feature.strip() for feature in raw_features.split(",") if feature.strip()]
    if not name.strip():
        raise ValueError(f"Feature-group name cannot be empty in {raw_value!r}.")
    if not features:
        raise ValueError(f"Feature group {name!r} does not contain any features.")
    return name.strip(), features


def get_feature_groups(args):
    feature_groups = {}

    if args.features:
        feature_groups["features_cli"] = list(args.features)

    for raw_group in args.feature_group or []:
        name, features = parse_cli_feature_group(raw_group)
        feature_groups[name] = features

    if args.feature_groups_json:
        file_groups = load_json_file(args.feature_groups_json)
        if not isinstance(file_groups, dict):
            raise ValueError("--feature-groups-json must contain a JSON object of name -> feature list.")
        for name, features in file_groups.items():
            if not isinstance(features, list) or not features:
                raise ValueError(f"Feature group {name!r} in {args.feature_groups_json} must be a non-empty list.")
            feature_groups[str(name)] = [str(feature) for feature in features]

    if not feature_groups:
        raise ValueError(
            "Provide at least one feature set using --features, --feature-group, or --feature-groups-json."
        )

    return feature_groups


def get_optuna_space(args):
    if args.optuna_space_json and args.optuna_space_file:
        raise ValueError("Use only one of --optuna-space-json or --optuna-space-file.")

    if args.optuna_space_json:
        optuna_space = parse_json_string(args.optuna_space_json, "--optuna-space-json")
    elif args.optuna_space_file:
        optuna_space = load_json_file(args.optuna_space_file)
    else:
        optuna_space = DEFAULT_OPTUNA_SPACE

    if not isinstance(optuna_space, dict) or not optuna_space:
        raise ValueError("Optuna search space must be a non-empty JSON object.")

    for key, spec in optuna_space.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Optuna parameter {key!r} must map to an object.")
        if spec.get("type") not in {"int", "float", "categorical"}:
            raise ValueError(f"Optuna parameter {key!r} has unsupported type {spec.get('type')!r}.")

    return optuna_space


def validate_columns(df, required_columns):
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")


def rmse(y_true, y_pred):
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(np.square(residuals))))


def build_model(model_params):
    estimator = XGBRegressor(**model_params)
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ]
    )


def suggest_optuna_params(trial, optuna_space):
    params = {}

    for name, spec in optuna_space.items():
        spec_type = spec["type"]

        if spec_type == "int":
            suggest_kwargs = {"low": int(spec["low"]), "high": int(spec["high"])}
            if "step" in spec:
                suggest_kwargs["step"] = int(spec["step"])
            if "log" in spec:
                suggest_kwargs["log"] = bool(spec["log"])
            params[name] = trial.suggest_int(name, **suggest_kwargs)
            continue

        if spec_type == "float":
            suggest_kwargs = {"low": float(spec["low"]), "high": float(spec["high"])}
            if "step" in spec:
                suggest_kwargs["step"] = float(spec["step"])
            if "log" in spec:
                suggest_kwargs["log"] = bool(spec["log"])
            params[name] = trial.suggest_float(name, **suggest_kwargs)
            continue

        if spec_type == "categorical":
            choices = spec.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ValueError(f"Categorical Optuna parameter {name!r} needs a non-empty 'choices' list.")
            params[name] = trial.suggest_categorical(name, choices)
            continue

    return params


def evaluate_model(train_df, holdout_df, features, target, baseline_column, model_params):
    model = build_model(model_params)
    model.fit(train_df[features], train_df[target])
    predictions = model.predict(holdout_df[features])

    prediction_df = holdout_df.copy()
    prediction_df["prediction"] = predictions
    prediction_df["model_residual"] = prediction_df[target] - prediction_df["prediction"]
    if baseline_column in prediction_df.columns:
        prediction_df["baseline_residual"] = prediction_df[target] - prediction_df[baseline_column]

    metrics = {
        "holdout_rmse": rmse(prediction_df[target], prediction_df["prediction"]),
        "holdout_mae": float(mean_absolute_error(prediction_df[target], prediction_df["prediction"])),
        "holdout_r2": float(r2_score(prediction_df[target], prediction_df["prediction"])),
    }

    if baseline_column in prediction_df.columns:
        metrics["baseline_rmse"] = rmse(prediction_df[target], prediction_df[baseline_column])
        metrics["baseline_mae"] = float(mean_absolute_error(prediction_df[target], prediction_df[baseline_column]))

    return metrics, prediction_df


def main():
    args = parse_args()
    feature_groups = get_feature_groups(args)
    optuna_space = get_optuna_space(args)
    base_model_params = parse_json_string(args.base_model_params, "--base-model-params")
    random_state = int(base_model_params.get("random_state", 42))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)

    all_features = []
    for features in feature_groups.values():
        all_features.extend(features)

    required_columns = [
        args.holdout_column,
        args.target,
        *all_features,
        *(args.dropna_subset or []),
    ]
    if args.baseline_column:
        required_columns.append(args.baseline_column)
    validate_columns(df, required_columns)

    required_non_null_columns = list(dict.fromkeys([args.target, *all_features, *(args.dropna_subset or [])]))
    model_df = df.dropna(subset=required_non_null_columns).copy()

    holdout_mask = model_df[args.holdout_column].astype(str) == str(args.holdout_value)
    holdout_df_full = model_df.loc[holdout_mask].copy()
    train_df_full = model_df.loc[~holdout_mask].copy()

    if holdout_df_full.empty:
        available_values = sorted(model_df[args.holdout_column].dropna().astype(str).unique())
        raise ValueError(
            f"No rows found for holdout value {args.holdout_value!r} in column {args.holdout_column!r}. "
            f"Available values include: {available_values[:10]}"
        )
    if train_df_full.empty:
        raise ValueError("Training set is empty after removing the holdout observation.")

    best_rows = []
    all_trial_tables = []
    all_best_prediction_tables = []
    optuna_best_trials = []

    metadata_columns = list(
        dict.fromkeys(
            column
            for column in [
                args.holdout_column,
                "observation_file",
                "corrected_file",
                "rjd",
                args.target,
                args.baseline_column,
            ]
            if column in holdout_df_full.columns
        )
    )

    for feature_group_name, features in feature_groups.items():
        summary_rows = []
        best_prediction_df = None

        sampler = optuna.samplers.TPESampler(seed=random_state)
        study = optuna.create_study(direction="minimize", sampler=sampler)

        def objective(trial):
            trial_params = suggest_optuna_params(trial, optuna_space)
            model_params = dict(base_model_params)
            model_params.update(trial_params)

            metrics, prediction_df = evaluate_model(
                train_df=train_df_full,
                holdout_df=holdout_df_full,
                features=features,
                target=args.target,
                baseline_column=args.baseline_column,
                model_params=model_params,
            )

            summary_row = {
                "trial_number": int(trial.number),
                "candidate_index": int(trial.number + 1),
                "n_features": len(features),
                "features_json": json.dumps(features),
                "n_train_rows": int(len(train_df_full)),
                "n_holdout_rows": int(len(holdout_df_full)),
                "holdout_rmse": metrics["holdout_rmse"],
                "holdout_mae": metrics["holdout_mae"],
                "holdout_r2": metrics["holdout_r2"],
                "baseline_rmse": metrics.get("baseline_rmse"),
                "baseline_mae": metrics.get("baseline_mae"),
                "params_json": json.dumps(model_params, sort_keys=True),
            }
            summary_rows.append(summary_row)
            return metrics["holdout_rmse"]

        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

        trial_df = pd.DataFrame(summary_rows).sort_values(
            ["holdout_rmse", "holdout_mae", "holdout_r2"],
            ascending=[True, True, False],
        ).reset_index(drop=True)
        trial_df["rank_rmse"] = np.arange(1, len(trial_df) + 1)
        trial_df.insert(0, "feature_group", feature_group_name)
        all_trial_tables.append(trial_df)

        best_params = dict(base_model_params)
        best_params.update(study.best_trial.params)
        best_metrics, best_prediction_raw = evaluate_model(
            train_df=train_df_full,
            holdout_df=holdout_df_full,
            features=features,
            target=args.target,
            baseline_column=args.baseline_column,
            model_params=best_params,
        )

        best_prediction_df = best_prediction_raw[metadata_columns].copy()
        best_prediction_df["feature_group"] = feature_group_name
        best_prediction_df["prediction"] = best_prediction_raw["prediction"]
        best_prediction_df["model_residual"] = best_prediction_raw["model_residual"]
        if "baseline_residual" in best_prediction_raw.columns:
            best_prediction_df["baseline_residual"] = best_prediction_raw["baseline_residual"]
        all_best_prediction_tables.append(best_prediction_df)

        best_row = trial_df.iloc[0].to_dict()
        best_row["n_rows_after_filtering"] = int(len(model_df))
        best_row["holdout_column"] = args.holdout_column
        best_row["holdout_value"] = args.holdout_value
        best_rows.append(best_row)

        optuna_best_trials.append(
            {
                "feature_group": feature_group_name,
                "best_trial_number": int(study.best_trial.number),
                "best_value_rmse": float(study.best_value),
                "best_params": study.best_trial.params,
            }
        )

    all_trials_df = pd.concat(all_trial_tables, ignore_index=True)
    best_models_df = pd.DataFrame(best_rows).sort_values(
        ["holdout_rmse", "holdout_mae", "holdout_r2"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    best_models_df["overall_rank_rmse"] = np.arange(1, len(best_models_df) + 1)
    best_predictions_df = pd.concat(all_best_prediction_tables, ignore_index=True)

    summary = {
        "input_csv": str(args.input_csv),
        "target": args.target,
        "holdout_column": args.holdout_column,
        "holdout_value": args.holdout_value,
        "baseline_column": args.baseline_column,
        "n_trials_per_feature_group": int(args.n_trials),
        "n_feature_groups": int(len(feature_groups)),
        "n_rows_after_filtering": int(len(model_df)),
        "n_train_rows": int(len(train_df_full)),
        "n_holdout_rows": int(len(holdout_df_full)),
        "feature_groups": feature_groups,
        "base_model_params": base_model_params,
        "optuna_space": optuna_space,
        "optuna_best_trials": optuna_best_trials,
        "dropna_columns_used": required_non_null_columns,
    }

    all_trials_df.to_csv(args.output_dir / "trial_results_all.csv", index=False)
    best_models_df.to_csv(args.output_dir / "best_models_summary.csv", index=False)
    best_predictions_df.to_csv(args.output_dir / "best_holdout_predictions.csv", index=False)
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
