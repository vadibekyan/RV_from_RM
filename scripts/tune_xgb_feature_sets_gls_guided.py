import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from scipy.signal import lombscargle as scipy_lombscargle
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

try:
    from astropy.timeseries import LombScargle
except ImportError:
    LombScargle = None

"""
Example:
python scripts/tune_xgb_feature_sets_gls_guided.py \
  --input-csv model_tables/rm_df_full.csv \
  --star-csv model_tables/hd22496_full.csv \
  --output-dir results/xgb_gls_guided_hd22496 \
  --target true_vrad \
  --feature-groups-json config/rm_feature_groups_staged_v1.json \
  --search-type optuna \
  --n-trials 200 \
  --target-window-min 5.05 \
  --target-window-max 5.15 \
  --background-window-min 6.0 \
  --background-window-max 8.0 \
  --gls-alpha 0.5 \
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
            "Tune XGBoost hyperparameters for one or more feature sets using grouped CV on the RM table, "
            "while also rewarding a stronger GLS signal in a target window on a second star table."
        )
    )
    parser.add_argument("--input-csv", type=Path, required=True, help="Path to the RM training table.")
    parser.add_argument("--star-csv", type=Path, required=True, help="Path to the target-star table, e.g. hd22496_full.csv.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where outputs will be saved.")
    parser.add_argument("--target", required=True, help="Target column to predict, e.g. true_vrad.")
    parser.add_argument(
        "--group-column",
        default="corrected_file",
        help="Column used to define grouped CV splits in the RM table. Default: corrected_file.",
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
            "Optional extra columns that must be non-null in both RM and star tables. "
            "Rows are filtered using the union of all requested features plus target."
        ),
    )
    parser.add_argument(
        "--cv-type",
        default="group_kfold",
        choices=["group_kfold", "leave_one_group_out"],
        help="Cross-validation scheme. Default: group_kfold.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of folds for group_kfold. Ignored for leave_one_group_out. Default: 5.",
    )
    parser.add_argument(
        "--search-type",
        default="optuna",
        choices=["optuna"],
        help="Hyperparameter search strategy. Currently only optuna is supported.",
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
    parser.add_argument("--gls-min-period", type=float, default=1.0, help="Minimum period for GLS. Default: 1.0.")
    parser.add_argument("--gls-max-period", type=float, default=50.0, help="Maximum period for GLS. Default: 50.0.")
    parser.add_argument("--target-window-min", type=float, default=5.05, help="Lower bound of target period window.")
    parser.add_argument("--target-window-max", type=float, default=5.15, help="Upper bound of target period window.")
    parser.add_argument("--background-window-min", type=float, default=6.0, help="Lower bound of background period window.")
    parser.add_argument("--background-window-max", type=float, default=8.0, help="Upper bound of background period window.")
    parser.add_argument(
        "--gls-alpha",
        type=float,
        default=0.5,
        help="Weight of the GLS ratio term in objective = cv_rmse - gls_alpha * gls_ratio. Default: 0.5.",
    )
    parser.add_argument(
        "--gls-epsilon",
        type=float,
        default=1.0e-12,
        help="Small floor applied to GLS denominator for numerical stability. Default: 1e-12.",
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
        raise ValueError(f"Invalid --feature-group value {raw_value!r}. Expected name=feature1,feature2,...")
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
        raise ValueError("Provide at least one feature set using --features, --feature-group, or --feature-groups-json.")

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
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])


def get_cv_splits(df, group_column, cv_type, n_splits):
    groups = df[group_column].to_numpy()
    unique_groups = pd.Series(groups).dropna().unique()
    n_unique_groups = len(unique_groups)

    if n_unique_groups < 2:
        raise ValueError(f"Need at least 2 non-null groups in {group_column!r} after filtering, found {n_unique_groups}.")

    if cv_type == "leave_one_group_out":
        splitter = LeaveOneGroupOut()
    else:
        if n_splits < 2:
            raise ValueError("--n-splits must be at least 2 for group_kfold.")
        if n_unique_groups < n_splits:
            raise ValueError(f"Requested n_splits={n_splits}, but only {n_unique_groups} non-null groups are available.")
        splitter = GroupKFold(n_splits=n_splits)

    splits = list(splitter.split(df, groups=groups))
    if not splits:
        raise ValueError("No CV splits were generated.")
    return splits


def suggest_optuna_params(trial, optuna_space):
    params = {}
    for name, spec in optuna_space.items():
        spec_type = spec["type"]
        if spec_type == "int":
            kwargs = {"low": int(spec["low"]), "high": int(spec["high"])}
            if "step" in spec:
                kwargs["step"] = int(spec["step"])
            if "log" in spec:
                kwargs["log"] = bool(spec["log"])
            params[name] = trial.suggest_int(name, **kwargs)
        elif spec_type == "float":
            kwargs = {"low": float(spec["low"]), "high": float(spec["high"])}
            if "step" in spec:
                kwargs["step"] = float(spec["step"])
            if "log" in spec:
                kwargs["log"] = bool(spec["log"])
            params[name] = trial.suggest_float(name, **kwargs)
        else:
            choices = spec.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ValueError(f"Categorical Optuna parameter {name!r} needs a non-empty 'choices' list.")
            params[name] = trial.suggest_categorical(name, choices)
    return params


def evaluate_cv_metrics(df, features, target, group_column, cv_splits, model_params, candidate_index):
    fold_rows = []
    for fold_index, (train_index, valid_index) in enumerate(cv_splits, start=1):
        train_df = df.iloc[train_index]
        valid_df = df.iloc[valid_index]

        model = build_model(model_params)
        model.fit(train_df[features], train_df[target])
        predictions = model.predict(valid_df[features])

        valid_group_values = sorted(str(value) for value in valid_df[group_column].dropna().unique())
        fold_rows.append(
            {
                "candidate_index": candidate_index,
                "fold_index": fold_index,
                "n_train_rows": int(len(train_df)),
                "n_valid_rows": int(len(valid_df)),
                "n_train_groups": int(train_df[group_column].nunique()),
                "n_valid_groups": int(valid_df[group_column].nunique()),
                "valid_groups": json.dumps(valid_group_values),
                "rmse": rmse(valid_df[target], predictions),
                "mae": float(mean_absolute_error(valid_df[target], predictions)),
                "r2": float(r2_score(valid_df[target], predictions)),
                "params_json": json.dumps(model_params, sort_keys=True),
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    return {
        "mean_rmse": float(fold_df["rmse"].mean()),
        "std_rmse": float(fold_df["rmse"].std(ddof=0)),
        "mean_mae": float(fold_df["mae"].mean()),
        "std_mae": float(fold_df["mae"].std(ddof=0)),
        "mean_r2": float(fold_df["r2"].mean()),
        "std_r2": float(fold_df["r2"].std(ddof=0)),
    }, fold_rows


def evaluate_gls_metrics(star_df, features, model, args):
    prediction_df = star_df.sort_values(["observation_file", "rjd"]).copy()
    prediction_df["predicted_true_vrad"] = model.predict(prediction_df[features])

    gls_input_df = prediction_df[["rjd", "predicted_true_vrad"]].dropna().sort_values("rjd").copy()
    if gls_input_df.empty:
        raise ValueError("No non-null rows available for GLS computation on star predictions.")

    min_frequency = 1.0 / args.gls_max_period
    max_frequency = 1.0 / args.gls_min_period

    if LombScargle is not None:
        gls = LombScargle(
            gls_input_df["rjd"].to_numpy(),
            gls_input_df["predicted_true_vrad"].to_numpy(),
            center_data=True,
            fit_mean=True,
        )
        frequency, power = gls.autopower(minimum_frequency=min_frequency, maximum_frequency=max_frequency)
    else:
        time_values = gls_input_df["rjd"].to_numpy(dtype=float)
        signal_values = gls_input_df["predicted_true_vrad"].to_numpy(dtype=float)

        signal_values = signal_values - np.mean(signal_values)
        frequency = np.linspace(min_frequency, max_frequency, 10000)
        angular_frequency = 2.0 * np.pi * frequency
        power = scipy_lombscargle(
            time_values,
            signal_values,
            angular_frequency,
            precenter=False,
            normalize=True,
        )

    period = 1.0 / frequency

    target_mask = (period >= args.target_window_min) & (period <= args.target_window_max)
    background_mask = (period >= args.background_window_min) & (period <= args.background_window_max)

    if not np.any(target_mask):
        raise ValueError("No GLS samples fall inside the requested target period window.")
    if not np.any(background_mask):
        raise ValueError("No GLS samples fall inside the requested background period window.")

    target_periods = period[target_mask]
    target_powers = power[target_mask]
    background_powers = power[background_mask]

    peak_idx = int(np.argmax(target_powers))
    peak_period = float(target_periods[peak_idx])
    peak_power = float(target_powers[peak_idx])
    background_median = float(np.median(background_powers))
    gls_ratio = float(peak_power / max(background_median, args.gls_epsilon))

    return {
        "peak_period_target_window": peak_period,
        "peak_power_target_window": peak_power,
        "median_power_background_window": background_median,
        "gls_ratio": gls_ratio,
        "n_gls_samples": int(len(period)),
    }, prediction_df, pd.DataFrame({"frequency": frequency, "period": period, "power": power}).sort_values("period")


def main():
    args = parse_args()
    feature_groups = get_feature_groups(args)
    optuna_space = get_optuna_space(args)
    base_model_params = parse_json_string(args.base_model_params, "--base-model-params")
    random_state = int(base_model_params.get("random_state", 42))

    if args.target_window_min >= args.target_window_max:
        raise ValueError("Target period window must have min < max.")
    if args.background_window_min >= args.background_window_max:
        raise ValueError("Background period window must have min < max.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rm_df = pd.read_csv(args.input_csv)
    star_df = pd.read_csv(args.star_csv)

    all_features = []
    for features in feature_groups.values():
        all_features.extend(features)

    required_columns = [args.group_column, args.target, "rjd", *all_features, *(args.dropna_subset or [])]
    validate_columns(rm_df, required_columns)
    validate_columns(star_df, ["rjd", *all_features, *(args.dropna_subset or [])])

    required_non_null_columns = list(dict.fromkeys([args.target, *all_features, *(args.dropna_subset or [])]))
    rm_model_df = rm_df.dropna(subset=[args.group_column, *required_non_null_columns]).copy()
    star_model_df = star_df.dropna(subset=[*all_features, *(args.dropna_subset or [])]).copy()

    if star_model_df.empty:
        raise ValueError("No rows remain in the star table after dropping missing feature values.")

    cv_splits = get_cv_splits(
        df=rm_model_df,
        group_column=args.group_column,
        cv_type=args.cv_type,
        n_splits=args.n_splits,
    )

    all_trial_tables = []
    all_fold_tables = []
    best_rows = []
    best_prediction_tables = []
    best_periodogram_tables = []
    optuna_best_trials = []

    for feature_group_name, features in feature_groups.items():
        summary_rows = []
        fold_rows = []

        sampler = optuna.samplers.TPESampler(seed=random_state)
        study = optuna.create_study(direction="minimize", sampler=sampler)

        def objective(trial):
            trial_params = suggest_optuna_params(trial, optuna_space)
            model_params = dict(base_model_params)
            model_params.update(trial_params)

            cv_metrics, candidate_fold_rows = evaluate_cv_metrics(
                df=rm_model_df,
                features=features,
                target=args.target,
                group_column=args.group_column,
                cv_splits=cv_splits,
                model_params=model_params,
                candidate_index=trial.number + 1,
            )

            full_model = build_model(model_params)
            full_model.fit(rm_model_df[features], rm_model_df[args.target])

            gls_metrics, _, _ = evaluate_gls_metrics(
                star_df=star_model_df,
                features=features,
                model=full_model,
                args=args,
            )

            objective_value = float(cv_metrics["mean_rmse"] - args.gls_alpha * gls_metrics["gls_ratio"])

            for row in candidate_fold_rows:
                row["trial_number"] = int(trial.number)
            fold_rows.extend(candidate_fold_rows)

            summary_rows.append(
                {
                    "trial_number": int(trial.number),
                    "candidate_index": int(trial.number + 1),
                    "n_features": len(features),
                    "features_json": json.dumps(features),
                    "mean_rmse": cv_metrics["mean_rmse"],
                    "std_rmse": cv_metrics["std_rmse"],
                    "mean_mae": cv_metrics["mean_mae"],
                    "std_mae": cv_metrics["std_mae"],
                    "mean_r2": cv_metrics["mean_r2"],
                    "std_r2": cv_metrics["std_r2"],
                    "peak_period_target_window": gls_metrics["peak_period_target_window"],
                    "peak_power_target_window": gls_metrics["peak_power_target_window"],
                    "median_power_background_window": gls_metrics["median_power_background_window"],
                    "gls_ratio": gls_metrics["gls_ratio"],
                    "objective_value": objective_value,
                    "params_json": json.dumps(model_params, sort_keys=True),
                }
            )
            return objective_value

        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

        trial_df = pd.DataFrame(summary_rows).sort_values(
            ["objective_value", "mean_rmse", "gls_ratio"],
            ascending=[True, True, False],
        ).reset_index(drop=True)
        trial_df["rank_objective"] = np.arange(1, len(trial_df) + 1)
        trial_df.insert(0, "feature_group", feature_group_name)
        fold_df = pd.DataFrame(fold_rows)
        fold_df.insert(0, "feature_group", feature_group_name)
        all_trial_tables.append(trial_df)
        all_fold_tables.append(fold_df)

        best_params = dict(base_model_params)
        best_params.update(study.best_trial.params)
        best_model = build_model(best_params)
        best_model.fit(rm_model_df[features], rm_model_df[args.target])
        best_gls_metrics, best_prediction_df, best_periodogram_df = evaluate_gls_metrics(
            star_df=star_model_df,
            features=features,
            model=best_model,
            args=args,
        )

        prediction_save_df = best_prediction_df.copy()
        prediction_save_df["feature_group"] = feature_group_name
        best_prediction_tables.append(prediction_save_df)

        periodogram_save_df = best_periodogram_df.copy()
        periodogram_save_df["feature_group"] = feature_group_name
        best_periodogram_tables.append(periodogram_save_df)

        best_row = trial_df.iloc[0].to_dict()
        best_row["cv_type"] = args.cv_type
        best_row["n_splits_used"] = len(cv_splits)
        best_row["n_rows_after_filtering"] = int(len(rm_model_df))
        best_row["n_groups_after_filtering"] = int(rm_model_df[args.group_column].nunique())
        best_row["n_star_rows_after_filtering"] = int(len(star_model_df))
        best_rows.append(best_row)

        optuna_best_trials.append(
            {
                "feature_group": feature_group_name,
                "best_trial_number": int(study.best_trial.number),
                "best_objective_value": float(study.best_value),
                "best_params": study.best_trial.params,
                "best_gls_metrics": best_gls_metrics,
            }
        )

    all_trials_df = pd.concat(all_trial_tables, ignore_index=True)
    all_folds_df = pd.concat(all_fold_tables, ignore_index=True)
    best_models_df = pd.DataFrame(best_rows).sort_values(
        ["objective_value", "mean_rmse", "gls_ratio"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    best_models_df["overall_rank_objective"] = np.arange(1, len(best_models_df) + 1)
    best_predictions_df = pd.concat(best_prediction_tables, ignore_index=True)
    best_periodograms_df = pd.concat(best_periodogram_tables, ignore_index=True)

    summary = {
        "input_csv": str(args.input_csv),
        "star_csv": str(args.star_csv),
        "target": args.target,
        "group_column": args.group_column,
        "cv_type": args.cv_type,
        "n_splits_requested": int(args.n_splits),
        "n_splits_used": int(len(cv_splits)),
        "n_trials_per_feature_group": int(args.n_trials),
        "n_feature_groups": int(len(feature_groups)),
        "n_rows_after_filtering": int(len(rm_model_df)),
        "n_groups_after_filtering": int(rm_model_df[args.group_column].nunique()),
        "n_star_rows_after_filtering": int(len(star_model_df)),
        "feature_groups": feature_groups,
        "base_model_params": base_model_params,
        "optuna_space": optuna_space,
        "gls_settings": {
            "gls_min_period": args.gls_min_period,
            "gls_max_period": args.gls_max_period,
            "target_window_min": args.target_window_min,
            "target_window_max": args.target_window_max,
            "background_window_min": args.background_window_min,
            "background_window_max": args.background_window_max,
            "gls_alpha": args.gls_alpha,
            "gls_epsilon": args.gls_epsilon,
        },
        "dropna_columns_used": required_non_null_columns,
        "optuna_best_trials": optuna_best_trials,
    }

    all_trials_df.to_csv(args.output_dir / "trial_results_all.csv", index=False)
    all_folds_df.to_csv(args.output_dir / "cv_fold_metrics.csv", index=False)
    best_models_df.to_csv(args.output_dir / "best_models_summary.csv", index=False)
    best_predictions_df.to_csv(args.output_dir / "best_star_predictions.csv", index=False)
    best_periodograms_df.to_csv(args.output_dir / "best_star_periodograms.csv", index=False)
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
