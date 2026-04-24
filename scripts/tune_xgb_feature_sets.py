import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, ParameterGrid, ParameterSampler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

"""
Examples:

Single feature set:
python scripts/tune_xgb_feature_sets.py \
  --input-csv model_tables/rm_df_full.csv \
  --output-dir results/xgb_tuning_run_01 \
  --target true_vrad \
  --features vrad_subtract_mean fwhm_fractional_mean bis_span_fractional_mean contrast_fractional_mean

Multiple named feature sets from CLI:
python scripts/tune_xgb_feature_sets.py \
  --input-csv model_tables/rm_df_full.csv \
  --output-dir results/xgb_tuning_run_01 \
  --target true_vrad \
  --feature-group baseline=vrad_subtract_mean,fwhm_fractional_mean,bis_span_fractional_mean,contrast_fractional_mean \
  --feature-group activity=vrad_subtract_mean,s_mw_fractional_mean,ha_fractional_mean,na_fractional_mean,ca_fractional_mean,rhk_fractional_mean

Multiple named feature sets from JSON:
python scripts/tune_xgb_feature_sets.py \
  --input-csv model_tables/rm_df_full.csv \
  --output-dir results/xgb_tuning_run_01 \
  --target true_vrad \
  --feature-groups-json config/rm_feature_groups_staged_v1.json \
  --search-type optuna \
  --n-trials 200 \
  --base-model-params '{"objective":"reg:squarederror","random_state":42,"tree_method":"hist"}'
"""


DEFAULT_PARAM_GRID = {
    "n_estimators": [300, 500, 800],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.8, 0.9, 1.0],
    "colsample_bytree": [0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5],
}

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
            "Tune XGBoost hyperparameters through CV for one or more feature sets, "
            "using the same splits for every feature set, and save performance summaries."
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
        help="Directory where CV tables, summaries, and metadata will be saved.",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Target column to predict, e.g. true_vrad.",
    )
    parser.add_argument(
        "--group-column",
        default="corrected_file",
        help="Column used to define grouped CV splits. Default: corrected_file.",
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
            "Optional extra columns that must be non-null. To keep CV identical across feature sets, "
            "rows are filtered using the union of all requested features plus the target."
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
        choices=["grid", "random", "optuna"],
        help="Hyperparameter search strategy. Default: optuna.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=25,
        help="Number of random samples when --search-type random. Default: 25.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=200,
        help="Number of Optuna trials when --search-type optuna. Default: 200.",
    )
    parser.add_argument(
        "--param-grid-json",
        default=None,
        help="JSON string with parameter grid/distributions. If omitted, a built-in grid is used.",
    )
    parser.add_argument(
        "--param-grid-file",
        type=Path,
        default=None,
        help="Path to a JSON file with parameter grid/distributions.",
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
        help=(
            "JSON string with fixed XGBoost parameters added to every hyperparameter setting. "
            "For example '{\"objective\":\"reg:squarederror\",\"random_state\":42,\"tree_method\":\"hist\"}'."
        ),
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


def get_param_grid(args):
    if args.param_grid_json and args.param_grid_file:
        raise ValueError("Use only one of --param-grid-json or --param-grid-file.")

    if args.param_grid_json:
        param_grid = parse_json_string(args.param_grid_json, "--param-grid-json")
    elif args.param_grid_file:
        param_grid = load_json_file(args.param_grid_file)
    else:
        param_grid = DEFAULT_PARAM_GRID

    if not isinstance(param_grid, dict) or not param_grid:
        raise ValueError("Parameter grid must be a non-empty JSON object.")

    for key, value in param_grid.items():
        if not isinstance(value, list) or not value:
            raise ValueError(f"Parameter {key!r} must map to a non-empty list.")

    return param_grid


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


def get_cv_splits(df, group_column, cv_type, n_splits):
    groups = df[group_column].to_numpy()
    unique_groups = pd.Series(groups).dropna().unique()
    n_unique_groups = len(unique_groups)

    if n_unique_groups < 2:
        raise ValueError(
            f"Need at least 2 non-null groups in {group_column!r} after filtering, found {n_unique_groups}."
        )

    if cv_type == "leave_one_group_out":
        splitter = LeaveOneGroupOut()
    else:
        if n_splits < 2:
            raise ValueError("--n-splits must be at least 2 for group_kfold.")
        if n_unique_groups < n_splits:
            raise ValueError(
                f"Requested n_splits={n_splits}, but only {n_unique_groups} non-null groups are available."
            )
        splitter = GroupKFold(n_splits=n_splits)

    splits = list(splitter.split(df, groups=groups))
    if not splits:
        raise ValueError("No CV splits were generated.")
    return splits


def get_param_candidates(param_grid, search_type, n_iter, random_state):
    if search_type == "grid":
        return list(ParameterGrid(param_grid))
    return list(ParameterSampler(param_grid, n_iter=n_iter, random_state=random_state))


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


def evaluate_model_params(df, features, target, group_column, cv_splits, model_params, candidate_index):
    fold_rows = []

    for fold_index, (train_index, valid_index) in enumerate(cv_splits, start=1):
        train_df = df.iloc[train_index]
        valid_df = df.iloc[valid_index]

        model = build_model(model_params)
        model.fit(train_df[features], train_df[target])
        predictions = model.predict(valid_df[features])

        fold_rmse = rmse(valid_df[target], predictions)
        fold_mae = float(mean_absolute_error(valid_df[target], predictions))
        fold_r2 = float(r2_score(valid_df[target], predictions))
        n_valid_groups = int(valid_df[group_column].nunique())
        valid_group_values = sorted(str(value) for value in valid_df[group_column].dropna().unique())

        fold_rows.append(
            {
                "candidate_index": candidate_index,
                "fold_index": fold_index,
                "n_train_rows": int(len(train_df)),
                "n_valid_rows": int(len(valid_df)),
                "n_train_groups": int(train_df[group_column].nunique()),
                "n_valid_groups": n_valid_groups,
                "valid_groups": json.dumps(valid_group_values),
                "rmse": fold_rmse,
                "mae": fold_mae,
                "r2": fold_r2,
                "params_json": json.dumps(model_params, sort_keys=True),
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    summary_row = {
        "candidate_index": candidate_index,
        "n_features": len(features),
        "features_json": json.dumps(features),
        "mean_rmse": float(fold_df["rmse"].mean()),
        "std_rmse": float(fold_df["rmse"].std(ddof=0)),
        "mean_mae": float(fold_df["mae"].mean()),
        "std_mae": float(fold_df["mae"].std(ddof=0)),
        "mean_r2": float(fold_df["r2"].mean()),
        "std_r2": float(fold_df["r2"].std(ddof=0)),
        "params_json": json.dumps(model_params, sort_keys=True),
    }
    return summary_row, fold_rows


def evaluate_feature_group(df, features, target, group_column, cv_splits, param_candidates, base_model_params):
    fold_rows = []
    summary_rows = []

    for candidate_index, candidate_params in enumerate(param_candidates, start=1):
        model_params = dict(base_model_params)
        model_params.update(candidate_params)
        summary_row, candidate_fold_rows = evaluate_model_params(
            df=df,
            features=features,
            target=target,
            group_column=group_column,
            cv_splits=cv_splits,
            model_params=model_params,
            candidate_index=candidate_index,
        )
        fold_rows.extend(candidate_fold_rows)
        summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["mean_rmse", "mean_mae", "mean_r2"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    summary_df["rank_rmse"] = np.arange(1, len(summary_df) + 1)
    fold_df = pd.DataFrame(fold_rows)
    return summary_df, fold_df


def evaluate_feature_group_optuna(
    df,
    features,
    target,
    group_column,
    cv_splits,
    base_model_params,
    optuna_space,
    n_trials,
    random_state,
):
    summary_rows = []
    fold_rows = []

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial):
        trial_params = suggest_optuna_params(trial, optuna_space)
        model_params = dict(base_model_params)
        model_params.update(trial_params)

        summary_row, candidate_fold_rows = evaluate_model_params(
            df=df,
            features=features,
            target=target,
            group_column=group_column,
            cv_splits=cv_splits,
            model_params=model_params,
            candidate_index=trial.number + 1,
        )
        summary_row["trial_number"] = int(trial.number)
        for row in candidate_fold_rows:
            row["trial_number"] = int(trial.number)

        summary_rows.append(summary_row)
        fold_rows.extend(candidate_fold_rows)
        return summary_row["mean_rmse"]

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["mean_rmse", "mean_mae", "mean_r2"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    summary_df["rank_rmse"] = np.arange(1, len(summary_df) + 1)
    fold_df = pd.DataFrame(fold_rows)
    return summary_df, fold_df, study


def main():
    args = parse_args()
    feature_groups = get_feature_groups(args)
    base_model_params = parse_json_string(args.base_model_params, "--base-model-params")
    random_state = int(base_model_params.get("random_state", 42))
    param_grid = None
    optuna_space = None

    if args.search_type == "optuna":
        optuna_space = get_optuna_space(args)
    else:
        param_grid = get_param_grid(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)

    all_features = []
    for features in feature_groups.values():
        all_features.extend(features)

    required_columns = [args.group_column, args.target, *all_features, *(args.dropna_subset or [])]
    validate_columns(df, required_columns)

    required_non_null_columns = list(dict.fromkeys([args.target, *all_features, *(args.dropna_subset or [])]))
    model_df = df.dropna(subset=required_non_null_columns).copy()

    cv_splits = get_cv_splits(
        df=model_df,
        group_column=args.group_column,
        cv_type=args.cv_type,
        n_splits=args.n_splits,
    )
    param_candidates = None
    if args.search_type != "optuna":
        param_candidates = get_param_candidates(
            param_grid=param_grid,
            search_type=args.search_type,
            n_iter=args.n_iter,
            random_state=random_state,
        )

    best_rows = []
    all_candidate_tables = []
    all_fold_tables = []
    optuna_best_trials = []

    for feature_group_name, features in feature_groups.items():
        if args.search_type == "optuna":
            candidate_df, fold_df, study = evaluate_feature_group_optuna(
                df=model_df,
                features=features,
                target=args.target,
                group_column=args.group_column,
                cv_splits=cv_splits,
                base_model_params=base_model_params,
                optuna_space=optuna_space,
                n_trials=args.n_trials,
                random_state=random_state,
            )
            optuna_best_trials.append(
                {
                    "feature_group": feature_group_name,
                    "best_trial_number": int(study.best_trial.number),
                    "best_value_rmse": float(study.best_value),
                    "best_params": study.best_trial.params,
                }
            )
        else:
            candidate_df, fold_df = evaluate_feature_group(
                df=model_df,
                features=features,
                target=args.target,
                group_column=args.group_column,
                cv_splits=cv_splits,
                param_candidates=param_candidates,
                base_model_params=base_model_params,
            )

        candidate_df.insert(0, "feature_group", feature_group_name)
        fold_df.insert(0, "feature_group", feature_group_name)
        all_candidate_tables.append(candidate_df)
        all_fold_tables.append(fold_df)

        best_row = candidate_df.iloc[0].to_dict()
        best_row["cv_type"] = args.cv_type
        best_row["n_splits_used"] = len(cv_splits)
        best_row["n_rows_after_filtering"] = int(len(model_df))
        best_row["n_groups_after_filtering"] = int(model_df[args.group_column].nunique())
        best_rows.append(best_row)

    all_candidates_df = pd.concat(all_candidate_tables, ignore_index=True)
    all_folds_df = pd.concat(all_fold_tables, ignore_index=True)
    best_models_df = pd.DataFrame(best_rows).sort_values(
        ["mean_rmse", "mean_mae", "mean_r2"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    best_models_df["overall_rank_rmse"] = np.arange(1, len(best_models_df) + 1)

    summary = {
        "input_csv": str(args.input_csv),
        "target": args.target,
        "group_column": args.group_column,
        "cv_type": args.cv_type,
        "n_splits_requested": int(args.n_splits),
        "n_splits_used": int(len(cv_splits)),
        "search_type": args.search_type,
        "n_param_candidates": int(len(param_candidates)) if param_candidates is not None else None,
        "n_trials_per_feature_group": int(args.n_trials) if args.search_type == "optuna" else None,
        "n_feature_groups": int(len(feature_groups)),
        "n_rows_after_filtering": int(len(model_df)),
        "n_groups_after_filtering": int(model_df[args.group_column].nunique()),
        "feature_groups": feature_groups,
        "base_model_params": base_model_params,
        "param_grid": param_grid,
        "optuna_space": optuna_space,
        "optuna_best_trials": optuna_best_trials,
        "dropna_columns_used": required_non_null_columns,
    }

    all_candidates_df.to_csv(args.output_dir / "cv_results_all.csv", index=False)
    all_folds_df.to_csv(args.output_dir / "cv_fold_metrics.csv", index=False)
    best_models_df.to_csv(args.output_dir / "best_models_summary.csv", index=False)
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
