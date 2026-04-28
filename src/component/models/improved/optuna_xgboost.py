from __future__ import annotations

import argparse
import logging
import sys
import time

import optuna
import pandas as pd
import xgboost as xgb

from src.component.models.utils.config_utils import load_config
from src.component.models.utils.data_utils import load_split, validate_feature_columns
from src.component.models.utils.io_utils import ensure_dir, save_csv, save_json
from src.component.models.utils.metric_utils import compute_binary_metrics, sweep_thresholds_for_f1
from src.component.models.utils.plot_utils import (
    save_confusion_matrix_plot,
    save_feature_importance_plot,
    save_pr_curve,
    save_roc_curve,
    save_threshold_plot,
)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def build_xgb_params(trial: optuna.Trial, base_params: dict) -> dict:
    params = dict(base_params)

    params.update(
        {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
    )
    return params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)

    experiment_name = cfg["experiment_name"]
    target_col = cfg["target_col"]
    output_dir = ensure_dir(cfg["paths"]["output_dir"])
    metrics_dir = ensure_dir(output_dir / "metrics")
    plots_dir = ensure_dir(output_dir / "plots")
    preds_dir = ensure_dir(output_dir / "predictions")
    models_dir = ensure_dir(output_dir / "models")
    study_dir = ensure_dir(output_dir / "study")

    meta_cols = cfg["data"]["meta_cols"]
    dtype = cfg["data"].get("dtype", "float32")
    default_threshold = float(cfg["evaluation"].get("default_threshold", 0.5))
    n_trials = int(cfg["optuna"]["n_trials"])
    study_name = cfg["optuna"].get("study_name", experiment_name)
    direction = cfg["optuna"].get("direction", "maximize")

    logging.info("Experiment: %s", experiment_name)
    logging.info("Loading train/val/test data...")

    X_train, y_train, train_cols, _ = load_split(
        cfg["paths"]["train_csv"], target_col, meta_cols, dtype
    )
    X_val, y_val, val_cols, _ = load_split(
        cfg["paths"]["val_csv"], target_col, meta_cols, dtype
    )
    X_test, y_test, test_cols, _ = load_split(
        cfg["paths"]["test_csv"], target_col, meta_cols, dtype
    )

    validate_feature_columns(train_cols, val_cols, test_cols)

    base_params = dict(cfg["model_params"])
    random_state = int(cfg["training"].get("random_state", 42))
    base_params.setdefault("random_state", random_state)
    base_params.setdefault("eval_metric", "logloss")
    base_params.setdefault("tree_method", "hist")
    if cfg["training"].get("use_gpu", True):
        base_params.setdefault("device", "cuda")

    def objective(trial: optuna.Trial) -> float:
        params = build_xgb_params(trial, base_params)
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        y_val_prob = model.predict_proba(X_val)[:, 1]
        metrics = compute_binary_metrics(y_val, y_val_prob, threshold=default_threshold)
        return metrics["aucpr"]

    logging.info("Starting Optuna with %d trials...", n_trials)
    start_study = time.time()

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    study_seconds = round(time.time() - start_study, 2)
    logging.info("Optuna finished in %.2f seconds", study_seconds)
    logging.info("Best trial value (val AUCPR): %.6f", study.best_value)
    logging.info("Best params: %s", study.best_params)

    best_params = dict(base_params)
    best_params.update(study.best_params)

    save_json(
        {
            "study_name": study_name,
            "direction": direction,
            "n_trials": n_trials,
            "study_seconds": study_seconds,
            "best_value_val_aucpr": study.best_value,
            "best_params": best_params,
        },
        study_dir / "best_study_result.json",
    )

    trials_df = study.trials_dataframe()
    save_csv(trials_df, study_dir / "optuna_trials.csv")

    logging.info("Retraining best XGBoost model on train split...")
    start_train = time.time()
    best_model = xgb.XGBClassifier(**best_params)
    best_model.fit(X_train, y_train)
    train_seconds = round(time.time() - start_train, 2)

    y_train_prob = best_model.predict_proba(X_train)[:, 1]
    y_val_prob = best_model.predict_proba(X_val)[:, 1]
    y_test_prob = best_model.predict_proba(X_test)[:, 1]

    train_metrics_default = compute_binary_metrics(y_train, y_train_prob, default_threshold)
    val_metrics_default = compute_binary_metrics(y_val, y_val_prob, default_threshold)
    test_metrics_default = compute_binary_metrics(y_test, y_test_prob, default_threshold)

    best_threshold, threshold_rows = sweep_thresholds_for_f1(y_val, y_val_prob)
    train_metrics_best = compute_binary_metrics(y_train, y_train_prob, best_threshold)
    val_metrics_best = compute_binary_metrics(y_val, y_val_prob, best_threshold)
    test_metrics_best = compute_binary_metrics(y_test, y_test_prob, best_threshold)

    save_json(
        {
            "experiment_name": experiment_name,
            "model_type": "xgboost",
            "train_seconds": train_seconds,
            "study_seconds": study_seconds,
            "default_threshold": default_threshold,
            "best_val_threshold_for_f1": best_threshold,
            "best_params": best_params,
        },
        metrics_dir / "run_info.json",
    )

    save_json(train_metrics_default, metrics_dir / "train_metrics_default.json")
    save_json(val_metrics_default, metrics_dir / "val_metrics_default.json")
    save_json(test_metrics_default, metrics_dir / "test_metrics_default.json")

    save_json(train_metrics_best, metrics_dir / "train_metrics_best_threshold.json")
    save_json(val_metrics_best, metrics_dir / "val_metrics_best_threshold.json")
    save_json(test_metrics_best, metrics_dir / "test_metrics_best_threshold.json")

    summary_df = pd.DataFrame(
        [
            {
                "experiment_name": experiment_name,
                "model_type": "xgboost",
                "train_seconds": train_seconds,
                "study_seconds": study_seconds,
                "best_val_aucpr_optuna": study.best_value,
                "train_aucpr_default": train_metrics_default["aucpr"],
                "train_f1_default": train_metrics_default["f1"],
                "val_aucpr_default": val_metrics_default["aucpr"],
                "val_f1_default": val_metrics_default["f1"],
                "val_aucpr_best_threshold": val_metrics_best["aucpr"],
                "val_f1_best_threshold": val_metrics_best["f1"],
                "best_val_threshold_for_f1": best_threshold,
                "test_aucpr_best_threshold": test_metrics_best["aucpr"],
                "test_f1_best_threshold": test_metrics_best["f1"],
            }
        ]
    )
    save_csv(summary_df, metrics_dir / "summary.csv")

    threshold_df = pd.DataFrame(threshold_rows)
    save_csv(threshold_df, metrics_dir / "val_threshold_sweep.csv")

    val_pred_df = pd.DataFrame(
        {
            "y_true": y_val,
            "y_prob": y_val_prob,
            "y_pred_default": (y_val_prob >= default_threshold).astype(int),
            "y_pred_best_threshold": (y_val_prob >= best_threshold).astype(int),
        }
    )
    test_pred_df = pd.DataFrame(
        {
            "y_true": y_test,
            "y_prob": y_test_prob,
            "y_pred_default": (y_test_prob >= default_threshold).astype(int),
            "y_pred_best_threshold": (y_test_prob >= best_threshold).astype(int),
        }
    )

    save_csv(val_pred_df, preds_dir / "val_predictions.csv")
    save_csv(test_pred_df, preds_dir / "test_predictions.csv")

    save_pr_curve(y_train, y_train_prob, plots_dir / "train_pr_curve.png", f"{experiment_name} - Train PR Curve")
    save_pr_curve(y_val, y_val_prob, plots_dir / "val_pr_curve.png", f"{experiment_name} - Val PR Curve")
    save_pr_curve(y_test, y_test_prob, plots_dir / "test_pr_curve.png", f"{experiment_name} - Test PR Curve")

    save_roc_curve(y_train, y_train_prob, plots_dir / "train_roc_curve.png", f"{experiment_name} - Train ROC Curve")
    save_roc_curve(y_val, y_val_prob, plots_dir / "val_roc_curve.png", f"{experiment_name} - Val ROC Curve")
    save_roc_curve(y_test, y_test_prob, plots_dir / "test_roc_curve.png", f"{experiment_name} - Test ROC Curve")

    save_threshold_plot(threshold_rows, plots_dir / "val_threshold_vs_f1.png", f"{experiment_name} - Val Threshold vs F1")

    save_confusion_matrix_plot(
        val_metrics_default["confusion_matrix"],
        plots_dir / "val_confusion_matrix_default.png",
        f"{experiment_name} - Val Confusion Matrix @ {default_threshold:.2f}",
    )
    save_confusion_matrix_plot(
        val_metrics_best["confusion_matrix"],
        plots_dir / "val_confusion_matrix_best_threshold.png",
        f"{experiment_name} - Val Confusion Matrix @ {best_threshold:.2f}",
    )

    if hasattr(best_model, "feature_importances_"):
        save_feature_importance_plot(
            train_cols,
            best_model.feature_importances_,
            plots_dir / "feature_importance_top25.png",
            f"{experiment_name} - Top 25 Feature Importances",
            top_k=25,
        )

    best_model.save_model(str(models_dir / "model.json"))
    logging.info("Saved tuned XGBoost outputs to %s", output_dir)
    logging.info("Done.")


if __name__ == "__main__":
    main()
