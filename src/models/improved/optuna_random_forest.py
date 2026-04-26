from __future__ import annotations

import argparse
import logging
import sys
import time

import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.models.utils.config_utils import load_config
from src.models.utils.data_utils import load_split, validate_feature_columns
from src.models.utils.io_utils import ensure_dir, save_csv, save_json
from src.models.utils.metric_utils import compute_binary_metrics, sweep_thresholds_for_f1
from src.models.utils.plot_utils import (
    save_confusion_matrix_plot,
    save_feature_importance_plot,
    save_pr_curve,
    save_roc_curve,
    save_threshold_plot,
)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def build_rf_params(trial, base_params):
    params = dict(base_params)

    params.update({
        "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
    })

    return params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)

    output_dir = ensure_dir(cfg["paths"]["output_dir"])
    metrics_dir = ensure_dir(output_dir / "metrics")
    plots_dir = ensure_dir(output_dir / "plots")
    preds_dir = ensure_dir(output_dir / "predictions")
    models_dir = ensure_dir(output_dir / "models")
    study_dir = ensure_dir(output_dir / "study")

    X_train, y_train, train_cols, _ = load_split(
        cfg["paths"]["train_csv"], cfg["target_col"], cfg["data"]["meta_cols"], cfg["data"]["dtype"]
    )
    X_val, y_val, val_cols, _ = load_split(
        cfg["paths"]["val_csv"], cfg["target_col"], cfg["data"]["meta_cols"], cfg["data"]["dtype"]
    )
    X_test, y_test, test_cols, _ = load_split(
        cfg["paths"]["test_csv"], cfg["target_col"], cfg["data"]["meta_cols"], cfg["data"]["dtype"]
    )

    validate_feature_columns(train_cols, val_cols, test_cols)

    base_params = dict(cfg["model_params"])

    def objective(trial):
        params = build_rf_params(trial, base_params)
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        y_val_prob = model.predict_proba(X_val)[:, 1]
        metrics = compute_binary_metrics(y_val, y_val_prob, threshold=0.5)
        return metrics["aucpr"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=cfg["optuna"]["n_trials"])

    best_params = dict(base_params)
    best_params.update(study.best_params)

    save_json(study.best_params, study_dir / "best_params.json")

    model = RandomForestClassifier(**best_params)
    model.fit(X_train, y_train)

    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_val_prob = model.predict_proba(X_val)[:, 1]
    y_test_prob = model.predict_proba(X_test)[:, 1]

    best_threshold, threshold_rows = sweep_thresholds_for_f1(y_val, y_val_prob)

    train_metrics = compute_binary_metrics(y_train, y_train_prob, best_threshold)
    val_metrics = compute_binary_metrics(y_val, y_val_prob, best_threshold)
    test_metrics = compute_binary_metrics(y_test, y_test_prob, best_threshold)

    save_json(train_metrics, metrics_dir / "train_metrics_best_threshold.json")
    save_json(val_metrics, metrics_dir / "val_metrics_best_threshold.json")
    save_json(test_metrics, metrics_dir / "test_metrics_best_threshold.json")

    save_threshold_plot(threshold_rows, plots_dir / "threshold.png", "Threshold vs F1")

    save_pr_curve(y_val, y_val_prob, plots_dir / "val_pr.png", "PR Curve")
    save_roc_curve(y_val, y_val_prob, plots_dir / "val_roc.png", "ROC Curve")

    save_feature_importance_plot(
        train_cols,
        model.feature_importances_,
        plots_dir / "feature_importance.png",
        "Feature Importance",
    )

    import joblib
    joblib.dump(model, models_dir / "model.joblib")

    logging.info("RF Optuna complete")


if __name__ == "__main__":
    main()
