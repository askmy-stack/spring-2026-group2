from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

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


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def build_model(model_type: str, model_params: dict, training_cfg: dict):
    model_type = model_type.lower()

    if model_type == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(**model_params)

    if model_type == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(**model_params)

    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        params = dict(model_params)
        if "random_state" not in params and "random_state" in training_cfg:
            params["random_state"] = training_cfg["random_state"]
        if "n_jobs" not in params and "n_jobs" in training_cfg:
            params["n_jobs"] = training_cfg["n_jobs"]
        return RandomForestClassifier(**params)

    raise ValueError(f"Unsupported model_type: {model_type}")


def save_model(model, model_type: str, model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)

    if model_type == "xgboost":
        model.save_model(str(model_dir / "model.json"))
    elif model_type == "lightgbm":
        model.booster_.save_model(str(model_dir / "model.txt"))
    elif model_type == "random_forest":
        import joblib
        joblib.dump(model, model_dir / "model.joblib")
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def get_feature_importances(model, model_type: str):
    if model_type in {"xgboost", "lightgbm", "random_forest"} and hasattr(model, "feature_importances_"):
        return model.feature_importances_
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)

    experiment_name = cfg["experiment_name"]
    model_type = cfg["model_type"].lower()
    target_col = cfg["target_col"]

    output_dir = ensure_dir(cfg["paths"]["output_dir"])
    metrics_dir = ensure_dir(output_dir / "metrics")
    plots_dir = ensure_dir(output_dir / "plots")
    preds_dir = ensure_dir(output_dir / "predictions")
    models_dir = ensure_dir(output_dir / "models")

    meta_cols = cfg["data"]["meta_cols"]
    dtype = cfg["data"].get("dtype", "float32")
    default_threshold = float(cfg["evaluation"].get("default_threshold", 0.5))

    logging.info("Experiment: %s", experiment_name)
    logging.info("Model type: %s", model_type)

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

    logging.info("Train shape: %s", X_train.shape)
    logging.info("Val shape: %s", X_val.shape)
    logging.info("Test shape: %s", X_test.shape)

    model = build_model(model_type, cfg["model_params"], cfg["training"])

    start = time.time()
    model.fit(X_train, y_train)
    train_seconds = round(time.time() - start, 2)
    logging.info("Training finished in %.2f seconds", train_seconds)

    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_val_prob = model.predict_proba(X_val)[:, 1]
    y_test_prob = model.predict_proba(X_test)[:, 1]

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
            "model_type": model_type,
            "train_seconds": train_seconds,
            "default_threshold": default_threshold,
            "best_val_threshold_for_f1": best_threshold,
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
                "model_type": model_type,
                "train_seconds": train_seconds,
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

    # Save only validation and test prediction files
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

    if cfg["plots"].get("save_pr_curve", True):
        save_pr_curve(y_train, y_train_prob, plots_dir / "train_pr_curve.png", f"{experiment_name} - Train PR Curve")
        save_pr_curve(y_val, y_val_prob, plots_dir / "val_pr_curve.png", f"{experiment_name} - Val PR Curve")
        save_pr_curve(y_test, y_test_prob, plots_dir / "test_pr_curve.png", f"{experiment_name} - Test PR Curve")

    if cfg["plots"].get("save_roc_curve", True):
        save_roc_curve(y_train, y_train_prob, plots_dir / "train_roc_curve.png", f"{experiment_name} - Train ROC Curve")
        save_roc_curve(y_val, y_val_prob, plots_dir / "val_roc_curve.png", f"{experiment_name} - Val ROC Curve")
        save_roc_curve(y_test, y_test_prob, plots_dir / "test_roc_curve.png", f"{experiment_name} - Test ROC Curve")

    if cfg["plots"].get("save_threshold_plot", True):
        save_threshold_plot(threshold_rows, plots_dir / "val_threshold_vs_f1.png", f"{experiment_name} - Val Threshold vs F1")

    if cfg["plots"].get("save_confusion_matrix", True):
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

    if cfg["plots"].get("save_feature_importance", True):
        importances = get_feature_importances(model, model_type)
        if importances is not None:
            save_feature_importance_plot(
                train_cols,
                importances,
                plots_dir / "feature_importance_top25.png",
                f"{experiment_name} - Top 25 Feature Importances",
                top_k=25,
            )

    save_model(model, model_type, models_dir)
    logging.info("Saved outputs to %s", output_dir)
    logging.info("Done.")


if __name__ == "__main__":
    main()
