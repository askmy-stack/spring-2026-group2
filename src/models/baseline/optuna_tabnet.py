#!/usr/bin/env python3
"""
Optuna hyperparameter tuning for TabNet.
Follows the same structure as optuna_lightgbm.py / optuna_xgboost.py.

Search space covers:
  - n_d, n_a (independently)
  - n_steps, gamma, lambda_sparse, momentum, mask_type
  - learning_rate, weight_decay
  - batch_size, virtual_batch_size
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import optuna
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from torch.utils.data import DataLoader, Dataset

from src.models.utils.config_utils import load_config
from src.models.utils.io_utils import ensure_dir, save_csv, save_json
from src.models.utils.metric_utils import compute_binary_metrics, sweep_thresholds_for_f1
from src.models.utils.plot_utils import (
    save_confusion_matrix_plot,
    save_pr_curve,
    save_roc_curve,
    save_threshold_plot,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def ensure_memmap(cfg: dict, chunksize: int = 50_000) -> None:
    """Build memmap files if they don't already exist."""
    from src.models.utils.prepare_memmap import (
        build_label_mapping,
        detect_feature_columns,
        write_split,
        already_exists,
    )
    import json as _json

    paths     = cfg["paths"]
    out_dir   = ensure_dir(Path(paths["memmap_dir"]).expanduser().resolve())
    target    = cfg["target_col"]
    train_csv = Path(paths["train_csv"]).expanduser().resolve()
    val_csv   = Path(paths["val_csv"]).expanduser().resolve()
    test_csv  = Path(paths["test_csv"]).expanduser().resolve()

    splits_needed = [
        s for s, p in [("train", train_csv), ("val", val_csv), ("test", test_csv)]
        if not already_exists(out_dir, s)
    ]

    if not splits_needed:
        logging.info("Memmap files already exist — skipping preparation.")
        return

    logging.info("Memmap files missing for splits: %s — building now...", splits_needed)

    feature_cols  = detect_feature_columns(train_csv, target)
    label_mapping = build_label_mapping(train_csv, target, chunksize)

    with open(out_dir / "feature_names.json", "w", encoding="utf-8") as f:
        _json.dump(feature_cols, f, indent=2)
    with open(out_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        _json.dump(label_mapping, f, indent=2)

    for split, csv_path in [("train", train_csv), ("val", val_csv), ("test", test_csv)]:
        if split in splits_needed:
            write_split(csv_path, split, feature_cols, label_mapping,
                        out_dir, target, chunksize)

    logging.info("Memmap preparation complete.")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── memmap dataset ───────────────────────────────────────────────────────────

class MemmapDataset(Dataset):
    def __init__(self, memmap_dir: Path, split: str):
        with open(memmap_dir / f"{split}_meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.rows       = meta["rows"]
        self.n_features = meta["features"]
        self.X = np.memmap(meta["x_path"], dtype="float32", mode="r",
                           shape=(meta["rows"], meta["features"]))
        self.y = np.memmap(meta["y_path"], dtype="int64",   mode="r",
                           shape=(meta["rows"],))

    def __len__(self) -> int:
        return self.rows

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(np.array(self.X[idx], dtype=np.float32, copy=True)),
            torch.tensor(np.int64(self.y[idx]), dtype=torch.long),
        )


def make_loader(ds: Dataset, batch_size: int, num_workers: int,
                shuffle: bool) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), drop_last=False)


def compute_class_weights(ds: MemmapDataset) -> torch.Tensor:
    counts = np.bincount(ds.y[:], minlength=2).astype(np.float64)
    w = counts.sum() / (2.0 * counts)
    return torch.tensor(w, dtype=torch.float32)


# ── model builder ────────────────────────────────────────────────────────────

def build_classifier(params: dict, input_dim: int,
                     device_name: str, seed: int) -> TabNetClassifier:
    clf = TabNetClassifier(
        input_dim=input_dim,
        output_dim=2,
        n_d=int(params["n_d"]),
        n_a=int(params["n_a"]),
        n_steps=int(params["n_steps"]),
        gamma=float(params["gamma"]),
        lambda_sparse=float(params["lambda_sparse"]),
        momentum=float(params["momentum"]),
        mask_type=str(params["mask_type"]),
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(
            lr=float(params["lr"]),
            weight_decay=float(params["weight_decay"]),
        ),
        device_name=device_name,
        verbose=0,
        seed=seed,
    )
    clf.classes_       = np.array([0, 1])
    clf.target_mapper  = {0: 0, 1: 1}
    clf.preds_mapper   = {0: 0, 1: 1}
    clf._default_loss  = torch.nn.functional.cross_entropy
    clf.loss_fn        = clf._default_loss
    clf._set_network()
    clf.network.virtual_batch_size = int(params["virtual_batch_size"])
    clf._set_optimizer()
    clf._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        clf._optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )
    return clf


# ── single training run ──────────────────────────────────────────────────────

def run_training(
    clf: TabNetClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    weight_tensor: torch.Tensor,
    lambda_sparse: float,
    grad_clip: float,
    max_epochs: int,
    patience: int,
    default_threshold: float,
    models_dir: Path | None = None,
    save_best: bool = False,
) -> tuple[float, int, list[dict]]:
    """Train and return (best_val_aucpr, best_epoch, history)."""

    best_val_aucpr  = -math.inf
    best_epoch      = -1
    epochs_no_improve = 0
    history: list[dict] = []
    w = weight_tensor.to(clf.device)

    for epoch in range(max_epochs):
        clf.network.train()
        running_loss = 0.0
        n_batches    = 0

        for Xb, yb in train_loader:
            Xb = Xb.to(clf.device).float()
            yb = yb.to(clf.device).long()
            for p in clf.network.parameters():
                p.grad = None
            logits, M_loss = clf.network(Xb)
            loss = torch.nn.functional.cross_entropy(logits, yb, weight=w) \
                   - lambda_sparse * M_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(clf.network.parameters(), grad_clip)
            clf._optimizer.step()
            running_loss += float(loss.detach().cpu())
            n_batches    += 1

        train_loss = running_loss / max(n_batches, 1)

        clf.network.eval()
        y_true_list, y_prob_list = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                logits, _ = clf.network(Xb.to(clf.device).float())
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                y_true_list.append(yb.numpy())
                y_prob_list.append(probs)
        y_val      = np.concatenate(y_true_list)
        y_val_prob = np.concatenate(y_prob_list)
        val_m      = compute_binary_metrics(y_val, y_val_prob, default_threshold)
        val_aucpr  = val_m["aucpr"]

        if hasattr(clf, "_scheduler") and clf._scheduler is not None:
            clf._scheduler.step(val_aucpr)

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_aucpr": val_aucpr, "val_roc_auc": val_m["roc_auc"],
            "val_f1": val_m["f1"],
        })

        if val_aucpr > best_val_aucpr:
            best_val_aucpr    = val_aucpr
            best_epoch        = epoch
            epochs_no_improve = 0
            if save_best and models_dir is not None:
                torch.save({
                    "network_state_dict":    clf.network.state_dict(),
                    "optimizer_state_dict":  clf._optimizer.state_dict(),
                    "epoch":                 epoch,
                    "best_val_aucpr":        best_val_aucpr,
                }, models_dir / "best_model.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    return best_val_aucpr, best_epoch, history


# ── optuna search space ──────────────────────────────────────────────────────

def build_trial_params(trial: optuna.Trial) -> dict:
    n_d = trial.suggest_categorical("n_d", [16, 32, 64, 128, 256])
    n_a = trial.suggest_categorical("n_a", [16, 32, 64, 128, 256])
    return {
        "n_d":              n_d,
        "n_a":              n_a,
        "n_steps":          trial.suggest_int("n_steps", 3, 10),
        "gamma":            trial.suggest_float("gamma", 1.0, 2.0),
        "lambda_sparse":    trial.suggest_float("lambda_sparse", 1e-5, 1e-1, log=True),
        "momentum":         trial.suggest_float("momentum", 0.01, 0.4),
        "mask_type":        trial.suggest_categorical("mask_type", ["sparsemax", "entmax"]),
        "lr":               trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay":     trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size":       trial.suggest_categorical("batch_size", [512, 1024, 2048]),
        "virtual_batch_size": trial.suggest_categorical(
            "virtual_batch_size", [64, 128, 256]
        ),
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)

    experiment_name   = cfg["experiment_name"]
    logging.info("Experiment: %s", experiment_name)

    seed = int(cfg["training"].get("random_state", 42))
    set_seed(seed)

    # build memmap if needed — must happen before dataset loading
    ensure_memmap(cfg)

    device_name = (
        "cuda" if cfg["training"].get("use_gpu", True)
        and torch.cuda.is_available() else "cpu"
    )
    logging.info("Device: %s", device_name)

    memmap_dir = Path(cfg["paths"]["memmap_dir"]).expanduser().resolve()
    output_dir = ensure_dir(cfg["paths"]["output_dir"])
    metrics_dir = ensure_dir(output_dir / "metrics")
    plots_dir   = ensure_dir(output_dir / "plots")
    preds_dir   = ensure_dir(output_dir / "predictions")
    models_dir  = ensure_dir(output_dir / "models")
    study_dir   = ensure_dir(output_dir / "study")

    default_threshold = float(cfg["evaluation"].get("default_threshold", 0.5))
    max_epochs        = int(cfg["training"]["max_epochs"])
    patience          = int(cfg["training"]["patience"])
    grad_clip         = float(cfg["training"].get("grad_clip", 1.0))
    n_trials          = int(cfg["optuna"]["n_trials"])
    study_name        = cfg["optuna"].get("study_name", experiment_name)
    num_workers       = int(cfg["data"].get("num_workers", 4))

    train_ds = MemmapDataset(memmap_dir, "train")
    val_ds   = MemmapDataset(memmap_dir, "val")
    test_ds  = MemmapDataset(memmap_dir, "test")

    logging.info("Train: %d | Val: %d | Test: %d | Features: %d",
                 len(train_ds), len(val_ds), len(test_ds), train_ds.n_features)

    weight_tensor = compute_class_weights(train_ds)

    # ── Optuna objective ───────────────────────────────────────────���─────────
    def objective(trial: optuna.Trial) -> float:
        params = build_trial_params(trial)
        train_loader = make_loader(train_ds, params["batch_size"], num_workers, True)
        val_loader   = make_loader(val_ds,   params["batch_size"], num_workers, False)
        clf = build_classifier(params, train_ds.n_features, device_name, seed)
        val_aucpr, _, _ = run_training(
            clf, train_loader, val_loader, weight_tensor,
            float(params["lambda_sparse"]), grad_clip,
            max_epochs, patience, default_threshold,
        )
        return val_aucpr

    logging.info("Starting Optuna with %d trials...", n_trials)
    t0 = time.time()

    study = optuna.create_study(study_name=study_name, direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    study_seconds = round(time.time() - t0, 2)
    logging.info("Optuna finished in %.2f seconds", study_seconds)
    logging.info("Best val AUCPR: %.6f", study.best_value)
    logging.info("Best params: %s", study.best_params)

    # merge best trial params with base defaults
    best_params = {
        "n_d":               64,
        "n_a":               64,
        "n_steps":           5,
        "gamma":             1.2,
        "lambda_sparse":     1e-3,
        "momentum":          0.02,
        "mask_type":         "entmax",
        "lr":                5e-3,
        "weight_decay":      1e-5,
        "batch_size":        1024,
        "virtual_batch_size": 128,
    }
    best_params.update(study.best_params)

    save_json({
        "study_name":          study_name,
        "direction":           "maximize",
        "n_trials":            n_trials,
        "study_seconds":       study_seconds,
        "best_value_val_aucpr": study.best_value,
        "best_params":         best_params,
    }, study_dir / "best_study_result.json")

    save_csv(study.trials_dataframe(), study_dir / "optuna_trials.csv")

    # ── retrain best model ───────────────────────────────────────────────────
    logging.info("Retraining best TabNet on full train split...")
    batch_size = int(best_params["batch_size"])
    train_loader = make_loader(train_ds, batch_size, num_workers, True)
    val_loader   = make_loader(val_ds,   batch_size, num_workers, False)
    test_loader  = make_loader(test_ds,  batch_size, num_workers, False)

    clf = build_classifier(best_params, train_ds.n_features, device_name, seed)

    t1 = time.time()
    best_val_aucpr, best_epoch, history = run_training(
        clf, train_loader, val_loader, weight_tensor,
        float(best_params["lambda_sparse"]), grad_clip,
        max_epochs, patience, default_threshold,
        models_dir=models_dir, save_best=True,
    )
    train_seconds = round(time.time() - t1, 2)
    logging.info("Retrain finished in %.2f seconds", train_seconds)

    # restore best checkpoint
    ckpt = torch.load(models_dir / "best_model.pt", map_location=device_name)
    clf.network.load_state_dict(ckpt["network_state_dict"])
    clf.network.eval()

    # ── evaluate ─────────────────────────────────────────────────────────────
    def eval_probs(loader):
        y_t, y_p = [], []
        with torch.no_grad():
            for Xb, yb in loader:
                logits, _ = clf.network(Xb.to(clf.device).float())
                y_t.append(yb.numpy())
                y_p.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        return np.concatenate(y_t), np.concatenate(y_p)

    y_val,  y_val_prob  = eval_probs(val_loader)
    y_test, y_test_prob = eval_probs(test_loader)

    val_metrics_default  = compute_binary_metrics(y_val,  y_val_prob,  default_threshold)
    test_metrics_default = compute_binary_metrics(y_test, y_test_prob, default_threshold)

    best_threshold, threshold_rows = sweep_thresholds_for_f1(y_val, y_val_prob)
    val_metrics_best  = compute_binary_metrics(y_val,  y_val_prob,  best_threshold)
    test_metrics_best = compute_binary_metrics(y_test, y_test_prob, best_threshold)

    # ── save metrics ─────────────────────────────────────────────────────────
    save_json({
        "experiment_name":            experiment_name,
        "model_type":                 "tabnet",
        "train_seconds":              train_seconds,
        "study_seconds":              study_seconds,
        "best_epoch":                 best_epoch,
        "default_threshold":          default_threshold,
        "best_val_threshold_for_f1":  best_threshold,
        "best_params":                best_params,
    }, metrics_dir / "run_info.json")

    save_json(val_metrics_default,  metrics_dir / "val_metrics_default.json")
    save_json(val_metrics_best,     metrics_dir / "val_metrics_best_threshold.json")
    save_json(test_metrics_default, metrics_dir / "test_metrics_default.json")
    save_json(test_metrics_best,    metrics_dir / "test_metrics_best_threshold.json")

    summary_df = pd.DataFrame([{
        "experiment_name":             experiment_name,
        "model_type":                  "tabnet",
        "train_seconds":               train_seconds,
        "study_seconds":               study_seconds,
        "best_val_aucpr_optuna":       study.best_value,
        "best_epoch":                  best_epoch,
        "best_val_threshold_for_f1":   best_threshold,
        "val_aucpr_default":           val_metrics_default["aucpr"],
        "val_f1_default":              val_metrics_default["f1"],
        "val_aucpr_best_threshold":    val_metrics_best["aucpr"],
        "val_f1_best_threshold":       val_metrics_best["f1"],
        "val_roc_auc":                 val_metrics_default["roc_auc"],
        "test_aucpr_best_threshold":   test_metrics_best["aucpr"],
        "test_f1_best_threshold":      test_metrics_best["f1"],
        "test_roc_auc":                test_metrics_best["roc_auc"],
        "test_precision_best":         test_metrics_best["precision"],
        "test_recall_best":            test_metrics_best["recall"],
    }])
    save_csv(summary_df, metrics_dir / "summary.csv")
    save_csv(pd.DataFrame(history),       metrics_dir / "history.csv")
    save_csv(pd.DataFrame(threshold_rows), metrics_dir / "val_threshold_sweep.csv")

    # ── save predictions ──────────────────────────────────────────────────────
    save_csv(pd.DataFrame({
        "y_true": y_val, "y_prob": y_val_prob,
        "y_pred_default":        (y_val_prob >= default_threshold).astype(int),
        "y_pred_best_threshold": (y_val_prob >= best_threshold).astype(int),
    }), preds_dir / "val_predictions.csv")

    save_csv(pd.DataFrame({
        "y_true": y_test, "y_prob": y_test_prob,
        "y_pred_default":        (y_test_prob >= default_threshold).astype(int),
        "y_pred_best_threshold": (y_test_prob >= best_threshold).astype(int),
    }), preds_dir / "test_predictions.csv")

    # ── save plots ────────────────────────────────────────────────────────────
    plot_cfg = cfg.get("plots", {})

    if plot_cfg.get("save_pr_curve", True):
        save_pr_curve(y_val,  y_val_prob,  plots_dir / "val_pr_curve.png",
                      f"{experiment_name} - Val PR Curve")
        save_pr_curve(y_test, y_test_prob, plots_dir / "test_pr_curve.png",
                      f"{experiment_name} - Test PR Curve")

    if plot_cfg.get("save_roc_curve", True):
        save_roc_curve(y_val,  y_val_prob,  plots_dir / "val_roc_curve.png",
                       f"{experiment_name} - Val ROC Curve")
        save_roc_curve(y_test, y_test_prob, plots_dir / "test_roc_curve.png",
                       f"{experiment_name} - Test ROC Curve")

    if plot_cfg.get("save_threshold_plot", True):
        save_threshold_plot(threshold_rows, plots_dir / "val_threshold_vs_f1.png",
                            f"{experiment_name} - Val Threshold vs F1")

    if plot_cfg.get("save_confusion_matrix", True):
        save_confusion_matrix_plot(
            val_metrics_best["confusion_matrix"],
            plots_dir / "val_confusion_matrix_best_threshold.png",
            f"{experiment_name} - Val Confusion Matrix @ {best_threshold:.2f}",
        )
        save_confusion_matrix_plot(
            test_metrics_best["confusion_matrix"],
            plots_dir / "test_confusion_matrix_best_threshold.png",
            f"{experiment_name} - Test Confusion Matrix @ {best_threshold:.2f}",
        )

    logging.info("Saved all outputs to %s", output_dir)
    logging.info("Done.")


if __name__ == "__main__":
    main()
