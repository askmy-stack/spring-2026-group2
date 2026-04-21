#!/usr/bin/env python3
"""
Standardized TabNet training script.
Follows the same structure as train_model.py for classical ML models.

Improvements over baseline:
  - ReduceLROnPlateau scheduler (adaptive, monitors val AUCPR)
  - weight_decay in Adam optimizer
  - Gradient clipping
  - Tunable GBN momentum
  - Independent n_d / n_a
  - Early stopping on val AUCPR (matches Optuna objective)
  - Full utils integration (metric_utils, plot_utils, io_utils)
  - YAML config driven (no argparse sprawl)
"""

from __future__ import annotations

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
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from torch.utils.data import DataLoader, Dataset

# ── shared utils ─────────────────────────────────────────────────────────────
from src.models.utils.config_utils import load_config
from src.models.utils.io_utils import ensure_dir, save_csv, save_json
from src.models.utils.metric_utils import compute_binary_metrics, sweep_thresholds_for_f1
from src.models.utils.plot_utils import (
    save_confusion_matrix_plot,
    save_pr_curve,
    save_roc_curve,
    save_threshold_plot,
)

# ────────────────────────────────────────────────────────────────────────────


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

    feat_names_path = out_dir / "feature_names.json"
    label_map_path  = out_dir / "label_mapping.json"

    with open(feat_names_path, "w", encoding="utf-8") as f:
        _json.dump(feature_cols, f, indent=2)
    with open(label_map_path, "w", encoding="utf-8") as f:
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
        meta_path = memmap_dir / f"{split}_meta.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.rows = meta["rows"]
        self.n_features = meta["features"]

        self.X = np.memmap(meta["x_path"], dtype="float32", mode="r",
                           shape=(meta["rows"], meta["features"]))
        self.y = np.memmap(meta["y_path"], dtype="int64", mode="r",
                           shape=(meta["rows"],))

    def __len__(self) -> int:
        return self.rows

    def __getitem__(self, idx: int):
        x = np.array(self.X[idx], dtype=np.float32, copy=True)
        y = np.int64(self.y[idx])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def make_loader(ds: Dataset, batch_size: int, num_workers: int,
                shuffle: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def compute_class_weights(ds: MemmapDataset) -> tuple[torch.Tensor, list[int]]:
    counts = np.bincount(ds.y[:], minlength=2).astype(np.float64)
    total = counts.sum()
    w = total / (2.0 * counts)
    return torch.tensor(w, dtype=torch.float32), counts.astype(int).tolist()


# ── model builder ────────────────────────────────────────────────────────────

def build_classifier(cfg: dict, input_dim: int,
                     device_name: str) -> TabNetClassifier:
    mp = cfg["model_params"]
    op = cfg["optimizer_params"]

    clf = TabNetClassifier(
        input_dim=input_dim,
        output_dim=2,
        n_d=int(mp["n_d"]),
        n_a=int(mp["n_a"]),
        n_steps=int(mp["n_steps"]),
        gamma=float(mp["gamma"]),
        lambda_sparse=float(mp["lambda_sparse"]),
        momentum=float(mp.get("momentum", 0.02)),
        mask_type=str(mp.get("mask_type", "entmax")),
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(
            lr=float(op["lr"]),
            weight_decay=float(op.get("weight_decay", 1e-5)),
        ),
        device_name=device_name,
        verbose=0,
        seed=int(cfg["training"].get("random_state", 42)),
    )

    clf.classes_ = np.array([0, 1])
    clf.target_mapper = {0: 0, 1: 1}
    clf.preds_mapper = {0: 0, 1: 1}
    clf._default_loss = torch.nn.functional.cross_entropy
    clf.loss_fn = clf._default_loss
    clf._set_network()
    clf.network.virtual_batch_size = int(
        cfg["data"].get("virtual_batch_size", 128)
    )
    clf._set_optimizer()
    clf._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        clf._optimizer,
        mode=cfg["scheduler_params"].get("mode", "max"),
        factor=float(cfg["scheduler_params"].get("factor", 0.5)),
        patience=int(cfg["scheduler_params"].get("patience", 3)),
        min_lr=float(cfg["scheduler_params"].get("min_lr", 1e-6)),
    )
    return clf


# ── evaluation ───────────────────────────────────────────────────────────────

def evaluate_probs(clf: TabNetClassifier,
                   loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    clf.network.eval()
    y_true_list, y_prob_list = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(clf.device).float()
            logits, _ = clf.network(Xb)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            y_true_list.append(yb.numpy())
            y_prob_list.append(probs)
    return np.concatenate(y_true_list), np.concatenate(y_prob_list)


# ── training loop ────────────────────────────────────────────────────────────

def train(clf: TabNetClassifier, train_loader: DataLoader,
          val_loader: DataLoader, cfg: dict,
          models_dir: Path) -> tuple[list[dict], int]:

    max_epochs = int(cfg["training"]["max_epochs"])
    patience = int(cfg["training"]["patience"])
    grad_clip = float(cfg["training"].get("grad_clip", 1.0))
    lambda_sparse = float(cfg["model_params"]["lambda_sparse"])
    batch_size = int(cfg["data"]["batch_size"])
    vbs = int(cfg["data"].get("virtual_batch_size", 128))

    weight_tensor, class_counts = compute_class_weights(
        train_loader.dataset  # type: ignore
    )
    weight_tensor = weight_tensor.to(clf.device)
    logging.info("Class counts: %s  |  Class weights: %s",
                 class_counts, weight_tensor.tolist())

    best_val_aucpr = -math.inf
    best_epoch = -1
    epochs_no_improve = 0
    history: list[dict] = []

    for epoch in range(max_epochs):
        clf.network.train()
        running_loss = 0.0
        n_batches = 0

        for Xb, yb in train_loader:
            Xb = Xb.to(clf.device).float()
            yb = yb.to(clf.device).long()

            for p in clf.network.parameters():
                p.grad = None

            logits, M_loss = clf.network(Xb)
            loss = torch.nn.functional.cross_entropy(
                logits, yb, weight=weight_tensor
            ) + lambda_sparse * M_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(clf.network.parameters(), grad_clip)
            clf._optimizer.step()

            running_loss += float(loss.detach().cpu())
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)
        y_val, y_val_prob = evaluate_probs(clf, val_loader)
        val_metrics = compute_binary_metrics(
            y_val, y_val_prob,
            float(cfg["evaluation"].get("default_threshold", 0.5))
        )
        val_aucpr = val_metrics["aucpr"]

        # step scheduler on val AUCPR
        if hasattr(clf, "_scheduler") and clf._scheduler is not None:
            clf._scheduler.step(val_aucpr)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_aucpr": val_aucpr,
            "val_roc_auc": val_metrics["roc_auc"],
            "val_f1": val_metrics["f1"],
        })

        logging.info(
            "Epoch %02d | loss=%.6f | val_aucpr=%.6f | "
            "val_roc_auc=%.6f | val_f1=%.6f",
            epoch, train_loss, val_aucpr,
            val_metrics["roc_auc"], val_metrics["f1"],
        )

        if val_aucpr > best_val_aucpr:
            best_val_aucpr = val_aucpr
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                "network_state_dict": clf.network.state_dict(),
                "optimizer_state_dict": clf._optimizer.state_dict(),
                "epoch": epoch,
                "best_val_aucpr": best_val_aucpr,
            }, models_dir / "best_model.pt")
            logging.info("  >> New best model saved (val_aucpr=%.6f)", best_val_aucpr)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logging.info("Early stopping at epoch %d", epoch)
                break

    ckpt = torch.load(models_dir / "best_model.pt", map_location=clf.device, weights_only=False)
    clf.network.load_state_dict(ckpt["network_state_dict"])
    clf.network.eval()
    logging.info("Restored best model from epoch %d", best_epoch)
    return history, best_epoch


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)

    experiment_name = cfg["experiment_name"]
    logging.info("Experiment: %s", experiment_name)

    set_seed(int(cfg["training"].get("random_state", 42)))

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

    default_threshold = float(cfg["evaluation"].get("default_threshold", 0.5))
    batch_size   = int(cfg["data"]["batch_size"])
    num_workers  = int(cfg["data"].get("num_workers", 4))

    train_ds = MemmapDataset(memmap_dir, "train")
    val_ds   = MemmapDataset(memmap_dir, "val")
    test_ds  = MemmapDataset(memmap_dir, "test")

    logging.info("Train: %d rows | Val: %d rows | Test: %d rows | Features: %d",
                 len(train_ds), len(val_ds), len(test_ds), train_ds.n_features)

    train_loader = make_loader(train_ds, batch_size, num_workers, shuffle=True)
    val_loader   = make_loader(val_ds,   batch_size, num_workers, shuffle=False)
    test_loader  = make_loader(test_ds,  batch_size, num_workers, shuffle=False)

    clf = build_classifier(cfg, train_ds.n_features, device_name)

    t0 = time.time()
    history, best_epoch = train(clf, train_loader, val_loader, cfg, models_dir)
    train_seconds = round(time.time() - t0, 2)
    logging.info("Training finished in %.2f seconds", train_seconds)

    y_val,  y_val_prob  = evaluate_probs(clf, val_loader)
    y_test, y_test_prob = evaluate_probs(clf, test_loader)

    val_metrics_default  = compute_binary_metrics(y_val,  y_val_prob,  default_threshold)
    test_metrics_default = compute_binary_metrics(y_test, y_test_prob, default_threshold)

    best_threshold, threshold_rows = sweep_thresholds_for_f1(y_val, y_val_prob)
    val_metrics_best  = compute_binary_metrics(y_val,  y_val_prob,  best_threshold)
    test_metrics_best = compute_binary_metrics(y_test, y_test_prob, best_threshold)

    # ── save metrics ────────────────────────────────────────────────────────
    save_json({
        "experiment_name": experiment_name,
        "model_type": "tabnet",
        "train_seconds": train_seconds,
        "best_epoch": best_epoch,
        "default_threshold": default_threshold,
        "best_val_threshold_for_f1": best_threshold,
        "model_params": cfg["model_params"],
        "optimizer_params": cfg["optimizer_params"],
    }, metrics_dir / "run_info.json")

    save_json(val_metrics_default,  metrics_dir / "val_metrics_default.json")
    save_json(val_metrics_best,     metrics_dir / "val_metrics_best_threshold.json")
    save_json(test_metrics_default, metrics_dir / "test_metrics_default.json")
    save_json(test_metrics_best,    metrics_dir / "test_metrics_best_threshold.json")

    summary_df = pd.DataFrame([{
        "experiment_name":            experiment_name,
        "model_type":                 "tabnet",
        "train_seconds":              train_seconds,
        "best_epoch":                 best_epoch,
        "best_val_threshold_for_f1":  best_threshold,
        "val_aucpr_default":          val_metrics_default["aucpr"],
        "val_f1_default":             val_metrics_default["f1"],
        "val_aucpr_best_threshold":   val_metrics_best["aucpr"],
        "val_f1_best_threshold":      val_metrics_best["f1"],
        "val_roc_auc":                val_metrics_default["roc_auc"],
        "test_aucpr_best_threshold":  test_metrics_best["aucpr"],
        "test_f1_best_threshold":     test_metrics_best["f1"],
        "test_roc_auc":               test_metrics_best["roc_auc"],
        "test_precision_best":        test_metrics_best["precision"],
        "test_recall_best":           test_metrics_best["recall"],
    }])
    save_csv(summary_df, metrics_dir / "summary.csv")

    history_df = pd.DataFrame(history)
    save_csv(history_df, metrics_dir / "history.csv")

    save_csv(pd.DataFrame(threshold_rows), metrics_dir / "val_threshold_sweep.csv")

    # ── save predictions ────────────────────────────────────────────────────
    save_csv(pd.DataFrame({
        "y_true": y_val,
        "y_prob": y_val_prob,
        "y_pred_default":       (y_val_prob >= default_threshold).astype(int),
        "y_pred_best_threshold": (y_val_prob >= best_threshold).astype(int),
    }), preds_dir / "val_predictions.csv")

    save_csv(pd.DataFrame({
        "y_true": y_test,
        "y_prob": y_test_prob,
        "y_pred_default":       (y_test_prob >= default_threshold).astype(int),
        "y_pred_best_threshold": (y_test_prob >= best_threshold).astype(int),
    }), preds_dir / "test_predictions.csv")

    # ── save plots ──────────────────────────────────────────────────────────
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
            val_metrics_default["confusion_matrix"],
            plots_dir / "val_confusion_matrix_default.png",
            f"{experiment_name} - Val Confusion Matrix @ {default_threshold:.2f}",
        )
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
