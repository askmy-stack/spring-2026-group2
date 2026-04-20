#!/usr/bin/env python3
"""
Advanced TabNet training script with two novel EEG-aware architectures.

ECT-TabNet  (architecture_type: ect)
  EEG Channel Transformer + TabNet.
  Reshapes the 528-feature input as 16 channel tokens (batch, 16, 33),
  applies multi-head self-attention to capture inter-channel seizure
  propagation dynamics, concatenates the pooled transformer output with
  the original features, and feeds the augmented vector into TabNet.

H-TabNet  (architecture_type: hier)
  Hierarchical TabNet with squeeze-excitation channel attention.
  A shared encoder maps each channel's 33 features to a d_ch-dimensional
  embedding, SE-attention reweights channels by importance, and the
  resulting channel representations are concatenated with original features
  before TabNet.

Both architectures:
  - Focal loss  (directly addresses seizure-window imbalance)
  - Class-weighted loss
  - Gradient clipping, ReduceLROnPlateau, early stopping on val AUCPR
  - Full utils integration (metric_utils, plot_utils, io_utils)

Usage:
    python tabnet/train_tabnet_advanced.py --config configs/tabnet_ect.yaml
    python tabnet/train_tabnet_advanced.py --config configs/tabnet_hier.yaml
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
import torch.nn as nn
import torch.nn.functional as F
from pytorch_tabnet.tab_network import TabNetNoEmbeddings
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


# ── memmap helper ─────────────────────────────────────────────────────────────

def ensure_memmap(cfg: dict, chunksize: int = 50_000) -> None:
    from src.models.utils.prepare_memmap import (
        build_label_mapping, detect_feature_columns,
        write_split, already_exists,
    )
    import json as _json

    paths   = cfg["paths"]
    out_dir = ensure_dir(Path(paths["memmap_dir"]).expanduser().resolve())
    target  = cfg["target_col"]
    train_csv = Path(paths["train_csv"]).expanduser().resolve()
    val_csv   = Path(paths["val_csv"]).expanduser().resolve()
    test_csv  = Path(paths["test_csv"]).expanduser().resolve()

    splits_needed = [
        s for s, p in [("train", train_csv), ("val", val_csv), ("test", test_csv)]
        if not already_exists(out_dir, s)
    ]
    if not splits_needed:
        logging.info("Memmap files already exist — skipping.")
        return

    logging.info("Building memmap for splits: %s", splits_needed)
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


# ── dataset ───────────────────────────────────────────────────────────────────

class MemmapDataset(Dataset):
    def __init__(self, memmap_dir: Path, split: str):
        with open(memmap_dir / f"{split}_meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.rows       = meta["rows"]
        self.n_features = meta["features"]
        self.X = np.memmap(meta["x_path"], dtype="float32", mode="r",
                           shape=(meta["rows"], meta["features"]))
        self.y = np.memmap(meta["y_path"], dtype="int64", mode="r",
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
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), drop_last=False,
    )


def compute_class_weights(ds: MemmapDataset) -> tuple[torch.Tensor, list[int]]:
    counts = np.bincount(ds.y[:], minlength=2).astype(np.float64)
    w = counts.sum() / (2.0 * counts)
    return torch.tensor(w, dtype=torch.float32), counts.astype(int).tolist()


# ── focal loss ────────────────────────────────────────────────────────────────

def focal_loss(logits: torch.Tensor, targets: torch.Tensor,
               weight: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """
    Focal loss with per-class weighting.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    ce = F.cross_entropy(logits, targets, weight=weight, reduction="none")
    pt = torch.exp(-ce)
    return ((1.0 - pt) ** gamma * ce).mean()


# ── Architecture 1: ECT-TabNet ────────────────────────────────────────────────

class ChannelTransformerEncoder(nn.Module):
    """
    Reshapes (batch, C*F) → (batch, C, F), applies multi-head self-attention
    over C EEG channel tokens, returns mean-pooled representation (batch, d_model).

    Learnable positional embeddings allow the transformer to distinguish
    channels by their spatial position on the scalp.
    """

    def __init__(self, n_channels: int, n_features: int, d_model: int,
                 n_heads: int, n_layers: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_features = n_features

        self.projection = nn.Linear(n_features, d_model)
        self.pos_embed  = nn.Parameter(torch.zeros(1, n_channels, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm        = nn.LayerNorm(d_model)
        self.output_dim  = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x_ch   = x.view(b, self.n_channels, self.n_features)  # (b, C, F)
        tokens = self.projection(x_ch) + self.pos_embed        # (b, C, d_model)
        out    = self.transformer(tokens)                       # (b, C, d_model)
        return self.norm(out.mean(dim=1))                       # (b, d_model)


class ECTTabNet(nn.Module):
    """
    EEG Channel Transformer + TabNet (ECT-TabNet).

    Forward:
      original (batch, D)
      + transformer(x) → (batch, d_model)
      → concat (batch, D + d_model)
      → TabNetNoEmbeddings
    """

    def __init__(
        self, input_dim: int,
        n_channels: int, n_features_per_ch: int,
        d_model: int, n_heads: int, n_layers: int, dropout: float,
        n_d: int, n_a: int, n_steps: int, gamma_tab: float,
        momentum: float, mask_type: str, virtual_batch_size: int,
    ) -> None:
        super().__init__()
        self.channel_transformer = ChannelTransformerEncoder(
            n_channels, n_features_per_ch, d_model, n_heads, n_layers, dropout,
        )
        self.tabnet = TabNetNoEmbeddings(
            input_dim=input_dim + d_model,
            output_dim=2,
            n_d=n_d, n_a=n_a, n_steps=n_steps, gamma=gamma_tab,
            n_independent=2, n_shared=2, epsilon=1e-15,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum, mask_type=mask_type,
        )

    def forward(self, x: torch.Tensor):
        ctx    = self.channel_transformer(x)          # (b, d_model)
        x_aug  = torch.cat([x, ctx], dim=1)           # (b, D + d_model)
        return self.tabnet(x_aug)                     # logits, M_loss


# ── Architecture 2: H-TabNet ─────────────────────────────────────────────────

class ChannelHierarchyEncoder(nn.Module):
    """
    Shared encoder + squeeze-excitation channel attention.

    Level 1 — Shared encoder (same weights applied to each channel's F features):
      (batch, C, F) → (batch, C, d_ch)

    Level 2 — Squeeze-excitation attention over channels:
      Global avg pool → FC squeeze → ReLU → FC excite → Sigmoid → scale channels

    Output: (batch, C * d_ch) flattened channel embeddings.
    """

    def __init__(self, n_channels: int, n_features: int,
                 d_ch: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_features = n_features
        self.d_ch       = d_ch

        # Shared encoder applied independently to each channel
        self.encoder = nn.Sequential(
            nn.Linear(n_features, d_ch * 2),
            nn.LayerNorm(d_ch * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ch * 2, d_ch),
            nn.LayerNorm(d_ch),
        )

        # Squeeze-excitation over the C channel dimension
        r = max(n_channels // 2, 4)
        self.se = nn.Sequential(
            nn.Linear(n_channels, r),
            nn.ReLU(),
            nn.Linear(r, n_channels),
            nn.Sigmoid(),
        )
        self.output_dim = n_channels * d_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b    = x.shape[0]
        x_ch = x.view(b, self.n_channels, self.n_features)  # (b, C, F)
        enc  = self.encoder(x_ch)                            # (b, C, d_ch)
        # SE: squeeze over d_ch → (b, C) → excite → (b, C, 1)
        se_w = self.se(enc.mean(dim=-1)).unsqueeze(-1)       # (b, C, 1)
        out  = enc * se_w                                    # (b, C, d_ch)
        return out.reshape(b, self.output_dim)               # (b, C*d_ch)


class HTabNet(nn.Module):
    """
    Hierarchical TabNet (H-TabNet).

    Forward:
      original (batch, D)
      + hierarchy(x) → (batch, C * d_ch)
      → concat (batch, D + C*d_ch)
      → TabNetNoEmbeddings
    """

    def __init__(
        self, input_dim: int,
        n_channels: int, n_features_per_ch: int,
        d_ch: int, dropout: float,
        n_d: int, n_a: int, n_steps: int, gamma_tab: float,
        momentum: float, mask_type: str, virtual_batch_size: int,
    ) -> None:
        super().__init__()
        self.hierarchy = ChannelHierarchyEncoder(
            n_channels, n_features_per_ch, d_ch, dropout,
        )
        self.tabnet = TabNetNoEmbeddings(
            input_dim=input_dim + self.hierarchy.output_dim,
            output_dim=2,
            n_d=n_d, n_a=n_a, n_steps=n_steps, gamma=gamma_tab,
            n_independent=2, n_shared=2, epsilon=1e-15,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum, mask_type=mask_type,
        )

    def forward(self, x: torch.Tensor):
        ch_repr = self.hierarchy(x)                    # (b, C*d_ch)
        x_aug   = torch.cat([x, ch_repr], dim=1)       # (b, D + C*d_ch)
        return self.tabnet(x_aug)                      # logits, M_loss


# ── model factory ─────────────────────────────────────────────────────────────

def build_model(cfg: dict, input_dim: int, device: torch.device) -> nn.Module:
    arch     = cfg.get("architecture_type", "ect")
    mp       = cfg["model_params"]
    arch_cfg = cfg.get("architecture", {})
    n_ch     = int(arch_cfg.get("n_channels", 16))
    n_f_ch   = int(arch_cfg.get("n_features_per_ch", 33))
    dropout  = float(arch_cfg.get("dropout", 0.1))
    vbs      = int(cfg["data"].get("virtual_batch_size", 128))

    if arch == "ect":
        model = ECTTabNet(
            input_dim=input_dim,
            n_channels=n_ch, n_features_per_ch=n_f_ch,
            d_model=int(arch_cfg.get("d_model", 64)),
            n_heads=int(arch_cfg.get("n_heads", 4)),
            n_layers=int(arch_cfg.get("n_transformer_layers", 2)),
            dropout=dropout,
            n_d=int(mp["n_d"]), n_a=int(mp["n_a"]),
            n_steps=int(mp["n_steps"]), gamma_tab=float(mp["gamma"]),
            momentum=float(mp.get("momentum", 0.02)),
            mask_type=str(mp.get("mask_type", "entmax")),
            virtual_batch_size=vbs,
        )
    elif arch == "hier":
        model = HTabNet(
            input_dim=input_dim,
            n_channels=n_ch, n_features_per_ch=n_f_ch,
            d_ch=int(arch_cfg.get("d_ch", 32)),
            dropout=dropout,
            n_d=int(mp["n_d"]), n_a=int(mp["n_a"]),
            n_steps=int(mp["n_steps"]), gamma_tab=float(mp["gamma"]),
            momentum=float(mp.get("momentum", 0.02)),
            mask_type=str(mp.get("mask_type", "entmax")),
            virtual_batch_size=vbs,
        )
    else:
        raise ValueError(f"Unknown architecture_type: '{arch}'. Use 'ect' or 'hier'.")

    model = model.to(device)
    # pytorch_tabnet's group_attention_matrix is not always registered as a
    # proper buffer — force-move it to the target device after .to()
    for module in model.modules():
        if hasattr(module, "group_attention_matrix"):
            module.group_attention_matrix = module.group_attention_matrix.to(device)
    return model


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate_probs(model: nn.Module, loader: DataLoader,
                   device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true_list, y_prob_list = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device).float()
            logits, _ = model(Xb)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            y_true_list.append(yb.numpy())
            y_prob_list.append(probs)
    return np.concatenate(y_true_list), np.concatenate(y_prob_list)


# ── training loop ─────────────────────────────────────────────────────────────

def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    models_dir: Path,
    weight_tensor: torch.Tensor,
    device: torch.device,
) -> tuple[list[dict], int]:

    max_epochs    = int(cfg["training"]["max_epochs"])
    patience      = int(cfg["training"]["patience"])
    grad_clip     = float(cfg["training"].get("grad_clip", 1.0))
    lambda_sparse = float(cfg["model_params"]["lambda_sparse"])
    focal_gamma   = float(cfg.get("loss", {}).get("gamma", 2.0))
    default_thr   = float(cfg["evaluation"].get("default_threshold", 0.5))

    w = weight_tensor.to(device)
    best_val_aucpr    = -math.inf
    best_epoch        = -1
    epochs_no_improve = 0
    history: list[dict] = []

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        n_batches    = 0

        for Xb, yb in train_loader:
            Xb = Xb.to(device).float()
            yb = yb.to(device).long()
            for p in model.parameters():
                p.grad = None
            logits, M_loss = model(Xb)
            loss = focal_loss(logits, yb, w, focal_gamma) - lambda_sparse * M_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            running_loss += float(loss.detach().cpu())
            n_batches    += 1

        train_loss = running_loss / max(n_batches, 1)
        y_val, y_val_prob = evaluate_probs(model, val_loader, device)
        val_m    = compute_binary_metrics(y_val, y_val_prob, default_thr)
        val_aucpr = val_m["aucpr"]
        scheduler.step(val_aucpr)

        history.append({
            "epoch":       epoch,
            "train_loss":  train_loss,
            "val_aucpr":   val_aucpr,
            "val_roc_auc": val_m["roc_auc"],
            "val_f1":      val_m["f1"],
        })
        logging.info(
            "Epoch %02d | loss=%.6f | val_aucpr=%.6f | "
            "val_roc_auc=%.6f | val_f1=%.6f",
            epoch, train_loss, val_aucpr, val_m["roc_auc"], val_m["f1"],
        )

        if val_aucpr > best_val_aucpr:
            best_val_aucpr    = val_aucpr
            best_epoch        = epoch
            epochs_no_improve = 0
            torch.save({
                "model_state_dict":    model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch":               epoch,
                "best_val_aucpr":      best_val_aucpr,
            }, models_dir / "best_model.pt")
            logging.info("  >> New best (val_aucpr=%.6f)", best_val_aucpr)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logging.info("Early stopping at epoch %d", epoch)
                break

    ckpt = torch.load(models_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logging.info("Restored best model from epoch %d", best_epoch)
    return history, best_epoch


# ── main ─────────────────────────────────────────────────────────────────────

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


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)

    experiment_name = cfg["experiment_name"]
    arch_type       = cfg.get("architecture_type", "ect")
    logging.info("Experiment: %s  |  Architecture: %s", experiment_name, arch_type)

    set_seed(int(cfg["training"].get("random_state", 42)))
    ensure_memmap(cfg)

    device = torch.device(
        "cuda" if cfg["training"].get("use_gpu", True)
        and torch.cuda.is_available() else "cpu"
    )
    logging.info("Device: %s", device)

    memmap_dir = Path(cfg["paths"]["memmap_dir"]).expanduser().resolve()
    output_dir = ensure_dir(cfg["paths"]["output_dir"])
    metrics_dir = ensure_dir(output_dir / "metrics")
    plots_dir   = ensure_dir(output_dir / "plots")
    preds_dir   = ensure_dir(output_dir / "predictions")
    models_dir  = ensure_dir(output_dir / "models")

    batch_size        = int(cfg["data"]["batch_size"])
    num_workers       = int(cfg["data"].get("num_workers", 4))
    default_threshold = float(cfg["evaluation"].get("default_threshold", 0.5))

    train_ds = MemmapDataset(memmap_dir, "train")
    val_ds   = MemmapDataset(memmap_dir, "val")
    test_ds  = MemmapDataset(memmap_dir, "test")
    logging.info("Train: %d | Val: %d | Test: %d | Features: %d",
                 len(train_ds), len(val_ds), len(test_ds), train_ds.n_features)

    weight_tensor, class_counts = compute_class_weights(train_ds)
    logging.info("Class counts: %s | Weights: %s",
                 class_counts, weight_tensor.tolist())

    train_loader = make_loader(train_ds, batch_size, num_workers, True)
    val_loader   = make_loader(val_ds,   batch_size, num_workers, False)
    test_loader  = make_loader(test_ds,  batch_size, num_workers, False)

    model = build_model(cfg, train_ds.n_features, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Trainable parameters: %d", n_params)

    op = cfg["optimizer_params"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(op["lr"]),
        weight_decay=float(op.get("weight_decay", 1e-5)),
    )
    sp = cfg["scheduler_params"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=sp.get("mode", "max"),
        factor=float(sp.get("factor", 0.5)),
        patience=int(sp.get("patience", 3)),
        min_lr=float(sp.get("min_lr", 1e-6)),
    )

    t0 = time.time()
    history, best_epoch = train(
        model, optimizer, scheduler,
        train_loader, val_loader,
        cfg, models_dir, weight_tensor, device,
    )
    train_seconds = round(time.time() - t0, 2)
    logging.info("Training done in %.2fs", train_seconds)

    # ── evaluation ───────────────────────────────────────────────────────────
    y_val,  y_val_prob  = evaluate_probs(model, val_loader,  device)
    y_test, y_test_prob = evaluate_probs(model, test_loader, device)

    val_metrics_default  = compute_binary_metrics(y_val,  y_val_prob,  default_threshold)
    test_metrics_default = compute_binary_metrics(y_test, y_test_prob, default_threshold)

    best_threshold, threshold_rows = sweep_thresholds_for_f1(y_val, y_val_prob)
    val_metrics_best  = compute_binary_metrics(y_val,  y_val_prob,  best_threshold)
    test_metrics_best = compute_binary_metrics(y_test, y_test_prob, best_threshold)

    # ── save metrics ─────────────────────────────────────────────────────────
    save_json({
        "experiment_name":            experiment_name,
        "architecture_type":          arch_type,
        "train_seconds":              train_seconds,
        "best_epoch":                 best_epoch,
        "n_params":                   n_params,
        "default_threshold":          default_threshold,
        "best_val_threshold_for_f1":  best_threshold,
        "model_params":               cfg["model_params"],
        "architecture":               cfg.get("architecture", {}),
        "optimizer_params":           cfg["optimizer_params"],
        "loss":                       cfg.get("loss", {}),
    }, metrics_dir / "run_info.json")

    save_json(val_metrics_default,  metrics_dir / "val_metrics_default.json")
    save_json(val_metrics_best,     metrics_dir / "val_metrics_best_threshold.json")
    save_json(test_metrics_default, metrics_dir / "test_metrics_default.json")
    save_json(test_metrics_best,    metrics_dir / "test_metrics_best_threshold.json")

    summary_df = pd.DataFrame([{
        "experiment_name":            experiment_name,
        "architecture_type":          arch_type,
        "n_params":                   n_params,
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
    save_csv(pd.DataFrame(history), metrics_dir / "history.csv")
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
            f"{experiment_name} - Val CM @ {best_threshold:.2f}",
        )
        save_confusion_matrix_plot(
            test_metrics_best["confusion_matrix"],
            plots_dir / "test_confusion_matrix_best_threshold.png",
            f"{experiment_name} - Test CM @ {best_threshold:.2f}",
        )

    logging.info("Saved all outputs to %s", output_dir)
    logging.info("Done.")


if __name__ == "__main__":
    main()
