"""
Improved LSTM Training — Directory 2 of 4.

All hyperparameters loaded from src/models/config.yaml.

Usage:
    python -m src.models.improved_lstm_models.train \
        --data_path src/data/processed/chbmit
"""
import argparse
import logging
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.utils.losses import FocalLoss
from src.models.utils.metrics import (
    compute_f1_score, compute_auc_roc, compute_sensitivity,
    compute_specificity, find_optimal_threshold,
)
from src.models.utils.callbacks import EarlyStopping, clip_gradients
from src.models.utils.config_validator import validate_config
from src.models.utils.checkpoint import save_checkpoint
from src.models.lstm_benchmark_models.train_baseline import (
    load_config, _get_device, _build_data_loaders,
)
from .augmentation import EEGAugmentation, augment_batch
from .architectures.hierarchical_lstm import HierarchicalLSTM

logger = logging.getLogger(__name__)


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Linear warmup followed by cosine annealing LR schedule.

    Args:
        optimizer: Wrapped optimizer.
        warmup_epochs: Number of linear warmup epochs.
        total_epochs: Total training epochs (warmup + cosine).
        last_epoch: Index of last epoch (default: -1).

    Example:
        >>> sched = WarmupCosineScheduler(optimizer, warmup_epochs=5, total_epochs=100)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute per-group LR for current epoch."""
        if self.last_epoch < self.warmup_epochs:
            return [
                base_lr * (self.last_epoch + 1) / max(self.warmup_epochs, 1)
                for base_lr in self.base_lrs
            ]
        cosine_progress = (self.last_epoch - self.warmup_epochs) / max(
            self.total_epochs - self.warmup_epochs, 1
        )
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * cosine_progress))
        return [base_lr * cosine_factor for base_lr in self.base_lrs]


def train_improved(data_path: Path, config: Dict) -> Dict:
    """
    Train improved LSTM with augmentation; return test metrics.

    Args:
        data_path: Directory with train/, val/, test/ tensor splits.
        config: Config dict from config.yaml.

    Returns:
        Dict with f1, auc_roc, sensitivity on test split.
    """
    validate_config(config, model_section="improved_lstm")
    device = _get_device()
    model = _build_improved_model(config).to(device)
    train_loader, val_loader, test_loader = _build_data_loaders(data_path, config)
    augmenter = EEGAugmentation(p=config["training"].get("aug_prob", 0.5))
    criterion = _build_criterion(config, device)
    optimizer = _build_optimizer(model, config)
    scheduler = _build_scheduler(optimizer, config)
    ckpt_dir = Path(config["outputs"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    data_cfg = config["data"]
    stopper = EarlyStopping(
        patience=config["models"]["improved_lstm"].get(
            "early_stopping_patience", config["training"]["early_stopping_patience"]
        ),
        checkpoint_path=ckpt_dir / "improved_lstm_best.pt",
        model_config=_model_config(config),
        input_spec={"channels": data_cfg["n_channels"], "time_steps": data_cfg["time_steps"]},
        preprocess=config.get("preprocess", {}),
    )
    _run_training_loop(model, train_loader, val_loader, augmenter, criterion, optimizer, scheduler, stopper, config, device)
    best_ckpt = ckpt_dir / "improved_lstm_best.pt"
    if best_ckpt.exists():
        # Unified checkpoints store a full payload dict; pull state_dict out of it.
        # Previously we passed the whole payload to load_state_dict which crashed
        # because it contains non-tensor metadata (epoch, optimizer_state_dict, …).
        payload = torch.load(best_ckpt, map_location=device, weights_only=False)
        state_dict = payload.get("model_state_dict", payload)
        model.load_state_dict(state_dict)
        logger.info("Loaded best checkpoint for threshold tuning and test evaluation.")

    val_probs, val_labels = _collect_probs(model, val_loader, device)
    threshold = find_optimal_threshold(val_labels, val_probs)
    val_pred = (val_probs >= threshold).astype(int)
    val_metrics = {
        "f1": compute_f1_score(val_labels, val_pred),
        "auroc": compute_auc_roc(val_labels, val_probs),
        "sens": compute_sensitivity(val_labels, val_pred),
        "spec": compute_specificity(val_labels, val_pred),
    }
    save_checkpoint(
        best_ckpt, model,
        model_config=_model_config(config),
        optimizer=optimizer,
        epoch=config["training"]["num_epochs"],
        val_metrics=val_metrics,
        optimal_threshold=threshold,
        input_spec={"channels": data_cfg["n_channels"], "time_steps": data_cfg["time_steps"]},
        preprocess=config.get("preprocess", {}),
    )
    return _evaluate_improved_on_test(model, test_loader, device, threshold=threshold)


def _collect_probs(model: nn.Module, loader: DataLoader, device: torch.device):
    """Run model on loader; return (probs, labels) as 1-D numpy arrays.

    Uses ``squeeze(-1)`` so batches of size 1 stay 1-D (bare ``squeeze()`` would
    collapse them to a 0-D scalar and break ``np.concatenate``).
    """
    model.train(False)
    probs_all, labels_all = [], []
    with torch.no_grad():
        for eeg_batch, label_batch in loader:
            eeg_batch = eeg_batch.to(device).unsqueeze(1)
            logits = model(eeg_batch).squeeze(-1)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            probs_all.append(np.atleast_1d(probs))
            labels_all.append(np.atleast_1d(label_batch.numpy()))
    return (
        np.concatenate(probs_all) if probs_all else np.array([], dtype=float),
        np.concatenate(labels_all) if labels_all else np.array([], dtype=int),
    )


def _model_config(config: Dict) -> Dict:
    """Kwargs needed to re-instantiate HierarchicalLSTM from config."""
    cfg = config["models"]["improved_lstm"]
    data_cfg = config["data"]
    return {
        "n_channels": data_cfg["n_channels"],
        "time_steps": data_cfg["time_steps"],
        "n_windows": cfg.get("n_windows", 1),
        "hidden_size": cfg["hidden_size"],
        "num_layers": cfg.get("num_layers", 2),
        "dropout": cfg["dropout"],
    }


def _run_training_loop(
    model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
    augmenter: EEGAugmentation, criterion: nn.Module,
    optimizer: torch.optim.Optimizer, scheduler: object,
    stopper: EarlyStopping, config: Dict, device: torch.device,
) -> None:
    """Run epochs with augmentation until early stopping or num_epochs reached."""
    num_epochs = config["training"]["num_epochs"]
    for epoch in range(num_epochs):
        train_loss = _train_one_epoch(model, train_loader, augmenter, criterion, optimizer, config, device)
        val_loss = _validate_one_epoch(model, val_loader, criterion, device)
        val_metrics = _evaluate_improved_on_test(model, val_loader, device)
        val_f1 = val_metrics["f1"]
        scheduler.step()
        # Use unified early stopping with checkpoint schema
        stopper.step(epoch, val_loss, model, val_metrics=val_metrics)
        logger.info("Epoch %d/%d — train=%.4f val_loss=%.4f val_f1=%.4f",
                    epoch + 1, num_epochs, train_loss, val_loss, val_f1)
        if stopper.should_stop:
            logger.info("Early stopping triggered at epoch %d.", epoch + 1)
            break


def _train_one_epoch(
    model: nn.Module, train_loader: DataLoader, augmenter: EEGAugmentation,
    criterion: nn.Module, optimizer: torch.optim.Optimizer,
    config: Dict, device: torch.device,
) -> float:
    """Train one epoch with data augmentation; return mean loss."""
    model.train()
    augmenter.train()
    total_loss = 0.0
    for eeg_batch, label_batch in train_loader:
        eeg_batch, label_batch = eeg_batch.to(device), label_batch.to(device)
        eeg_batch, label_batch = augment_batch(eeg_batch, label_batch, augmenter)
        eeg_batch = eeg_batch.unsqueeze(1)
        total_loss += _run_train_batch(model, eeg_batch, label_batch, criterion, optimizer, config)
    return total_loss / max(len(train_loader), 1)


def _run_train_batch(
    model: nn.Module, eeg_batch: torch.Tensor, label_batch: torch.Tensor,
    criterion: nn.Module, optimizer: torch.optim.Optimizer, config: Dict,
) -> float:
    """Forward + backward pass for one batch; return scalar loss."""
    optimizer.zero_grad()
    loss = criterion(model(eeg_batch).squeeze(-1), label_batch.float())
    loss.backward()
    clip_gradients(model, max_norm=config["training"]["gradient_clip"])
    optimizer.step()
    return loss.item()


def _validate_one_epoch(
    model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device
) -> float:
    """Validate one epoch; return mean loss."""
    model.train(False)
    total_loss = 0.0
    with torch.no_grad():
        for eeg_batch, label_batch in val_loader:
            eeg_batch, label_batch = eeg_batch.to(device), label_batch.to(device)
            eeg_batch = eeg_batch.unsqueeze(1)
            total_loss += criterion(model(eeg_batch).squeeze(-1), label_batch.float()).item()
    return total_loss / max(len(val_loader), 1)


def _evaluate_improved_on_test(
    model: nn.Module, test_loader: DataLoader, device: torch.device,
    threshold: float = 0.5,
) -> Dict:
    """Run inference on test split with 3D→4D unsqueeze; return F1/AUC/sensitivity."""
    y_score, y_true = _collect_probs(model, test_loader, device)
    y_pred = (y_score >= threshold).astype(int)
    return {
        "f1": compute_f1_score(y_true, y_pred),
        "auc_roc": compute_auc_roc(y_true, y_score),
        "sensitivity": compute_sensitivity(y_true, y_pred),
    }


def _build_improved_model(config: Dict) -> nn.Module:
    """Build HierarchicalLSTM from config hyperparameters."""
    cfg = config["models"]["improved_lstm"]
    data_cfg = config["data"]
    return HierarchicalLSTM(
        n_channels=data_cfg["n_channels"],
        time_steps=data_cfg["time_steps"],
        n_windows=cfg.get("n_windows", 1),
        hidden_size=cfg["hidden_size"],
        num_layers=cfg.get("num_layers", 2),
        dropout=cfg["dropout"],
    )


def _build_criterion(config: Dict, device: torch.device) -> nn.Module:
    """Build FocalLoss from config."""
    pos_weight = torch.tensor(config["training"]["pos_weight"]).to(device)
    return FocalLoss(
        gamma=config["focal_loss"]["gamma"],
        pos_weight=pos_weight,
        label_smoothing=config["training"].get("label_smoothing", 0.0),
        reduction=config["focal_loss"]["reduction"],
    )


def _build_optimizer(model: nn.Module, config: Dict) -> torch.optim.AdamW:
    """Build AdamW optimizer from config, with improved_lstm overrides if present."""
    lr = config["models"]["improved_lstm"].get(
        "learning_rate", config["training"]["learning_rate"]
    )
    weight_decay = config["models"]["improved_lstm"].get(
        "weight_decay", config["training"]["weight_decay"]
    )
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )


def _build_scheduler(optimizer: torch.optim.Optimizer, config: Dict) -> WarmupCosineScheduler:
    """Build WarmupCosineScheduler from config, with improved_lstm overrides if present."""
    cfg_m = config["models"]["improved_lstm"]
    return WarmupCosineScheduler(
        optimizer,
        warmup_epochs=cfg_m.get("warmup_epochs", config["training"]["warmup_epochs"]),
        total_epochs=config["training"]["num_epochs"],
    )


def main() -> None:
    """CLI entry point for improved LSTM training."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Train improved LSTM models")
    parser.add_argument("--data_path", default="src/data/processed/chbmit")
    parser.add_argument("--config", default="src/models/config.yaml")
    args = parser.parse_args()
    config = load_config(Path(args.config))
    results = train_improved(Path(args.data_path), config)
    logger.info("Test results: %s", results)


if __name__ == "__main__":
    main()
