"""
LSTM Benchmark Training — Models m1 through m6.

Usage:
    python -m src.models.lstm_benchmark_models.train_baseline \
        --model m1_vanilla_lstm --data_path src/data/processed/chbmit

All hyperparameters from src/models/config.yaml.
"""
import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.models.utils.losses import FocalLoss
from src.models.utils.metrics import (
    compute_f1_score, compute_auc_roc, compute_sensitivity,
    compute_specificity, find_optimal_threshold,
)
from src.models.utils.callbacks import EarlyStopping, clip_gradients
from src.models.utils.config_validator import validate_config
from src.models.utils.checkpoint import save_checkpoint
from .architectures import get_benchmark_model, MODEL_REGISTRY

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict:
    """
    Load YAML config from disk.

    Args:
        config_path: Path to config.yaml file.

    Returns:
        Parsed config dictionary.

    Raises:
        FileNotFoundError: If config_path does not exist.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as config_file:
        return yaml.safe_load(config_file)


def train_baseline(model_name: str, data_path: Path, config: Dict) -> Dict:
    """
    Train one benchmark model end-to-end; return test metrics.

    Args:
        model_name: Key in MODEL_REGISTRY (e.g. 'm1_vanilla_lstm').
        data_path: Directory with train/, val/, test/ tensor splits.
        config: Config dict from config.yaml.

    Returns:
        Dict with f1, auc_roc, sensitivity on test split.

    Raises:
        ValueError: If model_name is not in MODEL_REGISTRY.
    """
    validate_config(config, model_section="lstm_benchmark")
    device = _get_device()
    model_kwargs = {**config["models"]["lstm_benchmark"]}
    model_kwargs["n_channels"] = config["data"]["n_channels"]
    model_kwargs["time_steps"] = config["data"]["time_steps"]
    model = get_benchmark_model(model_name, **model_kwargs).to(device)
    train_loader, val_loader, test_loader = _build_data_loaders(data_path, config)
    criterion = _build_criterion(config, device)
    optimizer = _build_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["num_epochs"]
    )
    ckpt_dir = Path(config["outputs"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    stopper = EarlyStopping(
        patience=config["training"]["early_stopping_patience"],
        checkpoint_path=ckpt_dir / f"{model_name}_best.pt",
        model_config=model_kwargs,
        input_spec={"channels": config["data"]["n_channels"], "time_steps": config["data"]["time_steps"]},
        preprocess=config.get("preprocess", {}),
    )
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    _run_training_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, stopper, config, device, scaler)
    ckpt_path = ckpt_dir / f"{model_name}_best.pt"
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
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
        ckpt_path, model,
        model_config=model_kwargs,
        optimizer=optimizer,
        epoch=config["training"]["num_epochs"],
        val_metrics=val_metrics,
        optimal_threshold=threshold,
    )
    return _evaluate_on_test(model, test_loader, device, threshold=threshold)


def _run_training_loop(
    model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
    criterion: nn.Module, optimizer: torch.optim.Optimizer,
    scheduler: object, stopper: EarlyStopping, config: Dict, device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
) -> None:
    """Run epochs until early stopping or num_epochs reached."""
    num_epochs = config["training"]["num_epochs"]
    if scaler is None:
        scaler = torch.amp.GradScaler(enabled=False)
    for epoch in range(num_epochs):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, config, device, scaler)
        val_loss = _validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step()
        stopper.step(epoch, val_loss, model, val_metrics={"val_loss": val_loss})
        logger.info("Epoch %d/%d — train=%.4f val=%.4f", epoch + 1, num_epochs, train_loss, val_loss)
        if stopper.should_stop:
            logger.info("Early stopping triggered.")
            break


def _train_one_epoch(
    model: nn.Module, train_loader: DataLoader,
    criterion: nn.Module, optimizer: torch.optim.Optimizer,
    config: Dict, device: torch.device,
    scaler: torch.amp.GradScaler,
) -> float:
    """Train for one epoch with AMP; return mean training loss."""
    model.train()
    total_loss = 0.0
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    for eeg_batch, label_batch in train_loader:
        eeg_batch, label_batch = eeg_batch.to(device), label_batch.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            loss = criterion(model(eeg_batch).squeeze(-1), label_batch.float())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_gradients(model, max_norm=config["training"]["gradient_clip"])
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / max(len(train_loader), 1)


def _validate_one_epoch(
    model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device
) -> float:
    """Validate for one epoch; return mean validation loss."""
    model.train(False)
    total_loss = 0.0
    with torch.no_grad():
        for eeg_batch, label_batch in val_loader:
            eeg_batch, label_batch = eeg_batch.to(device), label_batch.to(device)
            total_loss += criterion(model(eeg_batch).squeeze(-1), label_batch.float()).item()
    return total_loss / max(len(val_loader), 1)


def _evaluate_on_test(
    model: nn.Module, test_loader: DataLoader, device: torch.device,
    threshold: float = 0.5,
) -> Dict:
    """Run inference on test split; return F1/AUC/sensitivity at the given threshold."""
    y_score, y_true = _collect_probs(model, test_loader, device)
    y_pred = (y_score >= threshold).astype(int)
    return {
        "f1": compute_f1_score(y_true, y_pred),
        "auc_roc": compute_auc_roc(y_true, y_score),
        "sensitivity": compute_sensitivity(y_true, y_pred),
    }


def _collect_probs(
    model: nn.Module, loader: DataLoader, device: torch.device,
):
    """Run model on loader; return ``(probs, labels)`` as 1-D numpy arrays.

    Uses ``squeeze(-1)`` (not bare ``squeeze()``) so the last batch of size 1
    still yields a 1-D array instead of a 0-D scalar that breaks ``.tolist()``.
    """
    model.train(False)
    probs_all, labels_all = [], []
    with torch.no_grad():
        for eeg_batch, label_batch in loader:
            eeg_batch = eeg_batch.to(device)
            logits = model(eeg_batch).squeeze(-1)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            probs_all.append(np.atleast_1d(probs))
            labels_all.append(np.atleast_1d(label_batch.numpy()))
    return (
        np.concatenate(probs_all) if probs_all else np.array([], dtype=float),
        np.concatenate(labels_all) if labels_all else np.array([], dtype=int),
    )


def _get_device() -> torch.device:
    """Return CUDA if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_criterion(config: Dict, device: torch.device) -> nn.Module:
    """Instantiate FocalLoss with pos_weight from config."""
    pos_weight = torch.tensor(config["training"]["pos_weight"]).to(device)
    return FocalLoss(
        gamma=config["focal_loss"]["gamma"],
        pos_weight=pos_weight,
        label_smoothing=config["training"].get("label_smoothing", 0.0),
        reduction=config["focal_loss"]["reduction"],
    )


def _build_optimizer(model: nn.Module, config: Dict) -> torch.optim.AdamW:
    """Build AdamW optimizer from config."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )


def _build_data_loaders(data_path: Path, config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train/val/test DataLoaders from pre-processed tensors.

    Args:
        data_path: Directory containing train/, val/, test/ subdirs with data.pt/labels.pt.
        config: Config dict with training.batch_size.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).

    Raises:
        FileNotFoundError: If any required split directory is missing.
    """
    batch_size = config["training"]["batch_size"]
    pin = torch.cuda.is_available()

    def _load_split(split: str) -> TensorDataset:
        split_path = data_path / split
        if not (split_path / "data.pt").exists():
            raise FileNotFoundError(f"Missing split at {split_path}")
        data = torch.load(split_path / "data.pt", weights_only=True).float()
        labels = torch.load(split_path / "labels.pt", weights_only=True).float().squeeze()
        return TensorDataset(data, labels)

    return (
        DataLoader(_load_split("train"), batch_size=batch_size, shuffle=True, pin_memory=pin),
        DataLoader(_load_split("val"), batch_size=batch_size, shuffle=False, pin_memory=pin),
        DataLoader(_load_split("test"), batch_size=batch_size, shuffle=False, pin_memory=pin),
    )


def main() -> None:
    """CLI entry point for training."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Train LSTM benchmark models m1-m6")
    parser.add_argument("--model", required=True, choices=["all"] + list(MODEL_REGISTRY.keys()))
    parser.add_argument("--data_path", default="src/data/processed/chbmit")
    parser.add_argument("--config", default="src/models/config.yaml")
    args = parser.parse_args()
    config = load_config(Path(args.config))
    models_to_train = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
    all_results = {}
    for model_name in models_to_train:
        logger.info("=" * 60)
        logger.info("Training model: %s", model_name)
        logger.info("=" * 60)
        results = train_baseline(model_name, Path(args.data_path), config)
        all_results[model_name] = results
        logger.info("Results for %s: %s", model_name, results)
    if len(models_to_train) > 1:
        logger.info("=" * 60)
        logger.info("ALL RESULTS SUMMARY:")
        for name, res in all_results.items():
            logger.info("  %s: %s", name, res)


if __name__ == "__main__":
    main()
