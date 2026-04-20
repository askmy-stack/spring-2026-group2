"""
Mamba + MoE Training — Directory 4 of 4.

All hyperparameters from src/models/config.yaml.

Usage:
    python -m src.models.hugging_face_mamba_moe.train_mamba \
        --model eeg_mamba --data_path src/data/processed/chbmit
"""
import argparse
import logging
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.utils.losses import FocalLoss
from src.models.utils.callbacks import EarlyStopping, clip_gradients
from src.models.lstm_benchmark_models.train_baseline import (
    load_config, _get_device, _build_data_loaders, _evaluate_on_test
)
from src.models.utils.config_validator import validate_config
from .architectures.eeg_mamba import EEGMamba, EEGMambaMoE

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "eeg_mamba": EEGMamba,
    "eeg_mamba_moe": EEGMambaMoE,
}


def train_mamba(model_name: str, data_path: Path, config: Dict) -> Dict:
    """
    Train a Mamba or Mamba-MoE model; return test metrics.

    Args:
        model_name: Key in MODEL_REGISTRY ('eeg_mamba' or 'eeg_mamba_moe').
        data_path: Directory with train/, val/, test/ tensor splits.
        config: Config dict from config.yaml.

    Returns:
        Dict with f1, auc_roc, sensitivity.

    Raises:
        ValueError: If model_name is not in MODEL_REGISTRY.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Options: {sorted(MODEL_REGISTRY)}")
    validate_config(config, model_section="hugging_face_mamba_moe")
    device = _get_device()
    model = _build_model(model_name, config).to(device)
    train_loader, val_loader, test_loader = _build_data_loaders(data_path, config)
    criterion = _build_criterion(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["num_epochs"]
    )
    ckpt_dir = Path(config["outputs"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    stopper = EarlyStopping(
        patience=config["training"]["early_stopping_patience"],
        checkpoint_path=ckpt_dir / f"{model_name}_best.pt",
    )
    _run_training_loop(model, model_name, train_loader, val_loader, criterion, optimizer, scheduler, stopper, config, device)
    return _evaluate_on_test(model, test_loader, device)


def _run_training_loop(
    model: nn.Module, model_name: str,
    train_loader: DataLoader, val_loader: DataLoader,
    criterion: nn.Module, optimizer: torch.optim.Optimizer,
    scheduler: object, stopper: EarlyStopping, config: Dict, device: torch.device,
) -> None:
    """Run training loop with optional MoE auxiliary loss until early stopping."""
    num_epochs = config["training"]["num_epochs"]
    is_moe = model_name == "eeg_mamba_moe"
    moe_weight = config["models"]["hugging_face_mamba_moe"].get("moe_loss_weight", 0.01)
    for epoch in range(num_epochs):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, config, device, is_moe, moe_weight)
        val_loss = _validate_one_epoch(model, val_loader, criterion, device, is_moe)
        scheduler.step()
        stopper.step(val_loss, model)
        logger.info("Epoch %d/%d — train=%.4f val=%.4f", epoch + 1, num_epochs, train_loss, val_loss)
        if stopper.should_stop:
            logger.info("Early stopping triggered.")
            break


def _train_one_epoch(
    model: nn.Module, train_loader: DataLoader, criterion: nn.Module,
    optimizer: torch.optim.Optimizer, config: Dict, device: torch.device,
    is_moe: bool, moe_weight: float,
) -> float:
    """Train one epoch; return mean loss."""
    model.train()
    total_loss = 0.0
    for eeg_batch, label_batch in train_loader:
        eeg_batch, label_batch = eeg_batch.to(device), label_batch.to(device)
        optimizer.zero_grad()
        loss = _compute_loss(model, eeg_batch, label_batch, criterion, is_moe, moe_weight)
        loss.backward()
        clip_gradients(model, max_norm=config["training"]["gradient_clip"])
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(train_loader), 1)


def _compute_loss(
    model: nn.Module, eeg_batch: torch.Tensor, label_batch: torch.Tensor,
    criterion: nn.Module, is_moe: bool, moe_weight: float,
) -> torch.Tensor:
    """Compute main loss + optional MoE load-balance auxiliary loss."""
    if is_moe:
        logits, lb_loss = model(eeg_batch)
        return criterion(logits.squeeze(-1), label_batch.float()) + moe_weight * lb_loss
    return criterion(model(eeg_batch).squeeze(-1), label_batch.float())


def _validate_one_epoch(
    model: nn.Module, val_loader: DataLoader,
    criterion: nn.Module, device: torch.device, is_moe: bool,
) -> float:
    """Validate one epoch; return mean loss."""
    model.train(False)
    total_loss = 0.0
    with torch.no_grad():
        for eeg_batch, label_batch in val_loader:
            eeg_batch, label_batch = eeg_batch.to(device), label_batch.to(device)
            if is_moe:
                logits, _ = model(eeg_batch)
            else:
                logits = model(eeg_batch)
            total_loss += criterion(logits.squeeze(-1), label_batch.float()).item()
    return total_loss / max(len(val_loader), 1)


def _build_model(model_name: str, config: Dict) -> nn.Module:
    """Instantiate the appropriate Mamba model from config."""
    cfg = config["models"]["hugging_face_mamba_moe"]
    data_cfg = config["data"]
    common_kwargs = dict(
        n_channels=data_cfg["n_channels"],
        time_steps=data_cfg["time_steps"],
        d_model=cfg["d_model"],
        d_state=cfg["d_state"],
        n_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
    )
    if model_name == "eeg_mamba":
        common_kwargs["d_conv"] = cfg.get("d_conv", 4)
        common_kwargs["bidirectional"] = cfg.get("bidirectional", True)
    elif model_name == "eeg_mamba_moe":
        common_kwargs["num_experts"] = cfg.get("num_experts", 8)
        common_kwargs["top_k"] = cfg.get("top_k", 2)
    return MODEL_REGISTRY[model_name](**common_kwargs)


def _build_criterion(config: Dict, device: torch.device) -> nn.Module:
    """Build FocalLoss from config."""
    pos_weight = torch.tensor(config["training"]["pos_weight"]).to(device)
    return FocalLoss(gamma=config["focal_loss"]["gamma"], pos_weight=pos_weight,
                     reduction=config["focal_loss"]["reduction"])


def main() -> None:
    """CLI entry point for Mamba/MoE training."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Train EEG Mamba or Mamba-MoE")
    parser.add_argument("--model", default="eeg_mamba", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--data_path", default="src/data/processed/chbmit")
    parser.add_argument("--config", default="src/models/config.yaml")
    args = parser.parse_args()
    config = load_config(Path(args.config))
    results = train_mamba(args.model, Path(args.data_path), config)
    logger.info("Test results: %s", results)


if __name__ == "__main__":
    main()
