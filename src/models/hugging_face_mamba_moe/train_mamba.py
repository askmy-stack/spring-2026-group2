"""
Mamba + MoE Training — Directory 4 of 4.

All hyperparameters from src/models/config.yaml.

Usage:
    python -m src.models.hugging_face_mamba_moe.train_mamba \
        --model eeg_mamba --data_path src/data/processed/chbmit
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
from src.models.utils.callbacks import EarlyStopping, clip_gradients
from src.models.lstm_benchmark_models.train_baseline import (
    load_config, _get_device, _build_data_loaders,
)
from src.models.utils.metrics import (
    compute_f1_score, compute_auc_roc, compute_sensitivity,
    compute_specificity, find_optimal_threshold,
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
    # Warmup+Cosine: linearly ramp up for ``warmup_epochs`` then cosine-decay.
    # Mamba-MoE is unstable at full LR from step 0 — caused NaN divergence in
    # earlier runs. Warmup avoids that without hurting non-MoE training.
    scheduler = _build_warmup_cosine_scheduler(optimizer, config)
    ckpt_dir = Path(config["outputs"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Build model_config for checkpoint schema
    cfg = config["models"]["hugging_face_mamba_moe"]
    data_cfg = config["data"]
    model_config = dict(
        n_channels=data_cfg["n_channels"],
        time_steps=data_cfg["time_steps"],
        d_model=cfg["d_model"],
        d_state=cfg["d_state"],
        n_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
    )
    if model_name == "eeg_mamba":
        model_config["d_conv"] = cfg.get("d_conv", 4)
        model_config["bidirectional"] = cfg.get("bidirectional", True)
    elif model_name == "eeg_mamba_moe":
        model_config["num_experts"] = cfg.get("num_experts", 8)
        model_config["top_k"] = cfg.get("top_k", 2)
    stopper = EarlyStopping(
        patience=config["training"]["early_stopping_patience"],
        checkpoint_path=ckpt_dir / f"{model_name}_best.pt",
        model_config=model_config,
        input_spec={"channels": data_cfg["n_channels"], "time_steps": data_cfg["time_steps"]},
        preprocess=config.get("preprocess", {}),
    )
    _run_training_loop(model, model_name, train_loader, val_loader, criterion, optimizer, scheduler, stopper, config, device)
    # Reload best checkpoint (EarlyStopping saves on val-loss improvement)
    _maybe_reload_best(model, stopper.checkpoint_path, device)
    # Tune threshold on val, apply to test. Previously we hardcoded 0.5 which
    # cost ~20 F1 points on eeg_mamba (AUC 0.71 → F1 0.34).
    threshold = _tune_threshold_on_val(model, model_name, val_loader, device)
    logger.info("Tuned threshold on val set: %.3f", threshold)
    return _evaluate_mamba(model, model_name, test_loader, device, threshold=threshold)


def _run_training_loop(
    model: nn.Module, model_name: str,
    train_loader: DataLoader, val_loader: DataLoader,
    criterion: nn.Module, optimizer: torch.optim.Optimizer,
    scheduler: object, stopper: EarlyStopping, config: Dict, device: torch.device,
) -> None:
    """Run training loop with optional MoE auxiliary loss until early stopping."""
    num_epochs = config["training"]["num_epochs"]
    is_moe = model_name == "eeg_mamba_moe"
    moe_weight = config["models"]["hugging_face_mamba_moe"].get("moe_loss_weight", 0.001)
    for epoch in range(num_epochs):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, config, device, is_moe, moe_weight)
        # NaN guard: if train loss diverges we must not keep looping — the
        # optimiser will keep writing NaN weights into the checkpoint. Abort
        # cleanly; the previously-saved best checkpoint (if any) is kept.
        if not math.isfinite(train_loss):
            logger.error(
                "Non-finite train loss at epoch %d (train_loss=%s) — aborting run.",
                epoch + 1, train_loss,
            )
            break
        val_loss = _validate_one_epoch(model, val_loader, criterion, device, is_moe)
        if not math.isfinite(val_loss):
            logger.error(
                "Non-finite val loss at epoch %d (val_loss=%s) — aborting run.",
                epoch + 1, val_loss,
            )
            break
        scheduler.step()
        stopper.step(epoch, val_loss, model, val_metrics={"val_loss": val_loss})
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


def _collect_probs(
    model: nn.Module, model_name: str, loader: DataLoader, device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference and return (y_score, y_true) as flat 1-D numpy arrays.

    Unpacks ``(logits, lb_loss)`` for ``eeg_mamba_moe``; for plain ``eeg_mamba``
    it treats the output as a tensor. Used by both threshold tuning (on val)
    and final evaluation (on test).
    """
    is_moe = model_name == "eeg_mamba_moe"
    model.eval()
    probs_all, labels_all = [], []
    with torch.no_grad():
        for eeg_batch, label_batch in loader:
            eeg_batch = eeg_batch.to(device)
            out = model(eeg_batch)
            logits = out[0] if is_moe else out
            probs = torch.sigmoid(logits.squeeze(-1)).detach().cpu().numpy()
            probs_all.append(np.atleast_1d(probs))
            labels_all.append(np.atleast_1d(label_batch.numpy()))
    return np.concatenate(probs_all), np.concatenate(labels_all)


def _tune_threshold_on_val(
    model: nn.Module, model_name: str, val_loader: DataLoader, device: torch.device,
) -> float:
    """Grid-search threshold in [0.05, 0.95] maximising F1 on val probs.

    Calibrates the decision boundary *before* test evaluation. Previously the
    evaluator hardcoded 0.5 which kept eeg_mamba's F1 far below what the AUC
    of 0.71 implied was achievable.
    """
    y_score, y_true = _collect_probs(model, model_name, val_loader, device)
    return float(find_optimal_threshold(y_true, y_score))


def _evaluate_mamba(
    model: nn.Module, model_name: str, test_loader: DataLoader, device: torch.device,
    threshold: float = 0.5,
) -> Dict:
    """Run inference on test split; unpack (logits, lb_loss) for MoE models.

    ``threshold`` should be tuned on the val split (see
    :func:`_tune_threshold_on_val`). A hardcoded 0.5 is only correct for
    perfectly-calibrated logits, which Mamba is not.
    """
    y_score, y_true = _collect_probs(model, model_name, test_loader, device)
    y_pred = (y_score >= threshold).astype(int)
    return {
        "f1": compute_f1_score(y_true, y_pred),
        "auc_roc": compute_auc_roc(y_true, y_score),
        "sensitivity": compute_sensitivity(y_true, y_pred),
        "specificity": compute_specificity(y_true, y_pred),
        "threshold": float(threshold),
    }


def _maybe_reload_best(
    model: nn.Module, checkpoint_path, device: torch.device,
) -> None:
    """Reload best-val-loss weights before final evaluation, if available."""
    if checkpoint_path is None or not Path(checkpoint_path).exists():
        logger.warning("No best checkpoint found at %s; evaluating current weights.", checkpoint_path)
        return
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    logger.info("Reloaded best checkpoint from %s (epoch=%s)", checkpoint_path, payload.get("epoch"))


def _build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer, config: Dict,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup over ``warmup_epochs``, then cosine decay to 0.

    Kept as an inline LambdaLR to avoid importing a heavier scheduler class.
    """
    total = int(config["training"]["num_epochs"])
    warmup = int(config["training"].get("warmup_epochs", 0))

    def lr_lambda(epoch: int) -> float:
        if warmup > 0 and epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(total - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


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
