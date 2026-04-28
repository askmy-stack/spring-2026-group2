"""
7-Model Ensemble: benchmark LSTM models (m1-m6) + m7 VQ-Transformer.

Import from here for ensemble evaluation.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.component.models.lstm_benchmark_models import get_benchmark_model, MODEL_REGISTRY
from src.component.models.utils.metrics import compute_f1_score, compute_auc_roc, compute_sensitivity
from .architectures.m7_vq_transformer import M7_VQTransformer

logger = logging.getLogger(__name__)

ALL_MODEL_NAMES = sorted(MODEL_REGISTRY.keys()) + ["m7_vq_transformer"]


def load_ensemble_checkpoints(
    checkpoint_dir: Path, config: Dict, device: torch.device
) -> Dict[str, nn.Module]:
    """
    Load all available model checkpoints from a directory.

    Args:
        checkpoint_dir: Directory containing *_best.pt files.
        config: Config dict with model hyperparameters.
        device: Device to load tensors onto.

    Returns:
        Dict mapping model_name -> loaded nn.Module.
    """
    loaded = {}
    for model_name in ALL_MODEL_NAMES:
        ckpt_path = checkpoint_dir / f"{model_name}_best.pt"
        if ckpt_path.exists():
            model = _load_single_checkpoint(model_name, ckpt_path, config, device)
            loaded[model_name] = model
            logger.info("Loaded checkpoint: %s", model_name)
        else:
            logger.warning("Missing checkpoint for %s at %s", model_name, ckpt_path)
    return loaded


def _load_single_checkpoint(
    model_name: str, path: Path, config: Dict, device: torch.device
) -> nn.Module:
    """
    Load one model checkpoint.

    Args:
        model_name: Key in MODEL_REGISTRY or 'm7_vq_transformer'.
        path: Path to the .pt state dict file.
        config: Config dict.
        device: Target device.

    Returns:
        Loaded nn.Module in eval mode.
    """
    if model_name == "m7_vq_transformer":
        cfg = config["models"]["ensemble_transformers"]
        model = M7_VQTransformer(
            n_channels=config["data"]["n_channels"],
            time_steps=config["data"]["time_steps"],
            patch_size=cfg["patch_size"],
            hidden_size=cfg["hidden_size"],
            codebook_size=cfg["codebook_size"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            dropout=cfg["dropout"],
        )
    else:
        model_kwargs = {**config["models"]["lstm_benchmark"]}
        model_kwargs["n_channels"] = config["data"]["n_channels"]
        model_kwargs["time_steps"] = config["data"]["time_steps"]
        model = get_benchmark_model(model_name, **model_kwargs)
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return model


def mean_ensemble_predict(
    models: Dict[str, nn.Module], eeg_batch: torch.Tensor
) -> np.ndarray:
    """
    Compute mean-probability ensemble prediction.

    Args:
        models: Dict of model_name -> nn.Module (all in eval mode).
        eeg_batch: Input tensor, shape (batch, n_channels, time_steps).

    Returns:
        Mean probabilities, shape (batch,).
    """
    all_probs = _collect_all_probs(models, eeg_batch)
    return all_probs.mean(axis=-1)


def weighted_ensemble_predict(
    models: Dict[str, nn.Module], eeg_batch: torch.Tensor, weights: np.ndarray
) -> np.ndarray:
    """
    Compute weighted-probability ensemble prediction.

    Args:
        models: Dict of model_name -> nn.Module (all in eval mode).
        eeg_batch: Input tensor, shape (batch, n_channels, time_steps).
        weights: Per-model weights, shape (n_models,). Will be normalised.

    Returns:
        Weighted probabilities, shape (batch,).
    """
    all_probs = _collect_all_probs(models, eeg_batch)
    normalised = weights / weights.sum()
    return (all_probs * normalised).sum(axis=-1)


def _collect_all_probs(
    models: Dict[str, nn.Module], eeg_batch: torch.Tensor
) -> np.ndarray:
    """
    Collect sigmoid probabilities from all models.

    Args:
        models: Dict of model_name -> nn.Module.
        eeg_batch: Input tensor.

    Returns:
        Array of shape (batch, n_models).
    """
    all_probs = []
    with torch.no_grad():
        for model in models.values():
            logits = model(eeg_batch)
            all_probs.append(torch.sigmoid(logits).squeeze(-1).cpu().numpy())
    return np.stack(all_probs, axis=-1)


def evaluate_ensemble(
    models: Dict[str, nn.Module],
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    strategy: str = "mean",
    weights: Optional[np.ndarray] = None,
) -> Dict:
    """
    Evaluate ensemble on a test DataLoader.

    Args:
        models: Dict of loaded models.
        test_loader: DataLoader for test split.
        device: Device for inference.
        strategy: 'mean' or 'weighted'.
        weights: Per-model weights if strategy='weighted'.

    Returns:
        Dict with f1, auc_roc, sensitivity.
    """
    y_true_list, y_score_list = _collect_test_predictions(models, test_loader, device, strategy, weights)
    y_true = np.array(y_true_list)
    y_score = np.array(y_score_list)
    y_pred = (y_score > 0.5).astype(int)
    return {
        "f1": compute_f1_score(y_true, y_pred),
        "auc_roc": compute_auc_roc(y_true, y_score),
        "sensitivity": compute_sensitivity(y_true, y_pred),
    }


def _collect_test_predictions(
    models: Dict[str, nn.Module],
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    strategy: str,
    weights: Optional[np.ndarray],
) -> Tuple[List, List]:
    """Iterate test_loader and collect ground truth + ensemble scores."""
    y_true_list, y_score_list = [], []
    for eeg_batch, label_batch in test_loader:
        eeg_batch = eeg_batch.to(device)
        if strategy == "weighted" and weights is not None:
            batch_probs = weighted_ensemble_predict(models, eeg_batch, weights)
        else:
            batch_probs = mean_ensemble_predict(models, eeg_batch)
        y_score_list.extend(batch_probs.tolist())
        y_true_list.extend(label_batch.numpy().tolist())
    return y_true_list, y_score_list
