"""
Training callbacks: early stopping and gradient clipping.

Import from here. Never implement inline in a training script.
"""
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Stop training when validation loss stops improving.

    Saves best model checkpoint automatically using unified checkpoint schema.

    Args:
        patience: Epochs to wait before stopping after last improvement.
        min_delta: Minimum improvement required to count as improvement.
        checkpoint_path: Where to save the best model checkpoint.
        model_config: Kwargs to reconstruct the model (for unified checkpoint).
        model_builder: Optional factory function path for model reconstruction.
        input_spec: EEG input specification (channels, sfreq, etc.).
        preprocess: Preprocessing parameters (resample, bandpass, etc.).

    Example:
        >>> stopper = EarlyStopping(patience=10, checkpoint_path=Path("best.pt"))
        >>> for epoch in range(100):
        ...     stopper.step(epoch, val_loss, model, model_config={...})
        ...     if stopper.should_stop:
        ...         break
    """

    def __init__(
        self,
        patience: int,
        min_delta: float = 1e-4,
        checkpoint_path: Optional[Path] = None,
        model_config: Optional[Dict[str, Any]] = None,
        model_builder: Optional[str] = None,
        input_spec: Optional[Dict[str, Any]] = None,
        preprocess: Optional[Dict[str, Any]] = None,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.model_config = model_config or {}
        self.model_builder = model_builder
        self.input_spec = input_spec
        self.preprocess = preprocess
        self.best_loss: float = float("inf")
        self.best_epoch: int = 0
        self.epochs_without_improvement: int = 0
        self.should_stop: bool = False

    def step(
        self,
        epoch: int,
        val_loss: float,
        model: nn.Module,
        val_metrics: Optional[Dict[str, float]] = None,
        optimal_threshold: float = 0.5,
    ) -> None:
        """
        Update state and optionally save checkpoint.

        Args:
            epoch: Current training epoch.
            val_loss: Validation loss for this epoch.
            model: Model to checkpoint if improved.
            val_metrics: Optional metrics dict for tracking.
            optimal_threshold: Decision threshold for classification.
        """
        if val_loss < self.best_loss - self.min_delta:
            self._record_improvement(
                epoch, val_loss, model, val_metrics, optimal_threshold
            )
        else:
            self._record_no_improvement()

    def _record_improvement(
        self,
        epoch: int,
        val_loss: float,
        model: nn.Module,
        val_metrics: Optional[Dict[str, float]] = None,
        optimal_threshold: float = 0.5,
    ) -> None:
        """Save checkpoint and reset counter on improvement."""
        self.best_loss = val_loss
        self.best_epoch = epoch
        self.epochs_without_improvement = 0
        if self.checkpoint_path is not None:
            from .checkpoint import save_checkpoint

            save_checkpoint(
                self.checkpoint_path,
                model,
                model_config=self.model_config,
                model_builder=self.model_builder,
                epoch=epoch,
                val_metrics=val_metrics,
                optimal_threshold=optimal_threshold,
                input_spec=self.input_spec,
                preprocess=self.preprocess,
            )
            logger.debug("Checkpoint saved (epoch=%d, val_loss=%.4f)", epoch, val_loss)

    def _record_no_improvement(self) -> None:
        """Increment counter; trigger stop if patience exceeded."""
        self.epochs_without_improvement += 1
        if self.epochs_without_improvement >= self.patience:
            self.should_stop = True
            logger.info(
                "Early stopping triggered after %d epochs without improvement.",
                self.patience,
            )


def clip_gradients(model: nn.Module, max_norm: float) -> float:
    """
    Clip gradient norms to prevent exploding gradients.

    Args:
        model: Model whose parameters to clip.
        max_norm: Maximum allowed gradient norm.

    Returns:
        Total gradient norm before clipping.

    Example:
        >>> grad_norm = clip_gradients(model, max_norm=1.0)
    """
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    return float(total_norm)
