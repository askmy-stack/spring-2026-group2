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

    Saves best model checkpoint automatically.

    Args:
        patience: Epochs to wait before stopping after last improvement.
        min_delta: Minimum improvement required to count as improvement.
        checkpoint_path: Where to save the best model state dict.

    Example:
        >>> stopper = EarlyStopping(patience=10, checkpoint_path=Path("best.pt"))
        >>> for epoch in range(100):
        ...     stopper.step(val_loss, model)
        ...     if stopper.should_stop:
        ...         break
    """

    def __init__(
        self,
        patience: int,
        min_delta: float = 1e-4,
        checkpoint_path: Optional[Path] = None,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.best_loss: float = float("inf")
        self.epochs_without_improvement: int = 0
        self.should_stop: bool = False

    def step(self, val_loss: float, model: nn.Module) -> None:
        """
        Update state and optionally save checkpoint.

        Args:
            val_loss: Validation loss for this epoch.
            model: Model to checkpoint if improved.
        """
        if val_loss < self.best_loss - self.min_delta:
            self._record_improvement(val_loss, model)
        else:
            self._record_no_improvement()

    def _record_improvement(self, val_loss: float, model: nn.Module) -> None:
        """Save checkpoint and reset counter on improvement."""
        self.best_loss = val_loss
        self.epochs_without_improvement = 0
        if self.checkpoint_path is not None:
            torch.save(model.state_dict(), self.checkpoint_path)
            logger.debug("Checkpoint saved (val_loss=%.4f)", val_loss)

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
