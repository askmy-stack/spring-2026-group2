"""
Loss functions for EEG seizure detection.

Consolidated from approach2/train.py. Import FocalLoss from here.
Never define it locally in a training script.
"""
import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for severe class imbalance (seizure vs. non-seizure).

    Down-weights easy negatives via (1-p_t)^gamma, then applies pos_weight
    AFTER focal modulation to avoid double-weighting.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

    Args:
        gamma: Focusing parameter. Higher = more focus on hard examples.
        pos_weight: Scalar weight tensor for positive (seizure) class.
        label_smoothing: Label smoothing factor in [0, 1).
        reduction: 'mean' or 'sum'.

    Example:
        >>> criterion = FocalLoss(gamma=1.0, pos_weight=torch.tensor(3.0))
        >>> loss = criterion(torch.randn(4, 1), torch.randint(0, 2, (4, 1)).float())
    """

    def __init__(
        self,
        gamma: float = 1.0,
        pos_weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Raw model outputs, shape (batch,) or (batch, 1)
            targets: Binary labels in {0, 1}, same shape as logits

        Returns:
            Scalar loss tensor.
        """
        logits_f32 = logits.float()
        targets_f32 = _apply_label_smoothing(targets.float(), self.label_smoothing)
        bce_per_sample = self.bce(logits_f32, targets_f32)
        probability_correct = torch.exp(-bce_per_sample)
        focal_weight = (1.0 - probability_correct) ** self.gamma
        focal_loss = focal_weight * bce_per_sample
        if self.pos_weight is not None:
            weight_mask = _build_pos_weight_mask(targets_f32, self.pos_weight)
            focal_loss = focal_loss * weight_mask
        return _reduce_loss(focal_loss, self.reduction)


def _apply_label_smoothing(targets: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    Apply label smoothing to binary targets.

    Args:
        targets: Binary labels, shape (batch,)
        epsilon: Smoothing factor in [0, 1)

    Returns:
        Smoothed targets tensor.
    """
    if epsilon <= 0.0:
        return targets
    return targets * (1.0 - epsilon) + 0.5 * epsilon


def _build_pos_weight_mask(
    targets: torch.Tensor, pos_weight: torch.Tensor
) -> torch.Tensor:
    """
    Build per-sample weight mask: pos_weight for seizure, 1.0 for non-seizure.

    Args:
        targets: Binary labels (possibly smoothed), shape (batch,)
        pos_weight: Scalar weight for the positive class

    Returns:
        Per-sample weight tensor, same shape as targets.
    """
    return torch.where(
        targets > 0.5,
        pos_weight.to(targets.device),
        torch.ones_like(targets),
    )


def _reduce_loss(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    """
    Apply reduction to per-sample loss.

    Args:
        loss: Per-sample losses, shape (batch,)
        reduction: 'mean' or 'sum'

    Returns:
        Scalar loss tensor.

    Raises:
        ValueError: If reduction is not 'mean' or 'sum'.
    """
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"Unknown reduction '{reduction}'. Expected 'mean' or 'sum'.")
