"""
Focal Loss for severe class imbalance
======================================
Focal loss down-weights easy negative examples so the model focuses on hard
seizure examples. Especially effective for severe imbalance (10:1 or worse).
Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss: (1 - p_t)^gamma * BCE(logits, targets)

    For class imbalance: alpha weights positive vs negative class,
    gamma focuses on hard examples.

    Standard hyperparameters: alpha=0.25, gamma=2.0
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Args:
            alpha: Weight for positive (seizure) class.
                   0.25 is RetinaNet default; matched to typical 3:1 neg:pos.
            gamma: Focusing parameter. 2.0 is standard; higher = focus more on hard examples.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model predictions, shape (batch, 1)
            targets: Binary labels, shape (batch, 1)

        Returns:
            Scalar loss
        """
        # Cross-entropy loss per sample
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Probability of correct class: p_t = exp(-BCE)
        # For well-calibrated predictions: p_t ≈ sigmoid(logits) if target=1, 1-sigmoid(logits) if target=0
        pt = torch.exp(-bce)

        # Alpha weighting: alpha for positive class, (1-alpha) for negative
        alpha_t = torch.where(
            targets == 1,
            torch.full_like(targets, self.alpha),
            torch.full_like(targets, 1 - self.alpha)
        )

        # Focal weight: (1 - p_t)^gamma — zero for easy examples (p_t ≈ 1), nonzero for hard
        focal_weight = (1 - pt) ** self.gamma

        # Combined loss
        loss = alpha_t * focal_weight * bce

        return loss.mean()
