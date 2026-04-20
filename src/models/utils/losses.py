from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        alpha_t = torch.where(
            targets == 1,
            torch.full_like(ce, self.alpha),
            torch.full_like(ce, 1.0 - self.alpha),
        )
        return (alpha_t * ((1.0 - pt) ** self.gamma) * ce).mean()


__all__ = ["BinaryFocalLoss"]
