from __future__ import annotations

import torch
from torch import nn

from models.utils.blocks import ConvBNAct1d, ensure_3d, make_mlp


class BaselineCNN1D(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        num_classes: int = 2,
        hidden_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct1d(in_channels, 32, kernel_size=7, dropout=dropout),
            nn.MaxPool1d(kernel_size=2),
            ConvBNAct1d(32, 64, kernel_size=5, dropout=dropout),
            nn.MaxPool1d(kernel_size=2),
            ConvBNAct1d(64, 128, kernel_size=3, dropout=dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = make_mlp((128, hidden_dim, num_classes), dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_3d(x)
        x = self.features(x).flatten(1)
        return self.classifier(x)


__all__ = ["BaselineCNN1D"]
