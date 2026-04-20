from __future__ import annotations

import torch
from torch import nn

from models.utils.blocks import (
    ConvBNAct1d,
    DepthwiseSeparableConv1d,
    MultiScaleBranch1d,
    ResidualBlock1d,
    SEBlock1d,
    ensure_3d,
    make_mlp,
)


class MultiScaleAttentionCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        num_classes: int = 2,
        branch_channels: int = 32,
        kernel_sizes: tuple[int, ...] = (3, 7, 15, 31),
        dropout: float = 0.35,
    ):
        super().__init__()
        self.branches = nn.ModuleList(
            [MultiScaleBranch1d(in_channels, branch_channels, kernel_size=k, dropout=dropout) for k in kernel_sizes]
        )
        fusion_channels = branch_channels * len(kernel_sizes)
        self.fusion = nn.Sequential(
            ConvBNAct1d(fusion_channels, 128, kernel_size=3, dropout=dropout),
            SEBlock1d(128, reduction=8),
            ResidualBlock1d(128, kernel_size=3, dropout=dropout),
            SEBlock1d(128, reduction=8),
            DepthwiseSeparableConv1d(128, 160, kernel_size=5, dropout=dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = make_mlp((160, 64, num_classes), dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_3d(x)
        multi_scale = [branch(x) for branch in self.branches]
        x = torch.cat(multi_scale, dim=1)
        x = self.fusion(x).flatten(1)
        return self.classifier(x)


__all__ = ["MultiScaleAttentionCNN"]
