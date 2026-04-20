from __future__ import annotations

import torch
from torch import nn

from models.utils.blocks import ensure_3d


class DeepConvNet(nn.Module):
    def __init__(self, in_channels: int = 16, num_classes: int = 2, dropout: float = 0.4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 10), padding=(0, 5), bias=False),
            nn.Conv2d(25, 25, kernel_size=(in_channels, 1), bias=False),
            nn.BatchNorm2d(25),
            nn.ELU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(
            self._make_block(25, 50, dropout),
            self._make_block(50, 100, dropout),
            self._make_block(100, 200, dropout),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(200, num_classes)

    @staticmethod
    def _make_block(in_filters: int, out_filters: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_filters, out_filters, kernel_size=(1, 10), padding=(0, 5), bias=False),
            nn.BatchNorm2d(out_filters),
            nn.ELU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_3d(x)
        x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.blocks(x)
        x = x.flatten(1)
        return self.classifier(x)


__all__ = ["DeepConvNet"]
