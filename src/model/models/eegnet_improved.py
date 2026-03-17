from __future__ import annotations

import torch
import torch.nn as nn


class EEGNetImproved(nn.Module):
    """
    EEGNet variant with less aggressive pooling and wider temporal refinement.

    Input:  (B, n_ch, n_samples)
    Output: (B,) logits
    """

    def __init__(
        self,
        n_ch: int = 16,
        n_samples: int = 256,
        F1: int = 16,
        D: int = 2,
        kernel_length: int = 32,
        dropout: float = 0.35,
        pool1: int = 2,
        pool2: int = 4,
    ):
        super().__init__()
        F2 = F1 * D

        self.temporal = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
        )

        self.spatial = nn.Sequential(
            nn.Conv2d(F1, F1 * D, kernel_size=(n_ch, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, pool1)),
            nn.Dropout(p=dropout),
        )

        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 32), padding=(0, 16), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, pool2)),
            nn.Dropout(p=dropout),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_ch, n_samples)
            feat = self.separable(self.spatial(self.temporal(dummy)))
            n_feat = feat.shape[1] * feat.shape[2] * feat.shape[3]

        self.classifier = nn.Linear(n_feat, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.separable(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x).squeeze(-1)
