from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    """
    EEGNet-style CNN for EEG window classification.

    Input:  (B, C, T)
    Output: (B,) logits for binary classification.
    """
    def __init__(
        self,
        n_ch: int,
        n_samples: int,
        F1: int = 8,
        D: int = 2,
        F2: int | None = None,
        kernel_length: int = 64,
        dropout: float = 0.25,
        pool1: int = 4,
        pool2: int = 8,
        sep_kernel: int = 16,
    ):
        super().__init__()
        if F2 is None:
            F2 = F1 * D

        self.n_ch = int(n_ch)

        # (B, C, T) -> (B, 1, C, T)
        self.conv_temporal = nn.Conv2d(
            1, F1, kernel_size=(1, kernel_length),
            padding=(0, kernel_length // 2), bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1)

        # depthwise spatial conv across channels
        self.conv_spatial = nn.Conv2d(
            F1, F1 * D, kernel_size=(self.n_ch, 1),
            groups=F1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)

        self.pool1 = nn.AvgPool2d(kernel_size=(1, pool1), stride=(1, pool1))
        self.drop1 = nn.Dropout(dropout)

        # separable temporal conv
        self.conv_sep_dw = nn.Conv2d(
            F1 * D, F1 * D, kernel_size=(1, sep_kernel),
            padding=(0, sep_kernel // 2), groups=F1 * D, bias=False
        )
        self.conv_sep_pw = nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)

        self.pool2 = nn.AvgPool2d(kernel_size=(1, pool2), stride=(1, pool2))
        self.drop2 = nn.Dropout(dropout)

        # infer classifier input dim
        with torch.no_grad():
            dummy = torch.zeros(1, self.n_ch, int(n_samples))
            fdim = self._forward_features(dummy).shape[-1]
        self.classifier = nn.Linear(fdim, 1)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (B, 1, C, T)

        x = self.conv_temporal(x)
        x = self.bn1(x)

        x = self.conv_spatial(x)
        x = self.bn2(x)
        x = F.elu(x)

        x = self.pool1(x)
        x = self.drop1(x)

        x = self.conv_sep_dw(x)
        x = self.conv_sep_pw(x)
        x = self.bn3(x)
        x = F.elu(x)

        x = self.pool2(x)
        x = self.drop2(x)

        return torch.flatten(x, start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._forward_features(x)
        return self.classifier(feats).squeeze(-1)