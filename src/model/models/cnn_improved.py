from __future__ import annotations

import torch
import torch.nn as nn


class _SEBlock1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1)
        return x * w


class _ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dropout: float = 0.2):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.SiLU()
        self.drop = nn.Dropout1d(dropout)
        self.se = _SEBlock1D(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        out = self.act(out + x)
        out = self.se(out)
        return self.drop(out)


class CNNImproved1D(nn.Module):
    """
    Stronger 1D CNN baseline for EEG.

    Input:  (B, 16, 256)
    Output: (B,) logits
    """

    def __init__(
        self,
        n_ch: int = 16,
        n_samples: int = 256,
        base_channels: int = 48,
        dropout: float = 0.25,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(n_ch, base_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.SiLU(),
        )
        self.res1 = _ResidualBlock1D(base_channels, kernel_size=5, dropout=dropout)
        self.down1 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(base_channels * 2),
            nn.SiLU(),
        )

        self.res2 = _ResidualBlock1D(base_channels * 2, kernel_size=5, dropout=dropout)
        self.down2 = nn.Sequential(
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(base_channels * 4),
            nn.SiLU(),
        )

        self.res3 = _ResidualBlock1D(base_channels * 4, kernel_size=3, dropout=dropout)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.res1(x)
        x = self.down1(x)
        x = self.res2(x)
        x = self.down2(x)
        x = self.res3(x)
        return self.head(x).squeeze(-1)
