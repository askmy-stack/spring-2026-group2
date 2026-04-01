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


class _ResidualConvBlock1D(nn.Module):
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
        self.se = _SEBlock1D(channels)
        self.act = nn.SiLU()
        self.drop = nn.Dropout1d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        y = self.se(y)
        y = self.act(y + x)
        return self.drop(y)


class _DilatedTCNBlock1D(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        k = 3
        pad = dilation
        self.block = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=k,
                padding=pad,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
            nn.SiLU(),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=k,
                padding=pad,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
        )
        self.se = _SEBlock1D(channels)
        self.act = nn.SiLU()
        self.drop = nn.Dropout1d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        y = self.se(y)
        y = self.act(y + x)
        return self.drop(y)


class CNNMixture1D(nn.Module):
    """
    Gated two-branch CNN for seizure detection.

    Branch A: local residual convolutions
    Branch B: dilated temporal convolutions (TCN-style)

    Input:  (B, 16, 256)
    Output: (B,) logits
    """

    def __init__(
        self,
        n_ch: int = 16,
        n_samples: int = 256,
        base_channels: int = 32,
        dropout: float = 0.25,
    ):
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.stem = nn.Sequential(
            nn.Conv1d(n_ch, c1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(c1),
            nn.SiLU(),
        )
        self.down1 = nn.Sequential(
            nn.Conv1d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(c2),
            nn.SiLU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv1d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(c3),
            nn.SiLU(),
        )

        self.local_branch = nn.Sequential(
            _ResidualConvBlock1D(c3, kernel_size=5, dropout=dropout),
            _ResidualConvBlock1D(c3, kernel_size=3, dropout=dropout),
        )
        self.tcn_branch = nn.Sequential(
            _DilatedTCNBlock1D(c3, dilation=1, dropout=dropout),
            _DilatedTCNBlock1D(c3, dilation=2, dropout=dropout),
            _DilatedTCNBlock1D(c3, dilation=4, dropout=dropout),
        )

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(c3, c3 // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(c3 // 2, 2),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(c3, c3 // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(c3 // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)

        y_local = self.local_branch(x)
        y_tcn = self.tcn_branch(x)

        gate_logits = self.gate(x)
        gate = torch.softmax(gate_logits, dim=1)
        mix = gate[:, 0].view(-1, 1, 1) * y_local + gate[:, 1].view(-1, 1, 1) * y_tcn

        return self.head(mix).squeeze(-1)

