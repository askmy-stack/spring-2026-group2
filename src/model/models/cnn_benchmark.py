from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBenchmark1D(nn.Module):
    """
    Benchmark CNN for EEG seizure detection.

    Input:  (B, 16, 256)
    Output: (B,) logits
    """

    def __init__(
        self,
        n_ch: int = 16,
        n_samples: int = 256,
        base_channels: int = 32,
        dropout: float = 0.3,
        k1: int = 7,
        k2: int = 5,
        k3: int = 3,
        pool_type: str = "max",  # "max" or "avg"
    ):
        super().__init__()
        assert pool_type in ("max", "avg")

        self.pool = nn.MaxPool1d(2) if pool_type == "max" else nn.AvgPool1d(2)
        self.drop = nn.Dropout(dropout)

        def conv_block(cin, cout, k):
            pad = k // 2
            return nn.Sequential(
                nn.Conv1d(cin, cout, kernel_size=k, padding=pad, bias=False),
                nn.BatchNorm1d(cout),
                nn.GELU(),
            )

        self.b1 = conv_block(n_ch, base_channels, k1)
        self.b2 = conv_block(base_channels, base_channels * 2, k2)
        self.b3 = conv_block(base_channels * 2, base_channels * 4, k3)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.b1(x))
        x = self.drop(x)

        x = self.pool(self.b2(x))
        x = self.drop(x)

        x = self.pool(self.b3(x))
        x = self.drop(x)

        return self.head(x).squeeze(-1)