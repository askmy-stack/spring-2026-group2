from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn


class _InceptionBlock1D(nn.Module):
    """
    Multi-scale temporal block for EEG windows.

    Branches:
    - multiple temporal kernels (e.g. 3, 7, 15, 31)
    - pooled branch for local invariance
    """

    def __init__(
        self,
        in_ch: int,
        out_per_branch: int,
        kernels: Iterable[int] = (3, 7, 15, 31),
        bottleneck_ch: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        kernels = tuple(int(k) for k in kernels)
        self.kernels: Tuple[int, ...] = kernels

        use_bottleneck = in_ch > bottleneck_ch
        self.bottleneck = (
            nn.Conv1d(in_ch, bottleneck_ch, kernel_size=1, bias=False)
            if use_bottleneck
            else nn.Identity()
        )
        conv_in = bottleneck_ch if use_bottleneck else in_ch

        self.conv_branches = nn.ModuleList(
            [
                nn.Conv1d(
                    conv_in,
                    out_per_branch,
                    kernel_size=k,
                    padding=k // 2,
                    bias=False,
                )
                for k in kernels
            ]
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_ch, out_per_branch, kernel_size=1, bias=False),
        )

        out_ch = out_per_branch * (len(kernels) + 1)
        self.post = nn.Sequential(
            nn.BatchNorm1d(out_ch),
            nn.SiLU(),
            nn.Dropout1d(dropout),
        )
        self.out_ch = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.bottleneck(x)
        ys = [conv(z) for conv in self.conv_branches]
        ys.append(self.pool_branch(x))
        y = torch.cat(ys, dim=1)
        return self.post(y)


class CNNMultiScale1D(nn.Module):
    """
    Inception-style multi-scale CNN for seizure detection.

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

        branch1 = max(base_channels // 2, 12)
        branch2 = max(base_channels, 24)

        self.stem = nn.Sequential(
            nn.Conv1d(n_ch, base_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.SiLU(),
        )

        self.inc1 = _InceptionBlock1D(
            in_ch=base_channels,
            out_per_branch=branch1,
            kernels=(3, 7, 15, 31),
            bottleneck_ch=base_channels,
            dropout=dropout,
        )
        self.inc2 = _InceptionBlock1D(
            in_ch=self.inc1.out_ch,
            out_per_branch=branch1,
            kernels=(3, 7, 15, 31),
            bottleneck_ch=base_channels,
            dropout=dropout,
        )
        self.res1 = nn.Sequential(
            nn.Conv1d(base_channels, self.inc2.out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.inc2.out_ch),
        )

        self.down = nn.MaxPool1d(kernel_size=2)

        self.inc3 = _InceptionBlock1D(
            in_ch=self.inc2.out_ch,
            out_per_branch=branch2,
            kernels=(3, 7, 15, 31),
            bottleneck_ch=self.inc2.out_ch // 2,
            dropout=dropout,
        )
        self.inc4 = _InceptionBlock1D(
            in_ch=self.inc3.out_ch,
            out_per_branch=branch2,
            kernels=(3, 7, 15, 31),
            bottleneck_ch=self.inc3.out_ch // 2,
            dropout=dropout,
        )
        self.res2 = nn.Sequential(
            nn.Conv1d(self.inc2.out_ch, self.inc4.out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.inc4.out_ch),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.inc4.out_ch, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(x)
        y1 = self.inc1(x0)
        y2 = self.inc2(y1)
        y2 = torch.silu(y2 + self.res1(x0))

        y2d = self.down(y2)
        y3 = self.inc3(y2d)
        y4 = self.inc4(y3)
        y4 = torch.silu(y4 + self.res2(y2d))

        return self.head(y4).squeeze(-1)
