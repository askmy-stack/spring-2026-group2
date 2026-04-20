"""
Drop-in regularisation primitives for improved benchmark models.

- ``DropPath``: stochastic depth — drops whole residual branches per-sample.
- ``SqueezeExcite1D``: channel-wise gating for ``(B, C, T)`` tensors.
- ``wrap_with_droppath``: swap a module's Dropout layer for Dropout + DropPath,
  used by im1/im2/im4/im7 when ``stochastic_depth > 0``.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class DropPath(nn.Module):
    """
    Stochastic Depth — randomly zero out the entire input with probability ``p``.

    At training time, each sample in the batch is independently dropped
    (multiplied by 0) with probability ``p``; surviving samples are
    rescaled by ``1 / (1 - p)`` so the expected activation is preserved.
    At eval time, acts as identity.

    Args:
        p: Drop probability in [0, 1). Default 0.0 (no-op).

    Example:
        >>> drop = DropPath(0.2)
        >>> drop.train()
        >>> out = drop(torch.randn(8, 256))
        >>> assert out.shape == (8, 256)
    """

    def __init__(self, p: float = 0.0):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"DropPath p must be in [0, 1); got {p}")
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep_prob = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # per-sample broadcast
        mask = x.new_empty(shape).bernoulli_(keep_prob).div_(keep_prob)
        return x * mask

    def extra_repr(self) -> str:
        return f"p={self.p}"


class SqueezeExcite1D(nn.Module):
    """
    Channel-wise squeeze-and-excitation for ``(B, C, T)`` tensors.

    Args:
        channels: Number of channels to gate.
        reduction: Bottleneck reduction factor (default 4).

    Example:
        >>> se = SqueezeExcite1D(16, reduction=4)
        >>> out = se(torch.randn(2, 16, 128))
        >>> assert out.shape == (2, 16, 128)
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, hidden)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden, channels)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) — pool over T, gate per channel
        pooled = x.mean(dim=-1)                        # (B, C)
        s = self.gate(self.fc2(self.act(self.fc1(pooled))))
        return x * s.unsqueeze(-1)


def wrap_with_droppath(
    model: nn.Module,
    attr_name: str = "dropout_layer",
    p: float = 0.1,
) -> nn.Module:
    """
    Replace ``model.<attr_name>`` with ``Sequential(original, DropPath(p))``.

    Used by improved benchmark subclasses to inject stochastic depth on
    their post-pool feature path without surgically editing the parent's
    forward pass. Returns the (mutated) model for convenience.

    Args:
        model: Target module holding a ``nn.Dropout`` attribute.
        attr_name: Attribute to wrap (must exist on ``model``).
        p: DropPath probability. Pass ``0.0`` to skip wrapping.

    Raises:
        AttributeError: If ``attr_name`` is missing on ``model``.
    """
    if p <= 0.0:
        return model
    if not hasattr(model, attr_name):
        raise AttributeError(
            f"{type(model).__name__} has no attribute '{attr_name}' to wrap"
        )
    existing = getattr(model, attr_name)
    setattr(model, attr_name, nn.Sequential(existing, DropPath(p)))
    return model
