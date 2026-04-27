"""
EEG Mamba and EEG Mamba-MoE — Directory 4 of 4.

O(n) linear complexity for long EEG sequences.

References:
    Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023)
    EEGMamba: Bidirectional State Space Model for EEG Decoding
"""
import logging
from typing import Tuple

import torch
import torch.nn as nn

from ..modules.mamba_block import MambaEncoder
from ..modules.mixture_of_experts import MoEWithUniversal

logger = logging.getLogger(__name__)


class EEGMamba(nn.Module):
    """
    Bidirectional Mamba encoder + classification head for EEG seizure detection.

    Architecture:
        Input (batch, n_channels, time_steps)
        -> MambaEncoder (N x BidirectionalMambaBlock)
        -> Mean pool -> LayerNorm -> FC Head -> Logits

    Args:
        n_channels: Number of EEG channels (default: 16).
        time_steps: Timesteps per window (default: 256).
        d_model: Model dimension (default: 128).
        d_state: SSM state dimension (default: 16).
        d_conv: Depthwise conv kernel size (default: 4).
        n_layers: Number of Mamba blocks (default: 4).
        dropout: Dropout probability (default: 0.1).
        bidirectional: Use bidirectional Mamba if True (default: True).

    Example:
        >>> model = EEGMamba(n_channels=16, d_model=128)
        >>> logits = model(torch.randn(4, 16, 256))
        >>> assert logits.shape == (4, 1)
    """

    def __init__(
        self,
        n_channels: int = 16,
        time_steps: int = 256,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.encoder = MambaEncoder(
            n_channels=n_channels,
            time_steps=time_steps,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            n_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        self.classifier = _build_classifier(d_model, dropout)

    def forward(self, eeg_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg_input: shape (batch, n_channels, time_steps)

        Returns:
            Logits, shape (batch, 1).
        """
        features = self.encoder(eeg_input)
        pooled = features.mean(dim=1)
        return self.classifier(pooled)


class EEGMambaMoE(nn.Module):
    """
    EEG Mamba with Mixture-of-Experts for multi-pattern seizure detection.

    Architecture:
        Input (batch, n_channels, time_steps)
        -> MambaEncoder -> N x MoEWithUniversal (per-layer)
        -> Mean pool -> LayerNorm -> FC Head -> Logits

    Args:
        n_channels: Number of EEG channels (default: 16).
        time_steps: Timesteps per window (default: 256).
        d_model: Model dimension (default: 128).
        d_state: SSM state dimension (default: 16).
        n_layers: Mamba encoder layers (default: 4).
        num_experts: MoE specialist count (default: 8).
        top_k: Experts activated per token (default: 2).
        dropout: Dropout probability (default: 0.1).

    Example:
        >>> model = EEGMambaMoE(n_channels=16, d_model=128, num_experts=4)
        >>> logits, lb_loss = model(torch.randn(2, 16, 256))
        >>> assert logits.shape == (2, 1)
    """

    def __init__(
        self,
        n_channels: int = 16,
        time_steps: int = 256,
        d_model: int = 128,
        d_state: int = 16,
        n_layers: int = 4,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = MambaEncoder(
            n_channels=n_channels, time_steps=time_steps, d_model=d_model,
            d_state=d_state, n_layers=n_layers, bidirectional=True, dropout=dropout,
        )
        self.moe_layers = nn.ModuleList([
            MoEWithUniversal(d_model, num_experts, top_k, dropout) for _ in range(n_layers)
        ])
        self.classifier = _build_classifier(d_model, dropout)

    def forward(self, eeg_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            eeg_input: shape (batch, n_channels, time_steps)

        Returns:
            (logits, load_balance_loss): logits shape (batch, 1).
        """
        encoded = self.encoder(eeg_input)
        moe_out, total_lb_loss = self._apply_moe_layers(encoded)
        pooled = moe_out.mean(dim=1)
        return self.classifier(pooled), total_lb_loss

    def _apply_moe_layers(self, encoded: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pass encoded features through all MoE layers; accumulate load balance loss."""
        total_lb = torch.tensor(0.0, device=encoded.device)
        hidden = encoded
        for moe_layer in self.moe_layers:
            moe_out, lb_loss = moe_layer(hidden)
            hidden = moe_out
            total_lb = total_lb + lb_loss
        return hidden, total_lb


def _build_classifier(d_model: int, dropout: float) -> nn.Sequential:
    """Build two-layer classification head with LayerNorm."""
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, d_model // 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_model // 2, 1),
    )
