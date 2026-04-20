"""
EEG Mamba: State-Space Model for Seizure Detection
---------------------------------------------------
O(n) linear complexity, 4× faster than transformers.

Variants:
- EEGMamba: Basic Mamba encoder + classifier
- EEGMambaMoE: Mamba with Mixture of Experts

References:
- Mamba: Linear-Time Sequence Modeling
- EEGMamba: Bidirectional State Space Model for EEG
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from modules.mamba_block import MambaEncoder, BidirectionalMamba, MambaBlock
from modules.mixture_of_experts import MixtureOfExperts, MoEWithUniversal


class EEGMamba(nn.Module):
    """
    EEG Mamba for seizure detection.

    Architecture:
        Input (batch, 16, 256)
        → Input Projection
        → N × Bidirectional Mamba Blocks
        → Global Pooling
        → Classification Head
        → Logits

    Args:
        n_channels: Number of EEG channels (default: 16)
        time_steps: Number of time steps (default: 256)
        d_model: Model dimension (default: 128)
        d_state: SSM state dimension (default: 16)
        n_layers: Number of Mamba layers (default: 4)
        dropout: Dropout rate (default: 0.1)
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
        self.n_channels = n_channels
        self.time_steps = time_steps
        self.d_model = d_model

        # Mamba encoder
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

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: EEG signal (batch, n_channels, time_steps)

        Returns:
            logits: Classification logits (batch, 1)
        """
        # Encode
        features = self.encoder(x)  # (batch, time_steps, d_model)

        # Pool
        pooled = features.mean(dim=1)  # (batch, d_model)

        # Classify
        logits = self.classifier(pooled)

        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get encoded features."""
        return self.encoder.get_features(x)


class EEGMambaMoE(nn.Module):
    """
    EEG Mamba with Mixture of Experts.

    Each expert specializes in different seizure patterns:
    - Expert 1: Focal seizures
    - Expert 2: Generalized seizures
    - Expert 3: Absence seizures
    - etc.

    Architecture:
        Input → Mamba Encoder → MoE Layer → Classifier

    Args:
        n_channels: Number of EEG channels
        time_steps: Number of time steps
        d_model: Model dimension
        n_layers: Number of Mamba layers
        num_experts: Number of MoE experts
        top_k: Number of experts per token
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
        self.n_channels = n_channels
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(n_channels, d_model),
            nn.LayerNorm(d_model),
        )

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, time_steps, d_model) * 0.02)

        # Mamba + MoE layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                nn.ModuleDict({
                    "mamba": BidirectionalMamba(d_model, d_state, dropout=dropout),
                    "moe": MoEWithUniversal(
                        input_dim=d_model,
                        hidden_dim=d_model * 4,
                        output_dim=d_model,
                        num_experts=num_experts,
                        top_k=top_k,
                        dropout=dropout,
                    ),
                    "norm": nn.LayerNorm(d_model),
                })
            )

        # Output normalization
        self.norm = nn.LayerNorm(d_model)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self, x: torch.Tensor, return_aux_loss: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass.

        Args:
            x: EEG signal (batch, n_channels, time_steps)
            return_aux_loss: Whether to return MoE auxiliary loss

        Returns:
            logits: Classification logits
            aux_loss: (optional) MoE load balancing loss
        """
        # Reshape and project
        x = x.permute(0, 2, 1)  # (batch, time_steps, n_channels)
        x = self.input_proj(x)

        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.pos_embed[:, :seq_len, :]

        # Process layers
        total_aux_loss = 0.0
        for layer in self.layers:
            # Mamba
            x = layer["mamba"](x)

            # MoE
            moe_out, aux_loss = layer["moe"](x)
            x = layer["norm"](x + moe_out)

            total_aux_loss = total_aux_loss + aux_loss

        # Final norm
        x = self.norm(x)

        # Pool and classify
        pooled = x.mean(dim=1)
        logits = self.classifier(pooled)

        if return_aux_loss:
            return logits, total_aux_loss
        return logits


class STAdaptiveMamba(nn.Module):
    """
    Spatio-Temporal Adaptive Mamba.
    Handles variable channel counts and sequence lengths.
    """

    def __init__(
        self,
        max_channels: int = 64,
        max_time_steps: int = 1024,
        d_model: int = 128,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Adaptive spatial convolution (handles variable channels)
        self.spatial_conv = nn.Conv1d(1, d_model, kernel_size=1)

        # CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Learnable positional encoding
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_time_steps + 1, d_model) * 0.02
        )

        # Mamba layers
        self.layers = nn.ModuleList([
            BidirectionalMamba(d_model, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, n_channels, time_steps = x.shape

        # Flatten spatial dimension
        x = x.view(batch * n_channels, 1, time_steps)
        x = self.spatial_conv(x)  # (batch * n_channels, d_model, time_steps)
        x = x.permute(0, 2, 1)  # (batch * n_channels, time_steps, d_model)

        # Reshape back
        x = x.view(batch, n_channels * time_steps, self.d_model)

        # Add CLS token
        cls = self.cls_token.expand(batch, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # Add positional encoding
        x = x + self.pos_embed[:, :x.size(1), :]

        # Mamba layers
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        # Use CLS token for classification
        cls_out = x[:, 0]
        logits = self.classifier(cls_out)

        return logits


if __name__ == "__main__":
    # Test EEGMamba
    print("Testing EEGMamba...")
    model = EEGMamba(n_channels=16, time_steps=256, d_model=128, n_layers=4)
    x = torch.randn(8, 16, 256)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test EEGMambaMoE
    print("\nTesting EEGMambaMoE...")
    model_moe = EEGMambaMoE(n_channels=16, time_steps=256, num_experts=8)
    out, aux_loss = model_moe(x, return_aux_loss=True)
    print(f"Output shape: {out.shape}")
    print(f"Aux loss: {aux_loss:.4f}")
    print(f"Parameters: {sum(p.numel() for p in model_moe.parameters()):,}")
