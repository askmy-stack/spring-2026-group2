"""
M6: GraphBiLSTM (GMAEEG-style)
------------------------------
Graph neural network over EEG electrode topology,
followed by BiLSTM for temporal modeling.

Expected F1: 0.78-0.85 (with pre-training)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from modules.graph_attention import GraphAttention, EEGGraphConv, build_eeg_adjacency
from modules.pretrained_encoders import load_pretrained_encoder


class M6_GraphBiLSTM(nn.Module):
    """
    Graph BiLSTM for EEG.

    Models EEG electrodes as graph nodes with learnable connectivity,
    then applies BiLSTM for temporal modeling.

    Architecture:
        Input (batch, 16, 256)
        → [Optional: Pretrained Encoder]
        → Temporal Feature Extraction (per channel)
        → Graph Attention (across channels)
        → BiLSTM (temporal modeling)
        → Multi-Head Attention
        → AvgPool + MaxPool
        → FC Head → Logits

    Args:
        n_channels: Number of EEG channels (default: 16)
        time_steps: Number of time steps (default: 256)
        hidden_size: Hidden size (default: 128)
        num_layers: Number of LSTM layers (default: 2)
        num_heads: Number of attention heads (default: 4)
        dropout: Dropout rate (default: 0.3)
        use_pretrained: Whether to use pretrained encoder
        pretrained_encoder: Name of pretrained encoder
    """

    def __init__(
        self,
        n_channels: int = 16,
        time_steps: int = 256,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
        use_pretrained: bool = False,
        pretrained_encoder: Optional[str] = "cbramod",
    ):
        super().__init__()
        self.n_channels = n_channels
        self.time_steps = time_steps
        self.hidden_size = hidden_size
        self.use_pretrained = use_pretrained

        # Optional pretrained encoder
        if use_pretrained and pretrained_encoder:
            self.pretrained = load_pretrained_encoder(
                pretrained_encoder,
                n_channels=n_channels,
                time_steps=time_steps,
                hidden_dim=hidden_size,
                pretrained=True,
                freeze=True,
            )
        else:
            self.pretrained = None

        # Temporal feature extraction per channel
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )

        # Number of time steps after convolution
        conv_time_steps = time_steps // 2

        # Graph attention layers
        self.graph_layers = nn.ModuleList([
            GraphAttention(
                in_features=64,
                out_features=hidden_size // num_heads,
                n_channels=n_channels,
                num_heads=num_heads,
                dropout=dropout,
            ),
            GraphAttention(
                in_features=hidden_size,
                out_features=hidden_size // num_heads,
                n_channels=n_channels,
                num_heads=num_heads,
                dropout=dropout,
            ),
        ])

        self.graph_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size),
            nn.LayerNorm(hidden_size),
        ])

        # BiLSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_size * n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_size = hidden_size * 2

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_out_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(lstm_out_size)

        # Pooling normalization
        self.pool_norm = nn.LayerNorm(lstm_out_size * 2)
        self.dropout_layer = nn.Dropout(dropout)

        # Classification head
        fc_input_size = lstm_out_size * 2
        if use_pretrained:
            fc_input_size += hidden_size

        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: EEG signal (batch, n_channels, time_steps)

        Returns:
            logits: Classification logits (batch, 1)
        """
        batch, n_channels, time_steps = x.shape

        # Get pretrained features if available
        pretrained_feat = None
        if self.pretrained is not None:
            pretrained_feat = self.pretrained(x)

        # Extract temporal features per channel
        # Process each channel independently
        channel_features = []
        for c in range(n_channels):
            c_input = x[:, c:c+1, :]  # (batch, 1, time_steps)
            c_feat = self.temporal_conv(c_input)  # (batch, 64, time_steps/2)
            channel_features.append(c_feat)

        # Stack: (batch, n_channels, 64, time_steps/2)
        channel_features = torch.stack(channel_features, dim=1)
        conv_time_steps = channel_features.size(-1)

        # Process each time step with graph attention
        time_features = []
        for t in range(conv_time_steps):
            # Get features at time t: (batch, n_channels, 64)
            t_feat = channel_features[:, :, :, t]

            # Apply graph attention layers
            for graph_layer, graph_norm in zip(self.graph_layers, self.graph_norms):
                t_feat = graph_layer(t_feat)
                t_feat = F.relu(t_feat)
                t_feat = graph_norm(t_feat)

            # Flatten channels: (batch, n_channels * hidden_size)
            t_feat = t_feat.view(batch, -1)
            time_features.append(t_feat)

        # Stack: (batch, conv_time_steps, n_channels * hidden_size)
        graph_out = torch.stack(time_features, dim=1)

        # BiLSTM
        lstm_out, _ = self.lstm(graph_out)  # (batch, conv_time_steps, hidden_size * 2)

        # Self-attention
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attended = self.attn_norm(attended + lstm_out)

        # Global pooling: avg + max
        avg_pool = attended.mean(dim=1)
        max_pool = attended.max(dim=1).values
        pooled = torch.cat([avg_pool, max_pool], dim=1)

        # Combine with pretrained features
        if pretrained_feat is not None:
            pooled = torch.cat([pooled, pretrained_feat], dim=1)

        # Normalize and dropout
        pooled_to_norm = pooled[:, :self.hidden_size * 4]
        pooled_normed = self.pool_norm(pooled_to_norm)
        if pretrained_feat is not None:
            pooled = torch.cat([pooled_normed, pretrained_feat], dim=1)
        else:
            pooled = pooled_normed
        pooled = self.dropout_layer(pooled)

        # Classify
        logits = self.fc(pooled)

        return logits

    def get_graph_weights(self) -> torch.Tensor:
        """Get learned graph adjacency for interpretability."""
        # Return adjacency from first graph layer
        return self.graph_layers[0].adj_mask


if __name__ == "__main__":
    # Test
    model = M6_GraphBiLSTM(n_channels=16, time_steps=256, use_pretrained=False)
    x = torch.randn(8, 16, 256)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
