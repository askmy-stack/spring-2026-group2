"""
Graph Attention Module for EEG
------------------------------
Models EEG electrodes as graph nodes with learnable connectivity.
Inspired by GMAEEG paper for non-Euclidean electrode layout.

Key insight: EEG electrodes are not arranged in a grid (like images)
but follow the 10-20 system's 3D brain topology.

References:
- GMAEEG: A Self-Supervised Graph Masked Autoencoder for EEG (IEEE JBHI 2024)
- Graph Attention Networks (Veličković et al., 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict


# Standard 10-20 system electrode positions and neighbors
EEG_ELECTRODE_POSITIONS = {
    # Channel name: (x, y, z) approximate 3D position
    "Fp1": (-0.31, 0.95, 0.00),
    "Fp2": (0.31, 0.95, 0.00),
    "F7": (-0.81, 0.59, 0.00),
    "F3": (-0.55, 0.67, 0.45),
    "Fz": (0.00, 0.72, 0.69),
    "F4": (0.55, 0.67, 0.45),
    "F8": (0.81, 0.59, 0.00),
    "T3": (-1.00, 0.00, 0.00),  # Also called T7
    "C3": (-0.71, 0.00, 0.71),
    "Cz": (0.00, 0.00, 1.00),
    "C4": (0.71, 0.00, 0.71),
    "T4": (1.00, 0.00, 0.00),  # Also called T8
    "T5": (-0.81, -0.59, 0.00),  # Also called P7
    "P3": (-0.55, -0.67, 0.45),
    "Pz": (0.00, -0.72, 0.69),
    "P4": (0.55, -0.67, 0.45),
    "T6": (0.81, -0.59, 0.00),  # Also called P8
    "O1": (-0.31, -0.95, 0.00),
    "O2": (0.31, -0.95, 0.00),
}

# Known electrode pairs with strong functional connectivity
GLOBAL_PAIRS = [
    ("F3", "F4"),  # Frontal bilateral
    ("C3", "C4"),  # Central bilateral
    ("P3", "P4"),  # Parietal bilateral
    ("F7", "F8"),  # Lateral frontal
    ("T3", "T4"),  # Temporal bilateral (key for seizures!)
    ("T5", "T6"),  # Posterior temporal
    ("O1", "O2"),  # Occipital bilateral
]

# CHB-MIT channel names (may differ from standard 10-20)
CHB_MIT_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
]


def build_eeg_adjacency(
    n_channels: int = 16,
    channel_names: Optional[list] = None,
    include_global: bool = True,
    self_loops: bool = True,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Build adjacency matrix for EEG electrode graph.

    Args:
        n_channels: Number of electrodes
        channel_names: List of channel names (for position-based adjacency)
        include_global: Include long-range bilateral connections
        self_loops: Add self-connections (diagonal = 1)
        normalize: Normalize adjacency matrix (D^-0.5 * A * D^-0.5)

    Returns:
        Adjacency matrix (n_channels, n_channels)
    """
    adj = torch.zeros(n_channels, n_channels)

    if channel_names is not None and len(channel_names) == n_channels:
        # Distance-based adjacency
        positions = []
        for name in channel_names:
            # Try to match channel name to known positions
            matched = False
            for key, pos in EEG_ELECTRODE_POSITIONS.items():
                if key.lower() in name.lower():
                    positions.append(pos)
                    matched = True
                    break
            if not matched:
                # Default position if not found
                positions.append((0, 0, 0))

        positions = torch.tensor(positions, dtype=torch.float32)

        # Compute pairwise distances
        for i in range(n_channels):
            for j in range(n_channels):
                dist = torch.norm(positions[i] - positions[j])
                # Neighboring electrodes (distance < 0.6 in normalized coords)
                if dist < 0.6:
                    adj[i, j] = 1.0
                # Medium distance (0.6-1.0) - weaker connection
                elif dist < 1.0:
                    adj[i, j] = 0.5
    else:
        # Default: sequential neighbors + wrap-around
        for i in range(n_channels):
            # Self
            if self_loops:
                adj[i, i] = 1.0
            # Immediate neighbors
            if i > 0:
                adj[i, i - 1] = 1.0
                adj[i - 1, i] = 1.0
            if i < n_channels - 1:
                adj[i, i + 1] = 1.0
                adj[i + 1, i] = 1.0

        # Add global bilateral connections (assumes symmetric montage)
        if include_global and n_channels >= 8:
            half = n_channels // 2
            for i in range(min(half, n_channels - half)):
                adj[i, i + half] = 1.0
                adj[i + half, i] = 1.0

    # Add self-loops
    if self_loops:
        adj = adj + torch.eye(n_channels)
        adj = torch.clamp(adj, 0, 1)

    # Normalize: D^-0.5 * A * D^-0.5
    if normalize:
        degree = adj.sum(dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
        D_inv_sqrt = torch.diag(degree_inv_sqrt)
        adj = D_inv_sqrt @ adj @ D_inv_sqrt

    return adj


class EEGGraphConv(nn.Module):
    """
    Graph Convolutional Layer for EEG.
    Aggregates information from neighboring electrodes.

    Input:  (batch, n_channels, features)
    Output: (batch, n_channels, out_features)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_channels: int = 16,
        learnable_adj: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_channels = n_channels

        # Linear transformation
        self.weight = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Adjacency matrix
        adj = build_eeg_adjacency(n_channels)
        if learnable_adj:
            # Learnable adjacency (initialized from prior)
            self.adj = nn.Parameter(adj)
        else:
            self.register_buffer("adj", adj)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_channels, features)
        batch, n_channels, _ = x.shape

        # Linear transform
        h = self.weight(x)  # (batch, n_channels, out_features)

        # Adjacency (possibly learnable)
        adj = self.adj[:n_channels, :n_channels]
        if hasattr(self, "adj") and isinstance(self.adj, nn.Parameter):
            # Normalize learnable adjacency
            adj = F.softmax(adj, dim=-1)

        # Graph convolution: A * H
        h = torch.einsum("ij,bjf->bif", adj, h)

        # Bias + activation
        h = h + self.bias
        h = F.relu(h)
        h = self.dropout(h)

        return h


class GraphAttention(nn.Module):
    """
    Graph Attention Network (GAT) layer for EEG.
    Learns attention weights between electrodes dynamically.

    Input:  (batch, n_channels, features)
    Output: (batch, n_channels, out_features)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_channels: int = 16,
        num_heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat

        # Multi-head attention projections
        self.W = nn.Linear(in_features, out_features * num_heads, bias=False)
        self.a = nn.Parameter(torch.zeros(num_heads, 2 * out_features))
        nn.init.xavier_uniform_(self.a)

        # Prior adjacency for masking
        self.register_buffer("adj_mask", build_eeg_adjacency(n_channels, normalize=False))

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

        if concat:
            self.out_dim = out_features * num_heads
        else:
            self.out_dim = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_channels, in_features)
        batch, n_channels, _ = x.shape

        # Linear transform: (batch, n_channels, num_heads * out_features)
        h = self.W(x)
        h = h.view(batch, n_channels, self.num_heads, self.out_features)

        # Compute attention coefficients
        # Self-attention: concat(Wh_i, Wh_j) for all pairs
        h_i = h.unsqueeze(2).expand(-1, -1, n_channels, -1, -1)  # (B, N, N, H, F)
        h_j = h.unsqueeze(1).expand(-1, n_channels, -1, -1, -1)  # (B, N, N, H, F)
        concat_h = torch.cat([h_i, h_j], dim=-1)  # (B, N, N, H, 2F)

        # Attention scores
        e = torch.einsum("bijnf,hf->bijh", concat_h, self.a)  # (B, N, N, H)
        e = self.leaky_relu(e)

        # Mask with prior adjacency (only attend to neighbors)
        mask = self.adj_mask[:n_channels, :n_channels]
        e = e.masked_fill(mask.unsqueeze(0).unsqueeze(-1) == 0, float("-inf"))

        # Softmax over neighbors
        alpha = F.softmax(e, dim=2)  # (B, N, N, H)
        alpha = self.dropout(alpha)

        # Aggregate: weighted sum of neighbor features
        out = torch.einsum("bijh,bjhf->bihf", alpha, h)  # (B, N, H, F)

        if self.concat:
            out = out.view(batch, n_channels, -1)  # (B, N, H*F)
        else:
            out = out.mean(dim=2)  # (B, N, F)

        return out


class EEGGraphNetwork(nn.Module):
    """
    Full Graph Neural Network for EEG electrode modeling.
    Combines multiple graph layers with residual connections.

    Input:  (batch, n_channels, time_steps)
    Output: (batch, hidden_dim)
    """

    def __init__(
        self,
        n_channels: int = 16,
        time_steps: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_channels = n_channels

        # Temporal feature extraction per channel
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # Graph layers
        self.graph_layers = nn.ModuleList()
        in_features = 64
        for i in range(num_layers):
            out_features = hidden_dim if i == num_layers - 1 else in_features
            self.graph_layers.append(
                GraphAttention(
                    in_features=in_features,
                    out_features=out_features // num_heads,
                    n_channels=n_channels,
                    num_heads=num_heads,
                    dropout=dropout,
                )
            )
            in_features = out_features

        # Global readout
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * n_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_channels, time_steps)
        batch, n_channels, time_steps = x.shape

        # Extract temporal features per channel
        channel_features = []
        for c in range(n_channels):
            c_feat = self.temporal_conv(x[:, c:c+1, :])  # (batch, 64, 1)
            channel_features.append(c_feat.squeeze(-1))

        # Stack: (batch, n_channels, 64)
        h = torch.stack(channel_features, dim=1)

        # Graph convolutions
        for layer in self.graph_layers:
            h = layer(h)
            h = F.relu(h)

        # Global readout: flatten and project
        h = h.view(batch, -1)
        h = self.readout(h)

        return h
