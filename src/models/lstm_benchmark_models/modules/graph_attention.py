"""
Graph Attention Module for LSTM Benchmark Models.

Models EEG electrodes as graph nodes with learnable connectivity.

References:
    GMAEEG: A Self-Supervised Graph Masked Autoencoder for EEG (IEEE JBHI 2024)
    Graph Attention Networks (Veličković et al., 2018)
"""
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

EEG_ELECTRODE_POSITIONS = {
    "Fp1": (-0.31, 0.95, 0.00), "Fp2": (0.31, 0.95, 0.00),
    "F7": (-0.81, 0.59, 0.00),  "F3": (-0.55, 0.67, 0.45),
    "Fz": (0.00, 0.72, 0.69),   "F4": (0.55, 0.67, 0.45),
    "F8": (0.81, 0.59, 0.00),   "T3": (-1.00, 0.00, 0.00),
    "C3": (-0.71, 0.00, 0.71),  "Cz": (0.00, 0.00, 1.00),
    "C4": (0.71, 0.00, 0.71),   "T4": (1.00, 0.00, 0.00),
    "T5": (-0.81, -0.59, 0.00), "P3": (-0.55, -0.67, 0.45),
    "Pz": (0.00, -0.72, 0.69),  "P4": (0.55, -0.67, 0.45),
    "T6": (0.81, -0.59, 0.00),  "O1": (-0.31, -0.95, 0.00),
    "O2": (0.31, -0.95, 0.00),
}


def build_eeg_adjacency(
    n_channels: int = 16,
    channel_names: Optional[list] = None,
    self_loops: bool = True,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Build adjacency matrix for the EEG electrode graph.

    Args:
        n_channels: Number of EEG electrodes.
        channel_names: Optional list of channel names for position-based connectivity.
        self_loops: Whether to add self-connections.
        normalize: Whether to apply symmetric normalisation (D^-0.5 A D^-0.5).

    Returns:
        Adjacency matrix of shape (n_channels, n_channels).

    Example:
        >>> adj = build_eeg_adjacency(n_channels=16)
        >>> assert adj.shape == (16, 16)
    """
    adj = _build_raw_adjacency(n_channels, channel_names, self_loops)
    if normalize:
        adj = _normalize_adjacency(adj)
    return adj


def _build_raw_adjacency(
    n_channels: int, channel_names: Optional[list], self_loops: bool
) -> torch.Tensor:
    """Build raw (un-normalised) adjacency from channel positions or sequential fallback."""
    adj = torch.zeros(n_channels, n_channels)
    if channel_names is not None and len(channel_names) == n_channels:
        adj = _position_based_adjacency(adj, channel_names, n_channels)
    else:
        adj = _sequential_adjacency(adj, n_channels, self_loops)
    if self_loops:
        adj = torch.clamp(adj + torch.eye(n_channels), 0, 1)
    return adj


def _position_based_adjacency(
    adj: torch.Tensor, channel_names: list, n_channels: int
) -> torch.Tensor:
    """Connect electrodes based on 3D proximity in the 10-20 system."""
    positions = _resolve_positions(channel_names)
    for i in range(n_channels):
        for j in range(n_channels):
            dist = torch.norm(positions[i] - positions[j])
            if dist < 0.6:
                adj[i, j] = 1.0
            elif dist < 1.0:
                adj[i, j] = 0.5
    return adj


def _resolve_positions(channel_names: list) -> torch.Tensor:
    """Map channel names to 3D coordinates, defaulting to origin if unknown."""
    positions = []
    for name in channel_names:
        matched = next(
            (pos for key, pos in EEG_ELECTRODE_POSITIONS.items() if key.lower() in name.lower()),
            (0.0, 0.0, 0.0),
        )
        positions.append(matched)
    return torch.tensor(positions, dtype=torch.float32)


def _sequential_adjacency(adj: torch.Tensor, n_channels: int, self_loops: bool) -> torch.Tensor:
    """Connect adjacent channels sequentially + bilateral long-range pairs."""
    for i in range(n_channels):
        if self_loops:
            adj[i, i] = 1.0
        for neighbor in [i - 1, i + 1]:
            if 0 <= neighbor < n_channels:
                adj[i, neighbor] = 1.0
    if n_channels >= 8:
        half = n_channels // 2
        for i in range(min(half, n_channels - half)):
            adj[i, i + half] = 1.0
            adj[i + half, i] = 1.0
    return adj


def _normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    """Apply symmetric normalisation: D^-0.5 A D^-0.5."""
    degree = adj.sum(dim=1)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
    diag = torch.diag(degree_inv_sqrt)
    return diag @ adj @ diag


class GraphAttention(nn.Module):
    """
    Graph Attention Network layer for EEG electrode graphs.

    Args:
        in_features: Input feature dimension per node.
        out_features: Output feature dimension per attention head.
        n_channels: Number of EEG electrodes (graph nodes).
        num_heads: Number of attention heads.
        dropout: Dropout probability on attention weights.
        concat: If True, concatenate head outputs; else average.

    Example:
        >>> layer = GraphAttention(in_features=64, out_features=32, n_channels=16)
        >>> out = layer(torch.randn(4, 16, 64))
        >>> assert out.shape == (4, 16, 128)  # concat with 4 heads
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
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.weight = nn.Linear(in_features, out_features * num_heads, bias=False)
        self.attention_vec = nn.Parameter(torch.zeros(num_heads, 2 * out_features))
        nn.init.xavier_uniform_(self.attention_vec)
        self.register_buffer("adj_mask", build_eeg_adjacency(n_channels, normalize=False))
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.out_dim = out_features * num_heads if concat else out_features

    def forward(self, eeg_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg_tensor: Node features, shape (batch, n_channels, in_features)

        Returns:
            Aggregated node features, shape (batch, n_channels, out_dim).
        """
        batch, n_channels, _ = eeg_tensor.shape
        node_features = self._project_nodes(eeg_tensor, batch, n_channels)
        attention_weights = self._compute_attention(node_features, n_channels)
        aggregated = torch.einsum("bijh,bjhf->bihf", attention_weights, node_features)
        return self._combine_heads(aggregated, batch, n_channels)

    def _project_nodes(self, eeg_tensor: torch.Tensor, batch: int, n_channels: int) -> torch.Tensor:
        """Linear projection split across heads."""
        return self.weight(eeg_tensor).view(batch, n_channels, self.num_heads, self.out_features)

    def _compute_attention(self, node_features: torch.Tensor, n_channels: int) -> torch.Tensor:
        """Compute masked and softmax-normalised attention weights."""
        features_i = node_features.unsqueeze(2).expand(-1, -1, n_channels, -1, -1)
        features_j = node_features.unsqueeze(1).expand(-1, n_channels, -1, -1, -1)
        concat_features = torch.cat([features_i, features_j], dim=-1)
        scores = self.leaky_relu(torch.einsum("bijnf,hf->bijh", concat_features, self.attention_vec))
        mask = self.adj_mask[:n_channels, :n_channels]
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(-1) == 0, float("-inf"))
        return self.dropout(F.softmax(scores, dim=2))

    def _combine_heads(self, aggregated: torch.Tensor, batch: int, n_channels: int) -> torch.Tensor:
        """Concat or average heads to produce final output."""
        if self.concat:
            return aggregated.reshape(batch, n_channels, -1)
        return aggregated.mean(dim=2)


class EEGGraphNetwork(nn.Module):
    """
    Full GNN for EEG: per-channel temporal CNN → multi-layer GAT → global readout.

    Args:
        n_channels: Number of EEG electrodes.
        time_steps: Temporal length of EEG window.
        hidden_dim: Final hidden dimension.
        num_layers: Number of graph attention layers.
        num_heads: Attention heads per layer.
        dropout: Dropout probability.

    Example:
        >>> net = EEGGraphNetwork(n_channels=16, hidden_dim=128)
        >>> out = net(torch.randn(4, 16, 256))
        >>> assert out.shape == (4, 128)
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
        self.temporal_conv = _build_temporal_conv()
        self.graph_layers = _build_graph_layers(n_channels, hidden_dim, num_layers, num_heads, dropout)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * n_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, eeg_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg_tensor: shape (batch, n_channels, time_steps)

        Returns:
            Graph-level embedding, shape (batch, hidden_dim).
        """
        batch, n_channels, _ = eeg_tensor.shape
        node_features = self._extract_channel_features(eeg_tensor, n_channels)
        node_features = self._propagate_graph(node_features)
        return self.readout(node_features.view(batch, -1))

    def _extract_channel_features(self, eeg_tensor: torch.Tensor, n_channels: int) -> torch.Tensor:
        """Apply temporal CNN per channel and stack into node features."""
        channel_feats = [
            self.temporal_conv(eeg_tensor[:, c:c + 1, :]).squeeze(-1)
            for c in range(n_channels)
        ]
        return torch.stack(channel_feats, dim=1)

    def _propagate_graph(self, node_features: torch.Tensor) -> torch.Tensor:
        """Run through all graph attention layers."""
        for layer in self.graph_layers:
            node_features = F.relu(layer(node_features))
        return node_features


def _build_temporal_conv() -> nn.Sequential:
    """Build per-channel temporal feature extractor (1D CNN)."""
    return nn.Sequential(
        nn.Conv1d(1, 32, kernel_size=7, padding=3),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Conv1d(32, 64, kernel_size=5, padding=2),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(1),
    )


def _build_graph_layers(
    n_channels: int, hidden_dim: int, num_layers: int, num_heads: int, dropout: float
) -> nn.ModuleList:
    """Build stack of GraphAttention layers."""
    layers = nn.ModuleList()
    in_features = 64
    for i in range(num_layers):
        out_features = hidden_dim if i == num_layers - 1 else in_features
        layers.append(GraphAttention(
            in_features=in_features,
            out_features=out_features // num_heads,
            n_channels=n_channels,
            num_heads=num_heads,
            dropout=dropout,
        ))
        in_features = out_features
    return layers
