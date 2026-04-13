"""
CNN-LSTM Hybrid Classifier
============================
3 PARALLEL multi-scale CNN branches extract features at different timescales,
then BiLSTM + multi-head attention for seizure detection.

Improvements (v2):
- Parallel multi-scale CNN: kernel=3 (fast spikes), k=15 (spike-wave), k=31 (ictal rhythm)
- Each branch: ConvBlock with SE channel attention + residual
- Concatenate all branches, align temporal dims, feed to BiLSTM
- Multi-head self-attention (4 heads) over LSTM outputs with residual
- Global avg+max pooling
- LayerNorm + 2-layer FC head
"""

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention for 1D feature maps."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.ReLU(),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, C, T)
        scale = self.se(x).unsqueeze(-1)  # (batch, C, 1)
        return x * scale


class ConvBlock(nn.Module):
    """Conv1d block with BN, ReLU, strided downsampling, SE attention, and residual."""
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,
                      stride=stride, padding=kernel_size // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.se = SEBlock(out_ch)

        # 1x1 projection for residual when dims change
        self.residual_proj = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride)
            if (in_ch != out_ch or stride != 1)
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.residual_proj(x)
        out = self.conv(x)
        out = self.se(out)
        # Align lengths in case of rounding differences
        min_len = min(out.size(-1), residual.size(-1))
        return out[..., :min_len] + residual[..., :min_len]


class CNNLSTM(nn.Module):
    def __init__(
        self,
        n_channels: int = 16,
        seq_len: int = 256,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 1,
        cnn_filters: int = 64,
        num_heads: int = 4,
    ):
        super().__init__()
        self.n_channels  = n_channels
        self.seq_len     = seq_len
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.dropout_p   = dropout
        self.num_classes = num_classes

        # --- 3 PARALLEL CNN branches at different scales ---
        # Branch A: Fast (kernel=3) — captures spike transients
        self.branch_a = ConvBlock(n_channels, 32, kernel_size=3, stride=2, dropout=dropout*0.5)
        self.branch_a2 = ConvBlock(32, 32, kernel_size=3, stride=1, dropout=dropout*0.5)

        # Branch B: Medium (kernel=15) — captures spike-wave complexes (~250ms)
        self.branch_b = ConvBlock(n_channels, 64, kernel_size=15, stride=2, dropout=dropout*0.5)
        self.branch_b2 = ConvBlock(64, 64, kernel_size=3, stride=1, dropout=dropout*0.5)

        # Branch C: Slow (kernel=31) — captures ictal rhythm (~1s)
        self.branch_c = ConvBlock(n_channels, 64, kernel_size=31, stride=1, dropout=dropout*0.5)
        self.branch_c2 = ConvBlock(64, 64, kernel_size=3, stride=1, dropout=dropout*0.5)

        # Total features after concatenation: 32 + 64 + 64 = 160
        cnn_out_features = 32 + 64 + 64

        # --- BiLSTM Sequence Modeler ---
        self.lstm = nn.LSTM(
            input_size=cnn_out_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_size = hidden_size * 2

        # --- Multi-head self-attention over LSTM outputs ---
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_out_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(lstm_out_size)

        # avg-pool + max-pool
        self.pool_norm = nn.LayerNorm(lstm_out_size * 2)
        self.dropout   = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(lstm_out_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, channels, time_steps) raw EEG tensor
        Returns:
            logits: (batch, 1)
        """
        # --- Process 3 parallel branches ---
        # Branch A
        a = self.branch_a(x)   # (batch, 32, T/2)
        a = self.branch_a2(a)  # (batch, 32, T/2)

        # Branch B
        b = self.branch_b(x)   # (batch, 64, T/2)
        b = self.branch_b2(b)  # (batch, 64, T/2)

        # Branch C (no initial stride)
        c = self.branch_c(x)   # (batch, 64, T)
        c = self.branch_c2(c)  # (batch, 64, T)

        # Align temporal dimensions: reduce C to match A/B
        # A, B are at T/2 after stride=2; C at T; downsample C by 2
        c_pool = torch.nn.functional.max_pool1d(c, kernel_size=2, stride=2)  # (batch, 64, T/2)

        # Concatenate: (batch, 32+64+64, T/2)
        merged = torch.cat([a, b, c_pool], dim=1)

        # Reshape for LSTM: (batch, T/2, 160)
        merged = merged.permute(0, 2, 1)

        # BiLSTM
        lstm_out, _ = self.lstm(merged)  # (batch, T/2, hidden_size * 2)

        # Multi-head self-attention with residual
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attended = self.attn_norm(attended + lstm_out)

        # Global avg + max pooling
        avg_pool = attended.mean(dim=1)
        max_pool = attended.max(dim=1).values
        out = torch.cat([avg_pool, max_pool], dim=1)

        out = self.pool_norm(out)
        out = self.dropout(out)
        return self.fc(out)
