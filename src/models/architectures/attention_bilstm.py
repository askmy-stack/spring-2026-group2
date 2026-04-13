"""
Attention-Based Bidirectional LSTM Classifier
================================================
BiLSTM + learned attention weights over time steps.

Improvements (v2):
- Learnable positional encoding: tells attention which timestep we're at
  (seizure spikes position-dependent: onset in early steps, spread in late steps)
- 4-head MultiheadAttention (replaced single Bahdanau)
- Pre-attention residual: attended + lstm_out
- Global avg+max pooling over attended sequence
- Input LayerNorm + projection
- 2-layer FC head with ReLU + dropout
"""

import torch
import torch.nn as nn


class AttentionBiLSTM(nn.Module):
    def __init__(
        self,
        n_channels: int = 16,
        seq_len: int = 256,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 1,
        num_heads: int = 4,
    ):
        super().__init__()
        self.n_channels  = n_channels
        self.seq_len     = seq_len
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.dropout_p   = dropout
        self.num_classes = num_classes

        # Input normalisation + projection
        self.input_norm = nn.LayerNorm(n_channels)
        self.input_proj = nn.Sequential(
            nn.Linear(n_channels, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_size = hidden_size * 2  # bidirectional

        # Learnable positional encoding: (1, seq_len, lstm_out_size)
        # Tells attention the position of each timestep in the window
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, seq_len, lstm_out_size)
        )
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # Multi-head self-attention over BiLSTM outputs
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_out_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(lstm_out_size)

        # avg-pool + max-pool -> lstm_out_size * 2
        self.pool_norm = nn.LayerNorm(lstm_out_size * 2)
        self.dropout   = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(lstm_out_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def _run(self, x):
        """Shared forward logic — returns logits and attention weight map."""
        x = x.permute(0, 2, 1)  # (batch, time_steps, channels)
        x = self.input_norm(x)
        x = self.input_proj(x)   # (batch, time_steps, hidden_size)

        lstm_out, _ = self.lstm(x)  # (batch, time_steps, hidden_size * 2)

        # Add learnable positional encoding
        # Trim pos_embedding to actual sequence length in case of batch variance
        pos_emb = self.pos_embedding[:, :lstm_out.size(1), :]
        lstm_out = lstm_out + pos_emb

        # Multi-head self-attention
        attended, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)

        # Residual connection: keep both attended and raw LSTM context
        attended = self.attn_norm(attended + lstm_out)

        # Global avg + max pooling
        avg_pool = attended.mean(dim=1)
        max_pool = attended.max(dim=1).values
        out = torch.cat([avg_pool, max_pool], dim=1)  # (batch, lstm_out_size * 2)

        out = self.pool_norm(out)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits, attn_weights

    def forward(self, x):
        """
        Args:
            x: (batch, channels, time_steps) raw EEG tensor
        Returns:
            logits: (batch, 1)
        """
        logits, _ = self._run(x)
        return logits

    def forward_with_attention(self, x):
        """Same as forward but also returns attention weights for visualisation.

        Returns:
            logits:       (batch, 1)
            attn_weights: (batch, num_heads, time_steps, time_steps)
        """
        return self._run(x)
