"""M5: FeatureBiLSTM + Temporal Attention — Benchmark Model. Expected F1: 0.68-0.75."""
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class M5_FeatureBiLSTM(nn.Module):
    """
    FeatureBiLSTM with temporal attention over sequential windows.

    Accepts either pre-extracted features (batch, seq_len, n_features)
    or raw EEG (batch, n_channels, time_steps).

    Args:
        n_features: Feature dimension per window (default: 226).
        seq_len: Number of sequential windows (default: 10).
        hidden_size: LSTM hidden dimension (default: 128).
        num_layers: BiLSTM layers (default: 2).
        num_heads: Temporal attention heads (default: 4).
        dropout: Dropout rate (default: 0.3).
        n_channels: EEG channels for raw input compatibility (default: 16).
        time_steps: EEG timesteps for raw input compatibility (default: 256).

    Example:
        >>> model = M5_FeatureBiLSTM(n_features=226, seq_len=10)
        >>> logits = model(torch.randn(4, 10, 226))
        >>> assert logits.shape == (4, 1)
    """

    def __init__(
        self,
        n_features: int = 226,
        seq_len: int = 10,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
        n_channels: int = 16,
        time_steps: int = 256,
    ):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.n_channels = n_channels
        self.time_steps = time_steps
        bilstm_out = hidden_size * 2
        self.raw_feature_extractor = _build_raw_extractor(n_channels, seq_len)
        self.input_bn = nn.BatchNorm1d(n_features)
        self.raw_input_bn = nn.BatchNorm1d(128)
        self.proj = _build_projection(n_features, hidden_size, dropout)
        self.proj_shortcut = nn.Linear(n_features, hidden_size)
        self.raw_proj = _build_projection(128, hidden_size, dropout)
        self.raw_proj_shortcut = nn.Linear(128, hidden_size)
        self.pre_lstm_norm = nn.LayerNorm(hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.temporal_attn = nn.MultiheadAttention(bilstm_out, num_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(bilstm_out)
        self.pool_norm = nn.LayerNorm(bilstm_out * 2)
        self.dropout_layer = nn.Dropout(dropout)
        self.classifier = _build_classifier(hidden_size, bilstm_out * 2, dropout)

    def forward(self, eeg_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg_input: Either raw EEG ``(batch, n_channels, time_steps)`` or
                pre-extracted features ``(batch, seq_len, n_features)``.

        Returns:
            Logits, shape (batch, 1).

        Raises:
            ValueError: If the last two dims match neither ``(n_channels, time_steps)``
                nor ``(seq_len, n_features)`` exactly.
        """
        if eeg_input.ndim != 3:
            raise ValueError(
                f"M5_FeatureBiLSTM expects a 3-D tensor; got shape {tuple(eeg_input.shape)}."
            )
        _, dim1, dim2 = eeg_input.shape
        is_raw = (dim1 == self.n_channels and dim2 == self.time_steps)
        is_feat = (dim1 == self.seq_len and dim2 == self.n_features)
        if is_raw and is_feat:
            # Defensive: if both interpretations are numerically valid (e.g. the
            # user happened to configure n_channels=seq_len AND time_steps=n_features),
            # prefer raw since that's the canonical CHB-MIT pipeline.
            logger.warning(
                "Input shape %s matches both raw and feature modes; defaulting to raw.",
                tuple(eeg_input.shape),
            )
            return self._forward_raw(eeg_input)
        if is_raw:
            return self._forward_raw(eeg_input)
        if is_feat:
            return self._forward_features(eeg_input)
        raise ValueError(
            f"M5_FeatureBiLSTM got shape {tuple(eeg_input.shape)} which matches "
            f"neither raw (_, {self.n_channels}, {self.time_steps}) nor "
            f"feature (_, {self.seq_len}, {self.n_features}) mode."
        )

    def _forward_features(self, eeg_input: torch.Tensor) -> torch.Tensor:
        """Pipeline for pre-extracted features (batch, seq_len, n_features)."""
        normalised = self._batch_norm_features(eeg_input, self.input_bn)
        projected = self.proj(normalised) + self.proj_shortcut(normalised)
        return self._lstm_classify(self.pre_lstm_norm(projected))

    def _forward_raw(self, eeg_input: torch.Tensor) -> torch.Tensor:
        """Pipeline for raw EEG (batch, n_channels, time_steps)."""
        features = self.raw_feature_extractor(eeg_input).permute(0, 2, 1)
        normalised = self._batch_norm_features(features, self.raw_input_bn)
        projected = self.raw_proj(normalised) + self.raw_proj_shortcut(normalised)
        return self._lstm_classify(self.pre_lstm_norm(projected))

    def _batch_norm_features(self, tensor: torch.Tensor, bn: nn.BatchNorm1d) -> torch.Tensor:
        """Apply BatchNorm1d across feature dimension (transpose in/out)."""
        return bn(tensor.permute(0, 2, 1)).permute(0, 2, 1)

    def _lstm_classify(self, projected: torch.Tensor) -> torch.Tensor:
        """Run BiLSTM -> temporal attention -> pool -> classify."""
        lstm_out, _ = self.lstm(projected)
        attended, _ = self.temporal_attn(lstm_out, lstm_out, lstm_out)
        attended = self.attn_norm(attended + lstm_out)
        avg_pooled = attended.mean(dim=1)
        max_pooled = attended.max(dim=1).values
        pooled = self.dropout_layer(self.pool_norm(torch.cat([avg_pooled, max_pooled], dim=-1)))
        return self.classifier(pooled)


def _build_raw_extractor(n_channels: int, seq_len: int) -> nn.Sequential:
    """Build CNN feature extractor for raw EEG input."""
    return nn.Sequential(
        nn.Conv1d(n_channels, 64, kernel_size=7, padding=3),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(seq_len),
    )


def _build_projection(in_dim: int, hidden_size: int, dropout: float) -> nn.Sequential:
    """Build MLP projection block."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_size * 2),
        nn.ReLU(),
        nn.Dropout(dropout * 0.5),
        nn.Linear(hidden_size * 2, hidden_size),
    )


def _build_classifier(hidden_size: int, input_size: int, dropout: float) -> nn.Sequential:
    """Build two-layer FC classification head."""
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, 1),
    )
