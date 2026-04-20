"""IM4: Improved CNN-LSTM — subclass of M4_CNNLSTM."""
from src.models.lstm_benchmark_models.architectures.m4_cnn_lstm import M4_CNNLSTM


class IM4_CNNLSTM(M4_CNNLSTM):
    """Improved M4 with wider defaults."""

    def __init__(
        self,
        n_channels: int = 16,
        time_steps: int = 256,
        hidden_size: int = 192,
        num_layers: int = 2,
        dropout: float = 0.4,
        stochastic_depth: float = 0.1,
    ):
        super().__init__(
            n_channels=n_channels,
            time_steps=time_steps,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.stochastic_depth = stochastic_depth
