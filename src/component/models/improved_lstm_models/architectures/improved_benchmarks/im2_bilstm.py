"""IM2: Improved BiLSTM — subclass of M2_BiLSTM."""
from src.models.lstm_benchmark_models.architectures.m2_bilstm import M2_BiLSTM
from src.models.improved_lstm_models.modules.regularization import wrap_with_droppath


class IM2_BiLSTM(M2_BiLSTM):
    """Improved M2 with wider defaults."""

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
        wrap_with_droppath(self, attr_name="dropout_layer", p=stochastic_depth)
