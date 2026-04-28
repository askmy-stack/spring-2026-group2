"""IM6: Improved Graph BiLSTM — subclass of M6_GraphBiLSTM."""
from src.component.models.lstm_benchmark_models.architectures.m6_graph_bilstm import (
    M6_GraphBiLSTM,
)
from src.component.models.improved_lstm_models.modules.regularization import wrap_with_droppath


class IM6_GraphBiLSTM(M6_GraphBiLSTM):
    """Improved M6 with wider defaults + 8 heads.

    Narrower backbone than im1/im2 (160 vs 192) because the graph layers
    and channel-wise LSTM already multiply effective width by ``n_channels``.
    """

    def __init__(
        self,
        n_channels: int = 16,
        time_steps: int = 256,
        hidden_size: int = 160,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.4,
        stochastic_depth: float = 0.1,
    ):
        super().__init__(
            n_channels=n_channels,
            time_steps=time_steps,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.stochastic_depth = stochastic_depth
        wrap_with_droppath(self, attr_name="dropout_layer", p=stochastic_depth)
