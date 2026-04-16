"""
Mamba State-Space Model Blocks for EEG.

O(n) linear-complexity sequence modeling for long EEG windows.

References:
    Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023)
    EEGMamba: Bidirectional State Space Model for EEG Decoding
"""
import logging
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (S6) — core Mamba operation.

    Input-dependent selection determines which information to retain.
    Processes sequences in O(n) time unlike O(n^2) for transformers.

    Args:
        d_model: Input feature dimension.
        d_state: SSM state dimension (default: 16).
        d_conv: Depthwise conv kernel size (default: 4).
        expand: Inner expansion factor (default: 2).
        dropout: Dropout probability (default: 0.1).

    Example:
        >>> ssm = SelectiveSSM(d_model=128)
        >>> out = ssm(torch.randn(2, 64, 128))
        >>> assert out.shape == (2, 64, 128)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                padding=d_conv - 1, groups=self.d_inner)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_tensor: shape (batch, seq_len, d_model)

        Returns:
            Output tensor, same shape as input.
        """
        batch, seq_len, _ = input_tensor.shape
        convolved, gated = self._project_and_conv(input_tensor, seq_len)
        ssm_out = self._apply_ssm(convolved, batch, seq_len)
        return self._project_output(ssm_out, gated, convolved)

    def _project_and_conv(
        self, input_tensor: torch.Tensor, seq_len: int
    ) -> tuple:
        """Project input and apply depthwise conv for local context."""
        xz = self.in_proj(input_tensor)
        convolved, gated = xz.chunk(2, dim=-1)
        convolved = F.silu(self.conv1d(convolved.transpose(1, 2))[:, :, :seq_len].transpose(1, 2))
        return convolved, gated

    def _apply_ssm(self, convolved: torch.Tensor, batch: int, seq_len: int) -> torch.Tensor:
        """Run selective scan recurrence over time steps."""
        x_dbl = self.x_proj(convolved)
        delta, B, C = torch.split(x_dbl, [1, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(delta)
        A = -torch.exp(self.A_log)
        deltaA = torch.exp(delta * A)
        deltaB = delta * B
        hidden_state = torch.zeros(batch, self.d_state, device=convolved.device)
        outputs: List[torch.Tensor] = []
        for t in range(seq_len):
            hidden_state = deltaA[:, t] * hidden_state + deltaB[:, t] * convolved[:, t, :self.d_state]
            outputs.append((hidden_state * C[:, t]).sum(dim=-1, keepdim=True))
        return torch.stack(outputs, dim=1).expand(-1, -1, self.d_inner)

    def _project_output(
        self, ssm_out: torch.Tensor, gated: torch.Tensor, convolved: torch.Tensor
    ) -> torch.Tensor:
        """Apply skip connection, gate, and output projection."""
        skip_out = ssm_out + convolved * self.D
        gated_out = skip_out * F.silu(gated)
        return self.dropout(self.out_proj(gated_out))


class MambaBlock(nn.Module):
    """
    Mamba residual block: LayerNorm -> SelectiveSSM -> residual.

    Args:
        d_model: Feature dimension.
        d_state: SSM state size (default: 16).
        d_conv: Conv kernel size (default: 4).
        expand: Inner expansion factor (default: 2).
        dropout: Dropout probability (default: 0.1).

    Example:
        >>> block = MambaBlock(d_model=128)
        >>> out = block(torch.randn(2, 64, 128))
        >>> assert out.shape == (2, 64, 128)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand, dropout)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_tensor: shape (batch, seq_len, d_model)

        Returns:
            Residual-connected SSM output, same shape.
        """
        return input_tensor + self.ssm(self.norm(input_tensor))


class BidirectionalMamba(nn.Module):
    """
    Bidirectional Mamba: forward + reversed backward SSM fused by linear.

    Args:
        d_model: Feature dimension.
        d_state: SSM state size (default: 16).
        d_conv: Conv kernel size (default: 4).
        expand: Expansion factor (default: 2).
        dropout: Dropout probability (default: 0.1).

    Example:
        >>> bi = BidirectionalMamba(d_model=128)
        >>> out = bi(torch.randn(2, 64, 128))
        >>> assert out.shape == (2, 64, 128)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.forward_mamba = MambaBlock(d_model, d_state, d_conv, expand, dropout)
        self.backward_mamba = MambaBlock(d_model, d_state, d_conv, expand, dropout)
        self.fusion = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_tensor: shape (batch, seq_len, d_model)

        Returns:
            Fused bidirectional SSM output, same shape.
        """
        fwd = self.forward_mamba(input_tensor)
        bwd = self.backward_mamba(input_tensor.flip(dims=[1])).flip(dims=[1])
        return self.norm(self.fusion(torch.cat([fwd, bwd], dim=-1)))


class MambaEncoder(nn.Module):
    """
    Full Mamba encoder: embedding -> N x (Bidirectional)MambaBlock -> LayerNorm.

    Args:
        n_channels: Number of EEG channels (default: 16).
        time_steps: Timesteps per window (default: 256).
        d_model: Model dimension (default: 128).
        d_state: SSM state size (default: 16).
        d_conv: Conv kernel size (default: 4).
        n_layers: Number of Mamba blocks (default: 4).
        bidirectional: Use BidirectionalMamba if True (default: True).
        dropout: Dropout probability (default: 0.1).

    Example:
        >>> enc = MambaEncoder(n_channels=16, d_model=128, n_layers=4)
        >>> out = enc(torch.randn(2, 16, 256))
        >>> assert out.shape == (2, 256, 128)
    """

    def __init__(
        self,
        n_channels: int = 16,
        time_steps: int = 256,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        n_layers: int = 4,
        bidirectional: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Sequential(nn.Linear(n_channels, d_model), nn.LayerNorm(d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, time_steps, d_model) * 0.02)
        block_class = BidirectionalMamba if bidirectional else MambaBlock
        self.layers = nn.ModuleList([
            block_class(d_model, d_state, d_conv, dropout=dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, eeg_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg_input: shape (batch, n_channels, time_steps)

        Returns:
            Temporal features, shape (batch, time_steps, d_model).
        """
        hidden = self._embed(eeg_input)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.norm(hidden)

    def _embed(self, eeg_input: torch.Tensor) -> torch.Tensor:
        """Project EEG channels to d_model and add positional encoding."""
        transposed = eeg_input.permute(0, 2, 1)
        embedded = self.input_proj(transposed)
        seq_len = embedded.size(1)
        return embedded + self.pos_embed[:, :seq_len, :]

    def get_features(self, eeg_input: torch.Tensor) -> torch.Tensor:
        """Return mean-pooled features for classification."""
        return self.forward(eeg_input).mean(dim=1)
