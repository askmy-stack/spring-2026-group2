"""
Mixture of Experts (MoE) for multi-task EEG processing.

Each expert specialises in different seizure pattern types.

References:
    EEGMamba: MoE for multi-task EEG
    Switch Transformers: Scaling to Trillion Parameter Models (Fedus et al., 2022)
"""
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class Expert(nn.Module):
    """
    Single expert feed-forward network.

    Args:
        input_dim: Input feature dimension.
        hidden_dim: Hidden layer width.
        output_dim: Output feature dimension.
        dropout: Dropout probability (default: 0.1).

    Example:
        >>> exp = Expert(input_dim=128, hidden_dim=256, output_dim=128)
        >>> out = exp(torch.randn(4, 128))
        >>> assert out.shape == (4, 128)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_tensor: shape (..., input_dim)

        Returns:
            Expert output, shape (..., output_dim).
        """
        return self.net(input_tensor)


class Router(nn.Module):
    """
    Sparse top-k router: assigns tokens to the k most relevant experts.

    Args:
        input_dim: Input feature dimension.
        num_experts: Total number of experts.
        top_k: Number of experts to activate per token (default: 2).
        noise_std: Gating noise for training exploration (default: 0.1).

    Example:
        >>> router = Router(input_dim=128, num_experts=8, top_k=2)
        >>> gates, indices, lb_loss = router(torch.randn(4, 16, 128))
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        top_k: int = 2,
        noise_std: float = 0.1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.gate = nn.Linear(input_dim, num_experts, bias=False)

    def forward(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            input_tensor: shape (batch, seq_len, input_dim) or (batch, input_dim)

        Returns:
            gates: Full gate weights (batch, [seq_len,] num_experts)
            indices: Top-k expert indices, same leading dims
            load_balance_loss: Scalar KL-divergence load balancing loss
        """
        logits = self.gate(input_tensor)
        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std
        gates, indices = self._top_k_gate(logits)
        load_balance_loss = self._load_balance_loss(gates, input_tensor.dim())
        return gates, indices, load_balance_loss

    def _top_k_gate(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select top-k experts and build sparse gate tensor."""
        top_k_logits, indices = torch.topk(logits, self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)
        gates = torch.zeros_like(logits).scatter_(-1, indices, top_k_gates)
        return gates, indices

    def _load_balance_loss(self, gates: torch.Tensor, n_dims: int) -> torch.Tensor:
        """KL divergence from expert usage to uniform prior."""
        reduce_dims = list(range(n_dims - 1))
        expert_usage = gates.sum(dim=reduce_dims) / (
            gates.shape[0] * (gates.shape[1] if n_dims == 3 else 1)
        )
        uniform = torch.ones_like(expert_usage) / self.num_experts
        return F.kl_div(expert_usage.log().clamp(min=-1e9), uniform, reduction="sum")


class MixtureOfExperts(nn.Module):
    """
    MoE layer: each token is routed to top-k experts and results are aggregated.

    Args:
        input_dim: Token feature dimension.
        num_experts: Number of expert networks (default: 8).
        top_k: Experts activated per token (default: 2).
        hidden_dim: Expert hidden dimension (default: input_dim * 4).
        dropout: Dropout probability (default: 0.1).

    Example:
        >>> moe = MixtureOfExperts(input_dim=128, num_experts=8)
        >>> out, lb_loss = moe(torch.randn(2, 32, 128))
        >>> assert out.shape == (2, 32, 128)
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        hidden_dim: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        hidden_dim = hidden_dim or input_dim * 4
        self.router = Router(input_dim, num_experts, top_k)
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, input_dim, dropout) for _ in range(num_experts)
        ])

    def forward(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_tensor: shape (batch, seq_len, input_dim)

        Returns:
            (output, load_balance_loss): output same shape, lb_loss scalar.
        """
        gates, _, load_balance_loss = self.router(input_tensor)
        expert_outputs = torch.stack([exp(input_tensor) for exp in self.experts], dim=-1)
        weighted = (expert_outputs * gates.unsqueeze(-2)).sum(dim=-1)
        return weighted, load_balance_loss


class MoEWithUniversal(nn.Module):
    """
    MoE layer with one universal shared expert always activated.

    The shared expert captures common EEG patterns while specialised
    experts handle specific seizure types.

    Args:
        input_dim: Token feature dimension.
        num_experts: Number of specialised experts (default: 8).
        top_k: Routed experts per token (default: 2).
        dropout: Dropout probability (default: 0.1).

    Example:
        >>> moe = MoEWithUniversal(input_dim=128)
        >>> out, lb_loss = moe(torch.randn(2, 32, 128))
        >>> assert out.shape == (2, 32, 128)
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.universal_expert = Expert(input_dim, input_dim * 4, input_dim, dropout)
        self.moe = MixtureOfExperts(input_dim, num_experts, top_k, dropout=dropout)

    def forward(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_tensor: shape (batch, seq_len, input_dim)

        Returns:
            (output, load_balance_loss): output same shape.
        """
        universal_out = self.universal_expert(input_tensor)
        routed_out, lb_loss = self.moe(input_tensor)
        return universal_out + routed_out, lb_loss
