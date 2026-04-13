"""
Mixture of Experts (MoE) Module
-------------------------------
Sparse gating for multi-task EEG processing.
Each expert specializes in different seizure patterns.

References:
- EEGMamba: MoE for multi-task EEG
- Switch Transformers: Scaling to Trillion Parameter Models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class Expert(nn.Module):
    """
    Single expert network.
    Each expert can specialize in different seizure patterns.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Router(nn.Module):
    """
    Router that assigns inputs to experts.
    Uses top-k gating for sparse activation.
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

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch, seq_len, input_dim) or (batch, input_dim)

        Returns:
            gates: Gating weights (batch, [seq_len,] num_experts)
            indices: Top-k expert indices
            load_balance_loss: Auxiliary loss for load balancing
        """
        # Compute logits
        logits = self.gate(x)  # (batch, [seq_len,] num_experts)

        # Add noise during training for exploration
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        # Top-k gating
        top_k_logits, indices = torch.topk(logits, self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)

        # Create full gate tensor
        gates = torch.zeros_like(logits)
        gates.scatter_(-1, indices, top_k_gates)

        # Load balance loss (encourage uniform expert usage)
        # Fraction of tokens routed to each expert
        if x.dim() == 3:
            expert_usage = gates.sum(dim=[0, 1]) / (gates.shape[0] * gates.shape[1])
        else:
            expert_usage = gates.sum(dim=0) / gates.shape[0]

        # Ideal uniform distribution
        uniform = torch.ones_like(expert_usage) / self.num_experts

        # KL divergence as load balance loss
        load_balance_loss = F.kl_div(
            expert_usage.log(), uniform, reduction="sum"
        )

        return gates, indices, load_balance_loss


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts layer.

    Each token is routed to top-k experts based on learned gating.
    Enables specialization for different EEG patterns.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Create experts
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim, dropout)
            for _ in range(num_experts)
        ])

        # Router
        self.router = Router(input_dim, num_experts, top_k)

        # Output normalization
        self.norm = nn.LayerNorm(output_dim)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch, seq_len, input_dim) or (batch, input_dim)

        Returns:
            output: MoE output
            aux_loss: Auxiliary load balancing loss
        """
        original_shape = x.shape
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add seq_len dim

        batch, seq_len, input_dim = x.shape

        # Get routing weights
        gates, indices, aux_loss = self.router(x)

        # Flatten for expert processing
        x_flat = x.view(-1, input_dim)  # (batch * seq_len, input_dim)
        gates_flat = gates.view(-1, self.num_experts)  # (batch * seq_len, num_experts)

        # Process through experts
        expert_outputs = torch.stack([
            expert(x_flat) for expert in self.experts
        ], dim=1)  # (batch * seq_len, num_experts, output_dim)

        # Weighted combination
        output = torch.einsum("bne,bn->be", expert_outputs, gates_flat)

        # Reshape
        output = output.view(batch, seq_len, -1)

        if len(original_shape) == 2:
            output = output.squeeze(1)

        output = self.norm(output)

        return output, aux_loss


class MoEBlock(nn.Module):
    """
    Complete MoE block with attention and feed-forward.
    Replaces standard transformer feed-forward with MoE.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(d_model)

        # MoE feed-forward
        self.moe = MixtureOfExperts(
            input_dim=d_model,
            hidden_dim=d_model * 4,
            output_dim=d_model,
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout,
        )
        self.moe_norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention
        attn_out, _ = self.attn(x, x, x)
        x = self.attn_norm(x + attn_out)

        # MoE feed-forward
        moe_out, aux_loss = self.moe(x)
        x = self.moe_norm(x + moe_out)

        return x, aux_loss


class UniversalExpert(nn.Module):
    """
    Universal expert that processes all inputs.
    Combined with sparse experts for robust predictions.
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
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoEWithUniversal(nn.Module):
    """
    MoE with additional universal expert (always active).
    Ensures baseline capability while allowing specialization.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Sparse MoE
        self.moe = MixtureOfExperts(
            input_dim, hidden_dim, output_dim, num_experts, top_k, dropout
        )

        # Universal expert
        self.universal = UniversalExpert(input_dim, hidden_dim, output_dim, dropout)

        # Combination weight
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # MoE output
        moe_out, aux_loss = self.moe(x)

        # Universal output
        universal_out = self.universal(x)

        # Combine
        alpha = torch.sigmoid(self.alpha)
        output = alpha * moe_out + (1 - alpha) * universal_out

        return output, aux_loss


if __name__ == "__main__":
    # Test MoE
    moe = MixtureOfExperts(
        input_dim=128, hidden_dim=256, output_dim=128, num_experts=8, top_k=2
    )
    x = torch.randn(8, 100, 128)  # (batch, seq_len, dim)
    out, loss = moe(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Aux loss: {loss.item():.4f}")
    print(f"Parameters: {sum(p.numel() for p in moe.parameters()):,}")
