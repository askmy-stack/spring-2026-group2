"""Mamba SSM, MoE, and HF CNN building-block modules."""
from .mamba_block import MambaBlock, BidirectionalMamba, MambaEncoder, SelectiveSSM
from .hf_blocks import (
    ConvBNAct1d, DepthwiseSeparableConv1d, ResidualBlock1d,
    DilatedResidualBlock1d, SEBlock1d, MultiScaleBranch1d,
    ensure_3d, make_mlp,
)

__all__ = [
    "MambaBlock", "BidirectionalMamba", "MambaEncoder", "SelectiveSSM",
    "ConvBNAct1d", "DepthwiseSeparableConv1d", "ResidualBlock1d",
    "DilatedResidualBlock1d", "SEBlock1d", "MultiScaleBranch1d",
    "ensure_3d", "make_mlp",
]
