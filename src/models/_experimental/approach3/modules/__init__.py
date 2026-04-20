"""
Approach 3 Cutting-Edge Modules
-------------------------------
Advanced components for state-of-the-art EEG seizure detection.
"""

from .mamba_block import MambaBlock, BidirectionalMamba, MambaEncoder
from .mixture_of_experts import MixtureOfExperts, Expert, Router
from .diffusion_eeg import EEGDiffusion, DiffusionScheduler, UNet1D
from .uncertainty import (
    MCDropout,
    mc_dropout_inference,
    EvidentialClassifier,
    compute_uncertainty,
)

__all__ = [
    # Mamba
    "MambaBlock",
    "BidirectionalMamba",
    "MambaEncoder",
    # MoE
    "MixtureOfExperts",
    "Expert",
    "Router",
    # Diffusion
    "EEGDiffusion",
    "DiffusionScheduler",
    "UNet1D",
    # Uncertainty
    "MCDropout",
    "mc_dropout_inference",
    "EvidentialClassifier",
    "compute_uncertainty",
]
