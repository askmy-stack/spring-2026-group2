"""
Approach 2 Advanced Modules
---------------------------
Reusable attention mechanisms and encoders for EEG seizure detection.
"""

from .channel_attention import ChannelAttention, SEBlock
from .criss_cross_attention import CrissCrossAttention, SpatialAttention, TemporalAttention
from .graph_attention import GraphAttention, EEGGraphConv, build_eeg_adjacency
from .vector_quantizer import VectorQuantizer, VQEncoder
from .pretrained_encoders import (
    load_pretrained_encoder,
    CBraModEncoder,
    EEGPTEncoder,
    BIOTEncoder,
    LaBraMEncoder,
)

__all__ = [
    # Channel Attention
    "ChannelAttention",
    "SEBlock",
    # Criss-Cross Attention
    "CrissCrossAttention",
    "SpatialAttention", 
    "TemporalAttention",
    # Graph Attention
    "GraphAttention",
    "EEGGraphConv",
    "build_eeg_adjacency",
    # Vector Quantizer
    "VectorQuantizer",
    "VQEncoder",
    # Pre-trained Encoders
    "load_pretrained_encoder",
    "CBraModEncoder",
    "EEGPTEncoder",
    "BIOTEncoder",
    "LaBraMEncoder",
]
