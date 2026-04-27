"""Shared attention modules for LSTM benchmark architectures."""
from .channel_attention import ChannelAttention, SpatialChannelAttention
from .criss_cross_attention import CrissCrossAttention, CrissCrossBlock
from .graph_attention import GraphAttention, EEGGraphNetwork

__all__ = [
    "ChannelAttention", "SpatialChannelAttention",
    "CrissCrossAttention", "CrissCrossBlock",
    "GraphAttention", "EEGGraphNetwork",
]
