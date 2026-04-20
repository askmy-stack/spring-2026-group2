"""
Approach 3 Model Architectures
------------------------------
Cutting-edge models for EEG seizure detection.

- EEGMamba: Mamba + Mixture of Experts
- HierarchicalLSTM: Long-context pre-ictal prediction
- TinySeizureNet: Distilled lightweight model
"""

from .eeg_mamba import EEGMamba, EEGMambaMoE
from .hierarchical_lstm import HierarchicalLSTM, PreIctalPredictor
from .tiny_seizure_net import TinySeizureNet, MicroSeizureNet

MODEL_REGISTRY = {
    "eeg_mamba": EEGMamba,
    "eeg_mamba_moe": EEGMambaMoE,
    "hierarchical_lstm": HierarchicalLSTM,
    "preictal_predictor": PreIctalPredictor,
    "tiny_seizure_net": TinySeizureNet,
    "micro_seizure_net": MicroSeizureNet,
}


def get_model(model_name: str, **kwargs):
    """Get model by name."""
    model_name = model_name.lower()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name](**kwargs)


def list_models():
    """List all available models."""
    return list(MODEL_REGISTRY.keys())


__all__ = [
    "EEGMamba",
    "EEGMambaMoE",
    "HierarchicalLSTM",
    "PreIctalPredictor",
    "TinySeizureNet",
    "MicroSeizureNet",
    "MODEL_REGISTRY",
    "get_model",
    "list_models",
]
