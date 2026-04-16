"""
src/models — Restructured EEG model package (2026-04-15).

Four sequenced directories:
  1. lstm_benchmark_models/   m1–m6 LSTM baselines
  2. improved_lstm_models/    HierarchicalLSTM + augmentation
  3. ensemble_transformers/   m7 VQ-Transformer + 7-model ensemble
  4. hugging_face_mamba_moe/  Mamba SSM + MoE + HF pretrained CNNs
"""
from .lstm_benchmark_models import get_benchmark_model  # noqa: F401
from .lstm_benchmark_models import MODEL_REGISTRY as BENCHMARK_MODEL_REGISTRY  # noqa: F401
from .utils import metrics, losses, callbacks  # noqa: F401
