"""
models — LSTM seizure classifier architectures.

Planned classifiers:
    1. VanillaLSTM        — single-direction baseline
    2. BiLSTM             — bidirectional LSTM
    3. AttentionBiLSTM    — BiLSTM with self-attention
    4. CNNLSTM            — CNN feature extractor + LSTM
    5. FeatureBiLSTM      — hand-crafted features + BiLSTM

Add model files here and register them in MODEL_REGISTRY.
"""

MODEL_REGISTRY = {}