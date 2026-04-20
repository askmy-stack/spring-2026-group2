# Approach 2: Advanced Ensemble with Pre-trained Models

This directory contains the **advanced EEG seizure detection pipeline** with:
- 7 diverse model architectures
- Pre-trained transformer integration (CBraMod, EEGPT, BIOT, LaBraM)
- Advanced attention mechanisms (Criss-Cross, Graph, Channel)
- Weighted ensemble with stacking meta-learner

## Architecture Overview

| Model | Architecture | Pre-training | Expected F1 |
|-------|-------------|--------------|-------------|
| M1 | VanillaLSTM + ChannelAttn | None | 0.62-0.70 |
| M2 | BiLSTM + ChannelAttn | None | 0.65-0.72 |
| M3 | AttentionBiLSTM + CrissCross | EEGPT/CBraMod | 0.75-0.82 |
| M4 | CNN-LSTM (multi-scale) | EEGPT/CBraMod | 0.80-0.88 |
| M5 | FeatureBiLSTM + TemporalAttn | None | 0.68-0.75 |
| M6 | GraphBiLSTM (GMAEEG-style) | EEGPT/CBraMod | 0.78-0.85 |
| M7 | VQ-Transformer (EEGFormer) | EEGPT/CBraMod | 0.80-0.87 |

## Directory Structure

```
approach2/
├── README.md
├── train.py                    # Training with optional pre-training
├── pretrain.py                 # EEGPT-style self-supervised pre-training
├── ensemble_7model.py          # 7-model weighted ensemble + stacking
├── architectures/
│   ├── __init__.py
│   ├── m1_vanilla_lstm.py      # VanillaLSTM + ChannelAttn
│   ├── m2_bilstm.py            # BiLSTM + ChannelAttn
│   ├── m3_criss_cross.py       # AttentionBiLSTM + CrissCross
│   ├── m4_cnn_lstm.py          # CNN-LSTM (multi-scale)
│   ├── m5_feature_bilstm.py    # FeatureBiLSTM + TemporalAttn
│   ├── m6_graph_bilstm.py      # GraphBiLSTM
│   └── m7_vq_transformer.py    # VQ-Transformer
└── modules/
    ├── __init__.py
    ├── channel_attention.py    # SE-style channel attention
    ├── criss_cross_attention.py # Spatial ⊥ Temporal attention
    ├── graph_attention.py      # Graph neural network for electrodes
    ├── vector_quantizer.py     # VQ-VAE codebook
    └── pretrained_encoders.py  # Load CBraMod/EEGPT/BIOT/LaBraM
```

## Quick Start

```bash
# Train single model
python train.py --model m3_criss_cross --pretrained cbramod --epochs 50

# Train all 7 models
python train.py --model all --epochs 50

# Run 7-model ensemble
python ensemble_7model.py --checkpoints ./checkpoints --output ./results

# Pre-train on background EEG (optional)
python pretrain.py --data_path ../../data --epochs 100
```

## Requirements

```bash
pip install braindecode torch torch-geometric
```

## Pre-trained Model Integration

Models M3, M4, M6, M7 can use pre-trained encoders:
- **CBraMod**: Criss-cross attention, 5M params
- **EEGPT**: Dual self-supervised, 25M params  
- **BIOT**: Lightweight, 3.2M params
- **LaBraM**: VQ neural spectrum, 5.8M params

## Performance Targets

| Metric | Baseline (Approach 1) | Target (Approach 2) |
|--------|----------------------|---------------------|
| F1 | 0.52-0.72 | 0.80-0.90 |
| Sensitivity | 0.26-0.57 | 0.85-0.95 |
| Specificity | 0.73-0.87 | 0.85-0.92 |
| AUC-ROC | 0.56-0.71 | 0.90-0.95 |
