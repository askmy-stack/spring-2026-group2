# EEG Seizure Detection — Model Directory

Four model directories for CHB-MIT EEG seizure prediction, with shared utilities.

## Directory Structure

```
src/models/
├── config.yaml                      # Single source of truth for all hyperparameters
├── utils/                           # Shared losses, metrics, callbacks, config validation
│
├── lstm_benchmark_models/           # Dir 1/4: Benchmark LSTM models (m1–m6)
│   ├── architectures/               # M1 Vanilla, M2 BiLSTM, M3 CrissCross, M4 CNN-LSTM,
│   │                                #   M5 FeatureBiLSTM, M6 Graph BiLSTM
│   ├── modules/                     # Channel attention, criss-cross attention, graph attention
│   └── train_baseline.py            # Training script for m1–m6
│
├── improved_lstm_models/            # Dir 2/4: Enhanced LSTM with augmentation
│   ├── architectures/               # HierarchicalLSTM (two-level CNN + BiLSTM)
│   ├── augmentation.py              # EEG augmentation (time shift, noise, channel dropout, etc.)
│   ├── ensemble.py                  # Multi-model ensemble predictor
│   └── train.py                     # Training with warmup cosine schedule + augmentation
│
├── ensemble_transformers/           # Dir 3/4: VQ-Transformer + 7-model ensemble
│   ├── architectures/               # M7 VQ-Transformer
│   ├── modules/                     # Vector quantizer with EMA codebook
│   ├── ensemble.py                  # 7-model ensemble (m1–m6 + m7)
│   └── train_ensemble.py            # Ensemble evaluation CLI
│
└── hugging_face_mamba_moe/          # Dir 4/4: Mamba SSM + MoE + HuggingFace CNNs
    ├── architectures/               # EEGMamba, EEGMambaMoE, 8 HF CNN models,
    │   │                            #   4 pretrained wrappers (BENDR, BIOT, EEGPT, ST-EEGFormer)
    │   └── pretrained/              # HF Hub pretrained model wrappers with input validation
    ├── modules/                     # Mamba SSM blocks, MoE routing, CNN building blocks
    ├── train_mamba.py               # Mamba/MoE training with auxiliary load-balance loss
    └── train_hf.py                  # Generic HuggingFace model training
```

## Prerequisites

```bash
pip install torch numpy scikit-learn pyyaml mne
# For HuggingFace pretrained models (optional):
pip install braindecode huggingface_hub safetensors
```

## Step 1: Prepare Data

Convert raw CHB-MIT EDF files into tensor splits:

```bash
python src/prepare_tensors.py \
    --raw_dir src/data/raw/chbmit \
    --out_dir src/data/processed/chbmit
```

This produces `train/`, `val/`, `test/` subdirectories, each containing:
- `data.pt` — shape `(N, 16, 256)` float32 (16 channels × 256 timesteps at 256 Hz)
- `labels.pt` — shape `(N,)` float32 (0=background, 1=seizure)

## Step 2: Train Models

All training scripts read hyperparameters from `src/models/config.yaml`.

### Directory 1 — LSTM Benchmarks (m1–m6)

```bash
python -m src.models.lstm_benchmark_models.train_baseline \
    --model m1_vanilla_lstm \
    --data_path src/data/processed/chbmit

# Available models: m1_vanilla_lstm, m2_bilstm, m3_criss_cross,
#                   m4_cnn_lstm, m5_feature_bilstm, m6_graph_bilstm
```

### Directory 2 — Improved LSTM (HierarchicalLSTM + Augmentation)

```bash
python -m src.models.improved_lstm_models.train \
    --data_path src/data/processed/chbmit
```

### Directory 3 — Ensemble Evaluation (7-model)

Requires trained checkpoints from Dir 1 + Dir 3 in `outputs/models/`:

```bash
python -m src.models.ensemble_transformers.train_ensemble \
    --data_path src/data/processed/chbmit \
    --checkpoint_dir outputs/models
```

### Directory 4 — Mamba / Mamba-MoE

```bash
python -m src.models.hugging_face_mamba_moe.train_mamba \
    --model eeg_mamba \
    --data_path src/data/processed/chbmit

# Or with Mixture of Experts:
python -m src.models.hugging_face_mamba_moe.train_mamba \
    --model eeg_mamba_moe \
    --data_path src/data/processed/chbmit
```

### Directory 4 — HuggingFace CNN Models

```bash
python -m src.models.hugging_face_mamba_moe.train_hf \
    --model baseline_cnn_1d \
    --batch_size 32

# Available: baseline_cnn_1d, enhanced_cnn_1d, eegnet_local, eegnet,
#            deepconvnet, multiscale_cnn, multiscale_attention_cnn,
#            st_eegformer, bendr_pretrained, biot_pretrained,
#            eegpt_pretrained, hf_st_eegformer
```

## Configuration

Edit `src/models/config.yaml` to change hyperparameters. Key sections:

| Section | Controls |
|---------|----------|
| `data` | n_channels, time_steps, split ratios |
| `training` | lr, batch_size, epochs, pos_weight, gradient_clip, label_smoothing |
| `focal_loss` | gamma (2.0), reduction |
| `models.*` | Per-directory model hyperparameters |
| `outputs` | Checkpoint and results directories |

## Outputs

- **Checkpoints**: `outputs/models/{model_name}_best.pt`
- **Metrics reported**: F1-score, AUC-ROC, Sensitivity (printed at end of training)
