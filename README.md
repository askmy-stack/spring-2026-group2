# Automated EEG Seizure Detection

**A Multi-Model Comparative Study on the CHB-MIT Dataset**  
Spring 2026 | GWU Research Group 2

---

## Overview

End-to-end pipeline for automated seizure detection from scalp EEG recordings. We compare classical ML, attention-based tabular models, and a hybrid deep learning architecture on the CHB-MIT Scalp EEG dataset.

**Key results (subject-independent evaluation):**

| Model | Test AUCPR | Test ROC-AUC | Test F1 |
|-------|-----------|-------------|---------|
| LightGBM (Optuna) | **0.334** | 0.901 | 0.235 |
| H-TabNet | 0.331 | **0.907** | **0.247** |
| XGBoost (Optuna) | 0.312 | 0.889 | 0.221 |
| TabNet Optuna | 0.318 | 0.893 | 0.218 |
| Random Forest | 0.285 | 0.871 | 0.198 |
| Hybrid TCN+Transformer+BiLSTM | 0.035 | — | — |

> Primary metric: **AUCPR** — correct metric for severe class imbalance (0.3% seizure rate)

---

## Dataset

**CHB-MIT Scalp EEG** (Boston Children's Hospital)
- 24 pediatric patients, ages 1.5–22
- 916 hours of continuous EEG
- 198 seizure events, 16 channels, 256 Hz
- 3.4 million 1-second windows
- Class imbalance: ~1 seizure window per 300 background windows

**Subject-independent split:**
- Train: chb02, 04, 06, 08, 09, 10, 11, 12, 13, 14, 15, 17, 19, 20, 22, 24 (16 subjects)
- Val: chb01, 03, 16, 18 (4 subjects)
- Test: chb05, 07, 21, 23 (4 subjects)

---

## Project Structure

```
eeg/
├── src/
│   ├── config/                    # All YAML configs
│   │   ├── feature_engineering.yaml
│   │   ├── baseline_lightgbm.yaml
│   │   ├── baseline_xgboost.yaml
│   │   ├── baseline_random_forest.yaml
│   │   ├── tabnet_baseline.yaml
│   │   ├── tabnet_ect.yaml
│   │   ├── tabnet_hier.yaml
│   │   └── tabnet_optuna.yaml
│   ├── feature_engineering/
│   │   ├── extractor.py           # AdvancedFeatureExtractor (528 features)
│   │   └── run_features_from_index.py  # Feature extraction pipeline
│   ├── models/
│   │   ├── baseline/
│   │   │   ├── train_model.py         # LightGBM / XGBoost / RF training
│   │   │   ├── optuna_lightgbm.py     # Optuna-tuned LightGBM
│   │   │   ├── optuna_xgboost.py      # Optuna-tuned XGBoost
│   │   │   ├── optuna_random_forest.py
│   │   │   ├── train_tabnet.py        # TabNet baseline
│   │   │   └── optuna_tabnet.py       # Optuna-tuned TabNet
│   │   ├── improved/
│   │   │   └── train_tabnet_advanced.py  # ECT-TabNet / H-TabNet
│   │   └── utils/
│   │       ├── config_utils.py        # Portable path resolution
│   │       ├── data_utils.py          # load_split, validate_feature_columns
│   │       ├── metric_utils.py        # AUCPR, ROC-AUC, F1, threshold sweep
│   │       ├── io_utils.py            # ensure_dir, save_csv, save_json
│   │       ├── plot_utils.py          # PR curve, ROC, confusion matrix
│   │       └── prepare_memmap.py      # CSV → memory-mapped arrays for TabNet
│   └── dataloaders/                   # EDF windowing & tensor cache
├── requirements.txt
├── check_setup.py                 # Structural verification (14 checks)
└── stress_test.py                 # Runtime verification (19 checks)
```

---

## Quickstart

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Verify setup

```bash
python check_setup.py    # 14 structural checks
python stress_test.py    # 19 runtime checks (uses synthetic data)
```

### 3. Feature extraction

```bash
# Train split
python -m src.feature_engineering.run_features_from_index \
    --config src/config/feature_engineering.yaml \
    --split train --n-jobs 4

# Val and test splits
python -m src.feature_engineering.run_features_from_index \
    --config src/config/feature_engineering.yaml \
    --split val --n-jobs 2

python -m src.feature_engineering.run_features_from_index \
    --config src/config/feature_engineering.yaml \
    --split test --n-jobs 2
```

Output: `results/features_raw/features_{train,val,test}.csv`

### 4. Train classical ML models

```bash
# LightGBM with Optuna tuning (50 trials)
python -m src.models.baseline.optuna_lightgbm \
    --config src/config/baseline_lightgbm.yaml

# XGBoost
python -m src.models.baseline.optuna_xgboost \
    --config src/config/baseline_xgboost.yaml

# Random Forest
python -m src.models.baseline.optuna_random_forest \
    --config src/config/baseline_random_forest.yaml
```

### 5. Train TabNet models

```bash
# Prepare memory-mapped arrays (required for TabNet)
python -m src.models.utils.prepare_memmap \
    --config src/config/tabnet_baseline.yaml

# Baseline TabNet
python -m src.models.baseline.train_tabnet \
    --config src/config/tabnet_baseline.yaml

# ECT-TabNet (Channel Transformer)
python -m src.models.improved.train_tabnet_advanced \
    --config src/config/tabnet_ect.yaml

# H-TabNet (Hierarchical encoder) — best overall
python -m src.models.improved.train_tabnet_advanced \
    --config src/config/tabnet_hier.yaml
```

---

## Features

**528 features per 1-second window:**

| Category | Features | Count |
|----------|---------|-------|
| Time-domain | Mean, variance, skewness, kurtosis, ZCR, line length, Hjorth parameters | 11 × 16 ch = 176 |
| Frequency-domain | Delta/Theta/Alpha/Beta/Gamma power, spectral entropy, peak frequency | 12 × 16 ch = 192 |
| Cross-channel | Correlation matrix, coherence, spatial synchrony | 10 × 16 ch = 160 |

---

## Model Architectures

### H-TabNet (Best ROC-AUC & F1)
Hierarchical channel encoder: shared per-channel encoder → Squeeze-Excitation attention → TabNet sequential attention.

### ECT-TabNet
16 EEG channels treated as tokens → multi-head self-attention (4 heads) → TabNet.

### Hybrid Deep Learning (TCN + Transformer + BiLSTM)
Three parallel encoders on raw EEG tensors (16 × 256):
- TCN: 5 dilated conv blocks (dilations 1→16), receptive field = 256 samples
- Channel Transformer: 2-layer self-attention across channels
- BiLSTM: bidirectional sequence model over 256 timesteps

---

## Configuration

All paths in YAML configs are relative to the config file and resolved automatically:

```yaml
# src/config/baseline_lightgbm.yaml
paths:
  train_csv: ../../results/features_raw/features_train.csv
  val_csv:   ../../results/features_raw/features_val.csv
  test_csv:  ../../results/features_raw/features_test.csv
  output_dir: ../../results/modeling/lightgbm/optuna
```

No hardcoded paths anywhere — clone and run from any location.

---

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- See `requirements.txt` for full list

---

## Citation / Dataset

PhysioNet CHB-MIT Scalp EEG Database:  
https://physionet.org/content/chbmit/1.0.0/
