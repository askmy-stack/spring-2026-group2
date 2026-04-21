# Automated EEG Seizure Detection

**A Multi-Model Comparative Study on the CHB-MIT Dataset**  
Spring 2026 | GWU Research Group 2

---

## Overview

End-to-end pipeline for automated seizure detection from scalp EEG recordings. We compare classical ML and attention-based tabular models on the CHB-MIT Scalp EEG dataset.

**Key results (subject-independent evaluation):**

| Model | Test AUCPR | Test ROC-AUC | Test F1 |
|-------|-----------|-------------|---------|
| LightGBM (Optuna) | **0.334** | **0.901** | **0.235** |
| XGBoost (Optuna) | 0.312 | 0.889 | 0.221 |
| TabNet Optuna | 0.318 | 0.893 | 0.218 |
| TabNet Baseline | 0.298 | 0.881 | 0.201 |
| Random Forest (Optuna) | 0.285 | 0.871 | 0.198 |

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
- Val: chb01, chb03, chb16, chb18 (4 subjects)
- Test: chb05, chb07, chb21, chb23 (4 subjects)

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
│   │   └── tabnet_optuna.yaml
│   ├── feature_engineering/
│   │   ├── extractor.py           # AdvancedFeatureExtractor (528 features)
│   │   └── run_features_from_index.py  # Feature extraction pipeline
│   ├── models/
│   │   ├── baseline/
│   │   │   ├── train_model.py         # LightGBM / XGBoost / RF training
│   │   │   └── train_tabnet.py        # TabNet baseline
│   │   ├── improved/
│   │   │   ├── optuna_lightgbm.py     # Optuna-tuned LightGBM
│   │   │   ├── optuna_xgboost.py      # Optuna-tuned XGBoost
│   │   │   ├── optuna_random_forest.py
│   │   │   └── optuna_tabnet.py       # Optuna-tuned TabNet
│   │   └── utils/
│   │       ├── config_utils.py        # Portable path resolution
│   │       ├── data_utils.py          # load_split, validate_feature_columns
│   │       ├── metric_utils.py        # AUCPR, ROC-AUC, F1, threshold sweep
│   │       ├── io_utils.py            # ensure_dir, save_csv, save_json
│   │       ├── plot_utils.py          # PR curve, ROC, confusion matrix
│   │       └── prepare_memmap.py      # CSV → memory-mapped arrays for TabNet
│   └── dataloaders/                   # EDF windowing & tensor cache
├── requirements.txt
├── check_setup.py                 # Structural verification
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
python check_setup.py    # structural checks
python stress_test.py    # runtime checks (uses synthetic data)
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
python -m src.models.improved.optuna_lightgbm \
    --config src/config/baseline_lightgbm.yaml

# XGBoost
python -m src.models.improved.optuna_xgboost \
    --config src/config/baseline_xgboost.yaml

# Random Forest
python -m src.models.improved.optuna_random_forest \
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

# Optuna-tuned TabNet
python -m src.models.improved.optuna_tabnet \
    --config src/config/tabnet_optuna.yaml
```

---

## Features

**528 features per 1-second window (33 per channel × 16 channels):**

| Category | Features | Count |
|----------|---------|-------|
| Time-domain | Mean, std, RMS, min, max, range, line length, ZCR, skewness, kurtosis | 10 × 16 ch = 160 |
| Hjorth parameters | Activity, mobility, complexity | 3 × 16 ch = 48 |
| Nonlinear | Sample entropy, permutation entropy | 2 × 16 ch = 32 |
| Frequency-domain | Band power (δ θ α β γ), relative power (×5), total power, spectral entropy | 12 × 16 ch = 192 |
| FFT | Dominant frequency | 1 × 16 ch = 16 |
| Wavelet | Approx energy, detail energy D1/D2/D3, wavelet entropy | 5 × 16 ch = 80 |
| **Total** | | **528** |

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
