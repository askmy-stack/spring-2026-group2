# EEG Seizure Detection — Models

All models are run from the **project root** using `python -m src.models.*`.  
Primary metric: **AUCPR** (Area Under Precision-Recall Curve) — correct for severe class imbalance (0.3% seizure rate).

---

## Results Summary

| Model | Val AUCPR | Test AUCPR | Test ROC-AUC | Test F1 |
|-------|-----------|-----------|-------------|---------|
| Random Forest (Optuna) | 0.031 | 0.285 | 0.871 | 0.198 |
| XGBoost (Optuna) | 0.038 | 0.312 | 0.889 | 0.221 |
| LightGBM (Optuna) | **0.042** | **0.334** | 0.901 | 0.235 |
| TabNet Baseline | 0.038 | 0.298 | 0.881 | 0.201 |
| TabNet Optuna | 0.041 | 0.318 | 0.893 | 0.218 |
| ECT-TabNet | 0.035 | 0.305 | 0.889 | 0.214 |
| H-TabNet | 0.044 | 0.331 | **0.907** | **0.247** |
| Hybrid TCN+Transformer+BiLSTM | 0.049 | 0.035 | — | — |

---

## Folder Structure

```
src/models/
├── baseline/
│   ├── train_model.py          # LightGBM / XGBoost / RF — single training run
│   ├── optuna_lightgbm.py      # LightGBM with Optuna Bayesian tuning
│   ├── optuna_xgboost.py       # XGBoost with Optuna Bayesian tuning
│   ├── optuna_random_forest.py # Random Forest with Optuna Bayesian tuning
│   ├── train_tabnet.py         # TabNet baseline training
│   └── optuna_tabnet.py        # TabNet with Optuna tuning
├── improved/
│   └── train_tabnet_advanced.py  # ECT-TabNet and H-TabNet
└── utils/
    ├── config_utils.py         # YAML loader with portable path resolution
    ├── data_utils.py           # load_split, validate_feature_columns
    ├── metric_utils.py         # AUCPR, ROC-AUC, F1, threshold sweep
    ├── io_utils.py             # ensure_dir, save_csv, save_json
    ├── plot_utils.py           # PR curve, ROC curve, confusion matrix, feature importance
    └── prepare_memmap.py       # Convert feature CSVs to memory-mapped arrays for TabNet
```

---

## Prerequisites

Feature CSVs must exist before running any model:

```
results/features_raw/
├── features_train.csv
├── features_val.csv
└── features_test.csv
```

See `src/feature_engineering/README.md` for how to generate them.

---

## Classical ML Models

### Single training run (no Optuna)

```bash
# LightGBM
python -m src.models.baseline.train_model \
    --config src/config/baseline_lightgbm.yaml

# XGBoost
python -m src.models.baseline.train_model \
    --config src/config/baseline_xgboost.yaml

# Random Forest
python -m src.models.baseline.train_model \
    --config src/config/baseline_random_forest.yaml
```

### With Optuna hyperparameter tuning (recommended)

```bash
# LightGBM — 50 Optuna trials, maximizes val AUCPR
python -m src.models.baseline.optuna_lightgbm \
    --config src/config/baseline_lightgbm.yaml

# XGBoost
python -m src.models.baseline.optuna_xgboost \
    --config src/config/baseline_xgboost.yaml

# Random Forest
python -m src.models.baseline.optuna_random_forest \
    --config src/config/baseline_random_forest.yaml
```

**What Optuna tunes:**

| Model | Tuned Parameters |
|-------|-----------------|
| LightGBM | n_estimators, learning_rate, num_leaves, max_depth, min_child_samples, subsample, colsample_bytree, reg_alpha, reg_lambda |
| XGBoost | n_estimators, learning_rate, max_depth, subsample, colsample_bytree, reg_alpha, reg_lambda, scale_pos_weight |
| Random Forest | n_estimators, max_depth, min_samples_leaf, min_samples_split, max_features |

**Outputs** (saved to `output_dir` from config):
- `best_model.joblib` — trained model
- `best_params.json` — winning hyperparameters
- `metrics_val.json` / `metrics_test.json` — AUCPR, ROC-AUC, F1, precision, recall
- `threshold_sweep.csv` — F1 at every threshold from 0.01 to 0.99
- `pr_curve.png`, `roc_curve.png`, `confusion_matrix.png`, `feature_importance.png`

---

## TabNet Models

TabNet requires memory-mapped arrays instead of loading the full CSV into RAM.

### Step 1 — Prepare memmap (run once per config)

```bash
python -m src.models.utils.prepare_memmap \
    --config src/config/tabnet_baseline.yaml
```

This creates:
```
results/modeling/tabnet/memmap/
├── X_train.dat / y_train.dat / train_meta.json
├── X_val.dat   / y_val.dat   / val_meta.json
└── X_test.dat  / y_test.dat  / test_meta.json
```

Use `--force` to overwrite existing memmap files.

### Step 2 — Train TabNet

```bash
# Baseline TabNet
python -m src.models.baseline.train_tabnet \
    --config src/config/tabnet_baseline.yaml

# Optuna-tuned TabNet
python -m src.models.baseline.optuna_tabnet \
    --config src/config/tabnet_optuna.yaml
```

---

## Advanced TabNet Models (ECT-TabNet & H-TabNet)

Both variants extend TabNet with EEG-specific channel encoding layers.

```bash
# ECT-TabNet — Channel Transformer pre-layer
python -m src.models.improved.train_tabnet_advanced \
    --config src/config/tabnet_ect.yaml

# H-TabNet — Hierarchical channel encoder (best ROC-AUC & F1)
python -m src.models.improved.train_tabnet_advanced \
    --config src/config/tabnet_hier.yaml
```

### ECT-TabNet Architecture
16 EEG channels treated as tokens → Multi-head self-attention (4 heads) with positional encoding → learns which channels are most active during seizure → TabNet sequential attention on the encoded features.

### H-TabNet Architecture
Shared per-channel encoder processes each of 16 channels independently → Squeeze-Excitation (SE) attention re-weights channels by importance → concatenated representation → TabNet sequential attention.

**Why H-TabNet wins on ROC-AUC and F1:**  
Hierarchical encoding captures spatial EEG topology — adjacent channels tend to show correlated seizure activity that SE attention learns to exploit before TabNet processes features.

---

## Utils

### config_utils.py
Loads YAML configs and resolves all relative paths from the config file's own directory — no hardcoded paths anywhere.

```python
from src.models.utils.config_utils import load_config
cfg = load_config("src/config/baseline_lightgbm.yaml")
```

### data_utils.py
```python
from src.models.utils.data_utils import load_split, validate_feature_columns

X_train, y_train, train_cols, meta = load_split(
    csv_path, target_col="label", meta_cols=["path","start_sec","end_sec"]
)
validate_feature_columns(train_cols, val_cols, test_cols)
```

### metric_utils.py
```python
from src.models.utils.metric_utils import compute_binary_metrics, sweep_thresholds_for_f1

metrics = compute_binary_metrics(y_true, y_prob, threshold=0.5)
# Returns: aucpr, roc_auc, f1, precision, recall, confusion_matrix

best_threshold, rows = sweep_thresholds_for_f1(y_true, y_prob)
```

### prepare_memmap.py
```bash
python -m src.models.utils.prepare_memmap \
    --config src/config/tabnet_baseline.yaml \
    --chunksize 50000   # rows per chunk (default: 50000)
    --force             # overwrite existing files
```

---

## Config Files

All configs live in `src/config/`. Key sections:

```yaml
model_type: lightgbm          # lightgbm | xgboost | random_forest | tabnet | tabnet_advanced

paths:
  train_csv:  ../../results/features_raw/features_train.csv
  val_csv:    ../../results/features_raw/features_val.csv
  test_csv:   ../../results/features_raw/features_test.csv
  output_dir: ../../results/modeling/lightgbm/optuna

data:
  meta_cols: [path, start_sec, end_sec]
  dtype: float32

target_col: label

optuna:
  n_trials: 50
  direction: maximize       # maximize val AUCPR
```

---

## Running in Background (HPC / AWS)

```bash
nohup python -m src.models.baseline.optuna_lightgbm \
    --config src/config/baseline_lightgbm.yaml \
    > results/logs/lgbm.log 2>&1 &

tail -f results/logs/lgbm.log
```
