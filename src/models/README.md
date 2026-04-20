# EEG Seizure Detection — Model Directory

Canonical pipeline: **three training directories + one shared utilities package + one orchestrator**. Everything else has been deleted (root-level duplicates, superseded `improved/` and `architectures/` folders) or archived under `_experimental/` (research WIP that is not wired into the canonical pipeline).

## Directory Layout

```
src/models/
├── config.yaml                   # Single source of truth for hyperparameters
├── run_all_models.py             # Orchestrator: runs all three phases sequentially
├── README.md                     # This file
│
├── utils/                        # Shared across every training/inference entry point
│   ├── losses.py                 # FocalLoss (with pos_weight + label smoothing)
│   ├── metrics.py                # F1 / AUROC / sens / spec / find_optimal_threshold
│   ├── callbacks.py              # EarlyStopping, gradient clipping
│   ├── config_validator.py       # Validates config.yaml shape
│   └── checkpoint.py             # UNIFIED .pt schema — save/load_checkpoint
│
├── lstm_benchmark_models/        # Phase 1: m1–m6 LSTM/CNN baselines
│   ├── architectures/            #   m1 Vanilla, m2 BiLSTM, m3 CrissCross,
│   │                             #   m4 CNN-LSTM, m5 FeatureBiLSTM, m6 GraphBiLSTM
│   ├── modules/                  #   attention blocks, graph attention
│   └── train_baseline.py         #   → writes unified-schema .pt
│
├── improved_lstm_models/         # Phase 2: HierarchicalLSTM + augmentation
│   ├── architectures/            #   two-level CNN + BiLSTM
│   ├── augmentation.py           #   time shift, noise, channel dropout, mixup
│   ├── train.py                  #   early-stops on val-F1 (not val-loss)
│   └── ensemble.py               #   EnsemblePredictor class + CLI over .pt dir
│
├── hugging_face_mamba_moe/       # Phase 3: Mamba SSM, MoE, HF CNNs, pretrained wrappers
│   ├── architectures/            #   EEGMamba, EEGMambaMoE, 8 HF CNNs,
│   │   └── pretrained/           #   4 HF-Hub wrappers (BENDR, BIOT, EEGPT, ST-EEGFormer)
│   ├── modules/                  #   Mamba SSM blocks, MoE routing, CNN primitives
│   ├── train_mamba.py            #   Mamba / MoE training (+ aux load-balance loss)
│   ├── train_hf.py               #   Generic HF model trainer (unified-schema save)
│   └── ensemble_hf.py            #   Ensemble over all trained HF checkpoints
│
├── tools/                        # Developer tools
│   └── infer_edf.py              #   Smoke test: load .pt + run on a raw .edf file
│
├── legacy_baseline/              # Preserved only so baseline_results.json stays reproducible
│
└── _experimental/                # Archived research WIP — NOT on the canonical path
    ├── approach2/                #   7-model stacking ensemble w/ BO meta-learner
    ├── approach3/                #   Mamba distillation + diffusion pretraining
    └── ensemble_transformers/    #   VQ-Transformer + 7-model ensemble
```

## Prerequisites

```bash
pip install torch numpy scikit-learn pyyaml mne
# For the HuggingFace pretrained wrappers (optional):
pip install 'braindecode[hug]' huggingface_hub safetensors
```

## End-to-End Workflow

### 1. Prepare Data

Convert raw CHB-MIT EDF files to tensor splits with the shared preprocessing
pipeline (`src/data_loader/core/signal.py`: resample → bandpass → notch → avg-reference):

```bash
python -m src.data_loader.precache \
    --raw_dir src/data/raw/chbmit \
    --out_dir src/data/processed/chbmit
```

Output layout:

```
src/data/processed/chbmit/
├── train/  data.pt  labels.pt
├── val/    data.pt  labels.pt
└── test/   data.pt  labels.pt
```

Tensors are `(N, 16, 256)` float32 at 256 Hz with 1-second windows; labels are `(N,)` long.

### 2. Train Models

All training scripts read `src/models/config.yaml` and write a **unified
checkpoint** (see below) to `outputs/models/` by default.

```bash
# Phase 1 — benchmarks (m1–m6, or 'all')
python -m src.models.lstm_benchmark_models.train_baseline \
    --model m1_vanilla_lstm --data_path src/data/processed/chbmit

# Phase 2 — improved LSTM (HierarchicalLSTM + augmentation)
python -m src.models.improved_lstm_models.train \
    --data_path src/data/processed/chbmit

# Phase 3a — Mamba / Mamba-MoE
python -m src.models.hugging_face_mamba_moe.train_mamba \
    --model eeg_mamba --data_path src/data/processed/chbmit

# Phase 3b — HuggingFace models (eegnet, deepconvnet, eegpt_pretrained, …)
python -m src.models.hugging_face_mamba_moe.train_hf \
    --model eegnet --data_path src/data/processed/chbmit

# Or run every phase at once
python -m src.models.run_all_models --data_path src/data/processed/chbmit
```

### 3. Ensemble

Once ≥ 2 checkpoints exist, ensemble them with the helper CLIs:

```bash
# Ensemble all improved-LSTM .pt files in a directory
python -m src.models.improved_lstm_models.ensemble \
    --data_path src/data/processed/chbmit \
    --ckpt_dir src/models/improved_lstm_models/checkpoints

# Ensemble all HF-model checkpoints
python -m src.models.hugging_face_mamba_moe.ensemble_hf \
    --data_path src/data/processed/chbmit --strategy weighted
```

Both auto-discover `.pt` files, rebuild models via `load_checkpoint`, run per-member
inference, and report **mean** and **val-F1-weighted** ensemble metrics with the
threshold tuned on validation data.

### 4. Inference Smoke Test

Verify that a checkpoint is loadable and runnable end-to-end on a raw EDF:

```bash
python -m src.models.tools.infer_edf \
    --edf src/data/raw/chbmit/chb01/chb01_01.edf \
    --ckpt outputs/models/m1_vanilla_lstm_best.pt
```

The tool reads the checkpoint's stored `input_spec` + `preprocess` + `optimal_threshold`,
so no external configuration is needed.

## Unified Checkpoint Schema

Every canonical training script writes `.pt` files via
`src.models.utils.checkpoint.save_checkpoint`. The on-disk payload is:

| Key                     | Type        | Purpose                                                       |
|-------------------------|-------------|---------------------------------------------------------------|
| `schema_version`        | int         | Currently `1`; bump on breaking changes.                      |
| `model_class`           | str         | `"pkg.module.ClassName"` — used to auto-rebuild the module.   |
| `model_config`          | dict        | Constructor kwargs for `_try_build_model`.                    |
| `model_state_dict`      | dict        | Tensor weights.                                               |
| `optimizer_state_dict`  | dict / None | For training resume.                                          |
| `epoch`                 | int         | Last epoch at save time.                                      |
| `val_metrics`           | dict        | `{"f1", "auroc", "sens", "spec"}` at save time.               |
| `optimal_threshold`     | float       | Decision threshold tuned on validation data.                  |
| `input_spec`            | dict        | `{"channels": 16, "sfreq": 256, "window_sec": 1.0}`.          |
| `preprocess`            | dict        | Params needed to reproduce preprocessing at inference time.   |
| `git_commit`            | str / None  | Short SHA for provenance.                                     |

Load any checkpoint uniformly:

```python
from src.models.utils.checkpoint import load_checkpoint
model, payload = load_checkpoint("outputs/models/m1_vanilla_lstm_best.pt")
threshold = payload["optimal_threshold"]
```

If auto-rebuild fails (e.g., HF pretrained wrappers that require a factory), the
caller can still access the raw `payload["model_state_dict"]` and build the
model manually.

## Configuration

All hyperparameters live in `src/models/config.yaml`:

| Section         | Controls                                                                |
|-----------------|-------------------------------------------------------------------------|
| `data`          | `n_channels`, `time_steps`, split layout                                |
| `training`      | `learning_rate`, `batch_size`, `num_epochs`, `pos_weight`, grad clip    |
| `focal_loss`    | `gamma`, `reduction`                                                    |
| `models.*`      | Per-phase model hyperparameters; may override `training` keys           |
| `outputs`       | `checkpoint_dir`, `results_dir`                                         |

`src/data_loader/config.yaml` is intentionally separate (controls preprocessing
and caching — not the model pipeline). The two are kept decoupled on purpose.

## What Was Deleted / Archived

- **Deleted** (superseded by canonical dirs): root-level `attention_bilstm.py`,
  `bilstm.py`, `cnn_lstm.py`, `feature_bilstm.py`, `vanilla_lstm.py`,
  `compare.py`, `train.py`, `ensemble.py`, `run_benchmark.py`, and the
  `improved/`, `improved_lstm/`, `architectures/` directories.
- **Archived** to `_experimental/` (non-canonical but non-trivial IP):
  `approach2/`, `approach3/`, `ensemble_transformers/`.
- **Kept** at top level for reproducibility of the research paper draft:
  `legacy_baseline/`.
