# EEG Seizure Detection Model Report (CHB-MIT, 16x256)

## Abstract
This report summarizes the EEG seizure detection pipeline and model experiments on CHB-MIT using 1-second windows (`16 channels x 256 samples`). We compared CNN and EEGNet baselines plus improved variants. The processed dataset produced subject-independent train/val/test tensors with balancing to 30% seizure in each split. In captured runs, `cnn_benchmark` and `eegnet` achieved similar final test F1 (~0.528), while `cnn_improved` and `eegnet_improved` showed different precision-recall tradeoffs. AUROC/AUPRC were `nan` in logs and should be recomputed after fixing metric dependency/runtime.

## 1. Objective
- Build strong CNN-based baselines for binary seizure detection.
- Standardized input: `(B, 16, 256)` EEG windows.
- Evaluate with threshold-tuned and fixed-threshold metrics.

## 2. Data and Pipeline
- Window size: `1.0s`
- Sampling rate: `256 Hz`
- Channels: `16`
- Stride: `1.0s`
- Subject-independent split

### Generated Data Summary
- Total windows (pre-balance): `114,511`
- Seizure windows (pre-balance): `11,611` (`10.1%`)
- Train: `96,857` (`30.0%` seizure)
- Val: `19,500` (`30.0%` seizure)
- Test: `30,642` (`30.0%` seizure)

Output locations:
- CSV index files: `results/dataloader/chbmit/`
- Tensor files: `results/tensors/chbmit/`

## 3. Models Implemented

### 3.1 CNN Benchmark (`cnn_benchmark`)
- 1D temporal CNN
- Conv stack: `16->32 (k=7)`, `32->64 (k=5)`, `64->128 (k=3)`
- Pooling after each block
- Dropout regularization
- Adaptive global average pooling + linear output head

### 3.2 EEGNet Baseline (`eegnet`)
- Temporal filtering conv
- Depthwise spatial conv across channels
- Separable temporal refinement
- Compact EEG-specific architecture

### 3.3 CNN Improved (`cnn_improved`)
- Residual-style 1D CNN with attention-like improvements
- Run config: `base_channels=48`, `dropout=0.25`
- Trainable parameters (logged): `425,563`

### 3.4 EEGNet Improved (`eegnet_improved`)
- Higher feature capacity and tuned temporal settings
- Run config: `F1=16`, `D=2`, `kernel_length=32`
- Trainable parameters (logged): `4,257`

## 4. Training Setup
- Loss: `BCEWithLogitsLoss(pos_weight=n_neg/n_pos)`
- Optimizer: `AdamW`
- LR scheduler: `ReduceLROnPlateau`
- Gradient clipping: `1.0`
- Epochs: `15`
- Device: `cuda:0`

## 5. Final Test Metrics (Captured)

| Model | Threshold | Accuracy | Precision | Recall | F1 | Specificity |
|---|---:|---:|---:|---:|---:|---:|
| `cnn_benchmark` | 0.04 | 0.59298 | 0.40499 | 0.76055 | 0.52854 | 0.52117 |
| `eegnet` | 0.27 | 0.52846 | 0.37728 | 0.87913 | 0.52798 | 0.37818 |
| `cnn_improved` | 0.04 | 0.51671 | 0.35385 | 0.73977 | 0.47872 | 0.42112 |
| `eegnet_improved` | 0.19 | 0.52967 | 0.36740 | 0.78677 | 0.50090 | 0.41949 |

Additional confusion counts from logs:
- `cnn_benchmark`: TP 6991, FP 10271, TN 11179, FN 2201
- `eegnet`: TP 8081, FP 13338, TN 8112, FN 1111
- `cnn_improved`: TP 6800, FP 12417, TN 9033, FN 2392
- `eegnet_improved`: TP 7232, FP 12452, TN 8998, FN 1960

## 6. Interpretation
- `cnn_benchmark` and `eegnet` delivered similar final F1, but with very different operating profiles.
- `eegnet` achieved highest recall, but low specificity (more false positives).
- `cnn_benchmark` gave better specificity and better overall balance.
- `cnn_improved` underperformed `cnn_benchmark` in this run (lower F1 and accuracy).
- `eegnet_improved` improved over `cnn_improved` F1, but did not exceed `cnn_benchmark`/`eegnet` baseline F1 in captured runs.

## 7. Metric Caveat
AUROC and AUPRC were `nan` in logs. These primary metrics are currently invalid in recorded runs and must be recomputed after fixing runtime metric dependencies.

## 8. Recommended Next Steps
1. Install/verify `scikit-learn` and rerun to recover valid AUROC/AUPRC.
2. Keep balancing on train only; evaluate val/test at natural prevalence.
3. Add focal loss (`gamma=2`, `alpha=0.25`) and compare against BCE.
4. Add early stopping and checkpointing on validation AUPRC.
5. Run controlled ablation with fixed seed across all 4 models.
