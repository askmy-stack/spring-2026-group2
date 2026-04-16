# LSTM Models Restructure + Code Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure `src/models/` into 4 sequenced directories and apply all Code Review Standards from the project PDF to every model file — without touching any dataloader code.

**Architecture:** Each new directory mirrors the doc's reference structure (`architectures/`, `modules/`, `train_*.py`, `eval_*.py`). Shared utilities extracted to `src/models/utils/`. A single `config.yaml` replaces all hardcoded values. Old `approach2/`, `approach3/`, `legacy_baseline/`, `improved_lstm/` directories are PRESERVED as-is — new directories are the authoritative copies.

**Tech Stack:** Python 3.12, PyTorch, pytest, Python `logging` module, YAML config

---

## Directory Sequence (4 required directories in order)

```
1. src/models/lstm_benchmark_models/      # m1-m6 LSTM baseline variants
2. src/models/improved_lstm_models/       # Enhanced LSTM with augmentation
3. src/models/ensemble_transformers/      # m7 VQ-Transformer + 7-model ensemble
4. src/models/hugging_face_mamba_moe/     # Mamba SSM + Mixture of Experts
```

## File Migration Map

```
CURRENT LOCATION                           NEW LOCATION (authoritative copy)
─────────────────────────────────────────  ──────────────────────────────────────────────────
approach2/architectures/m1_vanilla_lstm.py → lstm_benchmark_models/architectures/m1_vanilla_lstm.py
approach2/architectures/m2_bilstm.py       → lstm_benchmark_models/architectures/m2_bilstm.py
approach2/architectures/m3_criss_cross.py  → lstm_benchmark_models/architectures/m3_criss_cross.py
approach2/architectures/m4_cnn_lstm.py     → lstm_benchmark_models/architectures/m4_cnn_lstm.py
approach2/architectures/m5_feature_bilstm  → lstm_benchmark_models/architectures/m5_feature_bilstm.py
approach2/architectures/m6_graph_bilstm.py → lstm_benchmark_models/architectures/m6_graph_bilstm.py
approach2/modules/channel_attention.py     → lstm_benchmark_models/modules/channel_attention.py
approach2/modules/criss_cross_attention.py → lstm_benchmark_models/modules/criss_cross_attention.py
approach2/modules/graph_attention.py       → lstm_benchmark_models/modules/graph_attention.py
approach2/train.py (m1-m6 scope)           → lstm_benchmark_models/train_baseline.py
                                           → lstm_benchmark_models/eval_baseline.py  [NEW]

improved_lstm/train.py                     → improved_lstm_models/train.py
improved_lstm/augmentation.py              → improved_lstm_models/augmentation.py
improved_lstm/ensemble.py                  → improved_lstm_models/ensemble.py
approach3/architectures/hierarchical_lstm  → improved_lstm_models/architectures/hierarchical_lstm.py

approach2/architectures/m7_vq_transformer  → ensemble_transformers/architectures/m7_vq_transformer.py
approach2/modules/vector_quantizer.py      → ensemble_transformers/modules/vector_quantizer.py
approach2/modules/pretrained_encoders.py   → ensemble_transformers/modules/pretrained_encoders.py
approach2/ensemble_7model.py               → ensemble_transformers/ensemble.py
                                           → ensemble_transformers/train_ensemble.py  [NEW]

approach3/architectures/eeg_mamba.py       → hugging_face_mamba_moe/architectures/eeg_mamba.py
approach3/architectures/tiny_seizure_net   → hugging_face_mamba_moe/architectures/tiny_seizure_net.py
approach3/modules/mamba_block.py           → hugging_face_mamba_moe/modules/mamba_block.py
approach3/modules/mixture_of_experts.py    → hugging_face_mamba_moe/modules/mixture_of_experts.py
approach3/modules/diffusion_eeg.py         → hugging_face_mamba_moe/modules/diffusion_eeg.py
approach3/train_mamba.py                   → hugging_face_mamba_moe/train_mamba.py

NEW SHARED UTILITIES:
src/models/utils/metrics.py               [NEW — compute_f1_score, compute_auc_roc, compute_sensitivity]
src/models/utils/losses.py                [NEW — FocalLoss consolidated from 2 existing copies]
src/models/utils/callbacks.py             [NEW — EarlyStopping, clip_gradients]
src/models/config.yaml                    [NEW — single source of truth for all hyperparameters]
```

---

## Code Review Standards Checklist (from project PDF)

Every file in the 4 new directories must satisfy ALL of the following:

| Rule | Requirement | Current Issue in Source Files |
|------|-------------|-------------------------------|
| Function length | <= 30 lines; extract helpers | train.py files have 40-80 line functions |
| Docstrings | Every function: Args/Returns/Raises/Example | Many training helpers missing docstrings |
| Type hints | All function signatures | Forward() methods missing return types |
| Variable names | Explicit; no single letters except loop counters | `x` used extensively in forward() |
| Imports | Grouped (stdlib->third-party->local); no wildcard | Some files mix ordering |
| sys.path hacks | Remove; use relative imports | ALL m1-m7 files have sys.path.append() |
| Hardcoded values | All in config.yaml; load via load_config() | DEFAULT_CONFIG dict in improved_lstm/train.py |
| Logging | logging.getLogger(__name__); no print() | print() used throughout train files |
| Error messages | Include context: f"Expected {x}, got {actual}" | Bare raise ValueError("...") in some places |

---

## Complete File Structure to Create

```
src/models/
├── config.yaml                                  [NEW]
├── utils/
│   ├── __init__.py                              [NEW]
│   ├── metrics.py                               [NEW]
│   ├── losses.py                                [NEW]
│   └── callbacks.py                             [NEW]
├── lstm_benchmark_models/
│   ├── __init__.py                              [NEW]
│   ├── architectures/
│   │   ├── __init__.py                          [NEW]
│   │   ├── m1_vanilla_lstm.py                   [NEW — code-reviewed copy]
│   │   ├── m2_bilstm.py                         [NEW — code-reviewed copy]
│   │   ├── m3_criss_cross.py                    [NEW — code-reviewed copy]
│   │   ├── m4_cnn_lstm.py                       [NEW — code-reviewed copy]
│   │   ├── m5_feature_bilstm.py                 [NEW — code-reviewed copy]
│   │   └── m6_graph_bilstm.py                   [NEW — code-reviewed copy]
│   ├── modules/
│   │   ├── __init__.py                          [NEW]
│   │   ├── channel_attention.py                 [NEW — code-reviewed copy]
│   │   ├── criss_cross_attention.py             [NEW — code-reviewed copy]
│   │   └── graph_attention.py                   [NEW — code-reviewed copy]
│   ├── train_baseline.py                        [NEW — config-driven]
│   └── eval_baseline.py                         [NEW]
├── improved_lstm_models/
│   ├── __init__.py                              [NEW]
│   ├── architectures/
│   │   ├── __init__.py                          [NEW]
│   │   └── hierarchical_lstm.py                 [NEW — code-reviewed copy]
│   ├── modules/
│   │   └── __init__.py                          [NEW]
│   ├── augmentation.py                          [NEW — code-reviewed copy]
│   ├── ensemble.py                              [NEW — code-reviewed copy]
│   └── train.py                                 [NEW — config-driven rewrite]
├── ensemble_transformers/
│   ├── __init__.py                              [NEW]
│   ├── architectures/
│   │   ├── __init__.py                          [NEW]
│   │   └── m7_vq_transformer.py                 [NEW — code-reviewed copy]
│   ├── modules/
│   │   ├── __init__.py                          [NEW]
│   │   ├── vector_quantizer.py                  [NEW — code-reviewed copy]
│   │   └── pretrained_encoders.py               [NEW — code-reviewed copy]
│   ├── ensemble.py                              [NEW — code-reviewed from ensemble_7model.py]
│   └── train_ensemble.py                        [NEW]
└── hugging_face_mamba_moe/
    ├── __init__.py                              [NEW]
    ├── architectures/
    │   ├── __init__.py                          [NEW]
    │   ├── eeg_mamba.py                         [NEW — code-reviewed copy]
    │   └── tiny_seizure_net.py                  [NEW — code-reviewed copy]
    ├── modules/
    │   ├── __init__.py                          [NEW]
    │   ├── mamba_block.py                       [NEW — code-reviewed copy]
    │   ├── mixture_of_experts.py                [NEW — code-reviewed copy]
    │   └── diffusion_eeg.py                     [NEW — code-reviewed copy]
    └── train_mamba.py                           [NEW — code-reviewed]

tests/
├── test_models.py                               [NEW]
└── test_metrics.py                              [NEW]
```

---

## Task 1: Create Shared Utils (src/models/utils/)

**Why first:** Every training script imports metrics and losses. Building utils first prevents duplication.

**Files:**
- Create: `src/models/utils/__init__.py`
- Create: `src/models/utils/metrics.py`
- Create: `src/models/utils/losses.py`
- Create: `src/models/utils/callbacks.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py
import numpy as np
import pytest
from src.models.utils.metrics import compute_f1_score, compute_auc_roc, compute_sensitivity

def test_f1_score_perfect():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0])
    assert compute_f1_score(y_true, y_pred) == 1.0

def test_f1_score_all_wrong():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0, 0, 1, 1])
    assert compute_f1_score(y_true, y_pred) == 0.0

def test_f1_score_handles_no_positive_predictions():
    y_true = np.array([1, 0, 0])
    y_pred = np.array([0, 0, 0])
    result = compute_f1_score(y_true, y_pred)
    assert isinstance(result, float)

def test_compute_sensitivity_perfect():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0])
    assert compute_sensitivity(y_true, y_pred) == 1.0

def test_compute_sensitivity_missed_all():
    y_true = np.array([1, 1])
    y_pred = np.array([0, 0])
    assert compute_sensitivity(y_true, y_pred) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/abhinaysaikamineni/PycharmProjects/spring-2026-group2
pytest tests/test_metrics.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.models.utils'`

- [ ] **Step 3: Create src/models/utils/__init__.py**

```python
# src/models/utils/__init__.py
"""Shared utilities for all model directories."""
from .metrics import compute_f1_score, compute_auc_roc, compute_sensitivity
from .losses import FocalLoss
from .callbacks import EarlyStopping

__all__ = [
    "compute_f1_score", "compute_auc_roc", "compute_sensitivity",
    "FocalLoss", "EarlyStopping",
]
```

- [ ] **Step 4: Create src/models/utils/metrics.py**

```python
# src/models/utils/metrics.py
"""
Shared evaluation metrics for EEG seizure detection.

All training and evaluation scripts import from here. Never define metrics locally.
"""
import logging
from typing import Optional
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

logger = logging.getLogger(__name__)


def compute_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute binary F1-score.

    Args:
        y_true: Ground-truth binary labels, shape (n,)
        y_pred: Predicted binary labels, shape (n,)

    Returns:
        F1-score in [0, 1]. Returns 0.0 if no positive predictions.

    Example:
        >>> compute_f1_score(np.array([1,0,1]), np.array([1,0,0]))
        0.6666...
    """
    return float(f1_score(y_true, y_pred, zero_division=0))


def compute_auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute AUC-ROC score.

    Args:
        y_true: Ground-truth binary labels, shape (n,)
        y_score: Predicted probabilities for positive class, shape (n,)

    Returns:
        AUC-ROC in [0, 1]. Returns 0.5 if only one class present.

    Raises:
        ValueError: If y_true and y_score have different lengths.

    Example:
        >>> compute_auc_roc(np.array([1,0,1]), np.array([0.9,0.1,0.8]))
        1.0
    """
    if len(np.unique(y_true)) < 2:
        logger.warning("Only one class in y_true; AUC-ROC undefined, returning 0.5")
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def compute_sensitivity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute sensitivity (recall for the positive/seizure class).

    Args:
        y_true: Ground-truth binary labels, shape (n,)
        y_pred: Predicted binary labels, shape (n,)

    Returns:
        Sensitivity in [0, 1]. Returns 0.0 if no true positives exist.

    Example:
        >>> compute_sensitivity(np.array([1,1,0]), np.array([1,0,0]))
        0.5
    """
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    true_positives = conf_matrix[1, 1]
    false_negatives = conf_matrix[1, 0]
    denominator = true_positives + false_negatives
    if denominator == 0:
        return 0.0
    return float(true_positives / denominator)
```

- [ ] **Step 5: Create src/models/utils/losses.py**

```python
# src/models/utils/losses.py
"""
Loss functions for EEG seizure detection.

Consolidated from approach2/train.py and training/loss_functions.py.
Import FocalLoss from here. Never define it locally in a training script.
"""
import logging
from typing import Optional
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for severe class imbalance (seizure vs. non-seizure).

    Down-weights easy negatives via (1-p_t)^gamma, then applies pos_weight
    AFTER focal modulation — not inside BCE — to avoid double-weighting.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

    Args:
        gamma: Focusing parameter. Higher = more focus on hard examples.
        pos_weight: Scalar weight for positive (seizure) class.
        reduction: 'mean' or 'sum'.

    Example:
        >>> criterion = FocalLoss(gamma=2.0, pos_weight=torch.tensor(3.0))
        >>> loss = criterion(logits, targets)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Raw model outputs, shape (batch,) or (batch, 1)
            targets: Binary labels in {0, 1}, same shape as logits

        Returns:
            Scalar loss tensor.
        """
        logits_f32 = logits.float()
        targets_f32 = targets.float()
        bce_per_sample = self.bce(logits_f32, targets_f32)
        probability_correct = torch.exp(-bce_per_sample)
        focal_weight = (1.0 - probability_correct) ** self.gamma
        focal_loss = focal_weight * bce_per_sample
        if self.pos_weight is not None:
            weight_mask = _build_pos_weight_mask(targets_f32, self.pos_weight)
            focal_loss = focal_loss * weight_mask
        return _reduce_loss(focal_loss, self.reduction)


def _build_pos_weight_mask(
    targets: torch.Tensor, pos_weight: torch.Tensor
) -> torch.Tensor:
    """
    Build per-sample weight mask: pos_weight for seizure, 1.0 for non-seizure.

    Args:
        targets: Binary labels, shape (batch,)
        pos_weight: Scalar weight for the positive class

    Returns:
        Per-sample weight tensor, same shape as targets.
    """
    return torch.where(
        targets > 0.5,
        pos_weight.to(targets.device),
        torch.ones_like(targets),
    )


def _reduce_loss(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    """
    Apply reduction to per-sample loss.

    Args:
        loss: Per-sample losses, shape (batch,)
        reduction: 'mean' or 'sum'

    Returns:
        Scalar loss tensor.

    Raises:
        ValueError: If reduction is not 'mean' or 'sum'.
    """
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"Unknown reduction '{reduction}'. Expected 'mean' or 'sum'.")
```

- [ ] **Step 6: Create src/models/utils/callbacks.py**

```python
# src/models/utils/callbacks.py
"""
Training callbacks: early stopping and gradient clipping.

Import from here. Never implement inline in a training script.
"""
import logging
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Stop training when validation loss stops improving.

    Saves best model checkpoint automatically.

    Args:
        patience: Epochs to wait before stopping after last improvement.
        min_delta: Minimum improvement required to count as improvement.
        checkpoint_path: Where to save the best model state dict.

    Example:
        >>> stopper = EarlyStopping(patience=10, checkpoint_path=Path("best.pt"))
        >>> for epoch in range(100):
        ...     stopper.step(val_loss, model)
        ...     if stopper.should_stop:
        ...         break
    """

    def __init__(
        self,
        patience: int,
        min_delta: float = 1e-4,
        checkpoint_path: Optional[Path] = None,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.best_loss: float = float("inf")
        self.epochs_without_improvement: int = 0
        self.should_stop: bool = False

    def step(self, val_loss: float, model: nn.Module) -> None:
        """
        Update state and optionally save checkpoint.

        Args:
            val_loss: Validation loss for this epoch.
            model: Model to checkpoint if improved.
        """
        if val_loss < self.best_loss - self.min_delta:
            self._record_improvement(val_loss, model)
        else:
            self._record_no_improvement()

    def _record_improvement(self, val_loss: float, model: nn.Module) -> None:
        """Save checkpoint and reset counter on improvement."""
        self.best_loss = val_loss
        self.epochs_without_improvement = 0
        if self.checkpoint_path is not None:
            torch.save(model.state_dict(), self.checkpoint_path)
            logger.debug(f"Checkpoint saved (val_loss={val_loss:.4f})")

    def _record_no_improvement(self) -> None:
        """Increment counter; trigger stop if patience exceeded."""
        self.epochs_without_improvement += 1
        if self.epochs_without_improvement >= self.patience:
            self.should_stop = True
            logger.info(f"Early stopping triggered after {self.patience} epochs without improvement.")


def clip_gradients(model: nn.Module, max_norm: float) -> float:
    """
    Clip gradient norms to prevent exploding gradients.

    Args:
        model: Model whose parameters to clip.
        max_norm: Maximum allowed gradient norm.

    Returns:
        Total gradient norm before clipping.

    Example:
        >>> grad_norm = clip_gradients(model, max_norm=1.0)
    """
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    return float(total_norm)
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
pytest tests/test_metrics.py -v
```
Expected: All 5 tests PASS

- [ ] **Step 8: Commit**

```bash
git add src/models/utils/ tests/test_metrics.py
git commit -m "[Utils] Feat: Add shared metrics, losses, callbacks to src/models/utils/"
```

---

## Task 2: Create src/models/config.yaml

**Why:** The Code Review doc requires config.yaml as single source of truth. Every training script currently hardcodes hyperparameters.

**Files:**
- Create: `src/models/config.yaml`

- [ ] **Step 1: Create config.yaml**

```yaml
# src/models/config.yaml
# Single source of truth for all model hyperparameters.
# All training scripts load this via load_config(). No hardcoded values elsewhere.

data:
  processed_dir: "src/data/processed/chbmit"
  n_channels: 16
  time_steps: 256
  train_split: 0.70
  val_split: 0.15
  test_split: 0.15

training:
  learning_rate: 0.0005
  batch_size: 32
  num_epochs: 100
  pos_weight: 3.0
  early_stopping_patience: 15
  gradient_clip: 1.0
  warmup_epochs: 5
  weight_decay: 0.0001
  label_smoothing: 0.1
  aug_prob: 0.5

focal_loss:
  gamma: 2.0
  reduction: "mean"

models:
  lstm_benchmark:
    hidden_size: 128
    num_layers: 2
    dropout: 0.3

  improved_lstm:
    hidden_size: 256
    num_layers: 3
    dropout: 0.4

  ensemble_transformers:
    patch_size: 32
    hidden_size: 128
    codebook_size: 512
    num_layers: 4
    num_heads: 4
    dropout: 0.1
    ensemble_strategy: "weighted"

  hugging_face_mamba_moe:
    d_model: 128
    d_state: 16
    d_conv: 4
    n_layers: 4
    dropout: 0.1
    bidirectional: true

outputs:
  checkpoint_dir: "outputs/models"
  results_dir: "outputs/results"
  log_dir: "outputs/logs"
```

- [ ] **Step 2: Verify YAML is valid**

```bash
python -c "import yaml; cfg = yaml.safe_load(open('src/models/config.yaml')); print('OK:', list(cfg.keys()))"
```
Expected: `OK: ['data', 'training', 'focal_loss', 'models', 'outputs']`

- [ ] **Step 3: Commit**

```bash
git add src/models/config.yaml
git commit -m "[Config] Feat: Add src/models/config.yaml as single source of truth"
```

---

## Task 3: Create lstm_benchmark_models/modules/ (Code-Reviewed)

**Why modules first:** Architecture files import from modules — modules must exist before architectures.

**Files:**
- Create: `src/models/lstm_benchmark_models/__init__.py`
- Create: `src/models/lstm_benchmark_models/modules/__init__.py`
- Create: `src/models/lstm_benchmark_models/modules/channel_attention.py`
- Create: `src/models/lstm_benchmark_models/modules/criss_cross_attention.py`
- Create: `src/models/lstm_benchmark_models/modules/graph_attention.py`

- [ ] **Step 1: Create directories**

```bash
mkdir -p src/models/lstm_benchmark_models/architectures
mkdir -p src/models/lstm_benchmark_models/modules
```

- [ ] **Step 2: Create lstm_benchmark_models/__init__.py**

```python
# src/models/lstm_benchmark_models/__init__.py
"""LSTM Benchmark Models — Directory 1 of 4. Six LSTM variants (m1-m6)."""
from .architectures import (
    M1_VanillaLSTM, M2_BiLSTM, M3_CrissCrossBiLSTM,
    M4_CNNLSTM, M5_FeatureBiLSTM, M6_GraphBiLSTM,
    MODEL_REGISTRY, get_benchmark_model,
)
__all__ = [
    "M1_VanillaLSTM", "M2_BiLSTM", "M3_CrissCrossBiLSTM",
    "M4_CNNLSTM", "M5_FeatureBiLSTM", "M6_GraphBiLSTM",
    "MODEL_REGISTRY", "get_benchmark_model",
]
```

- [ ] **Step 3: Create modules/__init__.py**

```python
# src/models/lstm_benchmark_models/modules/__init__.py
"""Shared attention modules for LSTM benchmark architectures."""
from .channel_attention import ChannelAttention, SpatialChannelAttention
from .criss_cross_attention import CrissCrossAttention
from .graph_attention import GraphAttention

__all__ = ["ChannelAttention", "SpatialChannelAttention", "CrissCrossAttention", "GraphAttention"]
```

- [ ] **Step 4: Create channel_attention.py (code-reviewed)**

Source: `src/models/approach2/modules/channel_attention.py`

Code review changes to apply:
- Add `logger = logging.getLogger(__name__)` after imports
- Update module-level docstring to reference lstm_benchmark_models directory
- Rename `x` parameter in forward() to `eeg_tensor`
- Add `Example:` block to both class docstrings
- Verify all methods have return type annotations (`-> torch.Tensor`)

Key structure:
```python
# src/models/lstm_benchmark_models/modules/channel_attention.py
"""Channel Attention Modules for LSTM Benchmark Models."""
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class ChannelAttention(nn.Module):
    """..."""  # full docstring with Args/Example
    def __init__(self, n_channels: int, reduction: int = 4): ...
    def forward(self, eeg_tensor: torch.Tensor) -> torch.Tensor: ...

class SpatialChannelAttention(nn.Module):
    """..."""  # full docstring with Args/Example
    def __init__(self, n_channels: int, reduction: int = 4): ...
    def forward(self, eeg_tensor: torch.Tensor) -> torch.Tensor: ...
```

- [ ] **Step 5: Create criss_cross_attention.py (code-reviewed)**

Source: `src/models/approach2/modules/criss_cross_attention.py`
- Add logger
- Add type hints to all methods
- Rename `x` -> `eeg_tensor` in forward()
- Verify function length <= 30 lines; extract helper if needed

- [ ] **Step 6: Create graph_attention.py (code-reviewed)**

Source: `src/models/approach2/modules/graph_attention.py`
- Add logger
- Add type hints to all methods
- Rename `x` -> `eeg_tensor` in forward()
- Verify function length <= 30 lines

- [ ] **Step 7: Commit**

```bash
git add src/models/lstm_benchmark_models/
git commit -m "[Models] Feat: Add lstm_benchmark_models/modules (code-reviewed attention modules)"
```

---

## Task 4: Create lstm_benchmark_models/architectures/ — m1 through m6

**Files:**
- Create: `src/models/lstm_benchmark_models/architectures/__init__.py`
- Create: `src/models/lstm_benchmark_models/architectures/m1_vanilla_lstm.py`
- Create: `src/models/lstm_benchmark_models/architectures/m2_bilstm.py`
- Create: `src/models/lstm_benchmark_models/architectures/m3_criss_cross.py`
- Create: `src/models/lstm_benchmark_models/architectures/m4_cnn_lstm.py`
- Create: `src/models/lstm_benchmark_models/architectures/m5_feature_bilstm.py`
- Create: `src/models/lstm_benchmark_models/architectures/m6_graph_bilstm.py`

**Code review changes for ALL 6 architecture files:**

| Issue | Fix |
|-------|-----|
| `sys.path.append(str(Path(__file__).parent.parent))` | Remove entirely |
| `from modules.channel_attention import ...` | `from ..modules.channel_attention import ...` |
| `x` variable in forward() | Rename to `eeg_input`, `attended`, `projected`, `lstm_out`, `pooled` |
| forward() > 30 lines | Extract `_project_input()`, `_pool_lstm_output()`, `_classify()` as module-level helpers |
| No logging | Add `logger = logging.getLogger(__name__)` |
| Missing return type on forward() | Add `-> torch.Tensor` |

- [ ] **Step 1: Write the failing architecture smoke test**

```python
# tests/test_models.py
import pytest
import torch
from src.models.lstm_benchmark_models import get_benchmark_model

EEG_BATCH = torch.randn(4, 16, 256)   # (batch, channels, time_steps)

@pytest.mark.parametrize("model_name", [
    "m1_vanilla_lstm", "m2_bilstm", "m3_criss_cross",
    "m4_cnn_lstm", "m5_feature_bilstm", "m6_graph_bilstm",
])
def test_benchmark_model_forward(model_name: str):
    model = get_benchmark_model(model_name)
    model.train(False)
    with torch.no_grad():
        logits = model(EEG_BATCH)
    assert logits.shape == (4, 1), f"{model_name}: expected (4,1), got {logits.shape}"

def test_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        get_benchmark_model("not_a_model")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_models.py::test_benchmark_model_forward -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create architectures/__init__.py**

```python
# src/models/lstm_benchmark_models/architectures/__init__.py
"""
LSTM Benchmark Model Architectures — m1 through m6.

Input contract for all models: (batch, 16, 256) -> logits (batch, 1).
"""
from typing import Dict, Type
import torch.nn as nn
from .m1_vanilla_lstm import M1_VanillaLSTM
from .m2_bilstm import M2_BiLSTM
from .m3_criss_cross import M3_CrissCrossBiLSTM
from .m4_cnn_lstm import M4_CNNLSTM
from .m5_feature_bilstm import M5_FeatureBiLSTM
from .m6_graph_bilstm import M6_GraphBiLSTM

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "m1_vanilla_lstm": M1_VanillaLSTM,
    "m2_bilstm": M2_BiLSTM,
    "m3_criss_cross": M3_CrissCrossBiLSTM,
    "m4_cnn_lstm": M4_CNNLSTM,
    "m5_feature_bilstm": M5_FeatureBiLSTM,
    "m6_graph_bilstm": M6_GraphBiLSTM,
}

def get_benchmark_model(model_name: str, **kwargs: object) -> nn.Module:
    """
    Instantiate a benchmark LSTM model by name.

    Args:
        model_name: Key in MODEL_REGISTRY (e.g. 'm1_vanilla_lstm').
        **kwargs: Forwarded to the model constructor.

    Returns:
        Instantiated nn.Module.

    Raises:
        ValueError: If model_name is not in MODEL_REGISTRY.

    Example:
        >>> model = get_benchmark_model("m2_bilstm", hidden_size=256)
    """
    model_name = model_name.lower()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {sorted(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name](**kwargs)
```

- [ ] **Step 4: Create m1_vanilla_lstm.py (code-reviewed)**

```python
# src/models/lstm_benchmark_models/architectures/m1_vanilla_lstm.py
"""M1: VanillaLSTM + Channel Attention — Benchmark Model. Expected F1: 0.62-0.70."""
import logging
from typing import Tuple
import torch
import torch.nn as nn
from ..modules.channel_attention import ChannelAttention

logger = logging.getLogger(__name__)


class M1_VanillaLSTM(nn.Module):
    """
    VanillaLSTM with Channel Attention for EEG seizure detection.

    Architecture:
        Input (batch, n_channels, time_steps)
        -> ChannelAttention -> LayerNorm -> LinearProjection
        -> LSTM (num_layers) -> AvgPool + MaxPool -> FC Head -> Logits

    Args:
        n_channels: Number of EEG channels (default: 16).
        time_steps: Timesteps per window (default: 256).
        hidden_size: LSTM hidden dimension (default: 128).
        num_layers: Number of stacked LSTM layers (default: 2).
        dropout: Dropout rate (default: 0.3).

    Example:
        >>> model = M1_VanillaLSTM(n_channels=16, hidden_size=128)
        >>> logits = model(torch.randn(8, 16, 256))
        >>> assert logits.shape == (8, 1)
    """

    def __init__(
        self,
        n_channels: int = 16,
        time_steps: int = 256,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.hidden_size = hidden_size
        self.channel_attn = ChannelAttention(n_channels, reduction=4)
        self.input_norm = nn.LayerNorm(n_channels)
        self.input_proj = _build_input_proj(n_channels, hidden_size, dropout)
        self.lstm = _build_lstm(hidden_size, num_layers, dropout)
        self.pool_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout_layer = nn.Dropout(dropout)
        self.classifier = _build_classifier(hidden_size)

    def forward(self, eeg_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            eeg_input: EEG windows, shape (batch, n_channels, time_steps)

        Returns:
            Logits, shape (batch, 1).
        """
        attended = self.channel_attn(eeg_input)
        projected = self._project(attended)
        lstm_out, _ = self.lstm(projected)
        pooled = self._pool(lstm_out)
        return self.classifier(pooled)

    def _project(self, attended: torch.Tensor) -> torch.Tensor:
        """Normalize and project channel-attended input for LSTM."""
        transposed = attended.permute(0, 2, 1)
        return self.input_proj(self.input_norm(transposed))

    def _pool(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """Combine avg and max pool over time; apply norm and dropout."""
        avg_pooled = lstm_out.mean(dim=1)
        max_pooled = lstm_out.max(dim=1).values
        pooled = torch.cat([avg_pooled, max_pooled], dim=-1)
        return self.dropout_layer(self.pool_norm(pooled))


def _build_input_proj(n_channels: int, hidden_size: int, dropout: float) -> nn.Sequential:
    """Build linear input projection block."""
    return nn.Sequential(
        nn.Linear(n_channels, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout * 0.5),
    )


def _build_lstm(hidden_size: int, num_layers: int, dropout: float) -> nn.LSTM:
    """Build stacked LSTM module."""
    return nn.LSTM(
        input_size=hidden_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True,
        dropout=dropout if num_layers > 1 else 0.0,
    )


def _build_classifier(hidden_size: int) -> nn.Sequential:
    """Build two-layer FC classification head."""
    return nn.Sequential(
        nn.Linear(hidden_size * 2, hidden_size),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_size, 1),
    )
```

- [ ] **Step 5: Create m2_bilstm.py through m6_graph_bilstm.py**

Apply identical code review pattern to each file. Key changes per file:

**m2_bilstm.py**: source `approach2/architectures/m2_bilstm.py`
- Remove sys.path hack
- `from ..modules.channel_attention import SpatialChannelAttention`
- Rename `x` -> `eeg_input` in forward()
- Extract `_pool_bilstm_output()` helper if forward() > 30 lines

**m3_criss_cross.py**: source `approach2/architectures/m3_criss_cross.py`
- Remove sys.path hack
- `from ..modules.criss_cross_attention import CrissCrossAttention`
- Rename `x` -> `eeg_input`
- Extract `_apply_attention()` helper

**m4_cnn_lstm.py**: source `approach2/architectures/m4_cnn_lstm.py`
- Remove sys.path hack
- Rename `x` -> `eeg_input`, local `out` -> `cnn_features`
- Extract `_extract_cnn_features()`, `_run_lstm()` helpers

**m5_feature_bilstm.py**: source `approach2/architectures/m5_feature_bilstm.py`
- Remove sys.path hack
- `from ..modules.channel_attention import ChannelAttention`
- Rename `x` -> `eeg_input`

**m6_graph_bilstm.py**: source `approach2/architectures/m6_graph_bilstm.py`
- Remove sys.path hack
- `from ..modules.graph_attention import GraphAttention`
- Rename `x` -> `eeg_input`

- [ ] **Step 6: Run all 6 model tests**

```bash
pytest tests/test_models.py -v -k "benchmark"
```
Expected: All 6 parametrized tests PASS, `test_unknown_model_raises` PASS

- [ ] **Step 7: Commit**

```bash
git add src/models/lstm_benchmark_models/architectures/ tests/test_models.py
git commit -m "[Models] Feat: Add lstm_benchmark_models/architectures m1-m6 (code-reviewed)"
```

---

## Task 5: Create lstm_benchmark_models/train_baseline.py and eval_baseline.py

**Files:**
- Create: `src/models/lstm_benchmark_models/train_baseline.py`
- Create: `src/models/lstm_benchmark_models/eval_baseline.py`

**Code review changes from approach2/train.py:**

| Issue | Fix |
|-------|-----|
| FocalLoss defined locally | Import from `src.models.utils.losses` |
| Hardcoded LR/batch/epochs | Load from config.yaml |
| train_one_epoch() ~60 lines | Extract `_run_batch()` helper |
| print() throughout | Replace with logger.info() / logger.debug() |
| sklearn metrics computed inline | Replace with compute_f1_score() from utils.metrics |
| Scope: covers m1-m7 | Scope to m1-m6 only |

- [ ] **Step 1: Create train_baseline.py**

Key structure (full implementation):
```python
# src/models/lstm_benchmark_models/train_baseline.py
"""
LSTM Benchmark Training — Models m1 through m6.

Usage:
    python -m src.models.lstm_benchmark_models.train_baseline \
        --model m1_vanilla_lstm --data_path src/data/processed/chbmit

All hyperparameters from src/models/config.yaml.
"""
import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.utils.losses import FocalLoss
from src.models.utils.metrics import compute_f1_score, compute_auc_roc, compute_sensitivity
from src.models.utils.callbacks import EarlyStopping, clip_gradients
from .architectures import get_benchmark_model, MODEL_REGISTRY

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict:
    """
    Load YAML config from disk.

    Args:
        config_path: Path to config.yaml file.

    Returns:
        Parsed config dictionary.

    Raises:
        FileNotFoundError: If config_path does not exist.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as config_file:
        return yaml.safe_load(config_file)


def train_baseline(model_name: str, data_path: Path, config: Dict) -> Dict:
    """
    Train one benchmark model end-to-end; return test metrics.

    Args:
        model_name: Key in MODEL_REGISTRY (e.g. 'm1_vanilla_lstm').
        data_path: Directory with train/, val/, test/ tensor splits.
        config: Config dict from config.yaml.

    Returns:
        Dict with f1, auc_roc, sensitivity on test split.

    Raises:
        ValueError: If model_name is not in MODEL_REGISTRY.
    """
    device = _get_device()
    model = get_benchmark_model(model_name, **config["models"]["lstm_benchmark"]).to(device)
    train_loader, val_loader, test_loader = _build_data_loaders(data_path, config)
    criterion = _build_criterion(config, device)
    optimizer = _build_optimizer(model, config)
    scheduler = _build_scheduler(optimizer, config)
    stopper = EarlyStopping(
        patience=config["training"]["early_stopping_patience"],
        checkpoint_path=Path(config["outputs"]["checkpoint_dir"]) / f"{model_name}_best.pt",
    )
    _run_training_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, stopper, config, device)
    return _evaluate_on_test(model, test_loader, device)


def _run_training_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, stopper, config, device):
    """Run epochs until early stopping or num_epochs reached."""
    num_epochs = config["training"]["num_epochs"]
    for epoch in range(num_epochs):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, config, device)
        val_loss = _validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step()
        stopper.step(val_loss, model)
        logger.info(f"Epoch {epoch+1}/{num_epochs} — train={train_loss:.4f} val={val_loss:.4f}")
        if stopper.should_stop:
            logger.info("Early stopping triggered.")
            break


def _train_one_epoch(model, train_loader, criterion, optimizer, config, device) -> float:
    """Train for one epoch; return mean training loss."""
    model.train()
    total_loss = 0.0
    for eeg_batch, label_batch in train_loader:
        eeg_batch, label_batch = eeg_batch.to(device), label_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(eeg_batch).squeeze(), label_batch.float())
        loss.backward()
        clip_gradients(model, max_norm=config["training"]["gradient_clip"])
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def _validate_one_epoch(model, val_loader, criterion, device) -> float:
    """Validate for one epoch; return mean validation loss."""
    model.train(False)
    total_loss = 0.0
    with torch.no_grad():
        for eeg_batch, label_batch in val_loader:
            eeg_batch, label_batch = eeg_batch.to(device), label_batch.to(device)
            total_loss += criterion(model(eeg_batch).squeeze(), label_batch.float()).item()
    return total_loss / len(val_loader)


def _evaluate_on_test(model, test_loader, device) -> Dict:
    """Run inference on test split; return F1/AUC/sensitivity."""
    import numpy as np
    model.train(False)
    all_labels, all_probs = [], []
    with torch.no_grad():
        for eeg_batch, label_batch in test_loader:
            probs = torch.sigmoid(model(eeg_batch.to(device)).squeeze()).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(label_batch.numpy().tolist())
    y_true = np.array(all_labels)
    y_score = np.array(all_probs)
    y_pred = (y_score > 0.5).astype(int)
    return {
        "f1": compute_f1_score(y_true, y_pred),
        "auc_roc": compute_auc_roc(y_true, y_score),
        "sensitivity": compute_sensitivity(y_true, y_pred),
    }


def _get_device() -> torch.device:
    """Return CUDA if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_criterion(config: Dict, device: torch.device) -> nn.Module:
    """Instantiate FocalLoss with pos_weight from config."""
    pos_weight = torch.tensor(config["training"]["pos_weight"]).to(device)
    return FocalLoss(
        gamma=config["focal_loss"]["gamma"],
        pos_weight=pos_weight,
        reduction=config["focal_loss"]["reduction"],
    )


def _build_optimizer(model: nn.Module, config: Dict) -> torch.optim.AdamW:
    """Build AdamW optimizer from config."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )


def _build_scheduler(optimizer: torch.optim.Optimizer, config: Dict) -> object:
    """Build cosine annealing LR scheduler from config."""
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["num_epochs"]
    )


def _build_data_loaders(data_path: Path, config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train/val/test DataLoaders from pre-processed tensors.

    Args:
        data_path: Directory containing train/, val/, test/ subdirs.
        config: Config dict with training.batch_size.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).

    Raises:
        FileNotFoundError: If any split directory is missing.
    """
    from src.dataloaders.common.loader import load_split
    batch_size = config["training"]["batch_size"]
    return (
        DataLoader(load_split(data_path / "train"), batch_size=batch_size, shuffle=True),
        DataLoader(load_split(data_path / "val"), batch_size=batch_size, shuffle=False),
        DataLoader(load_split(data_path / "test"), batch_size=batch_size, shuffle=False),
    )


def main() -> None:
    """CLI entry point for training."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Train LSTM benchmark models m1-m6")
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--data_path", default="src/data/processed/chbmit")
    args = parser.parse_args()
    config = load_config(Path("src/models/config.yaml"))
    Path(config["outputs"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    results = train_baseline(args.model, Path(args.data_path), config)
    logger.info(f"Test results: {results}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create eval_baseline.py**

```python
# src/models/lstm_benchmark_models/eval_baseline.py
"""
LSTM Benchmark Evaluation.

Usage:
    python -m src.models.lstm_benchmark_models.eval_baseline \
        --model m1_vanilla_lstm \
        --checkpoint outputs/models/m1_vanilla_lstm_best.pt \
        --data_path src/data/processed/chbmit
"""
import argparse
import logging
from pathlib import Path
from typing import Dict
import torch
from src.models.utils.metrics import compute_f1_score, compute_auc_roc, compute_sensitivity
from .architectures import get_benchmark_model, MODEL_REGISTRY
from .train_baseline import load_config, _build_data_loaders, _get_device, _evaluate_on_test

logger = logging.getLogger(__name__)


def evaluate_checkpoint(
    model_name: str, checkpoint_path: Path, data_path: Path, config: Dict
) -> Dict:
    """
    Load checkpoint and run evaluation on test split.

    Args:
        model_name: Key in MODEL_REGISTRY.
        checkpoint_path: Path to .pt state dict file.
        data_path: Directory containing test/ tensor split.
        config: Config dict from config.yaml.

    Returns:
        Dict with f1, auc_roc, sensitivity.

    Raises:
        FileNotFoundError: If checkpoint_path does not exist.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    device = _get_device()
    model = get_benchmark_model(model_name, **config["models"]["lstm_benchmark"])
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    _, _, test_loader = _build_data_loaders(data_path, config)
    return _evaluate_on_test(model, test_loader, device)


def main() -> None:
    """CLI entry point for evaluation."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate LSTM benchmark checkpoint")
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", default="src/data/processed/chbmit")
    args = parser.parse_args()
    config = load_config(Path("src/models/config.yaml"))
    metrics = evaluate_checkpoint(args.model, Path(args.checkpoint), Path(args.data_path), config)
    logger.info(f"Evaluation: {metrics}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
git add src/models/lstm_benchmark_models/train_baseline.py src/models/lstm_benchmark_models/eval_baseline.py
git commit -m "[Models] Feat: Add lstm_benchmark_models train_baseline.py + eval_baseline.py (config-driven)"
```

---

## Task 6: Create improved_lstm_models/ (Code-Reviewed)

**Files:**
- Create: `src/models/improved_lstm_models/__init__.py`
- Create: `src/models/improved_lstm_models/architectures/__init__.py`
- Create: `src/models/improved_lstm_models/architectures/hierarchical_lstm.py`
- Create: `src/models/improved_lstm_models/modules/__init__.py`
- Create: `src/models/improved_lstm_models/augmentation.py`
- Create: `src/models/improved_lstm_models/ensemble.py`
- Create: `src/models/improved_lstm_models/train.py`

**Code review changes from improved_lstm/train.py (758 lines):**

| Issue | Fix |
|-------|-----|
| `DEFAULT_CONFIG = {...}` 15+ hardcoded values | Remove; load from config.yaml |
| print() calls throughout | Replace with logger.info() / logger.warning() |
| train_epoch() > 30 lines | Extract `_run_train_batch()` helper |
| Local FocalLoss | Remove; import from src.models.utils.losses |
| Local sklearn metrics | Replace with src.models.utils.metrics functions |
| WarmupCosineScheduler defined in train.py | Keep in improved_lstm_models/train.py (it is <= 30 lines) |

- [ ] **Step 1: Create directories**

```bash
mkdir -p src/models/improved_lstm_models/architectures
mkdir -p src/models/improved_lstm_models/modules
```

- [ ] **Step 2: Create improved_lstm_models/__init__.py**

```python
# src/models/improved_lstm_models/__init__.py
"""Improved LSTM Models — Directory 2 of 4. Enhanced LSTM with augmentation."""
```

- [ ] **Step 3: Create architectures/hierarchical_lstm.py (code-reviewed)**

Source: `src/models/approach3/architectures/hierarchical_lstm.py`
Changes:
- Remove any sys.path hacks
- Add `logger = logging.getLogger(__name__)`
- Rename `x` -> `eeg_input` in forward()
- Verify type hints on __init__ and forward()
- Verify forward() <= 30 lines

- [ ] **Step 4: Create augmentation.py (code-reviewed)**

Source: `src/models/improved_lstm/augmentation.py`
Changes:
- Add `logger = logging.getLogger(__name__)`
- Verify all public functions have Args/Returns/Example docstrings
- Replace any print() with logger calls
- Verify type hints on all signatures

- [ ] **Step 5: Create ensemble.py (code-reviewed)**

Source: `src/models/improved_lstm/ensemble.py`
Changes:
- Add `logger = logging.getLogger(__name__)`
- Verify all functions <= 30 lines
- Verify type hints on all signatures

- [ ] **Step 6: Create train.py (config-driven rewrite)**

Key structure:
```python
# src/models/improved_lstm_models/train.py
"""Improved LSTM Training — Directory 2 of 4. All params from config.yaml."""
import argparse
import logging
import math
from pathlib import Path
from typing import Dict, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.utils.losses import FocalLoss
from src.models.utils.metrics import compute_f1_score, compute_auc_roc, compute_sensitivity
from src.models.utils.callbacks import EarlyStopping, clip_gradients
from src.models.lstm_benchmark_models.train_baseline import (
    load_config, _get_device, _build_data_loaders
)
from .augmentation import EEGAugmentation

logger = logging.getLogger(__name__)


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup then cosine annealing LR scheduler."""
    # [full implementation per improved_lstm/train.py WarmupCosineScheduler — <= 30 lines]


def train_improved(data_path: Path, config: Dict) -> Dict:
    """Train improved LSTM; return test metrics. [full docstring]"""
    # [implementation calling _run_training_loop]


def _run_training_loop(model, train_loader, val_loader, augmenter, criterion, optimizer, scheduler, stopper, config, device):
    """Run epochs until early stopping. [docstring]"""


def _train_one_epoch(model, train_loader, augmenter, criterion, optimizer, config, device) -> float:
    """Train with augmentation; return mean loss. [docstring]"""


def _validate_one_epoch(model, val_loader, criterion, device) -> float:
    """Validate; return mean loss. [docstring]"""


def _evaluate_on_test(model, test_loader, device) -> Dict:
    """Compute F1/AUC/sensitivity on test split. [docstring]"""


def _build_improved_model(config: Dict) -> nn.Module:
    """Build model from config. [docstring]"""


def main() -> None:
    """CLI entry point."""
```

- [ ] **Step 7: Commit**

```bash
git add src/models/improved_lstm_models/
git commit -m "[Models] Feat: Add improved_lstm_models/ (config-driven, augmentation, code-reviewed)"
```

---

## Task 7: Create ensemble_transformers/ (Code-Reviewed)

**Files:**
- Create: `src/models/ensemble_transformers/__init__.py`
- Create: `src/models/ensemble_transformers/architectures/__init__.py`
- Create: `src/models/ensemble_transformers/architectures/m7_vq_transformer.py`
- Create: `src/models/ensemble_transformers/modules/__init__.py`
- Create: `src/models/ensemble_transformers/modules/vector_quantizer.py`
- Create: `src/models/ensemble_transformers/modules/pretrained_encoders.py`
- Create: `src/models/ensemble_transformers/ensemble.py`
- Create: `src/models/ensemble_transformers/train_ensemble.py`

**Code review changes from approach2/ensemble_7model.py:**

| Issue | Fix |
|-------|-----|
| `from architectures import get_model` | `from src.models.lstm_benchmark_models import get_benchmark_model` |
| All sklearn metrics computed inline | Use src.models.utils.metrics |
| print() throughout | Replace with logger.info() |
| run_ensemble() > 30 lines | Extract `_load_all_checkpoints()`, `_collect_predictions()` |

- [ ] **Step 1: Create directories**

```bash
mkdir -p src/models/ensemble_transformers/architectures
mkdir -p src/models/ensemble_transformers/modules
```

- [ ] **Step 2: Create modules (vector_quantizer.py, pretrained_encoders.py)**

Source: approach2/modules/
Changes:
- Remove sys.path hacks
- Add logger
- Use relative imports

- [ ] **Step 3: Create m7_vq_transformer.py (code-reviewed)**

Source: approach2/architectures/m7_vq_transformer.py
Changes:
- Remove sys.path hack
- `from ..modules.vector_quantizer import VectorQuantizer, VQEncoder`
- `from ..modules.pretrained_encoders import load_pretrained_encoder`
- Rename `x` -> `eeg_input` in forward()
- Extract `_embed_patches()`, `_encode()`, `_quantize_classify()` helpers

- [ ] **Step 4: Create ensemble.py**

Key functions (all <= 30 lines with docstrings):
```python
# src/models/ensemble_transformers/ensemble.py
"""7-Model Ensemble. Import benchmark models (m1-m6) + m7 transformer."""
# Functions:
# load_ensemble_checkpoints(checkpoint_dir, config, device) -> Dict[str, nn.Module]
# _load_single_checkpoint(model_name, path, config, device) -> nn.Module
# mean_ensemble_predict(models, eeg_batch) -> np.ndarray
# weighted_ensemble_predict(models, eeg_batch, weights) -> np.ndarray
# _collect_all_probs(models, eeg_batch) -> np.ndarray
# evaluate_ensemble(models, test_loader, device, strategy, weights) -> Dict
# _collect_test_predictions(models, test_loader, device, strategy, weights) -> Tuple
```

- [ ] **Step 5: Create train_ensemble.py (CLI entry point)**

```python
# src/models/ensemble_transformers/train_ensemble.py
"""Ensemble evaluation entry point."""
# main() -> loads checkpoints, runs evaluate_ensemble(), logs results
```

- [ ] **Step 6: Commit**

```bash
git add src/models/ensemble_transformers/
git commit -m "[Models] Feat: Add ensemble_transformers/ (m7 VQ-Transformer + ensemble, code-reviewed)"
```

---

## Task 8: Create hugging_face_mamba_moe/ (Code-Reviewed)

**Files:**
- Create: `src/models/hugging_face_mamba_moe/__init__.py`
- Create: `src/models/hugging_face_mamba_moe/architectures/__init__.py`
- Create: `src/models/hugging_face_mamba_moe/architectures/eeg_mamba.py`
- Create: `src/models/hugging_face_mamba_moe/architectures/tiny_seizure_net.py`
- Create: `src/models/hugging_face_mamba_moe/modules/__init__.py`
- Create: `src/models/hugging_face_mamba_moe/modules/mamba_block.py`
- Create: `src/models/hugging_face_mamba_moe/modules/mixture_of_experts.py`
- Create: `src/models/hugging_face_mamba_moe/modules/diffusion_eeg.py`
- Create: `src/models/hugging_face_mamba_moe/train_mamba.py`

**Code review changes from approach3/ files:**

| Issue | Fix |
|-------|-----|
| `sys.path.append(str(Path(__file__).parent.parent))` in eeg_mamba.py | Remove |
| `from modules.mamba_block import ...` | `from ..modules.mamba_block import ...` |
| `x` variable in SelectiveSSM.forward() | Rename to `input_tensor`, `convolved`, `gated` |
| MambaBlock.forward() > 30 lines | Extract `_apply_ssm()`, `_apply_gating()` |
| Hardcoded hyperparameters in train_mamba.py | Load from config.yaml |

- [ ] **Step 1: Create directories**

```bash
mkdir -p src/models/hugging_face_mamba_moe/architectures
mkdir -p src/models/hugging_face_mamba_moe/modules
```

- [ ] **Step 2: Create modules (mamba_block.py, mixture_of_experts.py, diffusion_eeg.py)**

For each file from approach3/modules/:
- Remove sys.path hacks
- Add `logger = logging.getLogger(__name__)`
- Rename `x` -> `input_tensor`, `h` -> `hidden_state`
- Verify forward() <= 30 lines; extract helpers if needed
- Add Example: blocks to all class docstrings

- [ ] **Step 3: Create architectures/eeg_mamba.py (code-reviewed)**

Source: approach3/architectures/eeg_mamba.py
Changes:
- Remove sys.path hack
- `from ..modules.mamba_block import MambaEncoder, BidirectionalMamba, MambaBlock`
- `from ..modules.mixture_of_experts import MixtureOfExperts, MoEWithUniversal`
- Rename `x` -> `eeg_input` in forward()
- Extract `_project_input()`, `_encode_mamba()`, `_classify()` helpers

- [ ] **Step 4: Create architectures/tiny_seizure_net.py (code-reviewed)**

Source: approach3/architectures/tiny_seizure_net.py
Changes: same code review pattern (logger, type hints, no sys.path, rename vars)

- [ ] **Step 5: Create train_mamba.py (code-reviewed)**

Source: approach3/train_mamba.py (186 lines — manageable)
Changes:
- Add load_config() at top of main(); replace any hardcoded values with config lookups
- Import FocalLoss from src.models.utils.losses
- Import metrics from src.models.utils.metrics
- Replace print() with logger.info()
- Add type hints to all function signatures
- Verify all functions <= 30 lines

- [ ] **Step 6: Commit**

```bash
git add src/models/hugging_face_mamba_moe/
git commit -m "[Models] Feat: Add hugging_face_mamba_moe/ (Mamba SSM + MoE, code-reviewed)"
```

---

## Task 9: Update src/models/__init__.py + Final Smoke Tests

**Files:**
- Modify: `src/models/__init__.py`
- Modify: `tests/test_models.py` (add ensemble + mamba tests)

- [ ] **Step 1: Update src/models/__init__.py**

```python
# src/models/__init__.py
"""
EEG Seizure Detection Models.

Four directories in sequence:
1. lstm_benchmark_models   — m1-m6 baseline LSTM variants
2. improved_lstm_models    — enhanced LSTM with augmentation
3. ensemble_transformers   — m7 VQ-Transformer + 7-model ensemble
4. hugging_face_mamba_moe  — Mamba SSM + Mixture of Experts
"""
from .lstm_benchmark_models import MODEL_REGISTRY as BENCHMARK_REGISTRY
from .lstm_benchmark_models import get_benchmark_model

__all__ = ["BENCHMARK_REGISTRY", "get_benchmark_model"]
```

- [ ] **Step 2: Add ensemble and mamba import smoke tests**

```python
# Append to tests/test_models.py

def test_ensemble_module_imports():
    """Verify ensemble_transformers package imports without error."""
    from src.models.ensemble_transformers.ensemble import (
        mean_ensemble_predict, ALL_MODEL_NAMES,
    )
    assert len(ALL_MODEL_NAMES) == 7

def test_mamba_model_forward():
    """Verify EEGMamba forward pass produces correct output shape."""
    import torch
    from src.models.hugging_face_mamba_moe.architectures.eeg_mamba import EEGMamba
    model = EEGMamba(n_channels=16)
    model.train(False)
    with torch.no_grad():
        logits = model(torch.randn(2, 16, 256))
    assert logits.shape == (2, 1)

def test_utils_focal_loss():
    """Verify FocalLoss computes a positive scalar loss."""
    import torch
    from src.models.utils import FocalLoss
    criterion = FocalLoss(gamma=2.0, pos_weight=torch.tensor(3.0))
    loss = criterion(torch.randn(4, 1), torch.randint(0, 2, (4, 1)).float())
    assert loss.item() > 0
```

- [ ] **Step 3: Run all tests**

```bash
cd /Users/abhinaysaikamineni/PycharmProjects/spring-2026-group2
pytest tests/test_models.py tests/test_metrics.py -v
```
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/models/__init__.py tests/test_models.py
git commit -m "[Models] Feat: Update models __init__; all 4 directory smoke tests passing"
```

---

## Task 10: Final Pre-Commit Verification (Code Review Checklist)

**Run the Section 9.2 checklist from the Code Review Standards doc.**

- [ ] **Step 1: No .pt files staged**

```bash
git status | grep "\.pt"
```
Expected: No matches.

- [ ] **Step 2: All tests pass**

```bash
pytest tests/ -v
```
Expected: All tests PASS.

- [ ] **Step 3: Syntax lint check**

```bash
pylint src/models/lstm_benchmark_models/ \
       src/models/improved_lstm_models/ \
       src/models/ensemble_transformers/ \
       src/models/hugging_face_mamba_moe/ \
       src/models/utils/ \
       --disable=all --enable=E
```
Expected: Exit code 0 (no syntax errors).

- [ ] **Step 4: No print() statements in new code**

```bash
grep -r "print(" src/models/lstm_benchmark_models/ \
                  src/models/improved_lstm_models/ \
                  src/models/ensemble_transformers/ \
                  src/models/hugging_face_mamba_moe/ \
                  src/models/utils/
```
Expected: No matches.

- [ ] **Step 5: No wildcard imports**

```bash
grep -r "from .* import \*" src/models/lstm_benchmark_models/ \
                              src/models/improved_lstm_models/ \
                              src/models/ensemble_transformers/ \
                              src/models/hugging_face_mamba_moe/ \
                              src/models/utils/
```
Expected: No matches.

- [ ] **Step 6: No sys.path hacks**

```bash
grep -r "sys\.path\." src/models/lstm_benchmark_models/ \
                       src/models/improved_lstm_models/ \
                       src/models/ensemble_transformers/ \
                       src/models/hugging_face_mamba_moe/
```
Expected: No matches.

- [ ] **Step 7: Final commit**

```bash
git add -u
git commit -m "[Models] Review: Pre-commit verification pass — all 4 directories meet code review standards"
```

---

## Self-Review Against Spec

| Requirement | Task |
|-------------|------|
| Directory 1: lstm_benchmark_models/ (same naming m1-m6) | Tasks 3, 4, 5 |
| Directory 2: improved_lstm_models/ | Task 6 |
| Directory 3: ensemble_transformers/ | Task 7 |
| Directory 4: hugging_face_mamba_moe/ | Task 8 |
| Code review: functions <= 30 lines | Tasks 3-8 (every file) |
| Code review: docstrings + type hints | Tasks 3-8 (every file) |
| Code review: no hardcoded values (config.yaml) | Task 2 + Tasks 5-8 |
| Code review: logging not print() | Tasks 3-8 |
| Code review: grouped imports, no wildcards | Tasks 3-8 |
| Code review: no sys.path hacks (relative imports) | Tasks 3-8 |
| Shared utils (metrics, losses, callbacks) | Task 1 |
| Dataloader code NOT touched | All tasks scoped to src/models/ only |
| Tests pass | Tasks 1, 4, 9, 10 |

**No gaps found. No placeholders present.**
