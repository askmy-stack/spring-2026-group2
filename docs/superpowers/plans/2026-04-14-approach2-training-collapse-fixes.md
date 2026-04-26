# Approach2 Training Collapse Fixes — Implementation Plan


> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 6 bugs in `src/models/approach2/train.py` that cause all models to collapse to predicting all-negatives (F1=0.0000, Loss=0.0000) from epoch 1.

**Architecture:** All fixes are concentrated in `train.py` (loss computation, threshold search, LR schedule). No architecture file changes are needed — the collapse is entirely a training-loop problem.

**Tech Stack:** Python 3, PyTorch 2.x, scikit-learn, CUDA AMP (GradScaler)

---

## Root Cause Diagnosis

The observed symptoms (Loss→0.0000 by epoch 2, F1=0.0000, Sens=0.0000, Spec=1.0000 throughout all 50 epochs) are caused by a cascade of three bugs:

1. **`FocalLoss` runs inside `torch.amp.autocast('cuda')`** → BCE computed in FP16 → once logits exceed ~±20, `sigmoid` underflows to 0 in FP16 → focal weight = 0 → reported loss = 0.0000 while the model is producing garbage logits.
2. **LR=3e-3 with no warmup** drives logits to extreme values within the first 100 batches (before the cosine scheduler has any effect), triggering the FP16 underflow above.
3. **`find_optimal_threshold` scans only [0.1, 0.9]** → once all sigmoid outputs are < 0.1 (due to collapsed negative logits), every threshold still predicts all-zero → F1=0 forever even if the loss were fixed.

---

## Files Changed

| File | Change |
|------|--------|
| `src/models/approach2/train.py` | All 6 fixes (FocalLoss, threshold, LR, warmup, pos_weight, torch.load) |

---

## Task 1: Fix FocalLoss FP16 underflow (Critical)

**Files:**
- Modify: `src/models/approach2/train.py` — `FocalLoss.forward` method (~lines 47–64)

**Problem:** `FocalLoss.forward` runs inside `torch.amp.autocast('cuda')`, so all tensor math is FP16. Once model logits exceed ±20 in magnitude, `sigmoid(logit)` underflows to 0 or saturates to 1 in FP16, causing both `focal_weight` and `bce_loss` to become 0. The optimizer still receives gradients (from unscaled path) but they are misleading.

**Fix:** Cast inputs to float32 at the top of `forward` so BCE and sigmoid are always in FP32.

- [ ] **Step 1: Replace FocalLoss.forward with FP32-safe version**

In `src/models/approach2/train.py`, replace the `FocalLoss.forward` method:

```python
def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Cast to float32 to prevent FP16 underflow inside torch.amp.autocast
    logits = logits.float()
    targets = targets.float()
    bce_loss = self.bce(logits, targets)
    probs = torch.sigmoid(logits)
    pt = torch.where(targets == 1, probs, 1 - probs)
    focal_weight = (1 - pt) ** self.gamma
    loss = focal_weight * bce_loss
    if self.reduction == "mean":
        return loss.mean()
    elif self.reduction == "none":
        return loss
    return loss.sum()
```

- [ ] **Step 2: Verify fix by running a quick sanity check**

```bash
cd src/models/approach2
python -c "
import torch, torch.nn as nn
from train import FocalLoss
# simulate collapsed logits that cause FP16 underflow
logits = torch.tensor([-30.0, -25.0, -20.0, 20.0], dtype=torch.float16).unsqueeze(1)
targets = torch.tensor([1.0, 1.0, 0.0, 0.0]).unsqueeze(1)
loss_fn = FocalLoss(gamma=1.0)
with torch.amp.autocast('cpu', dtype=torch.float16):
    loss = loss_fn(logits, targets)
print('Loss (should be non-zero):', loss.item())
assert loss.item() > 0.0, 'FocalLoss still underflowing!'
print('PASS')
"
```
Expected: `Loss (should be non-zero): <some positive value>` and `PASS`

- [ ] **Step 3: Commit**

```bash
git add src/models/approach2/train.py
git commit -m "fix: cast FocalLoss inputs to float32 to prevent FP16 underflow in AMP"
```

---

## Task 2: Fix pos_weight + focal interaction (Important)

**Files:**
- Modify: `src/models/approach2/train.py` — `FocalLoss.__init__` and `FocalLoss.forward`

**Problem:** `pos_weight` is passed to `BCEWithLogitsLoss` which upscales positive-class loss *before* focal modulation. This means easy positive examples (high `pt`, low focal weight) get their `pos_weight` boost suppressed by the focal term — the opposite of the intended behavior. Hard negatives get no `pos_weight` boost, hard positives get one, but easy positives get far less than `pos_weight` implies.

**Fix:** Remove `pos_weight` from the BCE constructor. Apply it as a flat per-sample multiplier *after* focal modulation.

- [ ] **Step 1: Update FocalLoss.__init__ and forward**

Replace the entire `FocalLoss` class in `src/models/approach2/train.py`:

```python
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, gamma: float = 2.0, pos_weight: Optional[torch.Tensor] = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight
        # No pos_weight here — applied manually after focal modulation
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Cast to float32 to prevent FP16 underflow inside torch.amp.autocast
        logits = logits.float()
        targets = targets.float()

        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * bce_loss

        # Apply pos_weight as flat per-sample multiplier (after focal, not inside BCE)
        if self.pos_weight is not None:
            class_weight = torch.where(
                targets == 1,
                self.pos_weight.expand_as(targets),
                torch.ones_like(targets),
            )
            focal_loss = focal_loss * class_weight

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "none":
            return focal_loss
        return focal_loss.sum()
```

- [ ] **Step 2: Quick sanity test**

```bash
python -c "
import torch
from train import FocalLoss
loss_fn = FocalLoss(gamma=1.0, pos_weight=torch.tensor([2.0]))
logits = torch.tensor([[-1.0], [1.0], [-0.5], [0.5]])
targets = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
loss = loss_fn(logits, targets)
print('Loss:', loss.item(), '(should be ~0.4-0.9)')
assert 0.1 < loss.item() < 3.0, 'Unexpected loss value'
print('PASS')
"
```

- [ ] **Step 3: Commit**

```bash
git add src/models/approach2/train.py
git commit -m "fix: apply pos_weight as flat multiplier after focal modulation, not inside BCE"
```

---

## Task 3: Fix threshold search range (Critical)

**Files:**
- Modify: `src/models/approach2/train.py` — `find_optimal_threshold` function (~lines 75–92)

**Problem:** `find_optimal_threshold` scans thresholds from 0.1 to 0.9. When the model collapses and outputs all probabilities < 0.1, the minimum threshold of 0.1 still predicts all-zero (F1=0). Even after fixing the loss collapse, the early-training probabilities can legitimately be small (0.01–0.08) before the model learns to separate the classes, causing a diagnostic blind spot.

**Fix:** Scan [0.01, 0.99] in 0.01 steps to capture the full probability range.

- [ ] **Step 1: Replace find_optimal_threshold**

```python
def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find optimal threshold using Youden's J statistic.
    Scans [0.01, 0.99] so that very low probability outputs are handled correctly.
    """
    if len(np.unique(y_true)) < 2:
        return 0.5

    thresholds = np.arange(0.01, 1.00, 0.01)
    best_j = -1.0
    best_thresh = 0.5

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(float)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        j = sens + spec - 1.0
        if j > best_j:
            best_j = j
            best_thresh = thresh

    return best_thresh
```

- [ ] **Step 2: Verify with a collapsed-model scenario**

```bash
python -c "
import numpy as np
from train import find_optimal_threshold
# simulate collapsed model: all probs < 0.05
y_true = np.array([0,0,0,1,0,0,1,0,0,0])
y_prob = np.array([0.01,0.02,0.01,0.04,0.02,0.01,0.03,0.01,0.02,0.01])
t = find_optimal_threshold(y_true, y_prob)
print('Threshold found:', t, '(should be ~0.03-0.04, not 0.5)')
assert t < 0.1, 'Threshold is too high for low-probability outputs'
print('PASS')
"
```
Expected: threshold ~0.03-0.04

- [ ] **Step 3: Commit**

```bash
git add src/models/approach2/train.py
git commit -m "fix: expand threshold scan to [0.01, 0.99] to handle low probability outputs"
```

---

## Task 4: Fix learning rate and add warmup (Critical)

**Files:**
- Modify: `src/models/approach2/train.py` — `train_model` function (~lines 260–345) and `main` parser

**Problem:** Default `lr=3e-3` is too aggressive for a 96K-sample dataset with AdamW and no warmup. The `CosineAnnealingWarmRestarts` with `T_0=10` starts at full LR on epoch 1 and drops over 10 epochs. In the first ~100 batches (well within epoch 1), the high LR drives model logits to extreme values, triggering the FP16 underflow and the all-negative collapse.

**Fix:** Lower default LR to `5e-4` and add a 5-epoch linear warmup before the cosine schedule.

- [ ] **Step 1: Add WarmupCosineScheduler class**

Add this class to `src/models/approach2/train.py` after the `apply_label_smoothing` function:

```python
class WarmupCosineScheduler:
    """Linear warmup followed by cosine annealing."""

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]

    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            factor = (epoch + 1) / max(self.warmup_epochs, 1)
        else:
            progress = (epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
            factor = 0.5 * (1 + np.cos(np.pi * progress))
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group['lr'] = max(self.min_lr, base_lr * factor)

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']
```

- [ ] **Step 2: Update train_model to use WarmupCosineScheduler with lr=5e-4 default**

In `train_model`, replace the scheduler creation block:

```python
# OLD — remove these lines:
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2
)

# NEW — replace with:
scheduler = WarmupCosineScheduler(
    optimizer,
    warmup_epochs=5,
    total_epochs=epochs,
)
```

And change the default `lr` parameter in `train_model` signature:
```python
def train_model(
    ...
    lr: float = 5e-4,   # was 3e-3
    ...
```

- [ ] **Step 3: Update scheduler.step() call inside the training loop**

The old `scheduler.step()` (no arg) was for PyTorch schedulers. Replace with:

```python
# Replace:
scheduler.step()

# With (call BEFORE train_one_epoch so warmup is active from epoch 1):
scheduler.step(epoch)   # epoch is 0-indexed (loop: for epoch in range(epochs))
```

Note: The training loop uses `for epoch in range(epochs)` (0-indexed). The scheduler's `step(epoch)` call goes at the TOP of the loop body, before `train_one_epoch`.

- [ ] **Step 4: Update CLI default**

In `main()`, update the argparse default:
```python
parser.add_argument("--lr", type=float, default=5e-4)   # was 3e-3
```

- [ ] **Step 5: Commit**

```bash
git add src/models/approach2/train.py
git commit -m "fix: lower default lr to 5e-4, add 5-epoch linear warmup before cosine schedule"
```

---

## Task 5: Fix torch.load missing weights_only (Minor)

**Files:**
- Modify: `src/models/approach2/train.py` — `load_data` function (~lines 195–230)

**Problem:** All `torch.load` calls lack `weights_only=True`, causing `FutureWarning` spam in PyTorch 2+ and a potential deserialization security issue for untrusted files.

- [ ] **Step 1: Add weights_only=True to all torch.load calls in load_data**

In `load_data`, replace all four `torch.load(...)` calls:

```python
# Replace:
X_train = torch.load(data_path / "train" / "data.pt").numpy()
y_train = torch.load(data_path / "train" / "labels.pt").numpy().squeeze()

# With:
X_train = torch.load(data_path / "train" / "data.pt", weights_only=True).numpy()
y_train = torch.load(data_path / "train" / "labels.pt", weights_only=True).numpy().squeeze()
```

```python
# Replace:
X_test = torch.load(data_path / "test" / "data.pt").numpy()
y_test = torch.load(data_path / "test" / "labels.pt").numpy().squeeze()
# and val variant:
X_test = torch.load(data_path / "val" / "data.pt").numpy()
y_test = torch.load(data_path / "val" / "labels.pt").numpy().squeeze()

# With:
X_test = torch.load(data_path / "test" / "data.pt", weights_only=True).numpy()
y_test = torch.load(data_path / "test" / "labels.pt", weights_only=True).numpy().squeeze()
# and val variant:
X_test = torch.load(data_path / "val" / "data.pt", weights_only=True).numpy()
y_test = torch.load(data_path / "val" / "labels.pt", weights_only=True).numpy().squeeze()
```

- [ ] **Step 2: Commit**

```bash
git add src/models/approach2/train.py
git commit -m "fix: add weights_only=True to all torch.load calls (suppress FutureWarning)"
```

---

## Task 6: Add early-epoch loss sanity check (Important)

**Files:**
- Modify: `src/models/approach2/train.py` — `train_model` training loop (~line 360)

**Problem:** When Loss=0.0000 appears in the output, there's no warning to the user that training has collapsed. The model silently wastes all remaining epochs computing zeros. A sanity check that warns loudly in the first few epochs would have caught this immediately.

- [ ] **Step 1: Add collapse detection inside the training loop**

In the training loop in `train_model`, after `train_loss = train_one_epoch(...)`, add:

```python
# Sanity check: warn if loss collapses in early epochs
if epoch < 5 and train_loss < 1e-3:
    print(
        f"  WARNING: Loss={train_loss:.6f} is suspiciously near zero at epoch {epoch+1}. "
        "Possible causes: AMP FP16 underflow, LR too high, or degenerate data. "
        "Check model logit magnitudes."
    )
```

- [ ] **Step 2: Commit**

```bash
git add src/models/approach2/train.py
git commit -m "fix: add early-epoch loss collapse warning for fast failure diagnosis"
```

---

## Task 7: Push all fixes

- [ ] **Step 1: Push to remote**

```bash
git push origin abhinaysai-lstm
```

- [ ] **Step 2: Verify on server**

On the remote server:
```bash
cd ~/spring-2026-group2 && git pull origin abhinaysai-lstm
cd src/models/approach2
python train.py --model m1_vanilla_lstm --data_path ../../results/tensors/chbmit --epochs 10 --batch_size 64
```

Expected output: Loss should start ~0.3–0.6 (not 0.0000), F1 and Sensitivity should be non-zero by epoch 5–10.

---

## Self-Review

**Spec coverage:** All 6 root causes diagnosed and addressed:
- FP16 underflow → Task 1
- pos_weight skew → Task 2
- Threshold range → Task 3
- LR + no warmup → Task 4
- torch.load warnings → Task 5
- Silent collapse → Task 6

**Placeholder scan:** No TBDs. All code blocks are complete and runnable.

**Type consistency:** `WarmupCosineScheduler.step(epoch)` takes `int`, consistent with `for epoch in range(epochs)` which produces `int`. `find_optimal_threshold` returns `float`, consumed by `evaluate` as `threshold: float`. Consistent throughout.
