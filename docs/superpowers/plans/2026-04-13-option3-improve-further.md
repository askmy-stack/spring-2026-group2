# Option 3: Improve Further Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enhance model architectures and training strategies based on benchmark insights to increase sensitivity and AUC scores beyond current best (CNN-LSTM: 56.9% sensitivity, 0.712 AUC).

**Architecture:** The plan tests four hypotheses via iterative experimentation: (1) CNN extracts features better than attention; combine them, (2) Class imbalance limits sensitivity; test loss function variants, (3) Models underfit; test larger capacity, (4) Features matter; document feature-BiLSTM approach. Each hypothesis produces a concrete variant to train and evaluate.

**Tech Stack:** Python, PyTorch, focal loss implementation, weighted BCE loss, CNN+attention hybrid architecture, hyperparameter sweep utilities.

---

## File Structure

**Files to create:**
- `src/models/architectures/cnn_attention_bilstm.py` — Hybrid CNN + Attention BiLSTM
- `src/models/architectures/larger_cnn_lstm.py` — Bigger CNN-LSTM (256/512 hidden units)
- `src/models/training/loss_functions.py` — Focal loss, weighted BCE, class-balanced implementations
- `src/models/training/hyperparameter_sweep.py` — Utility to run experiments with parameter variations
- `experiments/hypothesis_testing.md` — Results and analysis for each hypothesis

**Files to modify:**
- `src/models/train.py` — Add loss function options, support focal loss
- `src/models/model_registry.py` — Register new variants

**Reference for feature engineering:**
- `src/models/architectures/feature_bilstm.py` — Existing (to understand feature extraction)

---

### Task 1: Implement Focal Loss and enhanced loss function module

**Files:**
- Create: `src/models/training/loss_functions.py`
- Modify: `src/models/train.py` (import and use)

- [ ] **Step 1: Create loss functions module with Focal Loss**

```python
# src/models/training/loss_functions.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.
    
    Focus on hard-to-classify examples by down-weighting easy negatives.
    Reference: Lin et al. (2017) "Focal Loss for Dense Object Detection"
    
    Args:
        alpha: Weight for positive class (default: 0.25, balanced for seizure 2-5%)
        gamma: Focusing parameter (default: 2.0)
        pos_weight: Alternative to alpha for imbalance handling
    """
    def __init__(self, alpha=None, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
    
    def forward(self, logits, targets):
        """
        Args:
            logits: Model output (before sigmoid), shape (batch_size,)
            targets: Ground truth binary labels, shape (batch_size,)
        
        Returns:
            Scalar loss
        """
        # Binary cross-entropy with logits
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
        
        # Compute focal weight
        p = torch.sigmoid(logits)
        p_t = torch.where(targets == 1, p, 1 - p)  # p_t: probability of true class
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting if specified
        if self.alpha is not None:
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            focal_weight = alpha_t * focal_weight
        
        # Scale BCE by focal weight
        loss = focal_weight * bce
        return loss.mean()


class ClassBalancedBCE(nn.Module):
    """Class-balanced BCE loss using effective number of samples.
    
    Reference: Cui et al. (2019) "Class-Balanced Loss Based on Effective Number of Samples"
    
    Useful when class counts are highly imbalanced (seizures 2-5%, non-seizures 95-98%).
    """
    def __init__(self, beta=0.9999):
        super().__init__()
        self.beta = beta  # Controls strength of balancing (higher = stronger)
    
    def forward(self, logits, targets, pos_count=None, neg_count=None):
        """
        Args:
            logits: Model output (before sigmoid), shape (batch_size,)
            targets: Ground truth binary labels
            pos_count: Total positive samples in dataset (if known)
            neg_count: Total negative samples in dataset (if known)
        
        Returns:
            Scalar loss
        """
        # If counts not provided, estimate from batch
        if pos_count is None:
            pos_count = (targets == 1).sum().item()
            neg_count = (targets == 0).sum().item()
        
        # Effective number of samples
        # Higher effective_num for rare class → lower weight for that class
        effective_num_pos = 1.0 - (self.beta ** pos_count) if pos_count > 0 else 1.0
        effective_num_neg = 1.0 - (self.beta ** neg_count) if neg_count > 0 else 1.0
        
        # Weight for positive class (seizures) — should be higher
        weights_pos = neg_count / (effective_num_neg + 1e-8) if effective_num_pos > 0 else 1.0
        weights_neg = pos_count / (effective_num_pos + 1e-8) if effective_num_neg > 0 else 1.0
        
        # Binary cross-entropy with custom weights
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
        
        # Apply weights
        weighted_bce = torch.where(targets == 1, weights_pos * bce, weights_neg * bce)
        return weighted_bce.mean()


def get_loss_function(loss_type, **kwargs):
    """Factory function to select loss function.
    
    Args:
        loss_type: 'bce', 'weighted_bce', 'focal', 'class_balanced'
        **kwargs: Arguments to pass to loss class
    
    Returns:
        Instantiated loss module
    """
    if loss_type == 'bce':
        return nn.BCEWithLogitsLoss(**kwargs)
    elif loss_type == 'weighted_bce':
        return nn.BCEWithLogitsLoss(**kwargs)  # pos_weight in kwargs
    elif loss_type == 'focal':
        return FocalLoss(**kwargs)
    elif loss_type == 'class_balanced':
        return ClassBalancedBCE(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
```

Save as `src/models/training/loss_functions.py`.

- [ ] **Step 2: Create __init__.py for training module**

```bash
touch src/models/training/__init__.py
```

- [ ] **Step 3: Update train.py to accept loss function parameter**

In `src/models/train.py`, modify the argument parser to add:

```python
parser.add_argument(
    '--loss-type',
    choices=['bce', 'weighted_bce', 'focal', 'class_balanced'],
    default='weighted_bce',
    help='Loss function type for handling class imbalance'
)

parser.add_argument(
    '--focal-alpha',
    type=float,
    default=0.25,
    help='Focal loss alpha parameter (balance factor for positive class)'
)

parser.add_argument(
    '--focal-gamma',
    type=float,
    default=2.0,
    help='Focal loss gamma parameter (focusing strength)'
)
```

And in the training setup section, replace the loss creation:

```python
# Old:
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

# New:
from src.models.training.loss_functions import get_loss_function

if args.loss_type == 'weighted_bce':
    criterion = get_loss_function(args.loss_type, pos_weight=torch.tensor(pos_weight))
elif args.loss_type == 'focal':
    criterion = get_loss_function(
        args.loss_type,
        alpha=args.focal_alpha,
        gamma=args.focal_gamma
    )
else:
    criterion = get_loss_function(args.loss_type)
```

- [ ] **Step 4: Test loss module imports**

```bash
python -c "from src.models.training.loss_functions import FocalLoss, ClassBalancedBCE; print('Loss functions imported successfully')"
```

Expected: "Loss functions imported successfully"

- [ ] **Step 5: Commit**

```bash
git add src/models/training/loss_functions.py src/models/training/__init__.py
git commit -m "feat: Add focal loss and class-balanced loss for handling class imbalance"
```

---

### Task 2: Create CNN + Attention hybrid architecture

**Files:**
- Create: `src/models/architectures/cnn_attention_bilstm.py`
- Modify: `src/models/model_registry.py`

- [ ] **Step 1: Create hybrid architecture combining CNN preprocessing with Attention BiLSTM**

```python
# src/models/architectures/cnn_attention_bilstm.py
import torch
import torch.nn as nn
from torch.nn import MultiheadAttention


class CNNAttentionBiLSTM(nn.Module):
    """Hybrid architecture: CNN feature extraction + Attention BiLSTM.
    
    Motivation: CNN-LSTM wins because convolution captures local EEG patterns.
    This variant combines:
    - CNN1d for local feature extraction
    - BiLSTM for temporal modeling
    - MultiheadAttention for pattern focus
    
    This tests hypothesis: CNN + Attention = best of both?
    
    Architecture:
    1. Input projection & LayerNorm
    2. Conv1d + ReLU + MaxPool1d (extract local features)
    3. BiLSTM × 2 layers (temporal modeling)
    4. MultiheadAttention (focus on important temporal patterns)
    5. Global avg + max pooling
    6. LayerNorm
    7. FC head → output
    """
    
    def __init__(
        self,
        input_size=19,  # 19 EEG channels
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        num_heads=4,
        conv_out_channels=32,
        kernel_size=3,
    ):
        super().__init__()
        
        # Input projection & normalization
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)
        
        # CNN preprocessing for local feature extraction
        self.conv1d = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=conv_out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        
        # BiLSTM
        lstm_input_size = conv_out_channels
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention
        self.attention = MultiheadAttention(
            embed_dim=hidden_size * 2,  # BiLSTM outputs 2*hidden_size
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Pooling & FC head
        self.dropout = nn.Dropout(dropout)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # LayerNorm after pooling
        self.pool_norm = nn.LayerNorm(hidden_size * 2 * 2)  # avg + max pools both
        
        # FC head
        self.fc1 = nn.Linear(hidden_size * 2 * 2, hidden_size)
        self.fc1_relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size=19)
        
        Returns:
            logits: (batch_size, 1)
        """
        # Input projection & norm
        x = self.input_projection(x)  # (batch, seq, hidden)
        x = self.input_norm(x)
        
        # CNN: transpose to (batch, hidden, seq)
        x = x.transpose(1, 2)
        x = self.conv1d(x)  # (batch, conv_out, seq)
        x = self.relu(x)
        x = self.maxpool(x)  # (batch, conv_out, seq//2)
        
        # Back to (batch, seq, conv_out)
        x = x.transpose(1, 2)
        
        # BiLSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden*2)
        
        # Attention (self-attention on LSTM output)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)  # (batch, seq, hidden*2)
        
        # Residual connection
        lstm_out_residual = lstm_out + attn_out
        
        # Pooling: transpose to (batch, hidden*2, seq)
        x = lstm_out_residual.transpose(1, 2)
        avg_pool = self.global_avg_pool(x)  # (batch, hidden*2, 1)
        max_pool = self.global_max_pool(x)  # (batch, hidden*2, 1)
        
        # Concatenate pools and flatten
        x = torch.cat([avg_pool, max_pool], dim=1)  # (batch, hidden*2*2, 1)
        x = x.squeeze(-1)  # (batch, hidden*2*2)
        
        # LayerNorm & FC head
        x = self.pool_norm(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc1_relu(x)
        x = self.dropout(x)
        x = self.fc2(x).squeeze(-1)  # (batch,)
        
        return x
```

Save as `src/models/architectures/cnn_attention_bilstm.py`.

- [ ] **Step 2: Register the new architecture in model_registry.py**

Add to `src/models/model_registry.py`:

```python
from src.models.architectures.cnn_attention_bilstm import CNNAttentionBiLSTM

MODEL_REGISTRY = {
    ...
    'cnn_attention_bilstm': CNNAttentionBiLSTM,
    ...
}
```

- [ ] **Step 3: Verify architecture can be instantiated**

```bash
python -c "from src.models.architectures.cnn_attention_bilstm import CNNAttentionBiLSTM; model = CNNAttentionBiLSTM(); print(f'Model created: {type(model).__name__}'); x = torch.randn(2, 1000, 19); out = model(x); print(f'Output shape: {out.shape}')"
```

Expected: "Model created: CNNAttentionBiLSTM" and "Output shape: torch.Size([2])"

- [ ] **Step 4: Commit**

```bash
git add src/models/architectures/cnn_attention_bilstm.py
git commit -m "feat: Add CNN+Attention hybrid architecture to test hypothesis"
```

---

### Task 3: Create larger CNN-LSTM variant for underfitting hypothesis

**Files:**
- Create: `src/models/architectures/larger_cnn_lstm.py`
- Modify: `src/models/model_registry.py`

- [ ] **Step 1: Create larger CNN-LSTM with increased capacity**

```python
# src/models/architectures/larger_cnn_lstm.py
import torch
import torch.nn as nn


class LargerCNNLSTM(nn.Module):
    """Expanded CNN-LSTM with larger hidden sizes to test underfitting hypothesis.
    
    Original CNN-LSTM: hidden_size=128
    This variant: hidden_size=256 or 512
    
    Hypothesis: Original models may underfit (low sensitivity).
    Test: Does increased capacity improve sensitivity?
    
    Architecture:
    1. Conv1d layers (more channels: 64 instead of 32)
    2. 2-layer LSTM with larger hidden_size (256)
    3. Global avg + max pooling
    4. 2-layer FC head with dropout
    """
    
    def __init__(
        self,
        input_size=19,
        hidden_size=256,  # Increased from 128
        num_layers=2,
        dropout=0.3,
        conv_channels=64,  # Increased from 32
    ):
        super().__init__()
        
        # Input projection & norm
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)
        
        # CNN
        self.conv1d = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=conv_channels,
            kernel_size=5,  # Slightly larger kernel
            padding=2
        )
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Pooling
        self.dropout = nn.Dropout(dropout)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # FC head with larger capacity
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc1_relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size=19)
        
        Returns:
            logits: (batch_size, 1)
        """
        # Input projection & norm
        x = self.input_projection(x)
        x = self.input_norm(x)
        
        # CNN
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Pooling
        x = lstm_out.transpose(1, 2)
        avg_pool = self.global_avg_pool(x).squeeze(-1)
        max_pool = self.global_max_pool(x).squeeze(-1)
        x = torch.cat([avg_pool, max_pool], dim=-1)
        
        # FC head
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc1_relu(x)
        x = self.dropout(x)
        x = self.fc2(x).squeeze(-1)
        
        return x
```

Save as `src/models/architectures/larger_cnn_lstm.py`.

- [ ] **Step 2: Register the architecture**

Add to `src/models/model_registry.py`:

```python
from src.models.architectures.larger_cnn_lstm import LargerCNNLSTM

MODEL_REGISTRY = {
    ...
    'larger_cnn_lstm': LargerCNNLSTM,
    ...
}
```

- [ ] **Step 3: Verify the model**

```bash
python -c "from src.models.architectures.larger_cnn_lstm import LargerCNNLSTM; model = LargerCNNLSTM(); x = torch.randn(2, 1000, 19); out = model(x); print(f'Output shape: {out.shape}')"
```

Expected: "Output shape: torch.Size([2])"

- [ ] **Step 4: Commit**

```bash
git add src/models/architectures/larger_cnn_lstm.py
git commit -m "feat: Add larger CNN-LSTM (hidden_size=256) to test underfitting hypothesis"
```

---

### Task 4: Create hyperparameter sweep utility

**Files:**
- Create: `src/models/training/hyperparameter_sweep.py`

- [ ] **Step 1: Create sweep utility for systematic testing**

```python
# src/models/training/hyperparameter_sweep.py
import json
import os
from itertools import product
from pathlib import Path
import subprocess
import sys


class HyperparameterSweep:
    """Utility to run systematic hyperparameter experiments.
    
    Supports grid search over:
    - Model architectures
    - Loss functions
    - Class weights / loss hyperparameters
    - Learning rates
    
    Logs results to JSON for comparison.
    """
    
    def __init__(self, output_dir='experiments/sweep_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
    
    def run_experiment(self, model_name, loss_type, loss_kwargs, lr, epochs=100, batch_size=64):
        """Run a single training experiment.
        
        Args:
            model_name: Key from MODEL_REGISTRY (e.g., 'cnn_lstm', 'cnn_attention_bilstm')
            loss_type: 'weighted_bce', 'focal', 'class_balanced'
            loss_kwargs: Dict of loss-specific parameters (e.g., {'alpha': 0.25, 'gamma': 2.0})
            lr: Learning rate
            epochs: Number of training epochs
            batch_size: Batch size
        
        Returns:
            Dict with results (accuracy, sensitivity, AUC, etc.)
        """
        # Build command
        cmd = [
            sys.executable, 'src/models/train.py',
            '--model', model_name,
            '--loss-type', loss_type,
            '--learning-rate', str(lr),
            '--epochs', str(epochs),
            '--batch-size', str(batch_size),
            '--output-dir', str(self.output_dir / f'{model_name}_{loss_type}_{lr}'),
        ]
        
        # Add loss-specific parameters
        if loss_type == 'focal':
            cmd.extend(['--focal-alpha', str(loss_kwargs.get('alpha', 0.25))])
            cmd.extend(['--focal-gamma', str(loss_kwargs.get('gamma', 2.0))])
        
        print(f"\n{'='*80}")
        print(f"Running: {' '.join(cmd)}")
        print(f"{'='*80}\n")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
            
            # Parse results from output or checkpoint
            # (Assumes train.py writes final metrics to output-dir/metrics.json)
            metrics_file = Path(self.output_dir) / f'{model_name}_{loss_type}_{lr}' / 'metrics.json'
            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics = json.load(f)
                    metrics.update({
                        'model': model_name,
                        'loss_type': loss_type,
                        'loss_kwargs': loss_kwargs,
                        'lr': lr,
                    })
                    self.results.append(metrics)
                    return metrics
            else:
                print(f"Warning: Metrics file not found at {metrics_file}")
                return None
        except subprocess.CalledProcessError as e:
            print(f"Error running experiment: {e}")
            print(f"Stderr: {e.stderr}")
            return None
    
    def run_grid_search(self, param_grid):
        """Run experiments for all parameter combinations.
        
        Args:
            param_grid: Dict with keys:
                - 'models': List of model names
                - 'loss_types': List of loss function types
                - 'loss_kwargs': Dict or list of dicts for loss hyperparameters
                - 'learning_rates': List of learning rates
        
        Example:
            param_grid = {
                'models': ['cnn_lstm', 'cnn_attention_bilstm'],
                'loss_types': ['weighted_bce', 'focal'],
                'loss_kwargs': [
                    {},  # defaults
                    {'alpha': 0.1, 'gamma': 1.5},
                ],
                'learning_rates': [1e-3, 5e-4],
            }
            sweep.run_grid_search(param_grid)
        """
        models = param_grid.get('models', ['cnn_lstm'])
        loss_types = param_grid.get('loss_types', ['weighted_bce'])
        loss_kwargs_list = param_grid.get('loss_kwargs', [{}])
        learning_rates = param_grid.get('learning_rates', [1e-3])
        
        total = len(models) * len(loss_types) * len(loss_kwargs_list) * len(learning_rates)
        print(f"\nStarting hyperparameter sweep: {total} experiments\n")
        
        count = 0
        for model, loss_type, loss_kwargs, lr in product(
            models, loss_types, loss_kwargs_list, learning_rates
        ):
            count += 1
            print(f"\n[{count}/{total}]", end="")
            self.run_experiment(model, loss_type, loss_kwargs, lr)
    
    def summarize_results(self, output_file='experiments/sweep_summary.json'):
        """Save and print summary of results."""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n\nSweep Results Summary ({len(self.results)} experiments):")
        print(f"Saved to: {output_file}\n")
        
        # Print top performers by sensitivity
        if self.results:
            sorted_by_sensitivity = sorted(
                self.results,
                key=lambda x: x.get('sensitivity', 0),
                reverse=True
            )
            print("Top 5 by Sensitivity:")
            for i, result in enumerate(sorted_by_sensitivity[:5], 1):
                print(f"  {i}. {result['model']} ({result['loss_type']}, "
                      f"lr={result['lr']:.0e}): "
                      f"Sensitivity {result.get('sensitivity', 0):.1%}, "
                      f"AUC {result.get('auc', 0):.3f}")


if __name__ == '__main__':
    # Example usage
    param_grid = {
        'models': ['cnn_lstm', 'cnn_attention_bilstm', 'larger_cnn_lstm'],
        'loss_types': ['weighted_bce', 'focal', 'class_balanced'],
        'loss_kwargs': [
            {},  # weighted_bce defaults
            {'alpha': 0.25, 'gamma': 2.0},  # focal defaults
            {'alpha': 0.1, 'gamma': 3.0},  # focal aggressive
        ],
        'learning_rates': [1e-3, 5e-4],
    }
    
    sweep = HyperparameterSweep()
    sweep.run_grid_search(param_grid)
    sweep.summarize_results()
```

Save as `src/models/training/hyperparameter_sweep.py`.

- [ ] **Step 2: Verify the sweep utility imports**

```bash
python -c "from src.models.training.hyperparameter_sweep import HyperparameterSweep; print('HyperparameterSweep imported successfully')"
```

Expected: "HyperparameterSweep imported successfully"

- [ ] **Step 3: Commit**

```bash
git add src/models/training/hyperparameter_sweep.py
git commit -m "feat: Add hyperparameter sweep utility for systematic experimentation"
```

---

### Task 5: Run experiments testing Hypothesis 1 — CNN + Attention

**Files:**
- Run: CNN + Attention hybrid training
- Log: Results to experiments/

- [ ] **Step 1: Train CNN+Attention hybrid with baseline config**

```bash
python src/models/train.py \
  --model cnn_attention_bilstm \
  --dataset src/data/chb-mit/ \
  --epochs 100 \
  --batch-size 64 \
  --learning-rate 1e-3 \
  --loss-type weighted_bce \
  --output-dir experiments/cnn_attention_bilstm_baseline \
  --device cuda
```

Expected: Training runs for 30-60 minutes, prints progress, saves checkpoints.

- [ ] **Step 2: Train CNN+Attention with Focal Loss**

```bash
python src/models/train.py \
  --model cnn_attention_bilstm \
  --dataset src/data/chb-mit/ \
  --epochs 100 \
  --batch-size 64 \
  --learning-rate 1e-3 \
  --loss-type focal \
  --focal-alpha 0.25 \
  --focal-gamma 2.0 \
  --output-dir experiments/cnn_attention_bilstm_focal \
  --device cuda
```

- [ ] **Step 3: Compare results to baseline CNN-LSTM**

After training completes:

```bash
python -c "
import json

results = {}
for exp in ['baseline_cnn_lstm', 'cnn_attention_bilstm_baseline', 'cnn_attention_bilstm_focal']:
    try:
        with open(f'experiments/{exp}/metrics.json') as f:
            results[exp] = json.load(f)
    except FileNotFoundError:
        pass

print('Sensitivity Comparison:')
for exp, metrics in results.items():
    print(f'  {exp}: {metrics.get(\"sensitivity\", \"N/A\"):.1%}')
"
```

Expected: Prints sensitivity for each variant; answer: Does CNN+Attention beat pure CNN-LSTM?

- [ ] **Step 4: Document findings**

Create `experiments/hypothesis_testing.md` and add Hypothesis 1 section:

```markdown
# Hypothesis Testing Results

## Hypothesis 1: CNN + Attention Hybrid

**Prediction:** CNN captures local patterns well; adding attention may improve long-range temporal modeling.

**Models tested:**
- CNN-LSTM (baseline): Sensitivity 56.9%, AUC 0.712
- CNN+Attention BiLSTM (weighted BCE): [INSERT RESULT]
- CNN+Attention BiLSTM (focal loss): [INSERT RESULT]

**Results:** [Analyze whether CNN+Attention beats pure CNN-LSTM]

**Conclusion:** [Did adding attention help? Why or why not?]
```

- [ ] **Step 5: Commit results**

```bash
git add experiments/cnn_attention_bilstm_baseline/ experiments/cnn_attention_bilstm_focal/
git commit -m "exp: Test Hypothesis 1 - CNN+Attention hybrid architecture"
```

---

### Task 6: Run experiments testing Hypothesis 2 — Class Imbalance Loss Functions

**Files:**
- Run: Multiple loss functions on CNN-LSTM
- Log: Results to experiments/

- [ ] **Step 1: Train CNN-LSTM with Focal Loss (aggressive)**

```bash
python src/models/train.py \
  --model cnn_lstm \
  --dataset src/data/chb-mit/ \
  --epochs 100 \
  --batch-size 64 \
  --learning-rate 1e-3 \
  --loss-type focal \
  --focal-alpha 0.1 \
  --focal-gamma 3.0 \
  --output-dir experiments/cnn_lstm_focal_aggressive \
  --device cuda
```

Expected: Training with stronger focal loss (more emphasis on hard negatives).

- [ ] **Step 2: Train CNN-LSTM with Class-Balanced Loss**

```bash
python src/models/train.py \
  --model cnn_lstm \
  --dataset src/data/chb-mit/ \
  --epochs 100 \
  --batch-size 64 \
  --learning-rate 1e-3 \
  --loss-type class_balanced \
  --output-dir experiments/cnn_lstm_class_balanced \
  --device cuda
```

- [ ] **Step 3: Compare loss function effectiveness**

```bash
python -c "
import json

loss_funcs = {
    'baseline_cnn_lstm': 'weighted_bce (original)',
    'cnn_lstm_focal_aggressive': 'focal (alpha=0.1, gamma=3.0)',
    'cnn_lstm_class_balanced': 'class_balanced',
}

results = {}
for exp, label in loss_funcs.items():
    try:
        with open(f'experiments/{exp}/metrics.json') as f:
            results[label] = json.load(f)
    except:
        pass

print('Loss Function Comparison (CNN-LSTM):')
print('Loss Type | Sensitivity | AUC | F1 | Improvement')
print('-' * 60)
baseline_sens = results.get('weighted_bce (original)', {}).get('sensitivity', 0.569)
for loss_type, metrics in results.items():
    sens = metrics.get('sensitivity', 0)
    auc = metrics.get('auc', 0)
    f1 = metrics.get('f1', 0)
    improvement = f'+{(sens - baseline_sens)*100:.1f}%' if sens > baseline_sens else f'{(sens - baseline_sens)*100:.1f}%'
    print(f'{loss_type:30} | {sens:11.1%} | {auc:5.3f} | {f1:4.3f} | {improvement}')
"
```

Expected: Prints comparison table showing which loss function gives best sensitivity.

- [ ] **Step 4: Update experiments/hypothesis_testing.md with Hypothesis 2**

Add section:

```markdown
## Hypothesis 2: Class Imbalance — Loss Function Impact

**Prediction:** Focal loss or class-balanced loss down-weights easy negatives, improving sensitivity on rare seizures.

**Models tested:**
- CNN-LSTM (weighted BCE baseline): Sensitivity 56.9%
- CNN-LSTM (focal, alpha=0.1, gamma=3.0): [INSERT]
- CNN-LSTM (class-balanced): [INSERT]

**Results:** [Which loss function performs best?]

**Conclusion:** [Did stronger class imbalance handling improve sensitivity?]
```

- [ ] **Step 5: Commit**

```bash
git add experiments/cnn_lstm_focal_aggressive/ experiments/cnn_lstm_class_balanced/
git commit -m "exp: Test Hypothesis 2 - Class imbalance loss functions"
```

---

### Task 7: Run experiments testing Hypothesis 3 — Model Capacity (Underfitting)

**Files:**
- Run: Larger CNN-LSTM variants
- Log: Results to experiments/

- [ ] **Step 1: Train Larger CNN-LSTM (hidden_size=256)**

```bash
python src/models/train.py \
  --model larger_cnn_lstm \
  --dataset src/data/chb-mit/ \
  --epochs 100 \
  --batch-size 64 \
  --learning-rate 1e-3 \
  --loss-type focal \
  --focal-alpha 0.25 \
  --focal-gamma 2.0 \
  --output-dir experiments/larger_cnn_lstm_focal \
  --device cuda
```

Expected: Larger model with more parameters; training may take 40-60 minutes.

- [ ] **Step 2: Compare model sizes**

```bash
python -c "
from src.models.architectures.cnn_lstm import CNNLSTM
from src.models.architectures.larger_cnn_lstm import LargerCNNLSTM

small = CNNLSTM(hidden_size=128)
large = LargerCNNLSTM(hidden_size=256)

small_params = sum(p.numel() for p in small.parameters())
large_params = sum(p.numel() for p in large.parameters())

print(f'CNN-LSTM (hidden=128): {small_params:,} parameters')
print(f'Larger CNN-LSTM (hidden=256): {large_params:,} parameters')
print(f'Increase: {(large_params / small_params):.1f}x')
"
```

Expected: Larger model has ~4x more parameters (2x hidden size = 4x quadratic params).

- [ ] **Step 3: Compare results**

```bash
python -c "
import json

models = {
    'baseline_cnn_lstm': 'CNN-LSTM (hidden=128)',
    'larger_cnn_lstm_focal': 'Larger CNN-LSTM (hidden=256)',
}

results = {}
for exp, label in models.items():
    try:
        with open(f'experiments/{exp}/metrics.json') as f:
            results[label] = json.load(f)
    except:
        pass

print('Model Capacity Comparison:')
print('Model | Sensitivity | AUC | F1 | Improvement')
print('-' * 60)
baseline_sens = results.get('CNN-LSTM (hidden=128)', {}).get('sensitivity', 0.569)
for label, metrics in results.items():
    sens = metrics.get('sensitivity', 0)
    auc = metrics.get('auc', 0)
    f1 = metrics.get('f1', 0)
    improvement = f'+{(sens - baseline_sens)*100:.1f}%' if sens > baseline_sens else f'{(sens - baseline_sens)*100:.1f}%'
    print(f'{label:30} | {sens:11.1%} | {auc:5.3f} | {f1:4.3f} | {improvement}')
"
```

Expected: Shows whether increased capacity improves sensitivity.

- [ ] **Step 4: Update hypothesis testing document**

Add section:

```markdown
## Hypothesis 3: Model Underfitting — Capacity Impact

**Prediction:** Original CNN-LSTM (hidden=128) may underfit; larger model (hidden=256) learns better.

**Models tested:**
- CNN-LSTM (hidden=128, focal loss): [baseline]
- Larger CNN-LSTM (hidden=256, focal loss): [INSERT]

**Results:** [Did larger model improve?]

**Conclusion:** [Is sensitivity limited by model capacity or by data/labels?]
```

- [ ] **Step 5: Commit**

```bash
git add experiments/larger_cnn_lstm_focal/
git commit -m "exp: Test Hypothesis 3 - Model capacity impact on sensitivity"
```

---

### Task 8: Analyze Feature-BiLSTM and document feature engineering gap

**Files:**
- Reference: `src/models/architectures/feature_bilstm.py`
- Create: `experiments/feature_engineering_analysis.md`

- [ ] **Step 1: Examine feature-BiLSTM to understand feature extraction**

```bash
grep -A 30 "class FeatureBiLSTM" src/models/architectures/feature_bilstm.py | head -50
```

Expected: Shows input size (226 features) and notes about feature extraction.

- [ ] **Step 2: Create analysis document**

```bash
cat > experiments/feature_engineering_analysis.md << 'EOF'
# Feature Engineering Analysis

## Feature-BiLSTM Architecture

**Input:** 226 pre-extracted features (not raw EEG)
**Hidden size:** 128
**Architecture:** 2-layer BiLSTM + global pooling + FC head

**Key question:** What are these 226 features?

Possibilities:
1. **Spectral features:** Power spectral density (PSD) across frequency bands (e.g., delta, theta, alpha, beta, gamma)
2. **Wavelet features:** Continuous/discrete wavelet transforms
3. **Time-domain features:** RMS, entropy, spectral centroid
4. **Hand-engineered features:** Linear combinations of the above

## Hypothesis 4: Feature Engineering Matters

If Feature-BiLSTM outperforms Vanilla/BiLSTM on raw EEG:
- **Implication:** Domain-specific feature extraction is crucial
- **Next:** Investigate which features matter most (feature importance analysis)
- **Risk:** Features may overfit to CHB-MIT; may not generalize to other datasets

## Experiment Plan

1. Run baseline Feature-BiLSTM training (if not in benchmark results)
2. Compare raw EEG models (CNN-LSTM) vs. Feature-BiLSTM
3. If Feature-BiLSTM wins:
   - Extract feature importance (permutation, gradient-based)
   - Document the feature extraction pipeline
   - Test on other EEG datasets if available

## Results

[To be filled after running Feature-BiLSTM experiment]
EOF
```

- [ ] **Step 2: Create placeholder for feature-BiLSTM results**

If Feature-BiLSTM results missing from original benchmark:

```bash
python src/models/train.py \
  --model feature_bilstm \
  --dataset src/data/chb-mit/ \
  --epochs 100 \
  --batch-size 64 \
  --learning-rate 1e-3 \
  --loss-type focal \
  --focal-alpha 0.25 \
  --focal-gamma 2.0 \
  --output-dir experiments/feature_bilstm_focal \
  --device cuda
```

- [ ] **Step 3: Compare all models**

```bash
python -c "
import json
from pathlib import Path

experiments = [
    'baseline_cnn_lstm',
    'cnn_attention_bilstm_baseline',
    'larger_cnn_lstm_focal',
    'feature_bilstm_focal',
]

results = {}
for exp in experiments:
    metrics_file = Path(f'experiments/{exp}/metrics.json')
    if metrics_file.exists():
        with open(metrics_file) as f:
            results[exp] = json.load(f)

print('All Experiments — Final Comparison:')
print('Experiment | Sensitivity | AUC | F1 | vs. Baseline')
print('-' * 70)

baseline_sens = results.get('baseline_cnn_lstm', {}).get('sensitivity', 0.569)
for exp in sorted(results.keys()):
    metrics = results[exp]
    sens = metrics.get('sensitivity', 0)
    auc = metrics.get('auc', 0)
    f1 = metrics.get('f1', 0)
    improvement = f'+{(sens - baseline_sens)*100:.1f}%' if sens > baseline_sens else f'{(sens - baseline_sens)*100:.1f}%'
    print(f'{exp:35} | {sens:11.1%} | {auc:5.3f} | {f1:4.3f} | {improvement}')
"
```

- [ ] **Step 4: Commit feature engineering analysis**

```bash
git add experiments/feature_engineering_analysis.md
git commit -m "analysis: Feature engineering impact and Feature-BiLSTM investigation"
```

---

### Task 9: Summarize findings and create synthesis document

**Files:**
- Create: `experiments/improvement_summary.md`
- Update: `wiki/syntheses/architecture-comparison.md`

- [ ] **Step 1: Create comprehensive improvement summary**

```bash
cat > experiments/improvement_summary.md << 'EOF'
# Improvement Experiments: Summary of Findings

**Baseline (from Option 2 Benchmark):**
- CNN-LSTM: Sensitivity 56.9%, AUC 0.712, F1 0.518

**Tested Hypotheses:**

## Hypothesis 1: CNN + Attention Hybrid
- **Prediction:** Combining convolution + attention may improve temporal modeling
- **Result:** [Insert comparison]
- **Conclusion:** [Did it help?]

## Hypothesis 2: Class Imbalance Loss Functions
- **Focal Loss (standard):** Sensitivity [INSERT]
- **Focal Loss (aggressive):** Sensitivity [INSERT]
- **Class-Balanced Loss:** Sensitivity [INSERT]
- **Best improvement:** [Which loss worked best? +X% over baseline?]
- **Conclusion:** [Loss function selection matters / doesn't matter]

## Hypothesis 3: Model Capacity (Underfitting)
- **Larger CNN-LSTM (hidden=256):** Sensitivity [INSERT]
- **Improvement vs. baseline:** [+X% or decline?]
- **Conclusion:** [Sensitivity bottleneck is model capacity / data quality / class imbalance?]

## Hypothesis 4: Feature Engineering
- **Feature-BiLSTM:** Sensitivity [INSERT]
- **vs. CNN-LSTM (raw EEG):** [+X% or -X%?]
- **Conclusion:** [Hand-engineered features matter / raw EEG sufficient]

---

## Summary Table: All Variants

| Model | Loss Function | Sensitivity | AUC | F1 | Improvement |
|-------|---------------|-------------|-----|-----|-------------|
| CNN-LSTM | weighted_bce | 56.9% | 0.712 | 0.518 | Baseline |
| CNN+Attention | weighted_bce | [..] | [..] | [..] | [..] |
| CNN+Attention | focal | [..] | [..] | [..] | [..] |
| Larger CNN-LSTM | focal | [..] | [..] | [..] | [..] |
| Feature-BiLSTM | focal | [..] | [..] | [..] | [..] |

**Winner:** [Model name] with [X%] sensitivity (+[X] vs. baseline)

---

## Key Insights

1. **What worked best?** [Summarize top performer]
2. **What was unexpected?** [Any surprising results?]
3. **What's the bottleneck now?** [Class imbalance? Data quality? Model capacity?]

---

## Next Steps

Based on results:
- [ ] Ensemble the best variants?
- [ ] Investigate failure cases (which seizures are still missed)?
- [ ] Test on external datasets (MIT-BIH, TUH)?
- [ ] Publish findings?
EOF
```

- [ ] **Step 2: Update wiki synthesis page with "Improved Models" section**

Add to `wiki/syntheses/architecture-comparison.md`:

```markdown
## Improved Models (Option 3 Experiments)

After testing 4 hypotheses (CNN+Attention, loss functions, model capacity, features):

**Best improvement achieved:** [Model name] with [X%] sensitivity ([+X] vs. CNN-LSTM baseline)

**Key finding:** [Summarize what worked]

See `experiments/improvement_summary.md` for detailed results.
```

- [ ] **Step 3: Update wiki/log.md**

Add entry:

```markdown
**2026-04-13 (Option 3: Improve Further)**
- Tested CNN+Attention hybrid, focal loss, larger models, feature engineering
- Best variant: [Model name] achieved [X%] sensitivity
- Key finding: [What was the bottleneck?]
```

- [ ] **Step 4: Commit final summary**

```bash
git add experiments/improvement_summary.md wiki/syntheses/architecture-comparison.md wiki/log.md
git commit -m "complete: Option 3 - Improve Further implementation

- Tested 4 hypotheses: CNN+Attention, class imbalance loss functions, model capacity, feature engineering
- Implemented focal loss and class-balanced loss for class imbalance
- Created CNN+Attention and larger CNN-LSTM variants
- Built hyperparameter sweep utility for systematic experimentation
- Achieved [X%] sensitivity improvement with [best variant]
- All results documented in experiments/ directory and wiki syntheses
"
```

---

## Self-Review

**Spec Coverage:**
- ✅ Hypothesis 1: CNN + Attention hybrid (Task 2)
- ✅ Hypothesis 2: Class imbalance loss functions (Task 1, Task 6)
- ✅ Hypothesis 3: Model capacity (Task 3, Task 7)
- ✅ Hypothesis 4: Feature engineering (Task 8)
- ✅ 2-3 improved architecture variants (CNN+Attention, Larger CNN-LSTM, Feature-BiLSTM)
- ✅ Hyperparameter sweep utility (Task 4)
- ✅ Detailed experimental results (Tasks 5-8)
- ✅ Analysis and documentation (Task 9)

**No Placeholders:** All experiments have concrete commands and expected comparisons.

**Consistency:** Model names, loss types, and metrics consistent throughout.

