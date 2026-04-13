"""Baseline Benchmark"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, confusion_matrix
import numpy as np
import json
import time
import sys
from pathlib import Path

# Add models directory to path
models_dir = Path(__file__).parent.parent
sys.path.insert(0, str(models_dir))

# Import models
from architectures.vanilla_lstm import VanillaLSTM
from architectures.bilstm import BiLSTM
from architectures.attention_bilstm import AttentionBiLSTM
from architectures.cnn_lstm import CNNLSTM

MODEL_REGISTRY = {
    "vanilla_lstm": VanillaLSTM,
    "bilstm": BiLSTM,
    "attention_bilstm": AttentionBiLSTM,
    "cnn_lstm": CNNLSTM,
}

print("\n" + "="*60)
print("BASELINE BENCHMARK")
print("="*60)

# Load data
base_path = Path(__file__).parent.parent.parent / 'results/tensors/chbmit'
X_train = torch.load(base_path / 'train/data.pt')
y_train = torch.load(base_path / 'train/labels.pt')
X_val = torch.load(base_path / 'val/data.pt')
y_val = torch.load(base_path / 'val/labels.pt')
X_test = torch.load(base_path / 'test/data.pt')
y_test = torch.load(base_path / 'test/labels.pt')

print(f"\nTrain: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

# Dataloaders
class_counts = torch.bincount(y_train.long())
weights = 1.0 / class_counts.float()
sample_weights = weights[y_train.long()]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, sampler=sampler)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

models_to_test = ['vanilla_lstm', 'bilstm', 'attention_bilstm', 'cnn_lstm']
results = {}

for model_name in models_to_test:
    print(f"\n{'='*60}\n  {model_name}\n{'='*60}")
    
    model = MODEL_REGISTRY[model_name](n_channels=16, seq_len=256, hidden_size=128, num_layers=2, dropout=0.3)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    
    pos_weight = torch.tensor([float(len(y_train) - y_train.sum()) / y_train.sum()])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    patience = 0
    start = time.time()
    
    for epoch in range(30):
        model.train()
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_b.float()), y_b.float().unsqueeze(1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                val_loss += criterion(model(X_b.float()), y_b.float().unsqueeze(1)).item()
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience = 0
            
            Path('results/checkpoints').mkdir(parents=True, exist_ok=True)
            torch.save({'model_state_dict': best_state}, f'results/checkpoints/{model_name}_best.pt')
        else:
            patience += 1
            if patience >= 7:
                print(f"Early stop at epoch {epoch}")
                break
    
    model.load_state_dict(best_state)
    train_time = time.time() - start
    
    # Test
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            probs = torch.sigmoid(model(X_b.float())).numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(y_b.numpy())
    
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= 0.5).astype(float)
    all_labels = np.array(all_labels)
    
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0,1]).ravel()
    
    results[model_name] = {
        'accuracy': float(accuracy_score(all_labels, all_preds)),
        'sensitivity': float(recall_score(all_labels, all_preds, zero_division=0)),
        'specificity': float(tn/(tn+fp)) if (tn+fp)>0 else 0.0,
        'f1': float(f1_score(all_labels, all_preds, zero_division=0)),
        'auc': float(roc_auc_score(all_labels, all_probs)),
        'time': round(train_time, 1)
    }
    
    print(f"Test → Acc: {results[model_name]['accuracy']:.3f} | Sens: {results[model_name]['sensitivity']:.3f} | F1: {results[model_name]['f1']:.3f}")

print(f"\n{'='*70}")
print("BASELINE RESULTS")
print(f"{'='*70}")
for name, m in sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True):
    print(f"{name:20} | Acc: {m['accuracy']:.3f} | Sens: {m['sensitivity']:.3f} | F1: {m['f1']:.3f} | AUC: {m['auc']:.3f}")

Path('results').mkdir(exist_ok=True)
with open('results/baseline_results.json', 'w') as f:
    json.dump(results, f, indent=2)
    
print(f"\n✓ Results saved: results/baseline_results.json\n")
