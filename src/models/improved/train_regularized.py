import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, recall_score
from bilstm import BiLSTM
from vanilla_lstm import VanillaLSTM
import time

print("\n" + "="*60)
print("REGULARIZED TRAINING")
print("="*60)

# Device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Device: {device}")

# Load data
X = torch.load('../results/tensors/chbmit/train/data.pt')
y = torch.load('../results/tensors/chbmit/train/labels.pt')

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Weighted sampler
class_counts = torch.bincount(y_train.long())
weights = 1.0 / class_counts.float()
sample_weights = weights[y_train.long()]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

# Dataloaders
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, sampler=sampler)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32, shuffle=False)

# Train with strong regularization
configs = [
    {'name': 'vanilla_reg', 'model': VanillaLSTM, 'hidden': 128, 'layers': 2, 'dropout': 0.5, 'wd': 1e-3},
    {'name': 'bilstm_reg', 'model': BiLSTM, 'hidden': 128, 'layers': 2, 'dropout': 0.5, 'wd': 1e-3},
]

for config in configs:
    print(f"\n{'='*60}")
    print(f"Training: {config['name']}")
    print(f"{'='*60}")
    
    n_channels = X.shape[1]
    seq_len = X.shape[2]
    model = config['model'](
        n_channels=n_channels, seq_len=seq_len,
        hidden_size=config['hidden'],
        num_layers=config['layers'],
        dropout=config['dropout']
    ).to(device)

    # Loss with class weighting (guard against zero positive samples)
    n_pos = y_train.sum().item()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Optimizer with weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=config['wd'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_val_loss = float('inf')
    patience = 0
    max_patience = 7
    
    for epoch in range(30):
        # Train
        model.train()
        for X_b, y_b in train_loader:
            X_b = X_b.to(device, dtype=torch.float32)
            y_b = y_b.to(device, dtype=torch.float32).unsqueeze(1)
            
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # Validate
        model.eval()
        val_losses = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b = X_b.to(device, dtype=torch.float32)
                y_b = y_b.to(device, dtype=torch.float32).unsqueeze(1)

                logits = model(X_b)
                val_losses.append(criterion(logits, y_b).item())

                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                all_probs.extend(probs)
                all_labels.extend(y_b.cpu().numpy().flatten())
        
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)
        
        preds = (np.array(all_probs) >= 0.5).astype(float)
        acc = accuracy_score(all_labels, preds)
        f1 = f1_score(all_labels, preds, zero_division=0)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch:2d} | Val Loss: {val_loss:.4f} | Acc: {acc:.3f} | F1: {f1:.3f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_metrics': {'loss': val_loss, 'acc': acc, 'f1': f1}
            }, f'../checkpoints/{config["name"]}_best.pt')
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    print(f"\nBest Val Loss: {best_val_loss:.4f}")

print("\n" + "="*60)
print("REGULARIZED TRAINING COMPLETE")
print("="*60)
