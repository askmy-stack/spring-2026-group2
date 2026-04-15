"""
run_benchmark.py - Improved Models Benchmark
==============================================
Generates realistic synthetic EEG data matching the pipeline format
(batch, 16 channels, 256 timesteps at 256 Hz) and trains/evaluates
all 5 improved LSTM architectures.

Synthetic data mimics key EEG properties:
  - Background: pink-ish noise with realistic channel correlations
  - Seizure: high-amplitude rhythmic 3-5 Hz oscillation (spike-wave)
    + high-frequency bursts (gamma 30-50 Hz)
    + inter-channel coherence increase

Usage:
    python run_benchmark.py
    python run_benchmark.py --epochs 30 --n_samples 3000
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix, roc_curve,
)

# Make sure architectures are importable from this script's location
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from architectures import MODEL_REGISTRY


# ─────────────────────────────────────────────────────────────
# Synthetic EEG generator
# ─────────────────────────────────────────────────────────────

def generate_eeg_data(
    n_samples: int = 4000,
    n_channels: int = 16,
    seq_len: int = 256,
    seizure_ratio: float = 0.3,
    sfreq: int = 256,
    seed: int = 42,
):
    """
    Generate synthetic EEG windows that mimic CHB-MIT characteristics.

    Background windows: band-limited pink noise (1/f spectrum), amplitude ~50 µV.
    Seizure windows:    rhythmic 3 Hz spike-wave + gamma bursts, amplitude ~150 µV,
                        increased inter-channel coherence.

    Returns
    -------
    X : torch.FloatTensor  (n_samples, n_channels, seq_len)
    y : torch.LongTensor   (n_samples,)
    """
    rng = np.random.default_rng(seed)
    n_seizure = int(n_samples * seizure_ratio)
    n_bg      = n_samples - n_seizure
    t = np.linspace(0, seq_len / sfreq, seq_len)

    def pink_noise(shape, rng):
        """Generate 1/f pink noise via FFT filtering."""
        white = rng.standard_normal(shape)
        fft   = np.fft.rfft(white, axis=-1)
        freqs = np.fft.rfftfreq(shape[-1])
        freqs[0] = 1e-6  # avoid division by zero at DC
        fft  *= (1.0 / np.sqrt(freqs))
        return np.fft.irfft(fft, n=shape[-1], axis=-1)

    def make_background(n, rng):
        base = pink_noise((n, n_channels, seq_len), rng)
        # Band-limit 1–50 Hz via FFT zero-out
        fft   = np.fft.rfft(base, axis=-1)
        freqs = np.fft.rfftfreq(seq_len, d=1.0 / sfreq)
        mask  = (freqs >= 1.0) & (freqs <= 50.0)
        fft  *= mask[np.newaxis, np.newaxis, :]
        sig   = np.fft.irfft(fft, n=seq_len, axis=-1)
        # Normalise to ~50 µV std
        std = sig.std(axis=-1, keepdims=True).clip(min=1e-6)
        sig = sig / std * 50.0
        # Add cross-channel correlation (common background reference)
        common = pink_noise((n, 1, seq_len), rng) * 15.0
        return (sig + common).astype(np.float32)

    def make_seizure(n, rng):
        base = pink_noise((n, n_channels, seq_len), rng)
        fft  = np.fft.rfft(base, axis=-1)
        freqs= np.fft.rfftfreq(seq_len, d=1.0 / sfreq)
        mask = (freqs >= 1.0) & (freqs <= 50.0)
        fft *= mask[np.newaxis, np.newaxis, :]
        sig  = np.fft.irfft(fft, n=seq_len, axis=-1)

        # 3 Hz spike-wave discharge (dominant seizure feature)
        spike_wave = np.sin(2 * np.pi * 3.0 * t)  # (seq_len,)
        # Add sharp spike at each cycle peak
        spike_wave += 0.5 * np.sin(2 * np.pi * 9.0 * t)

        # 40 Hz gamma burst (ictal high-frequency activity)
        gamma = np.sin(2 * np.pi * 40.0 * t) * 0.4

        # Per-channel amplitude variation: temporal channels louder
        ch_amp = rng.uniform(0.7, 1.3, (n, n_channels, 1))

        ictal = (spike_wave + gamma)[np.newaxis, np.newaxis, :]  # (1, 1, T)
        ictal = ictal * ch_amp  # (n, n_channels, T)

        # Increased coherence: add strong common ictal reference
        common_ictal = np.sin(2 * np.pi * 3.0 * t) * rng.uniform(30, 60, (n, 1, 1))

        sig = sig * 0.3 + ictal * 100.0 + common_ictal
        std = sig.std(axis=-1, keepdims=True).clip(min=1e-6)
        sig = sig / std * 150.0
        return sig.astype(np.float32)

    X_bg  = make_background(n_bg,      rng)
    X_sz  = make_seizure   (n_seizure, rng)

    X = np.concatenate([X_bg, X_sz], axis=0)
    y = np.array([0] * n_bg + [1] * n_seizure, dtype=np.int64)

    # Shuffle
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]

    return torch.from_numpy(X), torch.from_numpy(y)


def generate_feature_data(
    n_samples: int = 4000,
    n_features: int = 226,
    seizure_ratio: float = 0.3,
    seed: int = 42,
):
    """
    Generate synthetic 226-dim feature vectors matching FeatureBiLSTM input.
    Seizure windows have elevated power features and reduced entropy.
    """
    rng = np.random.default_rng(seed)
    n_seizure = int(n_samples * seizure_ratio)
    n_bg      = n_samples - n_seizure

    # Background: Gaussian noise centred at 0, std 1
    X_bg = rng.standard_normal((n_bg, n_features)).astype(np.float32)

    # Seizure: elevated mean in first 50 features (band power), reduced variance in rest
    X_sz = rng.standard_normal((n_seizure, n_features)).astype(np.float32)
    X_sz[:, :50] += rng.uniform(1.5, 3.0, (n_seizure, 50))  # elevated band power
    X_sz[:, 50:120] *= 0.5                                    # reduced entropy
    X_sz[:, 120:180] += rng.uniform(-1.0, 1.0, (n_seizure, 60))

    X = np.concatenate([X_bg, X_sz], axis=0)
    y = np.array([0] * n_bg + [1] * n_seizure, dtype=np.int64)
    idx = rng.permutation(len(y))
    return torch.from_numpy(X[idx]), torch.from_numpy(y[idx])


# ─────────────────────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────────────────────

def make_loaders(X, y, val_split, batch_size, seed):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_split, random_state=seed, stratify=y
    )
    n_pos = y_tr.sum().item()
    n_neg = len(y_tr) - n_pos
    pos_weight = n_neg / max(n_pos, 1)

    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr), batch_size=batch_size,
        shuffle=True, pin_memory=False,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=batch_size,
        shuffle=False, pin_memory=False,
    )
    return train_loader, val_loader, pos_weight


def find_optimal_threshold(y_true, y_prob):
    """Youden's J: threshold that maximises sensitivity + specificity."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    return float(thresholds[best_idx])


def train_and_evaluate(model, train_loader, val_loader, pos_weight,
                       epochs, patience, lr, device):
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight]).to(device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    best_f1   = -1.0
    best_metrics = None
    wait = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        for X_b, y_b in train_loader:
            X_b = X_b.to(device, dtype=torch.float32)
            y_b = y_b.to(device, dtype=torch.float32).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step(epoch)

        # ── Validate ──
        model.eval()
        all_probs, all_labels = [], []
        val_loss_sum, n = 0.0, 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b = X_b.to(device, dtype=torch.float32)
                y_b = y_b.to(device, dtype=torch.float32).unsqueeze(1)
                logits = model(X_b)
                val_loss_sum += criterion(logits, y_b).item()
                n += 1
                all_probs.append(torch.sigmoid(logits).cpu())
                all_labels.append(y_b.cpu())

        probs  = torch.cat(all_probs).numpy().flatten()
        labels = torch.cat(all_labels).numpy().flatten()

        # Optimal threshold via Youden's J
        thresh = find_optimal_threshold(labels, probs)
        preds  = (probs >= thresh).astype(float)

        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        metrics = {
            "accuracy":    float(accuracy_score(labels, preds)),
            "sensitivity": float(recall_score(labels, preds, zero_division=0)),
            "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            "precision":   float(precision_score(labels, preds, zero_division=0)),
            "f1":          float(f1_score(labels, preds, zero_division=0)),
            "auc_roc":     float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.0,
            "threshold":   round(thresh, 4),
            "val_loss":    val_loss_sum / n,
            "epoch":       epoch,
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        }

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_metrics = metrics.copy()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    best_metrics["train_time_sec"] = round(time.time() - start_time, 1)
    best_metrics["total_params"]   = sum(p.numel() for p in model.parameters())
    return best_metrics


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=20)
    p.add_argument("--n_samples",  type=int,   default=4000)
    p.add_argument("--batch_size", type=int,   default=64)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--hidden_size",type=int,   default=128)
    p.add_argument("--num_layers", type=int,   default=2)
    p.add_argument("--dropout",    type=float, default=0.3)
    p.add_argument("--patience",   type=int,   default=7)
    p.add_argument("--val_split",  type=float, default=0.2)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--save_dir",   type=str,   default="./benchmark_results")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )

    os.makedirs(args.save_dir, exist_ok=True)

    N_CHANNELS = 16
    SEQ_LEN    = 256
    N_FEATURES = 226

    print(f"\n{'='*65}")
    print(f"  EEG LSTM Improved Models Benchmark")
    print(f"  Device: {device} | Epochs: {args.epochs} | Samples: {args.n_samples}")
    print(f"{'='*65}")

    # ── Generate data ──
    print("\nGenerating synthetic EEG data...")
    X_raw, y_raw = generate_eeg_data(
        n_samples=args.n_samples, n_channels=N_CHANNELS,
        seq_len=SEQ_LEN, seed=args.seed
    )
    X_feat, y_feat = generate_feature_data(
        n_samples=args.n_samples, n_features=N_FEATURES, seed=args.seed
    )
    n_sz = y_raw.sum().item()
    print(f"  Raw EEG shape:  {tuple(X_raw.shape)}  "
          f"(seizure={n_sz}, background={len(y_raw)-n_sz})")
    print(f"  Feature shape:  {tuple(X_feat.shape)}")

    raw_train,  raw_val,  raw_pw  = make_loaders(X_raw,  y_raw,  args.val_split, args.batch_size, args.seed)
    feat_train, feat_val, feat_pw = make_loaders(X_feat, y_feat, args.val_split, args.batch_size, args.seed)

    # ── Also print BASELINE for reference ──
    baseline_path = os.path.join(
        os.path.dirname(__file__), "baseline", "results", "baseline_results.json"
    )
    baseline = {}
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline = json.load(f)

    # ── Train all 5 models ──
    all_results = {}
    model_configs = {
        "vanilla_lstm":     {"n_channels": N_CHANNELS, "seq_len": SEQ_LEN},
        "bilstm":           {"n_channels": N_CHANNELS, "seq_len": SEQ_LEN},
        "attention_bilstm": {"n_channels": N_CHANNELS, "seq_len": SEQ_LEN},
        "cnn_lstm":         {"n_channels": N_CHANNELS, "seq_len": SEQ_LEN},
        "feature_bilstm":   {"n_features": N_FEATURES, "seq_len": 10},
    }

    for model_name, extra_kwargs in model_configs.items():
        print(f"\n{'─'*65}")
        print(f"  Training: {model_name}")
        print(f"{'─'*65}")

        ModelClass = MODEL_REGISTRY[model_name]
        # attention_bilstm: reduce seq_len to 64 via avg-pool so MHA stays fast on CPU
        if model_name == "attention_bilstm":
            extra_kwargs = dict(extra_kwargs)
            extra_kwargs["seq_len"] = SEQ_LEN // 4  # 64 steps after downsampling

        model = ModelClass(
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            **extra_kwargs,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        # For attention_bilstm: downsample raw EEG from 256 -> 64 steps via avg-pool
        if model_name == "attention_bilstm":
            X_attn = torch.nn.functional.avg_pool1d(X_raw, kernel_size=4, stride=4)
            at, av, apw = make_loaders(X_attn, y_raw, args.val_split, args.batch_size, args.seed)
            loader_train, loader_val, pw = at, av, apw
        elif model_name == "feature_bilstm":
            loader_train, loader_val, pw = feat_train, feat_val, feat_pw
        else:
            loader_train, loader_val, pw = raw_train, raw_val, raw_pw

        metrics = train_and_evaluate(
            model, loader_train, loader_val, pw,
            args.epochs, args.patience, args.lr, device,
        )
        all_results[model_name] = metrics

        base = baseline.get(model_name, {})
        print(f"  Best epoch:  {metrics['epoch']}")
        print(f"  Threshold:   {metrics['threshold']:.3f} (Youden's J, was 0.500)")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        f1_delta  = metrics['f1']      - base.get('f1', 0)
        sen_delta = metrics['sensitivity'] - base.get('sensitivity', 0)
        auc_delta = metrics['auc_roc'] - base.get('auc', 0)
        base_sen = base.get('sensitivity')
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}  "
              f"(baseline={base_sen:.3f if base_sen is not None else 'N/A'}  Δ={sen_delta:+.3f})")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        base_f1 = base.get('f1')
        print(f"  F1 Score:    {metrics['f1']:.4f}  "
              f"(baseline={base_f1:.3f if base_f1 is not None else 'N/A'}  Δ={f1_delta:+.3f})")
        base_auc = base.get('auc')
        print(f"  AUC-ROC:     {metrics['auc_roc']:.4f}  "
              f"(baseline={base_auc:.3f if base_auc is not None else 'N/A'}  Δ={auc_delta:+.3f})")
        print(f"  Confusion:   TP={metrics['tp']} FP={metrics['fp']} "
              f"TN={metrics['tn']} FN={metrics['fn']}")
        print(f"  Train time:  {metrics['train_time_sec']}s")

    # ── Comparison Table ──
    print(f"\n\n{'='*85}")
    print(f"  FINAL COMPARISON — IMPROVED vs BASELINE")
    print(f"{'='*85}")
    header = (f"{'Model':<22} | {'F1':>6} | {'Δ F1':>7} | "
              f"{'Sens':>6} | {'ΔSens':>7} | "
              f"{'Spec':>6} | {'AUC':>6} | {'ΔAUC':>7} | {'Params':>10}")
    print(header)
    print("─" * len(header))

    sorted_results = sorted(all_results.items(), key=lambda x: x[1]["f1"], reverse=True)
    for name, m in sorted_results:
        base = baseline.get(name, {})
        df1  = m["f1"]          - base.get("f1",          0)
        dsen = m["sensitivity"] - base.get("sensitivity",  0)
        dauc = m["auc_roc"]     - base.get("auc",          0)
        print(
            f"{name:<22} | {m['f1']:>6.3f} | {df1:>+7.3f} | "
            f"{m['sensitivity']:>6.3f} | {dsen:>+7.3f} | "
            f"{m['specificity']:>6.3f} | {m['auc_roc']:>6.3f} | {dauc:>+7.3f} | "
            f"{m['total_params']:>10,}"
        )

    best_name, best_m = sorted_results[0]
    print(f"\n  >>> Best model by F1: {best_name} "
          f"(F1={best_m['f1']:.4f}, AUC={best_m['auc_roc']:.4f}, "
          f"Sens={best_m['sensitivity']:.4f})")

    # Save results
    out_path = os.path.join(args.save_dir, "improved_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {out_path}\n")


if __name__ == "__main__":
    main()
