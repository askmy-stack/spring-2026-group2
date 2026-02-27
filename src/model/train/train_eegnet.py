from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.models.eegnet import EEGNet
from model.data.window_dataset import EEGWindowEDFDataset
from model.eval.metrics import find_best_f1_threshold, compute_binary_metrics
from model.utils.seed import set_seed


def collect_logits(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits = []
    all_y = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)            # (B,C,T)
            y = y.to(device)
            logits = model(x)           # (B,)
            all_logits.append(logits.detach().cpu())
            all_y.append(y.detach().cpu())
    logits = torch.cat(all_logits).numpy()
    y = torch.cat(all_y).numpy()
    return logits, y


def train_one_epoch(model, loader, optimizer, loss_fn, device) -> float:
    model.train()
    total = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        total += float(loss.detach().cpu()) * x.size(0)
        n += x.size(0)
    return total / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", default=".", help="repo root so CSV relative paths resolve")
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--test_csv", required=True)

    ap.add_argument("--n_channels", type=int, default=16)
    ap.add_argument("--n_samples", type=int, required=True, help="samples per window after crop (e.g., 1024)")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # EEGNet params
    ap.add_argument("--F1", type=int, default=8)
    ap.add_argument("--D", type=int, default=2)
    ap.add_argument("--F2", type=int, default=0, help="0 => auto (F1*D)")
    ap.add_argument("--kernel_length", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.25)

    # data
    ap.add_argument("--preload_raw", action="store_true", help="cache full EDF in RAM (can be heavy)")
    ap.add_argument("--target_sfreq", type=float, default=0.0, help="0 => no resample")

    args = ap.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)

    target_sfreq = None if args.target_sfreq <= 0 else float(args.target_sfreq)
    F2 = None if args.F2 == 0 else int(args.F2)

    train_ds = EEGWindowEDFDataset(
        args.train_csv, repo_root=args.repo_root, n_channels=args.n_channels,
        target_sfreq=target_sfreq, expected_samples=args.n_samples,
        preload_raw=args.preload_raw, apply_norm=True
    )
    val_ds = EEGWindowEDFDataset(
        args.val_csv, repo_root=args.repo_root, n_channels=args.n_channels,
        target_sfreq=target_sfreq, expected_samples=args.n_samples,
        preload_raw=args.preload_raw, apply_norm=True
    )
    test_ds = EEGWindowEDFDataset(
        args.test_csv, repo_root=args.repo_root, n_channels=args.n_channels,
        target_sfreq=target_sfreq, expected_samples=args.n_samples,
        preload_raw=args.preload_raw, apply_norm=True
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = EEGNet(
        n_ch=args.n_channels,
        n_samples=args.n_samples,
        F1=args.F1,
        D=args.D,
        F2=F2,
        kernel_length=args.kernel_length,
        dropout=args.dropout,
    ).to(device)

    # pos_weight for imbalance
    y_train = np.array([r.label for r in train_ds.rows], dtype=np.int64)
    n_pos = float((y_train == 1).sum())
    n_neg = float((y_train == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32, device=device)

    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_val_auprc = -1.0
    best_state = None
    best_threshold = 0.5

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)

        val_logits, val_y = collect_logits(model, val_loader, device)
        best_t, val_best = find_best_f1_threshold(val_y, logits=val_logits)
        # also track AUPRC for model selection
        val_metrics_t05 = compute_binary_metrics(val_y, logits=val_logits, threshold=0.5)
        # --- sanity: check if scores are inverted ---
        inv_auc = compute_binary_metrics(val_y, logits=-val_logits, threshold=0.5).auroc
        inv_auprc = compute_binary_metrics(val_y, logits=-val_logits, threshold=0.5).auprc
        print(f"  sanity inverted: auroc={inv_auc:.4f}  auprc={inv_auprc:.4f}")
        # choose which selection metric you want:
        select_score = val_best.auprc if np.isfinite(val_best.auprc) else val_best.f1

        if select_score > best_val_auprc:
            best_val_auprc = select_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_threshold = best_t

        scheduler.step(select_score)

        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  train_loss: {tr_loss:.4f}")
        print(f"  val@bestF1 threshold={best_t:.2f}  metrics={val_best.as_dict()}")
        print(f"  val@0.50                 metrics={val_metrics_t05.as_dict()}")

    # restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # final test
    test_logits, test_y = collect_logits(model, test_loader, device)
    test_metrics = compute_binary_metrics(test_y, logits=test_logits, threshold=best_threshold)
    print("\n==== FINAL TEST (best threshold from VAL) ====")
    print(test_metrics.as_dict())

    # save checkpoint
    out_dir = Path("checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "eegnet_best.pt"
    torch.save(
        {
            "model": "EEGNet",
            "state_dict": model.state_dict(),
            "best_threshold": best_threshold,
            "n_channels": args.n_channels,
            "n_samples": args.n_samples,
            "args": vars(args),
        },
        ckpt_path,
    )
    print(f"\nSaved: {ckpt_path}")


if __name__ == "__main__":
    main()