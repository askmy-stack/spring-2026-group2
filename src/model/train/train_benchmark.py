from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F

from dataloader.loader.loader import get_dataloaders
from model.models.registry import MODEL_REGISTRY
from model.models.registry import get_model
from model.eval.metrics import compute_binary_metrics, find_best_f1_threshold
from model.utils.seed import set_seed


class FocalLossWithLogits(torch.nn.Module):
    """Binary focal loss on logits with optional positive-class weighting."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, pos_weight: float | None = None):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.register_buffer(
            "pos_weight",
            torch.tensor([float(pos_weight)], dtype=torch.float32) if pos_weight is not None else None,
            persistent=False,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=self.pos_weight,
        )
        pt = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * ((1.0 - pt) ** self.gamma) * bce
        return loss.mean()


def collect_logits(model, loader, device):
    model.eval()
    all_logits, all_y = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)              # (B, 16, 256)
            y = y.to(device).float()      # (B,)
            logits = model(x)
            all_logits.append(logits.detach().cpu())
            all_y.append(y.detach().cpu())
    return torch.cat(all_logits).numpy(), torch.cat(all_y).numpy()


def train_one_epoch(model, loader, optimizer, loss_fn, device, scheduler=None, batch_scheduler: bool = False):
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device).float()

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # stabilizes
        optimizer.step()
        if batch_scheduler and scheduler is not None:
            scheduler.step()

        total += float(loss.detach().cpu()) * x.size(0)
        n += x.size(0)
    return total / max(1, n)


def estimate_pos_weight(train_dl) -> float:
    ys = []
    for _, y in train_dl:
        ys.append(y.detach().cpu().numpy())
    y = np.concatenate(ys).astype(int)
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    return n_neg / max(n_pos, 1.0)


def _flatten_config_dict(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Accept both flat and sectioned config structures and flatten to argparse keys.
    Supported sections: training, optimizer, loss, scheduler, model, output.
    """
    out: dict[str, Any] = {}
    for k, v in cfg.items():
        if isinstance(v, dict):
            if k in {"training", "optimizer", "loss", "scheduler", "model", "output"}:
                out.update(v)
            else:
                # Keep unknown sections isolated to avoid surprising overrides.
                continue
        else:
            out[k] = v
    return out


def _load_config_file(config_path: str) -> dict[str, Any]:
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("PyYAML is required for --config support. Install with: pip install pyyaml") from e
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a mapping at top level: {p}")
    return _flatten_config_dict(raw)


def main():
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=str, default="")
    boot_args, remaining = bootstrap.parse_known_args()

    config_defaults: dict[str, Any] = {}
    if boot_args.config:
        config_defaults = _load_config_file(boot_args.config)

    ap = argparse.ArgumentParser(parents=[bootstrap])
    ap.add_argument("--dataset", default="chbmit", choices=["chbmit", "siena"])
    ap.add_argument("--model", default="cnn_benchmark", choices=sorted(MODEL_REGISTRY.keys()))

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--loss", default="bce", choices=["bce", "focal"])
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--focal_alpha", type=float, default=0.25)
    ap.add_argument("--scheduler", default="plateau", choices=["plateau", "onecycle"])
    ap.add_argument("--early_stop_patience", type=int, default=5)
    ap.add_argument("--early_stop_min_delta", type=float, default=1e-4)

    # model hyperparams
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--base_channels", type=int, default=32)
    ap.add_argument("--F1", type=int, default=8)
    ap.add_argument("--D", type=int, default=2)
    ap.add_argument("--kernel_length", type=int, default=64)
    ap.add_argument("--save_metrics", action="store_true")
    ap.add_argument("--metrics_out", type=str, default="")
    ap.add_argument("--metrics_dir", type=str, default="")

    if config_defaults:
        ap.set_defaults(**config_defaults)
    args = ap.parse_args(remaining)
    set_seed(args.seed)
    device = torch.device(args.device)

    train_dl, val_dl, test_dl = get_dataloaders(args.dataset, batch_size=args.batch_size)

    n_channels = 16
    n_samples = 256

    def num_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    # build model
    if args.model == "cnn_benchmark":
        model = get_model(
            "cnn_benchmark",
            n_ch=n_channels,
            n_samples=n_samples,
            base_channels=args.base_channels,
            dropout=args.dropout,
        ).to(device)
    elif args.model == "cnn_improved":
        model = get_model(
            "cnn_improved",
            n_ch=n_channels,
            n_samples=n_samples,
            base_channels=args.base_channels,
            dropout=args.dropout,
        ).to(device)
    elif args.model == "cnn_mixture":
        model = get_model(
            "cnn_mixture",
            n_ch=n_channels,
            n_samples=n_samples,
            base_channels=args.base_channels,
            dropout=args.dropout,
        ).to(device)
    elif args.model == "cnn_multiscale":
        model = get_model(
            "cnn_multiscale",
            n_ch=n_channels,
            n_samples=n_samples,
            base_channels=args.base_channels,
            dropout=args.dropout,
        ).to(device)
    elif args.model == "eegnet":
        model = get_model(
            "eegnet",
            n_ch=n_channels,
            n_samples=n_samples,
            F1=args.F1,
            D=args.D,
            kernel_length=args.kernel_length,
            dropout=args.dropout,
        ).to(device)
    elif args.model == "eegnet_improved":
        model = get_model(
            "eegnet_improved",
            n_ch=n_channels,
            n_samples=n_samples,
            F1=args.F1,
            D=args.D,
            kernel_length=args.kernel_length,
            dropout=args.dropout,
        ).to(device)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    print(f"\nModel: {args.model}")
    print(f"Trainable params: {num_params(model):,}")

    pos_w = estimate_pos_weight(train_dl)
    if args.loss == "focal":
        loss_fn = FocalLossWithLogits(
            gamma=args.focal_gamma,
            alpha=args.focal_alpha,
            pos_weight=pos_w,
        ).to(device)
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=len(train_dl),
            pct_start=0.2,
            anneal_strategy="cos",
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_score = -1.0
    best_state = None
    best_threshold = 0.5
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(
            model,
            train_dl,
            optimizer,
            loss_fn,
            device,
            scheduler=scheduler,
            batch_scheduler=(args.scheduler == "onecycle"),
        )

        val_logits, val_y = collect_logits(model, val_dl, device)
        best_t, val_best = find_best_f1_threshold(val_y, logits=val_logits)
        val_t05 = compute_binary_metrics(val_y, logits=val_logits, threshold=0.5)

        # select by AUPRC (preferred) else F1
        score = val_best.auprc if np.isfinite(val_best.auprc) else val_best.f1
        if score > best_score + args.early_stop_min_delta:
            best_score = score
            best_threshold = best_t
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if args.scheduler == "plateau":
            scheduler.step(score)

        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  train_loss: {tr_loss:.4f}")
        print(f"  lr: {optimizer.param_groups[0]['lr']:.6g}")
        print(f"  val@bestF1 threshold={best_t:.2f}  metrics={val_best.as_dict()}")
        print(f"  val@0.50                 metrics={val_t05.as_dict()}")

        if bad_epochs >= args.early_stop_patience:
            print(f"  Early stopping: no improvement for {bad_epochs} epoch(s).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_logits, test_y = collect_logits(model, test_dl, device)
    test_metrics = compute_binary_metrics(test_y, logits=test_logits, threshold=best_threshold)

    print("\n==== FINAL TEST (best threshold from VAL) ====")
    print(test_metrics.as_dict())

    if args.save_metrics:
        payload = {
            "dataset": args.dataset,
            "model": args.model,
            "epochs_requested": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
            "device": str(args.device),
            "loss": str(args.loss),
            "scheduler": str(args.scheduler),
            "dropout": float(args.dropout),
            "base_channels": int(args.base_channels),
            "F1": int(args.F1),
            "D": int(args.D),
            "kernel_length": int(args.kernel_length),
            "final_test": test_metrics.as_dict(),
        }
        if args.metrics_out:
            out_path = Path(args.metrics_out)
        else:
            if args.metrics_dir:
                out_dir = Path(args.metrics_dir)
            else:
                out_dir = Path(__file__).resolve().parents[1] / "results" / "benchmark_metrics"
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = out_dir / f"{args.dataset}_{args.model}_{ts}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved metrics JSON: {out_path}")


if __name__ == "__main__":
    main()
