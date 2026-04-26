"""
Improved Benchmark Trainer — trains im1..im7 with the full upgrade stack.

Pipeline per improved model:
    for seed in seeds:
      for fold_idx, (train_idx, aux_val_idx) in kfold(train):
        model = IM_i(**config)
        train(model) with augmentation + MixUp + focal BCE + warmup-cosine LR
                     + AMP + F1-early-stop on aux_val
        capture SWA weights over the last 25% of epochs
        finalise SWA (update_bn) and save sub-run state_dict
    average all sub-run state_dicts -> final_model
    eval final_model with TTA on fixed val -> tune threshold
    eval final_model with TTA on fixed test -> report metrics
    save unified-schema checkpoint to
        src/models/improved_lstm_models/checkpoints/<im_i>_best.pt
    (sub-run state_dicts go under checkpoints/sub_runs/ and are
    skipped by the ensemble's _discover_checkpoints.)

Usage:
    python -m src.models.improved_lstm_models.training.train_improved_benchmark \\
        --model im7_attention_lstm --data_path $DATA

    python -m src.models.improved_lstm_models.training.train_improved_benchmark \\
        --model all --data_path $DATA
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.lstm_benchmark_models.train_baseline import load_config
from src.models.utils.callbacks import clip_gradients
from src.models.utils.checkpoint import save_checkpoint
from src.models.utils.metrics import (
    compute_f1_score, compute_auc_roc, compute_sensitivity,
    compute_specificity, find_optimal_threshold,
)
from src.models.improved_lstm_models.augmentation import EEGAugmentation
from src.models.improved_lstm_models.architectures.improved_benchmarks import (
    IMPROVED_REGISTRY, get_improved_model,
)
from .mixup import mixup_batch, MixUpBCELoss
from .swa import SWAHook
from .tta import predict_with_tta
from .kfold import subject_wise_kfold, load_train_tensors

logger = logging.getLogger(__name__)

DEFAULT_CKPT_DIR = Path("src/models/improved_lstm_models/checkpoints")
SUBRUN_SUBDIR = "sub_runs"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True,
                        help=f"One of {sorted(IMPROVED_REGISTRY.keys())} or 'all'.")
    parser.add_argument("--data_path", type=Path, required=True,
                        help="Tensor splits dir with train/val/test/{data,labels}.pt.")
    parser.add_argument("--config", default="src/models/config.yaml")
    parser.add_argument("--ckpt_dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--dry_run", action="store_true",
                        help="Reduce epochs / folds / seeds for a fast smoke run.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    config = load_config(Path(args.config))
    selected = _expand_model_arg(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device=%s; models=%s", device, selected)

    data, labels, subjects = load_train_tensors(args.data_path)
    if subjects is None:
        # Prominent banner: without subject_ids.pt the aux_f1 metrics reported
        # per-fold are window-wise (adjacent 1-second windows from the same
        # recording end up in both train and aux-val) and will be optimistic
        # by ~0.4 F1 vs a real subject-wise holdout. The final test metric
        # (fixed test split) is unaffected, but per-fold logs should be read
        # as "training-signal" only, not generalisation.
        logger.warning("=" * 72)
        logger.warning("subject_ids.pt not found under %s/train/", args.data_path)
        logger.warning("Falling back to stratified WINDOW-wise K-fold.")
        logger.warning("Per-fold aux_f1 metrics will be optimistic (data leakage).")
        logger.warning("Regenerate tensors with per-window subject ids for honest CV.")
        logger.warning("=" * 72)
    val_loader, test_loader = _build_val_test_loaders(args.data_path, config)

    for model_name in selected:
        logger.info("========== Training %s ==========", model_name)
        _train_one_improved(
            model_name, args.data_path, data, labels, subjects,
            val_loader, test_loader, config, args.ckpt_dir, device,
            dry_run=args.dry_run,
        )


# ---------------------------------------------------------------------------
# Top-level driver for a single im_i
# ---------------------------------------------------------------------------


def _train_one_improved(
    model_name: str,
    data_path: Path,
    data: torch.Tensor,
    labels: torch.Tensor,
    subjects: Optional[np.ndarray],
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: Dict,
    ckpt_dir: Path,
    device: torch.device,
    dry_run: bool = False,
) -> Dict:
    """Run K-fold × seeds for ``model_name``; save aggregated checkpoint."""
    cfg = _resolve_im_config(model_name, config)
    seeds, n_splits = cfg["seeds"], cfg["k_folds"]
    if dry_run:
        seeds, n_splits = seeds[:1], 2
        cfg["num_epochs"] = min(cfg["num_epochs"], 3)

    sub_states: List[Dict[str, torch.Tensor]] = []
    subrun_dir = ckpt_dir / SUBRUN_SUBDIR
    subrun_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        _set_seed(seed)
        folds = list(subject_wise_kfold(data_path, n_splits=n_splits, seed=seed))
        for fold_idx, (tr_idx, va_idx) in enumerate(folds):
            logger.info("[%s] seed=%d fold=%d/%d", model_name, seed, fold_idx + 1, n_splits)
            state = _train_one_subrun(
                model_name, cfg, data, labels, tr_idx, va_idx, device,
            )
            sub_states.append(state)
            sub_path = subrun_dir / f"{model_name}_seed{seed}_fold{fold_idx}.pt"
            torch.save(state, sub_path)
            logger.info("  saved sub-run -> %s", sub_path)

    # Aggregate into a final model via state-dict averaging.
    avg_state = _average_state_dicts(sub_states)
    final_model = _build_model(model_name, cfg).to(device)
    final_model.load_state_dict(avg_state)

    # TTA-evaluated probs on the fixed val/test splits.
    val_probs, val_labels_np = _predict_probs(final_model, val_loader, device, cfg)
    test_probs, test_labels_np = _predict_probs(final_model, test_loader, device, cfg)
    threshold = find_optimal_threshold(val_labels_np, val_probs, objective="f1")
    val_metrics = _metrics(val_labels_np, val_probs, threshold)
    test_metrics = _metrics(test_labels_np, test_probs, threshold)
    logger.info("[%s] val=%s  test=%s  threshold=%.3f",
                model_name, val_metrics, test_metrics, threshold)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{model_name}_best.pt"
    save_checkpoint(
        ckpt_path, final_model,
        model_config=_model_kwargs(model_name, cfg),
        epoch=cfg["num_epochs"],
        val_metrics=val_metrics,
        optimal_threshold=threshold,
    )
    (ckpt_dir / f"{model_name}_metrics.json").write_text(json.dumps({
        "val": val_metrics, "test": test_metrics, "threshold": threshold,
        "n_sub_runs": len(sub_states),
    }, indent=2))
    return test_metrics


# ---------------------------------------------------------------------------
# Sub-run training loop
# ---------------------------------------------------------------------------


def _train_one_subrun(
    model_name: str,
    cfg: Dict,
    data: torch.Tensor,
    labels: torch.Tensor,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Train one (seed × fold); return best-by-aux-F1 state_dict (CPU)."""
    model = _build_model(model_name, cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"],
    )
    scheduler = _warmup_cosine(optimizer, cfg)
    criterion = MixUpBCELoss(
        pos_weight=torch.tensor(cfg["pos_weight"], device=device),
        label_smoothing=cfg["label_smoothing"],
    )
    augmenter = EEGAugmentation(p=cfg["aug_prob"]).to(device)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    swa = SWAHook(model, start_frac=cfg["swa_start_frac"], total_epochs=cfg["num_epochs"])

    tr_loader = _fold_loader(data, labels, tr_idx, cfg["batch_size"], shuffle=True)
    aux_loader = _fold_loader(data, labels, va_idx, cfg["batch_size"], shuffle=False)

    best_f1, best_state = -1.0, None
    no_improve, patience = 0, cfg["early_stopping_patience"]
    for epoch in range(cfg["num_epochs"]):
        _train_epoch(model, tr_loader, augmenter, criterion, optimizer, cfg, device, scaler)
        scheduler.step()
        swa.step(epoch, model)
        aux_probs, aux_labels = _collect_probs_basic(model, aux_loader, device)
        t = find_optimal_threshold(aux_labels, aux_probs, objective="f1")
        aux_f1 = compute_f1_score(aux_labels, (aux_probs >= t).astype(int))
        logger.info("  epoch %d/%d aux_f1=%.4f (t=%.2f)", epoch + 1, cfg["num_epochs"], aux_f1, t)
        if aux_f1 > best_f1 + 1e-4:
            best_f1 = aux_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("  early stop at epoch %d (best aux_f1=%.4f)", epoch + 1, best_f1)
                break

    # Prefer SWA weights if SWA activated — seeds final_model in _train_one_improved
    swa_state = swa.state_dict()
    if swa_state is not None:
        try:
            swa.finalise(tr_loader, device)
        except Exception as exc:  # update_bn can fail on models without BN
            logger.warning("SWA update_bn skipped: %s", exc)
        return swa_state  # SWA weights typically beat best-aux-F1 weights
    return best_state if best_state is not None else {
        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
    }


def _train_epoch(
    model: nn.Module, loader: DataLoader, augmenter: EEGAugmentation,
    criterion: MixUpBCELoss, optimizer: torch.optim.Optimizer,
    cfg: Dict, device: torch.device, scaler: torch.amp.GradScaler,
) -> None:
    """Train for one epoch with augmentation + MixUp + AMP."""
    model.train()
    augmenter.train()
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        x = augmenter(x)
        mx, ya, yb, lam = mixup_batch(x, y, alpha=cfg["mixup_alpha"])
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=(device.type == "cuda")):
            logits = model(mx).squeeze(-1)
            loss = criterion(logits, ya, yb, lam)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_gradients(model, max_norm=cfg["gradient_clip"])
        scaler.step(optimizer)
        scaler.update()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_model(model_name: str, cfg: Dict) -> nn.Module:
    """Instantiate IM_i with config-driven kwargs."""
    return get_improved_model(model_name, **_model_kwargs(model_name, cfg))


def _model_kwargs(model_name: str, cfg: Dict) -> Dict:
    """Kwargs passed to the im_i constructor (filtered to the class signature).

    Each IM_i has a different constructor signature (e.g. im1/im2/im4 don't
    take num_heads; only im7 takes num_attn_blocks). We introspect the target
    class to keep only accepted kwargs, so the same cfg dict works for any
    model without a hardcoded per-model whitelist.
    """
    import inspect
    cls = IMPROVED_REGISTRY[model_name]
    accepted = set(inspect.signature(cls.__init__).parameters.keys()) - {"self"}
    return {k: v for k, v in cfg.items() if k in accepted}


def _resolve_im_config(model_name: str, config: Dict) -> Dict:
    """Merge training defaults, improved_benchmark.common, and per-model overrides."""
    training = config["training"]
    focal = config.get("focal_loss", {})
    im_root = config.get("models", {}).get("improved_benchmark", {})
    common = im_root.get("common", {})
    per_model = im_root.get(model_name, {})
    data_cfg = config["data"]
    cfg = {
        # Data / model geometry
        "n_channels": data_cfg["n_channels"],
        "time_steps": data_cfg["time_steps"],
        # Sensible defaults if improved_benchmark section isn't in config.yaml yet
        "hidden_size": 192,
        "num_layers": 2,
        "num_heads": 8,
        "dropout": 0.4,
        "stochastic_depth": 0.1,
        # Training
        "batch_size": training["batch_size"],
        "num_epochs": training["num_epochs"],
        "learning_rate": 3.0e-4,
        "weight_decay": 0.05,
        "pos_weight": training.get("pos_weight", 1.5),
        "label_smoothing": training.get("label_smoothing", 0.1),
        "aug_prob": training.get("aug_prob", 0.5),
        "gradient_clip": training.get("gradient_clip", 1.0),
        "warmup_epochs": training.get("warmup_epochs", 5),
        "early_stopping_patience": training.get("early_stopping_patience", 15),
        # Improved-benchmark specific
        "mixup_alpha": 0.2,
        "swa_start_frac": 0.75,
        "k_folds": 3,
        "seeds": [42, 43, 44],
        "tta_shifts": [-5, 0, 5],
    }
    cfg.update(common)
    cfg.update(per_model)
    return cfg


def _warmup_cosine(optimizer: torch.optim.Optimizer, cfg: Dict):
    """Linear warmup + cosine decay to 0."""
    warmup, total = cfg["warmup_epochs"], cfg["num_epochs"]

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            return float(epoch + 1) / max(warmup, 1)
        progress = (epoch - warmup) / max(total - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _fold_loader(
    data: torch.Tensor, labels: torch.Tensor, idx: np.ndarray,
    batch_size: int, shuffle: bool,
) -> DataLoader:
    """DataLoader over a subset of the training tensors."""
    ds = TensorDataset(data[idx], labels[idx].float())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=torch.cuda.is_available())


def _build_val_test_loaders(data_path: Path, config: Dict) -> Tuple[DataLoader, DataLoader]:
    """Fixed val and test loaders (no augmentation, no shuffle)."""
    bs = config["training"]["batch_size"]
    pin = torch.cuda.is_available()

    def _split(name: str) -> TensorDataset:
        x = torch.load(data_path / name / "data.pt", weights_only=True).float()
        y = torch.load(data_path / name / "labels.pt", weights_only=True).float().squeeze()
        return TensorDataset(x, y)

    return (
        DataLoader(_split("val"), batch_size=bs, shuffle=False, pin_memory=pin),
        DataLoader(_split("test"), batch_size=bs, shuffle=False, pin_memory=pin),
    )


def _predict_probs(
    model: nn.Module, loader: DataLoader, device: torch.device, cfg: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (probs, labels) with TTA averaging."""
    model.eval()
    probs_all, labels_all = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            p = predict_with_tta(model, x, shifts=cfg["tta_shifts"]).cpu().numpy()
            probs_all.append(np.atleast_1d(p))
            labels_all.append(np.atleast_1d(y.numpy()))
    return np.concatenate(probs_all), np.concatenate(labels_all)


def _collect_probs_basic(
    model: nn.Module, loader: DataLoader, device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Plain (no-TTA) probs for cheap per-epoch aux-F1 checks."""
    model.eval()
    probs_all, labels_all = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x).squeeze(-1)
            probs_all.append(np.atleast_1d(torch.sigmoid(logits).cpu().numpy()))
            labels_all.append(np.atleast_1d(y.numpy()))
    return np.concatenate(probs_all), np.concatenate(labels_all)


def _average_state_dicts(states: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Average N identically-shaped state dicts elementwise."""
    if not states:
        raise ValueError("Cannot average empty state_dict list")
    avg = copy.deepcopy(states[0])
    for k in avg:
        if not torch.is_floating_point(avg[k]):
            continue  # skip integer buffers
        stacked = torch.stack([s[k].float() for s in states], dim=0)
        avg[k] = stacked.mean(dim=0).to(avg[k].dtype)
    return avg


def _metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    """F1/AUROC/sens/spec at the given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "f1": compute_f1_score(y_true, y_pred),
        "auroc": compute_auc_roc(y_true, y_prob),
        "sens": compute_sensitivity(y_true, y_pred),
        "spec": compute_specificity(y_true, y_pred),
    }


def _set_seed(seed: int) -> None:
    """Deterministic seeding across random/numpy/torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _expand_model_arg(arg: str) -> List[str]:
    """Resolve 'all' or a single model name to a concrete list."""
    if arg == "all":
        return sorted(IMPROVED_REGISTRY.keys())
    if arg not in IMPROVED_REGISTRY:
        raise SystemExit(f"Unknown model '{arg}'. Options: {sorted(IMPROVED_REGISTRY.keys())}")
    return [arg]


if __name__ == "__main__":
    main()
