from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch


@dataclass
class BinaryMetrics:
    threshold: float
    auroc: float
    auprc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    specificity: float
    tp: int
    fp: int
    tn: int
    fn: int

    def as_dict(self) -> Dict[str, float]:
        return {
            "threshold": float(self.threshold),
            "auroc": float(self.auroc),
            "auprc": float(self.auprc),
            "accuracy": float(self.accuracy),
            "precision": float(self.precision),
            "recall": float(self.recall),
            "f1": float(self.f1),
            "specificity": float(self.specificity),
            "tp": int(self.tp),
            "fp": int(self.fp),
            "tn": int(self.tn),
            "fn": int(self.fn),
        }


def _to_numpy_1d(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x).reshape(-1)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


def _confusion(y: np.ndarray, yhat: np.ndarray) -> Tuple[int, int, int, int]:
    y = y.astype(int)
    yhat = yhat.astype(int)
    tp = int(np.sum((y == 1) & (yhat == 1)))
    fp = int(np.sum((y == 0) & (yhat == 1)))
    tn = int(np.sum((y == 0) & (yhat == 0)))
    fn = int(np.sum((y == 1) & (yhat == 0)))
    return tp, fp, tn, fn


def compute_binary_metrics(y_true, *, logits=None, probs=None, threshold: float = 0.5) -> BinaryMetrics:
    y = _to_numpy_1d(y_true).astype(int)

    if probs is None:
        if logits is None:
            raise ValueError("Provide logits or probs.")
        p = sigmoid(_to_numpy_1d(logits))
    else:
        p = _to_numpy_1d(probs).astype(np.float64)

    yhat = (p >= float(threshold)).astype(int)
    tp, fp, tn, fn = _confusion(y, yhat)

    acc = _safe_div(tp + tn, tp + tn + fp + fn)
    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    spec = _safe_div(tn, tn + fp)
    f1 = _safe_div(2 * prec * rec, prec + rec)

    auroc = float("nan")
    auprc = float("nan")
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        if len(np.unique(y)) == 2:
            auroc = float(roc_auc_score(y, p))
            auprc = float(average_precision_score(y, p))
    except Exception:
        pass

    return BinaryMetrics(
        threshold=float(threshold),
        auroc=float(auroc),
        auprc=float(auprc),
        accuracy=float(acc),
        precision=float(prec),
        recall=float(rec),
        f1=float(f1),
        specificity=float(spec),
        tp=tp, fp=fp, tn=tn, fn=fn,
    )


def find_best_f1_threshold(y_true, *, logits=None, probs=None, n_steps: int = 101) -> Tuple[float, BinaryMetrics]:
    y = _to_numpy_1d(y_true).astype(int)
    if probs is None:
        if logits is None:
            raise ValueError("Provide logits or probs.")
        p = sigmoid(_to_numpy_1d(logits))
    else:
        p = _to_numpy_1d(probs).astype(np.float64)

    best_t = 0.5
    best_m = None
    best_f = -1.0

    for t in np.linspace(0.0, 1.0, int(n_steps)):
        m = compute_binary_metrics(y, probs=p, threshold=float(t))
        if m.f1 > best_f:
            best_f = m.f1
            best_t = float(t)
            best_m = m

    assert best_m is not None
    return best_t, best_m