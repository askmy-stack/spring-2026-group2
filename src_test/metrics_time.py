from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import mne


def _kurtosis(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size < 4:
        return float("nan")
    mu = np.mean(x)
    v = np.mean((x - mu) ** 2)
    if v <= 0:
        return float("nan")
    m4 = np.mean((x - mu) ** 4)
    return float(m4 / (v ** 2))


def compute_epoch_stats(epochs: mne.Epochs) -> List[Dict[str, Any]]:
    """
    Returns epoch-level stats (aggregated across channels):
      - variance, ptp, kurtosis
    """
    X = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    rows: List[Dict[str, Any]] = []
    for i in range(X.shape[0]):
        e = X[i]
        var = float(np.var(e))
        ptp = float(np.ptp(e))  # in Volts
        kurt = float(np.mean([_kurtosis(e[ch]) for ch in range(e.shape[0])]))
        rows.append({
            "epoch_idx": i,
            "tmin_sec": float(epochs.events[i, 0] / epochs.info["sfreq"]) if epochs.events is not None else float("nan"),
            "var_V2": var,
            "ptp_V": ptp,
            "ptp_uV": ptp * 1e6,
            "mean_kurtosis": kurt,
        })
    return rows


def save_epoch_stats_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("epoch_idx,tmin_sec,var_V2,ptp_V,ptp_uV,mean_kurtosis\n")
        return

    cols = list(rows[0].keys())
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(str(r[c]) for c in cols))
    out_path.write_text("\n".join(lines))
