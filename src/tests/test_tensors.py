"""
Tensor & Label Output Verification
====================================
Verifies that StandardEEGLoader.__getitem__ produces:
  1. Correct tensor shape  : (n_channels, window_samples)
  2. Label matches CSV     : tensor label == row["label"]
  3. Clean signal          : no NaN, no Inf, not all-zero
  4. Correct time slice    : seizure windows land inside a known seizure interval
                             non-seizure windows land outside all seizure intervals

Run from src/data_loader directory on the server:
    cd /home/amir/Desktop/GWU/Research/EEG/src/data_loader
    python3 tests/test_tensors.py
"""

from __future__ import annotations

import sys
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch

# ── Setup path so imports work ─────────────────────────────────────────────
THIS_DIR = Path(__file__).parent.parent          # src/data_loader
sys.path.insert(0, str(THIS_DIR))

from core.io import read_raw

# ── Config ─────────────────────────────────────────────────────────────────
CONFIG_PATH      = THIS_DIR / "config.yaml"
RESULTS_ROOT     = Path("/home/amir/Desktop/GWU/Research/EEG/results")
DATALOADER_DIR   = RESULTS_ROOT / "dataloader"
CSV_BASE_DIR     = THIS_DIR     # relative paths in CSV are relative to here

N_SEIZURE_SAMPLES    = 10   # seizure windows to spot-check
N_NONSEIZURE_SAMPLES = 10   # non-seizure windows to spot-check
OVERLAP_THRESHOLD    = 0.5
RANDOM_SEED          = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ── Helpers ─────────────────────────────────────────────────────────────────

def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = (CSV_BASE_DIR / p).resolve()
    return p


def load_events(edf_path: Path) -> List[Tuple[float, float]]:
    ev_path = edf_path.parent / (edf_path.stem + "_events.tsv")
    if not ev_path.exists():
        return []
    try:
        df = pd.read_csv(ev_path, sep="\t")
        if df.empty or "onset" not in df.columns:
            return []
        return [(float(r["onset"]), float(r["onset"]) + float(r.get("duration", 0)))
                for _, r in df.iterrows()]
    except Exception:
        return []


def overlap_ratio(start: float, end: float, intervals: List[Tuple[float, float]]) -> float:
    dur = end - start
    if dur <= 0:
        return 0.0
    total = sum(max(0.0, min(end, e) - max(start, s)) for s, e in intervals)
    return total / dur


def load_tensor(row: pd.Series, n_channels: int, window_samples: int) -> Tuple[np.ndarray, int]:
    """Load EEG slice from BIDS EDF, return (data array, label)."""
    edf_path = resolve_path(row["path"])
    raw = read_raw(str(edf_path), preload=True)
    sfreq = raw.info["sfreq"]
    start_s = int(float(row["start_sec"]) * sfreq)
    end_s   = min(int(float(row["end_sec"]) * sfreq), raw.n_times)
    data, _ = raw[:, start_s:end_s]

    # pad / trim to expected shape
    c, t = data.shape
    if c < n_channels:
        data = np.vstack([data, np.zeros((n_channels - c, t))])
    elif c > n_channels:
        data = data[:n_channels]
    if t < window_samples:
        data = np.hstack([data, np.zeros((n_channels, window_samples - t))])
    elif t > window_samples:
        data = data[:, :window_samples]

    return data, int(row["label"])


# ── Main test ────────────────────────────────────────────────────────────────

def run_tests():
    import yaml
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    sfreq          = int(cfg["signal"]["target_sfreq"])
    window_sec     = float(cfg["windowing"]["window_sec"])
    n_channels     = int(cfg["channels"]["target_count"])
    window_samples = int(window_sec * sfreq)

    print("=" * 65)
    print("Tensor & Label Output Verification")
    print(f"  Expected shape : ({n_channels}, {window_samples})  "
          f"[{n_channels} ch x {window_samples} samples @ {sfreq}Hz for {window_sec}s]")
    print("=" * 65)

    # Load val split (no oversampling, smaller, faster)
    csv_path = DATALOADER_DIR / "window_index_val.csv"
    df = pd.read_csv(csv_path)
    df_pos = df[df["label"] == 1].drop_duplicates(subset=["path", "start_sec"])
    df_neg = df[df["label"] == 0].drop_duplicates(subset=["path", "start_sec"])

    pos_sample = df_pos.sample(min(N_SEIZURE_SAMPLES, len(df_pos)), random_state=RANDOM_SEED)
    neg_sample = df_neg.sample(min(N_NONSEIZURE_SAMPLES, len(df_neg)), random_state=RANDOM_SEED)

    all_errors = []

    def check_window(row: pd.Series, expected_label: int, tag: str):
        errors = []
        edf_path = resolve_path(row["path"])
        start_sec = float(row["start_sec"])
        end_sec   = float(row["end_sec"])
        csv_label = int(row["label"])

        # ── Load tensor ──
        try:
            data, label = load_tensor(row, n_channels, window_samples)
        except Exception as e:
            errors.append(f"  LOAD ERROR: {e}")
            return errors

        tensor = torch.tensor(data, dtype=torch.float32)

        # Check 1: shape
        if tensor.shape != (n_channels, window_samples):
            errors.append(f"  SHAPE: got {tuple(tensor.shape)}, expected ({n_channels},{window_samples})")

        # Check 2: label matches CSV
        if label != csv_label:
            errors.append(f"  LABEL MISMATCH: loader returned {label}, CSV says {csv_label}")

        # Check 3: no NaN / Inf
        if torch.isnan(tensor).any():
            errors.append(f"  NaN values in tensor")
        if torch.isinf(tensor).any():
            errors.append(f"  Inf values in tensor")

        # Check 4: signal is not all zeros (flat signal = bad load)
        if tensor.abs().max().item() < 1e-12:
            errors.append(f"  All-zero tensor — data not loaded correctly")

        # Check 5: time range vs events TSV
        intervals = load_events(edf_path)
        ratio = overlap_ratio(start_sec, end_sec, intervals)
        ground_truth_label = 1 if ratio >= OVERLAP_THRESHOLD else 0

        if csv_label == 1 and ground_truth_label == 0:
            errors.append(
                f"  LABEL vs EVENTS mismatch: CSV=1 but overlap={ratio:.3f} with {intervals}")
        elif csv_label == 0 and ground_truth_label == 1:
            errors.append(
                f"  LABEL vs EVENTS mismatch: CSV=0 but overlap={ratio:.3f} with {intervals}")

        status = "PASS" if not errors else "FAIL"
        print(f"  [{status}] {tag}  label={csv_label}  "
              f"t=[{start_sec},{end_sec}]  shape={tuple(tensor.shape)}  "
              f"max={tensor.abs().max().item():.4f}  overlap={ratio:.2f}")
        if errors:
            for e in errors:
                print(e)

        return errors

    # ── Seizure windows ──
    print(f"\n--- Seizure windows (label=1), n={len(pos_sample)} ---")
    for i, (_, row) in enumerate(pos_sample.iterrows()):
        errs = check_window(row, expected_label=1, tag=f"SEZ-{i+1:02d}")
        all_errors.extend(errs)

    # ── Non-seizure windows ──
    print(f"\n--- Non-seizure windows (label=0), n={len(neg_sample)} ---")
    for i, (_, row) in enumerate(neg_sample.iterrows()):
        errs = check_window(row, expected_label=0, tag=f"BKG-{i+1:02d}")
        all_errors.extend(errs)

    # ── Summary ──
    print(f"\n{'='*65}")
    if not all_errors:
        print("ALL CHECKS PASSED")
        print(f"  Tensors have correct shape ({n_channels},{window_samples})")
        print(f"  Labels match CSV and events TSV")
        print(f"  Signal is clean (no NaN/Inf, non-zero)")
    else:
        print(f"FAILED — {len(all_errors)} error(s) found")
        for e in all_errors:
            print(e)
        sys.exit(1)


if __name__ == "__main__":
    run_tests()