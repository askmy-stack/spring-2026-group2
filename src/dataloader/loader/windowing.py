"""
Windowing + Labeling + Balancing

Takes processed BIDS EDF paths + seizure intervals →
produces a DataFrame of windows with binary labels,
then balances to target seizure ratio.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def build_windows(
    edf_path: Path,
    seizure_intervals: List[Tuple[float, float]],
    subject_id: str,
    metadata: Dict,
    duration_sec: float = 0.0,
    window_sec: float = 1.0,
    stride_sec: float = 1.0,
    sfreq: float = 256.0,
    exclude_near_seizure_sec: float = 300.0,
    max_background_per_file: int = 150,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build window index for a single EDF file.

    Args:
        edf_path: Path to processed EDF
        seizure_intervals: [(start_sec, end_sec), ...] for this file
        subject_id: Subject identifier
        metadata: {"age": ..., "sex": ...}
        duration_sec: Duration of recording (auto-detected if 0)
        window_sec: Window size in seconds
        stride_sec: Stride between windows
        sfreq: Sampling frequency
        exclude_near_seizure_sec: Skip background windows within this range of seizures
        max_background_per_file: Cap background windows per file (0 = no cap).
            Seizure windows are NEVER capped.
        seed: Random seed for background sampling

    Returns:
        DataFrame with columns: path, subject_id, start_sec, end_sec, label, age, sex
    """
    # Get duration from file if not provided
    if duration_sec <= 0:
        try:
            import mne
            raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)
            duration_sec = raw.n_times / raw.info["sfreq"]
        except Exception:
            return pd.DataFrame()

    seizure_rows = []
    background_rows = []
    t = 0.0
    max_start = duration_sec - window_sec

    while t <= max_start:
        end = t + window_sec

        # Check seizure overlap
        label = 0
        for s_start, s_end in seizure_intervals:
            overlap_start = max(t, s_start)
            overlap_end = min(end, s_end)
            if overlap_end > overlap_start:
                overlap_frac = (overlap_end - overlap_start) / window_sec
                if overlap_frac >= 0.5:
                    label = 1
                    break

        # Skip background too close to seizures
        if label == 0 and exclude_near_seizure_sec > 0 and seizure_intervals:
            skip = False
            for s_start, s_end in seizure_intervals:
                if (s_start - exclude_near_seizure_sec) <= t <= (s_end + exclude_near_seizure_sec):
                    if not (t >= s_start and end <= s_end):
                        skip = True
                        break
            if skip:
                t += stride_sec
                continue

        row = {
            "path": str(edf_path),
            "subject_id": subject_id,
            "start_sec": round(t, 3),
            "end_sec": round(end, 3),
            "label": label,
            "age": metadata.get("age", "NA"),
            "sex": metadata.get("sex", "NA"),
        }

        if label == 1:
            seizure_rows.append(row)
        else:
            background_rows.append(row)

        t += stride_sec

    # Cap background windows (seizure windows are NEVER capped)
    if max_background_per_file > 0 and len(background_rows) > max_background_per_file:
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(background_rows), size=max_background_per_file, replace=False)
        background_rows = [background_rows[i] for i in sorted(indices)]

    all_rows = seizure_rows + background_rows
    return pd.DataFrame(all_rows)


def balance_windows(
    df: pd.DataFrame,
    seizure_ratio: float = 0.3,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Oversample seizure windows to reach target ratio.

    Args:
        df: Window index DataFrame with 'label' column
        seizure_ratio: Target fraction of seizure windows (default 0.3)
        seed: Random seed for reproducibility

    Returns:
        Balanced DataFrame
    """
    if df.empty:
        return df

    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]

    n_pos = len(pos)
    n_neg = len(neg)

    if n_pos == 0 or n_neg == 0:
        return df

    current_ratio = n_pos / len(df)
    if current_ratio >= seizure_ratio:
        return df

    # target: n_pos_new / (n_neg + n_pos_new) = seizure_ratio
    n_pos_target = int(seizure_ratio * n_neg / (1.0 - seizure_ratio))

    if n_pos_target <= n_pos:
        return df

    rng = np.random.RandomState(seed)
    n_repeats = n_pos_target // n_pos
    n_remainder = n_pos_target % n_pos

    parts = [neg]
    for _ in range(n_repeats):
        parts.append(pos)
    if n_remainder > 0:
        parts.append(pos.sample(n=n_remainder, random_state=rng))

    balanced = pd.concat(parts, ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=rng).reset_index(drop=True)
    return balanced