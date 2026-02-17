from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd


def label_windows(
    windows_df: pd.DataFrame,
    seizure_intervals: List[Tuple[float, float]],
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    label_cfg = cfg.get("labeling", {})
    mode = label_cfg.get("mode", "binary")
    threshold = float(label_cfg.get("overlap_threshold", 0.5))
    exclude_sec = cfg.get("windowing", {}).get("exclude_negatives_within_sec", 0)

    df = windows_df.copy()
    df["label"] = 0

    if not seizure_intervals:
        return df

    for idx, row in df.iterrows():
        start = float(row["start_sec"])
        end = float(row["end_sec"])
        duration = end - start
        overlap = _compute_overlap(start, end, seizure_intervals)
        ratio = overlap / duration if duration > 0 else 0.0
        if ratio >= threshold:
            df.at[idx, "label"] = 1

    if exclude_sec > 0:
        df = _exclude_near_seizure_negatives(df, seizure_intervals, exclude_sec)

    if mode == "multiclass":
        df = _assign_multiclass(df, seizure_intervals, cfg)

    return df


def _compute_overlap(
    start: float,
    end: float,
    intervals: List[Tuple[float, float]],
) -> float:
    total = 0.0
    for s, e in intervals:
        o = min(end, e) - max(start, s)
        if o > 0:
            total += o
    return total


def _exclude_near_seizure_negatives(
    df: pd.DataFrame,
    intervals: List[Tuple[float, float]],
    exclude_sec: float,
) -> pd.DataFrame:
    mask_keep = []
    for _, row in df.iterrows():
        if row["label"] == 1:
            mask_keep.append(True)
            continue
        start = float(row["start_sec"])
        end = float(row["end_sec"])
        near = any(
            start < e + exclude_sec and end > s - exclude_sec
            for s, e in intervals
        )
        mask_keep.append(not near)
    return df[mask_keep].reset_index(drop=True)


def _assign_multiclass(
    df: pd.DataFrame,
    intervals: List[Tuple[float, float]],
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    type_map = cfg.get("labeling", {}).get("seizure_types", {})
    background_label = type_map.get("background", 0)
    default_seizure_label = type_map.get("unknown", 1)

    new_labels = []
    for _, row in df.iterrows():
        if row["label"] == 0:
            new_labels.append(background_label)
            continue
        start = float(row["start_sec"])
        end = float(row["end_sec"])
        best_overlap = 0.0
        best_label = default_seizure_label
        for i, (s, e) in enumerate(intervals):
            o = min(end, e) - max(start, s)
            if o > best_overlap:
                best_overlap = o
                best_label = default_seizure_label
        new_labels.append(best_label)
    df = df.copy()
    df["label"] = new_labels
    return df


def extract_seizure_intervals(
    events_path: Optional[str],
    seizure_keywords: List[str],
) -> List[Tuple[float, float]]:
    if not events_path:
        return []
    p = Path(events_path)
    if not p.exists():
        return []
    df = _safe_read_tsv(p)
    if df.empty or "onset" not in df.columns:
        return []
    if "duration" not in df.columns:
        df["duration"] = 0.0
    label_cols = [c for c in ["trial_type", "description", "event_type", "value"] if c in df.columns]
    if not label_cols:
        label_cols = [c for c in df.columns if df[c].dtype == object]
    keys = [k.lower() for k in seizure_keywords]
    intervals: List[Tuple[float, float]] = []
    for _, row in df.iterrows():
        text = " ".join(str(row.get(c, "")) for c in label_cols).lower()
        if any(k in text for k in keys):
            onset = float(row["onset"])
            dur = float(row.get("duration", 0.0))
            end = onset + max(dur, 1.0)
            intervals.append((onset, end))
    return _merge_intervals(sorted(intervals))


def _safe_read_tsv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep="\t")
    except Exception:
        return pd.DataFrame()


def _merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    merged: List[Tuple[float, float]] = []
    for s, e in intervals:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def build_window_index(
    eeg_path: str,
    duration_sec: float,
    subject_id: str,
    seizure_intervals: List[Tuple[float, float]],
    cfg: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    win_cfg = cfg.get("windowing", {})
    window_sec = float(win_cfg.get("window_sec", 1.0))
    stride_sec = float(win_cfg.get("stride_sec", 1.0))

    starts = np.arange(0, duration_sec - window_sec + 1e-9, stride_sec)
    rows = []
    for s in starts:
        e = s + window_sec
        rows.append({
            "path": eeg_path,
            "subject_id": subject_id,
            "start_sec": round(float(s), 6),
            "end_sec": round(float(e), 6),
            "label": 0,
            "age": metadata.get("age", "NA") if metadata else "NA",
            "sex": metadata.get("sex", "NA") if metadata else "NA",
        })
    df = pd.DataFrame(rows)
    if seizure_intervals:
        df = label_windows(df, seizure_intervals, cfg)
    return df


def balance_index(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    bal_cfg = cfg.get("balance", {})
    if not bal_cfg.get("enable", True):
        return df
    method = bal_cfg.get("method", "oversample")
    ratio = float(bal_cfg.get("seizure_ratio", 0.3))
    seed = int(bal_cfg.get("seed", 42))
    rng = np.random.default_rng(seed)

    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]

    if len(pos) == 0:
        return df

    target_pos = int(len(neg) * ratio / (1 - ratio))

    if method == "oversample":
        if len(pos) < target_pos:
            extra = rng.choice(len(pos), size=target_pos - len(pos), replace=True)
            pos_extra = pos.iloc[extra]
            pos = pd.concat([pos, pos_extra], ignore_index=True)
    elif method == "undersample":
        n_neg = int(len(pos) / ratio * (1 - ratio))
        neg = neg.sample(n=min(n_neg, len(neg)), random_state=seed)

    return pd.concat([pos, neg], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)