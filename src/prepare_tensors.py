"""
Prepare processed tensor splits for new model training scripts.

Reads raw EDF files from src/data/raw/chbmit, windows them at 256 Hz into
1-second windows (16 channels × 256 timesteps), applies z-score normalisation,
labels seizure vs background, splits 70/15/15 by subject, and saves:

    src/data/processed/chbmit/
        train/data.pt    (N_train, 16, 256)  float32
        train/labels.pt  (N_train,)           float32  0=background 1=seizure
        val/data.pt
        val/labels.pt
        test/data.pt
        test/labels.pt

Usage:
    python src/prepare_tensors.py
    python src/prepare_tensors.py --raw_dir src/data/raw/chbmit --out_dir src/data/processed/chbmit
"""
import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

import mne
import numpy as np
import torch
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

N_CHANNELS = 16
SFREQ = 256
WINDOW_SEC = 1.0
WINDOW_SAMPLES = int(SFREQ * WINDOW_SEC)
STANDARD_CHANNELS = ["FP1", "FP2", "F3", "F4", "F7", "F8", "FZ", "CZ",
                     "T7", "T8", "P7", "P8", "C3", "C4", "O1", "O2"]


def main() -> None:
    """Entry point: parse args, run pipeline, save tensors."""
    parser = argparse.ArgumentParser(description="Prepare .pt tensor splits from CHB-MIT EDF files")
    parser.add_argument("--raw_dir", default="src/data/raw/chbmit")
    parser.add_argument("--out_dir", default="src/data/processed/chbmit")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    _run_pipeline(raw_dir, out_dir, args.seed)


def _run_pipeline(raw_dir: Path, out_dir: Path, seed: int) -> None:
    """Full pipeline: load -> window -> split -> save."""
    subjects = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
    logger.info("Found %d subjects at %s", len(subjects), raw_dir)
    all_windows, all_labels, all_subject_ids = [], [], []
    for subject_dir in subjects:
        windows, labels = _process_subject(subject_dir)
        if windows is not None and len(windows) > 0:
            all_windows.append(windows)
            all_labels.append(labels)
            all_subject_ids.extend([subject_dir.name] * len(windows))
            logger.info("%s: %d windows (%d seizure)", subject_dir.name, len(windows), int(labels.sum()))
    if not all_windows:
        logger.error("No windows extracted. Check raw data path and EDF files.")
        return
    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels, axis=0)
    subject_ids = np.array(all_subject_ids)
    logger.info("Total: %d windows, %d seizure (%.1f%%)", len(y), int(y.sum()), 100 * y.mean())
    _save_splits(X, y, subject_ids, out_dir, seed)


def _process_subject(subject_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load all EDF files for one subject; return (windows, labels)."""
    summary_path = next(subject_dir.glob("*-summary.txt"), None)
    seizure_intervals = _parse_summary(summary_path) if summary_path else {}
    edf_files = sorted(subject_dir.glob("*.edf"))
    all_windows, all_labels = [], []
    for edf_path in edf_files:
        windows, labels = _process_edf(edf_path, seizure_intervals.get(edf_path.name, []))
        if windows is not None:
            all_windows.append(windows)
            all_labels.append(labels)
    if not all_windows:
        return None, None
    return np.concatenate(all_windows, axis=0), np.concatenate(all_labels, axis=0)


def _process_edf(edf_path: Path, seizure_intervals: List[Tuple[float, float]]) -> Tuple:
    """Load one EDF, select channels, window, label, normalise."""
    try:
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    except Exception as exc:
        logger.warning("Could not read %s: %s", edf_path.name, exc)
        return None, None
    data = _extract_channels(raw)
    if data is None:
        return None, None
    data = _zscore_normalise(data)
    windows = _sliding_windows(data)
    labels = _label_windows(len(windows), seizure_intervals, raw.info["sfreq"])
    return windows, labels


def _extract_channels(raw: mne.io.BaseRaw) -> np.ndarray:
    """Select or approximate 16 standard EEG channels; return (16, T) array."""
    ch_names_upper = [ch.upper().replace("-", "").replace("EEG", "").strip() for ch in raw.ch_names]
    selected_indices = []
    for target in STANDARD_CHANNELS:
        matched = [i for i, ch in enumerate(ch_names_upper) if target in ch]
        if matched:
            selected_indices.append(matched[0])
    if len(selected_indices) < 4:
        return None
    data, _ = raw[selected_indices, :]
    if len(selected_indices) < N_CHANNELS:
        pad = np.zeros((N_CHANNELS - len(selected_indices), data.shape[1]), dtype=np.float32)
        data = np.vstack([data, pad])
    return data[:N_CHANNELS].astype(np.float32)


def _zscore_normalise(data: np.ndarray) -> np.ndarray:
    """Apply per-channel z-score normalisation."""
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True) + 1e-8
    return (data - mean) / std


def _sliding_windows(data: np.ndarray) -> np.ndarray:
    """Slice (16, T) into non-overlapping (N, 16, 256) windows."""
    n_windows = data.shape[1] // WINDOW_SAMPLES
    windows = []
    for i in range(n_windows):
        start = i * WINDOW_SAMPLES
        windows.append(data[:, start:start + WINDOW_SAMPLES])
    return np.stack(windows, axis=0) if windows else np.empty((0, N_CHANNELS, WINDOW_SAMPLES))


def _label_windows(
    n_windows: int, seizure_intervals: List[Tuple[float, float]], orig_sfreq: float
) -> np.ndarray:
    """Label each window 1 if it overlaps a seizure interval, else 0."""
    labels = np.zeros(n_windows, dtype=np.float32)
    for start_sec, end_sec in seizure_intervals:
        win_start = int(start_sec / WINDOW_SEC)
        win_end = min(int(np.ceil(end_sec / WINDOW_SEC)), n_windows - 1)
        labels[win_start:win_end + 1] = 1.0
    return labels


def _parse_summary(summary_path: Path) -> Dict[str, List[Tuple[float, float]]]:
    """Parse CHB-MIT *-summary.txt into {filename: [(start_sec, end_sec), ...]}."""
    intervals: Dict[str, List[Tuple[float, float]]] = {}
    current_file = None
    try:
        text = summary_path.read_text(errors="ignore")
        for line in text.splitlines():
            file_match = re.search(r"File Name:\s*(\S+\.edf)", line, re.IGNORECASE)
            if file_match:
                current_file = file_match.group(1).strip()
                intervals.setdefault(current_file, [])
            start_match = re.search(r"Seizure(?:\s+\d+)?\s+Start\s+Time:\s*(\d+)\s+sec", line, re.IGNORECASE)
            end_match = re.search(r"Seizure(?:\s+\d+)?\s+End\s+Time:\s*(\d+)\s+sec", line, re.IGNORECASE)
            if start_match and current_file:
                intervals.setdefault(current_file, []).append((float(start_match.group(1)), 0.0))
            if end_match and current_file and intervals.get(current_file):
                last = intervals[current_file][-1]
                intervals[current_file][-1] = (last[0], float(end_match.group(1)))
    except Exception as exc:
        logger.warning("Could not parse summary %s: %s", summary_path, exc)
    return intervals


def _save_splits(
    X: np.ndarray, y: np.ndarray, subject_ids: np.ndarray, out_dir: Path, seed: int
) -> None:
    """Subject-independent 70/15/15 split; save .pt files."""
    unique_subjects = np.unique(subject_ids)
    train_subs, temp_subs = train_test_split(unique_subjects, test_size=0.30, random_state=seed)
    val_subs, test_subs = train_test_split(temp_subs, test_size=0.50, random_state=seed)
    splits = {
        "train": np.isin(subject_ids, train_subs),
        "val": np.isin(subject_ids, val_subs),
        "test": np.isin(subject_ids, test_subs),
    }
    for split_name, mask in splits.items():
        split_dir = out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        torch.save(torch.tensor(X[mask], dtype=torch.float32), split_dir / "data.pt")
        torch.save(torch.tensor(y[mask], dtype=torch.float32), split_dir / "labels.pt")
        n_seizure = int(y[mask].sum())
        logger.info("Saved %s: %d windows (%d seizure) -> %s", split_name, mask.sum(), n_seizure, split_dir)
    logger.info("Done. All splits saved to %s", out_dir)


if __name__ == "__main__":
    main()
