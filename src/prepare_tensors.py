"""
Prepare processed tensor splits for model training scripts.

Reads raw EDF files from src/data/raw/chbmit, windows them at 256 Hz into
1-second windows (16 channels × 256 timesteps), applies z-score normalisation,
labels seizure vs background, and writes subject-independent 70/15/15 splits:

    src/data/processed/chbmit/
        train/data.pt    (N_train, 16, 256)  float32
        train/labels.pt  (N_train,)           float32  0=background 1=seizure
        val/data.pt
        val/labels.pt
        test/data.pt
        test/labels.pt

Memory design:
    * Subject→split mapping is decided up front (no EDFs loaded yet).
    * Each subject's windows are processed and written directly to a per-subject
      chunk file under the target split's ``_chunks/`` directory, then freed.
    * At the end, each split's chunks are concatenated into one data.pt +
      labels.pt pair and the chunks dir is deleted.
    * Peak RAM ≈ largest single split after subsampling (not all data at once).

Class balance:
    * CHB-MIT is extremely imbalanced (~0.05% seizure windows). By default we
      keep ``--background_ratio 10`` background windows per seizure window in
      each EDF — reduces ~1.6M windows to ~50-100k and improves training
      signal. Pass ``--background_ratio 0`` to disable subsampling (keep all
      background — WILL OOM on most machines).

Usage:
    python src/prepare_tensors.py                               # defaults
    python src/prepare_tensors.py --background_ratio 20         # more bg
    python src/prepare_tensors.py --background_ratio 0          # keep all (risky)
    python src/prepare_tensors.py --raw_dir src/data/raw/chbmit \
                                  --out_dir src/data/processed/chbmit
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
    parser.add_argument(
        "--background_ratio", type=float, default=10.0,
        help="Background windows kept per seizure window per EDF. "
             "Default 10.0 gives roughly balanced training. "
             "Pass 0 to keep every background window (very large; likely OOM).",
    )
    args = parser.parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    _run_pipeline(raw_dir, out_dir, args.seed, args.background_ratio)


def _run_pipeline(
    raw_dir: Path, out_dir: Path, seed: int, background_ratio: float,
) -> None:
    """Streaming pipeline: assign splits -> process each subject -> write chunks -> consolidate."""
    subjects = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
    logger.info("Found %d subjects at %s", len(subjects), raw_dir)
    if not subjects:
        logger.error("No subject directories under %s", raw_dir)
        return

    split_map = _assign_subjects_to_splits([s.name for s in subjects], seed)
    logger.info("Split assignment: train=%d val=%d test=%d",
                sum(v == "train" for v in split_map.values()),
                sum(v == "val"   for v in split_map.values()),
                sum(v == "test"  for v in split_map.values()))

    chunk_dirs = {
        split: (out_dir / split / "_chunks") for split in ("train", "val", "test")
    }
    for d in chunk_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    total_windows = total_seizure = 0
    for subject_dir in subjects:
        split = split_map[subject_dir.name]
        windows, labels = _process_subject(subject_dir, rng, background_ratio)
        if windows is None or len(windows) == 0:
            logger.warning("%s: no windows extracted, skipping", subject_dir.name)
            continue
        chunk_path = chunk_dirs[split] / f"{subject_dir.name}.pt"
        torch.save(
            {
                "data": torch.from_numpy(windows.astype(np.float32)),
                "labels": torch.from_numpy(labels.astype(np.float32)),
            },
            chunk_path,
        )
        total_windows += len(windows)
        total_seizure += int(labels.sum())
        logger.info(
            "%s -> %s: %d windows (%d seizure) saved to %s",
            subject_dir.name, split, len(windows), int(labels.sum()), chunk_path.name,
        )
        del windows, labels  # free per-subject memory before next subject

    logger.info(
        "Total processed: %d windows (%d seizure, %.2f%%)",
        total_windows, total_seizure,
        100.0 * total_seizure / max(total_windows, 1),
    )

    for split, chunk_dir in chunk_dirs.items():
        _consolidate_split(chunk_dir, out_dir / split)


def _assign_subjects_to_splits(
    subject_names: List[str], seed: int,
) -> Dict[str, str]:
    """Deterministic subject-independent 70/15/15 split by name."""
    unique = sorted(set(subject_names))
    train_subs, temp_subs = train_test_split(unique, test_size=0.30, random_state=seed)
    val_subs, test_subs = train_test_split(temp_subs, test_size=0.50, random_state=seed)
    mapping: Dict[str, str] = {}
    for s in train_subs:
        mapping[s] = "train"
    for s in val_subs:
        mapping[s] = "val"
    for s in test_subs:
        mapping[s] = "test"
    return mapping


def _consolidate_split(chunk_dir: Path, split_dir: Path) -> None:
    """Read every per-subject chunk under ``chunk_dir`` and emit data.pt / labels.pt.

    Peak RAM during consolidation ≈ total size of this one split (not all splits).
    Chunks are deleted afterwards.
    """
    chunk_files = sorted(chunk_dir.glob("*.pt"))
    if not chunk_files:
        logger.warning("No chunks found in %s — skipping consolidation", chunk_dir)
        return

    data_parts: List[torch.Tensor] = []
    label_parts: List[torch.Tensor] = []
    subj_parts: List[torch.Tensor] = []
    for chunk_path in chunk_files:
        payload = torch.load(chunk_path, weights_only=True)
        data_parts.append(payload["data"])
        label_parts.append(payload["labels"])
        m = re.search(r"\d+", chunk_path.stem)
        subj_id = int(m.group()) if m else hash(chunk_path.stem) % 10000
        subj_parts.append(torch.full((len(payload["data"]),), subj_id, dtype=torch.long))

    data = torch.cat(data_parts, dim=0)
    labels = torch.cat(label_parts, dim=0)
    del data_parts, label_parts

    split_dir.mkdir(parents=True, exist_ok=True)
    torch.save(data, split_dir / "data.pt")
    torch.save(labels, split_dir / "labels.pt")
    torch.save(torch.cat(subj_parts), split_dir / "subject_ids.pt")
    n_seizure = int(labels.sum().item())
    logger.info(
        "Consolidated %s: %d windows (%d seizure, %.2f%%) -> %s",
        split_dir.name, len(labels), n_seizure,
        100.0 * n_seizure / max(len(labels), 1),
        split_dir,
    )

    # Delete the chunk files once the consolidated tensors are safely saved.
    for chunk_path in chunk_files:
        chunk_path.unlink(missing_ok=True)
    try:
        chunk_dir.rmdir()
    except OSError:
        pass  # non-empty (something raced); leave it for manual cleanup


def _process_subject(
    subject_dir: Path, rng: np.random.Generator, background_ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load every EDF for one subject and return concatenated (windows, labels).

    Subsampling is applied per-EDF before concatenation so peak RAM stays
    bounded by the largest EDF, not the whole subject.
    """
    summary_path = next(subject_dir.glob("*-summary.txt"), None)
    seizure_intervals = _parse_summary(summary_path) if summary_path else {}
    edf_files = sorted(subject_dir.glob("*.edf"))
    all_windows: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    for edf_path in edf_files:
        windows, labels = _process_edf(
            edf_path, seizure_intervals.get(edf_path.name, []),
            rng, background_ratio,
        )
        if windows is not None and len(windows) > 0:
            all_windows.append(windows)
            all_labels.append(labels)
    if not all_windows:
        return None, None
    return np.concatenate(all_windows, axis=0), np.concatenate(all_labels, axis=0)


def _process_edf(
    edf_path: Path, seizure_intervals: List[Tuple[float, float]],
    rng: np.random.Generator, background_ratio: float,
) -> Tuple:
    """Load one EDF, select channels, window, label, normalise, subsample."""
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
    if background_ratio > 0:
        windows, labels = _subsample_background(windows, labels, background_ratio, rng)
    return windows, labels


def _subsample_background(
    windows: np.ndarray, labels: np.ndarray,
    ratio: float, rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Keep every seizure window and at most ``ratio * n_seizure`` background windows.

    If the EDF has no seizure windows we keep at most ``ratio`` background
    windows (tiny) so the background distribution isn't entirely lost.
    """
    seizure_mask = labels > 0.5
    n_seizure = int(seizure_mask.sum())
    n_bg_total = int((~seizure_mask).sum())
    if n_bg_total == 0:
        return windows, labels
    max_bg = max(int(np.ceil(ratio * max(n_seizure, 1))), 1)
    if max_bg >= n_bg_total:
        return windows, labels
    bg_idx = np.where(~seizure_mask)[0]
    chosen_bg = rng.choice(bg_idx, size=max_bg, replace=False)
    seiz_idx = np.where(seizure_mask)[0]
    keep = np.sort(np.concatenate([seiz_idx, chosen_bg]))
    return windows[keep], labels[keep]


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


if __name__ == "__main__":
    main()
