# src/bids_utils.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
import mne


SUPPORTED_EEG_EXTS = [".edf", ".bdf", ".vhdr", ".set", ".fif"]


def _safe_read_tsv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep="\t")
    except Exception:
        return pd.DataFrame()


def find_bids_recordings(bids_root: Path) -> List[Dict[str, Any]]:
    """
    Minimal BIDS scanner without needing mne-bids.
    Returns list of dict records:
      subject_id, session, task, run, eeg_path, events_path (optional), sidecar_json (optional)
    """
    bids_root = Path(bids_root)
    if not bids_root.exists():
        raise FileNotFoundError(f"BIDS root not found: {bids_root}")

    out: List[Dict[str, Any]] = []
    for sub_dir in sorted(bids_root.glob("sub-*")):
        if not sub_dir.is_dir():
            continue
        subject_id = sub_dir.name.replace("sub-", "")

        # sessions optional
        ses_dirs = list(sub_dir.glob("ses-*"))
        search_roots = ses_dirs if ses_dirs else [sub_dir]

        for root in search_roots:
            session = root.name.replace("ses-", "") if root.name.startswith("ses-") else None
            eeg_dir = root / "eeg"
            if not eeg_dir.exists():
                continue

            for eeg_path in eeg_dir.rglob("*"):
                if eeg_path.suffix.lower() not in SUPPORTED_EEG_EXTS:
                    continue

                base = eeg_path.name
                # BIDS events typically share the same stem before extension, e.g. *_events.tsv
                events_path = eeg_path.with_name(eeg_path.stem + "_events.tsv")
                if not events_path.exists():
                    # also try replacing extension in case of double suffix stems
                    events_path = eeg_path.parent / (eeg_path.stem.split(".")[0] + "_events.tsv")
                if not events_path.exists():
                    events_path = None

                json_path = eeg_path.with_suffix(".json")
                if not json_path.exists():
                    json_path = None

                # best-effort parse task/run from filename
                name = eeg_path.name
                task = None
                run = None
                parts = name.split("_")
                for p in parts:
                    if p.startswith("task-"):
                        task = p.replace("task-", "")
                    if p.startswith("run-"):
                        run = p.replace("run-", "").split(".")[0]

                out.append(
                    dict(
                        subject_id=subject_id,
                        session=session,
                        task=task,
                        run=run,
                        eeg_path=str(eeg_path),
                        events_path=str(events_path) if events_path else None,
                        json_path=str(json_path) if json_path else None,
                    )
                )
    return out


def read_raw_any(eeg_path: str) -> mne.io.BaseRaw:
    """
    Generic loader for common EEG formats supported by MNE.
    """
    p = Path(eeg_path)
    suf = p.suffix.lower()
    if suf in [".edf", ".bdf"]:
        return mne.io.read_raw_edf(eeg_path, preload=False, verbose=False)
    if suf == ".vhdr":
        return mne.io.read_raw_brainvision(eeg_path, preload=False, verbose=False)
    if suf == ".set":
        return mne.io.read_raw_eeglab(eeg_path, preload=False, verbose=False)
    if suf == ".fif":
        return mne.io.read_raw_fif(eeg_path, preload=False, verbose=False)
    raise ValueError(f"Unsupported EEG file type: {suf} ({eeg_path})")


def extract_seizure_intervals_from_events(
    events_path: Optional[str],
    seizure_keywords: List[str],
) -> List[Tuple[float, float]]:
    """
    Reads BIDS *_events.tsv and returns seizure-like intervals (onset, offset).
    We look in columns: trial_type, description, event_type, or any string cols.
    """
    if not events_path:
        return []

    df = _safe_read_tsv(Path(events_path))
    if df.empty:
        return []

    # basic BIDS columns: onset (sec), duration (sec) are common
    if "onset" not in df.columns:
        return []
    if "duration" not in df.columns:
        df["duration"] = 0.0

    # find a label column
    label_cols = [c for c in ["trial_type", "description", "event_type", "value"] if c in df.columns]
    if not label_cols:
        # fallback: scan any object columns
        label_cols = [c for c in df.columns if df[c].dtype == "object"]

    keys = [k.lower() for k in seizure_keywords]
    intervals: List[Tuple[float, float]] = []

    for _, r in df.iterrows():
        label_text = ""
        for c in label_cols:
            v = r.get(c, "")
            if isinstance(v, str):
                label_text += " " + v.lower()
        if any(k in label_text for k in keys):
            onset = float(r["onset"])
            dur = float(r.get("duration", 0.0))
            end = onset + max(0.0, dur)
            # if duration is 0, still keep a small interval marker
            if end == onset:
                end = onset + 1.0
            intervals.append((onset, end))

    # merge overlaps
    intervals = sorted(intervals)
    merged: List[Tuple[float, float]] = []
    for s, e in intervals:
        if not merged or s > merged[-1][1]:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
    return merged
