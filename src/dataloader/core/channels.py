from __future__ import annotations

from typing import Dict, Any, List

import mne
import numpy as np


STANDARD_16 = ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz", "Cz",
                "T7", "T8", "P7", "P8", "C3", "C4", "O1", "O2"]

POSITIONS_1020: Dict[str, np.ndarray] = {
    "Fp1": np.array([-0.30,  0.95,  0.10]),
    "Fp2": np.array([ 0.30,  0.95,  0.10]),
    "F7":  np.array([-0.71,  0.50,  0.10]),
    "F3":  np.array([-0.40,  0.60,  0.50]),
    "Fz":  np.array([ 0.00,  0.71,  0.60]),
    "F4":  np.array([ 0.40,  0.60,  0.50]),
    "F8":  np.array([ 0.71,  0.50,  0.10]),
    "T7":  np.array([-0.95,  0.00,  0.10]),
    "C3":  np.array([-0.50,  0.00,  0.70]),
    "Cz":  np.array([ 0.00,  0.00,  1.00]),
    "C4":  np.array([ 0.50,  0.00,  0.70]),
    "T8":  np.array([ 0.95,  0.00,  0.10]),
    "P7":  np.array([-0.71, -0.50,  0.10]),
    "P3":  np.array([-0.40, -0.60,  0.50]),
    "Pz":  np.array([ 0.00, -0.71,  0.60]),
    "P4":  np.array([ 0.40, -0.60,  0.50]),
    "P8":  np.array([ 0.71, -0.50,  0.10]),
    "O1":  np.array([-0.30, -0.95,  0.10]),
    "Oz":  np.array([ 0.00, -1.00,  0.00]),
    "O2":  np.array([ 0.30, -0.95,  0.10]),
    "AFz": np.array([ 0.00,  0.80,  0.40]),
    "FCz": np.array([ 0.00,  0.35,  0.85]),
    "CPz": np.array([ 0.00, -0.35,  0.85]),
}


def standardize_channels(raw: mne.io.BaseRaw, cfg: Dict[str, Any]) -> mne.io.BaseRaw:
    ch_cfg = cfg.get("channels", {})
    target = int(ch_cfg.get("target_count", 16))
    standard = ch_cfg.get("standard_set", STANDARD_16)[:target]

    n_current = len(raw.ch_names)

    if n_current > target:
        raw = _reduce_to_target(raw, standard, ch_cfg)
    elif n_current < target:
        raw = _expand_to_target(raw, standard, target)

    raw = _ensure_exactly(raw, standard, target)
    return raw


def _reduce_to_target(raw: mne.io.BaseRaw, standard: List[str], ch_cfg: Dict) -> mne.io.BaseRaw:
    available = [c.upper() for c in raw.ch_names]
    std_upper = [s.upper() for s in standard]

    direct_matches = []
    for i, s in enumerate(std_upper):
        if s in available:
            direct_matches.append(raw.ch_names[available.index(s)])

    if len(direct_matches) >= len(standard):
        raw.pick(direct_matches[:len(standard)])
        raw.rename_channels({old: new for old, new in zip(direct_matches[:len(standard)], standard)})
        return raw

    selected = list(direct_matches)
    remaining_std = [standard[i] for i, s in enumerate(std_upper) if s not in available]
    remaining_ch = [ch for ch in raw.ch_names if ch not in direct_matches]

    for std_ch in remaining_std:
        if not remaining_ch:
            break
        if std_ch in POSITIONS_1020:
            target_pos = POSITIONS_1020[std_ch]
            best_ch = _nearest_channel(target_pos, remaining_ch, raw.ch_names)
        else:
            best_ch = remaining_ch[0]
        selected.append(best_ch)
        remaining_ch = [c for c in remaining_ch if c != best_ch]

    raw.pick(selected[:len(standard)])
    rename_map = {old: new for old, new in zip(raw.ch_names, standard[:len(raw.ch_names)])}
    raw.rename_channels(rename_map)
    return raw


def _nearest_channel(target_pos: np.ndarray, candidates: List[str], all_ch: List[str]) -> str:
    best = candidates[0]
    best_dist = float("inf")
    for ch in candidates:
        ch_upper = ch.upper()
        if ch_upper in POSITIONS_1020:
            dist = float(np.linalg.norm(POSITIONS_1020[ch_upper] - target_pos))
            if dist < best_dist:
                best_dist = dist
                best = ch
    return best


def _expand_to_target(raw: mne.io.BaseRaw, standard: List[str], target: int) -> mne.io.BaseRaw:
    n_current = len(raw.ch_names)
    n_missing = target - n_current
    data = raw.get_data()
    sfreq = raw.info["sfreq"]
    ch_names = list(raw.ch_names)

    extra_names = []
    extra_data = []
    for i in range(n_missing):
        src_idx = i % n_current
        extra_data.append(data[src_idx].copy())
        new_name = f"CH_DUP_{i:02d}"
        extra_names.append(new_name)

    combined_data = np.vstack([data, np.array(extra_data)])
    combined_names = ch_names + extra_names
    info = mne.create_info(ch_names=combined_names, sfreq=sfreq, ch_types="eeg")
    return mne.io.RawArray(combined_data, info, verbose=False)


def _ensure_exactly(raw: mne.io.BaseRaw, standard: List[str], target: int) -> mne.io.BaseRaw:
    ch_names = list(raw.ch_names)
    if len(ch_names) >= target:
        raw.pick(ch_names[:target])
        rename_map = {old: new for old, new in zip(raw.ch_names, standard)}
        raw.rename_channels(rename_map)
    return raw


def get_channel_info(raw: mne.io.BaseRaw) -> Dict[str, Any]:
    return {
        "n_channels": len(raw.ch_names),
        "ch_names": raw.ch_names,
        "sfreq": raw.info["sfreq"],
    }