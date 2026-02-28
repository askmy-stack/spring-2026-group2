from __future__ import annotations

import warnings
from typing import Dict, Any

import mne
import numpy as np
from scipy.signal import resample_poly
from math import gcd


def preprocess(raw: mne.io.BaseRaw, cfg: Dict[str, Any]) -> mne.io.BaseRaw:
    raw = raw.copy().load_data()
    raw = _pick_eeg(raw)
    raw = _resample(raw, cfg)
    raw = _filter(raw, cfg)
    raw = _reference(raw, cfg)
    return raw


def _pick_eeg(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(picks) == 0:
        warnings.warn(
            f"No EEG channels detected in {len(raw.ch_names)} channels. "
            f"All channels will be treated as EEG: {raw.ch_names}",
            stacklevel=2,
        )
        picks = list(range(len(raw.ch_names)))
    raw.pick(picks)
    raw.set_channel_types({ch: "eeg" for ch in raw.ch_names})
    return raw


def _resample(raw: mne.io.BaseRaw, cfg: Dict[str, Any]) -> mne.io.BaseRaw:
    sig_cfg = cfg.get("signal", {})
    target = int(sig_cfg.get("target_sfreq", 256))
    src = int(raw.info["sfreq"])
    if src == target:
        return raw
    method = cfg.get("resampling", {}).get("default_method", "polyphase")
    if method == "polyphase":
        raw = _resample_polyphase(raw, src, target)
    else:
        raw.resample(target, npad="auto", verbose=False)
    return raw


def _resample_polyphase(raw: mne.io.BaseRaw, src: int, target: int) -> mne.io.BaseRaw:
    g = gcd(src, target)
    up = target // g
    down = src // g
    data = raw.get_data()
    resampled = np.array([
        resample_poly(ch, up, down) for ch in data
    ])
    info = mne.create_info(
        ch_names=raw.ch_names,
        sfreq=target,
        ch_types=["eeg"] * len(raw.ch_names),
    )
    return mne.io.RawArray(resampled, info, verbose=False)


def _filter(raw: mne.io.BaseRaw, cfg: Dict[str, Any]) -> mne.io.BaseRaw:
    sig_cfg = cfg.get("signal", {})
    notch = sig_cfg.get("notch")
    if notch is not None:
        raw.notch_filter(freqs=[float(notch)], verbose=False)
    bp = sig_cfg.get("bandpass")
    if bp is not None:
        raw.filter(float(bp[0]), float(bp[1]), verbose=False)
    return raw


def _reference(raw: mne.io.BaseRaw, cfg: Dict[str, Any]) -> mne.io.BaseRaw:
    ref = cfg.get("signal", {}).get("reference", "average")
    if ref == "average":
        raw.set_eeg_reference("average", verbose=False)
    elif ref and ref != "none":
        raw.set_eeg_reference(ref, verbose=False)
    return raw


def normalize_signal(data: np.ndarray, method: str = "zscore") -> np.ndarray:
    if method == "zscore":
        mean = data.mean(axis=-1, keepdims=True)
        std = data.std(axis=-1, keepdims=True)
        std[std == 0] = 1.0
        return (data - mean) / std
    if method == "minmax":
        mn = data.min(axis=-1, keepdims=True)
        mx = data.max(axis=-1, keepdims=True)
        rng = mx - mn
        rng[rng == 0] = 1.0
        return (data - mn) / rng
    return data