# src/preprocessing.py
from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import mne


class EEGPreprocessor:
    """
    Signal-only preprocessing (BIDS-friendly):
      - pick EEG channels only
      - resample to target_sfreq
      - notch + bandpass
      - average reference (optional)
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def preprocess(self, raw: mne.io.BaseRaw) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
        signal = self.cfg.get("signal", {})
        raw = raw.copy().load_data()

        # EEG only
        picks = mne.pick_types(raw.info, eeg=True, eog=False, ecg=False, emg=False, stim=False, exclude=[])
        raw.pick(picks)
        raw.set_channel_types({ch: "eeg" for ch in raw.ch_names})

        # resample
        target = int(signal.get("target_sfreq", 256))
        if int(raw.info["sfreq"]) != target:
            raw.resample(target, npad="auto", verbose=False)

        # notch
        notch = signal.get("notch_hz", 60)
        if notch is not None:
            raw.notch_filter(freqs=[float(notch)], verbose=False)

        # bandpass
        bp = signal.get("bandpass_hz", [1.0, 50.0])
        raw.filter(float(bp[0]), float(bp[1]), verbose=False)

        # reference
        ref = str(signal.get("reference", "average"))
        if ref == "average":
            raw.set_eeg_reference("average", verbose=False)

        meta = {
            "sfreq": float(raw.info["sfreq"]),
            "n_channels": int(len(raw.ch_names)),
            "duration_sec": float(raw.times[-1]) if len(raw.times) else 0.0,
        }
        return raw, meta
