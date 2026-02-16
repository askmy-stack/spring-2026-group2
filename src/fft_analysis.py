from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import mne

from yaml_utils import get


@dataclass
class FFTResult:
    freqs: np.ndarray          # Hz
    amp: np.ndarray            # amplitude spectrum
    power: np.ndarray          # power spectrum (amp^2)
    channel: str
    tmin: float
    tmax: float


class FFTAnalyzer:
    """
    Explicit FFT analysis (single window) for EEG.

    - Crops a time window [tmin, tmax]
    - Picks one EEG channel (or first EEG channel)
    - Optional detrend + Hann window
    - Computes rFFT and returns amplitude & power vs frequency
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.enabled = bool(get(cfg, "analysis.frequency_domain.fft.enabled", False))

        self.pick_channel = get(cfg, "analysis.frequency_domain.fft.pick_channel", None)
        self.tmin = float(get(cfg, "analysis.frequency_domain.fft.tmin", 0.0))
        self.tmax = float(get(cfg, "analysis.frequency_domain.fft.tmax", 10.0))

        self.detrend = get(cfg, "analysis.frequency_domain.fft.detrend", "constant")  # constant/linear/null
        self.window = get(cfg, "analysis.frequency_domain.fft.window", "hann")        # hann/null

        self.fmin = float(get(cfg, "analysis.frequency_domain.fft.fmin", 0.5))
        self.fmax = float(get(cfg, "analysis.frequency_domain.fft.fmax", 40.0))

    def compute(self, raw: mne.io.BaseRaw) -> Optional[FFTResult]:
        if not self.enabled:
            return None

        rr = raw.copy().crop(tmin=self.tmin, tmax=self.tmax, include_tmax=False)
        if not rr.preload:
            rr.load_data()

        # Pick a channel
        if self.pick_channel is not None and self.pick_channel in rr.ch_names:
            ch_name = str(self.pick_channel)
            pick_idx = mne.pick_channels(rr.ch_names, include=[ch_name])
        else:
            # fallback: first EEG channel
            eeg_picks = mne.pick_types(rr.info, eeg=True, meg=False, eog=False, ecg=False, stim=False, exclude=[])
            if len(eeg_picks) == 0:
                eeg_picks = np.arange(len(rr.ch_names))
            pick_idx = np.array([int(eeg_picks[0])], dtype=int)
            ch_name = rr.ch_names[int(pick_idx[0])]

        x = rr.get_data(picks=pick_idx).squeeze()  # volts, shape (n_times,)
        sfreq = float(rr.info["sfreq"])
        n = x.size
        if n < 8:
            return None

        # Detrend
        if self.detrend == "constant":
            x = x - np.mean(x)
        elif self.detrend == "linear":
            t = np.arange(n)
            coef = np.polyfit(t, x, 1)
            x = x - (coef[0] * t + coef[1])

        # Window
        if self.window == "hann":
            w = np.hanning(n)
            xw = x * w
        else:
            xw = x

        # rFFT
        X = np.fft.rfft(xw)
        freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)

        # Amplitude & power (simple definitions)
        amp = np.abs(X) / max(n, 1)
        power = amp ** 2

        # Band-limit
        mask = (freqs >= self.fmin) & (freqs <= self.fmax)
        freqs = freqs[mask]
        amp = amp[mask]
        power = power[mask]

        return FFTResult(freqs=freqs, amp=amp, power=power, channel=ch_name, tmin=self.tmin, tmax=self.tmax)
