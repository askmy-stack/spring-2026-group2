from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import mne

from yaml_utils import get

try:
    from scipy.signal import stft  # explicit FFT-based
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


class TimeFrequencyAnalyzer:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def stft_spectrogram_mean(
        self,
        raw: mne.io.BaseRaw,
        *,
        pick_channel: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy not installed. Install with: pip install scipy")

        fmin = float(get(self.cfg, "analysis.time_frequency.stft.fmin", 0.5))
        fmax = float(get(self.cfg, "analysis.time_frequency.stft.fmax", 40.0))
        win_sec = float(get(self.cfg, "analysis.time_frequency.stft.nperseg_sec", 2.0))
        overlap_frac = float(get(self.cfg, "analysis.time_frequency.stft.noverlap_frac", 0.5))

        sf = float(raw.info["sfreq"])
        nperseg = max(16, int(round(win_sec * sf)))
        noverlap = int(round(overlap_frac * nperseg))

        picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, ecg=False, stim=False, exclude=[])
        if pick_channel is not None and pick_channel in raw.ch_names:
            picks = [raw.ch_names.index(pick_channel)]
        if len(picks) == 0:
            picks = list(range(len(raw.ch_names)))

        X = raw.get_data(picks=picks)  # (n_ch, n_times)

        S_list = []
        freqs = None
        times = None
        for i in range(X.shape[0]):
            f, t, Zxx = stft(X[i], fs=sf, nperseg=nperseg, noverlap=noverlap, boundary=None)
            mag = np.abs(Zxx)
            if freqs is None:
                freqs, times = f, t
            S_list.append(mag)

        S = np.mean(np.stack(S_list, axis=0), axis=0)  # (n_freq, n_time)

        mask = (freqs >= fmin) & (freqs <= fmax)
        return freqs[mask], times, S[mask, :]

    def morlet_tfr(self, epochs: mne.Epochs) -> mne.time_frequency.EpochsTFR:
        fmin = float(get(self.cfg, "analysis.time_frequency.morlet_tfr.fmin", 2.0))
        fmax = float(get(self.cfg, "analysis.time_frequency.morlet_tfr.fmax", 40.0))
        n_freqs = int(get(self.cfg, "analysis.time_frequency.morlet_tfr.n_freqs", 30))
        n_cycles = float(get(self.cfg, "analysis.time_frequency.morlet_tfr.n_cycles", 7))

        freqs = np.linspace(fmin, fmax, n_freqs)
        power = mne.time_frequency.tfr_morlet(
            epochs,
            freqs=freqs,
            n_cycles=n_cycles,
            return_itc=False,
            average=True,
            verbose=False,
        )
        return power
