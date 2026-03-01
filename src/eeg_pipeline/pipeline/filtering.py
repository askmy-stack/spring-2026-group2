from __future__ import annotations

from typing import Optional, Dict, Any, List

import numpy as np
import mne


try:
    import pywt  # type: ignore
    PYWAVELETS_AVAILABLE = True
except Exception:
    PYWAVELETS_AVAILABLE = False


class WaveletDenoiser:
    def __init__(self, family: str = "db4", level: Optional[int] = None, threshold_mode: str = "soft"):
        if not PYWAVELETS_AVAILABLE:
            raise ImportError("PyWavelets not installed. Install with: pip install PyWavelets")
        if threshold_mode not in ("soft", "hard"):
            raise ValueError("threshold_mode must be 'soft' or 'hard'.")
        self.family = family
        self.level = level
        self.threshold_mode = threshold_mode

    @staticmethod
    def _universal_threshold(detail: np.ndarray) -> float:
        """
        Universal threshold using robust sigma estimate from the finest detail coefficients.
        """
        detail = np.asarray(detail, dtype=float)
        if detail.size == 0:
            return 0.0
        sigma = np.median(np.abs(detail - np.median(detail))) / 0.6745
        n = detail.size
        return float(sigma * np.sqrt(2.0 * np.log(max(n, 2))))

    def denoise_1d(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)

        max_level = pywt.dwt_max_level(len(x), pywt.Wavelet(self.family).dec_len)
        level = self.level if self.level is not None else min(6, max_level)

        coeffs = pywt.wavedec(x, self.family, level=level)
        thr = self._universal_threshold(np.asarray(coeffs[-1]))

        new_coeffs = [coeffs[0]]
        for d in coeffs[1:]:
            new_coeffs.append(pywt.threshold(d, value=thr, mode=self.threshold_mode))

        y = pywt.waverec(new_coeffs, self.family)
        return y[: len(x)]

    def denoise_nd(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        Y = np.zeros_like(X)
        for i in range(X.shape[0]):
            Y[i] = self.denoise_1d(X[i])
        return Y


class FilterApplier:
    """
    Applies:
      - Notch filter (e.g., 50/60 Hz)
      - Bandpass/highpass/lowpass (via l_freq/h_freq)
      - Optional wavelet denoising

    Robust to missing BIDS channel types:
      - tries EEG picks first
      - falls back to all non-stim channels
    """

    def __init__(
        self,
        *,
        l_freq: Optional[float],
        h_freq: Optional[float],
        notch_freqs: Optional[List[float]],
        method: str = "fir",
        iir_params: Optional[Dict[str, Any]] = None,
        use_wavelet: bool = False,
        wavelet_family: str = "db4",
        wavelet_level: Optional[int] = None,
        wavelet_threshold: str = "soft",
    ):
        if method not in ("fir", "iir"):
            raise ValueError("method must be 'fir' or 'iir'.")
        if wavelet_threshold not in ("soft", "hard"):
            raise ValueError("wavelet_threshold must be 'soft' or 'hard'.")

        self.l_freq = l_freq
        self.h_freq = h_freq
        self.notch_freqs = notch_freqs
        self.method = method
        self.iir_params = iir_params

        self.use_wavelet = use_wavelet
        self._denoiser: Optional[WaveletDenoiser] = None
        if use_wavelet:
            self._denoiser = WaveletDenoiser(
                family=wavelet_family,
                level=wavelet_level,
                threshold_mode=wavelet_threshold,
            )

    @staticmethod
    def _pick_data_channels(raw: mne.io.BaseRaw) -> np.ndarray:
        """
        Prefer EEG channels; if none are marked as EEG (common when channels.tsv has null types),
        fall back to all non-stim channels.
        """
        picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, ecg=False, stim=False, exclude=[])
        if len(picks) == 0:
            picks = mne.pick_types(raw.info, stim=False, exclude=[])
        return picks

    def apply(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        out = raw.copy()

        picks = self._pick_data_channels(out)
        if len(picks) == 0:
            # Extremely defensive: nothing to filter
            return out

        # Load data only when needed (wavelet writes into _data)
        if self.use_wavelet and not out.preload:
            out.load_data()

        # Notch
        if self.notch_freqs:
            out.notch_filter(
                freqs=self.notch_freqs,
                picks=picks,
                method=self.method,
                iir_params=self.iir_params,
                verbose=False,
            )

        # Band/high/low pass (single pass)
        if (self.l_freq is not None) or (self.h_freq is not None):
            out.filter(
                l_freq=self.l_freq,
                h_freq=self.h_freq,
                picks=picks,
                method=self.method,
                iir_params=self.iir_params,
                verbose=False,
            )

        # Wavelet denoise (use same picks so it works even if channels are "misc")
        if self.use_wavelet and self._denoiser is not None:
            X = out.get_data(picks=picks)
            Y = self._denoiser.denoise_nd(X)
            out._data[picks, :] = Y

        return out
