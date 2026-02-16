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
    """
    Generic wavelet denoiser for multi-channel signals (EEG).
    Uses PyWavelets. Works per-channel with universal threshold.
    """

    def __init__(
        self,
        wavelet: str = "db4",
        level: Optional[int] = None,
        threshold_mode: str = "soft",  # "soft" or "hard"
    ):
        if not PYWAVELETS_AVAILABLE:
            raise ImportError("PyWavelets not installed. Install with: pip install PyWavelets")

        if threshold_mode not in ("soft", "hard"):
            raise ValueError("threshold_mode must be 'soft' or 'hard'.")

        self.wavelet = wavelet
        self.level = level
        self.threshold_mode = threshold_mode

    @staticmethod
    def _universal_threshold(detail_coeffs: np.ndarray) -> float:
        if detail_coeffs.size == 0:
            return 0.0
        sigma = np.median(np.abs(detail_coeffs - np.median(detail_coeffs))) / 0.6745
        n = detail_coeffs.size
        return float(sigma * np.sqrt(2.0 * np.log(max(n, 2))))

    def denoise_1d(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)

        max_level = pywt.dwt_max_level(len(x), pywt.Wavelet(self.wavelet).dec_len)
        level = self.level if self.level is not None else min(6, max_level)

        coeffs = pywt.wavedec(x, self.wavelet, level=level)
        detail_finest = np.asarray(coeffs[-1])
        thr = self._universal_threshold(detail_finest)

        new_coeffs = [coeffs[0]]
        for d in coeffs[1:]:
            new_coeffs.append(pywt.threshold(d, value=thr, mode=self.threshold_mode))

        y = pywt.waverec(new_coeffs, self.wavelet)
        return y[: len(x)]

    def denoise_nd(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        Y = np.zeros_like(X)
        for ch in range(X.shape[0]):
            Y[ch] = self.denoise_1d(X[ch])
        return Y


class FilterApplier:
    """
    Step 4 filtering module:
      - notch 50/60
      - band/high/low-pass via l_freq/h_freq
      - FIR vs IIR via method + iir_params
      - optional wavelet denoise
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
        wavelet: str = "db4",
        wavelet_level: Optional[int] = None,
        wavelet_threshold: str = "soft",
    ):
        if method not in ("fir", "iir"):
            raise ValueError("method must be 'fir' or 'iir'.")

        self.l_freq = l_freq
        self.h_freq = h_freq
        self.notch_freqs = notch_freqs
        self.method = method
        self.iir_params = iir_params

        self.use_wavelet = use_wavelet
        self._wavelet_denoiser = None
        if use_wavelet:
            self._wavelet_denoiser = WaveletDenoiser(
                wavelet=wavelet,
                level=wavelet_level,
                threshold_mode=wavelet_threshold,
            )

    def apply(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        out = raw.copy()

        # Ensure data loaded if we later do wavelet direct assignment
        if self.use_wavelet and not out.preload:
            out.load_data()

        if self.notch_freqs:
            out.notch_filter(
                freqs=self.notch_freqs,
                method=self.method,
                iir_params=self.iir_params,
                verbose=False,
            )

        out.filter(
            l_freq=self.l_freq,
            h_freq=self.h_freq,
            method=self.method,
            iir_params=self.iir_params,
            verbose=False,
        )

        if self.use_wavelet and self._wavelet_denoiser is not None:
            eeg_picks = mne.pick_types(out.info, eeg=True, meg=False, eog=False, ecg=False, stim=False, exclude=[])
            if len(eeg_picks) > 0:
                X = out.get_data(picks=eeg_picks)
                Y = self._wavelet_denoiser.denoise_nd(X)
                out._data[eeg_picks, :] = Y

        return out
