from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import mne

from eeg_pipeline.core.yaml_utils import get


@dataclass
class PSDResult:
    freqs: np.ndarray
    psd_per_channel: np.ndarray     # (n_ch, n_freq)
    psd_mean: np.ndarray            # (n_freq,)
    ch_names: List[str]


@dataclass
class SpectrogramResult:
    freqs: np.ndarray               # (n_freq,)
    times: np.ndarray               # (n_win,)
    S: np.ndarray                   # (n_freq, n_win)
    units: str                      # "V2/Hz" or "dB"


@dataclass
class FFTResult:
    freqs: np.ndarray               # (n_freq,)
    amp: np.ndarray                 # amplitude spectrum (Volts)
    power: np.ndarray               # power spectrum (V^2)
    channel: str
    tmin: float
    tmax: float


@dataclass
class MorletResult:
    freqs: np.ndarray               # (n_freq,)
    times: np.ndarray               # (n_time,)
    P: np.ndarray                   # (n_freq, n_time) power in V^2 (or dB if scaled later)
    units: str                      # "V2" or "dB"


class FrequencyDomainAnalyzer:
    """
    Frequency + time-frequency analyzers in one module:
      - PSD (Welch / multitaper)
      - Bandpower (from PSD mean)
      - PSD spectrogram (time-resolved PSD slices)
      - FFT (explicit rFFT)
      - Morlet spectrogram (TFR averaged across channels or single channel)
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    # ---------------- PSD ----------------
    def psd(self, raw: mne.io.BaseRaw) -> PSDResult:
        method = str(get(self.cfg, "analysis.frequency_domain.psd.method", "welch")).lower()
        fmin = float(get(self.cfg, "analysis.frequency_domain.psd.fmin", 0.5))
        fmax = float(get(self.cfg, "analysis.frequency_domain.psd.fmax", 70.0))
        if method not in ("welch", "multitaper"):
            raise ValueError("PSD method must be 'welch' or 'multitaper'")

        picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, ecg=False, stim=False, exclude=[])
        if len(picks) == 0:
            picks = np.arange(len(raw.ch_names))

        spec = raw.compute_psd(method=method, fmin=fmin, fmax=fmax, picks=picks, verbose=False)

        # IMPORTANT: never drop all channels due to bads
        P = spec.get_data(picks="all", exclude=())      # (n_ch, n_freq)
        freqs = spec.freqs
        ch_names = list(spec.ch_names)
        mean = np.mean(P, axis=0)

        return PSDResult(freqs=freqs, psd_per_channel=P, psd_mean=mean, ch_names=ch_names)

    @staticmethod
    def _bandpower(freqs: np.ndarray, psd: np.ndarray, band: Tuple[float, float]) -> float:
        lo, hi = float(band[0]), float(band[1])
        mask = (freqs >= lo) & (freqs <= hi)
        if not np.any(mask):
            return 0.0
        return float(np.trapezoid(psd[mask], freqs[mask]))

    def bandpowers(self, psd_res: PSDResult) -> Dict[str, float]:
        bands = get(self.cfg, "analysis.frequency_domain.bandpower.bands", {}) or {}
        out: Dict[str, float] = {}
        for name, lim in bands.items():
            out[name] = self._bandpower(psd_res.freqs, psd_res.psd_mean, (lim[0], lim[1]))
        return out

    # ---------------- PSD Spectrogram ----------------
    def psd_spectrogram(
            self,
            raw: mne.io.BaseRaw,
            *,
            pick_channel: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          freqs: (n_freq,)
          times: (n_win,)
          P:     (n_freq, n_win) PSD in V^2/Hz
        """
        method = str(get(self.cfg, "analysis.frequency_domain.psd.method", "welch")).lower()
        fmin = float(get(self.cfg, "analysis.frequency_domain.psd.fmin", 0.5))
        fmax = float(get(self.cfg, "analysis.frequency_domain.psd.fmax", 70.0))

        win_sec = float(get(self.cfg, "analysis.frequency_domain.spectrogram.win_sec", 2.0))
        step_sec = float(get(self.cfg, "analysis.frequency_domain.spectrogram.step_sec", 0.25))

        bandwidth = get(self.cfg, "analysis.frequency_domain.spectrogram.bandwidth", None)
        adaptive = bool(get(self.cfg, "analysis.frequency_domain.spectrogram.adaptive", True))
        low_bias = bool(get(self.cfg, "analysis.frequency_domain.spectrogram.low_bias", True))

        sf = float(raw.info["sfreq"])
        nperseg = max(16, int(round(win_sec * sf)))
        step = max(1, int(round(step_sec * sf)))

        picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, ecg=False, stim=False, exclude=[])
        if pick_channel is not None and pick_channel in raw.ch_names:
            picks = [raw.ch_names.index(pick_channel)]
        if len(picks) == 0:
            picks = list(range(len(raw.ch_names)))

        X = raw.get_data(picks=picks)  # (n_ch, n_times) in Volts
        n_ch, n_times = X.shape

        starts = np.arange(0, max(1, n_times - nperseg + 1), step, dtype=int)
        times = (starts + nperseg / 2) / sf

        P_list = []
        freqs = None

        for s in starts:
            seg = X[:, s: s + nperseg]

            if method == "welch":
                psd, f = mne.time_frequency.psd_array_welch(
                    seg,
                    sfreq=sf,
                    fmin=fmin,
                    fmax=fmax,
                    n_fft=nperseg,
                    n_per_seg=nperseg,
                    n_overlap=0,
                    average="mean",
                    verbose=False,
                )
            elif method == "multitaper":
                psd, f = mne.time_frequency.psd_array_multitaper(
                    seg,
                    sfreq=sf,
                    fmin=fmin,
                    fmax=fmax,
                    adaptive=adaptive,
                    low_bias=low_bias,
                    normalization="full",
                    bandwidth=bandwidth,
                    verbose=False,
                )
            else:
                raise ValueError(f"Unknown PSD method: {method}")

            if freqs is None:
                freqs = f

            P_list.append(np.median(psd, axis=0))

        P = np.stack(P_list, axis=1)  # (n_freq, n_win)
        return freqs, times, P
    # ---------------- FFT ----------------
    def fft(self, raw: mne.io.BaseRaw) -> Optional[FFTResult]:
        if not bool(get(self.cfg, "analysis.frequency_domain.fft.enabled", False)):
            return None

        pick_channel = get(self.cfg, "analysis.frequency_domain.fft.pick_channel", None)
        tmin = float(get(self.cfg, "analysis.frequency_domain.fft.tmin", 0.0))
        tmax = float(get(self.cfg, "analysis.frequency_domain.fft.tmax", 10.0))
        detrend = str(get(self.cfg, "analysis.frequency_domain.fft.detrend", "constant"))
        window = str(get(self.cfg, "analysis.frequency_domain.fft.window", "hann"))
        fmin = float(get(self.cfg, "analysis.frequency_domain.fft.fmin", 0.5))
        fmax = float(get(self.cfg, "analysis.frequency_domain.fft.fmax", 70.0))

        rr = raw.copy().crop(tmin=tmin, tmax=tmax, include_tmax=False)
        if not rr.preload:
            rr.load_data()

        if pick_channel is not None and pick_channel in rr.ch_names:
            ch_name = str(pick_channel)
            pick_idx = mne.pick_channels(rr.ch_names, include=[ch_name])
        else:
            eeg_picks = mne.pick_types(rr.info, eeg=True, meg=False, eog=False, ecg=False, stim=False, exclude=[])
            if len(eeg_picks) == 0:
                eeg_picks = np.arange(len(rr.ch_names))
            pick_idx = np.array([int(eeg_picks[0])], dtype=int)
            ch_name = rr.ch_names[int(pick_idx[0])]

        x = rr.get_data(picks=pick_idx).squeeze()
        sfreq = float(rr.info["sfreq"])
        n = x.size
        if n < 8:
            return None

        if detrend == "constant":
            x = x - np.mean(x)
        elif detrend == "linear":
            t = np.arange(n)
            coef = np.polyfit(t, x, 1)
            x = x - (coef[0] * t + coef[1])

        if window == "hann":
            x = x * np.hanning(n)

        X = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)

        amp = np.abs(X) / max(n, 1)
        power = amp ** 2

        mask = (freqs >= fmin) & (freqs <= fmax)
        return FFTResult(freqs=freqs[mask], amp=amp[mask], power=power[mask], channel=ch_name, tmin=tmin, tmax=tmax)

    # ---------------- Morlet spectrogram ----------------
    def morlet_spectrogram(self, raw: mne.io.BaseRaw, *, pick_channel: Optional[str] = None) -> Optional[MorletResult]:
        if not bool(get(self.cfg, "analysis.frequency_domain.morlet.enabled", False)):
            return None

        fmin = float(get(self.cfg, "analysis.frequency_domain.morlet.fmin", 2.0))
        fmax = float(get(self.cfg, "analysis.frequency_domain.morlet.fmax", 40.0))
        n_freqs = int(get(self.cfg, "analysis.frequency_domain.morlet.n_freqs", 30))
        n_cycles = float(get(self.cfg, "analysis.frequency_domain.morlet.n_cycles", 7))
        decim = int(get(self.cfg, "analysis.frequency_domain.morlet.decim", 1))

        freqs = np.linspace(fmin, fmax, n_freqs)

        picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, ecg=False, stim=False, exclude=[])
        if pick_channel is not None and pick_channel in raw.ch_names:
            picks = [raw.ch_names.index(pick_channel)]
        if len(picks) == 0:
            picks = list(range(len(raw.ch_names)))

        # Use MNE Raw -> TFR directly (no epochs needed). average=False gives (n_ch, n_freq, n_time)
        power = mne.time_frequency.tfr_array_morlet(
            raw.get_data(picks=picks)[None, ...],       # shape (1, n_ch, n_times)
            sfreq=float(raw.info["sfreq"]),
            freqs=freqs,
            n_cycles=n_cycles,
            output="power",
            decim=decim,
            n_jobs=1,
        )  # (1, n_ch, n_freq, n_time)

        P = power[0]  # (n_ch, n_freq, n_time)

        # Average across channels -> (n_freq, n_time)
        Pmean = np.mean(P, axis=0)

        times = raw.times[::decim] if decim > 1 else raw.times
        return MorletResult(freqs=freqs, times=times, P=Pmean, units="V2")