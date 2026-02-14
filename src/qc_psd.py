from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import mne


def _kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis-free simple kurtosis (not Fisher-corrected)."""
    x = np.asarray(x, dtype=float)
    if x.size < 4:
        return float("nan")
    mu = np.mean(x)
    v = np.mean((x - mu) ** 2)
    if v <= 0:
        return float("nan")
    m4 = np.mean((x - mu) ** 4)
    return float(m4 / (v ** 2))


@dataclass
class PSDResult:
    freqs: np.ndarray
    psd_mean: np.ndarray         # averaged over channels
    psd_per_channel: np.ndarray  # shape: (n_channels, n_freqs)
    ch_names: List[str]


class PSDAnalyzer:
    """
    Computes PSD (Welch or multitaper) using MNE.
    Also computes line-noise ratio around 50/60 Hz.
    """

    def __init__(
        self,
        method: str = "welch",
        fmin: float = 0.5,
        fmax: float = 40.0,
    ):
        if method not in ("welch", "multitaper"):
            raise ValueError("PSD method must be 'welch' or 'multitaper'.")
        self.method = method
        self.fmin = fmin
        self.fmax = fmax

    def compute(self, raw: mne.io.BaseRaw) -> PSDResult:
        eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, ecg=False, stim=False, exclude=[])
        if len(eeg_picks) == 0:
            # fall back to all channels
            eeg_picks = np.arange(len(raw.ch_names))

        spec = raw.compute_psd(
            method=self.method,
            fmin=self.fmin,
            fmax=self.fmax,
            picks=eeg_picks,
            verbose=False,
        )
        psd = spec.get_data()  # (n_channels, n_freqs)
        freqs = spec.freqs
        ch_names = [raw.ch_names[i] for i in eeg_picks]

        psd_mean = np.mean(psd, axis=0)
        return PSDResult(freqs=freqs, psd_mean=psd_mean, psd_per_channel=psd, ch_names=ch_names)

    @staticmethod
    def line_noise_ratio(
        freqs: np.ndarray,
        psd_mean: np.ndarray,
        line_freq: float,
        bandwidth_hz: float = 1.0,
    ) -> float:
        """
        Ratio = power in [line_freq - bw, line_freq + bw] / total power.
        Uses mean PSD across channels.
        """
        freqs = np.asarray(freqs)
        psd_mean = np.asarray(psd_mean)

        total = np.trapz(psd_mean, freqs)
        if total <= 0:
            return 0.0

        lo = line_freq - bandwidth_hz
        hi = line_freq + bandwidth_hz
        mask = (freqs >= lo) & (freqs <= hi)
        if not np.any(mask):
            return 0.0

        band_power = np.trapz(psd_mean[mask], freqs[mask])
        return float(band_power / total)


class QCAnalyzer:
    """
    Computes basic QC metrics from time-domain EEG.
    """

    def __init__(self, max_abs_uV: float = 500.0, flat_var_thresh: float = 1e-12, nan_allowed: bool = False):
        self.max_abs_uV = max_abs_uV
        self.flat_var_thresh = flat_var_thresh
        self.nan_allowed = nan_allowed

    def compute(self, raw: mne.io.BaseRaw) -> Dict[str, Any]:
        eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, ecg=False, stim=False, exclude=[])
        if len(eeg_picks) == 0:
            eeg_picks = np.arange(len(raw.ch_names))

        data = raw.get_data(picks=eeg_picks)
        # Assume data in Volts for MNE; convert to microvolts for amplitude metrics
        data_uV = data * 1e6

        has_nan = bool(np.isnan(data_uV).any())
        if (not self.nan_allowed) and has_nan:
            nan_frac = float(np.mean(np.isnan(data_uV)))
        else:
            nan_frac = float(np.mean(np.isnan(data_uV))) if has_nan else 0.0

        # Replace NaNs for stats to avoid propagating
        data_uV = np.nan_to_num(data_uV, nan=0.0)

        ch_var = np.var(data_uV, axis=1)
        ch_maxabs = np.max(np.abs(data_uV), axis=1)
        ch_kurt = np.array([_kurtosis(data_uV[i]) for i in range(data_uV.shape[0])], dtype=float)

        flat_channels = np.where(ch_var <= self.flat_var_thresh)[0].tolist()
        clipped_channels = np.where(ch_maxabs >= self.max_abs_uV)[0].tolist()

        # Define “noisy” channels by high variance relative to median
        med_var = float(np.median(ch_var)) if ch_var.size else 0.0
        noisy = []
        if med_var > 0:
            noisy = np.where(ch_var >= 10.0 * med_var)[0].tolist()

        return {
            "sfreq": float(raw.info["sfreq"]),
            "n_channels_eeg": int(len(eeg_picks)),
            "has_nan": has_nan,
            "nan_frac": nan_frac,
            "median_var_uV2": med_var,
            "flat_channels_idx": flat_channels,
            "clipped_channels_idx": clipped_channels,
            "noisy_channels_idx": noisy,
            "noisy_channel_frac": float(len(noisy) / max(1, len(eeg_picks))),
            "mean_kurtosis": float(np.nanmean(ch_kurt)) if ch_kurt.size else float("nan"),
        }


def save_qc_json(qc: Dict[str, Any], out_path: Path) -> None:
    import json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(qc, indent=2))


def save_psd_csv(psd: PSDResult, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # CSV: freq, mean_psd
    lines = ["freq,psd_mean"]
    for f, p in zip(psd.freqs, psd.psd_mean):
        lines.append(f"{float(f)},{float(p)}")
    out_path.write_text("\n".join(lines))
