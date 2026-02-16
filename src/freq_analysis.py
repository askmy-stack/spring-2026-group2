from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import mne

from yaml_utils import get


@dataclass
class PSDResult:
    freqs: np.ndarray
    psd_per_channel: np.ndarray
    psd_mean: np.ndarray
    ch_names: List[str]


class FrequencyDomainAnalyzer:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def psd(self, raw: mne.io.BaseRaw) -> PSDResult:
        method = get(self.cfg, "analysis.frequency_domain.psd.method", "welch")
        fmin = float(get(self.cfg, "analysis.frequency_domain.psd.fmin", 0.5))
        fmax = float(get(self.cfg, "analysis.frequency_domain.psd.fmax", 40.0))
        if method not in ("welch", "multitaper"):
            raise ValueError("PSD method must be 'welch' or 'multitaper'")

        picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, ecg=False, stim=False, exclude=[])
        if len(picks) == 0:
            picks = np.arange(len(raw.ch_names))

        spec = raw.compute_psd(method=method, fmin=fmin, fmax=fmax, picks=picks, verbose=False)  # MNE
        P = spec.get_data()
        freqs = spec.freqs
        ch_names = [raw.ch_names[i] for i in picks]
        mean = np.mean(P, axis=0)

        return PSDResult(freqs=freqs, psd_per_channel=P, psd_mean=mean, ch_names=ch_names)

    @staticmethod
    def _bandpower(freqs: np.ndarray, psd: np.ndarray, band: Tuple[float, float]) -> float:
        lo, hi = float(band[0]), float(band[1])
        mask = (freqs >= lo) & (freqs <= hi)
        if not np.any(mask):
            return 0.0
        return float(np.trapz(psd[mask], freqs[mask]))

    def bandpowers(self, psd_res: PSDResult) -> Dict[str, float]:
        bands = get(self.cfg, "analysis.frequency_domain.bandpower.bands", {})
        out: Dict[str, float] = {}
        for name, lim in bands.items():
            out[name] = self._bandpower(psd_res.freqs, psd_res.psd_mean, (lim[0], lim[1]))
        return out
