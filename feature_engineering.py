import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis


class AdvancedFeatureExtractor:
    """
    Computes per-channel time + frequency features from a window.

    Input:  data shape (n_channels, n_times)
    Output: dict of features (tabular row)
    """

    def __init__(self, sfreq=256, cfg=None):
        self.sfreq = float(sfreq)
        self.cfg = cfg or {}

        # read bands from cfg["fe"]["frequency"]["bands"]
        fe_cfg = self.cfg.get("fe", {})
        bands = (
            fe_cfg.get("frequency", {}).get("bands")
            or {
                "delta": (0.5, 4),
                "theta": (4, 8),
                "alpha": (8, 13),
                "beta": (13, 30),
                "gamma": (30, 40),
            }
        )
        # Normalize YAML lists -> tuples
        self.bands = {k: tuple(v) for k, v in bands.items()}

    # -------------------
    # Time-domain helpers
    # -------------------
    def _line_length(self, x):
        return np.sum(np.abs(np.diff(x)))

    def _zero_crossing_rate(self, x):
        return float(np.mean(np.diff(np.signbit(x)) != 0))

    def _hjorth(self, x):
        activity = np.var(x)

        dx = np.diff(x)
        var_dx = np.var(dx)
        mobility = np.sqrt(var_dx / (activity + 1e-12))

        ddx = np.diff(dx)
        var_ddx = np.var(ddx)
        mobility_dx = np.sqrt(var_ddx / (var_dx + 1e-12))
        complexity = mobility_dx / (mobility + 1e-12)

        return float(activity), float(mobility), float(complexity)

    # ------------------------
    # Frequency-domain helpers
    # ------------------------
    def _bandpower(self, freqs, psd, band):
        lo, hi = float(band[0]), float(band[1])
        idx = (freqs >= lo) & (freqs <= hi)
        return float(np.trapezoid(psd[idx], freqs[idx])) if np.any(idx) else 0.0

    def _spectral_entropy(self, psd):
        p = psd / (np.sum(psd) + 1e-12)
        p = np.clip(p, 1e-12, 1.0)
        return float(-np.sum(p * np.log(p)))

    # -------------------
    # Main extraction
    # -------------------
    def extract(self, data):
        feats = {}

        fe_cfg = self.cfg.get("fe", {})
        time_cfg = fe_cfg.get("time", {})
        freq_cfg = fe_cfg.get("frequency", {})

        n_channels = data.shape[0]

        for ch in range(n_channels):
            x = data[ch, :].astype(np.float64, copy=False)
            prefix = f"ch{ch}_"

            # ---- time domain ----
            if time_cfg.get("mean", True):
                feats[prefix + "mean"] = float(np.mean(x))
            if time_cfg.get("std", True):
                feats[prefix + "std"] = float(np.std(x))
            if time_cfg.get("rms", True):
                feats[prefix + "rms"] = float(np.sqrt(np.mean(x ** 2)))
            if time_cfg.get("line_length", True):
                feats[prefix + "line_length"] = float(self._line_length(x))
            if time_cfg.get("zero_crossing_rate", True):
                feats[prefix + "zcr"] = float(self._zero_crossing_rate(x))
            if time_cfg.get("skew", True):
                feats[prefix + "skew"] = float(skew(x))
            if time_cfg.get("kurtosis", True):
                feats[prefix + "kurtosis"] = float(kurtosis(x))
            if time_cfg.get("hjorth", True):
                act, mob, comp = self._hjorth(x)
                feats[prefix + "hjorth_activity"] = act
                feats[prefix + "hjorth_mobility"] = mob
                feats[prefix + "hjorth_complexity"] = comp

            # ---- frequency domain ----
            if freq_cfg:
                welch_cfg = freq_cfg.get("welch", {})
                nperseg_sec = float(welch_cfg.get("nperseg_sec", 2))
                fmin = float(welch_cfg.get("fmin", 0.5))
                fmax = float(welch_cfg.get("fmax", 50))

                # Nyquist safety: fmax cannot exceed sfreq/2
                nyq = 0.5 * self.sfreq
                if fmax > nyq:
                    fmax = nyq

                nperseg = int(nperseg_sec * self.sfreq)
                nperseg = min(nperseg, len(x))

                freqs, psd = welch(
                    x,
                    fs=self.sfreq,
                    nperseg=nperseg,
                    noverlap=0,
                    scaling="density",
                )

                # limit freq range
                mask = (freqs >= fmin) & (freqs <= fmax)
                freqs = freqs[mask]
                psd = psd[mask]

                total_power = (
                    float(np.trapezoid(psd, freqs)) if len(freqs) > 1 else 0.0
                )
                feats[prefix + "total_power"] = total_power

                # band powers
                band_powers = {}
                for band_name, band_rng in self.bands.items():
                    bp = self._bandpower(freqs, psd, band_rng)
                    feats[prefix + f"{band_name}_power"] = bp
                    band_powers[band_name] = bp

                # relative power
                if freq_cfg.get("relative_power", True):
                    for band_name, bp in band_powers.items():
                        feats[prefix + f"{band_name}_rel"] = float(
                            bp / (total_power + 1e-12)
                        )

                # spectral entropy
                if freq_cfg.get("spectral_entropy", True):
                    feats[prefix + "spec_entropy"] = self._spectral_entropy(psd)

        return feats
