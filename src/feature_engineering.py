import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis
import pywt


class AdvancedFeatureExtractor:
    """
    Extracts window-level EEG features.

    Input:
        data: np.ndarray of shape (n_channels, n_times)

    Output:
        dict[str, float] containing per-channel and optional
        connectivity summary features.

    Feature groups (config-controlled):
        - Time-domain statistics + Hjorth parameters
        - Nonlinear metrics (entropy, complexity)
        - Frequency-domain (Welch PSD, band power, spectral entropy)
        - FFT dominant frequency
        - Wavelet energy + entropy
        - Channel connectivity (correlation summaries)
    """

    def __init__(self, sfreq=256, cfg=None):
        self.sfreq = float(sfreq)
        self.cfg = cfg or {}

        fe_cfg = self.cfg.get("fe", {})
        bands = (
            fe_cfg.get("frequency", {}).get("bands")
            or {
                "delta": (0.5, 4),
                "theta": (4, 8),
                "alpha": (8, 13),
                "beta": (13, 30),
                "gamma": (30, 50),
            }
        )
        self.bands = {k: tuple(v) for k, v in bands.items()}

    # ---------- Time-domain helpers ----------

    def _line_length(self, x):
        return float(np.sum(np.abs(np.diff(x))))

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

    # ---------- Frequency-domain helpers ----------

    def _bandpower(self, freqs, psd, band):
        lo, hi = float(band[0]), float(band[1])
        idx = (freqs >= lo) & (freqs <= hi)
        return float(np.trapezoid(psd[idx], freqs[idx])) if np.any(idx) else 0.0

    def _spectral_entropy(self, psd):
        p = psd / (np.sum(psd) + 1e-12)
        p = np.clip(p, 1e-12, 1.0)
        return float(-np.sum(p * np.log(p)))

    def _dominant_fft_freq(self, x, fmin=0.5, fmax=50.0, detrend=True, window=True):
        """
        Returns dominant frequency (Hz) within [fmin, fmax].
        """
        x = x.astype(np.float64, copy=False)

        if detrend:
            x = x - np.mean(x)

        if window:
            x = x * np.hamming(len(x))

        N = len(x)
        if N < 2:
            return 0.0

        freqs = np.fft.rfftfreq(N, d=1.0 / self.sfreq)
        mag = np.abs(np.fft.rfft(x))

        nyq = 0.5 * self.sfreq
        fmax = min(float(fmax), float(nyq))

        mask = (freqs >= float(fmin)) & (freqs <= float(fmax))
        if not np.any(mask):
            return 0.0

        freqs_band = freqs[mask]
        mag_band = mag[mask]

        return float(freqs_band[int(np.argmax(mag_band))])

    # ---------- Nonlinear helpers ----------

    def _sample_entropy(self, x, m=2, r=0.2):
        """
        Sample Entropy with Chebyshev distance.
        r is expressed as a fraction of std(x).
        """
        x = np.asarray(x, dtype=np.float64)
        n = len(x)

        if n < (m + 2):
            return 0.0

        sd = np.std(x)
        if sd < 1e-12:
            return 0.0

        tol = r * sd

        def _phi(mm):
            emb = np.array([x[i : i + mm] for i in range(n - mm + 1)])
            total = emb.shape[0]
            count = 0

            for i in range(total):
                dist = np.max(np.abs(emb - emb[i]), axis=1)
                count += np.sum(dist <= tol) - 1

            return count / (total * (total - 1) + 1e-12)

        A = _phi(m + 1)
        B = _phi(m)

        if B <= 1e-12 or A <= 1e-12:
            return 0.0

        return float(-np.log(A / B))

    def _perm_entropy(self, x, order=3, delay=1):
        """
        Permutation entropy (unnormalized).
        """
        x = np.asarray(x, dtype=np.float64)
        n = len(x)

        if n < order * delay + 1:
            return 0.0

        patterns = {}
        for i in range(n - delay * (order - 1)):
            window = x[i : i + delay * order : delay]
            key = tuple(np.argsort(window))
            patterns[key] = patterns.get(key, 0) + 1

        counts = np.array(list(patterns.values()), dtype=np.float64)
        p = counts / (np.sum(counts) + 1e-12)
        p = np.clip(p, 1e-12, 1.0)

        return float(-np.sum(p * np.log(p)))

    def _lz_complexity(self, x, n_bins=10):
        """
        Lempel–Ziv complexity on quantile-discretized signal.
        """
        x = np.asarray(x, dtype=np.float64)

        if len(x) < 10:
            return 0.0

        qs = np.quantile(x, np.linspace(0, 1, n_bins + 1))
        s = np.digitize(x, qs[1:-1])

        i, k = 0, 1
        c = 1
        n = len(s)

        while True:
            if i + k == n:
                c += 1
                break

            substring = tuple(s[i : i + k])
            found = False

            for j in range(0, i):
                if j + k <= n and tuple(s[j : j + k]) == substring:
                    found = True
                    break

            if found:
                k += 1
                if i + k > n:
                    c += 1
                    break
            else:
                c += 1
                i += k
                k = 1
                if i >= n:
                    break

        return float(c)

    # ---------- Wavelet helpers ----------

    def _wavelet_features(self, x, wavelet="db4", level=4):
        """
        Returns:
            - energy per coefficient level
            - wavelet entropy
        """
        x = np.asarray(x, dtype=np.float64)

        if len(x) < 8:
            return {}, 0.0

        max_level = pywt.dwt_max_level(len(x), pywt.Wavelet(wavelet).dec_len)
        level = int(min(level, max_level))

        if level < 1:
            return {}, 0.0

        coeffs = pywt.wavedec(x, wavelet, level=level)
        energies = [float(np.sum(c**2)) for c in coeffs]

        total = float(np.sum(energies) + 1e-12)
        p = np.clip(np.array(energies) / total, 1e-12, 1.0)
        wentropy = float(-np.sum(p * np.log(p)))

        feats = {"wav_E_A": energies[0]}

        for i in range(1, len(energies)):
            feats[f"wav_E_D{i}"] = energies[i]

        return feats, wentropy

    # ---------- Main extraction ----------

    def extract(self, data: np.ndarray) -> dict:
        feats = {}

        fe_cfg = self.cfg.get("fe", {})
        time_cfg = fe_cfg.get("time", {})
        nl_cfg = fe_cfg.get("nonlinear", {})
        freq_cfg = fe_cfg.get("frequency", {})
        wav_cfg = fe_cfg.get("wavelet", {})
        conn_cfg = fe_cfg.get("connectivity", {})

        n_channels = int(data.shape[0])

        for ch in range(n_channels):
            x = data[ch, :].astype(np.float64, copy=False)
            prefix = f"ch{ch}_"

            # Time-domain
            if time_cfg.get("mean", True):
                feats[prefix + "mean"] = float(np.mean(x))
            if time_cfg.get("std", True):
                feats[prefix + "std"] = float(np.std(x))
            if time_cfg.get("rms", True):
                feats[prefix + "rms"] = float(np.sqrt(np.mean(x**2)))
            if time_cfg.get("line_length", True):
                feats[prefix + "line_length"] = self._line_length(x)
            if time_cfg.get("zero_crossing_rate", True):
                feats[prefix + "zcr"] = self._zero_crossing_rate(x)
            if time_cfg.get("skew", True):
                feats[prefix + "skew"] = float(skew(x))
            if time_cfg.get("kurtosis", True):
                feats[prefix + "kurtosis"] = float(kurtosis(x))

            if time_cfg.get("hjorth", True):
                act, mob, comp = self._hjorth(x)
                feats[prefix + "hjorth_activity"] = act
                feats[prefix + "hjorth_mobility"] = mob
                feats[prefix + "hjorth_complexity"] = comp

            # Nonlinear
            if nl_cfg.get("sample_entropy", False):
                feats[prefix + "sampen"] = self._sample_entropy(
                    x,
                    m=int(nl_cfg.get("sampen_m", 2)),
                    r=float(nl_cfg.get("sampen_r", 0.2)),
                )

            if nl_cfg.get("perm_entropy", False):
                feats[prefix + "perm_entropy"] = self._perm_entropy(
                    x,
                    order=int(nl_cfg.get("perm_order", 3)),
                    delay=int(nl_cfg.get("perm_delay", 1)),
                )

            if nl_cfg.get("lz_complexity", False):
                feats[prefix + "lz_complexity"] = self._lz_complexity(
                    x,
                    n_bins=int(nl_cfg.get("lz_bins", 10)),
                )

            # Frequency-domain (Welch)
            if freq_cfg:
                welch_cfg = freq_cfg.get("welch", {})
                nperseg_sec = float(welch_cfg.get("nperseg_sec", 2))
                fmin = float(welch_cfg.get("fmin", 0.5))
                fmax = float(welch_cfg.get("fmax", 50))

                nyq = 0.5 * self.sfreq
                fmax = min(fmax, nyq)

                nperseg = min(int(nperseg_sec * self.sfreq), len(x))
                if nperseg < 4:
                    nperseg = len(x)

                freqs, psd = welch(
                    x,
                    fs=self.sfreq,
                    nperseg=nperseg,
                    noverlap=0,
                    scaling="density",
                )

                mask = (freqs >= fmin) & (freqs <= fmax)
                freqs = freqs[mask]
                psd = psd[mask]

                total_power = (
                    float(np.trapezoid(psd, freqs)) if len(freqs) > 1 else 0.0
                )
                feats[prefix + "total_power"] = total_power

                band_powers = {}
                for band_name, band_rng in self.bands.items():
                    bp = self._bandpower(freqs, psd, band_rng)
                    feats[prefix + f"{band_name}_power"] = bp
                    band_powers[band_name] = bp

                if freq_cfg.get("relative_power", True):
                    for band_name, bp in band_powers.items():
                        feats[prefix + f"{band_name}_rel"] = float(
                            bp / (total_power + 1e-12)
                        )

                if freq_cfg.get("spectral_entropy", True):
                    feats[prefix + "spec_entropy"] = self._spectral_entropy(psd)

                if freq_cfg.get("fft_dominant_freq", True):
                    fft_cfg = freq_cfg.get("fft", {})
                    feats[prefix + "fft_dom_freq"] = self._dominant_fft_freq(
                        x,
                        fmin=float(fft_cfg.get("fmin", 0.5)),
                        fmax=float(fft_cfg.get("fmax", 50)),
                        detrend=bool(fft_cfg.get("detrend", True)),
                        window=bool(fft_cfg.get("window", True)),
                    )

            # Wavelet
            if wav_cfg.get("enabled", False):
                wf, went = self._wavelet_features(
                    x,
                    wavelet=str(wav_cfg.get("wavelet", "db4")),
                    level=int(wav_cfg.get("level", 4)),
                )
                for k, v in wf.items():
                    feats[prefix + k] = float(v)

                if wav_cfg.get("entropy", True):
                    feats[prefix + "wav_entropy"] = float(went)

        # Connectivity (window-level)
        if conn_cfg.get("enabled", False) and n_channels >= 2:
            X = data.astype(np.float64, copy=False)
            C = np.corrcoef(X)

            iu = np.triu_indices_from(C, k=1)
            vals = C[iu]
            vals = vals[np.isfinite(vals)]

            if vals.size == 0:
                vals = np.array([0.0], dtype=np.float64)

            feats["conn_corr_mean"] = float(np.mean(vals))
            feats["conn_corr_std"] = float(np.std(vals))
            feats["conn_corr_max"] = float(np.max(vals))
            feats["conn_corr_min"] = float(np.min(vals))

        return feats
