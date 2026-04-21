import math
from collections import Counter
import numpy as np
import pywt
from scipy.signal import welch
from scipy.stats import kurtosis, skew
from numba import njit, prange
try:
    import cupy as cp
    _GPU = True
except ImportError:
    cp = None
    _GPU = False
def _to_numpy(x):
    if _GPU and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)
@njit(cache=True)
def _sampen_numba(x, m, r):
    n = x.shape[0]
    if n < m + 2:
        return np.nan
    sd = np.std(x)
    if sd < 1e-12:
        return 0.0
    tol = r * sd
    n_templates = n - m
    b = 0
    a = 0
    for i in range(n_templates - 1):
        for j in range(i + 1, n_templates):
            d = 0.0
            for k in range(m):
                diff = abs(x[i + k] - x[j + k])
                if diff > d:
                    d = diff
            if d <= tol:
                b += 1
                diff_extra = abs(x[i + m] - x[j + m])
                d_m1 = d if d > diff_extra else diff_extra
                if d_m1 <= tol:
                    a += 1
    if b == 0:
        return np.nan
    if a == 0:
        return np.inf
    return -np.log(a / b)
@njit(cache=True)
def _perm_entropy_numba(x, order, delay):
    n = x.shape[0]
    min_len = 1 + (order - 1) * delay
    if n < min_len:
        return 0.0
    n_patterns = n - (order - 1) * delay
    codes = np.empty(n_patterns, dtype=np.int64)
    for i in range(n_patterns):
        idx = np.empty(order, dtype=np.int64)
        for k in range(order):
            idx[k] = k
        vals = np.empty(order, dtype=np.float64)
        for k in range(order):
            vals[k] = x[i + k * delay]
        for a in range(1, order):
            key_v = vals[a]
            key_i = idx[a]
            b = a - 1
            while b >= 0 and vals[b] > key_v:
                vals[b + 1] = vals[b]
                idx[b + 1] = idx[b]
                b -= 1
            vals[b + 1] = key_v
            idx[b + 1] = key_i
        code = 0
        base = 1
        for k in range(order):
            code += idx[k] * base
            base *= order
        codes[i] = code
    codes.sort()
    counts_list = np.empty(n_patterns, dtype=np.int64)
    n_unique = 0
    prev = codes[0]
    cur_count = 1
    for i in range(1, n_patterns):
        if codes[i] == prev:
            cur_count += 1
        else:
            counts_list[n_unique] = cur_count
            n_unique += 1
            prev = codes[i]
            cur_count = 1
    counts_list[n_unique] = cur_count
    n_unique += 1
    total = np.float64(n_patterns)
    pe = 0.0
    for i in range(n_unique):
        p = counts_list[i] / total
        pe -= p * np.log2(p)
    return pe
class AdvancedFeatureExtractor:
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
    @staticmethod
    def _bandpower(freqs, psd, band):
        lo, hi = float(band[0]), float(band[1])
        idx = (freqs >= lo) & (freqs <= hi)
        _trapz = getattr(np, "trapezoid", np.trapz)
        return float(_trapz(psd[idx], freqs[idx])) if np.any(idx) else 0.0
    @staticmethod
    def _spectral_entropy(psd, normalize=True):
        psd = np.asarray(psd, dtype=np.float64)
        total = np.sum(psd)
        if total <= 1e-12:
            return 0.0
        p = np.clip(psd / total, 1e-12, 1.0)
        se = -np.sum(p * np.log(p))
        if normalize and len(p) > 1:
            se /= np.log(len(p))
        return float(se)
    @staticmethod
    def _wavelet_features(x, wavelet="db4", level=4):
        x = np.asarray(x, dtype=np.float64)

        def _zero_feats(lvl):
            """Return zero-valued dict with correct keys to prevent NaN."""
            zf = {"wav_E_A": 0.0}
            for i in range(1, lvl + 1):
                zf[f"wav_E_D{i}"] = 0.0
            return zf, 0.0

        if len(x) < 8:
            return _zero_feats(level)

        max_level = pywt.dwt_max_level(len(x), pywt.Wavelet(wavelet).dec_len)
        level = int(min(level, max_level))

        if level < 1:
            return _zero_feats(1)

        coeffs = pywt.wavedec(x, wavelet, level=level)
        energies = np.array([np.sum(c ** 2) for c in coeffs], dtype=np.float64)
        total = energies.sum()

        if total <= 1e-12:
            return _zero_feats(level)

        p = np.clip(energies / total, 1e-12, 1.0)
        wentropy = float(-np.sum(p * np.log(p)))
        feats = {"wav_E_A": float(energies[0])}
        for i in range(1, len(energies)):
            feats[f"wav_E_D{level - i + 1}"] = float(energies[i])
        return feats, wentropy
    def extract(self, data):
        feats = {}
        data_np = np.asarray(data, dtype=np.float64)
        if data_np.ndim != 2:
            raise ValueError("Input data must have shape (n_channels, n_times)")
        fe_cfg = self.cfg.get("fe", {})
        time_cfg = fe_cfg.get("time", {})
        nl_cfg = fe_cfg.get("nonlinear", {})
        freq_cfg = fe_cfg.get("frequency", {})
        wav_cfg = fe_cfg.get("wavelet", {})
        n_ch = int(data_np.shape[0])
        xp = cp if _GPU else np
        d = xp.asarray(data_np) if _GPU else data_np
        if time_cfg.get("mean", True):
            v = _to_numpy(xp.mean(d, axis=1))
            for ch in range(n_ch):
                feats[f"ch{ch}_mean"] = float(v[ch])
        if time_cfg.get("std", True):
            v = _to_numpy(xp.std(d, axis=1))
            for ch in range(n_ch):
                feats[f"ch{ch}_std"] = float(v[ch])
        if time_cfg.get("rms", True):
            v = _to_numpy(xp.sqrt(xp.mean(d ** 2, axis=1)))
            for ch in range(n_ch):
                feats[f"ch{ch}_rms"] = float(v[ch])
        if time_cfg.get("min", False):
            v = _to_numpy(xp.min(d, axis=1))
            for ch in range(n_ch):
                feats[f"ch{ch}_min"] = float(v[ch])
        if time_cfg.get("max", False):
            v = _to_numpy(xp.max(d, axis=1))
            for ch in range(n_ch):
                feats[f"ch{ch}_max"] = float(v[ch])
        if time_cfg.get("range", False):
            v = _to_numpy(xp.max(d, axis=1) - xp.min(d, axis=1))
            for ch in range(n_ch):
                feats[f"ch{ch}_range"] = float(v[ch])
        if time_cfg.get("line_length", True):
            v = _to_numpy(xp.sum(xp.abs(xp.diff(d, axis=1)), axis=1))
            for ch in range(n_ch):
                feats[f"ch{ch}_line_length"] = float(v[ch])
        if time_cfg.get("zero_crossing_rate", True):
            v = _to_numpy(
                xp.mean(xp.diff(xp.signbit(d).astype(xp.int8), axis=1) != 0, axis=1)
            )
            for ch in range(n_ch):
                feats[f"ch{ch}_zcr"] = float(v[ch])
        if time_cfg.get("skew", True):
            v = np.nan_to_num(skew(data_np, axis=1), nan=0.0)
            for ch in range(n_ch):
                feats[f"ch{ch}_skew"] = float(v[ch])
        if time_cfg.get("kurtosis", True):
            v = np.nan_to_num(kurtosis(data_np, axis=1), nan=0.0)
            for ch in range(n_ch):
                feats[f"ch{ch}_kurtosis"] = float(v[ch])
        if time_cfg.get("hjorth", True):
            activity = _to_numpy(xp.var(d, axis=1))
            dx = xp.diff(d, axis=1)
            var_dx = _to_numpy(xp.var(dx, axis=1))
            mobility = np.sqrt(var_dx / (activity + 1e-12))
            ddx = xp.diff(dx, axis=1)
            var_ddx = _to_numpy(xp.var(ddx, axis=1))
            mobility_dx = np.sqrt(var_ddx / (var_dx + 1e-12))
            complexity = mobility_dx / (mobility + 1e-12)
            for ch in range(n_ch):
                feats[f"ch{ch}_hjorth_activity"] = float(activity[ch])
                feats[f"ch{ch}_hjorth_mobility"] = float(mobility[ch])
                feats[f"ch{ch}_hjorth_complexity"] = float(complexity[ch])
        if nl_cfg.get("sample_entropy", False):
            m = int(nl_cfg.get("sampen_m", 2))
            r = float(nl_cfg.get("sampen_r", 0.2))
            for ch in range(n_ch):
                feats[f"ch{ch}_sampen"] = float(_sampen_numba(data_np[ch], m, r))
        if nl_cfg.get("perm_entropy", False):
            order = int(nl_cfg.get("perm_order", 3))
            delay = int(nl_cfg.get("perm_delay", 1))
            norm_denom = np.log2(math.factorial(order))
            for ch in range(n_ch):
                pe = _perm_entropy_numba(data_np[ch], order, delay)
                feats[f"ch{ch}_perm_entropy"] = float(pe / norm_denom)
        if freq_cfg:
            welch_cfg = freq_cfg.get("welch", {})
            nperseg_sec = float(welch_cfg.get("nperseg_sec", 2))
            fmin_w = float(welch_cfg.get("fmin", 0.5))
            fmax_w = float(welch_cfg.get("fmax", 50.0))
            nyq = 0.5 * self.sfreq
            fmax_w = min(fmax_w, nyq)
            nperseg = max(4, min(int(nperseg_sec * self.sfreq), data_np.shape[1]))
            freqs_w, psd_all = welch(
                data_np, fs=self.sfreq,
                nperseg=nperseg, noverlap=0,
                scaling="density", axis=-1,
            )
            mask_w = (freqs_w >= fmin_w) & (freqs_w <= fmax_w)
            freqs_w = freqs_w[mask_w]
            psd_all = psd_all[:, mask_w]
            for ch in range(n_ch):
                psd = psd_all[ch]
                pfx = f"ch{ch}_"
                _trapz = getattr(np, "trapezoid", np.trapz)
                total_power = float(_trapz(psd, freqs_w)) if len(freqs_w) > 1 else 0.0
                feats[pfx + "total_power"] = total_power
                band_powers = {}
                for bname, brng in self.bands.items():
                    bp = self._bandpower(freqs_w, psd, brng)
                    feats[pfx + f"{bname}_power"] = bp
                    band_powers[bname] = bp
                if freq_cfg.get("relative_power", True):
                    for bname, bp in band_powers.items():
                        feats[pfx + f"{bname}_rel"] = float(bp / (total_power + 1e-12))
                if freq_cfg.get("spectral_entropy", True):
                    feats[pfx + "spec_entropy"] = self._spectral_entropy(psd, normalize=True)
            if freq_cfg.get("fft_dominant_freq", True):
                fft_cfg = freq_cfg.get("fft", {})
                f_fmin = float(fft_cfg.get("fmin", 0.5))
                f_fmax = float(fft_cfg.get("fmax", 50.0))
                f_fmax = min(f_fmax, nyq)
                d_fft = d - xp.mean(d, axis=1, keepdims=True) if fft_cfg.get("detrend", True) else d
                if fft_cfg.get("window", True):
                    win = xp.asarray(np.hamming(data_np.shape[1]).astype(np.float64))
                    d_fft = d_fft * win
                freqs_fft = np.fft.rfftfreq(data_np.shape[1], d=1.0 / self.sfreq)
                fmask = (freqs_fft >= f_fmin) & (freqs_fft <= f_fmax)
                if np.any(fmask):
                    if _GPU:
                        mag = cp.asnumpy(cp.abs(cp.fft.rfft(d_fft, axis=1)))
                    else:
                        mag = np.abs(np.fft.rfft(d_fft, axis=1))
                    dom_idx = np.argmax(mag[:, fmask], axis=1)
                    dom_freqs = freqs_fft[fmask][dom_idx]
                    for ch in range(n_ch):
                        feats[f"ch{ch}_fft_dom_freq"] = float(dom_freqs[ch])
                else:
                    for ch in range(n_ch):
                        feats[f"ch{ch}_fft_dom_freq"] = 0.0
        if wav_cfg.get("enabled", False):
            wname = str(wav_cfg.get("wavelet", "db4"))
            level = int(wav_cfg.get("level", 4))
            do_entropy = wav_cfg.get("entropy", True)
            for ch in range(n_ch):
                wf, went = self._wavelet_features(data_np[ch], wavelet=wname, level=level)
                pfx = f"ch{ch}_"
                for k, v in wf.items():
                    feats[pfx + k] = float(v)
                if do_entropy:
                    feats[pfx + "wav_entropy"] = float(went)
        return feats
