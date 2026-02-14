from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import mne

from config_io import get
from filtering import FilterApplier
from qc_psd import QCAnalyzer, PSDAnalyzer, save_qc_json, save_psd_csv


class EEGPreprocessor:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

        out_root = get(cfg, "preprocess.out_root", "results/preprocess")
        self.out_root = Path(out_root)
        self.out_root.mkdir(parents=True, exist_ok=True)

        self.cleaned_dir = self.out_root / "cleaned_raw"
        self.qc_dir = self.out_root / "qc"
        self.psd_dir = self.out_root / "psd"

        for d in [self.cleaned_dir, self.qc_dir, self.psd_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Resample
        self.target_sfreq = get(cfg, "preprocess.resample.target_sfreq", None)

        # Filtering base config
        self.f_l = get(cfg, "preprocess.filter.l_freq", None)
        self.f_h = get(cfg, "preprocess.filter.h_freq", None)
        self.notch = get(cfg, "preprocess.filter.notch_freqs", None)
        self.method = get(cfg, "preprocess.filter.method", "fir")
        self.iir_params = get(cfg, "preprocess.filter.iir_params", None)

        # Wavelet params (manual flag remains, but auto may override)
        self.manual_use_wavelet = bool(get(cfg, "preprocess.filter.use_wavelet", False))
        self.wav_family = get(cfg, "preprocess.filter.wavelet.family", "db4")
        self.wav_level = get(cfg, "preprocess.filter.wavelet.level", None)
        self.wav_thr = get(cfg, "preprocess.filter.wavelet.threshold", "soft")

        # PSD settings
        self.psd_enabled = bool(get(cfg, "preprocess.psd.enabled", True))
        self.psd_method = get(cfg, "preprocess.psd.method", "welch")
        self.psd_fmin = float(get(cfg, "preprocess.psd.fmin", 0.5))
        self.psd_fmax = float(get(cfg, "preprocess.psd.fmax", 40.0))
        self.psd_save_csv = bool(get(cfg, "preprocess.psd.save_csv", True))

        # QC settings
        self.qc_enabled = bool(get(cfg, "preprocess.qc.enabled", True))
        self.qc_max_abs = float(get(cfg, "preprocess.qc.max_abs_uV", 500.0))
        self.qc_flat_var = float(get(cfg, "preprocess.qc.flat_var_thresh", 1e-12))
        self.qc_nan_allowed = bool(get(cfg, "preprocess.qc.nan_allowed", False))

        # Wavelet decision rules
        self.auto_wavelet = bool(get(cfg, "preprocess.wavelet.auto_enable", False))
        self.line_noise_ratio_threshold = float(get(cfg, "preprocess.wavelet.line_noise_ratio_threshold", 0.08))
        self.line_noise_bandwidth = float(get(cfg, "preprocess.wavelet.line_noise_bandwidth_hz", 1.0))
        self.noisy_channel_frac_threshold = float(get(cfg, "preprocess.wavelet.noisy_channel_frac_threshold", 0.30))

        # Build analyzers
        self.qc_analyzer = QCAnalyzer(
            max_abs_uV=self.qc_max_abs,
            flat_var_thresh=self.qc_flat_var,
            nan_allowed=self.qc_nan_allowed,
        )
        self.psd_analyzer = PSDAnalyzer(method=self.psd_method, fmin=self.psd_fmin, fmax=self.psd_fmax)

    def _build_filter(self, use_wavelet: bool) -> FilterApplier:
        return FilterApplier(
            l_freq=self.f_l,
            h_freq=self.f_h,
            notch_freqs=self.notch,
            method=self.method,
            iir_params=self.iir_params,
            use_wavelet=use_wavelet,
            wavelet=self.wav_family,
            wavelet_level=self.wav_level,
            wavelet_threshold=self.wav_thr,
        )

    def _should_enable_wavelet(self, qc: Dict[str, Any], psd_freqs, psd_mean) -> Dict[str, Any]:
        """
        Returns decision + reasons.
        """
        reasons = []

        # If notch list includes 50/60, compute ratios for those
        ratios = {}
        if self.notch:
            for lf in self.notch:
                ratios[str(lf)] = self.psd_analyzer.line_noise_ratio(
                    psd_freqs, psd_mean, line_freq=float(lf), bandwidth_hz=self.line_noise_bandwidth
                )
                if ratios[str(lf)] >= self.line_noise_ratio_threshold:
                    reasons.append(f"high_line_noise_ratio@{lf}Hz={ratios[str(lf)]:.3f}")

        noisy_frac = float(qc.get("noisy_channel_frac", 0.0))
        if noisy_frac >= self.noisy_channel_frac_threshold:
            reasons.append(f"high_noisy_channel_frac={noisy_frac:.2f}")

        enable = (len(reasons) > 0)
        return {"enable_wavelet": enable, "reasons": reasons, "line_noise_ratios": ratios}

    def run(self, raw: mne.io.BaseRaw, recording_id: str, *, save_clean_fif: bool = True) -> Dict[str, Any]:
        out = raw.copy()
        if not out.preload:
            out.load_data()

        # ---- Resample (signal processing)
        if self.target_sfreq is not None:
            cur = float(out.info["sfreq"])
            tgt = float(self.target_sfreq)
            if abs(cur - tgt) > 1e-6:
                out.resample(sfreq=tgt, npad="auto", verbose=False)

        # ---- Base filtering (wavelet OFF for evaluation unless manual says ON)
        base_filter = self._build_filter(use_wavelet=False)
        out_base = base_filter.apply(out)

        qc_base = self.qc_analyzer.compute(out_base) if self.qc_enabled else {}
        psd_base = self.psd_analyzer.compute(out_base) if self.psd_enabled else None

        decision = {"enable_wavelet": self.manual_use_wavelet, "reasons": ["manual_setting"], "line_noise_ratios": {}}

        if self.auto_wavelet and (psd_base is not None) and self.qc_enabled:
            decision = self._should_enable_wavelet(qc_base, psd_base.freqs, psd_base.psd_mean)

        # If manual use_wavelet is true, override auto decision
        if self.manual_use_wavelet:
            decision = {"enable_wavelet": True, "reasons": ["manual_setting"], "line_noise_ratios": decision.get("line_noise_ratios", {})}

        # ---- Optionally apply wavelet and re-evaluate QC/PSD
        final_raw = out_base
        qc_final = qc_base
        psd_final = psd_base
        used_wavelet = False

        if decision.get("enable_wavelet", False):
            wave_filter = self._build_filter(use_wavelet=True)
            final_raw = wave_filter.apply(out)
            used_wavelet = True
            qc_final = self.qc_analyzer.compute(final_raw) if self.qc_enabled else qc_base
            psd_final = self.psd_analyzer.compute(final_raw) if self.psd_enabled else psd_base

        # ---- Save QC + PSD artifacts
        if self.qc_enabled:
            qc_payload = {
                "recording_id": recording_id,
                "used_wavelet": used_wavelet,
                "decision": decision,
                "qc_base": qc_base,
                "qc_final": qc_final,
            }
            save_qc_json(qc_payload, self.qc_dir / f"{recording_id}_qc.json")

        if self.psd_enabled and self.psd_save_csv and (psd_final is not None):
            save_psd_csv(psd_final, self.psd_dir / f"{recording_id}_psd_mean.csv")

        # ---- Save cleaned raw
        out_fif = None
        if save_clean_fif:
            tag = "wave" if used_wavelet else "nowave"
            out_fif = self.cleaned_dir / f"{recording_id}_sf{int(final_raw.info['sfreq'])}_{tag}_raw.fif"
            final_raw.save(out_fif, overwrite=True)

        return {
            "recording_id": recording_id,
            "saved_clean_fif": str(out_fif) if out_fif else None,
            "final_sfreq": float(final_raw.info["sfreq"]),
            "used_wavelet": used_wavelet,
            "decision": decision,
        }
