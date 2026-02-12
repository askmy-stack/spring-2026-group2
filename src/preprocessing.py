# src/preprocessing.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import mne


class EEGPreprocessor:
    """
    YAML-only (dict) preprocessing:
      - keep EEG channels only
      - resample to target_sfreq
      - notch + bandpass
      - re-reference (optional)
      - channel QC + bad channel interpolation (optional)
    """

    def __init__(self, cfg: Dict[str, Any], cleaned_dir: Path):
        self.cfg = cfg
        self.cleaned_dir = cleaned_dir

    # ---------- helpers to read cfg safely ----------
    def _signal(self) -> Dict[str, Any]:
        return self.cfg.get("signal", {})

    def _qc(self) -> Dict[str, Any]:
        return self.cfg.get("qc", {})

    def _cache(self) -> Dict[str, Any]:
        return self.cfg.get("cache", {})

    # -------------------------
    # Channel QC (generic)
    # -------------------------
    def channel_qc(self, raw: mne.io.BaseRaw) -> pd.DataFrame:
        """
        Computes:
          - std_uv
          - ptp_uv
          - line_noise_ratio around notch frequency (cfg.signal.notch_hz, default 60)
        """
        signal = self._signal()
        qc_cfg = self._qc()

        data = raw.get_data()  # volts (n_ch, n_time)
        sf = float(raw.info["sfreq"])

        std_uv = (data * 1e6).std(axis=1)
        ptp_uv = (data * 1e6).ptp(axis=1)

        bandpass = signal.get("bandpass_hz", [1.0, 50.0])
        fmax = float(bandpass[1])

        # PSD for line-noise ratio
        psds, freqs = mne.time_frequency.psd_array_welch(
            data,
            sfreq=sf,
            fmin=1.0,
            fmax=fmax,
            n_fft=int(sf * 2),  # 2s Welch segment (generic default)
            verbose=False,
        )

        notch_hz = signal.get("notch_hz", 60.0)
        ln = float(notch_hz if notch_hz is not None else 60.0)

        band = (freqs >= (ln - 1.0)) & (freqs <= (ln + 1.0))
        neigh = (freqs >= (ln - 5.0)) & (freqs <= (ln + 5.0)) & (~band)
        line_ratio = (psds[:, band].mean(axis=1) + 1e-12) / (psds[:, neigh].mean(axis=1) + 1e-12)

        out = pd.DataFrame(
            {
                "ch_name": raw.ch_names,
                "std_uv": std_uv,
                "ptp_uv": ptp_uv,
                "line_noise_ratio": line_ratio,
            }
        )

        flat_std_uv = float(qc_cfg.get("flat_std_uv", 0.5))
        high_std_uv = float(qc_cfg.get("high_std_uv", 200.0))
        ln_ratio_thr = float(qc_cfg.get("line_noise_ratio_thresh", 3.0))

        out["is_flat"] = out["std_uv"] < flat_std_uv
        out["is_high_var"] = out["std_uv"] > high_std_uv
        out["is_line_noise"] = out["line_noise_ratio"] > ln_ratio_thr
        out["is_bad_suspect"] = out["is_flat"] | out["is_high_var"] | out["is_line_noise"]

        return out

    # -------------------------
    # Main preprocessing
    # -------------------------
    def preprocess(self, raw: mne.io.BaseRaw) -> Tuple[mne.io.BaseRaw, pd.DataFrame, Dict[str, Any]]:
        signal = self._signal()
        qc_cfg = self._qc()

        raw = raw.copy().load_data()

        # EEG only
        picks = mne.pick_types(
            raw.info,
            eeg=True,
            eog=False,
            ecg=False,
            emg=False,
            stim=False,
            exclude=[],
        )
        raw.pick(picks)
        raw.set_channel_types({ch: "eeg" for ch in raw.ch_names})

        # Resample
        target_sfreq = int(signal.get("target_sfreq", 256))
        if int(raw.info["sfreq"]) != target_sfreq:
            raw.resample(target_sfreq, npad="auto", verbose=False)

        # Notch (optional)
        notch_hz = signal.get("notch_hz", 60)
        if notch_hz is not None:
            raw.notch_filter(freqs=[float(notch_hz)], verbose=False)

        # Bandpass
        bandpass = signal.get("bandpass_hz", [1.0, 50.0])
        raw.filter(float(bandpass[0]), float(bandpass[1]), verbose=False)

        # Reference
        reference = str(signal.get("reference", "average"))
        if reference == "average":
            raw.set_eeg_reference("average", verbose=False)

        # QC
        qc = self.channel_qc(raw)

        # Mark bad channels (do not drop; interpolate later if feasible)
        bads = qc.loc[qc["is_bad_suspect"], "ch_name"].tolist()
        raw.info["bads"] = sorted(set(bads))

        max_bad_frac = float(qc_cfg.get("max_bad_channel_frac", 0.30))

        meta: Dict[str, Any] = {
            "sfreq": float(raw.info["sfreq"]),
            "n_channels": int(len(raw.ch_names)),
            "n_bads": int(len(raw.info["bads"])),
            "bad_frac": float(len(raw.info["bads"]) / max(1, len(raw.ch_names))),
            "reference": reference,
            "notch_hz": float(notch_hz) if notch_hz is not None else None,
            "bandpass_lo": float(bandpass[0]),
            "bandpass_hi": float(bandpass[1]),
        }

        # Interpolate bads if not too many
        if meta["bad_frac"] <= max_bad_frac and meta["n_bads"] > 0:
            try:
                raw.interpolate_bads(reset_bads=False, verbose=False)
                meta["interpolated_bads"] = True
            except Exception as ex:
                meta["interpolated_bads"] = False
                meta["interpolate_error"] = str(ex)
        else:
            meta["interpolated_bads"] = False

        return raw, qc, meta

    # -------------------------
    # Cache cleaned fif
    # -------------------------
    def cache_cleaned(self, raw_clean: mne.io.BaseRaw, subject_id: str, edf_file: str):
        cache_cfg = self._cache()
        save_cleaned = bool(cache_cfg.get("save_cleaned_raw", True))
        overwrite = bool(cache_cfg.get("overwrite_cleaned_raw", True))

        if not save_cleaned:
            return

        stem = edf_file.replace(".edf", "")
        out = self.cleaned_dir / f"{subject_id}__{stem}_clean_raw.fif"
        if overwrite or (not out.exists()):
            raw_clean.save(out.as_posix(), overwrite=True, verbose=False)
