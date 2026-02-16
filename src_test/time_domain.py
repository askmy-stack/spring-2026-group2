from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import mne

from config_io import get
from metrics_time import compute_epoch_stats, save_epoch_stats_csv


class TimeDomainProcessor:
    """
    Step 5: Time-domain processing on a cleaned Raw (.fif) from Steps 1–4.
    Includes:
      - re-referencing
      - bad channel detection + interpolation
      - epoching (fixed-length)
      - artifact rejection (threshold-based)
      - optional ICA
    Writes:
      - cleaned raw .fif
      - epochs .fif
      - epoch stats CSV
      - plots
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.out_root = Path(get(cfg, "preprocess.out_root", "results/preprocess"))
        self.td_root = self.out_root / "time_domain"
        self.td_root.mkdir(parents=True, exist_ok=True)

        self.save_clean_raw = bool(get(cfg, "preprocess.time_domain.save_clean_raw_fif", True))
        self.save_epochs = bool(get(cfg, "preprocess.time_domain.save_epochs_fif", True))
        self.save_stats = bool(get(cfg, "preprocess.time_domain.save_epoch_stats_csv", True))
        self.save_plots = bool(get(cfg, "preprocess.time_domain.save_plots", True))

    # ---------- Helpers ----------
    def _apply_reref(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        enabled = bool(get(self.cfg, "preprocess.time_domain.reref.enabled", True))
        if not enabled:
            return raw

        reref_type = get(self.cfg, "preprocess.time_domain.reref.type", "average")
        if reref_type == "average":
            raw.set_eeg_reference("average", projection=False, verbose=False)
        elif reref_type == "custom":
            chs = get(self.cfg, "preprocess.time_domain.reref.custom_ref_channels", None)
            if not chs:
                raise ValueError("custom_ref_channels must be provided when reref.type='custom'")
            raw.set_eeg_reference(ref_channels=chs, projection=False, verbose=False)
        else:
            raise ValueError("reref.type must be 'average' or 'custom'")
        return raw

    def _detect_bad_channels(self, raw: mne.io.BaseRaw) -> List[str]:
        enabled = bool(get(self.cfg, "preprocess.time_domain.bad_channels.enabled", True))
        if not enabled:
            return []

        method = get(self.cfg, "preprocess.time_domain.bad_channels.method", "variance_zscore")
        z_thresh = float(get(self.cfg, "preprocess.time_domain.bad_channels.z_thresh", 5.0))
        flat_thresh = float(get(self.cfg, "preprocess.time_domain.bad_channels.flat_var_thresh", 1e-12))

        picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, ecg=False, stim=False, exclude=[])
        if len(picks) == 0:
            return []

        X = raw.get_data(picks=picks)  # Volts
        var = np.var(X, axis=1)

        bad_idx: List[int] = []
        # flat channels
        bad_idx.extend(list(np.where(var <= flat_thresh)[0]))

        if method == "variance_zscore":
            med = np.median(var)
            mad = np.median(np.abs(var - med)) + 1e-12
            z = 0.6745 * (var - med) / mad
            bad_idx.extend(list(np.where(np.abs(z) >= z_thresh)[0]))
        else:
            raise ValueError("bad_channels.method must be 'variance_zscore'")

        bad_idx = sorted(set([i for i in bad_idx if 0 <= i < len(picks)]))
        bad_names = [raw.ch_names[picks[i]] for i in bad_idx]
        return bad_names

    def _interpolate_bads(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        interpolate = bool(get(self.cfg, "preprocess.time_domain.bad_channels.interpolate", True))
        if interpolate and raw.info.get("bads"):
            raw.interpolate_bads(reset_bads=False, verbose=False)
        return raw

    def _make_epochs(self, raw: mne.io.BaseRaw) -> mne.Epochs:
        enabled = bool(get(self.cfg, "preprocess.time_domain.epoching.enabled", True))
        if not enabled:
            raise ValueError("Epoching disabled: time-domain step expects epochs for stats/EDA")

        mode = get(self.cfg, "preprocess.time_domain.epoching.mode", "fixed")
        if mode != "fixed":
            raise ValueError("Only epoching.mode='fixed' is implemented in this Step 5 skeleton.")

        dur = float(get(self.cfg, "preprocess.time_domain.epoching.duration_sec", 2.0))
        overlap = float(get(self.cfg, "preprocess.time_domain.epoching.overlap_sec", 0.0))
        baseline = get(self.cfg, "preprocess.time_domain.epoching.baseline", None)

        step = max(dur - overlap, 1e-6)
        epochs = mne.make_fixed_length_epochs(
            raw,
            duration=dur,
            overlap=(dur - step),
            baseline=baseline,
            preload=True,
            verbose=False,
        )
        return epochs

    def _reject_epochs(self, epochs: mne.Epochs) -> mne.Epochs:
        enabled = bool(get(self.cfg, "preprocess.time_domain.reject.enabled", True))
        if not enabled:
            return epochs

        ptp_uV = float(get(self.cfg, "preprocess.time_domain.reject.reject_eeg_ptp_uV", 500.0))
        reject = dict(eeg=ptp_uV * 1e-6)  # MNE uses Volts
        return epochs.copy().drop_bad(reject=reject)

    def _run_ica(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        enabled = bool(get(self.cfg, "preprocess.time_domain.ica.enabled", False))
        if not enabled:
            return raw

        from mne.preprocessing import ICA

        n_components = get(self.cfg, "preprocess.time_domain.ica.n_components", 0.99)
        method = get(self.cfg, "preprocess.time_domain.ica.method", "fastica")
        random_state = int(get(self.cfg, "preprocess.time_domain.ica.random_state", 97))
        max_iter = int(get(self.cfg, "preprocess.time_domain.ica.max_iter", 512))

        ica = ICA(n_components=n_components, method=method, random_state=random_state, max_iter=max_iter)
        picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, ecg=False, stim=False, exclude="bads")
        ica.fit(raw, picks=picks, verbose=False)

        # Optional EOG detection if channels exist later
        detect_eog = bool(get(self.cfg, "preprocess.time_domain.ica.detect_eog", False))
        if detect_eog:
            eog_chs = get(self.cfg, "preprocess.time_domain.ica.eog_chs", None)
            if eog_chs:
                eog_inds, _ = ica.find_bads_eog(raw, ch_name=eog_chs[0], verbose=False)
            else:
                eog_inds, _ = ica.find_bads_eog(raw, verbose=False)
            ica.exclude = list(set(ica.exclude + list(eog_inds)))

        cleaned = raw.copy()
        ica.apply(cleaned, verbose=False)

        # Save ICA plots if enabled
        if self.save_plots:
            fig_dir = self.td_root / "plots"
            fig_dir.mkdir(parents=True, exist_ok=True)
            try:
                ica.plot_components(show=False).savefig(fig_dir / f"{raw.filenames[0].stem}_ica_components.png")
            except Exception:
                pass

        return cleaned

    def _save_plots_basic(self, raw_before: mne.io.BaseRaw, raw_after: mne.io.BaseRaw, epochs: mne.Epochs, rid: str):
        if not self.save_plots:
            return
        fig_dir = self.td_root / "plots"
        fig_dir.mkdir(parents=True, exist_ok=True)

        # Raw snapshot plots (non-interactive)
        try:
            fig = raw_before.copy().pick("eeg").plot(duration=10, n_channels=10, show=False)
            fig.savefig(fig_dir / f"{rid}_raw_before.png")
        except Exception:
            pass

        try:
            fig = raw_after.copy().pick("eeg").plot(duration=10, n_channels=10, show=False)
            fig.savefig(fig_dir / f"{rid}_raw_after.png")
        except Exception:
            pass

        # Epoch image view (very useful for inspection)
        try:
            fig = epochs.plot_image(picks="eeg", show=False)
            fig.savefig(fig_dir / f"{rid}_epochs_image.png")
        except Exception:
            pass

    # ---------- Main API ----------
    def run(self, raw_in: mne.io.BaseRaw, recording_id: str) -> Dict[str, Any]:
        raw = raw_in.copy()
        if not raw.preload:
            raw.load_data()

        # 1) re-reference
        raw = self._apply_reref(raw)

        # 2) bad channel detection + interpolation
        bads = self._detect_bad_channels(raw)
        raw.info["bads"] = sorted(set(raw.info.get("bads", []) + bads))
        raw = self._interpolate_bads(raw)

        # 3) optional ICA (usually after reref; before epoching is common)
        raw = self._run_ica(raw)

        # 4) epoching
        epochs = self._make_epochs(raw)

        # 5) artifact rejection on epochs
        epochs_clean = self._reject_epochs(epochs)

        # 6) save stats CSV
        stats_path = None
        if self.save_stats:
            rows = compute_epoch_stats(epochs_clean)
            stats_path = self.td_root / "epoch_stats" / f"{recording_id}_epoch_stats.csv"
            save_epoch_stats_csv(rows, stats_path)

        # 7) save plots
        self._save_plots_basic(raw_in, raw, epochs_clean, recording_id)

        # 8) save outputs
        out_raw_path = None
        out_epochs_path = None

        if self.save_clean_raw:
            out_raw_path = self.td_root / "clean_raw" / f"{recording_id}_timedomain_raw.fif"
            out_raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw.save(out_raw_path, overwrite=True)

        if self.save_epochs:
            out_epochs_path = self.td_root / "epochs" / f"{recording_id}_timedomain-epo.fif"
            out_epochs_path.parent.mkdir(parents=True, exist_ok=True)
            epochs_clean.save(out_epochs_path, overwrite=True)

        return {
            "recording_id": recording_id,
            "bads_detected": bads,
            "n_epochs_before": int(len(epochs)),
            "n_epochs_after_reject": int(len(epochs_clean)),
            "saved_raw_fif": str(out_raw_path) if out_raw_path else None,
            "saved_epochs_fif": str(out_epochs_path) if out_epochs_path else None,
            "epoch_stats_csv": str(stats_path) if stats_path else None,
        }
