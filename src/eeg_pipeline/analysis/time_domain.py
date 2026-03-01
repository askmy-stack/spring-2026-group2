# src/eeg_pipeline/analysis/time_domain.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
import mne

from eeg_pipeline.core.yaml_utils import get


def _kurtosis_simple(x: np.ndarray) -> float:
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
class EpochStats:
    n_epochs_before: int
    n_epochs_after: int
    epoch_len_sec: float
    mean_var_uV2: float
    mean_kurtosis: float


@dataclass(frozen=True)
class Interval:
    start: float
    end: float

    @property
    def dur(self) -> float:
        return max(0.0, self.end - self.start)


class TimeDomainModule:
    """
    Time-domain processing:
      - re-referencing
      - QC metrics
      - mark bad channels + interpolate
      - fixed-length epoching
      - artifact rejection (drop bad epochs)
      - epoch statistics (variance, kurtosis)
      - ICA (optional) for eye/muscle artifacts
      - seizure interval parsing from events.tsv
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    # -------------------------
    # Re-reference
    # -------------------------
    def rereference(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        if not bool(get(self.cfg, "analysis.time_domain.reref.enabled", False)):
            return raw

        ref_channels = get(self.cfg, "analysis.time_domain.reref.ref_channels", None)  # null => average
        projection = bool(get(self.cfg, "analysis.time_domain.reref.projection", False))

        out = raw.copy()
        out.set_eeg_reference(ref_channels=ref_channels, projection=projection, verbose=False)
        if not projection:
            out.apply_proj(verbose=False)
        return out

    # -------------------------
    # QC
    # -------------------------
    def qc(self, raw: mne.io.BaseRaw) -> Dict[str, Any]:
        if not bool(get(self.cfg, "analysis.time_domain.qc.enabled", True)):
            return {}

        max_abs_uV = float(get(self.cfg, "analysis.time_domain.qc.max_abs_uV", 500.0))
        flat_var_thresh = float(get(self.cfg, "analysis.time_domain.qc.flat_var_thresh_uV2", 1e-12))
        nan_allowed = bool(get(self.cfg, "analysis.time_domain.qc.nan_allowed", False))
        noisy_factor = float(get(self.cfg, "analysis.time_domain.qc.noisy_var_factor", 10.0))

        picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, ecg=False, stim=False, exclude=[])
        if len(picks) == 0:
            picks = np.arange(len(raw.ch_names))

        X = raw.get_data(picks=picks) * 1e6  # µV

        has_nan = bool(np.isnan(X).any())
        nan_frac = float(np.mean(np.isnan(X))) if has_nan else 0.0
        if (not nan_allowed) and has_nan:
            # keep flagging; downstream can decide
            pass

        X = np.nan_to_num(X, nan=0.0)

        # existing:
        ch_var = np.var(X, axis=1)
        ch_maxabs = np.max(np.abs(X), axis=1)
        med_var = float(np.median(ch_var)) if ch_var.size else 0.0

        flat = np.where(ch_var <= flat_var_thresh)[0].tolist()
        clipped = np.where(ch_maxabs >= max_abs_uV)[0].tolist()

        noisy = []
        if med_var > 0:
            noisy = np.where(ch_var >= noisy_factor * med_var)[0].tolist()

        ch_kurt = np.array([_kurtosis_simple(X[i]) for i in range(X.shape[0])], dtype=float)

        # ADD THESE summaries:
        var_summary = {
            "min": float(np.min(ch_var)) if ch_var.size else float("nan"),
            "median": float(np.median(ch_var)) if ch_var.size else float("nan"),
            "max": float(np.max(ch_var)) if ch_var.size else float("nan"),
        }
        maxabs_summary = {
            "min": float(np.min(ch_maxabs)) if ch_maxabs.size else float("nan"),
            "median": float(np.median(ch_maxabs)) if ch_maxabs.size else float("nan"),
            "max": float(np.max(ch_maxabs)) if ch_maxabs.size else float("nan"),
        }

        # Map indices -> channel names (so it’s human readable)
        ch_names = [raw.ch_names[p] for p in picks] if len(picks) else raw.ch_names
        flat_names = [ch_names[i] for i in flat if i < len(ch_names)]
        clipped_names = [ch_names[i] for i in clipped if i < len(ch_names)]
        noisy_names = [ch_names[i] for i in noisy if i < len(ch_names)]

        return {
            "sfreq": float(raw.info["sfreq"]),
            "n_channels_eeg": int(len(picks)),
            "has_nan": has_nan,
            "nan_frac": nan_frac,

            "median_var_uV2": med_var,
            "var_uV2_summary": var_summary,
            "maxabs_uV_summary": maxabs_summary,

            "flat_channels_idx": flat,
            "clipped_channels_idx": clipped,
            "noisy_channels_idx": noisy,

            "flat_channels": flat_names,
            "clipped_channels": clipped_names,
            "noisy_channels": noisy_names,

            "n_flat": int(len(flat)),
            "n_clipped": int(len(clipped)),
            "n_noisy": int(len(noisy)),
            "noisy_channel_frac": float(len(noisy) / max(1, len(picks))),

            "mean_kurtosis": float(np.nanmean(ch_kurt)) if ch_kurt.size else float("nan"),
        }

    # -------------------------
    # Bad channels
    # -------------------------
    def mark_bads_from_qc(self, raw: mne.io.BaseRaw, qc_payload: Dict[str, Any]) -> List[str]:

        if not bool(get(self.cfg, "analysis.time_domain.bad_channels.enabled", True)):
            return raw.info.get("bads", []) or []

        if not bool(get(self.cfg, "analysis.time_domain.bad_channels.use_qc_rules", True)):
            return raw.info.get("bads", []) or []

        bad_idx = set()
        for k in ("flat_channels_idx", "clipped_channels_idx", "noisy_channels_idx"):
            for i in qc_payload.get(k, []) or []:
                bad_idx.add(int(i))

        picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, ecg=False, stim=False, exclude=[])
        if len(picks) == 0:
            bad_names = [raw.ch_names[i] for i in sorted(bad_idx) if 0 <= i < len(raw.ch_names)]
        else:
            bad_names = [raw.ch_names[picks[i]] for i in sorted(bad_idx) if 0 <= i < len(picks)]
        min_good = int(get(self.cfg, "analysis.time_domain.bad_channels.min_good_channels", 4))
        n_total = len(raw.ch_names)
        n_bad = len(bad_names)

        # If we're about to mark too many bad channels, skip or cap
        if (n_total - n_bad) < min_good:
            # keep none (or keep only the worst few if you implement ranking)
            return raw.info.get("bads", []) or []
        raw.info["bads"] = sorted(set((raw.info.get("bads", []) or []) + bad_names))
        return raw.info["bads"]

    def interpolate_bads(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        if not bool(get(self.cfg, "analysis.time_domain.bad_channels.enabled", True)):
            return raw
        if not bool(get(self.cfg, "analysis.time_domain.bad_channels.interpolate", True)):
            return raw
        if len(raw.info.get("bads", [])) == 0:
            return raw

        require_montage = bool(get(self.cfg, "analysis.time_domain.bad_channels.require_montage", False))
        has_pos = raw.get_montage() is not None
        if require_montage and not has_pos:
            return raw

        out = raw.copy()
        out.interpolate_bads(reset_bads=False, verbose=False)
        return out

    # -------------------------
    # Epoching + artifact rejection
    # -------------------------
    def make_epochs(self, raw: mne.io.BaseRaw) -> Optional[mne.Epochs]:
        if not bool(get(self.cfg, "analysis.time_domain.epoching.enabled", False)):
            return None

        length_sec = float(get(self.cfg, "analysis.time_domain.epoching.length_sec", 2.0))
        overlap_sec = float(get(self.cfg, "analysis.time_domain.epoching.overlap_sec", 0.0))
        baseline = get(self.cfg, "analysis.time_domain.epoching.baseline", None)

        epochs = mne.make_fixed_length_epochs(
            raw,
            duration=length_sec,
            overlap=overlap_sec,
            preload=True,
            verbose=False,
        )
        if baseline is not None:
            epochs.apply_baseline(baseline)
        return epochs

    def reject_artifact_epochs(self, epochs: mne.Epochs) -> mne.Epochs:
        if not bool(get(self.cfg, "analysis.time_domain.artifact_rejection.enabled", False)):
            return epochs

        reject = get(self.cfg, "analysis.time_domain.artifact_rejection.reject", None)
        flat = get(self.cfg, "analysis.time_domain.artifact_rejection.flat", None)

        def _cast_dict(d):
            if not isinstance(d, dict):
                return d
            out = {}
            for k, v in d.items():
                if isinstance(v, str):
                    try:
                        out[k] = float(v)
                    except Exception:
                        out[k] = v
                else:
                    out[k] = v
            return out

        reject = _cast_dict(reject)
        flat = _cast_dict(flat)

        out = epochs.copy()
        out.drop_bad(reject=reject, flat=flat)
        return out

    def epoch_stats(self, epochs_before: mne.Epochs, epochs_after: mne.Epochs) -> EpochStats:
        X = epochs_after.get_data() * 1e6  # µV
        vars_uV2 = np.var(X, axis=2)
        mean_var = float(np.mean(vars_uV2)) if vars_uV2.size else float("nan")

        flat = X.reshape(-1, X.shape[-1]) if X.size else np.zeros((0, 0))
        kurt = np.array([_kurtosis_simple(flat[i]) for i in range(flat.shape[0])], dtype=float) if flat.size else np.array([], dtype=float)
        mean_k = float(np.nanmean(kurt)) if kurt.size else float("nan")

        return EpochStats(
            n_epochs_before=int(len(epochs_before)),
            n_epochs_after=int(len(epochs_after)),
            epoch_len_sec=float(epochs_after.tmax - epochs_after.tmin),
            mean_var_uV2=mean_var,
            mean_kurtosis=mean_k,
        )
    #
    # # -------------------------
    # # ICA
    # # -------------------------
    def apply_ica(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        if not bool(get(self.cfg, "analysis.time_domain.ica.enabled", False)):
            return raw

        method = str(get(self.cfg, "analysis.time_domain.ica.method", "fastica"))
        n_components = get(self.cfg, "analysis.time_domain.ica.n_components", None)
        random_state = int(get(self.cfg, "analysis.time_domain.ica.random_state", 42))
        max_iter = int(get(self.cfg, "analysis.time_domain.ica.max_iter", 512))

        eog_channel = get(self.cfg, "analysis.time_domain.ica.eog_channel", None)
        auto_detect = bool(get(self.cfg, "analysis.time_domain.ica.auto_detect", False))

        out = raw.copy()
        if not out.preload:
            out.load_data()

        picks = mne.pick_types(out.info, eeg=True, meg=False, eog=False, ecg=False, stim=False, exclude=[])
        if len(picks) == 0:
            picks = np.arange(len(out.ch_names))

        ica = mne.preprocessing.ICA(
            n_components=n_components,
            method=method,
            random_state=random_state,
            max_iter=max_iter,
            verbose=False,
        )
        ica.fit(out, picks=picks, verbose=False)

        if auto_detect and eog_channel and eog_channel in out.ch_names:
            inds, _ = ica.find_bads_eog(out, ch_name=str(eog_channel), verbose=False)
            ica.exclude.extend(inds)

        return ica.apply(out, verbose=False)

    # -------------------------
    # Seizure intervals from events.tsv
    # -------------------------
    def load_seizure_intervals_from_tsv(self, tsv_path: str) -> List[Interval]:
        onset_col = str(get(self.cfg, "analysis.labels.tsv.onset_col", "onset"))
        dur_col = str(get(self.cfg, "analysis.labels.tsv.duration_col", "duration"))
        label_col = str(get(self.cfg, "analysis.labels.tsv.label_col", "trial_type"))

        value_col = str(get(self.cfg, "analysis.labels.tsv.value_col", "value"))
        positive_values = get(self.cfg, "analysis.labels.tsv.positive_values", [1, "1", True, "true"])
        positive_values_norm = {str(v).strip().lower() for v in (positive_values or [])}

        seizure_values = get(self.cfg, "analysis.labels.tsv.seizure_values", ["seizure"])
        seizure_values_norm = {str(x).strip().lower() for x in (seizure_values or [])}

        try:
            df = pd.read_csv(tsv_path, sep="\t")
        except Exception:
            return []
        if df.empty:
            return []

        def _find_col(wanted: str) -> Optional[str]:
            if wanted in df.columns:
                return wanted
            wl = wanted.lower()
            for c in df.columns:
                if c.lower() == wl:
                    return c
            return None

        onset_c = _find_col(onset_col)
        dur_c = _find_col(dur_col)
        label_c = _find_col(label_col)
        value_c = _find_col(value_col)

        if onset_c is None or dur_c is None:
            return []

        out: List[Interval] = []
        for _, row in df.iterrows():
            is_seiz = False

            if label_c is not None:
                lab = str(row.get(label_c, "")).strip().lower()
                if lab in seizure_values_norm:
                    is_seiz = True

            if (not is_seiz) and (value_c is not None):
                v = str(row.get(value_c, "")).strip().lower()
                if v in positive_values_norm:
                    if label_c is None:
                        is_seiz = True
                    else:
                        lab = str(row.get(label_c, "")).strip().lower()
                        if (lab in seizure_values_norm) or (lab == "" or lab == "nan"):
                            is_seiz = True

            if not is_seiz:
                continue

            try:
                onset = float(row[onset_c])
                dur = float(row[dur_c])
            except Exception:
                continue

            if not np.isfinite(onset) or not np.isfinite(dur) or dur <= 0:
                continue

            out.append(Interval(start=float(onset), end=float(onset + dur)))

        return self._merge_intervals(out)

    @staticmethod
    def _merge_intervals(intervals: List[Interval], gap: float = 0.0) -> List[Interval]:
        if not intervals:
            return []
        xs = sorted(intervals, key=lambda z: z.start)
        merged = [xs[0]]
        for cur in xs[1:]:
            last = merged[-1]
            if cur.start <= last.end + gap:
                merged[-1] = Interval(start=last.start, end=max(last.end, cur.end))
            else:
                merged.append(cur)
        return merged