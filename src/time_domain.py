from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
import mne

from yaml_utils import get


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
    Time-domain processing (NO ICA):
      - QC metrics (bad channel detection signals)
      - mark bad channels + interpolate
      - fixed-length epoching
      - artifact rejection (epoch drop)
      - epoch statistics (variance, kurtosis)

    Also:
      - parse seizure intervals from *_events.tsv
      - sample matched non-seizure intervals
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    # ---------------- QC ----------------
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

        X = raw.get_data(picks=picks) * 1e6  # uV

        has_nan = bool(np.isnan(X).any())
        nan_frac = float(np.mean(np.isnan(X))) if has_nan else 0.0
        if (not nan_allowed) and has_nan:
            pass

        X = np.nan_to_num(X, nan=0.0)

        ch_var = np.var(X, axis=1)
        ch_maxabs = np.max(np.abs(X), axis=1)
        med_var = float(np.median(ch_var)) if ch_var.size else 0.0

        flat = np.where(ch_var <= flat_var_thresh)[0].tolist()
        clipped = np.where(ch_maxabs >= max_abs_uV)[0].tolist()

        noisy: List[int] = []
        if med_var > 0:
            noisy = np.where(ch_var >= noisy_factor * med_var)[0].tolist()

        ch_kurt = np.array([_kurtosis_simple(X[i]) for i in range(X.shape[0])], dtype=float)

        return {
            "sfreq": float(raw.info["sfreq"]),
            "n_channels_eeg": int(len(picks)),
            "has_nan": has_nan,
            "nan_frac": nan_frac,
            "median_var_uV2": med_var,
            "flat_channels_idx": flat,
            "clipped_channels_idx": clipped,
            "noisy_channels_idx": noisy,
            "noisy_channel_frac": float(len(noisy) / max(1, len(picks))),
            "mean_kurtosis": float(np.nanmean(ch_kurt)) if ch_kurt.size else float("nan"),
        }

    # -------- Bad channels: mark + interpolate --------
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
        out.interpolate_bads(reset_bads=False, verbose=False)  # MNE
        return out

    # ---------------- Epoching ----------------
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

    # ---------------- Artifact rejection ----------------
    def reject_artifact_epochs(self, epochs: mne.Epochs) -> mne.Epochs:
        if not bool(get(self.cfg, "analysis.time_domain.artifact_rejection.enabled", False)):
            return epochs

        reject = get(self.cfg, "analysis.time_domain.artifact_rejection.reject", None)
        flat = get(self.cfg, "analysis.time_domain.artifact_rejection.flat", None)

        out = epochs.copy()
        out.drop_bad(reject=reject, flat=flat)  # MNE
        return out

    # ---------------- Epoch stats ----------------
    def epoch_stats(self, epochs_before: mne.Epochs, epochs_after: mne.Epochs) -> EpochStats:
        X = epochs_after.get_data()  # Volts
        X_uV = X * 1e6

        vars_uV2 = np.var(X_uV, axis=2)
        mean_var = float(np.mean(vars_uV2)) if vars_uV2.size else float("nan")

        flat = X_uV.reshape(-1, X_uV.shape[-1]) if X_uV.size else np.zeros((0, 0))
        kurt = np.array([_kurtosis_simple(flat[i]) for i in range(flat.shape[0])], dtype=float) if flat.size else np.array([], dtype=float)
        mean_k = float(np.nanmean(kurt)) if kurt.size else float("nan")

        return EpochStats(
            n_epochs_before=int(len(epochs_before)),
            n_epochs_after=int(len(epochs_after)),
            epoch_len_sec=float(epochs_after.tmax - epochs_after.tmin),
            mean_var_uV2=mean_var,
            mean_kurtosis=mean_kurtosis if (mean_kurtosis := mean_k) else mean_k,
        )

    # =========================
    # TSV labels: seizure vs non-seizure
    # =========================
    def load_seizure_intervals_from_tsv(self, tsv_path: str) -> List[Interval]:
        """
        Reads BIDS *_events.tsv and returns seizure intervals.

        Uses config:
          analysis.labels.tsv.onset_col
          analysis.labels.tsv.duration_col
          analysis.labels.tsv.label_col
          analysis.labels.tsv.seizure_values
        """
        onset_col = str(get(self.cfg, "analysis.labels.tsv.onset_col", "onset"))
        dur_col = str(get(self.cfg, "analysis.labels.tsv.duration_col", "duration"))
        label_col = str(get(self.cfg, "analysis.labels.tsv.label_col", "trial_type"))
        seizure_values = [str(x).lower() for x in (get(self.cfg, "analysis.labels.tsv.seizure_values", ["seizure"]) or [])]

        p = tsv_path
        text = open(p, "r", encoding="utf-8").read().strip().splitlines()
        if not text:
            return []

        header = text[0].split("\t")
        idx = {h: i for i, h in enumerate(header)}

        def _get(row: List[str], col: str) -> str:
            i = idx.get(col, None)
            return row[i] if i is not None and i < len(row) else ""

        out: List[Interval] = []
        for line in text[1:]:
            if not line.strip():
                continue
            row = line.split("\t")
            lab = _get(row, label_col).strip().lower()
            if lab not in seizure_values:
                continue
            try:
                onset = float(_get(row, onset_col))
                dur = float(_get(row, dur_col))
            except Exception:
                continue
            if dur <= 0:
                continue
            out.append(Interval(start=onset, end=onset + dur))

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

    @staticmethod
    def _complement_intervals(total: Interval, blocked: List[Interval]) -> List[Interval]:
        """
        Returns intervals inside `total` not covered by `blocked`.
        Assumes `blocked` is merged + sorted.
        """
        if not blocked:
            return [total]
        out: List[Interval] = []
        cur = total.start
        for b in blocked:
            if b.start > cur:
                out.append(Interval(cur, min(b.start, total.end)))
            cur = max(cur, b.end)
            if cur >= total.end:
                break
        if cur < total.end:
            out.append(Interval(cur, total.end))
        return [i for i in out if i.dur > 0]

    def sample_nonseizure_intervals(
        self,
        *,
        raw: mne.io.BaseRaw,
        seizure_intervals: List[Interval],
        seed: int,
    ) -> List[Interval]:
        """
        Samples non-seizure intervals matching seizure durations and count.
        Sampling is done from the complement of seizure intervals.
        """
        total = Interval(0.0, float(raw.times[-1]) if raw.n_times else 0.0)
        seizure_intervals = self._merge_intervals([Interval(max(0.0, s.start), min(total.end, s.end)) for s in seizure_intervals])

        free = self._complement_intervals(total, seizure_intervals)
        if not free:
            return []

        policy = str(get(self.cfg, "analysis.comparisons.non_seizure_policy", "match_seizure_count"))
        rng = np.random.RandomState(seed)

        target = seizure_intervals if policy == "match_seizure_count" else seizure_intervals
        out: List[Interval] = []

        # Flatten free intervals into candidate start ranges per required duration
        for s in target:
            dur = s.dur
            candidates: List[Interval] = []
            for f in free:
                if f.dur >= dur:
                    candidates.append(Interval(f.start, f.end - dur))
            if not candidates:
                continue

            # Choose one candidate interval, then choose a start within it
            c = candidates[int(rng.randint(0, len(candidates)))]
            start = float(rng.uniform(c.start, c.end))
            out.append(Interval(start=start, end=start + dur))

            # Block this chosen region to avoid overlaps in later sampling
            seizure_intervals = self._merge_intervals(seizure_intervals + [out[-1]])
            free = self._complement_intervals(total, seizure_intervals)
            if not free:
                break

        return out
