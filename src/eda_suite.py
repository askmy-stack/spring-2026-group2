# src/eda_suite.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional, Set, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mne


# -------------------------
# Interval utilities
# -------------------------
def merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    out = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = out[-1]
        if s <= pe:
            out[-1] = (ps, max(pe, e))
        else:
            out.append((s, e))
    return out


def apply_buffer(intervals: List[Tuple[float, float]], buffer_sec: float, duration: float) -> List[Tuple[float, float]]:
    buffed = [(max(0.0, s - buffer_sec), min(duration, e + buffer_sec)) for s, e in intervals]
    return merge_intervals(buffed)


def overlaps_any(t0: float, t1: float, intervals: List[Tuple[float, float]]) -> bool:
    for s, e in intervals:
        if max(t0, s) < min(t1, e):
            return True
    return False


# -------------------------
# Subject sampler (no repeats across EDA topics)
# -------------------------
class SubjectSampler:
    """
    Ensures: for each EDA idea, we choose K subjects randomly,
    and we do NOT repeat subjects across different EDA ideas.
    """
    def __init__(self, subjects: List[str], seed: int):
        self.rng = np.random.default_rng(seed)
        self.pool = subjects[:]
        self.used: Set[str] = set()

    def sample(self, k: int) -> List[str]:
        avail = [s for s in self.pool if s not in self.used]
        if len(avail) == 0:
            return []
        k = min(k, len(avail))
        picked = list(self.rng.choice(avail, size=k, replace=False))
        for s in picked:
            self.used.add(s)
        return picked


# -------------------------
# EDA Engine (YAML-only dict config)
# -------------------------
class EDAEngine:
    def __init__(self, cfg: Dict[str, Any], figs_dir: Path, tables_dir: Path):
        self.cfg = cfg
        self.figs_dir = figs_dir
        self.tables_dir = tables_dir

    # ---------- helpers to read cfg safely ----------
    def _eda(self) -> Dict[str, Any]:
        return self.cfg.get("eda", {})

    def _signal(self) -> Dict[str, Any]:
        return self.cfg.get("signal", {})

    def _demo(self) -> Dict[str, Any]:
        return self.cfg.get("demographics", {})

    def clip_sec(self) -> float:
        return float(self._eda().get("clip_sec", 30.0))

    def seizure_buffer_sec(self) -> float:
        return float(self._eda().get("seizure_buffer_sec", 0.0))

    def nonseizure_min_gap_sec(self) -> float:
        return float(self._eda().get("nonseizure_min_gap_sec", 60.0))

    def max_channels_to_plot(self) -> int:
        return int(self._eda().get("max_channels_to_plot", 12))

    def channel_pick_strategy(self) -> str:
        return str(self._eda().get("channel_pick_strategy", "lowest_std"))

    def buffer_grid_sec(self) -> List[int]:
        return list(self._eda().get("buffer_grid_sec", [0, 10, 30, 60]))

    def fmax(self) -> float:
        bp = self._signal().get("bandpass_hz", [1.0, 50.0])
        return float(bp[1])

    # ---------- helpers ----------
    def pick_channels(self, raw: mne.io.BaseRaw) -> List[str]:
        k = self.max_channels_to_plot()
        if k >= len(raw.ch_names):
            return raw.ch_names

        strat = self.channel_pick_strategy()
        if strat == "first":
            return raw.ch_names[:k]

        # lowest_std = pick more stable channels for cleaner plots
        data_uv = raw.get_data() * 1e6
        std = data_uv.std(axis=1)
        idx = np.argsort(std)[:k]
        return [raw.ch_names[i] for i in idx]

    # ---------- clip selection ----------
    def choose_seizure_clip_start(
        self, seizure_intervals: List[Tuple[float, float]], duration: float
    ) -> Optional[float]:
        if not seizure_intervals:
            return None
        s, e = seizure_intervals[0]
        mid = 0.5 * (s + e)
        start = mid - self.clip_sec() / 2
        return float(max(0.0, min(duration - self.clip_sec(), start)))

    def choose_nonseizure_clip_start(
        self, duration: float, avoid_intervals: List[Tuple[float, float]]
    ) -> Optional[float]:
        # Expand avoid intervals by extra nonseizure gap
        expanded = apply_buffer(avoid_intervals, self.nonseizure_min_gap_sec(), duration)

        seed = int(self._eda().get("random_seed", 42))
        rng = np.random.default_rng(seed)

        clip = self.clip_sec()
        for _ in range(300):
            t0 = float(rng.uniform(0, max(1e-6, duration - clip)))
            if not overlaps_any(t0, t0 + clip, expanded):
                return t0

        # scan fallback
        step = max(1.0, clip / 2)
        t0 = 0.0
        while t0 + clip <= duration:
            if not overlaps_any(t0, t0 + clip, expanded):
                return t0
            t0 += step
        return None

    # ---------- plots ----------
    def plot_clip_with_labels(self, raw: mne.io.BaseRaw, t0: float, out_png: Path, title: str):
        clip = self.clip_sec()
        t1 = min(float(raw.times[-1]), t0 + clip)
        chs = self.pick_channels(raw)
        seg = raw.copy().pick(chs).crop(tmin=t0, tmax=t1, include_tmax=False)
        x = seg.get_data() * 1e6
        t = np.linspace(0, (t1 - t0), x.shape[1])

        plt.figure(figsize=(12, 7))
        offset = 0.0
        for i, ch in enumerate(chs):
            plt.plot(t, x[i] + offset)
            plt.text(t[0], offset, ch, fontsize=9, va="bottom")
            offset += np.percentile(np.abs(x[i]), 95) * 2 + 50.0

        plt.title(title)
        plt.xlabel("Time (sec)")
        plt.ylabel("Amplitude (µV) + offset")
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()

    def plot_psd_compare(self, raw: mne.io.BaseRaw, seiz_t0: float, non_t0: float, out_png: Path):
        chs = self.pick_channels(raw)
        sf = float(raw.info["sfreq"])
        clip = self.clip_sec()
        fmax = self.fmax()

        seiz = raw.copy().pick(chs).crop(
            tmin=seiz_t0, tmax=seiz_t0 + clip, include_tmax=False
        ).get_data()
        non = raw.copy().pick(chs).crop(
            tmin=non_t0, tmax=non_t0 + clip, include_tmax=False
        ).get_data()

        ps_s, freqs = mne.time_frequency.psd_array_welch(
            seiz, sfreq=sf, fmin=1.0, fmax=fmax, n_fft=int(sf * 2), verbose=False
        )
        ps_n, _ = mne.time_frequency.psd_array_welch(
            non, sfreq=sf, fmin=1.0, fmax=fmax, n_fft=int(sf * 2), verbose=False
        )

        plt.figure(figsize=(10, 5))
        plt.semilogy(freqs, ps_n.mean(axis=0), label=f"non-seizure {clip:.0f}s")
        plt.semilogy(freqs, ps_s.mean(axis=0), label=f"seizure {clip:.0f}s")
        plt.title("PSD compare (subject-wise clips)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power (V^2/Hz)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()

    def plot_spectrogram(self, raw: mne.io.BaseRaw, t0: float, out_png: Path, title: str):
        """
        Spectrogram for a clip:
          - pick channels
          - average channels into one trace
          - matplotlib.specgram
        """
        clip = self.clip_sec()
        t1 = min(float(raw.times[-1]), t0 + clip)
        chs = self.pick_channels(raw)
        seg = raw.copy().pick(chs).crop(tmin=t0, tmax=t1, include_tmax=False)
        x = seg.get_data()  # volts
        sf = float(seg.info["sfreq"])

        avg = x.mean(axis=0) * 1e6  # µV

        plt.figure(figsize=(10, 5))
        plt.specgram(avg, NFFT=int(sf * 2), Fs=sf, noverlap=int(sf * 1))
        plt.title(title)
        plt.xlabel("Time (sec)")
        plt.ylabel("Frequency (Hz)")
        plt.ylim(0, self.fmax())
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()

    # -------------------------
    # EDA Topic 1: subject-wise seizure vs non-seizure
    # -------------------------
    def eda_subject_clips_psd_spec(
        self, subject_id: str, edf_file: str, raw_clean: mne.io.BaseRaw, seizure_intervals: List[Tuple[float, float]]
    ):
        duration = float(raw_clean.times[-1])
        seizure_intervals = merge_intervals(seizure_intervals)
        if not seizure_intervals:
            return

        avoid = apply_buffer(seizure_intervals, self.seizure_buffer_sec(), duration)

        seiz_t0 = self.choose_seizure_clip_start(seizure_intervals, duration)
        non_t0 = self.choose_nonseizure_clip_start(duration, avoid)
        if seiz_t0 is None or non_t0 is None:
            return

        clip = self.clip_sec()
        base = f"{subject_id}__{edf_file}"

        self.plot_clip_with_labels(
            raw_clean,
            seiz_t0,
            self.figs_dir / f"{base}__EDA1_seizure_clip.png",
            title=f"EDA1 | {subject_id} | {edf_file} | Seizure clip {clip:.0f}s",
        )
        self.plot_clip_with_labels(
            raw_clean,
            non_t0,
            self.figs_dir / f"{base}__EDA1_nonseizure_clip.png",
            title=f"EDA1 | {subject_id} | {edf_file} | Non-seizure clip {clip:.0f}s | outside ±{self.seizure_buffer_sec():.0f}s",
        )
        self.plot_psd_compare(
            raw_clean,
            seiz_t0,
            non_t0,
            self.figs_dir / f"{base}__EDA1_psd_compare.png",
        )
        self.plot_spectrogram(
            raw_clean,
            seiz_t0,
            self.figs_dir / f"{base}__EDA1_spectrogram_seiz.png",
            title=f"EDA1 | Spectrogram | {subject_id} | Seizure",
        )
        self.plot_spectrogram(
            raw_clean,
            non_t0,
            self.figs_dir / f"{base}__EDA1_spectrogram_non.png",
            title=f"EDA1 | Spectrogram | {subject_id} | Non-seizure",
        )

    # -------------------------
    # EDA Topic 2: temporal neighborhood sanity
    # -------------------------
    def eda_temporal_neighborhood_sanity(
        self, subject_id: str, edf_file: str, raw_clean: mne.io.BaseRaw, seizure_intervals: List[Tuple[float, float]]
    ):
        duration = float(raw_clean.times[-1])
        seizure_intervals = merge_intervals(seizure_intervals)
        if not seizure_intervals:
            return

        s, e = seizure_intervals[0]
        b = self.seizure_buffer_sec()
        clip = self.clip_sec()

        pre_t0 = max(0.0, s - (b + clip))
        post_t0 = min(max(0.0, duration - clip), e + b)

        avoid = apply_buffer(seizure_intervals, b, duration)
        far_non = self.choose_nonseizure_clip_start(duration, avoid)
        if far_non is None:
            return

        base = f"{subject_id}__{edf_file}"
        self.plot_spectrogram(
            raw_clean,
            pre_t0,
            self.figs_dir / f"{base}__EDA2_spec_pre.png",
            title=f"EDA2 | {subject_id} | pre (outside buffer)",
        )
        self.plot_spectrogram(
            raw_clean,
            post_t0,
            self.figs_dir / f"{base}__EDA2_spec_post.png",
            title=f"EDA2 | {subject_id} | post (outside buffer)",
        )
        self.plot_spectrogram(
            raw_clean,
            far_non,
            self.figs_dir / f"{base}__EDA2_spec_far_non.png",
            title=f"EDA2 | {subject_id} | far non-seizure",
        )

    # -------------------------
    # EDA Topic 3: noise/artifact profile
    # -------------------------
    def eda_noise_artifact_profile(self, subject_id: str, edf_file: str, qc_channels: pd.DataFrame):
        df = qc_channels.copy()
        if df.empty:
            return

        base = f"{subject_id}__{edf_file}"

        top = df.sort_values("std_uv", ascending=False).head(min(15, len(df)))
        plt.figure(figsize=(10, 5))
        plt.bar(top["ch_name"].astype(str), top["std_uv"].astype(float))
        plt.xticks(rotation=45, ha="right")
        plt.title(f"EDA3 | {subject_id} | noisiest channels (std µV)")
        plt.tight_layout()
        plt.savefig(self.figs_dir / f"{base}__EDA3_noisy_channels.png", dpi=150)
        plt.close()

        top_ln = df.sort_values("line_noise_ratio", ascending=False).head(min(15, len(df)))
        plt.figure(figsize=(10, 5))
        plt.bar(top_ln["ch_name"].astype(str), top_ln["line_noise_ratio"].astype(float))
        plt.xticks(rotation=45, ha="right")
        plt.title(f"EDA3 | {subject_id} | line noise ratio (line band / neighbors)")
        plt.tight_layout()
        plt.savefig(self.figs_dir / f"{base}__EDA3_line_noise.png", dpi=150)
        plt.close()

    # -------------------------
    # Dataset-level EDA: seizure duration distribution
    # -------------------------
    def seizure_duration_distribution(self, evts: pd.DataFrame):
        if evts.empty:
            return

        df = evts.copy()
        df["seizure_start_sec"] = pd.to_numeric(df["seizure_start_sec"], errors="coerce")
        df["seizure_end_sec"] = pd.to_numeric(df["seizure_end_sec"], errors="coerce")
        df["seizure_dur_sec"] = df["seizure_end_sec"] - df["seizure_start_sec"]
        df = df.dropna(subset=["seizure_dur_sec"])
        df = df[df["seizure_dur_sec"] > 0]

        stats = df["seizure_dur_sec"].describe(
            percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]
        ).to_frame("value")
        stats.to_csv(self.tables_dir / "seizure_duration_stats.csv", index=True)

        plt.figure(figsize=(10, 5))
        plt.hist(df["seizure_dur_sec"].values, bins=50)
        plt.title("Seizure duration distribution (seconds)")
        plt.xlabel("Duration (sec)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(self.figs_dir / "seizure_duration_hist.png", dpi=150)
        plt.close()

    # -------------------------
    # Dataset-level EDA: recording quality summary
    # -------------------------
    def recording_quality_summary(self, preprocess_meta: pd.DataFrame, subject_file_summary: pd.DataFrame):
        if preprocess_meta.empty:
            return

        meta = preprocess_meta.copy()
        for c in ["bad_frac", "n_bads", "n_channels", "duration_sec"]:
            if c in meta.columns:
                meta[c] = pd.to_numeric(meta[c], errors="coerce")

        if not subject_file_summary.empty:
            sfs = subject_file_summary.copy()
            sfs["n_seizures_in_file"] = pd.to_numeric(sfs["n_seizures_in_file"], errors="coerce")
            meta = meta.merge(
                sfs[["subject_id", "edf_file", "n_seizures_in_file"]],
                on=["subject_id", "edf_file"],
                how="left",
            )
        else:
            meta["n_seizures_in_file"] = np.nan

        meta.to_csv(self.tables_dir / "recording_quality_summary.csv", index=False)

        if "bad_frac" in meta.columns and "n_seizures_in_file" in meta.columns:
            dfp = meta.dropna(subset=["bad_frac", "n_seizures_in_file"])
            if not dfp.empty:
                plt.figure(figsize=(7, 5))
                plt.scatter(dfp["bad_frac"].values, dfp["n_seizures_in_file"].values)
                plt.title("Bad channel fraction vs seizures per file")
                plt.xlabel("bad_channel_frac")
                plt.ylabel("n_seizures_in_file")
                plt.tight_layout()
                plt.savefig(self.figs_dir / "bad_channel_frac_vs_seizure_count.png", dpi=150)
                plt.close()

    # -------------------------
    # Dataset-level EDA: missing channel summary per file
    # -------------------------
    def missing_channel_summary(self, qc_all: pd.DataFrame):
        if qc_all.empty:
            return

        df = qc_all.copy()
        needed_cols = {"subject_id", "edf_file", "ch_name"}
        if not needed_cols.issubset(df.columns):
            return

        global_channels = set(df["ch_name"].astype(str).unique().tolist())

        rows = []
        for (sid, ef), g in df.groupby(["subject_id", "edf_file"]):
            present = set(g["ch_name"].astype(str).tolist())
            missing = sorted(list(global_channels - present))
            rows.append(
                {
                    "subject_id": sid,
                    "edf_file": ef,
                    "n_present_channels": len(present),
                    "n_missing_channels": len(missing),
                    "missing_channels": ", ".join(missing[:50]),
                }
            )

        out = pd.DataFrame(rows).sort_values("n_missing_channels", ascending=False)
        out.to_csv(self.tables_dir / "missing_channel_summary.csv", index=False)

    # -------------------------
    # Dataset-level EDA: seizure stats by age bin / sex
    # -------------------------
    def seizure_stats_by_demographics(self, subject_file_summary: pd.DataFrame):
        if subject_file_summary.empty:
            return

        df = subject_file_summary.copy()
        df["total_seizure_duration_sec"] = pd.to_numeric(df["total_seizure_duration_sec"], errors="coerce")

        demo = self._demo()
        age_bins = list(demo.get("age_bins", [0, 5, 12, 18, 150]))
        age_labels = list(demo.get("age_bin_labels", ["0-5", "6-12", "13-18", "19+"]))

        # age-bin
        if "age" in df.columns:
            df["age"] = pd.to_numeric(df["age"], errors="coerce")
            df["age_bin"] = pd.cut(df["age"], bins=age_bins, labels=age_labels, include_lowest=True)

            age_out = (
                df.groupby("age_bin")
                .agg(
                    n_files=("edf_file", "count"),
                    n_subjects=("subject_id", "nunique"),
                    total_seizure_sec=("total_seizure_duration_sec", "sum"),
                    mean_seizure_sec=("total_seizure_duration_sec", "mean"),
                )
                .reset_index()
            )
            age_out.to_csv(self.tables_dir / "seizure_stats_by_agebin.csv", index=False)

        # sex
        if "sex" in df.columns:
            sex_out = (
                df.groupby("sex")
                .agg(
                    n_files=("edf_file", "count"),
                    n_subjects=("subject_id", "nunique"),
                    total_seizure_sec=("total_seizure_duration_sec", "sum"),
                    mean_seizure_sec=("total_seizure_duration_sec", "mean"),
                )
                .reset_index()
            )
            sex_out.to_csv(self.tables_dir / "seizure_stats_by_sex.csv", index=False)

    # -------------------------
    # EDA Topic 4: buffer sensitivity PSD (0/10/30/60 etc)
    # -------------------------
    def eda_buffer_sensitivity_psd(
        self,
        subject_id: str,
        edf_file: str,
        raw_clean: mne.io.BaseRaw,
        seizure_intervals: List[Tuple[float, float]],
    ):
        duration = float(raw_clean.times[-1])
        seizure_intervals = merge_intervals(seizure_intervals)
        if not seizure_intervals:
            return

        seiz_t0 = self.choose_seizure_clip_start(seizure_intervals, duration)
        if seiz_t0 is None:
            return

        chs = self.pick_channels(raw_clean)
        sf = float(raw_clean.info["sfreq"])
        clip = self.clip_sec()
        fmax = self.fmax()

        seiz = raw_clean.copy().pick(chs).crop(
            tmin=seiz_t0, tmax=seiz_t0 + clip, include_tmax=False
        ).get_data()

        ps_s, freqs = mne.time_frequency.psd_array_welch(
            seiz, sfreq=sf, fmin=1.0, fmax=fmax, n_fft=int(sf * 2), verbose=False
        )
        ps_seiz = ps_s.mean(axis=0)

        buffers = self.buffer_grid_sec()
        ps_n_list = []
        labels = []

        for b in buffers:
            avoid = apply_buffer(seizure_intervals, buffer_sec=float(b), duration=duration)
            non_t0 = self.choose_nonseizure_clip_start(duration, avoid)
            if non_t0 is None:
                continue

            non = raw_clean.copy().pick(chs).crop(
                tmin=non_t0, tmax=non_t0 + clip, include_tmax=False
            ).get_data()

            ps_n, _ = mne.time_frequency.psd_array_welch(
                non, sfreq=sf, fmin=1.0, fmax=fmax, n_fft=int(sf * 2), verbose=False
            )
            ps_n_list.append(ps_n.mean(axis=0))
            labels.append(f"non (buffer={b}s)")

        if not ps_n_list:
            return

        base = f"{subject_id}__{edf_file}"
        plt.figure(figsize=(10, 5))
        plt.semilogy(freqs, ps_seiz, label="seizure clip")
        for arr, lab in zip(ps_n_list, labels):
            plt.semilogy(freqs, arr, label=lab)
        plt.title(f"EDA4 | Buffer sensitivity PSD | {subject_id} | {edf_file}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power (V^2/Hz)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.figs_dir / f"{base}__EDA4_buffer_sensitivity_psd.png", dpi=150)
        plt.close()

    # -------------------------
    # “Global” tables
    # -------------------------
    def age_bin_summary(self, subject_file_summary: pd.DataFrame):
        df = subject_file_summary.copy()
        if df.empty or "age" not in df.columns:
            return

        df["age"] = pd.to_numeric(df["age"], errors="coerce")

        demo = self._demo()
        age_bins = list(demo.get("age_bins", [0, 5, 12, 18, 150]))
        age_labels = list(demo.get("age_bin_labels", ["0-5", "6-12", "13-18", "19+"]))

        df["age_bin"] = pd.cut(df["age"], bins=age_bins, labels=age_labels, include_lowest=True)

        out = (
            df.groupby("age_bin")
            .agg(
                n_files=("edf_file", "count"),
                n_subjects=("subject_id", "nunique"),
                total_seizure_sec=("total_seizure_duration_sec", "sum"),
                mean_seizure_sec=("total_seizure_duration_sec", "mean"),
            )
            .reset_index()
        )
        out.to_csv(self.tables_dir / "agebin_subject_summary.csv", index=False)

    def channel_coverage_report(self, qc_all: pd.DataFrame):
        if qc_all.empty:
            return
        out = (
            qc_all.groupby("ch_name")
            .agg(
                n_files=("ch_name", "size"),
                mean_std_uv=("std_uv", "mean"),
                mean_line_noise_ratio=("line_noise_ratio", "mean"),
            )
            .reset_index()
            .sort_values("n_files", ascending=False)
        )
        out.to_csv(self.tables_dir / "channel_coverage_report.csv", index=False)
