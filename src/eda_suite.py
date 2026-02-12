# src/eda_suite.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne


def _apply_buffer(intervals: List[Tuple[float, float]], buffer_sec: float, duration: float) -> List[Tuple[float, float]]:
    out = []
    for s, e in intervals:
        out.append((max(0.0, s - buffer_sec), min(duration, e + buffer_sec)))
    out = sorted(out)
    merged = []
    for s, e in out:
        if not merged or s > merged[-1][1]:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
    return merged


def _overlaps_any(t0: float, t1: float, intervals: List[Tuple[float, float]]) -> bool:
    for s, e in intervals:
        if max(t0, s) < min(t1, e):
            return True
    return False


class SubjectSamplerNoRepeat:
    """
    Samples subjects without repeating across topics.
    We keep separate pools for seizure subjects and non-seizure subjects.
    """
    def __init__(self, seizure_subjects: List[str], non_subjects: List[str], seed: int):
        self.rng = np.random.default_rng(seed)
        self.seiz_pool = seizure_subjects[:]
        self.non_pool = non_subjects[:]
        self.used_seiz: Set[str] = set()
        self.used_non: Set[str] = set()

    def sample_group(self, pool: List[str], used: Set[str], k: int) -> List[str]:
        avail = [s for s in pool if s not in used]
        k = min(k, len(avail))
        if k <= 0:
            return []
        picked = list(self.rng.choice(avail, size=k, replace=False))
        used.update(picked)
        return picked

    def sample_for_topic(self, k_per_group: int) -> Tuple[List[str], List[str]]:
        seiz = self.sample_group(self.seiz_pool, self.used_seiz, k_per_group)
        non = self.sample_group(self.non_pool, self.used_non, k_per_group)
        return seiz, non


class EDAEngine:
    def __init__(self, cfg: Dict[str, Any], figs_dir: Path, tables_dir: Path):
        self.cfg = cfg
        self.figs_dir = Path(figs_dir)
        self.tables_dir = Path(tables_dir)
        self.figs_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

    def _eda(self) -> Dict[str, Any]:
        return self.cfg.get("eda", {})

    def _signal(self) -> Dict[str, Any]:
        return self.cfg.get("signal", {})

    def clip_sec(self) -> float:
        return float(self._eda().get("clip_sec", 30))

    def buffer_sec(self) -> float:
        return float(self._eda().get("seizure_buffer_sec", 30))

    def non_gap_sec(self) -> float:
        return float(self._eda().get("nonseizure_min_gap_sec", 60))

    def channels_for_viz(self) -> int:
        return int(self._eda().get("channels_for_viz", 16))

    def seizure_keywords(self) -> List[str]:
        return list(self._eda().get("seizure_keywords", ["seiz", "seizure", "ictal"]))

    def fmax(self) -> float:
        bp = self._signal().get("bandpass_hz", [1.0, 50.0])
        return float(bp[1])

    def pick_16_channels(self, raw: mne.io.BaseRaw) -> List[str]:
        k = self.channels_for_viz()
        return raw.ch_names[: min(k, len(raw.ch_names))]

    # -------------------------
    # Clip selection
    # -------------------------
    def choose_seizure_clip(self, intervals: List[Tuple[float, float]], duration: float) -> Optional[float]:
        if not intervals:
            return None
        s, e = intervals[0]
        mid = 0.5 * (s + e)
        t0 = mid - self.clip_sec() / 2
        return float(max(0.0, min(duration - self.clip_sec(), t0)))

    def choose_nonseizure_clip(self, duration: float, avoid: List[Tuple[float, float]], seed: int) -> Optional[float]:
        clip = self.clip_sec()
        expanded = _apply_buffer(avoid, self.non_gap_sec(), duration)
        rng = np.random.default_rng(seed)

        for _ in range(400):
            t0 = float(rng.uniform(0, max(1e-6, duration - clip)))
            if not _overlaps_any(t0, t0 + clip, expanded):
                return t0

        # fallback scan
        step = max(1.0, clip / 2)
        t0 = 0.0
        while t0 + clip <= duration:
            if not _overlaps_any(t0, t0 + clip, expanded):
                return t0
            t0 += step
        return None

    # -------------------------
    # Plot helpers
    # -------------------------
    def plot_waveform_16ch(self, raw: mne.io.BaseRaw, t0: float, out_png: Path, title: str):
        clip = self.clip_sec()
        t1 = min(float(raw.times[-1]), t0 + clip)
        chs = self.pick_16_channels(raw)
        seg = raw.copy().pick(chs).crop(tmin=t0, tmax=t1, include_tmax=False)
        x = seg.get_data() * 1e6
        t = np.linspace(0, t1 - t0, x.shape[1])

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

    def plot_psd(self, raw: mne.io.BaseRaw, t0: float, out_png: Path, title: str):
        clip = self.clip_sec()
        sf = float(raw.info["sfreq"])
        fmax = self.fmax()
        chs = self.pick_16_channels(raw)
        seg = raw.copy().pick(chs).crop(tmin=t0, tmax=t0 + clip, include_tmax=False).get_data()

        psd, freqs = mne.time_frequency.psd_array_welch(
            seg, sfreq=sf, fmin=1.0, fmax=fmax, n_fft=int(sf * 2), verbose=False
        )

        plt.figure(figsize=(10, 5))
        plt.semilogy(freqs, psd.mean(axis=0))
        plt.title(title)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power (V^2/Hz)")
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()

    def plot_spectrogram(self, raw: mne.io.BaseRaw, t0: float, out_png: Path, title: str):
        clip = self.clip_sec()
        t1 = min(float(raw.times[-1]), t0 + clip)
        sf = float(raw.info["sfreq"])
        chs = self.pick_16_channels(raw)
        seg = raw.copy().pick(chs).crop(tmin=t0, tmax=t1, include_tmax=False).get_data()
        avg_uv = (seg.mean(axis=0) * 1e6)

        plt.figure(figsize=(10, 5))
        plt.specgram(avg_uv, NFFT=int(sf * 2), Fs=sf, noverlap=int(sf * 1))
        plt.ylim(0, self.fmax())
        plt.title(title)
        plt.xlabel("Time (sec)")
        plt.ylabel("Frequency (Hz)")
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()

    # ============================================================
    # TOPIC 1: Seizure vs Non-seizure waveform (16ch) + PSD + Spec
    # ============================================================
    def topic1_seiz_vs_non(self, subject_id: str, raw: mne.io.BaseRaw, seizure_intervals: List[Tuple[float, float]], topic_tag: str):
        duration = float(raw.times[-1])
        seed = int(self._eda().get("random_seed", 42))
        base = f"{topic_tag}__sub-{subject_id}"

        if seizure_intervals:
            avoid = _apply_buffer(seizure_intervals, self.buffer_sec(), duration)
            seiz_t0 = self.choose_seizure_clip(seizure_intervals, duration)
            non_t0 = self.choose_nonseizure_clip(duration, avoid, seed=seed + 1)

            if seiz_t0 is not None:
                self.plot_waveform_16ch(raw, seiz_t0, self.figs_dir / f"{base}__seiz_wave.png",
                                        f"{topic_tag} | sub-{subject_id} | Seizure waveform (16ch)")
                self.plot_psd(raw, seiz_t0, self.figs_dir / f"{base}__seiz_psd.png",
                              f"{topic_tag} | sub-{subject_id} | Seizure PSD")
                self.plot_spectrogram(raw, seiz_t0, self.figs_dir / f"{base}__seiz_spec.png",
                                      f"{topic_tag} | sub-{subject_id} | Seizure Spectrogram")

            if non_t0 is not None:
                self.plot_waveform_16ch(raw, non_t0, self.figs_dir / f"{base}__non_wave.png",
                                        f"{topic_tag} | sub-{subject_id} | Non-seizure waveform (16ch)")
                self.plot_psd(raw, non_t0, self.figs_dir / f"{base}__non_psd.png",
                              f"{topic_tag} | sub-{subject_id} | Non-seizure PSD")
                self.plot_spectrogram(raw, non_t0, self.figs_dir / f"{base}__non_spec.png",
                                      f"{topic_tag} | sub-{subject_id} | Non-seizure Spectrogram")
        else:
            # no seizure: show 1 random non-seizure clip
            non_t0 = self.choose_nonseizure_clip(duration, [], seed=seed + 7)
            if non_t0 is not None:
                self.plot_waveform_16ch(raw, non_t0, self.figs_dir / f"{base}__non_wave.png",
                                        f"{topic_tag} | sub-{subject_id} | No events.tsv seizure -> random clip (16ch)")
                self.plot_psd(raw, non_t0, self.figs_dir / f"{base}__non_psd.png",
                              f"{topic_tag} | sub-{subject_id} | PSD (random)")
                self.plot_spectrogram(raw, non_t0, self.figs_dir / f"{base}__non_spec.png",
                                      f"{topic_tag} | sub-{subject_id} | Spectrogram (random)")

    # ============================================================
    # TOPIC 2 (CSV OUTPUT #1): Dataset recording summary CSV
    # ============================================================
    def topic2_dataset_summary_csv(self, records: List[Dict[str, Any]]):
        """
        CSV output: dataset_summary.csv
        Columns: subject_id, session, task, run, sfreq, duration_sec, n_channels, eeg_path, events_path
        """
        df = pd.DataFrame(records)
        out = self.tables_dir / "dataset_summary.csv"
        df.to_csv(out, index=False)

    # ============================================================
    # TOPIC 3 (CSV OUTPUT #2): Channel coverage + basic QC CSV
    # ============================================================
    def topic3_channel_coverage_csv(self, per_record_channels: List[Dict[str, Any]]):
        """
        CSV outputs:
          - channel_coverage.csv  (how often each channel appears)
          - per_record_channels.csv (channels list per recording)
        """
        per_df = pd.DataFrame(per_record_channels)
        per_df.to_csv(self.tables_dir / "per_record_channels.csv", index=False)

        # coverage
        rows = []
        for _, r in per_df.iterrows():
            chs = str(r.get("channels", "")).split("|") if pd.notna(r.get("channels", "")) else []
            for ch in chs:
                ch = ch.strip()
                if ch:
                    rows.append({"ch_name": ch, "recording_id": r.get("recording_id", "")})

        if rows:
            cov = pd.DataFrame(rows).groupby("ch_name").agg(n_recordings=("recording_id", "nunique")).reset_index()
            cov = cov.sort_values("n_recordings", ascending=False)
            cov.to_csv(self.tables_dir / "channel_coverage.csv", index=False)

    # ============================================================
    # TOPIC 4: Buffer sensitivity (visual) around seizure boundaries
    # ============================================================
    def topic4_buffer_sensitivity(self, subject_id: str, raw: mne.io.BaseRaw, seizure_intervals: List[Tuple[float, float]], topic_tag: str):
        """
        Shows why buffer matters by comparing spectrograms:
          - near seizure boundary (pre/post)
          - far non-seizure
        """
        if not seizure_intervals:
            return

        duration = float(raw.times[-1])
        clip = self.clip_sec()
        s, e = seizure_intervals[0]
        b = self.buffer_sec()
        seed = int(self._eda().get("random_seed", 42))

        pre_t0 = max(0.0, s - (b + clip))
        post_t0 = min(max(0.0, duration - clip), e + b)

        avoid = _apply_buffer(seizure_intervals, b, duration)
        far_non = self.choose_nonseizure_clip(duration, avoid, seed=seed + 99)
        if far_non is None:
            return

        base = f"{topic_tag}__sub-{subject_id}"
        self.plot_spectrogram(raw, pre_t0, self.figs_dir / f"{base}__pre_spec.png",
                              f"{topic_tag} | sub-{subject_id} | pre (outside buffer)")
        self.plot_spectrogram(raw, post_t0, self.figs_dir / f"{base}__post_spec.png",
                              f"{topic_tag} | sub-{subject_id} | post (outside buffer)")
        self.plot_spectrogram(raw, far_non, self.figs_dir / f"{base}__far_non_spec.png",
                              f"{topic_tag} | sub-{subject_id} | far non-seizure")
