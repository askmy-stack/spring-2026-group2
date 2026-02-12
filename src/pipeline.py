# src/pipeline.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd

from bids_utils import find_bids_recordings, read_raw_any, extract_seizure_intervals_from_events
from preprocessing import EEGPreprocessor
from eda_suite import EDAEngine, SubjectSamplerNoRepeat


class BIDSEDAPreprocessPipeline:
    """
    CSV-free pipeline:
      - scans BIDS root for recordings
      - uses events.tsv (if present) to detect seizure intervals
      - preprocessing (signal-only)
      - 4 EDA topics
      - 2 topics export CSV tables
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

        out_cfg = cfg.get("outputs", {})
        self.results_root = Path(out_cfg.get("results_root", "results"))
        self.figs_dir = Path(out_cfg.get("figs_dir", self.results_root / "figs"))
        self.tables_dir = Path(out_cfg.get("tables_dir", self.results_root / "tables"))
        self.figs_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

        self.prep = EEGPreprocessor(cfg)
        self.eda = EDAEngine(cfg, self.figs_dir, self.tables_dir)

        bids_root = cfg.get("paths", {}).get("bids_root", "")
        self.bids_root = Path(bids_root)

    def run(self):
        eda_cfg = self.cfg.get("eda", {})
        seed = int(eda_cfg.get("random_seed", 42))
        k = int(eda_cfg.get("subjects_per_group_per_topic", 3))
        n_topics = int(eda_cfg.get("n_topics", 4))

        seizure_keywords = list(eda_cfg.get("seizure_keywords", ["seiz", "seizure", "ictal"]))

        # 1) scan BIDS recordings
        recs = find_bids_recordings(self.bids_root)
        if not recs:
            raise FileNotFoundError(f"No EEG recordings found under BIDS root: {self.bids_root}")

        # 2) build dataset summary rows (CSV output #1)
        dataset_rows: List[Dict[str, Any]] = []
        per_record_channels: List[Dict[str, Any]] = []

        # subject-level seizure presence map
        subj_has_seiz: Dict[str, bool] = {}

        # We'll do a cheap first pass: determine seizure presence per subject
        for r in recs:
            sid = str(r["subject_id"])
            intervals = extract_seizure_intervals_from_events(r.get("events_path"), seizure_keywords)
            if sid not in subj_has_seiz:
                subj_has_seiz[sid] = False
            if intervals:
                subj_has_seiz[sid] = True

        seizure_subjects = sorted([s for s, v in subj_has_seiz.items() if v])
        non_subjects = sorted([s for s, v in subj_has_seiz.items() if not v])

        sampler = SubjectSamplerNoRepeat(seizure_subjects, non_subjects, seed=seed)

        # helper: pick ONE recording per subject for plotting (first one we see)
        first_record_by_subject: Dict[str, Dict[str, Any]] = {}
        for r in recs:
            sid = str(r["subject_id"])
            if sid not in first_record_by_subject:
                first_record_by_subject[sid] = r

        # 3) dataset summary and channel lists
        for r in recs:
            try:
                raw = read_raw_any(r["eeg_path"])
                raw_p, meta = self.prep.preprocess(raw)
                dataset_rows.append(
                    dict(
                        subject_id=r.get("subject_id"),
                        session=r.get("session"),
                        task=r.get("task"),
                        run=r.get("run"),
                        sfreq=meta["sfreq"],
                        duration_sec=meta["duration_sec"],
                        n_channels=meta["n_channels"],
                        eeg_path=r.get("eeg_path"),
                        events_path=r.get("events_path"),
                    )
                )
                per_record_channels.append(
                    dict(
                        recording_id=f"sub-{r.get('subject_id')}__{Path(r['eeg_path']).name}",
                        subject_id=r.get("subject_id"),
                        channels="|".join(raw_p.ch_names),
                        n_channels=int(len(raw_p.ch_names)),
                    )
                )
            except Exception as ex:
                dataset_rows.append(
                    dict(
                        subject_id=r.get("subject_id"),
                        session=r.get("session"),
                        task=r.get("task"),
                        run=r.get("run"),
                        sfreq=None,
                        duration_sec=None,
                        n_channels=None,
                        eeg_path=r.get("eeg_path"),
                        events_path=r.get("events_path"),
                        error=str(ex),
                    )
                )

        # CSV output #1 + #2
        self.eda.topic2_dataset_summary_csv(dataset_rows)
        self.eda.topic3_channel_coverage_csv(per_record_channels)

        # 4) EDA Topics (plots) — 4 topics, each topic: 3 seizure + 3 non-seizure, no repeats
        subjects_used_log = []

        for topic_idx in range(1, n_topics + 1):
            tag = f"EDA_T{topic_idx}"
            seiz_sids, non_sids = sampler.sample_for_topic(k)

            subjects_used_log.append(
                dict(topic=tag, seizure_subjects=",".join(seiz_sids), nonseizure_subjects=",".join(non_sids))
            )

            # Topic definitions:
            # T1/T2/T3: seizure vs non waveform/psd/spec
            # T4: buffer sensitivity around seizure
            for sid in seiz_sids:
                rec = first_record_by_subject.get(sid)
                if not rec:
                    continue
                raw = read_raw_any(rec["eeg_path"])
                raw_p, _ = self.prep.preprocess(raw)
                intervals = extract_seizure_intervals_from_events(rec.get("events_path"), seizure_keywords)

                if topic_idx in [1, 2, 3]:
                    self.eda.topic1_seiz_vs_non(sid, raw_p, intervals, topic_tag=tag)
                elif topic_idx == 4:
                    self.eda.topic4_buffer_sensitivity(sid, raw_p, intervals, topic_tag=tag)

            for sid in non_sids:
                rec = first_record_by_subject.get(sid)
                if not rec:
                    continue
                raw = read_raw_any(rec["eeg_path"])
                raw_p, _ = self.prep.preprocess(raw)
                intervals = []  # non-seizure group
                self.eda.topic1_seiz_vs_non(sid, raw_p, intervals, topic_tag=tag)

        pd.DataFrame(subjects_used_log).to_csv(self.tables_dir / "subjects_used_for_plots.csv", index=False)

        print("DONE ✅ (CSV-free, BIDS-based)")
        print("Figures:", self.figs_dir)
        print("Tables:", self.tables_dir)
