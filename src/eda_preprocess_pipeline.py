# src/eda_preprocess_pipeline.py
from __future__ import annotations

from pathlib import Path
import os
import pandas as pd
import mne

from io_utils import load_tables
from preprocessing import EEGPreprocessor
from eda_suite import EDAEngine, SubjectSampler


class EDAPreprocessPipeline:
    """
    Generic pipeline (YAML-only dict config):
      - loads teammate CSVs from results_root (recordings.csv, subjects.csv, seizure_events.csv)
      - preprocesses EDF
      - runs multiple EDA ideas
      - plots only N random subjects per idea (no repeats across ideas)
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

        results_root = cfg.get("paths", {}).get("results_root", "results")
        self.results_root = Path(results_root)

        self.figs_dir = self.results_root / "figs"
        self.tables_dir = self.results_root / "tables"
        self.cleaned_dir = self.results_root / "cleaned"

        self.figs_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.cleaned_dir.mkdir(parents=True, exist_ok=True)

        self.prep = EEGPreprocessor(cfg, self.cleaned_dir)
        self.eda = EDAEngine(cfg, self.figs_dir, self.tables_dir)

    def run(self):
        recs, subs, evts = load_tables(self.results_root)

        # ----------------------------
        # Dataset-level EDA (global)
        # ----------------------------
        self.eda.seizure_duration_distribution(evts)

        # ----------------------------
        # Subject sampling (no repeats across EDA topics)
        # ----------------------------
        eda_cfg = self.cfg.get("eda", {})
        seed = int(eda_cfg.get("random_seed", 42))
        k = int(eda_cfg.get("subjects_per_idea", 3))

        seizure_subjects = sorted(evts["subject_id"].astype(str).unique().tolist())
        sampler = SubjectSampler(seizure_subjects, seed=seed)

        idea1_subjects = sampler.sample(k)  # clips + PSD + spectrogram
        idea2_subjects = sampler.sample(k)  # temporal neighborhood sanity
        idea3_subjects = sampler.sample(k)  # noise profile
        idea4_subjects = sampler.sample(k)  # buffer sensitivity PSD

        # ----------------------------
        # Loop recordings -> preprocess -> per-subject EDA plots
        # ----------------------------
        qc_rows = []
        meta_rows = []
        subj_file_rows = []

        for _, r in recs.iterrows():
            subject_id = str(r["subject_id"])
            edf_path = str(r["edf_path"])

            if not os.path.exists(edf_path):
                print(f"[WARN] EDF not found: {edf_path}")
                continue

            edf_file = Path(edf_path).name

            # seizure intervals for this subject+file
            e = evts[
                (evts["subject_id"].astype(str) == subject_id)
                & (evts["edf_file"].astype(str) == edf_file)
            ].copy()

            seizure_intervals = [
                (float(a), float(b))
                for a, b in zip(e["seizure_start_sec"], e["seizure_end_sec"])
            ]

            # preprocess
            try:
                raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
                raw_clean, qc, meta = self.prep.preprocess(raw)
                self.prep.cache_cleaned(raw_clean, subject_id, edf_file)
            except Exception as ex:
                print(f"[WARN] Failed to read/preprocess {edf_path}: {ex}")
                continue

            # collect channel QC
            qc = qc.copy()
            qc["subject_id"] = subject_id
            qc["edf_file"] = edf_file
            qc_rows.append(qc)

            # collect preprocess meta
            meta = dict(meta)
            meta.update(
                {
                    "subject_id": subject_id,
                    "edf_file": edf_file,
                    "duration_sec": float(raw_clean.times[-1]),
                }
            )
            meta_rows.append(meta)

            # Save subject-file summary if seizures exist
            if len(seizure_intervals) > 0:
                subj_file_rows.append(
                    {
                        "subject_id": subject_id,
                        "edf_file": edf_file,
                        "sex": r.get("sex", None),
                        "age": r.get("age", None),
                        "n_seizures_in_file": int(len(seizure_intervals)),
                        "total_seizure_duration_sec": float(
                            sum(b - a for a, b in seizure_intervals)
                        ),
                    }
                )

            # ----------------------------
            # PLOTTING (K subjects per idea; no repeats across ideas)
            # ----------------------------
            if subject_id in idea1_subjects and len(seizure_intervals) > 0:
                self.eda.eda_subject_clips_psd_spec(
                    subject_id, edf_file, raw_clean, seizure_intervals
                )

            if subject_id in idea2_subjects and len(seizure_intervals) > 0:
                self.eda.eda_temporal_neighborhood_sanity(
                    subject_id, edf_file, raw_clean, seizure_intervals
                )

            if subject_id in idea3_subjects:
                # plot QC profile even if no seizure
                self.eda.eda_noise_artifact_profile(subject_id, edf_file, qc)

            if subject_id in idea4_subjects and len(seizure_intervals) > 0:
                self.eda.eda_buffer_sensitivity_psd(
                    subject_id, edf_file, raw_clean, seizure_intervals
                )

        # ----------------------------
        # Save tables + dataset-level summaries
        # ----------------------------
        qc_all = pd.concat(qc_rows, ignore_index=True) if qc_rows else pd.DataFrame()
        preprocess_meta_df = pd.DataFrame(meta_rows)
        subj_file_summary = pd.DataFrame(subj_file_rows)

        preprocess_meta_df.to_csv(self.tables_dir / "preprocess_meta.csv", index=False)
        qc_all.to_csv(self.tables_dir / "qc_channels.csv", index=False)
        subj_file_summary.to_csv(self.tables_dir / "subject_file_summary.csv", index=False)

        # dataset-level EDA tables
        self.eda.recording_quality_summary(preprocess_meta_df, subj_file_summary)
        self.eda.missing_channel_summary(qc_all)
        self.eda.seizure_stats_by_demographics(subj_file_summary)

        # optional global tables (kept if you have these methods)
        self.eda.age_bin_summary(subj_file_summary)
        self.eda.channel_coverage_report(qc_all)

        # Log which subjects were used per idea
        pd.DataFrame(
            {
                "EDA_idea": [
                    "EDA1_clips_psd_spec",
                    "EDA2_temporal_neighborhood",
                    "EDA3_noise_profile",
                    "EDA4_buffer_sensitivity_psd",
                ],
                "subjects_used": [
                    ", ".join(idea1_subjects),
                    ", ".join(idea2_subjects),
                    ", ".join(idea3_subjects),
                    ", ".join(idea4_subjects),
                ],
            }
        ).to_csv(self.tables_dir / "subjects_used_for_plots.csv", index=False)

        print("DONE ✅")
        print("Plots:", self.figs_dir)
        print("Tables:", self.tables_dir)
        print("Subjects used per idea saved to:", self.tables_dir / "subjects_used_for_plots.csv")
