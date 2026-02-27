from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
import argparse

import numpy as np
import pandas as pd

# __file__ = .../spring-2026-group2/src/eeg_pipeline/pipeline/run_pipeline.py
# parents[0]=pipeline, [1]=eeg_pipeline, [2]=src, [3]=spring-2026-group2 (repo root)
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"

from eeg_pipeline.core.yaml_utils import load_yaml, get
from eeg_pipeline.pipeline.preprocessor import EEGPreprocessor
from eeg_pipeline.analysis.time_domain import TimeDomainModule
from eeg_pipeline.analysis.freq_domain import FrequencyDomainAnalyzer
from eeg_pipeline.pipeline.eda_engine import EDAEngine
from eeg_pipeline.pipeline.bot_diagrams import DiagramBuilder
from eeg_pipeline.pipeline.dataset_overview import DatasetOverview
from eeg_pipeline.core.bids_derivatives import BIDSCleanedWriter


def _enabled(cfg: Dict[str, Any], path: str, default: bool = False) -> bool:
    return bool(get(cfg, path, default))


def _resolve_path(repo_root: Path, p: str | Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()


def _load_window_index(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Loads and concatenates dataloader window index CSVs.
    Expected columns:
      path, subject_id, start_sec, end_sec, label, age, sex
    """
    idx_cfg = get(cfg, "dataloader_index", {}) or {}
    paths = idx_cfg.get("csv_paths", [])
    if not paths:
        paths = [
            "results/dataloader/window_index_train.csv",
            "results/dataloader/window_index_val.csv",
            "results/dataloader/window_index_test.csv",
        ]

    cols_cfg = idx_cfg.get("columns", {}) or {}
    col_path = cols_cfg.get("path", "path")
    col_subject = cols_cfg.get("subject_id", "subject_id")
    col_start = cols_cfg.get("start_sec", "start_sec")
    col_end = cols_cfg.get("end_sec", "end_sec")
    col_label = cols_cfg.get("label", "label")
    col_age = cols_cfg.get("age", "age")
    col_sex = cols_cfg.get("sex", "sex")

    dfs = []
    for rel in paths:
        p = _resolve_path(REPO_ROOT, rel)
        if not p.exists():
            raise FileNotFoundError(f"Window index CSV not found: {p}")
        df = pd.read_csv(p)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    rename = {
        col_path: "path",
        col_subject: "subject_id",
        col_start: "start_sec",
        col_end: "end_sec",
        col_label: "label",
    }
    if col_age in df.columns:
        rename[col_age] = "age"
    if col_sex in df.columns:
        rename[col_sex] = "sex"

    df = df.rename(columns=rename)

    required = ["path", "subject_id", "start_sec", "end_sec", "label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Window index missing required columns: {missing}. Found: {list(df.columns)}")

    df["path"] = df["path"].astype(str)
    df["subject_id"] = df["subject_id"].astype(str)
    df["start_sec"] = df["start_sec"].astype(float)
    df["end_sec"] = df["end_sec"].astype(float)
    df["label"] = df["label"].astype(int)

    df = df[df["end_sec"] > df["start_sec"]].reset_index(drop=True)
    return df


def _write_participants_tsv(out_root: Path, subject_rows: pd.DataFrame) -> None:
    """
    subject_rows must have at least: subject_id
    optionally: age, sex
    """
    cols = ["subject_id"]
    if "age" in subject_rows.columns:
        cols.append("age")
    if "sex" in subject_rows.columns:
        cols.append("sex")

    part = (
        subject_rows[cols]
        .dropna(subset=["subject_id"])
        .drop_duplicates(subset=["subject_id"])
        .sort_values("subject_id")
        .copy()
    )

    part["participant_id"] = part["subject_id"].apply(
        lambda s: s if str(s).startswith("sub-") else f"sub-{s}"
    )
    part = part.drop(columns=["subject_id"])

    if "sex" in part.columns:
        part["sex"] = part["sex"].replace({"Female": "F", "Male": "M"})

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "participants.tsv").write_text(part.to_csv(sep="\t", index=False))


def _infer_subject_id_from_path(p: Path) -> str:
    """
    Try to infer subject id from BIDS-like path components:
      .../sub-XXXX/...  -> returns "sub-XXXX"
    fallback -> stem prefix
    """
    for part in p.parts:
        if part.startswith("sub-"):
            return part
    return "sub-unknown"


def _scan_bids_recordings(cfg: Dict[str, Any]) -> List[Tuple[str, Path]]:
    """
    Returns list of (subject_id, absolute_edf_path) discovered under dataset.bids_root.
    """
    bids_root = _resolve_path(REPO_ROOT, get(cfg, "dataset.bids_root", "results/bids_dataset"))
    ext = str(get(cfg, "bids.extension", ".edf"))
    ext = ext if ext.startswith(".") else f".{ext}"

    if not bids_root.exists():
        raise FileNotFoundError(f"BIDS root not found: {bids_root}")

    # scan all EDFs under root (covers eeg folder structure variations)
    edfs = sorted(bids_root.rglob(f"sub-*/**/eeg/*{ext}"))
    # Optional: only keep those in an /eeg/ directory (comment out if your structure differs)
    # edfs = [p for p in edfs if "eeg" in p.parts]

    out: List[Tuple[str, Path]] = []
    for p in edfs:
        sid = _infer_subject_id_from_path(p)
        out.append((sid, p.resolve()))

    return out


def run_from_dataloader_index(cfg: Dict[str, Any]) -> None:
    run_diagrams = _enabled(cfg, "run.diagrams", True)
    run_preprocess = _enabled(cfg, "run.preprocess", True)
    run_eda = _enabled(cfg, "run.eda", True)

    # EDA mode flags (enforced)
    rec_mode = bool(get(cfg, "eda.recording_level.enabled", False))
    win_mode = bool(get(cfg, "eda.window_level.enabled", True))

    print(f"[EDA MODE] recording_level={rec_mode} window_level={win_mode}")

    if rec_mode and win_mode:
        raise ValueError("Config error: both eda.recording_level.enabled and eda.window_level.enabled are True.")
    if (not rec_mode) and (not win_mode):
        raise ValueError("Config error: both eda.recording_level.enabled and eda.window_level.enabled are False.")

    overview_enabled = bool(get(cfg, "eda.dataset_overview.enabled", True))
    overview = DatasetOverview(cfg) if overview_enabled else None

    # preprocessing flags
    do_filter = run_preprocess and _enabled(cfg, "preprocess.filtering.enabled", True)
    do_reref = run_preprocess and _enabled(cfg, "analysis.time_domain.reref.enabled", False)
    do_ica = run_preprocess and _enabled(cfg, "analysis.time_domain.ica.enabled", False)

    # time-domain post-processing flags
    qc_enabled = _enabled(cfg, "analysis.time_domain.qc.enabled", True)
    bads_enabled = _enabled(cfg, "analysis.time_domain.bad_channels.enabled", True)
    interp_enabled = bads_enabled and _enabled(cfg, "analysis.time_domain.bad_channels.interpolate", False)

    # limits (in recording mode: treat as recordings limits; in window mode: windows limits)
    max_total = int(get(cfg, "eda.max_windows_total", 0))  # 0 = no limit
    max_per_subject = int(get(cfg, "eda.max_windows_per_subject", 0))  # 0 = no limit

    only_label = get(cfg, "eda.only_label", None)
    if only_label is not None:
        only_label = int(only_label)

    if run_diagrams:
        DiagramBuilder(cfg).save_all()

    export_enabled = _enabled(cfg, "export_cleaned.enabled", False)
    export_root = _resolve_path(REPO_ROOT, get(cfg, "export_cleaned.out_root", "results/preprocess/bids_dataset"))
    export_writer = BIDSCleanedWriter(export_root) if export_enabled else None

    # Modules
    td = TimeDomainModule(cfg)
    fd = FrequencyDomainAnalyzer(cfg)
    pre = EEGPreprocessor(cfg)
    eda = EDAEngine(cfg, td=td, fd=fd)

    # raw cache by absolute file path string
    raw_cache: Dict[str, Any] = {}

    # -------------------------
    # RECORDING-LEVEL MODE
    # -------------------------
    if rec_mode:
        if not run_eda and not export_enabled and overview is None:
            print("[INFO] Nothing to do (EDA off, export off, overview off).")
            return

        recs = _scan_bids_recordings(cfg)
        if not recs:
            print("[WARN] No EDF recordings found in dataset.bids_root.")
            return

        print(f"[INFO] BIDS scan found recordings={len(recs)} under {get(cfg, 'dataset.bids_root')}")

        subj_counts: Dict[str, int] = {}
        done = 0

        # participants for export participants.tsv
        participants_rows = []

        for (subject_id, abs_path) in recs:
            if max_per_subject > 0 and subj_counts.get(subject_id, 0) >= max_per_subject:
                continue
            if max_total > 0 and done >= max_total:
                break

            import mne
            key = str(abs_path)
            if key not in raw_cache:
                raw_full = mne.io.read_raw_edf(abs_path, preload=True, verbose=False)
                raw_cache[key] = raw_full
            raw_full = raw_cache[key]

            # overview recording metadata
            if overview is not None:
                overview.add_recording_from_path(
                    eeg_path=str(abs_path),
                    subject_id=subject_id,
                    sfreq=float(raw_full.info["sfreq"]),
                    n_channels=int(len(raw_full.ch_names)),
                    duration_sec=float(raw_full.times[-1]) if raw_full.n_times else 0.0,
                )

            raw_before = raw_full.copy()

            # preprocess full recording
            if run_preprocess:
                cleaned, steps = pre.process(
                    raw_full.copy(),
                    do_load=True,
                    do_resample=False,
                    do_reref=do_reref,
                    do_filter=do_filter,
                )
                if do_ica:
                    cleaned = td.apply_ica(cleaned)
                    steps.append("ica")
            else:
                cleaned, steps = raw_full.copy(), ["preprocess_skipped"]

            # QC + bads + interpolate on full recording
            qc = td.qc(cleaned) if qc_enabled else {}
            if bads_enabled and qc:
                td.mark_bads_from_qc(cleaned, qc)
            if interp_enabled:
                cleaned = td.interpolate_bads(cleaned)

            # Export cleaned full recording (optional)
            if export_writer is not None:
                export_writer.write_cleaned_raw(
                    cleaned,
                    subject=subject_id.replace("sub-", "").replace("-", "").replace("_", ""),
                    task="recording",
                    run="01",
                    datatype=get(cfg, "bids.datatype", "eeg"),
                    suffix=get(cfg, "bids.suffix", "eeg"),
                    overwrite=True,
                )

            # Run recording-level EDA
            rec_id = f"{subject_id}_{abs_path.stem}"
            if run_eda:
                eda.run_recording(
                    recording_id=rec_id,
                    raw_after=cleaned,
                    raw_before=raw_before if bool(get(cfg, "eda.plot_raw_vs_filtered", False)) else None,
                    seizure_intervals=None,  # add later if you build from events.tsv
                )

            participants_rows.append({"subject_id": subject_id})

            subj_counts[subject_id] = subj_counts.get(subject_id, 0) + 1
            done += 1

            print(f"[OK][REC] {rec_id} steps={steps}")

        # finalize overview
        if overview is not None:
            out = overview.finalize()
            print("[OVERVIEW] wrote dataset overview:")
            for k, v in out.items():
                print(f"  {k}: {v}")

        # participants.tsv for exported dataset
        if export_enabled:
            export_root.mkdir(parents=True, exist_ok=True)
            part_df = pd.DataFrame(participants_rows).drop_duplicates()
            _write_participants_tsv(export_root, part_df)

        print("[DONE][REC]")
        return

    # -------------------------
    # WINDOW-LEVEL MODE
    # -------------------------
    # Load window index (train/val/test)
    win_df = _load_window_index(cfg)
    print(f"[INFO] Loaded window index rows={len(win_df)}")

    # Optional filter by label
    if only_label in (0, 1):
        win_df = win_df[win_df["label"] == only_label].reset_index(drop=True)
        print(f"[INFO] Filtered by label={only_label} -> rows={len(win_df)}")

    subj_counts: Dict[str, int] = {}
    windows_done = 0

    for i, row in win_df.iterrows():
        eeg_path = str(row["path"])
        subject_id = str(row["subject_id"])
        start = float(row["start_sec"])
        end = float(row["end_sec"])
        label = int(row["label"])
        age = str(row["age"]) if "age" in row and not pd.isna(row["age"]) else None
        sex = str(row["sex"]) if "sex" in row and not pd.isna(row["sex"]) else None

        # subject budget
        if max_per_subject > 0 and subj_counts.get(subject_id, 0) >= max_per_subject:
            continue

        # global budget
        if max_total > 0 and windows_done >= max_total:
            break

        # Load raw for this file (cache)
        p = _resolve_path(REPO_ROOT, eeg_path)
        if not p.exists():
            raise FileNotFoundError(f"EEG file path does not exist: {p} (from index: {eeg_path})")

        import mne
        key = str(p.resolve())
        if key not in raw_cache:
            raw_full = mne.io.read_raw_edf(p, preload=True, verbose=False)
            raw_cache[key] = raw_full

            if overview is not None:
                overview.add_recording_from_path(
                    eeg_path=str(p),
                    subject_id=subject_id,
                    sfreq=float(raw_full.info["sfreq"]),
                    n_channels=int(len(raw_full.ch_names)),
                    duration_sec=float(raw_full.times[-1]) if raw_full.n_times else 0.0,
                )

        raw_full = raw_cache[key]

        # Crop window
        raw_win = raw_full.copy().crop(tmin=start, tmax=end, include_tmax=False)
        raw_before = raw_win.copy()

        # Preprocess window
        if run_preprocess:
            cleaned, steps = pre.process(
                raw_win,
                do_load=True,
                do_resample=False,  # already done in dataloader
                do_reref=do_reref,
                do_filter=do_filter,
            )
            if do_ica:
                cleaned = td.apply_ica(cleaned)
                steps.append("ica")
        else:
            cleaned, steps = raw_win, ["preprocess_skipped"]

        # QC + bads + interpolate
        qc = td.qc(cleaned) if qc_enabled else {}

        if bads_enabled and qc:
            td.mark_bads_from_qc(cleaned, qc)

        if interp_enabled:
            cleaned = td.interpolate_bads(cleaned)

        qc["bads"] = list(cleaned.info.get("bads", []) or [])
        qc["n_bads"] = len(qc["bads"])
        qc["noisy_channel_frac"] = qc["n_bads"] / max(1, len(cleaned.ch_names))

        # Stable window id
        rid = f"{subject_id}_{Path(eeg_path).stem}_s{int(start*1000):d}_e{int(end*1000):d}_y{label}"

        # Export cleaned window (optional)
        if export_writer is not None:
            run_id = f"{i:06d}"
            dur = max(0.0, end - start)
            desc = "seizure" if label == 1 else "nonseizure"
            cleaned.set_annotations(mne.Annotations(onset=[0.0], duration=[dur], description=[desc]))

            export_writer.write_cleaned_raw(
                cleaned,
                subject=subject_id,
                task="window",
                run=run_id,
                datatype=get(cfg, "bids.datatype", "eeg"),
                suffix=get(cfg, "bids.suffix", "eeg"),
                overwrite=True,
            )

        # Run EDA (window-level)
        if run_eda:
            eda_out = eda.run_window(
                window_id=rid,
                raw_after=cleaned,
                raw_before=raw_before if bool(get(cfg, "eda.plot_raw_vs_filtered", False)) else None,
                label=label,
            )
        else:
            eda_out = {}

        # Update overview
        if overview is not None:
            overview.add_window(
                rid=rid,
                eeg_path=eeg_path,
                subject_id=subject_id,
                label=label,
                start=start,
                end=end,
                age=age,
                sex=sex,
                qc=qc if qc_enabled else None,
                n_bads=len(cleaned.info.get("bads", []) or []),
                eda_artifacts=eda_out or {},
            )

        subj_counts[subject_id] = subj_counts.get(subject_id, 0) + 1
        windows_done += 1

        print(f"[OK][WIN] {rid} label={label} start={start:.3f} end={end:.3f} steps={steps}")

    # finalize overview once
    if overview is not None:
        out = overview.finalize()
        print("[OVERVIEW] wrote dataset overview:")
        for k, v in out.items():
            print(f"  {k}: {v}")

    # participants.tsv for exported dataset
    if export_enabled:
        export_root = _resolve_path(REPO_ROOT, get(cfg, "export_cleaned.out_root", "results/preprocess/bids_dataset"))
        # window mode: keep age/sex if present
        subj_df = win_df[["subject_id"] + ([c for c in ["age", "sex"] if c in win_df.columns])].copy()
        _write_participants_tsv(export_root, subj_df)

    print("[DONE][WIN]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="src/configs/config.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    run_from_dataloader_index(cfg)


if __name__ == "__main__":
    main()