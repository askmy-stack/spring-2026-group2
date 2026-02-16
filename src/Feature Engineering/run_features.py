import argparse
from pathlib import Path
import re

import yaml
import pandas as pd
import mne

from feature_engineering import AdvancedFeatureExtractor


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to FE config YAML")
    return ap.parse_args()


def find_events_tsv(edf_path: Path) -> Path | None:
    """
    BIDS convention: *_eeg.edf pairs with *_events.tsv in same folder.
    """
    # Example: sub-001_ses-01_task-szMonitoring_run-01_eeg.edf
    # Events:   sub-001_ses-01_task-szMonitoring_run-01_events.tsv
    name = edf_path.name
    events_name = re.sub(r"_eeg\.edf$", "_events.tsv", name)
    cand = edf_path.parent / events_name
    return cand if cand.exists() else None


def load_seizure_intervals(events_tsv: Path, cfg: dict) -> list[tuple[float, float]]:
    """
    Return [(start, end), ...] seizure intervals in seconds.
    BIDS-score style seizures: eventType starts with "sz" (e.g., sz_foc_ia_nm).
    """
    df = pd.read_csv(events_tsv, sep="\t")

    labeling = cfg.get("labeling", {})
    onset_col = labeling.get("onset_col", "onset")
    dur_col = labeling.get("duration_col", "duration")

    # SeizeIT2 uses 'eventType'
    possible_type_cols = labeling.get(
        "type_cols",
        ["eventType", "trial_type", "event_type", "value", "description"]
    )

    type_col = None
    for c in possible_type_cols:
        if c in df.columns:
            type_col = c
            break

    if onset_col not in df.columns or dur_col not in df.columns or type_col is None:
        return []

    seizure_prefix = labeling.get("seizure_prefix", "sz").lower()

    intervals = []
    for _, r in df.iterrows():
        txt = str(r[type_col]).strip().lower()

        # seizure if eventType starts with "sz"
        if txt.startswith(seizure_prefix):
            try:
                onset = float(r[onset_col])
                dur = float(r[dur_col])
                if dur > 0:
                    intervals.append((onset, onset + dur))
            except Exception:
                pass

    return intervals



def window_label(start: float, end: float, seizure_intervals: list[tuple[float, float]]) -> int:
    """
    1 if [start,end] overlaps any seizure interval, else 0
    """
    for s, e in seizure_intervals:
        if start < e and end > s:
            return 1
    return 0


def main():
    args = parse_args()
    cfg_path = Path(args.config).resolve()

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    bids_root = Path(cfg["io"]["bids_root"]).resolve()
    output_csv = Path(cfg["io"]["output_csv"]).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    modality = cfg["bids"].get("modality", "eeg")  # "eeg" for SeizeIT2
    window_sec = float(cfg["windows"]["window_sec"])
    step_sec = float(cfg["windows"]["step_sec"])

    max_files = cfg["limits"].get("max_files", None)
    max_windows = cfg["limits"].get("max_windows", None)

    # Find EDFs
    edfs = sorted(bids_root.rglob(f"*_{modality}.edf"))
    if max_files is not None:
        edfs = edfs[: int(max_files)]

    extractor = AdvancedFeatureExtractor(sfreq=cfg["fe"]["sfreq"], cfg=cfg)

    rows = []
    total_windows = 0

    for fi, edf_path in enumerate(edfs, start=1):
        print(f"[{fi}/{len(edfs)}] Reading:", edf_path)

        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

        print("Channels:", raw.ch_names)                     
        print("Number of channels:", len(raw.ch_names))      #code to see the number of channels

        # Optional: pick only EEG channels if the file contains multiple types
        if cfg["bids"].get("pick_eeg_only", True):
            raw.pick_types(eeg=True, ecg=False, emg=False, misc=False)

        duration = float(raw.times[-1])

        # Load seizure intervals (if events exist)
        events_tsv = find_events_tsv(edf_path)
        seizure_intervals = []
        if events_tsv is not None:
            seizure_intervals = load_seizure_intervals(events_tsv, cfg)

        start = 0.0
        while start + window_sec <= duration:
            end = start + window_sec
            y = window_label(start, end, seizure_intervals)

            # Crop window
            epoch = raw.copy().crop(tmin=start, tmax=end)
            data = epoch.get_data()  # (channels, time)

            feats = extractor.extract(data)

            # metadata
            feats["label"] = int(y)            # seizure=1, non-seizure=0
            feats["recording_path"] = str(edf_path)
            feats["events_path"] = str(events_tsv) if events_tsv else ""
            feats["start_sec"] = float(start)
            feats["end_sec"] = float(end)

            rows.append(feats)

            total_windows += 1
            if max_windows is not None and total_windows >= int(max_windows):
                print("Reached max_windows, stopping.")
                break

            start += step_sec

        if max_windows is not None and total_windows >= int(max_windows):
            break

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    print("✅ Saved:", output_csv)
    print("Shape:", df.shape)
    if "label" in df.columns:
        print(df["label"].value_counts(dropna=False))


if __name__ == "__main__":
    main()


