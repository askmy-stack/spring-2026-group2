from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List

import mne
import numpy as np
import pandas as pd


NA = "NA"


def convert_to_bids(
    raw: mne.io.BaseRaw,
    bids_root: Path,
    subject_id: str,
    metadata: Dict[str, Any],
    cfg: Dict[str, Any],
    session: Optional[str] = None,
    run: Optional[str] = None,
    events: Optional[List[Dict]] = None,
) -> Path:
    bids_root = Path(bids_root)
    bids_cfg = cfg.get("bids", {})
    task = bids_cfg.get("task_name", "eeg")

    sub_dir = bids_root / f"sub-{subject_id}"
    if session:
        sub_dir = sub_dir / f"ses-{session}"
    eeg_dir = sub_dir / "eeg"
    eeg_dir.mkdir(parents=True, exist_ok=True)

    run_str = f"_run-{run}" if run else ""
    ses_str = f"_ses-{session}" if session else ""
    basename = f"sub-{subject_id}{ses_str}_task-{task}{run_str}_eeg"

    eeg_path = eeg_dir / f"{basename}.edf"
    raw.export(str(eeg_path), fmt="edf", overwrite=True)

    if bids_cfg.get("write_sidecar", True):
        _write_sidecar(eeg_dir / f"{basename}.json", raw, metadata, cfg)

    if bids_cfg.get("write_channels_tsv", True):
        _write_channels_tsv(eeg_dir / f"{basename}_channels.tsv", raw)

    if bids_cfg.get("write_events", True) and events:
        _write_events_tsv(eeg_dir / f"{basename}_events.tsv", events)

    _update_participants_tsv(bids_root, subject_id, metadata, bids_cfg)
    _write_dataset_description(bids_root, cfg)

    return eeg_path


def _write_sidecar(path: Path, raw: mne.io.BaseRaw, metadata: Dict[str, Any], cfg: Dict[str, Any]):
    bids_cfg = cfg.get("bids", {})
    fill_na = bids_cfg.get("fill_missing_with_na", True)

    def _val(key: str, fallback=NA):
        v = metadata.get(key, fallback if fill_na else None)
        return v if v is not None else NA

    sidecar = {
        "TaskName": cfg.get("bids", {}).get("task_name", "eeg"),
        "SamplingFrequency": float(raw.info["sfreq"]),
        "PowerLineFrequency": float(cfg.get("signal", {}).get("notch", 60.0)),
        "EEGChannelCount": len(raw.ch_names),
        "RecordingDuration": float(raw.times[-1]) if len(raw.times) > 0 else NA,
        "RecordingType": _val("recording_type", "continuous"),
        "Manufacturer": _val("manufacturer"),
        "ManufacturersModelName": _val("model"),
        "SoftwareVersions": _val("software_version"),
        "InstitutionName": _val("institution"),
        "SubjectArtefactDescription": NA,
        "EEGReference": cfg.get("signal", {}).get("reference", "average"),
        "EEGGround": NA,
        "EEGPlacementScheme": "10-20",
        "CapManufacturer": NA,
        "CapManufacturersModelName": NA,
        "HardwareFilters": NA,
        "SoftwareFilters": {
            "Highpass": cfg.get("signal", {}).get("bandpass", [1.0, 50.0])[0],
            "Lowpass": cfg.get("signal", {}).get("bandpass", [1.0, 50.0])[1],
        },
    }

    with open(path, "w") as f:
        json.dump(sidecar, f, indent=2)


def _write_channels_tsv(path: Path, raw: mne.io.BaseRaw):
    rows = []
    for ch in raw.ch_names:
        rows.append({
            "name": ch,
            "type": "EEG",
            "units": "uV",
            "low_cutoff": NA,
            "high_cutoff": NA,
            "description": NA,
            "sampling_frequency": float(raw.info["sfreq"]),
            "reference": NA,
            "group": NA,
        })
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)

import json
from pathlib import Path

def ensure_dataset_description(bids_root: str | Path, bids_version: str = "1.7.0") -> None:
    bids_root = Path(bids_root)
    bids_root.mkdir(parents=True, exist_ok=True)
    dd = bids_root / "dataset_description.json"
    if dd.exists():
        return

    payload = {
        "Name": "EEG Seizure Dataset",
        "BIDSVersion": bids_version,
        "DatasetType": "raw",
        "GeneratedBy": [{"Name": "EEG Seizure Detection Dataloader", "Version": "1.0"}],
    }
    dd.write_text(json.dumps(payload, indent=2))


def _write_events_tsv(path: Path, events: List[Dict]):
    rows = []
    for ev in events:
        rows.append({
            "onset": ev.get("onset", NA),
            "duration": ev.get("duration", NA),
            "trial_type": ev.get("trial_type", NA),
            "value": ev.get("value", NA),
            "sample": ev.get("sample", NA),
        })
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def _update_participants_tsv(bids_root: Path, subject_id: str, metadata: Dict[str, Any], bids_cfg: Dict):
    if not bids_cfg.get("write_participants_tsv", True):
        return
    participants_file = bids_root / "participants.tsv"
    new_row = {
        "participant_id": f"sub-{subject_id}",
        "age": metadata.get("age", NA),
        "sex": str(metadata.get("sex", NA)).upper()[:1] if metadata.get("sex") not in (NA, None, "NA") else NA,
        "hand": metadata.get("hand", NA),
        "group": metadata.get("group", NA),
    }

    if participants_file.exists():
        df = pd.read_csv(participants_file, sep="\t")
        df = df[df["participant_id"] != f"sub-{subject_id}"]
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])

    df.to_csv(participants_file, sep="\t", index=False)


def _write_dataset_description(bids_root: Path, cfg: Dict[str, Any]):
    desc_file = bids_root / "dataset_description.json"
    if desc_file.exists():
        return
    desc = {
        "Name": "EEG Dataset",
        "BIDSVersion": cfg.get("bids", {}).get("version", "1.7.0"),
        "License": NA,
        "Authors": [NA],
        "Acknowledgements": NA,
        "HowToAcknowledge": NA,
        "Funding": [NA],
        "ReferencesAndLinks": [NA],
        "DatasetDOI": NA,
    }
    with open(desc_file, "w") as f:
        json.dump(desc, f, indent=2)


def load_participants(bids_root: Path) -> pd.DataFrame:
    p = Path(bids_root) / "participants.tsv"
    if not p.exists():
        return pd.DataFrame(columns=["participant_id", "age", "sex", "hand", "group"])
    return pd.read_csv(p, sep="\t")