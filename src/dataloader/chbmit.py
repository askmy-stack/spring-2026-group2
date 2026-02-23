"""
CHB-MIT → BIDS Loader
Download CHB-MIT from PhysioNet, write a BIDS layout, and produce train/val/test CSVs.

Usage:
    loader = CHBMITBIDSLoader("bids_chbmit")
    loader.run(subjects=["chb01", "chb02"])

    train_df = loader.splits["train"]
"""

from __future__ import annotations
import json, re, shutil, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

BASE = "https://physionet.org/files/chbmit/1.0.0"

# ── tiny helpers ────────────────────────────────────────────────────────────

def _fetch_tsv(url: str, cache: Path) -> pd.DataFrame:
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, cache)
    return pd.read_csv(cache, sep="\t")

def _fetch_list(url: str, cache: Path) -> List[str]:
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, cache)
    return [l.strip() for l in cache.read_text().splitlines() if l.strip()]

def _download(url: str, dest: Path, retries: int = 3) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for i in range(retries):
        try:
            urllib.request.urlretrieve(url, dest)
            return True
        except urllib.error.URLError:
            if i < retries - 1:
                time.sleep(2 ** i)
    return False

def _write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))

def _parse_summary(path: Path) -> Dict[str, List[Tuple[int, int]]]:
    """Return {edf_filename: [(start_sec, end_sec)]} from a subject summary.txt."""
    if not path.exists():
        return {}
    text = path.read_text(errors="ignore")
    out: Dict[str, List[Tuple[int, int]]] = {}
    for block in re.split(r"(?=File Name\s*[:\-])", text, flags=re.I):
        fm = re.search(r"File Name\s*[:\-]\s*(\S+\.edf)", block, re.I)
        if not fm:
            continue
        starts = [int(m.group(1)) for m in re.finditer(r"Seizure.*?Start.*?:\s*(\d+)\s*sec", block, re.I)]
        ends   = [int(m.group(1)) for m in re.finditer(r"Seizure.*?End.*?:\s*(\d+)\s*sec",   block, re.I)]
        out[fm.group(1)] = list(zip(starts, ends))
    return out

# ── main class ──────────────────────────────────────────────────────────────

class CHBMITBIDSLoader:
    def __init__(self, bids_root: str = "bids_chbmit", raw_cache: Optional[str] = None):
        self.bids  = Path(bids_root)
        self.cache = Path(raw_cache) if raw_cache else self.bids / ".raw_cache"
        self.splits: Dict[str, pd.DataFrame] = {}

    def run(
        self,
        subjects: Optional[List[str]] = None,
        val_size: float = 0.15,
        test_size: float = 0.15,
        seed: int = 42,
        force: bool = False,
        workers: int = 1
    ) -> "CHBMITBIDSLoader":
        # 1. Metadata
        meta_df = _fetch_tsv(f"{BASE}/SUBJECT-INFO", self.cache / "SUBJECT-INFO.txt") \
                    .rename(columns={"Case": "subject_id", "Age (years)": "age", "Gender": "sex"})

        def _group(paths: List[str]) -> Dict[str, List[str]]:
            d: Dict[str, List[str]] = {}
            for p in paths:
                p = p.strip()
                if p.endswith(".edf"):
                    d.setdefault(p.split("/")[0], []).append(p)
            return d

        edf_by_subj      = _group(_fetch_list(f"{BASE}/RECORDS",               self.cache / "RECORDS.txt"))
        seizure_by_subj  = _group(_fetch_list(f"{BASE}/RECORDS-WITH-SEIZURES", self.cache / "RECORDS-WITH-SEIZURES.txt"))

        all_subjects = meta_df["subject_id"].tolist()
        subjects = [s for s in (subjects or all_subjects) if s in edf_by_subj]

        # 2. Download — summaries first (fast), then all EDFs in parallel across subjects
        for subj in subjects:
            (self.cache / subj).mkdir(parents=True, exist_ok=True)
            _download(f"{BASE}/{subj}/{subj}-summary.txt", self.cache / subj / f"{subj}-summary.txt")

        def _dl_edf(rel: str):
            subj = rel.split("/")[0]
            dest = self.cache / subj / rel.split("/", 1)[1]
            if not dest.exists() or force:
                print(f"  ↓ {rel}")
                _download(f"{BASE}/{rel}", dest)

        all_edfs = [rel for subj in subjects for rel in edf_by_subj[subj]]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_dl_edf, rel): rel for rel in all_edfs}
            with tqdm(total=len(all_edfs), desc="Downloading EDFs", unit="file") as pbar:
                for fut in as_completed(futures):
                    if exc := fut.exception():
                        tqdm.write(f"  [WARN] {futures[fut]}: {exc}")
                    pbar.update(1)

        # 3. BIDS
        self.bids.mkdir(parents=True, exist_ok=True)
        _write_json(self.bids / "dataset_description.json", {
            "Name": "CHB-MIT Scalp EEG Database", "BIDSVersion": "1.8.0",
            "License": "ODC-By 1.0", "DatasetDOI": "https://doi.org/10.13026/C2K01R",
        })
        parts = meta_df[meta_df.subject_id.isin(subjects)].copy()
        parts.insert(0, "participant_id", "sub-" + parts.subject_id)
        parts.drop(columns="subject_id").to_csv(self.bids / "participants.tsv", sep="\t", index=False)

        records = []
        for subj in tqdm(subjects, desc="Writing BIDS", unit="subject"):
            subj_cache  = self.cache / subj
            intervals   = _parse_summary(subj_cache / f"{subj}-summary.txt")
            seizure_set = {p.split("/", 1)[1] for p in seizure_by_subj.get(subj, [])}
            subj_info   = meta_df[meta_df.subject_id == subj].iloc[0]
            session_rows = []

            for idx, rel in enumerate(tqdm(sorted(edf_by_subj[subj]), desc=f"  {subj}", unit="ses", leave=False), 1):
                edf_name = rel.split("/", 1)[1]
                ses      = f"ses-{idx:03d}"
                eeg_dir  = self.bids / f"sub-{subj}" / ses / "eeg"
                eeg_dir.mkdir(parents=True, exist_ok=True)
                prefix   = eeg_dir / f"sub-{subj}_{ses}_task-seizure_eeg"

                # Copy EDF
                src = subj_cache / edf_name
                if src.exists() and not prefix.with_suffix(".edf").exists():
                    shutil.copy2(src, prefix.with_suffix(".edf"))

                # Events TSV
                evts = intervals.get(edf_name, [])
                evts_path = prefix.parent / f"sub-{subj}_{ses}_task-seizure_events.tsv"
                if not evts_path.exists():
                    pd.DataFrame(
                        [{"onset": s, "duration": e - s, "trial_type": "seizure", "value": 1}
                         for s, e in evts],
                        columns=["onset", "duration", "trial_type", "value"],
                    ).to_csv(evts_path, sep="\t", index=False)

                # Sidecar JSON
                _write_json(prefix.with_suffix(".json"),
                            {"TaskName": "seizure_monitoring", "SamplingFrequency": 256,
                             "EEGReference": "linked mastoids", "PowerLineFrequency": 60})

                session_rows.append({"session_id": ses, "edf_original": edf_name,
                                     "has_seizure": edf_name in seizure_set, "n_seizures": len(evts)})
                records.append({
                    "participant_id": f"sub-{subj}", "session_id": ses,
                    "age": subj_info.age, "sex": subj_info.sex,
                    "edf_original": edf_name,
                    "bids_edf_path": str(prefix.with_suffix(".edf").relative_to(self.bids)),
                    "events_tsv_path": str(evts_path.relative_to(self.bids)),
                    "has_seizure": int(edf_name in seizure_set),
                    "n_seizures": len(evts),
                    "seizure_intervals_sec": str(evts),
                })

            pd.DataFrame(session_rows).to_csv(
                self.bids / f"sub-{subj}" / f"sub-{subj}_sessions.tsv", sep="\t", index=False)

        # 4. Splits
        df = pd.DataFrame(records)
        strat = df["has_seizure"] if df["has_seizure"].value_counts().min() >= 2 else None
        train, val_test = train_test_split(df, test_size=val_size + test_size, random_state=seed, stratify=strat)
        val, test = train_test_split(val_test, test_size=test_size / (val_size + test_size),
                                     random_state=seed,
                                     stratify=val_test["has_seizure"] if strat is not None else None)

        splits_dir = self.bids / "splits"
        splits_dir.mkdir(exist_ok=True)
        for name, sdf in [("train", train), ("val", val), ("test", test)]:
            sdf = sdf.copy(); sdf["split"] = name
            sdf.to_csv(splits_dir / f"{name}.csv", index=False)
            self.splits[name] = sdf
            print(f"  {name}: {len(sdf)} recordings")

        pd.concat(self.splits.values()).to_csv(splits_dir / "all_splits.csv", index=False)
        print("✓ Done.")
        return self


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--bids-root",  default="bids_chbmit")
    p.add_argument("--subjects",   nargs="*")
    p.add_argument("--val-size",   type=float, default=0.15)
    p.add_argument("--test-size",  type=float, default=0.15)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--force",      action="store_true")
    args = p.parse_args()

    CHBMITBIDSLoader(args.bids_root).run(
        subjects=args.subjects, val_size=args.val_size,
        test_size=args.test_size, seed=args.seed, force=args.force,
    )