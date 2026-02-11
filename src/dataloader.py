import os
import yaml
import numpy as np
import pandas as pd
import mne
import scipy.io
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timezone, date
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

try:
    from mne_bids import BIDSPath, write_raw_bids

    MNE_BIDS_AVAILABLE = True
except ImportError:
    MNE_BIDS_AVAILABLE = False


@dataclass
class EEGWindowIndex:
    recording_id: str
    start_sec: float
    end_sec: float
    label: int
    weight: float
    meta: Dict[str, Any]

def _inject_metadata(raw: mne.io.BaseRaw, subject_id: str, age: int = None, sex: str = None) -> mne.io.BaseRaw:
    """Injects Age/Sex into MNE object for BIDS compliance."""
    if raw.info['subject_info'] is None:
        raw.info['subject_info'] = {}

    # 1. Subject ID
    raw.info['subject_info']['his_id'] = str(subject_id)

    # 2. Sex (1=Male, 2=Female, 0=Unknown)
    if sex:
        sex_code = 1 if str(sex).lower() in ['m', 'male'] else 2 if str(sex).lower() in ['f', 'female'] else 0
        raw.info['subject_info']['sex'] = sex_code

    # 3. Age to Birthday Calculation (Required by MNE)
    if age is not None:
        meas_date = raw.info['meas_date']
        if meas_date is None:
            meas_date = datetime.now(timezone.utc)
            raw.set_meas_date(meas_date)

        birth_year = meas_date.year - int(age)
        try:
            raw.info['subject_info']['birthday'] = date(birth_year, meas_date.month, meas_date.day)
        except ValueError:
            raw.info['subject_info']['birthday'] = date(birth_year, 1, 1)

    return raw


def _process_single_file(row_data: Dict, config: Dict) -> Dict:
    """
    Worker function: Reads, Filters, Injects Metadata, Saves FIF & BIDS.
    """
    try:
        raw_path = Path(row_data['path'])
        processed_dir = Path(config["dataset"]["results_root"]) / "processed_signals"
        processed_dir.mkdir(parents=True, exist_ok=True)
        fif_save_path = processed_dir / f"{row_data['recording_id']}_raw.fif"

        if raw_path.suffix.lower() == '.mat':
            mat = scipy.io.loadmat(raw_path)
            data = next(v for k, v in mat.items() if isinstance(v, np.ndarray) and v.ndim == 2)
            if data.shape[0] > data.shape[1]: data = data.T
            info = mne.create_info([f"Ch{i}" for i in range(len(data))], config['signal']['target_sfreq'], 'eeg')
            raw = mne.io.RawArray(data, info, verbose='ERROR')
        elif raw_path.suffix.lower() == '.csv':
            df = pd.read_csv(raw_path)
            data = df.select_dtypes(include=[np.number]).values.T
            info = mne.create_info(list(df.columns), config['signal']['target_sfreq'], 'eeg')
            raw = mne.io.RawArray(data, info, verbose='ERROR')
        else:
            raw = mne.io.read_raw_edf(raw_path, preload=True, verbose='ERROR')

        raw = _inject_metadata(raw,
                               subject_id=row_data['subject_id'],
                               age=row_data.get('age'),
                               sex=row_data.get('sex'))

        if raw.info['sfreq'] != config['signal']['target_sfreq']:
            raw.resample(config['signal']['target_sfreq'])
        lf, hf = config['signal']['bandpass']
        raw.filter(lf, hf, verbose='ERROR')
        if config['signal'].get('notch'):
            raw.notch_filter(config['signal']['notch'], verbose='ERROR')

        raw.save(fif_save_path, overwrite=True, verbose='ERROR')

        if config.get("bids", {}).get("enable", True):
            bids_root = Path(config["dataset"]["bids_root"])
            subj_id = "".join(x for x in str(row_data['subject_id']) if x.isalnum())
            try:
                bids_path = BIDSPath(subject=subj_id, task='rest', root=bids_root, datatype='eeg')
                write_raw_bids(raw, bids_path, overwrite=True, verbose=False, allow_preload=True, format='EDF')
            except Exception as e:
                pass

        return {**row_data, "path": str(fif_save_path), "status": "success"}

    except Exception as e:
        return {**row_data, "status": "failed", "error": str(e)}


class EEGDataLoader:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        self.raw_root = Path(self.cfg["dataset"]["raw_root"]).expanduser()
        self.results_root = Path(self.cfg["dataset"]["results_root"]).expanduser()
        self.results_dir = self.results_root / "dataloader"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def discover_files(self, participants_csv: str = None) -> pd.DataFrame:
        extensions = ['*.edf', '*.EDF', '*.csv', '*.mat']
        files = []
        for ext in extensions:
            files.extend(list(self.raw_root.rglob(ext)))

        rows = []
        for p in files:
            subj = p.stem.split('_')[0]
            if p.parent.name.lower().startswith("sub"): subj = p.parent.name
            rows.append({"subject_id": subj, "recording_id": p.stem, "path": str(p)})

        df = pd.DataFrame(rows).drop_duplicates(subset="recording_id")

        if participants_csv and Path(participants_csv).exists():
            parts = pd.read_csv(participants_csv)
            df = df.merge(parts, on='subject_id', how='left')

        return df

    def preprocess_parallel(self, df_files: pd.DataFrame) -> pd.DataFrame:
        n_jobs = self.cfg['system'].get('n_jobs', -1)
        if n_jobs == -1: n_jobs = os.cpu_count() or 1

        print(f"[Preprocess] Starting parallel processing with {n_jobs} cores...")
        rows = df_files.to_dict('records')
        results = []

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            for res in executor.map(_process_single_file, rows, [self.cfg] * len(rows)):
                if res['status'] != 'failed':
                    results.append(res)
                else:
                    print(f"   [Error] {res['recording_id']}: {res.get('error')}")
        return pd.DataFrame(results)

    # --- RESTORED: ROBUST STRATIFICATION (Action Item 2) ---
    def split_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Splits data preserving Age/Sex balance."""
        seed = self.cfg['split']['seed']

        # 1. Create Stratification Key
        # If age/sex missing, use generic key
        if 'sex' in df.columns and 'age' in df.columns:
            df['age_bin'] = pd.cut(df['age'], bins=[0, 18, 40, 60, 100], labels=[0, 1, 2, 3])
            df['strat_key'] = df['sex'].fillna('U').astype(str) + "_" + df['age_bin'].astype(str)

            # Rare class handling
            counts = df.drop_duplicates('subject_id')['strat_key'].value_counts()
            rare = counts[counts < 2].index
            df.loc[df['strat_key'].isin(rare), 'strat_key'] = "_OTHER"
        else:
            df['strat_key'] = "_ALL"

        # 2. Split by Subject (No Data Leakage)
        subjects = df.groupby('subject_id')['strat_key'].first().reset_index()

        try:
            train_sub, temp = train_test_split(subjects, test_size=0.4, stratify=subjects['strat_key'],
                                               random_state=seed)
            val_sub, test_sub = train_test_split(temp, test_size=0.5, stratify=temp['strat_key'], random_state=seed)
        except ValueError:
            print("[Warn] Stratification failed (too small). Using random split.")
            train_sub, temp = train_test_split(subjects, test_size=0.4, random_state=seed)
            val_sub, test_sub = train_test_split(temp, test_size=0.5, random_state=seed)

        return {
            "train": df[df['subject_id'].isin(train_sub['subject_id'])],
            "val": df[df['subject_id'].isin(val_sub['subject_id'])],
            "test": df[df['subject_id'].isin(test_sub['subject_id'])]
        }

    def build_window_index(self, df_files: pd.DataFrame, labels_path: str = None) -> None:
        print("[Index] Building Window Index...")
        win_sec = self.cfg["windowing"]["window_sec"]
        stride = self.cfg["windowing"]["stride_sec"]

        events = pd.read_csv(labels_path) if labels_path and Path(labels_path).exists() else pd.DataFrame()
        index_rows = []

        for _, row in df_files.iterrows():
            try:
                raw = mne.io.read_raw_fif(row['path'], preload=False, verbose='ERROR')
                dur = raw.times[-1]
            except:
                continue

            rec_events = events[events['file'] == row['recording_id']] if not events.empty else pd.DataFrame()

            t = 0
            while t + win_sec <= dur:
                start, end = t, t + win_sec
                label = 0
                if not rec_events.empty:
                    overlaps = np.maximum(0,
                                          np.minimum(end, rec_events['end']) - np.maximum(start, rec_events['start']))
                    if np.any(overlaps >= (win_sec * 0.5)): label = 1

                index_rows.append({
                    "recording_id": row['recording_id'],
                    "path": row['path'],
                    "start_sec": start, "end_sec": end,
                    "label": label
                })
                t += stride

        df_index = pd.DataFrame(index_rows)
        if not df_index.empty:
            class_counts = df_index['label'].value_counts()
            total = len(df_index)
            weights = {cls: total / (2 * cnt) for cls, cnt in class_counts.items() if cnt > 0}
            df_index['weight'] = df_index['label'].map(weights)

            out_path = self.results_dir / "window_index.csv"
            df_index.to_csv(out_path, index=False)
            print(f"[Success] Saved {len(df_index)} windows to {out_path}")

    def run(self):
        # 1. Discover (Optional: pass 'participants.csv' path here)
        df = self.discover_files()
        print(f"Found {len(df)} files.")

        # 2. Parallel Preprocess (With Metadata Injection)
        df_clean = self.preprocess_parallel(df)
        if df_clean.empty: return

        # 3. Stratified Split
        splits = self.split_data(df_clean)
        for k, v in splits.items(): v.to_csv(self.results_dir / f"split_{k}.csv", index=False)

        # 4. Windowing
        self.build_window_index(splits['train'])
