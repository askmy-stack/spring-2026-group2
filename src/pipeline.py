import os
import yaml
import numpy as np
import pandas as pd
import mne
import scipy.io
import requests
import re
import torch
import warnings
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from collections import OrderedDict

warnings.filterwarnings('ignore')

try:
    from mne_bids import BIDSPath, write_raw_bids

    MNE_BIDS_AVAILABLE = True
except ImportError:
    MNE_BIDS_AVAILABLE = False

class LRUCache:
    """Keeps memory safe by limiting open files."""

    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache: return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


def _process_single_file(row_data: Dict, config: Dict) -> Dict:
    """Worker: Reads, Filters, Saves FIF, and Converts to BIDS."""
    try:
        raw_path = Path(row_data['path'])
        results_root = Path(config["dataset"]["results_root"])

        # Decide output folder based on mode
        subdir = "processed_signals" if row_data.get('is_batch', True) else "external_downloads/processed"
        processed_dir = results_root / subdir
        processed_dir.mkdir(parents=True, exist_ok=True)

        fif_save_path = processed_dir / f"{row_data['recording_id']}_raw.fif"

        if fif_save_path.exists():
            return {**row_data, "path": str(fif_save_path), "status": "exists"}

        # 1. READ
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
            # EDF
            raw = mne.io.read_raw_edf(raw_path, preload=True, verbose='ERROR')

        # 2. FILTER & RESAMPLE
        if raw.info['sfreq'] != config['signal']['target_sfreq']:
            raw.resample(config['signal']['target_sfreq'])

        lf, hf = config['signal']['bandpass']
        raw.filter(lf, hf, verbose='ERROR')
        if config['signal'].get('notch'):
            raw.notch_filter(config['signal']['notch'], verbose='ERROR')

        # 3. SAVE FIF (Fast Loading)
        raw.save(fif_save_path, overwrite=True, verbose='ERROR')

        # 4. BIDS CONVERSION (Action Item Requirement)
        if config.get("dataset", {}).get("bids_root"):
            try:
                bids_root = Path(config["dataset"]["bids_root"])
                # Sanitize subject ID
                subj_clean = "".join(x for x in str(row_data['subject_id']) if x.isalnum()) or "sub001"
                bids_path = BIDSPath(subject=subj_clean, task='rest', root=bids_root, datatype='eeg')
                write_raw_bids(raw, bids_path, overwrite=True, verbose=False, allow_preload=True, format='EDF')
            except Exception:
                pass

        return {**row_data, "path": str(fif_save_path), "status": "success"}

    except Exception as e:
        return {**row_data, "status": "failed", "error": str(e)}


class SeizurePipeline:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path).resolve()
        with open(self.config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)

        # Smart Path Anchoring
        self.script_dir = Path(__file__).parent.resolve()
        raw_root_cfg = self.cfg["dataset"]["raw_root"]
        self.raw_root = (self.script_dir / raw_root_cfg).resolve()

        res_root_cfg = self.cfg["dataset"]["results_root"]
        self.results_root = (self.script_dir / res_root_cfg).resolve()
        self.results_dir = self.results_root / "dataloader"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.cfg["dataset"]["results_root"] = str(self.results_root)

    # --- ACTION ITEM: SELF-HEALING ---
    def repair_edf_headers(self):
        print(f"🔧 [Self-Heal] Scanning headers in {self.raw_root}...")
        files = list(self.raw_root.rglob("*.edf")) + list(self.raw_root.rglob("*.EDF"))
        count = 0
        for p in files:
            try:
                with open(p, 'r+b') as f:
                    f.seek(168)
                    if f.read(8) != b'01.01.00':
                        f.seek(168);
                        f.write(b'01.01.00')
                        count += 1
            except:
                pass
        if count > 0: print(f" Repaired {count} files.")

    def fetch_missing_summaries(self):
        print(f" [Fetch] Checking for clinical labels...")
        subjects = set(p.name.split('_')[0] for p in self.raw_root.rglob("*.edf") if "chb" in p.name)
        base_url = "https://physionet.org/files/chbmit/1.0.0/"
        dl_count = 0
        for subj in subjects:
            f_name = f"{subj}-summary.txt"
            subj_files = list(self.raw_root.rglob(f"{subj}_*.edf"))
            if not subj_files: continue
            save_path = subj_files[0].parent / f_name
            if not save_path.exists():
                try:
                    r = requests.get(f"{base_url}{subj}/{f_name}", timeout=10)
                    if r.status_code == 200:
                        with open(save_path, 'wb') as f: f.write(r.content)
                        dl_count += 1
                except:
                    pass
        if dl_count > 0: print(f" Downloaded {dl_count} summary files.")

    # --- METADATA EXTRACTION (AGE/SEX) ---
    def parse_demographics(self):
        print("🕵️‍♂️ [Metadata] Extracting Age & Sex from summaries...")
        demographics = []
        for s_path in self.raw_root.rglob("*-summary.txt"):
            subj_id = s_path.name.split('-')[0]
            age, sex = None, None
            try:
                with open(s_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().lower()

                # Regex parsing
                age_match = re.search(r'age[:\s]+(\d+)', content)
                if age_match: age = int(age_match.group(1))

                if 'female' in content or ' woman ' in content:
                    sex = 'F'
                elif 'male' in content or ' man ' in content:
                    sex = 'M'

                if age or sex: demographics.append({"subject_id": subj_id, "age": age, "sex": sex})
            except:
                pass

        if demographics:
            out = self.raw_root / "participants.csv"
            pd.DataFrame(demographics).drop_duplicates(subset='subject_id').to_csv(out, index=False)
            print(f" Generated demographics for {len(demographics)} subjects.")

    def parse_labels(self):
        print(" [Labels] Parsing clinical summaries...")
        labels = []
        for s_path in self.raw_root.rglob("*-summary.txt"):
            current_file = None
            try:
                with open(s_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                for i, line in enumerate(lines):
                    line = line.strip()
                    if line.startswith("File Name:"):
                        current_file = line.split(": ")[1].replace('.edf', '')
                    if "Seizure" in line and "Start Time" in line:
                        start = int(re.search(r'\d+', line).group())
                        end = int(re.search(r'\d+', lines[i + 1]).group())
                        labels.append({"file": current_file, "start": start, "end": end, "seizure_type": "Seizure"})
            except:
                pass

        if labels:
            out = self.results_dir / "generated_labels.csv"
            pd.DataFrame(labels).to_csv(out, index=False)
            return str(out)
        return None

    # --- LEVEL 3: URL DOWNLOAD ---
    def download_external_data(self, url: str) -> Path:
        print(f" [Level 3] Downloading external data from: {url}")
        try:
            filename = url.split("/")[-1]
            if not filename.endswith(('.edf', '.EDF', '.fif')):
                filename = "downloaded_subject.edf"

            save_dir = self.raw_root / "external_downloads"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / filename

            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f" Saved to: {save_path}")
                # Repair header immediately after download
                self._repair_single_header(save_path)
                return save_path
            else:
                print(f" HTTP Error: {response.status_code}")
                return None
        except Exception as e:
            print(f" Download failed: {e}")
            return None

    def _repair_single_header(self, p: Path):
        try:
            with open(p, 'r+b') as f:
                f.seek(168)
                if f.read(8) != b'01.01.00':
                    f.seek(168);
                    f.write(b'01.01.00')
        except:
            pass

    # --- CORE PIPELINE ---
    def discover_files(self, custom_path: str = None) -> pd.DataFrame:
        search_path = Path(custom_path) if custom_path else self.raw_root
        print(f"[Discover] Scanning {search_path}...")

        files = []
        for ext in ['*.edf', '*.EDF', '*.fif']:
            files.extend(list(search_path.rglob(ext)))

        rows = []
        for p in files:
            if p.stem.startswith("chb"):
                subj = p.stem.split('_')[0]
            elif p.stem.startswith("PN"):
                subj = p.stem.split('-')[0]
            else:
                subj = "single_subject"

            rows.append({
                "subject_id": subj,
                "recording_id": p.stem,
                "path": str(p),
                "is_batch": custom_path is None
            })
        df = pd.DataFrame(rows).drop_duplicates(subset="recording_id")

        # Merge Participants for Age/Sex Report
        parts_csv = self.raw_root / "participants.csv"
        if parts_csv.exists():
            parts = pd.read_csv(parts_csv)
            df = df.merge(parts, on='subject_id', how='left')

        return df

    def preprocess(self, df_files: pd.DataFrame) -> pd.DataFrame:
        n_jobs = self.cfg['system'].get('n_jobs', -1)
        if n_jobs == -1: n_jobs = os.cpu_count() or 1
        print(f"[Preprocess] Processing {len(df_files)} files...")

        results = []
        rows = df_files.to_dict('records')
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            for res in executor.map(_process_single_file, rows, [self.cfg] * len(rows)):
                if res['status'] in ['success', 'exists']:
                    results.append(res)
                else:
                    print(f"   [Error] {res.get('error')}")
        return pd.DataFrame(results)

    def split_data(self, df: pd.DataFrame):
        seed = self.cfg['split']['seed']
        subs = df['subject_id'].unique()

        # Safe Split for Small Datasets
        if len(subs) < 3:
            return {"train": df, "val": df, "test": df}

        train, temp = train_test_split(subs, test_size=0.4, random_state=seed)
        val, test = train_test_split(temp, test_size=0.5, random_state=seed)
        return {
            "train": df[df['subject_id'].isin(train)],
            "val": df[df['subject_id'].isin(val)],
            "test": df[df['subject_id'].isin(test)]
        }

    def build_index(self, df: pd.DataFrame, labels_path: str = None, mode="train"):
        print(f"[Index] Building {mode} index...")
        win_sec = self.cfg["windowing"]["window_sec"]
        stride_norm = self.cfg["windowing"]["stride_normal_sec"]
        stride_sz = self.cfg["windowing"]["stride_seizure_sec"]
        reject = self.cfg["signal"].get("reject_threshold")

        events = pd.read_csv(labels_path) if labels_path and Path(labels_path).exists() else pd.DataFrame()
        rows = []

        for _, row in df.iterrows():
            try:
                raw = mne.io.read_raw_fif(row['path'], preload=True, verbose='ERROR')
                dur = raw.n_times / raw.info['sfreq']
                data = raw.get_data()
            except:
                continue

            rec_events = events[events['file'] == row['recording_id']] if not events.empty else pd.DataFrame()
            t = 0
            while t + win_sec <= dur:
                s, e = t, t + win_sec
                lbl = 0
                is_sz = False

                if not rec_events.empty:
                    ov = np.maximum(0, np.minimum(e, rec_events['end']) - np.maximum(s, rec_events['start']))
                    if np.any(ov >= (win_sec * 0.5)): is_sz = True; lbl = 1

                if reject and np.ptp(data[:, int(s * raw.info['sfreq']):int(e * raw.info['sfreq'])]) > float(reject):
                    t += stride_sz if is_sz else stride_norm
                    continue

                rows.append({"recording_id": row['recording_id'], "path": row['path'], "start_sec": s, "end_sec": e,
                             "label": lbl})
                t += stride_sz if is_sz else stride_norm

        out = self.results_dir / f"window_index_{mode}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        return str(out)

    def generate_report(self, df_files: pd.DataFrame, index_path: str):
        print("\n" + "=" * 50)
        print(" FINAL PIPELINE REPORT")
        print("=" * 50)
        print(f" Metadata Extraction:")
        print(f"   - Total Subjects: {df_files['subject_id'].nunique()}")
        print(f"   - Total Recordings: {len(df_files)}")

        if 'age' in df_files.columns and 'sex' in df_files.columns:
            print(f"\n Stratification (Age/Sex):")
            print(f"   - Sex Distribution: {df_files.drop_duplicates('subject_id')['sex'].value_counts().to_dict()}")
            print(
                f"   - Age Distribution: {df_files.drop_duplicates('subject_id')['age'].value_counts().sort_index().to_dict()}")
        else:
            print("\n Stratification (Age/Sex): Not available.")

        if Path(index_path).exists():
            df_idx = pd.read_csv(index_path)
            counts = df_idx['label'].value_counts()
            print(f"\n Label Classification:")
            print(f"   - Normal Windows (0): {counts.get(0, 0)}")
            print(f"   - Seizure Windows (1): {counts.get(1, 0)}")
            if 1 in counts:
                ratio = counts[1] / (counts[0] + counts[1]) * 100
                print(f"   - Seizure Ratio: {ratio:.2f}%")
        print("=" * 50 + "\n")


# --- PYTORCH DATASET ---
class EEGDataset(Dataset):
    def __init__(self, index_csv, config_path="config.yaml"):
        self.index = pd.read_csv(index_csv)
        with open(config_path) as f: self.cfg = yaml.safe_load(f)
        self.cache = LRUCache(self.cfg['system'].get('cache_size', 10))
        self.normalize = self.cfg['signal'].get('normalize', False)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        row = self.index.iloc[idx]
        raw = self.cache.get(row['recording_id'])
        if raw is None:
            raw = mne.io.read_raw_fif(row['path'], preload=True, verbose='ERROR')
            self.cache.put(row['recording_id'], raw)
        sfreq = raw.info['sfreq']
        data = raw._data[:, int(row['start_sec'] * sfreq):int(row['end_sec'] * sfreq)]

        if self.normalize:
            q75, q25 = np.percentile(data, [75, 25], axis=1, keepdims=True)
            iqr = q75 - q25
            iqr[iqr == 0] = 1.0
            data = (data - np.median(data, axis=1, keepdims=True)) / iqr

        return torch.from_numpy(data.astype(np.float32)), torch.tensor(row['label'], dtype=torch.long)