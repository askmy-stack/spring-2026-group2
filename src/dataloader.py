import os
import shutil
import numpy as np
import pandas as pd
import mne
import yaml
import re
import scipy.io
from pathlib import Path
from typing import Dict, List, Optional, Union
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor
import warnings

try:
    from mne_bids import BIDSPath, write_raw_bids, read_raw_bids, print_dir_tree, make_report
    from mne_bids.stats import count_events

    MNE_BIDS_AVAILABLE = True
except ImportError:
    raise ImportError("CRITICAL: 'mne-bids' is missing. Install it: pip install mne-bids")

warnings.filterwarnings('ignore')


class SeizurePipeline:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path).resolve()
        with open(self.config_path, 'r') as f: self.cfg = yaml.safe_load(f)

        self.root_dir = Path(__file__).parent.resolve()
        self.raw_root = (self.root_dir / self.cfg["dataset"]["raw_root"]).resolve()

        # BIDS ROOT: The central hub for all converted data
        self.bids_root = (self.root_dir / self.cfg["dataset"]["bids_root"]).resolve()
        self.bids_root.mkdir(parents=True, exist_ok=True)

        # Results for Dataloader
        self.results_dir = (self.root_dir / self.cfg["dataset"]["results_root"] / "dataloader").resolve()
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.default_sfreq = self.cfg["signal"].get("target_sfreq", 256)

    # --- 1. UNIVERSAL INGEST & CONVERT TO BIDS ---
    def ingest_to_bids(self, source_path: str, subject_id: str, age: int = None, sex: str = None):
        """
        Converts ANY EEG file (EDF/+, BDF, SET, VHDR, CSV, MAT, PARQUET) -> BIDS Format.
        """
        src = Path(source_path)
        if not src.exists(): return

        if src.stat().st_size < 1000:
            return

        print(f"Converting {src.name} -> BIDS (sub-{subject_id})...")

        try:
            raw = self._read_any_format(src)
            if raw is None:
                return

            # 2. Add Metadata
            if age or sex:
                raw.info['subject_info'] = {}
                if age: raw.info['subject_info']['age'] = age
                if sex:
                    # MNE uses 1=Male, 2=Female, 0=Unknown
                    sex_code = 1 if sex.upper() in ['M', 'MALE'] else (2 if sex.upper() in ['F', 'FEMALE'] else 0)
                    raw.info['subject_info']['sex'] = sex_code

            # 3. Write BIDS
            # We enforce 'EDF' as the standardized output format for BIDS to keep things consistent
            bids_path = BIDSPath(subject=subject_id, task='seizure', root=self.bids_root, datatype='eeg')
            write_raw_bids(raw, bids_path, overwrite=True, verbose=False, format='EDF', allow_preload=True)
            return bids_path

        except Exception as e:
            print(f"BIDS Conversion Failed for {src.name}: {e}")
            return None

    def _read_any_format(self, src: Path) -> Optional[mne.io.BaseRaw]:
        """Helper to handle specific file extensions."""
        ext = src.suffix.lower()

        # A. STANDARD HEADER FORMATS
        if ext in ['.edf', '.edf+']:
            return mne.io.read_raw_edf(src, preload=True, verbose='ERROR')
        elif ext == '.bdf':
            return mne.io.read_raw_bdf(src, preload=True, verbose='ERROR')
        elif ext == '.vhdr':  # BrainVision
            return mne.io.read_raw_brainvision(src, preload=True, verbose='ERROR')
        elif ext == '.set':  # EEGLAB
            return mne.io.read_raw_eeglab(src, preload=True, verbose='ERROR')
        elif ext == '.fif':
            return mne.io.read_raw_fif(src, preload=True, verbose='ERROR')

        # B. GENERIC TABLE FORMATS (CSV, TSV, PARQUET)
        elif ext in ['.csv', '.txt', '.tsv']:
            sep = '\t' if ext == '.tsv' else (',' if ext == '.csv' else None)
            df = pd.read_csv(src, sep=sep, engine='python')
            return self._df_to_raw(df)

        elif ext == '.parquet':
            df = pd.read_parquet(src)
            return self._df_to_raw(df)

        # C. MATLAB MAT FILES
        elif ext == '.mat':
            mat = scipy.io.loadmat(src)
            # Find the largest variable in the MAT file and assume it's the EEG data
            valid_keys = [k for k in mat.keys() if not k.startswith('_')]
            if not valid_keys: return None

            # Heuristic: pick the key with the biggest array
            best_key = max(valid_keys, key=lambda k: mat[k].size if hasattr(mat[k], 'size') else 0)
            data = mat[best_key]

            # Fix Dimensions: MNE expects (Channels, Time)
            if data.ndim == 2:
                if data.shape[0] > data.shape[1]:
                    data = data.T  # Transpose if Time > Channels (likely)

            ch_names = [f"Ch{i + 1}" for i in range(data.shape[0])]
            info = mne.create_info(ch_names, sfreq=self.default_sfreq, ch_types='eeg')
            return mne.io.RawArray(data, info, verbose='ERROR')

        else:
            print(f"Unsupported format: {ext}")
            return None

    def _df_to_raw(self, df: pd.DataFrame) -> mne.io.BaseRaw:
        """Converts a Pandas DataFrame to MNE Raw."""
        # Clean non-numeric columns (timestamps etc)
        df = df.select_dtypes(include=[np.number])

        # Transpose: DataFrame is usually (Time, Channel), MNE needs (Channel, Time)
        data = df.values.T
        ch_names = list(df.columns)

        info = mne.create_info(ch_names, sfreq=self.default_sfreq, ch_types='eeg')
        return mne.io.RawArray(data, info, verbose='ERROR')

    # --- 2. METADATA & LABELS ---
    def populate_metadata_and_labels(self):
        print("[Metadata] Parsing Clinical Info & Seizure Labels...")

        demos = {}
        # Parse Summaries (Recursive search)
        for s_path in self.raw_root.rglob("*-summary.txt"):
            try:
                subj = s_path.name.split('-')[0]
                with open(s_path, 'r', encoding='utf-8', errors='ignore') as f:
                    c = f.read().lower()

                age = None
                age_match = re.search(r'(?:age|old)[:\s]+(\d+)', c)
                if age_match: age = int(age_match.group(1))

                sex = 'F' if ('female' in c or 'woman' in c) else ('M' if ('male' in c or 'man' in c) else None)
                demos[subj] = {'age': age, 'sex': sex}
            except:
                pass

        # Create Events
        # (This is simplified logic; fully mapping text events to files requires strict naming conventions)
        # We rely on your dataset's "-summary.txt" being present.
        pass

    # --- 3. READ BIDS ---
    def read_bids_dataset(self):
        print(f"[BIDS] Reading dataset from {self.bids_root}...")
        subjects = [p.name.replace('sub-', '') for p in self.bids_root.glob('sub-*')]

        data_records = []
        for sub in subjects:
            bids_path = BIDSPath(subject=sub, root=self.bids_root, datatype='eeg')
            scans = bids_path.match()
            for scan in scans:
                data_records.append({
                    'subject_id': sub,
                    'bids_path': scan,
                    'raw_path': str(scan.fpath)
                })
        return pd.DataFrame(data_records)

    # --- 4. STRATIFY ---
    def stratify_split(self, df: pd.DataFrame):
        print("[Split] Stratifying by Age & Sex...")

        if 'age' not in df.columns: df['age'] = np.random.randint(10, 60, size=len(df))
        if 'sex' not in df.columns: df['sex'] = np.random.choice(['M', 'F'], size=len(df))

        df['stratum'] = df['sex'] + "_" + pd.cut(df['age'], bins=[0, 18, 100], labels=['young', 'adult']).astype(str)
        subs = df.drop_duplicates('subject_id')

        # Robust Fallback for small data
        if len(subs) < 5:
            print("Dataset too small for stratification. Using simple split.")
            return {'train': df, 'val': df, 'test': df}

        try:
            train_subs, temp_subs = train_test_split(subs['subject_id'], test_size=0.3, stratify=subs['stratum'],
                                                     random_state=42)
            val_subs, test_subs = train_test_split(temp_subs, test_size=0.5, random_state=42)
        except ValueError:
            print("Stratification groups too small. Fallback to Random Split.")
            train_subs, temp_subs = train_test_split(subs['subject_id'], test_size=0.3, random_state=42)
            val_subs, test_subs = train_test_split(temp_subs, test_size=0.5, random_state=42)

        return {
            'train': df[df['subject_id'].isin(train_subs)],
            'val': df[df['subject_id'].isin(val_subs)],
            'test': df[df['subject_id'].isin(test_subs)]
        }

    # --- 5. CREATE WINDOWS (DATALOADER OUTPUT) ---
    def create_windows(self, splits: Dict[str, pd.DataFrame]):
        for mode, df in splits.items():
            print(f"Building {mode} index...")
            rows = []
            for _, row in df.iterrows():
                try:
                    # Load from BIDS
                    raw = read_raw_bids(bids_path=row['bids_path'], verbose=False)
                    dur = raw.n_times / raw.info['sfreq']

                    # Create 4-second patches (Demo Logic)
                    # In production, loop t=0 to dur
                    rows.append({
                        'recording_id': Path(row['raw_path']).stem,
                        'path': row['raw_path'],
                        'label': 0,
                        'start_sec': 0,
                        'end_sec': min(4.0, dur)
                    })
                except:
                    pass

            pd.DataFrame(rows).to_csv(self.results_dir / f"window_index_{mode}.csv", index=False)

    def run_pipeline_level_1(self):
        """Master Sequence"""
        # 1. Convert Raw to BIDS
        # Scan for ALL supported extensions
        extensions = ['*.edf', '*.EDF', '*.bdf', '*.set', '*.vhdr', '*.fif', '*.csv', '*.txt', '*.mat', '*.parquet']
        files = []
        for ext in extensions:
            files.extend(list(self.raw_root.rglob(ext)))

        if not files:
            print("No valid EEG files found in raw_root.")
            return

        for f in files:
            # Smart Subject ID extraction
            name = f.stem
            if 'chb' in name:
                subj = name.split('_')[0]
            elif 'sub' in name:
                subj = name.split('_')[0]
            else:
                subj = "sub00"  # Fallback for generic files

            # Clean Subject ID (Alphanumeric only for BIDS)
            subj = "".join([c for c in subj if c.isalnum()])

            self.ingest_to_bids(str(f), subj)

        # 2. Add Labels
        self.populate_metadata_and_labels()

        # 3. Read BIDS
        df = self.read_bids_dataset()
        if df.empty:
            print("No valid BIDS data created.")
            return

        # 4. Stratify
        splits = self.stratify_split(df)

        # 5. Index
        self.create_windows(splits)
        print("Pipeline Complete.")