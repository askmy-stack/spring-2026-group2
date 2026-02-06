import os
import shutil
import numpy as np
import mne
import re
from dataloader import UniversalReader

# --- CONFIGURATION ---
TARGET_RATE = 256
WINDOW_SECONDS = 2
CHANNELS = [
    'FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2'
]

# External Labels (CHB-MIT Specific - Example)
# NOTE: If you switch datasets (e.g. to TUH or BIDS), you will need to
# update this logic to read from .tsv or .csv annotation files instead.
CHB_SEIZURES = {
    'chb01_03.edf': [(2996, 3036)],
    'chb01_04.edf': [(1467, 1494)],
    'chb01_15.edf': [(1732, 1772)],
    'chb01_16.edf': [(1015, 1066)],
    'chb01_18.edf': [(1720, 1759)],
    'chb01_26.edf': [(1862, 1963)],
}


class EEGJanitor:
    def __init__(self, target_rate=TARGET_RATE):
        self.target_rate = target_rate
        self.reader = UniversalReader(verbose=False)  # <--- INITIALIZE READER

    def process(self, file_path):
        try:
            # <--- UPDATED: USE UNIVERSAL READER INSTEAD OF READ_RAW_EDF
            raw = self.reader.read(file_path, preload=True)
        except Exception as e:
            print(f"Read Error: {e}")
            return None

        # 1. Standardize Names
        # This logic attempts to map various hospital naming conventions to a standard 10-20 system
        mapping = {}
        existing = set()
        for name in raw.ch_names:
            # Common cleanup for standardizing channel names
            clean = name.upper().replace('EEG ', '').replace('-REF', '').strip()
            clean = re.sub(r'-\d+$', '', clean)
            clean = clean.replace('T7', 'T3').replace('P7', 'T5').replace('T8', 'T4').replace('P8', 'T6')

            if clean in existing:
                clean += "-DUP"
            else:
                existing.add(clean)
            mapping[name] = clean

        try:
            raw.rename_channels(mapping)
        except:
            pass

        # 2. Resample & Filter
        if int(raw.info['sfreq']) != self.target_rate:
            raw.resample(self.target_rate, verbose=False)
        raw.filter(l_freq=1.0, h_freq=50.0, verbose=False)

        # 3. Pick Channels
        try:
            raw.pick(CHANNELS)
        except ValueError:
            return None

        return raw


class DatasetBuilder:
    def __init__(self, raw_dir, out_dir):
        self.raw_dir = raw_dir
        self.out_dir = out_dir
        self.janitor = EEGJanitor()

    def get_seizure_times(self, filename, raw):
        # 1. Check Hardcoded Dictionary (Legacy/CHB-MIT)
        if filename in CHB_SEIZURES:
            return CHB_SEIZURES[filename]

        # 2. Check Internal File Annotations (Standard for BIDS/EDF+)
        times = []
        if raw.annotations:
            for ann in raw.annotations:
                desc = ann['description'].lower()
                # Keywords for detecting seizures in annotation files
                if 'sz' in desc or 'seiz' in desc or 'ictal' in desc:
                    times.append((ann['onset'], ann['onset'] + ann['duration']))
        return times

    def run(self):
        if os.path.exists(self.out_dir): shutil.rmtree(self.out_dir)
        os.makedirs(self.out_dir)

        # Supports EDF, CNT, BDF, etc.
        valid_exts = ('.edf', '.bdf', '.cnt', '.vhdr')
        files = [f for f in os.listdir(self.raw_dir) if f.lower().endswith(valid_exts)]
        files.sort()

        stats = {'normal': 0, 'seizure': 0}

        for filename in files:
            print(f"Processing {filename}...")
            raw = self.janitor.process(os.path.join(self.raw_dir, filename))
            if not raw: continue

            seizures = self.get_seizure_times(filename, raw)
            is_seizure_file = len(seizures) > 0

            # Logic: 75% Overlap for Seizures, 0% for Normal
            overlap = 0.75 if is_seizure_file else 0.0

            data = raw.get_data()
            sfreq = raw.info['sfreq']

            win_samples = int(sfreq * WINDOW_SECONDS)
            stride_samples = int(win_samples * (1 - overlap))

            start = 0
            while start + win_samples < data.shape[1]:
                end = start + win_samples

                t_start, t_end = start / sfreq, end / sfreq
                label = 0
                for (s_start, s_end) in seizures:
                    if t_start < s_end and t_end > s_start:
                        label = 1;
                        break

                # Save slice
                save_name = f"{filename.split('.')[0]}_s{start}_lbl{label}.npy"
                np.save(os.path.join(self.out_dir, save_name), data[:, start:end])

                if label == 1:
                    stats['seizure'] += 1
                else:
                    stats['normal'] += 1

                start += stride_samples

        print(f"\nDone. Normal: {stats['normal']}, Seizure: {stats['seizure']}")


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Update this path to point to your specific dataset folder (e.g. BIDS root or EDF folder)
    raw = os.path.join(base, "seizure_system/data/raw_edf/chb_mit")
    out = os.path.join(base, "seizure_system/data/processed_tensors/chb_mit_labeled")
    DatasetBuilder(raw, out).run()