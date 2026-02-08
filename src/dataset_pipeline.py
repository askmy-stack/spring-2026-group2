import os
import shutil
import numpy as np
import pandas as pd
import mne
import re
from dataloader import UniversalReader
from feature_engineering import AdvancedFeatureExtractor

# --- CONFIGURATION ---
TARGET_RATE = 256
WINDOW_SECONDS = 2

# MASTER DICTIONARY OF SEIZURE TIMES
CHB_SEIZURES = {
    # --- Patient 01 ---
    'chb01_03.edf': [(2996, 3036)],
    'chb01_04.edf': [(1467, 1494)],
    'chb01_15.edf': [(1732, 1772)],
    'chb01_16.edf': [(1015, 1066)],
    'chb01_18.edf': [(1720, 1759)],
    'chb01_26.edf': [(1862, 1963)],

    # --- Patient 02 ---
    'chb02_16.edf': [(130, 212)],
    'chb02_16+.edf': [(130, 212)],
    'chb02_19.edf': [(3369, 3425)],

    # --- Patient 03 ---
    'chb03_01.edf': [(362, 414)],
    'chb03_02.edf': [(731, 796)],
    'chb03_03.edf': [(432, 501)],

    # --- Patient 05 ---
    'chb05_06.edf': [(417, 532)],
    'chb05_13.edf': [(1086, 1196)],
    'chb05_16.edf': [(2317, 2413)],

    # --- Patient 06 ---
    'chb06_01.edf': [(172, 217), (1025, 1079), (2804, 2872)],
    'chb06_04.edf': [(327, 347), (6208, 6231)],
    'chb06_09.edf': [(12500, 12516)],

    # --- Patient 07 ---
    'chb07_12.edf': [(4920, 5006)],
    'chb07_13.edf': [(3285, 3381)],
    'chb07_19.edf': [(13688, 13766)],

    # === CUSTOM PATIENT (PN06) ===
    # ACTION REQUIRED: Replace (0, 0) with actual start/end seconds
    'PN06-5.edf': [(0, 0)],
}

class EEGJanitor:
    def __init__(self, target_rate=TARGET_RATE):
        self.target_rate = target_rate
        self.reader = UniversalReader(verbose=False)

    def process(self, file_path):
        try:
            raw = self.reader.read(file_path, preload=True)
        except Exception as e:
            print(f"Read Error: {e}")
            return None, None

        # 1. Smart Notch Filter (Dynamic)
        # Prevents crashing if sampling rate is too low for 60Hz harmonics
        sfreq = raw.info['sfreq']
        nyquist = sfreq / 2
        target_freqs = np.arange(60, 241, 60)
        safe_freqs = target_freqs[target_freqs < nyquist]

        if len(safe_freqs) > 0:
            raw.notch_filter(safe_freqs, verbose=False)

        # 2. Bandpass (1-50Hz)
        raw.filter(l_freq=1.0, h_freq=50.0, verbose=False)

        # 3. Resample
        if int(raw.info['sfreq']) != self.target_rate:
            raw.resample(self.target_rate, verbose=False)

        # 4. Metadata
        meta = {
            'age': raw.info['subject_info'].get('age') if raw.info['subject_info'] else 0,
            'sex': raw.info['subject_info'].get('sex') if raw.info['subject_info'] else 0
        }

        return raw, meta

class DatasetBuilder:
    def __init__(self, raw_dir, out_dir):
        self.raw_dir = raw_dir
        self.out_dir = out_dir
        self.janitor = EEGJanitor()
        self.extractor = AdvancedFeatureExtractor()

    def get_seizure_times(self, filename, raw):
        if filename in CHB_SEIZURES:
            return CHB_SEIZURES[filename]

        for key in CHB_SEIZURES:
            if key in filename or filename in key:
                if len(key) == len(filename) or abs(len(key) - len(filename)) == 1:
                    return CHB_SEIZURES[key]

        # 3. Check Internal Annotations (BIDS/EDF+ style)
        times = []
        if raw.annotations:
            for ann in raw.annotations:
                desc = ann['description'].lower()
                if 'sz' in desc or 'seiz' in desc:
                    times.append((ann['onset'], ann['onset'] + ann['duration']))
        return times

    def run(self):
        if os.path.exists(self.out_dir): shutil.rmtree(self.out_dir)
        os.makedirs(self.out_dir)

        files = [f for f in os.listdir(self.raw_dir) if f.endswith(('.edf', '.mat', '.cnt'))]
        files.sort()

        metadata_records = []
        stats = {'normal': 0, 'seizure': 0}

        for filename in files:
            print(f"Processing {filename}...")
            raw, meta = self.janitor.process(os.path.join(self.raw_dir, filename))
            if not raw: continue

            seizures = self.get_seizure_times(filename, raw)
            is_seizure_file = len(seizures) > 0

            if 'PN06' in filename and seizures == [(0, 0)]:
                print("  [WARNING] PN06 has placeholder times (0,0). Treating as NORMAL.")
                is_seizure_file = False
                seizures = []

            if is_seizure_file:
                print(f"  -> Found Seizure(s): {seizures}")

            # 75% Overlap for Seizures to balance data
            overlap = 0.75 if is_seizure_file else 0.0

            data = raw.get_data()
            if data.shape[1] < int(raw.info['sfreq'] * WINDOW_SECONDS):
                print(f"  -> Skipped (Too Short)")
                continue

            sfreq = raw.info['sfreq']
            win_samples = int(sfreq * WINDOW_SECONDS)
            stride = int(win_samples * (1 - overlap))

            start = 0
            while start + win_samples < data.shape[1]:
                end = start + win_samples

                # Labeling Logic
                t_start, t_end = start / sfreq, end / sfreq
                label = 0
                for (s_start, s_end) in seizures:
                    # If window overlaps significantly with seizure
                    if t_start < s_end and t_end > s_start:
                        label = 1;
                        break

                # Slice Signal
                epoch = data[:, start:end]
                base_name = f"{filename.split('.')[0]}_s{start}"

                # Save Raw Data
                np.save(os.path.join(self.out_dir, f"{base_name}_raw.npy"), epoch)

                # Save Hand-Crafted Features
                feats = self.extractor.extract(epoch)
                np.save(os.path.join(self.out_dir, f"{base_name}_feat.npy"), feats)

                # Update Stats
                if label == 1:
                    stats['seizure'] += 1
                else:
                    stats['normal'] += 1

                metadata_records.append({
                    'filename': filename,
                    'slice_id': base_name,
                    'label': label,
                    'patient_id': re.split(r'[_-]', filename)[0]  # Extract Patient ID
                })

                start += stride

        if metadata_records:
            pd.DataFrame(metadata_records).to_csv(os.path.join(self.out_dir, "metadata.csv"), index=False)
            print(f"\n>>> PIPELINE COMPLETE <<<")
            print(f"Normal Windows: {stats['normal']}")
            print(f"Seizure Windows: {stats['seizure']}")
            if stats['seizure'] < 300:
                print("[!] Warning: Seizure count is low. Check CHB_SEIZURES dictionary.")
        else:
            print("\nNo data processed. Check raw directory.")


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw = os.path.join(base, "seizure_system/data/raw_edf/chb_mit")
    out = os.path.join(base, "seizure_system/data/processed_hybrid")
    DatasetBuilder(raw, out).run()
