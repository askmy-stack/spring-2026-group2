# EEG Seizure Detection – Feature Engineering

## Core Files (3 Files Only)

```
project/
├── run_features.py          # Main script to generate window-level features
├── feature_engineering.py   # AdvancedFeatureExtractor (all feature logic)
└── configs/
    └── fe.yaml              # Configuration (paths, windows, labeling, etc.)
```

## Quick Start

### 1. Install Dependencies

```bash
pip install numpy pandas scipy mne PyYAML
```

### 2. Configure Paths

Edit `configs/fe.yaml`:

```yaml
io:
  bids_root: "path/to/your/bids_dataset"
  output_csv: "results/features/features_output.csv"
```

Example:

```yaml
io:
  bids_root: "data/ds005873"
  output_csv: "results/features/features_ds005873.csv"
```

### 3. Run Feature Extraction

```bash
python run_features.py --config configs/fe.yaml
```

### 4. Verify Output

After running, the features will be saved at:

```
results/features/features_output.csv
```

The script also prints the label distribution:

- `0` = non-seizure
- `1` = seizure

---

## What This Module Does

This feature engineering step takes a BIDS dataset and produces a single CSV where each row is one EEG window.

High level flow:

1. Find all `*_eeg.edf` files under the BIDS root
2. For each EEG file, find the matching `*_events.tsv`
3. Create sliding windows over the recording (window size + step size from config)
4. Assign a label per window using the events file
5. Extract signal features for each window
6. Write everything to one CSV

---

## Seizure Labeling (0/1)

Labels are created from the BIDS events file (`*_events.tsv`).

A window is labeled seizure (`1`) if it overlaps any seizure interval defined by:

- `onset` (seconds)
- `duration` (seconds)
- `eventType` (or another type column depending on dataset)

For SeizeIT2 (and many BIDS seizure datasets), seizure event codes start with:

- `sz` (example: `sz_foc_ia_nm`)

So the rule is:

- `label = 1` if eventType starts with `"sz"` and the window overlaps that interval
- `label = 0` otherwise

---

## Windowing

Configured in `configs/fe.yaml`:

```yaml
windows:
  window_sec: 10
  step_sec: 5
```

This creates overlapping windows like:

- Window 1: 0–10s
- Window 2: 5–15s
- Window 3: 10–20s
- ...

Each window becomes one row in the final CSV.

---

## Features Extracted

All features are implemented in `feature_engineering.py` inside `AdvancedFeatureExtractor`.

Typical features include:

Time domain:
- mean
- std
- rms
- line length
- zero crossing rate
- hjorth parameters

Frequency domain:
- Welch PSD
- band power (delta/theta/alpha/beta/gamma)
- relative band power
- spectral entropy

Exact features can be turned on/off using the config.

---

## Configuration

### Example `configs/fe.yaml`

```yaml
io:
  bids_root: "data/ds005873"
  output_csv: "results/features/features_ds005873.csv"

bids:
  modality: "eeg"
  pick_eeg_only: true

windows:
  window_sec: 10
  step_sec: 5

limits:
  max_files: null
  max_windows: null

labeling:
  onset_col: "onset"
  duration_col: "duration"
  type_cols: ["eventType", "trial_type", "event_type", "value", "description"]
  seizure_prefix: "sz"

fe:
  sfreq: 250
```

### Limits (for quick testing)

To run a small test:

```yaml
limits:
  max_files: 5
  max_windows: 1000
```

To run full dataset:

```yaml
limits:
  max_files: null
  max_windows: null
```

---

## Expected BIDS Structure

The module expects standard BIDS EEG file naming:

```
bids_root/
├── sub-001/
│   └── ses-01/
│       └── eeg/
│           ├── sub-001_..._eeg.edf
│           └── sub-001_..._events.tsv
```

---

## Output Format

The output is one CSV file. Each row represents one EEG window.

Columns include:
- extracted features (from `AdvancedFeatureExtractor`)
- label (0/1)
- recording_path
- events_path
- start_sec
- end_sec

---

## Troubleshooting

### Only label = 0 showing up
This usually means the EEG files you processed do not contain seizure events.

To confirm, open a matching `*_events.tsv` and check if it contains any `eventType` values starting with `sz`.

### Only 2 channels detected
Some datasets (like wearable EEG datasets) naturally have only 2 EEG channels. This depends on the dataset, not the code.

### Too many non-seizure windows
Seizures are rare, so imbalance is expected. Balancing should be handled during training (class weights, sampling, etc.).

---

## Output Structure

After running:

```
results/
└── features/
    └── features_ds005873.csv
```

---

## Quick Reference

```bash
# Run full pipeline
python run_features.py --config configs/fe.yaml

# Run a small test (set max_files/max_windows in the YAML)
python run_features.py --config configs/fe.yaml
```
