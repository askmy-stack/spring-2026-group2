# EEG Seizure Detection – Feature Engineering Module

## Overview

This module converts BIDS-formatted EEG datasets into structured tabular feature datasets for machine learning.

It:
- Reads BIDS EEG recordings (*_eeg.edf)
- Reads corresponding seizure annotations (*_events.tsv)
- Applies sliding window segmentation
- Extracts time and frequency domain features
- Assigns binary seizure labels (0 = non-seizure, 1 = seizure)
- Exports a training-ready CSV file

---

# Core Files

project/
├── run_features.py
├── feature_engineering.py
└── configs/
    └── fe.yaml

---

# Quick Start

## 1. Install Dependencies

pip install numpy pandas scipy mne PyYAML

Optional virtual environment:

python -m venv .venv
.\.venv\Scripts\activate

---

## 2. Configure Paths

Edit configs/fe.yaml:

io:
  bids_root: "path/to/your/bids_dataset"
  output_csv: "results/features/features_output.csv"

Example:

io:
  bids_root: "data/ds005873"
  output_csv: "results/features/features_ds005873.csv"

---

## 3. Run Feature Extraction

python run_features.py --config configs/fe.yaml

---

## 4. Output

After execution:

results/features/features_ds005873.csv

Console output example:

Shape: (78996, 25)

0    78882
1      114

Where:
0 = non-seizure windows
1 = seizure windows

---

# How It Works

## 1. Locate EEG Files

The script recursively searches bids_root for:

*_eeg.edf

---

## 2. Match Events File

For each EEG file:

sub-001_ses-01_task-..._eeg.edf

The script looks for:

sub-001_ses-01_task-..._events.tsv

This follows the BIDS naming convention.

---

## 3. Seizure Labeling Logic

From events.tsv, the script reads:
- onset
- duration
- eventType

A window is labeled as seizure (1) if:

eventType starts with "sz"

Examples:
- sz
- sz_foc_ia_nm
- sz_gen_m_tonicClonic

Otherwise, the window is labeled 0.

---

## 4. Sliding Window Segmentation

Configured in fe.yaml:

windows:
  window_sec: 10
  step_sec: 5

Each recording is split into overlapping windows.

Each window becomes one row in the output CSV.

---

## 5. Feature Extraction

feature_engineering.py contains:

AdvancedFeatureExtractor

It extracts:

Time Domain Features:
- Mean
- Standard deviation
- RMS
- Line length
- Zero crossing rate
- Hjorth parameters

Frequency Domain Features:
- Welch Power Spectral Density
- Band powers:
  - Delta (0.5–4 Hz)
  - Theta (4–8 Hz)
  - Alpha (8–13 Hz)
  - Beta (13–30 Hz)
  - Gamma (30–50 Hz)
- Relative power
- Spectral entropy

---

# Output Format

Each row represents one EEG window.

Example columns:

ch0_mean
ch0_std
ch0_rms
delta_power
theta_power
alpha_power
beta_power
gamma_power
label
recording_path
events_path
start_sec
end_sec

---

# Label Definition

Binary classification:

0 → non-seizure  
1 → seizure  

A window is labeled 1 if it overlaps any seizure interval.

---

# Configuration Example (fe.yaml)

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
  type_cols: ["eventType", "trial_type", "event_type"]
  seizure_prefix: "sz"

fe:
  sfreq: 250
  bands:
    delta: [0.5, 4]
    theta: [4, 8]
    alpha: [8, 13]
    beta: [13, 30]
    gamma: [30, 50]

---

# Expected BIDS Structure

bids_root/
├── sub-001/
│   └── ses-01/
│       └── eeg/
│           ├── sub-001_..._eeg.edf
│           └── sub-001_..._events.tsv

---

# Testing Small Subsets

To test quickly:

limits:
  max_files: 2
  max_windows: 500

---

# Notes

- Seizures are rare; class imbalance is expected.
- The module is dataset-agnostic as long as the dataset follows BIDS.

