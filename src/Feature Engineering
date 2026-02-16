# EEG Seizure Detection – Feature Engineering

This folder contains the feature engineering part of our pipeline.

The goal of this module is simple:

Take BIDS-formatted EEG recordings as input  
Extract meaningful signal features  
Generate a CSV file with window-level seizure labels (0 and 1)

This output will later be used for classical ML models and deep learning baselines.

------------------------------------------------------------

Files in this module

run_features.py  
Main script that runs the full feature extraction pipeline.

feature_engineering.py  
Contains the AdvancedFeatureExtractor class where all features are implemented.

configs/fe.yaml  
Configuration file. All paths and parameters are controlled from here.

------------------------------------------------------------

How to Run

1) Install required libraries

pip install numpy pandas scipy mne PyYAML

2) Edit configs/fe.yaml

Update the dataset path:

io:
  bids_root: "path/to/your/bids_dataset"
  output_csv: "results/features/features_output.csv"

Example:

io:
  bids_root: "data/ds005873"
  output_csv: "results/features/features_ds005873.csv"

3) Run the script

python run_features.py --config configs/fe.yaml

After it finishes, you will see a CSV file inside:

results/features/

------------------------------------------------------------

What the Script Does

Step 1 – It scans the BIDS root directory and finds all files ending with:

*_eeg.edf

Step 2 – For each EEG file, it automatically finds the matching:

*_events.tsv

Step 3 – It reads seizure annotations from the events file.

If the eventType column starts with "sz", it is considered a seizure event.

Step 4 – The recording is split into sliding windows.

Window size and step size are controlled in fe.yaml:

windows:
  window_sec: 10
  step_sec: 5

Step 5 – For each window, features are extracted and a label is assigned:

0 → non-seizure  
1 → seizure  

A window gets label 1 if it overlaps with any seizure interval.

------------------------------------------------------------

Features Extracted

Time domain:
- Mean
- Standard deviation
- RMS
- Line length
- Zero crossing rate
- Hjorth parameters

Frequency domain:
- Welch power spectral density
- Delta band power
- Theta band power
- Alpha band power
- Beta band power
- Gamma band power
- Relative band power
- Spectral entropy

All features are computed per window.

------------------------------------------------------------

Output CSV Format

Each row represents one window.

Columns include:
- Extracted signal features
- label (0 or 1)
- recording_path
- events_path
- start_sec
- end_sec

Example:

label = 0 → background  
label = 1 → seizure  

------------------------------------------------------------

Important Notes

- Seizures are rare, so the dataset will be highly imbalanced.
- Some datasets may have only 2 EEG channels (e.g., wearable EEG). That is normal.
- The pipeline is dataset-agnostic as long as the dataset follows BIDS format.

------------------------------------------------------------

Testing on Small Subsets

If you don’t want to run the full dataset, update:

limits:
  max_files: 2
  max_windows: 500

This is useful for debugging.

------------------------------------------------------------

Summary

Input: BIDS EEG dataset  
Output: Tabular CSV with window-level seizure labels  

This module is fully controlled through fe.yaml and does not depend on dataset-specific hardcoding.

