# EEG Pipeline (Windowed Seizure Detection)

This repository contains an end-to-end EEG processing pipeline for **window-level seizure detection**.  
It consumes window indices produced by the dataloader stage (CSV files), loads EDF windows, applies optional preprocessing and QC, generates per-window EDA artifacts (plots + CSVs), builds a dataset overview summary, and can export cleaned windows into a **BIDS-derivatives** style dataset.

---

## Folder Structure
src/eeg_pipeline/
analysis/
time_domain.py # QC, bad channel rules, epoching, etc.
freq_domain.py # PSD, bandpower, FFT, spectrogram, Morlet, etc.
configs/
config.yaml # main pipeline config
core/
artifacts.py # JSON/CSV/plot writers
yaml_utils.py # config helpers (load_yaml, get)
bids_io.py # BIDS read utilities (if needed)
bids_derivatives.py # BIDS-derivative export writer (cleaned outputs)
pipeline/
run_pipeline.py # main runner (reads dataloader index CSVs)
preprocessor.py # preprocessing steps (filtering, reref, etc.)
filtering.py # filter implementation helpers
eda_engine.py # per-window EDA outputs
dataset_overview.py # windows.csv / recordings.csv / summary.json / charts
bot_diagrams.py # Mermaid diagram generator
verify.py # optional validation utilities
main.py #  single entrypoint to run the pipeline



---

## What the Pipeline Produces

Outputs (default paths from config):

### 1) Per-window EDA
`results/preprocess/eda/`

Examples (per window):
- `raw_before.png`, `raw_after.png`
- `qc.json`
- `psd_mean_uV2_per_hz.csv`
- `bandpower_uV2.csv`
- `psd_spectrogram_*.png` (if enabled)
- `morlet_spectrogram_*.png` (if enabled)
- `fft_*.csv/.png` (if enabled)

### 2) Dataset overview
`results/preprocess/overview/`
- `windows.csv` (window metadata + QC summary + pointers to artifacts)
- `recordings.csv` (recording-level info: duration, sfreq, channels)
- `summary.json` (counts, QC aggregates)
- charts in `results/preprocess/overview/charts/`

### 3) Diagrams
`results/preprocess/diagrams/`
- `eda.mmd`
- `modules.mmd`

### 4) Optional BIDS derivatives export
If enabled:
`results/preprocess/bids_dataset/`
- cleaned EDF windows in BIDS-like structure
- sidecars like `*_eeg.json`, `*_channels.tsv`, `*_events.tsv`

---

## Install

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Configure

Edit:
src/eeg_pipeline/configs/config.yaml

Important sections:

dataloader_index.csv_paths: points to train/val/test window index CSVs

run.*: enable/disable diagrams, preprocess, EDA

preprocess.* and analysis.*: filtering, QC, spectrograms, FFT, Morlet

outputs.*: where artifacts are written

export_cleaned.*: BIDS derivative export (optional)

## Run (Single Command)

Use the single entrypoint:
PYTHONPATH=src python3 src/eeg_pipeline/main.py --config src/eeg_pipeline/configs/config.yaml

## Notes / Tips

For debugging, limit output volume using:

eda.max_windows_total

eda.max_windows_per_subject

Window-level QC and EDA are designed to help validate labels, identify artifacts, and generate quick frequency-based features (bandpower/PSD/FFT).

Subject-wise splitting is controlled upstream by how the dataloader created the train/val/test CSVs.
