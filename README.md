```markdown
# EEG Pipeline for Window-Level Seizure Detection (CHB-MIT)

This repository contains a complete EEG processing pipeline designed for **window-level seizure detection**.  
It takes EEG windows indexed by the dataloader stage, applies optional preprocessing (filtering, re-referencing), runs **quality control (QC)** and **frequency/time-domain analysis**, produces per-window EDA artifacts (plots + CSVs), generates dataset-wide summaries, and optionally exports cleaned windows into a **BIDS-derivatives** style dataset.

The goal is to make the workflow **reproducible, inspectable, and ML-ready** (e.g., CNN training).

---

## Why this pipeline exists

EEG seizure detection often fails in practice because:
- raw EEG files are large and messy,
- labels are event-based but models are trained on short windows,
- preprocessing and QC decisions are hard to track,
- models silently learn artifacts if QC is not visible.

This pipeline addresses that by:
- working at the **window** level (the unit used for ML),
- generating **QC logs and EDA outputs** per window,
- producing **dataset-level summaries** to validate data quality and class balance,
- supporting **BIDS-format outputs** so the results are standardized and easier to share/debug.

---

## Input and Output Overview

### Inputs
1. **Window index CSVs** produced by your dataloader:
   - `results/dataloader/window_index_train.csv`
   - `results/dataloader/window_index_val.csv`
   - `results/dataloader/window_index_test.csv`

   Expected columns:
   - `path` (EDF path)
   - `subject_id`
   - `start_sec`, `end_sec`
   - `label` (0=nonseizure, 1=seizure)
   - optional: `age`, `sex`

2. **EDF files** (from CHB-MIT or other EEG sources) referenced by the CSV `path`.

3. **Config file**:
   - `src/eeg_pipeline/configs/config.yaml`

---

### Outputs (Default)
All pipeline outputs go under:

- `results/preprocess/eda/`  
  Per-window artifacts (QC + plots + CSVs)

- `results/preprocess/overview/`  
  Dataset-level overview: `windows.csv`, `summary.json`, charts

- `results/preprocess/diagrams/`  
  Mermaid diagrams describing enabled modules: `eda.mmd`, `modules.mmd`

Optional:
- `results/preprocess/bids_dataset/`  
  Cleaned EEG windows exported as BIDS derivatives (EDF + sidecars)

---

## Repository Structure

```

- src/eeg_pipeline/
- analysis/
- time_domain.py        # QC, bad channel logic, epoching, time-domain utilities
- freq_domain.py        # PSD, bandpower, FFT, spectrograms, Morlet, frequency utilities
- configs/
- config.yaml           # main pipeline config
- core/
- artifacts.py          # artifact writer (plots/CSVs/JSON)
- yaml_utils.py         # YAML helpers: load_yaml(), get()
- bids_io.py            # BIDS loading helpers (if needed)
- bids_derivatives.py   # export cleaned EEG to BIDS-derivatives
- pipeline/
- run_pipeline.py       # main runner (reads dataloader CSVs, processes windows)
- preprocessor.py       # preprocessing orchestration
- filtering.py          # filtering helpers
- eda_engine.py         # per-window EDA generation
- dataset_overview.py   # summary CSV/JSON + charts
- bot_diagrams.py       # Mermaid diagram generation from config flags
- verify.py             # optional sanity checks
- main.py                 # single entrypoint to run the whole pipeline

````

---

## Installation

### 1) Create an environment
```bash
python -m venv .venv
source .venv/bin/activate
````

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

---

## Configuration (config.yaml)

Main config location:

* `src/eeg_pipeline/configs/config.yaml`

Key sections:

### Pipeline switches

```yaml
run:
  diagrams: true
  preprocess: true
  eda: true
```

### Input dataset

```yaml
dataset:
  bids_root: "results/bids_dataset"
```

### Dataloader index inputs

```yaml
dataloader_index:
  csv_paths:
    - "results/dataloader/window_index_train.csv"
    - "results/dataloader/window_index_val.csv"
    - "results/dataloader/window_index_test.csv"
```

### Outputs

```yaml
outputs:
  eda_root: "results/preprocess/eda"
  diagrams_root: "results/preprocess/diagrams"
  overview_root: "results/preprocess/overview"
```

### Preprocessing

* Bandpass + notch filtering
* Optional reref / ICA depending on config

### QC + Bad Channels

QC settings include:

- * NaN checks
- * clipping threshold (`max_abs_uV`)
- * flatline detection (`flat_var_thresh_uV2`)
- * noisy channels by variance outliers (`noisy_var_factor`)

Bad channel module can:

- * mark bads using QC rules
- * interpolate bad channels (optional)

### Frequency-domain / Time-frequency analysis

- * PSD (Welch)
- * bandpower
- * PSD spectrogram
- * FFT
- * Morlet spectrogram (if enabled)

### export (BIDS derivatives)

```yaml
export_cleaned:
  enabled: true
  out_root: "results/preprocess/bids_dataset"
```

---

## Running the Pipeline (One Command)

- Use the single entrypoint:

```bash
PYTHONPATH=src python3 src/eeg_pipeline/main.py --config src/eeg_pipeline/configs/config.yaml
```

What this does:

- 1. Loads the window index CSVs
- 2. For each row/window:

  - * loads the EDF file (cached per file)
   - * crops the window `[start_sec, end_sec)`
   - * preprocesses (if enabled)
   - * runs QC and marks bad channels (if enabled)
   - * generates EDA artifacts (if enabled)
3. Generates dataset overview (summary JSON + CSV + charts)
4. Optionally exports cleaned windows to BIDS derivatives

---

## What Each Output Means (Important for Evaluation)

### Per-window QC (`qc.json`)

Saved in each window’s EDA folder. Typical fields include:

* NaN indicators (`has_nan`, `nan_frac`)
* variance summaries (`median_var_uV2`, etc.)
* lists/counts of flagged channels:

  * `flat_channels`, `noisy_channels`, `clipped_channels`
* `bads` and `n_bads` (channels marked bad)
* `noisy_channel_frac` (fraction of channels marked bad)

Why it matters:

* prevents training models on corrupted/noisy EEG windows
* provides traceability and reproducibility

---

### Per-window frequency artifacts

Examples:

* `psd_mean_uV2_per_hz.csv`
* `bandpower_uV2.csv`
* `fft_*.csv`
* `psd_spectrogram_*.png`
* `morlet_spectrogram_*.png`

Why it matters:

* verifies filtering and spectral structure
* provides baseline features (bandpower/PSD) for ML
* helps interpret differences between seizure/non-seizure windows

---

### Dataset overview (`results/preprocess/overview/`)

* `windows.csv`: a structured index of windows + QC + artifact pointers
* `summary.json`: dataset summary (counts, QC means)
* charts:

  * label distribution
  * windows per subject

Why it matters:

* verifies class balance and subject distribution
* provides high-level QC statistics so issues are caught early

---

## Notes on BIDS Export (Derivatives)

If export is enabled, cleaned windows can be written as:

* EDF in BIDS-structured folders (`sub-*/eeg/*.edf`)
* sidecar metadata (`*_eeg.json`, `*_channels.tsv`, `*_events.tsv`)

This helps:

* keep processed data in a standardized form
* preserve metadata needed for downstream tools
* enable validation and sharing

---

## Common Debug Tips (Things That Prevent Mistakes)

1. **Start with limited windows**
   Use:

   * `eda.max_windows_total`
   * `eda.max_windows_per_subject`
     to avoid generating thousands of artifacts while debugging.

2. **Check QC alignment**
   If summary QC looks correct but per-window `qc.json` doesn’t, ensure the same QC dict is passed consistently through pipeline → EDA writer.

3. **Subject-wise split**
   Ensure dataloader creates train/val/test splits by subject to avoid leakage.

4. **Artifacts ≠ Labels**
   A spectrogram showing strong low frequency power can be normal EEG or artifact. Use QC + raw plots to confirm.


```
```
