````markdown
# EEG Pipeline for Window-Level Seizure Detection (CHB-MIT)

This repository implements an end-to-end **window-level EEG preprocessing + QC + EDA pipeline** to support **seizure detection** modeling (e.g., CNNs).  
It consumes window indices produced by the dataloader stage, loads EDF windows, optionally preprocesses them (filtering / rereference), runs QC and frequency-domain analysis, generates per-window artifacts (plots/CSVs/JSON), and builds dataset-wide summaries for validation.

---

## 1) What this pipeline does

### Inputs (what you need before running)
- **Window index CSVs** from the dataloader:
  - `results/dataloader/window_index_train.csv`
  - `results/dataloader/window_index_val.csv`
  - `results/dataloader/window_index_test.csv`
- **EDF files** referenced by the `path` column in the CSVs.
- **Config file**:
  - `src/eeg_pipeline/configs/config.yaml`

**Expected columns in each window index CSV**
- `path` — EDF file path (absolute or repo-relative)
- `subject_id` — subject identifier (e.g., `chb01`)
- `start_sec`, `end_sec` — window boundaries in seconds
- `label` — `0` = non-seizure, `1` = seizure
- optional: `age`, `sex`

### Processing steps (high-level)
For each window row in the CSV:
1. Load the EDF (cached per EDF file to avoid repeated loads)
2. Crop the window `[start_sec, end_sec)`
3. (Optional) Preprocess  
   - bandpass filtering + notch filtering  
   - rereference (if enabled)
4. QC + bad-channel handling  
   - detect NaNs, flat channels, clipped channels, noisy channels  
   - mark bad channels and optionally interpolate
5. EDA outputs (per-window)
   - raw before/after plots  
   - QC JSON  
   - PSD / bandpower / FFT / spectrograms (based on config)
6. Dataset overview outputs
   - `windows.csv`, `recordings.csv`, `summary.json` + charts
7. (Optional) Export cleaned windows to **BIDS derivatives** structure (EDF + sidecars)

### Outputs (what gets generated)
Outputs are written under `results/preprocess/` (paths configurable):

**EDA outputs**
- `results/preprocess/eda/<window_id>/window/`
  - `qc.json`
  - `raw_before.png`, `raw_after.png`
  - `psd_mean_uV2_per_hz.csv`
  - `bandpower_uV2.csv`
  - spectrogram plots (PSD/Morlet) if enabled
  - FFT plot/CSV if enabled

**Dataset overview**
- `results/preprocess/overview/`
  - `windows.csv` — per-window metadata + QC summary + artifact pointers
  - `recordings.csv` — recording-level info (duration, sfreq, channels)
  - `summary.json` — dataset summary (counts + QC aggregates)
  - `charts/` — label counts and windows-per-subject charts

**Diagrams**
- `results/preprocess/diagrams/`
  - `eda.mmd`
  - `modules.mmd`

**(Optional) Export cleaned windows (BIDS derivatives)**
- `results/preprocess/bids_dataset/`
  - `sub-*/eeg/*.edf`
  - `*_eeg.json`, `*_channels.tsv`, `*_events.tsv`, etc.

---

## 2) Install dependencies

```bash
pip install -r requirements.txt
````

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
  columns:
    path: "path"
    subject_id: "subject_id"
    start_sec: "start_sec"
    end_sec: "end_sec"
    label: "label"
    age: "age"
    sex: "sex"
```

### Outputs

```yaml
outputs:
  eda_root: "results/preprocess/eda"
  diagrams_root: "results/preprocess/diagrams"
  overview_root: "results/preprocess/overview"
```

### Preprocessing

```yaml
preprocess:
  filtering:
    enabled: true
    l_freq: 1.0
    h_freq: 60.0
    notch_freqs: [60.0]
```

### QC + bad channels

```yaml
analysis:
  time_domain:
    qc:
      enabled: true
      max_abs_uV: 500.0
      flat_var_thresh_uV2: 1.0e-12
      nan_allowed: false
      noisy_var_factor: 10.0

    bad_channels:
      enabled: true
      use_qc_rules: true
      interpolate: false
```

### Frequency-domain EDA

```yaml
analysis:
  frequency_domain:
    psd:
      enabled: true
    bandpower:
      enabled: true
    spectrogram:
      enabled: true
    fft:
      enabled: true
    morlet:
      enabled: true
```

### Optional export to BIDS derivatives

```yaml
export_cleaned:
  enabled: true
  out_root: "results/preprocess/bids_dataset"
```

---

## Run the pipeline

Single entrypoint:

```bash
PYTHONPATH=src python3 src/eeg_pipeline/main.py --config src/eeg_pipeline/configs/config.yaml
```

---

## Key concepts (so results are interpretable)

### QC metrics and “bad channels”

* QC checks each window for:

  * NaNs
  * clipped amplitudes (`max_abs_uV`)
  * flat channels (`flat_var_thresh_uV2`)
  * noisy variance outliers (`noisy_var_factor`)
* Channels flagged by QC rules can be marked as bad (`raw.info["bads"]`).
* `qc_noisy_frac` / `noisy_channel_frac` is typically the fraction of channels marked bad in that window.

### Why per-window EDA is useful

Per-window artifacts help verify:

* the filter is working (PSD/FFT changes)
* seizure vs non-seizure spectral differences (bandpower/spectrograms)
* QC and bad-channel decisions are reasonable
* the pipeline is not silently learning artifacts

---

## Project structure

```
src/eeg_pipeline/
  analysis/
    time_domain.py
    freq_domain.py
  configs/
    config.yaml
  core/
    artifacts.py
    bids_derivatives.py
    bids_io.py
    yaml_utils.py
  pipeline/
    bot_diagrams.py
    dataset_overview.py
    eda_engine.py
    filtering.py
    preprocessor.py
    run_pipeline.py
    verify.py
  main.py
```


