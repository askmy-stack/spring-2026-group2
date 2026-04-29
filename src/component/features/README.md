# EEG Seizure Detection — Feature Engineering

Window-level feature extraction module for EEG seizure detection research.

This module converts pre-generated window index CSV files into structured feature matrices suitable for classical ML models (LightGBM, XGBoost, Random Forest) and TabNet variants.

Each output CSV row represents one EEG window and contains:

- Per-channel engineered features  
- Optional connectivity features  
- Binary label (0 = background, 1 = seizure)  
- Window metadata  

---

## Entry Point

Run from the **project root**:

```bash
# Train split
python -m src.component.features.run_features_from_index \
    --config src/config/feature_engineering.yaml \
    --split train --n-jobs 4

# Val split
python -m src.component.features.run_features_from_index \
    --config src/config/feature_engineering.yaml \
    --split val --n-jobs 2

# Test split
python -m src.component.features.run_features_from_index \
    --config src/config/feature_engineering.yaml \
    --split test --n-jobs 2
```

Optional flags:
- `--n-jobs` — parallel workers (default: 2)
- `--chunk-size` — rows per checkpoint file (default: 20000)
- `--merge-only` — skip extraction, just merge existing chunks

---

## Pipeline Overview

This module assumes the **data loader** has already generated window index CSV files.

Expected structure:

```
results/dataloader/
├── window_index_train.csv
├── window_index_val.csv
└── window_index_test.csv
```

The feature extraction workflow:

1. Load window index CSV
2. Group windows by EDF file
3. Load each EDF segment with MNE
4. Extract features using `AdvancedFeatureExtractor` (with cache per unique window)
5. Checkpoint every N rows for resume support
6. Merge all chunks into final feature CSV

Output:

```
results/features_raw/
├── features_train.csv
├── features_val.csv
└── features_test.csv
```

---

## Configuration (`src/config/feature_engineering.yaml`)

All paths are relative to the config file and resolved automatically.

```yaml
window_index:
  train_csv: ../../results/dataloader/window_index_train.csv
  val_csv:   ../../results/dataloader/window_index_val.csv
  test_csv:  ../../results/dataloader/window_index_test.csv

io:
  output_dir: ../../results/features_raw
```

---

## Feature Domains

| Domain | Scope | Includes |
|--------|-------|----------|
| **Time-Domain** | Per-channel | Mean, Std, RMS, Line Length, ZCR, Skew, Kurtosis, Hjorth parameters |
| **Frequency-Domain** | Per-channel | Welch PSD, Band Power (δ θ α β γ), Relative Power, Spectral Entropy, Peak Frequency |
| **Time-Frequency & Nonlinear** | Per-channel | Wavelet Energy, Wavelet Entropy, Sample Entropy, Permutation Entropy, Lempel-Ziv |
| **Cross-Channel** | Window-level | Pearson correlation matrix, coherence |

---

## Feature Count

| Category | Features | Per channel |
|----------|---------|------------|
| Time-domain | mean, std, rms, min, max, range, line_length, zcr, skew, kurtosis | 10 |
| Hjorth | hjorth_activity, hjorth_mobility, hjorth_complexity | 3 |
| Nonlinear | sampen, perm_entropy | 2 |
| Frequency | total_power, delta/theta/alpha/beta/gamma power + relative, spec_entropy | 12 |
| FFT | fft_dom_freq | 1 |
| Wavelet | wav_E_A, wav_E_D1, wav_E_D2, wav_E_D3, wav_entropy | 5 |
| **Total per channel** | | **33** |
| **Total (× 16 channels)** | | **528** |

---

## Reference Papers

1. EEG Signal Processing for Medical Diagnosis — IEEE: https://ieeexplore.ieee.org/document/10353995  
2. EEG Feature Extraction Review — Frontiers in AI: https://www.frontiersin.org/articles/10.3389/frai.2022.1072801  
3. EEG Signal Processing and Feature Extraction — IJMTST: https://www.researchgate.net/publication/374337940

---

## Time-Domain Features

| Method | Why Used | Formula | Source |
|--------|----------|---------|--------|
| Mean | Detect baseline shift | μ = (1/N) Σ xᵢ | A |
| Std | Amplitude variability | √[(1/N) Σ(xᵢ − μ)²] | A |
| RMS | Signal energy magnitude | √[(1/N) Σ xᵢ²] | A |
| Min | Minimum amplitude | min(x) | A |
| Max | Maximum amplitude | max(x) | A |
| Range | Amplitude spread | max(x) − min(x) | A |
| Line Length | Rapid transitions | Σ |xᵢ − xᵢ₋₁| | C |
| ZCR | Oscillatory activity | (1/N) Σ I(sign change) | A, C |
| Skew | Asymmetric spike detection | E[(x−μ)³]/σ³ | A |
| Kurtosis | Heavy-tailed bursts | E[(x−μ)⁴]/σ⁴ | A |

## Hjorth Parameters

| Method | Purpose | Formula | Source |
|--------|---------|---------|--------|
| Activity | Signal power | Var(x) | A, C |
| Mobility | Frequency proxy | √[Var(dx)/Var(x)] | A, C |
| Complexity | Irregularity | Mobility(dx)/Mobility(x) | A, C |

## Nonlinear Features

| Method | Purpose | Source |
|--------|---------|--------|
| Sample Entropy | Irregularity detection | A |
| Permutation Entropy | Temporal complexity | A |

## Frequency-Domain (Welch PSD)

| Method | Purpose | Source |
|--------|---------|--------|
| Total Power | Overall spectral energy | B |
| Band Power (δ θ α β γ) | Frequency redistribution across 5 bands | B, C |
| Relative Power (δ θ α β γ) | Normalized band energy (band / total) | C |
| Spectral Entropy | Spectral concentration | C |
| FFT Dominant Frequency | Frequency with highest FFT magnitude | A, B |

## Wavelet Features

| Method | Purpose | Source |
|--------|---------|--------|
| Approx Energy (wav_E_A) | Low-frequency seizure energy | A, C |
| Detail Energy D1 (wav_E_D1) | High-frequency transient bursts | A, C |
| Detail Energy D2 (wav_E_D2) | Mid-high frequency activity | A, C |
| Detail Energy D3 (wav_E_D3) | Mid-frequency transient detection | A, C |
| Wavelet Entropy (wav_entropy) | Multi-scale signal dispersion | C |

## Connectivity Features

| Method | Purpose | Source |
|--------|---------|--------|
| Pearson Correlation | Hypersynchronization | B |
| Corr Mean / Std / Max / Min | Network summary stats | B |

---

## Requirements

See root `requirements.txt`. Key dependencies:

- Python >= 3.9  
- numpy >= 1.24  
- pandas >= 2.0  
- scipy >= 1.10  
- mne >= 1.5  
- PyWavelets >= 1.4  
- joblib >= 1.3  
