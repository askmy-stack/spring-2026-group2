# EEG Seizure Detection — Feature Engineering

Window-level feature extraction module for EEG seizure detection research.

This module converts pre-generated window index CSV files into structured feature matrices suitable for classical ML models (XGBoost, Random Forest, SVM, etc.).

Each output CSV row represents one EEG window and contains:

- Per-channel engineered features  
- Optional connectivity features  
- Binary label (0 = background, 1 = seizure)  
- Window metadata  

---

# Entry Point

```bash
python run_features_from_index.py --config configs/fe.yaml
```

Optional:

```bash
python run_features_from_index.py --config configs/fe.yaml --split train
```

---

# Pipeline Overview

This module assumes the **Data Pipeline** has already generated window index CSV files.

Example structure:

```
results/dataloader/
├── window_index_train.csv
├── window_index_val.csv
└── window_index_test.csv
```

The feature extraction workflow:

1. Load window index CSV
2. Load corresponding EEG segment from EDF
3. Resample if configured
4. Extract features using `AdvancedFeatureExtractor`
5. Attach label + metadata
6. Save split-level feature CSV

---

# Configuration (`configs/fe.yaml`)

All behavior is controlled through `fe.yaml`.

## Window Index Inputs

```yaml
window_index:
  train_csv: "results/dataloader/window_index_train.csv"
  val_csv:   "results/dataloader/window_index_val.csv"
  test_csv:  "results/dataloader/window_index_test.csv"
```

These paths can point anywhere. The module does not assume a fixed folder structure.

---

## Output

```yaml
io:
  output_dir: "results/features"
```

Generated files:

```
results/features/
├── features_train.csv
├── features_val.csv
└── features_test.csv
```

---

## Sampling Control

```yaml
fe:
  sfreq_mode: "auto"   # auto | force
  target_sfreq: 256
```

- `auto` → use native EDF sampling rate  
- `force` → resample to `target_sfreq`  

---

# Feature Domains

Engineered features are grouped into three primary domains:

| Domain | Scope | Includes |
|--------|-------|----------|
| **Time-Domain** | Per-channel + Window-level | Mean, Std, RMS, Line Length, ZCR, Skew, Kurtosis, Hjorth, Pearson correlation |
| **Frequency-Domain** | Per-channel | Welch PSD, Band Power (δ, θ, α, β, γ), Relative Power, Spectral Entropy, FFT Dominant Frequency |
| **Time–Frequency & Nonlinear** | Per-channel | Wavelet Energy, Wavelet Entropy, Sample Entropy, Permutation Entropy, Lempel–Ziv Complexity |

---

# Detailed Data Dictionary

### Reference Papers

1. **Paper A:**  
   EEG Signal Processing for Medical Diagnosis, Healthcare and Monitoring — A Comprehensive Review  
   🔗 https://ieeexplore.ieee.org/document/10353995  

2. **Paper B:**  
   Frontiers in Artificial Intelligence (2022) — EEG Feature Extraction Review  
   🔗 https://www.frontiersin.org/articles/10.3389/frai.2022.1072801  

3. **Paper C:**  
   EEG Signal Processing and Feature Extraction — IJMTST  
   🔗 https://www.researchgate.net/publication/374337940_EEG_Signal_Processing_and_Feature_Extraction  

---

## Time-Domain Features

| Method | Why Used | Formula | Source |
|--------|----------|---------|--------|
| Mean | Detect baseline shift | μ = (1/N) Σ xᵢ | A |
| Std | Measures amplitude variability | √[(1/N) Σ(xᵢ − μ)²] | A |
| RMS | Signal energy magnitude | √[(1/N) Σ xᵢ²] | A |
| Line Length | Captures rapid transitions | Σ |xᵢ − xᵢ₋₁| | C |
| ZCR | Oscillatory activity shifts | (1/N) Σ I(sign change) | A, C |
| Skew | Asymmetric spike detection | E[(x−μ)³]/σ³ | A |
| Kurtosis | Heavy-tailed bursts | E[(x−μ)⁴]/σ⁴ | A |

---

## Hjorth Parameters

| Method | Purpose | Formula | Source |
|--------|---------|---------|--------|
| Activity | Signal power | Var(x) | A, C |
| Mobility | Frequency proxy | √[Var(dx)/Var(x)] | A, C |
| Complexity | Irregularity | Mobility(dx)/Mobility(x) | A, C |

---

## Nonlinear Features

| Method | Purpose | Formula | Source |
|--------|---------|---------|--------|
| Sample Entropy | Irregularity detection | −log(A/B) | A |
| Permutation Entropy | Temporal complexity | −Σ p log(p) | A |
| Lempel–Ziv | Signal compressibility | New substring count | A |

---

## Frequency-Domain (Welch PSD)

| Method | Purpose | Formula | Source |
|--------|---------|---------|--------|
| Total Power | Overall spectral energy | ∫ PSD(f) df | B |
| Band Power | Frequency redistribution | ∫ PSD over band | B, C |
| Relative Power | Normalized band energy | Band / Total | C |
| Spectral Entropy | Spectral concentration | −Σ p log(p) | C |
| FFT Dominant Freq | Strongest oscillation | argmax |FFT(f)| | A, B |

---

## Wavelet Features

| Method | Purpose | Formula | Source |
|--------|---------|---------|--------|
| Approx Energy | Low-frequency seizure energy | Σ cA² | A, C |
| Detail Energy | Transient burst detection | Σ cD² | A, C |
| Wavelet Entropy | Multi-scale dispersion | −Σ p log(p) | C |

---

## Connectivity Features

| Method | Purpose | Formula | Source |
|--------|---------|---------|--------|
| Pearson Corr | Hypersynchronization | Cov(X,Y)/(σxσy) | B |
| Corr Mean/Std/Max/Min | Network summary | Stats of upper triangle | B |

---

# Feature Count

With default configuration:

Per-channel features:

- Time-domain: 10  
- Nonlinear: 3  
- Frequency-domain: 13  
- Wavelet: 6  

**Total per channel = 32 features**

Connectivity (if enabled):

- 4 features

### Final Formula

```
Total Features = (C × 32) + 4
```

Where `C` = number of EEG channels.

### Example (16 channels)

```
16 × 32 = 512
+ 4 connectivity
= 516 features per window
```

Metadata columns (split, label, timing, path, subject_id) are not included in model features.

---

# Requirements

- Python ≥ 3.9  
- numpy ≥ 1.24  
- pandas ≥ 2.0  
- scipy ≥ 1.10  
- mne ≥ 1.5  
- pywavelets ≥ 1.4  
- PyYAML ≥ 6.0  

---

# Run Command

```bash
python run_features_from_index.py --config configs/fe.yaml
```
