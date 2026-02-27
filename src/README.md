```md
# EEG Seizure Detection — Feature Engineering

Window-level feature extraction module for EEG seizure detection research.

This module converts pre-generated window index CSV files into structured feature matrices for classical machine learning models.

Each output CSV row represents one EEG window and includes:

- Per-channel signal features  
- Multi-channel connectivity features  
- Binary label (0 = background, 1 = seizure)  
- Window metadata  

---

## Entry Point

```

python run_features_from_index.py --config configs/fe.yaml

```

---

## Pipeline Overview

This module assumes the Data Pipeline has already generated the csv files for train, test and val....for example:

```

results/dataloader/
├── window_index_train.csv
├── window_index_val.csv
└── window_index_test.csv

````

Feature extraction flow:

1. Load window index CSV
2. Load EEG window segment
3. Resample if configured
4. Apply `AdvancedFeatureExtractor`
5. Attach metadata + label
6. Save split-level feature CSV

---

## Configuration (configs/fe.yaml)

### Window Index Inputs

for exmaple:

```yaml
window_index:
  train_csv: "results/dataloader/window_index_train.csv"
  val_csv:   "results/dataloader/window_index_val.csv"
  test_csv:  "results/dataloader/window_index_test.csv"
````

### Output

```yaml
io:
  output_dir: "results/features"
```

### Sampling

```yaml
fe:
  sfreq_mode: "auto"     # auto | force
  target_sfreq: 256
```

* `auto` → use native sampling frequency
* `force` → resample to target_sfreq

---

# Extracted Feature Domains

The engineered features are organized into three primary domains:

| Domain | Type | Includes |
|--------|------|----------|
| **Time-Domain Features** | Per-channel + Window-level | Mean, Std, RMS, Line Length, ZCR, Skew, Kurtosis, Hjorth parameters, Pearson correlation |
| **Frequency-Domain Features** | Per-channel | Welch PSD, Band Power (δ, θ, α, β, γ), Relative Power, Spectral Entropy, FFT Dominant Frequency |
| **Time–Frequency & Nonlinear Features** | Per-channel | Wavelet Energy, Wavelet Entropy, Sample Entropy, Permutation Entropy, Lempel–Ziv Complexity |

---

## Domain Explanation

### 1. Time-Domain

These features operate directly on the signal amplitude over time.

They capture:
- Amplitude variability
- Waveform shape
- Rapid changes
- Channel-to-channel synchrony

Includes:
- Statistical moments
- Line length
- Hjorth parameters
- Pearson correlation summaries

---

### 2. Frequency-Domain

These features analyze the signal in the frequency spectrum using Fourier-based methods.

They capture:
- Energy distribution across frequency bands
- Rhythmic activity
- Dominant oscillatory behavior

Includes:
- Welch PSD
- Band power
- Relative power
- Spectral entropy
- FFT dominant frequency

---

### 3. Time–Frequency & Nonlinear

These features capture non-stationary and complex behavior.

They capture:
- Transient bursts
- Multi-scale patterns
- Signal unpredictability
- Structural complexity

Includes:
- Wavelet decomposition (multi-resolution)
- Entropy-based measures
- Complexity measures

---

---

# Detailed Data Dictionary

The table below documents:

* Method name
* Why it is used in seizure detection
* Mathematical formulation
* Source reference (from reviewed papers)

### Reference Papers

* **Paper A:** EEG Signal Processing for Medical Diagnosis — Comprehensive Review
* **Paper B:** Frontiers in Artificial Intelligence (2022) — EEG Feature Extraction Review
* **Paper C:** EEG Signal Processing and Feature Extraction — IJMTST

---

## Time-Domain Features

| Method             | Why We Use It                                    | Mathematical Definition                | Source     | 
| ------------------ | ------------------------------------------------ | -------------------------------------- | ---------- | 
| Mean               | Detect baseline drift                            | μ = (1/N) Σ xᵢ                         | Paper A    | 
| Standard Deviation | Seizures increase amplitude variance             | σ = √[(1/N) Σ(xᵢ − μ)²]                | Paper A    |
| RMS                | Measures signal energy magnitude                 | RMS = √[(1/N) Σ xᵢ²]                   | Paper A    |
| Line Length        | Captures waveform complexity & rapid transitions | LL = Σ xᵢ − xᵢ₋₁                       | Paper C    |
| Zero Crossing Rate | Detects oscillatory pattern shifts               | ZCR = (1/N) Σ I(sign(xᵢ) ≠ sign(xᵢ₋₁)) | Paper A, C |
| Skewness           | Detects asymmetric spike activity                | E[(x−μ)³] / σ³                         | Paper A    |
| Kurtosis           | Detects heavy-tailed spike bursts                | E[(x−μ)⁴] / σ⁴                         | Paper A    |   

---

## Hjorth Parameters

| Method            | Why We Use It                  | Mathematical Definition    | Source     |
| ----------------- | ------------------------------ | -------------------------- | ---------- |
| Hjorth Activity   | Measures signal power          | Var(x)                     | Paper A, C |
| Hjorth Mobility   | Proxy for dominant frequency   | √[Var(dx) / Var(x)]        | Paper A, C |
| Hjorth Complexity | Measures waveform irregularity | Mobility(dx) / Mobility(x) | Paper A, C |

---

## Nonlinear Features

| Method                | Why We Use It                                 | Mathematical Definition | Source  |
| --------------------- | --------------------------------------------- | ----------------------- | ------- |
| Sample Entropy        | Detects irregular transitions during seizures | SampEn = −log(A/B)      | Paper A |
| Permutation Entropy   | Measures temporal pattern complexity          | PE = −Σ p log(p)        | Paper A |
| Lempel–Ziv Complexity | Measures compressibility / novelty            | Count of new substrings | Paper A |

---

## Frequency-Domain (Welch PSD)

| Method                     | Why We Use It                               | Mathematical Definition | Source     |
| -------------------------- | ------------------------------------------- | ----------------------- | ---------- |
| Total Power                | Measures overall spectral energy            | ∫ PSD(f) df             | Paper B    |
| Band Power (δ, θ, α, β, γ) | Identifies seizure-related frequency shifts | ∫ PSD(f) over band      | Paper B, C |
| Relative Power             | Normalizes band energy distribution         | BandPower / TotalPower  | Paper C    |
| Spectral Entropy           | Measures spectral concentration vs spread   | −Σ p log(p)             | Paper C    |
| FFT Dominant Frequency     | Detects strongest oscillatory rhythm        | argmax FFT(f)           | Paper A, B |

---

## Wavelet Features (DWT)

| Method                       | Why We Use It                          | Mathematical Definition | Source     |
| ---------------------------- | -------------------------------------- | ----------------------- | ---------- |
| Wavelet Approximation Energy | Captures low-frequency seizure energy  | Σ cA²                   | Paper A, C |
| Wavelet Detail Energy        | Captures transient bursts              | Σ cD²                   | Paper A, C |
| Wavelet Entropy              | Measures multi-scale energy dispersion | −Σ p log(p)             | Paper C    |

---

## Connectivity Features

| Method                      | Why We Use It                     | Mathematical Definition      | Source  |
| --------------------------- | --------------------------------- | ---------------------------- | ------- |
| Pearson Correlation         | Detects hypersynchronization      | Cov(X,Y)/(σxσy)              | Paper B |
| Corr Mean / Std / Max / Min | Summarizes network-level coupling | Statistics of upper triangle | Paper B |


---

# Output Structure

```
results/features/
├── features_train.csv
├── features_val.csv
└── features_test.csv
```

Each row contains:

* Engineered features
* Label (0/1)
* Window timing
* Recording metadata

---

# Requirements

* Python ≥ 3.9
* numpy ≥ 1.24
* pandas ≥ 2.0
* scipy ≥ 1.10
* mne ≥ 1.5
* pywavelets ≥ 1.4
* PyYAML ≥ 6.0

---

# Run Command

```
python run_features_from_index.py --config configs/fe.yaml
```
