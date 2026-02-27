# EEG Seizure Detection – Feature Engineering (Window Index → Features CSV)

This module generates window-level EEG features from pre-built window index files.

Each output CSV row represents one EEG window and contains:
- Per-channel engineered features
- Optional connectivity features
- Label (0 = non-seizure, 1 = seizure)
- Metadata (window timing, file path, etc.)

---

# Project Structure

project/
├── run_features_from_index.py
├── feature_engineering.py
└── configs/
    └── fe.yaml

---

# Quick Start

1. Install Dependencies

pip install numpy pandas scipy mne PyYAML pywavelets

2. Configure fe.yaml

Edit:

configs/fe.yaml

Make sure window index CSV paths are correct:

window_index:
  train_csv: "results/dataloader/window_index_train.csv"
  val_csv:   "results/dataloader/window_index_val.csv"
  test_csv:  "results/dataloader/window_index_test.csv"

io:
  output_dir: "results/features"

3. Run Feature Extraction

python run_features_from_index.py --config configs/fe.yaml

Outputs will be saved inside:

results/features/

---

# What This Module Does

1. Reads window_index_train/val/test CSV files
2. Loads each EEG window
3. Applies AdvancedFeatureExtractor
4. Attaches label
5. Writes features CSV

This step assumes windowing and labeling are already completed.

---

# Data Dictionary

All per-channel features follow this naming pattern:

ch{channel_index}_{feature_name}

Example:
- ch0_mean
- ch1_alpha_power
- ch3_hjorth_mobility

Connectivity features are window-level:
- conn_corr_mean
- conn_corr_std
- conn_corr_max
- conn_corr_min

---

# Time-Domain Features (Per Channel)

chX_mean  
Mean amplitude of the window.

chX_std  
Standard deviation of amplitude.

chX_rms  
Root mean square value.

chX_line_length  
Sum of absolute consecutive differences:
LL = sum |x_t - x_{t-1}|

chX_zcr  
Zero Crossing Rate (fraction of sign changes).

chX_skew  
Skewness of amplitude distribution.

chX_kurtosis  
Kurtosis of amplitude distribution.

---

# Hjorth Parameters

Let:
dx = first difference
ddx = second difference

chX_hjorth_activity  
Variance of the signal.

chX_hjorth_mobility  
sqrt( Var(dx) / Var(x) )

Frequency proxy.

chX_hjorth_complexity  
Mobility(dx) / Mobility(x)

Waveform complexity measure.

---

# Nonlinear Features

chX_sampen  
Sample Entropy  
Parameters:
- m = 2
- r = 0.2 * std(x)

Measures irregularity.

chX_perm_entropy  
Permutation Entropy  
Parameters:
- order = 3
- delay = 1

Measures temporal pattern complexity.

chX_lz_complexity  
Lempel–Ziv complexity after quantile discretization (10 bins).

Measures sequence compressibility / novelty.

---

# Frequency-Domain Features (Welch PSD)

Welch Settings:
- nperseg = 2 seconds
- fmin = 0.5 Hz
- fmax = 50 Hz

chX_total_power  
Integrated PSD between fmin and fmax.

Band Powers:
- chX_delta_power (0.5–4 Hz)
- chX_theta_power (4–8 Hz)
- chX_alpha_power (8–13 Hz)
- chX_beta_power (13–30 Hz)
- chX_gamma_power (30–50 Hz)

Relative Powers:
- chX_delta_rel
- chX_theta_rel
- chX_alpha_rel
- chX_beta_rel
- chX_gamma_rel

Each relative power = band_power / total_power

chX_spec_entropy  
Entropy of normalized PSD.

chX_fft_dom_freq  
Dominant frequency (Hz) from FFT magnitude within 0.5–50 Hz.
Uses optional detrending and Hamming window.

---

# Wavelet Features (DWT)

Wavelet:
- db4
- Level = 4

chX_wav_E_A  
Approximation coefficient energy.

chX_wav_E_D1 ... chX_wav_E_D4  
Detail coefficient energies.

chX_wav_entropy  
Entropy of wavelet energy distribution.

---

# Connectivity Features (Window-Level)

Computed using Pearson correlation across channels.

conn_corr_mean  
Mean pairwise correlation.

conn_corr_std  
Standard deviation of correlations.

conn_corr_max  
Maximum correlation.

conn_corr_min  
Minimum correlation.

---

# Output CSV

Each row represents one EEG window.

Columns include:
- Engineered features
- label (0/1)
- start_sec
- end_sec
- recording metadata

---

# Run Command

python run_features_from_index.py --config configs/fe.yaml
