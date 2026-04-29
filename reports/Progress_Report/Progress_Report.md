# Weekly Progress Report
## EEG-Based Seizure Detection | spring-2026-group2
**Advisor:** Dr. Amir Jafari | **Clinical Collaborator:** Dr. Mohamad Koubeissi (GWU Neurology)
**Repo:** `spring-2026-group2` | **Branch:** `abhinaysai-lstm`

---

## Week 1 — January 13, 2026 (Tuesday Meeting)

**Topics Discussed**
- Reviewed capstone proposal options and selected EEG-based seizure detection
- Introduced project scope: seizure vs. non-seizure binary classification + onset detection
- Initial discussion on EEG signal properties and clinical context

**Decisions Made**
- Confirmed proposal: EEG seizure detection on CHB-MIT dataset
- Agreed on GitHub repo creation as immediate priority

**Action Items**
- [x] Create GitHub repo with correct folder structure
- [x] Begin exploring available EEG datasets
- [x] Look up foundational resources on EEG signal processing
- [x] Reach out to Dr. Mohamad Koubeissi for clinical guidance

---

## Week 2 — January 20, 2026 (Tuesday Meeting)

**Topics Discussed**
- Confirmed proposal and finalized project direction
- Discussed project outline and research scope
- Explored CHB-MIT, Siena Scalp EEG, and other publicly available datasets
- Introduced concept of data fusion across datasets

**Key Understandings from Meeting**
- Seizures manifest as increased frequency or amplitude — line length is a useful proxy feature
- Different frequency bands correspond to different seizure types
- Channel synchronization across electrodes is a diagnostic signal
- Persyst is the clinical EEG analysis tool used in practice — a relevant benchmark
- Model perspective: 16-channel input with potential for pretraining on raw EEG signals
- Contact: Dr. Mohamad Koubeissi (216-280-933) for clinical validation guidance

**Action Items**
- [x] Create GitHub repo with correct structure
- [x] Perform simple EDA on CHB-MIT dataset
- [x] Review literature on EEG modeling and signal processing
- [x] Reach out to Dr. Koubeissi

---

## Week 3 — January 27, 2026 (Tuesday Meeting)

**Topics Discussed**
- Presented initial EDA and dataset understanding
- Discussed CHB-MIT structure: 23 subjects, 256 Hz, EDF format, 198 seizure events
- Planned research and literature review scope
- Introduced initial modeling strategy

**Key Technical Decisions**
- Primary dataset: CHB-MIT Scalp EEG (PhysioNet)
- Secondary dataset: Siena Scalp EEG (pending preprocessing)
- Sampling rate target: 256 Hz (interpolate if lower, downsample if higher)
- Windowing strategy under discussion: 1-second vs. 4-second windows

**Action Items**
- [x] Write weekly progress report `.md` file starting from Jan 13
- [x] Begin literature review on time-frequency features for EEG
- [x] Set up data loader skeleton for EDF format

---

## Week 4 — February 3, 2026 (Tuesday Meeting)

**Topics Discussed**
- Established benchmark strategy and model roadmap
- Signal processing pipeline discussion: Fourier transforms, CNN-based models, autoencoder pretraining
- Dataloader requirements defined — must be universal for `.csv`, `.edf` formats
- EDA of signal processing required before end of week

**Benchmark Requirements**
- 16-channel standard montage as input
- EDA of signal processing pipeline
- Pretraining autoencoder for self-supervised representation learning
- CNN-based model as first deep learning baseline
- Fourier transform features for frequency-domain analysis
- Generalized dataloader supporting multiple formats

**Action Items**
- [x] Define action items for dataloader generalization
- [x] Begin EDA on signal processing pipeline
- [ ] Implement Fourier transform feature extraction
- [x] Create flowchart for end-to-end pipeline
- [x] Feature engineering plan drafted

---

## Week 5 — February 10, 2026 (Tuesday Meeting)

**Topics Discussed**
- Dataloader design refinements from team review
- Age and gender metadata integration
- Discussed dataset splitting strategy

**Dataloader Design Requirements Finalized**
- Config file generation after pipeline run (no hardcoding)
- Include age and gender fields; downsample to matched distributions
- Downsampling logic: >256 Hz → downsample; <256 Hz → interpolate; generate synthetic augmentation (5 methods per patient)
- Convert all formats to EDF / EDF+ compatible representation
- Convert to BIDS format as intermediate standard
- Electrode ordering: ascending by 10-20 system position
- Train / Val / Test split: **70 / 15 / 15 — subject-independent**
- Windowing: 4-second windows, 2-second overlap (50%) — also consider 1-second for label granularity
- Proportional stratified sampling after splitting

**Action Items**
- [x] Begin implementing config file in `config.yaml`
- [x] Implement subject-independent 70/15/15 split
- [x] Write test scripts for config and dataloader validation
- [x] Add age-matching logic from Arjun's input

---

## Week 6 — February 16, 2026 (Tuesday Meeting + Sunday Review)

**Topics Discussed**
- Caching strategy for dataloader using PyTorch Dataset
- Data serialization format decision
- Pipeline script naming and structure

**Technical Decisions**
- Cache input data using `pickle` format via `torch.data` caching
  - Reference: [torchdatasets cachers](https://szymonmaszke.github.io/torchdatasets/packages/torchdata.cachers.html)
- If cache exists → load from cache; if not → run pipeline and save
- Output formats: `.csv`, `.tsv`, `.parquet`, PyTorch tensors (`.pt`)
- JSON sidecar with age and sex metadata per recording
- Output should be BIDS-compliant and tensor-ready simultaneously
- Labeling: 1-second chunks with per-window binary label
- Channel location mapping: nearest-neighbor interpolation for missing electrodes
- `Pipeline.py` renamed for clarity
- `main_loader.py` → single entry point with argument flags for tensor and CSV generation

**Action Items**
- [x] Implement pickle-based caching in dataloader
- [x] Write download script for CHB-MIT from PhysioNet
- [x] Test full pipeline: dataloader → EDA → feature engineering
- [x] Write wrapper for event `.tsv` files
- [x] EEG channel names sorted in ascending order for EDA plots

---

## Week 7 — February 23, 2026 (Tuesday Meeting)

**Topics Discussed**
- Dataloader bug review — read code line by line
- Papers on time and frequency features for EEG reviewed
- Simple sync script connecting all pipeline modules

**Research Focus**
- Literature review: time-domain features (line length, amplitude, variance, energy)
- Literature review: frequency-domain features (band power, spectral entropy, peak frequency)
- Running PCA for feature selection and dimensionality reduction

**Action Items**
- [x] Read and debug dataloader code line by line
- [x] Review EEG modeling papers
- [x] Write sync script: `dataloader → EDA → feature engineering`
- [x] Implement wrapper code for `.tsv` event files
- [x] Ensure EEG channel names appear in ascending order in all EDA plots

---

## Week 8 — March 2, 2026 (Tuesday Meeting)

**Topics Discussed**
- Validated dataloader output: tensors, labels, correct source data
- Baseline model architecture confirmed: LSTM classifier
- Code structure review: `src/` folder reorganization

**Folder Structure Finalized**
```
src/
├── data_loader/       # main_loader.py + loaders per dataset
├── models/            # LSTM, CNN architectures
├── utils/             # metrics, callbacks, config
├── results/           # tensors, logs, outputs
```

**Model Pipeline Confirmed**
- `main_loader.py` → outputs PyTorch DataLoader
- First model: LSTM with classifier head
- Second model: CNN baseline
- Metrics module: F1, AUC-ROC, Sensitivity, Specificity

**Action Items**
- [x] Refactor `src/` into two dataloader structure
- [x] Implement `main_loader` with tensor and CSV output flags
- [x] Implement baseline LSTM classifier
- [x] Write test scripts to validate label correctness end-to-end
- [x] Add wrapper for additional dataset support

---

## Week 9 — March 10, 2026 (Tuesday Meeting)

**Topics Discussed**
- Feature engineering implementation: 528 features (33 per channel × 16 channels)
- Feature groups: time-domain, Hjorth, nonlinear, spectral (Welch), wavelet (db4 L3)
- Classical ML baselines with Optuna hyperparameter tuning

**Feature Engineering Completed**
| Group | Features | Count/Channel |
|-------|----------|---------------|
| Time-Domain | mean, std, RMS, min, max, range, line length, ZCR, skewness, kurtosis | 10 |
| Hjorth | activity, mobility, complexity | 3 |
| Nonlinear | sample entropy, permutation entropy | 2 |
| Spectral | 5 band powers + 5 relative + total + spectral entropy + FFT dominant freq | 13 |
| Wavelet (db4 L3) | energy A, D1, D2, D3 + wavelet entropy | 5 |
| **Total** | | **33 × 16 = 528** |

**Action Items**
- [x] Implement full feature extraction pipeline
- [x] Crash-safe checkpointing every 20,000 rows
- [x] Train Random Forest, XGBoost, LightGBM baselines
- [x] Run Optuna hyperparameter tuning on classical ML models

---

## Week 10 — March 17, 2026 (Tuesday Meeting)

**Topics Discussed**
- Data labeling issue identified in dataloader — fixed
- Siena dataset channel mapping created
- Introduced Transformer JEPA for time series as future direction
- Encoder-decoder architecture discussion for pretraining

**Key Bug Fix**
- Siena channel name parsing bug: `EEG Fp1-Ref` → parsed incorrectly as `EEG FP1` instead of `FP1`
- All Siena files were being skipped silently — fix committed

**Modeling Roadmap Confirmed**
- Pretrained models from HuggingFace: encoder layer fine-tuning
- Transformer-based models as next family after LSTM
- Ensemble layer: soft voting over all model families for web UI

**Action Items**
- [x] Fix Siena channel matching bug
- [x] Create PyTorch DataLoader for Siena dataset
- [x] Begin fine-tuning pretrained HuggingFace EEG transformer (ST-EEGFormer)
- [x] Encoder-decoder pretraining experiment

---

## Week 11 — March 24, 2026 (Tuesday Meeting)

**Topics Discussed**
- Log analysis and debugging of training runs
- Time-series gap handling in windowing
- Pretrained network integration from HuggingFace

**Training Infrastructure**
- HuggingFace encoder → freeze backbone → fine-tune classifier head only
- ST-EEGFormer adapted: zero-padding to 256 samples + non-strict checkpoint loading
- BENDR, BIOT: runtime resampling 256 Hz → 200 Hz

**Action Items**
- [x] Integrate ST-EEGFormer from HuggingFace
- [x] Fix checkpoint loading bug (`torch.load` without `model_state_dict` unwrap)
- [x] Run full benchmark: all model families
- [x] Build two-tier ensemble layer

---

## Week 12 — Week of April 7, 2026

**Completed**
- Baseline benchmark across all 7 LSTM variants
- CNN baselines: Enhanced CNN 1D, Multiscale Attention CNN
- Pretrained models: ST-EEGFormer, BENDR, BIOT, EEGPT
- EEGMamba (state-space model) — training complete, calibration issue flagged
- Two-tier meta-ensemble structure implemented

**Critical Bug Fixes**
| Bug | Commit | Impact |
|-----|--------|--------|
| `pos_weight` inverted | `f6460ac` | All pre-Apr 14 metrics biased low |
| Double `pos_weight` application | `0026241` | Over-penalized minority class |
| Checkpoint loading failure | `f57840d` | All 14 model evaluations broken |
| Mamba `squeeze()` scalar crash | `ff4011e` | Tail-batch crash on Mamba training |
| `NameError` in `callbacks.py` | `b7b1117` | EarlyStopping instantiation failure |

---

## Week 13 — Week of April 14, 2026

**Model Results Summary**

| Model | F1 | AUC-ROC | Sensitivity |
|-------|----|---------|-------------|
| Meta-Ensemble Tier 2 | ~0.60 | ~0.800 | ~0.85 |
| ST-EEGFormer | 0.572 | 0.747 | 0.744 |
| LSTM Ensemble Tier 1 | ~0.565 | ~0.745 | ~0.80 |
| FeatureBiLSTM | 0.520 | 0.704 | 0.766 |
| CNN-LSTM | 0.518 | 0.712 | 0.569 |
| Enhanced CNN 1D | 0.504 | 0.992 | 0.446 |
| EEGMamba | 0.462 | 0.456 | ~0.68 |
| TabNet | ~0.220 | ~0.850 | — |

**Key Finding:** Threshold tuning alone contributed +0.21 F1 — the single largest gain in the project.

---

## Week 14 — Week of April 21, 2026

**Completed**
- [x] Full benchmark table finalized with all metrics
- [x] Publication-quality figures generated (ROC, PR curves, confusion matrices)
- [x] Springer Nature manuscript drafted (43 pages)
- [x] Comprehensive modeling report generated (PDF)
- [x] Speaker notes and Q&A bank prepared for final presentation

**Pending**
- [x] Generate `subject_ids.pt` for honest subject-wise cross-validation
- [x] Temperature scaling for EEGMamba calibration
- [x] Brier score and calibration plots
- [x] External validation on Siena dataset (blocked by channel matching fix)

---

## Summary of Key Milestones

| Milestone | Status | Week |
|-----------|--------|------|
| GitHub repo created | ✅ | Week 2 |
| CHB-MIT dataloader complete | ✅ | Week 6 |
| Feature engineering (528 features) | ✅ | Week 9 |
| Classical ML baseline | ✅ | Week 10 |
| LSTM family benchmark (7 variants) | ✅ | Week 12 |
| CNN baseline benchmark | ✅ | Week 12 |
| ST-EEGFormer pretrained integration | ✅ | Week 11 |
| EEGMamba training | ✅ | Week 12 |
| Two-tier meta-ensemble | ✅ | Week 13 |
| Final benchmark table | ✅ | Week 14 |
| Manuscript draft | ✅ | Week 14 |
| Subject-wise CV (subject_ids.pt) | ✅ | Week 12|
| Siena dataset integration | ✅ | Week 12|

---

*Last updated: April 28, 2026 | Branch: `abhinaysai-lstm` | Commit: `60b84ec`*
