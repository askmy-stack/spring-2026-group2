# CHB01 Run-01 EDA Report

## Scope

This report explains the EDA outputs for:

- `sub-chb01_sub-chb01_ses-001_task-eeg_run-01_eeg`

and connects them to the seizure-detection code path in this repository.

The analysis is based on:

- the saved EDA artifacts under `src/results/preprocess/eda/sub-chb01_sub-chb01_ses-001_task-eeg_run-01_eeg/recording/`
- the preprocessing and EDA code in `src/eeg_pipeline`
- the labeling, tensorization, and model-training code in `src/dataloader` and `src/model`

## Short answer

Yes, the PNG and CSV outputs are interpretable.

For this specific recording, the strongest evidence is that it is more useful as a **background / reference EEG example** than as a seizure-positive example:

- there is no `seizure_vs_nonseizure.csv` for this run
- no matching `events.tsv` was found locally for `run-01`
- the current recording-level pipeline passes `seizure_intervals=None` into EDA
- the CHB-MIT summary parsing logic used elsewhere in the repo treats `chb01_01.edf` as a zero-seizure file in tests

So this run is still relevant to seizure detection, but mainly as a **negative-class recording** and as a way to inspect artifact burden, spectral structure, and preprocessing behavior.

## What the run-01 artifacts show

### 1. Signal quality and preprocessing behavior

From `qc.json`:

- sampling rate: `256 Hz`
- EEG channels analyzed: `19`
- median channel variance: `585.10 µV²`
- variance range: `216.39` to `2727.30 µV²`
- max absolute amplitude range: `136.33` to `747.76 µV`
- clipped channels flagged by QC: `10`
- flat channels: `0`
- noisy channels by variance rule: `0`
- mean kurtosis: `14.08`

Interpretation:

- The recording is not flat or missing, so it has usable physiological activity.
- It contains several high-amplitude excursions above the QC clipping threshold of `500 µV`.
- High kurtosis suggests heavy-tailed transients or artifacts, not a clean stationary rhythm.

Visually, `raw_before.png` and `raw_after.png` show:

- strong low-frequency, high-amplitude activity in several bipolar channels
- preprocessing reduces some visible contamination, but substantial large-amplitude structure remains
- several channels appear de-emphasized after QC/bad-channel marking, while the main retained channels still show structured rhythmic activity

### 2. Frequency-domain findings

From `bandpower_uV2.csv`:

- delta: `451.40 µV²`
- theta: `119.93 µV²`
- alpha: `35.99 µV²`
- beta: `125.01 µV²`
- gamma: `123.66 µV²`

Interpretation:

- Delta power is dominant by a large margin.
- Alpha is comparatively weak.
- Beta and gamma are not absent, but they are much smaller than the low-frequency component.

From `psd_mean_uV2_per_hz.csv`, the strongest mean PSD bins are around:

- `1.5 Hz`: `252.65 µV²/Hz`
- `1.375 Hz`: `244.56 µV²/Hz`
- `1.625 Hz`: `244.00 µV²/Hz`
- `1.75 Hz`: `218.80 µV²/Hz`
- `1.25 Hz`: `210.60 µV²/Hz`

Interpretation:

- The recording is dominated by very low-frequency energy around `1 to 2.5 Hz`.
- That pattern is consistent with slow-wave activity and/or low-frequency artifact contamination.

From `fft_FP1-F7_t0.0-2.0.csv`, the first analyzed segment of channel `FP1-F7` is dominated by:

- `2.5 Hz`: power `40.01 µV²`
- `2.0 Hz`: power `38.56 µV²`
- `5.0 Hz`: power `17.95 µV²`

Interpretation:

- The opening seconds of `FP1-F7` are also low-frequency dominated.
- There is no isolated narrow seizure frequency peak that by itself would label this segment as ictal.

### 3. Spectrogram findings

From `psd_spectrogram_db.png`:

- low-frequency power remains elevated through most of the recording
- stable horizontal bands are visible near roughly `16 Hz`, `32 Hz`, and `48 Hz`
- there is no single abrupt onset/offset block that stands out as a clean seizure interval in this artifact alone

Interpretation:

- Persistent low-frequency dominance suggests background structure plus artifact burden.
- The narrow horizontal lines near `16/32/48 Hz` look more like sustained rhythmic or harmonic structure than a transient seizure onset.
- This recording is informative for understanding what the model sees in non-seizure EEG that is still complex and noisy.

From `morlet_spectrogram_db.png`:

- power is strongest at low frequencies
- energy gradually falls as frequency increases
- vertical streaking is present, indicating repeated transient bursts across time

## Why this recording still matters for seizure detection

Seizure detection is not only about learning what seizures look like. It also requires learning what **non-seizure EEG with artifacts and slow activity** looks like, because false positives often come from:

- movement artifacts
- electrode pops
- rhythmic background slowing
- high-amplitude non-ictal bursts

This run is useful because it gives the model a hard negative example:

- it has strong low-frequency content
- it has high-amplitude excursions
- it triggers QC clipping flags
- but it does not currently carry seizure intervals in the saved EDA flow

That is exactly the kind of example that determines whether a seizure detector becomes clinically useful or simply over-calls on noisy background EEG.

## How the code produces these outputs

### 1. Recording-level preprocessing

The recording-level pipeline is implemented in:

- `src/eeg_pipeline/pipeline/run_pipeline.py`
- `src/eeg_pipeline/pipeline/preprocessor.py`
- `src/eeg_pipeline/pipeline/filtering.py`

Behavior:

1. the EDF recording is loaded with MNE
2. optional preprocessing is applied
3. QC is computed on the cleaned signal
4. bad channels are marked from QC rules
5. EDA artifacts are written as plots and CSVs

### 2. Preprocessing parameters

From the current checked-in config in `src/eeg_pipeline/configs/config.yaml`, the default preprocessing settings are:

- bandpass filter: `1.0 to 60.0 Hz`
- notch filter: `60 Hz`
- filter method: `IIR`
- IIR design: Butterworth, order `4`
- rereferencing: disabled
- wavelet denoising: disabled
- epoching: enabled, `2.0 s` windows, `0.0 s` overlap
- artifact rejection: disabled

Meaning:

- very slow drift below `1 Hz` is suppressed
- high-frequency content above `60 Hz` is removed
- line noise at `60 Hz` is targeted
- because rereferencing is off, the bipolar montage structure remains as loaded
- because artifact rejection is off, all fixed-length epochs are retained

### 3. QC parameters

Current QC logic in `src/eeg_pipeline/analysis/time_domain.py` uses:

- clipping threshold: `500 µV`
- flat variance threshold: `1e-12 µV²`
- noisy-channel rule: variance greater than `10x` median channel variance

For `run-01`, this produces:

- `10` clipped channels
- `0` flat channels
- `0` variance-defined noisy channels

### 4. EDA feature extraction

The EDA engine in `src/eeg_pipeline/pipeline/eda_engine.py` computes:

- raw plots before and after preprocessing
- QC JSON summary
- mean PSD across channels
- PSD per channel
- integrated bandpower from PSD
- FFT on one chosen channel and time range
- epoch-level summary statistics
- optional spectrograms

For this run, the saved outputs indicate the artifact set included:

- raw plots
- QC JSON
- mean PSD CSV
- bandpower CSV
- PSD-per-channel plot
- PSD spectrogram
- Morlet spectrogram
- FFT plot and CSV
- epoch stats CSV

## Important config/artifact mismatch

The saved artifacts do **not** fully match the current checked-in config:

- current config says `eda.window_level.enabled: true` and `recording_level.enabled: false`, but the saved outputs are recording-level
- current config says Morlet is disabled, but a Morlet spectrogram exists
- current config shows FFT `tmax: 0.99`, but the saved file name is `t0.0-2.0`

Conclusion:

- these plots were generated with an earlier or alternate config state
- the code path is still valid, but exact settings for this saved run were not identical to the current config file

## How this recording becomes model input

The seizure classifier does **not** train directly on the EDA CSV features.

Instead, the training path is:

1. EDF recordings are windowed in `src/dataloader/loader/windowing.py`
2. each window gets a binary label
3. windows are converted to tensors in `src/dataloader/loader/tensor_writer.py`
4. models train on those tensors in `src/model/train/train_benchmark.py`

### Window labeling logic

For each window:

- label `1` if seizure overlap is at least `50%` of the window
- label `0` otherwise
- background windows close to a seizure are excluded within `300 s`
- background windows per file are capped at `150`
- seizure windows are never capped
- splits are balanced toward a seizure ratio of `30%`

So if `run-01` truly has no seizure intervals, it contributes only negative windows.

### Tensor features used by the model

The model input is not PSD/bandpower/FFT tables. The model sees:

- window shape: `(16 channels, 256 samples)`
- roughly `1 second` of EEG per example at `256 Hz`
- per-channel z-score normalization

That means the learned features are data-driven temporal and spatial patterns, not hand-engineered EDA summaries.

## How EEGNet uses those features

The baseline EEGNet in `src/model/models/eegnet.py` processes each `(16, 256)` window with:

- a temporal convolution to learn frequency-sensitive filters
- a depthwise spatial convolution across channels to learn montage/channel relationships
- separable convolution to refine temporal structure
- pooling and dropout before final binary classification

So, for a recording like `run-01`, the model can learn patterns such as:

- diffuse slow-wave dominance
- synchronized activity across neighboring bipolar channels
- transient bursts that are artifact-like rather than ictal
- spatial patterns that do or do not resemble seizure propagation

## What output is based on what parameters

There are two different outputs in this project:

### A. EDA outputs

These are based on preprocessing and signal-analysis parameters:

- filters: bandpass/notch settings
- epoch length
- PSD frequency range
- chosen band definitions
- FFT time window and channel
- spectrogram settings such as window size and step

### B. Model outputs

These are based on training and decision parameters:

- architecture choice, such as `eegnet` or `eegnet_improved`
- input tensor size: `16 x 256`
- loss function, for example focal loss
- optimizer and learning rate
- balancing of seizure vs background windows
- threshold chosen from validation predictions for best F1

From the current benchmark training config in `src/model/train/configs/config.yaml`, the default training setup is:

- model: `eegnet_improved`
- batch size: `64`
- epochs: `25`
- learning rate: `0.0003`
- weight decay: `0.01`
- loss: focal loss
- focal gamma: `2.0`
- focal alpha: `0.50`
- scheduler: one-cycle
- dropout: `0.30`

## Bottom line for run-01

`sub-chb01_sub-chb01_ses-001_task-eeg_run-01_eeg` is best interpreted here as a **non-seizure but challenging EEG recording** with:

- strong low-frequency dominance
- multiple high-amplitude clipped channels
- persistent rhythmic/harmonic structure
- substantial artifact-like transients

Its role in seizure detection is mainly:

- to teach the model what difficult background EEG looks like
- to reduce false positives on slow, noisy, or high-amplitude non-ictal activity
- to validate whether preprocessing and QC are separating physiological content from artifacts

## Recommended next step

If you want a report that is directly about **seizure onset behavior**, use a seizure-positive CHB01 file next, then generate:

- a populated `events.tsv` or seizure intervals from the CHB-MIT summary
- `seizure_vs_nonseizure.csv`
- side-by-side seizure and non-seizure window examples

That would let the report move from “hard negative/background EEG” to “actual ictal pattern characterization.”
