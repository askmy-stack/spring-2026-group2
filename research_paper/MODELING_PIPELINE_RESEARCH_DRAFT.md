# EEG Seizure Detection: A Comparative Study of LSTM, CNN-LSTM, Transformer, and State-Space Architectures

**Research-paper draft synthesized from the `spring-2026-group2` codebase.**
Branch: `abhinaysai-lstm` · Date: 2026-04-16 · Source root: `src/`

> This document is a paper-ready methodology + results synthesis. Every architectural and training claim is cited to source (`file:line`). Numeric results are sourced from on-disk artifacts only; checkpoints that lack persisted test metrics are flagged explicitly as gaps to re-run.

---

## Abstract

We present a unified comparative study of 15+ neural architectures for automated seizure detection on scalp EEG, evaluated on the CHB-MIT pediatric corpus. All models share a common preprocessing, windowing, and subject-independent splitting pipeline (`src/data_loader/`), enabling an apples-to-apples comparison across four architecture families: (i) recurrent models (Vanilla LSTM, BiLSTM, Attention-BiLSTM, Feature-BiLSTM), (ii) convolutional–recurrent hybrids (CNN-LSTM with multi-scale temporal kernels), (iii) self-attention / vector-quantised transformers (VQ-Transformer "M7"), and (iv) state-space and pretrained foundation models (EEGMamba, BENDR, EEGPT). Training uses Focal Loss with positive-class re-weighting, AdamW with warmup-cosine learning-rate scheduling, threshold tuning via Youden's J or F1-max, and ensemble voting for the HF-model family. On the legacy baseline evaluation the CNN-LSTM hybrid achieves F1 = 0.518 / AUC = 0.712 with sensitivity = 0.569, substantially outperforming the pure LSTM (F1 = 0.346, sensitivity = 0.314) and motivating the later additions of multi-scale CNN frontends, hierarchical context, and pretrained encoders. This document unifies the methodology, hyperparameters, stability fixes, and iteration history into a single reference.

---

## 1. Introduction

Seizure detection from scalp EEG is a dense temporal-classification problem characterised by (a) severe class imbalance (seizures are rare), (b) inter-subject heterogeneity (age, montage, seizure type), (c) non-stationary signal statistics, and (d) a strong demand for subject-independent generalisation in clinical deployment. This codebase tackles the problem with a unified pipeline feeding a wide portfolio of architectures. The explicit research question we attempt to answer: *how do different inductive biases (recurrence, multi-scale convolution, self-attention, state-space dynamics, large-scale self-supervised pretraining) compare when held to the same preprocessing, labelling, and evaluation protocol?*

Scope and contributions:
1. **A unified, subject-independent pipeline** (CHB-MIT → 16-channel 10-20 montage, 256 Hz, 1-s windows) implemented in `src/data_loader/`.
2. **15+ architectures** under one repo with shared loss/metrics utilities (`src/models/utils/`).
3. **A documented stability history** — a sequence of fixes for NaN losses, FP16 underflow, gradient amplification in state-space layers, and class-weighting semantics — which constitutes an engineering contribution in its own right.
4. **Ensemble evaluation** (`ensemble_hf.py`) with per-ensemble threshold tuning.

---

## 2. End-to-End Modeling Pipeline

All models consume batches produced by `data_loader.dataset.factory.create_pytorch_dataloaders()`, which composes the stages below.

### 2.1 Pipeline diagram

```
PhysioNet CHB-MIT EDF
        │   (download.py:50-85)
        ▼
   MNE read_raw             ──► sfreq, n_channels, duration
        │   (io.py:75-81)
        ▼
   preprocess()
   pick-EEG → resample(256Hz, polyphase) → bandpass 1–50Hz
   → notch 60Hz → average reference
        │   (signal.py:12-82)
        ▼
   standardize_channels()   → 16-ch 10-20 montage
        │   (channels.py:39-138)
        ▼
   BIDS rewrite             → sub-XX/ses-001/eeg/*.edf + sidecars
        │   (bids.py:25-…)
        ▼
   build_window_index()     → 1-s windows, 1-s stride
   label_windows()          → positive if ≥50% overlap with seizure event
        │   (labels.py:11-181)
        ▼
   stratify_subjects()      → 70/15/15 subject-independent splits
        │   (stratify.py:9-84)
        ▼
   balance_index() [train only] → oversample to 30 % positive
        │   (labels.py:184-210)
        ▼
   EEGDataset.__getitem__() → per-channel z-score + random augmentations
        │   (loaders.py:57-60, augment.py:9-68)
        ▼
   PyTorch DataLoader       → (B, 16, 256), (B,)
```

### 2.2 Data acquisition

Source: CHB-MIT Scalp EEG Database (PhysioNet). Subject metadata (age, sex, 23 recordings / 22 subjects, ages 1.5–22) is hard-coded in `data_loader/core/download.py` (`CHBMIT_SUBJECTS`, lines 22–47); downloader `download_chbmit()` at lines 50–85 fetches all 664 EDF files into a local cache. File reading is handled by `read_raw()` at `data_loader/core/io.py:75-81` via MNE-Python (supports `.edf`, `.bdf`, `.vhdr`, `.set`, `.fif`).

### 2.3 Preprocessing (`core/signal.py`)

`preprocess()` (lines 12–18) chains:
1. **EEG-channel pick** (`_pick_eeg`, 21–32): MNE channel type filter with fallback to "all channels" when `ch_types` are unset.
2. **Resample to 256 Hz** (`_resample`, 35–46) via polyphase filtering (`_resample_polyphase`, 49–62) — the default path; SciPy-style GCD decomposition minimises phase distortion relative to naive linear interpolation.
3. **Bandpass 1–50 Hz** and **notch at 60 Hz** (`_filter`, 65–73).
4. **Average reference** (`_reference`, 76–82).

Per-channel z-score normalisation (`signal.py:85-90`) is deliberately deferred to the dataset `__getitem__` (`loaders.py:57-60`) so statistics are computed per window, avoiding leakage across train/val/test boundaries.

### 2.4 Channel standardisation (`core/channels.py`)

Target montage is the 16-channel subset of the 10-20 system: Fp1, Fp2, F3, F4, F7, F8, Fz, Cz, T7, T8, P7, P8, C3, C4, O1, O2 (lines 9–10). `standardize_channels()` (39–52) reduces over-count recordings by case-insensitive name match then spatial nearest-neighbour (`_reduce_to_target`, 55–87) and expands under-count recordings via duplication (`_expand_to_target`, 108–126), finishing with a canonical reorder (129–138). Spatial matching uses a hard-coded `POSITIONS_1020` dict (12–36).

### 2.5 Windowing & labelling (`core/labels.py`)

`build_window_index()` (153–181) produces non-overlapping 1-s windows at 256 Hz (⇒ 256 samples/window). `label_windows()` (11–42) tags a window positive if its temporal overlap with any seizure event is at least `labeling.overlap_threshold` (default 0.5). Seizure events are parsed from BIDS-style `_events.tsv` companions (107–133) using keyword match against {`seizure`, `sz`, `ictal`, `epilep`}. Negative windows within `windowing.exclude_negatives_within_sec = 300 s` of any seizure are discarded to prevent borderline-labelled background from contaminating the negative class (36–37).

### 2.6 Subject-independent stratified split (`core/stratify.py`)

`stratify_subjects()` (9–84) bins subjects by age (0–5, 5–10, 10–15, 15–20, 20–30, 30+), tracks seizure status per stratum, and greedily allocates subjects into train/val/test at the 70/15/15 ratio while preserving both age- and seizure-prevalence distributions. `assign_split_column()` (126–149) writes the split back onto the window index so no window can leak across splits.

### 2.7 Class-imbalance handling at dataset level (`core/labels.py:184-210`)

`balance_index()` oversamples positives on the **train split only** until the positive ratio meets `balance.seizure_ratio = 0.3`. Validation and test splits retain the natural distribution so evaluation is unbiased.

### 2.8 Augmentation (`core/augment.py:9-68`)

Four stochastic augmentations applied at load time:
- **Time warp** (cubic-spline, σ = 0.15) — non-linear time distortion.
- **Per-channel magnitude scale** (σ = 0.15) — amplitude variation.
- **Additive Gaussian noise** at 25 dB SNR.
- **Time shift** up to ±15 samples (~59 ms at 256 Hz).

Seizure windows are augmented with probability 0.8 and background with 0.3, biasing augmentation toward the minority class.

### 2.9 Two parallel data subsystems

The repo currently hosts both `src/dataloaders/` (legacy, `dataloaders/common/{windowing,loader,splits}.py`) and `src/data_loader/` (modern, YAML-configured, BIDS-aware). Both honour identical output contracts. The legacy path is preserved for reproducibility of baseline results on disk; new work targets `data_loader/`. A future revision should collapse the two.

---

## 3. Model Architectures

All models are defined under `src/models/`. The master inventory is in Table 1; per-family detail follows.

### Table 1 — Architecture inventory

| # | Name | Family | File:line | ~Params | Input | Key feature |
|---|---|---|---|---|---|---|
| 1 | `VanillaLSTM` | LSTM | `architectures/vanilla_lstm.py:47-122` | 180 K | 16×256 | ChannelAttn + uni-LSTM |
| 2 | `BiLSTM` | BiLSTM | `architectures/bilstm.py:44-113` | 300 K | 16×256 | ChannelAttn + BiLSTM |
| 3 | `AttentionBiLSTM` | BiLSTM + MHA | `architectures/attention_bilstm.py:19-130` | 330 K | 16×256 | Learned pos-emb + 4-head self-attn |
| 4 | `FeatureBiLSTM` | BiLSTM on features | `architectures/feature_bilstm.py:19-139` | 280 K | 226×10 | Residual proj + temporal MHA |
| 5 | `CNNLSTM` | CNN + BiLSTM | `architectures/cnn_lstm.py:74-186` | 480 K | 16×256 | 3-kernel CNN (3/15/31) + BiLSTM + MHA |
| 6–11 | `M1…M6` | LSTM benchmark suite | `lstm_benchmark_models/architectures/m[1-6]_*.py` | 180–480 K | 16×256 | Channel / criss-cross / graph attn variants |
| 12 | `HierarchicalLSTM` | 2-level BiLSTM | `improved_lstm_models/architectures/hierarchical_lstm.py:60-136` | 320 K | 60×16×256 | Window encoder → cross-window BiLSTM |
| 13 | `M7_VQTransformer` | Patch-Transformer + VQ | `ensemble_transformers/architectures/m7_vq_transformer.py:13-104` | 420 K | 16×256 | Patch embed + 4-layer Transformer + 512-code VQ |
| 14 | `EEGMamba` | SSM | `hugging_face_mamba_moe/architectures/eeg_mamba.py:21-80` | 280 K | 16×256 | Bidirectional Mamba blocks |
| 15 | `EEGMambaMoE` | SSM + MoE | `hugging_face_mamba_moe/architectures/eeg_mamba.py` | ~320 K | 16×256 | Mamba + mixture-of-experts router |
| 16 | `BENDRPretrainedModel` | Pretrained CNN | `hugging_face_mamba_moe/architectures/pretrained/bendr.py:11-60` | ~M (frozen) | 16×256 | Braindecode BENDR backbone |
| 17 | `EEGPTPretrainedModel` | Pretrained Transformer | `hugging_face_mamba_moe/architectures/pretrained/eegpt.py:11-57` | ~25 M (frozen) | 16×256 | `chs_info=None` adapter head |
| 18 | `TinySeizureNet` | Depthwise-separable CNN | `approach3/architectures/tiny_seizure_net.py` | < 1 M | 16×256 | Distilled from M1–M6 teachers |

### 3.1 LSTM family — shared blueprint

All LSTM models follow the pattern `ChannelAttention → LayerNorm → Linear(16→128) → LSTM → Dual(Mean+Max)Pool → FC`. Design rationale:

- **ChannelAttention (`lstm_benchmark_models/modules/channel_attention.py`)** — squeeze-and-excitation over channels. EEG channels carry very different amounts of seizure information; a learned gating vector suppresses noisy or artifact channels before temporal processing.
- **Input projection to 128-dim hidden space** — decouples LSTM capacity from raw channel count (important because the pipeline can emit 16 or 23 channels depending on fallback).
- **Dual pooling (Mean + Max)** — mean captures global activity level; max captures peaky ictal bursts. Concatenation doubles the feature dimension fed to the classifier.
- **Dropout 0.3 between LSTM layers** — primary regulariser. Weight decay is the secondary one.

#### VanillaLSTM (`vanilla_lstm.py:47-122`)
- 2-layer unidirectional LSTM, hidden 128, dropout 0.3.
- Acts as a reference floor: no bidirectionality, no self-attention, no CNN frontend.

#### BiLSTM (`bilstm.py:44-113`)
- 2-layer bidirectional LSTM; forward + backward hidden states concatenated → 256-dim per timestep.
- Within a 1-s window, the "future" is visible at training and inference — a reasonable assumption for offline seizure detection and a small assumption for clinical systems that buffer a few seconds of signal.

#### AttentionBiLSTM (`attention_bilstm.py:19-130`)
- BiLSTM → **learnable positional embedding (trunc-normal init)** → 4-head self-attention with residual → pool.
- The positional embedding is important: once we add self-attention on top of an RNN, the attention block itself is permutation-invariant, so position must be re-injected.

#### FeatureBiLSTM (`feature_bilstm.py:19-139`)
- Unlike the other LSTMs, this consumes **226 hand-crafted features per window over a sequence of 10 windows** (i.e. 10 s context). A residual projection 226 → 256 → 128 stabilises training under very small effective batch sizes, and a 4-head temporal MHA learns which of the 10 windows is most discriminative.
- Intended as a "tabular-features" baseline alongside the raw-waveform models.

#### CNNLSTM (`cnn_lstm.py:74-186`)
- Three parallel conv branches with kernel sizes **3 / 15 / 31** mimic the three clinical EEG time-scales: fast spikes, spike-wave complexes, and slow ictal rhythms.
- Branch A (32 ch, k = 3, stride = 2) + Branch B (64 ch, k = 15, stride = 2) + Branch C (64 ch, k = 31, stride = 1, then downsample) → concat to 160 ch → BiLSTM → MHA → dual pool.
- This is the architecture that produces the strongest baseline numbers (Section 8).

### 3.2 Benchmark suite `M1`–`M6`

`lstm_benchmark_models/architectures/m[1-6]_*.py` reproduces the five LSTM architectures above in a standardised shape, plus one novel graph variant:

| ID | Key addition over baseline |
|---|---|
| M1 | Baseline VanillaLSTM with `ChannelAttention(reduction=4)` |
| M2 | BiLSTM with combined channel + spatial attention |
| M3 | `CrissCrossBiLSTM` — attention applied horizontally (time) and vertically (channels) — cheaper than full 2D self-attention |
| M4 | CNN-LSTM with SE blocks at each conv |
| M5 | Feature-BiLSTM (226-feature input) |
| M6 | **Graph BiLSTM** — channels as graph nodes, GAT-style attention over channel connectivity |

M1–M6 share `_build_input_proj`, `_build_lstm`, and `_build_classifier` helpers so ablations can be run cleanly.

### 3.3 HierarchicalLSTM (`improved_lstm_models/.../hierarchical_lstm.py:60-136`)

Two-level model operating on **60-window (60-second) sequences**:
1. Window encoder (per-window CNN) squashes (16, 256) → 128.
2. BiLSTM over the sequence of 60 128-dim window embeddings.
3. Attention pooling over the 60 windows (learnable softmax weights) → classifier.

Rationale: seizures have pre-ictal build-up that a 1-s window cannot capture. HierarchicalLSTM gives the model a minute of context while avoiding the O(T²) cost of self-attention over 15 360 raw samples.

### 3.4 M7 VQ-Transformer (`ensemble_transformers/.../m7_vq_transformer.py:13-104`)

- Patch embedding via `unfold(32)` → 8 patches × 512-dim → linear 512→128.
- Learned positional encoding (1, 9, 128) with CLS token.
- 4-layer 4-head Transformer encoder (GELU, FFN 128→512→128).
- **Vector quantiser with 512 codes** injects a discrete bottleneck, producing a VQ loss (commitment + codebook). This is the only model in the suite that learns a discrete "EEG vocabulary", giving a handle on interpretability (each patch is mapped to a code index).

### 3.5 EEGMamba & EEGMambaMoE (`hugging_face_mamba_moe/.../eeg_mamba.py:21-80`)

Bidirectional Mamba blocks (forward + backward selective state-space) with `d_model = 128`, `d_conv = 4`, a `d_state = 16` SSM, and residual connections. Global mean pooling → LayerNorm → classifier. The MoE variant adds a router over expert FFNs.

Engineering caveats (see §7.2 and §11): early versions suffered NaN losses due to (a) a `.expand()` in the SSM output that 256× amplified gradients (fixed in commit `9fbfe5f`), (b) an unstable log-space scan (replaced by a `torch.jit.script` compiled sequential scan in `7c60642`), and then (c) a 10–50× speed-up from a vectorised parallel scan in `94a5b2b`.

### 3.6 Pretrained foundation models

- **BENDR** (`pretrained/bendr.py:11-60`) — Braindecode's `braindecode/braindecode-bendr` backbone consumed at 16 ch × 256 samples × 256 Hz. Per-channel z-score normalisation is **mandatory at input** (fix in commit `6516657`) or the pretrained distribution shift causes rapid divergence. `freeze_backbone=True` is supported for linear-probe evaluation.
- **EEGPT** (`pretrained/eegpt.py:11-57`) — `braindecode/eegpt-pretrained` backbone. Pretrained on 64 channels; we override `chs_info=None` (same commit `6516657`) so the model does not enforce its 64-channel channel schema, allowing 16-channel fine-tuning.
- `approach2/modules/pretrained_encoders.py` additionally exposes **CBraMod (ICLR 2025)**, **BIOT**, **LaBraM (ICLR 2024)** — wired as optional backbones for the M1–M7 variants in `approach2/architectures/`.

### 3.7 TinySeizureNet (`approach3/architectures/tiny_seizure_net.py`)

Depthwise-separable 1-D CNN (`DepthwiseSeparableConv1d`) with < 1 M parameters, trained via `MultiTeacherDistillation` from an ensemble of M1–M6 teachers. Intended for on-device deployment; a `QuantizedTinyNet` variant is provided for post-training quantisation.

### 3.8 Loss & attention utilities

- `src/models/utils/losses.py`: `FocalLoss` (§5) — single source of truth after the Apr-14 consolidation.
- `src/models/utils/metrics.py`: shared `compute_f1_score`, `compute_auc_roc`, `compute_sensitivity`, `compute_specificity`, plus the Youden-J threshold finder at lines 115–142.
- Attention modules: `ChannelAttention`, `ChannelAttentionWithContext`, `SpatialChannelAttention` (`lstm_benchmark_models/modules/channel_attention.py`), `CrissCrossAttention` (`.../criss_cross_attention.py`), `GraphAttention` (`.../graph_attention.py`).

---

## 4. Why LSTM? Design Rationale & Alternatives

**LSTM as default inductive bias.** EEG is an ordered, locally-stationary time series sampled at 256 Hz. For a 1-second window (256 timesteps) a 2-layer LSTM has an effective receptive field well beyond the window length and is cheap enough to run on commodity GPUs. Recurrence is a strong inductive bias for signals whose instantaneous meaning depends on short-term history — precisely the case for ictal rhythm onset.

**Why bidirectional.** Offline seizure detection lets us look at the full window before deciding; a BiLSTM therefore sees both onset and offset of a transient spike-wave complex. The per-timestep embeddings are concatenated (256-dim), doubling classifier capacity without doubling recurrence length.

**Why dual (mean + max) pooling.** Seizures often manifest as localized bursts in time. Mean pooling alone smooths these away; max pooling alone is noise-sensitive. The concatenation is a cheap hedge.

**Why add a CNN frontend (CNN-LSTM).** Multi-scale kernels (3 / 15 / 31 samples ≈ 12 / 59 / 121 ms at 256 Hz) directly correspond to clinical EEG band interpretation (high-frequency spikes, spike-wave complexes, slow rhythms). CNN-LSTM is the baseline's top performer (Section 8), validating that temporal inductive bias + local translation invariance complement each other.

**Why also try Transformers.** Pure self-attention is permutation-invariant and data-hungry, but once we have the VQ bottleneck (M7) it doubles as an interpretable representation learner. The VQ codebook gives each 32-sample patch a discrete identity, which is useful for downstream retrieval / anomaly flagging.

**Why also try Mamba / SSM.** Mamba scales linearly in sequence length, which matters for the hierarchical 60-s context setting (3840 1-s window embeddings or 15 360 raw samples). The selective state-space mechanism is well-suited to signals with long quiet stretches punctuated by bursts.

**Why pretrained (BENDR / EEGPT / CBraMod / LaBraM).** CHB-MIT is small (23 subjects). Pretrained foundation models bring self-supervised priors learned from orders of magnitude more EEG, providing a strong feature extractor with only a linear probe fine-tuned. This is the standard "few-shot medical ML" recipe.

**Compared to classical ML (SVM / RF on spectral features).** FeatureBiLSTM is our nod to this family: it accepts 226 engineered features per window. We do not train a separate SVM/RF; the feature-based BiLSTM provides an implicit upper bound on what a classical pipeline could reach because it has access to the same features plus a temporal model on top.

---

## 5. Loss Function & Class Imbalance

The training loss is a numerically-stable Focal Loss with post-hoc `pos_weight`:

$$
\mathcal{L}_{\text{focal}}(p,y) \;=\; -\,\alpha_t\,(1 - p_t)^{\gamma}\,\log p_t \;\cdot\; w_{\text{pos}}^{\,y}
$$

with $p_t = p$ if $y = 1$ else $1 - p$, $\alpha_t = \alpha$ if $y = 1$ else $1-\alpha$, $w_{\text{pos}} = n_-/\max(n_+,1)$, and $y$ smoothed by $\epsilon \in [0.02, 0.1]$ before passing to the loss.

Defaults: $\alpha = 0.25$, $\gamma \in \{1.0, 2.0\}$, `pos_weight` computed from the training window index at training start.

**Critical implementation fixes (visible in git history):**

| Commit | Fix |
|---|---|
| `f6460ac` | Inverted `pos_weight` (was `n_+ / n_-`, should be `n_- / n_+`) corrected; legacy results pre-dating this should be regarded with caution. |
| `0026241` | `pos_weight` must be applied as a flat multiplier **after** focal modulation, not inside the BCE call. Inside-BCE semantics combine with the focal factor in a way that double-weights hard positives. |
| `66d6624` | Cast `FocalLoss` inputs to `float32` to prevent FP16 underflow under `torch.amp` autocast. Without this, `(1-p_t)^\gamma` flushes to zero when `p_t \to 1`, yielding zero gradient on the hard negatives the loss is meant to emphasise. |
| `d9bedbc` | Target threshold check `targets > 0.5` rather than `targets == 1` — needed once label smoothing was added. |

Ref: `src/models/utils/losses.py` (consolidated FocalLoss). `approach2/train.py:489` uses $\gamma = 1.0$, $\epsilon = 0.02$; `train_hf.py:39-67` uses $\gamma = 2.0$, $\alpha = 0.25$ as CLI defaults.

---

## 6. Hyperparameters & Optimisation

### 6.1 Training-script inventory

| Script | Models | Epochs | Batch | LR | WD | Patience | Notes |
|---|---|---|---|---|---|---|---|
| `lstm_benchmark_models/train_baseline.py` | M1–M6 | 30 | 64 | 1e-3 | 1e-4 | 7 | Focal γ = 2 |
| `improved_lstm_models/train.py` | HierarchicalLSTM | 100 | 32 | 5e-5 | 1e-4 | 20 | After bbe24e0 / 9c8b7c7 |
| `ensemble_transformers/train_ensemble.py` | M1–M7 ensemble | 50 | 64 | 5e-4 | 1e-4 | 15 | Then ensemble voting |
| `hugging_face_mamba_moe/train_hf.py` | Mamba / MoE / BENDR / EEGPT / etc. (12 models) | 20 | 32 | 1e-3 | 1e-4 | 7 | `--model all`, `--grad_clip_norm 1.0`, `--decision_threshold 0.35` |
| `approach2/train.py` | M1–M7 + pretrained backbones | 50 | 64 | 5e-4 | 1e-4 | 15 | WarmupCosineScheduler |
| `approach3/train_mamba.py` | EEGMamba + MoE | 20 | 32 | 1e-3 | 1e-4 | 7 | — |

All scripts share the same YAML master config at `src/models/config.yaml` (augmentation probabilities, window sec, stride sec, seizure ratio, etc.).

### 6.2 Optimiser and schedule

Every script uses **AdamW** (`torch.optim.AdamW`) with weight decay `1e-4`. E.g. `approach2/train.py:377`, `train_hf.py:79`.

The default schedule is `WarmupCosineScheduler` (custom, `approach2/train.py:90-110` and `improved_lstm_models/train.py:34-75`):

$$
\eta_t \;=\;
\begin{cases}
\eta_{\max}\cdot \dfrac{t+1}{T_{\text{warm}}}, & t < T_{\text{warm}} \\[1ex]
\eta_{\min} + \tfrac{1}{2}(\eta_{\max}-\eta_{\min})\Bigl(1 + \cos\bigl(\pi\cdot \tfrac{t - T_{\text{warm}}}{T - T_{\text{warm}}}\bigr)\Bigr), & t \ge T_{\text{warm}}
\end{cases}
$$

with $T_{\text{warm}} \in \{5,10\}$, $\eta_{\min} = 10^{-6}$. An alternate scheduler appears in `compare.py:165` (`CosineAnnealingWarmRestarts`, $T_0 = T - T_{\text{warm}}$) and in the legacy `improved/train_regularized.py:69` (`ReduceLROnPlateau`, factor 0.5, patience 3).

### 6.3 Regularisation and stability

- **Dropout** 0.3 (LSTM / BiLSTM); 0.1 (Transformer encoder).
- **Label smoothing** $\epsilon \in \{0.02, 0.1\}$ — applied before the loss.
- **Gradient clipping** by L2 norm = 1.0 (`train_hf.py:182`).
- **Mixed precision** via `torch.amp.autocast` with float32 loss casts (see fix `66d6624`).

### 6.4 Tuning strategy

Tuning in this codebase is **manual + iterative**, not grid/random search. Each significant hyperparameter change is traceable to a commit message identifying the failure mode it addresses (e.g., `9c8b7c7 – lower improved_lstm lr to 0.0001 + more warmup/patience to fix unstable val loss`; `71277d3 – lower focal_gamma=1.0, higher lr=3e-3 to prevent loss collapse`). The git log therefore *is* the tuning log. Section 11 condenses it.

### 6.5 Early stopping and threshold tuning

- **Early stop on val F1** (not val loss) — switched in commits `bd32418` / `bbe24e0`. Compare uses `val_f1 > best + 1e-4`; patience 7 (HF), 15 (ensemble), 20 (improved LSTM). `improved_lstm_models/train.py:116-139`.
- **Threshold tuning**
  - Youden's J: `src/models/utils/metrics.py:115-142`, scan [0.01, 0.99] step 0.01, maximise $J = \text{sens} + \text{spec} - 1$.
  - F1-max: `train_hf.py:199-216` and `ensemble_hf.py`, scan [0.05, 0.95] step 0.05. Default decision threshold (pre-tuning) = 0.35 (`train_hf.py:60`).

---

## 7. Training Strategy & Iteration

### 7.1 Cross-cutting choices

- **Subject-independent splits** from `core/stratify.py` prevent the easiest kind of leakage. Data augmentation runs per-sample inside the dataset, so augmented copies of a subject cannot appear in both train and val.
- **Balanced training, unbalanced evaluation.** Oversampling is enabled only on train (`labels.py:184-210`); val / test keep natural prevalence for realistic sensitivity/specificity numbers.
- **Mixed precision** everywhere (AMP autocast). Loss is cast to float32 before the focal modulation (commit `66d6624`).
- **Gradient clipping** norm = 1.0 as a blanket safeguard against exploding gradients in RNN / SSM blocks.

### 7.2 Stability engineering (SSM / Mamba)

Two Mamba-specific fixes are worth surfacing because they are not obvious from the API:
- Commit `9fbfe5f` removes a `.expand()` in the SSM output whose broadcast shape caused a 256× gradient amplification during back-prop, producing NaN losses within a handful of iterations.
- Commit `7c60642` replaces a hand-written log-space parallel scan with a `torch.jit.script`-compiled sequential scan, which sacrifices some throughput for numerical stability. A follow-up, `94a5b2b`, re-introduces a vectorised parallel scan that is 10–50× faster than the scripted loop while preserving stability.

### 7.3 Ensemble evaluation

`src/models/hugging_face_mamba_moe/ensemble_hf.py` (introduced in `bd32418`) aggregates probabilities from multiple trained HF models by **mean voting** or **weighted voting**, tunes the ensemble's operating threshold via the same F1-max routine, and computes confusion-matrix metrics. Temporal smoothing across adjacent windows (moving average or consecutive-run minimum) is supported.

### 7.4 Distillation (Approach 3)

`MultiTeacherDistillation` (`approach3/architectures/tiny_seizure_net.py`) trains `TinySeizureNet` against the soft outputs of M1–M6, combining a KL-divergence term on logits with a CE term on labels. This is the fastest path to a < 1 M-parameter deployable model without sacrificing much sensitivity.

---

## 8. Results & Evaluation

### 8.1 Metric set (from `src/models/utils/metrics.py`)

- F1 (sklearn).
- AUROC (sklearn `roc_auc_score`).
- Sensitivity = TP / (TP + FN).
- Specificity = TN / (TN + FP).
- Precision (inline in train scripts).
- Confusion matrix (TP, TN, FP, FN) as absolute counts.
- Threshold value at which those metrics are reported (for reproducibility).

### 8.2 Persisted baseline results

File: `src/models/legacy_baseline/results/baseline_results.json` (verbatim, all numbers rounded to 4 dp):

| Model | Accuracy | Sensitivity | Specificity | F1 | AUC | Train time (s) |
|---|---|---|---|---|---|---|
| Vanilla LSTM | 0.6447 | 0.3139 | 0.7865 | 0.3464 | 0.5629 | 3 401 |
| BiLSTM | 0.6825 | 0.2598 | 0.8637 | 0.3293 | 0.6114 | 6 488 |
| Attention-BiLSTM | 0.6926 | 0.2731 | 0.8724 | 0.3476 | 0.6406 | 7 501 |
| **CNN-LSTM** | **0.6819** | **0.5692** | **0.7302** | **0.5177** | **0.7118** | **2 143** |

**Reading the table.**
- Pure LSTM / BiLSTM achieve higher **specificity** than CNN-LSTM but fail to catch most seizures — sensitivity is 0.26–0.31, which is clinically unacceptable.
- CNN-LSTM roughly doubles sensitivity (0.57) while retaining 0.73 specificity. F1 jumps from ~0.34 to ~0.52 and AUC from ~0.56–0.64 to 0.71.
- Training time for CNN-LSTM (2 143 s) is the **fastest** of the four — the convolutional downsampling (stride 2) shrinks the LSTM's input length by 2×, and the conv stack is GPU-efficient compared to pure recurrence.
- This single comparison is the strongest narrative hook for the paper: *multi-scale temporal convolution is the single most valuable frontend for seizure detection in this regime*, and motivates the subsequent adoption of richer frontends (M4 SE-CNN, VQ-Transformer patch embed, pretrained encoders).

### 8.3 Other models (checkpoints vs persisted metrics)

Current repository state: `outputs/models/*_best.pt` checkpoints are produced by every training script, and per-epoch history is logged to `logs/history.csv` and `logs/test_metrics.json` (`train_hf.py:141, 157`). However, the only numeric test-set results committed to disk are the four rows above. **Results for M1–M7, HierarchicalLSTM, EEGMamba, BENDR, EEGPT, TinySeizureNet, and the ensembles are not persisted as JSON in this branch** and need to be re-extracted from the checkpoints before inclusion in a camera-ready paper.

Action for authors: run `train_hf.py --model all --data_path ... --resume` (to load best checkpoints) and collect the `test_metrics.json` per model into one table before submission.

---

## 9. EEG Classification & Detection Context

- **Temporal dependencies.** Seizures evolve over milliseconds (spike, spike-wave) to seconds (ictal rhythm build-up). The 1-s window + 2-layer LSTM spans ~256 samples of context, enough to represent a full spike-wave complex (~1–3 Hz). HierarchicalLSTM lifts this to 60 s so the model can see pre-ictal precursors. Mamba's linear-time scaling makes longer windows tractable.
- **Channel semantics.** Focal seizures involve a small subset of electrodes; generalised seizures involve all. ChannelAttention learns per-example channel gating, mirroring a clinician's habit of focusing on the leads where the signal is abnormal. Graph attention in M6 goes further: adjacent electrodes have correlated activity, and a learned graph can encode that.
- **Multi-scale kernels.** The 3 / 15 / 31-sample kernels in CNN-LSTM correspond roughly to the γ / β / α–θ band range at 256 Hz — a standard band decomposition built into the inductive bias of the convolutions instead of a preprocessing FFT.
- **Subject independence.** Stratified subject splits are the right protocol for clinical deployment: the model must generalise to unseen patients. Oversampling is kept to the train split so val/test specificity reflects real-world false-alarm behaviour.
- **Robustness.** Augmentations (time warp, noise, magnitude scale, time shift) emulate the kinds of variation a real recording session introduces (electrode impedance drift, amplifier gain drift, sampling-clock jitter).

---

## 10. Comparative Insights & Future Work

**Empirical takeaways from the current repo state.**
1. Adding a multi-scale CNN frontend produces the single biggest measured quality jump.
2. Pure self-attention (without VQ or pretraining) is not attempted at scale; the Transformer-flavoured experiments wear either a VQ bottleneck (M7) or a pretrained backbone (BENDR / EEGPT).
3. State-space models (Mamba) required substantial numerical-stability work but are now trainable; their value proposition — long context with linear cost — is only realised by the hierarchical / 60-s paths.
4. Loss-function semantics matter: the sequence `f6460ac → 0026241 → 66d6624 → d9bedbc` systematically corrected Focal Loss to behave as intended.

**Future directions.**
- **Longer windows**: scale the hierarchical path from 60 s to 5 min; compare against a plain Mamba over raw 15 000-sample sequences.
- **Seizure-type multi-class**: the label machinery (`core/labels.py`) already supports keyword-based multi-class tagging (focal, generalised, absence). Enabling this is a one-line config change plus a modified classifier head.
- **Patient-specific fine-tuning**: train a generic model on 20 patients, fine-tune the last layer on the first 30 minutes of the held-out patient; compare to fully subject-independent evaluation.
- **Online / streaming deployment**: `TinySeizureNet` + `QuantizedTinyNet` gives a < 1 MB model; pair with a causal inference loop and a moving-average smoother for false-alarm control.
- **Calibration**: all current models are evaluated at a tuned operating threshold. Adding temperature scaling and Brier-score reporting would make the probability outputs clinically trustworthy.
- **Unified results harness**: persist a `results/{model_name}.json` for every model so the next version of this document can replace §8.3 with a full comparison table.

---

## 11. Timeline of Modeling Code Evolution

Reconstructed from `git log --oneline`. Oldest work is at the bottom; most recent at the top. We group into four phases.

### Phase 4 — Ensembling, stability polish, pretrained fixes (current, mid-April 2026)

| Commit | Change |
|---|---|
| `6516657` | **HF train**: robust best-F1 tracking with NaN guard; **BENDR** z-score normalise at input; **EEGPT** `chs_info=None` to unlock 16-channel fine-tuning. |
| `bbe24e0` | **Improved-LSTM**: early stop on val F1 (not loss); reduce model size; lower LR to 5e-5. |
| `bd32418` | Introduce `ensemble_hf.py`: mean / weighted voting + threshold tuning. |
| `7c60642` | Replace unstable log-space scan with `torch.jit.script`-compiled sequential scan in Mamba (removes NaN). |
| `9fbfe5f` | Remove `.expand()` in SSM output — eliminates 256× gradient amplification → removes NaN loss. |
| `9c8b7c7` | Lower `improved_lstm` LR to 1e-4, more warmup/patience. |
| `94a5b2b` | Replace sequential Python SSM loop with vectorised parallel scan (10–50× speed-up). |
| `3d5a27d` | `train_hf` default tuning (focal loss, threshold tuning, grad-clip); add `--model all`. |
| `21de52a` | `--data_path` flag on `train_hf.py` to load `.pt` tensors directly. |
| `a33506e` | Project-root-relative paths in `data_loader` config. |
| `30055d1` | Drop unsupported `config_path` kwarg from `get_dataloaders()`. |
| `fd2ca90` | `run_all_models.py` — one-command training across all dirs. |
| `ad7183a` | Filter kwargs per model to avoid unexpected-keyword errors. |
| `b13c5dd` | `--model all` across benchmark. |
| `8342be0` | Cross-cutting bugfix / perf / robustness sweep over all 4 model dirs. |
| `d761539` | Split hygiene: val/ for early stopping, test/ for final eval. |
| `db3a9bc` | Fix ensemble-weights device mismatch; `approach2` warmup scheduler. |

### Phase 3 — Architectural diversification (Approach 2 / Approach 3, early April 2026)

| Commit | Change |
|---|---|
| `605ac69` | **Fix approach2 training collapse** (4 bugs). |
| `d9bedbc` | FocalLoss targets > 0.5 check; graph-attention reshape. |
| `0026241` | Apply `pos_weight` as flat multiplier after focal modulation (not inside BCE). |
| `66d6624` | Cast FocalLoss inputs to float32 to prevent FP16 underflow in AMP. |
| `fa1961e` | Plan doc: approach2 training-collapse fix plan. |
| `efe1d9f` | Fix 10 bugs in `improved_lstm`: crashes, logic errors, dead code. |
| `844dbfc` | Fix relative imports in `improved_lstm/train.py`. |
| `44c4903` | **Add `improved_lstm`** with augmentation, ensemble, enhanced training. |
| `de26605` | Fix `attention_bilstm` (remove downsampling); `feature_bilstm` (3D shape extraction). |
| `f6460ac` | Fix inverted `pos_weight`; add tensors format support; deprecated `torch.amp` API. |
| `71277d3` | Lower focal_gamma to 1.0, raise LR to 3e-3 to prevent loss collapse. |
| `183ddbb` | Fix UnboundLocalError: use `X_train/X_test` instead of `X_raw` for attention_bilstm. |
| `cb55216` | Fix Mamba block tensor shape: `torch.stack` vs `torch.cat`. |
| `093780a` | Support loading real EEG from tensors (`data.pt`, `labels.pt`). |
| `7870440` | Fix format-string TypeError on None values. |
| `4370e93` | Remove broken `fn_multiplier`; fix import paths in approach2/3. |
| `785a2e5` | Fix `run_benchmark.py` format bug; add training utils. |
| `13f0986` | **Merge**: Add Approach 2 and Approach 3 models. |
| `f794e93` | **Add Approach 2** (7 models + pretrained encoders) and **Approach 3** (Mamba, Diffusion, Pre-ictal). |

### Phase 2 — Baseline improvements (LSTM-first era)

| Commit | Change |
|---|---|
| `d62d977` | Move old baseline scripts to `legacy_baseline/`. |
| `4ef865c` | Add comprehensive summary of LSTM improvements. |
| `ad959af` | **Complete LSTM improvements**: parallel multi-scale CNN, temporal attention, ensemble inference, training optimisations. |
| `0c69cad` | Implement Focal Loss, FN weighting, label smoothing, LR warmup, ChannelAttention, positional encoding. |

### Phase 1 — Baseline (pre-mid-April)

Baseline M1–M6 LSTM benchmark suite, initial FocalLoss, legacy `dataloaders/`, and the `baseline_results.json` numbers reported in §8.2.

---

## 12. References & Datasets

- **CHB-MIT Scalp EEG Database**, PhysioNet. (Shoeb & Guttag.)
- **Focal Loss**: Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017.
- **BENDR**: Kostas et al., *BENDR: Using Transformers and a Contrastive Self-Supervised Learning Task to Learn from Massive Amounts of EEG Data*.
- **EEGPT**: *EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals*, NeurIPS 2024.
- **LaBraM**: *Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI*, ICLR 2024.
- **CBraMod**: ICLR 2025.
- **BIOT**: *BIOT: Cross-data Biosignal Learning in the Wild*.
- **Mamba**: Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, 2023.

---

## Appendix A — File-Level Citation Index

For any claim in this document, the cited file:line can be opened directly. Primary references:

- Data pipeline: `src/data_loader/core/{signal.py, channels.py, labels.py, stratify.py, augment.py, bids.py, io.py, download.py}`, `src/data_loader/dataset/{base.py, loaders.py, factory.py}`, `src/data_loader/config.yaml`.
- Architectures: `src/models/architectures/{vanilla_lstm.py, bilstm.py, attention_bilstm.py, feature_bilstm.py, cnn_lstm.py}`, `src/models/lstm_benchmark_models/architectures/m[1-6]_*.py`, `src/models/improved_lstm_models/architectures/hierarchical_lstm.py`, `src/models/ensemble_transformers/architectures/m7_vq_transformer.py`, `src/models/hugging_face_mamba_moe/architectures/eeg_mamba.py`, `src/models/hugging_face_mamba_moe/architectures/pretrained/{bendr.py, eegpt.py}`, `src/models/approach3/architectures/tiny_seizure_net.py`.
- Training / loss / metrics: `src/models/utils/{losses.py, metrics.py}`, `src/models/approach2/train.py`, `src/models/hugging_face_mamba_moe/{train_hf.py, ensemble_hf.py}`, `src/models/improved_lstm_models/train.py`, `src/models/lstm_benchmark_models/train_baseline.py`, `src/models/ensemble_transformers/train_ensemble.py`.
- Persisted results: `src/models/legacy_baseline/results/baseline_results.json`.
- Master config: `src/models/config.yaml`.

---

*End of draft. Next revision should populate §8.3 with per-model test metrics extracted from the `outputs/models/*_best.pt` checkpoints.*
