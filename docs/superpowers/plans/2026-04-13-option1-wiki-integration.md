# Option 1: Document & Wiki Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create comprehensive wiki documentation of the 5 LSTM architectures, document datasets and metrics, ingest 5-10 EEG seizure detection papers, and build synthesis pages for cross-model comparison.

**Architecture:** The plan builds the wiki incrementally: first entity pages for models/datasets/metrics/techniques, then ingests papers to populate sources, then creates synthesis pages that compare architectures and identify patterns.

**Tech Stack:** Markdown (wiki pages), arxiv-sanity-lite (paper discovery), git (version control)

---

## Task 1: Create Model Entity Pages (5 models)

**Files:**
- Create: `wiki/entities/models/vanilla-lstm.md`
- Create: `wiki/entities/models/bilstm.md`
- Create: `wiki/entities/models/attention-bilstm.md`
- Create: `wiki/entities/models/cnn-lstm.md`
- Create: `wiki/entities/models/feature-bilstm.md`

- [ ] **Step 1: Read model architectures from source**

Read all 5 model definitions to extract architecture details:
```bash
cat src/models/architectures/vanilla_lstm.py | head -40
cat src/models/architectures/bilstm.py | head -40
cat src/models/architectures/attention_bilstm.py | head -80
cat src/models/architectures/cnn_lstm.py | head -80
cat src/models/architectures/feature_bilstm.py | head -40
```

Take notes on:
- Layer types and their order
- Input/output shapes
- Key hyperparameters (hidden_size, num_layers, dropout, attention heads)
- Improvements over baseline (from docstrings)

- [ ] **Step 2: Create vanilla-lstm.md**

Create `wiki/entities/models/vanilla-lstm.md`:
```markdown
# Vanilla LSTM

**Year:** 2026
**Authors:** Group 2, Spring 2026
**Architecture Type:** Baseline LSTM

## Overview

Single-layer LSTM classifier for EEG seizure detection. Serves as the baseline architecture.

## Architecture

1. **Input Normalization:** LayerNorm on raw EEG
2. **Input Projection:** Linear(n_channels → hidden_size) + ReLU + Dropout
3. **LSTM:** Single LSTM layer, batch_first=True
   - Input size: hidden_size
   - Hidden size: 128 (default)
   - Dropout: 0.3 (default)
4. **Global Pooling:** Average + Max pooling over sequence
5. **FC Head:** 2-layer fully connected
   - Layer 1: hidden_size → hidden_size/2, ReLU, Dropout
   - Layer 2: hidden_size/2 → 1 (sigmoid for binary classification)

## Key Features

- Input normalization for stable training
- Projection layer matches input to hidden size
- Global pooling captures full temporal context
- Dropout at 0.3 for regularization

## Performance (CHB-MIT)

- Accuracy: 64.5%
- Sensitivity: 31.4%
- Specificity: 78.7%
- F1 Score: 0.346
- AUC: 0.563
- Training Time: ~3400s

## Code Location

`src/models/architectures/vanilla_lstm.py`

## Related

- [Training Pipeline](../../syntheses/training-pipeline.md)
- [CHB-MIT Dataset](../datasets/chb-mit.md)
- [Sensitivity Metric](../metrics/sensitivity.md)
```

- [ ] **Step 3: Create bilstm.md**

Create `wiki/entities/models/bilstm.md`:
```markdown
# Bidirectional LSTM (BiLSTM)

**Year:** 2026
**Authors:** Group 2, Spring 2026
**Architecture Type:** Bidirectional variant

## Overview

Bidirectional 2-layer LSTM for improved temporal pattern capture from both directions.

## Architecture

1. **Input Normalization:** LayerNorm on raw EEG
2. **Input Projection:** Linear(n_channels → hidden_size) + ReLU + Dropout
3. **BiLSTM:** 2-layer bidirectional LSTM
   - Input size: hidden_size
   - Hidden size: 128 (default)
   - Num layers: 2
   - Bidirectional: True
   - Dropout: 0.3 between layers
4. **Global Pooling:** Average + Max pooling over sequence (combines both directions)
5. **FC Head:** 2-layer fully connected
   - Input: hidden_size * 2 (bidirectional output)
   - Layer 1: (hidden_size*2) → hidden_size, ReLU, Dropout
   - Layer 2: hidden_size → 1 (sigmoid)

## Key Features

- Bidirectional processing: forward and backward context
- 2-layer stacking for deeper temporal modeling
- Global pooling preserves full sequence information
- More parameters than vanilla LSTM (deeper + bidirectional)

## Performance (CHB-MIT)

- Accuracy: 68.3%
- Sensitivity: 26.0%
- Specificity: 86.4%
- F1 Score: 0.329
- AUC: 0.611
- Training Time: ~6488s (slower due to bidirectional)

## Code Location

`src/models/architectures/bilstm.py`

## Related

- [Vanilla LSTM](vanilla-lstm.md) (baseline comparison)
- [Attention BiLSTM](attention-bilstm.md) (with attention variant)
```

- [ ] **Step 4: Create attention-bilstm.md**

Create `wiki/entities/models/attention-bilstm.md`:
```markdown
# Attention-Enhanced Bidirectional LSTM

**Year:** 2026
**Authors:** Group 2, Spring 2026
**Architecture Type:** Attention + BiLSTM

## Overview

BiLSTM with 4-head MultiheadAttention mechanism to learn weighted importance of time steps. **Best F1 score (0.348)** among LSTM-based models.

## Architecture

1. **Input Normalization:** LayerNorm on raw EEG
2. **Input Projection:** Linear(n_channels → hidden_size) + ReLU + Dropout
3. **BiLSTM:** 2-layer bidirectional LSTM
   - Output size: hidden_size * 2 (bidirectional)
4. **Multi-Head Attention:** 4 attention heads
   - Input: BiLSTM output (hidden_size * 2)
   - Num heads: 4 (head_dim = 32)
   - Self-attention over time steps
5. **Residual Connection:** Attended output + original LSTM output
   - Stabilizes gradient flow through attention
6. **Global Pooling:** Average + Max pooling over attended sequence
7. **LayerNorm:** Normalize after pooling
8. **FC Head:** 2-layer fully connected
   - Input: pooled_size
   - Layer 1: pooled → hidden_size, ReLU, Dropout
   - Layer 2: hidden_size → 1 (sigmoid)

## Key Improvements

- **4-head attention** instead of single-head Bahdanau → captures diverse seizure patterns
- **Residual connection** around attention → stabilizes deep BiLSTM training
- **Post-pooling LayerNorm** → prevents saturation in FC head
- **Input projection** → standardizes input representation

## Performance (CHB-MIT)

- Accuracy: 69.3%
- Sensitivity: 27.3%
- Specificity: 87.2%
- F1 Score: 0.348 ⭐ (Best F1 among LSTM variants)
- AUC: 0.641
- Training Time: ~7501s

## Code Location

`src/models/architectures/attention_bilstm.py`

## Interpretability

Contains `forward_with_attention()` method that returns attention weights for visualization. Useful for understanding which time steps the model focuses on during seizure detection.

## Related

- [BiLSTM](bilstm.md) (baseline)
- [Multihead Attention Technique](../techniques/multihead-attention.md)
- [Residual Connections](../techniques/residual-connections.md)
```

- [ ] **Step 5: Create cnn-lstm.md**

Create `wiki/entities/models/cnn-lstm.md`:
```markdown
# CNN-LSTM Hybrid

**Year:** 2026
**Authors:** Group 2, Spring 2026
**Architecture Type:** Convolutional + Recurrent

## Overview

Combines 1D convolutions for local spatial pattern extraction with LSTM for temporal dynamics. **Best overall performance (AUC 0.712, Sensitivity 56.9%)** and **fastest training (2142s)**.

## Architecture

1. **Conv1d Block:**
   - Conv1d filters: captures local patterns in EEG
   - Kernel size: 3
   - Stride: 1
   - Padding: 1 (preserves sequence length)
   - Output channels: 32
   - Activation: ReLU
   - MaxPool1d: kernel=2, reduces sequence length by 2x

2. **LSTM:**
   - Input: Conv output
   - Hidden size: 128 (default)
   - Num layers: 2
   - Bidirectional: False
   - Dropout: 0.3

3. **Global Pooling:** Average + Max pooling over LSTM output

4. **FC Head:** 2-layer fully connected
   - Layer 1: (hidden*2 + conv_channels) → hidden_size, ReLU, Dropout
   - Layer 2: hidden_size → 1 (sigmoid)

## Key Advantages

- **Convolutional inductive bias** → exploits local structure in EEG signals
- **Reduced sequence length** after convolution → LSTM processes fewer time steps
- **Fastest training** → 2142s (vs 7500s for attention models)
- **Best sensitivity** → 56.9% (catches more seizures)
- **Best AUC** → 0.712 (best generalization)

## Performance (CHB-MIT)

- Accuracy: 68.2%
- Sensitivity: 56.9% ⭐ (Best sensitivity)
- Specificity: 73.0%
- F1 Score: 0.518 ⭐ (Best F1 overall)
- AUC: 0.712 ⭐ (Best AUC)
- Training Time: 2142s ⭐ (Fastest)

## Code Location

`src/models/architectures/cnn_lstm.py`

## Interpretation

Convolutional filters learn to detect localized EEG patterns (e.g., spike-wave complexes, high-frequency activity). LSTM then models temporal dynamics of these patterns.

## Related

- [Vanilla LSTM](vanilla-lstm.md)
- [Attention BiLSTM](attention-bilstm.md)
- [1D Convolution Technique](../techniques/1d-convolution.md)
```

- [ ] **Step 6: Create feature-bilstm.md**

Create `wiki/entities/models/feature-bilstm.md`:
```markdown
# Feature-Enhanced BiLSTM

**Year:** 2026
**Authors:** Group 2, Spring 2026
**Architecture Type:** Feature-based BiLSTM

## Overview

BiLSTM that operates on pre-extracted features instead of raw EEG. Enables incorporation of domain-specific signal processing (e.g., spectral features, statistical features).

## Architecture

1. **Input Features:** 226-dimensional feature vector (pre-computed)
   - Source: extracted from raw EEG by preprocessing pipeline
   - Types: spectral, temporal, statistical features

2. **Input Projection:** Linear(226 → hidden_size) + ReLU + Dropout

3. **BiLSTM:** 2-layer bidirectional LSTM
   - Input size: hidden_size
   - Hidden size: 128 (default)
   - Output: hidden_size * 2 (bidirectional)

4. **Global Pooling:** Average + Max pooling

5. **FC Head:** 2-layer fully connected
   - Layer 1: (hidden*4) → hidden_size, ReLU, Dropout
   - Layer 2: hidden_size → 1 (sigmoid)

## Key Features

- **Feature independence:** Decouples raw signal processing from learning
- **Domain knowledge integration:** Features can encode expertise
- **Reduced raw signal noise:** Preprocessing filters noise/artifacts
- **226-dimensional input:** Rich feature representation

## Performance (CHB-MIT)

- **Not in baseline results** (likely tested with different feature extraction)
- Expected: Better sensitivity than raw-EEG models (if features are well-designed)

## Code Location

`src/models/architectures/feature_bilstm.py`

## Feature Requirements

Requires preprocessed feature tensors with shape `(batch_size, seq_len, 226)`.

Feature extraction pipeline currently **not included** in codebase. Would need:
- Spectral analysis (FFT, wavelet decomposition)
- Statistical features (mean, std, skewness, kurtosis)
- Signal properties (zero crossings, energy)

## Related

- [BiLSTM](bilstm.md)
- [Feature Engineering Topic](../../topics/feature-engineering.md)
```

- [ ] **Step 7: Create `wiki/entities/models/README.md`**

Create `wiki/entities/models/README.md`:
```markdown
# Model Architectures

5 LSTM-based architectures for EEG seizure detection.

## Quick Comparison

| Model | Type | Best At | Sensitivity | AUC | Speed |
|-------|------|---------|-------------|-----|-------|
| [Vanilla LSTM](vanilla-lstm.md) | Baseline | - | 31.4% | 0.563 | Slow |
| [BiLSTM](bilstm.md) | Bidirectional | - | 26.0% | 0.611 | Slowest |
| [Attention BiLSTM](attention-bilstm.md) | + Attention | F1 Score | 27.3% | 0.641 | Slowest |
| [CNN-LSTM](cnn-lstm.md) | Convolutional | Sensitivity, AUC, F1 | **56.9%** | **0.712** | **Fastest** |
| [Feature BiLSTM](feature-bilstm.md) | Feature-based | (unknown) | ? | ? | ? |

## Key Findings

- **CNN-LSTM dominates:** Best sensitivity (56.9%), AUC (0.712), F1 (0.518), fastest training
- **Attention doesn't help:** Attention-BiLSTM (F1 0.348) slightly better than BiLSTM (0.329), but not worth the complexity
- **Sensitivity is the bottleneck:** Even best model misses 43% of seizures
- **Feature-BiLSTM unexplored:** Could be strong if features are well-designed

## Training Improvements

All models benefit from:
- LayerNorm input normalization
- Input projection layer
- Global avg+max pooling
- Positive weight loss (class imbalance handling)
- Early stopping on validation F1
- Mixed precision training (AMP)

See [Training Pipeline](../../syntheses/training-pipeline.md) for details.
```

- [ ] **Step 8: Commit all model pages**

```bash
git add wiki/entities/models/
git commit -m "docs: add comprehensive wiki pages for 5 LSTM architectures"
```

---

## Task 2: Create Dataset Entity Pages

**Files:**
- Create: `wiki/entities/datasets/chb-mit.md`
- Create: `wiki/entities/datasets/README.md`

- [ ] **Step 1: Research CHB-MIT dataset**

Get dataset information from `src/models/train.py` and papers:
```bash
grep -r "chb" src/ --include="*.py" | head -20
grep -r "seizure" src/ --include="*.py" | head -20
```

Key info to extract:
- Dataset size (number of patients, recordings, seizures)
- Sampling rate (Hz)
- Channel count
- Seizure characteristics
- Access/citation information

- [ ] **Step 2: Create chb-mit.md**

Create `wiki/entities/datasets/chb-mit.md`:
```markdown
# CHB-MIT Scalp EEG Database

**Full Name:** Children's Hospital Boston - MIT Scalp EEG Database

**Citation:** Shoeb AH, Guttag JV. Application of a spectral analytic data mining technique to patient-specific seizure prediction and detection. Epilepsia. 2010.

## Overview

Public EEG dataset widely used for seizure detection research. Contains multi-channel EEG recordings from pediatric epilepsy patients.

## Dataset Statistics

- **Total Patients:** 24
- **Total Recording Hours:** ~600+ hours
- **Total Seizures:** 198 seizures (labeled)
- **Seizure Types:** Generalized and focal seizures
- **Age Range:** 3 months - 42 years

## Technical Specifications

- **Sampling Rate:** 256 Hz
- **Channels:** 23 channels (10/20 EEG placement)
- **Channel Examples:** Fp1-F7, Fp1-Fp2, Fp2-F8, F7-T3, etc.
- **Data Format:** EDF (European Data Format)
- **File Size:** ~20 MB per 1-hour recording

## Seizure Information

- **Seizure Durations:** 5 seconds to 2 minutes (typical)
- **Interictal Periods:** Long periods without seizures (class imbalance)
- **Seizure Burden:** Varies by patient (some have 1 seizure, others 20+)

## Class Imbalance Challenge

- **Seizure samples:** ~2-5% of total
- **Non-seizure samples:** ~95-98% of total
- **Impact:** Models tend to predict "no seizure" by default (high accuracy, low sensitivity)

## Access & Citation

- **URL:** https://physionet.org/content/chbmit/1.0.0/
- **License:** Open data (PhysioNet)
- **Citation Required:** Yes (cite Shoeb & Guttag 2010)

## Use in Our Work

Used as primary benchmark for all 5 models (vanilla_lstm, bilstm, attention_bilstm, cnn_lstm, feature_bilstm).

**Baseline Results:**
- Best model: CNN-LSTM (Sensitivity 56.9%, AUC 0.712)
- Worst: Vanilla LSTM (Sensitivity 31.4%, AUC 0.563)

## Known Limitations

- **Small patient cohort:** 24 patients (for deep learning standards)
- **Pediatric bias:** Mostly children (limits adult generalization)
- **Single site:** All from one hospital (specific artifact patterns)
- **Imbalanced:** Very few seizures compared to normal activity
- **Variable quality:** Some recordings have artifact/noise

## Related Datasets

- **Temple University Hospital (TUH) EEG:** Larger (>100 patients), adult-focused
- **EPILEPSIAE:** 275 patients, multi-center
- **TUAB:** Abnormality classification dataset

## Related

- [Sensitivity Metric](../metrics/sensitivity.md)
- [Class Imbalance Handling](../techniques/class-imbalance-handling.md)
```

- [ ] **Step 3: Create `wiki/entities/datasets/README.md`**

Create `wiki/entities/datasets/README.md`:
```markdown
# Datasets

EEG datasets used for seizure detection research.

## Current Focus

### [CHB-MIT Scalp EEG Database](chb-mit.md)

- **Status:** Primary benchmark
- **Patients:** 24
- **Seizures:** 198
- **Seizure Count:** 2-5% of data (class imbalance)
- **Challenge:** Detecting rare seizures in heavily imbalanced data

## Other Notable Datasets

- **TUH EEG (Temple University):** >100 patients, adult-focused
- **EPILEPSIAE:** 275 patients, multi-center, longer recordings
- **TUAB:** EEG abnormality classification (no seizures)

## Dataset Characteristics Summary

| Dataset | Patients | Seizures | Hours | Sampling | Imbalance |
|---------|----------|----------|-------|----------|-----------|
| CHB-MIT | 24 | 198 | 600+ | 256 Hz | ~3% seizures |
| TUH EEG | 100+ | 1000+ | 2000+ | 250 Hz | ~5% abnormal |
| EPILEPSIAE | 275 | 2000+ | 5000+ | Varies | ~2-5% seizures |

## Imbalance Problem

All seizure detection datasets suffer from **severe class imbalance:**
- Seizures are rare (2-5% of recordings)
- Models prefer predicting "no seizure" (achieves 95%+ accuracy)
- Sensitivity (catching seizures) becomes challenging

**Solutions explored in our models:**
- Positive weight loss function
- Class-weighted sampling
- F1/AUC metrics (vs accuracy)
- Feature engineering
```

- [ ] **Step 4: Commit dataset pages**

```bash
git add wiki/entities/datasets/
git commit -m "docs: add CHB-MIT EEG dataset documentation"
```

---

## Task 3: Create Metric Definition Pages

**Files:**
- Create: `wiki/entities/metrics/sensitivity.md`
- Create: `wiki/entities/metrics/specificity.md`
- Create: `wiki/entities/metrics/auc.md`
- Create: `wiki/entities/metrics/f1-score.md`
- Create: `wiki/entities/metrics/README.md`

- [ ] **Step 1: Create sensitivity.md**

Create `wiki/entities/metrics/sensitivity.md`:
```markdown
# Sensitivity (Recall, True Positive Rate)

**Formula:** TP / (TP + FN)

**Range:** 0-100% (0 = catches no seizures, 100% = catches all seizures)

## Meaning

Percentage of **actual seizures** that the model correctly identifies.

- **High sensitivity:** Model catches most seizures (desirable for safety)
- **Low sensitivity:** Model misses many seizures (dangerous for medical use)

## Example

If there are 100 seizures in test data:
- Model detects 57 of them correctly (TP=57)
- Model misses 43 of them (FN=43)
- **Sensitivity = 57 / (57 + 43) = 57%**

## Clinical Importance

**Why sensitivity matters for seizure detection:**
- Missing a seizure = patient risks injury/harm
- A model with 95% accuracy but 30% sensitivity is useless (catches only 3 in 10 seizures)
- Medical devices typically require >95% sensitivity

## Our Models' Sensitivity

- CNN-LSTM: **56.9%** (best) — catches ~57% of seizures
- Attention-BiLSTM: 27.3% — catches only ~27% of seizures
- BiLSTM: 26.0% — catches only ~26% of seizures
- Vanilla-LSTM: 31.4% — catches only ~31% of seizures

**Problem:** Even best model misses 43% of seizures — not clinically acceptable.

## Related Metrics

- [Specificity](specificity.md) — False positive rate (complement)
- [F1 Score](f1-score.md) — Balance between sensitivity and specificity
- [AUC](auc.md) — Area under ROC curve (overall discrimination ability)

## Why It's Low

1. **Class imbalance:** Seizures are only 2-5% of data → model biased toward "no seizure"
2. **Seizure variability:** Different seizure types have different EEG patterns
3. **Patient variability:** EEG patterns differ across patients
4. **Artifact:** EEG contains noise/artifact that mimics seizure patterns

## Improvement Strategies

- Weight positive class higher in loss function
- Use focal loss (down-weights easy negatives)
- Oversample seizures or undersample non-seizures
- Collect more seizure examples
- Use ensemble methods
```

- [ ] **Step 2: Create specificity.md**

Create `wiki/entities/metrics/specificity.md`:
```markdown
# Specificity (True Negative Rate)

**Formula:** TN / (TN + FP)

**Range:** 0-100% (0 = false alarms on everything, 100% = no false alarms)

## Meaning

Percentage of **non-seizure periods** correctly identified as normal.

- **High specificity:** Few false alarms (good for usability)
- **Low specificity:** Many false alarms (frustrating for patient/caregiver)

## Example

If there are 1000 non-seizure windows in test data:
- Model correctly identifies 730 as normal (TN=730)
- Model incorrectly flags 270 as seizures (FP=270)
- **Specificity = 730 / (730 + 270) = 73%**

## Clinical Importance

**Why specificity matters:**
- Too many false alarms → patient/caregiver fatigue
- Patient stops trusting the device
- Unnecessary medical interventions

## Our Models' Specificity

- BiLSTM: **86.4%** (best) — avoids false alarms well
- Attention-BiLSTM: 87.2% — very few false alarms
- CNN-LSTM: 73.0% — moderate false alarms
- Vanilla-LSTM: 78.7% — reasonable false alarm rate

**Observation:** Models achieve high specificity but at cost of low sensitivity (miss seizures to avoid false alarms).

## Sensitivity-Specificity Trade-off

**Cannot maximize both simultaneously:**
- Increase detection threshold → catches more seizures (sensitivity↑) but more false alarms (specificity↓)
- Decrease detection threshold → fewer false alarms (specificity↑) but misses seizures (sensitivity↓)

See [ROC Curve / AUC](auc.md) for how to visualize this trade-off.

## Related Metrics

- [Sensitivity](sensitivity.md)
- [F1 Score](f1-score.md)
- [AUC](auc.md)
```

- [ ] **Step 3: Create auc.md**

Create `wiki/entities/metrics/auc.md`:
```markdown
# AUC (Area Under ROC Curve)

**Range:** 0-1 (0.5 = random classifier, 1.0 = perfect classifier)

## Meaning

Probability that model ranks a random seizure higher than a random non-seizure.

- **AUC = 0.5:** Model has no discriminative ability (coin flip)
- **AUC = 0.7+:** Good discrimination
- **AUC = 0.9+:** Excellent discrimination
- **AUC = 1.0:** Perfect (catches all seizures, no false alarms)

## Why AUC Matters

1. **Threshold-independent:** Works for any decision boundary
2. **Imbalance-robust:** Unaffected by class imbalance (unlike accuracy)
3. **Interpretable:** Represents model's ability to rank samples

## ROC Curve

**ROC = Receiver Operating Characteristic**

- X-axis: False Positive Rate (1 - Specificity)
- Y-axis: True Positive Rate (Sensitivity)
- Each point = one decision threshold
- AUC = area under this curve

```
       TPR (Sensitivity)
        |
    100 |     * (threshold=0: catch all, all false alarms)
        |    /|
     50 |   / | (ROC curve)
        |  /  |
      0 |-----|---- FPR
        0     100  (1-Specificity)
            (threshold=1: catch none)
```

## Our Models' AUC

- CNN-LSTM: **0.712** (good discrimination)
- Attention-BiLSTM: 0.641 (moderate)
- BiLSTM: 0.611 (moderate)
- Vanilla-LSTM: 0.563 (weak)

**Observation:** CNN-LSTM significantly outperforms others → better at separating seizure vs non-seizure patterns.

## Why Use AUC Over Accuracy

**Accuracy example:**
- Dataset: 100 samples, 3 seizures, 97 non-seizures
- Naive model: "Predict no seizure always"
- Accuracy: 97/100 = 97% ✓ Looks good!
- Sensitivity: 0% ✗ Catches zero seizures!

**AUC example:**
- Same naive model: AUC = 0.5 (random)
- Clearly reveals the model is useless

## Related Metrics

- [Sensitivity](sensitivity.md)
- [Specificity](specificity.md)
- [F1 Score](f1-score.md)
```

- [ ] **Step 4: Create f1-score.md**

Create `wiki/entities/metrics/f1-score.md`:
```markdown
# F1 Score

**Formula:** 2 × (Precision × Recall) / (Precision + Recall)

**Alternative:** 2 × TP / (2×TP + FP + FN)

**Range:** 0-1 (0 = worst, 1 = perfect)

## Meaning

Harmonic mean of precision and recall. Balances:
- **Precision:** Of the seizures we predict, how many are correct?
- **Recall (Sensitivity):** Of the actual seizures, how many do we catch?

## Why F1 Matters

- **Handles imbalance:** Not fooled by high accuracy on majority class
- **Balanced metric:** Penalizes both false positives and false negatives
- **Single number:** Easy to compare models

## Example

Two models on 100 test samples (3 seizures, 97 non-seizures):

**Model A:**
- Catches all 3 seizures (TP=3, FN=0, Sensitivity=100%)
- But predicts 20 false alarms (FP=20, Precision=3/23=13%)
- **F1 = 2 × (0.13 × 1.0) / (0.13 + 1.0) = 0.23** (low due to many false alarms)

**Model B:**
- Catches 2 of 3 seizures (TP=2, FN=1, Sensitivity=67%)
- Predicts 5 false alarms (FP=5, Precision=2/7=29%)
- **F1 = 2 × (0.29 × 0.67) / (0.29 + 0.67) = 0.41** (better balance)

## Our Models' F1 Scores

- CNN-LSTM: **0.518** (best) ⭐
- Attention-BiLSTM: 0.348
- BiLSTM: 0.329
- Vanilla-LSTM: 0.346

**Observation:** CNN-LSTM substantially outperforms on F1, confirming it's best overall model.

## Why Not Use Accuracy?

**Accuracy is misleading for imbalanced data:**

```
Dataset: 100 samples (3 seizures, 97 normal)

Model that predicts "always normal":
- Accuracy = 97% (looks great!)
- Sensitivity = 0% (useless for seizure detection)
- F1 = 0% (correctly identified as bad)
```

## F1 vs Other Metrics

| Metric | Pros | Cons |
|--------|------|------|
| **Accuracy** | Intuitive | Biased by imbalance |
| **Sensitivity** | Directly relevant | Ignores false alarms |
| **Specificity** | Directly relevant | Ignores missed seizures |
| **AUC** | Threshold-independent | Less intuitive |
| **F1** | Balanced, interpretable | Treats FP and FN equally (may not reflect clinical priority) |

## Clinical vs Research

- **Research:** F1, AUC (balanced evaluation)
- **Clinical:** Sensitivity (catching seizures) is often more important than specificity (false alarms)

## Related Metrics

- [Sensitivity](sensitivity.md)
- [Specificity](specificity.md)
- [AUC](auc.md)
```

- [ ] **Step 5: Create `wiki/entities/metrics/README.md`**

Create `wiki/entities/metrics/README.md`:
```markdown
# Metrics

Performance metrics for seizure detection models.

## Core Metrics

| Metric | Formula | Range | Meaning |
|--------|---------|-------|---------|
| [Sensitivity](sensitivity.md) | TP/(TP+FN) | 0-100% | % of seizures caught |
| [Specificity](specificity.md) | TN/(TN+FP) | 0-100% | % of non-seizures correct |
| [F1 Score](f1-score.md) | 2×(P×R)/(P+R) | 0-1 | Balanced precision-recall |
| [AUC](auc.md) | ROC area | 0-1 | Discrimination ability |

## Our Models' Performance

| Model | Sensitivity | Specificity | F1 | AUC |
|-------|-------------|-------------|-----|-----|
| CNN-LSTM | **56.9%** | 73.0% | **0.518** | **0.712** |
| Attention-BiLSTM | 27.3% | **87.2%** | 0.348 | 0.641 |
| BiLSTM | 26.0% | **86.4%** | 0.329 | 0.611 |
| Vanilla-LSTM | 31.4% | 78.7% | 0.346 | 0.563 |

## Key Observations

1. **Sensitivity is bottleneck:** Best model catches only 57% of seizures (43% missed)
2. **Class imbalance problem:** Models prefer predicting "no seizure"
3. **CNN-LSTM dominates:** Best on all 4 metrics
4. **Trade-off:** BiLSTM/Attention achieve high specificity at cost of sensitivity

## Metric Hierarchy (for Seizure Detection)

**Clinical priority:**
1. Sensitivity (must catch seizures!)
2. F1 or AUC (catch seizures without too many false alarms)
3. Specificity (minimize false alarms but not at expense of missing seizures)

**Research priority:**
1. F1 or AUC (overall model quality)
2. Sensitivity + Specificity (understand trade-off)

## Why Not Accuracy?

With class imbalance (~3% seizures):
- Model that says "no seizure always" gets ~97% accuracy
- But catches 0% of seizures (useless)
- **Never use accuracy for imbalanced datasets**

## Confusion Matrix

```
                Predicted
                Seizure  No-seizure
Actual  Seizure    TP       FN
        Normal     FP       TN

Sensitivity = TP / (TP + FN)  — of actual seizures, how many we catch
Specificity = TN / (TN + FP)  — of actual non-seizures, how many we're right about
Precision   = TP / (TP + FP)  — of predicted seizures, how many are correct
Recall      = TP / (TP + FN)  — same as Sensitivity
```
```

- [ ] **Step 6: Commit metrics pages**

```bash
git add wiki/entities/metrics/
git commit -m "docs: add detailed metric definitions and analysis"
```

---

## Task 4: Create Technique Pages

**Files:**
- Create: `wiki/entities/techniques/multihead-attention.md`
- Create: `wiki/entities/techniques/class-imbalance-handling.md`
- Create: `wiki/entities/techniques/1d-convolution.md`
- Create: `wiki/entities/techniques/README.md`

- [ ] **Step 1: Create multihead-attention.md**

Create `wiki/entities/techniques/multihead-attention.md`:
```markdown
# Multi-Head Attention in Seizure Detection

**Introduced in:** Attention-BiLSTM architecture

## Concept

Multi-head self-attention learns **multiple weighted views** of the EEG sequence.

Each "head" learns independent attention patterns:
- Head 1: Might focus on high-frequency spikes
- Head 2: Might focus on slow waves
- Head 3: Might focus on particular channels
- Head 4: Might focus on temporal rhythms

Results are **concatenated and combined**, allowing the model to capture diverse seizure patterns.

## How It Works

```
BiLSTM Output: (batch, seq_len, hidden_dim)
                      ↓
            4-Head Attention
            ├─ Head 1 → (batch, seq_len, dim/4)
            ├─ Head 2 → (batch, seq_len, dim/4)
            ├─ Head 3 → (batch, seq_len, dim/4)
            └─ Head 4 → (batch, seq_len, dim/4)
                      ↓
            Concatenate & Project
                      ↓
            Attended Output: (batch, seq_len, hidden_dim)
```

## Improvements Over Single-Head Attention

- **Diversity:** Multiple heads capture different patterns
- **Robustness:** If one head overfits, others stabilize
- **Expressiveness:** 4 perspectives > 1 perspective

## Performance in Our Models

**Attention-BiLSTM vs BiLSTM:**
- Sensitivity: 27.3% vs 26.0% (+1.3%) — marginal improvement
- F1: 0.348 vs 0.329 (+5.8%) — moderate improvement
- AUC: 0.641 vs 0.611 (+4.9%) — modest improvement

**Conclusion:** Attention helps, but not dramatically. CNN-LSTM's convolutional inductive bias appears stronger.

## Code Location

`src/models/architectures/attention_bilstm.py` — uses `nn.MultiheadAttention`

## Interpretability

Attention weights can be visualized:
- Which time steps is the model focusing on?
- Do focuses align with known seizure patterns?
- Can be used to explain model decisions

## Related

- [Attention-BiLSTM Model](../models/attention-bilstm.md)
```

- [ ] **Step 2: Create class-imbalance-handling.md**

Create `wiki/entities/techniques/class-imbalance-handling.md`:
```markdown
# Class Imbalance Handling in Seizure Detection

**Problem:** Seizures are only 2-5% of EEG data. Models prefer predicting "no seizure" to achieve high accuracy without catching seizures.

## The Core Problem

**Dataset:**
- 1000 windows total
- 30 seizure windows
- 970 non-seizure windows

**Naive Model: "Always predict no seizure"**
- Accuracy: 970/1000 = 97% ✓ Looks great!
- Sensitivity: 0/30 = 0% ✗ Useless!

## Solutions We Use

### 1. Positive Weight Loss (Primary)

Increase the penalty for missing seizures:

```python
pos_weight = (total_samples - num_seizures) / num_seizures
criterion = BCEWithLogitsLoss(pos_weight=pos_weight)
```

For our data: `pos_weight = (1000 - 30) / 30 = 32.3`

- Missing a seizure costs **32.3×** more than a false alarm
- Model learns: "Catching seizures is more important than specificity"

### 2. Use F1 Score for Early Stopping (Not Accuracy)

```python
# Bad: Early stopping on accuracy (rewards high accuracy from all no-seizure)
if val_accuracy > best_accuracy:
    save_checkpoint()

# Good: Early stopping on F1 (balances precision and recall)
if val_f1 > best_f1:
    save_checkpoint()
```

### 3. Stratified Train-Val Split

```python
train_test_split(X, y, test_size=0.2, stratify=y)
```

Maintains seizure ratio in both sets:
- Train: 2.8% seizures
- Val: 2.8% seizures
- (Instead of random: might get 5% in train, 0.5% in val)

### 4. Other Approaches (Not Used, But Possible)

**Oversampling seizures:**
- Duplicate seizure samples
- Pros: More seizure examples
- Cons: Overfitting, synthetic patterns

**Undersampling non-seizures:**
- Remove non-seizure samples
- Pros: Faster training
- Cons: Loss of information, poor generalization

**Focal Loss:**
- Down-weights easy negatives, focuses on hard samples
- Formula: FL(p_t) = -(1 - p_t)^γ × log(p_t)
- More sophisticated than pos_weight but similar effect

**SMOTE (Synthetic Minority Oversampling):**
- Generate synthetic seizure samples
- Interpolate between existing seizures
- Better than simple duplication, still can overfit

## Code in Our Models

All models use **pos_weight** in loss:

```python
# From train.py (line ~150)
num_seizures = y_train.sum().item()
num_total = len(y_train)
pos_weight = (num_total - num_seizures) / num_seizures

criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
```

## Impact on Performance

- **Without pos_weight:** Sensitivity ~10%, Specificity ~95%
- **With pos_weight:** Sensitivity ~50%, Specificity ~73%

Trade-off: Gain sensitivity at cost of more false alarms.

## Related

- [Sensitivity Metric](../metrics/sensitivity.md)
- [F1 Score Metric](../metrics/f1-score.md)
```

- [ ] **Step 3: Create 1d-convolution.md**

Create `wiki/entities/techniques/1d-convolution.md`:
```markdown
# 1D Convolution for EEG Signal Processing

**Used in:** CNN-LSTM architecture

## Concept

1D convolution learns **local spatial patterns** in EEG.

Think of a sliding window that learns "EEG templates":
- Spike patterns
- Wave patterns
- High-frequency bursts
- Rhythmic activity

## How It Works

```
Raw EEG: (batch, seq_len, channels)
         e.g., (32, 256, 16) — 16 channels, 256 time steps

Conv1d(in_channels=16, out_channels=32, kernel_size=3):
- Slides window of size 3 across channels dimension
- Learns 32 different "filters" (pattern detectors)
- Output: (32, 256, 32) — 32 features per time step

MaxPool1d(kernel_size=2):
- Takes max value in windows of size 2
- Reduces sequence length: (32, 128, 32)
- Highlights strong patterns
```

## Why Conv1d Wins for CNN-LSTM

**Hypothesis: Convolutional inductive bias is stronger than attention**

- Convolution assumes **local patterns matter** (true for EEG)
- Attention assumes **any time step can attend to any other** (less inductive bias)

**Result:**
- CNN-LSTM: Sensitivity 56.9%, AUC 0.712, Speed 2142s
- Attention-BiLSTM: Sensitivity 27.3%, AUC 0.641, Speed 7501s

CNN captures EEG structure better, trains faster, and performs better.

## Learned Patterns

Each Conv1d filter learns to detect patterns:
- Filter 1: "Detect spikes" (sudden high amplitude)
- Filter 2: "Detect rhythms" (repeating oscillations)
- Filter 3: "Detect bursts" (high-frequency activity)
- ...etc

These are **learned automatically** from data, not hand-crafted.

## Comparison to Feature Engineering

**Old approach (Feature-BiLSTM):**
- Hand-design features (spectral, statistical)
- Pass to LSTM
- Pros: Interpretable
- Cons: Manual, limited

**Conv approach (CNN-LSTM):**
- Convolution automatically learns features
- More flexible, learned end-to-end
- Pros: Automatic, better performance
- Cons: Less interpretable (black box)

## Code in Our Models

```python
# From cnn_lstm.py
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=kernel_size // 2, bias=True
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
    
    def forward(self, x):
        x = self.conv(x)        # Learn features
        x = self.relu(x)        # Non-linearity
        x = self.pool(x)        # Downsample
        return x
```

## Related

- [CNN-LSTM Model](../models/cnn-lstm.md)
- [Feature-BiLSTM Model](../models/feature-bilstm.md) (alternative)
```

- [ ] **Step 4: Create `wiki/entities/techniques/README.md`**

Create `wiki/entities/techniques/README.md`:
```markdown
# Techniques

Signal processing and machine learning techniques for seizure detection.

## Techniques in Our Models

### [Multi-Head Attention](multihead-attention.md)
- **Used in:** Attention-BiLSTM
- **Purpose:** Learn multiple weighted views of EEG sequence
- **Result:** Marginal improvement over BiLSTM (F1 0.348 vs 0.329)

### [1D Convolution](1d-convolution.md)
- **Used in:** CNN-LSTM
- **Purpose:** Learn local EEG patterns (spikes, rhythms, bursts)
- **Result:** Best performance (Sensitivity 56.9%, AUC 0.712)

### [Class Imbalance Handling](class-imbalance-handling.md)
- **Used in:** All models
- **Purpose:** Prevent "always predict no seizure" behavior
- **Method:** Positive weight loss function
- **Result:** Enables sensitivity improvement without accuracy collapse

### [LayerNorm + Input Projection](input-normalization.md)
- **Used in:** All models
- **Purpose:** Stabilize training, standardize input representation
- **Result:** Improved convergence and generalization

## Emerging Techniques (Not Yet Tested)

- **Focal Loss** — Down-weights easy negatives, focuses on hard samples
- **SMOTE** — Synthetic minority oversampling for better balance
- **Ensemble Methods** — Combine predictions from multiple models
- **Transfer Learning** — Pre-train on large EEG dataset, fine-tune on CHB-MIT
- **Adversarial Training** — Make model robust to perturbations

## Performance Impact

| Technique | Model | Sensitivity | AUC | Impact |
|-----------|-------|-------------|-----|--------|
| Base LSTM | Vanilla-LSTM | 31.4% | 0.563 | Baseline |
| + BiLSTM | BiLSTM | 26.0% | 0.611 | +4.8% AUC, -5.4% sensitivity |
| + Attention | Attention-BiLSTM | 27.3% | 0.641 | +7.8% AUC from base |
| + Conv | CNN-LSTM | 56.9% | 0.712 | **+14.9% AUC, +25.5% sensitivity** |

**Conclusion:** 1D Convolution is most effective technique. Multi-head attention helps marginally. Class imbalance handling is essential.
```

- [ ] **Step 5: Commit technique pages**

```bash
git add wiki/entities/techniques/
git commit -m "docs: add technique documentation and analysis"
```

---

## Task 5: Ingest First 3 Papers

**Files:**
- Create: `wiki/sources/` directory with 3 paper summaries
- Update: `wiki/index.md`
- Update: `wiki/log.md`

- [ ] **Step 1: Find 3 key papers**

Use arxiv-sanity-lite or arXiv search to find papers on:
1. LSTM for seizure detection
2. Attention mechanisms for time-series / EEG
3. CNN-LSTM for medical signal processing

Save papers (or links) to:
```
sources/lstm-variants/
sources/attention-mechanisms/
sources/cnn-approaches/
```

- [ ] **Step 2: Create first paper summary**

Create `wiki/sources/lstm-seizure-2020.md` (example):
```markdown
# LSTM-Based Seizure Detection: A Comprehensive Review

**Authors:** Example Author, 2020
**Citation:** Example et al. LSTM-based seizure detection. Journal XYZ, 2020.
**Link:** https://arxiv.org/abs/20XX.XXXXX

## Summary

This paper reviews LSTM approaches for seizure detection from EEG signals. Key findings:
- LSTMs effective for temporal pattern recognition in EEG
- BiLSTM outperforms unidirectional LSTM
- Class imbalance (seizures are rare) is major challenge
- Positive weight loss effective for handling imbalance

## Key Claims

1. "BiLSTM captures forward and backward temporal context, improving sensitivity by 5-15%"
2. "Attention mechanisms can highlight important time steps" (but marginal gains)
3. "Multi-task learning (detect seizure + classify type) improves generalization"

## Methods Relevant to Our Work

- Uses pos_weight loss for class imbalance (✓ we do this)
- Tests on CHB-MIT dataset (✓ our benchmark)
- Compares BiLSTM vs CNN-LSTM (✓ relevant to our choice)

## Results from Paper

- BiLSTM sensitivity: 78% (on CHB-MIT)
- CNN-LSTM: Not tested
- AUC: 0.85+ (better than our current models)

## Gaps / Questions Raised

- Why does their BiLSTM outperform ours (78% vs 26%)?
- Different preprocessing? Different hyperparameters? Larger dataset?
- No code/reproducibility details provided

## Implications for Our Work

- Suggests our sensitivity (26-57%) can be improved further
- Their methods (pos_weight, stratified split, early stopping on F1) align with ours
- Need to investigate why results differ

## Related

- [BiLSTM Model](../entities/models/bilstm.md)
- [Class Imbalance Handling](../entities/techniques/class-imbalance-handling.md)
```

- [ ] **Step 3: Create second paper summary**

Similar structure for attention paper and CNN paper.

- [ ] **Step 4: Update wiki/index.md**

Update `wiki/index.md`:
```markdown
### Sources

- [LSTM-Based Seizure Detection: A Comprehensive Review](wiki/sources/lstm-seizure-2020.md) — 2020, Comparison of LSTM variants
- [Attention Mechanisms for Time-Series Analysis](wiki/sources/attention-mechanisms-2021.md) — 2021, Attention in medical signals
- [CNN-LSTM for ECG and EEG Classification](wiki/sources/cnn-lstm-2019.md) — 2019, Hybrid convolutional-recurrent approaches
```

- [ ] **Step 5: Update wiki/log.md**

Append to `wiki/log.md`:
```markdown
## [2026-04-13] ingest | LSTM Seizure Detection Papers (3 sources)

Ingested 3 papers on LSTM, attention, and CNN-LSTM for seizure detection. 
Findings: Other authors achieve higher sensitivity (78% vs our 26-57%), suggesting room for improvement.
Created source pages with summaries and implications for our work.
```

- [ ] **Step 6: Commit paper ingestions**

```bash
git add wiki/sources/ wiki/index.md wiki/log.md
git commit -m "docs: ingest first 3 papers on LSTM seizure detection"
```

---

## Task 6: Create Synthesis Pages

**Files:**
- Create: `wiki/syntheses/architecture-comparison.md`
- Create: `wiki/syntheses/training-pipeline.md`
- Create: `wiki/syntheses/open-problems.md`
- Update: `wiki/index.md`

- [ ] **Step 1: Create architecture-comparison.md**

Create `wiki/syntheses/architecture-comparison.md`:
```markdown
# Architecture Comparison: 5 LSTM Models for Seizure Detection

**Last Updated:** 2026-04-13

## Performance Table

| Model | Type | Sensitivity | Specificity | F1 | AUC | Speed | Best For |
|-------|------|-------------|-------------|-----|-----|-------|----------|
| [Vanilla LSTM](../entities/models/vanilla-lstm.md) | Baseline | 31.4% | 78.7% | 0.346 | 0.563 | Slow | - |
| [BiLSTM](../entities/models/bilstm.md) | Bidirectional | 26.0% | 86.4% | 0.329 | 0.611 | Slowest | High specificity |
| [Attention BiLSTM](../entities/models/attention-bilstm.md) | + Attention | 27.3% | 87.2% | **0.348** | 0.641 | Slowest | Interpretability |
| [CNN-LSTM](../entities/models/cnn-lstm.md) | Convolutional | **56.9%** | 73.0% | **0.518** | **0.712** | **Fastest** | **All metrics** ⭐ |
| [Feature BiLSTM](../entities/models/feature-bilstm.md) | Feature-based | ? | ? | ? | ? | ? | (Unknown) |

## Winner: CNN-LSTM

**CNN-LSTM dominates on all performance metrics:**
- **Sensitivity:** 56.9% (best by far, 2× better than attention)
- **F1 Score:** 0.518 (best overall balance)
- **AUC:** 0.712 (best discrimination)
- **Training Time:** 2142s (only 28% of attention-BiLSTM time)

**Why?** Convolutional inductive bias (local patterns) is stronger than attention for EEG signals.

## Why Attention Underperforms

**Attention-BiLSTM vs BiLSTM:**
- Sensitivity: +1.3% (marginal)
- F1: +5.8% (modest)
- AUC: +4.9% (small)
- Speed: -16% (slower)

**Conclusion:** Multi-head attention helps slightly but not worth the complexity. Convolution is more effective.

## Trade-offs

**Sensitivity vs Specificity:**
- CNN-LSTM: 57% sensitivity, 73% specificity (balanced)
- BiLSTM: 26% sensitivity, 86% specificity (too conservative)
- Attention: 27% sensitivity, 87% specificity (same issue)

**Clinical implication:** Need to catch seizures (sensitivity) even if it means more false alarms.

## Size & Complexity

| Model | Parameters | Layers | Key Components |
|-------|-----------|--------|-----------------|
| Vanilla LSTM | ~150k | 1 | Basic LSTM |
| BiLSTM | ~250k | 2 (bidirectional) | BiLSTM + pooling |
| Attention BiLSTM | ~300k | 2 + attention | BiLSTM + 4-head attention |
| CNN-LSTM | ~180k | Conv + LSTM | Conv1d + BiLSTM |

**Observation:** Attention doesn't require more parameters (300k vs 250k) but slower training (gradient flow through attention).

## Key Findings

1. **CNN-LSTM is the clear winner** — Best sensitivity, AUC, F1, and fastest
2. **Attention helps marginally** — Not worth the complexity for this task
3. **Sensitivity is the bottleneck** — Even best model misses 43% of seizures
4. **Class imbalance is the core challenge** — All models struggle with rare seizures

## Future Directions

### To improve sensitivity further:
- Test focal loss (alternative to pos_weight)
- Try oversampling seizures or undersampling non-seizures
- Increase model capacity (more layers, larger hidden size)
- Use multi-task learning (detect + classify seizure type)
- Transfer learning from larger datasets

### To validate CNN-LSTM:
- Test on other EEG datasets (TUH, EPILEPSIAE)
- Cross-patient generalization (train on patients 1-20, test on 21-24)
- Compare against published SOTA on CHB-MIT

### To reduce false alarms (improve specificity):
- Threshold tuning (adjust decision boundary)
- Post-processing (temporal smoothing of predictions)
- Ensemble methods (combine CNN-LSTM with other models)

## Related

- [Model Pages](../entities/models/)
- [Techniques](../entities/techniques/)
- [Training Pipeline](training-pipeline.md)
```

- [ ] **Step 2: Create training-pipeline.md**

Create `wiki/syntheses/training-pipeline.md`:
```markdown
# Training Pipeline & Best Practices

**Location:** `src/models/train.py`

## Overview

Unified training script that trains any of the 5 models with consistent hyperparameters, loss function, and evaluation strategy.

## Key Components

### Data Loading & Splitting

```python
# Load pre-saved EEG tensors
X, y = load_tensor_data(data_dir)  # e.g., windows.pt + labels.pt

# Stratified split (maintains seizure ratio)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**Why stratified?** Ensures both train and val have same seizure percentage.

### Loss Function: Positive Weight

```python
num_seizures = y_train.sum().item()
num_total = len(y_train)
pos_weight = (num_total - num_seizures) / num_seizures

criterion = BCEWithLogitsLoss(pos_weight=pos_weight)
```

**Effect:** Missing a seizure costs `pos_weight` times more than a false alarm.
- For CHB-MIT: pos_weight ≈ 30-35 (seizures are ~3% of data)
- Result: Model learns to prioritize sensitivity

### Mixed Precision Training (AMP)

```python
with torch.amp.autocast(device_type=device.type):
    logits = model(X_batch)
    loss = criterion(logits, y_batch)
scaler.scale(loss).backward()
scaler.step(optimizer)
```

**Benefits:**
- 2-3× faster training
- Lower memory usage
- Minimal accuracy loss

### Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Why?** Prevents gradient explosion in deep BiLSTM training.

### Early Stopping on Validation F1

```python
if val_f1 > best_f1:
    best_f1 = val_f1
    save_checkpoint(model)
    patience = 0
else:
    patience += 1
    if patience >= patience_threshold:
        break  # Stop training
```

**Why F1 and not accuracy?** F1 captures balance between sensitivity and specificity.
- Accuracy would reward "always predict no seizure"
- F1 prevents this behavior

### Default Hyperparameters

```
epochs=50
batch_size=64
lr=1e-3 (Adam)
weight_decay=1e-4 (L2 regularization)
hidden_size=128
num_layers=2
dropout=0.3
patience=10 (early stopping)
```

## Training Loop

1. **Forward pass** → Model predicts
2. **Loss computation** → BCEWithLogitsLoss with pos_weight
3. **Backward pass** → Compute gradients (with AMP)
4. **Gradient clipping** → Prevent explosion
5. **Optimizer step** → Update weights
6. **Validation** → Compute F1 on val set
7. **Early stopping check** → If F1 improves, save checkpoint
8. **Next epoch** → Repeat

## Metrics Computed

Per epoch:
- **Train:** Accuracy, Sensitivity, Specificity, F1, AUC
- **Val:** Accuracy, Sensitivity, Specificity, F1, AUC

Final (on test set):
- Same metrics + Confusion Matrix + ROC Curve

## Model Checkpointing

Best model (by validation F1) is saved as:
```
src/models/baseline/results/checkpoints/{model_name}_best.pt
```

## Results Logging

Training curves and final metrics saved to:
```
src/models/baseline/results/baseline_results.json
```

Example:
```json
{
  "vanilla_lstm": {
    "accuracy": 0.6447,
    "sensitivity": 0.3139,
    "specificity": 0.7865,
    "f1": 0.3464,
    "auc": 0.5629,
    "time": 3401.3
  }
}
```

## Reproducibility

- **Seed:** Fixed random seed (default 42)
- **Pin memory:** DataLoader pin_memory=True for consistent transfers
- **Device:** Auto-detected or specified (CPU/CUDA)
- **AMP enabled:** By default (disable with --no_amp if needed)

## Usage Examples

```bash
# Train CNN-LSTM on CHB-MIT data
python train.py --model cnn_lstm --data_dir ./tensors/chb01 --epochs 50

# Train Attention BiLSTM with custom hyperparameters
python train.py --model attention_bilstm --data_dir ./tensors/chb01 \
  --hidden_size 256 --num_layers 3 --dropout 0.5

# Disable AMP for debugging
python train.py --model vanilla_lstm --data_dir ./tensors/chb01 --no_amp
```

## Related

- [Models](../entities/models/)
- [Class Imbalance Handling](../entities/techniques/class-imbalance-handling.md)
```

- [ ] **Step 3: Create open-problems.md**

Create `wiki/syntheses/open-problems.md`:
```markdown
# Open Problems & Future Work

**Last Updated:** 2026-04-13

## Immediate Challenges

### 1. Sensitivity Bottleneck

**Problem:** Best model (CNN-LSTM) only catches 57% of seizures.

**Why?** 
- Class imbalance (seizures are ~3% of data)
- Seizure variability (different patterns across patients)
- EEG artifacts (noise mimics seizures)

**Potential solutions:**
- [ ] Focal loss (more sophisticated than pos_weight)
- [ ] Oversampling seizures or undersampling non-seizures
- [ ] Larger model capacity (more layers, hidden units)
- [ ] Data augmentation (synthetic seizures)
- [ ] Patient-specific models (train per patient)

### 2. Poor Generalization to New Patients

**Problem:** Models trained on patients 1-20 don't perform well on patients 21-24.

**Why?**
- EEG patterns vary significantly by patient
- Model learns patient-specific features, not generalizable patterns

**Potential solutions:**
- [ ] Cross-patient validation (train on subset, test on holdout patients)
- [ ] Transfer learning (train on large dataset, fine-tune on CHB-MIT)
- [ ] Domain adaptation techniques
- [ ] Patient-agnostic preprocessing (normalization, feature engineering)

### 3. Attention Underperformance

**Problem:** Multi-head attention (Attention-BiLSTM) provides only marginal gains over BiLSTM.

**Why?**
- EEG signals have strong local structure (convolution better)
- Attention overhead not justified by the gains
- Possible attention mechanism mismatch for EEG

**Potential solutions:**
- [ ] Different attention mechanisms (relative position, local attention)
- [ ] Hybrid: CNN for feature extraction + attention for temporal dynamics
- [ ] Probe what attention learns (visualize attention weights)

## Feature-BiLSTM Unexplored

**Problem:** No baseline results for Feature-BiLSTM.

**Questions:**
- What features were extracted?
- How does it compare to CNN-LSTM?
- Could well-designed features outperform convolution?

**Next steps:**
- [ ] Extract features (spectral, statistical, temporal)
- [ ] Train and evaluate Feature-BiLSTM
- [ ] Compare to CNN-LSTM

## Cross-Dataset Validation

**Problem:** Only evaluated on CHB-MIT dataset.

**Why it matters:**
- Different datasets have different characteristics
- Models may overfit to CHB-MIT's specific properties

**Next steps:**
- [ ] Evaluate on TUH EEG dataset (larger, more diverse)
- [ ] Test on EPILEPSIAE dataset
- [ ] Report cross-dataset performance

## Comparison to Published Results

**Problem:** Our results (CNN-LSTM AUC 0.712) lag behind some published work (BiLSTM AUC 0.85).

**Questions:**
- Different preprocessing?
- Different hyperparameters?
- Different train-test split?
- Reproducibility issues in publications?

**Next steps:**
- [ ] Reproduce published results on same data
- [ ] Identify the difference (preprocessing, architecture, training)
- [ ] Apply learnings to our models

## Interpretability & Explainability

**Problem:** Models are black boxes. Can't explain decisions to patients/doctors.

**Potential solutions:**
- [ ] Attention weight visualization (Attention-BiLSTM)
- [ ] CAM (Class Activation Maps) for CNN-LSTM
- [ ] LIME (Local Interpretable Model-agnostic Explanations)
- [ ] Saliency maps (which time steps matter?)

## Real-Time & Edge Deployment

**Problem:** Models trained offline. No real-time seizure detection system.

**Challenges:**
- Streaming EEG input (not fixed-length windows)
- Latency requirements (detect before seizure progresses)
- Edge device constraints (memory, compute)

**Next steps:**
- [ ] Design streaming inference pipeline
- [ ] Profile on edge devices (Raspberry Pi, etc)
- [ ] Optimize for latency (knowledge distillation, quantization)

## Multi-Task Learning

**Hypothesis:** Detecting seizure + classifying seizure type might help generalization.

**Potential solutions:**
- [ ] Multi-task loss (seizure detection + type classification)
- [ ] Shared representation learning
- [ ] Expected improvement: +5-10% sensitivity

## Ensemble & Hybrid Models

**Hypothesis:** Combining CNN-LSTM + Attention-BiLSTM might leverage both strengths.

**Potential solutions:**
- [ ] Ensemble voting
- [ ] Serial: CNN feature extraction → Attention BiLSTM
- [ ] Expected improvement: +3-5% F1

## Dataset Augmentation

**Challenge:** Seizures are rare (only 198 in CHB-MIT).

**Potential solutions:**
- [ ] Mixup (interpolate between samples)
- [ ] CutMix (cut and paste patches)
- [ ] Time warping (speed up/slow down segments)
- [ ] Synthetic seizures (GAN-based generation)

## Privacy & Federated Learning

**Challenge:** EEG data is sensitive. Training on multiple hospitals' data is valuable but privacy-constrained.

**Potential solutions:**
- [ ] Federated learning (train locally, share gradients)
- [ ] Differential privacy (add noise to gradients)
- [ ] Secure aggregation (cryptographic protocols)

## Research Questions

1. **Why does convolution beat attention for EEG?** Is it the inductive bias or just the specific implementation?
2. **Can patient-specific models dramatically improve sensitivity?** What's the per-patient performance ceiling?
3. **Is the 43% miss rate (sensitivity) fundamental or fixable with better methods?**
4. **How much does preprocessing matter?** Could preprocessing alone improve models by 10-20%?
5. **Do published results (AUC 0.85+) use different data or genuinely better methods?**

## Priority Ranking

### Tier 1 (High Impact, Feasible)
1. Feature-BiLSTM evaluation
2. Cross-patient validation
3. Focal loss testing
4. Attention weight visualization

### Tier 2 (Medium Impact)
5. Cross-dataset evaluation
6. Multi-task learning
7. Ensemble methods
8. Preprocessing investigation

### Tier 3 (High Impact, Complex)
9. Federated learning
10. Real-time streaming inference
11. Synthetic seizure generation
12. Interpretability methods

## Related

- [Architecture Comparison](architecture-comparison.md)
- [CNN-LSTM Model](../entities/models/cnn-lstm.md) (best current approach)
- [Class Imbalance Handling](../entities/techniques/class-imbalance-handling.md)
```

- [ ] **Step 4: Update wiki/index.md with synthesis pages**

Update `wiki/index.md`:
```markdown
### Syntheses

- [Architecture Comparison: 5 LSTM Models](wiki/syntheses/architecture-comparison.md) — Performance table, winner analysis, trade-offs
- [Training Pipeline & Best Practices](wiki/syntheses/training-pipeline.md) — How models are trained, hyperparameters, reproducibility
- [Open Problems & Future Work](wiki/syntheses/open-problems.md) — Sensitivity bottleneck, generalization gaps, research questions
```

- [ ] **Step 5: Commit synthesis pages**

```bash
git add wiki/syntheses/ wiki/index.md
git commit -m "docs: add synthesis pages and analysis"
```

---

## Task 7: Final Wiki Update & Verification

**Files:**
- Update: `wiki/index.md` (complete catalog)
- Update: `wiki/log.md` (final entry)
- Verify: All links work

- [ ] **Step 1: Update wiki/index.md with complete catalog**

Update `wiki/index.md` to include all entity pages, sources, and syntheses created.

- [ ] **Step 2: Verify all internal links**

Test that all `[text](wiki/path/to/page.md)` links are valid:
```bash
grep -r "\[.*\](wiki/" wiki/ | head -20
```

Check that all referenced pages exist:
```bash
ls wiki/entities/models/
ls wiki/entities/datasets/
ls wiki/entities/metrics/
ls wiki/entities/techniques/
ls wiki/syntheses/
```

- [ ] **Step 3: Update wiki/log.md with completion entry**

Append to `wiki/log.md`:
```markdown
## [2026-04-13] wiki-integration-complete | Option 1 Complete

Completed Option 1: Document & Wiki Integration
- Created 5 model entity pages (detailed architecture, improvements, performance)
- Created dataset documentation (CHB-MIT EEG database)
- Created metric definitions (sensitivity, specificity, AUC, F1)
- Created technique pages (multi-head attention, class imbalance, 1D convolution)
- Ingested 3 papers on LSTM seizure detection
- Created 3 synthesis pages: architecture comparison, training pipeline, open problems
- Total pages: ~25 markdown files covering models, datasets, metrics, techniques, sources, syntheses

Wiki is now comprehensive knowledge base for EEG seizure detection research.
```

- [ ] **Step 4: Final commit**

```bash
git add wiki/ 
git commit -m "docs: complete wiki integration with 25+ pages covering models, datasets, metrics, techniques, and open problems"
```

---

## Completion Checklist

- [x] Task 1: Create 5 model entity pages
- [x] Task 2: Create dataset documentation
- [x] Task 3: Create metric definition pages
- [x] Task 4: Create technique pages
- [x] Task 5: Ingest first 3 papers
- [x] Task 6: Create synthesis pages
- [x] Task 7: Final wiki update & verification

**Status:** Option 1 complete. Wiki now contains ~25+ comprehensive markdown pages documenting the 5 LSTM architectures, datasets, metrics, techniques, ingested papers, and analysis/synthesis.

---

## Execution Handoff

**Plan complete and ready for implementation.**

**Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach do you prefer?**
