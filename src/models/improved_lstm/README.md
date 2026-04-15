# Improved LSTM Models for EEG Seizure Detection

Enhanced training pipeline for existing LSTM architectures with all recommended improvements.

## Improvements Implemented

### 1. Data-Level Improvements
| Technique | Implementation | Expected Gain |
|-----------|---------------|---------------|
| **Time Shifting** | Random circular shift ±20 samples | +2-3% F1 |
| **Noise Injection** | Gaussian noise (σ=0.1 × signal std) | +2-3% F1 |
| **Channel Dropout** | Random channel zeroing (p=0.1) | +1-2% F1 |
| **Amplitude Scaling** | Random scale 0.8-1.2× | +1-2% F1 |
| **MixUp** | Convex combination (α=0.2) | +2-3% F1 |

### 2. Architecture Improvements
| Parameter | Baseline | Improved |
|-----------|----------|----------|
| Hidden Size | 128 | **256** |
| Num Layers | 2 | **3** |
| Dropout | 0.3 | **0.4** |

### 3. Training Improvements
| Parameter | Baseline | Improved |
|-----------|----------|----------|
| Epochs | 20-50 | **100** |
| Batch Size | 64 | **32** |
| Learning Rate | 1e-3 | **5e-4** |
| Label Smoothing | 0.02 | **0.1** |
| Scheduler | Cosine Annealing | **Warmup (5 epochs) + Cosine** |
| Loss | BCE | **Focal Loss (γ=1.0)** |
| Early Stopping | 7 epochs | **15 epochs** |

### 4. Ensemble
- **Weighted Averaging**: Weights based on validation F1
- **Voting**: Majority voting with threshold
- **Stacking**: Meta-learner trained on base predictions

## Usage

### Train Single Model
```bash
cd src/models/improved_lstm

# Train with improved settings
python train.py --data_path ../../results/tensors/chbmit --model cnn_lstm

# Custom configuration
python train.py --data_path ../../results/tensors/chbmit \
    --model bilstm \
    --epochs 150 \
    --hidden_size 512 \
    --num_layers 4
```

### Train Ensemble
```bash
# Train all models and create ensemble
python train.py --data_path ../../results/tensors/chbmit --ensemble

# Or equivalently
python train.py --data_path ../../results/tensors/chbmit --model all
```

### Without Augmentation
```bash
python train.py --data_path ../../results/tensors/chbmit --model all --no_augment
```

## Expected Results

| Model | Baseline F1 | Improved F1 | Gain |
|-------|-------------|-------------|------|
| vanilla_lstm | 0.52 | ~0.58-0.62 | +10-15% |
| bilstm | 0.51 | ~0.57-0.61 | +10-15% |
| attention_bilstm | 0.00* | ~0.55-0.60 | Fixed |
| cnn_lstm | 0.54 | ~0.60-0.65 | +10-15% |
| **Ensemble** | N/A | ~0.65-0.70 | +20-25% |

*attention_bilstm was broken due to downsampling, now fixed

## Files

- `__init__.py` - Imports existing architectures + new components
- `augmentation.py` - EEG data augmentation (time shift, noise, channel dropout, MixUp)
- `ensemble.py` - Ensemble prediction (weighted, voting, stacking)
- `train.py` - Improved training with warmup, cosine annealing, focal loss

## Key Classes

### EEGAugmentation
```python
from improved_lstm import EEGAugmentation

augmenter = EEGAugmentation(
    time_shift_max=20,
    noise_std=0.1,
    channel_dropout_prob=0.1,
    p=0.5,  # Probability of applying each augmentation
)

# Use in training
X_aug = augmenter(X_batch)
```

### EnsemblePredictor
```python
from improved_lstm import EnsemblePredictor

# Create ensemble from trained models
ensemble = EnsemblePredictor(
    models=[model1, model2, model3],
    weights=[0.3, 0.3, 0.4],  # Based on validation F1
    strategy='weighted',  # 'average', 'voting', 'max', 'stacking'
)

# Predict
probs = ensemble.predict_proba(X_test)
preds = ensemble.predict(X_test)
```

### ImprovedTrainer
```python
from improved_lstm import ImprovedTrainer

trainer = ImprovedTrainer(model, device, config={
    'epochs': 100,
    'batch_size': 32,
    'lr': 5e-4,
    'augment': True,
})

metrics = trainer.train(train_loader, val_loader, pos_weight)
trainer.save('model_best.pt')
```
