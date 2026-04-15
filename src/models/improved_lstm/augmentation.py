"""
EEG Data Augmentation for Seizure Detection
============================================
Implements augmentation techniques that preserve seizure patterns:
- Time shifting
- Gaussian noise injection
- Channel dropout
- Amplitude scaling
- Time masking
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class EEGAugmentation(nn.Module):
    """
    EEG-specific data augmentation module.
    
    Applies random augmentations during training to improve generalization.
    All augmentations preserve the seizure patterns while adding variability.
    
    Args:
        time_shift_max: Maximum samples to shift (default: 20)
        noise_std: Standard deviation of Gaussian noise (default: 0.1)
        channel_dropout_prob: Probability of dropping a channel (default: 0.1)
        amplitude_scale_range: Range for amplitude scaling (default: (0.8, 1.2))
        time_mask_max: Maximum timesteps to mask (default: 20)
        p: Overall probability of applying augmentation (default: 0.5)
    """
    
    def __init__(
        self,
        time_shift_max: int = 20,
        noise_std: float = 0.1,
        channel_dropout_prob: float = 0.1,
        amplitude_scale_range: Tuple[float, float] = (0.8, 1.2),
        time_mask_max: int = 20,
        p: float = 0.5,
    ):
        super().__init__()
        self.time_shift_max = time_shift_max
        self.noise_std = noise_std
        self.channel_dropout_prob = channel_dropout_prob
        self.amplitude_scale_range = amplitude_scale_range
        self.time_mask_max = time_mask_max
        self.p = p
    
    def time_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly shift the signal in time (circular)."""
        if self.time_shift_max <= 0:
            return x
        shift = torch.randint(-self.time_shift_max, self.time_shift_max + 1, (1,)).item()
        return torch.roll(x, shifts=shift, dims=-1)
    
    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise scaled by signal amplitude."""
        if self.noise_std <= 0:
            return x
        noise = torch.randn_like(x) * self.noise_std * x.std()
        return x + noise
    
    def channel_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly zero out entire channels."""
        if self.channel_dropout_prob <= 0:
            return x
        mask = torch.rand(x.size(1), device=x.device) > self.channel_dropout_prob
        mask = mask.float().unsqueeze(0).unsqueeze(-1)
        return x * mask
    
    def amplitude_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly scale signal amplitude."""
        low, high = self.amplitude_scale_range
        scale = torch.empty(1, device=x.device).uniform_(low, high)
        return x * scale
    
    def time_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly mask a contiguous time segment."""
        if self.time_mask_max <= 0:
            return x
        seq_len = x.size(-1)
        mask_len = torch.randint(1, self.time_mask_max + 1, (1,)).item()
        max_start = seq_len - mask_len
        if max_start <= 0:
            return x
        start = torch.randint(0, max_start, (1,)).item()
        x = x.clone()
        x[..., start:start + mask_len] = 0
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations to input tensor."""
        if not self.training:
            return x
        
        # Apply each augmentation with probability p
        if torch.rand(1).item() < self.p:
            x = self.time_shift(x)
        if torch.rand(1).item() < self.p:
            x = self.add_noise(x)
        if torch.rand(1).item() < self.p:
            x = self.channel_dropout(x)
        if torch.rand(1).item() < self.p:
            x = self.amplitude_scale(x)
        if torch.rand(1).item() < self.p * 0.5:  # Less frequent
            x = self.time_mask(x)
        
        return x


def augment_batch(
    X: torch.Tensor,
    y: torch.Tensor,
    augmenter: EEGAugmentation,
    oversample_minority: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Augment a batch of EEG data.
    
    Args:
        X: Input tensor (batch, channels, time)
        y: Labels tensor (batch,)
        augmenter: EEGAugmentation instance
        oversample_minority: If True, duplicate and augment seizure samples
    
    Returns:
        Augmented X and y tensors
    """
    augmenter.train()
    
    if oversample_minority:
        # Find seizure samples
        seizure_mask = y == 1
        seizure_X = X[seizure_mask]
        seizure_y = y[seizure_mask]
        
        if len(seizure_X) > 0:
            # Augment seizure samples
            aug_seizure_X = augmenter(seizure_X)
            
            # Concatenate original + augmented seizure
            X = torch.cat([X, aug_seizure_X], dim=0)
            y = torch.cat([y, seizure_y], dim=0)
    
    # Augment all samples
    X_aug = augmenter(X)
    
    return X_aug, y


class MixUp(nn.Module):
    """
    MixUp augmentation for EEG data.
    
    Creates convex combinations of training examples.
    """
    
    def __init__(self, alpha: float = 0.2):
        super().__init__()
        self.alpha = alpha
    
    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply MixUp to a batch.
        
        Returns:
            mixed_x: Mixed input
            y_a: Original labels
            y_b: Shuffled labels
            lam: Mixing coefficient
        """
        if not self.training:
            return x, y, y, 1.0
        
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1.0
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Compute MixUp loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
