"""
EEG Data Augmentation for Improved LSTM Models.

All augmentations preserve seizure patterns while increasing variability.
Import from here — never define augmentation inline in a training script.
"""
import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EEGAugmentation(nn.Module):
    """
    EEG-specific data augmentation applied only during training.

    Args:
        time_shift_max: Max samples to shift (circular, default: 20).
        noise_std: Gaussian noise std relative to signal std (default: 0.1).
        channel_dropout_prob: Per-channel zero-out probability (default: 0.1).
        amplitude_scale_range: Range for uniform amplitude scaling (default: (0.8, 1.2)).
        time_mask_max: Max timesteps to zero-mask (default: 20).
        p: Probability to apply each augmentation (default: 0.5).

    Example:
        >>> aug = EEGAugmentation(p=0.5)
        >>> aug.train()
        >>> out = aug(torch.randn(4, 16, 256))
        >>> assert out.shape == (4, 16, 256)
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

    def forward(self, eeg_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply random augmentations to EEG batch (training mode only).

        Args:
            eeg_tensor: shape (batch, n_channels, time_steps)

        Returns:
            Augmented tensor, same shape.
        """
        if not self.training:
            return eeg_tensor
        augmented = eeg_tensor
        if torch.rand(1).item() < self.p:
            augmented = self.time_shift(augmented)
        if torch.rand(1).item() < self.p:
            augmented = self.add_noise(augmented)
        if torch.rand(1).item() < self.p:
            augmented = self.channel_dropout(augmented)
        if torch.rand(1).item() < self.p:
            augmented = self.amplitude_scale(augmented)
        if torch.rand(1).item() < self.p * 0.5:
            augmented = self.time_mask(augmented)
        return augmented

    def time_shift(self, eeg_tensor: torch.Tensor) -> torch.Tensor:
        """
        Randomly shift signal in time (circular).

        Args:
            eeg_tensor: shape (batch, n_channels, time_steps)

        Returns:
            Shifted tensor.
        """
        if self.time_shift_max <= 0:
            return eeg_tensor
        shift = torch.randint(-self.time_shift_max, self.time_shift_max + 1, (1,)).item()
        return torch.roll(eeg_tensor, shifts=shift, dims=-1)

    def add_noise(self, eeg_tensor: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise scaled by signal amplitude.

        Args:
            eeg_tensor: shape (batch, n_channels, time_steps)

        Returns:
            Noisy tensor.
        """
        if self.noise_std <= 0:
            return eeg_tensor
        noise = torch.randn_like(eeg_tensor) * self.noise_std * eeg_tensor.std()
        return eeg_tensor + noise

    def channel_dropout(self, eeg_tensor: torch.Tensor) -> torch.Tensor:
        """
        Randomly zero out entire channels.

        Args:
            eeg_tensor: shape (batch, n_channels, time_steps)

        Returns:
            Tensor with some channels zeroed.
        """
        if self.channel_dropout_prob <= 0:
            return eeg_tensor
        mask = (torch.rand(eeg_tensor.size(1), device=eeg_tensor.device) > self.channel_dropout_prob)
        return eeg_tensor * mask.float().unsqueeze(0).unsqueeze(-1)

    def amplitude_scale(self, eeg_tensor: torch.Tensor) -> torch.Tensor:
        """
        Randomly scale signal amplitude within a fixed range.

        Args:
            eeg_tensor: shape (batch, n_channels, time_steps)

        Returns:
            Amplitude-scaled tensor.
        """
        low, high = self.amplitude_scale_range
        scale = torch.empty(1, device=eeg_tensor.device).uniform_(low, high)
        return eeg_tensor * scale

    def time_mask(self, eeg_tensor: torch.Tensor) -> torch.Tensor:
        """
        Zero out a random contiguous time segment.

        Args:
            eeg_tensor: shape (batch, n_channels, time_steps)

        Returns:
            Time-masked tensor.
        """
        if self.time_mask_max <= 0:
            return eeg_tensor
        seq_len = eeg_tensor.size(-1)
        mask_len = int(torch.randint(1, self.time_mask_max + 1, (1,)).item())
        max_start = seq_len - mask_len
        if max_start <= 0:
            return eeg_tensor
        start = int(torch.randint(0, max_start, (1,)).item())
        masked = eeg_tensor.clone()
        masked[..., start:start + mask_len] = 0.0
        return masked


def augment_batch(
    eeg_batch: torch.Tensor,
    label_batch: torch.Tensor,
    augmenter: EEGAugmentation,
    oversample_minority: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Augment a training batch with optional seizure oversampling.

    Each sample is augmented exactly once. When ``oversample_minority=True`` the
    seizure (positive) samples are additionally duplicated with an *independent*
    augmentation draw, so the final batch contains:
      - all original samples, each augmented once
      - a second copy of seizure samples, each augmented once (different noise)

    Args:
        eeg_batch: EEG data, shape (batch, n_channels, time_steps).
        label_batch: Binary labels, shape (batch,).
        augmenter: EEGAugmentation instance (must be in train mode).
        oversample_minority: If True, add an extra augmented copy of seizures.

    Returns:
        Tuple of (augmented_eeg, labels) tensors.

    Example:
        >>> aug = EEGAugmentation(); aug.train()
        >>> X_aug, y_aug = augment_batch(torch.randn(8, 16, 256), torch.randint(0, 2, (8,)), aug)
    """
    augmenter.train()
    augmented_batch = augmenter(eeg_batch)
    if not oversample_minority:
        return augmented_batch, label_batch

    seizure_mask = label_batch == 1
    if seizure_mask.sum() == 0:
        return augmented_batch, label_batch

    # Draw a second, independent augmentation of the raw seizures (not of the
    # already-augmented ones) to avoid compounding noise on the duplicated copy.
    extra_seizures = augmenter(eeg_batch[seizure_mask])
    return (
        torch.cat([augmented_batch, extra_seizures], dim=0),
        torch.cat([label_batch, label_batch[seizure_mask]], dim=0),
    )


class MixUp(nn.Module):
    """
    MixUp augmentation for EEG data.

    Creates convex combinations of training examples.

    Args:
        alpha: Beta distribution concentration parameter (default: 0.2).

    Example:
        >>> mixup = MixUp(alpha=0.2); mixup.train()
        >>> x_mix, ya, yb, lam = mixup(torch.randn(4, 16, 256), torch.randint(0, 2, (4,)).float())
    """

    def __init__(self, alpha: float = 0.2):
        super().__init__()
        self.alpha = alpha

    def forward(
        self, eeg_tensor: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Args:
            eeg_tensor: shape (batch, n_channels, time_steps)
            labels: shape (batch,)

        Returns:
            (mixed_eeg, labels_a, labels_b, lambda) for use with mixup_criterion.
        """
        if not self.training:
            return eeg_tensor, labels, labels, 1.0
        lam = float(np.random.beta(self.alpha, self.alpha)) if self.alpha > 0 else 1.0
        index = torch.randperm(eeg_tensor.size(0), device=eeg_tensor.device)
        mixed_eeg = lam * eeg_tensor + (1.0 - lam) * eeg_tensor[index]
        return mixed_eeg, labels, labels[index], lam


def mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """
    Compute MixUp loss as convex combination of two label losses.

    Args:
        criterion: Loss function.
        pred: Model predictions.
        labels_a: Original labels.
        labels_b: Shuffled labels.
        lam: Mixing coefficient.

    Returns:
        Scalar MixUp loss.
    """
    return lam * criterion(pred, labels_a) + (1.0 - lam) * criterion(pred, labels_b)
