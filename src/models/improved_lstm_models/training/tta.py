"""
Test-Time Augmentation for EEG seizure classifiers.

At inference we run the model multiple times on slightly-perturbed copies
of the input (identity + circular time shifts, optional polarity flip)
and average the sigmoid probabilities. This is a cheap ensemble-of-one
that typically adds +0.5-1.5 F1 on EEG tasks with no extra training.

Works on any model whose forward accepts ``(batch, channels, time)`` and
returns ``(batch, 1)`` logits — i.e. all m_i and im_i variants, and the
HierarchicalLSTM via a pre-unsqueeze wrapper applied by the caller.
"""
from __future__ import annotations

from typing import Callable, List, Sequence

import torch
import torch.nn as nn


def predict_with_tta(
    model: nn.Module,
    eeg_batch: torch.Tensor,
    shifts: Sequence[int] = (-5, 0, 5),
    flip: bool = False,
    forward_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """
    Average sigmoid probabilities across augmented views.

    Args:
        model: Evaluated in ``eval()`` mode under ``torch.no_grad``.
        eeg_batch: ``(B, C, T)`` float tensor.
        shifts: Integer circular shifts applied along the time axis.
            ``0`` is always the identity view; negative values shift
            left, positive values shift right.
        flip: If True, also average in a channel-polarity-flipped view
            (multiply by -1). EEG signals are roughly zero-mean so this
            is a valid invariance for seizure spike morphology.
        forward_fn: Optional override called as ``forward_fn(model, x)``
            when the model needs shape adaptation (e.g. HierarchicalLSTM
            wants 4-D input). Default just calls ``model(x)``.

    Returns:
        ``(B,)`` tensor of averaged probabilities in [0, 1].
    """
    if forward_fn is None:
        forward_fn = lambda m, x: m(x)

    views = _build_views(eeg_batch, shifts, flip)
    probs: List[torch.Tensor] = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for view in views:
                logits = forward_fn(model, view).squeeze(-1)
                probs.append(torch.sigmoid(logits))
    finally:
        model.train(was_training)

    return torch.stack(probs, dim=0).mean(dim=0)


def _build_views(
    eeg_batch: torch.Tensor,
    shifts: Sequence[int],
    flip: bool,
) -> List[torch.Tensor]:
    """Build the list of augmented views to feed through the model."""
    views: List[torch.Tensor] = []
    for s in shifts:
        views.append(eeg_batch if s == 0 else torch.roll(eeg_batch, shifts=int(s), dims=-1))
    if flip:
        views.extend([-v for v in list(views)])
    return views
