"""
MixUp augmentation + label-smoothing-aware binary loss.

MixUp (Zhang et al. 2018) linearly interpolates pairs of (x, y) so the
model sees convex combinations of examples and soft labels, which acts
as a strong regulariser and reduces memorisation. Implementation here
is tailored to binary classification with logits (shape ``(B, 1)``) and
composes cleanly with BCEWithLogitsLoss + label smoothing.

Import from here — never define MixUp logic inline in a training script.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def mixup_batch(
    eeg_batch: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Sample a MixUp pair for one batch.

    Args:
        eeg_batch: ``(B, ...)`` input tensor (any trailing shape).
        labels: ``(B,)`` binary labels.
        alpha: Beta distribution concentration. 0 disables MixUp
            (returns original batch with lam=1.0).

    Returns:
        Tuple ``(mixed_x, labels_a, labels_b, lam)``:
          - ``mixed_x``: interpolated inputs, same shape as ``eeg_batch``.
          - ``labels_a``: original labels (float).
          - ``labels_b``: shuffled labels (float).
          - ``lam``: mixing coefficient in [0, 1].

    The caller computes ``lam * loss(logits, labels_a) + (1 - lam) *
    loss(logits, labels_b)``. See :func:`mixup_bce_loss` for the binary
    convenience wrapper.
    """
    if alpha <= 0.0:
        return eeg_batch, labels.float(), labels.float(), 1.0

    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    perm = torch.randperm(eeg_batch.size(0), device=eeg_batch.device)
    mixed_x = lam * eeg_batch + (1.0 - lam) * eeg_batch[perm]
    return mixed_x, labels.float(), labels[perm].float(), lam


def mixup_bce_loss(
    logits: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    lam: float,
    pos_weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Binary cross-entropy over a MixUp pair with optional label smoothing.

    Args:
        logits: ``(B,)`` or ``(B, 1)`` raw logits.
        labels_a: ``(B,)`` float labels (first pair element).
        labels_b: ``(B,)`` float labels (second pair element).
        lam: Mixing coefficient from :func:`mixup_batch`.
        pos_weight: Optional scalar tensor for positive class reweighting.
        label_smoothing: In [0, 0.5]. Shifts targets toward 0.5, i.e.
            ``y_smooth = y*(1 - eps) + 0.5*eps``.

    Returns:
        Scalar loss tensor (mean reduction).
    """
    logits = logits.squeeze(-1) if logits.ndim > 1 else logits
    ya = _smooth(labels_a, label_smoothing)
    yb = _smooth(labels_b, label_smoothing)
    loss_a = F.binary_cross_entropy_with_logits(
        logits, ya, pos_weight=pos_weight, reduction="mean"
    )
    loss_b = F.binary_cross_entropy_with_logits(
        logits, yb, pos_weight=pos_weight, reduction="mean"
    )
    return lam * loss_a + (1.0 - lam) * loss_b


class MixUpBCELoss(nn.Module):
    """
    Stateful wrapper around :func:`mixup_bce_loss`.

    Use when the trainer already holds a criterion-like object; stores
    ``pos_weight`` / ``label_smoothing`` so the train loop only passes
    ``(logits, labels_a, labels_b, lam)``.

    Example:
        >>> criterion = MixUpBCELoss(pos_weight=torch.tensor(1.5),
        ...                          label_smoothing=0.1)
        >>> loss = criterion(logits, y_a, y_b, lam=0.7)
    """

    def __init__(
        self,
        pos_weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None
        self.label_smoothing = float(label_smoothing)

    def forward(
        self,
        logits: torch.Tensor,
        labels_a: torch.Tensor,
        labels_b: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        return mixup_bce_loss(
            logits, labels_a, labels_b, lam,
            pos_weight=self.pos_weight,
            label_smoothing=self.label_smoothing,
        )


def _smooth(labels: torch.Tensor, eps: float) -> torch.Tensor:
    """Apply one-sided label smoothing toward 0.5."""
    if eps <= 0.0:
        return labels
    return labels * (1.0 - eps) + 0.5 * eps
