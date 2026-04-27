"""
Stochastic Weight Averaging helpers.

Thin wrapper over ``torch.optim.swa_utils`` with EEG-specific defaults:
- Averaging kicks in during the last ``start_frac`` fraction of training.
- ``update_bn`` is called on a dedicated loader before evaluation so any
  BatchNorm running stats reflect the averaged weights.

Typical trainer usage::

    swa = SWAHook(model, start_frac=0.75, total_epochs=num_epochs)
    for epoch in range(num_epochs):
        train_one_epoch(...)
        swa.step(epoch, model)               # copies weights when active
    swa.finalise(train_loader, device)       # updates BN on SWA model
    eval(swa.swa_model, ...)
"""
from __future__ import annotations

import copy
import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, update_bn

logger = logging.getLogger(__name__)


class SWAHook:
    """
    Lightweight Stochastic Weight Averaging driver.

    Args:
        model: Live training model. A deep copy is taken on first
            activation to seed the SWA average.
        start_frac: Fraction of total epochs after which averaging
            begins, in (0, 1). 0.75 means the last 25% contributes.
        total_epochs: Total epochs in the training run.

    Attributes:
        swa_model: ``AveragedModel`` holding the running average.
            ``None`` until the first :meth:`step` after ``start_epoch``.
        n_averaged: How many epochs have been averaged so far.
    """

    def __init__(
        self,
        model: nn.Module,
        start_frac: float = 0.75,
        total_epochs: int = 100,
    ):
        if not 0.0 < start_frac < 1.0:
            raise ValueError(f"start_frac must be in (0,1); got {start_frac}")
        self.start_epoch = int(start_frac * total_epochs)
        self.total_epochs = total_epochs
        self._model_ref = model
        self.swa_model: Optional[AveragedModel] = None
        self.n_averaged = 0

    def step(self, epoch: int, model: nn.Module) -> None:
        """Update the SWA running average if we're past ``start_epoch``."""
        if epoch < self.start_epoch:
            return
        if self.swa_model is None:
            self.swa_model = AveragedModel(copy.deepcopy(model))
            logger.info("SWA activated at epoch %d (total=%d)", epoch, self.total_epochs)
        self.swa_model.update_parameters(model)
        self.n_averaged += 1

    def finalise(
        self,
        train_loader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> Optional[nn.Module]:
        """
        Refresh BatchNorm stats on the SWA model using ``train_loader``.

        Returns the SWA module, or ``None`` if SWA never activated (run
        was shorter than ``start_epoch``).
        """
        if self.swa_model is None:
            logger.warning("SWA never activated (run shorter than start_epoch). "
                           "Skipping finalise().")
            return None
        self.swa_model.to(device)
        update_bn(_tensor_only_loader(train_loader, device), self.swa_model, device=device)
        logger.info("SWA finalised — averaged over %d epochs", self.n_averaged)
        return self.swa_model

    def state_dict(self) -> Optional[dict]:
        """Return the SWA inner-model state dict, or ``None`` if not active."""
        if self.swa_model is None:
            return None
        # Strip AveragedModel's "module." prefix so the dict loads back
        # into the original architecture.
        inner = self.swa_model.module
        return {k: v.detach().clone() for k, v in inner.state_dict().items()}


def _tensor_only_loader(loader: torch.utils.data.DataLoader, device: torch.device):
    """Yield just the input tensor from ``(x, y)`` batches for ``update_bn``."""
    for batch in loader:
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch
        yield x.to(device, non_blocking=True)
