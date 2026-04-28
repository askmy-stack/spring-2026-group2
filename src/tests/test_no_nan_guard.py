"""
Regression test: ``train_mamba._run_training_loop`` aborts on non-finite loss.

The earlier bug let ``eeg_mamba_moe`` continue looping after diverging to
NaN, which meant the saved "best" checkpoint ended up containing NaN weights.
This test forces the training epoch helper to return NaN and asserts the
outer loop breaks without calling ``stopper.step``.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from src.component.models.hugging_face_mamba_moe import train_mamba


class _DummyModel(nn.Module):
    """Smallest possible 1-logit classifier for sanity checks."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):  # noqa: D401 — trivial forward
        return self.fc(x.mean(dim=-1).mean(dim=-1, keepdim=True))


@pytest.fixture
def _config():
    """Bare-minimum config for the training-loop helper."""
    return {
        "training": {
            "num_epochs": 5,
            "early_stopping_patience": 10,
            "gradient_clip": 1.0,
            "warmup_epochs": 0,
        },
        "models": {
            "hugging_face_mamba_moe": {"moe_loss_weight": 0.01},
        },
    }


def _make_stopper():
    stopper = MagicMock()
    stopper.should_stop = False
    stopper.checkpoint_path = None
    return stopper


def test_nan_train_loss_aborts_loop(_config):
    """When train loss is NaN, the epoch loop must break before calling stopper."""
    model = _DummyModel()
    stopper = _make_stopper()

    with patch.object(train_mamba, "_train_one_epoch", return_value=float("nan")) as mock_train, \
         patch.object(train_mamba, "_validate_one_epoch", return_value=0.5) as mock_val:
        train_mamba._run_training_loop(
            model, "eeg_mamba", train_loader=[], val_loader=[],
            criterion=nn.BCEWithLogitsLoss(), optimizer=MagicMock(),
            scheduler=MagicMock(), stopper=stopper,
            config=_config, device=torch.device("cpu"),
        )

    # Only one epoch attempted before NaN triggered the abort.
    assert mock_train.call_count == 1
    # Validation should not run after NaN train loss, and stopper must not be poked.
    assert mock_val.call_count == 0
    stopper.step.assert_not_called()


def test_nan_val_loss_aborts_loop(_config):
    """NaN val loss must also abort (before next epoch)."""
    model = _DummyModel()
    stopper = _make_stopper()

    with patch.object(train_mamba, "_train_one_epoch", return_value=0.5), \
         patch.object(train_mamba, "_validate_one_epoch", return_value=float("inf")):
        train_mamba._run_training_loop(
            model, "eeg_mamba", train_loader=[], val_loader=[],
            criterion=nn.BCEWithLogitsLoss(), optimizer=MagicMock(),
            scheduler=MagicMock(), stopper=stopper,
            config=_config, device=torch.device("cpu"),
        )

    stopper.step.assert_not_called()


def test_finite_losses_complete_all_epochs(_config):
    """Sanity: with finite losses the loop runs ``num_epochs`` times."""
    model = _DummyModel()
    stopper = _make_stopper()

    with patch.object(train_mamba, "_train_one_epoch", return_value=0.5) as mock_train, \
         patch.object(train_mamba, "_validate_one_epoch", return_value=0.4):
        train_mamba._run_training_loop(
            model, "eeg_mamba", train_loader=[], val_loader=[],
            criterion=nn.BCEWithLogitsLoss(), optimizer=MagicMock(),
            scheduler=MagicMock(), stopper=stopper,
            config=_config, device=torch.device("cpu"),
        )

    assert mock_train.call_count == _config["training"]["num_epochs"]
    assert stopper.step.call_count == _config["training"]["num_epochs"]
