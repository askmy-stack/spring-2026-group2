"""Small callback-oriented helpers for the submission-facing package."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStoppingState:
    best_score: float = float("-inf")
    best_epoch: int = 0
    patience: int = 5
    epochs_without_improvement: int = 0

    def update(self, score: float, epoch: int) -> bool:
        if score > self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            return True
        self.epochs_without_improvement += 1
        return False

    def should_stop(self) -> bool:
        return self.epochs_without_improvement >= self.patience
