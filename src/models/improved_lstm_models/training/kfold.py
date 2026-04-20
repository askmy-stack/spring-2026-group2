"""
K-fold splitter for the improved benchmark trainer.

Two modes:

- **Subject-wise** (preferred): requires a per-window ``subject_ids.pt``
  (shape ``(N,)``, int or str) in the train split. Folds are built so no
  subject appears in both train and held-out partitions — faithful
  generalisation estimate.
- **Window-wise** (fallback): plain stratified K-fold over window
  indices. Automatically used when no subject_ids tensor is found;
  emits a warning so the user knows subject-wise is unavailable.

Import from here — never hand-roll fold logic in the trainer.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def subject_wise_kfold(
    data_path: Path,
    n_splits: int = 3,
    seed: int = 42,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Yield ``(train_idx, val_idx)`` arrays for each of ``n_splits`` folds.

    Splits the training partition (``data_path/train``) into folds by
    subject when ``subject_ids.pt`` is present, else falls back to
    stratified random indexing.

    Args:
        data_path: Directory holding the ``train/`` split with ``data.pt``,
            ``labels.pt``, and optionally ``subject_ids.pt``.
        n_splits: Number of folds (default 3).
        seed: Shuffle seed for deterministic fold assignments.

    Yields:
        ``(train_idx, val_idx)`` int64 numpy arrays indexing into the
        training tensors.

    Raises:
        FileNotFoundError: If the train split tensors are missing.
    """
    train_dir = Path(data_path) / "train"
    labels_path = train_dir / "labels.pt"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels at {labels_path}")

    labels = torch.load(labels_path, weights_only=True).long().squeeze().numpy()
    subj_path = train_dir / "subject_ids.pt"
    if subj_path.exists():
        yield from _subject_folds(subj_path, labels, n_splits, seed)
    else:
        logger.warning(
            "No subject_ids.pt at %s — falling back to stratified window-wise K-fold. "
            "Regenerate tensors with per-window subject ids to enable true "
            "subject-wise CV.",
            subj_path,
        )
        yield from _stratified_window_folds(labels, n_splits, seed)


def _subject_folds(
    subj_path: Path,
    labels: np.ndarray,
    n_splits: int,
    seed: int,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield subject-disjoint train/val index arrays for ``n_splits`` folds."""
    subjects = torch.load(subj_path, weights_only=True)
    subjects = np.asarray(subjects)
    unique = np.unique(subjects)
    if len(unique) < n_splits:
        raise ValueError(
            f"n_splits={n_splits} > n_subjects={len(unique)}; cannot build "
            "subject-disjoint folds."
        )
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    fold_assignments = np.array_split(shuffled, n_splits)
    for fold_idx, val_subjects in enumerate(fold_assignments):
        val_mask = np.isin(subjects, val_subjects)
        val_idx = np.flatnonzero(val_mask)
        train_idx = np.flatnonzero(~val_mask)
        logger.info(
            "Fold %d/%d — val subjects=%s  (n_train=%d, n_val=%d)",
            fold_idx + 1, n_splits, list(val_subjects),
            train_idx.size, val_idx.size,
        )
        yield train_idx, val_idx


def _stratified_window_folds(
    labels: np.ndarray,
    n_splits: int,
    seed: int,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Stratified K-fold over window indices (fallback when subject_ids missing)."""
    try:
        from sklearn.model_selection import StratifiedKFold
    except ImportError as exc:
        raise RuntimeError(
            "sklearn required for window-wise stratified K-fold fallback."
        ) from exc
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    indices = np.arange(len(labels))
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(indices, labels)):
        logger.info(
            "Fold %d/%d — stratified random (n_train=%d, n_val=%d)",
            fold_idx + 1, n_splits, train_idx.size, val_idx.size,
        )
        yield train_idx.astype(np.int64), val_idx.astype(np.int64)


def load_train_tensors(data_path: Path) -> Tuple[torch.Tensor, torch.Tensor, Optional[np.ndarray]]:
    """Load ``(data, labels, subject_ids)`` from the train split.

    ``subject_ids`` is ``None`` when the corresponding tensor is missing.
    """
    train_dir = Path(data_path) / "train"
    data = torch.load(train_dir / "data.pt", weights_only=True).float()
    labels = torch.load(train_dir / "labels.pt", weights_only=True).long().squeeze()
    subj_path = train_dir / "subject_ids.pt"
    subjects = (
        np.asarray(torch.load(subj_path, weights_only=True))
        if subj_path.exists() else None
    )
    return data, labels, subjects
