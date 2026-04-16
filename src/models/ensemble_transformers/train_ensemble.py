"""
7-Model Ensemble evaluation entry point.

Usage:
    python -m src.models.ensemble_transformers.train_ensemble \
        --checkpoint_dir outputs/models \
        --data_path src/data/processed/chbmit
"""
import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.lstm_benchmark_models.train_baseline import load_config, _get_device
from .ensemble import load_ensemble_checkpoints, evaluate_ensemble

logger = logging.getLogger(__name__)


def main() -> None:
    """CLI entry point for ensemble evaluation."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate 7-model ensemble on test split")
    parser.add_argument("--checkpoint_dir", default="outputs/models")
    parser.add_argument("--data_path", default="src/data/processed/chbmit")
    parser.add_argument("--config", default="src/models/config.yaml")
    parser.add_argument("--strategy", default="mean", choices=["mean", "weighted"])
    args = parser.parse_args()

    config = load_config(Path(args.config))
    device = _get_device()
    models = load_ensemble_checkpoints(Path(args.checkpoint_dir), config, device)

    if not models:
        logger.error("No model checkpoints found in %s. Train models first.", args.checkpoint_dir)
        return

    test_loader = _build_test_loader(Path(args.data_path), config)
    metrics = evaluate_ensemble(models, test_loader, device, strategy=args.strategy)
    logger.info("Ensemble (%s) results on test split: %s", args.strategy, metrics)


def _build_test_loader(data_path: Path, config: dict) -> DataLoader:
    """Build test DataLoader from tensor files."""
    test_path = data_path / "test"
    if not (test_path / "data.pt").exists():
        raise FileNotFoundError(f"No test data at {test_path}")
    data = torch.load(test_path / "data.pt", weights_only=True).float()
    labels = torch.load(test_path / "labels.pt", weights_only=True).float().squeeze()
    batch_size = config["training"]["batch_size"]
    return DataLoader(TensorDataset(data, labels), batch_size=batch_size, shuffle=False)


if __name__ == "__main__":
    main()
