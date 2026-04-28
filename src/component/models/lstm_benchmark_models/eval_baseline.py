"""
LSTM Benchmark Evaluation.

Usage:
    python -m src.models.lstm_benchmark_models.eval_baseline \
        --model m1_vanilla_lstm \
        --checkpoint outputs/models/m1_vanilla_lstm_best.pt \
        --data_path src/data/processed/chbmit
"""
import argparse
import logging
from pathlib import Path
from typing import Dict

import torch

from src.component.models.utils.metrics import compute_f1_score, compute_auc_roc, compute_sensitivity
from .architectures import get_benchmark_model, MODEL_REGISTRY
from .train_baseline import load_config, _build_data_loaders, _get_device, _evaluate_on_test

logger = logging.getLogger(__name__)


def evaluate_checkpoint(
    model_name: str, checkpoint_path: Path, data_path: Path, config: Dict
) -> Dict:
    """
    Load checkpoint and run evaluation on the test split.

    Args:
        model_name: Key in MODEL_REGISTRY.
        checkpoint_path: Path to .pt state dict file.
        data_path: Directory containing test/ tensor split.
        config: Config dict from config.yaml.

    Returns:
        Dict with f1, auc_roc, sensitivity.

    Raises:
        FileNotFoundError: If checkpoint_path does not exist.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    device = _get_device()
    model = get_benchmark_model(model_name, **config["models"]["lstm_benchmark"])
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    _, _, test_loader = _build_data_loaders(data_path, config)
    return _evaluate_on_test(model, test_loader, device)


def main() -> None:
    """CLI entry point for evaluation."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate LSTM benchmark checkpoint")
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", default="src/data/processed/chbmit")
    parser.add_argument("--config", default="src/models/config.yaml")
    args = parser.parse_args()
    config = load_config(Path(args.config))
    metrics = evaluate_checkpoint(args.model, Path(args.checkpoint), Path(args.data_path), config)
    logger.info("Evaluation: %s", metrics)


if __name__ == "__main__":
    main()
