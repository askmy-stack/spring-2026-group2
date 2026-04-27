"""
Run ALL models across all directories in one go.

Usage:
    python -m src.models.run_all_models --data_path src/data/processed/chbmit
"""
import argparse
import logging
from pathlib import Path
from typing import Dict

from src.models.lstm_benchmark_models.train_baseline import load_config, train_baseline
from src.models.lstm_benchmark_models.architectures import MODEL_REGISTRY as BENCHMARK_REGISTRY
from src.models.improved_lstm_models.train import train_improved
from src.models.hugging_face_mamba_moe.train_mamba import train_mamba, MODEL_REGISTRY as MAMBA_REGISTRY

logger = logging.getLogger(__name__)


def run_all(data_path: Path, config: Dict) -> Dict[str, Dict]:
    """Train all models sequentially; return combined results dict."""
    all_results = {}

    logger.info("=" * 60)
    logger.info("PHASE 1: LSTM Benchmark Models (m1-m6)")
    logger.info("=" * 60)
    for model_name in BENCHMARK_REGISTRY:
        logger.info("-" * 40)
        logger.info("Training: %s", model_name)
        logger.info("-" * 40)
        try:
            results = train_baseline(model_name, data_path, config)
            all_results[model_name] = results
            logger.info("  %s: %s", model_name, results)
        except Exception as e:
            logger.error("  %s FAILED: %s", model_name, e)
            all_results[model_name] = {"error": str(e)}

    logger.info("=" * 60)
    logger.info("PHASE 2: Improved LSTM (HierarchicalLSTM)")
    logger.info("=" * 60)
    try:
        results = train_improved(data_path, config)
        all_results["improved_hierarchical_lstm"] = results
        logger.info("  improved_hierarchical_lstm: %s", results)
    except Exception as e:
        logger.error("  improved_hierarchical_lstm FAILED: %s", e)
        all_results["improved_hierarchical_lstm"] = {"error": str(e)}

    logger.info("=" * 60)
    logger.info("PHASE 3: Mamba / Mamba-MoE")
    logger.info("=" * 60)
    for model_name in MAMBA_REGISTRY:
        logger.info("-" * 40)
        logger.info("Training: %s", model_name)
        logger.info("-" * 40)
        try:
            results = train_mamba(model_name, data_path, config)
            all_results[model_name] = results
            logger.info("  %s: %s", model_name, results)
        except Exception as e:
            logger.error("  %s FAILED: %s", model_name, e)
            all_results[model_name] = {"error": str(e)}

    logger.info("=" * 60)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("=" * 60)
    for name, res in all_results.items():
        logger.info("  %-30s %s", name, res)

    return all_results


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Train ALL EEG seizure detection models")
    parser.add_argument("--data_path", default="src/data/processed/chbmit")
    parser.add_argument("--config", default="src/models/config.yaml")
    args = parser.parse_args()
    config = load_config(Path(args.config))
    run_all(Path(args.data_path), config)


if __name__ == "__main__":
    main()
