#!/usr/bin/env python3
"""
Inference with Uncertainty Quantification
-----------------------------------------
Flag low-confidence predictions for clinical review.

Usage:
    python inference_uncertain.py --model eeg_mamba --checkpoint ./checkpoints/eeg_mamba_best.pt
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from architectures import get_model
from modules.uncertainty import mc_dropout_inference, compute_uncertainty, flag_uncertain_predictions


def run_inference(
    model: nn.Module,
    data: torch.Tensor,
    n_mc_samples: int = 30,
    uncertainty_threshold: float = 0.3,
) -> Dict:
    """
    Run inference with uncertainty quantification.

    Args:
        model: Trained model
        data: Input EEG data (batch, n_channels, time_steps)
        n_mc_samples: Number of MC dropout samples
        uncertainty_threshold: Threshold for flagging uncertain predictions

    Returns:
        Dictionary with predictions, uncertainties, and flags
    """
    # MC Dropout inference
    mean_pred, std_pred, all_preds = mc_dropout_inference(
        model, data, n_samples=n_mc_samples
    )

    # Compute uncertainty metrics
    metrics = compute_uncertainty(all_preds)

    # Flag uncertain predictions
    flagged, confidence = flag_uncertain_predictions(
        mean_pred, metrics["predictive_std"], threshold=uncertainty_threshold
    )

    # Binary predictions
    predictions = (mean_pred >= 0.5).float()

    return {
        "predictions": predictions.cpu().numpy(),
        "probabilities": mean_pred.cpu().numpy(),
        "uncertainty_std": std_pred.cpu().numpy(),
        "aleatoric": metrics["aleatoric"].cpu().numpy(),
        "epistemic": metrics["epistemic"].cpu().numpy(),
        "entropy": metrics["entropy"].cpu().numpy(),
        "confidence": confidence.cpu().numpy(),
        "flagged_for_review": flagged.cpu().numpy(),
        "n_flagged": flagged.sum().item(),
        "pct_flagged": flagged.float().mean().item() * 100,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="eeg_mamba")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/eeg_mamba_best.pt")
    parser.add_argument("--data_path", type=str, default="../../data/processed")
    parser.add_argument("--mc_samples", type=int, default=30)
    parser.add_argument("--uncertainty_threshold", type=float, default=0.3)
    parser.add_argument("--output", type=str, default="./results/inference_results.json")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # Load model
    model = get_model(args.model, n_channels=16, time_steps=256)
    if Path(args.checkpoint).exists():
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("Warning: No checkpoint found, using random weights")
    model = model.to(device)

    # Load test data
    data_path = Path(args.data_path)
    if (data_path / "X_windows.npy").exists():
        X = np.load(data_path / "X_windows.npy")
        y = np.load(data_path / "y_windows.npy")
        # Use subset for demo
        X = X[:100]
        y = y[:100]
    else:
        print("Generating synthetic test data...")
        X = np.random.randn(100, 16, 256).astype(np.float32)
        y = np.random.randint(0, 2, 100).astype(np.float32)

    data = torch.from_numpy(X).to(device)

    # Run inference
    print(f"\nRunning MC Dropout inference with {args.mc_samples} samples...")
    results = run_inference(
        model, data,
        n_mc_samples=args.mc_samples,
        uncertainty_threshold=args.uncertainty_threshold,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("INFERENCE RESULTS")
    print("=" * 60)
    print(f"Total samples: {len(X)}")
    print(f"Predicted seizures: {results['predictions'].sum():.0f}")
    print(f"Mean confidence: {results['confidence'].mean():.4f}")
    print(f"Mean uncertainty: {results['uncertainty_std'].mean():.4f}")
    print(f"Flagged for review: {results['n_flagged']} ({results['pct_flagged']:.1f}%)")

    # Per-class uncertainty
    seizure_mask = results['predictions'].flatten() == 1
    if seizure_mask.sum() > 0:
        print(f"\nSeizure predictions - Mean uncertainty: {results['uncertainty_std'][seizure_mask].mean():.4f}")
    normal_mask = results['predictions'].flatten() == 0
    if normal_mask.sum() > 0:
        print(f"Normal predictions - Mean uncertainty: {results['uncertainty_std'][normal_mask].mean():.4f}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to JSON-serializable
    json_results = {
        "n_samples": len(X),
        "n_seizures_predicted": int(results['predictions'].sum()),
        "n_flagged": int(results['n_flagged']),
        "pct_flagged": float(results['pct_flagged']),
        "mean_confidence": float(results['confidence'].mean()),
        "mean_uncertainty": float(results['uncertainty_std'].mean()),
        "mc_samples": args.mc_samples,
        "uncertainty_threshold": args.uncertainty_threshold,
    }

    with open(output_path, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Clinical recommendation
    print("\n" + "=" * 60)
    print("CLINICAL RECOMMENDATIONS")
    print("=" * 60)
    if results['pct_flagged'] > 20:
        print("⚠️  High uncertainty rate. Consider:")
        print("   - Reviewing flagged samples manually")
        print("   - Checking EEG signal quality")
        print("   - Retraining with more diverse data")
    else:
        print("✓ Model confidence is within acceptable range")
        print(f"  {results['n_flagged']} samples flagged for expert review")


if __name__ == "__main__":
    main()
