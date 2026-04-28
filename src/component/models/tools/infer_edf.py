"""
End-to-end inference smoke test.

Loads a unified-schema `.pt` checkpoint via
:func:`src.models.utils.checkpoint.load_checkpoint`, preprocesses one raw
CHB-MIT `.edf` recording with the same pipeline used during training
(resample → bandpass → notch → avg-reference → z-score → 1-s windows),
then prints a per-window probability + decision using the checkpoint's
stored ``optimal_threshold``.

Usage:

    python -m src.models.tools.infer_edf \\
        --edf src/data/raw/chbmit/chb01/chb01_01.edf \\
        --ckpt src/models/lstm_benchmark_models/checkpoints/m1_vanilla_lstm_best.pt

This is a validation / smoke test — not a production inference server.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

from src.component.models.utils.checkpoint import load_checkpoint

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trained checkpoint on a single .edf recording.")
    parser.add_argument("--edf", required=True, help="Path to a CHB-MIT .edf file.")
    parser.add_argument("--ckpt", required=True, help="Unified-schema .pt checkpoint.")
    parser.add_argument("--max_windows", type=int, default=30,
                        help="Limit how many windows to print (default: 30).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    """Run the smoke test; return 0 on success, non-zero on failure."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()

    edf_path = Path(args.edf)
    ckpt_path = Path(args.ckpt)
    if not edf_path.exists():
        logger.error("EDF not found: %s", edf_path)
        return 2
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        return 2

    device = torch.device(args.device if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    model, payload = load_checkpoint(ckpt_path, map_location=device, build_model=True)
    if model is None:
        logger.error("Auto-rebuild failed for %s; checkpoint lacks model_class/config.", ckpt_path)
        return 3
    model = model.to(device).eval()

    input_spec = payload.get("input_spec", {})
    preprocess_spec = payload.get("preprocess", {})
    threshold = float(payload.get("optimal_threshold", 0.5))
    logger.info("Checkpoint: model_class=%s  threshold=%.3f  epoch=%s",
                payload.get("model_class"), threshold, payload.get("epoch"))
    logger.info("Input spec: %s", input_spec)
    logger.info("Preprocess: %s", preprocess_spec)

    tensor = _load_and_window_edf(edf_path, input_spec, preprocess_spec)
    if tensor.numel() == 0:
        logger.error("No windows produced from %s.", edf_path)
        return 4
    logger.info("Produced %d windows of shape %s", tensor.shape[0], tuple(tensor.shape[1:]))

    probs = _run_inference(model, tensor.to(device))
    decisions = (probs >= threshold).astype(int)
    n_show = min(args.max_windows, len(probs))
    logger.info("First %d window predictions (threshold=%.2f):", n_show, threshold)
    for i in range(n_show):
        logger.info("  window %3d  p=%.4f  decision=%d", i, float(probs[i]), int(decisions[i]))
    logger.info("Summary: %d/%d windows flagged positive (%.1f%%)",
                int(decisions.sum()), len(decisions),
                100.0 * decisions.mean() if len(decisions) else 0.0)
    return 0


def _load_and_window_edf(
    edf_path: Path,
    input_spec: Dict[str, Any],
    preprocess_spec: Dict[str, Any],
) -> torch.Tensor:
    """Preprocess the EDF and slice it into non-overlapping windows.

    Returns a float tensor of shape ``(n_windows, channels, time_steps)``.
    """
    try:
        import mne  # local import so the test imports even without mne on path
    except ImportError as exc:
        raise ImportError("mne is required: pip install mne") from exc

    channels = int(input_spec.get("channels", 16))
    sfreq = int(input_spec.get("sfreq", 256))
    window_sec = float(input_spec.get("window_sec", 1.0))
    time_steps = int(window_sec * sfreq)

    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    cfg = _build_preprocess_config(preprocess_spec, sfreq)
    from src.data_loader.core.signal import preprocess
    raw = preprocess(raw, cfg)
    data = raw.get_data()  # (n_ch, n_samples)
    data = _fit_channels(data, channels)
    data = _zscore(data) if preprocess_spec.get("norm", "zscore") == "zscore" else data

    n_windows = data.shape[1] // time_steps
    if n_windows <= 0:
        return torch.empty(0)
    trimmed = data[:, : n_windows * time_steps]
    windows = trimmed.reshape(data.shape[0], n_windows, time_steps).transpose(1, 0, 2)
    return torch.from_numpy(windows).float()


def _build_preprocess_config(spec: Dict[str, Any], sfreq: int) -> Dict[str, Any]:
    """Shape the stored preprocess spec into the dict expected by ``core.signal.preprocess``."""
    bandpass = spec.get("bandpass", [1.0, 50.0])
    return {
        "signal": {
            "target_sfreq": int(spec.get("resample_hz", sfreq)),
            "bandpass": [float(bandpass[0]), float(bandpass[1])],
            "notch": float(spec.get("notch_hz", 60.0)),
            "reference": spec.get("reference", "average"),
        },
        "resampling": {"default_method": "polyphase"},
    }


def _fit_channels(data: np.ndarray, n_target: int) -> np.ndarray:
    """Pad with zero rows or truncate so the channel dim matches the model's expectation."""
    n_ch = data.shape[0]
    if n_ch == n_target:
        return data
    if n_ch > n_target:
        return data[:n_target]
    pad = np.zeros((n_target - n_ch, data.shape[1]), dtype=data.dtype)
    return np.concatenate([data, pad], axis=0)


def _zscore(data: np.ndarray) -> np.ndarray:
    """Per-channel zero-mean / unit-variance normalisation."""
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    std[std < 1e-6] = 1.0
    return (data - mean) / std


def _run_inference(model: torch.nn.Module, windows: torch.Tensor) -> np.ndarray:
    """Run model on a batch of windows; return per-window positive-class probabilities.

    Handles both binary-sigmoid (shape ``(B, 1)``) and 2-class-softmax
    (shape ``(B, 2)``) heads transparently.
    """
    model.eval()
    with torch.no_grad():
        try:
            logits = model(windows)
        except RuntimeError:
            logits = model(windows.unsqueeze(1))  # HierarchicalLSTM expects 4-D
    logits = logits.detach().cpu()
    if logits.ndim == 2 and logits.shape[-1] == 2:
        probs = torch.softmax(logits, dim=-1)[:, 1]
    else:
        probs = torch.sigmoid(logits.squeeze(-1))
    return probs.numpy()


if __name__ == "__main__":
    sys.exit(main())
