"""
Unified checkpoint I/O for all EEG seizure-detection models.

Every training script in the canonical pipeline (lstm_benchmark_models,
improved_lstm_models, hugging_face_mamba_moe) saves .pt files via
``save_checkpoint`` so downstream consumers (inference, ensembles, apps)
can load any model with a single ``load_checkpoint`` call — without needing
to know which training script produced it.

Schema written to disk:
    {
        "model_class": str,                   # "pkg.module.ClassName"
        "model_builder": str | None,          # "pkg.mod.factory_fn" — preferred over
                                              #   model_class when non-None (used by
                                              #   factory-based models like HF wrappers)
        "model_config": dict,                 # kwargs for rebuild (class or builder)
        "model_state_dict": dict,             # tensor weights
        "optimizer_state_dict": dict | None,  # for resume
        "epoch": int,
        "val_metrics": dict,                  # {"f1", "auroc", "sens", "spec", ...}
        "optimal_threshold": float,           # decision threshold tuned on val
        "input_spec": dict,                   # {"channels", "sfreq", "window_sec"}
        "preprocess": dict,                   # pipeline params for inference-time parity
        "git_commit": str | None,
        "schema_version": int,
    }
"""
from __future__ import annotations

import importlib
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

DEFAULT_INPUT_SPEC: Dict[str, Any] = {
    "channels": 16,
    "sfreq": 256,
    "window_sec": 1.0,
}

DEFAULT_PREPROCESS: Dict[str, Any] = {
    "resample_hz": 256,
    "bandpass": [1.0, 50.0],
    "notch_hz": 60.0,
    "reference": "avg",
    "norm": "zscore",
}


def _qualified_class_name(model: nn.Module) -> str:
    """Return ``pkg.mod.ClassName`` for a model instance."""
    cls = type(model)
    return f"{cls.__module__}.{cls.__qualname__}"


def _current_git_commit() -> Optional[str]:
    """Best-effort short SHA of HEAD; returns None if unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parent,
        )
        return out.decode().strip() or None
    except Exception:
        return None


def save_checkpoint(
    path: Path | str,
    model: nn.Module,
    *,
    model_config: Optional[Dict[str, Any]] = None,
    model_builder: Optional[str] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    val_metrics: Optional[Dict[str, float]] = None,
    optimal_threshold: float = 0.5,
    input_spec: Optional[Dict[str, Any]] = None,
    preprocess: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save a model checkpoint in the unified schema.

    Args:
        path: Destination file (``.pt``). Parent directory is created if missing.
        model: Module whose state_dict + class path will be stored.
        model_config: Kwargs sufficient to re-instantiate the model (via the
            class constructor, or via ``model_builder`` if provided).
        model_builder: Optional ``"pkg.mod.factory_fn"`` callable dotted path.
            When present, ``load_checkpoint`` will call
            ``factory_fn(**model_config)`` instead of
            ``type(model)(**model_config)``. Useful for factory-constructed
            models (e.g. HuggingFace wrappers instantiated via
            ``create_hf_model``) whose top-level class does not accept
            ``model_config`` directly.
        optimizer: Optional optimizer to capture for resume.
        epoch: Training epoch at save time.
        val_metrics: Metric dict for traceability (non-normative).
        optimal_threshold: Decision threshold (tuned on val, default 0.5).
        input_spec: Overrides :data:`DEFAULT_INPUT_SPEC`.
        preprocess: Overrides :data:`DEFAULT_PREPROCESS`.

    Returns:
        Absolute path of the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_class": _qualified_class_name(model),
        "model_builder": model_builder,
        "model_config": dict(model_config or {}),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch),
        "val_metrics": dict(val_metrics or {}),
        "optimal_threshold": float(optimal_threshold),
        "input_spec": dict(input_spec or DEFAULT_INPUT_SPEC),
        "preprocess": dict(preprocess or DEFAULT_PREPROCESS),
        "git_commit": _current_git_commit(),
    }
    torch.save(payload, path)
    logger.info("Saved checkpoint -> %s (epoch=%d, threshold=%.3f)", path, epoch, optimal_threshold)
    return path


def _import_class(qualified_name: str) -> type:
    """Resolve ``pkg.mod.ClassName`` to a class object."""
    module_name, _, class_name = qualified_name.rpartition(".")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def load_checkpoint(
    path: Path | str,
    *,
    map_location: str | torch.device = "cpu",
    build_model: bool = True,
) -> Tuple[Optional[nn.Module], Dict[str, Any]]:
    """
    Load a unified checkpoint. Returns (model_or_None, payload).

    When ``build_model=True`` (default) and ``model_class`` + ``model_config``
    are present, the model is re-instantiated and ``state_dict`` is loaded.
    If instantiation fails (e.g. missing kwargs, moved class), the model is
    returned as ``None`` and the caller can rebuild manually using the payload.

    Args:
        path: Checkpoint file.
        map_location: Device to map tensors to.
        build_model: Whether to attempt model reconstruction.

    Returns:
        Tuple of ``(model, payload)``. ``payload`` always contains the full
        on-disk dict (plus ``"path"``).
    """
    path = Path(path)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(
            f"{path} is not a unified checkpoint (missing 'model_state_dict'). "
            "Legacy bare-state-dict checkpoints require manual loading."
        )
    payload["path"] = str(path)
    if not build_model:
        return None, payload
    model = _try_build_model(payload)
    if model is not None:
        model.load_state_dict(payload["model_state_dict"])
        model.train(False)
    return model, payload


def _try_build_model(payload: Dict[str, Any]) -> Optional[nn.Module]:
    """Attempt to reconstruct the model via builder fn or class; return None on failure."""
    cfg = payload.get("model_config") or {}
    builder_path = payload.get("model_builder")
    if builder_path:
        try:
            builder = _import_class(builder_path)  # resolves fn the same way it does a class
            return builder(**cfg)
        except Exception as exc:
            logger.warning("Could not auto-build via builder %s(%s): %s", builder_path, cfg, exc)
            return None
    cls_name = payload.get("model_class")
    if not cls_name:
        logger.warning("Checkpoint lacks 'model_class' and 'model_builder'; caller must build manually.")
        return None
    try:
        cls = _import_class(cls_name)
        return cls(**cfg)
    except Exception as exc:
        logger.warning("Could not auto-build %s(%s): %s", cls_name, cfg, exc)
        return None
