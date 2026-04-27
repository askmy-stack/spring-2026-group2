"""Base spec and validation for pretrained HF EEG models."""
from __future__ import annotations

import warnings
from dataclasses import dataclass


@dataclass(frozen=True)
class PretrainedInputSpec:
    """
    Metadata contract for a pretrained model's expected inputs.

    Args:
        model_id: Short registry key.
        repo_id: HuggingFace Hub repo identifier.
        required_sfreq: If set, raises if sfreq doesn't match.
        required_channels: If set, raises if in_channels doesn't match.
        recommended_window_sec: Warns if window differs.
        max_window_sec: Raises if window exceeds this.
        notes: Human-readable guidance string.
    """
    model_id: str
    repo_id: str
    required_sfreq: int | None = None
    required_channels: int | None = None
    recommended_window_sec: float | None = None
    max_window_sec: float | None = None
    notes: str = ""


def validate_pretrained_inputs(
    spec: PretrainedInputSpec,
    *,
    in_channels: int,
    n_times: int,
    sfreq: int,
) -> None:
    """
    Validate that caller-supplied inputs satisfy pretrained model constraints.

    Args:
        spec: PretrainedInputSpec for the model being instantiated.
        in_channels: Number of EEG channels.
        n_times: Number of timesteps per window.
        sfreq: Sampling frequency in Hz.

    Raises:
        ValueError: If any hard constraint is violated.

    Warns:
        UserWarning: If recommended_window_sec differs from actual window.
    """
    if spec.required_sfreq is not None and sfreq != spec.required_sfreq:
        raise ValueError(
            f"{spec.model_id} requires sfreq={spec.required_sfreq}, got {sfreq}. "
            f"Use a config compatible with {spec.repo_id}."
        )
    if spec.required_channels is not None and in_channels != spec.required_channels:
        raise ValueError(
            f"{spec.model_id} requires n_channels={spec.required_channels}, got {in_channels}."
        )
    if sfreq <= 0:
        raise ValueError(f"{spec.model_id}: invalid sfreq={sfreq}.")
    window_sec = n_times / sfreq
    if spec.max_window_sec is not None and window_sec > spec.max_window_sec:
        raise ValueError(
            f"{spec.model_id} max window={spec.max_window_sec}s, got {window_sec:.2f}s."
        )
    if spec.recommended_window_sec is not None and abs(window_sec - spec.recommended_window_sec) > 1e-6:
        warnings.warn(
            f"{spec.model_id} recommended window={spec.recommended_window_sec}s, "
            f"got {window_sec:.2f}s. {spec.notes}".strip(),
            stacklevel=3,
        )
