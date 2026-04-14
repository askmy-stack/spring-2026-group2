from __future__ import annotations

import warnings
from dataclasses import dataclass


@dataclass(frozen=True)
class PretrainedInputSpec:
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
    if spec.required_sfreq is not None and sfreq != spec.required_sfreq:
        raise ValueError(
            f"{spec.model_id} expects sfreq={spec.required_sfreq}, got sfreq={sfreq}. "
            f"Use a model-specific config compatible with {spec.repo_id}."
        )

    if spec.required_channels is not None and in_channels != spec.required_channels:
        raise ValueError(
            f"{spec.model_id} expects n_channels={spec.required_channels}, got n_channels={in_channels}. "
            f"Use a model-specific config compatible with {spec.repo_id}."
        )

    if sfreq <= 0:
        raise ValueError(f"{spec.model_id} received invalid sfreq={sfreq}.")

    window_sec = n_times / sfreq
    if spec.max_window_sec is not None and window_sec > spec.max_window_sec:
        raise ValueError(
            f"{spec.model_id} expects window_sec<={spec.max_window_sec}, got {window_sec:.2f}s."
        )

    if spec.recommended_window_sec is not None and abs(window_sec - spec.recommended_window_sec) > 1e-6:
        warnings.warn(
            f"{spec.model_id} is typically used with window_sec={spec.recommended_window_sec}, "
            f"but received {window_sec:.2f}s. {spec.notes}".strip(),
            stacklevel=3,
        )
