from __future__ import annotations

from typing import Dict, Any

import numpy as np
from scipy.interpolate import CubicSpline


def augment(data: np.ndarray, label: int, cfg: Dict[str, Any]) -> np.ndarray:
    aug_cfg = cfg.get("augmentation", {})
    if not aug_cfg.get("enable", True):
        return data

    prob = aug_cfg.get("seizure_prob", 0.8) if label == 1 else aug_cfg.get("background_prob", 0.3)

    if np.random.random() >= prob:
        return data

    if aug_cfg.get("time_warp", {}).get("enable", True) and np.random.random() < 0.5:
        data = time_warp(data, aug_cfg.get("time_warp", {}).get("sigma", 0.15))

    if aug_cfg.get("magnitude_scale", {}).get("enable", True) and np.random.random() < 0.5:
        data = magnitude_scale(data, aug_cfg.get("magnitude_scale", {}).get("sigma", 0.15))

    if aug_cfg.get("add_noise", {}).get("enable", True) and np.random.random() < 0.3:
        data = add_noise(data, aug_cfg.get("add_noise", {}).get("snr_db", 25))

    if aug_cfg.get("time_shift", {}).get("enable", True) and np.random.random() < 0.3:
        data = time_shift(data, aug_cfg.get("time_shift", {}).get("max_samples", 15))

    return data


def time_warp(signal: np.ndarray, sigma: float = 0.15) -> np.ndarray:
    channels, time_steps = signal.shape
    orig_steps = np.arange(time_steps)
    warp_steps = np.linspace(0, time_steps - 1, num=5)
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=5)
    warper = CubicSpline(warp_steps, warp_steps * random_warps)
    warped_steps = np.clip(warper(orig_steps), 0, time_steps - 1)
    result = np.zeros_like(signal)
    for c in range(channels):
        result[c] = np.interp(warped_steps, orig_steps, signal[c])
    return result


def magnitude_scale(signal: np.ndarray, sigma: float = 0.15) -> np.ndarray:
    scale = np.random.normal(loc=1.0, scale=sigma, size=(signal.shape[0], 1))
    return signal * scale


def add_noise(signal: np.ndarray, snr_db: float = 25) -> np.ndarray:
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(max(noise_power, 1e-20)), signal.shape)
    return signal + noise


def time_shift(signal: np.ndarray, max_samples: int = 15) -> np.ndarray:
    shift = np.random.randint(-max_samples, max_samples + 1)
    if shift == 0:
        return signal
    result = np.zeros_like(signal)
    if shift > 0:
        result[:, shift:] = signal[:, :-shift]
    else:
        result[:, :shift] = signal[:, -shift:]
    return result