from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REQUIRED_TOP_LEVEL_KEYS = {
    "experiment_name",
    "model_type",
    "target_col",
    "paths",
    "data",
    "training",
    "model_params",
    "evaluation",
    "plots",
}


def _expand_value(value: Any, config_dir: Path) -> Any:
    """
    Recursively expand path strings in config values.

    Handles:
      - ~ (home directory expansion)
      - Relative paths (resolved against the config file's own directory,
        so configs are portable across machines and clone locations)

    Args:
        value: Any config value (str, dict, list, or scalar).
        config_dir: Directory of the config file, used as base for relative paths.

    Returns:
        Value with all path strings fully resolved.
    """
    if isinstance(value, str) and ("~" in value or value.startswith("./") or value.startswith("../")):
        resolved = Path(value).expanduser()
        if not resolved.is_absolute():
            resolved = (config_dir / resolved).resolve()
        return str(resolved)
    if isinstance(value, dict):
        return {k: _expand_value(v, config_dir) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_value(v, config_dir) for v in value]
    return value


def load_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load and validate a YAML config file.

    Resolves all relative paths against the config file's own directory,
    making configs portable — no hardcoded usernames or repo folder names.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Validated config dict with all paths fully resolved.

    Raises:
        ValueError: If required keys are missing.
    """
    config_path = Path(config_path).expanduser().resolve()
    config_dir  = config_path.parent

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {config_path} must be a YAML mapping.")

    missing = REQUIRED_TOP_LEVEL_KEYS - set(cfg.keys())
    if missing:
        raise ValueError(f"Missing required config keys: {sorted(missing)}")

    cfg = _expand_value(cfg, config_dir)

    path_keys = {"train_csv", "val_csv", "test_csv", "output_dir"}
    missing_path_keys = path_keys - set(cfg["paths"].keys())
    if missing_path_keys:
        raise ValueError(f"Missing required paths keys: {sorted(missing_path_keys)}")

    return cfg
