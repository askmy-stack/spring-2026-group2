"""
Config validation — ensures config.yaml has all required keys before training.

Usage:
    from src.component.models.utils.config_validator import validate_config
    validate_config(config)  # raises KeyError with helpful message if anything is missing
"""
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

REQUIRED_KEYS: Dict[str, List[str]] = {
    "data": ["n_channels", "time_steps"],
    "training": [
        "learning_rate", "batch_size", "num_epochs", "pos_weight",
        "early_stopping_patience", "gradient_clip", "weight_decay",
    ],
    "focal_loss": ["gamma", "reduction"],
    "models": [],
    "outputs": ["checkpoint_dir"],
}


def validate_config(config: Dict[str, Any], model_section: str | None = None) -> None:
    """
    Validate that all required config keys are present.

    Args:
        config: Parsed config dictionary.
        model_section: Optional model subsection to validate
            (e.g. 'lstm_benchmark', 'hugging_face_mamba_moe').

    Raises:
        KeyError: If any required key is missing, with a descriptive message.
    """
    missing = []
    for section, keys in REQUIRED_KEYS.items():
        if section not in config:
            missing.append(f"Top-level section '{section}'")
            continue
        for key in keys:
            if key not in config[section]:
                missing.append(f"{section}.{key}")

    if model_section is not None:
        if "models" in config:
            if model_section not in config["models"]:
                missing.append(f"models.{model_section}")
        else:
            missing.append(f"models (needed for models.{model_section})")

    if missing:
        raise KeyError(
            f"Missing required config keys: {', '.join(missing)}. "
            f"Check src/models/config.yaml."
        )
    logger.debug("Config validation passed.")
