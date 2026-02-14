from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("YAML must contain a top-level dictionary.")
    return data


def get(cfg: Dict[str, Any], path: str, default=None):
    """
    Safe nested get.
    path: "preprocess.filter.l_freq"
    """
    cur: Any = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur
