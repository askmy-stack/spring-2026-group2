from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_pipeline.pipeline.run_pipeline import main as run_pipeline_main


if __name__ == "__main__":
    default_config = PACKAGE_ROOT / "configs" / "config.yaml"
    args = sys.argv[1:]
    if "--config" not in args:
        args = ["--config", str(default_config), *args]
    sys.argv = [sys.argv[0], *args]
    run_pipeline_main()
