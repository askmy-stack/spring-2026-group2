from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tests.test_pipeline import run_all

if __name__ == "__main__":
    run_all()