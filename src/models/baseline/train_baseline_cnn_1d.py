from __future__ import annotations

import sys

from ._launcher import run_model_command


if __name__ == "__main__":
    raise SystemExit(run_model_command("baseline_cnn_1d", "train", sys.argv[1:]))
