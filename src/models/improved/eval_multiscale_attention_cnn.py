from __future__ import annotations

import sys

from ._launcher import run_model_command


if __name__ == "__main__":
    raise SystemExit(run_model_command("multiscale_attention_cnn", "eval", sys.argv[1:]))
