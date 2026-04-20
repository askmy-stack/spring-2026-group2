from __future__ import annotations

from models.utils.train_eval import build_eval_parser, build_train_parser, eval_model, train_model


def run_model_command(model_name: str, mode: str, extra_args: list[str] | None = None) -> int:
    if mode == "train":
        parser = build_train_parser(f"Train {model_name}.")
        args = parser.parse_args(["--model", model_name, *(extra_args or [])])
        return train_model(model_name, args)
    if mode == "eval":
        parser = build_eval_parser(f"Evaluate {model_name}.")
        args = parser.parse_args(["--model", model_name, *(extra_args or [])])
        return eval_model(model_name, args)
    raise ValueError(f"Unsupported mode: {mode}")
