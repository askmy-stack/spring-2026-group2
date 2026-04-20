from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

THIS_DIR = Path(__file__).parent
SRC_DIR = THIS_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from model.factory import create_model, list_models


@dataclass
class LayerRecord:
    name: str
    module_type: str
    output_shape: str
    trainable_params: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize EEG model architectures and save layer summaries.")
    parser.add_argument("--model", choices=list_models(), help="Single model to visualize.")
    parser.add_argument("--all", action="store_true", help="Visualize all available models.")
    parser.add_argument("--channels", type=int, default=16, help="Input channels.")
    parser.add_argument("--samples", type=int, default=256, help="Input samples per window.")
    parser.add_argument("--num-classes", type=int, default=2, help="Output classes.")
    return parser.parse_args()


def is_leaf_module(module: nn.Module) -> bool:
    return len(list(module.children())) == 0


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def format_shape(output: Any) -> str:
    if isinstance(output, torch.Tensor):
        return str(tuple(output.shape))
    if isinstance(output, (list, tuple)):
        parts = []
        for item in output:
            if isinstance(item, torch.Tensor):
                parts.append(str(tuple(item.shape)))
            else:
                parts.append(type(item).__name__)
        return "[" + ", ".join(parts) + "]"
    return type(output).__name__


def summarize_model(model: nn.Module, input_shape: tuple[int, int, int]) -> list[LayerRecord]:
    records: list[LayerRecord] = []
    hooks = []

    for name, module in model.named_modules():
        if not name or not is_leaf_module(module):
            continue

        def _make_hook(module_name: str, layer: nn.Module):
            def _hook(_module, _inputs, output):
                records.append(
                    LayerRecord(
                        name=module_name,
                        module_type=layer.__class__.__name__,
                        output_shape=format_shape(output),
                        trainable_params=count_params(layer),
                    )
                )
            return _hook

        hooks.append(module.register_forward_hook(_make_hook(name, module)))

    model.eval()
    dummy_input = torch.randn(*input_shape)
    with torch.no_grad():
        _ = model(dummy_input)

    for hook in hooks:
        hook.remove()
    return records


def save_records_csv(path: Path, records: list[LayerRecord]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "module_type", "output_shape", "trainable_params"])
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)


def save_architecture_text(path: Path, model_name: str, model: nn.Module, total_params: int, input_shape: tuple[int, int, int]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Input shape: {input_shape}\n")
        f.write(f"Trainable params: {total_params:,}\n")
        f.write("=" * 80 + "\n")
        f.write(str(model))
        f.write("\n")


def save_summary_json(path: Path, model_name: str, total_params: int, input_shape: tuple[int, int, int], records: list[LayerRecord]):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model_name,
        "input_shape": input_shape,
        "trainable_params": total_params,
        "num_layers": len(records),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def visualize_one(model_name: str, channels: int, samples: int, num_classes: int):
    model = create_model(model_name, in_channels=channels, num_classes=num_classes)
    input_shape = (1, channels, samples)
    total_params = count_params(model)
    records = summarize_model(model, input_shape)

    out_dir = SRC_DIR / "results" / "model_viz" / model_name
    save_architecture_text(out_dir / "architecture.txt", model_name, model, total_params, input_shape)
    save_records_csv(out_dir / "layer_summary.csv", records)
    save_summary_json(out_dir / "summary.json", model_name, total_params, input_shape, records)

    print("=" * 72)
    print(f"Model            : {model_name}")
    print(f"Input shape      : {input_shape}")
    print(f"Trainable params : {total_params:,}")
    print(f"Saved to         : {out_dir}")
    print("=" * 72)


def main():
    args = parse_args()
    if not args.all and not args.model:
        raise SystemExit("Pass either --model <name> or --all")

    model_names = list_models() if args.all else [args.model]
    for model_name in model_names:
        visualize_one(model_name, args.channels, args.samples, args.num_classes)


if __name__ == "__main__":
    main()
