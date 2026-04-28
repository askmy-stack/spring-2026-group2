"""
Publish trained seizure-detection checkpoints to the Hugging Face Hub.

Walks one or more local checkpoint directories, splits every matching
unified-schema ``.pt`` file into three portable artefacts
(``model.safetensors`` + ``config.json`` + ``metrics.json``), renders a
per-model Markdown card, and uploads the whole tree to a single monorepo
on the Hub in one atomic commit via ``HfApi.upload_folder``.

Auth:
    Reads the token from the ``HF_TOKEN`` env var or the cache file
    written by ``huggingface-cli login``.

Example:
    python -m src.models.utils.hf_publish \\
        --repo-id <your-user>/chbmit-seizure-models \\
        --ckpt-dirs src/models/improved_lstm_models/checkpoints \\
        --include 'im*_best.pt' 'improved_lstm_best.pt' \\
        --visibility public
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch

logger = logging.getLogger(__name__)

DEFAULT_CKPT_DIRS = [
    "src/models/lstm_benchmark_models/checkpoints",
    "src/models/improved_lstm_models/checkpoints",
]
DEFAULT_INCLUDE = ["im*_best.pt", "improved_lstm_best.pt"]
DEFAULT_STAGING = Path("hf_publish_staging")


@dataclass
class PublishPlan:
    """Describes one model ready to publish."""
    name: str                  # subfolder on the hub (e.g. "im7_attention_lstm")
    src_ckpt: Path             # local .pt path
    staging_dir: Path          # where we write safetensors + config + metrics + README


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo-id", required=True,
                        help="Target HF repo, e.g. 'your-user/chbmit-seizure-models'.")
    parser.add_argument("--ckpt-dirs", nargs="+", default=DEFAULT_CKPT_DIRS,
                        help="Local dirs to scan for checkpoints.")
    parser.add_argument("--include", nargs="+", default=DEFAULT_INCLUDE,
                        help="Glob patterns (checkpoint filenames) to publish.")
    parser.add_argument("--visibility", choices=["public", "private"], default="public")
    parser.add_argument("--staging-dir", type=Path, default=DEFAULT_STAGING,
                        help="Local staging dir for rendered artefacts.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Render artefacts locally but don't upload.")
    parser.add_argument("--commit-message",
                        default="publish improved benchmarks + ensemble")
    parser.add_argument("--overwrite-staging", action="store_true",
                        help="Wipe the staging dir before rendering.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()

    if args.overwrite_staging and args.staging_dir.exists():
        shutil.rmtree(args.staging_dir)
    args.staging_dir.mkdir(parents=True, exist_ok=True)

    ckpts = _discover_ckpts([Path(d) for d in args.ckpt_dirs], args.include)
    if not ckpts:
        logger.error("No checkpoints matched %s in %s", args.include, args.ckpt_dirs)
        return
    logger.info("Publishing %d checkpoint(s): %s", len(ckpts), [c.name for c in ckpts])

    plans = [_stage_checkpoint(ckpt, args.staging_dir) for ckpt in ckpts]
    _render_top_level_readme(args.staging_dir, plans, args.repo_id)

    if args.dry_run:
        logger.info("Dry run complete — artefacts at %s", args.staging_dir)
        return

    _upload_to_hub(
        repo_id=args.repo_id,
        folder=args.staging_dir,
        visibility=args.visibility,
        commit_message=args.commit_message,
    )
    logger.info("Pushed https://huggingface.co/%s", args.repo_id)


# ---------------------------------------------------------------------------
# Discovery / staging
# ---------------------------------------------------------------------------


def _discover_ckpts(dirs: Iterable[Path], include: List[str]) -> List[Path]:
    """Return sorted list of .pt files matching any ``include`` glob."""
    matches: List[Path] = []
    for d in dirs:
        if not d.exists():
            logger.warning("ckpt dir not found: %s", d)
            continue
        for p in d.glob("*.pt"):
            if p.parent.name == "sub_runs":
                continue
            if any(fnmatch.fnmatch(p.name, pat) for pat in include):
                matches.append(p)
    return sorted(matches)


def _stage_checkpoint(ckpt: Path, staging_root: Path) -> PublishPlan:
    """Split a unified-schema .pt into safetensors + config + metrics + README."""
    from safetensors.torch import save_file  # local import to keep util optional

    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "model_state_dict" not in payload:
        raise ValueError(f"{ckpt} is not a unified-schema checkpoint")

    name = _subfolder_name(ckpt)
    subdir = staging_root / name
    subdir.mkdir(parents=True, exist_ok=True)

    # Weights
    state = {k: v.contiguous() for k, v in payload["model_state_dict"].items()}
    save_file(state, str(subdir / "model.safetensors"))

    # Config
    cfg = {
        "model_class": payload.get("model_class"),
        "model_builder": payload.get("model_builder"),
        "model_config": payload.get("model_config", {}),
        "input_spec": payload.get("input_spec", {}),
        "preprocess": payload.get("preprocess", {}),
        "schema_version": payload.get("schema_version", 1),
        "optimal_threshold": float(payload.get("optimal_threshold", 0.5)),
        "git_commit": payload.get("git_commit"),
        "epoch": int(payload.get("epoch", 0)),
    }
    (subdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Metrics (val + any neighbouring <name>_metrics.json written by the trainer)
    metrics = {"val": payload.get("val_metrics", {})}
    sidecar = ckpt.with_name(ckpt.stem.replace("_best", "") + "_metrics.json")
    if sidecar.exists():
        metrics.update(json.loads(sidecar.read_text()))
    (subdir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Model card
    (subdir / "README.md").write_text(_render_model_card(name, cfg, metrics))

    logger.info("Staged %s -> %s", ckpt.name, subdir)
    return PublishPlan(name=name, src_ckpt=ckpt, staging_dir=subdir)


def _subfolder_name(ckpt: Path) -> str:
    """Derive the hub subfolder name from a checkpoint filename."""
    stem = ckpt.stem
    return stem.removesuffix("_best") or stem


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_model_card(name: str, cfg: Dict, metrics: Dict) -> str:
    """Render a per-model README.md from config + metrics."""
    val = metrics.get("val", {}) or {}
    test = metrics.get("test", {}) or {}
    threshold = cfg.get("optimal_threshold", 0.5)
    model_class = cfg.get("model_class", "(unknown class)")
    return _TEMPLATE.format(
        name=name,
        model_class=model_class,
        threshold=threshold,
        val_f1=val.get("f1", float("nan")),
        val_auroc=val.get("auroc", float("nan")),
        val_sens=val.get("sens", float("nan")),
        val_spec=val.get("spec", float("nan")),
        test_f1=test.get("f1", float("nan")),
        test_auroc=test.get("auroc", float("nan")),
        test_sens=test.get("sens", float("nan")),
        test_spec=test.get("spec", float("nan")),
    )


def _render_top_level_readme(staging: Path, plans: List[PublishPlan], repo_id: str) -> None:
    """Write the monorepo-level README listing all published models."""
    rows = "\n".join(f"| `{p.name}` | See [./{p.name}/](./{p.name}/) |" for p in plans)
    (staging / "README.md").write_text(
        f"# {repo_id}\n\n"
        "Seizure-detection models trained on CHB-MIT scalp EEG, "
        "subject-independent 70/15/15 split.\n\n"
        "## Published models\n\n"
        "| Subfolder | Details |\n|---|---|\n" + rows + "\n\n"
        "Each subfolder contains `model.safetensors`, `config.json`, "
        "`metrics.json`, and a model card README.\n"
    )


_TEMPLATE = """---
license: mit
tags: [eeg, seizure-detection, chb-mit, pytorch]
---
# {name} — CHB-MIT seizure classifier

Source class: `{model_class}`.

## Metrics

| Split | F1 | AUROC | Sensitivity | Specificity |
|---|---|---|---|---|
| val  | {val_f1:.4f}  | {val_auroc:.4f}  | {val_sens:.4f}  | {val_spec:.4f}  |
| test | {test_f1:.4f} | {test_auroc:.4f} | {test_sens:.4f} | {test_spec:.4f} |

Decision threshold (tuned on val for F1): **{threshold:.3f}**.

## Files

- `model.safetensors` — state_dict (safe, portable).
- `config.json` — model_class, model_config, input_spec, preprocess, threshold.
- `metrics.json` — val & test F1 / AUROC / sens / spec.

## Load

```python
from src.component.models.utils.hf_publish import rehydrate_from_hub
model, payload = rehydrate_from_hub(
    repo_id="...",       # e.g. "your-user/chbmit-seizure-models"
    subfolder="{name}",
)
```
"""


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


def _upload_to_hub(
    repo_id: str,
    folder: Path,
    visibility: str,
    commit_message: str,
) -> None:
    """Create the repo if needed, then upload ``folder`` in one commit."""
    from huggingface_hub import HfApi  # local import to keep util optional

    token = os.environ.get("HF_TOKEN") or None
    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=(visibility == "private"),
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(folder),
        commit_message=commit_message,
        repo_type="model",
    )


# ---------------------------------------------------------------------------
# Re-loading from the Hub
# ---------------------------------------------------------------------------


def rehydrate_from_hub(
    repo_id: str,
    subfolder: str,
    *,
    map_location: str = "cpu",
    token: Optional[str] = None,
):
    """
    Download ``model.safetensors`` + ``config.json`` from a HF repo and
    reconstruct a loaded ``nn.Module`` plus the unified payload dict.

    The caller gets exactly what :func:`load_checkpoint` would have
    returned for a local unified-schema ``.pt`` — so downstream code
    (ensemble, inference, eval scripts) needs no branching.

    Args:
        repo_id: ``<user-or-org>/<repo-name>`` on the Hugging Face Hub.
        subfolder: Per-model directory inside the repo (e.g.
            ``"im7_attention_lstm"``).
        map_location: Device for tensor loading (default ``"cpu"``).
        token: Optional HF access token override. When ``None``, uses
            ``HF_TOKEN`` env or the CLI-cached token; both work for
            public repos since no auth is required to download.

    Returns:
        Tuple ``(model, payload)`` — identical contract to
        :func:`src.models.utils.checkpoint.load_checkpoint`.
    """
    from huggingface_hub import hf_hub_download     # local import: keep util optional
    from safetensors.torch import load_file
    from .checkpoint import load_checkpoint

    token = token or os.environ.get("HF_TOKEN") or None
    weights_path = hf_hub_download(
        repo_id=repo_id, filename=f"{subfolder}/model.safetensors",
        token=token,
    )
    config_path = hf_hub_download(
        repo_id=repo_id, filename=f"{subfolder}/config.json",
        token=token,
    )
    cfg = json.loads(Path(config_path).read_text())
    state = load_file(weights_path, device=str(map_location))

    # Rehydrate into the unified payload schema so load_checkpoint can
    # delegate model reconstruction just as it does for local .pt files.
    staged_ckpt = _stage_pt_from_hub(cfg, state)
    return load_checkpoint(staged_ckpt, map_location=map_location, build_model=True)


def _stage_pt_from_hub(cfg: Dict, state: Dict[str, torch.Tensor]) -> Path:
    """Write a temporary unified-schema .pt file so load_checkpoint can read it."""
    import tempfile

    payload = {
        "schema_version": cfg.get("schema_version", 1),
        "model_class": cfg.get("model_class"),
        "model_builder": cfg.get("model_builder"),
        "model_config": cfg.get("model_config", {}),
        "model_state_dict": state,
        "optimizer_state_dict": None,
        "epoch": int(cfg.get("epoch", 0)),
        "val_metrics": {},
        "optimal_threshold": float(cfg.get("optimal_threshold", 0.5)),
        "input_spec": cfg.get("input_spec", {}),
        "preprocess": cfg.get("preprocess", {}),
        "git_commit": cfg.get("git_commit"),
    }
    tmp = Path(tempfile.mkstemp(suffix=".pt", prefix="hf_rehydrate_")[1])
    torch.save(payload, tmp)
    return tmp


if __name__ == "__main__":
    main()
