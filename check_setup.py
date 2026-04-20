"""
check_setup.py — Run this from the project root to verify everything is wired up.

Usage:
    cd C:/Users/anusu/Desktop/GWU/eeg
    python check_setup.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

# Make sure project root is on the path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"

results: list[tuple[str, str, str]] = []


def check(label: str, fn):
    """Run a check function and record result."""
    try:
        msg = fn()
        results.append((PASS, label, msg or ""))
    except Exception as exc:
        results.append((FAIL, label, str(exc)))


# ── 1. Utils imports ──────────────────────────────────────────────────────────

def _check_config_utils():
    from src.models.utils.config_utils import load_config
    return "load_config importable"

def _check_io_utils():
    from src.models.utils.io_utils import ensure_dir, save_csv, save_json
    return "ensure_dir, save_csv, save_json importable"

def _check_metric_utils():
    from src.models.utils.metric_utils import compute_binary_metrics, sweep_thresholds_for_f1
    return "compute_binary_metrics, sweep_thresholds_for_f1 importable"

def _check_plot_utils():
    from src.models.utils.plot_utils import save_pr_curve, save_roc_curve
    return "save_pr_curve, save_roc_curve importable"

def _check_data_utils():
    from src.models.utils.data_utils import load_split, validate_feature_columns
    return "load_split, validate_feature_columns importable"

check("src.models.utils.config_utils", _check_config_utils)
check("src.models.utils.io_utils",     _check_io_utils)
check("src.models.utils.metric_utils", _check_metric_utils)
check("src.models.utils.plot_utils",   _check_plot_utils)
check("src.models.utils.data_utils",   _check_data_utils)


# ── 2. Config loading (path resolution) ───────────────────────────────────────

def _check_config_lightgbm():
    from src.models.utils.config_utils import load_config
    cfg = load_config("src/config/baseline_lightgbm.yaml")
    train = cfg["paths"]["train_csv"]
    assert "spring-2026-group2" not in train, f"Hardcoded path found: {train}"
    assert not train.startswith("~"),          f"~ not expanded: {train}"
    assert Path(train).is_absolute(),          f"Path not absolute after resolve: {train}"
    return f"train_csv resolved -> {train}"

def _check_config_tabnet():
    from src.models.utils.config_utils import load_config
    cfg = load_config("src/config/tabnet_baseline.yaml")
    out = cfg["paths"]["output_dir"]
    assert "spring-2026-group2" not in out
    return f"output_dir resolved -> {out}"

def _check_config_fe():
    from pathlib import Path
    import yaml
    cfg_path = Path("src/config/feature_engineering.yaml")
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    train = raw["window_index"]["train_csv"]
    assert "spring-2026-group2" not in train, f"Hardcoded: {train}"
    return f"feature_engineering.yaml clean -> {train}"

check("config: baseline_lightgbm.yaml loads & paths resolve", _check_config_lightgbm)
check("config: tabnet_baseline.yaml loads & paths resolve",   _check_config_tabnet)
check("config: feature_engineering.yaml no hardcoding",       _check_config_fe)


# ── 3. Model file imports (top-level, no training triggered) ──────────────────

def _check_import_lgbm():
    with open("src/models/baseline/optuna_lightgbm.py") as f:
        src = f.read()
    assert "src.modeling" not in src, "Old src.modeling import still present"
    assert "src.models.utils" in src,  "New src.models.utils import missing"
    return "no old src.modeling references"

def _check_import_tabnet():
    with open("src/models/baseline/train_tabnet.py") as f:
        src = f.read()
    assert "sys.path.insert" not in src or "data_loader_dir" in src, \
        "sys.path.insert hack still present"
    assert "src.models.utils" in src
    return "no sys.path hacks"

def _check_import_tabnet_advanced():
    with open("src/models/improved/train_tabnet_advanced.py") as f:
        src = f.read()
    assert "tabnet.prepare_memmap" not in src, "Old tabnet.prepare_memmap import still present"
    assert "sys.path.insert" not in src, "sys.path.insert hack still present"
    assert "src.models.utils" in src
    return "train_tabnet_advanced imports clean"

check("optuna_lightgbm.py: no old imports",             _check_import_lgbm)
check("train_tabnet.py: no sys.path hacks",             _check_import_tabnet)
check("train_tabnet_advanced.py: clean imports",        _check_import_tabnet_advanced)


# ── 4. No hardcoded paths anywhere in src/models/ ─────────────────────────────

def _check_no_hardcoding():
    bad_patterns = ["spring-2026-group2", "/home/", "C:/Users/", "C:\\Users\\"]
    violations = []
    for py_file in Path("src/models").rglob("*.py"):
        text = py_file.read_text(encoding="utf-8", errors="ignore")
        for pat in bad_patterns:
            if pat in text:
                violations.append(f"{py_file.name}: contains '{pat}'")
    if violations:
        raise ValueError("\n  " + "\n  ".join(violations))
    return "all clean"

def _check_no_hardcoding_configs():
    bad_patterns = ["spring-2026-group2", "/home/", "C:/Users/"]
    violations = []
    for yaml_file in Path("src/config").rglob("*.yaml"):
        text = yaml_file.read_text(encoding="utf-8", errors="ignore")
        for pat in bad_patterns:
            if pat in text:
                violations.append(f"{yaml_file.name}: contains '{pat}'")
    if violations:
        raise ValueError("\n  " + "\n  ".join(violations))
    return "all configs clean"

check("no hardcoded paths in src/models/  *.py",   _check_no_hardcoding)
check("no hardcoded paths in src/config/ *.yaml",  _check_no_hardcoding_configs)


# ── 5. Existing tests ─────────────────────────────────────────────────────────

def _check_pytest_available():
    import pytest
    return f"pytest {pytest.__version__} available — run: pytest tests/ -v"

check("pytest available", _check_pytest_available)


# ── Print results ─────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  SETUP CHECK RESULTS")
print("=" * 65)

passed = sum(1 for r in results if r[0] == PASS)
failed = sum(1 for r in results if r[0] == FAIL)

for status, label, msg in results:
    print(f"\n{status}  {label}")
    if msg:
        prefix = "       "
        for line in msg.splitlines():
            print(f"{prefix}{line}")

print("\n" + "=" * 65)
print(f"  {passed} passed   {failed} failed   ({len(results)} total)")
print("=" * 65)

if failed > 0:
    print("\nFix the FAIL items above, then re-run this script.")
    sys.exit(1)
else:
    print("\nAll checks passed. Run full tests with:")
    print("  pytest tests/ -v")
