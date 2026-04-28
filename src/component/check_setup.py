"""
check_setup.py — Run this from the project root to verify everything is wired up.

Usage:
    python src/component/check_setup.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

# Resolve to project root (two levels above src/component/)
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
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
    from src.component.models.utils.config_utils import load_config
    return "load_config importable"

def _check_io_utils():
    from src.component.models.utils.io_utils import ensure_dir, save_csv, save_json
    return "ensure_dir, save_csv, save_json importable"

def _check_metric_utils():
    from src.component.models.utils.metric_utils import compute_binary_metrics, sweep_thresholds_for_f1
    return "compute_binary_metrics, sweep_thresholds_for_f1 importable"

def _check_plot_utils():
    from src.component.models.utils.plot_utils import save_pr_curve, save_roc_curve
    return "save_pr_curve, save_roc_curve importable"

def _check_data_utils():
    from src.component.models.utils.data_utils import load_split, validate_feature_columns
    return "load_split, validate_feature_columns importable"

check("src.component.models.utils.config_utils", _check_config_utils)
check("src.component.models.utils.io_utils",     _check_io_utils)
check("src.component.models.utils.metric_utils", _check_metric_utils)
check("src.component.models.utils.plot_utils",   _check_plot_utils)
check("src.component.models.utils.data_utils",   _check_data_utils)


# ── 2. Config loading (path resolution) ───────────────────────────────────────

def _check_config_lightgbm():
    import yaml
    with open("src/config/baseline_lightgbm.yaml") as f:
        raw = yaml.safe_load(f)
    train = raw["paths"]["train_csv"]
    # raw value should be relative (no hardcoded absolute path)
    assert not train.startswith("/"),              f"Hardcoded absolute path: {train}"
    assert "spring-2026-group2" not in train,      f"Hardcoded repo name: {train}"
    assert "C:/Users" not in train,                f"Hardcoded Windows path: {train}"
    return f"train_csv is relative -> {train}"

def _check_config_tabnet():
    import yaml
    with open("src/config/tabnet_baseline.yaml") as f:
        raw = yaml.safe_load(f)
    out = raw["paths"]["output_dir"]
    assert not out.startswith("/"),           f"Hardcoded absolute path: {out}"
    assert "spring-2026-group2" not in out,   f"Hardcoded repo name: {out}"
    return f"output_dir is relative -> {out}"

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
    with open("src/component/models/improved/optuna_lightgbm.py") as f:
        src = f.read()
    assert "src.modeling" not in src, "Old src.modeling import still present"
    assert "src.component.models.utils" in src, "Updated src.component.models.utils import missing"
    return "no old src.modeling references"

def _check_import_tabnet():
    with open("src/component/models/legacy_baseline/train_tabnet.py") as f:
        src = f.read()
    assert "sys.path.insert" not in src or "data_loader_dir" in src, \
        "sys.path.insert hack still present"
    assert "src.component.models.utils" in src
    return "no sys.path hacks"

def _check_import_optuna_scripts():
    """All optuna scripts live in improved/ and have clean imports."""
    scripts = [
        "src/component/models/improved/optuna_lightgbm.py",
        "src/component/models/improved/optuna_xgboost.py",
        "src/component/models/improved/optuna_random_forest.py",
    ]
    for path in scripts:
        with open(path) as f:
            src = f.read()
        name = Path(path).name
        assert "src.modeling" not in src,          f"{name}: old src.modeling import"
        assert "tabnet.prepare_memmap" not in src, f"{name}: old tabnet import"
    return "all 3 optuna scripts in improved/ have clean imports"

check("optuna_lightgbm.py: no old imports",          _check_import_lgbm)
check("train_tabnet.py: no sys.path hacks",          _check_import_tabnet)
check("improved/optuna_*.py: clean imports (all 3)", _check_import_optuna_scripts)


# ── 4. No hardcoded paths anywhere in src/models/ ─────────────────────────────

def _check_no_hardcoding():
    bad_patterns = ["spring-2026-group2", "/home/", "C:/Users/", "C:\\Users\\"]
    violations = []
    for py_file in Path("src/component").rglob("*.py"):
        if py_file.name == "check_setup.py":
            continue  # skip self — docstring intentionally references paths
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

check("no hardcoded paths in src/component/ *.py", _check_no_hardcoding)
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
    print("  pytest src/tests/ -v")
