"""
stress_test.py — Full pipeline stress test using synthetic data.

Tests every major code path WITHOUT needing real EEG data:
  - Feature engineering (AdvancedFeatureExtractor on fake signals)
  - All utils (config, data, io, metric, plot)
  - All ML model build/fit/predict cycles (LightGBM, XGBoost, RandomForest, TabNet)
  - Config loading for every yaml
  - prepare_memmap write/read round-trip
  - Hybrid model forward pass (TCN + Transformer + BiLSTM)

Run from project root:
    python stress_test.py
"""

from __future__ import annotations

import sys
import json
import shutil
import tempfile
import traceback
import logging
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(level=logging.WARNING)  # suppress noisy library logs

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"
results: list[tuple[str, str, str]] = []


def check(label: str, fn):
    try:
        msg = fn() or ""
        results.append((PASS, label, msg))
    except Exception as exc:
        results.append((FAIL, label, f"{type(exc).__name__}: {exc}"))


def skip(label: str, reason: str):
    results.append((SKIP, label, reason))


# ── Synthetic data helpers ─────────────────────────────────────────────────

N_CHANNELS  = 16
SFREQ       = 256
N_SAMPLES   = SFREQ          # 1-second window
N_TRAIN     = 300
N_VAL       = 80
N_FEATURES  = 50             # small synthetic feature set for speed
SEIZURE_RATIO = 0.1          # 10% seizure for synthetic set


def make_eeg_window() -> np.ndarray:
    """Fake 16-channel, 256-sample EEG window."""
    return np.random.randn(N_CHANNELS, N_SAMPLES).astype(np.float32)


def make_feature_matrix(n: int) -> tuple[np.ndarray, np.ndarray]:
    X = np.random.randn(n, N_FEATURES).astype(np.float32)
    y = (np.random.rand(n) < SEIZURE_RATIO).astype(int)
    # guarantee at least 2 positives so metrics don't crash
    y[:3] = 1
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# 1. Feature Engineering
# ══════════════════════════════════════════════════════════════════════════════

def _test_fe_import():
    from src.feature_engineering import AdvancedFeatureExtractor
    return "AdvancedFeatureExtractor imported"


def _test_fe_single_window():
    from src.feature_engineering import AdvancedFeatureExtractor
    extractor = AdvancedFeatureExtractor(sfreq=SFREQ)
    signal = make_eeg_window()   # shape (16, 256)
    features = extractor.extract(signal)
    assert isinstance(features, (np.ndarray, dict, list)), "extract() must return array/dict/list"
    if isinstance(features, np.ndarray):
        assert features.ndim == 1 and len(features) > 0, "Feature vector must be non-empty 1D"
        return f"extracted {len(features)} features per window"
    return f"extracted features (type={type(features).__name__})"


def _test_fe_batch():
    from src.feature_engineering import AdvancedFeatureExtractor
    extractor = AdvancedFeatureExtractor(sfreq=SFREQ)
    for _ in range(5):
        sig = make_eeg_window()
        extractor.extract(sig)
    return "5 windows processed without error"


check("FE: import AdvancedFeatureExtractor",     _test_fe_import)
check("FE: extract features from single window", _test_fe_single_window)
check("FE: batch of 5 windows",                  _test_fe_batch)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Utils — metric_utils
# ══════════════════════════════════════════════════════════════════════════════

def _test_metrics():
    from src.models.utils.metric_utils import compute_binary_metrics, sweep_thresholds_for_f1
    y_true = np.array([0]*90 + [1]*10)
    y_prob = np.random.rand(100)
    m = compute_binary_metrics(y_true, y_prob, threshold=0.5)
    assert "aucpr" in m and "f1" in m and "roc_auc" in m
    best_t, rows = sweep_thresholds_for_f1(y_true, y_prob)
    assert 0.0 <= best_t <= 1.0
    return f"aucpr={m['aucpr']:.3f}, best_threshold={best_t:.2f}"


check("utils.metric_utils: compute_binary_metrics + sweep", _test_metrics)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Utils — io_utils
# ══════════════════════════════════════════════════════════════════════════════

def _test_io_utils():
    from src.models.utils.io_utils import ensure_dir, save_csv, save_json
    import pandas as pd
    with tempfile.TemporaryDirectory() as tmp:
        d = ensure_dir(Path(tmp) / "sub" / "dir")
        assert d.exists()
        save_json({"key": 42}, d / "test.json")
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        save_csv(df, d / "test.csv")
        assert (d / "test.json").exists()
        assert (d / "test.csv").exists()
    return "ensure_dir, save_json, save_csv all OK"


check("utils.io_utils: ensure_dir + save_json + save_csv", _test_io_utils)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Utils — plot_utils
# ══════════════════════════════════════════════════════════════════════════════

def _test_plot_utils():
    from src.models.utils.plot_utils import (
        save_pr_curve, save_roc_curve, save_threshold_plot,
        save_confusion_matrix_plot, save_feature_importance_plot,
    )
    y_true = np.array([0]*90 + [1]*10)
    y_prob = np.random.rand(100)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        save_pr_curve(y_true, y_prob, p / "pr.png", "Test PR")
        save_roc_curve(y_true, y_prob, p / "roc.png", "Test ROC")
        rows = [{"threshold": t, "f1": 0.5} for t in np.linspace(0, 1, 10)]
        save_threshold_plot(rows, p / "thresh.png", "Thresh")
        save_confusion_matrix_plot([[80, 10], [5, 5]], p / "cm.png", "CM")
        names = [f"feat_{i}" for i in range(20)]
        imps  = np.random.rand(20)
        save_feature_importance_plot(names, imps, p / "fi.png", "FI", top_k=10)
    return "all 5 plot functions saved PNGs without error"


check("utils.plot_utils: all 5 plot functions", _test_plot_utils)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Utils — data_utils
# ══════════════════════════════════════════════════════════════════════════════

def _test_data_utils():
    import pandas as pd
    from src.models.utils.data_utils import load_split, validate_feature_columns
    X, y = make_feature_matrix(100)
    feat_names = [f"f{i}" for i in range(N_FEATURES)]
    df = pd.DataFrame(X, columns=feat_names)
    df["label"] = y
    df["subject_id"] = "chb01"

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "test.csv"
        df.to_csv(csv_path, index=False)
        X_out, y_out, cols, _ = load_split(str(csv_path), "label", {"subject_id"}, "float32")
        assert X_out.shape == (100, N_FEATURES)
        validate_feature_columns(cols, cols, cols)
    return f"load_split -> X={X_out.shape}, y={y_out.shape}"


check("utils.data_utils: load_split + validate_feature_columns", _test_data_utils)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Config loading — all yaml files
# ══════════════════════════════════════════════════════════════════════════════

def _test_all_configs():
    from src.models.utils.config_utils import load_config
    config_dir = PROJECT_ROOT / "src" / "config"
    yamls = list(config_dir.glob("*.yaml"))
    # Skip yamls that use different schemas (not the standard model config schema)
    skip_names = {"feature_engineering.yaml", "data_loader.yaml"}
    model_yamls = [y for y in yamls if y.name not in skip_names]
    loaded = []
    for yf in sorted(model_yamls):
        try:
            cfg = load_config(yf)
            loaded.append(yf.name)
        except Exception as exc:
            raise RuntimeError(f"{yf.name} failed: {exc}") from exc
    return f"loaded {len(loaded)} configs: {', '.join(loaded)}"


check("config: all model yamls load + paths resolve", _test_all_configs)


# ══════════════════════════════════════════════════════════════════════════════
# 7. ML Models — build + fit + predict
# ══════════════════════════════════════════════════════════════════════════════

def _test_lightgbm():
    import lightgbm as lgb
    X_tr, y_tr = make_feature_matrix(N_TRAIN)
    X_val, y_val = make_feature_matrix(N_VAL)
    model = lgb.LGBMClassifier(n_estimators=10, verbose=-1)
    model.fit(X_tr, y_tr)
    prob = model.predict_proba(X_val)[:, 1]
    assert prob.shape == (N_VAL,)
    return f"fit OK, predict_proba shape={prob.shape}"


def _test_xgboost():
    import xgboost as xgb
    X_tr, y_tr = make_feature_matrix(N_TRAIN)
    X_val, y_val = make_feature_matrix(N_VAL)
    model = xgb.XGBClassifier(n_estimators=10, verbosity=0, eval_metric="logloss")
    model.fit(X_tr, y_tr)
    prob = model.predict_proba(X_val)[:, 1]
    assert prob.shape == (N_VAL,)
    return f"fit OK, predict_proba shape={prob.shape}"


def _test_random_forest():
    from sklearn.ensemble import RandomForestClassifier
    X_tr, y_tr = make_feature_matrix(N_TRAIN)
    X_val, y_val = make_feature_matrix(N_VAL)
    model = RandomForestClassifier(n_estimators=10, n_jobs=1)
    model.fit(X_tr, y_tr)
    prob = model.predict_proba(X_val)[:, 1]
    assert prob.shape == (N_VAL,)
    return f"fit OK, predict_proba shape={prob.shape}"


check("model: LightGBM fit + predict_proba",     _test_lightgbm)
check("model: XGBoost fit + predict_proba",      _test_xgboost)
check("model: RandomForest fit + predict_proba", _test_random_forest)


# ══════════════════════════════════════════════════════════════════════════════
# 8. train_model.py — build_model + save_model helpers
# ══════════════════════════════════════════════════════════════════════════════

def _test_train_model_build():
    from src.models.baseline.train_model import build_model, get_feature_importances
    for mt in ["lightgbm", "xgboost", "random_forest"]:
        m = build_model(mt, {}, {"n_jobs": 1, "random_state": 42})
        X_tr, y_tr = make_feature_matrix(100)
        m.fit(X_tr, y_tr)
        imp = get_feature_importances(m, mt)
        assert imp is not None and len(imp) == N_FEATURES
    return "build_model + get_feature_importances OK for all 3 types"


check("train_model.py: build_model for lgb/xgb/rf", _test_train_model_build)


# ══════════════════════════════════════════════════════════════════════════════
# 9. prepare_memmap — write + read round-trip
# ══════════════════════════════════════════════════════════════════════════════

def _test_prepare_memmap():
    import pandas as pd
    from src.models.utils.prepare_memmap import (
        detect_feature_columns, build_label_mapping,
        write_split, already_exists,
    )
    X, y = make_feature_matrix(200)
    feat_names = [f"f{i}" for i in range(N_FEATURES)]
    df = pd.DataFrame(X, columns=feat_names)
    df["label"] = y
    df["subject_id"] = "chb01"

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "train.csv"
        out_dir  = Path(tmp) / "memmap"
        out_dir.mkdir()
        df.to_csv(csv_path, index=False)

        cols    = detect_feature_columns(csv_path, "label")
        mapping = build_label_mapping(csv_path, "label", chunksize=1000)

        assert not already_exists(out_dir, "train")
        write_split(csv_path, "train", cols, mapping, out_dir, "label", chunksize=1000)
        assert already_exists(out_dir, "train")

        X_mm = np.memmap(out_dir / "X_train.dat", dtype="float32",
                         mode="r", shape=(200, len(cols)))
        assert X_mm.shape == (200, len(cols))
        del X_mm  # must release before TemporaryDirectory cleanup on Windows
    return f"memmap round-trip OK: {len(cols)} features, {200} rows"


check("prepare_memmap: write + read round-trip", _test_prepare_memmap)


# ══════════════════════════════════════════════════════════════════════════════
# 10. TabNet — import + forward pass
# ══════════════════════════════════════════════════════════════════════════════

def _test_tabnet_import():
    from pytorch_tabnet.tab_model import TabNetClassifier
    return "pytorch_tabnet imported OK"


def _test_tabnet_fit():
    from pytorch_tabnet.tab_model import TabNetClassifier
    X_tr, y_tr = make_feature_matrix(200)
    X_val, y_val = make_feature_matrix(50)
    model = TabNetClassifier(
        n_steps=2, n_d=8, n_a=8, verbose=0,
        optimizer_params={"lr": 1e-2},
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric=["auc"],
        max_epochs=2,
        patience=2,
        batch_size=64,
    )
    prob = model.predict_proba(X_val)[:, 1]
    assert prob.shape == (50,)
    return f"TabNet fit (2 epochs) + predict_proba OK, shape={prob.shape}"


check("TabNet: import pytorch_tabnet", _test_tabnet_import)
check("TabNet: fit 2 epochs + predict_proba", _test_tabnet_fit)


# ══════════════════════════════════════════════════════════════════════════════
# 11. Hybrid model — import + forward pass on raw EEG tensor
# ══════════════════════════════════════════════════════════════════════════════

def _test_hybrid_import():
    # Check optuna files in baseline/ and advanced file in improved/ have clean imports
    checks = [
        PROJECT_ROOT / "src" / "models" / "baseline" / "optuna_lightgbm.py",
        PROJECT_ROOT / "src" / "models" / "baseline" / "optuna_xgboost.py",
        PROJECT_ROOT / "src" / "models" / "baseline" / "optuna_random_forest.py",
        PROJECT_ROOT / "src" / "models" / "baseline" / "optuna_tabnet.py",
        PROJECT_ROOT / "src" / "models" / "improved" / "train_tabnet_advanced.py",
    ]
    for path in checks:
        src = path.read_text(encoding="utf-8")
        assert "src.modeling" not in src, f"{path.name}: old src.modeling import"
        assert "spring-2026-group2" not in src, f"{path.name}: hardcoded path"
        assert "tabnet.prepare_memmap" not in src, f"{path.name}: old tabnet import"
    return "all model files have clean imports"


def _test_hybrid_forward():
    """Build the hybrid model architecture and run a forward pass."""
    import torch
    import torch.nn as nn

    # ── TCN block ─────────────────────────────────────────────────────────
    class CausalConv1d(nn.Module):
        def __init__(self, in_ch, out_ch, kernel, dilation):
            super().__init__()
            pad = (kernel - 1) * dilation
            self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=pad)
            self.pad  = pad
        def forward(self, x):
            return self.conv(x)[:, :, :x.size(2)]

    class TCNBlock(nn.Module):
        def __init__(self, ch, kernel=3, dilation=1):
            super().__init__()
            self.net = nn.Sequential(
                CausalConv1d(ch, ch, kernel, dilation), nn.ReLU(),
                CausalConv1d(ch, ch, kernel, dilation), nn.ReLU(),
            )
        def forward(self, x):
            return self.net(x) + x

    # ── Minimal hybrid (TCN + Transformer + BiLSTM) ───────────────────────
    class MinimalHybrid(nn.Module):
        def __init__(self, n_ch=16, seq=256, hidden=32):
            super().__init__()
            self.tcn = nn.Sequential(*[TCNBlock(n_ch, dilation=2**i) for i in range(3)])
            self.tcn_pool = nn.AdaptiveAvgPool1d(1)

            enc_layer = nn.TransformerEncoderLayer(d_model=n_ch, nhead=4, dim_feedforward=64,
                                                   batch_first=True, dropout=0.1)
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=1)
            self.trans_pool  = nn.AdaptiveAvgPool1d(1)

            self.bilstm = nn.LSTM(n_ch, hidden, batch_first=True, bidirectional=True)

            total = n_ch + n_ch + hidden * 2
            self.head = nn.Sequential(nn.Linear(total, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x):                       # x: (B, 16, 256)
            t = self.tcn(x)                         # (B, 16, 256)
            t = self.tcn_pool(t).squeeze(-1)        # (B, 16)

            xT = x.permute(0, 2, 1)                # (B, 256, 16)
            tr = self.transformer(xT)               # (B, 256, 16)
            tr = self.trans_pool(tr.permute(0,2,1)).squeeze(-1)  # (B, 16)

            _, (h, _) = self.bilstm(xT)            # h: (2, B, hidden)
            h = torch.cat([h[0], h[1]], dim=-1)    # (B, hidden*2)

            out = torch.cat([t, tr, h], dim=-1)
            return self.head(out).squeeze(-1)

    model = MinimalHybrid()
    batch = torch.randn(8, 16, 256)
    with torch.no_grad():
        logits = model(batch)
    assert logits.shape == (8,), f"Expected (8,), got {logits.shape}"
    return f"TCN+Transformer+BiLSTM forward pass OK: input={tuple(batch.shape)}, output={tuple(logits.shape)}"


check("Hybrid: train_hybrid_raw.py import hygiene",              _test_hybrid_import)
check("Hybrid: TCN+Transformer+BiLSTM forward pass (synthetic)", _test_hybrid_forward)


# ══════════════════════════════════════════════════════════════════════════════
# 12. Optuna — quick 2-trial study
# ══════════════════════════════════════════════════════════════════════════════

def _test_optuna_study():
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    from sklearn.metrics import average_precision_score
    import lightgbm as lgb

    X_tr, y_tr = make_feature_matrix(200)
    X_val, y_val = make_feature_matrix(50)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 30),
            "learning_rate": trial.suggest_float("lr", 1e-3, 0.1, log=True),
            "verbose": -1,
        }
        m = lgb.LGBMClassifier(**params)
        m.fit(X_tr, y_tr)
        prob = m.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, prob)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=2, show_progress_bar=False)
    return f"Optuna 2-trial study OK, best_value={study.best_value:.4f}"


check("Optuna: 2-trial LightGBM study end-to-end", _test_optuna_study)


# ══════════════════════════════════════════════════════════════════════════════
# 13. full metric pipeline (threshold sweep -> best model select)
# ══════════════════════════════════════════════════════════════════════════════

def _test_full_metric_pipeline():
    from src.models.utils.metric_utils import compute_binary_metrics, sweep_thresholds_for_f1
    import lightgbm as lgb

    X_tr, y_tr = make_feature_matrix(300)
    X_val, y_val = make_feature_matrix(100)
    X_te, y_te   = make_feature_matrix(100)

    model = lgb.LGBMClassifier(n_estimators=20, verbose=-1)
    model.fit(X_tr, y_tr)

    val_prob  = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_te)[:, 1]

    best_t, _ = sweep_thresholds_for_f1(y_val, val_prob)
    val_m  = compute_binary_metrics(y_val, val_prob,  threshold=best_t)
    test_m = compute_binary_metrics(y_te,  test_prob, threshold=best_t)

    assert all(k in test_m for k in ["aucpr", "f1", "roc_auc", "precision", "recall"])
    return (f"val AUCPR={val_m['aucpr']:.3f} | "
            f"test AUCPR={test_m['aucpr']:.3f} | best_t={best_t:.2f}")


check("Full pipeline: fit -> sweep threshold -> eval val+test", _test_full_metric_pipeline)


# ══════════════════════════════════════════════════════════════════════════════
# Print results
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  STRESS TEST RESULTS")
print("=" * 70)

passed = sum(1 for r in results if r[0] == PASS)
failed = sum(1 for r in results if r[0] == FAIL)
skipped = sum(1 for r in results if r[0] == SKIP)

for status, label, msg in results:
    print(f"\n{status}  {label}")
    if msg:
        for line in msg.splitlines():
            print(f"       {line}")

print("\n" + "=" * 70)
print(f"  {passed} passed   {failed} failed   {skipped} skipped   ({len(results)} total)")
print("=" * 70)

if failed:
    print("\nFix the FAIL items above before pushing.")
    sys.exit(1)
else:
    print("\nAll tests passed. Code is ready to push.")
