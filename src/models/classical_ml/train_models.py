#!/usr/bin/env python3
"""
Memory-efficient ML training - one model at a time.
Uses chunked loading, float32, and frees data between models.
"""

import os, gc, time, warnings, json, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    roc_curve, precision_recall_curve, classification_report
)
warnings.filterwarnings("ignore")

FEATURE_DIR = "/home/ubuntu/capstone_repo_dataloader/results/features_raw"
OUTPUT_DIR  = "/home/ubuntu/capstone_repo_dataloader/results/models_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

META_COLS = {"path", "start_sec", "end_sec", "label",
             "recording_path", "subject_id", "age", "sex"}

# ------------------------------------------------------------------ #
#  Memory-efficient CSV loader: reads in chunks, returns numpy arrays
# ------------------------------------------------------------------ #
def load_features_numpy(csv_path, chunksize=100000):
    """Load CSV directly into numpy float32 arrays, minimal memory."""
    import pandas as pd
    print(f"Loading {csv_path} ...")

    # First pass: get column names
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
    
    feat_cols = [c for c in header if c not in META_COLS]
    label_idx = header.index("label")
    feat_indices = [header.index(c) for c in feat_cols]
    
    # Count rows
    print("  Counting rows...")
    n_rows = 0
    with open(csv_path) as f:
        next(f)  # skip header
        for _ in f:
            n_rows += 1
    print(f"  Total rows: {n_rows:,}")
    
    # Pre-allocate arrays
    n_feats = len(feat_cols)
    X = np.zeros((n_rows, n_feats), dtype=np.float32)
    y = np.zeros(n_rows, dtype=np.int32)
    
    # Second pass: fill arrays
    print("  Loading data...")
    row_i = 0
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for line in reader:
            y[row_i] = int(float(line[label_idx]))
            for j, fi in enumerate(feat_indices):
                try:
                    val = float(line[fi])
                    if val != val or val == float('inf') or val == float('-inf'):
                        val = 0.0
                    X[row_i, j] = val
                except (ValueError, IndexError):
                    X[row_i, j] = 0.0
            row_i += 1
            if row_i % 500000 == 0:
                print(f"    ... {row_i:,} rows")
    
    print(f"  Loaded: {X.shape[0]:,} x {X.shape[1]} features")
    print(f"  Seizure: {y.sum():,} ({100*y.mean():.2f}%)")
    print(f"  Memory: X={X.nbytes/1e9:.2f} GB, y={y.nbytes/1e6:.1f} MB")
    return X, y, feat_cols

# ------------------------------------------------------------------ #
#  Plot helpers
# ------------------------------------------------------------------ #
def plot_roc(y_true, y_prob, name, path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, lw=2, label=f"AUC={auc:.4f}")
    plt.plot([0,1],[0,1],"k--",alpha=0.3)
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"ROC - {name}"); plt.legend(); plt.grid(True,alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def plot_pr(y_true, y_prob, name, path):
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(8,6))
    plt.plot(rec, prec, lw=2, label=f"AP={ap:.4f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR - {name}"); plt.legend(); plt.grid(True,alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def plot_cm(y_true, y_pred, name, path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar()
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    fontsize=14, color="white" if cm[i,j] > cm.max()/2 else "black")
    plt.xticks([0,1],["Normal","Seizure"]); plt.yticks([0,1],["Normal","Seizure"])
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title(f"CM - {name}")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def plot_feature_importance(importances, feat_names, name, path, top_n=30):
    idx = np.argsort(importances)[-top_n:]
    plt.figure(figsize=(10,8))
    plt.barh(range(len(idx)), importances[idx])
    plt.yticks(range(len(idx)), [feat_names[i] for i in idx], fontsize=8)
    plt.xlabel("Importance"); plt.title(f"Top {top_n} Features - {name}")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def evaluate(model, X, y, feat_names, name, split_name, is_xgb=False, is_lgb=False, scaler=None):
    print(f"\n--- {name} ({split_name}) ---")
    if is_xgb:
        import xgboost as xgb
        dm = xgb.DMatrix(X, feature_names=feat_names)
        y_prob = model.predict(dm)
        del dm
    elif is_lgb:
        y_prob = model.predict(X, num_iteration=model.best_iteration)
    elif scaler is not None:
        Xs = scaler.transform(X)
        y_prob = model.predict_proba(Xs)[:, 1]
        del Xs
    else:
        y_prob = model.predict_proba(X)[:, 1]

    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "AUROC": float(roc_auc_score(y, y_prob)),
        "AUPRC": float(average_precision_score(y, y_prob)),
        "F1": float(f1_score(y, y_pred)),
        "Precision": float(precision_score(y, y_pred, zero_division=0)),
        "Sensitivity": float(recall_score(y, y_pred)),
    }
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(classification_report(y, y_pred, target_names=["Normal","Seizure"]))

    prefix = f"{name}_{split_name}"
    plot_roc(y, y_prob, f"{name} ({split_name})", os.path.join(OUTPUT_DIR, f"{prefix}_roc.png"))
    plot_pr(y, y_prob, f"{name} ({split_name})", os.path.join(OUTPUT_DIR, f"{prefix}_pr.png"))
    plot_cm(y, y_pred, f"{name} ({split_name})", os.path.join(OUTPUT_DIR, f"{prefix}_cm.png"))
    return metrics

# ------------------------------------------------------------------ #
#  Main - train one model at a time
# ------------------------------------------------------------------ #
def main():
    t_start = time.time()
    all_test_metrics = {}

    # ============================================================== #
    #  1) XGBoost (GPU) - uses DMatrix, most memory efficient
    # ============================================================== #
    print("\n" + "="*60)
    print("  XGBOOST")
    print("="*60)
    import xgboost as xgb

    # Load train
    X_train, y_train, feat_names = load_features_numpy(
        os.path.join(FEATURE_DIR, "features_train.csv"))
    
    # Load val
    X_val, y_val, _ = load_features_numpy(
        os.path.join(FEATURE_DIR, "features_val.csv"))

    pos_weight = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1)
    print(f"  scale_pos_weight: {pos_weight:.2f}")

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feat_names)
    del X_train; gc.collect()
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feat_names)
    del X_val; gc.collect()

    params = {
        "objective": "binary:logistic",
        "eval_metric": ["aucpr", "auc"],
        "tree_method": "hist", "device": "cuda",
        "max_depth": 8, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "scale_pos_weight": pos_weight,
        "min_child_weight": 5, "gamma": 1,
        "reg_alpha": 0.1, "reg_lambda": 1.0, "seed": 42,
    }

    t0 = time.time()
    xgb_model = xgb.train(
        params, dtrain, num_boost_round=2000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50, verbose_eval=50
    )
    print(f"  Best iter: {xgb_model.best_iteration}, Time: {(time.time()-t0)/60:.1f}min")
    xgb_model.save_model(os.path.join(OUTPUT_DIR, "xgboost.json"))

    # Feature importance
    imp = xgb_model.get_score(importance_type="gain")
    imp_arr = np.array([imp.get(f, 0) for f in feat_names])
    plot_feature_importance(imp_arr, feat_names, "XGBoost",
        os.path.join(OUTPUT_DIR, "XGBoost_feat_imp.png"))

    # Val metrics
    y_prob_val = xgb_model.predict(dval)
    plot_roc(y_val, y_prob_val, "XGBoost (Val)", os.path.join(OUTPUT_DIR, "XGBoost_val_roc.png"))
    plot_pr(y_val, y_prob_val, "XGBoost (Val)", os.path.join(OUTPUT_DIR, "XGBoost_val_pr.png"))

    del dtrain, dval, y_train, y_val, y_prob_val; gc.collect()

    # Test
    X_test, y_test, _ = load_features_numpy(
        os.path.join(FEATURE_DIR, "features_test.csv"))
    all_test_metrics["XGBoost"] = evaluate(
        xgb_model, X_test, y_test, feat_names, "XGBoost", "test", is_xgb=True)
    del xgb_model, X_test, y_test; gc.collect()

    # ============================================================== #
    #  2) LightGBM (GPU)
    # ============================================================== #
    print("\n" + "="*60)
    print("  LIGHTGBM")
    print("="*60)
    import lightgbm as lgb

    X_train, y_train, _ = load_features_numpy(
        os.path.join(FEATURE_DIR, "features_train.csv"))
    X_val, y_val, _ = load_features_numpy(
        os.path.join(FEATURE_DIR, "features_val.csv"))

    pos_weight = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1)
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feat_names, free_raw_data=True)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=feat_names, free_raw_data=True, reference=dtrain)

    params = {
        "objective": "binary", "metric": ["average_precision", "auc"],
        "device": "gpu", "max_depth": 8, "learning_rate": 0.05,
        "num_leaves": 127, "subsample": 0.8, "colsample_bytree": 0.8,
        "scale_pos_weight": pos_weight,
        "min_child_weight": 5, "reg_alpha": 0.1, "reg_lambda": 1.0,
        "seed": 42, "verbose": -1,
    }

    t0 = time.time()
    lgb_model = lgb.train(
        params, dtrain, num_boost_round=2000,
        valid_sets=[dtrain, dval], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
    )
    print(f"  Best iter: {lgb_model.best_iteration}, Time: {(time.time()-t0)/60:.1f}min")
    lgb_model.save_model(os.path.join(OUTPUT_DIR, "lightgbm.txt"))

    imp_arr = lgb_model.feature_importance(importance_type="gain").astype(float)
    plot_feature_importance(imp_arr, feat_names, "LightGBM",
        os.path.join(OUTPUT_DIR, "LightGBM_feat_imp.png"))

    del dtrain, dval, X_train, y_train, X_val, y_val; gc.collect()

    # Test
    X_test, y_test, _ = load_features_numpy(
        os.path.join(FEATURE_DIR, "features_test.csv"))
    all_test_metrics["LightGBM"] = evaluate(
        lgb_model, X_test, y_test, feat_names, "LightGBM", "test", is_lgb=True)
    del lgb_model, X_test, y_test; gc.collect()

    # ============================================================== #
    #  3) Random Forest (sklearn) - subsample to fit in memory
    # ============================================================== #
    print("\n" + "="*60)
    print("  RANDOM FOREST")
    print("="*60)
    from sklearn.ensemble import RandomForestClassifier
    import joblib as jl

    X_train, y_train, _ = load_features_numpy(
        os.path.join(FEATURE_DIR, "features_train.csv"))

    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=20, min_samples_leaf=10,
        class_weight="balanced", n_jobs=4, random_state=42
    )
    rf.fit(X_train, y_train)
    print(f"  Time: {(time.time()-t0)/60:.1f}min")
    jl.dump(rf, os.path.join(OUTPUT_DIR, "random_forest.pkl"))

    imp_arr = rf.feature_importances_
    plot_feature_importance(imp_arr, feat_names, "RandomForest",
        os.path.join(OUTPUT_DIR, "RandomForest_feat_imp.png"))
    del X_train, y_train; gc.collect()

    X_test, y_test, _ = load_features_numpy(
        os.path.join(FEATURE_DIR, "features_test.csv"))
    all_test_metrics["RandomForest"] = evaluate(
        rf, X_test, y_test, feat_names, "RandomForest", "test")
    del rf, X_test, y_test; gc.collect()

    # ============================================================== #
    #  4) Logistic Regression
    # ============================================================== #
    print("\n" + "="*60)
    print("  LOGISTIC REGRESSION")
    print("="*60)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import joblib as jl

    X_train, y_train, _ = load_features_numpy(
        os.path.join(FEATURE_DIR, "features_train.csv"))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    jl.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))

    t0 = time.time()
    lr = LogisticRegression(
        class_weight="balanced", max_iter=1000,
        solver="saga", random_state=42, n_jobs=4
    )
    lr.fit(X_train, y_train)
    print(f"  Time: {(time.time()-t0)/60:.1f}min")
    jl.dump(lr, os.path.join(OUTPUT_DIR, "logistic_regression.pkl"))

    imp_arr = np.abs(lr.coef_[0])
    plot_feature_importance(imp_arr, feat_names, "LogisticRegression",
        os.path.join(OUTPUT_DIR, "LogisticRegression_feat_imp.png"))
    del X_train, y_train; gc.collect()

    X_test, y_test, _ = load_features_numpy(
        os.path.join(FEATURE_DIR, "features_test.csv"))
    all_test_metrics["LogisticRegression"] = evaluate(
        lr, X_test, y_test, feat_names, "LogisticRegression", "test", scaler=scaler)
    del lr, X_test, y_test, scaler; gc.collect()

    # ============================================================== #
    #  Comparison plot
    # ============================================================== #
    models = list(all_test_metrics.keys())
    metric_names = ["AUROC", "AUPRC", "F1", "Precision", "Sensitivity"]
    x = np.arange(len(metric_names))
    width = 0.8 / len(models)
    plt.figure(figsize=(14, 6))
    for i, m in enumerate(models):
        vals = [all_test_metrics[m].get(mn, 0) for mn in metric_names]
        plt.bar(x + i*width, vals, width, label=m)
    plt.xticks(x + width*(len(models)-1)/2, metric_names)
    plt.ylabel("Score"); plt.title("Model Comparison (Test)")
    plt.legend(); plt.grid(True, alpha=0.3, axis="y"); plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=150); plt.close()

    with open(os.path.join(OUTPUT_DIR, "test_metrics.json"), "w") as f:
        json.dump(all_test_metrics, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  ALL DONE! Total: {(time.time()-t_start)/60:.1f} min")
    print(f"  Results: {OUTPUT_DIR}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
