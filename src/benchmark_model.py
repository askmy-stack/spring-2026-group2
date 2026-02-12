import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
import random
import re


# --- 1. THE 1D CNN FEATURE EXTRACTOR ---
class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels):
        super(CNNFeatureExtractor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, 7, stride=2, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)


# --- 2. DATA LOADING & SMART SPLITTING ---
def load_hybrid_data(data_dir):
    meta_path = os.path.join(data_dir, "metadata.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata not found at {meta_path}")

    meta = pd.read_csv(meta_path)

    # 1. Extract Patient ID
    def get_patient_id(filename):
        return re.split(r'[_-]', filename)[0]

    meta['patient_id'] = meta['filename'].apply(get_patient_id)
    patients = sorted(meta['patient_id'].unique().tolist())
    print(f"DEBUG: Found Patients: {patients}")

    # 2. SMART SPLIT: Find a Test Patient who ACTUALLY HAS SEIZURES
    test_patient = None

    for p in reversed(patients):
        p_data = meta[meta['patient_id'] == p]
        if 1 in p_data['label'].values:
            test_patient = p
            break

    if test_patient is None:
        print("WARNING: No patient has seizures loaded! Defaulting to last patient.")
        test_patient = patients[-1]

    print(
        f"SPLIT STRATEGY: Testing on {test_patient} (Found Seizures: {len(meta[(meta['patient_id'] == test_patient) & (meta['label'] == 1)])})")
    print(f"               Training on {[p for p in patients if p != test_patient]}")

    train_meta = meta[meta['patient_id'] != test_patient]
    test_meta = meta[meta['patient_id'] == test_patient]

    train_sz = len(train_meta[train_meta['label'] == 1])
    test_sz = len(test_meta[test_meta['label'] == 1])
    print(f"  -> Train Seizures: {train_sz} | Test Seizures: {test_sz}")

    if train_sz == 0:
        raise ValueError("CRITICAL: No seizures in training set! Add more patient files.")

    # 4. Balance Training Data (Downsample Normal)
    seizure = train_meta[train_meta['label'] == 1]
    normal = train_meta[train_meta['label'] == 0]

    # Use 3x Normal to Seizure ratio
    n_samples = min(len(normal), len(seizure) * 3)
    if len(normal) > 0:
        normal = normal.sample(n=n_samples, random_state=42)

    train_meta_bal = pd.concat([seizure, normal]).sample(frac=1, random_state=42)

    # Helper Loader with Channel Cropping
    def load_batch(df):
        raw_list, feat_list, labels = [], [], []
        for _, row in df.iterrows():
            r = np.load(os.path.join(data_dir, f"{row['slice_id']}_raw.npy"))
            f = np.load(os.path.join(data_dir, f"{row['slice_id']}_feat.npy"))

            if np.std(r) > 0:
                r = (r - np.mean(r)) / (np.std(r) + 1e-6)

            raw_list.append(r)
            feat_list.append(f)
            labels.append(row['label'])

        if not raw_list: return np.array([]), np.array([]), np.array([])

        # Standardize Channels (Crop to minimum)
        min_ch = min(x.shape[0] for x in raw_list)
        raw_list = [x[:min_ch, :] for x in raw_list]

        print(f"  -> Standardized to {min_ch} Channels")
        return np.array(raw_list), np.array(feat_list), np.array(labels)

    print("Loading Training Data...")
    X_raw_train, X_hand_train, y_train = load_batch(train_meta_bal)
    print("Loading Test Data...")
    X_raw_test, X_hand_test, y_test = load_batch(test_meta)

    return X_raw_train, X_hand_train, y_train, X_raw_test, X_hand_test, y_test


# --- 3. MAIN EXECUTION ---
def run_pipeline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "seizure_system/data/processed_hybrid")

    try:
        X_rt, X_ht, yt, X_rv, X_hv, yv = load_hybrid_data(data_dir)
    except Exception as e:
        print(f"Error Loading Data: {e}")
        return

    if X_rt.size == 0:
        print("Error: No training data loaded.")
        return

    n_channels = X_rt.shape[1]

    # 2. Extract Deep Features
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using Device: {device}")

    cnn = CNNFeatureExtractor(in_channels=n_channels).to(device)
    cnn.eval()

    def get_deep_feats(data):
        batch_size = 32
        feats = []
        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                batch = torch.FloatTensor(data[i:i + batch_size]).to(device)
                out = cnn(batch)
                feats.append(out.cpu().numpy())
        return np.vstack(feats)

    print("Extracting CNN Features...")
    X_deep_train = get_deep_feats(X_rt)
    X_deep_test = get_deep_feats(X_rv)

    # 3. Fuse Features
    X_train_fused = np.hstack([X_ht, X_deep_train])
    X_test_fused = np.hstack([X_hv, X_deep_test])

    # 4. Train XGBoost
    print("Training XGBoost...")
    clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        eval_metric='logloss',
        base_score=0.5
    )
    clf.fit(X_train_fused, yt)

    print("Evaluating...")
    if len(np.unique(yv)) > 1:
        preds = clf.predict(X_test_fused)
        probs = clf.predict_proba(X_test_fused)[:, 1]
        print("\n>>> HYBRID CROSS-SUBJECT RESULTS <<<")
        print(classification_report(yv, preds, target_names=['Normal', 'Seizure']))
        print(f"AUC Score: {roc_auc_score(yv, probs):.4f}")
    else:
        print("Test set contains only one class. Cannot calculate AUC.")
        preds = clf.predict(X_test_fused)
        print(f"Accuracy: {np.mean(preds == yv):.4f}")

if __name__ == "__main__":
    run_pipeline()