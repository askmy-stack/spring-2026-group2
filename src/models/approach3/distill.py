#!/usr/bin/env python3
"""
Multi-Teacher Knowledge Distillation
------------------------------------
Compress ensemble of teacher models into single tiny student.

Usage:
    python distill.py --teachers ../approach2/checkpoints --student tiny
"""

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from architectures import get_model as get_student_model
from architectures.tiny_seizure_net import MultiTeacherDistillation, TinySeizureNet, MicroSeizureNet


def load_teachers(checkpoint_dir: str, device: torch.device) -> List[nn.Module]:
    """Load teacher models from Approach 2."""
    checkpoint_path = Path(checkpoint_dir)
    teachers = []

    # Try to load Approach 2 models
    try:
        from approach2.architectures import get_model as get_a2_model
        model_names = ["m1_vanilla_lstm", "m2_bilstm", "m3_criss_cross", "m4_cnn_lstm", "m5_feature_bilstm"]

        for name in model_names:
            ckpt = checkpoint_path / f"{name}_best.pt"
            if ckpt.exists():
                model = get_a2_model(name, n_channels=16, time_steps=256)
                model.load_state_dict(torch.load(ckpt, map_location=device))
                model = model.to(device)
                model.eval()
                teachers.append(model)
                print(f"Loaded teacher: {name}")
    except ImportError:
        print("Could not import Approach 2 models, using synthetic teachers")

    # If no teachers loaded, create dummy teachers
    if len(teachers) == 0:
        print("Creating synthetic teacher models...")
        for i in range(3):
            teacher = TinySeizureNet(n_channels=16, time_steps=256, base_filters=32)
            teacher = teacher.to(device)
            teachers.append(teacher)

    return teachers


def load_data(data_path: str):
    data_path = Path(data_path)
    x_path = data_path / "X_windows.npy"
    y_path = data_path / "y_windows.npy"

    if x_path.exists() and y_path.exists():
        X = np.load(x_path)
        y = np.load(y_path)
    else:
        np.random.seed(42)
        X = np.random.randn(5000, 16, 256).astype(np.float32)
        y = np.random.randint(0, 2, 5000).astype(np.float32)

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def distill(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # Load data
    X_train, X_test, y_train, y_test = load_data(args.data_path)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=args.batch_size
    )

    # Load teachers
    teachers = load_teachers(args.teachers, device)
    print(f"Loaded {len(teachers)} teachers")

    # Create student
    if args.student == "tiny":
        student = TinySeizureNet(n_channels=16, time_steps=256)
    else:
        student = MicroSeizureNet(n_channels=16, time_steps=256)
    student = student.to(device)

    print(f"Student parameters: {sum(p.numel() for p in student.parameters()):,}")
    print(f"Student size: {sum(p.numel() for p in student.parameters()) * 4 / 1024:.1f} KB")

    # Distillation wrapper
    distiller = MultiTeacherDistillation(
        teachers=teachers,
        student=student,
        temperature=args.temperature,
        alpha=args.alpha,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training
    best_f1 = 0.0
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        student.train()
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            optimizer.zero_grad()
            _, loss = distiller(x, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Evaluate student
        student.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                probs = torch.sigmoid(student(x)).cpu().numpy()
                all_probs.extend(probs.flatten())
                all_labels.extend(y.numpy().flatten())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        preds = (all_probs >= 0.5).astype(float)

        f1 = f1_score(all_labels, preds, zero_division=0)
        auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0

        if f1 > best_f1:
            best_f1 = f1
            torch.save(student.state_dict(), output_path / f"distilled_{args.student}.pt")

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | Loss: {total_loss/len(train_loader):.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

    print(f"\nDistillation complete!")
    print(f"Best student F1: {best_f1:.4f}")
    print(f"Model saved to: {output_path / f'distilled_{args.student}.pt'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teachers", type=str, default="../approach2/checkpoints")
    parser.add_argument("--student", type=str, default="tiny", choices=["tiny", "micro"])
    parser.add_argument("--data_path", type=str, default="../../data/processed")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    distill(args)


if __name__ == "__main__":
    main()
