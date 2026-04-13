#!/usr/bin/env python3
"""
Diffusion Model Pre-training for EEG Augmentation
--------------------------------------------------
Generate synthetic seizure EEG to balance imbalanced datasets.

Usage:
    python pretrain_diffusion.py --epochs 100 --generate 1000
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from modules.diffusion_eeg import EEGDiffusion


def train_diffusion(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Training diffusion model on {device}")

    # Load data
    data_path = Path(args.data_path)
    if (data_path / "X_windows.npy").exists():
        X = np.load(data_path / "X_windows.npy")
        y = np.load(data_path / "y_windows.npy")
        # Use only seizure samples for conditional generation
        X_seizure = X[y == 1]
        print(f"Loaded {len(X_seizure)} seizure samples for training")
    else:
        print("Generating synthetic seizure data...")
        np.random.seed(42)
        X_seizure = np.random.randn(500, 16, 256).astype(np.float32)
        # Add seizure-like patterns
        X_seizure += np.sin(np.linspace(0, 20 * np.pi, 256)) * 0.5

    # DataLoader
    dataset = TensorDataset(torch.from_numpy(X_seizure))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Model
    model = EEGDiffusion(
        n_channels=16,
        time_steps=256,
        num_diffusion_steps=args.diffusion_steps,
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for (x,) in loader:
            x = x.to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), output_path / "diffusion_best.pt")

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | Loss: {avg_loss:.4f}")

    print(f"\nTraining complete! Best loss: {best_loss:.4f}")

    # Generate synthetic samples
    if args.generate > 0:
        print(f"\nGenerating {args.generate} synthetic seizure samples...")
        model.eval()

        generated = []
        batch_size = min(args.generate, 32)
        n_batches = (args.generate + batch_size - 1) // batch_size

        for i in range(n_batches):
            n = min(batch_size, args.generate - i * batch_size)
            samples = model.generate(batch_size=n, device=str(device))
            generated.append(samples.cpu().numpy())

        generated = np.concatenate(generated, axis=0)
        print(f"Generated shape: {generated.shape}")

        # Save
        np.save(output_path / "synthetic_seizures.npy", generated)
        print(f"Saved to: {output_path / 'synthetic_seizures.npy'}")

        # Statistics
        print(f"\nSynthetic data statistics:")
        print(f"  Mean: {generated.mean():.4f}")
        print(f"  Std: {generated.std():.4f}")
        print(f"  Min: {generated.min():.4f}")
        print(f"  Max: {generated.max():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../../data/processed")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--diffusion_steps", type=int, default=100)
    parser.add_argument("--generate", type=int, default=1000, help="Number of samples to generate after training")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train_diffusion(args)


if __name__ == "__main__":
    main()
