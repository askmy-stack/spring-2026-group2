import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from pytorch_tabnet.tab_model import TabNetClassifier


def parse_args():
    p = argparse.ArgumentParser(description="Tune decision threshold for a trained streaming TabNet model.")
    p.add_argument("--memmap_dir", required=True)
    p.add_argument("--model_dir", required=True, help="Directory containing best_model.pt")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--threshold_start", type=float, default=0.05)
    p.add_argument("--threshold_end", type=float, default=0.95)
    p.add_argument("--threshold_step", type=float, default=0.05)
    p.add_argument("--fixed_threshold", type=float, default=None)
    return p.parse_args()


def get_device_name(device_arg: str) -> str:
    if device_arg == "cpu":
        return "cpu"
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_meta(memmap_dir: str, split: str):
    with open(Path(memmap_dir) / f"{split}_meta.json", "r", encoding="utf-8") as f:
        return json.load(f)


class MemmapTabularDataset(Dataset):
    def __init__(self, memmap_dir: str, split: str, limit=None):
        self.meta = load_meta(memmap_dir, split)
        self.rows = self.meta["rows"] if limit is None else min(limit, self.meta["rows"])
        self.n_features = self.meta["features"]

        self.X = np.memmap(
            self.meta["x_path"],
            dtype="float32",
            mode="r",
            shape=(self.meta["rows"], self.meta["features"]),
        )
        self.y = np.memmap(
            self.meta["y_path"],
            dtype="int64",
            mode="r",
            shape=(self.meta["rows"],),
        )

    def __len__(self):
        return self.rows

    def __getitem__(self, idx):
        x = np.array(self.X[idx], dtype=np.float32, copy=True)
        y = np.int64(self.y[idx])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def make_loader(dataset, batch_size, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def softmax_np(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


def evaluate_probs(clf, loader):
    clf.network.eval()
    y_true_all = []
    y_prob_all = []

    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(clf.device).float()
            logits, _ = clf.network(Xb)
            logits = logits.detach().cpu().numpy()
            probs = softmax_np(logits)[:, 1]

            y_true_all.append(yb.numpy())
            y_prob_all.append(probs)

    y_true = np.concatenate(y_true_all)
    y_prob = np.concatenate(y_prob_all)
    return y_true, y_prob


def metrics_at_threshold(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    return {
        "threshold": float(thr),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def main():
    args = parse_args()
    device_name = get_device_name(args.device)

    ckpt_path = Path(args.model_dir) / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Could not find {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device_name)
    train_args = ckpt["args"]

    dataset = MemmapTabularDataset(args.memmap_dir, args.split, args.limit)
    loader = make_loader(dataset, args.batch_size, args.num_workers)

    clf = TabNetClassifier(
        n_d=train_args["n_d"],
        n_a=train_args["n_a"],
        n_steps=train_args["n_steps"],
        gamma=train_args["gamma"],
        lambda_sparse=train_args["lambda_sparse"],
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=train_args["lr"]),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params={"step_size": 10, "gamma": 0.9},
        mask_type="entmax",
        device_name=device_name,
        verbose=1,
        seed=train_args["seed"],
    )

    clf.batch_size = train_args["batch_size"]
    clf.virtual_batch_size = train_args["virtual_batch_size"]
    clf.input_dim = dataset.n_features
    clf.output_dim = 2
    clf.classes_ = np.array([0, 1])
    clf.target_mapper = {0: 0, 1: 1}
    clf.preds_mapper = {0: 0, 1: 1}
    clf.device_name = device_name
    clf._default_loss = torch.nn.functional.cross_entropy
    clf.loss_fn = clf._default_loss
    clf._set_network()
    clf.network.virtual_batch_size = train_args["virtual_batch_size"]
    clf.network.load_state_dict(ckpt["network_state_dict"])
    clf.network.eval()

    y_true, y_prob = evaluate_probs(clf, loader)

    summary = {
        "split": args.split,
        "rows": int(len(dataset)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }

    thresholds = np.arange(
        args.threshold_start,
        args.threshold_end + 1e-12,
        args.threshold_step,
    )

    results = [metrics_at_threshold(y_true, y_prob, thr) for thr in thresholds]

    best_f1 = max(results, key=lambda x: x["f1"])
    best_precision = max(results, key=lambda x: x["precision"])
    best_recall = max(results, key=lambda x: x["recall"])

    out_dir = Path(args.model_dir)
    with open(out_dir / f"{args.split}_threshold_tuning.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "best_f1": best_f1,
                "best_precision": best_precision,
                "best_recall": best_recall,
                "all_thresholds": results,
            },
            f,
            indent=2,
        )

    
    if args.fixed_threshold is not None:
        print(f"\nUsing fixed threshold: {args.fixed_threshold}")
        fixed_metrics = metrics_at_threshold(y_true, y_prob, args.fixed_threshold)

        print("\nFIXED THRESHOLD RESULTS:")
        print(json.dumps(fixed_metrics, indent=2))

        with open(out_dir / f"{args.split}_fixed_threshold.json", "w", encoding="utf-8") as f:
            json.dump(fixed_metrics, f, indent=2)

        return

    print(f"\nBase ranking metrics on {args.split.upper()}:")
    print(f"roc_auc: {summary['roc_auc']:.6f}")
    print(f"pr_auc:  {summary['pr_auc']:.6f}")

    print("\nBest by F1:")
    print(json.dumps(best_f1, indent=2))

    print("\nBest by Precision:")
    print(json.dumps(best_precision, indent=2))

    print("\nBest by Recall:")
    print(json.dumps(best_recall, indent=2))

    print(f"\nSaved threshold results to: {out_dir / f'{args.split}_threshold_tuning.json'}")


if __name__ == "__main__":
    main()
