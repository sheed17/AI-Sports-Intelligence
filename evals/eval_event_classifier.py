"""Evaluation script: load model artifact, run on test split, print metrics + confusion matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from ml.datasets.event_dataset import EventWindowDataset
from ml.datasets.synthetic_dataset import SyntheticWindowDataset
from ml.models.lstm_classifier import LSTMEventClassifier
from ml.evaluate import evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Evaluate event classifier on test split")
    parser.add_argument("--features-dir", default="data/features")
    parser.add_argument("--annotations-dir", default="data/annotations")
    parser.add_argument("--synthetic-dir", default=None)
    parser.add_argument("--model-path", default="ml/artifacts/best_model.pth")
    parser.add_argument("--output-dir", default="ml/artifacts")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}")

    synthetic_dir = Path(args.synthetic_dir) if args.synthetic_dir else Path(args.features_dir) / "synthetic"
    if (synthetic_dir / "windows.npy").exists() and (synthetic_dir / "labels.npy").exists():
        print(f"Using pre-windowed synthetic dataset: {synthetic_dir}")
        dataset = SyntheticWindowDataset(synthetic_dir, augment=False)
        data_source = "synthetic"
    else:
        print(f"Using annotated/pseudo-labeled feature windows from: {args.features_dir}")
        dataset = EventWindowDataset(args.features_dir, args.annotations_dir)
        data_source = "feature_windows"

    if len(dataset) == 0:
        print("No evaluation data found. Generate synthetic data or add annotations first.")
        return

    labels = [dataset[i][1].item() for i in range(len(dataset))]
    indices = list(range(len(dataset)))
    _, test_idx = train_test_split(indices, test_size=0.15, stratify=labels, random_state=42)

    from torch.utils.data import Subset
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False)

    model = LSTMEventClassifier()
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"No model artifact at {model_path}. Train first with: python ml/train.py")
        return

    model.load_state_dict(torch.load(str(model_path), map_location=device, weights_only=True))
    model.to(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics, cm_path = evaluate_model(model, test_loader, device, output_dir)
    metrics = {**metrics, "eval_samples": float(len(test_idx))}

    metrics_path = output_dir / "metrics.json"
    metrics_payload = {
        "data_source": data_source,
        "model_path": str(model_path),
        "samples": {
            "total": len(dataset),
            "test": len(test_idx),
        },
        "metrics": metrics,
        "confusion_matrix": str(cm_path),
        "note": (
            "Synthetic-label metrics are suitable for this portfolio MVP's model smoke/eval story; "
            "replace data/features/synthetic with SoccerNet/manual annotations for real model validity."
        ),
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))

    print("\n=== Test Metrics ===")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")
    print(f"\nConfusion matrix saved to: {cm_path}")
    print(f"Metrics JSON saved to: {metrics_path}")


if __name__ == "__main__":
    main()
