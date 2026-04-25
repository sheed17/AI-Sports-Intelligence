"""Evaluate trained model: per-class F1, confusion matrix, classification report."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from ml.features import EVENT_CLASSES


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
) -> tuple[dict[str, float], Path]:
    """Run inference on loader and compute metrics.

    Returns:
        metrics: dict of scalar metric values for MLflow
        cm_path: path to saved confusion matrix PNG
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = model(x).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    report = classification_report(
        all_labels, all_preds, target_names=EVENT_CLASSES, output_dict=True, zero_division=0
    )
    print(classification_report(all_labels, all_preds, target_names=EVENT_CLASSES, zero_division=0))

    metrics = {
        "test_accuracy": float(report["accuracy"]),
        "test_macro_f1": float(report["macro avg"]["f1-score"]),
        "test_weighted_f1": float(report["weighted avg"]["f1-score"]),
    }
    for cls in EVENT_CLASSES:
        if cls in report:
            metrics[f"test_f1_{cls}"] = float(report[cls]["f1-score"])
            metrics[f"test_prec_{cls}"] = float(report[cls]["precision"])
            metrics[f"test_rec_{cls}"] = float(report[cls]["recall"])

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(EVENT_CLASSES))))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=EVENT_CLASSES,
        yticklabels=EVENT_CLASSES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Event Classifier — Confusion Matrix")
    plt.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    fig.savefig(str(cm_path), dpi=120)
    plt.close(fig)

    return metrics, cm_path


if __name__ == "__main__":
    import argparse
    from ml.datasets.event_dataset import EventWindowDataset
    from ml.models.lstm_classifier import LSTMEventClassifier

    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default="data/features")
    parser.add_argument("--annotations-dir", default="data/annotations")
    parser.add_argument("--model-path", default="ml/artifacts/best_model.pth")
    parser.add_argument("--output-dir", default="ml/artifacts")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMEventClassifier()
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.to(device)

    dataset = EventWindowDataset(args.features_dir, args.annotations_dir)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluate_model(model, loader, device, output_dir)
