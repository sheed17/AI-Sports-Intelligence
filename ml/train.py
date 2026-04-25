"""Train the LSTM event classifier with MLflow experiment tracking."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

from ml.datasets.event_dataset import EventWindowDataset
from ml.datasets.synthetic_dataset import SyntheticWindowDataset
from ml.features import EVENT_CLASSES, NUM_CLASSES
from ml.models.lstm_classifier import LSTMEventClassifier


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f).get("train", {})


def train_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += len(y)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * len(y)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += len(y)
    return total_loss / total, correct / total


def main(params_path: str = "params.yaml"):
    params = load_params(params_path)

    features_dir = params.get("features_dir", "data/features")
    annotations_dir = params.get("annotations_dir", "data/annotations")
    output_dir = Path(params.get("output_dir", "ml/artifacts"))
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = params.get("batch_size", 64)
    lr = params.get("lr", 1e-3)
    weight_decay = params.get("weight_decay", 1e-4)
    epochs = params.get("epochs", 50)
    hidden_size = params.get("hidden_size", 128)
    num_layers = params.get("num_layers", 2)
    dropout = params.get("dropout", 0.3)
    patience = params.get("patience", 10)
    val_split = params.get("val_split", 0.15)
    test_split = params.get("test_split", 0.15)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Use synthetic data if available (faster iteration, no annotation required)
    synthetic_dir = Path(features_dir) / "synthetic"
    if (synthetic_dir / "windows.npy").exists():
        print(f"Loading pre-windowed synthetic dataset from {synthetic_dir}")
        dataset = SyntheticWindowDataset(synthetic_dir, augment=False)
        train_dataset = SyntheticWindowDataset(synthetic_dir, augment=True)
        data_source = "synthetic"
    else:
        dataset = EventWindowDataset(features_dir=features_dir, annotations_dir=annotations_dir, augment=False)
        train_dataset = EventWindowDataset(features_dir=features_dir, annotations_dir=annotations_dir, augment=True)
        data_source = "feature_windows"

    if len(dataset) == 0:
        print("No training data found. Run: python ml/generate_synthetic_data.py")
        return

    # Stratified split
    labels = [dataset[i][1].item() for i in range(len(dataset))]
    indices = list(range(len(dataset)))
    train_idx, temp_idx = train_test_split(
        indices, test_size=val_split + test_split, stratify=labels, random_state=42
    )
    val_labels = [labels[i] for i in temp_idx]
    val_frac = val_split / (val_split + test_split)
    val_idx, test_idx = train_test_split(temp_idx, test_size=1 - val_frac, stratify=val_labels, random_state=42)

    train_loader = DataLoader(Subset(train_dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=0)

    # Class weights for imbalance
    train_labels = [labels[i] for i in train_idx]
    class_weights = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=train_labels)
    weight_tensor = torch.FloatTensor(class_weights).to(device)

    model = LSTMEventClassifier(hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Env var takes priority over params.yaml (lets Docker override without rebuild)
    import os
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or params.get("mlflow_tracking_uri", "file:///app/mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("soccer-event-classifier")

    with mlflow.start_run():
        mlflow.log_params({
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "epochs": epochs,
            "device": str(device),
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
        })

        best_val_loss = float("inf")
        no_improve = 0
        best_model_path = output_dir / "best_model.pth"

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
            scheduler.step()

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, step=epoch)

            print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} acc={train_acc:.3f} "
                  f"| val_loss={val_loss:.4f} acc={val_acc:.3f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
                torch.save(model.state_dict(), str(best_model_path))
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_artifact(str(best_model_path))

        # Run evaluation on test set
        from ml.evaluate import evaluate_model
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False)
        metrics, cm_path = evaluate_model(model, test_loader, device, output_dir)
        metrics_payload = {
            "data_source": data_source,
            "samples": {
                "total": len(dataset),
                "train": len(train_idx),
                "val": len(val_idx),
                "test": len(test_idx),
            },
            "best_val_loss": float(best_val_loss),
            "metrics": metrics,
            "confusion_matrix": str(cm_path),
            "model_path": str(best_model_path),
            "note": (
                "Synthetic-label metrics are suitable for this portfolio MVP's model smoke/eval story; "
                "replace data/features/synthetic with SoccerNet/manual annotations for real model validity."
            ),
        }
        (output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(output_dir / "metrics.json"))
        if cm_path.exists():
            mlflow.log_artifact(str(cm_path))

        print(f"Training complete. Best val loss: {best_val_loss:.4f}")
        print(f"Model saved to: {best_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    main(args.params)
