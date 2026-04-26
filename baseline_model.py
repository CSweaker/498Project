"""Stage 1: strong tabular baseline model for flow-based intrusion detection."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils import binary_metrics, ensure_dir, get_device, print_metrics, save_json, set_seed


class CNNClassifier(nn.Module):
    """Original 1D CNN kept for ablation/backward compatibility."""

    def __init__(self, input_dim: int, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.net(x)
        return self.classifier(x)


class ResidualBlock(nn.Module):
    """Residual MLP block for tabular features."""

    def __init__(self, hidden_dim: int, dropout: float = 0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabularResNet(nn.Module):
    """ResNet-style MLP baseline, better matched to tabular flow features."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.stem = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU())
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


def make_loaders(
    data_dir: str,
    batch_size: int = 512,
) -> Tuple[DataLoader, DataLoader, DataLoader, Tuple[np.ndarray, ...]]:
    """Load numpy arrays and create PyTorch loaders."""
    data_path = Path(data_dir)
    X_train = np.load(data_path / "X_train.npy").astype(np.float32)
    y_train = np.load(data_path / "y_train.npy").astype(np.int64)
    X_val = np.load(data_path / "X_val.npy").astype(np.float32)
    y_val = np.load(data_path / "y_val.npy").astype(np.int64)
    X_test = np.load(data_path / "X_test.npy").astype(np.float32)
    y_test = np.load(data_path / "y_test.npy").astype(np.int64)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
        batch_size=batch_size,
        shuffle=False,
    )
    arrays = (X_train, y_train, X_val, y_val, X_test, y_test)
    return train_loader, val_loader, test_loader, arrays


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    count = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total += loss.item() * len(x)
        count += len(x)
    return total / max(count, 1)


@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_logits, all_labels = [], []
    for x, y in loader:
        logits = model(x.to(device))
        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    preds = probs.argmax(axis=1)
    return preds, labels, probs, logits


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    preds, labels, probs, _ = predict_logits(model, loader, device)
    score = probs[:, 1] if probs.shape[1] > 1 else None
    return binary_metrics(labels, preds, score)


def load_classifier(model_path: str, device: torch.device | str = "cpu") -> nn.Module:
    """Load a classifier checkpoint saved by this file."""
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        cfg = ckpt.get("model_config", {})
        model_name = ckpt.get("model_name", "tabular_resnet")
        if model_name == "cnn":
            model = CNNClassifier(input_dim=int(cfg["input_dim"]), num_classes=int(cfg.get("num_classes", 2)))
        else:
            model = TabularResNet(
                input_dim=int(cfg["input_dim"]),
                num_classes=int(cfg.get("num_classes", 2)),
                hidden_dim=int(cfg.get("hidden_dim", 256)),
                num_blocks=int(cfg.get("num_blocks", 3)),
                dropout=float(cfg.get("dropout", 0.15)),
            )
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        raise ValueError("Unsupported checkpoint format. Re-train with baseline_model.py.")
    model.to(device)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed, deterministic=args.deterministic)
    device = get_device()
    model_dir = ensure_dir(args.model_dir)
    ensure_dir("results")

    train_loader, val_loader, test_loader, arrays = make_loaders(args.data_dir, args.batch_size)
    X_train = arrays[0]

    preprocess_path = Path(args.artifact_dir) / "preprocess.joblib"
    preprocess = joblib.load(preprocess_path) if preprocess_path.exists() else {}
    class_weights = preprocess.get("class_weights")
    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    model = TabularResNet(
        input_dim=X_train.shape[1],
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

    best_state = copy.deepcopy(model.state_dict())
    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_model(model, val_loader, device)
        val_f1 = val_metrics["f1"]
        scheduler.step(val_f1)
        print(
            f"Epoch {epoch:03d} | loss={train_loss:.4f} | "
            f"val_f1={val_f1:.4f} | val_bal_acc={val_metrics['balanced_accuracy']:.4f}"
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(best_state)
    val_metrics = evaluate_model(model, val_loader, device)
    test_metrics = evaluate_model(model, test_loader, device)
    print_metrics("Validation", val_metrics)
    print_metrics("Test clean", test_metrics)

    checkpoint = {
        "model_name": "tabular_resnet",
        "model_config": {
            "input_dim": int(X_train.shape[1]),
            "num_classes": 2,
            "hidden_dim": args.hidden_dim,
            "num_blocks": args.num_blocks,
            "dropout": args.dropout,
        },
        "model_state_dict": model.state_dict(),
        "best_epoch": best_epoch,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "seed": args.seed,
    }
    torch.save(checkpoint, model_dir / "tabular_baseline.pth")
    save_json({"validation": val_metrics, "test_clean": test_metrics}, "results/baseline_metrics.json")
    print(f"Saved best baseline to {model_dir / 'tabular_baseline.pth'}")


if __name__ == "__main__":
    main()
