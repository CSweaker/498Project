"""Shared utilities for the robust encrypted traffic analysis project."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """Create a directory if it does not exist and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Set common random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_device() -> torch.device:
    """Return CUDA if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_tensor(x: np.ndarray | torch.Tensor, device: torch.device | str) -> torch.Tensor:
    """Convert numpy array or tensor to float32 tensor on the requested device."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float32)
    return torch.tensor(x, dtype=torch.float32, device=device)


def prepare_feature_bounds(
    bounds: Optional[Dict[str, Any]], device: torch.device | str
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Convert stored feature bounds to tensors."""
    if bounds is None:
        return None
    lower = torch.tensor(bounds["lower"], dtype=torch.float32, device=device)
    upper = torch.tensor(bounds["upper"], dtype=torch.float32, device=device)
    return lower, upper


def clip_tensor(
    x: torch.Tensor, bounds: Optional[Tuple[torch.Tensor, torch.Tensor]]
) -> torch.Tensor:
    """Clip a tensor feature-wise using stored lower/upper bounds."""
    if bounds is None:
        return x
    lower, upper = bounds
    return torch.max(torch.min(x, upper), lower)


def clip_numpy(x: np.ndarray, bounds: Optional[Dict[str, Any]]) -> np.ndarray:
    """Clip a numpy array feature-wise using stored lower/upper bounds."""
    if bounds is None:
        return x
    return np.clip(x, np.asarray(bounds["lower"]), np.asarray(bounds["upper"]))


def binary_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    y_score: Optional[Iterable[float]] = None,
) -> Dict[str, Any]:
    """Compute binary classification metrics robustly."""
    y_true = np.asarray(list(y_true)).astype(int)
    y_pred = np.asarray(list(y_pred)).astype(int)
    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        y_score = np.asarray(list(y_score), dtype=float)
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            metrics["roc_auc"] = None
        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
        except ValueError:
            metrics["pr_auc"] = None
    return metrics


def print_metrics(name: str, metrics: Dict[str, Any]) -> None:
    """Pretty-print a metrics dictionary."""
    print(f"\n── {name} ──────────────────────────")
    for key in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        if key in metrics and metrics[key] is not None:
            print(f"{key:18s}: {metrics[key]:.4f}")
    if "attack_success_rate" in metrics:
        print(f"{'attack_success_rate':18s}: {metrics['attack_success_rate']:.4f}")
    if "confusion_matrix" in metrics:
        print("confusion_matrix:")
        print(np.asarray(metrics["confusion_matrix"]))


def save_json(obj: Dict[str, Any], path: str | os.PathLike[str]) -> None:
    """Save a dictionary as formatted JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
