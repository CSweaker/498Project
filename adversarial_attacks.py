"""Stage 2: bounded FGSM and PGD attacks for standardized tabular flow features."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from baseline_model import evaluate_model, load_classifier, predict_logits
from utils import binary_metrics, ensure_dir, get_device, prepare_feature_bounds, print_metrics, save_json, set_seed


def fgsm_attack(
    model: nn.Module,
    X: np.ndarray | torch.Tensor,
    y: np.ndarray | torch.Tensor,
    epsilon: float = 0.1,
    device: torch.device | str = "cpu",
    feature_bounds: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """Generate FGSM adversarial examples."""
    model.eval()
    bounds = prepare_feature_bounds(feature_bounds, device)
    x = torch.as_tensor(X, dtype=torch.float32, device=device).detach().clone().requires_grad_(True)
    y_t = torch.as_tensor(y, dtype=torch.long, device=device)
    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, y_t)
    model.zero_grad(set_to_none=True)
    loss.backward()
    x_adv = x + epsilon * x.grad.sign()
    if bounds is not None:
        lower, upper = bounds
        x_adv = torch.max(torch.min(x_adv, upper), lower)
    return x_adv.detach().cpu().numpy()


def pgd_attack(
    model: nn.Module,
    X: np.ndarray | torch.Tensor,
    y: np.ndarray | torch.Tensor,
    epsilon: float = 0.1,
    alpha: Optional[float] = None,
    num_steps: int = 20,
    random_start: bool = True,
    device: torch.device | str = "cpu",
    feature_bounds: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """Generate projected-gradient adversarial examples under an L-infinity budget."""
    model.eval()
    if alpha is None:
        alpha = max(epsilon / max(num_steps / 2, 1), 1e-6)
    bounds = prepare_feature_bounds(feature_bounds, device)

    x_clean = torch.as_tensor(X, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y, dtype=torch.long, device=device)
    if random_start:
        x_adv = x_clean + torch.empty_like(x_clean).uniform_(-epsilon, epsilon)
    else:
        x_adv = x_clean.clone()
    if bounds is not None:
        lower, upper = bounds
        x_adv = torch.max(torch.min(x_adv, upper), lower)

    for _ in range(num_steps):
        x_adv = x_adv.detach().clone().requires_grad_(True)
        loss = nn.CrossEntropyLoss()(model(x_adv), y_t)
        model.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            delta = torch.clamp(x_adv - x_clean, min=-epsilon, max=epsilon)
            x_adv = x_clean + delta
            if bounds is not None:
                lower, upper = bounds
                x_adv = torch.max(torch.min(x_adv, upper), lower)
    return x_adv.detach().cpu().numpy()


def generate_adversarial_dataset(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    attack: str = "fgsm",
    epsilon: float = 0.1,
    alpha: Optional[float] = None,
    num_steps: int = 20,
    batch_size: int = 512,
    device: torch.device | str = "cpu",
    feature_bounds: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """Generate adversarial examples for a full dataset in batches."""
    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=False,
    )
    out = []
    for x_batch, y_batch in loader:
        if attack.lower() == "fgsm":
            x_adv = fgsm_attack(model, x_batch, y_batch, epsilon, device, feature_bounds)
        elif attack.lower() == "pgd":
            x_adv = pgd_attack(model, x_batch, y_batch, epsilon, alpha, num_steps, True, device, feature_bounds)
        else:
            raise ValueError(f"Unknown attack: {attack}")
        out.append(x_adv)
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
    return np.concatenate(out, axis=0)


def evaluate_attack_success(
    model: nn.Module,
    X_clean: np.ndarray,
    X_adv: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate adversarial performance and attack success rate."""
    loader_clean = DataLoader(TensorDataset(torch.tensor(X_clean), torch.tensor(y)), batch_size=batch_size)
    loader_adv = DataLoader(TensorDataset(torch.tensor(X_adv), torch.tensor(y)), batch_size=batch_size)
    clean_pred, labels, _, _ = predict_logits(model, loader_clean, device)
    adv_pred, labels_adv, probs_adv, _ = predict_logits(model, loader_adv, device)
    metrics = binary_metrics(labels_adv, adv_pred, probs_adv[:, 1])
    initially_correct = clean_pred == labels
    became_wrong = adv_pred != labels
    denom = max(int(initially_correct.sum()), 1)
    metrics["attack_success_rate"] = float((initially_correct & became_wrong).sum() / denom)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--model-path", default="models/tabular_baseline.pth")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--attack", choices=["fgsm", "pgd", "both"], default="both")
    parser.add_argument("--epsilons", nargs="+", type=float, default=[0.05, 0.1, 0.2])
    parser.add_argument("--pgd-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    out_dir = ensure_dir(args.out_dir)
    X_test = np.load(Path(args.data_dir) / "X_test.npy").astype(np.float32)
    y_test = np.load(Path(args.data_dir) / "y_test.npy").astype(np.int64)

    preprocess = joblib.load(Path(args.artifact_dir) / "preprocess.joblib")
    feature_bounds = preprocess.get("feature_bounds")
    model = load_classifier(args.model_path, device=device)

    attacks = ["fgsm", "pgd"] if args.attack == "both" else [args.attack]
    results: Dict[str, Dict[str, float]] = {}
    for eps in args.epsilons:
        for attack in attacks:
            print(f"\nGenerating {attack.upper()} adversarial examples at epsilon={eps}")
            X_adv = generate_adversarial_dataset(
                model,
                X_test,
                y_test,
                attack=attack,
                epsilon=eps,
                num_steps=args.pgd_steps,
                batch_size=args.batch_size,
                device=device,
                feature_bounds=feature_bounds,
            )
            out_name = f"X_{attack}_eps{eps:g}.npy"
            np.save(out_dir / out_name, X_adv)
            # Also save to data/ for compatibility with simple scripts.
            np.save(Path(args.data_dir) / out_name, X_adv)
            metrics = evaluate_attack_success(model, X_test, X_adv, y_test, args.batch_size, device)
            key = f"{attack}_eps{eps:g}"
            results[key] = metrics
            print_metrics(key, metrics)

    save_json(results, out_dir / "attack_metrics.json")
    print(f"Saved adversarial samples and metrics to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
