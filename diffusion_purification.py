"""Stage 3: score-based diffusion purification for tabular flow features."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from adversarial_attacks import pgd_attack
from baseline_model import load_classifier, predict_logits
from utils import binary_metrics, ensure_dir, get_device, prepare_feature_bounds, print_metrics, save_json, set_seed


class ScoreNet(nn.Module):
    """Noise-conditional score network for tabular data."""

    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sigma_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        self.net = nn.Sequential(
            nn.Linear(input_dim + 64, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, input_dim),
        )

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma = sigma.view(-1, 1).to(x.dtype)
        emb = self.sigma_embed(torch.log(sigma + 1e-8))
        return self.net(torch.cat([x, emb], dim=1))


def get_sigma_schedule(T: int = 32, sigma_min: float = 0.01, sigma_max: float = 0.5) -> torch.Tensor:
    """Geometric noise schedule from low to high noise."""
    return torch.exp(torch.linspace(np.log(sigma_min), np.log(sigma_max), T))


def train_score_model(
    X_train: np.ndarray,
    input_dim: int,
    epochs: int = 30,
    batch_size: int = 512,
    lr: float = 1e-3,
    hidden_dim: int = 512,
    sigma_min: float = 0.01,
    sigma_max: float = 0.5,
    num_sigmas: int = 32,
    device: torch.device | str = "cpu",
) -> tuple[ScoreNet, torch.Tensor]:
    """Train by denoising score matching on the training distribution."""
    model = ScoreNet(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sigmas = get_sigma_schedule(num_sigmas, sigma_min, sigma_max).to(device)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    model.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        count = 0
        for (x,) in loader:
            x = x.to(device)
            idx = torch.randint(0, len(sigmas), (len(x),), device=device)
            sigma = sigmas[idx]
            noise = torch.randn_like(x)
            x_noisy = x + sigma[:, None] * noise
            target_score = -noise / sigma[:, None]
            pred_score = model(x_noisy, sigma)
            # DSM weighting keeps large-sigma and small-sigma terms balanced.
            loss = ((pred_score - target_score) ** 2 * sigma[:, None] ** 2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total += loss.item() * len(x)
            count += len(x)
        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch:03d} | score_loss={total / max(count, 1):.6f}")
    return model, sigmas.detach().cpu()


@torch.no_grad()
def purify(
    score_model: ScoreNet,
    X_adv: np.ndarray | torch.Tensor,
    sigmas: Optional[torch.Tensor] = None,
    start_fraction: float = 0.55,
    n_steps_each: int = 2,
    step_lr: float = 0.02,
    device: torch.device | str = "cpu",
    feature_bounds: Optional[Dict[str, np.ndarray]] = None,
    add_initial_noise: bool = True,
) -> np.ndarray:
    """Annealed Langevin purification aligned with the trained sigma schedule.

    The schedule is trained from low to high sigma. At inference we start from a
    medium noise level, optionally add that much noise, and then denoise toward
    lower sigma levels.
    """
    score_model.eval()
    x = torch.as_tensor(X_adv, dtype=torch.float32, device=device).clone()
    if sigmas is None:
        sigmas = get_sigma_schedule().to(device)
    else:
        sigmas = sigmas.to(device=device, dtype=torch.float32)
    bounds = prepare_feature_bounds(feature_bounds, device)

    start_idx = int(np.clip(round((len(sigmas) - 1) * start_fraction), 1, len(sigmas) - 1))
    selected = sigmas[: start_idx + 1].flip(0)  # medium -> low

    if add_initial_noise:
        x = x + selected[0] * torch.randn_like(x)
        if bounds is not None:
            lower, upper = bounds
            x = torch.max(torch.min(x, upper), lower)

    sigma_min = sigmas[0]
    for sigma in selected:
        sigma_batch = sigma.expand(x.shape[0])
        step_size = step_lr * (sigma / sigma_min) ** 2
        # Cap the step to avoid exploding updates on tabular data.
        step_size = torch.clamp(step_size, max=0.1)
        for _ in range(n_steps_each):
            score = score_model(x, sigma_batch)
            noise = torch.randn_like(x)
            x = x + 0.5 * step_size * score + torch.sqrt(step_size) * noise
            if bounds is not None:
                lower, upper = bounds
                x = torch.max(torch.min(x, upper), lower)
    return x.detach().cpu().numpy()


def load_score_model(model_path: str, device: torch.device | str = "cpu") -> tuple[ScoreNet, torch.Tensor, Dict]:
    """Load a score model checkpoint."""
    ckpt = torch.load(model_path, map_location=device)
    input_dim = int(ckpt["input_dim"])
    hidden_dim = int(ckpt.get("hidden_dim", 512))
    model = ScoreNet(input_dim=input_dim, hidden_dim=hidden_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    sigmas = torch.tensor(ckpt["sigma_schedule"], dtype=torch.float32, device=device)
    return model, sigmas, ckpt


def evaluate_numpy(model: nn.Module, X: np.ndarray, y: np.ndarray, batch_size: int, device: torch.device):
    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=False,
    )
    preds, labels, probs, _ = predict_logits(model, loader, device)
    return binary_metrics(labels, preds, probs[:, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--baseline-path", default="models/tabular_baseline.pth")
    parser.add_argument("--adv-dir", default="results")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--attack-eps", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    model_dir = ensure_dir(args.model_dir)
    ensure_dir("results")

    X_train = np.load(Path(args.data_dir) / "X_train.npy").astype(np.float32)
    X_test = np.load(Path(args.data_dir) / "X_test.npy").astype(np.float32)
    y_test = np.load(Path(args.data_dir) / "y_test.npy").astype(np.int64)
    preprocess = joblib.load(Path(args.artifact_dir) / "preprocess.joblib")
    feature_bounds = preprocess.get("feature_bounds")

    score_model, sigmas = train_score_model(
        X_train,
        input_dim=X_train.shape[1],
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
    )
    score_path = model_dir / "score_net.pth"
    torch.save(
        {
            "input_dim": int(X_train.shape[1]),
            "hidden_dim": score_model.hidden_dim,
            "model_state_dict": score_model.state_dict(),
            "sigma_schedule": sigmas.numpy().tolist(),
            "seed": args.seed,
        },
        score_path,
    )
    print(f"Saved score model to {score_path}")

    classifier = load_classifier(args.baseline_path, device=device)
    results = {"clean": evaluate_numpy(classifier, X_test, y_test, args.batch_size, device)}
    print_metrics("clean", results["clean"])

    for attack in ["fgsm", "pgd"]:
        adv_path = Path(args.adv_dir) / f"X_{attack}_eps{args.attack_eps:g}.npy"
        if not adv_path.exists():
            print(f"Missing {adv_path}; generating PGD-like fallback for {attack} is skipped unless file exists.")
            continue
        X_adv = np.load(adv_path).astype(np.float32)
        results[f"{attack}_attacked"] = evaluate_numpy(classifier, X_adv, y_test, args.batch_size, device)
        print_metrics(f"{attack}_attacked", results[f"{attack}_attacked"])
        X_pur = purify(
            score_model,
            X_adv,
            sigmas=sigmas,
            device=device,
            feature_bounds=feature_bounds,
        )
        np.save(Path(args.adv_dir) / f"X_{attack}_eps{args.attack_eps:g}_purified.npy", X_pur)
        results[f"{attack}_purified"] = evaluate_numpy(classifier, X_pur, y_test, args.batch_size, device)
        print_metrics(f"{attack}_purified", results[f"{attack}_purified"])

    save_json(results, "results/purification_metrics.json")
    print("Saved purification metrics to results/purification_metrics.json")


if __name__ == "__main__":
    main()
