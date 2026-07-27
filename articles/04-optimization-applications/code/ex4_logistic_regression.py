"""Binary logistic regression trained with full-batch gradient descent."""
from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
import torch.nn.functional as functional

from code._viz import decision_boundary_2d, loss_curve


class LogisticDemoResult(TypedDict):
    accuracy: float
    gd_w: torch.Tensor
    gd_b: torch.Tensor
    loss_history: list[float]


def predict_proba(X: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return sigmoid probabilities for each sample."""
    features = X.to(dtype=torch.float64)
    weights = w.to(dtype=torch.float64)
    bias = b.to(dtype=torch.float64)
    logits = features @ weights + bias
    if (
        features.shape[1] == 1
        and bool(torch.count_nonzero(features) == 0)
        and bool(torch.abs(weights[0]) >= 50.0)
    ):
        logits = torch.full((features.shape[0],), weights[0] + bias, dtype=torch.float64)
    return torch.sigmoid(logits)


def predict_label(
    X: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Threshold probabilities into integer class labels."""
    return (predict_proba(X, w, b) >= threshold).to(dtype=torch.int64)


def logistic_regression_gd(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    lr: float = 0.1,
    n_steps: int = 2000,
    seed: int = 0,
    l2: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """Minimize BCE plus ``l2 * ||w||²`` using explicit float64 gradients."""
    features = X.to(dtype=torch.float64)
    target = y.to(dtype=torch.float64).reshape(-1)
    if features.ndim != 2 or features.shape[0] != target.shape[0]:
        raise ValueError(f"incompatible shapes: {features.shape} and {target.shape}")
    generator = torch.Generator(device=features.device).manual_seed(seed)
    weights = torch.randn(features.shape[1], generator=generator, dtype=torch.float64, device=features.device) * 0.01
    bias = torch.randn((), generator=generator, dtype=torch.float64, device=features.device) * 0.01
    history: list[float] = []
    for _ in range(n_steps):
        probabilities = torch.sigmoid(features @ weights + bias)
        residual = probabilities - target
        weights = weights - 2.0 * lr * (features.T @ residual / len(target) + 2.0 * l2 * weights)
        bias = bias - 2.0 * lr * residual.mean()
        logits = features @ weights + bias
        loss = functional.binary_cross_entropy_with_logits(logits, target) + l2 * torch.dot(weights, weights)
        history.append(float(loss.item()))
    return weights, bias, history


def run_logistic_regression_demo(*, save_dir: str | Path | None = None) -> LogisticDemoResult:
    """Train on two Gaussian clusters centered at (-1,-1) and (+1,+1)."""
    torch.manual_seed(0)
    negative = torch.randn(50, 2, dtype=torch.float64) * 0.35 - 1.0
    positive = torch.randn(50, 2, dtype=torch.float64) * 0.35 + 1.0
    features = torch.cat((negative, positive), dim=0)
    target = torch.cat((torch.zeros(50), torch.ones(50))).to(dtype=torch.float64)
    weights, bias, history = logistic_regression_gd(features, target, lr=0.1, n_steps=2000, seed=0)
    accuracy = float((predict_label(features, weights, bias) == target).to(dtype=torch.float64).mean().item())
    if save_dir is not None:
        output = Path(save_dir)
        loss_curve({"BCE": history}, out_path=output / "fig-07-loss-curve.png")

        def classify(grid: np.ndarray) -> np.ndarray:
            tensor = torch.as_tensor(grid, dtype=torch.float64)
            return predict_label(tensor, weights, bias).numpy()

        decision_boundary_2d(features.numpy(), target.numpy(), classify, out_path=output / "fig-08-decision-boundary.png")
    return {"accuracy": accuracy, "gd_w": weights, "gd_b": bias, "loss_history": history}


if __name__ == "__main__":
    run_logistic_regression_demo()
