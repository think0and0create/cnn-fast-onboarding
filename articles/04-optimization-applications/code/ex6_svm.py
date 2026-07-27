"""code/ex6_svm.py — Ch 4 ex6: primal SVM via subgradient GD.

Primal (soft-margin) SVM:
  L(w, b) = C * mean(max(0, 1 - y_i * (w·x_i + b))) + 0.5 * ||w||^2
y is in {-1, +1}; subgradients flow only through hinge-active samples.

API contract (locked by tests/ex6_svm.py):
  - svm_subgradient_gd(X, y, *, C, lr, n_steps, seed) -> (w, b, history)
  - predict_svm(X, w, b) -> {-1, +1} Tensor, one per sample
  - run_svm_demo(*, save_dir) -> dict with accuracy, w, num_support_vectors,
    loss_history; saves fig-11-hinge-loss-curve.png + fig-12-svm-decision-boundary.png
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from code._viz import decision_boundary_2d, save_figure


# ---------------------------------------------------------------------------
# Core: training + prediction
# ---------------------------------------------------------------------------


def svm_subgradient_gd(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    C: float = 1.0,
    lr: float = 1e-2,
    n_steps: int = 3000,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """Primal SVM via subgradient descent. Returns (w, b, loss_history)."""
    torch.manual_seed(seed)
    X64 = X.to(dtype=torch.float64)
    y64 = y.to(dtype=torch.float64)
    n_samples, n_features = X64.shape

    w = torch.zeros(n_features, dtype=torch.float64)
    b = torch.tensor(0.0, dtype=torch.float64)
    loss_history: list[float] = []

    for _ in range(n_steps):
        margin = y64 * (X64 @ w + b)  # (n_samples,)
        hinge = torch.clamp(1.0 - margin, min=0.0)
        loss = C * hinge.mean() + 0.5 * (w * w).sum()

        # Subgradient: d hinge / d margin = -1 if margin < 1, else 0.
        active = (margin < 1.0).to(dtype=torch.float64)
        d_margin = -y64 * active
        grad_w = w + C * (X64.T @ d_margin) / n_samples
        grad_b = C * d_margin.sum() / n_samples

        w = w - lr * grad_w
        b = b - lr * grad_b

        loss_history.append(float(loss.item()))

    return w, b, loss_history


def predict_svm(
    X: torch.Tensor, w: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """Sign(X @ w + b) with the convention that 0 maps to +1."""
    score = X.to(dtype=torch.float64) @ w.to(dtype=torch.float64) + b.to(dtype=torch.float64)
    return torch.where(score >= 0, torch.ones_like(score), -torch.ones_like(score))


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def _make_separable_2d(
    seed: int = 11,
    n_per: int = 25,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    X_neg = torch.randn(n_per, 2, dtype=torch.float64) * 0.4 + torch.tensor(
        [-2.0, 0.0], dtype=torch.float64
    )
    X_pos = torch.randn(n_per, 2, dtype=torch.float64) * 0.4 + torch.tensor(
        [+2.0, 0.0], dtype=torch.float64
    )
    X = torch.cat([X_neg, X_pos], dim=0)
    y = torch.tensor([-1.0] * n_per + [1.0] * n_per, dtype=torch.float64)
    return X, y


def _count_support_vectors(
    X: torch.Tensor, y: torch.Tensor, w: torch.Tensor, b: torch.Tensor, tol: float = 0.10
) -> int:
    """Count samples whose hinge activation y_i*(w·x_i+b) falls within tol of 1.

    These are exactly the support vectors pinning the margin.
    """
    margin = y * (X @ w + b)
    return int((((margin >= 1.0 - tol) & (margin <= 1.0 + tol))).sum().item())


def run_svm_demo(*, save_dir: Path | None = None) -> dict[str, Any]:
    """Train primal SVM on two Gaussian clusters; render two PNGs."""
    X, y = _make_separable_2d()
    w, b, history = svm_subgradient_gd(X, y, n_steps=3000, lr=1e-2)

    preds = predict_svm(X, w, b)
    accuracy = float((preds == y).double().mean().item())
    n_sv = _count_support_vectors(X, y, w, b)

    result: dict[str, Any] = {
        "accuracy": accuracy,
        "w": w.detach(),
        "b": b.detach(),
        "num_support_vectors": n_sv,
        "loss_history": history,
    }

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

        # fig-11: hinge-loss curve (decays from C to 0 as data is separated)
        fig, ax = plt.subplots()
        ax.plot(np.arange(len(history)), history, color="#d62728")
        ax.set_title("SVM hinge loss (primal)")
        ax.set_xlabel("step")
        ax.set_ylabel("L(w,b)")
        save_figure(fig, save_dir / "fig-11-hinge-loss-curve.png")

        # fig-12: 2-D decision boundary + support vectors highlighted.
        # _viz.decision_boundary_2d maps predict_fn outputs as categorical {0,1,2,...},
        # so we wrap our {-1,+1} classifier into {0,1}.
        def _pred_fn(grid_np: np.ndarray) -> np.ndarray:
            t = torch.as_tensor(grid_np, dtype=torch.float64)
            p = predict_svm(t, w, b).cpu().numpy()
            return ((p + 1) / 2).astype(np.int64)

        X_np = X.cpu().numpy()
        y_int = ((y.cpu().numpy() + 1) / 2).astype(np.int64)
        fig = decision_boundary_2d(
            X_np,
            y_int,
            _pred_fn,
            title="SVM decision boundary (orange = support vectors)",
        )
        ax = fig.axes[0]
        margin = (y * (X @ w + b)).detach().cpu().numpy()
        sv_mask = (margin >= 0.85) & (margin <= 1.15)
        ax.scatter(
            X_np[sv_mask, 0],
            X_np[sv_mask, 1],
            s=140,
            facecolors="none",
            edgecolors="orange",
            linewidths=2.0,
            label="support vectors",
        )
        ax.legend(loc="upper right")
        save_figure(fig, save_dir / "fig-12-svm-decision-boundary.png")

    return result


if __name__ == "__main__":
    out = run_svm_demo()
    print(f"accuracy = {out['accuracy']:.4f}")
    print(f"num support vectors = {out['num_support_vectors']}")
    print(f"final loss = {out['loss_history'][-1]:.4f}")
