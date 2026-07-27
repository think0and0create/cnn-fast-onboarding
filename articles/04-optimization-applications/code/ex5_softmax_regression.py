"""code/ex5_softmax_regression.py — Ch 4 ex5: multi-class logistic regression.

Softmax regression = log-softmax + NLL loss + GD on (W, b).

API contract (locked by tests/ex5_softmax_regression.py):
  - softmax_regression_gd(X, y, n_classes, *, lr, n_steps, seed, l2) -> (W, b, history)
  - predict_class(X, W, b) -> long Tensor of class indices
  - predict_proba(X, W, b) -> softmax probabilities, rows sum to 1
  - run_softmax_regression_demo(*, save_dir) -> dict with accuracy, W, loss_history
    and saves fig-09-loss-curve.png + fig-10-confusion-matrix.png
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from code._viz import save_figure


# ---------------------------------------------------------------------------
# Core: training, prediction, probabilities
# ---------------------------------------------------------------------------


def softmax_regression_gd(
    X: torch.Tensor,
    y: torch.Tensor,
    n_classes: int,
    *,
    lr: float = 5e-1,
    n_steps: int = 2000,
    seed: int = 0,
    l2: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """Log-softmax + NLL loss, full-batch GD on (W, b). Tensors stay float64."""
    torch.manual_seed(seed)
    X64 = X.to(dtype=torch.float64)
    y64 = y.to(dtype=torch.long)
    n_samples, n_features = X64.shape

    # Xavier-ish init (very small for stability on separable data).
    W = (
        torch.randn(n_features, n_classes, dtype=torch.float64) * 0.01
    )
    b = torch.zeros(n_classes, dtype=torch.float64)

    loss_history: list[float] = []
    idx = torch.arange(n_samples, dtype=torch.long)

    for _ in range(n_steps):
        logits = X64 @ W + b  # (n_samples, n_classes)
        log_probs = F.log_softmax(logits, dim=1)
        nll = -log_probs[idx, y64].mean()
        reg = 0.5 * l2 * (W * W).sum()
        loss = nll + reg

        # Manual gradient: dL/d(logits) = softmax(logits) - one_hot(y)
        probs = log_probs.exp()
        d_logits = probs
        d_logits[idx, y64] -= 1.0
        d_logits /= n_samples
        grad_W = X64.T @ d_logits + l2 * W
        grad_b = d_logits.sum(dim=0)

        W = W - lr * grad_W
        b = b - lr * grad_b

        loss_history.append(float(loss.item()))

    return W, b, loss_history


def predict_class(X: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Argmax over class logits. dtype=long. With W=0, b=0 -> class 0 (first wins)."""
    logits = X.to(dtype=torch.float64) @ W.to(dtype=torch.float64) + b.to(dtype=torch.float64)
    return logits.argmax(dim=1).to(dtype=torch.long)


def predict_proba(X: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row-stochastic softmax probabilities."""
    logits = X.to(dtype=torch.float64) @ W.to(dtype=torch.float64) + b.to(dtype=torch.float64)
    return F.softmax(logits, dim=1)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def _make_three_class_data(
    seed: int = 42,
    n_per: int = 30,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same fixture as the test: 3 well-separated Gaussian clusters."""
    torch.manual_seed(seed)
    centers = torch.tensor(
        [[-2.0, -2.0], [0.0, 2.0], [2.0, -2.0]], dtype=torch.float64
    )
    Xs, ys = [], []
    for c in range(3):
        pts = torch.randn(n_per, 2, dtype=torch.float64) * 0.4 + centers[c]
        Xs.append(pts)
        ys.append(torch.full((n_per,), c, dtype=torch.long))
    X = torch.cat(Xs, dim=0).to(dtype=torch.float64)
    y = torch.cat(ys, dim=0)
    return X, y


def _confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> np.ndarray:
    """n_classes x n_classes count matrix: rows = true, cols = predicted."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm[t, p] += 1
    return cm


def run_softmax_regression_demo(
    *, save_dir: Path | None = None
) -> dict[str, Any]:
    """Train softmax regression on 3-class Gaussian data; render two PNGs."""
    X, y = _make_three_class_data()
    W, b, history = softmax_regression_gd(X, y, n_classes=3, n_steps=2000)

    preds = predict_class(X, W, b)
    accuracy = float((preds == y).double().mean().item())

    result: dict[str, Any] = {
        "accuracy": accuracy,
        "W": W.detach(),
        "b": b.detach(),
        "loss_history": history,
        "n_classes": 3,
    }

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        # fig-09: loss curve
        fig, ax = plt.subplots()
        ax.plot(np.arange(len(history)), history, color="#1f77b4")
        ax.set_title("softmax regression — NLL loss")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        save_figure(fig, save_dir / "fig-09-loss-curve.png")

        # fig-10: 3x3 confusion matrix heatmap
        cm = _confusion_matrix(y.cpu().numpy(), preds.cpu().numpy(), n_classes=3)
        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title("confusion matrix (rows=true, cols=pred)")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xlabel("predicted class")
        ax.set_ylabel("true class")
        for i in range(3):
            for j in range(3):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)
        fig.colorbar(im, ax=ax)
        save_figure(fig, save_dir / "fig-10-confusion-matrix.png")

    return result


if __name__ == "__main__":
    out = run_softmax_regression_demo()
    print(f"accuracy = {out['accuracy']:.4f}")
    print(f"final loss = {out['loss_history'][-1]:.4f}")
