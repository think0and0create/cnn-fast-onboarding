"""Closed-form and gradient-descent linear regression demo."""
from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np

from code._viz import loss_curve, save_figure


class LinearDemoResult(TypedDict):
    closed_form_w: float
    closed_form_b: float
    gd_w: float
    gd_b: float
    gd_final_loss: float
    demo_data_n: int


def _vectors(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(X, dtype=np.float64).reshape(-1)
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.shape != target.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {target.shape}")
    return x, target


def linear_regression_closed_form(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit ``y = w*x + b`` with the centered OLS formula."""
    x, target = _vectors(X, y)
    x_centered = x - x.mean()
    target_centered = target - target.mean()
    w = float(np.dot(x_centered, target_centered) / np.dot(x_centered, x_centered))
    return w, float(target.mean() - w * x.mean())


def linear_regression_gd(
    X: np.ndarray,
    y: np.ndarray,
    *,
    lr: float = 1e-2,
    n_steps: int = 1000,
    seed: int = 0,
) -> tuple[float, float, list[float]]:
    """Fit OLS by full-batch GD on standardized coordinates."""
    x, target = _vectors(X, y)
    x_mean, x_scale = float(x.mean()), float(x.std())
    y_mean, y_scale = float(target.mean()), max(float(target.std()), 1.0)
    z = (x - x_mean) / x_scale
    target_scaled = (target - y_mean) / y_scale
    rng = np.random.default_rng(seed)
    coefficient = float(rng.normal(scale=0.01))
    bias = float(rng.normal(scale=0.01))
    history: list[float] = []
    for _ in range(n_steps):
        residual = coefficient * z + bias - target_scaled
        coefficient -= lr * float(2.0 * np.mean(residual * z))
        bias -= lr * float(2.0 * residual.mean())
        updated = coefficient * z + bias - target_scaled
        history.append(float(np.mean(updated * updated)))
    w = coefficient * y_scale / x_scale
    b = y_mean + bias * y_scale - w * x_mean
    return float(w), float(b), history


def run_linear_regression_demo(*, save_dir: str | Path | None = None) -> LinearDemoResult:
    """Run the deterministic housing-like regression example."""
    rng = np.random.default_rng(0)
    x = np.linspace(80.0, 300.0, 50, dtype=np.float64)
    y = 2.5 * x + 100.0 + rng.standard_normal(50) * 5.0
    closed_w, closed_b = linear_regression_closed_form(x, y)
    gd_w, gd_b, history = linear_regression_gd(x, y, lr=1e-2, n_steps=5000, seed=0)
    if save_dir is not None:
        output = Path(save_dir)
        loss_curve({"gradient descent": history}, out_path=output / "fig-01-loss-curve.png")
        fig, ax = plt.subplots()
        ax.scatter(x, y, label="observed", s=20)
        ax.plot(x, 2.5 * x + 100.0, label="truth")
        ax.plot(x, gd_w * x + gd_b, label="fit")
        ax.set(xlabel="x", ylabel="y", title="fit vs truth")
        ax.legend()
        save_figure(fig, output / "fig-02-fit-vs-truth.png")
    return {
        "closed_form_w": closed_w,
        "closed_form_b": closed_b,
        "gd_w": gd_w,
        "gd_b": gd_b,
        "gd_final_loss": history[-1],
        "demo_data_n": len(x),
    }


if __name__ == "__main__":
    run_linear_regression_demo()
