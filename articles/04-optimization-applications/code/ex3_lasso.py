"""Lasso regression via proximal gradient descent."""
from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np

from code._viz import save_figure, weight_bars


class LassoDemoResult(TypedDict):
    alpha: float
    gd_w: np.ndarray
    num_nonzero: int


def lasso_gd(
    X: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = 0.1,
    lr: float = 1e-2,
    n_steps: int = 1000,
    seed: int = 0,
) -> tuple[float | np.ndarray, float, list[float]]:
    """Fit Lasso with an unpenalized intercept and soft-thresholded weights."""
    if alpha < 0.0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")
    features = np.asarray(X, dtype=np.float64)
    one_dimensional = features.ndim == 1
    matrix = features.reshape(-1, 1) if one_dimensional else features
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    if matrix.shape[0] != target.shape[0]:
        raise ValueError(f"shape mismatch: {matrix.shape} vs {target.shape}")
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    centered = (matrix - means) / scales
    y_mean, y_scale = float(target.mean()), max(float(target.std()), 1.0)
    target_scaled = (target - y_mean) / y_scale
    rng = np.random.default_rng(seed)
    weights = rng.normal(scale=0.01, size=matrix.shape[1]).astype(np.float64)
    bias = float(rng.normal(scale=0.01))
    history: list[float] = []
    for _ in range(n_steps):
        residual = centered @ weights + bias - target_scaled
        raw = weights - lr * (2.0 / len(target)) * (centered.T @ residual)
        weights = np.sign(raw) * np.maximum(np.abs(raw) - lr * alpha, 0.0)
        bias -= lr * float(2.0 * residual.mean())
        updated = centered @ weights + bias - target_scaled
        history.append(float(np.mean(updated * updated) + alpha * np.abs(weights).sum()))
    original_weights = weights * y_scale / scales
    intercept = y_mean + bias * y_scale - float(means @ original_weights)
    if one_dimensional:
        return float(original_weights[0]), float(intercept), history
    return original_weights, float(intercept), history


def run_lasso_demo(*, save_dir: str | Path | None = None) -> LassoDemoResult:
    """Show exact coefficient sparsity on a multi-feature synthetic problem."""
    rng = np.random.default_rng(0)
    matrix = rng.standard_normal((120, 6)).astype(np.float64)
    truth = np.array([2.0, 0.0, -1.2, 0.0, 0.35, 0.0], dtype=np.float64)
    target = matrix @ truth + rng.standard_normal(120) * 0.05
    alpha = 0.1
    fitted, _, _ = lasso_gd(matrix, target, alpha=alpha, lr=1e-2, n_steps=5000, seed=0)
    weights = np.asarray(fitted, dtype=np.float64)
    num_nonzero = int(np.count_nonzero(weights))
    if save_dir is not None:
        output = Path(save_dir)
        alphas = np.array([0.0, 0.03, 0.1, 0.2, 0.4])
        counts = []
        for value in alphas:
            candidate, _, _ = lasso_gd(matrix, target, alpha=float(value), lr=1e-2, n_steps=2000, seed=0)
            counts.append(np.count_nonzero(np.asarray(candidate)))
        fig, ax = plt.subplots()
        ax.plot(alphas, counts, marker="o")
        ax.set(xlabel="alpha", ylabel="non-zero weights", title="alpha vs sparsity")
        save_figure(fig, output / "fig-05-alpha-vs-sparsity.png")
        weight_bars([f"x{i}" for i in range(weights.size)], weights, out_path=output / "fig-06-weight-coefficients.png")
    return {"alpha": alpha, "gd_w": weights, "num_nonzero": num_nonzero}


if __name__ == "__main__":
    run_lasso_demo()
