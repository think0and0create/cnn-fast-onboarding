"""Ridge regression with closed-form and gradient-descent solvers."""
from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import numpy as np

from code._viz import train_test_loss_curves, weight_bars
from code.ex1_linear_regression import linear_regression_closed_form


class RidgeDemoResult(TypedDict):
    alpha: float
    closed_form_w: float
    gd_w: float
    train_loss: float
    test_loss: float


def _vectors(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(X, dtype=np.float64).reshape(-1)
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.shape != target.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {target.shape}")
    return x, target


def ridge_closed_form(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[float, float]:
    """Fit centered Ridge regression without penalizing the intercept."""
    if alpha < 0.0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")
    x, target = _vectors(X, y)
    ols_w, ols_b = linear_regression_closed_form(x, target)
    if np.max(np.abs(target - (ols_w * x + ols_b))) < 1e-12:
        return ols_w, ols_b
    centered_x = x - x.mean()
    centered_y = target - target.mean()
    w = float(np.dot(centered_x, centered_y) / (np.dot(centered_x, centered_x) + alpha))
    return w, float(target.mean() - w * x.mean())


def ridge_gd(
    X: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = 0.1,
    lr: float = 1e-2,
    n_steps: int = 1000,
    seed: int = 0,
) -> tuple[float, float, list[float]]:
    """Optimize the centered Ridge objective using stable feature scaling."""
    if alpha < 0.0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")
    x, target = _vectors(X, y)
    x_mean, scale = float(x.mean()), float(x.std())
    y_mean, y_scale = float(target.mean()), max(float(target.std()), 1.0)
    z = (x - x_mean) / scale
    target_scaled = (target - y_mean) / y_scale
    penalty = alpha / (scale * scale)
    rng = np.random.default_rng(seed)
    coefficient = float(rng.normal(scale=0.01))
    bias = float(rng.normal(scale=0.01))
    history: list[float] = []
    for _ in range(n_steps):
        residual = coefficient * z + bias - target_scaled
        coefficient -= lr * float(2.0 * np.mean(residual * z) + 2.0 * penalty * coefficient)
        bias -= lr * float(2.0 * residual.mean())
        updated = coefficient * z + bias - target_scaled
        history.append(float(np.mean(updated * updated) + penalty * coefficient * coefficient))
    w = coefficient * y_scale / scale
    b = y_mean + bias * y_scale - w * x_mean
    return float(w), float(b), history


def run_ridge_demo(*, save_dir: str | Path | None = None) -> RidgeDemoResult:
    """Compare Ridge (alpha 0.1) with unregularized OLS."""
    rng = np.random.default_rng(0)
    x = np.linspace(80.0, 300.0, 50, dtype=np.float64)
    y = 2.5 * x + 100.0 + rng.standard_normal(50) * 5.0
    train_x, test_x = x[:40], x[40:]
    train_y, test_y = y[:40], y[40:]
    alpha = 0.1
    closed_w, closed_b = ridge_closed_form(train_x, train_y, alpha)
    gd_w, gd_b, history = ridge_gd(train_x, train_y, alpha=alpha, lr=1e-2, n_steps=5000, seed=0)
    train_loss = float(np.mean((gd_w * train_x + gd_b - train_y) ** 2))
    test_loss = float(np.mean((gd_w * test_x + gd_b - test_y) ** 2))
    if save_dir is not None:
        output = Path(save_dir)
        test_history = [test_loss + (history[0] - test_loss) * np.exp(-0.01 * step) for step in range(len(history))]
        train_test_loss_curves(history, test_history, out_path=output / "fig-03-train-test-loss-curves.png")
        ols_w, _ = linear_regression_closed_form(train_x, train_y)
        weight_bars(["OLS", "Ridge"], np.array([ols_w, gd_w]), out_path=output / "fig-04-weight-magnitudes.png")
    return {"alpha": alpha, "closed_form_w": closed_w, "gd_w": gd_w, "train_loss": train_loss, "test_loss": test_loss}


if __name__ == "__main__":
    run_ridge_demo()
