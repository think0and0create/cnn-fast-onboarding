"""Matplotlib utilities for Ch 4 demos. Uses Agg backend, no display required."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # safety: headless

import matplotlib.pyplot as plt  # noqa: E402  (must come after use())
import numpy as np  # noqa: E402


def save_figure(fig: plt.Figure, path: str | Path) -> None:
    """Save figure to path as PNG with bbox_inches='tight', dpi=120. Closes the figure after saving."""
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def loss_curve(
    histories: dict[str, list[float]],
    *,
    title: str = "loss vs step",
    xlabel: str = "step",
    ylabel: str = "loss",
    out_path: str | Path | None = None,
) -> plt.Figure:
    """Plot multiple loss curves (one per label). Returns the Figure."""
    fig, ax = plt.subplots()
    for label, values in histories.items():
        steps = np.arange(len(values))
        ax.plot(steps, values, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    if out_path is not None:
        save_figure(fig, out_path)
    return fig


def train_test_loss_curves(
    train: list[float],
    test: list[float],
    *,
    title: str = "train vs test loss",
    out_path: str | Path | None = None,
) -> plt.Figure:
    """Two curves on same axes."""
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(train)), train, label="train")
    ax.plot(np.arange(len(test)), test, label="test")
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend()
    if out_path is not None:
        save_figure(fig, out_path)
    return fig


def decision_boundary_2d(
    X: np.ndarray,
    y: np.ndarray,
    predict_fn,
    *,
    title: str = "decision boundary",
    out_path: str | Path | None = None,
    h: float = 0.05,
) -> plt.Figure:
    """2D scatter of X colored by y, with decision region from predict_fn(X) -> {0,1}."""
    fig, ax = plt.subplots()
    if X.shape[1] < 2:
        raise ValueError(f"decision_boundary_2d needs 2D input, got shape {X.shape}")
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = np.asarray(predict_fn(grid)).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, levels=1)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", s=20)
    ax.set_title(title)
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    if out_path is not None:
        save_figure(fig, out_path)
    return fig


def trajectory_1d(
    t: np.ndarray,
    trajectories: dict[str, np.ndarray],
    *,
    title: str = "trajectories",
    ylabel: str = "value",
    out_path: str | Path | None = None,
) -> plt.Figure:
    """Plot 1D signals vs t. trajectories maps label -> (T+1,) or (T,) array."""
    fig, ax = plt.subplots()
    for label, values in trajectories.items():
        arr = np.asarray(values)
        ax.plot(t, arr, label=label)
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel(ylabel)
    ax.legend()
    if out_path is not None:
        save_figure(fig, out_path)
    return fig


def weight_bars(
    labels: list[str],
    values: np.ndarray,
    *,
    title: str = "weight coefficients",
    out_path: str | Path | None = None,
) -> plt.Figure:
    """Bar plot of named feature weights."""
    fig, ax = plt.subplots()
    vals = np.asarray(values).ravel()
    ax.bar(np.arange(len(vals)), vals)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(title)
    ax.set_xlabel("feature")
    ax.set_ylabel("weight")
    if out_path is not None:
        save_figure(fig, out_path)
    return fig