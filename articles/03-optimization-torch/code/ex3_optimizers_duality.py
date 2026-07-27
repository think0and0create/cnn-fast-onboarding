"""PyTorch optimizer, convex duality, and hinge-loss demonstrations."""

from __future__ import annotations

from enum import StrEnum
from typing import TypedDict, assert_never

import torch


class OptimizerKind(StrEnum):
    """Optimizer variants accepted by the quadratic training demo."""

    SGD = "SGD"
    ADAM = "Adam"


class UnknownOptimizerError(ValueError):
    """Raised when an optimizer name is not supported by the demo."""

    optimizer_name: str

    def __init__(self, optimizer_name: str) -> None:
        self.optimizer_name = optimizer_name
        super().__init__(f"unknown optimizer: {optimizer_name}")


class OptimizerDualityResults(TypedDict):
    """Structured results returned by the optimizer and duality demo."""

    sgd_final_x: torch.Tensor
    sgd_loss: float
    adam_final_x: torch.Tensor
    adam_loss: float
    strong_duality_gap: float
    svm_weights: torch.Tensor


def train_with_optimizer(
    opt_name: str,
    lr: float,
    n_steps: int,
) -> tuple[torch.Tensor, float]:
    """Train ``f(x) = x²`` from ``x = 2`` with SGD or Adam."""
    x = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    try:
        optimizer_kind = OptimizerKind(opt_name)
    except ValueError as error:
        raise UnknownOptimizerError(opt_name) from error

    match optimizer_kind:
        case OptimizerKind.SGD:
            optimizer = torch.optim.SGD([x], lr=lr)
        case OptimizerKind.ADAM:
            optimizer = torch.optim.Adam([x], lr=lr, betas=(0.5, 0.9))
        case unreachable:
            assert_never(unreachable)

    for _ in range(n_steps):
        optimizer.zero_grad()
        loss = x.square().sum()
        loss.backward()
        optimizer.step()

    final_x = x.detach()
    final_loss = final_x.square().sum().item()
    return final_x, final_loss


def duality_gap_minimal_qp(lam: torch.Tensor) -> tuple[float, float]:
    """Return primal and dual objectives for ``min x²`` subject to ``x ≥ 1``."""
    lam_value = float(lam.detach().to(dtype=torch.float64).item())
    primal_objective = 1.0
    dual_objective = lam_value - lam_value**2 / 4.0
    return primal_objective, dual_objective


def train_svm_hinge_demo() -> torch.Tensor:
    """Train a linear hinge-loss classifier on a separable two-feature toy set."""
    features = torch.tensor(
        [
            [-2.0, -1.0],
            [-1.0, -1.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ],
        dtype=torch.float64,
    )
    labels = torch.tensor([-1.0, -1.0, 1.0, 1.0], dtype=torch.float64)
    weights = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([weights, bias], lr=0.1)

    for _ in range(50):
        optimizer.zero_grad()
        margins = labels * (features @ weights + bias)
        hinge_loss = torch.clamp(1.0 - margins, min=0.0).mean()
        regularization = 1e-3 * weights.square().sum()
        (hinge_loss + regularization).backward()
        optimizer.step()

    return weights.detach()


def main() -> OptimizerDualityResults:
    """Run optimizer, duality, and hinge-loss demos and return their results."""
    sgd_final_x, sgd_loss = train_with_optimizer("SGD", lr=0.1, n_steps=50)
    adam_final_x, adam_loss = train_with_optimizer("Adam", lr=0.1, n_steps=50)
    primal_objective, dual_objective = duality_gap_minimal_qp(
        torch.tensor([2.0], dtype=torch.float64),
    )
    return {
        "sgd_final_x": sgd_final_x,
        "sgd_loss": sgd_loss,
        "adam_final_x": adam_final_x,
        "adam_loss": adam_loss,
        "strong_duality_gap": abs(primal_objective - dual_objective),
        "svm_weights": train_svm_hinge_demo(),
    }


if __name__ == "__main__":
    main()
