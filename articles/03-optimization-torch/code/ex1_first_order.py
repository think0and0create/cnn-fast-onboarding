"""First-order optimization demos with an analytic and autograd gradient."""

from __future__ import annotations

from typing import TypedDict

import torch


class FirstOrderResults(TypedDict):
    """Structured results returned by the first-order demo."""

    final_x_sgd: torch.Tensor
    final_loss_sgd: float


def f_quadratic(x: torch.Tensor) -> torch.Tensor:
    """Evaluate the elementwise quadratic objective ``f(x) = x²``."""
    return x.to(dtype=torch.float64).square()


def grad_quadratic(x: torch.Tensor) -> torch.Tensor:
    """Evaluate the analytic gradient ``2x`` in double precision."""
    return 2.0 * x.to(dtype=torch.float64)


def gradient_descent(
    x0: torch.Tensor,
    lr: float,
    n_steps: int,
) -> torch.Tensor:
    """Minimize the quadratic objective with fixed-step gradient descent."""
    x = x0.detach().clone().to(dtype=torch.float64)
    for _ in range(n_steps):
        x = x - lr * grad_quadratic(x)
    return x


def autograd_quadratic_gradient(x: torch.Tensor) -> torch.Tensor:
    """Differentiate the summed quadratic objective with ``torch.autograd``."""
    differentiable_x = x.detach().clone().to(dtype=torch.float64).requires_grad_(True)
    objective = f_quadratic(differentiable_x).sum()
    gradient, = torch.autograd.grad(
        objective,
        differentiable_x,
        create_graph=False,
    )
    return gradient


def main() -> FirstOrderResults:
    """Run the first-order demo and return verifiable numerical results."""
    final_x = gradient_descent(
        torch.tensor([2.0], dtype=torch.float64),
        lr=0.1,
        n_steps=100,
    )
    return {
        "final_x_sgd": final_x,
        "final_loss_sgd": f_quadratic(final_x).sum().item(),
    }


if __name__ == "__main__":
    main()
