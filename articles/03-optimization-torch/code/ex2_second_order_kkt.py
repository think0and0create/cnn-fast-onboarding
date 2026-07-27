"""Second-order optimization and KKT-condition demos in double precision."""

from __future__ import annotations

from typing import TypedDict

import torch


class SecondOrderResults(TypedDict):
    """Structured results returned by the Newton and KKT demo."""

    newton_final_x: torch.Tensor
    kkt_residual_at_optimum: float


def newton_step_quartic(x: torch.Tensor) -> torch.Tensor:
    """Apply one stable Newton step to ``f(x) = x⁴``."""
    x64 = x.detach().clone().to(dtype=torch.float64)
    first_derivative = 4.0 * x64.pow(3)
    second_derivative = 12.0 * x64.square()
    safe_second_derivative = torch.where(
        second_derivative == 0.0,
        torch.ones_like(second_derivative),
        second_derivative,
    )
    return x64 - first_derivative / safe_second_derivative


def kkt_residual_simple_qp(
    x: torch.Tensor,
    lam: torch.Tensor,
) -> torch.Tensor:
    """Return the aggregate KKT residual for ``min x²`` subject to ``x ≥ 1``."""
    stationarity, primal_feasibility, dual_feasibility = kkt_optimality_check(
        x,
        lam,
    )
    return stationarity + primal_feasibility + dual_feasibility


def kkt_optimality_check(
    x_star: torch.Tensor,
    lam_star: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return stationarity, primal, and dual feasibility residuals."""
    x64 = x_star.to(dtype=torch.float64)
    lam64 = lam_star.to(dtype=torch.float64)
    stationarity = torch.abs(2.0 * x64 - lam64)
    primal_feasibility = torch.clamp(1.0 - x64, min=0.0)
    dual_feasibility = torch.clamp(-lam64, min=0.0)
    return stationarity, primal_feasibility, dual_feasibility


def main() -> SecondOrderResults:
    """Run the Newton and KKT demos and return verifiable results."""
    x = torch.tensor([2.0], dtype=torch.float64)
    for _ in range(20):
        x = newton_step_quartic(x)

    residual = kkt_residual_simple_qp(
        torch.tensor([1.0], dtype=torch.float64),
        torch.tensor([2.0], dtype=torch.float64),
    )
    return {
        "newton_final_x": x,
        "kkt_residual_at_optimum": residual.sum().item(),
    }


if __name__ == "__main__":
    main()
