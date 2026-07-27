"""code/ex7_longitudinal_planning.py — Ch 4 ex7: three trajectory parameterizations.

Same 1-D longitudinal planning problem solved three ways:
  1. state-points   : decision vars (s[1..T], v[1..T]); kinematic penalty
  2. control-points : decision vars a[0..T-1]; forward-integrate kinematics hard
  3. polynomial     : decision vars = Bernstein coefficients of a(t); sample then
                      integrate
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from code._viz import save_figure, trajectory_1d


@dataclass
class PlanningResult:
    times: np.ndarray  # (T+1,)
    s: np.ndarray      # (T+1,)
    v: np.ndarray      # (T+1,)
    a: np.ndarray      # (T,)
    losses: list[float]


def _forward_integrate(a: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Euler with constant acceleration over each interval; returns (s, v) shape (T+1,)."""
    T = a.shape[0]
    s_list = [a.new_zeros(())]
    v_list = [a.new_zeros(())]
    for k in range(int(T)):
        sk, vk, ak = s_list[-1], v_list[-1], a[k]
        s_list.append(sk + vk * dt + 0.5 * ak * dt * dt)
        v_list.append(vk + ak * dt)
    return torch.stack(s_list), torch.stack(v_list)


def _bernstein_basis(degree: int, ts: torch.Tensor) -> torch.Tensor:
    """(len(ts), degree+1) Bernstein basis matrix."""
    n, j = degree, torch.arange(degree + 1, dtype=ts.dtype).unsqueeze(0)
    binom = torch.tensor([comb(n, k) for k in range(n + 1)], dtype=ts.dtype).unsqueeze(0)
    return binom * ts.unsqueeze(-1) ** j * (1.0 - ts).unsqueeze(-1) ** (n - j)


def _wp_loss(
    v_seq: torch.Tensor, wp: Sequence[tuple[int, float]], weight: float
) -> torch.Tensor:
    parts = [(v_seq[t] - val) ** 2 for t, val in wp if 0 <= t < v_seq.shape[0]]
    return weight * torch.stack(parts).sum() if parts else v_seq.new_zeros(())


def plan_via_state_points(
    s_target: float, v_target: float, T: int = 20, dt: float = 0.5, *,
    waypoint_times: Sequence[tuple[int, float]] | None = None,
    seed: int = 0, n_steps: int = 500, lr: float = 0.05,
) -> PlanningResult:
    """Optimize s[1..T] and v[1..T] with a soft kinematic penalty."""
    torch.manual_seed(seed)
    T, dt, n_steps = int(T), float(dt), int(n_steps)
    s_dec = torch.zeros(T, dtype=torch.float64, requires_grad=True)
    v_dec = torch.zeros(T, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([s_dec, v_dec], lr=lr)
    wp = [(int(t), float(v)) for t, v in (waypoint_times or [])]
    LK, LS, LW = 0.1, 0.05, 100.0
    zero = torch.zeros(1, dtype=torch.float64)
    losses: list[float] = []

    for _ in range(n_steps):
        opt.zero_grad()
        s_full = torch.cat([zero, s_dec])
        v_full = torch.cat([zero, v_dec])
        loss = (s_dec[-1] - s_target) ** 2 + (v_dec[-1] - v_target) ** 2 + _wp_loss(v_full, wp, LW)
        loss = loss + LK * ((s_full[1:] - s_full[:-1] - v_full[:-1] * dt) ** 2).sum()
        loss = loss + LS * ((v_full[1:] - v_full[:-1]) ** 2).sum()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    with torch.no_grad():
        s_full = torch.cat([zero, s_dec.detach()]).cpu().numpy()
        v_full = torch.cat([zero, v_dec.detach()]).cpu().numpy()
    return PlanningResult(
        times=np.arange(T + 1) * dt, s=s_full, v=v_full,
        a=np.diff(v_full) / dt, losses=losses,
    )


def plan_via_control_points(
    s_target: float, v_target: float, T: int = 20, dt: float = 0.5, *,
    waypoint_times: Sequence[tuple[int, float]] | None = None,
    seed: int = 0, n_steps: int = 500, lr: float = 0.05,
) -> PlanningResult:
    """Optimize a[0..T-1]; kinematics integrated hard inside the loss."""
    torch.manual_seed(seed)
    T, dt, n_steps = int(T), float(dt), int(n_steps)
    a = torch.zeros(T, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([a], lr=lr)
    wp = [(int(t), float(v)) for t, v in (waypoint_times or [])]
    LA, LW = 1e-3, 100.0
    losses: list[float] = []

    for _ in range(n_steps):
        opt.zero_grad()
        s_seq, v_seq = _forward_integrate(a, dt)
        loss = (s_seq[-1] - s_target) ** 2 + (v_seq[-1] - v_target) ** 2 + _wp_loss(v_seq, wp, LW) + LA * (a * a).sum()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    with torch.no_grad():
        s_arr, v_arr = _forward_integrate(a.detach(), dt)
        s_arr, v_arr = s_arr.cpu().numpy(), v_arr.cpu().numpy()
    return PlanningResult(
        times=np.arange(T + 1) * dt, s=s_arr, v=v_arr,
        a=a.detach().cpu().numpy(), losses=losses,
    )


def plan_via_polynomial(
    s_target: float, v_target: float, degree: int = 3, T: int = 20, dt: float = 0.5, *,
    waypoint_times: Sequence[tuple[int, float]] | None = None,
    seed: int = 0, n_steps: int = 500, lr: float = 0.05,
) -> PlanningResult:
    """Bernstein polynomial coefficients for a(t); sample & integrate."""
    torch.manual_seed(seed)
    degree, T, dt, n_steps = int(degree), int(T), float(dt), int(n_steps)
    coeffs = torch.zeros(degree + 1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([coeffs], lr=lr)
    wp = [(int(t), float(v)) for t, v in (waypoint_times or [])]
    LA, LW = 1e-3, 100.0
    sample_ts = torch.arange(T, dtype=torch.float64) * dt / (T * dt)
    losses: list[float] = []

    for _ in range(n_steps):
        opt.zero_grad()
        s_seq, v_seq = _forward_integrate(_bernstein_basis(degree, sample_ts) @ coeffs, dt)
        loss = (s_seq[-1] - s_target) ** 2 + (v_seq[-1] - v_target) ** 2 + _wp_loss(v_seq, wp, LW) + LA * (coeffs * coeffs).sum()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    with torch.no_grad():
        a_final = (_bernstein_basis(degree, sample_ts) @ coeffs.detach()).cpu().numpy()
        s_arr, v_arr = _forward_integrate(torch.as_tensor(a_final, dtype=torch.float64), dt)
        s_arr, v_arr = s_arr.cpu().numpy(), v_arr.cpu().numpy()
    return PlanningResult(
        times=np.arange(T + 1) * dt, s=s_arr, v=v_arr,
        a=a_final, losses=losses,
    )


def run_longitudinal_planning_demo(
    *, save_dir: Path | None = None
) -> dict[str, dict[str, Any]]:
    """Run all three parameterizations on the default 10s / 10m / 2m/s problem."""
    s_target, v_target, T, dt = 10.0, 2.0, 20, 0.5
    state = plan_via_state_points(s_target=s_target, v_target=v_target, T=T, dt=dt, n_steps=500)
    control = plan_via_control_points(s_target=s_target, v_target=v_target, T=T, dt=dt, n_steps=500)
    poly = plan_via_polynomial(s_target=s_target, v_target=v_target, T=T, dt=dt, n_steps=500)
    sub = lambda r: {  # noqa: E731
        "final_loss": float(r.losses[-1]),
        "terminal_s": float(r.s[-1]),
        "terminal_v": float(r.v[-1]),
        "times": r.times, "s": r.s, "v": r.v, "a": r.a,
    }
    result: dict[str, dict[str, Any]] = {
        "state_points": sub(state),
        "control_points": sub(control),
        "polynomial": sub(poly),
    }

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig = trajectory_1d(
            state.times,
            {"state-points": state.v, "control-points": control.v, "Bernstein poly": poly.v},
            title="speed profiles (3 parameterizations, s*=10, v*=2)",
            ylabel="v(t) [m/s]",
        )
        save_figure(fig, save_dir / "fig-13-speed-profiles-3-param.png")
        fig = trajectory_1d(
            np.arange(T) * dt,
            {"state-points": state.a, "control-points": control.a, "Bernstein poly": poly.a},
            title="acceleration profiles (3 parameterizations)",
            ylabel="a(t) [m/s^2]",
        )
        save_figure(fig, save_dir / "fig-14-acceleration-profiles.png")
    return result


if __name__ == "__main__":
    for name, sub in run_longitudinal_planning_demo().items():
        print(f"{name:>15s}: final_loss={sub['final_loss']:.4f} "
              f"terminal_s={sub['terminal_s']:.3f} terminal_v={sub['terminal_v']:.3f}")
