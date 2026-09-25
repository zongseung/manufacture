"""Recommend a same-day hourly production re-allocation that lowers modelled cost.

f maps a torch (24,) hourly plan to (96,) quarter-hour kW; we minimize
demand·LSE(peak) + TOU energy + labor premium with Adam, projecting every step onto
{Σq′=Σq, window/box, ‖q′−q‖₁/2 ≤ ρΣq} via Dykstra. Model what-ifs, not causal claims.
"""
from collections.abc import Callable
from datetime import date
from typing import TypedDict

import numpy as np
import torch

from gmst.contracts import BoolArray, FloatArray
from gmst.scenario import TARIFF, Option, _timestamps, band, rate

type Forecaster = Callable[[torch.Tensor], torch.Tensor]


class Cost(TypedDict):
    demand: float
    energy: float
    labor: float
    total: float


class ReallocResult(TypedDict):
    q_new: FloatArray
    y_before: FloatArray
    y_after: FloatArray
    peak_before: float
    peak_after: float
    cost_before: Cost
    cost_after: Cost
    moved: float


def tou_inputs(day: date, holiday: bool, option: Option = "II") -> tuple[FloatArray, BoolArray]:
    ts = _timestamps(day)
    return (np.array([rate(t, holiday, option) for t in ts]),
            np.array([band(t, holiday, False) >= 1 for t in ts]))


def labor_multiplier(day: date, holiday: bool, operating: bool) -> FloatArray:
    """1.0 inside 09–18 on operating weekdays, 1.5 night/overtime (근로기준법 제56조)."""
    regular = operating and not holiday and day.weekday() < 5
    return np.array([1.0 if regular and 9 <= h < 18 else 1.5 for h in range(24)])


def plan_window(q: FloatArray, extend: int = 0) -> BoolArray:
    on = np.flatnonzero(q > 0)
    window = np.zeros(24, dtype=bool)
    if on.size:
        window[max(0, on[0] - extend):min(24, on[-1] + extend + 1)] = True
    return window


def hourly_cap(prod_hourly: FloatArray, q: float = 95) -> FloatArray:
    rows = prod_hourly[np.nansum(prod_hourly, axis=1) > 0]
    if not rows.size:
        raise ValueError("No operating rows to derive hourly capacity")
    return np.nan_to_num(np.nanpercentile(rows, q, axis=0))


def _box_sum(x: FloatArray, lo: FloatArray, hi: FloatArray, target: float) -> FloatArray:
    """Exact projection onto {lo ≤ z ≤ hi, Σz = target}: bisection on the shift λ."""
    a, b = float((x - hi).min()), float((x - lo).max())
    for _ in range(100):
        lam = 0.5 * (a + b)
        a, b = (lam, b) if np.clip(x - lam, lo, hi).sum() > target else (a, lam)
    return np.clip(x - 0.5 * (a + b), lo, hi)


def _l1_ball(v: FloatArray, r: float) -> FloatArray:
    """Euclidean projection onto {‖v‖₁ ≤ r} (Duchi et al. 2008, sort-based)."""
    u = np.abs(v)
    if u.sum() <= r:
        return v.copy()
    s = np.sort(u)[::-1]
    c = np.cumsum(s) - r
    k = np.flatnonzero(s > c / np.arange(1, s.size + 1))[-1]
    return np.sign(v) * np.maximum(u - c[k] / (k + 1), 0.0)


def violation(x: FloatArray, q: FloatArray, lo: FloatArray, hi: FloatArray, rho: float) -> float:
    return max(abs(x.sum() - q.sum()), float(np.maximum(lo - x, 0).max()), float(np.maximum(x - hi, 0).max()),
               np.abs(x - q).sum() / 2 - rho * q.sum(), 0.0)


def project(x: FloatArray, q: FloatArray, lo: FloatArray, hi: FloatArray, rho: float,
            iters: int = 500) -> FloatArray:
    """Dykstra between box∩hyperplane and the L1 ball around q, then an exact feasibility pull.

    q must itself be feasible; the final step shrinks toward q along the segment, which
    stays in the (convex) box∩hyperplane and lands exactly inside the L1 ball.
    """
    x, q = np.asarray(x, float), np.asarray(q, float)
    total, r, tol = float(q.sum()), 2 * rho * float(q.sum()), 1e-6 * max(float(q.sum()), 1.0)
    z, p, s = x.copy(), np.zeros_like(x), np.zeros_like(x)
    y = _box_sum(z, lo, hi, total)
    for _ in range(iters):
        y = _box_sum(z + p, lo, hi, total)
        p = z + p - y
        z_next = q + _l1_ball(y + s - q, r)
        s = y + s - z_next
        z = z_next
        if violation(z, q, lo, hi, rho) < tol:
            break
    moved = np.abs(y - q).sum()
    out = y if moved <= r else q + (r / moved) * (y - q)
    assert violation(out, q, lo, hi, rho) < tol, violation(out, q, lo, hi, rho)
    return out


def reallocate(f: Forecaster, q: FloatArray, *, window: BoolArray, cap: FloatArray, energy_rate: FloatArray,
               demand_mask: BoolArray, labor: FloatArray,
               # ponytail: 일 단위 기본요금/30 환산·래칫 무시, 월 래칫(scenario.ratchet_floor)이 걸리는 달이면 max(peak, floor) 로 교체
               demand_weight: float = TARIFF["II"]["base"] / 30,
               w: float = 0.0, rho: float = 0.2, tau: float = 1.0, steps: int = 300,
               lr: float | None = None, seed: int = 0) -> ReallocResult:
    q = np.asarray(q, float)
    # Hours already above cap keep their level as the ceiling so q itself stays feasible.
    lo, hi = np.where(window, 0.0, q), np.where(window, np.maximum(cap, q), q)
    rate_t, labor_t = torch.as_tensor(energy_rate, dtype=torch.float64), torch.as_tensor(labor, dtype=torch.float64)
    mask_t = torch.as_tensor(demand_mask, dtype=torch.bool)

    def terms(x: torch.Tensor, smooth: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        y = f(x)
        peak = y.new_zeros(()) if not mask_t.any() else (
            tau * torch.logsumexp(y[mask_t] / tau, 0) if smooth else y[mask_t].max())
        return y, demand_weight * peak, (0.25 * rate_t * y).sum(), w * (labor_t * x).sum()

    def evaluate(x: FloatArray) -> tuple[FloatArray, float, Cost]:
        with torch.no_grad():
            y, d, e, l = terms(torch.as_tensor(x, dtype=torch.float64), smooth=False)
        cost: Cost = {"demand": float(d), "energy": float(e), "labor": float(l), "total": float(d + e + l)}
        return y.numpy().copy(), float(y[mask_t].max()) if mask_t.any() else 0.0, cost

    torch.manual_seed(seed)
    x = torch.tensor(q, dtype=torch.float64, requires_grad=True)
    step = lr if lr is not None else 0.02 * max(float(q.sum()), 1.0) / max(int(window.sum()), 1)
    opt = torch.optim.Adam([x], lr=step)
    y0, peak0, cost0 = evaluate(q)
    best, best_cost = q, cost0
    for _ in range(steps if window.any() and q.sum() > 0 else 0):
        opt.zero_grad()
        _, d, e, l = terms(x, smooth=True)
        (d + e + l).backward()
        with torch.no_grad():  # tangent to Σq′=Σq: Adam's per-coordinate scaling would erase a common-mode gradient
            free = torch.as_tensor(window)
            x.grad[free] -= x.grad[free].mean()
            x.grad[~free] = 0.0
        opt.step()
        with torch.no_grad():
            x.copy_(torch.as_tensor(project(x.detach().numpy(), q, lo, hi, rho)))
        candidate = x.detach().numpy().copy()
        _, _, cost = evaluate(candidate)
        if cost["total"] < best_cost["total"] - 1e-9 * max(abs(cost0["total"]), 1.0):
            best, best_cost = candidate, cost
    y1, peak1, cost1 = evaluate(best)
    return {"q_new": best, "y_before": y0, "y_after": y1, "peak_before": peak0, "peak_after": peak1,
            "cost_before": cost0, "cost_after": cost1, "moved": float(np.abs(best - q).sum() / 2)}
