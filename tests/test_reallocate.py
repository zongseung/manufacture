from datetime import date

import numpy as np
import torch

from gmst import reallocate as ra

SUMMER = date(2021, 7, 7)  # Wednesday
Q = np.r_[np.zeros(2), np.full(13, 10.0), np.zeros(9)]  # hours 2..14


def _toy(spike: float = 5.0) -> ra.Forecaster:
    J = torch.zeros(96, 24, dtype=torch.float64)
    for h in range(24):
        J[4 * h:4 * h + 4, h] = spike if h == 14 else 1.0
    return lambda x: 100.0 + J @ x


def _run(f: ra.Forecaster, **kw: object) -> ra.ReallocResult:
    energy, mask = ra.tou_inputs(SUMMER, False)
    args = {"window": ra.plan_window(Q), "cap": np.full(24, 20.0), "energy_rate": energy,
            "demand_mask": mask, "labor": ra.labor_multiplier(SUMMER, False, True), "steps": 150} | kw
    return ra.reallocate(f, Q, **args)  # type: ignore[arg-type]


def _check_rules(q_new: np.ndarray, q: np.ndarray, window: np.ndarray, cap: np.ndarray, rho: float) -> None:
    tol = 1e-6 * q.sum()
    assert abs(q_new.sum() - q.sum()) < tol
    assert np.allclose(q_new[~window], q[~window], atol=tol)
    assert (q_new >= -tol).all() and (q_new <= np.maximum(cap, q) + tol).all()
    assert np.abs(q_new - q).sum() / 2 <= rho * q.sum() + tol


def test_project_feasible_and_idempotent() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.uniform(0, 10, 24) * (rng.random(24) > 0.3)
        window = ra.plan_window(q, extend=1)
        cap = np.maximum(q, rng.uniform(0, 15, 24))
        lo, hi = np.where(window, 0.0, q), np.where(window, cap, q)
        rho = rng.uniform(0.05, 0.4)
        x = ra.project(q + rng.normal(0, 5, 24), q, lo, hi, rho)
        _check_rules(x, q, window, cap, rho)
        assert np.allclose(ra.project(x, q, lo, hi, rho), x, atol=1e-6 * q.sum())


def test_reallocate_lowers_afternoon_spike() -> None:
    res = _run(_toy())
    assert res["peak_after"] < res["peak_before"]
    assert res["cost_after"]["total"] < res["cost_before"]["total"]
    _check_rules(res["q_new"], Q, ra.plan_window(Q), np.full(24, 20.0), 0.2)
    assert res["moved"] <= 0.2 * Q.sum() + 1e-6


def test_flat_forecaster_returns_plan() -> None:
    res = _run(lambda x: 100.0 + 0.0 * x.sum() * torch.ones(96, dtype=torch.float64))
    assert np.array_equal(res["q_new"], Q)


def test_labor_premium_keeps_production_out_of_night() -> None:
    night = ra.labor_multiplier(SUMMER, False, True) > 1
    res = _run(_toy(), w=1e5)
    assert res["q_new"][night].sum() <= Q[night].sum() + 1e-6


def test_tou_inputs_summer_weekday_and_sunday() -> None:
    energy, mask = ra.tou_inputs(SUMMER, False)
    hours = np.arange(96) // 4
    assert np.array_equal(mask, (hours >= 9) & (hours < 23))
    top = (hours >= 10) & (hours < 12) | (hours >= 13) & (hours < 17)
    assert np.array_equal(energy == energy.max(), top)
    assert not ra.tou_inputs(date(2021, 7, 11), False)[1].any()


def test_small_helpers() -> None:
    q = np.zeros(24)
    q[[3, 7]] = 1.0
    assert np.flatnonzero(ra.plan_window(q)).tolist() == [3, 4, 5, 6, 7]
    assert np.flatnonzero(ra.plan_window(q, extend=4)).tolist() == list(range(12))
    assert not ra.plan_window(np.zeros(24)).any()
    lab = ra.labor_multiplier(SUMMER, False, True)
    assert lab[9:18].tolist() == [1.0] * 9 and lab[8] == lab[18] == 1.5
    assert (ra.labor_multiplier(SUMMER, True, True) == 1.5).all()
    prod = np.array([np.arange(24.0), np.zeros(24), np.arange(24.0) + 2, np.full(24, np.nan)])
    assert np.allclose(ra.hourly_cap(prod, 50), np.arange(24.0) + 1)
