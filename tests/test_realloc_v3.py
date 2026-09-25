from datetime import date
from functools import cache

import numpy as np
import torch

from gmst import bat
from gmst import evaluate as ev
from gmst import features as ft
from gmst import realloc_v3 as rv
from gmst import reallocate as ra
from gmst.scenario import _timestamps, energy_won, tariff_holidays


@cache
def fitted() -> tuple[dict, bat.BATState, int]:
    panel = ft.load_panel()
    tr = ft.role_idx(panel, "f1", "train")
    C, _ = ev.thresholds(panel, tr)
    val = ft.role_idx(panel, "f1", "val")
    return panel, bat.fit_bat(panel, tr, C, n_iter=200, burn=100), int(val[panel["op"][val] == 1][0])


def test_draw_curves_average_to_curve_fn() -> None:
    panel, state, d = fitted()
    q = np.nan_to_num(panel["X"]["생산량"][d, ::4])
    q_new = q.copy()
    on = np.flatnonzero(q > 0)
    moved = 0.5 * q[on[-1]]
    q_new[on[0]] += moved
    q_new[on[-1]] -= moved  # log channel moves, on/off unchanged: must match curve_fn exactly
    Y = rv.draw_curves(state, panel, d, np.stack([q, q_new]))
    f = bat.curve_fn(state, panel, d)
    for i, plan in enumerate((q, q_new)):
        np.testing.assert_allclose(Y[i].mean(0), f(torch.tensor(plan)).numpy(), rtol=1e-8, atol=1e-8)
    q_off = q.copy()
    q_off[on[-1]] = 0.0  # turning an hour off never raises any slot (w > 0, α ≥ 0)
    assert (rv.draw_curves(state, panel, d, q_off)[0].mean(0) <= Y[0].mean(0) + 1e-9).all()


def test_robust_rule_needs_p95_and_rates_match_energy_won() -> None:
    rng = np.random.default_rng(0)
    y0 = 100 + rng.normal(0, 1, (1000, 96))
    mask = np.zeros(96, bool)
    mask[40:60] = True
    rate = np.full(96, 100.0)
    for share, expect in ((0.94, False), (0.96, True)):
        y1 = y0.copy()
        y1[: int(share * 1000)] -= 50.0  # these draws lower every slot; the rest raise it
        y1[int(share * 1000):] += 50.0
        s = rv.robust_stats(y0, y1, mask, rate)
        assert abs(s["P_robust"] - share) < 1e-12 and s["recommended"] is expect
    assert rv.robust_stats(y0, y1, np.zeros(96, bool), rate)["recommended"] is False  # holiday: no demand band
    day = date(2021, 7, 7)
    r, _ = ra.tou_inputs(day, day in tariff_holidays(), rv.OPTION)
    y = rng.uniform(50, 200, 96)
    assert np.isclose(y @ (0.25 * r), energy_won(y, _timestamps(day), tariff_holidays(), rv.OPTION))
