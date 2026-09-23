from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from gmst import backbone as bb
from gmst.contracts import Panel


def synthetic_panel(n: int = 40) -> Panel:
    dates = [date(2021, 1, 1) + timedelta(days=d) for d in range(n)]
    dow = np.array([d.weekday() for d in dates], dtype=np.int8)
    y = 80 + np.arange(n)[:, None] + 20 * np.sin(np.arange(96)[None, :] / 96 * 2 * np.pi)
    return {
        "dates": dates, "Y": y, "X": {k: y / 10 for k in ("생산량", "기온", "풍속", "습도", "강수량_증분")},
        "is_missing": np.zeros_like(y, dtype=bool),
        "days": pl.DataFrame({"date": dates, "f1": ["train"] * (n - 5) + ["val"] * 5}),
        "op": np.ones(n, dtype=np.int8), "hol": np.zeros(n, dtype=np.int8),
        "dtype": np.where(dow == 6, 2, np.where(dow == 5, 1, 0)).astype(np.int8),
        "dow": dow, "month": np.ones(n, dtype=np.int8),
    }


def test_rw2_solution_matches_weighted_normal_equation() -> None:
    # Given a masked sample and the circular penalty.
    p = synthetic_panel()
    p["Y"][4, 17] = np.nan
    tr = np.arange(20)
    # When fitting one pooled operating profile.
    profiles, fallback = bb.fit_backbone(p["Y"], p["op"], p["dtype"], tr, 10, 60, 100)
    # Then its coefficients solve the stated weighted system without a ridge.
    w = 2 ** (-(19 - tr) / 60)
    mask = np.isfinite(p["Y"][tr])
    lhs = np.diag((w[:, None] * mask).sum(axis=0)) + 10 * bb.C2.T @ bb.C2
    rhs = (w[:, None] * np.nan_to_num(p["Y"][tr])).sum(axis=0)
    np.testing.assert_allclose(lhs @ profiles[1, 0], rhs, atol=1e-9)
    assert fallback == [(o, t) for o in range(2) for t in range(3)]


def test_asof_ignores_previous_day_and_future() -> None:
    # Given an issue day with earlier history.
    p = synthetic_panel()
    expected = bb.backbone_asof(p["Y"], p["op"], p["dtype"], 25, 10, 60)
    p["Y"][24:] = 9999
    # When changing data later than d-2.
    actual = bb.backbone_asof(p["Y"], p["op"], p["dtype"], 25, 10, 60)
    # Then the backbone feature remains exactly unchanged.
    np.testing.assert_array_equal(actual, expected)


def test_tau_zero_retains_nan_for_unobserved_slot() -> None:
    # Given an entirely missing slot and no smoothing.
    p = synthetic_panel()
    p["Y"][:, 0] = np.nan
    # When fitting cell means.
    profiles, _ = bb.fit_backbone(p["Y"], p["op"], p["dtype"], np.arange(30), 0, None, 1)
    # Then observed slots are means and unsupported slots remain unknown.
    assert np.isnan(profiles[:, :, 0]).all()
    assert profiles[1, 0, 1] == pytest.approx(np.mean(p["Y"][:30][p["dtype"][:30] == 0, 1]))


def test_tau_selection_uses_disjoint_tuning_labels_only() -> None:
    from gmst.features import internal_split

    p = synthetic_panel()
    _, tune, cal = internal_split(p, "f1")
    expected = bb.select_tau(p, "f1")
    p["Y"][cal[0]:] *= 90
    actual = bb.select_tau(p, "f1")
    assert actual[:2] == expected[:2]
    assert actual[2].equals(expected[2])
    assert actual[2]["n_tune"].to_list() == [len(tune)] * 15


def test_tau_empty_tuning_uses_registered_default() -> None:
    p = synthetic_panel(10)
    tau, h, table = bb.select_tau(p, "f1")
    assert (tau, h) == (10., 60)
    assert table["status"].to_list() == ["default_empty"] * 15


def test_bb_calibration_state_excludes_calibration_dates() -> None:
    from gmst.features import inner_idx

    p = synthetic_panel()
    model = bb.bb_model(10., 60)
    state = model["fit"](p, "f1", np.array([110., 130., 150.]))
    inner = model["inner_state"](state)
    assert inner is not None
    assert inner["train_idx"].max() < inner_idx(p, "f1").min()
