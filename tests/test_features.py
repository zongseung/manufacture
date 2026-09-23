from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from gmst.contracts import Panel


def synthetic_panel(n: int) -> Panel:
    dates = [date(2021, 1, 1) + timedelta(days=day) for day in range(n)]
    return {"dates": dates, "Y": np.ones((n, 96)), "X": {"생산량": np.ones((n, 96))},
            "is_missing": np.zeros((n, 96), dtype=np.bool_),
            "days": pl.DataFrame({"date": dates, "f1": ["train"] * n}),
            "op": np.ones(n, dtype=np.int8), "hol": np.zeros(n, dtype=np.int8),
            "dtype": np.zeros(n, dtype=np.int8), "dow": np.zeros(n, dtype=np.int8),
            "month": np.ones(n, dtype=np.int8)}


@pytest.mark.parametrize("copies,suspect,count", [(False, False, 11352), (True, False, 23064), (False, True, 11544), (True, True, 23256)])
def test_loader_applies_policy_before_any_model(copies: bool, suspect: bool, count: int) -> None:
    from gmst import features

    panel = features.load_panel(copies, suspect)
    assert panel["Y"].shape == (257, 96) and panel["Y"].dtype == np.float64
    assert np.isfinite(panel["Y"]).sum() == count
    september = features.role_idx(panel, "test", "test")
    assert np.isnan(panel["Y"][september]).all()
    assert all(np.isnan(values[september]).all() for values in panel["X"].values())
    if not copies and not suspect:
        assert np.isfinite(panel["Y"][:panel["dates"].index(date(2021, 8, 31))]).sum() == 11256
        assert np.isfinite(panel["Y"][:panel["dates"].index(date(2021, 7, 6))]).sum() == 6240
        assert all(np.isnan(panel["X"]["생산량"][panel["dates"].index(day)]).all() for day in (date(2021, 7, 13), date(2021, 7, 15)))
        assert np.isfinite(panel["X"]["기온"][panel["dates"].index(date(2021, 3, 1))]).all()


def test_protocol_calendar_permissions_and_usable_days() -> None:
    from gmst import features

    panel = features.load_panel()
    op, hol = features.cal_flags(panel, "A")
    assert np.all(op == 1) and np.all(hol == 0)
    op, hol = features.cal_flags(panel, "A+")
    np.testing.assert_array_equal(op, panel["op"])
    np.testing.assert_array_equal(hol, panel["hol"])
    assert features.PROTOCOLS == {"A": set(), "A+": {"op", "hol"}, "A+W*": {"op", "hol", "wx_obs"}, "B": {"op", "hol", "prod"}}
    outer = np.concatenate([features.role_idx(panel, fold, "val") for fold in ("f1", "f2", "f3", "f4")])
    assert np.isfinite(panel["Y"][outer]).any(axis=1).sum() == 53
    assert [features.usable_peak(panel)[features.role_idx(panel, fold, "val")].sum() for fold in ("f1", "f2", "f3", "f4")] == [12, 13, 14, 12]


@pytest.mark.parametrize("n,expected", [(3, (3, 0, 0)), (5, (5, 0, 0)), (9, (5, 4, 0)), (12, (5, 5, 2)), (19, (5, 7, 7))])
def test_internal_split_reserves_usable_days(n: int, expected: tuple[int, int, int]) -> None:
    from gmst import features

    panel = synthetic_panel(n)
    got = features.internal_split(panel, "f1")
    assert tuple(map(len, got)) == expected
    assert np.array_equal(np.concatenate(got), np.arange(n))
    np.testing.assert_array_equal(features.tuning_idx(panel, "f1"), got[1])
    np.testing.assert_array_equal(features.inner_idx(panel, "f1"), got[2])


def test_internal_split_keeps_masked_history_and_restricts_scenario() -> None:
    from gmst import features

    panel = synthetic_panel(24)
    panel["Y"][[1, 5, 10, 15, 20]] = np.nan
    fit, tune, cal = features.internal_split(panel, "f1")
    assert np.isfinite(panel["Y"][fit]).any(axis=1).sum() == 5
    assert {1, 5}.issubset(fit)
    assert len(tune) == len(cal) == 7
    assert fit[-1] < tune[0] <= tune[-1] < cal[0]
    real = features.load_panel()
    july = np.array([i for i, day in enumerate(real["dates"]) if day >= date(2021, 7, 1)], dtype=np.int64)
    fit, tune, cal = features.internal_split(real, "f1", july)
    assert len(fit) == 5 and len(tune) == len(cal) == 0
    assert all(real["dates"][i] <= date(2021, 7, 5) for i in fit)


def test_unseal_only_uses_synthetic_measurements(monkeypatch: pytest.MonkeyPatch) -> None:
    from gmst import features

    dates = [date(2021, 8, 31), date(2021, 9, 1)]
    frame = pl.DataFrame({"datetime": [datetime.combine(day, datetime.min.time()) + timedelta(minutes=15 * slot) for day in dates for slot in range(96)],
                          "전력": [17.] * 192, "생산량": [1.] * 192, "기온": [2.] * 192, "풍속": [3.] * 192,
                          "습도": [4.] * 192, "강수량_증분": [0.] * 192, "is_missing": [False] * 192})
    days = pl.DataFrame({"date": dates, "is_copy": [False] * 2, "is_suspect": [False] * 2,
                        "is_operating": [True] * 2, "is_holiday": [False] * 2, "daytype": ["wk"] * 2, "test": ["gap", "test"]})
    monkeypatch.setattr(features, "build_15min", lambda: frame)
    monkeypatch.setattr(features, "build_days", lambda _: days)
    opened = features.load_panel(unseal=True)
    np.testing.assert_array_equal(opened["Y"][1], np.full(96, 17.))
    assert np.isfinite(opened["X"]["생산량"][1]).all()
