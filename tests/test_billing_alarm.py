from datetime import date, timedelta

import numpy as np

from gmst import billing_alarm as ba
from gmst import features
from gmst.contracts import Panel
from gmst.scenario import RATCHET_MONTHS, _timestamps, band, tariff_holidays

H = tariff_holidays()


def test_band_mask_matches_the_tariff_bands_and_is_empty_on_sundays() -> None:
    day = date(2021, 7, 7)
    np.testing.assert_array_equal(ba.band_mask(day, H), [band(t, False, False) >= 1 for t in _timestamps(day)])
    assert not ba.band_mask(date(2021, 7, 11), H).any()  # Sunday


def test_baseline_uses_only_days_before_d_in_month_or_past_ratchet_months() -> None:
    panel = features.load_panel()
    dates = panel["dates"]
    for d in features.role_idx(panel, "f3", "val")[:6]:
        day = dates[d]
        hidden: Panel = {**panel, "Y": panel["Y"].copy()}
        hidden["Y"][d:] = 1e9
        assert ba.baseline(hidden, int(d), H) == ba.baseline(panel, int(d), H)
        eligible = [ba.band_peak(panel, j, H) for j in range(d) if (dates[j].year, dates[j].month) == (day.year, day.month)
                    or (dates[j].month in RATCHET_MONTHS and dates[j] > day - timedelta(days=365))]
        assert ba.baseline(panel, int(d), H) == np.nanmax(eligible)


def test_decision_rules_follow_expected_cost_and_probability() -> None:
    maxima = np.array([190., 200., 210., 230.])
    p, cost = ba.risk(maxima, 205.)
    assert p == .5 and cost == ba.BASE * (5 + 25) / 4
    assert ba.alarm(cost, p, "cost30000") == (cost >= 30_000)
    assert ba.alarm(cost, p, "p50") is True


def test_backtest_counts_caught_renewals_and_excess_at_stake() -> None:
    import polars as pl

    days = pl.DataFrame({"B": [200., 200., 200.], "observed": [210., 190., 205.], "p": [.9, .8, .1],
                         "cost": [5e4, 4e4, 1e3], "cbat_median": [205., 199., 199.], "b0_kind": [201., 150., float("nan")]})
    s = {r["rule"]: r for r in ba.backtest(days).iter_rows(named=True)}
    assert s["cost30000"]["renewals"] == 2 and s["cost30000"]["caught"] == 1 and s["cost30000"]["false_alarms"] == 1
    assert s["cost30000"]["excess_won_caught"] == 10 * ba.BASE and s["always"]["excess_won_all"] == 15 * ba.BASE
    assert s["b0_kind"]["alarms"] == 1 and s["cbat_point"]["caught"] == 1
