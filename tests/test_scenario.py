from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from gmst import scenario as sc
from gmst.contracts import FloatArray, Panel

H = {date(2021, 2, 13), date(2021, 8, 16)}
P0 = np.r_[np.zeros(8), np.full(10, 10.0), np.zeros(6)]


@pytest.mark.parametrize("ts,won", [
    (datetime(2021, 7, 7, 10), 4777.5), (datetime(2021, 7, 10, 11), 2725.0),
    (datetime(2021, 7, 11, 11), 1402.5), (datetime(2021, 8, 16, 11), 1402.5),
    (datetime(2021, 1, 5, 18), 4167.5), (datetime(2021, 2, 13, 11), 1577.5),
    (datetime(2021, 9, 8, 10), 2732.5),
])
def test_energy_when_historical_band_applies(ts: datetime, won: float) -> None:
    # Given one quarter-hour demand reading; when priced; then historical won.
    assert sc.energy_won(np.array([100.0]), [ts], H) == pytest.approx(won)


@pytest.mark.parametrize("ts,expected", [
    (datetime(2021, 7, 7, 9), 1), (datetime(2021, 7, 7, 8, 45), 0),
    (datetime(2021, 7, 7, 17), 1), (datetime(2021, 7, 7, 23), 0),
    (datetime(2021, 1, 5, 13), 1), (datetime(2021, 1, 5, 21), 1),
    (datetime(2021, 1, 5, 22, 30), 2), (datetime(2021, 7, 10, 11), 2),
])
def test_demand_band_when_boundary_or_saturday(ts: datetime, expected: int) -> None:
    # Given an interval start; when classified; then demand rules apply.
    assert sc.band(ts, holiday=False, for_energy=False) == expected


def test_demand_when_offpeak_readings_are_higher() -> None:
    # Given offpeak 200 and demand-band 150; when billed; then only 150 counts.
    ts = [datetime(2021, 7, 7) + timedelta(minutes=15 * q) for q in range(96)]
    y = np.where([t.hour < 9 or t.hour >= 23 for t in ts], 200.0, 150.0)
    assert sc.billing_demand(y, ts, H) == 150.0
    assert sc.expected_p_app(np.array([200.0, 230.0, 240.0]), 222.0) == pytest.approx(692 / 3)
    assert sc.basic_won(10.0) == 83200.0


def test_transforms_when_donors_and_receivers_exist() -> None:
    # Given a ten-hour schedule; when transformed; then exact movement conserves production.
    shifted = sc.shift_peak(P0, date(2021, 7, 7), 0.1, H)
    np.testing.assert_allclose(shifted[[8, 9, 12, 17]], 11.5)
    np.testing.assert_allclose(shifted[[10, 11, 13, 14, 15, 16]], 9.0)
    assert sc.c_change(P0, shifted) == pytest.approx(0.06)
    staggered = sc.stagger_start(P0, 0.2)
    np.testing.assert_allclose(staggered[[8, 9]], [8.0, 12.0])
    median = np.full(96, 100.0)
    median[45] = 250.0
    avoided = sc.avoid_high(P0, 0.3, median, 210.7)
    assert avoided[11] == 7.0
    eased = sc.ease_peak(P0, 0.1, 58)
    np.testing.assert_allclose(eased[[14, 15]], 9.0)
    for actual in (shifted, staggered, avoided, eased):
        assert actual.sum() == pytest.approx(P0.sum(), abs=1e-9)
        assert (actual >= 0).all()
    np.testing.assert_array_equal(P0, np.r_[np.zeros(8), np.full(10, 10.0), np.zeros(6)])


def test_transforms_when_no_receivers_or_holiday() -> None:
    # Given a holiday or isolated production; when moving; then the schedule is unchanged.
    np.testing.assert_array_equal(sc.shift_peak(P0, date(2021, 7, 11), 0.1, H), P0)
    only = np.r_[5.0, np.zeros(23)]
    np.testing.assert_array_equal(sc.stagger_start(only, 0.2), only)
    np.testing.assert_array_equal(sc.avoid_high(P0, 0.3, np.full(96, 300.0), 200.0), P0)
    assert sc.c_change(np.zeros(24), np.zeros(24)) == 0.0


@pytest.mark.parametrize("schedule,share", [(np.full(24, -1.0), 0.1), (np.ones(23), 0.1),
                                             (np.full(24, np.nan), 0.1), (np.ones(24), 1.1)])
def test_schedule_when_invalid_is_rejected(schedule: FloatArray, share: float) -> None:
    # Given an invalid boundary schedule; when transforming; then reject it.
    with pytest.raises(ValueError):
        sc.stagger_start(schedule, share)


def test_tariff_source_is_independent_of_dataset_rate() -> None:
    # Given the tariff table; when inspecting source; then the dataset rate is unused.
    assert "전기요금" not in Path(sc.__file__).read_text(encoding="utf-8")
    assert sc.TARIFF["I"]["base"] == 7220
    assert sc.TARIFF["III"]["winter"] == (62.5, 108.6, 155.5)


def test_monthly_when_paths_cross_the_ratchet_floor() -> None:
    # Given crossed path maxima across two dates; when billed; then mean(pathwise max) is used.
    dates = [date(2021, 7, 19), date(2021, 7, 21), date(2021, 7, 22)]
    y = np.zeros((3, 96))
    y[0, 45] = 222.0
    panel: Panel = {"dates": dates, "Y": y, "X": {}, "is_missing": np.zeros((3, 96), bool),
                    "days": pl.DataFrame({"date": dates}), "op": np.ones(3, dtype=np.int8),
                    "hol": np.zeros(3, dtype=np.int8), "dtype": np.zeros(3, dtype=np.int8),
                    "dow": np.array([0, 2, 3], dtype=np.int8), "month": np.full(3, 7, dtype=np.int8)}
    def stats(day: date, maxima: list[float], energy: float, kwh: float, movement: float) -> sc._DayStats:
        return {"day": day, "maxima": np.array(maxima), "energy": energy, "kwh": kwh,
                "production": 100.0, "movement": movement, "risk_ok": True}
    groups: dict[tuple[int, sc.Transform, float], list[sc._DayStats]] = {(7, "baseline", 0.0): [stats(dates[1], [100.0, 300.0], 1000.0, 10.0, 0.0), stats(dates[2], [400.0, 100.0], 2000.0, 20.0, 0.0)],
              (7, "shift_peak", 0.1): [stats(dates[1], [100.0, 250.0], 900.0, 9.0, 20.0), stats(dates[2], [300.0, 100.0], 1800.0, 18.0, 20.0)]}
    monthly = sc._monthly(panel, groups, H)
    shifted = monthly.row(1, named=True)
    assert monthly.columns == sc.SCN_MONTH_COLS
    assert shifted["E_P_app_base"] == 350.0 and shifted["E_P_app_scn"] == 275.0
    assert shifted["P_floor"] == 222.0 and shifted["d_demand_won"] == -624000.0
    assert shifted["d_energy_won"] == -300.0 and shifted["d_kwh"] == -3.0
    assert shifted["C_change"] == 0.1 and shifted["eta_star"] == 6243000.0


def test_floor_and_holiday_switch_when_real_panel_stays_sealed() -> None:
    # Given the ordinary sealed panel; when using history; then July/August floors remain 222.
    from gmst import features as ft

    p = ft.load_panel()
    holidays = sc.tariff_holidays()
    usable = ft.usable_peak(p)
    excluded = {p["dates"][int(i)] for fold in ("f2", "f3", "f4")
                for i in ft.role_idx(p, fold, "val") if usable[i] and p["op"][i] == 1}
    for month in (7, 8):
        assert sc.ratchet_floor(p, month, excluded, holidays) == (222.0, datetime(2021, 7, 19, 11, 15))
    assert sc.tariff_holidays(False) == holidays - {date(2021, 8, 16)}


def test_run_scenarios_when_f4_state_is_forward_and_sealed() -> None:
    # Given a small actual July-only SCENARIO fit; when simulated twice; then schemas and CRN agree.
    from gmst import analysis, evaluate, features, hmm

    panel = features.load_panel()
    model = hmm.hmm_model("B4", protocol="B", train_from=date(2021, 7, 1), max_epochs=1, N=20, device="cpu")
    state = model["fit"](panel, "f4", evaluate.thresholds(panel, features.role_idx(panel, "f4", "train"))[0])
    before = panel["X"]["생산량"].copy()
    daily, monthly = sc.run_scenarios(panel, {"f4": state}, N=20, max_days=2)
    again, again_month = sc.run_scenarios(panel, {"f4": state}, N=20, max_days=2)
    assert len(state["train_idx"]) == 47
    assert daily.columns == sc.SCN_DAY_COLS and monthly.columns == sc.SCN_MONTH_COLS
    assert daily.shape == (26, 14) and monthly.shape == (13, 15)
    assert daily.equals(again) and monthly.equals(again_month)
    np.testing.assert_array_equal(panel["X"]["생산량"], before)
    baseline = monthly.filter(pl.col("scenario") == "baseline")
    assert baseline["d_demand_won"][0] == baseline["d_kwh"][0] == baseline["d_total_won"][0] == 0.0
    assert monthly["P_floor"].unique().to_list() == [222.0]
    assert daily.filter(pl.col("scenario") == "baseline")["risk_ok"].all()
    transition = analysis.transitions(state, panel)
    assert transition.columns == ["op", "daytype", "hour", "from_state", "p_to_high"]
    assert transition["p_to_high"].is_between(0, 1).all()
    issue = int(features.role_idx(panel, "f4", "val")[0])
    changed: Panel = {**panel, "Y": panel["Y"].copy(), "X": {key: value.copy() for key, value in panel["X"].items()}}
    changed["Y"][issue:] = 999.0
    for key, values in changed["X"].items():
        values[issue + (key == "생산량"):] = 999.0
    altered, _ = sc.run_scenarios(changed, {"f4": state}, N=20, max_days=1)
    assert daily.head(13).equals(altered)
    empty_days, empty_months = sc.run_scenarios(panel, {"f4": state}, N=20, max_days=0)
    assert empty_days.schema == daily.schema and empty_months.schema == monthly.schema
