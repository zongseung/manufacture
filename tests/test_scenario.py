from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from gmst import scenario as sc

H = {date(2021, 2, 13), date(2021, 8, 16)}


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


def test_tariff_source_is_independent_of_dataset_rate() -> None:
    # Given the tariff table; when inspecting source; then the dataset rate is unused.
    assert "전기요금" not in Path(sc.__file__).read_text(encoding="utf-8")
    assert sc.TARIFF["I"]["base"] == 7220
    assert sc.TARIFF["III"]["winter"] == (62.5, 108.6, 155.5)


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
