"""Historical 2021 tariff: energy bands, billing demand and the ratchet floor.

Rates exclude tax and uniform kWh adders.
"""
from collections.abc import Sequence
from datetime import date, datetime, time, timedelta
from typing import Final, Literal, TypedDict

import numpy as np

from gmst.contracts import FloatArray, Panel

type Season = Literal["summer", "springfall", "winter"]
type Option = Literal["I", "II", "III"]


class Tariff(TypedDict):
    base: float
    summer: tuple[float, float, float]
    springfall: tuple[float, float, float]
    winter: tuple[float, float, float]


TARIFF: Final[dict[Option, Tariff]] = {
    "I": {"base": 7220, "summer": (61.6, 114.5, 196.6), "springfall": (61.6, 84.1, 114.8), "winter": (68.6, 114.7, 172.2)},
    "II": {"base": 8320, "summer": (56.1, 109.0, 191.1), "springfall": (56.1, 78.6, 109.3), "winter": (63.1, 109.2, 166.7)},
    "III": {"base": 9810, "summer": (55.2, 108.4, 178.7), "springfall": (55.2, 77.3, 101.0), "winter": (62.5, 108.6, 155.5)},
}
RATCHET_MONTHS: Final = (12, 1, 2, 7, 8, 9)


def season_name(month: int) -> Season:
    if not 1 <= month <= 12:
        raise ValueError(f"Month outside 1..12: {month}")
    if 6 <= month <= 8:
        return "summer"
    return "winter" if month <= 2 or month >= 11 else "springfall"


def tariff_holidays(tariff_0816: bool = True) -> set[date]:
    """Treat the 2021-08-16 substitute holiday as offpeak by assumption."""
    from gmst.splits import HOLIDAYS_2021

    return set(HOLIDAYS_2021) - (set() if tariff_0816 else {date(2021, 8, 16)})


def band(ts: datetime, holiday: bool, for_energy: bool) -> int:
    if holiday or ts.weekday() == 6 or ts.hour < 9 or ts.hour >= 23:
        return 0
    winter = ts.month <= 2 or ts.month >= 11
    peak = 10 <= ts.hour < 12 or (17 <= ts.hour < 20 or ts.hour == 22 if winter else 13 <= ts.hour < 17)
    return 2 if peak and not (for_energy and ts.weekday() == 5) else 1


def rate(ts: datetime, holiday: bool, option: Option = "II") -> float:
    return TARIFF[option][season_name(ts.month)][band(ts, holiday, True)]


def billing_demand(y: FloatArray, ts: Sequence[datetime], holidays: set[date]) -> float:
    return max((float(value) for value, t in zip(y, ts, strict=True)
                if np.isfinite(value) and band(t, t.date() in holidays, False) >= 1), default=0.0)


def _timestamps(day: date) -> list[datetime]:
    return [datetime.combine(day, time()) + timedelta(minutes=15 * q) for q in range(96)]