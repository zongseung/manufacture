"""Fixed day-level data policy and chronological fold roles."""
from datetime import date, timedelta
from typing import Final

import numpy as np
import polars as pl

from gmst import DATA
from gmst.contracts import FloatArray
from gmst.preprocess import build_15min

# 한국천문연구원 특일정보
# ponytail: 2021 공휴일 하드코딩, 다른 연도 데이터가 오면 특일정보 목록 교체 (FR-21)
HOLIDAYS_2021: Final = tuple(date(2021, month, day) for month, day in ((1, 1), (2, 11), (2, 12), (2, 13), (3, 1), (5, 5), (5, 19), (8, 16)))
SUSPECT: Final = (date(2021, 7, 13), date(2021, 7, 15))
EDITED: Final = tuple(date(2021, month, day) for month, day in ((1, 1), (1, 9), (1, 10), (1, 16), (3, 7), (3, 21), (3, 28)))
FOLDS: Final = {
    "f1": (date(2021, 7, 7), date(2021, 7, 20)),
    "f2": (date(2021, 7, 21), date(2021, 8, 3)),
    "f3": (date(2021, 8, 4), date(2021, 8, 17)),
    "f4": (date(2021, 8, 18), date(2021, 8, 31)),
    "test": (date(2021, 9, 1), date(2021, 9, 14)),
}
COLUMNS: Final = ["date", "event_id", "dup_size", "is_copy", "copy_kind", "copy_of", "is_suspect", "is_operating", "is_holiday", "daytype", "n_missing", "anomaly", "f1", "f2", "f3", "f4", "test"]
OUT: Final = DATA / "okm_cv_splits_2021.csv"


def active_same(a: FloatArray, b: FloatArray) -> int:
    return int(((a == b) & (a > 40)).sum())


def fold_roles(dates: list[date], events: list[str], start: date, end: date, val_label: str) -> list[str]:
    """Assign fixed windows and purge exact events shared with their targets."""
    window_events = {event for day, event in zip(dates, events, strict=True) if start <= day <= end}
    roles: list[str] = []
    # The gap day is excluded from targets but is input for the first validation day (FR-25).
    for day, event in zip(dates, events, strict=True):
        if start <= day <= end:
            roles.append(val_label)
        elif day == start - timedelta(days=1):
            roles.append("gap")
        elif day < start:
            roles.append("purged" if event in window_events else "train")
        else:
            roles.append("")
    return roles


def build_days(df15: pl.DataFrame) -> pl.DataFrame:
    """Derive copy identities before masking and then assign independent policy flags."""
    df15 = df15.sort("datetime")
    dates: list[date] = df15["datetime"].dt.date().unique(maintain_order=True).to_list()
    power = df15["전력"].to_numpy().astype(np.float64).reshape(-1, 96)
    production = df15["생산량"].to_numpy().reshape(-1, 96).sum(axis=1) / 4
    missing = df15["is_missing"].to_numpy().reshape(-1, 96).sum(axis=1)
    groups: dict[tuple[float, ...], list[int]] = {}
    for index, row in enumerate(power):
        groups.setdefault(tuple(np.nan_to_num(row, nan=-1)), []).append(index)
    events = [""] * len(dates)
    sizes = [0] * len(dates)
    copied = [False] * len(dates)
    kinds = [""] * len(dates)
    sources = [""] * len(dates)
    for number, members in enumerate(groups.values()):
        original = members[0]
        alternatives = [member for member in members if production[member] > 0]
        if (power[original] > 40).sum() >= 48 and production[original] == 0 and alternatives:
            original = alternatives[0]
        for member in members:
            events[member], sizes[member] = f"E{number:03d}", len(members)
            if member != original:
                copied[member], kinds[member], sources[member] = True, "exact", dates[original].strftime("%Y.%m.%d")
    exact_copies = copied.copy()
    for index in range(len(dates)):
        if sizes[index] != 1:
            continue
        matches = ((power == power[index]) & (power[index] > 40)).sum(axis=1)
        matches[index] = -1
        maximum = int(matches.max())
        if maximum >= 20:
            partner = min(np.flatnonzero(matches == maximum), key=lambda candidate: (exact_copies[candidate], dates[candidate]))
            copied[index], kinds[index], sources[index] = True, "edited", dates[partner].strftime("%Y.%m.%d")
    assert tuple(day for day, kind in zip(dates, kinds, strict=True) if kind == "edited") == EDITED
    jan24, feb24 = dates.index(date(2021, 1, 24)), dates.index(date(2021, 2, 24))
    assert kinds[jan24] == "exact" and sources[jan24] == "2021.02.24" and not copied[feb24]
    assert all(day < date(2021, 7, 1) for day, kind in zip(dates, kinds, strict=True) if kind == "edited")
    suspect = np.array([day in SUSPECT for day in dates])
    frame = pl.DataFrame({
        "date": dates, "event_id": events, "dup_size": sizes, "is_copy": copied,
        "copy_kind": kinds, "copy_of": sources, "is_suspect": suspect,
        "is_operating": (production > 0) | suspect,
        "is_holiday": [day in HOLIDAYS_2021 for day in dates],
        "daytype": ["wk" if day.weekday() < 5 else ("sat" if day.weekday() == 5 else "sun") for day in dates],
        "n_missing": missing.astype(np.int64), "anomaly": suspect | (missing > 0),
    })
    return frame.with_columns([
        pl.Series(fold, fold_roles(dates, events, start, end, "test" if fold == "test" else "val"))
        for fold, (start, end) in FOLDS.items()
    ]).select(COLUMNS)


def run() -> pl.DataFrame:
    days = build_days(build_15min())
    days.write_csv(OUT, date_format="%Y.%m.%d")
    return days


if __name__ == "__main__":
    run()
