"""Sealed, masked model inputs and chronological internal split policy."""
from typing import Final

import numpy as np

from gmst.contracts import BoolArray, IntArray, Panel
from gmst.lgbm_features import (
    DAY_FEATURES,
    PROTO_DAY,
    PROTO_SLOT,
    PROTOCOLS,
    SLOT_FEATURES,
    cal_flags,
    day_rows,
    lgbm_rows,
    season,
)
from gmst.preprocess import build_15min
from gmst.splits import build_days

__all__ = [
    "CHANNELS", "DAY_FEATURES", "PROTOCOLS", "PROTO_DAY", "PROTO_SLOT", "SLOT_FEATURES",
    "cal_flags", "day_rows", "inner_idx", "internal_split", "lgbm_rows", "load_panel", "role_idx",
    "season", "tuning_idx", "usable_peak",
]

CHANNELS: Final = ("생산량", "기온", "풍속", "습도", "강수량_증분")


def load_panel(include_copies: bool = False, include_suspect: bool = False, unseal: bool = False) -> Panel:
    """Build current measurement policy in memory and seal all September observations."""
    frame = build_15min()
    days = build_days(frame)
    targets = frame["전력"].to_numpy().astype(np.float64).reshape(-1, 96)
    inputs = {name: frame[name].to_numpy().astype(np.float64).reshape(-1, 96) for name in CHANNELS}
    missing = frame["is_missing"].to_numpy().reshape(-1, 96)
    targets[missing] = np.nan
    if not include_copies:
        targets[days["is_copy"].to_numpy()] = np.nan
    if not include_suspect:
        suspect = days["is_suspect"].to_numpy()
        targets[suspect] = np.nan
        inputs["생산량"][suspect] = np.nan
    if not unseal:
        sealed = (days["test"] == "test").to_numpy()
        targets[sealed] = np.nan
        for values in inputs.values():
            values[sealed] = np.nan
    dates = days["date"].to_list()
    return {
        "dates": dates, "Y": targets, "X": inputs, "is_missing": missing, "days": days,
        "op": days["is_operating"].to_numpy().astype(np.int8),
        "hol": days["is_holiday"].to_numpy().astype(np.int8),
        "dtype": np.array([0 if day.weekday() < 5 else day.weekday() - 4 for day in dates], dtype=np.int8),
        "dow": np.array([day.weekday() for day in dates], dtype=np.int8),
        "month": np.array([day.month for day in dates], dtype=np.int8),
    }


def role_idx(panel: Panel, fold: str, role: str) -> IntArray:
    return np.flatnonzero((panel["days"][fold] == role).to_numpy()).astype(np.int64)


def usable_peak(panel: Panel) -> BoolArray:
    """Only observed days with at most four outage slots support day-peak metrics."""
    return np.isfinite(panel["Y"]).any(axis=1) & (panel["is_missing"].sum(axis=1) <= 4)


def internal_split(panel: Panel, fold: str, train_idx: IntArray | None = None) -> tuple[IntArray, IntArray, IntArray]:
    """Reserve disjoint tuning and calibration days after at least five fit days."""
    train = role_idx(panel, fold, "train")
    if train_idx is not None:
        train = np.intersect1d(train, train_idx)
    usable = train[np.isfinite(panel["Y"][train]).any(axis=1)]
    n = len(usable)
    cal_count = min(7, max(0, n - 10))
    tune_count = min(7, max(0, n - cal_count - 5))
    cal = usable[n - cal_count:]
    tune = usable[n - cal_count - tune_count:n - cal_count]
    held = usable[n - cal_count - tune_count:]
    fit = train[train < held[0]] if len(held) else train
    return fit, tune, cal


def tuning_idx(panel: Panel, fold: str, train_idx: IntArray | None = None) -> IntArray:
    return internal_split(panel, fold, train_idx)[1]


def inner_idx(panel: Panel, fold: str, train_idx: IntArray | None = None) -> IntArray:
    return internal_split(panel, fold, train_idx)[2]
