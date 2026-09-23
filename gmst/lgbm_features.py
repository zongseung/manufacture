from typing import Final

import numpy as np

from gmst.contracts import FlagArray, FloatArray, IntArray, Panel

PROTOCOLS: Final = {"A": set(), "A+": {"op", "hol"}, "A+W*": {"op", "hol", "wx_obs"}, "B": {"op", "hol", "prod"}}

SLOT_FEATURES: Final = [
    "q", "q_mod4", "sin1", "cos1", "sin2", "cos2", "sin3", "cos3", "dow", "daytype", "month", "season",
    "hol", "op", "y_lag1", "y_lag2", "y_lag7", "prev_last4_mean", "prev_mean", "prev_max", "prev_min",
    "recent_same_mean", "op_lag1", "op_lag7", "backbone", "prev_temp", "prev_hum", "prev_wind", "prev_rain", "prev_prod",
]
DAY_FEATURES: Final = [
    "prev_mean", "prev_max", "prev_min", "y_lag7_mean", "y_lag7_max", "recent_same_max", "backbone_max",
    "dow", "daytype", "month", "hol", "op", "op_lag1",
]
PROTO_SLOT: Final = {"B": ["prod_h", "prod_on"], "A+W*": ["ob_temp", "ob_rain", "ob_wind", "ob_hum"]}
PROTO_DAY: Final = {"B": ["prod_sum", "prod_hours"], "A+W*": ["ob_temp_mean", "ob_rain_sum", "ob_wind_mean", "ob_hum_mean"]}


def season(month: int) -> int:
    return month % 12 // 3


def cal_flags(panel: Panel, protocol: str) -> tuple[FlagArray, FlagArray]:
    permitted = PROTOCOLS[protocol]
    return (panel["op"] if "op" in permitted else np.ones_like(panel["op"]),
            panel["hol"] if "hol" in permitted else np.zeros_like(panel["hol"]))


def _stats(values: FloatArray) -> tuple[float, float, float]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return float("nan"), float("nan"), float("nan")
    return float(finite.mean()), float(finite.max()), float(finite.min())


def lgbm_rows(
    panel: Panel, day_idx: IntArray, protocol: str, backbone: FloatArray,
) -> tuple[FloatArray, FloatArray, list[str]]:
    op, hol = cal_flags(panel, protocol)
    permitted = PROTOCOLS[protocol]
    names = SLOT_FEATURES + PROTO_SLOT.get(protocol, [])
    result = np.full((len(day_idx), 96, len(names)), np.nan)
    q = np.arange(96, dtype=float)
    waves = [f(2 * np.pi * r * q / 96) for r in (1, 2, 3) for f in (np.sin, np.cos)]
    empty = np.full(96, np.nan)
    for row, d in enumerate(day_idx):
        lag1 = panel["Y"][d - 1] if d >= 1 else empty
        lag2 = panel["Y"][d - 2] if d >= 2 else empty
        lag7 = panel["Y"][d - 7] if d >= 7 else empty
        prev_mean, prev_max, prev_min = _stats(lag1)
        candidates = np.arange(max(0, d - 7), d)
        same = candidates[(panel["dtype"][candidates] == panel["dtype"][d]) & (op[candidates] == op[d])]
        recent = panel["Y"][same]
        count = np.isfinite(recent).sum(axis=0)
        same_mean = np.divide(np.nansum(recent, axis=0), count, out=empty.copy(), where=count > 0)
        weather = [panel["X"][key][d - 1] if d >= 1 else empty for key in ("기온", "습도", "풍속", "강수량_증분", "생산량")]
        columns = [
            q, q % 4, *waves, panel["dow"][d], panel["dtype"][d], panel["month"][d], season(int(panel["month"][d])),
            hol[d], op[d], lag1, lag2, lag7, _stats(lag1[-4:])[0], prev_mean, prev_max, prev_min,
            same_mean, op[d - 1] if d >= 1 else np.nan, op[d - 7] if d >= 7 else np.nan, backbone[d],
            *(values.mean() for values in weather[:3]), *(values.sum() / 4 for values in weather[3:]),
        ]
        if "prod" in permitted:
            prod = panel["X"]["생산량"][d]
            columns.extend([prod, np.where(np.isfinite(prod), prod > 0, np.nan)])
        if "wx_obs" in permitted:
            columns.extend(panel["X"][key][d] for key in ("기온", "강수량_증분", "풍속", "습도"))
        for j, column in enumerate(columns):
            result[row, :, j] = column
    return result.reshape(-1, len(names)), panel["Y"][day_idx].reshape(-1), names


def day_rows(panel: Panel, day_idx: IntArray, protocol: str, backbone: FloatArray) -> tuple[FloatArray, list[str]]:
    permitted = PROTOCOLS[protocol]
    slots, _, slot_names = lgbm_rows(panel, day_idx, protocol, backbone)
    slots = slots.reshape(len(day_idx), 96, len(slot_names))
    names = DAY_FEATURES + PROTO_DAY.get(protocol, [])
    result = np.full((len(day_idx), len(names)), np.nan)
    for row, d in enumerate(day_idx):
        lag7 = panel["Y"][d - 7] if d >= 7 else np.full(96, np.nan)
        recent = slots[row, :, slot_names.index("recent_same_mean")]
        values = [slots[row, 0, slot_names.index(key)] for key in DAY_FEATURES[:3]]
        values += [*_stats(lag7)[:2], _stats(recent)[1], _stats(backbone[d])[1]]
        values += [slots[row, 0, slot_names.index(key)] for key in DAY_FEATURES[7:]]
        if "prod" in permitted:
            prod = panel["X"]["생산량"][d]
            values += [prod.sum() / 4, float((prod > 0).sum() / 4) if np.isfinite(prod).all() else np.nan]
        if "wx_obs" in permitted:
            values += [panel["X"]["기온"][d].mean(), panel["X"]["강수량_증분"][d].sum() / 4,
                       panel["X"]["풍속"][d].mean(), panel["X"]["습도"][d].mean()]
        result[row] = values
    return result, names
