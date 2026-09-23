"""Weighted cyclic RW2 profiles and fold-local tuning."""
from typing import Final, NotRequired, TypedDict

import numpy as np
import polars as pl

from gmst.contracts import (
    BoolArray,
    FlagArray,
    FloatArray,
    IntArray,
    Model,
    Panel,
    Prediction,
)

C2: Final = np.roll(np.eye(96), 1, axis=1) - 2 * np.eye(96) + np.roll(np.eye(96), -1, axis=1)
PENALTY: Final = C2.T @ C2
TAU_GRID: Final = (0.1, 1.0, 10.0, 100.0, 1000.0)
H_GRID: Final = (30, 60, 120)


class BackboneState(TypedDict):
    m: FloatArray
    C: FloatArray
    train_idx: IntArray
    inner_state: NotRequired["BackboneState"]


def fit_backbone(
    Y: FloatArray, op: FlagArray, daytype: FlagArray, train_idx: IntArray,
    tau: float, half_life: int | None = 60, min_days: int = 5,
) -> tuple[FloatArray, list[tuple[int, int]]]:
    """Solve the six weighted systems; sparse cells share their operating parent."""
    profiles = np.full((2, 3, 96), np.nan)
    fallback: list[tuple[int, int]] = []
    if train_idx.size == 0:
        return profiles, [(o, t) for o in range(2) for t in range(3)]
    weights = np.ones(len(train_idx)) if half_life is None else 2. ** (-(train_idx.max() - train_idx) / half_life)
    values = Y[train_idx]
    observed = np.isfinite(values)

    def solve(rows: BoolArray) -> FloatArray:
        w = weights[rows, None] * observed[rows]
        count = w.sum(axis=0)
        rhs = (w * np.nan_to_num(values[rows])).sum(axis=0)
        if not np.any(count):
            return np.full(96, np.nan)
        if tau == 0:
            return np.divide(rhs, count, out=np.full(96, np.nan), where=count > 0)
        return np.linalg.solve(np.diag(count) + tau * PENALTY, rhs)

    usable = observed.any(axis=1)
    for o in range(2):
        parent = op[train_idx] == o
        parent_profile = solve(parent)
        for t in range(3):
            rows = parent & (daytype[train_idx] == t)
            if int((rows & usable).sum()) < min_days:
                profiles[o, t] = parent_profile
                fallback.append((o, t))
            else:
                profiles[o, t] = solve(rows)
    return profiles, fallback


def predict_backbone(profiles: FloatArray, op: FlagArray, daytype: FlagArray) -> FloatArray:
    return profiles[op, daytype]


def backbone_asof(
    Y: FloatArray, op: FlagArray, daytype: FlagArray, d: int,
    tau: float, half_life: int, min_days: int = 5,
) -> FloatArray:
    """The d-2 cutoff matches the training gap for every feature row."""
    if d < 2:
        return np.full(96, np.nan)
    profiles, _ = fit_backbone(Y, op, daytype, np.arange(d - 1), tau, half_life, min_days)
    return profiles[op[d], daytype[d]]


def asof_matrix(panel: Panel, day_idx: IntArray, tau: float, half_life: int, protocol: str = "A+") -> FloatArray:
    from gmst.features import cal_flags

    op, _ = cal_flags(panel, protocol)
    result = np.full_like(panel["Y"], np.nan)
    for d in day_idx:
        result[d] = backbone_asof(panel["Y"], op, panel["dtype"], int(d), tau, half_life)
    return result


def fold_backbone(panel: Panel, train_idx: IntArray, tau: float, half_life: int, protocol: str = "A+") -> FloatArray:
    from gmst.features import cal_flags

    op, _ = cal_flags(panel, protocol)
    profiles, _ = fit_backbone(panel["Y"], op, panel["dtype"], train_idx, tau, half_life)
    return predict_backbone(profiles, op, panel["dtype"])


def select_tau(panel: Panel, fold: str, train_idx: IntArray | None = None) -> tuple[float, int, pl.DataFrame]:
    from gmst.features import internal_split

    fit_idx, tune_idx, _ = internal_split(panel, fold, train_idx)
    rows: list[tuple[str, float, int, float, int, str]] = []
    for tau in TAU_GRID:
        for h in H_GRID:
            pred = fold_backbone(panel, fit_idx, tau, h)[tune_idx]
            y = panel["Y"][tune_idx]
            mask = np.isfinite(y) & np.isfinite(pred)
            n = int(mask.sum())
            score = float(np.mean(np.abs(y[mask] - pred[mask]))) if n else float("nan")
            rows.append((fold, tau, h, score, len(tune_idx), "tuned" if n else "default_empty"))
    table = pl.DataFrame(rows, schema=["fold", "tau", "half_life", "mae_tune", "n_tune", "status"], orient="row")
    finite = [(i, row[3]) for i, row in enumerate(rows) if np.isfinite(row[3])]
    if not finite:
        return 10.0, 60, table
    best = rows[min(finite, key=lambda item: item[1])[0]]
    tau, h = best[1], best[2]
    if tau in (TAU_GRID[0], TAU_GRID[-1]) or h in (H_GRID[0], H_GRID[-1]):
        print(f"backbone grid edge: tau={tau} h={h}")
    return tau, h, table


def bb_model(tau: float, half_life: int, protocol: str = "A+") -> Model[BackboneState]:
    from gmst.evaluate import point_pred
    from gmst.features import inner_idx, role_idx

    def fit(panel: Panel, fold: str, C: FloatArray) -> BackboneState:
        tr = role_idx(panel, fold, "train")
        cal = inner_idx(panel, fold)
        before = tr[tr < cal[0]] if cal.size else tr
        inner: BackboneState = {"m": fold_backbone(panel, before, tau, half_life, protocol), "C": C, "train_idx": before}
        return {"m": fold_backbone(panel, tr, tau, half_life, protocol), "C": C, "train_idx": tr, "inner_state": inner}

    def predict(state: BackboneState, panel: Panel, d: int) -> Prediction:
        return point_pred(state["m"][d], state["C"])

    return {"name": "BB", "fit": fit, "predict": predict, "inner_state": lambda state: state.get("inner_state")}
