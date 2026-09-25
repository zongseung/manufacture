from typing import Final, NotRequired, TypedDict

import lightgbm as lgb
import numpy as np

from gmst import backbone as bb
from gmst.contracts import FloatArray, IntArray, Model, Panel, Prediction

LGB_PARAMS: Final = {"seed": 0, "deterministic": True, "force_col_wise": True, "verbose": -1}  # ponytail: 기본 하이퍼파라미터, B1이 f1–f4에서 B0′를 못 이기면 튜닝 (FR-49)
QUANTILES: Final = np.round(np.arange(1, 20) * .05, 2)


class NaiveState(TypedDict):
    C: FloatArray


class SlotState(TypedDict):
    tau: float
    h: int
    protocol: str
    rounds: int
    names: list[str]
    n_rows: int
    train_idx: IntArray
    q: list[lgb.Booster]


class B1State(SlotState):
    C: FloatArray
    mean: lgb.Booster
    Mq: list[lgb.Booster]
    clf: list[lgb.Booster | None]
    inner_state: NotRequired["B1State"]


def b0p_ref(panel: Panel, d: int, limit: int | None = 28) -> tuple[int | None, bool]:
    complete = np.isfinite(panel["Y"][:d]).all(axis=1)
    start = 0 if limit is None else max(0, d - limit)
    for j in range(d - 1, start - 1, -1):
        if complete[j] and panel["op"][j] == panel["op"][d] and panel["dtype"][j] == panel["dtype"][d]:
            return j, False
    for j in range(d - 1, -1, -1):
        if complete[j] and panel["op"][j] == panel["op"][d]:
            return j, True
    return None, True


def b0p_model(limit: int | None = 28) -> Model[NaiveState]:
    from gmst.evaluate import point_pred

    def predict(state: NaiveState, panel: Panel, d: int) -> Prediction:
        ref, _ = b0p_ref(panel, d, limit)
        y = np.full(96, np.nan) if ref is None else panel["Y"][ref].copy()
        return point_pred(y, state["C"])

    return {"name": "B0p", "fit": lambda panel, fold, C: {"C": C}, "predict": predict, "inner_state": lambda state: state}


def b0_model() -> Model[NaiveState]:
    from gmst.evaluate import point_pred

    fallback = b0p_model()

    def predict(state: NaiveState, panel: Panel, d: int) -> Prediction:
        ref = panel["Y"][d - 7] if d >= 7 else np.full(96, np.nan)
        y = np.where(np.isfinite(ref), ref, fallback["predict"](state, panel, d)["y_median"])
        return point_pred(y, state["C"])

    return {"name": "B0", "fit": fallback["fit"], "predict": predict, "inner_state": lambda state: state}


def _slot_data(
    panel: Panel, train_idx: IntArray, protocol: str, backbone: tuple[float, int],
) -> tuple[FloatArray, FloatArray, list[str]]:
    from gmst.features import lgbm_rows

    m = bb.asof_matrix(panel, train_idx, *backbone, protocol)
    X, y, names = lgbm_rows(panel, train_idx, protocol, m)
    keep = np.isfinite(y)
    if not keep.any():
        message = "B1 requires at least one observed training target"
        raise ValueError(message)
    return X[keep], y[keep], names


def _quantile_boosters(data: lgb.Dataset, rounds: int) -> list[lgb.Booster]:
    return [lgb.train({**LGB_PARAMS, "objective": "quantile", "alpha": float(q)}, data, num_boost_round=rounds) for q in QUANTILES]


def fit_b1(
    panel: Panel, train_idx: IntArray, protocol: str, backbone: tuple[float, int], C: FloatArray, rounds: int = 100,
) -> B1State:
    from gmst.features import day_rows, usable_peak

    X, y, names = _slot_data(panel, train_idx, protocol, backbone)
    data = lgb.Dataset(X, y, free_raw_data=False)
    mean = lgb.train({**LGB_PARAMS, "objective": "regression"}, data, num_boost_round=rounds)
    quantiles = _quantile_boosters(data, rounds)
    days = train_idx[usable_peak(panel)[train_idx]]
    if not len(days):
        message = "B1 requires at least one usable daily peak in training"
        raise ValueError(message)
    m = bb.asof_matrix(panel, days, *backbone, protocol)
    DX, _ = day_rows(panel, days, protocol, m)
    maxima = np.nanmax(panel["Y"][days], axis=1)
    Mq = _quantile_boosters(lgb.Dataset(DX, maxima, free_raw_data=False), rounds)
    classifiers: list[lgb.Booster | None] = []
    for j, threshold in enumerate(C):
        labels = (maxima > threshold).astype(float)
        if labels.min() == labels.max():
            print(f"B1: C{(50, 75, 90)[j]} single-class in training; risk = 1 - F_M(C)")
            classifiers.append(None)
        else:
            classifiers.append(lgb.train({**LGB_PARAMS, "objective": "binary"}, lgb.Dataset(DX, labels), num_boost_round=rounds))
    return {"tau": backbone[0], "h": backbone[1], "protocol": protocol, "rounds": rounds,
            "C": C, "names": names, "n_rows": len(y), "train_idx": train_idx.copy(), "mean": mean,
            "q": quantiles, "Mq": Mq, "clf": classifiers}


def predict_b1(state: B1State, panel: Panel, d: int) -> Prediction:
    from gmst.features import day_rows, lgbm_rows

    days = np.array([d])
    m = bb.asof_matrix(panel, days, state["tau"], state["h"], state["protocol"])
    X, _, _ = lgbm_rows(panel, days, state["protocol"], m)
    DX, _ = day_rows(panel, days, state["protocol"], m)
    q = np.sort(np.array([booster.predict(X) for booster in state["q"]]), axis=0)
    Mq = np.sort(np.array([booster.predict(DX) for booster in state["Mq"]]).reshape(19))
    risk = np.array([float(np.asarray(booster.predict(DX), dtype=float)[0]) if booster is not None else
                     1 - np.interp(state["C"][j], Mq, QUANTILES, left=0., right=1.)
                     for j, booster in enumerate(state["clf"])])
    return {"y_mean": np.asarray(state["mean"].predict(X), dtype=float), "y_median": q[9], "q": q, "paths": None,
            "M_hat_median": float(Mq[9]), "M_hat_mean": float(Mq.mean()), "peak_time_mode": int(np.argmax(q[9])),
            "risk_raw": risk, "Mq": Mq}


def b1_model(tau: float, half_life: int, protocol: str = "A+", rounds: int = 100) -> Model[B1State]:
    from gmst.features import inner_idx, role_idx

    def fit(panel: Panel, fold: str, C: FloatArray) -> B1State:
        tr = role_idx(panel, fold, "train")
        cal = inner_idx(panel, fold)
        before = tr[tr < cal[0]] if cal.size else tr
        inner = fit_b1(panel, before, protocol, (tau, half_life), C, rounds)
        state = fit_b1(panel, tr, protocol, (tau, half_life), C, rounds)
        state["inner_state"] = inner
        return state

    return {"name": "B1", "fit": fit, "predict": predict_b1, "inner_state": lambda state: state.get("inner_state")}
