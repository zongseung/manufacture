"""Stronger development baselines with the original protocol-B feature budget."""
from dataclasses import dataclass
from itertools import product
from typing import Literal

import lightgbm as lgb
import numpy as np
import polars as pl

from gmst import backbone as bb
from gmst import baselines as bl
from gmst.bat import kinds
from gmst.contracts import FloatArray, IntArray, Model, Panel, Prediction
from gmst.evaluate import FOLDS_CV, point_pred
from gmst.features import day_rows, internal_split, role_idx, usable_peak

type ReferenceStatus = Literal["same_kind_daytype_28d", "same_kind", "latest_complete", "no_complete_history"]


@dataclass(frozen=True, slots=True)
class Candidate:
    num_leaves: int = 31
    min_data_in_leaf: int = 20
    learning_rate: float = .1
    rounds: int = 100

    def params(self) -> bl.LGBOverrides:
        return {"num_leaves": self.num_leaves, "min_data_in_leaf": self.min_data_in_leaf,
                "learning_rate": self.learning_rate}


@dataclass(frozen=True, slots=True)
class TunedState:
    model: bl.B1State
    audit: pl.DataFrame
    inner: "TunedState | None" = None


def b0_kind_ref(panel: Panel, d: int) -> tuple[int | None, ReferenceStatus]:
    """Complete past references: same type/daytype ≤28 days, same type, then latest any type."""
    kind = kinds(panel, np.arange(d + 1))
    complete = np.flatnonzero(np.isfinite(panel["Y"][:d]).all(axis=1))
    same = complete[kind[complete] == kind[d]]
    recent = same[(same >= d - 28) & (panel["dtype"][same] == panel["dtype"][d])]
    if recent.size:
        return int(recent[-1]), "same_kind_daytype_28d"
    if same.size:
        return int(same[-1]), "same_kind"
    if complete.size:
        return int(complete[-1]), "latest_complete"
    return None, "no_complete_history"


def b0_kind_model() -> Model[bl.NaiveState]:
    def predict(state: bl.NaiveState, panel: Panel, d: int) -> Prediction:
        ref, _ = b0_kind_ref(panel, d)
        y = np.full(96, np.nan) if ref is None else panel["Y"][ref].copy()
        return point_pred(y, state["C"])

    return {"name": "B0_kind", "fit": lambda panel, fold, C: {"C": C},
            "predict": predict, "inner_state": lambda state: state}


def candidate_grid(quick: bool = False) -> tuple[Candidate, ...]:
    """Full 37-candidate grid; quick mode keeps the legacy default plus one shallow alternative."""
    if quick:
        return Candidate(), Candidate(7, 30, .03, 100)
    return (Candidate(), *(Candidate(*values) for values in product((7, 15, 31), (10, 30, 60), (.03, .1), (100, 300))))


def daily_candidate_grid(quick: bool = False) -> tuple[Candidate, ...]:
    """Daily sample capacity is tuned independently: legacy default plus 36 smaller-leaf candidates."""
    if quick:
        return Candidate(), Candidate(3, 5, .03, 100)
    return (Candidate(), *(Candidate(*values) for values in product((3, 7, 15), (2, 5, 10), (.03, .1), (100, 300))))


def _peak_data(panel: Panel, days: IntArray, backbone: tuple[float, int]) -> tuple[FloatArray, FloatArray]:
    m = bb.asof_matrix(panel, days, *backbone, "B")
    X, _ = day_rows(panel, days, "B", m)
    return X, np.nanmax(panel["Y"][days], axis=1)


def _tune(grid: tuple[Candidate, ...], fit_data: tuple[FloatArray, FloatArray],
          tune_data: tuple[FloatArray, FloatArray]) -> tuple[Candidate, pl.DataFrame]:
    scores = np.full(len(grid), np.nan)
    X, y = fit_data
    TX, ty = tune_data
    if y.size and ty.size:
        data = lgb.Dataset(X, y, free_raw_data=False, params={"feature_pre_filter": False})
        for i, candidate in enumerate(grid):
            booster = lgb.train({**bl.LGB_PARAMS, **candidate.params(), "feature_pre_filter": False,
                                 "objective": "quantile", "alpha": .5}, data, num_boost_round=candidate.rounds)
            scores[i] = np.mean(np.abs(ty - booster.predict(TX)))
    best = int(np.nanargmin(scores)) if np.isfinite(scores).any() else 0
    table = pl.DataFrame({
        "candidate": np.arange(len(grid)), "num_leaves": [c.num_leaves for c in grid],
        "min_data_in_leaf": [c.min_data_in_leaf for c in grid],
        "learning_rate": [c.learning_rate for c in grid], "rounds": [c.rounds for c in grid],
        "tune_mae": scores, "selected": np.arange(len(grid)) == best,
    }).with_columns(pl.lit(len(y)).alias("n_fit_rows"), pl.lit(len(ty)).alias("n_tune_rows"),
                   pl.lit("tuned" if np.isfinite(scores).any() else "default_empty").alias("status"))
    return grid[best], table


def tuning_table(state: TunedState) -> pl.DataFrame:
    """Candidate MAE, selection, and split audit for CSV export; shared by outer and inner state."""
    return state.audit.clone()


def tuned_model(tau: float, half_life: int, quick: bool = False) -> Model[TunedState]:
    """Select on fit/tune only; refit all B1 heads using unchanged protocol B (no BAT full-plan features)."""
    def fit(panel: Panel, fold: str, C: FloatArray) -> TunedState:
        if fold not in FOLDS_CV:
            message = "M2_tuned only supports development folds f1–f4"
            raise ValueError(message)
        fit_idx, tune_idx, cal_idx = internal_split(panel, fold)
        empty = (np.empty((0, 0)), np.empty(0))
        fit_data = tune_data = empty
        if tune_idx.size:
            X, y, _ = bl._slot_data(panel, fit_idx, "B", (tau, half_life))
            TX, ty, _ = bl._slot_data(panel, tune_idx, "B", (tau, half_life))
            fit_data, tune_data = (X, y), (TX, ty)
        selected, slot_audit = _tune(candidate_grid(quick), fit_data, tune_data)
        peak_fit, peak_tune = (idx[usable_peak(panel)[idx]] for idx in (fit_idx, tune_idx))
        peak, peak_audit = _tune(daily_candidate_grid(quick),
                                 _peak_data(panel, peak_fit, (tau, half_life)) if peak_fit.size else empty,
                                 _peak_data(panel, peak_tune, (tau, half_life)) if peak_tune.size else empty)
        audit = pl.concat([
            slot_audit.with_columns(pl.lit("slot").alias("target"), pl.lit(len(fit_idx)).alias("n_fit_days"),
                                    pl.lit(len(tune_idx)).alias("n_tune_days")),
            peak_audit.with_columns(pl.lit("peak").alias("target"), pl.lit(len(peak_fit)).alias("n_fit_days"),
                                    pl.lit(len(peak_tune)).alias("n_tune_days")),
        ]).with_columns(
            pl.lit(fold).alias("fold"),
            pl.lit("B").alias("protocol"), pl.lit(tau).alias("tau"), pl.lit(half_life).alias("half_life"),
            pl.lit(quick).alias("quick"), pl.lit(len(cal_idx)).alias("n_cal_days"),
            pl.lit(str(panel["dates"][fit_idx[-1]]) if fit_idx.size else None).alias("fit_end"),
            pl.lit(str(panel["dates"][tune_idx[0]]) if tune_idx.size else None).alias("tune_start"),
            pl.lit(str(panel["dates"][tune_idx[-1]]) if tune_idx.size else None).alias("tune_end"),
            pl.lit(str(panel["dates"][cal_idx[0]]) if cal_idx.size else None).alias("cal_start"),
            pl.lit(True).alias("exploratory_adaptive_development"),
        )
        train = role_idx(panel, fold, "train")
        before_cal = train[train < cal_idx[0]] if cal_idx.size else train
        inner = bl.fit_b1(panel, before_cal, "B", (tau, half_life), C, selected.rounds, selected.params(),
                          day_params=peak.params(), day_rounds=peak.rounds)
        outer = bl.fit_b1(panel, train, "B", (tau, half_life), C, selected.rounds, selected.params(),
                          day_params=peak.params(), day_rounds=peak.rounds)
        return TunedState(outer, audit, TunedState(inner, audit))

    return {"name": "M2_tuned", "fit": fit, "predict": lambda state, panel, d: bl.predict_b1(state.model, panel, d),
            "inner_state": lambda state: state.inner}
