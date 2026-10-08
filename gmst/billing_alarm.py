"""Month-max (billing demand) alarm backtest for chapter 4 (document/BILLING_alarm_preregistration.md).

A day matters for the demand charge only if its tariff-band peak exceeds the baseline B_d known at 00:00:
max(month-to-date band peak, band peaks of past ratchet months within a year). C-BAT paths give P(M_d > B_d) and the
expected excess charge; alarms are scored against observed band peaks. Not a counterfactual simulation: the effect of
acting on an alarm is not estimated.
"""
import argparse
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl

from gmst import ROOT, features
from gmst import evaluate as ev
from gmst.benchmark_models import b0_kind_ref
from gmst.conditional_bat import ConditionalConfig, fit_conditional, raw_paths
from gmst.contracts import BoolArray, FloatArray, Panel
from gmst.scenario import RATCHET_MONTHS, TARIFF, _timestamps, band, tariff_holidays

BASE: Final = TARIFF["II"]["base"]  # ₩ per kW-month, tariff option II
RULES: Final = ("cost10000", "cost30000", "cost100000", "p50", "cbat_point", "b0_kind", "always")


def band_mask(day: date, holidays: set[date]) -> BoolArray:
    return np.array([band(t, day in holidays, False) >= 1 for t in _timestamps(day)])


def band_peak(panel: Panel, d: int, holidays: set[date]) -> float:
    m = band_mask(panel["dates"][d], holidays) & np.isfinite(panel["Y"][d])
    return float(panel["Y"][d, m].max()) if m.any() else np.nan


def baseline(panel: Panel, d: int, holidays: set[date]) -> float:
    """B_d from days before d only: month-to-date band peak and past-year ratchet-month band peaks."""
    day, dates = panel["dates"][d], panel["dates"]
    peaks = [band_peak(panel, j, holidays) for j in range(d)
             if (dates[j].year, dates[j].month) == (day.year, day.month)
             or (dates[j].month in RATCHET_MONTHS and dates[j] > day - timedelta(days=365))]
    return float(np.nanmax(peaks)) if np.isfinite(peaks).any() else 0.


def risk(maxima: FloatArray, b: float) -> tuple[float, float]:
    """P(M > B_d) and expected excess demand charge (₩, current month) over path band maxima."""
    return float((maxima > b).mean()), float(BASE * np.maximum(maxima - b, 0).mean())


def alarm(cost: float, p: float, rule: str) -> bool:
    return p >= .5 if rule == "p50" else cost >= float(rule.removeprefix("cost"))


def job(args: tuple[str, int, int, int]) -> list[dict]:
    fold, seed, n_iter, burn = args
    panel, holidays = features.load_panel(), tariff_holidays()
    train = features.role_idx(panel, fold, "train")
    C, _ = ev.thresholds(panel, train)
    state = fit_conditional(panel, train, C, ConditionalConfig(n_iter=n_iter, burn=burn, seed=seed))
    rows = []
    for d in features.role_idx(panel, fold, "val"):
        mask = band_mask(panel["dates"][d], holidays)
        observed = band_peak(panel, int(d), holidays)
        if not mask.any() or np.isnan(observed):
            continue
        hidden: Panel = {**panel, "Y": panel["Y"].copy()}
        hidden["Y"][d:] = np.nan
        b = baseline(hidden, int(d), holidays)
        maxima = np.maximum(raw_paths(state, hidden, int(d)), 0)[:, mask].max(axis=1)
        p, cost = risk(maxima, b)
        ref, _ = b0_kind_ref(hidden, int(d))
        rows.append({"fold": fold, "seed": seed, "date": panel["dates"][d], "B": b, "observed": observed,
                     "p": p, "cost": cost, "cbat_median": float(np.median(maxima)),
                     "b0_kind": float(panel["Y"][ref, mask].max()) if ref is not None else np.nan})
    return rows


def backtest(days: pl.DataFrame) -> pl.DataFrame:
    """Per rule: renewals caught, alarms, false alarms, and excess charge at stake (current month, perfect action)."""
    event = (days["observed"] > days["B"]).to_numpy()
    excess = BASE * np.maximum(days["observed"] - days["B"], 0).to_numpy()
    out = []
    for rule in RULES:
        if rule == "always":
            fired = np.ones(len(days), bool)
        elif rule == "cbat_point":
            fired = (days["cbat_median"] > days["B"]).to_numpy()
        elif rule == "b0_kind":
            fired = np.nan_to_num(days["b0_kind"].to_numpy(), nan=-np.inf) > days["B"].to_numpy()
        else:
            fired = np.array([alarm(c, p, rule) for c, p in days.select("cost", "p").iter_rows()])
        out.append({"rule": rule, "days": len(days), "renewals": int(event.sum()), "caught": int((fired & event).sum()),
                    "alarms": int(fired.sum()), "false_alarms": int((fired & ~event).sum()),
                    "excess_won_all": float(excess.sum()), "excess_won_caught": float(excess[fired].sum())})
    return pl.DataFrame(out)


def main() -> None:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS"):
        os.environ[name] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results_v3/billing_alarm")
    parser.add_argument("--n-iter", type=int, default=6000)
    parser.add_argument("--burn", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    jobs = [(f, s, args.n_iter, args.burn) for f in ev.FOLDS_CV for s in (0, 1, 2)]
    with ProcessPoolExecutor(min(args.workers, len(jobs)), mp.get_context("spawn")) as pool:
        rows = [r for part in pool.map(job, jobs) for r in part]
    by_seed = pl.DataFrame(rows)
    days = (by_seed.group_by("fold", "date").agg(pl.col("B", "observed", "b0_kind").first(),
                                                 pl.col("p", "cost", "cbat_median").mean()).sort("date"))
    args.out.mkdir(parents=True, exist_ok=True)
    by_seed.write_csv(args.out / "days_by_seed.csv")
    days.write_csv(args.out / "days.csv")
    summary = backtest(days)
    summary.write_csv(args.out / "summary.csv")
    print(days.filter(pl.col("observed") > pl.col("B")), summary, sep="\n")


if __name__ == "__main__":
    main()
