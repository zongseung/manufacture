from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl

from gmst import evaluate as ev
from gmst import features
from gmst.contracts import FloatArray, Model, Panel


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    sigma: float = 0.0
    shift: int = 0


SCENARIOS: Final = (
    Scenario("clean"), Scenario("volume_05pct", sigma=0.05),
    Scenario("volume_10pct", sigma=0.1), Scenario("volume_20pct", sigma=0.2),
    Scenario("timing_early_1h", shift=-1), Scenario("timing_late_1h", shift=1),
)


def perturb_plan(q: FloatArray, scenario: Scenario, rng: np.random.Generator) -> FloatArray:
    noisy = q.copy()
    if scenario.sigma:
        noisy *= np.exp(scenario.sigma * rng.standard_normal(24) - scenario.sigma ** 2 / 2)
    if scenario.shift:
        moved = np.zeros(24)
        np.add.at(moved, np.clip(np.arange(24) + scenario.shift, 0, 23), noisy)
        noisy = moved
    return noisy


def issued_panel(panel: Panel, d: int, scenario: Scenario, repeat: int = 0) -> Panel:
    day_seed = panel["dates"][d].toordinal()
    rng = np.random.default_rng(np.random.SeedSequence([day_seed, repeat, 20260925]))
    production = panel["X"]["생산량"].copy()
    production[d] = np.repeat(perturb_plan(production[d, ::4], scenario, rng), 4)
    return {**panel, "X": {**panel["X"], "생산량": production}}


def evaluate_plan_errors[S](model: Model[S], state: S, panel: Panel, fold: str, repeats: int) -> pl.DataFrame:
    rows: list[dict[str, ev.Cell]] = []
    for d_raw in features.role_idx(panel, fold, "val"):
        d = int(d_raw)
        y = panel["Y"][d]
        if not np.isfinite(y).any():
            continue
        for scenario in SCENARIOS:
            for repeat in range(repeats if scenario.sigma else 1):
                issued = issued_panel(panel, d, scenario, repeat)
                pred = model["predict"](state, issued, d)
                mae, n = ev.mae(y, pred["y_median"])
                rmse, _ = ev.rmse(y, pred["y_mean"])
                q = pred["q"]
                crps = ev.crps(y, q.T)[0] if q is not None else float("nan")
                paths = pred["paths"]
                peak = float(np.median(ev.obs_max(paths, np.isfinite(y)))) if paths is not None else pred["M_hat_median"]
                usable = bool(features.usable_peak(panel)[d])
                original = panel["X"]["생산량"][d, ::4]
                noisy = issued["X"]["생산량"][d, ::4]
                rows.append({
                    "model": model["name"], "fold": fold, "date": str(panel["dates"][d]),
                    "scenario": scenario.name, "repeat": repeat, "sigma": scenario.sigma,
                    "shift": scenario.shift, "operating": bool(panel["op"][d]), "n_slots": n,
                    "ae_sum": mae * n, "se_sum": rmse ** 2 * n, "crps_sum": crps * n,
                    "peak_ae": abs(peak - float(np.nanmax(y))) if usable else float("nan"),
                    "usable_peak": usable, "mae": mae,
                    "plan_l1": float(np.abs(noisy - original).sum()),
                    "changed_on_hours": int(np.count_nonzero((noisy > 0) != (original > 0))),
                })
    return pl.DataFrame(rows)


def sensitivity_summary(rows: pl.DataFrame) -> pl.DataFrame:
    daily = rows.group_by("model", "fold", "date", "scenario", "operating").agg(
        pl.col("n_slots").first(), pl.col("usable_peak").first(), pl.len().alias("n_repeats"),
        *(pl.col(c).mean() for c in ("ae_sum", "se_sum", "crps_sum", "peak_ae", "plan_l1", "changed_on_hours")),
    )
    strata = pl.concat([
        daily.with_columns(pl.lit("all").alias("stratum")),
        daily.filter(pl.col("operating")).with_columns(pl.lit("op").alias("stratum")),
    ])
    strata = pl.concat([strata, strata.with_columns(pl.lit("pooled").alias("fold"))])
    summary = strata.group_by("model", "fold", "stratum", "scenario").agg(
        pl.len().alias("n_days"), pl.col("n_slots").sum(), pl.col("usable_peak").sum().alias("n_peak_days"),
        (pl.col("ae_sum").sum() / pl.col("n_slots").sum()).alias("mae"),
        (pl.col("se_sum").sum() / pl.col("n_slots").sum()).sqrt().alias("rmse"),
        (pl.col("crps_sum").sum() / pl.col("n_slots").sum()).alias("crps"),
        pl.col("peak_ae").fill_nan(None).mean().alias("peak_mae"),
        pl.col("plan_l1").mean(), pl.col("changed_on_hours").mean(), pl.col("n_repeats").first(),
    )
    keys = ["model", "fold", "stratum"]
    control = summary.filter(pl.col("scenario") == "clean").select(*keys, pl.col("mae").alias("mae_clean"))
    return summary.join(control, on=keys, validate="m:1").with_columns(
        (pl.col("mae") - pl.col("mae_clean")).alias("delta_mae"),
        (100 * (pl.col("mae") / pl.col("mae_clean") - 1)).alias("delta_mae_pct"),
    ).sort("fold", "stratum", "model", "scenario")
