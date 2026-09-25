"""Descriptive OOF diagnostics retaining each model and variant identity."""
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Final, TypedDict, assert_never

import numpy as np
import polars as pl

from gmst.contracts import FloatArray, Panel

type Selection = str | tuple[str, str]
type PStars = Mapping[tuple[str, str, str], Mapping[str, float]]
type Cell = str | float | int | bool


class Edges(TypedDict):
    production: FloatArray
    temperature: FloatArray


ERROR_SCHEMA: Final = {"model": pl.String, "variant": pl.String, "condition": pl.String,
                       "bin": pl.String, "n": pl.Int64, "mae": pl.Float64, "rmse": pl.Float64}
FN_SCHEMA: Final = {"model": pl.String, "variant": pl.String, "C": pl.String, "p_star_kind": pl.String,
                    "date": pl.String, "kind": pl.String, "daytype": pl.String, "op_prev": pl.Int64,
                    "op_next": pl.Int64, "prev_day_max": pl.Float64, "temp_mean": pl.Float64,
                    "prod_sum": pl.Float64, "risk_cal": pl.Float64, "M_true": pl.Float64}
SUMMARY_SCHEMA: Final = {"model": pl.String, "variant": pl.String, "C": pl.String, "p_star_kind": pl.String,
                         "kind": pl.String, "condition": pl.String, "bin": pl.String, "n": pl.Int64,
                         "share": pl.Float64, "n_all": pl.Int64, "share_all": pl.Float64}


def _pairs(models: Sequence[Selection], variants: Sequence[str] | None) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in models:
        match item:
            case str():
                pairs.extend((item, variant) for variant in variants or ("main",))
            case tuple():
                pairs.append(item)
            case unreachable:
                assert_never(unreachable)
    return list(dict.fromkeys(pairs))


def _quartiles(values: FloatArray) -> FloatArray:
    finite = values[np.isfinite(values)]
    return np.quantile(finite, [0.25, 0.5, 0.75]) if finite.size else np.full(3, np.nan)


def bin_edges(panel: Panel) -> Edges:
    """Design-only observed points; September never influences descriptive bins."""
    design = np.array([day <= date(2021, 8, 31) for day in panel["dates"]])[:, None] & np.isfinite(panel["Y"])
    p, t = panel["X"]["생산량"], panel["X"]["기온"]
    return {"production": _quartiles(p[design & (p > 0)]), "temperature": _quartiles(t[design])}


def _bins(values: FloatArray, edges: FloatArray, zero: bool = False) -> list[str]:
    return ["unknown" if not np.isfinite(v) else "0" if zero and v == 0 else f"q{np.searchsorted(edges, v, side='right') + 1}"
            for v in values]


def _context(panel: Panel) -> pl.DataFrame:
    """Retrospective conditions only; this table is never passed to a predictor."""
    maxima = [float(row[np.isfinite(row)].max()) if np.isfinite(row).any() else float("nan") for row in panel["Y"]]
    temp = [float(row[np.isfinite(row)].mean()) if np.isfinite(row).any() else float("nan") for row in panel["X"]["기온"]]
    return pl.DataFrame({"date": [d.strftime("%Y.%m.%d") for d in panel["dates"]],
                         "daytype": np.array(["wk", "sat", "sun"])[panel["dtype"]],
                         "operating": panel["op"].astype(str), "op_prev": np.r_[-1, panel["op"][:-1]],
                         "op_next": np.r_[panel["op"][1:], -1], "prev_day_max": [float("nan"), *maxima[:-1]],
                         "temp_mean": temp, "prod_sum": panel["X"]["생산량"][:, ::4].sum(axis=1)})


def error_by_condition(slots: pl.DataFrame, panel: Panel, models: Sequence[Selection],
                       variants: Sequence[str] | None = None, edges: Edges | None = None) -> pl.DataFrame:
    """Compare finite issued point forecasts across descriptive production conditions."""
    if "q" not in slots.columns:
        ts = pl.col("datetime").str.strptime(pl.Datetime, "%Y.%m.%d %H:%M:%S")
        slots = slots.with_columns((ts.dt.hour().cast(pl.Int64) * 4 + ts.dt.minute() // 15).alias("q"))
    edges = bin_edges(panel) if edges is None else edges
    context = _context(panel).with_columns(
        pl.when(pl.col("op_prev") == 0).then(pl.lit("prev_nonop"))
        .when(pl.col("op_next") == 0).then(pl.lit("next_nonop")).otherwise(pl.lit("other")).alias("holiday_adjacent"))
    qctx = pl.DataFrame({"date": np.repeat(context["date"].to_numpy(), 96), "q": np.tile(np.arange(96), len(panel["dates"])),
                        "production": _bins(panel["X"]["생산량"].ravel(), edges["production"], True),
                        "temperature": _bins(panel["X"]["기온"].ravel(), edges["temperature"])})
    frames: list[pl.DataFrame] = []
    for model, variant in _pairs(models, variants):
        part = slots.filter((pl.col("model") == model) & (pl.col("variant") == variant)).sort(["date", "q"])
        if part.is_empty():
            continue
        part = part.join(qctx, on=["date", "q"]).join(context, on="date").with_columns(
            (pl.col("y_median") - pl.col("y_true")).alias("error"), (pl.col("q") // 4).cast(pl.String).alias("hour"))
        conditions = ["production", "temperature", "operating", "daytype", "hour", "holiday_adjacent"]
        part = part.filter(pl.col("error").is_finite())
        for condition in conditions:
            eligible = part.filter(pl.col(condition).is_not_null())
            frames.append(eligible.group_by(condition).agg(pl.len().cast(pl.Int64).alias("n"),
                pl.col("error").abs().mean().alias("mae"), (pl.col("error") ** 2).mean().sqrt().alias("rmse"))
                .rename({condition: "bin"}).with_columns(pl.col("bin").cast(pl.String), pl.lit(model).alias("model"),
                pl.lit(variant).alias("variant"), pl.lit(condition).alias("condition")).select(list(ERROR_SCHEMA)))
    return pl.concat(frames).sort(["model", "variant", "condition", "bin"]) if frames else pl.DataFrame(schema=ERROR_SCHEMA)


def fn_fp(days: pl.DataFrame, panel: Panel, p_star: PStars, models: Sequence[Selection],
          variants: Sequence[str] | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Classify each usable OOF day at .5 and its own variant/fold p*; compare FN/FP conditions."""
    context = _context(panel).drop("operating")
    frames: list[pl.DataFrame] = []
    for model, variant in _pairs(models, variants):
        part = days.filter((pl.col("model") == model) & (pl.col("variant") == variant) & pl.col("usable_peak"))
        if part.is_empty():
            continue
        for C in ("C50", "C75", "C90"):
            if f"event_{C}" not in part.columns:
                continue
            risk = f"risk_platt_{C}"
            valid = part.filter(pl.col(risk).is_finite() & pl.col(f"event_{C}").is_not_null())
            folds: list[str] = valid["fold"].to_list()
            for kind in ("0.5", "pstar"):
                thresholds = [0.5] * len(folds) if kind == "0.5" else [p_star[(model, variant, f)][C] for f in folds]
                classified = valid.with_columns((pl.col(risk) >= pl.Series("p", thresholds, dtype=pl.Float64)).alias("alert"))
                hit = pl.col(f"event_{C}").cast(pl.Boolean)
                frames.append(classified.select("model", "variant", "date", "M_true", pl.col(risk).alias("risk_cal"),
                    pl.lit(C).alias("C"), pl.lit(kind).alias("p_star_kind"),
                    pl.when(hit & pl.col("alert")).then(pl.lit("TP")).when(hit).then(pl.lit("FN"))
                    .when(pl.col("alert")).then(pl.lit("FP")).otherwise(pl.lit("TN")).alias("kind"))
                    .join(context, on="date").select(list(FN_SCHEMA)).cast(pl.Schema(FN_SCHEMA)))
    daily = pl.concat(frames) if frames else pl.DataFrame(schema=FN_SCHEMA)
    rows: list[tuple[Cell, ...]] = []
    for key, part in daily.group_by(["model", "variant", "C", "p_star_kind"], maintain_order=True):
        for condition in ("daytype", "op_prev", "op_next", "prev_day_max", "temp_mean", "prod_sum"):
            labels = part[condition].cast(pl.String).to_list() if condition in ("daytype", "op_prev", "op_next") else _bins(part[condition].to_numpy(), _quartiles(part[condition].to_numpy()))
            bins = np.array(labels)
            for kind in ("all", "FN", "FP"):
                subset = np.ones(part.height, bool) if kind == "all" else part["kind"].to_numpy() == kind
                for label in sorted(set(labels)):
                    n_all, n = int((bins == label).sum()), int(((bins == label) & subset).sum())
                    rows.append((*key, kind, condition, label, n, n / subset.sum() if subset.any() else float("nan"), n_all, n_all / part.height))
    return daily, pl.DataFrame(rows, schema=SUMMARY_SCHEMA, orient="row")
