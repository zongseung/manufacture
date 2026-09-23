"""Descriptive OOF diagnostics retaining each model and variant identity."""
from collections.abc import Mapping, Sequence
from datetime import date
from typing import TYPE_CHECKING, Final, TypedDict, assert_never

import numpy as np
import polars as pl

from gmst.contracts import FloatArray, IntArray, Panel

if TYPE_CHECKING:
    from gmst.hmm import HMMState

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
        for state in ("state_filt", "state_smooth"):
            if state in part.columns:
                next_state = pl.col(state).shift(-1).over("date")
                next_q = pl.col("q").shift(-1).over("date")
                part = part.with_columns(pl.when((pl.col(state) >= 0) & (next_state >= 0) & (next_q == pl.col("q") + 1))
                    .then(pl.when(pl.col(state) != next_state).then(pl.lit("transition")).otherwise(pl.lit("stable")))
                    .alias(f"{state}_transition"))
                conditions.extend([state, f"{state}_transition"])
        part = part.filter(pl.col("error").is_finite())
        for condition in conditions:
            eligible = part.filter(pl.col(condition).is_not_null())
            if condition in ("state_filt", "state_smooth"):
                eligible = eligible.filter(pl.col(condition) >= 0)
            frames.append(eligible.group_by(condition).agg(pl.len().cast(pl.Int64).alias("n"),
                pl.col("error").abs().mean().alias("mae"), (pl.col("error") ** 2).mean().sqrt().alias("rmse"))
                .rename({condition: "bin"}).with_columns(pl.col("bin").cast(pl.String), pl.lit(model).alias("model"),
                pl.lit(variant).alias("variant"), pl.lit(condition).alias("condition")).select(list(ERROR_SCHEMA)))
    return pl.concat(frames).sort(["model", "variant", "condition", "bin"]) if frames else pl.DataFrame(schema=ERROR_SCHEMA)


def fn_fp(days: pl.DataFrame, panel: Panel, p_star: PStars, models: Sequence[Selection] | None = None,
          variants: Sequence[str] | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Classify each usable OOF day at .5 and its own variant/fold p*; compare FN/FP conditions."""
    context = _context(panel).drop("operating")
    frames: list[pl.DataFrame] = []
    for model, variant in _pairs(models or ("B4", "B1"), variants):
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


def _diagnostic_features(panel: Panel, idx: IntArray, holdout: IntArray) -> FloatArray:
    from gmst.backbone import asof_matrix
    from gmst.features import lgbm_rows

    masked: Panel = {**panel, "Y": panel["Y"].copy()}
    masked["Y"][holdout] = np.nan
    return lgbm_rows(masked, idx, "A+", asof_matrix(masked, idx, 10.0, 60))[0]


def leakage_gap(panel: Panel | None = None, num_boost_round: int = 100,
                max_groups: int | None = None) -> pl.DataFrame:
    """Compare random-day and event-held-out L2 diagnostics on identical pre-September slots.

    A supplied panel must include copies. This retrospective split comparison is
    a leakage diagnostic, not the forward forecast's validation performance.
    Held-out targets are masked in all training and validation features,
    including historical lags and the as-of backbone; scoring targets stay intact.
    """
    import lightgbm as lgb

    from gmst.baselines import LGB_PARAMS
    from gmst.features import load_panel

    if num_boost_round < 1 or (max_groups is not None and max_groups < 1):
        raise ValueError(f"Expected positive rounds/groups: rounds={num_boost_round}, groups={max_groups}")
    panel = load_panel(include_copies=True) if panel is None else panel
    idx = np.array([d for d, day in enumerate(panel["dates"]) if day <= date(2021, 8, 31)
                    and np.isfinite(panel["Y"][d]).any()], dtype=np.int64)
    event: list[str] = panel["days"]["event_id"].to_list()
    groups = list(dict.fromkeys(event[d] for d in idx))
    selected = set(groups if max_groups is None else groups[:max_groups])
    score_days = np.array([d for d in idx if event[d] in selected], dtype=np.int64)
    schema = {"split": pl.String, "mae": pl.Float64, "n": pl.Int64, "n_days": pl.Int64,
              "n_splits": pl.Int64, "max_same_slot": pl.Int64}
    if len(idx) < 2 or len(groups) < 2:
        return pl.DataFrame([(name, float("nan"), 0, 0, 0, 0) for name in ("random_day_5fold", "loeo")], schema=schema, orient="row")
    y = panel["Y"][idx].ravel()
    row_day = np.repeat(idx, 96)
    random = np.array_split(np.random.default_rng(0).permutation(idx), 5)
    held = [np.array([d for d in idx if event[d] == group], dtype=np.int64) for group in groups if group in selected]
    result: list[tuple[Cell, ...]] = []
    for name, folds in (("random_day_5fold", random), ("loeo", held)):
        absolute, n, n_splits, same_slot = 0.0, 0, 0, 0
        evaluated: set[int] = set()
        for holdout in folds:
            validation = np.intersect1d(holdout, score_days)
            train = idx[~np.isin(idx, holdout)]
            tr = np.isin(row_day, train) & np.isfinite(y)
            val = np.isin(row_day, validation) & np.isfinite(y)
            if not tr.any() or not val.any():
                continue
            X = _diagnostic_features(panel, idx, holdout)
            # ponytail: 누수 진단은 L2 모델만 사용, 분위수·위험 누수 격차가 필요하면 B1 전체를 비교 (FR-92)
            booster = lgb.train({**LGB_PARAMS, "objective": "regression"}, lgb.Dataset(X[tr], label=y[tr]), num_boost_round=num_boost_round)
            pred = np.asarray(booster.predict(X[val]), dtype=np.float64)
            absolute += float(np.abs(y[val] - pred).sum())
            n += int(val.sum())
            n_splits += 1
            evaluated.update(map(int, validation))
            for d in validation:
                matches = ((panel["Y"][train] == panel["Y"][d]) & (panel["Y"][d] > 40)).sum(axis=1)
                same_slot = max(same_slot, int(matches.max(initial=0)))
        result.append((name, absolute / n if n else float("nan"), n, len(evaluated), n_splits, same_slot))
    return pl.DataFrame(result, schema=schema, orient="row")


def transitions(state: "HMMState", panel: Panel) -> pl.DataFrame:
    from gmst.hmm import transition_table

    return transition_table(state["model"], panel, state["train_idx"], state["protocol"])
