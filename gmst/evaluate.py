"""Temporal forecast evaluation and exploratory development comparisons."""
# SIZE_OK: root-approved flat scientific evaluation contract shared by all models.
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import datetime, time, timedelta
from math import erfc, sqrt
from typing import Final, Literal, TypedDict, assert_never

import numpy as np
import polars as pl

from gmst import features as ft
from gmst.contracts import BoolArray, FloatArray, IntArray, Model, Panel, Prediction

TAUS: Final = np.round(np.arange(1, 20) * 0.05, 2)
QCOLS: Final = [f"q{q:02d}" for q in range(5, 100, 5)]
FOLDS_CV: Final = ("f1", "f2", "f3", "f4")
CS: Final = ("C50", "C75", "C90")
KEYS: Final = ["model", "variant", "fold"]
SLOT_COLS: Final = ["fold", "variant", "model", "date", "datetime", "y_true", "is_missing",
                    "y_mean", "y_median", *QCOLS]
DAY_COLS: Final = ["fold", "variant", "model", "date", "usable_peak", "n_obs", "M_true",
                   "M_hat_median", "M_hat_mean", "peak_slot_true", "peak_time_mode", *CS,
                   *[f"risk_{m}_{c}" for m in ("raw", "platt") for c in CS],
                   *[f"event_{c}" for c in CS]]
METRIC_SCHEMA: Final = {"model": pl.String, "variant": pl.String, "fold": pl.String,
                        "stratum": pl.String, "metric": pl.String, "value": pl.Float64, "n": pl.Int64}
type GroupKey = tuple[str, str, str]
type PStarTable = dict[GroupKey, dict[str, float]]
type Cell = str | float | int | bool | None
type Method = Literal["platt", "iso"]


class PRF(TypedDict):
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


class Calibration(TypedDict):
    params: FloatArray
    status: str
    n_cal: int


def _mean(values: FloatArray) -> tuple[float, int]:
    return (float(values.mean()), int(values.size)) if values.size else (float("nan"), 0)


def mae(y: FloatArray, yhat: FloatArray) -> tuple[float, int]:
    valid = np.isfinite(y) & np.isfinite(yhat)
    return _mean(np.abs(y[valid] - yhat[valid]))


def rmse(y: FloatArray, yhat: FloatArray) -> tuple[float, int]:
    valid = np.isfinite(y) & np.isfinite(yhat)
    value, n = _mean((y[valid] - yhat[valid]) ** 2)
    return sqrt(value), n


def crps(y: FloatArray, Q: FloatArray) -> tuple[float, int]:
    valid = np.isfinite(y) & np.isfinite(Q).all(axis=1)
    delta = y[valid, None] - Q[valid]
    return _mean(2 * np.maximum(TAUS * delta, (TAUS - 1) * delta).mean(axis=1))


def coverage(y: FloatArray, lo: FloatArray, hi: FloatArray) -> tuple[float, int]:
    valid = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    return _mean(((lo[valid] <= y[valid]) & (y[valid] <= hi[valid])).astype(float))


def brier(p: FloatArray, e: FloatArray) -> float:
    valid = np.isfinite(p) & np.isfinite(e)
    return _mean((p[valid] - e[valid]) ** 2)[0]


def auc(p: FloatArray, e: FloatArray) -> float:
    valid = np.isfinite(p) & np.isfinite(e)
    positive, negative = p[valid & (e == 1)], p[valid & (e == 0)]
    if not positive.size or not negative.size:
        return float("nan")
    return float(((positive[:, None] > negative) + 0.5 * (positive[:, None] == negative)).mean())


def prf(pred: BoolArray, true: BoolArray) -> PRF:
    tp, fp, fn = int((pred & true).sum()), int((pred & ~true).sum()), int((~pred & true).sum())
    return PRF(tp=tp, fp=fp, fn=fn, precision=tp / (tp + fp) if tp + fp else float("nan"),
               recall=tp / (tp + fn) if tp + fn else float("nan"),
               f1=2 * tp / (2 * tp + fp + fn) if tp + fn else float("nan"))


def obs_max(Y: FloatArray, obs: BoolArray) -> FloatArray:
    """Return same-slot maxima, including a scalar-shaped array for one day."""
    maxima = np.max(np.where(obs & np.isfinite(Y), Y, -np.inf), axis=-1)
    return np.asarray(np.where(np.isfinite(maxima), maxima, np.nan), dtype=float)


def peak_mae(M_true: FloatArray, M_hat: FloatArray) -> tuple[float, int]:
    return mae(M_true, M_hat)


def peak_hit(true_slot: FloatArray, pred_slot: FloatArray, k: int = 2) -> tuple[float, int]:
    valid = np.isfinite(true_slot) & np.isfinite(pred_slot)
    return _mean((np.abs(true_slot[valid] - pred_slot[valid]) <= k).astype(float))


def thresholds(panel: Panel, train_idx: IntArray) -> tuple[FloatArray, int]:
    idx = train_idx[(panel["op"][train_idx] == 1) & ft.usable_peak(panel)[train_idx]]
    maxima = obs_max(panel["Y"][idx], np.isfinite(panel["Y"][idx]))
    return (np.quantile(maxima, [0.5, 0.75, 0.9]) if idx.size else np.full(3, np.nan)), int(idx.size)


def point_pred(y: FloatArray, C: FloatArray) -> Prediction:
    finite = np.isfinite(y)
    maximum = float(obs_max(y, finite))
    return Prediction(y_mean=y, y_median=y, q=None, paths=None, M_hat_median=maximum,
                      M_hat_mean=maximum, peak_time_mode=int(np.nanargmax(y)) if finite.any() else 0,
                      risk_raw=(maximum > C).astype(float) if finite.any() else np.full(3, np.nan))


def _empty_days() -> pl.DataFrame:
    schema: dict[str, type[pl.DataType]] = {c: pl.Float64 for c in DAY_COLS}
    schema.update({c: pl.String for c in (*KEYS, "date")})
    schema.update({"usable_peak": pl.Boolean, "n_obs": pl.Int64, "peak_slot_true": pl.Int64,
                   "peak_time_mode": pl.Int64})
    return pl.DataFrame(schema=schema)


def _day_row(pred: Prediction, panel: Panel, d: int, identity: GroupKey, C: FloatArray) -> dict[str, Cell]:
    model, variant, fold = identity
    observed = np.isfinite(panel["Y"][d])
    maximum = float(obs_max(panel["Y"][d], observed))
    usable = bool(ft.usable_peak(panel)[d])
    median, mean = pred["M_hat_median"], pred["M_hat_mean"]
    paths = pred["paths"]
    risk = pred["risk_raw"]
    score_slots = observed if observed.any() else np.ones(96, dtype=bool)
    if paths is not None:
        maxima = obs_max(paths, score_slots)
        median, mean = float(np.median(maxima)), float(maxima.mean())
        if observed.any() and not observed.all():
            risk = np.where(np.isfinite(C), (maxima[:, None] > C).mean(axis=0), np.nan)
    elif pred["q"] is None:
        median = mean = float(obs_max(pred["y_median"], score_slots))
        if observed.any() and not observed.all():
            risk = np.where(np.isfinite(median) & np.isfinite(C), (median > C).astype(float), np.nan)
    # B1's separate day classifier has no joint paths to restrict; keep its issued risk.
    return {"model": model, "variant": variant, "fold": fold, "date": panel["dates"][d].strftime("%Y.%m.%d"),
            "usable_peak": usable, "n_obs": int(observed.sum()), "M_true": maximum,
            "M_hat_median": median, "M_hat_mean": mean,
            "peak_slot_true": int(np.nanargmax(panel["Y"][d])) if observed.any() else None,
            "peak_time_mode": pred["peak_time_mode"], **{c: float(C[j]) for j, c in enumerate(CS)},
            **{f"risk_raw_{c}": float(risk[j]) for j, c in enumerate(CS)},
            **{f"risk_platt_{c}": float("nan") for c in CS},
            **{f"event_{c}": float(maximum > C[j]) if usable and np.isfinite(C[j]) else float("nan") for j, c in enumerate(CS)}}


def rolling_origin[S](model: Model[S], panel: Panel, folds: Sequence[str] = FOLDS_CV,
                       variant: str = "main", states: Mapping[str, S] | None = None
                       ) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, S], pl.DataFrame]:
    """Issue ordered forecasts and train-period calibration predictions for every model."""
    fitted: dict[str, S] = {}
    slots: list[pl.DataFrame] = []
    days: list[dict[str, Cell]] = []
    inner: list[dict[str, Cell]] = []
    for fold in folds:
        C, _ = thresholds(panel, ft.role_idx(panel, fold, "train"))
        state = states[fold] if states is not None else model["fit"](panel, fold, C)
        fitted[fold] = state
        identity = model["name"], variant, fold
        outer_idx = ft.role_idx(panel, fold, "test" if fold == "test" else "val")
        for d in sorted(outer_idx):
            d = int(d)
            pred = model["predict"](state, panel, d)
            days.append(_day_row(pred, panel, d, identity, C))
            quantiles = pred["q"]
            q = quantiles if quantiles is not None else np.full((19, 96), np.nan)
            start = datetime.combine(panel["dates"][d], time())
            slots.append(pl.DataFrame({"fold": [fold] * 96, "variant": [variant] * 96,
                "model": [model["name"]] * 96, "date": [start.strftime("%Y.%m.%d")] * 96,
                "datetime": [(start + timedelta(minutes=15 * j)).strftime("%Y.%m.%d %H:%M:%S") for j in range(96)],
                "y_true": panel["Y"][d], "is_missing": panel["is_missing"][d],
                "y_mean": pred["y_mean"], "y_median": pred["y_median"],
                **{c: q[j] for j, c in enumerate(QCOLS)}}))
        cal_idx = ft.inner_idx(panel, fold)
        inner_state = model["inner_state"](state)
        if len(cal_idx) and inner_state is None:
            raise ValueError(f"Missing inner calibration state for {identity}")
        if inner_state is not None:
            inner.extend(_day_row(model["predict"](inner_state, panel, int(d)), panel, int(d), identity, C) for d in cal_idx)
    day_frame = pl.DataFrame(days, schema_overrides=_empty_days().schema) if days else _empty_days()
    day_frame = day_frame.select(*DAY_COLS)
    inner_frame = pl.DataFrame(inner, schema=_empty_days().schema) if inner else _empty_days()
    slot_schema: dict[str, type[pl.DataType]] = {c: pl.Float64 for c in SLOT_COLS}
    slot_schema.update({c: pl.String for c in (*KEYS, "date", "datetime")})
    slot_schema.update({"is_missing": pl.Boolean})
    return pl.concat(slots) if slots else pl.DataFrame(schema=slot_schema), day_frame, fitted, inner_frame


def _logit(p: FloatArray, N: int) -> FloatArray:
    clipped = np.clip(p, 1 / (2 * N), 1 - 1 / (2 * N))
    return np.log(clipped / (1 - clipped))


def _sigmoid(x: FloatArray) -> FloatArray:
    return np.exp(-np.logaddexp(0, -x))


def platt_fit(p: FloatArray, e: FloatArray, N: int = 2000) -> tuple[float, float]:
    valid = np.isfinite(p) & np.isfinite(e)
    p, e = p[valid], e[valid]
    if not e.size:
        return 1.0, 0.0
    single = np.unique(e).size == 1
    x = np.zeros_like(p) if single else _logit(p, N)
    design = np.column_stack((x, np.ones_like(x)))
    ab = np.array([0.0 if single else 1.0, 0.0])
    for _ in range(100):
        logits = design @ ab
        fitted = _sigmoid(logits)
        gradient = design.T @ (fitted - e) + 1e-3 * ab
        hessian = (design.T * (fitted * (1 - fitted))) @ design + 1e-3 * np.eye(2)
        step = np.linalg.solve(hessian, gradient)
        loss = float(np.sum(np.logaddexp(0, logits) - e * logits) + 5e-4 * (ab @ ab))
        scale = 1.0
        while scale > 1e-8:
            candidate = ab - scale * step
            candidate_logits = design @ candidate
            new_loss = float(np.sum(np.logaddexp(0, candidate_logits) - e * candidate_logits) + 5e-4 * (candidate @ candidate))
            if new_loss <= loss:
                break
            scale *= 0.5
        ab -= scale * step
        if np.max(np.abs(scale * step)) < 1e-10:
            break
    return float(ab[0]), float(ab[1])


def platt_apply(ab: tuple[float, float], p: FloatArray, N: int = 2000) -> FloatArray:
    if ab == (1.0, 0.0):
        return p.copy()
    return _sigmoid(ab[0] * _logit(p, N) + ab[1])


def pav_fit(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    valid = np.isfinite(x) & np.isfinite(y)
    if not valid.any():
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    xs, inverse, counts = np.unique(x[valid], return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=y[valid])
    blocks: list[tuple[int, int, float, int]] = []
    for i in range(len(xs)):
        blocks.append((i, i + 1, float(sums[i]), int(counts[i])))
        while len(blocks) > 1 and blocks[-2][2] / blocks[-2][3] > blocks[-1][2] / blocks[-1][3]:
            right, left = blocks.pop(), blocks.pop()
            blocks.append((left[0], right[1], left[2] + right[2], left[3] + right[3]))
    ys = np.empty_like(xs)
    for start, stop, total, weight in blocks:
        ys[start:stop] = total / weight
    return xs, ys


def pav_apply(model: tuple[FloatArray, FloatArray], x: FloatArray) -> FloatArray:
    return np.interp(x, *model)


def fit_calibrator(rows: pl.DataFrame, method: Method) -> dict[str, Calibration]:
    calibrated: dict[str, Calibration] = {}
    rows = rows.filter(pl.col("usable_peak"))
    for c in CS:
        p, e = rows[f"risk_raw_{c}"].to_numpy(), rows[f"event_{c}"].to_numpy()
        valid = np.isfinite(p) & np.isfinite(e)
        n = int(valid.sum())
        status = "default_empty" if not n else "single_class" if np.unique(e[valid]).size == 1 else "ok"
        match method:
            case "platt":
                params = np.asarray(platt_fit(p, e))
            case "iso":
                params = np.stack(pav_fit(p, e))
            case unreachable:
                assert_never(unreachable)
        calibrated[c] = Calibration(params=params, status=status, n_cal=n)
    return calibrated


def apply_calibrator(cal: Mapping[str, Calibration], rows: pl.DataFrame, method: Method) -> pl.DataFrame:
    probabilities: list[FloatArray] = []
    for c in CS:
        raw, params = rows[f"risk_raw_{c}"].to_numpy(), cal[c]["params"]
        match method:
            case "platt":
                probabilities.append(platt_apply((float(params[0]), float(params[1])), raw))
            case "iso":
                probabilities.append(pav_apply((params[0], params[1]), raw))
            case unreachable:
                assert_never(unreachable)
    monotone = np.minimum.accumulate(np.column_stack(probabilities), axis=1)
    return rows.with_columns(*[pl.Series(f"risk_{method}_{c}", monotone[:, j]) for j, c in enumerate(CS)],
        *[pl.lit(cal[c]["status"]).alias(f"calibration_status_{method}_{c}") for c in CS],
        *[pl.lit(cal[c]["n_cal"]).alias(f"n_cal_{method}_{c}") for c in CS],
        (pl.lit("ok") if method == "iso" else pl.col("isotonic_status").fill_null("not_run") if "isotonic_status" in rows.columns
         else pl.lit("not_run")).alias("isotonic_status"))


def _date_expr() -> pl.Expr:
    return pl.col("date").cast(pl.String).str.replace_all("-", ".", literal=True)


def _group_filter(key: GroupKey) -> pl.Expr:
    return pl.all_horizontal([pl.col(c) == value for c, value in zip(KEYS, key, strict=True)])


def calibrate_oof(days: pl.DataFrame, method: Method = "platt", inner: pl.DataFrame | None = None,
                  ref: pl.DataFrame | None = None) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """Fit only on same-fold inner rows, or earlier same-variant OOF for final issuance."""
    outputs: list[pl.DataFrame] = []
    inner_outputs: list[pl.DataFrame] = []
    for key, group in days.partition_by(KEYS, as_dict=True, maintain_order=True).items():
        model, variant, fold = str(key[0]), str(key[1]), str(key[2])
        identity = model, variant, fold
        calibration_rows = inner.filter(_group_filter(identity)) if inner is not None else days.head(0)
        if "date" in calibration_rows.columns and not calibration_rows.is_empty():
            cutoff = group.select(_date_expr().min()).item()
            if calibration_rows.filter(_date_expr() >= cutoff).height:
                raise ValueError(f"Calibration rows must precede outer dates for {identity}")
        if ref is not None:
            if fold != "test":
                raise ValueError("OOF reference calibration is reserved for test rows")
            cutoff = group.select(_date_expr().min()).item()
            calibration_rows = ref.filter((pl.col("model") == model) & (pl.col("variant") == variant)
                                          & pl.col("fold").is_in(FOLDS_CV) & (_date_expr() < cutoff))
        cal = fit_calibrator(calibration_rows, method)
        outputs.append(apply_calibrator(cal, group, method))
        if inner is not None:
            inner_outputs.append(apply_calibrator(cal, inner.filter(_group_filter(identity)), method))
    result = pl.concat(outputs, how="diagonal_relaxed") if outputs else days
    calibrated_inner = pl.concat(inner_outputs, how="diagonal_relaxed") if inner_outputs else inner
    return result, calibrated_inner


def p_star(p: FloatArray, e: FloatArray) -> float:
    valid = np.isfinite(p) & np.isfinite(e)
    p, e = p[valid], e[valid]
    if not p.size or not (e == 1).any():
        return 0.5
    candidates = np.unique(p)
    scores = [prf(p >= c, e.astype(bool))["f1"] for c in candidates]
    return float(candidates[int(np.argmax(scores))])


def p_star_table(inner: pl.DataFrame | None, expected_keys: Iterable[GroupKey]) -> PStarTable:
    result: PStarTable = {}
    for key in expected_keys:
        rows = inner.filter(_group_filter(key) & pl.col("usable_peak")) if inner is not None else _empty_days()
        result[key] = {c: p_star(rows[f"risk_platt_{c}"].to_numpy(), rows[f"event_{c}"].to_numpy()) for c in CS}
    return result


def bootstrap_days(panel: Panel) -> IntArray:
    idx = np.unique(np.concatenate([ft.role_idx(panel, fold, "val") for fold in FOLDS_CV]))
    return idx[np.isfinite(panel["Y"][idx]).any(axis=1)]


def block_bootstrap(num: FloatArray, den: FloatArray, B: int = 2000, seed: int = 0,
                    block_length: int = 1) -> tuple[float, float, float]:
    """Resample paired day sums and their own valid denominators."""
    if len(num) != len(den) or B < 1 or block_length < 1:
        raise ValueError("Invalid paired bootstrap dimensions or replicate count")
    valid = np.isfinite(num) & np.isfinite(den) & (den >= 0)
    num, den = num[valid], den[valid]
    n = len(num)
    if not n or den.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    if block_length == 1:
        indices = rng.integers(0, n, size=(B, n))
    else:
        length = min(block_length, n)
        starts = rng.integers(0, n - length + 1, size=(B, (n + length - 1) // length))
        indices = (starts[:, :, None] + np.arange(length)).reshape(B, -1)[:, :n]
    totals = den[indices].sum(axis=1)
    ratios = np.divide(num[indices].sum(axis=1), totals, out=np.full(B, np.nan), where=totals > 0)
    lo, hi = np.nanpercentile(ratios, [2.5, 97.5])
    return float(num.sum() / den.sum()), float(lo), float(hi)


def dm_test(d: FloatArray) -> tuple[float, float]:
    """DM-like normal day-loss check; variance is not autocorrelation robust."""
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return float("nan"), float("nan")
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    statistic = mean / (sd / sqrt(len(d))) if sd else 0.0 if mean == 0 else float(np.copysign(np.inf, mean))
    return statistic, erfc(abs(statistic) / sqrt(2))


def _with_op(frame: pl.DataFrame, panel: Panel) -> pl.DataFrame:
    calendar = pl.DataFrame({"date": [d.strftime("%Y.%m.%d") for d in panel["dates"]],
                             "op_true": panel["op"], "dtype": panel["dtype"]})
    return frame.drop("op_true", "dtype", strict=False).with_columns(_date_expr()).join(calendar, on="date", how="left", validate="m:1")


def _groups(frame: pl.DataFrame) -> Iterator[tuple[GroupKey, str, pl.DataFrame]]:
    for key, pair in frame.partition_by(["model", "variant"], as_dict=True, maintain_order=True).items():
        folds = pair["fold"].unique(maintain_order=True).to_list()
        if pair.filter(pl.col("fold").is_in(FOLDS_CV)).height:
            folds.append("pooled")
        for fold in folds:
            rows = pair.filter(pl.col("fold").is_in(FOLDS_CV) if fold == "pooled" else pl.col("fold") == fold)
            for stratum, predicate in (("all", pl.lit(True)), ("op", pl.col("op_true") == 1), ("nonop", pl.col("op_true") == 0)):
                yield (str(key[0]), str(key[1]), str(fold)), stratum, rows.filter(predicate)


def point_metrics(slots: pl.DataFrame, days: pl.DataFrame, panel: Panel) -> pl.DataFrame:
    rows: list[tuple[str, str, str, str, str, float, int]] = []
    for key, stratum, group in _groups(_with_op(slots, panel)):
        y, median, mean = (group[c].to_numpy() for c in ("y_true", "y_median", "y_mean"))
        metrics = {"mae": mae(y, median), "rmse": rmse(y, mean), "crps": crps(y, group.select(QCOLS).to_numpy())}
        for name, lo, hi in (("cov50", "q25", "q75"), ("cov80", "q10", "q90"), ("cov90", "q05", "q95")):
            metrics[name] = coverage(y, group[lo].to_numpy(), group[hi].to_numpy())
        rows.extend((*key, stratum, name, value, n) for name, (value, n) in metrics.items())
    for key, stratum, group in _groups(_with_op(days, panel)):
        usable = group.filter(pl.col("usable_peak"))
        metrics = {"peak_mae": peak_mae(usable["M_true"].to_numpy(), usable["M_hat_median"].to_numpy()),
                   "peak_hit2": peak_hit(usable["peak_slot_true"].to_numpy(), usable["peak_time_mode"].to_numpy())}
        for c in CS:
            p, e = usable[f"risk_raw_{c}"].to_numpy(), usable[f"event_{c}"].to_numpy()
            valid = np.isfinite(p) & np.isfinite(e)
            metrics[f"brier_raw_{c}"] = brier(p, e), int(valid.sum())
            metrics[f"auc_{c}"] = auc(p, e), int(valid.sum())
            metrics[f"n_events_{c}"] = float(np.sum(e[np.isfinite(e)])), int(np.isfinite(e).sum())
        risk = usable.select([f"risk_raw_{c}" for c in CS]).to_numpy()
        events = usable.select([f"event_{c}" for c in CS]).to_numpy()
        valid = np.isfinite(risk).all(axis=1) & np.isfinite(events).all(axis=1)
        metrics["brier_mean_raw"] = _mean(((risk[valid] - events[valid]) ** 2).mean(axis=1))
        rows.extend((*key, stratum, name, value, n) for name, (value, n) in metrics.items())
    return pl.DataFrame(rows, schema=METRIC_SCHEMA, orient="row")


def day_losses(slots: pl.DataFrame, days: pl.DataFrame, panel: Panel) -> pl.DataFrame:
    """Reduce a single model/variant to day sums without losing valid denominators."""
    if slots.select("model", "variant").unique().height > 1 or days.select("model", "variant").unique().height > 1:
        raise ValueError("day_losses requires one model/variant pair")
    permitted = [panel["dates"][int(i)].strftime("%Y.%m.%d") for i in bootstrap_days(panel)]
    slots = slots.with_columns(_date_expr()).filter(pl.col("date").is_in(permitted))
    days = days.with_columns(_date_expr()).filter(pl.col("date").is_in(permitted))
    rows: list[dict[str, Cell]] = []
    for key, group in slots.partition_by(["fold", "date"], as_dict=True, maintain_order=True).items():
        fold, date = str(key[0]), str(key[1])
        day = days.filter((pl.col("fold") == fold) & (pl.col("date") == date))
        if day.height != 1:
            raise ValueError(f"Expected one day record for {fold}/{date}")
        ae, n = mae(group["y_true"].to_numpy(), group["y_median"].to_numpy())
        score, nq = crps(group["y_true"].to_numpy(), group.select(QCOLS).to_numpy())
        p, e = day.select([f"risk_platt_{c}" for c in CS]).to_numpy()[0], day.select([f"event_{c}" for c in CS]).to_numpy()[0]
        usable = bool(day["usable_peak"][0]) and np.isfinite(p).all() and np.isfinite(e).all()
        rows.append({"fold": fold, "date": date, "ae_sum": ae * n if n else 0.0, "n_pts": n,
                     "crps_sum": score * nq if nq else 0.0, "n_crps": nq,
                     "brier_sum": float(((p - e) ** 2).mean()) if usable else 0.0, "n_brier": int(usable)})
    return pl.DataFrame(rows, schema={"fold": pl.String, "date": pl.String, "ae_sum": pl.Float64,
                         "n_pts": pl.Int64, "crps_sum": pl.Float64, "n_crps": pl.Int64,
                         "brier_sum": pl.Float64, "n_brier": pl.Int64}).sort("date")


def compare(slots_a: pl.DataFrame, days_a: pl.DataFrame, slots_b: pl.DataFrame, days_b: pl.DataFrame,
            panel: Panel, name: str, metrics: Sequence[str] = ("mae", "crps", "brier_mean"),
            B: int = 2000, seed: int = 0) -> list[dict[str, Cell]]:
    # ponytail: paired day bootstrap assumes independent dates, use 7-day sensitivity when lag-1 correlation is significant (FR-56).
    a, b = day_losses(slots_a, days_a, panel), day_losses(slots_b, days_b, panel)
    paired = a.join(b, on=["fold", "date"], suffix="_b", validate="1:1").sort("date")
    slot_pairs = slots_a.join(slots_b, on=["fold", "datetime"], suffix="_b", validate="1:1")
    truth, other_truth = slot_pairs["y_true"].to_numpy(), slot_pairs["y_true_b"].to_numpy()
    if not np.array_equal(truth, other_truth, equal_nan=True):
        raise ValueError(f"Paired observations differ for {name}")
    day_pairs = days_a.join(days_b, on=["fold", "date"], suffix="_b", validate="1:1")
    event_columns = [*CS, *[f"event_{c}" for c in CS]]
    if "brier_mean" in metrics and not np.array_equal(
        day_pairs.select(event_columns).to_numpy(),
        day_pairs.select([f"{c}_b" for c in event_columns]).to_numpy(), equal_nan=True,
    ):
        raise ValueError(f"Paired risk events or thresholds differ for {name}")
    output: list[dict[str, Cell]] = []
    for metric in metrics:
        if metric in ("mae", "crps"):
            columns = ["y_median"] if metric == "mae" else QCOLS
            mask = np.isfinite(truth) & np.isfinite(slot_pairs.select(columns).to_numpy()).all(axis=1)
            other_mask = np.isfinite(other_truth) & np.isfinite(slot_pairs.select([f"{c}_b" for c in columns]).to_numpy()).all(axis=1)
            if not np.array_equal(mask, other_mask):
                raise ValueError(f"Paired {metric} prediction masks differ for {name}")
        numerator, denominator = {"mae": ("ae_sum", "n_pts"), "crps": ("crps_sum", "n_crps"),
                                  "brier_mean": ("brier_sum", "n_brier")}[metric]
        den = paired[denominator].to_numpy().astype(float)
        other_den = paired[f"{denominator}_b"].to_numpy()
        if not np.array_equal(den, other_den):
            raise ValueError(f"Paired {metric} prediction masks differ for {name}")
        num = (paired[numerator] - paired[f"{numerator}_b"]).to_numpy()
        delta, lo, hi = block_bootstrap(num, den, B=B, seed=seed)
        daily = np.divide(num, den, out=np.full(len(den), np.nan), where=den > 0)
        stat, p = dm_test(daily)
        dates = paired["date"].str.strptime(pl.Date, "%Y.%m.%d").to_numpy()
        consecutive = (np.isfinite(daily[:-1]) & np.isfinite(daily[1:])
                       & (np.diff(dates) == np.timedelta64(1, "D")))
        left, right = daily[:-1][consecutive], daily[1:][consecutive]
        rho = float(np.corrcoef(left, right)[0, 1]) if len(left) >= 3 and np.std(left) > 0 and np.std(right) > 0 else float("nan")
        correlated = bool(np.isfinite(rho) and abs(rho) > 1.96 / sqrt(len(left)))
        block_lo, block_hi = float("nan"), float("nan")
        if correlated:
            positions = (dates - dates[0]).astype("timedelta64[D]").astype(np.int64)
            calendar_num = np.zeros(int(positions[-1]) + 1)
            calendar_den = np.zeros_like(calendar_num)
            calendar_num[positions], calendar_den[positions] = num, den
            _, block_lo, block_hi = block_bootstrap(calendar_num, calendar_den, B=B, seed=seed, block_length=7)
        output.append({"comparison": name, "metric": metric, "delta": delta, "ci_lo": lo, "ci_hi": hi,
                       "dm_stat": stat, "dm_p": p, "dm_method": "DM-like normal, nonrobust variance",
                       "n_days": paired.height, "n_valid_days": int((den > 0).sum()), "n_obs": int(den.sum()),
                       "lag1_correlation": rho, "serial_correlation": correlated,
                       "block7_ci_lo": block_lo, "block7_ci_hi": block_hi,
                       "block7_status": "ok" if correlated else "not_indicated", "exploratory": True})
    return output
