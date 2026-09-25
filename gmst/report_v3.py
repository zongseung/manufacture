"""Chapter 3 (영향요인 및 오류분석) tables and figures from the v3 OOF results."""
import argparse
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from gmst import ROOT, analysis, features, scenario
from gmst import evaluate as ev
from gmst.bat import groups_from_plan
from gmst.contracts import Panel

MODELS = ("B0", "M2", "BAT")  # M4 (BAT + cubic) is dropped from the report
KIND_NAMES = {0: "비가동", 1: "1–12h", 2: "13–19h", 3: "20h+"}


def slot_context(panel: Panel) -> pl.DataFrame:
    """Per (date, q): operating type, type × slot production on/off, TOU band (energy), hour."""
    group, parent = groups_from_plan(panel["X"]["생산량"][:, ::4])
    holidays = scenario.tariff_holidays()
    regime = np.array(["k0", "k1_off", "k1_on", "k2_off", "k2_on", "k3_off", "k3_on"])
    rows = [(d.strftime("%Y.%m.%d"), q, f"k{parent[g]}", regime[g],
             str(scenario.band(datetime(d.year, d.month, d.day, q // 4, 15 * (q % 4)), d in holidays, True)), f"{q // 4:02d}")
            for d, gd in zip(panel["dates"], group, strict=True) for q, g in enumerate(gd)]
    return pl.DataFrame(rows, schema=["date", "q", "optype", "regime", "tou", "hour"], orient="row")


def error_by_regime(slots: pl.DataFrame, ctx: pl.DataFrame, models: tuple[str, ...] = MODELS) -> pl.DataFrame:
    """MAE/RMSE of y_median per model × condition bin, with the BAT−M2-style delta against M2."""
    ts = pl.col("datetime").str.strptime(pl.Datetime, "%Y.%m.%d %H:%M:%S")
    part = (slots.filter(pl.col("model").is_in(models))
            .with_columns((ts.dt.hour().cast(pl.Int64) * 4 + ts.dt.minute() // 15).alias("q"),
                          (pl.col("y_median") - pl.col("y_true")).alias("error"))
            .filter(pl.col("error").is_finite()).join(ctx, on=["date", "q"]))
    frames = [part.group_by("model", "variant", c).agg(pl.len().cast(pl.Int64).alias("n"), pl.col("error").abs().mean().alias("mae"),
                                                        (pl.col("error") ** 2).mean().sqrt().alias("rmse"))
              .rename({c: "bin"}).with_columns(pl.lit(c).alias("condition")).select(list(analysis.ERROR_SCHEMA))
              for c in ("regime", "optype", "tou", "hour")]
    table = pl.concat(frames)
    ref = table.filter(pl.col("model") == "M2").select("variant", "condition", "bin", pl.col("mae").alias("mae_M2"))
    return (table.join(ref, on=["variant", "condition", "bin"], how="left")
            .with_columns((pl.col("mae") - pl.col("mae_M2")).alias("mae_minus_M2")).drop("mae_M2")
            .sort("condition", "bin", "model"))


def f2(tp: int, fp: int, fn: int) -> float:
    """F-beta with beta=2 (recall-weighted); NaN when there are no events, like ev.prf's F1."""
    return 5 * tp / (5 * tp + 4 * fn + fp) if tp + fn else float("nan")


def peak_events(days: pl.DataFrame, pstar: ev.PStarTable, models: tuple[str, ...] = MODELS) -> pl.DataFrame:
    """Auxiliary day-peak exceedance classification at calibrated risk ≥ p*(fold), per fold and pooled.

    Adds an "always_alarm" row per C/fold (every usable day flagged: precision = base rate, recall = 1)
    computed on the first model's usable days."""
    rows = []
    for model in (*models, "always_alarm"):
        part = days.filter((pl.col("model") == (models[0] if model == "always_alarm" else model)) & pl.col("usable_peak"))
        for c in ev.CS:
            cut = (np.zeros(len(part)) if model == "always_alarm" else
                   np.array([pstar[(model, v, f)][c] for v, f in part.select("variant", "fold").iter_rows()]))
            p, e, fold = part[f"risk_platt_{c}"].to_numpy(), part[f"event_{c}"].to_numpy(), part["fold"].to_numpy()
            if model == "always_alarm":
                p = np.where(np.isfinite(p), 1.0, np.nan)
            for f in ("all", *sorted(set(fold))):
                m = np.isfinite(p) & np.isfinite(e) & ((fold == f) | (f == "all"))
                s = ev.prf(p[m] >= cut[m], e[m] == 1)
                rows.append((model, c, f, int(m.sum()), int(e[m].sum()), float(e[m].mean()) if m.any() else float("nan"),
                             s["tp"], s["fp"], s["fn"], s["precision"], s["recall"], s["f1"], f2(s["tp"], s["fp"], s["fn"]),
                             ev.auc(p[m], e[m]), ev.brier(p[m], e[m]), float(np.mean(cut[m])) if m.any() else float("nan")))
    return pl.DataFrame(rows, orient="row", schema=["model", "C", "fold", "n_days", "n_events", "base_rate", "tp", "fp", "fn",
                                                     "precision", "recall", "f1", "f2", "auc", "brier", "p_star_mean"])


def fn_fp_conditions(days: pl.DataFrame, panel: Panel, pstar: ev.PStarTable, models: tuple[str, ...] = ("BAT", "M2")) -> pl.DataFrame:
    """analysis.fn_fp at p*, plus operating type / weekday / month shares of FN and FP days."""
    daily, summary = analysis.fn_fp(days, panel, pstar, models)
    daily = daily.filter(pl.col("p_star_kind") == "pstar")
    group, parent = groups_from_plan(panel["X"]["생산량"][:, ::4])
    kind = parent[group[:, 0]]
    extra = pl.DataFrame({"date": [d.strftime("%Y.%m.%d") for d in panel["dates"]], "optype": [f"k{k}" for k in kind],
                          "weekday": [d.strftime("%a") for d in panel["dates"]], "month": [f"{d.month:02d}" for d in panel["dates"]]})
    daily = daily.join(extra, on="date")
    frames = [summary.filter(pl.col("p_star_kind") == "pstar")]
    for cond in ("optype", "weekday", "month"):
        g = daily.group_by("model", "variant", "C", "p_star_kind", cond)
        counts = g.agg(pl.len().alias("n_all"), (pl.col("kind") == "FN").sum().alias("FN"), (pl.col("kind") == "FP").sum().alias("FP"))
        tot = pl.col("n_all").sum().over("model", "variant", "C")
        for k in ("all", "FN", "FP"):
            n = pl.col("n_all") if k == "all" else pl.col(k)
            frames.append(counts.select("model", "variant", "C", "p_star_kind", pl.lit(k).alias("kind"), pl.lit(cond).alias("condition"),
                                        pl.col(cond).alias("bin"), n.cast(pl.Int64).alias("n"),
                                        (n / n.sum().over("model", "variant", "C")).alias("share"),
                                        pl.col("n_all").cast(pl.Int64), (pl.col("n_all") / tot).alias("share_all")))
    return pl.concat(frames, how="vertical_relaxed").sort("model", "C", "condition", "kind", "bin")


def _fit_fold(fold: str, n_iter: int) -> dict[str, np.ndarray]:
    from gmst.bat import fit_bat

    panel = features.load_panel()
    train = features.role_idx(panel, fold, "train")
    state = fit_bat(panel, train, ev.thresholds(panel, train)[0], n_iter, n_iter // 2)
    print(f"[report_v3] BAT refit {fold} done", flush=True)
    return {"theta": state["theta"], "w": state["w"], "scale": np.array(state["scale"])}


def bat_draws(out: Path, folds: tuple[str, ...], n_iter: int, jobs: int) -> dict[str, dict[str, np.ndarray]]:
    """Refit BAT per fold (cached as npz; delete the file to refit after bat.py changes)."""
    todo = [f for f in folds if not (out / f"bat_draws_{f}.npz").exists()]
    if todo:
        with ProcessPoolExecutor(min(jobs, len(todo)), mp.get_context("spawn")) as pool:
            for f, draws in zip(todo, pool.map(_fit_fold, todo, [n_iter] * len(todo)), strict=True):
                np.savez(out / f"bat_draws_{f}.npz", **draws)
    return {f: dict(np.load(out / f"bat_draws_{f}.npz")) for f in folds}


def _q(x: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return x.mean(axis), *np.quantile(x, [0.05, 0.5, 0.95], axis=axis)


def attention_tables(draws: dict[str, dict[str, np.ndarray]]) -> tuple[dict[int, pl.DataFrame], pl.DataFrame, pl.DataFrame]:
    from gmst.bat import attention

    theta = np.concatenate([d["theta"] for d in draws.values()])  # pooled over folds (S·F, 4, 5)
    maps = {}
    for k in (1, 2, 3):
        a = np.array([attention(th[k]) for th in theta])  # (S, 96, 24)
        mean, lo, _, hi = _q(a)
        t, h = np.meshgrid(np.arange(96), np.arange(24), indexing="ij")
        maps[k] = pl.DataFrame({"slot": t.ravel(), "time": [f"{s // 4:02d}:{15 * (s % 4):02d}" for s in t.ravel()],
                                "hour": h.ravel(), "mean": mean.ravel(), "q05": lo.ravel(), "q95": hi.ravel()})
    gain, shape = [], []
    for fold, d in (*draws.items(), ("all", {"theta": theta, "w": np.concatenate([d["w"] * d["scale"] for d in draws.values()]), "scale": np.array(1.0)})):
        w_kw = d["w"] * d["scale"]  # (S, 2, 4) kW: on-channel = full-on step, log-channel = at q_max
        for k in (1, 2, 3):
            for ci, ch in enumerate(("on", "log")):
                gain.append((fold, k, ch, *map(float, _q(w_kw[:, ci, k]))))
            for name, v in (("sigma", np.exp(d["theta"][:, k, 0])), ("beta", np.exp(d["theta"][:, k, 1])), ("c_hours", d["theta"][:, k, 2])):
                shape.append((fold, k, name, *map(float, _q(v))))
    cols = ["fold", "kind", "channel", "mean", "q05", "median", "q95"]
    return (maps, pl.DataFrame(gain, orient="row", schema=[*cols[:3], "mean_kw", "q05_kw", "median_kw", "q95_kw"]),
            pl.DataFrame(shape, orient="row", schema=["fold", "kind", "param", *cols[3:]]))


def heatmap(table: pl.DataFrame, k: int, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 8))
    im = ax.imshow(table["mean"].to_numpy().reshape(96, 24), aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_yticks(range(0, 96, 8), [f"{s // 4:02d}:00" for s in range(0, 96, 8)])
    ax.set_xticks(range(0, 24, 2), [f"{h}" for h in range(0, 24, 2)])
    ax.set(xlabel="생산계획 시각 h (시)", ylabel="전력 슬롯 t (HH:MM)", title=f"BAT attention α(t,h) 사후평균 — 운전유형 {KIND_NAMES[k]}")
    fig.colorbar(im, ax=ax, label="α")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _set_font() -> None:
    import matplotlib
    from matplotlib import font_manager

    for name in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "Malgun Gothic"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            matplotlib.rcParams["font.family"] = name
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


def median_error_day(slots: pl.DataFrame) -> str:
    """Date whose daily MAE of y_median is closest to the median daily MAE (ties → earliest date)."""
    d = (slots.filter(pl.col("y_true").is_finite() & pl.col("y_median").is_finite())
         .group_by("date").agg((pl.col("y_median") - pl.col("y_true")).abs().mean().alias("mae")))
    return d.sort((pl.col("mae") - d["mae"].median()).abs(), "date").row(0, named=True)["date"]


def _mae(part: pl.DataFrame) -> float:
    ok = part.filter(pl.col("y_true").is_finite() & pl.col("y_median").is_finite())
    return float((ok["y_median"] - ok["y_true"]).abs().mean())  # type: ignore[arg-type]


def _forecast_ax(ax, bat: pl.DataFrame, m2: pl.DataFrame, bands: tuple[tuple[str, str], ...], labels: bool = False) -> None:  # type: ignore[no-untyped-def]
    """Actual (black), BAT median (blue) with quantile bands, M2 median (orange dashed) on one axis."""
    x = bat["datetime"].str.strptime(pl.Datetime, "%Y.%m.%d %H:%M:%S").to_numpy()
    for (lo, hi), alpha in zip(bands, (0.15, 0.3), strict=False):
        ax.fill_between(x, bat[lo].to_numpy(), bat[hi].to_numpy(), color="tab:blue", alpha=alpha, lw=0,
                        label=f"BAT {int(hi[1:]) - int(lo[1:])}% 구간")
    mae = {m: f" (MAE {_mae(f):.1f})" if labels else "" for m, f in (("BAT", bat), ("M2", m2))}
    ax.plot(x, bat["y_true"].to_numpy(), color="black", lw=1, label="실측")
    ax.plot(x, bat["y_median"].to_numpy(), color="tab:blue", lw=1.2, label="BAT 중앙값" + mae["BAT"])
    ax.plot(x, m2["y_median"].to_numpy(), color="tab:orange", lw=1.2, ls="--", label="M2 중앙값" + mae["M2"])
    ax.set_ylabel("kW")


def forecast_figures(val: pl.DataFrame, test: pl.DataFrame | None, ctx: pl.DataFrame, out: Path) -> dict[str, str]:
    """forecast_val.png (median-error operating day per fold), forecast_test.png, pred_vs_actual.png; returns picked days."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    optype = dict(ctx.filter(pl.col("q") == 0).select("date", "optype").iter_rows())
    pick = lambda f, m: f.filter(pl.col("model") == m).sort("datetime")  # noqa: E731
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    picked = {}
    for ax, (fold, part) in zip(axes.ravel(), sorted(val.partition_by("fold", as_dict=True).items()), strict=False):
        bat = pick(part, "BAT")
        day = picked[fold[0]] = median_error_day(bat.filter(pl.col("date").replace_strict(optype, default="k0") != "k0"))
        _forecast_ax(ax, bat.filter(pl.col("date") == day), pick(part, "M2").filter(pl.col("date") == day),
                     (("q05", "q95"), ("q25", "q75")), labels=True)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.set_title(f"{fold[0]} {day.replace('.', '-')} (가동유형 {KIND_NAMES[int(optype[day][1])]})")
        ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "forecast_val.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, m in zip(axes, ("BAT", "M2"), strict=True):
        ok = val.filter((pl.col("model") == m) & pl.col("y_true").is_finite() & pl.col("y_median").is_finite())
        ax.scatter(ok["y_true"], ok["y_median"], s=3, alpha=0.3, color="tab:blue" if m == "BAT" else "tab:orange")
        lim = [0, float(max(ok["y_true"].max(), ok["y_median"].max()))]  # type: ignore[arg-type]
        ax.plot(lim, lim, color="black", lw=0.8)
        ax.set(xlabel="실측 (kW)", ylabel="예측 중앙값 (kW)", title=f"{m} — 검증 f1–f4, MAE {_mae(ok):.2f} kW", aspect="equal")
    fig.tight_layout()
    fig.savefig(out / "pred_vs_actual.png", dpi=120)
    plt.close(fig)

    if test is not None:
        holidays = scenario.tariff_holidays()
        fig, ax = plt.subplots(figsize=(16, 5))
        bat = pick(test, "BAT")
        _forecast_ax(ax, bat, pick(test, "M2"), (("q05", "q95"),), labels=True)
        for d in sorted(set(bat["date"])):
            day = datetime.strptime(d, "%Y.%m.%d")
            if day.weekday() == 6 or day.date() in holidays:
                ax.axvspan(day, day.replace(hour=23, minute=59), color="grey", alpha=0.12, lw=0)
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.set(title="봉인 시험 구간 2021-09-01..14 (회색: 일요일·공휴일)")
        ax.margins(x=0)
        ax.legend(fontsize=8, loc="upper left", ncol=4)
        fig.tight_layout()
        fig.savefig(out / "forecast_test.png", dpi=120)
        plt.close(fig)
    return picked


def summary_md(err: pl.DataFrame, peaks: pl.DataFrame | None, fnfp: pl.DataFrame | None, shape: pl.DataFrame, gain: pl.DataFrame,
               slots: pl.DataFrame) -> str:
    ok = slots.filter(pl.col("model").is_in(["BAT", "M2"]) & pl.col("y_true").is_finite() & pl.col("y_median").is_finite())
    overall = dict(ok.group_by("model").agg((pl.col("y_median") - pl.col("y_true")).abs().mean()).iter_rows())
    L = [f"- 전체 슬롯 MAE: BAT {overall['BAT']:.2f} kW vs M2 {overall['M2']:.2f} kW (차이 {overall['BAT'] - overall['M2']:+.2f} kW, f1–f4 OOF 56일)."]
    bat = err.filter(pl.col("model") == "BAT")
    for cond, label in (("regime", "운전유형×슬롯 생산 on/off"), ("tou", "TOU 구간(0 경부하·1 중간·2 최대)")):
        part = bat.filter(pl.col("condition") == cond).sort("bin")
        cells = ", ".join(f"{b} {d:+.1f} (n={n})" for b, d, n in part.select("bin", "mae_minus_M2", "n").iter_rows())
        L.append(f"- {label}별 BAT−M2 MAE 차이(kW): {cells}.")
    hours = bat.filter(pl.col("condition") == "hour").sort("mae_minus_M2")
    best, worst = hours.row(0, named=True), hours.row(-1, named=True)
    L.append(f"- 시간대별: BAT 우위 최대 {best['bin']}시 ({best['mae_minus_M2']:+.1f} kW), 우위 최소(또는 열위) {worst['bin']}시 ({worst['mae_minus_M2']:+.1f} kW); "
             f"BAT 오차 최대 시간대 {bat.filter(pl.col('condition') == 'hour').sort('mae').row(-1, named=True)['bin']}시.")
    if peaks is not None:
        pooled = peaks.filter(pl.col("fold") == "all")
        L.append("- [보조 지표] 일 피크 초과 이벤트 분류 (본 과제는 전력 예측·조건 분석; 아래는 참고용, 항상경보 = 매일 경보 기준선):")
        for model in ("BAT", "M2", "always_alarm"):
            cells = ", ".join(f"{c} F1 {f1:.2f}/F2 {f2_:.2f} (P {p:.2f}/R {r:.2f}"
                              + ("" if model == "always_alarm" else f", AUC {a:.2f}, Brier {b:.3f}") + f", 사건 {n}/{d}일)"
                              for c, f1, f2_, p, r, a, b, n, d in pooled.filter(pl.col("model") == model)
                              .select("C", "f1", "f2", "precision", "recall", "auc", "brier", "n_events", "n_days").iter_rows())
            L.append(f"  - {'항상경보' if model == 'always_alarm' else model + ' (보정 위험 ≥ p*)'}: {cells}.")
    for kind in ("FN", "FP") if fnfp is not None else ():
        part = fnfp.filter((pl.col("model") == "BAT") & (pl.col("kind") == kind) & (pl.col("n") > 0)
                           & pl.col("condition").is_in(["optype", "weekday", "month", "prod_sum", "daytype"]))
        if part.is_empty():
            L.append(f"- [보조 지표] BAT {kind}: 해당 일 없음.")
            continue
        tot = part.group_by("C").agg((pl.col("n").sum() / pl.col("condition").n_unique()).alias("t"))
        top = (part.with_columns((pl.col("share") / pl.col("share_all")).alias("lift"))
               .filter(pl.col("n") >= 2).sort("lift", descending=True).head(4))
        cells = "; ".join(f"{c} {cond}={b}: {n}건, 비중 {s:.0%} (전체 {sa:.0%})"
                          for c, cond, b, n, s, sa in top.select("C", "condition", "bin", "n", "share", "share_all").iter_rows())
        L.append(f"- [보조 지표] BAT {kind} 집중 조건 (C별 {kind} 일수 {', '.join(f'{c}={t:.0f}' for c, t in tot.sort('C').iter_rows())}): {cells}.")
    s = shape.filter(pl.col("fold") == "all")
    for k in (1, 2, 3):
        get = {p: (m, lo, hi) for p, m, lo, hi in s.filter(pl.col("kind") == k).select("param", "median", "q05", "q95").iter_rows()}
        g = {c: (m, lo, hi) for c, m, lo, hi in gain.filter((pl.col("fold") == "all") & (pl.col("kind") == k))
             .select("channel", "median_kw", "q05_kw", "q95_kw").iter_rows()}
        L.append(f"- attention 유형 {KIND_NAMES[k]}: 지연 c {get['c_hours'][0]:+.2f}h [{get['c_hours'][1]:+.2f}, {get['c_hours'][2]:+.2f}], "
                 f"폭 σ {get['sigma'][0]:.2f} [{get['sigma'][1]:.2f}, {get['sigma'][2]:.2f}], 형상 β {get['beta'][0]:.2f}; "
                 f"이득 w_on {g['on'][0]:.1f} kW [{g['on'][1]:.1f}, {g['on'][2]:.1f}], w_log {g['log'][0]:.1f} kW [{g['log'][1]:.1f}, {g['log'][2]:.1f}] (90% 구간).")
    L.append("- 지연 c<0: 슬롯 t가 t+|c|시의 생산계획에 주목(전력이 계획 시각보다 먼저 움직임); c≈0은 동시각, σ가 클수록 넓은 시간 창.")
    return "# 3장 영향요인 및 오류분석 — 핵심 수치\n\n" + "\n".join(L) + "\n"


def main() -> None:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "results_v3")
    parser.add_argument("--folds", nargs="+", default=list(ev.FOLDS_CV))
    parser.add_argument("--n-iter", type=int, default=2000)
    parser.add_argument("--no-peak-class", action="store_true", help="skip the auxiliary peak-event classification outputs")
    parser.add_argument("--jobs", type=int, default=2, help="parallel BAT refits (keep total python processes ≤ 6)")
    a = parser.parse_args()
    out = a.results / "ch3"
    out.mkdir(parents=True, exist_ok=True)
    panel = features.load_panel()
    slots = pl.read_csv(a.results / "oof_slots.csv")
    days = pl.read_csv(a.results / "oof_days.csv")
    inner = pl.read_csv(a.results / "inner_days.csv")
    if "risk_platt_C50" not in days.columns:
        days, inner = ev.calibrate_oof(days, "platt", inner)
    pstar = ev.p_star_table(inner, days.select(ev.KEYS).unique().iter_rows())

    ctx = slot_context(panel)
    err = error_by_regime(slots, ctx)
    err = pl.concat([err, analysis.error_by_condition(slots.filter(pl.col("model").is_in(MODELS)), panel, MODELS)
                     .filter(pl.col("condition").is_in(["production", "temperature", "daytype"]))], how="diagonal_relaxed")
    err.write_csv(out / "error_by_regime.csv")
    peaks = fnfp = None
    if not a.no_peak_class:
        peaks = peak_events(days, pstar)
        peaks.write_csv(out / "peak_events.csv")
        fnfp = fn_fp_conditions(days, panel, pstar)
        fnfp.write_csv(out / "fn_fp_conditions.csv")

    maps, gain, shape = attention_tables(bat_draws(out, tuple(a.folds), a.n_iter, a.jobs))
    _set_font()
    for k, table in maps.items():
        table.write_csv(out / f"attention_k{k}.csv")
        heatmap(table, k, out / f"attention_k{k}.png")
    gain.write_csv(out / "transfer_gain.csv")
    shape.write_csv(out / "attention_shape.csv")
    final = a.results / "final" / "slots.csv"
    picked = forecast_figures(slots, pl.read_csv(final) if final.exists() else None, ctx, out)
    print("[report_v3] forecast_val days:", picked)
    (out / "summary.md").write_text(summary_md(err, peaks, fnfp, shape, gain, slots))
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
