"""Report tables T1–T7 from saved dev OOF predictions (no refits, September stays sealed)."""
import json
from datetime import date, datetime

import numpy as np
import polars as pl

from gmst import ROOT, features
from gmst import evaluate as ev
from gmst.contracts import Panel
from gmst.report_v3 import fn_fp_conditions, peak_events

R = ROOT / "results_v3"
OUT = R / "report_tables"
CBAT = "BAT_conditional_gaussian_fixed_attention"
SOURCES = {  # label -> (result dir with seed0-2, model name)
    "C-BAT": ("attention_ablation", CBAT),
    "BAT": ("attention_ablation", "BAT"),
    "B0_kind": ("attention_ablation", "B0_kind"),
    "LightGBM(튜닝)": ("attention_ablation", "M2_tuned"),
    "B0": ("attention_ablation", "B0"),
    "학습형 어텐션": ("attention_ablation", "BAT_conditional_gaussian"),
    "공통 잡음(학습형 어텐션)": ("conditional_collapsed", "BAT_pooled_gaussian"),
    "종류 기준 ref": ("ref_ablation", CBAT + "_kindref"),
    "BAT(전이 없음)": ("attention_ablation", "BAT_no_transfer"),
}
MAIN = ("C-BAT", "BAT", "B0_kind", "LightGBM(튜닝)", "B0")
SEEDS = (0, 1, 2)
B = 4000


def load(label: str) -> list[tuple[pl.DataFrame, pl.DataFrame]]:
    folder, model = SOURCES[label]
    return [tuple(pl.read_csv(R / folder / f"seed{s}" / f"oof_{k}.csv").filter(pl.col("model") == model)
                  for k in ("slots", "days")) for s in SEEDS]


def metrics(label: str, panel: Panel) -> pl.DataFrame:
    """Seed-mean pooled metrics exactly as gmst.rescore, plus slot interval widths."""
    frames = []
    for slots, days in load(label):
        m = ev.point_metrics(slots, days, panel)
        high = ev.point_metrics(slots.head(0), days.filter(pl.col("M_true") > pl.col("C90")), panel)
        widths = [(f"width{p}", float(slots.filter(pl.col("y_true").is_finite() & pl.col(lo).is_finite())
                                      .select((pl.col(hi) - pl.col(lo)).mean()).item() or np.nan))
                  for p, lo, hi in (("50", "q25", "q75"), ("80", "q10", "q90"), ("90", "q05", "q95"))]
        frames += [m.filter(pl.col("fold") == "pooled"),
                   high.filter((pl.col("fold") == "pooled") & (pl.col("stratum") == "all")).with_columns(stratum=pl.lit("high")),
                   pl.DataFrame([(w, v) for w, v in widths], schema=["metric", "value"], orient="row")
                   .with_columns(stratum=pl.lit("all"), n=pl.lit(0, pl.Int64))]
    return (pl.concat(frames, how="diagonal_relaxed").group_by("stratum", "metric")
            .agg(pl.col("value").mean(), pl.col("n").max()).with_columns(label=pl.lit(label)))


def paired_ci(num: np.ndarray, den: np.ndarray, dates: list[date], calendar: list[date]) -> dict[str, float]:
    """Day bootstrap on the given days, plus 7-day blocks on the full calendar with zero-filled gaps."""
    diff, lo, hi = ev.block_bootstrap(num, den, B=B, seed=0)
    pos = {d: i for i, d in enumerate(calendar)}
    cal_num, cal_den = np.zeros(len(calendar)), np.zeros(len(calendar))
    for d, n, w in zip(dates, num, den, strict=True):
        cal_num[pos[d]], cal_den[pos[d]] = n, w
    _, blo, bhi = ev.block_bootstrap(cal_num, cal_den, B=B, seed=0, block_length=7)
    return {"diff": diff, "ci_lo": lo, "ci_hi": hi, "block7_lo": blo, "block7_hi": bhi, "n_days": len(dates)}


def _day(text: str) -> date:
    return datetime.strptime(text.replace("-", "."), "%Y.%m.%d").date()


def paired(a: str, b: str, panel: Panel) -> list[dict[str, object]]:
    """Seed-average day loss differences first (a minus b), then bootstrap."""
    per_seed = []
    for (sa, da), (sb, db) in zip(load(a), load(b), strict=True):
        loss = ev.day_losses(sa, da, panel).join(ev.day_losses(sb, db, panel), on=["fold", "date"], suffix="_b", validate="1:1")
        peaks = da.join(db, on=["fold", "date"], suffix="_b", validate="1:1").filter(pl.col("usable_peak"))
        per_seed.append((loss.with_columns(mae=pl.col("ae_sum") - pl.col("ae_sum_b"), crps=pl.col("crps_sum") - pl.col("crps_sum_b")),
                         peaks.with_columns(peak=(pl.col("M_hat_median") - pl.col("M_true")).abs()
                                            - (pl.col("M_hat_median_b") - pl.col("M_true")).abs())))
    loss = pl.concat([x for x, _ in per_seed]).group_by("date").agg(
        pl.col("mae", "crps").mean(), pl.col("n_pts", "n_crps", "n_crps_b").first()).sort("date")
    peaks = pl.concat([y for _, y in per_seed]).group_by("date").agg(
        pl.col("peak").mean(), (pl.col("M_true") > pl.col("C90")).first().alias("high")).sort("date")
    calendar = sorted({_day(d) for d in pl.read_csv(R / "attention_ablation/seed0/oof_days.csv")["date"]})
    calendar = [date.fromordinal(o) for o in range(calendar[0].toordinal(), calendar[-1].toordinal() + 1)]
    rows = [("slot_mae", loss["mae"].to_numpy(), loss["n_pts"].to_numpy().astype(float), loss["date"])]
    if (loss["n_crps"] == loss["n_crps_b"]).all() and loss["n_crps"].sum() > 0:
        rows.append(("slot_crps", loss["crps"].to_numpy(), loss["n_crps"].to_numpy().astype(float), loss["date"]))
    for name, part in (("peak_mae_all", peaks), ("peak_mae_high", peaks.filter("high"))):
        rows.append((name, part["peak"].to_numpy(), np.ones(part.height), part["date"]))
    return [{"comparison": f"{a} − {b}", "metric": name, **paired_ci(num, den, [_day(d) for d in dates], calendar)}
            for name, num, den, dates in rows]


def alarms(panel: Panel) -> tuple[pl.DataFrame, pl.DataFrame]:
    """report_v3.peak_events / fn_fp_conditions per seed at inner-fold p*, then seed mean."""
    models = (CBAT, "BAT", "M2_tuned")
    events, conds = [], []
    for s in SEEDS:
        days = pl.read_csv(R / f"attention_ablation/seed{s}/oof_days.csv")
        inner = pl.read_csv(R / f"attention_ablation/seed{s}/inner_days.csv")
        pstar = ev.p_star_table(inner, days.select(ev.KEYS).unique().iter_rows())
        events.append(peak_events(days, pstar, models).filter(pl.col("fold") == "all"))
        conds.append(fn_fp_conditions(days, panel, pstar, (CBAT,)).filter(pl.col("C") == "C90"))
    ev_mean = pl.concat(events).group_by("model", "C", maintain_order=True).agg(pl.exclude("fold").mean())
    cond = pl.concat(conds).group_by("condition", "kind", "bin").agg(
        pl.col("n", "share", "n_all", "share_all").mean()).sort("condition", "kind", "bin")
    return ev_mean, cond


def _f(x: object, d: int = 2) -> str:
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}" if isinstance(x, float) else str(x)


def md_table(frame: pl.DataFrame, d: int = 2) -> str:
    head = "| " + " | ".join(frame.columns) + " |\n|" + "---|" * frame.width + "\n"
    return head + "".join("| " + " | ".join(_f(v, d) for v in row) + " |\n" for row in frame.iter_rows())


NOISE = {"대조(C-BAT)": CBAT, "kind": CBAT + "_noisekind", "slot": CBAT + "_noiseslot",
         "t": CBAT + "_tinnov", "kind+t": CBAT + "_noisekind_tinnov"}  # label -> model in results_v3/noise_ablation


def noise_ablation(folder: str = "noise_ablation", nu: float = 5.) -> str:
    """Noise-variant table, paired bootstrap vs control, noise posteriors and the pre-registered verdict."""
    out = R / folder
    SOURCES.update({label: (folder, model) for label, model in NOISE.items()})
    panel = features.load_panel()
    observed = pl.read_csv(R / "peak_conditions_v2/observed_days.csv").select("fold", "date", "kind", "high")
    strata = {"all": pl.lit(True), "nonop": pl.col("kind") == 0, "op": pl.col("kind") > 0, "high": pl.col("high")}
    convergence = pl.read_csv(out / "convergence_summary.csv")
    runtime = pl.DataFrame([json.loads(p.read_text()) for p in (out / "jobs").glob("*/runtime.json")]).select("model", "fit_seconds")
    rows, posterior = [], []
    for label, model in NOISE.items():
        m = metrics(label, panel).filter(pl.col("stratum") == "all")
        value = dict(zip(m["metric"], m["value"], strict=True))
        peaks = pl.concat([days.filter(pl.col("usable_peak")).join(observed, on=["fold", "date"], validate="1:1")
                           .with_columns(err=pl.col("M_hat_median") - pl.col("M_true"), seed=pl.lit(s))
                           for s, (_, days) in zip(SEEDS, load(label), strict=True)])
        assert peaks.height == 51 * len(SEEDS)
        row = {"label": label, "model": model, **{k: value[k] for k in ("mae", "rmse", "crps", "cov50", "width50", "cov90", "width90")}}
        for name, predicate in strata.items():  # median-prediction error, seed mean
            part = peaks.filter(predicate).group_by("seed").agg(mae=pl.col("err").abs().mean(), bias=pl.col("err").mean(), n=pl.len())
            row |= {f"peak_mae_{name}": part["mae"].mean(), f"peak_bias_{name}": part["bias"].mean(), f"n_{name}": part["n"].max()}
        row |= {"max_rhat": convergence.filter(pl.col("model") == model)["max_basic_split_rhat"].max(),
                "fit_seconds_median": runtime.filter(pl.col("model") == model)["fit_seconds"].median()}
        rows.append(row)
        trace = pl.concat([pl.read_csv(out / "jobs" / f"seed{s}_{f}_{model}" / "posterior_trace.csv")
                           for s in SEEDS for f in ev.FOLDS_CV]).select(pl.col("^(sigma_u_kw|sigma_eta_kw|rho)\\[.*$"))
        scale = np.sqrt(nu / (nu - 2)) if model.endswith("_tinnov") else 1.
        posterior += [{"label": label, "parameter": c, "median": trace[c].median(),
                       "innovation_sd_median": trace[c].median() * scale if c.startswith("sigma_eta") else None}
                      for c in trace.columns]
    table = pl.DataFrame(rows)
    control = table.row(0, named=True)
    verdict = table.slice(1).select(
        "label", "model", "peak_mae_all",
        a_op_bias=pl.col("peak_bias_op").abs() < abs(control["peak_bias_op"]),
        b_cov90=(pl.col("cov90") - .9).abs() < abs(control["cov90"] - .9),
        c_high=pl.col("peak_mae_high") - control["peak_mae_high"] <= .5,
        d_rhat=pl.col("max_rhat") <= 1.05).with_columns(passed=pl.all_horizontal("a_op_bias", "b_cov90", "c_high", "d_rhat"))
    best = verdict.filter("passed").sort("peak_mae_all")["label"].head(1)
    verdict = verdict.with_columns(decision=pl.when(pl.col("label").is_in(best)).then(pl.lit("채택 후보")).otherwise(pl.lit("")))
    boot = pl.DataFrame([r for label in list(NOISE)[1:] for r in paired(label, "대조(C-BAT)", panel)])
    for name, frame in (("noise_table", table), ("verdict", verdict), ("paired_bootstrap", boot),
                        ("noise_posterior", pl.DataFrame(posterior))):
        frame.write_csv(out / f"{name}.csv")
    return f"채택 후보: {best[0]} ({NOISE[best[0]]})" if len(best) else "채택 없음"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = features.load_panel()
    long = pl.concat([metrics(label, panel) for label in SOURCES])
    long.write_csv(OUT / "metrics_long.csv")

    def pick(labels: tuple[str, ...], stratum: str, cols: list[str]) -> pl.DataFrame:
        wide = long.filter(pl.col("stratum") == stratum).pivot(on="metric", index="label", values="value")
        return pl.DataFrame({"label": labels}).join(wide, on="label", how="left").select("label", *cols)

    t1 = pick(MAIN, "all", ["mae", "rmse", "r2", "wape_pct", "crps", "peak_mae", "peak_rmse", "peak_r2", "peak_hit2"])
    t2 = pl.concat([pick(MAIN[:4], s, ["peak_mae", "peak_bias"]).with_columns(stratum=pl.lit(s))
                    for s in ("all", "nonop", "op", "high")]).pivot(on="stratum", index="label", values=["peak_mae", "peak_bias"])
    t3 = pl.DataFrame([r for b in ("BAT", "B0_kind", "LightGBM(튜닝)", "학습형 어텐션") for r in paired("C-BAT", b, panel)])
    rhat = {label: pl.read_csv(R / SOURCES[label][0] / "convergence_summary.csv").filter(pl.col("model") == SOURCES[label][1])
            ["max_basic_split_rhat"].max() for label in ("C-BAT", "학습형 어텐션", "공통 잡음(학습형 어텐션)", "종류 기준 ref")}
    t4 = pick(("C-BAT", "학습형 어텐션", "공통 잡음(학습형 어텐션)", "종류 기준 ref", "BAT", "BAT(전이 없음)"), "all",
              ["mae", "rmse", "crps", "peak_mae"]).join(pick(tuple(SOURCES), "high", ["peak_mae"]).rename({"peak_mae": "peak_mae_high"}),
                                                        on="label", how="left").with_columns(
        max_split_rhat=pl.col("label").replace_strict(rhat, default=None, return_dtype=pl.Float64))
    t5, t5c = alarms(panel)
    t5 = t5.with_columns(pl.col("model").replace({CBAT: "C-BAT", "M2_tuned": "LightGBM(튜닝)"}))
    t5c = t5c.filter(pl.col("condition").is_in(["optype", "weekday", "month", "prod_sum"]) & (pl.col("kind") != "all"))
    peak_int = pl.read_csv(R / "attention_ablation/peak_interval_summary.csv").filter((pl.col("model") == CBAT) & (pl.col("stratum") == "all"))
    t6 = pick(("C-BAT",), "all", ["cov50", "width50", "cov80", "width80", "cov90", "width90"]).with_columns(
        peak_cov90=pl.lit(peak_int["coverage90"].item()), peak_width90=pl.lit(peak_int["width90_kw"].item()))
    t7 = (pl.read_csv(R / "attention_ablation/convergence_summary.csv").filter(pl.col("model") == CBAT)
          .select("fold", "stage", "n_parameters", "max_basic_split_rhat", "n_flagged").sort("fold", "stage"))
    for name, t in (("t1_main", t1), ("t2_peak_strata", t2), ("t3_paired_bootstrap", t3), ("t4_ablation", t4),
                    ("t5_alarms", t5), ("t5_fnfp_c90", t5c), ("t6_intervals", t6), ("t7_convergence", t7)):
        t.write_csv(OUT / f"{name}.csv")
    notes = {
        "T1": "개발 구간 f1–f4 합산(53일, 5,016슬롯, 피크 51일). MAE·WAPE는 중앙값, RMSE·R²는 평균 예측. 피크는 경로 최댓값(중앙값으로 MAE, 평균으로 RMSE·R²). 피크 시각 적중은 ±2슬롯(±30분). 확률 모델은 seed 0–2 평균.",
        "T2": "비가동 14일, 가동 37일, 고피크 13일(관측 피크 > 학습 fold C90). 편향은 예측 평균 − 관측(kW).",
        "T3": f"차이 = C-BAT − 비교 모델(음수가 C-BAT 유리). seed별 날짜 손실 차이를 먼저 평균하고 날짜 단위 재표집(B={B}, seed 0). 슬롯 지표는 날짜별 손실 합과 슬롯 수를 함께 재표집. 7일 블록은 56일 달력에서 빈 날을 0으로 채움. 학습형 어텐션 행은 기존 문서 검증용.",
        "T4": "C-BAT에서 한 가지씩 바꾼 모델. 공통 잡음 비교는 학습형 어텐션에서만 실행되어, 공통 잡음 행은 C-BAT이 아니라 '학습형 어텐션' 행과 비교해야 한다. 원래 BAT(라플라스, 학습형, 공통 잡음)와 전이 없는 BAT은 split-Rhat 기록이 없다.",
        "T5": "report_v3.peak_events 경로: Platt 보정 위험 ≥ 같은 fold 내부 p*(F1 최대), 전체 f1–f4 합산, seed 평균. always_alarm은 모든 날 경보.",
        "T5b": "C-BAT C90 경보의 FN·FP 날짜가 몰린 조건(seed 평균 건수, share=FN 또는 FP 중 비중, share_all=전체 51일 중 비중). prod_sum은 생산량 합 사분위(0 동점으로 q1이 q2에 합쳐짐).",
        "T6": "슬롯 구간은 분위수 q25–q75, q10–q90, q05–q95. 일 피크 90% 구간은 peak_interval_summary(51일, seed 평균).",
        "T7": "기본(classical) split-Rhat, seed 3개 체인. 상수 모수 제외. rank-normalized·tail ESS 진단 아님.",
    }
    tables = [("T1 주요 비교", t1, 3), ("T2 피크 조건별 MAE·편향", t2, 2), ("T3 짝지은 bootstrap", t3, 3),
              ("T4 절제 비교", t4, 3), ("T5 피크 경보", t5.select("model", "C", "n_events", "precision", "recall", "f1", "auc", "brier", "p_star_mean"), 3),
              ("T5b C-BAT C90 FN/FP 조건", t5c.select("condition", "kind", "bin", "n", "share", "share_all"), 2),
              ("T6 구간 적중률·폭", t6, 3), ("T7 수렴(C-BAT)", t7, 4)]
    text = "# 보고서 표 (개발 구간, 9월 봉인 유지)\n\n" + "".join(
        f"## {title}\n\n{notes[title.split()[0]]}\n\n{md_table(t, d)}\n" for title, t, d in tables)
    (OUT / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
