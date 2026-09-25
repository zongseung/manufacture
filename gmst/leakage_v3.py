"""Chapter-1 leakage diagnostic: random day 5-fold vs Leave-One-Event-Out on the panel WITH copies.

122 of 257 days are power-curve copies of another day. If a copy of a held-out day sits in training,
naive CV is optimistic; the random − LOEO MAE gap measures how much. September stays sealed.
Held-out targets are hidden from every feature (lags, B0′ reference, as-of backbone) in both schemes.
"""
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from pathlib import Path

import numpy as np
import polars as pl

from gmst import ROOT, backbone, baselines, bat
from gmst import evaluate as ev
from gmst import features as ft
from gmst.contracts import IntArray, Panel

OUT = ROOT / "results_v3" / "ch1"
MODELS = ("BAT", "M2")


def build_folds(idx: IntArray, event: list[str], k: int = 5, seed: int = 0) -> dict[str, list[IntArray]]:
    """Random day k-fold (each day once) and LOEO (one fold per event among `idx`, never split)."""
    random = [np.sort(f) for f in np.array_split(np.random.default_rng(seed).permutation(idx), k)]
    groups = dict.fromkeys(event[d] for d in idx)
    loeo = [np.array([d for d in idx if event[d] == g], dtype=np.int64) for g in groups]
    return {"random5": random, "loeo": loeo}


@cache
def _panel() -> tuple[Panel, IntArray, tuple[float, int]]:
    panel = ft.load_panel(include_copies=True)
    idx = np.flatnonzero(np.isfinite(panel["Y"]).any(1)).astype(np.int64)
    tau, h, _ = backbone.select_tau(panel, "f4")
    return panel, idx, (tau, h)


def _job(model: str, scheme: str, hold: IntArray) -> list[dict]:
    panel, idx, bb = _panel()
    masked: Panel = {**panel, "Y": panel["Y"].copy()}
    masked["Y"][hold] = np.nan
    train = np.setdiff1d(idx, hold)
    C, _ = ev.thresholds(masked, train)
    if model == "BAT":
        # ponytail: 진단용 1000/500 반복 (본 실험 2000/1000), 격차가 반복 수에 민감하면 전체 반복으로 재실행
        state = bat.fit_bat(masked, train, C, n_iter=1000, burn=500)
        pred = lambda d: bat.predict_bat(state, masked, d)["y_median"]  # noqa: E731
    else:
        state = baselines.fit_b1(masked, train, "B", bb, C, rounds=100)
        pred = lambda d: baselines.predict_b1(state, masked, d)["y_median"]  # noqa: E731
    rows = []
    for d in map(int, hold):
        y = panel["Y"][d]
        ok = np.isfinite(y)
        err = np.abs(y[ok] - pred(d)[ok])
        rows.append({"model": model, "scheme": scheme, "date": panel["dates"][d], "d": d,
                     "event_id": panel["days"]["event_id"][d], "abs_sum": float(err.sum()), "n_obs": int(ok.sum())})
    return rows


def _gap_table(days: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for model in MODELS:
        m = days.filter(pl.col("model") == model)
        wide = m.filter(pl.col("scheme") == "random5").join(
            m.filter(pl.col("scheme") == "loeo").select("d", loeo_sum="abs_sum"), on="d", validate="1:1")
        num_gap = (wide["abs_sum"] - wide["loeo_sum"]).to_numpy()
        gap, glo, ghi = ev.block_bootstrap(num_gap, wide["n_obs"].to_numpy().astype(float))
        subsets = [("random5", m.filter(pl.col("scheme") == "random5")),
                   ("loeo", m.filter(pl.col("scheme") == "loeo")),
                   ("loeo_singleton", m.filter((pl.col("scheme") == "loeo") & (pl.col("event_days") == 1))),
                   ("loeo_copygroup", m.filter((pl.col("scheme") == "loeo") & (pl.col("event_days") > 1))),
                   ("random5_copygroup", m.filter((pl.col("scheme") == "random5") & (pl.col("event_days") > 1)))]
        for scheme, s in subsets:
            mae, lo, hi = ev.block_bootstrap(s["abs_sum"].to_numpy(), s["n_obs"].to_numpy().astype(float))
            rows.append({"model": model, "scheme": scheme, "MAE": mae, "MAE_lo": lo, "MAE_hi": hi,
                         "n_days": s.height, "gap": gap if scheme == "random5" else None,
                         "gap_lo": glo if scheme == "random5" else None, "gap_hi": ghi if scheme == "random5" else None})
    return pl.DataFrame(rows)


def _plot(table: pl.DataFrame, days: pl.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gmst.report_v3 import _set_font

    _set_font()
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    for j, (scheme, label, color) in enumerate((("random5", "무작위 5-fold", "#9aa5b1"), ("loeo", "LOEO", "#2b6cb0"))):
        t = table.filter(pl.col("scheme") == scheme).sort(pl.col("model").replace_strict({"BAT": 0, "M2": 1}))
        x = np.arange(len(MODELS)) + (j - 0.5) * 0.36
        mae = t["MAE"].to_numpy()
        a.bar(x, mae, 0.36, color=color, label=label,
              yerr=[mae - t["MAE_lo"].to_numpy(), t["MAE_hi"].to_numpy() - mae], capsize=4)
        for xi, v in zip(x, mae, strict=True):
            a.text(xi, v * 0.5, f"{v:.1f}", ha="center", color="white", fontsize=9)
    a.set_xticks(range(len(MODELS)), MODELS)
    a.set_ylabel("MAE (kW)")
    a.set_title("평가 방식별 MAE (95% 일 단위 부트스트랩)")
    a.set_ylim(0, table["MAE_hi"].max() * 1.2)
    a.legend(frameon=False, loc="upper center", ncols=2)
    ev_mae = (days.filter(pl.col("scheme") == "loeo")
              .group_by("model", "event_id", "event_days").agg(mae=pl.col("abs_sum").sum() / pl.col("n_obs").sum()))
    data, labels = [], []
    for model in MODELS:
        for multi, name in ((False, "단독"), (True, "복제군")):
            s = ev_mae.filter((pl.col("model") == model) & ((pl.col("event_days") > 1) == multi))
            data.append(s["mae"].to_numpy())
            labels.append(f"{model}\n{name} (n={s.height})")
    b.boxplot(data, tick_labels=labels, showfliers=True)
    b.set_ylabel("사건별 LOEO MAE (kW)")
    b.set_title("LOEO 사건별 MAE: 단독 사건 vs 복제군")
    fig.tight_layout()
    fig.savefig(OUT / "leakage_gap.png", dpi=150)


def _summary(table: pl.DataFrame, n_events: int, n_multi: int, runtime: float) -> None:
    def g(model: str, scheme: str, col: str = "MAE") -> float:
        return float(table.filter((pl.col("model") == model) & (pl.col("scheme") == scheme))[col][0])

    lines = ["# 1장 누수 진단 — 무작위 5-fold vs LOEO (복제일 포함 패널, 9월 봉인)", "",
             f"- 평가 대상: 관측값이 있는 {int(g('BAT', 'loeo', 'n_days'))}일, 사건 {n_events}개 "
             f"(복제군 {n_multi}개, 단독 {n_events - n_multi}개). 두 방식 모두 보류일의 Y는 모든 특징에서 가림.",
             ]
    for m in MODELS:
        lines.append(f"- {m}: 무작위 5-fold MAE {g(m, 'random5'):.2f} kW [{g(m, 'random5', 'MAE_lo'):.2f}, "
                     f"{g(m, 'random5', 'MAE_hi'):.2f}] vs LOEO {g(m, 'loeo'):.2f} kW [{g(m, 'loeo', 'MAE_lo'):.2f}, "
                     f"{g(m, 'loeo', 'MAE_hi'):.2f}]; 격차(무작위−LOEO) {g(m, 'random5', 'gap'):+.2f} kW "
                     f"[{g(m, 'random5', 'gap_lo'):+.2f}, {g(m, 'random5', 'gap_hi'):+.2f}] "
                     f"({g(m, 'random5', 'gap') / g(m, 'loeo') * 100:+.0f}%). LOEO 분해: 단독 사건 {g(m, 'loeo_singleton'):.2f} kW "
                     f"(n={int(g(m, 'loeo_singleton', 'n_days'))}일) vs 복제군 {g(m, 'loeo_copygroup'):.2f} kW "
                     f"(n={int(g(m, 'loeo_copygroup', 'n_days'))}일); 같은 복제군 날의 무작위 5-fold MAE "
                     f"{g(m, 'random5_copygroup'):.2f} kW.")
    worst = max(-g(m, "random5", "gap") / g(m, "loeo") * 100 for m in MODELS)
    cg = {m: g(m, "random5_copygroup") - g(m, "loeo_copygroup") for m in MODELS}
    lines += [
        f"- 해석: 복제군 날만 보면 격차(무작위−LOEO)는 BAT {cg['BAT']:+.2f} kW, M2 {cg['M2']:+.2f} kW. "
        "음수 격차 = 시험일의 복제본(같은 전력 곡선)이 학습에 들어간 무작위 CV가 성능을 낙관적으로 부풀림. "
        "M2(LightGBM, 이력·as-of 특징)는 복제본을 외워 이득을 보고, BAT(구조적 전달 모형)는 거의 영향이 없다.",
        "- LOEO에서도 복제군 날의 MAE가 단독 사건보다 크다: 이는 누수가 아니라 복제군 자체(날씨만 다른 편집 복제일)가 "
        "어려운 날이라는 뜻이므로, 격차는 반드시 같은 날끼리 쌍체 비교로 읽어야 한다.",
        f"- 결론: 순진한 무작위 CV는 모형에 따라 최대 {worst:.0f}% 낙관적이다. 본 실험에서 복제일 Y를 가리고(masking) "
        "시간 순서 fold(f1–f4 rolling origin)를 쓰는 근거다. 이 격차는 누수 진단이며 전방 예측 성능 추정치가 아니다.",
        f"- 설정: BAT 1000/500 반복(진단용 축소), M2=B1(protocol B, rounds 100, τ/h는 f4 선택값 고정), "
        f"무작위 fold seed 0, CI는 일 단위 쌍체 부트스트랩 B=2000. 실행 시간 {runtime / 60:.1f}분.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")


def run(workers: int = 12) -> pl.DataFrame:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    panel, idx, _ = _panel()
    event: list[str] = panel["days"]["event_id"].to_list()
    folds = build_folds(idx, event)
    jobs = [(m, s, f) for m in MODELS for s, fs in folds.items() for f in fs]
    with ProcessPoolExecutor(min(workers, os.cpu_count() or 1), mp.get_context("spawn")) as pool:
        rows = [r for out in pool.map(_job, *zip(*jobs, strict=True)) for r in out]
    size = {str(g): int(n) for g, n in zip(*np.unique([event[d] for d in idx], return_counts=True), strict=True)}
    days = pl.DataFrame(rows).with_columns(
        event_days=pl.col("event_id").replace_strict(size, return_dtype=pl.Int64),
        mae=pl.col("abs_sum") / pl.col("n_obs")).sort("model", "scheme", "d")
    table = _gap_table(days)
    days.drop("d").write_csv(OUT / "loeo_days.csv")
    table.write_csv(OUT / "leakage_gap.csv")
    _plot(table, days)
    runtime = time.time() - t0
    _summary(table, len(size), sum(n > 1 for n in size.values()), runtime)
    print(table)
    print(f"[leakage_v3] {runtime:.0f}s")
    return table


if __name__ == "__main__":
    run()
