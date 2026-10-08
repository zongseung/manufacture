"""Model-free analog-day comparison: observed peaks of similar operating days whose start differs.

Pairs observed operating days before 2021-09-01 (September sealed, DC1 copies / DC14 suspect days
already masked by features.load_panel) with the same operating kind, date type (weekday/Sat, no
holidays) and daily production within ±10%. Observational association only, not a causal effect.
"""
import argparse
from datetime import date
from itertools import combinations
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl

from gmst import ROOT, bat, features
from gmst import evaluate as ev
from gmst.contracts import Panel
from gmst.scenario import _timestamps, billing_demand, tariff_holidays

CUTOFF: Final = date(2021, 9, 1)
TOL: Final = 0.10
SHARE_GAP: Final = 0.05  # 첫 2개 생산시간 비중 차이 ≥ 5%p 쌍만 고/저 비교


def day_table(panel: Panel) -> pl.DataFrame:
    holidays = tariff_holidays()
    Q = np.nan_to_num(panel["X"]["생산량"][:, ::4])
    ok = (features.usable_peak(panel) & (panel["op"] == 1) & (Q.sum(1) > 0) & (panel["dtype"] < 2)
          & (panel["hol"] == 0) & (np.array(panel["dates"]) < CUTOFF))
    idx = np.flatnonzero(ok)
    start = (Q[idx] > 0).argmax(1)
    Y = panel["Y"][idx]
    return pl.DataFrame({
        "date": [panel["dates"][d] for d in idx], "kind": bat.kinds(panel, idx), "dtype": panel["dtype"][idx],
        "total": Q[idx].sum(1), "start": start,
        "start_share": [Q[d, s:s + 2].sum() / Q[d].sum() for d, s in zip(idx, start, strict=True)],
        "peak_band": [billing_demand(Y[i], _timestamps(panel["dates"][d]), holidays) for i, d in enumerate(idx)],
        "peak_all": np.nanmax(Y, 1), "peak_hour": np.nanargmax(Y, 1) // 4,
    })


def pairs(days: pl.DataFrame, variant: str) -> pl.DataFrame:
    """Ordered pairs (a = earlier start / lower share, b = later start / higher share); diffs are b − a."""
    rows = []
    for a, b in combinations(days.iter_rows(named=True), 2):
        if a["kind"] != b["kind"] or a["dtype"] != b["dtype"] or abs(a["total"] - b["total"]) > TOL * min(a["total"], b["total"]):
            continue
        if variant == "start":
            if abs(a["start"] - b["start"]) < 1:
                continue
            x = b["start"] - a["start"]
        else:
            if a["start"] != b["start"] or abs(a["start_share"] - b["start_share"]) < SHARE_GAP:
                continue
            x = b["start_share"] - a["start_share"]
        if x < 0:
            a, b, x = b, a, -x
        rows.append({"variant": variant, "date_a": a["date"], "date_b": b["date"], "kind": a["kind"], "dtype": a["dtype"],
                     "start_a": a["start"], "start_b": b["start"], "share_a": a["start_share"], "share_b": b["start_share"],
                     "dx": float(x), "d_peak_band": b["peak_band"] - a["peak_band"], "d_peak_all": b["peak_all"] - a["peak_all"],
                     "peak_hour_a": a["peak_hour"], "peak_hour_b": b["peak_hour"]})
    return pl.DataFrame(rows)


def summarize(p: pl.DataFrame, unit: float) -> list[dict]:
    """Per outcome: mean/median diff and Σdiff/Σ(dx/unit) with pair-bootstrap 95% CI (ev.block_bootstrap)."""
    out = []
    dx = p["dx"].to_numpy() / unit
    for col in ("d_peak_band", "d_peak_all"):
        y = p[col].to_numpy()
        mean, lo, hi = ev.block_bootstrap(y, np.ones(len(y)))
        rate, rlo, rhi = ev.block_bootstrap(y, dx)
        out.append({"variant": p["variant"][0], "outcome": col, "n_pairs": p.height,
                    "n_days": len(set(p["date_a"]) | set(p["date_b"])), "mean": mean, "mean_lo95": lo, "mean_hi95": hi,
                    "median": float(np.median(y)), "median_per_unit": float(np.median(y / dx)),
                    "per_unit": rate, "per_unit_lo95": rlo, "per_unit_hi95": rhi,
                    "peak_hour_later_frac": float((p["peak_hour_b"] > p["peak_hour_a"]).mean()),
                    "peak_hour_shift_mean": float((p["peak_hour_b"] - p["peak_hour_a"]).mean())})
    return out


def summary_md(days: pl.DataFrame, res: pl.DataFrame) -> str:
    f = lambda v, o: res.filter((pl.col("variant") == v) & (pl.col("outcome") == o)).row(0, named=True)
    lines = ["# 유사일 실측 비교 (모델 없음, 관측 연관성)", "",
             f"- 대상: 2021-09-01 이전 관측 가동일 {days.height}일(평일·토요일, 공휴일 제외, 복사일·의심일·9월 봉인 구간 제외, 결측 슬롯 ≤ 4).",
             f"- 짝 조건: 같은 가동유형, 같은 날짜유형, 일 생산량 차이 ±{TOL:.0%} 이내. 부트스트랩은 쌍 단위 재표집 2000회 95% 구간이다. 한 날이 여러 쌍에 들어가므로 쌍끼리 독립이 아니고, 구간은 실제보다 좁게 나올 수 있다."]
    for v, head, unit in (("start", "시작 시각이 1시간 이상 다른 쌍 (b = 늦게 시작한 날)", "1시간 늦을 때"),
                          ("share", f"시작 시각은 같고 첫 2개 생산시간 비중이 {SHARE_GAP:.0%}p 이상 다른 쌍 (b = 비중 높은 날)", "비중 10%p 높을 때")):
        if not res.filter(pl.col("variant") == v).height:
            lines.append(f"- {head}: 조건을 만족하는 쌍 없음.")
            continue
        lines.append(f"- {head}: {f(v, 'd_peak_band')['n_pairs']}쌍({f(v, 'd_peak_band')['n_days']}일).")
        for o, name in (("d_peak_band", "수요대역 피크"), ("d_peak_all", "전체 슬롯 피크")):
            r = f(v, o)
            lines.append(f"  - {name} 차이(b−a): 평균 {r['mean']:+.1f} kW [{r['mean_lo95']:+.1f}, {r['mean_hi95']:+.1f}], 중앙 {r['median']:+.1f} kW; "
                         f"{unit} Σ차이/Σ간격 {r['per_unit']:+.1f} kW [{r['per_unit_lo95']:+.1f}, {r['per_unit_hi95']:+.1f}] (쌍별 비율 중앙 {r['median_per_unit']:+.1f}).")
        r = f(v, "d_peak_all")
        lines.append(f"  - 피크 시각: b의 피크 시각이 더 늦은 쌍 {r['peak_hour_later_frac']:.0%}, 평균 이동 {r['peak_hour_shift_mean']:+.1f}시간.")
    lines.append("- 해석: 관측일끼리의 연관성이며 인과 효과가 아니다. 시작 시각·초기 생산 비중은 수주, 설비 상태, 계절 같은 다른 요인과 함께 움직일 수 있다. 재배치 권고의 근거가 아니라 모델 반사실 추정과 방향이 맞는지 보는 보조 자료로만 쓴다.")
    return "\n".join(lines) + "\n"


def run(out: Path) -> pl.DataFrame:
    days = day_table(features.load_panel())
    parts = [(pairs(days, "start"), 1.0), (pairs(days, "share"), 0.1)]
    res = pl.DataFrame([r for p, unit in parts if p.height for r in summarize(p, unit)])
    out.mkdir(parents=True, exist_ok=True)
    days.write_csv(out / "days.csv")
    pl.concat([p for p, _ in parts if p.height], how="diagonal").write_csv(out / "pairs.csv")
    res.write_csv(out / "summary.csv")
    (out / "summary.md").write_text(summary_md(days, res))
    print((out / "summary.md").read_text())
    return res


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results_v3" / "analog_days")
    run(parser.parse_args().out)


if __name__ == "__main__":
    main()
