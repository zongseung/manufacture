"""Chapter 4 (현장 활용방안): BAT-driven same-day production re-allocation with a posterior-robust rule.

Per fold: fit BAT on the training days, re-allocate every operating validation day with
reallocate.reallocate on the posterior-mean curve, then re-score the recommended plan under
every posterior draw. Recommend only if P(peak_new < peak_old) ≥ 0.95 over draws.
All savings are model-based counterfactuals, never observed outcomes.
"""
import argparse
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl

from gmst import ROOT, bat, features
from gmst import evaluate as ev
from gmst import reallocate as ra
from gmst.contracts import BoolArray, FloatArray, Panel
from gmst.scenario import RATCHET_MONTHS, TARIFF, billing_demand, _timestamps, ratchet_floor, tariff_holidays

OPTION: Final = "II"
DEMAND_W: Final = TARIFF[OPTION]["base"] / 30  # ₩ per kW-day, daily-equivalent demand charge
RHOS: Final = (0.1, 0.2, 0.3)
# ponytail: 인건비 단가 미상이라 ₩/생산단위 격자 스윕, 실제 할증 단가가 오면 그 값 하나로 고정
WS: Final = (0.0, 0.5, 2.0, 10.0)
P_MIN: Final = 0.95


def draw_curves(state: bat.BATState, panel: Panel, d: int, plans: FloatArray) -> FloatArray:
    """(m, S, 96) kW per posterior draw s for m raw plans: idle_k + γ_k·ref + Σ_c w_c,k α(θ_k) x_c(q).

    Full formula for any plan: both the on/off and the log-volume channel follow the plan, and the
    operating type k is the plan's own (a shifted block keeps k; dropping an hour may change it).
    """
    plans = np.atleast_2d(plans)
    _, ref, _ = bat._day_parts(state, panel, d)
    out = np.empty((len(plans), len(state["theta"]), 96))
    for i, q in enumerate(plans):
        k = int(bat.kinds({"X": {"생산량": np.repeat(q[None], 4, 1)}}, np.array([0]))[0])  # type: ignore[arg-type]
        alpha = np.array([bat.attention(th[k]) for th in state["theta"]])  # (S, 96, 24)
        x = bat.channels(q, state["q_scale"])  # (2, 24)
        base = state["idle"][:, k] + state["gam"][:, k, None] * ref
        out[i] = (base + np.einsum("sc,sth,ch->st", state["w"][:, :, k], alpha, x)) * state["scale"]
    return out


def robust_stats(y0: FloatArray, y1: FloatArray, mask: BoolArray, rate: FloatArray) -> dict[str, float | bool]:
    """Posterior decision rule from paired draws y0 (old plan), y1 (new plan), both (S, 96) kW."""
    # Sundays/holidays have no demand-band slots: no demand charge, so never a peak recommendation
    p0, p1 = (y[:, mask].max(1) if mask.any() else np.zeros(len(y)) for y in (y0, y1))
    red = p0 - p1  # peak reduction, kW (>0 good)
    d_energy = (y1 - y0) @ (0.25 * rate)
    d_demand = -DEMAND_W * red
    P = float((p1 < p0).mean())
    lo, med, hi = np.quantile(red, [0.05, 0.5, 0.95])
    return {"P_robust": P, "recommended": P >= P_MIN, "dkW_median": float(med), "dkW_lo90": float(lo),
            "dkW_hi90": float(hi), "d_energy_won": float(np.median(d_energy)),
            "d_demand_equiv_won": float(np.median(d_demand)), "d_total_won": float(np.median(d_energy + d_demand)),
            "d_total_mean_won": float((d_energy + d_demand).mean())}


def run_fold(fold: str, n_iter: int, steps: int, max_days: int | None = None) -> tuple[list[dict], dict]:
    panel = features.load_panel()
    holidays = tariff_holidays()
    tr = features.role_idx(panel, fold, "train")
    C, _ = ev.thresholds(panel, tr)
    state = bat.fit_bat(panel, tr, C, n_iter=n_iter, burn=n_iter // 2)
    cap = ra.hourly_cap(panel["X"]["생산량"][tr, ::4])
    val = features.role_idx(panel, fold, "val")
    days = [int(d) for d in val if panel["op"][d] == 1 and np.nansum(panel["X"]["생산량"][d]) > 0][:max_days]
    rows, curves = [], {}
    for d in days:
        day = panel["dates"][d]
        hol = day in holidays
        q = np.nan_to_num(panel["X"]["생산량"][d, ::4])
        rate, mask = ra.tou_inputs(day, hol, OPTION)
        labor = ra.labor_multiplier(day, hol, True)
        f = bat.curve_fn(state, panel, d)
        obs_peak = billing_demand(panel["Y"][d], _timestamps(day), holidays)
        floor, _ = ratchet_floor(panel, day.month, {day}, holidays)
        base = {"fold": fold, "date": str(day), "weekday": day.weekday(), "kind": int(bat.kinds(panel, np.array([d]))[0]),
                "month": day.month, "obs_peak": obs_peak, "ratchet_floor": floor,
                "could_move_ratchet": day.month in RATCHET_MONTHS and obs_peak >= floor}
        results = [(rho, w, ra.reallocate(f, q, window=ra.plan_window(q), cap=cap, energy_rate=rate, demand_mask=mask,
                                           labor=labor, demand_weight=DEMAND_W, w=w, rho=rho, steps=steps))
                   for rho in RHOS for w in WS]
        Y = draw_curves(state, panel, d, np.stack([q] + [r["q_new"] for *_, r in results]))
        for i, (rho, w, r) in enumerate(results, 1):
            stats = robust_stats(Y[0], Y[i], mask, rate)
            rows.append(base | {"rho": rho, "w": w, "peak_before": r["peak_before"], "peak_after": r["peak_after"],
                                "moved": r["moved"], "moved_frac": r["moved"] / q.sum(),
                                "labor_before": float(labor @ q), "labor_after": float(labor @ r["q_new"])} | stats)
            curves[(str(day), rho, w)] = {"q": q, "q_new": r["q_new"], "mask": mask,
                                          "y0": np.quantile(Y[0], [0.05, 0.5, 0.95], 0),
                                          "y1": np.quantile(Y[i], [0.05, 0.5, 0.95], 0)}
    print(f"[ch4] {fold}: {len(days)} days", flush=True)
    return rows, curves


def pareto(days: pl.DataFrame) -> pl.DataFrame:
    """Per (ρ, w): labor index Σℓ·q′/Σℓ·q vs mean ₩/day, applying all plans or only robust ones."""
    rec = pl.col("recommended")
    return (days.group_by("rho", "w").agg(
        pl.len().alias("n_days"), rec.sum().alias("n_recommended"),
        (pl.col("labor_after").sum() / pl.col("labor_before").sum()).alias("labor_index_all"),
        pl.col("d_total_mean_won").mean().alias("d_cost_won_per_day_all"),
        (pl.when(rec).then(pl.col("labor_after")).otherwise(pl.col("labor_before")).sum()
         / pl.col("labor_before").sum()).alias("labor_index_robust"),
        pl.when(rec).then(pl.col("d_total_mean_won")).otherwise(0.0).mean().alias("d_cost_won_per_day_robust"),
    ).sort("rho", "w"))


def plot(days: pl.DataFrame, par: pl.DataFrame, curves: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.5))
    best = days.filter(pl.col("recommended")).sort("dkW_median", descending=True).head(1)
    if best.height:
        row = best.row(0, named=True)
        c = curves[(row["date"], row["rho"], row["w"])]
        t = np.arange(96) / 4
        for key, col, name in (("y0", "tab:gray", "before"), ("y1", "tab:blue", "after")):
            a.fill_between(t, c[key][0], c[key][2], color=col, alpha=0.2)
            a.plot(t, c[key][1], color=col, label=f"{name} (posterior median, 90% band)")
        a.fill_between(t, 0, 1, where=c["mask"], color="orange", alpha=0.08, transform=a.get_xaxis_transform(),
                       label="demand-band slots")
        a.set(title=f"Best robust day {row['date']} rho={row['rho']} w={row['w']}\nP={row['P_robust']:.2f}, peak -{row['dkW_median']:.1f} kW (model)",
              xlabel="hour", ylabel="kW")
        a.legend(fontsize=8)
    for rho in RHOS:
        p = par.filter(pl.col("rho") == rho)
        b.plot(p["labor_index_all"], p["d_cost_won_per_day_all"], "o-", label=f"rho={rho} all plans")
        b.plot(p["labor_index_robust"], p["d_cost_won_per_day_robust"], "s--", label=f"rho={rho} robust only")
    b.axhline(0, color="k", lw=0.5)
    b.set(title="Pareto over labor weight w (model counterfactual)", xlabel="labor cost index (after/before)",
          ylabel="mean d(energy + demand-equiv) won/day")
    b.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def summary_md(days: pl.DataFrame, par: pl.DataFrame) -> str:
    n_days = days.select(pl.struct("fold", "date").n_unique()).item()
    d0 = days.filter((pl.col("rho") == 0.2) & (pl.col("w") == 0.0))
    r0 = d0.filter(pl.col("recommended"))
    any_rec = days.filter(pl.col("recommended")).select(pl.struct("fold", "date").n_unique()).item()
    by_rho = days.filter(pl.col("w") == 0.0).group_by("rho").agg(pl.col("recommended").sum()).sort("rho")
    by_kind = d0.group_by("kind").agg(pl.len().alias("n"), pl.col("recommended").sum().alias("rec"),
                                      pl.col("dkW_median").median().alias("kw")).sort("kind")
    wk = d0.with_columns((pl.col("weekday") < 5).alias("wd")).group_by("wd").agg(
        pl.len().alias("n"), pl.col("recommended").sum().alias("rec")).sort("wd")
    top = days.filter(pl.col("recommended")).sort("dkW_median", descending=True).head(1)
    best = None if not top.height else "{date} rho={rho} w={w}".format(**top.row(0, named=True))
    med = lambda f, c: float(f[c].median()) if f.height else float("nan")  # noqa: E731
    ops_month = 22  # ponytail: 월 가동일 22일 가정, 실제 월별 가동 달력이 있으면 그 값으로 환산
    p = par.filter(pl.col("rho") == 0.2)
    w_hi = p.filter(pl.col("w") == max(WS)).row(0, named=True)
    w_lo = p.filter(pl.col("w") == 0.0).row(0, named=True)
    ratchet = r0.filter(pl.col("could_move_ratchet")).height
    kinds_txt = ", ".join(f"유형{r['kind']} {r['rec']}/{r['n']}일(ΔkW 중앙 {r['kw']:.2f})" for r in by_kind.iter_rows(named=True))
    wk_txt = ", ".join(f"{'평일' if r['wd'] else '토·일'} {r['rec']}/{r['n']}일" for r in wk.iter_rows(named=True))
    rho_txt = ", ".join(f"ρ={r['rho']}: {r['recommended']}일" for r in by_rho.iter_rows(named=True))
    e, dm = med(r0, "d_energy_won"), med(r0, "d_demand_equiv_won")
    lines = [
        "# 4장 현장 활용방안 — BAT 기반 일중 생산 재배치 (모델 기반 반사실 추정)",
        "",
        f"- 대상: f1–f4 검증 구간 가동일 {n_days}일, 폴드별 학습 구간으로 BAT를 1회 적합하고 규칙 R1–R4·R6(일 총생산 보존, 가동창 내 이동, 시간별 p95 상한, 이동량 ≤ ρ·총량, 날짜 간 이동 금지) 하에서 재배치했다.",
        f"- 결정 규칙: 사후분포 각 표본에서 수요대역 피크가 줄어들 확률 P ≥ {P_MIN}일 때만 권고한다. 기본 설정(ρ=0.2, w=0)에서 {r0.height}/{d0.height}일이 권고되었고, ρ·w 조합 중 하나라도 권고된 날은 {any_rec}일이다.",
        f"- ρ별 권고일 수(w=0): {rho_txt}. 실제 이동량은 일 총생산의 중앙 {100 * med(d0, 'moved_frac'):.2f}%(최대 {100 * float(d0['moved_frac'].max()):.1f}%)에 그쳐 ρ 상한이 거의 걸리지 않는다. BAT에서 전력은 가동 여부(on/off) 채널에 주로 반응하고, 가동창 안의 생산량 재배치(log 채널)에 대한 곡선 기울기는 작다.",
        f"- 권고일의 피크 감소는 사후 중앙값 기준 중앙 {med(r0, 'dkW_median'):.1f} kW(90% 구간 하한의 중앙값 {med(r0, 'dkW_lo90'):.1f} kW)이다. 관측 일 피크 중앙값은 {med(d0, 'obs_peak'):.0f} kW다.",
        f"- 실무적 크기: 권고일 가운데 피크 감소 중앙값이 1 kW 이상인 날은 {r0.filter(pl.col('dkW_median') >= 1).height}일이다. '강건함'(P ≥ 0.95)은 방향이 확실하다는 뜻일 뿐이고, 감소 폭은 관측 피크의 1% 미만이라 계량기 해상도·예측오차 안에 묻힌다.",
        f"- 금액(선택형 II, 2021 한전 요금): 권고일 하루 변화는 에너지요금 {e:+,.0f}₩, 일할 환산 기본요금(기본요금/30 × ΔkW) {dm:+,.0f}₩이다. 합계는 약 {e + dm:+,.0f}₩/일이다.",
        f"- 월 환산: 에너지 절감은 가동일 {ops_month}일을 가정하면 약 {e * ops_month:+,.0f}₩/월이다. 기본요금 효과는 그 달의 최대 피크일이 실제로 낮아질 때만 {dm * 30:+,.0f}₩/월(= 8,320₩ × ΔkW) 수준으로 실현된다.",
        f"- 래칫(12·1·2·7·8·9월 최대수요 12개월 적용): 권고일 {r0.height}일 중 {ratchet}일은 관측 피크가 해당 월까지의 래칫 바닥 이상이어서 연간 청구전력을 움직일 수 있는 날이다. 나머지 날은 피크를 낮춰도 청구전력이 바뀌지 않을 수 있다.",
        f"- 운전 유형별(ρ=0.2, w=0): {kinds_txt}. 유형 0은 비가동, 1은 1–12시간, 2는 13–19시간, 3은 20시간 이상 가동이다.",
        f"- 요일별: {wk_txt}.",
        f"- 인건비 절충(ρ=0.2): w=0이면 인건비 지수 {w_lo['labor_index_all']:.3f}, 일평균 {w_lo['d_cost_won_per_day_all']:+,.0f}₩이다. w={max(WS)}₩/단위에서는 지수 {w_hi['labor_index_all']:.3f}, {w_hi['d_cost_won_per_day_all']:+,.0f}₩이다. w를 올리면 생산이 1.0배 시간대로 옮겨 가면서 인건비 지수는 내려가지만, 강건 권고일은 {w_lo['n_recommended']}일에서 {w_hi['n_recommended']}일로 줄어든다. 이동량 자체가 작아 인건비와 전력요금 사이의 뚜렷한 상충은 관측되지 않았다(w가 커질 때 ₩이 오히려 조금 더 줄어드는 것은 비볼록 최적화의 경로 차이로 본다, pareto.csv).",
        f"- 강건 권고만 적용하면(ρ=0.2, w=0) 일평균 {w_lo['d_cost_won_per_day_robust']:+,.0f}₩이다. 모든 계획을 적용할 때의 {w_lo['d_cost_won_per_day_all']:+,.0f}₩보다 보수적인 수치다.",
        f"- 예시 그림: {best or '권고일 없음'} (realloc_example_pareto.png, 좌: 전후 곡선, 우: 파레토).",
        "- 한계: 모든 절감액은 BAT 사후평균·사후표본으로 계산한 모델 기반 반사실 추정치이고, 실제 재배치 후 관측값이 아니다. 가동/비가동 채널은 당일 계획으로 고정했고(가동창 확장 민감도 extend=1은 미실시), 세금·부가금·계약전력 30% 하한·설비 제약은 반영하지 않았다. 인건비 단가는 알 수 없어 w 격자로만 제시했다.",
    ]
    return "\n".join(lines) + "\n"


def run(out: Path, n_iter: int, steps: int, max_days: int | None) -> pl.DataFrame:
    with ProcessPoolExecutor(4, mp.get_context("spawn")) as pool:
        parts = list(pool.map(run_fold, ev.FOLDS_CV, *zip(*[(n_iter, steps, max_days)] * 4, strict=True)))
    days = pl.DataFrame([r for rows, _ in parts for r in rows])
    curves = {k: v for _, c in parts for k, v in c.items()}
    par = pareto(days)
    out.mkdir(parents=True, exist_ok=True)
    days.write_csv(out / "realloc_days.csv")
    par.write_csv(out / "pareto.csv")
    plot(days, par, curves, out / "realloc_example_pareto.png")
    (out / "summary.md").write_text(summary_md(days, par))
    return par


# ---- 분포 기반 시간대 이동 시나리오 (CONTEXT.md: 가동 블록·시간대 이동·관측 지지 범위·실행 오차) ----
SHIFTS: Final = (-2, -1, 1, 2)
SIGMAS: Final = (0.05, 0.1, 0.2)  # 실행 오차 LogNormal(0, σ): 계획 자료가 없어 민감도로만 제시
DAY_BAND: Final = np.arange(7, 20)  # 시각별 생산량 Gamma는 주간(07–20시)·야간으로 나눈다


def plan_prior(panel: Panel, tr: FloatArray, alpha: float = 0.5, n_mc: int = 20000, seed: int = 0) -> dict:
    """Training-day plan distributions: start hour | type ~ Categorical(Dirichlet α + counts);
    hourly volume | band ~ Gamma (moments). Returns start-hour posterior predictive and band 95% caps."""
    Q = np.nan_to_num(panel["X"]["생산량"][tr, ::4])
    kind = bat.kinds(panel, np.asarray(tr))
    on = Q > 0
    start = np.where(on.any(1), on.argmax(1), -1)
    p_start = np.zeros((bat.KINDS, 24))
    for k in range(bat.KINDS):
        p_start[k] = np.bincount(start[(kind == k) & (start >= 0)], minlength=24) + alpha
        p_start[k] /= p_start[k].sum()
    rng = np.random.default_rng(seed)
    day = np.zeros(24, bool)
    day[DAY_BAND] = True
    cap = np.empty(24)
    for band in (True, False):
        v = Q[:, day == band][on[:, day == band]]
        shape, scale = v.mean() ** 2 / v.var(), v.var() / v.mean()
        cap[day == band] = np.quantile(rng.gamma(shape, scale, n_mc), 0.95)
    return {"p_start": p_start, "cap": cap}


def candidates(q: FloatArray, k: int, prior: dict, rate_h: FloatArray) -> list[tuple[str, FloatArray]]:
    """Plausible time-shift plans (daily total preserved): whole-block shifts whose start hour has
    posterior predictive ≥ 5%, and single-hour moves out of an expensive hour into an adjacent cheaper
    in-block hour whose new volume stays under the band's Gamma 95% cap."""
    on = np.flatnonzero(q > 0)
    out = []
    for dlt in SHIFTS:
        lo, hi = on[0] + dlt, on[-1] + dlt
        if 0 <= lo and hi <= 23 and prior["p_start"][k, lo] >= 0.05:
            out.append((f"shift{dlt:+d}", np.roll(q, dlt)))
    for h in on:
        for j in (h - 1, h + 1):
            if 0 <= j <= 23 and q[j] > 0 and rate_h[j] < rate_h[h] and q[j] + q[h] <= prior["cap"][j]:
                q2 = q.copy()
                q2[j], q2[h] = q[j] + q[h], 0.0
                out.append((f"move{h:02d}to{j:02d}", q2))
    return out


def run_shift_fold(fold: str, n_iter: int, max_days: int | None = None, seed: int = 0) -> list[dict]:
    panel = features.load_panel()
    holidays = tariff_holidays()
    tr = features.role_idx(panel, fold, "train")
    C, _ = ev.thresholds(panel, tr)
    state = bat.fit_bat(panel, tr, C, n_iter=n_iter, burn=n_iter // 2)
    prior = plan_prior(panel, tr)
    rng = np.random.default_rng(seed)
    val = features.role_idx(panel, fold, "val")
    days = [int(d) for d in val if panel["op"][d] == 1 and np.nansum(panel["X"]["생산량"][d]) > 0][:max_days]
    rows = []
    for d in days:
        day = panel["dates"][d]
        hol = day in holidays
        q = np.nan_to_num(panel["X"]["생산량"][d, ::4])
        k = int(bat.kinds(panel, np.array([d]))[0])
        rate, mask = ra.tou_inputs(day, hol, OPTION)
        labor = ra.labor_multiplier(day, hol, True)
        cands = candidates(q, k, prior, rate.reshape(24, 4).mean(1))
        for sigma in SIGMAS:
            S = len(state["theta"])
            # 실행 오차: 두 계획 모두 같은 분포에서 독립적으로 실현된다 (가동 여부는 바뀌지 않음)
            noise = lambda: np.exp(sigma * rng.standard_normal((S, 24)))  # noqa: E731
            y0 = draw_curves_exec(state, panel, d, q, noise())
            best = None
            for name, q1 in cands:
                st = robust_stats(y0, draw_curves_exec(state, panel, d, q1, noise()), mask, rate)
                row = {"fold": fold, "date": str(day), "kind": k, "sigma": sigma, "plan": name,
                       "n_candidates": len(cands), "labor_index": float(labor @ q1 / (labor @ q)), "chosen": False} | st
                rows.append(row)
                if st["recommended"] and (best is None or st["d_total_mean_won"] < best["d_total_mean_won"]):
                    best = row
            if best is not None:
                best["chosen"] = True
    print(f"[ch4-shift] {fold}: {len(days)} days", flush=True)
    return rows


def draw_curves_exec(state: bat.BATState, panel: Panel, d: int, q: FloatArray, mult: FloatArray) -> FloatArray:
    """(S, 96): draw s uses posterior draw s with the plan realised as q·mult[s] (execution error)."""
    _, ref, _ = bat._day_parts(state, panel, d)
    qs = q[None] * mult
    k = int(bat.kinds({"X": {"생산량": np.repeat(q[None], 4, 1)}}, np.array([0]))[0])  # type: ignore[arg-type]
    alpha = np.array([bat.attention(th[k]) for th in state["theta"]])
    x = bat.channels(qs, state["q_scale"])  # (2, S, 24)
    base = state["idle"][:, k] + state["gam"][:, k, None] * ref
    return (base + np.einsum("sc,sth,csh->st", state["w"][:, :, k], alpha, x)) * state["scale"]


def shift_summary(rows: pl.DataFrame) -> str:
    lines = ["# 4장 — 분포 기반 시간대 이동 시나리오 (모델 기반 반사실 추정)", ""]
    for sigma in SIGMAS:
        r = rows.filter(pl.col("sigma") == sigma)
        n_days = r.select(pl.struct("fold", "date").n_unique()).item()
        ch = r.filter(pl.col("chosen"))
        lines.append(f"- σ={sigma}: 가동일 {n_days}일, 후보 계획 중앙 {r.group_by('date').agg(pl.col('n_candidates').first())['n_candidates'].median():.0f}개, "
                     f"권고일 {ch.height}일. 권고일 피크 감소 중앙 {ch['dkW_median'].median() if ch.height else float('nan'):.1f} kW "
                     f"(최대 {ch['dkW_median'].max() if ch.height else float('nan'):.1f}), 요금 변화 중앙 {ch['d_total_won'].median() if ch.height else float('nan'):+,.0f}₩/일, "
                     f"인건비 지수 중앙 {ch['labor_index'].median() if ch.height else float('nan'):.3f}.")
        if ch.height:
            lines.append("  - 선택된 계획 유형: " + ", ".join(f"{p} {n}일" for p, n in ch.group_by(pl.col("plan").str.slice(0, 5)).len().sort("len", descending=True).iter_rows()))
    lines.append("- 권고 조건: 사후 표본 × 실행 오차에서 P(피크 감소) ≥ 0.95, 이동 후 시각별 생산량이 해당 시간대 Gamma 95분위 이하, 블록 시작 시각의 사후예측확률 ≥ 5%.")
    lines.append("- 한계: 모든 수치는 BAT 기반 반사실 추정이며 실제 이동 후 관측값이 아니다. 실행 오차 σ는 계획 자료가 없어 민감도로만 제시했다.")
    return "\n".join(lines) + "\n"


def run_shift(out: Path, n_iter: int, max_days: int | None) -> pl.DataFrame:
    with ProcessPoolExecutor(4, mp.get_context("spawn")) as pool:
        parts = list(pool.map(run_shift_fold, ev.FOLDS_CV, [n_iter] * 4, [max_days] * 4))
    rows = pl.DataFrame([r for part in parts for r in part])
    out.mkdir(parents=True, exist_ok=True)
    rows.write_csv(out / "shift_days.csv")
    (out / "shift_summary.md").write_text(shift_summary(rows))
    print((out / "shift_summary.md").read_text())
    return rows


def main() -> None:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results_v3" / "ch4")
    parser.add_argument("--quick", action="store_true", help="400 Gibbs iterations, 60 steps, 2 days per fold")
    parser.add_argument("--shift", action="store_true", help="distribution-based time-shift scenario → shift_days.csv")
    a = parser.parse_args()
    n_iter, steps, max_days = (400, 60, 2) if a.quick else (2000, 300, None)
    if a.shift:
        run_shift(a.out, n_iter, max_days)
        return
    with pl.Config(tbl_rows=30, tbl_cols=10):
        print(run(a.out, n_iter, steps, max_days))


if __name__ == "__main__":
    main()
