"""v3 ladder: B0 → M2(B1, plan) → BAT on f1–f4; `--final` opens the sealed test fold (2021-09-01..14) once."""
import argparse
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

from gmst import ROOT, backbone, baselines, features
from gmst import evaluate as ev
from gmst.bat import bat_model
from gmst.contracts import Model

LADDER = ("B0", "M2", "BAT")
SUMMARY = ["mae", "rmse", "crps", "cov50", "cov80", "cov90", "peak_mae", "peak_hit2", "brier_mean_raw"]


def _asof[S](model: Model[S]) -> Model[S]:
    """Forecast day d from a panel with Y[d:] hidden, so no same-day or later Y can reach any model."""
    predict = model["predict"]

    def wrapped(state: S, panel: dict, d: int):  # type: ignore[no-untyped-def]
        Y = panel["Y"].copy()
        Y[d:] = np.nan
        return predict(state, {**panel, "Y": Y}, d)  # type: ignore[arg-type]
    return {**model, "predict": wrapped}


def ladder(tau: float, h: int, n_iter: int, rounds: int) -> list[Model]:
    m2 = baselines.b1_model(tau, h, "B", rounds)
    m2["name"] = "M2"
    return [_asof(m) for m in (baselines.b0_model(), m2, bat_model(n_iter))]


def _panel(final: bool) -> dict:
    return features.load_panel(unseal=True) if final else features.load_panel()  # type: ignore[return-value]


def _job(fold: str, i: int, n_iter: int, rounds: int, final: bool) -> tuple[pl.DataFrame, ...]:
    """One (fold, ladder model) pair in its own single-threaded process."""
    panel = _panel(final)
    tau, h, _ = backbone.select_tau(panel, fold)
    model = ladder(tau, h, n_iter, rounds)[i]
    slots, days, states, inner = ev.rolling_origin(model, panel, (fold,))
    if final and model["name"] == "BAT":
        # 발행값: _day_row는 관측 슬롯으로 M̂/위험을 다시 잘라 평가용으로 쓰므로, 제출 파일은 모델 출력 그대로 쓴다
        preds = [model["predict"](states[fold], panel, int(d)) for d in sorted(features.role_idx(panel, fold, "test"))]
        days = days.with_columns(
            M_hat_median=pl.Series([p["M_hat_median"] for p in preds]),
            M_hat_mean=pl.Series([p["M_hat_mean"] for p in preds]),
            peak_time_mode=pl.Series([p["peak_time_mode"] for p in preds]),
            **{f"risk_raw_{c}": pl.Series([float(p["risk_raw"][j]) for p in preds]) for j, c in enumerate(ev.CS)})
    print(f"[v3] done {fold} {model['name']}", flush=True)
    return slots, days, inner


def _final_outputs(out: Path, slots: pl.DataFrame, days: pl.DataFrame) -> None:
    bat_slots = slots.filter(pl.col("model") == "BAT").sort("datetime")
    bat_days = days.filter(pl.col("model") == "BAT").select(
        "date", "M_hat_median", "M_hat_mean",
        pl.format("{}:{}", (pl.col("peak_time_mode") // 4).cast(pl.String).str.zfill(2),
                  (pl.col("peak_time_mode") % 4 * 15).cast(pl.String).str.zfill(2)).alias("peak_time_mode"),
        *(pl.col(f"risk_platt_{c}").alias(f"risk_{c}") for c in ev.CS), *ev.CS)
    preds = bat_slots.join(bat_days, on="date", validate="m:1").select(
        "datetime", "model", "y_mean", "y_median", *ev.QCOLS, "M_hat_median", "M_hat_mean", "peak_time_mode",
        *(f"risk_{c}" for c in ev.CS), *ev.CS)
    assert preds.height == 14 * 96, preds.height
    preds.write_csv(out / "test_predictions.csv")
    bat_slots.select("datetime", is_missing=pl.col("y_true").is_nan() | pl.col("y_true").is_null()).write_csv(out / "eval_mask.csv")


def run(out: Path, folds: tuple[str, ...], n_iter: int, rounds: int, final: bool = False) -> pl.DataFrame:
    panel = _panel(final)
    # 작은 행렬(392×392)에서는 BLAS 다중 스레드가 오히려 느리다: 작업마다 1스레드, 작업을 프로세스로 병렬화
    jobs = [(fold, i, n_iter, rounds, final) for fold in folds for i in range(len(LADDER))]
    with ProcessPoolExecutor(min(len(jobs), os.cpu_count() or 1), mp.get_context("spawn")) as pool:
        results = list(pool.map(_job, *zip(*jobs, strict=True)))
    slots, days, inner = (pl.concat(list(v), how="diagonal_relaxed") for v in zip(*results, strict=True))
    days, inner = ev.calibrate_oof(days, "platt", inner)
    metrics = ev.point_metrics(slots, days, panel)
    out.mkdir(parents=True, exist_ok=True)
    summary = (metrics.filter((pl.col("stratum") == "all") & pl.col("metric").is_in(SUMMARY))
               .pivot(on="metric", index=["model", "fold"], values="value").sort("fold", "model"))
    if final:
        _final_outputs(out, slots, days)
        summary.write_csv(out / "test_metrics.csv")
        metrics.write_csv(out / "test_metrics_long.csv")
        return summary
    # ponytail: 부트스트랩은 ev.bootstrap_days(CV 검증일)만 받음, 시험 구간 신뢰구간이 필요해지면 그 필터를 폴드 인자로
    boot = []
    for ref, names, used in (("M2", ("BAT", "B0"), ("mae", "crps", "brier_mean")), ("B0", ("M2", "BAT"), ("mae",))):
        for name in names:
            used_here = ("mae",) if "B0" in (ref, name) else used
            pick = lambda f, m: f.filter(pl.col("model") == m)  # noqa: E731
            boot += ev.compare(pick(slots, name), pick(days, name), pick(slots, ref), pick(days, ref),
                               panel, f"{name}-{ref}", used_here)
    for name, frame in (("oof_slots", slots), ("oof_days", days), ("inner_days", inner), ("metrics", metrics),
                        ("bootstrap", pl.DataFrame(boot))):
        frame.write_csv(out / f"{name}.csv")
    summary.write_csv(out / "summary.csv")
    return summary


def main() -> None:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")  # spawn된 작업 프로세스가 상속
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results_v3")
    parser.add_argument("--folds", nargs="+", default=list(ev.FOLDS_CV))
    parser.add_argument("--quick", action="store_true", help="400 Gibbs iterations, 20 boosting rounds")
    parser.add_argument("--final", action="store_true", help="unseal and score the test fold once → <out>/final/")
    a = parser.parse_args()
    n_iter, rounds = (400, 20) if a.quick else (2000, 100)
    out, folds = (a.out / "final", ("test",)) if a.final else (a.out, tuple(a.folds))
    with pl.Config(tbl_rows=60, tbl_cols=12):
        print(run(out, folds, n_iter, rounds, a.final))


if __name__ == "__main__":
    main()
