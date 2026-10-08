"""Run sealed rolling-origin comparisons of pooled and conditional Gaussian BAT."""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Final

import numpy as np
import polars as pl

from gmst import ROOT, bat, features
from gmst import evaluate as ev
from gmst.conditional_bat import (
    ConditionalConfig,
    ConditionalState,
    conditional_model,
    raw_paths,
)
from gmst.conditional_diagnostics import (
    convergence_report,
    peak_interval_summary,
    summarize,
)
from gmst.contracts import Panel
from gmst.preprocess import RAW
from gmst.run_v3 import SUMMARY, _asof, _final_outputs, _panel

# (noise_state, innovation); the first is the default C-BAT control
NOISE_VARIANTS = (("op", "gaussian"), ("kind", "gaussian"), ("slot", "gaussian"), ("op", "t"), ("kind", "t"))


@dataclass(frozen=True, slots=True)
class Job:
    fold: str
    config: ConditionalConfig
    out: Path


def posterior_tables(state: ConditionalState, out: Path) -> None:
    columns = {}
    parameters = {
        "sigma_u_kw": state.sigma_u * state.mean["scale"],
        "sigma_eta_kw": state.sigma_eta * state.mean["scale"],
        "rho": state.rho, "gamma": state.mean["gam"],
        "weight": state.mean["w"], "theta": state.mean["theta"],
    }
    for name, values in parameters.items():
        for index in np.ndindex(values.shape[1:]):
            columns[f"{name}[{','.join(map(str, index))}]"] = values[(slice(None), *index)]
    trace = pl.DataFrame(columns).with_row_index("draw")
    trace.write_csv(out / "posterior_trace.csv")
    rows = [{"parameter": name, "mean": float(values.mean()), "sd": float(values.std()),
             "q05": float(np.quantile(values, .05)), "q50": float(np.quantile(values, .5)),
             "q95": float(np.quantile(values, .95))} for name, values in columns.items()]
    pl.DataFrame(rows).write_csv(out / "posterior_summary.csv")
    names = [f"transfer_kind_{k}" for k in range(4)] + [f"rho_group_{g}" for g in range(state.rho.shape[1])]
    pl.DataFrame({"parameter": names, "acceptance": state.acceptance}).write_csv(out / "acceptance.csv")


def run_job(job: Job) -> None:
    panel = features.load_panel()
    model = _asof(conditional_model(job.config))
    started = perf_counter()
    C, _ = ev.thresholds(panel, features.role_idx(panel, job.fold, "train"))
    state = model["fit"](panel, job.fold, C)
    fit_seconds = perf_counter() - started
    slots, days, states, inner = ev.rolling_origin(model, panel, (job.fold,), states={job.fold: state})
    rolling_seconds = perf_counter() - started
    out = job.out / "jobs" / f"seed{job.config.seed}_{job.fold}_{model['name']}"
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in (("slots", slots), ("days", days), ("inner", inner)):
        frame.write_csv(out / f"{name}.csv")
    state = states[job.fold]
    posterior_tables(state, out)
    if state.inner is not None:
        inner_out = out / "inner_posterior"
        inner_out.mkdir(exist_ok=True)
        posterior_tables(state.inner, inner_out)
    negatives, intervals = [], []
    usable = features.usable_peak(panel)
    for d in features.role_idx(panel, job.fold, "val"):
        observed = np.isfinite(panel["Y"][d])
        hidden = panel["Y"].copy()
        hidden[d:] = np.nan
        asof: Panel = {**panel, "Y": hidden}
        raw = raw_paths(state, asof, int(d))
        negatives.append({"date": panel["dates"][d].strftime("%Y.%m.%d"),
                          "kind": bat.kind_of(panel["X"]["생산량"][d, ::4]),
                          "negative_fraction_all": float((raw < 0).mean()),
                          "negative_fraction_observed": float((raw[:, observed] < 0).mean()) if observed.any() else np.nan,
                          "minimum_raw_kw": float(raw.min()), "n_observed": int(observed.sum())})
        maxima = np.maximum(raw[:, observed], 0).max(axis=1) if observed.any() else np.full(raw.shape[0], np.nan)
        quantiles = np.quantile(maxima, [.05, .1, .25, .5, .75, .9, .95])
        intervals.append({"model": model["name"], "fold": job.fold,
                          "date": panel["dates"][d].strftime("%Y.%m.%d"), "op": int(panel["op"][d]),
                          "usable_peak": bool(usable[d]), "C90": float(state.C[2]),
                          "M_true": float(panel["Y"][d, observed].max()) if observed.any() else np.nan,
                          **dict(zip(("q05", "q10", "q25", "q50", "q75", "q90", "q95"), quantiles, strict=True))})
    pl.DataFrame(negatives).write_csv(out / "negative_paths.csv")
    pl.DataFrame(intervals).write_csv(out / "peak_intervals.csv")
    (out / "runtime.json").write_text(json.dumps({
        "model": model["name"], "fold": job.fold, "seed": job.config.seed,
        "config": asdict(job.config), "rolling_seconds": rolling_seconds,
        "fit_seconds": fit_seconds, "forecast_scoring_seconds": rolling_seconds - fit_seconds,
        "job_seconds": perf_counter() - started,
    }, indent=2) + "\n")
    print(f"[conditional] done seed={job.config.seed} {job.fold} {model['name']}", flush=True)


def verify_alignment(slots: pl.DataFrame, days: pl.DataFrame,
                     baseline_slots: pl.DataFrame, baseline_days: pl.DataFrame) -> None:
    """Require identical evaluation dates, targets, missingness and peak eligibility."""
    slot_keys, day_keys = ["fold", "date", "datetime"], ["fold", "date"]
    reference_slots = baseline_slots.filter(pl.col("model") == "BAT").sort(slot_keys)
    reference_days = baseline_days.filter(pl.col("model") == "BAT").sort(day_keys)
    for group in slots.partition_by("model"):
        ordered = group.sort(slot_keys)
        assert ordered.select(*slot_keys, "is_missing").equals(reference_slots.select(*slot_keys, "is_missing"))
        np.testing.assert_equal(ordered["y_true"].to_numpy(), reference_slots["y_true"].to_numpy())
    for group in days.partition_by("model"):
        ordered = group.sort(day_keys)
        flags = [*day_keys, "usable_peak", "n_obs", "peak_slot_true"]
        assert ordered.select(flags).equals(reference_days.select(flags))
        truth = ["M_true", "C50", "C75", "C90", "event_C50", "event_C75", "event_C90"]
        np.testing.assert_equal(ordered.select(truth).to_numpy(), reference_days.select(truth).to_numpy())
    if set(reference_slots["fold"]) == set(ev.FOLDS_CV):
        assert np.isfinite(reference_slots["y_true"].to_numpy()).sum() == 5016


def collect(jobs: list[Job], source: Path) -> pl.DataFrame:
    panel = features.load_panel()
    out = jobs[0].out
    folds = tuple(dict.fromkeys(job.fold for job in jobs))
    baseline_slots, baseline_days, baseline_inner = (
        pl.read_csv(source / f"{name}.csv").filter(pl.col("fold").is_in(folds))
        for name in ("oof_slots", "oof_days", "inner_days")
    )
    meta = pl.DataFrame({"date": [d.strftime("%Y.%m.%d") for d in panel["dates"]],
                         "kind": bat.kinds(panel, np.arange(len(panel["dates"]))), "op": panel["op"]})
    summaries: list[pl.DataFrame] = []
    for seed in dict.fromkeys(job.config.seed for job in jobs):
        paths = [out / "jobs" / f"seed{seed}_{job.fold}_{conditional_model(job.config)['name']}"
                 for job in jobs if job.config.seed == seed]
        slots, days, inner = (pl.concat([pl.read_csv(p / f"{name}.csv") for p in paths], how="diagonal_relaxed")
                              for name in ("slots", "days", "inner"))
        intervals = pl.concat([pl.read_csv(path / "peak_intervals.csv") for path in paths])
        interval_keys = ["model", "fold", "date"]
        assert intervals.sort(interval_keys).select(*interval_keys, "usable_peak").equals(
            days.sort(interval_keys).select(*interval_keys, "usable_peak"))
        np.testing.assert_equal(intervals.sort(interval_keys)["M_true"].to_numpy(),
                                days.sort(interval_keys)["M_true"].to_numpy())
        verify_alignment(slots, days, baseline_slots, baseline_days)
        slots = pl.concat([slots, baseline_slots], how="diagonal_relaxed")
        days = pl.concat([days, baseline_days], how="diagonal_relaxed")
        inner = pl.concat([inner, baseline_inner], how="diagonal_relaxed")
        days, inner = ev.calibrate_oof(days, "platt", inner)
        metrics = ev.point_metrics(slots, days, panel)
        summary = (metrics.filter((pl.col("stratum") == "all") & pl.col("metric").is_in(SUMMARY))
                   .pivot(on="metric", index=["model", "fold"], values="value").sort("fold", "model"))
        peak = days.filter(pl.col("usable_peak")).join(meta, on="date", validate="m:1")
        outputs = {"oof_slots": slots, "oof_days": days, "inner_days": inner, "metrics": metrics,
                   "peak_intervals": intervals, "peak_interval_summary": peak_interval_summary(intervals),
                   "summary": summary, "peak_summary": summarize(peak, ["model"]),
                   "peak_by_kind": summarize(peak, ["model", "kind"]),
                   "peak_by_op": summarize(peak, ["model", "op"]),
                   "peak_by_fold": summarize(peak, ["model", "fold"]),
                   "high_peak": summarize(peak.filter(pl.col("M_true") > pl.col("C90")), ["model"])}
        seed_out = out / f"seed{seed}"
        seed_out.mkdir(exist_ok=True)
        for name, frame in outputs.items():
            if frame is not None:
                frame.with_columns(pl.lit(seed).alias("seed")).write_csv(seed_out / f"{name}.csv")
        summaries.append(summary.with_columns(pl.lit(seed).alias("seed")))
    combined = pl.concat(summaries)
    combined.write_csv(out / "summary_by_seed.csv")
    return combined


def source_hashes(files: list[Path]) -> dict[str, str]:
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def run_final(out: Path, config: ConditionalConfig) -> pl.DataFrame:
    """Open the sealed test fold once: fit on the test fold's training days (< 2021-09-01), score 09-01..14."""
    model = _asof(conditional_model(config))
    out.mkdir(parents=True, exist_ok=True)
    files = [*sorted((ROOT / "gmst").glob("*.py")), ROOT / "uv.lock", RAW]
    hashes = source_hashes(files)
    status = {"status": "running", "fold": "test", "model": model["name"], "config": asdict(config),
              "issued_risk": "raw path exceedance (no Platt)",
              "september_opened": True, "exploratory": False, "source_sha256": hashes}
    status_path = out / "run_status.json"
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    panel = _panel(final=True)  # DC10: the only sanctioned unseal path
    slots, days, states, inner = ev.rolling_origin(model, panel, ("test",))
    days, inner = ev.calibrate_oof(days, "platt", inner)
    metrics = ev.point_metrics(slots, days, panel)
    summary = (metrics.filter((pl.col("stratum") == "all") & pl.col("metric").is_in(SUMMARY))
               .pivot(on="metric", index=["model", "fold"], values="value"))
    for name, frame in (("test_slots", slots), ("test_days", days), ("inner_days", inner),
                        ("test_metrics_long", metrics), ("test_metrics", summary)):
        if frame is not None:
            frame.write_csv(out / f"{name}.csv")
    # 제출 파일: run_v3처럼 M̂/위험은 관측 슬롯으로 자르지 않은 발행값을 쓴다 (평가 지표는 위의 잘린 값)
    issued = [model["predict"](states["test"], panel, int(d)) for d in sorted(features.role_idx(panel, "test", "test"))]
    issued_days, _ = ev.calibrate_oof(days.with_columns(
        M_hat_median=pl.Series([p["M_hat_median"] for p in issued]),
        M_hat_mean=pl.Series([p["M_hat_mean"] for p in issued]),
        peak_time_mode=pl.Series([p["peak_time_mode"] for p in issued]),
        **{f"risk_raw_{c}": pl.Series([float(p["risk_raw"][j]) for p in issued]) for j, c in enumerate(ev.CS)}),
        "platt", inner)
    # 7일 내부 Platt는 9월 C90을 상수로 만든다 → 경로 위험 그대로 발행 (document/RISK_issuance_preregistration.md)
    _final_outputs(out, slots, issued_days, risk="raw", model=model["name"])
    posterior_tables(states["test"], out)
    assert hashes == source_hashes(files), "Sources changed during run"
    status_path.write_text(json.dumps({**status, "status": "complete"}, ensure_ascii=False, indent=2) + "\n")
    return summary


# flag → (results folder, config overrides per compared model, help); {} is the default C-BAT
COMPARISONS: Final[dict[str, tuple[str, tuple[dict, ...], str]]] = {
    "attention_ablation": ("attention_ablation", ({"fixed_attention": False}, {}),
                           "Compare learned vs prior-fixed attention with conditional noise in both models"),
    "pooled_noise_comparison": ("conditional", ({"conditional": False, "fixed_attention": False}, {"fixed_attention": False}),
                                "Reproduce the earlier pooled/conditional comparison with learned attention"),
    "ref_comparison": ("ref_ablation", ({}, {"ref": "kind"}), "Compare op vs operating-kind reference days"),
    "noise_comparison": ("noise_ablation", tuple({"noise_state": n, "innovation": i} for n, i in NOISE_VARIANTS),
                         "Default C-BAT with noise_state kind/slot and Student-t innovation variants"),
}


def configs(comparison: str | None, base: ConditionalConfig) -> list[ConditionalConfig]:
    return [replace(base, **overrides) for overrides in (COMPARISONS[comparison][1] if comparison else ({},))]


def main() -> None:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS"):
        os.environ[name] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--source", type=Path, default=ROOT / "results_v3")
    parser.add_argument("--folds", nargs="+", choices=ev.FOLDS_CV, default=list(ev.FOLDS_CV))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--n-iter", type=int, default=2000)
    parser.add_argument("--burn", type=int)
    parser.add_argument("--workers", type=int, default=8)
    comparison = parser.add_mutually_exclusive_group()
    for name, (_, _, text) in COMPARISONS.items():
        comparison.add_argument("--" + name.replace("_", "-"), dest="comparison", action="store_const", const=name, help=text)
    comparison.add_argument("--final", action="store_true",
                            help="Open the sealed test fold (2021-09-01..14) once with the default C-BAT (first seed)")
    args = parser.parse_args()
    burn = args.n_iter // 2 if args.burn is None else args.burn
    if args.workers < 1 or not 0 <= burn < args.n_iter or min(args.seeds) < 0:
        parser.error("require workers >= 1, 0 <= burn < n-iter and seeds >= 0")
    if args.out is None:
        folder = "cbat_final" if args.final else COMPARISONS[args.comparison][0] if args.comparison else "conditional_fixed"
        args.out = ROOT / "results_v3" / folder
    if args.final:
        print(run_final(args.out, ConditionalConfig(n_iter=args.n_iter, burn=burn, seed=args.seeds[0])))
        return
    folds, seeds = tuple(dict.fromkeys(args.folds)), tuple(dict.fromkeys(args.seeds))
    jobs = [Job(fold, config, args.out) for seed in seeds for fold in folds
            for config in configs(args.comparison, ConditionalConfig(n_iter=args.n_iter, burn=burn, seed=seed))]
    args.out.mkdir(parents=True, exist_ok=True)
    files = [*sorted((ROOT / "gmst").glob("*.py")), ROOT / "uv.lock", RAW,
             *(args.source / f"{name}.csv" for name in ("oof_slots", "oof_days", "inner_days"))]
    hashes = source_hashes(files)
    status = {"status": "running", "folds": folds, "seeds": seeds, "n_iter": args.n_iter, "burn": burn,
              "models": sorted({conditional_model(job.config)["name"] for job in jobs}),
              "half_life": jobs[0].config.half_life, "comparison": args.comparison,
              "configs": {conditional_model(job.config)["name"]: {k: v for k, v in asdict(job.config).items() if k != "seed"}
                          for job in jobs},
              "fixed_shape": {"width": 1.5, "shape": 2.0, "lag_hours": 0.0},
              "workers": min(args.workers, 8, len(jobs)),
              "september_opened": False, "exploratory": True,
              "baseline_source": str(args.source), "source_sha256": hashes,
              "seed_policy": "Metrics evaluated separately per seed; seeds are not independent days."}
    status_path = args.out / "run_status.json"
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    with ProcessPoolExecutor(min(args.workers, 8, len(jobs)), mp.get_context("spawn")) as pool:
        list(pool.map(run_job, jobs))
    summary = collect(jobs, args.source)
    convergence = convergence_report(args.out, folds, seeds, status["models"])
    assert hashes == source_hashes(files), "Sources changed during run"
    status_path.write_text(json.dumps({**status, "status": "complete", "alignment_verified": True,
                                      "basic_split_rhat_passed": bool(convergence["passed"].all())},
                                      ensure_ascii=False, indent=2) + "\n")
    print(summary.filter(pl.col("fold") == "pooled"))


if __name__ == "__main__":
    main()
