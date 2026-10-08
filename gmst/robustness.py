import argparse
import json
import multiprocessing as mp
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Final, Literal, assert_never

import polars as pl

from gmst import ROOT, backbone, baselines, bat, benchmark_models, features
from gmst import evaluate as ev
from gmst.conditional_bat import ConditionalConfig, conditional_model
from gmst.contracts import Model, Panel
from gmst.plan_sensitivity import evaluate_plan_errors, sensitivity_summary
from gmst.run_v3 import SUMMARY, _asof

type ModelName = Literal["B0", "B0_kind", "M2", "M2_tuned", "BAT", "BAT_no_transfer", "CBAT"]
NAMES: Final[tuple[ModelName, ...]] = ("B0", "B0_kind", "M2", "M2_tuned", "BAT", "BAT_no_transfer")
CBAT_NAME: Final = conditional_model(ConditionalConfig())["name"]


def _evaluate[S](model: Model[S], panel: Panel, fold: str, out: Path, repeats: int,
                 audit: Callable[[S], pl.DataFrame] | None = None) -> None:
    model = _asof(model)
    slots, days, states, inner = ev.rolling_origin(model, panel, (fold,))
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in (("slots", slots), ("days", days), ("inner", inner)):
        frame.write_csv(out / f"{name}.csv")
    if audit is not None:
        audit(states[fold]).write_csv(out / "tuning.csv")
    if model["name"] in ("BAT", "M2_tuned", CBAT_NAME):
        evaluate_plan_errors(model, states[fold], panel, fold, repeats).write_csv(out / "plan_errors.csv")
    print(f"[robustness] done {fold} {model['name']}", flush=True)


def _job(fold: str, name: ModelName, out: Path, quick: bool, repeats: int) -> None:
    panel = features.load_panel()
    path = out / "jobs" / f"{fold}_{name}"
    n_iter = 400 if quick else 2000
    match name:
        case "B0":
            _evaluate(baselines.b0_model(), panel, fold, path, repeats)
        case "B0_kind":
            _evaluate(benchmark_models.b0_kind_model(), panel, fold, path, repeats)
        case "BAT" | "BAT_no_transfer":
            _evaluate(bat.bat_model(n_iter, use_transfer=name == "BAT"), panel, fold, path, repeats)
        case "M2" | "M2_tuned":
            tau, half_life, _ = backbone.select_tau(panel, fold)
            if name == "M2":
                model = baselines.b1_model(tau, half_life, "B", 20 if quick else 100)
                model["name"] = "M2"
                _evaluate(model, panel, fold, path, repeats)
            else:
                _evaluate(benchmark_models.tuned_model(tau, half_life, quick), panel, fold, path,
                          repeats, benchmark_models.tuning_table)
        case "CBAT":
            config = ConditionalConfig(n_iter=400, burn=200) if quick else ConditionalConfig(n_iter=6000, burn=3000)
            _evaluate(conditional_model(config), panel, fold, path, repeats)
        case unreachable:
            assert_never(unreachable)


def collect(out: Path, folds: tuple[str, ...]) -> pl.DataFrame:
    panel = features.load_panel()
    paths = [out / "jobs" / f"{fold}_{name}" for fold in folds for name in NAMES]
    slots, days, inner = (pl.concat([pl.read_csv(p / f"{kind}.csv") for p in paths], how="diagonal_relaxed")
                          for kind in ("slots", "days", "inner"))
    days, inner = ev.calibrate_oof(days, "platt", inner)
    metrics = ev.point_metrics(slots, days, panel)
    summary = (metrics.filter((pl.col("stratum") == "all") & pl.col("metric").is_in(SUMMARY))
               .pivot(on="metric", index=["model", "fold"], values="value").sort("fold", "model"))
    comparisons: list[dict[str, ev.Cell]] = []
    for name in NAMES:
        if name == "BAT":
            continue
        used = ("mae",) if name in ("B0", "B0_kind") else ("mae", "crps")
        comparisons.extend(ev.compare(
            slots.filter(pl.col("model") == "BAT"), days.filter(pl.col("model") == "BAT"),
            slots.filter(pl.col("model") == name), days.filter(pl.col("model") == name),
            panel, f"BAT-{name}", used,
        ))
    for name, frame in (("oof_slots", slots), ("oof_days", days), ("inner_days", inner), ("metrics", metrics),
                        ("summary", summary), ("bootstrap", pl.DataFrame(comparisons))):
        frame.write_csv(out / f"{name}.csv")
    tuning = pl.concat([pl.read_csv(out / "jobs" / f"{fold}_M2_tuned" / "tuning.csv") for fold in folds])
    tuning.write_csv(out / "tuning.csv")
    collect_plan(out, paths)
    return summary


def collect_plan(out: Path, paths: list[Path]) -> pl.DataFrame:
    errors = pl.concat([pl.read_csv(p / "plan_errors.csv") for p in paths if (p / "plan_errors.csv").exists()])
    errors.write_csv(out / "plan_errors.csv")
    summary = sensitivity_summary(errors)
    summary.write_csv(out / "plan_summary.csv")
    return summary


def main() -> None:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS"):
        os.environ[name] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results_v3" / "robustness")
    parser.add_argument("--folds", nargs="+", choices=ev.FOLDS_CV, default=list(ev.FOLDS_CV))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--models", nargs="+", choices=(*NAMES, "CBAT"), default=list(NAMES))
    args = parser.parse_args()
    if args.repeats < 1 or args.workers < 1:
        parser.error("--repeats and --workers must be positive")
    folds = tuple(dict.fromkeys(args.folds))
    models = tuple(dict.fromkeys(args.models))
    jobs = [(fold, name, args.out, args.quick, args.repeats) for fold in folds for name in models]
    args.out.mkdir(parents=True, exist_ok=True)
    status = {"status": "running", "folds": folds, "models": models, "quick": args.quick,
              "repeats": args.repeats, "n_iter": 400 if args.quick else 2000, "cbat_n_iter_burn_seed": (400, 200, 0) if args.quick else (6000, 3000, 0), "september_opened": False,
              "selection": "separate slot and peak candidates selected on fold-local fit/tune only; calibration excluded",
              "baseline_revision": "independent_daily_head_tuning", "exploratory_adaptive_development": True,
              "exploratory": True}
    status_path = args.out / "run_status.json"
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    with ProcessPoolExecutor(min(args.workers, len(jobs)), mp.get_context("spawn")) as pool:
        list(pool.map(_job, *zip(*jobs, strict=True)))
    if set(models) == set(NAMES):
        summary = collect(args.out, folds)
    else:  # partial run (e.g. --models CBAT): only the plan-error experiment is pooled
        summary = collect_plan(args.out, [args.out / "jobs" / f"{fold}_{name}" for fold in folds for name in models])
    status_path.write_text(json.dumps({**status, "status": "complete"}, ensure_ascii=False, indent=2) + "\n")
    with pl.Config(tbl_rows=40, tbl_cols=12):
        print(summary)


if __name__ == "__main__":
    main()
