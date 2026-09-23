import argparse
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Final, TypedDict

import numpy as np
import polars as pl

from gmst import RESULTS, ROOT, backbone, baselines, evaluate, features, hmm, splits
from gmst.contracts import Model, Panel
from gmst.run_outputs import RunConfig

WINDOW_STARTS: Final = tuple(date(2021, 7, 1) + timedelta(days=7 * i) for i in range(7)) + (date(2021, 8, 18),)


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    smoke: bool = False
    seed: int = 0
    tail_u: float = 0.5
    tail_alpha: float = 0.05
    epochs: int | None = None
    paths: int | None = None
    rounds: int | None = None


class ExperimentDecision(TypedDict):
    status: str
    pc_qualified: bool
    winners_by_seed: dict[str, str]


def experiment_panel(panel: Panel) -> Panel:
    if any(start + timedelta(days=13) >= date(2021, 9, 1) for start in WINDOW_STARTS):
        raise ValueError("September must remain sealed in AR experiment windows")
    dates = panel["dates"]
    events = panel["days"]["event_id"].to_list()
    roles = [pl.Series(f"ar{i}", splits.fold_roles(dates, events, start, start + timedelta(days=13), "val"))
             for i, start in enumerate(WINDOW_STARTS, 1)]
    return {**panel, "days": panel["days"].with_columns(roles)}


def evaluate_fold(panel: Panel, fold: str, config: ExperimentConfig) -> pl.DataFrame:
    budget = RunConfig(config.smoke)
    rate = hmm.pc_rate_for_tail(config.tail_u, config.tail_alpha)
    tau, half_life, _ = backbone.select_tau(panel, fold)

    def score[S](candidate: str, model: Model[S]) -> dict[str, str | int | float]:
        slots, days, states, inner = evaluate.rolling_origin(model, panel, (fold,), candidate)
        calibrated, _ = evaluate.calibrate_oof(days, inner=inner)
        mae, n_slots = evaluate.mae(slots["y_true"].to_numpy(), slots["y_median"].to_numpy())
        crps, _ = evaluate.crps(slots["y_true"].to_numpy(), slots.select(evaluate.QCOLS).to_numpy())
        risk_metrics: dict[str, str | int | float] = {}
        briers = []
        for threshold in evaluate.CS:
            risk = calibrated[f"risk_platt_{threshold}"].to_numpy()
            event = calibrated[f"event_{threshold}"].to_numpy()
            valid = np.isfinite(risk) & np.isfinite(event)
            brier = evaluate.brier(risk, event)
            briers.append(brier)
            risk_metrics.update({f"brier_{threshold}": brier,
                                 f"brier_n_{threshold}": int(valid.sum()),
                                 f"events_{threshold}": int((event[valid] == 1).sum()),
                                 f"calibration_{threshold}": str(calibrated[f"calibration_status_platt_{threshold}"][0])})
        finite = [value for value in briers if np.isfinite(value)]
        state = states[fold]
        phi_mean = float("nan")
        best_epoch = config.rounds or budget.rounds
        if isinstance(state, dict) and "model" in state:
            phi_mean = float(state["model"].phi().mean().detach().cpu())
            best_epoch = state["best_epoch"]
        return {"fold": fold, "start": panel["dates"][int(features.role_idx(panel, fold, "val")[0])].isoformat(),
                "candidate": candidate, "seed": 0 if candidate == "B1" else config.seed,
                "mae": mae, "crps": crps, "brier_mean": float(np.mean(finite)) if finite else float("nan"),
                "n_slots": n_slots, "n_days": int(calibrated["usable_peak"].sum()),
                "phi_mean": phi_mean, "pc_rate": rate if candidate == "B4-PC" else 0.0,
                "epoch": best_epoch, "tau": tau, "half_life": half_life, **risk_metrics}

    common = {"tau": tau, "half_life": half_life, "max_epochs": config.epochs or budget.epochs,
              "N": config.paths or budget.paths, "seed": config.seed}
    rows = []
    if config.seed == 0:
        rows.append(score("B1", baselines.b1_model(tau, half_life, rounds=config.rounds or budget.rounds)))
    rows.append(score("B3", hmm.hmm_model("B3", **common)))
    rows.append(score("B4", hmm.hmm_model("B4", **common)))
    rows.append(score("B4-low", hmm.hmm_model("B4", phi_start=0.02, **common)))
    rows.append(score("B4-PC", hmm.hmm_model("B4", phi_start=0.02, pc_rate=rate, **common)))
    return pl.DataFrame(rows)


def summarize(metrics: pl.DataFrame, required_folds: int = 8) -> tuple[pl.DataFrame, ExperimentDecision]:
    expressions = [pl.col("fold").n_unique().alias("n_folds"),
                   pl.col("mae").mean().alias("mae_mean"),
                   pl.col("mae").median().alias("mae_median"),
                   pl.col("mae").max().alias("mae_worst"),
                   pl.col("crps").mean().alias("crps_mean")]
    if "brier_mean" in metrics.columns:
        expressions.append(pl.col("brier_mean").mean().alias("brier_mean"))
    summary = metrics.group_by("candidate", "seed").agg(expressions).sort("seed", "candidate")
    grouped = {(str(row["seed"]), str(row["candidate"])): row for row in summary.iter_rows(named=True)}
    winners: dict[str, str] = {}
    qualified = True
    for seed in range(3):
        key = str(seed)
        candidates = [grouped.get((key, candidate)) for candidate in ("B3", "B4", "B4-low", "B4-PC")]
        available = [row for row in candidates if row is not None and row["n_folds"] == required_folds
                     and np.isfinite(row["mae_mean"])]
        if available:
            winner = min(available, key=lambda row: (row["mae_mean"], row["candidate"] != "B3",
                                                     row["crps_mean"], row["candidate"]))
            winners[key] = str(winner["candidate"])
        if len(available) != 4:
            qualified = False
            continue
        b3, b4, low, pc = candidates
        qualified &= (pc["mae_mean"] < b3["mae_mean"] and pc["mae_mean"] < b4["mae_mean"]
                      and pc["mae_mean"] < low["mae_mean"] and winners[key] == "B4-PC")
    return summary, {"status": "development_only", "pc_qualified": qualified,
                     "winners_by_seed": winners}


def run(out: Path, config: ExperimentConfig) -> None:
    output = out.resolve()
    if output.is_relative_to(RESULTS.resolve()):
        raise ValueError("AR experiment cannot write to official results")
    rate = hmm.pc_rate_for_tail(config.tail_u, config.tail_alpha)
    if any(value is not None and value < 1 for value in (config.epochs, config.paths, config.rounds)):
        raise ValueError("experiment budgets must be positive")
    if output.exists():
        raise ValueError(f"experiment output already exists: {output}")
    panel = experiment_panel(features.load_panel())
    folds = ("ar8",) if config.smoke else tuple(f"ar{i}" for i in range(1, 9))
    seeds = (0,) if config.smoke else (0, 1, 2)
    frames = [evaluate_fold(panel, fold, ExperimentConfig(config.smoke, seed, config.tail_u,
              config.tail_alpha, config.epochs, config.paths, config.rounds))
              for fold in folds for seed in seeds]
    metrics = pl.concat(frames)
    summary, decision = summarize(metrics, required_folds=len(folds))
    if config.smoke:
        decision["status"] = "smoke_only"
        decision["pc_qualified"] = False
    output.mkdir(parents=True)
    metrics.write_csv(output / "fold_metrics.csv")
    summary.write_csv(output / "summary.csv")
    (output / "decision.json").write_text(json.dumps({**decision, "tail_u": config.tail_u,
        "tail_alpha": config.tail_alpha, "pc_rate": rate, "folds": list(folds), "seeds": list(seeds)},
        indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prior-only AR shrinkage comparison")
    parser.add_argument("--smoke", action="store_true", help="run ar8 and seed 0 with smaller budgets")
    parser.add_argument("--out", type=Path, default=ROOT / "results_ar")
    parser.add_argument("--u", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--paths", type=int)
    parser.add_argument("--rounds", type=int)
    args = parser.parse_args()
    try:
        run(args.out, ExperimentConfig(args.smoke, 0, args.u, args.alpha,
                                       args.epochs, args.paths, args.rounds))
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
