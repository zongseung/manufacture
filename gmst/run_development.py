"""Sealed rolling-origin model ladder and preregistered candidate selection."""

from datetime import date, datetime
from typing import Final
from zoneinfo import ZoneInfo

import polars as pl

from gmst import backbone, baselines, features, hmm
from gmst import evaluate as ev
from gmst.contracts import Model, Panel
from gmst.hmm_forecast import HMMState
from gmst.run_outputs import (
    MODEL_ID,
    Json,
    RunConfig,
    RunData,
    candidate,
    group,
    pstar_json,
    refresh,
    stage,
)

LADDER: Final = ("H", "IO", "KAN")
LOOP: Final = LADDER
LOOP_DEADLINE: Final = date(2026, 10, 3)


def model_flags(config: RunConfig, tau: float, half_life: int) -> hmm.HMMFlags:
    return {
        "tau": tau,
        "half_life": half_life,
        "max_epochs": config.epochs,
        "patience": 20,
        "N": config.paths,
        "rounds": config.rounds,
        "occ_floor": config.occ_floor,
    }


def tune_components(
    panel: Panel, fold: str, components: list[str], flags: hmm.HMMFlags, tables: dict[str, pl.DataFrame]
) -> hmm.HMMFlags:
    """Each outer fold selects only against its own earlier inner window."""
    flags = flags.copy()
    if "H" in components:
        flags["emission_source"] = "b1"
    if "IO" in components:
        value, table = hmm.select_io_lambda(panel, flags, fold)
        flags.update(decoder="io", lambda_io=value)
        tables["io_lambda.csv"] = (
            pl.concat([tables["io_lambda.csv"], table], how="diagonal_relaxed") if "io_lambda.csv" in tables else table
        )
    if "KAN" in components:
        (M, value), table = hmm.select_kan(panel, flags, fold)
        flags.update(kan=True, M=M, lambda_spl=value)
        tables["kan_grid.csv"] = (
            pl.concat([tables["kan_grid.csv"], table], how="diagonal_relaxed") if "kan_grid.csv" in tables else table
        )
    return flags


def development(config: RunConfig) -> RunData:
    """Prepare development artifacts without unsealing or writing result files."""
    panel = features.load_panel()
    tables: dict[str, pl.DataFrame] = {}
    states: dict[tuple[str, str], HMMState] = {}
    fold_flags: dict[tuple[str, str], hmm.HMMFlags] = {}
    parameters: dict[str, tuple[float, int]] = {}

    def record[S](
        model: Model[S], source: Panel, fold: str, variant: str = "main", saved: dict[str, S] | None = None
    ) -> dict[str, S]:
        slots, days, fitted, inner = ev.rolling_origin(model, source, (fold,), variant, states=saved)
        for name, frame in (("oof_slots.csv", slots), ("oof_days.csv", days), ("inner_days.csv", inner)):
            tables[name] = pl.concat([tables[name], frame], how="diagonal_relaxed") if name in tables else frame
        return fitted

    with stage(3, "oof"):
        tau_tables = []
        for fold in config.folds:
            tau, h, table = backbone.select_tau(panel, fold)
            parameters[fold] = tau, h
            tau_tables.append(table)
            flags = model_flags(config, tau, h)
            fold_flags["main", fold] = flags
            record(baselines.b0_model(), panel, fold)
            record(baselines.b0p_model(), panel, fold)
            record(backbone.bb_model(tau, h), panel, fold)
            record(baselines.b1_model(tau, h, rounds=config.rounds), panel, fold)
            for kind in ("B2", "B3", "B4"):
                fitted = record(hmm.hmm_model(kind, **flags), panel, fold)
                if kind == "B4":
                    states["main", fold] = fitted[fold]
            for variant, source, protocol in (
                ("copies", features.load_panel(include_copies=True), "A+"),
                ("suspect", features.load_panel(include_suspect=True), "A+"),
                ("protoA", panel, "A"),
            ):
                record(baselines.b1_model(tau, h, protocol, config.rounds), source, fold, variant)
                ablation_flags: hmm.HMMFlags = {**flags, "protocol": protocol}
                record(hmm.hmm_model("B4", **ablation_flags), source, fold, variant)
            k2_flags: hmm.HMMFlags = {**flags, "K": 2}
            record(hmm.hmm_model("B4", **k2_flags), panel, fold, "K2")
            record(hmm.hmm_model("B4", re=True, **flags), panel, fold, "re", {fold: states["main", fold]})
            record(baselines.b1_model(tau, h, "A+W*", config.rounds), panel, fold, "AWstar")
            if fold in ("f2", "f3", "f4"):
                scenario_flags: hmm.HMMFlags = {**flags, "protocol": "B"}
                fitted = record(hmm.hmm_model("B4", train_from=date(2021, 7, 1), **scenario_flags), panel, fold, "B")
                states["B", fold] = fitted[fold]
        tables["backbone_tau.csv"] = pl.concat(tau_tables)
        tables["lag7_skip.csv"] = baselines.lag7_skip_table(panel)
        tables["ablation_status.csv"] = pl.DataFrame(
            {
                "item": ["K4", "isotonic"],
                "status": ["not_run", "not_run"],
                "reason": ["approved_optional_cut", "approved_optional_cut"],
            }
        )
    with stage(4, "calibration"):
        stars = refresh(tables, panel)
    with stage(5, "metrics"):
        if tables["metrics.csv"].is_empty():
            raise ValueError("development produced no metrics")
    with stage(6, "risk_check"):
        risks = ev.risk_check(ev.add_climatology(tables["oof_days.csv"], panel))
    with stage(7, "bootstrap"):
        candidates = {"main": candidate(tables, panel, "main", config.bootstrap)}
        weather = ev.compare(
            group(tables["oof_slots.csv"], "B1", "AWstar"),
            group(tables["oof_days.csv"], "B1", "AWstar"),
            group(tables["oof_slots.csv"], "B1", "main"),
            group(tables["oof_days.csv"], "B1", "main"),
            panel,
            "AWstar-main",
            metrics=("mae", "brier_mean"),
            B=config.bootstrap,
        )
        tables["bootstrap.csv"] = pl.concat([tables["bootstrap.csv"], pl.DataFrame(weather)], how="diagonal_relaxed")
    components: dict[str, list[str]] = {"main": []}
    statuses = {"main": "run"}
    adopted: list[str] = []
    with stage(8, "loop"):
        for variant in LADDER:
            if any(item["gate_met"] for item in candidates.values()):
                statuses[variant] = "skipped_gate_met"
                continue
            if datetime.now(ZoneInfo("Asia/Seoul")).date() > LOOP_DEADLINE:
                statuses[variant] = "skipped_deadline"
                continue
            for fold in config.folds:
                tau, h = parameters[fold]
                flags = tune_components(panel, fold, [*adopted, variant], model_flags(config, tau, h), tables)
                fold_flags[variant, fold] = flags
                fitted = record(hmm.hmm_model("B4", **flags), panel, fold, variant)
                states[variant, fold] = fitted[fold]
            stars = refresh(tables, panel)
            candidates[variant] = candidate(tables, panel, variant, config.bootstrap)
            statuses[variant], components[variant] = "run", [*adopted, variant]
            adopted = components[ev.select_variant(candidates)["variant"]]
        risks = ev.risk_check(ev.add_climatology(tables["oof_days.csv"], panel))
    with stage(9, "selection"):
        selected = ev.select_variant(candidates)
        variant = selected["variant"]
        epochs = hmm.final_epochs([states[variant, fold]["best_epoch"] for fold in config.folds])
        final_fold = "f4" if config.smoke else "test"
        if config.smoke:
            final_flags = fold_flags[variant, "f4"]
            tau, h = parameters["f4"]
        else:
            tau, h, final_tau = backbone.select_tau(panel, "test")
            tables["backbone_tau.csv"] = pl.concat([tables["backbone_tau.csv"], final_tau])
            final_flags = tune_components(panel, "test", components[variant], model_flags(config, tau, h), tables)
        final_flags = final_flags.copy()
        final_flags["final_epochs"] = epochs
        thresholds, n_thresholds = ev.thresholds(panel, features.role_idx(panel, final_fold, "train"))
        candidates_json: list[Json] = [
            {
                "variant": v,
                "model_id": MODEL_ID[v],
                "components": components.get(v, []),
                "status": statuses[v],
                "mae": candidates[v]["mae"] if v in candidates else None,
                "brier_mean": candidates[v]["brier_mean"] if v in candidates else None,
                "gate_met": candidates[v]["gate_met"] if v in candidates else None,
            }
            for v in ("main", *LADDER)
        ]
        summary: dict[str, Json] = {
            "mode": "smoke" if config.smoke else "development",
            "frozen": True,
            "submitted_model": "B4",
            "submitted_variant": variant,
            "submitted": MODEL_ID[variant],
            "gate_met": selected["gate_met"],
            "tried": selected["tried"],
            "criteria": {k: candidates[variant]["gap"][k] for k in ("mae", "brier_mean")},
            "reported": {"crps": candidates[variant]["gap"]["crps"]},
            "remaining_gap_to_b1": selected["remaining_gap_to_b1"],
            "candidates": candidates_json,
            "n_candidates_run": selected["n_candidates_run"],
            "caveat": selected["caveat"],
            "components": components[variant],
            "tau": tau,
            "half_life": h,
            "final_epochs": epochs,
            "fold_parameters": {
                v: {
                    f: {
                        "tau": parameters[f][0],
                        "half_life": parameters[f][1],
                        "lambda_io": flags.get("lambda_io"),
                        "M": flags.get("M"),
                        "lambda_spl": flags.get("lambda_spl"),
                        "components": components[v],
                    }
                    for (candidate_variant, f), flags in fold_flags.items()
                    if candidate_variant == v
                }
                for v in candidates
            },
            "io_lambda": final_flags.get("lambda_io"),
            "kan": {"M": final_flags.get("M"), "lambda_spl": final_flags.get("lambda_spl")}
            if final_flags.get("kan", False)
            else None,
            "final_settings": {
                "lambda_io": final_flags.get("lambda_io", 0.01),
                "M": final_flags.get("M", 12),
                "lambda_spl": final_flags.get("lambda_spl", 0.01),
                "occ_floor": config.occ_floor,
                "N": config.paths,
                "rounds": config.rounds,
            },
            "thresholds": thresholds.tolist(),
            "n_thresholds": n_thresholds,
            "p_star": pstar_json(stars),
            "final_protocol": "A+",
            "final_features": [*features.SLOT_FEATURES, *features.DAY_FEATURES],
            "optional": {"K4": "not_run", "isotonic": "not_run"},
            "rule": "B1 benchmark; gate-passing B4 candidates first, then pooled MAE and mean Brier",
        }
    return {
        "panel": panel,
        "tables": tables,
        "metadata": {"selection.json": summary, "risk_check.json": risks},
        "states": states,
        "stars": stars,
        "variant": variant,
        "components": components[variant],
    }
