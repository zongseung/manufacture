import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

from gmst import ROOT, bat, features
from gmst.preprocess import RAW
from gmst.run_conditional import verify_alignment


def peak_times(slots: pl.DataFrame, days: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    rows = []
    contributions = []
    for row in days.sort("date").iter_rows(named=True):
        observed = slots.filter((pl.col("date") == row["date"]) & pl.col("y_true").is_finite()).sort("slot")
        maximum = observed["y_true"].max()
        assert maximum == row["M_true"]
        tied = observed.filter(pl.col("y_true") == maximum)["slot"].to_numpy()
        assert int(tied[0]) == row["peak_slot_true"]
        first = int(tied[0])
        rows.append({**row, "first_peak_hour": first // 4,
                     "first_peak_time": f"{first // 4:02}:{first % 4 * 15:02}",
                     "n_peak_slots": len(tied), "peak_slots": ",".join(map(str, tied))})
        for stratum, included in (("all", True), ("op", row["kind"] > 0),
                                   ("nonop", row["kind"] == 0), ("high", row["high"])):
            if included:
                for hour in range(24):
                    in_hour = int(np.sum(tied // 4 == hour))
                    contributions.append({"stratum": stratum, "hour": hour,
                                          "first_peak_days": int(first // 4 == hour),
                                          "any_peak_days": int(in_hour > 0),
                                          "tie_weighted_days": in_hour / len(tied)})
    hours = pl.DataFrame(contributions).group_by("stratum", "hour").agg(
        pl.len().alias("n_days"), pl.col("first_peak_days").sum(),
        pl.col("any_peak_days").sum(), pl.col("tie_weighted_days").sum(),
    ).with_columns((100 * pl.col("first_peak_days") / pl.col("n_days")).alias("first_peak_pct"),
                   (100 * pl.col("tie_weighted_days") / pl.col("n_days")).alias("tie_weighted_pct"))
    for group in hours.partition_by("stratum"):
        assert group["first_peak_days"].sum() == group["n_days"][0]
        np.testing.assert_allclose(group["tie_weighted_days"].sum(), group["n_days"][0])
    return pl.DataFrame(rows), hours.sort("stratum", "hour")


def start_hours(production: np.ndarray) -> np.ndarray:
    """First hour with planned production (q > 0) per day; -1 when nothing is planned."""
    producing = production > 0
    return np.where(producing.any(axis=1), producing.argmax(axis=1), -1)


def volume_bins(volume: np.ndarray, reference: np.ndarray) -> list[str]:
    """'none' for zero production, else quartile Q1..Q4 by edges of the reference operating days."""
    edges = np.quantile(reference, [.25, .5, .75])
    return ["none" if not v > 0 else f"Q{np.searchsorted(edges, v) + 1}" for v in volume]


def observed_summary(days: pl.DataFrame, key: str) -> pl.DataFrame:
    extra = [pl.col("first_peak_hour").mode().min().alias("modal_peak_hour")] if "first_peak_hour" in days.columns else []
    return days.group_by(key).agg(
        pl.len().alias("n_days"), (pl.col("kind") > 0).sum().alias("n_operating"),
        pl.col("M_true").mean().alias("peak_mean_kw"), pl.col("M_true").median().alias("peak_median_kw"),
        pl.col("M_true").min().alias("peak_min_kw"), pl.col("M_true").max().alias("peak_max_kw"),
        pl.col("high").sum().alias("n_high"), (100 * pl.col("high").mean()).alias("high_rate_pct"), *extra,
    ).sort(key)


def seed_mean(frame: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Average per-seed MAE/bias; counts are identical across seeds and taken once."""
    return frame.group_by(keys).agg(
        pl.exclude(*keys, "mae", "median_bias", "seed").first(), pl.col("mae").mean(),
        pl.col("mae").min().alias("seed_min_mae"), pl.col("mae").max().alias("seed_max_mae"),
        pl.col("median_bias").mean(), pl.col("seed").n_unique().alias("n_seeds")).sort(keys)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit observed peak conditions and conditional BAT errors from saved OOF.")
    parser.add_argument("--source", type=Path, default=ROOT / "results_v3/conditional_collapsed")
    parser.add_argument("--model", nargs="+", default=None,
                        help="Model names in oof files; default: all seed0 models plus conditional/pooled Gaussian seeds")
    parser.add_argument("--out", type=Path, default=ROOT / "results_v3/peak_conditions")
    args = parser.parse_args()
    panel = features.load_panel()
    dates = [d.strftime("%Y.%m.%d") for d in panel["dates"]]
    production = panel["X"]["생산량"][:, ::4]
    groups, parent = bat.groups_from_plan(production)
    meta = pl.DataFrame({"date": dates, "kind": parent[groups[:, 0]],
                         "production_hours": (production > 0).sum(axis=1),
                         "daily_production": np.nansum(production, axis=1),
                         "weekday": [f"{d.isoweekday()}_{d.strftime('%a')}" for d in panel["dates"]],
                         "start_hour": start_hours(production)})
    context = pl.DataFrame({"date": np.repeat(dates, 96), "slot": np.tile(np.arange(96), len(dates)),
                            "kind": parent[groups].ravel(), "regime": groups.ravel(),
                            "producing": np.repeat(production > 0, 4, axis=1).ravel()})
    plan_hours = pl.DataFrame({"date": np.repeat(dates, 24), "first_peak_hour": np.tile(np.arange(24), len(dates)),
                               "peak_in_production": (production > 0).ravel()})
    inputs: dict[str, str] = {}

    def read(path: Path) -> pl.DataFrame:
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return pl.read_csv(path)

    reference = args.model[0] if args.model else "BAT"
    base_slots = read(args.source / "seed0/oof_slots.csv")
    base_days = read(args.source / "seed0/oof_days.csv")
    peak_days = base_days.filter((pl.col("model") == reference) & pl.col("usable_peak"))
    operating_volume = meta.filter(pl.col("date").is_in(peak_days["date"].implode()) & (pl.col("kind") > 0))["daily_production"]
    meta = meta.with_columns(pl.Series("volume_bin", volume_bins(meta["daily_production"].to_numpy(), operating_volume.to_numpy())))
    truth = peak_days.join(meta, on="date", validate="1:1")
    truth = truth.select("fold", "date", "kind", "production_hours", "daily_production", "weekday", "start_hour",
                         "volume_bin", "M_true", "C90", "peak_slot_true").with_columns(
        (pl.col("M_true") > pl.col("C90")).alias("high"), pl.col("date").str.slice(5, 2).alias("month"))
    assert truth.height == truth["date"].n_unique() == 51
    assert set(truth["fold"]) == {"f1", "f2", "f3", "f4"}
    slot_expr = (pl.col("datetime").str.slice(11, 2).cast(pl.Int64) * 4
                 + pl.col("datetime").str.slice(14, 2).cast(pl.Int64) // 15).alias("slot")
    detail, hours = peak_times(base_slots.filter(pl.col("model") == reference).with_columns(slot_expr), truth)
    detail = detail.join(plan_hours, on=["date", "first_peak_hour"], validate="1:1")
    on_off = pl.when(pl.col("kind") == 0).then(pl.lit("nonop"))
    peak_state = on_off.otherwise(pl.format("kind{}_{}", pl.col("kind"),
                                            pl.when(pl.col("peak_in_production")).then(pl.lit("on")).otherwise(pl.lit("off"))))
    slot_state = on_off.otherwise(pl.format("kind{}_{}", pl.col("kind"),
                                            pl.when(pl.col("producing")).then(pl.lit("on")).otherwise(pl.lit("off"))))
    conditions = detail.select("date", "kind", "weekday", "volume_bin", "start_hour", "production_hours",
                               "first_peak_hour", "peak_in_production", "high", "M_true").with_columns(
        peak_state.alias("kind_production"))
    reference_frames = [f.filter(pl.col("model") == reference).with_columns(model=pl.lit("BAT")) for f in (base_slots, base_days)]
    errors, peak_errors, peak_condition_errors, day_errors = [], [], [], []
    for seed in (0, 1, 2):
        slots = base_slots if seed == 0 else read(args.source / f"seed{seed}/oof_slots.csv")
        days = base_days if seed == 0 else read(args.source / f"seed{seed}/oof_days.csv")
        if args.model or seed:
            predicate = pl.col("model").is_in(args.model or ["BAT_conditional_gaussian", "BAT_pooled_gaussian"])
            slots, days = slots.filter(predicate), days.filter(predicate)
        verify_alignment(slots, days, *reference_frames)
        valid = (slots.with_columns(slot_expr).join(context, on=["date", "slot"], validate="m:1")
                 .join(meta.select("date", "weekday", "start_hour", "volume_bin"), on="date", validate="m:1")
                 .with_columns((pl.col("y_median") - pl.col("y_true")).alias("error"),
                               (pl.col("datetime").str.slice(11, 2)).alias("hour"), slot_state.alias("kind_production"))
                 .filter(pl.col("error").is_finite()))
        for key in ("hour", "kind", "regime", "kind_production", "start_hour", "volume_bin", "weekday"):
            errors.append(valid.group_by("model", key).agg(
                pl.len().alias("n_slots"), pl.col("date").n_unique().alias("n_days"),
                pl.col("error").abs().mean().alias("mae"), pl.col("error").mean().alias("median_bias"),
            ).rename({key: "bin"}).with_columns(pl.col("bin").cast(pl.String),
                pl.lit(key).alias("condition"), pl.lit(seed).alias("seed")))
        valid_days = days.filter(pl.col("usable_peak")).join(conditions.drop("M_true", "kind"), on="date", validate="m:1").join(
            meta.select("date", "kind"), on="date", validate="m:1").with_columns(
            (pl.col("M_hat_median") - pl.col("M_true")).alias("error"))
        strata = [("all", pl.lit(True)), ("nonop", pl.col("kind") == 0),
                  ("op", pl.col("kind") > 0), ("high", pl.col("M_true") > pl.col("C90"))]
        strata.extend((f"kind{k}", pl.col("kind") == k) for k in range(4))
        for name, predicate in strata:
            peak_errors.append(valid_days.filter(predicate).group_by("model").agg(
                pl.len().alias("n_days"), pl.col("error").abs().mean().alias("mae"),
                pl.col("error").mean().alias("median_bias"),
                (pl.col("error") < 0).sum().alias("n_under"), (pl.col("error") > 0).sum().alias("n_over"),
            ).with_columns(pl.lit(name).alias("stratum"), pl.lit(seed).alias("seed")))
        for key in ("kind_production", "first_peak_hour", "start_hour", "volume_bin", "weekday"):
            peak_condition_errors.append(valid_days.group_by("model", key).agg(
                pl.len().alias("n_days"), pl.col("error").abs().mean().alias("mae"),
                pl.col("error").mean().alias("median_bias"),
            ).rename({key: "bin"}).with_columns(pl.col("bin").cast(pl.String),
                pl.lit(key).alias("condition"), pl.lit(seed).alias("seed")))
        day_errors.append(valid_days.select("model", "date", pl.col("error").abs().alias("peak_abs_error_kw"),
                                            pl.col("M_hat_median").alias("predicted_peak_kw")).join(
            valid.group_by("model", "date").agg(pl.col("error").abs().mean().alias("slot_mae_kw")),
            on=["model", "date"], validate="1:1"))
    error_detail, peak_detail = pl.concat(errors), pl.concat(peak_errors)
    peak_condition_detail = pl.concat(peak_condition_errors)
    peak_summary = peak_detail.group_by("model", "stratum").agg(
        pl.col("n_days").first(), pl.col("mae").mean(), pl.col("median_bias").mean(),
        pl.col("n_under").min().alias("n_under_min"), pl.col("n_under").max().alias("n_under_max"),
        pl.col("n_over").min().alias("n_over_min"), pl.col("n_over").max().alias("n_over_max"))
    worst_days = (pl.concat(day_errors).group_by("model", "date").agg(
        pl.col("peak_abs_error_kw", "slot_mae_kw", "predicted_peak_kw").mean(), pl.len().alias("n_seeds"))
        .join(conditions, on="date", validate="m:1").sort("peak_abs_error_kw", descending=True)
        .group_by("model", maintain_order=True).head(10))
    args.out.mkdir(parents=True, exist_ok=True)
    outputs = {"observed_days": detail, "peak_hours": hours,
               **{f"peak_by_{name}": observed_summary(detail, key) for name, key in (
                   ("kind", "kind"), ("month", "month"), ("weekday", "weekday"), ("volume", "volume_bin"),
                   ("start_hour", "start_hour"), ("production_hours", "production_hours"),
                   ("peak_in_production", "peak_in_production"))},
               "slot_errors_by_seed": error_detail, "slot_errors": seed_mean(error_detail, ["model", "condition", "bin"]),
               "peak_errors_by_seed": peak_detail, "peak_errors": peak_summary.sort("model", "stratum"),
               "peak_condition_errors_by_seed": peak_condition_detail,
               "peak_condition_errors": seed_mean(peak_condition_detail, ["model", "condition", "bin"]),
               "worst_days": worst_days}
    for name, frame in outputs.items():
        frame.write_csv(args.out / f"{name}.csv")
    inputs[str(RAW)] = hashlib.sha256(RAW.read_bytes()).hexdigest()
    for path in (Path(__file__), ROOT / "gmst/features.py", ROOT / "gmst/bat.py"):
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.out / "manifest.json").write_text(json.dumps({
        "source": str(args.source), "models": args.model, "sha256": inputs, "september_opened": False, "refitted": False,
        "peak_days": 51, "high_days": 13, "days_with_tied_maxima": detail.filter(pl.col("n_peak_slots") > 1).height,
        "peak_time_policy": "First observed maximum plus sensitivity: each tied slot gets 1/n_peak_slots daily mass; any_peak_days can double-count days across hours",
        "high_policy": "Observed peak > fold-training C90; threshold changes between folds; retrospective association, not causation",
        "errors": "Median predictions; selected model metrics averaged across seeds, no extra days; models present only in seed0 once",
        "regime": "0 nonoperating day; 2*k-1 operating type k with zero production slot; 2*k with positive production slot",
        "conditions": "start_hour = first plan hour with q>0 (-1 none); volume_bin = quartiles of daily planned production over peak-evaluable operating days (none = zero); peak_in_production/kind_production use plan q>0 at the first observed peak hour or at the slot",
        "volume_edges": np.quantile(operating_volume.to_numpy(), [.25, .5, .75]).tolist(),
    }, ensure_ascii=False, indent=2) + "\n")
    print(outputs["peak_by_kind"])
    print(hours.filter((pl.col("stratum") == "high") & (pl.col("first_peak_days") > 0)))


if __name__ == "__main__":
    main()
