import argparse
import hashlib
import json
from pathlib import Path

import polars as pl

from gmst import ROOT, features
from gmst import evaluate as ev
from gmst.run_conditional import verify_alignment
from gmst.run_v3 import SUMMARY


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute development metrics from saved OOF predictions.")
    parser.add_argument("--source", type=Path, default=ROOT / "results_v3/attention_ablation")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.out is None:
        args.out = args.source / "metrics"
    panel = features.load_panel()
    paths = sorted(args.source.glob("seed[0-9]*/oof_slots.csv"))
    if not paths:
        parser.error("No saved seed*/oof_slots.csv files found")
    reference_slots = pl.read_csv(paths[0])
    reference_days = pl.read_csv(paths[0].with_name("oof_days.csv"))
    records: list[pl.DataFrame] = []
    inputs: dict[str, str] = {}
    for path in paths:
        day_path = path.with_name("oof_days.csv")
        for item in (path, day_path):
            inputs[str(item)] = hashlib.sha256(item.read_bytes()).hexdigest()
        slots, days = pl.read_csv(path), pl.read_csv(day_path)
        assert set(slots["fold"]) == set(days["fold"]) == set(ev.FOLDS_CV)
        verify_alignment(slots, days, reference_slots, reference_days)
        if path != paths[0]:
            predicate = pl.col("model").str.contains("^BAT_(conditional|pooled)_gaussian")
            slots, days = slots.filter(predicate), days.filter(predicate)
        metrics = ev.point_metrics(slots, days, panel)
        high = ev.point_metrics(slots.head(0), days.filter(pl.col("M_true") > pl.col("C90")), panel)
        high = high.filter(pl.col("stratum") == "all").with_columns(pl.lit("high").alias("stratum"))
        records.append(pl.concat([metrics, high]).with_columns(pl.lit(int(path.parent.name[4:])).alias("seed")))
    detail = pl.concat(records)
    aggregate = detail.group_by("model", "variant", "fold", "stratum", "metric").agg(
        pl.col("value").mean().alias("value"), pl.col("value").min().alias("seed_min"),
        pl.col("value").max().alias("seed_max"), pl.col("seed").n_unique().alias("n_seeds"),
        pl.col("n").min().alias("n_min"), pl.col("n").max().alias("n_max"),
    ).sort("model", "fold", "stratum", "metric")
    summary = (aggregate.filter((pl.col("fold") == "pooled") & pl.col("metric").is_in(SUMMARY))
               .pivot(on="metric", index=["model", "variant", "stratum"], values="value")
               .sort("stratum", "model"))
    args.out.mkdir(parents=True, exist_ok=True)
    detail.write_csv(args.out / "metrics_by_seed.csv")
    aggregate.write_csv(args.out / "metrics.csv")
    summary.write_csv(args.out / "summary.csv")
    code = [Path(__file__), ROOT / "gmst/evaluate.py", ROOT / "gmst/run_v3.py",
            ROOT / "gmst/run_conditional.py", ROOT / "gmst/features.py"]
    manifest = {
        "source": str(args.source), "input_sha256": inputs,
        "code_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in code},
        "refitted": False, "september_opened": False, "alignment_verified": True,
        "r2": "1 - SSE/SST; mean predictions; constant target or n<2 -> NaN; pooled OOF, not fold mean",
        "bias": "mean prediction minus observation, kW",
        "wape_pct": "100 * sum(abs(median prediction - observation)) / sum(abs(observation)); zero denominator -> NaN",
        "peak": "usable_peak dates only; squared metrics and bias use M_hat_mean",
        "high": "observed M_true > training-fold C90; retrospective stratum",
        "seed_policy": "Mean of seed-specific metrics; repeated baseline predictions evaluated once; seeds are not extra days",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(summary.filter(pl.col("stratum") == "all").select(
        "model", "r2", "rmse", "mae", "peak_r2", "peak_rmse", "peak_mae"))


if __name__ == "__main__":
    main()
