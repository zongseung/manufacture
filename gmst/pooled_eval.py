"""Pooled time-ordered evaluation: dev folds f1–f4 (2021-07-07..08-31) + September (09-01..14) as a fifth origin.

Selection stays on f1–f4; this is a reporting view only. September B0/M2/BAT are re-scored with the frozen
run_v3 ladder (identical to results_v3/final, checked) so every model has per-day peaks; C-BAT comes from
results_v3/cbat_final and the dev seeds 0–2 of results_v3/attention_ablation (per-day losses averaged over seeds).
full_metrics.csv adds the remaining report metrics (RMSE, R², WAPE, CRPS, peak RMSE/R²/hit) with the definitions of
evaluate.point_metrics, computed per seed and then averaged over seeds like report_tables T1.
"""
import argparse
from pathlib import Path

import numpy as np
import polars as pl

from gmst import ROOT
from gmst import evaluate as ev
from gmst.run_v3 import _job

CBAT = "BAT_conditional_gaussian_fixed_attention"
MODELS = ("C-BAT", "BAT", "M2", "B0")
KEYS = ["model", "date"]
FULL = ("rmse", "r2", "wape_pct", "crps", "peak_rmse", "peak_r2", "peak_hit2")
SLOT_COLS = ["y_true", "y_mean", "y_median", *ev.QCOLS]
DAY_COLS = ["usable_peak", "M_true", "M_hat_mean", "peak_slot_true", "peak_time_mode"]


def per_day(slots: pl.DataFrame, days: pl.DataFrame, period: str) -> pl.DataFrame:
    """Day rows: slot absolute-error sum/count on observed slots, and peak truth/forecast on usable days."""
    err = (slots.filter(pl.col("y_true").is_not_nan())
           .group_by(KEYS).agg(((pl.col("y_true") - pl.col("y_median")).abs()).sum().alias("abs_sum"),
                               pl.len().alias("n_obs")))
    peak = days.select(*KEYS, "usable_peak", "M_true", "M_hat_median", "event_C90")
    return err.join(peak, on=KEYS, how="left").with_columns(period=pl.lit(period))


def dev_days(source: Path) -> pl.DataFrame:
    frames = []
    for seed in (0, 1, 2):
        slots = pl.read_csv(source / f"seed{seed}/oof_slots.csv").filter(pl.col("model").is_in([CBAT, "BAT", "M2", "B0"]))
        days = pl.read_csv(source / f"seed{seed}/oof_days.csv").filter(pl.col("model").is_in([CBAT, "BAT", "M2", "B0"]))
        frames.append(per_day(slots, days, "dev").with_columns(seed=pl.lit(seed)))
    return (pl.concat(frames).group_by(KEYS).agg(pl.col("abs_sum", "M_hat_median").mean(),
                                                 pl.col("n_obs", "usable_peak", "M_true", "event_C90", "period").first())
            .with_columns(pl.col("model").replace({CBAT: "C-BAT"})))


def september_days(cbat: Path, final: Path) -> tuple[pl.DataFrame, dict[str, tuple[pl.DataFrame, pl.DataFrame]]]:
    """Day rows and, per model, the (slots, days) frames needed by full_metrics."""
    cbat_slots, cbat_days = pl.read_csv(cbat / "test_slots.csv"), pl.read_csv(cbat / "test_days.csv")
    frames = [per_day(cbat_slots, cbat_days, "sep")]
    raw = {"C-BAT": (cbat_slots, cbat_days)}
    stored = pl.read_csv(final / "slots.csv")
    for i in range(3):  # B0, M2, BAT of the frozen run_v3 ladder
        slots, days, _ = _job("test", i, 2000, 100, True)
        name = slots["model"][0]
        ref = stored.filter(pl.col("model") == name).sort("datetime")["y_median"].to_numpy()
        assert np.allclose(slots.sort("datetime")["y_median"].to_numpy(), ref), f"{name} rerun differs from results_v3/final"
        frames.append(per_day(slots, days, "sep"))
        raw[name] = (slots, days)
    return pl.concat(frames, how="diagonal_relaxed").with_columns(pl.col("model").replace({CBAT: "C-BAT"})), raw


def summarize(days: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for model in MODELS:
        for period in ("dev", "sep", "all"):
            d = days.filter((pl.col("model") == model) & ((pl.col("period") == period) if period != "all" else True))
            p = d.filter(pl.col("usable_peak"))
            hi = p.filter(pl.col("event_C90") == 1)
            rows.append({"model": model, "period": period, "days": d.height, "slots": int(d["n_obs"].sum()),
                         "slot_mae": float(d["abs_sum"].sum() / d["n_obs"].sum()),
                         "peak_days": p.height, "peak_mae": float((p["M_hat_median"] - p["M_true"]).abs().mean()),
                         "high_days": hi.height,
                         "high_peak_mae": float((hi["M_hat_median"] - hi["M_true"]).abs().mean()) if hi.height else np.nan})
    return pl.DataFrame(rows)


def metrics_row(slots: pl.DataFrame, days: pl.DataFrame) -> dict[str, float]:
    """evaluate.point_metrics definitions: RMSE/R² on the mean forecast, WAPE on the median, CRPS on 19 quantiles."""
    y, mean, median = (slots[c].to_numpy() for c in ("y_true", "y_mean", "y_median"))
    usable = days.filter(pl.col("usable_peak"))
    truth, peak_mean = usable["M_true"].to_numpy(), usable["M_hat_mean"].to_numpy()
    return {"rmse": ev.rmse(y, mean)[0], "r2": ev.r2(y, mean)[0], "wape_pct": ev.wape(y, median)[0],
            "crps": ev.crps(y, slots.select(ev.QCOLS).to_numpy())[0],
            "peak_rmse": ev.rmse(truth, peak_mean)[0], "peak_r2": ev.r2(truth, peak_mean)[0],
            "peak_hit2": ev.peak_hit(usable["peak_slot_true"].to_numpy(), usable["peak_time_mode"].to_numpy())[0]}


def full_metrics(source: Path, sep: dict[str, tuple[pl.DataFrame, pl.DataFrame]]) -> pl.DataFrame:
    """Per seed: dev (seed-specific forecasts), September, and their union; then the mean over seeds 0–2."""
    names = {CBAT: "C-BAT", "BAT": "BAT", "M2": "M2", "B0": "B0"}
    rows = []
    for seed in (0, 1, 2):
        slots = pl.read_csv(source / f"seed{seed}/oof_slots.csv").filter(pl.col("model").is_in(list(names)))
        days = pl.read_csv(source / f"seed{seed}/oof_days.csv").filter(pl.col("model").is_in(list(names)))
        for raw, model in names.items():
            dev = (slots.filter(pl.col("model") == raw).select(SLOT_COLS), days.filter(pl.col("model") == raw).select(DAY_COLS))
            sep_slots, sep_days = sep[model]
            sept = (sep_slots.select(SLOT_COLS), sep_days.select(DAY_COLS))
            both = (pl.concat([dev[0], sept[0]]), pl.concat([dev[1], sept[1]]))
            for period, (sl, dy) in (("dev", dev), ("sep", sept), ("all", both)):
                rows.append({"model": model, "period": period, "seed": seed, **metrics_row(sl, dy)})
    return (pl.DataFrame(rows).group_by("model", "period", maintain_order=True).agg(pl.col(*FULL).mean())
            .sort(pl.col("model").replace_strict({m: i for i, m in enumerate(MODELS)}), pl.col("period").replace_strict({"dev": 0, "sep": 1, "all": 2})))


def paired(days: pl.DataFrame, b: int = 4000, seed: int = 0) -> pl.DataFrame:
    """C-BAT − other on pooled days: slot MAE (ratio of sums) and peak MAE, day bootstrap."""
    rng = np.random.default_rng(seed)
    base = days.filter(pl.col("model") == "C-BAT")
    rows = []
    for other in MODELS[1:]:
        j = base.join(days.filter(pl.col("model") == other), on="date", suffix="_o")
        a, o, n = j["abs_sum"].to_numpy(), j["abs_sum_o"].to_numpy(), j["n_obs"].to_numpy()
        p = j.filter(pl.col("usable_peak"))
        loss = ((p["M_hat_median"] - p["M_true"]).abs() - (p["M_hat_median_o"] - p["M_true"]).abs()).to_numpy()
        for metric, stat, size in (("slot_mae", lambda i: (a[i].sum() - o[i].sum()) / n[i].sum(), len(a)),
                                   ("peak_mae", lambda i: loss[i].mean(), len(loss))):
            boot = [stat(rng.integers(0, size, size)) for _ in range(b)]
            rows.append({"comparison": f"C-BAT − {other}", "metric": metric, "diff": float(stat(np.arange(size))),
                         "lo": float(np.percentile(boot, 2.5)), "hi": float(np.percentile(boot, 97.5)), "days": size})
    return pl.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results_v3/pooled_eval")
    args = parser.parse_args()
    sep_days, sep_raw = september_days(ROOT / "results_v3/cbat_final", ROOT / "results_v3/final")
    days = pl.concat([dev_days(ROOT / "results_v3/attention_ablation"), sep_days], how="diagonal_relaxed")
    args.out.mkdir(parents=True, exist_ok=True)
    days.write_csv(args.out / "days.csv")
    summary, boot = summarize(days), paired(days)
    full = full_metrics(ROOT / "results_v3/attention_ablation", sep_raw)
    full.write_csv(args.out / "full_metrics.csv")
    summary.write_csv(args.out / "summary.csv")
    boot.write_csv(args.out / "paired_bootstrap.csv")
    print(summary, boot, full, sep="\n")


if __name__ == "__main__":
    main()
